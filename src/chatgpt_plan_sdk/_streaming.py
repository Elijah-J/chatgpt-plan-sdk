"""Async SSE framing, terminal judgement and response ownership for Responses."""

from __future__ import annotations

import codecs
import copy
import json
import re
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import anyio
import httpx
from anyio.lowlevel import checkpoint

from chatgpt_plan_sdk._errors import ResponsesAPIError, ResponsesStreamIncomplete
from chatgpt_plan_sdk.models.result import ResponsesResult

__all__ = ["ResponsesSSEEvent"]

_LINE_END = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True, slots=True)
class ResponsesSSEEvent:
    """One SSE record: the event name (``message`` if absent) and decoded JSON."""

    event: str
    data: Any


class _SSEDecoder:
    """Incremental UTF-8 SSE framing over byte chunks.

    Handles a leading BOM, CR, LF and CRLF (including a CR at a chunk end),
    comments, one leading space after the colon and multiline data joined by
    a newline. ``feed`` and ``finish`` return the complete ``(name, data)``
    records; an unterminated final line or record is discarded at EOF.
    """

    def __init__(self) -> None:
        self._text = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
        self._line: list[str] = []
        self._skip_lf = False
        self._name = ""
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[tuple[str, str]]:
        return self._consume(self._text.decode(chunk))

    def finish(self) -> list[tuple[str, str]]:
        return self._consume(self._text.decode(b"", final=True))

    def _consume(self, text: str) -> list[tuple[str, str]]:
        records: list[tuple[str, str]] = []
        if self._skip_lf and text:
            self._skip_lf = False
            if text[0] == "\n":
                text = text[1:]
        start = 0
        for match in _LINE_END.finditer(text):
            self._line.append(text[start : match.start()])
            line = "".join(self._line)
            self._line.clear()
            start = match.end()
            self._process(line, records)
        self._line.append(text[start:])
        if text.endswith("\r"):
            self._skip_lf = True
        return records

    def _process(self, line: str, records: list[tuple[str, str]]) -> None:
        if line == "":
            if self._data:
                records.append((self._name or "message", "\n".join(self._data)))
            self._name = ""
            self._data = []
            return
        if line.startswith(":"):
            return
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._name = value
        elif field == "data":
            self._data.append(value)


class _Accumulator:
    """Judge fed events and assemble the terminal result (I3, R6)."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self._items: dict[int, dict[str, Any]] = {}
        self._completed: dict[str, Any] | None = None
        self._failure: tuple[str, Any, Any] | None = None

    def feed(self, name: str, data: Any) -> None:
        kind = data.get("type") if isinstance(data, dict) else None
        label = kind if isinstance(kind, str) else name
        self.counts[label] = self.counts.get(label, 0) + 1
        if not isinstance(kind, str):
            return
        if kind == "response.output_item.done":
            self._feed_item(data)
        elif kind == "response.completed":
            self._feed_completed(data)
        elif kind in ("response.failed", "response.incomplete"):
            response = data.get("response")
            self._fail(kind.removeprefix("response."), copy.deepcopy(response), None)
        elif kind == "error":
            self._fail("error", None, copy.deepcopy(data))

    def _fail(self, kind: str, response: Any, error: Any) -> None:
        # The first fed non-success outcome is the verdict; nothing overrides it.
        if self._failure is None:
            self._failure = (kind, response, error)

    def _feed_item(self, data: dict[str, Any]) -> None:
        index = data.get("output_index")
        item = data.get("item")
        # bool is not an integer index; duplicates and non-objects are protocol.
        if type(index) is not int or index in self._items or not isinstance(item, dict):
            self._fail("protocol", None, None)
            return
        self._items[index] = copy.deepcopy(item)

    def _feed_completed(self, data: dict[str, Any]) -> None:
        response = data.get("response")
        if (
            self._completed is not None
            or not isinstance(response, dict)
            or response.get("status") != "completed"
        ):
            self._fail("protocol", None, None)
            return
        self._completed = copy.deepcopy(response)

    def result(self) -> ResponsesResult:
        if self._failure is not None:
            kind, response, error = self._failure
            raise ResponsesStreamIncomplete(kind, response=response, error=error)
        if self._completed is None:
            raise ResponsesStreamIncomplete("truncated")
        if self._items:
            output = tuple(self._items[index] for index in sorted(self._items))
        else:
            raw = self._completed.get("output")
            if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
                raise ResponsesStreamIncomplete("protocol")
            output = tuple(raw)
        return ResponsesResult(
            response=self._completed, output=output, event_counts=dict(self.counts)
        )


class ResponsesStream:
    """One Responses SSE response: iteration, finalization and ownership.

    The stream owns its response and closes it on exhaustion, ``[DONE]``,
    parser or transport failure, cancellation, ``aclose()`` and context exit;
    the reusable SDK client stays open. Events are handed out as deep copies,
    so caller mutation cannot alter the result.
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._decoder = _SSEDecoder()
        self._queue: deque[tuple[str, str]] = deque()
        self._chunks: AsyncIterator[bytes] | None = None
        self._eof = False
        self._ended = False
        self._closed = False
        self._cancelled = False
        self._fault: Exception | None = None
        # One event per read in flight; ``aclose`` waits for each to end.
        self._readers: set[anyio.Event] = set()
        self._accumulator = _Accumulator()

    # ------------------------------------------------------------ metadata

    @property
    def response(self) -> httpx.Response:
        """The underlying HTTPX response."""
        return self._response

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def headers(self) -> httpx.Headers:
        return self._response.headers

    @property
    def request_id(self) -> str | None:
        """The ``x-request-id`` response header, or None."""
        return self._response.headers.get("x-request-id")

    @property
    def is_closed(self) -> bool:
        """Whether the response body has been closed."""
        return self._response.is_closed

    # ----------------------------------------------------------- iteration

    def __aiter__(self) -> ResponsesStream:
        return self

    async def __anext__(self) -> ResponsesSSEEvent:
        event = await self._step(deliver=True)
        if event is None:
            raise StopAsyncIteration
        return event

    @property
    def text_stream(self) -> AsyncIterator[str]:
        """Yield ``response.output_text.delta`` text as it arrives.

        Reads the same event stream as raw iteration.
        """
        return self._iter_text()

    async def _iter_text(self) -> AsyncIterator[str]:
        async for event in self:
            data = event.data
            if (
                isinstance(data, dict)
                and data.get("type") == "response.output_text.delta"
                and isinstance(data.get("delta"), str)
            ):
                yield data["delta"]

    async def _step(self, *, deliver: bool) -> ResponsesSSEEvent | None:
        if not self._response.is_success:
            raise ResponsesAPIError(self._response)
        if self._ended or self._closed:
            return None
        reader_done = anyio.Event()
        self._readers.add(reader_done)
        try:
            try:
                return await self._advance(deliver)
            except BaseException as exc:
                # Parser/transport faults and cancellation both end here; the
                # body is closed before the exception continues.
                self._ended = True
                explicit_close = self._closed
                if isinstance(exc, Exception):
                    if not explicit_close:
                        self._fault = exc
                else:
                    self._cancelled = True
                await self._close_body()
                if explicit_close and isinstance(exc, Exception):
                    return None
                raise
        finally:
            # The reader has terminated; only now may a pending close finish.
            self._readers.discard(reader_done)
            reader_done.set()

    async def _advance(self, deliver: bool) -> ResponsesSSEEvent | None:
        record = await self._next_record()
        if record is None:
            await self._finish()
            return None
        # A read can buffer many events and handing them out never suspends,
        # so a cancellation or close requested meanwhile would wait for the
        # whole buffer. Checkpoint before each event.
        await checkpoint()
        if self._closed:
            self._ended = True
            return None
        name, text = record
        if text == "[DONE]":
            await self._finish()
            return None
        data = json.loads(text)
        self._accumulator.feed(name, data)
        return ResponsesSSEEvent(event=name, data=copy.deepcopy(data)) if deliver else None

    async def _next_record(self) -> tuple[str, str] | None:
        while not self._queue:
            if self._eof:
                return None
            if self._chunks is None:
                self._chunks = self._response.aiter_bytes()
            try:
                chunk = await self._chunks.__anext__()
            except StopAsyncIteration:
                self._eof = True
                self._queue.extend(self._decoder.finish())
            else:
                self._queue.extend(self._decoder.feed(chunk))
        return self._queue.popleft()

    async def _finish(self) -> None:
        self._ended = True
        await self._close_body()

    # ------------------------------------------------------ final response

    async def get_final_response(self) -> ResponsesResult:
        """Finish the stream and return the completed result.

        Drains the remaining events unless the stream was explicitly closed,
        cancelled or ended. Success needs exactly one fed ``response.completed``
        with a mapping response whose status is ``completed`` and no fed
        failure. Raises ``ResponsesStreamIncomplete`` otherwise, and
        propagates a parser or transport fault, now and on later calls.
        """
        if not self._response.is_success:
            raise ResponsesAPIError(self._response)
        while not (self._ended or self._closed):
            await self._step(deliver=False)
        if self._fault is not None:
            raise self._fault
        if self._cancelled:
            raise ResponsesStreamIncomplete("truncated")
        return self._accumulator.result()

    # ----------------------------------------------------------- ownership

    async def aclose(self) -> None:
        """Stop reading and close the response, not the SDK client.

        Closing during a pending read requests body closure, then waits until
        that reader has terminated before returning. After the close the
        pending iteration stops or an owned cancellation propagates; later
        finalization may report truncated or succeed from a fed completion.
        Unread events are discarded and finalization judges only fed events. No timeout-free promise is made
        for a transport whose read never ends.
        """
        self._closed = True
        await self._close_body()
        for reader_done in tuple(self._readers):
            await reader_done.wait()
        # A generator that was mid-read can be closed now that it has ended.
        await self._close_body()

    async def _close_body(self) -> None:
        # Shielded so a cancelled reader still releases its connection.
        with anyio.CancelScope(shield=True):
            if not self._response.is_closed:
                await self._response.aclose()
            chunks = self._chunks
            if chunks is not None and not getattr(chunks, "ag_running", False):
                try:
                    await chunks.aclose()  # type: ignore[attr-defined]
                except RuntimeError:
                    pass
