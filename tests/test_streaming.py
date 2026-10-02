"""Offline checks for SSE framing, terminal judgement and close custody."""

import asyncio
import json

import httpx
import pytest

from chatgpt_plan_sdk import ResponsesClient, ResponsesRequest, ResponsesStreamIncomplete

REQUEST = ResponsesRequest(model="m", input=[])


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks, *, hold=False):
        self.chunks = chunks
        self.hold = hold
        self.closed = False
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.hold:
            self.waiting.set()
            await self.release.wait()

    async def aclose(self):
        self.closed = True
        self.release.set()


def sse(payload, name=None):
    head = f"event: {name}\n" if name else ""
    return (head + "data: " + json.dumps(payload) + "\n\n").encode()


def completed(status="completed", output=None):
    response = {"id": "r", "status": status, "output": [] if output is None else output}
    return sse({"type": "response.completed", "response": response})


def item(index, text):
    body = {"type": "message", "content": [{"type": "output_text", "text": text}]}
    return sse({"type": "response.output_item.done", "output_index": index, "item": body})


async def open_stream(body):
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=body))
    )
    return ResponsesClient(bearer="tok", http_client=http)


def final(chunks):
    async def scenario():
        body = Body(chunks)
        client = await open_stream(body)
        async with client.stream(REQUEST) as stream:
            result = await stream.get_final_response()
        return result, body

    return asyncio.run(scenario())


def verdict(chunks):
    async def scenario():
        client = await open_stream(Body(chunks))
        async with client.stream(REQUEST) as stream:
            with pytest.raises(ResponsesStreamIncomplete) as caught:
                await stream.get_final_response()
        return caught.value.kind

    return asyncio.run(scenario())


def test_items_are_ordered_by_index_and_text_is_joined():
    result, body = final([item(1, "ng"), item(0, "po"), completed(), b"data: [DONE]\n\n"])
    assert result.output_text == "pong" and body.closed
    assert result.event_counts == {"response.output_item.done": 2, "response.completed": 1}


def test_split_chunks_crlf_bom_and_multibyte_frame_correctly():
    wire = b"\xef\xbb\xbfevent: x\r\ndata: {\"t\":\"caf\xc3\xa9\",\r\ndata: \"u\":1}\r\n\r\n"

    async def scenario():
        client = await open_stream(Body([wire[i : i + 1] for i in range(len(wire))]))
        async with client.stream(REQUEST) as stream:
            return [(e.event, e.data) async for e in stream]

    assert asyncio.run(scenario()) == [("x", {"t": "café", "u": 1})]


def test_terminal_kinds():
    assert verdict([sse({"type": "response.created"})]) == "truncated"
    assert verdict([completed(), completed()]) == "protocol"
    assert verdict([completed(status="in_progress")]) == "protocol"
    assert verdict([item(0, "a"), item(0, "b"), completed()]) == "protocol"
    assert verdict([item(True, "a"), completed()]) == "protocol"
    assert verdict([completed(output={"not": "array"})]) == "protocol"
    failed = sse({"type": "response.failed", "response": {"status": "failed"}})
    assert verdict([failed, completed()]) == "failed"
    assert verdict([completed(), sse({"type": "error", "message": "m"})]) == "error"


def test_fallback_output_comes_from_the_terminal_response():
    result, _ = final([completed(output=[{"type": "message", "content": []}])])
    assert result.output == ({"type": "message", "content": []},)


def test_events_are_copies_and_close_leaves_the_client_open():
    async def scenario():
        body = Body([item(0, "po"), completed()])
        client = await open_stream(body)
        async with client.stream(REQUEST) as stream:
            async for event in stream:
                if "item" in event.data:
                    event.data["item"]["content"][0]["text"] = "CHANGED"
            result = await stream.get_final_response()
        open_after = not client._http.is_closed
        await client.close()
        return result, open_after

    result, open_after = asyncio.run(scenario())
    assert result.output_text == "po" and open_after


def test_held_open_body_is_not_success_until_eof_and_close_judges_fed_events():
    async def scenario():
        body = Body([completed()], hold=True)
        client = await open_stream(body)
        async with client.stream(REQUEST) as stream:
            final_task = asyncio.create_task(stream.get_final_response())
            await asyncio.wait_for(body.waiting.wait(), 5)
            for _ in range(10):
                await asyncio.sleep(0)
            pending = not final_task.done()
            await stream.aclose()
            result = await asyncio.wait_for(final_task, 5)
        return pending, result.status, body.closed

    assert asyncio.run(scenario()) == (True, "completed", True)


def test_cancellation_propagates_and_later_finalization_is_truncated():
    async def scenario():
        body = Body([sse({"type": "response.created"})], hold=True)
        client = await open_stream(body)
        async with client.stream(REQUEST) as stream:
            iterator = stream.__aiter__()
            await anext(iterator)
            pending = asyncio.create_task(anext(iterator))
            await asyncio.wait_for(body.waiting.wait(), 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert body.closed
            with pytest.raises(ResponsesStreamIncomplete) as caught:
                await stream.get_final_response()
        return caught.value.kind

    assert asyncio.run(scenario()) == "truncated"


def test_transport_fault_propagates_and_stays_a_fault():
    class Failing(Body):
        async def __aiter__(self):
            yield completed()
            raise httpx.ReadError("synthetic")

    async def scenario():
        client = await open_stream(Failing([]))
        async with client.stream(REQUEST) as stream:
            with pytest.raises(httpx.ReadError):
                await stream.get_final_response()
            with pytest.raises(httpx.ReadError):
                await stream.get_final_response()

    asyncio.run(scenario())


class DelayedReaderExit(httpx.AsyncByteStream):
    """Body whose closure and whose reader's termination are distinct events."""

    def __init__(self):
        self.started = asyncio.Event()
        self.body_closed = asyncio.Event()
        self.allow_reader_exit = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        await self.body_closed.wait()
        await self.allow_reader_exit.wait()
        if False:
            yield b""

    async def aclose(self):
        self.body_closed.set()


@pytest.mark.parametrize("reader", ["anext", "get_final_response"])
def test_close_waits_for_the_pending_reader_to_terminate(reader):
    async def scenario():
        body = DelayedReaderExit()
        client = await open_stream(body)
        async with client.stream(REQUEST) as stream:
            pending = asyncio.create_task(
                anext(stream) if reader == "anext" else stream.get_final_response()
            )
            await asyncio.wait_for(body.started.wait(), 5)
            closing = asyncio.create_task(stream.aclose())
            await asyncio.wait_for(body.body_closed.wait(), 5)
            for _ in range(20):
                await asyncio.sleep(0.01)
            close_done_early = closing.done()
            pending_done_early = pending.done()
            body.allow_reader_exit.set()
            await asyncio.wait_for(closing, 5)
            reader_done_at_close = pending.done()
            outcome = await asyncio.gather(pending, return_exceptions=True)
        return close_done_early, pending_done_early, reader_done_at_close, outcome[0]

    early_close, early_pending, done_at_close, outcome = asyncio.run(scenario())
    assert (early_close, early_pending, done_at_close) == (False, False, True)
    if reader == "anext":
        assert isinstance(outcome, StopAsyncIteration)
    else:
        assert isinstance(outcome, ResponsesStreamIncomplete) and outcome.kind == "truncated"


def test_close_after_owned_cancellation_of_the_pending_reader_returns():
    async def scenario():
        body = DelayedReaderExit()
        client = await open_stream(body)
        async with client.stream(REQUEST) as stream:
            pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(body.started.wait(), 5)
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await asyncio.wait_for(stream.aclose(), 5)
        return pending.cancelled()

    assert asyncio.run(scenario()) is True


def test_parser_fault_after_completion_is_a_fault_not_success():
    wire = completed() + b"data: {not json\n\n"

    async def scenario():
        client = await open_stream(Body([wire]))
        async with client.stream(REQUEST) as stream:
            for _ in range(2):
                with pytest.raises(ValueError) as caught:
                    await stream.get_final_response()
                assert not isinstance(caught.value, ResponsesStreamIncomplete)

    asyncio.run(scenario())
