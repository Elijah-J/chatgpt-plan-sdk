"""Async client for the Responses endpoints reachable with a SIWC grant."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Self

import httpx

from chatgpt_plan_sdk._streaming import ResponsesStream
from chatgpt_plan_sdk.models.request import ResponsesRequest

__all__ = ["ResponsesClient"]

_RESPONSES_URL = "https://api.openai.com/v1/responses"
_MODELS_URL = "https://api.openai.com/v1/models"
# The default HTTPX identity (httpx _client.py:119); the only non-negotiation
# header kept besides Authorization and Content-Type.
_USER_AGENT = f"python-httpx/{httpx.__version__}"


class _StreamContext:
    """Async context manager for one Responses stream.

    The POST begins on entry. Exit closes the response, not the client.
    """

    def __init__(self, open_response: Callable[[], Awaitable[httpx.Response]]) -> None:
        self._open_response = open_response
        self._entered = False
        self._stream: ResponsesStream | None = None

    async def __aenter__(self) -> ResponsesStream:
        if self._entered:
            raise RuntimeError("A Responses stream context can only be entered once.")
        self._entered = True
        stream = ResponsesStream(await self._open_response())
        self._stream = stream
        # An error body is not an SSE document: read it now so it stays
        # inspectable through the raw response.
        if not stream.response.is_success:
            try:
                await stream.response.aread()
            except BaseException:
                await stream.aclose()
                raise
        return stream

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            await self._stream.aclose()


class ResponsesClient:
    """Configure an async Responses client without making a request.

    ``bearer`` is an app-owned access token or a zero-argument provider; it is
    evaluated once per endpoint request and a blank or non-string value, or one
    containing CR, LF or NUL, is refused before anything is sent. ``timeout``
    (default 600 seconds) applies per HTTPX operation to every request,
    including through a supplied ``http_client``. Requests are sent once: no
    retries, no redirects and no inherited HTTPX auth. Every request the SDK
    builds carries only host, accept, user-agent, content-length,
    content-type and authorization (``Bearer <token>``): the HTTP client's
    default headers and cookie jar are not forwarded. Hooks, event handlers
    and transports on a supplied ``http_client`` are caller-owned and outside
    that guarantee. Closing this client also closes a supplied HTTP client.
    """

    def __init__(
        self,
        *,
        bearer: str | Callable[[], str],
        timeout: float | httpx.Timeout | None = 600.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if http_client is not None and not isinstance(http_client, httpx.AsyncClient):
            raise TypeError("http_client must be an httpx.AsyncClient")
        if not isinstance(bearer, str) and not callable(bearer):
            raise TypeError("bearer must be a string or a zero-argument callable")
        self._bearer = bearer
        self._timeout = httpx.Timeout(timeout)
        self._http = http_client if http_client is not None else httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=False
        )

    def stream(self, request: ResponsesRequest) -> _StreamContext:
        """Return an async context manager for one streamed response.

        The POST begins on context entry. The body is the request's native
        fields plus ``stream: true``; ``store`` is never added or changed. A
        request that supplies its own ``stream`` field is refused here.
        """
        if not isinstance(request, ResponsesRequest):
            raise TypeError("request must be a ResponsesRequest")
        body = request.to_body()
        if "stream" in body:
            raise ValueError("stream is owned by ResponsesClient.stream; remove it from the request")
        body["stream"] = True
        content = json.dumps(body, allow_nan=False).encode("utf-8")
        return _StreamContext(lambda: self._send("POST", _RESPONSES_URL, content))

    async def list_models(self) -> httpx.Response:
        """GET the model catalog and return the raw response, errors included."""
        return await self._send("GET", _MODELS_URL, None, stream=False)

    async def _send(
        self, method: str, url: str, content: bytes | None, *, stream: bool = True
    ) -> httpx.Response:
        if self._http.is_closed:
            raise RuntimeError("Cannot use a closed ResponsesClient.")
        bearer = self._bearer() if callable(self._bearer) else self._bearer
        if not isinstance(bearer, str) or not bearer.strip() or any(
            character in bearer for character in "\r\n\0"
        ):
            raise ValueError("bearer must be a nonblank string without control characters")
        headers = {
            "authorization": f"Bearer {bearer}",
            "accept": "text/event-stream" if stream else "application/json",
            "user-agent": _USER_AGENT,
        }
        if content is not None:
            headers["content-type"] = "application/json"
        # Build the request without ``AsyncClient.build_request``: that merges
        # the client's default headers and cookie jar (httpx _client.py:419,
        # :429), which would forward caller defaults and any cookie a response
        # set. A bare ``httpx.Request`` carries only these headers plus the
        # Host and Content-Length it derives itself, and the SDK timeout rides
        # its extension (httpx _client.py:377, _models.py:401), so it applies
        # even to a supplied client. Per-send auth=None and
        # follow_redirects=False disable inherited auth and redirects.
        wire_request = httpx.Request(
            method,
            url,
            headers=headers,
            content=content,
            extensions={"timeout": self._timeout.as_dict()},
        )
        return await self._http.send(
            wire_request, stream=stream, auth=None, follow_redirects=False
        )

    async def close(self) -> None:
        """Release the HTTP client, including a supplied one."""
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        if self._http.is_closed:
            raise RuntimeError("Cannot reopen a closed ResponsesClient.")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
