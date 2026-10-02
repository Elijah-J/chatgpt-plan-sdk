"""Typed errors for Responses statuses and unfinished streams."""

from __future__ import annotations

import copy
from typing import Any

import httpx

__all__ = ["ResponsesAPIError", "ResponsesStreamIncomplete"]

_INCOMPLETE_KINDS = frozenset({"failed", "incomplete", "error", "truncated", "protocol"})


class ResponsesAPIError(httpx.HTTPStatusError):
    """A non-2xx Responses or model-catalog HTTP response.

    The rendered message carries only the HTTP status. ``error``, ``detail``,
    ``request_id`` and ``response`` are sensitive raw diagnostics: ``error`` and
    ``detail`` are the decoded body's ``error`` and ``detail`` values (None when
    absent or the body is not a JSON object), and ``request_id`` is the value of
    the ``x-request-id`` response header. Do not log them.
    """

    def __init__(self, response: httpx.Response, *, request: httpx.Request | None = None) -> None:
        if request is None:
            try:
                request = response.request
            except RuntimeError:
                request = httpx.Request("GET", "https://api.openai.com/v1/")
        message = f"OpenAI Responses request failed with HTTP status {response.status_code}"
        super().__init__(message, request=request, response=response)

    def _body(self) -> dict[str, Any] | None:
        try:
            body = self.response.json()
        except (ValueError, RuntimeError):
            return None
        return body if isinstance(body, dict) else None

    @property
    def error(self) -> Any:
        """The body's ``error`` value, or None."""
        body = self._body()
        return copy.deepcopy(body["error"]) if body is not None and "error" in body else None

    @property
    def detail(self) -> Any:
        """The body's ``detail`` value, or None."""
        body = self._body()
        return copy.deepcopy(body["detail"]) if body is not None and "detail" in body else None

    @property
    def request_id(self) -> str | None:
        """The ``x-request-id`` response header, or None."""
        return self.response.headers.get("x-request-id")


class ResponsesStreamIncomplete(Exception):
    """The fed events are not a successful terminal Responses result.

    ``kind`` is one of failed, incomplete, error, truncated or protocol. The
    rendered message carries only the kind. ``response`` and ``error`` are the
    sensitive raw terminal response or error event when one was fed.
    """

    def __init__(self, kind: str, *, response: Any = None, error: Any = None) -> None:
        if kind not in _INCOMPLETE_KINDS:
            raise ValueError("unknown stream incompleteness kind")
        super().__init__(f"Responses stream did not complete: {kind}")
        self.kind = kind
        self.response = response
        self.error = error
