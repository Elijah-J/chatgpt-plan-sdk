"""Offline checks for ResponsesClient request policy."""

import asyncio
import json

import httpx
import pytest

from chatgpt_plan_sdk import ResponsesClient, ResponsesRequest

SSE = b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'


def _http(responder, **kwargs):
    seen = []

    def handler(request):
        seen.append(request)
        return responder(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs), seen


def _sse(request):
    return httpx.Response(200, content=SSE)


def test_stream_posts_once_on_entry_with_stream_true_and_explicit_bearer():
    async def scenario():
        http, seen = _http(_sse, auth=("u", "p"), headers={"Authorization": "Bearer inherited"})
        client = ResponsesClient(bearer="tok-1", http_client=http)
        context = client.stream(ResponsesRequest(model="m", input=[]))
        assert seen == []
        async with context as stream:
            await stream.get_final_response()
        await client.close()
        return seen

    (request,) = asyncio.run(scenario())
    assert request.method == "POST" and str(request.url) == "https://api.openai.com/v1/responses"
    assert json.loads(request.content) == {"model": "m", "input": [], "stream": True}
    assert request.headers["authorization"] == "Bearer tok-1"


def test_supplied_stream_field_is_refused_before_transport():
    async def scenario():
        http, seen = _http(_sse)
        client = ResponsesClient(bearer="tok", http_client=http)
        with pytest.raises(ValueError):
            client.stream(ResponsesRequest(model="m", input=[], stream=False))
        await client.close()
        return seen

    assert asyncio.run(scenario()) == []


@pytest.mark.parametrize(
    "bearer", ["  ", "a\nb", lambda: None, lambda: ""],
    ids=["whitespace", "newline", "non-string-provider", "blank-provider"],
)
def test_unusable_bearer_is_refused_locally(bearer):
    async def scenario():
        http, seen = _http(_sse)
        client = ResponsesClient(bearer=bearer, http_client=http)
        with pytest.raises(ValueError):
            await client.list_models()
        await client.close()
        return seen

    assert asyncio.run(scenario()) == []


def test_bearer_provider_runs_once_per_request_and_timeout_applies():
    calls = []

    def provider():
        calls.append(1)
        return f"tok-{len(calls)}"

    async def scenario():
        http, seen = _http(lambda r: httpx.Response(200, json={"models": []}), timeout=httpx.Timeout(1.0))
        client = ResponsesClient(bearer=provider, timeout=7.0, http_client=http)
        await client.list_models()
        await client.list_models()
        await client.close()
        return seen

    seen = asyncio.run(scenario())
    assert [r.headers["authorization"] for r in seen] == ["Bearer tok-1", "Bearer tok-2"]
    assert all(r.extensions["timeout"] == httpx.Timeout(7.0).as_dict() for r in seen)


def test_error_status_is_returned_by_list_models_and_redirects_are_not_followed():
    async def scenario():
        http, seen = _http(
            lambda r: httpx.Response(307, headers={"location": "https://other.invalid/"}),
            follow_redirects=True,
        )
        client = ResponsesClient(bearer="tok", http_client=http)
        response = await client.list_models()
        await client.close()
        return response.status_code, len(seen)

    assert asyncio.run(scenario()) == (307, 1)


ALLOWED_HEADERS = {
    "host", "accept", "accept-encoding", "connection", "user-agent",
    "content-length", "content-type", "authorization",
}


def _cookie_setting(request):
    headers = {"set-cookie": "affinity=SERVER; Path=/"}
    if request.method == "GET":
        return httpx.Response(200, headers=headers, json={"models": []})
    return httpx.Response(200, headers=headers, content=SSE)


async def _sequence(client):
    await client.list_models()
    async with client.stream(ResponsesRequest(model="m", input=[])) as stream:
        await stream.get_final_response()
    await client.list_models()


def test_supplied_client_default_headers_and_cookies_are_not_forwarded():
    async def scenario():
        http, seen = _http(
            _cookie_setting,
            headers={"ChatGPT-Account-Id": "acct", "X-Api-Key": "key", "X-Telemetry": "t"},
            cookies={"session": "jar"},
        )
        client = ResponsesClient(bearer="tok-1", http_client=http)
        await _sequence(client)
        await client.close()
        return seen

    seen = asyncio.run(scenario())
    assert [r.method for r in seen] == ["GET", "POST", "GET"]
    for request in seen:
        assert {name.lower() for name in request.headers.keys()} <= ALLOWED_HEADERS
        assert request.headers["authorization"] == "Bearer tok-1"
        assert "cookie" not in request.headers


def test_default_client_does_not_replay_response_cookies(monkeypatch):
    seen = []
    real = httpx.AsyncClient

    def handler(request):
        seen.append(request)
        return _cookie_setting(request)

    def factory(**kwargs):
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    async def scenario():
        async with ResponsesClient(bearer="tok-1") as client:
            await _sequence(client)

    asyncio.run(scenario())
    assert len(seen) == 3
    for request in seen:
        assert {name.lower() for name in request.headers.keys()} <= ALLOWED_HEADERS
        assert "cookie" not in request.headers
        assert request.extensions["timeout"] == httpx.Timeout(600.0).as_dict()
