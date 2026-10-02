# ChatGPT Plan SDK

An unofficial Python SDK for calling OpenAI's public Responses API through
[Sign in with ChatGPT](https://developers.openai.com/siwc/token-sharing-open-source),
using an eligible ChatGPT plan. Stream responses, discover available models,
replay completed output items, and explicitly renew a grant.

The package is self-contained. Its runtime dependencies are `httpx`, `pydantic`
and `anyio`; it requires Python 3.13 and a POSIX system for the credential
store's `fcntl` locks. It performs no automatic sign-in, model selection,
retries, refresh, history management or tool execution.

## Install

From this repository:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install .
```

The distribution is `chatgpt-plan-sdk`; the import is `chatgpt_plan_sdk`.

## Authentication

Obtain an app-owned OAuth grant through OpenAI's supported
[sign-in flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in).
ChatGPT plan usage requires both `resource.invoke` and
`chatgpt.tokens.use.direct`. Use your application's own identity and keep
persistent credentials local and under the user's control.

This SDK consumes the grant; it does not implement the first sign-in flow or
read Codex CLI credentials. See the
[SIWC terms](https://openai.com/policies/sign-in-with-chatgpt-terms/) for
eligibility, connected-application requirements and usage limits. Subscription
access is for the connected application, and does not grant extra usage.

Pass the resulting access token explicitly as `bearer`, or use
`SiwcCredentialStore` with an explicit protected file. The streaming example
below assumes `access_token` already contains that app-owned token. Choose a
model available in your signed-in account's catalog.

## Streaming

`ResponsesClient(bearer=...)` takes an access token or a zero-argument
provider, evaluated once per endpoint request. A `bearer` that is neither a
string nor a callable raises `TypeError` at construction; a blank bearer, a
non-string provider result, or a value containing CR, LF or NUL raises
`ValueError` before anything is sent. `stream(request)` returns a context manager
whose entry sends one `POST /v1/responses` with the request's native fields
plus `stream: true`. `store` is never added or changed, and a request that
supplies its own `stream` field is refused.

```python
request = ResponsesRequest(
    model="gpt-6-astra",
    input=[{"role": "user", "content": [{"type": "input_text", "text": "Say pong."}]}],
    instructions="Reply with exactly one word.",
    store=False,
)
async with ResponsesClient(bearer=access_token) as client:
    async with client.stream(request) as stream:
        async for event in stream:          # ResponsesSSEEvent(event, data)
            print(event.event, event.data)
        result = await stream.get_final_response()
print(result.output_text, result.total_tokens)
```

Each SSE record yields a `ResponsesSSEEvent` with the native event name and
decoded JSON; unknown events and fields pass through, and `[DONE]` ends
iteration without being yielded. `get_final_response()` returns a
`ResponsesResult` only when exactly one `response.completed` (mapping response,
status `completed`) was fed and no failure, protocol violation, fault or
cancellation; otherwise it raises `ResponsesStreamIncomplete` whose `kind` is
`failed`, `incomplete`, `error`, `truncated` or `protocol`. `result.response` is the
untouched terminal object; `result.output` holds the completed output items
assembled from `response.output_item.done` events; when no done item arrived it
falls back to a copy of the terminal response's `output` array. `aclose()`, leaving the
context, exhaustion, faults and cancellation close the response and leave the
client open. `aclose()` during a pending read closes the body, then returns
only after that reader has terminated; it makes no promise for a caller
transport whose read never ends. A non-2xx stream body is read on entry and
iteration raises `ResponsesAPIError`. Requests are sent once, without
redirects or inherited HTTPX auth; the SDK timeout (default 600 s) applies per
request even through a supplied `http_client`. Every request the SDK builds
carries only the headers host, accept, user-agent, content-length,
content-type and `authorization: Bearer <token>`: the HTTP client's default
headers and cookie jar are not forwarded. Hooks, event handlers and
transports on a supplied `http_client` are caller-owned and outside that
guarantee. `request_id` on a stream or `ResponsesAPIError` is the
`x-request-id` response header.

`result.output_json()` returns `json.loads(result.output_text)` and raises
`ValueError` when the result holds a refusal part, or when the text is empty or
not valid JSON; it validates no schema, so the caller checks the shape.
`result.to_input_items()` returns a new list of deep copies of every completed
item in `result.output`, in order and unchanged (unknown keys, `phase`,
`encrypted_content` and unknown item types included), never read from
`result.response`. Explicit history replay is the caller's concatenation into
the next request: `input=[*previous_input, *result.to_input_items(), new_user_item]`.

`await client.list_models()` returns the raw `httpx.Response`;
`ModelCatalog.from_response(response)` exposes `.models` and
`.visible_slugs()`; a successful response without a `models` array of objects
raises `ValueError`.

## Credentials

`SiwcCredentialStore(path)` works on one explicit protected file:
`load()`, `save(credentials)` and `refresh(http_client=None)`, all synchronous.
`load()` refuses a file with any group or world permission bit and, before
reading it, a containing directory with any group or world permission bit
(`store_permissions`); `save()` and `refresh()` refuse such a parent directory
too. `save()` and `refresh()` take a nonblocking exclusive flock on
`<path>.lock` (refused as `store_locked` when held), write an owner-only
temporary file, fsync it, replace the target and fsync the directory, all under
the lock. `refresh()` first refuses the bootstrap client id
`dynamic_agent_client` as `invalid_client` with no HTTP send, then sends one
form POST to the OpenAI OAuth token endpoint (no scope, no retry, no
redirects) and persists a validated replacement, including one that lost the
plan-use scopes. An `id_token` in the response is an optional hint: a nonblank
string replaces the stored hint, any other value leaves it. A record whose
`saved_at + expires_in` is beyond `datetime` range is valid: `is_expired` stays
exact and `redacted()["expires_at"]` is `None`. No store size limit applies. Failures raise `SiwcAuthError` with a safe `.code`
(`invalid_grant` and the other unusable-token codes, `invalid_client`,
`other`, `refresh_uncertain`, `persistence_uncertain`, or a local label);
`persistence_uncertain` retains the valid `.replacement`. Recovery
and reauthorization belong to the operator.

`siwc_preview_violations(body)` reports the captured SIWC preview restrictions
a request body breaks; consumers refuse before sending.

## Examples

The existing examples use an explicit protected credential store and an
explicit model slug. They run after installing this package:

```sh
.venv/bin/python examples/refresh.py --store /absolute/private/path/credentials.json
.venv/bin/python examples/ask.py --store /absolute/private/path/credentials.json --model MODEL_SLUG --receipt /absolute/private/path/ask.json
.venv/bin/python examples/json-history.py --store /absolute/private/path/credentials.json --model MODEL_SLUG --receipt /absolute/private/path/history.json
.venv/bin/python examples/tools/run.py --store /absolute/private/path/credentials.json --model MODEL_SLUG --receipt /absolute/private/path/tools.json
.venv/bin/python examples/vision/run.py --store /absolute/private/path/credentials.json --model MODEL_SLUG --receipt /absolute/private/path/vision.json
```

Use `--help` for each script's arguments. The store and its parent directory
must be accessible only to their owner.

`ask.py` refuses before sending unless the store has the required scopes,
is unexpired and passes the request checker; it reads the catalog, streams
a one-word answer and writes a completion receipt. `refresh.py` renews once
and prints only a safe summary. The other examples demonstrate strict JSON
with explicit history, a local arithmetic tool round trip, and a synthetic
image input. Tools execute in the example, not in the SDK.

## Tests

```sh
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest -q tests
```

The suite uses synthetic tokens, temporary credential stores and mocked HTTP
transports. It does not need live credentials.

## License

[MIT](LICENSE). This project is independent of OpenAI and is not endorsed by it.
OpenAI service use remains subject to its applicable terms.
