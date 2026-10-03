# ChatGPT Plan SDK

An unofficial Python SDK for OpenAI's Responses API using an eligible ChatGPT
plan through [Sign in with ChatGPT](https://developers.openai.com/siwc/token-sharing-open-source)
(SIWC). Stream responses, list models, replay completed output items, and
explicitly renew a stored grant.

Requires **Python 3.13 and POSIX** (`fcntl` credential-store locks).

## Install

From a clone of this repository:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install .
```

## Authentication

Obtain an app-owned OAuth grant through OpenAI's supported
[sign-in flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in).
It must include `resource.invoke` and `chatgpt.tokens.use.direct`. The SDK
consumes this grant; it does not implement first sign-in or read Codex credentials.

See the [SIWC terms](https://openai.com/policies/sign-in-with-chatgpt-terms/)
for eligibility and connected-application requirements. Your usual plan usage
limits apply; this grants no extra usage.

## Quick start

Set `SIWC_ACCESS_TOKEN` to your app-owned access token and `MODEL_SLUG` to a
model from your account's catalog, then run:

```python
import asyncio
import os

from chatgpt_plan_sdk import ResponsesClient, ResponsesRequest


async def main():
    request = ResponsesRequest(
        model=os.environ["MODEL_SLUG"],
        input=[{"role": "user", "content": [{"type": "input_text", "text": "Say pong."}]}],
        instructions="Reply with exactly one word.",
        store=False,
    )
    async with ResponsesClient(bearer=os.environ["SIWC_ACCESS_TOKEN"]) as client:
        async with client.stream(request) as stream:
            async for delta in stream.text_stream:
                print(delta, end="", flush=True)
            result = await stream.get_final_response()
    print(f"\nTokens: {result.total_tokens}")


asyncio.run(main())
```

`SiwcCredentialStore(path)` provides `load()`, `save()` and explicit
`refresh()` for a local credential file. Both the file and its parent must be
owner-only. Sign-in, model selection, retries, history and tool execution stay
with your application. Public-class docstrings describe the API details.

## Examples

Run each script with `--help` for its arguments:

- [ask.py](examples/ask.py): model catalog, streamed answer and receipt.
- [refresh.py](examples/refresh.py): renew a stored grant once.
- [json-history.py](examples/json-history.py): strict JSON and explicit history replay.
- [tools/run.py](examples/tools/run.py): a local function-tool round trip.
- [vision/run.py](examples/vision/run.py): synthetic image input.

## Tests

```sh
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest -q tests
```

Tests use synthetic tokens, temporary stores and mocked HTTP; no live credentials.

[MIT](LICENSE). Independent of OpenAI and not endorsed by it. OpenAI service
use remains subject to its applicable terms.
