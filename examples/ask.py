#!/usr/bin/env python3
"""Ask one explicitly chosen model for "pong" over a protected SIWC store.

Usage: ask.py --store PATH --model SLUG --receipt PATH

Refuses before any network send when the store is unreadable or group/world
accessible, lacks either plan-use scope, is expired, or the request breaks a
captured preview restriction. Then reads the model catalog, streams the
response through ``ResponsesClient`` and writes a receipt derived from the
completed result. Exits 0 only with a completed result. Stdout, stderr and the
receipt carry no token, client or identity value.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path

from chatgpt_plan_sdk import (
    ModelCatalog,
    ResponsesAPIError,
    ResponsesClient,
    ResponsesRequest,
    ResponsesStreamIncomplete,
    SiwcAuthError,
    SiwcCredentialStore,
    siwc_preview_violations,
)

INSTRUCTIONS = "Reply with exactly one word."
PROMPT = "Say pong."


def _request(model: str) -> ResponsesRequest:
    return ResponsesRequest(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": PROMPT}]}],
        instructions=INSTRUCTIONS,
        store=False,
        reasoning={"effort": "low"},
    )


async def _ask(access_token: str, request: ResponsesRequest) -> dict | None:
    """Run the catalog read and the stream; return the receipt document or None."""
    async with ResponsesClient(bearer=access_token) as client:
        listing = await client.list_models()
        catalog = ModelCatalog.from_response(listing)
        if request.model not in {m.get("slug") for m in catalog.models}:
            print("model is not in the catalog", file=sys.stderr)
            return None
        async with client.stream(request) as stream:
            async for delta in stream.text_stream:
                print(delta, end="", flush=True)
            print()
            result = await stream.get_final_response()
            http_status = stream.status_code
    if result.status != "completed":
        print("result is not completed", file=sys.stderr)
        return None
    return {
        "model": request.model,
        "catalog_http_status": listing.status_code,
        "stream_http_status": http_status,
        "status": result.status,
        "output_text": result.output_text,
        "input_tokens": result.input_tokens,
        "cached_tokens": result.cached_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "total_tokens": result.total_tokens,
        "event_counts": result.event_counts,
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _write_receipt(receipt: Path, document: dict) -> None:
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a model for pong over a SIWC store.")
    parser.add_argument("--store", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    receipt = Path(args.receipt)

    # Every refusal below happens before any network send.
    try:
        credentials = SiwcCredentialStore(Path(args.store)).load()
    except SiwcAuthError as exc:
        print(f"refused: store {exc.code}", file=sys.stderr)
        return 2
    if not credentials.has_plan_use:
        print("refused: grant lacks a plan-use scope", file=sys.stderr)
        return 2
    if credentials.is_expired(dt.datetime.now(dt.timezone.utc)):
        print("refused: access token is expired; refresh explicitly first", file=sys.stderr)
        return 2
    request = _request(args.model)
    violations = siwc_preview_violations(request.to_body())
    if violations:
        print("refused: request breaks the preview restrictions", file=sys.stderr)
        return 2
    if not receipt.parent.is_dir():
        print("refused: receipt directory does not exist", file=sys.stderr)
        return 2

    try:
        document = asyncio.run(_ask(credentials.access_token, request))
    except ResponsesAPIError as exc:
        print(f"request failed: HTTP {exc.response.status_code}", file=sys.stderr)
        return 1
    except ResponsesStreamIncomplete as exc:
        print(f"request failed: stream {exc.kind}", file=sys.stderr)
        return 1
    except Exception as exc:  # type name only: messages may carry raw provider text
        print(f"request failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    if document is None:
        return 1
    _write_receipt(receipt, document)
    print(f"completed: model={request.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
