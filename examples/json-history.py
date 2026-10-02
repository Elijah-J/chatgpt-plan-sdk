#!/usr/bin/env python3
"""Two strict-JSON turns with explicit replay of the first turn's output items.

Usage: json-history.py --store PATH --model SLUG --receipt PATH

Refuses before any network send when the store is unreadable or group/world
accessible, lacks either plan-use scope, is expired, or the first request
breaks a captured preview restriction. Turn 1 asks for a JSON object with an
integer value of 17. Turn 2 sends the original input, every completed turn-1
output item unchanged (``ResponsesResult.to_input_items()``) and one new user
item asking for the previous value plus 25. Each answer is decoded with
``ResponsesResult.output_json()`` and validated locally against the literal
schema. The receipt is written only when both turns pass. Stdout, stderr and
the receipt carry no token, response or item identifier, or encrypted content.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from chatgpt_plan_sdk import (
    ResponsesAPIError,
    ResponsesClient,
    ResponsesRequest,
    ResponsesResult,
    ResponsesStreamIncomplete,
    SiwcAuthError,
    SiwcCredentialStore,
    siwc_preview_violations,
)

INSTRUCTIONS = "Answer only with the JSON object that the response schema describes."
FIRST_PROMPT = "Return a JSON object whose value field is the integer 17."
SECOND_PROMPT = "Return a JSON object whose value field is the previous value plus 25."
FIRST_EXPECTED = 17
SECOND_EXPECTED = 42

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}
TEXT = {
    "format": {
        "type": "json_schema",
        "name": "sdk_history_value",
        "schema": SCHEMA,
        "strict": True,
    }
}


def _user_item(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _request(model: str, input_items: list[dict[str, Any]]) -> ResponsesRequest:
    # Every field except ``input`` is identical across the two turns.
    return ResponsesRequest(
        model=model,
        input=input_items,
        instructions=INSTRUCTIONS,
        store=False,
        reasoning={"effort": "low"},
        include=["reasoning.encrypted_content"],
        text=TEXT,
    )


def _local_value(answer: Any) -> int:
    """Validate the decoded answer against the literal schema; return its value."""
    if type(answer) is not dict or set(answer) != {"value"}:
        raise ValueError("answer is not an object whose only key is value")
    value = answer["value"]
    if type(value) is not int:  # a JSON boolean is not an integer here
        raise ValueError("value is not an integer")
    return value


def _turn_record(result: ResponsesResult, value: int, expected: int) -> dict[str, Any]:
    return {
        "value": value,
        "expected_value": expected,
        "matches_expected": value == expected,
        "status": result.status,
        "output_item_types": [item.get("type") for item in result.output],
        "input_tokens": result.input_tokens,
        "cached_tokens": result.cached_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "total_tokens": result.total_tokens,
        "event_counts": result.event_counts,
    }


async def _turn(client: ResponsesClient, request: ResponsesRequest, label: str) -> ResponsesResult | None:
    async with client.stream(request) as stream:
        result = await stream.get_final_response()
    if result.status != "completed":
        print(f"{label}: result is not completed", file=sys.stderr)
        return None
    return result


def _answer(result: ResponsesResult, label: str) -> int | None:
    try:
        return _local_value(result.output_json())
    except ValueError:  # fixed label only: messages may carry raw provider text
        print(f"{label}: answer rejected (refusal, invalid JSON or schema mismatch)", file=sys.stderr)
        return None


async def _converse(access_token: str, model: str, first_request: ResponsesRequest) -> dict[str, Any] | None:
    """Run both turns; return the receipt document or None after a printed stop."""
    first_input = first_request.input
    async with ResponsesClient(bearer=access_token) as client:
        first = await _turn(client, first_request, "turn 1")
        if first is None:
            return None
        first_value = _answer(first, "turn 1")
        if first_value is None:
            return None
        # Replay: original input, every completed turn-1 item unchanged, new user item.
        replay = first.to_input_items()
        second_input = [*first_input, *replay, _user_item(SECOND_PROMPT)]
        second_request = _request(model, second_input)
        if siwc_preview_violations(second_request.to_body()):
            print("refused: second request breaks the preview restrictions", file=sys.stderr)
            return None
        second = await _turn(client, second_request, "turn 2")
        if second is None:
            return None
        second_value = _answer(second, "turn 2")
        if second_value is None:
            return None
    return {
        "model": model,
        "first_turn": _turn_record(first, first_value, FIRST_EXPECTED),
        "second_turn": _turn_record(second, second_value, SECOND_EXPECTED),
        "replayed_item_count": len(replay),
        "second_input_item_count": len(second_input),
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _write_receipt(receipt: Path, document: dict[str, Any]) -> None:
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two strict-JSON turns with explicit history replay over a SIWC store.")
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
    first_request = _request(args.model, [_user_item(FIRST_PROMPT)])
    if siwc_preview_violations(first_request.to_body()):
        print("refused: request breaks the preview restrictions", file=sys.stderr)
        return 2
    if not receipt.parent.is_dir():
        print("refused: receipt directory does not exist", file=sys.stderr)
        return 2

    try:
        document = asyncio.run(_converse(credentials.access_token, args.model, first_request))
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
    print(f"completed: model={args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
