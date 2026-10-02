#!/usr/bin/env python3
"""Namespaced local arithmetic round trip: the caller owns policy, history and addition.

Usage: run.py --store PATH --model SLUG --receipt PATH

Refuses before any network send when the store is unreadable or group/world
accessible, lacks either plan-use scope, is expired, the receipt path is not
fresh, or a request breaks a captured preview restriction. Turn 1 offers the
literal SPEC D7 ``arithmetic`` namespace with ``tool_choice`` required and
``parallel_tool_calls`` false, and asks to add 17 and 25. Every completed
output item is then checked BEFORE any call is executed: each function call
must be namespace ``arithmetic``, name ``add``, status completed, carry a
nonblank unique ``call_id`` and arguments that parse to an object with exactly
the integers ``a`` = 17 and ``b`` = 25 (a JSON boolean is not an integer). A
refusal, a missing call, an unexpected call or any invalid call stops the run
with no second request and no receipt.

Only then does the script itself add ``a + b`` for each validated call. Turn 2
sends the original input, every completed turn-1 output item unchanged
(``ResponsesResult.to_input_items()``), one ``function_call_output`` per call
holding the JSON string of that sum under the exact ``call_id``, and one new
user item asking for the result, with the same offered tools and
``tool_choice`` none. The final answer must be completed text whose stripped
value is ``42``, with no new call and no refusal. The receipt is written only
then, mode 0600. Stdout, stderr and the receipt carry no token, response, item
or call identifier, and no encrypted content.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
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

INSTRUCTIONS = "Use the offered arithmetic tools for arithmetic. Answer briefly."
FIRST_PROMPT = "Use the arithmetic add tool to add 17 and 25."
SECOND_PROMPT = "Reply with only the result of adding 17 and 25."
NAMESPACE_NAME = "arithmetic"
FUNCTION_NAME = "add"
EXPECTED_A = 17
EXPECTED_B = 25
EXPECTED_FINAL = "42"

# SPEC D7, verbatim: the exact offered tool. Deep-copied into each request.
NAMESPACE_TOOL: dict[str, Any] = {
    "type": "namespace",
    "name": "arithmetic",
    "description": "Local arithmetic only.",
    "tools": [
        {
            "type": "function",
            "name": "add",
            "description": "Add two integers.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        }
    ],
}


def _user_item(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _request(model: str, input_items: list[dict[str, Any]], tool_choice: str) -> ResponsesRequest:
    # Every field except ``input`` and ``tool_choice`` is identical across the two turns.
    return ResponsesRequest(
        model=model,
        input=input_items,
        instructions=INSTRUCTIONS,
        store=False,
        reasoning={"effort": "low"},
        include=["reasoning.encrypted_content"],
        tools=[copy.deepcopy(NAMESPACE_TOOL)],
        tool_choice=tool_choice,
        parallel_tool_calls=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key in arguments")
    return dict(pairs)


def _validated_call(item: dict[str, Any], seen: set[str]) -> tuple[str, int, int]:
    """Return ``(call_id, a, b)`` for one acceptable call; ValueError otherwise.

    Nothing is executed here; the caller validates every call first.
    """
    if item.get("status") != "completed":
        raise ValueError("function call is not completed")
    if item.get("namespace") != NAMESPACE_NAME or item.get("name") != FUNCTION_NAME:
        raise ValueError("function call is not arithmetic/add")
    call_id = item.get("call_id")
    if type(call_id) is not str or not call_id.strip():
        raise ValueError("call_id is blank or not a string")
    if call_id in seen:
        raise ValueError("call_id is not unique")
    arguments = item.get("arguments")
    if type(arguments) is not str:
        raise ValueError("arguments are not a JSON string")
    parsed = json.loads(arguments, object_pairs_hook=_reject_duplicate_keys)
    if type(parsed) is not dict or set(parsed) != {"a", "b"}:
        raise ValueError("arguments are not an object whose only keys are a and b")
    a, b = parsed["a"], parsed["b"]
    if type(a) is not int or type(b) is not int:  # a JSON boolean is not an integer here
        raise ValueError("arguments are not integers")
    if (a, b) != (EXPECTED_A, EXPECTED_B):
        raise ValueError("arguments are not the requested pair")
    seen.add(call_id)
    return call_id, a, b


def _validate_first_turn(result: ResponsesResult) -> list[tuple[str, int, int]] | None:
    """Validate ALL completed output items; None (after a fixed label) on any problem."""
    if result.refusals:
        print("turn 1: refusal in the response", file=sys.stderr)
        return None
    calls: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for item in result.output:
        kind = item.get("type")
        if kind == "function_call":
            try:
                calls.append(_validated_call(item, seen))
            except ValueError:  # fixed label only: messages may carry raw provider text
                print("turn 1: invalid function call", file=sys.stderr)
                return None
        elif isinstance(kind, str) and kind.endswith("_call"):
            print("turn 1: unexpected tool call", file=sys.stderr)
            return None
    if not calls:
        print("turn 1: no function call returned", file=sys.stderr)
        return None
    return calls


def _add_outputs(calls: list[tuple[str, int, int]]) -> list[dict[str, Any]]:
    """The caller-owned execution: add each validated pair and build its output item."""
    outputs = []
    for call_id, a, b in calls:
        total = a + b
        outputs.append({"type": "function_call_output", "call_id": call_id, "output": json.dumps(total)})
    return outputs


def _turn_record(result: ResponsesResult) -> dict[str, Any]:
    return {
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


def _final_answer(result: ResponsesResult) -> str | None:
    """The stripped final text when it is plain and correct; None after a printed stop."""
    if result.refusals:
        print("turn 2: refusal in the response", file=sys.stderr)
        return None
    if any(isinstance(item.get("type"), str) and item["type"].endswith("_call") for item in result.output):
        print("turn 2: unexpected tool call", file=sys.stderr)
        return None
    text = result.output_text.strip()
    if text != EXPECTED_FINAL:
        print("turn 2: final text is not the expected result", file=sys.stderr)
        return None
    return text


async def _converse(access_token: str, model: str, first_request: ResponsesRequest) -> dict[str, Any] | None:
    """Run both turns; return the receipt document or None after a printed stop."""
    first_input = copy.deepcopy(first_request.input)
    async with ResponsesClient(bearer=access_token) as client:
        first = await _turn(client, first_request, "turn 1")
        if first is None:
            return None
        calls = _validate_first_turn(first)  # every call validated before any is executed
        if calls is None:
            return None
        outputs = _add_outputs(calls)
        # Replay: original input, every completed turn-1 item unchanged, one output per call, new user item.
        replay = first.to_input_items()
        second_input = [*first_input, *replay, *outputs, _user_item(SECOND_PROMPT)]
        second_request = _request(model, second_input, "none")
        if siwc_preview_violations(second_request.to_body()):
            print("refused: second request breaks the preview restrictions", file=sys.stderr)
            return None
        second = await _turn(client, second_request, "turn 2")
        if second is None:
            return None
        answer = _final_answer(second)
        if answer is None:
            return None
    return {
        "model": model,
        "call_count": len(calls),
        "executed_sums": [a + b for _, a, b in calls],
        "final_text": answer,
        "first_turn": _turn_record(first),
        "second_turn": _turn_record(second),
        "replayed_item_count": len(replay),
        "second_input_item_count": len(second_input),
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _write_receipt(receipt: Path, document: dict[str, Any]) -> None:
    # O_EXCL: the receipt is fresh, never an overwrite or a write through a link.
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Namespaced local arithmetic round trip over a SIWC store.")
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
    first_request = _request(args.model, [_user_item(FIRST_PROMPT)], "required")
    if siwc_preview_violations(first_request.to_body()):
        print("refused: request breaks the preview restrictions", file=sys.stderr)
        return 2
    if not receipt.parent.is_dir():
        print("refused: receipt directory does not exist", file=sys.stderr)
        return 2
    if os.path.lexists(receipt):
        print("refused: receipt path already exists", file=sys.stderr)
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
