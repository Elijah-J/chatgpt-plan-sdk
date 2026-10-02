#!/usr/bin/env python3
"""Synthetic image input: one request carrying a fixed 64x64 PNG data URL.

Usage: run.py --store PATH --model SLUG --receipt PATH

Refuses before any network send when the store is unreadable or group/world
accessible, lacks either plan-use scope, is expired, a request breaks a
captured preview restriction, or the receipt path is not fresh. There is no
catalog lookup and no refresh. Exactly one request is sent through the public
SDK surface: the CLI model, short instructions asking only for a color word,
``store`` false, low reasoning effort, and one user item holding exactly one
``input_text`` question about the image color and one ``input_image`` part
whose ``image_url`` is the fixed data URL below, sent unchanged with
``detail`` low. Neither the question nor the instructions names the color.

Success is judged only from the completed final result: status completed, no
refusal part, and the stripped, lowercased output text equal to the expected
color word. Streamed text deltas never decide success, so a delta-only or
truncated stream fails. The receipt is written only then, fresh and mode 0600;
stdout, stderr and the receipt carry no token, response or item identifier and
no encrypted content. Every other outcome exits nonzero with no receipt.
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

INSTRUCTIONS = "Answer with only one color word, in lowercase, and nothing else."
QUESTION = "What is the color of this image?"
EXPECTED_ANSWER = "blue"

# Root-measured fixture (64x64, 8-bit RGB, every pixel one solid color), sent unchanged.
DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAfElEQVR4nNXOQREAMAjAsK7+PTMRPLhGQR4MZRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncV4Htj5QhwF/4nP9ugAAAABJRU5ErkJggg=="  # noqa: E501


def _request(model: str) -> ResponsesRequest:
    image_part = {"type": "input_image", "image_url": DATA_URL, "detail": "low"}
    user_item = {"role": "user", "content": [{"type": "input_text", "text": QUESTION}, image_part]}
    return ResponsesRequest(
        model=model,
        input=[user_item],
        instructions=INSTRUCTIONS,
        store=False,
        reasoning={"effort": "low"},
    )


def _judge(result: ResponsesResult) -> str | None:
    """The stripped answer when the completed result is the expected color; None after a printed stop."""
    if result.status != "completed":
        print("result is not completed", file=sys.stderr)
        return None
    if result.refusals:
        print("refusal in the response", file=sys.stderr)
        return None
    answer = result.output_text.strip().lower()
    if answer != EXPECTED_ANSWER:
        print("final text is not the expected color", file=sys.stderr)
        return None
    return answer


async def _ask(access_token: str, model: str, request: ResponsesRequest) -> dict[str, Any] | None:
    """Send the one request; return the receipt document or None after a printed stop."""
    async with ResponsesClient(bearer=access_token) as client:
        async with client.stream(request) as stream:
            result = await stream.get_final_response()
    answer = _judge(result)
    if answer is None:
        return None
    return {
        "model": model,
        "final_text": answer,
        "status": result.status,
        "output_item_types": [item.get("type") for item in result.output],
        "input_tokens": result.input_tokens,
        "cached_tokens": result.cached_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "total_tokens": result.total_tokens,
        "event_counts": result.event_counts,
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _write_receipt(receipt: Path, document: dict[str, Any]) -> None:
    # O_EXCL: the receipt is fresh, never an overwrite or a write through a link.
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic image input over a SIWC store.")
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
    if siwc_preview_violations(request.to_body()):
        print("refused: request breaks the preview restrictions", file=sys.stderr)
        return 2
    if not receipt.parent.is_dir():
        print("refused: receipt directory does not exist", file=sys.stderr)
        return 2
    if os.path.lexists(receipt):
        print("refused: receipt path already exists", file=sys.stderr)
        return 2

    try:
        document = asyncio.run(_ask(credentials.access_token, args.model, request))
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
    try:
        _write_receipt(receipt, document)
    except OSError as exc:
        print(f"receipt write failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"completed: model={args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
