"""Offline checks for the synthetic image-input smoke consumer (hermetic: no network, no real store).

Every response is synthetic (``prodvision`` markers). The consumer runs
in-process with the HTTPX async transport replaced by a recording witness and
sockets refused; nothing here reads a credential, a real store or a live
endpoint. The frozen oracle under ``verification/vision`` owns the exact-body,
PNG and source-census checks; these cells cover the consumer's own judging,
receipt and failure paths.
"""

import datetime as dt
import importlib.util
import json
import os
import socket
import stat
from pathlib import Path

import httpx
import pytest

from chatgpt_plan_sdk import ResponsesResult

CONSUMER = Path(__file__).resolve().parents[2] / "examples" / "vision" / "run.py"
MODEL = "gpt-prod-vision-model"
ACCESS = "tok-prodvision-access"
PLAN = ["resource.invoke", "chatgpt.tokens.use.direct"]
MARKERS = ("tok-prodvision", "rs_prodvision", "msg_prodvision", "resp_prodvision", "ENC-PRODVISION")
RECEIPT_KEYS = {
    "model", "final_text", "status", "output_item_types", "input_tokens", "cached_tokens",
    "output_tokens", "reasoning_tokens", "total_tokens", "event_counts", "written_at",
}


def reasoning():
    return {"type": "reasoning", "id": "rs_prodvision_1", "summary": [], "encrypted_content": "ENC-PRODVISION-1"}


def message(text=None, *, refusal=None):
    content = []
    if text is not None:
        content.append({"type": "output_text", "text": text, "annotations": []})
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    return {"type": "message", "id": "msg_prodvision_1", "role": "assistant", "status": "completed", "content": content}


def response_object(items, status="completed"):
    return {
        "id": "resp_prodvision_1",
        "object": "response",
        "status": status,
        "model": MODEL,
        "output": items,
        "usage": {
            "input_tokens": 90,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 96,
        },
    }


def sse(items, *, deltas=(), terminal="completed"):
    records = [{"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": d} for d in deltas]
    records += [{"type": "response.output_item.done", "output_index": i, "item": item} for i, item in enumerate(items)]
    if terminal is not None:
        records.append({"type": f"response.{terminal}", "response": response_object(items, terminal)})
    return "".join(f"event: {r['type']}\ndata: {json.dumps(r)}\n\n" for r in records).encode("utf-8")


def make_store(tmp_path, **changes):
    raw = {
        "access_token": ACCESS,
        "refresh_token": "tok-prodvision-refresh",
        "client_id": "app_prodvision",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scopes": PLAN,
        "saved_at": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)).isoformat(),
    }
    raw.update(changes)
    directory = tmp_path / "store"
    directory.mkdir(mode=0o700)
    path = directory / "cred.json"
    path.write_text(json.dumps(raw))
    os.chmod(path, 0o600)
    return path


class Witness:
    """Record actual outgoing bodies; serve one scripted reply (or a status code); refuse sockets."""

    def __init__(self, monkeypatch, script=b"", status=200):
        self.bodies = []

        async def handle_async(transport, request):
            self.bodies.append(json.loads(request.content))
            if status != 200:
                return httpx.Response(status, headers={"content-type": "application/json"}, content=b"{}")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=script)

        def refuse(*args, **kwargs):
            raise OSError("socket use refused")

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
        monkeypatch.setattr(socket.socket, "connect", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)


def load_consumer():
    spec = importlib.util.spec_from_file_location("prodvision_consumer", CONSUMER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(tmp_path, capsys, store, receipt=None):
    receipt = receipt or tmp_path / "receipt.json"
    code = load_consumer().main(["--store", str(store), "--model", MODEL, "--receipt", str(receipt)])
    captured = capsys.readouterr()
    return code, receipt, captured.out + captured.err


def result(items, status="completed"):
    return ResponsesResult(response=response_object(items, status), output=tuple(items), event_counts={})


@pytest.mark.parametrize(
    ("items", "status", "expected"),
    [
        ([message("blue")], "completed", "blue"),
        ([reasoning(), message("  Blue ")], "completed", "blue"),
        ([message("BLUE\n")], "completed", "blue"),
        ([message("red")], "completed", None),
        ([message("blue!")], "completed", None),
        ([message("dark blue")], "completed", None),
        ([message("blue", refusal="I cannot help with that.")], "completed", None),
        ([message("blue")], "incomplete", None),
        ([reasoning()], "completed", None),
        ([message("")], "completed", None),
    ],
    ids=["plain", "padded-mixed-case", "upper-newline", "wrong-color", "punctuated", "extra-word",
         "refusal-part", "not-completed", "no-message", "empty-text"],
)
def test_judge_accepts_only_a_completed_refusal_free_exact_color(items, status, expected, capsys):
    assert load_consumer()._judge(result(items, status)) == expected
    if expected is None:
        assert capsys.readouterr().err.strip()


def test_request_is_one_user_item_with_the_unchanged_image_and_no_answer_disclosed():
    module = load_consumer()
    body = module._request(MODEL).to_body()
    assert set(body) == {"model", "input", "instructions", "store", "reasoning"}
    (item,) = body["input"]
    text_part, image_part = item["content"]
    assert image_part == {"type": "input_image", "image_url": module.DATA_URL, "detail": "low"}
    assert text_part["type"] == "input_text" and set(text_part) == {"type", "text"}
    for text in (body["instructions"], text_part["text"]):
        assert text.strip() and module.EXPECTED_ANSWER not in text.lower()
    assert "color" in text_part["text"].lower()


def test_success_writes_a_fresh_0600_receipt_with_only_the_allowed_keys(tmp_path, capsys, monkeypatch):
    witness = Witness(monkeypatch, sse([reasoning(), message("blue")], deltas=["bl", "ue"]))
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert code == 0, printed
    assert len(witness.bodies) == 1
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert set(document) == RECEIPT_KEYS
    assert document["final_text"] == "blue" and document["status"] == "completed"
    assert document["total_tokens"] == 96 and document["reasoning_tokens"] == 3
    assert not [m for m in MARKERS if m in receipt.read_text(encoding="utf-8") + printed]


@pytest.mark.parametrize(
    "script",
    [
        sse([message("blue", refusal="I cannot describe this image.")]),
        sse([], deltas=["blue"], terminal=None),
        sse([message("red")], deltas=["blue"]),
        sse([message("blue")], terminal="failed"),
    ],
    ids=["refusal", "delta-only", "delta-blue-final-red", "failed-terminal"],
)
def test_no_success_without_a_completed_exact_final_result(tmp_path, capsys, monkeypatch, script):
    witness = Witness(monkeypatch, script)
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert isinstance(code, int) and code != 0, printed
    assert len(witness.bodies) == 1
    assert not receipt.exists()
    assert not [m for m in MARKERS if m in printed]


def test_http_error_exits_nonzero_with_the_status_and_no_receipt(tmp_path, capsys, monkeypatch):
    witness = Witness(monkeypatch, status=500)
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert code == 1 and "HTTP 500" in printed
    assert len(witness.bodies) == 1 and not receipt.exists()


def test_missing_receipt_directory_refuses_before_any_send(tmp_path, capsys, monkeypatch):
    witness = Witness(monkeypatch, sse([message("blue")]))
    receipt = tmp_path / "absent-dir" / "receipt.json"
    code, _, printed = run(tmp_path, capsys, make_store(tmp_path), receipt)
    assert code == 2 and "receipt directory" in printed
    assert witness.bodies == [] and not receipt.parent.exists()


def test_dangling_symlink_receipt_path_refuses_before_any_send(tmp_path, capsys, monkeypatch):
    witness = Witness(monkeypatch, sse([message("blue")]))
    receipt = tmp_path / "receipt.json"
    receipt.symlink_to(tmp_path / "nowhere")
    code, _, printed = run(tmp_path, capsys, make_store(tmp_path), receipt)
    assert code == 2 and "already exists" in printed
    assert witness.bodies == [] and not (tmp_path / "nowhere").exists()
