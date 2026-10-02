"""Offline checks for the json-history smoke consumer (hermetic: no network, no real store)."""

import datetime as dt
import importlib.util
import json
import os
import socket
import stat
from pathlib import Path

import httpx
import pytest

CONSUMER = Path(__file__).resolve().parents[1] / "examples" / "json-history.py"
MODEL = "gpt-prod-test-model"
ACCESS = "tok-prodtest-access"
PLAN = ["resource.invoke", "chatgpt.tokens.use.direct"]
MARKERS = ("tok-prodtest", "rs_prodtest", "msg_prodtest", "resp_prodtest", "ENC-PRODTEST")
SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}
TEXT = {"format": {"type": "json_schema", "name": "sdk_history_value", "schema": SCHEMA, "strict": True}}


def message(text=None, *, refusal=None, ident="msg_prodtest_a"):
    content = []
    if text is not None:
        content.append({"type": "output_text", "text": text})
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    return {"id": ident, "type": "message", "role": "assistant", "status": "completed", "content": content}


def reasoning():
    return {"id": "rs_prodtest_a", "type": "reasoning", "summary": [], "encrypted_content": "ENC-PRODTEST-opaque"}


def sse(items, ident="resp_prodtest_1"):
    records = [
        {"type": "response.output_item.done", "output_index": index, "item": item}
        for index, item in enumerate(items)
    ]
    response = {"id": ident, "object": "response", "status": "completed", "model": MODEL, "output": []}
    records.append({"type": "response.completed", "response": response})
    return "".join(f"event: {r['type']}\ndata: {json.dumps(r)}\n\n" for r in records).encode("utf-8")


def make_store(tmp_path, **changes):
    raw = {
        "access_token": ACCESS,
        "refresh_token": "tok-prodtest-refresh",
        "client_id": "app_prodtest",
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
    """Record actual outgoing bodies; serve scripted SSE bytes; refuse sockets."""

    def __init__(self, monkeypatch, scripts):
        self.bodies = []
        self._scripts = list(scripts)

        async def handle_async(transport, request):
            self.bodies.append(json.loads(request.content))
            index = len(self.bodies) - 1
            if index >= len(self._scripts):
                return httpx.Response(500, headers={"content-type": "application/json"}, content=b"{}")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=self._scripts[index])

        def refuse(*args, **kwargs):
            raise OSError("socket use refused")

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
        monkeypatch.setattr(socket.socket, "connect", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)


def load_consumer():
    spec = importlib.util.spec_from_file_location("prodtest_json_history_consumer", CONSUMER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(tmp_path, capsys, store, module=None):
    module = module or load_consumer()
    receipt = tmp_path / "receipt.json"
    code = module.main(["--store", str(store), "--model", MODEL, "--receipt", str(receipt)])
    captured = capsys.readouterr()
    return code, receipt, captured.out + captured.err


def test_two_turns_replay_first_items_unchanged_and_write_a_private_receipt(tmp_path, capsys, monkeypatch):
    first_items = [reasoning(), message('{"value": 17}')]
    witness = Witness(monkeypatch, [sse(first_items), sse([message('{"value": 42}', ident="msg_prodtest_b")], "resp_prodtest_2")])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert code == 0, printed
    first, second = witness.bodies
    for body in (first, second):
        assert body["model"] == MODEL and body["store"] is False and body["stream"] is True
        assert body["reasoning"] == {"effort": "low"}
        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["text"] == TEXT
        assert body["instructions"].strip() and "previous_response_id" not in body
    assert {k: v for k, v in second.items() if k != "input"} == {k: v for k, v in first.items() if k != "input"}
    assert len(first["input"]) == 1 and first["input"][0]["role"] == "user"
    assert second["input"][:1] == first["input"]
    assert second["input"][1:3] == first_items
    assert len(second["input"]) == 4 and second["input"][3]["role"] == "user"
    assert second["input"][3]["content"][0]["type"] == "input_text"
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert document["first_turn"]["value"] == 17 and document["second_turn"]["value"] == 42
    assert document["replayed_item_count"] == 2
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert not [m for m in MARKERS if m in receipt.read_text(encoding="utf-8") + printed]


@pytest.mark.parametrize(
    "first_items",
    [
        [message(refusal="no")],
        [message('{"value": 17')],
        [message('{"value": true}')],
        [message('{"value": "17"}')],
        [message('{"value": 17, "extra": 1}')],
        [message('[17]')],
        [reasoning()],
    ],
    ids=["refusal", "invalid-json", "boolean", "string", "extra-key", "array", "no-message"],
)
def test_first_turn_failure_makes_no_second_send_and_no_receipt(tmp_path, capsys, monkeypatch, first_items):
    witness = Witness(monkeypatch, [sse(first_items)])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert code == 1 and len(witness.bodies) == 1 and not receipt.exists()
    assert not [m for m in MARKERS if m in printed]


@pytest.mark.parametrize(
    "second_items",
    [[message(refusal="no")], [message('{"value": "42"}')], [message('{"value": false}')]],
    ids=["refusal", "string", "boolean"],
)
def test_second_turn_failure_writes_no_receipt(tmp_path, capsys, monkeypatch, second_items):
    witness = Witness(monkeypatch, [sse([message('{"value": 17}')]), sse(second_items, "resp_prodtest_2")])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert code == 1 and len(witness.bodies) == 2 and not receipt.exists()
    assert not [m for m in MARKERS if m in printed]


@pytest.mark.parametrize(
    "changes",
    [{"saved_at": "2020-01-01T00:00:00+00:00"}, {"scopes": ["resource.invoke"]}],
    ids=["expired", "missing-plan-use"],
)
def test_store_refusals_return_two_before_any_send(tmp_path, capsys, monkeypatch, changes):
    witness = Witness(monkeypatch, [sse([message('{"value": 17}')])])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path, **changes))
    assert code == 2 and witness.bodies == [] and not receipt.exists()
    assert not [m for m in MARKERS if m in printed]


def test_unreadable_store_and_missing_receipt_directory_return_two(tmp_path, capsys, monkeypatch):
    witness = Witness(monkeypatch, [])
    store = make_store(tmp_path)
    os.chmod(store, 0o644)
    code, _, _ = run(tmp_path, capsys, store)
    assert code == 2 and witness.bodies == []
    os.chmod(store, 0o600)
    module = load_consumer()
    code = module.main(["--store", str(store), "--model", MODEL, "--receipt", str(tmp_path / "absent" / "r.json")])
    capsys.readouterr()
    assert code == 2 and witness.bodies == []


@pytest.mark.parametrize("violate_on", [1, 2])
def test_checker_runs_before_each_send(tmp_path, capsys, monkeypatch, violate_on):
    witness = Witness(monkeypatch, [sse([message('{"value": 17}')]), sse([message('{"value": 42}')], "resp_prodtest_2")])
    module = load_consumer()
    seen = []

    def checker(body):
        seen.append((len(witness.bodies), body["input"]))
        return ("prodtest_violation",) if len(seen) == violate_on else ()

    monkeypatch.setattr(module, "siwc_preview_violations", checker)
    code, receipt, _ = run(tmp_path, capsys, make_store(tmp_path), module)
    assert not receipt.exists()
    assert [sent for sent, _ in seen] == [0, 1][:violate_on]
    if violate_on == 1:
        assert code == 2 and witness.bodies == []
    else:
        assert code != 0 and len(witness.bodies) == 1
