"""Offline checks for the namespaced arithmetic smoke consumer (hermetic: no network, no real store).

Every response is synthetic (``prodtools`` markers). The consumer runs in-process
with the HTTPX async transport replaced by a recording witness and sockets
refused; nothing here reads a credential, a real store or a live endpoint.
"""

import ast
import datetime as dt
import importlib.util
import json
import os
import socket
import stat
from pathlib import Path

import httpx
import pytest

import chatgpt_plan_sdk

CONSUMER = Path(__file__).resolve().parents[2] / "examples" / "tools" / "run.py"
MODEL = "gpt-prod-tools-model"
ACCESS = "tok-prodtools-access"
PLAN = ["resource.invoke", "chatgpt.tokens.use.direct"]
MARKERS = ("tok-prodtools", "call_prodtools", "fc_prodtools", "rs_prodtools", "msg_prodtools", "resp_prodtools", "ENC-PRODTOOLS")
NAMESPACE = {
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
ABSENT = object()


def reasoning(n):
    return {"type": "reasoning", "id": f"rs_prodtools_{n}", "summary": [], "encrypted_content": f"ENC-PRODTOOLS-{n}"}


def fcall(suffix, arguments='{"a":17,"b":25}', *, namespace="arithmetic", name="add", call_id=None, status="completed"):
    item = {
        "type": "function_call",
        "id": f"fc_prodtools_{suffix}",
        "call_id": f"call_prodtools_{suffix}" if call_id is None else call_id,
        "namespace": namespace,
        "name": name,
        "arguments": arguments,
        "status": status,
    }
    if namespace is ABSENT:
        del item["namespace"]
    return item


def message(text=None, *, refusal=None, ident="msg_prodtools_a"):
    if refusal is not None:
        part = {"type": "refusal", "refusal": refusal}
    else:
        part = {"type": "output_text", "text": text, "annotations": []}
    return {"type": "message", "id": ident, "role": "assistant", "status": "completed", "content": [part]}


def output(call_id, value="42"):
    return {"type": "function_call_output", "call_id": call_id, "output": value}


def sse(items, ident="resp_prodtools_1", status="completed"):
    records = [
        {"type": "response.output_item.done", "output_index": index, "item": item}
        for index, item in enumerate(items)
    ]
    response = {"id": ident, "object": "response", "status": status, "model": MODEL, "output": []}
    records.append({"type": "response.completed", "response": response})
    return "".join(f"event: {r['type']}\ndata: {json.dumps(r)}\n\n" for r in records).encode("utf-8")


FINAL = [message("42", ident="msg_prodtools_final")]


def make_store(tmp_path, *, mode=0o600, **changes):
    raw = {
        "access_token": ACCESS,
        "refresh_token": "tok-prodtools-refresh",
        "client_id": "app_prodtools",
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
    os.chmod(path, mode)
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
    spec = importlib.util.spec_from_file_location("prodtools_consumer", CONSUMER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(tmp_path, capsys, store, module=None):
    module = module or load_consumer()
    receipt = tmp_path / "receipt.json"
    code = module.main(["--store", str(store), "--model", MODEL, "--receipt", str(receipt)])
    captured = capsys.readouterr()
    return code, receipt, captured.out + captured.err


def user_text(item):
    assert item["role"] == "user" and set(item) <= {"type", "role", "content"}
    return "".join(part["text"] for part in item["content"] if part["type"] == "input_text")


def assert_stopped(code, receipt, printed, witness, sends):
    assert code != 0, printed
    assert len(witness.bodies) == sends, witness.bodies
    assert not receipt.exists()
    assert not [m for m in MARKERS if m in printed]


def assert_round_trip(tmp_path, capsys, monkeypatch, first_items, call_ids, final_items=FINAL):
    witness = Witness(monkeypatch, [sse(first_items), sse(final_items, "resp_prodtools_2")])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert code == 0, printed
    first, second = witness.bodies
    for body in (first, second):
        assert body["model"] == MODEL and body["store"] is False and body["stream"] is True
        assert body["reasoning"] == {"effort": "low"}
        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["tools"] == [NAMESPACE]
        assert body["parallel_tool_calls"] is False
        assert body["instructions"].strip() and "previous_response_id" not in body
    assert first["tool_choice"] == "required" and second["tool_choice"] == "none"
    assert len(first["input"]) == 1
    text = user_text(first["input"][0])
    assert "17" in text and "25" in text
    history = second["input"]
    replay_end = 1 + len(first_items)
    assert len(history) == replay_end + len(call_ids) + 1
    assert history[0] == first["input"][0]
    assert history[1:replay_end] == first_items
    assert history[replay_end:replay_end + len(call_ids)] == [output(c) for c in call_ids]
    assert user_text(history[-1]).strip()
    written = receipt.read_text(encoding="utf-8")
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert json.loads(written)["call_count"] == len(call_ids)
    assert not [m for m in MARKERS if m in written + printed]
    return witness


def test_single_call_round_trip_replays_items_and_matches_the_call(tmp_path, capsys, monkeypatch):
    assert_round_trip(tmp_path, capsys, monkeypatch, [reasoning(1), fcall("a")], ["call_prodtools_a"])


def test_multi_call_round_trip_matches_every_call_and_accepts_padded_text(tmp_path, capsys, monkeypatch):
    first_items = [reasoning(1), fcall("a", '{"b": 25, "a": 17}'), reasoning(2), fcall("b", ' { "a" : 17 , "b" : 25 } ')]
    final = [reasoning(3), message("\n42 ", ident="msg_prodtools_final")]
    assert_round_trip(tmp_path, capsys, monkeypatch, first_items, ["call_prodtools_a", "call_prodtools_b"], final)


def test_sum_is_computed_from_the_validated_pair_not_a_constant():
    module = load_consumer()
    outputs = module._add_outputs([("call_x", 3, 4), ("call_y", -10, 2)])
    assert outputs == [
        {"type": "function_call_output", "call_id": "call_x", "output": "7"},
        {"type": "function_call_output", "call_id": "call_y", "output": "-8"},
    ]


@pytest.mark.parametrize(
    "first_items",
    [
        [reasoning(1), fcall("a"), message(refusal="prodtools refusal")],
        [reasoning(1), message("42")],
        [],
        [reasoning(1), fcall("a"), {"type": "web_search_call", "id": "ws_prodtools_a", "status": "completed"}],
    ],
    ids=["refusal-beside-valid-call", "no-call-text-only", "no-output", "unexpected-tool-call-type"],
)
def test_refusal_missing_or_unexpected_call_stops_before_second_send(tmp_path, capsys, monkeypatch, first_items):
    witness = Witness(monkeypatch, [sse(first_items), sse(FINAL, "resp_prodtools_2")])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert_stopped(code, receipt, printed, witness, 1)


@pytest.mark.parametrize(
    "calls",
    [
        [fcall("a", name="subtract")],
        [fcall("a", namespace="calculator")],
        [fcall("a", namespace=ABSENT)],
        [fcall("a", status="in_progress")],
        [fcall("a"), fcall("b", call_id="call_prodtools_a")],
        [fcall("a", call_id="  ")],
        [fcall("a", '{"a":17,"b":25')],
        [fcall("a", '{"a":true,"b":25}')],
        [fcall("a", '{"a":17,"b":true}')],
        [fcall("a", '{"a":17.0,"b":25}')],
        [fcall("a", '{"a":"17","b":25}')],
        [fcall("a", '{"a":25,"b":17}')],
        [fcall("a", '{"a":17,"b":25,"c":0}')],
        [fcall("a", '{"a":17}')],
        [fcall("a", '{"a":17,"a":17,"b":25}')],
        [fcall("a", "[17,25]")],
        [fcall("a"), fcall("b", name="subtract")],
        [fcall("a"), fcall("b", '{"a":17,"b":24}')],
    ],
    ids=[
        "unknown-function", "unknown-namespace", "absent-namespace", "not-completed", "duplicate-call-id",
        "blank-call-id", "invalid-json", "boolean-a", "boolean-b", "float-a", "string-a", "swapped-pair",
        "extra-key", "missing-key", "duplicate-key", "non-object", "second-call-unknown-function",
        "second-call-wrong-arguments",
    ],
)
def test_any_invalid_call_stops_with_one_send_and_no_receipt(tmp_path, capsys, monkeypatch, calls):
    witness = Witness(monkeypatch, [sse([reasoning(1), *calls]), sse(FINAL, "resp_prodtools_2")])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert_stopped(code, receipt, printed, witness, 1)


def test_validation_runs_before_any_call_is_executed(tmp_path, capsys, monkeypatch):
    module = load_consumer()
    executed = []
    real_add = module._add_outputs
    monkeypatch.setattr(module, "_add_outputs", lambda calls: executed.append(calls) or real_add(calls))
    witness = Witness(monkeypatch, [sse([reasoning(1), fcall("a"), fcall("b", name="subtract")])])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path), module)
    assert_stopped(code, receipt, printed, witness, 1)
    assert executed == []


@pytest.mark.parametrize(
    "final_items",
    [
        [fcall("z"), message("42", ident="msg_prodtools_final")],
        [message(refusal="prodtools refusal", ident="msg_prodtools_final")],
        [message("41", ident="msg_prodtools_final")],
        [message("The answer is 42", ident="msg_prodtools_final")],
        [reasoning(3)],
    ],
    ids=["new-call", "refusal", "wrong-text", "extra-words", "no-text"],
)
def test_second_turn_must_be_plain_42(tmp_path, capsys, monkeypatch, final_items):
    witness = Witness(monkeypatch, [sse([reasoning(1), fcall("a")]), sse(final_items, "resp_prodtools_2")])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert_stopped(code, receipt, printed, witness, 2)


def test_incomplete_second_turn_writes_no_receipt(tmp_path, capsys, monkeypatch):
    witness = Witness(monkeypatch, [sse([reasoning(1), fcall("a")]), sse(FINAL, "resp_prodtools_2", status="incomplete")])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert_stopped(code, receipt, printed, witness, 2)


def test_http_failure_on_first_send_is_reported_without_receipt(tmp_path, capsys, monkeypatch):
    witness = Witness(monkeypatch, [])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert_stopped(code, receipt, printed, witness, 1)


@pytest.mark.parametrize(
    "store_changes",
    [
        {"saved_at": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).isoformat()},
        {"mode": 0o644},
        {"scopes": ["openid"]},
    ],
    ids=["expired-grant", "group-world-readable-store", "no-plan-use-scope"],
)
def test_store_refusals_send_nothing(tmp_path, capsys, monkeypatch, store_changes):
    witness = Witness(monkeypatch, [sse([reasoning(1), fcall("a")])])
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path, **store_changes))
    assert_stopped(code, receipt, printed, witness, 0)


def test_receipt_directory_missing_or_receipt_present_sends_nothing(tmp_path, capsys, monkeypatch):
    module = load_consumer()
    witness = Witness(monkeypatch, [sse([reasoning(1), fcall("a")])])
    store = make_store(tmp_path)
    missing = tmp_path / "absent" / "receipt.json"
    assert module.main(["--store", str(store), "--model", MODEL, "--receipt", str(missing)]) != 0
    present = tmp_path / "receipt.json"
    present.write_text("keep")
    assert module.main(["--store", str(store), "--model", MODEL, "--receipt", str(present)]) != 0
    printed = capsys.readouterr()
    assert witness.bodies == [] and present.read_text() == "keep"
    assert not [m for m in MARKERS if m in printed.out + printed.err]


def test_preview_checker_runs_before_each_send_and_a_violation_stops(tmp_path, capsys, monkeypatch):
    events = []
    real_checker = chatgpt_plan_sdk.siwc_preview_violations
    witness = Witness(monkeypatch, [sse([reasoning(1), fcall("a")]), sse(FINAL, "resp_prodtools_2")])
    real_handler = httpx.AsyncHTTPTransport.handle_async_request

    async def spy(transport, request):
        events.append("send")
        return await real_handler(transport, request)

    def checker(body):
        events.append("check")
        return ("prodtools_violation",) if len(body["input"]) > 1 else real_checker(body)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", spy)
    monkeypatch.setattr(chatgpt_plan_sdk, "siwc_preview_violations", checker)
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert events == ["check", "send", "check"], events
    assert_stopped(code, receipt, printed, witness, 1)


def test_first_check_violation_sends_nothing(tmp_path, capsys, monkeypatch):
    witness = Witness(monkeypatch, [sse([reasoning(1), fcall("a")])])
    monkeypatch.setattr(chatgpt_plan_sdk, "siwc_preview_violations", lambda body: ("prodtools_violation",))
    code, receipt, printed = run(tmp_path, capsys, make_store(tmp_path))
    assert_stopped(code, receipt, printed, witness, 0)


def test_consumer_source_uses_public_sdk_names_only():
    tree = ast.parse(CONSUMER.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not [a.name for a in node.names if a.name.split(".")[0] in {"chatgpt_plan_sdk", "httpx", "urllib", "socket", "requests"}]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            root = (node.module or "").split(".")[0]
            assert root not in {"httpx", "urllib", "socket", "requests"}
            if root == "chatgpt_plan_sdk":
                assert node.module == "chatgpt_plan_sdk"
                imported |= {alias.name for alias in node.names}
    assert imported and imported <= set(chatgpt_plan_sdk.__all__), sorted(imported - set(chatgpt_plan_sdk.__all__))
