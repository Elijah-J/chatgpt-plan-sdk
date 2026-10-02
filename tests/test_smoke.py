"""Offline checks that the smoke consumers refuse before sending."""

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SMOKE = Path(__file__).resolve().parents[1] / "examples"
PLAN = ["resource.invoke", "chatgpt.tokens.use.direct"]


def make_store(tmp_path, **changes):
    raw = {
        "access_token": "tok-smoke-access",
        "refresh_token": "tok-smoke-refresh",
        "client_id": "app_smoke",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scopes": PLAN,
        "saved_at": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)).isoformat(),
    }
    raw.update(changes)
    directory = tmp_path / "s"
    directory.mkdir(mode=0o700)
    path = directory / "cred.json"
    path.write_text(json.dumps(raw))
    os.chmod(path, 0o600)
    return path


def run(script, *argv):
    # Every case must refuse before a send. The dead loopback proxy makes an
    # unexpected send fail locally instead of reaching any real host.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HTTPS_PROXY": "http://127.0.0.1:9",
    }
    return subprocess.run(
        [sys.executable, "-B", str(SMOKE / script), *argv],
        capture_output=True, text=True, env=env, timeout=60, check=False,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"scopes": ["resource.invoke"]},
        {"saved_at": "2020-01-01T00:00:00+00:00"},
    ],
)
def test_ask_refuses_ungranted_or_expired_store(tmp_path, changes):
    store = make_store(tmp_path, **changes)
    receipt = tmp_path / "receipt.json"
    proc = run("ask.py", "--store", str(store), "--model", "gpt-6-astra", "--receipt", str(receipt))
    assert proc.returncode == 2 and not receipt.exists()
    assert "tok-smoke" not in proc.stdout + proc.stderr


def test_ask_refuses_readable_store(tmp_path):
    store = make_store(tmp_path)
    os.chmod(store, 0o644)
    proc = run("ask.py", "--store", str(store), "--model", "m", "--receipt", str(tmp_path / "r.json"))
    assert proc.returncode == 2


def test_refresh_refuses_bootstrap_client(tmp_path):
    store = make_store(tmp_path, client_id="dynamic_agent_client")
    proc = run("refresh.py", "--store", str(store))
    assert proc.returncode == 1 and "invalid_client" in proc.stderr
    assert "tok-smoke" not in proc.stdout + proc.stderr
