"""Offline checks for credential validation, the protected store and refresh."""

import datetime as dt
import fcntl
import json
import os
import urllib.parse

import httpx
import pytest

from chatgpt_plan_sdk import SiwcAuthError, SiwcCredentials, SiwcCredentialStore

PLAN = ["openid", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"]


def record(**changes):
    raw = {
        "access_token": "tok-old-access",
        "refresh_token": "tok-old-refresh",
        "client_id": "app_test_client",
        "token_type": "Bearer",
        "expires_in": 600,
        "scopes": list(PLAN),
        "saved_at": "2026-01-01T00:00:00+00:00",
        "host_tag": "keep-me",
    }
    raw.update(changes)
    return raw


def grant(**changes):
    body = {
        "access_token": "tok-new-access",
        "refresh_token": "tok-new-refresh",
        "token_type": "Bearer",
        "expires_in": 900,
        "scope": " ".join(PLAN),
    }
    body.update(changes)
    return body


@pytest.fixture
def store_path(tmp_path):
    directory = tmp_path / "s"
    directory.mkdir(mode=0o700)
    os.chmod(directory, 0o700)
    path = directory / "cred.json"
    path.write_text(json.dumps(record()))
    os.chmod(path, 0o600)
    return path


def token_client(status=200, body=None):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(status, json=grant() if body is None else body)

    return httpx.Client(transport=httpx.MockTransport(handler)), sent


@pytest.mark.parametrize(
    "changes",
    [
        {"access_token": " "},
        {"token_type": "mac"},
        {"expires_in": True},
        {"expires_in": 0},
        {"scopes": "openid"},
        {"saved_at": "2026-01-01T00:00:00"},
    ],
)
def test_invalid_records_are_refused_without_echoing_values(changes):
    with pytest.raises(ValueError) as caught:
        SiwcCredentials(record(**changes))
    assert "tok-old" not in str(caught.value)


def test_expiry_plan_use_and_rendering():
    creds = SiwcCredentials(record())
    saved = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    assert not creds.is_expired(saved + dt.timedelta(seconds=599))
    assert creds.is_expired(saved + dt.timedelta(seconds=599), margin_seconds=5)
    assert creds.has_plan_use
    assert not SiwcCredentials(record(scopes=["resource.invoke"])).has_plan_use
    rendered = repr(creds) + str(creds) + json.dumps(creds.redacted())
    assert "tok-old" not in rendered and "app_test_client" not in rendered and "keep-me" not in rendered


def test_load_refuses_readable_file(store_path):
    os.chmod(store_path, 0o644)
    with pytest.raises(SiwcAuthError) as caught:
        SiwcCredentialStore(store_path).load()
    assert caught.value.code == "store_permissions"


def test_refresh_posts_the_exact_form_and_replaces_only_the_grant(store_path):
    client, sent = token_client(body=grant(client_id="OTHER", extra_key="x", id_token="hint"))
    result = SiwcCredentialStore(store_path).refresh(http_client=client)
    (request,) = sent
    form = urllib.parse.parse_qs(request.content.decode())
    assert form == {
        "grant_type": ["refresh_token"],
        "client_id": ["app_test_client"],
        "refresh_token": ["tok-old-refresh"],
        "resource": ["https://api.openai.com/v1"],
    }
    stored = json.loads(store_path.read_text())
    assert stored["access_token"] == "tok-new-access" and stored["refresh_token"] == "tok-new-refresh"
    assert stored["client_id"] == "app_test_client" and stored["host_tag"] == "keep-me"
    assert stored["id_token"] == "hint" and "extra_key" not in stored
    assert stored["token_response_extras"] == {"client_id": "OTHER", "extra_key": "x"}
    assert result.raw == stored and (os.stat(store_path).st_mode & 0o777) == 0o600
    assert not client.is_closed


@pytest.mark.parametrize(
    "status,body,code",
    [
        (400, {"error": "invalid_grant"}, "invalid_grant"),
        (400, {"error": "refresh_token_reused"}, "refresh_token_reused"),
        (401, {"error": "invalid_client"}, "invalid_client"),
        (400, {"error": "something_new"}, "other"),
        (200, {"access_token": "only"}, "refresh_uncertain"),
    ],
)
def test_refusals_keep_the_file(store_path, status, body, code):
    before = store_path.read_bytes()
    client, sent = token_client(status, body)
    with pytest.raises(SiwcAuthError) as caught:
        SiwcCredentialStore(store_path).refresh(http_client=client)
    assert caught.value.code == code and len(sent) == 1
    assert store_path.read_bytes() == before


def test_bootstrap_client_refuses_before_any_send(store_path):
    store_path.write_text(json.dumps(record(client_id="dynamic_agent_client")))
    client, sent = token_client()
    with pytest.raises(SiwcAuthError) as caught:
        SiwcCredentialStore(store_path).refresh(http_client=client)
    assert caught.value.code == "invalid_client" and sent == []


def test_held_lock_refuses_before_any_send(store_path):
    client, sent = token_client()
    fd = os.open(str(store_path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with pytest.raises(SiwcAuthError) as caught:
            SiwcCredentialStore(store_path).refresh(http_client=client)
    finally:
        os.close(fd)
    assert caught.value.code == "store_locked" and sent == []


def test_failed_persistence_retains_the_replacement(store_path, monkeypatch):
    def fail(fd):
        raise OSError("synthetic")

    monkeypatch.setattr(os, "fsync", fail)
    client, _ = token_client()
    with pytest.raises(SiwcAuthError) as caught:
        SiwcCredentialStore(store_path).refresh(http_client=client)
    assert caught.value.code == "persistence_uncertain"
    assert caught.value.replacement.access_token == "tok-new-access"
    assert "tok-new" not in str(caught.value) + repr(caught.value)


def test_load_refuses_a_group_or_world_accessible_parent(store_path):
    before = store_path.read_bytes()
    for mode in (0o755, 0o750, 0o705):
        os.chmod(store_path.parent, mode)
        try:
            with pytest.raises(SiwcAuthError) as caught:
                SiwcCredentialStore(store_path).load()
        finally:
            os.chmod(store_path.parent, 0o700)
        assert caught.value.code == "store_permissions"
    assert store_path.read_bytes() == before


def test_a_large_saved_record_reloads_unchanged(store_path):
    raw = record(opaque_native_metadata="m" * (2 << 20))
    store = SiwcCredentialStore(store_path)
    store.save(SiwcCredentials(raw))
    assert store.load().raw == raw


@pytest.mark.parametrize("hint", [None, "", "  ", 5, False, {"n": 1}, ["x"]])
def test_unusable_id_token_keeps_the_rotation_and_the_previous_hint(store_path, hint):
    store_path.write_text(json.dumps(record(id_token="old-hint")))
    client, sent = token_client(body=grant(id_token=hint))
    result = SiwcCredentialStore(store_path).refresh(http_client=client)
    stored = json.loads(store_path.read_text())
    assert len(sent) == 1
    assert stored["access_token"] == "tok-new-access" and stored["refresh_token"] == "tok-new-refresh"
    assert stored["id_token"] == "old-hint" and result.raw == stored


def test_missing_required_field_still_refuses_with_an_unusable_id_token(store_path):
    before = store_path.read_bytes()
    client, sent = token_client(body=grant(id_token=None, refresh_token=""))
    with pytest.raises(SiwcAuthError) as caught:
        SiwcCredentialStore(store_path).refresh(http_client=client)
    assert caught.value.code == "refresh_uncertain" and len(sent) == 1
    assert store_path.read_bytes() == before


@pytest.mark.parametrize("status,code", [(200, "refresh_uncertain"), (400, "other")])
def test_token_body_nested_beyond_the_parser_limit_is_classified(store_path, status, code):
    before = store_path.read_bytes()
    depth = 200_000
    nested = b"[" * depth + b"]" * depth
    sent = []

    def handler(request):
        sent.append(request)
        body = nested if status == 200 else b'{"error": ' + nested + b"}"
        return httpx.Response(status, content=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SiwcAuthError) as caught:
        SiwcCredentialStore(store_path).refresh(http_client=client)
    assert caught.value.code == code and len(sent) == 1
    assert caught.value.status_code == (None if status == 200 else status)
    assert store_path.read_bytes() == before


@pytest.mark.parametrize("expires_in", [10**12, 10**400])
def test_unrepresentable_expiry_is_exact_and_summarized_as_none(expires_in):
    creds = SiwcCredentials(record(expires_in=expires_in))
    saved = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    later = saved + dt.timedelta(days=1)
    assert creds.expires_in == expires_in
    assert creds.is_expired(later) is False
    assert creds.is_expired(dt.datetime.max.replace(tzinfo=dt.timezone.utc)) is False
    assert creds.redacted()["expires_at"] is None


def test_unrepresentable_expiry_honours_the_margin():
    creds = SiwcCredentials(record(expires_in=10**12))
    later = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
    assert creds.is_expired(later, margin_seconds=10**12) is True
    assert creds.is_expired(later, margin_seconds=10**12 - 2 * 86400) is False


def _deep_object_json(depth):
    return '{"d":' * depth + "0" + "}" * depth


def _rotating_store(store_path, old_extras):
    store_path.write_text(json.dumps(record(
        token_response_extras=old_extras, earliest_refresh_at="prev-opaque-time",
        native_note={"keep": "native-value"},
    )))
    return SiwcCredentialStore(store_path)


@pytest.mark.parametrize(
    "scope,plan_use",
    [(" ".join(PLAN), True), ("openid offline_access", False)],
    ids=["plan-use-present", "plan-use-lost"],
)
def test_rotation_survives_unrepresentable_new_extras(store_path, scope, plan_use):
    store = _rotating_store(store_path, {"prev_extra": 1})
    prefix = json.dumps(grant(scope=scope, id_token="new-hint"))[:-1]
    body = prefix + ',"beyond":' + _deep_object_json(700) + "}"
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, content=body.encode())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = store.refresh(http_client=client)
    stored = json.loads(store_path.read_text())
    assert len(sent) == 1
    assert stored["access_token"] == "tok-new-access" and stored["expires_in"] == 900
    assert stored["refresh_token"] == "tok-new-refresh" and stored["id_token"] == "new-hint"
    assert stored["scopes"] == scope.split()
    assert stored["token_response_extras"] == {"prev_extra": 1}
    assert stored["earliest_refresh_at"] == "prev-opaque-time"
    assert stored["native_note"] == {"keep": "native-value"} and stored["host_tag"] == "keep-me"
    assert "beyond" not in stored
    assert result.raw == stored and result.has_plan_use is plan_use
    assert store.load().has_plan_use is plan_use


def test_rotation_survives_unrepresentable_earliest_refresh_at(store_path):
    store = _rotating_store(store_path, {"prev_extra": 1})
    body = json.dumps(grant(extra_one="kept-new"))[:-1]
    body += ',"earliest_refresh_at":' + _deep_object_json(700) + "}"
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=body.encode())))
    store.refresh(http_client=client)
    stored = json.loads(store_path.read_text())
    assert stored["access_token"] == "tok-new-access"
    assert stored["earliest_refresh_at"] == "prev-opaque-time"
    assert stored["token_response_extras"] == {"extra_one": "kept-new"}


def test_representable_new_extras_are_kept_beside_an_unrepresentable_one(store_path):
    store = _rotating_store(store_path, {"prev_extra": 1})
    body = json.dumps(grant(alpha=[1, {"b": 2}], omega="last"))[:-1]
    body += ',"beyond":' + _deep_object_json(700) + ',"earliest_refresh_at":"new-opaque"}'
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=body.encode())))
    store.refresh(http_client=client)
    stored = json.loads(store_path.read_text())
    assert stored["token_response_extras"] == {"alpha": [1, {"b": 2}], "omega": "last"}
    assert stored["earliest_refresh_at"] == "new-opaque"
    assert "beyond" not in stored


def test_shallow_new_metadata_still_replaces_the_previous(store_path):
    store = _rotating_store(store_path, {"prev_extra": 1})
    client, _ = token_client(body=grant(fresh={"n": [1]}, earliest_refresh_at="new-opaque"))
    store.refresh(http_client=client)
    stored = json.loads(store_path.read_text())
    assert stored["token_response_extras"] == {"fresh": {"n": [1]}}
    assert stored["earliest_refresh_at"] == "new-opaque"


def test_missing_required_field_still_refuses_beside_unrepresentable_metadata(store_path):
    before = store_path.read_bytes()
    body = json.dumps(grant(refresh_token=""))[:-1] + ',"beyond":' + _deep_object_json(700) + "}"
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=body.encode())))
    with pytest.raises(SiwcAuthError) as caught:
        SiwcCredentialStore(store_path).refresh(http_client=client)
    assert caught.value.code == "refresh_uncertain"
    assert store_path.read_bytes() == before
