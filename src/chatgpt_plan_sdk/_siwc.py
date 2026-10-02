"""Sign in with ChatGPT credentials: record, protected store, refresh, checker."""

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import fcntl
import json
import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx

__all__ = [
    "SiwcAuthError",
    "SiwcCredentialStore",
    "SiwcCredentials",
    "siwc_preview_violations",
]

_TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
_RESOURCE = "https://api.openai.com/v1"
_BOOTSTRAP_CLIENT_ID = "dynamic_agent_client"
_PLAN_SCOPES = frozenset({"resource.invoke", "chatgpt.tokens.use.direct"})
_UNUSABLE_CODES = frozenset({
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
})
_REFRESH_TIMEOUT_SECONDS = 60.0
# Response keys the refresh replaces or interprets; every other key is kept
# apart in ``token_response_extras`` and never reaches the top level.
_TOKEN_KEYS = frozenset({
    "access_token", "refresh_token", "id_token", "token_type", "expires_in",
    "scope", "earliest_refresh_at",
})


class SiwcAuthError(Exception):
    """A credential operation was refused or its outcome is uncertain.

    ``code`` is a safe SDK label: one of the classification codes
    (``invalid_grant``, ``invalid_refresh_token``, ``token_expired``,
    ``refresh_token_expired``, ``refresh_token_invalidated``,
    ``refresh_token_reused``, ``invalid_client``, ``other``,
    ``refresh_uncertain``, ``persistence_uncertain``) or a local label
    (``connection_failed``, ``store_locked``, ``store_permissions``,
    ``store_unreadable``, ``store_invalid``). ``status_code`` is the HTTP
    status when a response supplied one. ``replacement`` is the sensitive,
    valid replacement credentials retained when persisting them failed.
    Provider text never appears in the rendered message.
    """

    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        replacement: SiwcCredentials | None = None,
    ) -> None:
        detail = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"Sign in with ChatGPT credential operation failed: {code}{detail}")
        self.code = code
        self.status_code = status_code
        self.replacement = replacement

    def __repr__(self) -> str:
        return f"SiwcAuthError(code={self.code!r}, status_code={self.status_code!r})"


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


class SiwcCredentials:
    """One validated SIWC grant record.

    ``raw`` is a deep copy of the sensitive JSON record, other fields
    included. Validation errors name the field, never its value; ``repr`` and
    ``str`` show no record content.
    """

    __slots__ = ("_raw", "_saved_at", "_scopes")

    def __init__(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping):
            raise ValueError("credentials must be a JSON object")
        record = copy.deepcopy(dict(raw))
        for field in ("access_token", "refresh_token", "client_id"):
            if not _nonblank(record.get(field)):
                raise ValueError(f"{field} must be a nonblank string")
        if record.get("token_type") != "Bearer":
            raise ValueError("token_type must be Bearer")
        if not _positive_int(record.get("expires_in")):
            raise ValueError("expires_in must be a positive integer")
        scopes = record.get("scopes")
        if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
            raise ValueError("scopes must be an array of strings")
        saved_at = record.get("saved_at")
        if not isinstance(saved_at, str):
            raise ValueError("saved_at must be an ISO 8601 string")
        try:
            parsed = dt.datetime.fromisoformat(saved_at)
        except ValueError:
            raise ValueError("saved_at must be an ISO 8601 string") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("saved_at must be timezone-aware")
        self._raw = record
        self._saved_at = parsed
        self._scopes = tuple(scopes)

    @property
    def raw(self) -> dict[str, Any]:
        """A deep copy of the sensitive record."""
        return copy.deepcopy(self._raw)

    @property
    def access_token(self) -> str:
        return self._raw["access_token"]

    @property
    def refresh_token(self) -> str:
        return self._raw["refresh_token"]

    @property
    def client_id(self) -> str:
        return self._raw["client_id"]

    @property
    def scopes(self) -> tuple[str, ...]:
        return self._scopes

    @property
    def expires_in(self) -> int:
        return self._raw["expires_in"]

    @property
    def saved_at(self) -> dt.datetime:
        return self._saved_at

    @property
    def has_plan_use(self) -> bool:
        """Whether both ``resource.invoke`` and ``chatgpt.tokens.use.direct`` are granted."""
        return _PLAN_SCOPES <= set(self._scopes)

    def _expires_at(self) -> dt.datetime | None:
        """saved_at + expires_in, or None when it is beyond ``datetime`` range."""
        try:
            return self._saved_at + dt.timedelta(seconds=self.expires_in)
        except OverflowError:
            return None

    def is_expired(self, now: dt.datetime, margin_seconds: float = 0) -> bool:
        """Whether ``now`` is within ``margin_seconds`` of saved_at + expires_in.

        Exact even when the expiry cannot be represented as a ``datetime``:
        ``now - saved_at >= expires_in - margin_seconds``, in exact seconds.
        """
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        try:
            expires_at = self._saved_at + dt.timedelta(seconds=self.expires_in)
            return now >= expires_at - dt.timedelta(seconds=margin_seconds)
        except OverflowError:
            pass
        elapsed = now - self._saved_at
        elapsed_seconds = Fraction(
            (elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds,
            1_000_000,
        )
        return elapsed_seconds >= self.expires_in - Fraction(margin_seconds)

    def redacted(self) -> dict[str, Any]:
        """Expiry, scopes and plan-use summary; no tokens, identity or extras.

        ``expires_at`` is None when the expiry is beyond ``datetime`` range.
        """
        expires_at = self._expires_at()
        return {
            "expires_at": None if expires_at is None else expires_at.isoformat(),
            "scopes": list(self._scopes),
            "has_plan_use": self.has_plan_use,
        }

    def __repr__(self) -> str:
        return "SiwcCredentials(<redacted>)"

    __str__ = __repr__


def _connection_error_types() -> tuple[type[Exception], ...]:
    return (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class SiwcCredentialStore:
    """One protected SIWC credential file at an explicit path.

    No discovery, login or automatic refresh. ``refresh`` and ``save`` hold a
    nonblocking exclusive flock on ``<path>.lock`` and write atomically.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock_path = Path(str(self._path) + ".lock")

    # --------------------------------------------------------------- reading

    def load(self) -> SiwcCredentials:
        """Read and validate the store; refuse a group- or world-accessible file."""
        return self._load()

    def _load(self) -> SiwcCredentials:
        code: str | None = None
        data = b""
        fd = None
        # Owner-only directories are required (SPEC D4): refuse a group- or
        # world-accessible containing directory before reading the file.
        try:
            if os.stat(self._path.parent).st_mode & 0o077:
                code = "store_permissions"
            else:
                fd = os.open(self._path, os.O_RDONLY)
        except OSError:
            code = "store_unreadable"
        if fd is not None:
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    code = "store_unreadable"
                elif stat.S_IMODE(info.st_mode) & 0o077:
                    code = "store_permissions"
                else:
                    blocks: list[bytes] = []
                    while True:
                        block = os.read(fd, 1 << 16)
                        if not block:
                            break
                        blocks.append(block)
                    data = b"".join(blocks)
            except OSError:
                code = "store_unreadable"
            finally:
                os.close(fd)
        if code is not None:
            raise SiwcAuthError(code)
        credentials: SiwcCredentials | None = None
        try:
            credentials = SiwcCredentials(json.loads(data.decode("utf-8")))
        except (ValueError, RecursionError):
            credentials = None
        if credentials is None:
            raise SiwcAuthError("store_invalid")
        return credentials

    # --------------------------------------------------------------- locking

    def _open_lock(self, *, create_parent: bool) -> tuple[int | None, str | None]:
        parent = self._path.parent
        try:
            if create_parent and not parent.exists():
                parent.mkdir(mode=0o700)
                os.chmod(parent, 0o700)
            if os.stat(parent).st_mode & 0o077:
                return None, "store_permissions"
            fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            return None, "store_unreadable"
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None, "store_locked"
        return fd, None

    @contextlib.contextmanager
    def _locked(self, *, create_parent: bool = False) -> Iterator[None]:
        fd, code = self._open_lock(create_parent=create_parent)
        if fd is None:
            raise SiwcAuthError(code or "store_locked")
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # --------------------------------------------------------------- writing

    def save(self, credentials: SiwcCredentials) -> None:
        """Atomically replace the store with ``credentials``.

        Takes the lock, writes an owner-only temporary file, fsyncs it,
        replaces the target, then fsyncs the containing directory through a
        read-only descriptor, all under the lock. Failing to persist raises
        ``SiwcAuthError`` with code ``persistence_uncertain`` and the
        replacement retained.
        """
        if not isinstance(credentials, SiwcCredentials):
            raise TypeError("credentials must be SiwcCredentials")
        with self._locked(create_parent=True):
            self._write(credentials)

    def _write(self, credentials: SiwcCredentials) -> None:
        # Caller holds the lock; this writer never takes it again.
        parent = self._path.parent
        temp = parent / f".{self._path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        published = False
        failed = False
        try:
            data = (json.dumps(credentials.raw, indent=2) + "\n").encode("utf-8")
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.fchmod(fd, 0o600)
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temp, self._path)
            published = True
            directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except Exception:
            failed = True
        if failed:
            if not published:
                with contextlib.suppress(OSError):
                    os.unlink(temp)
            raise SiwcAuthError("persistence_uncertain", replacement=credentials)

    # --------------------------------------------------------------- refresh

    def refresh(self, *, http_client: httpx.Client | None = None) -> SiwcCredentials:
        """Exchange the stored refresh token once and persist the replacement.

        One form POST, no scope, no retry, no redirects, no inherited auth.
        An injected client stays open; an internal one is closed. The lock is
        taken once around load, exchange and replacement.
        """
        with self._locked():
            current = self._load()
            if current.client_id == _BOOTSTRAP_CLIENT_ID:
                raise SiwcAuthError("invalid_client")
            replacement = self._exchange(current, http_client)
            self._write(replacement)
            return replacement

    def _exchange(
        self, current: SiwcCredentials, http_client: httpx.Client | None
    ) -> SiwcCredentials:
        form = {
            "grant_type": "refresh_token",
            "client_id": current.client_id,
            "refresh_token": current.refresh_token,
            "resource": _RESOURCE,
        }
        owned = http_client is None
        client = httpx.Client(timeout=_REFRESH_TIMEOUT_SECONDS) if owned else http_client
        response: httpx.Response | None = None
        code = "refresh_uncertain"
        try:
            try:
                response = client.post(_TOKEN_URL, data=form, auth=None, follow_redirects=False)
            except Exception as exc:
                # Failing before a connection exists cannot have used the token.
                if isinstance(exc, _connection_error_types()):
                    code = "connection_failed"
        finally:
            if owned:
                client.close()
        if response is None:
            raise SiwcAuthError(code)
        if response.status_code != 200:
            raise SiwcAuthError(_classify(response), status_code=response.status_code)
        replacement = _replacement(current, response)
        if replacement is None:
            raise SiwcAuthError("refresh_uncertain")
        return replacement


def _classify(response: httpx.Response) -> str:
    # RFC 6749 section 5.2: a string "error" member; nested objects are read too.
    label = "other"
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return label
    value = body.get("error") if isinstance(body, dict) else None
    if isinstance(value, dict):
        value = value.get("code")
    if value == "invalid_client":
        label = "invalid_client"
    elif isinstance(value, str) and value in _UNUSABLE_CODES:
        label = value
    return label


def _replacement(current: SiwcCredentials, response: httpx.Response) -> SiwcCredentials | None:
    """Validate a 200 token response and merge it into the stored record."""
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return None
    if not isinstance(body, dict):
        return None
    scope = body.get("scope")
    if (
        not _nonblank(body.get("access_token"))
        or not _nonblank(body.get("refresh_token"))
        or body.get("token_type") != "Bearer"
        or not _positive_int(body.get("expires_in"))
        or not isinstance(scope, str)
    ):
        return None
    record = current.raw
    record.update({
        "access_token": body["access_token"],
        "refresh_token": body["refresh_token"],
        "token_type": "Bearer",
        "expires_in": body["expires_in"],
        "scopes": scope.split(),
        "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    # The ID token is an optional hint: only a nonblank string replaces it;
    # anything else leaves the previous hint and is not retained.
    if _nonblank(body.get("id_token")):
        record["id_token"] = body["id_token"]
    extras = {key: value for key, value in body.items() if key not in _TOKEN_KEYS}
    full = dict(record)
    if "earliest_refresh_at" in body:
        full["earliest_refresh_at"] = body["earliest_refresh_at"]
    if extras:
        full["token_response_extras"] = extras
    try:
        return SiwcCredentials(full)
    except ValueError:
        return None
    except RecursionError:
        pass
    # The parser accepts deeper nesting than ``copy.deepcopy`` can represent.
    # Optional new metadata the record cannot hold is dropped, one item at a
    # time; the validated rotation and the previous metadata stay.
    try:
        kept = SiwcCredentials(record)
    except (ValueError, RecursionError):
        return None
    if "earliest_refresh_at" in body:
        trial = {**record, "earliest_refresh_at": body["earliest_refresh_at"]}
        try:
            kept = SiwcCredentials(trial)
            record = trial
        except (ValueError, RecursionError):
            pass
    held: dict[str, Any] = {}
    for key, value in extras.items():
        try:
            kept = SiwcCredentials({**record, "token_response_extras": {**held, key: value}})
        except (ValueError, RecursionError):
            continue
        held[key] = value
    return kept


# ------------------------------------------------------------ preview checker

_UNSUPPORTED_FIELDS = (
    "background", "conversation", "max_output_tokens", "max_tool_calls", "metadata",
    "moderation", "multi_agent", "prompt", "prompt_cache_retention", "safety_identifier",
    "temperature", "top_logprobs", "top_p", "truncation", "user",
)
_HOSTED_TOOL_TYPES = (
    "image_generation", "file_search", "code_interpreter", "computer",
    "computer_use_preview", "mcp", "tool_search", "programmatic_tool_calling",
)
_ASTRA_MODEL = "gpt-6-astra"


def siwc_preview_violations(body: Mapping[str, Any]) -> tuple[str, ...]:
    """Report each captured SIWC preview restriction the body breaks, once.

    Pure and bounded to the captured rules: not an exhaustive validator of
    model or account policy. The body is not modified.
    """
    if not isinstance(body, Mapping):
        raise TypeError("body must be a mapping")
    found: list[str] = []
    if body.get("store") is not False:
        found.append("store must be exactly false")
    items = body.get("input")
    if not isinstance(items, (list, tuple)):
        found.append("input must be an array")
    elif any(isinstance(item, Mapping) and item.get("role") == "system" for item in items):
        found.append("system-role input items are not supported; use instructions")
    if "previous_response_id" in body:
        found.append("previous_response_id is not supported")
    for name in _UNSUPPORTED_FIELDS:
        if name in body:
            found.append(f"field {name} is not supported")
    tools = body.get("tools")
    if isinstance(tools, (list, tuple)):
        kinds = [tool.get("type") for tool in tools if isinstance(tool, Mapping)]
        if "function" in kinds or "custom" in kinds:
            found.append("function and custom tools must be inside a namespace")
        for kind in _HOSTED_TOOL_TYPES:
            if kind in kinds:
                found.append(f"hosted tool type {kind} is not supported")
    reasoning = body.get("reasoning")
    if (
        body.get("model") == _ASTRA_MODEL
        and isinstance(reasoning, Mapping)
        and reasoning.get("effort") == "none"
    ):
        found.append("gpt-6-astra does not support reasoning effort none")
    return tuple(found)
