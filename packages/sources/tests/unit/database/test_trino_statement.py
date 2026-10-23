# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrinoStatementRunner — the Trino discovery statement primitive.

The behaviours tested hardest are the ones that would otherwise be silent:

* **retry classification**, because this module's whole retry posture rests on
  the two layers covering disjoint failure classes. A regression that retried
  transport errors here would multiply with the client's own policy; one that
  stopped retrying ``errorName`` failures would turn a briefly busy cluster into
  a failed scan.
* **config errors raised before any network call**, so a contradicting secret
  can never be reported as the cluster being down.
* **cell stringification**, which is what makes this interchangeable with
  ``AthenaStatementRunner`` to a caller — including ``None`` staying distinct
  from ``""``.

The fake connection models ``trino.dbapi``'s contract rather than its
internals: ``connect(**kwargs) -> conn``, ``conn.cursor() -> cur``,
``cur.execute(sql, params=None)``, ``cur.fetchall() -> rows``, ``conn.close()``.
``fetchall`` returning everything is the real client's documented behaviour (it
follows the coordinator's ``nextUri`` chain internally), so there is no
pagination to model.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import patch

import pytest
from coa_sources.database.connectors.trino_statement import (
    TrinoConfigError,
    TrinoConnectionFailed,
    TrinoStatementFailed,
    TrinoStatementRunner,
    TrinoStatementTimeout,
    quote_ident,
)
from trino import exceptions as trino_exc

pytestmark = pytest.mark.unit

_MODULE = "coa_sources.database.connectors.trino_statement"

_URI = "https://trino.example.com:8443"
_BASIC_SECRET = {"username": "coa", "password": "s3cret"}
_JWT_SECRET = {"token": "header.eyJzdWIiOiJjb2Etc3ZjIn0.sig"}  # {"sub": "coa-svc"}


def _query_error(error_name: str, message: str = "boom") -> trino_exc.TrinoQueryError:
    """A TrinoQueryError as the client builds it, from the coordinator's JSON."""
    return trino_exc.TrinoQueryError(
        {
            "errorName": error_name,
            "errorCode": 1,
            "errorType": "INTERNAL_ERROR",
            "message": message,
            "failureInfo": {},
        },
        query_id="20261023_000000_00000_abcde",
    )


class FakeCursor:
    """Cursor double.

    ``rows`` is returned by ``fetchall``; ``errors`` is a list of exceptions
    raised by successive ``execute`` calls, ``None`` meaning "succeed". It
    clamps to its last entry so a test need not enumerate every attempt.
    """

    def __init__(self, rows: list[tuple] | None = None, errors: list[Exception | None] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.errors = errors or [None]
        self.executed: list[tuple[str, Any]] = []
        self._call = 0

    def execute(self, statement: str, params: Any = None) -> None:
        self.executed.append((statement, params))
        error = self.errors[min(self._call, len(self.errors) - 1)]
        self._call += 1
        if error is not None:
            raise error

    def fetchall(self) -> list[tuple]:
        return self.rows


class FakeConnection:
    """Connection double recording whether it was closed."""

    def __init__(self, cursor: FakeCursor, *, close_error: Exception | None = None) -> None:
        self._cursor = cursor
        self._close_error = close_error
        self.closed = 0

    def cursor(self) -> FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed += 1
        if self._close_error is not None:
            raise self._close_error


def _connector(
    cursor: FakeCursor,
    *,
    connect_errors: list[Exception | None] | None = None,
    close_error: Exception | None = None,
) -> tuple[Any, list[dict]]:
    """Return a ``connect`` double and the list it records its kwargs into."""
    calls: list[dict] = []
    errors = connect_errors or [None]
    connection = FakeConnection(cursor, close_error=close_error)

    def connect(**kwargs: Any) -> FakeConnection:
        calls.append(kwargs)
        error = errors[min(len(calls) - 1, len(errors) - 1)]
        if error is not None:
            raise error
        return connection

    return connect, calls


def _runner(
    cursor: FakeCursor,
    *,
    secret: dict | None = None,
    auth_mode: str = "BASIC",
    connect_errors: list[Exception | None] | None = None,
    close_error: Exception | None = None,
    **kwargs: Any,
) -> tuple[TrinoStatementRunner, list[dict]]:
    """A runner wired to a fake connection, plus its recorded connect kwargs."""
    connect, calls = _connector(cursor, connect_errors=connect_errors, close_error=close_error)
    runner = TrinoStatementRunner(
        uri=_URI,
        auth_mode=auth_mode,
        secret=secret if secret is not None else _BASIC_SECRET,
        connect=connect,
        **kwargs,
    )
    return runner, calls


# ── URI parsing ──────────────────────────────────────────────────────────────


def test_uri_port_is_parsed() -> None:
    runner, calls = _runner(FakeCursor())
    runner.run("SELECT 1")
    assert (calls[0]["host"], calls[0]["port"]) == ("trino.example.com", 8443)


def test_uri_without_port_defaults_to_443() -> None:
    """443, not Trino's 8080: the scheme is pinned to https."""
    connect, calls = _connector(FakeCursor())
    TrinoStatementRunner(uri="https://trino.example.com", auth_mode="BASIC", secret=_BASIC_SECRET, connect=connect).run(
        "SELECT 1"
    )
    assert calls[0]["port"] == 443


def test_https_scheme_is_always_sent() -> None:
    runner, calls = _runner(FakeCursor())
    runner.run("SELECT 1")
    assert calls[0]["http_scheme"] == "https"


@pytest.mark.parametrize("uri", ["http://trino.example.com", "trino.example.com", "https://"])
def test_unusable_uri_raises_config_error(uri: str) -> None:
    """Raised at construction, before any network call."""
    with pytest.raises(TrinoConfigError):
        TrinoStatementRunner(uri=uri, auth_mode="BASIC", secret=_BASIC_SECRET, connect=lambda **_: None)


def test_non_numeric_port_raises_config_error() -> None:
    """``urlsplit`` defers this to attribute access, so it needs catching."""
    with pytest.raises(TrinoConfigError, match="port"):
        TrinoStatementRunner(
            uri="https://trino.example.com:notaport",
            auth_mode="BASIC",
            secret=_BASIC_SECRET,
            connect=lambda **_: None,
        )


# ── Auth from secret ─────────────────────────────────────────────────────────


def test_basic_auth_uses_username_as_trino_user() -> None:
    runner, calls = _runner(FakeCursor())
    runner.run("SELECT 1")
    assert calls[0]["user"] == "coa"
    assert calls[0]["auth"].__class__.__name__ == "BasicAuthentication"


def test_jwt_auth_takes_user_from_sub_claim() -> None:
    """The username is cosmetic, but Trino rejects a connection without one."""
    runner, calls = _runner(FakeCursor(), auth_mode="JWT", secret=_JWT_SECRET)
    runner.run("SELECT 1")
    assert calls[0]["user"] == "coa-svc"
    assert calls[0]["auth"].__class__.__name__ == "JWTAuthentication"


def test_jwt_without_readable_sub_falls_back_to_constant() -> None:
    """A malformed cosmetic field must not fail the connection."""
    runner, calls = _runner(FakeCursor(), auth_mode="JWT", secret={"token": "not-a-jwt"})
    runner.run("SELECT 1")
    assert calls[0]["user"] == "coa-discovery"


def test_jwt_sub_absent_from_claims_falls_back_to_constant() -> None:
    payload = base64.urlsafe_b64encode(json.dumps({"aud": "trino"}).encode()).decode().rstrip("=")
    runner, calls = _runner(FakeCursor(), auth_mode="JWT", secret={"token": f"h.{payload}.s"})
    runner.run("SELECT 1")
    assert calls[0]["user"] == "coa-discovery"


@pytest.mark.parametrize(
    ("auth_mode", "secret"),
    [
        ("BASIC", {"username": "coa"}),
        ("BASIC", {"password": "p"}),
        ("JWT", {}),
        ("OAUTH2_CLIENT_CREDENTIALS", {"client_id": "a", "client_secret": "b"}),
        ("", _BASIC_SECRET),
    ],
)
def test_unusable_secret_raises_config_error(auth_mode: str, secret: dict) -> None:
    """Including the dropped OAuth2 mode: loud, never a silent fallback to BASIC."""
    with pytest.raises(TrinoConfigError):
        TrinoStatementRunner(uri=_URI, auth_mode=auth_mode, secret=secret, connect=lambda **_: None)


@pytest.mark.parametrize(
    ("auth_mode", "secret"),
    [
        ("BASIC", {"username": "coa", "password": "p", "token": "t"}),
        ("JWT", {"token": "t", "username": "coa"}),
    ],
)
def test_secret_carrying_both_modes_is_refused(auth_mode: str, secret: dict) -> None:
    """A rotated secret that contradicts authMode must name the mismatch."""
    with pytest.raises(TrinoConfigError, match="both"):
        TrinoStatementRunner(uri=_URI, auth_mode=auth_mode, secret=secret, connect=lambda **_: None)


# ── Result shape ─────────────────────────────────────────────────────────────


def test_rows_are_stringified_and_none_preserved() -> None:
    """Matches AthenaStatementRunner's contract; None stays distinct from ""."""
    cursor = FakeCursor(rows=[("orders", 42, None, ""), ("items", True, None, "x")])
    runner, _ = _runner(cursor)
    assert runner.run("SELECT 1") == [
        ["orders", "42", None, ""],
        ["items", "True", None, "x"],
    ]


def test_empty_result_is_an_empty_list_not_an_error() -> None:
    runner, _ = _runner(FakeCursor(rows=[]))
    assert runner.run("SELECT 1") == []


def test_params_are_passed_through_as_bound_parameters() -> None:
    cursor = FakeCursor()
    runner, _ = _runner(cursor)
    runner.run("SELECT ? ", params=("sales",))
    assert cursor.executed == [("SELECT ? ", ("sales",))]


def test_no_params_executes_without_a_params_argument() -> None:
    """Some drivers treat execute(sql, None) differently from execute(sql)."""
    cursor = FakeCursor()
    runner, _ = _runner(cursor)
    runner.run("SELECT 1")
    assert cursor.executed == [("SELECT 1", None)]


def test_catalog_and_schema_are_session_defaults() -> None:
    runner, calls = _runner(FakeCursor())
    runner.run("SELECT 1", catalog="iceberg", schema="sales")
    assert (calls[0]["catalog"], calls[0]["schema"]) == ("iceberg", "sales")


# ── Retry classification: the disjoint-layer contract ────────────────────────


def test_client_keeps_its_own_http_retry_layer() -> None:
    """max_attempts>1 is deliberate — it is what honours 429 Retry-After."""
    runner, calls = _runner(FakeCursor())
    runner.run("SELECT 1")
    assert calls[0]["max_attempts"] > 1


@pytest.mark.parametrize("error_name", ["NO_NODES_AVAILABLE", "QUERY_QUEUE_FULL", "SERVER_STARTING_UP"])
def test_busy_cluster_error_names_are_retried(error_name: str) -> None:
    cursor = FakeCursor(rows=[("ok",)], errors=[_query_error(error_name), None])
    runner, calls = _runner(cursor)
    with patch(f"{_MODULE}.time.sleep"):
        assert runner.run("SELECT 1") == [["ok"]]
    assert len(calls) == 2


def test_retries_are_exhausted_then_the_last_error_raised() -> None:
    cursor = FakeCursor(errors=[_query_error("NO_NODES_AVAILABLE")])
    runner, calls = _runner(cursor)
    with patch(f"{_MODULE}.time.sleep"), pytest.raises(TrinoStatementFailed) as exc:
        runner.run("SELECT 1")
    assert exc.value.error_name == "NO_NODES_AVAILABLE"
    assert len(calls) == 3


@pytest.mark.parametrize("error_name", ["TABLE_NOT_FOUND", "SYNTAX_ERROR", "PERMISSION_DENIED"])
def test_query_errors_about_the_query_are_not_retried(error_name: str) -> None:
    """A wrong query fails identically every time; retrying hides the error."""
    cursor = FakeCursor(errors=[_query_error(error_name)])
    runner, calls = _runner(cursor)
    with pytest.raises(TrinoStatementFailed):
        runner.run("SELECT 1")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "exc",
    [
        trino_exc.Http503Error("error 503: service unavailable"),
        trino_exc.HttpError("error 429: slow down"),
        trino_exc.TrinoConnectionError("connection reset"),
        OSError("no route to host"),
    ],
)
def test_transport_and_http_failures_are_not_retried_here(exc: Exception) -> None:
    """The client already retried these; a second layer would multiply 3 x 3."""
    runner, calls = _runner(FakeCursor(), connect_errors=[exc])
    with pytest.raises(TrinoConnectionFailed):
        runner.run("SELECT 1")
    assert len(calls) == 1


def test_auth_failure_is_not_retried() -> None:
    """Retrying a rejected password risks tripping an account lockout."""
    runner, calls = _runner(FakeCursor(), connect_errors=[trino_exc.TrinoAuthError("bad credential")])
    with pytest.raises(TrinoConnectionFailed) as exc:
        runner.run("SELECT 1")
    assert exc.value.auth_failed is True
    assert len(calls) == 1


def test_query_error_preserves_trino_error_name() -> None:
    """errorName, not the message, distinguishes a grant problem from a bug."""
    cursor = FakeCursor(errors=[_query_error("PERMISSION_DENIED", "Access Denied: Cannot select")])
    runner, _ = _runner(cursor)
    with pytest.raises(TrinoStatementFailed) as exc:
        runner.run("SELECT 1")
    assert exc.value.error_name == "PERMISSION_DENIED"
    assert "Cannot select" in str(exc.value)


# ── Budget ───────────────────────────────────────────────────────────────────


def test_exhausted_budget_raises_timeout_before_connecting() -> None:
    runner, calls = _runner(FakeCursor(), timeout_seconds=0.0)
    with pytest.raises(TrinoStatementTimeout):
        runner.run("SELECT 1")
    assert calls == []


def test_per_call_timeout_overrides_the_instance_default() -> None:
    runner, calls = _runner(FakeCursor(), timeout_seconds=60.0)
    with pytest.raises(TrinoStatementTimeout):
        runner.run("SELECT 1", timeout_seconds=0.0)
    assert calls == []


def test_request_timeout_is_divided_by_the_clients_attempt_count() -> None:
    """So the client's own retries fit inside the budget rather than overrun it."""
    runner, calls = _runner(FakeCursor(), timeout_seconds=30.0, request_timeout_s=15.0)
    runner.run("SELECT 1")
    assert calls[0]["request_timeout"] == pytest.approx(10.0, abs=0.5)


def test_request_timeout_is_floored_at_one_second() -> None:
    """A near-exhausted budget still makes a real request, not a 0 s one."""
    runner, calls = _runner(FakeCursor(), timeout_seconds=0.2)
    runner.run("SELECT 1")
    assert calls[0]["request_timeout"] == 1.0


def test_retry_does_not_sleep_past_the_budget() -> None:
    """Better to report the timeout than to sleep and then fail anyway.

    The jitter is pinned rather than the clock: a backoff longer than the whole
    budget is the condition under test, and ``random.uniform`` is what decides
    the actual delay. Patching ``time.sleep`` alone cannot exercise this —
    nothing then advances the clock, so the deadline is never crossed.
    """
    cursor = FakeCursor(errors=[_query_error("NO_NODES_AVAILABLE")])
    runner, _ = _runner(cursor, timeout_seconds=1.0)
    with (
        patch(f"{_MODULE}.random.uniform", return_value=10.0),
        patch(f"{_MODULE}.time.sleep") as sleep,
        pytest.raises(TrinoStatementTimeout),
    ):
        runner.run("SELECT 1")
    assert sleep.call_count == 0


# ── Connection lifecycle ─────────────────────────────────────────────────────


def _tracking_connector(cursor: FakeCursor) -> tuple[Any, list[FakeConnection]]:
    """A ``connect`` double handing out a fresh connection per call, recorded."""
    made: list[FakeConnection] = []

    def connect(**_: Any) -> FakeConnection:
        connection = FakeConnection(cursor)
        made.append(connection)
        return connection

    return connect, made


def test_connection_is_closed_on_success() -> None:
    connect, made = _tracking_connector(FakeCursor())
    TrinoStatementRunner(uri=_URI, auth_mode="BASIC", secret=_BASIC_SECRET, connect=connect).run("SELECT 1")
    assert [c.closed for c in made] == [1]


def test_connection_is_closed_on_query_failure() -> None:
    connect, made = _tracking_connector(FakeCursor(errors=[_query_error("TABLE_NOT_FOUND")]))
    runner = TrinoStatementRunner(uri=_URI, auth_mode="BASIC", secret=_BASIC_SECRET, connect=connect)
    with pytest.raises(TrinoStatementFailed):
        runner.run("SELECT 1")
    assert [c.closed for c in made] == [1]


def test_every_retry_attempt_closes_its_connection() -> None:
    """One connection per attempt, each closed — none leaked across retries."""
    connect, made = _tracking_connector(FakeCursor(errors=[_query_error("NO_NODES_AVAILABLE")]))
    runner = TrinoStatementRunner(uri=_URI, auth_mode="BASIC", secret=_BASIC_SECRET, connect=connect)
    with patch(f"{_MODULE}.time.sleep"), pytest.raises(TrinoStatementFailed):
        runner.run("SELECT 1")
    assert [c.closed for c in made] == [1, 1, 1]


def test_close_failure_does_not_mask_the_real_error() -> None:
    cursor = FakeCursor(errors=[_query_error("TABLE_NOT_FOUND")])
    runner, _ = _runner(cursor, close_error=RuntimeError("socket already gone"))
    with pytest.raises(TrinoStatementFailed):
        runner.run("SELECT 1")


def test_close_failure_does_not_fail_a_successful_run() -> None:
    runner, _ = _runner(FakeCursor(rows=[("ok",)]), close_error=RuntimeError("socket already gone"))
    assert runner.run("SELECT 1") == [["ok"]]


# ── Identifier quoting ───────────────────────────────────────────────────────


def test_quote_ident_wraps_in_double_quotes() -> None:
    """Valid in every position, because Trino has one parser — unlike Athena."""
    assert quote_ident("sales") == '"sales"'


def test_quote_ident_doubles_interior_quotes() -> None:
    """What stops a crafted identifier closing the quote and appending SQL."""
    assert quote_ident('ev"il') == '"ev""il"'
    assert quote_ident('x" OR "1"="1') == '"x"" OR ""1""=""1"'
