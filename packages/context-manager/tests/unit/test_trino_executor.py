# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrinoQueryExecutor and the composite's Trino routing.

The behaviours that carry risk:

* **No transpilation.** The statement must reach the coordinator byte-identical
  apart from the LIMIT cap; a sqlglot round-trip that rewrote identifiers would
  break the identifier-agreement property the whole design rests on.
* **The breaker opens on reachability failures only.** A rejected query must not
  trip it, or one bad statement suppresses Tier 2 for the namespace.
* **Falling back to Athena when unwired.** The deployed Athena-only behaviour
  must never regress.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coa_serve.clients.composite_executor import CompositeQueryExecutor
from coa_serve.clients.trino import (
    TrinoQueryError,
    TrinoQueryExecutor,
    TrinoUnavailableError,
)
from coa_serve.tier2.sql_firewall import UnsafeSQLError

pytestmark = pytest.mark.unit

_MODULE = "coa_serve.clients.trino"
_NS = "11111111-2222-3333-4444-555555555555"
_SRC = "a29e4355-2c9e-447b-9f43-0e490ba0f556"

_CONFIG = {
    "uri": "https://trino.corp.com",
    "catalog": "iceberg",
    "schemaName": "sales",
    "authMode": "BASIC",
    "credentialSecretArn": "arn:aws:secretsmanager:us-east-1:123456789012:secret:t-AbCdEf",
}


def _registry(source: dict | None = None):
    reg = MagicMock()
    reg.get_source = AsyncMock(
        return_value=source
        if source is not None
        else {"queryEngine": "TRINO", "queryable": True, "configuration": json.dumps(_CONFIG)}
    )
    return reg


def _executor(**kwargs) -> TrinoQueryExecutor:
    with patch(f"{_MODULE}.boto3.client") as client:
        client.return_value.get_secret_value.return_value = {
            "SecretString": json.dumps({"username": "coa", "password": "pw"})
        }
        ex = TrinoQueryExecutor(sources_registry=kwargs.pop("registry", _registry()), region="us-east-1", **kwargs)
    ex._secrets.get_secret_value.return_value = {  # noqa: SLF001 - test seam
        "SecretString": json.dumps({"username": "coa", "password": "pw"})
    }
    return ex


# ── LIMIT capping, and the no-transpile property ─────────────────────────────


@pytest.mark.parametrize(
    ("sql", "expect"),
    [
        ("SELECT * FROM sales.orders", "LIMIT 10"),
        ("SELECT * FROM sales.orders LIMIT 5000", "LIMIT 10"),
        ("SELECT * FROM sales.orders LIMIT 3", "LIMIT 3"),
    ],
)
def test_outer_limit_is_capped_not_replaced(sql: str, expect: str) -> None:
    """A tighter existing LIMIT is respected; a looser one is capped."""
    out = TrinoQueryExecutor._inject_limit(sql, 10)  # noqa: SLF001 - the unit under test
    assert expect in out


def test_a_subquery_limit_is_left_alone() -> None:
    sql = "SELECT * FROM (SELECT * FROM sales.orders LIMIT 9999) t"
    out = TrinoQueryExecutor._inject_limit(sql, 10)  # noqa: SLF001
    assert "9999" in out


def test_unparseable_sql_still_gets_a_limit() -> None:
    """Falls back to appending rather than executing uncapped."""
    out = TrinoQueryExecutor._inject_limit("NOT VALID SQL ((", 10)  # noqa: SLF001
    assert out.endswith("LIMIT 10")


def test_identifiers_survive_the_limit_rewrite() -> None:
    """The round-trip must not rewrite names — identifier agreement depends on it."""
    out = TrinoQueryExecutor._inject_limit('SELECT "order_id" FROM "sales"."orders"', 10)  # noqa: SLF001
    assert "order_id" in out
    assert "orders" in out


# ── Validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("timeout", [0, -1, 301, 1.5, "30"])
async def test_invalid_timeout_is_refused(timeout: Any) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        await _executor().execute("SELECT 1", namespace=_NS, data_source_id=_SRC, timeout_seconds=timeout)


async def test_non_select_is_refused_by_the_firewall() -> None:
    with pytest.raises(UnsafeSQLError):
        await _executor().execute("DELETE FROM sales.orders", namespace=_NS, data_source_id=_SRC)


async def test_a_source_without_an_id_is_refused() -> None:
    """There is no sole-source fallback: each source names its own coordinator."""
    with pytest.raises(ValueError, match="data_source_id"):
        await _executor().execute("SELECT 1", namespace=_NS, data_source_id="")


async def test_a_source_without_a_uri_is_refused() -> None:
    reg = _registry({"queryEngine": "TRINO", "queryable": True, "configuration": json.dumps({"catalog": "x"})})
    with pytest.raises(ValueError, match="uri"):
        await _executor(registry=reg).execute("SELECT 1", namespace=_NS, data_source_id=_SRC)


async def test_a_non_allowlisted_host_is_refused_at_query_time() -> None:
    """Re-guarded per execute: an allowlisted host can be re-pointed later."""
    with (
        patch(f"{_MODULE}._ALLOWED_HOSTS", ()),
        patch("coa_common.net.url_guard.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.0.0.5", 0))]),
        pytest.raises(TrinoUnavailableError),
    ):
        await _executor().execute("SELECT 1", namespace=_NS, data_source_id=_SRC)


# ── Execution and result shape ───────────────────────────────────────────────


def _fake_run(rows, columns=("a",), truncated=False):
    return lambda *_a, **_k: (rows, list(columns), truncated)


def _fake_raise(error: Exception):
    def boom(*_a, **_k):
        raise error

    return boom


@contextlib.contextmanager
def _reachable(host: str = "trino.corp.com"):
    """Allowlist ``host`` and stub DNS — the guard resolves for real otherwise."""
    with (
        patch(f"{_MODULE}._ALLOWED_HOSTS", (host,)),
        patch(
            "coa_common.net.url_guard.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("10.0.0.5", 0))],
        ),
    ):
        yield


async def test_a_successful_query_is_tagged_with_the_engine() -> None:
    ex = _executor()
    with (
        _reachable(),
        patch.object(TrinoQueryExecutor, "_run", staticmethod(_fake_run([{"a": 1}]))),
    ):
        result = await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)
    assert result.engine == "trino"
    assert result.rows == [{"a": 1}]
    assert result.row_count == 1
    assert result.truncated is False


async def test_truncation_is_reported() -> None:
    ex = _executor()
    with (
        _reachable(),
        patch.object(TrinoQueryExecutor, "_run", staticmethod(_fake_run([{"a": 1}], truncated=True))),
    ):
        result = await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)
    assert result.truncated is True


# ── Circuit breaker ──────────────────────────────────────────────────────────


async def _fail_n_times(ex: TrinoQueryExecutor, n: int, error: Exception) -> None:
    def boom(*_a, **_k):
        raise error

    with (
        _reachable(),
        patch.object(TrinoQueryExecutor, "_run", staticmethod(boom)),
    ):
        for _ in range(n):
            with pytest.raises((TrinoUnavailableError, TrinoQueryError)):
                await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)


async def test_the_breaker_opens_after_the_threshold_and_fails_fast() -> None:
    """Open means Tier 2 is refused immediately, never cascading into Tier 3."""
    ex = _executor()
    with patch(f"{_MODULE}._BREAKER_THRESHOLD", 3):
        await _fail_n_times(ex, 3, TrinoUnavailableError("down"))
        with pytest.raises(TrinoUnavailableError, match="circuit breaker open"):
            await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)


async def test_a_rejected_query_does_not_trip_the_breaker() -> None:
    """Otherwise one bad statement suppresses Tier 2 for the whole namespace."""
    ex = _executor()
    with patch(f"{_MODULE}._BREAKER_THRESHOLD", 2):
        await _fail_n_times(ex, 4, TrinoQueryError("TABLE_NOT_FOUND"))
    with (
        _reachable(),
        patch.object(TrinoQueryExecutor, "_run", staticmethod(_fake_run([{"a": 1}]))),
    ):
        assert (await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)).row_count == 1


async def test_the_breaker_is_scoped_per_namespace() -> None:
    """One unreachable coordinator must not suppress Tier 2 for everyone."""
    ex = _executor()
    with patch(f"{_MODULE}._BREAKER_THRESHOLD", 2):
        await _fail_n_times(ex, 2, TrinoUnavailableError("down"))
    other = "99999999-2222-3333-4444-555555555555"
    with (
        _reachable(),
        patch.object(TrinoQueryExecutor, "_run", staticmethod(_fake_run([{"a": 1}]))),
    ):
        assert (await ex.execute("SELECT a FROM t", namespace=other, data_source_id=_SRC)).row_count == 1


async def test_a_success_clears_the_failure_count() -> None:
    ex = _executor()
    with patch(f"{_MODULE}._BREAKER_THRESHOLD", 3):
        await _fail_n_times(ex, 2, TrinoUnavailableError("down"))
        with (
            _reachable(),
            patch.object(TrinoQueryExecutor, "_run", staticmethod(_fake_run([{"a": 1}]))),
        ):
            await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)
        await _fail_n_times(ex, 2, TrinoUnavailableError("down"))
        with (
            _reachable(),
            patch.object(TrinoQueryExecutor, "_run", staticmethod(_fake_run([{"a": 2}]))),
        ):
            assert (await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)).row_count == 1


# ── Health and close ─────────────────────────────────────────────────────────


async def test_health_check_never_raises() -> None:
    health = await _executor().health_check()
    assert health["status"] == "ok"
    assert health["backend"] == "trino"


async def test_close_is_a_noop() -> None:
    assert await _executor().close() is None


# ── Composite routing (Step 8) ───────────────────────────────────────────────


def _composite(trino=None, source=None):
    athena = MagicMock()
    athena.execute = AsyncMock(return_value=MagicMock(engine="athena", row_count=0))
    athena.health_check = AsyncMock(return_value={"status": "ok"})
    return (
        CompositeQueryExecutor(
            athena_executor=athena,
            sources_registry=_registry(source),
            trino_executor=trino,
        ),
        athena,
    )


def _trino_double():
    t = MagicMock()
    t.execute = AsyncMock(return_value=MagicMock(engine="trino", row_count=1))
    t.health_check = AsyncMock(return_value={"status": "ok", "backend": "trino"})
    t.close = AsyncMock(return_value=None)
    return t


async def test_a_trino_source_routes_to_the_trino_executor() -> None:
    trino = _trino_double()
    composite, athena = _composite(trino)
    result = await composite.execute("SELECT 1", namespace=_NS, data_source_id=_SRC)
    assert result.engine == "trino"
    trino.execute.assert_awaited_once()
    athena.execute.assert_not_awaited()


async def test_falls_back_to_athena_when_the_executor_is_unwired() -> None:
    """The no-regression guarantee: unwired must behave exactly as before."""
    composite, athena = _composite(trino=None)
    result = await composite.execute("SELECT 1", namespace=_NS, data_source_id=_SRC)
    assert result.engine == "athena"
    athena.execute.assert_awaited_once()


@pytest.mark.parametrize(
    "source",
    [
        {"queryEngine": "TRINO", "queryable": False, "configuration": "{}"},
        {"queryEngine": "ATHENA", "queryable": True, "configuration": "{}"},
        {"queryable": True, "configuration": "{}"},
    ],
)
async def test_both_conditions_are_required_to_route(source: dict) -> None:
    """queryEngine=TRINO AND queryable — either alone stays on Athena."""
    trino = _trino_double()
    composite, athena = _composite(trino, source=source)
    result = await composite.execute("SELECT 1", namespace=_NS, data_source_id=_SRC)
    assert result.engine == "athena"
    trino.execute.assert_not_awaited()


async def test_the_engine_comparison_is_case_insensitive() -> None:
    trino = _trino_double()
    composite, _ = _composite(trino, source={"queryEngine": "trino", "queryable": True, "configuration": "{}"})
    assert (await composite.execute("SELECT 1", namespace=_NS, data_source_id=_SRC)).engine == "trino"


async def test_a_registry_error_falls_back_rather_than_mis_routing() -> None:
    trino = _trino_double()
    composite, athena = _composite(trino)
    composite._sources.get_source = AsyncMock(side_effect=RuntimeError("ddb down"))  # noqa: SLF001
    assert (await composite.execute("SELECT 1", namespace=_NS, data_source_id=_SRC)).engine == "athena"


async def test_health_and_close_include_trino() -> None:
    trino = _trino_double()
    composite, _ = _composite(trino)
    health = await composite.health_check()
    assert health["trino"]["backend"] == "trino"
    await composite.close()
    trino.close.assert_awaited_once()


# ── namespace-scope authorization on the Trino route ─────────────────────


def _scoped_registry(*databases: str, catalog_schemas: set | None = None):
    """A Trino-routed registry whose namespace scope authorizes ``databases``.

    ``catalog_schemas`` mirrors what the real ``sql_namespace_scope`` now returns
    for a TRINO row — the ``(catalog, schema)`` pair that authorizes a 3-part name.
    """
    from coa_serve.clients.sources_registry import SQLNamespaceScope

    reg = _registry(
        {
            "queryEngine": "TRINO",
            "queryable": True,
            "configuration": json.dumps({**_CONFIG, "catalog": "iceberg", "schemaName": "sales"}),
        }
    )
    reg.sql_namespace_scope = AsyncMock(
        return_value=SQLNamespaceScope(
            native_databases=frozenset(databases),
            federated_catalog_schemas=frozenset(catalog_schemas or set()),
        )
    )
    return reg


async def test_the_trino_route_consults_the_namespace_scope() -> None:
    """Mirrors the JDBC/Athena/Redshift routes: Trino was the only one without it."""
    trino = _trino_double()
    composite, _ = _composite(trino)
    composite._sources = _scoped_registry("sales")  # noqa: SLF001
    await composite.execute('SELECT * FROM "sales"."orders"', namespace=_NS, data_source_id=_SRC)
    composite._sources.sql_namespace_scope.assert_awaited()  # noqa: SLF001
    trino.execute.assert_awaited_once()


async def test_an_out_of_scope_reference_is_denied_before_trino_runs() -> None:
    """The statement reaches the coordinator verbatim, so this is the only gate:
    without it a qualified reference reads outside the namespace under the
    source's single service identity."""
    from coa_serve.tier2.sql_firewall import NamespaceSQLScopeError

    trino = _trino_double()
    composite, athena = _composite(trino)
    composite._sources = _scoped_registry("sales")  # noqa: SLF001
    with pytest.raises(NamespaceSQLScopeError):
        await composite.execute(
            'SELECT a.id FROM "sales"."orders" a JOIN "secret_hr"."salaries" b ON a.cid = b.cid',
            namespace=_NS,
            data_source_id=_SRC,
        )
    trino.execute.assert_not_awaited()
    athena.execute.assert_not_awaited()


async def test_a_bare_table_needs_no_scope_lookup() -> None:
    """Unqualified SQL stays on the session catalog/schema, so there is nothing
    that could escape the namespace."""
    trino = _trino_double()
    composite, _ = _composite(trino)
    composite._sources = _scoped_registry("sales")  # noqa: SLF001
    await composite.execute('SELECT * FROM "orders"', namespace=_NS, data_source_id=_SRC)
    composite._sources.sql_namespace_scope.assert_not_awaited()  # noqa: SLF001
    trino.execute.assert_awaited_once()


# ── auth mode comes from authMode, not from secret keys ──────────────────


def _auth(mode: str, secret: dict):
    from coa_serve.clients.trino import _auth_from_secret

    return _auth_from_secret(mode, secret)


def test_basic_uses_the_username_as_the_trino_user() -> None:
    auth, user = _auth("BASIC", {"username": "coa", "password": "pw"})
    assert auth.__class__.__name__ == "BasicAuthentication"
    assert user == "coa"


def test_jwt_builds_bearer_auth() -> None:
    auth, user = _auth("JWT", {"token": "h.e.s"})
    assert auth.__class__.__name__ == "JWTAuthentication"
    assert user == "coa-serve"


@pytest.mark.parametrize(
    ("mode", "secret"),
    [
        ("BASIC", {"username": "coa", "password": "pw", "token": "t"}),
        ("JWT", {"token": "t", "username": "coa"}),
    ],
)
def test_a_contradictory_secret_is_refused(mode: str, secret: dict) -> None:
    """Discovery refuses these; serve must not silently accept them under the
    other mode."""
    with pytest.raises(TrinoQueryError, match="both"):
        _auth(mode, secret)


def test_a_basic_secret_missing_a_key_raises_a_typed_error() -> None:
    """Not a bare KeyError, which would escape both error classes."""
    with pytest.raises(TrinoQueryError, match="BASIC"):
        _auth("BASIC", {"username": "coa"})


@pytest.mark.parametrize("mode", ["", "OAUTH2_CLIENT_CREDENTIALS", "mtls"])
def test_an_unsupported_mode_is_refused_rather_than_guessed(mode: str) -> None:
    with pytest.raises(TrinoQueryError, match="authMode"):
        _auth(mode, {"token": "t"})


def test_the_declared_mode_wins_over_the_keys_present() -> None:
    """A JWT-shaped secret under authMode BASIC is an error, not a silent switch."""
    with pytest.raises(TrinoQueryError):
        _auth("BASIC", {"token": "h.e.s"})


async def test_a_three_part_reference_to_the_sources_own_table_is_allowed() -> None:
    """Needs the TRINO arm in `sql_namespace_scope`; without it the catalog
    authorizes nothing and a correctly-qualified query is denied."""
    trino = _trino_double()
    composite, _ = _composite(trino)
    composite._sources = _scoped_registry(catalog_schemas={("iceberg", "sales")})  # noqa: SLF001
    await composite.execute('SELECT * FROM "iceberg"."sales"."orders"', namespace=_NS, data_source_id=_SRC)
    trino.execute.assert_awaited_once()


async def test_a_three_part_reference_to_another_schema_is_denied() -> None:
    from coa_serve.tier2.sql_firewall import NamespaceSQLScopeError

    trino = _trino_double()
    composite, athena = _composite(trino)
    composite._sources = _scoped_registry(catalog_schemas={("iceberg", "sales")})  # noqa: SLF001
    with pytest.raises(NamespaceSQLScopeError):
        await composite.execute('SELECT * FROM "iceberg"."secret_hr"."salaries"', namespace=_NS, data_source_id=_SRC)
    trino.execute.assert_not_awaited()
    athena.execute.assert_not_awaited()


# ── Credential cache ─────────────────────────────────────────────────────


async def _query(ex: TrinoQueryExecutor, namespace: str = _NS) -> None:
    with (
        _reachable(),
        patch.object(TrinoQueryExecutor, "_run", staticmethod(_fake_run([{"a": 1}]))),
    ):
        await ex.execute("SELECT a FROM t", namespace=namespace, data_source_id=_SRC)


async def test_a_repeat_query_reuses_the_cached_credential() -> None:
    """Uncached, every Tier-2 query spends a GetSecretValue against its rate limit."""
    ex = _executor()
    await _query(ex)
    await _query(ex)
    assert ex._secrets.get_secret_value.call_count == 1  # noqa: SLF001


async def test_an_expired_entry_is_refetched() -> None:
    """The TTL is what bounds how long a rotated credential stays in use."""
    ex = _executor()
    with patch(f"{_MODULE}._SECRET_CACHE_TTL_S", 0.0):
        await _query(ex)
        await _query(ex)
    assert ex._secrets.get_secret_value.call_count == 2  # noqa: SLF001


async def test_each_secret_is_cached_separately() -> None:
    ex = _executor()
    await _query(ex)
    other = dict(_CONFIG, credentialSecretArn="arn:aws:secretsmanager:us-east-1:123456789012:secret:other-ZyXwVu")
    ex._sources.get_source = AsyncMock(  # noqa: SLF001
        return_value={"queryEngine": "TRINO", "queryable": True, "configuration": json.dumps(other)}
    )
    await _query(ex)
    assert ex._secrets.get_secret_value.call_count == 2  # noqa: SLF001
    assert len(ex._secret_cache) == 2  # noqa: SLF001


async def test_the_cache_is_bounded() -> None:
    """A long-lived runtime must not accumulate credentials indefinitely."""
    ex = _executor()
    with patch(f"{_MODULE}._SECRET_CACHE_MAX", 2):
        for n in range(4):
            arn = f"arn:aws:secretsmanager:us-east-1:123456789012:secret:s{n}-AbCdEf"
            ex._sources.get_source = AsyncMock(  # noqa: SLF001
                return_value={
                    "queryEngine": "TRINO",
                    "queryable": True,
                    "configuration": json.dumps(dict(_CONFIG, credentialSecretArn=arn)),
                }
            )
            await _query(ex)
    assert len(ex._secret_cache) == 2  # noqa: SLF001


@pytest.mark.parametrize(
    ("configured", "expect"),
    [
        ("99999", 900.0),  # the docs promise 15 minutes; a bigger value cannot make that false
        ("60", 60.0),
        (None, 900.0),
        ("", 900.0),
        ("not-a-number", 900.0),  # a typo must not take serve down at import
        ("0", 0.0),  # explicit opt-out, logged rather than silent
        ("-5", 0.0),
    ],
)
def test_the_ttl_is_clamped_to_the_documented_ceiling(configured: str | None, expect: float) -> None:
    from coa_serve.clients.trino import _secret_cache_ttl

    assert _secret_cache_ttl(configured) == expect


async def test_an_auth_failure_evicts_the_cached_credential() -> None:
    """Otherwise a rotated credential fails every query until the TTL expires,
    while discovery — which has no cache — already works."""
    ex = _executor()
    await _query(ex)
    assert len(ex._secret_cache) == 1  # noqa: SLF001
    with (
        _reachable(),
        patch.object(
            TrinoQueryExecutor,
            "_run",
            staticmethod(_fake_raise(TrinoQueryError("Error 401: Unauthorized"))),
        ),
        pytest.raises(TrinoQueryError),
    ):
        await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)
    assert len(ex._secret_cache) == 0  # noqa: SLF001


async def test_a_rejected_statement_keeps_the_cached_credential() -> None:
    """A broad match would re-read the secret on every bad statement."""
    ex = _executor()
    await _query(ex)
    with (
        _reachable(),
        patch.object(
            TrinoQueryExecutor,
            "_run",
            staticmethod(_fake_raise(TrinoQueryError("TABLE_NOT_FOUND: no such table"))),
        ),
        pytest.raises(TrinoQueryError),
    ):
        await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)
    assert len(ex._secret_cache) == 1  # noqa: SLF001


@pytest.mark.parametrize(
    ("message", "expect"),
    [
        ("error 401: Invalid credentials", True),
        ("Authentication failed for user coa", True),
        ("Access Denied: Cannot select from table sales.orders", False),
        ("TABLE_NOT_FOUND", False),
        ("SYNTAX_ERROR: line 1:8", False),
    ],
)
def test_auth_failures_are_recognised_narrowly(message: str, expect: bool) -> None:
    from coa_serve.clients.trino import _is_auth_failure

    assert _is_auth_failure(TrinoQueryError(message)) is expect


# ── Total query budget ───────────────────────────────────────────────────


def _connect_double():
    """A stubbed ``trino.dbapi.connect`` returning one row."""
    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.fetchmany.return_value = [(1,)]
    cursor.description = [("a",)]
    return connection


def test_the_coordinator_enforces_the_total_budget() -> None:
    """request_timeout bounds each poll of the nextUri chain, so without this a
    long query outlives its budget and survives the client giving up."""
    connection = _connect_double()
    with patch("trino.dbapi.connect", return_value=connection) as connect:
        TrinoQueryExecutor._run(  # noqa: SLF001
            "SELECT a FROM t LIMIT 10",
            {**_CONFIG, "secret": {"username": "coa", "password": "pw"}},
            10,
            17,
        )
    assert connect.call_args.kwargs["session_properties"] == {"query_max_run_time": "17s"}
    assert connect.call_args.kwargs["request_timeout"] == 17.0


# ── Serve-side metrics ───────────────────────────────────────────────────


async def test_a_successful_query_emits_its_duration() -> None:
    """Discovery volume rides the SourceType-dimensioned scan metrics; serve has
    no equivalent, so without this the Tier-2 path is invisible to CloudWatch."""
    ex = _executor()
    with patch(f"{_MODULE}.emit_metric") as emit:
        await _query(ex)
    names = [c.args[1] for c in emit.call_args_list]
    assert "TrinoQueryDuration" in names


async def test_an_unreachable_coordinator_is_counted_by_class() -> None:
    ex = _executor()
    with (
        patch(f"{_MODULE}.emit_metric") as emit,
        _reachable(),
        patch.object(TrinoQueryExecutor, "_run", staticmethod(_fake_raise(TrinoUnavailableError("down")))),
        pytest.raises(TrinoUnavailableError),
    ):
        await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)
    failures = [c for c in emit.call_args_list if c.args[1] == "TrinoQueryFailures"]
    assert failures and failures[0].kwargs["FailureClass"] == "unavailable"


async def test_a_rejected_query_is_counted_under_its_own_class() -> None:
    """Separating these is what lets a dashboard tell a broken cluster from a
    broken statement — only the first should ever page."""
    ex = _executor()
    with (
        patch(f"{_MODULE}.emit_metric") as emit,
        _reachable(),
        patch.object(TrinoQueryExecutor, "_run", staticmethod(_fake_raise(TrinoQueryError("TABLE_NOT_FOUND")))),
        pytest.raises(TrinoQueryError),
    ):
        await ex.execute("SELECT a FROM t", namespace=_NS, data_source_id=_SRC)
    failures = [c for c in emit.call_args_list if c.args[1] == "TrinoQueryFailures"]
    assert failures and failures[0].kwargs["FailureClass"] == "TrinoQueryError"


async def test_the_breaker_opening_is_counted() -> None:
    ex = _executor()
    with patch(f"{_MODULE}._BREAKER_THRESHOLD", 2), patch(f"{_MODULE}.emit_metric") as emit:
        await _fail_n_times(ex, 2, TrinoUnavailableError("down"))
    assert "TrinoCircuitBreakerOpen" in [c.args[1] for c in emit.call_args_list]


def test_metrics_go_to_the_serve_namespace() -> None:
    from coa_serve.clients.trino import _METRICS_NAMESPACE

    assert _METRICS_NAMESPACE == "COA/Serve"
