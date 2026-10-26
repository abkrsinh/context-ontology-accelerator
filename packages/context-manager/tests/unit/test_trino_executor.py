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
