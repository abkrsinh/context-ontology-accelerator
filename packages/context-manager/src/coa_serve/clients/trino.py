# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trino query executor — serve-time counterpart to the TRINO source type.

Executes SELECT SQL on the same coordinator discovery read, which is what makes
the identifiers agree by construction: the names recorded at scan time are the
names queried here, with no translation layer between them.

**No transpilation.** The VKG emits Trino-dialect SQL and the target IS Trino, so
the statement reaches the coordinator byte-identical. Compare ``redshift_data``,
which must transpile Trino→Redshift via sqlglot and rewrite table references to
``awsdatacatalog`` 3-part names.

Security:
- SELECT-only, via the shared ``SQLFirewall.validate``.
- A root-level LIMIT is injected so a large scan does not fully materialise.
- The coordinator URI is re-validated through the shared SSRF guard on every
  execute: an allowlisted hostname can be re-pointed after the source exists.
- Trino's own fine-grained access control does NOT apply — one service identity
  serves every caller.

A failure here must not cost a Tier-3 synthesis, so a per-namespace circuit
breaker fails fast while open rather than letting the cascade fall through.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

import boto3
import sqlglot
import sqlglot.errors
import structlog
from coa_common import resolve_region
from coa_common.metrics import emit_metric
from coa_common.net import UrlGuardError, validate_url

from ..query_utils import validate_namespace
from ..tier2.sql_firewall import SQLFirewall
from .base import QueryResult, instrumented
from .sources_registry import SourcesRegistry

logger = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_ROWS = int(os.environ.get("TRINO_MAX_ROWS", "1000"))

# Hosts whose non-public addresses a coordinator URI may resolve to. Empty blocks
# every Trino query — the same fail-closed list the control plane validates against.
_ALLOWED_HOSTS = tuple(h.strip() for h in os.environ.get("TRINO_ALLOWED_HOSTS", "").split(",") if h.strip())

# Breaker, per namespace. Without it a Trino timeout makes Tier 2 report low
# confidence, the cascade falls to Tier 3, and a cheap failure becomes 20-90s of
# synthesis.
_BREAKER_THRESHOLD = int(os.environ.get("TRINO_BREAKER_FAILURE_THRESHOLD", "5"))
_BREAKER_WINDOW_S = float(os.environ.get("TRINO_BREAKER_WINDOW_S", "60"))
_BREAKER_OPEN_S = float(os.environ.get("TRINO_BREAKER_OPEN_S", "30"))

# Uncached, every query costs a GetSecretValue on the Tier-2 hot path. The TTL is
# also the window a rotated credential stays in use, so it is capped.
_SECRET_CACHE_TTL_CEILING_S = 900.0
_SECRET_CACHE_MAX = 128


def _secret_cache_ttl(raw: str | None) -> float:
    """Clamp the configured TTL to the ceiling, falling back on an unreadable value.

    A malformed value must not take serve down at import, and must not be read as
    "no caching" either — that would restore the per-query GetSecretValue.

    Zero or less is honoured as an explicit opt-out, but logged: it reinstates a
    GetSecretValue per query, so it must not be reachable by typo in silence.
    """
    try:
        requested = float(raw) if raw else _SECRET_CACHE_TTL_CEILING_S
    except ValueError:
        logger.warning("trino_secret_cache_ttl_unreadable", value=raw)
        requested = _SECRET_CACHE_TTL_CEILING_S
    if requested <= 0:
        logger.warning("trino_secret_cache_disabled", value=raw)
        return 0.0
    return min(requested, _SECRET_CACHE_TTL_CEILING_S)


_SECRET_CACHE_TTL_S = _secret_cache_ttl(os.environ.get("TRINO_SECRET_CACHE_TTL_S"))

# Stdout EMF, extracted by the runtime's log pipeline, so the serve role needs no
# PutMetricData grant. Discovery-side Trino volume is already covered by the
# SourceType-dimensioned metrics the scan pipeline emits.
_METRICS_NAMESPACE = "COA/Serve"

_firewall = SQLFirewall()


class TrinoQueryError(RuntimeError):
    """The coordinator rejected or failed the query."""


class TrinoUnavailableError(RuntimeError):
    """The coordinator was unreachable, or the breaker is open for this namespace."""


@dataclass
class _BreakerState:
    """Consecutive failures inside the window, and when the breaker opened."""

    failures: list[float] = field(default_factory=list)
    opened_at: float = 0.0


def _auth_from_secret(auth_mode: str, secret: dict[str, Any]) -> tuple[Any, str]:
    """Build the client auth from the stored ``authMode``, refusing ambiguity.

    Keyed on the declared mode rather than on which keys the secret happens to
    carry, so serve and discovery agree: a secret ``trino_statement`` rejects as
    contradictory is not silently accepted here under the other mode.
    """
    import trino  # noqa: PLC0415

    mode = (auth_mode or "").upper()
    if mode == "BASIC":
        username, password = secret.get("username"), secret.get("password")
        if not username or not password:
            raise TrinoQueryError("authMode BASIC requires a secret with 'username' and 'password'")
        if "token" in secret:
            raise TrinoQueryError("secret carries both BASIC and JWT keys; refusing to guess the mode")
        return trino.auth.BasicAuthentication(username, password), str(username)
    if mode == "JWT":
        token = secret.get("token")
        if not token:
            raise TrinoQueryError("authMode JWT requires a secret with 'token'")
        if "username" in secret or "password" in secret:
            raise TrinoQueryError("secret carries both JWT and BASIC keys; refusing to guess the mode")
        return trino.auth.JWTAuthentication(token), "coa-serve"
    raise TrinoQueryError(f"unsupported Trino authMode {auth_mode!r}")


def _is_auth_failure(exc: Exception) -> bool:
    """Whether the coordinator rejected the credential rather than the statement.

    Matched on the message because the client collapses both into one error type.
    A stale credential arrives as ``error 401: ...``; Trino's PERMISSION_DENIED
    ("Access Denied: Cannot select from table") is deliberately NOT matched — the
    credential is valid there, so re-reading it cannot help, and a least-privilege
    credential makes that the common denial rather than a rare one.
    """
    text = str(exc).lower()
    return any(marker in text for marker in ("401", "unauthorized", "authentication failed"))


class TrinoQueryExecutor:
    """Executes serve-time SQL on a Trino coordinator.

    One instance per deployment, shared across namespaces. Connections are opened
    per query rather than pooled: a pool would pin one source's credential for the
    process lifetime, and rotation has to propagate within SC-4's TTL.
    """

    def __init__(
        self,
        *,
        sources_registry: SourcesRegistry,
        region: str | None = None,
        max_rows: int = _DEFAULT_MAX_ROWS,
    ) -> None:
        """Wire the sources registry used to resolve a source's coordinator."""
        self._sources = sources_registry
        self._region = region or resolve_region()
        self._max_rows = max_rows
        self._secrets = boto3.client("secretsmanager", region_name=self._region)
        self._breakers: dict[str, _BreakerState] = defaultdict(_BreakerState)
        self._secret_cache: OrderedDict[str, tuple[dict[str, Any], float]] = OrderedDict()
        # Guards the dict only; a cold race fetches twice.
        self._secret_lock = threading.Lock()

    @instrumented("trino")
    async def execute(
        self,
        sql: str,
        *,
        namespace: str,
        data_source_id: str,
        params: dict[str, Any] | None = None,
        max_rows: int = _DEFAULT_MAX_ROWS,
        timeout_seconds: int = _DEFAULT_TIMEOUT,
    ) -> QueryResult:
        """Execute SELECT SQL on the source's Trino coordinator.

        Args:
            sql: Trino-dialect SELECT SQL. Sent unmodified apart from the LIMIT cap.
            namespace: Serving namespace (validated).
            data_source_id: TRINO source id, used to resolve the coordinator.
            params: Ignored — the firewall validates a fully-materialised statement.
            max_rows: Result cap, also injected as a root-level LIMIT.
            timeout_seconds: Per-query budget (1-300).

        Raises:
            TrinoQueryError: the coordinator rejected or failed the query.
            TrinoUnavailableError: unreachable, or the breaker is open.
            ValueError: invalid timeout, or the source carries no Trino config.
        """
        start = time.perf_counter()
        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0 or timeout_seconds > 300:
            raise ValueError(f"timeout_seconds must be an int in 1-300, got {timeout_seconds!r}")

        validate_namespace(namespace)
        _firewall.validate(sql)
        self._check_breaker(namespace)

        config = await self._resolve_config(namespace, data_source_id)
        statement = self._inject_limit(sql, min(max_rows, self._max_rows))

        try:
            rows, columns, truncated = await asyncio.to_thread(
                self._run, statement, config, min(max_rows, self._max_rows), timeout_seconds
            )
        except TrinoUnavailableError:
            emit_metric(_METRICS_NAMESPACE, "TrinoQueryFailures", 1, "Count", FailureClass="unavailable")
            self._record_failure(namespace)
            raise
        except Exception as exc:
            emit_metric(_METRICS_NAMESPACE, "TrinoQueryFailures", 1, "Count", FailureClass=type(exc).__name__)
            # Without this a rotated credential fails every query until the entry ages out.
            if _is_auth_failure(exc):
                self._evict_secret(config.get("credentialSecretArn", ""))
            raise
        self._record_success(namespace)

        duration_ms = int((time.perf_counter() - start) * 1000)
        emit_metric(_METRICS_NAMESPACE, "TrinoQueryDuration", duration_ms, "Milliseconds")
        logger.info(
            "trino_query_executed",
            namespace=namespace,
            data_source_id=data_source_id,
            duration_ms=duration_ms,
            row_count=len(rows),
            truncated=truncated,
        )
        return QueryResult(
            rows=rows,
            columns=columns,
            row_count=len(rows),
            truncated=truncated,
            duration_ms=duration_ms,
            engine="trino",
        )

    async def health_check(self) -> dict[str, Any]:
        """Report reachability without a coordinator to probe.

        There is no deployment-wide Trino endpoint — each source names its own — so
        this reports wiring, not reachability, and never raises.
        """
        return {"status": "ok", "backend": "trino", "allowed_hosts": len(_ALLOWED_HOSTS)}

    async def close(self) -> None:
        """No-op; connections are per-query and already closed."""
        return None

    # ── Internals ────────────────────────────────────────────────────────────

    async def _resolve_config(self, namespace: str, data_source_id: str) -> dict[str, Any]:
        """Read the source's Trino configuration and credential."""
        if not data_source_id:
            raise ValueError("Trino execution requires an explicit data_source_id")
        source = await self._sources.get_source(namespace, data_source_id)
        raw = (source or {}).get("configuration") or "{}"
        config = json.loads(raw) if isinstance(raw, str) else raw
        uri = config.get("uri", "")
        if not uri:
            raise ValueError(f"Source {namespace}/{data_source_id} carries no Trino uri")
        try:
            validate_url(uri, allowed_hosts=_ALLOWED_HOSTS, field_name="trinoConfiguration.uri")
        except UrlGuardError as exc:
            raise TrinoUnavailableError(str(exc)) from exc
        secret = await asyncio.to_thread(self._fetch_secret, config.get("credentialSecretArn", ""))
        return {**config, "secret": secret}

    def _fetch_secret(self, secret_arn: str) -> dict[str, Any]:
        """Read the credential, reusing a cached copy within the TTL."""
        if not secret_arn:
            raise ValueError("Trino source carries no credentialSecretArn")
        now = time.monotonic()
        with self._secret_lock:
            cached = self._secret_cache.get(secret_arn)
            if cached is not None and now - cached[1] < _SECRET_CACHE_TTL_S:
                self._secret_cache.move_to_end(secret_arn)
                return cached[0]

        response = self._secrets.get_secret_value(SecretId=secret_arn)
        secret = json.loads(response["SecretString"])

        with self._secret_lock:
            self._secret_cache[secret_arn] = (secret, time.monotonic())
            self._secret_cache.move_to_end(secret_arn)
            while len(self._secret_cache) > _SECRET_CACHE_MAX:
                self._secret_cache.popitem(last=False)
        return secret

    def _evict_secret(self, secret_arn: str) -> None:
        """Drop a cached credential so the next query re-reads it."""
        if not secret_arn:
            return
        with self._secret_lock:
            if self._secret_cache.pop(secret_arn, None) is not None:
                logger.info("trino_secret_cache_evicted", secret_arn=secret_arn)

    @staticmethod
    def _run(
        statement: str, config: dict[str, Any], max_rows: int, timeout_seconds: int
    ) -> tuple[list[dict[str, Any]], list[str], bool]:
        """Open a connection, run the statement, return capped rows.

        Imported here rather than at module scope so serve starts without the
        ``trino`` package when TRINO_ENABLED is off.
        """
        from urllib.parse import urlsplit  # noqa: PLC0415

        import trino  # noqa: PLC0415

        parts = urlsplit(config["uri"])
        auth, user = _auth_from_secret(
            config.get("auth_mode") or config.get("authMode") or "", config.get("secret") or {}
        )

        connection = trino.dbapi.connect(
            host=parts.hostname,
            port=parts.port or 443,
            user=user,
            auth=auth,
            http_scheme="https",
            catalog=config.get("catalog"),
            schema=config.get("schemaName"),
            source="coa-serve",
            request_timeout=float(timeout_seconds),
            # request_timeout bounds each poll of the nextUri chain, not the query,
            # and to_thread cannot be cancelled — so the coordinator enforces the
            # total, and abandoning the client does not leave the query running.
            session_properties={"query_max_run_time": f"{timeout_seconds}s"},
        )
        try:
            cursor = connection.cursor()
            cursor.execute(statement)
            fetched = cursor.fetchmany(max_rows + 1)
            columns = [d[0] for d in (cursor.description or [])]
        except Exception as exc:
            name = type(exc).__name__
            # Reachability failures feed the breaker; a rejected query does not —
            # a bad statement would otherwise trip it and suppress Tier 2.
            if "Connection" in name or "Timeout" in name or isinstance(exc, OSError):
                raise TrinoUnavailableError(f"Trino coordinator unreachable: {exc}") from exc
            raise TrinoQueryError(str(exc)) from exc
        finally:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - the query's own error is the one worth raising
                logger.debug("trino_close_failed", exc_info=True)

        truncated = len(fetched) > max_rows
        rows = [dict(zip(columns, row, strict=False)) for row in fetched[:max_rows]]
        return rows, columns, truncated

    @staticmethod
    def _inject_limit(sql: str, max_rows: int) -> str:
        """Cap the OUTER query's LIMIT, leaving subquery LIMITs alone.

        Same approach as ``redshift_data._inject_limit``, minus the transpile: the
        statement is already Trino dialect, so it parses and renders in one pass.
        """
        try:
            parsed = sqlglot.parse_one(sql, read="trino")
        except sqlglot.errors.ParseError:
            return f"{sql.rstrip(';')} LIMIT {int(max_rows)}"
        if parsed is None:
            return f"{sql.rstrip(';')} LIMIT {int(max_rows)}"
        existing = parsed.args.get("limit")
        if existing is None:
            return parsed.limit(max_rows).sql(dialect="trino")
        try:
            if int(existing.expression.this) > max_rows:
                return parsed.limit(max_rows).sql(dialect="trino")
        except (AttributeError, TypeError, ValueError):
            return parsed.limit(max_rows).sql(dialect="trino")
        return parsed.sql(dialect="trino")

    def _check_breaker(self, namespace: str) -> None:
        """Fail fast while open, so Tier 2 does not cascade into Tier 3."""
        state = self._breakers[namespace]
        if not state.opened_at:
            return
        if time.monotonic() - state.opened_at < _BREAKER_OPEN_S:
            raise TrinoUnavailableError(
                f"Trino is unavailable for namespace {namespace}; retrying in "
                f"{_BREAKER_OPEN_S:g}s (circuit breaker open)"
            )
        # Half-open: one probe decides.
        state.opened_at = 0.0
        state.failures.clear()

    def _record_failure(self, namespace: str) -> None:
        state = self._breakers[namespace]
        now = time.monotonic()
        state.failures = [t for t in state.failures if now - t < _BREAKER_WINDOW_S]
        state.failures.append(now)
        if len(state.failures) >= _BREAKER_THRESHOLD:
            state.opened_at = now
            emit_metric(_METRICS_NAMESPACE, "TrinoCircuitBreakerOpen", 1, "Count")
            logger.warning("trino_circuit_breaker_open", namespace=namespace, failures=len(state.failures))

    def _record_success(self, namespace: str) -> None:
        state = self._breakers[namespace]
        state.failures.clear()
        state.opened_at = 0.0
