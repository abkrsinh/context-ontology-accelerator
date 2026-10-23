# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover metadata from a Trino catalog/schema via ``information_schema``.

**Two queries per source, no per-table fan-out.** ``information_schema.columns``
carries type, ``ordinal_position``, ``is_nullable`` AND ``comment`` in one result
set — strictly more than ``DESCRIBE`` returns, which has no nullability at all
(``custom_connector.py`` says so outright, leaving ``@notnull`` comment tags as
its only channel). That is why this does not reuse :class:`CustomConnector` by
injecting a Trino runner, even though its ``__init__(self, runner=None)`` makes
that mechanically possible: doing so would inherit the per-table ``DESCRIBE``
fan-out and the Hive-style tab-packed row parsing in order to re-derive *less*
information.

Two Trino behaviours drive the implementation and are easy to get wrong:

* ``comment`` is a ``hiddenColumn`` on ``information_schema.columns``, so it is
  NOT returned by ``SELECT *`` — it must be named explicitly. Selecting ``*``
  here is a silent-metadata-loss bug, which is why a test asserts the emitted SQL
  names it.
* ``information_schema`` prunes COLUMNS/TABLES reads to a schema prefix derived
  from the predicate, and **drops the predicate entirely when the value is not
  already lowercase** — yielding zero rows and no error. The schema name is
  therefore resolved from ``information_schema.schemata`` first; see
  :meth:`TrinoConnector._resolve_schema`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import structlog
from coa_common.domain_models import (
    BusinessMetadata,
    Column,
    DiscoveredMetadata,
    EnrichmentSource,
    PrimaryKey,
    Table,
    TechnicalMetadata,
)

from coa_sources.database.metrics import emit_metric

from .base import ConnectionCheck, ConnectionTestResult
from .constraint_tags import ParsedComment, assemble_constraints, parse_comment
from .filters import compile_filter, unmatched_regex_warning
from .trino_statement import (
    TrinoConfigError,
    TrinoStatementError,
    TrinoStatementRunner,
    quote_ident,
)

logger = structlog.get_logger(__name__)

# One column from `information_schema.columns`: name, data_type, parsed comment,
# and Trino's own `is_nullable`. Nullability sits outside ParsedComment, which is
# frozen and models comment tags only.
_ParsedColumn = tuple[str, str, ParsedComment, bool]

# Only the schema listing is needed to resolve the identifier, and SCHEMATA is
# exempt from the prefix pruning that makes a mis-cased predicate return nothing.
_SCHEMATA_SQL = "SELECT schema_name FROM information_schema.schemata"

# `table_type` is filtered rather than left to chance so views are excluded
# deliberately. It is not free: Trino sets `needsTableType` when the column is
# requested and resolves relation types per prefix to answer it.
_TABLES_SQL = (
    "SELECT table_name FROM information_schema.tables "
    "WHERE table_schema = ? AND table_type = 'BASE TABLE' ORDER BY table_name"
)

# `comment` is named explicitly because it is a hiddenColumn — see the module
# docstring. `extra_info` and `column_comment` are deliberately NOT selected:
# `column_comment` carries the same value as `comment` (the page source writes
# the column's comment into both positions), so there is no fallback to be had,
# and nothing downstream consumes `extra_info`.
#
# ORDER BY ordinal_position is load-bearing, not cosmetic: assemble_constraints
# takes column order as a composite primary key's column order.
_COLUMNS_SQL = (
    "SELECT table_name, column_name, data_type, is_nullable, comment "
    "FROM information_schema.columns "
    "WHERE table_schema = ? ORDER BY table_name, ordinal_position"
)

# Probe for the network/auth checks. Cheap, and it needs no catalog to exist.
_PING_SQL = "SELECT 1"


class SchemaNotFoundError(TrinoStatementError):
    """The configured schema matches nothing the credential can see.

    Carries the available schemas so the error can name them: this is raised at
    source-create, where "which schemas are there?" is the operator's next
    question, and the list has already been fetched to answer it.
    """

    def __init__(self, requested: str, available: list[str]) -> None:
        """Record the requested name and what the credential could actually see."""
        self.requested = requested
        self.available = available
        shown = ", ".join(sorted(available)[:20]) or "(none visible to this credential)"
        super().__init__(f"Schema {requested!r} not found in catalog. Available: {shown}")


def _compute_hash(columns: list[Column]) -> str:
    """Deterministic hash of the column schema, for re-scan change detection.

    Mirrors ``CustomConnector._compute_hash`` and ``GlueCatalogConnector``'s so
    every connector's hash changes under the same conditions.
    """
    normalized = sorted(
        [{"n": c.name, "t": c.data_type, "p": c.is_partition_key} for c in columns],
        key=lambda x: str(x["n"]),
    )
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:16]


def _default_runner_factory(config: dict) -> TrinoStatementRunner:
    """Build a runner from a source's stored configuration."""
    return TrinoStatementRunner(
        uri=config.get("uri", ""),
        auth_mode=config.get("auth_mode", ""),
        secret=config.get("secret") or {},
    )


class TrinoConnector:
    """Metadata connector for the ``TRINO`` sub-type."""

    def __init__(self, runner_factory: Callable[[dict], TrinoStatementRunner] | None = None) -> None:
        """Accept a runner factory rather than a runner.

        ``CustomConnector`` can take a ready-made runner because its Athena
        client is configured from the environment. A Trino runner needs the
        source's own URI and credential, which arrive in ``config`` per call, so
        the injectable seam has to be the factory. ``get_connector()`` constructs
        with no arguments, which this still supports.
        """
        self._runner_factory = runner_factory or _default_runner_factory

    # ── Connection test ──────────────────────────────────────────────────────

    def test_connection(self, config: dict) -> ConnectionTestResult:
        """Verify connectivity: network, auth, and catalog/schema access.

        One :class:`ConnectionCheck` per concern, as ``JdbcConnector`` does, and
        the summary message names the FIRST failure rather than the last.
        """
        checks: list[ConnectionCheck] = []
        catalog, requested = config.get("catalog", ""), config.get("schema_name", "")
        if not catalog or not requested:
            checks.append(
                ConnectionCheck(
                    check="catalog_access",
                    status="failed",
                    message="Missing 'catalog' or 'schema_name' configuration",
                )
            )
            return ConnectionTestResult(success=False, message="Invalid Trino configuration", checks=checks)

        try:
            runner = self._runner_factory(config)
        except TrinoConfigError as exc:
            # Raised before any network call, so this is the operator's URI or
            # secret shape — never the cluster being unreachable.
            checks.append(ConnectionCheck(check="auth", status="failed", message=str(exc)))
            return ConnectionTestResult(success=False, message=str(exc), checks=checks)

        try:
            runner.run(_PING_SQL)
        except TrinoStatementError as exc:
            auth_failed = getattr(exc, "auth_failed", False)
            check = "auth" if auth_failed else "network"
            checks.append(ConnectionCheck(check=check, status="failed", message=str(exc)))
            return ConnectionTestResult(success=False, message=str(exc), checks=checks)
        checks.append(ConnectionCheck(check="network", status="ok", message="Trino coordinator reachable"))
        checks.append(ConnectionCheck(check="auth", status="ok", message="Credential accepted by Trino"))

        try:
            resolved, available = self._resolve_schema(runner, catalog, requested)
        except SchemaNotFoundError as exc:
            checks.append(ConnectionCheck(check="catalog_access", status="failed", message=str(exc)))
            return ConnectionTestResult(success=False, message=str(exc), checks=checks)
        except TrinoStatementError as exc:
            checks.append(ConnectionCheck(check="catalog_access", status="failed", message=str(exc)))
            return ConnectionTestResult(success=False, message=str(exc), checks=checks)
        checks.append(
            ConnectionCheck(
                check="catalog_access",
                status="ok",
                message=f"Schema {resolved!r} found in catalog {catalog!r}",
            )
        )

        # SC-2: COA substitutes its own authorization for Trino's, so a credential
        # broader than the source it serves widens what every COA caller reaches.
        # "Use a least-privilege credential" is unenforceable advice, but it IS
        # observable from the schema list already fetched above — so report it at
        # the moment an operator is looking at the result.
        extra = [s for s in available if s.casefold() != resolved.casefold() and s != "information_schema"]
        if extra:
            checks.append(
                ConnectionCheck(
                    check="credential_scope",
                    status="warning",
                    message=(
                        f"This credential can see {len(extra)} schema(s) beyond {resolved!r}. "
                        "Trino's own per-user authorization does not apply to queries from this "
                        "service, so scope the credential to just what this source should expose."
                    ),
                )
            )

        return ConnectionTestResult(success=True, message="Connection successful", checks=checks)

    # ── Discovery ────────────────────────────────────────────────────────────

    def discover_metadata(self, config: dict) -> DiscoveredMetadata:
        """Discover every table and column in the source's schema.

        Two queries, regardless of table count. Raises rather than returning a
        partial result when either query fails: a scan that reported zero tables
        for a populated schema would reach PENDING_REVIEW looking legitimate.
        """
        catalog = config.get("catalog", "")
        runner = self._runner_factory(config)
        resolved, _ = self._resolve_schema(runner, catalog, config.get("schema_name", ""))

        include = compile_filter(config.get("table_filter"), "table_filter")
        exclude = compile_filter(config.get("table_exclude_filter"), "table_exclude_filter")

        all_names = [row[0] for row in runner.run(_TABLES_SQL, catalog=catalog, params=(resolved,)) if row and row[0]]
        listed = [name for name in all_names if self._is_included(name, include, exclude)]
        # Silent for a glob that legitimately matches nothing, so it does not cover a
        # mis-cased filter — registration lowercases those.
        filter_warnings = [
            w
            for w in (
                unmatched_regex_warning(config.get("table_filter"), "table_filter", all_names),
                unmatched_regex_warning(config.get("table_exclude_filter"), "table_exclude_filter", all_names),
            )
            if w
        ]
        column_rows = runner.run(_COLUMNS_SQL, catalog=catalog, params=(resolved,))

        by_table = self._group_columns(column_rows, keep=set(listed))
        tables: list[Table] = []
        failed: list[str] = []
        for name in listed:
            parsed = by_table.get(name)
            if not parsed:
                # Listed in `tables` but absent from `columns`. A table with no
                # readable columns is indistinguishable from one we failed to
                # read, and shipping it would put an empty table into the
                # ontology for enrichment to invent descriptions over — so it is
                # a scan-level signal, not a log line.
                logger.warning("trino_table_no_columns", catalog=catalog, schema=resolved, table=name)
                failed.append(f"{resolved}.{name}")
                continue
            tables.append(self._build_table(resolved, name, parsed, config))

        emit_metric("TrinoTablesDiscovered", len(tables), "Count")
        if failed:
            emit_metric("TrinoTablesFailed", len(failed), "Count")
        logger.info(
            "trino_discovery_complete",
            catalog=catalog,
            schema=resolved,
            tables=len(tables),
            failed=len(failed),
        )
        return DiscoveredMetadata(tables=tables, failed_tables=sorted(failed), filter_warnings=filter_warnings)

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_schema(runner: TrinoStatementRunner, catalog: str, requested: str) -> tuple[str, list[str]]:
        """Return the schema name as Trino spells it, plus every schema visible.

        Resolved from the system of record rather than lowercased defensively.
        ``information_schema`` prunes COLUMNS/TABLES reads to a schema prefix and
        filters the candidates through an ``isLowerCase`` check, so a predicate of
        ``table_schema = 'Sales'`` is dropped and the read returns **zero rows
        with no error** — the worst available failure shape, since discovery would
        report a real schema as empty and the scan would reach PENDING_REVIEW
        looking legitimately so.

        ``SCHEMATA`` is exempt from that pruning, so asking it for the canonical
        spelling is a structural fix rather than a guess: the value used in the
        predicate is whatever Trino itself reported, which the ``isLowerCase``
        filter cannot drop. It also moves the failure to source-create with an
        error naming the available schemas, and makes "resolved but zero columns"
        an unambiguous data condition rather than a possible casing bug.
        """
        rows = runner.run(_SCHEMATA_SQL, catalog=catalog)
        available = [row[0] for row in rows if row and row[0]]
        for name in available:
            if name.casefold() == requested.casefold():
                return name, available
        raise SchemaNotFoundError(requested, available)

    @staticmethod
    def _is_included(table: str, include: Any, exclude: Any) -> bool:
        """Apply the include filter then the exclude filter, in that order."""
        if include is not None and not include.match(table):
            return False
        return not (exclude is not None and exclude.match(table))

    @staticmethod
    def _group_columns(rows: list[list[str | None]], *, keep: set[str]) -> dict[str, list[_ParsedColumn]]:
        """Group column rows by table, preserving the query's ordinal ordering.

        ``keep`` drops columns of tables the filters excluded and of views, which
        the TABLES query already filtered out — the COLUMNS query has no
        ``table_type`` to filter on, so it returns both.
        """
        grouped: dict[str, list[_ParsedColumn]] = {}
        for row in rows:
            if len(row) < 5:
                continue
            table, column, data_type, is_nullable, comment = row[0], row[1], row[2], row[3], row[4]
            if not table or not column or table not in keep:
                continue
            grouped.setdefault(table, []).append(
                (
                    column,
                    # data_type verbatim: Trino's toTrinoType already produced it,
                    # and COA's downstream reduction is `.split("(")[0].upper()`
                    # against a dict with a safe default, so row(...),
                    # decimal(18,4) and timestamp(6) with time zone all behave.
                    # We maintain no type table.
                    data_type or "",
                    # @pk / @fk / @notnull still parsed from the comment via the
                    # shared, engine-independent helper, so a source that encodes
                    # keys in comments keeps working through Trino.
                    parse_comment(comment),
                    (is_nullable or "").upper() != "NO",
                )
            )
        return grouped

    @staticmethod
    def _build_table(schema: str, table: str, parsed_columns: list[_ParsedColumn], config: dict) -> Table:
        """Assemble a :class:`Table`, mirroring ``CustomConnector._build_table``."""
        columns = [
            Column(
                name=name,
                data_type=data_type,
                # The real improvement over the Athena path, where "DESCRIBE
                # carries no nullability … so `@notnull` in the comment is the
                # only channel". `is_nullable` is the source's own answer; the tag
                # can still TIGHTEN it, so a connector-style source sitting behind
                # Trino is not regressed.
                nullable=source_nullable and not comment.is_not_null,
                business_metadata=(
                    BusinessMetadata(
                        description=comment.description,
                        enrichment_source=EnrichmentSource.DETERMINISTIC,
                        confidence=1.0,
                    )
                    if comment.description
                    else BusinessMetadata()
                ),
            )
            for name, data_type, comment, source_nullable in parsed_columns
        ]
        # Order matters: assemble_constraints takes column order as the composite
        # primary key's column order, which is why the query sorts by
        # ordinal_position.
        primary_key, foreign_keys = assemble_constraints([(name, comment) for name, _, comment, _ in parsed_columns])
        return Table(
            name=table,
            database=schema,
            data_source_id=config.get("data_source_id", ""),
            namespace_id=config.get("namespace_id", ""),
            # No `format`: information_schema does not say which connector answered.
            technical_metadata=TechnicalMetadata(column_count=len(columns)),
            # Deliberately no table-level description: Trino's
            # information_schema.tables writes `table_comment` as literal null on
            # every row, so there is nothing to read and table descriptions come
            # from enrichment Pass 1 — the same posture CustomConnector documents.
            business_metadata=BusinessMetadata(),
            # An empty PrimaryKey, not None: the field is non-optional, and an
            # empty one reads as "no declared key" everywhere downstream.
            primary_key=primary_key or PrimaryKey(),
            foreign_keys=foreign_keys,
            columns=columns,
            technical_metadata_hash=_compute_hash(columns),
        )


__all__ = ["SchemaNotFoundError", "TrinoConnector", "quote_ident"]
