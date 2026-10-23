# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrinoConnector — discovery via ``information_schema``.

The tests that would have caught real bugs, and why each exists:

* **two queries, never a fan-out** — the property the whole design rests on.
* **``comment`` named explicitly** — it is a ``hiddenColumn``, so ``SELECT *``
  would return everything *except* the comments, silently, and enrichment would
  then invent descriptions over the gap.
* **schema resolved from ``information_schema.schemata``** — a mis-cased
  predicate makes Trino drop the predicate and return zero rows with no error,
  which would present as a real schema discovered empty.
* **``is_nullable`` with ``@notnull`` still tightening it** — the improvement
  over the Athena path, without regressing sources that use the tag.
* **a listed table with no columns lands in ``failed_tables``** — not a log line,
  or a steward reviews a silently incomplete ontology as complete.

The fake runner records every statement so the SQL itself can be asserted on,
which is what makes the ``SELECT *`` guard possible.
"""

from __future__ import annotations

from typing import Any

import pytest
from coa_common.domain_models import EnrichmentSource
from coa_sources.database.connectors.trino_source import (
    SchemaNotFoundError,
    TrinoConnector,
)
from coa_sources.database.connectors.trino_statement import (
    TrinoConfigError,
    TrinoConnectionFailed,
    TrinoStatementFailed,
)

pytestmark = pytest.mark.unit

_CONFIG = {
    "uri": "https://trino.example.com:8443",
    "catalog": "iceberg",
    "schema_name": "sales",
    "auth_mode": "JWT",
    "secret": {"token": "h.e.s"},
    "data_source_id": "ds-123",
    "namespace_id": "ns-456",
}


class FakeRunner:
    """Runner double that answers by matching the statement's relation.

    ``schemata`` / ``tables`` / ``columns`` are the rows returned for each of the
    three queries; ``errors`` maps the same keys to an exception to raise
    instead.
    """

    def __init__(
        self,
        *,
        schemata: list[list[str | None]] | None = None,
        tables: list[list[str | None]] | None = None,
        columns: list[list[str | None]] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.schemata = schemata if schemata is not None else [["sales"]]
        self.tables = tables if tables is not None else []
        self.columns = columns if columns is not None else []
        self.errors = errors or {}
        self.calls: list[dict[str, Any]] = []

    def run(self, statement: str, **kwargs: Any) -> list[list[str | None]]:
        self.calls.append({"sql": statement, **kwargs})
        key = self._key(statement)
        if key in self.errors:
            raise self.errors[key]
        return {"schemata": self.schemata, "tables": self.tables, "columns": self.columns, "ping": [["1"]]}[key]

    @staticmethod
    def _key(statement: str) -> str:
        for relation in ("schemata", "columns", "tables"):
            if f"information_schema.{relation}" in statement:
                return relation
        return "ping"

    @property
    def statements(self) -> list[str]:
        return [c["sql"] for c in self.calls]

    def discovery_statements(self) -> list[str]:
        """Statements excluding the schema resolution, which is not a scan query."""
        return [s for s in self.statements if "information_schema.schemata" not in s]


def _col(table: str, name: str, dtype: str = "varchar", nullable: str = "YES", comment: str | None = None) -> list:
    return [table, name, dtype, nullable, comment]


def _connector(runner: FakeRunner) -> TrinoConnector:
    return TrinoConnector(runner_factory=lambda _config: runner)


# ── The two-query property ───────────────────────────────────────────────────


def test_discovery_issues_exactly_two_queries_regardless_of_table_count() -> None:
    """The property the design rests on: O(2), not O(tables)."""
    tables = [[f"t{i}"] for i in range(50)]
    columns = [_col(f"t{i}", "id", "bigint", "NO") for i in range(50)]
    runner = FakeRunner(tables=tables, columns=columns)
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert len(result.tables) == 50
    assert len(runner.discovery_statements()) == 2


def test_comment_is_selected_explicitly_and_never_star() -> None:
    """`comment` is a hiddenColumn: SELECT * returns everything but the comments."""
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    _connector(runner).discover_metadata(dict(_CONFIG))
    columns_sql = next(s for s in runner.statements if "information_schema.columns" in s)
    assert "comment" in columns_sql
    assert "SELECT *" not in columns_sql.upper()


def test_columns_query_orders_by_ordinal_position() -> None:
    """Load-bearing: assemble_constraints reads column order as composite PK order."""
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    _connector(runner).discover_metadata(dict(_CONFIG))
    columns_sql = next(s for s in runner.statements if "information_schema.columns" in s)
    assert "ordinal_position" in columns_sql


def test_views_are_excluded_by_table_type() -> None:
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    _connector(runner).discover_metadata(dict(_CONFIG))
    tables_sql = next(s for s in runner.statements if "information_schema.tables" in s)
    assert "table_type = 'BASE TABLE'" in tables_sql


def test_schema_predicate_is_a_bound_parameter_not_interpolated() -> None:
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    _connector(runner).discover_metadata(dict(_CONFIG))
    scans = [c for c in runner.calls if "information_schema.schemata" not in c["sql"]]
    assert len(scans) == 2, "expected both scan queries"
    for call in scans:
        assert "?" in call["sql"]
        assert call["params"] == ("sales",)
        assert "'sales'" not in call["sql"]


# ── Schema resolution ────────────────────────────────────────────────────────


def test_schema_is_resolved_from_trino_not_trusted_from_the_caller() -> None:
    """Caller says `Sales`; Trino says `sales`; the predicate must use `sales`.

    A mis-cased predicate is dropped by Trino's prefix pruning and returns zero
    rows with no error — a real schema reported empty.
    """
    runner = FakeRunner(schemata=[["sales"]], tables=[["orders"]], columns=[_col("orders", "id")])
    config = dict(_CONFIG, schema_name="Sales")
    result = _connector(runner).discover_metadata(config)
    scans = [c for c in runner.calls if "information_schema.schemata" not in c["sql"]]
    assert len(scans) == 2, "expected both scan queries"
    assert [c["params"] for c in scans] == [("sales",), ("sales",)]
    assert result.tables[0].database == "sales"


def test_unknown_schema_raises_and_names_what_exists() -> None:
    runner = FakeRunner(schemata=[["public"], ["staging"]])
    with pytest.raises(SchemaNotFoundError) as exc:
        _connector(runner).discover_metadata(dict(_CONFIG))
    assert "public" in str(exc.value)
    assert "staging" in str(exc.value)
    assert exc.value.requested == "sales"


def test_resolved_schema_with_zero_columns_is_reported_not_silently_empty() -> None:
    """Unambiguous once the name is resolved: a data condition, not a casing bug."""
    runner = FakeRunner(tables=[["orders"]], columns=[])
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert result.tables == []
    assert result.failed_tables == ["sales.orders"]


def test_schemata_query_is_not_prefix_filtered() -> None:
    """It must not carry a table_schema predicate — that is what exempts it."""
    runner = FakeRunner(tables=[], columns=[])
    _connector(runner).discover_metadata(dict(_CONFIG))
    schemata_sql = next(s for s in runner.statements if "information_schema.schemata" in s)
    assert "where" not in schemata_sql.lower()


# ── Column mapping ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(("is_nullable", "expected"), [("YES", True), ("NO", False)])
def test_is_nullable_maps_to_column_nullable(is_nullable: str, expected: bool) -> None:
    """The real improvement over the Athena path, which has no nullability at all."""
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id", "bigint", is_nullable)])
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert result.tables[0].columns[0].nullable is expected


def test_notnull_tag_still_tightens_a_nullable_column() -> None:
    """No regression for a connector-style source sitting behind Trino."""
    runner = FakeRunner(
        tables=[["orders"]],
        columns=[_col("orders", "id", "bigint", "YES", "the id @notnull")],
    )
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert result.tables[0].columns[0].nullable is False


def test_comment_becomes_a_deterministic_description() -> None:
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id", "bigint", "NO", "Order identifier")])
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    meta = result.tables[0].columns[0].business_metadata
    assert meta.description == "Order identifier"
    assert meta.enrichment_source == EnrichmentSource.DETERMINISTIC
    assert meta.confidence == 1.0


def test_absent_comment_leaves_business_metadata_empty() -> None:
    """Not a blank DETERMINISTIC description, which would outrank enrichment."""
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id", "bigint", "NO", None)])
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    meta = result.tables[0].columns[0].business_metadata
    assert meta.description == ""
    assert meta.enrichment_source != EnrichmentSource.DETERMINISTIC


@pytest.mark.parametrize(
    "dtype",
    [
        "row(street varchar, zip integer)",
        "array(varchar)",
        "map(varchar, bigint)",
        "decimal(18,4)",
        "timestamp(6) with time zone",
        "varbinary",
    ],
)
def test_data_type_is_taken_verbatim(dtype: str) -> None:
    """We maintain no Iceberg type table — Trino's toTrinoType already ran."""
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "c", dtype)])
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert result.tables[0].columns[0].data_type == dtype


def test_column_order_follows_the_query_order() -> None:
    runner = FakeRunner(
        tables=[["orders"]],
        columns=[_col("orders", "a"), _col("orders", "b"), _col("orders", "c")],
    )
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert [c.name for c in result.tables[0].columns] == ["a", "b", "c"]


# ── Constraint tags ──────────────────────────────────────────────────────────


def test_pk_tag_is_parsed_from_the_comment() -> None:
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id", "bigint", "NO", "@pk")])
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert result.tables[0].primary_key.columns == ["id"]


def test_composite_pk_column_order_follows_ordinal_position() -> None:
    """The reason ORDER BY ordinal_position is load-bearing rather than cosmetic."""
    runner = FakeRunner(
        tables=[["line_items"]],
        columns=[
            _col("line_items", "order_id", "bigint", "NO", "@pk"),
            _col("line_items", "line_no", "integer", "NO", "@pk"),
        ],
    )
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert result.tables[0].primary_key.columns == ["order_id", "line_no"]


def test_fk_tag_is_parsed_from_the_comment() -> None:
    runner = FakeRunner(
        tables=[["line_items"]],
        columns=[_col("line_items", "order_id", "bigint", "NO", "@fk(orders.id)")],
    )
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    fks = result.tables[0].foreign_keys
    assert len(fks) == 1
    assert (fks[0].column, fks[0].target_table, fks[0].target_column) == ("order_id", "orders", "id")


def test_tags_are_stripped_from_the_stored_description() -> None:
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id", "bigint", "NO", "Order id @pk")])
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert result.tables[0].columns[0].business_metadata.description == "Order id"


# ── Table assembly ───────────────────────────────────────────────────────────


def test_table_carries_schema_ids_and_column_count() -> None:
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id"), _col("orders", "total")])
    table = _connector(runner).discover_metadata(dict(_CONFIG)).tables[0]
    assert (table.name, table.database) == ("orders", "sales")
    assert (table.data_source_id, table.namespace_id) == ("ds-123", "ns-456")
    assert table.technical_metadata.column_count == 2


def test_table_description_is_left_for_enrichment() -> None:
    """Trino writes information_schema.tables.table_comment as literal null."""
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    table = _connector(runner).discover_metadata(dict(_CONFIG)).tables[0]
    assert table.business_metadata.description == ""


def test_hash_is_stable_across_scans_and_changes_with_the_schema() -> None:
    def scan(columns: list[list]) -> str:
        runner = FakeRunner(tables=[["orders"]], columns=columns)
        return _connector(runner).discover_metadata(dict(_CONFIG)).tables[0].technical_metadata_hash

    base = [_col("orders", "id", "bigint")]
    assert scan(base) == scan(base)
    assert scan(base) != scan([_col("orders", "id", "varchar")])
    assert scan(base) != scan(base + [_col("orders", "total", "double")])


def test_columns_of_unlisted_tables_are_ignored() -> None:
    """The COLUMNS query has no table_type, so it returns views too."""
    runner = FakeRunner(
        tables=[["orders"]],
        columns=[_col("orders", "id"), _col("orders_view", "id")],
    )
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert [t.name for t in result.tables] == ["orders"]


# ── Filters ──────────────────────────────────────────────────────────────────


def test_exclude_filter_accepts_multiple_patterns() -> None:
    """`staging_*|temp_*` must exclude both — the compile_filter trap."""
    runner = FakeRunner(
        tables=[["orders"], ["staging_x"], ["temp_y"]],
        columns=[_col("orders", "id"), _col("staging_x", "id"), _col("temp_y", "id")],
    )
    config = dict(_CONFIG, table_exclude_filter="staging_*|temp_*")
    result = _connector(runner).discover_metadata(config)
    assert [t.name for t in result.tables] == ["orders"]


def test_include_filter_restricts_to_matches() -> None:
    runner = FakeRunner(
        tables=[["orders"], ["customers"]],
        columns=[_col("orders", "id"), _col("customers", "id")],
    )
    result = _connector(runner).discover_metadata(dict(_CONFIG, table_filter="orders"))
    assert [t.name for t in result.tables] == ["orders"]


def test_exclude_wins_over_include() -> None:
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    config = dict(_CONFIG, table_filter="order*", table_exclude_filter="orders")
    assert _connector(runner).discover_metadata(config).tables == []


def test_excluded_tables_never_reach_failed_tables() -> None:
    """An excluded table is not a failure, even with no columns returned."""
    runner = FakeRunner(tables=[["orders"], ["staging_x"]], columns=[_col("orders", "id")])
    result = _connector(runner).discover_metadata(dict(_CONFIG, table_exclude_filter="staging_*"))
    assert result.failed_tables == []


# ── Partial failure ──────────────────────────────────────────────────────────


def test_table_listed_without_columns_lands_in_failed_tables() -> None:
    runner = FakeRunner(
        tables=[["orders"], ["broken"]],
        columns=[_col("orders", "id")],
    )
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert [t.name for t in result.tables] == ["orders"]
    assert result.failed_tables == ["sales.broken"]


def test_a_failed_query_raises_rather_than_returning_a_partial_result() -> None:
    """Zero tables for a populated schema would reach PENDING_REVIEW looking fine."""
    runner = FakeRunner(errors={"columns": TrinoStatementFailed("sql", "PERMISSION_DENIED", "denied")})
    with pytest.raises(TrinoStatementFailed):
        _connector(runner).discover_metadata(dict(_CONFIG))


def test_a_short_column_row_is_skipped_rather_than_unpacked() -> None:
    """Guards the five-way unpack against a result set that is not the shape asked for."""
    runner = FakeRunner(tables=[["orders"]], columns=[["orders", "id"], _col("orders", "total")])
    result = _connector(runner).discover_metadata(dict(_CONFIG))
    assert [c.name for c in result.tables[0].columns] == ["total"]


# ── test_connection ──────────────────────────────────────────────────────────


def test_connection_reports_a_check_per_concern() -> None:
    runner = FakeRunner(schemata=[["sales"]])
    result = _connector(runner).test_connection(dict(_CONFIG))
    assert result.success is True
    assert {c.check for c in result.checks} >= {"network", "auth", "catalog_access"}
    assert all(c.status in {"ok", "warning"} for c in result.checks)


def test_connection_failure_names_the_first_failing_check() -> None:
    runner = FakeRunner(errors={"ping": TrinoConnectionFailed("SELECT 1", RuntimeError("no route"))})
    result = _connector(runner).test_connection(dict(_CONFIG))
    assert result.success is False
    assert [c.check for c in result.checks if c.status == "failed"] == ["network"]


def test_connection_distinguishes_auth_failure_from_unreachable() -> None:
    runner = FakeRunner(errors={"ping": TrinoConnectionFailed("SELECT 1", RuntimeError("401"), auth_failed=True)})
    result = _connector(runner).test_connection(dict(_CONFIG))
    assert [c.check for c in result.checks if c.status == "failed"] == ["auth"]


def test_connection_reports_an_unknown_schema_as_catalog_access() -> None:
    runner = FakeRunner(schemata=[["public"]])
    result = _connector(runner).test_connection(dict(_CONFIG))
    assert result.success is False
    failed = [c for c in result.checks if c.status == "failed"]
    assert failed[0].check == "catalog_access"
    assert "public" in failed[0].message


def test_connection_warns_when_the_credential_is_broader_than_the_source() -> None:
    """SC-2: unenforceable as advice, but observable from a query already made."""
    runner = FakeRunner(schemata=[["sales"], ["finance"], ["hr"]])
    result = _connector(runner).test_connection(dict(_CONFIG))
    warnings = [c for c in result.checks if c.status == "warning"]
    assert result.success is True
    assert len(warnings) == 1
    assert warnings[0].check == "credential_scope"
    assert "2 schema(s) beyond" in warnings[0].message


def test_connection_does_not_warn_on_a_correctly_scoped_credential() -> None:
    """information_schema is always visible and must not count as over-broad."""
    runner = FakeRunner(schemata=[["sales"], ["information_schema"]])
    result = _connector(runner).test_connection(dict(_CONFIG))
    assert [c for c in result.checks if c.status == "warning"] == []


@pytest.mark.parametrize("missing", ["catalog", "schema_name"])
def test_connection_rejects_incomplete_configuration(missing: str) -> None:
    config = dict(_CONFIG)
    config[missing] = ""
    result = _connector(FakeRunner()).test_connection(config)
    assert result.success is False
    assert [c.check for c in result.checks] == ["catalog_access"]


def test_connection_reports_a_catalog_error_that_is_not_a_missing_schema() -> None:
    """A non-existent catalog fails the schemata read itself, not the name match."""
    runner = FakeRunner(errors={"schemata": TrinoStatementFailed("sql", "CATALOG_NOT_FOUND", "no catalog 'iceberg'")})
    result = _connector(runner).test_connection(dict(_CONFIG))
    assert result.success is False
    failed = [c for c in result.checks if c.status == "failed"]
    assert failed[0].check == "catalog_access"
    assert "CATALOG_NOT_FOUND" in failed[0].message


def test_the_default_runner_factory_builds_a_runner_from_config() -> None:
    """The production path: no factory injected, so config alone must suffice."""
    connector = TrinoConnector()
    runner = connector._runner_factory(dict(_CONFIG))  # noqa: SLF001 - the seam under test
    assert (runner._host, runner._port) == ("trino.example.com", 8443)  # noqa: SLF001


def test_the_default_runner_factory_surfaces_a_bad_secret_as_a_config_error() -> None:
    with pytest.raises(TrinoConfigError):
        TrinoConnector()._runner_factory(dict(_CONFIG, secret={}))  # noqa: SLF001 - the seam under test


def test_connection_reports_a_config_error_before_any_network_call() -> None:
    """A bad URI or contradicting secret is the operator's, not the cluster's."""

    def factory(_config: dict) -> Any:
        raise TrinoConfigError("authMode JWT requires a secret with 'token'")

    result = TrinoConnector(runner_factory=factory).test_connection(dict(_CONFIG))
    assert result.success is False
    assert [c.check for c in result.checks] == ["auth"]
    assert "token" in result.message


def test_regex_shaped_filter_that_matched_nothing_is_warned_about() -> None:
    """Mirrors every sibling connector: a filter with no effect reaches the scan job."""
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    result = _connector(runner).discover_metadata(dict(_CONFIG, table_filter="^(staging|tmp)$"))
    assert len(result.filter_warnings) == 1
    assert "table_filter" in result.filter_warnings[0]


def test_both_filters_can_warn_independently() -> None:
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    config = dict(_CONFIG, table_filter="^orders$", table_exclude_filter="^tmp$")
    result = _connector(runner).discover_metadata(config)
    assert len(result.filter_warnings) == 2


def test_a_glob_that_legitimately_matches_nothing_stays_silent() -> None:
    """An exclude for a table that does not exist yet is not a mistake."""
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    result = _connector(runner).discover_metadata(dict(_CONFIG, table_exclude_filter="staging_*"))
    assert result.filter_warnings == []


def test_no_filters_means_no_warnings() -> None:
    runner = FakeRunner(tables=[["orders"]], columns=[_col("orders", "id")])
    assert _connector(runner).discover_metadata(dict(_CONFIG)).filter_warnings == []
