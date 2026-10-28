# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrinoSampler.

The behaviours that carry risk:

* **The query count stays constant.** The whole reason sampling is affordable
  here is that it does not fan out per column, so a wide schema must still issue
  two statements per chunk. A regression to per-column queries would reintroduce
  the timeout risk AthenaSampler's docstring warns about.
* **High-cardinality columns are never collected.** ``array_agg(DISTINCT col)``
  accumulates every value it sees, so the cardinality gate is what keeps a
  free-text column from being pulled into memory.
* **Values arrive byte-identical.** Their only purpose is literal matching, so a
  case change would defeat the point of sampling at all.
* **Failure is non-fatal.** Sampling is a quality improvement; it must never
  cost the scan.
"""

from __future__ import annotations

import json

import pytest
from coa_sources.database.connectors.trino_sampler import TrinoSampler, is_sampleable_type
from coa_sources.database.connectors.trino_statement import quote_ident, quote_literal

pytestmark = pytest.mark.unit


class _Runner:
    """Records statements and replays canned JSON rows per call."""

    def __init__(self, responses: list[list[list[str]]] | None = None, error: Exception | None = None) -> None:
        self.statements: list[str] = []
        self._responses = responses or []
        self._error = error

    def run(self, statement, *, catalog=None, schema=None, timeout_seconds=None):
        self.statements.append(statement)
        if self._error:
            raise self._error
        if not self._responses:
            return []
        return self._responses.pop(0)


def _json_rows(payload: dict[str, dict]) -> list[list[str]]:
    return [[table, json.dumps(cols)] for table, cols in payload.items()]


_ORDERS = ("orders", [("status", "varchar(10)"), ("note", "varchar"), ("total", "decimal(18,2)")])


# ── Candidate selection ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("data_type", "expect"),
    [
        ("varchar", True),
        ("varchar(25)", True),
        ("char(2)", True),
        ("VARCHAR(10)", True),
        ("decimal(18,2)", False),
        ("bigint", False),
        ("date", False),
        ("timestamp(6) with time zone", False),
        ("array(varchar)", False),
        ("", False),
    ],
)
def test_only_string_columns_are_candidates(data_type: str, expect: bool) -> None:
    """Parameterised types arrive verbatim from information_schema, so the check
    is on the prefix — and `array(varchar)` must not match on a substring."""
    assert is_sampleable_type(data_type) is expect


def test_a_schema_with_no_string_columns_issues_no_queries() -> None:
    runner = _Runner()
    out = TrinoSampler(runner).sample_all(
        [("metrics", [("value", "double"), ("day", "date")])], catalog="c", schema="s"
    )
    assert out == {}
    assert runner.statements == []


# ── The constant-query-count property ────────────────────────────────────


def test_sampling_adds_two_queries_not_per_column() -> None:
    """The property the design rests on: one statement per pass, not per column."""
    cols = [(f"c{i}", "varchar") for i in range(40)]
    runner = _Runner(
        [
            _json_rows({"wide": {f"c{i}": 3 for i in range(40)}}),
            _json_rows({"wide": {f"c{i}": ["A", "B", "C"] for i in range(40)}}),
        ]
    )
    out = TrinoSampler(runner).sample_all([("wide", cols)], catalog="c", schema="s")
    assert len(runner.statements) == 2
    assert len(out["wide"]) == 40


def test_many_tables_share_one_statement_per_pass() -> None:
    tables = [(f"t{i}", [("status", "varchar")]) for i in range(10)]
    runner = _Runner(
        [
            _json_rows({f"t{i}": {"status": 2} for i in range(10)}),
            _json_rows({f"t{i}": {"status": ["N", "Y"]} for i in range(10)}),
        ]
    )
    TrinoSampler(runner).sample_all(tables, catalog="c", schema="s")
    assert len(runner.statements) == 2
    assert runner.statements[0].count("UNION ALL") == 9


def test_candidates_are_chunked_past_the_ceiling() -> None:
    """A wide schema becomes a few bounded statements, never one unbounded one."""
    cols = [(f"c{i}", "varchar") for i in range(10)]
    runner = _Runner([_json_rows({"t": {}})] * 10)
    TrinoSampler(runner, max_columns_per_query=4).sample_all([("t", cols)], catalog="c", schema="s")
    # 10 candidates at 4 per statement → 3 chunks; the cardinality pass of each
    # returns nothing sampleable, so only the first pass runs per chunk.
    assert len(runner.statements) == 3
    for stmt in runner.statements:
        assert stmt.count("approx_distinct") <= 4


# ── The cardinality gate ─────────────────────────────────────────────────


def test_high_cardinality_column_is_not_collected() -> None:
    """Without this gate, array_agg would accumulate every value of a free-text
    column — the failure mode the two-pass split exists to prevent."""
    runner = _Runner(
        [
            _json_rows({"orders": {"status": 3, "note": 995}}),
            _json_rows({"orders": {"status": ["F", "O", "P"]}}),
        ]
    )
    out = TrinoSampler(runner, max_distinct=25).sample_all([_ORDERS], catalog="c", schema="s")
    assert out == {"orders": {"status": ["F", "O", "P"]}}
    assert "note" not in runner.statements[1]
    assert "array_agg" in runner.statements[1]


def test_a_zero_cardinality_column_is_skipped() -> None:
    """An empty column has nothing to match on, so collecting it is wasted work."""
    runner = _Runner([_json_rows({"orders": {"status": 0, "note": 0}})])
    out = TrinoSampler(runner).sample_all([_ORDERS], catalog="c", schema="s")
    assert out == {}
    assert len(runner.statements) == 1


def test_the_threshold_is_inclusive() -> None:
    runner = _Runner(
        [
            _json_rows({"orders": {"status": 25}}),
            _json_rows({"orders": {"status": ["a"]}}),
        ]
    )
    out = TrinoSampler(runner, max_distinct=25).sample_all(
        [("orders", [("status", "varchar")])], catalog="c", schema="s"
    )
    assert out["orders"]["status"] == ["a"]


# ── Values reach the caller unchanged ────────────────────────────────────


def test_values_are_preserved_verbatim() -> None:
    """Case and spacing are the whole point: NL→SQL has to emit the literal the
    data actually stores."""
    runner = _Runner(
        [
            _json_rows({"nation": {"name": 3}}),
            _json_rows({"nation": {"name": ["FRANCE", "SAUDI ARABIA", "United Kingdom"]}}),
        ]
    )
    out = TrinoSampler(runner).sample_all([("nation", [("name", "varchar(25)")])], catalog="c", schema="s")
    assert out["nation"]["name"] == ["FRANCE", "SAUDI ARABIA", "United Kingdom"]


def test_nulls_are_dropped_from_the_values() -> None:
    runner = _Runner(
        [
            _json_rows({"orders": {"status": 3}}),
            _json_rows({"orders": {"status": ["F", None, "O"]}}),
        ]
    )
    out = TrinoSampler(runner).sample_all([("orders", [("status", "varchar")])], catalog="c", schema="s")
    assert out["orders"]["status"] == ["F", "O"]


# ── Non-fatal behaviour and SQL shape ────────────────────────────────────


def test_a_sampling_failure_returns_empty_rather_than_raising() -> None:
    """Sampling costs literal quality when it fails, never the scan."""
    runner = _Runner(error=RuntimeError("coordinator busy"))
    assert TrinoSampler(runner).sample_all([_ORDERS], catalog="c", schema="s") == {}


def test_an_unparseable_row_is_skipped_not_fatal() -> None:
    runner = _Runner([[["orders", "not json"]]])
    assert TrinoSampler(runner).sample_all([_ORDERS], catalog="c", schema="s") == {}


def test_identifiers_are_quoted_in_the_emitted_sql() -> None:
    """A column named like a keyword, or carrying a quote, must not break out of
    its identifier position."""
    runner = _Runner([_json_rows({"t": {}})])
    TrinoSampler(runner).sample_all([("t", [('we"ird', "varchar")])], catalog="c", schema="s")
    assert '"we""ird"' in runner.statements[0]


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("o'brien", "it's"),
        ("t", "it's"),
        ("o'brien", "c"),
    ],
)
def test_an_apostrophe_in_a_name_does_not_break_the_statement(table: str, column: str) -> None:
    """Names also reach LITERAL positions — the table label and the key array —
    where doubling `"` protects nothing. A legal name carrying `'` previously
    produced a syntax error, and `sample_all` swallows it, so sampling was lost
    for the whole chunk with nothing surfaced."""
    runner = _Runner([_json_rows({table: {}})])
    TrinoSampler(runner).sample_all([(table, [(column, "varchar")])], catalog="c", schema="s")
    stmt = runner.statements[0]
    # Doubled where it is DATA (the table label, the key array), raw inside
    # double quotes where it is an IDENTIFIER. Both escapes, each in its place.
    assert quote_literal(table) in stmt
    assert quote_literal(column) in stmt
    assert quote_ident(table) in stmt


def test_a_crafted_name_cannot_rewrite_the_statement() -> None:
    """Whoever can create a table in the scanned schema must not be able to
    redirect the arm at another relation, whose rows would otherwise land in
    distinct_values and from there in the Tier-2 prompt."""
    payload = "x' AS tbl, cast(map(ARRAY['a'],ARRAY[(SELECT 1)]) AS json) AS v FROM other --"
    runner = _Runner([_json_rows({payload: {}})])
    TrinoSampler(runner).sample_all([(payload, [("a", "varchar")])], catalog="c", schema="s")
    stmt = runner.statements[0]
    # The payload survives only inside a properly closed literal and a properly
    # closed identifier — so it is text, never syntax.
    assert quote_literal(payload) in stmt
    assert quote_ident(payload) in stmt


def test_the_runner_is_required() -> None:
    """The old default constructed a TrinoStatementRunner with no arguments,
    which raises: uri, auth_mode and secret are required keyword-only."""
    with pytest.raises(TypeError):
        TrinoSampler()  # type: ignore[call-arg]


def test_the_cardinality_pass_casts_to_json() -> None:
    """The statement runner's contract is str|None rows, so a native MAP would
    decode differently per client."""
    runner = _Runner([_json_rows({"orders": {"status": 3}}), _json_rows({"orders": {"status": ["F"]}})])
    TrinoSampler(runner).sample_all([("orders", [("status", "varchar")])], catalog="c", schema="s")
    assert "AS json" in runner.statements[0]
