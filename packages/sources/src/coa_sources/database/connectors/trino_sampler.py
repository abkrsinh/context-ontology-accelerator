# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distinct-value sampling for Trino tables, in a constant number of queries.

Tier-2 NL→SQL matches literals from ``Column.distinct_values``, which
``tbox_context`` pastes into the prompt. Without them the model has to guess how
a categorical value is spelled and cased in the data, and a wrong guess produces
a statement that is valid, runs, and returns nothing — the hardest failure to
diagnose, because there is no error anywhere.

``AthenaSampler`` supplies those values by fanning out one query per column,
which is why its own docstring flags a Lambda-timeout risk at scale. Trino does
not need that: one statement can carry every candidate column of every table,
since the per-column aggregates collapse into a map and the tables into a UNION
ALL. Sampling therefore costs TWO queries for a whole schema rather than one per
column, and each pass reads a table once no matter how many of its columns are
candidates.

Two passes rather than one, because ``array_agg(DISTINCT col)`` accumulates every
value it encounters: ``approx_distinct`` is the cheap probe that decides which
columns are small enough to collect.

Both passes read at most :data:`PROBE_MAX_ROWS` rows per table, and the gate and
the collection apply the same bounds ``AthenaSampler`` does — the cap, the
repetition factor, the value-shape backstop and the per-value length cap all come
from :mod:`dialects`, so a column treated as categorical on a Glue source is
treated that way here. The cap is what makes cost O(rows read) rather than
O(table): without it a 100M-row fact table costs two full scans to learn 25
values, and the consequence is the one ``probe_gate_sql`` documents — a value
rarer than the cap is invisible, so the collected set describes a prefix of the
table rather than all of it.
"""

from __future__ import annotations

import json

import structlog

from coa_sources.database.metrics import emit_metric

from .dialects import (
    MAX_ENUM_DISTINCT,
    MIN_REPETITION_FACTOR,
    PROBE_MAX_ROWS,
    values_look_categorical,
)
from .trino_statement import TrinoStatementRunner, quote_ident, quote_literal

logger = structlog.get_logger(__name__)

# Same prefixes AthenaSampler and JdbcConnector sample on, so a column treated as
# categorical on one source type is treated that way on all of them.
_SAMPLEABLE_TYPE_PREFIXES = ("char", "varchar", "text", "string")

# Ceiling on candidates per statement. A wide schema becomes a few bounded
# statements instead of one whose text grows without limit.
MAX_COLUMNS_PER_QUERY = 60

# Matches AthenaSampler's own truncation. A low-cardinality column of long values
# is already rejected by values_look_categorical; this bounds what a value that
# passes can contribute to the prompt.
_MAX_VALUE_CHARS = 100


def is_sampleable_type(data_type: str) -> bool:
    """Whether a Trino ``data_type`` is a string type worth sampling.

    Matches on the prefix because Trino reports parameterised types verbatim
    (``varchar(25)``), exactly as ``information_schema`` returned them.
    """
    return (data_type or "").strip().lower().startswith(_SAMPLEABLE_TYPE_PREFIXES)


class TrinoSampler:
    """Collect enum-like values for many tables in two queries per chunk.

    Deliberately the same shape as ``AthenaSampler.sample_all`` —
    ``{table: {column: [values]}}`` — so the connectors stay diffable even though
    the query strategy differs.
    """

    def __init__(
        self,
        runner: TrinoStatementRunner,
        *,
        max_distinct: int = MAX_ENUM_DISTINCT,
        max_columns_per_query: int = MAX_COLUMNS_PER_QUERY,
        row_cap: int = PROBE_MAX_ROWS,
    ) -> None:
        """Wire the statement runner and the cardinality/width/row bounds."""
        self._runner = runner
        self._max_distinct = max_distinct
        self._max_columns_per_query = max(1, max_columns_per_query)
        self._row_cap = max(1, row_cap)

    def sample_all(
        self,
        tables: list[tuple[str, list[tuple[str, str]]]],
        *,
        catalog: str,
        schema: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        """Return ``{table: {column: [values]}}`` for the categorical columns.

        ``tables`` is ``[(table_name, [(col_name, data_type), ...]), ...]``, the
        same input ``AthenaSampler.sample_all`` takes.

        Never raises: sampling is a quality improvement, so a failure costs
        literal matching and leaves the scan intact.
        """
        candidates = [(table, [col for col, dtype in cols if is_sampleable_type(dtype)]) for table, cols in tables]
        candidates = [(t, cols) for t, cols in candidates if cols]
        if not candidates:
            return {}

        out: dict[str, dict[str, list[str]]] = {}
        for chunk in self._chunk(candidates):
            try:
                probed = self._probe(chunk, catalog, schema, timeout_seconds)
                small = [(t, [c for c in cols if self._is_enum_like(probed.get(t, {}).get(c))]) for t, cols in chunk]
                small = [(t, cols) for t, cols in small if cols]
                if not small:
                    continue
                collected = self._collect(small, catalog, schema, timeout_seconds)
            except Exception:
                emit_metric("TrinoSamplingFailed", 1, "Count")
                logger.warning(
                    "trino_enum_sampling_failed",
                    catalog=catalog,
                    schema=schema,
                    tables=[t for t, _ in chunk],
                    exc_info=True,
                )
                continue
            for table, cols in collected.items():
                for col, vals in cols.items():
                    values = self._clean(vals)
                    if values:
                        out.setdefault(table, {})[col] = values
        return out

    def _chunk(self, candidates: list[tuple[str, list[str]]]) -> list[list[tuple[str, list[str]]]]:
        """Split candidates so no statement carries more than the ceiling.

        A single table with more candidates than the ceiling is split across
        chunks rather than exempted, so the bound actually holds.
        """
        chunks: list[list[tuple[str, list[str]]]] = []
        current: list[tuple[str, list[str]]] = []
        budget = self._max_columns_per_query
        for table, cols in candidates:
            remaining = list(cols)
            while remaining:
                take, remaining = remaining[:budget], remaining[budget:]
                current.append((table, take))
                budget -= len(take)
                if budget == 0:
                    chunks.append(current)
                    current, budget = [], self._max_columns_per_query
        if current:
            chunks.append(current)
        return chunks

    def _is_enum_like(self, probe: tuple[int, int] | None) -> bool:
        """Whether ``(distinct, non_null_rows)`` describes a category column.

        The repetition factor is what excludes a unique-identifier column, which
        a distinct ceiling alone admits in any table smaller than the ceiling: 25
        rows of 25 distinct order ids would otherwise reach the prompt as
        ``allowed values``. Same gate ``AthenaSampler`` applies.
        """
        if not probe:
            return False
        distinct, non_null = probe
        return 0 < distinct <= self._max_distinct and non_null >= MIN_REPETITION_FACTOR * distinct

    def _clean(self, values: object) -> list[str]:
        """Drop nulls and blanks, truncate, then apply the value-shape backstop.

        Truncation precedes the shape check, as in ``AthenaSampler``: the check's
        own length threshold is well below the cap, so free text still fails it.
        """
        if not isinstance(values, list):
            return []
        cleaned = [str(v)[:_MAX_VALUE_CHARS] for v in values if v is not None and str(v) != ""]
        return cleaned if values_look_categorical(cleaned) else []

    def _scan(self, table: str, cols: list[str]) -> str:
        """A row-capped subquery projecting only ``cols``.

        One scan serves every aggregate over it, which is the whole efficiency of
        this sampler against the per-column fan-out. No ``ORDER BY``: that would
        reintroduce a full sort, and the prefix only has to answer "does this
        column hold a small set of repeated labels".
        """
        projected = ", ".join(quote_ident(c) for c in cols)
        return f"(SELECT {projected} FROM {quote_ident(table)} LIMIT {self._row_cap}) probe_sample"

    def _probe(
        self,
        chunk: list[tuple[str, list[str]]],
        catalog: str,
        schema: str,
        timeout_seconds: float | None,
    ) -> dict[str, dict[str, tuple[int, int]]]:
        """Cardinality and non-null count per column, one statement per chunk."""
        arms = []
        for table, cols in chunk:
            keys = ", ".join(quote_literal(c) for c in cols)
            distincts = ", ".join(f"approx_distinct({quote_ident(c)})" for c in cols)
            # count(c), not count(*): the repetition factor has to compare against
            # this column's non-null values, or a mostly-null column reads as
            # repetitive on the strength of other columns' rows.
            non_nulls = ", ".join(f"count({quote_ident(c)})" for c in cols)
            arms.append(
                f"SELECT {quote_literal(table)} AS tbl, "
                f"cast(map(ARRAY[{keys}], ARRAY[{distincts}]) AS json) AS d, "
                f"cast(map(ARRAY[{keys}], ARRAY[{non_nulls}]) AS json) AS n "
                f"FROM {self._scan(table, cols)}"
            )
        rows = self._runner.run(
            "\nUNION ALL\n".join(arms),
            catalog=catalog,
            schema=schema,
            timeout_seconds=timeout_seconds,
        )
        out: dict[str, dict[str, tuple[int, int]]] = {}
        for row in rows:
            if not row or len(row) < 3 or row[0] is None or row[1] is None or row[2] is None:
                continue
            try:
                distinct_map, non_null_map = json.loads(row[1]), json.loads(row[2])
            except (TypeError, ValueError):
                logger.debug("trino_enum_sampling_unparseable_row", table=row[0])
                continue
            if not isinstance(distinct_map, dict) or not isinstance(non_null_map, dict):
                continue
            out[str(row[0])] = {
                col: (_as_int(distinct_map.get(col)), _as_int(non_null_map.get(col))) for col in distinct_map
            }
        return out

    def _collect(
        self,
        chunk: list[tuple[str, list[str]]],
        catalog: str,
        schema: str,
        timeout_seconds: float | None,
    ) -> dict[str, dict]:
        """Distinct values for the columns that passed, one statement per chunk.

        Reads the same capped prefix the probe did, so the values are the ones its
        counts described rather than a set the gate never saw.
        """
        arms = []
        for table, cols in chunk:
            keys = ", ".join(quote_literal(c) for c in cols)
            aggs = ", ".join(f"array_agg(DISTINCT {quote_ident(c)})" for c in cols)
            # Cast to json so the row stays a string: the statement runner's
            # contract is str|None, and a native MAP would decode per client.
            arms.append(
                f"SELECT {quote_literal(table)} AS tbl, "
                f"cast(map(ARRAY[{keys}], ARRAY[{aggs}]) AS json) AS v "
                f"FROM {self._scan(table, cols)}"
            )
        rows = self._runner.run(
            "\nUNION ALL\n".join(arms),
            catalog=catalog,
            schema=schema,
            timeout_seconds=timeout_seconds,
        )
        parsed: dict[str, dict] = {}
        for row in rows:
            if not row or len(row) < 2 or row[0] is None or row[1] is None:
                continue
            try:
                values = json.loads(row[1])
            except (TypeError, ValueError):
                logger.debug("trino_enum_sampling_unparseable_row", table=row[0])
                continue
            if isinstance(values, dict):
                parsed[str(row[0])] = values
        return parsed


def _as_int(value: object) -> int:
    """Coerce a JSON number to int, treating anything unusable as zero.

    Zero fails the gate, so an unreadable count leaves the column unsampled
    rather than collected on the strength of a value nobody could parse.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
