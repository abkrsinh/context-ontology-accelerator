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
column, which is what makes it affordable here at all — this connector's
discovery is deliberately a fixed number of queries regardless of schema size.

Two passes rather than one, because ``array_agg(DISTINCT col)`` accumulates every
value it encounters: ``approx_distinct`` is the cheap probe that decides which
columns are small enough to collect.
"""

from __future__ import annotations

import json

import structlog

from coa_sources.database.metrics import emit_metric

from .dialects import MAX_ENUM_DISTINCT
from .trino_statement import TrinoStatementRunner, quote_ident, quote_literal

logger = structlog.get_logger(__name__)

# Same prefixes AthenaSampler and JdbcConnector sample on, so a column treated as
# categorical on one source type is treated that way on all of them.
_SAMPLEABLE_TYPE_PREFIXES = ("char", "varchar", "text", "string")

# Ceiling on candidates per statement. A wide schema becomes a few bounded
# statements instead of one whose text grows without limit.
MAX_COLUMNS_PER_QUERY = 60


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
    ) -> None:
        """Wire the statement runner and the cardinality/width bounds."""
        self._runner = runner
        self._max_distinct = max_distinct
        self._max_columns_per_query = max(1, max_columns_per_query)

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
                counts = self._run_pass(chunk, "approx_distinct", catalog, schema, timeout_seconds)
                small = [
                    (table, [c for c in cols if 0 < int(counts.get(table, {}).get(c, -1) or -1) <= self._max_distinct])
                    for table, cols in chunk
                ]
                small = [(t, cols) for t, cols in small if cols]
                if not small:
                    continue
                values = self._run_pass(small, "array_agg_distinct", catalog, schema, timeout_seconds)
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
            for table, cols in values.items():
                bucket = out.setdefault(table, {})
                for col, vals in cols.items():
                    if isinstance(vals, list):
                        bucket[col] = [str(v) for v in vals if v is not None]
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

    def _run_pass(
        self,
        chunk: list[tuple[str, list[str]]],
        mode: str,
        catalog: str,
        schema: str,
        timeout_seconds: float | None,
    ) -> dict[str, dict]:
        """Issue one UNION ALL statement for ``chunk`` and parse its JSON maps."""
        arms = []
        for table, cols in chunk:
            keys = ", ".join(quote_literal(c) for c in cols)
            if mode == "approx_distinct":
                aggs = ", ".join(f"approx_distinct({quote_ident(c)})" for c in cols)
            else:
                aggs = ", ".join(f"array_agg(DISTINCT {quote_ident(c)})" for c in cols)
            # Cast to json so the row stays a string: the statement runner's
            # contract is str|None, and a native MAP would decode per client.
            arms.append(
                f"SELECT {quote_literal(table)} AS tbl, "
                f"cast(map(ARRAY[{keys}], ARRAY[{aggs}]) AS json) AS v "
                f"FROM {quote_ident(table)}"
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
                parsed[str(row[0])] = json.loads(row[1])
            except (TypeError, ValueError):
                logger.debug("trino_enum_sampling_unparseable_row", table=row[0])
        return parsed
