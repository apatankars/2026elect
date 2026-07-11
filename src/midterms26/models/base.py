"""Shared model I/O: member input assembly, quantile-grid storage, standardization.

Deliberately pure-Python (polars/duckdb, no numpy) so it imports and unit-tests in
the light CI stack — the heavy members (:mod:`~midterms26.models.bayes`,
:mod:`~midterms26.models.tabpfn_member`) import numpy/numpyro/tabpfn lazily inside
their fit routines and consume the :class:`MemberInput` this module builds.

A ``race_id`` is ``{cycle}-{office}-{state}-{district}``; cycle/state/office are
parsed back out here so the hierarchical member can pool over them without the
grouping keys having to survive as features.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import duckdb

# Canonical predictive quantile levels every member emits. Chosen so the conformal
# layer's three intervals (alpha 0.5/0.2/0.1 -> the 0.25/0.75, 0.10/0.90, 0.05/0.95
# pairs) and the median are all present, with 0.01/0.99 tails for CDF/CRPS work.
QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

LIVE_FOLD = "live"


def parse_race_id(race_id: str) -> tuple[int, str, str, str]:
    """Return ``(cycle, office, state, district)`` from a canonical race id."""
    parts = race_id.split("-", 3)
    if len(parts) != 4:
        raise ValueError(f"malformed race_id {race_id!r}; expected cycle-office-state-district")
    cycle_s, office, state, district = parts
    return int(cycle_s), office, state, district


@dataclass(frozen=True)
class MemberInput:
    """Aligned design for a member fit: features, targets, and grouping keys.

    ``y[i]`` is ``None`` for a live/future race (no realized margin). ``X`` is the
    raw (un-standardized) feature matrix aligned to ``feature_names``; call
    :meth:`standardized` to get model-ready columns.
    """

    race_ids: list[str]
    feature_names: list[str]
    X: list[list[float | None]]
    y: list[float | None]
    cycles: list[int]
    states: list[str]
    offices: list[str]

    def __post_init__(self) -> None:
        n = len(self.race_ids)
        if not all(
            len(seq) == n for seq in (self.X, self.y, self.cycles, self.states, self.offices)
        ):
            raise ValueError("MemberInput columns must all align to race_ids")

    @property
    def n_rows(self) -> int:
        return len(self.race_ids)

    def labeled_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.y) if v is not None]

    def live_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.y) if v is None]

    def labeled_cycles(self) -> list[int]:
        return sorted({self.cycles[i] for i in self.labeled_indices()})

    def standardized(self, ref_indices: Sequence[int] | None = None) -> list[list[float]]:
        """Z-score features using ``ref_indices`` (default: labeled rows) as the base.

        Missing values map to ``0.0`` (the standardized mean), and zero-variance
        columns are left centered so priors stay well-conditioned.
        """
        ref = list(ref_indices) if ref_indices is not None else self.labeled_indices()
        if not ref:
            ref = list(range(self.n_rows))
        means: list[float] = []
        sds: list[float] = []
        for j in range(len(self.feature_names)):
            vals: list[float] = []
            for i in ref:
                v = self.X[i][j]
                if v is not None:
                    vals.append(v)
            mu = sum(vals) / len(vals) if vals else 0.0
            var = sum((v - mu) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
            means.append(mu)
            sds.append(math.sqrt(var) if var > 0 else 1.0)
        out: list[list[float]] = []
        for i in range(self.n_rows):
            row = self.X[i]
            out_row: list[float] = []
            for j in range(len(self.feature_names)):
                v = row[j]
                out_row.append(0.0 if v is None else (v - means[j]) / sds[j])
            out.append(out_row)
        return out


def load_member_input(
    conn: duckdb.DuckDBPyConnection, cutoff_date: date, plan_generation: int = 0
) -> MemberInput:
    """Assemble a :class:`MemberInput` from ``feature_matrix`` at one freeze date."""
    rows = conn.execute(
        """
        SELECT race_id, features, target_margin
        FROM feature_matrix
        WHERE cutoff_date = ? AND plan_generation = ?
        ORDER BY race_id
        """,
        [cutoff_date, plan_generation],
    ).fetchall()
    parsed = [(rid, json.loads(feats), tgt) for rid, feats, tgt in rows]
    names: set[str] = set()
    for _, feats, _ in parsed:
        names.update(feats.keys())
    feature_names = sorted(names)

    race_ids: list[str] = []
    X: list[list[float | None]] = []
    y: list[float | None] = []
    cycles: list[int] = []
    states: list[str] = []
    offices: list[str] = []
    for rid, feats, tgt in parsed:
        cycle, office, state, _district = parse_race_id(rid)
        race_ids.append(rid)
        X.append([_as_float(feats.get(name)) for name in feature_names])
        y.append(None if tgt is None else float(tgt))
        cycles.append(cycle)
        states.append(state)
        offices.append(office)
    return MemberInput(race_ids, feature_names, X, y, cycles, states, offices)


def _as_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return float(v) if isinstance(v, bool) else None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def grid_to_json(grid: Mapping[float, float]) -> str:
    """Serialize a level -> value quantile grid with stable string keys."""
    return json.dumps({f"{lvl:.4f}": grid[lvl] for lvl in sorted(grid)})


def grid_from_json(text: str) -> dict[float, float]:
    """Inverse of :func:`grid_to_json`."""
    return {float(k): float(v) for k, v in json.loads(text).items()}


def load_member_grids(
    conn: duckdb.DuckDBPyConnection,
    cutoff_date: date,
    plan_generation: int = 0,
    model_member: str | None = None,
) -> dict[tuple[str, str, str], dict[float, float]]:
    """Read member grids as ``{(model_member, fold, race_id): grid}``."""
    sql = (
        "SELECT model_member, fold, race_id, quantiles FROM member_predictions "
        "WHERE cutoff_date = ? AND plan_generation = ?"
    )
    params: list[Any] = [cutoff_date, plan_generation]
    if model_member is not None:
        sql += " AND model_member = ?"
        params.append(model_member)
    return {
        (m, fold, rid): grid_from_json(q)
        for m, fold, rid, q in conn.execute(sql, params).fetchall()
    }


def load_targets(
    conn: duckdb.DuckDBPyConnection, cutoff_date: date, plan_generation: int = 0
) -> dict[str, float]:
    """Read realized ``target_margin`` per race (non-null only) from feature_matrix."""
    rows = conn.execute(
        "SELECT race_id, target_margin FROM feature_matrix "
        "WHERE cutoff_date = ? AND plan_generation = ? AND target_margin IS NOT NULL",
        [cutoff_date, plan_generation],
    ).fetchall()
    return {rid: float(t) for rid, t in rows}


def write_member_predictions(
    conn: duckdb.DuckDBPyConnection,
    *,
    cutoff_date: date,
    plan_generation: int,
    model_member: str,
    fold: str,
    grids: Mapping[str, Mapping[float, float]],
) -> int:
    """Upsert one member's quantile grids for ``fold`` (``grids`` keyed by race_id)."""
    payload = [
        (
            rid,
            cutoff_date,
            plan_generation,
            model_member,
            fold,
            grid.get(0.5),
            grid_to_json(grid),
        )
        for rid, grid in grids.items()
    ]
    if not payload:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO member_predictions
            (race_id, cutoff_date, plan_generation, model_member, fold, median_margin, quantiles)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def write_latent_factors(
    conn: duckdb.DuckDBPyConnection,
    *,
    cutoff_date: date,
    plan_generation: int,
    loadings: Mapping[str, Mapping[str, float]],
    idiosyncratic_sd: Mapping[str, float],
) -> int:
    """Upsert per-race latent-factor loadings + idiosyncratic sd for the copula."""
    payload = [
        (
            rid,
            cutoff_date,
            plan_generation,
            json.dumps(dict(load)),
            float(idiosyncratic_sd[rid]),
        )
        for rid, load in loadings.items()
    ]
    if not payload:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO latent_factors
            (race_id, cutoff_date, plan_generation, loadings, idiosyncratic_sd)
        VALUES (?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)
