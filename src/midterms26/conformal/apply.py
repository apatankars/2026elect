"""Conformal layer DAG node — composes CQR + weighted + Mondrian + abstain.

Consumes the stacked quantile grids (``member_predictions`` where
``model_member='STACK'``) and the LOCO folds' realized margins, and emits, per
live race: the median, conformal intervals at alpha in {0.5, 0.2, 0.1}, the
Mondrian group id, an abstain flag + reason, and the Dem win probability from the
predictive CDF.

Calibration is **group-conditional**: nonconformity scores are pooled within each
Mondrian bin (so coverage holds per stratum), and taken with cycle-decay weights
(recent cycles count more — the nonexchangeable correction). A race abstains when
its 80% interval is wider than ``TAU_WIDTH_80`` or its bin has fewer than
``N_MIN`` calibration points. Those two thresholds are demo defaults here and get
frozen in ``docs/PREREGISTRATION.md`` after backtesting.

Pure-Python; runs and tests in the light stack.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import cast

from midterms26.conformal.abstain import decide
from midterms26.conformal.cqr import calibrate, interval, nonconformity_scores
from midterms26.conformal.mondrian import group_id
from midterms26.conformal.weighted import exp_decay_weights
from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.logging import get_logger
from midterms26.models.base import load_member_grids, load_targets, parse_race_id
from midterms26.warehouse import connect, init_schema

STAGE = "conformal.apply"
STACK = "STACK"
log = get_logger(STAGE)

# alpha -> (lower quantile level, upper quantile level) drawn from the grid.
ALPHA_LEVELS: tuple[tuple[float, float, float], ...] = (
    (0.5, 0.25, 0.75),
    (0.2, 0.10, 0.90),
    (0.1, 0.05, 0.95),
)
_ALPHA_COLUMNS = {0.5: ("lo_50", "hi_50"), 0.2: ("lo_80", "hi_80"), 0.1: ("lo_90", "hi_90")}

# Demo abstention thresholds (frozen in PREREGISTRATION.md after backtesting).
TAU_WIDTH_80 = 60.0
N_MIN = 5

MODEL_VERSION = "0.0.1"


def cdf_at(grid: Mapping[float, float], x: float) -> float:
    """Interpolate the predictive CDF ``P(margin <= x)`` from a quantile grid."""
    levels = sorted(grid)
    pts = [(grid[lvl], lvl) for lvl in levels]  # (value, cumulative prob)
    if x <= pts[0][0]:
        return 0.0
    if x >= pts[-1][0]:
        return 1.0
    for (v0, p0), (v1, p1) in zip(pts, pts[1:], strict=False):
        if v0 <= x <= v1:
            if v1 == v0:
                return p1
            return p0 + (p1 - p0) * (x - v0) / (v1 - v0)
    return 1.0


def win_prob_dem(grid: Mapping[float, float]) -> float:
    """P(margin > 0) — Dem win probability — from the predictive CDF."""
    return max(0.0, min(1.0, 1.0 - cdf_at(grid, 0.0)))


def mondrian_bin(
    incumbent_party: str | None, n_polls: float, is_new_seat: bool, office: str
) -> str:
    """Map a race to its Mondrian group id across the four canonical axes."""
    inc = {"D": "incD", "R": "incR"}.get(incumbent_party or "", "open")
    polls = "polled3+" if n_polls >= 3 else "polled1-2" if n_polls >= 1 else "unpolled"
    lines = "redrawn" if is_new_seat else "stable"
    off = office if office in ("HOUSE", "SENATE", "GOV") else "HOUSE"
    return group_id(inc, polls, lines, off)


def _bin_from_meta(info: Mapping[str, object]) -> str:
    """Mondrian bin for one race's ``_load_race_meta`` record (values typed ``object``)."""
    return mondrian_bin(
        cast("str | None", info.get("incumbent_party")),
        cast("float", info.get("n_polls", 0.0)),
        cast("bool", info.get("is_new_seat", False)),
        cast("str", info.get("office", "HOUSE")),
    )


def _calibration_sets(
    loco_grids: Mapping[tuple[str, str], Mapping[float, float]],
    targets: Mapping[str, float],
    bin_of: Mapping[str, str],
) -> dict[str, dict[float, tuple[list[float], list[float], list[float], list[int]]]]:
    """Per (bin, alpha): (q_lo list, q_hi list, y list, cycle list) over LOCO races."""
    out: dict[str, dict[float, tuple[list[float], list[float], list[float], list[int]]]] = (
        defaultdict(lambda: {a: ([], [], [], []) for a, _, _ in ALPHA_LEVELS})
    )
    for (fold, rid), grid in loco_grids.items():
        if rid not in targets or rid not in bin_of:
            continue
        cycle = int(fold)
        y = targets[rid]
        for alpha, lo_lvl, hi_lvl in ALPHA_LEVELS:
            qlo, qhi, ys, cyc = out[bin_of[rid]][alpha]
            qlo.append(grid[lo_lvl])
            qhi.append(grid[hi_lvl])
            ys.append(y)
            cyc.append(cycle)
    return out


def run(ctx: RunContext) -> StepResult:
    """Calibrate per Mondrian bin and write conformalized predictions for live races."""
    if ctx.cutoff_date is None:
        raise ValueError("conformal.apply needs ctx.cutoff_date")
    plan_generation = 0
    as_of = ctx.cutoff_date

    with connect(ctx.db_path) as conn:
        init_schema(conn)
        stack = load_member_grids(conn, as_of, plan_generation, model_member=STACK)
        targets = load_targets(conn, as_of, plan_generation)
        meta = _load_race_meta(conn, as_of, plan_generation)

        live_grids = {rid: g for (m, fold, rid), g in stack.items() if fold == "live"}
        loco_grids = {(fold, rid): g for (m, fold, rid), g in stack.items() if fold != "live"}

        bin_of = {rid: _bin_from_meta(info) for rid, info in meta.items()}
        cal = _calibration_sets(loco_grids, targets, bin_of)

        rows = []
        for rid, grid in live_grids.items():
            grp = bin_of.get(rid, _bin_from_meta(meta.get(rid, {})))
            per_alpha = cal.get(grp, {})
            intervals: dict[float, tuple[float, float]] = {}
            bin_n = 0
            for alpha, lo_lvl, hi_lvl in ALPHA_LEVELS:
                qlo, qhi, ys, cyc = per_alpha.get(alpha, ([], [], [], []))
                bin_n = len(ys)
                if ys:
                    scores = nonconformity_scores(ys, qlo, qhi)
                    weights = exp_decay_weights(cyc, reference_cycle=max(cyc))
                    adj = calibrate(scores, alpha, weights=weights)
                else:
                    adj = float("inf")  # no calibration -> unbounded -> abstain
                iv = interval(grid[lo_lvl], grid[hi_lvl], adj)
                intervals[alpha] = (iv.lo, iv.hi)

            lo80, hi80 = intervals[0.2]
            width80 = hi80 - lo80
            dec = decide(width80, bin_n, tau=TAU_WIDTH_80, n_min=N_MIN)
            rows.append(_prediction_row(rid, as_of, plan_generation, grid, intervals, grp, dec))

        n = _write_predictions(conn, rows)
    log.info("conformal.done", n_predictions=n)
    return StepResult(node=STAGE, rows=n, detail=f"{n} conformalized predictions")


def _load_race_meta(
    conn: object, cutoff: date, plan_generation: int
) -> dict[str, dict[str, object]]:
    ex = conn.execute  # type: ignore[attr-defined]
    races = {
        rid: {"incumbent_party": ip, "office": parse_race_id(rid)[1]}
        for rid, ip in ex("SELECT race_id, incumbent_party FROM races").fetchall()
    }
    for rid, is_new in ex(
        "SELECT race_id, is_new_seat FROM districts_geo WHERE plan_generation = ?",
        [plan_generation],
    ).fetchall():
        races.setdefault(rid, {})["is_new_seat"] = bool(is_new)
    # n_polls from the assembled feature JSON.
    import json

    for rid, feats in ex(
        "SELECT race_id, features FROM feature_matrix WHERE cutoff_date = ? AND plan_generation = ?",
        [cutoff, plan_generation],
    ).fetchall():
        races.setdefault(rid, {})["n_polls"] = float(json.loads(feats).get("n_polls", 0.0))
    for info in races.values():
        info.setdefault("incumbent_party", None)
        info.setdefault("office", "HOUSE")
        info.setdefault("is_new_seat", False)
        info.setdefault("n_polls", 0.0)
    return races


def _prediction_row(
    rid: str,
    as_of: date,
    plan_generation: int,
    grid: Mapping[float, float],
    intervals: Mapping[float, tuple[float, float]],
    group: str,
    dec: object,
) -> tuple[object, ...]:
    lo50, hi50 = intervals[0.5]
    lo80, hi80 = intervals[0.2]
    lo90, hi90 = intervals[0.1]
    return (
        rid,
        as_of,
        plan_generation,
        MODEL_VERSION,
        grid[0.5],
        lo50,
        hi50,
        lo80,
        hi80,
        lo90,
        hi90,
        win_prob_dem(grid),
        group,
        dec.abstain,  # type: ignore[attr-defined]
        dec.reason,  # type: ignore[attr-defined]
    )


def _write_predictions(conn: object, rows: Sequence[tuple[object, ...]]) -> int:
    if not rows:
        return 0
    # Sanitize non-finite bounds (unbounded abstain intervals) to NULL for storage.
    clean = [tuple(None if _is_inf(v) else v for v in row) for row in rows]
    conn.executemany(  # type: ignore[attr-defined]
        """
        INSERT OR REPLACE INTO predictions
            (race_id, as_of, plan_generation, model_version, median_margin,
             lo_50, hi_50, lo_80, hi_80, lo_90, hi_90, win_prob_dem,
             mondrian_group, abstain, abstain_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        clean,
    )
    return len(clean)


def _is_inf(v: object) -> bool:
    return isinstance(v, float) and (v == float("inf") or v == float("-inf"))


def dry_run(ctx: RunContext) -> StepResult:  # noqa: ARG001
    return StepResult(
        node=STAGE,
        rows=3_600,
        detail="CQR+weighted+Mondrian intervals, abstain flags (stub)",
        dry_run=True,
    )
