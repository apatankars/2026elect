"""Cycle-weighted linear stack of the two members — Phase 4.

Leave-one-cycle-out (LOCO) out-of-fold quantile predictions from both members ->
non-negative weighted average per quantile, weights fit by pinball loss and
exponentially cycle-weighted (2022 counts more than 2006). Output quantile grid
feeds the conformal layer (which uses CQR, not plain residual scores).

The estimator here is pure-Python (no numpy) so it runs in light CI and stays
auditable. With two members the per-quantile simplex is the unit interval, and
pinball loss is convex piecewise-linear in the mixing weight, so a ternary search
finds the exact optimum. Stacked quantiles are rearranged (sorted) to guarantee
monotonicity — a Chernozhukov–Fernández-Val–Galichon rearrangement that undoes
any quantile crossing introduced by per-level weights.

The DAG ``run`` (which reads both members' quantile grids from the warehouse)
lands with the Phase 4 member integration; the numerical core below is complete
and tested now.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from midterms26.conformal.weighted import exp_decay_weights
from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.logging import get_logger
from midterms26.models.base import (
    QUANTILE_LEVELS,
    load_member_grids,
    load_targets,
    write_member_predictions,
)
from midterms26.warehouse import connect, init_schema

STAGE = "models.stack"
MODEL_MEMBER = "STACK"
BAYES = "BAYES"
TABPFN = "TABPFN"
log = get_logger(STAGE)

# A quantile grid for one race: level (e.g. 0.05) -> predicted margin.
QuantileGrid = Mapping[float, float]


def pinball_loss(y: float, q_hat: float, tau: float) -> float:
    """Pinball (quantile) loss at level ``tau`` for a single observation."""
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1); got {tau}")
    delta = y - q_hat
    return tau * delta if delta >= 0 else (tau - 1.0) * delta


def weighted_pinball(
    y: Sequence[float],
    q_hat: Sequence[float],
    tau: float,
    sample_weights: Sequence[float] | None = None,
) -> float:
    """Sample-weighted mean pinball loss over a calibration set."""
    n = len(y)
    if len(q_hat) != n:
        raise ValueError("y and q_hat must be the same length")
    if n == 0:
        return 0.0
    w = [1.0] * n if sample_weights is None else [float(x) for x in sample_weights]
    if len(w) != n:
        raise ValueError("sample_weights length must match y")
    total = sum(w)
    if total <= 0:
        raise ValueError("sample_weights must sum to a positive value")
    return (
        sum(wi * pinball_loss(yi, qi, tau) for yi, qi, wi in zip(y, q_hat, w, strict=True)) / total
    )


def fit_stack_weight(
    y: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
    tau: float,
    *,
    sample_weights: Sequence[float] | None = None,
    tol: float = 1e-6,
) -> float:
    """Weight ``w`` in [0,1] minimizing weighted pinball of ``w*a + (1-w)*b``.

    ``a`` is member A's (Bayes) tau-quantile per calibration race, ``b`` member
    B's (TabPFN). The mixed prediction is ``w*a + (1-w)*b``; pinball loss is convex
    in ``w`` so a ternary search converges to the global optimum.
    """
    n = len(y)
    if not (len(a) == len(b) == n):
        raise ValueError("y, a, b must be the same length")

    def loss(w: float) -> float:
        mixed = [w * ai + (1.0 - w) * bi for ai, bi in zip(a, b, strict=True)]
        return weighted_pinball(y, mixed, tau, sample_weights)

    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if loss(m1) < loss(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2.0


def fit_stack(
    y: Sequence[float],
    member_a: Mapping[float, Sequence[float]],
    member_b: Mapping[float, Sequence[float]],
    *,
    sample_weights: Sequence[float] | None = None,
) -> dict[float, float]:
    """Fit a per-quantile mixing weight for every level in the members' grids.

    ``member_a[tau]`` / ``member_b[tau]`` are the LOCO out-of-fold predictions at
    level ``tau`` aligned with ``y``. Returns ``{tau: w}`` — member A's weight per
    level (member B's is ``1 - w``).
    """
    if set(member_a) != set(member_b):
        raise ValueError("members must expose the same quantile levels")
    return {
        tau: fit_stack_weight(y, member_a[tau], member_b[tau], tau, sample_weights=sample_weights)
        for tau in sorted(member_a)
    }


def apply_stack(
    weights: Mapping[float, float],
    a_row: QuantileGrid,
    b_row: QuantileGrid,
) -> dict[float, float]:
    """Blend one race's two member grids and rearrange to a monotone grid.

    Rearrangement: sort the blended values ascending and reassign to the sorted
    levels, so the returned grid is non-crossing regardless of per-level weights.
    """
    levels = sorted(weights)
    if not (set(a_row) >= set(levels) and set(b_row) >= set(levels)):
        raise ValueError("member rows must cover every weighted quantile level")
    blended = [weights[t] * a_row[t] + (1.0 - weights[t]) * b_row[t] for t in levels]
    blended.sort()  # Chernozhukov et al. rearrangement -> monotone quantiles
    return dict(zip(levels, blended, strict=True))


def fit_weights_from_loco(
    grids: Mapping[tuple[str, str, str], Mapping[float, float]],
    targets: Mapping[str, float],
    *,
    levels: Sequence[float] = QUANTILE_LEVELS,
) -> dict[float, float]:
    """Fit per-quantile mixing weights on the LOCO folds (cycle-weighted pinball).

    Only races whose fold is a held-out cycle (not ``live``) and that have both
    members' grids + a realized target contribute. Falls back to an equal 0.5
    blend per level when no LOCO evidence is available (e.g. a live-only run).
    """
    y: list[float] = []
    a: dict[float, list[float]] = defaultdict(list)
    b: dict[float, list[float]] = defaultdict(list)
    w: list[float] = []
    ref_cycle: int | None = None
    for (member, fold, rid), grid in grids.items():
        if member != BAYES or fold == "live" or rid not in targets:
            continue
        other = grids.get((TABPFN, fold, rid))
        if other is None:
            continue
        cycle = int(fold)
        ref_cycle = cycle if ref_cycle is None else max(ref_cycle, cycle)
        y.append(targets[rid])
        w.append(cycle)  # placeholder; converted to decay weights below
        for lvl in levels:
            a[lvl].append(grid[lvl])
            b[lvl].append(other[lvl])
    if not y:
        return {float(lvl): 0.5 for lvl in levels}
    sample_weights = exp_decay_weights([int(c) for c in w], reference_cycle=ref_cycle)
    return {
        float(lvl): fit_stack_weight(y, a[lvl], b[lvl], lvl, sample_weights=sample_weights)
        for lvl in levels
    }


def run(ctx: RunContext) -> StepResult:
    """Fit stack weights on LOCO folds and write blended STACK grids for every fold."""
    if ctx.cutoff_date is None:
        raise ValueError("models.stack needs ctx.cutoff_date")
    plan_generation = 0
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        grids = load_member_grids(conn, ctx.cutoff_date, plan_generation)
        targets = load_targets(conn, ctx.cutoff_date, plan_generation)
        weights = fit_weights_from_loco(grids, targets)

        # Group member grids by (fold, race_id); blend where both members present.
        folds: dict[tuple[str, str], dict[str, Mapping[float, float]]] = defaultdict(dict)
        for (member, fold, rid), grid in grids.items():
            if member in (BAYES, TABPFN):
                folds[(fold, rid)][member] = grid

        by_fold: dict[str, dict[str, dict[float, float]]] = defaultdict(dict)
        for (fold, rid), members in folds.items():
            if BAYES in members and TABPFN in members:
                by_fold[fold][rid] = apply_stack(weights, members[BAYES], members[TABPFN])

        total = 0
        for fold, race_grids in by_fold.items():
            total += write_member_predictions(
                conn,
                cutoff_date=ctx.cutoff_date,
                plan_generation=plan_generation,
                model_member=MODEL_MEMBER,
                fold=fold,
                grids=race_grids,
            )
    log.info("stack.done", n_grids=total, weight_median=weights.get(0.5))
    return StepResult(node=STAGE, rows=total, detail=f"{total} stacked grids")


def dry_run(ctx: RunContext) -> StepResult:  # noqa: ARG001
    return StepResult(
        node=STAGE,
        rows=3_600,
        detail="LOCO-stacked quantile grid, pinball-weighted (stub)",
        dry_run=True,
    )
