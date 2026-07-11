"""TabPFN v2 second learner — Phase 4.

Tabular foundation model, in-context learning, no hyperparameter tuning; our
~3-4k race-cycle x ~100 feature dataset sits in its sweet spot. Emits a full
predictive quantile grid per race. Captures nonlinear interactions the Bayesian
linear predictor misses.

Wrapped in a versioned adapter with a **fallback member** (per METHODOLOGY §4:
NGBoost / quantile forest, not LightGBM). The fallback is a scikit-learn
quantile Gradient Boosting regressor — one model per quantile level, rearranged to
a monotone grid. It exists for real deployment reasons: TabPFN downloads a model
checkpoint on first use and caps training rows, so a nightly job must degrade
gracefully when the checkpoint is unavailable or the data is out of range. Backend
selection is explicit (``auto`` tries TabPFN then falls back).

Requires the ``models`` extra; imported lazily so the package stays importable and
testable in the light CI stack.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.logging import get_logger
from midterms26.models.base import (
    QUANTILE_LEVELS,
    MemberInput,
    load_member_input,
    write_member_predictions,
)
from midterms26.warehouse import connect, init_schema

STAGE = "models.tabpfn"
MODEL_MEMBER = "TABPFN"
log = get_logger(STAGE)

# Deployment knob: force the fallback member (e.g. when the TabPFN checkpoint is
# unavailable in CI / the nightly runner) via MIDTERMS26_TABPFN_BACKEND=sklearn.
_BACKEND_ENV = "MIDTERMS26_TABPFN_BACKEND"


def _monotone_grid(levels: Sequence[float], values: Sequence[float]) -> dict[float, float]:
    """Assemble a level -> value grid, sorting values to prevent quantile crossing."""
    ordered = sorted(values)
    return {float(lvl): float(v) for lvl, v in zip(sorted(levels), ordered, strict=True)}


def _predict_tabpfn(
    x_tr: Any, y_tr: Any, x_pr: Any, levels: Sequence[float], seed: int
) -> list[list[float]]:
    """Per-row quantile predictions from TabPFN. Returns rows x levels."""
    import numpy as np
    from tabpfn import TabPFNRegressor

    reg = TabPFNRegressor(random_state=seed)
    reg.fit(x_tr, y_tr)
    out = reg.predict(x_pr, output_type="quantiles", quantiles=list(levels))
    # TabPFN returns a list (per quantile) of length-n arrays; transpose to rows.
    arr = np.asarray(out, dtype=float)
    if arr.shape[0] == len(levels):
        arr = arr.T
    return [list(map(float, row)) for row in arr]


def _predict_sklearn(
    x_tr: Any, y_tr: Any, x_pr: Any, levels: Sequence[float], seed: int
) -> list[list[float]]:
    """Fallback: one quantile GBR per level. Returns rows x levels."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor

    cols: list[Any] = []
    for lvl in levels:
        gbr = GradientBoostingRegressor(
            loss="quantile", alpha=float(lvl), n_estimators=100, max_depth=3, random_state=seed
        )
        gbr.fit(x_tr, y_tr)
        cols.append(gbr.predict(x_pr))
    mat = np.column_stack(cols) if cols else np.empty((len(x_pr), 0))
    return [list(map(float, row)) for row in mat]


def fit_predict(
    mi: MemberInput,
    *,
    target_cycle: int | None = None,
    quantile_levels: Sequence[float] = QUANTILE_LEVELS,
    backend: str = "auto",
    seed: int = 0,
) -> dict[str, dict[float, float]]:
    """Fit on training rows and predict the target rows' quantile grids.

    ``target_cycle=None`` predicts live rows (train on all labeled); ``=c`` is the
    LOCO fold (train on cycles != c, predict c). ``backend`` is ``auto`` (TabPFN,
    then fallback on any failure), ``tabpfn``, or ``sklearn``.
    """
    import numpy as np

    labeled = mi.labeled_indices()
    if target_cycle is None:
        train_idx, pred_idx = labeled, mi.live_indices()
    else:
        train_idx = [i for i in labeled if mi.cycles[i] != target_cycle]
        pred_idx = [i for i in labeled if mi.cycles[i] == target_cycle]
    if not train_idx:
        raise ValueError("no training rows for the TabPFN member")
    if not pred_idx:
        return {}

    x_std = mi.standardized(ref_indices=train_idx)
    p = len(mi.feature_names)
    x_tr = np.asarray([x_std[i] for i in train_idx], dtype=float).reshape(len(train_idx), p)
    y_tr = np.asarray([mi.y[i] for i in train_idx], dtype=float)
    x_pr = np.asarray([x_std[i] for i in pred_idx], dtype=float).reshape(len(pred_idx), p)
    levels = list(quantile_levels)

    rows = _predict(x_tr, y_tr, x_pr, levels, backend, seed)
    return {mi.race_ids[i]: _monotone_grid(levels, rows[k]) for k, i in enumerate(pred_idx)}


def _predict(
    x_tr: Any, y_tr: Any, x_pr: Any, levels: Sequence[float], backend: str, seed: int
) -> list[list[float]]:
    if backend == "sklearn":
        return _predict_sklearn(x_tr, y_tr, x_pr, levels, seed)
    if backend == "tabpfn":
        return _predict_tabpfn(x_tr, y_tr, x_pr, levels, seed)
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r}; use auto|tabpfn|sklearn")
    try:
        return _predict_tabpfn(x_tr, y_tr, x_pr, levels, seed)
    except Exception as exc:  # noqa: BLE001 — deployment: degrade to the fallback
        log.warning("tabpfn.fallback", error=str(exc)[:200])
        return _predict_sklearn(x_tr, y_tr, x_pr, levels, seed)


def run(ctx: RunContext) -> StepResult:
    """Fit the TabPFN (or fallback) member and write its quantile grids."""
    if ctx.cutoff_date is None:
        raise ValueError("models.tabpfn needs ctx.cutoff_date")
    plan_generation = 0
    backend = os.getenv(_BACKEND_ENV, "auto")
    total = 0
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        mi = load_member_input(conn, ctx.cutoff_date, plan_generation)
        if not mi.labeled_indices():
            raise ValueError("feature_matrix has no labeled rows; run features first")

        total += write_member_predictions(
            conn,
            cutoff_date=ctx.cutoff_date,
            plan_generation=plan_generation,
            model_member=MODEL_MEMBER,
            fold="live",
            grids=fit_predict(mi, target_cycle=None, backend=backend),
        )
        if ctx.do_loco:
            for cycle in mi.labeled_cycles():
                total += write_member_predictions(
                    conn,
                    cutoff_date=ctx.cutoff_date,
                    plan_generation=plan_generation,
                    model_member=MODEL_MEMBER,
                    fold=str(cycle),
                    grids=fit_predict(mi, target_cycle=cycle, backend=backend),
                )
    log.info("tabpfn.done", n_grids=total, do_loco=ctx.do_loco)
    return StepResult(node=STAGE, rows=total, detail=f"{total} TabPFN member grids")


def dry_run(ctx: RunContext) -> StepResult:  # noqa: ARG001
    return StepResult(
        node=STAGE, rows=3_600, detail="TabPFN predictive quantile grid (stub)", dry_run=True
    )
