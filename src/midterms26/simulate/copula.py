"""Joint simulation via Gaussian copula (Phase 4).

Marginals = per-race conformalized predictive CDFs (interpolated from the stacked
quantile grid). Correlation = the Bayesian model's shared latent factors
(national + state), read from ``latent_factors`` — this replaces the old
SHAP-correlation hack. We draw ``N_DRAWS`` correlated elections, invert each race's
marginal, and count seats per draw -> seat distribution, majority probability,
and seat quantiles per office.

The factor covariance ``L L^T + diag(idiosyncratic^2)`` is PSD by construction, so
its correlation has a Cholesky factor for the draw. numpy is imported lazily
(``models`` extra); the seat-counting logic is separated out so it tests without a
GPU/BLAS-heavy path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.logging import get_logger
from midterms26.models.base import load_member_grids, parse_race_id
from midterms26.warehouse import connect, init_schema

STAGE = "simulate.copula"
STACK = "STACK"
N_DRAWS = 50_000
log = get_logger(STAGE)


def inv_cdf(grid: Mapping[float, float], u: float) -> float:
    """Invert a quantile grid at cumulative prob ``u`` (piecewise-linear)."""
    levels = sorted(grid)
    if u <= levels[0]:
        return grid[levels[0]]
    if u >= levels[-1]:
        return grid[levels[-1]]
    for lo, hi in zip(levels, levels[1:], strict=False):
        if lo <= u <= hi:
            v0, v1 = grid[lo], grid[hi]
            if hi == lo:
                return v1
            return v0 + (v1 - v0) * (u - lo) / (hi - lo)
    return grid[levels[-1]]


def correlation_matrix(
    race_ids: Sequence[str],
    loadings: Mapping[str, Mapping[str, float]],
    idiosyncratic: Mapping[str, float],
) -> Any:
    """Build the race correlation matrix from latent-factor loadings.

    ``Cov(i,j) = dot(loadings_i, loadings_j)`` for i != j and
    ``Var(i) = dot(loadings_i, loadings_i) + idiosyncratic_i^2``; correlation is
    the standardized covariance. Returns an ``(n, n)`` numpy array.
    """
    import numpy as np

    n = len(race_ids)
    components = sorted({c for rid in race_ids for c in loadings.get(rid, {})})
    comp_ix = {c: k for k, c in enumerate(components)}
    load = np.zeros((n, len(components)))
    for i, rid in enumerate(race_ids):
        for c, v in loadings.get(rid, {}).items():
            load[i, comp_ix[c]] = v
    cov = load @ load.T
    idio = np.array([idiosyncratic.get(rid, 1.0) ** 2 for rid in race_ids])
    cov = cov + np.diag(idio)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    return np.clip(corr, -1.0, 1.0)


def simulate_office(
    grids: Mapping[str, Mapping[float, float]],
    corr: Any,
    *,
    majority_threshold: int,
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Draw correlated elections and summarize the Dem seat distribution."""
    import numpy as np
    from scipy.stats import norm

    race_ids = list(grids)
    n = len(race_ids)
    rng = np.random.default_rng(seed)
    # Cholesky with a tiny jitter for numerical PSD safety.
    chol = np.linalg.cholesky(corr + 1e-9 * np.eye(n))
    z = chol @ rng.standard_normal((n, n_draws))
    u = norm.cdf(z)

    wins = np.zeros((n, n_draws), dtype=bool)
    for i, rid in enumerate(race_ids):
        grid = grids[rid]
        margins = np.array([inv_cdf(grid, float(uu)) for uu in u[i]])
        wins[i] = margins > 0.0
    dem_seats = wins.sum(axis=0)

    hist_counts = np.bincount(dem_seats, minlength=n + 1)
    histogram = {str(k): float(c) / n_draws for k, c in enumerate(hist_counts) if c > 0}
    return {
        "n_races": n,
        "n_draws": n_draws,
        "majority_threshold": majority_threshold,
        "expected_dem_seats": float(dem_seats.mean()),
        "p_dem_majority": float((dem_seats >= majority_threshold).mean()),
        "seats_p10": float(np.percentile(dem_seats, 10)),
        "seats_p50": float(np.percentile(dem_seats, 50)),
        "seats_p90": float(np.percentile(dem_seats, 90)),
        "histogram": histogram,
    }


def run(ctx: RunContext) -> StepResult:
    """Simulate the joint seat distribution per office and write ``seat_forecast``."""
    if ctx.cutoff_date is None:
        raise ValueError("simulate.copula needs ctx.cutoff_date")
    plan_generation = 0
    as_of = ctx.cutoff_date

    with connect(ctx.db_path) as conn:
        init_schema(conn)
        stack = load_member_grids(conn, as_of, plan_generation, model_member=STACK)
        live = {rid: g for (m, fold, rid), g in stack.items() if fold == "live"}
        loadings, idio = _load_latent(conn, as_of, plan_generation)

        # Group live races by office; simulate each independently.
        by_office: dict[str, dict[str, Mapping[float, float]]] = {}
        for rid, grid in live.items():
            by_office.setdefault(parse_race_id(rid)[1], {})[rid] = grid

        n_written = 0
        for office, grids in by_office.items():
            race_ids = list(grids)
            corr = correlation_matrix(race_ids, loadings, idio)
            threshold = len(race_ids) // 2 + 1
            summary = simulate_office(
                grids, corr, majority_threshold=threshold, n_draws=N_DRAWS, seed=0
            )
            _write_forecast(conn, as_of, plan_generation, office, summary)
            n_written += 1
    log.info("copula.done", offices=n_written, draws=N_DRAWS)
    return StepResult(node=STAGE, rows=n_written, detail=f"{n_written} office seat forecasts")


def _load_latent(
    conn: object, cutoff: date, plan_generation: int
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    rows = conn.execute(  # type: ignore[attr-defined]
        "SELECT race_id, loadings, idiosyncratic_sd FROM latent_factors "
        "WHERE cutoff_date = ? AND plan_generation = ?",
        [cutoff, plan_generation],
    ).fetchall()
    loadings = {rid: json.loads(load) for rid, load, _ in rows}
    idio = {rid: float(sd) for rid, _, sd in rows}
    return loadings, idio


def _write_forecast(
    conn: object, as_of: date, plan_generation: int, office: str, s: Mapping[str, Any]
) -> None:
    conn.execute(  # type: ignore[attr-defined]
        """
        INSERT OR REPLACE INTO seat_forecast
            (as_of, plan_generation, office, n_races, n_draws, majority_threshold,
             expected_dem_seats, p_dem_majority, seats_p10, seats_p50, seats_p90, histogram)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            as_of,
            plan_generation,
            office,
            s["n_races"],
            s["n_draws"],
            s["majority_threshold"],
            s["expected_dem_seats"],
            s["p_dem_majority"],
            s["seats_p10"],
            s["seats_p50"],
            s["seats_p90"],
            json.dumps(s["histogram"]),
        ],
    )


def dry_run(ctx: RunContext) -> StepResult:  # noqa: ARG001
    return StepResult(
        node=STAGE,
        rows=N_DRAWS,
        detail="50k copula draws -> seat dist, P(majority), tipping (stub)",
        dry_run=True,
    )
