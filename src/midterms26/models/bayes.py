"""Hierarchical Bayesian fundamentals model (NumPyro) — Phase 4.

``margin_i ~ Normal(mu_i, sigma)`` with ``mu_i`` = national latent (partially
pooled per cycle) + state random effect + ``beta . features``. Fit with NUTS;
nightly refits are fine at this data size.

Leakage-safe cross-cycle prediction is the crux: a race in the target cycle (a
held-out LOCO cycle, or live 2026) has **no** outcome data, so its national
environment is drawn fresh from the ``Normal(mu_nat, tau_nat)`` hyperprior and an
unseen state's effect from ``Normal(0, tau_state)`` — the target cycle can never
peek at its own margins. That is exactly why unpolled / novel races get wide,
honest posteriors rather than falsely confident ones.

Emits, per race: a predictive quantile grid over
:data:`~midterms26.models.base.QUANTILE_LEVELS`, plus the shared latent-factor
loadings (national + per-state) and idiosyncratic sd that yield the analytic
race-correlation matrix the copula simulator consumes.

Requires the ``models`` extra (numpyro/jax); imported lazily so the module stays
importable — and the rest of the package testable — in the light CI stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.logging import get_logger
from midterms26.models.base import (
    QUANTILE_LEVELS,
    MemberInput,
    load_member_input,
    write_latent_factors,
    write_member_predictions,
)
from midterms26.warehouse import connect, init_schema

STAGE = "models.bayes"
MODEL_MEMBER = "BAYES"
log = get_logger(STAGE)

NATIONAL = "national"


@dataclass
class MemberFit:
    """One member fit: predictive grids for the target rows + copula loadings."""

    grids: dict[str, dict[float, float]]
    loadings: dict[str, dict[str, float]]
    idiosyncratic_sd: dict[str, float]


def fit_predict(
    mi: MemberInput,
    *,
    target_cycle: int | None = None,
    quantile_levels: Sequence[float] = QUANTILE_LEVELS,
    num_warmup: int = 400,
    num_samples: int = 400,
    num_chains: int = 1,
    seed: int = 0,
) -> MemberFit:
    """Fit on training rows and predict the target rows (leakage-safe).

    ``target_cycle=None`` predicts the live rows (``y is None``) using every
    labeled row for training; ``target_cycle=c`` is the LOCO fold — train on all
    labeled cycles except ``c`` and predict ``c``'s rows out-of-fold.
    """
    import jax
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(num_chains)

    labeled = mi.labeled_indices()
    if target_cycle is None:
        train_idx = labeled
        pred_idx = mi.live_indices()
    else:
        train_idx = [i for i in labeled if mi.cycles[i] != target_cycle]
        pred_idx = [i for i in labeled if mi.cycles[i] == target_cycle]
    if not train_idx:
        raise ValueError("no training rows for the Bayes member")
    if not pred_idx:
        return MemberFit({}, {}, {})

    x_std = mi.standardized(ref_indices=train_idx)
    p = len(mi.feature_names)

    train_cycles = sorted({mi.cycles[i] for i in train_idx})
    train_states = sorted({mi.states[i] for i in train_idx})
    cyc_ix = {c: k for k, c in enumerate(train_cycles)}
    st_ix = {s: k for k, s in enumerate(train_states)}

    x_tr = np.asarray([x_std[i] for i in train_idx], dtype=float).reshape(len(train_idx), p)
    y_tr = np.asarray([mi.y[i] for i in train_idx], dtype=float)
    cyc_tr = np.asarray([cyc_ix[mi.cycles[i]] for i in train_idx])
    st_tr = np.asarray([st_ix[mi.states[i]] for i in train_idx])

    def model(x: Any, cyc: Any, st: Any, y: Any = None) -> None:
        mu_nat = numpyro.sample("mu_nat", dist.Normal(0.0, 10.0))
        tau_nat = numpyro.sample("tau_nat", dist.HalfNormal(10.0))
        with numpyro.plate("cycles", len(train_cycles)):
            alpha = numpyro.sample("alpha", dist.Normal(mu_nat, tau_nat))
        tau_state = numpyro.sample("tau_state", dist.HalfNormal(10.0))
        with numpyro.plate("states", len(train_states)):
            u_state = numpyro.sample("u_state", dist.Normal(0.0, tau_state))
        sigma = numpyro.sample("sigma", dist.HalfNormal(20.0))
        mean = alpha[cyc] + u_state[st]
        if p:
            beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([p]).to_event(1))
            mean = mean + x @ beta
        numpyro.sample("y", dist.Normal(mean, sigma), obs=y)

    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), x=x_tr, cyc=cyc_tr, st=st_tr, y=y_tr)
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    s = post["sigma"].shape[0]
    rng = np.random.default_rng(seed + 1)

    grids: dict[str, dict[float, float]] = {}
    loadings: dict[str, dict[str, float]] = {}
    idio: dict[str, float] = {}
    # National component: a fresh draw from the hyperprior per posterior sample —
    # shared across every target race (the unknown target-cycle environment).
    alpha_pred = post["mu_nat"] + post["tau_nat"] * rng.standard_normal(s)
    national_sd = float(np.std(alpha_pred))
    idio_sd = float(np.mean(post["sigma"]))
    levels = list(quantile_levels)

    for i in pred_idx:
        rid = mi.race_ids[i]
        state = mi.states[i]
        if state in st_ix:
            u_pred = post["u_state"][:, st_ix[state]]
        else:  # unseen state -> fresh draw from its population
            u_pred = post["tau_state"] * rng.standard_normal(s)
        mean = alpha_pred + u_pred
        if p:
            xi = np.asarray(x_std[i], dtype=float)
            mean = mean + post["beta"] @ xi
        draws = mean + post["sigma"] * rng.standard_normal(s)
        qs = np.quantile(draws, levels)
        grids[rid] = {float(lvl): float(q) for lvl, q in zip(levels, qs, strict=True)}
        loadings[rid] = {NATIONAL: national_sd, f"state:{state}": float(np.std(u_pred))}
        idio[rid] = idio_sd

    return MemberFit(grids, loadings, idio)


def run(ctx: RunContext) -> StepResult:
    """Fit the Bayes member and write member grids (+ latent factors for live)."""
    if ctx.cutoff_date is None:
        raise ValueError("models.bayes needs ctx.cutoff_date")
    plan_generation = 0
    total = 0
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        mi = load_member_input(conn, ctx.cutoff_date, plan_generation)
        if not mi.labeled_indices():
            raise ValueError("feature_matrix has no labeled rows; run features first")

        live = fit_predict(mi, target_cycle=None)
        total += write_member_predictions(
            conn,
            cutoff_date=ctx.cutoff_date,
            plan_generation=plan_generation,
            model_member=MODEL_MEMBER,
            fold="live",
            grids=live.grids,
        )
        if live.grids:
            write_latent_factors(
                conn,
                cutoff_date=ctx.cutoff_date,
                plan_generation=plan_generation,
                loadings=live.loadings,
                idiosyncratic_sd=live.idiosyncratic_sd,
            )
        if ctx.do_loco:
            for cycle in mi.labeled_cycles():
                fold = fit_predict(mi, target_cycle=cycle)
                total += write_member_predictions(
                    conn,
                    cutoff_date=ctx.cutoff_date,
                    plan_generation=plan_generation,
                    model_member=MODEL_MEMBER,
                    fold=str(cycle),
                    grids=fold.grids,
                )
    log.info("bayes.done", n_grids=total, do_loco=ctx.do_loco)
    return StepResult(node=STAGE, rows=total, detail=f"{total} Bayes member grids")


def dry_run(ctx: RunContext) -> StepResult:  # noqa: ARG001
    return StepResult(
        node=STAGE,
        rows=3_600,
        detail="posterior quantile grid + latent loadings (stub)",
        dry_run=True,
    )
