"""Hierarchical Bayesian fundamentals model (NumPyro) — Phase 4.

``margin_i ~ Normal(mu_i, sigma_i)`` where ``mu_i`` = national latent + state
random effect + beta.(fundamentals) with hierarchical priors and incumbent random
effects; ``sigma_i`` is regressed on poll coverage (heteroskedastic — unpolled
races get wider posteriors). Fit with NUTS; nightly refits are fine at this data
size.

Emits: posterior predictive quantile grid (1%..99%) per race, plus the latent
factor loadings (national + regional + state) that yield the analytic
race-correlation matrix used by the copula simulator.

Requires the ``models`` extra. Phase 4 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "models.bayes"

run = not_implemented(STAGE, "Phase 4")
dry_run = stub(STAGE, rows=3_600, detail="posterior quantile grid + latent loadings (stub)")
