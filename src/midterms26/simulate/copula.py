"""Joint simulation via Gaussian copula (Phase 4).

Marginals = per-race conformal predictive CDFs (interpolate the conformalized
quantile grid). Correlation matrix = from the Bayesian model's shared latent
factors (national + regional + state) — this replaces the old SHAP-correlation
hack. Sample 50k elections nightly -> seat distributions, majority probabilities,
tipping-point races, and conditional queries ("if D wins NC-01, P(House) = ...").

Phase 4 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "simulate.copula"
N_DRAWS = 50_000

run = not_implemented(STAGE, "Phase 4")
dry_run = stub(
    STAGE, rows=N_DRAWS, detail="50k copula draws -> seat dist, P(majority), tipping (stub)"
)
