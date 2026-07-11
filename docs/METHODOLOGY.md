# Methodology

The point model is deliberately not the star. The contribution is a UQ stack that
is *provably honest*: any reasonable point model, wrapped in conformal prediction
with finite-sample coverage, group-conditional calibration, online adaptation, and
explicit abstention.

## Target

Two-party margin **D% − R%** per race. Uncontested races carry an imputation flag
and are excluded from calibration sets.

## Base model — two-member stack

1. **Hierarchical Bayesian fundamentals (NumPyro).** Latent national environment
   (generic-ballot anchor), partial pooling across cycles / states / race types,
   incumbent random effects, explicit redistricting covariates. `σ_i` is
   regressed on poll coverage (heteroskedastic — unpolled races get wider
   posteriors). This is the interpretable spine and the source of the
   race-correlation structure: shared latent factors (national + regional +
   state) → an analytic correlation matrix, replacing the old SHAP-correlation
   hack. Emits a posterior predictive quantile grid (1%…99%) per race.
2. **TabPFN v2.** Tabular foundation model, in-context learning, no
   hyperparameter tuning; ~3–4k race-cycle rows × ~100 features. Captures
   nonlinear interactions the Bayesian linear predictor misses. Emits a native
   predictive quantile grid. Wrapped in a versioned adapter so it can be swapped
   (fallback: NGBoost / quantile forest — not LightGBM).

**Stack:** cycle-weighted linear stacking fit on leave-one-cycle-out (LOCO)
out-of-fold quantile predictions; non-negative weights per quantile fit by pinball
loss, exponentially cycle-weighted. Both members emit quantiles because the
conformal layer uses **Conformalized Quantile Regression (CQR)**, not plain
residual scores.

## Conformal stack (the contribution)

- **CQR** nonconformity scores on the stacked quantile predictions.
- **Weighted / nonexchangeable conformal** (Barber–Candès–Ramdas–Tibshirani):
  exponential decay weights over cycles (2022 > 2006) for robustness to drift.
- **Mondrian / group-conditional calibration** over strata — coverage must hold
  *within* group, audited in backtests. Axes: {incumbent-D, incumbent-R, open} ×
  {polled ≥3, polled 1–2, unpolled} × {redistricted, stable} × {House, Senate,
  Gov}.
- **Adaptive Conformal Inference (ACI):** online α-adjustment so empirical
  coverage self-corrects as the cycle drifts (live phase).
- **Selective prediction / abstention ("No Call"):** a race abstains when (a) its
  α=0.2 conformal interval width exceeds τ, or (b) its Mondrian bin has < `n_min`
  calibration points. First-class output, rendered with its reason. τ and `n_min`
  are fit in backtesting and frozen in `PREREGISTRATION.md`.

## Joint simulation

Gaussian copula. Marginals = per-race conformal predictive CDFs (interpolate the
conformalized quantile grid). Correlation = Bayesian shared latent factors. 50k
draws nightly → seat distributions, majority probabilities, tipping-point index,
conditional queries.

## Redistricting-native pipeline

`maup` areal interpolation: precinct → block (VAP-weighted) → new district.
Validation gate: reaggregated statewide totals match official within 0.1%. Per
2026 district: reaggregated PVI on new lines (level + 2016→2024 trend),
`incumbent_constituency_overlap` (incumbency advantage scales with overlap),
`is_new_seat`, `plan_enacted_date`, `plan_generation`. Predictions are immutable
per `plan_generation` — a mid-cycle redraw creates a new generation, never
overwrites history.

## Leakage discipline

Every feature carries an `as_of` date. The assembler takes `(race_id,
cutoff_date)` and refuses to read anything after the cutoff
(`features/leakage.py`, property-tested). Money is stored as quarterly snapshots;
polls and ratings are as-of-dated; pollster skill uses a rolling leakage-free
scheme (ratings for cycle Y trained only on polls from Y−12…Y−2).

## Phase 1b sources & candidate quality

Four additional ingest sources (see `docs/ROADMAP.md` for rationale): ACS 5-yr
demographics (aggregated onto 2026 lines in Phase 2), Daily Kos pres-by-CD
results (PVI cross-check on redrawn lines), DIME CFscores (candidate ideology —
a cycle-Y score is dated Dec 31 of cycle Y so it can never leak into a cycle-Y
forecast; `dime.scores_as_of` returns the latest usable score), and FEC
Schedule E independent expenditures (insider revealed preference; aggregated
leakage-safe by `fec_ie.race_ie_totals`). Candidate-quality features derived
from already-ingested FEC data (small-dollar share, self-funding, fundraising
slope, divisive-primary margin) land in Phase 3.

## Interpretability outputs (Phase 4+, roadmap workstream 2)

Alongside the quantile grids, the Bayes member emits: a per-race margin
decomposition (national + state + PVI + incumbency×overlap + candidate quality
+ money + residual), an effect ledger (posterior CIs per named effect, tracked
over the cycle), and an interval-width decomposition (national vs. state vs.
idiosyncratic). Abstentions already carry reasons; wide intervals get them too.
The TabPFN member is explained by exception: stack-member divergence per race
plus nearest historical analogs, with ALE/fANOVA for global structure.

## Phase → code map

| Phase | Deliverable | Code |
|-------|-------------|------|
| 0 | Scaffolding, warehouse schema, walkable DAG | `warehouse.py`, `dag.py`, `pipeline.py`, `pipelines/` |
| 1 | Historical backfill 2006–2024 | `ingest/` |
| 2 | Redistricting-native geo | `geo/` |
| 3 | Feature matrix | `features/` |
| 4 | Models + conformal + simulate | `models/`, `conformal/`, `simulate/` |
| 5 | Backtesting + pre-registration | `backtest/`, `docs/PREREGISTRATION.md` |
| 6 | Live pipeline + site JSON | `publish/`, GitHub Actions |

## Known risks (tracked, not silently resolved)

- Pre-2018 poll archives are patchy → thin "polled" strata in old cycles; may
  collapse strata for calibration.
- TabPFN licensing/inference for a nightly job → verify; fallback member ready.
- 2024 precinct returns incomplete in some states on RDH → track coverage
  state-by-state.
- Senate: ~35 races/cycle → pool with House via office fixed effects; check
  whether Senate needs its own σ inflation.
- States redrawing after a plan snapshot → `plan_generation` versioning keeps
  historical predictions immutable.
