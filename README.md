# midterms26 — Calibration-Native 2026 Midterms Forecast

A House/Senate/Gov forecast for the 2026 U.S. midterms whose **uncertainty is the
product**. Any reasonable point model, wrapped in a UQ stack that is *provably
honest*: conformal prediction intervals with finite-sample coverage guarantees,
group-conditional (Mondrian) coverage, online adaptation through the cycle, and
explicit abstention ("No Call"). Grading criteria are pre-registered before
election day.

Two integrated workstreams:

- **(A) Redistricting-native fundamentals** — a consistent historical baseline
  for every 2026 district, including mid-cycle redraws (areal interpolation onto
  enacted lines, per `plan_generation`).
- **(B) Calibration-native forecast** — Conformalized Quantile Regression on a
  two-member stack (hierarchical Bayes + TabPFN), weighted/nonexchangeable
  conformal, Mondrian calibration, Adaptive Conformal Inference, abstention, and
  a Gaussian-copula joint simulation for seat distributions.

## Status: full modeling spine runs end-to-end

The complete pipeline — ingest → geo → features → both members → stack →
conformal → copula → publish — now runs end to end and emits the site JSON. Run
it in one command on a self-contained synthetic warehouse:

```bash
uv pip install -e ".[dev,models]"   # models extra: NumPyro/JAX (+ sklearn fallback)
uv run midterms26 demo              # seeds a synthetic warehouse, runs the whole spine
# -> Forecast artifacts written to: data/site  (races, expected_seats,
#    seat_distribution, no_call, calibration, manifest)
```

The demo doubles as an end-to-end smoke test: its synthetic generative model is
known (`national[cycle] + district_pvi + incumbency_bump + noise`), so the
conformal intervals should cover and the stack should not underperform either
member. TabPFN is optional — when its checkpoint is absent the member degrades to
the scikit-learn quantile-GBR fallback automatically, so `demo` needs only the
`models` extra (NumPyro/JAX), not TabPFN.

Phase 0 (scaffolding) is complete: the DAG is wired end to end and walkable on
stub data. Phase 1 + 1b implement all ten tabular ingest sources for real —
parse → normalize → idempotent upsert into DuckDB, run against cached raw files
(the six originals plus ACS demographics, Daily Kos pres-by-CD, DIME CFscores,
and FEC Schedule E independent expenditures).

The distinctive UQ / eval / publish layers are now implemented and unit-tested in
pure Python (no heavy stack — they run in the light CI):

- **Conformal core** (`conformal/`): CQR nonconformity + calibrated intervals,
  Barber et al. weighted/nonexchangeable quantile with a `+inf` test-point mass
  (→ "No Call"), exponential cycle-decay weights, and online ACI. The marginal
  finite-sample coverage guarantee is exercised by Monte-Carlo tests.
- **Stacking** (`models/stack.py`): pinball-loss LOCO mixing weights per quantile
  (convex ternary search) + Chernozhukov rearrangement to non-crossing grids.
- **Backtest metrics** (`backtest/metrics.py`): coverage gap, interval width,
  CRPS-from-quantiles, Brier, log loss, and the Mondrian per-group coverage audit
  — the eval-gated-CI backbone.
- **Publish** (`publish/emit.py`): runnable JSON emitters (race table, No-Call,
  expected-seats via linearity of expectation, calibration dashboard, run
  manifest) reading the `predictions` table + appending to the `calibration_log`
  honesty ledger.

Both **model members** are now implemented against the `[models]` extra (imported
lazily so the package still installs and tests light):

- **Hierarchical Bayes** (`models/bayes.py`, NumPyro NUTS): national latent
  (partially pooled per cycle) + state random effects + linear fundamentals.
  Cross-cycle prediction is leakage-safe by construction — a target/live cycle
  draws its national environment from the hyperprior, never its own margins — so
  novel/unpolled races get honestly wide posteriors. Also emits the shared
  latent-factor loadings that give the copula its analytic race-correlation matrix.
- **TabPFN member** (`models/tabpfn_member.py`): versioned adapter with a
  scikit-learn quantile-GBR **fallback** (forced via `MIDTERMS26_TABPFN_BACKEND=sklearn`)
  so a nightly run degrades gracefully when the TabPFN checkpoint is unavailable.

Both write per-race quantile grids to the `member_predictions` warehouse table
(`live` fold + per-cycle LOCO folds in backfill), consumed by the stacking
estimator. A separate CI job installs `.[models]` and exercises them for real.

Feature assembly (`features/assemble.py`) and the full mid-spine (`geo.reaggregate`,
`models.stack`, `conformal.apply`, `simulate.copula`, `publish.emit`) are all
implemented and exercised by `midterms26 demo`. Geo runs on the tabular
pres-by-CD PVI path; the **one** path still to land is shapefile areal
interpolation (the `maup` GIS path in `geo/reaggregate.py` + `ingest/plans.py`),
which fails loudly if plan shapefiles are present rather than silently ignoring
them. The remaining roadmap work is the differentiators layered on top of this
spine — interpretability decomposition, the LLM ablation, and the public site.
See `docs/ROADMAP.md` for the full plan.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# Phase 0 acceptance — walks the full backfill DAG on stub data:
uv run python pipelines/backfill.py --dry-run

# Full spine on a synthetic warehouse — writes site JSON to data/site (needs .[models]):
uv run midterms26 demo

# Phase 1 — real ingest of the six tabular sources from cached raw files:
uv run midterms26 ingest --raw-dir data/raw --db-path data/warehouse.duckdb

# Inspect / utilities:
uv run midterms26 show-dag ingest
uv run midterms26 init-db
```

### Ingest raw-file layout

Ingest runs offline against cached, immutable downloads. Live fetching is gated
behind `--allow-fetch` (needs network + FEC/FRED API keys); by default the
pipeline only reads what's already on disk:

```
data/raw/
  mit_edsl/*.csv     MIT returns (one CSV per office; candidate-level rows)
  fec/*.{parquet,csv} itemized receipts (race_id, candidate_id, contributor_type, is_self, amount, receipt_date)
  polls/*.csv        poll rows (race_id, as_of, pollster_id, shares/margin, mode, population, n)
  specials/*.csv     specials (special_id, cycle, election_date, seat_pvi, result_margin, turnout)
  ratings/*.csv      dated rating snapshots (race_id, source, as_of, rating)
  econ/<series>.csv  FRED / approval series (date, value); series id from filename
  acs/*.csv          tidy ACS 5-yr rows (geoid, geo_level, vintage, release_date, variable, value)
  pres_by_cd/*.csv   Daily Kos pres-by-CD (state, district, plan_label, pres_year, dem_votes, rep_votes)
  dime/*.csv         DIME CFscores (candidate_id, cycle, cfscore[, cfscore_dyn, n_donors])
  fec_ie/*.csv       FEC Schedule E IEs (ie_id, race_id, candidate_id, committee_id, support_oppose, amount, expenditure_date)
```

Each source module documents its exact schema and assumptions; see
`src/midterms26/ingest/` and `docs/METHODOLOGY.md`. Fixtures in
`tests/fixtures/raw/` show the expected shapes.

Install the heavier stacks only when their phase needs them:

```bash
uv pip install -e ".[models]"   # NumPyro/JAX, TabPFN, scikit-learn
uv pip install -e ".[geo]"      # GeoPandas, shapely, maup
uv pip install -e ".[ingest]"   # httpx, tenacity
```

## Repo layout

```
src/midterms26/
  ingest/     one module per source (results, fec, polls, ratings, specials, econ, plans)
  geo/        redistricting-native areal interpolation (Phase 2)
  features/   feature-matrix assembly + the leakage guard (non-negotiable)
  models/     bayes.py (NumPyro) · tabpfn_member.py · stack.py (LOCO)
  conformal/  cqr · weighted · mondrian · aci · abstain · apply
  simulate/   Gaussian-copula joint simulation
  backtest/   LOCO harness + coverage audits
  publish/    static JSON emitters + calibration dashboard
  warehouse.py  DuckDB schema · dag.py  DAG runner · pipeline.py  wiring · cli.py
pipelines/    nightly.py · backfill.py (orchestration entrypoints)
docs/         GUIDE.md (run + what's left) · METHODOLOGY.md · ROADMAP.md · PREREGISTRATION.md
```

## Stack

Python 3.12 · `uv` · Polars + DuckDB · NumPyro (JAX) · TabPFN · `maup`/GeoPandas ·
GitHub Actions nightly cron · outputs as static JSON. No AWS/DynamoDB — static-file
simple.

## Design decisions (frozen; see `docs/METHODOLOGY.md`)

Base = two-member stack (hierarchical Bayes + TabPFN v2), conformal on top. Target
= two-party margin (D% − R%). Race correlation from the Bayesian model's shared
latent factors (national + regional + state), feeding a Gaussian copula. All
features are as-of-date parameterized and pass through the leakage guard.
