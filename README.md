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

## Status: Phase 1 (historical ingest) in progress

Phase 0 (scaffolding) is complete: the DAG is wired end to end and walkable on
stub data. Phase 1 implements the six tabular ingest sources for real —
parse → normalize → idempotent upsert into DuckDB, run against cached raw files.
Phase 1b adds four more sources (ACS demographics, Daily Kos pres-by-CD, DIME
CFscores, FEC Schedule E independent expenditures) — see `docs/ROADMAP.md` for
the full plan from here to launch. Downstream stages (geo, models, conformal,
simulate) still fail loudly until their phase lands.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# Phase 0 acceptance — walks the full backfill DAG on stub data:
uv run python pipelines/backfill.py --dry-run

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
docs/         METHODOLOGY.md · PREREGISTRATION.md (frozen before Oct 1, 2026)
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
