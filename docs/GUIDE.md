# Guide — running midterms26 and what's left to build

A practical checklist: how to get the pipeline running end-to-end today, verify
it, and where the remaining roadmap work picks up. For the *why* behind design
choices see `docs/METHODOLOGY.md`; for the full roadmap see `docs/ROADMAP.md`.

## 1. One-time setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev,models]"   # core + dev tools + NumPyro/JAX (models spine)
```

- `.[dev]` alone is enough for the light stack (ingest, conformal core, stack,
  backtest metrics, publish) and is exactly what CI installs for the fast job.
- `.[models]` adds NumPyro/JAX for the Bayes member. TabPFN is **optional** — if
  its checkpoint isn't present the TabPFN member auto-falls back to a
  scikit-learn quantile-GBR, so you do *not* need to install `tabpfn` to run the
  demo.
- `.[geo]` (GeoPandas/maup) is only needed for the shapefile geo path, which is
  the one piece not yet implemented (see §4).

## 2. Run the demo (the whole spine, one command)

```bash
uv run midterms26 demo
```

This seeds a self-contained synthetic warehouse and runs
`geo → features → bayes → tabpfn → stack → conformal → copula → publish`, writing
six JSON artifacts to `data/site/`:

| File | Contents |
|------|----------|
| `races.json` | per-race median margin, conformal intervals (50/80/90%), Mondrian group, abstain flag |
| `no_call.json` | abstained races + reason |
| `expected_seats.json` | expected Dem seats by office (linearity of expectation) + bucket counts |
| `seat_distribution.json` | full seat histogram from the 50k-draw copula simulation |
| `calibration.json` | coverage-so-far dashboard rows |
| `manifest.json` | run id, as-of date, model version, git-traceable provenance |

You'll see a benign `Unable to initialize backend 'tpu'` line from JAX — it just
means JAX is running on CPU, which is expected on a laptop.

## 3. Verify the build

```bash
uv run pytest                    # 183 tests, light + models paths (models via importorskip)
uv run mypy                      # strict, src only
uv run ruff check src tests pipelines
uv run ruff format --check src tests pipelines conftest.py
uv run python pipelines/backfill.py --dry-run   # Phase 0 DAG acceptance
```

All of the above should pass green. CI mirrors these in `.github/workflows/ci.yml`
(a fast light-stack job + a separate `models-test` job that installs `.[models]`).

## 4. What's implemented vs. what's left

**Done and runnable end-to-end (Phases 0–4 core):**

- Phase 0 — scaffolding, DuckDB schema, walkable typed DAG.
- Phase 1 + 1b — all ten tabular ingest sources (parse → normalize → idempotent
  upsert), run offline against cached raw files.
- Phase 2 — geo on the **tabular pres-by-CD PVI path** (no GIS deps).
- Phase 3 — feature-matrix assembly (leakage-guarded).
- Phase 4 core — Bayes (NumPyro NUTS) + TabPFN/sklearn members → pinball-loss
  LOCO stack → conformal (CQR + weighted + Mondrian + ACI + abstain) → Gaussian
  copula → JSON publish.
- Backtest metrics (coverage/CRPS/Brier/log-loss/Mondrian) + LOCO harness.

**Remaining work, in dependency order:**

1. **Shapefile geo path (`maup`)** — the only piece of the core spine still
   stubbed. `geo/reaggregate.py` and `ingest/plans.py` fall back to the tabular
   path when no plan shapefiles are present, and **fail loudly** if shapefiles
   *are* present (rather than silently ignoring them). To finish: install
   `.[geo]`, drop enacted 2026 plan shapefiles under `data/raw/plans/`, and
   implement the areal-interpolation branch. Guarded by
   `tests/test_backfill_dryrun.py::test_unimplemented_path_fails_loudly`.
2. **Interpretability layer** (roadmap workstream 2) — per-race margin waterfall,
   effect ledger, uncertainty decomposition, black-box delta view, what-if
   engine, poll-placement optimizer. This is the project's headline
   differentiator and builds on the Bayes member's latent factors.
3. **Advanced features** (Phase 3 remainder) — candidate-quality / extremism / IE
   features and the bitemporal leakage guard.
4. **LLM ablation** (Phase 5) — dated news-event features behind the
   pre-registered ablation in `docs/PREREGISTRATION.md`, plus VOI machinery.
5. **Public static site** (Phase 6) — the interactive front end over the JSON
   artifacts, including the backtest replay.

## 5. Real ingest (optional, beyond the synthetic demo)

```bash
uv run midterms26 ingest --raw-dir data/raw --db-path data/warehouse.duckdb
```

Reads cached immutable downloads (see the raw-file layout in `README.md`). Live
network fetching is gated behind `--allow-fetch` and needs FEC/FRED API keys.
