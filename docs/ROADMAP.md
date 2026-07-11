# Roadmap — from calibrated forecast to flagship project

Positioning: **the only 2026 forecast whose uncertainty is provably honest — with
every prediction decomposed into named, quantified causes, graded publicly against
a pre-registered standard.** Three pillars: honest uncertainty (the conformal
stack, already designed) + causes-not-just-predictions (interpretability layer
below) + production ML discipline (eval-gated CI, bitemporal data).

Market context: no public forecaster (Silver Bulletin, Split Ticket/The Argument,
markets) publishes intervals with coverage guarantees. That gap is the moat; this
roadmap builds everything else around making it *visible and explainable*.

## Workstream 1 — Data

### 1a. Phase 1b ingest sources (all free, same parse → normalize → upsert pattern)

| Source | Module | Feeds |
|---|---|---|
| Census ACS 5-yr | `ingest/acs.py` | District demographics (% BA+, income, race, age) — core covariates; aggregated onto 2026 lines by the Phase 2 `maup` pipeline |
| Daily Kos Elections pres-by-CD | `ingest/pres_by_cd.py` | PVI cross-check on new lines; fallback where RDH precinct coverage is thin |
| DIME / CFscores (Stanford) | `ingest/dime.py` | `candidate_extremism`; the moderates-overperform effect. 2026 candidates not in DIME: crude CFscore from donor overlap in FEC itemized receipts we already ingest |
| FEC Schedule E IEs | `ingest/fec_ie.py` | "Insider revealed preference" competitiveness signal — where parties/PACs spend late money |
| Derived candidate quality | (features phase) | Small-dollar share, out-of-district donor share, self-funding, fundraising slope, divisive-primary margin, prior office — WAR-style features, leakage-safe |

### 1b. Correctness upgrades

- **Bitemporal warehouse**: event date vs. knowledge date (`recorded_at`) so any
  run can answer "what did we know on date X"; leakage guard enforces both.
- **Data contracts** per source: schema/range/freshness assertions before upsert;
  failures quarantine the file, never partially load.
- **Freshness & coverage monitors** emitted into published JSON (data-health page).
- **Entity resolution**: one `candidates` crosswalk (FEC ↔ MIT-EDSL ↔ DIME),
  deterministic first, reviewable fuzzy exceptions second.

## Workstream 2 — Interpretability core (the centerpiece)

Derived natively from the hierarchical Bayes member (not post-hoc):

1. **Per-race margin waterfall** — national environment + state effect + PVI +
   incumbency×overlap + candidate quality + money + residual; JSON per race.
2. **Effect ledger** — posterior CIs for every named effect (incumbency, midterm
   penalty, extremism penalty, special-swing coefficient), tracked over the cycle.
3. **Uncertainty decomposition** — interval width split into national / state /
   idiosyncratic; every wide interval and abstention carries a reason.
4. **Black-box delta view** — where TabPFN disagrees with the interpretable spine,
   with nearest historical analog races as rationale; ALE/fANOVA for global structure.
5. **What-if engine** — conditional copula queries exposed on the site (generic
   ballot ±5, approval −3, extremism swap → seat distribution shift).
6. **Poll-placement optimizer** — rank races by expected seat-entropy reduction
   per marginal poll (value-of-information; nobody public does this).

## Workstream 3 — LLM track (pre-registered ablation)

- **Grounded race briefs** (product output, never a model input): generated
  strictly from the decomposition JSON, every sentence traceable to a feature.
- **LLM news-event features** (model input, ablation-gated): dated event flags
  (scandal/retirement/dropout/health) through the leakage guard; stack-with vs.
  stack-without graded on CRPS/coverage in LOCO backtests. Null result gets
  published. Design frozen in `PREREGISTRATION.md` before Oct 1, 2026.
- **No LLM synthetic polling** — unreliable per 2025 research (prompt sensitivity,
  partisan asymmetry).

## Workstream 4 — Engineering discipline

- **Eval-gated CI**: LOCO backtest runs on every PR touching `features/`,
  `models/`, `conformal/`; posts coverage/CRPS/width diffs as a PR comment.
- Golden-snapshot regression tests; run manifests (git SHA, data versions, config
  hash) on every nightly artifact; feature-drift monitoring on the data-health page.
- Keep static-JSON no-server architecture — auditable, cheap, immutable artifacts
  in git history = free hash-chained honesty ledger.

## Workstream 5 — Public site (static-hosted, interactive)

National seat distribution → per-race pages (waterfall, interval + reason, brief,
black-box delta, analogs) → what-if simulator → calibration dashboard (coverage-
so-far as specials/primaries resolve) → **backtest replay** (scrub 2018/2022
day-by-day watching ACI intervals adapt — the flagship demo) → data-health page →
model-vs-markets divergence tracker with feature-level attribution.

## Sequencing (extends the phase map in METHODOLOGY.md)

1. **Phase 1b (now)**: four new ingest sources + data contracts + crosswalk.
2. **Phase 2**: unchanged, plus Daily Kos crosswalk validation + ACS aggregation.
3. **Phase 3**: candidate-quality/extremism/IE features; bitemporal leakage guard.
4. **Phase 4**: unchanged core + decomposition & uncertainty-split emitters.
5. **Phase 5**: LLM ablation + VPI machinery in the backtest harness; freeze prereg.
6. **Phase 6**: interactive site + briefs + replay. Eval-gated CI lands as soon as
   the backtest harness exists — it protects Phase 4.

## Key sources

- Split Ticket 2026 model: <https://www.theargumentmag.com/p/split-ticket-2026-midterms-model>
- Split Ticket WAR: <https://split-ticket.org/full-wins-above-replacement-war-database/>, critique: <https://split-ticket.org/2025/08/15/deconstructing-war/>
- DIME v4 / CFscores: <https://data.stanford.edu/dime>
- Moderates & electability (Bonica–Grumbach): <https://data4democracy.substack.com/p/do-moderates-do-better>
- Nonexchangeable conformal (Barber et al.): <https://www.stat.berkeley.edu/~ryantibs/papers/nexcp.pdf>
- LLM election-prediction reliability: <https://arxiv.org/pdf/2502.16280>, <https://arxiv.org/html/2412.15291v1>
- Specials as environment signal: <https://www.brookings.edu/articles/what-do-special-elections-mean-for-the-midterm-elections/>
- fANOVA vs SHAP: <https://arxiv.org/pdf/2208.09970>
