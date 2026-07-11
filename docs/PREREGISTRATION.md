# Pre-Registration — DRAFT (freeze by 2026-10-01)

> This document is a **draft** until frozen. Once frozen (git tag `prereg-frozen`
> before Oct 1, 2026), the metrics, comparison set, thresholds, and grading date
> below are committed and results will be published regardless of outcome.

## Commitment

We publish the graded results after the 2026 election **regardless of how the
model performs**. The claim under test is *calibration*, not point accuracy: we
predict that our conformal intervals achieve their nominal coverage, marginally
and within Mondrian groups, where competitors' intervals are unvalidated.

## Primary metrics (headline)

For α ∈ {0.5, 0.2, 0.1}, reported marginally and per Mondrian group:

1. **Empirical coverage vs. nominal** — the primary claim. Target: empirical
   coverage within finite-sample tolerance of (1 − α).
2. **Mean interval width** — sharpness, conditional on coverage holding.

## Secondary metrics

- Brier score, log loss, CRPS (per race and aggregated).
- Seat-count error (predicted vs. realized majority margin).
- Abstention rate and a qualitative description of abstained ("No Call") races.

## Abstention thresholds (fit in Phase 5 backtesting; fill before freeze)

- `τ` (max α=0.2 interval width before "No Call"): **TBD**
- `n_min` (min Mondrian-bin calibration points): **TBD**

## Pre-registered ablation: LLM-derived event features

One model input is gated behind an ablation committed here before the freeze:
LLM-classified, as-of-dated race event flags (scandal, retirement, dropout,
health) extracted from news. Design:

- **Arms:** full stack *with* vs. *without* the event-flag features; identical
  otherwise (same members, same conformal stack, same calibration sets).
- **Metrics:** LOCO-backtest CRPS and empirical coverage (primary), Brier and
  interval width (secondary), evaluated marginally and per Mondrian group.
- **Decision rule (fill before freeze):** the with-arm ships live only if it
  improves backtest CRPS without degrading coverage; otherwise the without-arm
  ships and the null result is published.
- LLM outputs never enter as free text — only the dated categorical flags, and
  they pass through the same leakage guard as every other feature. LLM
  *synthetic polling* is explicitly excluded as a feature under any arm.
- Grounded per-race narrative briefs are a rendering of the model's own
  decomposition JSON, not a model input, and are out of scope for this ablation.

## Baselines / comparison set

Internal baselines: PVI-only linear model, ratings-only model, raw poll average.
External comparison (as visible): Race to the WH, Split Ticket / The Argument,
Silver Bulletin (where paywall-visible), Kalshi, Polymarket.

Honest framing: we do not claim to beat everyone on Brier. We claim *valid
coverage* where others' intervals are vibes.

## Grading

- **Grading date:** TBD (post-certification window, fill before freeze).
- **Frozen predictions:** every nightly run appends to `calibration_log` with
  timestamps; the audit trail is the graded artifact.
- **Immutability:** predictions are keyed by `plan_generation`; mid-cycle redraws
  never overwrite prior-generation predictions.

## Open items to resolve before freeze

- [ ] Fit and record `τ`, `n_min`.
- [ ] Fill the LLM event-feature ablation decision rule thresholds.
- [ ] Confirm Mondrian strata that survive (may collapse thin pre-2018 poll bins).
- [ ] Final external comparison set and data-capture cadence.
- [ ] Exact grading date and certification cutoffs.
