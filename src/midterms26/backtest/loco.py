"""LOCO backtest harness + coverage audits (Phase 5).

Leave-one-cycle-out over midterms {2006, 2010, 2014, 2018, 2022} (train on all
other cycles incl. presidentials), features frozen at E-1, E-30, E-90 days.

Reports, per cycle and per Mondrian group: empirical vs. nominal coverage at all
three alpha (the headline table), interval width, Brier, CRPS, log loss,
seat-count error, abstention rate. Baselines: PVI-only linear, ratings-only, raw
poll average. Honest framing: the claim is valid coverage, not universally best
Brier.

Phase 5 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "backtest.loco"
MIDTERM_CYCLES = (2006, 2010, 2014, 2018, 2022)
FREEZE_HORIZONS_DAYS = (1, 30, 90)

run = not_implemented(STAGE, "Phase 5")
dry_run = stub(
    STAGE,
    rows=len(MIDTERM_CYCLES) * len(FREEZE_HORIZONS_DAYS),
    detail="LOCO coverage/width/Brier/CRPS per cycle x horizon (stub)",
)
