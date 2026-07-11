"""Feature matrix assembly (Phase 3).

Assembles the ~80-120 column feature matrix, all as-of-date parameterized. The
assembler takes ``(race_id, cutoff_date)`` and routes every time-varying source
through :mod:`midterms26.features.leakage` before it can contribute a feature.

Feature families: fundamentals, national environment, money, polls (race-level),
pollster ratings, redistricting. Uncontested races carry an imputation flag and
are excluded from calibration sets.

Phase 3 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "features.assemble"

run = not_implemented(STAGE, "Phase 3")
dry_run = stub(STAGE, rows=3_600, detail="race-cycle rows x ~100 features, leakage-guarded (stub)")
