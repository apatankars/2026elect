"""Redistricting-native areal interpolation (Phase 2).

`maup`-based precinct -> block (VAP-weighted disaggregation) -> new-district
aggregation. Produces per-2026-district reaggregated electoral history, recomputed
PVI on new lines, ``pct_population_carried``, ``incumbent_constituency_overlap``,
``is_new_seat``, ``plan_enacted_date``, ``plan_generation``.

Validation gate: reaggregated statewide totals must match official statewide
totals within 0.1% (enforced in tests/geo).

Phase 2 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "geo.reaggregate"

run = not_implemented(STAGE, "Phase 2")
dry_run = stub(STAGE, rows=470, detail="reaggregated 2026 districts, all plan generations (stub)")
