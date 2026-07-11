"""Enacted redistricting plan ingest.

Enacted 2026 plan shapefiles (Redistricting Data Hub, DRA exports). States that
redraw mid-cycle are re-pulled by the nightly job and get a new
``plan_generation`` so historical predictions stay immutable (plan §4). Feeds the
Phase 2 geo pipeline.

Phase 1/2 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "ingest.plans"

run = not_implemented(STAGE, "Phase 2")
dry_run = stub(STAGE, rows=470, detail="Enacted 2026 plan shapefiles + plan_generation (stub)")
