"""Enacted redistricting plan ingest.

Enacted 2026 plan shapefiles (Redistricting Data Hub, DRA exports). States that
redraw mid-cycle are re-pulled by the nightly job and get a new
``plan_generation`` so historical predictions stay immutable (plan §4). Feeds the
Phase 2 geo pipeline.

Shapefile-based areal interpolation (the ``maup`` path) needs the ``geo`` extra and
cached plan geometries. When neither is present this step is a tolerant no-op: the
geo stage falls back to the tabular pres-by-CD path (Daily Kos crosswalks), so the
pipeline stays runnable without heavy GIS deps. Real geometry ingest lands with the
full Phase 2 ``geo`` pipeline.
"""

from __future__ import annotations

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.ingest import base
from midterms26.logging import get_logger

STAGE = "ingest.plans"
SOURCE = "plans"
log = get_logger(STAGE)


def run(ctx: RunContext) -> StepResult:
    """Ingest cached plan shapefiles if present; otherwise no-op (tabular geo path)."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.shp")
    if not paths:
        log.info("plans.skip", reason="no cached shapefiles; geo uses the tabular pres-by-CD path")
        return StepResult(node=STAGE, rows=0, detail="no plan shapefiles; tabular geo path")
    # Real shapefile ingest (maup/geopandas) arrives with the full Phase 2 pipeline.
    raise NotImplementedError(
        f"found {len(paths)} plan shapefile(s) but shapefile ingest (maup) is Phase 2; "
        "remove them to use the tabular geo path."
    )


def dry_run(ctx: RunContext) -> StepResult:  # noqa: ARG001
    return StepResult(
        node=STAGE,
        rows=470,
        detail="Enacted 2026 plan shapefiles + plan_generation (stub)",
        dry_run=True,
    )
