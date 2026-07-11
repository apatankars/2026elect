"""Phase 0 acceptance test: backfill --dry-run walks the DAG end to end."""

from __future__ import annotations

import pipelines.backfill as backfill
import pipelines.nightly as nightly

from midterms26.pipeline import build_backfill_dag


def test_backfill_dry_run_exits_zero() -> None:
    assert backfill.main(["--dry-run"]) == 0


def test_nightly_dry_run_exits_zero() -> None:
    assert nightly.main(["--dry-run"]) == 0


def test_backfill_dry_run_visits_every_node() -> None:
    dag = build_backfill_dag()
    # Walking the graph in dry-run must produce one result per node.
    from midterms26.pipeline import RunContext

    results = dag.walk(RunContext(dry_run=True))
    assert {r.node for r in results} == set(dag.nodes)


def test_unimplemented_path_fails_loudly(tmp_path: object) -> None:
    # Guard rail: the one path that hasn't landed — shapefile areal interpolation
    # (maup, Phase 2) — must raise when plan geometries are present, so a run can
    # never silently ignore them. Without shapefiles both geo and ingest.plans fall
    # back to the tabular pres-by-CD path (exercised end-to-end by the demo).
    from pathlib import Path

    from midterms26.context import RunContext
    from midterms26.geo import reaggregate
    from midterms26.ingest import plans

    raw_dir = Path(tmp_path)  # type: ignore[arg-type]
    (raw_dir / "plans").mkdir(parents=True)
    (raw_dir / "plans" / "enacted_2026.shp").write_bytes(b"")
    ctx = RunContext(raw_dir=raw_dir, db_path=raw_dir / "wh.duckdb")

    for stage in (plans, reaggregate):
        try:
            stage.run(ctx)
        except NotImplementedError:
            continue
        raise AssertionError(
            f"{stage.__name__}.run should raise NotImplementedError on a Phase 2 shapefile"
        )
