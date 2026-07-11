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


def test_unimplemented_stage_fails_loudly() -> None:
    # Guard rail: stages whose phase hasn't landed must raise, so a non-dry-run
    # cannot silently produce empty artifacts. geo.reaggregate lands in Phase 2.
    from midterms26.geo import reaggregate

    try:
        reaggregate.run(object())  # type: ignore[arg-type]
    except NotImplementedError:
        return
    raise AssertionError("geo.reaggregate.run should raise NotImplementedError until Phase 2")
