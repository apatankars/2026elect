"""DAG runner: topological order, cycle/missing-dep detection, dry-run walk."""

from __future__ import annotations

import pytest

from midterms26.dag import DAG, Node, StepResult
from midterms26.pipeline import RunContext, build_backfill_dag, build_nightly_dag


def _n(name: str, deps: tuple[str, ...]) -> Node:
    return Node(
        name=name,
        deps=deps,
        run=lambda ctx: StepResult(name),
        dry_run=lambda ctx: StepResult(name, rows=1, dry_run=True),
    )


def test_topological_order_respects_dependencies() -> None:
    dag = DAG()
    dag.add(_n("a", ()))
    dag.add(_n("b", ("a",)))
    dag.add(_n("c", ("a", "b")))
    order = dag.topological_order()
    assert order.index("a") < order.index("b") < order.index("c")


def test_missing_dependency_raises() -> None:
    dag = DAG()
    dag.add(_n("b", ("a",)))
    with pytest.raises(ValueError, match="unknown node"):
        dag.topological_order()


def test_cycle_detected() -> None:
    dag = DAG()
    dag.add(_n("a", ("b",)))
    dag.add(_n("b", ("a",)))
    with pytest.raises(ValueError, match="cycle"):
        dag.topological_order()


def test_duplicate_node_rejected() -> None:
    dag = DAG()
    dag.add(_n("a", ()))
    with pytest.raises(ValueError, match="duplicate"):
        dag.add(_n("a", ()))


@pytest.mark.parametrize("build", [build_backfill_dag, build_nightly_dag])
def test_real_dags_are_acyclic_and_walkable(build) -> None:
    dag = build()
    order = dag.topological_order()
    assert len(order) == len(dag.nodes)
    results = dag.walk(RunContext(dry_run=True))
    assert len(results) == len(dag.nodes)
    assert all(r.dry_run for r in results)


def test_backfill_terminates_in_backtest_and_nightly_in_publish() -> None:
    assert "backtest.loco" in build_backfill_dag().nodes
    assert "backtest.loco" not in build_nightly_dag().nodes
    assert "publish.emit" in build_nightly_dag().nodes
    assert "publish.emit" not in build_backfill_dag().nodes
