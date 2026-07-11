"""Pipeline assembly: wire stage modules into the backfill / nightly DAGs.

The two graphs share a spine — ingest -> geo -> features -> members -> stack ->
conformal -> simulate — and differ only at the terminal node:
  * backfill terminates in ``backtest.loco`` (LOCO coverage audits).
  * nightly terminates in ``publish.emit`` (static JSON + calibration_log).
"""

from __future__ import annotations

from midterms26.backtest import loco
from midterms26.conformal import apply as conformal_apply
from midterms26.context import RunContext
from midterms26.dag import DAG, Node
from midterms26.features import assemble as features_assemble
from midterms26.geo import reaggregate
from midterms26.ingest import (
    acs,
    dime,
    econ,
    fec,
    fec_ie,
    plans,
    polls,
    pres_by_cd,
    ratings,
    results,
    specials,
)
from midterms26.models import bayes, stack, tabpfn_member
from midterms26.publish import emit
from midterms26.simulate import copula

__all__ = [
    "RunContext",
    "build_backfill_dag",
    "build_ingest_dag",
    "build_nightly_dag",
]

# Ingest roots (no deps). ``ingest.plans`` also feeds geo. The last four are
# the Phase 1b sources (see docs/ROADMAP.md workstream 1a).
_INGEST = {
    "ingest.results": results,
    "ingest.fec": fec,
    "ingest.polls": polls,
    "ingest.ratings": ratings,
    "ingest.specials": specials,
    "ingest.econ": econ,
    "ingest.plans": plans,
    "ingest.acs": acs,
    "ingest.pres_by_cd": pres_by_cd,
    "ingest.dime": dime,
    "ingest.fec_ie": fec_ie,
}


def _node(name: str, deps: tuple[str, ...], module: object) -> Node:
    return Node(name=name, deps=deps, run=module.run, dry_run=module.dry_run)  # type: ignore[attr-defined]


# Phase 1 covers the six original tabular sources plus the four Phase 1b
# sources (acs, pres_by_cd, dime, fec_ie). ``ingest.plans`` (shapefiles) lands
# with the Phase 2 geo pipeline, so it is excluded from the ingest-only graph.
PHASE1_INGEST = tuple(n for n in _INGEST if n != "ingest.plans")


def build_ingest_dag() -> DAG:
    """Ingest-only graph (Phase 1 + 1b) — the tabular source nodes."""
    dag = DAG()
    for name in PHASE1_INGEST:
        dag.add(_node(name, (), _INGEST[name]))
    return dag


def _spine(dag: DAG) -> None:
    """Add the shared ingest -> ... -> simulate spine to ``dag``."""
    for name, module in _INGEST.items():
        dag.add(_node(name, (), module))

    # Geo needs enacted plans + precinct/statewide returns.
    dag.add(_node("geo.reaggregate", ("ingest.plans", "ingest.results"), reaggregate))

    # Features consume every source + geo, all leakage-guarded.
    feature_deps = (*_INGEST.keys(), "geo.reaggregate")
    dag.add(_node("features.assemble", feature_deps, features_assemble))

    # Two members, then the stack.
    dag.add(_node("models.bayes", ("features.assemble",), bayes))
    dag.add(_node("models.tabpfn", ("features.assemble",), tabpfn_member))
    dag.add(_node("models.stack", ("models.bayes", "models.tabpfn"), stack))

    # Conformal on the stacked quantiles.
    dag.add(_node("conformal.apply", ("models.stack",), conformal_apply))

    # Copula needs conformal marginals + Bayesian latent-factor correlation.
    dag.add(_node("simulate.copula", ("conformal.apply", "models.bayes"), copula))


def build_backfill_dag() -> DAG:
    """Historical backfill graph, terminating in LOCO backtests."""
    dag = DAG()
    _spine(dag)
    dag.add(_node("backtest.loco", ("conformal.apply",), loco))
    return dag


def build_nightly_dag() -> DAG:
    """Live nightly graph, terminating in JSON publish."""
    dag = DAG()
    _spine(dag)
    dag.add(_node("publish.emit", ("simulate.copula",), emit))
    return dag
