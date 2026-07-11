"""Helpers for Phase 0 stub steps.

Every pipeline stage ships a real ``run`` (raising :class:`NotImplementedError`
until its phase lands) and a ``dry_run`` that returns a synthetic
:class:`~midterms26.dag.StepResult`. This module keeps that boilerplate honest
and in one place so the DAG stays walkable end to end from day one.
"""

from __future__ import annotations

from midterms26.dag import RunContext, StepResult


def not_implemented(stage: str, phase: str) -> object:
    """Return a ``run`` callable that fails loudly until ``phase`` lands."""

    def _run(ctx: RunContext) -> StepResult:  # noqa: ARG001
        raise NotImplementedError(
            f"{stage!r} real execution arrives in {phase}; only --dry-run is wired in Phase 0."
        )

    return _run


def stub(stage: str, rows: int, detail: str) -> object:
    """Return a ``dry_run`` callable emitting a fixed synthetic result."""

    def _dry(ctx: RunContext) -> StepResult:  # noqa: ARG001
        return StepResult(node=stage, rows=rows, detail=detail, dry_run=True)

    return _dry
