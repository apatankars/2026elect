"""A tiny typed DAG runner for the forecast pipeline.

Each :class:`Node` names its upstream dependencies and provides two callables:
  * ``run(ctx)``     — the real step (Phase 1+ fills these in).
  * ``dry_run(ctx)`` — a stub that touches no external source and returns a
    :class:`StepResult` describing what it *would* do. Phase 0 ships only these.

``pipelines/backfill.py --dry-run`` topologically walks the graph and calls
``dry_run`` on every node, which is the Phase 0 acceptance test.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from midterms26.logging import get_logger

log = get_logger("dag")


@dataclass(frozen=True)
class StepResult:
    """Outcome of a single node execution."""

    node: str
    rows: int = 0
    detail: str = ""
    dry_run: bool = False


class RunContext(Protocol):
    """What node callables receive. Kept minimal for Phase 0."""

    dry_run: bool


StepFn = Callable[["RunContext"], StepResult]


@dataclass(frozen=True)
class Node:
    name: str
    deps: tuple[str, ...]
    run: StepFn
    dry_run: StepFn


@dataclass
class DAG:
    nodes: dict[str, Node] = field(default_factory=dict)

    def add(self, node: Node) -> None:
        if node.name in self.nodes:
            raise ValueError(f"duplicate node: {node.name}")
        self.nodes[node.name] = node

    def topological_order(self) -> list[str]:
        """Kahn's algorithm; raises on cycles or missing dependencies."""
        indeg: dict[str, int] = {n: 0 for n in self.nodes}
        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for node in self.nodes.values():
            for dep in node.deps:
                if dep not in self.nodes:
                    raise ValueError(f"node {node.name!r} depends on unknown node {dep!r}")
                adj[dep].append(node.name)
                indeg[node.name] += 1

        queue: deque[str] = deque(sorted(n for n, d in indeg.items() if d == 0))
        order: list[str] = []
        while queue:
            cur = queue.popleft()
            order.append(cur)
            for nxt in sorted(adj[cur]):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self.nodes):
            unresolved = sorted(set(self.nodes) - set(order))
            raise ValueError(f"cycle detected among nodes: {unresolved}")
        return order

    def walk(self, ctx: RunContext) -> list[StepResult]:
        """Execute every node in dependency order, returning per-step results."""
        results: list[StepResult] = []
        for name in self.topological_order():
            node = self.nodes[name]
            fn = node.dry_run if ctx.dry_run else node.run
            result = fn(ctx)
            mode = "dry-run" if ctx.dry_run else "run"
            log.info("step.done", node=name, mode=mode, rows=result.rows, detail=result.detail)
            results.append(result)
        return results
