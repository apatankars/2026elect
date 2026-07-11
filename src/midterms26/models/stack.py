"""Cycle-weighted linear stack of the two members — Phase 4.

Leave-one-cycle-out (LOCO) out-of-fold quantile predictions from both members ->
non-negative weighted average per quantile, weights fit by pinball loss and
exponentially cycle-weighted (2022 counts more than 2006). Output quantile grid
feeds the conformal layer (which uses CQR, not plain residual scores).

Phase 4 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "models.stack"

run = not_implemented(STAGE, "Phase 4")
dry_run = stub(STAGE, rows=3_600, detail="LOCO-stacked quantile grid, pinball-weighted (stub)")
