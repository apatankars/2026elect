"""Conformal layer DAG node — composes CQR + weighted + Mondrian + abstain.

Consumes the stacked quantile grid and calibration (LOCO) folds; emits per race:
median, conformal intervals at alpha in {0.5, 0.2, 0.1}, Mondrian group id,
abstain flag + reason, and win probability from the conformalized CDF.

Phase 4 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "conformal.apply"

run = not_implemented(STAGE, "Phase 4")
dry_run = stub(STAGE, rows=3_600, detail="CQR+weighted+Mondrian intervals, abstain flags (stub)")
