"""JSON emitters + calibration dashboard data (Phase 6).

Emits static JSON artifacts consumed by the (separate) site repo: seat
distribution, race table with interval bars, "No Call" races with reasons, the
live calibration dashboard (reliability diagram), redistricting explainer data,
herding tracker. Every run appends to ``calibration_log`` with frozen timestamps
— the audit trail is the product.

JSON schemas are snapshot-tested (tests/publish). Election-night phase-two (an
ACI-updated needle) will consume the same JSON contracts.

Phase 6 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "publish.emit"

run = not_implemented(STAGE, "Phase 6")
dry_run = stub(STAGE, rows=1, detail="static JSON artifacts + calibration_log append (stub)")
