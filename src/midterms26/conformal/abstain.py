"""Selective prediction / abstention — "No Call" (Phase 4).

A race gets "No Call" status when either:
  (a) its conformal interval at alpha=0.2 spans more than a width threshold tau
      (fit in backtesting), or
  (b) its Mondrian bin has < ``n_min`` calibration points.
"No Call" is a first-class output rendered on the site with the reason.

This module ships the decision rule now (pure, testable); thresholds ``tau`` and
``n_min`` are fit in Phase 5 backtesting and frozen in PREREGISTRATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AbstainDecision:
    abstain: bool
    reason: str | None


def decide(
    interval_width_80: float,
    bin_calibration_n: int,
    *,
    tau: float,
    n_min: int,
) -> AbstainDecision:
    """Return the "No Call" decision and its reason.

    ``interval_width_80`` is the width of the alpha=0.2 (80%) conformal interval.
    Reasons are stable strings so the site and ``predictions.abstain_reason`` agree.
    """
    if bin_calibration_n < n_min:
        return AbstainDecision(True, f"bin<{n_min}")
    if interval_width_80 > tau:
        return AbstainDecision(True, f"width>{tau:g}")
    return AbstainDecision(False, None)
