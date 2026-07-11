"""Mondrian / group-conditional calibration (Phase 4).

Coverage must hold *within* strata, audited in backtests. Strata are the cross of:
  * incumbency:    {incumbent-D, incumbent-R, open}
  * poll coverage: {polled >=3, polled 1-2, unpolled}
  * lines:         {redistricted, stable}
  * office:        {House, Senate, Gov}

Each race maps to one ``mondrian_group`` id; conformal quantiles are computed per
bin. Bins with < ``n_min`` calibration points trigger abstention (see
:mod:`midterms26.conformal.abstain`). Thin old-cycle strata may be collapsed
(plan §4).
"""

from __future__ import annotations

STAGE = "conformal.mondrian"

# Canonical stratum axes; the group id is a '|'-joined tuple in this order.
INCUMBENCY_LEVELS = ("incD", "incR", "open")
POLL_LEVELS = ("polled3+", "polled1-2", "unpolled")
LINES_LEVELS = ("redrawn", "stable")
OFFICE_LEVELS = ("HOUSE", "SENATE", "GOV")


def group_id(incumbency: str, polls: str, lines: str, office: str) -> str:
    """Compose the canonical Mondrian group id, validating each axis."""
    for value, allowed, axis in (
        (incumbency, INCUMBENCY_LEVELS, "incumbency"),
        (polls, POLL_LEVELS, "polls"),
        (lines, LINES_LEVELS, "lines"),
        (office, OFFICE_LEVELS, "office"),
    ):
        if value not in allowed:
            raise ValueError(f"invalid {axis} level {value!r}; expected one of {allowed}")
    return "|".join((incumbency, polls, lines, office))
