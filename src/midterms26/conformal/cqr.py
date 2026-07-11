"""Conformalized Quantile Regression scores (Phase 4).

Nonconformity scores on the stacked quantile predictions:
    E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))
Calibrated interval = [q_lo - Q_{1-alpha}(E), q_hi + Q_{1-alpha}(E)]. The
quantile of E is taken with the nonexchangeable weights from
:mod:`midterms26.conformal.weighted`.

Phase 4 fills in the estimator.
"""

from __future__ import annotations

STAGE = "conformal.cqr"
