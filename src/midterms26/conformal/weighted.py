"""Weighted (nonexchangeable) conformal quantiles (Phase 4).

Barber-Candes-Ramdas-Tibshirani weighted conformal with exponential decay over
cycles so 2022 counts more than 2006 — robustness to distribution drift is the
whole point. Implements the weighted empirical quantile of nonconformity scores
with a point mass at +inf for the test point.

Phase 4 fills in the estimator; edge cases (all-equal weights -> ordinary
conformal; single calibration point) are covered in tests/conformal.
"""

from __future__ import annotations

STAGE = "conformal.weighted"
