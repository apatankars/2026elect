"""Conformalized Quantile Regression scores (Phase 4).

Nonconformity scores on the stacked quantile predictions:

    E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))

Calibrated interval = ``[q_lo - Q, q_hi + Q]`` where ``Q`` is the ``(1 - alpha)``
weighted quantile of the calibration scores ``{E_i}`` (see
:mod:`midterms26.conformal.weighted` for the nonexchangeable weighting and the
``+inf`` test-point mass). CQR adapts to heteroskedasticity — the base quantiles
already widen where the model is unsure, and the conformal step only corrects
their *coverage*, so intervals stay tight where the model is confident.

Pure-Python by design (no numpy) so it runs in light CI and stays auditable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from midterms26.conformal.weighted import weighted_conformal_quantile

STAGE = "conformal.cqr"


def nonconformity_scores(
    y: Sequence[float],
    q_lo: Sequence[float],
    q_hi: Sequence[float],
) -> list[float]:
    """CQR nonconformity score per calibration point.

    ``E_i = max(q_lo_i - y_i, y_i - q_hi_i)``: signed distance outside the base
    interval, negative when ``y_i`` lands comfortably inside ``[q_lo, q_hi]``.
    """
    n = len(y)
    if not (len(q_lo) == len(q_hi) == n):
        raise ValueError("y, q_lo, q_hi must be the same length")
    return [max(q_lo[i] - y[i], y[i] - q_hi[i]) for i in range(n)]


@dataclass(frozen=True)
class ConformalInterval:
    """A conformalized interval and the adjustment that produced it."""

    lo: float
    hi: float
    adjustment: float  # Q_{1-alpha}; +inf means an unbounded (No Call) interval

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def is_finite(self) -> bool:
        return math.isfinite(self.adjustment)


def calibrate(
    scores: Sequence[float],
    alpha: float,
    *,
    weights: Sequence[float] | None = None,
    test_weight: float | None = None,
) -> float:
    """Return the CQR adjustment ``Q`` = weighted ``(1 - alpha)`` score quantile."""
    return weighted_conformal_quantile(scores, weights, alpha, test_weight=test_weight)


def interval(q_lo: float, q_hi: float, adjustment: float) -> ConformalInterval:
    """Widen a base quantile interval by the conformal adjustment on both sides."""
    if q_hi < q_lo:
        raise ValueError(f"q_hi ({q_hi}) must be >= q_lo ({q_lo})")
    if math.isinf(adjustment):
        return ConformalInterval(-math.inf, math.inf, adjustment)
    return ConformalInterval(q_lo - adjustment, q_hi + adjustment, adjustment)


@dataclass(frozen=True)
class CQRCalibrator:
    """A fitted CQR calibrator: holds scores + weights, applies to new races.

    Built once per calibration set (e.g. one Mondrian bin) from held-out LOCO
    predictions, then applied to every test race sharing that bin. ``n`` is
    exposed so the abstention rule can check the bin against ``n_min``.
    """

    scores: tuple[float, ...]
    weights: tuple[float, ...] | None

    @classmethod
    def fit(
        cls,
        y: Sequence[float],
        q_lo: Sequence[float],
        q_hi: Sequence[float],
        *,
        weights: Sequence[float] | None = None,
    ) -> CQRCalibrator:
        scores = nonconformity_scores(y, q_lo, q_hi)
        w = tuple(float(x) for x in weights) if weights is not None else None
        if w is not None and len(w) != len(scores):
            raise ValueError("weights length must match calibration set size")
        return cls(scores=tuple(scores), weights=w)

    @property
    def n(self) -> int:
        return len(self.scores)

    def adjustment(self, alpha: float, *, test_weight: float | None = None) -> float:
        return calibrate(self.scores, alpha, weights=self.weights, test_weight=test_weight)

    def apply(
        self, q_lo: float, q_hi: float, alpha: float, *, test_weight: float | None = None
    ) -> ConformalInterval:
        """Conformalize one race's base quantiles at level ``alpha``."""
        return interval(q_lo, q_hi, self.adjustment(alpha, test_weight=test_weight))
