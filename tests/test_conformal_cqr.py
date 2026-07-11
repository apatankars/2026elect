"""CQR nonconformity scores, interval construction, and marginal coverage."""

from __future__ import annotations

import math
import random

import pytest

from midterms26.conformal.cqr import (
    ConformalInterval,
    CQRCalibrator,
    interval,
    nonconformity_scores,
)


def test_scores_negative_inside_positive_outside() -> None:
    # y=5 inside [0,10] -> negative score; y=12 above hi -> +2; y=-3 below lo -> +3.
    scores = nonconformity_scores([5.0, 12.0, -3.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
    assert scores[0] == -5.0
    assert scores[1] == 2.0
    assert scores[2] == 3.0


def test_scores_length_mismatch() -> None:
    with pytest.raises(ValueError):
        nonconformity_scores([1.0], [0.0], [1.0, 2.0])


def test_interval_widens_symmetrically() -> None:
    iv = interval(q_lo=-2.0, q_hi=3.0, adjustment=1.5)
    assert (iv.lo, iv.hi) == (-3.5, 4.5)
    assert iv.width == 8.0
    assert iv.is_finite


def test_interval_infinite_adjustment() -> None:
    iv = interval(q_lo=-2.0, q_hi=3.0, adjustment=math.inf)
    assert iv.lo == -math.inf and iv.hi == math.inf
    assert not iv.is_finite


def test_interval_rejects_inverted_base() -> None:
    with pytest.raises(ValueError):
        interval(q_lo=5.0, q_hi=1.0, adjustment=0.0)


def test_calibrator_fit_apply() -> None:
    cal = CQRCalibrator.fit(
        y=[1.0, -1.0, 0.5, -0.5],
        q_lo=[-1.0, -1.0, -1.0, -1.0],
        q_hi=[1.0, 1.0, 1.0, 1.0],
    )
    assert cal.n == 4
    iv = cal.apply(q_lo=-1.0, q_hi=1.0, alpha=0.5)
    assert isinstance(iv, ConformalInterval)
    # Symmetric base + symmetric data -> symmetric conformal interval.
    assert iv.lo == pytest.approx(-iv.hi)


def test_calibrator_weight_length_checked() -> None:
    with pytest.raises(ValueError):
        CQRCalibrator.fit([1.0, 2.0], [0.0, 0.0], [1.0, 1.0], weights=[1.0])


def test_cqr_marginal_coverage_heteroskedastic() -> None:
    """CQR keeps >= 1 - alpha coverage even when noise scale varies with x."""
    rng = random.Random(7)
    alpha = 0.1
    n_cal = 200
    trials = 1500
    covered = 0

    def sample() -> tuple[float, float, float]:
        # x in [0,1] drives the noise scale; base quantiles track +/- 1.64*scale.
        x = rng.random()
        scale = 0.5 + 2.0 * x
        y = rng.gauss(0.0, scale)
        q_lo = -1.6449 * scale
        q_hi = 1.6449 * scale
        return y, q_lo, q_hi

    for _ in range(trials):
        cal = [sample() for _ in range(n_cal)]
        y_c = [s[0] for s in cal]
        lo_c = [s[1] for s in cal]
        hi_c = [s[2] for s in cal]
        calibrator = CQRCalibrator.fit(y_c, lo_c, hi_c)
        y_t, lo_t, hi_t = sample()
        iv = calibrator.apply(lo_t, hi_t, alpha)
        if iv.lo <= y_t <= iv.hi:
            covered += 1
    assert covered / trials >= 1 - alpha - 0.02
