"""Weighted / nonexchangeable conformal quantile — recovery, edges, coverage."""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from midterms26.conformal.weighted import (
    exp_decay_weights,
    weighted_conformal_quantile,
)


def _ordinary_quantile(scores: list[float], alpha: float) -> float:
    """Textbook split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest."""
    n = len(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return sorted(scores)[k - 1]


@pytest.mark.parametrize("alpha", [0.5, 0.2, 0.1, 0.05])
@pytest.mark.parametrize("n", [1, 2, 5, 10, 19, 50])
def test_uniform_weights_recover_ordinary_conformal(n: int, alpha: float) -> None:
    rng = random.Random(n * 100 + int(alpha * 100))
    scores = [rng.gauss(0, 1) for _ in range(n)]
    got = weighted_conformal_quantile(scores, None, alpha)
    assert got == pytest.approx(_ordinary_quantile(scores, alpha)) or (
        math.isinf(got) and math.isinf(_ordinary_quantile(scores, alpha))
    )


def test_weight_scaling_is_invariant() -> None:
    # Scaling every weight (calibration + test point) by a constant leaves the
    # normalized quantile unchanged — the estimator depends only on weight ratios.
    scores = [0.1, 0.5, 0.3, 0.9, 0.2]
    w = [1.0, 2.0, 0.5, 3.0, 1.5]
    base = weighted_conformal_quantile(scores, w, 0.2, test_weight=1.0)
    for c in (0.5, 7.0):
        scaled = weighted_conformal_quantile(scores, [c * x for x in w], 0.2, test_weight=c * 1.0)
        assert scaled == pytest.approx(base)


def test_infinite_when_mass_below_target() -> None:
    # n=3, alpha=0.1 -> ceil(4*0.9)=4 > 3 -> quantile on the test point's +inf mass.
    assert weighted_conformal_quantile([0.0, 1.0, 2.0], None, 0.1) == math.inf


def test_empty_calibration_is_infinite() -> None:
    assert weighted_conformal_quantile([], None, 0.2) == math.inf


def test_single_point() -> None:
    # n=1, alpha=0.5 -> ceil(2*0.5)=1 -> the point itself; alpha=0.1 -> inf.
    assert weighted_conformal_quantile([3.0], None, 0.5) == 3.0
    assert weighted_conformal_quantile([3.0], None, 0.1) == math.inf


def test_recent_weight_pulls_quantile() -> None:
    # Heavily up-weighting the large recent score raises the quantile vs uniform.
    scores = [0.0, 0.0, 0.0, 10.0]
    up = weighted_conformal_quantile(scores, [1, 1, 1, 100], 0.2)
    uniform = weighted_conformal_quantile(scores, None, 0.2)
    assert up >= uniform


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_alpha_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1.0, 2.0], None, bad)


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1.0, 2.0], [1.0, -1.0], 0.2)


def test_weight_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1.0, 2.0], [1.0], 0.2)


# -- exp_decay_weights ------------------------------------------------------


def test_exp_decay_reference_is_unit_and_monotone() -> None:
    w = exp_decay_weights([2006, 2010, 2014, 2018, 2022], half_life=8.0)
    assert w[-1] == 1.0  # most recent cycle
    assert all(a <= b for a, b in zip(w, w[1:], strict=False))  # older <= newer


def test_exp_decay_half_life() -> None:
    # A point exactly half_life older than the reference has half the weight.
    w = exp_decay_weights([2014, 2022], reference_cycle=2022, half_life=8.0)
    assert w[0] == pytest.approx(0.5)


def test_exp_decay_future_cycles_clamped() -> None:
    w = exp_decay_weights([2026], reference_cycle=2022, half_life=8.0)
    assert w == [1.0]


def test_exp_decay_large_half_life_is_uniform() -> None:
    w = exp_decay_weights([2006, 2014, 2022], half_life=1e9)
    assert all(x == pytest.approx(1.0) for x in w)


def test_exp_decay_rejects_nonpositive_half_life() -> None:
    with pytest.raises(ValueError):
        exp_decay_weights([2020], half_life=0.0)


# -- properties -------------------------------------------------------------


@given(
    scores=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=1, max_size=40
    ),
    alpha=st.floats(min_value=0.01, max_value=0.99),
)
def test_finite_result_is_a_score(scores: list[float], alpha: float) -> None:
    q = weighted_conformal_quantile(scores, None, alpha)
    assert math.isinf(q) or any(q == pytest.approx(s) for s in scores)


@given(
    scores=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=2, max_size=40
    ),
    a1=st.floats(min_value=0.02, max_value=0.5),
    a2=st.floats(min_value=0.02, max_value=0.5),
)
def test_monotone_in_alpha(scores: list[float], a1: float, a2: float) -> None:
    # Smaller alpha (higher confidence) => quantile no smaller.
    lo, hi = sorted((a1, a2))
    q_lo_alpha = weighted_conformal_quantile(scores, None, lo)
    q_hi_alpha = weighted_conformal_quantile(scores, None, hi)
    assert q_lo_alpha >= q_hi_alpha - 1e-9 or math.isinf(q_lo_alpha)


def test_marginal_coverage_holds_exchangeable() -> None:
    """The headline guarantee: split conformal covers >= 1 - alpha, exchangeable."""
    rng = random.Random(0)
    alpha = 0.2
    n_cal = 50
    trials = 4000
    covered = 0
    for _ in range(trials):
        sample = [rng.gauss(0, 1) for _ in range(n_cal + 1)]
        cal, test = sample[:-1], sample[-1]
        # Base quantile "model" is the constant [0, 0]; scores are |y| distances.
        scores = [max(0.0 - y, y - 0.0) for y in cal]
        q = weighted_conformal_quantile(scores, None, alpha)
        if -q <= test <= q:
            covered += 1
    assert covered / trials >= 1 - alpha - 0.02
