"""Pinball-weighted LOCO stacking: loss, weight fitting, monotone rearrangement."""

from __future__ import annotations

import pytest

from midterms26.models.stack import (
    apply_stack,
    fit_stack,
    fit_stack_weight,
    pinball_loss,
    weighted_pinball,
)


def test_pinball_asymmetry() -> None:
    # Under-prediction (y > q) penalized by tau; over-prediction by (1 - tau).
    assert pinball_loss(y=10.0, q_hat=8.0, tau=0.9) == pytest.approx(0.9 * 2)
    assert pinball_loss(y=8.0, q_hat=10.0, tau=0.9) == pytest.approx(0.1 * 2)
    assert pinball_loss(y=5.0, q_hat=3.0, tau=0.5) == pytest.approx(0.5 * 2)


def test_pinball_rejects_bad_tau() -> None:
    with pytest.raises(ValueError):
        pinball_loss(1.0, 0.0, tau=0.0)


def test_weighted_pinball_weights_applied() -> None:
    y = [1.0, 1.0]
    q = [0.0, 2.0]  # each off by 1 in opposite directions
    # At tau=0.5 both cost 0.5; weighting shouldn't change the mean here.
    assert weighted_pinball(y, q, 0.5, [3.0, 1.0]) == pytest.approx(0.5)


def test_fit_weight_bracketing_optimum() -> None:
    # y=c constant, a=c-1, b=c+1, tau=0.5 -> loss 0.5*|2w-1|, minimized at w=0.5.
    c = 4.0
    y = [c] * 5
    a = [c - 1.0] * 5
    b = [c + 1.0] * 5
    assert fit_stack_weight(y, a, b, 0.5) == pytest.approx(0.5, abs=1e-3)


def test_fit_weight_prefers_accurate_member() -> None:
    y = [0.0, 2.0, -1.0, 3.0, 1.0]
    a = list(y)  # member A is exact
    b = [v + 5.0 for v in y]  # member B biased high
    assert fit_stack_weight(y, a, b, 0.5) > 0.9


def test_fit_stack_same_levels_required() -> None:
    y = [0.0, 1.0]
    with pytest.raises(ValueError):
        fit_stack(y, {0.5: [0.0, 1.0]}, {0.9: [0.0, 1.0]})


def test_fit_stack_returns_weight_per_level() -> None:
    y = [0.0, 1.0, 2.0]
    a = {0.1: [0.0, 1.0, 2.0], 0.9: [0.0, 1.0, 2.0]}
    b = {0.1: [1.0, 2.0, 3.0], 0.9: [1.0, 2.0, 3.0]}
    w = fit_stack(y, a, b)
    assert set(w) == {0.1, 0.9}
    assert all(0.0 <= v <= 1.0 for v in w.values())


def test_apply_stack_blends_and_rearranges() -> None:
    # Per-level weights chosen so the raw blend crosses; output must be monotone.
    weights = {0.1: 0.0, 0.5: 1.0, 0.9: 0.0}
    a_row = {0.1: 5.0, 0.5: -2.0, 0.9: 6.0}  # A dips at the median
    b_row = {0.1: -3.0, 0.5: 0.0, 0.9: 1.0}
    grid = apply_stack(weights, a_row, b_row)
    values = [grid[t] for t in sorted(grid)]
    assert values == sorted(values)  # non-crossing after rearrangement


def test_apply_stack_requires_full_rows() -> None:
    with pytest.raises(ValueError):
        apply_stack({0.1: 0.5, 0.9: 0.5}, {0.1: 1.0}, {0.1: 0.0, 0.9: 2.0})
