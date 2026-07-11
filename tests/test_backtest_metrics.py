"""Backtest scoring metrics: coverage, width, CRPS, Brier, log loss, Mondrian."""

from __future__ import annotations

import math

import pytest

from midterms26.backtest.metrics import (
    brier_score,
    coverage_by_group,
    coverage_row,
    crps_from_quantiles,
    interval_coverage,
    log_loss,
    mean_interval_width,
    seat_count_error,
)


def test_interval_coverage_counts_hits() -> None:
    y = [0.0, 5.0, -3.0, 10.0]
    lo = [-1.0, 0.0, -2.0, 0.0]
    hi = [1.0, 4.0, 0.0, 20.0]  # index 1 (5 not in [0,4]) and 2 (-3 not in [-2,0]) miss
    assert interval_coverage(y, lo, hi) == pytest.approx(0.5)


def test_infinite_bounds_count_as_covered() -> None:
    assert interval_coverage([100.0], [-math.inf], [math.inf]) == 1.0


def test_mean_width_rejects_unbounded() -> None:
    with pytest.raises(ValueError):
        mean_interval_width([-math.inf], [math.inf])


def test_mean_width_value() -> None:
    assert mean_interval_width([0.0, 1.0], [2.0, 4.0]) == pytest.approx(2.5)


def test_crps_degenerate_is_absolute_error() -> None:
    # A point mass at q on a symmetric midpoint grid -> CRPS == |y - q|.
    levels = [(k + 0.5) / 200 for k in range(200)]
    q = 3.0
    quantiles = [q] * 200
    assert crps_from_quantiles(5.0, levels, quantiles) == pytest.approx(2.0, abs=1e-9)


def test_crps_rewards_sharper_correct_forecast() -> None:
    levels = [(k + 0.5) / 100 for k in range(100)]
    y = 0.0
    tight = [(-1.0 + 2.0 * t) for t in levels]  # spread ~[-1, 1] around 0
    wide = [(-5.0 + 10.0 * t) for t in levels]  # spread ~[-5, 5] around 0
    assert crps_from_quantiles(y, levels, tight) < crps_from_quantiles(y, levels, wide)


def test_brier_score() -> None:
    assert brier_score([0.9, 0.2], [True, False]) == pytest.approx((0.01 + 0.04) / 2)


def test_log_loss_clips_extremes() -> None:
    # A confident wrong call is heavily but finitely penalized.
    val = log_loss([0.0], [True])
    assert math.isfinite(val) and val > 30


def test_coverage_by_group() -> None:
    groups = ["A", "A", "B", "B", "B"]
    covered = [True, False, True, True, False]
    table = coverage_by_group(groups, covered)
    assert table["A"] == (2, pytest.approx(0.5))
    assert table["B"] == (3, pytest.approx(2 / 3))


def test_coverage_row_gap_and_width_excludes_abstentions() -> None:
    y = [0.0, 0.0, 0.0]
    lo = [-1.0, -2.0, -math.inf]
    hi = [1.0, 2.0, math.inf]  # third is an abstention (unbounded)
    row = coverage_row("incR|polled3+|redrawn|HOUSE", alpha=0.2, y=y, lo=lo, hi=hi)
    assert row.n == 3
    assert row.empirical_coverage == 1.0
    assert row.nominal_coverage == pytest.approx(0.8)
    assert row.coverage_gap == pytest.approx(0.2)
    assert row.mean_width == pytest.approx(3.0)  # mean of finite widths 2 and 4


def test_coverage_row_all_abstained_has_no_width() -> None:
    row = coverage_row("g", 0.1, [0.0], [-math.inf], [math.inf])
    assert row.mean_width is None


def test_seat_count_error_signed() -> None:
    assert seat_count_error(218.4, 215) == pytest.approx(3.4)
