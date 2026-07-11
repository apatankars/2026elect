"""Adaptive Conformal Inference — update direction, clamping, drift convergence."""

from __future__ import annotations

import random

import pytest

from midterms26.conformal.aci import (
    ACIState,
    realized_miscoverage,
    run_aci,
    update_alpha,
)


def test_miss_lowers_alpha_hit_raises_it() -> None:
    a = 0.1
    after_miss = update_alpha(a, covered=False, alpha_target=0.1, gamma=0.03)
    after_hit = update_alpha(a, covered=True, alpha_target=0.1, gamma=0.03)
    assert after_miss < a  # missed -> shrink alpha -> wider next interval
    assert after_hit > a  # covered -> grow alpha -> tighter next interval


def test_alpha_clamped_to_unit_interval() -> None:
    # A long run of misses cannot push alpha below 0.
    a = 0.05
    for _ in range(100):
        a = update_alpha(a, covered=False, alpha_target=0.1, gamma=0.5)
    assert a == 0.0
    # A long run of hits cannot push it above 1.
    a = 0.9
    for _ in range(100):
        a = update_alpha(a, covered=True, alpha_target=0.9, gamma=0.5)
    assert a == 1.0


def test_state_step_matches_functional_update() -> None:
    st = ACIState(alpha=0.2, alpha_target=0.2, gamma=0.03)
    got = st.step(covered=False)
    assert got == update_alpha(0.2, covered=False, alpha_target=0.2, gamma=0.03)
    assert st.alpha == got


def test_run_aci_reports_pre_step_levels() -> None:
    levels = run_aci([True, False, True], alpha_target=0.2, gamma=0.05)
    assert len(levels) == 3
    assert levels[0] == 0.2  # starts at alpha_target by default


def test_bad_params_rejected() -> None:
    with pytest.raises(ValueError):
        update_alpha(0.1, covered=True, alpha_target=1.5, gamma=0.03)
    with pytest.raises(ValueError):
        update_alpha(0.1, covered=True, alpha_target=0.1, gamma=0.0)
    with pytest.raises(ValueError):
        realized_miscoverage([])


def test_converges_to_target_under_drift() -> None:
    """Realized miscoverage tracks alpha_target even as the miss process drifts.

    Coverage model: at working level ``alpha`` the interval misses with
    probability ``clip(alpha + drift_t)`` — a moving bias ACI must chase.
    """
    rng = random.Random(3)
    alpha_target = 0.2
    gamma = 0.03
    state = ACIState(alpha=alpha_target, alpha_target=alpha_target, gamma=gamma)
    covered_stream: list[bool] = []
    n = 30000
    # Bias kept small enough that the equilibrium level (target - bias) and the
    # miss probability stay well inside (0, 1) — no clamping to distort the mean.
    for t in range(n):
        bias = 0.05 * (1 if (t // 3000) % 2 == 0 else -1)  # square-wave drift
        miss_prob = min(0.99, max(0.01, state.alpha + bias))
        covered = rng.random() >= miss_prob
        covered_stream.append(covered)
        state.step(covered)
    # Discard burn-in; long-run miscoverage should sit near the target.
    tail = covered_stream[3000:]
    assert realized_miscoverage(tail) == pytest.approx(alpha_target, abs=0.02)
