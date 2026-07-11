"""Adaptive Conformal Inference — online alpha-adjustment (Phase 4, live).

Once the daily update stream is live, ACI (Gibbs & Candès, 2021) self-corrects
empirical coverage as the cycle drifts:

    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

where ``err_t`` is 1 if the previous interval missed, else 0. The long-run
average miscoverage converges to ``alpha_target`` regardless of drift, so the
realized coverage tracks the nominal ``1 - alpha_target`` even when the score
distribution is nonstationary. Verified on simulated drift in
``tests/test_conformal_aci.py``.

Pure-Python and stateless-by-value: :class:`ACIState` carries the current
``alpha`` so the nightly job can persist and resume it across runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

STAGE = "conformal.aci"

DEFAULT_GAMMA = 0.03


def update_alpha(
    alpha_t: float,
    covered: bool,
    *,
    alpha_target: float,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """One ACI step. ``covered`` is the realized coverage of the *last* interval.

    ``err_t = 0`` when covered, ``1`` when missed. The result is clamped to
    ``[0, 1]``: at ``alpha = 0`` the interval is maximally wide (never misses), at
    ``alpha = 1`` maximally narrow — the two absorbing guards ACI needs so a run of
    misses (or hits) can't push the working level out of range.
    """
    if not 0.0 <= alpha_target <= 1.0:
        raise ValueError(f"alpha_target must be in [0, 1]; got {alpha_target}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive; got {gamma}")
    err_t = 0.0 if covered else 1.0
    nxt = alpha_t + gamma * (alpha_target - err_t)
    return min(1.0, max(0.0, nxt))


@dataclass
class ACIState:
    """Resumable ACI level for one (race-group, nominal-alpha) stream."""

    alpha: float
    alpha_target: float
    gamma: float = DEFAULT_GAMMA

    def step(self, covered: bool) -> float:
        """Advance one observation, mutating and returning the working ``alpha``."""
        self.alpha = update_alpha(
            self.alpha, covered, alpha_target=self.alpha_target, gamma=self.gamma
        )
        return self.alpha


def run_aci(
    covered_stream: Iterable[bool],
    *,
    alpha_target: float,
    gamma: float = DEFAULT_GAMMA,
    alpha0: float | None = None,
) -> list[float]:
    """Roll ACI over a coverage stream, returning the working alpha *before* each step.

    Element ``t`` is the level used to build the interval that was then scored by
    ``covered_stream[t]``. Starts from ``alpha0`` (default ``alpha_target``).
    """
    alpha = alpha_target if alpha0 is None else alpha0
    state = ACIState(alpha=alpha, alpha_target=alpha_target, gamma=gamma)
    levels: list[float] = []
    for covered in covered_stream:
        levels.append(state.alpha)
        state.step(covered)
    return levels


def realized_miscoverage(covered_stream: Sequence[bool]) -> float:
    """Empirical miscoverage rate of a coverage stream (fraction that missed)."""
    if not covered_stream:
        raise ValueError("empty coverage stream")
    misses = sum(0 if c else 1 for c in covered_stream)
    return misses / len(covered_stream)
