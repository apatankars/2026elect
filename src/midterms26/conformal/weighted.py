"""Weighted (nonexchangeable) conformal quantiles (Phase 4).

Barber-Candes-Ramdas-Tibshirani weighted conformal with exponential decay over
cycles so 2022 counts more than 2006 — robustness to distribution drift is the
whole point. Implements the weighted empirical quantile of nonconformity scores
with a point mass at ``+inf`` for the test point.

The estimator here is deliberately pure-Python (no numpy) so it runs in the light
CI stack and stays trivially auditable — it is the core of the project's coverage
guarantee. Edge cases (all-equal weights -> ordinary conformal; single
calibration point; total weight below ``1 - alpha`` -> ``+inf``) are covered in
``tests/test_conformal_weighted.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

STAGE = "conformal.weighted"


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")


def weighted_conformal_quantile(
    scores: Sequence[float],
    weights: Sequence[float] | None,
    alpha: float,
    *,
    test_weight: float | None = None,
) -> float:
    """Weighted ``(1 - alpha)`` empirical quantile of nonconformity ``scores``.

    Implements the Barber et al. (2023) nonexchangeable construction: calibration
    point ``i`` carries normalized mass ``w_i / (sum_j w_j + w_test)`` and the
    test point carries the remaining mass at ``+inf``. The returned value is the
    smallest score whose cumulative normalized weight reaches ``1 - alpha``; if the
    calibration mass never reaches that level, the quantile falls on the test
    point's ``+inf`` mass and we return ``math.inf`` (an infinite interval — the
    honest answer, which downstream turns into a "No Call").

    * ``weights=None`` gives every calibration point weight ``1`` and
      ``test_weight`` defaults to ``1`` — exactly the ordinary split-conformal
      finite-sample ``ceil((n + 1)(1 - alpha)) / n`` quantile.
    * A single calibration point yields either that point's score (if its mass
      covers ``1 - alpha``) or ``+inf``.
    """
    _validate_alpha(alpha)
    n = len(scores)
    if n == 0:
        return math.inf

    if weights is None:
        w = [1.0] * n
    else:
        if len(weights) != n:
            raise ValueError(f"weights length {len(weights)} != scores length {n}")
        w = [float(x) for x in weights]
    if any(x < 0 for x in w):
        raise ValueError("weights must be non-negative")

    tw = float(test_weight) if test_weight is not None else 1.0
    if tw < 0:
        raise ValueError("test_weight must be non-negative")

    total = math.fsum(w) + tw
    if total <= 0:
        return math.inf

    order = sorted(range(n), key=lambda i: scores[i])
    target = 1.0 - alpha
    cumulative = 0.0
    for i in order:
        cumulative += w[i] / total
        # Guard against float drift right at the boundary.
        if cumulative >= target - 1e-12:
            return float(scores[i])
    # All calibration mass exhausted without reaching 1 - alpha: the quantile
    # sits on the test point's +inf mass.
    return math.inf


def exp_decay_weights(
    cycles: Sequence[int],
    *,
    reference_cycle: int | None = None,
    half_life: float = 8.0,
) -> list[float]:
    """Exponential-decay calibration weights: recent cycles count more.

    ``weight = 0.5 ** ((reference_cycle - cycle) / half_life)`` so a point
    ``half_life`` years older than the reference carries half the weight. The
    reference defaults to the most recent cycle present, giving it weight ``1``.
    Future cycles (``cycle > reference``) are clamped to weight ``1`` rather than
    up-weighted — calibration never extrapolates forward.

    ``half_life`` in the same units as ``cycles`` (years). Passing a huge
    ``half_life`` recovers uniform weights, i.e. ordinary exchangeable conformal.
    """
    if half_life <= 0:
        raise ValueError(f"half_life must be positive; got {half_life}")
    if not cycles:
        return []
    ref = max(cycles) if reference_cycle is None else reference_cycle
    out: list[float] = []
    for c in cycles:
        age = ref - c
        out.append(1.0 if age <= 0 else 0.5 ** (age / half_life))
    return out
