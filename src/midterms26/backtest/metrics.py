"""Scoring metrics for the LOCO backtest + eval-gated CI (Phase 5).

Pure-Python (no numpy) so the eval gate runs in the light CI stack. These are the
functions that turn a set of conformalized predictions + realized margins into the
headline honesty table: empirical-vs-nominal coverage (overall and per Mondrian
group), interval width, CRPS, Brier, and seat-count error.

The framing the project commits to: the claim is *valid coverage*, not universally
best Brier — so ``coverage_gap`` (empirical − nominal) is the primary number and is
reported alongside sharpness (width) rather than instead of it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from midterms26.models.stack import pinball_loss

__all__ = [
    "CoverageRow",
    "brier_score",
    "coverage_by_group",
    "coverage_row",
    "crps_from_quantiles",
    "interval_coverage",
    "log_loss",
    "mean_interval_width",
    "seat_count_error",
]


def interval_coverage(y: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> float:
    """Fraction of realized ``y`` inside ``[lo, hi]``. Infinite bounds count as covered."""
    n = len(y)
    if not (len(lo) == len(hi) == n):
        raise ValueError("y, lo, hi must be the same length")
    if n == 0:
        raise ValueError("empty coverage set")
    hits = sum(1 for i in range(n) if lo[i] <= y[i] <= hi[i])
    return hits / n


def mean_interval_width(lo: Sequence[float], hi: Sequence[float]) -> float:
    """Mean finite interval width; raises if any interval is unbounded.

    Abstentions (unbounded "No Call" intervals) must be excluded upstream — an
    infinite width is not a sharpness number, it is a refusal to predict.
    """
    n = len(lo)
    if len(hi) != n:
        raise ValueError("lo and hi must be the same length")
    if n == 0:
        raise ValueError("empty width set")
    widths = [hi[i] - lo[i] for i in range(n)]
    if any(not math.isfinite(w) for w in widths):
        raise ValueError("unbounded interval in width set; drop abstentions first")
    return math.fsum(widths) / n


def crps_from_quantiles(y: float, levels: Sequence[float], quantiles: Sequence[float]) -> float:
    """CRPS approximated from a predictive quantile grid.

    ``CRPS(F, y) = 2 * integral_0^1 pinball_tau(y, F^{-1}(tau)) dtau``; here it is
    the midpoint average over the supplied ``levels`` times 2. Exact for a
    degenerate forecast on a symmetric grid (reduces to ``|y - q|``), and
    convergent to the true CRPS as the grid densifies.
    """
    m = len(levels)
    if len(quantiles) != m:
        raise ValueError("levels and quantiles must be the same length")
    if m == 0:
        raise ValueError("empty quantile grid")
    return 2.0 * math.fsum(pinball_loss(y, quantiles[k], levels[k]) for k in range(m)) / m


def brier_score(prob: Sequence[float], outcome: Sequence[bool]) -> float:
    """Mean squared error of probabilistic win calls: ``mean((p - 1{win})^2)``."""
    n = len(prob)
    if len(outcome) != n:
        raise ValueError("prob and outcome must be the same length")
    if n == 0:
        raise ValueError("empty Brier set")
    return math.fsum((prob[i] - (1.0 if outcome[i] else 0.0)) ** 2 for i in range(n)) / n


def log_loss(prob: Sequence[float], outcome: Sequence[bool], *, eps: float = 1e-15) -> float:
    """Mean binary cross-entropy; probabilities clipped to ``[eps, 1-eps]``."""
    n = len(prob)
    if len(outcome) != n:
        raise ValueError("prob and outcome must be the same length")
    if n == 0:
        raise ValueError("empty log-loss set")
    total = 0.0
    for i in range(n):
        p = min(1.0 - eps, max(eps, prob[i]))
        total += -math.log(p) if outcome[i] else -math.log(1.0 - p)
    return total / n


def seat_count_error(predicted_seats: float, actual_seats: int) -> float:
    """Signed seat-count error (predicted − actual); a summary of the joint sim."""
    return predicted_seats - actual_seats


@dataclass(frozen=True)
class CoverageRow:
    """Empirical-vs-nominal coverage for one (group, alpha) cell."""

    group: str
    alpha: float
    n: int
    empirical_coverage: float
    mean_width: float | None  # None when every interval in the cell abstained

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def coverage_gap(self) -> float:
        """Empirical − nominal. Negative = under-coverage (the failure that matters)."""
        return self.empirical_coverage - self.nominal_coverage


def coverage_row(
    group: str,
    alpha: float,
    y: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
) -> CoverageRow:
    """Build a :class:`CoverageRow`, excluding unbounded intervals from the width."""
    finite = [i for i in range(len(y)) if math.isfinite(hi[i] - lo[i])]
    width = (
        mean_interval_width([lo[i] for i in finite], [hi[i] for i in finite]) if finite else None
    )
    return CoverageRow(
        group=group,
        alpha=alpha,
        n=len(y),
        empirical_coverage=interval_coverage(y, lo, hi),
        mean_width=width,
    )


def coverage_by_group(
    groups: Sequence[str],
    covered: Sequence[bool],
) -> dict[str, tuple[int, float]]:
    """Per-group ``(n, empirical_coverage)`` — the Mondrian audit.

    Coverage must hold *within* each group, so this is the table the calibration
    dashboard renders and the eval gate checks for per-stratum under-coverage.
    """
    n = len(groups)
    if len(covered) != n:
        raise ValueError("groups and covered must be the same length")
    counts: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    for g, c in zip(groups, covered, strict=True):
        counts[g] += 1
        hits[g] += 1 if c else 0
    return {g: (counts[g], hits[g] / counts[g]) for g in counts}
