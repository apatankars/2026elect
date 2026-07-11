"""Unit tests for the conformal primitives that already carry logic in Phase 0."""

from __future__ import annotations

import pytest

from midterms26.conformal.abstain import decide
from midterms26.conformal.mondrian import group_id


def test_group_id_composes_canonical_order() -> None:
    assert group_id("incR", "polled3+", "redrawn", "HOUSE") == "incR|polled3+|redrawn|HOUSE"


@pytest.mark.parametrize(
    "args",
    [
        ("incX", "polled3+", "redrawn", "HOUSE"),
        ("incR", "sometimes", "redrawn", "HOUSE"),
        ("incR", "polled3+", "wiggly", "HOUSE"),
        ("incR", "polled3+", "redrawn", "MAYOR"),
    ],
)
def test_group_id_rejects_invalid_levels(args: tuple[str, str, str, str]) -> None:
    with pytest.raises(ValueError):
        group_id(*args)


def test_abstain_on_thin_bin() -> None:
    d = decide(interval_width_80=5.0, bin_calibration_n=3, tau=40.0, n_min=10)
    assert d.abstain and d.reason == "bin<10"


def test_abstain_on_wide_interval() -> None:
    d = decide(interval_width_80=50.0, bin_calibration_n=100, tau=40.0, n_min=10)
    assert d.abstain and d.reason == "width>40"


def test_no_abstain_when_confident() -> None:
    d = decide(interval_width_80=12.0, bin_calibration_n=100, tau=40.0, n_min=10)
    assert not d.abstain and d.reason is None


def test_thin_bin_takes_precedence_over_width() -> None:
    d = decide(interval_width_80=99.0, bin_calibration_n=1, tau=40.0, n_min=10)
    assert d.reason == "bin<10"
