"""Specials: overperformance + leakage-safe recency/turnout-weighted swing index."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from midterms26.ingest.specials import normalize_specials, swing_index

FIXTURE = Path(__file__).parent / "fixtures" / "raw" / "specials" / "specials_2022.csv"


def _specials() -> pl.DataFrame:
    return normalize_specials(pl.read_csv(FIXTURE))


def test_overperformance_is_margin_minus_pvi() -> None:
    df = _specials()
    s1 = df.filter(pl.col("special_id") == "S1").row(0, named=True)
    assert s1["overperformance"] == pytest.approx(8.0)  # -2 - (-10)
    s2 = df.filter(pl.col("special_id") == "S2").row(0, named=True)
    assert s2["overperformance"] == pytest.approx(4.0)  # 9 - 5


def test_swing_index_excludes_future_specials() -> None:
    df = _specials()
    # As-of Oct 1 2022: only S1 (+8) and S2 (+4) count; S3 (+20, Dec) is excluded.
    idx = swing_index(df, date(2022, 10, 1))
    assert idx is not None
    assert 4.0 <= idx <= 8.0

    # Including the strongly-Dem December special pulls the index up.
    idx_later = swing_index(df, date(2023, 1, 1))
    assert idx_later is not None
    assert idx_later > idx


def test_swing_index_none_when_no_eligible() -> None:
    df = _specials()
    assert swing_index(df, date(2000, 1, 1)) is None


def test_swing_index_recency_weighting() -> None:
    # Two equal-turnout specials with opposite overperformance; the more recent
    # one (relative to as_of) dominates.
    df = pl.DataFrame(
        {
            "special_id": ["A", "B"],
            "race_id": [None, None],
            "cycle": [2022, 2022],
            "state": ["X", "Y"],
            "district": ["01", "01"],
            "election_date": [date(2022, 1, 1), date(2022, 9, 1)],
            "seat_pvi": [0.0, 0.0],
            "result_margin": [-10.0, 10.0],
            "turnout": [1000, 1000],
            "overperformance": [-10.0, 10.0],
        }
    )
    idx = swing_index(df, date(2022, 10, 1), half_life_days=60.0)
    assert idx is not None and idx > 0  # recent +10 outweighs older -10
