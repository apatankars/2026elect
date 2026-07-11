"""FEC snapshots: individual-only, self/PAC excluded, cumulative, quarter-end as_of."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from midterms26.ingest.fec import snapshot_individual_donations

FIXTURE = Path(__file__).parent / "fixtures" / "raw" / "fec" / "receipts_r1.csv"


def _snaps() -> pl.DataFrame:
    return snapshot_individual_donations(pl.read_csv(FIXTURE)).sort("report_quarter")


def test_excludes_self_and_pac_money() -> None:
    snaps = _snaps()
    # Individual, non-self only: Q1 = 150 + 500 = 650 (PAC 1000 and self 2000 excluded).
    q1 = snaps.filter(pl.col("report_quarter") == "2018Q1").row(0, named=True)
    assert q1["individual_donations"] == pytest.approx(650.0)


def test_cumulative_across_quarters() -> None:
    snaps = _snaps()
    q2 = snaps.filter(pl.col("report_quarter") == "2018Q2").row(0, named=True)
    assert q2["individual_donations"] == pytest.approx(750.0)  # 650 + 100


def test_small_dollar_share_cumulative() -> None:
    snaps = _snaps()
    q1 = snaps.filter(pl.col("report_quarter") == "2018Q1").row(0, named=True)
    q2 = snaps.filter(pl.col("report_quarter") == "2018Q2").row(0, named=True)
    assert q1["small_dollar_share"] == pytest.approx(150 / 650)
    assert q2["small_dollar_share"] == pytest.approx(250 / 750)


def test_as_of_is_quarter_end() -> None:
    snaps = _snaps()
    q1 = snaps.filter(pl.col("report_quarter") == "2018Q1").row(0, named=True)
    q2 = snaps.filter(pl.col("report_quarter") == "2018Q2").row(0, named=True)
    assert q1["as_of"] == date(2018, 3, 31)
    assert q2["as_of"] == date(2018, 6, 30)


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        snapshot_individual_donations(pl.DataFrame({"race_id": ["x"]}))
