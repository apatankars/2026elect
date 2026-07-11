"""MIT returns normalization: aggregation, mode dedup, margin, uncontested."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from midterms26.ingest.results import normalize_returns

FIXTURES = Path(__file__).parent / "fixtures" / "raw" / "mit_edsl"


def _house() -> tuple[pl.DataFrame, pl.DataFrame]:
    return normalize_returns(pl.read_csv(FIXTURES / "house_2018.csv"))


def test_mode_dedup_prefers_total() -> None:
    _, results = _house()
    ca12 = results.filter(pl.col("race_id") == "2018-HOUSE-CA-12").row(0, named=True)
    # TOTAL rows (D=200, R=120) win over ELECTION DAY rows.
    assert ca12["dem_votes"] == 200
    assert ca12["rep_votes"] == 120
    assert ca12["two_party_margin"] == 200 / (200 + 120) * 100 - 120 / (200 + 120) * 100


def test_president_and_primary_rows_excluded() -> None:
    races, _ = _house()
    ids = set(races["race_id"].to_list())
    assert ids == {"2018-HOUSE-CA-12", "2018-HOUSE-CA-13"}


def test_uncontested_flagged() -> None:
    races, _ = _house()
    ca13 = races.filter(pl.col("race_id") == "2018-HOUSE-CA-13").row(0, named=True)
    assert ca13["is_uncontested"] is True
    ca12 = races.filter(pl.col("race_id") == "2018-HOUSE-CA-12").row(0, named=True)
    assert ca12["is_uncontested"] is False


def test_senate_ignores_district_and_buckets_third_party() -> None:
    _, results = normalize_returns(pl.read_csv(FIXTURES / "senate_2018.csv"))
    tx = results.filter(pl.col("race_id") == "2018-SENATE-TX-SEN").row(0, named=True)
    assert tx["dem_votes"] == 400
    assert tx["rep_votes"] == 600
    # Libertarian counts toward total but not two-party margin.
    assert tx["total_votes"] == 1050
    assert tx["two_party_margin"] == (400 - 600) / 1000 * 100


def test_summation_when_no_total_mode() -> None:
    # No TOTAL rows -> modes are summed.
    df = pl.DataFrame(
        {
            "year": [2020, 2020, 2020],
            "state_po": ["NY", "NY", "NY"],
            "office": ["US HOUSE"] * 3,
            "district": ["01", "01", "01"],
            "stage": ["GEN"] * 3,
            "special": [False] * 3,
            "party": ["DEMOCRAT", "DEMOCRAT", "REPUBLICAN"],
            "candidatevotes": [100, 50, 200],
            "mode": ["ELECTION DAY", "ABSENTEE", "ELECTION DAY"],
        }
    )
    _, results = normalize_returns(df)
    row = results.row(0, named=True)
    assert row["dem_votes"] == 150
    assert row["rep_votes"] == 200
