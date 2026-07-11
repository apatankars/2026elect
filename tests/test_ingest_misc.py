"""Polls, ratings, and econ normalization."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from midterms26.ingest.econ import normalize_series
from midterms26.ingest.polls import normalize_polls
from midterms26.ingest.ratings import normalize_ratings

RAW = Path(__file__).parent / "fixtures" / "raw"


def test_polls_derive_margin_and_synthesize_id() -> None:
    polls = normalize_polls(pl.read_csv(RAW / "polls" / "polls_2018.csv"))
    assert "poll_id" in polls.columns
    assert polls.select(pl.col("poll_id").is_null().any()).item() is False
    ca12 = polls.filter(pl.col("race_id") == "2018-HOUSE-CA-12").row(0, named=True)
    assert ca12["margin"] == pytest.approx(4.0)  # 49 - 45
    assert ca12["as_of"] == date(2018, 10, 5)


def test_polls_require_race_and_as_of() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        normalize_polls(pl.DataFrame({"race_id": ["x"]}))


def test_ratings_map_to_numeric_scale() -> None:
    ratings = normalize_ratings(pl.read_csv(RAW / "ratings" / "ratings_2018.csv"))
    by_key = {
        (r["race_id"], r["source"], r["as_of"]): r["rating_numeric"]
        for r in ratings.iter_rows(named=True)
    }
    assert by_key[("2018-HOUSE-CA-12", "Cook", date(2018, 9, 1))] == pytest.approx(1.0)
    assert by_key[("2018-HOUSE-CA-13", "Cook", date(2018, 9, 1))] == pytest.approx(3.0)
    assert by_key[("2018-SENATE-TX-SEN", "Sabato", date(2018, 9, 1))] == pytest.approx(0.0)


def test_ratings_dedupe_keeps_latest_snapshot() -> None:
    ratings = normalize_ratings(pl.read_csv(RAW / "ratings" / "ratings_2018.csv"))
    # Two Cook snapshots of CA-12 on different dates both survive (distinct as_of).
    ca12 = ratings.filter(pl.col("race_id") == "2018-HOUSE-CA-12")
    assert ca12.height == 2


def test_econ_normalize_series() -> None:
    rows = normalize_series(pl.read_csv(RAW / "econ" / "unrate.csv"), series_id="UNRATE")
    assert set(rows.columns) == {"series_id", "as_of", "value", "source"}
    assert rows.height == 3
    assert rows.select(pl.col("series_id").unique()).item() == "UNRATE"
