"""Leakage guard property tests — the guard MUST throw on post-cutoff data."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from midterms26.features.leakage import LeakageError, enforce_as_of, filter_as_of

CUTOFF = date(2022, 11, 8)

_dates = st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 1, 1))


@given(rows=st.lists(_dates, min_size=1, max_size=50))
def test_enforce_throws_iff_any_post_cutoff(rows: list[date]) -> None:
    df = pl.DataFrame({"as_of": rows, "value": list(range(len(rows)))})
    has_leak = any(d > CUTOFF for d in rows)
    if has_leak:
        with pytest.raises(LeakageError):
            enforce_as_of(df, CUTOFF)
    else:
        # Clean frame is returned unchanged.
        out = enforce_as_of(df, CUTOFF)
        assert out.height == df.height


def test_enforce_passes_on_boundary_date() -> None:
    df = pl.DataFrame({"as_of": [CUTOFF], "value": [1]})
    assert enforce_as_of(df, CUTOFF).height == 1


def test_enforce_rejects_missing_column() -> None:
    df = pl.DataFrame({"value": [1, 2]})
    with pytest.raises(LeakageError, match="missing required leakage column"):
        enforce_as_of(df, CUTOFF)


def test_enforce_rejects_null_as_of() -> None:
    df = pl.DataFrame({"as_of": [CUTOFF, None], "value": [1, 2]})
    with pytest.raises(LeakageError, match="null"):
        enforce_as_of(df, CUTOFF)


def test_empty_frame_is_clean() -> None:
    df = pl.DataFrame({"as_of": [], "value": []}, schema={"as_of": pl.Date, "value": pl.Int64})
    assert enforce_as_of(df, CUTOFF).height == 0


@given(rows=st.lists(_dates, min_size=0, max_size=50))
def test_filter_keeps_only_pre_cutoff_and_is_then_clean(rows: list[date]) -> None:
    df = pl.DataFrame(
        {"as_of": rows, "value": list(range(len(rows)))},
        schema={"as_of": pl.Date, "value": pl.Int64},
    )
    kept = filter_as_of(df, CUTOFF)
    assert kept.select((pl.col("as_of") > CUTOFF).any()).item() in (False, None)
    # After filtering, the strict guard must pass.
    enforce_as_of(kept, CUTOFF)


def test_filter_drops_future_rows() -> None:
    df = pl.DataFrame(
        {
            "as_of": [CUTOFF - timedelta(days=1), CUTOFF, CUTOFF + timedelta(days=1)],
            "value": [1, 2, 3],
        }
    )
    kept = filter_as_of(df, CUTOFF)
    assert kept.height == 2
