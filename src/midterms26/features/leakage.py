"""Leakage guard — non-negotiable (project plan Phase 1).

Every feature carries an ``as_of`` date. The feature assembler takes
``(race_id, cutoff_date)`` and must refuse to read anything after the cutoff.
This module is the enforcement point and is unit-tested with property tests.

The guard is deliberately *strict*: it raises rather than silently dropping, so a
mis-specified query surfaces immediately instead of quietly leaking or quietly
producing an empty feature. Callers that legitimately want the pre-cutoff slice
use :func:`filter_as_of`, which enforces then filters.
"""

from __future__ import annotations

from datetime import date

import polars as pl


class LeakageError(ValueError):
    """Raised when data carries an ``as_of`` strictly after the cutoff date."""


def enforce_as_of(
    df: pl.DataFrame,
    cutoff: date,
    *,
    column: str = "as_of",
) -> pl.DataFrame:
    """Raise :class:`LeakageError` if any row's ``as_of`` is after ``cutoff``.

    Returns ``df`` unchanged when clean, so it composes as an assertion in a
    pipeline. A missing ``as_of`` column is itself a leak risk and is rejected.
    """
    if column not in df.columns:
        raise LeakageError(f"frame is missing required leakage column {column!r}")

    if df.height == 0:
        return df

    # Null as_of cannot be proven pre-cutoff -> treat as a leak.
    n_null = df.select(pl.col(column).is_null().sum()).item()
    if n_null:
        raise LeakageError(f"{n_null} row(s) have null {column!r}; cannot prove pre-cutoff")

    latest = df.select(pl.col(column).max()).item()
    if latest is not None and latest > cutoff:
        n_bad = df.select((pl.col(column) > cutoff).sum()).item()
        raise LeakageError(
            f"{n_bad} row(s) have {column} > cutoff {cutoff.isoformat()} "
            f"(latest={latest}); post-cutoff data must never reach features"
        )
    return df


def filter_as_of(
    df: pl.DataFrame,
    cutoff: date,
    *,
    column: str = "as_of",
) -> pl.DataFrame:
    """Return only rows at or before ``cutoff`` (drops null ``as_of``)."""
    if column not in df.columns:
        raise LeakageError(f"frame is missing required leakage column {column!r}")
    return df.filter(pl.col(column).is_not_null() & (pl.col(column) <= cutoff))
