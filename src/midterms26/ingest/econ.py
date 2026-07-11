"""Economic / national environment ingest.

Sources: FRED (real disposable income growth, unemployment, gas prices) and
presidential approval (VoteHub / archives). Each source is a tidy time series
normalized into ``national_indicators`` as (series_id, as_of, value), so the
national-environment features are as-of-date correct.

Cached files live in ``data/raw/econ/<series>.csv`` with columns ``date,value``;
the series id is taken from the filename stem. Live FRED pulls (need an API key)
are gated by ``allow_fetch``.
"""

from __future__ import annotations

import polars as pl

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.ingest import base
from midterms26.ingest.normalize import date_col
from midterms26.logging import get_logger
from midterms26.stubs import stub
from midterms26.warehouse import connect, init_schema

STAGE = "ingest.econ"
SOURCE = "econ"
log = get_logger(STAGE)


def normalize_series(df: pl.DataFrame, series_id: str, source: str = "FRED") -> pl.DataFrame:
    """Normalize a (date, value) frame into ``national_indicators`` rows."""
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "value" not in cols:
        raise ValueError(f"series {series_id!r} needs date,value columns; got {df.columns}")
    return (
        df.rename({cols["date"]: "as_of", cols["value"]: "value"})
        .with_columns(
            date_col("as_of"),
            pl.col("value").cast(pl.Float64, strict=False),
            pl.lit(series_id).alias("series_id"),
            pl.lit(source).alias("source"),
        )
        .drop_nulls(subset=["value"])
        .select("series_id", "as_of", "value", "source")
    )


def run(ctx: RunContext) -> StepResult:
    """Parse cached econ series and upsert national_indicators."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.csv")
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached econ series in {ctx.raw_dir / SOURCE}; add files or enable allow_fetch."
        )
    frames = [normalize_series(pl.read_csv(p), series_id=p.stem.upper()) for p in paths]
    rows = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n = base.upsert_dataframe(conn, "national_indicators", rows)
    return StepResult(
        node=STAGE, rows=n, detail=f"{n} indicator observations from {len(paths)} series"
    )


dry_run = stub(STAGE, rows=2_400, detail="FRED macro + presidential approval series (stub)")
