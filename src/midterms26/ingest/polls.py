"""Polls ingest.

VoteHub for the live cycle; 538 GitHub archives for 2018-2024; BPR/hand-collected
for older cycles where available. Each poll row is normalized into ``polls`` with
an ``as_of`` release date (the leakage key). Heavy meta-analysis (weighted margin,
herding statistic) lives in the feature phase; ingest only normalizes rows.

Risk (plan §4): pre-2018 archives are patchy -> thin "polled" strata in old
cycles; calibration may collapse those strata.

Cached files live in ``data/raw/polls/``.
"""

from __future__ import annotations

import hashlib

import polars as pl

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.ingest import base
from midterms26.ingest.normalize import date_col
from midterms26.logging import get_logger
from midterms26.stubs import stub
from midterms26.warehouse import connect, init_schema

STAGE = "ingest.polls"
SOURCE = "polls"
log = get_logger(STAGE)


def _poll_id(row: dict[str, object]) -> str:
    key = f"{row.get('pollster_id')}|{row.get('race_id')}|{row.get('field_end')}|{row.get('sample_n')}|{row.get('margin')}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def normalize_polls(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw poll rows into the ``polls`` schema.

    Requires at least ``race_id`` and ``as_of``. ``margin`` is derived from
    ``dem_share``/``rep_share`` when absent. A stable ``poll_id`` is synthesized
    when missing so re-ingest is idempotent.
    """
    required = {"race_id", "as_of"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"polls missing columns: {sorted(missing)}")

    for opt, dtype in {
        "pollster_id": pl.Utf8,
        "field_start": pl.Date,
        "field_end": pl.Date,
        "mode": pl.Utf8,
        "population": pl.Utf8,
        "sample_n": pl.Int64,
        "dem_share": pl.Float64,
        "rep_share": pl.Float64,
        "margin": pl.Float64,
    }.items():
        if opt not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(opt))

    df = df.with_columns(
        date_col("as_of"),
        date_col("field_start"),
        date_col("field_end"),
        pl.col("sample_n").cast(pl.Int64, strict=False),
        pl.col("dem_share").cast(pl.Float64, strict=False),
        pl.col("rep_share").cast(pl.Float64, strict=False),
        pl.col("margin").cast(pl.Float64, strict=False),
    ).with_columns(
        pl.when(
            pl.col("margin").is_null()
            & pl.col("dem_share").is_not_null()
            & pl.col("rep_share").is_not_null()
        )
        .then(pl.col("dem_share") - pl.col("rep_share"))
        .otherwise(pl.col("margin"))
        .alias("margin")
    )

    if "poll_id" not in df.columns:
        df = df.with_columns(
            pl.struct(["pollster_id", "race_id", "field_end", "sample_n", "margin"])
            .map_elements(_poll_id, return_dtype=pl.Utf8)
            .alias("poll_id")
        )

    return df.select(
        "poll_id",
        "race_id",
        "pollster_id",
        "field_start",
        "field_end",
        "as_of",
        "mode",
        "population",
        "sample_n",
        "dem_share",
        "rep_share",
        "margin",
    ).unique(subset=["poll_id"], keep="last")


def run(ctx: RunContext) -> StepResult:
    """Parse cached polls and upsert the polls table."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.csv")
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached polls in {ctx.raw_dir / SOURCE}; add files or enable allow_fetch."
        )
    df = pl.concat(
        [pl.read_csv(p, infer_schema_length=2000) for p in paths], how="vertical_relaxed"
    )
    rows = normalize_polls(df)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n = base.upsert_dataframe(conn, "polls", rows)
    return StepResult(node=STAGE, rows=n, detail=f"{n} polls")


dry_run = stub(STAGE, rows=28_000, detail="VoteHub + 538 archives 2018-2026 (stub)")
