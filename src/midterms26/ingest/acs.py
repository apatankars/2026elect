"""ACS 5-year demographics ingest (Phase 1b).

Source: Census ACS 5-yr via the census API, pre-shaped into tidy CSVs. Raw files
in ``data/raw/acs/*.csv`` with columns:

    geoid, geo_level, vintage, release_date, variable, value

Stored at source geography (block group / tract / CD); the Phase 2 geo pipeline
aggregates onto enacted 2026 lines with ``maup``. ``as_of`` = the vintage's
public release date, so a not-yet-released vintage can never pass a cutoff.

Contract (enforced in :func:`normalize_acs`): known ``geo_level``; non-null
``value``; ``pct_*`` variables must lie in [0, 100].
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

STAGE = "ingest.acs"
SOURCE = "acs"
log = get_logger(STAGE)

GEO_LEVELS = frozenset({"BG", "TRACT", "CD"})


def normalize_acs(df: pl.DataFrame) -> pl.DataFrame:
    """Validate tidy ACS rows and rename ``release_date`` to the leakage key."""
    required = {"geoid", "geo_level", "vintage", "release_date", "variable", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"acs missing columns: {sorted(missing)}")

    out = df.with_columns(
        pl.col("geoid").cast(pl.Utf8),
        pl.col("geo_level").cast(pl.Utf8).str.to_uppercase(),
        pl.col("vintage").cast(pl.Utf8),
        date_col("release_date").alias("as_of"),
        pl.col("variable").cast(pl.Utf8),
        pl.col("value").cast(pl.Float64),
    )

    bad_level = out.filter(~pl.col("geo_level").is_in(sorted(GEO_LEVELS)))
    if bad_level.height:
        levels = sorted(bad_level["geo_level"].unique().to_list())
        raise ValueError(f"acs unknown geo_level(s): {levels}")

    n_null = out.select(pl.col("value").is_null().sum()).item()
    if n_null:
        raise ValueError(f"acs has {n_null} null value row(s); drop them upstream")

    bad_pct = out.filter(
        pl.col("variable").str.starts_with("pct_")
        & ((pl.col("value") < 0.0) | (pl.col("value") > 100.0))
    )
    if bad_pct.height:
        raise ValueError(f"acs has {bad_pct.height} pct_* row(s) outside [0, 100]")

    return out.select("geoid", "geo_level", "vintage", "as_of", "variable", "value")


def run(ctx: RunContext) -> StepResult:
    """Parse cached ACS files and upsert the acs_demographics table."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.csv")
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached acs files in {ctx.raw_dir / SOURCE}; add files or enable allow_fetch."
        )
    df = pl.concat(
        [pl.read_csv(p, infer_schema_length=2000) for p in paths], how="vertical_relaxed"
    )
    rows = normalize_acs(df)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n = base.upsert_dataframe(conn, "acs_demographics", rows)
    return StepResult(node=STAGE, rows=n, detail=f"{n} acs rows")


dry_run = stub(STAGE, rows=250_000, detail="ACS 5-yr tidy demographics (stub)")
