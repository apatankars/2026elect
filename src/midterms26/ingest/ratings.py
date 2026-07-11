"""Expert ratings ingest.

Cook, Inside Elections, Sabato — dated snapshots keyed by ``as_of`` so features
are as-of-date correct. Rating labels are mapped to a signed numeric scale
(+ = Democratic advantage) for use as a feature and for the ratings-only baseline.

Cached files live in ``data/raw/ratings/``.
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

STAGE = "ingest.ratings"
SOURCE = "ratings"
log = get_logger(STAGE)

# Signed 7-point scale, + = Democratic. Covers common label spellings.
RATING_SCALE: dict[str, float] = {
    "SAFE D": 3.0,
    "SOLID D": 3.0,
    "LIKELY D": 2.0,
    "LEAN D": 1.0,
    "TILT D": 0.5,
    "TOSS-UP": 0.0,
    "TOSSUP": 0.0,
    "TOSS UP": 0.0,
    "TILT R": -0.5,
    "LEAN R": -1.0,
    "LIKELY R": -2.0,
    "SAFE R": -3.0,
    "SOLID R": -3.0,
}


def _to_numeric(label: str | None) -> float | None:
    if label is None:
        return None
    return RATING_SCALE.get(label.strip().upper())


def normalize_ratings(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw ratings into the ``ratings`` schema with numeric scores."""
    required = {"race_id", "source", "as_of", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ratings missing columns: {sorted(missing)}")
    out = df.with_columns(
        date_col("as_of"),
        pl.col("rating").map_elements(_to_numeric, return_dtype=pl.Float64).alias("rating_numeric"),
    )
    unmapped = out.filter(pl.col("rating_numeric").is_null() & pl.col("rating").is_not_null())
    if unmapped.height:
        labels = unmapped.select(pl.col("rating").unique()).to_series().to_list()
        log.warning("ratings.unmapped_labels", labels=labels)
    return out.select("race_id", "source", "as_of", "rating", "rating_numeric").unique(
        subset=["race_id", "source", "as_of"], keep="last"
    )


def run(ctx: RunContext) -> StepResult:
    """Parse cached ratings and upsert the ratings table."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.csv")
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached ratings in {ctx.raw_dir / SOURCE}; add files or enable allow_fetch."
        )
    df = pl.concat(
        [pl.read_csv(p, infer_schema_length=2000) for p in paths], how="vertical_relaxed"
    )
    rows = normalize_ratings(df)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n = base.upsert_dataframe(conn, "ratings", rows)
    return StepResult(node=STAGE, rows=n, detail=f"{n} rating snapshots")


dry_run = stub(STAGE, rows=9_800, detail="Cook/InsideElections/Sabato dated snapshots (stub)")
