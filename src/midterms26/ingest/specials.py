"""Special elections ingest + special-election swing index.

Source: Ballotpedia scrape + manual verification. Per special we store seat PVI,
result margin, turnout, and the derived ``overperformance`` = result_margin −
seat_pvi (how much the Democrat beat the seat's partisan baseline).

The per-cycle "special election swing index" = pooled overperformance vs. seat
partisanship, weighted by recency and turnout — a top-tier leading indicator. It
is computed as-of a cutoff so it never uses specials after the forecast date.

Real execution runs against cached files in ``data/raw/specials/``.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.ingest import base
from midterms26.ingest.normalize import date_col
from midterms26.logging import get_logger
from midterms26.stubs import stub
from midterms26.warehouse import connect, init_schema

STAGE = "ingest.specials"
SOURCE = "specials"
log = get_logger(STAGE)

DEFAULT_HALF_LIFE_DAYS = 180.0


def normalize_specials(df: pl.DataFrame) -> pl.DataFrame:
    """Validate raw specials and compute per-special overperformance."""
    required = {"special_id", "cycle", "election_date", "seat_pvi", "result_margin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"specials missing columns: {sorted(missing)}")

    out = df.with_columns(
        date_col("election_date"),
        pl.col("seat_pvi").cast(pl.Float64),
        pl.col("result_margin").cast(pl.Float64),
        (pl.col("result_margin").cast(pl.Float64) - pl.col("seat_pvi").cast(pl.Float64)).alias(
            "overperformance"
        ),
    )
    for opt in ("race_id", "state", "district", "turnout"):
        if opt not in out.columns:
            out = out.with_columns(pl.lit(None).alias(opt))
    return out.select(
        "special_id",
        "race_id",
        "cycle",
        "state",
        "district",
        "election_date",
        "seat_pvi",
        "result_margin",
        "turnout",
        "overperformance",
    )


def swing_index(
    specials: pl.DataFrame,
    as_of: date,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    cycle: int | None = None,
) -> float | None:
    """Pooled overperformance as-of ``as_of``, weighted by recency and turnout.

    Recency weight = 2 ** (−age_days / half_life). Turnout weight = sqrt(turnout)
    (null turnout -> weight 1). Only specials with ``election_date <= as_of`` are
    used, so the index is leakage-safe. Returns ``None`` if no eligible specials.
    """
    df = specials.filter(pl.col("election_date") <= as_of)
    if cycle is not None:
        df = df.filter(pl.col("cycle") == cycle)
    if df.height == 0:
        return None

    num = 0.0
    den = 0.0
    for row in df.iter_rows(named=True):
        age = (as_of - row["election_date"]).days
        recency = 2.0 ** (-age / half_life_days)
        turnout = row["turnout"]
        turnout_w = math.sqrt(turnout) if turnout and turnout > 0 else 1.0
        w = recency * turnout_w
        num += w * row["overperformance"]
        den += w
    return num / den if den > 0 else None


def run(ctx: RunContext) -> StepResult:
    """Parse cached specials and upsert the specials table."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.csv")
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached specials in {ctx.raw_dir / SOURCE}; add files or enable allow_fetch."
        )
    df = pl.concat(
        [pl.read_csv(p, infer_schema_length=2000) for p in paths], how="vertical_relaxed"
    )
    rows = normalize_specials(df)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n = base.upsert_dataframe(conn, "specials", rows)
    return StepResult(node=STAGE, rows=n, detail=f"{n} specials")


dry_run = stub(STAGE, rows=220, detail="Ballotpedia specials + swing index (stub)")
