"""FEC finance ingest — individual-donor signal, quarterly as-of snapshots.

Source: FEC itemized receipts by candidate-cycle. We compute INDIVIDUAL-DONOR
totals ONLY — excluding self-funding and PAC money, because individual-donor
fundraising is the validated signal. Snapshots are *cumulative through each
quarter end* with an ``as_of`` date, so a backtest at election E only sees money
reported before E.

Input receipt schema (normalized upstream / by the FEC connector):
  race_id, candidate_id, party, contributor_type ('IND'|'PAC'|...),
  is_self (bool, candidate self-funding), amount (float), receipt_date (date).

Real execution runs against cached parquet/csv in ``data/raw/fec/``; the live
API pull (needs an API key) is gated by ``allow_fetch``.
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

STAGE = "ingest.fec"
SOURCE = "fec"
log = get_logger(STAGE)

# Individual "small-dollar" contributions are conventionally < $200 (the FEC
# itemization threshold). Used for the small_dollar_share feature.
SMALL_DOLLAR_THRESHOLD = 200.0

_QUARTER_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


def snapshot_individual_donations(receipts: pl.DataFrame) -> pl.DataFrame:
    """Cumulative quarterly individual-donor snapshots -> ``fec_finance`` rows.

    Returns one row per (race_id, candidate_id, report_quarter) with cumulative
    ``individual_donations`` reported through the quarter end and the cumulative
    ``small_dollar_share``.
    """
    required = {
        "race_id",
        "candidate_id",
        "party",
        "contributor_type",
        "is_self",
        "amount",
        "receipt_date",
    }
    missing = required - set(receipts.columns)
    if missing:
        raise ValueError(f"fec receipts missing columns: {sorted(missing)}")

    if receipts.height == 0:
        return pl.DataFrame(schema={"race_id": pl.Utf8})

    ind = receipts.filter(
        (pl.col("contributor_type").cast(pl.Utf8).str.to_uppercase() == "IND")
        & (~pl.col("is_self").cast(pl.Boolean))
    )
    if ind.height == 0:
        return pl.DataFrame(schema={"race_id": pl.Utf8})

    ind = ind.with_columns(
        date_col("receipt_date"),
        pl.col("amount").cast(pl.Float64),
    ).with_columns(
        pl.col("receipt_date").dt.year().alias("_yr"),
        pl.col("receipt_date").dt.quarter().alias("_q"),
        (pl.col("amount") < SMALL_DOLLAR_THRESHOLD).alias("_is_small"),
    )

    # Per-quarter sums, then cumulative within candidate ordered by time.
    per_q = (
        ind.group_by(["race_id", "candidate_id", "party", "_yr", "_q"])
        .agg(
            pl.col("amount").sum().alias("q_total"),
            pl.col("amount").filter(pl.col("_is_small")).sum().fill_null(0.0).alias("q_small"),
        )
        .sort(["race_id", "candidate_id", "_yr", "_q"])
        .with_columns(
            pl.col("q_total")
            .cum_sum()
            .over(["race_id", "candidate_id"])
            .alias("individual_donations"),
            pl.col("q_small").cum_sum().over(["race_id", "candidate_id"]).alias("cum_small"),
        )
        .with_columns(
            pl.when(pl.col("individual_donations") > 0)
            .then(pl.col("cum_small") / pl.col("individual_donations"))
            .otherwise(None)
            .alias("small_dollar_share"),
            (pl.col("_yr").cast(pl.Utf8) + "Q" + pl.col("_q").cast(pl.Utf8)).alias(
                "report_quarter"
            ),
        )
    )

    # as_of = quarter-end date, derived from _QUARTER_END.
    per_q = per_q.with_columns(
        pl.date(
            pl.col("_yr"),
            pl.col("_q").replace_strict(
                {q: m for q, (m, _d) in _QUARTER_END.items()}, return_dtype=pl.Int32
            ),
            pl.col("_q").replace_strict(
                {q: d for q, (_m, d) in _QUARTER_END.items()}, return_dtype=pl.Int32
            ),
        ).alias("as_of")
    )

    return per_q.select(
        "race_id",
        "candidate_id",
        "party",
        "report_quarter",
        "as_of",
        "individual_donations",
        "small_dollar_share",
    )


def run(ctx: RunContext) -> StepResult:
    """Parse cached FEC receipts and upsert cumulative quarterly snapshots."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.parquet") + base.cached_files(
        ctx, SOURCE, pattern="*.csv"
    )
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached FEC receipts in {ctx.raw_dir / SOURCE}; add files or enable allow_fetch."
        )
    frames = [
        pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p, infer_schema_length=2000)
        for p in paths
    ]
    receipts = pl.concat(frames, how="vertical_relaxed")
    snaps = snapshot_individual_donations(receipts)

    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n = base.upsert_dataframe(conn, "fec_finance", snaps)
    return StepResult(node=STAGE, rows=n, detail=f"{n} candidate-quarter snapshots")


dry_run = stub(STAGE, rows=42_000, detail="FEC individual-donor quarterly snapshots (stub)")
