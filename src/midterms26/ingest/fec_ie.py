"""FEC Schedule E independent-expenditure ingest (Phase 1b).

Source: FEC bulk Schedule E, pre-shaped so each row already carries the
canonical ``race_id`` (mapped from the FEC candidate file upstream). Raw files
in ``data/raw/fec_ie/*.csv`` with columns:

    ie_id, race_id, candidate_id, committee_id, support_oppose, amount,
    expenditure_date

Why this source: independent expenditures are "insider revealed preference" —
where parties and PACs actually spend late money marks the races they believe
are in play, a competitiveness signal no public model uses explicitly.

Leakage: ``as_of`` = expenditure date. :func:`race_ie_totals` aggregates net
support per (race, candidate) strictly as-of a cutoff.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.ingest import base
from midterms26.ingest.normalize import date_col
from midterms26.logging import get_logger
from midterms26.stubs import stub
from midterms26.warehouse import connect, init_schema

STAGE = "ingest.fec_ie"
SOURCE = "fec_ie"
log = get_logger(STAGE)

_SUPPORT_OPPOSE = {"S": "S", "SUPPORT": "S", "O": "O", "OPPOSE": "O"}


def normalize_fec_ie(df: pl.DataFrame) -> pl.DataFrame:
    """Validate raw Schedule E rows; canonicalize support/oppose and dates."""
    required = {
        "ie_id",
        "race_id",
        "candidate_id",
        "support_oppose",
        "amount",
        "expenditure_date",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fec_ie missing columns: {sorted(missing)}")

    out = df.with_columns(
        pl.col("ie_id").cast(pl.Utf8),
        pl.col("race_id").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8),
        pl.col("support_oppose")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .replace_strict(_SUPPORT_OPPOSE, default=None),
        pl.col("amount").cast(pl.Float64),
        date_col("expenditure_date").alias("as_of"),
    )
    if "committee_id" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("committee_id"))
    else:
        out = out.with_columns(pl.col("committee_id").cast(pl.Utf8))

    n_bad_so = out.select(pl.col("support_oppose").is_null().sum()).item()
    if n_bad_so:
        raise ValueError(f"fec_ie has {n_bad_so} row(s) with unrecognized support_oppose")

    n_neg = out.select((pl.col("amount") < 0).sum()).item()
    if n_neg:
        raise ValueError(f"fec_ie has {n_neg} negative amount row(s); refunds go upstream")

    return out.select(
        "ie_id", "race_id", "candidate_id", "committee_id", "support_oppose", "amount", "as_of"
    )


def race_ie_totals(ie: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """Net IE support per (race, candidate) using only rows with as_of <= cutoff.

    Net = sum(support amounts) − sum(oppose amounts). Leakage-safe: strictly
    filters on the expenditure date before aggregating.
    """
    eligible = ie.filter(pl.col("as_of") <= as_of)
    signed = eligible.with_columns(
        pl.when(pl.col("support_oppose") == "S")
        .then(pl.col("amount"))
        .otherwise(-pl.col("amount"))
        .alias("signed_amount")
    )
    return (
        signed.group_by("race_id", "candidate_id")
        .agg(
            pl.col("signed_amount").sum().alias("net_support"),
            pl.col("amount").sum().alias("gross_total"),
        )
        .sort("race_id", "candidate_id")
    )


def run(ctx: RunContext) -> StepResult:
    """Parse cached Schedule E files and upsert the fec_ie table."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.csv")
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached fec_ie files in {ctx.raw_dir / SOURCE}; add files or enable allow_fetch."
        )
    df = pl.concat(
        [pl.read_csv(p, infer_schema_length=2000) for p in paths], how="vertical_relaxed"
    )
    rows = normalize_fec_ie(df)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n = base.upsert_dataframe(conn, "fec_ie", rows)
    return StepResult(node=STAGE, rows=n, detail=f"{n} IE rows")


dry_run = stub(STAGE, rows=40_000, detail="FEC Schedule E IEs (stub)")
