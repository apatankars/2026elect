"""DIME / CFscore candidate-ideology ingest (Phase 1b).

Source: Stanford DIME (Bonica), pre-shaped into per-cycle CSVs. Raw files in
``data/raw/dime/*.csv`` with columns:

    candidate_id, cycle, cfscore [, cfscore_dyn, n_donors, source]

``candidate_id`` is the FEC id (crosswalked upstream so it joins fec_finance).

Leakage: DIME scores are estimated from a *full cycle's* receipts, so using a
cycle-Y score to forecast cycle Y would leak. ``as_of`` is therefore set to
Dec 31 of the score's cycle; any pre-election cutoff excludes the same-cycle
score, and :func:`scores_as_of` returns each candidate's latest usable score.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.ingest import base
from midterms26.logging import get_logger
from midterms26.stubs import stub
from midterms26.warehouse import connect, init_schema

STAGE = "ingest.dime"
SOURCE = "dime"
log = get_logger(STAGE)

# CFscores live on roughly [-2, 2]; anything beyond this is a parse error.
MAX_ABS_CFSCORE = 6.0
DEFAULT_SOURCE_LABEL = "DIME-v4"


def normalize_dime(df: pl.DataFrame) -> pl.DataFrame:
    """Validate raw DIME rows and derive the cycle-end ``as_of`` leakage key."""
    required = {"candidate_id", "cycle", "cfscore"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dime missing columns: {sorted(missing)}")

    out = df.with_columns(
        pl.col("candidate_id").cast(pl.Utf8),
        pl.col("cycle").cast(pl.Int64),
        pl.col("cfscore").cast(pl.Float64),
    )
    for opt, dtype in (("cfscore_dyn", pl.Float64), ("n_donors", pl.Int64)):
        out = (
            out.with_columns(pl.col(opt).cast(dtype))
            if opt in out.columns
            else out.with_columns(pl.lit(None, dtype=dtype).alias(opt))
        )
    if "source" not in out.columns:
        out = out.with_columns(pl.lit(DEFAULT_SOURCE_LABEL).alias("source"))

    n_null = out.select(pl.col("cfscore").is_null().sum()).item()
    if n_null:
        raise ValueError(f"dime has {n_null} null cfscore row(s); drop them upstream")

    bad = out.filter(pl.col("cfscore").abs() > MAX_ABS_CFSCORE)
    if bad.height:
        raise ValueError(f"dime has {bad.height} cfscore row(s) with |score| > {MAX_ABS_CFSCORE}")

    out = out.with_columns(
        pl.date(pl.col("cycle"), 12, 31).alias("as_of"),
    )
    return out.select(
        "candidate_id", "cycle", "as_of", "cfscore", "cfscore_dyn", "n_donors", "source"
    )


def scores_as_of(scores: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """Latest usable score per candidate: max cycle with ``as_of <= cutoff``.

    Leakage-safe accessor for the feature assembler — a same-cycle score
    (as_of = Dec 31) can never be selected by a pre-election cutoff.
    """
    eligible = scores.filter(pl.col("as_of") <= cutoff)
    if eligible.height == 0:
        return eligible
    return (
        eligible.sort("cycle", descending=True)
        .group_by("candidate_id", maintain_order=True)
        .first()
    )


def run(ctx: RunContext) -> StepResult:
    """Parse cached DIME files and upsert the candidate_ideology table."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.csv")
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached dime files in {ctx.raw_dir / SOURCE}; add files or enable allow_fetch."
        )
    df = pl.concat(
        [pl.read_csv(p, infer_schema_length=2000) for p in paths], how="vertical_relaxed"
    )
    rows = normalize_dime(df)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n = base.upsert_dataframe(conn, "candidate_ideology", rows)
    return StepResult(node=STAGE, rows=n, detail=f"{n} cfscore rows")


dry_run = stub(STAGE, rows=9_000, detail="DIME CFscores 2006-2024 (stub)")
