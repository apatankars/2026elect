"""Historical returns ingest — the target variable.

Source: MIT Election Data + Science Lab (House/Senate/Gov returns, district level).
MIT ships one CSV per office; each row is a candidate. We aggregate to race level,
bucket votes into D / R / other, compute the two-party margin (D% − R%), and
populate ``races`` + ``results_history``.

Assumptions (validate against the live download — see docs/METHODOLOGY risks):
  * General-election rows only (``stage == 'GEN'``).
  * When a ``mode`` column contains a 'TOTAL' row, per-mode rows are dropped to
    avoid double counting; otherwise modes are summed.
  * Party bucket: DEMOCRAT -> D, REPUBLICAN -> R, everything else -> other.
    Fusion-ticket votes for the same party bucket sum together.

Real execution runs against cached CSVs in ``data/raw/mit_edsl/``; fetching the
files is gated by ``allow_fetch`` (see :mod:`midterms26.ingest.base`).
"""

from __future__ import annotations

import polars as pl

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.ingest import base
from midterms26.ingest.normalize import (
    NormalizationError,
    normalize_district,
    normalize_office,
    normalize_state,
    two_party_margin,
)
from midterms26.logging import get_logger
from midterms26.stubs import stub
from midterms26.warehouse import connect, init_schema

STAGE = "ingest.results"
SOURCE = "mit_edsl"
log = get_logger(STAGE)

# Columns we consume; others are ignored. Missing optional cols are backfilled null.
_REQUIRED = ("year", "state_po", "office", "party", "candidatevotes")
_OPTIONAL = ("district", "stage", "special", "mode")


def _standardize(df: pl.DataFrame) -> pl.DataFrame:
    """Lowercase column names and ensure optional columns exist."""
    df = df.rename({c: c.lower() for c in df.columns})
    for col in _REQUIRED:
        if col not in df.columns:
            raise NormalizationError(f"results file missing required column {col!r}")
    for col in _OPTIONAL:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    return df.select([*_REQUIRED, *_OPTIONAL])


def _party_bucket(party: str | None) -> str:
    p = (party or "").strip().upper()
    if p in {"DEMOCRAT", "DEMOCRATIC", "DEMOCRATIC-FARMER-LABOR", "DEM", "D"}:
        return "D"
    if p in {"REPUBLICAN", "REP", "R", "GOP"}:
        return "R"
    return "O"


def _canonical_row(
    cycle: int, office_raw: str, state_raw: str, district_raw: object
) -> tuple[str, str, str, str] | None:
    """Return (race_id, office, state, district) or ``None`` if office is out of scope."""
    try:
        office = normalize_office(office_raw)
    except NormalizationError:
        return None  # president / other offices in the same file
    try:
        state = normalize_state(state_raw)
        district = normalize_district(office, district_raw)
    except NormalizationError as exc:
        log.warning("results.skip_row", reason=str(exc), office=office_raw, state=state_raw)
        return None
    return f"{cycle}-{office}-{state}-{district}", office, state, district


def normalize_returns(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Turn a raw MIT returns frame into (races_df, results_history_df)."""
    df = _standardize(df)

    # General election only, when a stage column is present.
    if df.select(pl.col("stage").is_not_null().any()).item():
        df = df.filter(
            pl.col("stage").is_null() | (pl.col("stage").cast(pl.Utf8).str.to_uppercase() == "GEN")
        )

    # Mode dedup: prefer TOTAL rows if any exist in the file.
    has_total = df.select((pl.col("mode").cast(pl.Utf8).str.to_uppercase() == "TOTAL").any()).item()
    if has_total:
        df = df.filter(pl.col("mode").cast(pl.Utf8).str.to_uppercase() == "TOTAL")

    records: list[dict[str, object]] = []
    for row in df.iter_rows(named=True):
        canon = _canonical_row(
            int(row["year"]), str(row["office"]), str(row["state_po"]), row["district"]
        )
        if canon is None:
            continue
        rid, office, state, district = canon
        special = bool(row["special"]) if row["special"] is not None else False
        votes = row["candidatevotes"]
        records.append(
            {
                "race_id": rid,
                "cycle": int(row["year"]),
                "office": office,
                "state": state,
                "district": district,
                "is_special": special,
                "bucket": _party_bucket(row["party"]),
                "votes": int(votes) if votes is not None else 0,
            }
        )

    if not records:
        empty_races = pl.DataFrame(schema={"race_id": pl.Utf8})
        return empty_races, pl.DataFrame(schema={"race_id": pl.Utf8})

    long = pl.DataFrame(records)
    # Sum votes per race x bucket, then pivot buckets to columns.
    agg = (
        long.group_by(["race_id", "cycle", "office", "state", "district", "is_special", "bucket"])
        .agg(pl.col("votes").sum().alias("votes"))
        .pivot(
            values="votes",
            index=["race_id", "cycle", "office", "state", "district", "is_special"],
            on="bucket",
        )
    )
    for b in ("D", "R", "O"):
        if b not in agg.columns:
            agg = agg.with_columns(pl.lit(0).alias(b))
    agg = agg.with_columns(
        pl.col("D").fill_null(0).alias("dem_votes"),
        pl.col("R").fill_null(0).alias("rep_votes"),
        pl.col("O").fill_null(0).alias("other_votes"),
    ).with_columns(
        (pl.col("dem_votes") + pl.col("rep_votes") + pl.col("other_votes")).alias("total_votes"),
        pl.struct(["dem_votes", "rep_votes"])
        .map_elements(
            lambda s: two_party_margin(s["dem_votes"], s["rep_votes"]), return_dtype=pl.Float64
        )
        .alias("two_party_margin"),
        ((pl.col("D").fill_null(0) == 0) | (pl.col("R").fill_null(0) == 0)).alias("is_uncontested"),
    )

    races_df = agg.select(
        "race_id", "cycle", "office", "state", "district", "is_special", "is_uncontested"
    )
    results_df = agg.select(
        "race_id",
        "cycle",
        "office",
        "state",
        "district",
        "dem_votes",
        "rep_votes",
        "total_votes",
        "two_party_margin",
    ).with_columns(pl.lit("MIT-EDSL").alias("source"))
    return races_df, results_df


def run(ctx: RunContext) -> StepResult:
    """Parse cached MIT returns and upsert races + results_history."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.csv")
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached MIT returns in {ctx.raw_dir / SOURCE}; add CSVs or enable allow_fetch."
        )
    frames = [pl.read_csv(p, infer_schema_length=2000) for p in paths]
    races_all, results_all = [], []
    for frame in frames:
        r, res = normalize_returns(frame)
        if r.height:
            races_all.append(r)
            results_all.append(res)

    races_df = pl.concat(races_all, how="vertical_relaxed") if races_all else pl.DataFrame()
    results_df = pl.concat(results_all, how="vertical_relaxed") if results_all else pl.DataFrame()
    # De-dup race spine across files (a race should appear once).
    if races_df.height:
        races_df = races_df.unique(subset=["race_id"], keep="first")
    if results_df.height:
        results_df = results_df.unique(subset=["race_id"], keep="first")

    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n_races = base.upsert_dataframe(conn, "races", races_df)
        n_results = base.upsert_dataframe(conn, "results_history", results_df)
    return StepResult(node=STAGE, rows=n_results, detail=f"{n_races} races, {n_results} results")


dry_run = stub(STAGE, rows=13_500, detail="MIT-EDSL House/Senate/Gov 2006-2024 (stub)")
