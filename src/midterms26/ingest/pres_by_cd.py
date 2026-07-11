"""Presidential results by congressional district ingest (Phase 1b).

Source: Daily Kos Elections pres-by-CD crosswalks. Raw files in
``data/raw/pres_by_cd/*.csv`` with columns:

    state, district, plan_label, pres_year, dem_votes, rep_votes [, source]

``plan_label`` names the district lines the results are aggregated onto (e.g.
'2022', 'gen1'), so the same seat can carry results under multiple plans —
that is the whole point for redrawn 2026 lines.

Leakage: ``as_of`` = a certification proxy, Jan 6 of ``pres_year + 1`` — a
presidential result cannot inform a forecast dated before it was counted.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.ingest import base
from midterms26.ingest.normalize import normalize_district, normalize_state, two_party_margin
from midterms26.logging import get_logger
from midterms26.stubs import stub
from midterms26.warehouse import connect, init_schema

STAGE = "ingest.pres_by_cd"
SOURCE = "pres_by_cd"
log = get_logger(STAGE)

DEFAULT_SOURCE_LABEL = "DailyKos"


def certification_proxy(pres_year: int) -> date:
    """Jan 6 of the following year — when the result is unambiguously known."""
    return date(pres_year + 1, 1, 6)


def normalize_pres_by_cd(df: pl.DataFrame) -> pl.DataFrame:
    """Validate raw rows, canonicalize geography, derive shares and margin."""
    required = {"state", "district", "plan_label", "pres_year", "dem_votes", "rep_votes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pres_by_cd missing columns: {sorted(missing)}")

    rows = []
    for row in df.iter_rows(named=True):
        state = normalize_state(str(row["state"]))
        district = normalize_district("HOUSE", row["district"])
        dem = float(row["dem_votes"])
        rep = float(row["rep_votes"])
        total = dem + rep
        margin = two_party_margin(dem, rep)
        pres_year = int(row["pres_year"])
        rows.append(
            {
                "state": state,
                "district": district,
                "plan_label": str(row["plan_label"]),
                "pres_year": pres_year,
                "dem_share": dem / total * 100.0 if total > 0 else None,
                "rep_share": rep / total * 100.0 if total > 0 else None,
                "two_party_margin": margin,
                "as_of": certification_proxy(pres_year),
                "source": str(row.get("source") or DEFAULT_SOURCE_LABEL),
            }
        )
    return pl.DataFrame(rows)


def run(ctx: RunContext) -> StepResult:
    """Parse cached pres-by-CD files and upsert pres_results_by_district."""
    paths = base.cached_files(ctx, SOURCE, pattern="*.csv")
    if not paths:
        raise base.FetchNotAllowedError(
            f"no cached pres_by_cd files in {ctx.raw_dir / SOURCE}; "
            "add files or enable allow_fetch."
        )
    df = pl.concat(
        [pl.read_csv(p, infer_schema_length=2000) for p in paths], how="vertical_relaxed"
    )
    rows = normalize_pres_by_cd(df)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        n = base.upsert_dataframe(conn, "pres_results_by_district", rows)
    return StepResult(node=STAGE, rows=n, detail=f"{n} pres-by-CD rows")


dry_run = stub(STAGE, rows=1_800, detail="Daily Kos pres-by-CD crosswalks (stub)")
