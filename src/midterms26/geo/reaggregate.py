"""Redistricting-native district features (Phase 2).

The full pipeline reaggregates precinct -> block (VAP-weighted) -> enacted district
with ``maup`` (the ``geo`` extra) to recompute electoral history on new lines. When
plan geometries aren't cached, this falls back to the **tabular pres-by-CD path**:
Daily Kos crosswalks already give presidential results on the enacted district
lines, which is exactly the PVI signal the features need — the roadmap designates
pres-by-CD as the cross-check/fallback where precinct coverage is thin.

Either way the output is ``districts_geo``, one row per (race, plan_generation):

  * ``pvi_reaggregated``      — district two-party pres margin minus the national
    pres margin (Cook-style PVI, + = D-leaning), on the current lines.
  * ``pvi_trend_2016_2024``   — change in that PVI between the two most recent
    presidential years (drift of the seat's partisanship).
  * ``incumbent_constituency_overlap`` — 1.0 in the tabular path (no geometry to
    measure carry-over); the maup path fills the real 0..1 value.
  * ``is_new_seat``           — no prior presidential result on these lines.
  * ``reaggregation_error``   — 0.0 in the tabular path (nothing is interpolated).

Pure-Polars, so it runs and tests without the GIS stack.
"""

from __future__ import annotations

import polars as pl

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.ingest import base
from midterms26.logging import get_logger
from midterms26.warehouse import connect, init_schema

STAGE = "geo.reaggregate"
log = get_logger(STAGE)


def pvi_table(pres: pl.DataFrame) -> pl.DataFrame:
    """Per (state, district, plan_label) PVI level + trend from pres-by-CD rows.

    PVI(year) = district two_party_margin(year) − national margin(year), where the
    national margin is the mean district margin that year. Level uses the most
    recent presidential year; trend = latest PVI − previous-year PVI.
    """
    if pres.height == 0:
        return pl.DataFrame(
            schema={
                "state": pl.Utf8,
                "district": pl.Utf8,
                "plan_label": pl.Utf8,
                "pvi_reaggregated": pl.Float64,
                "pvi_trend_2016_2024": pl.Float64,
                "n_years": pl.UInt32,
            }
        )
    national = pres.group_by("pres_year").agg(
        pl.col("two_party_margin").mean().alias("national_margin")
    )
    pvi = (
        pres.join(national, on="pres_year")
        .with_columns((pl.col("two_party_margin") - pl.col("national_margin")).alias("pvi"))
        .sort("pres_year")
    )
    return pvi.group_by(["state", "district", "plan_label"]).agg(
        pl.col("pvi").last().alias("pvi_reaggregated"),
        (pl.col("pvi").last() - pl.col("pvi").first()).alias("pvi_trend_2016_2024"),
        pl.col("pres_year").n_unique().alias("n_years"),
    )


def build_districts_geo(
    races: pl.DataFrame, pres: pl.DataFrame, prior_seats: set[tuple[str, str]]
) -> pl.DataFrame:
    """Join per-race spine to district PVI; derive is_new_seat / overlap / error."""
    pvi = pvi_table(pres)
    rows: list[dict[str, object]] = []
    pvi_lookup = {(r["state"], r["district"]): r for r in pvi.to_dicts()}
    for race in races.iter_rows(named=True):
        key = (race["state"], race["district"])
        p = pvi_lookup.get(key)
        rows.append(
            {
                "race_id": race["race_id"],
                "state": race["state"],
                "district": race["district"],
                "plan_generation": race.get("plan_generation", 0) or 0,
                "plan_enacted_date": None,
                "is_new_seat": key not in prior_seats,
                "pvi_reaggregated": None if p is None else p["pvi_reaggregated"],
                "pvi_trend_2016_2024": None if p is None else p["pvi_trend_2016_2024"],
                "incumbent_constituency_overlap": 1.0,
                "reaggregation_error": 0.0,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def run(ctx: RunContext) -> StepResult:
    """Populate ``districts_geo`` from pres-by-CD (tabular geo path)."""
    if base.cached_files(ctx, "plans", pattern="*.shp"):
        raise NotImplementedError(
            "maup areal-interpolation path is Phase 2; see geo/reaggregate.py"
        )
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        races = conn.execute(
            "SELECT race_id, state, district, plan_generation FROM races WHERE office = 'HOUSE'"
        ).pl()
        pres = conn.execute(
            "SELECT state, district, plan_label, pres_year, two_party_margin "
            "FROM pres_results_by_district"
        ).pl()
        # Seats with any prior district-level results are not "new".
        prior = conn.execute(
            "SELECT DISTINCT state, district FROM results_history WHERE office = 'HOUSE'"
        ).fetchall()
        prior_seats = {(s, d) for s, d in prior}
        geo = build_districts_geo(races, pres, prior_seats)
        n = base.upsert_dataframe(conn, "districts_geo", geo) if geo.height else 0
    log.info("geo.done", rows=n)
    return StepResult(node=STAGE, rows=n, detail=f"{n} districts (tabular pres-by-CD PVI)")


def dry_run(ctx: RunContext) -> StepResult:  # noqa: ARG001
    return StepResult(
        node=STAGE,
        rows=470,
        detail="reaggregated 2026 districts, all plan generations (stub)",
        dry_run=True,
    )
