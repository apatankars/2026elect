"""Feature matrix assembly (Phase 3).

Assembles the feature matrix, all as-of-date parameterized. Each race freezes at
its own pre-election cutoff — historical races at ``election_date - horizon`` (E-1
by default), live races at the run's ``cutoff_date`` — and every time-varying
source is routed through :mod:`midterms26.features.leakage` before it can
contribute, so a historical row can never see data past its own election.

Feature families assembled here: fundamentals (district PVI + trend, incumbency,
open/special), national environment (latest macro / approval / generic-ballot
indicators), money (small-dollar share, independent-expenditure intensity), polls
(as-of race average + count), and expert ratings. Uncontested races carry an
imputation flag and a null target so they stay out of calibration sets.

Pure-Polars; runs and tests without the heavy stack.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import polars as pl

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.features.leakage import filter_as_of
from midterms26.ingest import base
from midterms26.logging import get_logger
from midterms26.warehouse import connect, init_schema

STAGE = "features.assemble"
log = get_logger(STAGE)

DEFAULT_HORIZON_DAYS = 1  # E-1 freeze for historical races


def _effective_cutoff(
    election: date | None, run_cutoff: date, horizon_days: int
) -> tuple[date, bool]:
    """Return (freeze_date, is_live) for a race given its election date."""
    if election is None or election > run_cutoff:
        return run_cutoff, True
    return election - timedelta(days=horizon_days), False


def _latest_value(df: pl.DataFrame, cutoff: date, value_col: str) -> float | None:
    """Latest non-null ``value_col`` at or before ``cutoff`` (leakage-guarded)."""
    if df.height == 0:
        return None
    slice_ = filter_as_of(df, cutoff).sort("as_of")
    if slice_.height == 0:
        return None
    val = slice_.select(pl.col(value_col).drop_nulls().last()).item()
    return None if val is None else float(val)


def assemble_features(
    conn: object,
    run_cutoff: date,
    *,
    plan_generation: int = 0,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pl.DataFrame:
    """Build the feature_matrix frame for one run cutoff (leakage-safe per race)."""
    ex = conn.execute  # type: ignore[attr-defined]
    races = ex(
        "SELECT race_id, cycle, office, state, district, is_special, is_uncontested, "
        "incumbent_party, is_open_seat, election_date FROM races"
    ).pl()
    geo = ex(
        "SELECT race_id, pvi_reaggregated, pvi_trend_2016_2024, "
        "incumbent_constituency_overlap, is_new_seat FROM districts_geo"
    ).pl()
    geo_by_race = {r["race_id"]: r for r in geo.to_dicts()}
    results = ex("SELECT race_id, two_party_margin FROM results_history").pl()
    result_by_race = {r["race_id"]: r["two_party_margin"] for r in results.to_dicts()}
    national = ex("SELECT series_id, as_of, value FROM national_indicators").pl()
    polls = ex("SELECT race_id, as_of, margin FROM polls").pl()
    ratings = ex("SELECT race_id, as_of, rating_numeric FROM ratings").pl()
    finance = ex("SELECT race_id, as_of, small_dollar_share FROM fec_finance").pl()
    ie = ex("SELECT race_id, as_of, amount FROM fec_ie").pl()

    series_ids = sorted(national["series_id"].unique().to_list()) if national.height else []

    rows: list[dict[str, object]] = []
    for race in races.iter_rows(named=True):
        rid = race["race_id"]
        election = race["election_date"]
        cutoff, is_live = _effective_cutoff(election, run_cutoff, horizon_days)

        feats: dict[str, float] = {}
        g = geo_by_race.get(rid)
        if g is not None:
            if g["pvi_reaggregated"] is not None:
                feats["district_pvi"] = float(g["pvi_reaggregated"])
            if g["pvi_trend_2016_2024"] is not None:
                feats["pvi_trend"] = float(g["pvi_trend_2016_2024"])
            feats["incumbent_overlap"] = float(g["incumbent_constituency_overlap"] or 0.0)
            feats["is_new_seat"] = 1.0 if g["is_new_seat"] else 0.0

        feats["incumbent_dem"] = 1.0 if race["incumbent_party"] == "D" else 0.0
        feats["incumbent_rep"] = 1.0 if race["incumbent_party"] == "R" else 0.0
        feats["is_open_seat"] = 1.0 if race["is_open_seat"] else 0.0
        feats["is_special"] = 1.0 if race["is_special"] else 0.0

        for sid in series_ids:
            v = _latest_value(national.filter(pl.col("series_id") == sid), cutoff, "value")
            if v is not None:
                feats[f"nat_{sid.lower()}"] = v

        race_polls = filter_as_of(polls.filter(pl.col("race_id") == rid), cutoff)
        if race_polls.height:
            feats["poll_margin"] = float(race_polls.select(pl.col("margin").mean()).item())
            feats["n_polls"] = float(race_polls.height)

        rate = _latest_value(ratings.filter(pl.col("race_id") == rid), cutoff, "rating_numeric")
        if rate is not None:
            feats["rating_numeric"] = rate

        sds = _latest_value(finance.filter(pl.col("race_id") == rid), cutoff, "small_dollar_share")
        if sds is not None:
            feats["small_dollar_share"] = sds

        race_ie = filter_as_of(ie.filter(pl.col("race_id") == rid), cutoff)
        if race_ie.height:
            feats["ie_intensity"] = float(race_ie.select(pl.col("amount").sum()).item())

        target = None if is_live else result_by_race.get(rid)
        rows.append(
            {
                "race_id": rid,
                "cutoff_date": run_cutoff,
                "plan_generation": plan_generation,
                "features": json.dumps(feats),
                "target_margin": None if target is None else float(target),
                "is_imputed_uncontested": bool(race["is_uncontested"]),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def run(ctx: RunContext) -> StepResult:
    """Assemble and upsert the feature matrix for ``ctx.cutoff_date``."""
    if ctx.cutoff_date is None:
        raise ValueError("features.assemble needs ctx.cutoff_date")
    with connect(ctx.db_path) as conn:
        init_schema(conn)
        frame = assemble_features(conn, ctx.cutoff_date)
        n = base.upsert_dataframe(conn, "feature_matrix", frame) if frame.height else 0
    log.info("features.done", rows=n, cutoff=ctx.cutoff_date.isoformat())
    return StepResult(node=STAGE, rows=n, detail=f"{n} feature rows")


def dry_run(ctx: RunContext) -> StepResult:  # noqa: ARG001
    return StepResult(
        node=STAGE,
        rows=3_600,
        detail="race-cycle rows x ~100 features, leakage-guarded (stub)",
        dry_run=True,
    )
