"""Phase 1b sources: ACS, pres-by-CD, DIME CFscores, FEC Schedule E IEs."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from midterms26.context import RunContext
from midterms26.ingest.acs import normalize_acs
from midterms26.ingest.dime import normalize_dime, scores_as_of
from midterms26.ingest.fec_ie import normalize_fec_ie, race_ie_totals
from midterms26.ingest.pres_by_cd import certification_proxy, normalize_pres_by_cd
from midterms26.pipeline import build_ingest_dag
from midterms26.warehouse import connect, init_schema

RAW = Path(__file__).parent / "fixtures" / "raw"


def _read(source: str, name: str) -> pl.DataFrame:
    return pl.read_csv(RAW / source / name, infer_schema_length=2000)


# -- ACS ---------------------------------------------------------------------


def test_acs_normalize_parses_release_date_as_of() -> None:
    df = normalize_acs(_read("acs", "acs_2023.csv"))
    assert df.height == 5
    assert df["as_of"].unique().to_list() == [date(2024, 12, 12)]
    assert set(df["geo_level"].unique().to_list()) <= {"BG", "TRACT", "CD"}
    # geoid must survive as text (leading-zero safety)
    assert df["geoid"].dtype == pl.Utf8


def test_acs_rejects_out_of_range_pct() -> None:
    bad = _read("acs", "acs_2023.csv").with_columns(
        pl.when(pl.col("variable") == "pct_ba_plus")
        .then(pl.lit(140.0))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    with pytest.raises(ValueError, match=r"pct_\* row"):
        normalize_acs(bad)


def test_acs_rejects_unknown_geo_level() -> None:
    bad = _read("acs", "acs_2023.csv").with_columns(pl.lit("COUNTY").alias("geo_level"))
    with pytest.raises(ValueError, match="geo_level"):
        normalize_acs(bad)


# -- Pres-by-CD ---------------------------------------------------------------


def test_pres_by_cd_canonicalizes_geography_and_margin() -> None:
    df = normalize_pres_by_cd(_read("pres_by_cd", "pres_by_cd_pa.csv"))
    pa07_2020 = df.filter((pl.col("district") == "07") & (pl.col("pres_year") == 2020)).row(
        0, named=True
    )
    assert pa07_2020["state"] == "PA"
    # (182000 - 175000) / 357000 * 100
    assert pa07_2020["two_party_margin"] == pytest.approx(1.9608, abs=1e-3)
    assert pa07_2020["dem_share"] + pa07_2020["rep_share"] == pytest.approx(100.0)

    # lowercase state + 'AL' at-large district both canonicalize
    al = df.filter(pl.col("district") == "00").row(0, named=True)
    assert al["state"] == "PA"
    assert al["two_party_margin"] < 0


def test_pres_by_cd_as_of_is_certification_proxy() -> None:
    df = normalize_pres_by_cd(_read("pres_by_cd", "pres_by_cd_pa.csv"))
    for row in df.iter_rows(named=True):
        assert row["as_of"] == date(row["pres_year"] + 1, 1, 6)
    assert certification_proxy(2020) == date(2021, 1, 6)


# -- DIME ---------------------------------------------------------------------


def test_dime_as_of_is_cycle_end() -> None:
    df = normalize_dime(_read("dime", "dime_scores.csv"))
    assert df.height == 3
    r = df.filter((pl.col("candidate_id") == "H8PA07123") & (pl.col("cycle") == 2022))
    assert r["as_of"].item() == date(2022, 12, 31)


def test_dime_same_cycle_score_never_passes_pre_election_cutoff() -> None:
    df = normalize_dime(_read("dime", "dime_scores.csv"))
    # Forecasting the 2024 election on Nov 3 2024: the 2024 score (as_of
    # Dec 31 2024, full-cycle receipts) must be invisible; the 2022 score wins.
    picked = scores_as_of(df, date(2024, 11, 3))
    pa = picked.filter(pl.col("candidate_id") == "H8PA07123").row(0, named=True)
    assert pa["cycle"] == 2022
    assert pa["cfscore"] == pytest.approx(-0.62)
    # The TX candidate only has a 2024 score -> not usable yet.
    assert picked.filter(pl.col("candidate_id") == "H6TX02999").height == 0


def test_dime_rejects_absurd_cfscore() -> None:
    bad = _read("dime", "dime_scores.csv").with_columns(pl.lit(9.5).alias("cfscore"))
    with pytest.raises(ValueError, match="cfscore"):
        normalize_dime(bad)


# -- FEC Schedule E -----------------------------------------------------------


def test_fec_ie_canonicalizes_support_oppose() -> None:
    df = normalize_fec_ie(_read("fec_ie", "schedule_e_2026.csv"))
    assert df.height == 4
    assert set(df["support_oppose"].unique().to_list()) == {"S", "O"}
    assert df.filter(pl.col("ie_id") == "SB3")["support_oppose"].item() == "S"
    assert df.filter(pl.col("ie_id") == "SB4")["support_oppose"].item() == "O"


def test_fec_ie_rejects_unknown_support_oppose_and_negative_amounts() -> None:
    raw = _read("fec_ie", "schedule_e_2026.csv")
    with pytest.raises(ValueError, match="support_oppose"):
        normalize_fec_ie(raw.with_columns(pl.lit("MAYBE").alias("support_oppose")))
    with pytest.raises(ValueError, match="negative"):
        normalize_fec_ie(raw.with_columns(pl.lit(-5.0).alias("amount")))


def test_race_ie_totals_is_leakage_safe() -> None:
    df = normalize_fec_ie(_read("fec_ie", "schedule_e_2026.csv"))
    # As-of Oct 1: only SB1 (S 250k, Sep 15) counts for PA-07 incumbent —
    # the Oct 20 oppose spend must be excluded.
    early = race_ie_totals(df, date(2026, 10, 1))
    pa = early.filter(
        (pl.col("race_id") == "2026-HOUSE-PA-07") & (pl.col("candidate_id") == "H8PA07123")
    ).row(0, named=True)
    assert pa["net_support"] == pytest.approx(250_000.0)

    # As-of Nov 1: net = 250k support - 100k oppose.
    late = race_ie_totals(df, date(2026, 11, 1))
    pa = late.filter(
        (pl.col("race_id") == "2026-HOUSE-PA-07") & (pl.col("candidate_id") == "H8PA07123")
    ).row(0, named=True)
    assert pa["net_support"] == pytest.approx(150_000.0)
    assert pa["gross_total"] == pytest.approx(350_000.0)


# -- DAG wiring + warehouse round-trip ----------------------------------------


def test_ingest_dag_includes_phase1b_nodes() -> None:
    nodes = set(build_ingest_dag().nodes)
    assert {"ingest.acs", "ingest.pres_by_cd", "ingest.dime", "ingest.fec_ie"} <= nodes
    assert "ingest.plans" not in nodes


def test_phase1b_run_round_trip_is_idempotent(tmp_path: Path) -> None:
    from midterms26.ingest import dime as dime_mod
    from midterms26.ingest import fec_ie as fec_ie_mod

    ctx = RunContext(dry_run=False, db_path=tmp_path / "wh.duckdb", raw_dir=RAW)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
    for mod in (dime_mod, fec_ie_mod):
        mod.run(ctx)
        mod.run(ctx)  # second run must not duplicate
    with connect(ctx.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM candidate_ideology").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM fec_ie").fetchone()[0] == 4
