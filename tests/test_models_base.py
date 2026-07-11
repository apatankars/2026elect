"""Pure model-I/O layer: race-id parsing, standardization, warehouse round-trips."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from midterms26.models.base import (
    MemberInput,
    grid_from_json,
    grid_to_json,
    load_member_input,
    parse_race_id,
    write_latent_factors,
    write_member_predictions,
)
from midterms26.warehouse import connect, init_schema


def test_parse_race_id() -> None:
    assert parse_race_id("2026-HOUSE-CA-01") == (2026, "HOUSE", "CA", "01")
    assert parse_race_id("2022-SENATE-OH-SEN") == (2022, "SENATE", "OH", "SEN")


def test_parse_race_id_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_race_id("garbage")


def test_member_input_alignment_checked() -> None:
    with pytest.raises(ValueError):
        MemberInput(
            race_ids=["2026-HOUSE-CA-01"],
            feature_names=["a"],
            X=[[1.0]],
            y=[0.0, 1.0],  # misaligned
            cycles=[2026],
            states=["CA"],
            offices=["HOUSE"],
        )


def test_labeled_and_live_partition() -> None:
    mi = MemberInput(
        race_ids=["2018-HOUSE-CA-01", "2026-HOUSE-CA-01"],
        feature_names=["pvi"],
        X=[[1.0], [2.0]],
        y=[5.0, None],
        cycles=[2018, 2026],
        states=["CA", "CA"],
        offices=["HOUSE", "HOUSE"],
    )
    assert mi.labeled_indices() == [0]
    assert mi.live_indices() == [1]
    assert mi.labeled_cycles() == [2018]


def test_standardized_centers_and_imputes_missing() -> None:
    mi = MemberInput(
        race_ids=["a-HOUSE-CA-01", "b-HOUSE-CA-02", "c-HOUSE-CA-03"],
        feature_names=["f"],
        X=[[0.0], [2.0], [None]],  # mean 1.0 over the two non-null labeled rows
        y=[1.0, -1.0, 3.0],
        cycles=[2014, 2018, 2022],
        states=["CA", "CA", "CA"],
        offices=["HOUSE", "HOUSE", "HOUSE"],
    )
    z = mi.standardized()
    assert z[0][0] == pytest.approx(-1.0)
    assert z[1][0] == pytest.approx(1.0)
    assert z[2][0] == 0.0  # missing -> standardized mean


def test_grid_json_roundtrip() -> None:
    grid = {0.05: -3.0, 0.5: 1.5, 0.95: 6.0}
    assert grid_from_json(grid_to_json(grid)) == grid


def _seed_features(db: Path, cutoff: str) -> None:
    with connect(db) as conn:
        init_schema(conn)
        rows = [
            ("2018-HOUSE-CA-01", json.dumps({"pvi": 3.0, "money": 0.4}), 5.0),
            ("2022-HOUSE-CA-01", json.dumps({"pvi": 4.0}), -2.0),  # missing 'money'
            ("2026-HOUSE-CA-01", json.dumps({"pvi": 3.5, "money": 0.6}), None),  # live
        ]
        conn.executemany(
            "INSERT INTO feature_matrix (race_id, cutoff_date, plan_generation, features, "
            "target_margin) VALUES (?, ?, 0, ?, ?)",
            [(rid, cutoff, feats, tgt) for rid, feats, tgt in rows],
        )


def test_load_member_input_from_warehouse(tmp_path: Path) -> None:
    db = tmp_path / "wh.duckdb"
    cutoff = "2026-10-01"
    _seed_features(db, cutoff)
    with connect(db) as conn:
        mi = load_member_input(conn, date.fromisoformat(cutoff))
    assert mi.feature_names == ["money", "pvi"]  # sorted union
    assert mi.n_rows == 3
    assert mi.live_indices() == [2]  # the null-target row
    assert mi.cycles == [2018, 2022, 2026]
    # Missing 'money' on the 2022 row surfaces as None (imputed later).
    assert mi.X[1][0] is None


def test_write_member_predictions_and_latent_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "wh.duckdb"
    cutoff = date(2026, 10, 1)
    with connect(db) as conn:
        init_schema(conn)
        n = write_member_predictions(
            conn,
            cutoff_date=cutoff,
            plan_generation=0,
            model_member="BAYES",
            fold="live",
            grids={"2026-HOUSE-CA-01": {0.05: -3.0, 0.5: 1.0, 0.95: 5.0}},
        )
        assert n == 1
        m = write_latent_factors(
            conn,
            cutoff_date=cutoff,
            plan_generation=0,
            loadings={"2026-HOUSE-CA-01": {"national": 2.0, "state:CA": 1.0}},
            idiosyncratic_sd={"2026-HOUSE-CA-01": 3.0},
        )
        assert m == 1
        median, quantiles = conn.execute(
            "SELECT median_margin, quantiles FROM member_predictions WHERE model_member='BAYES'"
        ).fetchone()
        assert median == pytest.approx(1.0)
        assert grid_from_json(quantiles)[0.95] == pytest.approx(5.0)
        idio = conn.execute("SELECT idiosyncratic_sd FROM latent_factors").fetchone()[0]
        assert idio == pytest.approx(3.0)
