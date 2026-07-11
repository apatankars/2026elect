"""Phase 6 publish: JSON builders + full run() against a seeded warehouse."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from midterms26.context import RunContext
from midterms26.publish import emit
from midterms26.warehouse import connect, init_schema


def _pred(race_id: str, office: str, *, win: float, abstain: bool = False, reason=None) -> dict:
    return {
        "race_id": race_id,
        "office": office,
        "median_margin": 2.0,
        "lo_50": 0.0,
        "hi_50": 4.0,
        "lo_80": -3.0,
        "hi_80": 7.0,
        "lo_90": -6.0,
        "hi_90": 10.0,
        "win_prob_dem": win,
        "mondrian_group": "incD|polled3+|stable|HOUSE",
        "abstain": abstain,
        "abstain_reason": reason,
    }


def test_race_card_shapes_intervals_by_percent() -> None:
    card = emit.race_card(_pred("2026-HOUSE-CA-01", "HOUSE", win=0.7))
    assert set(card["intervals"]) == {"50", "80", "90"}
    assert card["intervals"]["80"] == {"lo": -3.0, "hi": 7.0}
    assert card["win_prob_dem"] == 0.7


def test_race_table_sorted_by_id() -> None:
    preds = [
        _pred("2026-HOUSE-CA-02", "HOUSE", win=0.6),
        _pred("2026-HOUSE-CA-01", "HOUSE", win=0.4),
    ]
    table = emit.build_race_table(preds)
    assert [c["race_id"] for c in table] == ["2026-HOUSE-CA-01", "2026-HOUSE-CA-02"]


def test_no_call_only_abstained() -> None:
    preds = [
        _pred("2026-HOUSE-CA-01", "HOUSE", win=0.5, abstain=True, reason="width>40"),
        _pred("2026-HOUSE-CA-02", "HOUSE", win=0.6),
    ]
    nc = emit.build_no_call(preds)
    assert len(nc) == 1
    assert nc[0]["race_id"] == "2026-HOUSE-CA-01"
    assert nc[0]["reason"] == "width>40"


def test_expected_seats_uses_linearity() -> None:
    preds = [
        _pred("2026-HOUSE-CA-01", "HOUSE", win=0.9),
        _pred("2026-HOUSE-CA-02", "HOUSE", win=0.5),
        _pred("2026-SENATE-OH-SEN", "SENATE", win=0.2),
        _pred("2026-HOUSE-CA-03", "HOUSE", win=0.1, abstain=True, reason="bin<10"),
    ]
    seats = emit.build_expected_seats(preds)
    # HOUSE: 0.9 + 0.5 = 1.4 expected (abstention excluded), 3 races, 1 abstain.
    assert seats["HOUSE"]["expected_dem_seats"] == pytest.approx(1.4)
    assert seats["HOUSE"]["n_races"] == 3
    assert seats["HOUSE"]["n_abstain"] == 1
    assert seats["SENATE"]["expected_dem_seats"] == pytest.approx(0.2)


def test_calibration_ignores_unscored_rows() -> None:
    rows = [
        {"alpha": 0.2, "covered": True},
        {"alpha": 0.2, "covered": False},
        {"alpha": 0.2, "covered": None},  # not yet resolved
        {"alpha": 0.1, "covered": True},
    ]
    cal = emit.build_calibration(rows)
    by_alpha = {r["alpha"]: r for r in cal}
    assert by_alpha[0.2]["n"] == 2
    assert by_alpha[0.2]["empirical_coverage"] == pytest.approx(0.5)
    assert by_alpha[0.2]["nominal_coverage"] == pytest.approx(0.8)
    assert by_alpha[0.1]["empirical_coverage"] == pytest.approx(1.0)


def _seed(db: Path, as_of: str) -> None:
    with connect(db) as conn:
        init_schema(conn)
        conn.execute(
            "INSERT INTO races (race_id, cycle, office, state, district) VALUES "
            "('2026-HOUSE-CA-01', 2026, 'HOUSE', 'CA', '01'),"
            "('2026-HOUSE-CA-02', 2026, 'HOUSE', 'CA', '02')"
        )
        for rid, win, ab, reason in (
            ("2026-HOUSE-CA-01", 0.72, False, None),
            ("2026-HOUSE-CA-02", 0.30, True, "width>40"),
        ):
            conn.execute(
                """
                INSERT INTO predictions
                    (race_id, as_of, plan_generation, model_version, median_margin,
                     lo_50, hi_50, lo_80, hi_80, lo_90, hi_90, win_prob_dem,
                     mondrian_group, abstain, abstain_reason)
                VALUES (?, ?, 0, ?, 3.0, 1,5, -2,8, -5,11, ?, 'incD|polled3+|stable|HOUSE', ?, ?)
                """,
                [rid, as_of, emit.MODEL_VERSION, win, ab, reason],
            )


def test_run_emits_all_artifacts(tmp_path: Path) -> None:
    db = tmp_path / "warehouse.duckdb"
    as_of = "2026-10-15"
    _seed(db, as_of)
    ctx = RunContext(db_path=db, cutoff_date=date.fromisoformat(as_of))

    result = emit.run(ctx)
    assert result.rows == 2

    site = db.parent / "site"
    for name in (
        "manifest.json",
        "races.json",
        "no_call.json",
        "expected_seats.json",
        "calibration.json",
    ):
        assert (site / name).exists(), f"missing {name}"

    races = json.loads((site / "races.json").read_text())
    assert {c["race_id"] for c in races} == {"2026-HOUSE-CA-01", "2026-HOUSE-CA-02"}
    no_call = json.loads((site / "no_call.json").read_text())
    assert [c["race_id"] for c in no_call] == ["2026-HOUSE-CA-02"]

    # Each prediction logged its three intervals to the audit trail.
    with connect(db) as conn:
        n = conn.execute("SELECT count(*) FROM calibration_log").fetchone()[0]
    assert n == 2 * 3


def test_run_without_predictions_fails_loud(tmp_path: Path) -> None:
    db = tmp_path / "warehouse.duckdb"
    with connect(db) as conn:
        init_schema(conn)
    ctx = RunContext(db_path=db, cutoff_date=date(2026, 10, 15))
    with pytest.raises(ValueError, match="no predictions"):
        emit.run(ctx)


def test_manifest_is_deterministic_given_timestamp() -> None:
    m = emit.build_manifest(
        run_id="2026-10-15-0.0.1",
        generated_at=datetime(2026, 10, 15, 12, 0, 0),
        as_of="2026-10-15",
        plan_generation=0,
        n_races=435,
    )
    assert m["run_id"] == "2026-10-15-0.0.1"
    assert m["n_races"] == 435
    assert m["generated_at"] == "2026-10-15T12:00:00"
