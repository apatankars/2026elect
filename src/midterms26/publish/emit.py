"""JSON emitters + calibration dashboard data (Phase 6).

Emits static JSON artifacts consumed by the (separate) site repo: the per-race
table with interval bars, "No Call" races with reasons, an expected-seats summary,
the live calibration dashboard (reliability data), and a run manifest — the
audit trail *is* the product. Every run also appends its predicted intervals to
``calibration_log`` (``covered`` NULL until the outcome resolves), so coverage-so-
far can be scored later without re-running the model.

Pure-Python + DuckDB (no numpy): the builders below take plain prediction dicts
and return JSON-able structures, so they snapshot-test without a database; the DB
layer (:func:`load_predictions`, :func:`run`) is a thin read/write shell. The
expected-seats number uses linearity of expectation — ``sum(win_prob_dem)`` is the
exact expected D seat count even under race correlation, so this summary is honest
without the full copula joint (which lands the seat *distribution* separately).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from midterms26.context import RunContext
from midterms26.dag import StepResult
from midterms26.logging import get_logger
from midterms26.warehouse import connect, init_schema

STAGE = "publish.emit"
log = get_logger(STAGE)

MODEL_VERSION = "0.0.1"

# alpha -> (lo column, hi column) on the predictions table.
_INTERVAL_COLUMNS: tuple[tuple[float, str, str], ...] = (
    (0.5, "lo_50", "hi_50"),
    (0.2, "lo_80", "hi_80"),
    (0.1, "lo_90", "hi_90"),
)

Pred = Mapping[str, Any]


def _intervals(pred: Pred) -> dict[str, dict[str, float | None]]:
    """Pull the three conformal intervals off a prediction row, keyed by nominal %."""
    out: dict[str, dict[str, float | None]] = {}
    for alpha, lo_col, hi_col in _INTERVAL_COLUMNS:
        pct = round((1.0 - alpha) * 100)
        out[str(pct)] = {"lo": pred.get(lo_col), "hi": pred.get(hi_col)}
    return out


def race_card(pred: Pred) -> dict[str, Any]:
    """One race's site payload: point, intervals, win prob, group, No-Call state."""
    return {
        "race_id": pred["race_id"],
        "office": pred.get("office"),
        "median_margin": pred.get("median_margin"),
        "intervals": _intervals(pred),
        "win_prob_dem": pred.get("win_prob_dem"),
        "mondrian_group": pred.get("mondrian_group"),
        "abstain": bool(pred.get("abstain", False)),
        "abstain_reason": pred.get("abstain_reason"),
    }


def build_race_table(preds: Sequence[Pred]) -> list[dict[str, Any]]:
    """Per-race cards, sorted by race id for stable diffs."""
    return [race_card(p) for p in sorted(preds, key=lambda p: str(p["race_id"]))]


def build_no_call(preds: Sequence[Pred]) -> list[dict[str, Any]]:
    """The "No Call" races, each with its stable abstention reason."""
    return [
        {"race_id": p["race_id"], "office": p.get("office"), "reason": p.get("abstain_reason")}
        for p in sorted(preds, key=lambda p: str(p["race_id"]))
        if bool(p.get("abstain", False))
    ]


def _seat_bucket(win_prob_dem: float | None) -> str:
    if win_prob_dem is None:
        return "unknown"
    if win_prob_dem >= 0.85:
        return "safe_d"
    if win_prob_dem >= 0.6:
        return "lean_d"
    if win_prob_dem > 0.4:
        return "tossup"
    if win_prob_dem > 0.15:
        return "lean_r"
    return "safe_r"


def build_expected_seats(preds: Sequence[Pred]) -> dict[str, dict[str, Any]]:
    """Per-office expected D seats + rating-bucket counts.

    ``expected_dem_seats`` = ``sum(win_prob_dem)`` over contested, non-abstaining
    races — exact in expectation even with correlated races. Abstentions are
    counted separately and excluded from the expectation.
    """
    n_races: dict[str, int] = defaultdict(int)
    n_abstain: dict[str, int] = defaultdict(int)
    expected: dict[str, float] = defaultdict(float)
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in preds:
        office = str(p.get("office") or "UNKNOWN")
        n_races[office] += 1
        if bool(p.get("abstain", False)):
            n_abstain[office] += 1
            continue
        wp = p.get("win_prob_dem")
        if wp is not None:
            expected[office] += float(wp)
        buckets[office][_seat_bucket(wp)] += 1
    return {
        office: {
            "n_races": n_races[office],
            "n_abstain": n_abstain[office],
            "expected_dem_seats": round(expected[office], 3),
            "buckets": dict(buckets[office]),
        }
        for office in sorted(n_races)
    }


def build_calibration(log_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reliability data from scored ``calibration_log`` rows, grouped by alpha.

    Only rows with a non-null ``covered`` (outcome resolved) contribute. Emits
    nominal vs empirical coverage per alpha — the calibration dashboard's series.
    """
    counts: dict[float, int] = defaultdict(int)
    hits: dict[float, int] = defaultdict(int)
    for r in log_rows:
        if r.get("covered") is None:
            continue
        alpha = float(r["alpha"])
        counts[alpha] += 1
        hits[alpha] += 1 if r["covered"] else 0
    return [
        {
            "alpha": alpha,
            "nominal_coverage": round(1.0 - alpha, 4),
            "empirical_coverage": round(hits[alpha] / counts[alpha], 4),
            "n": counts[alpha],
        }
        for alpha in sorted(counts)
    ]


def build_seat_distribution(forecast_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-office joint seat forecast for the site (from ``seat_forecast`` rows)."""
    out: dict[str, Any] = {}
    for r in forecast_rows:
        out[str(r["office"])] = {
            "n_races": r["n_races"],
            "majority_threshold": r["majority_threshold"],
            "expected_dem_seats": round(float(r["expected_dem_seats"]), 2),
            "p_dem_majority": round(float(r["p_dem_majority"]), 4),
            "seats_p10": r["seats_p10"],
            "seats_p50": r["seats_p50"],
            "seats_p90": r["seats_p90"],
            "histogram": json.loads(r["histogram"]) if r.get("histogram") else {},
        }
    return out


def build_manifest(
    *,
    run_id: str,
    generated_at: datetime,
    as_of: str,
    plan_generation: int,
    n_races: int,
) -> dict[str, Any]:
    """Run manifest — the honesty-ledger header stamped on every artifact set."""
    return {
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "model_version": MODEL_VERSION,
        "as_of": as_of,
        "plan_generation": plan_generation,
        "n_races": n_races,
    }


def load_predictions(
    conn: Any, as_of: str, plan_generation: int, model_version: str = MODEL_VERSION
) -> list[dict[str, Any]]:
    """Read prediction rows for one (as_of, plan_generation), joined to office."""
    rows = conn.execute(
        """
        SELECT p.*, r.office
        FROM predictions p
        LEFT JOIN races r USING (race_id)
        WHERE p.as_of = ? AND p.plan_generation = ? AND p.model_version = ?
        """,
        [as_of, plan_generation, model_version],
    )
    cols = [d[0] for d in rows.description]
    return [dict(zip(cols, row, strict=True)) for row in rows.fetchall()]


def append_calibration_log(
    conn: Any, run_id: str, logged_at: datetime, preds: Sequence[Pred], as_of: str
) -> int:
    """Append each prediction's intervals to ``calibration_log`` (covered NULL)."""
    n = 0
    for p in preds:
        for alpha, lo_col, hi_col in _INTERVAL_COLUMNS:
            conn.execute(
                """
                INSERT OR REPLACE INTO calibration_log
                    (run_id, logged_at, race_id, as_of, model_version, alpha,
                     interval_lo, interval_hi, mondrian_group, realized_margin, covered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                [
                    run_id,
                    logged_at,
                    p["race_id"],
                    as_of,
                    MODEL_VERSION,
                    alpha,
                    p.get(lo_col),
                    p.get(hi_col),
                    p.get("mondrian_group"),
                ],
            )
            n += 1
    return n


def _write_json(out_dir: Path, name: str, payload: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def write_artifacts(
    out_dir: Path,
    preds: Sequence[Pred],
    calibration_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    seat_forecast_rows: Sequence[Mapping[str, Any]] = (),
) -> list[Path]:
    """Write the full JSON artifact set; returns the paths written."""
    return [
        _write_json(out_dir, "manifest.json", manifest),
        _write_json(out_dir, "races.json", build_race_table(preds)),
        _write_json(out_dir, "no_call.json", build_no_call(preds)),
        _write_json(out_dir, "expected_seats.json", build_expected_seats(preds)),
        _write_json(out_dir, "seat_distribution.json", build_seat_distribution(seat_forecast_rows)),
        _write_json(out_dir, "calibration.json", build_calibration(calibration_rows)),
    ]


def run(ctx: RunContext) -> StepResult:
    """Emit static JSON artifacts + append to ``calibration_log`` for one as-of."""
    if ctx.cutoff_date is None:
        raise ValueError("publish.emit needs ctx.cutoff_date (the as-of date)")
    as_of = ctx.cutoff_date.isoformat()
    plan_generation = 0
    generated_at = datetime.now(UTC)
    run_id = f"{as_of}-{MODEL_VERSION}"
    out_dir = ctx.db_path.parent / "site"

    with connect(ctx.db_path) as conn:
        init_schema(conn)
        preds = load_predictions(conn, as_of, plan_generation)
        if not preds:
            raise ValueError(
                f"no predictions at as_of={as_of} plan_generation={plan_generation}; "
                "run the conformal stage first"
            )
        append_calibration_log(conn, run_id, generated_at, preds, as_of)
        cal_rows = conn.execute("SELECT * FROM calibration_log").pl().to_dicts()
        seat_rows = (
            conn.execute(
                "SELECT * FROM seat_forecast WHERE as_of = ? AND plan_generation = ?",
                [as_of, plan_generation],
            )
            .pl()
            .to_dicts()
        )

    manifest = build_manifest(
        run_id=run_id,
        generated_at=generated_at,
        as_of=as_of,
        plan_generation=plan_generation,
        n_races=len(preds),
    )
    paths = write_artifacts(out_dir, preds, cal_rows, manifest, seat_rows)
    log.info("publish.done", as_of=as_of, n_races=len(preds), files=len(paths))
    return StepResult(
        node=STAGE, rows=len(preds), detail=f"{len(paths)} JSON artifacts -> {out_dir}"
    )


def dry_run(ctx: RunContext) -> StepResult:  # noqa: ARG001
    return StepResult(
        node=STAGE,
        rows=1,
        detail="static JSON artifacts + calibration_log append (stub)",
        dry_run=True,
    )
