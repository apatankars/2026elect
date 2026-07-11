"""Bayes + TabPFN members. Skipped unless the ``models`` extra is installed.

These exercise the real estimators (NUTS sampling, quantile GBR fallback) on small
synthetic data plus a full ``run()`` against a seeded warehouse. They import
numpy/numpyro/sklearn lazily via ``importorskip`` so the light CI stack skips them.
"""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import pytest

from midterms26.context import RunContext
from midterms26.models import bayes, tabpfn_member
from midterms26.models.base import MemberInput
from midterms26.warehouse import connect, init_schema

pytest.importorskip("numpy")


def _synth(seed: int = 0, *, live: bool = True) -> MemberInput:
    """Margins driven by cycle national + state effect + a strong linear feature."""
    rng = random.Random(seed)
    national = {2014: -3.0, 2018: 6.0, 2022: 1.0}
    state_eff = {"CA": 4.0, "TX": -5.0}
    race_ids, X, y, cycles, states, offices = [], [], [], [], [], []
    for cyc, nat in national.items():
        for k in range(12):
            st = "CA" if k % 2 == 0 else "TX"
            x = rng.uniform(-2, 2)
            margin = nat + state_eff[st] + 3.0 * x + rng.gauss(0, 0.5)
            race_ids.append(f"{cyc}-HOUSE-{st}-{k:02d}")
            X.append([x])
            y.append(margin)
            cycles.append(cyc)
            states.append(st)
            offices.append("HOUSE")
    if live:
        for k, (st, x) in enumerate([("CA", 1.5), ("CA", -1.5)]):
            race_ids.append(f"2026-HOUSE-{st}-{k:02d}")
            X.append([x])
            y.append(None)
            cycles.append(2026)
            states.append(st)
            offices.append("HOUSE")
    return MemberInput(race_ids, ["x"], X, y, cycles, states, offices)


# -- Bayes ------------------------------------------------------------------


def test_bayes_live_prediction_recovers_feature_sign() -> None:
    pytest.importorskip("numpyro")
    mi = _synth()
    fit = bayes.fit_predict(mi, num_warmup=120, num_samples=120, seed=1)
    hi = fit.grids["2026-HOUSE-CA-00"]  # x = +1.5
    lo = fit.grids["2026-HOUSE-CA-01"]  # x = -1.5
    # Positive beta: the higher-x race has the higher median.
    assert hi[0.5] > lo[0.5]
    # Quantiles are monotone and the interval is non-degenerate.
    assert hi[0.05] < hi[0.5] < hi[0.95]
    assert hi[0.95] - hi[0.05] > 0.5


def test_bayes_emits_latent_factors() -> None:
    pytest.importorskip("numpyro")
    mi = _synth()
    fit = bayes.fit_predict(mi, num_warmup=120, num_samples=120, seed=2)
    load = fit.loadings["2026-HOUSE-CA-00"]
    assert bayes.NATIONAL in load
    assert "state:CA" in load
    assert fit.idiosyncratic_sd["2026-HOUSE-CA-00"] > 0.0


def test_bayes_loco_predicts_held_out_cycle() -> None:
    pytest.importorskip("numpyro")
    mi = _synth(live=False)
    fit = bayes.fit_predict(mi, target_cycle=2018, num_warmup=120, num_samples=120, seed=3)
    assert fit.grids  # non-empty
    assert all(rid.startswith("2018-") for rid in fit.grids)


# -- TabPFN / fallback ------------------------------------------------------


def test_fallback_sklearn_monotone_and_ordered() -> None:
    pytest.importorskip("sklearn")
    mi = _synth()
    grids = tabpfn_member.fit_predict(mi, backend="sklearn", seed=0)
    hi = grids["2026-HOUSE-CA-00"]
    lo = grids["2026-HOUSE-CA-01"]
    assert list(hi.values()) == sorted(hi.values())  # monotone after rearrangement
    assert hi[0.5] > lo[0.5]  # tracks the feature


def test_auto_backend_falls_back_on_tabpfn_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sklearn")

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("no checkpoint")

    monkeypatch.setattr(tabpfn_member, "_predict_tabpfn", boom)
    mi = _synth()
    grids = tabpfn_member.fit_predict(mi, backend="auto", seed=0)
    assert grids  # fallback produced predictions despite the TabPFN failure


# -- Full run() wiring ------------------------------------------------------


def _seed_feature_matrix(db: Path, cutoff: str) -> None:
    import json

    mi = _synth(seed=5)
    with connect(db) as conn:
        init_schema(conn)
        payload = [
            (mi.race_ids[i], cutoff, json.dumps({"x": mi.X[i][0]}), mi.y[i])
            for i in range(mi.n_rows)
        ]
        conn.executemany(
            "INSERT INTO feature_matrix (race_id, cutoff_date, plan_generation, features, "
            "target_margin) VALUES (?, ?, 0, ?, ?)",
            payload,
        )


def test_tabpfn_run_writes_live_and_loco(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sklearn")
    monkeypatch.setenv("MIDTERMS26_TABPFN_BACKEND", "sklearn")
    db = tmp_path / "wh.duckdb"
    cutoff = "2026-10-01"
    _seed_feature_matrix(db, cutoff)
    ctx = RunContext(db_path=db, cutoff_date=date.fromisoformat(cutoff), do_loco=True)

    result = tabpfn_member.run(ctx)
    assert result.rows > 0
    with connect(db) as conn:
        folds = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT fold FROM member_predictions WHERE model_member='TABPFN'"
            ).fetchall()
        }
    # Live rows + one LOCO fold per labeled cycle.
    assert folds == {"live", "2014", "2018", "2022"}


def test_bayes_run_writes_grids_and_latent(tmp_path: Path) -> None:
    pytest.importorskip("numpyro")
    # Keep this fast: live-only (do_loco off), tiny sampling via a monkeypatched default.
    db = tmp_path / "wh.duckdb"
    cutoff = "2026-10-01"
    _seed_feature_matrix(db, cutoff)
    ctx = RunContext(db_path=db, cutoff_date=date.fromisoformat(cutoff), do_loco=False)

    result = bayes.run(ctx)
    assert result.rows == 2  # two live rows
    with connect(db) as conn:
        n_live = conn.execute(
            "SELECT count(*) FROM member_predictions WHERE model_member='BAYES' AND fold='live'"
        ).fetchone()[0]
        n_latent = conn.execute("SELECT count(*) FROM latent_factors").fetchone()[0]
    assert n_live == 2
    assert n_latent == 2
