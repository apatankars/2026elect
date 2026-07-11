"""End-to-end: run the Phase 1 ingest DAG against fixtures into a temp warehouse."""

from __future__ import annotations

from pathlib import Path

import pytest

from midterms26.context import RunContext
from midterms26.ingest import base
from midterms26.pipeline import build_ingest_dag
from midterms26.warehouse import AS_OF_TABLES, connect, init_schema

RAW = Path(__file__).parent / "fixtures" / "raw"


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(dry_run=False, db_path=tmp_path / "wh.duckdb", raw_dir=RAW)


def test_ingest_dag_populates_warehouse(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
    results = build_ingest_dag().walk(ctx)
    assert all(r.rows > 0 for r in results), {r.node: r.rows for r in results}

    with connect(ctx.db_path) as conn:
        counts = {
            t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in (
                "races",
                "results_history",
                "fec_finance",
                "specials",
                "polls",
                "ratings",
                "national_indicators",
            )
        }
    # 2 house + 1 senate race across the two MIT files.
    assert counts["races"] == 3
    assert counts["results_history"] == 3
    assert counts["fec_finance"] == 2  # Q1 + Q2 snapshots for one candidate
    assert counts["specials"] == 3
    assert counts["polls"] == 2
    assert counts["ratings"] == 4
    assert counts["national_indicators"] == 3


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
    build_ingest_dag().walk(ctx)
    build_ingest_dag().walk(ctx)  # second run must not duplicate
    with connect(ctx.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM races").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM fec_finance").fetchone()[0] == 2


def test_as_of_columns_are_populated(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with connect(ctx.db_path) as conn:
        init_schema(conn)
    build_ingest_dag().walk(ctx)
    with connect(ctx.db_path) as conn:
        for table in ("polls", "ratings", "fec_finance", "national_indicators"):
            if table not in AS_OF_TABLES:
                continue
            n_null = conn.execute(f"SELECT count(*) FROM {table} WHERE as_of IS NULL").fetchone()[0]
            assert n_null == 0, f"{table} has null as_of rows"


def test_missing_raw_raises_fetch_not_allowed(tmp_path: Path) -> None:
    ctx = RunContext(dry_run=False, db_path=tmp_path / "wh.duckdb", raw_dir=tmp_path / "empty")
    with connect(ctx.db_path) as conn:
        init_schema(conn)
    from midterms26.ingest import results

    with pytest.raises(base.FetchNotAllowedError):
        results.run(ctx)
