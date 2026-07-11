"""Warehouse schema: all tables created, idempotent, leakage columns present."""

from __future__ import annotations

from pathlib import Path

from midterms26.warehouse import (
    ALL_TABLES,
    AS_OF_TABLES,
    connect,
    existing_tables,
    init_schema,
)


def test_init_schema_creates_all_tables(tmp_path: Path) -> None:
    with connect(tmp_path / "wh.duckdb") as conn:
        init_schema(conn)
        assert existing_tables(conn) >= set(ALL_TABLES)


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "wh.duckdb"
    with connect(db) as conn:
        init_schema(conn)
        init_schema(conn)  # must not raise
        assert existing_tables(conn) >= set(ALL_TABLES)


def test_as_of_tables_have_as_of_column(tmp_path: Path) -> None:
    with connect(tmp_path / "wh.duckdb") as conn:
        init_schema(conn)
        for table in AS_OF_TABLES:
            cols = {
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                    [table],
                ).fetchall()
            }
            assert "as_of" in cols, f"{table} must carry an as_of leakage column"


def test_predictions_keyed_by_plan_generation(tmp_path: Path) -> None:
    with connect(tmp_path / "wh.duckdb") as conn:
        init_schema(conn)
        cols = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'predictions'"
            ).fetchall()
        }
        # Immutability-by-plan-generation requires these keys.
        assert {"plan_generation", "as_of", "model_version"} <= cols
