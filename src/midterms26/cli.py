"""``midterms26`` CLI — thin wrapper over pipeline entrypoints.

Subcommands:
  * ``init-db``   — create the DuckDB warehouse schema.
  * ``backfill``  — walk the historical backfill DAG (``--dry-run`` for Phase 0).
  * ``nightly``   — walk the live nightly DAG (``--dry-run`` for Phase 0).
  * ``show-dag``  — print the topological order of a graph.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from midterms26 import __version__
from midterms26.context import DEFAULT_RAW_DIR
from midterms26.logging import configure, get_logger
from midterms26.pipeline import (
    RunContext,
    build_backfill_dag,
    build_ingest_dag,
    build_nightly_dag,
)
from midterms26.warehouse import DEFAULT_DB_PATH, connect, init_schema

app = typer.Typer(add_completion=False, help="Calibration-native 2026 midterms forecast.")
log = get_logger("cli")

DEFAULT_DEMO_DB = Path("data/demo.duckdb")


@app.callback()
def _root() -> None:
    configure()


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("init-db")
def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the DuckDB warehouse schema (idempotent)."""
    with connect(db_path) as conn:
        init_schema(conn)
    log.info("warehouse.initialized", db_path=str(db_path))


def _run_dag(name: str, dry_run: bool, db_path: Path, cutoff: date | None) -> None:
    dag = build_backfill_dag() if name == "backfill" else build_nightly_dag()
    # Backfill fits LOCO folds (the stack + backtest need them); nightly is live-only.
    ctx = RunContext(
        dry_run=dry_run, db_path=db_path, cutoff_date=cutoff, do_loco=name == "backfill"
    )
    order = dag.topological_order()
    log.info("dag.start", pipeline=name, dry_run=dry_run, n_nodes=len(order))
    results = dag.walk(ctx)
    total = sum(r.rows for r in results)
    log.info("dag.done", pipeline=name, n_steps=len(results), total_rows=total)


@app.command()
def backfill(
    dry_run: bool = typer.Option(False, "--dry-run", help="Walk the DAG on stub data."),
    db_path: Path = DEFAULT_DB_PATH,
    cutoff: str | None = typer.Option(None, help="Feature freeze date YYYY-MM-DD."),
) -> None:
    """Walk the historical backfill DAG (2006-2024)."""
    _run_dag("backfill", dry_run, db_path, date.fromisoformat(cutoff) if cutoff else None)


@app.command()
def nightly(
    dry_run: bool = typer.Option(False, "--dry-run", help="Walk the DAG on stub data."),
    db_path: Path = DEFAULT_DB_PATH,
    cutoff: str | None = typer.Option(None, help="As-of date YYYY-MM-DD (defaults to today)."),
) -> None:
    """Walk the live nightly DAG."""
    as_of = date.fromisoformat(cutoff) if cutoff else date.today()
    _run_dag("nightly", dry_run, db_path, as_of)


@app.command()
def ingest(
    db_path: Path = DEFAULT_DB_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    allow_fetch: bool = typer.Option(
        False, "--allow-fetch", help="Permit live network downloads (needs API keys)."
    ),
) -> None:
    """Run the Phase 1 ingest-only DAG against cached raw files in --raw-dir."""
    with connect(db_path) as conn:
        init_schema(conn)
    ctx = RunContext(dry_run=False, db_path=db_path, raw_dir=raw_dir, allow_fetch=allow_fetch)
    dag = build_ingest_dag()
    log.info("ingest.start", raw_dir=str(raw_dir), db_path=str(db_path), allow_fetch=allow_fetch)
    results = dag.walk(ctx)
    log.info("ingest.done", n_steps=len(results), total_rows=sum(r.rows for r in results))


@app.command()
def demo(
    db_path: Path = DEFAULT_DEMO_DB,
    cutoff: str = typer.Option("2026-10-15", help="As-of date YYYY-MM-DD for the live forecast."),
    seed: int = typer.Option(0, help="Synthetic-data seed."),
) -> None:
    """Seed a synthetic warehouse and run the full model spine end-to-end to JSON.

    Requires the ``models`` extra (numpyro/tabpfn). Emits the site JSON artifacts
    next to ``--db-path`` (in a ``site/`` folder).
    """
    from midterms26.demo import run_demo  # local import: pulls the models stack

    site = run_demo(db_path, cutoff=date.fromisoformat(cutoff), seed=seed)
    typer.echo(f"\nForecast artifacts written to: {site}")
    for f in sorted(site.glob("*.json")):
        typer.echo(f"  - {f.name}")


@app.command("show-dag")
def show_dag(pipeline: str = typer.Argument("backfill")) -> None:
    """Print the topological execution order of a pipeline graph."""
    dag = (
        build_ingest_dag()
        if pipeline == "ingest"
        else build_backfill_dag()
        if pipeline == "backfill"
        else build_nightly_dag()
    )
    for i, name in enumerate(dag.topological_order(), start=1):
        deps = ", ".join(dag.nodes[name].deps) or "(root)"
        typer.echo(f"{i:2d}. {name:<22} <- {deps}")


if __name__ == "__main__":
    app()
