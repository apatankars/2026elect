"""Nightly live pipeline orchestration entrypoint.

ingest -> features -> fit -> conformal -> simulate -> emit JSON -> publish.
Every run appends to ``calibration_log`` with frozen timestamps (Phase 6). Runs
under GitHub Actions cron. Phase 0 supports ``--dry-run`` only.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from midterms26.context import DEFAULT_RAW_DIR
from midterms26.logging import configure, get_logger
from midterms26.pipeline import RunContext, build_nightly_dag
from midterms26.warehouse import DEFAULT_DB_PATH, connect, init_schema

log = get_logger("nightly")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly live DAG runner.")
    parser.add_argument("--dry-run", action="store_true", help="Walk the DAG on stub data.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Cached raw downloads."
    )
    parser.add_argument("--cutoff", type=str, default=None, help="As-of date YYYY-MM-DD.")
    parser.add_argument(
        "--allow-fetch", action="store_true", help="Permit live network downloads (needs keys)."
    )
    args = parser.parse_args(argv)

    configure()
    dag = build_nightly_dag()
    order = dag.topological_order()
    log.info("nightly.start", dry_run=args.dry_run, n_nodes=len(order))

    if not args.dry_run:
        with connect(args.db_path) as conn:
            init_schema(conn)
        log.info("warehouse.ready", db_path=str(args.db_path))

    as_of = date.fromisoformat(args.cutoff) if args.cutoff else date.today()
    ctx = RunContext(
        dry_run=args.dry_run,
        db_path=args.db_path,
        raw_dir=args.raw_dir,
        cutoff_date=as_of,
        allow_fetch=args.allow_fetch,
    )
    results = dag.walk(ctx)

    total = sum(r.rows for r in results)
    log.info("nightly.done", n_steps=len(results), total_rows=total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
