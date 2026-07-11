"""Historical backfill orchestration entrypoint (2006-2024).

Phase 0 acceptance test:

    python pipelines/backfill.py --dry-run

walks the full DAG end to end on stub data (ingest -> geo -> features ->
members -> stack -> conformal -> simulate -> backtest), touching no external
source. Real execution arrives per-phase as stage ``run`` functions land.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from midterms26.context import DEFAULT_RAW_DIR
from midterms26.logging import configure, get_logger
from midterms26.pipeline import RunContext, build_backfill_dag
from midterms26.warehouse import DEFAULT_DB_PATH, connect, init_schema

log = get_logger("backfill")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Historical backfill DAG runner.")
    parser.add_argument("--dry-run", action="store_true", help="Walk the DAG on stub data.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Cached raw downloads."
    )
    parser.add_argument("--cutoff", type=str, default=None, help="Feature freeze date YYYY-MM-DD.")
    parser.add_argument(
        "--allow-fetch", action="store_true", help="Permit live network downloads (needs keys)."
    )
    args = parser.parse_args(argv)

    configure()
    dag = build_backfill_dag()
    order = dag.topological_order()
    log.info("backfill.start", dry_run=args.dry_run, n_nodes=len(order))

    if not args.dry_run:
        with connect(args.db_path) as conn:
            init_schema(conn)
        log.info("warehouse.ready", db_path=str(args.db_path))

    cutoff = date.fromisoformat(args.cutoff) if args.cutoff else None
    ctx = RunContext(
        dry_run=args.dry_run,
        db_path=args.db_path,
        raw_dir=args.raw_dir,
        cutoff_date=cutoff,
        allow_fetch=args.allow_fetch,
    )
    results = dag.walk(ctx)

    total = sum(r.rows for r in results)
    log.info("backfill.done", n_steps=len(results), total_rows=total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
