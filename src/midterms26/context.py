"""Run context shared by the DAG runner, pipeline wiring, and stage modules.

Kept in its own module so ``ingest``/``geo``/... can import the concrete
:class:`RunContext` without a circular dependency on :mod:`midterms26.pipeline`
(which imports every stage). It structurally satisfies
:class:`midterms26.dag.RunContext`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from midterms26.warehouse import DEFAULT_DB_PATH

DEFAULT_RAW_DIR = Path("data/raw")


@dataclass
class RunContext:
    """Concrete run context passed to every node.

    ``dry_run``     — stub walk, touches no source.
    ``db_path``     — DuckDB warehouse to read/write.
    ``raw_dir``     — cached immutable downloads (``data/raw/<source>/...``).
    ``cutoff_date`` — feature-freeze / as-of date for the run (may be ``None``).
    ``allow_fetch`` — permit live network downloads (default off; Phase 1 runs
                      against cached raw files unless explicitly enabled).
    ``do_loco``     — fit leave-one-cycle-out folds (backfill: needed by the
                      stack + backtest). Off for nightly, which predicts live only
                      and reuses stack weights frozen in backfill.
    """

    dry_run: bool = False
    db_path: Path = DEFAULT_DB_PATH
    raw_dir: Path = DEFAULT_RAW_DIR
    cutoff_date: date | None = None
    allow_fetch: bool = False
    do_loco: bool = False
