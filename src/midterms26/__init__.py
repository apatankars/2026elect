"""midterms26 — calibration-native 2026 midterms election forecast.

Two integrated workstreams:
  A. Redistricting-native fundamentals pipeline (consistent historical baseline
     for every 2026 district, including mid-cycle redraws).
  B. A forecast whose *uncertainty* is the product: conformal prediction with
     finite-sample coverage, group-conditional calibration, online adaptation,
     and explicit abstention ("No Call").

Phase 0 (this scaffold): package layout, DuckDB warehouse schema, and a DAG
that `pipelines/backfill.py --dry-run` walks end to end on stub data.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
