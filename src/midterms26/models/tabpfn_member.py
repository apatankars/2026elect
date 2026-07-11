"""TabPFN v2 second learner — Phase 4.

Tabular foundation model, in-context learning, no hyperparameter tuning; our
~3-4k race-cycle x ~100 feature dataset sits in its sweet spot. Emits a full
predictive quantile grid per race natively. Wrapped in a versioned adapter so the
member can be swapped.

Risk (plan §4): verify TabPFN licensing / inference constraints for a nightly
job; fallback is an NGBoost or quantile-forest member (still not LightGBM).

Requires the ``models`` extra. Phase 4 fills in ``run``.
"""

from __future__ import annotations

from midterms26.stubs import not_implemented, stub

STAGE = "models.tabpfn"

run = not_implemented(STAGE, "Phase 4")
dry_run = stub(STAGE, rows=3_600, detail="TabPFN predictive quantile grid (stub)")
