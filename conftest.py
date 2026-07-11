"""Test bootstrap: ensure ``src`` and the repo root are importable.

The root-level presence of this file makes pytest add the repo root to
``sys.path``; we also add ``src`` so ``import midterms26`` and
``import pipelines`` both resolve without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
