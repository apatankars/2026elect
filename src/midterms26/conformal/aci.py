"""Adaptive Conformal Inference — online alpha-adjustment (Phase 4, live).

Once the daily update stream is live, ACI self-corrects empirical coverage as the
cycle drifts:
    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
where ``err_t`` is 1 if the previous interval missed, else 0. Long-run coverage
converges to ``alpha_target`` regardless of drift; verified on simulated drift in
tests/conformal.
"""

from __future__ import annotations

STAGE = "conformal.aci"

DEFAULT_GAMMA = 0.03
