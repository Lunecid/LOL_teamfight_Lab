from __future__ import annotations

from typing import Dict, List

from core.config import (
    cfg,
    NODE_FEATURE_NAMES, EVENT_FEATURE_NAMES, GLOBAL_FEATURE_NAMES,
    # [P2-STRUCT-1] SSoT: Import indices directly from config.
    # Previously: dict(FEATURE_CONTRACT.node_idx) ← independent copy.
    # Now: single import path through config.py.
    NODE_IDX, EVENT_IDX, GLOBAL_IDX,
    F_NODE, F_EVENT, F_GLOBAL,
    FEATURE_CONTRACT,
)

from gameplay.features import get_spatial_feature_names

# ---------------------------------------------------------------------
# [P2-STRUCT-1] Feature/time contracts — unified via config.py
# ---------------------------------------------------------------------
# NODE_IDX, EVENT_IDX, GLOBAL_IDX, F_NODE, F_EVENT, F_GLOBAL
# are now imported directly from config.py (single source of truth).
# The try/except fallback is kept for robustness but should never fire.

class _TimeContractFallback:
    """
    Minimal fallback for TIME_CONTRACT.

    Only provides methods that this project calls.
    """
    def coerce_t_start_minute_idx(self, minute_ts, t_raw) -> int:
        # minute_ts: list/np.ndarray of minute anchor timestamps (ms)
        try:
            t = int(t_raw)
        except Exception:
            return 0
        try:
            import numpy as _np  # local import to avoid hard deps in config-only envs
            mts = _np.asarray(minute_ts, dtype=_np.int64)
            if mts.size <= 0:
                return max(0, t)
            T = int(mts.size)
            # If t already looks like a minute index, keep it.
            if 0 <= t < T:
                return t
            # If t looks like a timestamp (ms), map to the nearest prior minute index.
            idx = int(_np.searchsorted(mts, t, side="right") - 1)
            return max(0, min(T - 1, idx))
        except Exception:
            return max(0, t)

try:
    from core.config import TIME_CONTRACT as TIME_CONTRACT  # type: ignore
except Exception:
    TIME_CONTRACT = _TimeContractFallback()

# Spatial feature names are defined in features.py (single source).
SPATIAL_NAMES = list(get_spatial_feature_names())
F_SPATIAL = len(SPATIAL_NAMES)

# Item hashing feature dimension (used in aggregate_events)
ITEM_HASH_DIM = int(getattr(cfg, "ITEM_HASH_DIM", 32))

# Boolean-ish node columns that should be clipped into [0,1] after interpolation
_BOOL_NODE_COLS: List[int] = []
for nm, j in NODE_IDX.items():
    if j is None:
        continue
    if nm == "alive" or nm == "flash_ready":
        _BOOL_NODE_COLS.append(j)
    elif nm.startswith("has_") or nm.startswith("soul_"):
        _BOOL_NODE_COLS.append(j)
_BOOL_NODE_COLS = sorted(set(_BOOL_NODE_COLS))

__all__ = [
    "NODE_IDX", "EVENT_IDX", "GLOBAL_IDX",
    "F_NODE", "F_EVENT", "F_GLOBAL",
    "TIME_CONTRACT",
    "SPATIAL_NAMES", "F_SPATIAL",
    "ITEM_HASH_DIM", "_BOOL_NODE_COLS",
]