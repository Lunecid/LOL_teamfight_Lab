from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimeContract:
    """Minimal time/index contract used by legacy config/indices."""

    frame_ms: int = 60000

    def coerce_t_start_minute_idx(self, minute_ts: Any, t_raw: Any) -> int:
        try:
            t = int(t_raw)
        except Exception:
            return 0

        try:
            mts = np.asarray(minute_ts, dtype=np.int64)
        except Exception:
            return max(0, t)

        if mts.size <= 0:
            return max(0, t)

        T = int(mts.size)
        if 0 <= t < T:
            return t

        idx = int(np.searchsorted(mts, t, side="right") - 1)
        return max(0, min(T - 1, idx))

