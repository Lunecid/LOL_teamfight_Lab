from __future__ import annotations

from common import Any, Dict, List, Optional, np
from config import cfg


def _get_horizon_ms() -> int:
    """Prefer cfg.FIGHT_HORIZON_SEC if exists; fallback to cfg.FIGHT_HORIZON_MIN."""
    if hasattr(cfg, "FIGHT_HORIZON_SEC"):
        sec = int(getattr(cfg, "FIGHT_HORIZON_SEC", 60))
        return max(1, sec) * 1000
    horizon_min = int(getattr(cfg, "FIGHT_HORIZON_MIN", 1))
    return max(1, horizon_min) * 60_000


def _get_context_ms() -> int:
    """Context window in ms."""
    ctx_min = int(getattr(cfg, "FIGHT_CONTEXT_MIN", 1))
    return max(1, ctx_min) * 60_000


def _get_bin_ms() -> int:
    """Bin size in ms."""
    return int(getattr(cfg, "BIN_MS", 10000))


def gold_at_ms(cache: Dict[str, Any], q_ms: int, method: str = "linear") -> np.ndarray:
    """
    gold_team_minute is (T,2) sampled at cache['minute_ts'] boundaries.
    Return gold at arbitrary ms.

    Supported methods:
      - "linear": linear interpolation between neighbor minute frames
      - "ffill": forward-fill / zero-order hold (piecewise constant)
      - "zoh":   alias of ffill
      - "none":  alias of ffill
      - "bfill": backward-fill (use next frame)
    """
    ts = cache["minute_ts"]
    g = cache["gold_team_minute"]

    if len(ts) == 0:
        return np.zeros((2,), dtype=np.float32)

    if len(ts) == 1:
        return g[0].astype(np.float32)

    idx = int(np.searchsorted(ts, q_ms) - 1)
    i = max(0, idx)
    j = min(len(ts) - 1, idx + 1)

    m = str(method or "linear").lower().strip()

    # ---- ffill / zoh / none ----
    if m in ("ffill", "zoh", "none"):
        return g[i].astype(np.float32)

    # ---- bfill ----
    if m in ("bfill",):
        return g[j].astype(np.float32)

    # ---- linear ----
    if ts[j] == ts[i]:
        return g[i].astype(np.float32)

    alpha = float(q_ms - ts[i]) / float(ts[j] - ts[i])
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - alpha) * g[i] + alpha * g[j]).astype(np.float32)


def ts_to_minute_idx(minute_ts: np.ndarray, q_ms: int) -> int:
    """Convert ms timestamp to minute index (compat)."""
    if len(minute_ts) == 0:
        return 0
    idx = int(np.searchsorted(minute_ts, q_ms, side="right") - 1)
    return int(np.clip(idx, 0, len(minute_ts) - 1))


def minute_idx_to_ts(minute_ts: np.ndarray, m_idx: int) -> int:
    """Convert minute index to ms timestamp."""
    if len(minute_ts) == 0:
        return 0
    m_idx = int(np.clip(m_idx, 0, len(minute_ts) - 1))
    return int(minute_ts[m_idx])


def validate_engage_ts(cache: Dict[str, Any], engage_ts: int, horizon_ms: int, ctx_ms: int) -> bool:
    """
    Validate that engage_ts has enough context before and horizon after.
    Returns True if valid.
    """
    minute_ts = cache.get("minute_ts", np.array([]))
    if len(minute_ts) < 2:
        return False

    t_min = int(minute_ts[0])
    t_max = int(minute_ts[-1])

    # Need ctx_ms before engage_ts
    if int(engage_ts) - int(ctx_ms) < t_min:
        return False

    # Need horizon_ms after engage_ts
    if int(engage_ts) + int(horizon_ms) > t_max:
        return False

    return True
