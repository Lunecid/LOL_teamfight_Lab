"""time_norm in the feature path must depend on game time only, never on the match's length."""
import numpy as np

from core.config import cfg, GLOBAL_IDX
from gameplay.pipeline_interp import interpolate_node_global


def _cache(n_frames: int):
    T = n_frames
    ts = np.arange(T, dtype=np.int64) * 60000
    node = np.zeros((T, 10, len(__import__("core.contract", fromlist=["NODE_IDX"]).NODE_IDX)), dtype=np.float32)
    glob = np.zeros((T, len(GLOBAL_IDX)), dtype=np.float32)
    tj = GLOBAL_IDX["time_norm"]
    glob[:, tj] = np.arange(T) / max(1, T - 1)      # the cached (match-length-relative) value
    return {"minute_ts": ts, "node_minute": node, "global_minute": glob}


def test_time_norm_is_absolute_and_length_independent(monkeypatch):
    monkeypatch.setattr(cfg, "TIME_NORM_ABSOLUTE", True)
    monkeypatch.setattr(cfg, "TIME_NORM_DENOM_MIN", 45.0)
    q = 10 * 60000 + 30000                                  # 10.5 minutes into the game
    tj = GLOBAL_IDX["time_norm"]
    _, g_short = interpolate_node_global(_cache(20), q, max_snapshot_ms=q)
    _, g_long = interpolate_node_global(_cache(45), q, max_snapshot_ms=q)
    assert abs(float(g_short[tj]) - 10.5 / 45.0) < 1e-6
    assert abs(float(g_long[tj]) - 10.5 / 45.0) < 1e-6


def test_legacy_value_depends_on_match_length(monkeypatch):
    monkeypatch.setattr(cfg, "TIME_NORM_ABSOLUTE", False)
    monkeypatch.setattr(cfg, "ZERO_XY_NODE_FEATURES", False)
    q = 10 * 60000
    tj = GLOBAL_IDX["time_norm"]
    _, g_short = interpolate_node_global(_cache(20), q, max_snapshot_ms=q)
    _, g_long = interpolate_node_global(_cache(45), q, max_snapshot_ms=q)
    assert float(g_short[tj]) != float(g_long[tj])
