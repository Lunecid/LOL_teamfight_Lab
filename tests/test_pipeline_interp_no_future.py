from __future__ import annotations

import numpy as np

from core.config import F_EVENT, F_GLOBAL, F_NODE, ITEM_HASH_NAMES, NODE_IDX, cfg
from gameplay.pipeline import build_ms_sequence
from gameplay.pipeline_interp import interpolate_node_global


def _make_interp_cache() -> dict:
    ts = np.array([0, 60_000, 120_000], dtype=np.int64)
    node_minute = np.zeros((3, 10, F_NODE), dtype=np.float32)
    global_minute = np.zeros((3, F_GLOBAL), dtype=np.float32)
    xy_raw_minute = np.zeros((3, 10, 2), dtype=np.float32)

    # Only the future frame (t=120s) has a large X so interpolation leakage is visible.
    xy_raw_minute[2, :, 0] = 16_000.0

    return {
        "minute_ts": ts,
        "node_minute": node_minute,
        "global_minute": global_minute,
        "xy_raw_minute": xy_raw_minute,
    }


def test_interpolate_node_global_can_cap_future_snapshot(monkeypatch):
    cache = _make_interp_cache()
    x_idx = int(NODE_IDX["x_norm"])

    monkeypatch.setattr(cfg, "INTERP_XY", True, raising=False)
    monkeypatch.setattr(cfg, "INTERP_XY_CURVE", "linear", raising=False)
    monkeypatch.setattr(cfg, "XY_DISCONT_DIST_RAW", 1e9, raising=False)
    monkeypatch.setattr(cfg, "INTERP_SCALARS_METHOD", "ffill", raising=False)
    monkeypatch.setattr(cfg, "ZERO_XY_NODE_FEATURES", False, raising=False)
    monkeypatch.setattr(cfg, "USE_RELATIVE_XY", False, raising=False)

    # q=90s sits between t=60s and t=120s.
    node_unclamped, _ = interpolate_node_global(cache, 90_000)
    node_clamped, _ = interpolate_node_global(cache, 90_000, max_snapshot_ms=60_000)

    assert float(node_unclamped[:, x_idx].mean()) > 0.1
    assert float(np.abs(node_clamped[:, x_idx]).max()) < 1e-6


def test_build_ms_sequence_passes_engage_guard_to_node_interp(monkeypatch):
    calls = []

    def _fake_interp(cache, q_ms, *, max_snapshot_ms=None):
        calls.append(int(max_snapshot_ms) if max_snapshot_ms is not None else None)
        node = np.zeros((10, F_NODE), dtype=np.float32)
        glob = np.zeros((F_GLOBAL,), dtype=np.float32)
        return node, glob

    def _fake_global_from_prev(cache, ref_ms, *, strict_before=True):
        return np.zeros((F_GLOBAL,), dtype=np.float32), int(ref_ms)

    def _fake_agg(cache, tm, s_ms, e_ms):
        ev = np.zeros((F_EVENT,), dtype=np.float32)
        it = np.zeros((len(ITEM_HASH_NAMES),), dtype=np.float32)
        return ev, it

    def _fake_label_targets(
        cache,
        tm,
        t_start,
        *,
        engage_ts=None,
        label_end_ts=None,
        horizon_ms=None,
        first_kill_ts=None,
        last_kill_ts=None,
    ):
        st = int(engage_ts if engage_ts is not None else 0)
        ed = int(label_end_ts if label_end_ts is not None else st + 30_000)
        return {
            "y": 0,
            "kill_diff_norm": 0.0,
            "gold_diff_norm": 0.0,
            "obj_diff_norm": 0.0,
            "kill_diff": 0.0,
            "gold_diff": 0.0,
            "obj_diff": 0.0,
            "alive_diff": 0.0,
            "summoner_spells": 0.0,
            "label_start_ms": st,
            "label_end_ms": ed,
        }

    monkeypatch.setattr(cfg, "USE_EVENT_TOKENS", False, raising=False)
    monkeypatch.setattr("gameplay.pipeline.interpolate_node_global", _fake_interp)
    monkeypatch.setattr("gameplay.pipeline.global_from_prev_snapshot", _fake_global_from_prev)
    monkeypatch.setattr("gameplay.pipeline.aggregate_events", _fake_agg)
    monkeypatch.setattr("gameplay.pipeline.compute_label_targets", _fake_label_targets)

    cache = {
        "minute_ts": np.array([0, 60_000, 120_000, 180_000], dtype=np.int64),
        "node_minute": np.zeros((4, 10, F_NODE), dtype=np.float32),
        "global_minute": np.zeros((4, F_GLOBAL), dtype=np.float32),
        "xy_raw_minute": np.zeros((4, 10, 2), dtype=np.float32),
        "events": [],
        "meta": {},
    }
    tm = {pid: (100 if pid <= 5 else 200) for pid in range(1, 11)}

    sample = build_ms_sequence(
        cache,
        tm,
        -1,
        engage_ts=125_000,
        label_end_ts=155_000,
        ctx_ms=60_000,
        bin_ms=5_000,
        horizon_ms=30_000,
        prediction_gap_ms=0,
    )

    assert sample is not None
    assert calls
    assert all(v == 124_999 for v in calls)
