"""Frame-held features must come from the last frame at or before the cutoff, and time_norm
in the built sequence must be absolute game time."""
import numpy as np

from core.config import cfg, GLOBAL_IDX
from core.contract import NODE_IDX
from gameplay.pipeline import build_ms_sequence
from gameplay.pipeline_interp import interpolate_node_global


def _cache(T=12):
    ts = np.arange(T, dtype=np.int64) * 60000
    node = np.zeros((T, 10, len(NODE_IDX)), dtype=np.float32)
    node[:, :, NODE_IDX["level_norm"]] = (np.arange(T) / 18.0)[:, None]      # level = frame index
    node[:, :, NODE_IDX["alive"]] = 1.0
    node[:, :, NODE_IDX["x_norm"]] = 0.3
    node[:, :, NODE_IDX["y_norm"]] = 0.3
    glob = np.zeros((T, len(GLOBAL_IDX)), dtype=np.float32)
    glob[:, GLOBAL_IDX["time_norm"]] = np.arange(T) / (T - 1)
    xy = np.full((T, 10, 2), 4800.0, dtype=np.float32)
    kill_ts = 8 * 60000 + 55000
    events = [{"type": "CHAMPION_KILL", "timestamp": kill_ts, "killerId": 1, "victimId": 6, "bounty": 300,
               "shutdownBounty": 100, "assistingParticipantIds": [], "position": {"x": 4800, "y": 4800}}]
    return {"minute_ts": ts, "node_minute": node, "global_minute": glob, "gold_team_minute": np.zeros((T, 2), np.float32),
            "xy_raw_minute": xy, "events": events, "events_ts": np.array([kill_ts], dtype=np.int64),
            "meta": {"team_map": {i: (100 if i <= 5 else 200) for i in range(1, 11)}, "anchors": None}}


def test_query_at_frame_time_reads_that_frame():
    c = _cache()
    node, _ = interpolate_node_global(c, 5 * 60000, max_snapshot_ms=5 * 60000)
    assert abs(float(node[0, NODE_IDX["level_norm"]]) - 5 / 18.0) < 1e-6


def test_sequence_holds_last_frame_before_cutoff_and_absolute_time(monkeypatch):
    monkeypatch.setattr(cfg, "TIME_NORM_ABSOLUTE", True)
    monkeypatch.setattr(cfg, "TIME_NORM_DENOM_MIN", 45.0)
    monkeypatch.setattr(cfg, "LABEL_TYPE", "market_event")
    monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", "drop", raising=False)
    c = _cache()
    tau = 8 * 60000 + 40000                      # 8:40 -> last frame at or before the cutoff is frame 8
    raw = build_ms_sequence(c, c["meta"]["team_map"], -1, engage_ts=tau, label_end_ts=tau + 35000,
                            first_kill_ts=tau + 15000, last_kill_ts=tau + 15000)
    assert raw is not None
    node_seq = np.asarray(raw["node_seq"])
    assert abs(float(node_seq[-1, 0, NODE_IDX["level_norm"]]) - 8 / 18.0) < 1e-6
    glob_seq = np.asarray(raw["glob_seq"])
    q_last = (tau - 30000) + 5 * 5000 + 2500
    assert abs(float(glob_seq[-1, GLOBAL_IDX["time_norm"]]) - q_last / (45 * 60000.0)) < 1e-6
