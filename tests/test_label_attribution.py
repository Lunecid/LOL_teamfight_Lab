"""Events on the other side of the map must not decide an engagement's label."""
import numpy as np

from core.config import cfg
from core.contract import NODE_IDX
from gameplay import labels as L


def _cache(events):
    T = 3
    node = np.zeros((T, 10, len(NODE_IDX)), dtype=np.float32)
    node[:, :, NODE_IDX["alive"]] = 1.0
    return {"minute_ts": np.array([0, 60000, 120000], dtype=np.int64), "node_minute": node,
            "global_minute": np.zeros((T, 26), np.float32), "gold_team_minute": np.zeros((T, 2), np.float32),
            "events": events, "events_ts": np.array([e["timestamp"] for e in events], dtype=np.int64)}


def _interp(cache, ms):
    return cache["node_minute"][0], cache["global_minute"][0]


TM = {i: (100 if i <= 5 else 200) for i in range(1, 11)}
NEAR = {"x": 5000, "y": 5000}
FAR = {"x": 13000, "y": 13000}


def _events():
    return [{"type": "CHAMPION_KILL", "timestamp": 20000, "killerId": 1, "victimId": 6, "bounty": 300, "shutdownBounty": 0,
             "assistingParticipantIds": [], "position": NEAR},
            {"type": "CHAMPION_KILL", "timestamp": 22000, "killerId": 7, "victimId": 2, "bounty": 300, "shutdownBounty": 700,
             "assistingParticipantIds": [], "position": FAR}]


def _run(label_type, monkeypatch):
    monkeypatch.setattr(cfg, "LABEL_TYPE", label_type)
    monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", "drop", raising=False)
    monkeypatch.setattr(cfg, "LABEL_EVENT_ATTRIBUTION", "engagement")
    monkeypatch.setattr(cfg, "LABEL_ATTRIBUTION_RADIUS_U", 4264.0)
    monkeypatch.setattr(L, "_event_price_table", lambda: {})
    ev = _events()
    return L.compute_label(_cache(ev), TM, -1, engage_ts=10000, label_end_ts=40000, horizon_ms=30000,
                           first_kill_ts=20000, last_kill_ts=20000, interp_node_global=_interp, anchor_xy=(5000.0, 5000.0))


def test_far_kill_is_not_attributed(monkeypatch):
    assert _run("market_event", monkeypatch) == 1          # only the near blue kill counts


def test_window_suffix_keeps_map_wide_behaviour(monkeypatch):
    assert _run("market_event@window", monkeypatch) == 0    # far red kill worth 1,000 outweighs 300


def test_without_anchor_everything_is_kept(monkeypatch):
    monkeypatch.setattr(cfg, "LABEL_TYPE", "market_event")
    monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", "drop", raising=False)
    monkeypatch.setattr(L, "_event_price_table", lambda: {})
    ev = _events()
    assert L.compute_label(_cache(ev), TM, -1, engage_ts=10000, label_end_ts=40000, horizon_ms=30000,
                           first_kill_ts=20000, last_kill_ts=20000, interp_node_global=_interp) == 0
