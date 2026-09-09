"""market_event: kill gold from the event, other events from the price table, dead zone, refinement."""
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


def _label(events, table, monkeypatch, deadzone=300.0):
    monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", "drop", raising=False)
    monkeypatch.setattr(cfg, "LABEL_GOLD_DEADZONE", deadzone)
    monkeypatch.setattr(L, "_event_price_table", lambda: table)
    cache = _cache(events)
    return L._compute_label_market_event(events, TM, cache, 10000, 40000, interp_node_global=_interp,
                                         first_kill_ts=20000, last_kill_ts=30000)


def test_kill_gold_alone_decides_beyond_deadzone(monkeypatch):
    ev = [{"type": "CHAMPION_KILL", "timestamp": 20000, "killerId": 1, "victimId": 6, "bounty": 300, "shutdownBounty": 150,
           "assistingParticipantIds": []}]
    assert _label(ev, {}, monkeypatch) == 1


def test_priced_structure_flips_a_kill_trade(monkeypatch):
    ev = [{"type": "CHAMPION_KILL", "timestamp": 20000, "killerId": 1, "victimId": 6, "bounty": 300, "shutdownBounty": 0, "assistingParticipantIds": []},
          {"type": "CHAMPION_KILL", "timestamp": 25000, "killerId": 6, "victimId": 1, "bounty": 300, "shutdownBounty": 0, "assistingParticipantIds": []},
          {"type": "ELITE_MONSTER_KILL", "timestamp": 35000, "killerTeamId": 200, "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON"}]
    assert _label(ev, {"dragon": 400.0}, monkeypatch) == 0          # red: kills even, dragon worth 400 > dead zone
    assert _label(ev, {"dragon": 100.0}, monkeypatch) == 0          # inside dead zone -> refinement: kills even, survivors even, structures -> red


def test_assist_gold_counts_per_assister(monkeypatch):
    ev = [{"type": "CHAMPION_KILL", "timestamp": 20000, "killerId": 1, "victimId": 6, "bounty": 200, "shutdownBounty": 0,
           "assistingParticipantIds": [2, 3]}]
    assert _label(ev, {"assists": 75.0}, monkeypatch) == 1           # 200 + 2*75 = 350 > 300
    assert _label(ev, {}, monkeypatch) == 1                           # 200 inside dead zone -> refinement: blue has the kill
