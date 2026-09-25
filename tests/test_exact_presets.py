"""v4-exact preset: applies cleanly, leaves v3.3 untouched, new CFG defaults keep v3.3 behaviour."""
from core.config import CFG
from core.presets import PRESETS, apply_preset

# Hard-coded snapshot of PRESETS["v3.3"] before the v4-exact stage-0 change (2026-09-25).
V33_SNAPSHOT = {
    "TF2_KILL_CLUSTER_GAP_MS": 13700, "CLUSTER_MAX_DIAMETER": 4264.0,
    "TF2_VALIDITY_RADIUS": 1600.0, "TF2_ENGAGE_PRE_KILL_MS": 15000, "FIGHT_HORIZON_SEC": 35,
    "LABEL_TYPE": "market_event", "LABEL_EVENT_ATTRIBUTION": "engagement", "LABEL_GOLD_DEADZONE": 300.0,
    "TIME_NORM_ABSOLUTE": True, "TIME_NORM_DENOM_MIN": 45.0, "ANCHORS_CAUSAL": True, "TAB_FRAME_AGE_FEATURE": True,
}

NEW_FIELD_DEFAULTS = {
    "ENG_ALIVE_SOURCE": "frame", "ENG_PARTICIPATION": "kill_plus_interactions",
    "ENG_OVERLAP_RULE": "legacy_priority", "ENG_MERGE_A6": False, "ENG_ISOLATION": False,
    "ENG_CLEAN_MAX_AGE_MS": 0, "ENG_BOUNDARIES_LOCKED": False,
    "DD_TABLE_DIR": "config/game_rules/datadragon",
}

V4_OVERRIDES = {
    "ENG_ALIVE_SOURCE": "event", "ENG_PARTICIPATION": "kill_credit", "ENG_OVERLAP_RULE": "none",
    "ENG_MERGE_A6": True, "ENG_ISOLATION": True, "ENG_CLEAN_MAX_AGE_MS": 10000,
    "ENG_BOUNDARIES_LOCKED": True, "DD_TABLE_DIR": "config/game_rules/datadragon_v2",
    "REQUIRE_ALIVE_PER_TEAM": 0, "LABEL_TIE_STRATEGY": "exclude",
    "TF2_KILL_CLUSTER_GAP_MS": 14000, "CLUSTER_MAX_DIAMETER": 4300.0,
}


def test_v33_unchanged_vs_snapshot():
    assert PRESETS["v3.3"] == V33_SNAPSHOT
    c = CFG()
    apply_preset(c, "v3.3")
    for k in NEW_FIELD_DEFAULTS:
        assert k not in PRESETS["v3.3"]
        assert getattr(c, k) == NEW_FIELD_DEFAULTS[k], k


def test_new_config_defaults_keep_v33_behaviour():
    c = CFG()
    for k, v in NEW_FIELD_DEFAULTS.items():
        assert getattr(c, k) == v, k
    # untouched globals the preset overrides
    assert c.LABEL_TIE_STRATEGY == "random"
    assert c.REQUIRE_ALIVE_PER_TEAM == 2


def test_v4_exact_applies():
    c = CFG()
    vals = apply_preset(c, "v4-exact")
    assert set(vals) == set(PRESETS["v4-exact"])
    for k, v in V33_SNAPSHOT.items():
        if k not in V4_OVERRIDES:
            assert getattr(c, k) == v, k
    for k, v in V4_OVERRIDES.items():
        assert getattr(c, k) == v, k
    # G / D locked by E1 (G = 14 s, D = 4,300).
    assert c.TF2_KILL_CLUSTER_GAP_MS == 14000 and c.CLUSTER_MAX_DIAMETER == 4300.0
    assert c.ENG_BOUNDARIES_LOCKED is True


def test_v4_exact_does_not_leak_into_defaults():
    apply_preset(CFG(), "v4-exact")
    c = CFG()
    for k, v in NEW_FIELD_DEFAULTS.items():
        assert getattr(c, k) == v, k


def test_exclude_tie_strategy_drops_market_event_tie(monkeypatch):
    """v4-exact sets LABEL_TIE_STRATEGY='exclude'; the existing label code must return None on a tie."""
    import numpy as np
    from core.config import F_GLOBAL, F_NODE, cfg
    from gameplay.labels import compute_label

    def _dummy_interp(cache, q_ms):
        return np.zeros((10, F_NODE), dtype=np.float32), np.zeros((F_GLOBAL,), dtype=np.float32)

    monkeypatch.setattr(cfg, "LABEL_TYPE", "market_event@window", raising=False)
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_TABLE", "", raising=False)
    cache = {"minute_ts": np.array([0, 60_000], dtype=np.int64), "events": [
        {"timestamp": 1_000, "type": "CHAMPION_KILL", "killerId": 1, "victimId": 6, "bounty": 300},
        {"timestamp": 2_000, "type": "CHAMPION_KILL", "killerId": 6, "victimId": 1, "bounty": 300}]}
    tm = {pid: (100 if pid <= 5 else 200) for pid in range(1, 11)}
    monkeypatch.setattr(cfg, "LABEL_TIE_STRATEGY", "exclude", raising=False)
    assert compute_label(cache, tm, 0, horizon_ms=30_000, interp_node_global=_dummy_interp) is None
    monkeypatch.setattr(cfg, "LABEL_TIE_STRATEGY", "random", raising=False)
    assert compute_label(cache, tm, 0, horizon_ms=30_000, interp_node_global=_dummy_interp) in (0, 1)
