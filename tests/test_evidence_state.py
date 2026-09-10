"""Evidence-based state: causality, event replay, and the formulas the reconstruction rests on."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import NODE_FEATURE_NAMES  # noqa: E402
from gameplay.evidence_state import (  # noqa: E402
    DEN, EvidenceStateBuilder, FOUNTAIN, XP_AT_LEVEL, level_growth, respawn_seconds,
)

IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
ITEMS = {1001: {"gold_total": 300, "gold_base": 300, "gold_sell": 210, "from": [], "into": [3006],
                "stats": {"movementSpeed": 25.0}},
         1036: {"gold_total": 350, "gold_base": 350, "gold_sell": 245, "from": [], "into": [3134],
                "stats": {"attackDamage": 10.0}},
         3134: {"gold_total": 1100, "gold_base": 400, "gold_sell": 770, "from": [1036, 1036], "into": [],
                "stats": {"attackDamage": 30.0}},
         2003: {"gold_total": 50, "gold_base": 50, "gold_sell": 20, "from": [], "into": [], "stats": {}}}
CHAMPS = {266: {"name": "Aatrox", "base": {"attackDamage": 60.0, "attackSpeed": 0.651, "healthMax": 650.0},
                "growth": {"attackDamage": 5.0, "attackSpeed_pct": 2.5, "healthMax": 114.0}}}


def _pack(events, n_frames=4):
    ts = np.array([120_000 + 60_000 * k for k in range(n_frames)], dtype=np.int64)
    node = np.zeros((n_frames, 10, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    xy = np.zeros((n_frames, 10, 2))
    for k in range(n_frames):
        for p in range(10):
            node[k, p, IDX["level_norm"]] = 3 / DEN["level"]
            node[k, p, IDX["xp_norm"]] = (700 + 200 * k) / DEN["xp"]
            node[k, p, IDX["totalGold_norm"]] = (1000 + 300 * k) / DEN["tot_g"]
            node[k, p, IDX["curGold_norm"]] = (500 + 300 * k) / DEN["cur_g"]
            node[k, p, IDX["cs_attackDamage"]] = 70.0 / 600.0
            node[k, p, IDX["cs_attackSpeed"]] = 0.7 / 3.0
            node[k, p, IDX["cs_healthMax"]] = 900.0 / 8500.0
            node[k, p, IDX["hp_pct"]] = 0.8
            node[k, p, IDX["alive"]] = 1.0
            xy[k, p] = (4000.0 + 100 * k, 7000.0)
    return {"minute_ts": ts, "node_minute": node, "xy_raw_minute": xy, "events": events,
            "meta": {"patch": "15.15", "team_map": {str(i): (100 if i <= 5 else 200) for i in range(1, 11)},
                     "static_meta": {"champion_by_pid": {"1": 266}}}}


def builder(events, **kw):
    return EvidenceStateBuilder(_pack(events, **kw), tables=(ITEMS, CHAMPS))


def test_never_reads_a_frame_after_tau():
    b = builder([])
    assert b.frame_index(179_999) == 0 and b.frame_index(180_000) == 1
    with pytest.raises(ValueError):
        b.frame_index(100)


def test_level_up_and_ultimate_come_from_events_not_the_frame():
    b = builder([{"type": "LEVEL_UP", "timestamp": 190_000, "participantId": 1, "level": 4},
                 {"type": "LEVEL_UP", "timestamp": 195_000, "participantId": 1, "level": 5},
                 {"type": "SKILL_LEVEL_UP", "timestamp": 196_000, "participantId": 1, "skillSlot": 4, "levelUpType": "NORMAL"}])
    s = b.at(200_000)[1]                                   # frame at 180s says level 3
    assert s["level_norm"] == pytest.approx(5 / 18)
    assert s["ult_level_norm"] == pytest.approx(1 / 3)
    assert s["xp_norm"] * DEN["xp"] >= XP_AT_LEVEL[5] - 1e-6   # a level-up proves at least that much xp
    assert b.at(189_000)[1]["level_norm"] == pytest.approx(3 / 18)


def test_inventory_replay_handles_buy_combine_sell_and_undo():
    ev = [{"type": "ITEM_PURCHASED", "timestamp": 185_000, "participantId": 1, "itemId": 1036},
          {"type": "ITEM_PURCHASED", "timestamp": 186_000, "participantId": 1, "itemId": 1036},
          {"type": "ITEM_DESTROYED", "timestamp": 190_000, "participantId": 1, "itemId": 1036},
          {"type": "ITEM_DESTROYED", "timestamp": 190_000, "participantId": 1, "itemId": 1036},
          {"type": "ITEM_PURCHASED", "timestamp": 190_000, "participantId": 1, "itemId": 3134},
          {"type": "ITEM_PURCHASED", "timestamp": 192_000, "participantId": 1, "itemId": 2003},
          {"type": "ITEM_UNDO", "timestamp": 193_000, "participantId": 1, "beforeId": 2003, "afterId": 0, "goldGain": 50}]
    b = builder(ev)
    inv, gold_delta, stat_delta = b._replay_inventory(1, 200_000, since=180_000)
    assert inv == [3134]
    # paid 350 + 350 + 400 (recipe, components destroyed at the same timestamp) + 50, undo refunded 50
    assert gold_delta == pytest.approx(-(350 + 350 + 400 + 50) + 50)
    assert stat_delta["attackDamage"] == pytest.approx(30.0)   # 10 + 10 - 10 - 10 + 30
    s = b.at(200_000)[1]
    assert s["cs_attackDamage"] * 600 == pytest.approx(70.0 + 30.0)
    assert s["item_count"] == 1.0


def test_level_growth_formula_and_attack_speed_growth():
    assert level_growth(1) == 0.0
    assert level_growth(2) == pytest.approx(0.72)            # (L-1) * (0.7025 + 0.0175 * (L-1))
    assert level_growth(18) == pytest.approx(17.0)           # the growth multiplier reaches exactly 17 at 18
    b = builder([{"type": "LEVEL_UP", "timestamp": 190_000, "participantId": 1, "level": 4}])
    s = b.at(200_000)[1]
    dg = level_growth(4) - level_growth(3)
    assert s["cs_attackDamage"] * 600 == pytest.approx(70.0 + 5.0 * dg)
    assert s["cs_attackSpeed"] * 3 == pytest.approx(0.7 + 0.651 * 0.025 * dg)
    assert s["cs_healthMax"] * 8500 == pytest.approx(900.0 + 114.0 * dg)


def test_death_respawn_and_hp_reset():
    ev = [{"type": "CHAMPION_KILL", "timestamp": 185_000, "victimId": 1, "killerId": 6,
           "position": {"x": 9000, "y": 9000}, "bounty": 300, "shutdownBounty": 0}]
    b = builder(ev)                                          # no LEVEL_UP events: the frame's level 3 is the bound
    timer = respawn_seconds(3, 185_000)
    assert timer == 12                                      # level 3, before minute 25: base only
    during = b.at(185_000 + int(timer * 1000) - 500)[1]
    after = b.at(185_000 + int(timer * 1000) + 500)[1]
    assert during["alive"] == 0.0 and during["hp_pct"] == 0.0 and during["respawn_in_s"] > 0
    assert after["alive"] == 1.0 and after["hp_pct"] == 1.0 and after["mp_pct"] == 1.0
    killer = b.at(190_000)[6]
    assert killer["curGold_norm"] * DEN["cur_g"] >= 500 + 300      # bounty landed at 185s
    assert respawn_seconds(18, 40 * 60_000) > respawn_seconds(18, 10 * 60_000)


def test_position_uses_the_latest_evidence_and_never_the_engagement_kill():
    ev = [{"type": "CHAMPION_KILL", "timestamp": 185_000, "victimId": 2, "killerId": 1,
           "assistingParticipantIds": [3], "position": {"x": 12000, "y": 3000}, "bounty": 300, "shutdownBounty": 0},
          {"type": "ITEM_PURCHASED", "timestamp": 195_000, "participantId": 4, "itemId": 1001},
          {"type": "CHAMPION_KILL", "timestamp": 230_000, "victimId": 5, "killerId": 7,
           "position": {"x": 1000, "y": 1000}, "bounty": 300, "shutdownBounty": 0}]
    b = builder(ev)
    s = b.at(200_000)
    assert (s[1]["x_norm"] * 16000, s[1]["y_norm"] * 16000) == (12000, 3000)   # killer near the kill
    assert s[1]["pos_evidence_kind"] == 2.0 and s[1]["pos_evidence_age_s"] == pytest.approx(15.0)
    assert s[3]["pos_evidence_kind"] == 3.0                                     # assist
    # a shop visit is weak evidence (players leave the fountain at once): the calibrated error curve
    # keeps the 20 s-old frame (3,300 u expected) over a 5 s-old purchase (5,833 u)
    assert s[4]["pos_evidence_kind"] == 0.0
    assert s[4]["x_norm"] * 16000 == pytest.approx(4100.0)
    # victim 2 died at 185s at level 3 (12 s timer): dead at 190s (frame position kept, flagged),
    # respawned at 197s -> a 3 s-old respawn (2,803 u) beats the 20 s-old frame (3,300 u)
    d = b.at(190_000)[2]
    assert d["alive"] == 0.0 and d["pos_evidence_kind"] == 7.0
    assert s[2]["alive"] == 1.0 and s[2]["pos_evidence_kind"] == 6.0
    assert (s[2]["x_norm"] * 16000, s[2]["y_norm"] * 16000) == FOUNTAIN[100]
    # but once the respawn is 15 s old (6,360 u) a 30 s-old frame (3,900 u) wins again
    late = b.at(212_000)[2]
    assert late["pos_evidence_kind"] == 0.0
    assert s[5]["pos_evidence_kind"] == 0.0                                     # kill at 230s is after tau
    assert s[5]["x_norm"] * 16000 == pytest.approx(4100.0)                      # frame at 180s, held
    assert s[5]["pos_evidence_age_s"] == pytest.approx(20.0)


def test_continuous_accumulators_extrapolate_from_previous_frames_only():
    b = builder([])
    s = b.at(210_000)[1]                       # frame at 180s, previous at 120s: 300 gold / 60s = 5 g/s
    assert s["totalGold_norm"] * DEN["tot_g"] == pytest.approx(1300 + 5.0 * 30)
    assert s["xp_norm"] * DEN["xp"] == pytest.approx(900 + (200 / 60) * 30)
    first = b.at(150_000)[1]                   # only one frame available: no rate, plain hold
    assert first["totalGold_norm"] * DEN["tot_g"] == pytest.approx(1000)


def test_baron_buff_is_lost_on_death():
    ev = [{"type": "ELITE_MONSTER_KILL", "timestamp": 185_000, "killerId": 1, "killerTeamId": 100,
           "monsterType": "BARON_NASHOR", "monsterSubType": "", "position": {"x": 5000, "y": 10000}},
          {"type": "CHAMPION_KILL", "timestamp": 200_000, "victimId": 2, "killerId": 6,
           "position": {"x": 6000, "y": 6000}, "bounty": 300, "shutdownBounty": 0}]
    b = builder(ev)
    s = b.at(210_000)
    assert s[1]["has_baron"] == 1.0 and 0 < s[1]["baron_remain_norm"] < 1
    assert s[2]["has_baron"] == 0.0                                            # died after taking it
    assert s[6]["has_baron"] == 0.0                                            # other team
    assert b.at(185_000 + 180_000 + 1)[1]["has_baron"] == 0.0                  # expired
