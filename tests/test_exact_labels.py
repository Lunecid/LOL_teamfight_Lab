"""Tests for gameplay.labels_exact (v4-exact stage 1, task B3).

Synthetic engagements check every tier and boundary; 20 real patch-15.14 engagements from
outputs/reest_exact_v4_20260925/records/baselines/finals_r2_kill_full.tsv check the functions on
Match-V5 events (skipped when the D: cache or the TSV is missing).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import gameplay.labels as LB
from core.config import cfg
from gameplay import labels_exact as LX
from gameplay.event_survival import death_intervals

TM = {i: (100 if i <= 5 else 200) for i in range(1, 11)}
PATCH = "15.14"
D = 4300.0


def kill(ts, killer, victim, assists=(), bounty=300, shutdown=0, xy=(5000, 5000)):
    return {"type": "CHAMPION_KILL", "timestamp": ts, "killerId": killer, "victimId": victim,
            "assistingParticipantIds": list(assists), "bounty": bounty, "shutdownBounty": shutdown,
            "position": {"x": xy[0], "y": xy[1]}}


def plate(ts, lost_team, xy=(5000, 5000)):
    return {"type": "TURRET_PLATE_DESTROYED", "timestamp": ts, "teamId": lost_team, "killerId": 0,
            "laneType": "MID_LANE", "position": {"x": xy[0], "y": xy[1]}}


def tower(ts, lost_team, tier="OUTER_TURRET", xy=(5000, 5000)):
    return {"type": "BUILDING_KILL", "timestamp": ts, "teamId": lost_team, "killerId": 1,
            "buildingType": "TOWER_BUILDING", "towerType": tier, "laneType": "MID_LANE",
            "position": {"x": xy[0], "y": xy[1]}}


def monster(ts, team, mtype="DRAGON", sub="FIRE_DRAGON", xy=(9800, 4400), killer=None):
    e = {"type": "ELITE_MONSTER_KILL", "timestamp": ts, "killerTeamId": team, "monsterType": mtype,
         "killerId": killer if killer is not None else (1 if team == 100 else 6 if team == 200 else 0),
         "position": {"x": xy[0], "y": xy[1]}}
    if mtype == "DRAGON":
        e["monsterSubType"] = sub
    return e


def eng(kills, tau, **kw):
    d = {"tau": tau, "last_kill_ts": max(k["timestamp"] for k in kills), "kills": kills}
    d.update(kw)
    return d


def outcome(e, events, prices, e_h, **kw):
    kw.setdefault("radius", D)
    kw.setdefault("patch", PATCH)
    kw.setdefault("team_map", TM)
    return LX.exchange_outcome(e, events, prices, e_h, **kw)


@pytest.fixture(autouse=True)
def no_tie_coin(monkeypatch):
    """Any call of labels._seeded_tie_coin fails the test; cfg asks for random ties."""
    def boom(*a, **k):
        raise AssertionError("labels._seeded_tie_coin must never be called")
    monkeypatch.setattr(LB, "_seeded_tie_coin", boom)
    monkeypatch.setattr(cfg, "LABEL_TIE_STRATEGY", "random", raising=False)
    monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", "random", raising=False)
    yield


PRICES = {"kills": 20.0, "assists": 45.0, "plates": 120.0, "tower_outer_turret": 540.0,
          "inhibitor": 85.0, "dragon": 0.0, "baron_nashor": 1555.0, "first_tower": 100.0}


# ------------------------------------------------------------------ gold tier
def test_gold_tier_blue_and_kill_price_rule():
    ks = [kill(200_000, 1, 6, assists=(2,), bounty=300)]
    o = outcome(eng(ks, 185_000), ks, PRICES, 230_000)
    assert o["gold_diff"] == pytest.approx(300 + 20 + 45)
    assert (o["label"], o["decided_by"]) == (1, "gold")


def test_gold_overrides_kill_count_and_kill_diff_label_differs():
    ks = [kill(200_000, 1, 6, bounty=100), kill(201_000, 2, 7, bounty=100),
          kill(202_000, 8, 3, bounty=300, shutdown=400)]
    e = eng(ks, 185_000)
    o = outcome(e, ks, {}, 230_000)
    assert o["gold_diff"] == pytest.approx(200 - 700)
    assert (o["label"], o["decided_by"]) == (0, "gold")
    assert LX.kill_diff_label(e, ks, TM) == 1


def test_deadzone_is_strict_and_falls_to_kills():
    ks = [kill(200_000, 1, 6, bounty=300)]
    o = outcome(eng(ks, 185_000), ks, {}, 230_000)
    assert o["gold_diff"] == 300.0 and o["decided_by"] == "kills" and o["label"] == 1
    ks2 = [kill(200_000, 1, 6, bounty=301)]
    assert outcome(eng(ks2, 185_000), ks2, {}, 230_000)["decided_by"] == "gold"
    o3 = outcome(eng(ks, 185_000), ks, {}, 230_000, deadzone=200.0)
    assert o3["decided_by"] == "gold"


def test_negative_and_nan_bounty_are_clipped():
    k = kill(200_000, 1, 6, bounty=-50)
    k["shutdownBounty"] = float("nan")
    assert LX.priced_kill_gold(k, {}) == 0.0


# ------------------------------------------------------------------ non-kill attribution
def test_non_kill_window_and_radius_boundaries():
    ks = [kill(200_000, 1, 6, bounty=150, xy=(5000, 5000))]
    e = eng(ks, 185_000)
    e_h = 230_000
    at_d = plate(210_000, 200, xy=(5000 + D, 5000))            # distance == D: counted
    beyond = plate(210_000, 200, xy=(5000 + D + 1, 5000))      # counted nowhere
    at_tau = plate(185_000, 200)                                # tau inclusive
    before = plate(184_999, 200)
    at_end = plate(e_h, 200)                                    # e_h inclusive
    after = plate(e_h + 1, 200)
    evs = ks + [at_d, beyond, at_tau, before, at_end, after]
    o = outcome(e, evs, PRICES, e_h)
    assert o["n_priced_other"] == 3
    assert o["gold_other"] == pytest.approx(3 * 120.0)
    assert o["structure_diff"] == 3


def test_non_kill_signs_and_ignored_types():
    ks = [kill(200_000, 1, 6, bounty=0)]
    e = eng(ks, 185_000)
    soul = {"type": "DRAGON_SOUL_GIVEN", "timestamp": 205_000, "teamId": 100, "name": "Fire"}
    despawn = monster(206_000, 300, "HORDE")
    despawn["position"] = {"x": 5000, "y": 5000}
    ward = {"type": "WARD_KILL", "timestamp": 207_000, "killerId": 1, "wardType": "YELLOW_TRINKET"}
    red_plate = plate(208_000, 100)            # blue lost a plate -> red +120
    evs = ks + [soul, despawn, ward, red_plate]
    o = outcome(e, evs, {"kills": 0.0, "plates": 120.0, "horde": 40.0, "ward_kills": 25.0}, 230_000)
    assert o["gold_other"] == pytest.approx(-120.0)
    assert o["n_unplaced_skipped"] == 1        # the ward kill has no position
    assert o["structure_diff"] == -1


def test_foreign_kills_are_not_own_kills():
    ks = [kill(200_000, 1, 6, bounty=300)]
    foreign = kill(201_000, 7, 2, bounty=1000, xy=(5100, 5100))   # same place, not in the kill set
    o = outcome(eng(ks, 185_000), ks + [foreign], {}, 230_000)
    assert o["kill_diff"] == 1 and o["gold_kills"] == 300.0 and o["label"] == 1


def test_first_tower_bonus_only_for_the_match_first_tower():
    ks = [kill(200_000, 1, 6, bounty=0)]
    e = eng(ks, 185_000)
    t_in = tower(210_000, 200)
    o = outcome(e, ks + [t_in], PRICES, 230_000)
    assert o["gold_other"] == pytest.approx(540 + 100)
    earlier = tower(100_000, 100, xy=(1000, 1000))                 # earlier tower elsewhere
    o2 = outcome(e, ks + [earlier, t_in], PRICES, 230_000)
    assert o2["gold_other"] == pytest.approx(540)


# ------------------------------------------------------------------ lexicographic tiers
def test_alive_tier_uses_kill_event_survival_at_L():
    # blue 1 dies at 100 s (level 1 -> 10 s timer, back at 110 s); red 6 dies at 115 s = L
    ks = [kill(100_000, 6, 1, bounty=300), kill(115_000, 1, 6, bounty=300)]
    e = eng(ks, 85_000)
    iv = death_intervals(ks, PATCH)
    assert iv[1][0][1] <= 115_000
    o = outcome(e, ks, {}, 140_000)
    assert o["kill_diff"] == 0 and o["alive_diff"] == 1
    assert (o["label"], o["decided_by"]) == (1, "alive")


def test_alive_tier_ignores_events_after_L():
    ks = [kill(100_000, 6, 1, bounty=300), kill(115_000, 1, 6, bounty=300)]
    late = kill(115_001, 7, 2, bounty=0, xy=(14000, 14000))       # foreign death just after L
    o = outcome(eng(ks, 85_000), ks + [late], {}, 140_000)
    assert o["alive_diff"] == 1 and o["label"] == 1


def test_structures_tier_and_full_tie_excluded():
    ks = [kill(200_000, 1, 6, bounty=300), kill(200_500, 7, 2, bounty=300)]
    e = eng(ks, 185_000)
    o = outcome(e, ks + [plate(210_000, 200)], {}, 230_000)
    assert o["alive_diff"] == 0
    assert (o["label"], o["decided_by"]) == (1, "structures")
    o_tie = outcome(e, ks, {}, 230_000)
    assert o_tie["label"] is None and o_tie["decided_by"] == "excluded"
    assert LX.exchange_label(e, ks, {}, 230_000, radius=D, patch=PATCH, team_map=TM) is None
    assert LX.kill_diff_label(e, ks, TM) is None


def test_execution_counts_as_victim_team_loss_with_zero_gold():
    """Author decision 2026-09-25: killer 0 = loss for the victim's team; gold contribution 0."""
    ks = [kill(200_000, 0, 6, assists=(2,), bounty=300)]     # red 6 executed
    e = eng(ks, 185_000)
    o = outcome(e, ks, {"kills": 20.0, "assists": 45.0}, 230_000)
    assert o["gold_kills"] == 0.0 and o["gold_diff"] == 0.0 and o["n_executions"] == 1
    assert o["kill_diff"] == 1 and (o["label"], o["decided_by"]) == (1, "kills")
    assert o["alive_diff"] is None                              # the kill tier decided
    assert LX.kill_diff_label(e, ks, TM) == 1
    kb = [kill(200_000, 0, 3, bounty=300)]                      # blue 3 executed
    eb = eng(kb, 185_000)
    ob = outcome(eb, kb, {"kills": 20.0}, 230_000)
    assert (ob["gold_kills"], ob["kill_diff"], ob["label"], ob["decided_by"]) == (0.0, -1, 0, "kills")
    assert LX.kill_diff_label(eb, kb, TM) == 0


def test_execution_offsets_a_champion_kill():
    # blue 1 kills red 6 (+1), blue 2 is executed by a turret (-1): kill tier ties -> alive tier
    ks = [kill(200_000, 1, 6, bounty=250), kill(201_000, 0, 2, bounty=300)]
    e = eng(ks, 185_000)
    o = outcome(e, ks, {}, 230_000)
    assert o["gold_kills"] == 250.0 and o["kill_diff"] == 0 and o["n_executions"] == 1
    assert (o["alive_diff"], o["label"], o["decided_by"]) == (0, None, "excluded")
    assert LX.kill_diff_label(e, ks, TM) is None
    # the execution does not shift the gold tier: a large bounty on the executed player is not paid
    ks2 = [kill(200_000, 1, 6, bounty=400), kill(201_000, 0, 2, bounty=1000)]
    o2 = outcome(eng(ks2, 185_000), ks2, {}, 230_000)
    assert (o2["gold_diff"], o2["label"], o2["decided_by"]) == (400.0, 1, "gold")


def test_execution_sign_helpers():
    assert LX.is_execution({"killerId": 0}) and LX.is_execution({}) and not LX.is_execution({"killerId": 4})
    assert LX.kill_team_sign({"killerId": 0, "victimId": 7}, TM) == 1
    assert LX.kill_team_sign({"killerId": 0, "victimId": 4}, TM) == -1
    assert LX.kill_team_sign({"killerId": 0, "victimId": 0}, TM) == 0      # no victim team: nobody
    assert LX.kill_team_sign({"killerId": 8, "victimId": 1}, TM) == -1
    swapped = {p: (200 if p <= 5 else 100) for p in range(1, 11)}
    assert LX.kill_team_sign({"killerId": 0, "victimId": 7}, swapped) == -1
    assert LX.kill_difference([{"killerId": 0, "victimId": 7}, {"killerId": 0, "victimId": 8},
                               {"killerId": 9, "victimId": 2}], TM) == 1


def test_tie_policy_in_cfg_is_ignored(monkeypatch):
    ks = [kill(200_000, 1, 6, bounty=300), kill(200_500, 7, 2, bounty=300)]
    for pol in ("random", "blue", "red", "drop"):
        monkeypatch.setattr(cfg, "LABEL_TIE_STRATEGY", pol, raising=False)
        monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", pol, raising=False)
        assert outcome(eng(ks, 185_000), ks, {}, 230_000)["label"] is None


def test_source_never_references_tie_coin_or_frames():
    src = Path(LX.__file__).read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]  # skip the module docstring
    for bad in ("_seeded_tie_coin", "import random", "np.random", "xy_raw_minute", "node_minute",
                "build_5s_position_grid", "interp_node_global", "LABEL_TIE"):
        assert bad not in code, bad


# ------------------------------------------------------------------ inputs and errors
def test_errors():
    ks = [kill(200_000, 1, 6)]
    e = eng(ks, 185_000)
    with pytest.raises(ValueError):
        outcome(e, ks, {}, 199_999)                              # e_h < L
    with pytest.raises(ValueError):
        LX.exchange_outcome(e, ks, {}, 230_000, radius=D, team_map=TM)          # no patch
    with pytest.raises(ValueError):
        LX.exchange_outcome(e, ks, {}, 230_000, patch=PATCH, team_map=TM)       # no radius
    with pytest.raises(KeyError):
        outcome({"tau": 1, "last_kill_ts": 2}, ks, {}, 230_000)  # no kill set
    with pytest.raises(ValueError):
        outcome(dict(e, last_kill_ts=201_000), ks, {}, 230_000)
    with pytest.raises(ValueError):
        outcome(eng([{"killer_id": 1, "victim_id": 6, "timestamp": 1}], 0), ks, {}, 230_000)
    with pytest.raises(ValueError):
        outcome(eng(ks, 200_001), ks, {}, 230_000)              # kill before tau


def test_kill_idx_equals_kills_and_extract_order():
    from gameplay.fights import _extract_kill_events
    evs = [kill(300_000, 2, 7), plate(250_000, 200), kill(200_000, 1, 6), kill(200_000, 8, 3),
           {"type": "CHAMPION_KILL", "killerId": 1, "victimId": 9}]  # no timestamp: dropped by both
    raw = LX.raw_kill_events(evs)
    ext = _extract_kill_events(evs)
    assert [(k["timestamp"], k["killerId"], k["victimId"]) for k in raw] == \
           [(k["timestamp"], k["killer_id"], k["victim_id"]) for k in ext]
    by_idx = {"tau": 185_000, "last_kill_ts": 200_000, "kill_idx": [1, 0]}
    by_evs = eng([evs[2], evs[3]], 185_000)
    o1, o2 = outcome(by_idx, evs, PRICES, 230_000), outcome(by_evs, evs, PRICES, 230_000)
    assert o1 == o2


def test_anchor_priority():
    ks = [kill(200_000, 1, 6, xy=(100, 100))]
    assert LX.engagement_anchor({}, ks) == (100.0, 100.0)
    assert LX.engagement_anchor({"anchor_xy": (7, 8)}, ks) == (7.0, 8.0)
    nopos = dict(ks[0])
    nopos.pop("position")
    assert LX.engagement_anchor({"centroid_x": 3, "centroid_y": 4}, [nopos]) == (3.0, 4.0)


def test_events_after_e_h_never_matter():
    ks = [kill(200_000, 1, 6, bounty=300), kill(200_500, 7, 2, bounty=300)]
    e = eng(ks, 185_000)
    base = outcome(e, ks, PRICES, 230_000)
    noise = [tower(230_001, 100), plate(240_000, 100), kill(230_001, 6, 3, xy=(5000, 5000)),
             {"type": "GAME_END", "timestamp": 250_000, "winningTeam": 200}]
    assert outcome(e, ks + noise, PRICES, 230_000) == base


def test_load_price_table():
    pooled = LX.load_price_table()
    p1514 = LX.load_price_table(patch="15.14")
    assert pooled["assists"] == 45.0 and p1514["tower_outer_turret"] == 535.0


# ------------------------------------------------------------------ next objective
def test_next_objective_rules():
    e_h = 300_000
    ge = {"type": "GAME_END", "timestamp": 2_000_000}
    f = LX.next_objective_label
    assert f([monster(e_h, 100), monster(e_h + 5, 200), ge], e_h) == 0             # (e_h, ...] open left
    assert f([monster(e_h + 180_000, 100), ge], e_h) == 1                           # right closed
    assert f([monster(e_h + 180_001, 100), ge], e_h) is None
    assert LX.next_objective_outcome([monster(e_h + 180_001, 100), ge], e_h) == "none"
    assert LX.next_objective_outcome([monster(e_h + 10, 100), monster(e_h + 10, 200, "HORDE"), ge], e_h) == "tie_ambiguous"
    both_blue = [monster(e_h + 10, 100, "HORDE"), monster(e_h + 10, 100, "HORDE"), ge]
    assert f(both_blue, e_h) == 1
    desp = [monster(e_h + 10, 300, "HORDE", killer=0), monster(e_h + 20, 200, "BARON_NASHOR"), ge]
    assert f(desp, e_h) == 0
    assert LX.next_objective_outcome(desp, e_h, skip_despawn=False) == "tie_ambiguous"
    assert LX.next_objective_outcome([{"type": "GAME_END", "timestamp": e_h}], e_h) == "game_ended_before_objective"
    assert LX.next_objective_outcome([{"type": "GAME_END", "timestamp": e_h + 60_000}], e_h) == \
        "game_ended_in_window_no_objective"
    assert LX.next_objective_outcome([], e_h) == "observation_censored"
    # killerTeamId missing -> team from killerId
    m = monster(e_h + 10, 0, "ATAKHAN", killer=7)
    m.pop("killerTeamId")
    assert f([m, ge], e_h) == 0


# ------------------------------------------------------------------ label end (DR rule, unchanged)
def test_label_endpoint_uses_dr_rule():
    try:
        LX.load_dr_rules()
    except FileNotFoundError:
        pytest.skip("DR/engagement_labels_v3_rules.py not found")
    ks = [kill(200_000, 1, 6), kill(250_000, 2, 7), {"type": "GAME_END", "timestamp": 1_000_000}]
    assert LX.label_endpoint(200_000, ks, h_s=90) == (249_999, ("next_kill",))
    assert LX.label_endpoint(250_000, ks, h_s=90) == (340_000, ("horizon",))
    assert LX.label_endpoint(250_000, ks, h_s=90, next_start=300_000) == (299_999, ("next_engagement_start",))
    assert LX.label_endpoint(250_000, ks, h_s=90, next_start=2_000_000, game_end=260_000) == (259_999, ("game_end",))


# ------------------------------------------------------------------ 20 real 15.14 engagements
ROOT = Path("C:/Users/todtj/문서/LOL_Teamfight")
TSV = ROOT / "outputs/reest_exact_v4_20260925/records/baselines/finals_r2_kill_full.tsv"
CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
R2_G, R2_D = 13700, 4264.0


def _real_engagements(n=20, seed=20260925):
    """n single-segment R2 engagements (15.14) with their own kill set rebuilt by the R2 clustering
    (G = 13.7 s, D = 4,264), matched on first/last kill and the TSV participant set."""
    if not TSV.exists() or not CACHE.exists():
        return None
    from gameplay import fight_clustering as FC
    from gameplay.fights import _extract_kill_events
    rows = list(csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"))
    by_match = {}
    for r in rows:
        by_match.setdefault(r["match_id"], []).append(r)
    rng = np.random.default_rng(seed)
    mids = sorted(by_match)
    out = []
    for mi in rng.permutation(len(mids)):
        mid = mids[int(mi)]
        mp = CACHE / f"{mid}.meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if str(meta.get("patch")) != "15.14":
            continue
        evs = json.loads((CACHE / f"{mid}.events.json").read_text(encoding="utf-8"))
        tm = {int(k): int(v) for k, v in meta["team_map"].items()}
        kills = _extract_kill_events(evs)
        kid = {id(k): i for i, k in enumerate(kills)}
        clusters = []
        for cl in FC.cluster_kills_temporal(kills, R2_G):
            clusters.extend(FC.split_kill_cluster_spatial(cl, max_diameter=R2_D))
        mrows = sorted(by_match[mid], key=lambda r: int(r["engage_ts"]))
        for j, r in enumerate(mrows):
            if int(r["n_segments"]) != 1:
                continue
            fk, lk = int(r["first_kill_ts"]), int(r["last_kill_ts"])
            parts = sorted(int(x) for x in r["parts"].split(","))
            hit = [c for c in clusters if c["first_kill_ts"] == fk and c["last_kill_ts"] == lk
                   and sorted(c["participants"]) == parts]
            if len(hit) != 1:
                continue
            nxt = int(mrows[j + 1]["engage_ts"]) if j + 1 < len(mrows) else None
            if nxt is not None and nxt <= lk:  # overlapping R2 rows (not isolated): e_h < L is invalid
                continue
            e = {"match_id": mid, "tau": int(r["engage_ts"]), "last_kill_ts": lk, "patch": "15.14",
                 "team_map": tm, "kill_idx": sorted(kid[id(k)] for k in hit[0]["kills"]), "next_start": nxt}
            out.append((e, evs))
            break  # one engagement per match
        if len(out) >= n:
            break
    return out


@pytest.fixture(scope="module")
def real():
    got = _real_engagements()
    if not got or len(got) < 20:
        pytest.skip("15.14 cache or finals_r2_kill_full.tsv not available")
    return got


def _independent_gold(e, evs, prices, e_h):
    """Straight re-implementation of the rule for the cross-check."""
    ks = sorted([x for x in evs if x.get("type") == "CHAMPION_KILL"], key=lambda x: x["timestamp"])
    own = [ks[i] for i in e["kill_idx"]]
    tm = e["team_map"]
    ax, ay = own[0]["position"]["x"], own[0]["position"]["y"]
    g = 0.0
    for k in own:
        t = tm.get(k.get("killerId", 0), 0)
        s = 1 if t == 100 else -1 if t == 200 else 0
        g += s * (max(0, k.get("bounty", 0)) + max(0, k.get("shutdownBounty", 0)) + prices["kills"]
                  + prices["assists"] * len(k.get("assistingParticipantIds", [])))
    ft = min([x["timestamp"] for x in evs if x.get("type") == "BUILDING_KILL"
              and x.get("buildingType") != "INHIBITOR_BUILDING"] or [None], key=lambda v: (v is None, v))
    for x in evs:
        if x.get("type") not in ("TURRET_PLATE_DESTROYED", "BUILDING_KILL", "ELITE_MONSTER_KILL"):
            continue
        if not (e["tau"] <= x["timestamp"] <= e_h) or "position" not in x:
            continue
        if (x["position"]["x"] - ax) ** 2 + (x["position"]["y"] - ay) ** 2 > D * D:
            continue
        g += LB._label_event_team_sign(x, tm) * LB._priced_event_gold(x, prices, ft)
    return g


def test_real_engagements(real):
    prices = LX.load_price_table(patch="15.14")
    seen = {}
    for e, evs in real:
        e_h, _ = LX.label_endpoint(e["last_kill_ts"], evs, h_s=90, next_start=e["next_start"])
        assert e_h >= e["last_kill_ts"]
        o = LX.exchange_outcome(e, evs, prices, e_h, radius=D)
        assert o["label"] in (0, 1, None) and o["decided_by"] in LX.DECIDED_BY
        assert o["gold_diff"] == pytest.approx(_independent_gold(e, evs, prices, e_h))
        own = LX.own_kill_events(e, evs)
        assert [k["timestamp"] for k in own] == sorted(k["timestamp"] for k in own)
        assert own[-1]["timestamp"] == e["last_kill_ts"] and own[0]["timestamp"] >= e["tau"]
        # outcome functions read nothing after e_h
        cut = [x for x in evs if int(x.get("timestamp", -1)) <= e_h]
        assert LX.exchange_outcome(e, cut, prices, e_h, radius=D) == o
        # kill label = sign of own kills
        def _sgn(k, tm=e["team_map"]):
            if k.get("killerId", 0) == 0:  # execution: loss for the victim's team
                v = tm.get(k.get("victimId", 0), 0)
                return -1 if v == 100 else 1 if v == 200 else 0
            t = tm.get(k["killerId"], 0)
            return 1 if t == 100 else -1 if t == 200 else 0
        kd = sum(_sgn(k) for k in own)
        assert LX.kill_diff_label(e, evs) == (None if kd == 0 else int(kd > 0))
        assert o["kill_diff"] == kd
        # next objective: only (e_h, e_h + 180 s] matters
        y = LX.next_objective_label(evs, e_h, team_map=e["team_map"])
        cut2 = [x for x in evs if int(x.get("timestamp", -1)) <= e_h + LX.NEXT_OBJECTIVE_WINDOW_MS
                or x.get("type") == "GAME_END"]
        assert LX.next_objective_label(cut2, e_h, team_map=e["team_map"]) == y
        seen[o["decided_by"]] = seen.get(o["decided_by"], 0) + 1
    assert sum(seen.values()) == 20


def test_real_matches_legacy_market_event_on_gold_and_kill_tiers(real, monkeypatch):
    """Where the legacy engagement attribution (events in [tau, e_h] within D of the anchor) holds
    exactly the own kills, the legacy market_event label agrees on the gold and kill tiers."""
    prices = LX.load_price_table(patch="15.14")
    monkeypatch.setattr(LB, "_event_price_table", lambda: dict(prices))
    monkeypatch.setattr(cfg, "LABEL_ATTRIBUTION_RADIUS_U", D, raising=False)
    monkeypatch.setattr(cfg, "LABEL_GOLD_DEADZONE", 300.0, raising=False)

    def no_frames(*a, **k):
        raise LookupError("alive tier reached")
    n_cmp = 0
    for e, evs in real:
        e_h, _ = LX.label_endpoint(e["last_kill_ts"], evs, h_s=90, next_start=e["next_start"])
        o = LX.exchange_outcome(e, evs, prices, e_h, radius=D)
        own = LX.own_kill_events(e, evs)
        win = [x for x in evs if e["tau"] <= int(x.get("timestamp", -1)) <= e_h]
        att = LB.attribute_events(win, own[0]["position"] and (own[0]["position"]["x"], own[0]["position"]["y"]), "engagement")
        att_k = [x for x in att if x.get("type") == "CHAMPION_KILL"]
        if sorted(map(id, att_k)) != sorted(map(id, own)) or o["decided_by"] not in ("gold", "kills"):
            continue
        if o["decided_by"] == "kills" and o["n_executions"]:
            continue  # the legacy kill tier counts an execution for nobody; v4 counts it (author decision)
        cache = {"events": evs, "minute_ts": np.array([0, 10 ** 9])}
        y = LB._compute_label_market_event(att, e["team_map"], cache, e["tau"], e_h, interp_node_global=no_frames,
                                           first_kill_ts=own[0]["timestamp"], last_kill_ts=e["last_kill_ts"])
        assert y == o["label"]
        n_cmp += 1
    assert n_cmp >= 5
