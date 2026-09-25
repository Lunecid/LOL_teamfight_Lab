"""gameplay.objective_timers: exact objective timers, buffs, structures and dragon soul at t."""
import json
import math
import random
from pathlib import Path

import pytest

from gameplay.objective_timers import (ObjectiveTimeline, RULES_PATH, feature_names, load_objective_rules,
                                       objective_state_at, rule, uncertain_rules)

CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
PATCH_INDEX = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json")


# ---------------------------------------------------------------------------- event builders
def drake(ts, team, sub="FIRE_DRAGON", killer=None):
    return {"type": "ELITE_MONSTER_KILL", "timestamp": ts, "monsterType": "DRAGON", "monsterSubType": sub,
            "killerTeamId": team, "killerId": killer or (1 if team == 100 else 6)}


def monster(ts, team, mt):
    return {"type": "ELITE_MONSTER_KILL", "timestamp": ts, "monsterType": mt, "killerTeamId": team,
            "killerId": 1 if team == 100 else (6 if team == 200 else 0)}


def death(ts, victim, killer):
    return {"type": "CHAMPION_KILL", "timestamp": ts, "victimId": victim, "killerId": killer}


def building(ts, owner, lane, btype="TOWER_BUILDING", tower=None, pos=(0, 0)):
    e = {"type": "BUILDING_KILL", "timestamp": ts, "teamId": owner, "laneType": lane, "buildingType": btype,
         "killerId": 1 if owner == 200 else 6, "position": {"x": pos[0], "y": pos[1]}}
    if tower:
        e["towerType"] = tower
    return e


def soul_given(ts, team, name):
    return {"type": "DRAGON_SOUL_GIVEN", "timestamp": ts, "teamId": team, "name": name}


def same_state(a, b):
    return a.keys() == b.keys() and all((math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in a)


# ---------------------------------------------------------------------------- rule table
def test_rule_table_well_formed():
    table = load_objective_rules()
    legend = set(table["status_legend"])
    for name, entries in table["rules"].items():
        for ent in entries:
            assert ent["status"] in legend, name
            assert len(ent["patch_range"]) == 2, name
            assert ent["sources"] and set(ent["sources"]) <= set(table["sources"]), name


@pytest.mark.parametrize("patch", ["15.14", "15.15", "15.16", "15.18", "15.22", "16.13", "16.14", "16.15"])
def test_every_rule_covers_study_patches(patch):
    for name in load_objective_rules()["rules"]:
        rule(name, patch)          # raises KeyError if a study patch is not covered


def test_rule_refuses_uncovered_patch():
    with pytest.raises(KeyError):
        rule("herald_spawn_ms", "15.5")          # before 25.09 herald spawned at 16:00: no entry
    with pytest.raises(KeyError):
        rule("dragon_respawn_ms", "17.1")


def test_patch_specific_rules():
    assert rule("baron_first_spawn_ms", "15.14").value == 1_500_000
    assert rule("baron_first_spawn_ms", "16.15").value == 1_200_000
    assert rule("atakhan_spawn_ms", "16.15").value is None
    assert rule("turret_plates_fall_ms", "15.14").value == 840_000
    assert rule("turret_plates_fall_ms", "16.15").value is None
    assert uncertain_rules("15.14") == []
    assert uncertain_rules("16.15") == ["herald_despawn_ms"]


# ---------------------------------------------------------------------------- spawn timers
def test_dragon_first_spawn_and_respawn():
    ev = [drake(330_000, 100)]
    tl = ObjectiveTimeline(ev, "15.14")
    s = tl.state(0)
    assert s["dragon_next_spawn_s"] == 300.0 and s["dragon_up"] == 0.0
    assert tl.state(300_000)["dragon_up"] == 1.0
    assert tl.state(329_999)["dragon_up"] == 1.0                 # kill not yet known
    s = tl.state(330_000)                                        # kill at t is known at t
    assert s["dragon_up"] == 0.0 and s["dragon_next_spawn_s"] == pytest.approx(300.0)
    assert tl.state(629_999)["dragon_next_spawn_s"] == pytest.approx(0.001)
    assert tl.state(630_000)["dragon_up"] == 1.0


def test_soul_and_elder_timers():
    ev = [drake(300_000, 100, "FIRE_DRAGON"), drake(600_000, 200, "WATER_DRAGON"),
          soul_given(600_500, 0, "Mountain"),                    # Rift element, NOT a soul
          drake(900_000, 100, "EARTH_DRAGON"), drake(1_200_000, 100, "EARTH_DRAGON"),
          drake(1_500_000, 100, "EARTH_DRAGON"), soul_given(1_500_500, 100, "Mountain"),
          drake(1_900_000, 200, "ELDER_DRAGON")]
    tl = ObjectiveTimeline(ev, "15.14")
    s = tl.state(600_499)
    assert s["rift_element_known"] == 0.0
    s = tl.state(700_000)
    assert s["rift_element_earth"] == 1.0 and s["blue_has_soul"] == 0.0 and s["red_has_soul"] == 0.0
    s = tl.state(1_499_999)
    assert s["blue_dragons"] == 3 and s["blue_has_soul"] == 0.0 and s["dragon_next_is_elder"] == 0.0
    s = tl.state(1_500_000)                                      # 4th drake: soul, next is Elder in 6:00
    assert s["blue_has_soul"] == 1.0 and s["dragon_next_is_elder"] == 1.0
    assert s["dragon_next_spawn_s"] == pytest.approx(360.0)
    assert s["blue_dragon_earth"] == 3 and s["blue_dragon_fire"] == 1 and s["red_dragon_water"] == 1
    assert tl.state(1_860_000)["dragon_up"] == 1.0
    s = tl.state(1_900_000)
    assert s["red_elders"] == 1 and s["dragon_next_spawn_s"] == pytest.approx(360.0)
    assert tl.soul_holder(1_900_000) == 100


def test_team_zero_soul_event_never_gives_soul():
    ev = [drake(300_000, 100), drake(600_000, 100), soul_given(600_500, 0, "Ocean")]
    s = ObjectiveTimeline(ev, "15.14").state(2_000_000)
    assert s["blue_has_soul"] == 0.0 and s["red_has_soul"] == 0.0 and s["rift_element_water"] == 1.0


def test_baron_herald_atakhan_voidgrub_15_14():
    ev = [monster(500_000, 100, "HORDE"), monster(505_000, 100, "HORDE"), monster(510_000, 200, "HORDE"),
          monster(950_000, 200, "RIFTHERALD"), monster(1_250_000, 100, "ATAKHAN"),
          monster(1_600_000, 200, "BARON_NASHOR")]
    tl = ObjectiveTimeline(ev, "15.14")
    s = tl.state(479_999)
    assert s["voidgrub_up"] == 0.0 and s["voidgrub_next_spawn_s"] == pytest.approx(0.001)
    s = tl.state(505_000)
    assert s["voidgrub_up"] == 1.0 and s["voidgrub_left"] == 1.0 and s["blue_voidgrubs"] == 2
    assert math.isnan(tl.state(510_000)["voidgrub_next_spawn_s"])
    assert tl.state(900_000)["herald_up"] == 1.0
    assert math.isnan(tl.state(950_000)["herald_next_spawn_s"])
    s = tl.state(1_200_000)
    assert s["atakhan_up"] == 1.0 and s["baron_next_spawn_s"] == pytest.approx(300.0)
    assert math.isnan(tl.state(1_250_000)["atakhan_next_spawn_s"])
    assert tl.state(1_500_000)["baron_up"] == 1.0
    s = tl.state(1_600_000)
    assert s["baron_up"] == 0.0 and s["baron_next_spawn_s"] == pytest.approx(360.0) and s["red_barons"] == 1


def test_despawn_windows_15_14():
    tl = ObjectiveTimeline([], "15.14")
    assert tl.state(884_999)["voidgrub_up"] == 1.0
    s = tl.state(885_000)
    assert s["voidgrub_up"] == 0.0 and math.isnan(s["voidgrub_next_spawn_s"]) and s["voidgrub_in_despawn_grace"] == 1.0
    assert tl.state(1_484_999)["herald_up"] == 1.0
    s = tl.state(1_485_000)
    assert s["herald_up"] == 0.0 and s["herald_in_despawn_grace"] == 1.0
    assert tl.state(1_495_000)["herald_in_despawn_grace"] == 0.0


def test_16_15_rules_and_flags():
    tl = ObjectiveTimeline([], "16.15")
    s = tl.state(0)
    assert s["baron_next_spawn_s"] == 1200.0
    assert math.isnan(s["atakhan_next_spawn_s"]) and s["atakhan_in_patch"] == 0.0
    assert s["rule_uncertain_herald"] == 1.0 and s["rule_uncertain_dragon"] == 0.0
    assert tl.state(1_185_000)["herald_up"] == 0.0               # uncertain 19:45 despawn applied
    assert ObjectiveTimeline([], "15.14").state(0)["rule_uncertain_herald"] == 0.0


# ---------------------------------------------------------------------------- buffs
def test_baron_buff_alive_only_and_lost_on_death():
    ev = [death(1_595_000, 2, 7),                                # pid 2 dead at the kill (timer >= 10 s)
          monster(1_600_000, 100, "BARON_NASHOR"),
          death(1_650_000, 1, 6)]                                # holder pid 1 dies 50 s later
    tl = ObjectiveTimeline(ev, "15.14")
    r = tl.buff_remaining_by_player("baron", 1_600_000)
    assert list(r[:5]) == [180.0, 0.0, 180.0, 180.0, 180.0] and r[5:].sum() == 0
    s = tl.state(1_649_999)
    assert s["blue_baron_buff_n"] == 4 and s["blue_baron_buff_s"] == pytest.approx(130.001)
    r = tl.buff_remaining_by_player("baron", 1_650_000)
    assert r[0] == 0.0 and r[2] == pytest.approx(130.0)
    s = tl.state(1_650_000)
    assert s["blue_baron_buff_n"] == 3 and s["red_baron_buff_n"] == 0
    s = tl.state(1_780_000)
    assert s["blue_baron_buff_n"] == 0 and s["blue_baron_buff_s"] == 0.0


def test_future_death_does_not_remove_buff_before_it_happens():
    ev = [monster(1_600_000, 200, "BARON_NASHOR"), death(1_700_000, 6, 1)]
    full = ObjectiveTimeline(ev, "15.14").state(1_650_000)
    assert full["red_baron_buff_n"] == 5 and full["red_baron_buff_s"] == pytest.approx(130.0)
    assert same_state(full, objective_state_at(ev, "15.14", 1_650_000))


def test_elder_buff_duration():
    ev = [drake(300_000 * i, 200, "EARTH_DRAGON") for i in range(1, 5)] + [drake(2_000_000, 200, "ELDER_DRAGON")]
    tl = ObjectiveTimeline(ev, "15.14")
    assert tl.state(2_000_000)["red_elder_buff_s"] == pytest.approx(150.0)
    assert tl.state(2_000_000)["red_elder_buff_n"] == 5
    assert tl.state(2_150_000)["red_elder_buff_s"] == 0.0


# ---------------------------------------------------------------------------- structures
def test_inhibitor_and_nexus_respawn_and_tiers():
    ev = [building(600_000, 200, "TOP_LANE", tower="OUTER_TURRET"),
          building(1_000_000, 200, "TOP_LANE", tower="INNER_TURRET"),
          building(1_100_000, 200, "TOP_LANE", tower="BASE_TURRET"),
          building(1_200_000, 200, "TOP_LANE", btype="INHIBITOR_BUILDING"),
          building(1_300_000, 200, "MID_LANE", tower="NEXUS_TURRET", pos=(13052, 12612))]
    tl = ObjectiveTimeline(ev, "15.14")
    s = tl.state(1_300_000)
    assert (s["red_towers_outer"], s["red_towers_inner"], s["red_towers_base"], s["red_towers_nexus"]) == (2, 2, 2, 1)
    assert s["blue_towers_outer"] == 3 and s["blue_towers_nexus"] == 2
    assert s["red_inhib_top_respawn_s"] == pytest.approx(200.0) and s["red_inhibs_down"] == 1
    assert s["red_nexus_turret_respawn_max_s"] == pytest.approx(180.0)
    s = tl.state(1_480_000)                                      # nexus turret back after 3:00
    assert s["red_towers_nexus"] == 2 and s["red_nexus_turret_respawn_max_s"] == 0.0
    s = tl.state(1_500_000)                                      # inhibitor back after 5:00
    assert s["red_inhib_top_respawn_s"] == 0.0 and s["red_inhibs_down"] == 0
    assert s["red_towers_outer"] == 2                            # lane turrets never come back


def test_plates():
    ev = [{"type": "TURRET_PLATE_DESTROYED", "timestamp": 400_000, "teamId": 100, "laneType": "BOT_LANE", "killerId": 6},
          building(700_000, 100, "MID_LANE", tower="OUTER_TURRET")]
    tl = ObjectiveTimeline(ev, "15.14")
    s = tl.state(700_000)
    assert s["blue_plates_bot"] == 4 and s["blue_plates_mid"] == 0 and s["red_plates_top"] == 5
    assert tl.state(840_000)["blue_plates_bot"] == 0             # 15.x plates fall at 14:00
    assert ObjectiveTimeline(ev, "16.15").state(900_000)["blue_plates_bot"] == 4   # permanent in 16.x


# ---------------------------------------------------------------------------- exactness
def test_state_uses_only_events_up_to_t_synthetic():
    rnd = random.Random(0)
    ev = [drake(330_000, 100), drake(640_000, 200), soul_given(640_500, 0, "Cloud"),
          monster(500_000, 100, "HORDE"), monster(960_000, 100, "RIFTHERALD"),
          monster(1_560_000, 100, "BARON_NASHOR"), death(1_600_000, 3, 8), death(1_555_000, 4, 9),
          building(1_700_000, 200, "BOT_LANE", btype="INHIBITOR_BUILDING")]
    tl = ObjectiveTimeline(ev, "15.14")
    for _ in range(200):
        t = rnd.randint(0, 2_000_000)
        assert same_state(tl.state(t), objective_state_at(ev, "15.14", t)), t


def test_feature_vector_names_stable():
    names = feature_names("15.14")
    assert len(names) == len(set(names)) == len(ObjectiveTimeline([], "15.14").features(0))
    assert names == feature_names("16.15")


# ---------------------------------------------------------------------------- real 15.14 events
def _real_1514(n):
    if not PATCH_INDEX.exists():
        pytest.skip("15.14 cache not available")
    idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in idx.items() if v == "15.14")
    random.Random(11).shuffle(ids)
    out = []
    for mid in ids[:n]:
        meta = json.loads((CACHE / f"{mid}.meta.json").read_text(encoding="utf-8"))
        assert meta["patch"] == "15.14"
        out.append((mid, meta, json.loads((CACHE / f"{mid}.events.json").read_text(encoding="utf-8"))))
    return out


def test_real_1514_kills_happen_when_timer_says_up():
    mon = {"BARON_NASHOR": "baron", "RIFTHERALD": "herald", "ATAKHAN": "atakhan", "HORDE": "voidgrub"}
    n_kills = 0
    for mid, meta, ev in _real_1514(40):
        tl = ObjectiveTimeline(ev, "15.14", meta["team_map"])
        for e in tl.events:
            if e["type"] != "ELITE_MONSTER_KILL":
                continue
            s = tl.state(int(e["timestamp"]) - 1)
            if e["monsterType"] == "DRAGON":
                assert s["dragon_up"] == 1.0, (mid, e)
                assert s["dragon_next_is_elder"] == float(e.get("monsterSubType") == "ELDER_DRAGON"), (mid, e)
            else:
                k = mon[e["monsterType"]]
                assert s[f"{k}_up"] == 1.0 or s.get(f"{k}_in_despawn_grace") == 1.0, (mid, e)
            n_kills += 1
    assert n_kills > 200


def test_real_1514_dragon_soul_given_meaning():
    """teamId 0 = Rift element 0.5 s after the 2nd drake; teamId 100/200 = the soul, which the
    drake count gives independently."""
    n0 = nt = 0
    for mid, meta, ev in _real_1514(60):
        tl = ObjectiveTimeline(ev, "15.14", meta["team_map"])
        d = tl.soul_event_diagnostics()
        for r in d["events"]:
            if r["teamId"] == 0:
                n0 += 1
                assert r["drakes_before"] == 2 and 0 <= r["lag_ms"] <= 1000, (mid, r)
                assert tl.rift_element(r["ts"]) is not None
                assert all(el == tl.rift_element(r["ts"]) for el in d["later_drake_elements"]), mid
            else:
                nt += 1
                assert d["derived_soul"] is not None and d["derived_soul"][1] == r["teamId"], (mid, r)
                assert 0 <= r["ts"] - d["derived_soul"][0] <= 1000, (mid, r)
        if d["derived_soul"] is not None:
            assert any(r["teamId"] == d["derived_soul"][1] for r in d["events"]), mid
    assert n0 > 20 and nt >= 1


def test_real_1514_no_future_events():
    rnd = random.Random(3)
    for mid, meta, ev in _real_1514(25):
        tl = ObjectiveTimeline(ev, "15.14", meta["team_map"])
        end = max(int(e["timestamp"]) for e in ev)
        for _ in range(8):
            t = rnd.randint(0, end)
            assert same_state(tl.state(t), objective_state_at(ev, "15.14", t, meta["team_map"])), (mid, t)


def test_rules_file_is_the_default():
    assert RULES_PATH.name == "objective_rules.json" and RULES_PATH.exists()
