"""Tests for gameplay/state_value_v3.py and gameplay/setup_features.py (v4-exact stage 1, task C)."""
from __future__ import annotations

import copy
import inspect
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from core.config import NODE_IDX, cfg
from gameplay import setup_features as sf
from gameplay import state_value_v3 as sv
from gameplay.setup_features import SETUP_COLUMNS, SETUP_NAME_HASH, build_setup
from gameplay.state_value_v2 import StateBuilder as StateBuilderV2
from gameplay.state_value_v3 import (PLAYER_FIELDS, SLOT_PREFIXES, STATE_V3_COLUMNS, STATE_V3_NAME_HASH,
                                     STATE_VERSION, StateBuilderV3, name_hash, q_columns, state_matrix_v3)
from tests.test_exact_roles import STD, make_pack, truncate_pack

CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
PATCH_INDEX = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json")
NODE_NAMES = sorted(NODE_IDX, key=NODE_IDX.get)
CHAMPS = {1: 266, 2: 64, 3: 103, 4: 22, 5: 412, 6: 86, 7: 11, 8: 1, 9: 51, 10: 89}  # 15.14 champions

# node columns StateV3 / setup must never read
STATUS_COLS = ["alive", "has_baron", "has_elder", "baron_remain_norm", "elder_remain_norm", "soul_infernal",
               "soul_ocean", "soul_mountain", "soul_cloud", "soul_hextech", "soul_chemtech", "ult_level_norm"]
CORRUPT_COLS = [NODE_IDX[c] for c in STATUS_COLS] + [j for n, j in NODE_IDX.items() if n.startswith(("cs_", "ds_"))]
READ_COLS = {"totalGold_norm", "curGold_norm", "level_norm", "xp_norm", "hp_pct", "mp_pct", "laneCS_norm",
             "jgCS_norm", "x_norm", "y_norm"}


# --------------------------------------------------------------------------------------------- helpers
def synth(extra_events=(), champions=CHAMPS, n_frames=12, positions=None):
    pack = make_pack(STD, champions=champions, extra_events=extra_events, n_frames=n_frames, positions=positions)
    nm = pack["node_minute"]
    for name, val in (("totalGold_norm", 0.1), ("curGold_norm", 0.05), ("level_norm", 3 / 18.), ("xp_norm", 0.05),
                      ("hp_pct", 0.8), ("mp_pct", 0.6), ("alive", 1.0), ("laneCS_norm", 0.1), ("jgCS_norm", 0.0)):
        nm[:, :, NODE_IDX[name]] = val
    return pack


def same(a, b):
    """dict equality with NaN == NaN and identical key order."""
    if list(a) != list(b):
        return False
    return all((x == y) or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))
               for x, y in zip(a.values(), b.values()))


def kill(ts, victim, killer, assists=(), x=7000, y=7000):
    return {"type": "CHAMPION_KILL", "timestamp": ts, "victimId": victim, "killerId": killer,
            "assistingParticipantIds": list(assists), "position": {"x": x, "y": y}, "bounty": 300}


def _real_ids(n, seed):
    if not (CACHE.exists() and PATCH_INDEX.exists()):
        pytest.skip("15.14 cache not available")
    idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in idx.items() if v == "15.14")
    random.Random(seed).shuffle(ids)
    return ids[:n]


def _load(m, xy=False):
    meta = json.loads((CACHE / f"{m}.meta.json").read_text(encoding="utf-8"))
    assert str(meta["patch"]) == "15.14"
    ev = json.loads((CACHE / f"{m}.events.json").read_text(encoding="utf-8"))
    with np.load(CACHE / f"{m}.npz") as z:
        pack = {"minute_ts": z["minute_ts"], "node_minute": z["node_minute"], "events": ev, "meta": meta}
        if xy:
            pack["xy_raw_minute"] = z["xy_raw_minute"]
    return pack


class NoXYPack(dict):
    """A pack whose xy_raw_minute cannot be read."""

    def __getitem__(self, k):
        if k == "xy_raw_minute":
            raise RuntimeError("xy_raw_minute must not be read")
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k == "xy_raw_minute":
            raise RuntimeError("xy_raw_minute must not be read")
        return super().get(k, default)

    def __contains__(self, k):
        if k == "xy_raw_minute":
            raise RuntimeError("xy_raw_minute must not be read")
        return super().__contains__(k)


# --------------------------------------------------------------------------------------------- contract
def test_column_hash_is_frozen_and_blocks_add_up():
    assert name_hash(STATE_V3_COLUMNS) == STATE_V3_NAME_HASH == sv.STATE_V3_NAME_HASH_COMPUTED
    assert sf.name_hash(SETUP_COLUMNS) == SETUP_NAME_HASH
    assert len(set(STATE_V3_COLUMNS)) == len(STATE_V3_COLUMNS)
    blocks = sv.column_blocks()
    assert sum(v for k, v in blocks.items() if k != "total") == blocks["total"] == len(STATE_V3_COLUMNS)
    assert STATE_VERSION == "exact_v3.1"
    assert sf.COORD_NORM_DIV == float(cfg.COORD_NORM_DIV)
    # role-slot order of the player blocks
    assert SLOT_PREFIXES == ("blue_top_", "blue_jungle_", "blue_middle_", "blue_bottom_", "blue_utility_",
                             "red_top_", "red_jungle_", "red_middle_", "red_bottom_", "red_utility_")
    first = [c for c in STATE_V3_COLUMNS if c.endswith("_totalGold_norm")]
    assert first == [p + "totalGold_norm" for p in SLOT_PREFIXES]
    assert "snapshot_age_s" in STATE_V3_COLUMNS and "snapshot_age_s" not in q_columns()
    assert not any(c.endswith("_champion_id") for c in STATE_V3_COLUMNS)
    # stage-1 decisions: role inference and patch-rule constants are metadata, not inputs
    assert not any(c.startswith("role_") for c in STATE_V3_COLUMNS)
    assert not any(c.startswith("obj_rule_uncertain_") for c in STATE_V3_COLUMNS)
    assert "obj_atakhan_in_patch" not in STATE_V3_COLUMNS
    assert len(sv.OBJ_METADATA_FIELDS) == 9
    assert len(PLAYER_FIELDS) == 16 + 6 + 22 + 28
    assert sum(c.endswith("_itm_flag_lifeline") for c in STATE_V3_COLUMNS) == 10
    assert blocks == {"global_v2": 4, "player_v2": 160, "player_event": 60, "player_item": 220,
                      "player_champion": 280, "team_v2": 188, "team_objective": 84, "total": 996}
    assert len(STATE_V3_COLUMNS) == 996 and len(q_columns()) == 995


def test_state_matrix_guards_order_and_version():
    b = StateBuilderV3(synth(), "15.14")
    s1, s2 = b.at(200_000), b.at(400_000)
    X = state_matrix_v3([s1, s2])
    assert X.shape == (2, len(STATE_V3_COLUMNS))
    assert X[1, STATE_V3_COLUMNS.index("time_minutes")] == pytest.approx(400_000 / 60000)
    Xq = state_matrix_v3([s1], q_columns())
    assert Xq.shape == (1, len(STATE_V3_COLUMNS) - 1)
    bad = copy.deepcopy(s1)
    bad.state_version = "objective_history_v2_participant_order"
    with pytest.raises(ValueError):
        state_matrix_v3([bad])
    bad = copy.deepcopy(s1)
    bad.values = dict(reversed(list(bad.values.items())))
    with pytest.raises(ValueError):
        state_matrix_v3([bad])
    with pytest.raises(ValueError):
        state_matrix_v3([s1], ["no_such_column"])


def test_setup_has_no_anchor_argument():
    params = list(inspect.signature(build_setup).parameters.values())
    assert [p.name for p in params] == ["pack", "t", "roles"]
    assert not any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in params)
    assert not any("anchor" in n or "kill" in n for n in SETUP_COLUMNS)
    pack = synth()
    with pytest.raises(TypeError):
        build_setup(pack, 300_000, {p: p - 1 for p in range(1, 11)}, anchor=(7000, 7000))  # noqa
    with pytest.raises(TypeError):
        build_setup(pack, 300_000, {p: p - 1 for p in range(1, 11)}, (7000, 7000))  # noqa
    at = inspect.signature(StateBuilderV3.at).parameters
    assert list(at) == ["self", "t", "setup"]


# --------------------------------------------------------------------------------------------- overrides
def test_alive_hp_level_assists_and_respawn_overrides():
    # pid 6 dies at 245 s (level 1 at 15.14: 10 s timer) and respawns at 255 s, frame at 240.037 s
    ev = [{"type": "LEVEL_UP", "timestamp": 241_000, "participantId": 1, "level": 7},
          kill(245_000, 6, 1, assists=(2, 3)), kill(250_000, 7, 2, assists=(1,))]
    b = StateBuilderV3(synth(ev), "15.14")
    s = b.at(247_000).values
    assert s["red_top_alive"] == 0.0 and s["red_top_hp_pct"] == 0.0 and s["red_top_mp_pct"] == 0.0
    assert s["red_top_respawn_remaining_s"] == pytest.approx(8.0)
    assert s["red_top_deaths"] == 1.0 and s["red_top_death_since_snapshot"] == 1.0
    assert s["blue_top_level_norm"] == pytest.approx(7 / 18.)          # frame says level 3
    assert s["blue_top_kills"] == 1.0 and s["blue_top_kills_since_snapshot"] == 1.0
    assert s["blue_jungle_assists"] == 1.0 and s["blue_middle_assists"] == 1.0
    assert s["blue_top_hp_pct"] == pytest.approx(0.8)                    # alive, not respawned: frame value
    s = b.at(256_000).values                                             # respawned after the frame
    assert s["red_top_alive"] == 1.0 and s["red_top_respawned_since_snapshot"] == 1.0
    assert s["red_top_hp_pct"] == 1.0 and s["red_top_mp_pct"] == 1.0 and s["red_top_respawn_remaining_s"] == 0.0
    assert s["blue_top_assists"] == 1.0 and s["blue_jungle_kills_since_snapshot"] == 1.0
    s = b.at(300_037).values                                             # next frame: no longer 'since snapshot'
    assert s["red_top_respawned_since_snapshot"] == 0.0 and s["red_top_hp_pct"] == pytest.approx(0.8)
    assert s["blue_top_kills_since_snapshot"] == 0.0 and s["snapshot_age_s"] == 0.0
    # boundaries: death at exactly t counts, respawn at exactly 255 000 ms is alive
    assert b.at(245_000).values["red_top_alive"] == 0.0
    assert b.at(244_999).values["red_top_alive"] == 1.0
    assert b.at(255_000).values["red_top_alive"] == 1.0


def test_frame_alive_status_columns_are_ignored():
    pack = synth([kill(245_000, 6, 1)])
    ref = StateBuilderV3(pack, "15.14").at(250_000)
    bad = copy.deepcopy(pack)
    bad["node_minute"][:, :, CORRUPT_COLS] = np.nan
    assert same(StateBuilderV3(bad, "15.14").at(250_000).values, ref.values)
    bad["node_minute"][:, :, NODE_IDX["alive"]] = 0.0     # frame says everybody is dead
    assert same(StateBuilderV3(bad, "15.14").at(250_000).values, ref.values)


def test_player_blocks_follow_participant_order_roles_are_metadata(monkeypatch):
    # evidence says pid 1 plays UTILITY etc.; the blocks still follow participant order (pid 1 = blue TOP)
    from tests.test_exact_roles import SHUFFLED
    pack = make_pack(SHUFFLED, champions=CHAMPS)
    for name in ("hp_pct", "mp_pct", "level_norm"):
        pack["node_minute"][:, :, NODE_IDX[name]] = 0.5
    pack["node_minute"][:, :, NODE_IDX["totalGold_norm"]] = np.arange(1, 11) / 100.
    st = StateBuilderV3(pack, "15.14").at(560_000)
    for p in range(1, 11):
        assert st.values[SLOT_PREFIXES[p - 1] + "totalGold_norm"] == pytest.approx(p / 100.)
    assert st.slot_by_pid == {p: p - 1 for p in range(1, 11)} and st.role_source == "participant_order"
    assert st.role_fallback is None and st.role_ambiguous is None and st.role_inferred_agree is None
    # role_check: inference outcome as metadata only; the values do not change
    chk = StateBuilderV3(pack, "15.14", role_check=True).at(560_000)
    assert same(chk.values, st.values) and chk.slot_by_pid == st.slot_by_pid
    assert chk.role_fallback is False and chk.role_inferred_agree == {100: False, 200: False}
    assert {p: chk.inferred_slot_by_pid[p] for p in SHUFFLED} == {
        p: (0 if p <= 5 else 5) + ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"].index(r) for p, r in SHUFFLED.items()}
    def boom(*a, **k):
        raise ValueError("no positions")
    monkeypatch.setattr(sv, "infer_roles", boom)
    bad = StateBuilderV3(pack, "15.14", role_check=True).at(560_000)
    assert bad.role_fallback is True and same(bad.values, st.values)


def test_participant_slots_and_meta_role_slots():
    assert sv.participant_slots({str(p): (100 if p <= 5 else 200) for p in range(1, 11)}) == {p: p - 1 for p in range(1, 11)}
    swapped = {p: (200 if p <= 5 else 100) for p in range(1, 11)}                  # blue = pids 6-10
    assert sv.participant_slots(swapped) == {**{p: p - 6 for p in range(6, 11)}, **{p: p + 4 for p in range(1, 6)}}
    with pytest.raises(ValueError):
        sv.participant_slots({p: 100 for p in range(1, 11)})
    slots = {p: p - 1 for p in range(1, 11)}
    assert sv.meta_role_slots_agree({}, slots) is None
    assert sv.meta_role_slots_agree({"role_slots": {str(p): p - 1 for p in range(1, 11)}}, slots) is True
    assert sv.meta_role_slots_agree({"role_slots": {**{str(p): p - 1 for p in range(1, 11)}, "1": 4, "5": 0}}, slots) is False
    pack = synth()
    pack["meta"]["role_slots"] = {str(p): p - 1 for p in range(1, 11)}
    assert StateBuilderV3(pack, "15.14").at(300_000).role_slots_meta_agree is True


def test_objective_rule_constants_are_metadata():
    st = StateBuilderV3(synth(), "15.14").at(300_000)
    assert set(st.objective_rule_flags) == set(sv.OBJ_METADATA_FIELDS)
    assert st.objective_rule_flags["atakhan_in_patch"] == 1.0
    assert all(v == 0.0 for k, v in st.objective_rule_flags.items() if k.startswith("rule_uncertain_"))


def test_unknown_champion_is_the_only_nan():
    champs = {**CHAMPS, 4: 0, 9: 60001}
    st = StateBuilderV3(synth(champions=champs), "15.14").at(300_000)
    nan = [k for k, v in st.values.items() if not math.isfinite(v)]
    assert nan and set(nan) <= sv.NAN_ALLOWED_COLUMNS
    assert {k.split("_ch_")[0] for k in nan} == {"blue_bottom", "red_bottom"}
    assert st.values["blue_bottom_ch_champ_unknown"] == 1.0 and st.values["red_bottom_ch_champ_unknown"] == 1.0
    assert st.values["blue_top_ch_champ_unknown"] == 0.0


def test_objective_nan_timers_are_encoded():
    ev = [{"type": "ELITE_MONSTER_KILL", "timestamp": 400_000, "monsterType": "RIFTHERALD", "killerTeamId": 100,
           "killerId": 2, "position": {"x": 4750, "y": 9990}}]
    s = StateBuilderV3(synth(ev), "15.14").at(420_000).values
    assert s["obj_herald_spawn_none"] == 1.0 and s["obj_herald_next_spawn_s"] == 0.0 and s["obj_herald_up"] == 0.0
    assert s["obj_blue_heralds"] == 1.0 and s["blue_herald"] == 1.0
    assert s["obj_dragon_spawn_none"] == 0.0 and s["obj_dragon_up"] == 1.0   # first drake at 5:00
    s = StateBuilderV3(synth(ev), "15.14").at(200_000).values
    assert s["obj_herald_spawn_none"] == 0.0 and s["obj_dragon_next_spawn_s"] == pytest.approx(100.0)


def test_patch_argument_is_checked():
    with pytest.raises(ValueError):
        StateBuilderV3(synth(), "15.15")
    assert StateBuilderV3(synth(), None).patch == "15.14"


# --------------------------------------------------------------------------------------------- setup
def test_setup_masks_dead_places_respawned_at_fountain_and_counts_pairs():
    pos = {p: (7000.0, 7000.0) for p in range(1, 11)}
    pos[1] = (7000.0, 8000.0)
    ev = [kill(245_000, 6, 1), kill(246_000, 7, 1)]
    pack = synth(ev, positions=pos)
    roles = {p: p - 1 for p in range(1, 11)}
    s = build_setup(pack, 250_000, roles)
    assert s["red_top_present"] == 0.0 and s["red_top_x"] == 0.0 and s["red_top_dist_own_tower"] == sf.FAR_NORM
    assert s["red_n_present"] == 3.0 and s["blue_n_present"] == 5.0
    assert s["n_pairs"] == 15.0 and s["pairs_within_1600"] == 15.0
    assert s["pair_min_dist"] == 0.0
    assert s["blue_largest_group"] == 5.0
    assert s["blue_top_zone_lane"] + s["blue_top_zone_river"] + s["blue_top_zone_own_jungle"] \
        + s["blue_top_zone_enemy_jungle"] + s["blue_top_zone_own_base"] + s["blue_top_zone_enemy_base"] == 1.0
    s = build_setup(pack, 256_500, roles)                  # pid 6 back (10 s timer), frame still 240 s
    assert s["red_top_present"] == 1.0 and s["red_top_at_fountain"] == 1.0
    assert s["red_top_zone_own_base"] == 1.0 and s["red_n_fountain"] == 2.0   # pid 7 back at 256 s too
    assert build_setup(pack, 255_999, roles)["red_n_fountain"] == 1.0
    assert (s["red_top_x"], s["red_top_y"]) == (sf.FOUNTAIN[200][0] / 16000., sf.FOUNTAIN[200][1] / 16000.)


def test_setup_zones_and_towers():
    # mid lane point, river, own/enemy jungle for blue and red
    polys = sf.lane_polylines()
    assert sf._zone(7400, 7400, 100, polys) == "zone_lane"
    assert sf._zone(9800, 4400, 100, polys) == "zone_river"      # dragon pit
    assert sf._zone(3800, 7900, 100, polys) == "zone_own_jungle"
    assert sf._zone(3800, 7900, 200, polys) == "zone_enemy_jungle"
    assert sf._zone(1000, 1000, 200, polys) == "zone_enemy_base"
    ev = [{"type": "BUILDING_KILL", "timestamp": 300_000, "buildingType": "TOWER_BUILDING", "teamId": 200,
           "laneType": "MID_LANE", "towerType": "OUTER_TURRET", "position": {"x": 8955, "y": 8510}}]
    pos = {p: (8955.0, 8510.0) for p in range(1, 11)}
    pack = synth(ev, positions=pos)
    roles = {p: p - 1 for p in range(1, 11)}
    before = build_setup(pack, 299_999, roles)
    after = build_setup(pack, 300_000, roles)
    assert before["blue_middle_dist_enemy_tower"] == pytest.approx(0.0, abs=1e-6)
    assert after["blue_middle_dist_enemy_tower"] > 0.05
    assert after["red_middle_dist_own_tower"] == after["blue_middle_dist_enemy_tower"]


def test_setup_accepts_role_assignment_and_slot_list():
    from gameplay.role_inference import infer_roles
    pack = synth()
    ra = infer_roles(pack, 400_000)
    a = build_setup(pack, 400_000, ra)
    assert same(a, build_setup(pack, 400_000, ra.slot_by_pid))
    assert same(a, build_setup(pack, 400_000, ra.pid_by_slot))
    with pytest.raises(ValueError):
        build_setup(pack, 400_000, {1: 0})


# --------------------------------------------------------------------------------------------- real 15.14
def test_real_future_information_50x10():
    """StateV3 and setup features from a pack truncated to frames / events <= t equal the full pack's."""
    ids = _real_ids(50, 20260925)
    rng = random.Random(7)
    n = 0
    for m in ids:
        pack = _load(m)
        full = StateBuilderV3(pack, "15.14")
        ts = np.asarray(pack["minute_ts"])
        times = [rng.randint(int(ts[0]), int(ts[-1])) for _ in range(8)] + [rng.randint(60_000, 539_999) for _ in range(2)]
        for t in times:
            a = full.at(t, setup=True)
            b = StateBuilderV3(truncate_pack(pack, t), "15.14").at(t, setup=True)
            assert same(a.values, b.values), (m, t, [k for k in a.values if a.values[k] != b.values[k]][:5])
            assert same(a.setup, b.setup), (m, t)
            assert a.slot_by_pid == b.slot_by_pid and a.snapshot_ms == b.snapshot_ms
            assert all(math.isfinite(v) for v in a.values.values()), (m, t)
            n += 1
    assert n == 500


def test_real_corrupted_inputs_do_not_change_output():
    """node alive / status / cs_* / ds_* columns NaN and every other unread column NaN -> identical output;
    xy_raw_minute access raising -> StateV3 (and setup, which reads node x/y) still runs."""
    for m in _real_ids(5, 99):
        pack = _load(m, xy=True)
        ts = np.asarray(pack["minute_ts"])
        times = (150_000, 420_000, int(ts[-1]) - 1500, int(ts[len(ts) // 2]) + 7_000)
        ref = StateBuilderV3(pack, "15.14")
        refs = [ref.at(t, setup=True) for t in times]
        bad = dict(pack)
        nm = np.array(pack["node_minute"], dtype=np.float32, copy=True)
        nm[:, :, CORRUPT_COLS] = np.nan
        bad["node_minute"] = nm
        bad.pop("xy_raw_minute")
        bad = NoXYPack(bad)
        b = StateBuilderV3(bad, "15.14")
        for t, r in zip(times, refs):
            got = b.at(t, setup=True)
            assert same(got.values, r.values) and same(got.setup, r.setup), (m, t)
        worse = dict(bad)
        nm2 = nm.copy()
        nm2[:, :, [j for n, j in NODE_IDX.items() if n not in READ_COLS]] = np.nan
        worse["node_minute"] = nm2
        b = StateBuilderV3(NoXYPack(worse), "15.14")
        for t, r in zip(times, refs):
            assert same(b.at(t).values, r.values), (m, t)


def test_real_statev2_fields_are_carried_over():
    """Non-overridden StateV2 values equal gameplay.state_value_v2 at the same t (mapped through slots)."""
    over = {"alive", "hp_pct", "mp_pct", "level_norm"}
    for m in _real_ids(8, 5):
        pack = _load(m)
        tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
        order = sorted(tm, key=lambda p: (tm[p], p))
        b3 = StateBuilderV3(pack, "15.14")
        b2 = StateBuilderV2(pack, NODE_NAMES)
        ts = np.asarray(pack["minute_ts"])
        for t in (int(ts[2]) + 5_000, int(ts[len(ts) // 2]) + 31_000, int(ts[-1])):
            s3, s2 = b3.at(t), b2.at(t)
            pid_by_slot = {s: p for p, s in s3.slot_by_pid.items()}
            for k, v in s2.values.items():
                if k.startswith("participant_slot"):
                    slot, field = k[len("participant_slot"):].split("_", 1)
                    if field in over or field == "champion_id":
                        continue
                    pid = order[int(slot)]
                    s = s3.slot_by_pid[pid]
                    assert s3.values[SLOT_PREFIXES[s] + field] == v, (m, t, k)
                else:
                    assert s3.values[k] == v, (m, t, k)
            # the overrides agree with the frame when nothing happened since it and the player is alive
            for s, pre in enumerate(SLOT_PREFIXES):
                p = pid_by_slot[s]
                lvl2 = s2.values[f"participant_slot{order.index(p)}_level_norm"]
                assert s3.values[pre + "level_norm"] >= lvl2
