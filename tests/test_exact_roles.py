"""Tests for gameplay/role_inference.py (v4-exact stage 1, task B2)."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest

from core.config import NODE_IDX, cfg
from gameplay import role_inference as ri
from gameplay.role_inference import ROLES, infer_roles, lane_polylines, point_polyline_distance, role_slots

CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
PATCH_INDEX = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json")

# lane-phase positions (game units): blue / red side of each role
POS = {
    100: {"TOP": (1300, 11500), "JUNGLE": (3800, 7900), "MIDDLE": (6300, 6600), "BOTTOM": (11500, 1300),
          "UTILITY": (11400, 1500)},
    200: {"TOP": (3300, 13500), "JUNGLE": (11000, 7000), "MIDDLE": (8500, 8200), "BOTTOM": (13500, 3300),
          "UTILITY": (13300, 3400)},
}
FOUNTAIN = {100: (400, 400), 200: (14400, 14400)}


def make_pack(role_of_pid, *, smite=None, support=None, champions=None, purchases=(), extra_events=(),
              n_frames=11, positions=None, node=True, patch="15.14"):
    """Synthetic pack.  role_of_pid: {pid: role}; pids 1-5 team 100, 6-10 team 200.
    smite / support: pids with Smite / a support-item event (ITEM_DESTROYED 3865 at 7:30);
    default = the JUNGLE / UTILITY players.  positions: {pid: (x, y)} or {pid: [(x, y) per frame]}."""
    team = {p: 100 if p <= 5 else 200 for p in range(1, 11)}
    smite = {p for p, r in role_of_pid.items() if r == "JUNGLE"} if smite is None else set(smite)
    support = {p for p, r in role_of_pid.items() if r == "UTILITY"} if support is None else set(support)
    ts = np.array([0] + [j * 60_000 + 37 for j in range(1, n_frames)], dtype=np.int64)
    xy = np.zeros((n_frames, 10, 2), dtype=np.float32)
    for p in range(1, 11):
        pos = (positions or {}).get(p, POS[team[p]][role_of_pid[p]])
        for j in range(n_frames):
            xy[j, p - 1] = FOUNTAIN[team[p]] if j == 0 else (pos[j] if isinstance(pos, list) else pos)
    events = [{"type": "ITEM_DESTROYED", "itemId": 3865, "participantId": p, "timestamp": 450_000} for p in support]
    events += [{"type": "ITEM_PURCHASED", "itemId": it, "participantId": p, "timestamp": ts_}
               for p, it, ts_ in purchases]
    events += list(extra_events)
    events.sort(key=lambda e: e["timestamp"])
    meta = {
        "patch": patch,
        "team_map": {str(p): team[p] for p in range(1, 11)},
        "static_meta": {
            "champion_by_pid": {str(p): int((champions or {}).get(p, 0)) for p in range(1, 11)},
            "summoner_spells_by_pid": {str(p): {"summoner_spell_1_id": 4, "summoner_spell_2_id": 11 if p in smite else 14}
                                       for p in range(1, 11)},
        },
    }
    pack = {"minute_ts": ts, "events": events, "meta": meta}
    if node:
        nm = np.zeros((n_frames, 10, max(NODE_IDX.values()) + 1), dtype=np.float32)
        div = float(cfg.COORD_NORM_DIV)
        nm[:, :, NODE_IDX["x_norm"]] = xy[:, :, 0] / div
        nm[:, :, NODE_IDX["y_norm"]] = xy[:, :, 1] / div
        pack["node_minute"] = nm
    else:
        pack["xy_raw_minute"] = xy
    return pack


STD = {1: "TOP", 2: "JUNGLE", 3: "MIDDLE", 4: "BOTTOM", 5: "UTILITY",
       6: "TOP", 7: "JUNGLE", 8: "MIDDLE", 9: "BOTTOM", 10: "UTILITY"}
SHUFFLED = {1: "UTILITY", 2: "MIDDLE", 3: "JUNGLE", 4: "TOP", 5: "BOTTOM",
            6: "BOTTOM", 7: "UTILITY", 8: "TOP", 9: "JUNGLE", 10: "MIDDLE"}

PID_KEYS = ("participantId", "killerId", "victimId", "creatorId")


def permute_pack(pack, perm):
    """Relabel participants: old pid p becomes perm[p] (perm must map within a team)."""
    pk = copy.deepcopy(pack)
    inv = {v: k for k, v in perm.items()}
    for e in pk["events"]:
        for k in PID_KEYS:
            if k in e and 1 <= int(e[k] or 0) <= 10:
                e[k] = perm[int(e[k])]
        if "assistingParticipantIds" in e:
            e["assistingParticipantIds"] = [perm[int(a)] for a in e["assistingParticipantIds"]]
    for key in ("node_minute", "xy_raw_minute"):
        if key in pk:
            arr = np.asarray(pk[key])
            pk[key] = arr[:, [inv[p] - 1 for p in range(1, 11)]]
    meta = pk["meta"]
    meta["team_map"] = {str(perm[int(p)]): v for p, v in meta["team_map"].items()}
    sm = meta.get("static_meta") or {}
    if isinstance(sm, str):
        import ast
        sm = ast.literal_eval(sm)
    for k, d in list(sm.items()):
        if isinstance(d, dict) and set(map(str, d)) <= {str(p) for p in range(1, 11)} and len(d) == 10:
            sm[k] = {str(perm[int(p)]): v for p, v in d.items()}
    meta["static_meta"] = sm
    return pk


def within_team_perm(seed):
    rng = random.Random(seed)
    a, b = list(range(1, 6)), list(range(6, 11))
    rng.shuffle(a)
    rng.shuffle(b)
    return {**{p: a[p - 1] for p in range(1, 6)}, **{p: b[p - 6] for p in range(6, 11)}}


def truncate_pack(pack, t):
    pk = dict(pack)
    keep = np.asarray(pack["minute_ts"]) <= t
    pk["minute_ts"] = np.asarray(pack["minute_ts"])[keep]
    for key in ("node_minute", "xy_raw_minute"):
        if key in pack:
            pk[key] = np.asarray(pack[key])[keep]
    pk["events"] = [e for e in pack["events"] if int(e["timestamp"]) <= t]
    return pk


# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------
def test_lane_polylines_pass_through_towers():
    polys = lane_polylines()
    assert set(polys) == {"TOP", "MIDDLE", "BOTTOM"}
    assert polys["TOP"].shape == (9, 2) and polys["BOTTOM"].shape == (9, 2) and polys["MIDDLE"].shape == (8, 2)
    d = json.loads(ri.ANCHORS_PATH.read_text(encoding="utf-8"))
    lane = {"TOP_LANE": "TOP", "MID_LANE": "MIDDLE", "BOT_LANE": "BOTTOM"}
    for t in d["towers"]:
        if t["tower"] == "NEXUS_TURRET":
            continue
        dist = point_polyline_distance(np.array([[t["x"], t["y"]]]), polys[lane[t["lane"]]])[0]
        assert dist < 1e-6
    # the corners: top-left for TOP, bottom-right for BOTTOM
    assert polys["TOP"][4][0] < 2000 and polys["TOP"][4][1] > 13000
    assert polys["BOTTOM"][4][0] > 13000 and polys["BOTTOM"][4][1] < 2000


def test_point_polyline_distance_segment_cases():
    poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    pts = np.array([[5.0, 3.0], [-4.0, 3.0], [13.0, 5.0], [12.0, 14.0]])
    np.testing.assert_allclose(point_polyline_distance(pts, poly), [3.0, 5.0, 3.0, np.hypot(2, 4)])


# ---------------------------------------------------------------------------------------------
# synthetic assignments
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("roles", [STD, SHUFFLED])
def test_standard_and_shuffled_participant_order(roles):
    ra = infer_roles(make_pack(roles))
    assert ra.roles == roles
    assert not ra.ambiguous
    for team in (100, 200):
        tr = ra.teams[team]
        assert tr.n_smite == 1 and tr.n_support == 1 and tr.margin >= ri.MARGIN_MIN
    # slots follow core.roles layout [blue TOP..UTILITY, red TOP..UTILITY]
    for p, s in ra.slot_by_pid.items():
        assert s == (0 if p <= 5 else 5) + ROLES.index(roles[p])
        assert ra.pid_by_slot[s] == p
    assert role_slots(make_pack(roles)) == ra.slot_by_pid


def test_roles_match_core_role_order():
    from core.config import ROLE_ORDER
    assert list(ROLES) == list(ROLE_ORDER)


def test_xy_raw_fallback_equals_node_path():
    a = infer_roles(make_pack(SHUFFLED, node=True))
    b = infer_roles(make_pack(SHUFFLED, node=False))
    assert a.roles == b.roles


@pytest.mark.parametrize("seed", range(5))
def test_permutation_equivariance_synthetic(seed):
    pack = make_pack(STD, purchases=[(4, 1055, 30_000), (9, 1055, 30_000)])
    perm = within_team_perm(seed)
    a = infer_roles(pack)
    b = infer_roles(permute_pack(pack, perm))
    assert {perm[p]: r for p, r in a.roles.items()} == b.roles
    for team in (100, 200):
        assert a.teams[team].margin == pytest.approx(b.teams[team].margin)


def test_missing_smite_jungle_by_position():
    ra = infer_roles(make_pack(STD, smite=[7]))  # blue team has no Smite
    assert ra.roles == STD
    tr = ra.teams[100]
    assert tr.n_smite == 0 and "smite_count=0" in tr.reasons and tr.ambiguous
    assert not ra.teams[200].ambiguous


def test_missing_summoner_spells_falls_back_to_node_minute_then_flags():
    pack = make_pack(STD)
    del pack["meta"]["static_meta"]["summoner_spells_by_pid"]
    j1, j2 = NODE_IDX["summoner_spell_1_id"], NODE_IDX["summoner_spell_2_id"]
    pack["node_minute"][:, :, j1] = 4
    pack["node_minute"][:, :, j2] = 14
    pack["node_minute"][:, [1, 6], j2] = 11
    ra = infer_roles(pack)
    assert ra.roles == STD and not ra.ambiguous
    pack2 = make_pack(STD, node=False)
    del pack2["meta"]["static_meta"]["summoner_spells_by_pid"]
    ra2 = infer_roles(pack2)
    assert all("summoner_spells_missing" in tr.reasons and tr.ambiguous for tr in ra2.teams.values())


def test_two_smites_the_jungle_positioned_one_is_jungler():
    ra = infer_roles(make_pack(STD, smite=[1, 2, 7]))  # blue top laner also runs Smite
    assert ra.roles == STD
    assert ra.teams[100].n_smite == 2 and ra.teams[100].ambiguous


def test_two_supports_bot_one_is_utility():
    # blue: the mid laner also has a support-item event; red: the support is swapped into the
    # participant slot of the ADC
    roles = dict(STD)
    ra = infer_roles(make_pack(roles, support=[3, 5, 10]))
    assert ra.roles == roles
    assert ra.teams[100].n_support == 2 and "support_item_count=2" in ra.teams[100].reasons
    assert ra.teams[100].ambiguous and not ra.teams[200].ambiguous


def test_support_item_on_smite_player_is_overruled():
    ra = infer_roles(make_pack(STD, support=[2, 5, 7, 10]))
    assert ra.roles == STD
    assert ra.teams[100].n_support == 1 and not ra.ambiguous


def test_no_support_item_tiebreak_by_champion_tags():
    # 15.14 table: 22 Ashe (Marksman, Support), 51 Caitlyn (Marksman), 412 Thresh (Support, Tank)
    champs = {4: 412, 5: 51, 9: 51, 10: 412}  # blue: participant 4 is the support, 5 the ADC
    roles = {**STD, 4: "UTILITY", 5: "BOTTOM"}
    ra = infer_roles(make_pack(roles, support=[], champions=champs))
    assert ra.roles == roles
    for team in (100, 200):
        assert "no_support_item" in ra.teams[team].reasons


def test_no_support_item_tiebreak_by_gold():
    # no champion ids (unknown tags); the ADC starts Doran's Blade and buys a Long Sword, the support
    # starts with a potion only (World Atlas is credited to participant 0 in Match-V5)
    purchases = [(4, 1055, 20_000), (4, 1036, 200_000), (5, 2003, 20_000),
                 (9, 1055, 20_000), (9, 1036, 200_000), (10, 2003, 20_000)]
    pack = make_pack(STD, support=[], purchases=purchases)
    ra = infer_roles(pack)  # gold_mode='start'
    assert ra.roles == STD and not ra.ambiguous
    assert ra.evidence[5]["low_start"] == 1.0 and ra.evidence[4]["low_start"] == 0.0
    assert ra.evidence[4]["gold"] == pytest.approx(450.0)  # starting inventory: 1036 (at 200 s) not counted
    assert infer_roles(pack, gold_mode="inventory").roles == STD  # v1 rule
    # without the tie-break the bot pair is a cost tie -> flagged
    rb = infer_roles(pack, use_tiebreak=False)
    assert rb.teams[100].margin < ri.MARGIN_MIN and rb.teams[100].ambiguous
    with pytest.raises(ValueError):
        infer_roles(pack, gold_mode="nope")


def test_starting_inventory_beats_misleading_tags_and_inventory_value():
    # 16.15-style lane: a Support-tagged mage bottom laner (Seraphine 147: Support, Mage) with a fighter
    # support (Pantheon 80: Fighter, Assassin) who has out-bought the carry by 6 min
    champs = {4: 147, 5: 80, 9: 147, 10: 80}
    purchases = []
    for adc, sup in ((4, 5), (9, 10)):
        purchases += [(adc, 1056, 10_000), (adc, 2003, 10_000), (sup, 2003, 15_000), (sup, 2003, 15_000),
                      (sup, 1036, 188_000), (sup, 1036, 309_000), (sup, 1001, 320_000)]
    pack = make_pack(STD, support=[], champions=champs, purchases=purchases)
    ra = infer_roles(pack, t=360_000)
    assert ra.roles == STD and not ra.ambiguous
    rv1 = infer_roles(pack, t=360_000, gold_mode="inventory")
    assert rv1.roles[4] == "UTILITY" and rv1.roles[5] == "BOTTOM"  # the v1 failure mode


def test_lane_swap_is_assigned_by_position():
    # blue participant 1 plays mid and participant 3 plays top: position, not participant order, decides
    swapped = {**STD, 1: "MIDDLE", 3: "TOP"}
    ra = infer_roles(make_pack(swapped))
    assert ra.roles[1] == "MIDDLE" and ra.roles[3] == "TOP"


def test_lane_swap_during_window_uses_median():
    # participant 1 is top for minutes 3-6 and wanders mid for minutes 7-8: median over 6 frames
    track = [POS[100]["TOP"]] * 7 + [POS[100]["MIDDLE"]] * 2 + [POS[100]["TOP"]] * 2
    ra = infer_roles(make_pack(STD, positions={1: track}))
    assert ra.roles == STD


def test_identical_positions_are_flagged_ambiguous():
    pos = {1: POS[100]["MIDDLE"]}  # blue top and mid stand on the same spot
    ra = infer_roles(make_pack(STD, positions=pos))
    tr = ra.teams[100]
    assert tr.margin < ri.MARGIN_MIN and tr.ambiguous


# ---------------------------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------------------------
def test_support_event_after_t_or_horizon_is_ignored():
    pack = make_pack(STD)  # support events at 450 000 ms
    assert infer_roles(pack, t=449_999).teams[100].n_support == 0
    assert infer_roles(pack, t=450_000).teams[100].n_support == 1
    late = make_pack(STD, support=[])
    late["events"].append({"type": "ITEM_DESTROYED", "itemId": 3865, "participantId": 1, "timestamp": 540_000})
    ra = infer_roles(late)  # after the default horizon 539 999 ms
    assert ra.teams[100].n_support == 0 and ra.roles == STD
    assert infer_roles(late, horizon_ms=600_000).roles[1] == "UTILITY"


@pytest.mark.parametrize("t", [150_000, 240_037, 300_000, 419_999, 450_000, None])
def test_no_future_frames_or_events(t):
    base = make_pack(SHUFFLED, purchases=[(5, 1055, 30_000), (6, 1055, 30_000)])
    # corrupt everything strictly after t (or after the horizon when t is None)
    cut = ri.HORIZON_MS if t is None else t
    bad = copy.deepcopy(base)
    later = np.asarray(bad["minute_ts"]) > cut
    rng = np.random.default_rng(0)
    bad["node_minute"][later, :, NODE_IDX["x_norm"]] = rng.random((later.sum(), 10))
    bad["node_minute"][later, :, NODE_IDX["y_norm"]] = rng.random((later.sum(), 10))
    bad["events"] = [e for e in bad["events"] if e["timestamp"] <= cut] + [
        {"type": "ITEM_DESTROYED", "itemId": 3865, "participantId": p, "timestamp": int(cut) + 1} for p in range(1, 11)]
    a, b = infer_roles(base, t), infer_roles(bad, t)
    assert a.roles == b.roles
    assert [a.teams[k].margin for k in (100, 200)] == pytest.approx([b.teams[k].margin for k in (100, 200)])
    c = infer_roles(truncate_pack(base, cut), t)
    assert c.roles == a.roles


def test_early_t_uses_fallback_frames_then_flags():
    pack = make_pack(SHUFFLED)
    ra = infer_roles(pack, t=150_000)  # frames at 60 s and 120 s; support events (7:30) not yet seen
    assert all(ra.roles[p] == r for p, r in SHUFFLED.items() if r in ("TOP", "JUNGLE", "MIDDLE"))
    assert {ra.roles[p] for p, r in SHUFFLED.items() if r in ("BOTTOM", "UTILITY")} == {"BOTTOM", "UTILITY"}
    for tr in ra.teams.values():
        # no support item, no tags, no gold: the bottom pair is a cost tie -> flagged
        assert "fallback_frames" in tr.reasons and "no_support_item" in tr.reasons and tr.ambiguous
    rb = infer_roles(pack, t=90_000)  # one frame -> few_frames
    assert all(tr.ambiguous and "few_frames" in tr.reasons for tr in rb.teams.values())
    rc = infer_roles(pack, t=30_000)  # no frame in (60 s, t]
    assert all(tr.ambiguous and "no_position" in tr.reasons for tr in rc.teams.values())


def test_dead_frames_are_excluded():
    # blue top laner killed at 4:59 (dead at the 5:00 frame, 10+ s timer); the frame shows him mid
    track = [POS[100]["TOP"]] * 11
    track[5] = POS[100]["MIDDLE"]
    kill = {"type": "CHAMPION_KILL", "victimId": 1, "killerId": 8, "timestamp": 299_990, "assistingParticipantIds": []}
    with_kill = make_pack(STD, positions={1: track}, extra_events=[kill])
    ev = infer_roles(with_kill).evidence[1]["lane"]
    assert ev["n_frames"] == 5  # minutes 3, 4, 6, 7, 8
    no_frame = make_pack(STD, extra_events=[kill])
    assert infer_roles(no_frame).evidence[1]["lane"]["TOP"] == pytest.approx(ev["TOP"])


def test_contaminated_inputs_do_not_matter():
    pack = make_pack(SHUFFLED)
    ref = infer_roles(pack)

    class NoXY(dict):
        def __getitem__(self, k):
            if k == "xy_raw_minute":
                raise RuntimeError("xy_raw_minute must not be read")
            return super().__getitem__(k)

        def get(self, k, default=None):
            if k == "xy_raw_minute":
                raise RuntimeError("xy_raw_minute must not be read")
            return super().get(k, default)

    guarded = NoXY(pack)
    guarded["xy_raw_minute"] = np.zeros((11, 10, 2))
    assert infer_roles(guarded).roles == ref.roles
    nm = pack["node_minute"].copy()
    for name, j in NODE_IDX.items():
        if name in ("x_norm", "y_norm") or name.startswith("summoner_spell"):
            continue
        nm[:, :, j] = np.nan
    assert infer_roles({**pack, "node_minute": nm}).roles == ref.roles


def test_bad_team_map_raises():
    pack = make_pack(STD)
    pack["meta"]["team_map"]["1"] = 200
    with pytest.raises(ValueError):
        infer_roles(pack)


def test_unknown_patch_still_assigns_with_notes():
    pack = make_pack(STD, patch="99.1")
    ra = infer_roles(pack)
    assert ra.roles == STD
    notes = " ".join(ra.teams[100].reasons)
    assert "no_survival_rule" in notes and "no_champion_table" in notes and "no_item_table" in notes


# ---------------------------------------------------------------------------------------------
# real 15.14 matches (proxy truth: participant order, which equals teamPosition in the 553
# 16.15 KR matches; see the stage-1 validation report)
# ---------------------------------------------------------------------------------------------
def _real_ids(n, seed):
    if not (CACHE.exists() and PATCH_INDEX.exists()):
        pytest.skip("15.14 cache not available")
    idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in idx.items() if v == "15.14")
    random.Random(seed).shuffle(ids)
    return ids[:n]


def _load(m):
    meta = json.loads((CACHE / f"{m}.meta.json").read_text(encoding="utf-8"))
    assert str(meta["patch"]) == "15.14"
    ev = json.loads((CACHE / f"{m}.events.json").read_text(encoding="utf-8"))
    z = np.load(CACHE / f"{m}.npz")
    return {"minute_ts": z["minute_ts"], "node_minute": z["node_minute"], "events": ev, "meta": meta}


def test_real_1514_accuracy_permutation_and_no_future():
    ids = _real_ids(30, 4242)
    correct = total = 0
    for i, m in enumerate(ids):
        pack = _load(m)
        ra = infer_roles(pack)
        correct += sum(ra.roles[p] == ROLES[(p - 1) % 5] for p in range(1, 11))
        total += 10
        perm = within_team_perm(i)
        rb = infer_roles(permute_pack(pack, perm))
        assert {perm[p]: r for p, r in ra.roles.items()} == rb.roles
        for t in (200_000, 400_000):
            assert infer_roles(truncate_pack(pack, t), t).roles == infer_roles(pack, t).roles
    assert correct / total >= 0.97
