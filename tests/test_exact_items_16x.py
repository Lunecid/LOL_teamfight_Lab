"""16.x item rules (gameplay/item_state.py _R16Context, config/game_rules/item_rules_16x.json): gating, each rule,
causality, and the 15.14 byte-identity on 50 cache matches."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gameplay.item_state import (  # noqa: E402
    ItemStateIndex, load_item_table_v2, load_rules_16x, patch_major, replay_inventory, role_quest_start,
    shop_events_by_participant, tier3_boots_map, uses_rules_16x,
)

T16 = load_item_table_v2("16.15")
T15 = load_item_table_v2("15.14")
R16 = load_rules_16x()
NO_TP = {str(p): {"summoner_spell_1_id": 4, "summoner_spell_2_id": 14} for p in range(1, 11)}


def buy(ts, iid, pid=1):
    return {"type": "ITEM_PURCHASED", "timestamp": ts, "participantId": pid, "itemId": iid}


def destroy(ts, iid, pid=1):
    return {"type": "ITEM_DESTROYED", "timestamp": ts, "participantId": pid, "itemId": iid}


def sell(ts, iid, pid=1):
    return {"type": "ITEM_SOLD", "timestamp": ts, "participantId": pid, "itemId": iid}


def undo(ts, before, after, gold, pid=1):
    return {"type": "ITEM_UNDO", "timestamp": ts, "participantId": pid, "beforeId": before, "afterId": after,
            "goldGain": gold}


def kill(ts, killer, victim=7):
    return {"type": "CHAMPION_KILL", "timestamp": ts, "killerId": killer, "victimId": victim,
            "assistingParticipantIds": []}


def level(ts, lv, pid=1):
    return {"type": "LEVEL_UP", "timestamp": ts, "participantId": pid, "level": lv}


def x16(events, **kw):
    kw.setdefault("summoner_spells_by_pid", NO_TP)
    return ItemStateIndex(events, T16, patch="16.15", **kw)


def held(ix, pid, t=10 ** 9, diag=None):
    return sorted(ix.inventory(pid, t, diagnostics=diag))


# ------------------------------------------------------------------ gating (15.x untouched)
def test_patch_major_gate():
    assert patch_major("16.15") == 16 and patch_major("15.14") == 15 and patch_major(None) is None
    assert uses_rules_16x("16.15") and uses_rules_16x("16.1") and uses_rules_16x("17.2")
    assert not uses_rules_16x("15.14") and not uses_rules_16x(None) and not uses_rules_16x("x")
    ev = [buy(1000, 1036)]
    assert ItemStateIndex(ev, T16, patch="16.15").rules16 is not None
    assert ItemStateIndex(ev, T15, patch="15.14").rules16 is None
    assert ItemStateIndex(ev, T16).rules16 is None                               # no patch: never on
    assert ItemStateIndex(ev, T16, patch="16.15", exact=False).rules16 is None   # legacy mode untouched
    assert ItemStateIndex(ev, T16, patch="16.15", rules_16x=False).rules16 is None
    assert ItemStateIndex(ev, T15, patch="15.14", rules_16x=True).rules16 is not None


def test_15x_index_equals_plain_exact_replay():
    ev = [buy(1000, 1036), buy(2000, 1036), destroy(3000, 1036), destroy(3000, 1036), buy(3000, 3133),
          destroy(5000, 2001), buy(6000, 3340), destroy(7000, 3340), buy(7000, 3364)]
    # rules_15x=False: the 15.x rules (tests/test_exact_items_15x.py) are off, so no 16.x rule may change the replay
    ix = ItemStateIndex(ev, T15, patch="15.14", summoner_spells_by_pid=NO_TP, rules_15x=False)
    shop = shop_events_by_participant(ev, T15, assign_support_atlas=True)[1]
    assert ix.grants[1] == [] and ix.role[1] is None and ix.rules16 is None
    assert ix.replay(1, 10 ** 9, since_ms=0) == replay_inventory(shop, T15, 10 ** 9, 0, exact=True)


# ------------------------------------------------------------------ R16-1 / R16-7 / R16-13 start items
def test_role_quest_start_by_slot_and_teleport():
    spells = dict(NO_TP, **{"6": {"summoner_spell_1_id": 12, "summoner_spell_2_id": 4}})
    ix = x16([], summoner_spells_by_pid=spells, champion_by_pid={"7": 9})
    start = {pid: sorted(ix.inventory(pid, 0)) for pid in range(1, 11)}
    assert start[1] == [1200, 3340] and start[6] == [1222, 3340]                 # TOP: Teleport or not
    assert start[2] == [1204, 3340] and start[3] == [1201, 3340]
    assert start[4] == [1202, 3340] and start[5] == [1203, 3340] and start[10] == [1203, 3340]
    assert start[7] == [1204, 3330]                                              # Fiddlesticks: Scarecrow Effigy
    assert role_quest_start(R16, "TOP", [12, 4]) == 1222 and role_quest_start(R16, "TOP", None) == 1200


def test_trinket_purchase_replaces_start_trinket_and_undo_restores_it():
    ix = x16([buy(8000, 3340), buy(9000, 3364), undo(9500, 0, 3364, 0)])
    assert held(ix, 1, 8500) == [1200, 3340]                                     # not two 3340
    assert held(ix, 1, 9200) == [1200, 3364]
    assert held(ix, 1) == [1200, 3340]
    ix = x16([destroy(9000, 3340), buy(9000, 3364)])                              # logged swap
    assert held(ix, 1) == [1200, 3364]


# ------------------------------------------------------------------ R16-2 quest tokens -> rewards
def test_top_and_jungle_quest_rewards():
    ev = [destroy(600_000, 1200, 1), buy(1000, 1101, 2), destroy(700_000, 1204, 2), destroy(700_000, 1101, 2)]
    ix = x16(ev)
    assert held(ix, 1) == [1220, 3340]
    assert held(ix, 2, 699_999) == [1101, 1204, 3340] and held(ix, 2) == [1209, 3340]
    spells = dict(NO_TP, **{"1": {"summoner_spell_1_id": 12, "summoner_spell_2_id": 4}})
    assert held(x16([destroy(600_000, 1222, 1)], summoner_spells_by_pid=spells), 1) == [1221, 3340]


# ------------------------------------------------------------------ R16-4 mid tier-3 boots
def test_tier3_map_includes_unlabelled_gunmetal():
    t3 = tier3_boots_map(T16)
    assert t3[3006] == 3172 and t3[3020] == 3175 and t3[3111] == 3173 and 1001 not in t3


def test_mid_quest_upgrades_held_boots_at_completion():
    ev = [buy(300_000, 1001, 3), destroy(400_000, 1001, 3), buy(400_000, 3020, 3),
          destroy(600_000, 3020, 3), destroy(600_000, 1201, 3)]
    ix = x16(ev)
    assert held(ix, 3, 599_999) == [1201, 3020, 3340]
    assert held(ix, 3) == [1206, 3175, 3340]


def test_mid_boots_bought_after_completion_upgrade_at_once_and_blind_undo():
    ev = [destroy(500_000, 1201, 3), buy(600_000, 1001, 3),
          destroy(630_000, 1001, 3), destroy(630_000, 3006, 3), buy(630_000, 3006, 3),
          undo(631_000, 0, 0, 800, 3)]
    ix = x16(ev)
    assert held(ix, 3, 630_500) == [1206, 3172, 3340]
    assert held(ix, 3) == [1001, 1206, 3340]                                     # the upgrade is undone
    d = {}
    ix.inventory(3, 10 ** 9, diagnostics=d)
    assert d["r16_blind_undo"] == 1 and d["unowned_removal"] == 0


# ------------------------------------------------------------------ R16-5 bot boots into the role slot
def test_bot_quest_moves_boots_into_role_slot():
    ev = [buy(300_000, 1001, 4), destroy(600_000, 1202, 4), destroy(600_000, 1001, 4), destroy(600_000, 1207, 4)]
    assert held(x16(ev), 4) == [1001, 3340]
    ev = [destroy(600_000, 1202, 4), buy(700_000, 1001, 4), destroy(700_000, 1207, 4)]
    ix = x16(ev)
    assert held(ix, 4, 650_000) == [1207, 3340] and held(ix, 4) == [1001, 3340]


# ------------------------------------------------------------------ R16-3 support ward slot
def test_support_ward_slot_and_quest_stage():
    s = 5
    ev = [destroy(100_000, 2055, s), destroy(100_000, 1203, s), buy(100_000, 2055, s),   # ward into the slot
          destroy(150_000, 2055, s),                                                     # placed: 1203 back
          destroy(200_000, 2055, s), destroy(200_000, 1203, s), buy(200_000, 2055, s),
          destroy(300_000, 3866, s),                                  # quest done while the slot holds a ward
          destroy(350_000, 2055, s),                                  # placed: the reward placeholder comes back
          destroy(400_000, 2055, s), destroy(400_000, 1208, s), buy(400_000, 2055, s),
          buy(401_000, 2055, s)]                                      # a second ward stacks
    ix = x16(ev)
    assert held(ix, s, 120_000) == [2055, 3340]
    assert held(ix, s, 160_000) == [1203, 3340]
    assert held(ix, s, 360_000) == [1208, 3340, 3867]
    assert held(ix, s) == [2055, 2055, 3340, 3867]
    d = {}
    ix.inventory(s, 10 ** 9, diagnostics=d)
    assert d["unowned_removal"] == 1                                  # only the 3866 (never bought here)
    ev2 = [destroy(300_000, 3866, s), destroy(300_000, 1203, s)]      # completion with the placeholder held
    assert held(x16(ev2), s) == [1208, 3340, 3867]


def test_support_blind_undo_of_slot_ward():
    s = 5
    ev = [destroy(100_000, 2055, s), destroy(100_000, 1203, s), buy(100_000, 2055, s), undo(101_000, 0, 0, 75, s)]
    assert held(x16(ev), s) == [1203, 3340]


# ------------------------------------------------------------------ R16-6 / R16-8 recall items, herald eye
def test_recall_items_dropped_and_herald_eye_exchange():
    ev = [buy(200_000, 3364), destroy(200_000, 3340), destroy(900_000, 2001), destroy(900_000, 3364),
          destroy(950_000, 3513), destroy(1_000_000, 2002)]
    ix = x16(ev)
    d = {}
    assert held(ix, 1, 920_000) == [1200, 3513]
    assert held(ix, 1, 10 ** 9, d) == [1200, 3364]
    assert d["unowned_removal"] == 0 and d["r16_transient_dropped"] == 2
    # a lone trinket destroy without a Recall-item destroy stays a plain removal
    assert held(x16([destroy(900_000, 3340)]), 1) == [1200]


# ------------------------------------------------------------------ R16-10 Viego
def test_viego_quest_completion_is_not_a_possession_swap():
    v = 2
    ev = [buy(1000, 1102, v), kill(690_000, v), destroy(700_000, 1204, v), destroy(700_000, 1102, v),
          kill(800_000, v), destroy(805_000, 1036, v), destroy(805_000, 3071, v), destroy(805_000, 1206, v),
          destroy(805_000, 3047, v)]                                   # possessed champion's items leave: ignored
    ix = x16(ev, champion_by_pid={str(v): 234})
    assert held(ix, v, 750_000) == [1209, 3340]
    assert held(ix, v) == [1209, 3340]


# ------------------------------------------------------------------ R16-11 / R16-12 rune grants
def test_rune_grants_biscuits_and_triple_tonic():
    runes = {"1": {"primary_rune_1": 8010, "sub_rune_1": 8345, "sub_rune_2": 8313}}
    ev = [level(60_000, 2), level(150_000, 3), level(400_000, 6), level(700_000, 9), destroy(151_000, 2151)]
    ix = x16(ev, runes_by_pid=runes)
    assert held(ix, 1, 119_999) == [1200, 3340]
    assert held(ix, 1, 150_500) == [1200, 2010, 2151, 3340]
    assert held(ix, 1) == [1200, 2010, 2010, 2010, 2152, 3340]         # no 2150 at level 9
    none = x16(ev, runes_by_pid={"1": {"primary_rune_1": 8010}})
    d = {}
    assert held(none, 1, 10 ** 9, d) == [1200, 3340] and d["unowned_removal"] == 1


# ------------------------------------------------------------------ causality
def test_16x_state_uses_only_events_up_to_t():
    runes = {"1": {"sub_rune_1": 8345, "sub_rune_2": 8313}}
    ev = [level(150_000, 3), destroy(300_000, 1201, 3), buy(400_000, 3020, 3), destroy(401_000, 3020, 3),
          destroy(500_000, 1202, 4), buy(600_000, 1001, 4), destroy(600_000, 1207, 4),
          destroy(100_000, 2055, 5), destroy(100_000, 1203, 5), buy(100_000, 2055, 5), destroy(150_000, 2055, 5),
          destroy(700_000, 2001, 1), destroy(700_000, 3340, 1), destroy(720_000, 3513, 1), level(800_000, 6)]
    full = x16(ev, runes_by_pid=runes)
    for t in range(0, 900_001, 25_000):
        cut = x16([e for e in ev if e["timestamp"] <= t], runes_by_pid=runes)
        for pid in (1, 3, 4, 5):
            assert held(full, pid, t) == held(cut, pid, t), (pid, t)


# ------------------------------------------------------------------ rule file
def test_rule_file_lists_every_rule_with_evidence():
    spec = json.loads((ROOT / "config/game_rules/item_rules_16x.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in spec["rules"]]
    assert ids == [f"R16-{k}" for k in range(1, 14)]
    assert all(r["rule"] and r["evidence"] for r in spec["rules"])
    assert R16["transforms"][1201] == 1206 and R16["chains"]["UTILITY"] == frozenset({1203, 1208})


# ------------------------------------------------------------------ 15.14 byte-identity (50 cache matches)
@pytest.mark.slow
def test_15_14_item_vectors_identical_to_pre_16x_module():
    sys.path.insert(0, str(ROOT / "scripts/exact_v4"))
    import ev4_s1x_items_16x as S
    if not S.PATCH_INDEX.exists() or not S.CACHE.exists():
        pytest.skip("15.14 cache not available")
    digest, n = S.identity_digest()
    assert n == 14_350
    assert digest == S.REF_HASH_1514
