"""15.x item rules (gameplay/item_state.py _R15Context, config/game_rules/item_rules_15x.json): gating, each rule,
gold neutrality, causality, the rule file, the 15.14 identity with the rules off, and role inference's start gold."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gameplay.item_state import (  # noqa: E402
    ItemStateIndex, load_item_table_v2, load_rules_15x, player_item_vector, uses_rules_15x,
)

T15 = load_item_table_v2("15.14")
R15 = load_rules_15x()
BISCUIT = {"1": {"primary_rune_1": 8010, "sub_rune_1": 8345}}
TONIC = {"1": {"primary_rune_1": 8010, "sub_rune_1": 8313}}


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


def x15(events, **kw):
    return ItemStateIndex(events, T15, patch="15.14", **kw)


def held(ix, pid=1, t=10 ** 9, diag=None):
    return sorted(ix.inventory(pid, t, diagnostics=diag))


def spent(ix, pid=1, t=10 ** 9):
    return -ix.replay(pid, t, since_ms=-1)[1]


# ------------------------------------------------------------------ gating
def test_gate_is_patch_major_15_only():
    assert uses_rules_15x("15.14") and uses_rules_15x("15.18") and uses_rules_15x("15.22")
    assert not uses_rules_15x("16.15") and not uses_rules_15x("14.1") and not uses_rules_15x(None)
    assert not uses_rules_15x("nope")
    assert x15([]).rules15 is not None and x15([]).rules16 is None
    assert ItemStateIndex([], T15).rules15 is None                               # no patch: never on
    assert x15([], rules_15x=False).rules15 is None
    assert x15([], exact=False).rules15 is None                                  # legacy mode untouched
    t16 = load_item_table_v2("16.15")
    ix16 = ItemStateIndex([], t16, patch="16.15")
    assert ix16.rules15 is None and ix16.rules16 is not None                    # 16.x keeps its own rules
    assert ItemStateIndex([], t16, patch="16.15", rules_16x=False).rules15 is None


def test_rules_off_reproduces_the_pre_15x_replay():
    ev = [buy(1000, 1001), destroy(400_000, 1001), buy(400_000, 3010), destroy(900_000, 3010)]
    assert held(x15(ev, rules_15x=False)) == [] and held(ItemStateIndex(ev, T15)) == []
    assert held(x15(ev)) == [3013, 3340]


# ------------------------------------------------------------------ R15-1 Symbiotic Soles
def test_r15_1_loose_3010_destroy_becomes_3013():
    assert T15[3013]["from"] == [3010] and T15[3013]["gold_base"] == 0 and not T15[3013].get("specialRecipe")
    ev = [buy(1000, 1001), destroy(400_000, 1001), buy(400_000, 3010), destroy(900_000, 3010)]
    d = {}
    assert held(x15(ev), t=899_999) == [3010, 3340]
    assert held(x15(ev), diag=d) == [3013, 3340] and d["r15_loose_transform"] == 1 and d["unowned_removal"] == 0
    assert spent(x15(ev)) == pytest.approx(spent(x15(ev, rules_15x=False)))     # free: no gold
    v = player_item_vector(x15(ev).inventory(1, 10 ** 9), T15)
    assert v["item_movementSpeed"] == pytest.approx(45.0) and v["boots_tier"] == 2.0
    # 3013 is later built into 3176 Forever Forward: the 3013 destroy at the purchase is a combine
    up = ev + [destroy(1_200_000, 3013), buy(1_200_000, 3176)]
    d = {}
    assert held(x15(up), diag=d) == [3176, 3340] and d["unowned_removal"] == 0
    d = {}
    assert held(x15([destroy(900_000, 3010)]), diag=d) == [3340] and d["unowned_ids"] == {3010: 1}  # not held


def test_r15_1_not_applied_to_a_viego_possession_swap():
    ev = [buy(1000, 3010), buy(2000, 1036), kill(895_000, 1), destroy(900_000, 3010), destroy(900_000, 1036)]
    ix = x15(ev, champion_by_pid={"1": 234})
    assert held(ix) == [1036, 3010, 3340]                                        # swap: nothing consumed


# ------------------------------------------------------------------ R15-2 / R15-3 trinket slot
def test_r15_2_start_trinket_and_fiddlesticks():
    ix = x15([], champion_by_pid={"1": 9, "2": 1})
    assert held(ix, 1, 0) == [3330] and held(ix, 2, 0) == [3340]
    assert all(held(ix, p, 0) == [3340] for p in range(3, 11))
    assert player_item_vector([3340], T15) == player_item_vector([], T15)      # a trinket changes no feature
    assert x15([]).grants[1] == [(0, 3340)]


def test_r15_3_trinket_purchase_replaces_the_held_trinket():
    d = {}
    assert held(x15([buy(9000, 3340)]), diag=d) == [3340] and d["r15_trinket_replaced"] == 1
    assert held(x15([buy(9000, 3364)])) == [3364]
    ev = [buy(9000, 3364), undo(12_000, 0, 3364, 0)]                             # undo of the free pick
    assert held(x15(ev)) == [3340]
    swap = [destroy(600_000, 3340), buy(600_000, 3364)]                          # ordinary swap
    d = {}
    assert held(x15(swap), diag=d) == [3364] and d["unowned_removal"] == 0 and d["r15_trinket_replaced"] == 0
    assert held(x15(swap + [undo(601_000, 0, 3364, 0)])) == [3340]
    assert held(x15([buy(9000, 3340), destroy(700_000, 3340), buy(700_000, 3363)])) == [3363]
    assert spent(x15(swap)) == 0.0


# ------------------------------------------------------------------ R15-4 rune grant (R15-5 removed)
def test_r15_4_biscuit_delivery():
    ev = [destroy(181_000, 2010), destroy(340_000, 2010)]
    ix = x15(ev, runes_by_pid=BISCUIT)
    assert held(ix, t=119_999) == [3340] and held(ix, t=120_000) == [2010, 3340]
    d = {}
    assert held(ix, t=359_999, diag=d) == [3340] and d["unowned_removal"] == 0
    assert held(ix) == [2010, 3340]                                             # the 6:00 biscuit is kept
    assert player_item_vector(ix.inventory(1, 10 ** 9), T15)["item_gold_owned"] == 50.0
    assert spent(ix) == 0.0
    none = x15(ev, runes_by_pid={"1": {"primary_rune_1": 8010}})
    d = {}
    assert held(none, diag=d) == [3340] and d["unowned_ids"] == {2010: 2}
    s = x15([sell(250_000, 2010)], runes_by_pid=BISCUIT)
    assert held(s, t=250_000) == [2010, 3340] and held(s) == [2010, 2010, 3340]
    assert spent(s) == pytest.approx(-T15[2010]["gold_sell"])


def test_r15_5_triple_tonic_is_removed():
    # author decision records/items_confirmations_20260925T134510Z.json: R15-5 removed; 15.x grants no elixir
    ev = [level(60_000, 2), level(150_000, 3), destroy(151_000, 2151), level(400_000, 6), level(700_000, 9)]
    ix = x15(ev, runes_by_pid=TONIC)
    assert ix.grants[1] == [(0, 3340)]
    d = {}
    assert held(ix, t=150_500) == [3340] and held(ix, diag=d) == [3340]
    assert d["unowned_removal"] == 1 and d["unowned_ids"] == {2151: 1}         # the destroy hits an unowned item
    assert held(x15(ev, runes_by_pid={"1": {"sub_rune_1": 8345}}), t=150_500) == [2010, 3340]
    # 16.x keeps its own Triple Tonic grant (R16-12)
    ix16 = ItemStateIndex(ev, load_item_table_v2("16.15"), patch="16.15", runes_by_pid=TONIC)
    assert 2152 in held(ix16)


# ------------------------------------------------------------------ R15-6 / R15-7 Eye of the Herald, Viego
def test_r15_6_eye_of_the_herald():
    ev = [destroy(600_000, 3340), buy(600_000, 3364), destroy(840_000, 3364), destroy(900_000, 3513)]
    d = {}
    assert held(x15(ev), t=840_000) == [3513]
    assert held(x15(ev), diag=d) == [3364] and d["r15_herald_swap"] == 1 and d["r15_herald_used"] == 1
    assert d["unowned_removal"] == 0
    assert held(x15(ev[:3])) == [3513]                                          # held at game end
    fid = [destroy(840_000, 3330), destroy(900_000, 3513)]
    assert held(x15(fid, champion_by_pid={"1": 9}), t=850_000) == [3513]
    assert held(x15(fid, champion_by_pid={"1": 9})) == [3330]
    # rules off: the trinket is lost and the 3513 destroy hits an unowned item
    d = {}
    assert held(x15(ev, rules_15x=False), diag=d) == [] and d["unowned_ids"] == {3340: 1, 3513: 1}


def test_r15_7_viego_herald_is_not_a_possession():
    ev = [buy(1000, 1036), kill(830_000, 1), destroy(840_000, 3340), destroy(900_000, 3513)]
    ix = x15(ev, champion_by_pid={"1": 234})
    assert held(ix, t=840_000) == [1036, 3513]
    assert held(ix) == [1036, 3340]
    swap = [kill(590_000, 1), destroy(600_000, 3340), buy(600_000, 3364)]    # a swap right after a takedown
    assert held(x15(swap, champion_by_pid={"1": 234})) == [3364]
    poss = [buy(1000, 1036), buy(1000, 2055), kill(895_000, 1), destroy(900_000, 1036), destroy(900_000, 2055)]
    assert held(x15(poss, champion_by_pid={"1": 234})) == [1036, 2055, 3340]  # possession swaps unchanged


# ------------------------------------------------------------------ gold neutrality and causality
def test_rules_change_no_gold():
    ev = [buy(9000, 3364), buy(20_000, 1055), buy(20_000, 2003), level(150_000, 3), destroy(181_000, 2010),
          destroy(400_000, 1001), buy(390_000, 1001), buy(400_000, 3010), destroy(840_000, 3364),
          destroy(900_000, 3513), destroy(950_000, 3010), sell(1_000_000, 2010), undo(1_000_500, 0, 2010, -20)]
    runes = {"1": {"sub_rune_1": 8345, "sub_rune_2": 8313}}
    for t in range(0, 1_100_001, 50_000):
        assert spent(x15(ev, runes_by_pid=runes), t=t) == pytest.approx(
            spent(x15(ev, runes_by_pid=runes, rules_15x=False), t=t)), t


def test_15x_state_uses_only_events_up_to_t():
    runes = {"1": {"sub_rune_1": 8345, "sub_rune_2": 8313}, "2": {"sub_rune_1": 8345}}
    ev = [buy(9000, 3364), level(150_000, 3), destroy(181_000, 2010), buy(400_000, 3010),
          destroy(840_000, 3364), destroy(900_000, 3513), destroy(950_000, 3010), level(400_000, 6),
          buy(8000, 3340, 2), destroy(600_000, 3340, 2), buy(600_000, 3363, 2), undo(601_000, 0, 3363, 0, 2),
          kill(830_000, 3), destroy(840_000, 3340, 3), destroy(900_000, 3513, 3)]
    champ = {"3": 234}
    full = x15(ev, runes_by_pid=runes, champion_by_pid=champ)
    for t in range(0, 1_000_001, 10_000):
        cut = x15([e for e in ev if e["timestamp"] <= t], runes_by_pid=runes, champion_by_pid=champ)
        for pid in (1, 2, 3):
            assert held(full, pid, t) == held(cut, pid, t), (pid, t)


# ------------------------------------------------------------------ rule file
def test_rule_file_lists_every_rule_with_evidence():
    spec = json.loads((ROOT / "config/game_rules/item_rules_15x.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in spec["rules"]] == [f"R15-{k}" for k in (1, 2, 3, 4, 6, 7)]     # R15-5 removed
    (rm,) = spec["removed_rules"]
    assert rm["id"] == "R15-5" and rm["removed_on"] == "2026-09-25"
    assert "items_confirmations_20260925T134510Z.json" in rm["decision_record"] and "8313" in rm["removed_rune_grant"]
    assert "8313" not in spec["rune_grants"]
    for r in spec["rules"]:
        assert r["rule"] and set(r["evidence"]) == {"15.14 rule-check", "KR 15.18 first sample"}
    assert R15["loose_transforms"] == {3010: 3013}
    assert R15["start_trinket"] == 3340 and R15["start_trinket_by_champion"] == {9: 3330} and R15["herald_eye"] == 3513
    assert R15["rune_grants_at_ms"] == {8345: (2010, (120_000, 240_000, 360_000))}
    assert R15["rune_grants_at_level"] == {}
    for patch in ("15.14", "15.18"):                                            # ids exist with the same meaning
        t = load_item_table_v2(patch)
        assert t[3013]["from"] == [3010] and "Trinket" in t[3513]["tags"] and "Trinket" in t[3330]["tags"]
        assert t[3330]["requiredChampion"] == "FiddleSticks" and 2010 in t and 2151 in t and 2152 in t


# ------------------------------------------------------------------ 15.14 identity with the rules off (50 cache matches)
@pytest.mark.slow
def test_15_14_identity_with_rules_off_equals_the_pre_15x_reference():
    import types
    sys.path.insert(0, str(ROOT / "scripts/exact_v4"))
    import ev4_s1x_items_16x as S
    from gameplay import item_state as IS
    if not S.PATCH_INDEX.exists() or not S.CACHE.exists():
        pytest.skip("15.14 cache not available")

    class Off(ItemStateIndex):
        @classmethod
        def from_pack(cls, pack, items_table=None, **kw):
            return super().from_pack(pack, items_table, rules_15x=False, **kw)

    digest, n = S.identity_digest(types.SimpleNamespace(ItemStateIndex=Off, player_item_vector=IS.player_item_vector))
    assert n == 14_350
    assert digest == S.REF_HASH_1514_NO_15X == "8ff6d349744d310a8f87a6b087d1490466d2d23ab16da44851b1cd21ca824d9f"
    assert S.REF_HASH_1514 != S.REF_HASH_1514_NO_15X


# ------------------------------------------------------------------ role inference: participant-0 World Atlas
def test_role_start_gold_ignores_the_participant_zero_world_atlas():
    from tests.test_exact_roles import SHUFFLED, make_pack, permute_pack, within_team_perm
    from gameplay.role_inference import infer_roles
    atlas = [{"type": "ITEM_PURCHASED", "timestamp": 0, "participantId": 0, "itemId": 3865}] * 2
    sup = [p for p, r in SHUFFLED.items() if r == "UTILITY"]
    adc = [p for p, r in SHUFFLED.items() if r == "BOTTOM"]
    purchases = [(p, 1055, 20_000) for p in adc] + [(p, 2003, 20_000) for p in sup]
    pack = make_pack(SHUFFLED, support=[], purchases=purchases, extra_events=atlas)
    ra = infer_roles(pack)
    assert ra.roles == SHUFFLED and not ra.ambiguous
    assert all(ra.evidence[p]["gold"] == pytest.approx(50.0) and ra.evidence[p]["low_start"] == 1.0 for p in sup)
    assert ra.evidence[5]["gold"] == pytest.approx(450.0) and ra.evidence[10]["gold"] == pytest.approx(0.0)
    for seed in (1, 2, 3):                                                      # equivariant: no participant order
        perm = within_team_perm(seed)
        rb = infer_roles(permute_pack(pack, perm))
        assert all(rb.evidence[perm[p]]["gold"] == ra.evidence[p]["gold"] for p in range(1, 11))
        assert all(rb.roles[perm[p]] == ra.roles[p] for p in range(1, 11))
