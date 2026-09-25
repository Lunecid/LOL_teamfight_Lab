"""v4-exact item state (gameplay/item_state.py): event replay, recipe rules, features, effect flags."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gameplay.item_state import (  # noqa: E402
    EFFECT_FLAGS, ITEM_STAT_KEYS, ITEM_VECTOR_NAMES, ItemStateIndex, UnknownItemRateError, boots_tier,
    check_unknown_rate, is_completed_item, item_vectors_at, load_effect_flags, load_item_table_v2,
    magical_footwear_grant_ms, player_item_vector, proxy_sources, purchase_price, replay_inventory,
    resolve_ids_by_name, shop_events_by_participant, transform_map, verify_effect_flags,
)

PATCHES = ["15.14", "15.15", "15.16", "15.18", "15.19", "15.20", "15.21", "15.22", "16.13", "16.14", "16.15"]
T15 = load_item_table_v2("15.14")


def buy(ts, iid, pid=1):
    return {"type": "ITEM_PURCHASED", "timestamp": ts, "participantId": pid, "itemId": iid}


def destroy(ts, iid, pid=1):
    return {"type": "ITEM_DESTROYED", "timestamp": ts, "participantId": pid, "itemId": iid}


def sell(ts, iid, pid=1):
    return {"type": "ITEM_SOLD", "timestamp": ts, "participantId": pid, "itemId": iid}


def undo(ts, before, after, gold, pid=1):
    return {"type": "ITEM_UNDO", "timestamp": ts, "participantId": pid, "beforeId": before, "afterId": after,
            "goldGain": gold}


def inv(events, t=10**9, exact=True, table=T15, **kw):
    shop = shop_events_by_participant(events, table)[1]
    return replay_inventory(shop, table, t, exact=exact, **kw)


# ------------------------------------------------------------------ tables and names
def test_stat_keys_match_ddragon_stat_map():
    idx = json.loads((ROOT / "config/game_rules/datadragon_v2/index.json").read_text(encoding="utf-8"))
    assert tuple(idx["stat_map"].values()) == ITEM_STAT_KEYS
    from scripts.fetch_datadragon_tables import ITEM_STAT_MAP
    assert tuple(ITEM_STAT_MAP.values()) == ITEM_STAT_KEYS
    assert len(ITEM_VECTOR_NAMES) == len(set(ITEM_VECTOR_NAMES)) == 22
    assert EFFECT_FLAGS == ("stasis", "revive", "cleanse", "shield", "lifeline")
    assert ITEM_VECTOR_NAMES[-2:] == ("flag_lifeline", "unknown_item_count")


@pytest.mark.parametrize("patch", PATCHES)
def test_table_drops_mode_copies_keeps_sr(patch):
    t = load_item_table_v2(patch)
    assert all(v["map_sr"] or k < 10_000 for k, v in t.items())
    assert not any(200_000 <= k < 230_000 for k in t)          # Arena 22xxxx
    assert not any(773_000 <= k < 774_000 for k in t)          # 16.15 mode variants
    assert 3513 in t and 3040 in t and 2421 in t               # held without being in store / on the map flag


# ------------------------------------------------------------------ legacy replay (moved body)
def test_evidence_state_method_calls_the_moved_function():
    from gameplay import evidence_state
    assert evidence_state.replay_inventory is replay_inventory
    ev = [buy(1000, 1036), buy(2000, 1036), destroy(3000, 1036), destroy(3000, 1036), buy(3000, 3134),
          buy(4000, 2003), undo(5000, 2003, 0, 50)]
    shop = shop_events_by_participant(ev, T15)[1]
    legacy = replay_inventory(shop, T15, 10_000, since_ms=0)
    assert legacy[0] == [3134]
    assert legacy[1] == pytest.approx(-(350 + 350 + T15[3134]["gold_base"] + 50) + 50)


# ------------------------------------------------------------------ event algebra
@pytest.mark.parametrize("exact", [False, True])
def test_purchase_and_sell(exact):
    items, gold, stats = inv([buy(1000, 1036), buy(2000, 1001), sell(3000, 1036)], exact=exact, since_ms=0)
    assert items == [1001]
    assert gold == pytest.approx(-350 - 300 + T15[1036]["gold_sell"])
    assert stats["movementSpeed"] == pytest.approx(25.0) and stats["attackDamage"] == pytest.approx(0.0)
    assert inv([buy(1000, 1036), buy(2000, 1001), sell(3000, 1036)], t=2500, exact=exact)[0] == [1036, 1001]


@pytest.mark.parametrize("exact", [False, True])
def test_undo_purchase_and_undo_sell(exact):
    assert inv([buy(1000, 1036), undo(1500, 1036, 0, 350)], exact=exact)[0] == []
    assert inv([buy(1000, 1036), sell(2000, 1036), undo(2500, 0, 1036, -245)], exact=exact)[0] == [1036]


@pytest.mark.parametrize("exact", [False, True])
def test_combine_destroys_components_at_same_ms(exact):
    ev = [buy(1000, 1036), buy(2000, 1036), destroy(3000, 1036), destroy(3000, 1036), buy(3000, 3134)]
    items, gold, stats = inv(ev, exact=exact, since_ms=0)
    assert items == [3134]
    assert gold == pytest.approx(-(350 + 350 + T15[3134]["gold_base"]))       # recipe cost, not total
    assert stats["attackDamage"] == pytest.approx(T15[3134]["stats"]["attackDamage"])
    assert inv(ev, t=2999, exact=exact)[0] == [1036, 1036]                       # nothing at 3000 is read


def test_undo_of_a_combine_restores_components_exact_only():
    ev = [buy(1000, 1052), destroy(2000, 1052), buy(2000, 3113), undo(2500, 3113, 0, 500)]
    assert inv(ev, exact=True)[0] == [1052]
    assert inv(ev, exact=False)[0] == []                                          # legacy loses the component


@pytest.mark.parametrize("exact", [False, True])
def test_consumable_used_and_unowned_destroy(exact):
    assert inv([buy(1000, 2003), buy(1100, 2003), destroy(5000, 2003)], exact=exact)[0] == [2003]
    d = {}
    assert inv([destroy(5000, 2010)], exact=True, diagnostics=d)[0] == []       # never held: counted, not guessed
    assert d["unowned_removal"] == 1 and d["unowned_ids"] == {2010: 1}


def test_consumable_bought_and_drunk_in_one_ms_is_not_held():
    ev = [destroy(9000, 2140), buy(9000, 2140)]                                  # timeline lists destroy first
    assert inv(ev, exact=True)[0] == []
    assert inv(ev, exact=False)[0] == [2140]


def test_transformations_without_purchase():
    assert transform_map(T15)[3003] == 3040 and transform_map(T15)[2420] == 2421
    assert inv([buy(1000, 3003), destroy(9000, 3003)])[0] == [3040]                 # Archangel's -> Seraph's
    armguard = [buy(1000, 2420), destroy(9000, 2420)]                              # stasis used -> Shattered
    assert inv(armguard)[0] == [2421]
    combine = [buy(1000, 2420), buy(2000, 1058), destroy(9000, 2420), destroy(9000, 1058), buy(9000, 3157)]
    assert inv(combine)[0] == [3157]                                               # a combine is not a transform


def test_support_quest_chain_and_proxy():
    assert 3867 in proxy_sources(T15) and 1001 not in proxy_sources(T15)
    chain = [destroy(400_000, 3865), destroy(800_000, 3866), destroy(900_000, 3867)]   # 3865 bought by pid 0
    assert inv(chain, t=500_000)[0] == [3866]
    assert inv(chain)[0] == [3867]                                                 # proxy for the 387x upgrade
    assert inv(chain + [buy(900_000, 3869)])[0] == [3869]


def test_viego_possession_is_not_consumption():
    ev = [buy(1000, 1036), buy(1000, 2055), destroy(5000, 1036), destroy(5000, 2055), destroy(9000, 3078),
          destroy(9000, 1029)]
    assert sorted(inv(ev, champion_id=234)[0]) == [1036, 2055]
    assert inv(ev, champion_id=1)[0] == []


def test_magical_footwear_grant():
    kills = [{"type": "CHAMPION_KILL", "timestamp": 200_000, "killerId": 1, "assistingParticipantIds": []},
             {"type": "CHAMPION_KILL", "timestamp": 300_000, "killerId": 6, "assistingParticipantIds": [1]}]
    runes = {"primary_rune_1": 8010, "sub_rune_1": 8304}
    assert magical_footwear_grant_ms(kills, 1, runes) == 720_000 - 2 * 45_000
    assert magical_footwear_grant_ms(kills, 2, runes) == 720_000
    assert magical_footwear_grant_ms(kills, 1, {"sub_rune_1": 8345}) is None
    many = [{"type": "CHAMPION_KILL", "timestamp": 10_000 * k, "killerId": 1} for k in range(1, 20)]
    assert magical_footwear_grant_ms(many, 1, runes) == 135_000                  # first t >= 720 s - 45 s * N(t)
    x = ItemStateIndex(kills + [buy(700_000, 1036)], T15, runes_by_pid={"1": runes})
    assert x.inventory(1, 629_999) == []
    assert x.inventory(1, 630_000) == [2422]
    assert x.vector(1, 700_000)["boots_tier"] == 1.0


# ------------------------------------------------------------------ features
def test_completed_rule_and_boots_tier():
    assert is_completed_item(3031, T15) and is_completed_item(3157, T15)
    assert not is_completed_item(1038, T15)
    assert not is_completed_item(3006, T15) and boots_tier(3006, T15) == 2
    assert boots_tier(1001, T15) == 1 and boots_tier(2422, T15) == 1 and boots_tier(3031, T15) == 0
    assert not is_completed_item(2003, T15) and not is_completed_item(2420, T15)


def test_player_item_vector_and_unknown_ids():
    v = player_item_vector([3157, 3006, 1038, 2003, 999_999], T15)
    assert tuple(v) == ITEM_VECTOR_NAMES
    assert v["unknown_item_count"] == 1.0
    assert v["item_abilityPower"] == pytest.approx(105.0) and v["item_armor"] == pytest.approx(50.0)
    assert v["item_attackDamage"] == pytest.approx(40.0)
    assert v["item_attackSpeed_pct"] == pytest.approx(0.25)
    assert v["item_gold_owned"] == pytest.approx(3250 + 1100 + 1300 + 50)
    assert v["completed_item_count"] == 1.0 and v["boots_tier"] == 2.0
    assert v["flag_stasis"] == 1.0 and v["flag_revive"] == v["flag_cleanse"] == v["flag_shield"] == 0.0
    assert v["flag_lifeline"] == 0.0
    assert player_item_vector([2421], T15)["flag_stasis"] == 0.0
    assert player_item_vector([3222], T15)["flag_cleanse"] == 0.0                 # Mikael's excluded
    assert player_item_vector([3140, 3026, 323190], T15)["flag_cleanse"] == 1.0


def test_all_ten_players_and_no_future_events():
    ev = [buy(1000, 1036, pid) for pid in range(1, 11)] + [buy(90_000, 3157, 3), destroy(95_000, 1036, 4)]
    full = item_vectors_at(ev, T15, 60_000)
    cut = item_vectors_at([e for e in ev if e["timestamp"] <= 60_000], T15, 60_000)
    assert set(full) == set(range(1, 11)) and full == cut
    assert full[3]["flag_stasis"] == 0.0
    assert item_vectors_at(ev, T15, 90_000)[3]["flag_stasis"] == 1.0


# ------------------------------------------------------------------ effect flags
LIFELINE_NAMES = ["Sterak's Gage", "Immortal Shieldbow", "Maw of Malmortius", "Seraph's Embrace"]
LIFELINE = {3053, 6673, 3156, 3040, 323040}


def test_effect_flag_sets_are_the_reviewer_mapping():
    flags = load_effect_flags()
    assert set(flags) == set(EFFECT_FLAGS)
    assert flags["stasis"] == {3157, 2420} and flags["revive"] == {3026}
    assert flags["cleanse"] == {3140, 3139} and flags["shield"] == {3190, 323190}
    assert flags["lifeline"] == LIFELINE
    every = set().union(*flags.values())
    assert not {2421, 3222, 3003, 3155} & every                                   # Archangel's / Hexdrinker: no lifeline yet
    assert not any(200_000 <= i < 230_000 or 773_000 <= i < 774_000 for i in every)


@pytest.mark.parametrize("patch", PATCHES)
def test_lifeline_is_a_per_patch_set_resolved_by_name(patch):
    spec = json.loads((ROOT / "config/game_rules/item_effect_flags.json").read_text(encoding="utf-8"))
    lf = spec["flags"]["lifeline"]
    assert "ids" not in lf and set(lf["ids_by_patch"]) == set(PATCHES) and lf["resolve_names"] == LIFELINE_NAMES
    table = load_item_table_v2(patch)
    res = resolve_ids_by_name(table, LIFELINE_NAMES)
    assert all(res[n] for n in LIFELINE_NAMES), res                                 # no name missing / renamed
    assert set().union(*map(set, res.values())) == set(lf["ids_by_patch"][patch]) == LIFELINE
    assert load_effect_flags(table, patch=patch)["lifeline"] == LIFELINE
    # Arena / mode copies carry the names but are not SR entries
    raw = json.loads((ROOT / f"config/game_rules/datadragon_v2/items_{patch}.json").read_text(encoding="utf-8"))["items"]
    copies = {int(k) for k, v in raw.items() if v.get("name") in LIFELINE_NAMES and not v.get("map_sr")}
    assert copies and not copies & LIFELINE and {223053, 226673, 223156, 223040} <= copies


def test_lifeline_flag_in_vectors_and_patch_selection():
    assert all(player_item_vector([i], T15)["flag_lifeline"] == 1.0 for i in LIFELINE)
    assert player_item_vector([3003, 3155, 6670], T15)["flag_lifeline"] == 0.0
    assert inv([buy(1000, 3003), destroy(9000, 3003)])[0] == [3040]               # Archangel's -> Seraph's
    x = ItemStateIndex([buy(1000, 3003, 2), destroy(9000, 3003, 2), buy(5000, 3053, 7)], T15, patch="15.14")
    assert x.vector(2, 8999)["flag_lifeline"] == 0.0 and x.vector(2, 9000)["flag_lifeline"] == 1.0
    assert x.vector(7, 4999)["flag_lifeline"] == 0.0 and x.vector(7, 5000)["flag_lifeline"] == 1.0
    assert x.flags["lifeline"] == LIFELINE
    with pytest.raises(KeyError):
        load_effect_flags(T15, patch="14.1")                                     # no id set for that patch
    pack = {"events": [buy(5000, 6673, 4)], "meta": {"patch": "15.14", "static_meta": {}}}
    y = ItemStateIndex.from_pack(pack)
    assert y.patch == "15.14" and y.vector(4, 5000)["flag_lifeline"] == 1.0


def test_check_unknown_rate():
    assert check_unknown_rate({}, 397_018) == 0.0
    assert check_unknown_rate({999_999: 5}, 1000) == pytest.approx(0.005)          # exactly 0.5 %: passes
    with pytest.raises(UnknownItemRateError, match="999999"):
        check_unknown_rate({999_999: 5, 888_888: 1}, 1000)                          # 0.6 %
    assert check_unknown_rate(4, 1000) == pytest.approx(0.004)                      # plain count
    with pytest.raises(UnknownItemRateError):
        check_unknown_rate(6, 1000)
    with pytest.raises(UnknownItemRateError):
        check_unknown_rate(2, 1000, max_rate=0.001)
    assert check_unknown_rate(0, 0) == 0.0
    with pytest.raises(UnknownItemRateError):
        check_unknown_rate({1: 1}, 0)
    with pytest.raises(ValueError):
        check_unknown_rate(-1, 10)
    with pytest.raises(ValueError):
        check_unknown_rate(11, 10)
    assert issubclass(UnknownItemRateError, ValueError)


def test_effect_flags_resolve_in_all_patches():
    res = verify_effect_flags(PATCHES)
    assert set(res) == set(PATCHES)
    assert all(v == {} for v in res.values()), {p: v for p, v in res.items() if v}
    for p in PATCHES:
        load_effect_flags(load_item_table_v2(p))                                  # raises when an id is missing
        load_effect_flags(load_item_table_v2(p), patch=p)


# ------------------------------------------------------------------ stage 1 completion fixes (B1-B3) and gold rules
def gold_spent(events, t=10**9, table=T15, **kw):
    """Net item spending of pid 1 (purchases at the charged price - sells - undo refunds)."""
    return -inv(events, t=t, table=table, since_ms=-1, **kw)[1]


def test_b1_undo_restores_only_the_destroyed_copies():
    assert T15[3133]["from"] == [1036, 2022, 1036]
    one = [buy(1000, 1036), buy(2000, 2022), destroy(3000, 1036), destroy(3000, 2022), buy(3000, 3133),
           undo(3500, 3133, 0, 450)]
    d = {}
    assert sorted(inv(one, diagnostics=d)[0]) == [1036, 2022]                    # not [1036, 1036, 2022]
    assert d["undo_restored"] == 2 and d["unowned_removal"] == 0
    assert inv(one, t=3000)[0] == [3133]
    # price: one Long Sword owned, so the shop also charges the missing one: 1050 - 350 - 250
    assert gold_spent(one, t=3000) == pytest.approx(350 + 250 + 450)
    assert gold_spent(one) == pytest.approx(350 + 250)
    two = [buy(1000, 1036), buy(1500, 1036), buy(2000, 2022), destroy(3000, 1036), destroy(3000, 1036),
           destroy(3000, 2022), buy(3000, 3133), undo(3500, 3133, 0, 100)]
    assert sorted(inv(two)[0]) == [1036, 1036, 2022]
    assert gold_spent(two, t=3000) == pytest.approx(350 + 350 + 250 + T15[3133]["gold_base"])


def test_price_counts_components_inside_components():
    # Youmuu = [3134 Serrated Dirk, 6690, 1036]; the one Long Sword held is consumed
    ev = [buy(1000, 1036), destroy(5000, 1036), buy(5000, 3142)]
    d = {}
    assert inv(ev, diagnostics=d)[0] == [3142] and d["unowned_removal"] == 0
    assert gold_spent(ev) == pytest.approx(350 + 2800 - 350)
    assert gold_spent([buy(5000, 3142)]) == pytest.approx(2800)                   # nothing held: gold_total


@pytest.mark.parametrize("champion", [234, 1])                                     # Viego and a normal champion
def test_b2_magical_footwear_upgrade_is_a_combine(champion):
    grant = [(600_000, 2422)]
    ev = [buy(1000, 1029), destroy(700_000, 2422), destroy(700_000, 1029), buy(700_000, 3047)]
    d = {}
    assert inv(ev, grants=grant, champion_id=champion, diagnostics=d)[0] == [3047]
    assert d["viego_ignored"] == 0 and d["unowned_removal"] == 0 and d["substituted"] == 1
    assert gold_spent(ev, grants=grant, champion_id=champion) == pytest.approx(300 + T15[3047]["gold_base"])
    only = [destroy(700_000, 2422), buy(700_000, 3009)]                           # 3009 = [1001]
    assert inv(only, grants=grant, champion_id=champion)[0] == [3009]
    assert gold_spent(only, grants=grant, champion_id=champion) == pytest.approx(T15[3009]["gold_base"])
    part = [destroy(700_000, 2422), buy(700_000, 3047)]                           # no Cloth Armor held
    assert gold_spent(part, grants=grant, champion_id=champion) == pytest.approx(1200 - 300)
    assert 3010 not in T15[2422]["into"]                                           # still counts as Boots
    soles = [destroy(700_000, 2422), buy(700_000, 3010)]
    assert inv(soles, grants=grant, champion_id=champion)[0] == [3010]
    assert gold_spent(soles, grants=grant, champion_id=champion) == pytest.approx(900 - 300)
    t3 = [destroy(700_000, 2422), buy(700_000, 3170)]                              # tier 3 = [3009]
    assert inv(t3, grants=grant, champion_id=champion)[0] == [3170]
    assert gold_spent(t3, grants=grant, champion_id=champion) == pytest.approx(1500 - 300)
    assert inv(t3 + [undo(701_000, 3170, 0, 1200)], grants=grant, champion_id=champion)[0] == [2422]


def test_shattered_armguard_builds_zhonyas():
    ev = [buy(1000, 2420), destroy(9000, 2420), buy(20_000, 1058), destroy(30_000, 2421), destroy(30_000, 1058),
          buy(30_000, 3157)]
    d = {}
    assert inv(ev, diagnostics=d)[0] == [3157] and d["unowned_removal"] == 0
    assert gold_spent(ev) == pytest.approx(1600 + 1200 + 3250 - 1600 - 1200)


def test_b3_grant_no_later_than_the_first_2422_event():
    kills = [{"type": "CHAMPION_KILL", "timestamp": 200_000, "killerId": 1, "assistingParticipantIds": []}]
    runes = {"sub_rune_1": 8304}
    ev = kills + [destroy(466_985, 2422), buy(466_985, 3020)]
    assert magical_footwear_grant_ms(kills, 1, runes) == 720_000 - 45_000
    assert magical_footwear_grant_ms(ev, 1, runes) == 466_985
    assert magical_footwear_grant_ms(ev, 2, runes) == 720_000                     # another player's events
    x = ItemStateIndex(ev, T15, runes_by_pid={"1": runes})
    assert x.inventory(1, 466_984) == []                                           # nothing before the event
    d = {}
    assert x.inventory(1, 466_985, diagnostics=d) == [3020] and d["unowned_removal"] == 0
    assert -x.replay(1, 10**9, since_ms=-1)[1] == pytest.approx(T15[3020]["gold_base"])
    cut = [e for e in ev if e["timestamp"] <= 466_000]                            # causal: events <= t only
    assert magical_footwear_grant_ms(cut, 1, runes) == 675_000
    y = ItemStateIndex(kills + [sell(500_000, 2422)], T15, runes_by_pid={"1": runes})
    assert y.inventory(1, 500_000) == [] and y.inventory(1, 499_999) == []       # granted at the sell, then sold
    assert -y.replay(1, 500_000, since_ms=-1)[1] == pytest.approx(-T15[2422]["gold_sell"])


def test_world_atlas_bought_by_participant_zero_goes_to_the_support_slots():
    atlas = [{"type": "ITEM_PURCHASED", "timestamp": 0, "participantId": 0, "itemId": 3865}] * 2
    ev = atlas + [destroy(400_000, 3865, 5), buy(60_000, 1054, 10)]
    x = ItemStateIndex(ev, T15)
    assert x.inventory(5, 0) == [3865] and x.inventory(10, 0) == [3865]
    assert all(x.inventory(p, 0) == [] for p in (1, 2, 3, 4, 6, 7, 8, 9))
    assert x.inventory(5, 400_000) == [3866]
    assert -x.replay(10, 60_000, since_ms=-1)[1] == pytest.approx(400 + 450)
    assert ItemStateIndex(ev, T15, exact=False).inventory(5, 0) == []            # legacy mode unchanged
    assert shop_events_by_participant(ev, T15)[5] == [(400_000, "destroy", 3865, 0.0, 0, 0)]


def test_selling_the_upgrade_of_a_proxy_sells_the_proxy():
    chain = [destroy(400_000, 3865), destroy(800_000, 3866), destroy(900_000, 3867)]
    ev = chain + [sell(1_700_000, 3870), buy(1_701_000, 3876)]
    d = {}
    assert inv(ev, diagnostics=d)[0] == [3876] and d["unowned_ids"] == {3865: 1}       # 3865 held by pid 0 here
    assert gold_spent(ev) == pytest.approx(-T15[3870]["gold_sell"] + 400)          # bought again at full price
    assert gold_spent(chain + [buy(1_000_000, 3869)]) == pytest.approx(0.0)       # upgrade of a held proxy: free


def test_viego_possession_needs_a_recent_takedown():
    ev = [buy(1000, 1102), buy(2000, 1036), destroy(900_000, 1102)]               # lone jungle-pet destroy
    assert sorted(inv(ev, champion_id=234)[0]) == [1036, 1102]                    # no takedown info: a swap
    assert inv(ev, champion_id=234, takedowns_ms=[100_000])[0] == [1036]          # none within 30 s: consumed
    assert sorted(inv(ev, champion_id=234, takedowns_ms=[880_000])[0]) == [1036, 1102]
    assert inv(ev, champion_id=234, takedowns_ms=[910_000])[0] == [1036]          # a later takedown is not used
    kills = [{"type": "CHAMPION_KILL", "timestamp": 895_000, "killerId": 6, "assistingParticipantIds": [1]}]
    x = ItemStateIndex(ev + kills, T15, champion_by_pid={"1": 234})
    assert x.takedowns[1] == [895_000] and x.takedowns[2] is None
    assert sorted(x.inventory(1, 10**9)) == [1036, 1102]


def test_undo_of_a_free_purchase_is_logged_as_after_id():
    # support-quest upgrade chosen, undone (beforeId 0, afterId 3876, goldGain 0), chosen again
    chain = [destroy(400_000, 3865), destroy(800_000, 3866)]
    ev = chain + [destroy(821_076, 3867), buy(821_076, 3876), undo(828_858, 0, 3876, 0),
                  destroy(833_467, 3867), buy(833_467, 3876)]
    d = {}
    assert inv(ev, t=828_858)[0] == [3867]
    assert inv(ev, diagnostics=d)[0] == [3876] and d["unowned_ids"] == {3865: 1}
    # trinket swap undone: the old trinket comes back, no second copy of the new one
    sw = [buy(1000, 3340), destroy(600_000, 3340), buy(600_000, 3364), undo(601_000, 0, 3364, 0)]
    assert inv(sw)[0] == [3340]
    # an undo of a sale (the item is not held) still re-adds it
    assert inv([buy(1000, 3340), sell(2000, 3340), undo(2500, 0, 3340, 0)])[0] == [3340]


def test_purchase_price_rule():
    assert purchase_price(3133, [1036, 2022, 1036], T15) == T15[3133]["gold_base"]
    assert purchase_price(3133, [1036], T15) == pytest.approx(1050 - 350)
    assert purchase_price(3133, [], T15) == T15[3133]["gold_total"]
    for iid, it in T15.items():                                                    # gold_base = total - components
        if it.get("from") and all(c in T15 for c in it["from"]):
            assert purchase_price(iid, list(it["from"]), T15) == pytest.approx(
                it["gold_total"] - sum(T15[c]["gold_total"] for c in it["from"]))
