"""16.x item rules (v4-exact, task 16X): evidence, 16.15 end-of-game validation, 15.14 identity.  Single process.

The rules live in config/game_rules/item_rules_16x.json and gameplay/item_state.py (_R16Context, ItemStateIndex);
they apply only when the patch major is >= 16.  Outputs go to <OUT> (stage1/item_state_16x/).

  evidence  16.15 KR raw, all 553 matches (D:/LOL_Project/fusion_2615/raw/kr).  Reads, per match, the timeline
            events (GAME_END is skipped; no outcome field is read) and, from the detail, only participantId,
            teamPosition, championId, the pre-game summoner spell ids, the rune selections, item0-6 and
            roleBoundItem.  Counts the event patterns that motivated each rule (event types, quest-item destroys and
            what else the player's events hold at that ms, Recall items and elite kills, trinket / Eye of the Herald,
            support ward-slot groups, blind undos, rune grants) -> rules_evidence_16.15.json.
  validate  the same 553 matches through the cache pipeline's pack (ev4_s1_validate.build_1615_pack), replayed with
            ItemStateIndex.from_pack at the GAME_END timestamp, 'before' = rules_16x=False (the replay after the
            B1-B3 FIX, no 16.x rule), 'after' = the 16.x rules.  Per player (5,530):
              primary (99 % rule)  set of non-zero item0-5 + roleBoundItem (non-trinket) == set of the replay's
                                   non-trinket ids (exact-set agreement);
              item6                replay trinket set == {item6} (both empty when item6 == 0);
              also: multiset, item0-5 only (the earlier headline convention), 'rbq' (role-quest ids 1200-1222
              removed from both sides), per role, and a diagnostic where the replay's 3867 Bounty of Worlds (the
              documented proxy of its unobserved upgrade, item_state rule 3) counts as matching the detail's upgrade.
            Residual causes of every 'after' mismatch are listed.  Slot feasibility: normal (item0-5) slots the
            replayed end inventory needs, role-slot content excluded (<= 6 expected).  Causality: for 60 matches x 8
            random times, the index built from events <= t gives the same inventory at t as the full index.
            -> inventory_end_of_game_16.15_16x.json
  identity  50 15.14 cache matches (patch index == '15.14' only; sorted, random.Random(2026092516).shuffle, first
            50): inventories, gold deltas, stat deltas and item vectors of all 10 players at every minute and at the
            last event; sha256 compared with the value computed by the pre-16X module (REF_HASH_1514).
            -> identity_15.14.json

Usage: python scripts/exact_v4/ev4_s1x_items_16x.py [evidence|validate|identity|all]
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/exact_v4"))

from gameplay.item_state import (  # noqa: E402
    ItemStateIndex, is_boots, is_consumable, is_trinket, load_item_table_v2, load_rules_16x,
)

OUT = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925/stage1/item_state_16x")
RAW = Path("D:/LOL_Project/fusion_2615/raw/kr")
CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
PATCH_INDEX = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json")
PASS_RATE = 0.99
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
ROLE_QUEST_IDS = frozenset(range(1200, 1223))
SUPPORT_UPGRADES = frozenset({3869, 3870, 3871, 3876, 3877})
TRINKET_IDS = frozenset({3330, 3340, 3363, 3364})
# identity reference: identity_digest() below.  Old value 8ff6d349744d310a8f87a6b087d1490466d2d23ab16da44851b1cd21ca824d9f
# = gameplay/item_state.py as it was before task 16X (file sha256 ce5a341b..., after FIX B1-B3); the current module still
# reproduces it with the 15.x rules off (REF_HASH_1514_NO_15X, tests/test_exact_items_15x.py).  New value (task RULES15,
# 2026-09-25): the 15.x rules of config/game_rules/item_rules_15x.json (R15-1 .. R15-7) change 15.14 replays on purpose
# (0-ms trinket, rune grants, 3010 -> 3013, Eye of the Herald); stage1/item_state_15x/identity_15.14.json lists what changed.
# That RULES15 value was 0ca245e108e9b64855f661d1ec53566f036523f621daccccaf2b7d0c28f70138.  Current value (2026-09-25):
# R15-5 (Triple Tonic 2151 / 2152 grants) removed by author decision records/items_confirmations_20260925T134510Z.json
# (item_rules_15x.json 'removed_rules'); no other rule changed; rules-off digest still REF_HASH_1514_NO_15X;
# stage1/validation/strict_noR155/identity_15.14.json.  0ca245e1... -> e7cb8e51...
REF_HASH_1514 = "e7cb8e51d5874e7355a06e5bdc20ed510503ad75316905c5639df452f232e377"
REF_HASH_1514_NO_15X = "8ff6d349744d310a8f87a6b087d1490466d2d23ab16da44851b1cd21ca824d9f"
IDENTITY_SEED, IDENTITY_N = 2026092516, 50


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _ids_1615():
    return sorted(p.stem for p in (RAW / "detail").glob("*.json") if (RAW / "timeline" / p.name).exists())


def _detail_items(detail):
    """Only the fields this task may read: slot, position, champion, pre-game spells / runes, items."""
    out = {}
    for p in detail["info"]["participants"]:
        runes = [int(s["perk"]) for st in (p.get("perks") or {}).get("styles", []) for s in st.get("selections", [])]
        out[int(p["participantId"])] = {
            "pos": p.get("teamPosition"), "champion": int(p.get("championId", 0) or 0),
            "champion_name": p.get("championName"),
            "spells": (int(p.get("summoner1Id", 0) or 0), int(p.get("summoner2Id", 0) or 0)), "runes": runes,
            "items": [int(p.get(f"item{k}", 0) or 0) for k in range(6)], "item6": int(p.get("item6", 0) or 0),
            "rbi": int(p.get("roleBoundItem", 0) or 0)}
    return out


def _load_raw(mid):
    detail = json.loads((RAW / "detail" / f"{mid}.json").read_text(encoding="utf-8"))
    tl = json.loads((RAW / "timeline" / f"{mid}.json").read_text(encoding="utf-8"))
    types = Counter()
    events = []
    for f in tl["info"]["frames"]:
        for e in f["events"]:
            types[e["type"]] += 1
            if e["type"] != "GAME_END":                     # carries the winning team: never kept
                events.append(e)
    return _detail_items(detail), events, types


# ============================================================================================ evidence
def run_evidence():
    t0 = time.time()
    table = load_item_table_v2("16.15")
    name = lambda i: table.get(int(i), {}).get("name", "?")                              # noqa: E731
    boots = lambda i: "boots" if is_boots(table.get(int(i), {})) else str(i)             # noqa: E731
    ids = _ids_1615()
    E = defaultdict(Counter)
    for mid in ids:
        parts, events, types = _load_raw(mid)
        E["event_types"].update(types)
        elite = [(int(e["timestamp"]), e.get("monsterType"), e.get("killerTeamId")) for e in events
                 if e.get("type") == "ELITE_MONSTER_KILL"]
        herald_kills = [t for t, m, _ in elite if m == "RIFTHERALD"]
        item_ev = [e for e in events if e.get("type", "").startswith("ITEM_")]
        by = defaultdict(list)
        for e in item_ev:
            by[(int(e.get("participantId", 0) or 0), int(e["timestamp"]))].append(e)
        lvl_first = defaultdict(dict)
        for e in events:
            if e.get("type") == "LEVEL_UP":
                lvl_first[int(e["participantId"])].setdefault(int(e["level"]), int(e["timestamp"]))
        for pid, p in parts.items():
            slot_role = ROLES[(pid - 1) % 5]
            E["slot_role_equals_teamPosition"][str(p["pos"] == slot_role)] += 1
            E["roleBoundItem_by_position"][f"{p['pos']} {p['rbi']} {name(p['rbi'])}"] += 1
            if p["pos"] == "TOP":
                tp = 12 in p["spells"]
                q = [int(e["itemId"]) for e in item_ev if int(e.get("participantId", 0) or 0) == pid
                     and e["type"] == "ITEM_DESTROYED" and int(e.get("itemId", 0) or 0) in (1200, 1222)]
                E["top_quest_by_teleport"][f"teleport={tp} destroyed={q[:1]} roleBoundItem={p['rbi']}"] += 1
            mine = [e for e in item_ev if int(e.get("participantId", 0) or 0) == pid]
            # first quest-chain destroy vs slot role
            qfirst = next((int(e["itemId"]) for e in mine if e["type"] == "ITEM_DESTROYED"
                           and int(e.get("itemId", 0) or 0) in (1200, 1222, 1201, 1202, 1203, 1204)), None)
            qrole = {1200: "TOP", 1222: "TOP", 1201: "MIDDLE", 1202: "BOTTOM", 1203: "UTILITY", 1204: "JUNGLE"}
            E["first_quest_destroy_matches_slot_role"][str(None if qfirst is None else qrole[qfirst] == slot_role)] += 1
            # trinkets
            tr = [e for e in mine if int(e.get("itemId", 0) or 0) in TRINKET_IDS]
            if not tr:
                E["trinket_players_without_trinket_event_by_item6"][str(p["item6"])] += 1
            elif tr[0]["type"] == "ITEM_DESTROYED":
                E["trinket_first_event_is_destroy"]["n"] += 1
            if p["item6"] == 3330:
                E["item6_3330_by_champion"][p["champion_name"]] += 1
            # runes
            if 8345 in p["runes"]:
                bis = sorted(int(e["timestamp"]) for e in mine if int(e.get("itemId", 0) or 0) == 2010
                             and e["type"] in ("ITEM_DESTROYED", "ITEM_SOLD"))
                ok = all(t >= 120_000 * (k + 1) for k, t in enumerate(bis))
                E["biscuit_delivery_players_by_events"][f"n={len(bis)} kth_at_or_after_k*2min={ok}"] += 1
            for iid in (2151, 2152, 2150, 2010):
                has = any(int(e.get("itemId", 0) or 0) == iid for e in mine)
                if has:
                    E["rune_item_event_players"][f"{iid} rune8313={8313 in p['runes']} rune8345={8345 in p['runes']}"] += 1
            if 8313 in p["runes"]:
                for lv, iid in ((3, 2151), (6, 2152), (9, 2150)):
                    d = [int(e["timestamp"]) for e in mine if int(e.get("itemId", 0) or 0) == iid
                         and e["type"] == "ITEM_DESTROYED"]
                    if lv in lvl_first[pid]:
                        E["triple_tonic_reached_level"][f"level {lv} -> {iid} event={bool(d)}"] += 1
                        if d:
                            E["triple_tonic_destroy_after_levelup"][f"{iid} >=0"] += int(d[0] >= lvl_first[pid][lv])
        for (pid, ts), g in by.items():
            if pid == 0:
                continue
            for e0 in g:
                iid = int(e0.get("itemId", 0) or 0)
                if e0["type"] == "ITEM_DESTROYED" and (1200 <= iid <= 1222 or iid in (2001, 2002)):
                    others = sorted(f"{e['type'][5:8]} {boots(e.get('itemId', 0) or 0)}" for e in g if e is not e0)
                    key = " + ".join(others) if len(others) <= 3 else f"{len(others)} other destroys (Viego swap)"
                    E[f"destroy_{iid}_{name(iid)}_with"][key or "(alone)"] += 1
                if e0["type"] == "ITEM_DESTROYED" and iid in TRINKET_IDS and not any(
                        e["type"] == "ITEM_PURCHASED" and int(e.get("itemId", 0) or 0) in TRINKET_IDS for e in g):
                    w2001 = any(int(e.get("itemId", 0) or 0) in (2001, 2002) and e["type"] == "ITEM_DESTROYED"
                                for e in g)
                    later = any(int(e.get("itemId", 0) or 0) == 3513 and e["type"] == "ITEM_DESTROYED"
                                and int(e["timestamp"]) >= ts and int(e.get("participantId", 0) or 0) == pid
                                for e in item_ev)
                    end3513 = parts[pid]["item6"] == 3513
                    hk = any(0 <= ts - h <= 5000 for h in herald_kills)
                    E["loose_trinket_destroys"][f"with_recall_destroy={w2001} later_3513_destroy={later} "
                                                f"end_item6_3513={end3513} herald_kill_within_5s={hk}"] += 1
                if e0["type"] == "ITEM_DESTROYED" and iid in (2001, 2002):
                    team = 100 if pid <= 5 else 200
                    near = [m for t, m, k in elite if 0 <= ts - t <= 300_000 and k == team]
                    E["recall_item_destroy_last_own_elite_kill_within_5min"][f"{iid} {near[-1] if near else None}"] += 1
                if e0["type"] == "ITEM_UNDO" and not int(e0.get("beforeId", 0) or 0) \
                        and not int(e0.get("afterId", 0) or 0):
                    prev = [e for e in item_ev if int(e.get("participantId", 0) or 0) == pid
                            and e["type"] == "ITEM_PURCHASED" and int(e["timestamp"]) <= ts]
                    last = int(prev[-1]["itemId"]) if prev else 0
                    E["undo_before0_after0_last_purchase"][f"{parts[pid]['pos']} {last} {name(last)}"] += 1
            sig = [(e["type"][5:8], int(e.get("itemId", 0) or 0)) for e in g]
            if any(i in (1203, 1208) and t == "DES" for t, i in sig):
                E["support_group_at_1203_1208_destroy"][" + ".join(f"{t} {i}" for t, i in sig
                                                                     if i in (2055, 1203, 1208, 3866))] += 1
            dboots = [i for t, i in sig if t == "DES" and is_boots(table.get(i, {}))]
            if parts[pid]["pos"] == "MIDDLE" and any(t == "DES" and i == 1201 for t, i in sig):
                E["mid_quest_destroy_boots_destroyed_same_ms"][",".join(map(str, dboots)) or "none"] += 1
            if parts[pid]["pos"] == "MIDDLE":
                for t, i in sig:
                    if t == "PUR" and ("DES", i) in sig and is_boots(table.get(i, {})):
                        E["mid_destroy_X_and_buy_X_same_ms_boots"][f"{i} {name(i)}"] += 1
        # end-of-game tier-3 boots of mids (detail)
        for pid, p in parts.items():
            if p["pos"] == "MIDDLE":
                t3 = [i for i in p["items"] + [p["rbi"]] if 3168 <= i <= 3176]
                E["mid_end_tier3_boots_in_detail"][str(bool(t3))] += 1
    out = {"check": "16.15 event patterns behind the 16.x item rules", "data": str(RAW), "n_matches": len(ids),
           "fields_read": "timeline events except GAME_END; detail participantId, teamPosition, championId, "
                          "championName, summoner1Id/2Id, perks selections, item0-6, roleBoundItem",
           "counts": {k: dict(v.most_common()) for k, v in E.items()},
           "script": "scripts/exact_v4/ev4_s1x_items_16x.py evidence", "elapsed_s": round(time.time() - t0, 1)}
    _dump("rules_evidence_16.15.json", out)
    return {"n_matches": len(ids), "keys": sorted(E)}


# ============================================================================================ validate
def normal_slots(inv, table, role):
    """item0-5 slots the replayed inventory needs: one per copy, one per id for stackables (consumables, 2010
    biscuits); role-slot content excluded (role-quest ids; the bot's boots once 1202 / 1207 are gone; the support's
    control wards)."""
    c = Counter(int(i) for i in inv if not is_trinket(table.get(int(i), {})) and int(i) not in ROLE_QUEST_IDS)
    role_slot = None
    if role == "BOTTOM" and 1202 not in inv and 1207 not in inv:
        role_slot = next((i for i in c if is_boots(table.get(i, {}))), None)
    if role == "UTILITY" and 2055 in c:
        role_slot = 2055
    n = 0
    for i, k in c.items():
        m = 1 if (is_consumable(table.get(i, {})) or i == 2010) else k
        n += max(0, m - (1 if i == role_slot else 0))
    return n


def _classify(miss, extra, champion_name):
    miss, extra = set(miss), set(extra)
    if miss and miss <= SUPPORT_UPGRADES and extra == {3867}:
        return "support_upgrade_unobserved_3867_proxy"
    cats = []
    if champion_name == "Viego":
        cats.append("viego_possession")
    if (miss & SUPPORT_UPGRADES) and 3867 in extra:
        cats.append("support_upgrade_unobserved_3867_proxy")
    rest_m, rest_x = miss - SUPPORT_UPGRADES, extra - {3867}
    if 2150 in rest_m:
        cats.append("elixir_of_skill_held_no_event")
    if 2055 in rest_m | rest_x or 1208 in rest_m:
        cats.append("control_ward_or_ward_slot")
    if not cats or (rest_m - {2150, 2055, 1208}) or (rest_x - {2055}):
        cats.append("other")
    return " + ".join(sorted(set(cats)))


def run_validate():
    from ev4_s1_validate import build_1615_pack
    t0 = time.time()
    ids = _ids_1615()
    table = load_item_table_v2("16.15")
    names = {k: v.get("name", "") for k, v in table.items()}
    V = {v: {"c": Counter(), "role": defaultdict(Counter), "missing": Counter(), "extra": Counter(),
             "diag": Counter(), "unowned": Counter(), "item6_pairs": Counter()} for v in ("before", "after")}
    residual, cause_after = [], Counter()
    slot_hist, slot_over = Counter(), []
    rng = random.Random(20260927)
    caus = Counter()
    caus_examples = []
    caus_ids = set(rng.sample(ids, 60))
    for mid in ids:
        pack, detail = build_1615_pack(mid)
        parts = _detail_items(detail)
        assert pack["meta"]["patch"] == "16.15"
        ends = [int(e["timestamp"]) for e in pack["events"] if e.get("type") == "GAME_END"]
        t_end = max(ends) if ends else max(int(e["timestamp"]) for e in pack["events"])
        idx = {"before": ItemStateIndex.from_pack(pack, table, rules_16x=False),
               "after": ItemStateIndex.from_pack(pack, table)}
        assert idx["after"].rules16 is not None and idx["before"].rules16 is None
        for pid, p in parts.items():
            det_main = Counter(i for i in p["items"] if i and not is_trinket(table.get(i, {})))
            rb = p["rbi"]
            det_rb = det_main + (Counter([rb]) if rb and not is_trinket(table.get(rb, {})) else Counter())
            det_tr = {p["item6"]} if p["item6"] else set()
            role = ROLES[(pid - 1) % 5]
            for vn, ix in idx.items():
                W = V[vn]
                diag = {}
                inv = ix.inventory(pid, t_end, diagnostics=diag)
                for k, v in diag.items():
                    if k != "unowned_ids":
                        W["diag"][k] += v
                for k, v in diag.get("unowned_ids", {}).items():
                    W["unowned"][int(k)] += v
                rep_main = Counter(int(i) for i in inv if not is_trinket(table.get(int(i), {})))
                rep_tr = {int(i) for i in inv if is_trinket(table.get(int(i), {}))}
                ok = set(rep_main) == set(det_rb)
                c = W["c"]
                c["players"] += 1
                c["set_rb"] += ok
                c["multiset_rb"] += rep_main == det_rb
                c["set_item0_5_only"] += set(rep_main) == set(det_main)
                c["set_rbq"] += ({i for i in rep_main if i not in ROLE_QUEST_IDS}
                                 == {i for i in det_rb if i not in ROLE_QUEST_IDS})
                c["item6"] += rep_tr == det_tr
                if rep_tr != det_tr:
                    W["item6_pairs"][(p["item6"], tuple(sorted(rep_tr)))] += 1
                W["role"][role]["players"] += 1
                W["role"][role]["set_rb"] += ok
                miss, extra = set(det_rb) - set(rep_main), set(rep_main) - set(det_rb)
                proxy_ok = ok or (bool(miss) and miss <= SUPPORT_UPGRADES and extra == {3867}
                                  and len(miss) == 1)
                c["set_rb_proxy_counted"] += proxy_ok
                if not ok:
                    for i in miss:
                        W["missing"][i] += 1
                    for i in extra:
                        W["extra"][i] += 1
                if vn == "after":
                    ns = normal_slots(inv, table, role)
                    slot_hist[min(ns, 8)] += 1
                    if ns > 6 and len(slot_over) < 40:
                        slot_over.append({"match": mid, "pid": pid, "champion": p["champion_name"], "role": role,
                                          "normal_slots": ns, "replay": sorted(inv)})
                    if not ok:
                        cause = _classify(miss, extra, p["champion_name"])
                        cause_after[cause] += 1
                        if cause != "support_upgrade_unobserved_3867_proxy":
                            residual.append({"match": mid, "pid": pid, "champion": p["champion_name"],
                                             "role": role, "cause": cause,
                                             "missing_in_replay": [[i, names.get(i, "?")] for i in sorted(miss)],
                                             "extra_in_replay": [[i, names.get(i, "?")] for i in sorted(extra)],
                                             "diagnostics": {k: v for k, v in diag.items() if v}})
        if mid in caus_ids:
            ev = pack["events"]
            last = max(int(e["timestamp"]) for e in ev)
            for t in sorted(rng.randint(1, last) for _ in range(8)):
                cut = dict(pack, events=[e for e in ev if int(e["timestamp"]) <= t])
                a, b = idx["after"], ItemStateIndex.from_pack(cut, table)
                for pid in range(1, 11):
                    same = sorted(a.inventory(pid, t)) == sorted(b.inventory(pid, t))
                    caus["player_times"] += 1
                    caus["identical"] += same
                    if not same and len(caus_examples) < 10:
                        caus_examples.append({"match": mid, "pid": pid, "t": t,
                                              "full": sorted(a.inventory(pid, t)), "cut": sorted(b.inventory(pid, t))})
    named = lambda cnt, k=40: [[int(i), names.get(int(i), "?"), v] for i, v in cnt.most_common(k)]  # noqa: E731
    res = {}
    for vn, W in V.items():
        n = W["c"]["players"]
        res[vn] = {
            "n_players": n,
            "agreement_exact_set_item0_5_plus_roleBoundItem": W["c"]["set_rb"] / n,
            "pass_99pct_rule": W["c"]["set_rb"] / n >= PASS_RATE,
            "agreement_exact_multiset_item0_5_plus_roleBoundItem": W["c"]["multiset_rb"] / n,
            "agreement_exact_set_item0_5_only": W["c"]["set_item0_5_only"] / n,
            "agreement_exact_set_role_quest_ids_removed_rbq": W["c"]["set_rbq"] / n,
            "agreement_item6_trinket": W["c"]["item6"] / n,
            "diagnostic_agreement_3867_proxy_counted_as_its_upgrade": W["c"]["set_rb_proxy_counted"] / n,
            "agreement_by_role": {r: W["role"][r]["set_rb"] / W["role"][r]["players"] for r in ROLES},
            "mismatch_missing_in_replay_by_id": named(W["missing"]),
            "mismatch_extra_in_replay_by_id": named(W["extra"]),
            "item6_mismatch_pairs_detail_vs_replay": [[k[0], names.get(k[0], "?"), list(k[1]), v]
                                                      for k, v in W["item6_pairs"].most_common(15)],
            "replay_diagnostics_totals": dict(W["diag"]),
            "unowned_removal_ids": named(W["unowned"], 25),
        }
    n = V["after"]["c"]["players"]
    out = {
        "check": "16.15 end-of-game inventory: replay at GAME_END vs detail item0-5 + roleBoundItem (item6 apart)",
        "data": str(RAW), "n_matches": len(ids), "match_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "comparison": "primary = exact set of non-zero non-trinket ids, detail item0-5 + roleBoundItem vs replay; "
                      "before = ItemStateIndex(rules_16x=False) (replay after FIX B1-B3), after = 16.x rules",
        "before": res["before"], "after": res["after"],
        "after_mismatch_players_by_cause": dict(cause_after.most_common()),
        "after_mismatch_players_not_3867_proxy": len(residual),
        "after_agreement_if_only_3867_proxy_mismatches_counted_as_match": (n - len(residual)) / n,
        "after_residual_not_3867_proxy": residual,
        "slot_feasibility_after": {"normal_slots_hist": {str(k): v for k, v in sorted(slot_hist.items())},
                                   "players_over_6": sum(v for k, v in slot_hist.items() if k > 6),
                                   "examples_over_6": slot_over},
        "causality_after": {"matches": len(caus_ids), "player_times": caus["player_times"],
                            "identical_full_vs_events_upto_t": caus["identical"], "examples": caus_examples},
        "script": "scripts/exact_v4/ev4_s1x_items_16x.py validate", "elapsed_s": round(time.time() - t0, 1),
    }
    _dump("inventory_end_of_game_16.15_16x.json", out)
    brief = {vn: {k: v for k, v in r.items() if k.startswith(("agreement", "pass", "diagnostic"))}
             for vn, r in res.items()}
    return brief | {"after_causes": dict(cause_after.most_common()), "residual_not_proxy": len(residual),
                    "slots_over_6": out["slot_feasibility_after"]["players_over_6"],
                    "causality": out["causality_after"]["identical_full_vs_events_upto_t"],
                    "causality_n": caus["player_times"]}


# ============================================================================================ identity
def identity_ids():
    idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in idx.items() if v == "15.14")
    random.Random(IDENTITY_SEED).shuffle(ids)
    return sorted(ids[:IDENTITY_N])


def identity_digest(mod=None):
    """sha256 over (match, t, pid) -> (inventory, gold delta, stat delta, item vector) for the 50 15.14 matches."""
    import gameplay.item_state as default_mod
    m = mod or default_mod
    out = {}
    for mid in identity_ids():
        meta = json.loads((CACHE / f"{mid}.meta.json").read_text(encoding="utf-8"))
        assert str(meta["patch"]) == "15.14"
        ev = json.loads((CACHE / f"{mid}.events.json").read_text(encoding="utf-8"))
        ix = m.ItemStateIndex.from_pack({"events": ev, "meta": meta})
        assert ix.rules16 is None
        tl = max(int(e.get("timestamp", 0)) for e in ev)
        for t in list(range(0, tl + 60000, 60000)) + [tl]:
            for pid in range(1, 11):
                inv, g, s = ix.replay(pid, t, since_ms=max(0, t - 60000))
                out[(mid, t, pid)] = (tuple(inv), g, tuple(sorted(s.items())),
                                      tuple(sorted(m.player_item_vector(inv, ix.items, ix.flags).items())))
    return hashlib.sha256(repr(sorted(out.items())).encode()).hexdigest(), len(out)


def run_identity():
    h, n = identity_digest()
    out = {"check": "15.14 byte-identity of the item replay before / after the 16.x rules",
           "matches": identity_ids(), "n_match_time_players": n, "sha256_after": h, "sha256_before": REF_HASH_1514,
           "identical": h == REF_HASH_1514,
           "before_module": "REF_HASH_1514 (since task RULES15: the module with the 15.x rules; the pre-16X module "
                            "ce5a341b... gives REF_HASH_1514_NO_15X, reproduced with rules_15x=False)",
           "script": "scripts/exact_v4/ev4_s1x_items_16x.py identity"}
    _dump("identity_15.14.json", out)
    return {"identical": out["identical"], "n": n}


def main(which="all"):
    for name, fn in (("evidence", run_evidence), ("validate", run_validate), ("identity", run_identity)):
        if which in (name, "all"):
            print(name, json.dumps(fn(), ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    load_rules_16x()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
