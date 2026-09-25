"""Stage 1 validation reports (v4-exact, task Z).  Single process.

Three checks, each written to <OUT>/stage1/validation/.  The rules they validate (gameplay.item_state,
gameplay.event_survival, gameplay.state_value_v3, gameplay.setup_features) are NOT changed by this script; a
check below 99 % is reported with its causes (plan section "tests" 9: stop until the cause is found).

  inventory   16.15 KR raw, all 553 matches (D:/LOL_Project/fusion_2615/raw/kr/{detail,timeline}).
              Pack = what the cache pipeline would build (data.cache_io._parse_timeline_to_minute_cache_compat
              events; static_meta from _extract_static_meta_from_detail, pre-game fields only).
              replay_inventory(exact=True) through ItemStateIndex.from_pack (Viego rule, Magical Footwear grant)
              at the GAME_END event timestamp, per player, against detail item0-item5 (and item6 separately).
              Comparison conventions fixed before the first run:
                * trinket = Data Dragon tag 'Trinket'; replay items split into trinket / non-trinket;
                * 'exact set'      : set of non-zero non-trinket ids == set of non-zero item0-5 ids;
                * 'exact multiset' : the same with multiplicity (stackable consumables occupy one slot, so a
                                     multiset miss can be a stack, reported separately);
                * item6            : replay trinket set == {item6} (both empty when item6 == 0);
                * all7             : set over item0-6 == set over the whole replay inventory.
              99 % rule: applies to the exact-set agreement over item0-5.
  vision      the 1,333 pre-fight windows (D:/LOL_Project/fusion_2615/vision_windows/*/engagement_*/window.json;
              4 have no frames).  HUD alive per player = lol_fight_vision.hud_panel_reader.read_frame (health bar
              fill > 0) on the last kept frame with replay_clock_seconds <= prediction_cutoff_s; left panel =
              teamId 100.  Slot k of a panel = k-th participant of that team in participant order (pids 1-5 /
              6-10), fixed before the first run.  Event alive = gameplay.event_survival from the 16.15 timeline
              events <= t (death_intervals(events, patch, t=t)), t = that frame's replay clock in ms (primary) and
              the prediction cutoff (secondary).  Abstentions (health None) are counted, not scored.
              Order-free cross-check: per-team alive count.  99 % rule: per-player agreement at the frame clock.
  tau         200 random 15.14 matches (cache match_cache_fresh_v3_engage_status13, patch index == '15.14',
              sorted, random.Random(TAU_SEED).shuffle, first 200) x 20 query times: v4 engagements
              (detect_engagements_exact(mode='v4'), preset 'v4-exact' with G = 14 000 ms, D = 4 300) sampled
              without replacement (random.Random(TAU_SEED + 1)), topped up with uniform random t in
              [first frame + 1, last frame].  Query q = tau - 1 ms.  StateV3 (+ setup) from the full pack vs from
              the pack truncated to minute_ts <= q and events with timestamp <= q must be identical (NaN == NaN);
              non-finite values are counted per column.

Usage: python scripts/exact_v4/ev4_s1_validate.py [inventory|vision|tau|all]
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925/stage1/validation")
RAW = Path("D:/LOL_Project/fusion_2615/raw/kr")
VISION = Path("D:/LOL_Project/fusion_2615/vision_windows")
VISION_SRC = Path(r"C:/Users/todtj/문서/LOL_Teamfight/lol_fight_vision/src")
CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
PATCH_INDEX = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json")
TAU_PATCH = "15.14"
TAU_N_MATCHES, TAU_PER_MATCH, TAU_SEED = 200, 20, 2026092531
G_MS, D_UNITS = 14000, 4300.0
PASS_RATE = 0.99


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# ============================================================================================ 16.15 packs
def build_1615_pack(mid):
    from data.cache_io import _extract_static_meta_from_detail, _parse_timeline_to_minute_cache_compat
    from gameplay.fights import normalize_patch
    detail = json.loads((RAW / "detail" / f"{mid}.json").read_text(encoding="utf-8"))
    tl = json.loads((RAW / "timeline" / f"{mid}.json").read_text(encoding="utf-8"))
    parts = detail["info"]["participants"]
    tm = {int(p["participantId"]): int(p["teamId"]) for p in parts}
    cache = _parse_timeline_to_minute_cache_compat(tl, tm, detail=detail)
    patch = normalize_patch(str(detail["info"]["gameVersion"]))
    meta = {"match_id": mid, "patch": patch, "team_map": {str(k): v for k, v in tm.items()},
            "static_meta": _extract_static_meta_from_detail(detail)}
    pack = {"minute_ts": cache["minute_ts"], "node_minute": cache["node_minute"], "events": cache["events"],
            "meta": meta}
    return pack, detail


def _ids_1615():
    return sorted(p.stem for p in (RAW / "detail").glob("*.json") if (RAW / "timeline" / p.name).exists())


# ============================================================================================ 1. inventory
# POST-HOC cause labels (added after the first run, to explain the failure; they change no replay rule and no pass
# criterion).  'support_quest_proxy_by_design': the replay keeps 3867 as the documented proxy of its unobserved
# upgrade (item_state rule 3, identical stats); the other two are 16.x mechanics without a timeline event.
DESIGN_OR_16X = frozenset({"support_quest_proxy_by_design", "tier3_boots_upgrade_no_event", "control_ward_extra"})
SUPPORT_CHAIN = (3865, 3866, 3867)
JUNGLE_PETS = range(1101, 1108)
ROLE_QUEST_IDS = tuple(range(1200, 1223))
RECALL_IDS = (2001, 2002)
TIER3_BOOTS = tuple(range(3168, 3176))
SUPPORT_UPGRADES = (3869, 3870, 3871, 3876, 3877)
TRINKETS = (3340, 3363, 3364, 3330)


def classify_diff(miss, extra, table):
    from gameplay.item_state import is_boots, is_consumable
    miss, extra, cats = set(miss), set(extra), set()
    upg = {k for k, v in table.items() if list(v.get("from") or []) == [3867]}
    t3 = {k: v["from"][0] for k, v in table.items() if len(v.get("from") or []) == 1
          and is_boots(table.get(v["from"][0], {})) and (table[v["from"][0]].get("depth") or 0) >= 2}
    for m in sorted(miss):
        if m in upg:
            cats.add("support_quest_proxy_by_design" if 3867 in extra else "support_quest_upgrade_missing_other")
            miss.discard(m)
            extra.discard(3867)
        elif m in t3:
            cats.add("tier3_boots_upgrade_no_event" if t3[m] in extra else "tier3_boots_other")
            miss.discard(m)
            extra.discard(t3[m])
    for x in sorted(miss):
        it = table.get(x, {})
        cats.add("control_ward_missing" if x == 2055 else "support_chain_missing" if x in SUPPORT_CHAIN
                 else "consumable_missing" if is_consumable(it) else "component_missing" if it.get("into")
                 else "completed_or_other_missing")
    for x in sorted(extra):
        it = table.get(x, {})
        cats.add("control_ward_extra" if x == 2055 else "support_chain_extra" if x in SUPPORT_CHAIN
                 else "jungle_pet_extra" if x in JUNGLE_PETS else "magical_footwear_extra" if x == 2422
                 else "boots_extra_other" if is_boots(it) else "consumable_extra" if is_consumable(it) else "component_extra" if it.get("into")
                 else "completed_or_other_extra")
    return cats


def undo_dup_scan(ids, build, table):
    """ITEM_UNDO of a purchase whose recipe lists a component twice (3133 = 1036 + 2022 + 1036, ...): replay rule 1
    restores [c for c in recipe if c was destroyed at that ms], i.e. BOTH copies when only one destroy event was
    emitted at that ms.  Counts undo events where that restores more copies than destroy events of that id."""
    from gameplay.item_state import SHOP_TYPES
    n_undo, n_over, n_extra, by_item = 0, 0, 0, Counter()
    for m in ids:
        pack = build(m)
        if pack is None:
            continue
        per = defaultdict(list)
        for e in sorted((e for e in pack["events"] if e.get("type") in SHOP_TYPES), key=lambda e: int(e["timestamp"])):
            per[int(e.get("participantId", 0) or 0)].append(e)
        for pid, evs in per.items():
            if not 1 <= pid <= 10:
                continue
            destroyed = defaultdict(Counter)
            for e in evs:
                if e["type"] == "ITEM_DESTROYED":
                    destroyed[int(e["timestamp"])][int(e.get("itemId", 0) or 0)] += 1
            purchases = defaultdict(list)
            for e in evs:
                ts = int(e["timestamp"])
                if e["type"] == "ITEM_PURCHASED":
                    iid = int(e.get("itemId", 0) or 0)
                    frm = list((table.get(iid) or {}).get("from") or [])
                    purchases[iid].append((frm, ts))
                elif e["type"] == "ITEM_UNDO":
                    b, a = int(e.get("beforeId", 0) or 0), int(e.get("afterId", 0) or 0)
                    if b and not a and purchases.get(b):
                        frm, pts = purchases[b].pop()
                        n_undo += 1
                        restored = Counter(c for c in frm if destroyed[pts][c] > 0)
                        over = sum(max(0, k - destroyed[pts][c]) for c, k in restored.items())
                        if over:
                            n_over += 1
                            n_extra += over
                            by_item[b] += 1
    return {"undo_of_purchase_events": n_undo, "undo_events_overrestoring": n_over,
            "extra_components_restored": n_extra, "by_undone_item": dict(by_item.most_common())}


def prevalence(ids, load):
    """Counts of the 16.15 failure mechanisms in the ITEM_* events of `ids` (loaded by `load`)."""
    c = Counter()
    for m in ids:
        pack = load(m)
        evs = sorted(pack["events"], key=lambda e: int(e["timestamp"]))
        first_tr_buy, first_tr_destroy = {}, {}
        cw = defaultdict(lambda: [0, 0])
        for e in evs:
            et = e.get("type", "")
            if not et.startswith("ITEM_"):
                continue
            c["item_events"] += 1
            iid = int(e.get("itemId", 0) or 0)
            pid = int(e.get("participantId", 0) or 0)
            ts = int(e["timestamp"])
            c["role_quest_item_events"] += int(iid in ROLE_QUEST_IDS)
            c["recall_item_events"] += int(iid in RECALL_IDS)
            c["tier3_boots_item_events"] += int(iid in TIER3_BOOTS)
            c["tier3_boots_purchases"] += int(iid in TIER3_BOOTS and et == "ITEM_PURCHASED")
            c["support_upgrade_item_events"] += int(iid in SUPPORT_UPGRADES)
            c["undo_before0_after0"] += int(et == "ITEM_UNDO" and not int(e.get("beforeId", 0) or 0)
                                          and not int(e.get("afterId", 0) or 0))
            if iid in TRINKETS:
                d = first_tr_buy if et == "ITEM_PURCHASED" else first_tr_destroy if et == "ITEM_DESTROYED" else None
                if d is not None and pid not in d:
                    d[pid] = ts
            if iid == 2055 and et in ("ITEM_PURCHASED", "ITEM_DESTROYED"):
                cw[pid][0 if et == "ITEM_PURCHASED" else 1] += 1
        for pid in range(1, 11):
            c["players"] += 1
            if pid in first_tr_destroy and first_tr_destroy[pid] < first_tr_buy.get(pid, 10 ** 12):
                c["players_trinket_destroyed_before_any_trinket_purchase"] += 1
            if pid not in first_tr_buy:
                c["players_without_trinket_purchase"] += 1
            if cw[pid][1] > cw[pid][0]:
                c["players_control_ward_destroys_gt_purchases"] += 1
    out = dict(c)
    out["n_matches"] = len(ids)
    return out


def prevalence_1514():
    """The same counts on 15.14 (the tau sample: 200 cached 15.14 matches), plus the undo over-restore scan."""
    ids = _tau_ids()
    out = prevalence(ids, _load_1514)
    out["undo_duplicate_component_overrestore"] = undo_dup_scan(ids, _load_1514, _tables_1514())
    return out


def _tables_1514():
    from gameplay.item_state import load_item_table_v2
    return load_item_table_v2(TAU_PATCH)


def run_inventory():
    from gameplay.item_state import ItemStateIndex, load_item_table_v2
    t0 = time.time()
    ids = _ids_1615()
    tables = {}
    names = {}
    c = Counter()
    missing_by_id, extra_by_id = Counter(), Counter()
    ms_only_missing, ms_only_extra = Counter(), Counter()
    item6_pairs = Counter()
    diag_tot = Counter()
    unowned_ids = Counter()
    champ_mism = Counter()
    examples, ms_examples = [], []
    cause_players, cause_combo = Counter(), Counter()
    var = {vn: {"c": Counter(), "cause": Counter(), "combo": Counter(), "missing": Counter(), "extra": Counter(),
                "residual": []} for vn in ("rb", "rbq")}
    patches = Counter()
    per_match_mism = Counter()
    for mid in ids:
        pack, detail = build_1615_pack(mid)
        patch = pack["meta"]["patch"]
        patches[patch] += 1
        if patch not in tables:
            tables[patch] = load_item_table_v2(patch)
            names.update({k: v.get("name", "") for k, v in tables[patch].items()})
        table = tables[patch]
        idx = ItemStateIndex.from_pack(pack, table)
        ends = [int(e["timestamp"]) for e in pack["events"] if e.get("type") == "GAME_END"]
        t_end = max(ends) if ends else max(int(e["timestamp"]) for e in pack["events"])
        c["game_end_event"] += int(bool(ends))
        for p in detail["info"]["participants"]:
            pid = int(p["participantId"])
            diag = {}
            inv = idx.inventory(pid, t_end, diagnostics=diag)
            for k, v in diag.items():
                if k != "unowned_ids":
                    diag_tot[k] += v
            for k, v in diag.get("unowned_ids", {}).items():
                unowned_ids[int(k)] += v

            def trink(i):
                return "Trinket" in ((table.get(int(i)) or {}).get("tags") or [])
            rep_main = Counter(int(i) for i in inv if not trink(i))
            rep_tr = Counter(int(i) for i in inv if trink(i))
            det_main = Counter(int(p[f"item{k}"]) for k in range(6) if int(p.get(f"item{k}", 0) or 0))
            it6 = int(p.get("item6", 0) or 0)
            det_tr = Counter([it6] if it6 else [])
            c["players"] += 1
            set_ok = set(rep_main) == set(det_main)
            ms_ok = rep_main == det_main
            tr_ok = set(rep_tr) == set(det_tr)
            all7_ok = (set(rep_main) | set(rep_tr)) == (set(det_main) | set(det_tr))
            c["set_ok"] += set_ok
            c["multiset_ok"] += ms_ok
            c["item6_ok"] += tr_ok
            c["all7_ok"] += all7_ok
            c["replay_over_6_nontrinket"] += int(sum(rep_main.values()) > 6)
            if not set_ok:
                per_match_mism[mid] += 1
                miss = set(det_main) - set(rep_main)
                extra = set(rep_main) - set(det_main)
                cats = classify_diff(miss, extra, table)
                for k in cats:
                    cause_players[k] += 1
                cause_combo[" + ".join(sorted(cats))] += 1
                if cats <= DESIGN_OR_16X:
                    c["set_mismatch_only_design_or_16x"] += 1
                if cats <= {"support_quest_proxy_by_design"}:
                    c["set_mismatch_only_proxy"] += 1
                for i in miss:
                    missing_by_id[i] += 1
                for i in extra:
                    extra_by_id[i] += 1
                champ_mism[str(p.get("championName"))] += 1
                if len(examples) < 60:
                    examples.append({"match": mid, "pid": pid, "champion": p.get("championName"),
                                     "missing_in_replay": [[i, names.get(i, "?")] for i in sorted(miss)],
                                     "extra_in_replay": [[i, names.get(i, "?")] for i in sorted(extra)],
                                     "detail_item0_5": sorted(det_main.elements()), "replay": sorted(inv),
                                     "diagnostics": {k: v for k, v in diag.items() if v}})
            elif not ms_ok:
                for i, n in (det_main - rep_main).items():
                    ms_only_missing[i] += n
                for i, n in (rep_main - det_main).items():
                    ms_only_extra[i] += n
                if len(ms_examples) < 20:
                    ms_examples.append({"match": mid, "pid": pid, "detail": sorted(det_main.elements()),
                                        "replay_nontrinket": sorted(rep_main.elements())})
            if not tr_ok:
                item6_pairs[(it6, tuple(sorted(rep_tr)))] += 1
            # Listed comparison changes after the first run (the replay is unchanged):
            #   rb  : 16.x detail moves the role-bound item (role-quest token / reward, boots) out of item0-5 into
            #         'roleBoundItem'; compare the replay with item0-5 + roleBoundItem.
            #   rbq : rb, with the 16.x role-quest ids 1200-1222 (quest tokens and rewards; never purchased, so absent
            #         from ITEM_PURCHASED) removed from both sides.
            rb = int(p.get("roleBoundItem", 0) or 0)
            c["players_with_roleBoundItem"] += int(bool(rb))
            c["players_roleBoundItem_is_role_quest_id"] += int(rb in ROLE_QUEST_IDS)
            det_rb = det_main + (Counter([rb]) if rb and not trink(rb) else Counter())
            variants = {
                "rb": (det_rb, rep_main),
                "rbq": (Counter({k: v for k, v in det_rb.items() if k not in ROLE_QUEST_IDS}),
                        Counter({k: v for k, v in rep_main.items() if k not in ROLE_QUEST_IDS})),
            }
            for vn, (dv, rv) in variants.items():
                V = var[vn]
                ok = set(rv) == set(dv)
                V["c"]["set_ok"] += ok
                V["c"]["multiset_ok"] += rv == dv
                if ok:
                    continue
                miss, extra = set(dv) - set(rv), set(rv) - set(dv)
                cats = classify_diff(miss, extra, table)
                for k in cats:
                    V["cause"][k] += 1
                V["combo"][" + ".join(sorted(cats))] += 1
                for i in miss:
                    V["missing"][i] += 1
                for i in extra:
                    V["extra"][i] += 1
                if cats <= {"support_quest_proxy_by_design"}:
                    V["c"]["only_proxy"] += 1
                if cats <= DESIGN_OR_16X:
                    V["c"]["only_design_or_16x"] += 1
                elif len(V["residual"]) < 40:
                    V["residual"].append({"match": mid, "pid": pid, "champion": p.get("championName"),
                                          "causes": sorted(cats),
                                          "missing_in_replay": [[i, names.get(i, "?")] for i in sorted(miss)],
                                          "extra_in_replay": [[i, names.get(i, "?")] for i in sorted(extra)],
                                          "detail_compared": sorted(dv.elements()), "replay": sorted(inv),
                                          "diagnostics": {k: v for k, v in diag.items() if v}})
    n = c["players"]
    rate = c["set_ok"] / max(1, n)
    named = lambda cnt: [[int(i), names.get(int(i), "?"), v] for i, v in cnt.most_common()]
    out = {
        "check": "end-of-game inventory, replay_inventory(exact=True) at GAME_END vs detail item0-5 / item6",
        "data": str(RAW), "n_matches": len(ids), "match_ids_sha256": _sha(ids), "patches": dict(patches),
        "n_players": n, "matches_with_game_end_event": c["game_end_event"],
        "agreement_exact_set_item0_5": rate,
        "agreement_exact_multiset_item0_5": c["multiset_ok"] / max(1, n),
        "agreement_item6_trinket": c["item6_ok"] / max(1, n),
        "agreement_set_item0_6_vs_whole_replay": c["all7_ok"] / max(1, n),
        "n_set_mismatch_players": n - c["set_ok"], "n_matches_with_set_mismatch": len(per_match_mism),
        "players_replay_holding_over_6_nontrinket": c["replay_over_6_nontrinket"],
        "pass_99pct_rule_exact_set": rate >= PASS_RATE,
        "set_mismatch_missing_in_replay_by_id": named(missing_by_id),
        "set_mismatch_extra_in_replay_by_id": named(extra_by_id),
        "set_mismatch_by_champion": dict(champ_mism.most_common()),
        "cause_classification_note": "POST-HOC diagnostic labels of each set-mismatch player's missing/extra ids "
                                     "(classify_diff); written after the first run to explain the failure, not a "
                                     "rule change and not a pass criterion",
        "set_mismatch_players_by_cause": dict(cause_players.most_common()),
        "set_mismatch_players_by_cause_combination": dict(cause_combo.most_common(40)),
        "diagnostic_rate_if_design_proxy_counted_as_match": (c["set_ok"] + c["set_mismatch_only_proxy"]) / max(1, n),
        "diagnostic_rate_if_design_proxy_and_16x_only_causes_counted_as_match":
            (c["set_ok"] + c["set_mismatch_only_design_or_16x"]) / max(1, n),
        "listed_comparison_changes_after_first_run": [
            "rb: detail item0-5 + roleBoundItem (16.x role-bound slot: role-quest token / reward or boots)",
            "rbq: rb with the 16.x role-quest ids 1200-1222 removed from both sides (never purchased)",
            "diagnostic only: the replay's documented 3867 proxy (item_state rule 3) counted as its unobserved "
            "upgrade; and, additionally, the 16.x-only causes (tier-3 boots without event, control-ward extra)"],
        "n_players_with_roleBoundItem": c["players_with_roleBoundItem"],
        "n_players_roleBoundItem_is_role_quest_id": c["players_roleBoundItem_is_role_quest_id"],
        "variants": {vn: {
            "agreement_exact_set": V["c"]["set_ok"] / max(1, n),
            "agreement_exact_multiset": V["c"]["multiset_ok"] / max(1, n),
            "pass_99pct_rule": V["c"]["set_ok"] / max(1, n) >= PASS_RATE,
            "diagnostic_agreement_proxy_counted_as_upgrade": (V["c"]["set_ok"] + V["c"]["only_proxy"]) / max(1, n),
            "diagnostic_agreement_proxy_and_16x_causes_counted_as_match":
                (V["c"]["set_ok"] + V["c"]["only_design_or_16x"]) / max(1, n),
            "mismatch_players_by_cause": dict(V["cause"].most_common()),
            "mismatch_players_by_cause_combination": dict(V["combo"].most_common(40)),
            "mismatch_missing_in_replay_by_id": named(V["missing"]),
            "mismatch_extra_in_replay_by_id": named(V["extra"]),
            "residual_examples_not_proxy_or_16x": V["residual"]} for vn, V in var.items()},
        "undo_duplicate_component_overrestore": undo_dup_scan(ids, build=lambda m: build_1615_pack(m)[0],
                                                              table=tables["16.15"]),
        "prevalence_of_causes_16.15": prevalence(ids, lambda m: build_1615_pack(m)[0]),
        "prevalence_of_causes_15.14_tau_sample": prevalence_1514(),
        "multiset_only_mismatch_missing_by_id": named(ms_only_missing),
        "multiset_only_mismatch_extra_by_id": named(ms_only_extra),
        "item6_mismatch_pairs_detail_vs_replay_trinkets": [[k[0], names.get(k[0], "?"), list(k[1]), v]
                                                            for k, v in item6_pairs.most_common()],
        "replay_diagnostics_totals": dict(diag_tot),
        "replay_unowned_removal_ids": named(unowned_ids),
        "set_mismatch_examples": examples, "multiset_only_examples": ms_examples,
        "script": "scripts/exact_v4/ev4_s1_validate.py inventory", "elapsed_s": round(time.time() - t0, 1),
    }
    _dump("inventory_end_of_game_16.15.json", out)
    return {k: out[k] for k in ("n_players", "agreement_exact_set_item0_5", "agreement_exact_multiset_item0_5",
                                "agreement_item6_trinket", "agreement_set_item0_6_vs_whole_replay",
                                "pass_99pct_rule_exact_set")} | {
        "top_missing": out["set_mismatch_missing_in_replay_by_id"][:15],
        "top_extra": out["set_mismatch_extra_in_replay_by_id"][:15],
        "item6_pairs": out["item6_mismatch_pairs_detail_vs_replay_trinkets"][:10],
        "ms_missing": out["multiset_only_mismatch_missing_by_id"][:10],
        "ms_extra": out["multiset_only_mismatch_extra_by_id"][:10], "diag": dict(diag_tot)}


# ============================================================================================ 2. vision
def run_vision():
    from PIL import Image
    sys.path.insert(0, str(VISION_SRC))
    from lol_fight_vision.hud_panel_reader import read_frame
    from gameplay.event_survival import death_intervals, event_alive
    t0 = time.time()
    wins = sorted(VISION.glob("*/engagement_*/window.json"))
    c = Counter()
    pack_cache = {}
    per_slot = defaultdict(lambda: [0, 0])
    conf = Counter()                      # (event_alive, hud_alive) at the frame clock
    conf_cut = Counter()
    boundary = []                         # mismatches: signed seconds to the nearest interval end point
    team_cnt = Counter()
    lag = []
    examples = []
    res = Counter()
    by_scale = defaultdict(lambda: [0, 0])
    abst_slot = Counter()
    for wp in wins:
        w = json.loads(wp.read_text(encoding="utf-8"))
        c["windows"] += 1
        frames = [f for f in (w.get("frames") or []) if f.get("replay_clock_seconds") is not None]
        cut = float(w["prediction_cutoff_s"])
        cand = [f for f in frames if float(f["replay_clock_seconds"]) <= cut]
        if not cand:
            c["windows_no_frame_le_cutoff"] += 1
            continue
        fr = max(cand, key=lambda f: float(f["replay_clock_seconds"]))
        mid = w["match_id"]
        if mid not in pack_cache:
            pack_cache.clear()
            pack_cache[mid] = build_1615_pack(mid)
        pack, detail = pack_cache[mid]
        tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
        patch = pack["meta"]["patch"]
        img = np.asarray(Image.open(wp.parent / fr["file"]).convert("RGB"))
        res[f"{img.shape[1]}x{img.shape[0]}"] += 1
        readings = read_frame(img)
        t_fr = float(fr["replay_clock_seconds"]) * 1000.0
        lag.append(cut - float(fr["replay_clock_seconds"]))
        a_fr = event_alive(death_intervals(pack["events"], patch, t=t_fr), t_fr)
        iv_all = death_intervals(pack["events"], patch)
        a_cut = event_alive(death_intervals(pack["events"], patch, t=cut * 1000.0), cut * 1000.0)
        c["windows_scored"] += 1
        order = {100: sorted(p for p in tm if tm[p] == 100), 200: sorted(p for p in tm if tm[p] == 200)}
        hud_cnt = {100: 0, 200: 0}
        hud_abst = {100: 0, 200: 0}
        win_ok = True
        for r in readings:
            team = 100 if r.team == "left" else 200
            pid = order[team][r.slot]
            if r.health is None:
                c["player_abstain"] += 1
                abst_slot[f"{r.team}{r.slot}"] += 1
                hud_abst[team] += 1
                continue
            hud = bool(r.alive)
            hud_cnt[team] += int(hud)
            ev = bool(a_fr[pid - 1] > 0.5)
            evc = bool(a_cut[pid - 1] > 0.5)
            c["players_scored"] += 1
            c["agree_frame"] += int(ev == hud)
            c["agree_cutoff"] += int(evc == hud)
            conf[(int(ev), int(hud))] += 1
            conf_cut[(int(evc), int(hud))] += 1
            per_slot[f"{r.team}{r.slot}"][0] += int(ev == hud)
            per_slot[f"{r.team}{r.slot}"][1] += 1
            by_scale[str(w.get("fight_scale"))][0] += int(ev == hud)
            by_scale[str(w.get("fight_scale"))][1] += 1
            if ev != hud:
                win_ok = False
                pts = [x for iv in iv_all.get(pid, []) for x in iv]
                near = min(pts, key=lambda x: abs(x - t_fr)) if pts else None
                d = None if near is None else (t_fr - near) / 1000.0
                boundary.append(d)
                if len(examples) < 40:
                    examples.append({"window": str(wp.parent.relative_to(VISION)), "pid": pid, "slot": f"{r.team}{r.slot}",
                                     "t_frame_s": t_fr / 1000.0, "event_alive": ev, "hud_alive": hud,
                                     "hud_health": r.health,
                                     "death_intervals_s": [[a / 1000.0, b / 1000.0] for a, b in iv_all.get(pid, [])
                                                           if a - 120_000 <= t_fr <= b + 120_000],
                                     "seconds_from_nearest_interval_endpoint": d})
        c["windows_all_players_agree"] += int(win_ok)
        for team in (100, 200):
            if hud_abst[team]:
                team_cnt["abstain"] += 1
                continue
            ev_cnt = int(sum(a_fr[p - 1] > 0.5 for p in order[team]))
            team_cnt["teams"] += 1
            team_cnt["agree"] += int(ev_cnt == hud_cnt[team])
    n = c["players_scored"]
    rate = c["agree_frame"] / max(1, n)
    bd = [x for x in boundary if x is not None]
    near_bins = Counter()
    for x in boundary:
        if x is None:
            near_bins["no_death_interval"] += 1
        else:
            ax = abs(x)
            near_bins["<=1s" if ax <= 1 else "<=3s" if ax <= 3 else "<=10s" if ax <= 10 else ">10s"] += 1
    out = {
        "check": "HUD alive (hud_panel_reader) vs event_survival alive at the last frame <= prediction cutoff",
        "data": str(VISION), "n_windows": c["windows"], "n_windows_scored": c["windows_scored"],
        "n_windows_no_frame_le_cutoff": c["windows_no_frame_le_cutoff"], "frame_resolutions": dict(res),
        "cutoff_minus_frame_clock_s": {"mean": float(np.mean(lag)) if lag else None,
                                       "max": float(np.max(lag)) if lag else None,
                                       "p95": float(np.percentile(lag, 95)) if lag else None},
        "n_player_readings": n, "n_player_abstain": c["player_abstain"], "abstain_by_slot": dict(abst_slot),
        "agreement_player_at_frame_clock": rate,
        "agreement_player_at_prediction_cutoff": c["agree_cutoff"] / max(1, n),
        "pass_99pct_rule": rate >= PASS_RATE,
        "confusion_event_hud_at_frame": {f"event{a}_hud{b}": v for (a, b), v in sorted(conf.items())},
        "confusion_event_hud_at_cutoff": {f"event{a}_hud{b}": v for (a, b), v in sorted(conf_cut.items())},
        "windows_all_10_agree": c["windows_all_players_agree"] / max(1, c["windows_scored"]),
        "team_alive_count_agreement_order_free": team_cnt["agree"] / max(1, team_cnt["teams"]),
        "n_teams_count_scored": team_cnt["teams"], "n_teams_with_abstain": team_cnt["abstain"],
        "agreement_by_hud_slot": {k: v[0] / max(1, v[1]) for k, v in sorted(per_slot.items())},
        "agreement_by_fight_scale": {k: [v[0] / max(1, v[1]), v[1]] for k, v in by_scale.items()},
        "mismatch_seconds_from_nearest_death_or_respawn": near_bins,
        "mismatch_signed_offsets_s": sorted(round(x, 3) for x in bd),
        "mismatch_examples": examples,
        "slot_mapping": "left panel = teamId 100, slot k = k-th pid of the team in participant order",
        "script": "scripts/exact_v4/ev4_s1_validate.py vision", "elapsed_s": round(time.time() - t0, 1),
    }
    _dump("vision_alive_16.15.json", out)
    return {k: out[k] for k in ("n_windows_scored", "n_player_readings", "n_player_abstain",
                                "agreement_player_at_frame_clock", "agreement_player_at_prediction_cutoff",
                                "pass_99pct_rule", "confusion_event_hud_at_frame", "team_alive_count_agreement_order_free",
                                "windows_all_10_agree", "mismatch_seconds_from_nearest_death_or_respawn",
                                "agreement_by_hud_slot", "cutoff_minus_frame_clock_s")}


# ============================================================================================ 3. tau
def _tau_ids():
    idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in idx.items() if v == TAU_PATCH)
    random.Random(TAU_SEED).shuffle(ids)
    return ids[:TAU_N_MATCHES]


def _load_1514(m):
    meta = json.loads((CACHE / f"{m}.meta.json").read_text(encoding="utf-8"))
    if str(meta.get("patch")) != TAU_PATCH:
        raise RuntimeError(f"{m}: patch {meta.get('patch')} != {TAU_PATCH}")
    ev = json.loads((CACHE / f"{m}.events.json").read_text(encoding="utf-8"))
    with np.load(CACHE / f"{m}.npz") as z:
        return {"minute_ts": z["minute_ts"], "node_minute": z["node_minute"], "events": ev, "meta": meta}


def _trunc(pack, t):
    keep = np.asarray(pack["minute_ts"]) <= t
    return {"minute_ts": pack["minute_ts"][keep], "node_minute": pack["node_minute"][keep],
            "events": [e for e in pack["events"] if int(e["timestamp"]) <= t], "meta": copy.deepcopy(pack["meta"])}


def _same_dict(a, b):
    if list(a) != list(b):
        return False, ["<column order>"]
    bad = [k for k in a if not (a[k] == b[k] or (isinstance(a[k], float) and isinstance(b[k], float)
                                                 and math.isnan(a[k]) and math.isnan(b[k])))]
    return not bad, bad


def run_tau():
    from core.config import cfg
    from core.presets import apply_preset
    from gameplay.exact_population import ExactParams, detect_engagements_exact
    from gameplay.setup_features import SETUP_COLUMNS
    from gameplay.state_value_v3 import STATE_V3_COLUMNS, StateBuilderV3
    t0 = time.time()
    c0 = copy.copy(cfg)
    apply_preset(c0, "v4-exact")
    c0.TF2_KILL_CLUSTER_GAP_MS, c0.CLUSTER_MAX_DIAMETER = G_MS, D_UNITS
    params = ExactParams.from_cfg(c0, "v4")
    ids = _tau_ids()
    rng = random.Random(TAU_SEED + 1)
    c = Counter()
    mism, mism_cols = [], Counter()
    nonfin_state, nonfin_setup = Counter(), Counter()
    errors = []
    for m in ids:
        pack = _load_1514(m)
        tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
        epack = {"meta": dict(pack["meta"], match_id=m), "events": pack["events"],
                 "minute_ts": np.asarray(pack["minute_ts"]).astype(np.int64)}
        recs, _ = detect_engagements_exact(epack, tm, params, mode="v4")
        taus = sorted({int(r["tau"]) for r in recs})
        ts = np.asarray(pack["minute_ts"], dtype=np.int64)
        pick = rng.sample(taus, min(TAU_PER_MATCH, len(taus)))
        src = ["engagement"] * len(pick)
        while len(pick) < TAU_PER_MATCH:
            pick.append(rng.randint(int(ts[0]) + 1, int(ts[-1])))
            src.append("random")
        c["matches"] += 1
        c["engagements_detected"] += len(taus)
        full_b = StateBuilderV3(pack, TAU_PATCH)
        for tau, s in zip(pick, src):
            q = tau - 1
            c[f"queries_{s}"] += 1
            if q < int(ts[0]):
                c["skipped_before_first_frame"] += 1
                continue
            try:
                a = full_b.at(q, setup=True)
                b = StateBuilderV3(_trunc(pack, q), TAU_PATCH).at(q, setup=True)
            except Exception as e:                           # reported, not hidden
                errors.append([m, tau, s, f"{type(e).__name__}: {e}"[:300]])
                continue
            c["queries"] += 1
            ok1, bad1 = _same_dict(a.values, b.values)
            ok2, bad2 = _same_dict(a.setup, b.setup)
            ok3 = (a.snapshot_ms == b.snapshot_ms and a.slot_by_pid == b.slot_by_pid
                   and a.objective_rule_flags == b.objective_rule_flags)
            if ok1 and ok2 and ok3:
                c["identical"] += 1
            else:
                mism.append([m, tau, s, bad1[:5], bad2[:5], ok3])
                for k in bad1 + bad2:
                    mism_cols[k] += 1
            for k, v in a.values.items():
                if not math.isfinite(v):
                    nonfin_state[k] += 1
            for k, v in a.setup.items():
                if not math.isfinite(v):
                    nonfin_setup[k] += 1
    n = c["queries"]
    out = {
        "check": "future information: StateV3 + setup at q = tau - 1 ms, full pack vs pack truncated to data <= q",
        "patch": TAU_PATCH, "n_matches": len(ids), "match_ids_sha256": _sha(ids), "sample_seed": TAU_SEED,
        "params": {"G_ms": G_MS, "D": D_UNITS, "preset": "v4-exact", "mode": "v4",
                   "clean_max_age_ms": params.clean_max_age_ms, "merge": params.merge},
        "counts": dict(c), "n_queries_compared": n, "n_identical": c["identical"],
        "n_mismatch": len(mism), "pass": len(mism) == 0 and not errors and n == TAU_N_MATCHES * TAU_PER_MATCH,
        "mismatch_columns": dict(mism_cols.most_common()), "mismatch_examples": mism[:20],
        "errors": errors[:20], "n_errors": len(errors),
        "n_state_columns": len(STATE_V3_COLUMNS), "n_setup_columns": len(SETUP_COLUMNS),
        "nonfinite_state_values_total": int(sum(nonfin_state.values())),
        "nonfinite_state_by_column": dict(nonfin_state.most_common()),
        "nonfinite_setup_values_total": int(sum(nonfin_setup.values())),
        "nonfinite_setup_by_column": dict(nonfin_setup.most_common()),
        "script": "scripts/exact_v4/ev4_s1_validate.py tau", "elapsed_s": round(time.time() - t0, 1),
    }
    _dump("tau_minus_1_exactness_15.14.json", out)
    return {k: out[k] for k in ("counts", "n_queries_compared", "n_identical", "n_mismatch", "n_errors", "pass",
                                "nonfinite_state_values_total", "nonfinite_setup_values_total", "elapsed_s")} | {
        "nonfinite_state_cols": len(nonfin_state), "mism_cols": dict(mism_cols.most_common(10)),
        "errors": errors[:3]}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    for name, fn in (("inventory", run_inventory), ("vision", run_vision), ("tau", run_tau)):
        if which in (name, "all"):
            res[name] = fn()
            print(json.dumps({name: res[name]}, indent=1, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
