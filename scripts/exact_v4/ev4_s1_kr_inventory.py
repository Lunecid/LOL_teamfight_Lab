"""Stage 1 check (b): end-of-game inventory on a 500-match KR 15.18 sample (v4-exact, task KR).  Single process.

Author decision records/items_validation_decisions_20260925T113050Z.json: "(b) end-of-game inventory on a 500-match
KR 15.18 sample (detail item0-5, item6 separately).  The 99% rule applies ... The 15.18 sample is added to the
replication exclusion list."

Data: Match-V5 KR raw, C:/Users/todtj/PycharmProjects/Lol_project/data/raw/matches/kr/{detail,timeline}.
Sample (fixed before the first run):
  * every detail file name = match id; ids ordered by sha256(match id) hex (ascending);
  * walked in that order; a match is taken when detail info.gameVersion normalises to 15.18, info.queueId == 420,
    info.mapId == 11 and its timeline file exists; stop at N_SAMPLE = 500.  A byte-level regex on gameVersion /
    queueId / mapId only pre-screens files; the taken match is confirmed on the parsed fields;
  * the ids (sample order) -> records/replication_exclusions_inventory_1518.txt (one id per line), to be excluded
    from the replication.
Fields read.  Both files are parsed with an object_pairs_hook that drops every outcome key (OUTCOME_KEYS: win,
  winningTeam, teams, surrender / nexus fields, ...) before anything else sees the object, so no outcome value is
  kept, counted or printed.  From the detail only: info.gameVersion / queueId / mapId, and per participant
  participantId, championId, championName, the pre-game summoner spell ids and rune selections (the replay's Viego and
  Magical Footwear rules need champion and runes), item0-item6 and roleBoundItem (when present).  From the timeline:
  frames[].events (type + timestamp present; GAME_END keeps only type and timestamp).  Frames' participantFrames are
  not used.
Replay: gameplay.item_state.ItemStateIndex(exact=True, patch='15.18', table items_15.18.json) with champion_by_pid /
  runes_by_pid / summoner_spells_by_pid built by data.cache_io._extract_static_meta_from_detail from the filtered
  detail (the cache pipeline's format), at the GAME_END timestamp (fallback: the last event).  Patch 15.18 has major
  15, so no 16.x rule is active.
Comparison conventions (the ev4_s1_validate.py inventory conventions, fixed before the first run):
  * trinket = Data Dragon tag 'Trinket' (15.18 table); replay items split into trinket / non-trinket;
  * primary, 99 % rule : set of non-zero item0-5 ids == set of the replay's non-trinket ids (exact set);
  * multiset           : the same with multiplicity (a stack of consumables takes one slot: reported, not scored);
  * item6              : replay trinket set == {item6} (both empty when item6 == 0), reported separately;
  * all7               : set over item0-6 == set over the whole replay.
  Diagnostic only (not the 99 % rule): the replay's 3867 Bounty of Worlds (item_state rule 3, the documented proxy
  of its unobserved upgrade, identical stats) counted as matching a detail support upgrade (items built only from
  3867).  Cause labels of the mismatches are descriptive (post-hoc, per missing / extra id).
Also: slot feasibility (replay non-trinket count > 6), replay diagnostics (unowned removals by id), per-slot-role
  agreement (slot order = role, as in Stage 1), causality spot check (index from events <= t vs full index at random
  t), and 15.18-specific mechanics: event types, item ids absent from the 15.18 / 15.14 tables, ids whose recipe or
  price differs between the 15.14 and 15.18 tables among the ids the sample touches, participantId-0 item events.
Outputs: <OUT> = stage1/validation/kr1518/: inventory_end_of_game_15.18.json, sample_match_ids_15.18.json.

Usage: python scripts/exact_v4/ev4_s1_kr_inventory.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.cache_io import _extract_static_meta_from_detail  # noqa: E402
from gameplay.fights import normalize_patch  # noqa: E402
from gameplay.item_state import (  # noqa: E402
    ItemStateIndex, is_boots, is_consumable, is_trinket, load_item_table_v2, uses_rules_16x,
)

KR = Path("C:/Users/todtj/PycharmProjects/Lol_project/data/raw/matches/kr")
BASE = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925")
OUT = BASE / "stage1/validation/kr1518"
EXCLUSIONS = BASE / "records/replication_exclusions_inventory_1518.txt"
PATCH, REF_PATCH = "15.18", "15.14"
QUEUE, MAP = 420, 11
N_SAMPLE = 500
PASS_RATE = 0.99
CAUSAL_SEED, CAUSAL_MATCHES, CAUSAL_TIMES = 2026092518, 60, 8
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
SUPPORT_CHAIN = (3865, 3866, 3867)
JUNGLE_PETS = frozenset(range(1101, 1108))
VIEGO = 234

# Any key that reveals the result is dropped at parse time (object_pairs_hook), wherever it occurs.
OUTCOME_KEYS = frozenset({
    "win", "winningTeam", "teams", "gameEndedInSurrender", "gameEndedInEarlySurrender", "teamEarlySurrendered",
    "nexusKills", "nexusLost", "nexusTakedowns", "inhibitorsLost", "turretsLost", "endOfGameResult",
    "challenges", "missions", "placement", "subteamPlacement", "playerSubteamId",
})
RX_VER = re.compile(rb'"gameVersion"\s*:\s*"([^"]*)"')
RX_Q = re.compile(rb'"queueId"\s*:\s*(\d+)')
RX_MAP = re.compile(rb'"mapId"\s*:\s*(\d+)')


def _hook(pairs):
    return {k: v for k, v in pairs if k not in OUTCOME_KEYS}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_hook)


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _sha_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ============================================================================================ sample
def select_sample():
    ids = [f[:-5] for f in os.listdir(KR / "detail") if f.endswith(".json")]
    ids.sort(key=_sha_hex)
    scan = Counter()
    taken = []
    for mid in ids:
        if len(taken) >= N_SAMPLE:
            break
        scan["scanned"] += 1
        b = (KR / "detail" / f"{mid}.json").read_bytes()
        v, q, m = RX_VER.search(b), RX_Q.search(b), RX_MAP.search(b)
        if not (v and q and m):
            scan["skip_fields_missing"] += 1
            continue
        if normalize_patch(v.group(1).decode()) != PATCH:
            scan["skip_other_patch"] += 1
            continue
        if int(q.group(1)) != QUEUE or int(m.group(1)) != MAP:
            scan["skip_queue_or_map"] += 1
            continue
        if not (KR / "timeline" / f"{mid}.json").exists():
            scan["skip_no_timeline"] += 1
            continue
        info = _load(KR / "detail" / f"{mid}.json")["info"]         # confirm on the parsed fields
        if (normalize_patch(str(info["gameVersion"])) != PATCH or int(info["queueId"]) != QUEUE
                or int(info["mapId"]) != MAP):
            scan["skip_parsed_mismatch"] += 1
            continue
        taken.append(mid)
    return taken, dict(scan), len(ids)


# ============================================================================================ loading
def load_match(mid):
    raw = _load(KR / "detail" / f"{mid}.json")
    info = raw["info"]
    keep = ("participantId", "championId", "championName", "summoner1Id", "summoner2Id", "perks",
            *(f"item{k}" for k in range(7)), "roleBoundItem")
    parts = [{k: p[k] for k in keep if k in p} for p in info["participants"]]
    detail = {"info": {"gameVersion": info["gameVersion"], "queueId": info["queueId"], "mapId": info["mapId"],
                       "participants": parts}}
    tl = _load(KR / "timeline" / f"{mid}.json")
    events = []
    for fr in tl["info"]["frames"]:
        for e in fr.get("events") or []:
            if isinstance(e, dict) and "timestamp" in e and "type" in e:
                events.append({"type": e["type"], "timestamp": e["timestamp"]} if e["type"] == "GAME_END" else e)
    return detail, events


def build_index(detail, events, table):
    sm = _extract_static_meta_from_detail(detail)
    return ItemStateIndex(events, table, exact=True, patch=PATCH, champion_by_pid=sm["champion_by_pid"],
                          runes_by_pid=sm["runes_by_pid"], summoner_spells_by_pid=sm["summoner_spells_by_pid"])


# ============================================================================================ causes
def classify(miss, extra, table, champion_id):
    """Descriptive labels of one mismatch (post-hoc; no rule or criterion depends on them)."""
    miss, extra, cats = set(miss), set(extra), set()
    upg = {k for k, v in table.items() if list(v.get("from") or []) == [3867]}
    if (miss & upg) and 3867 in extra:
        cats.add("support_upgrade_unobserved_3867_proxy")
        miss -= upg
        extra.discard(3867)
    if champion_id == VIEGO:
        cats.add("viego")

    def lab(x, side):
        it = table.get(x)
        if it is None:
            return f"id_not_in_{PATCH}_table_{side}"
        if x == 2055:
            return f"control_ward_{side}"
        if x in SUPPORT_CHAIN or list(it.get("from") or []) == [3867]:
            return f"support_chain_{side}"
        if x in JUNGLE_PETS:
            return f"jungle_pet_{side}"
        if x == 2422:
            return f"magical_footwear_{side}"
        if is_boots(it):
            return f"boots_{side}"
        if is_consumable(it):
            return f"consumable_{side}"
        if it.get("into"):
            return f"component_{side}"
        return f"completed_or_other_{side}"
    for x in miss:
        cats.add(lab(x, "missing"))
    for x in extra:
        cats.add(lab(x, "extra"))
    return cats


# ============================================================================================ main check
def run():
    t0 = time.time()
    ids, scan, n_files = select_sample()
    if len(ids) < N_SAMPLE:
        raise SystemExit(f"only {len(ids)} eligible matches")
    EXCLUSIONS.parent.mkdir(parents=True, exist_ok=True)
    EXCLUSIONS.write_text("\n".join(ids) + "\n", encoding="utf-8")
    ids_sha = _sha_hex("\n".join(ids))
    _dump("sample_match_ids_15.18.json", {
        "procedure": "detail file ids ordered by sha256(match id) hex ascending; first 500 with gameVersion 15.18, "
                     "queueId 420, mapId 11 and a timeline file", "n_detail_files": n_files, "scan": scan,
        "n": len(ids), "sha256_of_newline_joined_ids": ids_sha, "exclusion_list": str(EXCLUSIONS), "ids": ids})

    table = load_item_table_v2(PATCH)
    ref = load_item_table_v2(REF_PATCH)
    names = {k: v.get("name", "") for k, v in table.items()}
    assert not uses_rules_16x(PATCH)

    c = Counter()
    role_c = {r: Counter() for r in ROLES}
    cause_players, cause_combo = Counter(), Counter()
    missing_by_id, extra_by_id, ms_missing, ms_extra = Counter(), Counter(), Counter(), Counter()
    item6_pairs, diag_tot, unowned_ids = Counter(), Counter(), Counter()
    champ_mism = Counter()
    ev_types, item_ev_types, pid0_items = Counter(), Counter(), Counter()
    ids_touched = Counter()
    unknown_event_ids, unknown_detail_ids = Counter(), Counter()
    residual, proxy_examples, over6 = [], [], []
    rb_present = Counter()
    slot_dist = Counter()
    upg_ids = {k for k, v in table.items() if list(v.get("from") or []) == [3867]}
    rng = random.Random(CAUSAL_SEED)
    causal_mids = set(rng.sample(ids, CAUSAL_MATCHES))
    causal = Counter()
    causal_fail = []
    for mid in ids:
        detail, events = load_match(mid)
        for e in events:
            ev_types[e["type"]] += 1
            if str(e["type"]).startswith("ITEM_"):
                item_ev_types[e["type"]] += 1
                for f in ("itemId", "beforeId", "afterId"):
                    iid = int(e.get(f, 0) or 0)
                    if iid:
                        ids_touched[iid] += 1
                        if iid not in table:
                            unknown_event_ids[iid] += 1
                if int(e.get("participantId", 0) or 0) == 0:
                    pid0_items[(e["type"], int(e.get("itemId", 0) or 0), int(e["timestamp"]) == 0)] += 1
        idx = build_index(detail, events, table)
        ends = [int(e["timestamp"]) for e in events if e["type"] == "GAME_END"]
        t_end = max(ends) if ends else max(int(e["timestamp"]) for e in events)
        c["game_end_event"] += int(bool(ends))
        for p in detail["info"]["participants"]:
            pid = int(p["participantId"])
            role = ROLES[(pid - 1) % 5]
            diag = {}
            inv = idx.inventory(pid, t_end, diagnostics=diag)
            for k, v in diag.items():
                if k != "unowned_ids":
                    diag_tot[k] += v
            for k, v in diag.get("unowned_ids", {}).items():
                unowned_ids[int(k)] += v
            rep_main = Counter(int(i) for i in inv if not is_trinket(table.get(int(i), {})))
            rep_tr = Counter(int(i) for i in inv if is_trinket(table.get(int(i), {})))
            det_main = Counter(int(p.get(f"item{k}", 0) or 0) for k in range(6) if int(p.get(f"item{k}", 0) or 0))
            it6 = int(p.get("item6", 0) or 0)
            det_tr = Counter([it6] if it6 else [])
            for i in list(det_main) + ([it6] if it6 else []):
                ids_touched[i] += 1
                if i not in table:
                    unknown_detail_ids[i] += 1
            rb = int(p.get("roleBoundItem", 0) or 0)
            rb_present["field_present"] += int("roleBoundItem" in p)
            rb_present["nonzero"] += int(bool(rb))
            c["players"] += 1
            set_ok = set(rep_main) == set(det_main)
            ms_ok = rep_main == det_main
            tr_ok = set(rep_tr) == set(det_tr)
            all7_ok = (set(rep_main) | set(rep_tr)) == (set(det_main) | set(det_tr))
            c["set_ok"] += set_ok
            c["multiset_ok"] += ms_ok
            c["item6_ok"] += tr_ok
            c["all7_ok"] += all7_ok
            role_c[role]["n"] += 1
            role_c[role]["set_ok"] += set_ok
            n_slots = sum(1 if is_consumable(table.get(i, {})) else k for i, k in rep_main.items())
            slot_dist[n_slots] += 1
            if n_slots > 6:                                     # a stack of one consumable id takes one slot
                c["replay_over_6_nontrinket_slots"] += 1
                if len(over6) < 20:
                    over6.append({"match": mid, "pid": pid, "champion": p.get("championName"), "slots": n_slots,
                                  "replay": sorted(inv), "detail_item0_5": sorted(det_main.elements())})
            if not tr_ok:
                item6_pairs[(it6, tuple(sorted(rep_tr.elements())))] += 1
            if not set_ok:
                miss, extra = set(det_main) - set(rep_main), set(rep_main) - set(det_main)
                cats = classify(miss, extra, table, int(p.get("championId", 0) or 0))
                for k in cats:
                    cause_players[k] += 1
                cause_combo[" + ".join(sorted(cats))] += 1
                for i in miss:
                    missing_by_id[i] += 1
                for i in extra:
                    extra_by_id[i] += 1
                champ_mism[str(p.get("championName"))] += 1
                role_c[role]["set_mismatch"] += 1
                only_proxy = cats == {"support_upgrade_unobserved_3867_proxy"}
                # post-hoc diagnostic: the Symbiotic Soles upgrade (a loose destroy of 3010 -> 3013 Synchronized
                # Souls, Data Dragon from=[3010] without specialRecipe) is not a replay rule
                m13, x13 = miss - {3013}, set(extra)
                if (m13 & upg_ids) and 3867 in x13:
                    m13, x13 = m13 - upg_ids, x13 - {3867}
                c["set_mismatch_only_proxy_or_3013"] += not m13 and not x13
                c["set_mismatch_only_3867_proxy"] += only_proxy
                role_c[role]["set_ok_or_proxy"] += only_proxy
                rec = {"match": mid, "pid": pid, "slot_role": role, "champion": p.get("championName"),
                       "causes": sorted(cats),
                       "missing_in_replay": [[i, names.get(i, "?")] for i in sorted(miss)],
                       "extra_in_replay": [[i, names.get(i, "?")] for i in sorted(extra)],
                       "detail_item0_5": sorted(det_main.elements()), "replay": sorted(inv),
                       "diagnostics": {k: v for k, v in diag.items() if v}}
                if only_proxy:
                    if len(proxy_examples) < 5:
                        proxy_examples.append(rec)
                elif len(residual) < 80:
                    residual.append(rec)
            else:
                role_c[role]["set_ok_or_proxy"] += 1
                if not ms_ok:
                    for i, n in (det_main - rep_main).items():
                        ms_missing[i] += n
                    for i, n in (rep_main - det_main).items():
                        ms_extra[i] += n
        if mid in causal_mids:                                  # index from events <= t only vs full index
            t_last = max(int(e["timestamp"]) for e in events)
            for _ in range(CAUSAL_TIMES):
                t = rng.randint(0, t_last)
                cut = build_index(detail, [e for e in events if int(e["timestamp"]) <= t], table)
                for pid in range(1, 11):
                    same = sorted(cut.inventory(pid, t)) == sorted(idx.inventory(pid, t))
                    causal["player_times"] += 1
                    causal["identical"] += same
                    if not same and len(causal_fail) < 10:
                        causal_fail.append({"match": mid, "pid": pid, "t": t})

    n = c["players"]
    rate = c["set_ok"] / n
    proxy_rate = (c["set_ok"] + c["set_mismatch_only_3867_proxy"]) / n
    named = lambda cnt: [[int(i), names.get(int(i), ref.get(int(i), {}).get("name", "?")), v]
                         for i, v in cnt.most_common()]
    # 15.18 vs 15.14 tables, restricted to the ids this sample touches
    table_diff = []
    for i in sorted(ids_touched):
        a, b = ref.get(i), table.get(i)
        if a is None and b is None:
            continue
        if a is None or b is None:
            table_diff.append({"id": i, "name": (a or b).get("name"), "in_15.14": a is not None,
                               "in_15.18": b is not None, "touches": ids_touched[i]})
            continue
        ch = {k: [a.get(k), b.get(k)] for k in ("from", "into", "gold_total", "gold_base", "tags", "stats")
              if a.get(k) != b.get(k)}
        if ch.get("from") is not None or ch.get("into") is not None or "gold_total" in ch or "gold_base" in ch \
                or "tags" in ch:
            table_diff.append({"id": i, "name": b.get("name"), "touches": ids_touched[i], "changed": ch})
    out = {
        "check": "(b) end-of-game inventory, KR Match-V5 15.18 sample: ItemStateIndex(exact=True, patch 15.18) at "
                 "GAME_END vs detail item0-5 (exact set, 99 % rule) and item6 separately",
        "data": str(KR), "patch": PATCH, "queue": QUEUE, "map": MAP, "n_matches": len(ids),
        "match_ids_sha256": ids_sha, "sample_scan": scan, "n_detail_files": n_files,
        "exclusion_list": str(EXCLUSIONS),
        "fields_read": "detail: gameVersion, queueId, mapId, participantId, championId, championName, summoner1Id/2Id,"
                       " perks, item0-6, roleBoundItem; timeline: frames[].events (GAME_END: type + timestamp). "
                       "Outcome keys dropped at parse: " + ", ".join(sorted(OUTCOME_KEYS)),
        "rules_16x_active": False,
        "n_players": n, "matches_with_game_end_event": c["game_end_event"],
        "agreement_exact_set_item0_5": rate,
        "pass_99pct_rule_exact_set": rate >= PASS_RATE,
        "agreement_exact_multiset_item0_5": c["multiset_ok"] / n,
        "agreement_item6_trinket": c["item6_ok"] / n,
        "agreement_set_item0_6_vs_whole_replay": c["all7_ok"] / n,
        "diagnostic_agreement_3867_proxy_counted_as_its_upgrade": proxy_rate,
        "diagnostic_pass_99pct_if_proxy_counted": proxy_rate >= PASS_RATE,
        "diagnostic_agreement_3867_proxy_and_3010_to_3013_counted": (c["set_ok"] + c["set_mismatch_only_proxy_or_3013"]) / n,
        "diagnostic_note": "post-hoc, not the 99 % rule: (i) the 3867 proxy counted as the detail's support upgrade; "
                           "(ii) additionally the unmodelled Symbiotic Soles upgrade (3010 destroyed without a combine "
                           "-> 3013) counted as a match",
        "n_set_mismatch_players": n - c["set_ok"],
        "n_set_mismatch_only_3867_proxy": c["set_mismatch_only_3867_proxy"],
        "by_slot_role": {r: {"n": v["n"], "agreement_exact_set": v["set_ok"] / max(1, v["n"]),
                             "agreement_proxy_counted": v["set_ok_or_proxy"] / max(1, v["n"])}
                         for r, v in role_c.items()},
        "cause_labels_note": "descriptive post-hoc labels per missing / extra id (classify); no rule or pass "
                             "criterion depends on them",
        "set_mismatch_players_by_cause": dict(cause_players.most_common()),
        "set_mismatch_players_by_cause_combination": dict(cause_combo.most_common(40)),
        "set_mismatch_missing_in_replay_by_id": named(missing_by_id),
        "set_mismatch_extra_in_replay_by_id": named(extra_by_id),
        "set_mismatch_by_champion": dict(champ_mism.most_common(30)),
        "multiset_only_missing_by_id": named(ms_missing), "multiset_only_extra_by_id": named(ms_extra),
        "item6_mismatch_pairs_detail_vs_replay_trinkets": [[k[0], names.get(k[0], "?"), list(k[1]), v]
                                                            for k, v in item6_pairs.most_common(30)],
        "slot_feasibility_players_replay_over_6_nontrinket_slots": c["replay_over_6_nontrinket_slots"],
        "slot_feasibility_rule": "non-trinket replay items; a consumable id held more than once takes one slot "
                                 "(ev4_s1_item_gold.py slots convention)",
        "slot_count_distribution": {str(k): v for k, v in sorted(slot_dist.items())},
        "slot_feasibility_examples": over6,
        "replay_diagnostics_totals": dict(diag_tot),
        "replay_unowned_removal_ids": named(unowned_ids),
        "causality_spot_check": {"matches": CAUSAL_MATCHES, "times_per_match": CAUSAL_TIMES, **dict(causal),
                                 "failures": causal_fail},
        "patch_mechanics_15_18": {
            "event_types": dict(ev_types.most_common()),
            "item_event_types": dict(item_ev_types),
            "participant0_item_events_type_item_at0ms": [[k[0], k[1], k[2], v] for k, v in pid0_items.most_common()],
            "event_item_ids_not_in_15.18_table": named(unknown_event_ids),
            "detail_item_ids_not_in_15.18_table": named(unknown_detail_ids),
            "roleBoundItem": dict(rb_present),
            "table_diff_15.14_vs_15.18_ids_touched_by_sample": table_diff,
        },
        "residual_examples_not_proxy": residual, "proxy_examples": proxy_examples,
        "script": "scripts/exact_v4/ev4_s1_kr_inventory.py", "elapsed_s": round(time.time() - t0, 1),
    }
    _dump("inventory_end_of_game_15.18.json", out)
    brief = {k: out[k] for k in ("n_matches", "n_players", "agreement_exact_set_item0_5", "pass_99pct_rule_exact_set",
                                 "agreement_exact_multiset_item0_5", "agreement_item6_trinket",
                                 "agreement_set_item0_6_vs_whole_replay",
                                 "diagnostic_agreement_3867_proxy_counted_as_its_upgrade", "n_set_mismatch_players",
                                 "n_set_mismatch_only_3867_proxy", "diagnostic_agreement_3867_proxy_and_3010_to_3013_counted",
                                 "slot_feasibility_players_replay_over_6_nontrinket_slots", "slot_count_distribution",
                                 "replay_diagnostics_totals", "by_slot_role", "sample_scan")}
    brief |= {"causes": out["set_mismatch_players_by_cause"], "top_missing": out["set_mismatch_missing_in_replay_by_id"][:15],
              "top_extra": out["set_mismatch_extra_in_replay_by_id"][:15], "item6": out["item6_mismatch_pairs_detail_vs_replay_trinkets"][:10],
              "unowned": out["replay_unowned_removal_ids"][:15], "causal": out["causality_spot_check"],
              "unknown_event_ids": out["patch_mechanics_15_18"]["event_item_ids_not_in_15.18_table"][:10],
              "pid0": out["patch_mechanics_15_18"]["participant0_item_events_type_item_at0ms"],
              "item_event_types": out["patch_mechanics_15_18"]["item_event_types"],
              "roleBoundItem": out["patch_mechanics_15_18"]["roleBoundItem"]}
    return brief


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, ensure_ascii=False, default=str))
