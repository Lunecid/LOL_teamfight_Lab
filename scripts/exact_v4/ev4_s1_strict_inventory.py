"""Stage 1 strict end-of-game inventory validation (v4-exact, task VALIDATE).  Single process.

Author decision records/items_strict_rule_20260925T125703Z.json: "strict item-ID set equality for item0-5 AND item6
(trinket), per player, at GAME_END, with exactly one exception class: {3867, 3869, 3870, 3871, 3876, 3877} are treated
as one item"; the 99 % rule applies; "a fresh 500-match KR 15.18 sample, disjoint from the first; added to the
replication exclusion list".  This script adds no replay rule: it only measures gameplay/item_state.py as it is.

Samples (all fixed before the first run):
  fresh    KR Match-V5 raw (C:/Users/todtj/PycharmProjects/Lol_project/data/raw/matches/kr).  Detail file ids ordered by
           sha256(match id) hex ascending (the first sample's order, ev4_s1_kr_inventory.select_sample); a match is
           eligible when detail info.gameVersion normalises to 15.18, queueId == 420, mapId == 11 and its timeline
           exists (byte regex pre-screen, confirmed on the parsed fields, as in the first sample); eligible ids in
           records/replication_exclusions_inventory_1518.txt (the first sample) are skipped; the next 500 are taken.
           Their ids -> records/replication_exclusions_inventory_1518_fresh.txt (new file, one id per line, trailing
           newline; the first list is not touched).  On a re-run the file must hold exactly the same ids.
           THE OUT-OF-SAMPLE TEST: the 15.x rules (config/game_rules/item_rules_15x.json) were derived on 15.14
           events and the first KR sample, never on this one.
  first    the first KR 15.18 sample (records/replication_exclusions_inventory_1518.txt, 500 ids) - the
           RULE-DERIVATION SAMPLE, reported for comparison only (in-sample, not a clean test).
  16.15    all 553 matches of D:/LOL_Project/fusion_2615/raw/kr with a timeline (the ev4_s1x_items_16x.py set), 16.x
           rules on.
Data handling.  Every raw file (KR and 16.15) is parsed with ev4_s1_kr_inventory._hook, the object_pairs_hook that
  drops every outcome key (win, winningTeam, teams, surrender / nexus fields, ...) at parse time; the kept fields are
  those of ev4_s1_kr_inventory.load_match: gameVersion / queueId / mapId, participantId, championId / Name, pre-game
  summoner spells, perks, item0-6, roleBoundItem; timeline events (GAME_END reduced to type + timestamp).  No outcome
  field is read, counted or printed.  No cache data is loaded.
Replay: ItemStateIndex(events, table, exact=True, patch=<match patch>, champion_by_pid / runes_by_pid /
  summoner_spells_by_pid from data.cache_io._extract_static_meta_from_detail) at the GAME_END timestamp (fallback: the
  last event); table = Data Dragon items of the match patch.  Default rules by patch (15.18 -> 15.x rules,
  16.15 -> 16.x rules).  A reference replay with the patch rules switched off (rules_15x=False / rules_16x=False) is
  reported next to it (not the measure).
Measure per player (the author rule):
  trinket     = Data Dragon tag 'Trinket' of the match-patch table (includes 3513 / 1104 Eye of the Herald);
  norm(x)     = x, except every id of SUPPORT_CLASS -> one token;
  item0_5     : set(norm(detail item0-5 non-zero)) == set(norm(replay non-trinket ids));
                16.15: detail side = item0-5 + roleBoundItem (non-zero, non-trinket), because 16.x has a separate
                role slot (the ev4_s1x_items_16x.py primary); item0-5 alone is reported as a variant;
  item6       : set(replay trinket ids) == {item6} (both empty when item6 == 0); a replay holding two trinkets fails;
  strict      = item0_5 AND item6  -> the 99 % rule (share of players passing >= 0.99);
  variants    : multiset (norm, with multiplicity; the detail shows a stack in one slot, so stacked consumables fail
                it by construction - reported, not scored), raw ids (no support class), per slot role (pid order,
                TOP JUNGLE MIDDLE BOTTOM UTILITY), 95 % Wilson interval (players independent) and a match-cluster
                bootstrap interval (2,000 resamples of matches, seed 2026092540).
Mismatch causes: every failing player is listed (match, pid, champion, role, missing / extra ids with names, item6
  pair) with descriptive post-hoc labels per id (control ward, jungle pet, Viego, consumable, ...) and context flags
  (Viego, no own ITEM_* event, game shorter than 5 min).  Labels explain; they change no rule or criterion.
Outputs: <OUT> = stage1/validation/strict/: strict_inventory_fresh_kr_15.18.json,
  strict_inventory_first_kr_15.18_rule_derivation.json, strict_inventory_16.15.json, sample_fresh_kr_15.18.json,
  summary.json.

Usage: python scripts/exact_v4/ev4_s1_strict_inventory.py [fresh|first|1615|all] [--out DIR]
  --out DIR  write the outputs to DIR instead of <OUT> (the fresh id list in records/ is still checked, never
             rewritten when it exists).  Used for stage1/validation/strict_noR155/ (2026-09-25: R15-5 removed by author
             decision records/items_confirmations_20260925T134510Z.json; the fresh result changes no rule).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/exact_v4"))

import ev4_s1_kr_inventory as K  # noqa: E402  (outcome-stripping hook, KR loader, sample regexes)
from data.cache_io import _extract_static_meta_from_detail  # noqa: E402
from gameplay.fights import normalize_patch  # noqa: E402
from gameplay.item_state import (  # noqa: E402
    ItemStateIndex, is_boots, is_consumable, is_trinket, load_item_table_v2, uses_rules_15x, uses_rules_16x,
)

BASE = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925")
OUT = BASE / "stage1/validation/strict"
FIRST_LIST = BASE / "records/replication_exclusions_inventory_1518.txt"
FRESH_LIST = BASE / "records/replication_exclusions_inventory_1518_fresh.txt"
KR = K.KR
RAW_1615 = Path("D:/LOL_Project/fusion_2615/raw/kr")
N_FRESH = 500
PASS_RATE = 0.99
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
SUPPORT_CLASS = frozenset({3867, 3869, 3870, 3871, 3876, 3877})
SUPPORT_TOKEN = "support_upgraded_class"
SUPPORT_CHAIN_LOW = frozenset({3865, 3866})
JUNGLE_PETS = frozenset({1101, 1102, 1103, 1105, 1106, 1107})
ROLE_QUEST_IDS = frozenset(range(1200, 1223))
VIEGO = 234
BOOT_SEED, BOOT_N = 2026092540, 2000
SHORT_GAME_MS = 300_000


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _sha(s) -> str:
    return hashlib.sha256(s if isinstance(s, bytes) else s.encode("utf-8")).hexdigest()


def _read_ids(path: Path):
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(ids) == len(set(ids)), path
    return ids


def _list_hashes(path: Path, ids):
    return {"path": str(path), "n": len(ids), "sha256_newline_joined_ids": _sha("\n".join(ids)),
            "sha256_file_bytes": _sha(path.read_bytes())}


# ============================================================================================ fresh sample
def select_fresh():
    first = _read_ids(FIRST_LIST)
    first_set = set(first)
    assert len(first) == 500
    ids = [f[:-5] for f in os.listdir(KR / "detail") if f.endswith(".json")]
    ids.sort(key=K._sha_hex)
    scan = Counter()
    taken, first_seen = [], []
    for mid in ids:
        if len(taken) >= N_FRESH:
            break
        scan["scanned"] += 1
        b = (KR / "detail" / f"{mid}.json").read_bytes()
        v, q, m = K.RX_VER.search(b), K.RX_Q.search(b), K.RX_MAP.search(b)
        if not (v and q and m):
            scan["skip_fields_missing"] += 1
            continue
        if normalize_patch(v.group(1).decode()) != K.PATCH:
            scan["skip_other_patch"] += 1
            continue
        if int(q.group(1)) != K.QUEUE or int(m.group(1)) != K.MAP:
            scan["skip_queue_or_map"] += 1
            continue
        if not (KR / "timeline" / f"{mid}.json").exists():
            scan["skip_no_timeline"] += 1
            continue
        info = K._load(KR / "detail" / f"{mid}.json")["info"]
        if (normalize_patch(str(info["gameVersion"])) != K.PATCH or int(info["queueId"]) != K.QUEUE
                or int(info["mapId"]) != K.MAP):
            scan["skip_parsed_mismatch"] += 1
            continue
        if mid in first_set:
            scan["skip_in_first_sample"] += 1
            first_seen.append(mid)
            continue
        taken.append(mid)
    if len(taken) < N_FRESH:
        raise SystemExit(f"only {len(taken)} fresh eligible matches")
    # the first sample is the first 500 eligible ids of the same walk: all of them come before any fresh id
    assert first_seen == first, "first sample is not the prefix of the eligible walk"
    assert not set(taken) & first_set
    if FRESH_LIST.exists():
        prev = _read_ids(FRESH_LIST)
        assert prev == taken, "existing fresh list differs from the re-derived sample"
    else:
        FRESH_LIST.parent.mkdir(parents=True, exist_ok=True)
        FRESH_LIST.write_text("\n".join(taken) + "\n", encoding="utf-8")
    info = {"procedure": "KR detail ids ordered by sha256(match id) hex ascending; eligible = gameVersion 15.18, "
                         "queueId 420, mapId 11, timeline present (regex pre-screen, confirmed parsed); eligible ids of "
                         "the first sample skipped; next 500 taken",
            "n_detail_files": len(ids), "scan": dict(scan),
            "first_sample_is_prefix_of_eligible_walk": True,
            "fresh_list": _list_hashes(FRESH_LIST, taken), "first_list_unchanged": _list_hashes(FIRST_LIST, first),
            "overlap_with_first": 0, "ids": taken}
    _dump("sample_fresh_kr_15.18.json", info)
    return taken, info


# ============================================================================================ loading
def load_1615(mid):
    """16.15 raw through the same outcome-stripping hook and the same kept fields as K.load_match."""
    raw = K._load(RAW_1615 / "detail" / f"{mid}.json")
    info = raw["info"]
    keep = ("participantId", "championId", "championName", "summoner1Id", "summoner2Id", "perks",
            *(f"item{k}" for k in range(7)), "roleBoundItem")
    parts = [{k: p[k] for k in keep if k in p} for p in info["participants"]]
    detail = {"info": {"gameVersion": info["gameVersion"], "queueId": info.get("queueId"), "mapId": info.get("mapId"),
                       "participants": parts}}
    tl = K._load(RAW_1615 / "timeline" / f"{mid}.json")
    events = []
    for fr in tl["info"]["frames"]:
        for e in fr.get("events") or []:
            if isinstance(e, dict) and "timestamp" in e and "type" in e:
                events.append({"type": e["type"], "timestamp": e["timestamp"]} if e["type"] == "GAME_END" else e)
    return detail, events


def ids_1615():
    return sorted(p.stem for p in (RAW_1615 / "detail").glob("*.json") if (RAW_1615 / "timeline" / p.name).exists())


# ============================================================================================ measure
def _norm(i):
    return SUPPORT_TOKEN if i in SUPPORT_CLASS else i


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 5), round(c + h, 5)]


def _cluster_boot(per_match, seed=BOOT_SEED, reps=BOOT_N):
    """per_match: list of (passes, players); percentile 2.5 / 97.5 of the pooled share over match resamples."""
    rng = random.Random(seed)
    m = len(per_match)
    vals = []
    for _ in range(reps):
        k = n = 0
        for _ in range(m):
            a, b = per_match[rng.randrange(m)]
            k += a
            n += b
        vals.append(k / n)
    vals.sort()
    return [round(vals[int(0.025 * reps)], 5), round(vals[int(0.975 * reps) - 1], 5)]


def _label(i, side, table, champion_id):
    if isinstance(i, str):
        return f"support_upgraded_class_{side}"
    it = table.get(i)
    pre = "viego:" if champion_id == VIEGO else ""
    if it is None:
        return f"{pre}id_not_in_table_{side}"
    if i == 2055:
        return f"{pre}control_ward_{side}"
    if i in JUNGLE_PETS:
        return f"{pre}jungle_pet_{side}"
    if i in ROLE_QUEST_IDS:
        return f"{pre}role_quest_item_{side}"
    if i in SUPPORT_CHAIN_LOW:
        return f"{pre}support_chain_{side}"
    if i == 2422:
        return f"{pre}magical_footwear_{side}"
    if i == 2010:
        return f"{pre}biscuit_{side}"
    if i in (2150, 2151, 2152, 2138, 2139, 2140):
        return f"{pre}elixir_{side}"
    if is_boots(it):
        return f"{pre}boots_{side}"
    if is_consumable(it):
        return f"{pre}consumable_{side}"
    if it.get("into"):
        return f"{pre}component_{side}"
    return f"{pre}completed_or_other_{side}"


def evaluate(label, match_ids, loader, patch_expected, rules_note, sample_role):
    t0 = time.time()
    table = load_item_table_v2(patch_expected)
    names = {k: v.get("name", "") for k, v in table.items()}
    trink = lambda i: is_trinket(table.get(int(i), {}))  # noqa: E731
    is16 = uses_rules_16x(patch_expected)
    assert is16 or uses_rules_15x(patch_expected)
    variants = ("rules", "reference_rules_off")
    C = {v: Counter() for v in variants}
    R = {v: defaultdict(Counter) for v in variants}
    per_match = {v: [] for v in variants}
    mism_players, cause_labels, cause_combo = [], Counter(), Counter()
    miss_ids, extra_ids, item6_pairs, ms_only = Counter(), Counter(), Counter(), Counter()
    flags_c = Counter()
    meta = Counter()
    patches = Counter()
    for mid in match_ids:
        detail, events = loader(mid)
        info = detail["info"]
        patch = normalize_patch(str(info["gameVersion"]))
        patches[patch] += 1
        assert patch == patch_expected, (mid, patch)
        meta[f"queue_{info.get('queueId')}_map_{info.get('mapId')}"] += 1
        sm = _extract_static_meta_from_detail(detail)
        kw = dict(exact=True, patch=patch, champion_by_pid=sm["champion_by_pid"], runes_by_pid=sm["runes_by_pid"],
                  summoner_spells_by_pid=sm["summoner_spells_by_pid"])
        off = {"rules_16x": False} if is16 else {"rules_15x": False}
        idx = {"rules": ItemStateIndex(events, table, **kw), "reference_rules_off": ItemStateIndex(events, table, **kw, **off)}
        if is16:
            assert idx["rules"].rules16 is not None and idx["reference_rules_off"].rules16 is None
        else:
            assert idx["rules"].rules15 is not None and idx["reference_rules_off"].rules15 is None
        ends = [int(e["timestamp"]) for e in events if e["type"] == "GAME_END"]
        meta["game_end_event"] += bool(ends)
        t_end = max(ends) if ends else max(int(e["timestamp"]) for e in events)
        own_item_events = Counter(int(e.get("participantId", 0) or 0) for e in events
                                  if str(e["type"]).startswith("ITEM_"))
        pm = {v: [0, 0] for v in variants}
        for p in info["participants"]:
            pid = int(p["participantId"])
            role = ROLES[(pid - 1) % 5]
            cid = int(p.get("championId", 0) or 0)
            d05 = [int(p.get(f"item{k}", 0) or 0) for k in range(6)]
            d05 = [i for i in d05 if i]
            meta["detail_item0_5_trinket_ids"] += sum(trink(i) for i in d05)
            rb = int(p.get("roleBoundItem", 0) or 0)
            meta["roleBoundItem_nonzero"] += bool(rb)
            det_main = list(d05)
            if is16 and rb and not trink(rb):
                det_main.append(rb)
            it6 = int(p.get("item6", 0) or 0)
            det_tr = {it6} if it6 else set()
            for vn, ix in idx.items():
                inv = [int(i) for i in ix.inventory(pid, t_end)]
                rep_main = [i for i in inv if not trink(i)]
                rep_tr = [i for i in inv if trink(i)]
                sr, sd = {_norm(i) for i in rep_main}, {_norm(i) for i in det_main}
                ok05 = sr == sd
                ok6 = set(rep_tr) == det_tr and len(rep_tr) == len(det_tr)
                ok = ok05 and ok6
                okms = Counter(_norm(i) for i in rep_main) == Counter(_norm(i) for i in det_main)
                c = C[vn]
                c["players"] += 1
                c["strict_item0_6"] += ok
                c["item0_5"] += ok05
                c["item6"] += ok6
                c["multiset_item0_6"] += okms and ok6
                c["raw_ids_item0_6"] += (set(rep_main) == set(det_main)) and ok6
                if is16:
                    c["item0_5_only_no_roleBoundItem"] += {_norm(i) for i in rep_main} == {_norm(i) for i in d05}
                    c["rbq_item0_6"] += ok6 and ({_norm(i) for i in rep_main if i not in ROLE_QUEST_IDS}
                                                 == {_norm(i) for i in det_main if i not in ROLE_QUEST_IDS})
                R[vn][role]["n"] += 1
                R[vn][role]["strict_item0_6"] += ok
                R[vn][role]["item0_5"] += ok05
                R[vn][role]["item6"] += ok6
                pm[vn][0] += ok
                pm[vn][1] += 1
                if vn != "rules":
                    continue
                if ok and not okms:
                    for i, k in (Counter(_norm(i) for i in det_main) - Counter(_norm(i) for i in rep_main)).items():
                        ms_only[f"detail_more:{i}:{names.get(i, '') if isinstance(i, int) else ''}"] += k
                    for i, k in (Counter(_norm(i) for i in rep_main) - Counter(_norm(i) for i in det_main)).items():
                        ms_only[f"replay_more:{i}:{names.get(i, '') if isinstance(i, int) else ''}"] += k
                if ok:
                    continue
                miss, extra = sd - sr, sr - sd
                labels = set()
                for i in miss:
                    labels.add(_label(i, "missing_in_replay", table, cid))
                    miss_ids[i] += 1
                for i in extra:
                    labels.add(_label(i, "extra_in_replay", table, cid))
                    extra_ids[i] += 1
                if not ok6:
                    pair = (it6, tuple(sorted(rep_tr)))
                    item6_pairs[pair] += 1
                    if it6 == 0:
                        labels.add("item6_detail_empty_replay_has_trinket")
                    elif not rep_tr:
                        labels.add("item6_replay_no_trinket")
                    elif len(rep_tr) > 1:
                        labels.add("item6_replay_two_trinkets")
                    else:
                        labels.add("item6_different_trinket")
                flags = []
                if cid == VIEGO:
                    flags.append("viego")
                if own_item_events.get(pid, 0) == 0:
                    flags.append("no_own_item_event")
                if t_end < SHORT_GAME_MS:
                    flags.append("game_under_5min")
                for f in flags:
                    flags_c[f] += 1
                for lb in labels:
                    cause_labels[lb] += 1
                cause_combo[" + ".join(sorted(labels)) + (f"  [{', '.join(flags)}]" if flags else "")] += 1
                mism_players.append({
                    "match": mid, "pid": pid, "slot_role": role, "champion": p.get("championName"),
                    "game_end_s": round(t_end / 1000, 1), "own_item_events": own_item_events.get(pid, 0),
                    "flags": flags, "labels": sorted(labels),
                    "missing_in_replay": [[i, names.get(i, "?") if isinstance(i, int) else "support class"]
                                          for i in sorted(miss, key=str)],
                    "extra_in_replay": [[i, names.get(i, "?") if isinstance(i, int) else "support class"]
                                        for i in sorted(extra, key=str)],
                    "item6_detail": [it6, names.get(it6, "")], "replay_trinkets": [[i, names.get(i, "?")] for i in rep_tr],
                    "detail_item0_5": d05, "detail_roleBoundItem": rb, "replay": sorted(inv)})
        for vn in variants:
            per_match[vn].append(tuple(pm[vn]))

    def summ(vn):
        c = C[vn]
        n = c["players"]
        k = c["strict_item0_6"]
        out = {"n_players": n, "strict_item0_6_share": k / n, "n_fail": n - k,
               "pass_99pct_rule": k / n >= PASS_RATE, "wilson95": _wilson(k, n),
               "match_cluster_bootstrap95": _cluster_boot(per_match[vn]),
               "item0_5_share": c["item0_5"] / n, "item6_share": c["item6"] / n,
               "multiset_item0_6_share": c["multiset_item0_6"] / n,
               "raw_ids_no_support_class_item0_6_share": c["raw_ids_item0_6"] / n,
               "by_slot_role": {r: {"n": R[vn][r]["n"],
                                    **{m: R[vn][r][m] / max(1, R[vn][r]["n"]) for m in ("strict_item0_6", "item0_5", "item6")}}
                                for r in ROLES}}
        if is16:
            out["item0_5_only_no_roleBoundItem_share"] = c["item0_5_only_no_roleBoundItem"] / n
            out["role_quest_ids_removed_item0_6_share"] = c["rbq_item0_6"] / n
        return out

    named = lambda cnt: [[i, names.get(i, "?") if isinstance(i, int) else "support class", v]  # noqa: E731
                         for i, v in cnt.most_common()]
    ids_sha = _sha("\n".join(match_ids))
    res = {
        "sample": label, "sample_role": sample_role, "patch": patch_expected, "rules": rules_note,
        "n_matches": len(match_ids), "match_ids_sha256_newline_joined": ids_sha,
        "patches": dict(patches), "match_meta": dict(meta),
        "measure": "per player at GAME_END: set(norm(detail item0-5" + (" + roleBoundItem" if is16 else "")
                   + ")) == set(norm(replay non-trinket ids)) AND set(replay trinket ids) == {item6} (both empty when "
                   "item6 == 0); norm maps {3867, 3869, 3870, 3871, 3876, 3877} to one token; 99 % rule on the "
                   "share of players passing",
        "result": summ("rules"),
        "reference_rules_off": summ("reference_rules_off"),
        "mismatch_cause_labels_note": "descriptive post-hoc labels (per missing / extra id and item6 pair); "
                                      "one player can carry several; no rule or criterion depends on them",
        "mismatch_players_by_label": dict(cause_labels.most_common()),
        "mismatch_players_by_label_combination": dict(cause_combo.most_common()),
        "mismatch_context_flags": dict(flags_c),
        "mismatch_missing_in_replay_by_id": named(miss_ids),
        "mismatch_extra_in_replay_by_id": named(extra_ids),
        "mismatch_item6_pairs_detail_vs_replay": [[k[0], names.get(k[0], ""), list(k[1]), v]
                                                   for k, v in item6_pairs.most_common()],
        "multiset_only_failures_among_strict_passes": dict(ms_only.most_common()),
        "mismatch_players": mism_players,
        "script": "scripts/exact_v4/ev4_s1_strict_inventory.py", "elapsed_s": round(time.time() - t0, 1),
    }
    return res


def _brief(res):
    r, o = res["result"], res["reference_rules_off"]
    keys = ("strict_item0_6_share", "n_fail", "pass_99pct_rule", "wilson95", "match_cluster_bootstrap95",
            "item0_5_share", "item6_share", "multiset_item0_6_share", "raw_ids_no_support_class_item0_6_share",
            "item0_5_only_no_roleBoundItem_share", "role_quest_ids_removed_item0_6_share")
    return {"sample": res["sample"], "sample_role": res["sample_role"], "n_matches": res["n_matches"],
            "n_players": r["n_players"], "match_ids_sha256": res["match_ids_sha256_newline_joined"],
            **{k: r[k] for k in keys if k in r},
            "by_slot_role_strict": {k: round(v["strict_item0_6"], 4) for k, v in r["by_slot_role"].items()},
            "reference_rules_off": {k: o[k] for k in ("strict_item0_6_share", "item0_5_share", "item6_share")},
            "mismatch_players_by_label": res["mismatch_players_by_label"],
            "mismatch_players_by_label_combination": res["mismatch_players_by_label_combination"],
            "mismatch_context_flags": res["mismatch_context_flags"],
            "item6_pairs": res["mismatch_item6_pairs_detail_vs_replay"],
            "multiset_only": res["multiset_only_failures_among_strict_passes"]}


# ============================================================================================ runs
def run_fresh():
    ids, info = select_fresh()
    res = evaluate("fresh KR 15.18 (500, disjoint from the first sample)", ids, lambda m: K.load_match(m), "15.18",
                   "15.x rules (item_rules_15x.json) as derived on 15.14 + the first KR sample", "OUT-OF-SAMPLE TEST")
    res["fresh_list"] = info["fresh_list"]
    res["first_list_unchanged"] = info["first_list_unchanged"]
    res["selection_scan"] = info["scan"]
    _dump("strict_inventory_fresh_kr_15.18.json", res)
    return _brief(res) | {"fresh_list": info["fresh_list"], "first_list": info["first_list_unchanged"],
                          "scan": info["scan"]}


def run_first():
    ids = _read_ids(FIRST_LIST)
    res = evaluate("first KR 15.18 (500)", ids, lambda m: K.load_match(m), "15.18",
                   "15.x rules (item_rules_15x.json)",
                   "RULE-DERIVATION SAMPLE (in-sample; the 15.x rules were derived on it - not a clean test)")
    res["first_list"] = _list_hashes(FIRST_LIST, ids)
    _dump("strict_inventory_first_kr_15.18_rule_derivation.json", res)
    return _brief(res)


def run_1615():
    ids = ids_1615()
    res = evaluate("16.15 KR raw (all matches with a timeline)", ids, load_1615, "16.15",
                   "16.x rules (item_rules_16x.json)",
                   "16.x validation set (the 16.x rules were derived on these 553 matches: in-sample for 16.x)")
    _dump("strict_inventory_16.15.json", res)
    return _brief(res)


def main(which="all"):
    out = {}
    for name, fn in (("fresh", run_fresh), ("first", run_first), ("1615", run_1615)):
        if which in (name, "all"):
            out[name] = fn()
    if which == "all":
        _dump("summary.json", out)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--out" in args:
        k = args.index("--out")
        OUT = Path(args[k + 1])
        del args[k:k + 2]
    main(args[0] if args else "all")
