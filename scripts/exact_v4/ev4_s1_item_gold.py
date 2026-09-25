"""Stage 1 item-replay checks on 15.14 (v4-exact, author decision 2026-09-25 (a) and the slot check).  Single process.

Reads only the 15.14 cache matches (patch index == '15.14'); no other patch is listed, loaded or counted.

gold   (author's 99 % check a)  For every player p and every minute frame j of a match:
         replayed net spending S(p, T_j) = sum of purchases at the charged price (item_state.purchase_price, via
         ItemStateIndex.replay (replay_inventory(exact=True): Viego rule with takedown window, Magical Footwear
         grant, World Atlas slot rule) - sell gold (Data Dragon gold_sell) - undo refunds (event goldGain); free grants and
         transformations 0.  Events with timestamp <= T_j only.
         observed O(p, T_j) = totalGold - currentGold of the frame, recovered from node_minute:
           totalGold = round(totalGold_norm * 25 000), currentGold = round(curGold_norm * 4 000)
           (gameplay.pipeline_cache DEN_TOT_G / DEN_CUR_G; clip ceilings 125 000 / 20 000 gold are never reached,
           checked below).  float32 keeps ~7 significant digits, so the stored value is within 0.01 gold of the
           integer frame value and rounding recovers it exactly.
         error E = S - O.  Tolerance |E| <= 1 gold: the frame shows totalGold and currentGold as two separately
           rounded integers of an internal fractional gold value (passive income is fractional), so their
           difference can be one off even when the replay is exact; nothing larger is tolerated.
         Three observed-side variants are reported (the replay is the same in all):
           raw            O as above;
           ward_adjusted  O - 5 * (WARD_KILL events with killerId p, timestamp <= T_j, wardType in
                          CONTROL_WARD / SIGHT_WARD / YELLOW_TRINKET).  Evidence (first 40 rule-check matches,
                          least squares of the per-minute change of E on ward kills by type): -4.95 / -5.00 / -5.03
                          gold per kill; BLUE_TRINKET -1.5 (about half of the kills move the frame by 5, half by 0;
                          no time pattern) and UNDEFINED 0.3, so those are not adjusted.  A killed ward adds 5 to
                          totalGold - currentGold without any shop event; it is frame bookkeeping, not item spending.
           ward_bounded   ward_adjusted, and in addition each of the b BLUE_TRINKET / UNDEFINED ward kills <= T_j
                          may or may not have added 5 (not identifiable from the events): the frame is within
                          tolerance when E + 5k is within +-1 for some k in 0..b.
         Samples: 'rule_check' = the 300 matches (seed 20260925) the replay rules were derived on (the sample the
         author named); 'held_out' = the 300 validation matches of validate_item_stats.py (seed 20260926, disjoint).
         Causes: each player-frame outside tolerance is attributed to the minute interval in which |E| first left
         the tolerance band (onset) and classified by the player's events in that interval; and, independently, by
         the error value: 'abs_error_2' (|E| = 2), 'ward_kill_without_5' (E > 1 and within +-1 of 5k for
         1 <= k <= the player's adjusted ward kills <= T_j: kills that did not move the frame), 'other' (listed in
         full: the misses that can come from the item replay).
slots  end-of-game slot feasibility on both samples and on the completion reviewer's 200 matches (seed 424242,
         46/2000 players over 6 before the fixes): the replayed inventory at the last event of the match has at
         most 6 non-trinket slots (Data Dragon tag 'Trinket' excluded; a stackable consumable (potions, wards,
         elixirs, 2010 biscuits) occupies one slot per id).  Reported: players with > 6, and the item ids involved.

Usage: python scripts/exact_v4/ev4_s1_item_gold.py [gold|slots|all]
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.config import NODE_FEATURE_NAMES  # noqa: E402
from gameplay.item_state import (  # noqa: E402
    ItemStateIndex, _static_meta, is_consumable, load_item_table_v2,
)

OUT = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925/stage1/item_state")
CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
PATCH_INDEX = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json")
PATCH = "15.14"
N_MATCHES = 300
DEN_TOT_G, DEN_CUR_G, CLIP = 25_000.0, 4_000.0, 5.0
TOL = 1
WARD_GOLD = 5
WARD_TYPES_ADJUSTED = ("CONTROL_WARD", "SIGHT_WARD", "YELLOW_TRINKET")


def samples():
    idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in idx.items() if v == PATCH)
    a = list(ids)
    random.Random(20260925).shuffle(a)
    rule = a[:N_MATCHES]
    rs = set(rule)
    rest = [i for i in ids if i not in rs]
    random.Random(20260926).shuffle(rest)
    return {"rule_check": rule, "held_out": rest[:N_MATCHES]}


def load(m):
    meta = json.loads((CACHE / f"{m}.meta.json").read_text(encoding="utf-8"))
    if str(meta["patch"]) != PATCH:
        raise RuntimeError(f"{m} is not {PATCH}")
    events = json.loads((CACHE / f"{m}.events.json").read_text(encoding="utf-8"))
    z = np.load(CACHE / f"{m}.npz")
    return meta, events, np.asarray(z["minute_ts"], np.int64), np.asarray(z["node_minute"])


def classify(ev_int, pid, champion, grants_in, table):
    """Class of one minute interval from the player's events in it (first matching rule wins)."""
    types = Counter()
    for e in ev_int:
        t = e["type"]
        if t == "WARD_KILL":
            types["ward_kill_" + str(e.get("wardType"))] += 1
        elif t.startswith("ITEM"):
            types[t] += 1
            if 2422 in (e.get("itemId"), e.get("beforeId"), e.get("afterId")):
                types["item_2422"] += 1
            if e.get("itemId") in (3865, 3866, 3867):
                types["atlas_family"] += 1
    if champion == 234 and any(k.startswith("ITEM") for k in types):
        return "viego_item_event"
    if types["item_2422"] or grants_in:
        return "magical_footwear"
    if types["ITEM_UNDO"]:
        return "undo"
    if types["ITEM_SOLD"]:
        return "sell"
    if types["atlas_family"]:
        return "world_atlas_family"
    if types["ITEM_PURCHASED"] and types["ITEM_DESTROYED"]:
        return "purchase_with_destroy"
    if types["ITEM_PURCHASED"]:
        return "purchase_only"
    if types["ward_kill_BLUE_TRINKET"] or types["ward_kill_UNDEFINED"]:
        return "ward_kill_blue_or_undefined"
    if any(k.startswith("ward_kill") for k in types):
        return "ward_kill_other"
    if types["ITEM_DESTROYED"]:
        return "destroy_only"
    return "no_item_or_ward_event"


def summarize(errs):
    e = np.asarray(errs, float)
    a = np.abs(e)
    buckets = {"0": int((a == 0).sum()), "1": int((a == 1).sum()), "2-5": int(((a >= 2) & (a <= 5)).sum()),
               "6-50": int(((a > 5) & (a <= 50)).sum()), "51-500": int(((a > 50) & (a <= 500)).sum()),
               ">500": int((a > 500).sum())}
    return {"n": int(e.size), "share_within_tol": float((a <= TOL).mean()), "share_exact": float((a == 0).mean()),
            "pass_99pct_rule": bool((a <= TOL).mean() >= 0.99),
            "abs_error_quantiles": {q: float(np.percentile(a, q)) for q in (50, 90, 95, 99, 99.9, 100)},
            "mean_signed_error": float(e.mean()), "abs_error_buckets": buckets,
            "signed_error_top": [[int(k), int(v)] for k, v in Counter(e.astype(int).tolist()).most_common(15)]}


def run_gold(name, ids, table):
    t0 = time.time()
    VARS = ("raw", "ward_adjusted", "ward_bounded")
    errs = {v: [] for v in VARS}
    by_frame0 = {v: [] for v in VARS}
    onset_cls = {v: Counter() for v in VARS}
    miss_players = {v: set() for v in VARS}
    examples = defaultdict(list)
    clip_hits = 0
    by_champ_miss = Counter()
    sig_cls = {v: Counter() for v in VARS}
    other_list = {v: [] for v in VARS}
    for mi, m in enumerate(ids):
        meta, events, ts, node = load(m)
        sm = _static_meta(meta)
        champ = {int(k): int(v) for k, v in (sm.get("champion_by_pid") or {}).items()}
        x = ItemStateIndex.from_pack({"events": events, "meta": meta}, table)
        tg, cg = NODE_FEATURE_NAMES.index("totalGold_norm"), NODE_FEATURE_NAMES.index("curGold_norm")
        wk, wb = defaultdict(list), defaultdict(list)
        for e in events:
            if e.get("type") == "WARD_KILL":
                (wk if e.get("wardType") in WARD_TYPES_ADJUSTED else wb)[int(e.get("killerId", 0) or 0)].append(
                    int(e["timestamp"]))
        for pid in range(1, 11):
            state = {v: None for v in VARS}
            for j in range(len(ts)):
                T = int(ts[j])
                tn, cn = float(node[j, pid - 1, tg]), float(node[j, pid - 1, cg])
                clip_hits += int(tn >= CLIP or cn >= CLIP)
                obs = int(round(tn * DEN_TOT_G)) - int(round(cn * DEN_CUR_G))
                _, gd, _ = x.replay(pid, T, since_ms=-1)
                spent = -gd
                wadj = WARD_GOLD * sum(1 for t in wk[pid] if t <= T)
                nb = sum(1 for t in wb[pid] if t <= T)
                na = wadj // WARD_GOLD
                e_adj = int(round(spent - (obs - wadj)))
                e_bnd = min((e_adj + WARD_GOLD * k for k in range(nb + 1)), key=abs)
                for var, e_ in (("raw", int(round(spent - obs))), ("ward_adjusted", e_adj), ("ward_bounded", e_bnd)):
                    errs[var].append(e_)
                    if j == 0:
                        by_frame0[var].append(e_)
                    if abs(e_) > TOL:
                        miss_players[var].add((m, pid))
                        by_champ_miss[(var, champ.get(pid))] += 1
                        if state[var] is None:              # onset: first frame out of tolerance (per excursion)
                            lo = int(ts[j - 1]) if j > 0 else -1
                            ev_int = [e for e in events if lo < int(e.get("timestamp", 0)) <= T and (
                                e.get("participantId") == pid or e.get("killerId") == pid)]
                            gin = [g for g in x.grants[pid] if lo < g[0] <= T]
                            state[var] = classify(ev_int, pid, champ.get(pid), gin, table)
                            if len(examples[(var, state[var])]) < 5:
                                examples[(var, state[var])].append({"match": m, "pid": pid, "frame": j, "t_ms": T,
                                                                    "error": e_})
                        onset_cls[var][state[var]] += 1
                        if abs(e_) == 2:
                            sc = "abs_error_2"
                        elif e_ > 1 and any(abs(e_ - WARD_GOLD * k) <= TOL for k in range(1, na + 1)):
                            sc = "ward_kill_without_5"
                        else:
                            sc = "other"
                            other_list[var].append({"match": m, "pid": pid, "champion": champ.get(pid), "frame": j,
                                                    "t_ms": T, "error": e_, "onset_class": state[var]})
                        sig_cls[var][sc] += 1
                    else:
                        state[var] = None
        if (mi + 1) % 100 == 0:
            print(f"  [{name} {mi + 1}/{len(ids)}] {time.time() - t0:.0f}s", flush=True)
    out = {"n_matches": len(ids), "tolerance_gold": TOL, "clip_ceiling_hits": clip_hits}
    for var in errs:
        s = summarize(errs[var])
        s["frame0_only"] = summarize(by_frame0[var])
        s["players_with_any_miss"] = len(miss_players[var])
        s["players"] = len(ids) * 10
        s["miss_player_frames_by_onset_class"] = dict(onset_cls[var].most_common())
        s["miss_player_frames_by_error_value_class"] = dict(sig_cls[var].most_common())
        s["other_misses_first_frame_per_player"] = [o for k, o in enumerate(other_list[var])
                                                    if k == 0 or (o["match"], o["pid"]) != (
                                                        other_list[var][k - 1]["match"], other_list[var][k - 1]["pid"])][:200]
        s["n_other_miss_frames"] = len(other_list[var])
        s["onset_examples"] = {c: examples[(var, c)] for c in onset_cls[var]}
        s["miss_player_frames_top_champions"] = [[c, n] for (v, c), n in by_champ_miss.most_common() if v == var][:10]
        out[var] = s
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run_slots(name, ids, table):
    over, lens, trinkets = [], Counter(), Counter()
    item_counter = Counter()
    for m in ids:
        meta, events, ts, node = load(m)
        x = ItemStateIndex.from_pack({"events": events, "meta": meta}, table)
        t_end = max(int(e.get("timestamp", 0)) for e in events)
        for pid in range(1, 11):
            inv = x.inventory(pid, t_end)
            slots = []
            trinkets[min(2, sum(1 for i in inv if "Trinket" in (table.get(i, {}).get("tags") or [])))] += 1
            for i in inv:
                it = table.get(i, {})
                if "Trinket" in (it.get("tags") or []):
                    continue
                if is_consumable(it) and i in slots:          # a stack occupies one slot
                    continue
                slots.append(i)
            lens[len(slots)] += 1
            if len(slots) > 6:
                over.append({"match": m, "pid": pid, "champion": x.champion.get(pid), "slots": len(slots),
                             "inventory": sorted(inv)})
                item_counter.update(Counter(slots))
    n = len(ids) * 10
    return {"n_matches": len(ids), "players": n, "players_over_6_slots": len(over),
            "share_feasible": 1 - len(over) / n, "slot_count_distribution": {str(k): v for k, v in sorted(lens.items())},
            "trinket_count_distribution_(2=two_or_more)": {str(k): v for k, v in sorted(trinkets.items())},
            "items_in_over_capacity_inventories_top": [[i, table.get(i, {}).get("name"), c]
                                                      for i, c in item_counter.most_common(15)],
            "over_capacity_players": over[:50]}


def main(which):
    table = load_item_table_v2(PATCH)
    ss = samples()
    assert not set(ss["rule_check"]) & set(ss["held_out"])
    OUT.mkdir(parents=True, exist_ok=True)
    if which in ("gold", "all"):
        res = {"check": "15.14 gold consistency: replayed net item spending vs frame totalGold - currentGold",
               "doc": __doc__.split("gold   ")[1].split("slots  ")[0].strip()}
        for name, ids in ss.items():
            res[name] = run_gold(name, ids, table)
            for var in ("raw", "ward_adjusted", "ward_bounded"):
                r = res[name][var]
                print(f"{name:10s} {var:13s} n={r['n']} within+-1={r['share_within_tol']:.4f} "
                      f"exact={r['share_exact']:.4f} pass={r['pass_99pct_rule']}")
        (OUT / "gold_consistency_15.14.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    if which in ("slots", "all"):
        res = {"check": "end-of-game slot feasibility (<= 6 non-trinket slots, stackables one slot per id)"}
        idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
        rev = sorted(k for k, v in idx.items() if v == PATCH)
        random.Random(424242).shuffle(rev)                   # the completion reviewer's sample (46/2000 before)
        for name, ids in list(ss.items()) + [("reviewer_sample_424242", rev[:200])]:
            res[name] = run_slots(name, ids, table)
            print(f"{name:10s} slots>6: {res[name]['players_over_6_slots']}/{res[name]['players']}")
        (OUT / "slot_feasibility_15.14.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
