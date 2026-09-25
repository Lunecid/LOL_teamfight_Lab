"""15.x item rules (v4-exact, task RULES15): evidence, 15.14 identity hash, 15.14 gold / slot re-check, first-KR-15.18
in-sample inventory diagnostic, role-inference start gold.  Single process.

The rules live in config/game_rules/item_rules_15x.json and gameplay/item_state.py (_R15Context, ItemStateIndex);
they apply only when the patch major is 15.  Author decision records/items_strict_rule_20260925T125703Z.json: strict
item-ID set equality for item0-5 AND item6 with one exception class {3867, 3869, 3870, 3871, 3876, 3877}; 15.x rules
for 3010 -> 3013, Biscuit Delivery and the trinket slot.  Every rule is derived from events of the 15.14 rule-check
sample (the 300 matches of ev4_s1_item_gold.samples()['rule_check'], seed 20260925) or the FIRST KR 15.18 sample
(records/replication_exclusions_inventory_1518.txt, 500 ids).  The fresh KR 15.18 validation sample is never read.

Data handling.  15.14 cache only (patch index rows == '15.14'; no other patch is loaded, counted or printed).  KR raw
only for the first 15.18 sample, through ev4_s1_kr_inventory.load_match (outcome keys dropped at parse time by its
object_pairs_hook; the kept fields: gameVersion / queueId / mapId, participantId, championId / Name, summoner
spells, perks, item0-6, roleBoundItem; timeline events, GAME_END reduced to type + timestamp).  No outcome field is
read or printed.

  evidence  event counts behind R15-1 .. R15-7 on the 15.14 rule-check sample and the first KR 15.18 sample
            -> <OUT>/rules_evidence_15x.json
  kr_first  the first KR 15.18 sample replayed before (rules_15x=False) and after (15.x rules) at GAME_END:
            strict set equality item0-5, item6, both, with the support-upgrade class mapped to one token (IN-SAMPLE:
            the rules were derived on it; the out-of-sample test is the fresh sample, not run here)
            -> <OUT>/inventory_first_kr_15.18_insample.json
  identity  the 50-match 15.14 identity digest (ev4_s1x_items_16x.identity_digest) with the 15.x rules off (must
            equal the old REF_HASH_1514 8ff6d349...) and on (the new reference) -> <OUT>/identity_15.14.json.
            Reference with the 15.x rules on: 0ca245e108e9b648... (RULES15, R15-1 .. R15-7) -> e7cb8e51d5874e73...
            (2026-09-25, R15-5 Triple Tonic removed by author decision records/items_confirmations_20260925T134510Z.json;
            run with OUT = stage1/validation/strict_noR155).  The 'evidence' / 'kr_first' passes still count the
            Triple Tonic events under the label R15-5 (the evidence of the removed rule, kept in 'removed_rules').
  gold      ev4_s1_item_gold.run_gold (raw / ward_adjusted / ward_bounded) plus the reviewer's path-consistent
            variant (review/review_items_gold_path.py best_path, copied) on rule_check (300), held_out (300) and a
            fresh 200 (seed 20260928, disjoint from rule_check, held_out, reviewer-424242 200, fresh-20260927 200 and
            the identity 50), before (rules_15x=False) and after -> <OUT>/gold_consistency_15.14_15x.json
  slots     ev4_s1_item_gold.run_slots (after) on the same three samples + reviewer-424242 200
            -> <OUT>/slot_feasibility_15.14_15x.json
  roles     role_inference start gold / 'low_start' and team ambiguity on the 500 StateV3-validation matches
            (ev4_s1x_state_v3_checks.sample_ids) at the full horizon: participant-0 World Atlas kept (given to
            participants 5 / 10, role_inference before this task) vs dropped (current role_inference._inventory_gold)
            -> <OUT>/role_start_gold_15.14.json

Usage: python scripts/exact_v4/ev4_s1y_items_15x.py [evidence|kr_first|identity|gold|slots|roles|all]
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/exact_v4"))

import ev4_s1_item_gold as G  # noqa: E402
from gameplay import item_state as IS  # noqa: E402
from gameplay.item_state import (  # noqa: E402
    ItemStateIndex, _static_meta, is_trinket, load_item_table_v2, load_rules_15x,
)

BASE = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925")
OUT = BASE / "stage1/item_state_15x"
FIRST_KR = BASE / "records/replication_exclusions_inventory_1518.txt"
SUPPORT_CLASS = frozenset({3867, 3869, 3870, 3871, 3876, 3877})
SUPPORT_TOKEN = -3867
FRESH_SEED, FRESH_N = 20260928, 200
HERALD_WINDOW_MS = 120_000
VIEGO, FIDDLESTICKS = 234, 9
BISCUIT_RUNE, BISCUIT, TONIC_RUNE = 8345, 2010, 8313


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _q(xs, qs=(0, 1, 50, 99, 100)):
    return {str(q): round(float(np.percentile(xs, q)), 3) for q in qs} if len(xs) else {}


def _iid(e, f="itemId"):
    return int(e.get(f, 0) or 0)


def _runes(runes_map):
    return {int(v) for k, v in (runes_map or {}).items() if "rune" in str(k) and v is not None}


def first_kr_ids():
    ids = [ln.strip() for ln in FIRST_KR.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(ids) == len(set(ids)) == 500
    return ids


def load_kr(mid):
    import ev4_s1_kr_inventory as K                     # outcome keys dropped by its object_pairs_hook
    from data.cache_io import _extract_static_meta_from_detail
    detail, events = K.load_match(mid)
    return detail, events, _extract_static_meta_from_detail(detail)


def load_1514(m):
    meta = json.loads((G.CACHE / f"{m}.meta.json").read_text(encoding="utf-8"))
    if str(meta["patch"]) != G.PATCH:
        raise RuntimeError(f"{m} is not {G.PATCH}")
    events = json.loads((G.CACHE / f"{m}.events.json").read_text(encoding="utf-8"))
    return meta, events


# ============================================================================================ evidence
class Ev:
    """Per-player event-pattern counter shared by the 15.14 and the KR pass (events only)."""

    def __init__(self, table):
        self.t = table
        self.tr = {i for i, v in table.items() if is_trinket(v)}
        self.c = Counter()
        self.lists = defaultdict(list)

    def player(self, pe, ev_all, pid, champ, runes, team_of, herald_kills):
        c, TR = self.c, self.tr
        tre = [e for e in pe if _iid(e) in TR or _iid(e, "afterId") in TR or _iid(e, "beforeId") in TR]
        by_ms = defaultdict(list)
        for e in pe:
            by_ms[int(e["timestamp"])].append(e)
        c["players"] += 1
        # R15-2 start trinket
        if not tre:
            c["R15-2 players_without_any_trinket_event"] += 1
        else:
            f = tre[0]
            c[f"R15-2 first_trinket_event {f['type']} {_iid(f) or _iid(f, 'afterId')}"] += 1
            if f["type"] == "ITEM_DESTROYED":
                self.lists["R15-2 first_event_is_destroy_ms"].append(int(f["timestamp"]))
        if champ == FIDDLESTICKS:
            c["R15-2 fiddlesticks_players"] += 1
            c[f"R15-2 fiddlesticks_first_trinket_event {tre[0]['type'] + ' ' + str(_iid(tre[0])) if tre else 'none'}"] += 1
        # R15-3 trinket purchases without a same-ms trinket destroy, R15-6 loose trinket destroys
        herald_held, stored = False, None
        for e in tre:
            ts, iid = int(e["timestamp"]), _iid(e)
            same = by_ms[ts]
            tr_des = any(x["type"] == "ITEM_DESTROYED" and _iid(x) in TR for x in same)
            tr_buy = any(x["type"] == "ITEM_PURCHASED" and _iid(x) in TR for x in same)
            if e["type"] == "ITEM_PURCHASED":
                if tr_des:
                    c["R15-3 trinket_purchase_with_same_ms_trinket_destroy (swap)"] += 1
                else:
                    c[f"R15-3 trinket_purchase_without_trinket_destroy {iid}"] += 1
                    self.lists[f"R15-3 purchase_without_destroy_ms {iid}"].append(ts)
                    if herald_held:
                        c["R15-3 trinket_purchase_while_3513_held"] += 1
            elif e["type"] == "ITEM_UNDO" and not _iid(e, "beforeId") and _iid(e, "afterId") in TR:
                c["R15-3 undo_beforeId0_afterId_trinket"] += 1
            elif e["type"] == "ITEM_DESTROYED" and iid != 3513 and not tr_buy:
                n_other = sum(1 for x in same if x["type"] == "ITEM_DESTROYED") - 1
                own = [h for h in herald_kills if h[1] == team_of[pid] and 0 <= ts - h[0] <= HERALD_WINDOW_MS]
                c["R15-6 loose_trinket_destroys"] += 1
                c[f"R15-6 loose_trinket_destroy_id {iid}"] += 1
                c["R15-6 loose_trinket_destroy_other_destroys_same_ms"] += int(n_other > 0)
                if own:
                    c["R15-6 own_team_RIFTHERALD_kill_within_2min_before"] += 1
                    self.lists["R15-6 delay_after_own_herald_kill_s"].append((ts - own[-1][0]) / 1000)
                nxt = [x for x in tre if int(x["timestamp"]) > ts]
                c["R15-6 next_trinket_event_is_3513_destroy" if nxt and nxt[0]["type"] == "ITEM_DESTROYED"
                  and _iid(nxt[0]) == 3513 else ("R15-6 no_later_trinket_event" if not nxt
                                                 else "R15-6 next_trinket_event_other")] += 1
                herald_held, stored = True, iid
                if champ == VIEGO:
                    c["R15-7 viego_loose_trinket_destroys"] += 1
                    c["R15-7 viego_loose_trinket_destroy_alone_at_ms"] += int(n_other == 0)
            elif e["type"] == "ITEM_DESTROYED" and iid == 3513:
                c["R15-6 3513_destroys"] += 1
                c["R15-6 3513_destroy_after_a_loose_trinket_destroy" if herald_held else "R15-6 3513_destroy_without_prior"] += 1
                nxt = [x for x in tre if int(x["timestamp"]) > ts]
                if nxt:
                    c["R15-6 after_3513_next_trinket_event_" + ("destroy_of_stored" if nxt[0]["type"] == "ITEM_DESTROYED"
                                                                and _iid(nxt[0]) == stored else "other")] += 1
                herald_held = False
        if champ == VIEGO:
            for ts, grp in by_ms.items():
                ds = [x for x in grp if x["type"] == "ITEM_DESTROYED"]
                if len(ds) >= 2 and any(_iid(x) in TR for x in ds):
                    c["R15-7 viego_multi_destroy_groups_with_a_trinket"] += 1
                elif len(ds) >= 2:
                    c["R15-7 viego_multi_destroy_groups_without_trinket"] += 1
        # R15-1 Symbiotic Soles
        for e in pe:
            if e["type"] == "ITEM_DESTROYED" and _iid(e) == 3010 and champ != VIEGO:
                ts = int(e["timestamp"])
                bought = [_iid(x) for x in by_ms[ts] if x["type"] == "ITEM_PURCHASED"]
                later = [x for x in pe if int(x["timestamp"]) > ts and 3013 in (_iid(x), _iid(x, "beforeId"), _iid(x, "afterId"))]
                c["R15-1 3010_destroys_non_viego"] += 1
                c["R15-1 3010_destroy_with_same_ms_purchase" if bought else "R15-1 3010_destroy_without_purchase"] += 1
                c["R15-1 later_3013_event" if later else "R15-1 no_later_3013_event"] += 1
                if later:
                    c[f"R15-1 first_later_3013_event {later[0]['type']}"] += 1
            if _iid(e) == 3013 and champ != VIEGO:
                prior = any(x["type"] == "ITEM_DESTROYED" and _iid(x) == 3010 and int(x["timestamp"]) < int(e["timestamp"])
                            for x in pe)
                c["R15-1 3013_events_non_viego " + ("after_a_3010_destroy" if prior else "WITHOUT_prior_3010_destroy")] += 1
            if e["type"] == "ITEM_PURCHASED" and _iid(e) in (3010, 3013):
                c[f"R15-1 purchases_of {_iid(e)}"] += 1
        # R15-4 biscuits
        b = [e for e in pe if BISCUIT in (_iid(e), _iid(e, "afterId"))]
        has = BISCUIT_RUNE in runes
        c[f"R15-4 players_{'with' if has else 'without'}_rune n_2010_events={min(len(b), 4)}"] += 1
        if has:
            for k, e in enumerate(b[:3], 1):
                self.lists[f"R15-4 kth_2010_event_minus_k*2min_s k={k}"].append((int(e["timestamp"]) - 120_000 * k) / 1000)
        elif b:
            c[f"R15-4 2010_events_without_rune champion={'Viego' if champ == VIEGO else champ}"] += len(b)
        c["R15-4 2010_purchases"] += sum(1 for e in b if e["type"] == "ITEM_PURCHASED")
        # R15-5 Triple Tonic
        tonic = [e for e in pe if _iid(e) in (2150, 2151, 2152)]
        if TONIC_RUNE in runes:
            lv = {}
            for e in ev_all:
                if e["type"] == "LEVEL_UP" and int(e.get("participantId", 0) or 0) == pid:
                    lv.setdefault(int(e["level"]), int(e["timestamp"]))
            c["R15-5 players_with_rune"] += 1
            for item, L in ((2151, 3), (2152, 6), (2150, 9)):
                es = sorted(int(e["timestamp"]) for e in tonic if _iid(e) == item)
                if L in lv:
                    c[f"R15-5 level{L}_reached {item}_event={'yes' if es else 'no'}"] += 1
                    if es:
                        c[f"R15-5 level{L} first_{item}_event_at_or_after_levelup={'yes' if es[0] >= lv[L] else 'NO'}"] += 1
                        self.lists[f"R15-5 {item}_delay_after_levelup_s"].append((es[0] - lv[L]) / 1000)
                else:
                    c[f"R15-5 level{L}_not_reached {item}_events={len(es)}"] += 1
            c["R15-5 tonic_purchases"] += sum(1 for e in tonic if e["type"] == "ITEM_PURCHASED")
        elif tonic:
            c["R15-5 tonic_events_without_rune"] += len(tonic)

    def summary(self):
        out = {k: v for k, v in sorted(self.c.items())}
        out |= {k + " (quantiles)": _q(v) for k, v in sorted(self.lists.items())}
        out |= {k + " (n)": len(v) for k, v in sorted(self.lists.items())}
        return out


def _herald_kills(events):
    return [(int(e["timestamp"]), int(e.get("killerTeamId", 0) or 0)) for e in events
            if e.get("type") == "ELITE_MONSTER_KILL" and e.get("monsterType") == "RIFTHERALD"]


def run_evidence():
    t0 = time.time()
    ss = G.samples()
    t14 = load_item_table_v2("15.14")
    e14 = Ev(t14)
    for m in ss["rule_check"]:
        meta, events = load_1514(m)
        events = sorted(events, key=lambda e: int(e.get("timestamp", 0)))
        sm = _static_meta(meta)
        tm = meta["team_map"]
        tm = {int(k): int(v) for k, v in (tm if isinstance(tm, dict) else json.loads(tm.replace("'", '"'))).items()}
        hk = _herald_kills(events)
        for pid in range(1, 11):
            pe = [e for e in events if str(e["type"]).startswith("ITEM_") and int(e.get("participantId", 0) or 0) == pid]
            e14.player(pe, events, pid, int(sm["champion_by_pid"][str(pid)]),
                       _runes(sm["runes_by_pid"].get(str(pid))), tm, hk)
    t18 = load_item_table_v2("15.18")
    e18 = Ev(t18)
    kr = Counter()
    for mid in first_kr_ids():
        detail, events, sm = load_kr(mid)
        events = sorted(events, key=lambda e: int(e["timestamp"]))
        t_end = max(int(e["timestamp"]) for e in events)
        parts = detail["info"]["participants"]
        tm = {int(p["participantId"]): (100 if int(p["participantId"]) <= 5 else 200) for p in parts}
        hk = _herald_kills(events)
        for p in parts:
            pid = int(p["participantId"])
            champ = int(p.get("championId", 0) or 0)
            runes = _runes(sm["runes_by_pid"].get(str(pid)))
            pe = [e for e in events if str(e["type"]).startswith("ITEM_") and int(e.get("participantId", 0) or 0) == pid]
            e18.player(pe, events, pid, champ, runes, tm, hk)
            # end-of-game item slots (detail), for the rules' end states
            it6 = int(p.get("item6", 0) or 0)
            main = [int(p.get(f"item{k}", 0) or 0) for k in range(6)]
            tre = [e for e in pe if _iid(e) in e18.tr or _iid(e, "afterId") in e18.tr]
            if champ == FIDDLESTICKS:
                kr[f"R15-2 fiddlesticks item6={it6} trinket_events={'yes' if tre else 'no'}"] += 1
            elif not tre:
                kr[f"R15-2 no_trinket_event item6={it6}"] += 1
                if it6 == 0:
                    kr["R15-2 no_trinket_event item6=0: players with 0 ITEM_* events"] += int(not pe)
                    kr["R15-2 no_trinket_event item6=0: game_end_s_max"] = max(kr["R15-2 no_trinket_event item6=0: game_end_s_max"],
                                                                              t_end // 1000)
            holding, stored = False, None
            for e in tre:
                ts = int(e["timestamp"])
                tr_buy = any(x["type"] == "ITEM_PURCHASED" and _iid(x) in e18.tr for x in pe if int(x["timestamp"]) == ts)
                if e["type"] == "ITEM_DESTROYED" and _iid(e) != 3513 and not tr_buy:
                    holding, stored = True, _iid(e)
                elif e["type"] == "ITEM_DESTROYED" and _iid(e) == 3513:
                    holding = False
            if stored is not None:
                if holding:
                    kr[f"R15-6 herald player, no 3513 destroy after the last loose trinket destroy: item6={it6}"] += 1
                else:
                    later_swap = any(e["type"] == "ITEM_PURCHASED" and _iid(e) in e18.tr and int(e["timestamp"]) > max(
                        int(x["timestamp"]) for x in tre if _iid(x) == 3513) for e in tre) if any(_iid(x) == 3513 for x in tre) else False
                    kr["R15-6 herald player, 3513 used: item6 == stored trinket" if it6 == stored else (
                        "R15-6 herald player, 3513 used: item6 != stored, trinket bought after the 3513 destroy"
                        if later_swap else f"R15-6 herald player, 3513 used: item6 != stored, NO later purchase item6={it6}")] += 1
            loose10 = [e for e in pe if e["type"] == "ITEM_DESTROYED" and _iid(e) == 3010 and not any(
                x["type"] == "ITEM_PURCHASED" and int(x["timestamp"]) == int(e["timestamp"]) for x in pe)]
            left13 = any(e["type"] in ("ITEM_DESTROYED", "ITEM_SOLD") and _iid(e) == 3013 for e in pe)
            if champ != VIEGO:
                if loose10:
                    kr["R15-1 loose 3010 destroy: " + ("3013 in item0-5" if 3013 in main else (
                        "3013 left by a later destroy / sale" if left13 else "NO 3013"))] += 1
                if 3013 in main:
                    kr["R15-1 3013 in item0-5: " + ("after a loose 3010 destroy" if loose10 else "WITHOUT a loose 3010 destroy")] += 1
                if 3010 in main:
                    kr["R15-1 3010 in item0-5: " + ("after a loose 3010 destroy" if loose10 else "no 3010 destroy")] += 1
            b = [e for e in pe if _iid(e) == BISCUIT]
            if BISCUIT_RUNE in runes:
                n_due = sum(1 for t in (120_000, 240_000, 360_000) if t <= t_end)
                held = main.count(BISCUIT) and 1
                kr[f"R15-4 rune: due_by_game_end={n_due} 2010_events={min(len(b), 4)} 2010_in_item0_5={'yes' if held else 'no'}"] += 1
            elif BISCUIT in main:
                kr["R15-4 2010 in item0-5 without the rune"] += 1
            for tid in (2150, 2151, 2152):
                if tid in main:
                    kr[f"R15-5 {tid} in item0-5 rune={'yes' if TONIC_RUNE in runes else 'no'}"] += 1
    out = {
        "note": "event counts behind item_rules_15x.json R15-1 .. R15-7.  15.14: rule-check sample (300 matches, "
                "ev4_s1_item_gold.samples, seed 20260925, 3000 players), cache events.  KR 15.18: the FIRST sample "
                "(records/replication_exclusions_inventory_1518.txt, 500 matches, 5000 players), timeline events and "
                "detail item0-6 at game end; outcome keys dropped at parse.  The fresh KR 15.18 sample is not read.",
        "15.14_rule_check": e14.summary(),
        "kr_15.18_first_sample_events": e18.summary(),
        "kr_15.18_first_sample_end_of_game": dict(sorted(kr.items())),
        "script": "scripts/exact_v4/ev4_s1y_items_15x.py evidence", "elapsed_s": round(time.time() - t0, 1),
    }
    _dump("rules_evidence_15x.json", out)
    return out


# ============================================================================================ first KR (in-sample)
def _norm(ids):
    return {SUPPORT_TOKEN if i in SUPPORT_CLASS else i for i in ids}


def run_kr_first():
    t0 = time.time()
    table = load_item_table_v2("15.18")
    assert IS.uses_rules_15x("15.18")
    res = {}
    resid = {"before": Counter(), "after": Counter()}
    resid_ex = []
    for mid in first_kr_ids():
        detail, events, sm = load_kr(mid)
        ends = [int(e["timestamp"]) for e in events if e["type"] == "GAME_END"]
        t_end = max(ends) if ends else max(int(e["timestamp"]) for e in events)
        kw = dict(exact=True, patch="15.18", champion_by_pid=sm["champion_by_pid"], runes_by_pid=sm["runes_by_pid"],
                  summoner_spells_by_pid=sm["summoner_spells_by_pid"])
        idx = {"before": ItemStateIndex(events, table, rules_15x=False, **kw), "after": ItemStateIndex(events, table, **kw)}
        assert idx["after"].rules15 is not None and idx["before"].rules15 is None
        for p in detail["info"]["participants"]:
            pid = int(p["participantId"])
            det_main = [int(p.get(f"item{k}", 0) or 0) for k in range(6) if int(p.get(f"item{k}", 0) or 0)]
            it6 = int(p.get("item6", 0) or 0)
            for name, ix in idx.items():
                c = res.setdefault(name, Counter())
                inv = ix.inventory(pid, t_end)
                main = [i for i in inv if not is_trinket(table.get(i, {}))]
                tr = [i for i in inv if is_trinket(table.get(i, {}))]
                ok_main = _norm(main) == _norm(det_main)
                ok_raw = set(main) == set(det_main)
                ok6 = set(tr) == ({it6} if it6 else set())
                c["n"] += 1
                c["item0_5_strict_set"] += ok_raw
                c["item0_5_strict_set_support_class"] += ok_main
                c["item6"] += ok6
                c["item0_6_support_class"] += ok_main and ok6
                if not (ok_main and ok6):
                    miss = sorted(_norm(det_main) - _norm(main)) + ([f"item6:{it6}"] if not ok6 else [])
                    extra = sorted(_norm(main) - _norm(det_main)) + ([f"trinkets:{sorted(tr)}"] if not ok6 else [])
                    resid[name][f"missing {miss} extra {extra}"] += 1
                    if name == "after" and len(resid_ex) < 40:
                        resid_ex.append({"match": mid, "pid": pid, "champion": p.get("championName"),
                                         "detail_item0_5": sorted(det_main), "item6": it6, "replay": sorted(inv)})
    out = {"note": "IN-SAMPLE diagnostic: the first KR 15.18 sample (500 matches) is the sample R15-1 .. R15-7 were "
                   "derived on (with 15.14 events); the out-of-sample check is the fresh KR 15.18 sample (other task). "
                   "Strict set equality per player at GAME_END; 'support_class' maps {3867, 3869, 3870, 3871, 3876, "
                   "3877} to one token (author decision items_strict_rule_20260925T125703Z); item6 = replay trinket set "
                   "== {item6} (both empty when item6 == 0).  'before' = rules_15x=False (the replay of the first KR "
                   "run), 'after' = the 15.x rules."}
    for name, c in res.items():
        n = c["n"]
        out[name] = {"n_players": n, **{k: c[k] / n for k in ("item0_5_strict_set", "item0_5_strict_set_support_class",
                                                               "item6", "item0_6_support_class")},
                     "n_mismatch_item0_6_support_class": n - c["item0_6_support_class"],
                     "mismatch_patterns_top": resid[name].most_common(40)}
    out["after_mismatch_examples"] = resid_ex
    out["elapsed_s"] = round(time.time() - t0, 1)
    _dump("inventory_first_kr_15.18_insample.json", out)
    return {k: {kk: vv for kk, vv in v.items() if kk != "mismatch_patterns_top"} if isinstance(v, dict) else v
            for k, v in out.items() if k != "after_mismatch_examples"}


# ============================================================================================ identity
class _NoRules15Index(ItemStateIndex):
    @classmethod
    def from_pack(cls, pack, items_table=None, **kw):
        kw.setdefault("rules_15x", False)
        return super().from_pack(pack, items_table, **kw)


def identity_digests():
    import ev4_s1x_items_16x as S
    off = types.SimpleNamespace(ItemStateIndex=_NoRules15Index, player_item_vector=IS.player_item_vector)
    h_off, n_off = S.identity_digest(off)
    h_on, n_on = S.identity_digest()
    return {"old_ref_REF_HASH_1514_before_15x": "8ff6d349744d310a8f87a6b087d1490466d2d23ab16da44851b1cd21ca824d9f",
            "digest_rules_15x_off": h_off, "digest_rules_15x_on": h_on, "n": [n_off, n_on],
            "off_equals_old_ref": h_off == "8ff6d349744d310a8f87a6b087d1490466d2d23ab16da44851b1cd21ca824d9f",
            "ref_in_ev4_s1x_items_16x": S.REF_HASH_1514, "ref_equals_on": S.REF_HASH_1514 == h_on}


def run_identity():
    t0 = time.time()
    d = identity_digests()
    # what changed: per (match, t, pid) diff counts between off and on over the same 50 matches
    import ev4_s1x_items_16x as S
    changed = Counter()
    for mid in S.identity_ids():
        meta, ev = load_1514(mid)
        on = ItemStateIndex.from_pack({"events": ev, "meta": meta})
        off = ItemStateIndex.from_pack({"events": ev, "meta": meta}, rules_15x=False)
        tl = max(int(e.get("timestamp", 0)) for e in ev)
        for t in list(range(0, tl + 60000, 60000)) + [tl]:
            for pid in range(1, 11):
                a, b = off.replay(pid, t, since_ms=max(0, t - 60000)), on.replay(pid, t, since_ms=max(0, t - 60000))
                changed["n"] += 1
                changed["inventory_differs"] += sorted(a[0]) != sorted(b[0])
                changed["gold_delta_differs"] += a[1] != b[1]
                va, vb = IS.player_item_vector(a[0], off.items, off.flags), IS.player_item_vector(b[0], on.items, on.flags)
                changed["item_vector_differs"] += va != vb
                for k in va:
                    if va[k] != vb[k]:
                        changed[f"vector_field_differs {k}"] += 1
    d |= {"changes_off_to_on_over_the_same_match_time_players": dict(changed),
          "why": "the 15.x rules (item_rules_15x.json 'rules'; R15-5 Triple Tonic removed 2026-09-25) change 15.14 "
                 "replays: the 0-ms trinket grant, the Biscuit Delivery grant, 3010 -> 3013, the Eye of the Herald "
                 "exchange.  With "
                 "rules_15x=False the module reproduces the old reference exactly, so nothing else changed.",
          "script": "scripts/exact_v4/ev4_s1y_items_15x.py identity", "elapsed_s": round(time.time() - t0, 1)}
    _dump("identity_15.14.json", d)
    return d


# ============================================================================================ gold / slots
def best_path(e_adj, b_int, tol=1, w=5):
    """Copied from review/review_items_gold_path.py (reviewer's path-consistent ward_bounded variant): max #frames
    within tol over a non-decreasing k with increments <= b_int[j]; returns (n_ok, per-frame ok flags)."""
    K = int(sum(b_int))
    NEG = -10 ** 9
    score = np.full(K + 1, NEG, dtype=np.int64)
    back = []
    for j, (e, b) in enumerate(zip(e_adj, b_int)):
        new = np.full(K + 1, NEG, dtype=np.int64)
        arg = np.zeros(K + 1, dtype=np.int64)
        ok = np.array([1 if abs(e + w * k) <= tol else 0 for k in range(K + 1)], dtype=np.int64)
        for k in range(K + 1):
            if j == 0:
                if k <= b:
                    new[k], arg[k] = ok[k], k
                continue
            lo = max(0, k - b)
            seg = score[lo:k + 1]
            mm = int(np.argmax(seg))
            if seg[mm] > NEG:
                new[k], arg[k] = seg[mm] + ok[k], lo + mm
        back.append((arg, ok))
        score = new
    k = int(np.argmax(score))
    n_ok = int(score[k])
    flags = []
    for j in range(len(e_adj) - 1, -1, -1):
        arg, ok = back[j]
        flags.append(int(ok[k]))
        k = int(arg[k])
    return n_ok, flags[::-1]


def gold_path(ids, table, rules_15x=None):
    from core.config import NODE_FEATURE_NAMES
    tg, cg = NODE_FEATURE_NAMES.index("totalGold_norm"), NODE_FEATURE_NAMES.index("curGold_norm")
    n = n_path = 0
    spend_digest = hashlib.sha256()
    for m in ids:
        meta, events, ts, node = G.load(m)
        kw = {} if rules_15x is None else {"rules_15x": rules_15x}
        x = ItemStateIndex.from_pack({"events": events, "meta": meta}, table, **kw)
        wk, wb = defaultdict(list), defaultdict(list)
        for e in events:
            if e.get("type") == "WARD_KILL":
                (wk if e.get("wardType") in G.WARD_TYPES_ADJUSTED else wb)[int(e.get("killerId", 0) or 0)].append(
                    int(e["timestamp"]))
        for pid in range(1, 11):
            e_adj, b_int, prev = [], [], -1
            for j in range(len(ts)):
                T = int(ts[j])
                obs = int(round(float(node[j, pid - 1, tg]) * G.DEN_TOT_G)) - int(round(float(node[j, pid - 1, cg]) * G.DEN_CUR_G))
                spent = -x.replay(pid, T, since_ms=-1)[1]
                spend_digest.update(f"{m},{pid},{T},{spent!r};".encode())
                e_adj.append(int(round(spent - (obs - G.WARD_GOLD * sum(1 for t in wk[pid] if t <= T)))))
                b_int.append(sum(1 for t in wb[pid] if prev < t <= T))
                prev = T
            n_ok, _ = best_path(e_adj, b_int)
            n += len(ts)
            n_path += n_ok
    return {"n": n, "ward_path_consistent": n_path / n, "pass_99": n_path / n >= 0.99,
            "replayed_spending_sha256": spend_digest.hexdigest()}


def gold_samples():
    ss = G.samples()
    idx = json.loads(G.PATCH_INDEX.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in idx.items() if v == G.PATCH)
    rev = list(ids)
    random.Random(424242).shuffle(rev)
    used = set(ss["rule_check"]) | set(ss["held_out"]) | set(rev[:200])
    f27 = [i for i in ids if i not in used]
    random.Random(20260927).shuffle(f27)
    import ev4_s1x_items_16x as S
    used |= set(f27[:200]) | set(S.identity_ids())
    fresh = [i for i in ids if i not in used]
    random.Random(FRESH_SEED).shuffle(fresh)
    fresh = fresh[:FRESH_N]
    return {"rule_check": ss["rule_check"], "held_out": ss["held_out"], f"fresh_{FRESH_SEED}": fresh}, rev[:200]


def run_gold():
    t0 = time.time()
    table = load_item_table_v2(G.PATCH)
    samples, _ = gold_samples()
    fresh = samples[f"fresh_{FRESH_SEED}"]
    out = {"check": "15.14 gold consistency with the 15.x item rules (ev4_s1_item_gold.run_gold + path-consistent "
                    "variant); before = rules_15x=False, after = default (15.x rules on)",
           "fresh_sample": {"seed": FRESH_SEED, "n": len(fresh), "disjoint_from": "rule_check 300, held_out 300, "
                            "reviewer-424242 200, fresh-20260927 200, identity 50",
                            "ids_sha256": hashlib.sha256("\n".join(fresh).encode()).hexdigest(), "ids": fresh}}
    for name, ids in samples.items():
        r = G.run_gold(name, ids, table)
        row = {}
        for var in ("raw", "ward_adjusted", "ward_bounded"):
            x = r[var]
            row[var] = {"n": x["n"], "share_within_tol": x["share_within_tol"], "share_exact": x["share_exact"],
                        "pass_99": x["pass_99pct_rule"], "abs_error_quantiles": x["abs_error_quantiles"],
                        "miss_by_error_value_class": x["miss_player_frames_by_error_value_class"],
                        "n_other_miss_frames": x["n_other_miss_frames"],
                        "other_misses_first_frame_per_player": x["other_misses_first_frame_per_player"][:30]}
        after = gold_path(ids, table)
        before = gold_path(ids, table, rules_15x=False)
        row["ward_path_consistent"] = after
        row["replayed_spending_identical_before_after"] = (before["replayed_spending_sha256"]
                                                          == after["replayed_spending_sha256"])
        row["ward_path_consistent_before"] = before["ward_path_consistent"]
        out[name] = row
        print(name, {v: round(row[v]["share_within_tol"], 4) for v in ("raw", "ward_adjusted", "ward_bounded")},
              "path", round(after["ward_path_consistent"], 4), "spend identical",
              row["replayed_spending_identical_before_after"], flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    _dump("gold_consistency_15.14_15x.json", out)
    return {k: v for k, v in out.items() if k != "fresh_sample"}


def run_slots():
    table = load_item_table_v2(G.PATCH)
    samples, rev200 = gold_samples()
    out = {"check": "end-of-game slot feasibility with the 15.x rules (<= 6 non-trinket slots, stackables one slot)"}
    for name, ids in list(samples.items()) + [("reviewer_sample_424242", rev200)]:
        out[name] = G.run_slots(name, ids, table)
        print(name, out[name]["players_over_6_slots"], "/", out[name]["players"], flush=True)
    _dump("slot_feasibility_15.14_15x.json", out)
    return {k: (v["players_over_6_slots"], v["players"], v["trinket_count_distribution_(2=two_or_more)"])
            for k, v in out.items() if isinstance(v, dict)}


# ============================================================================================ roles
def _inventory_gold_with_atlas(pack, horizon):
    """role_inference._inventory_gold as it was before this task (World Atlas of participant 0 kept, so item_state
    gives it to participants 5 / 10)."""
    idx = ItemStateIndex.from_pack(pack)
    return {p: float(idx.vector(p, int(horizon))["item_gold_owned"]) for p in range(1, 11)}, None


def run_roles():
    import ev4_s1x_state_v3_checks as V
    from gameplay import role_inference as cur
    t0 = time.time()
    keys = [(q, name) for q in ("full", "early") for name in ("before", "after")]
    c = {k: Counter() for k in keys}
    gold_s = {k: defaultdict(list) for k in keys}
    current = cur._inventory_gold
    rng = random.Random(V.T_SEED)                   # the query times of ev4_s1x_state_v3_checks.main
    slots = {p: p - 1 for p in range(1, 11)}        # participant order (the StateV3 slots)
    try:
        for m in V.sample_ids():
            pack = V.load(m)
            ts = pack["minute_ts"]
            times = [rng.randint(int(ts[0]), int(ts[-1])) for _ in range(8)] + [rng.randint(60_000, 539_999)
                                                                                for _ in range(2)]
            for q, qt in [("full", None)] + [("early", t) for t in times[8:]]:
                for name, fn in (("before", _inventory_gold_with_atlas), ("after", current)):
                    cur._inventory_gold = fn
                    ra = cur.infer_roles(pack, qt)
                    cc = c[(q, name)]
                    for team, tr in ra.teams.items():
                        cc["teams"] += 1
                        cc["teams_ambiguous"] += tr.ambiguous
                    for p, s in slots.items():
                        cc["players"] += 1
                        cc["players_agree_participant_order"] += ra.slot_by_pid[p] == s
                        ev = ra.evidence[p]
                        role = V.ROLES[s % 5]
                        gold_s[(q, name)][role].append(ev["gold"])
                        cc[f"low_start {role}"] += ev["low_start"]
    finally:
        cur._inventory_gold = current
    out = {"note": "500 StateV3-validation 15.14 matches (ev4_s1x_state_v3_checks.sample_ids); infer_roles at the full "
                   "horizon 8:59.999 and at the 2 early query times per match of ev4_s1x_state_v3_checks.main "
                   "(random.Random(7), [60 s, 540 s)); start gold = inventory at min(H, 90 s).  before = the World "
                   "Atlas purchases of participant 0 kept (item_state gives them to participants 5 / 10; "
                   "role_inference before this task); after = dropped before the replay (current "
                   "role_inference._inventory_gold).  Both with the 15.x item rules.  Participant order is only the "
                   "comparison target (validation metadata)."}
    for q, name in keys:
        cc = c[(q, name)]
        out[f"{q}_{name}"] = {
            "n_teams": cc["teams"], "team_ambiguous_rate": cc["teams_ambiguous"] / cc["teams"],
            "player_agreement_with_participant_order": cc["players_agree_participant_order"] / cc["players"],
            "low_start_share_by_participant_order_role": {r: cc[f"low_start {r}"] / (cc["players"] / 5) for r in V.ROLES},
            "start_gold_median_by_participant_order_role": {r: float(np.median(v)) for r, v in gold_s[(q, name)].items()}}
    out["elapsed_s"] = round(time.time() - t0, 1)
    _dump("role_start_gold_15.14.json", out)
    return out


def main(which="all"):
    for name, fn in (("evidence", run_evidence), ("kr_first", run_kr_first), ("identity", run_identity),
                     ("gold", run_gold), ("slots", run_slots), ("roles", run_roles)):
        if which in (name, "all"):
            print(name, json.dumps(fn(), ensure_ascii=False, default=str)[:6000], flush=True)


if __name__ == "__main__":
    load_rules_15x()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
