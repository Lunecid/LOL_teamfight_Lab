"""Stage 1 follow-up (v4-exact, task X): checks behind the item / StateV3 decisions, patch 15.14 only.

Pre-specified sample: 500 patch-15.14 matches of match_cache_fresh_v3_engage_status13 (patch index rows == "15.14",
sorted, random.Random(20260925).shuffle, first 500: the sample of stage1/state_value_v3/validate_state_value_v3.py).
Query times: per match 8 uniform ms in [first frame, last frame] and 2 in [60 s, 540 s) (random.Random(7)), as in
tests/test_exact_state_v3.py.  Single process.  No other patch's engagement data is read (15.15 appears only as a
patch-rule lookup in config/game_rules/objective_rules.json, without events).

Writes
  <OUT>/stage1/roles/agreement_15.14.json
      role inference (gameplay.role_inference.infer_roles at the full horizon 8:59.999 and at the 2 early query
      times) against participant order (the StateV3 slots), plus meta role_slots against participant order.
  <OUT>/stage1/state_value_v3/constant_columns_15.14.json, state_v3_columns.txt
      STATE_V3_COLUMNS constant over the 5000 rows; the dropped objective fields (OBJ_METADATA_FIELDS, now metadata)
      over the same rows and as 15.15 rule values; the full-vs-truncated (future information) check on every row.
  <OUT>/stage1/item_state/lifeline_unknown_15.14.json
      unknown item id rate of the ITEM_* events (check_unknown_rate), 32xxxx SR-variant ids seen, lifeline flag
      prevalence over the player rows.
"""
from __future__ import annotations

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
from gameplay import state_value_v3 as sv  # noqa: E402
from gameplay.item_state import (EFFECT_FLAGS, UnknownItemRateError, check_unknown_rate, load_effect_flags,  # noqa: E402
                                 load_item_table_v2)
from gameplay.objective_timers import ObjectiveTimeline  # noqa: E402
from gameplay.role_inference import HORIZON_MS, ROLES, infer_roles  # noqa: E402
from gameplay.state_value_v3 import SLOT_PREFIXES, STATE_V3_COLUMNS, StateBuilderV3  # noqa: E402

CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
PATCH_INDEX = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json")
OUT = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925/stage1")
PATCH = "15.14"
N_MATCHES, SEED, T_SEED = 500, 20260925, 7
NEAR_CONSTANT_SHARE = 0.001
ITEM_TYPES = ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO")


def sample_ids():
    idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in idx.items() if v == PATCH)
    random.Random(SEED).shuffle(ids)
    return ids[:N_MATCHES]


def load(m):
    meta = json.loads((CACHE / f"{m}.meta.json").read_text(encoding="utf-8"))
    if str(meta.get("patch")) != PATCH:
        raise RuntimeError(f"{m}: patch {meta.get('patch')} != {PATCH}")
    ev = json.loads((CACHE / f"{m}.events.json").read_text(encoding="utf-8"))
    with np.load(CACHE / f"{m}.npz") as z:
        return {"minute_ts": z["minute_ts"], "node_minute": z["node_minute"], "events": ev, "meta": meta}


def trunc(pack, t):
    keep = np.asarray(pack["minute_ts"]) <= t
    return {"minute_ts": pack["minute_ts"][keep], "node_minute": pack["node_minute"][keep],
            "events": [e for e in pack["events"] if int(e["timestamp"]) <= t], "meta": pack["meta"]}


def same(a, b):
    return list(a) == list(b) and all(x == y or (math.isnan(x) and math.isnan(y)) for x, y in zip(a.values(), b.values()))


class RoleTally:
    def __init__(self):
        self.c = Counter()
        self.by_role = defaultdict(lambda: [0, 0])
        self.conf = defaultdict(Counter)            # participant-order role -> inferred role counts
        self.team_split = {"confident": [0, 0], "ambiguous": [0, 0]}
        self.examples = []

    def add(self, m, ra, slots):
        self.c["matches"] += 1
        all10 = True
        for team in (100, 200):
            tr = ra.teams[team]
            pids = [p for p in slots if (slots[p] < 5) == (team == 100)]
            ok = all(ra.slot_by_pid[p] == slots[p] for p in pids)
            all10 &= ok
            self.c["teams"] += 1
            self.c["teams_agree"] += int(ok)
            key = "ambiguous" if tr.ambiguous else "confident"
            self.team_split[key][0] += int(ok)
            self.team_split[key][1] += 1
            if not ok and len(self.examples) < 15:
                self.examples.append({"match": m, "team": team, "ambiguous": bool(tr.ambiguous),
                                      "reasons": list(tr.reasons),
                                      "inferred_by_pid": {str(p): ra.roles[p] for p in sorted(pids)}})
        self.c["matches_agree"] += int(all10)
        for p, s in slots.items():
            po_role = ROLES[s % 5]
            self.c["players"] += 1
            hit = ra.slot_by_pid[p] == s
            self.c["players_agree"] += int(hit)
            self.by_role[po_role][0] += int(hit)
            self.by_role[po_role][1] += 1
            self.conf[po_role][ra.roles[p]] += 1

    def summary(self):
        c = self.c
        return {
            "n_matches": c["matches"], "n_teams": c["teams"], "n_players": c["players"],
            "player_agreement": c["players_agree"] / max(1, c["players"]),
            "team_agreement_all5": c["teams_agree"] / max(1, c["teams"]),
            "match_agreement_all10": c["matches_agree"] / max(1, c["matches"]),
            "agreement_by_participant_order_role": {r: v[0] / max(1, v[1]) for r, v in self.by_role.items()},
            "team_agreement_confident": self.team_split["confident"][0] / max(1, self.team_split["confident"][1]),
            "n_teams_confident": self.team_split["confident"][1],
            "team_agreement_ambiguous": self.team_split["ambiguous"][0] / max(1, self.team_split["ambiguous"][1]),
            "n_teams_ambiguous": self.team_split["ambiguous"][1],
            "confusion_participant_order_to_inferred": {r: dict(self.conf[r]) for r in ROLES},
            "disagreement_examples": self.examples,
        }


def main():
    t0 = time.time()
    ids = sample_ids()
    rng = random.Random(T_SEED)
    table = load_item_table_v2(PATCH)
    flags = load_effect_flags(table, patch=PATCH)

    role_full, role_early = RoleTally(), RoleTally()
    role_fail = Counter()
    meta_agree = Counter()
    role_slot_swaps = []
    rows = []                                      # value vectors in STATE_V3_COLUMNS order
    obj_meta_values = defaultdict(set)
    mism = []
    unknown_counts, n_item_obs = Counter(), 0
    variant_ids = Counter()
    lifeline_rows = Counter()
    lifeline_by_id = Counter()
    player_rows = 0
    for m in ids:
        pack = load(m)
        b = StateBuilderV3(pack, PATCH)
        slots = b.slot_by_pid
        meta_agree[str(b.role_slots_meta_agree)] += 1
        # --- roles: inference vs participant order
        try:
            ra_full = infer_roles(pack, None)
            role_full.add(m, ra_full, slots)
        except (KeyError, ValueError, IndexError, TypeError) as e:
            ra_full = None
            role_fail[f"full:{type(e).__name__}"] += 1
        if b.role_slots_meta_agree is False:
            rs = {int(k): int(v) for k, v in pack["meta"]["role_slots"].items()}
            for team, lo in ((100, 0), (200, 5)):
                pids = [q for q in slots if lo <= slots[q] < lo + 5]
                if any(rs[q] != slots[q] for q in pids):
                    role_slot_swaps.append({
                        "match": m, "team": team,
                        "slot_participant_order_vs_meta_role_slots_vs_inferred": {
                            str(q): [slots[q], rs[q], None if ra_full is None else ra_full.slot_by_pid[q]]
                            for q in pids if rs[q] != slots[q]},
                        "inferred_equals_meta_role_slots": None if ra_full is None else all(ra_full.slot_by_pid[q] == rs[q] for q in pids),
                        "inferred_equals_participant_order": None if ra_full is None else all(ra_full.slot_by_pid[q] == slots[q] for q in pids)})
        ts = pack["minute_ts"]
        times = [rng.randint(int(ts[0]), int(ts[-1])) for _ in range(8)] + [rng.randint(60_000, 539_999) for _ in range(2)]
        for t in times[8:]:
            try:
                role_early.add(m, infer_roles(pack, t), slots)
            except (KeyError, ValueError, IndexError, TypeError) as e:
                role_fail[f"early:{type(e).__name__}"] += 1
        # --- item ids in events
        for e in pack["events"]:
            if e.get("type") not in ITEM_TYPES:
                continue
            for key in ("itemId", "beforeId", "afterId"):
                iid = int(e.get(key, 0) or 0)
                if iid:
                    n_item_obs += 1
                    if iid not in table:
                        unknown_counts[iid] += 1
                    if 320_000 <= iid < 330_000:
                        variant_ids[iid] += 1
        # --- states
        for t in times:
            st = b.at(t)
            tr = StateBuilderV3(trunc(pack, t), PATCH).at(t)
            if not (same(st.values, tr.values) and st.slot_by_pid == tr.slot_by_pid
                    and st.objective_rule_flags == tr.objective_rule_flags):
                mism.append([m, t])
            rows.append([st.values[c] for c in STATE_V3_COLUMNS])
            for k, v in st.objective_rule_flags.items():
                obj_meta_values[k].add(v)
            for s, pre in enumerate(SLOT_PREFIXES):
                player_rows += 1
                lifeline_rows[int(st.values[pre + "itm_flag_lifeline"])] += 1
            for p in range(1, 11):
                for iid in set(b._item_index.inventory(p, t)) & flags["lifeline"]:
                    lifeline_by_id[iid] += 1

    X = np.asarray(rows, dtype=float)
    constant, near = [], []
    for j, c in enumerate(STATE_V3_COLUMNS):
        col = X[:, j]
        fin = col[np.isfinite(col)]
        vals = np.unique(fin)
        if len(vals) <= 1 and np.isfinite(col).all():
            constant.append({"column": c, "value": float(vals[0]) if len(vals) else None})
        elif len(vals) >= 2:
            mode_share = max(np.mean(fin == v) for v in vals[:50]) if len(vals) <= 50 else 0.0
            if mode_share >= 1 - NEAR_CONSTANT_SHARE:
                near.append({"column": c, "n_distinct": int(len(vals)), "share_not_mode": float(1 - mode_share)})
    # group constant columns by field for readability
    by_field = defaultdict(list)
    for d in constant:
        c = d["column"]
        pre = next((p for p in SLOT_PREFIXES if c.startswith(p)), None)
        by_field[c[len(pre):] if pre else c].append(d["value"])
    tm = {p: (100 if p <= 5 else 200) for p in range(1, 11)}
    rule_1515 = {k: v for k, v in ObjectiveTimeline([], "15.15", tm).state(600_000).items() if k in sv.OBJ_METADATA_FIELDS}
    rule_1514 = {k: v for k, v in ObjectiveTimeline([], "15.14", tm).state(600_000).items() if k in sv.OBJ_METADATA_FIELDS}

    sample_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    common = {"patch": PATCH, "n_matches": len(ids), "sample_seed": SEED, "time_seed": T_SEED,
              "sample_sha256": sample_sha, "script": "scripts/exact_v4/ev4_s1x_state_v3_checks.py"}

    roles_out = dict(common)
    roles_out.update({
        "slots_used_by_statev3": "participant order (team_map sorted by (team, pid)); role inference is validation only",
        "meta_role_slots_vs_participant_order": dict(meta_agree),
        "meta_role_slots_note": "meta role_slots = core.roles.get_role_slots_from_detail (Match-V5 teamPosition, "
                                "Riot's post-game position); participant order = champ-select position",
        "teams_where_meta_role_slots_differ": role_slot_swaps,
        "n_teams_where_meta_role_slots_differ": len(role_slot_swaps),
        "n_of_those_inferred_equals_meta_role_slots": sum(bool(r["inferred_equals_meta_role_slots"]) for r in role_slot_swaps),
        "inference_failures": dict(role_fail),
        "at_full_horizon_ms": HORIZON_MS,
        "full_horizon": role_full.summary(),
        "early_t_in_60_540s": role_early.summary(),
        "note": "agreement = inferred slot == participant-order slot. Participant order is the champ-select assigned "
                "position (author decision 2026-09-25; equals teamPosition in 553/553 16.15 matches); a disagreement is "
                "an inference error or an off-role / lane-swap game, not a slot error.",
    })
    (OUT / "roles").mkdir(parents=True, exist_ok=True)
    (OUT / "roles" / "agreement_15.14.json").write_text(json.dumps(roles_out, indent=2), encoding="utf-8")

    const_out = dict(common)
    const_out.update({
        "n_rows": int(X.shape[0]), "n_columns": len(STATE_V3_COLUMNS), "name_hash": sv.STATE_V3_NAME_HASH,
        "column_blocks": sv.column_blocks(),
        "future_info_mismatches": len(mism), "mismatch_examples": mism[:10],
        "dropped_objective_fields_now_metadata": {
            k: {"values_in_15.14_rows": sorted(obj_meta_values[k]), "constant_in_15.14_rows": len(obj_meta_values[k]) == 1,
                "rule_value_15.14": rule_1514[k], "rule_value_15.15": rule_1515[k]}
            for k in sv.OBJ_METADATA_FIELDS},
        "dropped_role_columns_now_metadata": ["role_fallback", "role_ambiguous_blue", "role_ambiguous_red", "role_final"],
        "other_constant_columns": constant,
        "other_constant_columns_by_field": {k: {"n_columns": len(v), "values": sorted(set(v))} for k, v in by_field.items()},
        "near_constant_columns": near,
        "near_constant_rule": f"finite, >= 2 distinct values, one value in >= {1 - NEAR_CONSTANT_SHARE:.1%} of rows",
        "note": "Constant columns other than the dropped objective fields are REPORTED, not dropped.",
        "elapsed_s": round(time.time() - t0, 1),
    })
    (OUT / "state_value_v3").mkdir(parents=True, exist_ok=True)
    (OUT / "state_value_v3" / "constant_columns_15.14.json").write_text(json.dumps(const_out, indent=2), encoding="utf-8")
    (OUT / "state_value_v3" / "state_v3_columns.txt").write_text("\n".join(STATE_V3_COLUMNS) + "\n", encoding="utf-8")

    try:
        rate = check_unknown_rate(unknown_counts, n_item_obs)
        halted = False
    except UnknownItemRateError:
        rate, halted = sum(unknown_counts.values()) / max(1, n_item_obs), True
    item_out = dict(common)
    item_out.update({
        "effect_flags": list(EFFECT_FLAGS), "lifeline_ids_15.14": sorted(flags["lifeline"]),
        "item_id_observations": n_item_obs, "unknown_ids": {str(k): v for k, v in unknown_counts.most_common()},
        "unknown_rate": rate, "unknown_rate_max": 0.005, "check_unknown_rate_raised": halted,
        "sr_variant_32xxxx_ids_in_events": {str(k): v for k, v in variant_ids.most_common()},
        "player_rows": player_rows, "lifeline_flag_share": lifeline_rows[1] / max(1, player_rows),
        "lifeline_holdings_by_id": {str(k): v for k, v in lifeline_by_id.most_common()},
    })
    (OUT / "item_state").mkdir(parents=True, exist_ok=True)
    (OUT / "item_state" / "lifeline_unknown_15.14.json").write_text(json.dumps(item_out, indent=2), encoding="utf-8")
    print(json.dumps({"roles_full": {k: v for k, v in role_full.summary().items() if k not in ("disagreement_examples",)},
                      "roles_early": {k: role_early.summary()[k] for k in ("player_agreement", "team_agreement_all5")},
                      "meta_agree": dict(meta_agree), "role_fail": dict(role_fail),
                      "mism": len(mism), "constant": by_field and {k: len(v) for k, v in by_field.items()},
                      "near": near, "objmeta": {k: sorted(v) for k, v in obj_meta_values.items()},
                      "unknown_rate": rate, "n_item_obs": n_item_obs, "variants": dict(variant_ids),
                      "lifeline_share": lifeline_rows[1] / max(1, player_rows), "lifeline_ids": dict(lifeline_by_id),
                      "elapsed": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
