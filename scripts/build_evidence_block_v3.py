"""Build the evidence-corrected state block at each engagement's cutoff, next to a hold block.

For every engagement in a split, query tau = cutoff - 1 ms (the same instant the frozen X was
built at) through EvidenceStateBuilder and through the plain last-frame hold, and write both as
dense blocks keyed by engagement id.  The pilot then trains the outcome predictor on
X, X + hold block and X + evidence block: the last two differ only in whether events were used
to correct what the frame could not show, so their gap is the value of the evidence.

Per participant (ordered by role slot, blue then red) the block carries the fields the audit
showed evidence improves, plus the evidence metadata; six team differences are appended.

--pos-error-curve selects the position-fusion curve: 'legacy' (default; reproduces the v1 pilot blocks,
whose curve was calibrated on predict_test) or a scripts/calibrate_position_error_v3.py JSON measured on
disjoint matches.  --tag suffixes the output names (e.g. _v2) so earlier blocks are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = ["level_norm", "ult_level_norm", "alive", "respawn_in_s", "hp_pct", "mp_pct",
          "has_baron", "baron_remain_norm", "has_elder", "elder_remain_norm",
          "totalGold_norm", "curGold_norm", "xp_norm", "laneCS_norm", "jgCS_norm",
          "item_count", "item_gold_owned",
          "cs_attackDamage", "cs_abilityPower", "cs_armor", "cs_magicResist", "cs_healthMax", "cs_health",
          "cs_attackSpeed", "cs_movementSpeed", "x_norm", "y_norm",
          "pos_evidence_kind", "pos_evidence_age_s", "frame_age_s"]
TEAM_DIFF = ["totalGold_norm", "xp_norm", "level_norm", "alive", "item_gold_owned", "curGold_norm"]
SLOT_TAGS = ["bTOP", "bJNG", "bMID", "bBOT", "bSUP", "rTOP", "rJNG", "rMID", "rBOT", "rSUP"]


def hold_values(builder, i, pid, tau):
    """The plain last-frame state, with the same extra columns zeroed so the blocks line up."""
    fr = builder.frame_values(i, pid)
    age = (tau - int(builder.ts[i])) / 1000.0
    v = {f: fr.get(f, 0.0) for f in FIELDS}
    v.update({"respawn_in_s": 0.0, "item_count": 0.0, "item_gold_owned": 0.0,
              "pos_evidence_kind": 0.0, "pos_evidence_age_s": age, "frame_age_s": age})
    return v


def block_names():
    return [f"{tag}_{f}" for tag in SLOT_TAGS for f in FIELDS] + [f"teamdiff_{f}" for f in TEAM_DIFF]


def slot_order(pack):
    """Participant ids by role slot, blue then red."""
    slots = {int(p): int(s) for p, s in (pack["meta"].get("role_slots") or {}).items()}
    return sorted(range(1, 11), key=lambda p: slots.get(p, p - 1))


def state_rows(b, tau, order):
    """(evidence_row, hold_row) at tau in block_names() order; ValueError if tau precedes the first frame."""
    est = b.at(tau)
    i = b.frame_index(tau)
    rows = []
    for get in (lambda p: est[p], lambda p: hold_values(b, i, p, tau)):
        per = {p: get(p) for p in order}
        vals = [per[p][f] for p in order for f in FIELDS]
        blue, red = order[:5], order[5:]
        vals += [sum(per[p][f] for p in blue) - sum(per[p][f] for p in red) for f in TEAM_DIFF]
        rows.append(vals)
    return rows[0], rows[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("predict_train", "predict_test"), required=True)
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/state_value_main_50k")
    ap.add_argument("--previous-eval", type=Path, default=ROOT / "outputs/state_value_main_50k_eval")
    ap.add_argument("--cache-dir", type=Path, default=Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13"))
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/evidence_state")
    ap.add_argument("--pos-error-curve", default="legacy",
                    help="'legacy' (v1 pilot, default) or a calibrate_position_error_v3.py JSON")
    ap.add_argument("--tag", default="", help="output name suffix, e.g. _v2 -> block_<split>_v2.npz")
    ap.add_argument("--max-matches", type=int, default=0, help="smoke tests only: first N matches of the split")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOL_OUTPUT_ROOT"] = str(a.out_dir / "runtime")
    os.environ["LOL_CFG_PRESET"] = "v3.3"
    os.environ["LOL_CFG_OVERRIDES"] = json.dumps({"CACHE_DIRNAME": str(a.cache_dir.resolve()),
        "FIGHT_INDEX_CACHE_ENABLED": False, "FIGHT_INDEX_NUM_WORKERS": 1, "DUMP_FIGHTS": False})
    from data.cache_io import load_match_cache
    from gameplay.evidence_state import EvidenceStateBuilder, load_pos_error_curve, load_tables
    from scripts.calibrate_position_error_v3 import git_state, ids_sha256

    out_npz = a.out_dir / f"block_{a.split}{a.tag}.npz"
    out_report = a.out_dir / f"block_{a.split}{a.tag}_report.json"
    if a.tag and (out_npz.exists() or out_report.exists()):
        raise SystemExit(f"{out_npz} already exists; choose a new --tag")
    curve, curve_provenance = load_pos_error_curve(a.pos_error_curve)
    splits = json.loads((a.previous_eval / "match_splits.json").read_text(encoding="utf-8"))
    ids = splits[a.split][:a.max_matches] if a.max_matches > 0 else splits[a.split]
    calibration_overlap = None
    if curve_provenance["source"] not in ("legacy", "dict"):
        cal_ids = set(json.loads(Path(curve_provenance["source"]).read_text(encoding="utf-8"))["match_ids"])
        calibration_overlap = len(cal_ids & set(splits[a.split]))
        if a.split == "predict_test" and calibration_overlap:
            raise SystemExit(f"calibration matches overlap the evaluation split predict_test ({calibration_overlap})")
    names = block_names()
    tables, rows_ev, rows_hold, eids, matches, skipped = {}, [], [], [], [], {}
    started, wall_start = time.time(), time.strftime("%Y-%m-%dT%H:%M:%S")
    for k, mid in enumerate(ids, 1):
        with np.load(a.dataset / "matches" / (mid + ".npz"), allow_pickle=False) as f:
            eng_ids, cutoffs = f["engagement_id"].astype(str), f["cutoff"]
        if not len(eng_ids):
            continue
        pack = load_match_cache(mid)
        if pack is None:
            skipped[mid] = "cache missing"
            continue
        patch = str(pack["meta"]["patch"])
        if patch not in tables:
            tables[patch] = load_tables(patch)
        try:
            b = EvidenceStateBuilder(pack, patch=patch, tables=tables[patch], pos_error_curve=curve)
        except Exception as exc:                                    # unsupported pack: record, move on
            skipped[mid] = repr(exc)
            continue
        order = slot_order(pack)
        for eid, cutoff in zip(eng_ids, cutoffs):
            tau = int(cutoff) - 1
            try:
                ev_row, hold_row = state_rows(b, tau, order)
            except ValueError as exc:
                skipped[eid] = str(exc)
                continue
            rows_ev.append(ev_row)
            rows_hold.append(hold_row)
            eids.append(eid)
            matches.append(mid)
        if k % 2000 == 0:
            print(f"[block] {a.split} {k}/{len(ids)} matches, {len(eids)} engagements ({time.time()-started:.0f}s)", flush=True)
    ev = np.asarray(rows_ev, dtype=np.float32)
    hold = np.asarray(rows_hold, dtype=np.float32)
    np.savez_compressed(out_npz, id=np.array(eids), match=np.array(matches),
                        evidence=ev, hold=hold, names=np.array(names))
    out_report.write_text(json.dumps({
        "split": a.split, "engagements": len(eids), "matches": len(set(matches)), "columns": len(names),
        "skipped": skipped, "elapsed_seconds": round(time.time() - started, 1),
        "evidence_minus_hold_mean_abs": {n: float(np.abs(ev[:, j] - hold[:, j]).mean()) for j, n in enumerate(names)
                                         if n.startswith("bMID_") or n.startswith("teamdiff_")},
        "provenance": {"git": git_state(), "preset": "v3.3", "dataset": str(a.dataset),
                       "match_splits": str(a.previous_eval / "match_splits.json"), "split_matches": len(ids),
                       "split_match_ids_sha256": ids_sha256(ids), "max_matches": a.max_matches, "tag": a.tag,
                       "pos_error_curve": curve_provenance, "calibration_matches_in_this_split": calibration_overlap,
                       "wall_clock": {"start": wall_start, "end": time.strftime("%Y-%m-%dT%H:%M:%S")}}},
        indent=2), encoding="utf-8")
    print(f"[block] done: {len(eids)} engagements x {len(names)} columns, skipped {len(skipped)} ({time.time()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
