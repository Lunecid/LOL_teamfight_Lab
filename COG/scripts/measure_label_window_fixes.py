"""Impact of the two label-contract changes on real engagements (external audit items 1 and 4).

For a seeded match sample under the v3 definition:
  endpoint     share of engagements whose fight outlasts the minimum horizon (last kill >= cutoff +
               H), i.e. whose label window end moved by +1 ms, and the label flip rate among them
               when the window is closed on the last kill (old) versus one past it (new)
  attribution  label agreement between map-wide ("@window") and engagement-attributed events, the
               share of engagements whose window contains far events, and flips by scale class

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/measure_label_window_fixes.py --n-matches 553 \\
        --output D:/LOL_Project/fusion_2615/features/label_window_fixes.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V3 = {"TF2_KILL_CLUSTER_GAP_MS": 13700, "CLUSTER_MAX_DIAMETER": 4264.0, "TF2_VALIDITY_RADIUS": 1600.0,
      "TF2_ENGAGE_PRE_KILL_MS": 15000, "FIGHT_HORIZON_SEC": 35, "LABEL_TIE_POLICY": "drop",
      "FIGHT_INDEX_NUM_WORKERS": 1, "FIGHT_INDEX_CACHE_ENABLED": False, "DUMP_FIGHTS": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-matches", type=int, default=553)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--label", default="market_event")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    from core.config import CACHE_DIR, cfg
    for k, v in V3.items():
        setattr(cfg, k, v)
    from data.cache_io import load_match_cache
    from data.index_split import build_fight_index
    from gameplay.labels import compute_label, attribute_events
    from data.events_index import _events_in_window
    from gameplay.pipeline_interp import interpolate_node_global

    mids = [p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json")]
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))
    t0 = time.time()
    refs = build_fight_index(cache_match_ids=mids)
    H = int(cfg.FIGHT_HORIZON_SEC) * 1000
    stats = collections.Counter(); flips_by_class = collections.Counter(); n_by_class = collections.Counter()
    far_share = []
    for r in refs:
        pack = load_match_cache(r.match_id)
        if not pack:
            continue
        tm = {int(k): int(v) for k, v in (pack["meta"].get("team_map") or {}).items()}
        tau, last = int(r.t_start_ts), int(r.last_kill_ts)
        anchor = (float(r.anchor_x), float(r.anchor_y)) if r.anchor_x >= 0 else None
        common = dict(engage_ts=tau, first_kill_ts=int(r.first_kill_ts), last_kill_ts=last, interp_node_global=interpolate_node_global)
        stats["engagements"] += 1
        smaller = min(int(r.det_cluster_blue), int(r.det_cluster_red))
        cls = "pick" if smaller <= 1 else ("skirmish" if smaller <= 3 else "teamfight")
        n_by_class[cls] += 1
        # --- endpoint: fights that outlast the minimum horizon
        if last >= tau + H:
            stats["outlast_horizon"] += 1
            cfg.LABEL_TYPE = args.label
            y_old = compute_label(pack, tm, -1, label_end_ts=last, anchor_xy=anchor, **common)        # old: window closed on the last kill
            y_new = compute_label(pack, tm, -1, label_end_ts=last + 1, anchor_xy=anchor, **common)    # new: last kill included
            if y_old != y_new:
                stats["endpoint_flip"] += 1
        # --- attribution: window vs engagement on the (new) window
        end = int(r.label_end_ts) if int(r.label_end_ts) > 0 else max(last + 1, tau + H)
        cfg.LABEL_TYPE = f"{args.label}@window"
        y_win = compute_label(pack, tm, -1, label_end_ts=end, anchor_xy=anchor, **common)
        cfg.LABEL_TYPE = f"{args.label}@engagement"
        y_eng = compute_label(pack, tm, -1, label_end_ts=end, anchor_xy=anchor, **common)
        evs = _events_in_window(pack, tau, end)
        positioned = [e for e in evs if isinstance(e.get("position"), dict) and str(e.get("type", "")).upper() in
                      ("CHAMPION_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED", "ELITE_MONSTER_KILL")]
        kept = attribute_events(positioned, anchor, "engagement")
        if positioned:
            far_share.append(1.0 - len(kept) / len(positioned))
            if len(kept) < len(positioned):
                stats["has_far_events"] += 1
        if y_win is not None and y_eng is not None:
            stats["both_labelled"] += 1
            if y_win != y_eng:
                stats["attribution_flip"] += 1; flips_by_class[cls] += 1
        elif (y_win is None) != (y_eng is None):
            stats["draw_status_changed"] += 1
    n = max(stats["engagements"], 1)
    out = {"n_matches": len(mids), "n_engagements": int(n), "label": args.label,
           "endpoint": {"outlast_horizon_share": stats["outlast_horizon"] / n,
                        "flip_share_among_outlasting": stats["endpoint_flip"] / max(stats["outlast_horizon"], 1),
                        "flip_share_overall": stats["endpoint_flip"] / n},
           "attribution": {"has_far_events_share": stats["has_far_events"] / n,
                           "mean_far_event_share": float(np.mean(far_share)) if far_share else 0.0,
                           "flip_share_among_both_labelled": stats["attribution_flip"] / max(stats["both_labelled"], 1),
                           "draw_status_changed_share": stats["draw_status_changed"] / n,
                           "flip_share_by_class": {c: flips_by_class[c] / max(n_by_class[c], 1) for c in ("pick", "skirmish", "teamfight")},
                           "n_by_class": dict(n_by_class)},
           "elapsed_s": round(time.time() - t0)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
