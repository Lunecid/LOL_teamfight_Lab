"""Do shop events inflate the participant counts?

Interaction actors are placed by interpolating their own 5 s position
because the timeline gives no coordinates for most event types.  Shop
events (ITEM_PURCHASED / ITEM_SOLD / ITEM_UNDO) fire at the fountain, so a
fight near the base can count a player who is shopping as a participant --
and participant counts are what the pick/skirmish/teamfight classes are
built from.

This detects the same matches twice, with and without shop events feeding
the interaction actor set, and reports how many engagements change scale
class.  Detection runs serially with the fight-index disk cache disabled so
the runtime override provably reaches the detector.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_shop_event_sensitivity.py ^
        --n-matches 3000 --seed 7 ^
        --output D:/LOL_Project/fusion_2615/features/shop_event_sensitivity.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASSES = ("pick", "skirmish", "teamfight")


def index_by_key(refs) -> dict[tuple[str, int], object]:
    return {(r.match_id, int(r.t_start_ts)): r for r in refs}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-matches", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    from core.config import CACHE_DIR, cfg
    cfg.FIGHT_INDEX_CACHE_ENABLED = False
    cfg.FIGHT_INDEX_NUM_WORKERS = 1
    cfg.DUMP_FIGHTS = False

    from data.index_split import build_fight_index

    mids = sorted(p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json"))
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))

    runs = {}
    for label, exclude in (("with_shop", False), ("without_shop", True)):
        cfg.TF2_EXCLUDE_SHOP_INTERACTIONS = exclude
        refs = build_fight_index(cache_match_ids=mids)
        runs[label] = index_by_key(refs)
        counts = collections.Counter(r.fight_scale for r in refs)
        print(f"[{label}] refs={len(refs)} {dict(counts)}", flush=True)
    cfg.TF2_EXCLUDE_SHOP_INTERACTIONS = False

    base, alt = runs["with_shop"], runs["without_shop"]
    shared = sorted(set(base) & set(alt))
    print(f"engagements present in both runs: {len(shared)} "
          f"(with_shop {len(base)}, without_shop {len(alt)})")

    changed = collections.Counter()
    drop_blue, drop_red = [], []
    for key in shared:
        a, b = base[key], alt[key]
        drop_blue.append(a.det_cluster_blue - b.det_cluster_blue)
        drop_red.append(a.det_cluster_red - b.det_cluster_red)
        if a.fight_scale != b.fight_scale:
            changed[(a.fight_scale, b.fight_scale)] += 1

    n_changed = sum(changed.values())
    drop = np.array(drop_blue + drop_red, dtype=float)
    results = {
        "n_matches": len(mids),
        "n_engagements_with_shop": len(base),
        "n_engagements_without_shop": len(alt),
        "n_shared": len(shared),
        "class_shares": {
            label: {k: v / max(1, len(run)) for k, v in
                    collections.Counter(r.fight_scale for r in run.values()).items()}
            for label, run in runs.items()
        },
        "n_class_changes": n_changed,
        "class_change_rate": n_changed / max(1, len(shared)),
        "class_transitions": {f"{a}->{b}": c for (a, b), c in changed.most_common()},
        "participant_drop_per_team": {
            "mean": float(drop.mean()),
            "share_nonzero": float(np.mean(drop != 0)),
            "max": float(drop.max()) if len(drop) else 0.0,
        },
    }
    print(json.dumps({k: results[k] for k in
                      ("class_shares", "n_class_changes", "class_change_rate",
                       "class_transitions", "participant_drop_per_team")}, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
