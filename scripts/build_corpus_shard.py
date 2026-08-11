"""Build one shard of the corpus tabular matrix (worker for the full-corpus run).

``build_tabular_Xy`` walks matches serially, so a full-corpus build is
single-threaded and hours long.  Every match is independent, so the corpus
is sharded by match id and the shards run as separate processes; the
decomposition driver merges them.

Each shard writes X (float32), y, groups (match id), patch, engage_ts and
the two scale-count pairs, so the merged corpus can be split, stratified,
and bootstrapped without touching the cache again.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/build_corpus_shard.py ^
        --shard 0 --num-shards 8 --n-matches 30000 --seed 7 ^
        --out-dir D:/LOL_Project/fusion_2615/corpus_shards
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--n-matches", type=int, default=None,
                        help="sample this many matches from the cache (default: all)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feature-set", default="full")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    from core.config import CACHE_DIR, cfg
    # every shard indexes a different match set, so the shared index cache
    # would only churn; dumps are irrelevant here
    cfg.FIGHT_INDEX_CACHE_ENABLED = False
    cfg.DUMP_FIGHTS = False

    from data.index_split import build_fight_index
    from train.baseline import build_tabular_Xy

    mids = sorted(p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json"))
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))
    mine = [m for i, m in enumerate(mids) if i % args.num_shards == args.shard]
    started = time.time()
    print(f"[shard {args.shard}/{args.num_shards}] matches={len(mine)}", flush=True)

    refs = build_fight_index(cache_match_ids=mine)
    print(f"[shard {args.shard}] refs={len(refs)} "
          f"({time.time() - started:.0f}s)", flush=True)

    X, y, feat_names, used = build_tabular_Xy(refs, feature_set=args.feature_set)
    elapsed = time.time() - started
    print(f"[shard {args.shard}] X={X.shape} in {elapsed:.0f}s "
          f"({len(mine) / max(elapsed, 1e-9):.1f} matches/s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"shard_{args.shard:03d}.npz"
    np.savez_compressed(
        out,
        X=X.astype(np.float32),
        y=y.astype(np.int8),
        groups=np.array([r.match_id for r in used]),
        patch=np.array([r.patch for r in used]),
        engage_ts=np.array([r.t_start_ts for r in used], dtype=np.int64),
        cluster_blue=np.array([r.det_cluster_blue for r in used], dtype=np.int16),
        cluster_red=np.array([r.det_cluster_red for r in used], dtype=np.int16),
        present_blue=np.array([r.det_present_blue for r in used], dtype=np.int16),
        present_red=np.array([r.det_present_red for r in used], dtype=np.int16),
    )
    if args.shard == 0:
        (args.out_dir / "feature_names.json").write_text(
            json.dumps({"feature_set": args.feature_set, "names": list(feat_names)}),
            encoding="utf-8",
        )
    print(f"[shard {args.shard}] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
