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
    parser.add_argument("--label-type", default=None,
                        help="override cfg.LABEL_TYPE (e.g. market_lex)")
    parser.add_argument("--tie-policy", default=None,
                        help="override cfg.LABEL_TIE_POLICY; with 'drop', "
                             "genuine draws are excluded from the shard")
    parser.add_argument("--extra-tie-policy", default=None,
                        help="tie policy while computing --extra-labels (default: same as --tie-policy); "
                             "'drop' stores -1 for draws so every label keeps its own draw mask on the common rows")
    parser.add_argument("--extra-labels", default="",
                        help="comma-separated LABEL_TYPEs computed on the same rows and stored as "
                             "y_<type> (-1 where that label is a draw), e.g. market_event,attention_value_win")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    from core.config import CACHE_DIR, cfg
    if args.label_type:
        cfg.LABEL_TYPE = args.label_type
    if args.tie_policy:
        cfg.LABEL_TIE_POLICY = args.tie_policy
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

    extra = [s.strip() for s in str(args.extra_labels or "").split(",") if s.strip()]
    extra_cols: dict[str, np.ndarray] = {}
    if extra:
        from data.cache_io import load_match_cache
        from gameplay.labels import compute_label
        from gameplay.pipeline_interp import interpolate_node_global
        default_type = str(cfg.LABEL_TYPE)
        default_tie = getattr(cfg, "LABEL_TIE_POLICY", None)
        if args.extra_tie_policy:
            cfg.LABEL_TIE_POLICY = args.extra_tie_policy
        by_match: dict[str, list[int]] = {}
        for i, r in enumerate(used):
            by_match.setdefault(r.match_id, []).append(i)
        for lt in extra:
            extra_cols[lt] = np.full(len(used), -1, dtype=np.int8)
        for mid_, idxs in by_match.items():
            pack = load_match_cache(mid_)
            if not pack:
                continue
            tm = {int(k): int(v) for k, v in (pack["meta"].get("team_map") or {}).items()}
            for lt in extra:
                cfg.LABEL_TYPE = lt
                for i in idxs:
                    r = used[i]
                    lab = compute_label(
                        pack, tm, -1,
                        engage_ts=int(r.t_start_ts),
                        label_end_ts=(int(r.label_end_ts) if int(r.label_end_ts) >= 0 else None),
                        first_kill_ts=(int(r.first_kill_ts) if int(getattr(r, "first_kill_ts", -1)) >= 0 else None),
                        last_kill_ts=(int(r.last_kill_ts) if int(getattr(r, "last_kill_ts", -1)) >= 0 else None),
                        interp_node_global=interpolate_node_global,
                        anchor_xy=((float(r.anchor_x), float(r.anchor_y)) if float(getattr(r, "anchor_x", -1.0)) >= 0 else None),
                    )
                    extra_cols[lt][i] = -1 if lab is None else int(lab)
        cfg.LABEL_TYPE = default_type
        if args.extra_tie_policy:
            if default_tie is None:
                try:
                    delattr(cfg, "LABEL_TIE_POLICY")
                except Exception:
                    pass
            else:
                cfg.LABEL_TIE_POLICY = default_tie
        for lt, col in extra_cols.items():
            ok = col >= 0
            share = float(ok.mean()) * 100.0
            pos = float((col[ok] == 1).mean()) if ok.any() else float("nan")
            agree = float((col[ok] == y[ok]).mean()) if ok.any() else float("nan")
            print(f"[shard {args.shard}] y_{lt}: labelled {share:.1f}% positives {pos:.3f} agreement with y {agree:.3f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"shard_{args.shard:03d}.npz"
    np.savez_compressed(
        out,
        **{f"y_{lt}": col for lt, col in extra_cols.items()},
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
