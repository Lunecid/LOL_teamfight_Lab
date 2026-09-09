"""Build resumable per-match value states and causal engagement inputs.

Requires existing caches; never downloads, rewrites, or repairs source caches.
The v3.3 detector and label window are retained. A compact input is useful for
smoke tests; the default full input reuses the existing tabular feature pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def role_for_match(mid: str, seed: int) -> str:
    u = int(hashlib.sha256(f"{seed}:split:{mid}".encode()).hexdigest()[:8], 16) / 2**32
    return "value_train" if u < .2 else "value_validation" if u < .3 else "predict_train" if u < .8 else "predict_test"


def atomic_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-matches", type=int, default=5000, help="0 = all; deterministic nested sample")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--input", choices=("full", "compact"), default="full")
    ap.add_argument("--patches", default="15.14,15.15,15.16")
    ap.add_argument("--exclude-selection", type=Path, help="Exclude every match in a previous selection.json before sampling")
    args = ap.parse_args(argv)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve()
    if cache == out or cache in out.parents:
        raise ValueError("outputs must be outside irreplaceable cache")
    os.environ["LOL_OUTPUT_ROOT"] = str(out / "runtime")
    os.environ["LOL_CFG_PRESET"] = "v3.3"
    os.environ["LOL_CFG_OVERRIDES"] = json.dumps({
        "CACHE_DIRNAME": str(cache), "FIGHT_INDEX_CACHE_ENABLED": False,
        "FIGHT_INDEX_NUM_WORKERS": 1, "DUMP_FIGHTS": False})
    from core.config import cfg, NODE_FEATURE_NAMES, CACHE_DIR
    cfg.LABEL_TIE_POLICY = "random"
    from data.cache_io import load_match_cache
    from data.index_split import build_fight_index
    from train.baseline import build_tabular_Xy
    from gameplay.labels import compute_label
    from gameplay.pipeline_interp import interpolate_node_global
    from gameplay.state_value import StateBuilder, STATE_VERSION, final_outcome
    assert CACHE_DIR.resolve() == cache
    excluded_selection = json.loads(args.exclude_selection.read_text(encoding="utf-8"))["match_ids"] if args.exclude_selection else []
    source_paths = ("scripts/build_state_value_dataset.py", "gameplay/state_value.py", "core/config.py",
                    "data/index_split.py", "train/baseline.py", "gameplay/labels.py", "gameplay/pipeline_interp.py")
    source_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    settings = {"state_version": STATE_VERSION, "input": args.input, "seed": args.seed,
                "cache_dir": str(cache), "patches": args.patches, "preset": "v3.3",
                "label_window": "existing FightRef.label_end_ts; inclusive state query end-1",
                "sampling": "sha256 rank independent of outcomes", "value_grid_ms": 60000,
                "source_sha256": source_hashes,
                "excluded_selection_sha256": hashlib.sha256(json.dumps(sorted(excluded_selection)).encode()).hexdigest()}
    settings_path = out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("dataset settings changed; use another output directory")
    atomic_json(settings_path, settings)
    excluded_ids = set(excluded_selection)
    mids = [p.name[:-10] for p in cache.glob("*.meta.json") if p.name[:-10] not in excluded_ids]
    mids.sort(key=lambda m: hashlib.sha256(f"{args.seed}:sample:{m}".encode()).hexdigest())
    mids = mids[:args.n_matches] if args.n_matches else mids
    atomic_json(out / "selection.json", {"match_ids": mids})
    manifest = {"status": "building", "settings": settings, "requested_matches": len(mids),
                "completed": [], "excluded": {}}
    match_dir = out / "matches"
    match_dir.mkdir(exist_ok=True)
    schema_path = out / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else None
    started = time.time()
    for n, mid in enumerate(mids, 1):
        dest = match_dir / f"{mid}.npz"
        if dest.exists():
            # A file becomes visible only after successful atomic write.
            with np.load(dest, allow_pickle=False) as z:
                if str(z["match_id"]) != mid:
                    raise ValueError("checkpoint match ID mismatch")
            manifest["completed"].append(mid)
            continue
        pack = load_match_cache(mid)
        if pack is None:
            manifest["excluded"][mid] = "unreadable_or_incompatible_cache"
            continue
        if args.patches and pack["meta"]["patch"] not in args.patches.split(","):
            manifest["excluded"][mid] = "patch_not_selected"
            continue
        try:
            winner, terminal = final_outcome(pack["events"])
            builder = StateBuilder(pack, NODE_FEATURE_NAMES)
        except ValueError as exc:
            manifest["excluded"][mid] = str(exc)
            continue
        # Terminal frame and final outcome never enter the value input.
        grid = list(range(120000, min(terminal, int(builder.ts[-1]) + 1), 60000))
        if not grid:
            manifest["excluded"][mid] = "no_nonterminal_state"
            continue
        refs = build_fight_index(cache_match_ids=[mid])
        if args.input == "full" and refs:
            X, _, x_names, used = build_tabular_Xy(refs, feature_set="full")
        else:
            X, used, x_names = None, refs, None
        pre, post, before, kept, indices = [], [], [], [], []
        rejected = []
        for row, ref in enumerate(used):
            q, end = int(ref.t_start_ts), int(ref.label_end_ts)
            if q < 30000 or end <= q or end > terminal:
                rejected.append([q, "invalid_or_terminal_window"])
                continue
            try:
                a, b, c = builder.at(q), builder.at(end - 1), builder.at(q - 30000)
            except ValueError as exc:
                rejected.append([q, str(exc)])
                continue
            pre.append(a); post.append(b); before.append(c); kept.append(ref); indices.append(row)
        grid_states = [builder.at(t) for t in grid]
        names = list(grid_states[0].values)
        if args.input == "compact":
            x_names = ["now_" + k for k in names] + ["history30s_" + k for k in names]
            X = np.asarray([list(a.values.values()) + list(c.values.values()) for a, c in zip(pre, before)], dtype=np.float32).reshape(len(kept), len(x_names))
        elif X is not None:
            X = np.asarray(X[indices], dtype=np.float32)
        # Matches without engagements still contribute to the valuation dataset.
        new_schema = {"state_names": names, "input_names": list(x_names) if x_names else None}
        if schema is None or schema["input_names"] is None:
            schema = new_schema
            atomic_json(schema_path, schema)
        elif schema["state_names"] != names or (x_names and schema["input_names"] != list(x_names)):
            raise ValueError("feature order changed across matches")
        state_array = lambda states: np.asarray([list(s.values.values()) for s in states], dtype=np.float32).reshape(len(states), len(names))
        labels = {}
        old_type, old_tie = cfg.LABEL_TYPE, cfg.LABEL_TIE_POLICY
        try:
            cfg.LABEL_TIE_POLICY = "drop"
            for lt in ("market_event", "attention_value_win"):
                cfg.LABEL_TYPE = lt
                vals = []
                for r in kept:
                    lab = compute_label(pack, builder.tm, -1, engage_ts=r.t_start_ts, label_end_ts=r.label_end_ts,
                        first_kill_ts=r.first_kill_ts, last_kill_ts=r.last_kill_ts,
                        anchor_xy=(r.anchor_x, r.anchor_y), interp_node_global=interpolate_node_global)
                    vals.append(-1 if lab is None else int(lab))
                labels["y_" + lt] = np.asarray(vals, dtype=np.int8)
        finally:
            cfg.LABEL_TYPE, cfg.LABEL_TIE_POLICY = old_type, old_tie
        ids = [f"{mid}:{r.t_start_ts}" for r in kept]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate engagement IDs")
        rec = dict(match_id=mid, role=role_for_match(mid, args.seed), patch=pack["meta"]["patch"],
                   winner=np.int8(winner), value_states=state_array(grid_states), value_times=np.asarray(grid),
                   pre=state_array(pre), post=state_array(post),
                   X=X if X is not None else np.empty((0, 0), dtype=np.float32),
                   engagement_id=np.asarray(ids, dtype=str), cutoff=np.asarray([r.t_start_ts for r in kept], dtype=np.int64),
                   post_time=np.asarray([s.query_ms for s in post], dtype=np.int64),
                   pre_snapshot=np.asarray([s.snapshot_ms for s in pre], dtype=np.int64),
                   post_snapshot=np.asarray([s.snapshot_ms for s in post], dtype=np.int64),
                   scale=np.asarray([min(r.det_cluster_blue, r.det_cluster_red) for r in kept]),
                   rejected=json.dumps(rejected), detector_refs=len(refs), used_refs=len(used), **labels)
        tmp = dest.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **rec)
        tmp.replace(dest)
        manifest["completed"].append(mid)
        if n % 25 == 0:
            atomic_json(out / "manifest.json", manifest)
            print(f"[build] {n}/{len(mids)} elapsed={time.time()-started:.0f}s", flush=True)
    manifest["status"] = "complete"
    atomic_json(out / "manifest.json", manifest)
    print(f"[complete] cached={len(manifest['completed'])} excluded={len(manifest['excluded'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
