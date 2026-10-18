"""Full-corpus engagement-outcome predictability, decomposed by scale.

The ToG extension's headline experiment.  Merges the corpus shards built by
build_corpus_shard.py, fits the paper's tabular baseline once under
match-grouped folds, then evaluates the *frozen* out-of-fold predictions
inside each scale class.  Models are never refit per class, so class sample
size cannot confound the comparison.

Scale classes use observed participation (min(blue, red) >= 3 teamfight,
== 2 skirmish, <= 1 pick) -- a post-hoc property, reported as a subgroup
breakdown and never fed to a model.  Presence at the cutoff is reported
alongside as the pre-fight counterpart.  See docs/ENGAGEMENT_SCALE_DEFINITION.md.

Memory: constant columns (a third of the 7,105) are dropped during the
merge, and the merge streams shard by shard, so peak usage stays near the
final matrix rather than twice it.

    python scripts/run_scale_decomposition.py ^
        --shards D:/LOL_Project/fusion_2615/corpus_shards ^
        --output D:/LOL_Project/fusion_2615/features/scale_decomposition.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

SEED = 7
CLASSES = ("pick", "skirmish", "teamfight")


def scale_class(blue: np.ndarray, red: np.ndarray, teamfight_min: int = 3) -> np.ndarray:
    """pick: smaller side <= 1; skirmish: 2 .. teamfight_min-1; teamfight: >= teamfight_min.

    The published decomposition used teamfight_min = 3; the v3 definition
    (docs/DEFINITION_EVIDENCE.md section 17) uses 4, the {4,5}x{4,5} block.
    """
    smaller = np.minimum(blue, red)
    out = np.full(len(smaller), "unknown", dtype=object)
    out[smaller <= 1] = "pick"
    out[(smaller >= 2) & (smaller < teamfight_min)] = "skirmish"
    out[smaller >= teamfight_min] = "teamfight"
    out[(blue < 0) | (red < 0)] = "unknown"
    return out


def verify_shard_manifest(shard_dir: Path, allow_partial: bool = False) -> dict:
    """Refuse to evaluate an incomplete or mixed corpus.

    The build writes manifest.json with the expected shard count, every child's return code
    and the feature-name hash; the shard files on disk must match it exactly.
    """
    paths = sorted(shard_dir.glob("shard_*.npz"))
    mp = shard_dir / "manifest.json"
    if not mp.exists():
        if allow_partial:
            print("WARNING: no manifest.json; evaluating whatever shards are present", flush=True)
            return {}
        raise SystemExit(f"{shard_dir} has no manifest.json (pass --allow-partial to evaluate anyway)")
    m = json.load(open(mp, encoding="utf-8"))
    problems = []
    if not m.get("complete", False):
        problems.append("manifest marks the build incomplete")
    expected = {f"shard_{int(i):03d}.npz" for i in range(int(m.get("num_shards", 0)))}
    found = {p.name for p in paths}
    if found != expected:
        problems.append(f"shard files differ from manifest: missing={sorted(expected - found)} extra={sorted(found - expected)}")
    bad = [i for i, s in (m.get("shards") or {}).items() if int(s.get("rc", 1)) != 0]
    if bad:
        problems.append(f"shards with non-zero rc: {bad}")
    names_path = shard_dir / "feature_names.json"
    if m.get("feature_names_sha1") and names_path.exists():
        import hashlib
        if hashlib.sha1(open(names_path, "rb").read()).hexdigest() != m["feature_names_sha1"]:
            problems.append("feature_names.json hash differs from manifest")
    if problems:
        msg = "; ".join(problems)
        if allow_partial:
            print("WARNING: " + msg, flush=True)
        else:
            raise SystemExit("corpus integrity: " + msg)
    return m


def merge_shards(shard_dir: Path, matrix_path: Path, y_key: str = "y") -> dict:
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {shard_dir}")
    print(f"shards: {len(paths)}")

    # pass 1: row count and per-column range, one shard resident at a time
    total, n_features = 0, None
    col_min = col_max = None
    for path in paths:
        with np.load(path, allow_pickle=True) as blob:
            X = blob["X"]
            total += X.shape[0]
            if n_features is None:
                n_features = X.shape[1]
                col_min, col_max = X.min(axis=0), X.max(axis=0)
            else:
                np.minimum(col_min, X.min(axis=0), out=col_min)
                np.maximum(col_max, X.max(axis=0), out=col_max)
    keep = np.flatnonzero(col_max > col_min)
    names_path = shard_dir / "feature_names.json"
    all_names = json.load(open(names_path, encoding="utf-8"))["names"] if names_path.exists() else None
    kept_names = [all_names[j] for j in keep] if all_names and len(all_names) == n_features else None
    print(f"rows={total} features={n_features} non-constant={len(keep)}"
          f" ({len(keep) / n_features * 100:.1f}%)")

    # the merged matrix is tens of GB; back it with a disk memmap so the base
    # array lives in evictable page cache and only the per-fold training slice
    # occupies resident memory
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    X_all = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(total, len(keep))
    )
    y_all = np.empty(total, dtype=np.int8)
    parts: dict[str, list] = {k: [] for k in
                              ("groups", "patch", "engage_ts", "cluster_blue",
                               "cluster_red", "present_blue", "present_red")}
    offset = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as blob:
            X = blob["X"]
            rows = X.shape[0]
            X_all[offset:offset + rows] = X[:, keep]
            y_all[offset:offset + rows] = blob[y_key]
            for name in parts:
                parts[name].append(blob[name])
            offset += rows
    assert offset == total
    X_all.flush()
    merged = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    merged["X"] = np.load(matrix_path, mmap_mode="r")
    merged["feature_names"] = kept_names
    merged["y"], merged["keep"] = y_all, keep
    return merged


def oof_predictions(X, y, groups, n_splits=5, categorical=None) -> np.ndarray:
    """Paper's tabular learner, one config: at ~10^6 rows the small-sample
    regularization search that the fusion study needed is unnecessary."""
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), 1):
        model = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED,
            n_jobs=-1, verbose=-1,
        )
        X_train = np.ascontiguousarray(X[train])  # materialize once, free early
        if categorical:
            model.fit(X_train, y[train], categorical_feature=list(categorical))
        else:
            model.fit(X_train, y[train])
        del X_train
        oof[test] = model.predict_proba(np.ascontiguousarray(X[test]))[:, 1]
        print(f"  fold {fold}/{n_splits}: train={len(train)} test={len(test)}"
              f" auc={roc_auc_score(y[test], oof[test]):.4f}", flush=True)
    assert not np.isnan(oof).any()
    return oof


def cluster_bootstrap(y, pred, groups, masks, n_boot=1000):
    """Per-class AUC intervals and pairwise gaps, resampling matches."""
    rng = np.random.default_rng(SEED)
    unique, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    ends = np.append(starts[1:], len(order))
    collected = {name: [] for name in masks}
    gaps: list[float] = []
    for _ in range(n_boot):
        sample = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([order[starts[g]:ends[g]] for g in sample])
        per_class = {}
        for name, mask in masks.items():
            sub = idx[mask[idx]]
            if len(sub) < 50 or len(np.unique(y[sub])) < 2:
                continue
            value = roc_auc_score(y[sub], pred[sub])
            collected[name].append(value)
            per_class[name] = value
        if "pick" in per_class and "teamfight" in per_class:
            gaps.append(per_class["pick"] - per_class["teamfight"])

    def summarize(values):
        arr = np.asarray(values)
        return {
            "mean": float(arr.mean()),
            "ci_2.5": float(np.percentile(arr, 2.5)),
            "ci_97.5": float(np.percentile(arr, 97.5)),
            "n_boot": int(len(arr)),
        }

    out = {name: summarize(vals) for name, vals in collected.items() if vals}
    if gaps:
        out["pick_minus_teamfight"] = summarize(gaps)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--allow-partial", action="store_true", help="evaluate even if the manifest check fails")
    parser.add_argument("--y-key", default="y",
                        help="label column in the shards: y (the build's LABEL_TYPE) or y_<type> from --extra-labels")
    parser.add_argument("--categorical", action="store_true",
                        help="treat identifier columns (champion / spell / rune / item-hash / ban ids, __last only) "
                             "as LightGBM categorical features instead of numbers")
    parser.add_argument("--teamfight-min", type=int, default=3,
                        help="smaller side's participation at which an engagement is a teamfight (3 = published, 4 = v3)")
    parser.add_argument("--matrix", type=Path, default=None,
                        help="path for the merged memmap (default: alongside --output)")
    args = parser.parse_args(argv)

    matrix_path = args.matrix or args.output.with_suffix(".matrix.npy")
    manifest = verify_shard_manifest(args.shards, allow_partial=args.allow_partial)
    data = merge_shards(args.shards, matrix_path, y_key=args.y_key)
    valid = data["y"] >= 0
    if not valid.all():
        idx = np.flatnonzero(valid)
        print(f"label {args.y_key}: dropping {int((~valid).sum())} rows without a label (draws)", flush=True)
        n_rows = int(len(data["y"]))
        for k in list(data):
            v = data[k]
            if hasattr(v, "shape") and len(v.shape) >= 1 and int(v.shape[0]) == n_rows:
                data[k] = np.asarray(v[idx])   # row-aligned arrays only; X is materialised from the memmap
    X, y, groups = data["X"], data["y"], data["groups"]
    classes = scale_class(data["cluster_blue"], data["cluster_red"], teamfight_min=args.teamfight_min)
    presence = scale_class(data["present_blue"], data["present_red"], teamfight_min=args.teamfight_min)
    print(f"matches={len(np.unique(groups))} positives={y.mean():.3f}")
    for name in CLASSES:
        share = float(np.mean(classes == name))
        print(f"  participation {name:10s} n={int((classes == name).sum()):7d} ({share*100:5.1f}%)")

    cat_idx = []
    if args.categorical and data.get("feature_names"):
        import re as _re
        role_re = _re.compile(r"^(b|r)(TOP|JNG|MID|BOT|SUP)_(.+)$")
        cat_bases = {"champion_id", "champion_name_id", "summoner_spell_1_id", "summoner_spell_2_id", "primary_style_id",
                     "sub_style_id", "primary_rune_1", "primary_rune_2", "primary_rune_3", "primary_rune_4", "sub_rune_1",
                     "sub_rune_2", "stat_perk_offense", "stat_perk_flex", "stat_perk_defense"} \
                    | {f"{tm}_ban_{i}" for tm in ("blue", "red") for i in range(5)}
        # item hashes (itemhash0..15) are bucket COUNTS of the inventory, not identifiers: never categorical
        for i, n in enumerate(data["feature_names"]):
            if not n.endswith("__last"):
                continue
            b = n[: -len("__last")]
            m = role_re.match(b)
            if (m and m.group(3) in cat_bases) or (b in cat_bases):
                cat_idx.append(i)
        print(f"categorical columns: {len(cat_idx)}", flush=True)
    pred = oof_predictions(X, y, groups, categorical=cat_idx or None)
    overall = float(roc_auc_score(y, pred))
    print(f"overall AUC: {overall:.4f}")

    masks = {name: classes == name for name in CLASSES if (classes == name).sum() >= 50}
    results: dict[str, object] = {
        "n": int(len(y)),
        "n_matches": int(len(np.unique(groups))),
        "n_features_used": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "overall_auc": overall,
        "patch_distribution": {
            str(p): int(c) for p, c in zip(*np.unique(data["patch"], return_counts=True))
        },
        "teamfight_min": int(args.teamfight_min),
        "y_key": str(args.y_key),
        "corpus_manifest": {k: manifest.get(k) for k in ("run_id", "git_commit", "num_shards", "complete", "feature_names_sha1")} if manifest else None,
        "categorical_columns": int(len(cat_idx)),
        "categorical_column_names": [data["feature_names"][i] for i in cat_idx] if cat_idx else [],
        "by_participation_scale": {},
        "by_presence_scale": {},
    }
    for name, mask in masks.items():
        results["by_participation_scale"][name] = {
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "positive_rate": float(y[mask].mean()),
            "auc": float(roc_auc_score(y[mask], pred[mask])),
            "asymmetry_mean": float(np.mean(np.abs(
                data["cluster_blue"][mask].astype(int) - data["cluster_red"][mask].astype(int)))),
        }
        print(f"participation {name:10s} n={mask.sum():7d}"
              f" auc={results['by_participation_scale'][name]['auc']:.4f}")
    for name in CLASSES:
        mask = presence == name
        if mask.sum() < 50:
            continue
        results["by_presence_scale"][name] = {
            "n": int(mask.sum()),
            "auc": float(roc_auc_score(y[mask], pred[mask])),
        }
    # robustness: teamfight at min>=4 instead of min>=3
    smaller = np.minimum(data["cluster_blue"], data["cluster_red"])
    strict = smaller >= 4
    if strict.sum() >= 50:
        results["teamfight_min4"] = {
            "n": int(strict.sum()),
            "auc": float(roc_auc_score(y[strict], pred[strict])),
        }
    results["bootstrap"] = cluster_bootstrap(y, pred, groups, masks, n_boot=args.n_boot)
    print(json.dumps(results["bootstrap"], indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output.with_suffix(".preds.npz"),
        y=y, pred=pred, groups=groups, scale=classes.astype(str),
        presence=presence.astype(str), patch=data["patch"],
        cluster_blue=data["cluster_blue"], cluster_red=data["cluster_red"],
    )
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
