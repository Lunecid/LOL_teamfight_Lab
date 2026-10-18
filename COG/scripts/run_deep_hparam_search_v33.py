"""Declared hyper-parameter search for the patch-holdout learner comparison (ToG A1; R2, R4).

CoG review R2 asked for "a more competitive and equitable experimental setting" for the deep
models, and R4 doubted their "true potential ... given the current architecture".  No deep model
was ever tuned.  This script gives FT-Transformer, SAINT (with and without self-supervised
pre-training), the MLP, TabNet and LightGBM each a declared grid under one protocol:

  1. Load the shards once; drop unlabelled rows; standardise with TRAIN-patch statistics only.
  2. SEARCH.  Every grid point is fitted on the train patch (15.14) -- or on a seeded MATCH
     subsample of it (--search-train-matches) -- with early stopping on the validation patch
     (15.15, optionally a seeded match subsample, --search-val-matches), and is scored on 15.15
     only.  Search fits are never given the test rows.
  3. SELECT the point with the highest validation AUC per family (ties: first in grid order).
  4. REFIT the selected point on the FULL train patch with early stopping on the FULL 15.15 and
     score test 15.16 exactly once.  Test predictions are saved; test AUCs carry match-clustered
     bootstrap CIs and paired deltas resample matches jointly (also against the published runs'
     predictions when their test rows are identical).

Resume: every finished grid point and refit is an atomic JSON under --out-dir, keyed by a hash of
its configuration and protocol.  Rerunning skips finished points (failed points are retried), and
a family's refit only runs once all its points have a record; an existing refit is never rescored.

Arguments this script does not define (--epochs, --patience, --amp, --grad-checkpoint,
--eval-batch-size, --lgbm-threads, --torch-threads, --max-gpu-mem-gb, --saint-arch, --ft-arch,
--contrastive-tau, --lambda-denoise, --cutmix-p, --mixup-alpha, --saint-pretrain-rows, --y-key,
patch names, ...) are passed through to run_deep_tabular_baselines.py's parser, whose defaults are
the published configuration; --epochs/--patience are the refit budget.

    LOL_OUTPUT_ROOT=D:/LOL_Project LOL_CFG_PRESET=v3.3 python scripts/run_deep_hparam_search_v33.py ^
        --shards D:/LOL_Project/fusion_2615/corpus_shards_v33 --n-matches 0 ^
        --out-dir D:/LOL_Project/fusion_2615/features/tog_revision/A1-deep-baselines/hparam_search
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_deep_tabular_baselines as dtb  # noqa: E402

FAMILIES = ("lightgbm", "mlp", "tabnet", "ft_transformer", "saint", "saint_pretrain")
MODEL_OF = {"saint_pretrain": "saint"}
_TRANSFORMER_GRID = {"lr": [1e-4, 3e-4, 1e-3], "n_layers": [2, 3], "d_token": [32, 128],
                     "lr_schedule": ["constant", "cosine_warmup"]}
GRIDS = {
    # the brief's suggested grids (docs/CLAUDE_TOG_PAPER_PLAN.md, item A1)
    "suggested": {
        "lightgbm": {"num_leaves": [31, 127, 255], "learning_rate": [0.03, 0.1],
                     "min_child_samples": [20, 100]},
        "mlp": {"hidden": [512, 1024], "layers": [2, 3, 4], "dropout": [0.1, 0.3],
                "lr": [1e-4, 1e-3]},
        "tabnet": {"n_d": [16, 32, 64], "n_steps": [3, 5], "lambda_sparse": [1e-4, 1e-3]},
        "ft_transformer": {**_TRANSFORMER_GRID, "d_token": [32, 128, 192]},
        "saint": dict(_TRANSFORMER_GRID),
        "saint_pretrain": dict(_TRANSFORMER_GRID),
    },
    # one tiny point per family, for smoke tests
    "smoke": {
        "lightgbm": {"num_leaves": [7], "learning_rate": [0.1], "min_child_samples": [20],
                     "n_estimators": [50]},
        "mlp": {"hidden": [32], "layers": [1], "dropout": [0.1], "lr": [1e-3]},
        "tabnet": {"n_d": [8], "n_steps": [2], "lambda_sparse": [1e-3]},
        "ft_transformer": {"d_token": [8], "n_layers": [1], "n_heads": [2], "lr": [1e-3],
                           "lr_schedule": ["cosine_warmup"]},
        "saint": {"d_token": [8], "n_layers": [1], "n_heads": [2], "d_misa": [8], "lr": [1e-3]},
        "saint_pretrain": {"d_token": [8], "n_layers": [1], "n_heads": [2], "d_misa": [8],
                           "lr": [1e-3]},
    },
}
# fixed on top of the published configuration for every point of a grid
FIXED = {
    "suggested": {"lightgbm": {"n_estimators": 5000}},
    "smoke": {},
}
PRETRAIN_KEYS = ("pretrain_rows", "pretrain_batch_size", "tau", "lambda_denoise", "p_cutmix",
                 "mixup_alpha", "denoise_reduction", "contrastive_sim")
TORCH_ONLY_OPTS = ("warmup_frac", "lr_decay_rate", "lr_decay_steps")
TOKEN_ONLY_OPTS = ("amp", "grad_checkpoint")


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def family_config(family: str, point: dict, configs: dict, grid_name: str,
                  pretrain_epochs: int) -> dict:
    model = MODEL_OF.get(family, family)
    cfg = dict(configs[model])
    if model == "saint":
        cfg["pretrain_epochs"] = pretrain_epochs if family == "saint_pretrain" else 0
    cfg.update(FIXED.get(grid_name, {}).get(family, {}))
    cfg.update(point)
    if model == "tabnet":
        cfg["n_a"] = cfg["n_d"]  # the paper ties N_a = N_d
    if model == "saint" and not cfg["pretrain_epochs"]:
        for key in PRETRAIN_KEYS:
            cfg.pop(key, None)
    return cfg


def relevant_options(model: str, opts: dict) -> dict:
    """Options that change a fit of ``model`` (device, eval batch and threads do not)."""
    keep = {"epochs", "patience", "seed"} if model != "lightgbm" else {"seed"}
    if model in dtb.TORCH_MODELS:
        keep |= set(TORCH_ONLY_OPTS)
    if model in ("ft_transformer", "saint"):
        keep |= set(TOKEN_ONLY_OPTS)
    return {k: v for k, v in opts.items() if k in keep}


def key_of(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def subsample_matches(rows: np.ndarray, groups: np.ndarray, n_matches: int, seed: int):
    """Rows of ``n_matches`` matches drawn without replacement from the matches in ``rows``."""
    matches = np.unique(groups[rows])
    if not n_matches or n_matches >= len(matches):
        return rows, {"kind": "full", "rows": int(len(rows)), "matches": int(len(matches))}
    chosen = np.random.default_rng(seed).choice(matches, size=n_matches, replace=False)
    picked = rows[np.isin(groups[rows], chosen)]
    return picked, {"kind": "match_subsample", "seed": int(seed), "rows": int(len(picked)),
                    "matches": int(n_matches), "of_matches": int(len(matches)),
                    "of_rows": int(len(rows))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-matches", type=int, default=0,
                        help="corpus match subsample before splitting (0: every match)")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--grid", choices=sorted(GRIDS), default="suggested")
    parser.add_argument("--phase", choices=("all", "search", "refit"), default="all")
    parser.add_argument("--search-train-matches", type=int, default=0,
                        help="search on a seeded match subsample of the train patch (0: full)")
    parser.add_argument("--search-val-matches", type=int, default=0,
                        help="early-stop and select on a seeded match subsample of the "
                             "validation patch (0: full)")
    parser.add_argument("--search-epochs", type=int, default=None,
                        help="max epochs per search point (default: --epochs)")
    parser.add_argument("--search-patience", type=int, default=None,
                        help="early-stopping patience per search point (default: --patience)")
    parser.add_argument("--pretrain-epochs", type=int, default=3,
                        help="SAINT pre-training epochs for the saint_pretrain family")
    parser.add_argument("--no-retry-failed", action="store_true",
                        help="treat failed points as final instead of retrying them")
    parser.add_argument("--allow-partial-grid", action="store_true",
                        help="refit even if some grid points have no record (recorded)")
    parser.add_argument("--points", default=None,
                        help="comma-separated grid indices to run per family (negative counts from "
                             "the end; smoke tests: -1 is each grid's largest point); other points "
                             "are neither run nor selected unless already on disk (recorded)")
    parser.add_argument("--published-preds", nargs="*", type=Path, default=[
        Path("D:/LOL_Project/fusion_2615/features/deep_tabular_v33_patch_full.preds.npz"),
        Path("D:/LOL_Project/fusion_2615/features/deep_tabular_v33_patch_ft.preds.npz"),
        Path("D:/LOL_Project/fusion_2615/features/deep_tabular_v33_patch_saint.preds.npz"),
    ], help="published prediction files to pair against when their test rows are identical")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true", help="print the declared plan and exit")
    return parser


def search_deviations(args, grids: dict, configs: dict) -> list[str]:
    archs = {"ft_transformer": ("gorishniy", "Gorishniy et al. defaults"),
             "saint": ("paper", "Somepalli et al. Eqs. (1)-(2)"),
             "tabnet": ("paper", "Arik & Pfister")}
    used = ", ".join(f"{m} '{configs[m]['arch']}'" for m in archs)
    off_reference = [m for m, (ref, _) in archs.items() if configs[m]["arch"] != ref]
    sizes = {f: len(expand_grid(g)) for f, g in grids.items()}
    out = [
        "grids are the declared ones below (brief's suggestion); MLP, TabNet and LightGBM grids "
        "have no learning-rate-schedule axis, FT-Transformer/SAINT grids do",
        "equal effort means one protocol, not one grid size: every family gets the same search "
        "rows, epoch cap, patience, selection rule and a single test evaluation, but the declared "
        "grids hold " + ", ".join(f"{f} {n}" for f, n in sizes.items()) + " points",
        "a 'published_saint' pairing, when present, is against the CoG SAINT row whose evaluation "
        "batches were in load order (intersample attention could reach later engagements of the "
        "same match); the refit SAINT uses match-disjoint batches, so that delta mixes tuning "
        "with the batching fix",
        "hyper-parameters outside each grid stay at the configuration the pass-through flags "
        f"define (architectures: {used})",
        *(f"{m} uses the '{configs[m]['arch']}' architecture, not the reference one "
          f"('{archs[m][0]}', {archs[m][1]})" for m in off_reference),
        "per-model deviations from each reference are listed in every refit record",
        "selection by validation AUC of the best epoch / iteration; ties go to the first point "
        "in grid order",
        f"saint_pretrain uses a fixed {args.pretrain_epochs} pre-training epoch(s); pre-training "
        "rows are the search-train rows in the search and the full train patch in the refit",
    ]
    if args.search_train_matches:
        out.append(f"search fits use a {args.search_train_matches}-match subsample of the train "
                   "patch; the refit uses the full train patch")
    if args.search_val_matches:
        out.append(f"search early stopping and selection use a {args.search_val_matches}-match "
                   "subsample of the validation patch; the refit early-stops on the full patch")
    if "lightgbm" in grids and args.grid == "suggested":
        out.append("LightGBM cap raised from 2,000 to 5,000 trees (early stopping 50 rounds) so "
                   "the lr 0.03 points are not truncated")
    if args.allow_partial_grid:
        out.append("--allow-partial-grid: a refit may have run before every grid point finished")
    if args.points is not None:
        out.append(f"--points {args.points}: only these grid indices were run in this invocation")
    return out


def parse_points(spec: str | None, n_points: int) -> set[int] | None:
    """Grid indices named by --points for a grid of ``n_points`` (None: every point)."""
    if spec is None:
        return None
    chosen = set()
    for item in spec.split(","):
        if item.strip():
            index = int(item)
            if not -n_points <= index < n_points:
                raise SystemExit(f"--points index {index} outside a grid of {n_points} points")
            chosen.add(index % n_points)
    return chosen


def main(argv=None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    base = dtb.build_parser().parse_args(
        ["--shards", str(args.shards), "--output", "unused.json",
         "--n-matches", str(args.n_matches), *rest])
    if base.split != "patch" or base.models != dtb.build_parser().get_default("models") \
            or base.skip_ft or base.pair_preds:
        raise SystemExit("--split, --models, --skip-ft and --pair-preds do not apply to the search "
                         "(patch split only; use --families and --published-preds)")
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise SystemExit(f"unknown families {unknown}; known: {FAMILIES}")
    grids = {f: GRIDS[args.grid][f] for f in families}
    points = {f: expand_grid(grids[f]) for f in families}
    only = {f: parse_points(args.points, len(points[f])) for f in families}
    if args.dry_run:
        for f in families:
            print(f"{f}: {len(points[f])} points over {grids[f]}"
                  + (f"; running {sorted(only[f])}" if only[f] is not None else ""))
        print(f"total points: {sum(len(p) for p in points.values())}")
        return 0

    started = time.time()
    dtb.configure_runtime(base.torch_threads, base.max_gpu_mem_gb)
    data = dtb.load_subsample(args.shards, args.n_matches or None, dtb.SEED, y_key=base.y_key,
                              drop_unlabelled=True)
    if data["n_unlabelled_dropped"]:
        print(f"label {base.y_key}: dropping {data['n_unlabelled_dropped']} rows without a label"
              " (draws)")
    names = json.loads((args.shards / "feature_names.json").read_text(encoding="utf-8"))["names"]
    n_bases, extras = dtb.token_layout(names)
    layout = (n_bases, len(extras))
    tr, va, te = dtb.split_by_patch(data["patch"], base.train_patch, base.val_patch,
                                    base.test_patch)
    dtb.prepare_matrix(data, tr)
    X, y, groups = data["X"], data["y"], data["groups"]
    tr_idx, va_idx, te_idx = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
    s_tr, s_tr_info = subsample_matches(tr_idx, groups, args.search_train_matches, dtb.SEED)
    s_va, s_va_info = subsample_matches(va_idx, groups, args.search_val_matches, dtb.SEED + 1)
    print(f"rows={len(y)} train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)};"
          f" search train={s_tr_info} search val={s_va_info}")

    fingerprint = {
        "shards": str(args.shards.resolve()), "n_matches_arg": args.n_matches,
        "n_rows": int(len(y)), "n_matches": int(len(np.unique(groups))),
        "n_features": int(X.shape[1]),
        "feature_names_sha1": hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest(),
        "y_key": base.y_key, "positive_rate": round(float(y.mean()), 6),
        "split": {"kind": "patch", "train_patch": base.train_patch, "val_patch": base.val_patch,
                  "test_patch": base.test_patch, "train": int(len(tr_idx)),
                  "val": int(len(va_idx)), "test": int(len(te_idx))},
        "standardisation": "train-patch mean/std (full train patch)",
    }
    configs = dtb.published_configs(base)
    refit_opts = dtb.training_options(base)
    search_opts = {**refit_opts,
                   "epochs": args.search_epochs or base.epochs,
                   "patience": args.search_patience or base.patience}
    meta = dtb.run_metadata(argv)
    out = args.out_dir
    dtb.write_json_atomic(out / "search_manifest.json", {
        **meta, "label_key": base.y_key, "data": fingerprint, "grid_name": args.grid,
        "grids": grids, "fixed": {f: FIXED[args.grid].get(f, {}) for f in families},
        "n_points": {f: len(points[f]) for f in families},
        "search_train": s_tr_info, "search_val": s_va_info,
        "search_options": search_opts, "refit_options": refit_opts,
        "pretrain_epochs_saint_pretrain": args.pretrain_epochs, "points_filter": args.points,
        "published_configs": configs, "deviations": search_deviations(args, grids, configs),
    })
    ran_points = skipped_points = 0
    declared_keys: dict[str, list[str]] = {}

    for family in families:
        model = MODEL_OF.get(family, family)
        records = []
        declared_keys[family] = []
        for index, point in enumerate(points[family]):
            cfg = family_config(family, point, configs, args.grid, args.pretrain_epochs)
            key_cfg = {k: v for k, v in cfg.items() if k != "n_jobs"}
            key = key_of({"phase": "search", "family": family, "config": key_cfg,
                          "data": fingerprint, "train": s_tr_info, "val": s_va_info,
                          "options": relevant_options(model, search_opts)})
            declared_keys[family].append(key)
            path = out / "points" / family / f"{key}.json"
            record = read_json(path)
            done = record is not None and (record.get("status") == "done" or (
                args.no_retry_failed and record.get("status") == "failed"))
            if done:
                skipped_points += 1
                records.append(record)
                continue
            if args.phase == "refit" or (only[family] is not None and index not in only[family]):
                records.append(record)
                continue
            print(f"[search:{family}] point {index + 1}/{len(points[family])} {point}", flush=True)
            point_started = time.time()
            record = {"family": family, "model": model, "index": index, "point": point,
                      "config": cfg, "key": key, "train": s_tr_info, "val": s_va_info,
                      "options": relevant_options(model, search_opts),
                      "git_commit": meta["git_commit"]}
            try:
                entry = dtb.fit_model(model, cfg, search_opts, X, y, s_tr, s_va, None, layout,
                                      label=f"{family}#{index}", groups=groups)
                entry.pop("_pred", None)
                entry.pop("test_auc", None)
                record.update(status="done", **entry)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # e.g. CUDA OOM on one point must not kill the grid
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:2000])
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            record["wall_clock_s"] = round(time.time() - point_started, 1)
            dtb.write_json_atomic(path, record)
            ran_points += 1
            records.append(record)
            print(f"  -> {record['status']} val_auc={record.get('val_auc')}"
                  f" ({record['wall_clock_s']}s)", flush=True)

        if args.phase == "search":
            continue
        missing = [i for i, r in enumerate(records) if r is None]
        finished = [r for r in records if r is not None and r.get("status") == "done"]
        if (missing and not args.allow_partial_grid) or not finished:
            print(f"[refit:{family}] skipped: {len(missing)} grid point(s) without a record,"
                  f" {len(finished)} finished")
            continue
        selected = max(finished, key=lambda r: (r["val_auc"], -r["index"]))
        failed = [r["index"] for r in records if r is not None and r.get("status") != "done"]
        if failed:
            print(f"[refit:{family}] WARNING: selecting among {len(finished)} finished point(s); "
                  f"failed grid indices {failed} are excluded (see their records)", flush=True)
        refit_path = out / "refit" / f"{family}.json"
        refit_key = key_of({"phase": "refit", "family": family, "config": {
            k: v for k, v in selected["config"].items() if k != "n_jobs"},
            "data": fingerprint, "options": relevant_options(model, refit_opts)})
        existing = read_json(refit_path)
        if existing is not None and existing.get("status") == "done":
            if existing.get("key") != refit_key:
                raise SystemExit(
                    f"{refit_path} scored test for a different selection/protocol; refusing to "
                    "score the test patch again.  Move that file away deliberately to rescore.")
            print(f"[refit:{family}] already scored; skipping")
            continue
        attempts = int((existing or {}).get("attempts", 0)) + 1
        dtb.write_json_atomic(refit_path, {"family": family, "key": refit_key, "status": "running",
                                           "attempts": attempts, "selected": selected["point"]})
        print(f"[refit:{family}] selected {selected['point']} (search val AUC"
              f" {selected['val_auc']:.4f}); full train patch, test scored once", flush=True)
        refit_started = time.time()
        entry = dtb.fit_model(model, selected["config"], refit_opts, X, y, tr_idx, va_idx, te_idx,
                              layout, label=f"{family}:refit", groups=groups)
        pred = entry.pop("_pred")
        boot = dtb.cluster_bootstrap_auc(y[te_idx], {family: pred}, groups[te_idx],
                                         n_boot=args.n_boot)
        dtb.write_model_preds(out / "refit" / f"{family}.preds.npz", y[te_idx], pred,
                              groups[te_idx])
        dtb.write_json_atomic(refit_path, {
            "family": family, "model": model, "key": refit_key, "status": "done",
            "attempts": attempts, "n_test_evaluations": 1,
            "n_points_declared": len(points[family]), "n_points_selected_from": len(finished),
            "failed_point_indices": failed, "missing_point_indices": missing,
            "selected_point": selected["point"], "selected_index": selected["index"],
            "selected_search_key": selected["key"], "search_val_auc": selected["val_auc"],
            "train": {"kind": "full train patch", "rows": int(len(tr_idx)),
                      "matches": int(len(np.unique(groups[tr_idx])))},
            "val": {"kind": "full val patch", "rows": int(len(va_idx)),
                    "matches": int(len(np.unique(groups[va_idx])))},
            "test": {"rows": int(len(te_idx)), "matches": int(len(np.unique(groups[te_idx])))},
            "options": relevant_options(model, refit_opts), "git_commit": meta["git_commit"],
            "git_dirty": meta["git_dirty"], "preset": meta["preset"], "seed": meta["seed"],
            "label_key": base.y_key, "split": fingerprint["split"],
            "test_auc_ci95": boot["auc"][family]["ci95"],
            "test_auc_bootstrap": {k: boot[k] for k in ("method", "n_boot", "seed", "n_matches",
                                                         "n_rows")},
            "deviations": dtb.model_deviations(model, selected["config"], refit_opts),
            **entry, "wall_clock_s": round(time.time() - refit_started, 1),
        })
        print(f"  -> test AUC={entry['test_auc']:.4f} CI={boot['auc'][family]['ci95']}", flush=True)

    summary = summarise(out, families, points, configs, args, meta, fingerprint, s_tr_info,
                        s_va_info, search_opts, refit_opts, y[te_idx], groups[te_idx], grids,
                        declared_keys)
    summary.update(points_run_this_invocation=ran_points,
                   points_skipped_this_invocation=skipped_points,
                   wall_clock_s=round(time.time() - started, 1))
    dtb.write_json_atomic(out / "summary.json", summary)
    print(f"wrote {out / 'summary.json'} (ran {ran_points} point(s), skipped {skipped_points})")
    return 0


def summarise(out: Path, families, points, configs, args, meta, fingerprint, s_tr_info, s_va_info,
              search_opts, refit_opts, y_te, g_te, grids, declared_keys) -> dict:
    """Aggregate this protocol's points (records under other keys -- an earlier grid, other search
    rows or options -- stay on disk but are not listed), selections and refits; pair the refits'
    test predictions."""
    per_family = {}
    preds = {}
    for family in families:
        model = MODEL_OF.get(family, family)
        rows = []
        for key in declared_keys[family]:
            r = read_json(out / "points" / family / f"{key}.json")
            if r is None:
                continue
            rows.append({k: r.get(k) for k in (
                "index", "point", "status", "val_auc", "best_epoch", "best_iteration",
                "epochs_run", "seconds", "wall_clock_s", "n_params", "peak_gpu_mem_allocated_gb",
                "error", "key")})
            if r.get("pretrain"):
                rows[-1]["pretrain_seconds"] = r["pretrain"].get("seconds")
        rows.sort(key=lambda r: (r["index"] if r["index"] is not None else 1e9))
        refit = read_json(out / "refit" / f"{family}.json")
        per_family[family] = {
            "model": model, "n_declared_points": len(points[family]), "grid": grids[family],
            "n_done": sum(r["status"] == "done" for r in rows),
            "n_failed": sum(r["status"] == "failed" for r in rows),
            "search_seconds": round(sum((r["wall_clock_s"] or 0) for r in rows), 1),
            "points": rows,
            "refit": None if refit is None else {k: refit.get(k) for k in (
                "status", "selected_point", "search_val_auc", "val_auc", "test_auc",
                "test_auc_ci95", "best_epoch", "best_iteration", "n_params", "seconds",
                "wall_clock_s", "attempts", "n_test_evaluations", "n_points_declared",
                "n_points_selected_from", "failed_point_indices", "missing_point_indices",
                "deviations", "config")},
        }
        pred_path = out / "refit" / f"{family}.preds.npz"
        if refit and refit.get("status") == "done" and pred_path.exists():
            with np.load(pred_path, allow_pickle=True) as blob:
                preds[family] = blob["pred"]
    refit_names = list(preds)
    published, published_notes = dtb.load_reference_preds(args.published_preds, y_te, g_te)
    preds.update(published)
    pairs = [(a, b) for i, a in enumerate(refit_names) for b in refit_names[i + 1:]]
    pairs += [(f, f"published_{MODEL_OF.get(f, f)}") for f in refit_names
              if f"published_{MODEL_OF.get(f, f)}" in preds]
    paired = None
    if preds and args.n_boot:
        paired = dtb.cluster_bootstrap_auc(y_te, preds, g_te, n_boot=args.n_boot, pairs=pairs)
    return {
        **meta, "label_key": fingerprint["y_key"], "data": fingerprint, "grid_name": args.grid,
        "search_train": s_tr_info, "search_val": s_va_info, "search_options": search_opts,
        "refit_options": refit_opts, "pretrain_epochs_saint_pretrain": args.pretrain_epochs,
        "protocol": ("select on validation patch only; refit on full train patch with early "
                     "stopping on the full validation patch; test patch scored once per family"),
        "families": per_family, "test_bootstrap": paired, "published_pairing": published_notes,
        "deviations": search_deviations(args, grids, configs),
    }


if __name__ == "__main__":
    raise SystemExit(main())
