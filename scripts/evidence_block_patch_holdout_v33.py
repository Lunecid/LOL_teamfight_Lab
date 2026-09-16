"""Evidence-corrected state block on the full v3.3 corpus, compared under the paper's patch holdout.

The v1 pilot (docs/experiments/claude_evidence_pilot_results.json, AUC 0.6649 -> 0.6708) ran on 16.6 % of a
50k exploratory split, with a position-fusion curve calibrated on its own evaluation matches.  This script
moves the comparison to the headline corpus and split, with the disjoint calibration.

  build    For every engagement row of corpus_shards_v33 (join key (groups, engage_ts)) query
           tau = engage_ts - 1 ms through EvidenceStateBuilder (position curve: the calibration JSON of
           scripts/calibrate_position_error_v3.py, default gameplay.evidence_state.DEFAULT_POS_ERROR_CALIBRATION)
           and through the plain last-frame hold.  engage_ts is the cutoff the corpus X was built at
           (build_corpus_shard.py: engage_ts = t_start_ts; the 50k dataset's cutoff equals it, 1,107 / 1,107
           checked).  Columns and construction are scripts/build_evidence_block_v3.py state_rows(), unchanged
           (306 columns).  One block shard per corpus shard, row-aligned, resumable.
  compare  X-only, X + hold block (control), X + evidence block: fit on patch 15.14, early-stop on 15.15,
           test on 15.16, with LightGBM at the lgbm_paper settings of scripts/run_model_comparison_v33.py
           run_model() (n_estimators 400, learning_rate .05, num_leaves 31, subsample .9, colsample_bytree .9,
           random_state 7, n_jobs -1, early_stopping(100) on validation AUC).  X is assembled as merge_shards()
           in scripts/run_scale_decomposition.py does it (shards in name order, columns non-constant over all
           rows, --y-key with draws dropped), so X-only should reproduce lgbm_paper in
           model_comparison_v33_patch.json (checked and recorded).
           Paired CIs: match-clustered percentile bootstrap (Efron & Tibshirani 1993, An Introduction to the
           Bootstrap, ch. 13; resampling whole clusters, Field & Welsh 2007, JRSS-B 69(3)).  Each draw is one
           multiplicity vector over the test matches shared by every model, comparison and band (overall, the
           game-minute bands, and the participation / presence scale classes), so all differences are paired
           and resampled jointly; every model's own AUC gets a CI from the same draws.  AUC per draw is the
           Mann-Whitney statistic with the multiplicities as integer weights, which equals roc_auc_score on the
           duplicated rows; the resampling stream is that of run_model_comparison_v33.paired
           (default_rng(7).integers(M, size=M)), so its first 300 draws are the reference's draws.

Leak discipline: the calibration must record zero intersection with every evaluation set it was checked against
(load_pos_error_curve refuses it otherwise); build re-checks its match ids against the corpus matches of the
validation and test patches, compare re-checks them against the loaded rows, and both stop on any overlap.  The
legacy curve (calibrated on predict_test) is refused.  Nothing from 15.15 / 15.16 reaches training except the
15.15 early-stopping signal the reference itself uses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 7
SHARDS = Path("D:/LOL_Project/fusion_2615/corpus_shards_v33")
OUT_DIR = Path("D:/LOL_Project/fusion_2615/features/tog_revision/A4-evidence-state-leak")
CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))
VARIANTS = ("X_plus_hold", "X_plus_evidence", "X")          # X last: its column slice is the only copy
COMPARISONS = {"evidence_minus_X": ("X", "X_plus_evidence"), "hold_minus_X": ("X", "X_plus_hold"),
               "evidence_minus_hold": ("X_plus_hold", "X_plus_evidence")}
LGBM_PAPER = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
                  colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1)
CLASSES = ("pick", "skirmish", "teamfight")
DEVIATIONS = {
    "build": [
        "Any exception while building one engagement or one match is recorded (row ok = False, block NaN) "
        "instead of aborting the run; build_evidence_block_v3.py aborts on non-ValueError errors.",
        "Rows are keyed by (groups, engage_ts) of the corpus, not by the 50k dataset's engagement ids.",
    ],
    "compare": [
        "Block rows that failed to build are NaN in both the hold and the evidence block (LightGBM missing values); "
        "the count per split is recorded.",
        "n_jobs is a flag (default -1 as in the reference); LightGBM results can differ in the last digits across "
        "thread counts.",
        "Bootstrap uses --n-boot draws (default 1000) instead of the reference's 300; the first 300 coincide.",
        "Smoke runs (--shard-limit / --max-matches) compute the non-constant column mask on the loaded shards only.",
        "Additions beyond the reference, from the same joint draws: a CI for every model's own AUC, and paired CIs "
        "within each participation / presence scale class (the reference reports scale-class AUCs without CIs).",
        "Game-minute bands start at minute 2 (TIME_BANDS of the v1 pilot); test rows outside every band are counted "
        "in rows_outside_minute_bands and still enter 'overall'.",
    ],
}
LEAK_NOTES = [
    "Cutoff: the block is queried at tau = engage_ts - 1 ms.  Corpus X is built by gameplay/pipeline.py with "
    "label_start_ms = engage_ts, event buckets ending at label_start_ms - PREDICTION_GAP_MS (0 under v3.3) and "
    "causal anchors at label_start_ms - 1 ms, so the block reads nothing X could not.  Checked: 583 / 583 cutoffs of "
    "the 50k dataset (200 matches of shard_000) equal a corpus engage_ts.",
    "Position-error calibration: matches from the 50k predict_train split, patch not 15.15 / 15.16; recorded disjoint "
    "from predict_test, value_validation and every corpus match of 15.15 / 15.16 when fitted, re-checked by build "
    "against the corpus hold-out patches and here against the loaded validation and test rows (any overlap stops).",
    "Non-constant column mask over all rows of the loaded shards, validation and test patches included, exactly as "
    "merge_shards builds the lgbm_paper matrix.  It is label-free, and a column constant on the training patch offers "
    "no split, so the fitted trees cannot depend on it.",
    "Early stopping reads the validation patch, as the reference does; the test patch is read only to predict and to "
    "bootstrap.",
    "The calibration matches are training-side (the 50k predict_train split); the ones that fall in the training patch "
    "are allowed here (calibration_vs_loaded_rows.train_patch_*).  A whole-corpus match-grouped CV that used this "
    "block would put them in test folds: such a run must drop them from its test folds or refit the curve per fold.",
]
SOURCE_FILES = ["scripts/evidence_block_patch_holdout_v33.py", "gameplay/evidence_state.py",
                "scripts/build_evidence_block_v3.py", "scripts/calibrate_position_error_v3.py"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_keys(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def smoke_match_subset(groups, patch, n, patches, salt="smoke:7:"):
    """Smoke tests only: up to n matches split evenly over `patches`, ordered by sha256(salt + id)."""
    from scripts.calibrate_position_error_v3 import rank
    per = max(1, -(-n // max(1, len(patches))))
    chosen = []
    for p in patches:
        chosen += sorted(set(groups[patch == p].tolist()), key=lambda m: rank(salt + m))[:per]
    return set(chosen[:n])


def calibration_match_ids(provenance):
    src = provenance.get("source")
    if src in (None, "legacy", "dict"):
        raise SystemExit(f"position curve source {src!r} cannot be checked for disjointness; "
                         "use a calibrate_position_error_v3.py JSON")
    return set(json.loads(Path(src).read_text(encoding="utf-8"))["match_ids"])


def overlap_with_patches(cal_ids, groups, patch, patches):
    out = {}
    for p in patches:
        ms = set(groups[patch == p].tolist())
        out[f"corpus_patch_{p}"] = {"n_matches": len(ms), "intersection": len(cal_ids & ms)}
    return out


# ------------------------------------------------------------------ build

_W = {}


def _init_worker(curve):
    from data.cache_io import load_match_cache
    from gameplay.evidence_state import EvidenceStateBuilder, load_tables
    _W.update(load=load_match_cache, builder=EvidenceStateBuilder, load_tables=load_tables, curve=curve, tables={})


def _build_match(task):
    from scripts.build_evidence_block_v3 import block_names, slot_order, state_rows
    mid, cutoffs = task
    ev = np.full((len(cutoffs), len(block_names())), np.nan, dtype=np.float32)
    hold = np.full_like(ev, np.nan)
    ok = np.zeros(len(cutoffs), dtype=bool)
    try:
        pack = _W["load"](mid)
        if pack is None:
            return mid, cutoffs, ev, hold, ok, "cache missing"
        patch = str(pack["meta"]["patch"])
        if patch not in _W["tables"]:
            _W["tables"][patch] = _W["load_tables"](patch)
        b = _W["builder"](pack, patch=patch, tables=_W["tables"][patch], pos_error_curve=_W["curve"])
        order = slot_order(pack)
    except Exception as exc:                                        # unsupported pack: record, move on
        return mid, cutoffs, ev, hold, ok, repr(exc)
    errors = []
    for k, cutoff in enumerate(cutoffs.tolist()):
        try:
            e, h = state_rows(b, int(cutoff) - 1, order)
        except Exception as exc:
            errors.append(f"{int(cutoff)}: {exc!r}")
            continue
        ev[k], hold[k], ok[k] = e, h, True
    return mid, cutoffs, ev, hold, ok, ("; ".join(errors[:3]) if errors else None)


def build(a):
    import multiprocessing as mp
    from scripts.calibrate_position_error_v3 import git_state, ids_sha256, setup_env, source_hashes
    started, wall_start = time.time(), now()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    setup_env(a.out_dir / "manifest.json", a.cache_dir)
    from gameplay.evidence_state import load_pos_error_curve
    from run_scale_decomposition import verify_shard_manifest
    from scripts.build_evidence_block_v3 import block_names

    smoke = bool(a.max_matches or a.shard_ids)
    corpus_manifest = (verify_shard_manifest(a.shards) if not smoke
                       else json.loads((a.shards / "manifest.json").read_text(encoding="utf-8")))
    if isinstance(a.pos_error_curve, str) and a.pos_error_curve == "legacy":
        raise SystemExit("the legacy curve was calibrated on evaluation matches; not allowed here")
    curve, provenance = load_pos_error_curve(a.pos_error_curve)
    paths = sorted(a.shards.glob("shard_*.npz"))
    meta = {p.name: read_keys(p, ("groups", "engage_ts", "patch")) for p in paths}
    all_groups = np.concatenate([m["groups"].astype(str) for m in meta.values()])
    all_patch = np.concatenate([m["patch"].astype(str) for m in meta.values()])
    cal_ids = calibration_match_ids(provenance)
    checks = overlap_with_patches(cal_ids, all_groups, all_patch, a.holdout_patches.split(","))
    bad = {k: v["intersection"] for k, v in checks.items() if v["intersection"]}
    print(f"[build] calibration {len(cal_ids)} matches vs corpus hold-out patches: "
          f"{ {k: v['intersection'] for k, v in checks.items()} }", flush=True)
    if bad:
        raise SystemExit(f"calibration matches intersect corpus hold-out patches: {bad}")
    names = block_names()
    git = git_state()
    source_start = source_hashes(SOURCE_FILES)                     # the code that runs, not what is on disk at the end
    record = {"pos_error_curve": provenance, "calibration_vs_corpus": checks, "git": git, "preset": "v3.3",
              "corpus_run_id": corpus_manifest.get("run_id"), "corpus_git_commit": corpus_manifest.get("git_commit"),
              "tau": "engage_ts - 1 ms", "columns": "scripts/build_evidence_block_v3.py state_rows", "smoke": smoke}
    selected = [p for p in paths if not a.shard_ids or int(p.stem.split("_")[1]) in a.shard_ids]
    status = {}
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(curve,)) as pool:
        for p in selected:
            out = a.out_dir / p.name
            if out.exists() and not a.overwrite:
                side = out.with_suffix(".json")
                status[p.name] = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {"resumed": True}
                print(f"[build] {p.name} exists, skipped", flush=True)
                continue
            t0 = time.time()
            g = meta[p.name]["groups"].astype(str)
            ts = meta[p.name]["engage_ts"].astype(np.int64)
            pa = meta[p.name]["patch"].astype(str)
            rows_of = {}
            for r, (m, t) in enumerate(zip(g.tolist(), ts.tolist())):
                rows_of.setdefault(m, {}).setdefault(t, []).append(r)
            mids = list(rows_of)
            if a.max_matches:
                keep = smoke_match_subset(g, pa, a.max_matches, a.smoke_patches.split(","))
                mids = [m for m in mids if m in keep]
            tasks = [(m, np.array(sorted(rows_of[m]), dtype=np.int64)) for m in mids]
            ev = np.full((len(g), len(names)), np.nan, dtype=np.float32)
            hold = np.full_like(ev, np.nan)
            ok = np.zeros(len(g), dtype=bool)
            failures = {}
            for k, (mid, cutoffs, e, h, o, err) in enumerate(pool.imap_unordered(_build_match, tasks, chunksize=16), 1):
                for q, c in enumerate(cutoffs.tolist()):
                    for r in rows_of[mid][c]:
                        ev[r], hold[r], ok[r] = e[q], h[q], o[q]
                if err:
                    failures[mid] = err
                if k % 2000 == 0:
                    print(f"[build] {p.name} {k}/{len(tasks)} matches ({time.time()-t0:.0f}s)", flush=True)
            built = np.zeros(len(g), dtype=bool)
            for m in mids:
                for rr in rows_of[m].values():
                    built[rr] = True
            partial = out.with_name(out.stem + ".partial.npz")
            np.savez_compressed(partial, groups=g, engage_ts=ts, patch=pa, evidence=ev, hold=hold, ok=ok,
                                names=np.array(names), provenance=np.array(json.dumps(record)))
            os.replace(partial, out)
            status[p.name] = {"rows": int(len(g)), "rows_attempted": int(built.sum()), "rows_ok": int(ok.sum()),
                              "matches": len(mids), "failed_matches": len(failures),
                              "failures_sample": dict(list(failures.items())[:20]), "seconds": round(time.time() - t0, 1)}
            out.with_suffix(".json").write_text(json.dumps(status[p.name], indent=2), encoding="utf-8")
            print(f"[build] {p.name}: {status[p.name]['rows_ok']}/{status[p.name]['rows_attempted']} rows ok, "
                  f"{len(failures)} failed matches ({time.time()-t0:.0f}s, total {time.time()-started:.0f}s)", flush=True)
    complete = (not smoke) and all((a.out_dir / p.name).exists() for p in paths)
    manifest = {"item": "A4-evidence-state-leak", "stage": "build", "complete": complete, "smoke": smoke,
                "num_shards": len(paths), "shards": status, **record,
                "rows": int(sum(s.get("rows", 0) for s in status.values())),
                "rows_ok": int(sum(s.get("rows_ok", 0) for s in status.values())),
                "calibration_match_ids_sha256": ids_sha256(cal_ids), "workers": a.workers,
                "source_sha256": source_start, "source_changed_during_run": source_hashes(SOURCE_FILES) != source_start,
                "deviations": DEVIATIONS["build"],
                "wall_clock": {"start": wall_start, "end": now()}, "elapsed_seconds": round(time.time() - started, 1)}
    name = "manifest.json" if not smoke else "smoke_manifest.json"
    (a.out_dir / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[build] done: {manifest['rows_ok']}/{manifest['rows']} rows ok, complete={complete} "
          f"({time.time()-started:.0f}s)", flush=True)


# ------------------------------------------------------------------ compare

class RankedAUC:
    """Tie groups of one score vector, so a weighted AUC costs two bincounts per draw.

    AUC = sum_g P_g (N_below_g + N_g / 2) / (P N): the Mann-Whitney statistic with ties counted one half,
    which is what sklearn.metrics.roc_auc_score computes (with or without sample weights).
    """

    def __init__(self, y, p):
        order = np.argsort(p, kind="mergesort")
        ps = p[order]
        gid_sorted = np.cumsum(np.r_[True, ps[1:] != ps[:-1]]) - 1
        gid = np.empty(len(p), dtype=np.int64)
        gid[order] = gid_sorted
        self.n_groups = int(gid_sorted[-1]) + 1 if len(p) else 0
        yb = np.asarray(y).astype(bool)
        self.pos_idx, self.neg_idx = np.flatnonzero(yb), np.flatnonzero(~yb)
        self.gid_pos, self.gid_neg = gid[self.pos_idx], gid[self.neg_idx]

    def auc(self, w):
        pos = np.bincount(self.gid_pos, weights=w[self.pos_idx], minlength=self.n_groups)
        neg = np.bincount(self.gid_neg, weights=w[self.neg_idx], minlength=self.n_groups)
        P, N = pos.sum(), neg.sum()
        if P <= 0 or N <= 0:
            return np.nan
        return float((pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (P * N))


def bootstrap_bands(minute, extra_masks=None):
    """{band: bool mask}: 'overall', the TIME_BANDS game-minute bands, then extra_masks in their given order."""
    minute = np.asarray(minute, dtype=float)
    bands = {"overall": np.ones(len(minute), dtype=bool)}
    bands.update({f"{lo}-{hi}": (minute >= lo) & (minute < hi) for lo, hi in TIME_BANDS})
    for name, mask in (extra_masks or {}).items():
        if name in bands:
            raise ValueError(f"band name {name!r} is taken")
        bands[name] = np.asarray(mask, dtype=bool)
    return bands


def joint_paired_bootstrap(y, preds, groups, minute, n_boot=1000, seed=SEED, extra_masks=None, model_cis=False):
    """Percentile CIs for every comparison x band from one shared match resample per draw.

    Bands: bootstrap_bands(minute, extra_masks).  A band with fewer than 50 rows or a single class is skipped.
    Draw b is counts_b = bincount(default_rng(seed).integers(M, size=M)) over the M test matches; every row carries
    its match's multiplicity, and each band's AUCs are computed on that band's rows with those weights, so every
    model, comparison and band inside one draw sees the same resampled matches (paired, resampled jointly).
    Returns (comparisons, meta), or (comparisons, meta, model_cis) with model_cis=True: a percentile CI for each
    model's own AUC per band, from the same draws.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    n_clusters = len(uniq)
    comps = {c: ab for c, ab in COMPARISONS.items() if ab[0] in preds and ab[1] in preds}
    bands = {}
    for bname, mask in bootstrap_bands(minute, extra_masks).items():
        idx = np.flatnonzero(mask)
        if len(idx) < 50 or len(np.unique(y[idx])) < 2:
            continue
        bands[bname] = (idx, inv[idx], {k: RankedAUC(y[idx], np.asarray(p)[idx]) for k, p in preds.items()})
    model_draws = {k: {b: [] for b in bands} for k in preds}
    comp_draws = {c: {b: [] for b in bands} for c in comps}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(n_clusters, size=n_clusters), minlength=n_clusters).astype(np.float64)
        for bname, (idx, inv_b, ranked) in bands.items():
            w = counts[inv_b]
            aucs = {k: r.auc(w) for k, r in ranked.items()}
            for k, v in aucs.items():
                model_draws[k][bname].append(v)
            for c, (ka, kb) in comps.items():
                comp_draws[c][bname].append(aucs[kb] - aucs[ka])

    def ci(values):
        d = np.asarray(values, dtype=float)
        d = d[np.isfinite(d)]
        return d, ({"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                    "n_boot": int(len(d))} if len(d) else None)

    point = {b: {k: float(roc_auc_score(y[idx], np.asarray(p)[idx])) for k, p in preds.items()}
             for b, (idx, _, _) in bands.items()}
    size = {b: {"n_rows": int(len(idx)), "n_matches": int(len(np.unique(inv_b)))} for b, (idx, inv_b, _) in bands.items()}
    out = {}
    for c, (ka, kb) in comps.items():
        out[c] = {}
        for bname in bands:
            d, stats = ci(comp_draws[c][bname])
            if stats is None:
                continue
            out[c][bname] = {"delta": point[bname][kb] - point[bname][ka], **stats,
                             "p_draws_le_0": float((d <= 0).mean()), **size[bname]}
    models = {}
    for k in preds:
        models[k] = {}
        for bname in bands:
            _, stats = ci(model_draws[k][bname])
            if stats is not None:
                models[k][bname] = {"auc": point[bname][k], **stats, **size[bname]}
    check_w = np.bincount(np.random.default_rng(seed + 1).integers(n_clusters, size=n_clusters), minlength=n_clusters)[inv].astype(np.float64)
    check = 0.0
    for k, p in preds.items():
        full = RankedAUC(y, np.asarray(p))
        check = max(check, abs(full.auc(np.ones(len(y))) - roc_auc_score(y, p)),
                    abs(full.auc(check_w) - roc_auc_score(y, p, sample_weight=check_w)))
    meta = {"n_boot": n_boot, "seed": seed, "clusters": int(n_clusters), "ci": "percentile 2.5 / 97.5",
            "bands": list(bands), "auc_implementation_max_abs_diff_vs_sklearn": float(check)}
    return (out, meta, models) if model_cis else (out, meta)


def reference_params_verified():
    src = (ROOT / "scripts/run_model_comparison_v33.py")
    if not src.exists():
        return False
    text = re.sub(r"\s+", " ", src.read_text(encoding="utf-8"))
    return ("params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9, "
            "colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1)") in text and "early_stopping(100" in text


def predict_chunked(model, A, cols=None, chunk=50_000):
    out = np.empty(len(A), dtype=np.float64)
    for s in range(0, len(A), chunk):
        part = A[s:s + chunk] if cols is None else np.ascontiguousarray(A[s:s + chunk, :cols])
        out[s:s + len(part)] = model.predict_proba(part)[:, 1]
    return out


def compare(a):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import roc_auc_score
    from run_scale_decomposition import scale_class, verify_shard_manifest
    from scripts.calibrate_position_error_v3 import curve_disjointness_record, git_state, ids_sha256, source_hashes
    started, wall_start = time.time(), now()
    log = lambda *s: print(*s, flush=True)
    if a.out.exists() and not a.overwrite:
        raise SystemExit(f"{a.out} exists; pass --overwrite or choose a new name")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    git_start, source_start = git_state(), source_hashes(SOURCE_FILES)   # the code that runs
    smoke = bool(a.max_matches or a.shard_limit)
    corpus_manifest = (verify_shard_manifest(a.shards) if not smoke
                       else json.loads((a.shards / "manifest.json").read_text(encoding="utf-8")))
    block_manifest_path = a.block_dir / ("manifest.json" if not smoke else "smoke_manifest.json")
    block_manifest = (json.loads(block_manifest_path.read_text(encoding="utf-8")) if block_manifest_path.exists() else None)
    if not smoke and not (block_manifest or {}).get("complete"):
        raise SystemExit(f"{block_manifest_path} is missing or not complete; run the build stage to completion first")
    paths = sorted(a.shards.glob("shard_*.npz"))[:a.shard_limit or None]
    split_patch = {"train": a.train_patch, "val": a.val_patch, "test": a.test_patch}
    splits = tuple(split_patch)

    # pass 0: labels, keys and row selection (no X)
    keys = (a.y_key, "groups", "patch", "engage_ts", "cluster_blue", "cluster_red", "present_blue", "present_red")
    meta = [read_keys(p, keys) for p in paths]
    subset = None
    if a.max_matches:
        subset = smoke_match_subset(np.concatenate([m["groups"].astype(str) for m in meta]),
                                    np.concatenate([m["patch"].astype(str) for m in meta]),
                                    a.max_matches, list(split_patch.values()))
    sel = []
    for m in meta:
        base = (m[a.y_key] >= 0)
        if subset is not None:
            base &= np.isin(m["groups"].astype(str), sorted(subset))
        sel.append({s: np.flatnonzero(base & (m["patch"].astype(str) == split_patch[s])) for s in splits})
    n = {s: int(sum(len(x[s]) for x in sel)) for s in splits}
    if not all(n.values()):
        raise SystemExit(f"empty patch split: {n}")
    cat = lambda key, s: np.concatenate([m[key][x[s]] for m, x in zip(meta, sel)])
    y = {s: cat(a.y_key, s).astype(np.int8) for s in splits}
    groups = {s: cat("groups", s).astype(str) for s in splits}
    engage_ts = {s: cat("engage_ts", s).astype(np.int64) for s in splits}
    for s1 in splits:
        for s2 in splits:
            if s1 < s2 and set(groups[s1].tolist()) & set(groups[s2].tolist()):
                raise SystemExit(f"match overlap between {s1} and {s2}")
    log(f"rows {n} matches { {s: len(set(groups[s].tolist())) for s in splits} }")
    # rows whose block failed to build, counted before any X is loaded so an incomplete block fails fast
    block_ok = [read_keys(a.block_dir / p.name, ("ok",))["ok"] for p in paths]
    missing = {s: int(sum(int((~ok[x[s]]).sum()) for ok, x in zip(block_ok, sel))) for s in splits}
    missing_share = {s: missing[s] / n[s] for s in splits}
    log(f"block rows not built (NaN): {missing}")
    if not smoke and max(missing_share.values()) > a.max_missing_block_share:
        raise SystemExit(f"block rows not built exceed --max-missing-block-share {a.max_missing_block_share}: {missing_share}")
    del block_ok

    # block provenance: one curve behind every loaded block shard, its calibration disjoint from the val / test rows
    block_provs = {p.name: json.loads(str(read_keys(a.block_dir / p.name, ("provenance",))["provenance"])) for p in paths}
    block_prov = block_provs[paths[0].name]
    bnames = read_keys(a.block_dir / paths[0].name, ("names",))["names"].astype(str).tolist()
    try:
        cal_check = curve_disjointness_record(
            {k: v.get("pos_error_curve") for k, v in block_provs.items()},
            {f"val_patch_{a.val_patch}": groups["val"], f"test_patch_{a.test_patch}": groups["test"]})
    except ValueError as exc:
        raise SystemExit(f"position-error calibration refused: {exc}")
    cal_ids = calibration_match_ids(block_prov["pos_error_curve"])
    calibration_doc = json.loads(Path(block_prov["pos_error_curve"]["source"]).read_text(encoding="utf-8"))
    cal_check["train_patch_" + a.train_patch] = {
        "n_matches": len(set(groups["train"].tolist())), "intersection": len(cal_ids & set(groups["train"].tolist())),
        "note": "allowed: the calibration is feature construction fitted on training-side matches"}
    cal_check["disjoint_from_test_patch"] = cal_check["disjointness_from_this_run"][f"test_patch_{a.test_patch}"]["intersection"] == 0
    cal_check["disjoint_from_val_patch"] = cal_check["disjointness_from_this_run"][f"val_patch_{a.val_patch}"]["intersection"] == 0
    log(f"calibration vs loaded rows: { {k: v['intersection'] for k, v in cal_check['disjointness_from_this_run'].items()} }")

    # pass 1: non-constant columns over all rows of the loaded shards (merge_shards rule)
    col_min = col_max = None
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            X = z["X"]
        if col_min is None:
            col_min, col_max = X.min(axis=0), X.max(axis=0)
        else:
            np.minimum(col_min, X.min(axis=0), out=col_min)
            np.maximum(col_max, X.max(axis=0), out=col_max)
        del X
    keep = np.flatnonzero(col_max > col_min)
    d, width = len(keep), len(keep) + len(bnames)
    all_names = json.loads((a.shards / "feature_names.json").read_text(encoding="utf-8"))["names"]
    x_names = [all_names[j] for j in keep]
    log(f"columns: {d} non-constant X + {len(bnames)} block ({time.time()-started:.0f}s)")

    # pass 2: fill X and the two blocks, per split, in shard order
    M = {s: np.empty((n[s], width), dtype=np.float32) for s in splits}
    EV = {s: np.empty((n[s], len(bnames)), dtype=np.float32) for s in splits}
    HD = {s: np.empty((n[s], len(bnames)), dtype=np.float32) for s in splits}
    OK = {s: np.empty(n[s], dtype=bool) for s in splits}
    off = {s: 0 for s in splits}
    for p, m, x in zip(paths, meta, sel):
        with np.load(p, allow_pickle=False) as z:
            X = z["X"]
        blk = read_keys(a.block_dir / p.name, ("groups", "engage_ts", "evidence", "hold", "ok"))
        if not (np.array_equal(blk["groups"].astype(str), m["groups"].astype(str))
                and np.array_equal(blk["engage_ts"], m["engage_ts"])):
            raise SystemExit(f"block shard {p.name} is not row-aligned with the corpus shard")
        for s in splits:
            r, o = x[s], off[s]
            M[s][o:o + len(r), :d] = X[r][:, keep]
            EV[s][o:o + len(r)], HD[s][o:o + len(r)], OK[s][o:o + len(r)] = blk["evidence"][r], blk["hold"][r], blk["ok"][r]
            off[s] += len(r)
        del X, blk
        log(f"  loaded {p.name} ({time.time()-started:.0f}s)")
    if {s: int((~OK[s]).sum()) for s in splits} != missing:
        raise SystemExit("block ok masks changed between the first and the second pass")

    te_minute = (engage_ts["test"] - 1) / 60000.0
    ym = {s: y[s] for s in splits}
    cm = lambda key: np.concatenate([m[key][x["test"]] for m, x in zip(meta, sel)])
    classes = scale_class(cm("cluster_blue"), cm("cluster_red"), teamfight_min=4)
    present = scale_class(cm("present_blue"), cm("present_red"), teamfight_min=4)
    params = dict(LGBM_PAPER, n_jobs=a.n_jobs)
    results, preds = {}, {}
    for name in [v for v in VARIANTS if v in a.variants.split(",")]:
        t0 = time.time()
        log(f"\n=== {name} ===")
        if name == "X":
            Xtr, Xva, cols, fnames = np.ascontiguousarray(M["train"][:, :d]), np.ascontiguousarray(M["val"][:, :d]), d, x_names
        else:
            blk, prefix = (EV, "ev_") if name == "X_plus_evidence" else (HD, "hold_")
            for s in splits:
                M[s][:, d:] = blk[s]
            Xtr, Xva, cols, fnames = M["train"], M["val"], None, x_names + [prefix + b for b in bnames]
        model = LGBMClassifier(**params)
        model.fit(Xtr, ym["train"], eval_set=[(Xva, ym["val"])], eval_metric="auc",
                  callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        del Xtr, Xva
        val_auc = float(roc_auc_score(ym["val"], predict_chunked(model, M["val"], cols)))
        p = predict_chunked(model, M["test"], cols)
        preds[name] = p
        gain = model.booster_.feature_importance(importance_type="gain")
        top = np.argsort(gain)[::-1][:15]
        cell = {"overall_auc": float(roc_auc_score(ym["test"], p)), "val_auc": val_auc,
                "best_iteration": getattr(model, "best_iteration_", None), "n_features": int(model.n_features_in_),
                "seconds": round(time.time() - t0, 1), "by_game_minute": {},
                "top_features": [{"feature": fnames[i], "gain_share": float(gain[i] / max(gain.sum(), 1e-12))} for i in top]}
        for label, arr in (("by_participation_scale", classes), ("by_presence_scale", present)):
            for cname in CLASSES:
                mk = arr == cname
                if mk.sum() >= 50 and len(set(ym["test"][mk].tolist())) == 2:
                    cell.setdefault(label, {})[cname] = {"n": int(mk.sum()), "auc": float(roc_auc_score(ym["test"][mk], p[mk]))}
        for lo, hi in TIME_BANDS:
            mk = (te_minute >= lo) & (te_minute < hi)
            if mk.sum() >= 50 and len(set(ym["test"][mk].tolist())) == 2:
                cell["by_game_minute"][f"{lo}-{hi}"] = {"n": int(mk.sum()), "matches": int(len(set(groups["test"][mk].tolist()))),
                                                        "auc": float(roc_auc_score(ym["test"][mk], p[mk]))}
        results[name] = cell
        log(f"  {name}: TEST({a.test_patch}) AUC {cell['overall_auc']:.4f} | val {val_auc:.4f} "
            f"best_iter {cell['best_iteration']} ({cell['seconds']:.0f}s)")
        np.savez_compressed(a.preds, y=ym["test"], groups=groups["test"], engage_ts=engage_ts["test"],
                            minute=te_minute, classes=classes.astype(str), present=present.astype(str),
                            **{f"p_{k}": v for k, v in preds.items()})

    scale_masks = {**{f"participation_{c}": classes == c for c in CLASSES},
                   **{f"presence_{c}": present == c for c in CLASSES}}
    paired, boot, model_ci = joint_paired_bootstrap(ym["test"], preds, groups["test"], te_minute, n_boot=a.n_boot,
                                                    extra_masks=scale_masks, model_cis=True)
    in_bands = np.zeros(len(te_minute), dtype=bool)
    for lo, hi in TIME_BANDS:
        in_bands |= (te_minute >= lo) & (te_minute < hi)
    reference = None
    if a.reference_json.exists() and "X" in results:
        ref = json.loads(a.reference_json.read_text(encoding="utf-8")).get("models", {}).get("lgbm_paper", {})
        if "overall_auc" in ref:
            reference = {"path": str(a.reference_json), "lgbm_paper_test_auc": ref["overall_auc"],
                         "x_only_test_auc": results["X"]["overall_auc"],
                         "abs_diff": abs(ref["overall_auc"] - results["X"]["overall_auc"]),
                         "comparable": not smoke}
    payload = {
        "item": "A4-evidence-state-leak", "stage": "compare", "smoke": smoke,
        "question": "does evidence-corrected state at the cutoff improve outcome prediction under the paper's patch holdout?",
        "reference": {"learner": "scripts/run_model_comparison_v33.py run_model lgbm_paper",
                      "matrix": "scripts/run_scale_decomposition.py merge_shards",
                      "bootstrap": "match-clustered percentile bootstrap (Efron & Tibshirani 1993 ch. 13; Field & Welsh 2007)",
                      "block": "scripts/build_evidence_block_v3.py state_rows"},
        "deviations": DEVIATIONS["compare"], "leak_notes": LEAK_NOTES,
        "calibration_deviations": calibration_doc.get("deviations"),
        "block_deviations": (block_manifest or {}).get("deviations"),
        "git": git_start, "git_end": git_state(),
        "source_sha256": source_start, "source_changed_during_run": source_hashes(SOURCE_FILES) != source_start,
        "preset": "v3.3", "label_key": a.y_key, "label_policy": "draws (y < 0) dropped", "seed": SEED,
        "split": {"kind": "patch holdout", **split_patch, "rows": n,
                  "matches": {s: int(len(set(groups[s].tolist()))) for s in splits}},
        "n_rows": int(sum(n.values())), "n_matches": int(len(set().union(*[set(groups[s].tolist()) for s in splits]))),
        "n_features_X": d, "block_columns": len(bnames), "block_rows_not_built": missing,
        "block_rows_not_built_share": missing_share, "max_missing_block_share": a.max_missing_block_share,
        "corpus_manifest": {k: corpus_manifest.get(k) for k in ("run_id", "git_commit", "num_shards", "feature_names_sha1")},
        "shards_loaded": len(paths), "block_dir": str(a.block_dir), "block_provenance": block_prov,
        "block_manifest": ({k: block_manifest.get(k) for k in ("complete", "rows", "rows_ok", "git", "source_sha256",
                                                                "source_changed_during_run", "wall_clock")}
                           if block_manifest else None),
        "calibration_vs_loaded_rows": cal_check, "calibration_match_ids_sha256": ids_sha256(cal_ids),
        "lgbm_params": params, "reference_params_verified": reference_params_verified(),
        "models": results, "reproduces_lgbm_paper": reference,
        "paired": paired, "model_auc_ci": model_ci, "bootstrap": boot,
        "rows_outside_minute_bands": int((~in_bands).sum()),
        "wall_clock": {"start": wall_start, "end": now()}, "elapsed_seconds": round(time.time() - started, 1),
    }
    a.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("\n" + json.dumps({c: {b: [round(v["lo"], 4), round(v["hi"], 4)] for b, v in cell.items()}
                           for c, cell in paired.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build", help="evidence + hold block for every corpus engagement row")
    b.add_argument("--shards", type=Path, default=SHARDS)
    b.add_argument("--cache-dir", type=Path, default=CACHE)
    b.add_argument("--pos-error-curve", default=None,
                   help="calibrate_position_error_v3.py JSON (default: gameplay.evidence_state default calibration)")
    b.add_argument("--out-dir", type=Path, default=OUT_DIR / "evidence_block_v33_v2")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--holdout-patches", default="15.15,15.16")
    b.add_argument("--shard-ids", type=int, nargs="*", default=[], help="smoke tests only")
    b.add_argument("--max-matches", type=int, default=0, help="smoke tests only: matches per shard")
    b.add_argument("--smoke-patches", default="15.14,15.15,15.16")
    b.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("compare", help="X / X + hold / X + evidence under the patch holdout")
    c.add_argument("--shards", type=Path, default=SHARDS)
    c.add_argument("--block-dir", type=Path, default=OUT_DIR / "evidence_block_v33_v2")
    c.add_argument("--out", type=Path, default=OUT_DIR / "evidence_block_patch_holdout_v33_v2.json")
    c.add_argument("--preds", type=Path, default=OUT_DIR / "evidence_block_patch_holdout_v33_v2.preds.npz")
    c.add_argument("--y-key", default="y_market_event")
    c.add_argument("--train-patch", default="15.14")
    c.add_argument("--val-patch", default="15.15")
    c.add_argument("--test-patch", default="15.16")
    c.add_argument("--variants", default="X_plus_hold,X_plus_evidence,X")
    c.add_argument("--n-jobs", type=int, default=-1)
    c.add_argument("--n-boot", type=int, default=1000)
    c.add_argument("--max-missing-block-share", type=float, default=0.01,
                   help="full runs stop if a split has more rows whose block failed to build (NaN) than this share")
    c.add_argument("--reference-json", type=Path, default=Path("D:/LOL_Project/fusion_2615/features/model_comparison_v33_patch.json"))
    c.add_argument("--shard-limit", type=int, default=0, help="smoke tests only")
    c.add_argument("--max-matches", type=int, default=0, help="smoke tests only")
    c.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.stage == "build":
        build(a)
    else:
        compare(a)


if __name__ == "__main__":
    main()
