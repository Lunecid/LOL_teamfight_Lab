"""Label family on corpus v3.3: is the model learning the game or the labelling heuristic?  (R2)

Reviewer R2 (CoG 2026 submission 118, major): the engagement outcome aggregates events with hand-set
coefficients, so a model might learn the labelling heuristic; the reviewer asked for ablations with
alternative labelling schemes (e.g. raw kill advantage) and a sensitivity analysis on the weighting
parameters.  For every variant of gameplay/labels.py LABEL_VARIANTS, computed on the corpus rows by
scripts/build_label_sidecars_v33.py, this script reports on two row sets

    own     the rows that variant labels (its own draws dropped)
    common  the rows every scored variant labels (identical rows and folds for all variants)

    OOF AUC        match-grouped GroupKFold OOF LightGBM: run_scale_decomposition.py's
                   oof_predictions itself (imported, not copied) -- same GroupKFold on the rows in
                   corpus order, same LGBMClassifier parameters, frozen OOF predictions
    CI             match-clustered bootstrap with run_scale_decomposition.cluster_bootstrap's draws
                   (default_rng(7), matches resampled with replacement); all statistics of one
                   replicate share the draw, so every difference is a paired comparison
    scale classes  per-class AUC of the frozen OOF predictions (scale_class, teamfight_min 4:
                   pick <= 1, skirmish 2-3, teamfight >= 4 on the smaller side's participation)
    agreement      label agreement, flip rate and Cohen's kappa vs market_event
    transfer       the market_event model's OOF predictions scored against the variant's labels
                   (same folds on the common rows; market_event's own-row folds on own rows)

Reference followed: scripts/run_scale_decomposition.py (oof_predictions, scale_class,
verify_shard_manifest, the cluster_bootstrap resampling scheme), the headline 5-fold protocol.
The feature matrix is never rebuilt: the merged memmap of corpus_shards_v33 (--matrix of
run_scale_decomposition.py; 566,452 x 6,164 float32 with the columns constant over all corpus rows
removed) is opened read only and verified against the shards (shape, and exact equality of sampled
rows of every shard on the non-constant columns) before any fit.

Leak discipline: this learner has no model selection, early stopping or calibration, so nothing of
an evaluation fold reaches training; labels are post-cutoff outcomes and never features.

    LOL_OUTPUT_ROOT=D:/LOL_Project LOKY_MAX_CPU_COUNT=8 .venv/Scripts/python.exe scripts/run_label_family_v33.py \
        --rowsets common,own --variants required --threads 8 \
        --sidecars D:/LOL_Project/fusion_2615/features/tog_revision/A2-label-family/label_sidecars_v33 \
        --keep-cache D:/LOL_Project/fusion_2615/features/tog_revision/A2-label-family/keep_v33.npz \
        --output D:/LOL_Project/fusion_2615/features/tog_revision/A2-label-family/label_family_v33.json

Writes the JSON (git commit, preset, label keys, split, n rows / matches, seed, wall clock,
deviations) and <output stem>.preds/<rowset>__<variant>.npz per fit (sidecar position, matrix row,
y, OOF prediction, fold, match, scale class, patch).  --resume reuses a preds file whose spec hash
(rows, labels, folds setting, learner source, matrix) matches.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 7                                   # run_scale_decomposition.SEED
REFERENCE = "market_event"
CLASSES = ("pick", "skirmish", "teamfight")
DEFAULT_CORPUS = Path("D:/LOL_Project/fusion_2615/corpus_shards_v33")
DEFAULT_MATRIX = Path("D:/LOL_Project/fusion_2615/features/scale_decomposition_v33_market_event.matrix.npy")
DEFAULT_REFERENCE_PREDS = Path("D:/LOL_Project/fusion_2615/features/scale_decomposition_v33_market_event.preds.npz")
V33_FEATURE_NAMES_SHA1 = "3c97f60383d303aa9d53e107ecb350cc0e1b4886"

DEVIATIONS = [
    "X rows are read from the existing merged memmap (opened read only) through a row view; run_scale_decomposition.py "
    "materialises the labelled rows in RAM first.  Values, row order and folds are the same: the matrix is verified "
    "against every shard, and GroupKFold sees the identical rows in corpus order.",
    "Bootstrap CIs use per-row match-count weights with a weighted Mann-Whitney AUC instead of concatenating resampled "
    "row indices.  The draws are cluster_bootstrap's (default_rng(7), integers over the row set's unique matches) and the "
    "statistic is identical (tests/test_label_variants.py).  Every statistic of a replicate shares the draw, so "
    "differences are paired.  Replicates with < 50 weighted rows or a missing class are skipped, as in cluster_bootstrap.",
    "On the own row sets the transfer model is market_event's own-row OOF model, scored only on rows both labels "
    "decide, so its folds are those of market_event's rows; only on the common row set are rows and folds identical "
    "across variants.  The own-row bootstrap resamples the matches of the variant's own rows.",
    "Constant columns are those constant over all 566,452 corpus rows (the merged matrix's column set), as in the "
    "published decompositions; they are not recomputed per row set.",
]


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------

def load_scale_decomposition():
    spec = importlib.util.spec_from_file_location("run_scale_decomposition", PROJECT_ROOT / "scripts" / "run_scale_decomposition.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha1_bytes(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(PROJECT_ROOT),
                                    capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except Exception:
        return "", None


class RowView:
    """Row subset of a read-only memmap that oof_predictions indexes like an array (X[idx] reads only those rows)."""

    def __init__(self, base, rows):
        self.base = base
        self.rows = np.asarray(rows, dtype=np.int64)
        self.shape = (int(len(self.rows)), int(base.shape[1]))
        self.dtype = base.dtype
        self.ndim = 2

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        return np.asarray(self.base[self.rows[idx]])


class SortedScores:
    """Predictions grouped into tie blocks once, so a weighted AUC is O(n) per bootstrap replicate."""

    def __init__(self, pred):
        pred = np.asarray(pred, dtype=np.float64)
        order = np.argsort(pred, kind="mergesort")
        sp = pred[order]
        blocks = np.concatenate([[0], np.cumsum(sp[1:] != sp[:-1])]) if len(sp) else np.zeros(0, np.int64)
        self.block = np.empty(len(pred), dtype=np.int64)
        self.block[order] = blocks
        self.n_blocks = int(blocks[-1]) + 1 if len(sp) else 0


def weighted_auc(scores: SortedScores, y, w) -> float:
    """Mann-Whitney AUC with row weights, ties counted 1/2.

    Equals roc_auc_score on the sample in which row i appears w[i] times; rows with y not in {0, 1}
    carry no weight.  NaN when either class has zero weight.
    """
    y = np.asarray(y)
    wp = np.bincount(scores.block, weights=w * (y == 1), minlength=scores.n_blocks)
    wn = np.bincount(scores.block, weights=w * (y == 0), minlength=scores.n_blocks)
    P, N = float(wp.sum()), float(wn.sum())
    if P <= 0.0 or N <= 0.0:
        return float("nan")
    below = np.cumsum(wn) - wn
    return float(np.dot(wp, below + 0.5 * wn) / (P * N))


def match_bootstrap_weights(groups, n_boot: int, seed: int = SEED):
    """Row weights of each match-clustered bootstrap replicate.

    The draws of run_scale_decomposition.cluster_bootstrap (default_rng(seed); for every replicate,
    len(unique) integers over the unique matches), expressed as per-row match counts instead of
    concatenated row indices.
    """
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    for _ in range(int(n_boot)):
        sample = rng.integers(0, len(unique), size=len(unique))
        yield np.bincount(sample, minlength=len(unique))[inverse].astype(np.float64)


def _summarize(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "ci_2.5": float(np.percentile(arr, 2.5)),
            "ci_97.5": float(np.percentile(arr, 97.5)), "n_boot": int(len(arr))}


def paired_bootstrap(groups, stats: dict, diffs: dict, n_boot: int):
    """stats: name -> (SortedScores, y, mask | None); diffs: name -> (stat a, stat b), a - b per replicate."""
    vals = {k: [] for k in stats}
    dvals = {k: [] for k in diffs}
    for w in match_bootstrap_weights(groups, n_boot):
        cur = {}
        for k, (sc, y, mask) in stats.items():
            ww = w if mask is None else w * mask
            yy = np.asarray(y)
            if float(ww[(yy == 0) | (yy == 1)].sum()) < 50:
                continue
            a = weighted_auc(sc, yy, ww)
            if a == a:
                vals[k].append(a)
                cur[k] = a
        for k, (a, b) in diffs.items():
            if a in cur and b in cur:
                dvals[k].append(cur[a] - cur[b])
    return {k: _summarize(v) for k, v in vals.items()}, {k: _summarize(v) for k, v in dvals.items()}


def _auc(y, pred, mask=None):
    y = np.asarray(y)
    pred = np.asarray(pred)
    sel = (y >= 0) if mask is None else ((y >= 0) & mask)
    if sel.sum() < 50 or len(np.unique(y[sel])) < 2:
        return None
    return float(roc_auc_score(y[sel], pred[sel]))


def agreement(a, b) -> dict:
    """Label agreement of variant labels ``a`` with reference labels ``b`` (-1 = draw) on rows both decide."""
    a, b = np.asarray(a), np.asarray(b)
    la, lb = a >= 0, b >= 0
    both = la & lb
    n_both = int(both.sum())
    out = {"n_rows": int(len(a)), "labelled_share_variant": float(la.mean()) if len(a) else None,
           "labelled_share_market_event": float(lb.mean()) if len(a) else None, "n_both": n_both,
           "n_variant_only": int((la & ~lb).sum()), "n_market_event_only": int((lb & ~la).sum())}
    if n_both:
        agree = float((a[both] == b[both]).mean())
        pa, pb = float((a[both] == 1).mean()), float((b[both] == 1).mean())
        pe = pa * pb + (1 - pa) * (1 - pb)
        out.update(agreement=agree, flip_rate=1.0 - agree,
                   flips_to_blue=int(((a == 1) & (b == 0) & both).sum()), flips_to_red=int(((a == 0) & (b == 1) & both).sum()),
                   cohen_kappa=(agree - pe) / (1 - pe) if pe < 1 else None)
    return out


def fold_ids(y, groups, n_splits: int) -> np.ndarray:
    """The fold of every row under oof_predictions' GroupKFold (X is not used by GroupKFold)."""
    from sklearn.model_selection import GroupKFold
    f = np.full(len(y), -1, dtype=np.int8)
    for k, (_, test) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros((len(y), 1), np.int8), y, groups)):
        f[test] = k
    return f


# ---------------------------------------------------------------------------------------------------------
# corpus, matrix and sidecar checks
# ---------------------------------------------------------------------------------------------------------

def corpus_layout(corpus_dir: Path) -> dict:
    paths = sorted(Path(corpus_dir).glob("shard_*.npz"))
    groups, engage = [], []
    for p in paths:
        with np.load(p, allow_pickle=False) as blob:
            groups.append(blob["groups"])
            engage.append(blob["engage_ts"])
    counts = np.array([len(g) for g in groups], dtype=np.int64)
    return {"paths": paths, "groups": groups, "engage_ts": engage, "counts": counts,
            "offsets": np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64), "total": int(counts.sum())}


def keep_and_samples(layout: dict, cache_path: Path, run_id: str, rows_per_shard: int = 6) -> dict:
    """Non-constant columns over all corpus rows (merge_shards' pass 1) plus sampled raw rows of every shard.

    Cached to ``cache_path`` keyed by the corpus run id and the shard file sizes.
    """
    sizes = np.array([p.stat().st_size for p in layout["paths"]], dtype=np.int64)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as c:
            if str(c["run_id"]) == str(run_id) and np.array_equal(c["shard_bytes"], sizes):
                print(f"keep columns from cache {cache_path}", flush=True)
                return {"keep": c["keep"], "sample_rows": c["sample_rows"], "sample_X": c["sample_X"], "cached": True}
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    col_min = col_max = None
    rows_all, X_all = [], []
    for p, off in zip(layout["paths"], layout["offsets"]):
        with np.load(p, allow_pickle=False) as blob:
            X = blob["X"]
        if col_min is None:
            col_min, col_max = X.min(axis=0), X.max(axis=0)
        else:
            np.minimum(col_min, X.min(axis=0), out=col_min)
            np.maximum(col_max, X.max(axis=0), out=col_max)
        rows = np.unique(np.concatenate([[0, len(X) - 1], rng.integers(0, len(X), rows_per_shard - 2)]))
        rows_all.append(int(off) + rows)
        X_all.append(X[rows].copy())
        del X
    keep = np.flatnonzero(col_max > col_min)
    out = {"keep": keep, "sample_rows": np.concatenate(rows_all).astype(np.int64), "sample_X": np.concatenate(X_all), "cached": False}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.stem + ".partial.npz")
    np.savez_compressed(tmp, keep=keep, sample_rows=out["sample_rows"], sample_X=out["sample_X"],
                        run_id=np.array(str(run_id)), shard_bytes=sizes)
    os.replace(tmp, cache_path)
    print(f"keep columns: {len(keep)} of {len(col_min)} in {time.time() - t0:.0f}s -> {cache_path}", flush=True)
    return out


def verify_matrix(M, layout: dict, ks: dict) -> dict:
    keep, rows, sx = ks["keep"], ks["sample_rows"], ks["sample_X"]
    problems = []
    if tuple(M.shape) != (layout["total"], len(keep)):
        problems.append(f"shape {tuple(M.shape)} != ({layout['total']}, {len(keep)})")
    if M.dtype != np.float32:
        problems.append(f"dtype {M.dtype}")
    bad = -1
    if not problems:
        got = np.asarray(M[rows])
        bad = int((~np.all(got == sx[:, keep], axis=1)).sum())
        if bad:
            problems.append(f"{bad} of {len(rows)} sampled rows differ from the shards")
    if problems:
        raise SystemExit("matrix does not match corpus_shards_v33: " + "; ".join(problems))
    return {"shape": list(M.shape), "n_sampled_rows_checked": int(len(rows)), "n_shards": int(len(layout["paths"])),
            "keep_sha1": _sha1_bytes(keep), "n_keep": int(len(keep))}


def load_sidecars(sidecar_dir: Path, layout: dict, corpus_run_id: str, allow_partial: bool) -> dict:
    mp = Path(sidecar_dir) / "manifest.json"
    if not mp.exists():
        raise SystemExit(f"{sidecar_dir} has no manifest.json")
    man = json.load(open(mp, encoding="utf-8"))
    if (man.get("corpus") or {}).get("run_id") != corpus_run_id:
        raise SystemExit(f"sidecars were built against corpus run {(man.get('corpus') or {}).get('run_id')}, not {corpus_run_id}")
    if not man.get("full") and not allow_partial:
        raise SystemExit("sidecar manifest is not a full build (every shard, no match limit); pass --allow-partial-sidecars for smoke tests")
    files = sorted(p for p in Path(sidecar_dir).glob("labels_shard_*.npz") if ".partial" not in p.name)
    if not files:
        raise SystemExit(f"no labels_shard_*.npz under {sidecar_dir}")
    parts: dict[str, list] = {}
    shard_ids = []
    for f in files:
        s = int(f.stem.split("_")[-1])
        info_path = f.with_suffix(".json")
        info = json.load(open(info_path, encoding="utf-8")) if info_path.exists() else {}
        if not (info.get("identity") or {}).get("ok"):
            raise SystemExit(f"{f.name}: identity check not recorded as passed")
        with np.load(f, allow_pickle=False) as blob:
            arrs = {k: blob[k] for k in blob.files}
        n = len(arrs["groups"])
        cr = arrs["corpus_row"].astype(np.int64)
        if not np.array_equal(cr, np.arange(n)):
            raise SystemExit(f"{f.name}: corpus_row is not the corpus prefix")
        if n > layout["counts"][s] or not np.array_equal(arrs["groups"].astype(str), layout["groups"][s][:n].astype(str)) \
                or not np.array_equal(arrs["engage_ts"], layout["engage_ts"][s][:n]):
            raise SystemExit(f"{f.name}: rows differ from corpus shard {s}")
        arrs["matrix_row"] = layout["offsets"][s] + cr
        arrs["shard"] = np.full(n, s, dtype=np.int16)
        for k, v in arrs.items():
            parts.setdefault(k, []).append(v)
        shard_ids.append(s)
    if not allow_partial and shard_ids != list(range(len(layout["paths"]))):
        raise SystemExit(f"sidecar shards {shard_ids} do not cover the corpus")
    side = {k: np.concatenate(v) for k, v in parts.items()}
    side["manifest"] = man
    side["shard_ids"] = shard_ids
    return side


# ---------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidecars", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX,
                    help="existing merged memmap of corpus_shards_v33 (opened read only, verified, never written)")
    ap.add_argument("--keep-cache", type=Path, default=None, help="non-constant columns + sampled rows (default: next to --output)")
    ap.add_argument("--variants", default="required", help="'required' (market_event + the 11 R2 variants), 'all', or a comma list")
    ap.add_argument("--rowsets", default="common,own")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--teamfight-min", type=int, default=4)
    ap.add_argument("--reference-preds", type=Path, default=DEFAULT_REFERENCE_PREDS,
                    help="published market_event OOF predictions to compare the own-row market_event fit with")
    ap.add_argument("--threads", type=int, default=None, help="set LOKY_MAX_CPU_COUNT / OMP_NUM_THREADS (LightGBM n_jobs=-1)")
    ap.add_argument("--resume", action="store_true", help="reuse preds files whose spec hash matches")
    ap.add_argument("--allow-partial-sidecars", action="store_true", help="smoke tests on a subset of shards / matches")
    args = ap.parse_args(argv)
    if args.threads:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    t_run = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rowsets = [s.strip() for s in args.rowsets.split(",") if s.strip()]
    if not set(rowsets) <= {"own", "common"}:
        raise SystemExit(f"--rowsets must be own and/or common, got {rowsets}")

    # lightgbm 4.6 + sklearn 1.8 warn on every predict_proba of an ndarray; the published decomposition runs the same
    # call, so the message carries no information here
    import warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    rsd = load_scale_decomposition()
    import lightgbm
    import sklearn
    from lightgbm import LGBMClassifier
    lgbm_threads = int(LGBMClassifier()._process_n_jobs(-1))
    from gameplay.labels import LABEL_VARIANTS, REQUIRED_LABEL_VARIANTS

    corpus_manifest = rsd.verify_shard_manifest(args.corpus, allow_partial=False)
    if corpus_manifest.get("feature_names_sha1") != V33_FEATURE_NAMES_SHA1:
        raise SystemExit(f"corpus feature_names_sha1 {corpus_manifest.get('feature_names_sha1')} is not v3.3")
    layout = corpus_layout(args.corpus)
    keep_cache = args.keep_cache or args.output.parent / "keep_v33.npz"
    ks = keep_and_samples(layout, keep_cache, str(corpus_manifest.get("run_id")))
    matrix_stat0 = args.matrix.stat()
    M = np.load(args.matrix, mmap_mode="r")
    matrix_report = verify_matrix(M, layout, ks)
    matrix_report.update(path=str(args.matrix), bytes=int(matrix_stat0.st_size), mtime_ns=int(matrix_stat0.st_mtime_ns))
    print(f"matrix verified: {matrix_report['shape']} ({matrix_report['n_sampled_rows_checked']} sampled rows exact)", flush=True)

    side = load_sidecars(args.sidecars, layout, str(corpus_manifest.get("run_id")), args.allow_partial_sidecars)
    built = side["manifest"].get("variants") or {}
    if args.variants == "required":
        names = list(REQUIRED_LABEL_VARIANTS)
    elif args.variants == "all":
        names = list(built)
    else:
        names = [s.strip() for s in args.variants.split(",") if s.strip()]
    if REFERENCE in names:
        names.remove(REFERENCE)
    names = [REFERENCE] + names
    for n in names:
        if n not in built or f"y_{n}" not in side:
            raise SystemExit(f"variant {n!r} is not in the sidecars")
        cur = LABEL_VARIANTS[n].as_dict() if n in LABEL_VARIANTS else None
        if cur is None or any(cur[k] != built[n][k] for k in ("label_type", "attribution", "overrides")):
            raise SystemExit(f"variant {n!r} differs between the sidecars and gameplay/labels.py: {built[n]} vs {cur}")
    labels = {n: side[f"y_{n}"].astype(np.int8) for n in names}
    groups = side["groups"].astype(str)
    classes = rsd.scale_class(side["cluster_blue"], side["cluster_red"], teamfight_min=args.teamfight_min)
    N = len(groups)
    print(f"sidecar rows={N} matches={len(np.unique(groups))} shards={side['shard_ids']} variants={names}", flush=True)

    preds_dir = args.output.with_suffix(".preds")
    preds_dir.mkdir(parents=True, exist_ok=True)
    oof_src_sha1 = hashlib.sha1(inspect.getsource(rsd.oof_predictions).encode("utf-8")).hexdigest()

    def fit(rowset: str, name: str, pos: np.ndarray):
        y = labels[name][pos]
        g = groups[pos]
        rows = side["matrix_row"][pos]
        spec = {"rowset": rowset, "variant": name, "n_splits": args.n_splits, "rows": _sha1_bytes(rows), "y": _sha1_bytes(y),
                "groups": _sha1_bytes(g), "oof_predictions_sha1": oof_src_sha1, "matrix": str(args.matrix),
                "keep": matrix_report["keep_sha1"]}
        spec_hash = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        path = preds_dir / f"{rowset}__{name}.npz"
        if args.resume and path.exists():
            with np.load(path, allow_pickle=False) as c:
                if str(c["spec_hash"]) == spec_hash:
                    print(f"[{rowset}] {name}: reusing {path.name}", flush=True)
                    return c["pred"].astype(np.float64), c["fold"], float(c["wall_s"])
        print(f"[{rowset}] {name}: OOF on n={len(pos)} matches={len(np.unique(g))} positives={float(y.mean()):.3f}", flush=True)
        t0 = time.time()
        pred = rsd.oof_predictions(RowView(M, rows), y, g, n_splits=args.n_splits)
        wall = time.time() - t0
        folds = fold_ids(y, g, args.n_splits)
        np.savez_compressed(path, pos=pos.astype(np.int32), matrix_row=rows, y=y, pred=pred, fold=folds, groups=g,
                            scale=classes[pos].astype(str), patch=side["patch"][pos].astype(str),
                            spec_hash=np.array(spec_hash), wall_s=np.array(wall))
        print(f"[{rowset}] {name}: AUC {roc_auc_score(y, pred):.4f} in {wall / 60:.1f} min", flush=True)
        return pred, folds, wall

    def fold_aucs(y, pred, folds):
        return [(_auc(y[folds == k], pred[folds == k])) for k in range(args.n_splits)]

    def base_block(y, pred, folds, g, cls, wall):
        return {"n": int(len(y)), "n_matches": int(len(np.unique(g))), "positive_rate": float(np.mean(y == 1)),
                "auc": _auc(y, pred), "fold_auc": fold_aucs(y, pred, folds), "fold_sha1": _sha1_bytes(folds),
                "oof_wall_s": round(float(wall), 1),
                "by_scale": {c: {"n": int((cls == c).sum()), "positive_rate": float(np.mean(y[cls == c] == 1)) if (cls == c).any() else None,
                                 "auc": _auc(y, pred, cls == c)} for c in CLASSES}}

    results: dict = {}
    reproduction = None
    for rowset in rowsets:
        t_set = time.time()
        if rowset == "common":
            pos = np.flatnonzero(np.logical_and.reduce([labels[n] >= 0 for n in names]))
            g, cls = groups[pos], classes[pos]
            fits = {n: fit("common", n, pos) for n in names}
            ref_pred = fits[REFERENCE][0]
            ref_sc = SortedScores(ref_pred)
            masks = {c: (cls == c) for c in CLASSES if (cls == c).sum() >= 50}
            stats, diffs = {}, {}
            for n in names:
                yv = labels[n][pos]
                sc = ref_sc if n == REFERENCE else SortedScores(fits[n][0])
                stats[f"{n}|own|all"] = (sc, yv, None)
                for c, mk in masks.items():
                    stats[f"{n}|own|{c}"] = (sc, yv, mk)
                if n != REFERENCE:
                    stats[f"{n}|transfer|all"] = (ref_sc, yv, None)
                    for c, mk in masks.items():
                        stats[f"{n}|transfer|{c}"] = (ref_sc, yv, mk)
                    diffs[f"{n}|own_minus_market_event"] = (f"{n}|own|all", f"{REFERENCE}|own|all")
                    diffs[f"{n}|transfer_minus_own"] = (f"{n}|transfer|all", f"{n}|own|all")
            print(f"[common] bootstrap {args.n_boot} x {len(stats)} statistics on {len(pos)} rows", flush=True)
            boot, dboot = paired_bootstrap(g, stats, diffs, args.n_boot)
            block = {"n": int(len(pos)), "n_matches": int(len(np.unique(g))), "share_of_sidecar_rows": float(len(pos) / max(N, 1)),
                     "class_counts": {c: int((cls == c).sum()) for c in CLASSES}, "variants": {}}
            for n in names:
                yv = labels[n][pos]
                pred, folds, wall = fits[n]
                r = base_block(yv, pred, folds, g, cls, wall)
                r["auc_ci"] = boot[f"{n}|own|all"]
                for c in CLASSES:
                    r["by_scale"][c]["auc_ci"] = boot.get(f"{n}|own|{c}")
                if n != REFERENCE:
                    r["transfer"] = {"auc": _auc(yv, ref_pred), "auc_ci": boot[f"{n}|transfer|all"],
                                     "transfer_minus_own": dboot[f"{n}|transfer_minus_own"],
                                     "by_scale": {c: {"auc": _auc(yv, ref_pred, cls == c), "auc_ci": boot.get(f"{n}|transfer|{c}")} for c in CLASSES}}
                    r["own_minus_market_event"] = dboot[f"{n}|own_minus_market_event"]
                    r["agreement_vs_market_event"] = agreement(yv, labels[REFERENCE][pos])
                block["variants"][n] = r
                print(f"[common] {n:32s} auc={r['auc']:.4f}" + (f" transfer={r['transfer']['auc']:.4f}" if n != REFERENCE else ""), flush=True)
        else:
            ref_pos = np.flatnonzero(labels[REFERENCE] >= 0)
            ref_pred_rows, ref_folds, ref_wall = fit("own", REFERENCE, ref_pos)
            ref_full = np.zeros(N, dtype=np.float64)
            ref_full[ref_pos] = ref_pred_rows
            ref_has = labels[REFERENCE] >= 0
            block = {"variants": {}}
            for n in names:
                pos = np.flatnonzero(labels[n] >= 0)
                g, cls = groups[pos], classes[pos]
                pred, folds, wall = (ref_pred_rows, ref_folds, ref_wall) if n == REFERENCE else fit("own", n, pos)
                yv = labels[n][pos]
                both = ref_has[pos]
                y_both = np.where(both, yv, -1).astype(np.int8)
                sc = SortedScores(pred)
                masks = {c: (cls == c) for c in CLASSES if (cls == c).sum() >= 50}
                stats = {"own|all": (sc, yv, None)}
                stats.update({f"own|{c}": (sc, yv, mk) for c, mk in masks.items()})
                diffs = {}
                if n != REFERENCE:
                    tr_sc = SortedScores(ref_full[pos])
                    stats["own_on_both|all"] = (sc, y_both, None)
                    stats["transfer|all"] = (tr_sc, y_both, None)
                    stats.update({f"transfer|{c}": (tr_sc, y_both, mk) for c, mk in masks.items()})
                    diffs["transfer_minus_own_on_both"] = ("transfer|all", "own_on_both|all")
                print(f"[own] {n}: bootstrap {args.n_boot} x {len(stats)} statistics on {len(pos)} rows", flush=True)
                boot, dboot = paired_bootstrap(g, stats, diffs, args.n_boot)
                r = base_block(yv, pred, folds, g, cls, wall)
                r["share_of_sidecar_rows"] = float(len(pos) / max(N, 1))
                r["auc_ci"] = boot["own|all"]
                for c in CLASSES:
                    r["by_scale"][c]["auc_ci"] = boot.get(f"own|{c}")
                if n != REFERENCE:
                    r["transfer"] = {"n_both": int(both.sum()), "auc": _auc(y_both, ref_full[pos]), "auc_ci": boot["transfer|all"],
                                     "own_auc_on_both": _auc(y_both, pred), "own_auc_on_both_ci": boot["own_on_both|all"],
                                     "transfer_minus_own_on_both": dboot["transfer_minus_own_on_both"],
                                     "by_scale": {c: {"auc": _auc(y_both, ref_full[pos], cls == c), "auc_ci": boot.get(f"transfer|{c}")} for c in CLASSES}}
                block["variants"][n] = r
                print(f"[own] {n:32s} n={len(pos)} auc={r['auc']:.4f}" + (f" transfer={r['transfer']['auc']:.4f}" if n != REFERENCE else ""), flush=True)
            if args.reference_preds and args.reference_preds.exists():
                q = np.load(args.reference_preds, allow_pickle=False)
                comparable = len(q["y"]) == len(ref_pos) and np.array_equal(q["groups"].astype(str), groups[ref_pos]) \
                    and np.array_equal(q["y"], labels[REFERENCE][ref_pos])
                reproduction = {"reference_preds": str(args.reference_preds), "rows_identical": bool(comparable)}
                if comparable:
                    reproduction.update(auc_published=float(roc_auc_score(q["y"], q["pred"])), auc_now=float(roc_auc_score(q["y"], ref_pred_rows)),
                                        preds_identical=bool(np.array_equal(q["pred"], ref_pred_rows)),
                                        max_abs_pred_diff=float(np.max(np.abs(q["pred"] - ref_pred_rows))),
                                        pearson=float(np.corrcoef(q["pred"], ref_pred_rows)[0, 1]))
                print(f"reproduction of the published market_event OOF: {reproduction}", flush=True)
        block["wall_s"] = round(time.time() - t_set, 1)
        results[rowset] = block

    matrix_stat1 = args.matrix.stat()
    matrix_changed = (matrix_stat1.st_size, matrix_stat1.st_mtime_ns) != (matrix_stat0.st_size, matrix_stat0.st_mtime_ns)
    git_commit, git_dirty = _git_state()
    deviations = list(DEVIATIONS)
    if args.n_splits != 5:
        deviations.append(f"SMOKE: {args.n_splits} folds instead of the headline 5.")
    if args.n_boot != 1000:
        deviations.append(f"SMOKE: {args.n_boot} bootstrap replicates instead of 1000.")
    if args.allow_partial_sidecars:
        deviations.append("SMOKE: partial sidecars (subset of shards and/or matches); numbers are not corpus estimates.")
    out = {
        "item": "A2-label-family", "script": "scripts/run_label_family_v33.py",
        "reference_followed": "scripts/run_scale_decomposition.py oof_predictions / scale_class / verify_shard_manifest / cluster_bootstrap draws",
        "git_commit": git_commit, "git_dirty": git_dirty, "preset": side["manifest"].get("preset"),
        "label_key_reference": f"y_{REFERENCE}", "label_keys": {n: f"y_{n}" for n in names},
        "split": f"match-grouped GroupKFold(n_splits={args.n_splits}) OOF within each row set, all patches (headline protocol)",
        "seed": SEED, "n_splits": args.n_splits, "n_boot": args.n_boot, "teamfight_min": args.teamfight_min,
        "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_clock_s": round(time.time() - t_run, 1),
        "python": sys.executable, "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__},
        "lgbm_num_threads": lgbm_threads, "env": {k: os.environ.get(k) for k in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "LOL_OUTPUT_ROOT")},
        "corpus": {"dir": str(args.corpus), **{k: corpus_manifest.get(k) for k in ("run_id", "git_commit", "num_shards", "complete", "feature_names_sha1")}},
        "sidecars": {"dir": str(args.sidecars), "shards": side["shard_ids"], "full": side["manifest"].get("full"),
                     "limit_matches_per_shard": side["manifest"].get("limit_matches_per_shard"),
                     "git_commit": side["manifest"].get("git_commit"), "run_id": side["manifest"].get("run_id"),
                     "n_rows": int(N), "n_matches": int(len(np.unique(groups)))},
        "matrix": {**matrix_report, "keep_cache": str(keep_cache), "changed_during_run": bool(matrix_changed)},
        "partial": bool(args.allow_partial_sidecars and not side["manifest"].get("full")),
        "variants": {n: built[n] for n in names},
        "agreement_vs_market_event": {n: agreement(labels[n], labels[REFERENCE]) for n in names if n != REFERENCE},
        "rowsets": results, "reproduction_market_event_own": reproduction,
        "deviations": deviations,
        "variant_notes": {n: {"description": built[n].get("description"), "source": built[n].get("source"),
                              "verification": built[n].get("verification")} for n in names},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({(time.time() - t_run) / 60:.1f} min)", flush=True)
    if matrix_changed:
        raise SystemExit(f"{args.matrix} changed during the run; results flagged in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
