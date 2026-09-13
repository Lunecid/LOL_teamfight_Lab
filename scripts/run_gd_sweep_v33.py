"""G x D sensitivity of the corpus-v3.3 engagement-outcome result (CoG 2026 #118, reviewer R1).

R1: "Why is clustering done with kills within an 18 seconds interval? ... why specifically 4,000 game
units?"  Corpus v3.3 derives both constants from data (G = 13.7246 s kill-interval KDE antimode, D =
4,263.87 u same-champion sharing crossover; docs/DEFINITION_EVIDENCE.md section 17, injected as 13,700 ms
and 4,264 u), but the objection is retired only if the result does not hinge on where they sit.  This
sweep re-detects engagements on a declared match subsample at every

    G in {10, 12, 13.7, 16, 18} s  x  D in {3,500, 4,000, 4,264, 4,750} u        (20 settings)

with every other definition constant at the v3.3 preset (core/presets.py: R 1,600 u, B 15 s, M 2, 35 s
horizon, market_event label attributed to the engagement with a 300 g dead zone, clean feature path).
G is TF2_KILL_CLUSTER_GAP_MS (temporal chaining of consecutive kills); D is CLUSTER_MAX_DIAMETER, which
gameplay/fights.py uses both to split a temporal cluster spatially and as the post-merge spacing radius.
Draws are dropped (tie policy "drop", as for y_market_event in the shards).

Pipeline per setting, the path of scripts/run_threshold_sensitivity.py and scripts/build_corpus_shard.py:
data.index_split.build_fight_index -> train.baseline.build_tabular_Xy(feature_set="full"), run in worker
processes that receive the setting through LOL_CFG_PRESET / LOL_CFG_OVERRIDES before core.config is
imported; the fight-index disk cache is off, so no index built under another setting can be reused.

Learner and split: the published LightGBM configuration (Ke et al. 2017, "LightGBM: A Highly Efficient
Gradient Boosting Decision Tree", NeurIPS; LightGBM 4.6 LGBMClassifier) as refit under the patch holdout by
``lgbm_paper`` in scripts/run_model_comparison_v33.py of the engagement-state-value worktree (400 trees,
learning rate 0.05, 31 leaves, subsample 0.9, colsample_bytree 0.9, early stopping after 100 rounds on the
validation patch, eval_metric auc; features/model_comparison_v33_patch.json): trained on patch 15.14,
early-stopped on 15.15, tested once on 15.16.  Nothing from 15.16 reaches training, early stopping or any
selection, and nothing is calibrated.  Workers are checked to have run preset v3.3 (clean feature path) with
only G and D changed and draws dropped; a setting whose workers did not is refused.

Per setting: detected engagements, engagements per sampled match, labelled rows and positive rate per
split, test AUC with a match-clustered percentile bootstrap CI (Efron & Tibshirani 1993 ch. 13; matches
resampled whole as in Field & Welsh 2007, JRSS-B 69(3)), and pick / skirmish / teamfight test AUCs
(smaller-side participation, teamfight >= 4) with CIs.

Cross-setting, against the operating point G 13.7 s / D 4,264 u on the same subsample:
  identity   an engagement is (match id, first-kill time).  B is fixed at 15 s, so the cutoff moves with the
             first kill and the two identities are equivalent.
  overlap    engagements of setting s and of the operating point are matched one-to-one inside a match:
             candidate pairs have |t_s - t_op| <= tolerance (default 5,000 ms, the MATCH_TOLERANCE_MS of
             run_threshold_sensitivity.py) and are accepted greedily by increasing |dt|.  Reported: share of
             operating-point engagements matched, share of s matched, Jaccard, and matches at dt = 0.
  shared     labelled test rows matched the same way.  On them: label agreement, AUC of s (own model and
             labels), AUC of the operating point, their paired difference, and the operating-point model
             scored against s's labels, which splits the difference into a label effect (op model: s labels
             minus op labels) and a model effect (s labels: s model minus op model).  AUCs on rows only one
             side has are reported too.
  full test  AUC_s - AUC_op on each setting's own test rows.
  Every paired statistic resamples the union of both settings' test matches jointly.

Resumable: a finished setting writes settings/<name>.json and settings/<name>.npz (engagement keys, labels,
predictions; never X) and is skipped on the next run when its signature matches.  --aggregate-only rebuilds
the cross-setting table from disk.  --validate-shards checks that the operating point reproduces the corpus
rows (keys, labels, counts, and X for the shards holding most sampled matches).

    LOL_OUTPUT_ROOT=D:/LOL_Project .venv/Scripts/python.exe scripts/run_gd_sweep_v33.py --n-matches 10000 \\
        --workers 8 --n-jobs 8 --validate-shards D:/LOL_Project/fusion_2615/corpus_shards_v33
    # smoke: two settings on 100 matches
    LOL_OUTPUT_ROOT=D:/LOL_Project .venv/Scripts/python.exe scripts/run_gd_sweep_v33.py --n-matches 100 \\
        --settings 13.7:4264,18:4000 --n-boot 200 --out-dir <scratch dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRESET = "v3.3"
LABEL_KEY = "market_event (preset LABEL_TYPE, engagement attribution, 300 g dead zone), tie policy drop"
G_VALUES_S = "10,12,13.7,16,18"
D_VALUES_U = "3500,4000,4264,4750"
OPERATING_POINT = (13700, 4264.0)
PAPER_LGBM = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9)
EARLY_STOPPING_ROUNDS = 100
MATCH_TOLERANCE_MS = 5000
KEY_BASE = 10 ** 10  # > any in-game millisecond timestamp plus tolerance: keys of different matches never collide
CFG_KEYS_RECORDED = (
    "TF2_KILL_CLUSTER_GAP_MS", "CLUSTER_MAX_DIAMETER", "TF2_VALIDITY_RADIUS", "TF2_ENGAGE_PRE_KILL_MS",
    "TF2_MIN_PER_TEAM", "TF2_INTERACTION_RADIUS", "TF2_POST_FIGHT_WINDOW_MS", "FIGHT_HORIZON_SEC",
    "FIGHT_CONTEXT_SEC", "START_OFFSET_MIN", "CONTINUOUS_FIGHT_MERGE", "CONTINUOUS_FIGHT_MERGE_RADIUS",
    "MAX_MERGED_FIGHT_DURATION_MS", "LABEL_TYPE", "LABEL_EVENT_ATTRIBUTION", "LABEL_GOLD_DEADZONE",
    "LABEL_TIE_STRATEGY", "TIME_NORM_ABSOLUTE", "TIME_NORM_DENOM_MIN", "ANCHORS_CAUSAL", "TAB_FRAME_AGE_FEATURE",
    "FEATURE_VERSION", "CACHE_DIRNAME", "FIGHT_INDEX_CACHE_ENABLED", "FIGHT_INDEX_NUM_WORKERS", "DUMP_FIGHTS",
)
REF_FIELDS = ("match_id", "patch", "engage_ts", "first_kill_ts", "last_kill_ts",
              "cluster_blue", "cluster_red", "present_blue", "present_red")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_LOG_PATH = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _LOG_PATH is not None:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def setting_name(g_ms: int, d_u: float) -> str:
    return f"G{int(g_ms)}_D{int(round(d_u))}"


def parse_settings(args) -> list:
    if args.settings:
        grid = []
        for tok in str(args.settings).split(","):
            g, d = tok.strip().split(":")
            grid.append((int(round(float(g) * 1000)), float(d)))
    else:
        grid = [(int(round(float(g) * 1000)), float(d))
                for g in str(args.g_values).split(",") for d in str(args.d_values).split(",")]
    ordered, seen = [], set()
    for s in [OPERATING_POINT] + grid:  # operating point first: every other setting is compared with it
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    return ordered


def operational_overrides() -> dict:
    """Worker overrides that are not definition constants: draws dropped, no fight-index disk cache (an index
    built under another setting can never be reused), detection in-process, no fight dumps."""
    return {"LABEL_TIE_STRATEGY": "drop", "FIGHT_INDEX_CACHE_ENABLED": False, "FIGHT_INDEX_NUM_WORKERS": 1,
            "DUMP_FIGHTS": False}


def worker_overrides(g_ms: int, d_u: float) -> dict:
    return {"TF2_KILL_CLUSTER_GAP_MS": int(g_ms), "CLUSTER_MAX_DIAMETER": float(d_u), **operational_overrides()}


# ----------------------------------------------------------------------------- worker processes
def _worker_init(overrides_json: str, preset: str = PRESET) -> None:
    """Runs in a fresh spawned worker before any task: the preset and overrides must be in the environment
    before core.config is imported, because core.config applies both at import time."""
    if "core.config" in sys.modules:
        raise RuntimeError("core.config was imported before the worker received its preset and overrides")
    os.environ["LOL_CFG_PRESET"] = str(preset)
    os.environ["LOL_CFG_OVERRIDES"] = overrides_json
    import warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    from core.config import cfg
    cfg.LABEL_TIE_POLICY = "drop"  # read before LABEL_TIE_STRATEGY by gameplay/labels.py; not a CFG field
    for k, v in json.loads(overrides_json).items():
        if getattr(cfg, k) != v:
            raise RuntimeError(f"override {k}={v!r} did not reach cfg (got {getattr(cfg, k)!r})")
    if "torch" in sys.modules:
        try:
            sys.modules["torch"].set_num_threads(1)
        except Exception:
            pass


def make_pool(overrides: dict, workers: int, preset: str = PRESET):
    """Process pool whose workers run ``_work_chunk`` under ``preset`` + ``overrides``."""
    from concurrent.futures import ProcessPoolExecutor
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"  # inherited by the workers only; the parent's BLAS pools are already sized
    return ProcessPoolExecutor(max_workers=max(1, int(workers)), initializer=_worker_init,
                               initargs=(json.dumps(overrides), str(preset)))


def _work_chunk(task):
    idx, mids = task
    from core.config import cfg
    from data.index_split import build_fight_index
    from train.baseline import build_tabular_Xy
    t0 = time.time()
    refs = build_fight_index(cache_match_ids=list(mids))
    X, y, names, used = build_tabular_Xy(refs, feature_set="full")
    pos = {id(r): i for i, r in enumerate(refs)}
    arrays = {
        "match_id": np.array([r.match_id for r in refs], dtype=str),
        "patch": np.array([r.patch for r in refs], dtype=str),
        "engage_ts": np.array([r.t_start_ts for r in refs], dtype=np.int64),
        "first_kill_ts": np.array([r.first_kill_ts for r in refs], dtype=np.int64),
        "last_kill_ts": np.array([r.last_kill_ts for r in refs], dtype=np.int64),
        "cluster_blue": np.array([r.det_cluster_blue for r in refs], dtype=np.int16),
        "cluster_red": np.array([r.det_cluster_red for r in refs], dtype=np.int16),
        "present_blue": np.array([r.det_present_blue for r in refs], dtype=np.int16),
        "present_red": np.array([r.det_present_red for r in refs], dtype=np.int16),
    }
    effective = None
    if idx == 0:
        effective = {k: getattr(cfg, k, None) for k in CFG_KEYS_RECORDED}
        effective["LABEL_TIE_POLICY"] = getattr(cfg, "LABEL_TIE_POLICY", None)
        effective["LOL_CFG_PRESET"] = os.environ.get("LOL_CFG_PRESET")
    return {"idx": idx, "n_matches": len(mids), "seconds": time.time() - t0, "refs": arrays,
            "used_idx": np.array([pos[id(r)] for r in used], dtype=np.int64),
            "X": np.asarray(X, dtype=np.float32) if len(used) else None,
            "y": np.asarray(y, dtype=np.int8) if len(used) else np.zeros(0, dtype=np.int8),
            "names": list(names) if len(used) else None, "cfg": effective}


def build_setting(g_ms: int, d_u: float, mids: list, workers: int, chunk_size: int) -> dict:
    tasks = [(i, mids[s:s + chunk_size]) for i, s in enumerate(range(0, len(mids), chunk_size))]
    parts, names, names_sha1, effective = [], None, None, None
    t0 = time.time()
    with make_pool(worker_overrides(g_ms, d_u), workers) as ex:
        for k, res in enumerate(ex.map(_work_chunk, tasks)):
            if res["cfg"] is not None:
                effective = res["cfg"]
            if res["names"]:
                sha = hashlib.sha1("\n".join(res["names"]).encode("utf-8")).hexdigest()
                if names_sha1 is None:
                    names, names_sha1 = res["names"], sha
                elif sha != names_sha1:
                    raise RuntimeError("feature names differ between chunks")
                res["names"] = None
            parts.append(res)
            if (k + 1) % max(1, len(tasks) // 10) == 0 or k + 1 == len(tasks):
                done = sum(p["n_matches"] for p in parts)
                log(f"    {done:,}/{len(mids):,} matches ({time.time() - t0:.0f}s)")
    refs = {key: np.concatenate([p["refs"][key] for p in parts]) for key in REF_FIELDS}
    offsets = np.cumsum([0] + [len(p["refs"]["match_id"]) for p in parts])
    row_ref_idx = np.concatenate([p["used_idx"] + off for p, off in zip(parts, offsets[:-1])]).astype(np.int64)
    y = np.concatenate([p["y"] for p in parts]).astype(np.int8)
    xs = [p["X"] for p in parts if p["X"] is not None]
    X = np.concatenate(xs, axis=0) if xs else np.zeros((0, 0), dtype=np.float32)
    del parts, xs
    return {"refs": refs, "row_ref_idx": row_ref_idx, "y": y, "X": X, "names": names, "names_sha1": names_sha1,
            "effective_cfg": effective, "build_seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- evaluation helpers
def smaller_side(blue: np.ndarray, red: np.ndarray, negative_as: str) -> np.ndarray:
    """min(blue, red).  A negative count is data/index_split._fight_to_ref_row's ``int(x or -1)`` applied to a
    zero count, so it is read as 0 by default; ``negative_as='unknown'`` keeps it out of every class."""
    b, r = blue.astype(np.int64), red.astype(np.int64)
    if negative_as == "zero":
        return np.minimum(np.maximum(b, 0), np.maximum(r, 0))
    out = np.minimum(b, r)
    out[(b < 0) | (r < 0)] = -1
    return out


def scale_masks(nmin: np.ndarray, teamfight_min: int) -> dict:
    return {"pick": (nmin >= 0) & (nmin <= 1), "skirmish": (nmin >= 2) & (nmin < teamfight_min),
            "teamfight": nmin >= teamfight_min}


def point_auc(y, p, min_rows: int) -> float | None:
    from sklearn.metrics import roc_auc_score
    if len(y) < min_rows or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def match_one_to_one(mid_a, t_a, mid_b, t_b, tol: int):
    """One-to-one pairs (i, j) with mid_a[i] == mid_b[j] and |t_a[i] - t_b[j]| <= tol, accepted greedily by
    increasing |dt| (ties: lower i, then lower j).  Returns index arrays into a and b and |dt|."""
    mid_a, mid_b = np.asarray(mid_a), np.asarray(mid_b)
    if len(mid_a) == 0 or len(mid_b) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    universe = np.unique(np.concatenate([mid_a, mid_b]))
    ka = np.searchsorted(universe, mid_a).astype(np.int64) * KEY_BASE + np.asarray(t_a, dtype=np.int64)
    kb = np.searchsorted(universe, mid_b).astype(np.int64) * KEY_BASE + np.asarray(t_b, dtype=np.int64)
    ob = np.argsort(kb, kind="stable")
    kbs = kb[ob]
    lo = np.searchsorted(kbs, ka - tol, side="left")
    hi = np.searchsorted(kbs, ka + tol, side="right")
    ci, cj, cd = [], [], []
    for i in np.flatnonzero(hi > lo):
        for jj in range(int(lo[i]), int(hi[i])):
            j = int(ob[jj])
            ci.append(int(i))
            cj.append(j)
            cd.append(abs(int(ka[i]) - int(kb[j])))
    if not ci:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    ci, cj, cd = np.asarray(ci), np.asarray(cj), np.asarray(cd)
    order = np.lexsort((cj, ci, cd))
    used_a = np.zeros(len(mid_a), dtype=bool)
    used_b = np.zeros(len(mid_b), dtype=bool)
    pa, pb, pd = [], [], []
    for k in order:
        i, j = ci[k], cj[k]
        if not used_a[i] and not used_b[j]:
            used_a[i] = used_b[j] = True
            pa.append(i)
            pb.append(j)
            pd.append(cd[k])
    return np.asarray(pa, np.int64), np.asarray(pb, np.int64), np.asarray(pd, np.int64)


def anchor_time(refs: dict, b_ms: int) -> np.ndarray:
    fk = refs["first_kill_ts"].astype(np.int64)
    return np.where(fk >= 0, fk, refs["engage_ts"].astype(np.int64) + int(b_ms))


def test_auc_block(yt, pt, gt, nmin, teamfight_min: int, n_boot: int, seed: int, boot_mod,
                   extra_masks: dict | None = None) -> dict:
    """Test AUC with a match-clustered percentile interval, AUC per participation class (pick / skirmish /
    teamfight on the smaller side ``nmin``) and the pick - teamfight gap, all on the same replicates: matches
    are drawn with replacement (Field & Welsh 2007) and match multiplicities weight the rows of the
    Mann-Whitney AUC (boot_mod.WeightedAUC).  ``extra_masks`` (name -> row mask) adds more subsets under
    ``extra`` without changing the replicate stream."""
    yt = np.asarray(yt).astype(np.int64)
    pt = np.asarray(pt, dtype=np.float64)
    masks = {"overall": np.ones(len(yt), dtype=bool), **scale_masks(nmin, teamfight_min)}
    extra = dict(extra_masks or {})
    all_masks = {**masks, **{f"extra/{k}": np.asarray(m, dtype=bool) for k, m in extra.items()}}
    point = {k: point_auc(yt[m], pt[m], boot_mod.MIN_ROWS) for k, m in all_masks.items()}
    wa = boot_mod.WeightedAUC(yt, pt)
    boot = boot_mod.MatchBootstrap(gt, seed)
    live = [k for k in all_masks if point[k] is not None]
    msorted = {k: wa.sort(all_masks[k]) for k in live}
    draws = {k: [] for k in live}
    gap = []
    for _ in range(int(n_boot)):
        counts, _sample = boot.draw()
        ws = wa.sort(counts[boot.inverse])
        vals = {k: wa.from_sorted(ws * msorted[k]) for k in live}
        for k, v in vals.items():
            draws[k].append(v)
        if "pick" in vals and "teamfight" in vals and np.isfinite(vals["pick"]) and np.isfinite(vals["teamfight"]):
            gap.append(vals["pick"] - vals["teamfight"])
    test = {"auc": boot_mod.summarize(draws.get("overall", []), point["overall"]), "classes": {}}
    for k in ("pick", "skirmish", "teamfight"):
        test["classes"][k] = {"n": int(masks[k].sum()), "positive_rate": (float(yt[masks[k]].mean()) if masks[k].any() else None),
                              "auc": boot_mod.summarize(draws.get(k, []), point[k])}
    pt_gap = (point["pick"] - point["teamfight"]) if (point["pick"] is not None and point["teamfight"] is not None) else None
    test["pick_minus_teamfight"] = boot_mod.summarize(gap, pt_gap, gap=True)
    if extra:
        test["extra"] = {k: {"n": int(all_masks[f"extra/{k}"].sum()),
                             "positive_rate": (float(yt[all_masks[f"extra/{k}"]].mean()) if all_masks[f"extra/{k}"].any() else None),
                             "auc": boot_mod.summarize(draws.get(f"extra/{k}", []), point[f"extra/{k}"])}
                         for k in extra}
    return test


def paired_vs_reference(ref: dict, cand: dict, tol: int, n_boot: int, seed: int, boot_mod) -> dict:
    """Overlap, shared test rows and paired AUC differences of a candidate definition against a reference.

    ``ref`` / ``cand``: ``match_id`` and ``t`` (anchor time) of every detected engagement, and ``test_mid``,
    ``test_t``, ``y``, ``p`` of the labelled test rows.  Engagements, and separately test rows, are matched
    one-to-one inside a match within ``tol`` ms (match_one_to_one).  Every AUC and every difference is computed
    on the same replicates, which resample the union of both sides' test matches jointly.  On shared rows the
    difference splits into a label effect (reference model scored against the candidate's labels minus the
    reference) and a model effect (candidate model minus reference model, both on the candidate's labels)."""
    pa, pb, dt = match_one_to_one(ref["match_id"], ref["t"], cand["match_id"], cand["t"], tol)
    n_op, n_s, m = len(ref["t"]), len(cand["t"]), len(pa)
    overlap = {"tolerance_ms": tol, "identity": "(match id, first-kill time)", "n_operating_point": n_op, "n_setting": n_s,
               "matched": m, "share_of_operating_point_matched": m / max(1, n_op), "share_of_setting_matched": m / max(1, n_s),
               "jaccard": m / max(1, n_op + n_s - m), "matched_at_dt0": int((dt == 0).sum()),
               "share_of_operating_point_matched_at_dt0": float((dt == 0).sum() / max(1, n_op))}
    op_rmid, op_rt, op_y, op_p = ref["test_mid"], ref["test_t"], np.asarray(ref["y"]).astype(np.int64), ref["p"]
    s_rmid, s_rt, s_y, s_p = cand["test_mid"], cand["test_t"], np.asarray(cand["y"]).astype(np.int64), cand["p"]
    ra, rb, rdt = match_one_to_one(op_rmid, op_rt, s_rmid, s_rt, tol)
    only_s = np.ones(len(s_y), dtype=bool)
    only_s[rb] = False
    only_op = np.ones(len(op_y), dtype=bool)
    only_op[ra] = False
    boot = boot_mod.MatchBootstrap(np.concatenate([op_rmid, s_rmid]), seed)
    stats = {
        "setting_full": (boot_mod.WeightedAUC(s_y, s_p), boot.index_of(s_rmid)),
        "operating_point_full": (boot_mod.WeightedAUC(op_y, op_p), boot.index_of(op_rmid)),
        "setting_shared": (boot_mod.WeightedAUC(s_y[rb], s_p[rb]), boot.index_of(s_rmid[rb])),
        "operating_point_shared": (boot_mod.WeightedAUC(op_y[ra], op_p[ra]), boot.index_of(op_rmid[ra])),
        "op_model_setting_labels_shared": (boot_mod.WeightedAUC(s_y[rb], op_p[ra]), boot.index_of(s_rmid[rb])),
        "setting_only": (boot_mod.WeightedAUC(s_y[only_s], s_p[only_s]), boot.index_of(s_rmid[only_s])),
        "operating_point_only": (boot_mod.WeightedAUC(op_y[only_op], op_p[only_op]), boot.index_of(op_rmid[only_op])),
    }
    diffs = {"full_setting_minus_op": ("setting_full", "operating_point_full"),
             "shared_setting_minus_op": ("setting_shared", "operating_point_shared"),
             "shared_label_effect": ("op_model_setting_labels_shared", "operating_point_shared"),
             "shared_model_effect": ("setting_shared", "op_model_setting_labels_shared")}
    point = {k: a(np.ones(len(i))) for k, (a, i) in stats.items()}
    point = {k: (v if np.isfinite(v) else None) for k, v in point.items()}
    draws = {k: [] for k in stats}
    ddraws = {k: [] for k in diffs}
    for _ in range(int(n_boot)):
        counts, _sample = boot.draw()
        vals = {k: (a(counts[i]) if point[k] is not None else float("nan")) for k, (a, i) in stats.items()}
        for k, v in vals.items():
            draws[k].append(v)
        for k, (u, v) in diffs.items():
            if np.isfinite(vals[u]) and np.isfinite(vals[v]):
                ddraws[k].append(vals[u] - vals[v])
    auc = {k: boot_mod.summarize(draws[k], point[k]) for k in stats}
    paired = {k: boot_mod.summarize(ddraws[k], (point[u] - point[v]) if (point[u] is not None and point[v] is not None) else None, gap=True)
              for k, (u, v) in diffs.items()}
    shared = {"tolerance_ms": tol, "n_test_setting": int(len(s_y)), "n_test_operating_point": int(len(op_y)),
              "n_shared": int(len(ra)), "n_only_setting": int(only_s.sum()), "n_only_operating_point": int(only_op.sum()),
              "shared_at_dt0": int((rdt == 0).sum()),
              "label_agreement_shared": (float(np.mean(s_y[rb] == op_y[ra])) if len(ra) else None),
              "n_bootstrap_matches": int(len(boot.unique))}
    return {"overlap": overlap, "shared": shared, "auc": auc, "paired": paired}


# ----------------------------------------------------------------------------- one setting
def sweep_deviations(args, n_matches: int) -> list:
    """Departures from the headline protocol, written into every per-setting JSON and the summary."""
    return [
        (f"Match subsample of {n_matches:,} cache matches (seed {args.seed}), not the full corpus."
         if args.n_matches else "Every cached match (no subsample)."),
        "Learner protocol is the patch-holdout refit of the paper configuration (trained on 15.14, early stopping on "
        "15.15, as lgbm_paper in run_model_comparison_v33.py of the engagement-state-value worktree), not the "
        "match-grouped 5-fold OOF of the 0.6699 headline.",
        "LightGBM gets all 7,106 columns; columns constant on the training rows are never split on, so this equals "
        "dropping them, but it differs from the headline's corpus-wide constant filter (6,164 columns).",
        "Draws are dropped through the label tie policy inside build_tabular_Xy instead of building rows with a "
        "random tie policy and masking y_market_event afterwards; --validate-shards measures the agreement.",
        f"Negative participation counts are read as {args.negative_count_as} (the int(x or -1) coercion of a zero "
        "count in data/index_split._fight_to_ref_row).",
        "Overlap and shared rows use a one-to-one greedy match within the tolerance; the existing "
        "run_threshold_sensitivity.overlap_fraction counts any match within tolerance (not one-to-one).",
        "subsample=0.9 is inert because subsample_freq stays 0 (LightGBM bagging off), exactly as in lgbm_paper "
        "and the OOF headline; it is kept so the parameter dict is the published one.",
        f"LightGBM runs with n_jobs={args.n_jobs} instead of -1; the thread count does not change the fitted trees.",
    ]


def fit_and_score(data: dict, args, boot_mod) -> tuple:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    refs, ridx, X, y = data["refs"], data["row_ref_idx"], data["X"], data["y"].astype(np.int64)
    patch = refs["patch"][ridx]
    groups = refs["match_id"][ridx]
    split = np.full(len(y), -1, dtype=np.int8)
    split[patch == args.train_patch] = 0
    split[patch == args.val_patch] = 1
    split[patch == args.test_patch] = 2
    tr, va, te = split == 0, split == 1, split == 2
    for name, m in (("train", tr), ("val", va), ("test", te)):
        if int(m.sum()) < 2 or len(np.unique(y[m])) < 2:
            raise RuntimeError(f"{name} split has {int(m.sum())} rows / one class; enlarge --n-matches")
    params = dict(PAPER_LGBM, random_state=int(args.seed), n_jobs=int(args.n_jobs), verbose=-1)
    model = LGBMClassifier(**params)
    t0 = time.time()
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
              callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)])
    fit_s = time.time() - t0
    pred = np.full(len(y), np.nan)
    pred[va] = model.predict_proba(X[va])[:, 1]
    pred[te] = model.predict_proba(X[te])[:, 1]

    yt, pt, gt = y[te], pred[te], groups[te]
    nmin = smaller_side(refs["cluster_blue"][ridx], refs["cluster_red"][ridx], args.negative_count_as)[te]
    test = test_auc_block(yt, pt, gt, nmin, args.teamfight_min, args.n_boot, args.seed, boot_mod)

    def split_stats(m):
        return {"rows": int(m.sum()), "matches": int(len(np.unique(groups[m]))),
                "positive_rate": (float(y[m].mean()) if m.any() else None)}

    importance = model.booster_.feature_importance("split")
    model_info = {"learner": "lightgbm.LGBMClassifier", "params": params, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                  "best_iteration": (int(model.best_iteration_) if getattr(model, "best_iteration_", None) else None),
                  "val_auc": point_auc(y[va], pred[va], 2), "train_rows": int(tr.sum()),
                  "n_features": int(X.shape[1]), "n_features_split_on": int((importance > 0).sum()),
                  "fit_seconds": round(fit_s, 1)}
    splits = {"train": split_stats(tr), "val": split_stats(va), "test": split_stats(te),
              "other_patch_rows": int((split < 0).sum())}
    return pred, split, model_info, splits, test


def validate_corpus_reproduction(shard_dir: Path, all_mids_sorted: list, sample: list, data: dict, x_shards: int,
                                 boot_mod) -> dict:
    """Does the operating point reproduce corpus_shards_v33 on the sampled matches?"""
    manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
    num_shards = int(manifest["num_shards"])
    keys = ["groups", "engage_ts", "y_market_event", "cluster_blue", "cluster_red", "present_blue", "present_red"]
    sh = boot_mod.load_shard_arrays(shard_dir, keys)
    sel = np.flatnonzero(np.isin(sh["groups"], np.asarray(sample)))
    shard_key = {(str(sh["groups"][i]), int(sh["engage_ts"][i])): int(i) for i in sel}
    refs, ridx, y = data["refs"], data["row_ref_idx"], data["y"]
    our_key = {(str(m), int(t)): i for i, (m, t) in enumerate(zip(refs["match_id"], refs["engage_ts"]))}
    common = sorted(set(our_key) & set(shard_key))
    out = {"shard_rows_for_sample": len(shard_key), "our_engagements": len(our_key), "common_engagements": len(common),
           "only_in_shards": len(set(shard_key) - set(our_key)), "only_ours": len(set(our_key) - set(shard_key)),
           "examples_only_in_shards": [list(k) for k in sorted(set(shard_key) - set(our_key))[:5]],
           "examples_only_ours": [list(k) for k in sorted(set(our_key) - set(shard_key))[:5]]}
    counts_equal = 0
    for k in common:
        i, j = our_key[k], shard_key[k]
        counts_equal += all(int(refs[f][i]) == int(sh[f][j]) for f in ("cluster_blue", "cluster_red", "present_blue", "present_red"))
    out["scale_counts_equal_on_common"] = counts_equal
    our_lab = {(str(refs["match_id"][r]), int(refs["engage_ts"][r])): int(v) for r, v in zip(ridx, y)}
    shard_lab = {k: int(sh["y_market_event"][j]) for k, j in shard_key.items() if int(sh["y_market_event"][j]) >= 0}
    common_lab = sorted(set(our_lab) & set(shard_lab))
    out.update({"our_labelled_rows": len(our_lab), "shard_labelled_rows": len(shard_lab),
                "common_labelled_rows": len(common_lab),
                "label_agreement_on_common": (float(np.mean([our_lab[k] == shard_lab[k] for k in common_lab])) if common_lab else None)})
    names_path = shard_dir / "feature_names.json"
    shard_names = json.loads(names_path.read_text(encoding="utf-8"))["names"] if names_path.exists() else None
    out["feature_names_equal"] = bool(shard_names is not None and data["names"] is not None and list(shard_names) == list(data["names"]))
    # X: shards holding the most sampled matches (shard index = position in the sorted cache listing mod num_shards)
    pos = {m: i for i, m in enumerate(all_mids_sorted)}
    per_shard = {}
    for m in sample:
        per_shard[pos[m] % num_shards] = per_shard.get(pos[m] % num_shards, 0) + 1
    chosen = sorted(per_shard, key=lambda s: (-per_shard[s], s))[:max(0, int(x_shards))]
    row_of = {(str(refs["match_id"][r]), int(refs["engage_ts"][r])): k for k, r in enumerate(ridx)}
    compared, equal, max_diff = 0, 0, 0.0
    for sid in chosen:
        with np.load(shard_dir / f"shard_{sid:03d}.npz", allow_pickle=False) as z:
            g, t = z["groups"], z["engage_ts"]
            wanted = [(k, j) for j, k in enumerate(zip(g.tolist(), t.tolist())) if (str(k[0]), int(k[1])) in row_of]
            if not wanted:
                continue
            Xs = z["X"]
            for k, j in wanted:
                a = data["X"][row_of[(str(k[0]), int(k[1]))]]
                b = Xs[j]
                compared += 1
                equal += bool(np.array_equal(a, b, equal_nan=True))
                diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
                if np.isfinite(diff).any():
                    max_diff = max(max_diff, float(np.nanmax(diff)))
            del Xs
    out["X"] = {"shards_compared": chosen, "rows_compared": compared, "rows_bitwise_equal": equal, "max_abs_diff": max_diff}
    return out


def run_setting(g_ms: int, d_u: float, mids: list, all_mids: list, signature: dict, args, boot_mod, git: dict) -> None:
    name = setting_name(g_ms, d_u)
    js_path, npz_path = args.out_dir / "settings" / f"{name}.json", args.out_dir / "settings" / f"{name}.npz"
    if js_path.exists() and npz_path.exists() and not args.force:
        prev = json.loads(js_path.read_text(encoding="utf-8"))
        if prev.get("complete") and prev.get("signature") == signature:
            log(f"[{name}] finished earlier; skipped")
            return
        if prev.get("complete"):
            raise SystemExit(f"[{name}] exists with a different signature; pass --force or a fresh --out-dir")
    log(f"[{name}] G={g_ms / 1000:g} s D={d_u:g} u: detecting and building rows on {len(mids):,} matches")
    t0 = time.time()
    data = build_setting(g_ms, d_u, mids, args.workers, args.chunk_size)
    refs = data["refs"]
    n_refs, n_rows = int(len(refs["match_id"])), int(len(data["y"]))
    log(f"[{name}] engagements={n_refs:,} labelled rows={n_rows:,} ({data['build_seconds']:.0f}s)")
    eff = data["effective_cfg"] or {}
    if int(eff.get("TF2_KILL_CLUSTER_GAP_MS", -1)) != int(g_ms) or float(eff.get("CLUSTER_MAX_DIAMETER", -1)) != float(d_u):
        raise SystemExit(f"[{name}] the workers did not run the requested setting: {eff}")
    # every other definition constant must be the preset's: a worker that missed LOL_CFG_PRESET would run the
    # CoG 2026 defaults (Eq.3 label, leaky time_norm / anchors) with only G and D changed
    expected = {**boot_mod.preset_values(PRESET), **worker_overrides(g_ms, d_u)}
    bad = {k: (eff.get(k), v) for k, v in expected.items() if k in eff and eff.get(k) != v}
    if eff.get("LOL_CFG_PRESET") != PRESET or eff.get("LABEL_TIE_POLICY") != "drop" or bad:
        raise SystemExit(f"[{name}] workers did not run preset {PRESET} with draws dropped: "
                         f"preset={eff.get('LOL_CFG_PRESET')} tie={eff.get('LABEL_TIE_POLICY')} {bad}")
    reproduction = None
    if args.validate_shards and (int(g_ms), float(d_u)) == OPERATING_POINT:
        reproduction = validate_corpus_reproduction(args.validate_shards, all_mids, mids, data, args.validate_x_shards, boot_mod)
        log(f"[{name}] corpus reproduction: {json.dumps(reproduction)}")
    pred, split, model_info, splits, test = fit_and_score(data, args, boot_mod)
    nmin = smaller_side(refs["cluster_blue"], refs["cluster_red"], args.negative_count_as)
    with_eng = int(len(np.unique(refs["match_id"])))
    result = {
        "item": boot_mod.ITEM, "complete": True, "signature": signature,
        "setting": {"name": name, "G_ms": int(g_ms), "G_s": g_ms / 1000.0, "D_u": float(d_u),
                    "is_operating_point": (int(g_ms), float(d_u)) == OPERATING_POINT},
        "provenance": {**git, "script": "scripts/run_gd_sweep_v33.py", "python": sys.executable, "preset": PRESET,
                       "label_key": LABEL_KEY, "split": {"kind": "patch holdout", "train": args.train_patch,
                                                         "val": args.val_patch, "test": args.test_patch},
                       "seed": int(args.seed), "n_matches_sampled": len(mids), "sample_digest": signature["sample_digest"],
                       "lol_output_root": os.environ.get("LOL_OUTPUT_ROOT"), "effective_cfg": eff,
                       "worker_overrides": worker_overrides(g_ms, d_u), "feature_names_sha1": data["names_sha1"],
                       "wall_clock_s": None},
        "counts": {"n_matches_sampled": len(mids), "n_matches_with_engagement": with_eng,
                   "n_engagements": n_refs, "engagements_per_sampled_match": n_refs / max(1, len(mids)),
                   "engagements_per_match_with_engagement": n_refs / max(1, with_eng),
                   "n_rows_labelled": n_rows, "engagements_without_label": n_refs - n_rows,
                   "positive_rate": (float(data["y"].mean()) if n_rows else None), "by_split": splits,
                   "engagement_scale_share": {k: float(m.mean()) for k, m in scale_masks(nmin, args.teamfight_min).items()}},
        "model": model_info, "test": test, "corpus_reproduction": reproduction,
        "deviations": sweep_deviations(args, len(mids)),
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **{f"ref_{k}": v for k, v in refs.items()}, row_ref_idx=data["row_ref_idx"],
                        y=data["y"], pred=pred, split=split)
    result["provenance"]["wall_clock_s"] = round(time.time() - t0, 1)
    js_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    a = test["auc"]
    log(f"[{name}] test AUC {boot_mod.fmt_ci(a)} rows={splits['test']['rows']:,} best_iter={model_info['best_iteration']} "
        f"({result['provenance']['wall_clock_s']:.0f}s)")


# ----------------------------------------------------------------------------- cross-setting table
def aggregate(args, settings: list, boot_mod, git: dict) -> None:
    t0 = time.time()
    sdir = args.out_dir / "settings"
    op_name = setting_name(*OPERATING_POINT)
    loaded = {}
    for g_ms, d_u in settings:
        name = setting_name(g_ms, d_u)
        jp, zp = sdir / f"{name}.json", sdir / f"{name}.npz"
        if jp.exists() and zp.exists():
            js = json.loads(jp.read_text(encoding="utf-8"))
            if js.get("complete"):
                with np.load(zp, allow_pickle=False) as z:
                    loaded[name] = (js, {k: z[k] for k in z.files})
    missing = [setting_name(*s) for s in settings if setting_name(*s) not in loaded]
    if op_name not in loaded:
        log(f"operating point {op_name} not finished; cross-setting table skipped")
        return
    signatures = {json.dumps(js["signature"], sort_keys=True) for js, _ in loaded.values()}
    if len(signatures) != 1:
        raise SystemExit("finished settings carry different signatures; they cannot share one table")
    b_ms = int(loaded[op_name][0]["provenance"]["effective_cfg"].get("TF2_ENGAGE_PRE_KILL_MS", 15000))
    tol = int(args.tolerance_ms)

    def unpack(z):
        refs = {k[4:]: z[k] for k in z if k.startswith("ref_")}
        t_ref = anchor_time(refs, b_ms)
        te = z["split"] == 2
        ridx = z["row_ref_idx"][te]
        return refs, t_ref, refs["match_id"][ridx], t_ref[ridx], z["y"][te].astype(np.int64), z["pred"][te]

    op_refs, op_t, op_rmid, op_rt, op_y, op_p = unpack(loaded[op_name][1])
    rows = []
    for g_ms, d_u in settings:
        name = setting_name(g_ms, d_u)
        if name not in loaded:
            continue
        js, z = loaded[name]
        s_refs, s_t, s_rmid, s_rt, s_y, s_p = unpack(z)
        cmp = paired_vs_reference(
            {"match_id": op_refs["match_id"], "t": op_t, "test_mid": op_rmid, "test_t": op_rt, "y": op_y, "p": op_p},
            {"match_id": s_refs["match_id"], "t": s_t, "test_mid": s_rmid, "test_t": s_rt, "y": s_y, "p": s_p},
            tol, args.n_boot, args.seed, boot_mod)
        rows.append({"name": name, "G_s": g_ms / 1000.0, "D_u": float(d_u), "is_operating_point": name == op_name,
                     "counts": js["counts"], "model": {k: js["model"][k] for k in ("best_iteration", "val_auc", "train_rows")},
                     "test": js["test"], "overlap_engagements": cmp["overlap"], "test_rows_vs_operating_point": cmp["shared"],
                     "auc_vs_operating_point": cmp["auc"], "paired_vs_operating_point": cmp["paired"],
                     "wall_clock_s": js["provenance"]["wall_clock_s"]})

    sig = loaded[op_name][0]["signature"]
    summary = {
        "item": boot_mod.ITEM,
        "what": "G x D sensitivity of the v3.3 result on a match subsample, patch holdout, paper LightGBM",
        "provenance": {**git, "script": "scripts/run_gd_sweep_v33.py", "preset": PRESET, "label_key": LABEL_KEY,
                       "split": {"kind": "patch holdout", "train": args.train_patch, "val": args.val_patch, "test": args.test_patch},
                       "seed": int(args.seed), "n_matches_sampled": sig["n_matches"], "sample_digest": sig["sample_digest"],
                       "n_boot": int(args.n_boot), "tolerance_ms": tol, "negative_count_as": args.negative_count_as,
                       "teamfight_min": int(args.teamfight_min), "operating_point": op_name,
                       "setting_commits": sorted({js["provenance"].get("git_commit") for js, _ in loaded.values()}),
                       "settings_wall_clock_s": round(sum(js["provenance"]["wall_clock_s"] for js, _ in loaded.values()), 1),
                       "aggregation_wall_clock_s": None},
        "signature": sig, "settings_requested": [setting_name(*s) for s in settings], "settings_missing": missing,
        "settings": rows,
        "deviations": sweep_deviations(args, sig["n_matches"]) + [
            "Per-setting intervals and the paired differences come from separate replicate streams with the same seed; "
            "the paired stream resamples the union of both settings' test matches.",
        ],
    }
    summary["provenance"]["aggregation_wall_clock_s"] = round(time.time() - t0, 1)
    out = args.out_dir / "gd_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(args.out_dir / "gd_sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "G_s", "D_u", "engagements", "per_sampled_match", "rows", "positive_rate", "share_op_matched",
                    "share_setting_matched", "jaccard", "auc_test", "auc_lo", "auc_hi", "d_full", "d_full_lo", "d_full_hi",
                    "n_shared", "label_agreement_shared", "auc_shared_setting", "auc_shared_op", "d_shared", "d_shared_lo",
                    "d_shared_hi", "auc_pick", "auc_skirmish", "auc_teamfight", "pick_minus_teamfight"])
        for r in rows:
            c, t, o, p, s, a = r["counts"], r["test"], r["overlap_engagements"], r["paired_vs_operating_point"], \
                r["test_rows_vs_operating_point"], r["auc_vs_operating_point"]
            w.writerow([r["name"], r["G_s"], r["D_u"], c["n_engagements"], round(c["engagements_per_sampled_match"], 4),
                        c["n_rows_labelled"], c["positive_rate"], round(o["share_of_operating_point_matched"], 4),
                        round(o["share_of_setting_matched"], 4), round(o["jaccard"], 4), t["auc"].get("point"),
                        t["auc"].get("ci_2.5"), t["auc"].get("ci_97.5"), p["full_setting_minus_op"].get("point"),
                        p["full_setting_minus_op"].get("ci_2.5"), p["full_setting_minus_op"].get("ci_97.5"), s["n_shared"],
                        s["label_agreement_shared"], a["setting_shared"].get("point"), a["operating_point_shared"].get("point"),
                        p["shared_setting_minus_op"].get("point"), p["shared_setting_minus_op"].get("ci_2.5"),
                        p["shared_setting_minus_op"].get("ci_97.5"), t["classes"]["pick"]["auc"].get("point"),
                        t["classes"]["skirmish"]["auc"].get("point"), t["classes"]["teamfight"]["auc"].get("point"),
                        t["pick_minus_teamfight"].get("point")])
    log("\n| G (s) | D (u) | engagements | per match | pos rate | op matched | Jaccard | test AUC | vs op (full) | shared n | vs op (shared) | pick | skirmish | teamfight |")
    log("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c, t, o, p = r["counts"], r["test"], r["overlap_engagements"], r["paired_vs_operating_point"]
        f = boot_mod.fmt_ci
        cls = t["classes"]
        log(f"| {r['G_s']:g} | {r['D_u']:g} | {c['n_engagements']:,} | {c['engagements_per_sampled_match']:.3f} | "
            f"{(c['positive_rate'] or float('nan')):.3f} | {o['share_of_operating_point_matched']:.3f} | {o['jaccard']:.3f} | "
            f"{f(t['auc'])} | {f(p['full_setting_minus_op'], True)} | {r['test_rows_vs_operating_point']['n_shared']:,} | "
            f"{f(p['shared_setting_minus_op'], True)} | {f(cls['pick']['auc'])} | {f(cls['skirmish']['auc'])} | {f(cls['teamfight']['auc'])} |")
    log(f"wrote {out} (missing: {missing})")


def main(argv=None) -> int:
    global _LOG_PATH
    boot_mod = load_sibling("scale_cut_sensitivity_v33")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g-values", default=G_VALUES_S, help="kill-gap values in seconds")
    ap.add_argument("--d-values", default=D_VALUES_U, help="cluster-diameter values in game units")
    ap.add_argument("--settings", default=None, help="explicit G_s:D_u list instead of the grid, e.g. 13.7:4264,18:4000")
    ap.add_argument("--n-matches", type=int, default=10000, help="declared cache-match subsample (0 = every cached match)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-patch", default="15.14")
    ap.add_argument("--val-patch", default="15.15")
    ap.add_argument("--test-patch", default="15.16")
    ap.add_argument("--workers", type=int, default=4, help="row-building worker processes")
    ap.add_argument("--chunk-size", type=int, default=40, help="matches per worker task")
    ap.add_argument("--n-jobs", type=int, default=4, help="LightGBM threads")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tolerance-ms", type=int, default=MATCH_TOLERANCE_MS)
    ap.add_argument("--teamfight-min", type=int, default=4)
    ap.add_argument("--negative-count-as", choices=("zero", "unknown"), default="zero")
    ap.add_argument("--out-dir", type=Path, default=boot_mod.DEFAULT_OUT_DIR / "gd_sweep")
    ap.add_argument("--validate-shards", type=Path, default=None,
                    help="corpus_shards_v33: check the operating point reproduces the corpus on the sampled matches")
    ap.add_argument("--validate-x-shards", type=int, default=1, help="shards whose X rows are compared (each ~0.5 GB)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute settings that already finished")
    args = ap.parse_args(argv)

    inherited_preset = os.environ.get("LOL_CFG_PRESET", "").strip()
    if inherited_preset and inherited_preset != PRESET:
        raise SystemExit(f"LOL_CFG_PRESET={inherited_preset!r}; this sweep is defined on preset {PRESET}")
    if os.environ.get("LOL_CFG_OVERRIDES", "").strip():
        raise SystemExit("unset LOL_CFG_OVERRIDES: the sweep passes the detector constants to its workers itself")
    os.environ["LOL_CFG_PRESET"] = PRESET
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = args.out_dir / "gd_sweep.log"
    settings = parse_settings(args)
    git = boot_mod.git_provenance()

    from core.config import CACHE_DIR
    all_mids = sorted(p.name[: -len(".meta.json")] for p in CACHE_DIR.glob("*.meta.json"))
    mids = list(all_mids)
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))
    digest = hashlib.blake2b("\n".join(mids).encode("utf-8"), digest_size=12).hexdigest()
    signature = {"preset": PRESET, "cache_dir": str(CACHE_DIR), "n_matches": len(mids), "seed": int(args.seed),
                 "sample_digest": digest, "patches": [args.train_patch, args.val_patch, args.test_patch],
                 "lgbm": PAPER_LGBM, "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "feature_set": "full",
                 "label": "market_event/drop", "teamfight_min": int(args.teamfight_min),
                 "negative_count_as": args.negative_count_as, "n_boot": int(args.n_boot)}
    log(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} git={git['git_commit'][:10]} preset={PRESET} matches={len(mids):,}/"
        f"{len(all_mids):,} digest={digest} settings={[setting_name(*s) for s in settings]}")
    if not args.aggregate_only:
        for g_ms, d_u in settings:
            run_setting(g_ms, d_u, mids, all_mids, signature, args, boot_mod, git)
    aggregate(args, settings, boot_mod, git)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
