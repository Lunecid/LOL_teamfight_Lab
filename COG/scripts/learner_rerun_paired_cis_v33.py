"""Paired intervals for the re-run learners, sequence-learner gain differences and the published
SAINT evaluation batches (v3.3, ToG revision item C5-archive-learner-cis).

sec_learners.tex prints three sets of numbers that no committed artefact stores.  This script
recomputes all three from frozen prediction files and writes them, with the git commit and the
SHA-1 of every input, to

    D:/LOL_Project/fusion_2615/features/tog_revision/C5-learner-cis/learner_rerun_paired_cis_v33.json

It reads predictions only: nothing is trained, selected or calibrated.  It is safe to re-run at any
time; runs that have not finished are listed under ``pending`` and join every block (intervals,
comparisons, the answer block) on the next run once their files exist.

(1) learner_cis -- re-run learners on the A8 bootstrap draws
    The first comparison's paired intervals (REV/A8-input-audit-and-cis/paired_learner_cis_v33.json,
    "A8") come from 2,000 match-bootstrap draws, seed 7, of the 55,449 test matches of patch 15.16.
    The re-runs (REV/A1-deep-baselines/<stem>.json) carry their own 1,000-draw bootstrap, which
    cannot be paired with A8's intervals.  This block scores every learner -- the nine of A8 and every
    finished re-run or searched refit -- on A8's own draws, so every row of the learner table shares
    one set of resamples.  The draw generator, match index and weighted AUC are those of the worktree
    script that wrote A8 (scripts/paired_learner_cis_v33.py, script_sha1 5400ecb0; prepare(),
    weighted_auc(), bootstrap()): uniq, inv = np.unique(groups, return_inverse=True); rng =
    default_rng(7); per replicate w = bincount(rng.integers(G, size=G), minlength=G)[inv].  The draws
    do not depend on which learners are scored, so adding a learner never moves another's interval.
    Checks: every A8 learners.*.test_auc / ci95 and comparisons.*.diff / ci95 / boot_mean / boot_se /
    p_two_sided / p_holm reproduced (the run aborts on a gap above 1e-9); the tree prediction file's
    SHA-1 equals A8's provenance; every re-run file has y and match ids equal to the tree file element by
    element (and min_participants_test equal to the first comparison's, where the file stores it); the
    per-model file <stem>.<model>.preds.npz holds the same rows and predictions; the point AUC equals
    the run JSON's test_auc and sklearn's roc_auc_score.

(2) sequence_gain_differences -- sequence learners' window gain minus LightGBM's
    (AUC_m,w - AUC_m,ref) - (AUC_lgbm,w - AUC_lgbm,ref) on the draws of
    REV/A3-temporal-windows/sequence_baseline/sequence_baseline_v33.json ("SEQ"), whose bootstrap
    stores each AUC and each paired difference but not this difference of differences.  The draw
    loop is run_window_sweep_v33.match_bootstrap (imported from scripts/, not copied): match_weights,
    _auc_plan and _auc_from_plan, default_rng(7), a replicate skipped when any AUC is undefined.
    Check: every SEQ bootstrap.auc and bootstrap.paired entry reproduced (abort above 1e-9).  Reported
    at SEQ's resample count (1,000) and at 2,000 (the first 1,000 draws are shared).  Every setting
    and seed present in the prediction file enters, so further sequence arms are picked up.

(3) published_saint_batches -- which evaluation batches of the withdrawn SAINT run held one match twice
    The published SAINT run (FEAT/deep_tabular_v33_patch_saint.preds.npz) scored validation and test
    in consecutive chunks of 1,024 rows in load order: evaluate() of scripts/run_deep_tabular_baselines.py,
    "for i in range(0, len(y_eval), 1024)", identical at 314ee78 and 60945ed (the committed versions
    either side of the run, A8 caveats[0]); the patch split, "tr, va, te = patch == train, ...", boolean
    masks that keep load order, exists only from 60945ed on (314ee78 splits by match), and
    groups_test = groups[te] is saved in that same order.  The block counts the test chunks that
    contain two or more rows of one match and the rows that share their chunk with a row of their own
    match (validation match ids were not saved).  The git source of both commits is searched for those
    three lines and the result recorded.  For contrast, the corrected runs' test batches are rebuilt
    with the current match_disjoint_batches() at each run's eval_batch_size and seed + 1, the generator
    evaluate() uses, and must share no match.

Run (light: CPU only, 4 threads, well under 1 GB):

    python scripts/learner_rerun_paired_cis_v33.py

References: Efron & Tibshirani (1993) ch. 13 (percentile interval); Davison & Hinkley (1997) s. 3.8
(resampling clusters); Holm (1979) step-down adjustment.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ITEM = "C5-archive-learner-cis"
SEED = 7
TOL = 1e-9
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = Path("D:/LOL_Project/fusion_2615/features")
REV = FEATURES / "tog_revision"
A1 = REV / "A1-deep-baselines"
DEFAULT_A8 = REV / "A8-input-audit-and-cis" / "paired_learner_cis_v33.json"
DEFAULT_SEQ = REV / "A3-temporal-windows" / "sequence_baseline" / "sequence_baseline_v33.json"
DEFAULT_PUBLISHED_SAINT = FEATURES / "deep_tabular_v33_patch_saint.preds.npz"
DEFAULT_QUEUE_STATE = REV / "queue_state"
DEFAULT_SEARCH_DIR = A1 / "hparam_search"
DEFAULT_OUT = REV / "C5-learner-cis" / "learner_rerun_paired_cis_v33.json"

# first-comparison deep prediction keys -> A8 learner names (worktree paired_learner_cis_v33.DEEP_NAMES)
DEEP_NAMES = {"lightgbm": "lgbm_7106", "mlp": "mlp", "tabnet": "tabnet",
              "ft_transformer": "ft_transformer", "saint": "saint"}
TREE_LEARNERS = {"lead_only", "lgbm_paper", "lgbm_deep", "lgbm_7106", "search_lightgbm"}
LINEAR_LEARNERS = {"linear"}
WITHDRAWN = {
    "tabnet": "published TabNet subtracted its sparsity term (A8 caveats[0]); replaced by tabnet_fixed",
    "saint": "published SAINT evaluated in load-order batches of 1,024 rows that shared matches "
             "(published_saint_batches); replaced by saint_c0",
}
# Re-runs of REV/queue_wave5.json: (learner name, output stem under A1, model key, queue job, prior learner)
RERUNS = (
    ("tabnet_fixed", "a_tabnet_sign_fixed_published_config", "tabnet",
     "a_tabnet_sign_fixed_published_config", "tabnet"),
    ("ft_d192", "b_ft_transformer_d192_published_budget", "ft_transformer",
     "b1_ft_transformer_d192_published_budget", "ft_transformer"),
    ("saint_c0", "c0_saint_d32_published_config_match_disjoint", "saint",
     "c0_saint_d32_published_config_match_disjoint", "saint"),
    ("saint_d128", "b_saint_d128_published_budget", "saint",
     "b2_saint_d128_published_budget", "saint_c0"),
    ("saint_pretrain5", "c_saint_d32_pretrain5_published_config", "saint",
     "c_saint_d32_pretrain5_published_config", "saint_c0"),
)
# Declared search (scripts/run_deep_hparam_search_v33.py): refit/<family>.json + refit/<family>.preds.npz
SEARCH_FAMILIES = ("lightgbm", "mlp", "tabnet", "ft_transformer", "saint", "saint_pretrain")
SEARCH_PRIOR = {"lightgbm": "lgbm_paper", "mlp": "mlp", "tabnet": "tabnet_fixed",
                "ft_transformer": "ft_d192", "saint": "saint_c0", "saint_pretrain": "saint_c0"}
# every re-run / searched learner is also compared with these first-comparison learners
REFERENCES = ("lgbm_paper", "lead_only", "linear", "lgbm_deep", "lgbm_7106", "mlp", "ft_transformer")
PUBLISHED_CHUNK = 1024
PUBLISHED_CODE_LINES = ("for i in range(0, len(y_eval), 1024):",
                        "tr, va, te = patch == train, patch == val, patch == test",
                        "y_test=y[te], groups_test=groups[te],")
PUBLISHED_COMMITS = ("314ee78", "60945ed")


# ----------------------------------------------------------------------------- provenance
def sha1_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Inputs:
    """Every file the run reads, with SHA-1, size and modification time."""

    def __init__(self):
        self.files: dict[str, dict] = {}

    def add(self, path: Path) -> Path:
        path = Path(path)
        key = path.as_posix()
        if key not in self.files:
            st = path.stat()
            self.files[key] = {"sha1": sha1_of(path), "bytes": st.st_size,
                               "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}
        return path

    def json(self, path: Path) -> dict:
        return json.loads(self.add(path).read_text(encoding="utf-8"))

    def npz(self, path: Path) -> dict:
        with np.load(self.add(path), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60)
        return r.stdout if r.returncode == 0 else None
    except Exception:  # provenance must never abort the run
        return None


def git_state() -> dict:
    rel = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    # porcelain lines are 'XY path'; the first status column may be a space, so lines are never stripped
    porcelain = (git("status", "--porcelain") or "").splitlines()
    tracked_dirty = [ln[3:] for ln in porcelain if ln and not ln.startswith("??")]
    own = [ln[:2] for ln in porcelain if ln[3:] == rel]
    return {"commit": (git("rev-parse", "HEAD") or "").strip() or None,
            "branch": (git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
            "dirty_tracked_files": tracked_dirty,
            "script_path": rel,
            "script_status_vs_head": ("untracked" if own and own[0] == "??" else "modified" if own else "committed"),
            "note": "script_sha1 identifies the code that ran; when script_status_vs_head is not 'committed', "
                    "match it against the SHA-1 of the file in a later commit"}


def load_module(name: str):
    """A sibling script imported by path (scripts/ is not a package); its SHA-1 goes to the inputs."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- A8 AUC machinery
def prepare(score, y):
    """Copied from worktree scripts/paired_learner_cis_v33.py (script_sha1 5400ecb0): sort once per learner."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return order, starts, y[order] == 1


def weighted_auc(prep, w):
    """Copied from the same script: Mann-Whitney AUC over weighted rows, ties scored 1/2."""
    order, starts, pos = prep
    ws = w[order]
    gp = np.add.reduceat(np.where(pos, ws, 0.0), starts)
    gn = np.add.reduceat(np.where(pos, 0.0, ws), starts)
    denom = gp.sum() * gn.sum()
    if denom == 0:
        return float("nan")
    return float((gp * (np.cumsum(gn) - 0.5 * gn)).sum() / denom)


def interval(d: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


def mc_p_two_sided(d: np.ndarray) -> float:
    """A8's Monte Carlo p: 2 (1 + #{replicates on the far side of 0}) / (R + 1), capped at 1."""
    far = min(int((d <= 0).sum()), int((d >= 0).sum()))
    return min(1.0, 2 * (1 + far) / (len(d) + 1))


def holm(pvals: dict) -> dict:
    """Holm (1979) step-down adjusted p over the keys of ``pvals`` (A8's running-max form)."""
    ordered = sorted(pvals, key=lambda k: pvals[k])
    out, running, m = {}, 0.0, len(ordered)
    for i, k in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def kind_of(name: str) -> str:
    return "tree" if name in TREE_LEARNERS else "linear" if name in LINEAR_LEARNERS else "neural"


# ----------------------------------------------------------------------------- (1) learner_cis
def load_first_comparison(inp: Inputs, a8: dict, log):
    files = {Path(p).as_posix(): v for p, v in a8["provenance"]["input_files"].items()}
    tree_preds = next(Path(p) for p in files if p.endswith("model_comparison_v33_patch.preds.npz"))
    deep_preds = [Path(p) for p in files if p.endswith(".preds.npz") and "deep_tabular_v33_patch_" in p]
    learners, row_identity = {}, {}
    for p, meta in files.items():
        inp.add(Path(p))
        got = inp.files[Path(p).as_posix()]["sha1"]
        if got != meta["sha1"]:
            raise SystemExit(f"{p}: SHA-1 {got} differs from A8 provenance {meta['sha1']}; not A8's inputs")
    tree_json = inp.json(Path(str(tree_preds).replace(".preds.npz", ".json")))
    z = inp.npz(tree_preds)
    y, groups = z["y"].astype(np.int64), z["groups"]
    for k in ("lead_only", "linear", "lgbm_paper", "lgbm_deep"):
        learners[k] = {"pred": z[k].astype(np.float64), "file": tree_preds.as_posix(), "model_key": k,
                       "run_json": str(tree_preds).replace(".preds.npz", ".json").replace("\\", "/"),
                       "json_test_auc": tree_json["models"][k]["overall_auc"],
                       "val_auc": tree_json["models"][k].get("val_auc"),
                       "status": "first comparison"}
    mp_ref = None
    for pp in sorted(deep_preds):
        d = inp.npz(pp)
        run = inp.json(Path(str(pp).replace(".preds.npz", ".json")))
        same_y = bool(np.array_equal(d["y_test"].astype(np.int64), y))
        same_g = bool(np.array_equal(d["groups_test"], groups))
        if not (same_y and same_g):
            raise SystemExit(f"{pp}: y_test / groups_test differ from {tree_preds}; rows are not paired")
        if mp_ref is None and "full" in pp.name:
            mp_ref = d["min_participants_test"]
        row_identity[pp.as_posix()] = {"y_equal_tree_file": same_y, "groups_equal_tree_file": same_g}
        for key in d:
            if not key.startswith("pred_"):
                continue
            base = key[len("pred_"):]
            name = DEEP_NAMES.get(base, base)
            learners[name] = {"pred": d[key].astype(np.float64), "file": pp.as_posix(), "model_key": key,
                              "run_json": str(pp).replace(".preds.npz", ".json").replace("\\", "/"),
                              "json_test_auc": run["models"][base]["test_auc"],
                              "val_auc": run["models"][base].get("val_auc"),
                              "status": "withdrawn" if name in WITHDRAWN else "first comparison"}
    for pp in sorted(deep_preds):  # min_participants_test of every first-comparison deep file
        d = inp.npz(pp)
        row_identity[pp.as_posix()]["min_participants_equal_full_file"] = bool(
            np.array_equal(d["min_participants_test"], mp_ref))
        if not row_identity[pp.as_posix()]["min_participants_equal_full_file"]:
            raise SystemExit(f"{pp}: min_participants_test differs from the full deep file")
    log(f"first comparison: {len(learners)} learners, {len(y):,} test rows")
    return y, groups, mp_ref, learners, row_identity, tree_preds


def sentinel_state(queue_state: Path, job: str) -> str:
    for suffix in ("ok", "running", "failed"):
        if (queue_state / f"{job}.{suffix}").exists():
            return suffix
    return "no entry"


def load_reruns(inp: Inputs, y, groups, mp_ref, learners, row_identity, args, log):
    available, pending = [], []
    for name, stem, model, job, prior in RERUNS:
        a1 = args.a1_dir
        run_json, preds, per_model = a1 / f"{stem}.json", a1 / f"{stem}.preds.npz", a1 / f"{stem}.{model}.preds.npz"
        state = sentinel_state(args.queue_state, job)
        ready = run_json.exists() and preds.exists() and (state == "ok" or args.ignore_queue_state)
        if not ready:
            pending.append({"learner": name, "queue_job": job, "queue_state": state,
                            "expects": [run_json.as_posix(), preds.as_posix(), per_model.as_posix()]})
            continue
        run = inp.json(run_json)
        d = inp.npz(preds)
        checks = {"y_equal_tree_file": bool(np.array_equal(d["y_test"].astype(np.int64), y)),
                  "groups_equal_tree_file": bool(np.array_equal(d["groups_test"], groups)),
                  "min_participants_equal_first_comparison": bool(
                      np.array_equal(d["min_participants_test"], mp_ref))}
        pred = d[f"pred_{model}"]
        if per_model.exists():
            s = inp.npz(per_model)
            checks.update({"per_model_file_y_equal": bool(np.array_equal(s["y"].astype(np.int64), y)),
                           "per_model_file_groups_equal": bool(np.array_equal(s["groups"], groups)),
                           "per_model_file_pred_equal": bool(np.array_equal(s["pred"], pred))})
        if not all(checks.values()):
            raise SystemExit(f"{name}: row identity failed {checks}")
        row_identity[preds.as_posix()] = checks
        m = run["models"][model]
        learners[name] = {"pred": pred.astype(np.float64), "file": preds.as_posix(), "model_key": f"pred_{model}",
                          "run_json": run_json.as_posix(), "json_test_auc": m["test_auc"],
                          "val_auc": m.get("val_auc"), "n_params": m.get("n_params"),
                          "best_epoch": m.get("best_epoch"), "epochs_run": m.get("epochs_run"),
                          "config": m.get("config"), "training_options": run.get("training_options"),
                          "run_git_commit": run.get("git_commit"), "run_git_dirty": run.get("git_dirty"),
                          "queue_job": job, "queue_state": state, "prior": prior, "status": "re-run"}
        available.append(name)
        log(f"re-run {name}: rows paired ({preds.name})")
    search_dir = args.search_dir
    for family in SEARCH_FAMILIES:
        name = f"search_{family}"
        rj, rp = search_dir / "refit" / f"{family}.json", search_dir / "refit" / f"{family}.preds.npz"
        refit = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else None
        if not (refit and refit.get("status") == "done" and rp.exists()):
            pending.append({"learner": name, "refit_status": (refit or {}).get("status", "absent"),
                            "expects": [rj.as_posix(), rp.as_posix()]})
            continue
        inp.add(rj)
        d = inp.npz(rp)
        checks = {"y_equal_tree_file": bool(np.array_equal(d["y"].astype(np.int64), y)),
                  "groups_equal_tree_file": bool(np.array_equal(d["groups"], groups))}
        if not all(checks.values()):
            raise SystemExit(f"{name}: row identity failed {checks}")
        row_identity[rp.as_posix()] = checks
        learners[name] = {"pred": d["pred"].astype(np.float64), "file": rp.as_posix(), "model_key": "pred",
                          "run_json": rj.as_posix(), "json_test_auc": refit.get("test_auc"),
                          "val_auc": refit.get("val_auc"), "n_params": refit.get("n_params"),
                          "best_epoch": refit.get("best_epoch"), "best_iteration": refit.get("best_iteration"),
                          "config": refit.get("config"), "selected_point": refit.get("selected_point"),
                          "search_val_auc": refit.get("search_val_auc"),
                          "run_git_commit": refit.get("git_commit"), "run_git_dirty": refit.get("git_dirty"),
                          "prior": SEARCH_PRIOR[family], "status": "searched"}
        available.append(name)
        log(f"searched refit {name}: rows paired ({rp.name})")
    return available, pending


def learner_cis(inp: Inputs, args, log) -> dict:
    started = time.time()
    a8 = inp.json(args.a8)
    prov = a8["provenance"]
    if (prov["n_boot"], prov["seed"]) != (args.n_boot, args.seed):
        raise SystemExit(f"--n-boot/--seed {args.n_boot}/{args.seed} differ from A8's {prov['n_boot']}/{prov['seed']}: "
                         "this block exists to score on A8's own draws")
    y, groups, mp_ref, learners, row_identity, tree_preds = load_first_comparison(inp, a8, log)
    available, pending = load_reruns(inp, y, groups, mp_ref, learners, row_identity, args, log)
    names = list(learners)
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    prep = {k: prepare(learners[k]["pred"], y) for k in names}
    point = {k: weighted_auc(prep[k], np.ones(len(y))) for k in names}
    gap_sklearn = max(abs(point[k] - roc_auc_score(y, learners[k]["pred"])) for k in names)
    gap_json = max(abs(point[k] - learners[k]["json_test_auc"]) for k in names
                   if learners[k]["json_test_auc"] is not None)
    if gap_sklearn > TOL or gap_json > TOL:
        raise SystemExit(f"point AUCs do not reproduce: sklearn gap {gap_sklearn}, run-JSON gap {gap_json}")
    log(f"scoring {len(names)} learners on {args.n_boot} draws of {G:,} matches (seed {args.seed}) ...")
    rng = np.random.default_rng(args.seed)
    B = np.empty((args.n_boot, len(names)))
    for b in range(args.n_boot):
        w = np.bincount(rng.integers(G, size=G), minlength=G).astype(np.float64)[inv]
        for j, k in enumerate(names):
            B[b, j] = weighted_auc(prep[k], w)
        if (b + 1) % 500 == 0:
            log(f"  replicate {b + 1}/{args.n_boot} ({time.time() - started:.0f}s)")
    if np.isnan(B).any():
        raise SystemExit("an undefined AUC in a replicate; A8 had none")
    col = {k: j for j, k in enumerate(names)}

    def diff_entry(a: str, b: str) -> dict:
        d = B[:, col[a]] - B[:, col[b]]
        lo, hi = interval(d)
        return {"a": a, "b": b, "difference": "AUC(a) - AUC(b)", "diff": point[a] - point[b],
                "ci95": [lo, hi], "boot_mean": float(d.mean()), "boot_se": float(d.std(ddof=1)),
                "excludes_zero": bool(lo > 0 or hi < 0), "p_two_sided_mc": mc_p_two_sided(d)}

    # reproduce A8 in full
    gaps = {"test_auc": 0.0, "learner_ci95": 0.0, "diff": 0.0, "comparison_ci95": 0.0, "boot_mean": 0.0,
            "boot_se": 0.0, "p_two_sided": 0.0, "p_holm": 0.0}
    for k, v in a8["learners"].items():
        gaps["test_auc"] = max(gaps["test_auc"], abs(point[k] - v["test_auc"]))
        gaps["learner_ci95"] = max(gaps["learner_ci95"], *(abs(p - q) for p, q in zip(interval(B[:, col[k]]), v["ci95"])))
    pv, pairs_of = {}, {}
    for key, v in a8["comparisons"].items():
        e = diff_entry(v["b"], v["a"])  # A8 writes a->b as AUC(b) - AUC(a)
        gaps["diff"] = max(gaps["diff"], abs(e["diff"] - v["diff"]))
        gaps["comparison_ci95"] = max(gaps["comparison_ci95"], *(abs(p - q) for p, q in zip(e["ci95"], v["ci95"])))
        gaps["boot_mean"] = max(gaps["boot_mean"], abs(e["boot_mean"] - v["boot_mean"]))
        gaps["boot_se"] = max(gaps["boot_se"], abs(e["boot_se"] - v["boot_se"]))
        gaps["p_two_sided"] = max(gaps["p_two_sided"], abs(e["p_two_sided_mc"] - v["p_two_sided"]))
        pair = frozenset((v["a"], v["b"]))
        pv.setdefault(pair, e["p_two_sided_mc"])
        pairs_of[key] = pair
    adj = holm(pv)
    for key, v in a8["comparisons"].items():
        gaps["p_holm"] = max(gaps["p_holm"], abs(adj[pairs_of[key]] - v["p_holm"]))
    reproduces = {"n_learners": len(a8["learners"]), "n_comparisons": len(a8["comparisons"]),
                  "max_gap": gaps, "tolerance": TOL, "passed": all(g <= TOL for g in gaps.values())}
    if not reproduces["passed"]:
        raise SystemExit(f"A8 not reproduced: {gaps}")
    log(f"A8 reproduced: max gap {max(gaps.values()):.1e}")

    out_learners = {}
    for k in sorted(names, key=lambda k: -point[k]):
        info = {kk: vv for kk, vv in learners[k].items() if kk != "pred"}
        out_learners[k] = {"test_auc": point[k], "ci95": interval(B[:, col[k]]),
                           "boot_mean": float(B[:, col[k]].mean()), "boot_se": float(B[:, col[k]].std(ddof=1)),
                           "kind": kind_of(k), **({"withdrawn": WITHDRAWN[k]} if k in WITHDRAWN else {}), **info}

    comparisons = {}
    new = [k for k in available]
    for k in new:
        refs = list(REFERENCES)
        prior = learners[k].get("prior")
        if prior in col and prior not in refs:
            refs.append(prior)
        for r in refs:
            if r in col and r != k:
                comparisons[f"{k} - {r}"] = {**diff_entry(k, r), "family": "re-run or searched learner vs first comparison"
                                             if r != prior else "re-run or searched learner vs the run it corrects or extends"}
    for i, a in enumerate(new):
        for b in new[i + 1:]:
            comparisons[f"{a} - {b}"] = {**diff_entry(a, b), "family": "between re-run and searched learners"}

    # the answer block the manuscript defers: every valid learner vs the tree model with the highest validation AUC
    valid = [k for k in names if k not in WITHDRAWN]
    trees = [k for k in valid if kind_of(k) == "tree" and learners[k].get("val_auc") is not None]
    best_tree = max(trees, key=lambda k: learners[k]["val_auc"])
    family = {k: diff_entry(k, best_tree) for k in valid if k != best_tree}
    adj = holm({k: e["p_two_sided_mc"] for k, e in family.items()})
    for k, e in family.items():
        e.update(p_holm=adj[k], kind=kind_of(k), val_auc=learners[k].get("val_auc"), status=learners[k]["status"])
    answer = {
        "complete": not pending,
        "note": ("provisional while 'pending' is non-empty: the reference tree and the Holm family change when "
                 "pending learners are added" if pending else "every registered learner is present"),
        "reference_rule": "tree learner (LightGBM of any configuration) with the highest validation AUC (patch 15.15)",
        "reference": best_tree, "reference_val_auc": learners[best_tree]["val_auc"],
        "tree_val_aucs": {k: learners[k]["val_auc"] for k in trees},
        "family": "every learner of the table except the reference and the withdrawn rows; Holm (1979) across it",
        "p_floor_unadjusted": 2 / (args.n_boot + 1),
        "comparisons": dict(sorted(family.items(), key=lambda kv: kv[1]["diff"], reverse=True)),
    }
    return {
        "method": "match-clustered paired percentile bootstrap on A8's draws (Efron & Tibshirani 1993 ch. 13; "
                  "Davison & Hinkley 1997 s. 3.8); Monte Carlo two-sided p = 2 (1 + far) / (R + 1); Holm (1979)",
        "draws": {"source": args.a8.as_posix(), "n_boot": args.n_boot, "seed": args.seed,
                  "generator": "numpy default_rng(seed); replicate: bincount(rng.integers(G, size=G), minlength=G)[inv], "
                               "uniq, inv = np.unique(groups, return_inverse=True)",
                  "n_matches": int(G), "n_rows": int(len(y)), "tree_prediction_file": tree_preds.as_posix()},
        "intervals_cover": "test-set sampling of matches only; every learner is one fitted model (one training seed)",
        "point_auc_checks": {"max_gap_vs_sklearn_roc_auc_score": gap_sklearn, "max_gap_vs_run_json_test_auc": gap_json},
        "row_identity": row_identity,
        "reproduces_a8": reproduces,
        "available_reruns": available,
        "pending": pending,
        "learners": out_learners,
        "comparisons": comparisons,
        "vs_best_validation_tree": answer,
        "wall_clock_s": round(time.time() - started, 1),
    }


# ----------------------------------------------------------------------------- (2) sequence gains
def sequence_gain_differences(inp: Inputs, args, log) -> dict:
    started = time.time()
    SW = load_module("run_window_sweep_v33")
    inp.add(Path(SW.__file__))
    seq = inp.json(args.seq)
    preds_path = Path(str(args.seq).replace(".json", ".preds.npz"))
    z = inp.npz(preds_path)
    boot = seq["bootstrap"]
    y = np.asarray(z["y"]).astype(np.int8)
    groups = z["match_id"]
    names = [k[len("test__"):] for k in z if k.startswith("test__")]
    preds = {k: np.asarray(z["test__" + k], dtype=np.float64) for k in names}
    row_checks = {"n_rows_equal_seq_bootstrap": len(y) == boot["n_rows"],
                  "n_test_rows_equal_seq_rows": len(y) == seq["rows"]["test_rows"],
                  "every_prediction_has_n_rows": all(len(p) == len(y) for p in preds.values()),
                  "learners_equal_seq_bootstrap_auc_keys": sorted(names) == sorted(boot["auc"])}
    # the same rows, labels and matches for every learner are guaranteed by one preds file; the run's own
    # tabular consistency check is recorded for completeness
    row_checks["seq_tabular_consistency_rows_bit_identical"] = {
        s: v.get("tabular_consistency", {}).get("rows_bit_identical") for s, v in seq["settings"].items()}
    if not all(v for k, v in row_checks.items() if isinstance(v, bool)):
        raise SystemExit(f"sequence rows: {row_checks}")
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    inverse = inverse.reshape(-1)
    n_groups = int(inverse.max()) + 1
    if n_groups != boot["n_matches"]:
        raise SystemExit(f"{n_groups} matches, SEQ bootstrap has {boot['n_matches']}")
    plans = {k: SW._auc_plan(y, preds[k]) for k in names}
    point = {k: SW._auc_from_plan(plans[k], None) for k in names}
    n_max = max(args.seq_n_boot, int(boot["n_boot_requested"]))
    rng = np.random.default_rng(int(boot["seed"]))
    draws = np.empty((n_max, len(names)))
    for b in range(n_max):
        w = SW.match_weights(rng, inverse, n_groups)
        draws[b] = [SW._auc_from_plan(plans[k], w) for k in names]
    col = {k: j for j, k in enumerate(names)}

    def usable_upto(n):  # match_bootstrap skips a replicate in which any AUC is undefined
        rows = draws[:n]
        return rows[~np.isnan(rows).any(axis=1)]

    # reproduce SEQ's own bootstrap
    B = usable_upto(int(boot["n_boot_requested"]))
    gaps = {"auc": 0.0, "auc_ci": 0.0, "auc_boot_mean": 0.0, "paired_delta": 0.0, "paired_ci": 0.0,
            "paired_boot_mean": 0.0, "p_delta_leq_0": 0.0, "n_boot": 0.0}
    for k, v in boot["auc"].items():
        arr = B[:, col[k]]
        lo, hi = interval(arr)
        gaps["auc"] = max(gaps["auc"], abs(point[k] - v["auc"]))
        gaps["auc_ci"] = max(gaps["auc_ci"], abs(lo - v["ci_lo"]), abs(hi - v["ci_hi"]))
        gaps["auc_boot_mean"] = max(gaps["auc_boot_mean"], abs(arr.mean() - v["boot_mean"]))
        gaps["n_boot"] = max(gaps["n_boot"], abs(len(arr) - v["n_boot"]))
    for key, v in boot["paired"].items():
        a, b = key.split(" - ")
        d = B[:, col[a]] - B[:, col[b]]
        lo, hi = interval(d)
        gaps["paired_delta"] = max(gaps["paired_delta"], abs(point[a] - point[b] - v["delta"]))
        gaps["paired_ci"] = max(gaps["paired_ci"], abs(lo - v["ci_lo"]), abs(hi - v["ci_hi"]))
        gaps["paired_boot_mean"] = max(gaps["paired_boot_mean"], abs(d.mean() - v["boot_mean"]))
        gaps["p_delta_leq_0"] = max(gaps["p_delta_leq_0"], abs(float((d <= 0).mean()) - v["p_delta_leq_0"]))
    reproduces = {"n_auc": len(boot["auc"]), "n_paired": len(boot["paired"]), "max_gap": gaps, "tolerance": TOL,
                  "passed": all(g <= TOL for g in gaps.values())}
    if not reproduces["passed"]:
        raise SystemExit(f"SEQ bootstrap not reproduced: {gaps}")
    log(f"SEQ bootstrap reproduced: max gap {max(gaps.values()):.1e}")

    ref = seq["reference_setting"]
    lgbm_ref = f"lgbm__{ref}"
    seq_keys = [k for k in names if not k.startswith("lgbm__")]

    def gains(n: int) -> dict:
        Bn = usable_upto(n)
        out = {"n_boot_requested": n, "n_boot_usable": int(len(Bn)), "seed": int(boot["seed"]),
               "difference": "(AUC(model, setting) - AUC(model, reference)) - (AUC(lgbm, setting) - AUC(lgbm, reference))",
               "gain_minus_lgbm_gain": {}}
        for key in seq_keys:
            model, setting, seed_tag = key.split("__")
            if setting == ref:
                continue
            m_ref, l_set = f"{model}__{ref}__{seed_tag}", f"lgbm__{setting}"
            if m_ref not in col or l_set not in col or lgbm_ref not in col:
                continue
            d = (Bn[:, col[key]] - Bn[:, col[m_ref]]) - (Bn[:, col[l_set]] - Bn[:, col[lgbm_ref]])
            lo, hi = interval(d)
            out["gain_minus_lgbm_gain"][key] = {
                "model": model, "setting": setting, "reference": ref, "seed": seed_tag,
                "gain": point[key] - point[m_ref], "lgbm_gain": point[l_set] - point[lgbm_ref],
                "diff": (point[key] - point[m_ref]) - (point[l_set] - point[lgbm_ref]),
                "ci95": [lo, hi], "boot_mean": float(d.mean()), "boot_se": float(d.std(ddof=1)),
                "excludes_zero": bool(lo > 0 or hi < 0), "p_delta_leq_0": float((d <= 0).mean()),
                "p_two_sided_mc": mc_p_two_sided(d)}
        return out

    blocks = {f"n_boot_{n}": gains(n) for n in sorted({int(boot["n_boot_requested"]), args.seq_n_boot})}
    return {"source": args.seq.as_posix(), "predictions": preds_path.as_posix(),
            "draw_code": "scripts/run_window_sweep_v33.py match_bootstrap(): match_weights, _auc_plan, _auc_from_plan "
                         "(imported), default_rng(SEQ bootstrap.seed); the first n draws of a longer run are the n-draw run",
            "reference_setting": ref, "n_rows": int(len(y)), "n_matches": n_groups,
            "row_checks": row_checks, "reproduces_seq_bootstrap": reproduces, **blocks,
            "wall_clock_s": round(time.time() - started, 1)}


# ----------------------------------------------------------------------------- (3) SAINT batches
def batch_sharing(groups: np.ndarray, batches) -> dict:
    n_shared_batches, n_rows_sharing, covered = 0, 0, np.zeros(len(groups), dtype=np.int64)
    for rows in batches:
        g = groups[rows]
        covered[rows] += 1
        _, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
        mult = counts[inv.reshape(-1)]
        if (counts > 1).any():
            n_shared_batches += 1
        n_rows_sharing += int((mult > 1).sum())
    return {"n_batches": len(batches), "n_batches_with_a_shared_match": n_shared_batches,
            "n_rows_sharing_batch_with_own_match": n_rows_sharing, "n_rows": int(len(groups)),
            "every_row_in_exactly_one_batch": bool((covered == 1).all())}


def published_saint_batches(inp: Inputs, args, learner_block: dict | None, log) -> dict:
    started = time.time()
    d = inp.npz(args.published_saint)
    groups = d["groups_test"]
    code = {}
    for commit in PUBLISHED_COMMITS:
        src = git("show", f"{commit}:scripts/run_deep_tabular_baselines.py")
        code[commit] = ("git unavailable" if src is None else
                        {line: line in src for line in PUBLISHED_CODE_LINES})
    tree = learner_block["draws"]["tree_prediction_file"] if learner_block else None
    row_checks = {}
    if tree:
        t = inp.npz(Path(tree))
        row_checks = {"groups_equal_tree_file": bool(np.array_equal(groups, t["groups"])),
                      "y_equal_tree_file": bool(np.array_equal(d["y_test"].astype(np.int64), t["y"].astype(np.int64)))}
        if not all(row_checks.values()):
            raise SystemExit(f"published SAINT rows differ from the tree file: {row_checks}")
    chunks = [np.arange(i, min(i + PUBLISHED_CHUNK, len(groups))) for i in range(0, len(groups), PUBLISHED_CHUNK)]
    published = batch_sharing(groups, chunks)
    runs = int(1 + np.count_nonzero(groups[1:] != groups[:-1]))
    n_matches = int(len(np.unique(groups)))
    log(f"published SAINT: {published['n_batches_with_a_shared_match']} of {published['n_batches']} test batches "
        f"share a match; {published['n_rows_sharing_batch_with_own_match']:,} of {published['n_rows']:,} rows")
    corrected = {}
    if learner_block:
        dtb = load_module("run_deep_tabular_baselines")
        inp.add(Path(dtb.__file__))
        for name, info in learner_block["learners"].items():
            cfg = info.get("config") or {}
            if info.get("status") not in ("re-run", "searched") or cfg.get("batching") != "match_disjoint":
                continue
            ebs = int(cfg.get("eval_batch_size") or cfg.get("batch_size"))
            seed = int((info.get("training_options") or {}).get("seed", dtb.SEED))
            g = inp.npz(Path(info["file"]))
            g = g["groups_test"] if "groups_test" in g else g["groups"]
            batches = dtb.match_disjoint_batches(g, ebs, np.random.default_rng(seed + 1))
            corrected[name] = {"eval_batch_size": ebs, "generator": f"default_rng({seed} + 1)", **batch_sharing(g, batches)}
            if corrected[name]["n_batches_with_a_shared_match"] or not corrected[name]["every_row_in_exactly_one_batch"]:
                raise SystemExit(f"{name}: rebuilt match-disjoint batches share a match: {corrected[name]}")
    return {"source": args.published_saint.as_posix(), "split": "test patch 15.16 (validation match ids were not saved)",
            "published_code": {"evaluate_loop": PUBLISHED_CODE_LINES[0], "batch_rows": PUBLISHED_CHUNK,
                               "lines_found_in_git_source": code},
            "row_checks": row_checks, **published,
            "n_match_runs_in_row_order": runs, "n_matches": n_matches, "rows_contiguous_by_match": runs == n_matches,
            "corrected_runs_rebuilt_batches": corrected,
            "corrected_batch_code": "scripts/run_deep_tabular_baselines.py match_disjoint_batches() (imported)",
            "wall_clock_s": round(time.time() - started, 1)}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a8", type=Path, default=DEFAULT_A8)
    ap.add_argument("--seq", type=Path, default=DEFAULT_SEQ)
    ap.add_argument("--published-saint", type=Path, default=DEFAULT_PUBLISHED_SAINT)
    ap.add_argument("--a1-dir", type=Path, default=A1, help="directory of the re-run outputs <stem>.json / .preds.npz")
    ap.add_argument("--queue-state", type=Path, default=DEFAULT_QUEUE_STATE)
    ap.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    ap.add_argument("--ignore-queue-state", action="store_true",
                    help="include a re-run whose JSON and predictions exist even without its queue .ok sentinel")
    ap.add_argument("--n-boot", type=int, default=2000, help="must equal A8's")
    ap.add_argument("--seed", type=int, default=SEED, help="must equal A8's")
    ap.add_argument("--seq-n-boot", type=int, default=2000,
                    help="second resample count for the sequence gain differences (SEQ's own count is always run)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    log = lambda *s: print(*s, flush=True)  # noqa: E731
    started = time.time()
    inp = Inputs()
    inp.add(Path(__file__).resolve())
    c1 = learner_cis(inp, args, log)
    c2 = sequence_gain_differences(inp, args, log)
    c3 = published_saint_batches(inp, args, c1, log)
    payload = {
        "item": ITEM,
        "script": "scripts/learner_rerun_paired_cis_v33.py",
        "script_sha1": sha1_of(Path(__file__).resolve()),
        "git": git_state(),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "python": sys.version.split()[0], "numpy": np.__version__,
        "learner_cis": c1,
        "sequence_gain_differences": c2,
        "published_saint_batches": c3,
        "inputs": inp.files,
        "wall_clock_s": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, args.out)
    for k in c1["available_reruns"]:
        v, c = c1["learners"][k], c1["comparisons"].get(f"{k} - lgbm_paper")
        log(f"  {k:18s} {v['test_auc']:.6f} [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]  - lgbm_paper "
            f"{c['diff']:+.6f} [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}]")
    log(f"pending: {[p['learner'] for p in c1['pending']]}")
    log(f"wrote {args.out} ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
