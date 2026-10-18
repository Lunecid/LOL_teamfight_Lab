"""Per-class paired contrast of the full LightGBM against the scoreboard reference, patch holdout (v3.3).

Question.  On the held-out test patch (train 15.14, select 15.15, test 15.16) a LightGBM given eight
scoreboard columns (``lead_only``) reaches AUC 0.626 and a LightGBM given all 6,164 non-constant pre-fight
columns reaches 0.669.  Where is the gap?  A readout built on the scoreboard alone is only adequate for the
engagement classes in which the full model adds little, so the gain is reported inside each scale class
(pick / skirmish / teamfight) with an interval, and the teamfight cut ``n_min >= 4`` is moved to 3 and 5.

What is computed.  Nothing is refit.  The frozen test-patch predictions of the patch-holdout learner
comparison are scored inside each class:

  AUC(model | class)                      for lead_only and every full model
  gain(class)   = AUC(full | class) - AUC(lead_only | class)
  gain(a) - gain(b)                       for every pair of classes at one cut

with 95 % match-clustered paired percentile bootstrap intervals.

Inputs (the files the A8 input audit identifies; its recorded sha1 is asserted):
  FEAT/model_comparison_v33_patch.preds.npz   y, groups, classes (cut 4), present, lead_only, linear,
                                              lgbm_paper, lgbm_deep  (worktree scripts/run_model_comparison_v33.py)
  FEAT/model_comparison_v33_patch.json        per-class point AUCs at cut 4, split, corpus manifest
  FEAT/scale_decomposition_v33_market_event.preds.npz
                                              cluster_blue / cluster_red for every labelled row; the rows with
                                              patch == test are the test rows above, in the same order
  REV/A8-input-audit-and-cis/model_comparison_input_audit_extended.json   prediction_files.tree.sha1
  REV/A8-input-audit-and-cis/paired_learner_cis_v33.json                  overall intervals to reproduce

Row alignment.  Both prediction files hold rows in the order of run_scale_decomposition.merge_shards, the
patch file restricted to the test patch.  The script asserts, row by row and in order, that the two files
carry identical match ids and labels, that the stored cut-4 classes equal scale_class() recomputed from the
participation counts, and that the scale file's own scale and presence arrays equal the patch file's
classes and present arrays.  Equality of those arrays cannot see a swap of two rows of one match that share
label, class and presence, so the scale file's five-fold out-of-fold prediction is correlated with the
patch run's lgbm_paper prediction as stored and after shuffling rows within such strata; the numbers are
written under validation.alignment.

Classes (run_scale_decomposition.scale_class, the classes the learner comparison reports):
n_min = min(cluster_blue, cluster_red), the smaller side's kill-cluster participants.  pick n_min <= 1,
skirmish 2 .. c-1, teamfight n_min >= c, for the cut c in {3, 4, 5}.  A negative count marks a team with no
kill-cluster participant (see scripts/scale_cut_sensitivity_v33.py); scale_class calls those rows
"unknown" and they are in no class (22 test rows) but are in "overall".  ``pick_negative_as_zero`` reports
the other reading, in which they are picks; skirmish and teamfight do not change.

Method.  Identical to REV/A8-input-audit-and-cis/paired_learner_cis_v33.json (worktree
scripts/paired_learner_cis_v33.py):
  resampling   Davison & Hinkley (1997), Bootstrap Methods and their Application, section 3.8: resample
               the test matches with replacement and keep every row of a drawn match.  One generator,
               numpy.random.default_rng(seed), one integers(n_matches, size=n_matches) per replicate.
               Every replicate resamples ALL test matches; a class AUC uses that class's rows under the
               replicate's weights, so every model, class and cut is scored on the same resample and every
               difference is paired.
  AUC          Mann-Whitney statistic with ties scored 1/2 (Hanley & McNeil 1982), with a row weighted by
               the number of times its match was drawn; equal to roc_auc_score on the concatenated resample
               (checked on the first replicates).
  interval     percentile, 2.5 / 97.5 (Efron & Tibshirani 1993, An Introduction to the Bootstrap, ch. 13).
  p value      Monte Carlo, 2 (1 + #{replicates on the far side of 0}) / (R + 1), capped at 1.
  Monte Carlo  the whole bootstrap is repeated with --mc-check-seed and the largest endpoint move is
  error        written out; the reported intervals are those of --seed.

Because the generator, seed and draw order are A8's, the overall intervals reproduce A8's to the bit;
the run checks that and writes the gap.

    .venv/Scripts/python.exe scripts/perclass_lead_contrast_v33.py
"""
from __future__ import annotations

import os

# a compute queue shares this machine: keep every numerical library to at most four threads
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_scale_decomposition import scale_class  # noqa: E402

ITEM = "C1-perclass-contrast"
FEATURES = Path("D:/LOL_Project/fusion_2615/features")
REV = FEATURES / "tog_revision"
A8 = REV / "A8-input-audit-and-cis"
DEFAULT_OUT = REV / "A10-perclass-contrast" / "perclass_lead_contrast_v33.json"
SEED = 7
REFERENCE = "lead_only"
HEADLINE_FULL = "lgbm_deep"
CUTS = (3, 4, 5)
PRIMARY_CUT = 4
CLASS_NAMES = ("pick", "skirmish", "teamfight")
CLASS_PAIRS = (("pick", "skirmish"), ("pick", "teamfight"), ("skirmish", "teamfight"))

LEARNERS = {
    "lead_only": "LightGBM, 8 columns: goldDiff__last, xpDiff__last, avgLevelDiff__last, killDiff_cum__last, "
                 "csDiff_total__last, aliveDiff__last, time_norm__last, frame_age_s (worktree "
                 "scripts/run_model_comparison_v33.py lines 214-216 at commit 3fb00c3); n_estimators 400, "
                 "learning_rate 0.05, num_leaves 31, subsample 0.9, colsample_bytree 0.9 (line 124), early "
                 "stopping patience 100 on the validation patch (line 131)",
    "lgbm_paper": "LightGBM, all 6,164 non-constant columns, the same configuration as lead_only (the published "
                  "configuration)",
    "lgbm_deep": "LightGBM, all 6,164 non-constant columns, n_estimators 3000, learning_rate 0.02, num_leaves 127, "
                 "min_child_samples 100, reg_lambda 1.0 (lines 126-128), early stopping patience 100 (line 131); "
                 "the higher-capacity model the manuscript calls the full model",
}
DEVIATIONS = [
    "rows are weighted by match multiplicity instead of concatenating the resample; the AUC is identical (the gap "
    "on the checked replicates is written under validation)",
    "percentile interval, not BCa or studentised",
    "every replicate resamples all test matches and scores a class on its rows under those weights; the number of "
    "rows in a class therefore varies across replicates, as it does across samples of matches",
    "matches are the only clustering level; the same player or account across matches is not clustered",
    "each learner is one fitted model (one training seed, 7): the intervals cover test-set sampling only, not "
    "training-seed variability",
    "intervals are per statistic and unadjusted; nothing here is simultaneous across classes, cuts or learners",
    "p values are Monte Carlo from the same replicates, so the smallest attainable two-sided p is 2 / (R + 1)",
]
CAVEATS = [
    "scale class is a post-hoc property: participation counts kill-cluster participants and in-radius interaction "
    "actors, known only once the engagement has resolved.  It splits frozen test predictions and is never a model "
    "input; no model is refit per class",
    "the three cuts share the pick class and every replicate, so the per-cut results are not independent evidence",
    "the 22 test rows with a negative participation count are in 'overall' and in no class; "
    "'pick_negative_as_zero' gives the pick results with those rows read as picks",
    "row identity between the two prediction files rests on the merge order both runs share; the equality and "
    "correlation checks under validation.alignment corroborate it and cannot prove it for rows that agree on "
    "match, label, class, presence and participation counts (such rows are interchangeable for every statistic "
    "here)",
]


# ----------------------------------------------------------------------------- provenance
def sha1_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path: Path) -> dict:
    st = path.stat()
    return {"sha1": sha1_of(path), "bytes": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}


def git_state(script: Path) -> dict:
    def run(*cmd):
        try:
            return subprocess.run(["git", "-C", str(PROJECT_ROOT), *cmd], capture_output=True, text=True,
                                  timeout=60).stdout
        except Exception as exc:  # provenance must never abort the run
            return f"unavailable: {exc}"
    rel = script.relative_to(PROJECT_ROOT).as_posix()
    # porcelain lines are 'XY path'; the first status column may be a space, so the output is never stripped
    status = run("status", "--porcelain", "--", rel)
    tree_line = run("ls-tree", "HEAD", "--", rel).split()  # '<mode> blob <sha> <path>' or nothing if untracked
    return {"repo": str(PROJECT_ROOT), "commit": run("rev-parse", "HEAD").strip(),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "script": rel,
            "script_status_at_run": status[:2] if status else "clean (committed as run)",
            "script_blob_at_head": tree_line[2] if len(tree_line) >= 3 else None,
            "dirty_tracked_files": [ln[3:] for ln in run("status", "--porcelain", "--untracked-files=no").splitlines()]}


# ----------------------------------------------------------------------------- AUC and bootstrap
def prepare(score: np.ndarray, y: np.ndarray, rows: np.ndarray):
    """Sort a class's rows by score once; tie groups do not depend on the bootstrap weights.

    Returns (row indices into the full test vector in score order, tie-group starts, positive mask), so one
    gather of the replicate's row weights serves the class.  Same arithmetic as A8's weighted_auc.
    """
    s_class = score[rows]
    order = np.argsort(s_class, kind="mergesort")
    s = s_class[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return rows[order], starts, y[rows][order] == 1


def weighted_auc(prep, w: np.ndarray) -> float:
    """Mann-Whitney AUC over weighted rows, ties scored 1/2."""
    full_order, starts, pos = prep
    ws = w[full_order]
    gp = np.add.reduceat(np.where(pos, ws, 0.0), starts)
    gn = np.add.reduceat(np.where(pos, 0.0, ws), starts)
    denom = gp.sum() * gn.sum()
    if denom == 0:
        return float("nan")
    return float((gp * (np.cumsum(gn) - 0.5 * gn)).sum() / denom)


def bootstrap(prep, set_names, models, inv, n_groups, n_boot, seed, log, started):
    """(n_boot, sets, models) AUCs; replicate b draws n_groups matches with replacement from one generator."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(set_names), len(models)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups).astype(np.float64)[inv]
        for i, s in enumerate(set_names):
            for j, m in enumerate(models):
                out[b, i, j] = weighted_auc(prep[(s, m)], w)
        if (b + 1) % 250 == 0:
            log(f"  seed {seed}: replicate {b + 1}/{n_boot} ({time.time() - started:.0f}s)")
    return out


def interval(d: np.ndarray) -> list:
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


def summarise_difference(point: float, d: np.ndarray, d_mc, label_hi: str, label_lo: str) -> dict:
    lo, hi = interval(d)
    far = min(int((d <= 0).sum()), int((d >= 0).sum()))
    out = {"diff": point, "boot_mean": float(d.mean()), "boot_se": float(d.std(ddof=1)), "ci95": [lo, hi],
           "excludes_zero": bool(lo > 0 or hi < 0), "p_two_sided": min(1.0, 2 * (1 + far) / (len(d) + 1)),
           "reading": "tie: the 95% interval contains 0" if lo <= 0 <= hi else (label_hi if lo > 0 else label_lo)}
    if d_mc is not None:
        out["ci95_mc_check_seed"] = interval(d_mc)
    return out


# ----------------------------------------------------------------------------- inputs and alignment
def load_inputs(a, log):
    files = {}
    for p in (a.tree_results, a.tree_preds, a.scale_preds, a.a8_audit, a.a8_cis):
        files[str(p)] = file_info(p)
    audit = json.loads(a.a8_audit.read_text(encoding="utf-8"))
    audited = audit["prediction_files"]["tree"]
    if Path(audited["path"]) != a.tree_preds or audited["sha1"] != files[str(a.tree_preds)]["sha1"]:
        raise SystemExit(f"{a.tree_preds} is not the tree prediction file the A8 audit identifies "
                         f"({audited['path']}, sha1 {audited['sha1']})")
    if audit.get("verdict") != "PASS":
        raise SystemExit(f"A8 audit verdict is {audit.get('verdict')!r}, not PASS")
    tree = json.loads(a.tree_results.read_text(encoding="utf-8"))
    a8 = json.loads(a.a8_cis.read_text(encoding="utf-8"))
    a8_tree = [v for k, v in a8["provenance"]["input_files"].items() if Path(k) == a.tree_preds]
    if len(a8_tree) != 1 or a8_tree[0]["sha1"] != files[str(a.tree_preds)]["sha1"]:
        raise SystemExit("paired_learner_cis_v33.json was computed from a different tree prediction file")

    with np.load(a.tree_preds, allow_pickle=False) as z:
        y = z["y"].astype(np.int64)
        groups = z["groups"]
        classes = z["classes"].astype(str)
        present = z["present"].astype(str)
        preds = {k: z[k].astype(np.float64) for k in (REFERENCE, *a.full_models)}
    test_patch = tree["split"]["test"]
    with np.load(a.scale_preds, allow_pickle=False) as z:
        mask = z["patch"] == test_patch
        sc = {k: z[k][mask] for k in ("y", "groups", "scale", "presence", "cluster_blue", "cluster_red", "pred")}
        n_scale_rows = int(len(mask))
    log(f"test patch {test_patch}: {len(y):,} rows in the patch file, {int(mask.sum()):,} in the scale file")

    checks = {}

    def check(name, ok, detail=None):
        checks[name] = {"pass": bool(ok), **({"detail": detail} if detail is not None else {})}
        if not ok:
            raise SystemExit(f"alignment check failed: {name} {detail or ''}")

    check("row count: scale-file rows with patch == test equal the patch file's rows",
          int(mask.sum()) == len(y) == tree["split"]["rows"]["test"],
          f"{int(mask.sum())} / {len(y)} / {tree['split']['rows']['test']}")
    check("match ids identical row by row", np.array_equal(sc["groups"], groups))
    check("labels identical row by row", np.array_equal(sc["y"].astype(np.int64), y))
    check("stored classes equal scale_class(cluster_blue, cluster_red, 4)",
          np.array_equal(scale_class(sc["cluster_blue"], sc["cluster_red"], teamfight_min=4).astype(str), classes))
    check("scale file's own scale array equals the patch file's classes", np.array_equal(sc["scale"].astype(str), classes))
    check("scale file's presence array equals the patch file's present", np.array_equal(sc["presence"].astype(str), present))
    return tree, a8, files, y, groups, classes, present, preds, sc, checks, n_scale_rows


def within_strata_order_check(pred_a, pred_b, strata_keys, n_shuffles, seed):
    """Pearson r of two row-aligned prediction vectors, as stored and after permuting rows within strata."""
    _, sid = np.unique(np.rec.fromarrays(strata_keys), return_inverse=True)
    sid = sid.ravel()
    sizes = np.bincount(sid)
    in_multi = sizes[sid] > 1
    rng = np.random.default_rng(seed)
    base = np.lexsort((np.arange(len(sid)), sid))
    rs = []
    for _ in range(n_shuffles):
        perm = np.empty(len(sid), dtype=np.int64)
        perm[base] = np.lexsort((rng.random(len(sid)), sid))
        rs.append(float(np.corrcoef(pred_a[perm], pred_b)[0, 1]))
    return {"r_aligned": float(np.corrcoef(pred_a, pred_b)[0, 1]),
            "r_shuffled_mean": float(np.mean(rs)), "r_shuffled_max": float(np.max(rs)),
            "n_shuffles": n_shuffles, "n_strata": int(len(sizes)), "rows_in_multi_row_strata": int(in_multi.sum())}


# ----------------------------------------------------------------------------- prior scratch computation
def compare_prior(prior_cut, prior_ci, res_by_cut, overall, full):
    """Largest absolute gap between this run and the session-scratchpad computation it archives."""
    gaps = []

    def gap(a_, b_):
        gaps.append(abs(float(a_) - float(b_)))
    if prior_cut is not None:
        for cut in CUTS:
            p = prior_cut.get(str(cut))
            if p is None:
                continue
            r = res_by_cut[f"cut_{cut}"]
            for c in CLASS_NAMES:
                rc = r["classes"][c]
                gap(p[c]["n"], rc["n"])
                gap(p[c]["share_of_test"], rc["share_of_test"])
                for m in (REFERENCE, full):
                    gap(p[c]["auc"][m], rc["auc"][m]["point"])
                gap(p[c]["gain"], rc["contrasts"][f"{REFERENCE}->{full}"]["diff"])
                for x, z in zip(p[c]["gain_ci95"], rc["contrasts"][f"{REFERENCE}->{full}"]["ci95"]):
                    gap(x, z)
            for ca, cb in CLASS_PAIRS:
                pp = p[f"gain_{ca}_minus_{cb}"]
                rr = r["gain_differences"][full][f"{ca}-{cb}"]
                gap(pp["point"], rr["diff"])
                for x, z in zip(pp["ci95"], rr["ci95"]):
                    gap(x, z)
    if prior_ci is not None:
        r4 = res_by_cut[f"cut_{PRIMARY_CUT}"]
        for c in ("all", *CLASS_NAMES):
            node = overall if c == "all" else r4["classes"][c]
            for m in (REFERENCE, full):
                gap(prior_ci["point"][c][m], node["auc"][m]["point"])
                for x, z in zip(prior_ci["seed7"][c][f"{m}_ci95"], node["auc"][m]["ci95"]):
                    gap(x, z)
                if "seed8_mc_check" in prior_ci and "ci95_mc_check_seed" in node["auc"][m]:
                    for x, z in zip(prior_ci["seed8_mc_check"][c][f"{m}_ci95"], node["auc"][m]["ci95_mc_check_seed"]):
                        gap(x, z)
            con = node["contrasts"][f"{REFERENCE}->{full}"]
            gap(prior_ci["seed7"][c]["gain_point"], con["diff"])
            for x, z in zip(prior_ci["seed7"][c]["gain_ci95"], con["ci95"]):
                gap(x, z)
        for ca, cb in CLASS_PAIRS:
            pp = prior_ci["seed7"][f"gain_{ca}_minus_gain_{cb}"]
            rr = r4["gain_differences"][full][f"{ca}-{cb}"]
            gap(pp["point"], rr["diff"])
            for x, z in zip(pp["ci95"], rr["ci95"]):
                gap(x, z)
    return (max(gaps) if gaps else None), len(gaps)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree-results", type=Path, default=FEATURES / "model_comparison_v33_patch.json")
    ap.add_argument("--tree-preds", type=Path, default=FEATURES / "model_comparison_v33_patch.preds.npz")
    ap.add_argument("--scale-preds", type=Path, default=FEATURES / "scale_decomposition_v33_market_event.preds.npz")
    ap.add_argument("--a8-audit", type=Path, default=A8 / "model_comparison_input_audit_extended.json")
    ap.add_argument("--a8-cis", type=Path, default=A8 / "paired_learner_cis_v33.json")
    ap.add_argument("--full-models", default="lgbm_deep,lgbm_paper",
                    help="comma-separated; each is contrasted with lead_only.  The first is the headline")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mc-check-seed", type=int, default=8, help="-1 skips the Monte Carlo replication")
    ap.add_argument("--validate-reps", type=int, default=3)
    ap.add_argument("--prior-cut-json", type=Path, default=None,
                    help="scratchpad perclass_cut.json of the computation this script archives (optional)")
    ap.add_argument("--prior-ci-json", type=Path, default=None,
                    help="scratchpad perclass_ci.json of the computation this script archives (optional)")
    ap.add_argument("--prior-scripts", default="",
                    help="comma-separated scratchpad scripts of that computation; recorded by sha1 only")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    a.full_models = tuple(s.strip() for s in a.full_models.split(",") if s.strip())
    log = lambda *s: print(*s, flush=True)  # noqa: E731
    started = time.time()
    if a.n_boot < 1000:
        raise SystemExit("--n-boot must be >= 1000 for reported intervals")
    if a.mc_check_seed == a.seed:
        raise SystemExit("--mc-check-seed must differ from --seed")
    if HEADLINE_FULL not in a.full_models:
        raise SystemExit(f"--full-models must include the headline model {HEADLINE_FULL}")

    tree, a8, files, y, groups, classes, present, preds, sc, checks, n_scale_rows = load_inputs(a, log)
    models = (REFERENCE, *a.full_models)
    uniq, inv = np.unique(groups, return_inverse=True)
    n = len(y)
    log(f"rows {n:,}  matches {len(uniq):,}  positive {y.mean():.4f}  models {list(models)}")

    # class row sets
    blue, red = sc["cluster_blue"], sc["cluster_red"]
    unknown = (blue < 0) | (red < 0)
    by_cut = {c: scale_class(blue, red, teamfight_min=c).astype(str) for c in CUTS}
    if not all(np.array_equal(by_cut[c] == "pick", by_cut[PRIMARY_CUT] == "pick") for c in CUTS):
        raise SystemExit("pick rows differ between cuts")
    checks["pick rows identical at every cut"] = {"pass": True}
    checks["unknown rows are exactly the rows with a negative count, at every cut"] = {
        "pass": bool(all(np.array_equal(by_cut[c] == "unknown", unknown) for c in CUTS))}
    sets = {"all": np.arange(n), "pick": np.flatnonzero(by_cut[PRIMARY_CUT] == "pick"),
            "pick_negative_as_zero": np.flatnonzero((by_cut[PRIMARY_CUT] == "pick") | unknown)}
    for c in CUTS:
        for k in ("skirmish", "teamfight"):
            sets[f"{k}@{c}"] = np.flatnonzero(by_cut[c] == k)
    set_names = list(sets)

    def set_of(cls, cut):
        return "pick" if cls == "pick" else f"{cls}@{cut}"

    prep = {(s, m): prepare(preds[m], y, sets[s]) for s in set_names for m in models}
    point = {(s, m): float(roc_auc_score(y[sets[s]], preds[m][sets[s]])) for s in set_names for m in models}
    unit = np.ones(n)
    gap_unit = max(abs(weighted_auc(prep[k], unit) - point[k]) for k in prep)
    log(f"point AUCs computed; unit-weight vs roc_auc_score gap {gap_unit:.1e}")

    # within-match row order: scale-file OOF prediction vs the patch run's lgbm_paper
    order_check = None
    if "lgbm_paper" in preds:
        order_check = {
            "compared": "scale_decomposition five-fold OOF 'pred' (published configuration) vs the patch run's "
                        "lgbm_paper test prediction",
            "within_match": within_strata_order_check(sc["pred"], preds["lgbm_paper"], [groups], 5, 11),
            "within_match_label_class_presence": within_strata_order_check(
                sc["pred"], preds["lgbm_paper"], [groups, y, classes, present], 5, 12)}
        cnt_strata = [groups, y, classes, present]
        _, sid = np.unique(np.rec.fromarrays(cnt_strata), return_inverse=True)
        sid = sid.ravel()
        mixed = np.zeros(sid.max() + 1, dtype=bool)
        for c in CUTS:
            _, cid = np.unique(by_cut[c], return_inverse=True)
            lo = np.full(sid.max() + 1, 99)
            hi = np.full(sid.max() + 1, -1)
            np.minimum.at(lo, sid, cid.ravel())
            np.maximum.at(hi, sid, cid.ravel())
            mixed |= lo != hi
        order_check["rows_in_match_label_class_presence_strata_whose_cut3_or_cut5_class_differs"] = int(mixed[sid].sum())
        order_check["note"] = (
            "equality of match ids, labels, classes and presence cannot see a swap of two rows of one match that "
            "agree on all four; the correlation of two independently fitted models' predictions drops when rows "
            "are shuffled within those strata, so the stored order is the order the rows share.  The last count is "
            "the number of rows such a swap could move to a different class at cut 3 or 5")
        log(f"row-order check: r aligned {order_check['within_match']['r_aligned']:.4f}; shuffled within match "
            f"{order_check['within_match']['r_shuffled_max']:.4f}; within match/label/class/presence "
            f"{order_check['within_match_label_class_presence']['r_shuffled_max']:.4f}")

    boot = bootstrap(prep, set_names, models, inv, len(uniq), a.n_boot, a.seed, log, started)
    usable = ~np.isnan(boot).any(axis=(1, 2))
    B = boot[usable]
    si = {s: i for i, s in enumerate(set_names)}
    mi = {m: j for j, m in enumerate(models)}

    # the weighted AUC must equal roc_auc_score on the concatenated resample
    rng_v = np.random.default_rng(a.seed)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    member = {s: np.isin(np.arange(n), sets[s]) for s in set_names}
    concat_gap = 0.0
    for b in range(min(a.validate_reps, a.n_boot)):
        idx = np.concatenate([rows_of[i] for i in rng_v.integers(len(uniq), size=len(uniq))])
        for s in set_names:
            r = idx[member[s][idx]]
            for m in models:
                concat_gap = max(concat_gap, abs(roc_auc_score(y[r], preds[m][r]) - boot[b, si[s], mi[m]]))
    log(f"weighted vs concatenated resample gap {concat_gap:.1e}")

    B2 = None
    if a.mc_check_seed >= 0:
        log(f"independent Monte Carlo replication with seed {a.mc_check_seed} ...")
        boot2 = bootstrap(prep, set_names, models, inv, len(uniq), a.n_boot, a.mc_check_seed, log, started)
        B2 = boot2[~np.isnan(boot2).any(axis=(1, 2))]
    mc_gaps, flips = [], []

    def auc_node(s, m):
        node = {"point": point[(s, m)], "ci95": interval(B[:, si[s], mi[m]]),
                "boot_se": float(B[:, si[s], mi[m]].std(ddof=1))}
        if B2 is not None:
            node["ci95_mc_check_seed"] = interval(B2[:, si[s], mi[m]])
            mc_gaps.append(max(abs(p - q) for p, q in zip(node["ci95"], node["ci95_mc_check_seed"])))
        return node

    def gain_series(Bx, s, full):
        return Bx[:, si[s], mi[full]] - Bx[:, si[s], mi[REFERENCE]]

    def diff_node(pt, d, d2, hi, lo, key):
        node = summarise_difference(pt, d, d2, hi, lo)
        if d2 is not None:
            mc_gaps.append(max(abs(p - q) for p, q in zip(node["ci95"], node["ci95_mc_check_seed"])))
            r2 = interval(d2)
            reading2 = "tie: the 95% interval contains 0" if r2[0] <= 0 <= r2[1] else (hi if r2[0] > 0 else lo)
            if reading2 != node["reading"]:
                flips.append(key)
        return node

    def class_node(s):
        rows = sets[s]
        node = {"n": int(len(rows)), "n_matches": int(len(np.unique(groups[rows]))),
                "share_of_test": len(rows) / n, "positive_rate": float(y[rows].mean()),
                "auc": {m: auc_node(s, m) for m in models}, "contrasts": {}}
        for full in a.full_models:
            key = f"{REFERENCE}->{full}"
            node["contrasts"][key] = {
                "difference": f"AUC({full}) - AUC({REFERENCE})",
                **diff_node(point[(s, full)] - point[(s, REFERENCE)], gain_series(B, s, full),
                            None if B2 is None else gain_series(B2, s, full),
                            f"{full} higher", f"{REFERENCE} higher", f"{s}:{key}")}
        return node

    overall = class_node("all")
    overall.pop("share_of_test")
    res_by_cut = {}
    for cut in CUTS:
        cls_nodes = {c: class_node(set_of(c, cut)) for c in CLASS_NAMES}
        gd = {}
        for full in a.full_models:
            gd[full] = {}
            for ca, cb in CLASS_PAIRS:
                sa, sb = set_of(ca, cut), set_of(cb, cut)
                pt = (point[(sa, full)] - point[(sa, REFERENCE)]) - (point[(sb, full)] - point[(sb, REFERENCE)])
                d = gain_series(B, sa, full) - gain_series(B, sb, full)
                d2 = None if B2 is None else gain_series(B2, sa, full) - gain_series(B2, sb, full)
                gd[full][f"{ca}-{cb}"] = {"difference": f"gain({ca}) - gain({cb}), gain = AUC({full}) - AUC({REFERENCE})",
                                          **diff_node(pt, d, d2, f"gain larger in {ca}", f"gain larger in {cb}",
                                                      f"cut{cut}:{full}:{ca}-{cb}")}
        ordering = {}
        for full in a.full_models:
            g = {c: cls_nodes[c]["contrasts"][f"{REFERENCE}->{full}"]["diff"] for c in CLASS_NAMES}
            ordering[full] = {
                "point_order_largest_first": sorted(CLASS_NAMES, key=lambda c: -g[c]),
                "pick_gt_skirmish_gt_teamfight_with_every_pairwise_interval_above_zero": bool(
                    all(gd[full][f"{ca}-{cb}"]["ci95"][0] > 0 for ca, cb in CLASS_PAIRS))}
        res_by_cut[f"cut_{cut}"] = {
            "definition": f"n_min = min(cluster_blue, cluster_red); pick n_min <= 1, skirmish 2..{cut - 1}, "
                          f"teamfight n_min >= {cut}; negative count -> unknown (excluded)",
            "unknown_rows_excluded": int(unknown.sum()),
            "classes": cls_nodes, "gain_differences": gd, "ordering": ordering}
    pick_zero = class_node("pick_negative_as_zero")

    # ------------------------------------------------------------------ validation against published artefacts
    tree_gap, tree_n_equal = 0.0, True
    for m in models:
        tree_gap = max(tree_gap, abs(point[("all", m)] - tree["models"][m]["overall_auc"]))
        for c in CLASS_NAMES:
            ref = tree["models"][m]["by_participation_scale"][c]
            tree_gap = max(tree_gap, abs(point[(set_of(c, PRIMARY_CUT), m)] - ref["auc"]))
            tree_n_equal &= int(ref["n"]) == len(sets[set_of(c, PRIMARY_CUT)])
    a8_repro, a8_gap = {}, None
    a8_prov = a8["provenance"]
    same_design = (a8_prov["seed"] == a.seed and a8_prov["n_boot"] == a.n_boot
                   and a8["validation"]["monte_carlo_replication"].get("seed") == a.mc_check_seed)
    gaps = []
    for m in models:
        L = a8["learners"][m]
        mine = overall["auc"][m]
        a8_repro[f"learners.{m}"] = {"a8": [L["test_auc"], *L["ci95"]], "here": [mine["point"], *mine["ci95"]]}
        gaps += [abs(p - q) for p, q in zip(a8_repro[f"learners.{m}"]["a8"], a8_repro[f"learners.{m}"]["here"])]
    for full in a.full_models:
        key = f"{REFERENCE}->{full}"
        C = a8["comparisons"][key]
        mine = overall["contrasts"][key]
        a8_repro[f"comparisons.{key}"] = {"a8": [C["diff"], *C["ci95"], *C.get("ci95_mc_check_seed", [])],
                                          "here": [mine["diff"], *mine["ci95"], *mine.get("ci95_mc_check_seed", [])]}
        gaps += [abs(p - q) for p, q in zip(a8_repro[f"comparisons.{key}"]["a8"], a8_repro[f"comparisons.{key}"]["here"])]
    a8_gap = max(gaps)
    log(f"per-class point AUCs vs model_comparison_v33_patch.json: max gap {tree_gap:.1e}, n equal {tree_n_equal}; "
        f"overall vs A8: max gap {a8_gap:.1e} (same seed/n_boot/mc seed: {same_design})")

    prior = None
    if a.prior_cut_json is not None or a.prior_ci_json is not None:
        pc = json.loads(a.prior_cut_json.read_text(encoding="utf-8")) if a.prior_cut_json else None
        pci = json.loads(a.prior_ci_json.read_text(encoding="utf-8")) if a.prior_ci_json else None
        pgap, pcount = compare_prior(pc, pci, res_by_cut, overall, HEADLINE_FULL)
        prior = {"what": "the session-scratchpad computation of 2026-09-14 that this script archives "
                         "(cited in docs/tog_manuscript/sec_intro.tex comments)",
                 "files": {str(p): file_info(p) for p in (a.prior_cut_json, a.prior_ci_json) if p is not None},
                 "scripts": {s: file_info(Path(s)) for s in a.prior_scripts.split(",") if s.strip()},
                 "values_compared": pcount, "max_abs_gap": pgap}
        log(f"scratchpad computation reproduced: {pcount} values, max gap {pgap:.1e}")

    # ------------------------------------------------------------------ headline
    def r4(x):
        return round(float(x), 4)

    headline = {"model": HEADLINE_FULL, "reference": REFERENCE, "cuts": {}}
    for cut in CUTS:
        rc = res_by_cut[f"cut_{cut}"]
        h = {}
        for c in CLASS_NAMES:
            node = rc["classes"][c]
            con = node["contrasts"][f"{REFERENCE}->{HEADLINE_FULL}"]
            h[c] = {"n": node["n"],
                    REFERENCE: [r4(node["auc"][REFERENCE]["point"]), [r4(x) for x in node["auc"][REFERENCE]["ci95"]]],
                    HEADLINE_FULL: [r4(node["auc"][HEADLINE_FULL]["point"]), [r4(x) for x in node["auc"][HEADLINE_FULL]["ci95"]]],
                    "gain": [r4(con["diff"]), [r4(x) for x in con["ci95"]]]}
        for ca, cb in CLASS_PAIRS:
            g = rc["gain_differences"][HEADLINE_FULL][f"{ca}-{cb}"]
            h[f"gain {ca}-{cb}"] = [r4(g["diff"]), [r4(x) for x in g["ci95"]]]
        h["ordering_holds"] = rc["ordering"][HEADLINE_FULL][
            "pick_gt_skirmish_gt_teamfight_with_every_pairwise_interval_above_zero"]
        headline["cuts"][str(cut)] = h
    oc = overall["contrasts"][f"{REFERENCE}->{HEADLINE_FULL}"]
    headline["overall"] = {REFERENCE: [r4(overall["auc"][REFERENCE]["point"]), [r4(x) for x in overall["auc"][REFERENCE]["ci95"]]],
                           HEADLINE_FULL: [r4(overall["auc"][HEADLINE_FULL]["point"]), [r4(x) for x in overall["auc"][HEADLINE_FULL]["ci95"]]],
                           "gain": [r4(oc["diff"]), [r4(x) for x in oc["ci95"]]]}
    headline["format"] = "[point, [2.5 %, 97.5 %]]"

    script = Path(__file__).resolve()
    manifest = tree.get("corpus_manifest", {})
    payload = {
        "item": ITEM,
        "question": "on the held-out test patch, how much does the full LightGBM add over the scoreboard-only "
                    "LightGBM inside each engagement scale class, and does the ordering depend on the teamfight cut",
        "method": "frozen test-patch predictions scored per class; match-clustered paired percentile bootstrap "
                  "(Davison & Hinkley 1997 s3.8; Efron & Tibshirani 1993 ch13), A8's generator, seed and draw order",
        "headline": headline,
        "learners": {m: LEARNERS.get(m, "") for m in models},
        "overall": overall,
        "by_cut": res_by_cut,
        "pick_negative_as_zero": {"definition": "pick rows plus the rows with a negative participation count "
                                                "(read as zero participants); skirmish and teamfight unchanged",
                                  **pick_zero},
        "validation": {
            "alignment": {"checks": checks, "within_match_row_order": order_check},
            "max_gap_unit_weight_auc_vs_sklearn": gap_unit,
            "max_gap_weighted_vs_concatenated_resample": concat_gap,
            "replicates_checked_by_concatenation": int(min(a.validate_reps, a.n_boot)),
            "model_comparison_v33_patch_json": {"max_gap_point_auc_overall_and_cut4_classes": tree_gap,
                                                "cut4_class_counts_equal": bool(tree_n_equal)},
            "a8_paired_learner_cis_reproduction": {"same_seed_n_boot_and_mc_seed": bool(same_design),
                                                   "max_abs_gap": a8_gap, "values": a8_repro},
            "monte_carlo_replication": (
                {"seed": a.mc_check_seed, "n_boot_usable": int(len(B2)),
                 "max_endpoint_gap_between_seeds": float(max(mc_gaps)),
                 "median_endpoint_gap_between_seeds": float(np.median(mc_gaps)),
                 "differences_whose_reading_changes": flips,
                 "note": "the reported intervals are those of --seed; the second seed measures how far an endpoint "
                         "moves under an independent set of resamples"}
                if B2 is not None else "skipped (--mc-check-seed -1)"),
            "prior_scratchpad_computation": prior,
        },
        "deviations": DEVIATIONS,
        "caveats": CAVEATS,
        "provenance": {
            "script": str(script), "script_sha1": sha1_of(script), "git": git_state(script),
            "argv": sys.argv[1:], "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "corpus_run_id": manifest.get("run_id"), "corpus_git_commit": manifest.get("git_commit"),
            "detector": manifest.get("detector"), "label_key": tree.get("y_key"),
            "split": {"kind": "patch holdout", "train": tree["split"]["train"], "val": tree["split"]["val"],
                      "test": tree["split"]["test"], "rows": tree["split"]["rows"]},
            "tree_run_code": "worktree C:/Users/todtj/문서/LOL_Teamfight/worktrees/engagement-state-value "
                             "scripts/run_model_comparison_v33.py, first committed in 3fb00c3 (the run finished "
                             "before that commit; A8 audit run_code_provenance)",
            "n_rows_test": int(n), "n_matches_test": int(len(uniq)), "positive_rate_test": float(y.mean()),
            "scale_file_rows_all_patches": n_scale_rows,
            "seed": a.seed, "n_boot": a.n_boot, "n_boot_usable": int(usable.sum()),
            "mc_check_seed": a.mc_check_seed, "input_files": files,
            "input_audit": str(a.a8_audit),
            "wall_clock_s": round(time.time() - started, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def fmt(v):
        return f"{v[0]:.4f} [{v[1][0]:.4f}, {v[1][1]:.4f}]"
    log(f"\noverall  {REFERENCE} {fmt(headline['overall'][REFERENCE])}  {HEADLINE_FULL} "
        f"{fmt(headline['overall'][HEADLINE_FULL])}  gain {fmt(headline['overall']['gain'])}")
    for cut in CUTS:
        h = headline["cuts"][str(cut)]
        log(f"cut {cut}:")
        for c in CLASS_NAMES:
            log(f"  {c:9s} n={h[c]['n']:6d}  {REFERENCE} {fmt(h[c][REFERENCE])}  {HEADLINE_FULL} "
                f"{fmt(h[c][HEADLINE_FULL])}  gain {fmt(h[c]['gain'])}")
        for ca, cb in CLASS_PAIRS:
            log(f"  gain {ca}-{cb}: {fmt(h[f'gain {ca}-{cb}'])}")
        log(f"  ordering pick > skirmish > teamfight with every pairwise interval above 0: {h['ordering_holds']}")
    if B2 is not None:
        log(f"Monte Carlo replication (seed {a.mc_check_seed}): endpoints move by at most {max(mc_gaps):.5f} "
            f"(median {np.median(mc_gaps):.5f}); readings that change: {flips or 'none'}")
    log(f"wrote {a.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
