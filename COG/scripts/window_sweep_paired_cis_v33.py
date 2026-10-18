"""Paired, match-clustered bootstrap CIs for the v3.3 observation-window sweep (item C2-window-paired-cis).

Answers CoG 2026 review R2 ("alternative temporal resolutions or extended observation windows") with
uncertainty attached.  scripts/run_window_sweep_v33.py (item A3-temporal-windows) fitted LightGBM once per
setting on the same 50,766 engagements (18,335 matches) and cached the predictions:

    ctx 15 / 30 / 60 / 120 s at 5 s bins      (L = 3 / 6 / 12 / 24)
    ctx 30 s at 2.5 s and 10 s bins            (L = 12 / 3)

This script reads nothing but those cached predictions and the row metadata; it fits no model.  For every
setting against the reference ctx30_bin5000 it

  1. proves the rows are the same rows: the setting cache's own row keys (match_id, engage_ts, position in the
     shared fight index), labels, match ids and patches, aligned onto common_rows.npz, must equal the
     reference's element for element; the prediction sidecars must have been written under the SHA-1 of those
     common rows; every prediction vector must reproduce the AUC its sidecar recorded from the stored labels;
     and the per-setting .npz vectors must be bit-identical to the consolidated window_sweep_v33.preds.npz;
  2. computes, on (a) the patch-holdout test rows (patch 15.16) and (b) the match-grouped 5-fold out-of-fold
     rows, each setting's AUC with a 95 % CI and the paired AUC difference setting - reference with a 95 % CI.

Why a match-clustered paired bootstrap.  One match contributes several engagements whose outcomes share the
same players, draft and gold state, so rows are not independent; resampling rows would understate the
spread.  Whole matches are therefore drawn with replacement (a cluster bootstrap, Efron & Tibshirani 1993,
ch. 13; Field & Welsh 2007, JRSS-B 69(3), sec. 2.1), and a drawn match enters with its multiplicity.  The
same draw is applied to both prediction vectors of a pair, so the variation the two settings share (which
matches happen to be easy) cancels in the difference -- that is what makes a paired CI much narrower than
the two marginal CIs suggest.  AUC is the weighted Mann-Whitney statistic (Hanley & McNeil 1982, eq. 1; ties
count one half), numerically identical to concatenating the drawn matches' rows.  Intervals are percentile
intervals.  The replicate stream (numpy default_rng(seed), one integers(0, n_matches, n_matches) call per
replicate, matches indexed by np.unique order) is the one run_window_sweep_v33.py used with 1,000 replicates,
so the first 1,000 of the 2,000 replicates must reproduce that run's intervals; the script asserts it.

What the intervals above do not cover.  The fitted models are held fixed: those CIs are evaluation-sample
(match) uncertainty only.  Five settings are compared with one reference per protocol; alongside the marginal
95 % CI the output carries a Bonferroni family interval (alpha 0.05 / 5, percentiles 0.5 / 99.5, about 10
replicates per tail at 2,000 draws) and Holm-adjusted bootstrap p-values.  A bootstrap p-value cannot fall
below 2 / (B + 1) (0.0010 at B = 2,000), so a Holm-adjusted p of 5 x that floor is a bound, not an estimate;
the JSON marks such values (p_at_floor, p_two_sided_holm_is_bound, p_report).

Seed replication (section seed_replication; skip with --no-refit).  The one learner randomness in these fits
is LightGBM's column subsampling (colsample_bytree 0.9, seeded by random_state); subsample 0.9 is inert
because subsample_freq is 0 (LightGBM: bagging needs bagging_freq > 0).  The patch-holdout fit of every
setting is repeated for random_state in --refit-seeds (default ten seeds including the published 7) with
everything else unchanged -- same cached feature matrix aligned to the common rows, same parameters, same
column filter (run_window_sweep_v33.nonconstant_columns), early stopping on 15.15, scoring on 15.16 -- at
<= 4 threads.  The seed-7 refit must reproduce the cached published predictions bit for bit (checked).  Two
intervals then cover seed and match sampling together, both on the same match draws as the paired CIs:
  * single_refit: per replicate each setting gets its own uniformly drawn seed, so the interval is that of
    the difference between two independent refits (the conservative one to cite);
  * seed_mean:    per replicate each setting's seeds are resampled with replacement and averaged, the
    interval of the seed-averaged difference.
The out-of-fold protocol is not refitted (five fits per seed per setting); its CIs remain match-only.

    python scripts/window_sweep_paired_cis_v33.py            # 2,000 replicates, seed 7, 10 refit seeds, 4 threads
    python scripts/window_sweep_paired_cis_v33.py --no-refit # bootstrap over cached predictions only (~15 s)
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITEM = "C2-window-paired-cis"
SOURCE_ITEM = "A3-temporal-windows"
IN_DIR = Path("D:/LOL_Project/fusion_2615/features/tog_revision") / SOURCE_ITEM
SETTINGS = ("ctx15_bin5000", "ctx30_bin5000", "ctx60_bin5000", "ctx120_bin5000", "ctx30_bin2500", "ctx30_bin10000")
REFERENCE = "ctx30_bin5000"
N_BOOT = 2000
SEED = 7
ALPHA = 0.05
TEST_PATCH = "15.16"
AUC_TOL = 1e-9                # recorded vs recomputed AUC
REPRO_TOL = 1e-9              # 1,000-replicate prefix vs window_sweep_v33.json intervals
ROW_MEMBERS = ("row_match_id", "row_t_start_ts", "row_ref_ordinal", "y", "row_patch")
# signature fields that must not vary across settings (everything that defines rows, labels and learner)
SHARED_SIG_FIELDS = ("feature_set", "index_sha1", "item", "label", "match_list_sha1", "n_matches", "preset",
                     "preset_values", "row_key", "script_sha1", "source_sha1", "tie_policy")
REFIT_SEEDS = "7,1,2,3,4,5,6,8,9,10"   # 7 is the published seed (run_window_sweep_v33.py SEED)
REFIT_THREADS = 4
REFIT_DIRNAME = "seed_refit_c2"        # per (setting, seed) prediction caches, under the input directory
SEED_PICK_STREAM = (SEED, 1)           # SeedSequence entropy of the seed-draw stream, separate from the match draws
SWEEP_SCRIPT = PROJECT_ROOT / "scripts" / "run_window_sweep_v33.py"


class IdentityError(AssertionError):
    """Rows, labels, match groups or predictions are not what the pairing assumes."""


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise IdentityError(msg)


# ------------------------------------------------------------------ provenance

def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha1_lines(items) -> str:
    h = hashlib.sha1()
    for it in items:
        h.update(str(it).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def common_rows_sha1(groups, engage_ts, y) -> str:
    """The digest run_window_sweep_v33.py::common_rows_sha1 writes into every prediction signature."""
    return sha1_lines(f"{g}|{int(t)}|{int(v)}" for g, t, v in zip(
        np.asarray(groups).astype(str).tolist(), np.asarray(engage_ts).astype(np.int64).tolist(),
        np.asarray(y).astype(np.int64).tolist()))


def git_info() -> dict:
    def run(*cmd, strip=True):
        try:
            out = subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60).stdout
            return out.strip() if strip else out
        except Exception:
            return ""
    rel = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    status = run("status", "--porcelain", "--untracked-files=no", strip=False)   # porcelain columns: "XY path"
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked_files": [line[3:] for line in status.splitlines() if line.strip()],
            "this_script_tracked": bool(run("ls-files", rel)),
            "this_script_status": run("status", "--porcelain", "--", rel) or "clean"}


def npz_member_sha1(path: Path, members) -> dict:
    """SHA-1 of selected raw .npy members of an .npz without decompressing the rest (the feature matrix X)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for m in members:
            h = hashlib.sha1()
            with z.open(f"{m}.npy") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            out[m] = h.hexdigest()
    return out


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def log(*parts) -> None:
    print(*parts, flush=True)


# ------------------------------------------------------------------ AUC and bootstrap

def auc_plan(y: np.ndarray, score: np.ndarray):
    """Sort once; a replicate then needs only per-tie-group weight sums."""
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    new = np.ones(len(s), dtype=bool)
    new[1:] = s[1:] != s[:-1]
    tie = np.cumsum(new) - 1
    n_tie = int(tie[-1]) + 1 if len(s) else 0
    pos = (np.asarray(y)[order] == 1).astype(np.float64)
    return order, tie, n_tie, pos


def auc_weighted(plan, weight: np.ndarray | None) -> float:
    """Mann-Whitney AUC with non-negative row weights: P(score_pos > score_neg) + 0.5 P(tie)."""
    order, tie, n_tie, pos = plan
    w = np.ones(len(order)) if weight is None else np.asarray(weight, dtype=np.float64)[order]
    wp = np.bincount(tie, weights=w * pos, minlength=n_tie)
    wn = np.bincount(tie, weights=w * (1.0 - pos), minlength=n_tie)
    tot_p, tot_n = float(wp.sum()), float(wn.sum())
    if tot_p <= 0.0 or tot_n <= 0.0:
        return float("nan")
    below = np.cumsum(wn) - wn
    return float(np.dot(wp, below + 0.5 * wn) / (tot_p * tot_n))


def percentile_interval(values: np.ndarray, alpha: float) -> tuple[float, float]:
    return (float(np.percentile(values, 100.0 * alpha / 2.0)), float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0))))


def bootstrap_replicates(y: np.ndarray, preds: dict, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    """AUC of every prediction vector under the same match draw per replicate.

    Draw scheme (identical to run_window_sweep_v33.py::match_bootstrap): matches indexed in np.unique order;
    per replicate counts = bincount(rng.integers(0, G, size=G)); every row carries its match's count."""
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    inverse = inverse.reshape(-1)
    n_groups = int(inverse.max()) + 1
    names = list(preds)
    plans = {k: auc_plan(y, preds[k]) for k in names}
    point = {k: auc_weighted(plans[k], None) for k in names}
    rng = np.random.default_rng(seed)
    reps = np.full((int(n_boot), len(names)), np.nan, dtype=np.float64)
    skipped = []
    for b in range(int(n_boot)):
        counts = np.bincount(rng.integers(0, n_groups, size=n_groups), minlength=n_groups)
        w = counts[inverse].astype(np.float64)
        row = [auc_weighted(plans[k], w) for k in names]
        if any(np.isnan(v) for v in row):
            skipped.append(b)
            continue
        reps[b] = row
    return {"names": names, "point": point, "reps": reps, "skipped": skipped, "n_groups": n_groups}


def holm(pvalues: dict) -> dict:
    """Holm (1979) step-down adjustment; monotone, capped at 1."""
    keys = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def p_value_fields(v: dict, p_holm: float) -> dict:
    """Holm value plus its resolution: with B replicates p_two_sided cannot fall below 2 / (B + 1).

    At that floor no replicate reached zero, so p_two_sided and its Holm value are upper bounds, not estimates."""
    n = int(v["n_boot"])
    floor = 2.0 / (n + 1.0)
    at_floor = bool(v["p_two_sided"] <= floor * (1.0 + 1e-12))
    n_far = int(round(min(v["share_delta_leq_0"], v["share_delta_geq_0"]) * n))
    return {"p_two_sided_holm": p_holm, "p_resolution_floor": floor, "p_at_floor": at_floor,
            "n_replicates_zero_or_opposite_sign": n_far, "p_two_sided_holm_is_bound": at_floor,
            "p_report": (f"Holm-adjusted bootstrap p <= {p_holm:.4f} (resolution floor 2/(B+1) at B = {n:,}; "
                         f"{n_far} of {n:,} replicates had delta of opposite sign or zero)") if at_floor
            else f"Holm-adjusted bootstrap p = {p_holm:.4f} (B = {n:,})"}


def difference_block(d: np.ndarray, point: float, alpha: float, family_alpha: float) -> dict:
    """Percentile and Bonferroni-family intervals, tail shares and two-sided p of bootstrap differences d."""
    n = d.size
    lo, hi = percentile_interval(d, alpha)
    flo, fhi = percentile_interval(d, family_alpha)
    p_le = (1.0 + float((d <= 0).sum())) / (n + 1.0)
    p_ge = (1.0 + float((d >= 0).sum())) / (n + 1.0)
    return {"delta": float(point), "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "boot_se": float(d.std(ddof=1)), "boot_mean": float(d.mean()),
            "family_ci_lo": flo, "family_ci_hi": fhi, "family_alpha": family_alpha,
            "family_ci_excludes_zero": bool(flo > 0.0 or fhi < 0.0),
            "share_delta_leq_0": float((d <= 0).mean()), "share_delta_geq_0": float((d >= 0).mean()),
            "p_one_sided_leq_0": p_le, "p_two_sided": min(1.0, 2.0 * min(p_le, p_ge)), "n_boot": int(n)}


def summarise_protocol(boot: dict, reference: str, n_boot: int, alpha: float) -> dict:
    names, reps = boot["names"], boot["reps"]
    keep = ~np.isnan(reps).any(axis=1)
    reps = reps[keep]
    col = {k: i for i, k in enumerate(names)}
    others = [k for k in names if k != reference]
    family_alpha = alpha / len(others)
    auc = {}
    for k in names:
        v = reps[:, col[k]]
        lo, hi = percentile_interval(v, alpha)
        auc[k] = {"auc": boot["point"][k], "ci_lo": lo, "ci_hi": hi, "boot_se": float(v.std(ddof=1)),
                  "boot_mean": float(v.mean()), "n_boot": int(v.size)}
    paired, p_two = {}, {}
    for k in others:
        d = reps[:, col[k]] - reps[:, col[reference]]
        key = f"{k} - {reference}"
        lo, hi = percentile_interval(d, alpha)
        flo, fhi = percentile_interval(d, family_alpha)
        n = d.size
        p_le = (1.0 + float((d <= 0).sum())) / (n + 1.0)
        p_ge = (1.0 + float((d >= 0).sum())) / (n + 1.0)
        p_two[key] = min(1.0, 2.0 * min(p_le, p_ge))
        paired[key] = {
            "setting": k, "reference": reference,
            "delta": boot["point"][k] - boot["point"][reference], "ci_lo": lo, "ci_hi": hi,
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "boot_se": float(d.std(ddof=1)), "boot_mean": float(d.mean()),
            "family_ci_lo": flo, "family_ci_hi": fhi, "family_alpha": family_alpha,
            "family_ci_excludes_zero": bool(flo > 0.0 or fhi < 0.0),
            "share_delta_leq_0": float((d <= 0).mean()), "share_delta_geq_0": float((d >= 0).mean()),
            "p_one_sided_leq_0": p_le, "p_two_sided": p_two[key],
            "marginal_ci_overlap": bool(auc[k]["ci_lo"] <= auc[reference]["ci_hi"]
                                        and auc[reference]["ci_lo"] <= auc[k]["ci_hi"]),
            "corr_replicate_auc": float(np.corrcoef(reps[:, col[k]], reps[:, col[reference]])[0, 1]),
            "n_boot": int(n),
        }
    for key, p in holm(p_two).items():
        paired[key].update(p_value_fields(paired[key], p))
    return {"n_matches": boot["n_groups"], "n_boot_requested": int(n_boot),
            "n_boot_skipped_single_class": len(boot["skipped"]), "auc": auc, "paired": paired}


def prefix_matches_sweep(boot: dict, sweep_boot: dict, reference: str, n_prefix: int) -> dict:
    """Percentile intervals of the first n_prefix replicates vs the intervals window_sweep_v33.json stored."""
    names, reps = boot["names"], boot["reps"][:n_prefix]
    reps = reps[~np.isnan(reps).any(axis=1)]
    col = {k: i for i, k in enumerate(names)}
    alpha = float(sweep_boot["alpha"])
    diffs = []
    for k in names:
        lo, hi = percentile_interval(reps[:, col[k]], alpha)
        ref = sweep_boot["auc"][k]
        diffs += [abs(lo - ref["ci_lo"]), abs(hi - ref["ci_hi"]), abs(boot["point"][k] - ref["auc"])]
    for k in names:
        if k == reference:
            continue
        lo, hi = percentile_interval(reps[:, col[k]] - reps[:, col[reference]], alpha)
        ref = sweep_boot["paired"][f"{k} - {reference}"]
        diffs += [abs(lo - ref["ci_lo"]), abs(hi - ref["ci_hi"])]
    max_diff = float(max(diffs))
    return {"n_prefix": int(n_prefix), "sweep_n_boot": int(sweep_boot["n_boot_requested"]),
            "sweep_seed": int(sweep_boot["seed"]), "max_abs_difference": max_diff, "passed": max_diff <= REPRO_TOL}


# ------------------------------------------------------------------ identity checks

def load_common(in_dir: Path) -> dict:
    with np.load(in_dir / "common_rows.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def setting_rows(in_dir: Path, name: str) -> dict:
    """Row metadata of the setting's own feature cache (the matrix X is never decompressed)."""
    with np.load(in_dir / f"setting_{name}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in ROW_MEMBERS}


def verify(in_dir: Path, sweep: dict, reference: str) -> tuple[dict, dict, dict]:
    """Returns (common arrays, predictions {protocol: {setting: vector}}, check record)."""
    from sklearn.metrics import roc_auc_score   # independent AUC for the recorded-AUC check

    rec: dict = {"common_rows": {}, "settings": {}}
    c = load_common(in_dir)
    mid, ts, ordn = c["match_id"].astype(str), c["engage_ts"].astype(np.int64), c["ref_ordinal"].astype(np.int64)
    y, patch = c["y"].astype(np.int8), c["patch"].astype(str)
    n = len(y)
    check(all(len(c[k]) == n for k in c), "common_rows.npz arrays differ in length")
    keys = list(zip(mid.tolist(), ts.tolist(), ordn.tolist()))
    check(len(set(keys)) == n, "common rows: duplicate (match_id, engage_ts, ref_ordinal) keys")
    check(keys == sorted(keys), "common rows are not in the sorted-key order run_window_sweep_v33.py writes")
    check(set(np.unique(y).tolist()) == {0, 1}, "labels are not binary 0/1")
    digest = common_rows_sha1(mid, ts, y)
    check(digest == sweep["common_rows"]["sha1"], "common_rows.npz does not hash to window_sweep_v33.json common_rows.sha1")
    check(n == sweep["common_rows"]["n_rows"], "common row count differs from window_sweep_v33.json")
    n_matches = len(set(mid.tolist()))
    check(n_matches == sweep["common_rows"]["n_matches"], "common match count differs from window_sweep_v33.json")
    patches_per_match = {}
    for m_, p_ in zip(mid.tolist(), patch.tolist()):
        patches_per_match.setdefault(m_, set()).add(p_)
    multi_patch = sum(len(v) > 1 for v in patches_per_match.values())
    check(multi_patch == 0, f"{multi_patch} matches carry rows from more than one patch")
    te = patch == TEST_PATCH
    split = sweep["split"]
    check(split["test"] == TEST_PATCH and int(te.sum()) == split["rows"]["test"], "patch-holdout test rows differ")
    check(len(set(mid[te].tolist())) == split["matches"]["test"], "patch-holdout test match count differs")
    rec["common_rows"] = {"n_rows": n, "n_matches": n_matches, "sha1_recomputed": digest,
                          "keys_unique_and_sorted": True, "matches_spanning_patches": multi_patch,
                          "patch_test_rows": int(te.sum()), "patch_test_matches": int(len(set(mid[te].tolist()))),
                          "positive_rate": float(y.mean()), "positive_rate_test": float(y[te].mean())}

    with np.load(in_dir / "window_sweep_v33.preds.npz", allow_pickle=False) as z:
        store = {k: z[k] for k in z.files}
    check(np.array_equal(store["match_id"].astype(str), mid) and np.array_equal(store["engage_ts"], ts)
          and np.array_equal(store["y"].astype(np.int8), y) and np.array_equal(store["patch"].astype(str), patch),
          "window_sweep_v33.preds.npz row metadata differs from common_rows.npz")
    check(np.array_equal(store["patch_test_index"], np.flatnonzero(te)), "patch_test_index is not the 15.16 rows in common order")
    rec["common_rows"]["consolidated_preds_rows_identical"] = True

    preds = {"patch": {}, "oof": {}}
    aligned_all: dict = {}
    sigs: dict = {}
    for name in SETTINGS:
        r: dict = {}
        # (1) the setting cache's own rows, aligned onto the common rows
        rows = setting_rows(in_dir, name)
        own = list(zip(rows["row_match_id"].astype(str).tolist(), rows["row_t_start_ts"].astype(np.int64).tolist(),
                       rows["row_ref_ordinal"].astype(np.int64).tolist()))
        check(len(set(own)) == len(own), f"{name}: duplicate row keys in the setting cache")
        pos = {k: i for i, k in enumerate(own)}
        missing = sum(k not in pos for k in keys)
        check(missing == 0, f"{name}: {missing} common rows absent from the setting cache")
        take = np.fromiter((pos[k] for k in keys), dtype=np.int64, count=n)
        aligned = {"match_id": rows["row_match_id"].astype(str)[take],
                   "engage_ts": rows["row_t_start_ts"].astype(np.int64)[take],
                   "ref_ordinal": rows["row_ref_ordinal"].astype(np.int64)[take],
                   "y": rows["y"].astype(np.int8)[take], "patch": rows["row_patch"].astype(str)[take]}
        check(np.array_equal(aligned["match_id"], mid) and np.array_equal(aligned["engage_ts"], ts)
              and np.array_equal(aligned["ref_ordinal"], ordn), f"{name}: aligned row keys differ from common rows")
        check(np.array_equal(aligned["y"], y), f"{name}: labels differ from the common-row labels")
        check(np.array_equal(aligned["patch"], patch), f"{name}: patches differ from the common rows")
        aligned_all[name] = aligned
        r["setting_cache"] = {"n_rows_built": len(own), "common_rows_found": n,
                              "row_order_labels_groups_patch_equal_common": True,
                              "label_mismatch_vs_common": int((aligned["y"] != y).sum())}

        # (2) signatures: same corpus, index, preset, label and learner script for every setting
        s_side = read_json(in_dir / f"setting_{name}.json")["signature"]
        check(s_side["setting"] == name and s_side["kind"] == "setting", f"{name}: setting sidecar names another setting")
        sigs[name] = s_side
        for proto in ("patch", "oof"):
            p_side = read_json(in_dir / f"pred_{proto}_{name}.json")
            sig = p_side["signature"]
            check(sig["kind"] == f"pred_{proto}" and sig["setting"] == name, f"{name}/{proto}: sidecar kind or setting")
            check(sig["common_rows_sha1"] == digest, f"{name}/{proto}: predictions written under another common-row set")
            for f in SHARED_SIG_FIELDS + ("ctx_sec", "bin_ms"):
                check(sig[f] == s_side[f], f"{name}/{proto}: signature field {f} differs from its setting cache")
            with np.load(in_dir / f"pred_{proto}_{name}.npz", allow_pickle=False) as z:
                check(z.files == ["pred"], f"{name}/{proto}: unexpected members {z.files}")
                p = z["pred"].astype(np.float64)
            check(bool(np.isfinite(p).all()), f"{name}/{proto}: non-finite predictions")
            check(np.array_equal(p, store[f"{proto}__pred__{name}"]), f"{name}/{proto}: differs from consolidated preds")
            if proto == "patch":
                check(sig["split"]["test"] == TEST_PATCH and sig["split"]["rows"]["test"] == int(te.sum()),
                      f"{name}/patch: split differs")
                check(len(p) == int(te.sum()), f"{name}/patch: {len(p)} predictions for {int(te.sum())} test rows")
                y_eval, recorded = y[te], float(p_side["fit"]["test_auc"])
            else:
                check(sig["split"] == "GroupKFold(5)", f"{name}/oof: split {sig['split']}")
                check(len(p) == n, f"{name}/oof: {len(p)} predictions for {n} rows")
                check(sum(f["test"] for f in p_side["fit"]["folds"]) == n, f"{name}/oof: fold sizes do not cover the rows")
                y_eval, recorded = y, float(p_side["fit"]["auc"])
            recomputed = float(roc_auc_score(y_eval, p))
            check(abs(recomputed - recorded) <= AUC_TOL,
                  f"{name}/{proto}: AUC {recomputed} from stored labels != recorded {recorded} (row order broken?)")
            preds[proto][name] = p
            r[proto] = {"n_predictions": int(len(p)), "recorded_auc": recorded, "recomputed_auc_sklearn": recomputed,
                        "abs_diff": abs(recomputed - recorded), "sidecar_common_rows_sha1_matches": True,
                        "bit_identical_to_consolidated_preds": True}
        rec["settings"][name] = r

    # (3) every setting vs the reference: identical rows, labels, groups; shared signature; different predictions
    ref_sig, aligned_ref = sigs[reference], aligned_all[reference]
    for name in SETTINGS:
        diff_fields = [f for f in SHARED_SIG_FIELDS if sigs[name][f] != ref_sig[f]]
        check(not diff_fields, f"{name}: signature fields {diff_fields} differ from the reference setting")
        pair = {"shared_signature_fields_equal": True}
        for field in ("match_id", "engage_ts", "ref_ordinal", "y", "patch"):
            check(np.array_equal(aligned_all[name][field], aligned_ref[field]),
                  f"{name}: {field} differs from the reference after alignment")
        pair["row_order_equal"] = pair["labels_equal"] = pair["match_groups_equal"] = pair["patch_equal"] = True
        pair["patch_test_groups_equal"] = bool(np.array_equal(aligned_all[name]["match_id"][te], aligned_ref["match_id"][te]))
        check(pair["patch_test_groups_equal"], f"{name}: test-row match groups differ from the reference")
        for proto in ("patch", "oof"):
            same = bool(np.array_equal(preds[proto][name], preds[proto][reference]))
            check(name == reference or not same, f"{name}/{proto}: predictions identical to the reference")
            pair[f"{proto}_predictions_differ_from_reference"] = (not same) if name != reference else None
        rec["settings"][name]["vs_reference"] = pair
    rec["passed"] = True
    return {"match_id": mid, "engage_ts": ts, "ref_ordinal": ordn, "y": y, "patch": patch, "test_mask": te,
            "keys": keys, "sha1": digest}, preds, rec


def input_manifest(in_dir: Path) -> dict:
    files = ["common_rows.npz", "window_sweep_v33.json", "window_sweep_v33.preds.npz"]
    for name in SETTINGS:
        files += [f"pred_patch_{name}.npz", f"pred_patch_{name}.json", f"pred_oof_{name}.npz", f"pred_oof_{name}.json",
                  f"setting_{name}.json"]
    out = {f: {"sha1": file_sha1(in_dir / f), "bytes": (in_dir / f).stat().st_size} for f in files}
    for name in SETTINGS:
        path = in_dir / f"setting_{name}.npz"
        out[f"setting_{name}.npz"] = {"bytes": path.stat().st_size, "hashed": "row-metadata members only",
                                      "member_sha1": npz_member_sha1(path, ROW_MEMBERS),
                                      "X_npy_zip_crc32": x_member_crc32(path)}
    return out


def x_member_crc32(path: Path) -> str:
    """CRC-32 of the X.npy member as stored in the .npz central directory (identifies X without reading 1.3 GB)."""
    with zipfile.ZipFile(path) as z:
        return f"{z.getinfo('X.npy').CRC:08x}"


def manifest_counts(manifest: dict) -> dict:
    return {"n_entries": len(manifest), "n_fully_hashed_sha1": sum("sha1" in v for v in manifest.values()),
            "n_row_member_hashed": sum("member_sha1" in v for v in manifest.values()),
            "note": "setting_<name>.npz feature caches are identified by the SHA-1 of their row-metadata members "
                    "and the stored CRC-32 of X.npy; every other input is hashed in full"}


# ------------------------------------------------------------------ seed replication (patch holdout)

def import_sweep_module():
    """scripts/run_window_sweep_v33.py as a module (its column filter and parameters); it defines, it does not run."""
    scripts_dir = str(SWEEP_SCRIPT.parent)
    for p in (str(PROJECT_ROOT), scripts_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_window_sweep_v33 as sw   # noqa: E402  (main() is guarded by __name__)
    return sw


def load_common_X(in_dir: Path, name: str, keys: list) -> np.ndarray:
    """The setting's cached feature matrix, rows aligned to the common rows (no copy when already aligned)."""
    with np.load(in_dir / f"setting_{name}.npz", allow_pickle=False) as z:
        own = list(zip(z["row_match_id"].astype(str).tolist(), z["row_t_start_ts"].astype(np.int64).tolist(),
                       z["row_ref_ordinal"].astype(np.int64).tolist()))
        pos = {k: i for i, k in enumerate(own)}
        take = np.fromiter((pos[k] for k in keys), dtype=np.int64, count=len(keys))
        X = z["X"]
    if X.shape[0] == len(take) and np.array_equal(take, np.arange(len(take))):
        return X
    return X[take]


def fit_patch_seeded(X, y, tr, va, te, params: dict, es_rounds: int, n_jobs: int, nonconstant_columns) -> dict:
    """run_window_sweep_v33.fit_patch_lgbm with the parameter dict passed in (only random_state varies)."""
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import roc_auc_score
    started = time.time()
    tr_i, va_i, te_i = (np.flatnonzero(m) for m in (tr, va, te))
    keep = nonconstant_columns(X, tr_i)
    model = LGBMClassifier(**params, n_jobs=int(n_jobs))
    model.fit(X[np.ix_(tr_i, keep)], y[tr_i], eval_set=[(X[np.ix_(va_i, keep)], y[va_i])], eval_metric="auc",
              callbacks=[early_stopping(int(es_rounds), verbose=False), log_evaluation(0)])
    pred_val = model.predict_proba(X[np.ix_(va_i, keep)])[:, 1]
    pred_test = model.predict_proba(X[np.ix_(te_i, keep)])[:, 1]
    got = model.get_params()
    return {"pred_test": pred_test.astype(np.float64), "val_auc": float(roc_auc_score(y[va_i], pred_val)),
            "test_auc": float(roc_auc_score(y[te_i], pred_test)),
            "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
            "n_features_nonconstant": int(len(keep)), "subsample_freq": int(got.get("subsample_freq", 0)),
            "random_state": int(got["random_state"]), "seconds": round(time.time() - started, 1)}


def run_refits(in_dir: Path, sweep: dict, common: dict, keys: list, seeds: list, n_jobs: int, digest: str) -> dict:
    """Patch-holdout refit of every setting for every seed, cached per (setting, seed) under in_dir/REFIT_DIRNAME."""
    import inspect
    import warnings
    import lightgbm
    sw = import_sweep_module()
    sweep_sha1 = file_sha1(SWEEP_SCRIPT)
    check(sweep_sha1 == sweep["script_sha1"], f"{SWEEP_SCRIPT.name} SHA-1 {sweep_sha1} != the {sweep['script_sha1']} "
                                              "that produced the cached predictions")
    base = dict(sweep["lgbm_params"])
    check(base == sw.LGBM_PARAMS, "window_sweep_v33.json lgbm_params differ from run_window_sweep_v33.LGBM_PARAMS")
    es_rounds = int(sweep["patch_early_stopping_rounds"])
    check(es_rounds == sw.PATCH_EARLY_STOPPING, "early-stopping rounds differ from run_window_sweep_v33")
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    out_dir = in_dir / REFIT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    y, patch = common["y"], common["patch"]
    split = sweep["split"]
    tr, va, te = patch == split["train"], patch == split["val"], patch == split["test"]
    fn_sha1 = hashlib.sha1(inspect.getsource(fit_patch_seeded).encode("utf-8")).hexdigest()
    preds: dict = {}
    fits: dict = {}
    caches: dict = {}
    for name in SETTINGS:
        preds[name], fits[name] = {}, {}
        setting_npz = in_dir / f"setting_{name}.npz"
        X = None
        for s in seeds:
            params = {**base, "random_state": int(s)}
            sig = {"item": ITEM, "kind": "seed_refit_patch", "setting": name, "seed": int(s), "lgbm_params": params,
                   "early_stopping": es_rounds, "n_jobs": int(n_jobs), "lightgbm": lightgbm.__version__,
                   "common_rows_sha1": digest, "split": split, "setting_json_sha1": file_sha1(in_dir / f"setting_{name}.json"),
                   "setting_X_npy_zip_crc32": x_member_crc32(setting_npz), "sweep_script_sha1": sweep_sha1,
                   "fit_fn_sha1": fn_sha1, "column_filter": "run_window_sweep_v33.nonconstant_columns on train-patch rows"}
            npz_path, json_path = out_dir / f"pred_patch_{name}_seed{s}.npz", out_dir / f"pred_patch_{name}_seed{s}.json"
            cached = None
            if npz_path.exists() and json_path.exists():
                side = read_json(json_path)
                if side.get("signature") == json.loads(json.dumps(sig)):
                    with np.load(npz_path, allow_pickle=False) as z:
                        cached = z["pred"].astype(np.float64)
                    if len(cached) != int(te.sum()):
                        cached = None
                    else:
                        fit = side["fit"]
            if cached is None:
                if X is None:
                    t0 = time.time()
                    X = load_common_X(in_dir, name, keys)
                    log(f"  [refit] {name}: X {X.shape} loaded in {time.time() - t0:.1f}s")
                fit = fit_patch_seeded(X, y, tr, va, te, params, es_rounds, n_jobs, sw.nonconstant_columns)
                cached = fit.pop("pred_test")
                check(fit["subsample_freq"] == 0 and fit["random_state"] == int(s), f"{name}/seed {s}: unexpected params")
                tmp = npz_path.with_name(npz_path.stem + ".partial.npz")
                np.savez(tmp, pred=cached)
                os.replace(tmp, npz_path)
                write_json_atomic(json_path, {"signature": sig, "fit": fit})
                log(f"  [refit] {name} seed {s:>2}: val {fit['val_auc']:.4f} test {fit['test_auc']:.4f} "
                    f"best_iter {fit['best_iteration']} ({fit['seconds']}s)")
            else:
                log(f"  [refit] {name} seed {s:>2}: cached, test {fit['test_auc']:.4f}")
            check(bool(np.isfinite(cached).all()), f"{name}/seed {s}: non-finite predictions")
            preds[name][int(s)] = cached
            fits[name][int(s)] = fit
            caches[f"{REFIT_DIRNAME}/{npz_path.name}"] = {"sha1": file_sha1(npz_path), "bytes": npz_path.stat().st_size}
        del X
    return {"preds": preds, "fits": fits, "caches": caches, "sweep_script_sha1": sweep_sha1, "fit_fn_sha1": fn_sha1,
            "lightgbm": lightgbm.__version__, "base_params": base, "early_stopping": es_rounds}


def seed_analysis(common: dict, refit: dict, published: dict, main_patch_boot: dict, seeds: list, reference: str,
                  n_boot: int, seed: int, alpha: float) -> dict:
    """Seed variability of the patch-holdout AUCs and paired differences, and intervals over seeds and matches."""
    te = common["test_mask"]
    y, groups = common["y"][te], common["match_id"][te]
    K, S = len(seeds), len(SETTINGS)
    j7 = seeds.index(SEED)
    rec: dict = {"published_seed": SEED, "published_seed_reproduced": {}}
    # (1) the refit at the published seed must be the published model
    for name in SETTINGS:
        same = bool(np.array_equal(refit["preds"][name][SEED], published[name]))
        diff = float(np.abs(refit["preds"][name][SEED] - published[name]).max())
        check(same, f"{name}: seed-{SEED} refit differs from the cached published predictions (max abs {diff})")
        rec["published_seed_reproduced"][name] = {"bit_identical": same, "max_abs_pred_diff": diff}
    # (2) seeds must change the model (otherwise column subsampling is inert and the section is vacuous)
    for name in SETTINGS:
        distinct = len({refit["preds"][name][s].tobytes() for s in seeds})
        check(distinct == K, f"{name}: only {distinct} distinct prediction vectors over {K} seeds")
    # (3) match draws identical to the paired CIs (same stream), seeds x settings AUC per replicate
    vecs = {f"{name}@{s}": refit["preds"][name][s] for name in SETTINGS for s in seeds}
    boot = bootstrap_replicates(y, vecs, groups, n_boot, seed)
    check(not boot["skipped"], "single-class bootstrap replicates in the seed analysis")
    A = boot["reps"].reshape(int(n_boot), S, K)
    check(bool(np.array_equal(A[:, :, j7], main_patch_boot["reps"])),
          "seed-7 replicate AUCs differ from the paired-CI replicates (match draws not shared)")
    rec["match_draws_identical_to_paired_cis"] = True
    point = np.array([[boot["point"][f"{name}@{s}"] for s in seeds] for name in SETTINGS])   # S x K
    col = {name: i for i, name in enumerate(SETTINGS)}
    r = col[reference]
    per_setting = {}
    for name in SETTINGS:
        a = point[col[name]]
        per_setting[name] = {
            "test_auc_by_seed": {str(s): float(a[j]) for j, s in enumerate(seeds)},
            "val_auc_by_seed": {str(s): float(refit["fits"][name][s]["val_auc"]) for s in seeds},
            "best_iteration_by_seed": {str(s): int(refit["fits"][name][s]["best_iteration"]) for s in seeds},
            "test_auc_mean": float(a.mean()), "test_auc_sd": float(a.std(ddof=1)), "test_auc_min": float(a.min()),
            "test_auc_max": float(a.max()), "published_seed_test_auc": float(a[j7]),
            "published_seed_rank_among_seeds_desc": int((a > a[j7]).sum()) + 1}
    # seed picks: independent stream, one row per replicate
    pick_rng = np.random.default_rng(np.random.SeedSequence(list(SEED_PICK_STREAM)))
    single_pick = pick_rng.integers(0, K, size=(int(n_boot), S))            # one seed per setting per replicate
    mean_pick = pick_rng.integers(0, K, size=(int(n_boot), S, K))           # K seeds with replacement per setting
    rows = np.arange(int(n_boot))
    single_auc = A[rows[:, None], np.arange(S)[None, :], single_pick]      # B x S
    mean_auc = np.take_along_axis(A, mean_pick, axis=2).mean(axis=2)        # B x S
    others = [k for k in SETTINGS if k != reference]
    family_alpha = alpha / len(others)
    paired, p_single, p_mean = {}, {}, {}
    for name in others:
        i = col[name]
        key = f"{name} - {reference}"
        d_seed = point[i] - point[r]
        per_seed_ci = {}
        for j, s in enumerate(seeds):
            lo, hi = percentile_interval(A[:, i, j] - A[:, r, j], alpha)
            per_seed_ci[str(s)] = {"delta": float(d_seed[j]), "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)}
        seed_mean_point = float(point[i].mean() - point[r].mean())
        single = difference_block(single_auc[:, i] - single_auc[:, r], seed_mean_point, alpha, family_alpha)
        smean = difference_block(mean_auc[:, i] - mean_auc[:, r], seed_mean_point, alpha, family_alpha)
        p_single[key], p_mean[key] = single["p_two_sided"], smean["p_two_sided"]
        match_se_pub = float((A[:, i, j7] - A[:, r, j7]).std(ddof=1))
        paired[key] = {
            "setting": name, "reference": reference,
            "same_seed_delta_by_seed": {str(s): float(d_seed[j]) for j, s in enumerate(seeds)},
            "same_seed_delta_mean": float(d_seed.mean()), "same_seed_delta_sd": float(d_seed.std(ddof=1)),
            "same_seed_delta_min": float(d_seed.min()), "same_seed_delta_max": float(d_seed.max()),
            "n_seeds_delta_gt_0": int((d_seed > 0).sum()), "n_seeds": K,
            "published_seed_delta": float(d_seed[j7]),
            "per_seed_match_ci": per_seed_ci,
            "n_seeds_match_ci_excludes_zero": int(sum(v["ci_excludes_zero"] for v in per_seed_ci.values())),
            "match_bootstrap_se_published_seed": match_se_pub,
            "seed_sd_of_independent_refit_difference": float(np.sqrt(point[i].var(ddof=1) + point[r].var(ddof=1))),
            "single_refit": single, "seed_mean": smean}
    for key, p in holm(p_single).items():
        paired[key]["single_refit"].update(p_value_fields(paired[key]["single_refit"], p))
    for key, p in holm(p_mean).items():
        paired[key]["seed_mean"].update(p_value_fields(paired[key]["seed_mean"], p))
    rec.update({
        "protocol": "patch holdout (fit 15.14, early stopping 15.15, score 15.16), same common rows as protocols.patch",
        "seeds": [int(s) for s in seeds], "n_threads": REFIT_THREADS,
        "varied": "LightGBM random_state only; it seeds colsample_bytree = 0.9 column subsampling (subsample = 0.9 "
                  "is inert because subsample_freq = 0, checked on every fitted model)",
        "held_fixed": "rows, labels, cached feature matrices, parameters, column filter, early-stopping rule, split",
        "estimands": {
            "same_seed_delta": "AUC(setting, seed s) - AUC(reference, seed s) on the 15.16 rows; seed labels do not "
                               "couple the two models, so this is a descriptive per-seed table",
            "single_refit": "interval of AUC(setting, one random seed) - AUC(reference, another random seed): per "
                            "replicate the match draw of the paired CIs plus an independent uniform seed per setting; "
                            "point = difference of seed-mean AUCs",
            "seed_mean": "interval of the difference of seed-averaged AUCs: per replicate the same match draw plus "
                         f"{K} seeds drawn with replacement per setting and averaged"},
        "seed_pick_rng": f"numpy.random.default_rng(SeedSequence({list(SEED_PICK_STREAM)})); integers(0, K, (B, S)) "
                         "for single_refit, then integers(0, K, (B, S, K)) for seed_mean",
        "n_boot": int(n_boot), "match_draw_seed": int(seed),
        "per_setting": per_setting, "paired": paired})
    return rec


# ------------------------------------------------------------------ main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=IN_DIR)
    ap.add_argument("--output", type=Path, default=None, help="default <in-dir>/window_sweep_paired_cis_v33.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--reference", default=REFERENCE, choices=SETTINGS)
    ap.add_argument("--refit-seeds", default=REFIT_SEEDS, help="comma list of LightGBM random_state values; must include 7")
    ap.add_argument("--refit-threads", type=int, default=REFIT_THREADS)
    ap.add_argument("--no-refit", action="store_true", help="skip the seed replication (bootstrap over cached predictions only)")
    return ap.parse_args(argv)


def fmt(v: float) -> str:
    return f"{v:+.4f}"


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    in_dir = args.in_dir
    output = args.output or in_dir / "window_sweep_paired_cis_v33.json"
    sweep = read_json(in_dir / "window_sweep_v33.json")
    check(list(sweep["settings_order"]) == list(SETTINGS), f"settings order {sweep['settings_order']}")
    check(sweep["reference"] == REFERENCE, f"window_sweep_v33.json reference {sweep['reference']} != {REFERENCE}")

    log(f"[{ITEM}] verifying rows, labels, match groups and predictions in {in_dir}")
    common, preds, checks = verify(in_dir, sweep, args.reference)
    log("  identity checks passed")
    manifest = input_manifest(in_dir)

    protocols, boots = {}, {}
    for proto in ("patch", "oof"):
        mask = common["test_mask"] if proto == "patch" else np.ones(len(common["y"]), dtype=bool)
        t0 = time.time()
        boot = bootstrap_replicates(common["y"][mask], preds[proto], common["match_id"][mask], args.n_boot, args.seed)
        boots[proto] = boot
        summary = summarise_protocol(boot, args.reference, args.n_boot, ALPHA)
        sweep_boot = sweep["protocols"][proto]["bootstrap"]
        repro = None
        if args.seed == int(sweep_boot["seed"]) and args.n_boot >= int(sweep_boot["n_boot_requested"]) \
                and args.reference == sweep["reference"]:
            repro = prefix_matches_sweep(boot, sweep_boot, args.reference, int(sweep_boot["n_boot_requested"]))
            check(repro["passed"], f"{proto}: first {repro['n_prefix']} replicates do not reproduce window_sweep_v33.json "
                                   f"(max abs diff {repro['max_abs_difference']})")
        protocols[proto] = {
            "rows": "patch-holdout test rows (patch 15.16; model fit on 15.14, early stopping on 15.15)" if proto == "patch"
            else "all common rows, out-of-fold predictions from match-grouped GroupKFold(5)",
            "n_rows": int(mask.sum()), "positive_rate": float(common["y"][mask].mean()), **summary,
            "reproduces_window_sweep_v33_bootstrap": repro, "seconds": round(time.time() - t0, 1)}
        log(f"[{proto}] {int(mask.sum())} rows, {summary['n_matches']} matches, {args.n_boot} replicates "
            f"({protocols[proto]['seconds']}s); 1,000-replicate prefix reproduces sweep: {repro and repro['passed']}")
        for k, v in summary["auc"].items():
            log(f"  {k:15s} AUC {v['auc']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
        for k, v in summary["paired"].items():
            log(f"  {k:31s} {fmt(v['delta'])} [{fmt(v['ci_lo'])}, {fmt(v['ci_hi'])}]  family "
                f"[{fmt(v['family_ci_lo'])}, {fmt(v['family_ci_hi'])}]  {v['p_report']}")

    seed_rep = None
    if args.no_refit:
        seed_rep = {"skipped": "--no-refit"}
    else:
        seeds = [int(s) for s in str(args.refit_seeds).split(",") if s.strip()]
        check(SEED in seeds and len(set(seeds)) == len(seeds) and len(seeds) >= 3,
              f"--refit-seeds must hold >= 3 distinct seeds including the published {SEED}: {seeds}")
        check(args.reference == REFERENCE and args.seed == SEED,
              "the seed replication shares the published reference and match-draw seed")
        t0 = time.time()
        log(f"[seed replication] patch holdout, {len(SETTINGS)} settings x {len(seeds)} seeds, {args.refit_threads} threads")
        refit = run_refits(in_dir, sweep, common, common["keys"], seeds, args.refit_threads, common["sha1"])
        seed_rep = seed_analysis(common, refit, preds["patch"], boots["patch"], seeds, args.reference,
                                 args.n_boot, args.seed, ALPHA)
        seed_rep["code"] = {"sweep_script": SWEEP_SCRIPT.relative_to(PROJECT_ROOT).as_posix(),
                            "sweep_script_sha1": refit["sweep_script_sha1"], "fit_fn_sha1": refit["fit_fn_sha1"],
                            "lightgbm": refit["lightgbm"], "base_params": refit["base_params"],
                            "early_stopping_rounds": refit["early_stopping"]}
        seed_rep["refit_caches"] = refit["caches"]
        seed_rep["seconds"] = round(time.time() - t0, 1)
        for name, v in seed_rep["per_setting"].items():
            log(f"  {name:15s} test AUC over seeds {v['test_auc_mean']:.4f} sd {v['test_auc_sd']:.4f} "
                f"[{v['test_auc_min']:.4f}, {v['test_auc_max']:.4f}]")
        for key, v in seed_rep["paired"].items():
            s1, sm = v["single_refit"], v["seed_mean"]
            log(f"  {key:31s} same-seed {fmt(v['same_seed_delta_mean'])} sd {v['same_seed_delta_sd']:.4f} "
                f"({v['n_seeds_delta_gt_0']}/{v['n_seeds']} > 0, {v['n_seeds_match_ci_excludes_zero']} CIs excl. 0)  "
                f"single-refit [{fmt(s1['ci_lo'])}, {fmt(s1['ci_hi'])}] fam [{fmt(s1['family_ci_lo'])}, {fmt(s1['family_ci_hi'])}]  "
                f"seed-mean [{fmt(sm['ci_lo'])}, {fmt(sm['ci_hi'])}] fam [{fmt(sm['family_ci_lo'])}, {fmt(sm['family_ci_hi'])}]")

    settings_meta = {k: {"ctx_sec": sweep["settings"][k]["ctx_sec"], "bin_ms": sweep["settings"][k]["bin_ms"],
                         "L": sweep["settings"][k]["L"]} for k in SETTINGS}
    result = {
        "item": ITEM, "source_item": SOURCE_ITEM, "script": "scripts/window_sweep_paired_cis_v33.py",
        "script_sha1": file_sha1(Path(__file__)), "git": git_info(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "thread_limits": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "in_dir": str(in_dir), "reference": args.reference, "settings": settings_meta,
        "source_run": {"script": sweep["script"], "script_sha1": sweep["script_sha1"], "git": sweep["git"],
                       "preset": sweep["preset"], "label_key": sweep["label_key"], "n_matches_sampled": sweep["n_matches_sampled"],
                       "common_rows_sha1": sweep["common_rows"]["sha1"], "split": sweep["split"],
                       "lgbm_params": sweep["lgbm_params"]},
        "method": {
            "estimator": "ROC AUC as the weighted Mann-Whitney statistic, ties counted one half",
            "resampling": "match-clustered bootstrap: matches drawn with replacement, each drawn match enters with its "
                          "multiplicity (weights), rows never resampled individually",
            "pairing": "one match draw per replicate shared by all six prediction vectors; paired difference = "
                       "AUC(setting) - AUC(reference) within the replicate",
            "n_boot": int(args.n_boot), "seed": int(args.seed), "rng": "numpy.random.default_rng(seed); per replicate "
            "integers(0, n_matches, size=n_matches); matches indexed in numpy.unique order",
            "interval": f"percentile, {100 * (1 - ALPHA):.0f} % two-sided",
            "family_interval": f"Bonferroni percentile interval at alpha {ALPHA} / {len(SETTINGS) - 1} per protocol",
            "p_values": "p_one_sided_leq_0 = (1 + #{delta* <= 0}) / (B + 1); p_two_sided = min(1, 2 min(p_leq, p_geq)); "
                        "Holm step-down across the five comparisons of a protocol",
            "protocols": {"patch": "patch-holdout test rows", "oof": "match-grouped 5-fold out-of-fold rows"},
            "p_value_resolution": "p_two_sided >= 2 / (B + 1); where p_at_floor is true no replicate reached zero and "
                                  "p_two_sided / p_two_sided_holm are upper bounds (see p_report), not estimates",
            "held_fixed": "protocols.*: the fitted LightGBM models; those CIs cover match sampling of the evaluation rows "
                          "only.  seed_replication: refits over LightGBM random_state on the patch holdout, with intervals "
                          "over seeds and matches; the OOF protocol is not refitted",
            "references": ["efron1993bootstrap (ch. 13)", "Field & Welsh 2007, JRSS-B 69(3) sec. 2.1 (not in bib)",
                           "Hanley & McNeil 1982, Radiology 143(1) eq. 1 (not in bib)", "Holm 1979, Scand. J. Stat. 6(2) (not in bib)"],
        },
        "checks": checks, "inputs": manifest, "inputs_counts": manifest_counts(manifest), "protocols": protocols,
        "seed_replication": seed_rep,
        "notes": [
            "The per-setting pred_*.npz files hold only a prediction vector; their row order is established by (a) the "
            "common_rows_sha1 under which run_window_sweep_v33.py wrote them, (b) exact reproduction of each recorded "
            "AUC from the common-row labels, and (c) bit identity with window_sweep_v33.preds.npz.",
            "Row identity across settings is checked on each setting cache's own row metadata (setting_<name>.npz "
            "members row_match_id, row_t_start_ts, row_ref_ordinal, y, row_patch), aligned onto common_rows.npz.",
            "Setting feature caches (141-437 MB) are hashed by their row-metadata members only (plus the stored CRC-32 "
            "of X.npy); their JSON sidecars are hashed in full.  inputs_counts gives the tally.",
            "seed_replication refit predictions are cached under <in_dir>/seed_refit_c2/ (SHA-1s in "
            "seed_replication.refit_caches); a cache is reused only when its signature (setting, seed, parameters, "
            "LightGBM version, thread count, common-row SHA-1, setting sidecar SHA-1, X.npy CRC-32, sweep-script SHA-1, "
            "fit-function SHA-1) matches.",
            "OOF AUC pools the five folds' predictions (the recorded fit.auc); fold membership is not stored, so the "
            "match-disjointness of folds is inherited from GroupKFold in run_window_sweep_v33.py and not re-checked here.",
            "run_window_sweep_v33.py already stored a 1,000-replicate version of this bootstrap; this file supersedes it "
            "with 2,000 replicates, whose first 1,000 replicates reproduce it (reproduces_window_sweep_v33_bootstrap).",
        ],
        "wall_clock_s": None,
    }
    result["wall_clock_s"] = round(time.time() - started, 1)
    write_json_atomic(output, result)
    log(f"wrote {output} in {result['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
