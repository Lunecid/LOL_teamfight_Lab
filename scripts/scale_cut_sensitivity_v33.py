"""Teamfight-cut sensitivity of the frozen corpus-v3.3 predictions, with presence as a second scale.

Reviewer R1 (CoG 2026 #118) called the detector constants arbitrary; the teamfight cut
``n_min >= 4`` is one of them and it was read from a v2 participation distribution.  The headline
decomposition (``scripts/run_scale_decomposition.py``) fits the tabular baseline once under
match-grouped 5-fold CV and evaluates the frozen out-of-fold predictions inside each scale class, so
moving the cut only re-scores the stored predictions: no refit, and the cut cannot reach the model.

Two definitions of scale, both on the smaller side ``n_min = min(blue, red)``:

  participation  ``cluster_blue/red``: champions in the kill cluster (killers, assisters, victims)
                 plus in-radius interaction actors.  Known only once the fight has resolved.
  presence       ``present_blue/red``: alive champions within R of the first kill at the prediction
                 cutoff.  A pre-fight quantity, >= M = 2 by the presence gate, so it has no pick class
                 and is reported as small (n_min < c) against large (n_min >= c).

Classes at a cut c: pick ``n_min <= 1``, skirmish ``2 .. c-1``, teamfight ``n_min >= c`` (the
``scale_class`` of run_scale_decomposition.py).

Negative participation counts.  gameplay/fights.py always writes ``det_cluster_blue/red`` as an int
count, and data/index_split._fight_to_ref_row stores ``int(count or -1)``, so a -1 is a team with zero
kill-cluster participants (42 of the 532,547 labelled rows).  ``--negative-count-as zero`` (default)
reads it as 0, i.e. a pick; ``unknown`` keeps those rows out of every class, which is what
run_scale_decomposition.scale_class does.  Only the pick class differs between the two readings; the run
reports the other reading on the same replicates, and the main-loop pick - teamfight table was computed
under the zero reading (it reproduces exactly under zero, and is 1e-4 lower under unknown).

Intervals are match-clustered percentile bootstraps: the percentile interval of Efron & Tibshirani
(1993, An Introduction to the Bootstrap, ch. 13) with whole matches resampled, the cluster bootstrap of
Field & Welsh (2007, "Bootstrapping clustered data", JRSS-B 69(3)).  A replicate draws n_matches matches
with replacement and every statistic is computed on that same replicate, so gaps are paired.  The draw
stream is the one ``run_scale_decomposition.cluster_bootstrap`` uses (``default_rng(seed)``, one
``integers(0, n_matches, n_matches)`` per replicate), so the same seed gives the same replicates.
The AUC of a replicate uses match multiplicities as row weights: the Mann-Whitney statistic with ties
counted 1/2 (Hanley & McNeil 1982, Radiology 143(1)), which equals ``roc_auc_score`` on the explicitly
concatenated replicate; the script asserts that equality on the first replicate.  As in
``cluster_bootstrap``, a statistic is skipped in a replicate whose subset has < 50 rows or one class.

``WeightedAUC``, ``MatchBootstrap``, ``git_provenance`` and ``corpus_provenance`` are reused by
scripts/scale_participation_v33.py, scripts/run_gd_sweep_v33.py and scripts/run_presence_gate_points.py.

    .venv/Scripts/python.exe scripts/scale_cut_sensitivity_v33.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = Path("D:/LOL_Project/fusion_2615/features")
ITEM = "A6-definition-sensitivity"
DEFAULT_OUT_DIR = FEATURES / "tog_revision" / ITEM
DEFAULT_PREDS = FEATURES / "scale_decomposition_v33_market_event.preds.npz"
DEFAULT_SHARDS = Path("D:/LOL_Project/fusion_2615/corpus_shards_v33")
MIN_ROWS = 50
# Quoted by the main loop from the same preds file (400 match bootstraps, seed 7): pick - teamfight
# point [2.5 %, 97.5 %] per participation cut.  The run records whether it reproduces them.
MAIN_LOOP_REFERENCE = {3: (0.0108, 0.0070, 0.0147), 4: (-0.0019, -0.0065, 0.0028), 5: (-0.0105, -0.0170, -0.0041)}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ----------------------------------------------------------------------------- shared helpers
def git_provenance(root: Path = PROJECT_ROOT) -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception:
            return ""
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty_tracked": bool(_git("status", "--porcelain", "--untracked-files=no"))}


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preset_values(name: str) -> dict:
    """core/presets.py loaded by path, so reading a preset never imports core.config."""
    spec = importlib.util.spec_from_file_location("_presets_by_path", PROJECT_ROOT / "core" / "presets.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return dict(module.PRESETS[name])


def corpus_provenance(shard_dir: Path) -> dict:
    mp = shard_dir / "manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    files = sorted(p.name for p in shard_dir.glob("shard_*.npz"))
    expected = [f"shard_{i:03d}.npz" for i in range(int(m.get("num_shards", 0) or 0))]
    det = m.get("detector") or {}
    preset = preset_values("v3.3")
    return {"shards": str(shard_dir), "run_id": m.get("run_id"), "git_commit": m.get("git_commit"),
            "complete": bool(m.get("complete", False)), "shard_files_match_manifest": files == expected,
            "feature_names_sha1": m.get("feature_names_sha1"), "detector": det, "label": m.get("label"),
            "scale": m.get("scale"), "preset_field_in_manifest": m.get("preset"),
            "detector_matches_preset_v3.3": bool(det) and all(preset.get(k) == v for k, v in det.items())}


def load_shard_arrays(shard_dir: Path, keys) -> dict:
    """Row arrays of every shard in merge order (sorted file names), X never touched."""
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {shard_dir}")
    parts = {k: [] for k in keys}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            for k in keys:
                parts[k].append(np.asarray(z[k]))
    return {k: np.concatenate(v) for k, v in parts.items()}


class WeightedAUC:
    """ROC AUC of row-weighted subsets of one fixed (y, score) vector.

    The scores are sorted once; a call is O(n): weighted positives P_g and negatives N_g are summed per
    tied-score group g and AUC = sum_g P_g (N_below(g) + N_g / 2) / (P N), the Mann-Whitney statistic with
    ties counted 1/2.  With integer weights it is exactly the AUC of the multiset repeating row i w_i times.
    """

    def __init__(self, y, score):
        y = np.asarray(y).astype(np.int64)
        score = np.asarray(score, dtype=np.float64)
        self.order = np.argsort(score, kind="mergesort")
        s = score[self.order]
        self.pos = (y[self.order] == 1).astype(np.float64)
        new = np.ones(len(s), dtype=bool)
        new[1:] = s[1:] != s[:-1]
        self.gid = np.cumsum(new) - 1
        self.n_groups = int(self.gid[-1]) + 1 if len(s) else 0

    def sort(self, v) -> np.ndarray:
        return np.asarray(v, dtype=np.float64)[self.order]

    def from_sorted(self, w_sorted: np.ndarray, min_rows: int = MIN_ROWS) -> float:
        wp = w_sorted * self.pos
        wn = w_sorted - wp
        p_tot, n_tot = float(wp.sum()), float(wn.sum())
        if p_tot + n_tot < min_rows or p_tot <= 0 or n_tot <= 0:
            return float("nan")
        gp = np.bincount(self.gid, weights=wp, minlength=self.n_groups)
        gn = np.bincount(self.gid, weights=wn, minlength=self.n_groups)
        below = np.cumsum(gn) - gn
        return float(np.dot(gp, below + 0.5 * gn) / (p_tot * n_tot))

    def __call__(self, w, min_rows: int = MIN_ROWS) -> float:
        return self.from_sorted(self.sort(w), min_rows=min_rows)


class MatchBootstrap:
    """Match-clustered resampling over a fixed universe of matches.

    ``draw()`` returns the multiplicity of every universe match in one replicate (and the raw draw);
    rows take their match's multiplicity as weight: ``counts[index_of(groups)]``.
    """

    def __init__(self, universe, seed: int):
        self.unique, self.inverse = np.unique(np.asarray(universe), return_inverse=True)
        self.rng = np.random.default_rng(seed)

    def index_of(self, groups) -> np.ndarray:
        groups = np.asarray(groups)
        idx = np.searchsorted(self.unique, groups)
        idx_c = np.minimum(idx, len(self.unique) - 1)
        if len(groups) and not np.array_equal(self.unique[idx_c], groups):
            raise ValueError("rows reference matches outside the bootstrap universe")
        return idx_c

    def draw(self):
        sample = self.rng.integers(0, len(self.unique), size=len(self.unique))
        return np.bincount(sample, minlength=len(self.unique)).astype(np.float64), sample


def smaller_side_count(blue, red, negative_count_as: str = "zero") -> tuple:
    """(n_min, known) for per-team participation counts; see the module docstring on negative counts."""
    b, r = np.asarray(blue).astype(np.int64), np.asarray(red).astype(np.int64)
    if negative_count_as == "zero":
        return np.minimum(np.maximum(b, 0), np.maximum(r, 0)), np.ones(len(b), dtype=bool)
    if negative_count_as == "unknown":
        return np.minimum(b, r), (b >= 0) & (r >= 0)
    raise ValueError(f"negative_count_as must be 'zero' or 'unknown', not {negative_count_as!r}")


def concatenated_replicate(inverse: np.ndarray, n_unique: int, sample: np.ndarray) -> np.ndarray:
    """Row indices of a replicate built the way run_scale_decomposition.cluster_bootstrap builds it."""
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(n_unique))
    ends = np.append(starts[1:], len(order))
    return np.concatenate([order[starts[g]:ends[g]] for g in sample])


def summarize(values, point=None, gap: bool = False) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    out = {"point": point, "n_boot": int(len(arr))}
    if len(arr):
        out.update({"mean": float(arr.mean()), "ci_2.5": float(np.percentile(arr, 2.5)),
                    "ci_97.5": float(np.percentile(arr, 97.5))})
        if gap:
            out["p_le_0"] = float((arr <= 0).mean())
    return out


def fmt_ci(d: dict, signed: bool = False) -> str:
    if not d or d.get("point") is None or "ci_2.5" not in d:
        return "-"
    f = "{:+.4f}" if signed else "{:.4f}"
    return f"{f.format(d['point'])} [{f.format(d['ci_2.5'])}, {f.format(d['ci_97.5'])}]"


# ----------------------------------------------------------------------------- analysis
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, default=DEFAULT_PREDS,
                    help="stored OOF predictions (y, pred, groups, cluster_blue, cluster_red)")
    ap.add_argument("--shards", type=Path, default=DEFAULT_SHARDS,
                    help="corpus shards the predictions were fitted on (presence counts, row alignment)")
    ap.add_argument("--y-key", default="y_market_event", help="label column the predictions were fitted on")
    ap.add_argument("--cuts", default="3,4,5")
    ap.add_argument("--cross-cut", type=int, default=4, help="cut used for the participation x presence cells")
    ap.add_argument("--negative-count-as", choices=("zero", "unknown"), default="zero",
                    help="reading of a -1 participation count (int(0 or -1)); the other reading is reported too")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR / "scale_cut_sensitivity_v33.json")
    args = ap.parse_args(argv)
    started = time.time()
    cuts = sorted({int(c) for c in str(args.cuts).split(",") if c.strip()})

    with np.load(args.preds, allow_pickle=False) as z:
        y = z["y"].astype(np.int64)
        pred = z["pred"].astype(np.float64)
        groups = np.asarray(z["groups"])
        cb = z["cluster_blue"].astype(np.int64)
        cr = z["cluster_red"].astype(np.int64)
    n = int(len(y))
    if not np.isin(y, (0, 1)).all():
        raise SystemExit("preds y must be 0/1 (draws already dropped)")

    # presence counts come from the shards; the preds rows are the shards in merge order with the label's
    # draws dropped (run_scale_decomposition.merge_shards), so the alignment is checked, not assumed
    sh = load_shard_arrays(args.shards, [args.y_key, "groups", "cluster_blue", "cluster_red", "present_blue", "present_red"])
    keep = sh[args.y_key] >= 0
    alignment = {"shard_rows": int(len(keep)), "shard_rows_with_label": int(keep.sum()), "pred_rows": n}
    if alignment["shard_rows_with_label"] != n:
        raise SystemExit(f"row count differs between preds and shards: {alignment}")
    alignment["groups_equal"] = bool(np.array_equal(sh["groups"][keep], groups))
    alignment["cluster_blue_equal"] = bool(np.array_equal(sh["cluster_blue"][keep].astype(np.int64), cb))
    alignment["cluster_red_equal"] = bool(np.array_equal(sh["cluster_red"][keep].astype(np.int64), cr))
    alignment["label_equal"] = bool(np.array_equal(sh[args.y_key][keep].astype(np.int64), y))
    if not all(v for k, v in alignment.items() if k.endswith("_equal")):
        raise SystemExit(f"preds do not align with the shards: {alignment}")
    pb = sh["present_blue"][keep].astype(np.int64)
    pr = sh["present_red"][keep].astype(np.int64)
    del sh

    negative = (cb < 0) | (cr < 0)
    alt = "unknown" if args.negative_count_as == "zero" else "zero"
    part_min, part_ok = smaller_side_count(cb, cr, args.negative_count_as)
    alt_min, alt_ok = smaller_side_count(cb, cr, alt)
    pres_min, pres_ok = smaller_side_count(pb, pr, "unknown")
    if not np.array_equal(part_ok & (part_min >= 2), alt_ok & (alt_min >= 2)):
        raise SystemExit("the two negative-count readings disagree outside the pick class")
    alt_pick = f"alt_{alt}/participation/pick"
    masks = {"overall": np.ones(n, dtype=bool), "participation/pick": part_ok & (part_min <= 1),
             alt_pick: alt_ok & (alt_min <= 1)}
    for c in cuts:
        masks[f"participation/c{c}/skirmish"] = part_ok & (part_min >= 2) & (part_min < c)
        masks[f"participation/c{c}/teamfight"] = part_ok & (part_min >= c)
        masks[f"presence/c{c}/small"] = pres_ok & (pres_min < c)
        masks[f"presence/c{c}/large"] = pres_ok & (pres_min >= c)
    for k in range(6):
        masks[f"participation/level{k}"] = part_ok & (part_min == k)
        masks[f"presence/level{k}"] = pres_ok & (pres_min == k)
    xc = int(args.cross_cut)
    part_cls = {"pick": part_ok & (part_min <= 1), "skirmish": part_ok & (part_min >= 2) & (part_min < xc),
                "teamfight": part_ok & (part_min >= xc)}
    pres_cls = {"small": pres_ok & (pres_min < xc), "large": pres_ok & (pres_min >= xc)}
    for pc, pm in part_cls.items():
        for sc, sm in pres_cls.items():
            masks[f"cross/c{xc}/participation_{pc}/presence_{sc}"] = pm & sm
    gaps = {}
    for c in cuts:
        gaps[f"participation/c{c}/pick_minus_teamfight"] = ("participation/pick", f"participation/c{c}/teamfight")
        gaps[f"participation/c{c}/pick_minus_skirmish"] = ("participation/pick", f"participation/c{c}/skirmish")
        gaps[f"participation/c{c}/skirmish_minus_teamfight"] = (f"participation/c{c}/skirmish", f"participation/c{c}/teamfight")
        gaps[f"presence/c{c}/small_minus_large"] = (f"presence/c{c}/small", f"presence/c{c}/large")
        gaps[f"alt_{alt}/participation/c{c}/pick_minus_teamfight"] = (alt_pick, f"participation/c{c}/teamfight")
        gaps[f"alt_{alt}/participation/c{c}/pick_minus_skirmish"] = (alt_pick, f"participation/c{c}/skirmish")

    auc = WeightedAUC(y, pred)
    point, max_point_diff = {}, 0.0
    for name, m in masks.items():
        k = int(m.sum())
        if k >= MIN_ROWS and 0 < int(y[m].sum()) < k:
            point[name] = float(roc_auc_score(y[m], pred[m]))
            max_point_diff = max(max_point_diff, abs(auc(m) - point[name]))
        else:
            point[name] = None
    if max_point_diff > 1e-9:
        raise SystemExit(f"weighted AUC disagrees with roc_auc_score on the full data by {max_point_diff:.2e}")
    print(f"rows={n:,} matches={len(np.unique(groups)):,} overall AUC={point['overall']:.4f}", flush=True)

    boot = MatchBootstrap(groups, args.seed)
    live = [name for name in masks if point[name] is not None]
    msorted = {name: auc.sort(masks[name]) for name in live}
    draws = {name: [] for name in live}
    gap_draws = {g: [] for g in gaps}
    replicate_check = None
    t_boot = time.time()
    for b in range(int(args.n_boot)):
        counts, sample = boot.draw()
        w_sorted = auc.sort(counts[boot.inverse])
        vals = {name: auc.from_sorted(w_sorted * ms) for name, ms in msorted.items()}
        if b == 0:
            idx = concatenated_replicate(boot.inverse, len(boot.unique), sample)
            diffs = {}
            for name in ("overall", "participation/pick", alt_pick, f"participation/c{cuts[-1]}/teamfight",
                         f"presence/c{cuts[0]}/large"):
                if name in vals:
                    sub = idx[masks[name][idx]]
                    diffs[name] = abs(float(roc_auc_score(y[sub], pred[sub])) - vals[name])
            replicate_check = {"statistics": sorted(diffs), "max_abs_diff": max(diffs.values())}
            if replicate_check["max_abs_diff"] > 1e-9:
                raise SystemExit(f"weighted replicate AUC disagrees with the concatenated replicate: {diffs}")
            del idx
        for name, v in vals.items():
            draws[name].append(v)
        for g, (a, c) in gaps.items():
            va, vc = vals.get(a, float("nan")), vals.get(c, float("nan"))
            if np.isfinite(va) and np.isfinite(vc):
                gap_draws[g].append(va - vc)
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot} ({time.time() - t_boot:.0f}s)", flush=True)

    def cls_stats(name):
        m = masks[name]
        k = int(m.sum())
        return {"n": k, "share": float(k / n), "positive_rate": (float(y[m].mean()) if k else None),
                "auc": summarize(draws.get(name, []), point[name])}

    def gap_stats(g):
        a, c = gaps[g]
        pt = (point[a] - point[c]) if (point[a] is not None and point[c] is not None) else None
        return summarize(gap_draws[g], pt, gap=True)

    participation = {"definition": "n_min = min(cluster_blue, cluster_red); pick <= 1, skirmish 2..c-1, teamfight >= c",
                     "negative_count_as": args.negative_count_as,
                     "rows_with_negative_count": int(negative.sum()),
                     "negative_count_rows_by_label": {str(v): int(((y == v) & negative).sum()) for v in (0, 1)},
                     "rows_without_class": int((~part_ok).sum()),
                     "cuts": {}, "levels": {}}
    presence = {"definition": ("n_min = min(present_blue, present_red), alive champions within R of the first kill at "
                               "the cutoff; >= M by the gate, so small (< c) vs large (>= c)"),
                "rows_with_n_min_le_1": int((pres_ok & (pres_min <= 1)).sum()), "cuts": {}, "levels": {}}
    for c in cuts:
        participation["cuts"][str(c)] = {
            "classes": {"pick": cls_stats("participation/pick"), "skirmish": cls_stats(f"participation/c{c}/skirmish"),
                        "teamfight": cls_stats(f"participation/c{c}/teamfight")},
            "gaps": {"pick_minus_teamfight": gap_stats(f"participation/c{c}/pick_minus_teamfight"),
                     "pick_minus_skirmish": gap_stats(f"participation/c{c}/pick_minus_skirmish"),
                     "skirmish_minus_teamfight": gap_stats(f"participation/c{c}/skirmish_minus_teamfight")}}
        presence["cuts"][str(c)] = {
            "classes": {"small": cls_stats(f"presence/c{c}/small"), "large": cls_stats(f"presence/c{c}/large")},
            "gaps": {"small_minus_large": gap_stats(f"presence/c{c}/small_minus_large")}}
    for k in range(6):
        participation["levels"][str(k)] = cls_stats(f"participation/level{k}")
        presence["levels"][str(k)] = cls_stats(f"presence/level{k}")
    cross = {"cut": xc, "cells": {name.split("/", 2)[2]: cls_stats(name) for name in masks if name.startswith("cross/")}}
    other_reading = {
        "negative_count_as": alt,
        "note": "same replicates; skirmish and teamfight are identical under both readings, only pick changes",
        "pick": cls_stats(alt_pick),
        "cuts": {str(c): {"pick_minus_teamfight": gap_stats(f"alt_{alt}/participation/c{c}/pick_minus_teamfight"),
                          "pick_minus_skirmish": gap_stats(f"alt_{alt}/participation/c{c}/pick_minus_skirmish")}
                 for c in cuts}}

    def rounded(g):
        if g.get("point") is None or "ci_2.5" not in g:
            return None
        return (round(g["point"], 4), round(g["ci_2.5"], 4), round(g["ci_97.5"], 4))

    reproduction = {"reference": "pick - teamfight [2.5 %, 97.5 %] quoted by the main loop from the same preds file "
                                 "(400 match bootstraps, seed 7)", "by_reading": {}}
    for reading, block_of in ((args.negative_count_as, lambda c: participation["cuts"][str(c)]["gaps"]),
                              (alt, lambda c: other_reading["cuts"][str(c)])):
        rows = {}
        for c, ref in MAIN_LOOP_REFERENCE.items():
            if c not in cuts:
                continue
            ours = rounded(block_of(c)["pick_minus_teamfight"])
            rows[str(c)] = {"main_loop": list(ref), "this_run": list(ours) if ours else None,
                            "reproduced_to_4dp": bool(ours) and all(abs(a - b) < 1e-9 for a, b in zip(ours, ref))}
        reproduction["by_reading"][reading] = rows
    reproduction["reproduced_under"] = [r for r, rows in reproduction["by_reading"].items()
                                        if rows and all(v["reproduced_to_4dp"] for v in rows.values())]

    git = git_provenance()
    results = {
        "item": ITEM,
        "what": "teamfight-cut sensitivity of the stored v3.3 OOF predictions; participation and presence scale",
        "provenance": {**git, "script": "scripts/scale_cut_sensitivity_v33.py", "python": sys.executable,
                       "preds": str(args.preds), "preds_sha1": file_sha1(args.preds),
                       "corpus": corpus_provenance(args.shards), "preset": "v3.3", "label_key": args.y_key,
                       "draws": "dropped (label < 0)",
                       "split": "match-grouped 5-fold out-of-fold predictions, stored by scripts/run_scale_decomposition.py",
                       "n_rows": n, "n_matches": int(len(boot.unique)), "seed": int(args.seed),
                       "n_boot": int(args.n_boot), "wall_clock_s": None},
        "alignment_with_shards": alignment,
        "weighted_auc_checks": {"max_abs_diff_full_data": max_point_diff, "first_replicate": replicate_check},
        "overall": cls_stats("overall"),
        "participation": participation,
        "participation_other_negative_count_reading": other_reading,
        "presence": presence,
        "participation_x_presence": cross,
        "reproduces_main_loop_table": reproduction,
        "deviations": [
            "Intervals describe evaluation-sample uncertainty of the frozen OOF predictions only; the model is not "
            "refit inside replicates (same as run_scale_decomposition.cluster_bootstrap).",
            "Replicate AUCs use match multiplicities as row weights instead of concatenating rows; identical by "
            "construction and asserted on the first replicate.",
            "Presence classes are small/large rather than pick/skirmish/teamfight because the presence gate forces "
            "n_min >= M = 2.",
            f"A -1 participation count is read as {args.negative_count_as} (int(0 or -1) in "
            "data/index_split._fight_to_ref_row); run_scale_decomposition.scale_class reads it as unknown. The other "
            "reading is reported under participation_other_negative_count_reading.",
        ],
    }
    results["provenance"]["wall_clock_s"] = round(time.time() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n| cut | pick AUC | skirmish AUC | teamfight AUC | pick - teamfight | presence small - large |")
    print("|---|---|---|---|---|---|")
    for c in cuts:
        pc, sc = participation["cuts"][str(c)], presence["cuts"][str(c)]
        print(f"| n_min >= {c} | {fmt_ci(pc['classes']['pick']['auc'])} | {fmt_ci(pc['classes']['skirmish']['auc'])} | "
              f"{fmt_ci(pc['classes']['teamfight']['auc'])} | {fmt_ci(pc['gaps']['pick_minus_teamfight'], True)} | "
              f"{fmt_ci(sc['gaps']['small_minus_large'], True)} |")
    print("\n| n_min | participation n | participation AUC | presence n | presence AUC |")
    print("|---|---|---|---|---|")
    for k in range(6):
        a, b = participation["levels"][str(k)], presence["levels"][str(k)]
        print(f"| {k} | {a['n']:,} | {fmt_ci(a['auc'])} | {b['n']:,} | {fmt_ci(b['auc'])} |")
    print(f"\nother negative-count reading ({alt}): pick {fmt_ci(other_reading['pick']['auc'])}; pick - teamfight " +
          ", ".join(f"c{c} {fmt_ci(other_reading['cuts'][str(c)]['pick_minus_teamfight'], True)}" for c in cuts))
    print("reproduction of the main-loop table:", json.dumps(reproduction))
    print(f"wrote {args.output} ({results['provenance']['wall_clock_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
