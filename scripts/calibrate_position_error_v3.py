"""Position-error calibration for the evidence fusion, measured on matches disjoint from every evaluation set.

Why: gameplay/evidence_state.POS_ERROR_CURVE (pilot v1) was read from scripts/audit_evidence_state.py run
on protocol['splits']['engagement'] = predict_test, the pilot's own evaluation matches (9,844 / 9,844),
and its frame row was lowered by hand.  This script re-measures the table on disjoint matches and fits
the monotone curve by a written rule.

Reference (measurement) - the leak-free audit of scripts/audit_evidence_state.py: for every frame
j >= 2 of a match, query tau = minute_ts[j] - 1 ms; the reconstruction may use frame j-1 and events
<= tau only; the truth is frame j's node x_norm / y_norm times MAP_MAX (16,000).  The legacy table was
read from that audit's position_error_u.by_kind_and_age_bin_10s (outputs/evidence_state/audit_1000.json),
where each participant-frame is credited to the kind of evidence the first fusion rule selected,
"the most recent evidence wins" (docs/CLAUDE_EVIDENCE_STATE_REPORT.md section 3), binned by
min(5, age_s // 10).

  --measure most_recent (default)  that selection: a dead participant is set aside (kind "dead", not
                                   fitted); otherwise the candidate of EvidenceStateBuilder.position_candidates
                                   with the largest timestamp, ties to the LAST in candidate order
                                   (frame, respawn, killer, assist, objective, shop).  Of the tie rules
                                   tried, only this one reproduces audit_1000.json (see DEVIATIONS).
  --measure latest_per_kind        every kind's most recent candidate is credited (independent of any
                                   fusion rule).  Both tables are always written; only the chosen one is fitted.

Fit rule (no hand edits), per kind over the 6 age bins 0-10 .. 50-60+ s:
  isotonic_weighted (default)  least-squares non-decreasing fit of the bin means with the bin counts as
                               weights: pool-adjacent-violators (Barlow, Bartholomew, Bremner & Brunk
                               1972, "Statistical Inference under Order Restrictions", ch. 1), computed
                               by sklearn.isotonic.IsotonicRegression(increasing=True).  It equals the
                               isotonic regression of the individual errors on the bin index, so a bin
                               of a few hundred cannot outvote one of tens of thousands.
  cummax                       running maximum of the bin means from the youngest bin upward.
  Empty bins take the fitted value of the nearest non-empty younger bin (older if there is none).
  "victim" is never a fusion candidate (a dead champion keeps the frame); it is set equal to "killer",
  as in the legacy table.  Values are rounded to 0.1 u (rounding is monotone, so order is kept).

Matches: split --split (default predict_train) of outputs/state_value_main_50k_eval/match_splits.json,
ordered by sha256("poscal:7:" + id), keeping the first --n-matches whose patch is NOT a hold-out patch of
the paper's patch holdout (--holdout-patches, default 15.15 validation and 15.16 test).  The run raises
unless the selected set has an empty intersection with predict_test (pilot evaluation), value_validation
(value-model test) and the matches of every hold-out patch in corpus_shards_v33; all counts are written.

--reproduce-legacy-audit AUDIT_JSON re-runs the measurement on the legacy audit's own matches only to show
the measurement code reproduces it (n and mean per kind x bin).  It writes a comparison, never a fitted
table, and its output carries no disjointness record, so load_pos_error_curve refuses to load it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAND_ORDER = ("frame", "respawn", "killer", "assist", "objective", "shop")   # position_candidates order
FIT_KINDS = ("frame", "killer", "assist", "objective", "shop", "respawn")
N_BINS, BIN_S = 6, 10
SALT = "poscal:7:"
OUT_DIR = Path("D:/LOL_Project/fusion_2615/features/tog_revision/A4-evidence-state-leak")
# must equal gameplay.evidence_state.DEFAULT_POS_ERROR_CALIBRATION (checked at run time, after setup_env)
DEFAULT_CALIBRATION = OUT_DIR / "position_error_calibration_v2.json"
DEVIATIONS = [
    "The first-rule fusion code that produced outputs/evidence_state/audit_1000.json is not in git history "
    "(commit 8998201 already carries the calibrated rule); 'most_recent' is reconstructed: largest candidate "
    "timestamp, ties to the LAST candidate in position_candidates order.  On the legacy audit's 1,000 matches "
    "this reproduces the published n per kind x bin to 14 participant-frames in total (of ~230k alive) and "
    "every bin mean to <= 2.0 u; ties-to-first gave 152 / 10.2 u, and including Ornn purchases 562-654 / ~20 u "
    "(--reproduce-legacy-audit).",
    "Candidates come from the current EvidenceStateBuilder.position_candidates (death position dropped after "
    "respawn, Ornn purchases excluded), i.e. the evidence the fusion actually sees now.",
    "fused_error_in_sample_u reduces each kind to its latest candidate (ties to last), so a same-kind tie at "
    "one millisecond may pick a different position than the builder (diagnostic only; not used by the fit).",
    "Only positions are measured (the audit also measured gold / level / stats); the fusion curve needs positions only.",
    "Truth for frame-kind ages < 50 s comes only from irregular frames (in practice the match's final frame), "
    "because tau = next frame - 1 ms makes a regular frame ~60 s old; frame_final_frame_share records this.",
    "Match patch is read from corpus_shards_v33 (groups/patch) when the match is in the corpus, otherwise from "
    "the 50k per-match npz; a match with neither is excluded as unverifiable.",
]


def rank(text: str) -> int:
    """Same formula as train.temporal_winprob.stable_int."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(set(map(str, ids)))).encode()).hexdigest()


def git_state(root: Path = ROOT) -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                               text=True, timeout=120).stdout.strip()
        return {"commit": commit or None, "dirty": bool(dirty)}
    except Exception as exc:                                        # git missing: record, do not fail
        return {"commit": None, "error": repr(exc)}


def source_hashes(paths) -> dict:
    return {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths if (ROOT / p).exists()}


def corpus_patch_matches(shard_dir: Path) -> dict:
    """patch -> set of match ids in the corpus shards; reads only the groups / patch members."""
    paths = sorted(Path(shard_dir).glob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no corpus shards under {shard_dir}")
    out = defaultdict(set)
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            for g, pa in zip(z["groups"].tolist(), z["patch"].tolist()):
                out[str(pa)].add(str(g))
    return dict(out)


def disjointness(selected, evaluation_sets: dict) -> dict:
    sel = set(map(str, selected))
    return {name: {"n_eval": len(set(ids)), "intersection": len(sel & set(map(str, ids))),
                   "eval_ids_sha256": ids_sha256(ids)} for name, ids in evaluation_sets.items()}


def curve_disjointness_record(curve_provenances: dict, evaluation_sets: dict, *, allow_legacy: bool = False) -> dict:
    """Leak guard for a model run that consumes evidence blocks (pilot, patch-holdout comparison).

    curve_provenances {block name: gameplay.evidence_state.load_pos_error_curve provenance, or None for blocks built
    before provenance was recorded} must all name one curve (same source and sha256).  A calibration JSON must still
    hash to the sha256 recorded when the blocks were built, and its match ids must not intersect any of
    evaluation_sets {name: ids} - the run's own validation / test matches; either failure raises ValueError.
    The legacy table (calibrated on predict_test) and unrecorded curves raise unless allow_legacy, and are then
    returned flagged evaluation_calibrated_curve = True (reproduction of the v1 pilot only).
    """
    provs = {k: (v or {}) for k, v in curve_provenances.items()}
    sources = {k: v.get("source") for k, v in provs.items()}
    hashes = {k: v.get("sha256") for k, v in provs.items()}
    if len(set(sources.values())) > 1 or len(set(hashes.values())) > 1:
        raise ValueError(f"blocks were built with different position-error curves: {sources}")
    src = next(iter(sources.values()), None)
    if src in (None, "legacy"):
        if not allow_legacy:
            raise ValueError(f"position-error curve {src or 'unrecorded'!r} was calibrated on evaluation matches "
                             "(predict_test); not allowed in this run")
        return {"source": src or "unrecorded (blocks built before provenance was recorded: the legacy curve)",
                "evaluation_calibrated_curve": True,
                "note": "gameplay.evidence_state.POS_ERROR_CURVE was calibrated on predict_test with a hand-lowered "
                        "frame row; valid only to reproduce the v1 pilot"}
    if src == "dict":
        raise ValueError("an inline curve records no calibration matches; its disjointness cannot be checked")
    raw = Path(src).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    recorded = next(iter(hashes.values()), None)
    if recorded and sha != recorded:
        raise ValueError(f"{src} changed after the blocks were built (sha256 {sha[:12]} != {recorded[:12]})")
    cal_ids = set(map(str, json.loads(raw.decode("utf-8"))["match_ids"]))
    checks = disjointness(cal_ids, evaluation_sets)
    bad = {k: v["intersection"] for k, v in checks.items() if v["intersection"]}
    if bad:
        raise ValueError(f"calibration matches intersect evaluation sets of this run: {bad}")
    return {"source": str(src), "sha256": sha, "evaluation_calibrated_curve": False,
            "calibration_matches": len(cal_ids), "calibration_match_ids_sha256": ids_sha256(cal_ids),
            "disjointness_from_this_run": checks}


def select_calibration_matches(pool, n, *, patch_of, holdout_patches, split_evaluation_sets,
                               patch_holdout_sets, salt=SALT):
    """First n of `pool` by sha256(salt + id) whose patch is known and not a hold-out patch.

    split_evaluation_sets {name: ids} must not intersect the pool at all - an overlap is a split bug and
    raises instead of being filtered.  patch_holdout_sets {name: ids} are the corpus matches of the
    hold-out patches: filtered through patch_of, then asserted empty (a disagreement between patch_of
    and the corpus raises).  patch_of(id) -> patch string or None (unknown: excluded).
    """
    pool = list(dict.fromkeys(map(str, pool)))
    pool_set = set(pool)
    for name, ids in split_evaluation_sets.items():
        inter = pool_set & set(map(str, ids))
        if inter:
            raise ValueError(f"calibration pool intersects evaluation split {name!r} ({len(inter)} matches)")
    holdout = {str(p) for p in holdout_patches}
    selected, stats = [], {"pool": len(pool), "considered": 0, "excluded_holdout_patch": 0, "excluded_unknown_patch": 0}
    for mid in sorted(pool, key=lambda m: rank(salt + m)):
        if len(selected) >= n:
            break
        stats["considered"] += 1
        patch = patch_of(mid)
        if patch is None:
            stats["excluded_unknown_patch"] += 1
        elif str(patch) in holdout:
            stats["excluded_holdout_patch"] += 1
        else:
            selected.append(mid)
    checks = disjointness(selected, {**split_evaluation_sets, **patch_holdout_sets})
    bad = {k: v["intersection"] for k, v in checks.items() if v["intersection"]}
    if bad:
        raise ValueError(f"calibration matches intersect evaluation sets: {bad}")
    return selected, stats, checks


def fit_monotone(means, counts, rule="isotonic_weighted"):
    """Non-decreasing curve over the age bins by `rule` (module docstring); empty bins are filled."""
    means, counts = np.asarray(means, dtype=float), np.asarray(counts, dtype=float)
    ok = np.flatnonzero((counts > 0) & np.isfinite(means))
    if not len(ok):
        raise ValueError("no non-empty bin to fit")
    if rule == "isotonic_weighted":
        from sklearn.isotonic import IsotonicRegression
        vals = IsotonicRegression(increasing=True).fit(ok.astype(float), means[ok], sample_weight=counts[ok]).predict(ok.astype(float))
    elif rule == "cummax":
        vals = np.maximum.accumulate(means[ok])
    else:
        raise ValueError(f"unknown fit rule {rule!r}")
    out = np.full(len(means), np.nan)
    out[ok] = vals
    for b in range(len(out)):
        if not np.isfinite(out[b]):
            younger = [k for k in ok if k < b]
            out[b] = out[max(younger)] if younger else out[min(k for k in ok if k > b)]
    return [round(float(v), 1) for v in out]


def measure_match(b, xy_scale: float):
    """Per alive participant-frame: latest candidate age / error per kind (CAND_ORDER), frame flags.

    Returns dict of arrays; dead participant-frames are returned separately (frame age, frame error).
    Nothing after tau is read by the candidates; frame j is used only as the truth.
    """
    ix, iy = b.idx["x_norm"], b.idx["y_norm"]
    ages, errs, final, dead_age, dead_err = [], [], [], [], []
    ts = b.ts
    for j in range(2, len(ts)):
        tau = int(ts[j]) - 1
        i = b.frame_index(tau)
        for pid in range(1, 11):
            tx = float(b.node[j, pid - 1, ix]) * xy_scale
            ty = float(b.node[j, pid - 1, iy]) * xy_scale
            dead, cands = b.position_candidates(pid, tau, i)
            if dead:
                c = cands[0]
                dead_age.append((tau - c[0]) / 1000.0)
                dead_err.append(float(np.hypot(tx - c[1], ty - c[2])))
                continue
            best = {}
            for c in cands:                                          # later or equal wins: ties to the last
                k = c[3]
                if k not in best or c[0] >= best[k][0]:
                    best[k] = c
            a = np.full(len(CAND_ORDER), np.nan)
            e = np.full(len(CAND_ORDER), np.nan)
            for k, c in best.items():
                q = CAND_ORDER.index(k)
                a[q] = (tau - c[0]) / 1000.0
                e[q] = float(np.hypot(tx - c[1], ty - c[2]))
            ages.append(a)
            errs.append(e)
            final.append(j == len(ts) - 1)
    return {"ages": ages, "errs": errs, "final": final, "dead_age": dead_age, "dead_err": dead_err}


def bin_of(age):
    return np.minimum(N_BINS - 1, np.floor(np.asarray(age) / BIN_S)).astype(int)


def most_recent_choice(ages):
    """Index into CAND_ORDER of the most recent candidate (smallest age; ties to the LAST kind)."""
    a = np.where(np.isfinite(ages), ages, np.inf)[:, ::-1]
    return ages.shape[1] - 1 - np.argmin(a, axis=1)


def table(ages, errs, final, mode):
    n = len(ages)
    rows = {}
    if mode == "most_recent":
        pick = most_recent_choice(ages)
        credited = [(pick == q) for q in range(len(CAND_ORDER))]
    else:
        credited = [np.isfinite(ages[:, q]) for q in range(len(CAND_ORDER))]
    for q, kind in enumerate(CAND_ORDER):
        m = credited[q]
        bins = bin_of(ages[m, q]) if m.any() else np.zeros(0, dtype=int)
        e, fin = errs[m, q], final[m]
        cell = {"n": [], "mean": [], "median": [], "p90": [], "final_frame_share": []}
        for bb in range(N_BINS):
            sel = bins == bb
            v = e[sel]
            cell["n"].append(int(sel.sum()))
            cell["mean"].append(float(v.mean()) if len(v) else None)
            cell["median"].append(float(np.median(v)) if len(v) else None)
            cell["p90"].append(float(np.quantile(v, .9)) if len(v) else None)
            cell["final_frame_share"].append(float(fin[sel].mean()) if len(v) else None)
        rows[kind] = cell
    rows["_participant_frames"] = int(n)
    return rows


def fused_error(ages, errs, curve):
    """Mean error of the fusion argmin(expected error, then age) under `curve`, vectorised."""
    exp = np.full(ages.shape, np.inf)
    for q, kind in enumerate(CAND_ORDER):
        ok = np.isfinite(ages[:, q])
        exp[ok, q] = np.asarray(curve[kind], dtype=float)[bin_of(ages[ok, q])]
    best = exp.min(axis=1, keepdims=True)
    age_tie = np.where(exp == best, ages, np.inf)
    pick = np.argmin(age_tie, axis=1)
    err = errs[np.arange(len(pick)), pick]
    return {"mean": float(err.mean()), "median": float(np.median(err)),
            "selected_share": {k: float((pick == q).mean()) for q, k in enumerate(CAND_ORDER)}}


def run_measurement(ids, cache_dir: Path):
    from data.cache_io import load_match_cache
    from gameplay.evidence_state import DEN, EvidenceStateBuilder, load_tables
    tables, acc, loaded, failed = {}, defaultdict(list), [], {}
    started = time.time()
    for k, mid in enumerate(ids, 1):
        pack = load_match_cache(mid)
        if pack is None:
            failed[mid] = "cache missing"
            continue
        patch = str(pack["meta"]["patch"])
        if patch not in tables:
            tables[patch] = load_tables(patch)
        try:
            # the curve is irrelevant to measurement (candidates only); "legacy" avoids a file dependency
            b = EvidenceStateBuilder(pack, patch=patch, tables=tables[patch], pos_error_curve="legacy")
            rec = measure_match(b, float(DEN["coord"]))
        except Exception as exc:                                    # unsupported pack: record, move on
            failed[mid] = repr(exc)
            continue
        for key, v in rec.items():
            acc[key].extend(v)
        loaded.append(mid)
        if k % 100 == 0:
            print(f"[poscal] {k}/{len(ids)} matches, {len(acc['ages'])} alive participant-frames "
                  f"({time.time()-started:.0f}s)", flush=True)
    ages = np.asarray(acc["ages"], dtype=float).reshape(-1, len(CAND_ORDER))
    errs = np.asarray(acc["errs"], dtype=float).reshape(-1, len(CAND_ORDER))
    final = np.asarray(acc["final"], dtype=bool)
    dead = (np.asarray(acc["dead_age"], dtype=float), np.asarray(acc["dead_err"], dtype=float))
    return ages, errs, final, dead, loaded, failed


def dead_table(dead_age, dead_err):
    bins = bin_of(dead_age) if len(dead_age) else np.zeros(0, dtype=int)
    return {"n": [int((bins == b).sum()) for b in range(N_BINS)],
            "mean": [float(dead_err[bins == b].mean()) if (bins == b).any() else None for b in range(N_BINS)]}


def setup_env(out: Path, cache_dir: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LOL_OUTPUT_ROOT"] = str(out.parent / "runtime")
    os.environ["LOL_CFG_PRESET"] = "v3.3"
    os.environ["LOL_CFG_OVERRIDES"] = json.dumps({"CACHE_DIRNAME": str(cache_dir.resolve()),
        "FIGHT_INDEX_CACHE_ENABLED": False, "FIGHT_INDEX_NUM_WORKERS": 1, "DUMP_FIGHTS": False})


def reproduce_legacy(a, started):
    """Measurement-code equivalence check on the legacy audit's matches; writes a comparison only."""
    protocol = json.loads((a.v3_dir / "protocol.json").read_text(encoding="utf-8"))
    ids = sorted(protocol["splits"]["engagement"], key=lambda m: rank("audit:7:" + m))[:a.n_matches]
    ages, errs, final, dead, loaded, failed = run_measurement(ids, a.cache_dir)
    mine = table(ages, errs, final, "most_recent")
    mine["dead"] = dead_table(*dead)
    ref = json.loads(a.reproduce_legacy_audit.read_text(encoding="utf-8"))["position_error_u"]["by_kind_and_age_bin_10s"]
    diffs, count_mismatch, compared = [], [], 0
    for key, cell in ref.items():
        kind, span = key.split("|")
        bb = int(span.split("-")[0]) // BIN_S
        if kind not in mine:
            count_mismatch.append({"bin": key, "reference_n": cell["n"], "measured_n": 0})
            continue
        n_m, mean_m = mine[kind]["n"][bb], mine[kind]["mean"][bb]
        compared += 1
        if n_m != cell["n"]:
            count_mismatch.append({"bin": key, "reference_n": cell["n"], "measured_n": n_m})
        if mean_m is not None:
            diffs.append(abs(mean_m - cell["evidence_mean"]))
    out = {"purpose": "code-equivalence check of the measurement against the published legacy audit; "
                      "evaluation matches are read, nothing is fitted or written for later use",
           "reference": str(a.reproduce_legacy_audit), "matches": len(ids), "loaded": len(loaded),
           "bins_compared": compared, "count_mismatches": count_mismatch,
           "max_abs_mean_diff_u": float(max(diffs)) if diffs else None,
           "reproduced": (not count_mismatch) and bool(diffs) and max(diffs) < 0.5,
           "git": git_state(), "elapsed_seconds": round(time.time() - started, 1)}
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("bins_compared", "max_abs_mean_diff_u", "reproduced")}), flush=True)
    print(f"[poscal] count mismatches: {count_mismatch[:5]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-matches", type=int, default=1000)
    ap.add_argument("--split", default="predict_train")
    ap.add_argument("--match-splits", type=Path, default=ROOT / "outputs/state_value_main_50k_eval/match_splits.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/state_value_main_50k")
    ap.add_argument("--corpus-shards", type=Path, default=Path("D:/LOL_Project/fusion_2615/corpus_shards_v33"))
    ap.add_argument("--holdout-patches", default="15.15,15.16")
    ap.add_argument("--cache-dir", type=Path, default=Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13"))
    ap.add_argument("--measure", choices=("most_recent", "latest_per_kind"), default="most_recent")
    ap.add_argument("--fit-rule", choices=("isotonic_weighted", "cummax"), default="isotonic_weighted")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"default {OUT_DIR / 'position_error_calibration_v2.json'}; with --reproduce-legacy-audit "
                         f"{OUT_DIR / 'legacy_audit_reproduction_check.json'} (never the calibration path)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reproduce-legacy-audit", type=Path, default=None,
                    help="legacy audit JSON to reproduce (reads its evaluation matches; writes a comparison only)")
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v3_buckets")
    a = ap.parse_args()
    started, wall_start = time.time(), time.strftime("%Y-%m-%dT%H:%M:%S")
    # resolved without importing gameplay.* : core.config must first see the environment setup_env writes
    default_calibration = Path(os.environ.get("LOL_POS_ERROR_CALIBRATION") or DEFAULT_CALIBRATION)
    if a.out is None:
        a.out = (OUT_DIR / "legacy_audit_reproduction_check.json") if a.reproduce_legacy_audit is not None else default_calibration
    if a.reproduce_legacy_audit is not None and a.out.resolve() in (default_calibration.resolve(), DEFAULT_CALIBRATION.resolve()):
        raise SystemExit("--reproduce-legacy-audit reads evaluation matches; it must not write the calibration path")
    if a.out.exists() and not a.overwrite:
        raise SystemExit(f"{a.out} exists; pass --overwrite or choose a new name")
    setup_env(a.out, a.cache_dir)
    from gameplay.evidence_state import DEFAULT_POS_ERROR_CALIBRATION
    if DEFAULT_POS_ERROR_CALIBRATION.resolve() != DEFAULT_CALIBRATION.resolve():
        raise SystemExit(f"default calibration path disagrees with gameplay.evidence_state: {DEFAULT_POS_ERROR_CALIBRATION}")
    if a.reproduce_legacy_audit is not None:
        return reproduce_legacy(a, started)

    splits = json.loads(a.match_splits.read_text(encoding="utf-8"))
    if a.split in ("predict_test", "value_validation"):
        raise SystemExit(f"{a.split} is an evaluation split; calibration must not read it")
    holdout = [p.strip() for p in a.holdout_patches.split(",") if p.strip()]
    corpus = corpus_patch_matches(a.corpus_shards)
    corpus_patch = {m: p for p, ms in corpus.items() for m in ms}

    def patch_of(mid):
        if mid in corpus_patch:
            return corpus_patch[mid]
        f = a.dataset / "matches" / (mid + ".npz")
        if f.exists():
            with np.load(f, allow_pickle=False) as z:
                return str(z["patch"])
        return None

    split_eval = {k: splits[k] for k in ("predict_test", "value_validation") if k in splits}
    patch_eval = {f"corpus_patch_{p}": sorted(corpus.get(p, set())) for p in holdout}
    ids, sel_stats, checks = select_calibration_matches(
        splits[a.split], a.n_matches, patch_of=patch_of, holdout_patches=holdout,
        split_evaluation_sets=split_eval, patch_holdout_sets=patch_eval)
    print(f"[poscal] {len(ids)} matches from {a.split} (considered {sel_stats['considered']}, excluded hold-out "
          f"patch {sel_stats['excluded_holdout_patch']}); intersections "
          f"{ {k: v['intersection'] for k, v in checks.items()} }", flush=True)

    ages, errs, final, dead, loaded, failed = run_measurement(ids, a.cache_dir)
    if not len(ages):
        raise SystemExit("no participant-frames measured")
    measured = {m: table(ages, errs, final, m) for m in ("most_recent", "latest_per_kind")}
    measured["most_recent"]["dead"] = dead_table(*dead)
    from gameplay.evidence_state import POS_ERROR_CURVE, validate_pos_error_curve
    fits = {}
    for rule in ("isotonic_weighted", "cummax"):
        curve = {k: fit_monotone([np.nan if v is None else v for v in measured[a.measure][k]["mean"]],
                                 measured[a.measure][k]["n"], rule) for k in FIT_KINDS}
        curve["victim"] = list(curve["killer"])
        fits[rule] = validate_pos_error_curve(curve)
    fitted = fits[a.fit_rule]
    legacy = {k: [float(v) for v in vals] for k, vals in POS_ERROR_CURVE.items()}
    out = {
        "version": "position_error_calibration_v2",
        "reference": "scripts/audit_evidence_state.py measurement (tau = next frame - 1 ms, truth = next frame); "
                     "fit: isotonic regression, Barlow et al. 1972 ch. 1 (see script docstring)",
        "deviations": DEVIATIONS,
        "git": git_state(), "source_sha256": source_hashes(["gameplay/evidence_state.py", "scripts/calibrate_position_error_v3.py"]),
        "preset": "v3.3", "seed": 7, "salt": SALT,
        "split": {"match_splits": str(a.match_splits), "name": a.split, "holdout_patches": holdout,
                  "corpus_shards": str(a.corpus_shards), "selection": sel_stats},
        "n_matches": len(ids), "n_matches_loaded": len(loaded), "failed": failed,
        "match_ids": ids, "match_ids_sha256": ids_sha256(ids),
        "loaded_match_ids_sha256": ids_sha256(loaded),
        "disjointness": checks,
        "alive_participant_frames": int(len(ages)), "dead_participant_frames": int(len(dead[0])),
        "bins_s": BIN_S, "n_bins": N_BINS, "age_bin_rule": "min(5, floor(age_s / 10))",
        "measure": a.measure, "measured": measured,
        "fit_rule": a.fit_rule, "fitted": fitted, "fitted_alternatives": fits,
        "legacy_curve": legacy,
        "fitted_minus_legacy": {k: [round(f - l, 1) for f, l in zip(fitted[k], legacy[k])] for k in legacy},
        "fused_error_in_sample_u": {"note": "on the calibration matches themselves (in-sample for 'fitted')",
                                    "hold": {"mean": float(errs[:, 0].mean()), "median": float(np.median(errs[:, 0]))},
                                    "legacy": fused_error(ages, errs, legacy), "fitted": fused_error(ages, errs, fitted)},
        "wall_clock": {"start": wall_start, "end": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"fitted_frame": fitted["frame"], "measured_frame_mean": measured[a.measure]["frame"]["mean"],
                      "measured_frame_n": measured[a.measure]["frame"]["n"],
                      "fused_mean_u": {k: out["fused_error_in_sample_u"][k]["mean"] for k in ("hold", "legacy", "fitted")},
                      "elapsed_s": out["elapsed_seconds"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
