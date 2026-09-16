"""Validation B: how much does the reported delta-V depend on where we put the boundary?

The output contract is not fixed yet: end (last cluster kill) is a proxy for combat end, and
+30s / +60s are candidate follow-up horizons.  This measures what actually changes between
them - sign, magnitude, how often the direction flips, how many engagements a neutral band
would swallow - and decomposes it by snapshot age, engagement scale and game phase.

It also tests one specific hypothesis.  In v2 the aggregate immediate delta has OPPOSITE
signs for the two model families (expanded -1.88pp, maymin +0.02pp).  If pre and end resolve
to the SAME source snapshot, then no observed game state changed between them and the whole
delta is produced by the clock alone - so a clock-driven drift, differing between a 7-input
model and a 361-input model with 94 interaction columns, would explain the sign split.

Read-only: consumes the frozen probabilities and the validation-A row table, never refits.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HORIZONS = ("last_kill_proxy", "plus_30s", "plus_60s")   # delta = p[point] - p[pre]
POINT_INDEX = {"pre": 0, "last_kill_proxy": 1, "plus_30s": 2, "plus_60s": 3}
FAMILIES = ("maymin", "expanded")
NEUTRAL_BANDS = (0.0, 0.01, 0.02, 0.05)                   # probability units: 0 / 1 / 2 / 5 pp
AGE_BANDS = ((0, 15), (15, 30), (30, 45), (45, 10_000))
TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 10_000))
BOOT_SEED = 7
BOOT_REPLICATES = 500
MIN_MATCHES_FOR_CI = 30


def match_weights(groups):
    _, ix, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / counts[ix]
    return w / w.mean()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- delta summary

def describe(delta, groups, *, label=""):
    """Both unweighted (v2 convention) and match-weighted views of one delta column."""
    if len(delta) == 0:
        return dict(label=label, n=0, matches=0, note="empty subgroup")
    w = match_weights(groups)
    pp = delta * 100.
    return dict(
        label=label,
        n=int(len(delta)),
        matches=int(len(set(groups.tolist()))),
        mean_pp_unweighted=float(pp.mean()),
        mean_pp_match_weighted=float(np.average(pp, weights=w)),
        median_pp=float(np.median(pp)),
        median_abs_pp=float(np.median(np.abs(pp))),
        abs_pp_quantiles={str(q): float(np.quantile(np.abs(pp), q))
                          for q in (.25, .5, .75, .9, .95, .99)},
        blue_increase_fraction=float((delta > 0).mean()),
        red_increase_fraction=float((delta < 0).mean()),
        exact_zero_fraction=float((delta == 0).mean()),
        blue_increase_fraction_match_weighted=float(np.average(delta > 0, weights=w)),
    )


def bootstrap_mean_delta(deltas_by_key, groups, *, replicates=BOOT_REPLICATES, seed=BOOT_SEED):
    """Match-level CI for each mean delta and for the between-model difference."""
    uniq, inverse = np.unique(groups, return_inverse=True)
    if len(uniq) < MIN_MATCHES_FOR_CI:
        return dict(na_reason=f"only {len(uniq)} matches (< {MIN_MATCHES_FOR_CI})")
    rows_of = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in deltas_by_key}
    diff = []
    for _ in range(replicates):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        w = np.concatenate([np.full(len(rows_of[i]), 1. / len(rows_of[i])) for i in pick])
        w = w / w.mean()
        means = {k: float(np.average(v[idx] * 100., weights=w)) for k, v in deltas_by_key.items()}
        for k, m in means.items():
            draws[k].append(m)
        if set(means) == set(FAMILIES):
            diff.append(means["expanded"] - means["maymin"])

    def interval(values):
        a = np.asarray(values)
        return dict(replicates=int(len(a)), mean_pp=float(a.mean()),
                    lo_pp=float(np.percentile(a, 2.5)), hi_pp=float(np.percentile(a, 97.5)),
                    excludes_zero=bool(np.percentile(a, 2.5) > 0 or np.percentile(a, 97.5) < 0))
    out = {k: interval(v) for k, v in draws.items()}
    if diff:
        out["expanded_minus_maymin"] = interval(diff)
    out["seed"], out["matches"] = seed, int(len(uniq))
    return out


# ------------------------------------------------------------------- banding

def band_of(values, bands):
    out = np.full(len(values), "", dtype=object)
    for lo, hi in bands:
        mask = np.isfinite(values) & (values >= lo) & (values < hi)
        out[mask] = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return out


def scale_class(scale):
    """-1 is the 'unknown participants' sentinel and must NOT fall into the pick bucket."""
    out = np.full(len(scale), "", dtype=object)
    out[scale < 0] = "unknown(-1)"
    out[(scale >= 0) & (scale <= 1)] = "pick(<=1)"
    out[(scale >= 2) & (scale <= 3)] = "skirmish(2-3)"
    out[scale >= 4] = "teamfight(>=4)"
    return out


def clock_interaction_diagnostic(dataset, v2_dir, *, sample=3000, advances=(15, 30, 60)):
    """Advance ONLY the clock on a real state and watch the probability move.

    state_value.py writes `key + "_x_time" = value * q / 1800000` for every team field, so the
    query timestamp multiplies ~94 columns on top of time_minutes and time_minutes_sq.  Here we
    hold every event- and frame-derived value fixed and recompute just those clock columns.  Any
    movement is therefore pure clock sensitivity, with no change in observed game state at all.
    This is deliberately conservative: recency features (death_age_minutes, *_age_minutes,
    *_acquired_last_*s, death_last_30s) are left untouched, so it UNDERSTATES the total effect.
    """
    import joblib
    models = {f: joblib.load(v2_dir / f"{f}_model.joblib") for f in FAMILIES}
    schema = json.loads(io.open(dataset / "schema.json", encoding="utf-8").read())
    names = schema["state_names"] if isinstance(schema, dict) else schema
    idx = {n: i for i, n in enumerate(names)}
    x_time = [(i, idx[n[:-len("_x_time")]]) for i, n in enumerate(names) if n.endswith("_x_time")]
    t_i, t2_i = idx["time_minutes"], idx["time_minutes_sq"]

    sampled = json.loads(io.open(v2_dir / "sampled_minutes.json", encoding="utf-8").read())
    used = {r["match"] for r in sampled}                       # never touch fit/calibrate/select
    states, qs = [], []
    for path in sorted((dataset / "matches").glob("*.npz")):
        if path.stem in used:
            continue
        with np.load(path, allow_pickle=False) as z:
            if not len(z["cutoff"]):
                continue
            states.append(z["pre"][0].astype(np.float64))
            qs.append(int(z["cutoff"][0]))
        if len(states) >= sample:
            break
    states, qs = np.asarray(states), np.asarray(qs, dtype=np.float64)

    def advanced(delta_s):
        out = states.copy()
        q_new = qs + delta_s * 1000.
        out[:, t_i] = q_new / 60000.
        out[:, t2_i] = (q_new / 60000.) ** 2
        for col, base in x_time:
            out[:, col] = out[:, base] * q_new / 1800000.
        return out

    base_rows = advanced(0)                                    # rebuild at +0 for an exact control
    report = {"n_states": int(len(states)), "source": "engagement-partition pre states, unused by any model fit",
              "columns_moved": len(x_time) + 2,
              "held_fixed": "all event- and frame-derived values, including recency features",
              "families": {}}
    for family, model in models.items():
        p0 = model.predict_proba(base_rows)[:, 1]
        cell = {"reconstruction_max_abs_error_pp": float(
            np.max(np.abs(p0 - model.predict_proba(states)[:, 1])) * 100.)}
        for d in advances:
            p = model.predict_proba(advanced(d))[:, 1]
            move = (p - p0) * 100.
            cell[f"+{d}s"] = dict(mean_pp=float(move.mean()), median_pp=float(np.median(move)),
                                  median_abs_pp=float(np.median(np.abs(move))),
                                  toward_red_fraction=float((move < 0).mean()),
                                  p90_abs_pp=float(np.quantile(np.abs(move), .9)))
        report["families"][family] = cell
    return report


def by_group(delta, groups, labels, *, order=None):
    keys = order if order is not None else sorted(set(labels) - {""})
    table = {}
    for key in keys:
        mask = labels == key
        table[key] = describe(delta[mask], groups[mask], label=key) if mask.any() \
            else dict(n=0, note="no rows")
    empty = (labels == "").sum()
    if empty:
        table["_unbanded"] = dict(n=int(empty), note="value outside every band (missing query)")
    return table


# ---------------------------------------------------------------------- main

def run(args):
    started = time.time()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def status(stage, **extra):
        (out / "status_b.json").write_text(json.dumps(
            dict(stage=stage, pid=os.getpid(),
                 updated_utc=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), **extra),
            indent=2, ensure_ascii=False), encoding="utf-8")

    status("running", step="load")
    with np.load(args.rows, allow_pickle=False) as z:
        rows = {k: z[k] for k in z.files}

    match = rows["match"].astype(str)
    valid = rows["valid"]
    scale = rows["scale"]
    snapshot = rows["snapshot_ms"]
    age = rows["age_s"]
    minute = rows["minute"]
    complete = rows["complete"].astype(bool)
    probs = {f: rows[f] for f in FAMILIES}
    deltas = {f: probs[f][:, 1:] - probs[f][:, :1] for f in FAMILIES}   # vs pre
    n = len(match)

    results = {
        "question": "How much does the reported delta-V depend on the choice of output boundary?",
        "interpretation_guard": (
            "Descriptive only. Neutral-band shares below are a SENSITIVITY table, not a proposed "
            "threshold; nothing here selects a winner, and no row is randomly assigned. Deltas are "
            "observed differences between two model evaluations, not causal effects of the fight."),
        "rows": int(n),
        "matches": int(len(set(match.tolist()))),
        "rows_with_all_four_points": int(complete.sum()),
    }

    # ---- B1: distribution per horizon, all available rows -------------------
    status("running", step="B1_distributions")
    results["B1_delta_distribution"] = {}
    for h in HORIZONS:
        j = POINT_INDEX[h]
        mask = valid[:, j] & valid[:, 0]
        cell = {"rows": int(mask.sum()), "matches": int(len(set(match[mask].tolist()))), "families": {}}
        for f in FAMILIES:
            cell["families"][f] = describe(deltas[f][mask, j - 1], match[mask], label=f"{h}:{f}")
        cell["bootstrap_mean_pp"] = bootstrap_mean_delta(
            {f: deltas[f][mask, j - 1] for f in FAMILIES}, match[mask])
        results["B1_delta_distribution"][h] = cell

    # ---- B2: common rows only - horizon and model agreement -----------------
    status("running", step="B2_agreement")
    cm = complete
    common = {"rows": int(cm.sum()), "matches": int(len(set(match[cm].tolist()))),
              "note": "all comparisons below use the 4-point common sample; validation A showed "
                      "the all-available sample is easier and would bias horizon comparisons",
              "horizon_distribution": {}, "direction_flip_between_horizons": {},
              "between_model": {}}
    for h in HORIZONS:
        j = POINT_INDEX[h]
        common["horizon_distribution"][h] = {
            f: describe(deltas[f][cm, j - 1], match[cm], label=f"common:{h}:{f}") for f in FAMILIES}
    pairs = (("last_kill_proxy", "plus_30s"), ("plus_30s", "plus_60s"), ("last_kill_proxy", "plus_60s"))
    for a, b in pairs:
        ja, jb = POINT_INDEX[a] - 1, POINT_INDEX[b] - 1
        entry = {}
        for f in FAMILIES:
            da, db = deltas[f][cm, ja], deltas[f][cm, jb]
            flip = np.sign(da) != np.sign(db)
            entry[f] = dict(
                flip_fraction=float(flip.mean()),
                flip_rows=int(flip.sum()),
                mean_abs_change_pp=float(np.mean(np.abs(db - da)) * 100.),
                median_abs_change_pp=float(np.median(np.abs(db - da)) * 100.),
                pearson_r=float(np.corrcoef(da, db)[0, 1]))
        common["direction_flip_between_horizons"][f"{a}->{b}"] = entry
    for h in HORIZONS:
        j = POINT_INDEX[h] - 1
        dm, de = deltas["maymin"][cm, j], deltas["expanded"][cm, j]
        agree = np.sign(dm) == np.sign(de)
        w = match_weights(match[cm])
        common["between_model"][h] = dict(
            direction_agreement=float(agree.mean()),
            direction_agreement_match_weighted=float(np.average(agree, weights=w)),
            median_abs_maymin_pp=float(np.median(np.abs(dm)) * 100.),
            median_abs_expanded_pp=float(np.median(np.abs(de)) * 100.),
            expanded_over_maymin_median_abs_ratio=float(
                np.median(np.abs(de)) / np.median(np.abs(dm))),
            pearson_r=float(np.corrcoef(dm, de)[0, 1]),
            mean_pp_maymin=float(dm.mean() * 100.), mean_pp_expanded=float(de.mean() * 100.))
    results["B2_common_sample"] = common

    # ---- B3: neutral-band sensitivity (reported, never adopted) -------------
    status("running", step="B3_neutral_band")
    band_table = {}
    for h in HORIZONS:
        j = POINT_INDEX[h] - 1
        band_table[h] = {}
        for f in FAMILIES:
            d = deltas[f][cm, j]
            band_table[h][f] = {f"|delta|<={t:g}": dict(
                threshold_pp=t * 100.,
                neutral_fraction=float((np.abs(d) <= t).mean()),
                neutral_rows=int((np.abs(d) <= t).sum())) for t in NEUTRAL_BANDS}
    results["B3_neutral_band_sensitivity"] = {
        "population": "common 4-point sample",
        "warning": "sensitivity only - do not adopt as the output threshold and do not assign "
                   "neutral rows to a winner at random",
        "table": band_table}

    # ---- B4: same-snapshot test, age, scale, phase --------------------------
    status("running", step="B4_decomposition")
    same_pre_end = (snapshot[:, 0] == snapshot[:, 1]) & valid[:, 0] & valid[:, 1]
    b4 = {"same_source_snapshot": {
        "hypothesis": "If pre and end resolve to the same source frame, no observed state changed "
                      "between them and the entire delta is produced by the clock alone.",
        "pre_end_same_frame_rows": int(same_pre_end.sum()),
        "pre_end_same_frame_fraction": float(same_pre_end.mean()),
        "groups": {}}}
    for name, mask in (("same_frame", same_pre_end),
                       ("different_frame", (~same_pre_end) & valid[:, 0] & valid[:, 1])):
        entry = {f: describe(deltas[f][mask, 0], match[mask], label=f"{name}:{f}") for f in FAMILIES}
        entry["bootstrap_mean_pp"] = bootstrap_mean_delta(
            {f: deltas[f][mask, 0] for f in FAMILIES}, match[mask])
        b4["same_source_snapshot"]["groups"][name] = entry
    # elapsed real time between the two queries, to separate "same frame" from "short fight"
    gap_s = (rows["query_ms"][:, 1] - rows["query_ms"][:, 0]) / 1000.
    b4["same_source_snapshot"]["pre_to_end_gap_s"] = {
        "all": {str(q): float(np.quantile(gap_s, q)) for q in (.25, .5, .75, .9)},
        "same_frame": {str(q): float(np.quantile(gap_s[same_pre_end], q)) for q in (.25, .5, .75, .9)},
        "different_frame": {str(q): float(np.quantile(gap_s[~same_pre_end], q)) for q in (.25, .5, .75, .9)}}

    b4["by_scale"] = {}
    classes = scale_class(scale)
    for h in HORIZONS:
        j = POINT_INDEX[h] - 1
        mask = valid[:, POINT_INDEX[h]] & valid[:, 0]
        b4["by_scale"][h] = {f: by_group(deltas[f][mask, j], match[mask], classes[mask],
                                         order=["pick(<=1)", "skirmish(2-3)", "teamfight(>=4)", "unknown(-1)"])
                             for f in FAMILIES}
    b4["scale_value_counts"] = {k: int((classes == k).sum()) for k in
                                ("pick(<=1)", "skirmish(2-3)", "teamfight(>=4)", "unknown(-1)")}

    b4["by_pre_snapshot_age_s"], b4["by_game_time_min"] = {}, {}
    age_labels, time_labels = band_of(age[:, 0], AGE_BANDS), band_of(minute[:, 0], TIME_BANDS)
    for h in HORIZONS:
        j = POINT_INDEX[h] - 1
        mask = valid[:, POINT_INDEX[h]] & valid[:, 0]
        b4["by_pre_snapshot_age_s"][h] = {
            f: by_group(deltas[f][mask, j], match[mask], age_labels[mask]) for f in FAMILIES}
        b4["by_game_time_min"][h] = {
            f: by_group(deltas[f][mask, j], match[mask], time_labels[mask]) for f in FAMILIES}
    b4["clock_interaction_diagnostic"] = clock_interaction_diagnostic(args.dataset, args.v2_dir)
    results["B4_decomposition"] = b4

    # ---- B5: who loses the follow-up observations ---------------------------
    status("running", step="B5_dropout")
    dropped60 = ~valid[:, 3]
    b5 = {"dropped_at_plus_60s": int(dropped60.sum()), "retained": int((~dropped60).sum()),
          "reason": "query past game end or past the last source snapshot",
          "comparison": {}}
    for name, mask in (("dropped", dropped60), ("retained", ~dropped60)):
        cell = {"rows": int(mask.sum()), "matches": int(len(set(match[mask].tolist()))),
                "game_time_min": {str(q): float(np.quantile(minute[mask, 0], q)) for q in (.25, .5, .75, .9)},
                "scale_shares": {k: float((scale_class(scale)[mask] == k).mean()) for k in
                                 ("pick(<=1)", "skirmish(2-3)", "teamfight(>=4)", "unknown(-1)")},
                "blue_win_rate": float(rows["y"][mask].mean()),
                "immediate_delta": {f: describe(deltas[f][mask, 0], match[mask], label=f"{name}:{f}")
                                    for f in FAMILIES}}
        b5["comparison"][name] = cell
    results["B5_followup_dropout"] = b5

    status("running", step="write")
    results["elapsed_seconds"] = round(time.time() - started, 2)
    results["status"] = "complete"
    (out / "results_b.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    from importlib.metadata import version
    (out / "manifest_b.json").write_text(json.dumps({
        "command": " ".join([sys.executable, *sys.argv]),
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "scikit-learn", "joblib")},
        "inputs": {"engagement_rows.npz": sha256(args.rows)},
        "source": {"scripts/validate_boundary_sensitivity_claude.py": sha256(Path(__file__))},
        "elapsed_seconds": results["elapsed_seconds"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    status("complete", elapsed_seconds=results["elapsed_seconds"], results="results_b.json")
    print(json.dumps({"rows": results["rows"], "matches": results["matches"],
                      "same_frame_fraction": b4["same_source_snapshot"]["pre_end_same_frame_fraction"],
                      "elapsed_seconds": results["elapsed_seconds"]}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--rows", type=Path,
                    default=root / "outputs/temporal_winprob_claude_validation/engagement_rows.npz")
    ap.add_argument("--out-dir", type=Path,
                    default=root / "outputs/temporal_winprob_claude_validation")
    ap.add_argument("--dataset", type=Path, default=root / "outputs/state_value_main_50k")
    ap.add_argument("--v2-dir", type=Path, default=root / "outputs/temporal_winprob_v2")
    args = ap.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "status_b.json").write_text(
            json.dumps({"stage": "failed", "error": repr(exc), "pid": os.getpid()}, indent=2),
            encoding="utf-8")
        raise


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
