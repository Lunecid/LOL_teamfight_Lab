"""Validation A: is the frozen win probability trustworthy AT ENGAGEMENT QUERY TIMES?

The v2 run established reliability on a random-minute population (one sampled minute per
match).  Every delta-V number, however, is read at engagement times: pre = cutoff-1ms,
end = last cluster kill, and end+30s / end+60s.  Those queries are conditioned on a kill
having happened and their snapshot ages are distributed differently from the training
grid, so random-minute calibration does not transfer by assumption.  This script scores
BOTH populations with the SAME frozen models and the SAME match-weighted metric code.

Read-only with respect to every frozen input: models, engagement probabilities, splits and
the source dataset are consumed, never rewritten.  Nothing here retrains, recalibrates or
reselects anything, and the source match cache is never opened.
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
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POINTS = ("pre", "last_kill_proxy", "plus_30s", "plus_60s")
FAMILIES = ("maymin", "expanded")
TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 10_000))
AGE_BANDS = ((0, 15), (15, 30), (30, 45), (45, 10_000))
BOOT_SEED = 7
BOOT_REPLICATES = 500
MIN_MATCHES_FOR_CI = 30


# ---------------------------------------------------------------- metric core

def match_weights(groups):
    """Every match contributes the same total weight, mean-normalised (v2 convention)."""
    _, ix, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / counts[ix]
    return w / w.mean()


def reliability(y, p, w, bins=10):
    """Fixed equal-width bins on [0, 1]; both the gap and ECE are match-weighted."""
    edges = np.linspace(0., 1., bins + 1)
    total = w.sum()
    rows, ece = [], 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi + (1e-10 if hi >= 1. else 0.))
        if not mask.any():
            rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=0, weight_share=0.,
                             predicted=None, observed=None, gap=None))
            continue
        ww = w[mask]
        pred = float(np.average(p[mask], weights=ww))
        obs = float(np.average(y[mask], weights=ww))
        share = float(ww.sum() / total)
        ece += share * abs(obs - pred)
        rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=int(mask.sum()), weight_share=share,
                         predicted=pred, observed=obs, gap=float(obs - pred)))
    return rows, float(ece)


def metrics(y, p, groups, *, label=""):
    """Match-weighted AUC / Brier / log loss / ECE plus the raw denominators."""
    if len(y) == 0:
        return dict(n=0, matches=0, note="empty subgroup", label=label)
    w = match_weights(groups)
    single_class = len(set(y.tolist())) < 2
    rows, ece = reliability(y, p, w)
    return dict(
        label=label,
        n=int(len(y)),
        matches=int(len(set(groups.tolist()))),
        positive_rate=float(np.average(y, weights=w)),
        auc=None if single_class else float(roc_auc_score(y, p, sample_weight=w)),
        auc_na_reason="single class in subgroup" if single_class else None,
        brier=float(brier_score_loss(y, p, sample_weight=w)),
        log_loss=float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
        ece=ece,
        reliability=rows,
    )


# ------------------------------------------------------------------ bootstrap

def match_bootstrap(y, p_by_family, groups, *, replicates=BOOT_REPLICATES, seed=BOOT_SEED):
    """Resample MATCHES with replacement; repeated matches count as distinct matches."""
    uniq, inverse = np.unique(groups, return_inverse=True)
    if len(uniq) < MIN_MATCHES_FOR_CI:
        return dict(na_reason=f"only {len(uniq)} matches (< {MIN_MATCHES_FOR_CI})", matches=int(len(uniq)))
    rows_of = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    sizes = np.array([len(r) for r in rows_of], dtype=float)
    rng = np.random.default_rng(seed)
    draws = {f: {"brier": [], "log_loss": []} for f in p_by_family}
    diff = {"brier": [], "log_loss": []}
    skipped = Counter()
    for _ in range(replicates):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        # A duplicated match keeps its own 1/n_rows weight, i.e. it is a distinct match.
        w = np.concatenate([np.full(len(rows_of[i]), 1. / sizes[i]) for i in pick])
        w = w / w.mean()
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            skipped["single_class_replicate"] += 1
            continue
        per_family = {}
        for family, p in p_by_family.items():
            pp = p[idx]
            per_family[family] = (float(brier_score_loss(yy, pp, sample_weight=w)),
                                  float(log_loss(yy, pp, labels=[0, 1], sample_weight=w)))
            draws[family]["brier"].append(per_family[family][0])
            draws[family]["log_loss"].append(per_family[family][1])
        if set(per_family) == set(FAMILIES):
            diff["brier"].append(per_family["expanded"][0] - per_family["maymin"][0])
            diff["log_loss"].append(per_family["expanded"][1] - per_family["maymin"][1])

    def interval(values):
        if len(values) < 2:
            return dict(na_reason="fewer than 2 usable replicates", replicates=len(values))
        a = np.asarray(values)
        return dict(replicates=int(len(a)), mean=float(a.mean()),
                    lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)))

    out = {"seed": seed, "replicates_requested": replicates, "matches": int(len(uniq)),
           "resample_unit": "match", "skipped": dict(skipped),
           "families": {f: {k: interval(v) for k, v in d.items()} for f, d in draws.items()}}
    out["expanded_minus_maymin"] = {k: interval(v) for k, v in diff.items()}
    for key, band in out["expanded_minus_maymin"].items():
        if "lo" in band:
            band["excludes_zero"] = bool(band["lo"] > 0 or band["hi"] < 0)
    return out


# ------------------------------------------------------------------- loading

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_path, workspace):
    """Re-check every frozen input the handoff pinned; report per file, never mutate."""
    manifest = json.loads(io.open(manifest_path, encoding="utf-8").read())
    checked, mismatched, missing = [], [], []
    for entry in manifest["files"]:
        target = workspace / entry["path"].replace("\\", "/")
        if not target.exists():
            missing.append(entry["path"])
            continue
        got = sha256(target)
        (checked if got == entry["sha256"] else mismatched).append(entry["path"])
    return dict(manifest=str(manifest_path), base_commit=manifest.get("base_commit"),
                files=len(manifest["files"]), ok=len(checked),
                mismatched=mismatched, missing=missing,
                verdict="all match" if not mismatched and not missing else "DISCREPANCY")


def match_winners(dataset, match_ids):
    """Final match outcome per match, straight from the frozen per-match npz."""
    winners = {}
    for mid in match_ids:
        with np.load(dataset / "matches" / (mid + ".npz"), allow_pickle=False) as z:
            winners[mid] = int(z["winner"])
    return winners


def engagement_population(v2_dir, dataset):
    """26,712 engagements x 4 query points, flattened to one row per (engagement, point)."""
    with np.load(v2_dir / "engagement_changes.npz", allow_pickle=False) as z:
        table = {k: z[k] for k in z.files}
    matches = table["match"].astype(str)
    winners = match_winners(dataset, sorted(set(matches.tolist())))
    y_match = np.array([winners[m] for m in matches])

    query = table["query_ms"].astype(np.int64)
    snapshot = table["snapshot_ms"].astype(np.int64)
    probs = {f: table[f].astype(np.float64) for f in FAMILIES}

    valid = np.isfinite(probs["expanded"]) & np.isfinite(probs["maymin"]) & (snapshot >= 0) & (query >= 0)
    age_s = np.where(valid, (query - snapshot) / 1000., np.nan)
    minute = np.where(valid, query / 60000., np.nan)
    complete = valid.all(axis=1)          # rows where all four points are observable
    return dict(id=table["id"].astype(str), match=matches, y=y_match, scale=table["scale"],
                query_ms=query, snapshot_ms=snapshot, probs=probs, valid=valid,
                age_s=age_s, minute=minute, complete=complete)


def random_minute_population(v2_dir, dataset, models):
    """The frozen v2 test set: one sampled minute per held-out match, rescored here."""
    sampled = json.loads(io.open(v2_dir / "sampled_minutes.json", encoding="utf-8").read())
    test = [r for r in sampled if r["role"] == "test"]
    states, y, groups = [], [], []
    for row in test:
        with np.load(dataset / "matches" / (row["match"] + ".npz"), allow_pickle=False) as z:
            k = int(np.flatnonzero(z["value_times"] == row["time_ms"])[0])
            states.append(z["value_states"][k])
            y.append(int(z["winner"]))
            groups.append(row["match"])
    states = np.asarray(states, dtype=np.float64)
    probs = {f: m.predict_proba(states)[:, 1] for f, m in models.items()}
    minute = np.array([r["time_ms"] for r in test]) / 60000.
    # snapshot_age_s is a state column; it is audit metadata, never an expanded covariate.
    schema = json.loads(io.open(dataset / "schema.json", encoding="utf-8").read())
    names = schema["state_names"] if isinstance(schema, dict) else schema
    if "snapshot_age_s" in names:
        age_idx = names.index("snapshot_age_s")
    age_s = states[:, age_idx] if age_idx is not None else np.full(len(states), np.nan)
    return dict(y=np.array(y), match=np.array(groups), probs=probs, minute=minute, age_s=age_s)


# ------------------------------------------------------------------ analysis

def band_of(values, bands):
    out = np.full(len(values), "", dtype=object)
    for lo, hi in bands:
        mask = np.isfinite(values) & (values >= lo) & (values < hi)
        out[mask] = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return out


def decompose(y, p, groups, values, bands, name):
    labels = band_of(values, bands)
    table = {}
    for lo, hi in bands:
        key = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        mask = labels == key
        table[key] = metrics(y[mask], p[mask], groups[mask], label=f"{name}:{key}") if mask.any() \
            else dict(n=0, matches=0, note="no rows in band")
    return table


def analyse_engagement(pop, *, subset, subset_name):
    """Per query point: headline metrics, decompositions and a match-level bootstrap."""
    result = {}
    for j, point in enumerate(POINTS):
        mask = pop["valid"][:, j] & subset
        y, groups = pop["y"][mask], pop["match"][mask]
        p_by_family = {f: pop["probs"][f][mask, j] for f in FAMILIES}
        entry = {"population": subset_name,
                 "rows_available": int(mask.sum()),
                 "rows_dropped_here": int((~pop["valid"][:, j] & subset).sum()),
                 "unique_matches": int(len(set(groups.tolist()))),
                 "families": {}}
        for family, p in p_by_family.items():
            entry["families"][family] = metrics(y, p, groups, label=f"{point}:{family}")
            entry["families"][family]["by_game_time_min"] = decompose(
                y, p, groups, pop["minute"][mask, j], TIME_BANDS, "time")
            entry["families"][family]["by_snapshot_age_s"] = decompose(
                y, p, groups, pop["age_s"][mask, j], AGE_BANDS, "age")
        entry["bootstrap"] = match_bootstrap(y, p_by_family, groups)
        result[point] = entry
    return result


def analyse_random_minute(pop):
    entry = {"population": "random_minute_test", "rows_available": int(len(pop["y"])),
             "unique_matches": int(len(set(pop["match"].tolist()))), "families": {}}
    for family, p in pop["probs"].items():
        entry["families"][family] = metrics(pop["y"], p, pop["match"], label=f"random_minute:{family}")
        entry["families"][family]["by_game_time_min"] = decompose(
            pop["y"], p, pop["match"], pop["minute"], TIME_BANDS, "time")
        entry["families"][family]["by_snapshot_age_s"] = decompose(
            pop["y"], p, pop["match"], pop["age_s"], AGE_BANDS, "age")
    entry["bootstrap"] = match_bootstrap(pop["y"], pop["probs"], pop["match"])
    return entry


# -------------------------------------------------------------------- output

def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def status(out, stage, **extra):
    write_json(out / "status.json", dict(stage=stage, pid=os.getpid(),
                                         updated_utc=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                                         **extra))


def reliability_figure(results, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                   # pragma: no cover
        return {"drawn": False, "reason": repr(exc)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, family in zip(axes, FAMILIES):
        ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls="--", label="perfect")
        rm = results["random_minute"]["families"][family]
        ax.plot([r["predicted"] for r in rm["reliability"] if r["n"]],
                [r["observed"] for r in rm["reliability"] if r["n"]],
                marker="s", color="black", label=f"random minute (ECE {rm['ece']:.4f})")
        for point in POINTS:
            cell = results["engagement_all_available"][point]["families"][family]
            ax.plot([r["predicted"] for r in cell["reliability"] if r["n"]],
                    [r["observed"] for r in cell["reliability"] if r["n"]],
                    marker="o", alpha=.85, label=f"{point} (ECE {cell['ece']:.4f})")
        ax.set_title(family)
        ax.set_xlabel("predicted P(blue win)")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("observed frequency")
    fig.suptitle("Reliability: random-minute population vs engagement query times")
    fig.tight_layout()
    path = out / "reliability_engagement_vs_random_minute.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"drawn": True, "path": path.name}


def save_rows(pop, out):
    """Row-level evidence so every number above can be recomputed without this script."""
    path = out / "engagement_rows.npz"
    np.savez_compressed(
        path, id=pop["id"], match=pop["match"], y=pop["y"], scale=pop["scale"],
        query_ms=pop["query_ms"], snapshot_ms=pop["snapshot_ms"], valid=pop["valid"],
        age_s=pop["age_s"], minute=pop["minute"], complete=pop["complete"],
        maymin=pop["probs"]["maymin"], expanded=pop["probs"]["expanded"],
        points=np.array(POINTS))
    return path.name


# ---------------------------------------------------------------------- main

def run(args):
    started = time.time()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    status(out, "running", step="verify_manifest")

    workspace = args.workspace
    integrity = verify_manifest(args.manifest, workspace)

    status(out, "running", step="load_models")
    import joblib
    models = {f: joblib.load(args.v2_dir / f"{f}_model.joblib") for f in FAMILIES}
    model_ids = {f: sha256(args.v2_dir / f"{f}_model.joblib") for f in FAMILIES}

    status(out, "running", step="load_engagement_population")
    pop = engagement_population(args.v2_dir, args.dataset)

    status(out, "running", step="score_random_minute")
    rm = random_minute_population(args.v2_dir, args.dataset, models)

    status(out, "running", step="analyse")
    results = {
        "question": "Does the frozen win probability stay calibrated AT ENGAGEMENT QUERY TIMES?",
        "interpretation_guard": (
            "This is an event-conditioned exploratory check. The engagement population is "
            "conditioned on a kill having occurred, so it is NOT exchangeable with the "
            "random-minute population; differences between the two tables are not evidence "
            "that one model or one population is causally better."),
        "integrity": integrity,
        "model_ids": model_ids,
        "environment_note": (
            "Frozen models were pickled under scikit-learn 1.6.1. They are unusable under the "
            "1.8.0 interpreter in the project .venv (SimpleImputer raises AttributeError), so "
            "this run uses the 1.6.1 environment and reproduces the stored v2 test metrics."),
        "denominators": {
            "engagements": int(len(pop["id"])),
            "engagement_matches": int(len(set(pop["match"].tolist()))),
            "valid_rows_per_point": {p: int(pop["valid"][:, j].sum()) for j, p in enumerate(POINTS)},
            "dropped_rows_per_point": {p: int((~pop["valid"][:, j]).sum()) for j, p in enumerate(POINTS)},
            "drop_reason": "query point past game end or past the last source snapshot; recorded as -1 by observed_queries()",
            "rows_with_all_four_points": int(pop["complete"].sum()),
        },
    }
    results["random_minute"] = analyse_random_minute(rm)
    results["engagement_all_available"] = analyse_engagement(
        pop, subset=np.ones(len(pop["id"]), dtype=bool), subset_name="all_available")
    results["engagement_common_four_points"] = analyse_engagement(
        pop, subset=pop["complete"], subset_name="common_four_points")

    status(out, "running", step="write_outputs")
    results["figures"] = reliability_figure(results, out)
    results["row_level_evidence"] = save_rows(pop, out)
    results["elapsed_seconds"] = round(time.time() - started, 2)
    results["status"] = "complete"
    write_json(out / "results.json", results)

    from importlib.metadata import version
    write_json(out / "manifest.json", {
        "command": " ".join([sys.executable, *sys.argv]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "scikit-learn", "joblib")},
        "inputs": {name: sha256(path) for name, path in (
            ("engagement_changes.npz", args.v2_dir / "engagement_changes.npz"),
            ("sampled_minutes.json", args.v2_dir / "sampled_minutes.json"),
            ("results.json(v2)", args.v2_dir / "results.json"),
            ("maymin_model.joblib", args.v2_dir / "maymin_model.joblib"),
            ("expanded_model.joblib", args.v2_dir / "expanded_model.joblib"))},
        "source": {name: sha256(workspace / name) for name in (
            "scripts/validate_temporal_winprob_claude.py",
            "train/temporal_winprob.py",
            "gameplay/state_value.py")},
        "manifest_verification": integrity,
        "elapsed_seconds": results["elapsed_seconds"],
        "wrote_nothing_outside": str(out),
    })
    status(out, "complete", elapsed_seconds=results["elapsed_seconds"], results="results.json")
    print(json.dumps({k: results[k] for k in ("denominators", "integrity", "elapsed_seconds")},
                     indent=2, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--workspace", type=Path, default=root)
    ap.add_argument("--v2-dir", type=Path, default=root / "outputs/temporal_winprob_v2")
    ap.add_argument("--dataset", type=Path, default=root / "outputs/state_value_main_50k")
    ap.add_argument("--manifest", type=Path, default=root / "docs/CLAUDE_NEXT_STAGE_MANIFEST.json")
    ap.add_argument("--out-dir", type=Path, default=root / "outputs/temporal_winprob_claude_validation")
    args = ap.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        status(args.out_dir, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
