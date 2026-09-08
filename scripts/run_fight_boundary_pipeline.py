"""Run the fight-boundary definition pipeline per patch and pooled.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_fight_boundary_pipeline.py ^
        --n-matches-per-patch 3000 --seed 7 --out-dir D:/LOL_Project/fusion_2615/features/fight_boundary

Outputs, in --out-dir:
    spec_pooled.json, spec_<patch>.json     the definitions (BoundarySpec)
    details_<scope>.json                    every intermediate estimate
    drift.json, drift.md                    per-patch vs pooled comparison and verdict
    pairs_<patch>.npz                       kill records + pairs (re-analysis without the cache)
    fight_boundary.png                      G and D per patch against the pooled bands
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOL_OUTPUT_ROOT", "D:/LOL_Project")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + scikit-learn OpenMP clash on Windows/conda
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from analysis.boundary_spec import BoundarySpec  # noqa: E402
from analysis.fight_boundary_pipeline import build_spec, drift_decision, drift_markdown  # noqa: E402
from analysis.kill_pairs import extract_kill_pairs, group_by_patch, list_cache_match_ids, load_pairs, sample_ids, save_pairs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-matches-per-patch", type=int, default=3000, help="0 = every match of the patch")
    ap.add_argument("--patches", nargs="*", default=None, help="default: every patch in the cache")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--bandwidth", type=float, default=0.08)
    ap.add_argument("--datadragon", action="store_true", help="estimate R per patch from Data Dragon (network)")
    ap.add_argument("--reuse-pairs", action="store_true", help="load pairs_<patch>.npz from out-dir instead of the cache")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731

    # ---- 1. corpus slices ----
    slices = {}
    if args.reuse_pairs:
        for p in sorted(args.out_dir.glob("pairs_*.npz")):
            patch = p.stem.replace("pairs_", "")
            if args.patches and patch not in args.patches:
                continue
            slices[patch] = load_pairs(p)
            log(f"loaded {p.name}: {len(slices[patch][0])} matches")
    else:
        mids = list_cache_match_ids()
        log(f"cache: {len(mids):,} matches; scanning patches ...")
        by_patch = group_by_patch(mids, progress=log)
        log("patches: " + ", ".join(f"{k}={len(v):,}" for k, v in sorted(by_patch.items())))
        for patch in sorted(by_patch):
            if args.patches and patch not in args.patches:
                continue
            ids = sample_ids(by_patch[patch], args.n_matches_per_patch, args.seed)
            log(f"[{patch}] extracting {len(ids)} matches")
            records, pairs = extract_kill_pairs(ids, progress=log)
            save_pairs(args.out_dir / f"pairs_{patch}.npz", records, pairs)
            slices[patch] = (records, pairs)
    if not slices:
        raise SystemExit("no corpus slices")

    # ---- 2. R per patch (optional) ----
    radius = {}
    if args.datadragon:
        from analysis.ability_range import estimate_validity_radius
        for patch in slices:
            radius[patch] = estimate_validity_radius(patch, cache_dir=args.out_dir / "datadragon")
            log(f"[{patch}] R: {radius[patch]}")

    # ---- 3. pooled spec first (it is the fallback for thin slices) ----
    all_records = [r for recs, _ in slices.values() for r in recs]
    keys = [k for k in next(iter(slices.values()))[1].keys()]
    pooled_pairs = {}
    offset = 0
    for patch, (recs, pairs) in slices.items():
        for k in keys:
            v = pairs[k]
            if k == "match_idx":
                v = v + offset
            pooled_pairs.setdefault(k, []).append(v)
        offset += len(recs)
    pooled_pairs = {k: np.concatenate(v) for k, v in pooled_pairs.items()}
    pooled_radius = next((r for r in radius.values() if r.get("ok")), None)
    pooled, pooled_details = build_spec("pooled", sorted(slices), all_records, pooled_pairs,
                                        n_boot=args.n_boot, seed=args.seed, bandwidth=args.bandwidth,
                                        radius_info=pooled_radius)
    pooled.to_json(args.out_dir / "spec_pooled.json")
    json.dump(pooled_details, open(args.out_dir / "details_pooled.json", "w"), indent=1, default=_json_default)
    log(pooled.summary())

    # ---- 4. per patch ----
    per_patch = {}
    for patch, (recs, pairs) in slices.items():
        spec, details = build_spec(f"patch:{patch}", [patch], recs, pairs, n_boot=args.n_boot, seed=args.seed,
                                   bandwidth=args.bandwidth, pooled=pooled, radius_info=radius.get(patch))
        spec.to_json(args.out_dir / f"spec_{patch}.json")
        json.dump(details, open(args.out_dir / f"details_{patch}.json", "w"), indent=1, default=_json_default)
        per_patch[patch] = spec
        log(spec.summary())

    # ---- 5. drift verdict ----
    decision = drift_decision(per_patch, pooled)
    json.dump(decision, open(args.out_dir / "drift.json", "w"), indent=1)
    md = drift_markdown(decision, per_patch, pooled)
    (args.out_dir / "drift.md").write_text(md, encoding="utf-8")
    log("\n" + md)

    try:
        _figure(per_patch, pooled, decision, args.out_dir / "fight_boundary.png")
    except Exception as e:
        log(f"figure failed: {e}")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _figure(per_patch, pooled: BoundarySpec, decision, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#5a5a58", "#e6e6e3"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    fig.subplots_adjust(wspace=0.35, left=0.12, right=0.98, top=0.82, bottom=0.16)
    names = list(per_patch)
    y = np.arange(len(names))

    ax = axes[0]
    if pooled.gap_plateau_s:
        ax.axvspan(pooled.gap_plateau_s[0], pooled.gap_plateau_s[1], color=BLUE, alpha=0.10)
        ax.text(pooled.gap_plateau_s[0] + 0.2, -0.65, f"pooled plateau (ARI >= {pooled.plateau_ari:g})", fontsize=8, color=BLUE)
    ax.axvline(pooled.gap_s, color=BLUE, lw=1.4, ls="-."); ax.axvline(18, color=INK, lw=1.0)
    for i, n in enumerate(names):
        s = per_patch[n]
        if s.gap_ci_s:
            ax.plot(s.gap_ci_s, [i, i], color=GRID, lw=4)
        ax.plot(s.gap_s, i, "o", color=ORANGE if s.gap_source == "valley" else INK2, ms=7, mec="#fcfcfb")
        ax.text(s.gap_s + 0.4, i - 0.25, f"{s.gap_s:.1f} s" + ("" if s.gap_source == "valley" else f" ({s.gap_source})"), fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_ylim(len(names) - 0.4, -0.9); ax.set_xlim(5, 30)
    ax.set_xlabel("temporal boundary G, s"); ax.set_title("a. G per patch vs pooled", loc="left", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    ax = axes[1]
    tol = decision["diameter_rel_tol"]
    ax.axvspan(pooled.diameter_u * (1 - tol), pooled.diameter_u * (1 + tol), color=BLUE, alpha=0.10)
    ax.text(pooled.diameter_u * (1 - tol) + 30, -0.65, f"pooled +-{int(tol * 100)}%", fontsize=8, color=BLUE)
    ax.axvline(pooled.diameter_u, color=BLUE, lw=1.4, ls="-."); ax.axvline(4000, color=INK, lw=1.0)
    for i, n in enumerate(names):
        s = per_patch[n]
        if s.diameter_ci_u:
            ax.plot(s.diameter_ci_u, [i, i], color=GRID, lw=4)
        ax.plot(s.diameter_u, i, "o", color=ORANGE if s.diameter_source == "crossover" else INK2, ms=7, mec="#fcfcfb")
        ax.text(s.diameter_u + 60, i - 0.25, f"{s.diameter_u:,.0f} u", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_ylim(len(names) - 0.4, -0.9); ax.set_xlim(3000, 6000)
    ax.set_xlabel("spatial boundary D, game units"); ax.set_title("b. D per patch vs pooled", loc="left", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    fig.suptitle(f"Fight boundary per patch: verdict = {decision['verdict']}  (pooled G {pooled.gap_s:.1f} s, D {pooled.diameter_u:,.0f} u)",
                 x=0.12, ha="left", fontsize=11)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    raise SystemExit(main())
