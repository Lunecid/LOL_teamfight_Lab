"""Estimate the kill-episode temporal boundary from the corpus and report its stability.

Reads champion-kill events from the match cache, builds consecutive inter-kill
intervals per match, and runs the estimators in analysis/temporal_boundary.py:
KDE valley (with a bandwidth sweep), two-component mixture crossing, a cluster
bootstrap over matches, strata by patch / game phase / match kill count, and
the clustering-agreement plateau (ARI) across gap values.  Also stores every
consecutive kill pair with positions and participant overlap, so the spatial
boundary can be analysed by map region without re-reading the cache.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_temporal_boundary.py ^
        --n-matches 10000 --seed 7 --out-dir D:/LOL_Project/fusion_2615/features/temporal_boundary
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOL_OUTPUT_ROOT", "D:/LOL_Project")
# torch (pulled in by gameplay.fights) and scikit-learn each ship an OpenMP runtime;
# without this the process aborts with "OMP: Error #15" on Windows/conda.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from analysis.temporal_boundary import (  # noqa: E402
    bandwidth_sweep, cluster_bootstrap, consecutive_intervals, episode_counts,
    kde_valley, mixture_crossing, plateau_ari, stratified_valleys, to_log,
)
from analysis.map_regions import classify_points  # noqa: E402

GAPS = [6, 8, 10, 12, 13, 14, 15, 16, 18, 20, 22, 24, 27, 30, 36, 45, 60]


def phase_of_minute(m: float) -> str:
    if m < 10:
        return "0-10 min"
    if m < 20:
        return "10-20 min"
    if m < 30:
        return "20-30 min"
    return "30+ min"


def extract(n_matches: int, seed: int):
    from core.config import CACHE_DIR
    from data.cache_io import load_match_cache
    from gameplay.fights import _extract_kill_events

    mids = sorted(p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json"))
    if n_matches and n_matches < len(mids):
        mids = sorted(random.Random(seed).sample(mids, n_matches))

    records = []          # per match: ts (s), log_dt, patch, n_kills
    pairs = {k: [] for k in ("match_idx", "dt", "dd", "shared", "x1", "y1", "x2", "y2", "minute", "same_victim_team")}
    patches = []
    t0 = time.time()
    for done, mid in enumerate(mids, 1):
        pack = load_match_cache(mid)
        if not pack:
            continue
        kills = _extract_kill_events(pack.get("events") or [])
        if len(kills) < 2:
            continue
        tm = pack["meta"].get("team_map", {}) or {}
        patch = str(pack["meta"].get("patch", "?"))
        ts = np.array([k["timestamp"] for k in kills], dtype=np.float64) / 1000.0
        dt = consecutive_intervals(ts)
        rec = {"match_id": mid, "patch": patch, "ts": ts, "log_dt": to_log(dt), "n_kills": len(kills)}
        mi = len(records)
        records.append(rec)
        patches.append(patch)
        parts = []
        for k in kills:
            s = set()
            for pid in [k.get("killer_id", 0), k.get("victim_id", 0)] + list(k.get("assisting_ids", []) or []):
                try:
                    pid = int(pid)
                except Exception:
                    continue
                if 1 <= pid <= 10:
                    s.add(pid)
            parts.append(s)
        vteam = [tm.get(str(k.get("victim_id", 0)), tm.get(k.get("victim_id", 0), -1)) for k in kills]
        for i in range(len(kills) - 1):
            j = i + 1
            p1, p2 = kills[i].get("position"), kills[j].get("position")
            if not p1 or not p2:
                continue
            d = ts[j] - ts[i]
            if d <= 0:
                continue
            pairs["match_idx"].append(mi); pairs["dt"].append(d)
            pairs["dd"].append(float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
            pairs["shared"].append(len(parts[i] & parts[j]) > 0)
            pairs["x1"].append(p1[0]); pairs["y1"].append(p1[1]); pairs["x2"].append(p2[0]); pairs["y2"].append(p2[1])
            pairs["minute"].append(ts[i] / 60.0)
            pairs["same_victim_team"].append(vteam[i] == vteam[j])
        if done % 1000 == 0:
            print(f"  {done}/{len(mids)} matches, {len(pairs['dt']):,} pairs, {time.time() - t0:.0f}s", flush=True)
    P = {k: np.asarray(v) for k, v in pairs.items()}
    P["patch"] = np.asarray(patches)
    return records, P


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-matches", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--bandwidth", type=float, default=0.08)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("extracting ...", flush=True)
    records, P = extract(args.n_matches, args.seed)
    all_log = np.concatenate([r["log_dt"] for r in records])
    R = {"n_matches": len(records), "n_kills": int(sum(r["n_kills"] for r in records)),
         "n_intervals": int(all_log.size), "seed": args.seed, "bandwidth": args.bandwidth,
         "patches": sorted(set(r["patch"] for r in records))}
    print(f"matches={R['n_matches']} kills={R['n_kills']:,} intervals={R['n_intervals']:,}", flush=True)

    # 1. point estimates
    R["kde"] = kde_valley(all_log, bandwidth=args.bandwidth).as_dict()
    R["bandwidth_sweep"] = bandwidth_sweep(all_log)
    R["mixture"] = mixture_crossing(all_log, seed=args.seed)
    R["interval_quantiles_s"] = {str(q): float(10 ** np.percentile(all_log, q)) for q in (10, 25, 50, 75, 90)}
    print("kde:", R["kde"], "\nmixture:", R["mixture"], flush=True)

    # 2. bootstrap over matches
    R["bootstrap"] = cluster_bootstrap([r["log_dt"] for r in records], n_boot=args.n_boot, seed=args.seed,
                                       bandwidth=args.bandwidth)
    print("bootstrap valley:", R["bootstrap"].get("valley_s"), flush=True)

    # 3. strata
    R["by_patch"] = stratified_valleys(records, key=lambda r: r["patch"], bandwidth=args.bandwidth)
    # game phase: split each match's intervals by the minute of the earlier kill
    phase_recs = []
    for r in records:
        ts = r["ts"]; d = np.diff(np.sort(ts)); m = ts[:-1] / 60.0
        for ph in ("0-10 min", "10-20 min", "20-30 min", "30+ min"):
            sel = np.array([phase_of_minute(x) == ph for x in m]) & (d > 0)
            if sel.any():
                phase_recs.append({"log_dt": np.log10(d[sel]), "phase": ph})
    R["by_phase"] = stratified_valleys(phase_recs, key=lambda r: r["phase"], bandwidth=args.bandwidth)
    kills = np.array([r["n_kills"] for r in records])
    terc = np.percentile(kills, [33.3, 66.7])
    R["by_kill_count"] = stratified_valleys(
        records, key=lambda r: ("low" if r["n_kills"] <= terc[0] else "mid" if r["n_kills"] <= terc[1] else "high"),
        bandwidth=args.bandwidth)
    R["kill_count_tertiles"] = [float(t) for t in terc]

    # 4. plateau: clustering agreement across gaps, vs 18 s and vs the valley
    per_ts = [r["ts"] for r in records]
    R["ari_vs_18s"] = plateau_ari(per_ts, GAPS, ref_gap_s=18.0)
    v = R["kde"]["valley_s"]
    R["ari_vs_valley"] = plateau_ari(per_ts, GAPS, ref_gap_s=v) if np.isfinite(v) else {}
    R["episodes_per_match"] = episode_counts(per_ts, GAPS)

    # 5. save
    json.dump(R, open(args.out_dir / "temporal_boundary.json", "w"), indent=1)
    labels1, tan1 = classify_points(np.stack([P["x1"], P["y1"]], axis=1))
    labels2, _ = classify_points(np.stack([P["x2"], P["y2"]], axis=1))
    np.savez_compressed(args.out_dir / "kill_pairs.npz", **P, region1=labels1, region2=labels2, tangent1=tan1)
    print("saved", args.out_dir / "temporal_boundary.json", "and kill_pairs.npz", flush=True)

    # 6. figure
    try:
        _figure(R, records, args.out_dir / "temporal_boundary.png", args.bandwidth)
    except Exception as e:  # figure is optional
        print("figure failed:", e)
    return 0


def _figure(R, records, path, bw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    BLUE, ORANGE, AQUA, YEL, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#1a1a19", "#5a5a58", "#e6e6e3"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor="#fcfcfb")
    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.99, top=0.84, bottom=0.15)
    g = np.linspace(-0.5, 3.0, 800)

    ax = axes[0]
    all_log = np.concatenate([r["log_dt"] for r in records])
    ax.plot(g, gaussian_kde(all_log, bw_method=bw)(g), color=BLUE, lw=2.2, label="all")
    cols = [ORANGE, AQUA, YEL, "#e87ba4"]
    phase_log = {}
    for r in records:
        ts = np.sort(r["ts"]); d = np.diff(ts); m = ts[:-1] / 60.0
        for ph, dd in zip([phase_of_minute(x) for x in m], d):
            if dd > 0:
                phase_log.setdefault(ph, []).append(np.log10(dd))
    for (k, v), c in zip(R["by_phase"].items(), cols):
        x = np.asarray(phase_log.get(k, []))
        if x.size > 200:
            ax.plot(g, gaussian_kde(x, bw_method=bw)(g), color=c, lw=1.3,
                    label=f"{k}: valley {v['valley_s']:.0f} s" if v["ok"] else f"{k}: no valley")
    ax.axvline(np.log10(R["kde"]["valley_s"]), color=ORANGE, ls="-.", lw=1.4)
    ax.axvline(np.log10(18), color=INK, lw=1.2)
    b = R["bootstrap"].get("valley_s") or {}
    if b:
        ax.axvspan(np.log10(b["p2.5"]), np.log10(b["p97.5"]), color=ORANGE, alpha=0.15)
    ax.text(np.log10(R["kde"]["valley_s"]) + 0.03, 0.05, f"valley {R['kde']['valley_s']:.1f} s\n95% CI {b.get('p2.5', float('nan')):.1f}-{b.get('p97.5', float('nan')):.1f}", fontsize=8, color=ORANGE)
    ax.text(np.log10(18) + 0.03, 0.3, "18 s", fontsize=8)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5]); ax.set_xticklabels(["1", "3", "10", "32", "100", "316"])
    ax.set_xlabel("consecutive inter-kill interval, s (log)"); ax.set_ylabel("density")
    ax.set_title("a. Interval density and the valley", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    groups = [("patch", R["by_patch"]), ("phase", R["by_phase"]), ("kills", R["by_kill_count"])]
    y = 0; ticks = []; labs = []
    for gname, d in groups:
        for k, v in d.items():
            if v["ok"]:
                ax.plot([np.log10(v["mode1_s"]), np.log10(v["mode2_s"])], [y, y], color=GRID, lw=3)
                ax.plot(np.log10(v["valley_s"]), y, "o", color=ORANGE, ms=7, mec="#fcfcfb")
                ax.text(np.log10(v["mode2_s"]) + 0.05, y, f"{v['valley_s']:.0f} s, depth {v['depth']:.2f}", va="center", fontsize=8)
            else:
                ax.text(0.35, y, f"no valley (search {v['search_lo_s']:.0f}-{v['search_hi_s']:.0f} s, depth {v['depth']:.2f})",
                        va="center", fontsize=8, color=INK2)
            ticks.append(y); labs.append(f"{gname}: {k}"); y += 1
    ax.axvline(np.log10(R["kde"]["valley_s"]), color=ORANGE, ls="-.", lw=1.2); ax.axvline(np.log10(18), color=INK, lw=1.0)
    ax.set_yticks(ticks); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xticks([0.5, 1, 1.5, 2]); ax.set_xticklabels(["3", "10", "32", "100"]); ax.set_xlim(0.3, 2.6)
    ax.set_xlabel("s (log): grey = mode-to-mode span, dot = valley"); ax.set_title("b. Valley by stratum", loc="left", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    ax = axes[2]
    xs = [float(k) for k in R["ari_vs_18s"]]
    ax.plot(np.log10(xs), list(R["ari_vs_18s"].values()), color=BLUE, lw=2, marker="o", ms=4, mfc="#fcfcfb", label="ARI vs 18 s")
    if R["ari_vs_valley"]:
        ax.plot(np.log10(xs), list(R["ari_vs_valley"].values()), color=ORANGE, lw=2, marker="o", ms=4, mfc="#fcfcfb", label=f"ARI vs valley {R['kde']['valley_s']:.0f} s")
    ax.axhline(0.9, color=GRID, lw=1); ax.text(np.log10(6), 0.905, "0.90", fontsize=8, color=INK2)
    ax.set_xticks(np.log10([6, 10, 18, 30, 60])); ax.set_xticklabels(["6", "10", "18", "30", "60"])
    ax.set_ylim(0.5, 1.02); ax.set_xlabel("gap G, s (log)"); ax.set_ylabel("adjusted Rand index")
    ax.set_title("c. How much the episodes change with G", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    fig.suptitle(f"Temporal boundary from data: {R['n_matches']:,} matches, {R['n_intervals']:,} intervals, patches {', '.join(R['patches'])}",
                 x=0.05, ha="left", fontsize=11)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    raise SystemExit(main())
