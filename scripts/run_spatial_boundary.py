"""Spatial boundary of kill episodes, by map region and lane geometry.

Consumes kill_pairs.npz written by run_temporal_boundary.py (consecutive kill
pairs with positions, participant overlap and region labels) and asks, for
pairs inside the temporal window:

* at what distance does P(the two kills share a champion) cross 0.5, and does
  that crossover differ between lanes, jungle, river, pits and bases?
* in lanes, is the boundary anisotropic -- do same-fight kills stretch further
  along the lane than across it?
* how often do consecutive kills within the window fall in different regions,
  and how much sharing survives a region change?

    python scripts/run_spatial_boundary.py ^
        --pairs D:/LOL_Project/fusion_2615/features/temporal_boundary/kill_pairs.npz ^
        --gap-s 18 --out-dir D:/LOL_Project/fusion_2615/features/temporal_boundary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.map_regions import REGIONS, along_across  # noqa: E402

DIST_EDGES = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000, 12000, 1e9])


def crossover(dd: np.ndarray, sh: np.ndarray, level: float = 0.5, min_n: int = 40):
    """Distance at which the binned sharing rate first drops below ``level``.

    Linear interpolation between bin centres; None if the rate never drops
    (too few far pairs) or never exceeds the level.
    """
    xs, ys = [], []
    for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:]):
        m = (dd > lo) & (dd <= hi)
        if m.sum() >= min_n:
            xs.append((lo + min(hi, 14000)) / 2); ys.append(float(sh[m].mean()))
    if len(xs) < 2 or ys[0] < level:
        return None, list(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        if y0 >= level > y1:
            return float(x0 + (y0 - level) / max(y0 - y1, 1e-9) * (x1 - x0)), list(zip(xs, ys))
    return None, list(zip(xs, ys))


def bootstrap_crossover(dd, sh, match_idx, n_boot=200, seed=7):
    rng = np.random.default_rng(seed)
    mids = np.unique(match_idx)
    by = {m: np.where(match_idx == m)[0] for m in mids}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(mids, mids.size, replace=True)
        idx = np.concatenate([by[m] for m in pick])
        c, _ = crossover(dd[idx], sh[idx])
        if c is not None:
            vals.append(c)
    if not vals:
        return None
    v = np.asarray(vals)
    return {"p2.5": float(np.percentile(v, 2.5)), "p50": float(np.percentile(v, 50)),
            "p97.5": float(np.percentile(v, 97.5)), "found": len(vals) / n_boot}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--gap-s", type=float, default=18.0)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    Z = np.load(args.pairs, allow_pickle=True)
    dt, dd, sh = Z["dt"], Z["dd"], Z["shared"].astype(bool)
    r1, r2 = Z["region1"].astype(str), Z["region2"].astype(str)
    tan = Z["tangent1"]
    mi = Z["match_idx"]
    disp = np.stack([Z["x2"] - Z["x1"], Z["y2"] - Z["y1"]], axis=1)
    win = dt <= args.gap_s
    R = {"n_pairs": int(dt.size), "n_pairs_in_window": int(win.sum()), "gap_s": args.gap_s,
         "n_matches": int(np.unique(mi).size)}

    # 1. region mix of first kills inside the window
    u, c = np.unique(r1[win], return_counts=True)
    R["region_share_first_kill"] = {str(k): float(v / win.sum()) for k, v in zip(u, c)}

    # 2. crossover distance by region of the first kill
    R["by_region"] = {}
    for reg in REGIONS:
        m = win & (r1 == reg)
        if m.sum() < 200:
            continue
        cx, curve = crossover(dd[m], sh[m])
        boot = bootstrap_crossover(dd[m], sh[m], mi[m], n_boot=args.n_boot)
        R["by_region"][reg] = {"n": int(m.sum()), "crossover_u": cx, "bootstrap": boot,
                               "share_le_2000": float(np.mean(dd[m] <= 2000)), "share_le_4000": float(np.mean(dd[m] <= 4000)),
                               "shared_rate_le_4000": float(sh[m & (dd <= 4000)].mean()) if (m & (dd <= 4000)).any() else None,
                               "shared_rate_gt_4000": float(sh[m & (dd > 4000)].mean()) if (m & (dd > 4000)).any() else None,
                               "curve": curve}
    cx_all, curve_all = crossover(dd[win], sh[win])
    R["all"] = {"n": int(win.sum()), "crossover_u": cx_all, "bootstrap": bootstrap_crossover(dd[win], sh[win], mi[win], n_boot=args.n_boot),
                "curve": curve_all}

    # 3. lane anisotropy: along vs across the lane tangent at the first kill
    R["lane_anisotropy"] = {}
    for reg in ("top", "mid", "bot"):
        m = win & (r1 == reg) & (np.abs(tan).sum(axis=1) > 0)
        if m.sum() < 200:
            continue
        along, across = along_across(disp[m], tan[m])
        s = sh[m]
        out = {"n": int(m.sum())}
        # Conditional crossovers: vary one component while holding the other
        # small, so a displacement into the jungle is not counted against the
        # along-lane boundary and vice versa.  (Marginal crossovers mix the two
        # directions and are misleading.)
        for name, comp, other, cap in (("along", along, across, 500.0), ("across", across, along, 1000.0)):
            keep = other <= cap
            cx, curve = crossover(comp[keep], s[keep])
            out[name] = {"crossover_u": cx, "conditioned_on": f"other component <= {cap:.0f} u", "n": int(keep.sum()),
                         "median_u_shared": float(np.median(comp[s])) if s.any() else None,
                         "p90_u_shared": float(np.percentile(comp[s], 90)) if s.any() else None, "curve": curve}
        # sharing rate on a 2-D along x across grid (coarse)
        grid = {}
        for lo_a, hi_a in ((0, 1000), (1000, 2000), (2000, 3000), (3000, 4500), (4500, 7000)):
            for lo_c, hi_c in ((0, 500), (500, 1000), (1000, 2000), (2000, 4000)):
                g = (along > lo_a) & (along <= hi_a) & (across > lo_c) & (across <= hi_c)
                if g.sum() >= 40:
                    grid[f"along{lo_a}-{hi_a}|across{lo_c}-{hi_c}"] = {"n": int(g.sum()), "shared": float(s[g].mean())}
        out["grid"] = grid
        R["lane_anisotropy"][reg] = out

    # 4. region transitions inside the window
    same = r1 == r2
    R["region_transition"] = {"same_region_share": float(np.mean(same[win])),
                              "shared_rate_same_region": float(sh[win & same].mean()),
                              "shared_rate_diff_region": float(sh[win & ~same].mean()),
                              "shared_rate_diff_region_le_4000": float(sh[win & ~same & (dd <= 4000)].mean()) if (win & ~same & (dd <= 4000)).any() else None}
    trans = {}
    for a in REGIONS:
        for b in REGIONS:
            m = win & (r1 == a) & (r2 == b)
            if m.sum() >= 100:
                trans[f"{a}->{b}"] = {"n": int(m.sum()), "shared": float(sh[m].mean()), "dd_median": float(np.median(dd[m]))}
    R["transitions"] = trans

    json.dump(R, open(args.out_dir / "spatial_boundary.json", "w"), indent=1)
    print(json.dumps({k: (v if k != "by_region" else {r: {"n": d["n"], "crossover_u": d["crossover_u"], "boot": d["bootstrap"]} for r, d in v.items()})
                      for k, v in R.items() if k in ("n_pairs", "n_pairs_in_window", "region_share_first_kill", "by_region", "region_transition")}, indent=1))
    print("all crossover:", R["all"]["crossover_u"], R["all"]["bootstrap"])
    for reg, d in R["lane_anisotropy"].items():
        print(reg, "along:", d["along"]["crossover_u"], "across:", d["across"]["crossover_u"], "n", d["n"])
    try:
        _figure(R, args.out_dir / "spatial_boundary.png")
    except Exception as e:
        print("figure failed:", e)
    return 0


def _figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA, YEL, MAG, GREEN, VIO, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
    INK, INK2, GRID = "#1a1a19", "#5a5a58", "#e6e6e3"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor="#fcfcfb")
    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.99, top=0.84, bottom=0.15)
    palette = [BLUE, ORANGE, AQUA, YEL, MAG, GREEN, VIO, RED, "#7a7a78"]

    ax = axes[0]
    for (reg, d), col in zip(R["by_region"].items(), palette):
        xs = [c[0] for c in d["curve"]]; ys = [c[1] * 100 for c in d["curve"]]
        ax.plot(xs, ys, color=col, lw=1.8, marker="o", ms=3.5, mfc="#fcfcfb", label=f"{reg} (n={d['n']:,})")
    ax.axhline(50, color=GRID, lw=1); ax.axvline(4000, color=INK, lw=1.2); ax.text(4080, 92, "4,000 u", fontsize=8)
    ax.set_xlim(0, 9000); ax.set_ylim(0, 100)
    ax.set_xlabel(f"distance between consecutive kills within {R['gap_s']:.0f} s, units"); ax.set_ylabel("% sharing a champion")
    ax.set_title("a. Same-champion rate vs distance, by region of first kill", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    ax = axes[1]
    regs = list(R["by_region"].keys())
    y = np.arange(len(regs))
    for i, reg in enumerate(regs):
        d = R["by_region"][reg]; b = d["bootstrap"] or {}
        if d["crossover_u"] is not None:
            if b:
                ax.plot([b["p2.5"], b["p97.5"]], [i, i], color=GRID, lw=4)
            ax.plot(d["crossover_u"], i, "o", color=ORANGE, ms=7, mec="#fcfcfb")
            ax.text(max(d["crossover_u"], b.get("p97.5", 0)) + 150, i, f"{d['crossover_u']:,.0f} u", va="center", fontsize=8)
        else:
            ax.text(200, i, "no crossover (too few far pairs)", va="center", fontsize=8, color=INK2)
    ax.axvline(4000, color=INK, lw=1.2)
    if R["all"]["crossover_u"] is not None:
        ax.axvline(R["all"]["crossover_u"], color=ORANGE, lw=1.2, ls="-.")
    ax.set_yticks(y); ax.set_yticklabels(regs); ax.invert_yaxis(); ax.set_xlim(0, 9000)
    ax.set_xlabel("distance where sharing drops to 50%, with 95% match-bootstrap band"); ax.set_title("b. Spatial boundary by region", loc="left", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    ax = axes[2]
    lanes = list(R["lane_anisotropy"].keys())
    w = 0.36
    for i, lane in enumerate(lanes):
        d = R["lane_anisotropy"][lane]
        for xx, comp, col, lab in ((i - w / 2, d["along"], BLUE, "along lane"), (i + w / 2, d["across"], ORANGE, "across lane")):
            v = comp["crossover_u"]
            if v is not None:
                ax.bar(xx, v, w, color=col, label=lab if i == 0 else None)
                ax.text(xx, v + 80, f"{v:,.0f}", ha="center", fontsize=8)
            else:
                # sharing never dropped to 50% within the populated bins: draw the
                # last populated bin as an open bar and report what was still shared
                last_x, last_y = comp["curve"][-1]
                ax.bar(xx, last_x, w, color="none", edgecolor=col, lw=1.6, hatch="////", label=lab if i == 0 else None)
                ax.text(xx, last_x + 80, f"> {last_x:,.0f}" + chr(10) + f"({last_y * 100:.0f}% still shared)", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(lanes))); ax.set_xticklabels(lanes); ax.set_ylabel("50% crossover distance, units"); ax.set_ylim(0, 6300)
    ax.set_title("c. Lane anisotropy: along (across<=500 u) vs across (along<=1,000 u)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    fig.suptitle(f"Spatial boundary by map region: {R['n_pairs_in_window']:,} consecutive kill pairs within {R['gap_s']:.0f} s, {R['n_matches']:,} matches",
                 x=0.05, ha="left", fontsize=11)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    raise SystemExit(main())
