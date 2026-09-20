#!/usr/bin/env python3
"""Visualize pre->post winprob direction change around engagements (new V).

Shows the volatility that SVI is meant to capture: not overall WP accuracy alone,
but how V-hat moves across the fight window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "outputs" / "newv_engagement_value_verify_20260920" / "engagement_table_h90_MAIN_TEST.npz"
OUT = REPO / "docs" / "figures"


def main() -> int:
    z = np.load(TABLE, allow_pickle=False)
    ok = z["missing_score"] == 0
    p0 = z["p_pre"][ok].astype(float)
    p1 = z["p_post"][ok].astype(float)
    dV = z["delta_V"][ok].astype(float)
    tmin = z["s"][ok].astype(float) / 60000.0
    b40 = z["B40"][ok].astype(bool)
    n = len(dV)

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # --- Figure 1: the core story — pre→post movement ---
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)

    ax = axes[0]
    # density scatter subsample for readability
    idx = rng.choice(n, size=min(12000, n), replace=False)
    sc = ax.scatter(
        p0[idx],
        p1[idx],
        c=dV[idx],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.35, vmax=0.35),
        s=6,
        alpha=0.35,
        linewidths=0,
    )
    ax.plot([0, 1], [0, 1], color="#333333", lw=1.2, ls="--", label="no change")
    ax.axhline(0.5, color="#888888", lw=0.8, alpha=0.6)
    ax.axvline(0.5, color="#888888", lw=0.8, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$p_{pre}$  (before fight)")
    ax.set_ylabel(r"$p_{post}$  (after fight)")
    ax.set_title("Fight window: where winprob moves")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Delta V = p_{post}-p_{pre}$")

    ax = axes[1]
    # signed histogram
    bins = np.linspace(-0.6, 0.6, 61)
    ax.hist(dV[dV > 0], bins=bins, color="#2166ac", alpha=0.85, label=f"SVI=1  n={(dV>0).sum():,}")
    ax.hist(dV[dV < 0], bins=bins, color="#b2182b", alpha=0.85, label=f"SVI=0  n={(dV<=0).sum():,}")
    ax.axvline(0, color="#222222", lw=1.2)
    ax.set_xlabel(r"$\Delta V = p_{post}-p_{pre}$")
    ax.set_ylabel("Engagements")
    ax.set_title("Signed change size (direction volatility)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    # mean |ΔV| and P(SVI=1) by p_pre
    edges = np.linspace(0, 1, 11)
    mids = 0.5 * (edges[:-1] + edges[1:])
    pos_rate, mean_abs, mean_d = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p0 >= lo) & (p0 < hi)
        if m.sum() < 80:
            pos_rate.append(np.nan)
            mean_abs.append(np.nan)
            mean_d.append(np.nan)
        else:
            pos_rate.append(float(np.mean(dV[m] > 0)))
            mean_abs.append(float(np.mean(np.abs(dV[m]))))
            mean_d.append(float(np.mean(dV[m])))
    ax2 = ax.twinx()
    ax.plot(mids, mean_abs, color="#1f77b4", lw=2.0, marker="o", ms=4, label=r"mean $|\Delta V|$")
    ax2.plot(mids, pos_rate, color="#2ca02c", lw=2.0, marker="s", ms=4, label=r"$P(SVI=1)$")
    ax2.plot(mids, mean_d, color="#ff7f0e", lw=1.4, ls="--", marker="^", ms=3.5, label=r"$E[\Delta V]$")
    ax.set_xlabel(r"$p_{pre}$ bin mid")
    ax.set_ylabel(r"mean $|\Delta V|$")
    ax2.set_ylabel("rate / signed mean")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.25, np.nanmax(mean_abs) * 1.15))
    ax2.set_ylim(-0.15, 0.85)
    ax.set_title(r"How change depends on pre-fight $V$")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper center")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"New V (fit85 MLP) — engagement pre→post direction change  (MAIN_TEST T∩h90, n={n:,})",
        fontsize=11,
    )
    path1 = OUT / "mlp_fit85_engagement_delta_volatility_20260920.png"
    fig.savefig(path1, dpi=170)
    plt.close()
    print("wrote", path1)

    # --- Figure 2: arrow sample — explicit direction ---
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)

    ax = axes[0]
    # sample near-even fights + random for context
    near = np.where(np.abs(p0 - 0.5) <= 0.15)[0]
    far = np.where(np.abs(p0 - 0.5) > 0.15)[0]
    take_near = rng.choice(near, size=min(400, len(near)), replace=False) if len(near) else np.array([], int)
    take_far = rng.choice(far, size=min(200, len(far)), replace=False) if len(far) else np.array([], int)
    take = np.concatenate([take_near, take_far])
    for i in take:
        col = "#2166ac" if dV[i] > 0 else "#b2182b"
        ax.annotate(
            "",
            xy=(1.0, p1[i]),
            xytext=(0.0, p0[i]),
            arrowprops=dict(arrowstyle="->", color=col, alpha=0.25, lw=0.9),
        )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"pre", r"post"])
    ax.set_ylabel(r"$V$ (blue winprob)")
    ax.axhline(0.5, color="#888888", lw=1.0, ls=":")
    ax.set_title("Sample trajectories (blue↑ / red↓)")
    ax.text(0.02, 0.97, "blue = SVI+", transform=ax.transAxes, color="#2166ac", fontsize=8, va="top")
    ax.text(0.02, 0.92, "red = SVI-", transform=ax.transAxes, color="#b2182b", fontsize=8, va="top")

    ax = axes[1]
    # |ΔV| vs time, with B40 highlighted
    ax.scatter(tmin[~b40], np.abs(dV[~b40]), s=4, alpha=0.15, color="#7f7f7f", label="other", rasterized=True)
    ax.scatter(tmin[b40], np.abs(dV[b40]), s=6, alpha=0.35, color="#e6550d", label="B40 (new p_pre)", rasterized=True)
    # smooth mean |ΔV| by minute
    edges_t = np.arange(2, 41, 1.0)
    mids_t, means_t = [], []
    for lo, hi in zip(edges_t[:-1], edges_t[1:]):
        m = (tmin >= lo) & (tmin < hi)
        if m.sum() < 80:
            continue
        mids_t.append(0.5 * (lo + hi))
        means_t.append(float(np.mean(np.abs(dV[m]))))
    ax.plot(mids_t, means_t, color="#000000", lw=2.0, label=r"mean $|\Delta V|$ / min")
    ax.set_xlabel("Match time at onset (min)")
    ax.set_ylabel(r"$|\Delta V|$")
    ax.set_title("Change magnitude over match time")
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Directionality of strategic-value change around engagements", fontsize=11)
    path2 = OUT / "mlp_fit85_engagement_delta_trajectories_20260920.png"
    fig.savefig(path2, dpi=170)
    plt.close()
    print("wrote", path2)

    # --- Figure 3: compact 2-panel for paper-ish ---
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    ax = axes[0]
    # hexbin of movement
    hb = ax.hexbin(p0, p1, gridsize=45, cmap="viridis", mincnt=5, bins="log")
    ax.plot([0, 1], [0, 1], color="white", lw=1.4, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$p_{pre}$")
    ax.set_ylabel(r"$p_{post}$")
    ax.set_title("Density of pre→post states")
    ax.set_aspect("equal")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="log count")

    ax = axes[1]
    # CDF of |ΔV|
    abs_d = np.sort(np.abs(dV))
    cdf = np.arange(1, len(abs_d) + 1) / len(abs_d)
    ax.plot(abs_d, cdf, color="#1f77b4", lw=2.0)
    for thr, lab in ((0.02, "2pp"), (0.05, "5pp"), (0.10, "10pp")):
        ax.axvline(thr, color="#999999", lw=0.9, ls=":")
        frac = float(np.mean(np.abs(dV) > thr))
        ax.text(thr + 0.005, 0.08, f"{lab}\nP(>|·|)={frac:.2f}", fontsize=7.5, color="#444444")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$|\Delta V|$")
    ax.set_ylabel("CDF")
    ax.set_title("How often the fight moves winprob")
    ax.grid(True, alpha=0.3)

    path3 = OUT / "mlp_fit85_engagement_delta_density_cdf_20260920.png"
    fig.savefig(path3, dpi=170)
    plt.close()
    print("wrote", path3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
