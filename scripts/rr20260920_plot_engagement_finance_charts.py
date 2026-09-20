#!/usr/bin/env python3
"""Stock/finance-style charts for engagement winprob moves (new V).

Metaphor: frozen V is the 'price' of blue winprob; each fight is a bar/candle
from p_pre (open) to p_post (close). Direction = SVI; size = |ΔV|.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
WAVE4 = REPO / "outputs" / "v_redesign_wave4_corrected_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"
TABLE = REPO / "outputs" / "newv_engagement_value_verify_20260920" / "engagement_table_h90_MAIN_TEST.npz"
OUT = REPO / "docs" / "figures"

# finance palette (blue-team ΔV+: teal up / crimson down — not Riot team colors)
UP = "#0b6e4f"
DOWN = "#9b2226"
NEUTRAL = "#4a4e69"
GRID = "#e9ecef"
BG = "#fafafa"


def _setup():
    import os
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            data_root = p
            break
    else:
        data_root = Path.home() / "문서" / "LOL_Teamfight"
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path[:0] = [str(REPO), str(data_root / "scripts"), str(wt)]
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )
    return data_root


def draw_candle(ax, x, o, c, width=0.55, alpha=0.95):
    col = UP if c >= o else DOWN
    lo, hi = min(o, c), max(o, c)
    # wick
    ax.plot([x, x], [lo, hi], color=col, lw=1.2, solid_capstyle="round", zorder=2)
    # if flat, still mark
    h = max(hi - lo, 0.008)
    y0 = lo if c >= o else c
    if abs(c - o) < 1e-6:
        ax.plot([x - width / 2, x + width / 2], [o, o], color=col, lw=2.0, zorder=3)
    else:
        ax.add_patch(
            Rectangle(
                (x - width / 2, min(o, c)),
                width,
                abs(c - o) if abs(c - o) > 1e-6 else h,
                facecolor=col,
                edgecolor=col,
                alpha=alpha,
                zorder=3,
                linewidth=0.6,
            )
        )


def fig_market_overview(p0, p1, dV, tmin):
    """Aggregate 'returns' board: candlestick strip + return hist + vol by time."""
    fig = plt.figure(figsize=(13.2, 8.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.28)

    # --- A: sample candlesticks ordered by time (like a session tape) ---
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(BG)
    rng = np.random.default_rng(11)
    # stratified sample across time for a readable tape
    order = np.argsort(tmin)
    # take every k-th from sorted for a continuous-looking session
    step = max(1, len(order) // 180)
    idx = order[::step][:180]
    xs = np.arange(len(idx))
    for j, i in enumerate(idx):
        draw_candle(ax, xs[j], float(p0[i]), float(p1[i]), width=0.7)
    ax.axhline(0.5, color=NEUTRAL, lw=1.0, ls="--", alpha=0.7, label="even (0.5)")
    ax.set_xlim(-1, len(idx))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Blue winprob  V")
    ax.set_xlabel("Engagements in match-time order (subsample tape)")
    ax.set_title(
        "Candlestick tape — each fight is open=p_pre → close=p_post  "
        f"(teal=up / crimson=down, n_show={len(idx)})"
    )
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # --- B: return distribution (like daily returns) ---
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(BG)
    bins = np.linspace(-0.55, 0.55, 55)
    ax.hist(dV[dV > 0], bins=bins, color=UP, alpha=0.88, label=f"up  n={(dV>0).sum():,}")
    ax.hist(dV[dV <= 0], bins=bins, color=DOWN, alpha=0.88, label=f"down n={(dV<=0).sum():,}")
    ax.axvline(0, color="#222222", lw=1.2)
    # annotate fat-tail fractions like finance
    for thr, y in ((0.05, 0.72), (0.10, 0.58), (0.20, 0.44)):
        frac = float(np.mean(np.abs(dV) >= thr))
        ax.text(
            0.98,
            y,
            f"P(|r|≥{int(thr*100)}pp) = {frac:.2f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color="#333333",
            family="monospace",
        )
    ax.set_xlabel(r"fight return  r = ΔV = p_post − p_pre")
    ax.set_ylabel("count")
    ax.set_title("Return distribution (signed ΔV)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color=GRID)

    # --- C: realized vol by clock (like intraday vol) ---
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(BG)
    edges = np.arange(2, 41, 2.0)
    mids, vol, n_bin = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (tmin >= lo) & (tmin < hi)
        if m.sum() < 120:
            continue
        mids.append(0.5 * (lo + hi))
        # realized vol proxy: std of returns + mean abs
        vol.append(float(np.std(dV[m])))
        n_bin.append(int(m.sum()))
    ax.fill_between(mids, vol, color=NEUTRAL, alpha=0.25)
    ax.plot(mids, vol, color=NEUTRAL, lw=2.2, marker="o", ms=4)
    ax.set_xlabel("match time (min)")
    ax.set_ylabel(r"realized vol  std(ΔV)")
    ax.set_title("Intraday-style volatility of fight returns")
    ax.grid(True, color=GRID)
    ax.set_xlim(0, 42)

    fig.suptitle(
        "Economic-style view of engagement winprob moves  |  frozen fit85 MLP V   "
        f"(MAIN_TEST T∩h90, n={len(dV):,})",
        fontsize=12,
        y=0.995,
    )
    path = OUT / "mlp_fit85_engagement_finance_market_20260920.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close()
    print("wrote", path)
    return path


def fig_match_tickers(data_root, eng_match, eng_s, p0, p1, dV):
    """Per-match price charts with fight candles overlaid on V timeline."""
    import fc20260915_data as D
    from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated

    L = D.Layout(False)
    TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    ev = load_evaluator(BUNDLE)
    qm = TE["match"].astype(str)
    qt = TE["query_ms"].astype(float) / 60000.0
    qp = predict_calibrated(ev, TE["X"])

    # choose matches with several fights and large moves
    cnt = Counter(eng_match.tolist())
    scores = []
    for mid, n_e in cnt.items():
        if n_e < 3:
            continue
        m = eng_match == mid
        if m.sum() < 3:
            continue
        scores.append((float(np.mean(np.abs(dV[m]))), int(n_e), mid))
    scores.sort(reverse=True)
    picks = [s[2] for s in scores[:6]]

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.0), sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, mid in zip(axes, picks):
        ax.set_facecolor(BG)
        mq = qm == mid
        t_line = qt[mq]
        p_line = qp[mq]
        order = np.argsort(t_line)
        t_line, p_line = t_line[order], p_line[order]
        ax.plot(t_line, p_line, color="#1d3557", lw=1.6, marker="o", ms=3.5, label="V timeline", zorder=2)
        ax.axhline(0.5, color=NEUTRAL, lw=0.9, ls=":", alpha=0.8)

        me = eng_match == mid
        # place candles at onset time; width ~ duration visual
        for t0, o, c in zip(eng_s[me] / 60000.0, p0[me], p1[me]):
            col = UP if c >= o else DOWN
            ax.annotate(
                "",
                xy=(t0 + 0.4, c),
                xytext=(t0, o),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=10),
                zorder=4,
            )
            ax.scatter([t0], [o], s=28, color=col, zorder=5, edgecolors="white", linewidths=0.4)
            ax.scatter([t0 + 0.4], [c], s=28, color=col, zorder=5, marker="s", edgecolors="white", linewidths=0.4)

        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(max(0, float(t_line.min()) - 1), float(t_line.max()) + 2)
        ax.grid(True, color=GRID, lw=0.7)
        short = mid if len(mid) < 28 else mid[:12] + "…" + mid[-10:]
        ax.set_title(f"match {short}   fights={int(me.sum())}", fontsize=9)
        ax.set_xlabel("match time (min)", fontsize=8)
        ax.set_ylabel("V", fontsize=8)

    legend = [
        Line2D([0], [0], color="#1d3557", lw=1.6, marker="o", ms=4, label="bucket V path"),
        Line2D([0], [0], color=UP, lw=2, label="fight up (SVI+)"),
        Line2D([0], [0], color=DOWN, lw=2, label="fight down (SVI−)"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Match tickers — winprob path with fight open→close moves (finance chart metaphor)",
        fontsize=12,
        y=1.04,
    )
    path = OUT / "mlp_fit85_engagement_finance_tickers_20260920.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close()
    print("wrote", path)
    return path


def fig_ohlc_board(p0, p1, dV, tmin, b40):
    """Classic OHLC-style summary: open/close bars + volume(|ΔV|) by p_pre regime."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.set_facecolor(BG)
    # bin by p_pre like price levels; show mean open/close as a thick candle per bin
    edges = np.linspace(0, 1, 11)
    xs = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p0 >= lo) & (p0 < hi)
        if m.sum() < 80:
            continue
        x = i
        xs.append(x)
        o = float(np.mean(p0[m]))
        # mean close among ups and downs separately as range whisker
        c = float(np.mean(p1[m]))
        draw_candle(ax, x, o, c, width=0.65)
        ax.text(x, -0.06, f"{int(lo*100)}-{int(hi*100)}", ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color=NEUTRAL, ls="--", lw=1.0)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.8, 10.5)
    ax.set_xticks([])
    ax.set_ylabel("mean V")
    ax.set_title("Level board — mean open→close by pre-fight V bin")
    ax.grid(True, axis="y", color=GRID)

    ax = axes[1]
    ax.set_facecolor(BG)
    # waterfalls of mean signed return vs mean abs by B40 vs rest — bar chart like sector returns
    groups = [
        ("all", np.ones(len(dV), dtype=bool)),
        ("B40", b40.astype(bool)),
        ("outside B40", ~b40.astype(bool)),
        ("early <15m", tmin < 15),
        ("mid 15–25m", (tmin >= 15) & (tmin < 25)),
        ("late ≥25m", tmin >= 25),
    ]
    labels, means, abs_m = [], [], []
    for name, m in groups:
        if m.sum() < 50:
            continue
        labels.append(name)
        means.append(float(np.mean(dV[m])))
        abs_m.append(float(np.mean(np.abs(dV[m]))))
    ypos = np.arange(len(labels))
    colors = [UP if v >= 0 else DOWN for v in means]
    ax.barh(ypos, means, color=colors, height=0.55, alpha=0.9, label="mean return E[ΔV]")
    ax.plot(abs_m, ypos, "D", color="#1d3557", ms=6, label=r"mean |ΔV|")
    ax.axvline(0, color="#222", lw=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("ΔV")
    ax.set_title("Sector-style returns by regime")
    ax.grid(True, axis="x", color=GRID)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Finance board summaries of fight returns under frozen V", fontsize=11)
    path = OUT / "mlp_fit85_engagement_finance_board_20260920.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close()
    print("wrote", path)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data_root = _setup()
    z = np.load(TABLE, allow_pickle=False)
    ok = z["missing_score"] == 0
    p0 = z["p_pre"][ok].astype(float)
    p1 = z["p_post"][ok].astype(float)
    dV = z["delta_V"][ok].astype(float)
    tmin = z["s"][ok].astype(float) / 60000.0
    b40 = z["B40"][ok].astype(bool)
    em = z["match"][ok].astype(str)
    es = z["s"][ok].astype(np.int64)

    fig_market_overview(p0, p1, dV, tmin)
    fig_ohlc_board(p0, p1, dV, tmin, b40)
    fig_match_tickers(data_root, em, es, p0, p1, dV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
