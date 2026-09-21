#!/usr/bin/env python3
"""Result charts for new-V q primary + 16.x transfer."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
QOUT = REPO / "outputs" / "q_newv_fit85_20260920"
OUT = REPO / "docs" / "figures"

ORDER = ["constant", "b_p", "PT", "lgbm_state", "logit_state"]
LABELS = {
    "constant": "constant",
    "b_p": "b(p)",
    "PT": "PT",
    "lgbm_state": "LGBM",
    "logit_state": "logit (q*)",
}
COLORS = {
    "constant": "#9e9e9e",
    "b_p": "#90caf9",
    "PT": "#42a5f5",
    "lgbm_state": "#66bb6a",
    "logit_state": "#ef6c00",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    primary = json.loads((QOUT / "primary_table.json").read_text(encoding="utf-8"))
    transfer = json.loads((QOUT / "transfer_16x" / "results.json").read_text(encoding="utf-8"))
    winner = primary["selected_q"]

    # ── Figure 1: TEST primary dashboard ──
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)

    ax = axes[0]
    xs = np.arange(len(ORDER))
    briers = [primary["TEST"][m]["brier"] for m in ORDER]
    cols = [COLORS[m] for m in ORDER]
    ax.bar(xs, briers, color=cols, width=0.72)
    ax.axhline(primary["TEST"]["PT"]["brier"], color="#1565c0", ls="--", lw=1.2, alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[m] for m in ORDER], rotation=20, ha="right")
    ax.set_ylabel("Brier (↓ better)")
    ax.set_title("TEST 15.16 T — Brier")
    ax.set_ylim(0.22, 0.255)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    aucs = [primary["TEST"][m]["auc"] for m in ORDER]
    ax.bar(xs, aucs, color=cols, width=0.72)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[m] for m in ORDER], rotation=20, ha="right")
    ax.set_ylabel("AUC (↑ better)")
    ax.set_title("TEST 15.16 T — AUC")
    ax.set_ylim(0.48, 0.68)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    boot = primary["bootstrap_delta_brier"]
    key = f"{winner}_minus_PT"
    d = boot[key]
    # also B40 if present
    items = [("All T", d)]
    b40k = f"B40_{winner}_minus_PT"
    if b40k in boot:
        items.append(("B40", boot[b40k]))
    ypos = np.arange(len(items))
    est = [it[1]["estimate"] for it in items]
    lo = [it[1]["ci95"][0] for it in items]
    hi = [it[1]["ci95"][1] for it in items]
    ax.axvline(0, color="#333", lw=1.0)
    ax.axvline(-0.001, color="#999", ls=":", lw=1.0, label="τ=−0.001")
    ax.errorbar(
        est,
        ypos,
        xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
        fmt="o",
        color="#ef6c00",
        capsize=4,
        ms=7,
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels([it[0] for it in items])
    ax.set_xlabel(r"$\Delta$Brier = Brier(q*) − Brier(PT)")
    ax.set_title("Primary contrast (neg ⇒ q better)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"q primary — selected {winner}  |  TRAIN OOF + frozen fit85 V  |  n_TEST={primary['census']['test']:,}",
        fontsize=11,
    )
    p1 = OUT / "q_newv_fit85_primary_dashboard_20260920.png"
    fig.savefig(p1, dpi=170, facecolor="white")
    plt.close()
    print("wrote", p1)

    # ── Figure 2: All T vs B40 Brier comparison ──
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    w = 0.36
    b_all = [primary["TEST"][m]["brier"] for m in ORDER]
    b_b40 = [primary["TEST_B40"][m]["brier"] for m in ORDER]
    ax.bar(xs - w / 2, b_all, width=w, color="#42a5f5", label="All T")
    ax.bar(xs + w / 2, b_b40, width=w, color="#ef6c00", label="B40")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[m] for m in ORDER])
    ax.set_ylabel("Brier")
    ax.set_title("TEST: All T vs B40 (new p_pre)")
    ax.set_ylim(0.22, 0.26)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    p2 = OUT / "q_newv_fit85_test_b40_brier_20260920.png"
    fig.savefig(p2, dpi=170, facecolor="white")
    plt.close()
    print("wrote", p2)

    # ── Figure 3: Transfer ΔBrier ──
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    cohorts = ["KR_16.13", "NA1_16.13", "KR_16.15", "KR_16.14_pilot"]
    labels = [transfer["cohorts"][c]["label"] for c in cohorts]
    delta = [transfer["cohorts"][c]["delta_brier_q_minus_PT"] for c in cohorts]
    q_br = [transfer["cohorts"][c]["scores"][winner]["brier"] for c in cohorts]
    pt_br = [transfer["cohorts"][c]["scores"]["PT"]["brier"] for c in cohorts]
    ns = [transfer["cohorts"][c]["n"] for c in cohorts]

    ax = axes[0]
    cols_d = ["#c62828" if d > 0 else "#2e7d32" for d in delta]
    ax.barh(np.arange(len(cohorts)), delta, color=cols_d, height=0.55)
    ax.axvline(0, color="#222", lw=1.0)
    ax.axvline(-0.001, color="#999", ls=":", lw=1.0)
    ax.axvline(0.001, color="#999", ls=":", lw=1.0)
    ax.set_yticks(np.arange(len(cohorts)))
    ax.set_yticklabels([f"{lb}\n(n={n})" for lb, n in zip(labels, ns)])
    ax.set_xlabel(r"$\Delta$Brier (q* − PT)")
    ax.set_title("16.x transfer — ΔBrier (score-only)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    ax = axes[1]
    x = np.arange(len(cohorts))
    ax.bar(x - 0.2, q_br, 0.4, color="#ef6c00", label="q* (logit)")
    ax.bar(x + 0.2, pt_br, 0.4, color="#42a5f5", label="PT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Brier")
    ax.set_title("16.x transfer — Brier levels")
    ax.set_ylim(0.23, 0.26)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Frozen q on EXT 16.x (no refit)  |  green Δ < 0 = q better than PT", fontsize=11)
    p3 = OUT / "q_newv_fit85_transfer_16x_dashboard_20260920.png"
    fig.savefig(p3, dpi=170, facecolor="white")
    plt.close()
    print("wrote", p3)

    # ── Figure 4: SELECT vs TEST consistency ──
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for m in ORDER:
        ax.scatter(
            primary["Q_SELECT"][m]["brier"],
            primary["TEST"][m]["brier"],
            s=120 if m == winner else 80,
            color=COLORS[m],
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
            label=LABELS[m],
        )
        ax.annotate(LABELS[m], (primary["Q_SELECT"][m]["brier"], primary["TEST"][m]["brier"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    lims = [0.230, 0.252]
    ax.plot(lims, lims, color="#888", ls="--", lw=1.0)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Q_SELECT Brier")
    ax.set_ylabel("TEST Brier")
    ax.set_title("Selection → TEST (no TEST peek for choice)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    p4 = OUT / "q_newv_fit85_select_vs_test_20260920.png"
    fig.savefig(p4, dpi=170, facecolor="white")
    plt.close()
    print("wrote", p4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
