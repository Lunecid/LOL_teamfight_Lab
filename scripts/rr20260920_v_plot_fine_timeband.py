#!/usr/bin/env python3
"""Fine-grained (per-minute) WP metrics vs match time from raw query rows.

Not the 10-minute L_time bands — bins align to actual query_ms / frame grid.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
WAVE4 = REPO / "outputs" / "v_redesign_wave4_corrected_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    w = (1.0 / c[inv]).astype(np.float64)
    return w / w.mean()


def bin_metrics(y, p, g, tmin, bin_width=1.0, t_max=45.0, min_n=200):
    """Per bin_width-minute metrics on raw query times (floor to bin)."""
    w = match_weights(g)
    edges = np.arange(0.0, t_max + bin_width, bin_width)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (tmin >= lo) & (tmin < hi) & np.isfinite(p)
        n = int(m.sum())
        if n < min_n:
            continue
        yy, pp, ww, gg = y[m], p[m], w[m], g[m]
        br = float(np.average((pp - yy) ** 2, weights=ww))
        eps = 1e-15
        pc = np.clip(pp, eps, 1 - eps)
        ll = float(np.average(-(yy * np.log(pc) + (1 - yy) * np.log(1 - pc)), weights=ww))
        try:
            from sklearn.metrics import roc_auc_score

            # need both classes
            if len(np.unique(yy)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(yy, pp, sample_weight=ww))
        except Exception:
            auc = float("nan")
        rows.append(
            dict(
                t_lo=float(lo),
                t_mid=float(lo + bin_width / 2),
                t_hi=float(hi),
                n=n,
                n_matches=int(len(np.unique(gg))),
                brier=br,
                logloss=ll,
                auc=auc,
                mean_p=float(np.average(pp, weights=ww)),
                mean_y=float(np.average(yy, weights=ww)),
            )
        )
    return rows


def main() -> int:
    import argparse

    import matplotlib.pyplot as plt

    from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated

    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-width", type=float, default=1.0, help="minutes per bin (default 1)")
    ap.add_argument("--t-max", type=float, default=40.0)
    ap.add_argument("--split", choices=("TEST", "V_SELECT"), default="TEST")
    ap.add_argument("--min-n", type=int, default=300)
    args = ap.parse_args()

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D

    if not BUNDLE.is_file():
        raise SystemExit(f"missing {BUNDLE}")

    L = D.Layout(False)
    role = ["TEST"] if args.split == "TEST" else ["V_SELECT"]
    print(f"load {args.split} bucket rows…", flush=True)
    R = D.load_v_rows(L, "MAIN", role, bucket_only=True)
    W = D.load_outcomes(L, "MAIN", role, purpose="fine timeband plot")
    y = np.asarray([W[m][0] for m in R["match"].tolist()], dtype=np.float64)
    t = R["query_ms"].astype(float) / 60000.0
    g = R["match"].astype(str)

    print("score MLP evaluator…", flush=True)
    ev = load_evaluator(BUNDLE)
    p = predict_calibrated(ev, R["X"])

    # also raw query-level scatter density for calibration-by-time (mean p vs mean y)
    rows = bin_metrics(y, p, g, t, bin_width=args.bin_width, t_max=args.t_max, min_n=args.min_n)
    out_json = WAVE4 / f"fine_timeband_{args.split.lower()}_bin{args.bin_width:g}m.json"
    out_json.write_text(
        json.dumps(
            dict(
                split=args.split,
                bin_width_min=args.bin_width,
                t_max=args.t_max,
                evaluator="A_MLP_expanded_evaluator.joblib",
                n_queries=int(len(y)),
                rows=rows,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("wrote", out_json, "nbins", len(rows), flush=True)

    tm = np.array([r["t_mid"] for r in rows])
    br = np.array([r["brier"] for r in rows])
    auc = np.array([r["auc"] for r in rows])
    ll = np.array([r["logloss"] for r in rows])
    n = np.array([r["n"] for r in rows])
    mean_p = np.array([r["mean_p"] for r in rows])
    mean_y = np.array([r["mean_y"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(tm, br, color="#1f77b4", linewidth=1.8)
    ax.set_title(f"Brier vs match time ({args.bin_width:g}-min bins)")
    ax.set_xlabel("Match time (min)")
    ax.set_ylabel("Brier (↓ better)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(tm, auc, color="#2ca02c", linewidth=1.8)
    ax.set_title(f"AUC vs match time ({args.bin_width:g}-min bins)")
    ax.set_xlabel("Match time (min)")
    ax.set_ylabel("AUC (↑ better)")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(tm, ll, color="#d62728", linewidth=1.8)
    ax.set_title(f"Log loss vs match time ({args.bin_width:g}-min bins)")
    ax.set_xlabel("Match time (min)")
    ax.set_ylabel("Log loss (↓ better)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(tm, mean_y, label="mean W (label)", color="#333333", linewidth=1.6)
    ax.plot(tm, mean_p, label="mean V̂", color="#1f77b4", linewidth=1.6)
    ax.set_title("Mean label vs mean prediction")
    ax.set_xlabel("Match time (min)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.35, 0.65)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"A_MLP_expanded fit85 — {args.split} raw queries, {args.bin_width:g}-min bins "
        f"(n={len(y):,})",
        fontsize=11,
    )
    out_dir = REPO / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mlp_fit85_{args.split.lower()}_per_minute_performance_20260920.png"
    fig.savefig(path, dpi=160)
    plt.close()
    print("wrote", path)

    # query count per bin (shows frame/query density)
    fig2, ax2 = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax2.bar(tm, n, width=args.bin_width * 0.85, color="#7f7f7f", alpha=0.85)
    ax2.set_xlabel("Match time (min)")
    ax2.set_ylabel("Query count")
    ax2.set_title(f"{args.split} query density by {args.bin_width:g}-min bin (bucket samples)")
    ax2.grid(True, axis="y", alpha=0.3)
    path2 = out_dir / f"mlp_fit85_{args.split.lower()}_per_minute_density_20260920.png"
    fig2.savefig(path2, dpi=160)
    plt.close()
    print("wrote", path2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
