#!/usr/bin/env python3
"""RR4 — binary SVI retention + small-|ΔV| sensitivity (λ·s_Q and absolute cutoffs).

Design: docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §8
Lit: Christoffersen–Diebold triad; Brown–Warner scale vs mean; RR3 s_Q fallback.

Locks:
  - Primary target remains Y=1[ΔV>0]; no continuous-target swap
  - Fixed RR12 preds (q_base identity, PT_flex); no retuning
  - λ∈{0,.25,.5,1}: keep |ΔV| ≥ λ·max(s_Q(z),1e-6); λ=0 = all eligible
  - Do NOT pick best TEST λ as primary result
  - Absolute cutoffs {0,.005,.01,.02} are a *different* auxiliary definition
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from forecast_diagnostics import (  # noqa: E402
    match_weights,
    triad_by_p_bins,
    weighted_mean,
)

RR12 = REPO / "outputs" / "review_response_rr12_20260920"
RR3 = REPO / "outputs" / "review_response_rr3_quiet_20260920"
OUT = REPO / "outputs" / "review_response_rr4_margin_20260920"
ROLE = "EXPLORATORY_RR4_MARGIN_SENSITIVITY_PRIOR_TEST_EXPOSURE"

LAMBDAS = (0.0, 0.25, 0.5, 1.0)
ABS_CUTS = (0.0, 0.005, 0.01, 0.02)
EPS = 1e-6


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{float(x):.{nd}f}"


def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def bootstrap_delta_brier(y, p_a, p_b, g, reps=2000, seed=7):
    """ΔBrier = Brier(a) - Brier(b); negative ⇒ a better."""
    g = np.asarray(g).astype(str)
    y = np.asarray(y).astype(float)
    p_a = np.asarray(p_a).astype(float)
    p_b = np.asarray(p_b).astype(float)
    if len(y) == 0:
        return dict(estimate=float("nan"), ci95=[float("nan"), float("nan")], bootstrap_fraction_positive=float("nan"))
    matches, inv = np.unique(g, return_inverse=True)
    n_m = len(matches)
    _, counts = np.unique(inv, return_counts=True)
    w = (1.0 / counts[inv]).astype(np.float64)
    ea = w * (p_a - y) ** 2
    eb = w * (p_b - y) ** 2
    sum_a = np.zeros(n_m)
    sum_b = np.zeros(n_m)
    np.add.at(sum_a, inv, ea)
    np.add.at(sum_b, inv, eb)
    obs = float((sum_a.sum() - sum_b.sum()) / n_m)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        samp = rng.integers(0, n_m, size=n_m)
        draws[i] = float((sum_a[samp].sum() - sum_b[samp].sum()) / n_m)
    return dict(
        estimate=obs,
        ci95=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        bootstrap_fraction_positive=float(np.mean(draws > 0)),
    )


def metrics(y, p, g) -> Dict[str, Any]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    g = np.asarray(g)
    if len(y) == 0:
        return dict(n=0, n_matches=0, brier=float("nan"), logloss=float("nan"), auc=float("nan"), p_pos=float("nan"))
    w = match_weights(g)
    br = float(np.average((p - y) ** 2, weights=w))
    eps = 1e-15
    pc = np.clip(p, eps, 1 - eps)
    ll = float(np.average(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)), weights=w))
    if y.min() == y.max():
        auc = float("nan")
    else:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y, p, sample_weight=w))
        except Exception:
            auc = float("nan")
    return dict(
        n=int(len(y)),
        n_matches=int(len(np.unique(g.astype(str)))),
        brier=br,
        logloss=ll,
        auc=auc,
        p_pos=float(np.average(y, weights=w)),
    )


def p_bin_index(p: float, edges: List[float]) -> int:
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            if lo <= p < hi:
                return i
        else:
            if lo <= p <= hi:
                return i
    return len(edges) - 2


def band_index(x: float, bands: List[List[float]]) -> int:
    for i, (lo, hi) in enumerate(bands):
        if lo <= x < hi:
            return i
    raise ValueError(f"value {x} outside bands {bands}")


def build_s_lookup(mu_s: Dict[str, Any]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    return {(int(c["p_bin"]), int(c["t_bin"]), int(c["L_bin"])): c for c in mu_s["cells"]}


def resolve_s_Q(
    p: np.ndarray,
    t_min: np.ndarray,
    L_s: np.ndarray,
    mu_s: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return s_Q, mu_Q, level_code for each row."""
    edges = mu_s["p_edges"]
    t_bands = mu_s["t_bands"]
    L_bands = mu_s["L_bands"]
    lookup = build_s_lookup(mu_s)
    n = len(p)
    s = np.empty(n, float)
    mu = np.empty(n, float)
    level = np.empty(n, object)
    for i in range(n):
        pi = p_bin_index(float(p[i]), edges)
        ti = band_index(float(t_min[i]), t_bands)
        Li = band_index(float(L_s[i]), L_bands)
        c = lookup[(pi, ti, Li)]
        s[i] = float(c["s"])
        mu[i] = float(c["mu"])
        level[i] = c["level"]
    return s, mu, level


def slice_metrics(
    mask: np.ndarray,
    y,
    q,
    pt,
    g,
    p_pre,
    tmin,
    delta,
    b40,
    boot_reps: int,
    seed: int,
) -> Dict[str, Any]:
    m = np.asarray(mask, bool)
    n0 = int(len(y))
    if not m.any():
        return dict(n=0, coverage=0.0, n_matches=0)
    ym, qm, ptm, gm = y[m], q[m], pt[m], g[m]
    sc_q = metrics(ym, qm, gm)
    sc_pt = metrics(ym, ptm, gm)
    d = bootstrap_delta_brier(ym, qm, ptm, gm, reps=boot_reps, seed=seed)
    w = match_weights(gm)
    b40m = b40[m]
    out = dict(
        n=int(m.sum()),
        coverage=float(m.sum() / n0),
        n_matches=sc_q["n_matches"],
        mean_p_pre=weighted_mean(p_pre[m], match_weights(g[m])),
        mean_t_min=weighted_mean(tmin[m], match_weights(g[m])),
        p_pos=sc_q["p_pos"],
        mean_abs_delta=weighted_mean(np.abs(delta[m]), w),
        E_delta=weighted_mean(delta[m], w),
        q=sc_q,
        PT_flex=sc_pt,
        delta_brier_q_minus_PT_flex=d,
        B40_n=int(b40m.sum()),
        B40_coverage_within_slice=float(b40m.mean()) if m.any() else float("nan"),
    )
    # optional B40-restricted Δ within the λ-slice
    if b40m.any() and (~b40m).any():
        out["B40_delta_brier"] = bootstrap_delta_brier(ym[b40m], qm[b40m], ptm[b40m], gm[b40m], reps=boot_reps, seed=seed + 1)
    elif b40m.any():
        out["B40_delta_brier"] = bootstrap_delta_brier(ym[b40m], qm[b40m], ptm[b40m], gm[b40m], reps=boot_reps, seed=seed + 1)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    pred_path = RR12 / "prediction_table.npz"
    s_path = RR3 / "quiet_mu_s_Q_CAL_fallback.json"
    for need in (pred_path, s_path):
        if not need.is_file():
            raise SystemExit(f"missing {need}")

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D

    print("load RR12 preds + engagements for L…", flush=True)
    pred = np.load(pred_path, allow_pickle=False)
    mu_s = json.loads(s_path.read_text(encoding="utf-8"))
    Llayout = D.Layout(False)
    E = D.load_engagements(Llayout, "MAIN", ["TEST"], states=False, counts=False)
    e_key = {
        (a, int(b)): i
        for i, (a, b) in enumerate(zip(E["match"].astype(str).tolist(), E["s"].astype(np.int64).tolist()))
    }
    match = pred["match"].astype(str)
    s = pred["s"].astype(np.int64)
    idx_e = np.asarray([e_key[(a, int(b))] for a, b in zip(match.tolist(), s.tolist())], dtype=int)
    L_ms = E["endpoint_h90"][idx_e].astype(np.int64) - s
    L_sec = L_ms.astype(float) / 1000.0
    tmin = s.astype(float) / 60000.0
    p_pre = pred["p_pre"].astype(float)
    delta = pred["delta_V"].astype(float)
    y = pred["y"].astype(float)
    g = match
    q = pred["p_q_base"].astype(float)
    pt = pred["p_PT_flex"].astype(float)
    b40 = pred["B40"].astype(bool)

    print("resolve s_Q(z) from Q_CAL quiet fallback…", flush=True)
    s_Q, mu_Q, level = resolve_s_Q(p_pre, tmin, L_sec, mu_s)
    thresh = np.maximum(s_Q, EPS)

    # triad (all rows) — Christoffersen–Diebold axes
    triad = triad_by_p_bins(delta, p_pre, g)

    level_counts = {str(k): int(v) for k, v in zip(*np.unique(level, return_counts=True))}
    n_L_gt_120 = int(np.sum(L_sec > 120.0))

    lambda_rows = []
    lambda_payload = {}
    for lam in LAMBDAS:
        if lam == 0.0:
            mask = np.ones(len(y), dtype=bool)
        else:
            mask = np.abs(delta) >= (lam * thresh)
        block = slice_metrics(mask, y, q, pt, g, p_pre, tmin, delta, b40, args.boot_reps, args.seed)
        block["lambda"] = lam
        block["filter"] = "abs_dV_ge_lambda_sQ" if lam > 0 else "all"
        block["n_excluded"] = int((~mask).sum())
        block["level_counts_kept"] = {
            str(k): int(np.sum(level[mask] == k)) for k in np.unique(level)
        }
        lambda_payload[f"lambda_{lam}"] = block
        lambda_rows.append(
            dict(
                filter="sQ",
                lambda_or_cut=lam,
                n=block["n"],
                coverage=block["coverage"],
                n_matches=block["n_matches"],
                n_excluded=block["n_excluded"],
                mean_p_pre=block["mean_p_pre"],
                mean_t_min=block["mean_t_min"],
                p_pos=block["p_pos"],
                mean_abs_delta=block["mean_abs_delta"],
                q_brier=block["q"]["brier"],
                PT_flex_brier=block["PT_flex"]["brier"],
                delta_brier=block["delta_brier_q_minus_PT_flex"]["estimate"],
                delta_brier_lo=block["delta_brier_q_minus_PT_flex"]["ci95"][0],
                delta_brier_hi=block["delta_brier_q_minus_PT_flex"]["ci95"][1],
                q_auc=block["q"]["auc"],
                PT_flex_auc=block["PT_flex"]["auc"],
            )
        )
        print(
            f"  λ={lam}: n={block['n']} cov={fmt(block['coverage'],3)} "
            f"ΔBrier={fmt(block['delta_brier_q_minus_PT_flex']['estimate'])} "
            f"CI=[{fmt(block['delta_brier_q_minus_PT_flex']['ci95'][0])},"
            f"{fmt(block['delta_brier_q_minus_PT_flex']['ci95'][1])}]",
            flush=True,
        )

    abs_payload = {}
    for cut in ABS_CUTS:
        if cut == 0.0:
            mask = np.ones(len(y), dtype=bool)
        else:
            mask = np.abs(delta) >= cut
        block = slice_metrics(mask, y, q, pt, g, p_pre, tmin, delta, b40, args.boot_reps, args.seed + 11)
        block["abs_cut"] = cut
        block["filter"] = "abs_dV_ge_cut" if cut > 0 else "all"
        block["n_excluded"] = int((~mask).sum())
        abs_payload[f"cut_{cut}"] = block
        lambda_rows.append(
            dict(
                filter="abs",
                lambda_or_cut=cut,
                n=block["n"],
                coverage=block["coverage"],
                n_matches=block["n_matches"],
                n_excluded=block["n_excluded"],
                mean_p_pre=block["mean_p_pre"],
                mean_t_min=block["mean_t_min"],
                p_pos=block["p_pos"],
                mean_abs_delta=block["mean_abs_delta"],
                q_brier=block["q"]["brier"],
                PT_flex_brier=block["PT_flex"]["brier"],
                delta_brier=block["delta_brier_q_minus_PT_flex"]["estimate"],
                delta_brier_lo=block["delta_brier_q_minus_PT_flex"]["ci95"][0],
                delta_brier_hi=block["delta_brier_q_minus_PT_flex"]["ci95"][1],
                q_auc=block["q"]["auc"],
                PT_flex_auc=block["PT_flex"]["auc"],
            )
        )

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "margin_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lambda_rows[0].keys()))
        w.writeheader()
        for r in lambda_rows:
            w.writerow(r)

    triad_path = OUT / "triad_by_p.csv"
    with triad_path.open("w", newline="", encoding="utf-8") as f:
        keys = list(triad[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in triad:
            w.writerow(r)

    n_tot = int(len(y))
    level_share = {k: dict(n=v, share=float(v / n_tot)) for k, v in level_counts.items()}
    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §8 RR4",
        lit="docs/ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md",
        preds="outputs/review_response_rr12_20260920/prediction_table.npz",
        s_Q="outputs/review_response_rr3_quiet_20260920/quiet_mu_s_Q_CAL_fallback.json",
        n_test=n_tot,
        s_Q_level_counts=level_counts,
        s_Q_level_share=level_share,
        n_L_gt_120s=n_L_gt_120,
        share_L_gt_120s=float(n_L_gt_120 / n_tot),
        note_L_gt_120=(
            "L>120s fights were out of RR3 match support; s_Q still resolved via fallback "
            "hierarchy but long-L cells are weakly identified — do not over-interpret λ filters there."
        ),
        triad_by_p=triad,
        lambda_sQ=lambda_payload,
        absolute_cut_auxiliary=abs_payload,
        reading=(
            "Under the tested small-|ΔV| exclusion rules, all-T q−PT_flex ΔBrier sign is preserved. "
            "This does not prove labels are fine for small ΔV, nor that pre-selecting large moves is a deployable rule. "
            "λ filters are post-hoc (use post |ΔV|). Do not promote best TEST λ. "
            "Do not assume all-T sensitivity implies B40 sensitivity — report B40 ΔBrier per λ separately. "
            "Most rows use coarse s_Q fallback (p-only), not full p,t,L support."
        ),
    )
    (OUT / "rr4_margin_results.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")

    md = [
        "# RR4 — small-|ΔV| sensitivity (binary SVI retained)",
        "",
        f"Generated: {payload['generated']}",
        "",
        r"Primary target stays \(Y=1[\Delta V>0]\) because that is the research question "
        r"(direction), not because the triad “proves” direction is predictable. "
        r"λ-grid uses DEV quiet \(s_Q(z)\) from RR3 fallback; absolute cutoffs are auxiliary only.",
        "",
        f"- TEST n={n_tot}; L>120s rows={n_L_gt_120} ({100.0 * n_L_gt_120 / n_tot:.1f}%) — "
        r"outside RR3 equal-L match support; \(s_Q\) there is a coarser fallback reference.",
        "",
        r"## \(s_Q\) fallback composition (all TEST rows)",
        "",
        "| Level used | n | share |",
        "|---|---:|---:|",
    ]
    for lev in ("p_t_L", "p_t", "p", "global"):
        if lev in level_counts:
            md.append(f"| `{lev}` | {level_counts[lev]} | {100.0 * level_counts[lev] / n_tot:.1f}% |")
    md += [
        "",
        r"Most rows use **p-only** quiet scale, not full \(p,t,L\). Do not describe \(s_Q\) as if every fight had dense same-length quiet support.",
        "",
        r"## Direction / mean / scale triad by \(p_{\mathrm{pre}}\)",
        "",
        r"Role: show the three objects differ. Direction predictability is evidenced by RR1/RR2 holdout ΔBrier, not by large \(E[|\Delta V|]\) alone.",
        "",
        "| p bin | n | P(ΔV>0) | E[ΔV] | E[|ΔV|] |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in triad:
        if r.get("n", 0) == 0:
            continue
        md.append(
            f"| [{r['lo']:.1f},{r['hi']:.1f}] | {r['n']} | {fmt(r['P_delta_gt0'])} | "
            f"{fmt(r['E_delta'])} | {fmt(r['E_abs_delta'])} |"
        )
    md += [
        "",
        "## λ · s_Q sensitivity (post-hoc) — all-T",
        "",
        "| λ | n | cov | p_pos | E[|ΔV|] | ΔBrier all-T | CI95 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for lam in LAMBDAS:
        b = lambda_payload[f"lambda_{lam}"]
        d = b["delta_brier_q_minus_PT_flex"]
        md.append(
            f"| {lam} | {b['n']} | {fmt(b['coverage'],3)} | {fmt(b['p_pos'])} | {fmt(b['mean_abs_delta'])} | "
            f"{fmt(d['estimate'])} | [{fmt(d['ci95'][0])}, {fmt(d['ci95'][1])}] |"
        )
    md += [
        "",
        "## λ · s_Q — B40 within each λ-slice (exploratory; not assumed equal to all-T)",
        "",
        "| λ | B40 n | ΔBrier B40 | CI95 |",
        "|---:|---:|---:|---|",
    ]
    for lam in LAMBDAS:
        b = lambda_payload[f"lambda_{lam}"]
        d40 = b.get("B40_delta_brier")
        if not d40:
            md.append(f"| {lam} | — | NA | — |")
            continue
        md.append(
            f"| {lam} | {b.get('B40_n', 'NA')} | {fmt(d40['estimate'])} | "
            f"[{fmt(d40['ci95'][0])}, {fmt(d40['ci95'][1])}] |"
        )
    md += [
        "",
        "## Absolute |ΔV| cutoffs (auxiliary, different definition)",
        "",
        "| cut | n | cov | ΔBrier(q−PT_flex) | CI95 |",
        "|---:|---:|---:|---:|---|",
    ]
    for cut in ABS_CUTS:
        b = abs_payload[f"cut_{cut}"]
        d = b["delta_brier_q_minus_PT_flex"]
        md.append(
            f"| {cut} | {b['n']} | {fmt(b['coverage'],3)} | {fmt(d['estimate'])} | "
            f"[{fmt(d['ci95'][0])}, {fmt(d['ci95'][1])}] |"
        )
    md += [
        "",
        "## Reading",
        "",
        "- Under tested small-change exclusion rules, **all-T** q−PT_flex improvement **does not disappear**.",
        "- That is **not** “small ΔV labels are fine” and **not** a deployable large-move selector.",
        "- Do **not** select λ by TEST performance; do **not** infer B40 from all-T.",
        "- Artifacts: `outputs/review_response_rr4_margin_20260920/`",
        "",
    ]
    doc = REPO / "docs" / "REVIEW_RESPONSE_RR4_MARGIN_20260920.md"
    doc.write_text("\n".join(md), encoding="utf-8")
    (OUT / "REVIEW_RESPONSE_RR4_MARGIN_20260920.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "rr4_margin_results.json", csv_path, doc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
