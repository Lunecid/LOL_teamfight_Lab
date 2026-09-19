#!/usr/bin/env python3
"""Paper primary prediction table — identical 15.16 T rows for every model.

Primary metric: Brier (match-weighted). Primary contrast: q − PT (negative ⇒ q better).
AUC / log loss secondary. Raw p_pre is diagnostic ordering only (not an SVI-probability baseline).

Sources (aligned by match,s_ms):
  - incremental_q: PT, logistic, LightGBM, old_p_pre_logistic≈b(p), p_pre
  - track_a_mlp: MLP, residual MLP
  - svi_lean_tabm: TabM
  - overnight stage_tier_b JSON: TabNet / FT point estimates (no re-bootstrap)

Writes outputs/svi_primary_table_20260919/{results.json,REPORT.md}
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROLE = "EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_NOT_CONFIRMATORY"


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "incremental_q_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    br = float(np.average((p - y) ** 2, weights=w))
    # clip for logloss
    eps = 1e-15
    pc = np.clip(p, eps, 1 - eps)
    ll = float(np.average(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)), weights=w))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        auc = float("nan")
    return dict(brier=br, logloss=ll, auc=auc)


def align(keys_ref: List[Tuple[str, int]], z, col: str) -> np.ndarray:
    keys_z = list(zip(z["match"].astype(str).tolist(), z["s_ms"].astype(np.int64).tolist()))
    idx = {k: i for i, k in enumerate(keys_z)}
    order = np.array([idx[k] for k in keys_ref])
    return z[col][order].astype(np.float64)


def bootstrap_delta_brier(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, g: np.ndarray,
    reps: int = 1000, seed: int = 7,
) -> Dict[str, Any]:
    """Match-clustered bootstrap of Brier(a)−Brier(b).

    Weights are recomputed on the passed row set (callers must pass the evaluation
    cell's rows so point metrics and CI share cell-internal match-equal weights).
    ``estimate`` = observed ΔBrier on those rows; ``boot_mean`` = mean of replicates.
    """
    g = np.asarray(g).astype(str)
    matches, inv = np.unique(g, return_inverse=True)
    n_m = len(matches)
    _, counts = np.unique(inv, return_counts=True)
    w = (1.0 / counts[inv]).astype(np.float64)
    ea = w * (p_a - y) ** 2
    eb = w * (p_b - y) ** 2
    sum_a = np.zeros(n_m, dtype=np.float64)
    sum_b = np.zeros(n_m, dtype=np.float64)
    sum_w = np.zeros(n_m, dtype=np.float64)
    np.add.at(sum_a, inv, ea)
    np.add.at(sum_b, inv, eb)
    np.add.at(sum_w, inv, w)
    den0 = float(sum_w.sum())
    obs = float(sum_a.sum() / den0 - sum_b.sum() / den0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_m, size=(reps, n_m))
    num_a = sum_a[draws].sum(axis=1)
    num_b = sum_b[draws].sum(axis=1)
    den = sum_w[draws].sum(axis=1)
    deltas = num_a / den - num_b / den
    return dict(
        estimate=obs,
        boot_mean=float(np.mean(deltas)),
        ci95=[float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        finite_replicates=int(np.isfinite(deltas).sum()),
        fraction_a_better=float(np.mean(deltas < 0)),
        scope="fixed-model evaluation-sample uncertainty; no re-fit / selection",
        weighting="cell_internal_match_equal",
        reps=reps,
        seed=seed,
        n_matches=int(n_m),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "svi_primary_table_20260919")
    ap.add_argument("--boot-reps", type=int, default=1000)
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    iq_path = data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"
    ta_path = data_root / "outputs" / "track_a_mlp_20260916" / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"
    tabm_path = REPO / "outputs" / "svi_lean_tabm_20260919" / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"
    iq_res = json.loads(
        (data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "results.json").read_text(encoding="utf-8"))
    ta_res = json.loads(
        (data_root / "outputs" / "track_a_mlp_20260916" / "eval" / "results.json").read_text(encoding="utf-8"))
    overnight_tb = REPO / "outputs" / "svi_overnight_20260919" / "stage_tier_b" / "results.json"

    z = np.load(iq_path, allow_pickle=False)
    y = z["y"].astype(int)
    g = z["match"].astype(str)
    p_pre = z["p_pre"].astype(float)
    keys = list(zip(g.tolist(), z["s_ms"].astype(np.int64).tolist()))
    w = match_weights(g)
    b40 = z["cell__B40"].astype(bool) if "cell__B40" in z.files else (p_pre >= 0.4) & (p_pre <= 0.6)

    preds = {
        "p_pre_raw": p_pre,  # diagnostic only for AUC/order; Brier vs SVI mixes estimands
        "b(p)≈old_p_pre_logistic": z["named__old_p_pre_logistic"].astype(float),
        "PT": z["named__pt_winner"].astype(float),
        "logistic": z["named__logit_winner"].astype(float),
        "LightGBM": z["named__lgbm_winner"].astype(float),
    }
    zt = np.load(ta_path, allow_pickle=False)
    preds["MLP"] = align(keys, zt, "named__mlp_winner")
    preds["residual_MLP"] = align(keys, zt, "named__resmlp_winner")
    zm = np.load(tabm_path, allow_pickle=False)
    preds["TabM"] = align(keys, zm, "tabm_winner")

    # Point estimates from overnight for TabNet / FT (already sealed on same rows)
    tb = json.loads(overnight_tb.read_text(encoding="utf-8")) if overnight_tb.is_file() else {}

    def cell_table(mask: np.ndarray) -> Dict[str, Any]:
        # Cell-internal match-equal weights (same rule as bootstrap on this cell)
        ym = y[mask]
        wm = match_weights(g[mask])
        rows = {}
        for name, p in preds.items():
            rows[name] = metrics(ym, p[mask], wm)
        pt_br = rows["PT"]["brier"]
        bp_br = rows["b(p)≈old_p_pre_logistic"]["brier"]
        for name, row in rows.items():
            row["delta_brier_vs_PT"] = row["brier"] - pt_br
            row["delta_brier_vs_b(p)"] = row["brier"] - bp_br
        return dict(
            n=int(mask.sum()),
            n_matches=int(len(np.unique(g[mask]))),
            models=rows,
            weighting="cell_internal_match_equal",
        )

    all_cell = cell_table(np.ones(len(y), dtype=bool))
    b40_cell = cell_table(b40)

    # Bootstrap primary contrasts (fixed predictions)
    print("bootstrap LightGBM - PT ...", flush=True)
    boot_lgbm_pt = bootstrap_delta_brier(y, preds["LightGBM"], preds["PT"], g, args.boot_reps)
    print("bootstrap LightGBM - b(p) ...", flush=True)
    boot_lgbm_bp = bootstrap_delta_brier(y, preds["LightGBM"], preds["b(p)≈old_p_pre_logistic"], g, args.boot_reps)
    print("bootstrap TabM-style - PT ...", flush=True)
    boot_tabm_pt = bootstrap_delta_brier(y, preds["TabM"], preds["PT"], g, args.boot_reps)
    print("bootstrap B40 LightGBM - PT ...", flush=True)
    boot_b40 = bootstrap_delta_brier(
        y[b40], preds["LightGBM"][b40], preds["PT"][b40], g[b40], args.boot_reps)

    # Reuse sealed iq bootstrap as cross-check
    iq_boot = iq_res["results"]["MAIN_TEST"]["T"]["bootstrap"]["all"]["pairs"][0]["a_minus_b"]["brier"]
    iq_b40 = iq_res["results"]["MAIN_TEST"]["T"]["bootstrap"]["B40"]["pairs"][0]["a_minus_b"]["brier"]

    # Diagnostic: p_pre AUC on THIS sample (not pooled)
    diag = dict(
        p_pre_auc_vs_SVI_on_15_16_T=all_cell["models"]["p_pre_raw"]["auc"],
        note="Diagnostic ordering only; do not read as SVI-probability Brier lift",
    )

    tier_b_points = {
        "TabNet": (tb.get("TabNet") or {}).get("sealed"),
        "FT_Transformer": (tb.get("FT_Transformer") or {}).get("sealed"),
    }

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        sample=dict(set="MAIN_TEST", cohort="T", patch_role="15.16 holdout",
                    n=all_cell["n"], n_matches=all_cell["n_matches"]),
        primary_metric="match-weighted Brier",
        primary_contrast="q − PT (negative ⇒ q better)",
        secondary_contrast="q − b(p)",
        all=all_cell,
        B40=b40_cell,
        bootstrap=dict(
            LightGBM_minus_PT=boot_lgbm_pt,
            LightGBM_minus_b_p=boot_lgbm_bp,
            TabM_minus_PT=boot_tabm_pt,
            B40_LightGBM_minus_PT=boot_b40,
            iq_sealed_crosscheck=dict(all=iq_boot, B40=iq_b40),
        ),
        diagnostic_p_pre=diag,
        tier_b_point_estimates_same_rows=tier_b_points,
        winner=dict(
            family="LightGBM",
            reason="Q_SELECT / sealed incremental_q lgbm_winner (selection not on TEST); primary q for this slate",
            selection_source="incremental_q_training_20260915 + svi_reselection_20260919 winner_manifest",
        ),
        naming_notes=dict(
            TabM="TabM-style shared-stem multi-head MLP (train/svi_tabular_meta.py); not claimed as faithful TabM paper reimplementation",
        ),
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x, nd=6):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "NA"
        return f"{x:.{nd}f}"

    lines = [
        "# Primary prediction table — 15.16 T (identical rows)",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        f"Sample: n={all_cell['n']} / matches={all_cell['n_matches']} (15.16 sealed T).",
        "Primary metric: **match-weighted Brier**. Primary contrast: **q − PT**.",
        "AUC secondary. Raw \(p_{pre}\) is diagnostic only.",
        "",
        "## All rows",
        "",
        "| Model | Brier | AUC | log loss | ΔBrier vs PT | ΔBrier vs b(p) | Role |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    order = ["p_pre_raw", "b(p)≈old_p_pre_logistic", "PT", "logistic", "LightGBM",
             "MLP", "residual_MLP", "TabM"]
    role = {
        "p_pre_raw": "diagnostic",
        "b(p)≈old_p_pre_logistic": "SVI baseline (prior only)",
        "PT": "SVI baseline (prior×time) — primary comparator",
        "logistic": "q",
        "LightGBM": "q (winner)",
        "MLP": "q",
        "residual_MLP": "q",
        "TabM": "q (TabM-style sketch)",
    }
    for name in order:
        r = all_cell["models"][name]
        lines.append(
            f"| {name} | {fmt(r['brier'])} | {fmt(r['auc'], 4)} | {fmt(r['logloss'])} | "
            f"{fmt(r['delta_brier_vs_PT'])} | {fmt(r['delta_brier_vs_b(p)'])} | {role[name]} |"
        )
    # tier B points
    if tier_b_points.get("TabNet"):
        s = tier_b_points["TabNet"]
        lines.append(
            f"| TabNet (overnight) | {fmt(s['brier'])} | {fmt(s.get('auc'), 4)} | — | "
            f"{fmt(s.get('delta_vs_pt'))} | — | q (Tier B) |"
        )
    if tier_b_points.get("FT_Transformer"):
        s = tier_b_points["FT_Transformer"]
        lines.append(
            f"| FT-Transformer (overnight) | {fmt(s['brier'])} | {fmt(s.get('auc'), 4)} | — | "
            f"{fmt(s.get('delta_vs_pt'))} | — | q (collapsed) |"
        )

    lines += [
        "",
        "## B40 (p_pre ∈ [0.4, 0.6])",
        "",
        f"n={b40_cell['n']} / matches={b40_cell['n_matches']}",
        "",
        "| Model | Brier | AUC | ΔBrier vs PT |",
        "|---|---:|---:|---:|",
    ]
    for name in ["PT", "b(p)≈old_p_pre_logistic", "logistic", "LightGBM", "MLP", "TabM"]:
        r = b40_cell["models"][name]
        lines.append(f"| {name} | {fmt(r['brier'])} | {fmt(r['auc'], 4)} | {fmt(r['delta_brier_vs_PT'])} |")

    lines += [
        "",
        "## Bootstrap (match-clustered, fixed models)",
        "",
        f"- **LightGBM − PT:** {fmt(boot_lgbm_pt['estimate'])} "
        f"[{fmt(boot_lgbm_pt['ci95'][0])}, {fmt(boot_lgbm_pt['ci95'][1])}] "
        f"(P(Δ<0)={boot_lgbm_pt['fraction_a_better']:.3f})",
        f"- **LightGBM − b(p):** {fmt(boot_lgbm_bp['estimate'])} "
        f"[{fmt(boot_lgbm_bp['ci95'][0])}, {fmt(boot_lgbm_bp['ci95'][1])}]",
        f"- **TabM-style − PT:** {fmt(boot_tabm_pt['estimate'])} "
        f"[{fmt(boot_tabm_pt['ci95'][0])}, {fmt(boot_tabm_pt['ci95'][1])}]",
        f"- **B40 LightGBM − PT:** {fmt(boot_b40['estimate'])} "
        f"[{fmt(boot_b40['ci95'][0])}, {fmt(boot_b40['ci95'][1])}] "
        f"(P(Δ<0)={boot_b40['fraction_a_better']:.3f})"
        + (" — CI includes 0" if boot_b40["ci95"][0] <= 0 <= boot_b40["ci95"][1] else ""),
        "",
        f"iq sealed cross-check LGBM−PT: {iq_boot}",
        "",
        "## Reading",
        "",
        "- Primary \(q\): **LightGBM** (Q_SELECT / sealed incremental_q winner — not selected on TEST).",
        "- **TabM** column = in-repo TabM-style shared-stem multi-head sketch; not claimed as paper-faithful TabM.",
        "- Small overall lift vs PT is a result, not failure; B40 lift remains uncertain (CI includes 0).",
        "- Do not juxtapose pooled-T \(p_{pre}\) AUC with these ΔBrier numbers as one lift story.",
        "- Cell metrics and bootstrap use **cell-internal match-equal** weights.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "out": str(out_dir),
        "lgbm_minus_pt": boot_lgbm_pt["estimate"],
        "lgbm_brier": all_cell["models"]["LightGBM"]["brier"],
        "b40_ci_includes_0": boot_b40["ci95"][0] < 0 < boot_b40["ci95"][1],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
