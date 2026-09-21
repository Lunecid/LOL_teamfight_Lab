#!/usr/bin/env python3
"""Time-banded model performance — primary report cut on recorded engagement clock.

Clock: sealed MAIN_TEST_h90_T `s_ms` / `time_minutes` (Match-V5-derived onset ms).
Bands: [0,10), [10,20), [20,30), [30,inf) — same as lift / V̂ contracts.

Per band (cell-internal match-equal weights):
  - full q table: p_pre, b(p), PT, logistic, LightGBM, MLP, residual_MLP, TabM
  - bootstrap LightGBM − PT
Also: V̂→W on engagement p_pre / p_post_h90 by the same onset bands;
      EXT T LGBM−PT by time band (headline cohorts).

Writes outputs/svi_time_banded_models_20260919/{results.json,REPORT.md}
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
T_BANDS = (
    ("all", None, None),
    ("t_0_10", 0.0, 10.0),
    ("t_10_20", 10.0, 20.0),
    ("t_20_30", 20.0, 30.0),
    ("t_30_inf", 30.0, 1e9),
)
EXT = ("EXT_KR_16.13", "EXT_NA1_16.13", "EXT_KR_16.15")


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
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        auc = float("nan")
    return dict(brier=br, auc=auc)


def align(keys_ref: List[Tuple[str, int]], z, col: str) -> np.ndarray:
    keys_z = list(zip(z["match"].astype(str).tolist(), z["s_ms"].astype(np.int64).tolist()))
    idx = {k: i for i, k in enumerate(keys_z)}
    return z[col][np.array([idx[k] for k in keys_ref])].astype(np.float64)


def bootstrap_delta_brier(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, g: np.ndarray,
    reps: int = 1000, seed: int = 7,
) -> Dict[str, Any]:
    g = np.asarray(g).astype(str)
    if len(y) < 50 or len(np.unique(g)) < 25:
        return dict(estimate=None, ci95=None, skipped="too few", n=int(len(y)))
    matches, inv = np.unique(g, return_inverse=True)
    n_m = len(matches)
    _, counts = np.unique(inv, return_counts=True)
    w = (1.0 / counts[inv]).astype(np.float64)
    ea = w * (p_a - y) ** 2
    eb = w * (p_b - y) ** 2
    sum_a = np.zeros(n_m)
    sum_b = np.zeros(n_m)
    sum_w = np.zeros(n_m)
    np.add.at(sum_a, inv, ea)
    np.add.at(sum_b, inv, eb)
    np.add.at(sum_w, inv, w)
    den0 = float(sum_w.sum())
    obs = float(sum_a.sum() / den0 - sum_b.sum() / den0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_m, size=(reps, n_m))
    den = sum_w[draws].sum(axis=1)
    deltas = sum_a[draws].sum(axis=1) / den - sum_b[draws].sum(axis=1) / den
    return dict(
        estimate=obs,
        boot_mean=float(np.mean(deltas)),
        ci95=[float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        fraction_q_better=float(np.mean(deltas < 0)),
        n=int(len(y)),
        n_matches=int(n_m),
        weighting="cell_internal_match_equal",
    )


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def band_mask(tmin: np.ndarray, lo, hi) -> np.ndarray:
    if lo is None:
        return np.ones(len(tmin), dtype=bool)
    return (tmin >= lo) & (tmin < hi)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "svi_time_banded_models_20260919")
    ap.add_argument("--boot-reps", type=int, default=1000)
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    iq_path = data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"
    ta_path = data_root / "outputs" / "track_a_mlp_20260916" / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"
    tabm_path = REPO / "outputs" / "svi_lean_tabm_20260919" / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"

    z = np.load(iq_path, allow_pickle=False)
    y = z["y"].astype(int)
    g = z["match"].astype(str)
    s_ms = z["s_ms"].astype(np.int64)
    tmin = z["time_minutes"].astype(float) if "time_minutes" in z.files else s_ms / 60000.0
    p_pre = z["p_pre"].astype(float)
    keys = list(zip(g.tolist(), s_ms.tolist()))

    preds = {
        "p_pre_raw": p_pre,
        "b(p)": z["named__old_p_pre_logistic"].astype(float),
        "PT": z["named__pt_winner"].astype(float),
        "logistic": z["named__logit_winner"].astype(float),
        "LightGBM": z["named__lgbm_winner"].astype(float),
    }
    zt = np.load(ta_path, allow_pickle=False)
    preds["MLP"] = align(keys, zt, "named__mlp_winner")
    preds["residual_MLP"] = align(keys, zt, "named__resmlp_winner")
    zm = np.load(tabm_path, allow_pickle=False)
    preds["TabM"] = align(keys, zm, "tabm_winner")

    # Engagement V̂ anchors (same onset clock)
    lab = np.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
        allow_pickle=False)
    coh = np.load(
        data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz",
        allow_pickle=False)
    # Align W via outcomes
    match_w: Dict[str, int] = {}
    od = data_root / "outputs" / "full_corpus_training_20260915" / "extract" / "MAIN" / "outcomes_SEALED"
    for p in sorted(od.glob("chunk_*.npz")):
        oz = np.load(p, allow_pickle=False)
        for m, w in zip(oz["o_match"].astype(str), oz["o_winner_blue"].astype(int)):
            match_w[m] = int(w)

    m_eng = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    g_e = lab["match"][m_eng].astype(str)
    t_e = lab["s"][m_eng].astype(float) / 60000.0
    p_pre_e = lab["p_pre"][m_eng].astype(float)
    p_post_e = lab["p_post_h90"][m_eng].astype(float)
    y_e = np.array([match_w.get(x, -1) for x in g_e], dtype=int)
    ok = y_e >= 0
    g_e, t_e, p_pre_e, p_post_e, y_e = g_e[ok], t_e[ok], p_pre_e[ok], p_post_e[ok], y_e[ok]

    q_by_band: Dict[str, Any] = {}
    v_by_band: Dict[str, Any] = {}

    for name, lo, hi in T_BANDS:
        print(f"band {name}…", flush=True)
        mq = band_mask(tmin, lo, hi)
        ym, gm = y[mq], g[mq]
        wm = match_weights(gm)
        models = {}
        for mn, p in preds.items():
            models[mn] = metrics(ym, p[mq], wm)
        pt_br = models["PT"]["brier"]
        for mn, row in models.items():
            row["delta_brier_vs_PT"] = row["brier"] - pt_br
        boot = bootstrap_delta_brier(
            ym, preds["LightGBM"][mq], preds["PT"][mq], gm, args.boot_reps)
        q_by_band[name] = dict(
            n=int(mq.sum()),
            n_matches=int(len(np.unique(gm))),
            clock="s_ms / time_minutes (recorded onset)",
            models=models,
            boot_LGBM_minus_PT=boot,
        )

        mv = band_mask(t_e, lo, hi)
        v_by_band[name] = dict(
            n=int(mv.sum()),
            n_matches=int(len(np.unique(g_e[mv]))),
            pre=metrics(y_e[mv], p_pre_e[mv], match_weights(g_e[mv])),
            post_h90=metrics(y_e[mv], p_post_e[mv], match_weights(g_e[mv])),
        )

    # EXT q by time
    ext_out: Dict[str, Any] = {}
    pred_dir = data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "predictions"
    for cohort in EXT:
        path = pred_dir / f"{cohort}_h90_T.npz"
        if not path.is_file():
            ext_out[cohort] = dict(error="missing")
            continue
        ez = np.load(path, allow_pickle=False)
        ey = ez["y"].astype(int)
        eg = ez["match"].astype(str)
        et = ez["time_minutes"].astype(float) if "time_minutes" in ez.files else ez["s_ms"].astype(float) / 60000.0
        e_lgbm = ez["named__lgbm_winner"].astype(float)
        e_pt = ez["named__pt_winner"].astype(float)
        bands = {}
        for name, lo, hi in T_BANDS:
            m = band_mask(et, lo, hi)
            if int(m.sum()) < 50:
                bands[name] = dict(n=int(m.sum()), skipped=True)
                continue
            wm = match_weights(eg[m])
            bands[name] = dict(
                n=int(m.sum()),
                n_matches=int(len(np.unique(eg[m]))),
                LightGBM=metrics(ey[m], e_lgbm[m], wm),
                PT=metrics(ey[m], e_pt[m], wm),
                delta_brier_LGBM_minus_PT=float(
                    metrics(ey[m], e_lgbm[m], wm)["brier"] - metrics(ey[m], e_pt[m], wm)["brier"]),
            )
        ext_out[cohort] = bands

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        clock_source="MAIN_TEST_h90_T.s_ms / time_minutes; engagement lab['s']; Match-V5 timeline ms",
        bands=[b[0] for b in T_BANDS],
        q_15_16_T=q_by_band,
        V_engagement_anchors=v_by_band,
        EXT_q=ext_out,
    )

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

    (out_dir / "results.json").write_text(
        json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # REPORT — band-first
    lines = [
        "# Model performance by recorded time band",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        "Clock = sealed engagement onset `s_ms` / `time_minutes` (Match-V5-derived).",
        "Band tables are **primary**; `all` row is summary only.",
        "Weights: cell-internal match-equal. Contrast: ΔBrier = Brier(q)−Brier(PT) (negative ⇒ q better).",
        "",
        "## 1. q → SVI — 15.16 T by time band",
        "",
    ]
    for name, _, _ in T_BANDS:
        cell = q_by_band[name]
        boot = cell["boot_LGBM_minus_PT"]
        lines += [
            f"### {name}  (n={cell['n']} / matches={cell['n_matches']})",
            "",
            "| Model | Brier | AUC | ΔBrier vs PT |",
            "|---|---:|---:|---:|",
        ]
        for mn in ("p_pre_raw", "b(p)", "PT", "logistic", "LightGBM", "MLP", "residual_MLP", "TabM"):
            r = cell["models"][mn]
            lines.append(
                f"| {mn} | {fmt(r['brier'], 6)} | {fmt(r['auc'], 4)} | {fmt(r['delta_brier_vs_PT'], 6)} |"
            )
        if boot.get("estimate") is not None:
            ci = boot["ci95"]
            lines += [
                "",
                f"LightGBM − PT: **{fmt(boot['estimate'], 6)}** "
                f"[{fmt(ci[0], 6)}, {fmt(ci[1], 6)}]  "
                f"P(Δ<0)={fmt(boot['fraction_q_better'], 3)}",
                "",
            ]
        else:
            lines += ["", f"Bootstrap skipped: {boot.get('skipped')}", ""]

    lines += [
        "## 2. Headline — LightGBM − PT across bands",
        "",
        "| Band | n | ΔBrier | 95% CI | P(Δ<0) |",
        "|---|---:|---:|---|---:|",
    ]
    for name, _, _ in T_BANDS:
        b = q_by_band[name]["boot_LGBM_minus_PT"]
        if b.get("estimate") is None:
            lines.append(f"| {name} | {q_by_band[name]['n']} | NA | NA | NA |")
            continue
        ci = b["ci95"]
        lines.append(
            f"| {name} | {q_by_band[name]['n']} | {fmt(b['estimate'], 6)} | "
            f"[{fmt(ci[0], 6)}, {fmt(ci[1], 6)}] | {fmt(b['fraction_q_better'], 3)} |"
        )

    lines += [
        "",
        "## 3. Vhat -> W — same onset bands (engagement anchors)",
        "",
        "| Band | n | pre Brier | pre AUC | post Brier | post AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, _, _ in T_BANDS:
        v = v_by_band[name]
        lines.append(
            f"| {name} | {v['n']} | {fmt(v['pre']['brier'], 6)} | {fmt(v['pre']['auc'], 4)} | "
            f"{fmt(v['post_h90']['brier'], 6)} | {fmt(v['post_h90']['auc'], 4)} |"
        )

    lines += [
        "",
        "## 4. Transfer EXT T — LGBM − PT by time band",
        "",
    ]
    for cohort, bands in ext_out.items():
        if "error" in bands:
            lines.append(f"- {cohort}: missing")
            continue
        lines += [f"### {cohort}", "", "| Band | n | LGBM Brier | PT Brier | ΔBrier |", "|---|---:|---:|---:|---:|"]
        for name, _, _ in T_BANDS:
            r = bands[name]
            if r.get("skipped"):
                continue
            lines.append(
                f"| {name} | {r['n']} | {fmt(r['LightGBM']['brier'], 6)} | "
                f"{fmt(r['PT']['brier'], 6)} | {fmt(r['delta_brier_LGBM_minus_PT'], 6)} |"
            )
        lines.append("")

    lines += [
        "## Reading",
        "",
        "- Early band can have CI covering 0; mid/late bands often carry the sealed lift.",
        "- Do not replace band tables with a single pooled ΔBrier in the paper body.",
        "- V̂ skill also rises with the same recorded clock — I2 warrant, not fight-win truth.",
        "",
    ]
    # Fix latex escapes in file - write without bad escapes
    text = "\n".join(lines).replace("\\(\\widehat{V}\\to W\\)", "V̂→W")
    (out_dir / "REPORT.md").write_text(text, encoding="utf-8")
    print("wrote", out_dir)
    for name, _, _ in T_BANDS:
        b = q_by_band[name]["boot_LGBM_minus_PT"]
        print(name, "n", q_by_band[name]["n"], "Δ", b.get("estimate"), "CI", b.get("ci95"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
