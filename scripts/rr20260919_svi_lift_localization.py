#!/usr/bin/env python3
"""Lift localization: where does q beat PT?

Primary contrast LightGBM − PT (ΔBrier, match-weighted) on identical 15.16 T rows,
broken by p_pre bands and time bands (and their crossing). Match-clustered bootstrap
per cell (fixed models). Also scores the same cells on 2026 EXT_* T predictions.

Answers: small overall ΔBrier — is it concentrated in balanced / late game, or diffuse?

Writes outputs/svi_lift_localization_20260919/{results.json,REPORT.md}
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

P_BANDS = (
    ("B_all", None, None),
    ("B20", 0.0, 0.2),
    ("B40", 0.4, 0.6),
    ("B45", 0.45, 0.55),
    ("B60", 0.6, 0.8),
    ("B80", 0.8, 1.01),
    ("B_skew_low", 0.0, 0.4),
    ("B_skew_high", 0.6, 1.01),
)
T_BANDS = (
    ("T_all", None, None),
    ("t_0_10", 0.0, 10.0),
    ("t_10_20", 10.0, 20.0),
    ("t_20_30", 20.0, 30.0),
    ("t_30_inf", 30.0, 1e9),
)
EXT = (
    "EXT_KR_16.13",
    "EXT_NA1_16.13",
    "EXT_KR_16.15",
    "EXT_KR_16.14_pilot",
)


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "incremental_q_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def brier(y, p, w) -> float:
    return float(np.average((p - y) ** 2, weights=w))


def bootstrap_delta(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, g: np.ndarray,
    reps: int = 1000, seed: int = 7,
) -> Dict[str, Any]:
    g = np.asarray(g).astype(str)
    if len(y) < 30 or len(np.unique(g)) < 20:
        return dict(estimate=None, ci95=None, n=int(len(y)), n_matches=int(len(np.unique(g))),
                    skipped="too few rows/matches")
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
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_m, size=(reps, n_m))
    den = sum_w[draws].sum(axis=1)
    deltas = sum_a[draws].sum(axis=1) / den - sum_b[draws].sum(axis=1) / den
    return dict(
        estimate=float(np.mean(deltas)),
        ci95=[float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        fraction_q_better=float(np.mean(deltas < 0)),
        n=int(len(y)),
        n_matches=int(n_m),
        reps=reps,
        seed=seed,
        scope="fixed-model eval-sample uncertainty",
    )


def band_mask(p_pre: np.ndarray, tmin: np.ndarray, p_name: str, plo, phi, t_name: str, tlo, thi) -> np.ndarray:
    m = np.ones(len(p_pre), dtype=bool)
    if plo is not None:
        m &= (p_pre >= plo) & (p_pre < phi)
    if tlo is not None:
        m &= (tmin >= tlo) & (tmin < thi)
    return m


def cell_stats(y, p_q, p_pt, p_bp, g, mask) -> Dict[str, Any]:
    if mask.sum() < 30:
        return dict(n=int(mask.sum()), skipped="n<30")
    ym, gm = y[mask], g[mask]
    w = match_weights(gm)
    br_q = brier(ym, p_q[mask], w)
    br_pt = brier(ym, p_pt[mask], w)
    br_bp = brier(ym, p_bp[mask], w)
    try:
        from sklearn.metrics import roc_auc_score
        auc_q = float(roc_auc_score(ym, p_q[mask], sample_weight=w))
        auc_pt = float(roc_auc_score(ym, p_pt[mask], sample_weight=w))
    except Exception:
        auc_q = auc_pt = float("nan")
    return dict(
        n=int(mask.sum()),
        n_matches=int(len(np.unique(gm))),
        brier_q=br_q,
        brier_pt=br_pt,
        brier_bp=br_bp,
        delta_q_minus_pt=br_q - br_pt,
        delta_q_minus_bp=br_q - br_bp,
        auc_q=auc_q,
        auc_pt=auc_pt,
    )


def load_main_test(data_root: Path):
    z = np.load(
        data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "predictions" / "MAIN_TEST_h90_T.npz",
        allow_pickle=False)
    y = z["y"].astype(int)
    g = z["match"].astype(str)
    p_pre = z["p_pre"].astype(float)
    # time: prefer column; else from cell masks only for known bands
    if "time_minutes" in z.files:
        tmin = z["time_minutes"].astype(float)
    else:
        # reconstruct coarse time from cell indicators if present
        tmin = np.full(len(y), np.nan)
        for name, lo, hi in (("cell__time_0_10", 5.0), ("cell__time_10_20", 15.0),
                             ("cell__time_20_30", 25.0), ("cell__time_30_inf", 35.0)):
            if name in z.files:
                tmin[z[name].astype(bool)] = lo
        if not np.isfinite(tmin).all():
            # fallback: s_ms if available elsewhere — use 0
            tmin = np.where(np.isfinite(tmin), tmin, 15.0)
    return dict(
        y=y, g=g, p_pre=p_pre, tmin=tmin,
        q=z["named__lgbm_winner"].astype(float),
        pt=z["named__pt_winner"].astype(float),
        bp=z["named__old_p_pre_logistic"].astype(float),
    )


def load_ext(data_root: Path, set_name: str) -> Optional[Dict[str, np.ndarray]]:
    path = data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "predictions" / f"{set_name}_h90_T.npz"
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    tmin = z["time_minutes"].astype(float) if "time_minutes" in z.files else np.full(len(z["y"]), 15.0)
    return dict(
        y=z["y"].astype(int),
        g=z["match"].astype(str),
        p_pre=z["p_pre"].astype(float),
        tmin=tmin,
        q=z["named__lgbm_winner"].astype(float),
        pt=z["named__pt_winner"].astype(float),
        bp=z["named__old_p_pre_logistic"].astype(float) if "named__old_p_pre_logistic" in z.files
        else z["named__pt_winner"].astype(float),
    )


def run_grid(data: Dict[str, np.ndarray], boot: bool, reps: int, seed: int,
             crossings: bool = True) -> Dict[str, Any]:
    y, g = data["y"], data["g"]
    p_pre, tmin = data["p_pre"], data["tmin"]
    q, pt, bp = data["q"], data["pt"], data["bp"]
    cells = {}
    # marginal p bands
    for p_name, plo, phi in P_BANDS:
        m = band_mask(p_pre, tmin, p_name, plo, phi, "T_all", None, None)
        st = cell_stats(y, q, pt, bp, g, m)
        if boot and "skipped" not in st:
            st["boot_q_minus_pt"] = bootstrap_delta(y[m], q[m], pt[m], g[m], reps, seed)
        cells[p_name] = st
    # marginal time bands
    for t_name, tlo, thi in T_BANDS:
        if t_name == "T_all":
            continue
        m = band_mask(p_pre, tmin, "B_all", None, None, t_name, tlo, thi)
        st = cell_stats(y, q, pt, bp, g, m)
        if boot and "skipped" not in st:
            st["boot_q_minus_pt"] = bootstrap_delta(y[m], q[m], pt[m], g[m], reps, seed)
        cells[t_name] = st
    # crossings of interest
    if crossings:
        for p_name, plo, phi in (("B40", 0.4, 0.6), ("B45", 0.45, 0.55)):
            for t_name, tlo, thi in T_BANDS[1:]:
                key = f"{p_name}_x_{t_name}"
                m = band_mask(p_pre, tmin, p_name, plo, phi, t_name, tlo, thi)
                st = cell_stats(y, q, pt, bp, g, m)
                if boot and "skipped" not in st and st["n"] >= 80:
                    st["boot_q_minus_pt"] = bootstrap_delta(y[m], q[m], pt[m], g[m], reps, seed)
                cells[key] = st
    return cells


def fmt(x, nd=6):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Lift localization — LightGBM − PT by strata",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        "Primary contrast: **ΔBrier = Brier(LightGBM) − Brier(PT)** on identical rows (negative ⇒ q better).",
        "Bootstrap = fixed-model, match-clustered eval-sample uncertainty.",
        "",
        "## 15.16 T — p_pre bands",
        "",
        "| Band | n | ΔBrier q−PT | 95% CI | P(Δ<0) | Δ vs b(p) |",
        "|---|---:|---:|---|---:|---:|",
    ]
    main = payload["MAIN_TEST_T"]
    for key in ("B_all", "B_skew_low", "B40", "B45", "B_skew_high", "B20", "B60", "B80"):
        c = main.get(key) or {}
        if "skipped" in c:
            lines.append(f"| {key} | {c.get('n', 0)} | — | — | — | — |")
            continue
        boot = c.get("boot_q_minus_pt") or {}
        ci = boot.get("ci95")
        ci_s = f"[{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "—"
        lines.append(
            f"| {key} | {c['n']} | {fmt(c['delta_q_minus_pt'])} | {ci_s} | "
            f"{fmt(boot.get('fraction_q_better'), 3)} | {fmt(c['delta_q_minus_bp'])} |"
        )
    lines += [
        "",
        "## 15.16 T — time bands",
        "",
        "| Band | n | ΔBrier q−PT | 95% CI | P(Δ<0) |",
        "|---|---:|---:|---|---:|",
    ]
    for key in ("t_0_10", "t_10_20", "t_20_30", "t_30_inf"):
        c = main.get(key) or {}
        if "skipped" in c:
            continue
        boot = c.get("boot_q_minus_pt") or {}
        ci = boot.get("ci95")
        ci_s = f"[{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "—"
        lines.append(
            f"| {key} | {c['n']} | {fmt(c['delta_q_minus_pt'])} | {ci_s} | "
            f"{fmt(boot.get('fraction_q_better'), 3)} |"
        )
    lines += [
        "",
        "## 15.16 T — B40 × time",
        "",
        "| Cell | n | ΔBrier q−PT | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for key in sorted(k for k in main if k.startswith("B40_x_")):
        c = main[key]
        if "skipped" in c:
            lines.append(f"| {key} | {c.get('n', 0)} | — | — |")
            continue
        boot = c.get("boot_q_minus_pt") or {}
        ci = boot.get("ci95")
        ci_s = f"[{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "—"
        lines.append(f"| {key} | {c['n']} | {fmt(c['delta_q_minus_pt'])} | {ci_s} |")

    lines += ["", "## Transfer (2026) — headline cells", ""]
    lines.append("| Cohort | all Δ | B40 Δ | t_20_30 Δ | n_all |")
    lines.append("|---|---:|---:|---:|---:|")
    for set_name, block in (payload.get("transfer") or {}).items():
        if "error" in block:
            lines.append(f"| {set_name} | error | — | — | — |")
            continue
        a = block.get("B_all") or {}
        b = block.get("B40") or {}
        t = block.get("t_20_30") or {}
        lines.append(
            f"| {set_name} | {fmt(a.get('delta_q_minus_pt'))} | {fmt(b.get('delta_q_minus_pt'))} | "
            f"{fmt(t.get('delta_q_minus_pt'))} | {a.get('n', 'NA')} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- Overall lift can be small while some strata carry most of the signal (or none).",
        "- B40 CI including 0 remains an honest uncertainty statement for balanced fights.",
        "- Transfer strata: score-only; no selection on external.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "svi_lift_localization_20260919")
    ap.add_argument("--boot-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("load MAIN_TEST T…", flush=True)
    main_data = load_main_test(data_root)
    print("strata + bootstrap on 15.16…", flush=True)
    main_cells = run_grid(main_data, boot=True, reps=args.boot_reps, seed=args.seed)

    transfer = {}
    for set_name in EXT:
        print(f"transfer {set_name}…", flush=True)
        d = load_ext(data_root, set_name)
        if d is None:
            transfer[set_name] = dict(error="missing predictions")
            continue
        # point estimates only on external (no bootstrap to keep runtime modest; optional)
        transfer[set_name] = run_grid(d, boot=False, reps=args.boot_reps, seed=args.seed, crossings=False)

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        primary_contrast="LightGBM − PT (ΔBrier)",
        sample_main=dict(n=int(len(main_data["y"])), n_matches=int(len(np.unique(main_data["g"])))),
        MAIN_TEST_T=main_cells,
        transfer=transfer,
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    # headline print
    for k in ("B_all", "B40", "B45", "t_20_30", "t_30_inf"):
        c = main_cells.get(k) or {}
        print(k, "n=", c.get("n"), "d=", c.get("delta_q_minus_pt"), "ci=", (c.get("boot_q_minus_pt") or {}).get("ci95"))
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
