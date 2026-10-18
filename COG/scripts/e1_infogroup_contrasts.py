#!/usr/bin/env python3
"""E1 contrasts D_E|F and D_F|E from saved TEST predictions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/supplementary_e1_infogroups_20260921"
DOCS_JSON = REPO / "docs/SUPPLEMENTARY_E1_INFOGROUPS_20260921.json"
DOCS_MD = REPO / "docs/SUPPLEMENTARY_E1_INFOGROUPS_20260921.md"


def load_pred(path: Path):
    z = np.load(path, allow_pickle=True)
    return z["match"].astype(str), z["y"].astype(float), z["p"].astype(float)


def brier(y, p, g) -> float:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / c[inv]
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def paired_delta(y, p_a, p_b, g) -> Dict[str, Any]:
    """L(a)-L(b): negative means a better."""
    y = np.asarray(y, float)
    p_a = np.asarray(p_a, float)
    p_b = np.asarray(p_b, float)
    g = np.asarray(g).astype(str)
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / c[inv]
    row = (p_a - y) ** 2 - (p_b - y) ** 2
    sums = np.zeros(len(c), dtype=np.float64)
    counts = np.zeros(len(c), dtype=np.float64)
    np.add.at(sums, inv, row)
    np.add.at(counts, inv, 1.0)
    d_m = sums / counts
    rng = np.random.default_rng(7)
    boots = d_m[rng.integers(0, len(d_m), size=(2000, len(d_m)))].mean(axis=1)
    return dict(
        delta_row_weighted=float(np.sum(w * row) / np.sum(w)),
        delta_match_mean=float(np.mean(d_m)),
        ci95=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        n_matches=int(len(d_m)),
        n_rows=int(len(y)),
    )


def cohort_block(cohort: str) -> Dict[str, Any]:
    summary = json.loads((ROOT / cohort / "summary.json").read_text(encoding="utf-8"))
    preds = {}
    for arm in ("M-F0", "M-F1", "M-F2", "M-F3"):
        path = ROOT / cohort / f"{arm}_TEST_pred.npz"
        g, y, p = load_pred(path)
        preds[arm] = dict(g=g, y=y, p=p, brier=brier(y, p, g), selected=summary["arms"][arm]["selected"])

    # align on match+order assumption: same TEST pack → identical row order
    y = preds["M-F3"]["y"]
    g = preds["M-F3"]["g"]
    for arm in preds:
        assert len(preds[arm]["y"]) == len(y)

    contrasts = {
        "D_E_given_F": paired_delta(y, preds["M-F3"]["p"], preds["M-F1"]["p"], g),  # L3-L1
        "D_F_given_E": paired_delta(y, preds["M-F3"]["p"], preds["M-F2"]["p"], g),  # L3-L2
        "M-F1_minus_M-F0": paired_delta(y, preds["M-F1"]["p"], preds["M-F0"]["p"], g),
        "M-F2_minus_M-F0": paired_delta(y, preds["M-F2"]["p"], preds["M-F0"]["p"], g),
        "M-F3_minus_M-F0": paired_delta(y, preds["M-F3"]["p"], preds["M-F0"]["p"], g),
    }
    # rename keys for clarity: design D_E|F = L(F3)-L(F1)
    return dict(
        cohort=cohort,
        arm_brier={a: preds[a]["brier"] for a in preds},
        selected={a: summary["arms"][a]["selected"] for a in summary["arms"]},
        contrasts=contrasts,
    )


def main() -> int:
    blocks = {}
    for cohort in ("T", "S"):
        if not (ROOT / cohort / "summary.json").is_file():
            raise SystemExit(f"missing summary for {cohort}")
        blocks[cohort] = cohort_block(cohort)

    doc = dict(
        schema="SUPPLEMENTARY_E1_INFOGROUPS_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        contract="docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#4",
        task=".ai/tasks/T018.md",
        primary_family=[
            "T:D_E_given_F",
            "T:D_F_given_E",
            "S:D_E_given_F",
            "S:D_F_given_E",
        ],
        reading=(
            "Negative D means the fuller arm has lower Brier. "
            "Absence of gain ≠ information absence. Not a game mechanism. "
            "Not comparable to frozen logit_state vs PT_flex headlines."
        ),
        cohorts=blocks,
    )
    DOCS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Supplementary E1 — Frame×event info-group LGBM",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        "",
        doc["reading"],
        "",
    ]
    for cohort, block in blocks.items():
        lines.append(f"## Cohort {cohort}")
        lines.append("")
        lines.append("| Arm | TEST Brier | selected cand | dim |")
        lines.append("|---|---:|---|---:|")
        for arm, br in block["arm_brier"].items():
            sel = block["selected"][arm]
            lines.append(
                f"| {arm} | {br:.5f} | {sel['cand_id']} (iter {sel['best_iteration']}) | {sel['n_features']} |"
            )
        lines.append("")
        lines.append("| Contrast | Δ (row-wtd) | Δ (match-mean) | CI95 |")
        lines.append("|---|---:|---:|---|")
        for name, c in block["contrasts"].items():
            lines.append(
                f"| {name} | {c['delta_row_weighted']:+.5f} | {c['delta_match_mean']:+.5f} | "
                f"[{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] |"
            )
        lines.append("")
    DOCS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOCS_JSON}")
    print(f"wrote {DOCS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
