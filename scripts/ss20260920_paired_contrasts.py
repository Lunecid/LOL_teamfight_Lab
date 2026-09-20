#!/usr/bin/env python3
"""Scale-split paired ΔBrier contrasts (§5) + CORP point values.

--variant sel (default): T009 sigmoid-selected packs (_qS/_qT/_qTS)
--variant id: T010 identity packs (_qS_id/_qT_id/_qTS_id)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from forecast_diagnostics import metrics_bundle  # noqa: E402
from rr20260920_review_response_rr12 import bootstrap_delta_brier, metrics  # noqa: E402

OUT = REPO / "outputs" / "scale_split_TvsS_20260920"
ROLE = "EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE"
T_QTS = REPO / "outputs" / "review_response_rr12_20260920_qTS_on_T"
T_FROZEN = REPO / "outputs" / "review_response_rr12_20260920"

VARIANT = {
    "sel": dict(
        suffix="",
        out_name="paired_contrasts.json",
        note="RR12 two-stage calibrator selection (sensitivity)",
        primary_label="sensitivity",
    ),
    "id": dict(
        suffix="_id",
        out_name="paired_contrasts_id.json",
        note="contract §4 identity calibrator (main)",
        primary_label="primary",
    ),
}


def load_pred(path: Path) -> Dict[str, np.ndarray]:
    p = np.load(path / "prediction_table.npz", allow_pickle=False)
    return {k: p[k] for k in p.files}


def align_keys(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> None:
    ka = list(zip(a["match"].astype(str).tolist(), a["s"].astype(np.int64).tolist()))
    kb = list(zip(b["match"].astype(str).tolist(), b["s"].astype(np.int64).tolist()))
    if ka != kb:
        raise SystemExit(f"prediction key/order mismatch {len(ka)} vs {len(kb)}")


def contrast(name: str, y, p_a, p_b, g, n_note: str, primary: bool = False) -> Dict[str, Any]:
    boot = bootstrap_delta_brier(y, p_a, p_b, g, reps=2000, seed=7)
    return dict(
        name=name,
        primary=primary,
        n=int(len(y)),
        n_matches=int(len(np.unique(np.asarray(g).astype(str)))),
        seed=7,
        boot_reps=2000,
        note=n_note,
        delta_brier=boot,
        metrics_a=metrics(y, p_a, g),
        metrics_b=metrics(y, p_b, g),
    )


def corp_cell(y, p, g, label: str) -> Dict[str, Any]:
    bun = metrics_bundle(y, p, g)
    return dict(
        label=label,
        n=bun["n"],
        n_matches=bun["n_matches"],
        brier=bun["brier"],
        CORP=dict(
            MCB=bun["corp"]["MCB"],
            DSC=bun["corp"]["DSC"],
            UNC=bun["corp"]["UNC"],
            ECE=bun.get("ece"),
            gap=bun["corp"]["reconstruction_gap"],
        ),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["sel", "id"], default="sel")
    args = ap.parse_args(argv)
    cfg = VARIANT[args.variant]
    sfx = cfg["suffix"]
    s_qs = REPO / f"outputs/review_response_rr12_20260920_S_qS{sfx}"
    s_qt = REPO / f"outputs/review_response_rr12_20260920_S_qT{sfx}"
    s_qts = REPO / f"outputs/review_response_rr12_20260920_S_qTS{sfx}"

    for need in (s_qs, s_qt, s_qts, T_QTS, T_FROZEN):
        if not (need / "prediction_table.npz").is_file():
            raise SystemExit(f"missing {need / 'prediction_table.npz'}")

    qs = load_pred(s_qs)
    qt = load_pred(s_qt)
    qts = load_pred(s_qts)
    align_keys(qs, qt)
    align_keys(qs, qts)

    y = qs["y"].astype(float)
    g = qs["match"].astype(str)
    b40 = qs["B40"].astype(bool)
    p_qs = qs["p_q_base"].astype(float)
    p_pt = qs["p_PT_flex"].astype(float)
    p_qt = qt["p_q_base"].astype(float)
    p_qts_s = qts["p_q_base"].astype(float)

    is_main = args.variant == "id"
    contrasts = [
        contrast(
            "q_S_minus_PT_flex_S",
            y,
            p_qs,
            p_pt,
            g,
            f"S TEST; {cfg['note']}",
            primary=is_main,
        ),
        contrast("q_S_minus_q_T_to_S", y, p_qs, p_qt, g, "S TEST; secondary transfer"),
        contrast("q_S_minus_q_TS", y, p_qs, p_qts_s, g, "S TEST; secondary pooled"),
        contrast(
            "q_S_minus_PT_flex_S_B40",
            y[b40],
            p_qs[b40],
            p_pt[b40],
            g[b40],
            "S TEST ∩ B40; secondary (within-S only)",
            primary=False,
        ),
    ]

    t_qts = load_pred(T_QTS)
    t_fr = load_pred(T_FROZEN)
    align_keys(t_qts, t_fr)
    yt = t_fr["y"].astype(float)
    gt = t_fr["match"].astype(str)
    contrasts.append(
        contrast(
            "q_TS_minus_q_T",
            yt,
            t_qts["p_q_base"].astype(float),
            t_fr["p_q_base"].astype(float),
            gt,
            "T TEST; secondary pooling cost",
        )
    )

    corp = dict(
        q_S=corp_cell(y, p_qs, g, "q_S"),
        PT_flex_S=corp_cell(y, p_pt, g, "PT_flex_S"),
    )

    table1 = {
        "q_S": metrics(y, p_qs, g),
        "q_T_to_S": metrics(y, p_qt, g),
        "q_TS": metrics(y, p_qts_s, g),
        "PT_flex_S": metrics(y, p_pt, g),
        "PT_linear_S": metrics(y, qs["p_PT_linear"].astype(float), g),
        "b_spline_S": metrics(y, qs["p_b_spline"].astype(float), g),
        "b_linear_S": metrics(y, qs["p_b_linear"].astype(float), g),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / cfg["out_name"]
    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        variant=args.variant,
        calibrator_note=cfg["note"],
        contract="docs/SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md",
        sources=dict(
            S_qS=str(s_qs.as_posix()),
            S_qT=str(s_qt.as_posix()),
            S_qTS=str(s_qts.as_posix()),
            T_qTS_on_T=str(T_QTS.as_posix()),
            T_frozen_rr12=str(T_FROZEN.as_posix()),
        ),
        table1_S_TEST=table1,
        contrasts=contrasts,
        CORP=corp,
        primary_name="q_S_minus_PT_flex_S",
    )
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_path, flush=True)
    for c in contrasts:
        d = c["delta_brier"]
        tag = "PRIMARY" if c["primary"] else "secondary"
        print(
            f"  [{tag}] {c['name']}: {d['estimate']:+.5f} "
            f"[{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}] p_gt0={d['p_gt0']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
