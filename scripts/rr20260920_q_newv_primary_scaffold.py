#!/usr/bin/env python3
"""Primary q table scaffold under new-V labels (fit only when TRAIN OOF exists).

Baselines: constant, b(p), PT, logistic(state), LightGBM(state).
Selection on Q_SELECT Brier; seal on TEST. Does not run until TRAIN_oof_h90.npz exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
OUT = REPO / "outputs" / "q_newv_fit85_20260920"


def main() -> int:
    train = LAB / "TRAIN_oof_h90.npz"
    status = {
        "ready_for_q_fit": train.is_file(),
        "have_labels": {
            "TEST": (LAB / "TEST_h90.npz").is_file(),
            "Q_CAL": (LAB / "Q_CAL_h90.npz").is_file(),
            "Q_SELECT": (LAB / "Q_SELECT_h90.npz").is_file(),
            "TRAIN_oof": train.is_file(),
        },
        "next": (
            "Run fold MLP OOF worker to write TRAIN_oof_h90.npz, then re-run this script "
            "with --fit (to be implemented once OOF labels land)."
            if not train.is_file()
            else "TRAIN OOF present — implement/fit primary table next."
        ),
        "contract": "docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md",
        "candidates": ["constant", "b_p", "PT", "logit_state", "lgbm_state", "mlp_state"],
        "primary_contrast": "q - PT ΔBrier on identical 15.16 T rows",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "PRIMARY_STATUS.json"
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status["have_labels"]["TEST"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
