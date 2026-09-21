#!/usr/bin/env python3
"""RR0 — response manifest: digests + TEST key parity (no model refit)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
QDIR = REPO / "outputs" / "q_newv_fit85_20260920"
RR = REPO / "outputs" / "review_response_rr12_20260920"
OUT = REPO / "outputs" / "review_response_rr0_20260920"
DOCS = REPO / "docs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha16(path: Path) -> str:
    return sha256(path)[:16]


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def test_key_check() -> dict:
    lab = np.load(LAB / "TEST_h90.npz", allow_pickle=False)
    pred = np.load(RR / "prediction_table.npz", allow_pickle=False)
    lab_keys = list(zip(lab["match"].astype(str).tolist(), lab["s"].astype(np.int64).tolist()))
    ok = lab["Y_SVI"] >= 0
    if "missing_score" in lab.files:
        ok &= lab["missing_score"] == 0
    lab_ok = [k for k, flag in zip(lab_keys, ok) if flag]
    pred_keys = list(zip(pred["match"].astype(str).tolist(), pred["s"].astype(np.int64).tolist()))
    set_lab, set_pred = set(lab_ok), set(pred_keys)
    return dict(
        n_label_eligible=len(lab_ok),
        n_label_unique=len(set_lab),
        n_label_duplicate_keys=len(lab_ok) - len(set_lab),
        n_prediction=len(pred_keys),
        n_prediction_unique=len(set_pred),
        n_prediction_duplicate_keys=len(pred_keys) - len(set_pred),
        prediction_equals_label_keyset=set_pred == set_lab,
        same_row_order=lab_ok == pred_keys,
        only_in_label=len(set_lab - set_pred),
        only_in_prediction=len(set_pred - set_lab),
        PASS=bool(
            set_pred == set_lab
            and lab_ok == pred_keys
            and len(lab_ok) - len(set_lab) == 0
            and len(pred_keys) - len(set_pred) == 0
        ),
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "TRAIN_oof_STATUS": LAB / "TRAIN_oof_STATUS.json",
        "TRAIN_oof_h90_meta": LAB / "TRAIN_oof_h90_meta.json",
        "TEST_h90": LAB / "TEST_h90.npz",
        "logit_state": QDIR / "models" / "logit_state.joblib",
        "primary_table": QDIR / "primary_table.json",
        "selection_freeze": QDIR / "selection_freeze.json",
        "rr12_paired_ci": RR / "paired_ci.json",
        "rr12_baseline_selection": RR / "baseline_selection.json",
        "rr12_prediction_table": RR / "prediction_table.npz",
        "rr12_PT_flex": RR / "models" / "PT_flex.joblib",
        "rr12_PT_linear": RR / "models" / "PT_linear.joblib",
        "rr12_b_spline": RR / "models" / "b_spline.joblib",
    }
    digests = {}
    for name, path in artifacts.items():
        digests[name] = dict(
            path=str(path.relative_to(REPO)).replace("\\", "/"),
            exists=path.is_file(),
            sha256=sha256(path) if path.is_file() else None,
            sha16=sha16(path) if path.is_file() else None,
            bytes=path.stat().st_size if path.is_file() else None,
        )

    oof = json.loads((LAB / "TRAIN_oof_STATUS.json").read_text(encoding="utf-8"))
    sel = json.loads((RR / "baseline_selection.json").read_text(encoding="utf-8"))
    keys = test_key_check()

    manifest = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic="EXPLORATORY_REVIEW_RESPONSE_PRIOR_TEST_EXPOSURE",
        source_commit=git_head(),
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md",
        execution_addendum="docs/REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md",
        V_bundle_sha256="AC459CC4397630A953D672730CAA687B50FC1EDCD5A029928FAD236FC5B204E3",
        q_base="frozen logit_state (no refit in RR1)",
        g_V="unchanged (fit85 PosSlopeSigmoid inside V bundle)",
        g_q="optional; RR1 selected identity for q and PT_flex",
        OOF_integrity=oof.get("integrity", oof),
        TEST_key_parity=keys,
        digests=digests,
        rr1_execution_as_run=dict(
            spline_knots="sklearn SplineTransformer default knots='uniform' (NOT TRAIN weighted quantiles)",
            fit_sample_weight="match weight 1/n_match (NOT mean-1 renormalized)",
            selection_procedure="two-stage: pick spline hyperparams on raw Q_SELECT Brier, then identity vs sigmoid",
            joint_config_x_calibrator=False,
            calibrator_choice=sel.get("calibrator_choice"),
            PT_flex=sel.get("PT_flex"),
            b_spline=sel.get("b_spline"),
        ),
        bootstrap_field_note=(
            "paired_ci.json field p_gt0 == bootstrap_fraction_positive "
            "(fraction of match-bootstrap draws with ΔBrier>0); not a classical p-value."
        ),
        PASS=bool(keys["PASS"] and oof.get("integrity", {}).get("PASS", False)),
    )
    (OUT / "response_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (DOCS / "REVIEW_RESPONSE_RR0_MANIFEST_20260920.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("PASS" if manifest["PASS"] else "FAIL", OUT / "response_manifest.json")
    return 0 if manifest["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
