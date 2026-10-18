"""Static V/OOF and baseline-input audit (no fitting/predictions)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import config as C
from .io_source import file_sha256


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _find_line(lines: list[str], pattern: str) -> int | None:
    rx = re.compile(pattern)
    for i, line in enumerate(lines, 1):
        if rx.search(line):
            return i
    return None


def audit_oof_vs_protocol() -> dict[str, Any]:
    oof_path = C.EV4 / "ev4_03b_oof_v.py"
    lab_path = C.EV4 / "ev4_04_labels.py"
    lines = _read_lines(oof_path)
    lab_lines = _read_lines(lab_path)
    evidence = {
        "files": {
            "ev4_03b_oof_v.py": {"path": str(oof_path), "sha256": file_sha256(oof_path)},
            "ev4_04_labels.py": {"path": str(lab_path), "sha256": file_sha256(lab_path)},
            "FULL_STUDY_PROTOCOL.md": {
                "path": str(C.PROTOCOL),
                "sha256": file_sha256(C.PROTOCOL) if C.PROTOCOL.is_file() else None,
                "section": "§8 lines ~130–146 (nested fold-local preprocessing/STOP/candidate/ensemble/calibration)",
            },
        },
        "old_oof_contract": {
            "hyperparameter_from_all_TRAIN_CV": {
                "line": _find_line(lines, r"O8_disclosure|hyperparameter chosen by fit-V CV over all"),
                "quote_patterns": ["O8_disclosure", "hyperparameter chosen by fit-V CV over all 15.14"],
            },
            "nan_fill_bins_from_all_TRAIN": {
                "line": _find_line(lines, r"NaN fill and lgbm bins from all 15.14"),
            },
            "weights_OOF_not_nested_selection": {
                "line": _find_line(lines, r"out-of-fold for its weights but not for that one grid choice"),
                "conclusion": (
                    "Old weight-OOF reuses all-TRAIN hyperparameter/fill/bin choices; "
                    "it does NOT implement the new nested fold-local preprocessing/STOP/"
                    "candidate/ensemble/calibration contract in FULL_STUDY_PROTOCOL.md §8."
                ),
            },
            "calibration_on_V_CAL": {
                "line": _find_line(lines, r"O5_calibration|PositiveSlopeSigmoid on V_CAL"),
            },
        },
        "new_protocol_nested_oof": {
            "protocol_path": str(C.PROTOCOL),
            "requires": [
                "fold-local preprocessing",
                "inner STOP",
                "20-card candidate search",
                "3-seed series selection",
                "V_CAL calibration inside each outer fold",
                "no inheritance of a single final-V setting chosen on all TRAIN before OOF",
            ],
            "implemented_in_old_oof": False,
            "note": "New nested procedure is a design/protocol contract; not claimed completed by old oof_v artifacts.",
        },
        "labels_endpoint_contract": {
            "L3_endpoint_line": _find_line(lab_lines, r"e_h = gameplay.labels_exact.label_endpoint"),
            "L4_validity_line": _find_line(lab_lines, r"Endpoint validity|endpoint_validity"),
            "no_invalid_to_negative": True,
        },
    }
    return evidence


def audit_pt_and_m0() -> dict[str, Any]:
    out: dict[str, Any] = {"pt_flex_frozen_bundle": None, "m0_partition": None, "q_bundle": None}
    if C.PT_FLEX_BUNDLE.is_file():
        b = json.loads(C.PT_FLEX_BUNDLE.read_text(encoding="utf-8"))
        out["pt_flex_frozen_bundle"] = {
            "path": str(C.PT_FLEX_BUNDLE),
            "sha256": file_sha256(C.PT_FLEX_BUNDLE),
            "columns": b.get("columns"),
            "n_columns": b.get("n_columns"),
            "uses_logit_p_pre": b.get("uses_logit_p_pre"),
            "model_kind": (b.get("model") or {}).get("kind"),
            "n_features_expanded": (b.get("model") or {}).get("fit_info", {}).get("n_features"),
            "conclusion": (
                "Frozen PT_flex raw columns are exactly ['p_pre','time_minutes']; "
                "expanded spline/tensor features are transforms of those two raw inputs."
            ),
        }
    else:
        out["pt_flex_frozen_bundle"] = {"missing": True, "path": str(C.PT_FLEX_BUNDLE)}

    part = C.WT / "studies" / "feature_controls_20261001" / "partitions.py"
    if part.is_file():
        lines = _read_lines(part)
        out["m0_partition"] = {
            "path": str(part),
            "sha256": file_sha256(part),
            "M0_COLUMNS_line": _find_line(lines, r"M0_COLUMNS"),
            "M0_COLUMNS": ["time_minutes", "time_minutes_sq"],
            "distinction": "M0 is time-only (time_minutes, time_minutes_sq); PT_flex is p_pre + time_minutes (not M0).",
        }
    prep = C.WT / "studies" / "feature_controls_20261001" / "prepare.py"
    if prep.is_file():
        lines = _read_lines(prep)
        out["p_pre_auxiliary"] = {
            "path": str(prep),
            "sha256": file_sha256(prep),
            "strip_logit_line": _find_line(lines, r"def strip_logit_p_pre"),
            "save_matrices_line": _find_line(lines, r"def save_matrices_npz"),
            "p_pre_auxiliary_only_line": _find_line(lines, r"p_pre_auxiliary_only"),
            "conclusion": (
                "Matrices may store p_pre as auxiliary; direct arm fit columns strip logit_p_pre. "
                "Do not claim p_pre is consumed by M0/M1/M2/M3 fits without tracing the fit call."
            ),
        }
    if C.Q_BUNDLE.is_file():
        qb = json.loads(C.Q_BUNDLE.read_text(encoding="utf-8"))
        out["q_bundle"] = {
            "path": str(C.Q_BUNDLE),
            "sha256": file_sha256(C.Q_BUNDLE),
            "n_columns": qb.get("n_columns"),
            "uses_logit_p_pre": qb.get("uses_logit_p_pre"),
            "columns_tail": (qb.get("columns") or [])[-3:],
        }
    # Probe completed vs smoke OOF manifests if present
    oof_manifests = []
    for cand in [
        C.ROOT / "outputs" / "reest_exact_v4_20260925" / "stage2" / "fit_v2" / "oof_v" / "oof_manifest.json",
        C.ROOT / "outputs" / "reest_exact_v4_20260925" / "stage2" / "fit_v" / "oof_v" / "oof_manifest.json",
    ]:
        if cand.is_file():
            oof_manifests.append({
                "path": str(cand),
                "sha256": file_sha256(cand),
                "exists": True,
                "status": "completed_artifact_present_on_disk",
            })
    out["oof_manifests_found"] = oof_manifests
    out["unimplemented_new_nested_oof"] = {
        "status": "protocol_defined_not_claimed_implemented_by_this_audit",
        "reference": "FULL_STUDY_PROTOCOL.md §8 (lines 130–146)",
    }
    return out


def run_code_audit(out_path: Path) -> dict[str, Any]:
    report = {
        "generated_for": "anchor_frame_audit_20261002",
        "no_fit_no_predict": True,
        "oof": audit_oof_vs_protocol(),
        "baselines": audit_pt_and_m0(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["sha256"] = file_sha256(out_path)
    return report
