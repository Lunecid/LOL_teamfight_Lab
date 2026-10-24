"""EXTERNAL_SCORE_LOCK.json: binds the protocol, the external code, the sealed labels and every external prediction.

README
  Written once, before any external label, q metric or V outcome metric is read (protocol 'order' step 4).  It records
  sha256 of: the protocol and the frozen identities it names (V_LOCK final, V training acceptance, LABEL_LOCK,
  q_run_1 SCORE_LOCK / RUN_RELEASE / Q_SELECTION, P1_BEST_RULE, DECISIONS) - each compared with the protocol's
  'frozen' values; every external code file (E2a/E2b/E2c package) and the frozen modules they import; per dataset the
  E2a bank / projection / history manifests, the engagement table, PRED_final, public label tables, the sealed label
  file (hash and row counts only), the q row table, raw-store build record, XY, and every cell's prediction file and
  meta; both parity reports.  It refuses to write if a parity report is not PASS, if any dataset or cell is missing,
  or if an external evaluation output already exists.  It never opens a sealed file or computes a metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402
from studies.external_validation_20261009 import e2bc_common as B  # noqa: E402

FROZEN_FILES = {
    "v_lock_final_sha256": B.RUN_3 / "final" / "V_LOCK.json",
    "v_training_acceptance_final_sha256": B.RUN_3 / "final" / "TRAINING_ACCEPTANCE.json",
    "label_lock_sha256": B.LABELS_1 / "LABEL_LOCK.json",
    "q_score_lock_sha256": B.Q_RUN_1 / "SCORE_LOCK.json",
    "q_run_release_sha256": B.Q_RUN_1 / "RUN_RELEASE.json",
    "q_selection_sha256": B.Q_RUN_1 / "Q_SELECTION.json",
    "p1_best_rule_sha256": B.P1_BEST_RULE,
    "decisions_sha256": C.ROOT / "reports" / "architecture_20261005" / "DECISIONS_20261005.json",
}


def _frozen_code_paths() -> list[Path]:
    folders = ("label_production_20261005", "q_production_20261005", "v_full_preprocess_20261003", "v_model_fit_input_20261003",
               "v_model_input_20261003", "v_production_20261004", "v_training_gate_20261003")
    paths = []
    for folder in folders:
        paths.extend((C.ROOT / "studies" / folder).glob("*.py"))
    paths.append(C.ROOT / "reports" / "label_production_20261005" / "run_labels.py")
    return paths


def _dataset_record(dataset: str) -> dict:
    from studies.external_validation_20261009.e2c_scores import cell_keys

    labels, q_dir, scores = B.labels_dir(dataset), B.q_inputs_dir(dataset), B.scores_dir(dataset)
    files = {}

    def bind(label: str, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{dataset}: missing {path}")
        files[label] = {"path": str(path), "sha256": B.sha256_file(path), "bytes": path.stat().st_size}

    for stage, name in (("bank", "RUN_MANIFEST.json"), ("preparation", "RUN_MANIFEST.json"), ("items", "RUN_MANIFEST.json"),
                        ("projection", "RUN_MANIFEST.json"), ("history", "ACCEPTANCE.json"), ("history", "PLAN.json")):
        bind(f"e2a:{stage}/{name}", C.stage_dir(dataset, stage) / name)
    for name in ("ENGAGEMENTS.npz", "ENGAGEMENTS_MANIFEST.json", f"QUERIES_{B.V_CONTEXT}.npy", f"PRED_{B.V_CONTEXT}.npz",
                 f"PRED_{B.V_CONTEXT}_MANIFEST.json", "LABELS_MANIFEST_EXTERNAL.json",
                 *[f"LABELS_PUBLIC_h{h}.npz" for h in B.HORIZONS]):
        bind(f"labels:{name}", labels / name)
    bind("labels:SEALED", B.sealed_labels_path(dataset))
    manifest = C.read_json(labels / "LABELS_MANIFEST_EXTERNAL.json")
    if manifest["sealed_sha256"] != files["labels:SEALED"]["sha256"]:
        raise ValueError(f"{dataset}: sealed label file differs from its manifest")
    for name in (f"Q_ROWS_h{B.MAIN_HORIZON}.npz", f"Q_ROWS_h{B.MAIN_HORIZON}.json", "XY.npy", "XY.json", "raw/store/BUILD.json",
                 "raw/store/CHUNK.json", "raw/store/queries.i64.npy"):
        bind(f"q_inputs:{name}", q_dir / name)
    cells = {}
    for key in cell_keys():
        npz, meta = scores / f"{key}.npz", scores / f"{key}.json"
        if not npz.is_file() or not meta.is_file():
            raise FileNotFoundError(f"{dataset}: cell {key} has no prediction")
        record = C.read_json(meta)
        sha = B.sha256_file(npz)
        if record["pred_sha256"] != sha:
            raise ValueError(f"{dataset}: {key} prediction differs from its meta")
        cells[key] = {"pred_sha256": sha, "meta_sha256": B.sha256_file(meta), "rows": record["rows"], "kind": record["kind"],
                      "calibration": record["calibration"], "reload_max_abs_logit_diff": record["reload_max_abs_logit_diff"],
                      "model_sha256": record.get("model_sha256"), "calibrator_sha256": record.get("calibrator_sha256")}
    q_meta = C.read_json(q_dir / f"Q_ROWS_h{B.MAIN_HORIZON}.json")
    eng = C.read_json(labels / "ENGAGEMENTS_MANIFEST.json")
    return {"files": files, "cells": cells,
            "counts": {"engagement_rows": eng["rows"], "valid_label_rows": {h: c["valid"] for h, c in manifest["counts"].items()},
                       "q_rows": q_meta["rows"], "q_matches": q_meta["matches"], "q_rows_by_patch": q_meta["rows_by_patch"],
                       "cells_scored": len(cells)},
            "sealed": {"sha256": manifest["sealed_sha256"], "columns": manifest["sealed_columns"], "opened": False}}


def write_lock(log) -> dict:
    from studies.external_validation_20261009.e2c_scores import protocol_cells

    if B.SCORE_LOCK_OUT.exists():
        raise FileExistsError(f"{B.SCORE_LOCK_OUT} exists; the external score lock is written once")
    proto = B.protocol()
    problems = []
    for path, name in ((B.E2B_PARITY, "E2b"), (B.E2C_PARITY, "E2c")):
        if C.read_json(path).get("status") != "PASS":
            problems.append(f"{name} parity is not PASS")
    frozen = {}
    for name, path in FROZEN_FILES.items():
        actual = B.sha256_file(path)
        frozen[name] = {"path": str(path), "sha256": actual, "protocol": proto["frozen"].get(name)}
        if proto["frozen"].get(name) not in (None, actual):
            problems.append(f"{name}: {actual} differs from the protocol value")
    release = C.read_json(B.Q_RUN_1 / "RUN_RELEASE.json")
    frozen["q_scientific_release_json"] = {"path": str(release["release"]), "sha256": B.sha256_file(Path(release["release"])),
                                           "digest": release["digest"]}
    evaluation_outputs = sorted(str(p) for p in C.REPORTS.glob("EXTERNAL_EVAL*")) + sorted(str(p) for p in C.REPORTS.glob("*EXTERNAL_REPORT*"))
    if evaluation_outputs:
        problems.append(f"external evaluation outputs already exist: {evaluation_outputs}")
    datasets = {}
    for dataset in B.EXTERNAL_DATASETS:
        try:
            datasets[dataset] = _dataset_record(dataset)
        except (FileNotFoundError, ValueError) as error:
            problems.append(str(error))
    if problems:
        raise RuntimeError("cannot lock: " + "; ".join(problems))
    payload = {
        "version": B.VERSION, "kind": "EXTERNAL_SCORE_LOCK", "locked_utc": C.utc_now(),
        "protocol": {"path": str(B.PROTOCOL), "sha256": B.sha256_file(B.PROTOCOL), "version": proto["version"]},
        "frozen": frozen,
        "parity": {name: {"path": str(path), "sha256": B.sha256_file(path), "status": C.read_json(path)["status"]}
                   for name, path in (("E2b", B.E2B_PARITY), ("E2c", B.E2C_PARITY), ("E2a", C.PARITY_REPORT))},
        "parity_attempts": {"E2b_attempt1_unpadded": B.sha256_file(C.REPORTS / "E2b_PARITY_REPORT_attempt1_unpadded.json")
                            if (C.REPORTS / "E2b_PARITY_REPORT_attempt1_unpadded.json").is_file() else None,
                            "E2c_attempt1_as_scored": B.sha256_file(C.REPORTS / "E2c_PARITY_REPORT_attempt1_as_scored.json")
                            if (C.REPORTS / "E2c_PARITY_REPORT_attempt1_as_scored.json").is_file() else None},
        "cells": protocol_cells(),
        "code": {"external": B.code_hashes(B.e2bc_code_files() + list(C.PACKAGE.glob("e2a_*.py"))),
                 "frozen_imported": B.code_hashes(_frozen_code_paths())},
        "datasets": datasets,
        "e2a_report_sha256": B.sha256_file(C.E2A_REPORT),
        "stage_times": {"path": str(B.STAGE_TIMES), "sha256": B.sha256_file(B.STAGE_TIMES)},
        "rules": {"labels_sealed": "EXTERNAL_LABELS_SEALED_<dataset>.npz; never opened before this lock",
                  "q_rows_y": -1, "no_metric_computed": True, "padding": ("V scoring and neural q scoring pad the final "
                  "partial batch to the full batch shape (copies of the last row, dropped); see the parity reports"),
                  "evaluation": "once, by the author, after checking this lock (protocol 'order' step 5)"},
    }
    B.write_json_once(B.SCORE_LOCK_OUT, payload)
    log(f"[lock] wrote {B.SCORE_LOCK_OUT} sha256={B.sha256_file(B.SCORE_LOCK_OUT)}")
    return payload
