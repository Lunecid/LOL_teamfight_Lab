"""L6 of label production: write LABEL_LOCK.json once every gate has passed (never edited afterwards).

Gates: all six V_LOCKs scored (PRED manifests), label rows assembled from all six,
layer-1 validation passed, layer-2 diagnostics and the clock-only sidecar
written over the frozen Q_DEV keys (and, when it exists, the non-event control
sidecar).  The lock binds every file by SHA256; q may start only after it exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import assemble, score  # noqa: E402

VERSION = "label_lock_v1_20261005"
REQUIRED = ("ENGAGEMENTS.npz", "ENGAGEMENTS_MANIFEST.json", "Q_DEV_H90_KEYS.npy", "Q_DEV_H90_KEYS.json", "LABELS_MANIFEST.json",
            "LABELS_h60.npz", "LABELS_h90.npz", "LABELS_h120.npz", "STABILITY_h90.npz", "VALIDATION_LAYER1.json",
            "DIAGNOSTICS_LAYER2.json", "CLOCK_ONLY_h90.npz", "CLOCK_ONLY_h90.json")
OPTIONAL = ("CONTROLS_h90.npz", "CONTROLS_h90.json")


def lock(labels: Path, run_root: Path) -> dict[str, Any]:
    labels = Path(labels)
    target = labels / "LABEL_LOCK.json"
    if target.exists():
        raise FileExistsError("labels are already locked; a change needs a new label version")
    missing = [name for name in REQUIRED if not (labels / name).is_file()]
    missing += [f"PRED_{c}.npz" for c in assemble.CONTEXTS if not (labels / f"PRED_{c}.npz").is_file()]
    if missing:
        raise ValueError(f"cannot lock, missing: {missing}")
    validation = json.loads((labels / "VALIDATION_LAYER1.json").read_text(encoding="utf-8"))
    if validation.get("passed") is not True:
        raise ValueError(f"layer-1 validation did not pass: {validation.get('blocking_problems')}")
    manifest = json.loads((labels / "LABELS_MANIFEST.json").read_text(encoding="utf-8"))
    if set(manifest["predictions"]) != set(assemble.CONTEXTS):
        raise ValueError("label rows were not assembled from all six contexts")
    files = {name: score.sha256_file(labels / name) for name in (*REQUIRED, *OPTIONAL) if (labels / name).is_file()}
    files.update({f"PRED_{c}.npz": score.sha256_file(labels / f"PRED_{c}.npz") for c in assemble.CONTEXTS})
    files.update({f"PRED_{c}_MANIFEST.json": score.sha256_file(labels / f"PRED_{c}_MANIFEST.json") for c in assemble.CONTEXTS})
    v_locks = {c: score.sha256_file(Path(run_root) / c / "V_LOCK.json") for c in assemble.CONTEXTS}
    current_code = score.code_hashes()
    for c in assemble.CONTEXTS:
        if v_locks[c] != manifest["predictions"][c]["v"]["v_lock_sha256"]:
            raise ValueError(f"{c}: V_LOCK changed after scoring")
        scored_code = json.loads((labels / f"PRED_{c}_MANIFEST.json").read_text(encoding="utf-8")).get("code_sha256")
        if scored_code != current_code:
            raise ValueError(f"{c}: label input/scoring code changed after scoring; rescore under a new label version")
    record = {"version": VERSION, "locked_utc": datetime.now(timezone.utc).isoformat(), "contract_version": assemble.CONTRACT_VERSION,
              "contract_sha256": score.sha256_file(assemble.CONTRACT), "seed": 20261001, "phase": "one-seed phase 1 (seed 20261001)",
              "main_horizon_s": 90, "sensitivity_horizons_s": [60, 120], "files": files, "v_locks_sha256": v_locks,
              "code_sha256": {path.name: score.sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
              "counts": manifest["counts"], "controls_present": all((labels / name).is_file() for name in OPTIONAL),
              "q_may_start": True}
    pending = labels / f".LABEL_LOCK.json.{os.getpid()}.pending"
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, target)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    record = lock(Path(args.labels), Path(args.run_root))
    print(json.dumps({"locked_utc": record["locked_utc"], "counts": record["counts"]["h90"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
