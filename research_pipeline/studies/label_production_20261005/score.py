"""L2 scoring: a context's locked V on label query rows (raw logit, raw probability, calibrated V).

The model is rebuilt the way ``worker.run_select`` rebuilds the selected
checkpoint (``build_locked_model`` with the REFIT dimensions + ``model_refit``
state, or the REFIT pickle for table families) and the probability path is the
one the V_CAL calibrator was fitted on: ``_sigmoid(_model_logits(...))`` and
``PositiveSlopeCalibrator.predict`` (float64).

Scoring is cut into canonical 1,024-row batches of the sorted query list and
grouped into parts; a part is written once (fsync + atomic rename) with its
hash, so an interrupted run continues from the parts it already has, and the
result does not depend on how many workers ran.

Usage (one worker): python -B studies/label_production_20261005/score.py --run-root D:/LOL_Training_20261004/run_3
                     --labels D:/LOL_Training_20261004/labels_1 --context final --parts 0 4 8
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from studies.label_production_20261005 import queries  # noqa: E402

VERSION = "label_scoring_v1_20261005"
SEED = 20261001
TABLE_FAMILIES = ("logistic", "lightgbm")
BATCHES_PER_PART = 32
CALIBRATOR_FIELDS = ("a", "b", "loss", "projected_gradient_inf", "converged", "class_counts")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def code_hashes() -> dict[str, str]:
    """The label input/scoring code a prediction depends on (the V code is bound through the V_LOCK's scientific release)."""

    here = Path(__file__).resolve().parent
    files = [here / "score.py", here / "queries.py",
             ROOT / "studies/v_full_preprocess_20261003/surface.py", ROOT / "studies/v_model_fit_input_20261003/common.py",
             ROOT / "studies/v_model_input_20261003/view.py"]
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files}


class LockedV:
    """The selected, calibrated V of one context, rebuilt from its V_LOCK.json."""

    def __init__(self, run_root: Path, context: str, *, device: str = "cuda", lock_path: Path | None = None) -> None:
        from studies.v_production_20261004.train import PositiveSlopeCalibrator, configure_runtime
        from studies.v_training_gate_20261003.guard import build_locked_model

        root = Path(run_root) / context
        lock_path = Path(lock_path) if lock_path is not None else root / "V_LOCK.json"  # an explicit path is for tests only
        lock = read_json(lock_path)
        if lock.get("complete") is not True or lock.get("context") != context or int(lock.get("seed", -1)) != SEED:
            raise ValueError(f"{context}: V_LOCK is not a complete phase-1 lock")
        selected = lock["selected"]
        self.family, self.card_id = str(selected["family"]), str(selected["card_id"])
        checkpoint = Path(selected["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        if sha256_file(checkpoint) != lock["checkpoint_sha256"]:
            raise ValueError(f"{context}: selected checkpoint differs from its V_LOCK hash")
        self.calibrator = PositiveSlopeCalibrator(**{name: lock["calibrator"][name] for name in CALIBRATOR_FIELDS})
        plan = read_json(root / "CONTEXT_PLAN.json")
        if self.family in TABLE_FAMILIES:
            configure_runtime(SEED, "cpu")
            with checkpoint.open("rb") as stream:
                self.model = pickle.load(stream)
            self.device = None
        else:
            import torch

            configure_runtime(SEED, device)
            self.device = torch.device(device)
            self.model = build_locked_model(self.family, self.card_id, acceptance_path=root / "TRAINING_ACCEPTANCE.json", seed=SEED,
                                            dimensions=plan["dimensions"][f"REFIT_TRAIN:{self.family}"], cpu_threads=1)
            self.model.to(self.device)
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)["model_state"]
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
        self.identity = {"context": context, "family": self.family, "card_id": self.card_id, "checkpoint": str(checkpoint),
                         "checkpoint_sha256": lock["checkpoint_sha256"], "v_lock_sha256": sha256_file(lock_path),
                         "scientific_sha256": lock.get("scientific_sha256"), "training_acceptance_sha256": lock.get("training_acceptance_sha256"),
                         "calibrator": {name: lock["calibrator"][name] for name in CALIBRATOR_FIELDS},
                         "device": str(self.device) if self.device is not None else "cpu"}

    def score(self, structured: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(logit, raw probability, calibrated V), the last two in float64."""

        from studies.v_production_20261004.train import _model_logits, _sigmoid

        inputs = queries.family_inputs(structured, self.family)
        if self.device is not None:
            import torch

            with torch.no_grad():
                values = _model_logits(self.model, self.family, inputs, self.device).detach().cpu().numpy()
        else:
            values = _model_logits(self.model, self.family, inputs, None)
        raw = _sigmoid(values)  # the V_CAL fitting path in worker.run_select
        calibrated = self.calibrator.predict(raw)
        return np.asarray(values, dtype=np.float64), np.asarray(raw, dtype=np.float64), np.asarray(calibrated, dtype=np.float64)


def query_list(labels_root: Path, context: str) -> tuple[np.ndarray, Path]:
    """The sorted query list of a context, written once next to the engagement table."""

    path = Path(labels_root) / f"QUERIES_{context}.npy"
    if path.is_file():
        return np.load(path, allow_pickle=False), path
    engagements = dict(np.load(Path(labels_root) / "ENGAGEMENTS.npz", allow_pickle=False))
    indices = queries.context_queries(engagements, context)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.save(stream, indices, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return indices, path


def part_ranges(n_rows: int) -> list[tuple[int, int]]:
    rows_per_part = BATCHES_PER_PART * queries.BATCH_ROWS
    return [(start, min(start + rows_per_part, n_rows)) for start in range(0, n_rows, rows_per_part)]


def part_path(labels_root: Path, context: str, part: int) -> Path:
    return Path(labels_root) / f"PRED_{context}" / f"part_{part:05d}.npz"


def part_complete(path: Path, expected: np.ndarray) -> bool:
    receipt = path.with_suffix(".json")
    if not path.is_file() or not receipt.is_file():
        return False
    try:
        record = read_json(receipt)
        if record.get("sha256") != sha256_file(path):
            return False
        with np.load(path, allow_pickle=False) as stored:
            return np.array_equal(stored["query_index"], expected)
    except (OSError, ValueError, KeyError):
        return False


def score_parts(run_root: Path, labels_root: Path, context: str, parts: list[int], *, device: str = "cuda",
                lock_path: Path | None = None) -> list[int]:
    indices, list_path = query_list(labels_root, context)
    ranges = part_ranges(len(indices))
    todo = [part for part in parts if not part_complete(part_path(labels_root, context, part), indices[slice(*ranges[part])])]
    if not todo:
        return []
    v = LockedV(run_root, context, device=device, lock_path=lock_path)
    surface, bundle = queries.open_inputs(context)
    try:
        for part in todo:
            start, stop = ranges[part]
            chunk = indices[start:stop]
            logits, raw, calibrated = [], [], []
            for b_start in range(0, len(chunk), queries.BATCH_ROWS):  # canonical batches: fixed by the sorted list
                batch = chunk[b_start:b_start + queries.BATCH_ROWS]
                rows, seen = [], []
                for query_index, row, _meta in surface.iter_query_rows(batch):
                    rows.append(row)
                    seen.append(query_index)
                if not np.array_equal(np.asarray(seen, dtype=np.int64), batch):
                    raise ValueError(f"{context}: query rows came back out of order in part {part}")
                structured = queries.transform(rows, bundle)
                values = v.score(structured)
                for store, value in zip((logits, raw, calibrated), values):
                    store.append(value)
            payload = {"query_index": chunk, "logit": np.concatenate(logits), "raw_p": np.concatenate(raw), "v": np.concatenate(calibrated)}
            for name in ("raw_p", "v"):
                if not np.isfinite(payload[name]).all():
                    raise ValueError(f"{context}: non-finite {name} in part {part}")
            target = part_path(labels_root, context, part)
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
            with pending.open("wb") as stream:
                np.savez(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, target)
            receipt = {"version": VERSION, "context": context, "part": part, "rows": int(len(chunk)), "range": [int(start), int(stop)],
                       "sha256": sha256_file(target), "queries_sha256": sha256_file(list_path), "v": v.identity,
                       "code_sha256": code_hashes(), "batch_rows": queries.BATCH_ROWS, "utc": datetime.now(timezone.utc).isoformat()}
            pending_receipt = target.with_suffix(".json.pending")
            pending_receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(pending_receipt, target.with_suffix(".json"))
    finally:
        surface.close()
    return todo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--context", required=True, choices=queries.CONTEXTS)
    parser.add_argument("--parts", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    from studies.v_production_20261004.worker import keep_awake

    keep_awake()
    done = score_parts(Path(args.run_root), Path(args.labels), args.context, list(args.parts), device=args.device)
    print(json.dumps({"context": args.context, "scored_parts": done}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
