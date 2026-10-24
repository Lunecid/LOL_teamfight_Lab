"""E2b/E2c common: frozen release paths, the protocol, layout, compute gates and the EXTERNAL query surface.

README (E2b labels, E2c q scores; protocol reports/external_validation_20261009/EXTERNAL_PROTOCOL_20261009.json,
DECISION_LOG_KO.md 10-09 18:34)
  E2b rebuilds, from the E2a outputs of a dataset, the labels_1 engagement table (label_production engagements.build),
  scores every pre/post query with the frozen final-context V (score.LockedV, run_3/final, V_CAL calibrator) and
  assembles the label rows (assemble.label_table).  Label-bearing columns are written only to the sealed file
  EXTERNAL_LABELS_SEALED_<dataset>.npz; manifests carry hashes and row counts, never a label value or rate.
  E2c builds the q inputs of the external h90 rows (features.build_chunk raw store, spatial.read_query_xy XY, the frozen
  main_h90 REFIT_TRAIN context of q_run_1) and scores the protocol cells with the models re-loaded as qtest.score does.
  Nothing is fitted or tuned on external data; the q row tables carry y = -1.

  ExternalQuerySurface: label_production queries.QuerySurface with _records_for_match minus the 2025 role check
  (surface._match_role accepts only patches 15.14-15.16); every other check of the frozen reader is kept.

Layout (DATA_ROOT = D:/LOL_External_20261009): labels/<dataset>, q_inputs/<dataset>, scores/<dataset>; the E2a
inputs are e2a_common.stage_dir(dataset, 'bank' | 'projection' | 'history').
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402
from studies.external_validation_20261009 import e2a_na1  # noqa: E402,F401  (registers na1_16_13)

VERSION = "external_validation_e2bc_20261009"
PROTOCOL = C.REPORTS / "EXTERNAL_PROTOCOL_20261009.json"
STAGE_TIMES = C.REPORTS / "E2bc_STAGE_TIMES.jsonl"
SCORE_LOCK_OUT = C.REPORTS / "EXTERNAL_SCORE_LOCK.json"
E2B_PARITY = C.REPORTS / "E2b_PARITY_REPORT.json"
E2C_PARITY = C.REPORTS / "E2c_PARITY_REPORT.json"

TRAINING = Path(r"D:\LOL_Training_20261004")
RUN_3 = TRAINING / "run_3"
LABELS_1 = TRAINING / "labels_1"
Q_RUN_1 = TRAINING / "q_run_1"
FLAGS_LOG = TRAINING / "boundary_flags_1" / "RUN_FLAGS.log"
V_CONTEXT = "final"
P1_BEST_RULE = C.ROOT / "reports" / "q_production_20261005" / "P1_BEST_RULE_20261008.json"
HORIZONS = (60, 90, 120)
MAIN_HORIZON = 90
EXTERNAL_DATASETS = (C.DATASET_PARITY, C.DATASET_2025, C.DATASET_2026, e2a_na1.DATASET_NA1)

# Columns of assemble.label_table that carry the label or reveal it (post state, SVI, sign/zero/tie flags).
SEALED_LABEL_COLUMNS = ("direction_positive", "svi_delta", "exact_zero", "near_zero_bits", "logit_post", "raw_post",
                        "p_post", "raw_delta", "clip_saturation_tie_bits")


def labels_dir(dataset: str) -> Path:
    return C.DATA_ROOT / "labels" / dataset


def q_inputs_dir(dataset: str) -> Path:
    return C.DATA_ROOT / "q_inputs" / dataset


def scores_dir(dataset: str) -> Path:
    return C.DATA_ROOT / "scores" / dataset


def sealed_labels_path(dataset: str) -> Path:
    return labels_dir(dataset) / f"EXTERNAL_LABELS_SEALED_{dataset}.npz"


def sha256_file(path) -> str:
    return C.sha256_file(path)


def protocol() -> dict:
    return C.read_json(PROTOCOL)


def cpu_workers_allowed(requested: int) -> int:
    """At most 2 CPU workers until RUN_FLAGS.log reports 'evaluate rc', then at most 6 (coordinator rule)."""
    try:
        released = "evaluate rc" in FLAGS_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        released = False
    cap = C.MAX_WORKERS if released else 2
    return max(1, min(int(requested), cap))


def record_time(row: dict) -> None:
    STAGE_TIMES.parent.mkdir(parents=True, exist_ok=True)
    with STAGE_TIMES.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Timer:
    """Context manager appending one E2bc_STAGE_TIMES.jsonl row (status ok/failed)."""

    def __init__(self, dataset: str, stage: str, log, **extra):
        self.dataset, self.stage, self.log, self.extra = dataset, stage, log, extra

    def __enter__(self):
        self.started, self.t0 = C.utc_now(), time.perf_counter()
        return self

    def __exit__(self, kind, value, tb):
        seconds = round(time.perf_counter() - self.t0, 1)
        status = "ok" if kind is None else "failed"
        record_time({"dataset": self.dataset, "stage": self.stage, "started_utc": self.started, "completed_utc": C.utc_now(),
                     "seconds": seconds, "status": status, "error": None if kind is None else f"{kind.__name__}: {value}"[:500],
                     "argv": sys.argv, **self.extra})
        self.log(f"[time] {self.dataset}/{self.stage}: {status} {seconds} s")
        return False


def write_npz(path: Path, arrays: dict) -> str:
    """assemble.write_npz semantics (fsync + atomic rename); returns sha256."""
    from studies.label_production_20261005.assemble import write_npz as frozen_write_npz

    return frozen_write_npz(Path(path), arrays)


def write_json_once(path: Path, payload: dict) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; it is written once")
    return C.write_json(path, payload)


def external_surface(dataset: str):
    """ExternalQuerySurface over the dataset's E2a shared history and projection."""
    return make_surface_class()(C.stage_dir(dataset, "history"), C.stage_dir(dataset, "projection"), seed=20261001,
                                verify_projection_hashes=False)


_SURFACE_CLASS = None


def make_surface_class():
    global _SURFACE_CLASS
    if _SURFACE_CLASS is not None:
        return _SURFACE_CLASS
    from collections.abc import Mapping

    from studies.label_production_20261005.queries import QuerySurface
    from studies.v_full_preprocess_20261003.surface import _field, _integer, _key

    class ExternalQuerySurface(QuerySurface):
        """QuerySurface for EXTERNAL matches: _records_for_match without the 2025 patch/role decoding."""

        def _records_for_match(self, match_index: int):
            match = self.loader.matches[match_index]
            if int(match["role"]) != C.EXTERNAL_ROLE_CODE or int(match["patch_role"]) != C.EXTERNAL_PATCH_ROLE_CODE:
                raise ValueError(f"match {match_index} is not an EXTERNAL match")
            key = _key(_field(match, "key"))
            records = self._reader_for(match_index).match_records(match_index)
            count = _integer(self.loader.match_queries[match_index]["n_queries"], "n_queries")
            first_query = _integer(self.loader.match_queries[match_index]["first_query"], "first_query")
            expected_ids = list(range(first_query, first_query + count))
            actual_ids = [_integer(record.get("query_id"), "shard query_id") for record in records if isinstance(record, Mapping)]
            if len(actual_ids) != len(records) or actual_ids != expected_ids:
                raise ValueError(f"shard query records differ for match {match_index}")
            return match, key, C.EXTERNAL_ROLE, False, None, records

    _SURFACE_CLASS = ExternalQuerySurface
    return ExternalQuerySurface


def e2bc_code_files() -> list[Path]:
    return sorted(set(C.PACKAGE.glob("e2b_*.py")) | set(C.PACKAGE.glob("e2c_*.py")) |
                  {C.PACKAGE / "e2bc_common.py", C.PACKAGE / "e2bc_run.py", C.PACKAGE / "e2_lock.py",
                   C.PACKAGE / "e2a_common.py", C.PACKAGE / "e2a_na1.py"} |
                  set((C.PACKAGE / "tests").glob("test_e2bc*.py")))


def code_hashes(paths) -> dict:
    return C.code_hashes([p for p in paths if Path(p).is_file()])


def digest_array(array) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def env_runtime() -> None:
    """Thread limits like score.py / qtest (set before torch/numpy work); GPU stays visible for V / neural q."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    try:
        import psutil

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
