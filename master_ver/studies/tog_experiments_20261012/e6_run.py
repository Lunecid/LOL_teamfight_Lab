"""E6: pre-registered confirmatory test of claim 1 under the kill-advantage label on a never-processed slice of the KR
15.18-15.22 raw matches (reports/tog_experiments_20261012/PRESPEC_E6_KO.md, approved 2026-10-14 at d11d63c, approval
line c4d4ee8).  The PRESPEC is binding; section numbers below refer to it.

Subcommands (every output under OUT = D:/LOL_Training_20261004/tog_e6_1, each file written once):
  run [--until STEP]   supervisor: runs the steps below in order, each as its own process, skipping completed ones
                       (STAGE_<step>.json); waits while e6_run.py differs from HEAD or OUT/HOLD exists; stops a step
                       over its time limit (history 6 h, others 2 h) or above 95.5 GB system commit (section 9).
  select               A1 environment census (ENV_CENSUS_before.json, start checks) and A2 slice selection
                       (section 2.2) -> E6_SAMPLE_MANIFEST.json (+ .sha256).
  build                A3 build_cache.mode_build then mode_check for kr_e6; zero build failures and passed: true.
  process --step S     A4-A6 GPU-masked steps: audit, bank, preparation, items, projection, history (E2a),
                       engagements and assemble (E2b), e2c (q rows, raw store, XY).  Frozen functions are called
                       directly with their outputs redirected to OUT (bind_outputs; no file is edited).
  score-v              A5 V scoring (e2b_labels.run_scoring, device cuda) in a fresh process whose environment is
                       exactly score_v_env() (the section 4 list plus USERNAME, 2026-10-14 운영 정정, Ruling 69); a
                       device check (one visible RTX 5080) is written to V_SCORING_DEVICE.json before any scoring; a
                       failure is BLOCKED (never another device).
  players, score, labels, rehearse, lock   A7-A12;  test   part B (B0-B2), written now and run only after the review.
Every stage records its run id, E6_STAGE, the modules its process and its workers loaded (worker capture: E6Executor
replaces concurrent.futures.ProcessPoolExecutor before any frozen import; its initializer writes a start marker and
registers a multiprocessing.util.Finalize(exitpriority=100) termination record; submit counts tasks) and reconciles
them (section 4 "module records"), then applies the acceptance rule (three provenance records, the 30 old-repository
builder modules, the 09-15 parent record).  STAGE_MODULES_<stage>.json, stage_runs/<stage>/<run>.json,
worker_modules/<stage>/<run>/<run>_<executor>_<pid>{.start,}.json (one subfolder per attempt; stages that ran before
Ruling 69 have them under <stage>/), STAGE_<stage>.json and STAGE_TIMES.jsonl hold them.  Ruling 69: an attempt that
died before the lock may be rerun; the lock reconciles each stage's final attempt and lists the others as superseded.

Launch (so that workers never re-run a main module):
  python -B -c "import sys; sys.path.insert(0, r'<WS>'); from studies.tog_experiments_20261012 import e6_run as m;
                raise SystemExit(m.main(sys.argv[1:]))" <subcommand> ...
No path whose name contains SEALED is opened for reading (guard_read; byte hashes only), and the slice's sealed SVI file
and PRED_final are refused always.  No player identifier is read except inside e3_players, and none is written.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
import concurrent.futures  # noqa: E402
import concurrent.futures.process  # noqa: E402
import contextlib  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import multiprocessing.util  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Callable, Iterable, Mapping  # noqa: E402
import uuid  # noqa: E402

import numpy as np  # noqa: E402

import studies.external_validation_20261009 as EXT  # noqa: E402  (package constants only; no side effects)
from studies.external_validation_20261009 import e2a_common as C  # noqa: E402
from studies.external_validation_20261009 import e2bc_common as B  # noqa: E402  (registers na1_16_13, as the frozen runs did)

VERSION = "tog_e6_run_v1_20261014"
EXPERIMENT = "E6"
DATASET = "kr_e6"
HASH_TAG = "tog_e6_20261014:"
SLICE_SIZE = 40000
PATCHES = ("15.18", "15.19", "15.20", "15.21", "15.22")
DETAIL_FILES, TIMELINE_FILES = 301744, 301724                         # section 2.1 (SAMPLE_MANIFEST 10-09 count)
CANDIDATES = {"15.18": 57153, "15.19": 73909, "15.20": 71706, "15.21": 62106, "15.22": 36794}
EXISTING_PER_PATCH = 8000
REMAINING = {"15.18": 49153, "15.19": 65909, "15.20": 63706, "15.21": 54106, "15.22": 28794}
QUOTAS = {"15.18": 7514, "15.19": 10075, "15.20": 9738, "15.21": 8271, "15.22": 4402}
EXISTING_DATASET = "kr_15_18_22"
MAX_WORKERS = 4                                                        # section 13; WS/CLAUDE.md
WORKER_EXIT_PRIORITY = 100                                             # section 4: Finalize exitpriority 100
EXCLUSION_MAX_SHARE = 0.01                                             # section 9: audit + bank exclusions
STAGE_LIMIT_S = {"history": 6 * 3600.0}
DEFAULT_LIMIT_S = 2 * 3600.0
COMMIT_LIMIT_BYTES = int(95.5 * 2**30)                                 # review_followup_20261011/memwatch.ps1 threshold
MIN_FREE_BYTES = 60 * 10**9                                            # section 9: D: free 60 GB
DEVICE_NAME = "RTX 5080"                                               # INFRA 5.1 main PC
PRESPEC_REL = "reports/tog_experiments_20261012/PRESPEC_E6_KO.md"
PRESPEC_APPROVAL_COMMIT = "c4d4ee8d8182f8a200ea8c2cec65ee9c1f3cdd45"   # approval line added to the d11d63c text
PRESPEC_APPROVED = "d11d63c51729d68867709c95bef841dd2db01ab9"
# The lock hashes the PRESPEC as last committed: the 2026-10-14 운영 정정 (Ruling 69) note followed the approval.
CODE_REL = "studies/tog_experiments_20261012/e6_run.py"
TEST_REL = "tests/test_tog_e6_run.py"

TRAINING = Path(r"D:\LOL_Training_20261004")
OUT_DEFAULT = TRAINING / "tog_e6_1"
OLD_REPO = Path(r"C:\Users\todtj\PycharmProjects\LOL_teamfight")
RAW_ROOT = Path(r"C:\Users\todtj\PycharmProjects\Lol_project\data\raw\matches\kr")
EX_ROOT = Path(r"D:\LOL_External_20261009")
REF_PATCH_INDEX_2025 = Path(r"D:\LOL_Project\cache\match_cache_fresh_v3_engage_status13_patch_index.json")
KA_POSTHOC_LOCK = TRAINING / "tog_ka_1" / "KA_POSTHOC_LOCK_r2.json"
PARENT_RUN = ROOT / "outputs" / "postkill_objective_delay_full" / "run.json"

P1 = "final__module__LGBM_S1_A1_M1_E1_C1_R1_D1__lightgbm__selected~lightgbm"
PT = "final__reference__PT_FLEX__PT_flex__selected~PT_flex"


def lgbm(s: int, d: int) -> str:
    return f"final__module__LGBM_S{s}_A0_M0_E1_C1_R1_D{d}__lightgbm__selected~lightgbm"


S10, S00, S11, S01 = lgbm(1, 0), lgbm(0, 0), lgbm(1, 1), lgbm(0, 1)
CELLS = (P1, S10, S00, S11, S01, PT)                                   # kill_label.py:74 order
S_CELLS = (S10, S00, S11, S01)

SCORE_V_ALLOW = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP", "USERPROFILE",
                 "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
                 "CONDA_PREFIX",
                 "USERNAME")       # 2026-10-14 운영 정정 (Ruling 69): torch's getpass.getuser() in the frozen V path
SCORE_V_FIXED = {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "KMP_DUPLICATE_LIB_OK": "TRUE",
                 "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "E6_STAGE": "score_v"}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

PROCESS_STEPS = ("audit", "bank", "preparation", "items", "projection", "history", "engagements", "assemble", "e2c")
STEP_ORDER = ("select", "build", "audit", "bank", "preparation", "items", "projection", "history", "engagements",
              "score_v", "assemble", "e2c", "players", "score", "labels", "rehearse")
RULE_TEXT = (
    "PRESPEC_E6 section 2.2 (approved 2026-10-14, d11d63c). Candidates: the external path's filter (sample.py "
    "_scan_details/_classify: info.queueId 420, info.mapId 11, info.gameMode CLASSIC, timeline/<match_id>.json present, "
    "patch = major.minor of info.gameVersion in 15.18-15.22) over the KR raw root; the existing external sample "
    "(SAMPLE_MANIFEST.json dataset kr_15_18_22, 40,000 matches) is removed. Per patch the remaining candidates are "
    "ranked by h6 = SHA256('tog_e6_20261014:' + match_id).hexdigest() (UTF-8, lowercase hex; ties by match_id) and the "
    "q_p smallest are taken, q_p = largest-remainder shares of 40,000 proportional to the remaining candidates "
    "(equal remainders in patch-name order). Matches lost later are not replaced. Selection reads match ids only.")


# ----------------------------------------------------------------------------- errors
class Blocked(RuntimeError):
    """A section 9 condition: the stage stops and reports BLOCKED (with an optional record)."""

    def __init__(self, message: str, record: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.record = dict(record) if record is not None else None


class SealedPathError(PermissionError):
    """A label-bearing path asked for reading (byte hashes are allowed; section 4, Ruling 52)."""


# ----------------------------------------------------------------------------- small io
def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def guard_read(path: Path | str, *, sealed_ok: bool = False) -> Path:
    """Refuse reading a path with SEALED in a component (unless sealed_ok) and, always, the slice's sealed SVI file
    and PRED_final with its parts (section 4: those two are never opened by E6 code)."""

    path = Path(path)
    parts = [part.upper() for part in path.parts]
    if any(part.startswith("EXTERNAL_LABELS_SEALED") or part.startswith("PRED_FINAL") for part in parts):
        raise SealedPathError(f"E6 code never opens this file (byte hash only): {path}")
    if not sealed_ok and any("SEALED" in part for part in parts):
        raise SealedPathError(f"refusing to read a path whose name contains SEALED: {path}")
    return path


def sha256_file(path: Path | str) -> str:
    """SHA-256 of a file's bytes (allowed for every file, sealed ones included: Ruling 52)."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(guard_read(path).read_text(encoding="utf-8-sig"))


def load_npz(path: Path | str, names: Iterable[str] | None = None, *, sealed_ok: bool = False) -> dict[str, np.ndarray]:
    with np.load(guard_read(path, sealed_ok=sealed_ok), allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in (list(names) if names is not None else data.files)}


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [jsonable(v) for v in items]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("ascii", "replace")
    return value


def json_bytes(payload: Any, list_key: str | None = None) -> bytes:
    """indent=1 sorted JSON (LF only, non-finite floats as null); one top-level list may be written one record per
    line (list_key)."""

    obj = jsonable(payload)
    if list_key is None or list_key not in obj:
        return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    head = {k: v for k, v in obj.items() if k != list_key}
    text = json.dumps(head, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False)
    rows = ",\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
                      for r in obj[list_key])
    body = "[\n" + rows + "\n]" if rows else "[]"
    text = text[:-1].rstrip() + (",\n" if head else "\n") + f" {json.dumps(list_key)}: " + body + "\n}\n"
    return text.encode("utf-8")


def _place_once(path: Path, data: bytes) -> str:
    """Write bytes through a pending file; an existing target is refused (written once)."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; it is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.pending")
    with open(pending, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        pending.unlink()
        raise FileExistsError(f"{path} exists; it is written once")
    os.rename(pending, path)
    return hashlib.sha256(data).hexdigest()


def write_json_once(path: Path | str, payload: Any, list_key: str | None = None) -> str:
    return _place_once(Path(path), json_bytes(payload, list_key))


def write_bytes_once(path: Path | str, data: bytes) -> str:
    return _place_once(Path(path), data)


def write_npz_once(path: Path | str, arrays: Mapping[str, np.ndarray]) -> str:
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **{k: np.asarray(v) for k, v in arrays.items()})
    return _place_once(Path(path), buffer.getvalue())


def next_free(path: Path) -> Path:
    """path, or path_r1, path_r2, ... the first that does not exist (earlier versions stay)."""

    path = Path(path)
    if not path.exists():
        return path
    k = 1
    while path.with_name(f"{path.stem}_r{k}{path.suffix}").exists():
        k += 1
    return path.with_name(f"{path.stem}_r{k}{path.suffix}")


def append_line(path: Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")


def peak_memory_bytes() -> int | None:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", 0) or info.rss)
    except Exception:  # noqa: BLE001
        return None


def commit_bytes() -> int | None:
    """System commit charge (GetPerformanceInfo CommitTotal * PageSize; ka_posthoc.commit_bytes)."""

    try:
        import ctypes
        from ctypes import wintypes

        class PerformanceInformation(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                        ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                        ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                        ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                        ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                        ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD),
                        ("ThreadCount", wintypes.DWORD)]

        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        if not ctypes.WinDLL("psapi").GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None
        return int(info.CommitTotal) * int(info.PageSize)
    except Exception:  # noqa: BLE001
        return None


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)


def _qualname(fn: Any) -> str | None:
    if fn is None:
        return None
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"


# ----------------------------------------------------------------------------- paths
@dataclass(frozen=True)
class Paths:
    out: Path = OUT_DEFAULT
    repo: Path = ROOT
    raw: Path = RAW_ROOT
    external_data: Path = EX_ROOT
    external_reports: Path = ROOT / "reports" / "external_validation_20261009"
    training: Path = TRAINING
    old_repo: Path = OLD_REPO
    ref_patch_index: Path = REF_PATCH_INDEX_2025
    ka_posthoc_lock: Path = KA_POSTHOC_LOCK
    parent_run: Path = PARENT_RUN
    openings: Path = ROOT / "reports" / "tog_experiments_20261012" / "OPENINGS.json"
    exposure: Path = ROOT / "reports" / "tog_experiments_20261012" / "EXPOSURE_KO.md"
    prespec_rel: str = PRESPEC_REL
    prespec_commit: str | None = None                                  # None: the last commit of the PRESPEC path

    @property
    def sample_manifest(self) -> Path:
        return self.external_reports / "SAMPLE_MANIFEST.json"

    @property
    def e6_manifest(self) -> Path:
        return self.out / "E6_SAMPLE_MANIFEST.json"

    @property
    def e6_manifest_sha(self) -> Path:
        return self.out / "E6_SAMPLE_MANIFEST.sha256"

    @property
    def patch_index(self) -> Path:
        return self.out / "match_cache_kr_e6_patch_index.json"

    @property
    def cache(self) -> Path:
        return self.out / "cache_kr_e6"

    def ext(self, stage: str) -> Path:
        return self.out / "ext" / stage / DATASET

    @property
    def q_ka(self) -> Path:
        return self.training / "q_ka_1"

    @property
    def q_run(self) -> Path:
        return self.training / "q_run_1"

    def ka_cell(self, key: str) -> Path:
        return self.q_ka / "cells" / key


# ----------------------------------------------------------------------------- registration and output binding (section 4)
_ORIGINAL = {"stage_dir": C.stage_dir, "labels_dir": B.labels_dir, "q_inputs_dir": B.q_inputs_dir,
             "scores_dir": B.scores_dir}
_BOUND: dict[str, Any] = {}


def register_dataset() -> None:
    """Add kr_e6 to the external registries (additive and idempotent, as e2a_na1.register)."""

    if DATASET not in C.DATASETS:
        C.DATASETS = C.DATASETS + (DATASET,)
    C.EXPECTED_PATCHES[DATASET] = PATCHES
    EXT.RAW[DATASET] = RAW_ROOT
    out = Path(_BOUND.get("out") or os.environ.get("E6_OUT") or OUT_DEFAULT)
    for registry in (EXT.CACHE_DIR, C.CACHE):
        registry.setdefault(DATASET, out / "cache_kr_e6")
    for registry in (EXT.PATCH_INDEX, C.PATCH_INDEX):
        registry.setdefault(DATASET, out / "match_cache_kr_e6_patch_index.json")


def _e6_stage_dir(dataset: str, stage: str) -> Path:
    if dataset == DATASET:
        if stage not in C.STAGES:
            raise ValueError(f"unknown dataset/stage {dataset}/{stage}")
        return Path(_BOUND["out"]) / "ext" / stage / DATASET
    return _ORIGINAL["stage_dir"](dataset, stage)


def _e6_labels_dir(dataset: str) -> Path:
    return Path(_BOUND["out"]) / "ext" / "labels" / DATASET if dataset == DATASET else _ORIGINAL["labels_dir"](dataset)


def _e6_q_inputs_dir(dataset: str) -> Path:
    return Path(_BOUND["out"]) / "ext" / "q_inputs" / DATASET if dataset == DATASET else _ORIGINAL["q_inputs_dir"](dataset)


def _e6_scores_dir(dataset: str) -> Path:
    return Path(_BOUND["out"]) / "ext" / "scores" / DATASET if dataset == DATASET else _ORIGINAL["scores_dir"](dataset)


def bind_outputs(out: Path | str) -> dict[str, Any]:
    """Point every output of the external package at OUT by rebinding module attributes (section 4 "경로"); no file
    is edited.  kr_e6 stage folders are OUT/ext/<stage>/kr_e6; the other datasets keep the frozen layout."""

    out = Path(out)
    register_dataset()
    _BOUND["out"] = str(out)
    EXT.REPORTS, EXT.LOGS, EXT.WORK, EXT.RUNTIME_REPO = out, out / "logs", out / "work", out / "runtime_repo"
    EXT.SAMPLE_MANIFEST, EXT.SAMPLE_MANIFEST_SHA = out / "E6_SAMPLE_MANIFEST.json", out / "E6_SAMPLE_MANIFEST.sha256"
    EXT.CACHE_DIR[DATASET] = C.CACHE[DATASET] = out / "cache_kr_e6"
    EXT.PATCH_INDEX[DATASET] = C.PATCH_INDEX[DATASET] = out / "match_cache_kr_e6_patch_index.json"
    C.LOGS = out / "logs"
    C.stage_dir = _e6_stage_dir
    B.labels_dir, B.q_inputs_dir, B.scores_dir = _e6_labels_dir, _e6_q_inputs_dir, _e6_scores_dir
    B.STAGE_TIMES = out / "E2bc_STAGE_TIMES.jsonl"
    return {"out": str(out), "rebindings": rebinding_list(out)}


def rebinding_list(out: Path | str) -> list[dict[str, str]]:
    """The module-attribute rebindings of bind_outputs and install_executor (bound in the lock like FROZEN_HOOKS)."""

    out = Path(out)
    pkg, a, b = "studies.external_validation_20261009", "studies.external_validation_20261009.e2a_common", \
        "studies.external_validation_20261009.e2bc_common"
    return [
        {"module": pkg, "name": "REPORTS", "value": str(out)},
        {"module": pkg, "name": "LOGS", "value": str(out / "logs")},
        {"module": pkg, "name": "WORK", "value": str(out / "work")},
        {"module": pkg, "name": "RUNTIME_REPO", "value": str(out / "runtime_repo"),
         "note": "before build_cache import (BUILDER_ENV LOL_OUTPUT_ROOT)"},
        {"module": pkg, "name": "SAMPLE_MANIFEST", "value": str(out / "E6_SAMPLE_MANIFEST.json")},
        {"module": pkg, "name": "SAMPLE_MANIFEST_SHA", "value": str(out / "E6_SAMPLE_MANIFEST.sha256")},
        {"module": pkg, "name": "RAW/CACHE_DIR/PATCH_INDEX[kr_e6]", "value": f"{RAW_ROOT} | {out / 'cache_kr_e6'} | "
                                                                          f"{out / 'match_cache_kr_e6_patch_index.json'}"},
        {"module": a, "name": "DATASETS/EXPECTED_PATCHES/CACHE/PATCH_INDEX[kr_e6]", "value": "kr_e6 added (additive)"},
        {"module": a, "name": "LOGS", "value": str(out / "logs")},
        {"module": a, "name": "stage_dir", "value": f"kr_e6 -> {out / 'ext'}/<stage>/kr_e6; other datasets unchanged"},
        {"module": b, "name": "labels_dir", "value": f"kr_e6 -> {out / 'ext' / 'labels' / DATASET}"},
        {"module": b, "name": "q_inputs_dir", "value": f"kr_e6 -> {out / 'ext' / 'q_inputs' / DATASET}"},
        {"module": b, "name": "scores_dir", "value": f"kr_e6 -> {out / 'ext' / 'scores' / DATASET}"},
        {"module": b, "name": "STAGE_TIMES", "value": str(out / "E2bc_STAGE_TIMES.jsonl")},
        {"module": "concurrent.futures", "name": "ProcessPoolExecutor", "value": f"{__name__}.E6Executor"},
        {"module": "concurrent.futures.process", "name": "ProcessPoolExecutor", "value": f"{__name__}.E6Executor"},
    ]


# ----------------------------------------------------------------------------- worker capture (section 4 "module records")
_ORIGINAL_PPE = getattr(concurrent.futures.process, "_e6_original_ppe", concurrent.futures.process.ProcessPoolExecutor)
_STAGE: "StageContext | None" = None
_WORKER: dict[str, Any] | None = None


class E6Executor(_ORIGINAL_PPE):
    """ProcessPoolExecutor that wraps the worker initializer (start marker, Finalize termination record, then the
    original initializer with its original arguments) and every submitted call (task count, result unchanged)."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
        ctx = _STAGE
        if ctx is None:
            raise RuntimeError("an E6 worker pool needs an active E6 stage (begin_stage)")
        if max_workers is None or int(max_workers) < 1 or int(max_workers) > MAX_WORKERS:
            raise Blocked(f"at most {MAX_WORKERS} CPU worker processes per pool (asked {max_workers})")
        with ctx.lock:
            number = len(ctx.executors)
            ctx.executors[number] = {"executor_no": number, "max_workers": int(max_workers), "submitted": 0,
                                     "created_utc": utc(), "shutdown_utc": None,
                                     "original_initializer": _qualname(initializer)}
        self._e6_ctx, self._e6_no = ctx, number
        super().__init__(max_workers=max_workers, mp_context=mp_context, initializer=_worker_initializer,
                         initargs=(ctx.stage, ctx.run_id, number, str(ctx.records_dir), initializer, tuple(initargs)),
                         **kwargs)

    def submit(self, fn, /, *args, **kwargs):
        future = super().submit(_E6Task(fn), *args, **kwargs)
        with self._e6_ctx.lock:
            self._e6_ctx.executors[self._e6_no]["submitted"] += 1
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        with self._e6_ctx.lock:
            entry = self._e6_ctx.executors[self._e6_no]
            if entry["shutdown_utc"] is None:
                entry["shutdown_utc"] = utc()


def install_executor() -> dict[str, str]:
    """Replace both ProcessPoolExecutor names before any frozen module is imported (section 4)."""

    concurrent.futures.process._e6_original_ppe = _ORIGINAL_PPE
    concurrent.futures.process.ProcessPoolExecutor = E6Executor
    concurrent.futures.ProcessPoolExecutor = E6Executor
    return {"concurrent.futures.ProcessPoolExecutor": _qualname(E6Executor),
            "concurrent.futures.process.ProcessPoolExecutor": _qualname(E6Executor)}


def restore_executor() -> None:
    concurrent.futures.process.ProcessPoolExecutor = _ORIGINAL_PPE
    concurrent.futures.ProcessPoolExecutor = _ORIGINAL_PPE


def check_frozen_executors(modules: Iterable[Any]) -> dict[str, bool]:
    """Every frozen module that took the name at its top must hold the E6 subclass (else BLOCKED)."""

    seen = {getattr(m, "__name__", repr(m)): getattr(m, "ProcessPoolExecutor", None) is E6Executor for m in modules}
    bad = sorted(name for name, ok in seen.items() if not ok)
    if bad:
        raise Blocked(f"ProcessPoolExecutor is not the E6 subclass in {bad}")
    return seen


class _E6Task:
    """A submitted call: counts one task in this worker and returns the original result unchanged."""

    __slots__ = ("fn",)

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def __call__(self, *args, **kwargs):
        if _WORKER is None:
            raise RuntimeError("an E6 task ran outside an E6 worker")
        _WORKER["tasks"] += 1
        return self.fn(*args, **kwargs)


def _worker_path(records_dir: str | Path, run_id: str, executor_no: int, pid: int, *, start: bool) -> Path:
    return Path(records_dir) / f"{run_id}_{int(executor_no)}_{int(pid)}{'.start' if start else ''}.json"


def _worker_initializer(stage: str, run_id: str, executor_no: int, records_dir: str, original: Callable | None,
                        original_args: tuple) -> None:
    """E6 initializer: identity, start marker, Finalize(exitpriority=100) termination record, then the original
    initializer with its original arguments."""

    global _WORKER
    parent = multiprocessing.parent_process()
    _WORKER = {"stage": stage, "run_id": run_id, "executor_no": int(executor_no), "records_dir": str(records_dir),
               "pid": os.getpid(), "os_ppid": os.getppid(), "mp_parent_pid": parent.pid if parent is not None else None,
               "process_name": multiprocessing.current_process().name, "env_stage": os.environ.get("E6_STAGE"),
               "started_utc": utc(), "tasks": 0, "original_initializer": _qualname(original)}
    marker = {k: v for k, v in _WORKER.items() if k not in ("tasks", "records_dir")}
    write_json_once(_worker_path(records_dir, run_id, executor_no, os.getpid(), start=True), marker)
    multiprocessing.util.Finalize(None, _write_worker_record, exitpriority=WORKER_EXIT_PRIORITY)
    if original is not None:
        original(*original_args)


def _write_worker_record() -> None:
    """Termination record: identity, tasks executed, loaded workspace and old-repository files with SHA-256."""

    w = _WORKER
    if w is None:
        return
    record = {k: v for k, v in w.items() if k != "records_dir"}
    record.update(finished_utc=utc(), env_stage_at_exit=os.environ.get("E6_STAGE"), peak_memory_bytes=peak_memory_bytes(),
                  modules=loaded_files())
    write_json_once(_worker_path(w["records_dir"], w["run_id"], w["executor_no"], w["pid"], start=False), record)


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _under(path: str | Path, root: str | Path) -> bool:
    p, r = _norm(path), _norm(root)
    return p.startswith(r + os.sep)


def loaded_files(modules: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Absolute path -> SHA-256 of every loaded module file under the workspace or the old repository."""

    out: dict[str, str] = {}
    for module in list((modules if modules is not None else sys.modules).values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        try:
            path = str(Path(name).resolve())
        except OSError:
            continue
        if (_under(path, ROOT) or _under(path, OLD_REPO)) and os.path.isfile(path):
            out[path] = sha256_file(path)
    return dict(sorted(out.items()))


def worker_record_files(root: Path | str) -> tuple[list[Path], list[Path]]:
    """(start markers, termination records) under root, recursively (section 8.1 A12, 8.3 B0)."""

    root = Path(root)
    if not root.is_dir():
        return [], []
    files = sorted(p for p in root.rglob("*.json") if p.is_file())
    return [p for p in files if p.name.endswith(".start.json")], [p for p in files if not p.name.endswith(".start.json")]


def record_run_id(path: Path | str) -> str:
    """The run nonce in a worker-record file name <run>_<executor>_<pid>[.start].json (run ids are uuid hex)."""

    return Path(path).name.split("_", 1)[0]


def reconcile(root: Path | str, runs: Mapping[str, Mapping[str, Any]], run_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Start markers against termination records by (stage, run, executor, pid); each termination record valid with
    the stage's pid as parent; per executor the workers' task total equals the stage's submissions (section 4)."""

    wanted = set(run_ids) if run_ids is not None else None
    root = Path(root)
    starts, ends = worker_record_files(root)
    problems: list[str] = []
    start_map: dict[tuple, tuple[Path, dict]] = {}
    end_map: dict[tuple, tuple[Path, dict]] = {}
    for paths_, target, kind in ((starts, start_map, "start marker"), (ends, end_map, "termination record")):
        for path in paths_:
            if wanted is not None and record_run_id(path) not in wanted:
                continue                                                # another attempt (Ruling 69: superseded ones)
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                key = (str(doc["stage"]), str(doc["run_id"]), int(doc["executor_no"]), int(doc["pid"]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                problems.append(f"invalid {kind} {path}: {type(error).__name__}")
                continue
            expected = _worker_path(path.parent, key[1], key[2], key[3], start=kind == "start marker").name
            parts = path.relative_to(root).parts
            folder_ok = parts[0] == key[0] and (len(parts) == 2 or (len(parts) == 3 and parts[1] == key[1]))
            if path.name != expected or not folder_ok:
                problems.append(f"invalid {kind} {path}: name or folder differs from its content")
                continue
            target[key] = (path, doc)
    for key in sorted(set(start_map) - set(end_map)):
        problems.append(f"no termination record for start marker {start_map[key][0]}")
    for key in sorted(set(end_map) - set(start_map)):
        problems.append(f"termination record without a start marker {end_map[key][0]}")
    tasks: Counter = Counter()
    by_run: dict[str, dict[str, str]] = {}                             # module versions agree within a run (stage attempt)
    peak = 0
    norm_runs = {str(r): {"stage": str(v["stage"]), "pid": int(v["pid"]),
                          "executors": {int(k): int(n) for k, n in (v.get("executors") or {}).items()}}
                 for r, v in runs.items()}
    for key, (path, doc) in sorted(end_map.items()):
        stage, run, number, _pid = key
        reasons = []
        if doc.get("env_stage") != stage:
            reasons.append(f"inherited E6_STAGE {doc.get('env_stage')!r} differs from the initializer stage {stage!r}")
        if not isinstance(doc.get("tasks"), int) or doc["tasks"] < 0:
            reasons.append("tasks is not a count")
        if not isinstance(doc.get("modules"), dict):
            reasons.append("modules missing")
        info = norm_runs.get(run)
        if info is None:
            problems.append(f"no stage run record for run {run} ({path})")
            continue
        if info["stage"] != stage:
            reasons.append(f"run {run} belongs to stage {info['stage']}")
        if reasons:
            problems.append(f"invalid termination record {path}: {'; '.join(reasons)}")
            continue
        if doc.get("mp_parent_pid") != info["pid"] or doc.get("os_ppid") != info["pid"]:
            problems.append(f"parent pid {doc.get('mp_parent_pid')}/{doc.get('os_ppid')} differs from the stage pid "
                            f"{info['pid']} in {path}")
            continue
        tasks[f"{run}:{number}"] += int(doc["tasks"])
        peak = max(peak, int(doc.get("peak_memory_bytes") or 0))
        seen = by_run.setdefault(run, {})
        for name, sha in doc["modules"].items():
            if seen.setdefault(name, sha) != sha:
                problems.append(f"workers of run {run} loaded two versions of {name}")
    for run, info in sorted(norm_runs.items()):
        if wanted is not None and run not in wanted:
            continue
        for number, submitted in sorted(info["executors"].items()):
            done = tasks.get(f"{run}:{number}", 0)
            if done != submitted:
                problems.append(f"executor {run}:{number}: worker task total {done} differs from the submissions {submitted}")
    for label in sorted(tasks):
        run, number = label.split(":")
        if int(number) not in norm_runs.get(run, {}).get("executors", {}):
            problems.append(f"executor {label} has worker records but no submission count")
    return {"start_markers": len(start_map), "termination_records": len(end_map), "tasks": dict(sorted(tasks.items())),
            "problems": problems, "passed": not problems,
            "modules": dict(sorted(next(iter(by_run.values())).items())) if len(by_run) == 1 else {},
            "modules_by_run": {run: dict(sorted(m.items())) for run, m in sorted(by_run.items())},
            "worker_peak_memory_bytes": peak or None,
            "rule": ("start markers and termination records (OUT/worker_modules, recursive) match by (stage, run, executor, "
                     "pid); every termination record is readable, its inherited E6_STAGE equals the initializer stage and "
                     "its parent pid is the stage process; per executor the task total equals the stage's submissions")}


# ----------------------------------------------------------------------------- acceptance rule (section 4)
@dataclass(frozen=True)
class Refs:
    ws: dict[str, str]                       # workspace-relative posix path -> SHA-256 (three records)
    old30: dict[str, str]                    # E1_REPORT code.old_repo_builder_modules: absolute path -> SHA-256
    parent: dict[str, str]                   # 09-15 parent run.json source_hashes: absolute path -> SHA-256
    sizes: dict[str, int] = field(default_factory=dict)
    records: dict[str, str] = field(default_factory=dict)   # record file -> SHA-256


def load_refs(paths: Paths) -> Refs:
    """The three provenance records (section 4 first paragraph) and the 09-15 parent run record."""

    rep = paths.external_reports
    lock = read_json(rep / "EXTERNAL_SCORE_LOCK.json")
    e2a = read_json(rep / "E2a_REPORT.json")
    e1 = read_json(rep / "E1_REPORT.json")
    blocks = {"EXTERNAL_SCORE_LOCK.code.external": lock["code"]["external"],
              "EXTERNAL_SCORE_LOCK.code.frozen_imported": lock["code"]["frozen_imported"],
              "E2a_REPORT.code.package": e2a["code"]["package"],
              "E2a_REPORT.code.frozen_dependencies": e2a["code"]["frozen_dependencies"]}
    ws: dict[str, str] = {}
    for label, block in blocks.items():
        for rel, sha in block.items():
            if ws.setdefault(rel, sha) != sha:
                raise Blocked(f"the provenance records disagree on {rel}")
    old30 = {path: entry["sha256"] for path, entry in e1["code"]["old_repo_builder_modules"].items()}
    parent = dict(read_json(paths.parent_run)["source_hashes"])
    sizes = {**{label: len(block) for label, block in blocks.items()}, "E1_REPORT.code.old_repo_builder_modules": len(old30)}
    records = {str(p): sha256_file(p) for p in (rep / "EXTERNAL_SCORE_LOCK.json", rep / "E2a_REPORT.json",
                                                rep / "E1_REPORT.json", paths.parent_run)}
    return Refs(ws=ws, old30=old30, parent=parent, sizes=sizes, records=records)


def acceptance(files: Mapping[str, str], refs: Refs) -> dict[str, Any]:
    """Section 4 acceptance rule for loaded files: a workspace file named in the three records has their hash; an
    old-repository file is one of the 30 builder modules with their hash, or else in the 09-15 parent record with its
    hash; anything else under the old repository is BLOCKED."""

    old30 = {_norm(k): v for k, v in refs.old30.items()}
    parent = {_norm(k): v for k, v in refs.parent.items()}
    out: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for path, sha in sorted(files.items()):
        if _under(path, ROOT):
            rel = Path(path).resolve().relative_to(ROOT).as_posix() if Path(path).exists() else \
                Path(os.path.relpath(path, ROOT)).as_posix()
            if rel in refs.ws:
                rule, ok = "three_records", refs.ws[rel] == sha
            else:
                rule, ok = "workspace_not_in_records", True
        elif _under(path, OLD_REPO):
            key = _norm(path)
            if key in old30:
                rule, ok = "old_repo_30", old30[key] == sha
            elif key in parent:
                rule, ok = "parent_record", parent[key] == sha
            else:
                rule, ok = "old_repo_unbound", False
        else:
            rule, ok = "outside", True
        out[path] = {"sha256": sha, "rule": rule, "ok": ok}
        if not ok:
            problems.append(f"{path}: {rule} rule fails (sha256 {sha})")
    return {"files": out, "problems": problems, "passed": not problems,
            "counts": dict(Counter(v["rule"] for v in out.values())),
            "rule": ("workspace files named in EXTERNAL_SCORE_LOCK code, E2a_REPORT code or E1_REPORT "
                     "old_repo_builder_modules carry the recorded SHA-256; old-repository files are among the 30 "
                     "builder modules with the same SHA-256, otherwise in outputs/postkill_objective_delay_full/run.json "
                     "source_hashes with the same SHA-256; any other old-repository file is BLOCKED")}


# ----------------------------------------------------------------------------- stages
class StageLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{utc()} {message}"
        with open(self.path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
        try:
            print(line, flush=True)
        except Exception:  # noqa: BLE001  (a detached process may have no console)
            pass


@dataclass
class StageContext:
    paths: Paths
    stage: str
    run_id: str
    pid: int
    started_utc: str
    tick: float
    cpu0: float
    log: StageLog
    env_at_start: dict[str, str]
    executors: dict[int, dict[str, Any]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    superseded_outputs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def records_dir(self) -> Path:
        """A fresh worker_modules subfolder per attempt (2026-10-14 운영 정정, Ruling 69)."""

        return self.paths.out / "worker_modules" / self.stage / self.run_id


# Outputs of the E6-owned stages (frozen stages resume through their own validated receipts): a rerun after an attempt
# that died before its stage record moves these aside (kept, not read) and starts clean (Ruling 69).
STAGE_OUTPUTS = {"players": ("PLAYERS_E6.npz", "PLAYERS_E6.json", "PLAYERS_E6_UNAVAILABLE.json"),
                 "score": ("scores",),
                 "labels": ("KA_GATE.json", "KA_LABELS_E6_SEALED.npz", "KA_LABELS_E6_MANIFEST.json"),
                 "rehearse": ("E6_QDEV_REHEARSAL.json", "KA_LABELS_E6_SYNTH.npz", "E6_SYNTH_REHEARSAL.json")}


def supersede_leftovers(ctx: StageContext) -> list[dict[str, Any]]:
    """Move the outputs an earlier, crashed attempt of this stage left behind to OUT/superseded/<stage>/<run>/ (file
    bytes hashed, never opened for reading; nothing deleted)."""

    moved = []
    for name in STAGE_OUTPUTS.get(ctx.stage, ()):
        source = ctx.paths.out / name
        if not source.exists():
            continue
        target = ctx.paths.out / "superseded" / ctx.stage / ctx.run_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
        hashes = {str(p.relative_to(ctx.paths.out)): sha256_file(p) for p in files}
        os.rename(source, target)
        entry = {"from": str(source), "to": str(target), "files": len(files)}
        if source.suffix:
            entry["sha256"] = next(iter(hashes.values()), None)
        else:
            entry["sha256_by_file"] = hashes
        moved.append(entry)
        ctx.log(f"[{ctx.stage}] moved an earlier attempt's {name} to {target} (Ruling 69)")
    ctx.superseded_outputs.extend(moved)
    return moved


def begin_stage(paths: Paths, stage: str) -> StageContext:
    """The stage process: E6_STAGE in its own environment, a random run id, the stage log (section 4)."""

    global _STAGE
    if _STAGE is not None:
        raise RuntimeError(f"stage {_STAGE.stage} is still active in this process")
    os.environ["E6_STAGE"] = stage
    ctx = StageContext(paths=paths, stage=stage, run_id=uuid.uuid4().hex, pid=os.getpid(), started_utc=utc(),
                       tick=time.perf_counter(), cpu0=time.process_time(),
                       log=StageLog(paths.out / "logs" / f"{stage}.log"), env_at_start=dict(os.environ))
    _STAGE = ctx
    ctx.log(f"[{stage}] start run_id={ctx.run_id} pid={ctx.pid}")
    return ctx


def stage_record_path(paths: Paths, stage: str) -> Path:
    return paths.out / f"STAGE_{stage}.json"


def stage_done(paths: Paths, stage: str) -> bool:
    path = stage_record_path(paths, stage)
    return path.is_file() and read_json(path).get("status") == "completed"


def code_identity(paths: Paths) -> dict[str, Any]:
    status = _git(paths.repo, "status", "--porcelain", "--untracked-files=all", "--", CODE_REL)
    return {"path": CODE_REL, "sha256": sha256_file(paths.repo / CODE_REL),
            "head": _git(paths.repo, "rev-parse", "HEAD").stdout.decode("ascii", "replace").strip(),
            "last_commit": _git(paths.repo, "log", "-1", "--format=%H", "--", CODE_REL).stdout.decode("ascii", "replace").strip(),
            "status_porcelain": status.stdout.decode("utf-8", "replace"), "status_returncode": status.returncode}


def code_committed(paths: Paths) -> tuple[bool, dict[str, Any]]:
    ident = code_identity(paths)
    return ident["status_returncode"] == 0 and not ident["status_porcelain"].strip() and bool(ident["last_commit"]), ident


def end_stage(ctx: StageContext, status: str, error: str | None, result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Module record, reconciliation and acceptance; the run record always; STAGE_MODULES and STAGE records only for a
    completed stage whose checks pass."""

    global _STAGE
    paths, stage = ctx.paths, ctx.stage
    try:
        modules = loaded_files()
        runs = {ctx.run_id: {"stage": stage, "pid": ctx.pid,
                             "executors": {no: e["submitted"] for no, e in ctx.executors.items()}}}
        rec = reconcile(paths.out / "worker_modules", runs, run_ids={ctx.run_id})
        merged = dict(modules)
        conflicts = [name for name, sha in rec["modules"].items() if merged.setdefault(name, sha) != sha]
        acc = acceptance(merged, load_refs(paths))
        problems = list(rec["problems"]) + [f"stage and workers loaded two versions of {n}" for n in conflicts] + list(acc["problems"])
        if status == "completed" and problems:
            status, error = "blocked", f"module record checks failed: {problems[:5]}"
        wall = time.perf_counter() - ctx.tick
        run_record = {
            "version": VERSION, "experiment": EXPERIMENT, "stage": stage, "run_id": ctx.run_id, "pid": ctx.pid,
            "status": status, "error": error, "started_utc": ctx.started_utc, "finished_utc": utc(),
            "wall_seconds": round(wall, 3), "cpu_seconds": round(time.process_time() - ctx.cpu0, 3),
            "peak_memory_bytes": peak_memory_bytes(), "worker_peak_memory_bytes": rec["worker_peak_memory_bytes"],
            "executors": {str(no): e for no, e in ctx.executors.items()},
            "env_stage": os.environ.get("E6_STAGE"), "stage_modules": modules,
            "reconciliation": {k: v for k, v in rec.items() if k not in ("modules", "modules_by_run")}, "worker_modules": rec["modules"],
            "acceptance": {k: v for k, v in acc.items() if k != "files"}, "code": code_identity(paths),
            "records_dir": str(ctx.records_dir), "superseded_outputs": ctx.superseded_outputs}
        run_path = paths.out / "stage_runs" / stage / f"{ctx.run_id}.json"
        run_sha = write_json_once(run_path, run_record)
        if status == "blocked":
            write_json_once(paths.out / f"{stage.upper()}_BLOCKED_{stamp()}.json",
                            {"version": VERSION, "experiment": EXPERIMENT, "stage": stage, "status": "BLOCKED",
                             "utc": utc(), "reason": error, "run_record": str(run_path),
                             "details": (result or {}).get("blocked_details")})
        times = {"stage": stage, "run_id": ctx.run_id, "status": status, "started_utc": ctx.started_utc,
                 "finished_utc": run_record["finished_utc"], "wall_seconds": run_record["wall_seconds"],
                 "peak_memory_bytes": run_record["peak_memory_bytes"],
                 "worker_peak_memory_bytes": run_record["worker_peak_memory_bytes"], "error": error}
        append_line(paths.out / "STAGE_TIMES.jsonl", times)
        if status == "completed":
            modules_doc = {"version": VERSION, "experiment": EXPERIMENT, "stage": stage, "run_id": ctx.run_id,
                           "pid": ctx.pid, "stage_process_modules": modules, "worker_modules": rec["modules"],
                           "merged_modules": dict(sorted(merged.items())),
                           "executors": {str(no): e for no, e in ctx.executors.items()},
                           "reconciliation": {k: v for k, v in rec.items() if k not in ("modules", "modules_by_run")},
                           "acceptance": acc, "run_record": {"path": str(run_path), "sha256": run_sha}, "utc": utc()}
            modules_sha = write_json_once(paths.out / f"STAGE_MODULES_{stage}.json", modules_doc)
            body = dict(result or {})
            record = {"version": VERSION, "experiment": EXPERIMENT, "stage": stage, "status": "completed",
                      "run_id": ctx.run_id, "pid": ctx.pid, "started_utc": ctx.started_utc,
                      "finished_utc": run_record["finished_utc"], "wall_seconds": run_record["wall_seconds"],
                      "cpu_seconds": run_record["cpu_seconds"], "peak_memory_bytes": run_record["peak_memory_bytes"],
                      "worker_peak_memory_bytes": run_record["worker_peak_memory_bytes"],
                      "workers": {str(no): {"max_workers": e["max_workers"], "submitted": e["submitted"]}
                                  for no, e in ctx.executors.items()},
                      "env": {k: ctx.env_at_start.get(k) for k in ("E6_STAGE", "E6_OUT", "CUDA_VISIBLE_DEVICES",
                                                                   "PYTHONDONTWRITEBYTECODE", *THREAD_VARIABLES)},
                      "counts": body.get("counts", {}), "details": body.get("details", {}),
                      "outputs": {label: {"path": str(p), "sha256": sha256_file(p), "bytes": Path(p).stat().st_size}
                                  for label, p in (body.get("outputs") or {}).items()},
                      "run": run_block(body, paths), "code": run_record["code"],
                      "modules_record": {"path": str(paths.out / f"STAGE_MODULES_{stage}.json"), "sha256": modules_sha},
                      "run_record": {"path": str(run_path), "sha256": run_sha},
                      "superseded_outputs": ctx.superseded_outputs}
            write_json_once(stage_record_path(paths, stage), record)
        ctx.log(f"[{stage}] {status} in {wall:.1f} s (run {ctx.run_id}){'' if not error else ': ' + error}")
        return {"status": status, "error": error, "run_record": str(run_path)}
    finally:
        _STAGE = None


def run_block(body: Mapping[str, Any], paths: Paths) -> dict[str, Any]:
    """Seed, row and match counts, input and code SHA-256 (common.run_record fields)."""

    inputs = body.get("inputs") or {}
    return {"seed": body.get("seed"), "rows": body.get("rows"), "matches": body.get("matches"),
            "input_sha256": {label: sha256_file(p) for label, p in inputs.items()},
            "code_sha256": {CODE_REL: sha256_file(paths.repo / CODE_REL)}, "utc": utc(),
            "python": platform.python_version(), "numpy": np.__version__}


def run_stage(paths: Paths, stage: str, body: Callable[[StageContext], Mapping[str, Any] | None], *,
              require_committed: bool | None = None) -> int:
    """Run one stage once: skipped when STAGE_<stage>.json says completed; 0 completed, 2 BLOCKED, 1 failed."""

    if stage_done(paths, stage):
        print(f"[{stage}] already completed ({stage_record_path(paths, stage)})", flush=True)
        return 0
    if require_committed is None:
        require_committed = Path(paths.out) == OUT_DEFAULT
    paths.out.mkdir(parents=True, exist_ok=True)
    ctx = begin_stage(paths, stage)
    status, error, result = "failed", None, {}
    try:
        if require_committed:
            ok, ident = code_committed(paths)
            if not ok:
                raise Blocked(f"{CODE_REL} must be committed and unchanged before a real stage runs: {ident}")
        supersede_leftovers(ctx)
        result = dict(body(ctx) or {})
        status = "completed"
    except Blocked as exc:
        status, error = "blocked", str(exc)
        result = {"blocked_details": exc.record}
        ctx.log(f"[{stage}] BLOCKED: {exc}")
    except Exception as exc:  # noqa: BLE001  (recorded; the supervisor stops on a missing STAGE record)
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        ctx.log(f"[{stage}] FAILED:\n{traceback.format_exc()}")
    except BaseException as exc:
        end_stage(ctx, "failed", f"{type(exc).__name__}: {exc}", result)
        raise
    outcome = end_stage(ctx, status, error, result)
    return {"completed": 0, "blocked": 2}.get(outcome["status"], 1)


# ----------------------------------------------------------------------------- A1 environment census (section 8.1)
def check_equal(what: str, found: Any, expected: Any) -> None:
    if found != expected:
        raise Blocked(f"{what}: found {found}, expected {expected}")


def check_disjoint(what: str, selected: Iterable[str], other: Iterable[str]) -> None:
    common = set(selected) & set(other)
    if common:
        raise Blocked(f"selection overlaps the {what} in {len(common)} matches")


def prespec_commit(paths: Paths) -> str:
    """The PRESPEC commit the run binds: an explicit one, else the last commit of the PRESPEC path (the approved text
    plus the approval line, plus dated operational notes such as Ruling 69's)."""

    if paths.prespec_commit:
        return paths.prespec_commit
    last = _git(paths.repo, "log", "-1", "--format=%H", "--", paths.prespec_rel).stdout.decode("ascii", "replace").strip()
    if not last:
        raise Blocked(f"{paths.prespec_rel} has no commit")
    return last


def prespec_blob(paths: Paths, commit: str | None = None) -> tuple[str, int, bytes]:
    commit = commit or prespec_commit(paths)
    blob = _git(paths.repo, "cat-file", "blob", f"{commit}:{paths.prespec_rel}")
    if blob.returncode:
        raise Blocked(f"git cat-file blob {commit}:{paths.prespec_rel} failed: {blob.stderr.decode('utf-8', 'replace')}")
    return hashlib.sha256(blob.stdout).hexdigest(), len(blob.stdout), blob.stdout


def start_checks(paths: Paths) -> dict[str, Any]:
    """Section 9 start conditions: author approval, the three records and the 30 builder modules, the frozen hashes,
    the q_ka_1 models, the three parity reports and D: free space."""

    problems: list[str] = []
    refs = load_refs(paths)
    rep = paths.external_reports
    lock = read_json(rep / "EXTERNAL_SCORE_LOCK.json")
    e2a = read_json(rep / "E2a_REPORT.json")
    e1 = read_json(rep / "E1_REPORT.json")
    three = {"checked": 0, "mismatches": []}
    for label, block in (("EXTERNAL_SCORE_LOCK.code.external", lock["code"]["external"]),
                         ("EXTERNAL_SCORE_LOCK.code.frozen_imported", lock["code"]["frozen_imported"]),
                         ("E2a_REPORT.code.package", e2a["code"]["package"]),
                         ("E2a_REPORT.code.frozen_dependencies", e2a["code"]["frozen_dependencies"])):
        for rel, sha in block.items():
            three["checked"] += 1
            path = paths.repo / rel
            now = sha256_file(path) if path.is_file() else None
            if now != sha:
                three["mismatches"].append({"record": label, "path": rel, "recorded": sha, "now": now})
    old = {}
    for path, entry in e1["code"]["old_repo_builder_modules"].items():
        three["checked"] += 1
        now = sha256_file(path) if Path(path).is_file() else None
        old[path] = now
        if now != entry["sha256"]:
            three["mismatches"].append({"record": "E1_REPORT.code.old_repo_builder_modules", "path": path,
                                        "recorded": entry["sha256"], "now": now})
    if three["mismatches"]:
        problems.append(f"{len(three['mismatches'])} files differ from the three records")
    frozen = {}
    for name, entry in lock["frozen"].items():
        now = sha256_file(entry["path"]) if Path(entry["path"]).is_file() else None
        frozen[name] = {"path": entry["path"], "recorded": entry["sha256"], "now": now, "equal": now == entry["sha256"]}
        if now != entry["sha256"]:
            problems.append(f"frozen {name} differs")
    ka_lock = read_json(paths.ka_posthoc_lock)
    models = {}
    for key in CELLS:
        path = paths.ka_cell(key) / "model_refit.pkl"
        now = sha256_file(path) if path.is_file() else None
        bound = ((ka_lock.get("verify") or {}).get(f"q_ka_1:{key}:model_refit.pkl") or {}).get("sha256")
        result = read_json(paths.ka_cell(key) / "RESULT.json")
        entry = {"path": str(path), "sha256": now, "ka_posthoc_lock_r2": bound, "equal_lock": now == bound,
                 "result_model_sha256": result.get("model_sha256"), "status": result.get("status")}
        if key != PT:
            entry["equal_result"] = now == result.get("model_sha256")
            if not entry["equal_result"]:
                problems.append(f"{key}: model_refit.pkl differs from RESULT.json model_sha256")
        if not entry["equal_lock"]:
            problems.append(f"{key}: model_refit.pkl differs from KA_POSTHOC_LOCK_r2")
        models[key] = entry
    parity = {}
    for name in ("E2a_PARITY_REPORT.json", "E2b_PARITY_REPORT.json", "E2c_PARITY_REPORT.json"):
        doc = read_json(rep / name)
        parity[name] = {"status": doc.get("status"), "sha256": sha256_file(rep / name)}
        if doc.get("status") != "PASS":
            problems.append(f"{name} is not PASS")
    free = shutil.disk_usage(Path(paths.out).anchor or str(paths.out)).free
    if free < MIN_FREE_BYTES:
        problems.append(f"free space {free} below 60 GB")
    sha, size, text = prespec_blob(paths)
    approved = "- 저자 승인: 2026-10-14".encode("utf-8") in text
    if not approved:
        problems.append("the PRESPEC has no author approval line")
    return {"three_records": three, "old_repo_builder_modules_now": old, "frozen": frozen, "q_ka_1_models": models,
            "parity_reports": parity, "disk_free_bytes": free, "min_free_bytes": MIN_FREE_BYTES,
            "prespec": {"path": paths.prespec_rel, "commit": prespec_commit(paths), "blob_sha256": sha, "blob_bytes": size,
                        "approval_line_present": approved, "approved_text_commit": PRESPEC_APPROVED,
                        "approval_line_commit": PRESPEC_APPROVAL_COMMIT},
            "record_files": refs.records, "problems": problems, "passed": not problems}


def census(paths: Paths) -> dict[str, Any]:
    """A1 / A11: every file under reports/external_validation_20261009 with SHA-256, the external store listing
    (path, kind, size, modification time) and the start checks (section 8.1)."""

    tick = time.perf_counter()
    reports = {p.relative_to(paths.external_reports).as_posix(): sha256_file(p)
               for p in sorted(paths.external_reports.rglob("*")) if p.is_file()}
    entries = []
    for top, dirs, files in os.walk(paths.external_data):
        dirs.sort()
        for name in dirs:
            st = os.stat(os.path.join(top, name))
            entries.append([os.path.relpath(os.path.join(top, name), paths.external_data).replace("\\", "/"), "d", 0,
                            st.st_mtime_ns])
        for name in sorted(files):
            st = os.stat(os.path.join(top, name))
            entries.append([os.path.relpath(os.path.join(top, name), paths.external_data).replace("\\", "/"), "f",
                            st.st_size, st.st_mtime_ns])
    entries.sort()
    digest = hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()
    checks = start_checks(paths)
    old_status = _git(paths.old_repo, "status", "--porcelain", "--untracked-files=all")
    return {"version": VERSION, "experiment": EXPERIMENT, "utc": utc(),
            "external_reports": {"root": str(paths.external_reports), "files": reports, "count": len(reports)},
            "external_data": {"root": str(paths.external_data), "count": len(entries), "listing_sha256": digest,
                              "fields": ["relative path", "kind (f file, d folder)", "bytes", "mtime_ns"]},
            "external_data_entries": entries, "checks": checks,
            "old_repo_git_status": old_status.stdout.decode("utf-8", "replace"),
            "seconds": round(time.perf_counter() - tick, 1)}


def compare_census(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """A11 equals A1 for everything outside OUT (reports folder hashes, external store listing, start checks)."""

    diffs = {}
    if before["external_reports"]["files"] != after["external_reports"]["files"]:
        a, b = before["external_reports"]["files"], after["external_reports"]["files"]
        diffs["external_reports"] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if before["external_data"]["listing_sha256"] != after["external_data"]["listing_sha256"]:
        a = {e[0]: e for e in before["external_data_entries"]}
        b = {e[0]: e for e in after["external_data_entries"]}
        diffs["external_data"] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))[:200]
    if not after["checks"]["passed"]:
        diffs["start_checks_after"] = after["checks"]["problems"]
    return {"equal": not diffs, "differences": diffs}


# ----------------------------------------------------------------------------- A2 slice selection (section 2.2)
def match_hash6(match_id: str) -> str:
    """h6 = SHA256('tog_e6_20261014:' + match_id).hexdigest() (UTF-8, lowercase hex)."""

    return hashlib.sha256((HASH_TAG + str(match_id)).encode("utf-8")).hexdigest()


def rank_h6(ids: Iterable[str]) -> list[str]:
    return sorted({str(m) for m in ids}, key=lambda m: (match_hash6(m), m))


def largest_remainder(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """q_p = floor(total * n_p / N) plus one for the largest remainders (equal remainders in name order); exact
    integer arithmetic."""

    names = sorted(counts)
    n = sum(int(counts[p]) for p in names)
    if n <= 0 or total < 0 or total > n:
        raise ValueError(f"cannot allocate {total} over {n}")
    base = {p: (int(total) * int(counts[p])) // n for p in names}
    remainder = {p: int(total) * int(counts[p]) - base[p] * n for p in names}
    left = int(total) - sum(base.values())
    for p in sorted(names, key=lambda q: (-remainder[q], q))[:left]:
        base[p] += 1
    return base


def select_slice(candidates: Mapping[str, Iterable[str]], existing: Iterable[str], total: int = SLICE_SIZE) -> dict[str, Any]:
    """Remove the existing sample, allocate total by largest remainder over the remaining candidates and take the q_p
    smallest h6 per patch (no replacement)."""

    cand = {p: sorted({str(m) for m in ids}) for p, ids in candidates.items()}
    every = set().union(*(set(v) for v in cand.values())) if cand else set()
    existing = {str(m) for m in existing}
    missing = existing - every
    if missing:
        raise Blocked(f"{len(missing)} matches of the existing sample are not candidates")
    remaining = {p: sorted(set(ids) - existing) for p, ids in cand.items()}
    quotas = largest_remainder({p: len(v) for p, v in remaining.items()}, total)
    selected = {p: rank_h6(remaining[p])[: quotas[p]] for p in sorted(remaining)}
    chosen = set().union(*(set(v) for v in selected.values())) if selected else set()
    check_disjoint("existing sample", chosen, existing)
    return {"candidates": {p: len(v) for p, v in cand.items()},
            "existing": {p: len(set(v) & existing) for p, v in cand.items()},
            "remaining": {p: len(v) for p, v in remaining.items()}, "quotas": quotas, "selected": selected,
            "remaining_ids": remaining}


def _ids_sha(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def select_body(ctx: StageContext, workers: int) -> dict[str, Any]:
    paths, log = ctx.paths, ctx.log
    before = paths.out / "ENV_CENSUS_before.json"
    if not before.exists():
        doc = census(paths)
        write_json_once(before, doc, list_key="external_data_entries")
        log(f"[select] ENV_CENSUS_before.json written: {doc['external_reports']['count']} report files, "
            f"{doc['external_data']['count']} external entries, checks passed={doc['checks']['passed']}")
        checks = doc["checks"]
    else:
        checks = start_checks(paths)
    if not checks["passed"]:
        raise Blocked(f"start conditions fail: {checks['problems']}", checks)
    if paths.e6_manifest.exists():
        raise Blocked(f"{paths.e6_manifest} exists without a completed select stage record")
    from studies.external_validation_20261009 import sample as S

    t0 = time.perf_counter()
    manifest_sha = sha256_file(paths.sample_manifest)
    frozen_sha = (paths.external_reports / "SAMPLE_MANIFEST.sha256").read_text(encoding="ascii").split()[0]
    check_equal("SAMPLE_MANIFEST.json SHA-256 against SAMPLE_MANIFEST.sha256", manifest_sha, frozen_sha)
    old = read_json(paths.sample_manifest)
    old_ds = old["datasets"][EXISTING_DATASET]
    d_stems = S._stems(paths.raw / "detail")
    t_stems = set(S._stems(paths.raw / "timeline"))
    check_equal("detail files", len(d_stems), DETAIL_FILES)
    check_equal("timeline files", len(t_stems), TIMELINE_FILES)
    log(f"[select] listed detail={len(d_stems)} timeline={len(t_stems)} in {time.perf_counter() - t0:.1f}s")
    tasks = [dict(folder=str(paths.raw / "detail"), stems=d_stems[i:i + 500]) for i in range(0, len(d_stems), 500)]
    rows = []
    with E6Executor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for res in pool.map(S._scan_details, tasks):
            rows.extend(res)
    scan_seconds = round(time.perf_counter() - t0, 1)
    reasons, by_patch, builds, matchid_ne = Counter(), {}, {}, Counter()
    for row in rows:
        reason, patch = S._classify(row, t_stems)
        reasons[reason] += 1
        if reason == "candidate":
            by_patch.setdefault(patch, []).append(row[0])
            builds[row[0]] = str(row[5])
            if row[6] is False:
                matchid_ne[patch] += 1
    other = {p: len(v) for p, v in sorted(by_patch.items()) if p not in PATCHES}
    check_equal("candidates outside 15.18-15.22", other, old_ds["candidates_other_patches"])
    check_equal("scan reasons", dict(sorted(reasons.items())), old_ds["scan_reasons"])
    cands = {p: sorted(by_patch.get(p, [])) for p in PATCHES}
    check_equal("candidates", {p: len(v) for p, v in cands.items()}, CANDIDATES)
    check_equal("candidates (SAMPLE_MANIFEST per patch)", {p: len(v) for p, v in cands.items()},
                {p: old_ds["per_patch"][p]["candidates"] for p in PATCHES})
    check_equal("candidate id lists (SAMPLE_MANIFEST candidate_ids_sha256)", {p: _ids_sha(v) for p, v in cands.items()},
                {p: old_ds["per_patch"][p]["candidate_ids_sha256"] for p in PATCHES})
    existing = [m for m in old["matches"] if m["dataset"] == EXISTING_DATASET]
    check_equal("existing sample size", len(existing), EXISTING_PER_PATCH * len(PATCHES))
    existing_ids = {m["match_id"] for m in existing}
    out = select_slice(cands, existing_ids)
    check_equal("existing sample per patch", out["existing"], {p: EXISTING_PER_PATCH for p in PATCHES})
    check_equal("remaining candidates", out["remaining"], REMAINING)
    check_equal("quotas", out["quotas"], QUOTAS)
    check_equal("selected per patch", {p: len(v) for p, v in out["selected"].items()}, QUOTAS)
    chosen = sorted(set().union(*(set(v) for v in out["selected"].values())))
    check_equal("selected total", len(chosen), SLICE_SIZE)
    check_disjoint("existing sample", chosen, existing_ids)
    ref_index = read_json(paths.ref_patch_index)
    check_disjoint("2025 cache index", chosen, ref_index.keys())
    t1 = time.perf_counter()
    order = [(p, m) for p in PATCHES for m in out["selected"][p]]
    items = [(str(paths.raw), m) for _, m in order]
    raw_info = {}
    with E6Executor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for res in pool.map(S._hash_raw, [dict(items=items[i:i + 200]) for i in range(0, len(items), 200)]):
            raw_info.update(dict(res))
    hash_seconds = round(time.perf_counter() - t1, 1)
    matches = []
    for patch in PATCHES:
        for rank, mid in enumerate(out["selected"][patch]):
            r = raw_info[mid]
            matches.append(dict(match_id=mid, patch=patch, build=builds[mid], dataset=DATASET, h=match_hash6(mid),
                                rank_in_patch=rank, detail_bytes=r["detail_bytes"], detail_sha256=r["detail_sha256"],
                                timeline_bytes=r["timeline_bytes"], timeline_sha256=r["timeline_sha256"]))
    per_patch = {}
    for p in PATCHES:
        sel = out["selected"][p]
        per_patch[p] = dict(candidates=len(cands[p]), existing_sample=out["existing"][p], remaining=out["remaining"][p],
                            quota=out["quotas"][p], selected=len(sel), inclusion_fraction=len(sel) / out["remaining"][p],
                            candidate_ids_sha256=_ids_sha(cands[p]), remaining_ids_sha256=_ids_sha(out["remaining_ids"][p]),
                            selected_ids_sha256=_ids_sha(sel), h_min=match_hash6(sel[0]), h_max_selected=match_hash6(sel[-1]),
                            builds=dict(Counter(builds[s] for s in sel).most_common()),
                            candidate_builds=dict(Counter(builds[s] for s in cands[p]).most_common()),
                            metadata_matchId_ne_stem_candidates=matchid_ne.get(p, 0))
    summary = {DATASET: dict(raw_root=str(paths.raw), detail_files=len(d_stems), timeline_files=len(t_stems),
                             detail_without_timeline=len(set(d_stems) - t_stems),
                             timeline_without_detail=len(t_stems - set(d_stems)), scan_reasons=dict(sorted(reasons.items())),
                             candidates_other_patches=other, per_patch=per_patch, scan_seconds=scan_seconds)}
    manifest = dict(
        version=VERSION, kind="external_sample_manifest", frozen_utc=utc(), rule=RULE_TEXT, hash_tag=HASH_TAG,
        slice_size=SLICE_SIZE, filters=EXT.FILTERS, datasets=summary,
        counts={f"{DATASET}|{p}": len(out["selected"][p]) for p in PATCHES}, total_selected=len(matches),
        allocation={"rule": "largest remainder of 40,000 over the remaining candidates (equal remainders in patch order)",
                    "remaining": out["remaining"], "quotas": out["quotas"]},
        existing_sample={"manifest": str(paths.sample_manifest), "sha256": manifest_sha, "sha256_file": frozen_sha,
                         "dataset": EXISTING_DATASET, "matches": len(existing_ids), "per_patch": out["existing"],
                         "all_existing_are_candidates": True},
        checks={"overlap_with_existing_sample": 0, "overlap_with_2025_cache_index": 0,
                "ref_patch_index": str(paths.ref_patch_index), "ref_patch_index_sha256": sha256_file(paths.ref_patch_index),
                "candidate_ids_equal_sample_manifest": True},
        candidate_ids_sha256=_ids_sha(set().union(*(set(v) for v in cands.values()))),
        environment=dict(python=sys.version.split()[0], platform=platform.platform(), json_parser="stdlib json",
                         workers=workers, script=str(Path(__file__).resolve()), script_sha256=sha256_file(Path(__file__)),
                         sample_module_sha256=sha256_file(Path(S.__file__)),
                         package_init_sha256=sha256_file(Path(EXT.__file__))),
        timing=dict(scan_seconds=scan_seconds, raw_hash_seconds=hash_seconds,
                    total_seconds=round(time.perf_counter() - t0, 1)),
        identifiers="no player identifiers are read into this file (match ids, versions, counts and file hashes only)",
        matches=matches)
    data = EXT.json_bytes(manifest, list_key="matches")
    digest = write_bytes_once(paths.e6_manifest, data)
    write_bytes_once(paths.e6_manifest_sha, f"{digest}  {paths.e6_manifest.name}\n".encode("ascii"))
    log(f"[select] frozen {paths.e6_manifest} sha256={digest} selected={len(matches)} {manifest['counts']}")
    return {"counts": {"detail_files": len(d_stems), "timeline_files": len(t_stems), "candidates": out["candidates"],
                       "existing": out["existing"], "remaining": out["remaining"], "quotas": out["quotas"],
                       "selected": len(matches), "scan_reasons": dict(sorted(reasons.items()))},
            "details": {"scan_seconds": scan_seconds, "raw_hash_seconds": hash_seconds},
            "outputs": {"E6_SAMPLE_MANIFEST.json": paths.e6_manifest, "E6_SAMPLE_MANIFEST.sha256": paths.e6_manifest_sha,
                        "ENV_CENSUS_before.json": before},
            "inputs": {"SAMPLE_MANIFEST.json": paths.sample_manifest, "ref_patch_index": paths.ref_patch_index},
            "rows": len(matches), "matches": len(matches)}


# ----------------------------------------------------------------------------- A3 cache build (section 2.3 (3))
def check_build_manifest(doc: Mapping[str, Any]) -> int:
    failed = int(doc["summary"]["failed"])
    if failed:
        raise Blocked(f"{failed} build failure(s): zero are allowed (build_cache.py:680, :717-720)")
    return int(doc["summary"]["ok"])


def check_postbuild(doc: Mapping[str, Any]) -> None:
    if doc.get("passed") is not True:
        raise Blocked("the post-build check did not pass")


def check_exclusions(excluded: int, built: int) -> dict[str, Any]:
    """Audit and bank exclusions together, against the built matches (separate from build failures)."""

    if built <= 0:
        raise ValueError("no built match")
    share = int(excluded) / int(built)
    if share > EXCLUSION_MAX_SHARE:
        raise Blocked(f"audit and bank excluded {excluded} of {built} built matches, more than 1 %")
    return {"excluded": int(excluded), "built": int(built), "share": share, "max_share": EXCLUSION_MAX_SHARE}


def build_body(ctx: StageContext, workers: int) -> dict[str, Any]:
    paths, log = ctx.paths, ctx.log
    for path in (paths.e6_manifest, paths.e6_manifest_sha):
        if not path.is_file():
            raise Blocked(f"{path} is missing: run select first")
    if EXT.RUNTIME_REPO != paths.out / "runtime_repo" or EXT.SAMPLE_MANIFEST != paths.e6_manifest:
        raise Blocked("outputs are not bound to OUT before the build_cache import")
    from studies.external_validation_20261009 import build_cache as BC

    check_frozen_executors([BC])
    if BC.BUILDER_ENV["LOL_OUTPUT_ROOT"] != str(paths.out / "runtime_repo"):
        raise Blocked(f"build_cache BUILDER_ENV points at {BC.BUILDER_ENV['LOL_OUTPUT_ROOT']}")
    bm_path, check_path = paths.out / f"BUILD_MANIFEST_{DATASET}.json", paths.out / f"POSTBUILD_CHECK_{DATASET}.json"
    if not bm_path.exists():
        BC.mode_build(SimpleNamespace(dataset=DATASET, workers=workers, chunk=100, retry_failed=False), log)
    bm = read_json(bm_path)
    built = check_build_manifest(bm)
    if not check_path.exists():
        BC.mode_check(SimpleNamespace(dataset=DATASET, workers=workers, determinism_n=5), log)
    chk = read_json(check_path)
    check_postbuild(chk)
    return {"counts": {"selected": len(read_json(paths.e6_manifest)["matches"]), "built": built,
                       "failed": int(bm["summary"]["failed"]), "status": bm["summary"]["per_patch"],
                       "determinism_equal": chk["determinism"]["equal"], "determinism_n": chk["determinism"]["n"],
                       "rehash_mismatching_files": chk["rehash"]["mismatching_files"]},
            "details": {"builder_env": BC.BUILDER_ENV, "build_seconds_this_run": bm.get("seconds_this_run"),
                        "check_seconds": chk.get("seconds"), "frozen_executor_check": True},
            "outputs": {f"BUILD_MANIFEST_{DATASET}.json": bm_path, f"POSTBUILD_CHECK_{DATASET}.json": check_path,
                        "patch_index": paths.patch_index},
            "inputs": {"E6_SAMPLE_MANIFEST.json": paths.e6_manifest}, "rows": built, "matches": built}


# ----------------------------------------------------------------------------- A4-A6 GPU-masked steps
def _built_matches(paths: Paths) -> int:
    return check_build_manifest(read_json(paths.out / f"BUILD_MANIFEST_{DATASET}.json"))


def process_body(ctx: StageContext, step: str, workers: int) -> dict[str, Any]:
    paths, log = ctx.paths, ctx.log
    if not stage_done(paths, "build"):
        raise Blocked("the build stage is not complete")
    counts: dict[str, Any] = {}
    outputs: dict[str, Path] = {}
    details: dict[str, Any] = {}
    if step == "audit":
        from studies.external_validation_20261009 import e2a_audit

        details["frozen_executor_check"] = check_frozen_executors([e2a_audit])
        m = e2a_audit.run_audit(DATASET, workers, log)
        fail = int(m["status_counts"].get("fail", 0))
        counts = {"audited": m["n_matches"], "status_counts": m["status_counts"], "entries": m["n_matches"] - fail,
                  "audit_fail": fail}
        outputs = {"AUDIT_MANIFEST.json": paths.ext("audit") / "AUDIT_MANIFEST.json"}
    elif step == "bank":
        from studies.external_validation_20261009 import e2a_bank

        details["frozen_executor_check"] = check_frozen_executors([e2a_bank])
        m = e2a_bank.run_bank(DATASET, workers, log)
        built = _built_matches(paths)
        counts = {"bank_matches": m["n_bank_matches"], "excluded": m["n_excluded"],
                  "entries_after_audit": m["n_entries_after_audit"], "built": built,
                  **{k: m["counts"].get(k) for k in ("n_components", "n_kills", "n_supported_primary", "n_unique_queries")}}
        details["exclusions"] = check_exclusions(m["n_excluded"], built)
        outputs = {"RUN_MANIFEST.json": paths.ext("bank") / "RUN_MANIFEST.json",
                   "EXCLUDED.json": paths.ext("bank") / "EXCLUDED.json",
                   "BANK_ENTRIES.jsonl": paths.ext("bank") / "BANK_ENTRIES.jsonl"}
    elif step == "preparation":
        from studies.external_validation_20261009 import e2a_preparation

        m = e2a_preparation.run_preparation(DATASET, log)
        counts = dict(m.get("counts") or {})
        outputs = {"RUN_MANIFEST.json": paths.ext("preparation") / "RUN_MANIFEST.json"}
    elif step == "items":
        from studies.external_validation_20261009 import e2a_items

        details["frozen_executor_check"] = check_frozen_executors([e2a_items])
        m = e2a_items.run_items(DATASET, workers, log)
        counts = dict(m.get("counts") or {})
        outputs = {"RUN_MANIFEST.json": paths.ext("items") / "RUN_MANIFEST.json"}
    elif step == "projection":
        from studies.external_validation_20261009 import e2a_projection

        m = e2a_projection.run_projection(DATASET, log)
        counts = dict(m.get("counts") or {})
        outputs = {"RUN_MANIFEST.json": paths.ext("projection") / "RUN_MANIFEST.json",
                   "TEAM_KILLS.npy": paths.ext("projection") / "TEAM_KILLS.npy"}
    elif step == "history":
        from studies.external_validation_20261009 import e2a_history

        details["frozen_executor_check"] = check_frozen_executors([e2a_history])
        m = e2a_history.run_history(DATASET, workers, log)
        counts = {"matches": m["matches"], "queries": m["queries"], "shards": len(m["shards"]),
                  "empty_query_matches": m["empty_query_matches"]}
        outputs = {"ACCEPTANCE.json": paths.ext("history") / "ACCEPTANCE.json", "PLAN.json": paths.ext("history") / "PLAN.json"}
    elif step == "engagements":
        from studies.external_validation_20261009 import e2b_labels

        r = e2b_labels.run_engagements(DATASET, log)
        counts = {"engagements": r["rows"], "valid_counts": r["valid_counts"]}
        outputs = {"ENGAGEMENTS.npz": paths.ext("labels") / "ENGAGEMENTS.npz",
                   "ENGAGEMENTS_MANIFEST.json": paths.ext("labels") / "ENGAGEMENTS_MANIFEST.json"}
    elif step == "assemble":
        if not stage_done(paths, "score_v"):
            raise Blocked("V scoring (score_v) is not complete")
        from studies.external_validation_20261009 import e2b_labels

        r = e2b_labels.assemble_labels(DATASET, log)
        counts = {"label_rows": r["counts"], "label_values_read": r.get("label_values_read")}
        outputs = {"LABELS_PUBLIC_h90.npz": paths.ext("labels") / "LABELS_PUBLIC_h90.npz",
                   "LABELS_MANIFEST_EXTERNAL.json": paths.ext("labels") / "LABELS_MANIFEST_EXTERNAL.json"}
        details["sealed_files_hashed_only"] = {"EXTERNAL_LABELS_SEALED_kr_e6.npz": sha256_file(B.sealed_labels_path(DATASET))}
    elif step == "e2c":
        from studies.external_validation_20261009 import e2c_scores

        rows = e2c_scores.build_rows(DATASET, log)
        store = e2c_scores.build_store(DATASET, workers, log)
        xy = e2c_scores.build_xy(DATASET, log)
        counts = {"q_rows": rows["rows"], "q_row_matches": rows["matches"], "rows_by_patch": rows["rows_by_patch"],
                  "excluded_invalid_rows": rows["excluded_invalid_rows"], "store_rows": store["rows"], "xy_rows": xy["rows"]}
        outputs = {"Q_ROWS_h90.npz": paths.ext("q_inputs") / "Q_ROWS_h90.npz",
                   "Q_ROWS_h90.json": paths.ext("q_inputs") / "Q_ROWS_h90.json",
                   "store/BUILD.json": paths.ext("q_inputs") / "raw" / "store" / "BUILD.json",
                   "XY.npy": paths.ext("q_inputs") / "XY.npy", "XY.json": paths.ext("q_inputs") / "XY.json"}
    else:
        raise ValueError(f"unknown process step {step!r}")
    log(f"[{step}] counts {json.dumps(jsonable(counts), sort_keys=True)}")
    return {"counts": counts, "outputs": outputs, "details": details}


# ----------------------------------------------------------------------------- A5 V scoring in a fresh CUDA process
def score_v_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The score-v environment: the allowlisted operating-system / Python entries of the parent plus the fixed values,
    nothing else (section 4 GPU)."""

    base = os.environ if base is None else base
    upper = {str(k).upper(): str(v) for k, v in base.items()}
    env = {name: upper[name] for name in SCORE_V_ALLOW if name in upper}
    env.update(SCORE_V_FIXED)
    return env


def check_device(torch: Any) -> dict[str, Any]:
    """CUDA available, exactly one visible device, and that device the RTX 5080; otherwise BLOCKED (scoring never
    moves to another device)."""

    record: dict[str, Any] = {"torch": getattr(torch, "__version__", None),
                              "cuda": getattr(getattr(torch, "version", None), "cuda", None), "utc": utc()}
    problems: list[str] = []
    try:
        record["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        record["cudnn"] = None
    record["is_available"] = bool(torch.cuda.is_available())
    if not record["is_available"]:
        problems.append("torch.cuda.is_available() is false (CUDA not available)")
    else:
        record["device_count"] = int(torch.cuda.device_count())
        if record["device_count"] != 1:
            problems.append(f"visible device count {record['device_count']} != 1")
        if record["device_count"] >= 1:
            record["device_name"] = str(torch.cuda.get_device_name(0))
            try:
                record["device_capability"] = list(torch.cuda.get_device_capability(0))
            except Exception:  # noqa: BLE001
                record["device_capability"] = None
            if DEVICE_NAME not in record["device_name"]:
                problems.append(f"device {record['device_name']!r} is not the {DEVICE_NAME}")
    record.update(problems=problems, passed=not problems,
                  rule=f"is_available true, device_count 1, device name contains {DEVICE_NAME!r}; any failure is BLOCKED "
                       "and nothing is scored (PRESPEC_E6 section 4)")
    if problems:
        raise Blocked("V scoring device check failed: " + "; ".join(problems), record)
    return record


def _import_torch():
    import torch

    return torch


def _frozen_run_scoring(dataset: str, workers: int, device: str, log: Callable[[str], None]) -> dict[str, Any]:
    from studies.external_validation_20261009 import e2b_labels

    return e2b_labels.run_scoring(dataset, workers, device, log)


def score_v_body(ctx: StageContext, workers: int) -> dict[str, Any]:
    paths, log = ctx.paths, ctx.log
    if not stage_done(paths, "engagements") and Path(paths.out) == OUT_DEFAULT:
        raise Blocked("the engagement step is not complete")
    env = dict(ctx.env_at_start)
    target = next_free(paths.out / "V_SCORING_DEVICE.json")
    try:
        record = check_device(_import_torch())
    except Blocked as exc:
        write_json_once(target, {**(exc.record or {}), "environment": env, "stage_run_id": ctx.run_id})
        raise
    record.update(environment=env, stage_run_id=ctx.run_id, workers=workers,
                  environment_rule="exactly score_v_env(): allowlisted entries of the launching process plus fixed values")
    write_json_once(target, record)
    log(f"[score_v] device check passed: {record.get('device_name')} torch {record.get('torch')} cuda {record.get('cuda')}")
    pred = _frozen_run_scoring(DATASET, workers, "cuda", log)
    return {"counts": {"queries_scored": pred.get("rows"), "parts": pred.get("parts")},
            "details": {"device": record, "pred_manifest_fields": {k: pred.get(k) for k in ("rows", "parts", "queries_sha256")},
                        "pred_files_hashed_only": True},
            "outputs": {"V_SCORING_DEVICE.json": target}}


def score_v_stage(paths: Paths, workers: int = MAX_WORKERS, *, require_committed: bool | None = None) -> int:
    return run_stage(paths, "score_v", lambda ctx: score_v_body(ctx, workers), require_committed=require_committed)


# ============================================================================= analysis side (A7-A12, part B)
KA_COLUMNS = ("engagement_key", "ka", "blue_kills", "red_kills", "eligible", "y_ka", "weight_raw_ka")
KA_SEALED_NAME = "KA_LABELS_E6_SEALED.npz"
KA_SYNTH_NAME = "KA_LABELS_E6_SYNTH.npz"
P1_CONTRAST = {P1: 1.0, PT: -1.0}
S_CONTRAST = {S10: 0.5, S00: -0.5, S11: 0.5, S01: -0.5}                  # kill_label.py:76 order
SEED_BOOT, SEED_SYNTH, RESAMPLES, LEVEL, Z = 20261011, 20261014, 2000, 0.95, 1.96
SYNTH_TIE = 0.098
QDEV_TOL = 1e-9                                                         # PRESPEC 8.1 A10
INCOMPLETE_MAX_SHARE = 0.01                                             # PRESPEC 5.3
KILL_ORDER = ("1", "2-3", "4+")
PART_B_LIMIT_S = 3600.0
SIGN = "Brier(P1 LightGBM) - Brier(PT_flex), match weighted, raw probabilities; negative = LightGBM better"
DECISION_TEXT = {"confirmed": "확증 (upper < 0)", "unresolved": "미결 (the interval includes 0)",
                 "opposite": "반대 방향 (lower > 0)"}
OPENING_NOTE = ("first opening of the E6 kill-advantage labels of a never-processed KR 15.18-15.22 slice; pre-registered "
                "confirmatory test, PRESPEC_E6_KO.md; the slice's sealed SVI file and PRED_final are not opened")
ANALYSIS_STEPS = ("players", "score", "labels", "rehearse")
LOCK_BASE = "E6_LOCK"


class LabelAccessError(RuntimeError):
    """E6 KA labels asked for before the OPENINGS.json entry of the TEST step was saved (PRESPEC 8.3 B1)."""


def require_stage(paths: Paths, stage: str) -> None:
    if not stage_done(paths, stage):
        raise Blocked(f"the {stage} stage is not complete")


# ----------------------------------------------------------------------------- KA_v1 (PRESPEC 3; kill_label.py:115-141)
def ka_labels(rows: Mapping[str, Any], labels: Mapping[str, Any], team_kills: Any) -> dict[str, np.ndarray]:
    """KA = Blue kills gained - Red kills gained between the pre query and the h90 post query, from cumulative team
    kills; rows aligned to the label table by (canonical_match_key, component_key); eligible = KA != 0; y = 1[KA > 0];
    weight 1/m over the eligible rows of each match (kill_label.cmd_prepare lines 124-141)."""

    index = {(m, c): i for i, (m, c) in enumerate(zip(np.asarray(labels["canonical_match_key"]).tolist(),
                                                      np.asarray(labels["component_key"]).tolist()))}
    try:
        pick = np.asarray([index[(m, c)] for m, c in zip(np.asarray(rows["canonical_match_key"]).tolist(),
                                                         np.asarray(rows["component_key"]).tolist())], dtype=np.int64)
    except KeyError as error:
        raise Blocked("a q-row key is not in the label table (key alignment failed)") from error
    pre = np.asarray(labels["query_pre"], dtype=np.int64)[pick]
    post = np.asarray(labels["query_post"], dtype=np.int64)[pick]
    if not np.array_equal(pre, np.asarray(rows["query_pre"], dtype=np.int64)):
        raise Blocked("q rows and label rows disagree on the pre query (kill_label.py:127-128)")
    n_queries = len(team_kills)
    if pre.size and (pre.min() < 0 or post.min() < 0 or pre.max() >= n_queries or post.max() >= n_queries):
        raise Blocked("a query reference is outside the projection's query table")
    blue = np.asarray(team_kills[post, 0], dtype=np.int64) - np.asarray(team_kills[pre, 0], dtype=np.int64)
    red = np.asarray(team_kills[post, 1], dtype=np.int64) - np.asarray(team_kills[pre, 1], dtype=np.int64)
    if (blue < 0).any() or (red < 0).any():
        raise Blocked("cumulative team kills decrease inside a window (kill_label.py:132-133)")
    ka = blue - red
    eligible = ka != 0
    y_ka = (ka > 0).astype(np.float64)
    match = np.asarray(rows["match_index"], dtype=np.int64)
    m = np.bincount(match[eligible], minlength=int(match.max()) + 1).astype(np.float64)
    weight = np.where(eligible, 1.0 / np.maximum(m[match], 1.0), 0.0)
    return {"engagement_key": np.asarray(rows["engagement_key"]), "ka": ka.astype(np.int32),
            "blue_kills": blue.astype(np.int32), "red_kills": red.astype(np.int32), "eligible": eligible, "y_ka": y_ka,
            "weight_raw_ka": weight}


def gate_compare(mine: Mapping[str, np.ndarray], ref: Mapping[str, np.ndarray], keep: Any) -> dict[str, Any]:
    """Array-for-array equality (dtype, shape, values) of the recomputed labels and KA_LABELS.npz on the kept rows."""

    keep = np.asarray(keep, dtype=bool)
    arrays = {}
    for name in KA_COLUMNS:
        a, r = np.asarray(mine[name]), np.asarray(ref[name])[keep]
        arrays[name] = bool(a.dtype == r.dtype and a.shape == r.shape and np.array_equal(a, r))
    return {"arrays": arrays, "rows": int(keep.sum()), "passed": all(arrays.values()),
            "rule": "np.array_equal and equal dtype for ka, blue_kills, red_kills, eligible, y_ka, weight_raw_ka and the keys"}


def ka_gate(paths: Paths) -> dict[str, Any]:
    """A9 gate: the new KA function on the TRAIN, Q_DEV and Q_CAL q rows of q_run_1 (keys, query_pre and query_post of
    labels_1/LABELS_h90.npz only; TEST rows neither computed nor compared) equals q_ka_1/KA_LABELS.npz."""

    tick = time.perf_counter()
    rows_path, labels_path = paths.q_run / "Q_ROWS_h90.npz", paths.training / "labels_1" / "LABELS_h90.npz"
    kills_path = ROOT / "outputs" / "v_input_projection_20261002" / "full_raw_20261002_1" / "TEAM_KILLS.npy"
    prepare = read_json(paths.q_ka / "KA_PREPARE.json")
    hashes = {"Q_ROWS_h90.npz": sha256_file(rows_path), "TEAM_KILLS.npy": sha256_file(kills_path),
              "KA_LABELS.npz": sha256_file(paths.q_ka / "KA_LABELS.npz"), "LABELS_h90.npz": sha256_file(labels_path)}
    for name, key in (("Q_ROWS_h90.npz", "rows_sha256"), ("TEAM_KILLS.npy", "team_kills_sha256"), ("KA_LABELS.npz", "ka_labels_sha256")):
        if hashes[name] != prepare.get(key):
            raise Blocked(f"{name} differs from KA_PREPARE.json {key}")
    rows = load_npz(rows_path, ["canonical_match_key", "component_key", "engagement_key", "query_pre", "match_index", "role"])
    role = np.asarray(rows["role"])
    keep = np.isin(role, (0, 3, 4))
    if np.intersect1d(np.asarray(rows["match_index"])[keep], np.asarray(rows["match_index"])[~keep]).size:
        raise Blocked("a match has rows inside and outside TRAIN/Q_DEV/Q_CAL (per-match weights would differ)")
    sub = {name: np.asarray(value)[keep] for name, value in rows.items() if name != "role"}
    labels = load_npz(labels_path, ["canonical_match_key", "component_key", "query_pre", "query_post"])
    kills = np.load(kills_path, mmap_mode="r", allow_pickle=False)
    mine = ka_labels(sub, labels, kills)
    del kills
    ref = load_npz(paths.q_ka / "KA_LABELS.npz", KA_COLUMNS)
    if not np.array_equal(np.asarray(ref["engagement_key"]), np.asarray(rows["engagement_key"])):
        raise Blocked("KA_LABELS.npz is not aligned with the q_run_1 row table")
    out = gate_compare(mine, ref, keep)
    out.update(per_role_rows={name: int((role == code).sum()) for name, code in (("TRAIN", 0), ("Q_DEV", 3), ("Q_CAL", 4))},
               test_rows_computed=0, inputs={**{k: str(v) for k, v in (("rows", rows_path), ("labels", labels_path),
                                                                          ("team_kills", kills_path))}},
               input_sha256=hashes, label_table_columns_read=["canonical_match_key", "component_key", "query_pre", "query_post"],
               seconds=round(time.perf_counter() - tick, 1))
    return out


def labels_body(ctx: StageContext) -> dict[str, Any]:
    """A9: the gate, then the E6 KA labels computed in memory and written straight to the sealed file (nothing about
    them is printed, summarised or recorded beyond path, SHA-256, row count and code hashes)."""

    paths, log = ctx.paths, ctx.log
    require_stage(paths, "score")
    gate_path = paths.out / "KA_GATE.json"
    if gate_path.exists():
        gate = read_json(gate_path)
    else:
        gate = ka_gate(paths)
        gate.update(version=VERSION, experiment=EXPERIMENT, utc=utc(), code_sha256=sha256_file(paths.repo / CODE_REL))
        write_json_once(gate_path, gate)
    if not gate.get("passed"):
        raise Blocked(f"the KA label function gate failed: {gate.get('arrays')}")
    log(f"[labels] gate passed on {gate['rows']} TRAIN/Q_DEV/Q_CAL rows")
    sealed, manifest_path = paths.out / KA_SEALED_NAME, paths.out / "KA_LABELS_E6_MANIFEST.json"
    if sealed.exists() or manifest_path.exists():
        raise Blocked(f"{sealed} or its manifest exists without a completed labels stage")
    q = paths.ext("q_inputs")
    rows_meta = read_json(q / "Q_ROWS_h90.json")
    rows_path, public = q / "Q_ROWS_h90.npz", paths.ext("labels") / "LABELS_PUBLIC_h90.npz"
    kills_path = paths.ext("projection") / "TEAM_KILLS.npy"
    if sha256_file(rows_path) != rows_meta["rows_sha256"]:
        raise Blocked("the E6 row table differs from Q_ROWS_h90.json")
    rows = load_npz(rows_path, ["canonical_match_key", "component_key", "engagement_key", "query_pre", "match_index", "y"])
    if (np.asarray(rows["y"]) != -1).any():
        raise Blocked("the E6 row table holds a label value (y must be -1)")
    labels = load_npz(public, ["canonical_match_key", "component_key", "query_pre", "query_post"])
    kills = np.load(kills_path, mmap_mode="r", allow_pickle=False)
    arrays = ka_labels(rows, labels, kills)
    del kills
    sha = write_npz_once(sealed, arrays)
    n = int(arrays["engagement_key"].size)
    del arrays
    manifest = {"version": VERSION, "experiment": EXPERIMENT, "path": str(sealed), "sha256": sha, "rows": n,
                "columns": list(KA_COLUMNS), "q_row_order": True,
                "code_sha256": {CODE_REL: sha256_file(paths.repo / CODE_REL),
                                "reports/q_production_20261005/kill_label.py":
                                    sha256_file(paths.repo / "reports/q_production_20261005/kill_label.py")},
                "inputs_sha256": {"Q_ROWS_h90.npz": rows_meta["rows_sha256"], "LABELS_PUBLIC_h90.npz": sha256_file(public),
                                  "TEAM_KILLS.npy": sha256_file(kills_path)},
                "gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)}, "utc": utc(),
                "note": "computed in memory and written straight to the sealed file; no count or rate of the labels was computed"}
    write_json_once(manifest_path, manifest)
    log(f"[labels] sealed {sealed.name} sha256={sha} rows={n}")
    return {"counts": {"gate_rows": gate["rows"], "ka_rows": n}, "rows": n,
            "outputs": {"KA_GATE.json": gate_path, "KA_LABELS_E6_MANIFEST.json": manifest_path},
            "details": {"sealed_sha256": sha, "sealed_hashed_only": True},
            "inputs": {"Q_ROWS_h90.npz": rows_path, "TEAM_KILLS.npy": kills_path}}


# ----------------------------------------------------------------------------- A7 player identification (PRESPEC 5.3)
def incomplete_identification(player_idx: Any, q_row_matches: int) -> dict[str, Any]:
    """k_m = identified players (entries >= 0) of each matrix row; every match with k_m < 10 is incomplete, failure
    reason or not; denominator = matches with E6 q rows."""

    idx = np.asarray(player_idx)
    if idx.ndim != 2 or idx.shape[0] != int(q_row_matches):
        raise Blocked(f"the player matrix ({idx.shape}) does not cover the {q_row_matches} q-row matches")
    k = (idx >= 0).sum(axis=1)
    n = int((k < 10).sum())
    return {"available": True, "incomplete_matches": n, "q_row_matches": int(q_row_matches), "share": n / int(q_row_matches)}


def choose_branch(block: Mapping[str, Any]) -> dict[str, Any]:
    """(가) two-way primary when incomplete identification <= 1 %; (나) match bootstrap when above; (다) match bootstrap
    when the identification source is unavailable."""

    if not block.get("available"):
        branch, primary, reason = "다", "match_bootstrap", "player_identification_unavailable"
    elif float(block["share"]) <= INCOMPLETE_MAX_SHARE:
        branch, primary, reason = "가", "twoway", None
    else:
        branch, primary, reason = "나", "match_bootstrap", "incomplete_identification_over_1pct"
    return {"branch": branch, "primary_interval": primary, "twoway_unavailable_reason": reason,
            "incomplete_identification": dict(block),
            "rule": "PRESPEC_E6 5.3: (가) incomplete <= 1 % two-way; (나) > 1 % match bootstrap; (다) detail folder absent "
                    "or not listable, match bootstrap; fixed before any label is read"}


def detail_folder_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {"path": str(path), "exists": Path(path).is_dir(), "listable": False, "error": None}
    if not status["exists"]:
        status["error"] = {"name": "FileNotFoundError", "message": "the detail folder does not exist"}
        return status
    try:
        with os.scandir(path) as entries:
            next(entries, None)
        status["listable"] = True
    except OSError as error:
        status["error"] = {"name": type(error).__name__, "message": str(error)}
    return status


def _e3_build(rows_path: Path, patch_index_path: Path, detail_dir: Path):
    from studies.tog_experiments_20261012 import e3_players

    return e3_players.build(e3_players.DATASET, rows_path=rows_path, patch_index_path=patch_index_path, detail_dir=detail_dir)


def player_distribution(player_idx: Any) -> dict[str, Any]:
    idx = np.asarray(player_idx)
    known = idx >= 0
    if not known.any():
        return {"players": 0, "identified_slots": 0}
    counts = np.bincount(np.unique(idx[known], return_inverse=True)[1].reshape(-1))
    q50, q90, q99 = np.quantile(counts, [0.5, 0.9, 0.99]).tolist()
    return {"players": int(counts.size), "identified_slots": int(known.sum()), "unidentified_slots": int((~known).sum()),
            "matches_per_player": {"min": int(counts.min()), "median": q50, "p90": q90, "p99": q99, "max": int(counts.max()),
                                   "mean": float(counts.mean()), "share_in_one_match": float((counts == 1).mean()),
                                   "histogram": [[k, int(v)] for k, v in enumerate(np.bincount(counts)) if v]}}


def players_body(ctx: StageContext) -> dict[str, Any]:
    paths, log = ctx.paths, ctx.log
    require_stage(paths, "e2c")
    rows_path, index_path = paths.ext("q_inputs") / "Q_ROWS_h90.npz", paths.patch_index
    missing = [str(p) for p in (rows_path, index_path) if not p.is_file()]
    if missing:
        raise Blocked(f"mandatory analysis inputs are missing: {missing}")
    try:
        with np.load(guard_read(rows_path), allow_pickle=False) as data:
            q_matches = int(np.unique(np.asarray(data["match_index"])).size)
        index = read_json(index_path)
        if not isinstance(index, dict) or not index:
            raise ValueError("empty patch index")
    except Exception as error:  # noqa: BLE001
        raise Blocked(f"a mandatory analysis input cannot be read: {type(error).__name__}: {error}") from error
    detail = paths.raw / "detail"
    status = detail_folder_status(detail)
    inputs = {"Q_ROWS_h90.npz": rows_path, "patch_index": index_path}
    if not status["listable"]:
        block = {"available": False, "reason": "player_identification_unavailable", "q_row_matches": q_matches}
        branch = choose_branch(block)
        target = paths.out / "PLAYERS_E6_UNAVAILABLE.json"
        write_json_once(target, {"version": VERSION, "experiment": EXPERIMENT, "reason": "player_identification_unavailable",
                                 "detail_folder": status, "incomplete_identification": block, "branch": branch, "utc": utc(),
                                 "identifiers": "none"})
        log(f"[players] branch 다: detail folder unavailable ({status['error']})")
        return {"counts": {"q_row_matches": q_matches, "branch": "다"}, "details": {"branch": branch},
                "outputs": {"PLAYERS_E6_UNAVAILABLE.json": target}, "inputs": inputs, "matches": q_matches}
    try:
        built = _e3_build(rows_path, index_path, detail)
    except Exception as error:  # noqa: BLE001  (section 5.3: build raising is BLOCKED, not branch 다)
        raise Blocked(f"e3_players.build raised {type(error).__name__}: {error}") from error
    block = incomplete_identification(built.player_idx, q_matches)
    branch = choose_branch(block)
    from studies.tog_experiments_20261012 import e3_players

    k_m = np.bincount((built.player_idx >= 0).sum(axis=1), minlength=11)
    arrays = {"match_index": built.match_index, "canonical_match_key": built.match_keys, "player_idx": built.player_idx,
              "map_status": built.map_status, "players_by_match_count": e3_players.players_by_match_count(built.player_idx)}
    npz_path, json_path = paths.out / "PLAYERS_E6.npz", paths.out / "PLAYERS_E6.json"
    npz_sha = write_npz_once(npz_path, arrays)
    summary = e3_players.summarize(built)
    summary["mapping"].pop("blocked", None)
    doc = {"version": VERSION, "experiment": EXPERIMENT, "branch": branch,
           "order": "one row per match of the E6 row table in ascending match_index (np.unique)",
           "columns": e3_players.COLUMNS, "privacy": e3_players.PRIVACY, "npz": str(npz_path), "npz_sha256": npz_sha,
           "npz_arrays": {name: {"dtype": str(a.dtype), "shape": list(a.shape)} for name, a in arrays.items()},
           "k_m_distribution": {str(k): int(v) for k, v in enumerate(k_m)},
           "failure_counts": e3_players.failure_counts(built.map_status), "summary": summary,
           "players": player_distribution(built.player_idx),
           "detail_files": {"read": built.files_read, "sha256_of_file_digests": built.files_digest,
                            "digest_rule": e3_players.DIGEST_RULE, "folder": status},
           "build_seconds": built.seconds, "inputs_sha256": {k: sha256_file(v) for k, v in inputs.items()}, "utc": utc(),
           "identifiers": "integers and aggregates only; puuid strings never left e3_players._number_players"}
    write_json_once(json_path, doc)
    log(f"[players] branch {branch['branch']}: incomplete {block['incomplete_matches']}/{q_matches} "
        f"players {doc['players'].get('players')}")
    return {"counts": {"q_row_matches": q_matches, "branch": branch["branch"], "incomplete_matches": block["incomplete_matches"],
                       "players": doc["players"].get("players")},
            "details": {"branch": branch}, "outputs": {"PLAYERS_E6.npz": npz_path, "PLAYERS_E6.json": json_path},
            "inputs": inputs, "matches": q_matches, "rows": built.rows}


def players_stage(paths: Paths, *, require_committed: bool | None = None) -> int:
    return run_stage(paths, "players", players_body, require_committed=require_committed)


# ----------------------------------------------------------------------------- A8 scoring the q_ka_1 REFIT cells
def load_ka_cell(paths: Paths, key: str, device: str = "cpu") -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """qtest.LoadedCell built from the q_run_1 source record named in the q_ka_1 RESULT.json (rows, store, XY,
    REFIT_TRAIN context, condition), with the model replaced by q_ka_1's model_refit.pkl (LightGBM adapter, PT_flex
    {model, design})."""

    import pickle

    from studies.q_production_20261005 import qtest

    result_path = paths.ka_cell(key) / "RESULT.json"
    result = read_json(result_path)
    if result.get("status") != "completed":
        raise Blocked(f"{key}: q_ka_1 RESULT.json is not completed")
    source = Path(result["source_dir"])
    spec_sha = sha256_file(source / "SPEC.json")
    if spec_sha != result.get("source_spec_sha256"):
        raise Blocked(f"{key}: the q_run_1 source SPEC.json differs from RESULT.json source_spec_sha256")
    model_path = paths.ka_cell(key) / "model_refit.pkl"
    model_sha = sha256_file(model_path)
    bound = ((read_json(paths.ka_posthoc_lock).get("verify") or {}).get(f"q_ka_1:{key}:model_refit.pkl") or {}).get("sha256")
    if model_sha != bound:
        raise Blocked(f"{key}: model_refit.pkl differs from KA_POSTHOC_LOCK_r2")
    if key != PT and model_sha != result.get("model_sha256"):
        raise Blocked(f"{key}: model_refit.pkl differs from RESULT.json model_sha256")
    loaded = qtest.LoadedCell({"key": key, "source_dir": str(source)}, device=device)
    with open(model_path, "rb") as stream:
        model = pickle.load(stream)
    if loaded.kind == "table":
        loaded._model = model
    elif loaded.kind == "reference" and loaded.family == "PT_flex":
        loaded._model, loaded._design = model["model"], model["design"]
    else:
        raise Blocked(f"{key}: unexpected family {loaded.family}")
    info = {"key": key, "kind": loaded.kind, "family": loaded.family, "source_dir": str(source), "spec_sha256": spec_sha,
            "model_sha256": model_sha, "result_sha256": sha256_file(result_path)}
    return loaded, info, result


def reload_check(paths: Paths, key: str, loaded: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    from studies.q_production_20261005 import qtest

    path = paths.ka_cell(key) / "pred_q_dev.npz"
    sha = sha256_file(path)
    if sha != ((result.get("scores") or {}).get("q_dev") or {}).get("pred_sha256"):
        raise Blocked(f"{key}: pred_q_dev.npz differs from RESULT.json")
    stored = load_npz(path, ["row", "logit"])
    rows = np.asarray(stored["row"], dtype=np.int64)[: qtest.RELOAD_ROWS]
    again = loaded.logits(rows)
    diff = float(np.abs(again - np.asarray(stored["logit"], dtype=np.float64)[: qtest.RELOAD_ROWS]).max()) if rows.size else 0.0
    atol = qtest.RELOAD_ATOL[loaded.kind]
    if not diff <= atol:
        raise Blocked(f"{key}: the reloaded model differs from q_ka_1 pred_q_dev.npz by {diff} > {atol}")
    return {"rows": int(rows.size), "max_abs_logit_diff": diff, "atol": atol, "pred_q_dev_sha256": sha}


def score_body(ctx: StageContext) -> dict[str, Any]:
    """A8: the six q_ka_1 REFIT cells, each reload-checked on its first 4,096 stored Q_DEV rows, score every E6 q row
    (no label is read; scores/<cell>.npz holds row, engagement_key, logit, p)."""

    import copy

    from studies.external_validation_20261009 import e2c_scores
    from studies.q_production_20261005 import features as qf, qtest

    paths, log = ctx.paths, ctx.log
    require_stage(paths, "players")
    q = paths.ext("q_inputs")
    rows_path = q / "Q_ROWS_h90.npz"
    if sha256_file(rows_path) != read_json(q / "Q_ROWS_h90.json")["rows_sha256"]:
        raise Blocked("the E6 row table differs from Q_ROWS_h90.json")
    e6_rows = load_npz(rows_path)
    if (np.asarray(e6_rows["y"]) != -1).any():
        raise Blocked("the E6 row table holds a label value (y must be -1)")
    store = qf.RawStore(q / "raw" / "store")
    xy = np.load(guard_read(q / "XY.npy"), allow_pickle=False)
    n = int(len(e6_rows["query_pre"]))
    all_rows = np.arange(n, dtype=np.int64)
    out_dir = paths.out / "scores"
    contexts: dict[str, Any] = {}
    metas, unscorable = {}, {}
    for key in CELLS:
        meta_path, npz_path = out_dir / f"{key}.json", out_dir / f"{key}.npz"
        if meta_path.is_file():
            meta = read_json(meta_path)
            if meta["pred_sha256"] != sha256_file(npz_path):
                raise Blocked(f"{npz_path} differs from its record")
            metas[key] = meta
            continue
        if npz_path.exists():
            raise Blocked(f"{npz_path} exists without its record")
        tick = time.perf_counter()
        try:
            loaded, info, result = load_ka_cell(paths, key)
            reload = reload_check(paths, key, loaded, result)
            ext = copy.copy(loaded)
            ext.rows = e6_rows
            saved = None
            if loaded.kind != "reference":
                saved = Path(loaded.spec["inputs"]["contexts"]["REFIT_TRAIN"])
                if str(saved) not in contexts:
                    contexts[str(saved)] = e2c_scores.external_context_class()(
                        store, e6_rows, xy, "REFIT_TRAIN", saved=saved,
                        expected_context_json_sha256=e2c_scores._prepared_context_sha(saved))
                ext._ctx = contexts[str(saved)]
            logits = np.asarray(qtest.LoadedCell.logits(ext, all_rows), dtype=np.float64)
            if logits.shape != (n,) or not np.isfinite(logits).all():
                raise Blocked(f"{key}: non-finite or missing E6 logits")
        except Exception as error:  # noqa: BLE001
            if key in (P1, PT):
                raise Blocked(f"{key}: {type(error).__name__}: {error}") from error
            unscorable[key] = f"{type(error).__name__}: {error}"
            log(f"[score] {key}: not scorable ({unscorable[key]}); secondary (a) becomes 채점 불가")
            continue
        p = qtest.sigmoid(logits)
        sha = write_npz_once(npz_path, {"row": all_rows, "engagement_key": np.asarray(e6_rows["engagement_key"]),
                                        "logit": logits, "p": p})
        meta = {"key": key, "rows": n, "pred_sha256": sha, **info, "reload": reload,
                "context": None if saved is None else str(saved),
                "context_json_sha256": None if saved is None else sha256_file(saved / "CONTEXT.json"),
                "rows_table": str(rows_path), "rows_table_sha256": sha256_file(rows_path), "device": "cpu",
                "probability": "p = sigmoid(logit), no calibrator (PRESPEC 3)", "seconds": round(time.perf_counter() - tick, 1),
                "utc": utc()}
        write_json_once(meta_path, meta)
        metas[key] = meta
        log(f"[score] {key}: {loaded.kind} rows={n} reload_diff={reload['max_abs_logit_diff']:.2e} {meta['seconds']} s")
    s_scored = all(k in metas for k in S_CELLS)
    return {"counts": {"q_rows": n, "cells_scored": sorted(metas), "s_scored": s_scored},
            "details": {"unscorable": unscorable, "reload": {k: m.get("reload") for k, m in metas.items()}},
            "outputs": {**{f"{k}.npz": out_dir / f"{k}.npz" for k in metas}, **{f"{k}.json": out_dir / f"{k}.json" for k in metas}},
            "inputs": {"Q_ROWS_h90.npz": rows_path}, "rows": n}


# ----------------------------------------------------------------------------- evaluation (PRESPEC 5-7; part B and both rehearsals)
@dataclass(frozen=True)
class PlayerInput:
    available: bool
    reason: str | None
    match_index: np.ndarray | None
    player_idx: np.ndarray | None
    record: str | None


def decide(interval: Iterable[float]) -> str:
    """Section 6: upper < 0 confirmed, lower > 0 opposite, otherwise (the interval includes 0) unresolved."""

    lower, upper = (float(v) for v in interval)
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
        raise ValueError(f"not an interval: {lower}, {upper}")
    if upper < 0.0:
        return "confirmed"
    if lower > 0.0:
        return "opposite"
    return "unresolved"


def _boot_entry(contrast: Mapping[str, Any], boot: Mapping[str, Any]) -> dict[str, Any]:
    return {"point": contrast["point"], "interval": list(contrast["interval"]), "level": contrast["level"],
            "quantiles": list(contrast["quantiles"]), "resamples": boot["resamples"], "seed": boot["seed"],
            "matches": boot["matches"], "method": "qeval.paired_match_bootstrap (percentile, numpy.quantile linear)"}


def _twoway_entry(values: Any, player_idx: Any) -> dict[str, Any]:
    from studies.tog_experiments_20261012 import boot

    tw = boot.twoway_interval(values, player_idx, LEVEL)
    entry = {"available": True, **tw, "interval": list(tw["interval"])}
    v2 = max(tw["V1"], tw["V2"])
    half = tw["z"] * math.sqrt(v2)
    entry["conservative_V2"] = {"V": v2, "interval": [tw["point"] - half, tw["point"] + half],
                                "design_effect": math.sqrt(v2 / tw["V1"]) if tw["V1"] > 0 else None,
                                "rule": "max(V1, V2), reported only"}
    entry["players"] = player_distribution(player_idx)
    entry["method"] = "boot.twoway_interval: theta -/+ 1.96 sqrt(max(V1, VU))"
    return entry


def _kill_strata(n_kills: Any) -> np.ndarray:
    from studies.q_production_20261005 import qeval

    try:
        return qeval.kill_count_strata(np.asarray(n_kills))
    except ValueError as error:
        raise Blocked(f"kill counts must be positive integers: {error}") from error


def evaluate(y: Any, preds: Mapping[str, Any], match: Any, n_kills: Any, all_n_kills: Any, players: PlayerInput, *,
             s_scored: bool) -> dict[str, Any]:
    """Sections 5-7 on the non-tie rows: per-cell match-weighted Brier and AUC, theta = P1 contrast, the primary
    interval of the branch (two-way or match bootstrap), the decision, secondary (a) position share, (b) AUC and (c)
    kill-record strata (ties counted from all_n_kills)."""

    from studies.q_production_20261005 import qeval

    y = np.asarray(y, dtype=np.float64)
    match = np.asarray(match)
    strata = _kill_strata(n_kills)
    all_strata = _kill_strata(all_n_kills)
    cells = [c for c in CELLS if s_scored or c in (P1, PT)]
    metrics = {c: {m: qeval.evaluate(y, preds[c], match)["match_weighted"][m] for m in ("auc", "brier")} for c in cells}
    losses = {c: qeval.per_match_loss(y, preds[c], match) for c in cells}
    shared, aligned = qeval.align_cells(losses)
    contrasts = {"P1": dict(P1_CONTRAST), **({"S_effect_meanD": dict(S_CONTRAST)} if s_scored else {})}
    boot_all = qeval.paired_match_bootstrap(shared, aligned, contrasts, levels={name: LEVEL for name in contrasts},
                                            resamples=RESAMPLES, seed=SEED_BOOT)
    bootstrap = _boot_entry(boot_all["contrasts"]["P1"], boot_all)
    D = qeval.contrast_per_match(aligned, P1_CONTRAST)
    sub = None
    if players.available:
        pos = np.searchsorted(players.match_index, shared)
        if pos.size and (pos.max() >= players.match_index.size or not np.array_equal(players.match_index[pos], shared)):
            raise Blocked("a match with non-tie rows is not a row of the player matrix")
        sub = np.asarray(players.player_idx)[pos]
        twoway = _twoway_entry(D, sub)
        primary = {"method": "twoway", "point": twoway["point"], "interval": twoway["interval"], "level": LEVEL}
    else:
        twoway = {"available": False, "reason": players.reason, **({"record": players.record} if players.record else {})}
        primary = {"method": "match_bootstrap", "point": bootstrap["point"], "interval": bootstrap["interval"], "level": LEVEL}
    decision = decide(primary["interval"])
    by_interval = {"match_bootstrap": decide(bootstrap["interval"])}
    if players.available:
        by_interval["twoway"] = decide(twoway["interval"])
    if s_scored:
        s_boot = _boot_entry(boot_all["contrasts"]["S_effect_meanD"], boot_all)
        a: dict[str, Any] = {"scored": True, "point": s_boot["point"], "bootstrap": s_boot}
        if players.available:
            a["twoway"] = _twoway_entry(qeval.contrast_per_match(aligned, S_CONTRAST), sub)
            a["primary_interval"] = "twoway"
            a["reading"] = decide(a["twoway"]["interval"])
        else:
            a["primary_interval"] = "match_bootstrap"
            a["reading"] = decide(s_boot["interval"])
    else:
        a = {"scored": False, "reason": "채점 불가 (a position cell failed the reload check or scoring; PRESPEC 7 (a))"}
    c_strata: dict[str, Any] = {}
    for s in KILL_ORDER:
        mask = strata == s
        ties = int((all_strata == s).sum() - mask.sum())
        if not mask.any():
            c_strata[s] = {"rows": 0, "matches": 0, "ties": ties, "brier": None, "auc": None, "p1": None}
            continue
        sl = {k: qeval.per_match_loss(y[mask], preds[k][mask], match[mask]) for k in (P1, PT)}
        s_shared, s_aligned = qeval.align_cells(sl)
        sb = qeval.paired_match_bootstrap(s_shared, s_aligned, {"P1": dict(P1_CONTRAST)}, levels=LEVEL,
                                          resamples=RESAMPLES, seed=SEED_BOOT)
        sm = {k: qeval.evaluate(y[mask], preds[k][mask], match[mask])["match_weighted"] for k in (P1, PT)}
        c_strata[s] = {"rows": int(mask.sum()), "matches": int(s_shared.size), "ties": ties,
                       "brier": {k: v["brier"] for k, v in sm.items()}, "auc": {k: v["auc"] for k, v in sm.items()},
                       "p1": _boot_entry(sb["contrasts"]["P1"], sb)}
    return {"sign": SIGN, "rows": int(y.size), "matches": int(shared.size), "cells": metrics,
            "theta_hat": bootstrap["point"], "primary_interval": "twoway" if players.available else "match_bootstrap",
            "primary": primary, "decision": decision, "decision_text": DECISION_TEXT[decision],
            "decisions_by_interval": by_interval, "intervals_disagree": len(set(by_interval.values())) > 1,
            "twoway": twoway, "bootstrap": bootstrap,
            "secondary": {"a_position_share": a, "b_auc": {c: metrics[c]["auc"] for c in cells},
                          "c_kill_strata": {"strata": c_strata, "note": "descriptive; strata chosen by the kill record of "
                                                                        "the outcome window (not prediction-time scale)",
                                            "interval": "paired match bootstrap inside each stratum, 2,000, PCG64(20261011), 95 %"}},
            "rules": {"level": LEVEL, "z": Z, "resamples": RESAMPLES, "seed": SEED_BOOT, "contrasts": contrasts}}


# ----------------------------------------------------------------------------- labels loader, openings, rehearsals (A10, B1-B2)
def synthetic_labels(engagement_key: Any, match_index: Any) -> dict[str, np.ndarray]:
    """A10 fake labels: u = PCG64(20261014).random((n, 2)); tie = u1 < 0.098; y = u2 < 0.5; in the sealed file's format
    (KA = 0 for a tie, else +1 / -1; blue / red kills 1[KA > 0] / 1[KA < 0]; weights by the KA rule)."""

    keys = np.asarray(engagement_key)
    u = np.random.Generator(np.random.PCG64(SEED_SYNTH)).random((keys.size, 2))
    tie, positive = u[:, 0] < SYNTH_TIE, u[:, 1] < 0.5
    ka = np.where(tie, 0, np.where(positive, 1, -1)).astype(np.int32)
    eligible = ka != 0
    match = np.asarray(match_index, dtype=np.int64)
    m = np.bincount(match[eligible], minlength=int(match.max()) + 1).astype(np.float64)
    return {"engagement_key": keys, "ka": ka, "blue_kills": (ka > 0).astype(np.int32), "red_kills": (ka < 0).astype(np.int32),
            "eligible": eligible, "y_ka": (ka > 0).astype(np.float64),
            "weight_raw_ka": np.where(eligible, 1.0 / np.maximum(m[match], 1.0), 0.0)}


def opening_saved(paths: Paths, opening: Mapping[str, Any] | None) -> bool:
    if not opening or not Path(paths.openings).is_file():
        return False
    fields = ("experiment", "attempt", "utc", "lock_sha256")
    return any(all(o.get(k) == opening.get(k) for k in fields) for o in read_json(paths.openings).get("openings", []))


def append_opening(path: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Append the TEST step's entry to OPENINGS.json (existing entries unchanged) and make sure it is on disk."""

    path = Path(path)
    doc = read_json(path) if path.is_file() else {"rule": ("EXPOSURE_KO.md rule 1: every ToG experiment that opens sealed "
                                                         "labels appends experiment, UTC time and lock SHA-256"), "openings": []}
    before = list(doc.get("openings", []))
    attempts = [int(o.get("attempt", 0)) for o in before if o.get("experiment") == entry["experiment"]]
    if int(entry["attempt"]) != len(attempts) + 1:
        raise RuntimeError(f"attempt {entry['attempt']} does not follow the recorded attempts {attempts}")
    doc["openings"] = before + [dict(entry)]
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with open(pending, "wb") as stream:
        stream.write((json.dumps(jsonable(doc), indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(pending, path)
    except BaseException:
        try:
            pending.unlink()
        except OSError:
            pass
        raise
    if not opening_saved(SimpleNamespace(openings=path), entry):
        raise RuntimeError("the OPENINGS.json entry is not on disk")
    return before


def load_ka_labels(path: Path | str, rows: Mapping[str, Any], *, paths: Paths, opening: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The labels of the eligible (non-tie) rows of a KA label file in q-row order.  A file whose name contains SEALED
    is read only after the OPENINGS.json entry of the TEST step is on disk; the synthetic rehearsal file uses the same
    loader."""

    path = Path(path)
    sealed = any("SEALED" in part.upper() for part in path.parts)
    guard_read(path, sealed_ok=True)
    if sealed and not opening_saved(paths, opening):
        raise LabelAccessError("the E6 KA labels are read only after the OPENINGS.json entry of the TEST step is saved")
    data = load_npz(path, KA_COLUMNS, sealed_ok=sealed)
    if not np.array_equal(np.asarray(data["engagement_key"]), np.asarray(rows["engagement_key"])):
        raise Blocked("the KA label file is not in q-row order")
    eligible = np.asarray(data["eligible"], dtype=bool)
    idx = np.flatnonzero(eligible)
    y = np.asarray(data["y_ka"], dtype=np.float64)[idx]
    if not np.isin(y, (0.0, 1.0)).all():
        raise Blocked("KA labels are not binary on the eligible rows")
    match = np.asarray(rows["match_index"])[idx]
    _, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
    weights_ok = bool(np.array_equal(np.asarray(data["weight_raw_ka"], dtype=np.float64)[idx], 1.0 / counts[inverse]))
    out = {"idx": idx, "y": y, "rows": int(eligible.size), "ties": int((~eligible).sum()), "sha256": sha256_file(path),
           "weights_equal_match_weights": weights_ok}
    del data
    return out


def e6_inputs(paths: Paths) -> dict[str, Any]:
    """Every label-free input of the evaluation: the row table, the cell scores in row order, the player input of the
    branch and whether the four S cells were scored (B0 3 checks)."""

    q = paths.ext("q_inputs")
    rows_path = q / "Q_ROWS_h90.npz"
    meta = read_json(q / "Q_ROWS_h90.json")
    if sha256_file(rows_path) != meta["rows_sha256"]:
        raise Blocked("the E6 row table differs from Q_ROWS_h90.json")
    rows = load_npz(rows_path, ["engagement_key", "match_index", "canonical_match_key", "eval_n_kills", "y"])
    keys = np.asarray(rows["engagement_key"])
    if np.unique(keys).size != keys.size:
        raise Blocked("engagement keys of the E6 row table repeat")
    if (np.asarray(rows["y"]) != -1).any():
        raise Blocked("the E6 row table holds a label value")
    if int(np.asarray(rows["eval_n_kills"]).min()) < 1:
        raise Blocked("an E6 kill count is below 1")
    score_stage = read_json(stage_record_path(paths, "score"))
    s_scored = bool(score_stage["counts"]["s_scored"])
    preds = {}
    for key in (CELLS if s_scored else (P1, PT)):
        meta_c = read_json(paths.out / "scores" / f"{key}.json")
        path = paths.out / "scores" / f"{key}.npz"
        if sha256_file(path) != meta_c["pred_sha256"]:
            raise Blocked(f"{path} differs from its record")
        with np.load(guard_read(path), allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        missing = [n for n in ("row", "engagement_key", "logit", "p") if n not in arrays]
        if missing:
            raise Blocked(f"{key}: score arrays missing {missing}")
        row, logit, p = (np.asarray(arrays[n]) for n in ("row", "logit", "p"))
        if row.ndim != 1 or logit.ndim != 1 or p.ndim != 1 or not row.size == logit.size == p.size == keys.size:
            raise Blocked(f"{key}: score arrays are not one-dimensional of the row-table length")
        if row.dtype.kind not in "iu" or logit.dtype.kind != "f" or p.dtype.kind != "f":
            raise Blocked(f"{key}: score dtypes are not integer rows with floating logit and p")
        if not (np.isfinite(logit).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
            raise Blocked(f"{key}: logit or p not finite, or p outside [0, 1]")
        if not np.array_equal(row, np.arange(keys.size)) or not np.array_equal(np.asarray(arrays["engagement_key"]), keys):
            raise Blocked(f"{key}: score rows are not the row-table order")
        preds[key] = np.asarray(p, dtype=np.float64)
    players, branch = load_player_input(paths, rows)
    return {"rows": rows, "preds": preds, "players": players, "branch": branch, "s_scored": s_scored,
            "rows_sha256": meta["rows_sha256"]}


def load_player_input(paths: Paths, rows: Mapping[str, Any]) -> tuple[PlayerInput, dict[str, Any]]:
    unavailable = paths.out / "PLAYERS_E6_UNAVAILABLE.json"
    if unavailable.is_file():
        doc = read_json(unavailable)
        return PlayerInput(False, "player_identification_unavailable", None, None, str(unavailable)), doc["branch"]
    doc = read_json(paths.out / "PLAYERS_E6.json")
    branch = doc["branch"]
    if branch["branch"] == "나":
        return PlayerInput(False, "incomplete_identification_over_1pct", None, None, None), branch
    if branch["branch"] != "가":
        raise Blocked(f"unknown branch {branch['branch']}")
    if sha256_file(paths.out / "PLAYERS_E6.npz") != doc["npz_sha256"]:
        raise Blocked("PLAYERS_E6.npz differs from PLAYERS_E6.json")
    data = load_npz(paths.out / "PLAYERS_E6.npz", ["match_index", "canonical_match_key", "player_idx"])
    match = np.asarray(rows["match_index"])
    unique, first = np.unique(match, return_index=True)
    if not np.array_equal(np.asarray(data["match_index"]), unique) or \
            not np.array_equal(np.asarray(data["canonical_match_key"]), np.asarray(rows["canonical_match_key"])[first]):
        raise Blocked("the player matrix is not aligned with the E6 row table (match order or canonical keys)")
    return PlayerInput(True, None, np.asarray(data["match_index"]), np.asarray(data["player_idx"]), None), branch


def analyse_core(inputs: Mapping[str, Any], labels: Mapping[str, Any]) -> dict[str, Any]:
    """Sections 5-7 on the eligible rows of a label file (the synthetic rehearsal and part B call this)."""

    idx = labels["idx"]
    rows = inputs["rows"]
    preds = {k: np.asarray(v)[idx] for k, v in inputs["preds"].items()}
    out = evaluate(labels["y"], preds, np.asarray(rows["match_index"])[idx], np.asarray(rows["eval_n_kills"])[idx],
                   np.asarray(rows["eval_n_kills"]), inputs["players"], s_scored=inputs["s_scored"])
    out["scale"] = {"q_rows": int(labels["rows"]), "ties": int(labels["ties"]), "non_tie_rows": int(idx.size),
                    "matches": out["matches"]}
    out["branch"] = {k: inputs["branch"].get(k) for k in ("branch", "primary_interval", "twoway_unavailable_reason",
                                                           "incomplete_identification")}
    return out


def qdev_inputs(paths: Paths) -> dict[str, Any]:
    """Q_DEV rehearsal inputs: the already opened Q_DEV KA labels (KA_LABELS.npz, Q_DEV eligible rows only) and the q_ka_1
    cells' saved pred_q_dev.npz, in prediction row order (kill_label.py:316-328)."""

    prepare = read_json(paths.q_ka / "KA_PREPARE.json")
    if sha256_file(paths.q_ka / "KA_LABELS.npz") != prepare.get("ka_labels_sha256"):
        raise Blocked("KA_LABELS.npz differs from KA_PREPARE.json")
    rows = load_npz(paths.q_run / "Q_ROWS_h90.npz", ["engagement_key", "match_index", "role", "eval_n_kills"])
    ka = load_npz(paths.q_ka / "KA_LABELS.npz", ["engagement_key", "eligible", "y_ka"])
    if not np.array_equal(np.asarray(ka["engagement_key"]), np.asarray(rows["engagement_key"])):
        raise Blocked("KA_LABELS.npz is not aligned with the q_run_1 row table")
    split_rows = np.flatnonzero(np.asarray(rows["role"]) == 3)
    eligible = np.asarray(ka["eligible"], dtype=bool)[split_rows]
    idx = split_rows[eligible]
    y_all = np.asarray(ka["y_ka"], dtype=np.float64)[idx]
    del ka
    preds, rows_ref = {}, None
    for key in CELLS:
        result = read_json(paths.ka_cell(key) / "RESULT.json")
        path = paths.ka_cell(key) / "pred_q_dev.npz"
        if sha256_file(path) != ((result.get("scores") or {}).get("q_dev") or {}).get("pred_sha256"):
            raise Blocked(f"{key}: pred_q_dev.npz differs from RESULT.json")
        data = load_npz(path, ["row", "p"])
        row = np.asarray(data["row"], dtype=np.int64)
        if rows_ref is None:
            rows_ref = row
        elif not np.array_equal(row, rows_ref):
            raise Blocked(f"{key}: Q_DEV prediction rows differ between cells")
        preds[key] = np.asarray(data["p"], dtype=np.float64)
    if not np.array_equal(np.sort(rows_ref), idx):
        raise Blocked("the Q_DEV prediction rows are not the eligible Q_DEV rows")
    position = np.searchsorted(idx, rows_ref)
    kills = np.asarray(rows["eval_n_kills"])
    return {"y": y_all[position], "preds": preds, "match": np.asarray(rows["match_index"])[rows_ref],
            "n_kills": kills[rows_ref], "all_n_kills": kills[split_rows], "split_rows": int(split_rows.size),
            "ties": int((~eligible).sum())}


def compare_ka_report(result: Mapping[str, Any], block: Mapping[str, Any], tol: float = QDEV_TOL) -> dict[str, Any]:
    """The Q_DEV rehearsal against KA_REPORT splits.q_dev.ka_trained_on_ka: per-cell AUC and Brier, P1 and
    S_effect_meanD point and match-bootstrap interval."""

    items = []
    for cell in CELLS:
        for metric in ("auc", "brier"):
            items.append((f"metrics/{cell}/{metric}", result["cells"][cell][metric], block["metrics"][cell][metric]))
    for name, mine in (("P1", result["bootstrap"]), ("S_effect_meanD", result["secondary"]["a_position_share"]["bootstrap"])):
        ref = block["contrasts"]["contrasts"][name]
        items.append((f"contrasts/{name}/point", mine["point"], ref["point"]))
        for i in (0, 1):
            items.append((f"contrasts/{name}/interval/{i}", mine["interval"][i], ref["interval"][i]))
    diffs = {label: abs(float(a) - float(b)) for label, a, b in items}
    worst = max(diffs, key=diffs.get)
    return {"items": len(items), "max_abs_diff": diffs[worst], "worst": worst, "tolerance": tol,
            "passed": bool(diffs[worst] <= tol), "differences": diffs}


def qdev_rehearsal(paths: Paths) -> dict[str, Any]:
    d = qdev_inputs(paths)
    result = evaluate(d["y"], d["preds"], d["match"], d["n_kills"], d["all_n_kills"],
                      PlayerInput(False, "no_player_ids", None, None, None), s_scored=True)
    report = read_json(paths.q_ka / "KA_REPORT.json")
    block = report["splits"]["q_dev"]["ka_trained_on_ka"]
    comparison = compare_ka_report(result, block)
    counts = {"rows": int(d["y"].size), "matches": result["matches"], "split_rows": d["split_rows"], "ties": d["ties"]}
    comparison["counts_equal_ka_report"] = (counts["rows"], counts["matches"]) == (report["splits"]["q_dev"]["rows"],
                                                                                 report["splits"]["q_dev"]["matches"])
    comparison["passed"] = bool(comparison["passed"] and comparison["counts_equal_ka_report"])
    return {"result": {"evaluation": result, "counts": counts}, "ka_report_comparison": comparison}


def synth_rehearsal(paths: Paths, inputs: Mapping[str, Any]) -> dict[str, Any]:
    labels = load_ka_labels(paths.out / KA_SYNTH_NAME, inputs["rows"], paths=paths)
    return {"result": {"evaluation": analyse_core(inputs, labels), "labels_sha256": labels["sha256"],
                       "weights_equal_match_weights": labels["weights_equal_match_weights"]}}


def canonical(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def warm_up() -> None:
    """Run the evaluation once on small synthetic rows (both branches) so that every module it needs is loaded."""

    rng = np.random.default_rng(20261014)
    per = np.tile([1, 2, 3, 4], 25)
    match = np.repeat(np.arange(per.size) * 3 + 1, per)
    n = match.size
    y = np.tile([0.0, 1.0], n // 2 + 1)[:n]
    rng.shuffle(y)
    preds = {key: 1.0 / (1.0 + np.exp(-((0.3 + 0.1 * (i % 3)) * (2 * y - 1) + rng.normal(0, 1, n))))
             for i, key in enumerate(CELLS)}
    kills = np.tile(np.arange(1, 7, dtype=np.int16), n // 6 + 1)[:n]
    players = np.asarray([rng.permutation(200)[:10] for _ in range(per.size)], dtype=np.int64)
    evaluate(y, preds, match, kills, kills, PlayerInput(True, None, np.unique(match), players, None), s_scored=True)
    evaluate(y, preds, match, kills, kills, PlayerInput(False, "warm_up", None, None, None), s_scored=False)


def rehearse_body(ctx: StageContext) -> dict[str, Any]:
    """A10: Q_DEV rehearsal (KA_REPORT within 1e-9) and the fake-label rehearsal on the real E6 scores, rows and player
    input; each written once (the lock binds both; part B recomputes them exactly)."""

    paths, log = ctx.paths, ctx.log
    require_stage(paths, "labels")
    qdev_path, synth_path = paths.out / "E6_QDEV_REHEARSAL.json", paths.out / "E6_SYNTH_REHEARSAL.json"
    synth_labels = paths.out / KA_SYNTH_NAME
    if not qdev_path.exists():
        tick = time.perf_counter()
        doc = qdev_rehearsal(paths)
        doc["run"] = {"utc": utc(), "seconds": round(time.perf_counter() - tick, 1), "seed": SEED_BOOT,
                      "inputs_sha256": {"KA_LABELS.npz": sha256_file(paths.q_ka / "KA_LABELS.npz"),
                                        "KA_REPORT.json": sha256_file(paths.q_ka / "KA_REPORT.json"),
                                        "Q_ROWS_h90.npz (q_run_1)": sha256_file(paths.q_run / "Q_ROWS_h90.npz")},
                      "code_sha256": sha256_file(paths.repo / CODE_REL),
                      "note": "re-reads the already opened Q_DEV KA labels (Q_DEV eligible rows only); no TEST row"}
        write_json_once(qdev_path, doc)
    qdev = read_json(qdev_path)
    if not qdev["ka_report_comparison"]["passed"]:
        raise Blocked(f"the Q_DEV rehearsal differs from KA_REPORT: {qdev['ka_report_comparison']['worst']} "
                      f"{qdev['ka_report_comparison']['max_abs_diff']}")
    log(f"[rehearse] Q_DEV rehearsal equals KA_REPORT (max |diff| {qdev['ka_report_comparison']['max_abs_diff']})")
    inputs = e6_inputs(paths)
    if not synth_labels.exists():
        write_npz_once(synth_labels, synthetic_labels(inputs["rows"]["engagement_key"], inputs["rows"]["match_index"]))
    if not synth_path.exists():
        tick = time.perf_counter()
        doc = synth_rehearsal(paths, inputs)
        doc["run"] = {"utc": utc(), "seconds": round(time.perf_counter() - tick, 1), "seeds": [SEED_SYNTH, SEED_BOOT],
                      "code_sha256": sha256_file(paths.repo / CODE_REL),
                      "note": "fake labels (meaningless); removes post-opening crashes (Ruling 55)"}
        write_json_once(synth_path, doc)
    log("[rehearse] fake-label rehearsal written")
    return {"counts": {"qdev_rows": qdev["result"]["counts"]["rows"], "qdev_matches": qdev["result"]["counts"]["matches"],
                       "qdev_max_abs_diff": qdev["ka_report_comparison"]["max_abs_diff"], "branch": inputs["branch"]["branch"]},
            "outputs": {"E6_QDEV_REHEARSAL.json": qdev_path, "E6_SYNTH_REHEARSAL.json": synth_path, KA_SYNTH_NAME: synth_labels},
            "seed": SEED_SYNTH}


# ----------------------------------------------------------------------------- A11-A12 lock
def lock_versions(out: Path) -> list[Path]:
    found, base = [], Path(out) / f"{LOCK_BASE}.json"
    if base.exists():
        found.append(base)
        k = 1
        while (Path(out) / f"{LOCK_BASE}_r{k}.json").exists():
            found.append(Path(out) / f"{LOCK_BASE}_r{k}.json")
            k += 1
    return found


def latest_lock(out: Path) -> Path | None:
    versions = lock_versions(out)
    return versions[-1] if versions else None


def previous_locks_record(paths: Paths) -> list[dict[str, Any]]:
    """Every lock version of this folder with its SHA-256 and whether an E6 opening used it."""

    opened = {o.get("lock_sha256") for o in e6_openings(paths)}
    out = []
    for path in lock_versions(paths.out):
        sha = sha256_file(path)
        out.append({"path": str(path), "sha256": sha, "opened": sha in opened, "superseded_before_any_look": sha not in opened})
    return out


def lock_step(paths: Paths) -> str | None:
    """The stage of the next lock version (PRESPEC 8.1 A12, earlier versions stay): "lock" writes E6_LOCK.json,
    "lock_r<k>" writes E6_LOCK_r<k>.json when the committed e6_run.py or its test file differ from the latest lock's;
    None when the latest lock binds them.  Refused after an E6 opening of a lock of this folder (Ruling 31)."""

    versions = lock_versions(paths.out)
    if not versions:
        return "lock"
    verify = read_json(versions[-1]).get("verify") or {}
    same = all((verify.get(f"code:{rel}") or {}).get("sha256") == sha256_file(paths.repo / rel) for rel in (CODE_REL, TEST_REL))
    if same:
        return None
    if any(p["opened"] for p in previous_locks_record(paths)):
        raise Blocked("a lock of this folder was used by an opening; a rerun needs a new folder and lock (Ruling 31)")
    return f"lock_r{len(versions)}"


def lock_files(paths: Paths, branch: str) -> dict[str, Path]:
    """Every file the lock binds by SHA-256 (PRESPEC 8.1 A12); labels_hash_only entries are hashed, never opened."""

    out, rep = paths.out, paths.external_reports
    files: dict[str, Path] = {
        "provenance:EXTERNAL_SCORE_LOCK.json": rep / "EXTERNAL_SCORE_LOCK.json", "provenance:E2a_REPORT.json": rep / "E2a_REPORT.json",
        "provenance:E1_REPORT.json": rep / "E1_REPORT.json", "provenance:parent_run.json": paths.parent_run}
    for path in read_json(rep / "E1_REPORT.json")["code"]["old_repo_builder_modules"]:
        files[f"old_repo_builder:{path}"] = Path(path)
    files.update({"input:SAMPLE_MANIFEST.json": paths.sample_manifest, "input:SAMPLE_MANIFEST.sha256": rep / "SAMPLE_MANIFEST.sha256",
                  "input:E6_SAMPLE_MANIFEST.json": paths.e6_manifest, "input:E6_SAMPLE_MANIFEST.sha256": paths.e6_manifest_sha,
                  f"input:BUILD_MANIFEST_{DATASET}.json": out / f"BUILD_MANIFEST_{DATASET}.json",
                  f"input:POSTBUILD_CHECK_{DATASET}.json": out / f"POSTBUILD_CHECK_{DATASET}.json",
                  "input:patch_index": paths.patch_index, "input:ENV_CENSUS_before.json": out / "ENV_CENSUS_before.json",
                  "input:ENV_CENSUS_after.json": out / "ENV_CENSUS_after.json", "input:V_SCORING_DEVICE.json": out / "V_SCORING_DEVICE.json"})
    for path in sorted(out.glob("ENV_CENSUS_after_r*.json")):                # one census per later lock version
        files[f"input:{path.name}"] = path
    for path in sorted(out.glob("V_SCORING_DEVICE_r*.json")):
        files[f"input:{path.name}"] = path
    stage_files = {"audit": ("AUDIT_MANIFEST.json", "row_ledger.jsonl"),
                   "bank": ("RUN_MANIFEST.json", "EXCLUDED.json", "BANK_ENTRIES.jsonl", "FIELD_MANIFEST.json", "FIELD_SUPPORT_COUNTS.json"),
                   "preparation": ("RUN_MANIFEST.json",), "items": ("RUN_MANIFEST.json",),
                   "projection": ("RUN_MANIFEST.json", "TEAM_KILLS.npy"), "history": ("PLAN.json", "ACCEPTANCE.json"),
                   "labels": ("ENGAGEMENTS.npz", "ENGAGEMENTS_MANIFEST.json", "LABELS_PUBLIC_h90.npz", "LABELS_MANIFEST_EXTERNAL.json",
                              "QUERIES_final.npy"),
                   "q_inputs": ("Q_ROWS_h90.npz", "Q_ROWS_h90.json", "XY.npy", "XY.json")}
    for stage, names in stage_files.items():
        for name in names:
            files[f"input:{stage}/{name}"] = paths.ext(stage) / name
    store = paths.ext("q_inputs") / "raw" / "store"
    if store.is_dir():
        for path in sorted(p for p in store.rglob("*") if p.is_file()):
            files[f"input:q_inputs/raw/store/{path.relative_to(store).as_posix()}"] = path
    if branch in ("가", "나"):
        files["input:PLAYERS_E6.npz"], files["input:PLAYERS_E6.json"] = out / "PLAYERS_E6.npz", out / "PLAYERS_E6.json"
    else:
        files["input:PLAYERS_E6_UNAVAILABLE.json"] = out / "PLAYERS_E6_UNAVAILABLE.json"
    labels_dir = paths.ext("labels")
    hash_only = [out / KA_SEALED_NAME, labels_dir / f"EXTERNAL_LABELS_SEALED_{DATASET}.npz"]
    if labels_dir.is_dir():
        hash_only += sorted(p for p in labels_dir.glob("PRED_final*") if p.is_file())
        if (labels_dir / "PRED_final").is_dir():
            hash_only += sorted(p for p in (labels_dir / "PRED_final").rglob("*") if p.is_file())
    for path in hash_only:
        files[f"labels_hash_only:{path.relative_to(out).as_posix()}"] = path
    for key in CELLS:
        for name in ("RESULT.json", "model_refit.pkl", "pred_q_dev.npz"):
            files[f"model:{key}:{name}"] = paths.ka_cell(key) / name
        result = paths.ka_cell(key) / "RESULT.json"
        if result.is_file():
            files[f"model:{key}:q_run_1_source_SPEC.json"] = Path(read_json(result)["source_dir"]) / "SPEC.json"
    context = paths.q_run / "inputs" / "main_h90"
    files.update({"context:CONTEXT.json": context / "REFIT_TRAIN" / "CONTEXT.json",
                  "context:CONTEXT.npz": context / "REFIT_TRAIN" / "CONTEXT.npz",
                  "context:bundle/MANIFEST.json": context / "REFIT_TRAIN" / "bundle" / "MANIFEST.json",
                  "context:PREPARED.json": context / "PREPARED.json",
                  "context:V_LOCK.json": paths.training / "run_3" / "final" / "V_LOCK.json"})
    for path in sorted((out / "scores").glob("*")) if (out / "scores").is_dir() else []:
        if path.is_file():
            files[f"score:{path.name}"] = path
    files.update({"rehearsal:KA_LABELS.npz": paths.q_ka / "KA_LABELS.npz", "rehearsal:KA_PREPARE.json": paths.q_ka / "KA_PREPARE.json",
                  "rehearsal:KA_REPORT.json": paths.q_ka / "KA_REPORT.json",
                  "rehearsal:E6_QDEV_REHEARSAL.json": out / "E6_QDEV_REHEARSAL.json",
                  "rehearsal:E6_SYNTH_REHEARSAL.json": out / "E6_SYNTH_REHEARSAL.json", f"rehearsal:{KA_SYNTH_NAME}": out / KA_SYNTH_NAME,
                  "labels:KA_LABELS_E6_MANIFEST.json": out / "KA_LABELS_E6_MANIFEST.json", "labels:KA_GATE.json": out / "KA_GATE.json"})
    for path in sorted(out.glob("STAGE_*.json")):
        files[f"{'stage_modules' if path.name.startswith('STAGE_MODULES_') else 'stage'}:{path.name}"] = path
    for path in sorted((out / "stage_runs").rglob("*.json")) if (out / "stage_runs").is_dir() else []:
        files[f"stage_run:{path.relative_to(out).as_posix()}"] = path
    starts, ends = worker_record_files(out / "worker_modules")
    for path in starts + ends:
        files[f"worker_record:{path.relative_to(out).as_posix()}"] = path
    superseded = out / "superseded"                                    # Ruling 69: outputs of crashed attempts, hash only
    for path in sorted(p for p in superseded.rglob("*") if p.is_file()) if superseded.is_dir() else []:
        files[f"superseded_hash_only:{path.relative_to(out).as_posix()}"] = path
    files[f"code:{CODE_REL}"], files[f"code:{TEST_REL}"] = paths.repo / CODE_REL, paths.repo / TEST_REL
    return files


def verify_entries(files: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """{label: {path, sha256, bytes}} hashed now (byte hashes only; sealed files included)."""

    out = {}
    for label, path in files.items():
        path = Path(path)
        out[label] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return out


def load_stage_runs(paths: Paths) -> dict[str, dict[str, Any]]:
    runs = {}
    root = paths.out / "stage_runs"
    for path in sorted(root.rglob("*.json")) if root.is_dir() else []:
        doc = read_json(path)
        runs[doc["run_id"]] = {"stage": doc["stage"], "pid": doc["pid"], "status": doc["status"], "error": doc.get("error"),
                               "executors": {no: e["submitted"] for no, e in (doc.get("executors") or {}).items()}}
    return runs


def attempts_record(paths: Paths, stages: Iterable[str]) -> dict[str, Any]:
    """2026-10-14 운영 정정 (Ruling 69): the final attempt of each stage (the run id of its STAGE record) must reconcile
    completely; every other attempt found in stage_runs or worker_modules is superseded, listed with its reason and its
    records, and left out of reconciliation."""

    final = {}
    for stage in stages:
        path = stage_record_path(paths, stage)
        doc = read_json(path) if path.is_file() else {}
        if doc.get("status") != "completed":
            raise Blocked(f"the {stage} stage is not complete")
        final[stage] = str(doc["run_id"])
    runs = load_stage_runs(paths)
    rec = reconcile(paths.out / "worker_modules", runs, run_ids=set(final.values()))
    starts, ends = worker_record_files(paths.out / "worker_modules")
    by_run: dict[str, list[str]] = {}
    stage_of: dict[str, str] = {}
    for path in starts + ends:
        run = record_run_id(path)
        by_run.setdefault(run, []).append(path.relative_to(paths.out).as_posix())
        stage_of.setdefault(run, path.relative_to(paths.out / "worker_modules").parts[0])
    superseded = []
    for run in sorted((set(runs) | set(by_run)) - set(final.values())):
        info = runs.get(run)
        if info is None:
            reason = "no stage run record: the attempt ended without one (for example the host ended the job)"
        elif info["status"] == "completed" and str(info["stage"]).startswith("lock"):
            reason = ("an earlier lock version: its lock file stays, and previous_locks says whether an opening used it "
                      "(fix round 1)")
        elif info["status"] == "completed":
            reason = "completed its run record but no stage record was written; superseded by a rerun"
        else:
            reason = f"{info['status']}: {info.get('error')}"
        stage = info["stage"] if info else stage_of.get(run)
        superseded.append({"stage": stage, "run_id": run, "status": info["status"] if info else None, "reason": reason,
                           "worker_records": sorted(by_run.get(run, [])),
                           "stage_run_record": f"stage_runs/{stage}/{run}.json" if info else None})
    return {"final": final, "superseded": superseded,
            "reconciliation": {k: v for k, v in rec.items() if k not in ("modules", "modules_by_run")},
            "rule": ("Ruling 69: each stage's final attempt reconciles completely; superseded attempts (died before the "
                     "lock) keep their records, are listed here with their reason and are left out of reconciliation")}


def stage_code_versions(paths: Paths, modules_docs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per stage, the e6_run.py version it ran and the version its module record holds (ka_posthoc.code_revision rule).

    run_stage starts a real stage only when e6_run.py equals HEAD, so the stage process and the workers it spawned
    loaded the last commit of e6_run.py made before the stage started (`commit_at_start`).  Module records hash every
    loaded file when they are written, at the end of the stage (a worker's record at its exit), so a file edited on disk
    during a stage appears there in its end-of-stage version (`record_sha256`, `record_commit`) and the stage is listed
    in `edited_during_stage` (2026-10-14: bank ran 81fbb37 while the Ruling 69 edit, committed as 86e1541 after bank
    ended, was made)."""

    code_path = str((paths.repo / CODE_REL).resolve())
    now = sha256_file(paths.repo / CODE_REL)
    history = []                                                    # (commit, committer time, blob SHA-256), newest first
    for line in _git(paths.repo, "log", "--format=%H %cI", "--", CODE_REL).stdout.decode("ascii", "replace").splitlines():
        if not line.strip():
            continue
        commit, when = line.split()
        blob = _git(paths.repo, "cat-file", "blob", f"{commit}:{CODE_REL}")
        history.append((commit, datetime.fromisoformat(when),
                        hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None))
    blobs: dict[str, str] = {}
    for commit, _, sha in history:
        if sha is not None:
            blobs.setdefault(sha, commit)
    per_stage, problems, edited = {}, [], []
    for stage, doc in modules_docs.items():
        sha = (doc.get("stage_process_modules") or {}).get(code_path)
        record = stage_record_path(paths, stage)
        started = read_json(record).get("started_utc") if record.is_file() else None
        at_start = None
        if started:
            moment = datetime.fromisoformat(started)
            at_start = next(((c, s) for c, when, s in history if when <= moment), None)
        entry = {"record_sha256": sha, "record_commit": blobs.get(sha), "equal_now": sha == now, "started_utc": started,
                 "commit_at_start": at_start[0] if at_start else None, "sha256_at_start": at_start[1] if at_start else None,
                 "record_equals_start": bool(at_start and sha is not None and sha == at_start[1])}
        per_stage[stage] = entry
        if sha is None or entry["record_commit"] is None:
            problems.append(f"{stage}: e6_run.py version {sha} in the module record is not a commit in git")
        if at_start is None:
            problems.append(f"{stage}: no commit of e6_run.py was made before the stage started ({started})")
        elif not entry["record_equals_start"]:
            edited.append(stage)
    return {"now_sha256": now, "stages": per_stage, "problems": problems, "edited_during_stage": sorted(edited),
            "rule": ("each stage ran the last e6_run.py commit made before it started (run_stage refuses to start a real "
                     "stage unless the file equals HEAD); module records hash files at the end of the stage, so a stage "
                     "whose record holds another version had the file edited on disk during it and is listed in "
                     "edited_during_stage; later commits only add later steps' functions and the attempt bookkeeping")}


def lock_body(ctx: StageContext, reason: str | None = None) -> dict[str, Any]:
    paths, log = ctx.paths, ctx.log
    for stage in (*STEP_ORDER, *ANALYSIS_STEPS):
        require_stage(paths, stage)
    suffix = "" if ctx.stage == "lock" else "_" + ctx.stage[len("lock_"):]
    target = paths.out / f"{LOCK_BASE}{suffix}.json"
    if target != next_free(paths.out / f"{LOCK_BASE}.json"):
        raise Blocked(f"the {ctx.stage} step would write {target.name}, but the next free lock is "
                      f"{next_free(paths.out / f'{LOCK_BASE}.json').name}")
    previous_locks = previous_locks_record(paths)
    if any(p["opened"] for p in previous_locks):
        raise Blocked("a lock of this folder was used by an opening; a rerun needs a new folder and lock (Ruling 31)")
    before_path, after_path = paths.out / "ENV_CENSUS_before.json", paths.out / f"ENV_CENSUS_after{suffix}.json"
    if not after_path.exists():
        write_json_once(after_path, census(paths), list_key="external_data_entries")
    census_cmp = compare_census(read_json(before_path), read_json(after_path))
    if not census_cmp["equal"]:
        raise Blocked(f"ENV_CENSUS_after differs from ENV_CENSUS_before: {census_cmp['differences']}")
    attempts = attempts_record(paths, STEP_ORDER)
    rec = attempts["reconciliation"]
    if not rec["passed"]:
        raise Blocked(f"the final attempts' worker records do not reconcile: {rec['problems'][:5]}")
    modules_docs = {p.name[len("STAGE_MODULES_"):-len(".json")]: read_json(p) for p in sorted(paths.out.glob("STAGE_MODULES_*.json"))}
    warm_up()
    lock_modules = loaded_files()
    recorded: dict[str, set[str]] = {}
    for doc in modules_docs.values():
        for name, sha in (doc.get("merged_modules") or {}).items():
            recorded.setdefault(name, set()).add(sha)
    for name, sha in lock_modules.items():
        recorded.setdefault(name, set()).add(sha)
    code_now = {name: sha256_file(name) for name in sorted(recorded) if Path(name).is_file()}
    code_path = str((paths.repo / CODE_REL).resolve())
    drift = sorted(name for name, shas in recorded.items() if name != code_path and shas != {code_now.get(name)})
    if drift:
        raise Blocked(f"modules changed after a stage loaded them: {drift[:5]}")
    acc = acceptance(code_now, load_refs(paths))
    if not acc["passed"]:
        raise Blocked(f"acceptance rule fails: {acc['problems'][:5]}")
    versions = stage_code_versions(paths, modules_docs)
    if versions["problems"]:
        raise Blocked(f"stage code versions: {versions['problems']}")
    ok, ident = code_committed(paths)
    test_status = _git(paths.repo, "status", "--porcelain", "--untracked-files=all", "--", TEST_REL).stdout.decode("utf-8", "replace")
    if not ok or test_status.strip():
        raise Blocked(f"commit {CODE_REL} and {TEST_REL} before the lock: {ident['status_porcelain']!r} {test_status!r}")
    ws_paths = sorted(Path(n).resolve().relative_to(paths.repo.resolve()).as_posix() for n in code_now if _under(n, paths.repo))
    porcelain = _git(paths.repo, "status", "--porcelain", "--untracked-files=all", "--", *ws_paths).stdout.decode("utf-8", "replace")
    players = read_json(paths.out / "PLAYERS_E6_UNAVAILABLE.json") if (paths.out / "PLAYERS_E6_UNAVAILABLE.json").is_file() \
        else read_json(paths.out / "PLAYERS_E6.json")
    branch = players["branch"]
    files = lock_files(paths, branch["branch"])
    tick = time.perf_counter()
    verify = verify_entries(files)
    log(f"[lock] {len(verify)} files hashed in {time.perf_counter() - tick:.0f} s")
    score_stage = read_json(stage_record_path(paths, "score"))
    import scipy
    import lightgbm
    import torch

    commit = prespec_commit(paths)
    sha, size, _ = prespec_blob(paths, commit)
    unchanged = _git(paths.repo, "diff", "--quiet", commit, "--", paths.prespec_rel).returncode == 0
    if not unchanged:
        raise Blocked(f"the PRESPEC working copy differs from {commit}")
    numstat = _git(paths.repo, "diff", "--numstat", PRESPEC_APPROVED, commit, "--", paths.prespec_rel).stdout.decode("utf-8", "replace")
    score_v_record = read_json(Path(read_json(stage_record_path(paths, "score_v"))["outputs"]["V_SCORING_DEVICE.json"]["path"]))
    starts, ends = worker_record_files(paths.out / "worker_modules")
    payload = {
        "version": VERSION, "experiment": EXPERIMENT, "locked_utc": utc(),
        "prespec": {"path": paths.prespec_rel, "commit": commit, "approved_text_commit": PRESPEC_APPROVED,
                    "approval_line_commit": PRESPEC_APPROVAL_COMMIT, "blob_sha256": sha, "blob_bytes": size,
                    "working_copy_unchanged": unchanged, "diff_numstat_since_approved_text": numstat,
                    "notes_after_approval": "header approval line (c4d4ee8) and the dated 2026-10-14 운영 정정 (Ruling 69)",
                    "rule": "SHA-256 of `git cat-file blob <commit>:<path>` stdout; OPENINGS prespec_sha256 is this value"},
        "lock_version": {"step": ctx.stage, "file": target.name, "reason": reason,
                         "rule": "PRESPEC 8.1 A12: a changed lock gets a new version; earlier versions stay"},
        "previous_locks": previous_locks,
        "code": {"files": code_now, "identity": ident, "status_porcelain_code_paths": porcelain, "stage_versions": versions,
                 "rebindings": rebinding_list(paths.out), "start_checks": read_json(before_path)["checks"]},
        "module_records": {"stages": {stage: {"stage_process": doc.get("stage_process_modules"), "workers": doc.get("worker_modules"),
                                              "reconciliation": doc.get("reconciliation"),
                                              "acceptance": {k: v for k, v in (doc.get("acceptance") or {}).items() if k != "files"}}
                                      for stage, doc in modules_docs.items()},
                           "reconciliation_final_attempts": rec, "final_attempts": attempts["final"],
                           "superseded_attempts": attempts["superseded"], "attempts_rule": attempts["rule"],
                           "acceptance_all": {k: v for k, v in acc.items() if k != "files"}, "lock_process": lock_modules,
                           "worker_records": sorted(p.relative_to(paths.out).as_posix() for p in starts + ends)},
        "provenance": {"records": load_refs(paths).records,
                       "old_repo_builder_modules": {k[len("old_repo_builder:"):]: verify[k]["sha256"] for k in verify
                                                    if k.startswith("old_repo_builder:")},
                       "old_repo_files_outside_30": {p: v for p, v in acc["files"].items() if v["rule"] == "parent_record"}},
        "v_scoring_boundary": {"device_record": score_v_record, "environment": score_v_record.get("environment")},
        "census": census_cmp,
        "labels_hash_only": {k: v for k, v in verify.items() if k.startswith("labels_hash_only:")},
        "rules": {"primary_interval": branch["primary_interval"], "twoway_unavailable_reason": branch["twoway_unavailable_reason"],
                  "branch": branch["branch"], "incomplete_identification": branch["incomplete_identification"],
                  "secondary_a_scored": bool(score_stage["counts"]["s_scored"]),
                  "secondary_a_unscorable": score_stage["details"].get("unscorable"),
                  "seeds": {"bootstrap": SEED_BOOT, "synthetic": SEED_SYNTH}, "resamples": RESAMPLES, "level": LEVEL, "z": Z,
                  "tolerances": {"qdev_rehearsal": QDEV_TOL, "reload": "qtest.RELOAD_ATOL"}, "sign": SIGN,
                  "decision": DECISION_TEXT},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "lightgbm": lightgbm.__version__, "torch": torch.__version__, "platform": platform.platform()},
        "test_step": {"labels": str(paths.out / KA_SEALED_NAME), "openings": str(paths.openings), "note": OPENING_NOTE,
                      "order": ["B0 warm-up, lock recheck, label-free checks, rehearsal recomputation, modules", "B1 OPENINGS entry",
                                "B2 sealed KA labels read once, E6_REPORT.json once"]},
        "exposure_ledger": {"path": str(paths.exposure), "sha256": sha256_file(paths.exposure),
                            "note": "the planned-look row with this lock's SHA-256 is appended after the lock"},
        "verify": verify}
    lock_sha = write_json_once(target, payload)
    log(f"[lock] wrote {target} sha256={lock_sha}")
    return {"counts": {"files": len(verify), "worker_records": len(starts) + len(ends)},
            "details": {"lock": str(target), "lock_sha256": lock_sha}, "outputs": {after_path.name: after_path}}


# ----------------------------------------------------------------------------- part B (B0-B2): written now, run after the review
@dataclass
class TestDeps:
    warm_up: Callable[[], None]
    verify_lock: Callable[[Mapping[str, Any], str, Paths], Mapping[str, Any]]
    preflight: Callable[[Paths, Mapping[str, Any]], Mapping[str, Any]]
    reproduce: Callable[[Paths, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
    modules_outside_lock: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    load_labels: Callable[[Paths, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
    analyse: Callable[[Paths, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def exposure_problems(lock_sha: str, paths: Paths) -> dict[str, Any]:
    text = Path(paths.exposure).read_text(encoding="utf-8") if Path(paths.exposure).is_file() else ""
    return {} if lock_sha in text else {"exposure_row_missing": f"EXPOSURE_KO.md has no planned-look row with {lock_sha}"}


def modules_outside_lock(lock: Mapping[str, Any]) -> dict[str, str]:
    code = (lock.get("code") or {}).get("files") or {}
    return {name: sha for name, sha in loaded_files().items() if code.get(name) != sha}


def verify_lock(lock: Mapping[str, Any], lock_sha: str, paths: Paths) -> dict[str, Any]:
    """B0 2: every bound file, the worker-record listing, the PRESPEC blob, the loaded modules, the versions and the
    EXPOSURE row (empty when the lock holds)."""

    problems: dict[str, Any] = {}
    prespec = lock.get("prespec") or {}
    if not prespec.get("blob_sha256"):
        problems["prespec_hash_missing"] = True
    else:
        try:
            sha, _, _ = prespec_blob(paths, prespec["commit"])
            unchanged = _git(paths.repo, "diff", "--quiet", prespec["commit"], "--", prespec["path"]).returncode == 0
            if sha != prespec["blob_sha256"] or not unchanged:
                problems["prespec_changed"] = {"locked": prespec["blob_sha256"], "now": sha, "unchanged": unchanged}
        except Blocked as error:
            problems["prespec"] = str(error)
    changed = {}
    for label, entry in (lock.get("verify") or {}).items():
        path = Path(entry["path"])
        now = sha256_file(path) if path.is_file() else None
        if now != entry["sha256"]:
            changed[label] = {"locked": entry["sha256"], "now": now}
    if changed:
        problems["changed_since_lock"] = changed
    starts, ends = worker_record_files(paths.out / "worker_modules")
    listing = sorted(p.relative_to(paths.out).as_posix() for p in starts + ends)
    if listing != (lock.get("module_records") or {}).get("worker_records"):
        problems["worker_records_listing"] = "the worker_modules listing differs from the lock"
    outside = modules_outside_lock(lock)
    if outside:
        problems["loaded_modules_not_in_lock"] = outside
    import scipy
    import lightgbm
    import torch

    env = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "lightgbm": lightgbm.__version__, "torch": torch.__version__}
    locked_env = {k: (lock.get("environment") or {}).get(k) for k in env}
    if env != locked_env:
        problems["environment"] = {"locked": locked_env, "now": env}
    problems.update(exposure_problems(lock_sha, paths))
    return problems


def preflight(paths: Paths, lock: Mapping[str, Any]) -> dict[str, Any]:
    """B0 3: label-free inputs (row table, scores, player input, sealed KA hash and row count, KA_REPORT Q_DEV numbers)."""

    inputs = dict(e6_inputs(paths))
    sealed = paths.out / KA_SEALED_NAME
    bound = ((lock.get("verify") or {}).get(f"labels_hash_only:{KA_SEALED_NAME}") or {}).get("sha256")
    if sha256_file(sealed) != bound:
        raise Blocked("the sealed KA file differs from the lock")
    manifest = read_json(paths.out / "KA_LABELS_E6_MANIFEST.json")
    if int(manifest["rows"]) != int(np.asarray(inputs["rows"]["engagement_key"]).size) or manifest["sha256"] != bound:
        raise Blocked("KA_LABELS_E6_MANIFEST rows or hash differ from the row table / lock")
    block = read_json(paths.q_ka / "KA_REPORT.json")["splits"]["q_dev"]["ka_trained_on_ka"]
    for name in ("P1", "S_effect_meanD"):
        entry = block["contrasts"]["contrasts"][name]
        lo, hi = entry["interval"]
        if not (math.isfinite(entry["point"]) and math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise Blocked(f"KA_REPORT Q_DEV {name} reference is not a finite ordered interval")
    inputs["checks"] = {"rows": int(np.asarray(inputs["rows"]["engagement_key"]).size), "sealed_sha256": bound,
                        "cells": sorted(inputs["preds"]), "branch": inputs["branch"]["branch"]}
    return inputs


def reproduce(paths: Paths, lock: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """B0 4: both rehearsals recomputed with this code equal the lock-bound files exactly (canonical JSON); Q_DEV also
    within 1e-9 of KA_REPORT."""

    record: dict[str, Any] = {"equal": False, "error": None}
    try:
        stored_qdev = read_json(paths.out / "E6_QDEV_REHEARSAL.json")
        stored_synth = read_json(paths.out / "E6_SYNTH_REHEARSAL.json")
        qdev = qdev_rehearsal(paths)
        synth = synth_rehearsal(paths, inputs)
        record.update(qdev_equal=canonical(qdev["result"]) == canonical(stored_qdev["result"]),
                      qdev_comparison_equal=canonical(qdev["ka_report_comparison"]) == canonical(stored_qdev["ka_report_comparison"]),
                      qdev_ka_report_passed=bool(qdev["ka_report_comparison"]["passed"]),
                      synth_equal=canonical(synth["result"]) == canonical(stored_synth["result"]))
        record["equal"] = bool(record["qdev_equal"] and record["qdev_comparison_equal"] and record["qdev_ka_report_passed"]
                               and record["synth_equal"])
    except Exception as error:  # noqa: BLE001  (a reproduction that cannot run is not equal)
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def analyse_part_b(paths: Paths, lock: Mapping[str, Any], inputs: Mapping[str, Any], labels: Mapping[str, Any]) -> dict[str, Any]:
    out = analyse_core(inputs, labels)
    out["selected"] = len(read_json(paths.e6_manifest)["matches"])
    out["stage_counts"] = {stage: read_json(stage_record_path(paths, stage)).get("counts")
                           for stage in ("build", "audit", "bank", "engagements", "e2c")}
    return out


def default_deps() -> TestDeps:
    return TestDeps(warm_up=warm_up, verify_lock=verify_lock, preflight=preflight, reproduce=reproduce,
                    modules_outside_lock=modules_outside_lock,
                    load_labels=lambda p, inputs, opening: load_ka_labels(p.out / KA_SEALED_NAME, inputs["rows"], paths=p,
                                                                         opening=opening),
                    analyse=analyse_part_b)


def _test_blocked(paths: Paths, reason: str, **details: Any) -> int:
    write_json_once(paths.out / f"TEST_BLOCKED_{stamp()}.json",
                    {"version": VERSION, "experiment": EXPERIMENT, "stage": "test", "status": "BLOCKED", "utc": utc(),
                     "reason": reason, "before_exposure": True, **details})
    print(f"TEST BLOCKED: {reason}", flush=True)
    return 2


def e6_openings(paths: Paths) -> list[dict[str, Any]]:
    if not Path(paths.openings).is_file():
        return []
    return [o for o in read_json(paths.openings).get("openings", []) if o.get("experiment") == EXPERIMENT]


# Fix round 1 (Astra review of part A): every step after the OPENINGS entry starts to be saved runs under failure
# handling; each failure writes a record that says whether the entry is on disk, whether the labels were read and
# whether label-derived values reached files or the screen ("unknown" where that cannot be established).
SINGLE_WRITER_NOTE = ("part B runs as one single invocation, the only writer of OPENINGS.json while it runs; "
                      "OPENINGS.json has no cross-process lock (Astra review of part A, parked: concurrent writers could "
                      "lose entries)")
WATCHDOG_LOCK_WAIT_S = 60.0
EXPOSURE_RULE = ("labels_read is 'no' until the label loader is called, 'unknown' when the loader failed (it may have "
                 "read bytes) and 'yes' once it returned. While the labels are in memory, Python output goes to an "
                 "in-memory buffer. With labels read, exception text and captured output are written only to files whose "
                 "names contain SEALED (never opened; status 'unknown'), and the record keeps the exception type and "
                 "frame locations. Streams after the labels were read: a regular file behind stdout or stderr that did "
                 "not grow is 'no'; one that grew, a pipe or an unknown target makes both files and screen 'unknown' (a "
                 "redirected stream is a file); a console makes the screen 'unknown'. If the full record cannot be "
                 "prepared, a minimal record of fixed fields (statuses 'unknown' once labels may have been read) is "
                 "written, else a fixed line goes to stderr; no exception text leaves the recorder")


class _Capture(io.TextIOBase):
    """Thread-safe in-memory text sink for sys.stdout and sys.stderr while the labels are in this process."""

    def __init__(self) -> None:
        super().__init__()
        self._guard = threading.Lock()
        self._parts: list[str] = []

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        with self._guard:
            self._parts.append(str(text))
        return len(text)

    def text(self) -> str:
        with self._guard:
            return "".join(self._parts)


@dataclass
class PartBState:
    """What part B has done since it armed its watchdog: the phase, whether the OPENINGS entry is on disk, whether the
    labels were read, and the Python output captured while they are in memory (shared with the watchdog thread)."""

    paths: Paths
    attempt: int
    report_path: Path
    phase: str = "before the OPENINGS entry"
    entry: dict[str, Any] | None = None
    persisted: str = "no"                                              # "no", "yes", "unknown"
    labels: str = "no"                                                 # "no", "reading", "yes", "unknown"
    fd_state: dict[str, dict[str, Any]] | None = None
    capture: _Capture = field(default_factory=_Capture)
    finish: threading.Lock = field(default_factory=threading.Lock)
    done: threading.Event = field(default_factory=threading.Event)
    fired: bool = False


def _hard_exit(code: int) -> None:
    """End the process at once (part B time limit); tests replace os._exit."""

    os._exit(code)


def _flush_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass


def _fd_state() -> dict[str, dict[str, Any]]:
    """What fds 1 and 2 point at: a regular file (with its byte size), a console, a pipe, another kind, or unknown."""

    state: dict[str, dict[str, Any]] = {}
    for name, fd in (("stdout", 1), ("stderr", 2)):
        try:
            info = os.fstat(fd)
            mode = info.st_mode
            kind = ("file" if stat.S_ISREG(mode) else "console" if stat.S_ISCHR(mode)
                    else "pipe" if stat.S_ISFIFO(mode) else "other")
            state[name] = {"kind": kind, "bytes": int(info.st_size) if kind == "file" else None}
        except (OSError, ValueError):
            state[name] = {"kind": "unknown", "bytes": None}
    return state


def _stream_exposure(before: Mapping[str, Any] | None, now: Mapping[str, Any] | None) -> tuple[str, str]:
    """(screen, files) for label-derived values through stdout and stderr after the labels were read."""

    screen = files = "no"
    for name in ("stdout", "stderr"):
        b, n = (before or {}).get(name) or {}, (now or {}).get(name) or {}
        if b.get("kind") == n.get("kind") == "file" and b.get("bytes") is not None and b.get("bytes") == n.get("bytes"):
            continue                                                   # a redirected file that did not grow
        screen = "unknown"
        if not b.get("kind") == n.get("kind") == "console":
            files = "unknown"                                          # a file that grew, a pipe or an unknown target
    return screen, files


def _opening_on_disk(paths: Paths, entry: Mapping[str, Any] | None) -> str:
    """'yes', 'no' or 'unknown': is this attempt's entry in OPENINGS.json (read independently of append_opening)."""

    if not entry:
        return "no"
    try:
        path = Path(paths.openings)
        if not path.is_file():
            return "no"
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        fields = ("experiment", "attempt", "utc", "lock_sha256")
        return "yes" if any(all(o.get(k) == entry.get(k) for k in fields) for o in doc.get("openings", [])) else "no"
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_sealed_text(path: Path, text: str, what: str) -> dict[str, Any]:
    """Text that may hold label-derived values, written only to a file whose name contains SEALED (never opened)."""

    entry: dict[str, Any] = {"file": str(path), "holds": what, "status": "unknown"}
    try:
        entry["sha256"] = write_bytes_once(path, text.encode("utf-8", "replace"))
        entry["bytes"] = len(text.encode("utf-8", "replace"))
    except Exception as error:  # noqa: BLE001  (a partial pending file may exist; still unknown)
        entry["write_error"] = type(error).__name__
    return entry


def _exposure(state: PartBState, sealed_files: list[dict[str, Any]]) -> dict[str, Any]:
    out = Path(state.paths.out)
    labels = "unknown" if state.labels == "reading" else state.labels
    derived = labels != "no"
    report_written = state.report_path.exists()
    partial = sorted(p.name for p in out.glob(f".{state.report_path.name}.*.pending"))
    now = _fd_state()
    screen, stream_files = _stream_exposure(state.fd_state, now) if derived else ("no", "no")
    if report_written or (derived and partial):
        files = "yes"
    elif derived and (sealed_files or stream_files != "no"):
        files = "unknown"
    else:
        files = "no"
    return {"opening_persisted": state.persisted, "labels_read": labels,
            "report_file": "written" if report_written else "not written", "partial_report_files": partial,
            "sealed_files": sealed_files, "label_derived_values_in_files": files,
            "label_derived_values_on_screen": screen,
            "stdout_stderr": {"when_labels_read": state.fd_state, "at_record": now}, "rule": EXPOSURE_RULE}


def _emergency_line(state: PartBState, code: int, *errors: BaseException) -> None:
    """A fixed, label-free line on stderr (exception type names only) when no failure record could be written."""

    try:
        what = "E6 TEST FAILED AFTER THE OPENING" if code == 3 else "E6 TEST BLOCKED BEFORE THE OPENING"
        types = ", ".join(type(e).__name__ for e in errors)
        text = (f"{what} (attempt {state.attempt}, phase {state.phase}): no failure record could be written ({types}); "
                f"opening persisted {state.persisted}; labels read {state.labels}\n")
    except BaseException:  # noqa: BLE001
        text = "E6 TEST FAILED: no failure record could be written\n"
    try:
        stream = sys.__stderr__ if isinstance(sys.stderr, _Capture) or sys.stderr is None else sys.stderr
        stream.write(text)
        stream.flush()
    except BaseException:  # noqa: BLE001
        try:
            os.write(2, text.encode("utf-8", "replace"))
        except BaseException:  # noqa: BLE001
            pass


def _write_failure_record(state: PartBState, *, reason: str, error: BaseException | None = None) -> int:
    """The full failure record: E6_FAILED_<n>.json when the OPENINGS entry is (or may be) on disk, TEST_BLOCKED_<utc>.json
    when it is not.  Label-derived text never goes into the record itself."""

    paths = state.paths
    if state.persisted != "yes":
        state.persisted = _opening_on_disk(paths, state.entry)
    derived = state.labels != "no"
    tag = stamp()
    sealed: list[dict[str, Any]] = []
    record: dict[str, Any] = {"version": VERSION, "experiment": EXPERIMENT, "stage": "test", "utc": utc(),
                              "attempt": state.attempt, "phase": state.phase, "reason": reason, "opening": state.entry,
                              "ledger_writer": SINGLE_WRITER_NOTE, "minimal": False}
    if error is not None:
        record["error_type"] = type(error).__name__
        if derived:
            detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            sealed.append(_write_sealed_text(paths.out / f"E6_FAILED_DETAIL_SEALED_{tag}.txt", detail,
                                             "exception message and traceback"))
        else:
            record["error"] = str(error)
            record["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        record["frames"] = [{"file": f.filename, "line": f.lineno, "function": f.name}
                            for f in traceback.extract_tb(error.__traceback__)]
    captured = state.capture.text()
    if captured:
        if derived:
            sealed.append(_write_sealed_text(paths.out / f"E6_FAILED_OUTPUT_SEALED_{tag}.txt", captured,
                                             "Python output captured after the labels were read"))
        else:
            record["captured_output"] = captured
    record["exposure"] = _exposure(state, sealed)
    if state.persisted == "no":
        record.update(status="BLOCKED", before_exposure=True)
        target, code = paths.out / f"TEST_BLOCKED_{tag}.json", 2
    else:
        record.update(status="failed_after_exposure", before_exposure=False)
        target, code = next_free(paths.out / "E6_FAILED_1.json"), 3
    write_json_once(target, record)                                    # a failure here goes to the minimal record
    stream = sys.__stderr__ if isinstance(sys.stdout, _Capture) else sys.stdout
    try:
        stream.write(("TEST BLOCKED" if code == 2 else f"E6 TEST FAILED after the opening (attempt {state.attempt})")
                     + f": {reason} during {state.phase}; record {target.name}\n")
        stream.flush()
    except Exception:  # noqa: BLE001
        pass
    return code


def _minimal_failure_record(state: PartBState, reason: str, error: BaseException | None,
                            recorder_error: BaseException) -> int:
    """When the full record could not be prepared: a record of fixed fields only (no exception text), else a fixed
    line on stderr.  Never raises."""

    code = 3
    try:
        persisted = state.persisted if state.persisted in ("yes", "no", "unknown") else "unknown"
        labels = "unknown" if state.labels == "reading" else state.labels
        code = 2 if persisted == "no" and labels == "no" else 3
        status = "unknown" if labels != "no" else "no"
        record = {"version": VERSION, "experiment": EXPERIMENT, "stage": "test", "utc": utc(), "minimal": True,
                  "attempt": int(state.attempt), "phase": str(state.phase), "reason": str(reason),
                  "status": "BLOCKED" if code == 2 else "failed_after_exposure", "before_exposure": code == 2,
                  "error_type": type(error).__name__ if error is not None else None,
                  "recorder_error_type": type(recorder_error).__name__, "ledger_writer": SINGLE_WRITER_NOTE,
                  "exposure": {"opening_persisted": persisted, "labels_read": labels,
                               "label_derived_values_in_files": status, "label_derived_values_on_screen": status,
                               "note": ("the full failure record could not be prepared; files whose names contain SEALED "
                                        "may have been written before that and are never opened"),
                               "rule": EXPOSURE_RULE}}
        name = f"{'TEST_BLOCKED' if code == 2 else 'E6_FAILED'}_MIN_{stamp()}_{uuid.uuid4().hex[:8]}.json"
        try:
            write_json_once(state.paths.out / name, record)
        except BaseException as write_error:  # noqa: BLE001
            _emergency_line(state, code, recorder_error, write_error)
            return code
        try:
            stream = sys.__stderr__ if isinstance(sys.stdout, _Capture) else sys.stdout
            stream.write(("TEST BLOCKED" if code == 2 else f"E6 TEST FAILED after the opening (attempt {state.attempt})")
                         + f": {reason} during {state.phase}; minimal record {name}" + chr(10))
            stream.flush()
        except BaseException:  # noqa: BLE001
            pass
    except BaseException as other:  # noqa: BLE001
        _emergency_line(state, code, recorder_error, other)
    return code


def _record_failure(state: PartBState, *, reason: str, error: BaseException | None = None) -> int:
    """Record a part B attempt that did not finish.  Guarded as a whole: a failure while preparing or writing the record
    still leaves a minimal record or a fixed line on stderr, and no exception (so no chained, possibly label-bearing
    context) leaves it."""

    try:
        return _write_failure_record(state, reason=reason, error=error)
    except BaseException as recorder_error:  # noqa: BLE001
        return _minimal_failure_record(state, reason, error, recorder_error)


def _fail(state: PartBState, reason: str, error: BaseException) -> int:
    """The main thread's failure path: stop the watchdog, then record unless the watchdog already did.  Never raises."""

    got = False
    try:
        state.done.set()
        got = state.finish.acquire(timeout=WATCHDOG_LOCK_WAIT_S)
        if state.fired:
            return 3
        return _record_failure(state, reason=reason, error=error)
    except BaseException as fail_error:  # noqa: BLE001  (never re-raised: the original may carry label-derived text)
        code = 2 if state.persisted == "no" and state.labels == "no" else 3
        _emergency_line(state, code, fail_error)
        return code
    finally:
        if got:
            state.finish.release()


def _watchdog(state: PartBState, limit_s: float) -> None:
    """Part B time limit: after limit_s without the report, record the attempt and end the process."""

    if state.done.wait(limit_s):
        return
    got = state.finish.acquire(timeout=WATCHDOG_LOCK_WAIT_S)
    code = 3
    try:
        if state.done.is_set():
            return
        state.fired = True
        code = _record_failure(state, reason="part B time limit")
    except BaseException as error:  # noqa: BLE001  (last resort; the process still ends below)
        _emergency_line(state, code, error)
    finally:
        if state.fired:
            _hard_exit(code)
        if got:
            state.finish.release()


def test_stage(paths: Paths, deps: TestDeps | None = None, *, time_limit_s: float | None = PART_B_LIMIT_S) -> int:
    """Part B (B0-B2): checks with no label, then the OPENINGS.json entry, then the sealed KA labels once, then
    E6_REPORT.json once.  Not run in part A.  The watchdog is armed before the entry (a thread that cannot start
    stops the run with no exposure); every later step is under failure handling (fix round 1), and the recorder is
    guarded as a whole (fix round 2)."""

    deps = deps or default_deps()
    started, tick = utc(), time.perf_counter()
    lock_path = latest_lock(paths.out)
    if lock_path is None:
        return _test_blocked(paths, "no E6_LOCK.json")
    lock, lock_sha = read_json(lock_path), sha256_file(lock_path)
    previous = e6_openings(paths)
    if any(o.get("lock_sha256") == lock_sha for o in previous):
        return _test_blocked(paths, "this lock was already used by an opening (Ruling 31: a rerun needs a new folder and lock)")
    prespec = (lock.get("prespec") or {}).get("blob_sha256")
    if any(o.get("prespec_sha256") == prespec for o in previous) and Path(paths.out).name == OUT_DEFAULT.name:
        return _test_blocked(paths, "an earlier E6 opening of this PRESPEC exists; a rerun runs in tog_e6_2 with a new lock")
    try:
        deps.warm_up()
    except Exception as error:  # noqa: BLE001
        return _test_blocked(paths, f"warm-up failed: {type(error).__name__}: {error}", traceback=traceback.format_exc())
    problems = deps.verify_lock(lock, lock_sha, paths)
    if problems:
        return _test_blocked(paths, "the lock does not hold", lock=str(lock_path), problems=problems)
    try:
        inputs = deps.preflight(paths, lock)
    except Exception as error:  # noqa: BLE001
        return _test_blocked(paths, f"a label-free input check failed: {type(error).__name__}: {error}", lock=str(lock_path))
    repro = deps.reproduce(paths, lock, inputs)
    if not repro.get("equal"):
        return _test_blocked(paths, "the rehearsals recomputed with this code differ from the lock-bound files", reproduction=repro)
    late = deps.modules_outside_lock(lock)
    if late:
        return _test_blocked(paths, "modules outside the lock were loaded before the opening", modules=late)
    attempt = len(previous) + 1
    sealed = paths.out / KA_SEALED_NAME
    report_path = paths.out / ("E6_REPORT.json" if attempt == 1 else f"E6_REPORT_a{attempt}.json")
    state = PartBState(paths=paths, attempt=attempt, report_path=report_path)
    if time_limit_s is not None:
        try:
            threading.Thread(target=_watchdog, args=(state, float(time_limit_s)), daemon=True,
                             name="e6-part-b-watchdog").start()
        except Exception as error:  # noqa: BLE001
            return _test_blocked(paths, f"the part B watchdog could not start: {type(error).__name__}: {error}")
    entry = {"experiment": EXPERIMENT, "attempt": attempt, "utc": utc(), "lock": str(lock_path), "lock_sha256": lock_sha,
             "prespec_sha256": prespec, "labels": [str(sealed)], "sealed_sha256": sha256_file(sealed), "note": OPENING_NOTE}
    state.entry, state.phase = entry, "saving the OPENINGS entry"
    try:
        with state.finish:
            before = append_opening(paths.openings, entry)               # exposure starts once this entry is on disk
            state.persisted = "yes"
    except BaseException as error:  # noqa: BLE001  (recorded: E6_FAILED if the entry reached disk, else TEST_BLOCKED)
        return _fail(state, "exception", error)
    try:
        state.phase = "writing E6_OPENING.json"
        write_json_once(paths.out / ("E6_OPENING.json" if attempt == 1 else f"E6_OPENING_a{attempt}.json"),
                        {**entry, "openings_before": before, "ledger_writer": SINGLE_WRITER_NOTE,
                         "time_limit_s": time_limit_s, "exposure_rule": EXPOSURE_RULE})
        state.phase = "reading the sealed KA labels"
        _flush_streams()
        state.fd_state = _fd_state()
        with contextlib.redirect_stdout(state.capture), contextlib.redirect_stderr(state.capture):
            state.labels = "reading"
            labels = deps.load_labels(paths, inputs, entry)
            state.labels = "yes"
            state.phase = "analysing"
            result = deps.analyse(paths, lock, inputs, labels)
            state.phase = "checking the modules after the opening"
            late = deps.modules_outside_lock(lock)
            if late:
                raise Blocked(f"modules outside the lock were loaded after the opening: {sorted(late)}")
            state.phase = "writing E6_REPORT.json"
            with state.finish:
                if state.fired:
                    return 3                                             # the watchdog recorded the time limit
                payload = {"version": VERSION, "experiment": EXPERIMENT, "status": "completed", "attempt": attempt,
                           "opening": entry, "openings_before": before,
                           "lock": {"path": str(lock_path), "sha256": lock_sha, "locked_utc": lock.get("locked_utc")},
                           "prespec": lock.get("prespec"), "result": result,
                           "pre_exposure_checks": {"lock": "held", "inputs": (inputs or {}).get("checks"),
                                                   "reproduction": repro},
                           "environment": {"python": platform.python_version(), "numpy": np.__version__},
                           "started_utc": started, "finished_utc": utc(),
                           "wall_seconds": round(time.perf_counter() - tick, 1), "peak_memory_bytes": peak_memory_bytes(),
                           "captured_output_after_opening": state.capture.text(), "ledger_writer": SINGLE_WRITER_NOTE,
                           "run": {"seed": SEED_BOOT, "rows": result.get("rows"), "matches": result.get("matches"),
                                   "code_sha256": {CODE_REL: sha256_file(paths.repo / CODE_REL)}}}
                write_json_once(report_path, payload)
                state.done.set()
    except BaseException as error:  # noqa: BLE001  (recorded; the opening stays)
        if state.labels == "reading":
            state.labels = "no" if isinstance(error, LabelAccessError) else "unknown"
        return _fail(state, "exception", error)
    try:
        print(f"E6 TEST written (attempt {attempt})", flush=True)
    except Exception:  # noqa: BLE001  (the report is written; a closed stdout changes nothing)
        pass
    return 0


# ----------------------------------------------------------------------------- supervisor (resumable, detached)
def launcher_command(args: list[str]) -> list[str]:
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
            "from studies.tog_experiments_20261012 import e6_run as m; raise SystemExit(m.main(sys.argv[1:]))")
    return [sys.executable, "-B", "-c", code, *args]


def step_command(step: str, paths: Paths, workers: int) -> list[str]:
    common = ["--out", str(paths.out), "--workers", str(workers), "--tag", f"e6step:{step}"]
    if step in PROCESS_STEPS:
        return launcher_command(["process", "--step", step, *common])
    return launcher_command([step.replace("_", "-"), *common])


def step_env(step: str, paths: Paths, base: Mapping[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ if base is None else base)
    if step == "score_v":
        env = score_v_env(base)
    else:
        env = {k: v for k, v in base.items()}
        env.update({"E6_STAGE": step, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE", **{v: "1" for v in THREAD_VARIABLES}})
        env.pop("E6_OUT", None)
    if Path(paths.out) != OUT_DEFAULT:
        env["E6_OUT"] = str(paths.out)                                 # dry runs into another folder only
    return env


def _find_step_process(step: str) -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
        except Exception:  # noqa: BLE001
            continue
        if f"e6step:{step}" in cmd and proc.info["pid"] != os.getpid():
            return int(proc.info["pid"])
    return None


def _kill_tree(pid: int) -> None:
    try:
        import psutil

        root = psutil.Process(pid)
        for child in root.children(recursive=True):
            child.kill()
        root.kill()
    except Exception:  # noqa: BLE001
        pass


def code_gate(paths: Paths) -> tuple[bool, str]:
    if (paths.out / "HOLD").exists():
        return False, "OUT/HOLD exists"
    if Path(paths.out) != OUT_DEFAULT:
        return True, "dry run folder"
    ok, ident = code_committed(paths)
    return ok, "" if ok else f"{CODE_REL} differs from HEAD: {ident['status_porcelain'].strip()}"


def run_supervisor(paths: Paths, until: str, workers: int, *, poll_s: float = 15.0) -> int:
    """Run the steps up to `until`, one process at a time; resumable (completed steps are skipped, a live step process
    of an earlier supervisor is waited for)."""

    if until not in STEP_ORDER:
        raise ValueError(f"unknown step {until!r}")
    paths.out.mkdir(parents=True, exist_ok=True)
    log = StageLog(paths.out / "logs" / "supervisor.log")
    log(f"[run] supervisor pid={os.getpid()} until={until} workers={workers}")
    for step in STEP_ORDER[: STEP_ORDER.index(until) + 1]:
        if stage_done(paths, step):
            continue
        pid = _find_step_process(step)
        adopted = pid is not None
        if adopted:
            log(f"[run] {step}: waiting for the live step process pid={pid}")
            handle: Any = None
        else:
            waited = 0.0
            while True:
                ok, why = code_gate(paths)
                if ok:
                    break
                if waited % 300 == 0:
                    log(f"[run] {step}: waiting ({why})")
                time.sleep(30)
                waited += 30
            env = step_env(step, paths)
            cmd = step_command(step, paths, workers)
            out_log = paths.out / "logs" / f"{step}.stdout.log"
            stream = open(out_log, "ab")
            handle = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            pid = handle.pid
            shown = sorted(env) if step == "score_v" else f"{len(env)} entries (inherited, E6_STAGE={env.get('E6_STAGE')})"
            log(f"[run] {step}: started pid={pid} environment {shown}")
            if step == "score_v":
                append_line(paths.out / "logs" / "score_v_env.jsonl", {"utc": utc(), "pid": pid, "environment": env})
        limit = STAGE_LIMIT_S.get(step, DEFAULT_LIMIT_S)
        begun, peak = time.monotonic(), 0
        while True:
            alive = handle.poll() is None if handle is not None else _pid_alive(pid)
            if not alive:
                break
            charge = commit_bytes() or 0
            peak = max(peak, charge)
            if charge > COMMIT_LIMIT_BYTES or time.monotonic() - begun > limit:
                reason = "commit" if charge > COMMIT_LIMIT_BYTES else "time"
                _kill_tree(pid)
                write_json_once(paths.out / f"{step.upper()}_BLOCKED_{stamp()}.json",
                                {"version": VERSION, "stage": step, "status": "BLOCKED", "utc": utc(),
                                 "reason": f"{reason} limit (section 9)", "commit_bytes": charge, "limit_s": limit})
                log(f"[run] {step}: BLOCKED by the {reason} limit")
                return 2
            time.sleep(poll_s)
        code = handle.returncode if handle is not None else None
        append_line(paths.out / "logs" / "supervisor.jsonl",
                    {"step": step, "pid": pid, "adopted": adopted, "exit_code": code, "utc": utc(),
                     "wall_seconds": round(time.monotonic() - begun, 1), "peak_system_commit_bytes": peak})
        if not stage_done(paths, step):
            log(f"[run] {step}: ended (exit {code}) without a completed stage record; stopping")
            return 1
        log(f"[run] {step}: completed (peak system commit {peak / 2**30:.1f} GiB)")
    log(f"[run] done up to {until}")
    return 0


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- command line
def _stage_entry(paths: Paths, stage: str) -> None:
    """Before any frozen import: E6_STAGE, OUT binding, the executor subclass."""

    os.environ["E6_STAGE"] = stage
    if stage != "score_v":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                        # GPU only in score-v (PRESPEC 4, 13)
    if Path(paths.out) != OUT_DEFAULT:
        os.environ["E6_OUT"] = str(paths.out)
    bind_outputs(paths.out)
    install_executor()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("command", choices=("run", "select", "build", "process", "score-v", "players", "score", "labels",
                                            "rehearse", "lock", "test"))
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--step", choices=PROCESS_STEPS)
    parser.add_argument("--until", default="rehearse")
    parser.add_argument("--tag", default=None, help="process marker for the supervisor (ignored)")
    parser.add_argument("--reason", default=None, help="lock: why a new lock version is written (recorded in the lock)")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")
    paths = Paths(out=Path(args.out))
    if args.command == "run":
        return run_supervisor(paths, args.until, args.workers)
    if args.command == "process":
        if not args.step:
            raise SystemExit("process needs --step (the supervisor runs the steps in order: run --until e2c)")
        _stage_entry(paths, args.step)
        return run_stage(paths, args.step, lambda ctx: process_body(ctx, args.step, args.workers))
    stage = args.command.replace("-", "_")
    if stage == "test":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        bind_outputs(paths.out)
        return test_stage(paths)
    if stage == "lock":
        step = lock_step(paths)
        if step is None:
            print(f"[lock] {latest_lock(paths.out)} already binds the committed code; no new lock version", flush=True)
            return 0
        _stage_entry(paths, step)
        return run_stage(paths, step, lambda ctx: lock_body(ctx, reason=args.reason))
    _stage_entry(paths, stage)
    bodies = {"select": lambda ctx: select_body(ctx, args.workers), "build": lambda ctx: build_body(ctx, args.workers),
              "players": players_body, "score": score_body, "labels": labels_body, "rehearse": rehearse_body}
    if stage == "score_v":
        return score_v_stage(paths, args.workers)
    return run_stage(paths, stage, bodies[stage])


register_dataset()
if os.environ.get("E6_STAGE"):
    bind_outputs(Path(os.environ.get("E6_OUT") or OUT_DEFAULT))       # spawn workers of a stage (section 4 "0 등록")

if __name__ == "__main__":
    from studies.tog_experiments_20261012 import e6_run as _module

    raise SystemExit(_module.main())
