"""T4.3 E4: q retraining on the second measurer's labels, the lock, and the single TEST step (PLAN_E1_E5_KO.md Task 12).

Binding specification: reports/tog_experiments_20261012/PRESPEC_E4_KO.md (sections 3-7, 10) and EXPOSURE_KO.md (SDD
Rulings 19, 28, 43-47).  Stages, run in this order and each once; every output is written once under
D:/LOL_Training_20261004/tog_e4_1.  Part A (Ruling 47) ends with the lock and opens no TEST label; part B is the test
stage, run only after the review of part A.

  rows     The E4 row table Q_ROWS_h90.npz: the rows and order of q_run_1/Q_ROWS_h90.npz with y, label_svi_delta,
           pt_p_pre, exact_zero and near_zero_bits replaced by the E4 values (e4_run.direction_labels / zero_flags on
           Task 10's PRED_<ctx>.npz; TRAIN rows oof<fold>, the other rows final).  The TRAIN/Q_DEV/Q_CAL values are
           recomputed with e4_run.nontest_labels and must equal Task 11's E4_LABELS_nonTEST.npz (Ruling 46).  The TEST
           labels (y4, label_svi_delta4, exact_zero, near_zero_bits) are made from PRED_final and go straight into
           Q_TEST_SEALED_h90.npz through the frozen rows.seal_test; the row table keeps y = -1, NaN, -1, 0 there and the
           new measurer's p_pre (pt_p_pre, an input).  No TEST label statistic is computed, printed or stored.  The
           record Q_ROWS_h90.json has the frozen rows.write_rows keys; its counts hold label figures for TRAIN, Q_DEV and
           Q_CAL only, and it is handled as a sealed summary (Ruling 28): never printed, hashed into the lock, and code
           reads only its TRAIN/Q_DEV/Q_CAL counts.  E4_ROWS.json is the stage record (hashes and checks only).
  prepare  raw/store = a directory junction to q_run_1/raw/store; inputs/main_h90/XY.npy = a hash-checked copy of
           q_run_1's; then the frozen q_prepare.py --run tog_e4_1 --variant main_h90 --release SCIENTIFIC_Q_15638e1f...
           The new FIT and REFIT_TRAIN CONTEXT.npz and bundle/MANIFEST.json must equal q_run_1's (the execution evidence
           that inputs and preprocessing read no label), else BLOCKED.  E4_PREPARE.json.
  train    The 16 job specs in the q_supervisor.py:447-449 shape (the frozen plan's search jobs: LightGBM lgbm_l15_m100,
           lgbm_l15_m200, lgbm_l31_m100, lgbm_l31_m200 on LGBM_S1_A1_M1_E1_C1_R1_D1; PT_flex ptflex_k{4,6,8}_C{0.001,
           0.01,0.1,1}), each run as a q_worker.py subprocess: LightGBM one at a time, at most four worker processes,
           CPU only.  Resumable: a completed job (RESULT.json completed for its SPEC.json) is never relaunched; a failed
           attempt is kept as RESULT_attempt<k>.json before its one retry; live workers are adopted after a driver
           restart.  All 16 must complete, else BLOCKED.  E4_TRAIN.json.
  select   qselect.select_within on Q_DEV raw match-weighted Brier within each family; every card must have completed,
           else BLOCKED.  E4_SELECTION.json.
  qdev     Rehearsal of the TEST evaluation on Q_DEV (development evidence that decides nothing; PRESPEC_E4 section 6):
           the selected cells' stored Q_DEV predictions through evaluate(); its agreement must equal Task 11's
           E4_QDEV_AGREEMENT.json, else BLOCKED.  E4_QDEV.json.
  lock     E4_LOCK.json: PRESPEC_E4's SHA-256 (on disk and git blob) and its commit c8e78b4, code, inputs, the Task 10/11
           outputs, the row table, its record (sealed summary), both sealed files (bytes only), the preprocessing, every
           job's files, the two selected cells as qtest.cell_record, row counts, rules, versions and the UTC time.
  test     (part B, only after the review of part A) lock re-check -> the frozen qtest scoring path (qtest.LoadedCell,
           the Q_DEV reload check, the sealed-row check) -> TEST_SCORES/ -> the opening in E4_OPENING.json, then
           OPENINGS.json -> both sealed files read once -> comparison and decision (PRESPEC_E4 sections 4-5), agreement,
           kappa, exact-zero share, the comparison without exact-zero rows (Ruling 46) -> E4_REPORT.json.  A failure
           after the opening writes E4_FAILED_test_<n>.json.

A BLOCKED condition (PRESPEC_E4 section 7) writes E4_BLOCKED_<stage>_<n>.json and ends the stage with status 2.  No
player identifier is read or written.

Usage: python -B -m studies.tog_experiments_20261012.e4_q {rows,prepare,train,select,qdev,lock}
       python -B -m studies.tog_experiments_20261012.e4_q test --confirm-single-opening --lock-sha256 <reviewed lock>
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005.assemble import PROJECTION  # noqa: E402
from studies.q_production_20261005 import jobs as qjobs, qselect, qtest, release as qrelease, rows as q_rows_module  # noqa: E402
from studies.q_production_20261005.rows import Q_ROLES, SEALED_COLUMNS  # noqa: E402
from studies.tog_experiments_20261012 import e2_run, e4_run as er, e4_vscore as vs  # noqa: E402
from studies.tog_experiments_20261012.common import (  # noqa: E402
    LABELS, Q_RUN, document_record, git_head, peak_memory_bytes, run_record, sha256_file, utc, write_once)

VERSION = "tog_e4_q_v1_20261013"
ROWS_VERSION = "tog_e4_rows_v1_20261013"
TOG_E4 = vs.TOG_E4
PRESPEC = vs.PRESPEC
PRESPEC_COMMIT = "c8e78b4c563ed27110072f0d11301acd977250b6"     # the author's approval of PRESPEC_E1..E5 (as of d78912e)
EXPOSURE = ROOT / "reports/tog_experiments_20261012/EXPOSURE_KO.md"
OPENINGS = ROOT / "reports/tog_experiments_20261012/OPENINGS.json"
RELEASE = ROOT / "reports/q_production_20261005/SCIENTIFIC_Q_15638e1f9d2fc7ce.json"
RELEASE_DIGEST = "15638e1f9d2fc7ce44110fc1dfbdc0731966b4322c3b29f927e4d8cbee417a1a"
WORKER = ROOT / "studies/q_production_20261005/q_worker.py"
PREPARE_SCRIPT = ROOT / "studies/q_production_20261005/q_prepare.py"
VARIANT, HORIZON = "main_h90", 90
SCOPES = ("FIT", "REFIT_TRAIN")
CONTEXTS = vs.CONTEXTS
TEST_ROLE = Q_ROLES["TEST"]
REPLACED = {"y": "y4", "label_svi_delta": "label_svi_delta4", "exact_zero": "exact_zero", "near_zero_bits": "near_zero_bits",
            "pt_p_pre": "p_pre"}                                   # row-table column <- E4 label field (PRESPEC_E4 section 3)
TEST_PLACEHOLDERS = {"y": -1, "label_svi_delta": float("nan"), "exact_zero": -1, "near_zero_bits": 0}   # rows.seal_test
LGBM_CONDITION = qjobs.FULL_CONDITIONS["lightgbm"]              # LGBM_S1_A1_M1_E1_C1_R1_D1
CARDS = {"lightgbm": ("lgbm_l15_m100", "lgbm_l15_m200", "lgbm_l31_m100", "lgbm_l31_m200"),
         "PT_flex": tuple(f"ptflex_k{k}_C{c}" for k in (4, 6, 8) for c in ("0.001", "0.01", "0.1", "1"))}
FAMILIES = tuple(CARDS)                                          # lightgbm first: the contrast is LightGBM - PT_flex
SPEC_KEYS = ("job", "variant", "inputs", "out_dir", "device", "release", "release_digest", "written_utc")  # q_supervisor.py:447-449
MAX_WORKERS = 4
THREADS = 2                     # q_run_1's DEFAULT_CONTROL threads (its CONTROL.json sets none); LightGBM itself uses 3 (qtrain)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
RETRIES = 1                     # q_supervisor DEFAULT_CONTROL retries
MAX_INTERRUPTS = 5              # q_supervisor DEFAULT_CONTROL max_interrupts
RULING_46 = {"labels": "3af35bca6247d86e6a769a1e81ce38f37d56f1e15b7e16a1c6c424e2fe766018",       # E4_LABELS_nonTEST.npz
             "agreement": "9b3680610e3749bb05913eb54d2a2f2e312ea801a8e324bd29df729cd978b9eb"}   # E4_QDEV_AGREEMENT.json
SEED, RESAMPLES, LEVEL = e2_run.SEED, e2_run.RESAMPLES, e2_run.LEVEL   # 20261011, 2000, 0.95 (PRESPEC_E4 section 4)
RECOMPUTE_ATOL = 1e-10
TIE_SHARE = 0.01                # PRESPEC_E4 section 4: the supplementary comparison when the TEST exact-zero share > 1 %
PRIMARY_AGREEMENT = ("share_equal", "kappa", "exact_zero_share_new")
MAIN_P1 = {"point": -0.01386, "interval": [-0.01445, -0.01328], "level": 0.975}   # PRESPEC_E4 section 4 (printed)
ENV_KEYS = ("python", "numpy", "scipy", "lightgbm", "sklearn")
ROWS_NAME, ROWS_RECORD, SEALED_NAME = "Q_ROWS_h90.npz", "Q_ROWS_h90.json", "Q_TEST_SEALED_h90.npz"
ROWS_STAGE, PREPARE_STAGE, TRAIN_STAGE = "E4_ROWS.json", "E4_PREPARE.json", "E4_TRAIN.json"
SELECTION_NAME, QDEV_NAME, LOCK_NAME = "E4_SELECTION.json", "E4_QDEV.json", "E4_LOCK.json"
OPENING_NAME, REPORT_NAME, SCORES_DIR = "E4_OPENING.json", "E4_REPORT.json", "TEST_SCORES"
STATE_NAME, LEDGER_NAME, DRIVER_NAME = "TRAIN_STATE.json", "TRAIN_LEDGER.jsonl", "TRAIN_DRIVER.json"
CONTEXT_FILES = ("CONTEXT.npz", "bundle/MANIFEST.json", "bundle/evidence.json", "bundle/numeric.json", "bundle/vocab.json")
CONTEXT_FIELDS = ("version", "scope", "scope_rows", "scope_positions_sha256", "sigma", "m_stats", "m_median", "release_digest",
                  "store_build_sha256", "xy_sha256", "rows_file", "xy_file", "variant")
TASK10_PATTERNS = ("E4_VSCORE.json", "PRED_*.npz", "PRED_*.json", "PRED_*/*", "QUERIES_*.npy", "VDEV_CHECK_*.json", "VDEV_*.npz",
                   "VDEV_*/*")
TOG_CODE = ("studies/tog_experiments_20261012/e4_q.py", "studies/tog_experiments_20261012/e4_run.py",
            "studies/tog_experiments_20261012/e4_vscore.py", "studies/tog_experiments_20261012/e2_run.py",
            "studies/tog_experiments_20261012/e2_groups.py", "studies/tog_experiments_20261012/common.py",
            "studies/tog_experiments_20261012/__init__.py")
FROZEN_CODE = ("studies/q_production_20261005/__init__.py", "studies/q_production_20261005/rows.py",
               "studies/q_production_20261005/qeval.py", "studies/q_production_20261005/qtest.py",
               "studies/q_production_20261005/qselect.py", "studies/q_production_20261005/jobs.py",
               "studies/q_production_20261005/release.py", "studies/q_production_20261005/q_worker.py",
               "studies/q_production_20261005/q_prepare.py", "studies/q_production_20261005/qtrain.py",
               "studies/q_production_20261005/qbatch.py", "studies/q_production_20261005/qbaselines.py",
               "studies/q_production_20261005/qcalibrate.py", "studies/q_production_20261005/qmodels.py",
               "studies/q_production_20261005/features.py", "studies/q_production_20261005/spatial.py",
               "studies/label_production_20261005/assemble.py", "studies/label_production_20261005/queries.py",
               "studies/label_production_20261005/engagements.py", "studies/v_production_20261004/source.py",
               "studies/v_production_20261004/train.py", "studies/v_model_fit_input_20261003/models.py",
               "studies/v_model_fit_input_20261003/common.py",
               "reports/q_production_20261005/q_supervisor.py")    # the spec shape (447-449); referenced, not imported
CODE_FILES = TOG_CODE + FROZEN_CODE
CORE_LABELS = ("prespec", "release", "rows:Q_ROWS_h90.npz", "rows:Q_ROWS_h90.json", "rows:Q_TEST_SEALED_h90.npz",
               "q_run_1:Q_TEST_SEALED_h90.npz", "q_run_1:Q_ROWS_h90.npz", "q_run_1:TEST_REPORT.json", "task10:PRED_final.npz",
               "labels_1:ENGAGEMENTS.npz", "labels_1:LABEL_LOCK.json", "task11:E4_LABELS_nonTEST.npz",
               "task11:E4_QDEV_AGREEMENT.json", "select:E4_SELECTION.json")
DECISIONS = {  # PRESPEC_E4 section 5, on the 95 % TEST interval of Brier(LightGBM) - Brier(PT_flex) (verbatim)
    "same_direction": "상한 < 0: \"같은 방향\"으로 주장 1에 측정기 강건성을 더한다(강건성 점검으로 서술).",
    "opposite_direction": "하한 > 0: \"반대 방향\"으로 그 사실을 표 1과 한계에 적고, 주장 1의 측정기 의존을 저자와 다시 본다.",
    "undetermined": "0을 포함: \"두 번째 측정기에서는 방향을 확인하지 못함\"으로 표 1과 한계에 적는다.",
}
DECISION_RULE = ("PRESPEC_E4 section 5: the 95% interval of the TEST difference Brier(LightGBM) - Brier(PT_flex) only; upper < 0 "
                 "-> same_direction; lower > 0 -> opposite_direction; otherwise (0 inside or on a bound) undetermined")
CONTRAST_TEXT = ("match-weighted raw Brier(selected LightGBM) - Brier(selected PT_flex) against the E4 labels y4; negative means "
                 "LightGBM is better (PRESPEC_E4 line 5; the opposite sign of E1's G)")
SEALED_SUMMARY = ("Ruling 28: handled as a sealed summary; nobody views it before the lock, code reads only its TRAIN/Q_DEV/Q_CAL "
                  "counts, its SHA-256 is in the lock.  It holds no TEST label figure: TEST counts are rows and matches only")
OPENING_NOTE = ("E4 TEST step: first opening of the E4 TEST labels (tog_e4_1/Q_TEST_SEALED_h90.npz) and a repeated look at the "
                "main TEST labels (q_run_1/Q_TEST_SEALED_h90.npz; first opened 2026-10-08 under q_run_1 SCORE_LOCK, again by E2 "
                "on 2026-10-12) for the agreement (PRESPEC_E4 sections 2 and 6)")
QDEV_POLICY = ("PRESPEC_E4 section 6: Q_DEV results are development evidence and decide nothing; this stage rehearses the TEST "
               "evaluation path on Q_DEV and checks it against Task 11's Q_DEV agreement")


@dataclass(frozen=True)
class Paths:
    """Where the stages read and write; the defaults are the real run (the tests point them at tmp_path)."""

    out: Path = TOG_E4
    q_run: Path = Q_RUN
    labels: Path = LABELS
    projection: Path = PROJECTION
    prespec: Path = PRESPEC
    exposure: Path = EXPOSURE
    openings: Path = OPENINGS
    release: Path = RELEASE

    @property
    def source_rows(self) -> Path:
        return self.q_run / "Q_ROWS_h90.npz"

    @property
    def source_sealed(self) -> Path:
        return self.q_run / "Q_TEST_SEALED_h90.npz"

    @property
    def rows(self) -> Path:
        return self.out / ROWS_NAME

    @property
    def rows_record(self) -> Path:
        return self.out / ROWS_RECORD

    @property
    def sealed(self) -> Path:
        return self.out / SEALED_NAME

    @property
    def inputs(self) -> Path:
        return self.out / "inputs" / VARIANT

    @property
    def base_inputs(self) -> Path:
        return self.q_run / "inputs" / VARIANT

    @property
    def store_link(self) -> Path:
        return self.out / "raw" / "store"

    @property
    def store_target(self) -> Path:
        return self.q_run / "raw" / "store"

    @property
    def jobs(self) -> Path:
        return self.out / "jobs"


# ----------------------------------------------------------------------------- small helpers
def _read(path: Path) -> Any:
    return vs._read(path)


def _clean(value: Any) -> Any:
    """A JSON-ready copy equal to what reading the written file back gives (numpy to Python, NaN to null)."""

    return json.loads(json.dumps(vs._json_safe(value)))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _load_fields(path: Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in fields}


def _hash(path: Path, cache: dict[str, str]) -> str:
    return vs._hashes([path], cache)[Path(path).as_posix()]


def _file_entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _atomic_json(path: Path, payload: Any) -> None:
    """Bookkeeping files of the train driver (state, driver claim), rewritten atomically; results are written once."""

    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    pending.write_text(json.dumps(_clean(payload), indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending, path)


def _next_free(out: Path, stem: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    k = 1
    while (out / f"{stem}_{k}.json").exists():
        k += 1
    return out / f"{stem}_{k}.json"


def _blocked(out: Path, stage: str, reasons: Any, **extra: Any) -> int:
    """Write E4_BLOCKED_<stage>_<n>.json once (n = the first free number, so earlier attempts stay) and return 2."""

    path = _next_free(Path(out), f"E4_BLOCKED_{stage}")
    write_once(path, _clean({"version": VERSION, "experiment": "E4", "blocked_stage": stage, "blocked": True, "utc": utc(),
                             "reasons": reasons, **extra}))
    print(json.dumps({"BLOCKED": stage, "file": str(path), "reasons": _clean(reasons)}, ensure_ascii=False), flush=True)
    return 2


def _check_files(files: Mapping[str, Mapping[str, str]], stage: str) -> None:
    changed = [label for label, entry in files.items()
               if (sha256_file(entry["path"]) if Path(entry["path"]).is_file() else None) != entry["sha256"]]
    if changed:
        raise RuntimeError(f"files written by the {stage} stage changed: {changed}")


def _passed(path: Path, stage: str) -> dict[str, Any]:
    """The record of an earlier stage: present, passed, and every file it lists unchanged."""

    if not Path(path).is_file():
        raise RuntimeError(f"{Path(path).name} is missing: run the {stage} stage first")
    record = _read(path)
    if record.get("passed") is not True:
        raise RuntimeError(f"the {stage} stage has not passed")
    _check_files(record.get("files") or {}, stage)
    return record


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def environment() -> dict[str, str]:
    import lightgbm
    import scipy
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__, "implementation": platform.python_implementation(),
            "executable": sys.executable, "platform": platform.platform()}


def _common(paths: Paths) -> dict[str, Any]:
    return {"code": document_record(Path(__file__)), "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
            "prespec": document_record(paths.prespec), "git_head": git_head(), "versions": vs._versions()}


# ----------------------------------------------------------------------------- the row table (PRESPEC_E4 section 3)
def labels_of_test_rows(rows: Mapping[str, np.ndarray], query_post: Any, pred_final: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """E4 labels of the row table's TEST rows in row-table order: the final context at both ends
    (e4_vscore.row_contexts), y4 / exact_zero / near_zero_bits by e4_run.direction_labels and zero_flags, the difference
    p_post - p_pre, and p_pre (the PT_flex input).  These values go only into the sealed file; nothing summarizes them."""

    role = np.asarray(rows["role"]).astype(np.int64)
    query_post = np.asarray(query_post, dtype=np.int64)
    if query_post.shape != role.shape:
        raise ValueError("query_post must give every row's endpoint query")
    q_row = np.flatnonzero(role == TEST_ROLE)
    if np.any(vs.row_contexts(role[q_row], np.asarray(rows["fold"])[q_row]) != 0):
        raise ValueError("a TEST row is not scored by the final context")
    p_pre = er.probabilities_at(pred_final, np.asarray(rows["query_pre"], dtype=np.int64)[q_row], "final TEST p_pre")
    p_post = er.probabilities_at(pred_final, query_post[q_row], "final TEST p_post")
    return {"q_row": q_row.astype(np.int64), "engagement_key": np.asarray(rows["engagement_key"])[q_row],
            "y4": er.direction_labels(p_pre, p_post), "label_svi_delta4": p_post - p_pre, **er.zero_flags(p_pre, p_post),
            "p_pre": p_pre, "p_post": p_post}


def e4_row_table(source: Mapping[str, np.ndarray], nontest: Mapping[str, np.ndarray],
                 test: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(row table, sealed TEST labels): the source table with the REPLACED columns set from the E4 labels of every row
    (``nontest``: e4_run.nontest_labels; ``test``: labels_of_test_rows), then split by the frozen rows.seal_test, so the
    TEST rows keep y = -1, NaN, -1, 0 and their labels go to the sealed dict (engagement_key + rows.SEALED_COLUMNS).
    The two label parts must cover every row once with the row table's keys; a value the column dtype cannot hold
    raises."""

    role = np.asarray(source["role"]).astype(np.int64)
    n = role.size
    count = np.zeros(n, dtype=np.int64)
    for part in (nontest, test):
        q_row = np.asarray(part["q_row"], dtype=np.int64)
        if q_row.ndim != 1 or np.any(q_row < 0) or np.any(q_row >= n):
            raise ValueError("label rows outside the row table")
        np.add.at(count, q_row, 1)
        if not np.array_equal(np.asarray(source["engagement_key"])[q_row], np.asarray(part["engagement_key"])):
            raise ValueError("the labelled rows' engagement keys differ from the row table's")
    if not (count == 1).all():
        raise ValueError("the E4 labels must cover every row-table row exactly once")
    if np.any(role[np.asarray(test["q_row"])] != TEST_ROLE) or np.any(role[np.asarray(nontest["q_row"])] == TEST_ROLE):
        raise ValueError("the TEST labels must be exactly the TEST rows")
    full = {name: np.array(value, copy=True) for name, value in source.items()}
    for column, field in REPLACED.items():
        for part in (nontest, test):
            q_row = np.asarray(part["q_row"], dtype=np.int64)
            values = np.asarray(part[field])
            full[column][q_row] = values
            if not np.array_equal(full[column][q_row], values):
                raise ValueError(f"{column}: the E4 values do not fit the row table's dtype {full[column].dtype}")
    return q_rows_module.seal_test(full)


def _placeholder_ok(column: np.ndarray, value: float | int) -> bool:
    column = np.asarray(column)
    if isinstance(value, float) and math.isnan(value):
        return bool(np.isnan(column).all())
    return bool((column == value).all())


def public_problems(source: Mapping[str, np.ndarray], public: Mapping[str, np.ndarray]) -> list[str]:
    """Differences between the E4 row table and its source other than the five replaced columns: column set and order,
    dtypes, every other value, and the sealed placeholders on TEST rows (in both tables)."""

    problems: list[str] = []
    if list(public) != list(source):
        problems.append("the columns or their order differ from the source row table")
    test = np.asarray(source["role"]).astype(np.int64) == TEST_ROLE
    for name, value in source.items():
        if name not in public:
            continue
        left, right = np.asarray(value), np.asarray(public[name])
        if left.dtype != right.dtype or left.shape != right.shape:
            problems.append(f"{name}: dtype or shape differs from the source row table")
            continue
        if name in REPLACED:
            continue                     # replaced on purpose; their TEST rows are checked against the placeholders below
        if not np.array_equal(left, right, equal_nan=left.dtype.kind == "f"):
            problems.append(f"{name}: values differ from the source row table")
    for name, value in TEST_PLACEHOLDERS.items():
        for label, table in (("source", source), ("E4", public)):
            if name in table and not _placeholder_ok(np.asarray(table[name])[test], value):
                problems.append(f"{name}: the {label} row table does not hold the sealed placeholder {value} on TEST rows")
    return problems


def sealed_problems(source: Mapping[str, np.ndarray], sealed: Mapping[str, np.ndarray]) -> list[str]:
    """The sealed file's layout (engagement_key + rows.SEALED_COLUMNS, the row table's TEST keys in order and dtypes) and
    validity checks that stop the run (labels 0/1, a tie negative, finite differences); nothing is counted."""

    expected = ("engagement_key", *SEALED_COLUMNS)
    if tuple(sealed) != expected:
        return [f"the sealed columns are {list(sealed)}, not {list(expected)} (rows.py SEALED_COLUMNS)"]
    problems: list[str] = []
    test = np.asarray(source["role"]).astype(np.int64) == TEST_ROLE
    keys = np.asarray(source["engagement_key"])[test]
    if np.asarray(sealed["engagement_key"]).dtype != keys.dtype or not np.array_equal(sealed["engagement_key"], keys):
        problems.append("the sealed engagement keys are not the row table's TEST rows in order")
    for name in SEALED_COLUMNS:
        column = np.asarray(sealed[name])
        if column.dtype != np.asarray(source[name]).dtype or column.shape != keys.shape:
            problems.append(f"sealed {name}: dtype or length differs from the row table's column")
    if problems:
        return problems
    y, exact = np.asarray(sealed["y"]), np.asarray(sealed["exact_zero"])
    if not np.isin(y, (0, 1)).all():
        problems.append("sealed y holds values other than 0 and 1")
    if not np.isin(exact, (0, 1)).all() or np.any((exact == 1) & (y == 1)):
        problems.append("sealed exact_zero is not 0/1 or a tie is labelled 1")
    if not np.isfinite(np.asarray(sealed["label_svi_delta"], dtype=np.float64)).all():
        problems.append("sealed label_svi_delta is not finite")
    if np.any(np.asarray(sealed["near_zero_bits"]) > 7):
        problems.append("sealed near_zero_bits uses bits beyond the three thresholds")
    return problems


def nontest_differences(recomputed: Mapping[str, np.ndarray], stored: Mapping[str, np.ndarray]) -> list[str]:
    """Columns in which the recomputed TRAIN/Q_DEV/Q_CAL labels differ from E4_LABELS_nonTEST.npz (dtype, shape or value)."""

    differ: list[str] = []
    if set(recomputed) != set(stored):
        differ.append(f"columns differ: {sorted(set(recomputed) ^ set(stored))}")
    for name in sorted(set(recomputed) & set(stored)):
        left, right = np.asarray(recomputed[name]), np.asarray(stored[name])
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            differ.append(name)
    return differ


def row_counts(public: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The counts block of the frozen rows.write_rows record (rows.py:148-158) for the E4 row table, with label figures
    for TRAIN, Q_DEV and Q_CAL only: TEST holds rows and matches (public, PRESPEC_E4 section 2), no label figure."""

    role = np.asarray(public["role"]).astype(np.int64)
    match = np.asarray(public["match_index"])
    y, weight = np.asarray(public["y"]), np.asarray(public["weight_raw"], dtype=np.float64)
    counts: dict[str, Any] = {"rows": int(role.size)}
    for name, code in Q_ROLES.items():
        mask = role == code
        entry: dict[str, Any] = {"rows": int(mask.sum()), "matches": int(np.unique(match[mask]).size)}
        if code == TEST_ROLE:
            entry["labels"] = "sealed in Q_TEST_SEALED_h90.npz; no TEST label figure is computed (PRESPEC_E4 section 3)"
        elif mask.any():
            values, w = y[mask].astype(np.float64), weight[mask]
            if not np.isin(values, (0.0, 1.0)).all():
                raise ValueError(f"{name} rows lack a binary E4 label")
            entry.update(positive=int(values.sum()), positive_rate_unit=float(values.mean()),
                         positive_rate_match_weighted=float((w * values).sum() / w.sum()),
                         exact_zero=int((np.asarray(public["exact_zero"])[mask] == 1).sum()))
        counts[name] = entry
    train, stop = role == Q_ROLES["TRAIN"], np.asarray(public["q_stop"], dtype=bool)
    counts["TRAIN"].update(stop_rows=int((train & stop).sum()), stop_matches=int(np.unique(match[train & stop]).size),
                           fit_rows=int((train & ~stop).sum()), fit_matches=int(np.unique(match[train & ~stop]).size))
    return counts


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """np.savez bytes (uncompressed, fixed zip timestamps): the same bytes write_npz_once / assemble.write_npz write."""

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def write_bytes_once(path: Path, data: bytes) -> str:
    """Write bytes once and return their SHA-256.  An existing file with the same bytes is kept (a restarted stage
    recomputes deterministically and compares hashes without opening the file); other bytes raise."""

    path, digest = Path(path), hashlib.sha256(data).hexdigest()
    if path.exists():
        if sha256_file(path) != digest:
            raise FileExistsError(f"{path} exists with other bytes; it is written once")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    if sha256_file(path) != digest:
        raise RuntimeError(f"{path}: the written bytes differ")
    return digest


def rows_stage(paths: Paths = Paths(), *, role_rows: Mapping[str, int] = vs.Q_ROLE_ROWS,
               expected: Mapping[str, str] = RULING_46) -> int:
    """PRESPEC_E4 section 6 step 4 (first part): the E4 row table, the sealed TEST labels, the record and E4_ROWS.json."""

    started, tick = utc(), time.perf_counter()
    out = paths.out
    stage_path = out / ROWS_STAGE
    if stage_path.exists():
        _check_files(_read(stage_path)["files"], "rows")
        print(json.dumps({"done": str(stage_path)}), flush=True)
        return 0
    hashes: dict[str, str] = {}
    stages = []

    # inputs: Task 10's records and tables, the label lock, the two Task 11 files bound by Ruling 46 --------------
    row_inputs = (paths.source_rows, paths.labels / "ENGAGEMENTS.npz", paths.labels / "LABEL_LOCK.json",
                  paths.projection / "CONTEXT_MEMBERSHIP.npy")
    _vscore, _records, task10_problems = er.check_task10(out, hashes, row_inputs, role_rows)
    problems = list(task10_problems)
    try:
        lock = q_rows_module.verify_lock(paths.labels, HORIZON)
    except (OSError, ValueError) as error:
        lock = {}
        problems.append(f"label lock: {error}")
    labels_path, agreement_path = out / er.LABELS_NAME, out / er.AGREEMENT_NAME
    ruling_46 = {}
    for path, digest in ((labels_path, expected["labels"]), (agreement_path, expected["agreement"])):
        now = sha256_file(path) if path.is_file() else None
        ruling_46[path.name] = now == digest
        if now != digest:
            problems.append(f"{path.name} has SHA-256 {now}, not the file bound by Ruling 46 ({digest})")
    if not problems:
        agreement = _read(agreement_path)
        if (agreement.get("labels") or {}).get("sha256") != expected["labels"]:
            problems.append(f"{agreement_path.name} describes another {labels_path.name}")
        if (agreement.get("inputs") or {}).get(paths.source_rows.as_posix()) != _hash(paths.source_rows, hashes):
            problems.append("Task 11 labelled another row table than the source Q_ROWS_h90.npz")
    stages.append(vs._stage("inputs", started, tick))
    if problems:
        return _blocked(out, "rows", problems)

    # rows: the source table aligned to the locked engagements, role counts, keys, OOF exclusion of every row --------
    s, t = utc(), time.perf_counter()
    source = _load_npz(paths.source_rows)
    with np.load(paths.labels / "ENGAGEMENTS.npz", allow_pickle=False) as stored:
        engagements = {name: stored[name] for name in (*vs.ALIGN_COLUMNS, "query_h90", "valid_h90")}
    try:
        keep = vs.align_q_rows(engagements, source)
    except ValueError as error:
        return _blocked(out, "rows", [f"row table key mismatch: {error}"])
    query_post = engagements["query_h90"][keep].astype(np.int64)
    del engagements
    role = source["role"].astype(np.int64)
    by_role = {name: int((role == code).sum()) for name, code in Q_ROLES.items()}
    matches_by_role = {name: int(np.unique(source["match_index"][role == code]).size) for name, code in Q_ROLES.items()}
    role_counts_equal = by_role == dict(role_rows)
    if not role_counts_equal:
        problems.append(f"rows by role {by_role} differ from PRESPEC_E4's {dict(role_rows)}")
    canonical = np.array_equal(np.char.add(np.char.add(source["canonical_match_key"], b":"), source["component_key"]),
                               source["engagement_key"])
    unique = int(np.unique(source["engagement_key"]).size) == role.size
    if not (canonical and unique):
        problems.append("engagement keys are not unique canonical_match_key:component_key keys")
    frozen_keys = np.load(paths.labels / "Q_DEV_H90_KEYS.npy", allow_pickle=False)
    outside = int((~np.isin(source["engagement_key"][role == Q_ROLES["Q_DEV"]], frozen_keys)).sum())
    if outside:
        problems.append(f"{outside} Q_DEV rows are outside the keys frozen before predictions")
    codes = vs.row_contexts(role, source["fold"])
    membership = np.load(paths.projection / "CONTEXT_MEMBERSHIP.npy", allow_pickle=False)
    leaked = vs.oof_leaks(codes, role, source["match_index"], membership)
    del membership
    oof = {"rule": "assemble.py:109-114 (e4_vscore.oof_leaks) on every row: split bits 1-16 of the scoring context are 0 "
                   "for the row's match; a TRAIN match carries the held-out target bit 32 of its oof context",
           "rows_checked": int(role.size), "leaked": int(leaked.sum()), "passed": bool(not leaked.any())}
    if leaked.any():
        problems.append(f"OOF exclusion failed on {int(leaked.sum())} rows")
    stages.append(vs._stage("rows", s, t))
    if problems:
        return _blocked(out, "rows", problems, oof_check=oof)

    # labels: TRAIN/Q_DEV/Q_CAL recomputed (= Task 11's file), TEST made from PRED_final; the table and the seal --------
    s, t = utc(), time.perf_counter()
    preds = {ctx: _load_fields(out / f"PRED_{ctx}.npz", ("query_index", "raw_p")) for ctx in CONTEXTS}
    try:
        nontest = er.nontest_labels(source, query_post, preds)
        test = labels_of_test_rows(source, query_post, preds["final"])
    except ValueError as error:
        return _blocked(out, "rows", [f"labels: {error}"])
    del preds
    differ = nontest_differences(nontest, _load_npz(labels_path))
    if differ:
        return _blocked(out, "rows", [f"the recomputed TRAIN/Q_DEV/Q_CAL labels differ from {labels_path.name} in {differ}"])
    try:
        public, sealed = e4_row_table(source, nontest, test)
    except ValueError as error:
        return _blocked(out, "rows", [f"row table: {error}"])
    del test, nontest
    table_problems = public_problems(source, public) + sealed_problems(source, sealed)
    if table_problems:
        return _blocked(out, "rows", table_problems)
    stages.append(vs._stage("labels_and_table", s, t))

    # write: the sealed file, the row table, the frozen-format record (rows.write_rows order), then E4_ROWS.json ------
    s, t = utc(), time.perf_counter()
    if paths.rows_record.exists():
        raise FileExistsError(f"{paths.rows_record} exists without {ROWS_STAGE}: an earlier rows run stopped part way")
    sealed_sha = write_bytes_once(paths.sealed, npz_bytes(sealed))
    del sealed
    rows_sha = write_bytes_once(paths.rows, npz_bytes(public))
    source_sha = _hash(paths.source_rows, hashes)
    counts = row_counts(public)
    counts["q_dev_frozen_keys"] = {"frozen": int(frozen_keys.size), "q_rows": by_role["Q_DEV"], "q_rows_outside_frozen": outside}
    record = {"version": ROWS_VERSION, "format": "the frozen rows.write_rows record (rows.py:233-239) for the E4 row table",
              "utc": utc(), "horizon_s": HORIZON, "label_lock_sha256": _hash(paths.labels / "LABEL_LOCK.json", hashes),
              "label_lock_utc": lock.get("locked_utc"), "labels_sha256": _hash(labels_path, hashes),
              "test_labels_source_sha256": _hash(out / "PRED_final.npz", hashes), "q_stop_salt": q_rows_module.Q_STOP_SALT,
              "restricted_columns": list(q_rows_module.RESTRICTED), "sealed_test_columns": list(SEALED_COLUMNS),
              "code_sha256": sha256_file(Path(__file__)), "counts": counts, "test_sealed_sha256": sealed_sha,
              "rows_sha256": rows_sha, "source_rows_sha256": source_sha, "replaced_columns": list(REPLACED),
              "sealed_summary": SEALED_SUMMARY}
    write_once(paths.rows_record, _clean(record))
    record_sha = sha256_file(paths.rows_record)
    stages.append(vs._stage("write", s, t))
    inputs = {**vs._hashes([out / "E4_VSCORE.json"], hashes),
              **vs._hashes([out / f"PRED_{ctx}.{ext}" for ctx in CONTEXTS for ext in ("npz", "json")], hashes),
              **vs._hashes([labels_path, agreement_path, *row_inputs, paths.labels / "Q_DEV_H90_KEYS.npy"], hashes)}
    stage = {
        "version": VERSION, "experiment": "E4", "stage": "rows", "passed": True,
        "files": {"rows": {"path": str(paths.rows), "sha256": rows_sha}, "sealed": {"path": str(paths.sealed), "sha256": sealed_sha},
                  "record": {"path": str(paths.rows_record), "sha256": record_sha}},
        "source": {"path": str(paths.source_rows), "sha256": source_sha},
        "replaced_columns": dict(REPLACED), "sealed_columns": ["engagement_key", *SEALED_COLUMNS],
        "test_placeholders": dict(TEST_PLACEHOLDERS), "rows": int(role.size), "rows_by_role": by_role,
        "matches_by_role": matches_by_role,
        "definitions": {
            "labels": "y4 = 1[p_post - p_pre > 0]; a tie is 0 with exact_zero 1; near_zero_bits bit k when |p_post - p_pre| < "
                      "(0.005, 0.01, 0.02)[k]; label_svi_delta = p_post - p_pre; pt_p_pre = p_pre (raw p of PRED_<ctx>.npz)",
            "contexts": "TRAIN rows oof<fold>, Q_DEV/Q_CAL/TEST rows final; p_pre at query_pre, p_post at ENGAGEMENTS query_h90",
            "test_rows": "labels made from PRED_final and written only into Q_TEST_SEALED_h90.npz by rows.seal_test; the file is "
                         "hashed, not opened again before the lock",
            "record": SEALED_SUMMARY},
        "checks": {"task10_records_and_hashes": not task10_problems, "label_lock_verified": bool(lock),
                   "ruling_46_files": all(ruling_46.values()),
                   "row_table_aligned_to_engagements": bool(keep.sum() == role.size), "role_counts_equal_prespec": role_counts_equal,
                   "engagement_keys_unique_and_canonical": bool(canonical and unique), "q_dev_keys_inside_frozen_keys": outside == 0,
                   "oof_exclusion": oof["passed"], "nontest_labels_equal_task11": not differ,
                   "table_and_seal_checks": not table_problems},
        "oof_check": oof, "inputs": inputs, **_common(paths), "timing": stages, "peak_memory_bytes": peak_memory_bytes(),
        "run_started_utc": started, "utc": utc(),
        "notes": ["no TEST label figure was computed, printed or stored (PRESPEC_E4 section 3); Q_TEST_SEALED_h90.npz is "
                  "known here by its SHA-256 only", "Q_ROWS_h90.json: " + SEALED_SUMMARY]}
    write_once(stage_path, _clean(stage))
    print(json.dumps({"done": str(stage_path), "rows_by_role": by_role, "rows_sha256": rows_sha, "test_sealed_sha256": sealed_sha,
                      "record_sha256": record_sha}), flush=True)
    return 0


# ----------------------------------------------------------------------------- preprocessing (PRESPEC_E4 section 3)
def ensure_store_link(link: Path, target: Path) -> dict[str, Any]:
    """``link`` as a directory junction to ``target`` (made with mklink /J when missing); an existing path must already
    be a junction to the same directory."""

    link, target = Path(link), Path(target)
    created = False
    if not os.path.lexists(link):
        if not target.is_dir():
            raise FileNotFoundError(f"{target} is not a directory")
        link.parent.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False)
        if done.returncode != 0:
            raise OSError(f"mklink /J failed: {done.stdout.decode(errors='replace')} {done.stderr.decode(errors='replace')}")
        created = True
    if not os.path.isjunction(link):
        raise ValueError(f"{link} exists and is not a directory junction")
    if not os.path.samefile(link, target):
        raise ValueError(f"{link} is a junction to another directory than {target}")
    return {"link": str(link), "target": str(target), "created": created, "is_junction": True,
            "readlink": os.readlink(link), "realpath": os.path.realpath(link)}


def copy_verified(source: Path, target: Path, sha256: str) -> dict[str, Any]:
    """Copy ``source`` to ``target`` once after checking its SHA-256; the copy must hash the same.  An existing target
    with the same bytes is kept; other bytes raise."""

    source, target = Path(source), Path(target)
    if sha256_file(source) != sha256:
        raise ValueError(f"{source} differs from its recorded SHA-256")
    created = False
    if target.exists():
        if sha256_file(target) != sha256:
            raise FileExistsError(f"{target} exists with other bytes")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
        shutil.copyfile(source, pending)
        if sha256_file(pending) != sha256:
            raise ValueError(f"the copy {pending} differs from {source}")
        os.replace(pending, target)
        created = True
    return {"source": str(source), "target": str(target), "sha256": sha256, "created": created}


def context_comparison(new: Path, base: Path) -> dict[str, Any]:
    """One scope's new context against q_run_1's: CONTEXT.npz, bundle/MANIFEST.json and the bundle data files (new file,
    q_run_1 file and q_run_1's CONTEXT.json), and the CONTEXT.json fields; rows_sha256 differs by design and is shown,
    not compared."""

    new, base = Path(new), Path(base)
    new_record, base_record = _read(new / "CONTEXT.json"), _read(base / "CONTEXT.json")
    files = {name: {"new": sha256_file(new / name), "q_run_1": sha256_file(base / name)} for name in CONTEXT_FILES}
    npz, manifest = files["CONTEXT.npz"], files["bundle/MANIFEST.json"]
    return {"files": files,
            "context_npz_equal": npz["new"] == npz["q_run_1"] == base_record.get("context_npz_sha256"),
            "bundle_manifest_equal": manifest["new"] == manifest["q_run_1"] == base_record.get("bundle_manifest_sha256"),
            "bundle_files_equal": all(files[name]["new"] == files[name]["q_run_1"] for name in CONTEXT_FILES[2:]),
            "new_record_names_its_files": (new_record.get("context_npz_sha256") == npz["new"]
                                           and new_record.get("bundle_manifest_sha256") == manifest["new"]),
            "record_fields_equal": {key: new_record.get(key) == base_record.get(key) for key in CONTEXT_FIELDS},
            "q_run_1_context_json": {"context_npz_sha256": base_record.get("context_npz_sha256"),
                                     "bundle_manifest_sha256": base_record.get("bundle_manifest_sha256")},
            "rows_sha256": {"new": new_record.get("rows_sha256"), "q_run_1": base_record.get("rows_sha256"),
                            "note": "differs by design (PRESPEC_E4 section 3); not compared"}}


def context_problems(scope: str, comparison: Mapping[str, Any]) -> list[str]:
    problems = []
    if not comparison["context_npz_equal"]:
        problems.append(f"{scope}: CONTEXT.npz differs from q_run_1's (PRESPEC_E4 section 7)")
    if not comparison["bundle_manifest_equal"]:
        problems.append(f"{scope}: bundle/MANIFEST.json differs from q_run_1's (PRESPEC_E4 section 7)")
    if not comparison["bundle_files_equal"]:
        problems.append(f"{scope}: bundle data files differ from q_run_1's")
    if not comparison["new_record_names_its_files"]:
        problems.append(f"{scope}: the new CONTEXT.json does not name its own files")
    problems += [f"{scope}: CONTEXT.json {key} differs from q_run_1's" for key, ok in comparison["record_fields_equal"].items() if not ok]
    return problems


def worker_env() -> dict[str, str]:
    """The q supervisor's worker environment (q_supervisor.py _env, threads 2) with the GPU hidden (CPU only)."""

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
           "CUDA_VISIBLE_DEVICES": ""}
    env.update({name: str(THREADS) for name in THREAD_VARIABLES})
    return env


def run_q_prepare(paths: Paths) -> dict[str, Any]:
    """The frozen q_prepare.py as a subprocess (cwd = repository root), stdout and stderr to tog_e4_1/logs."""

    logs = paths.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = utc().replace(":", "").replace("-", "")[:15]
    command = [sys.executable, "-B", str(PREPARE_SCRIPT), "--run", paths.out.as_posix(), "--variant", VARIANT,
               "--release", _relative(paths.release)]
    stdout, stderr = logs / f"q_prepare_{stamp}.stdout.log", logs / f"q_prepare_{stamp}.stderr.log"
    started, tick = utc(), time.perf_counter()
    with stdout.open("wb") as out_stream, stderr.open("wb") as err_stream:
        done = subprocess.run(command, cwd=str(ROOT), env=worker_env(), stdout=out_stream, stderr=err_stream, check=False)
    return {"command": command, "cwd": str(ROOT), "returncode": done.returncode, "stdout_log": str(stdout),
            "stderr_log": str(stderr), "started_utc": started, "finished_utc": utc(), "wall_seconds": time.perf_counter() - tick,
            "threads": THREADS}


def prepare_stage(paths: Paths = Paths(), *, runner: Callable[[Paths], Mapping[str, Any]] | None = None) -> int:
    """PRESPEC_E4 section 6 step 4 (second part): store junction, XY copy, the frozen q_prepare, context equality."""

    started, tick = utc(), time.perf_counter()
    out = paths.out
    target = out / PREPARE_STAGE
    if target.exists():
        _check_files(_read(target)["files"], "prepare")
        print(json.dumps({"done": str(target)}), flush=True)
        return 0
    rows = _passed(out / ROWS_STAGE, "rows")
    base = _read(paths.base_inputs / "PREPARED.json")
    hashes: dict[str, str] = {}
    build_sha = _hash(paths.store_target / "BUILD.json", hashes)
    xy_sha = _hash(paths.base_inputs / "XY.npy", hashes)
    problems = []
    if build_sha != base.get("store_build_sha256"):
        problems.append("q_run_1 raw/store/BUILD.json differs from q_run_1 PREPARED.json store_build_sha256 (PRESPEC_E4 section 7)")
    if xy_sha != base.get("xy_sha256"):
        problems.append("q_run_1 inputs/main_h90/XY.npy differs from q_run_1 PREPARED.json xy_sha256 (PRESPEC_E4 section 7)")
    if base.get("release_digest") != RELEASE_DIGEST:
        problems.append("q_run_1 was prepared under another q release")
    try:
        release = qrelease.verify(paths.release)
        if release["digest"] != RELEASE_DIGEST:
            problems.append(f"release digest {release['digest']} is not {RELEASE_DIGEST}")
    except RuntimeError as error:
        problems.append(f"release: {error}")
    if problems:
        return _blocked(out, "prepare", problems)
    link = ensure_store_link(paths.store_link, paths.store_target)
    link["build_sha256_via_link"] = sha256_file(paths.store_link / "BUILD.json")
    if link["build_sha256_via_link"] != build_sha:
        return _blocked(out, "prepare", ["BUILD.json read through the junction differs from q_run_1's"], store_link=link)
    xy = copy_verified(paths.base_inputs / "XY.npy", paths.inputs / "XY.npy", xy_sha)
    prepared_path = paths.inputs / "PREPARED.json"
    run = None
    if not prepared_path.exists():
        run = dict((runner or run_q_prepare)(paths))
        if run["returncode"] != 0 or not prepared_path.exists():
            return _blocked(out, "prepare", [f"q_prepare.py exited with {run['returncode']} (logs: {run.get('stderr_log')})"],
                            q_prepare=run, store_link=link, xy=xy)
    prepared = _read(prepared_path)
    comparisons = {scope: context_comparison(paths.inputs / scope, paths.base_inputs / scope) for scope in SCOPES}
    problems = [problem for scope in SCOPES for problem in context_problems(scope, comparisons[scope])]
    prepared_checks = {"rows_sha256_is_the_e4_row_table": prepared.get("rows_sha256") == rows["files"]["rows"]["sha256"],
                       "rows_file": prepared.get("rows_file") == ROWS_NAME,
                       "rows": prepared.get("rows") == rows["rows"],
                       "xy_sha256_equals_q_run_1": prepared.get("xy_sha256") == xy_sha,
                       "store_build_sha256_equals_q_run_1": prepared.get("store_build_sha256") == build_sha,
                       "release_digest": prepared.get("release_digest") == RELEASE_DIGEST,
                       "variant": prepared.get("variant") == VARIANT,
                       "contexts_listed": sorted(prepared.get("contexts") or {}) == sorted(SCOPES)}
    problems += [f"PREPARED.json check failed: {name}" for name, ok in prepared_checks.items() if not ok]
    if problems:
        return _blocked(out, "prepare", problems, comparisons=comparisons, prepared_checks=prepared_checks, q_prepare=run)
    files = {f"inputs/{VARIANT}/{path.relative_to(paths.inputs).as_posix()}": _file_entry(path)
             for path in sorted(paths.inputs.rglob("*")) if path.is_file()}
    files["raw/store/BUILD.json (junction)"] = _file_entry(paths.store_link / "BUILD.json")
    record = {"version": VERSION, "experiment": "E4", "stage": "prepare", "passed": True, "files": files,
              "command": run["command"] if run else "PREPARED.json already present: q_prepare was not run again",
              "q_prepare": run, "store_link": link, "xy": xy, "prepared": prepared, "comparisons": comparisons,
              "prepared_checks": prepared_checks,
              "q_run_1": {"prepared": str(paths.base_inputs / "PREPARED.json"), "store_build_sha256": build_sha, "xy_sha256": xy_sha},
              "evidence": "CONTEXT.npz and bundle/MANIFEST.json of FIT and REFIT_TRAIN equal q_run_1's: inputs and "
                          "preprocessing read no label (PRESPEC_E4 section 3)",
              **_common(paths), "timing": vs._stage("prepare", started, tick), "utc": utc()}
    write_once(target, _clean(record))
    print(json.dumps({"done": str(target), "contexts_equal": {scope: comparisons[scope]["context_npz_equal"]
                                                               and comparisons[scope]["bundle_manifest_equal"] for scope in SCOPES}}),
          flush=True)
    return 0


# ----------------------------------------------------------------------------- jobs and the train driver
def plan_jobs(plan: Mapping[str, Sequence[Any]]) -> list[Any]:
    """The frozen plan's search jobs of PRESPEC_E4's 16 cards, LightGBM first, each family in card order."""

    search = list(plan["search"])
    chosen = {"lightgbm": [j for j in search if j.block == "auxiliary" and j.family == "lightgbm" and j.condition_id == LGBM_CONDITION],
              "PT_flex": [j for j in search if j.block == "reference" and j.family == "PT_flex"]}
    for family, cards in CARDS.items():
        found = tuple(job.card_id for job in chosen[family])
        if found != cards:
            raise ValueError(f"the frozen plan's {family} cards {found} are not PRESPEC_E4's {cards}")
        if any(job.stage != "search" or job.variant != VARIANT for job in chosen[family]):
            raise ValueError(f"a {family} job is not a main_h90 search job")
    return chosen["lightgbm"] + chosen["PT_flex"]


def variant_inputs(run: Path, prepared: Mapping[str, Any]) -> dict[str, Any]:
    """q_supervisor.Supervisor.variant_inputs for the prepared main_h90 inputs of ``run``."""

    return {"rows": str(run / prepared["rows_file"]), "rows_sha256": prepared["rows_sha256"],
            "xy": str(run / prepared.get("xy_file", f"inputs/{VARIANT}/XY.npy")), "xy_sha256": prepared["xy_sha256"],
            "store": str(run / "raw" / "store"),
            "contexts": {scope: str(run / info["dir"]) for scope, info in prepared["contexts"].items()}}


def job_spec(job: Any, run: Path, prepared: Mapping[str, Any], release: Path, digest: str, written_utc: str) -> dict[str, Any]:
    """One job spec in the q_supervisor.py:447-449 shape (device cpu)."""

    run = Path(run)
    return {"job": job.to_dict(), "variant": job.variant, "inputs": variant_inputs(run, prepared),
            "out_dir": str(run / "jobs" / qtest.key_of(job.job_id)), "device": "cpu", "release": str(release),
            "release_digest": digest, "written_utc": written_utc}


def ensure_spec(directory: Path, spec: Mapping[str, Any]) -> Path:
    """SPEC.json written once; an existing one must hold the same spec apart from written_utc (the worker binds its
    result to the spec's SHA-256, so a finished job keeps its spec file)."""

    path = Path(directory) / "SPEC.json"

    def strip(value: Mapping[str, Any]) -> dict[str, Any]:
        return {key: item for key, item in _clean(value).items() if key != "written_utc"}

    if path.exists():
        if strip(_read(path)) != strip(spec):
            raise ValueError(f"{path} holds another spec")
        return path
    Path(directory).mkdir(parents=True, exist_ok=True)
    write_once(path, _clean(spec))
    return path


def next_launches(pending: Sequence[tuple[str, str]], running_families: Sequence[str], max_workers: int = MAX_WORKERS) -> list[str]:
    """Keys to start now, in pending order: at most ``max_workers`` workers in all and one LightGBM job at a time."""

    slots = int(max_workers) - len(running_families)
    lightgbm_busy = "lightgbm" in running_families
    chosen: list[str] = []
    for key, family in pending:
        if slots <= 0:
            break
        if family == "lightgbm":
            if lightgbm_busy:
                continue
            lightgbm_busy = True
        chosen.append(key)
        slots -= 1
    return chosen


def job_state(directory: Path) -> str:
    """'completed' (RESULT.json completed for this SPEC.json), 'terminal', 'failed', or 'pending' (no RESULT.json); a
    completed result of another spec raises."""

    directory = Path(directory)
    result_path = directory / "RESULT.json"
    if not result_path.is_file():
        return "pending"
    result = _read(result_path)
    if result.get("status") == "completed":
        if result.get("spec_sha256") != sha256_file(directory / "SPEC.json"):
            raise ValueError(f"{result_path} completed under another SPEC.json")
        return "completed"
    return "terminal" if result.get("terminal") else "failed"


class _Adopted:
    """A worker known only by its PID (left running by an earlier driver)."""

    def __init__(self, pid: int) -> None:
        self.pid = int(pid)

    def poll(self) -> int | None:
        try:
            import psutil

            if psutil.pid_exists(self.pid) and psutil.Process(self.pid).status() != psutil.STATUS_ZOMBIE:
                return None
        except Exception:  # noqa: BLE001  (a vanished process counts as ended)
            pass
        return -1


def _alive_with(pid: int, marker: str) -> bool:
    try:
        import psutil

        process = psutil.Process(int(pid))
        return process.is_running() and any(marker in str(part) for part in process.cmdline())
    except Exception:  # noqa: BLE001
        return False


def launch_worker(spec_path: Path, directory: Path, launch: int, env: Mapping[str, str]) -> subprocess.Popen:
    """``python -B q_worker.py --spec SPEC.json`` (the supervisor's command), stdout and stderr to the job directory."""

    with (directory / f"worker_{launch}.stdout.log").open("ab") as stdout, (directory / f"worker_{launch}.stderr.log").open("ab") as stderr:
        return subprocess.Popen([sys.executable, "-B", str(WORKER), "--spec", str(spec_path)], cwd=str(ROOT), env=dict(env),
                                stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _ledger(jobs_dir: Path, event: str, **fields: Any) -> None:
    with (jobs_dir / LEDGER_NAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_clean({"utc": utc(), "event": event, **fields})) + "\n")


def _interrupts(directory: Path) -> int:
    path = Path(directory) / "INTERRUPTS.json"
    return int(_read(path).get("count", 0)) if path.is_file() else 0


def _count_interrupt(directory: Path) -> None:
    _atomic_json(Path(directory) / "INTERRUPTS.json", {"count": _interrupts(directory) + 1, "last_utc": utc()})


def _save_state(jobs_dir: Path, running: Mapping[str, Mapping[str, Any]]) -> None:
    _atomic_json(jobs_dir / STATE_NAME, {"driver_pid": os.getpid(), "heartbeat_utc": utc(),
                                         "running": {key: {name: info[name] for name in ("pid", "launch", "started_utc", "spec")}
                                                     for key, info in running.items()}})


def _claim_driver(jobs_dir: Path) -> None:
    path = jobs_dir / DRIVER_NAME
    current = _read(path) if path.is_file() else {}
    pid = current.get("pid")
    if pid and int(pid) != os.getpid() and _alive_with(int(pid), "e4_q"):
        raise RuntimeError(f"another train driver (pid {pid}) is running")
    _atomic_json(path, {"pid": os.getpid(), "started_utc": utc(), "argv": sys.argv})


def _adopt(jobs_dir: Path, spec_paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Workers of an earlier driver that are still alive (PID and SPEC path in the command line) are adopted; the
    others count as interrupted when they left no RESULT.json."""

    state = _read(jobs_dir / STATE_NAME) if (jobs_dir / STATE_NAME).is_file() else {}
    running: dict[str, dict[str, Any]] = {}
    for key, info in (state.get("running") or {}).items():
        if key not in spec_paths:
            continue
        if _alive_with(int(info["pid"]), str(spec_paths[key])):
            running[key] = {**info, "handle": _Adopted(int(info["pid"])), "tick": time.perf_counter()}
            _ledger(jobs_dir, "adopted", key=key, pid=info["pid"])
        else:
            _ledger(jobs_dir, "not_running_at_driver_start", key=key, pid=info["pid"])
            if job_state(jobs_dir / key) == "pending":
                _count_interrupt(jobs_dir / key)
    return running


def job_record(directory: Path, job: Any, digest: str, role_rows: Mapping[str, int]) -> tuple[dict[str, Any], list[str]]:
    """A completed job's files, scores and resources, and every disagreement with its spec and the plan."""

    directory = Path(directory)
    result = _read(directory / "RESULT.json")
    model = next((directory / name for name in ("model_refit.pkl", "model_refit.json", "model_refit.pt")
                  if (directory / name).is_file()), None)
    names = ["SPEC.json", "RESULT.json", "pred_q_dev.npz", "pred_q_cal.npz", *([model.name] if model else []),
             *(["FIT_DONE.json"] if (directory / "FIT_DONE.json").is_file() else []),
             *sorted(path.name for path in directory.glob("RESULT_attempt*.json"))]
    files = {name: sha256_file(directory / name) for name in names}
    scores = result.get("scores") or {}
    q_dev, q_cal = scores.get("q_dev") or {}, scores.get("q_cal") or {}
    record = {"key": directory.name, "job_id": job.job_id, "family": job.family, "card_id": job.card_id, "identity": job.identity,
              "state": "completed", "files": files, "model_file": model.name if model else None,
              "q_dev_raw_match_brier": q_dev.get("raw_match_brier"), "q_cal_raw_match_brier": q_cal.get("raw_match_brier"),
              "q_dev_rows": q_dev.get("rows"), "q_cal_rows": q_cal.get("rows"), "q_dev_keys_sha256": result.get("q_dev_keys_sha256"),
              "best_iteration": result.get("best_iteration"), "seconds": result.get("seconds"),
              "peak_rss_bytes": result.get("peak_rss_bytes"), "threads": result.get("threads"),
              "finished_utc": result.get("finished_utc"), "interrupts": _interrupts(directory),
              "launches": len(list(directory.glob("worker_*.stdout.log"))),
              "attempts_kept": sorted(path.name for path in directory.glob("RESULT_attempt*.json"))}
    checks = {"spec_sha256": result.get("spec_sha256") == files["SPEC.json"], "identity": result.get("identity") == job.identity,
              "job_id": result.get("job_id") == job.job_id, "release_digest": result.get("release_digest") == digest,
              "variant": result.get("variant") == VARIANT, "model_file": model is not None,
              "model_sha256": model is not None and result.get("model_sha256") == files.get(model.name),
              "pred_q_dev_sha256": q_dev.get("pred_sha256") == files["pred_q_dev.npz"],
              "pred_q_cal_sha256": q_cal.get("pred_sha256") == files["pred_q_cal.npz"],
              "q_dev_rows": q_dev.get("rows") == role_rows["Q_DEV"], "q_cal_rows": q_cal.get("rows") == role_rows["Q_CAL"],
              "q_dev_brier_finite": vs._finite(q_dev.get("raw_match_brier"))}
    record["checks"] = checks
    return record, [f"{directory.name}: {name}" for name, ok in checks.items() if not ok]


def _ledger_wall(jobs_dir: Path) -> dict[str, Any]:
    path = jobs_dir / LEDGER_NAME
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
    starts = [e["utc"] for e in events if e.get("event") == "started"]
    ends = [e["utc"] for e in events if e.get("event") == "ended"]
    if not starts or not ends:
        return {"first_start_utc": None, "last_end_utc": None, "seconds": None, "launches": len(starts)}
    first, last = min(starts), max(ends)
    return {"first_start_utc": first, "last_end_utc": last, "launches": len(starts),
            "seconds": (datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds(),
            "driver_starts": sum(e.get("event") == "driver_started" for e in events)}


def train_stage(paths: Paths = Paths(), *, workers: int = MAX_WORKERS, poll_s: float = 5.0,
                role_rows: Mapping[str, int] = vs.Q_ROLE_ROWS) -> int:
    """PRESPEC_E4 section 6 step 4 (third part): the 16 q jobs through the frozen worker; resumable, all must complete."""

    started, tick = utc(), time.perf_counter()
    out = paths.out
    target = out / TRAIN_STAGE
    if target.exists():
        print(json.dumps({"done": str(target)}), flush=True)
        return 0
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f"workers must be 1-{MAX_WORKERS}")
    _passed(out / PREPARE_STAGE, "prepare")
    prepared = _read(paths.inputs / "PREPARED.json")
    release = qrelease.verify(paths.release)
    if release["digest"] != RELEASE_DIGEST:
        return _blocked(out, "train", [f"release digest {release['digest']} is not {RELEASE_DIGEST}"])
    jobs = plan_jobs(qjobs.plan())
    jobs_dir = paths.jobs
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _claim_driver(jobs_dir)
    keys = [qtest.key_of(job.job_id) for job in jobs]
    family = {qtest.key_of(job.job_id): job.family for job in jobs}
    spec_paths = {key: ensure_spec(jobs_dir / key, job_spec(job, out, prepared, paths.release, release["digest"], utc()))
                  for key, job in zip(keys, jobs)}
    env = worker_env()
    running = _adopt(jobs_dir, spec_paths)
    blocked: dict[str, str] = {}
    _ledger(jobs_dir, "driver_started", pid=os.getpid(), workers=int(workers), adopted=sorted(running))
    print(json.dumps({"driver": os.getpid(), "jobs": len(keys), "adopted": sorted(running)}), flush=True)
    while True:
        for key in list(running):
            info = running[key]
            code = info["handle"].poll()
            if code is None:
                continue
            running.pop(key)
            state = job_state(jobs_dir / key)
            if state == "pending":                     # ended without a result: an interruption
                _count_interrupt(jobs_dir / key)
            seconds = time.perf_counter() - info["tick"]
            _ledger(jobs_dir, "ended", key=key, pid=info["pid"], launch=info["launch"], returncode=code, state=state,
                    seconds=seconds)
            print(json.dumps({"end": key, "state": state, "returncode": code, "seconds": round(seconds, 1)}), flush=True)
        pending: list[tuple[str, str]] = []
        for key in keys:
            if key in running or key in blocked:
                continue
            directory = jobs_dir / key
            state = job_state(directory)
            if state == "completed":
                continue
            if state == "terminal":
                blocked[key] = "terminal failure (RESULT.json)"
                continue
            if state == "failed":
                attempt = len(list(directory.glob("RESULT_attempt*.json"))) + 1
                if attempt > RETRIES:
                    blocked[key] = f"failed in {attempt} attempts"
                    continue
                os.replace(directory / "RESULT.json", directory / f"RESULT_attempt{attempt}.json")
                _ledger(jobs_dir, "kept_failed_attempt", key=key, attempt=attempt)
            if _interrupts(directory) > MAX_INTERRUPTS:
                blocked[key] = f"interrupted more than {MAX_INTERRUPTS} times"
                continue
            pending.append((key, family[key]))
        if not pending and not running:
            break
        for key in next_launches(pending, [family[k] for k in running], workers):
            directory = jobs_dir / key
            launch = len(list(directory.glob("worker_*.stdout.log"))) + 1
            process = launch_worker(spec_paths[key], directory, launch, env)
            running[key] = {"handle": process, "pid": process.pid, "launch": launch, "started_utc": utc(),
                            "tick": time.perf_counter(), "spec": str(spec_paths[key])}
            _ledger(jobs_dir, "started", key=key, pid=process.pid, launch=launch, family=family[key])
            print(json.dumps({"start": key, "pid": process.pid, "launch": launch}), flush=True)
        _save_state(jobs_dir, running)
        time.sleep(poll_s)
    _save_state(jobs_dir, running)

    # every job must have completed under its spec, the release and the main q run's Q_DEV keys ------------------
    records, problems = [], []
    for job, key in zip(jobs, keys):
        state = job_state(jobs_dir / key)
        if state != "completed":
            records.append({"key": key, "job_id": job.job_id, "family": job.family, "card_id": job.card_id, "state": state,
                            "reason": blocked.get(key)})
            problems.append(f"{key}: {blocked.get(key, state)}")
            continue
        record, found = job_record(jobs_dir / key, job, release["digest"], role_rows)
        records.append(record)
        problems += found
    keys_sha = sorted({r["q_dev_keys_sha256"] for r in records if r.get("state") == "completed"})
    main_keys = sorted({(_read(paths.q_run / "jobs" / key / "RESULT.json") or {}).get("q_dev_keys_sha256") for key in keys
                        if (paths.q_run / "jobs" / key / "RESULT.json").is_file()})
    if len(keys_sha) > 1:
        problems.append(f"the jobs scored different Q_DEV keys: {keys_sha}")
    if keys_sha and keys_sha != main_keys:
        problems.append(f"Q_DEV keys {keys_sha} differ from the main q run's {main_keys} (row-table key mismatch)")
    if problems:
        return _blocked(out, "train", problems, jobs=records)
    files = {f"jobs/{r['key']}/{name}": {"path": str(jobs_dir / r["key"] / name), "sha256": digest}
             for r in records for name, digest in r["files"].items()}
    record = {"version": VERSION, "experiment": "E4", "stage": "train", "passed": True, "files": files, "jobs": records,
              "cards": dict(CARDS), "q_dev_keys_sha256": keys_sha[0], "main_q_dev_keys_sha256": main_keys[0],
              "release_digest": release["digest"], "spec_shape": "reports/q_production_20261005/q_supervisor.py:447-449",
              "worker": _relative(WORKER),
              "scheduling": {"max_workers": int(workers), "lightgbm_at_once": 1, "env_threads": THREADS,
                             "lightgbm_num_threads": "3 (qtrain.fit_lightgbm, frozen)", "gpu": "none: device cpu, CUDA_VISIBLE_DEVICES=''",
                             "retries": RETRIES, "max_interrupts": MAX_INTERRUPTS,
                             "resume": "rerun this stage: completed jobs are skipped, live workers adopted, interrupted ones relaunched"},
              "wall": _ledger_wall(jobs_dir), "ledger": str(jobs_dir / LEDGER_NAME), **_common(paths),
              "timing": vs._stage("train", started, tick), "utc": utc()}
    write_once(target, _clean(record))
    print(json.dumps({"done": str(target), "q_dev_raw_match_brier": {r["card_id"]: r["q_dev_raw_match_brier"] for r in records}}),
          flush=True)
    return 0


# ----------------------------------------------------------------------------- selection and the Q_DEV rehearsal
def select_cards(results: Mapping[str, Mapping[str, Any]], orders: Mapping[str, Sequence[str]]) -> tuple[dict[str, Any], list[str]]:
    """qselect.select_within per family on Q_DEV raw match-weighted Brier (ties by card order); every card must have
    completed (qselect also picks among completed cards only, so an incomplete family is a problem: PRESPEC_E4 §3)."""

    records, problems = {}, []
    for family in FAMILIES:
        record = qselect.select_within(results[family], orders[family])
        records[family] = record
        if not record["complete"] or record["selected"] is None:
            states = {card: entry["status"] for card, entry in record["cards"].items()}
            problems.append(f"{family}: not every card completed ({states})")
    return records, problems


def select_stage(paths: Paths = Paths()) -> int:
    """PRESPEC_E4 section 6 step 4 (last part): selection within each family on Q_DEV.  E4_SELECTION.json."""

    started, tick = utc(), time.perf_counter()
    out = paths.out
    target = out / SELECTION_NAME
    if target.exists():
        print(json.dumps({"done": str(target)}), flush=True)
        return 0
    train = _passed(out / TRAIN_STAGE, "train")
    jobs = plan_jobs(qjobs.plan())
    results: dict[str, dict[str, Any]] = {family: {} for family in FAMILIES}
    orders: dict[str, list[str]] = {family: [] for family in FAMILIES}
    key_of: dict[str, dict[str, str]] = {family: {} for family in FAMILIES}
    files = {}
    for job in jobs:
        key = qtest.key_of(job.job_id)
        path = paths.jobs / key / "RESULT.json"
        results[job.family][job.card_id] = _read(path)
        orders[job.family].append(job.card_id)
        key_of[job.family][job.card_id] = key
        files[f"jobs/{key}/RESULT.json"] = _file_entry(path)
    records, problems = select_cards(results, orders)
    if problems:
        return _blocked(out, "select", problems, selection=records)
    selection = {family: {**records[family], "selected_key": key_of[family][records[family]["selected"]],
                          "card_order": orders[family], "jobs": len(orders[family])} for family in FAMILIES}
    record = {"version": VERSION, "experiment": "E4", "stage": "select", "passed": True, "files": files, "selection": selection,
              "rule": "qselect.select_within per family: lowest Q_DEV raw match-weighted Brier (RESULT.json "
                      "scores.q_dev.raw_match_brier) among completed cards, ties by card order (the frozen plan's order); "
                      "every card completed (PRESPEC_E4 section 3)",
              "q_dev_keys_sha256": train["q_dev_keys_sha256"], **_common(paths), "timing": vs._stage("select", started, tick),
              "utc": utc()}
    write_once(target, _clean(record))
    print(json.dumps({"done": str(target), "selected": {family: [selection[family]["selected"], selection[family]["brier"]]
                                                         for family in FAMILIES}}), flush=True)
    return 0


def _renamed(pair: Mapping[str, Any]) -> dict[str, Any]:
    """e2_run.evaluate_pair names its first cell P1; here it is the selected LightGBM cell."""

    out = dict(pair)
    for field in ("brier", "auc", "metrics"):
        out[field] = {("lightgbm" if name == "P1" else name): value for name, value in pair[field].items()}
    out["contrast"] = "lightgbm_minus_PT_flex"
    return out


def decide(interval: Sequence[float]) -> str:
    """PRESPEC_E4 section 5 on the 95 % TEST interval of Brier(LightGBM) - Brier(PT_flex)."""

    low, high = (float(value) for value in interval)
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"not an interval: {low}, {high}")
    if high < 0.0:
        return "same_direction"
    if low > 0.0:
        return "opposite_direction"
    return "undetermined"


def evaluate(y4: Any, exact4: Any, y_main: Any, svi_main: Any, p_lgbm: Any, p_pt: Any, match: Any) -> dict[str, Any]:
    """The TEST evaluation of PRESPEC_E4 sections 3-5 on one set of rows (Q_DEV in the rehearsal):

    comparison  e2_run.evaluate_pair(y4, p_lgbm, p_pt, match): match-weighted raw Brier of both cells, the difference
                LightGBM - PT_flex with its 95 % percentile interval (qeval.paired_match_bootstrap: 2,000 draws, PCG64 seed
                20261011, matches in ascending id, draws shared by both cells, numpy.quantile linear), AUC of both cells
                and their difference on the same draws;
    decision    decide() on that interval (section 5);
    agreement   e4_run.agreement(y_main, y4, svi_main, match, exact_zero_new=exact4): share equal, kappa and exact-zero
                share, match weighted (primary) and row weighted, overall and by |main SVI|; every overall figure of
                share_equal, kappa and exact_zero_share_new must be present (Task 11 review m5);
    supplementary  the same comparison on the rows with exact_zero4 == 0 (section 4; included by Ruling 46), with the
                section 4 condition (TEST exact-zero share > 1 %) recorded in both weightings; not used for the decision."""

    y4 = er._binary(y4, "y4")
    n = y4.size
    exact = er._binary(exact4, "exact_zero", n)
    if np.any((exact == 1) & (y4 == 1)):
        raise ValueError("an exact-zero row must be negative")
    match = np.asarray(match)
    p_lgbm, p_pt = np.asarray(p_lgbm, dtype=np.float64), np.asarray(p_pt, dtype=np.float64)
    if match.shape != (n,) or p_lgbm.shape != (n,) or p_pt.shape != (n,):
        raise ValueError("match ids and probabilities must align with the labels")
    comparison = _renamed(e2_run.evaluate_pair(y4, p_lgbm, p_pt, match))
    difference = comparison["brier_difference"]
    outcome = decide(difference["interval"])
    decision = {"point": difference["point"], "interval": difference["interval"], "level": difference["level"], "outcome": outcome,
                "text": DECISIONS[outcome], "rule": DECISION_RULE}
    agreement = er.agreement(y_main, y4, svi_main, match, exact_zero_new=exact)
    overall = agreement["overall"]
    missing = [f"{name}.{weighting}" for name in PRIMARY_AGREEMENT for weighting in er.WEIGHTINGS
               if (overall.get(name) or {}).get(weighting) is None]
    if missing:
        raise ValueError(f"agreement figures missing: {missing} (pass match ids and exact_zero_new; Task 11 review m5)")
    share = overall["exact_zero_share_new"]
    keep = exact == 0
    supplementary = {"rule": "the rows with exact_zero4 == 0 (an E4 tie, p_post == p_pre, removed), the same comparison; match "
                             "weights recomputed on the kept rows; not used for the decision (PRESPEC_E4 section 4)",
                     "included_by": "Ruling 46 (the E4 tie share exceeds 1 % in every non-TEST role)", "used_for_decision": False,
                     "rows": int(keep.sum()), "rows_excluded": int((~keep).sum()), "matches": int(np.unique(match[keep]).size),
                     "prespec_condition_share_over_0.01": {name: bool(share[name] > TIE_SHARE) for name in er.WEIGHTINGS},
                     "comparison": _renamed(e2_run.evaluate_pair(y4[keep], p_lgbm[keep], p_pt[keep], match[keep]))}
    return {"rows": int(n), "matches": int(np.unique(match).size), "comparison": comparison, "decision": decision,
            "agreement": agreement, "exact_zero_share": share, "supplementary_without_exact_zero": supplementary}


def qdev_stage(paths: Paths = Paths()) -> int:
    """The TEST evaluation path rehearsed on Q_DEV with the selected cells' stored predictions (development evidence):
    its agreement must equal Task 11's E4_QDEV_AGREEMENT.json and each cell's Brier its worker's.  E4_QDEV.json."""

    started, tick = utc(), time.perf_counter()
    out = paths.out
    target = out / QDEV_NAME
    if target.exists():
        print(json.dumps({"done": str(target)}), flush=True)
        return 0
    selection = _passed(out / SELECTION_NAME, "select")["selection"]
    _passed(out / ROWS_STAGE, "rows")
    public = _load_fields(paths.rows, ("engagement_key", "match_index", "role", "y", "exact_zero"))
    source = _load_fields(paths.source_rows, ("engagement_key", "role", "y", "label_svi_delta"))
    role = public["role"].astype(np.int64)
    if not (np.array_equal(public["engagement_key"], source["engagement_key"]) and np.array_equal(role, source["role"].astype(np.int64))):
        raise RuntimeError("the E4 row table and its source differ in keys or roles")
    q_dev = np.flatnonzero(role == Q_ROLES["Q_DEV"])
    preds, files = {}, {}
    for family in FAMILIES:
        key = selection[family]["selected_key"]
        directory = paths.jobs / key
        result = _read(directory / "RESULT.json")
        pred_path = directory / "pred_q_dev.npz"
        if sha256_file(pred_path) != result["scores"]["q_dev"]["pred_sha256"]:
            raise RuntimeError(f"{pred_path} differs from its RESULT.json")
        pred = _load_npz(pred_path)
        if not np.array_equal(np.asarray(pred["row"], dtype=np.int64), q_dev):
            raise RuntimeError(f"{pred_path}: the stored predictions are not the Q_DEV rows in order")
        preds[family] = {"p": np.asarray(pred["p"], dtype=np.float64), "brier": result["scores"]["q_dev"]["raw_match_brier"],
                         "key": key, "card": selection[family]["selected"]}
        files[f"jobs/{key}/pred_q_dev.npz"] = _file_entry(pred_path)
        files[f"jobs/{key}/RESULT.json"] = _file_entry(directory / "RESULT.json")
    result = evaluate(public["y"][q_dev], public["exact_zero"][q_dev], source["y"][q_dev], source["label_svi_delta"][q_dev],
                      preds["lightgbm"]["p"], preds["PT_flex"]["p"], public["match_index"][q_dev])
    task11 = _read(out / er.AGREEMENT_NAME)["q_dev_agreement"]
    reproduced = _clean(result["agreement"]) == task11
    brier_diff = {family: abs(result["comparison"]["brier"][family] - preds[family]["brier"]) for family in FAMILIES}
    problems = []
    if not reproduced:
        problems.append("the TEST-step agreement path does not reproduce Task 11's Q_DEV agreement")
    problems += [f"{family}: Q_DEV Brier differs from its worker's by {diff}" for family, diff in brier_diff.items() if diff > RECOMPUTE_ATOL]
    if problems:
        return _blocked(out, "qdev", problems)
    files[er.AGREEMENT_NAME] = _file_entry(out / er.AGREEMENT_NAME)
    record = {"version": VERSION, "experiment": "E4", "stage": "qdev", "passed": True, "split": "Q_DEV", "policy": QDEV_POLICY,
              "files": files, "cells": {family: {k: preds[family][k] for k in ("key", "card", "brier")} for family in FAMILIES},
              "rehearsal": result,
              "checks": {"agreement_equals_task11": reproduced, "brier_abs_diff_to_worker": brier_diff,
                         "main_labels": "q_run_1/Q_ROWS_h90.npz y and label_svi_delta at Q_DEV rows (equal to LABELS_h90: Task 11)"},
              **_common(paths), "timing": vs._stage("qdev", started, tick), "utc": utc()}
    write_once(target, _clean(record))
    print(json.dumps({"done": str(target), "q_dev_difference": result["decision"]["point"],
                      "q_dev_interval": result["decision"]["interval"], "agreement_equals_task11": reproduced}), flush=True)
    return 0


# ----------------------------------------------------------------------------- lock (PRESPEC_E4 section 6)
def rules() -> dict[str, Any]:
    """The rules of the TEST step; the lock binds them and the step refuses others."""

    return {
        "contrast": CONTRAST_TEXT,
        "cells": "the qselect.select_within winners on Q_DEV raw match-weighted Brier: LightGBM (4 cards on "
                 f"{LGBM_CONDITION}) and PT_flex (12 cards); E4_SELECTION.json",
        "probabilities": "raw p = qtest.sigmoid(qtest.LoadedCell(cell).logits(TEST rows)); no calibration",
        "scoring_checks": {"reload_rows": qtest.RELOAD_ROWS, "reload_atol": dict(qtest.RELOAD_ATOL),
                           "sealed_rows": "the row table's y is -1 on every TEST row", "order": "before any sealed file is read"},
        "weights": "every match weighs 1 and its rows 1/m (qeval match weights)",
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "level": LEVEL, "quantiles": list(e2_run.qeval.bootstrap_quantiles(LEVEL)),
                      "unit": "match, ascending match id; draws shared by both cells and by the AUC difference",
                      "generator": "qeval.bootstrap_indices: numpy.random.Generator(numpy.random.PCG64(seed)).integers(0, M, size=M)",
                      "interval": "percentile; numpy.quantile method='linear'", "implementation": "e2_run.evaluate_pair"},
        "auc": "secondary: match-weighted AUC of both cells and the difference LightGBM - PT_flex on the same draws",
        "decision": {"split": "TEST only", "rule": DECISION_RULE, "outcomes": DECISIONS},
        "main_comparison": {"source": "q_run_1/TEST_REPORT.json bootstrap.contrasts.P1", "prespec_printed": MAIN_P1,
                            "note": "reported side by side; its interval is at level 0.975, E4's at 0.95"},
        "agreement": {"function": "e4_run.agreement(y_main, y4, svi_main, match_index, exact_zero_new=exact_zero4)",
                      "main_labels": "q_run_1/Q_TEST_SEALED_h90.npz y and label_svi_delta (re-opened)",
                      "e4_labels": "tog_e4_1/Q_TEST_SEALED_h90.npz y and exact_zero", "bins": list(er.BINS),
                      "primary": "match_weighted", "required": [f"{name}.{w}" for name in PRIMARY_AGREEMENT for w in er.WEIGHTINGS]},
        "supplementary": {"rule": "TEST rows with exact_zero4 == 0, the same comparison; not used for the decision",
                          "prespec_condition": "TEST exact-zero share > 0.01 (PRESPEC_E4 section 4)", "included_by": "Ruling 46"},
        "checks_after_opening": ["the sealed E4 labels equal labels_of_test_rows on the locked PRED_final.npz",
                                 "the main sealed labels follow y = 1[SVI > 0]"],
        "order": ["lock re-check (every bound file, rules, environment, release, the reviewed lock SHA-256)",
                  "qtest.score checks and TEST scoring into TEST_SCORES/", "E4_OPENING.json, then OPENINGS.json",
                  "both sealed files read once", "evaluation", "E4_REPORT.json"],
    }


def task10_files(out: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for pattern in TASK10_PATTERNS:
        for path in sorted(Path(out).glob(pattern)):
            if path.is_file():
                found[f"task10:{path.relative_to(out).as_posix()}"] = path
    return found


def loaded_modules() -> list[str]:
    """Workspace modules loaded in this process that CODE_FILES does not list (bound as module:<path>)."""

    return sorted(name for name in qrelease.workspace_files(list(sys.modules.values()))
                  if name.startswith(("studies/", "reports/")) and name not in CODE_FILES)


def bound_files(paths: Paths, train: Mapping[str, Any]) -> dict[str, Path]:
    """Every file the lock binds by SHA-256 (EXPOSURE_KO.md and OPENINGS.json are appended to, so recorded, not bound)."""

    out, q = paths.out, paths.q_run
    files: dict[str, Path] = {"prespec": paths.prespec, "release": paths.release}
    files.update({f"code:{name}": ROOT / name for name in CODE_FILES})
    files.update({f"module:{name}": ROOT / name for name in loaded_modules()})
    vscore = _read(out / "E4_VSCORE.json")
    files.update({f"measurer:{ctx}": Path(vscore["measurers"][ctx]["checkpoint"]) for ctx in CONTEXTS})
    files.update({"q_run_1:Q_ROWS_h90.npz": paths.source_rows, "q_run_1:Q_TEST_SEALED_h90.npz": paths.source_sealed,
                  "q_run_1:raw/store/BUILD.json": paths.store_target / "BUILD.json",
                  "q_run_1:inputs/main_h90/XY.npy": paths.base_inputs / "XY.npy",
                  "q_run_1:inputs/main_h90/PREPARED.json": paths.base_inputs / "PREPARED.json",
                  "q_run_1:TEST_REPORT.json": q / "TEST_REPORT.json", "q_run_1:SCORE_LOCK.json": q / "SCORE_LOCK.json"})
    for scope in SCOPES:
        for name in ("CONTEXT.json", *CONTEXT_FILES):
            files[f"q_run_1:inputs/main_h90/{scope}/{name}"] = paths.base_inputs / scope / name
    files.update({"labels_1:LABEL_LOCK.json": paths.labels / "LABEL_LOCK.json", "labels_1:ENGAGEMENTS.npz": paths.labels / "ENGAGEMENTS.npz",
                  "labels_1:Q_DEV_H90_KEYS.npy": paths.labels / "Q_DEV_H90_KEYS.npy",
                  "projection:CONTEXT_MEMBERSHIP.npy": paths.projection / "CONTEXT_MEMBERSHIP.npy"})
    files.update(task10_files(out))
    files.update({"task11:E4_LABELS_nonTEST.npz": out / er.LABELS_NAME, "task11:E4_QDEV_AGREEMENT.json": out / er.AGREEMENT_NAME,
                  "rows:Q_ROWS_h90.npz": paths.rows, "rows:Q_ROWS_h90.json": paths.rows_record,
                  "rows:Q_TEST_SEALED_h90.npz": paths.sealed, "rows:E4_ROWS.json": out / ROWS_STAGE})
    for path in sorted(paths.inputs.rglob("*")):
        if path.is_file():
            files[f"prepare:{path.relative_to(paths.inputs).as_posix()}"] = path
    files["prepare:E4_PREPARE.json"] = out / PREPARE_STAGE
    files["store:raw/store/BUILD.json"] = paths.store_link / "BUILD.json"
    for record in train["jobs"]:
        for name in record["files"]:
            files[f"job:{record['key']}:{name}"] = paths.jobs / record["key"] / name
    files.update({"train:E4_TRAIN.json": out / TRAIN_STAGE, "select:E4_SELECTION.json": out / SELECTION_NAME,
                  "qdev:E4_QDEV.json": out / QDEV_NAME})
    return files


def lock_cross_checks(paths: Paths, verify: Mapping[str, Mapping[str, str]], records: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    """The bound hashes against the stage records, the Task 10/11 records, q_run_1's records and Ruling 46."""

    def sha(label: str) -> str | None:
        return (verify.get(label) or {}).get("sha256")

    rows, prepare, train, qdev = records["rows"], records["prepare"], records["train"], records["qdev"]
    vscore, agreement = _read(paths.out / "E4_VSCORE.json"), _read(paths.out / er.AGREEMENT_NAME)
    base = _read(paths.base_inputs / "PREPARED.json")
    score_lock_files = (_read(paths.q_run / "SCORE_LOCK.json") or {}).get("files") or {}
    return {
        "row_table_files_equal_the_rows_record": all(sha(f"rows:{Path(entry['path']).name}") == entry["sha256"]
                                                     for entry in rows["files"].values()),
        "source_rows_equal_q_run_1_prepared_and_the_rows_record": sha("q_run_1:Q_ROWS_h90.npz") == base.get("rows_sha256")
                                                                  == rows["source"]["sha256"],
        "q_run_1_sealed_equals_its_score_lock": sha("q_run_1:Q_TEST_SEALED_h90.npz") == score_lock_files.get("Q_TEST_SEALED_h90.npz"),
        "store_build_equals_q_run_1_prepared": (sha("q_run_1:raw/store/BUILD.json") == base.get("store_build_sha256")
                                                == sha("store:raw/store/BUILD.json")),
        "xy_equals_q_run_1_prepared": sha("q_run_1:inputs/main_h90/XY.npy") == base.get("xy_sha256") == sha("prepare:XY.npy"),
        "contexts_equal_q_run_1": all(
            prepare["comparisons"][scope]["context_npz_equal"] and prepare["comparisons"][scope]["bundle_manifest_equal"]
            and sha(f"prepare:{scope}/{name}") == sha(f"q_run_1:inputs/main_h90/{scope}/{name}")
            for scope in SCOPES for name in ("CONTEXT.npz", "bundle/MANIFEST.json")),
        "ruling_46_files": (sha("task11:E4_LABELS_nonTEST.npz") == RULING_46["labels"]
                            and sha("task11:E4_QDEV_AGREEMENT.json") == RULING_46["agreement"]),
        "task10_tables_equal_e4_vscore": all(sha(f"task10:PRED_{ctx}.npz") == vscore["contexts"][ctx]["pred_sha256"]
                                             and sha(f"task10:PRED_{ctx}.json") == vscore["contexts"][ctx]["record_sha256"]
                                             for ctx in CONTEXTS),
        "measurers_equal_e4_vscore": all(sha(f"measurer:{ctx}") == vscore["measurers"][ctx]["model_sha256"] for ctx in CONTEXTS),
        "e4_vscore_unchanged_since_task10": (sha("code:studies/tog_experiments_20261012/e4_vscore.py")
                                             == (vscore.get("code_sha256") or {}).get("studies/tog_experiments_20261012/e4_vscore.py")),
        "e4_run_unchanged_since_task11": (sha("code:studies/tog_experiments_20261012/e4_run.py")
                                          == (agreement.get("code_sha256") or {}).get("studies/tog_experiments_20261012/e4_run.py")),
        "job_files_equal_the_train_record": all(sha(f"job:{r['key']}:{name}") == digest for r in train["jobs"]
                                                for name, digest in r["files"].items()),
        "qdev_rehearsal_reproduced_task11": qdev["checks"]["agreement_equals_task11"] is True,
    }


def lock_stage(paths: Paths = Paths(), *, require_committed_code: bool = True) -> int:
    """E4_LOCK.json (PRESPEC_E4 section 6, Ruling 28).  The sealed files and the record JSON are hashed as bytes only."""

    started, tick = utc(), time.perf_counter()
    out = paths.out
    target = out / LOCK_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; it is written once")
    records = {"rows": _passed(out / ROWS_STAGE, "rows"), "prepare": _passed(out / PREPARE_STAGE, "prepare"),
               "train": _passed(out / TRAIN_STAGE, "train"), "select": _passed(out / SELECTION_NAME, "select"),
               "qdev": _passed(out / QDEV_NAME, "qdev")}
    prespec = document_record(paths.prespec)
    if not (prespec["sha256"] and prespec["commit"] == PRESPEC_COMMIT and prespec["unchanged_since_commit"]):
        return _blocked(out, "lock", {"prespec_hash_or_commit": prespec}, stage=vs._stage("lock", started, tick))
    code = {name: document_record(ROOT / name) for name in CODE_FILES}
    if require_committed_code and not all(code[name]["commit"] and code[name]["unchanged_since_commit"] for name in TOG_CODE
                                          if name != "studies/tog_experiments_20261012/__init__.py"):
        raise RuntimeError("commit the E4 code before writing the lock")
    release = qrelease.verify(paths.release)
    if release["digest"] != RELEASE_DIGEST:
        return _blocked(out, "lock", {"release_digest": release["digest"]})
    tick_hash = time.perf_counter()
    files = bound_files(paths, records["train"])
    verify = {label: {"path": str(path), "sha256": sha256_file(path)} for label, path in files.items()}
    cross = lock_cross_checks(paths, verify, records)
    if not all(cross.values()):
        raise RuntimeError(f"lock cross-checks failed: {sorted(name for name, ok in cross.items() if not ok)}")
    hashing = {"wall_seconds": time.perf_counter() - tick_hash, "files": len(verify)}
    jobs = {job.card_id: job for job in plan_jobs(qjobs.plan())}
    selection = records["select"]["selection"]
    cells = {}
    for family in FAMILIES:
        job = jobs[selection[family]["selected"]]
        cells[family] = qtest.cell_record(out, job.to_dict())
        cell = cells[family]
        if cell["status"] != "completed" or cell["key"] != selection[family]["selected_key"]:
            raise RuntimeError(f"{family}: the selected cell is not a completed job")
        for name, field in (("SPEC.json", "spec_sha256"), ("RESULT.json", "result_sha256"), (cell["model_file"], "model_sha256"),
                            ("pred_q_dev.npz", "pred_q_dev_sha256")):
            if verify[f"job:{cell['key']}:{name}"]["sha256"] != cell[field]:
                raise RuntimeError(f"{family}: {name} differs between qtest.cell_record and the bound files")
    rows_record = _read(paths.rows_record)              # sealed summary: only its TRAIN/Q_DEV/Q_CAL counts are read
    non_test_counts = {name: rows_record["counts"][name] for name in ("TRAIN", "Q_DEV", "Q_CAL")}
    vscore = _read(out / "E4_VSCORE.json")
    stage_times = {name: record.get("timing") for name, record in records.items()}
    stage_times["train_wall"] = records["train"].get("wall")
    payload = {
        "version": VERSION, "experiment": "E4", "locked_utc": utc(),
        "prespec": {**prespec, "approved_commit": PRESPEC_COMMIT}, "exposure_ledger": document_record(paths.exposure),
        "code": code, "git_head": git_head(),
        "release": {"path": str(paths.release), "digest": release["digest"]},
        "measurers": {ctx: {k: vscore["measurers"][ctx][k] for k in ("card_id", "checkpoint", "model_sha256", "v_dev_raw_logloss")}
                      for ctx in CONTEXTS},
        "selection": {family: {"selected": selection[family]["selected"], "selected_key": selection[family]["selected_key"],
                               "brier": selection[family]["brier"],
                               "cards": {card: entry["brier"] for card, entry in selection[family]["cards"].items()}}
                      for family in FAMILIES},
        "cells": cells,
        "row_counts": {"rows_by_role": records["rows"]["rows_by_role"], "matches_by_role": records["rows"]["matches_by_role"],
                       "label_counts_non_test": non_test_counts,
                       "note": "TEST: rows and matches only; Q_ROWS_h90.json read for its TRAIN/Q_DEV/Q_CAL counts only (Ruling 28)"},
        "rules": rules(), "environment": environment(), "stages": stage_times,
        "label_policy": ("Q_TEST_SEALED_h90.npz (E4) and q_run_1/Q_TEST_SEALED_h90.npz are hashed as file bytes only and not "
                         "opened before the TEST step; Q_ROWS_h90.json is a sealed summary (Ruling 28); PRED_final.npz holds "
                         "TEST probabilities outside any seal and nothing summarized them (Ruling 43)"),
        "openings_before_lock": (_read(paths.openings) or {}).get("openings", []) if paths.openings.is_file() else [],
        "cross_checks": cross, "verify": verify, "hashing": hashing,
        "run": run_record(rows=records["rows"]["rows_by_role"], matches=records["rows"]["matches_by_role"], inputs=[],
                          code=[ROOT / name for name in CODE_FILES], seed=SEED, started_utc=started),
    }
    payload["stages"]["lock"] = vs._stage("lock", started, tick)       # every stage's wall time and peak memory (PRESPEC §10)
    write_once(target, _clean(payload))
    print(json.dumps({"lock": str(target), "sha256": sha256_file(target), "files": len(verify), "prespec_commit": prespec["commit"],
                      "selected": {family: selection[family]["selected"] for family in FAMILIES}}), flush=True)
    return 0


# ----------------------------------------------------------------------------- part B: the single TEST step
class ScoringRefused(RuntimeError):
    """A qtest.score check failed before any sealed file was read."""


def score_cell(cell: Mapping[str, Any], *, device: str = "cpu") -> dict[str, Any]:
    """The frozen qtest TEST scoring path for one cell (qtest.score:307-318): qtest.LoadedCell rebuilds the model as its
    worker did, re-scores the first RELOAD_ROWS stored Q_DEV rows (must agree within RELOAD_ATOL), checks that the row
    table seals every TEST row (y == -1), then scores the TEST rows; p = qtest.sigmoid(logit), no calibration."""

    loaded = qtest.LoadedCell(cell, device=device)
    stored = qtest.load_npz(Path(cell["source_dir"]) / "pred_q_dev.npz")
    check_rows = np.asarray(stored["row"], dtype=np.int64)[:qtest.RELOAD_ROWS]
    again = loaded.logits(check_rows)
    diff = float(np.abs(again - np.asarray(stored["logit"])[:qtest.RELOAD_ROWS]).max()) if len(check_rows) else 0.0
    if diff > qtest.RELOAD_ATOL[loaded.kind]:
        raise ScoringRefused(f"{cell['key']}: re-loaded model differs from its stored Q_DEV predictions by {diff}")
    rows = np.flatnonzero(np.asarray(loaded.rows["role"]) == qtest.TEST_ROLE)
    if (np.asarray(loaded.rows["y"])[rows] != -1).any():
        raise ScoringRefused(f"{cell['key']}: TEST labels are not sealed in the row table")
    logits = np.asarray(loaded.logits(rows), dtype=np.float64)
    return {"row": rows, "engagement_key": np.asarray(loaded.rows["engagement_key"])[rows], "logit": logits,
            "p": qtest.sigmoid(logits), "reload_max_abs_logit_diff": diff, "reload_rows": int(len(check_rows)), "kind": loaded.kind}


def required_labels(lock: Mapping[str, Any]) -> list[str]:
    labels = [*CORE_LABELS, *(f"code:{name}" for name in CODE_FILES)]
    for family in FAMILIES:
        cell = (lock.get("cells") or {}).get(family) or {}
        labels += [f"job:{cell.get('key')}:{name}" for name in ("SPEC.json", "RESULT.json", cell.get("model_file"), "pred_q_dev.npz")]
    return labels


def verify_lock(lock: Mapping[str, Any], paths: Paths = Paths()) -> dict[str, Any]:
    """Why the TEST step may not run (empty when the lock holds): no PRESPEC hash or another commit (section 7), a missing
    entry, any bound file changed, other rules, another environment, cells that differ from the bound files, or a q
    release whose code changed."""

    problems: dict[str, Any] = {}
    prespec, verify = lock.get("prespec") or {}, lock.get("verify") or {}
    if not (prespec.get("sha256") and prespec.get("commit")):
        problems["prespec_hash_missing"] = True
    else:
        if prespec["commit"] != PRESPEC_COMMIT:
            problems["prespec_not_the_approved_commit"] = prespec["commit"]
        if (verify.get("prespec") or {}).get("sha256") != prespec["sha256"]:
            problems["prespec_entry_differs"] = True
    missing = [label for label in required_labels(lock) if label not in verify]
    if missing:
        problems["entries_missing"] = missing
    changed = {}
    for label, entry in verify.items():
        path = Path(entry["path"])
        now = sha256_file(path) if path.is_file() else None
        if now != entry["sha256"]:
            changed[label] = {"path": entry["path"], "locked": entry["sha256"], "now": now}
    if changed:
        problems["changed_since_lock"] = changed
    if lock.get("rules") != _clean(rules()):
        problems["rules_differ"] = True
    now_env, locked_env = environment(), lock.get("environment") or {}
    differs = {key: {"locked": locked_env.get(key), "now": now_env[key]} for key in ENV_KEYS if locked_env.get(key) != now_env[key]}
    if differs:
        problems["environment_differs"] = differs
    for family in FAMILIES:
        cell = (lock.get("cells") or {}).get(family) or {}
        for name, field in (("SPEC.json", "spec_sha256"), ("RESULT.json", "result_sha256"), (cell.get("model_file"), "model_sha256"),
                            ("pred_q_dev.npz", "pred_q_dev_sha256")):
            if not cell or (verify.get(f"job:{cell.get('key')}:{name}") or {}).get("sha256") != cell.get(field):
                problems.setdefault("cells_differ_from_bound_files", []).append(f"{family}:{name}")
    try:
        release = qrelease.verify(Path((verify.get("release") or {}).get("path") or paths.release))
        if release["digest"] != RELEASE_DIGEST:
            problems["release_digest"] = release["digest"]
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        problems["release"] = str(error)
    return problems


def _openings(path: Path) -> list[dict[str, Any]]:
    return list((_read(path) or {}).get("openings", [])) if Path(path).exists() else []


def append_opening(path: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Append one opening to OPENINGS.json (created when missing; EXPOSURE_KO.md rule 1); returns the earlier ones."""

    path = Path(path)
    payload = _read(path) if path.exists() else {
        "rule": "EXPOSURE_KO.md rule 1: every ToG experiment that opens sealed labels appends experiment, UTC time and lock "
                "SHA-256", "openings": []}
    before = list(payload.get("openings", []))
    if any(opening.get("experiment") == entry["experiment"] for opening in before):
        raise RuntimeError(f"{entry['experiment']} has already opened its sealed labels")
    payload["openings"] = before + [dict(entry)]
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    pending.write_text(json.dumps(_clean(payload), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(pending, path)
    return before


def _part(tick: float) -> dict[str, Any]:
    return {"wall_seconds": time.perf_counter() - tick, "peak_memory_bytes_main_process": peak_memory_bytes()}


def _failure(out: Path, entry: Mapping[str, Any], error: BaseException, parts: Mapping[str, Any]) -> None:
    """E4_FAILED_test_<n>.json: a failure after the opening leaves a record (no data value; message cut to 500)."""

    path = _next_free(Path(out), "E4_FAILED_test")
    lines = traceback.format_exception(type(error), error, error.__traceback__)
    write_once(path, _clean({"version": VERSION, "experiment": "E4", "failed": True, "utc": utc(), "opening": dict(entry),
                             "error": {"type": type(error).__name__, "message": str(error)[:500],
                                       "traceback_tail": "".join(lines[-6:])[-3000:]},
                             "stages": dict(parts),
                             "rule": "do not re-run under this lock; the opening stays recorded (Ruling 31 practice)"}))
    print(json.dumps({"FAILED": "test", "file": str(path), "error": type(error).__name__}), flush=True)


def _test_scores(out: Path, cell: Mapping[str, Any], scorer: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
    """TEST scores of one cell in TEST_SCORES/<key>.npz (row, engagement_key, logit, p) with <key>.json, written once; a
    rerun before the opening reuses verified scores (and identical bytes)."""

    directory = Path(out) / SCORES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    npz, meta_path = directory / f"{cell['key']}.npz", directory / f"{cell['key']}.json"
    if meta_path.is_file():
        meta = _read(meta_path)
        if sha256_file(npz) != meta.get("pred_sha256"):
            raise RuntimeError(f"{npz} differs from {meta_path}")
        return {**_load_npz(npz), "meta": meta}
    tick = time.perf_counter()
    scored = scorer(cell)
    arrays = {"row": np.asarray(scored["row"], dtype=np.int64), "engagement_key": np.asarray(scored["engagement_key"]),
              "logit": np.asarray(scored["logit"], dtype=np.float64), "p": np.asarray(scored["p"], dtype=np.float64)}
    vs._check_probabilities(arrays["p"], f"{cell['key']} TEST p")
    digest = write_bytes_once(npz, npz_bytes(arrays))
    meta = {"key": cell["key"], "rows": int(arrays["row"].size), "pred_sha256": digest, "kind": scored.get("kind"),
            "reload_max_abs_logit_diff": scored.get("reload_max_abs_logit_diff"), "reload_rows": scored.get("reload_rows"),
            "loader": "qtest.LoadedCell (frozen)", "calibration": "none: raw probabilities (PRESPEC_E4 section 4)",
            "seconds": time.perf_counter() - tick, "utc": utc()}
    write_once(meta_path, _clean(meta))
    return {**arrays, "meta": _read(meta_path)}


def run_test_step(paths: Paths = Paths(), *, expected_lock_sha256: str,
                  scorer: Callable[[Mapping[str, Any]], Mapping[str, Any]] = score_cell) -> int:
    """Part B (PRESPEC_E4 section 6 step 6): verify the lock, score TEST through the frozen qtest path, record the
    opening, read both sealed files once, evaluate, and write E4_REPORT.json once."""

    started, tick = utc(), time.perf_counter()
    out = paths.out
    for name in (REPORT_NAME, OPENING_NAME):
        if (out / name).exists():
            raise FileExistsError(f"{out / name} exists; the sealed TEST labels are opened once under this lock")
    lock_path = out / LOCK_NAME
    if not lock_path.is_file():
        raise RuntimeError("no E4_LOCK.json: run the lock stage first")
    lock = _read(lock_path)
    problems = verify_lock(lock, paths)
    lock_sha = sha256_file(lock_path)
    if lock_sha != expected_lock_sha256:
        problems["lock_is_not_the_reviewed_lock"] = {"expected": expected_lock_sha256, "now": lock_sha}
    before = _openings(paths.openings)
    if any(opening.get("experiment") == "E4" for opening in before):
        problems["e4_already_opened"] = True
    if problems:
        return _blocked(out, "test", problems, lock=str(lock_path))
    parts: dict[str, Any] = {"verify_lock": _part(tick)}
    verify = lock["verify"]

    def path_of(label: str) -> Path:
        return Path(verify[label]["path"])

    # the frozen qtest scoring path, before any sealed file is read ----------------------------------------------
    tick_part = time.perf_counter()
    public = _load_fields(path_of("rows:Q_ROWS_h90.npz"), (*vs.ALIGN_COLUMNS, "engagement_key", "y"))
    test_rows = np.flatnonzero(public["role"].astype(np.int64) == TEST_ROLE)
    keys, match = public["engagement_key"][test_rows], public["match_index"][test_rows].astype(np.int64)
    if np.any(public["y"][test_rows] != -1):
        return _blocked(out, "test", {"row_table_not_sealed": "a TEST row of the row table holds a label"})
    scores: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        cell = lock["cells"][family]
        try:
            scores[family] = _test_scores(out, cell, scorer)
        except Exception as error:  # noqa: BLE001  (no sealed file was read: refused, diagnosable, rerunnable)
            return _blocked(out, "test", {"scoring_refused": {family: f"{type(error).__name__}: {error}"}})
        if not (np.array_equal(np.asarray(scores[family]["row"], dtype=np.int64), test_rows)
                and np.array_equal(scores[family]["engagement_key"], keys)):
            return _blocked(out, "test", {"scores_are_not_the_test_rows": family})
    parts["score"] = _part(tick_part)

    # the opening: the record first, then OPENINGS.json, then the sealed files (EXPOSURE_KO.md rule 1) -------------
    entry = {"experiment": "E4", "utc": utc(), "lock": str(lock_path), "lock_sha256": lock_sha,
             "prespec_sha256": lock["prespec"]["sha256"],
             "labels": [str(path_of("rows:Q_TEST_SEALED_h90.npz")), str(path_of("q_run_1:Q_TEST_SEALED_h90.npz"))],
             "sealed_sha256": {"e4": verify["rows:Q_TEST_SEALED_h90.npz"]["sha256"],
                               "main": verify["q_run_1:Q_TEST_SEALED_h90.npz"]["sha256"]},
             "note": OPENING_NOTE}
    write_once(out / OPENING_NAME, _clean({**entry, "openings_before": before}))
    append_opening(paths.openings, entry)
    try:
        return _evaluate_test(paths, lock, lock_path, entry, before, public, test_rows, keys, match, scores, parts, started, tick)
    except BaseException as error:
        _failure(out, entry, error, parts)
        raise


def _evaluate_test(paths: Paths, lock: Mapping[str, Any], lock_path: Path, entry: Mapping[str, Any], before: list,
                   public: Mapping[str, np.ndarray], test_rows: np.ndarray, keys: np.ndarray, match: np.ndarray,
                   scores: Mapping[str, Mapping[str, Any]], parts: dict[str, Any], started: str, tick: float) -> int:
    verify = lock["verify"]

    def path_of(label: str) -> Path:
        return Path(verify[label]["path"])

    tick_part = time.perf_counter()
    e4 = _load_fields(path_of("rows:Q_TEST_SEALED_h90.npz"), ("engagement_key", *SEALED_COLUMNS))      # first opening
    main = _load_fields(path_of("q_run_1:Q_TEST_SEALED_h90.npz"), ("engagement_key", "y", "label_svi_delta"))  # repeated look
    pos_e4, pos_main = align_keys(keys, e4["engagement_key"]), align_keys(keys, main["engagement_key"])
    labels = {name: np.asarray(e4[name])[pos_e4] for name in SEALED_COLUMNS}
    y_main = np.asarray(main["y"])[pos_main]
    svi_main = np.asarray(main["label_svi_delta"], dtype=np.float64)[pos_main]
    with np.load(path_of("labels_1:ENGAGEMENTS.npz"), allow_pickle=False) as stored:
        engagements = {name: stored[name] for name in (*vs.ALIGN_COLUMNS, "query_h90", "valid_h90")}
    query_post = engagements["query_h90"][vs.align_q_rows(engagements, public)].astype(np.int64)
    del engagements
    again = labels_of_test_rows(public, query_post, _load_fields(path_of("task10:PRED_final.npz"), ("query_index", "raw_p")))
    sealed_equal = bool(np.array_equal(again["engagement_key"], keys)
                        and all(np.array_equal(again[field], labels[column]) and again[field].dtype == labels[column].dtype
                                for column, field in (("y", "y4"), ("label_svi_delta", "label_svi_delta4"),
                                                      ("exact_zero", "exact_zero"), ("near_zero_bits", "near_zero_bits"))))
    if not sealed_equal:
        raise RuntimeError("the sealed E4 TEST labels differ from the labels of the locked PRED_final.npz")
    direction_rule = bool(np.isin(y_main, (0, 1)).all() and np.isfinite(svi_main).all()
                          and np.array_equal(np.asarray(y_main, dtype=np.int64), (svi_main > 0).astype(np.int64)))
    if not direction_rule:
        raise RuntimeError("the main sealed TEST labels do not follow y = 1[SVI > 0]")
    parts["labels"] = _part(tick_part)
    tick_part = time.perf_counter()
    result = evaluate(labels["y"], labels["exact_zero"], y_main, svi_main, scores["lightgbm"]["p"], scores["PT_flex"]["p"], match)
    parts["evaluate"] = _part(tick_part)
    p1 = ((_read(path_of("q_run_1:TEST_REPORT.json")).get("bootstrap") or {}).get("contrasts") or {}).get("P1") or {}
    main_comparison = {"test_report": {"point": p1.get("point"), "interval": p1.get("interval"), "level": p1.get("level")},
                       "prespec_printed": MAIN_P1,
                       "note": "the main labels' P1 (LightGBM - PT_flex) on TEST, interval at level 0.975; E4's interval is at "
                               "0.95 (PRESPEC_E4 section 4)"}
    rows = int(test_rows.size)
    matches = int(np.unique(match).size)
    inputs = [path_of(label) for label in ("rows:Q_ROWS_h90.npz", "rows:Q_TEST_SEALED_h90.npz", "q_run_1:Q_TEST_SEALED_h90.npz",
                                           "task10:PRED_final.npz", "labels_1:ENGAGEMENTS.npz", "q_run_1:TEST_REPORT.json")]
    inputs += [paths.out / SCORES_DIR / f"{lock['cells'][family]['key']}.npz" for family in FAMILIES]
    report = {
        "version": VERSION, "experiment": "E4", "split": "TEST", "utc": utc(),
        "prespec": {"path": verify["prespec"]["path"], "sha256": lock["prespec"]["sha256"], "commit": lock["prespec"]["commit"]},
        "lock": {"path": str(lock_path), "sha256": entry["lock_sha256"], "locked_utc": lock.get("locked_utc")},
        "opening": dict(entry), "openings_before": before, "labels_opened": entry["labels"], "rules": lock["rules"],
        "selection": lock.get("selection"), "cells": {family: lock["cells"][family]["key"] for family in FAMILIES},
        "rows": rows, "matches": matches, "scores": {family: scores[family]["meta"] for family in FAMILIES},
        "comparison": result["comparison"], "decision": result["decision"], "main_comparison": main_comparison,
        "agreement": result["agreement"], "exact_zero_share": result["exact_zero_share"],
        "supplementary_without_exact_zero": result["supplementary_without_exact_zero"],
        "checks": {"sealed_e4_labels_equal_pred_final": sealed_equal, "main_labels_follow_the_direction_rule": direction_rule,
                   "scores_are_the_test_rows": all(np.array_equal(np.asarray(scores[family]["row"], dtype=np.int64), test_rows)
                                                   and np.array_equal(scores[family]["engagement_key"], keys) for family in FAMILIES),
                   "agreement_primary_figures_present": all(result["agreement"]["overall"][name][weighting] is not None
                                                            for name in PRIMARY_AGREEMENT for weighting in er.WEIGHTINGS)},
        "row_counts": lock.get("row_counts"),
        "stages": {**(lock.get("stages") or {}),
                   "test": {"started_utc": started, "finished_utc": utc(), "wall_seconds": time.perf_counter() - tick,
                            "peak_memory_bytes_main_process": peak_memory_bytes(), "parts": parts,
                            "processes": "main process only, no worker, no GPU"}},
        "environment": environment(),
        "run": run_record(rows=rows, matches=matches, inputs=inputs, code=[ROOT / name for name in CODE_FILES], seed=SEED,
                          started_utc=started),
    }
    write_once(paths.out / REPORT_NAME, _clean(report))
    difference = result["decision"]
    print(json.dumps({"done": str(paths.out / REPORT_NAME), "difference": difference["point"], "interval": difference["interval"],
                      "outcome": difference["outcome"]}), flush=True)
    return 0


def align_keys(keys: Any, table_keys: Any) -> np.ndarray:
    """Positions of ``keys`` in a sealed table's engagement keys (each key once there); a missing key raises."""

    table_keys, keys = np.asarray(table_keys), np.asarray(keys)
    if np.unique(table_keys).size != table_keys.size:
        raise ValueError("a sealed file repeats an engagement key")
    index = {key: position for position, key in enumerate(table_keys.tolist())}
    absent = sum(key not in index for key in keys.tolist())
    if absent:
        raise ValueError(f"{absent} rows have no sealed label")
    return np.asarray([index[key] for key in keys.tolist()], dtype=np.int64)


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("rows", "prepare", "train", "select", "qdev", "lock", "test"))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="train: worker processes at once (1-4)")
    parser.add_argument("--poll", type=float, default=5.0, help="train: seconds between polls")
    parser.add_argument("--confirm-single-opening", action="store_true",
                        help="test: required; the step opens the sealed TEST labels once")
    parser.add_argument("--lock-sha256", help="test: required; the SHA-256 of the reviewed E4_LOCK.json")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if args.stage == "rows":
        return rows_stage()
    if args.stage == "prepare":
        return prepare_stage()
    if args.stage == "train":
        return train_stage(workers=args.workers, poll_s=args.poll)
    if args.stage == "select":
        return select_stage()
    if args.stage == "qdev":
        return qdev_stage()
    if args.stage == "lock":
        return lock_stage()
    if not (args.confirm_single_opening and args.lock_sha256):
        parser.error("the test stage opens the sealed TEST labels once; pass --confirm-single-opening and --lock-sha256")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # part B scores LightGBM and PT_flex cells on the CPU
    return run_test_step(expected_lock_sha256=args.lock_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
