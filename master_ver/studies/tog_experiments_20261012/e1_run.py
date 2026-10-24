"""T1.5 E1: training of the placebo and size-matched engagement models, Q_DEV selection, the Q_CAL development figures,
the rehearsal of the TEST evaluation path and the lock (part A, SDD Ruling 56), and the single TEST step (part B)
(PLAN_E1_E5_KO.md Task 16).

Binding specification: reports/tog_experiments_20261012/PRESPEC_E1_KO.md as amended 2026-10-13 and re-approved (commit
7a841db): section 3 모델 and 균형, section 4 (quantities and intervals), section 5 (decision rule), section 6 steps 4-7
with 잠금 파일, 잠금 전에 보지 않는 것 and 행 표 기록 JSON, section 7 (BLOCKED) and section 10 (outputs).  SDD Rulings 16
(decide() on G_eng, L1, L2 and the net gain), 27/28 (TEST-step order, lock list), 31 (no rerun after an opening), 50 (the
re-approval sentence), 52 (hashing sealed bytes is not an opening), 55 (rehearsal; no kill or credit stratum on the
placebo table), 56 (two parts), 57 (Task 13-15 code byte-identical) and 58 (sealed summaries in the lock).

Sign: G = Brier(PT_flex) - Brier(LightGBM); positive means LightGBM (the detailed state) is better than the game-state
model, the opposite of E2-E4.

Runs (written by Task 15 under D:/LOL_Training_20261004/tog_e1_1): plc, model (가) on the placebo windows, and eng, model
(나) on the matched engagements (raw/store a junction to q_run_1's).  Placebo row i and engagement row i are the same
matched engagement (plc/PLC_KEY_MAP_h90.npz).  Stages, each written once:
  train   The 16 cards of PRESPEC_E1 section 3 모델 (verbatim) per run as the frozen q_worker.py subprocesses: specs in the
          q_supervisor.py:447-449 shape (e4_q.job_spec), at most four workers and one LightGBM at a time across both runs,
          CPU only; resumable (a completed job is never relaunched, live workers are adopted after a driver restart, a
          failed attempt is kept before its one retry).  Every card must complete and score its run's own Q_DEV keys,
          else BLOCKED.  E1_TRAIN.json.
  select  qselect.select_within per run and family on Q_DEV raw match-weighted Brier (e4_q.select_cards); every card
          completed, else BLOCKED.  E1_SELECTION.json.
  qcal    Section 6 step 5 through the function the TEST step calls after its opening (evaluate_opened), run end to end
          on Q_CAL (Ruling 55): the selected cells re-loaded by qtest.LoadedCell and scored on Q_CAL (equal to their
          stored Q_CAL predictions); non-TEST stand-ins in the sealed files' layouts for the five sealed files (placebo
          labels, placebo end V, both sealed summaries, q_run_1's labels) and for q_run_1's two main-model TEST_SCORES
          cells.  Gains, the joint bootstrap, decide(), every sensitivity (stratified and de-duplicated r regardless of
          their triggers) and every auxiliary.  E1_QCAL.json: development evidence that decides nothing (section 6).
  lock    E1_LOCK.json (section 6 잠금 파일, Ruling 28 and 58), written once from committed code.
  test    Part B, only after the review of part A: lock re-check -> both row tables seal TEST -> e4_q.score_cell (the
          frozen qtest.score path: row-table hash, Q_DEV reload, TEST y = -1) for the four cells -> E1_OPENING.json and
          OPENINGS.json -> the five sealed files read once -> evaluate_opened -> E1_REPORT.json.  A failure after the
          opening writes E1_FAILED_test_<n>.json and is not rerun under this lock (Ruling 31).

A BLOCKED condition (PRESPEC_E1 section 7, or an input that differs from its record) writes E1_BLOCKED_<stage>_<n>.json and
ends the stage with status 2.  Before the TEST step nothing about TEST is printed, logged, recorded or reported: public
records cover TRAIN, Q_DEV and Q_CAL only, sealed files are hashed as bytes (Ruling 52), and the frozen-format files with
TEST-inclusive counts are bound as sealed summaries by path pattern, never opened (Ruling 58).  qtest.evaluate,
qeval.kill_count_strata and qeval.credit_strata are never called (the placebo eval_* columns are 0 by design, Ruling 55).
No player identifier is read or written; no GPU.

Usage: python -B -m studies.tog_experiments_20261012.e1_run {train,select,qcal,lock} [--workers 4] [--out DIR]
       python -B -m studies.tog_experiments_20261012.e1_run test --confirm-single-opening --lock-sha256 <reviewed lock>
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
from dataclasses import dataclass  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005.assemble import NEAR_ZERO, PROJECTION  # noqa: E402
from studies.q_production_20261005 import jobs as qjobs, qeval, qtest, release as qrelease  # noqa: E402
from studies.q_production_20261005.rows import Q_ROLES, SEALED_COLUMNS  # noqa: E402
from studies.tog_experiments_20261012 import (e1_candidates as ec, e1_rows as er1, e1_states as es, e2_run,  # noqa: E402
                                              e4_q as eq, e4_vscore as vs)
from studies.tog_experiments_20261012.common import (  # noqa: E402
    LABELS, Q_RUN, TOG_E1, document_record, git_head, peak_memory_bytes, run_record, sha256_file, utc, write_once)

VERSION = "tog_e1_run_v1_20261013"
Blocked = es.Blocked                       # a PRESPEC_E1 section 7 stop or an input that differs from its record
VARIANT = eq.VARIANT                       # main_h90
TRAIN, Q_DEV, Q_CAL, TEST = (Q_ROLES[name] for name in ("TRAIN", "Q_DEV", "Q_CAL", "TEST"))
ROLE_CODES = dict(Q_ROLES)
PRESPEC, AMENDMENT = es.PRESPEC, es.AMENDMENT
EXPOSURE, OPENINGS = eq.EXPOSURE, eq.OPENINGS
RELEASE, RELEASE_DIGEST = eq.RELEASE, eq.RELEASE_DIGEST
PRESPEC_COMMIT = "7a841dbf4793d0c9702dfb4347ea156462e126fc"     # the author's re-approval of the amended PRESPEC_E1 (Ruling 50)
AMENDMENT_COMMIT = "bd300bf0a22091c413edd32f6274634a7143719b"   # the 2026-10-13 amendment; AMENDMENT_E1 unchanged since
PRESPEC_AT_BD300BF = "fe1baedfc8635f222a65bd8def112a9019734806afe7dcfb5b8b1a88d1a69e02"   # cited by the Task 13 r1 records
ARMS = ("eng", "plc")                      # (나) size-matched engagements, (가) placebo windows: the order of the four arrays
ENG, PLC = ARMS
FAMILIES = eq.FAMILIES                     # lightgbm, PT_flex
CARDS = {"lightgbm": ("lgbm_l15_m100", "lgbm_l15_m200", "lgbm_l31_m100", "lgbm_l31_m200"),     # PRESPEC_E1 section 3 모델
         "PT_flex": ("ptflex_k4_C0.001", "ptflex_k4_C0.01", "ptflex_k4_C0.1", "ptflex_k4_C1", "ptflex_k6_C0.001",
                     "ptflex_k6_C0.01", "ptflex_k6_C0.1", "ptflex_k6_C1", "ptflex_k8_C0.001", "ptflex_k8_C0.01",
                     "ptflex_k8_C0.1", "ptflex_k8_C1")}
MAX_WORKERS = eq.MAX_WORKERS
SEED, RESAMPLES, LEVEL = qeval.BOOTSTRAP_SEED, qeval.BOOTSTRAP_RESAMPLES, qeval.EXPLORATORY_LEVEL   # 20261011, 2000, 0.95
L1_WEIGHT, L2_WEIGHT = 0.25, 0.75
QUANTITIES = ("G_eng", "G_plc", "L1", "L2", "net", "r")
R001 = NEAR_ZERO[1]                        # 0.01, the middle near-zero threshold (assemble.py:40)
MAIN_TEST_GAIN = 0.01386                   # PRESPEC_E1 line 6: -TEST_REPORT.json bootstrap.contrasts.P1.point, printed
QUINTILES = (0.2, 0.4, 0.6, 0.8)
RECOMPUTE_ATOL = 1e-10
SHARD_GROUP = {TRAIN: "train", Q_DEV: "validation", Q_CAL: "validation", TEST: "test"}     # STATES_h90 parts by bank shard
MAIN_CELLS = {"lightgbm": e2_run.CELLS["P1"], "PT_flex": e2_run.CELLS["PT_flex"]}   # q_run_1 final cell, the search job it reuses
SEALED_LABELS = ("plc_labels", "pred_end", "windows_summary", "balance_summary", "eng_labels")
SEALED_LOCK_LABEL = {"plc_labels": "sealed:plc/Q_TEST_SEALED_h90.npz", "pred_end": "sealed:PRED_PLACEBO_h90_TEST_SEALED.npz",
                     "windows_summary": "sealed:WINDOWS_h90_TEST_SEALED.json", "balance_summary": "sealed:BALANCE_TEST_SEALED.json",
                     "eng_labels": "sealed:q_run_1/Q_TEST_SEALED_h90.npz"}
OUTCOMES = ("no_gain", "inconclusive", "keep", "rewrite", "net_gain", "indistinguishable")
DECISIONS = {  # PRESPEC_E1 section 5, verbatim
    "no_gain": "G_eng_size 구간 상한 < 0 → \"no_gain\": 크기를 맞춘 교전 모델에 판세 대비 이득이 없다. 보고하고 주장 1을 저자와 다시 본다.",
    "inconclusive": "G_eng_size 구간 하한 ≤ 0 ≤ 상한 → \"inconclusive\".",
    "keep": "L₁ 97.5 백분위 < 0 → \"keep\": 주장 1을 유지하고 위약 이득을 함께 보고한다.",
    "rewrite": "L₂ 2.5 백분위 > 0 → \"rewrite\": 주장 1을 킬 우위 라벨 중심으로 다시 쓰고 원고의 틀을 저자와 다시 정한다.",
    "net_gain": "그 밖: 순이득 구간 하한 > 0 → \"net_gain\": 순이득을 주 결과로 쓰고 킬 우위 라벨 결과를 주장 1의 둘째 근거로 올린다.",
    "indistinguishable": "아니면 → \"indistinguishable\": 측정기 표류와 구별되지 않는다. 순이득 구간 상한 < 0이면 위약 이득이 교전 이득보다 "
                         "크다는 것도 적는다.",
}
NET_NOTE = "순이득 구간 상한 < 0: 위약 이득이 교전 이득보다 크다 (PRESPEC_E1 section 5 rule 3c)"
DEFAULT_OUTCOMES = ("no_gain", "inconclusive", "indistinguishable")
DEFAULT_HANDLING = ("no_gain, inconclusive, indistinguishable의 기본 처리: 주장 1을 킬 우위 라벨 결과를 둘째 근거로 올려 쓰고, E1 결과는 "
                    "있는 그대로 표 2와 한계에 보고한다. 저자가 다르게 정하면 그 결정을 날짜와 함께 DECISION_LOG에 적은 뒤 따른다.")
DECISION_RULE = ("PRESPEC_E1 section 5 on the 95% TEST intervals of one joint match bootstrap: G_eng_size upper < 0 -> no_gain; "
                 "lower <= 0 <= upper -> inconclusive; lower > 0: L1 97.5th percentile < 0 -> keep, else L2 2.5th percentile > 0 "
                 "-> rewrite, else net gain lower > 0 -> net_gain, else indistinguishable (decided by L also when exactly 50 of "
                 "2,000 draws sit on a threshold, section 4)")
SIGN_TEXT = ("G = Brier(PT_flex) - Brier(LightGBM), match-weighted raw-probability Brier; positive means LightGBM is better "
             "(PRESPEC_E1 line 6; the opposite sign of E2-E4).  L1, L2 and the net gain share the sign")
QCAL_POLICY = ("PRESPEC_E1 section 6 step 5 and 개발 결과 정책: the Q_CAL figures are development evidence, reported first and used "
               "for no decision or definition; the decision uses the TEST intervals only")
OPENING_NOTE = ("E1 TEST step: first opening of the placebo TEST labels (plc/Q_TEST_SEALED_h90.npz), the placebo end V "
                "(PRED_PLACEBO_h90_TEST_SEALED.npz) and the two sealed TEST summaries, and a repeated look at the main TEST labels "
                "(q_run_1/Q_TEST_SEALED_h90.npz; first opened 2026-10-08, again by E2 and E4) for model (나) (PRESPEC_E1 sections 2 "
                "and 6)")
PRESPEC_NOTE = ("Ruling 50: the author's re-approval was recorded as one sentence appended to PRESPEC_E1's 2026-10-13 header line "
                "(commit 7a841db).  The PRESPEC SHA-256 bound here (the approved text) therefore differs from the bd300bf hash "
                f"{PRESPEC_AT_BD300BF} cited by the Task 13 r1 records (WINDOWS_h90.json, WINDOWS_h90_QDEV_GATE_r1.json); the "
                "text diff between the two is that sentence only (git diff bd300bf 7a841db: one line).  Every later E1 record "
                "(Tasks 14-16) cites the approved hash.  AMENDMENT_E1 is unchanged since bd300bf.")
SEALED_SUMMARY_RULE = ("Ruling 58 (re-review of Task 15's Important finding): every frozen-format file under tog_e1_1 that carries a "
                       "TEST-inclusive count, found by path pattern and bound by SHA-256 as a sealed summary; nobody reads their "
                       "counts and no code opens them (the frozen q_worker and q_prepare read the store records and PREPARED.json "
                       "as their format requires); the row-table record JSONs are sealed summaries by Ruling 28")
SEALED_SUMMARY_PATTERNS = (
    ("plc/raw/chunks/chunk_*/CHUNK.json", "per-chunk raw-store record of the placebo store (features.RawStoreWriter.close): "
                                          "'rows' is the chunk's row count; chunks of test_* state parts hold TEST rows only"),
    ("plc/raw/store/CHUNK.json", "merged raw-store record (features.merge_chunks): 'rows' counts every placebo row, TEST included"),
    ("plc/raw/store/BUILD.json", "the placebo store's build record: 'rows' counts every placebo row, TEST included"),
    ("plc/inputs/main_h90/PREPARED.json", "frozen q_prepare record: 'rows' is the placebo table's total, TEST included"),
    ("eng/inputs/main_h90/PREPARED.json", "frozen q_prepare record: 'rows' is the engagement table's total, TEST included"),
    ("plc/logs/q_prepare_*.stdout.log", "frozen q_prepare CLI output: prints 'rows', TEST included"),
    ("eng/logs/q_prepare_*.stdout.log", "frozen q_prepare CLI output: prints 'rows', TEST included"),
    ("plc/Q_ROWS_h90.json", "row-table record (rows.write_rows keys; Ruling 28 sealed summary; Task 15 wrote no TEST figure)"),
    ("eng/Q_ROWS_h90.json", "row-table record (rows.write_rows keys; Ruling 28 sealed summary; Task 15 wrote no TEST figure)"),
)
ROWS_NAME, ROWS_RECORD, PLC_SEALED, KEY_MAP = "Q_ROWS_h90.npz", "Q_ROWS_h90.json", "Q_TEST_SEALED_h90.npz", "PLC_KEY_MAP_h90.npz"
TRAIN_NAME, SELECTION_NAME, QCAL_NAME, LOCK_NAME = "E1_TRAIN.json", "E1_SELECTION.json", "E1_QCAL.json", "E1_LOCK.json"
OPENING_NAME, REPORT_NAME, DRIVER_NAME = "E1_OPENING.json", "E1_REPORT.json", "E1_TRAIN_DRIVER.json"
SCORES_DIR, REHEARSAL_DIR = "TEST_SCORES", "REHEARSAL_QCAL"
TASK13_FILES = ("WINDOWS_h90_QDEV_GATE_r0.json", "WINDOWS_h90_QDEV_GATE_r0.npz", "WINDOWS_h90_QDEV_GATE_r1.json",
                "WINDOWS_h90_QDEV_GATE_r1.npz", "GOLD_EDGES_h90.json", "GOLD_TRAIN_h90.npz", "EQUALITY_GATE_TOG.json")
TASK15_RECORDS = ("E1_ROWS.json", "E1_STORE_CHECK.json", "E1_STORE.json", "E1_ENG_INPUTS.json", "E1_PREPARE.json",
                  "E1_ROWS_RUN.json", "BALANCE.json")
PRESPEC_CITED = {  # hashes printed in PRESPEC_E1 (section 6 r0 record, Appendix A)
    "task13:WINDOWS_h90_QDEV_GATE_r0.json": "c67ce5564225a5c115f55700c1189ddf04063ef3cf0a143113cb5e74393723c8",
    "task13:WINDOWS_h90_QDEV_GATE_r0.npz": "b1fd21c0955672519f97c4ca4cb4ca6abd06c0f580ea63ecf5dd96c00d6f4930",
    "task13:GOLD_EDGES_h90.json": "42e0dd2d76eaf24319f175938e8f140509633a5508e5fe23d4849e2322e6a4eb",
    "task13:GOLD_TRAIN_h90.npz": "1e79fafb699afe981d8cdfd7594cd013df403850011fbf0c68072167d26251c6"}
E1_MODULES = ("studies/tog_experiments_20261012/e1_features.py", "studies/tog_experiments_20261012/e1_candidates.py",
              "studies/tog_experiments_20261012/e1_gate.py", "studies/tog_experiments_20261012/e1_states.py",
              "studies/tog_experiments_20261012/e1_rows.py")                   # Ruling 57: byte-identical since their runs
TOG_CODE = ("studies/tog_experiments_20261012/e1_run.py", *E1_MODULES, "studies/tog_experiments_20261012/e4_q.py",
            "studies/tog_experiments_20261012/e4_run.py", "studies/tog_experiments_20261012/e4_vscore.py",
            "studies/tog_experiments_20261012/e2_run.py", "studies/tog_experiments_20261012/e2_groups.py",
            "studies/tog_experiments_20261012/common.py", "studies/tog_experiments_20261012/__init__.py")
FROZEN_CODE = tuple(dict.fromkeys((*eq.FROZEN_CODE, "studies/label_production_20261005/controls_states.py",
                                   "studies/label_production_20261005/diagnostics.py",
                                   "studies/v_full_preprocess_20261003/artifacts.py",
                                   "studies/v_model_fit_input_20261003/preprocess.py")))
CODE_FILES = TOG_CODE + FROZEN_CODE
PUBLIC_ENG = ("engagement_key", "match_index", "role", "pt_p_pre", "y", "label_svi_delta", "exact_zero", "near_zero_bits")
PUBLIC_PLC = PUBLIC_ENG
PRED_FIELDS = ("window_id", "engagement_key", "match_index", "role", "s_ms", "end_ms", "p_start")
WINDOW_FIELDS = ("engagement_key", "match_index", "role", "s_ms", "D_ms", "start_age_plc_ms")
Q_FIELDS = ("engagement_key", "canonical_match_key", "component_key", "match_index", "role", "query_pre", "pre_ms")


@dataclass(frozen=True)
class Paths:
    """Where the stages read and write; the defaults are the real run (the tests point them at tmp_path).  ``source`` holds
    the Task 13-15 outputs and the run folders plc/ and eng/ (whose jobs/ the train stage fills); ``out`` takes this task's
    records (a scratch folder for a dry run)."""

    source: Path = TOG_E1
    out: Path = TOG_E1
    q_run: Path = Q_RUN
    labels: Path = LABELS
    projection: Path = PROJECTION
    prespec: Path = PRESPEC
    amendment: Path = AMENDMENT
    exposure: Path = EXPOSURE
    openings: Path = OPENINGS
    release: Path = RELEASE

    def run(self, arm: str) -> Path:
        return self.source / arm

    def rows(self, arm: str) -> Path:
        return self.run(arm) / ROWS_NAME

    def rows_record(self, arm: str) -> Path:
        return self.run(arm) / ROWS_RECORD

    def jobs(self, arm: str) -> Path:
        return self.run(arm) / "jobs"

    def inputs(self, arm: str) -> Path:
        return self.run(arm) / "inputs" / VARIANT

    @property
    def key_map(self) -> Path:
        return self.run(PLC) / KEY_MAP

    @property
    def plc_sealed(self) -> Path:
        return self.run(PLC) / PLC_SEALED

    @property
    def pred(self) -> Path:
        return self.source / es.PRED_NAME

    @property
    def pred_record(self) -> Path:
        return self.source / es.PRED_RECORD

    @property
    def pred_sealed(self) -> Path:
        return self.source / es.SEALED_NAME

    @property
    def windows(self) -> Path:
        return self.source / "WINDOWS_h90.npz"

    @property
    def windows_record(self) -> Path:
        return self.source / "WINDOWS_h90.json"

    @property
    def windows_sealed(self) -> Path:
        return self.source / ec.SEALED_NAME

    @property
    def balance(self) -> Path:
        return self.source / er1.BALANCE_NAME

    @property
    def balance_sealed(self) -> Path:
        return self.source / er1.BALANCE_SEALED_NAME

    @property
    def states_dir(self) -> Path:
        return self.source / es.STATES_DIR

    @property
    def states_record(self) -> Path:
        return self.states_dir / es.STATES_RECORD

    @property
    def q_rows(self) -> Path:
        return self.q_run / "Q_ROWS_h90.npz"

    @property
    def q_sealed(self) -> Path:
        return self.q_run / "Q_TEST_SEALED_h90.npz"

    @property
    def engagements(self) -> Path:
        return self.labels / "ENGAGEMENTS.npz"

    @property
    def objective_counts(self) -> Path:
        return self.projection / "OBJECTIVE_COUNTS.npy"

    @property
    def rehearsal(self) -> Path:
        return self.out / REHEARSAL_DIR

    def main_scores(self, family: str) -> Path:
        return self.q_run / qtest.SCORES_DIR / f"{MAIN_CELLS[family][0]}.npz"

    def main_meta(self, family: str) -> Path:
        return self.q_run / qtest.SCORES_DIR / f"{MAIN_CELLS[family][0]}.json"


# ----------------------------------------------------------------------------- small helpers
def _clean(value: Any) -> Any:
    return eq._clean(value)


def _read(path: Path) -> Any:
    return eq._read(path)


def _entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _blocked(out: Path, stage: str, reasons: Any, **extra: Any) -> int:
    """Write E1_BLOCKED_<stage>_<n>.json once (first free n) and return 2; reasons never carry a TEST figure."""

    path = eq._next_free(Path(out), f"E1_BLOCKED_{stage}")
    write_once(path, _clean({"version": VERSION, "experiment": "E1", "blocked_stage": stage, "blocked": True, "utc": utc(),
                             "reasons": reasons, **extra}))
    print(json.dumps({"BLOCKED": stage, "file": str(path), "reasons": _clean(reasons)}, ensure_ascii=False), flush=True)
    return 2


def _passed(path: Path, stage: str) -> dict[str, Any]:
    """An earlier stage's record: present, passed and every file it lists unchanged, else BLOCKED."""

    path = Path(path)
    if not path.is_file():
        raise Blocked(f"{path.name} is missing: run the {stage} stage first")
    record = _read(path)
    if record.get("passed") is not True:
        raise Blocked(f"{path.name} did not pass")
    try:
        eq._check_files(record.get("files") or {}, stage)
    except RuntimeError as error:
        raise Blocked(str(error)) from error
    return record


def _common(paths: Paths) -> dict[str, Any]:
    return {"code": document_record(Path(__file__)), "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
            "prespec": document_record(paths.prespec), "amendment": document_record(paths.amendment), "git_head": git_head(),
            "versions": vs._versions()}


def _json_bytes(payload: Any) -> bytes:
    """Deterministic JSON bytes (no timestamp inside), so a rerun reproduces a stand-in file byte for byte."""

    return json.dumps(_clean(payload), indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")


# ----------------------------------------------------------------------------- gains, joint bootstrap, decision (sections 4-5)
def _per_match(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite 1-D array of per-match mean losses")
    return array


def gains_and_ratio(eng_q: Any, eng_pt: Any, plc_q: Any, plc_pt: Any) -> dict[str, float | None]:
    """G_eng (= G_eng_size), G_plc, L1, L2, net and r from the four per-match mean-loss arrays in one match order (eng
    LightGBM, eng PT_flex, plc LightGBM, plc PT_flex): the mean of a per-match array is its match-weighted Brier.  r is
    None when G_eng is exactly 0 (L1 and L2 stay defined, section 4)."""

    arrays = [_per_match(value, name) for value, name in ((eng_q, "eng_q"), (eng_pt, "eng_pt"), (plc_q, "plc_q"), (plc_pt, "plc_pt"))]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("the four arrays must hold one value per match in one match order")
    mean_eng_q, mean_eng_pt, mean_plc_q, mean_plc_pt = (float(array.mean()) for array in arrays)
    g_eng, g_plc = mean_eng_pt - mean_eng_q, mean_plc_pt - mean_plc_q
    return {"G_eng": g_eng, "G_plc": g_plc, "L1": g_plc - L1_WEIGHT * g_eng, "L2": g_plc - L2_WEIGHT * g_eng, "net": g_eng - g_plc,
            "r": None if g_eng == 0.0 else g_plc / g_eng}


def derived_draws(g_eng: Any, g_plc: Any) -> dict[str, np.ndarray]:
    """Every quantity of section 4 from the same draws of G_eng and G_plc (a draw with G_eng = 0 gives a non-finite r)."""

    g_eng, g_plc = np.asarray(g_eng, dtype=np.float64), np.asarray(g_plc, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = g_plc / g_eng
    return {"G_eng": g_eng, "G_plc": g_plc, "L1": g_plc - L1_WEIGHT * g_eng, "L2": g_plc - L2_WEIGHT * g_eng,
            "net": g_eng - g_plc, "r": ratio}


def percentile_interval(draws: Any, level: float = LEVEL) -> list[float]:
    """The percentile interval of section 4: numpy.quantile (method linear) at qeval.bootstrap_quantiles(level)."""

    low, high = np.quantile(np.asarray(draws, dtype=np.float64), qeval.bootstrap_quantiles(level), method="linear")
    return [float(low), float(high)]


def joint_bootstrap(eng_q: Any, eng_pt: Any, plc_q: Any, plc_pt: Any, resamples: int = RESAMPLES, seed: int = SEED,
                    level: float = LEVEL, *, return_draws: bool = False) -> dict[str, Any]:
    """The joint match bootstrap of section 4: qeval.paired_match_bootstrap over the four cells (one stream of
    qeval.bootstrap_indices: PCG64 seed, match positions drawn with replacement, shared by every cell), G_eng and G_plc
    draws, and L1, L2, the net gain and r derived from those same draws.  Percentile intervals at ``level``.  The draws
    with G_eng <= 0 are counted and kept; r's non-finite draws are counted (none is dropped)."""

    point = gains_and_ratio(eng_q, eng_pt, plc_q, plc_pt)
    cells = {"eng_q": eng_q, "eng_pt": eng_pt, "plc_q": plc_q, "plc_pt": plc_pt}
    matches = int(np.asarray(eng_q).size)
    boot = qeval.paired_match_bootstrap(np.arange(matches), {name: np.asarray(value, dtype=np.float64) for name, value in cells.items()},
                                        {"G_eng": {"eng_pt": 1.0, "eng_q": -1.0}, "G_plc": {"plc_pt": 1.0, "plc_q": -1.0}},
                                        levels=level, resamples=resamples, seed=seed, return_draws=True)
    draws = derived_draws(boot["contrasts"]["G_eng"]["draws"], boot["contrasts"]["G_plc"]["draws"])
    out: dict[str, Any] = {"point": point, "matches": matches, "resamples": int(resamples), "seed": int(seed), "level": float(level),
                           "quantiles": list(qeval.bootstrap_quantiles(level)),
                           "unit": "match; the four arrays share every draw (qeval.paired_match_bootstrap)",
                           "generator": boot["generator"], "interval_method": "percentile; numpy.quantile method='linear'"}
    for name in QUANTITIES:
        out[f"{name}_ci"] = percentile_interval(draws[name], level)
    g = draws["G_eng"]
    out["draws_G_eng_le_0"] = int((g <= 0).sum())
    out["draws_L1_ge_0"] = int((draws["L1"] >= 0).sum())
    out["draws_L2_le_0"] = int((draws["L2"] <= 0).sum())
    out["r_nonfinite_draws"] = int((~np.isfinite(draws["r"])).sum())
    l_reading = {"L1_upper_lt_0": out["L1_ci"][1] < 0.0, "L2_lower_gt_0": out["L2_ci"][0] > 0.0}
    r_reading = {"r_upper_lt_0.25": bool(out["r_ci"][1] < L1_WEIGHT), "r_lower_gt_0.75": bool(out["r_ci"][0] > L2_WEIGHT)}
    out["equivalence"] = {
        "every_draw_G_eng_positive": bool((g > 0).all()), "L_reading": l_reading, "r_reading": r_reading,
        "readings_agree": l_reading["L1_upper_lt_0"] == r_reading["r_upper_lt_0.25"]
                          and l_reading["L2_lower_gt_0"] == r_reading["r_lower_gt_0.75"],
        "exactly_50_draws_on_a_threshold": {"L1_ge_0": out["draws_L1_ge_0"] == 50, "L2_le_0": out["draws_L2_le_0"] == 50},
        "rule": "PRESPEC_E1 section 4: with G_eng > 0 in every draw the L and r readings agree except when exactly 50 of 2,000 "
                "draws sit on the threshold side (linear interpolation); the decision is made by L"}
    if return_draws:
        out["draws"] = draws
    return out


def _bounds(value: Any, name: str) -> tuple[float, float]:
    low, high = (float(item) for item in value)
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"{name} is not an interval: {low}, {high}")
    return low, high


def decide(out: Mapping[str, Any]) -> str:
    """PRESPEC_E1 section 5 (Ruling 16) on the intervals G_eng_ci, L1_ci, L2_ci and net_ci: no_gain, inconclusive, keep,
    rewrite, net_gain or indistinguishable."""

    bounds = {name: _bounds(out[name], name) for name in ("G_eng_ci", "L1_ci", "L2_ci", "net_ci")}
    g_low, g_high = bounds["G_eng_ci"]
    if g_high < 0.0:
        return "no_gain"
    if g_low <= 0.0:
        return "inconclusive"
    if bounds["L1_ci"][1] < 0.0:
        return "keep"
    if bounds["L2_ci"][0] > 0.0:
        return "rewrite"
    return "net_gain" if bounds["net_ci"][0] > 0.0 else "indistinguishable"


def decision(out: Mapping[str, Any]) -> dict[str, Any]:
    """decide() with its section 5 text, the rule 3c note and the rule 4 default handling."""

    outcome = decide(out)
    exceeds = outcome == "indistinguishable" and _bounds(out["net_ci"], "net_ci")[1] < 0.0
    return {"outcome": outcome, "text": DECISIONS[outcome], "placebo_gain_exceeds_engagement_gain": bool(exceeds),
            "note": NET_NOTE if exceeds else None, "default_handling": DEFAULT_HANDLING if outcome in DEFAULT_OUTCOMES else None,
            "rule": DECISION_RULE,
            "intervals": {name: [float(v) for v in out[name]] for name in ("G_eng_ci", "L1_ci", "L2_ci", "net_ci")}}


# ----------------------------------------------------------------------------- the four arrays and the sensitivities
def _arm_losses(y: Any, p: Mapping[str, Any], match: np.ndarray, mask: Any) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    y = np.asarray(y)
    select = np.ones(y.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if select.shape != y.shape or match.shape != y.shape:
        raise ValueError("labels, match ids and the row mask must be aligned")
    if not select.any():
        return match[:0], {"q": np.zeros(0), "pt": np.zeros(0)}, match[:0]
    ids, q = qeval.per_match_loss(y[select], np.asarray(p["lightgbm"], dtype=np.float64)[select], match[select])
    ids_pt, pt = qeval.per_match_loss(y[select], np.asarray(p["PT_flex"], dtype=np.float64)[select], match[select])
    if not np.array_equal(ids, ids_pt):
        raise ValueError("the two cells of an arm cover different matches")
    return ids, {"q": q, "pt": pt}, match[select]


def four_arrays(match: Any, y_eng: Any, p_eng: Mapping[str, Any], y_plc: Any, p_plc: Mapping[str, Any], *,
                eng_mask: Any = None, plc_mask: Any = None, intersect: bool = False
                ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """(ascending match ids, the four per-match mean-loss arrays eng_q, eng_pt, plc_q, plc_pt, counts).  Pair i of both
    arms is in match[i].  Without ``intersect`` both arms must cover the same matches (qeval.align_cells raises); with it
    only the matches with rows in all four arrays are kept (the r_0.01 rule, section 4)."""

    match = np.asarray(match)
    ids_e, eng, rows_e = _arm_losses(y_eng, p_eng, match, eng_mask)
    ids_p, plc, rows_p = _arm_losses(y_plc, p_plc, match, plc_mask)
    info: dict[str, Any] = {"matches_eng": int(ids_e.size), "matches_plc": int(ids_p.size)}
    if intersect:
        keep = np.intersect1d(ids_e, ids_p)
        info["matches_dropped"] = int(np.union1d(ids_e, ids_p).size - keep.size)
        at_e, at_p = np.isin(ids_e, keep), np.isin(ids_p, keep)
        ids_e, eng = ids_e[at_e], {k: v[at_e] for k, v in eng.items()}
        ids_p, plc = ids_p[at_p], {k: v[at_p] for k, v in plc.items()}
        rows_e, rows_p = rows_e[np.isin(rows_e, keep)], rows_p[np.isin(rows_p, keep)]
        if keep.size == 0:
            empty = {name: np.zeros(0) for name in ("eng_q", "eng_pt", "plc_q", "plc_pt")}
            return keep, empty, {**info, "rows_eng": 0, "rows_plc": 0, "matches": 0}
    shared, aligned = qeval.align_cells({"eng_q": (ids_e, eng["q"]), "eng_pt": (ids_e, eng["pt"]),
                                         "plc_q": (ids_p, plc["q"]), "plc_pt": (ids_p, plc["pt"])})
    info.update(rows_eng=int(rows_e.size), rows_plc=int(rows_p.size), matches=int(shared.size))
    return shared, aligned, info


def r001_masks(svi_eng: Any, w_plc: Any, threshold: float = R001) -> tuple[np.ndarray, np.ndarray]:
    """Rows kept by the label-size sensitivity: |engagement SVI| >= 0.01 and |placebo W| >= 0.01 (section 4 (1))."""

    return np.abs(np.asarray(svi_eng, dtype=np.float64)) >= threshold, np.abs(np.asarray(w_plc, dtype=np.float64)) >= threshold


def objective_free_mask(obj_eng: Any, obj_plc: Any) -> np.ndarray:
    """Pairs where neither window includes an objective (section 4 (2))."""

    return ~np.asarray(obj_eng, dtype=bool) & ~np.asarray(obj_plc, dtype=bool)


def quintile_strata(v_tau: Any) -> tuple[np.ndarray, np.ndarray]:
    """Quintile edges of the evaluation role's matched engagement V(tau) (numpy.quantile linear) and each pair's stratum
    0-4 by its engagement's V(tau) (numpy.searchsorted side='right', the gold-bin convention of PRESPEC_E1 section 3)."""

    v = np.asarray(v_tau, dtype=np.float64)
    edges = np.quantile(v, QUINTILES, method="linear")
    return edges, np.searchsorted(edges, v, side="right")


def stratified_r(match: Any, v_tau: Any, y_eng: Any, p_eng: Mapping[str, Any], y_plc: Any, p_plc: Mapping[str, Any]) -> dict[str, Any]:
    """r_strat = sum_s w_s G_plc,s / sum_s w_s G_eng,s with w_s the stratum's share of the engagements and G_s match
    weighted inside the stratum (each match with rows there weighs equally; scale_headcount.py:177-205)."""

    match, y_eng, y_plc = np.asarray(match), np.asarray(y_eng), np.asarray(y_plc)
    edges, strata = quintile_strata(v_tau)
    n = strata.size
    table: dict[str, Any] = {}
    numerator = denominator = 0.0
    for stratum in range(len(QUINTILES) + 1):
        rows = strata == stratum
        if not rows.any():
            table[str(stratum)] = {"rows": 0, "matches": 0, "weight": 0.0, "G_eng": None, "G_plc": None, "r": None}
            continue
        ids, arrays, _ = four_arrays(match[rows], y_eng[rows], {k: np.asarray(v)[rows] for k, v in p_eng.items()}, y_plc[rows],
                                     {k: np.asarray(v)[rows] for k, v in p_plc.items()})
        gains = gains_and_ratio(arrays["eng_q"], arrays["eng_pt"], arrays["plc_q"], arrays["plc_pt"])
        weight = float(rows.sum()) / n
        numerator += weight * gains["G_plc"]
        denominator += weight * gains["G_eng"]
        table[str(stratum)] = {"rows": int(rows.sum()), "matches": int(ids.size), "weight": weight, "G_eng": gains["G_eng"],
                               "G_plc": gains["G_plc"], "r": gains["r"]}
    return {"edges": [float(edge) for edge in edges], "strata": table, "numerator": numerator, "denominator": denominator,
            "r": numerator / denominator if denominator != 0.0 else None,
            "rule": "quintile edges of the role's matched engagement V(tau) (numpy.quantile linear, searchsorted right); a pair "
                    "is assigned by its engagement's V(tau); w_s = the stratum's share of the engagements; G_s match weighted "
                    "within the stratum (PRESPEC_E1 section 3 균형)"}


def dedup_keep(match: Any, s_ms: Any, eng_keys: Any) -> np.ndarray:
    """Among pairs that chose the same placebo window (match, s), the one with the smallest selection hash
    (e1_candidates.selection_digest of the engagement key and s; PRESPEC_E1 section 3 선택 and 균형)."""

    match, s = np.asarray(match).astype(np.int64), np.asarray(s_ms).astype(np.int64)
    keys = np.asarray(eng_keys)
    best: dict[tuple[int, int], tuple[str, int]] = {}
    for i in range(match.size):
        digest = ec.selection_digest(keys[i], int(s[i]))
        group = (int(match[i]), int(s[i]))
        if group not in best or digest < best[group][0]:
            best[group] = (digest, i)
    keep = np.zeros(match.size, dtype=bool)
    keep[[i for _, i in best.values()]] = True
    return keep


def gain(y: Any, p_q: Any, p_pt: Any, match: Any, *, level: float = LEVEL) -> dict[str, Any]:
    """G = Brier(PT_flex) - Brier(LightGBM), match weighted, with its percentile interval on shared match draws
    (qeval.paired_match_bootstrap, 2,000 draws, seed 20261011).  An empty row set is reported, not raised."""

    y = np.asarray(y)
    if y.size == 0:
        return {"rows": 0, "matches": 0, "point": None, "interval": None, "level": float(level), "brier": None, "note": "no rows"}
    losses = {"lightgbm": qeval.per_match_loss(y, p_q, match), "PT_flex": qeval.per_match_loss(y, p_pt, match)}
    shared, aligned = qeval.align_cells(losses)
    boot = qeval.paired_match_bootstrap(shared, aligned, {"G": {"PT_flex": 1.0, "lightgbm": -1.0}}, levels=level,
                                        resamples=RESAMPLES, seed=SEED)
    contrast = boot["contrasts"]["G"]
    return {"rows": int(y.size), "matches": int(shared.size), "point": contrast["point"], "interval": contrast["interval"],
            "level": contrast["level"], "brier": {name: boot["cells"][name]["point"] for name in ("lightgbm", "PT_flex")},
            "sign": "Brier(PT_flex) - Brier(LightGBM)"}


def _subset_joint(ids: np.ndarray, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> dict[str, Any]:
    if ids.size == 0:
        return {**info, "note": "no match with rows in all four arrays; nothing to estimate"}
    return {**info, **joint_bootstrap(arrays["eng_q"], arrays["eng_pt"], arrays["plc_q"], arrays["plc_pt"])}


def _pair_subset(pairs: Mapping[str, Any], keep: np.ndarray) -> dict[str, Any]:
    keep = np.asarray(keep, dtype=bool)
    if not keep.any():
        return {"rows_eng": 0, "rows_plc": 0, "matches": 0, "note": "no pair; nothing to estimate"}
    take = {k: np.asarray(v)[keep] for k, v in pairs["eng_p"].items()}, {k: np.asarray(v)[keep] for k, v in pairs["plc_p"].items()}
    ids, arrays, info = four_arrays(np.asarray(pairs["match"])[keep], np.asarray(pairs["eng_y"])[keep], take[0],
                                    np.asarray(pairs["plc_y"])[keep], take[1])
    return _subset_joint(ids, arrays, info)


def evaluate_pairs(pairs: Mapping[str, Any]) -> dict[str, Any]:
    """Sections 4-5 on one role's pairs: the four arrays, the joint bootstrap and decision, the sensitivities (r_0.01,
    objective-free pairs, stratified r and de-duplicated r, the last two computed whatever their triggers say) and the
    auxiliaries (AUC of both models in both arms, G_eng_full with the main models on the matched engagements,
    G_plc / 0.01386, the main-model gain on matched against unmatched engagements).  Nothing here reads a file."""

    match = np.asarray(pairs["match"])
    y_eng, y_plc = np.asarray(pairs["eng_y"]), np.asarray(pairs["plc_y"])
    p_eng = {family: np.asarray(pairs["eng_p"][family], dtype=np.float64) for family in FAMILIES}
    p_plc = {family: np.asarray(pairs["plc_p"][family], dtype=np.float64) for family in FAMILIES}
    ids, arrays, info = four_arrays(match, y_eng, p_eng, y_plc, p_plc)
    primary = joint_bootstrap(arrays["eng_q"], arrays["eng_pt"], arrays["plc_q"], arrays["plc_pt"])
    cells = {"eng": {"lightgbm": float(arrays["eng_q"].mean()), "PT_flex": float(arrays["eng_pt"].mean())},
             "plc": {"lightgbm": float(arrays["plc_q"].mean()), "PT_flex": float(arrays["plc_pt"].mean())}}
    triggers = {name: bool(pairs["triggers"][name]) for name in ("stratified_r", "deduplicated_r")}

    eng_mask, plc_mask = r001_masks(pairs["eng_svi"], pairs["plc_w"])
    r_ids, r_arrays, r_info = four_arrays(match, y_eng, p_eng, y_plc, p_plc, eng_mask=eng_mask, plc_mask=plc_mask, intersect=True)
    free = objective_free_mask(pairs["obj_eng"], pairs["obj_plc"])
    keep = dedup_keep(match, pairs["s_ms"], pairs["eng_key"])
    sensitivities = {
        "r_0.01": {"rule": "rows with |engagement SVI| >= 0.01 and placebo rows with |W| >= 0.01; matches with rows in all four "
                           "arrays (section 4 (1); 0.01 = NEAR_ZERO[1], assemble.py:40)", "threshold": R001,
                   **_subset_joint(r_ids, r_arrays, r_info)},
        "objective_free": {"rule": "pairs where neither window includes an objective (section 4 (2); diagnostics.py:152-158)",
                           "pairs": int(free.sum()), "pairs_with_an_objective": int((~free).sum()),
                           **_pair_subset(pairs, free)},
        "stratified_r": {"trigger": triggers["stratified_r"], "trigger_rule": "|SMD| > 0.1 on V(start) or game time",
                         "computed_regardless_of_trigger": True, **stratified_r(match, pairs["v_tau"], y_eng, p_eng, y_plc, p_plc)},
        "deduplicated_r": {"trigger": triggers["deduplicated_r"], "trigger_rule": "reused placebo windows > 5% of the placebo rows",
                           "computed_regardless_of_trigger": True, "pairs_kept": int(keep.sum()), "pairs_dropped": int((~keep).sum()),
                           **_pair_subset(pairs, keep)},
    }

    auc = {}
    for arm, y, p in ((ENG, y_eng, p_eng), (PLC, y_plc, p_plc)):
        auc[arm] = eq._renamed(e2_run.evaluate_pair(y, p["lightgbm"], p["PT_flex"], match))
        for family in FAMILIES:
            if abs(auc[arm]["brier"][family] - cells[arm][family]) > RECOMPUTE_ATOL:
                raise RuntimeError(f"{arm} {family}: the match-weighted Brier differs between qeval.evaluate and the per-match losses")
    main = pairs["main"]
    matched = np.asarray(main["matched"], dtype=bool)
    main_y, main_match = np.asarray(main["y"]), np.asarray(main["match"])
    main_p = {family: np.asarray(main["p"][family], dtype=np.float64) for family in FAMILIES}

    def main_gain(rows: np.ndarray) -> dict[str, Any]:
        return gain(main_y[rows], main_p["lightgbm"][rows], main_p["PT_flex"][rows], main_match[rows])

    full, unmatched, every = main_gain(matched), main_gain(~matched), main_gain(np.ones(matched.size, dtype=bool))
    g_plc_ci = primary["G_plc_ci"]
    auxiliary = {
        "auc": auc,
        "G_eng_full": {**full, "rule": "q_run_1's main models (selected LightGBM lgbm_l15_m200 and PT_flex ptflex_k8_C0.001 cells) "
                                       "on the matched engagements of the role"},
        "G_plc_over_main_test_gain": {"point": primary["point"]["G_plc"] / MAIN_TEST_GAIN,
                                      "interval": [g_plc_ci[0] / MAIN_TEST_GAIN, g_plc_ci[1] / MAIN_TEST_GAIN],
                                      "reference": MAIN_TEST_GAIN,
                                      "note": "reference = the full engagement TEST gain printed in PRESPEC_E1 (97.5% level there)"},
        "main_matched_vs_unmatched": {"matched": full, "unmatched": unmatched, "all": every,
                                      "difference_point": (None if full["point"] is None or unmatched["point"] is None
                                                           else full["point"] - unmatched["point"]),
                                      "rule": "descriptive: the main-model gain on the role's matched engagements and on its "
                                              "unmatched engagements, each with its own match bootstrap"},
    }
    return {"rows": int(match.size), "matches": int(ids.size), "four_arrays": info, "cells": cells, "primary": primary,
            "decision": decision(primary), "sensitivities": sensitivities, "auxiliary": auxiliary}


# ----------------------------------------------------------------------------- reading one role (shared by rehearsal and TEST)
def read_sealed(label: str, path: Path, fields: Sequence[str] | None = None) -> Any:
    """The only reader of a sealed file (or of its non-TEST stand-in in the same layout): the named npz fields, or the JSON."""

    if label not in SEALED_LABELS:
        raise ValueError(f"{label} is not a sealed file of PRESPEC_E1 section 6 step 7")
    if fields is None:
        return _read(path)
    return eq._load_fields(Path(path), fields)


def sealed_paths(paths: Paths) -> dict[str, Path]:
    return {"plc_labels": paths.plc_sealed, "pred_end": paths.pred_sealed, "windows_summary": paths.windows_sealed,
            "balance_summary": paths.balance_sealed, "eng_labels": paths.q_sealed}


def main_paths(paths: Paths) -> dict[str, Path]:
    return {family: paths.main_scores(family) for family in FAMILIES}


def load_public(paths: Paths, code: int) -> dict[str, Any]:
    """Both row tables, the key map, the public placebo V, the windows and q_run_1's rows, checked for the pairing (placebo
    row i = engagement row i = q_run_1 row window_id[i]), and the role's pair positions.  No label of a sealed role."""

    eng, plc = eq._load_fields(paths.rows(ENG), PUBLIC_ENG), eq._load_fields(paths.rows(PLC), PUBLIC_PLC)
    kmap = eq._load_fields(paths.key_map, ("plc_row", "eng_row", "window_id", "engagement_key_plc", "engagement_key_eng"))
    pred, windows = eq._load_fields(paths.pred, PRED_FIELDS), eq._load_fields(paths.windows, WINDOW_FIELDS)
    q = eq._load_fields(paths.q_rows, Q_FIELDS)
    n = eng["role"].size
    if not (plc["role"].size == kmap["window_id"].size == pred["window_id"].size == n):
        raise Blocked("the placebo table, the engagement table, the key map and PRED_PLACEBO_h90 differ in length")
    window = kmap["window_id"].astype(np.int64)
    pair = np.arange(n)
    checks = {
        "key_map_rows": np.array_equal(kmap["plc_row"], pair) and np.array_equal(kmap["eng_row"], pair),
        "key_map_keys": (np.array_equal(kmap["engagement_key_plc"], plc["engagement_key"])
                         and np.array_equal(kmap["engagement_key_eng"], eng["engagement_key"])),
        "same_role_and_match": (np.array_equal(eng["role"].astype(np.int64), plc["role"].astype(np.int64))
                                and np.array_equal(eng["match_index"].astype(np.int64), plc["match_index"].astype(np.int64))),
        "windows_increasing": bool(n and np.all(np.diff(window) > 0) and window[0] >= 0 and window[-1] < q["role"].size),
    }
    if all(checks.values()):
        checks.update({
            "pred_is_the_pairs": (np.array_equal(pred["window_id"].astype(np.int64), window)
                                  and np.array_equal(pred["engagement_key"], eng["engagement_key"])
                                  and np.array_equal(pred["role"].astype(np.int64), eng["role"].astype(np.int64))
                                  and np.array_equal(pred["match_index"].astype(np.int64), eng["match_index"].astype(np.int64))),
            "q_run_1_rows_are_the_engagements": (np.array_equal(q["engagement_key"][window], eng["engagement_key"])
                                                 and np.array_equal(q["role"][window].astype(np.int64), eng["role"].astype(np.int64))
                                                 and np.array_equal(q["match_index"][window].astype(np.int64),
                                                                    eng["match_index"].astype(np.int64))),
            "windows_are_the_pairs": (np.array_equal(windows["engagement_key"][window], eng["engagement_key"])
                                      and np.array_equal(windows["s_ms"][window].astype(np.int64), pred["s_ms"].astype(np.int64))
                                      and np.array_equal(pred["end_ms"].astype(np.int64),
                                                         pred["s_ms"].astype(np.int64) + windows["D_ms"][window].astype(np.int64))
                                      and bool(np.all(pred["s_ms"] >= 0))),
            "placebo_v_start_is_pt_p_pre": np.array_equal(plc["pt_p_pre"], pred["p_start"]),
        })
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise Blocked(f"row-table key mismatch (PRESPEC_E1 section 7): {failed}")
    rows = np.flatnonzero(eng["role"].astype(np.int64) == code)
    if rows.size == 0:
        raise Blocked(f"no pair of role {code}")
    return {"eng": eng, "plc": plc, "pred": pred, "windows": windows, "q": q, "window": window, "rows": rows,
            "start_age": windows["start_age_plc_ms"][window].astype(np.int64), "checks": {name: True for name in checks}}


def objective_flags(paths: Paths, code: int, pub: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Objective inclusion of both windows of the role's pairs (diagnostics.py:152-158): the engagement's OBJECTIVE_COUNTS
    sum at query_h90 above that at query_pre (projection; e1_rows.engagement_endpoints), the placebo's at s + D above
    that at s (the STATES parts of the role's shard group, read by e1_rows.part_index with every matched window of that
    group; only the role's pairs are kept)."""

    rows, window = pub["rows"], pub["window"]
    engagements = eq._load_fields(paths.engagements, er1.ENGAGEMENT_COLUMNS)
    endpoints = er1.engagement_endpoints(engagements, pub["q"])
    counts = np.load(paths.objective_counts, mmap_mode="r", allow_pickle=False)
    q_rows = window[rows]
    obj_eng = er1.objective_gain(np.asarray(counts[endpoints["query_pre"][q_rows]]),
                                 np.asarray(counts[endpoints["query_h90"][q_rows]])) > 0
    group = SHARD_GROUP[code]
    codes = [role for role, name in SHARD_GROUP.items() if name == group]
    record = _read(paths.states_record)
    parts = {name: entry for name, entry in record["parts"].items() if name.rsplit("_", 1)[0] == group}
    changed = [entry["file"] for entry in parts.values() if sha256_file(paths.states_dir / entry["file"]) != entry["sha256"]]
    if not parts or changed:
        raise Blocked(f"the {group} state parts are missing or differ from STATES_h90.json: {changed}")
    role = pub["eng"]["role"].astype(np.int64)
    select = np.flatnonzero(np.isin(role, codes))
    index = er1.part_index(paths.states_dir, {"parts": parts}, window[select], pub["eng"]["match_index"][select],
                           pub["pred"]["s_ms"][select], pub["pred"]["end_ms"][select], pub["start_age"][select])
    gained = er1.objective_gain(index["objective_start"], index["objective_end"]) > 0
    obj_plc = gained[np.searchsorted(select, rows)]
    return obj_eng, obj_plc, {"shard_group": group, "parts": sorted(entry["file"] for entry in parts.values()),
                              "rule": "OBJECTIVE_COUNTS summed at the end state above the start state (diagnostics.py:152-158)"}


def parts_follow_shard_groups(paths: Paths) -> bool:
    """Every state part holds windows of its shard group's roles only and every matched window lies in exactly one part:
    what objective_flags needs to index one group (test_* at the TEST step).  Reads each part's window_id member and the
    public role of each window; returns a boolean only (nothing is counted)."""

    record = _read(paths.states_record)
    pred = eq._load_fields(paths.pred, ("window_id", "role"))
    window, role = np.asarray(pred["window_id"], dtype=np.int64), np.asarray(pred["role"]).astype(np.int64)
    seen = np.zeros(window.size, dtype=np.int64)
    for name, entry in record["parts"].items():
        codes = [code for code, group in SHARD_GROUP.items() if group == name.rsplit("_", 1)[0]]
        with np.load(paths.states_dir / entry["file"], allow_pickle=False) as part:
            ids = np.asarray(part["window_id"], dtype=np.int64)
        at = np.searchsorted(window, ids)
        if ids.size and (np.any(at >= window.size) or not np.array_equal(window[np.minimum(at, window.size - 1)], ids)):
            return False
        if not codes or not np.isin(role[at], codes).all():
            return False
        np.add.at(seen, at, 1)
    return bool((seen == 1).all())


def _scores_of(scores: Mapping[str, Mapping[str, Any]], arm: str, keys: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for family in FAMILIES:
        scored = scores[arm][family]
        if not (np.array_equal(np.asarray(scored["row"], dtype=np.int64), rows) and np.array_equal(np.asarray(scored["engagement_key"]), keys[rows])):
            raise Blocked(f"{arm} {family}: the scores are not the role's rows of the {arm} table in order")
        p = np.asarray(scored["p"], dtype=np.float64)
        vs._check_probabilities(p, f"{arm} {family} p")
        out[family] = p
    return out


def load_pairs(paths: Paths, role: str, opened: Mapping[str, Path], scores: Mapping[str, Mapping[str, Any]],
               main: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The role's pairs for evaluate_pairs from the public files, the five ``opened`` files (sealed at TEST, stand-ins in
    the rehearsal), the four cells' scores and the two main-model score files, with every consistency check."""

    code = ROLE_CODES[role]
    pub = load_public(paths, code)
    rows, window = pub["rows"], pub["window"][pub["rows"]]
    eng, plc = pub["eng"], pub["plc"]
    # placebo labels, checked against W = V(s + D) - V(s) of the placebo end V -----------------------------------------
    plc_sealed = read_sealed("plc_labels", opened["plc_labels"], ("engagement_key", *SEALED_COLUMNS))
    at = eq.align_keys(plc["engagement_key"][rows], plc_sealed["engagement_key"])
    plc_labels = {name: np.asarray(plc_sealed[name])[at] for name in SEALED_COLUMNS}
    pred_end = read_sealed("pred_end", opened["pred_end"], ("window_id", "engagement_key", "p_end"))
    at_end = eq.align_keys(window, np.asarray(pred_end["window_id"], dtype=np.int64))
    if not np.array_equal(np.asarray(pred_end["engagement_key"])[at_end], eng["engagement_key"][rows]):
        raise Blocked("the placebo end V rows are not the role's windows")
    expected = er1.placebo_labels(pub["pred"]["p_start"][rows], np.asarray(pred_end["p_end"], dtype=np.float64)[at_end])
    labels_equal = all(np.array_equal(np.asarray(expected[name]).astype(np.int64), np.asarray(plc_labels[name]).astype(np.int64))
                       for name in ("y", "exact_zero", "near_zero_bits")) and np.array_equal(expected["label_svi_delta"],
                                                                                          plc_labels["label_svi_delta"])
    if not labels_equal:
        raise Blocked("the placebo labels differ from W = V(s + D) - V(s) of the placebo end V (PRESPEC_E1 section 3 라벨)")
    # engagement labels (q_run_1's file covers every q row of the role) ------------------------------------------------
    eng_sealed = read_sealed("eng_labels", opened["eng_labels"], ("engagement_key", "y", "label_svi_delta"))
    y_all, svi_all = np.asarray(eng_sealed["y"]), np.asarray(eng_sealed["label_svi_delta"], dtype=np.float64)
    if not (np.isin(y_all, (0, 1)).all() and np.isfinite(svi_all).all()
            and np.array_equal(y_all.astype(np.int64), (svi_all > 0).astype(np.int64))):
        raise Blocked("the engagement labels do not follow y = 1[SVI > 0]")
    at_eng = eq.align_keys(eng["engagement_key"][rows], eng_sealed["engagement_key"])
    # sealed summaries ---------------------------------------------------------------------------------------------
    windows_block = read_sealed("windows_summary", opened["windows_summary"])[role]
    balance_block = read_sealed("balance_summary", opened["balance_summary"])["roles"][role]
    q_rows = np.flatnonzero(pub["q"]["role"].astype(np.int64) == code)
    obj_eng, obj_plc, objective = objective_flags(paths, code, pub)
    keep = dedup_keep(eng["match_index"][rows], pub["pred"]["s_ms"][rows], eng["engagement_key"][rows])
    shares, reuse = balance_block["objective_share"], balance_block["reuse"]
    summary_checks = {"windows_matched_equals_pairs": windows_block["matched"] == rows.size,
                      "windows_engagements_equal_q_rows": windows_block["engagements"] == q_rows.size,
                      "balance_pairs_equal_pairs": balance_block["pairs"] == rows.size,
                      "objective_flags_equal_the_balance_table": (
                          int(obj_eng.sum()) == shares["engagement"]["with_objective_gain"]
                          and int(obj_plc.sum()) == shares["placebo"]["with_objective_gain"]
                          and int((~obj_eng & ~obj_plc).sum()) == shares["pairs_without_objective_in_either"]),
                      "reused_pairs_equal_the_balance_table": int((~keep).sum()) == reuse["rows"] - reuse["distinct_windows"]}
    if not all(summary_checks.values()):
        raise Blocked(f"the sealed summaries do not describe the role's pairs: {sorted(k for k, v in summary_checks.items() if not v)}")
    triggers = {"stratified_r": bool(balance_block["triggers"]["stratified_r"]["flag"]),
                "deduplicated_r": bool(balance_block["triggers"]["deduplicated_r"]["flag"])}
    # the four cells' scores and the main models ------------------------------------------------------------------------
    p_eng, p_plc = _scores_of(scores, ENG, eng["engagement_key"], rows), _scores_of(scores, PLC, plc["engagement_key"], rows)
    at_main = eq.align_keys(pub["q"]["engagement_key"][q_rows], eng_sealed["engagement_key"])
    main_p = {}
    for family in FAMILIES:
        stored = eq._load_fields(main[family], ("row", "engagement_key", "p"))
        if not (np.array_equal(np.asarray(stored["row"], dtype=np.int64), q_rows)
                and np.array_equal(np.asarray(stored["engagement_key"]), pub["q"]["engagement_key"][q_rows])):
            raise Blocked(f"main {family}: the score file does not hold q_run_1's rows of the role in order")
        main_p[family] = np.asarray(stored["p"], dtype=np.float64)
        vs._check_probabilities(main_p[family], f"main {family} p")
    matched = np.isin(q_rows, window)
    if int(matched.sum()) != rows.size:
        raise Blocked("the matched engagements are not q_run_1 rows of the role")
    pairs = {"match": eng["match_index"][rows].astype(np.int64), "eng_y": y_all[at_eng], "eng_svi": svi_all[at_eng], "eng_p": p_eng,
             "plc_y": plc_labels["y"].astype(np.int64), "plc_w": plc_labels["label_svi_delta"], "plc_p": p_plc,
             "v_tau": eng["pt_p_pre"][rows], "eng_key": eng["engagement_key"][rows], "s_ms": pub["pred"]["s_ms"][rows].astype(np.int64),
             "obj_eng": obj_eng, "obj_plc": obj_plc, "triggers": triggers,
             "main": {"y": y_all[at_main], "p": main_p, "match": pub["q"]["match_index"][q_rows].astype(np.int64), "matched": matched}}
    info = {"split": role, "summaries": {"windows": windows_block, "balance": balance_block}, "objective": objective,
            "checks": {**pub["checks"], "placebo_labels_equal_W_of_the_end_v": True, "engagement_labels_follow_the_direction_rule": True,
                       **summary_checks, "scores_are_the_role_rows": True, "main_scores_are_the_q_rows": True}}
    return pairs, info


def evaluate_opened(paths: Paths, role: str, opened: Mapping[str, Path], scores: Mapping[str, Mapping[str, Any]],
                    main: Mapping[str, Path]) -> dict[str, Any]:
    """The evaluation the TEST step runs once after its opening (role TEST, the five sealed files and q_run_1's TEST_SCORES)
    and the rehearsal runs on Q_CAL with stand-ins (Ruling 55): load_pairs, then evaluate_pairs."""

    pairs, info = load_pairs(paths, role, opened, scores, main)
    return {**info, **evaluate_pairs(pairs)}


# ----------------------------------------------------------------------------- train (PRESPEC_E1 section 6 step 4)
@dataclass(frozen=True)
class Entry:
    """One job of one run for the driver."""

    arm: str
    key: str
    family: str
    jobs_dir: Path
    spec: Path
    job: Any = None

    @property
    def name(self) -> str:
        return entry_name(self.arm, self.key)

    @property
    def directory(self) -> Path:
        return Path(self.jobs_dir) / self.key


def entry_name(arm: str, key: str) -> str:
    return f"{arm}/{key}"


def job_order(jobs: Sequence[Any]) -> list[tuple[str, str, Any]]:
    """(arm, family, job) in launch order: the LightGBM jobs of both runs first (only one runs at a time, so they set the
    wall time), then PT_flex."""

    return [(arm, family, job) for family in FAMILIES for arm in ARMS for job in jobs if job.family == family]


def _save_states(dirs: Sequence[Path], running: Mapping[str, Mapping[str, Any]], by_name: Mapping[str, Entry]) -> None:
    for jobs_dir in dirs:
        eq._save_state(jobs_dir, {by_name[name].key: info for name, info in running.items() if by_name[name].jobs_dir == jobs_dir})


def drive(entries: Sequence[Entry], *, workers: int = MAX_WORKERS, poll_s: float = 5.0,
          launcher: Callable[..., Any] = eq.launch_worker, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The resumable scheduling loop of e4_q.train_stage over both runs: at most ``workers`` worker processes and one
    LightGBM at a time (e4_q.next_launches); a completed job is skipped, live workers of an earlier driver are adopted
    (e4_q._adopt), a failed attempt is kept as RESULT_attempt<k>.json before its one retry, a terminal failure or too many
    interruptions blocks the job.  Ledger and state files per run (jobs/TRAIN_LEDGER.jsonl, jobs/TRAIN_STATE.json)."""

    by_name = {entry.name: entry for entry in entries}
    dirs = list(dict.fromkeys(Path(entry.jobs_dir) for entry in entries))
    running: dict[str, dict[str, Any]] = {}
    for jobs_dir in dirs:
        own = [entry for entry in entries if Path(entry.jobs_dir) == jobs_dir]
        for key, info in eq._adopt(jobs_dir, {entry.key: entry.spec for entry in own}).items():
            running[entry_name(own[0].arm, key)] = info
    blocked: dict[str, str] = {}
    events: list[dict[str, Any]] = [{"event": "driver_started", "adopted": sorted(running)}]
    for jobs_dir in dirs:
        eq._ledger(jobs_dir, "driver_started", pid=os.getpid(), workers=int(workers),
                   adopted=sorted(by_name[name].key for name in running if Path(by_name[name].jobs_dir) == jobs_dir))
    print(json.dumps({"driver": os.getpid(), "jobs": len(entries), "adopted": sorted(running)}), flush=True)
    while True:
        for name in list(running):
            info = running[name]
            code = info["handle"].poll()
            if code is None:
                continue
            running.pop(name)
            entry = by_name[name]
            state = eq.job_state(entry.directory)
            if state == "pending":                     # ended without a result: an interruption
                eq._count_interrupt(entry.directory)
            seconds = time.perf_counter() - info["tick"]
            eq._ledger(entry.jobs_dir, "ended", key=entry.key, pid=info["pid"], launch=info["launch"], returncode=code, state=state,
                       seconds=seconds)
            events.append({"event": "ended", "name": name, "state": state})
            print(json.dumps({"end": name, "state": state, "returncode": code, "seconds": round(seconds, 1)}), flush=True)
        pending: list[tuple[str, str]] = []
        for entry in entries:
            if entry.name in running or entry.name in blocked:
                continue
            state = eq.job_state(entry.directory)
            if state == "completed":
                continue
            if state == "terminal":
                blocked[entry.name] = "terminal failure (RESULT.json)"
                continue
            if state == "failed":
                attempt = len(list(entry.directory.glob("RESULT_attempt*.json"))) + 1
                if attempt > eq.RETRIES:
                    blocked[entry.name] = f"failed in {attempt} attempts"
                    continue
                os.replace(entry.directory / "RESULT.json", entry.directory / f"RESULT_attempt{attempt}.json")
                eq._ledger(entry.jobs_dir, "kept_failed_attempt", key=entry.key, attempt=attempt)
            if eq._interrupts(entry.directory) > eq.MAX_INTERRUPTS:
                blocked[entry.name] = f"interrupted more than {eq.MAX_INTERRUPTS} times"
                continue
            pending.append((entry.name, entry.family))
        if not pending and not running:
            break
        for name in eq.next_launches(pending, [by_name[n].family for n in running], workers):
            entry = by_name[name]
            launch = len(list(entry.directory.glob("worker_*.stdout.log"))) + 1
            process = launcher(entry.spec, entry.directory, launch, dict(env or {}))
            running[name] = {"handle": process, "pid": process.pid, "launch": launch, "started_utc": utc(),
                             "tick": time.perf_counter(), "spec": str(entry.spec)}
            eq._ledger(entry.jobs_dir, "started", key=entry.key, pid=process.pid, launch=launch, family=entry.family)
            events.append({"event": "started", "name": name, "running": {n: by_name[n].family for n in running}})
            print(json.dumps({"start": name, "pid": process.pid, "launch": launch}), flush=True)
        _save_states(dirs, running, by_name)
        time.sleep(poll_s)
    _save_states(dirs, running, by_name)
    return {"blocked": blocked, "events": events}


def completed_records(entries: Sequence[Entry], digest: str, role_rows: Mapping[str, int],
                      keys_sha256: Mapping[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    """e4_q.job_record of every job (spec, identity, release, model and prediction hashes, Q_DEV/Q_CAL row counts) and its
    Q_DEV keys against its own run's row table (a row-table key mismatch otherwise)."""

    records, problems = [], []
    for entry in entries:
        state = eq.job_state(entry.directory)
        if state != "completed":
            records.append({"arm": entry.arm, "key": entry.key, "family": entry.family, "state": state})
            problems.append(f"{entry.name}: {state}")
            continue
        record, found = eq.job_record(entry.directory, entry.job, digest, role_rows)
        record["arm"] = entry.arm
        if record.get("q_dev_keys_sha256") != keys_sha256[entry.arm]:
            found.append(f"{entry.name}: Q_DEV keys differ from the {entry.arm} row table's (row-table key mismatch)")
        records.append(record)
        problems += found
    return records, problems


def q_dev_keys_sha256(rows_path: Path) -> str:
    """The Q_DEV key hash a worker writes for this row table (q_worker.keys_sha256 over the Q_DEV engagement keys)."""

    from studies.q_production_20261005.q_worker import keys_sha256

    rows = eq._load_fields(rows_path, ("engagement_key", "role"))
    return keys_sha256(rows["engagement_key"][rows["role"].astype(np.int64) == Q_DEV])


def worker_env() -> dict[str, str]:
    """The q supervisor's worker environment (threads 2) with CUDA hidden by -1 (e1_rows.prepare_env): CPU only."""

    return er1.prepare_env()


def prepared_view(paths: Paths, arm: str) -> dict[str, Any]:
    """The run's PREPARED.json without its 'rows' field (a TEST-inclusive total, Ruling 58), checked against the files."""

    prepared = _read(paths.inputs(arm) / "PREPARED.json")
    prepared.pop("rows", None)
    run = paths.run(arm)
    checks = {"rows_file": prepared.get("rows_file") == ROWS_NAME,
              "rows_sha256": prepared.get("rows_sha256") == sha256_file(paths.rows(arm)),
              "xy_sha256": prepared.get("xy_sha256") == sha256_file(run / prepared.get("xy_file", f"inputs/{VARIANT}/XY.npy")),
              "store_build_sha256": prepared.get("store_build_sha256") == sha256_file(run / "raw" / "store" / "BUILD.json"),
              "release_digest": prepared.get("release_digest") == RELEASE_DIGEST, "variant": prepared.get("variant") == VARIANT,
              "contexts": sorted(prepared.get("contexts") or {}) == sorted(eq.SCOPES)
                          and all((run / info["dir"] / "CONTEXT.json").is_file() for info in prepared["contexts"].values())}
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise Blocked(f"{arm}/inputs/main_h90/PREPARED.json disagrees with the run's files: {failed}")
    return prepared


def _eng_junction(paths: Paths) -> dict[str, Any]:
    link, target = paths.run(ENG) / "raw" / "store", paths.q_run / "raw" / "store"
    ok = os.path.lexists(link) and os.path.isjunction(link) and os.path.samefile(link, target)
    if not ok:
        raise Blocked("eng/raw/store is not the directory junction to q_run_1/raw/store")
    return {"link": str(link), "target": str(target), "is_junction": True, "realpath": os.path.realpath(link)}


def role_rows(paths: Paths) -> dict[str, int]:
    """Q_DEV and Q_CAL rows of the two row tables (equal: placebo row i = engagement row i)."""

    counts = {}
    for arm in ARMS:
        role = eq._load_fields(paths.rows(arm), ("role",))["role"].astype(np.int64)
        counts[arm] = {"Q_DEV": int((role == Q_DEV).sum()), "Q_CAL": int((role == Q_CAL).sum())}
    if counts[ENG] != counts[PLC]:
        raise Blocked("the two row tables differ in their Q_DEV or Q_CAL rows")
    return counts[ENG]


def _claim_driver(out: Path) -> None:
    path = Path(out) / DRIVER_NAME
    current = _read(path) if path.is_file() else {}
    pid = current.get("pid")
    if pid and int(pid) != os.getpid() and eq._alive_with(int(pid), "e1_run"):
        raise RuntimeError(f"another E1 train driver (pid {pid}) is running")
    eq._atomic_json(path, {"pid": os.getpid(), "started_utc": utc(), "argv": sys.argv})


def train_stage(paths: Paths = Paths(), *, workers: int = MAX_WORKERS, poll_s: float = 5.0,
                launcher: Callable[..., Any] | None = None) -> int:
    """Both runs' 16 cards through the frozen q worker (PRESPEC_E1 section 3 모델); E1_TRAIN.json once all completed."""

    started, tick = utc(), time.perf_counter()
    target = paths.out / TRAIN_NAME
    if target.exists():
        print(json.dumps({"done": str(target)}), flush=True)
        return 0
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f"workers must be 1-{MAX_WORKERS}")
    records = {name: _passed(paths.source / name, name) for name in ("E1_ROWS.json", "E1_STORE.json", "E1_ENG_INPUTS.json",
                                                                    "E1_PREPARE.json", "E1_ROWS_RUN.json")}
    for arm in ARMS:
        if sha256_file(paths.rows(arm)) != records["E1_ROWS.json"]["files"][f"{arm}/{ROWS_NAME}"]["sha256"]:
            raise Blocked(f"{arm}/{ROWS_NAME} differs from E1_ROWS.json")
    junction = _eng_junction(paths)
    prepared = {arm: prepared_view(paths, arm) for arm in ARMS}
    release = qrelease.verify(paths.release)
    if release["digest"] != RELEASE_DIGEST:
        return _blocked(paths.out, "train", [f"release digest {release['digest']} is not {RELEASE_DIGEST}"])
    jobs = eq.plan_jobs(qjobs.plan())
    if {family: tuple(job.card_id for job in jobs if job.family == family) for family in FAMILIES} != CARDS:
        return _blocked(paths.out, "train", ["the frozen plan's cards are not PRESPEC_E1 section 3's"])
    paths.out.mkdir(parents=True, exist_ok=True)
    _claim_driver(paths.out)
    entries = []
    for arm, family, job in job_order(jobs):
        key = qtest.key_of(job.job_id)
        jobs_dir = paths.jobs(arm)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        spec = eq.ensure_spec(jobs_dir / key, eq.job_spec(job, paths.run(arm), prepared[arm], paths.release, release["digest"], utc()))
        entries.append(Entry(arm=arm, key=key, family=family, jobs_dir=jobs_dir, spec=spec, job=job))
    outcome = drive(entries, workers=workers, poll_s=poll_s, launcher=launcher or eq.launch_worker, env=worker_env())
    counts = role_rows(paths)
    keys = {arm: q_dev_keys_sha256(paths.rows(arm)) for arm in ARMS}
    job_records, problems = completed_records(entries, release["digest"], counts, keys)
    problems += [f"{name}: {reason}" for name, reason in outcome["blocked"].items() if not any(name in p for p in problems)]
    if problems:
        return _blocked(paths.out, "train", problems, jobs=job_records)
    files = {f"{r['arm']}/jobs/{r['key']}/{name}": {"path": str(paths.jobs(r["arm"]) / r["key"] / name), "sha256": digest}
             for r in job_records for name, digest in r["files"].items()}
    record = {
        "version": VERSION, "experiment": "E1", "stage": "train", "passed": True, "files": files, "jobs": job_records,
        "cards": {family: list(cards) for family, cards in CARDS.items()}, "lightgbm_condition": eq.LGBM_CONDITION,
        "q_dev_keys_sha256": keys, "row_counts_q_dev_q_cal": counts, "release_digest": release["digest"],
        "spec_shape": "reports/q_production_20261005/q_supervisor.py:447-449 (e4_q.job_spec)", "worker": "studies/q_production_20261005/q_worker.py",
        "eng_store_junction": junction,
        "scheduling": {"max_workers": int(workers), "lightgbm_at_once": 1, "order": "LightGBM of both runs first, then PT_flex",
                       "env_threads": eq.THREADS, "lightgbm_num_threads": "3 (qtrain.fit_lightgbm, frozen)",
                       "gpu": "none: device cpu, CUDA_VISIBLE_DEVICES=-1", "retries": eq.RETRIES, "max_interrupts": eq.MAX_INTERRUPTS,
                       "resume": "rerun this stage: completed jobs are skipped, live workers adopted, interrupted ones relaunched"},
        "wall": {arm: eq._ledger_wall(paths.jobs(arm)) for arm in ARMS},
        "ledgers": {arm: str(paths.jobs(arm) / eq.LEDGER_NAME) for arm in ARMS},
        **_common(paths), "timing": vs._stage("train", started, tick), "utc": utc(),
        "run": run_record(rows=counts, matches=None, inputs=[paths.rows(arm) for arm in ARMS], code=[ROOT / name for name in TOG_CODE],
                          seed=None, started_utc=started),
        "seed_note": "no seed of its own: the frozen q worker's training seed rules apply (qtrain)"}
    write_once(target, _clean(record))
    print(json.dumps({"done": str(target), "q_dev_raw_match_brier": {f"{r['arm']}:{r['card_id']}": r["q_dev_raw_match_brier"]
                                                                       for r in job_records}}), flush=True)
    return 0


# ----------------------------------------------------------------------------- select (PRESPEC_E1 section 3 모델)
def select_stage(paths: Paths = Paths()) -> int:
    """qselect.select_within per run and family on Q_DEV (e4_q.select_cards); E1_SELECTION.json."""

    started, tick = utc(), time.perf_counter()
    target = paths.out / SELECTION_NAME
    if target.exists():
        print(json.dumps({"done": str(target)}), flush=True)
        return 0
    train = _passed(paths.out / TRAIN_NAME, "train")
    jobs = eq.plan_jobs(qjobs.plan())
    selection, files, problems = {}, {}, []
    for arm in ARMS:
        results = {family: {} for family in FAMILIES}
        orders = {family: [] for family in FAMILIES}
        key_of = {family: {} for family in FAMILIES}
        for job in jobs:
            key = qtest.key_of(job.job_id)
            path = paths.jobs(arm) / key / "RESULT.json"
            results[job.family][job.card_id] = _read(path)
            orders[job.family].append(job.card_id)
            key_of[job.family][job.card_id] = key
            files[f"{arm}/jobs/{key}/RESULT.json"] = _entry(path)
        chosen, found = eq.select_cards(results, orders)
        problems += [f"{arm}: {problem}" for problem in found]
        selection[arm] = {family: {**chosen[family], "selected_key": key_of[family].get(chosen[family]["selected"]),
                                   "card_order": orders[family], "jobs": len(orders[family])} for family in FAMILIES}
    if problems:
        return _blocked(paths.out, "select", problems, selection=selection)
    record = {"version": VERSION, "experiment": "E1", "stage": "select", "passed": True, "files": files, "selection": selection,
              "rule": "qselect.select_within per run and family: the lowest Q_DEV raw match-weighted Brier (RESULT.json "
                      "scores.q_dev.raw_match_brier) among completed cards, ties by the frozen plan's card order; every card "
                      "completed (PRESPEC_E1 section 3)", "q_dev_keys_sha256": train["q_dev_keys_sha256"], **_common(paths),
              "timing": vs._stage("select", started, tick), "utc": utc()}
    write_once(target, _clean(record))
    print(json.dumps({"done": str(target), "selected": {arm: {family: [selection[arm][family]["selected"], selection[arm][family]["brier"]]
                                                                 for family in FAMILIES} for arm in ARMS}}), flush=True)
    return 0


def selected_cells(paths: Paths, selection: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """qtest.cell_record of the four selected cells (completed, and the job the selection names)."""

    jobs = {job.card_id: job for job in eq.plan_jobs(qjobs.plan())}
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        cells[arm] = {}
        for family in FAMILIES:
            chosen = selection[arm][family]
            job = jobs[chosen["selected"]]
            cell = qtest.cell_record(paths.run(arm), job.to_dict())
            if job.family != family or cell["status"] != "completed" or cell["key"] != chosen["selected_key"]:
                raise Blocked(f"{arm} {family}: the selected cell is not a completed job of the selection")
            cells[arm][family] = cell
    return cells


# ----------------------------------------------------------------------------- the Q_CAL rehearsal (section 6 step 5, Ruling 55)
def rehearse_scoring(cell: Mapping[str, Any], *, role: str = "Q_CAL", device: str = "cpu") -> dict[str, Any]:
    """The TEST scoring path of e4_q.score_cell on the role's rows: qtest.LoadedCell (the row table's hash equals the
    spec's), the first RELOAD_ROWS stored Q_DEV predictions reproduced, every TEST row of the table sealed (y = -1),
    then the role's rows scored, which must equal the worker's stored Q_CAL predictions (RELOAD_ATOL)."""

    code = ROLE_CODES[role]
    loaded = qtest.LoadedCell(cell, device=device)
    source = Path(cell["source_dir"])
    stored = qtest.load_npz(source / "pred_q_dev.npz")
    check_rows = np.asarray(stored["row"], dtype=np.int64)[:qtest.RELOAD_ROWS]
    again = loaded.logits(check_rows)
    diff = float(np.abs(again - np.asarray(stored["logit"])[:qtest.RELOAD_ROWS]).max()) if check_rows.size else 0.0
    if diff > qtest.RELOAD_ATOL[loaded.kind]:
        raise Blocked(f"{cell['key']}: re-loaded model differs from its stored Q_DEV predictions by {diff}")
    sealed_rows = np.flatnonzero(np.asarray(loaded.rows["role"]) == TEST)
    if (np.asarray(loaded.rows["y"])[sealed_rows] != -1).any():
        raise Blocked(f"{cell['key']}: TEST labels are not sealed in the row table")
    rows = np.flatnonzero(np.asarray(loaded.rows["role"]) == code)
    logits = np.asarray(loaded.logits(rows), dtype=np.float64)
    reference = qtest.load_npz(source / f"pred_{role.lower()}.npz")
    if not np.array_equal(np.asarray(reference["row"], dtype=np.int64), rows):
        raise Blocked(f"{cell['key']}: the stored {role} predictions are not the table's {role} rows in order")
    stored_diff = float(np.abs(logits - np.asarray(reference["logit"], dtype=np.float64)).max()) if rows.size else 0.0
    if stored_diff > qtest.RELOAD_ATOL[loaded.kind]:
        raise Blocked(f"{cell['key']}: the re-loaded model's {role} logits differ from the stored ones by {stored_diff}")
    return {"row": rows, "engagement_key": np.asarray(loaded.rows["engagement_key"])[rows], "logit": logits,
            "p": qtest.sigmoid(logits), "reload_max_abs_logit_diff": diff, "reload_rows": int(check_rows.size), "kind": loaded.kind,
            "stored_q_cal_max_abs_logit_diff": stored_diff}


def load_scores(directory: Path, cell: Mapping[str, Any]) -> dict[str, Any]:
    """Saved scores of a cell (checked against their meta's SHA-256)."""

    npz, meta_path = Path(directory) / f"{cell['key']}.npz", Path(directory) / f"{cell['key']}.json"
    meta = _read(meta_path)
    if sha256_file(npz) != meta.get("pred_sha256"):
        raise Blocked(f"{npz} differs from {meta_path.name}")
    return {**eq._load_npz(npz), "meta": meta}


def write_scores(directory: Path, cell: Mapping[str, Any], scored: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    """Scores of one cell in <directory>/<key>.npz (row, engagement_key, logit, p) and <key>.json, written once."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"row": np.asarray(scored["row"], dtype=np.int64), "engagement_key": np.asarray(scored["engagement_key"]),
              "logit": np.asarray(scored["logit"], dtype=np.float64), "p": np.asarray(scored["p"], dtype=np.float64)}
    vs._check_probabilities(arrays["p"], f"{cell['key']} {split} p")
    digest = eq.write_bytes_once(directory / f"{cell['key']}.npz", eq.npz_bytes(arrays))
    meta = {"key": cell["key"], "split": split, "rows": int(arrays["row"].size), "pred_sha256": digest, "kind": scored.get("kind"),
            "reload_max_abs_logit_diff": scored.get("reload_max_abs_logit_diff"), "reload_rows": scored.get("reload_rows"),
            "stored_q_cal_max_abs_logit_diff": scored.get("stored_q_cal_max_abs_logit_diff"), "loader": "qtest.LoadedCell (frozen)",
            "calibration": "none: raw probabilities (PRESPEC_E1 section 4)", "utc": utc()}
    write_once(directory / f"{cell['key']}.json", _clean(meta))
    return {**arrays, "meta": _read(directory / f"{cell['key']}.json")}


def _scored(directory: Path, cell: Mapping[str, Any], scorer: Callable[[Mapping[str, Any]], Mapping[str, Any]], *, split: str) -> dict[str, Any]:
    """Saved scores of a cell when present (a rerun before the opening reuses verified scores), else score and save."""

    if (Path(directory) / f"{cell['key']}.json").is_file():
        return load_scores(directory, cell)
    return write_scores(directory, cell, scorer(cell), split=split)


def write_standins(paths: Paths, role: str = "Q_CAL") -> tuple[dict[str, Path], dict[str, Path], dict[str, Any]]:
    """Non-TEST stand-ins for the inputs that exist only at TEST, in the sealed files' layouts (Ruling 55):
      plc_labels       the placebo table's role rows as engagement_key + rows.SEALED_COLUMNS (rows.seal_test layout);
      pred_end         the public PRED_PLACEBO_h90 role rows as e1_states.SEALED_COLUMNS;
      windows_summary  WINDOWS_h90.json's role block under the role's key (the sealed file holds TEST's under 'TEST');
      balance_summary  BALANCE.json's role block under roles.<role>;
      eng_labels       q_run_1's public role rows as engagement_key + rows.SEALED_COLUMNS (q_run_1/Q_TEST_SEALED_h90.npz layout);
      main:<family>    the role's predictions of the search job each q_run_1 main cell reuses, in the qtest.score layout
                       (row, engagement_key, logit, p and p_cal when q_run_1 has a fitted calibrator for the cell).
    Files are written once with fixed bytes under <out>/REHEARSAL_QCAL."""

    code = ROLE_CODES[role]
    base = paths.rehearsal
    base.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}

    def put(label: str, name: str, data: bytes, source: str) -> Path:
        path = base / name
        files[label] = {"path": str(path), "sha256": eq.write_bytes_once(path, data), "source": source}
        return path

    plc = eq._load_npz(paths.rows(PLC))
    own = plc["role"].astype(np.int64) == code
    opened = {"plc_labels": put("plc_labels", "plc_labels.npz", eq.npz_bytes(
        {"engagement_key": plc["engagement_key"][own], **{name: plc[name][own] for name in SEALED_COLUMNS}}),
        f"{paths.rows(PLC)} rows of {role}")}
    pred = eq._load_npz(paths.pred)
    own = pred["role"].astype(np.int64) == code
    if not np.isfinite(pred["p_end"][own]).all():
        raise Blocked(f"the public placebo end V of {role} is not finite")
    end_is_start = es.end_is_a_start(pred["start_state"], pred["end_state"])
    columns = {"window_id": pred["window_id"][own], "engagement_key": pred["engagement_key"][own], "ctx": pred["ctx"][own],
               "end_state": pred["end_state"][own], "logit_end": pred["logit_end"][own], "raw_end": pred["raw_end"][own],
               "p_end": pred["p_end"][own], "end_equals_a_start": end_is_start[own]}
    opened["pred_end"] = put("pred_end", "pred_end.npz", eq.npz_bytes({name: columns[name] for name in es.SEALED_COLUMNS}),
                             f"{paths.pred} rows of {role}")
    windows_record = _read(paths.windows_record)
    opened["windows_summary"] = put("windows_summary", "windows_summary.json", _json_bytes({
        "version": windows_record.get("version"), "experiment": "E1", "stage": f"windows_{role}_standin", "smoke": False,
        "note": f"rehearsal stand-in in the layout of WINDOWS_h90_TEST_SEALED.json: the public {role} summary of WINDOWS_h90.json",
        "windows_npz_sha256": (windows_record.get("windows_npz") or {}).get("sha256"), role: windows_record["roles"][role]}),
        f"{paths.windows_record} roles.{role}")
    balance = _read(paths.balance)
    opened["balance_summary"] = put("balance_summary", "balance_summary.json", _json_bytes({
        "version": balance.get("version"), "experiment": "E1", "stage": f"balance_{role}_standin",
        "note": f"rehearsal stand-in in the layout of BALANCE_TEST_SEALED.json: the public {role} block of BALANCE.json",
        "roles": {role: balance["roles"][role]}}), f"{paths.balance} roles.{role}")
    q = eq._load_npz(paths.q_rows)
    q_role = q["role"].astype(np.int64)
    own = q_role == code
    opened["eng_labels"] = put("eng_labels", "eng_labels.npz", eq.npz_bytes(
        {"engagement_key": q["engagement_key"][own], **{name: q[name][own] for name in SEALED_COLUMNS}}), f"{paths.q_rows} rows of {role}")
    main = {}
    for family, (final, search) in MAIN_CELLS.items():
        reuse = _read(paths.q_run / "jobs" / final / "REUSE.json")
        if reuse.get("reuse_of_key") != search:
            raise Blocked(f"q_run_1 {final}: REUSE.json does not point to {search}")
        stored = eq._load_npz(paths.q_run / "jobs" / search / f"pred_{role.lower()}.npz")
        rows = np.asarray(stored["row"], dtype=np.int64)
        if not np.array_equal(rows, np.flatnonzero(own)):
            raise Blocked(f"q_run_1 {search}: the stored {role} predictions are not q_run_1's {role} rows in order")
        arrays = {"row": rows, "engagement_key": q["engagement_key"][rows], "logit": np.asarray(stored["logit"], dtype=np.float64),
                  "p": np.asarray(stored["p"], dtype=np.float64)}
        calibrator_path = paths.q_run / "calibrators" / f"{final}.json"
        if calibrator_path.is_file():
            from studies.q_production_20261005 import qcalibrate

            calibrator = qcalibrate.QCalibrator.from_dict(_read(calibrator_path)["calibrator"])
            if calibrator.fitted:
                arrays["p_cal"] = np.asarray(calibrator.predict(arrays["p"]), dtype=np.float64)
        main[family] = put(f"main:{family}", f"main_{family}.npz", eq.npz_bytes(arrays),
                           f"q_run_1 {search}/pred_{role.lower()}.npz (the search job {final} reuses)")
    return opened, main, files


def qcal_stage(paths: Paths = Paths(), *, scorer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None) -> int:
    """The Q_CAL development figures through the TEST step's evaluation path with stand-ins (module docstring)."""

    started, tick = utc(), time.perf_counter()
    target = paths.out / QCAL_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; it is written once")
    selection_record = _passed(paths.out / SELECTION_NAME, "select")
    _passed(paths.source / "E1_ROWS.json", "rows")
    cells = selected_cells(paths, selection_record["selection"])
    scorer = scorer or rehearse_scoring
    parts: dict[str, Any] = {}
    tick_part = time.perf_counter()
    scores = {arm: {family: _scored(paths.rehearsal / "SCORES" / arm, cells[arm][family], scorer, split="Q_CAL")
                    for family in FAMILIES} for arm in ARMS}
    parts["scoring"] = eq._part(tick_part)
    tick_part = time.perf_counter()
    opened, main, standins = write_standins(paths, "Q_CAL")
    parts["stand_ins"] = eq._part(tick_part)
    tick_part = time.perf_counter()
    result = evaluate_opened(paths, "Q_CAL", opened, scores, main)
    parts["evaluate"] = eq._part(tick_part)

    # rehearsal checks: the joins reproduce the public tables and each cell's Brier its worker's --------------------------
    eng = eq._load_fields(paths.rows(ENG), ("engagement_key", "role", "y", "label_svi_delta"))
    plc = eq._load_fields(paths.rows(PLC), ("engagement_key", "role", *SEALED_COLUMNS))
    joined = {}
    for arm, table, label, columns in ((ENG, eng, "eng_labels", ("y", "label_svi_delta")), (PLC, plc, "plc_labels", SEALED_COLUMNS)):
        own = table["role"].astype(np.int64) == Q_CAL
        standin = eq._load_fields(opened[label], ("engagement_key", *columns))
        at = eq.align_keys(table["engagement_key"][own], standin["engagement_key"])
        joined[arm] = all(np.array_equal(np.asarray(standin[name])[at], table[name][own], equal_nan=False) for name in columns)
    brier_diff = {}
    for arm in ARMS:
        for family in FAMILIES:
            worker = _read(Path(cells[arm][family]["source_dir"]) / "RESULT.json")["scores"]["q_cal"]["raw_match_brier"]
            brier_diff[f"{arm}:{family}"] = abs(result["cells"][arm][family] - float(worker))
    checks = {"joined_engagement_labels_equal_the_eng_table": joined[ENG], "joined_placebo_labels_equal_the_plc_table": joined[PLC],
              "state_parts_follow_their_shard_groups": parts_follow_shard_groups(paths),
              "brier_abs_diff_to_worker": brier_diff,
              "scores_reproduce_the_stored_q_cal_predictions": {f"{arm}:{family}": scores[arm][family]["meta"].get("stored_q_cal_max_abs_logit_diff")
                                                                for arm in ARMS for family in FAMILIES}}
    problems = [name for name in ("joined_engagement_labels_equal_the_eng_table", "joined_placebo_labels_equal_the_plc_table",
                                  "state_parts_follow_their_shard_groups") if not checks[name]]
    problems += [f"{cell}: Q_CAL Brier differs from its worker's by {diff}" for cell, diff in brier_diff.items() if diff > RECOMPUTE_ATOL]
    if problems:
        raise Blocked(f"the rehearsal failed: {problems}")
    files = {f"rehearsal:{Path(entry['path']).relative_to(paths.rehearsal).as_posix()}": {"path": entry["path"], "sha256": entry["sha256"]}
             for entry in standins.values()}
    for arm in ARMS:
        for family in FAMILIES:
            key = cells[arm][family]["key"]
            for name in (f"{key}.npz", f"{key}.json"):
                path = paths.rehearsal / "SCORES" / arm / name
                files[f"rehearsal:SCORES/{arm}/{name}"] = _entry(path)
            for name in ("RESULT.json", "pred_q_cal.npz", "pred_q_dev.npz", "SPEC.json"):
                files[f"{arm}/jobs/{key}/{name}"] = _entry(Path(cells[arm][family]["source_dir"]) / name)
    record = {
        "version": VERSION, "experiment": "E1", "stage": "qcal", "passed": True, "split": "Q_CAL", "policy": QCAL_POLICY,
        "sign": SIGN_TEXT, "files": files,
        "cells": {arm: {family: {"key": cells[arm][family]["key"], "card": selection_record["selection"][arm][family]["selected"],
                                 "q_dev_brier": selection_record["selection"][arm][family]["brier"]} for family in FAMILIES} for arm in ARMS},
        "result": result,
        "rehearsal": {
            "passed": True, "checks": checks, "stand_ins": standins,
            "scores": {arm: {family: scores[arm][family]["meta"] for family in FAMILIES} for arm in ARMS},
            "exercised": ["qtest.LoadedCell of each selected cell: the row-table hash, the Q_DEV reload check and the sealed-row "
                          "check of e4_q.score_cell, then the role's rows scored (Q_CAL in place of TEST) and saved by the "
                          "TEST step's score writer",
                          "read_sealed of five files in the sealed layouts and the two main-model files in the qtest.score layout",
                          "the key joins (e4_q.align_keys) of the placebo labels, the placebo end V and q_run_1's labels",
                          "the placebo-label check against W = V(s + D) - V(s) and the direction-rule check of the engagement labels",
                          "the sealed-summary checks and triggers", "the objective flags (projection OBJECTIVE_COUNTS, STATES parts "
                          "of the role's shard group by e1_rows.part_index)",
                          "the four arrays, the joint bootstrap, decide(), every sensitivity (stratified and de-duplicated r "
                          "regardless of their triggers) and every auxiliary (AUC, G_eng_full, G_plc / 0.01386, matched vs "
                          "unmatched)"],
            "differs_at_test": ["the role (TEST) and its rows", "the state parts read are the test_* shard group's",
                                "the five files are the sealed ones and the main scores q_run_1's TEST_SCORES cells",
                                "scoring through e4_q.score_cell (the same loader and checks, on TEST rows)",
                                "the OPENINGS.json append before the first sealed read (tested on synthetic data)"],
            "not_called": "qtest.evaluate, qeval.kill_count_strata and qeval.credit_strata (Ruling 55)"},
        "stages": parts, **_common(paths), "timing": vs._stage("qcal", started, tick), "utc": utc(),
        "run": run_record(rows=result["rows"], matches=result["matches"], inputs=[paths.rows(arm) for arm in ARMS] + [paths.pred, paths.q_rows],
                          code=[ROOT / name for name in TOG_CODE], seed=SEED, started_utc=started)}
    write_once(target, _clean(record))
    primary = result["primary"]
    print(json.dumps({"done": str(target), "split": "Q_CAL", **{name: [primary["point"][name], primary[f"{name}_ci"]]
                                                                for name in QUANTITIES},
                      "draws_G_eng_le_0": primary["draws_G_eng_le_0"], "decide": result["decision"]["outcome"]}), flush=True)
    return 0


# ----------------------------------------------------------------------------- lock (PRESPEC_E1 section 6 잠금 파일)
def core_files(paths: Paths) -> dict[str, Path]:
    """The files the TEST step reads, by lock label (the lock binds them and the step reads exactly these paths)."""

    files = {"prespec": paths.prespec, "amendment": paths.amendment, "release": paths.release,
             "plc:Q_ROWS_h90.npz": paths.rows(PLC), "plc:Q_ROWS_h90.json": paths.rows_record(PLC), "plc:PLC_KEY_MAP_h90.npz": paths.key_map,
             "eng:Q_ROWS_h90.npz": paths.rows(ENG), "eng:Q_ROWS_h90.json": paths.rows_record(ENG),
             "windows:WINDOWS_h90.npz": paths.windows, "windows:WINDOWS_h90.json": paths.windows_record,
             "pred:PRED_PLACEBO_h90.npz": paths.pred, "pred:PRED_PLACEBO_h90.json": paths.pred_record,
             "states:STATES_h90.json": paths.states_record, "balance:BALANCE.json": paths.balance,
             "q_run_1:Q_ROWS_h90.npz": paths.q_rows, "labels_1:ENGAGEMENTS.npz": paths.engagements,
             "projection:OBJECTIVE_COUNTS.npy": paths.objective_counts}
    files.update({SEALED_LOCK_LABEL[label]: path for label, path in sealed_paths(paths).items()})
    for family in FAMILIES:
        files[f"main:{MAIN_CELLS[family][0]}.npz"] = paths.main_scores(family)
        files[f"main:{MAIN_CELLS[family][0]}.json"] = paths.main_meta(family)
    return files


def cell_label(arm: str, key: str, name: str) -> str:
    return f"job:{arm}:{key}:{name}"


def sealed_summaries(root: Path) -> dict[str, dict[str, str]]:
    """Ruling 58: the frozen-format files under ``root`` with a TEST-inclusive count, found by path pattern and hashed as
    bytes (never opened)."""

    found: dict[str, dict[str, str]] = {}
    for pattern, why in SEALED_SUMMARY_PATTERNS:
        for path in sorted(Path(root).glob(pattern)):
            if path.is_file():
                found[path.relative_to(root).as_posix()] = {"path": str(path), "sha256": sha256_file(path), "pattern": pattern, "why": why}
    return found


def sealed_summary_gaps(root: Path) -> list[str]:
    """Patterns of SEALED_SUMMARY_PATTERNS that match no file (the lock stops on any)."""

    return [pattern for pattern, _ in SEALED_SUMMARY_PATTERNS if not any(path.is_file() for path in Path(root).glob(pattern))]


def public_counts(counts: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The TRAIN/Q_DEV/Q_CAL counts per arm (E1_ROWS.json counts); any other role is BLOCKED (no TEST count in the lock)."""

    out = {}
    for arm, roles in counts.items():
        extra = sorted(set(roles) - set(es.PUBLIC_ROLES))
        if extra:
            raise Blocked(f"{arm}: the counts hold non-public roles {extra}")
        out[arm] = dict(roles)
    return out


def rules() -> dict[str, Any]:
    """The rules of the evaluation and the TEST step; the lock binds them and the step refuses others."""

    return {
        "sign": SIGN_TEXT,
        "arms": {ENG: "model (나): the 16 cards trained on the matched TRAIN engagements, selected on the matched Q_DEV engagements",
                 PLC: "model (가): the 16 cards trained on the placebo TRAIN windows, selected on the placebo Q_DEV windows"},
        "four_arrays": "per-match mean raw-probability Brier of eng LightGBM, eng PT_flex, plc LightGBM and plc PT_flex in one "
                       "ascending match order (qeval.per_match_loss, qeval.align_cells); placebo row i and engagement row i are "
                       "the same matched engagement, so both arms cover the same matches",
        "quantities": {"G_eng_size": "Brier(PT_flex) - Brier(LightGBM), model (나) on the matched engagements",
                       "G_plc": "the same quantity, model (가) on the placebo windows", "L1": "G_plc - 0.25 G_eng_size",
                       "L2": "G_plc - 0.75 G_eng_size", "net": "G_eng_size - G_plc", "r": "G_plc / G_eng_size (descriptive only)"},
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "level": LEVEL, "quantiles": list(qeval.bootstrap_quantiles(LEVEL)),
                      "generator": "qeval.bootstrap_indices: numpy.random.Generator(numpy.random.PCG64(seed)).integers(0, M, size=M)",
                      "unit": "match; every quantity from the same draws (qeval.paired_match_bootstrap over the four cells)",
                      "interval": "percentile; numpy.quantile method='linear'",
                      "nonpositive_G_eng": "draws with G_eng_size <= 0 are counted and kept"},
        "decision": {"split": "TEST only", "rule": DECISION_RULE, "outcomes": DECISIONS, "default_handling": DEFAULT_HANDLING,
                     "net_note": NET_NOTE},
        "sensitivities": {"r_0.01": f"|engagement SVI| >= {R001} and |placebo W| >= {R001}; matches with rows in all four arrays",
                          "objective_free": "pairs where neither window includes an objective (diagnostics.py:152-158)",
                          "stratified_r": "V(tau) quintiles of the role's matched engagements; computed always, reported with "
                                          "its trigger (|SMD| > 0.1 on V(start) or game time; TEST trigger from "
                                          "BALANCE_TEST_SEALED.json)",
                          "deduplicated_r": "one pair per reused window (smallest selection hash); computed always, reported "
                                            "with its trigger (reused windows > 5% of the placebo rows)",
                          "report_only": True},
        "auxiliary": {"auc": "match-weighted AUC of both models in both arms and the difference LightGBM - PT_flex (e2_run.evaluate_pair)",
                      "G_eng_full": "q_run_1's main models on the matched engagements: Brier(PT_flex) - Brier(LightGBM)",
                      "G_plc_over_main_test_gain": f"G_plc / {MAIN_TEST_GAIN}",
                      "main_matched_vs_unmatched": "the main-model gain on matched and on unmatched engagements of the role",
                      "main_cells": {family: {"final": final, "reuses": search} for family, (final, search) in MAIN_CELLS.items()}},
        "scoring": {"function": "e4_q.score_cell (qtest.LoadedCell; Q_DEV reload of the first RELOAD_ROWS rows; TEST y = -1)",
                    "reload_rows": qtest.RELOAD_ROWS, "reload_atol": dict(qtest.RELOAD_ATOL), "probabilities": "raw p, no calibration",
                    "device": "cpu"},
        "sealed_files": {label: SEALED_LOCK_LABEL[label] for label in SEALED_LABELS},
        "checks_after_opening": ["the placebo labels equal W = V(s + D) - V(s) of the placebo end V",
                                 "the engagement labels follow y = 1[SVI > 0]", "the sealed summaries describe the role's pairs"],
        "order": ["lock re-check (every bound file, rules, environment, release, the reviewed lock SHA-256)",
                  "both row tables seal TEST (y = -1, NaN, -1, 0)", "e4_q.score_cell for the four cells, scores in <arm>/TEST_SCORES",
                  "E1_OPENING.json, then OPENINGS.json", "the five sealed files read once", "evaluate_opened", "E1_REPORT.json"],
        "not_called": "qtest.evaluate, qeval.kill_count_strata and qeval.credit_strata (Ruling 55)",
    }


def loaded_modules() -> list[str]:
    """Workspace modules loaded in this process that CODE_FILES does not list (bound as module:<path>)."""

    return sorted(name for name in qrelease.workspace_files(list(sys.modules.values()))
                  if name.startswith(("studies/", "reports/")) and name not in CODE_FILES)


def bound_files(paths: Paths, records: Mapping[str, Mapping[str, Any]]) -> dict[str, Path]:
    """Every file the lock binds by SHA-256 (EXPOSURE_KO.md and OPENINGS.json are appended to, so recorded, not bound)."""

    files: dict[str, Path] = dict(core_files(paths))
    files.update({f"code:{name}": ROOT / name for name in CODE_FILES})
    files.update({f"module:{name}": ROOT / name for name in loaded_modules()})
    files.update({f"task13:{name}": paths.source / name for name in TASK13_FILES})
    states = _read(paths.states_record)
    files.update({f"states:{entry['file']}": paths.states_dir / entry["file"] for entry in states["parts"].values()})
    files["task14:V_REPRO_h90.npz"] = paths.source / "V_REPRO_h90.npz"
    files.update({f"task15:{name}": paths.source / name for name in TASK15_RECORDS})
    for arm in ARMS:
        run = paths.run(arm)
        for path in sorted(paths.inputs(arm).rglob("*")):
            if path.is_file():
                files[f"{arm}:inputs/{VARIANT}/{path.relative_to(paths.inputs(arm)).as_posix()}"] = path
        for path in sorted((run / "logs").glob("*")):
            if path.is_file():
                files[f"{arm}:logs/{path.name}"] = path
        for path in sorted((run / "raw" / "store").iterdir()):
            if path.is_file():
                files[f"{arm}:raw/store/{path.name}"] = path
    for rel in sealed_summaries(paths.source):
        files[f"sealed_summary:{rel}"] = paths.source / rel
    q = paths.q_run
    files.update({"q_run_1:raw/store/BUILD.json": q / "raw" / "store" / "BUILD.json", "q_run_1:inputs/main_h90/XY.npy": q / "inputs" / VARIANT / "XY.npy",
                  "q_run_1:inputs/main_h90/PREPARED.json": q / "inputs" / VARIANT / "PREPARED.json",
                  "q_run_1:SCORE_LOCK.json": q / "SCORE_LOCK.json", "q_run_1:TEST_REPORT.json": q / "TEST_REPORT.json"})
    for family, (final, search) in MAIN_CELLS.items():
        files[f"q_run_1:jobs/{final}/REUSE.json"] = q / "jobs" / final / "REUSE.json"
        files[f"q_run_1:jobs/{search}/pred_q_cal.npz"] = q / "jobs" / search / "pred_q_cal.npz"
        if (q / "calibrators" / f"{final}.json").is_file():
            files[f"q_run_1:calibrators/{final}.json"] = q / "calibrators" / f"{final}.json"
    files.update({"labels_1:LABEL_LOCK.json": paths.labels / "LABEL_LOCK.json",
                  "labels_1:DIAGNOSTICS_LAYER2.json": paths.labels / "DIAGNOSTICS_LAYER2.json"})
    for record in records["train"]["jobs"]:
        for name in record["files"]:
            files[cell_label(record["arm"], record["key"], name)] = paths.jobs(record["arm"]) / record["key"] / name
    files.update({f"e1:{name}": paths.out / name for name in (TRAIN_NAME, SELECTION_NAME, QCAL_NAME)})
    for path in sorted(paths.rehearsal.rglob("*")):
        if path.is_file():
            files[f"rehearsal:{path.relative_to(paths.rehearsal).as_posix()}"] = path
    return files


def _tree_sha(verify: Mapping[str, Mapping[str, str]], prefix: str) -> str:
    """e1_rows.tree_digest over the bound files under ``prefix`` (sorted '<file>:<sha256>' lines; no file size)."""

    entries = {label[len(prefix):]: {"sha256": entry["sha256"]} for label, entry in verify.items() if label.startswith(prefix)}
    return er1.tree_digest(entries)


def lock_cross_checks(paths: Paths, verify: Mapping[str, Mapping[str, str]], records: Mapping[str, Mapping[str, Any]],
                      summaries: Mapping[str, Mapping[str, str]]) -> dict[str, bool]:
    """The bound hashes against the Task 13-16 records, q_run_1's records and the hashes PRESPEC_E1 prints."""

    def sha(label: str) -> str | None:
        return (verify.get(label) or {}).get("sha256")

    rows, prepare, store = records["E1_ROWS.json"], records["E1_PREPARE.json"], records["E1_STORE.json"]
    eng_inputs, balance = records["E1_ENG_INPUTS.json"], records["BALANCE.json"]
    windows, pred, states = _read(paths.windows_record), _read(paths.pred_record), _read(paths.states_record)
    score_lock = (_read(paths.q_run / "SCORE_LOCK.json") or {}).get("files") or {}
    q_prepared = _read(paths.q_run / "inputs" / VARIANT / "PREPARED.json")
    listing = "".join(sorted(f"{entry['file']}:{sha('states:' + entry['file'])}\n" for entry in states["parts"].values()))
    code_records = {name: records[name].get("code_sha256") or {} for name in ("E1_ROWS.json", "BALANCE.json", "E1_ROWS_RUN.json")}
    code_records.update({"WINDOWS_h90.json": windows.get("code_sha256") or {}, "PRED_PLACEBO_h90.json": pred.get("code_sha256") or {},
                         "STATES_h90.json": states.get("code_sha256") or {}})
    return {
        "row_tables_and_seal_equal_the_rows_record": all(
            sha(label) == rows["files"][name]["sha256"] for label, name in (
                ("plc:Q_ROWS_h90.npz", "plc/Q_ROWS_h90.npz"), ("eng:Q_ROWS_h90.npz", "eng/Q_ROWS_h90.npz"),
                ("plc:Q_ROWS_h90.json", "plc/Q_ROWS_h90.json"), ("eng:Q_ROWS_h90.json", "eng/Q_ROWS_h90.json"),
                ("plc:PLC_KEY_MAP_h90.npz", "plc/PLC_KEY_MAP_h90.npz"), ("sealed:plc/Q_TEST_SEALED_h90.npz", "plc/Q_TEST_SEALED_h90.npz"))),
        "inputs_equal_the_prepare_record": all(sha(name.replace("/", ":", 1)) == entry["sha256"] for name, entry in prepare["files"].items()),
        "xy_equal_the_store_and_eng_records": (sha("plc:inputs/main_h90/XY.npy") == store["files"]["plc/inputs/main_h90/XY.npy"]["sha256"]
                                               and sha("eng:inputs/main_h90/XY.npy") == eng_inputs["files"]["eng/inputs/main_h90/XY.npy"]["sha256"]),
        "plc_store_equals_the_store_record": _tree_sha(verify, "plc:raw/store/") == store["store_tree_sha256"],
        "eng_store_is_q_run_1s_unchanged": (_tree_sha(verify, "eng:raw/store/") == prepare["q_run_1_untouched"]["store_tree_sha256_after"]
                                            and sha("eng:raw/store/BUILD.json") == q_prepared.get("store_build_sha256")
                                            == sha("q_run_1:raw/store/BUILD.json")),
        "windows_equal_the_windows_record": (sha("windows:WINDOWS_h90.npz") == windows["windows_npz"]["sha256"]
                                             and sha("sealed:WINDOWS_h90_TEST_SEALED.json") == windows["test_sealed_summary"]["sha256"]),
        "gate_r1_equals_the_windows_record": (sha("task13:WINDOWS_h90_QDEV_GATE_r1.json") == windows["gate_record"]["sha256"]
                                              and sha("task13:WINDOWS_h90_QDEV_GATE_r1.npz") == windows["gate_record"]["npz_sha256"]),
        "gate_r0_and_gold_equal_the_prespec": all(sha(label) == digest for label, digest in PRESPEC_CITED.items()),
        "pred_equal_the_pred_record": (sha("pred:PRED_PLACEBO_h90.npz") == pred["outputs"]["public"]["sha256"]
                                       and sha("sealed:PRED_PLACEBO_h90_TEST_SEALED.npz") == pred["outputs"]["sealed"]["sha256"]
                                       and sha("states:STATES_h90.json") == pred["states"]["record_sha256"]),
        "state_parts_equal_the_states_record": (all(sha(f"states:{entry['file']}") == entry["sha256"] for entry in states["parts"].values())
                                                and hashlib.sha256(listing.encode("utf-8")).hexdigest() == states["parts_list_sha256"]),
        "balance_seal_equals_the_balance_record": sha("sealed:BALANCE_TEST_SEALED.json") == balance["files"]["BALANCE_TEST_SEALED.json"]["sha256"],
        "q_run_1_rows_and_seal_equal_its_score_lock": (sha("sealed:q_run_1/Q_TEST_SEALED_h90.npz") == score_lock.get("Q_TEST_SEALED_h90.npz")
                                                       and sha("q_run_1:Q_ROWS_h90.npz") == score_lock.get("Q_ROWS_h90.npz")),
        "main_scores_equal_their_meta": all(sha(f"main:{MAIN_CELLS[family][0]}.npz") == _read(paths.main_meta(family)).get("pred_sha256")
                                            for family in FAMILIES),
        "job_files_equal_the_train_record": all(sha(cell_label(r["arm"], r["key"], name)) == digest for r in records["train"]["jobs"]
                                                for name, digest in r["files"].items()),
        "qcal_rehearsal_passed": records["qcal"]["rehearsal"]["passed"] is True,
        "sealed_summaries_bound": bool(summaries) and all(sha(f"sealed_summary:{rel}") == entry["sha256"] for rel, entry in summaries.items()),
        "task13_15_code_unchanged_since_their_runs": all(sha(f"code:{name}") == digest for record in code_records.values()
                                                         for name, digest in record.items() if name in E1_MODULES),
    }


def lock_stage(paths: Paths = Paths(), *, require_committed_code: bool = True) -> int:
    """E1_LOCK.json (PRESPEC_E1 section 6 잠금 파일, Rulings 28, 50 and 58).  Sealed files and sealed summaries are hashed
    as bytes only."""

    started, tick = utc(), time.perf_counter()
    target = paths.out / LOCK_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; it is written once")
    records = {"train": _passed(paths.out / TRAIN_NAME, "train"), "select": _passed(paths.out / SELECTION_NAME, "select"),
               "qcal": _passed(paths.out / QCAL_NAME, "qcal")}
    records.update({name: _passed(paths.source / name, name) for name in TASK15_RECORDS})
    prespec, amendment = document_record(paths.prespec), document_record(paths.amendment)
    approval = {"prespec": bool(prespec["sha256"] and prespec["commit"] == PRESPEC_COMMIT and prespec["unchanged_since_commit"]),
                "amendment": bool(amendment["sha256"] and amendment["commit"] == AMENDMENT_COMMIT and amendment["unchanged_since_commit"])}
    if not all(approval.values()):
        return _blocked(paths.out, "lock", {"prespec_or_amendment_hash_or_commit": {"prespec": prespec, "amendment": amendment}})
    code = {name: document_record(ROOT / name) for name in CODE_FILES}
    if require_committed_code and not all(code[name]["commit"] and code[name]["unchanged_since_commit"] for name in TOG_CODE
                                          if not name.endswith("__init__.py")):
        raise RuntimeError("commit the E1 code before writing the lock")
    release = qrelease.verify(paths.release)
    if release["digest"] != RELEASE_DIGEST:
        return _blocked(paths.out, "lock", {"release_digest": release["digest"]})
    gaps = sealed_summary_gaps(paths.source)
    if gaps:
        return _blocked(paths.out, "lock", {"sealed_summary_patterns_without_a_file": gaps})
    summaries = sealed_summaries(paths.source)
    junction = _eng_junction(paths)
    edges = np.asarray(_read(paths.source / "GOLD_EDGES_h90.json")["edges"], dtype=np.float64)
    if not np.array_equal(edges, ec.read_appendix_edges(paths.prespec.read_text(encoding="utf-8"))):
        return _blocked(paths.out, "lock", ["the gold edges differ from PRESPEC_E1 Appendix A (section 7)"])
    tick_hash = time.perf_counter()
    files = bound_files(paths, records)
    verify = {label: {"path": str(path), "sha256": sha256_file(path)} for label, path in files.items()}
    cross = lock_cross_checks(paths, verify, records, summaries)
    if not all(cross.values()):
        return _blocked(paths.out, "lock", {"cross_checks_failed": sorted(name for name, ok in cross.items() if not ok)})
    hashing = {"wall_seconds": time.perf_counter() - tick_hash, "files": len(verify)}
    cells = selected_cells(paths, records["select"]["selection"])
    for arm in ARMS:
        for family in FAMILIES:
            cell = cells[arm][family]
            for name, field in (("SPEC.json", "spec_sha256"), ("RESULT.json", "result_sha256"), (cell["model_file"], "model_sha256"),
                                ("pred_q_dev.npz", "pred_q_dev_sha256")):
                if verify[cell_label(arm, cell["key"], name)]["sha256"] != cell[field]:
                    raise Blocked(f"{arm} {family}: {name} differs between qtest.cell_record and the bound files")
    selection = records["select"]["selection"]
    qcal = records["qcal"]["result"]
    stage_times = {name: records[name].get("timing") for name in ("train", "select", "qcal")}
    stage_times["train_wall"] = records["train"].get("wall")
    payload = {
        "version": VERSION, "experiment": "E1", "locked_utc": utc(),
        "prespec": {**prespec, "approved_commit": PRESPEC_COMMIT}, "amendment": {**amendment, "approved_commit": AMENDMENT_COMMIT},
        "prespec_note": PRESPEC_NOTE, "exposure_ledger": document_record(paths.exposure), "code": code, "git_head": git_head(),
        "release": {"path": str(paths.release), "digest": release["digest"]},
        "selection": {arm: {family: {"selected": selection[arm][family]["selected"], "selected_key": selection[arm][family]["selected_key"],
                                     "q_dev_brier": selection[arm][family]["brier"],
                                     "cards": {card: entry["brier"] for card, entry in selection[arm][family]["cards"].items()}}
                            for family in FAMILIES} for arm in ARMS},
        "cells": cells,
        "main_cells": {family: {"final": final, "reuses": search, "npz": verify[f"main:{final}.npz"], "meta": verify[f"main:{final}.json"]}
                       for family, (final, search) in MAIN_CELLS.items()},
        "sealed_files": {label: verify[SEALED_LOCK_LABEL[label]] for label in SEALED_LABELS},
        "sealed_summaries": {"rule": SEALED_SUMMARY_RULE, "patterns": [{"pattern": p, "why": w} for p, w in SEALED_SUMMARY_PATTERNS],
                             "files": summaries},
        "gold_edges": {"values": edges.tolist(), "appendix_a_equal": True, "file": verify["task13:GOLD_EDGES_h90.json"],
                       "rows_file": verify["task13:GOLD_TRAIN_h90.npz"]},
        "eng_store_junction": junction,
        "row_counts": {"non_test": public_counts(records["E1_ROWS.json"]["counts"]),
                       "note": "TRAIN, Q_DEV and Q_CAL only (E1_ROWS.json); no TEST count is in this lock"},
        "qcal_headline": {"point": qcal["primary"]["point"], **{f"{name}_ci": qcal["primary"][f"{name}_ci"] for name in QUANTITIES},
                          "draws_G_eng_le_0": qcal["primary"]["draws_G_eng_le_0"], "decide_on_q_cal": qcal["decision"]["outcome"],
                          "policy": QCAL_POLICY},
        "rules": rules(), "environment": eq.environment(), "stages": stage_times,
        "label_policy": ("the five sealed files are hashed as file bytes only and opened once in the TEST step (Ruling 52); the "
                         "row-table record JSONs and the frozen-format files listed under sealed_summaries are sealed summaries "
                         "(Rulings 28 and 58); q_run_1's TEST_SCORES cells are hashed only and read in the TEST step"),
        "openings_before_lock": eq._openings(paths.openings),
        "cross_checks": cross, "verify": verify, "hashing": hashing,
        "run": run_record(rows=records["E1_ROWS.json"]["counts"], matches=None, inputs=[], code=[ROOT / name for name in CODE_FILES],
                          seed=SEED, started_utc=started),
    }
    payload["stages"]["lock"] = vs._stage("lock", started, tick)
    write_once(target, _clean(payload))
    print(json.dumps({"lock": str(target), "sha256": sha256_file(target), "files": len(verify), "prespec_commit": prespec["commit"],
                      "selected": {arm: {family: selection[arm][family]["selected"] for family in FAMILIES} for arm in ARMS}}), flush=True)
    return 0


# ----------------------------------------------------------------------------- part B: the single TEST step
def required_labels(lock: Mapping[str, Any]) -> list[str]:
    labels = [*core_files(Paths()), *(f"code:{name}" for name in CODE_FILES)]
    for arm in ARMS:
        for family in FAMILIES:
            cell = ((lock.get("cells") or {}).get(arm) or {}).get(family) or {}
            labels += [cell_label(arm, cell.get("key"), name) for name in ("SPEC.json", "RESULT.json", cell.get("model_file"), "pred_q_dev.npz")]
    return labels


def verify_lock(lock: Mapping[str, Any], paths: Paths = Paths()) -> dict[str, Any]:
    """Why the TEST step may not run (empty when the lock holds): a PRESPEC or AMENDMENT hash missing or of another commit
    (section 7), a missing entry, a bound path other than this run's, any bound file changed, other rules, another
    environment, cells that differ from the bound files, or a q release whose code changed."""

    problems: dict[str, Any] = {}
    verify = lock.get("verify") or {}
    for name, commit in (("prespec", PRESPEC_COMMIT), ("amendment", AMENDMENT_COMMIT)):
        record = lock.get(name) or {}
        if not (record.get("sha256") and record.get("commit")):
            problems[f"{name}_hash_missing"] = True
            continue
        if record["commit"] != commit:
            problems[f"{name}_not_the_approved_commit"] = record["commit"]
        if (verify.get(name) or {}).get("sha256") != record["sha256"]:
            problems[f"{name}_entry_differs"] = True
    missing = [label for label in required_labels(lock) if label not in verify]
    if missing:
        problems["entries_missing"] = missing
    moved = [label for label, path in core_files(paths).items()
             if label in verify and Path(verify[label]["path"]).resolve() != Path(path).resolve()]
    if moved:
        problems["paths_differ_from_this_run"] = moved
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
    now_env, locked_env = eq.environment(), lock.get("environment") or {}
    differs = {key: {"locked": locked_env.get(key), "now": now_env[key]} for key in eq.ENV_KEYS if locked_env.get(key) != now_env[key]}
    if differs:
        problems["environment_differs"] = differs
    for arm in ARMS:
        for family in FAMILIES:
            cell = ((lock.get("cells") or {}).get(arm) or {}).get(family) or {}
            for name, field in (("SPEC.json", "spec_sha256"), ("RESULT.json", "result_sha256"), (cell.get("model_file"), "model_sha256"),
                                ("pred_q_dev.npz", "pred_q_dev_sha256")):
                if not cell or (verify.get(cell_label(arm, cell.get("key"), name)) or {}).get("sha256") != cell.get(field):
                    problems.setdefault("cells_differ_from_bound_files", []).append(f"{arm}:{family}:{name}")
    try:
        release = qrelease.verify(Path((verify.get("release") or {}).get("path") or paths.release))
        if release["digest"] != RELEASE_DIGEST:
            problems["release_digest"] = release["digest"]
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        problems["release"] = str(error)
    return problems


def unsealed_test_rows(paths: Paths) -> list[str]:
    """Why the row tables do not seal TEST (empty when they do): both must hold the sealed placeholders on their TEST rows
    (y -1, label_svi_delta NaN, exact_zero -1, near_zero_bits 0) and their TEST rows must be the same pairs; nothing is
    counted."""

    problems = []
    tables = {arm: eq._load_fields(paths.rows(arm), ("role", "match_index", *eq.TEST_PLACEHOLDERS)) for arm in ARMS}
    test = {arm: tables[arm]["role"].astype(np.int64) == TEST for arm in ARMS}
    if not (np.array_equal(test[ENG], test[PLC]) and np.array_equal(tables[ENG]["match_index"][test[ENG]], tables[PLC]["match_index"][test[PLC]])):
        problems.append("the TEST rows of the two row tables are not the same pairs")
    for arm in ARMS:
        for name, value in eq.TEST_PLACEHOLDERS.items():
            if not eq._placeholder_ok(np.asarray(tables[arm][name])[test[arm]], value):
                problems.append(f"{arm}: {name} does not hold the sealed placeholder on TEST rows")
    return problems


def _failure(out: Path, entry: Mapping[str, Any], error: BaseException, parts: Mapping[str, Any]) -> None:
    """E1_FAILED_test_<n>.json: a failure after the opening leaves a record; no rerun under this lock (Ruling 31)."""

    path = eq._next_free(Path(out), "E1_FAILED_test")
    lines = traceback.format_exception(type(error), error, error.__traceback__)
    write_once(path, _clean({"version": VERSION, "experiment": "E1", "failed": True, "utc": utc(), "opening": dict(entry),
                             "error": {"type": type(error).__name__, "message": str(error)[:500],
                                       "traceback_tail": "".join(lines[-6:])[-3000:]},
                             "stages": dict(parts),
                             "rule": "do not re-run under this lock; the opening stays recorded; keep logs and partial state and "
                                     "report (Ruling 31)"}))
    print(json.dumps({"FAILED": "test", "file": str(path), "error": type(error).__name__}), flush=True)


def run_test_step(paths: Paths = Paths(), *, expected_lock_sha256: str,
                  scorer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None) -> int:
    """Part B (PRESPEC_E1 section 6 step 7): verify the lock, check that both row tables seal TEST, score the TEST rows of
    the four cells through the frozen qtest path (e4_q.score_cell), record the opening, read the five sealed files once,
    evaluate, and write E1_REPORT.json once."""

    scorer = scorer or eq.score_cell
    started, tick = utc(), time.perf_counter()
    out = paths.out
    for name in (REPORT_NAME, OPENING_NAME):
        if (out / name).exists():
            raise FileExistsError(f"{out / name} exists; the sealed TEST files are opened once under this lock")
    lock_path = out / LOCK_NAME
    if not lock_path.is_file():
        raise RuntimeError("no E1_LOCK.json: run the lock stage first")
    lock = _read(lock_path)
    problems = verify_lock(lock, paths)
    lock_sha = sha256_file(lock_path)
    if lock_sha != expected_lock_sha256:
        problems["lock_is_not_the_reviewed_lock"] = {"expected": expected_lock_sha256, "now": lock_sha}
    before = eq._openings(paths.openings)
    if any(opening.get("experiment") == "E1" for opening in before):
        problems["e1_already_opened"] = True
    if problems:
        return _blocked(out, "test", problems, lock=str(lock_path))
    parts: dict[str, Any] = {"verify_lock": eq._part(tick)}
    unsealed = unsealed_test_rows(paths)                   # checks before scoring
    if unsealed:
        return _blocked(out, "test", {"row_tables_not_sealed": unsealed})
    tick_part = time.perf_counter()
    scores: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        scores[arm] = {}
        for family in FAMILIES:
            cell = lock["cells"][arm][family]
            try:
                scores[arm][family] = _scored(out / arm / SCORES_DIR, cell, scorer, split="TEST")
            except Exception as error:  # noqa: BLE001  (no sealed file was read: refused, diagnosable, rerunnable)
                return _blocked(out, "test", {"scoring_refused": {f"{arm}:{family}": f"{type(error).__name__}: {error}"}})
    parts["score"] = eq._part(tick_part)

    # the opening: the record first, then OPENINGS.json, then the sealed files (EXPOSURE_KO.md rule 1) ---------------------
    sealed = sealed_paths(paths)
    entry = {"experiment": "E1", "utc": utc(), "lock": str(lock_path), "lock_sha256": lock_sha, "prespec_sha256": lock["prespec"]["sha256"],
             "labels": [str(sealed[label]) for label in SEALED_LABELS],
             "sealed_sha256": {label: lock["verify"][SEALED_LOCK_LABEL[label]]["sha256"] for label in SEALED_LABELS},
             "note": OPENING_NOTE}
    write_once(out / OPENING_NAME, _clean({**entry, "openings_before": before}))
    eq.append_opening(paths.openings, entry)
    try:
        return _evaluate_test(paths, lock, lock_path, entry, before, scores, parts, started, tick)
    except BaseException as error:
        _failure(out, entry, error, parts)
        raise


def _evaluate_test(paths: Paths, lock: Mapping[str, Any], lock_path: Path, entry: Mapping[str, Any], before: list,
                   scores: Mapping[str, Mapping[str, Any]], parts: dict[str, Any], started: str, tick: float) -> int:
    tick_part = time.perf_counter()
    result = evaluate_opened(paths, "TEST", sealed_paths(paths), scores, main_paths(paths))
    parts["evaluate"] = eq._part(tick_part)
    inputs = [Path(lock["verify"][label]["path"]) for label in (*SEALED_LOCK_LABEL.values(), "plc:Q_ROWS_h90.npz", "eng:Q_ROWS_h90.npz",
                                                                 "pred:PRED_PLACEBO_h90.npz", "windows:WINDOWS_h90.npz")]
    inputs += [paths.out / arm / SCORES_DIR / f"{lock['cells'][arm][family]['key']}.npz" for arm in ARMS for family in FAMILIES]
    report = {
        "version": VERSION, "experiment": "E1", "split": "TEST", "utc": utc(),
        "prespec": {"path": lock["verify"]["prespec"]["path"], "sha256": lock["prespec"]["sha256"], "commit": lock["prespec"]["commit"]},
        "lock": {"path": str(lock_path), "sha256": entry["lock_sha256"], "locked_utc": lock.get("locked_utc")},
        "opening": dict(entry), "openings_before": before, "labels_opened": entry["labels"], "rules": lock["rules"],
        "selection": lock.get("selection"), "cells": {arm: {family: lock["cells"][arm][family]["key"] for family in FAMILIES} for arm in ARMS},
        "scores": {arm: {family: scores[arm][family]["meta"] for family in FAMILIES} for arm in ARMS},
        "result": _clean(result), "qcal_headline": lock.get("qcal_headline"), "row_counts": lock.get("row_counts"),
        "stages": {**(lock.get("stages") or {}),
                   "test": {"started_utc": started, "finished_utc": utc(), "wall_seconds": time.perf_counter() - tick,
                            "peak_memory_bytes_main_process": peak_memory_bytes(), "parts": parts,
                            "processes": "main process only, no worker, no GPU"}},
        "environment": eq.environment(),
        "run": run_record(rows=result["rows"], matches=result["matches"], inputs=inputs, code=[ROOT / name for name in CODE_FILES],
                          seed=SEED, started_utc=started),
    }
    write_once(paths.out / REPORT_NAME, _clean(report))
    primary = result["primary"]
    print(json.dumps({"done": str(paths.out / REPORT_NAME), "outcome": result["decision"]["outcome"],
                      **{name: [primary["point"][name], primary[f"{name}_ci"]] for name in QUANTITIES}}), flush=True)
    return 0


# ----------------------------------------------------------------------------- CLI
STAGES: dict[str, Callable[..., int]] = {"train": train_stage, "select": select_stage, "qcal": qcal_stage, "lock": lock_stage}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=(*STAGES, "test"))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="train: worker processes at once (1-4)")
    parser.add_argument("--poll", type=float, default=5.0, help="train: seconds between polls")
    parser.add_argument("--out", type=Path, help="select/qcal/lock: write the records here (a dry run); the run folders stay")
    parser.add_argument("--uncommitted", action="store_true", help="lock dry run only: allow uncommitted code")
    parser.add_argument("--confirm-single-opening", action="store_true", help="test: required; the step opens the sealed files once")
    parser.add_argument("--lock-sha256", help="test: required; the SHA-256 of the reviewed E1_LOCK.json")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = es.HIDE_CUDA      # CPU only; inherited by every worker
    es._below_normal_priority()
    paths = Paths(out=args.out) if args.out else Paths()
    if args.out and args.stage in ("train", "test"):
        parser.error("--out is for the select, qcal and lock dry runs")
    if args.uncommitted and not (args.stage == "lock" and args.out):
        parser.error("--uncommitted is for a lock dry run with --out")
    try:
        if args.stage == "train":
            return train_stage(paths, workers=args.workers, poll_s=args.poll)
        if args.stage == "lock":
            return lock_stage(paths, require_committed_code=not args.uncommitted)
        if args.stage in STAGES:
            return STAGES[args.stage](paths)
        if not (args.confirm_single_opening and args.lock_sha256):
            parser.error("the test stage opens the sealed TEST files once; pass --confirm-single-opening and --lock-sha256")
        return run_test_step(paths, expected_lock_sha256=args.lock_sha256)
    except Blocked as error:
        return _blocked(paths.out, args.stage, [str(error)])


if __name__ == "__main__":
    raise SystemExit(main())
