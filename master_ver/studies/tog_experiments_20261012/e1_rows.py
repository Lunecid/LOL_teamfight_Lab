"""T1.4 E1 placebo labels, the placebo and size-matched engagement row tables, the placebo raw store and XY, the balance
tables and q_prepare for both folders (PRESPEC_E1 section 6 step 3; PLAN_E1_E5_KO.md Task 15).

Binding specification: reports/tog_experiments_20261012/PRESPEC_E1_KO.md as amended 2026-10-13 and re-approved (commit
7a841db): section 3 균형, 라벨, 입력과 전처리, 실행 세부; section 6 step 3, "잠금 전에 보지 않는 것" and "행 표 기록 JSON
(Ruling 28)"; section 7 (BLOCKED).  SDD Rulings 24, 27, 28, 48, 51-54.

Inputs (read only, each checked against its record before use): tog_e1_1/WINDOWS_h90.npz (Task 13; one row per
q_run_1/Q_ROWS_h90.npz row, s_ms = -1 unmatched), PRED_PLACEBO_h90.npz and its record (Task 14; V(s) of every matched
window, V(s + D) of the TRAIN/Q_DEV/Q_CAL windows), STATES_h90/ (Task 14; the start and end states), q_run_1's row table,
raw/store and inputs/main_h90.  PRED_PLACEBO_h90_TEST_SEALED.npz is read in memory by the rows stage only, to make the TEST
placebo labels, which go straight into plc/Q_TEST_SEALED_h90.npz (Ruling 54); every other stage hashes its bytes only.

Stages (CLI), each written once under D:/LOL_Training_20261004/tog_e1_1 and resumable: a finished stage is verified by the
SHA-256 of its files and skipped; data files have fixed bytes, so a stage interrupted before its record recomputes them and
keeps the existing files by hash (a file with other bytes is never overwritten).
  rows       plc/Q_ROWS_h90.npz: one row per matched window, in WINDOWS row order, with exactly q_run_1's columns
             (COLUMN_MAP): the matched engagement's match identity, query_pre = 0..n-1 (the placebo raw-store key),
             pre_ms = s, pt_p_pre = V(s), W = V(s + D) - V(s), y = 1[W > 0] (an exact zero is negative with exact_zero 1),
             near_zero_bits by assemble.NEAR_ZERO; TEST rows y = -1, NaN, -1, 0 through the frozen rows.seal_test, their
             labels only in plc/Q_TEST_SEALED_h90.npz.  plc/PLC_KEY_MAP_h90.npz pairs placebo row i with engagement row i.
             eng/Q_ROWS_h90.npz: the q_run_1 public rows of the matched engagements in q_run_1 order (every role; TEST sealed
             as there), weight_raw by rows.match_weights on the subset, q_stop by rows.q_inner_stop (must equal q_run_1's).
             Row-table records <arm>/Q_ROWS_h90.json: rows.write_rows keys without any TEST figure (Ruling 48), still
             handled as sealed summaries (Ruling 28).  E1_ROWS.json.
  check      Store path check: per public role the 500 matched engagements with the smallest SHA-256('<salt>:<engagement
             key>'); their states at tau built by Task 14's builder (e1_states._build_match_states) and stored as state
             parts, then written by this module's chunk writer and merged like the placebo store; every q input field
             (e1_gate.store_fields) must equal q_run_1/raw/store at the same query, and F(tau) XY, the held frame and the
             objective counts must equal q_run_1's XY.npy, ENGAGEMENTS F_pre_ms and the projection's OBJECTIVE_COUNTS.npy,
             else BLOCKED.  STORE_PATH_CHECK_h90/, E1_STORE_CHECK.json.
  store      plc/raw/store: the start-state semantic rows (controls_states.decode_states -> semantic_rows) written with
             features.RawStoreWriter in chunks of consecutive state parts (at most 4 worker processes), merged by
             features.merge_chunks, BUILD.json last (features.build_raw_store's record).  plc/inputs/main_h90/XY.npy: the
             parts' F(s) XY in placebo row order, finite for every row, else BLOCKED; it exists before q_prepare runs
             (q_prepare.py:115-117).  E1_STORE.json.
  eng_inputs eng/raw/store = a directory junction to q_run_1/raw/store; eng/inputs/main_h90/XY.npy = q_run_1's XY.npy rows
             of the engagement rows (source SHA-256 = q_run_1 PREPARED.json xy_sha256 and rows matched by engagement_key,
             else BLOCKED).  E1_ENG_INPUTS.json.
  prepare    The frozen q_prepare.py --run <plc|eng> --variant main_h90 --release reports/q_production_20261005/
             SCIENTIFIC_Q_15638e1f9d2fc7ce.json as subprocesses (FIT = non-STOP TRAIN, REFIT_TRAIN = all TRAIN); every file of
             q_run_1/raw/store hashed before and after (any change BLOCKED).  E1_PREPARE.json.
  balance    BALANCE.json (TRAIN, Q_DEV, Q_CAL) and BALANCE_TEST_SEALED.json (TEST): SMDs of V(start), game time, gold
             difference, start frame age and time since the last kill (pairs with an earlier kill in both arms), the
             objective share per arm (diagnostics.py:152-158), per-feature bin counts and match rates, the reused-window
             share (5 % trigger) and the stratified-r flags; public extras (match rate by D quartile, matched and unmatched
             medians) and the precision disclosure (exact zeros and |W| < 1e-6 per arm).
  record     E1_ROWS_RUN.json: the run record (non-TEST counts, input and code SHA-256, UTC times, stage wall times and
             peak memory).

Nothing about TEST is printed, logged, recorded or reported except paths and SHA-256 (PRESPEC_E1 section 6; Ruling 54):
public records carry TRAIN, Q_DEV and Q_CAL figures only, and a TEST check that fails stops the run.  The frozen formats
BUILD.json, CHUNK.json and PREPARED.json carry the total row count; nobody back-computes a TEST count from them (PRESPEC_E1
section 6).  CPU only (CUDA hidden), at most 4 worker processes, below-normal priority.  No player identifier.

Usage: python -B -m studies.tog_experiments_20261012.e1_rows {rows,check,store,eng_inputs,prepare,balance,record,run}
       [--workers 4]
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Callable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import controls_states as cs  # noqa: E402
from studies.label_production_20261005.assemble import NEAR_ZERO, PROJECTION  # noqa: E402
from studies.q_production_20261005 import features as qf, release as qrelease, rows as q_rows_module  # noqa: E402
from studies.q_production_20261005.rows import Q_ROLES, SEALED_COLUMNS  # noqa: E402
from studies.tog_experiments_20261012 import e1_candidates as ec, e1_states as es, e4_q as eq, e4_run as er  # noqa: E402
from studies.tog_experiments_20261012.common import (  # noqa: E402
    LABELS, Q_RUN, TOG_E1, document_record, git_head, peak_memory_bytes, sha256_file, utc, write_once)

VERSION = "tog_e1_rows_v1_20261013"
ROWS_VERSION = "tog_e1_row_table_v1_20261013"
Blocked = es.Blocked                       # a PRESPEC_E1 section 7 stop or an input that differs from its record
VARIANT, HORIZON = eq.VARIANT, eq.HORIZON  # main_h90, 90
PRESPEC, AMENDMENT = es.PRESPEC, es.AMENDMENT
RELEASE, RELEASE_DIGEST = eq.RELEASE, eq.RELEASE_DIGEST
PREPARE_SCRIPT = "studies/q_production_20261005/q_prepare.py"
RELEASE_ARGUMENT = "reports/q_production_20261005/SCIENTIFIC_Q_15638e1f9d2fc7ce.json"
TRAIN, TEST = es.TRAIN, es.TEST
PUBLIC_ROLES = es.PUBLIC_ROLES             # TRAIN, Q_DEV, Q_CAL (TEST only in sealed files)
MAX_WORKERS = es.MAX_WORKERS
PLC, ENG, CHECK_DIR = "plc", "eng", "STORE_PATH_CHECK_h90"
ROWS_NAME, ROWS_RECORD, SEALED_NAME, MAP_NAME = "Q_ROWS_h90.npz", "Q_ROWS_h90.json", "Q_TEST_SEALED_h90.npz", "PLC_KEY_MAP_h90.npz"
ROWS_STAGE, CHECK_STAGE, STORE_STAGE = "E1_ROWS.json", "E1_STORE_CHECK.json", "E1_STORE.json"
ENG_STAGE, PREPARE_STAGE, RUN_RECORD = "E1_ENG_INPUTS.json", "E1_PREPARE.json", "E1_ROWS_RUN.json"
BALANCE_NAME, BALANCE_SEALED_NAME = "BALANCE.json", "BALANCE_TEST_SEALED.json"
PLC_COMPONENT_PREFIX, PLC_COMPONENT_KEEP = b"plc-", 60
PARTS_PER_CHUNK = 16                       # chunks follow the state parts, so their number depends on the parts only
CHECK_SALT, CHECK_PER_ROLE = "tog_e1_rows_store_check_20261013", 500
SMD_TRIGGER, REUSE_TRIGGER = 0.1, 0.05     # PRESPEC_E1 section 3 균형 (stratified r; de-duplicated r)
TINY_W = 1e-6                              # precision disclosure: CPU batch noise of the placebo V is about 3e-7
EXPECTED = {  # the Task 13/14 outputs this task was dispatched on (each must also equal its record)
    "windows": "9cf280700d77ed6be3467864eb00ba0af07fef43eb460a1cb57c50b9c1ebca07",
    "pred": "703f98dabc8598c06230a23fe9a00269b78a1ac1ef3c512cbcd124e909079213",
    "pred_sealed": "76310ce5b8f05d028babb29398075386c650b4e91cdcb46b8351e2176e879a0c",
    "pred_record": "dd2efe2bd0de7b3d26b07acf6a0c7d06bd687dee465f13b16870b41db09c4255",
    "states_record": "1e400af00a2dc73cbb0ed7f2d22e5f914b8fa1e907049d1f06b74bd9feebaff4",
    "q_rows": "eacc0f732dd15dc48e5a16ce32fb8f86187ff8afe0e88b08d6b9ade658d4974a"}
PRED_COLUMNS = ("window_id", "engagement_key", "match_index", "role", "fold", "ctx", "s_ms", "end_ms", "p_start", "p_end")
SEALED_PRED_COLUMNS = ("window_id", "engagement_key", "p_end")
ENGAGEMENT_COLUMNS = ("canonical_match_key", "component_key", "role", "valid_h90", "query_pre", "pre_ms", "query_h90",
                      "post_ms_h90", "F_pre_ms")
PART_MEMBERS = ("state_id", "state_match", "state_ms", "frame_ms", "xy", "xy_support", "window_id", "window_start_state",
                "window_end_state", "feature_objective_counts")
WINDOW_BALANCE_COLUMNS = ("engagement_key", "match_index", "role", "tau_ms", "D_ms", "s_ms", "start_age_bin", "n_bound",
                          "end_age_bin", "time_bin", "kill_gap_bin", "gold_bin", "start_age_eng_ms", "start_age_plc_ms",
                          "kill_gap_eng_ms", "kill_gap_plc_ms", "gold_eng", "gold_plc", "n_candidates", "n_survivors",
                          "reject_counts", "xy_finite_plc", "meets_component_window_plc")
CROSS_CHECK_KEYS = ("engagements", "matched", "match_rate", "bins", "reuse", "kill_gap_balance")
CODE_FILES = ("studies/tog_experiments_20261012/e1_rows.py", "studies/tog_experiments_20261012/common.py",
              "studies/tog_experiments_20261012/e1_states.py", "studies/tog_experiments_20261012/e1_candidates.py",
              "studies/tog_experiments_20261012/e1_features.py", "studies/tog_experiments_20261012/e1_gate.py",
              "studies/tog_experiments_20261012/e4_q.py", "studies/tog_experiments_20261012/e4_run.py",
              "studies/q_production_20261005/rows.py", "studies/q_production_20261005/features.py",
              "studies/q_production_20261005/q_prepare.py", "studies/q_production_20261005/qbatch.py",
              "studies/q_production_20261005/release.py", "studies/q_production_20261005/spatial.py",
              "studies/q_production_20261005/qtest.py", "studies/label_production_20261005/assemble.py",
              "studies/label_production_20261005/controls_states.py", "studies/label_production_20261005/diagnostics.py")
COLUMN_MAP = {  # q_run_1 column -> the placebo value, the frozen code that reads it, and why (PRESPEC_E1 section 3 실행 세부)
    "canonical_match_key": {"value": "the matched engagement's", "readers": "qbatch.ScopeContext (game keys of the fit evidence)",
                            "why": "the window lies in the engagement's match; STOP, folds and the fit evidence go by match"},
    "component_key": {"value": "b'plc-' + the first 60 characters of the matched engagement's component key (S64)",
                      "readers": "none on the E1 path (rows.build_rows writes it; q_production report scripts read q_run_1's)",
                      "why": "a kill-free window has no component; 'plc-' never occurs in a q_run_1 component key (64 "
                             "lowercase hex characters), so the key is new and keeps engagement_key = canonical_match_key:"
                             "component_key (rows.py:133)"},
    "engagement_key": {"value": "canonical_match_key + b':' + component_key (S129)",
                       "readers": "q_worker (Q_DEV key hash), qtest.score (TEST_SCORES keys), any join with a label file",
                       "why": "differs from every q_run_1 engagement key, so no join mixes the arms; one to one with the "
                              "matched engagement (plc/PLC_KEY_MAP_h90.npz)"},
    "match_index": {"value": "the matched engagement's", "readers": "q_worker (normalized weights), qtest", "why": "same match"},
    "role": {"value": "the matched engagement's", "readers": "qbatch.ScopeContext, q_worker.splits, qtest.score",
             "why": "the window lies in the engagement's match"},
    "fold": {"value": "the matched engagement's", "readers": "none on the E1 path (Task 14 used it for the V context)",
             "why": "same match"},
    "q_stop": {"value": "rows.q_inner_stop(canonical_match_key) on TRAIN rows, else False",
               "readers": "qbatch.ScopeContext (FIT scope), q_worker.splits", "why": "the q inner STOP set is whole matches"},
    "y": {"value": "1[W > 0] on TRAIN/Q_DEV/Q_CAL rows (an exact zero is negative); -1 on TEST rows",
          "readers": "qbatch.source (training labels; refuses -1), q_worker references, qtest.score (TEST rows must be -1)",
          "why": "PRESPEC_E1 section 3 라벨; TEST labels sealed by rows.seal_test"},
    "weight_raw": {"value": "rows.match_weights over the placebo table", "readers": "qbatch.source, q_worker (raw Brier)",
                   "why": "PRESPEC_E1 section 3 실행 세부"},
    "query_pre": {"value": "0..n-1 in placebo row order",
                  "readers": "qbatch.ScopeContext (raw-store positions); q_prepare reads it only when XY.npy is missing",
                  "why": "the synthetic key of the placebo raw store; XY.npy is written before q_prepare"},
    "pre_ms": {"value": "s", "readers": "none on the E1 path", "why": "the window start"},
    "elapsed_min": {"value": "s / 60000", "readers": "q_worker and qtest.LoadedCell (time_only, PT_linear, PT_flex); "
                    "qtest.evaluate game-time strata", "why": "PT_flex's game-time input at the window start"},
    "frame_age_s": {"value": "(s - F(s)) / 1000 = start_age_plc_ms / 1000",
                    "readers": "qtest.evaluate frame-age strata (accept [0, 10])",
                    "why": "the start frame age; inside [0, 10] by the start_age_out rule"},
    "truncation_bits": {"value": "0", "readers": "qtest.evaluate endpoint strata (bit 0 = horizon reached), not on the E1 path",
                        "why": "no endpoint rule (horizon, next kill, next anchor, game end) ends a kill-free window; its "
                               "length D is the engagement's"},
    "near_zero_bits": {"value": "bit k when |W| < NEAR_ZERO[k] (assemble.py:40, :125-127); 0 on TEST rows",
                       "readers": "none on the E1 path (sealed column)", "why": "the main labels' rule"},
    "exact_zero": {"value": "1[W == 0]; -1 on TEST rows", "readers": "none on the E1 path (sealed column)",
                   "why": "the main labels' rule (assemble.py:124)"},
    "label_svi_delta": {"value": "W = V(s + D) - V(s) in float64; NaN on TEST rows",
                        "readers": "no model (restricted column, rows.py:12-14)", "why": "PRESPEC_E1 section 3 라벨"},
    "pt_p_pre": {"value": "V(s) (PRED_PLACEBO_h90 p_start)", "readers": "q_worker and qtest.LoadedCell (p_only, PT_linear, "
                 "PT_flex only)", "why": "PT_flex's V input (PRESPEC_E1 section 3 라벨, 모델)"},
    "eval_n_kills": {"value": "0", "readers": "qtest.evaluate kill-count strata and q_production report scripts, none on "
                     "the E1 path (qeval.kill_count_strata refuses 0)",
                     "why": "a kill-free window holds no kill; kill strata are engagement strata (D-Q17), so refusing is "
                            "the right outcome"},
    "eval_firstkill_credit_blue": {"value": "0", "readers": "qtest.evaluate credit strata, none on the E1 path "
                                   "(qeval.credit_strata refuses 0)", "why": "no first kill, no first-kill credit"},
    "eval_firstkill_credit_red": {"value": "0", "readers": "as eval_firstkill_credit_blue", "why": "no first kill"},
}
KEY_FORM = ("placebo component_key = b'plc-' + first 60 characters of the matched engagement's component key; engagement_key "
            "= canonical_match_key + b':' + component_key (S129, ASCII, deterministic).  q_run_1 component keys are 64 "
            "lowercase hex characters, so no placebo key can equal one; the code also checks disjointness from every "
            "q_run_1 engagement and component key and uniqueness within the placebo table, and PLC_KEY_MAP_h90.npz records "
            "the pairing (placebo row i = engagement row i = q_run_1 row window_id[i])")
LABEL_RULE = ("W = V(s + D) - V(s) in float64 with the role's context at both ends (Task 14); y = 1[W > 0], an exact zero "
              "negative with exact_zero 1; near_zero_bits bit k when |W| < (0.005, 0.01, 0.02)[k] (e4_run.direction_labels "
              "and zero_flags, the frozen assemble.label_table rule)")
TEST_NOTE = ("the TEST rows are in both row tables (PRESPEC_E1 section 3 실행 세부); no TEST count, rate or summary is "
             "recorded here; TEST labels only in plc/Q_TEST_SEALED_h90.npz, TEST balance only in BALANCE_TEST_SEALED.json")
SEALED_SUMMARY = ("Ruling 28/48: the rows.write_rows record keys with label figures for TRAIN, Q_DEV and Q_CAL only (no TEST "
                  "figure, no total row count); still handled as a sealed summary: never printed, hash bound in the E1 lock")
DEFINITIONS = {
    "smd": "(mean placebo - mean engagement) / sqrt((var engagement + var placebo) / 2), population variances, row "
           "weighted, over the matched pairs of the role (e1_candidates.smd)",
    "v_start": "engagement V(tau) = q_run_1 pt_p_pre (labels, CUDA); placebo V(s) = PRED_PLACEBO_h90 p_start (CPU)",
    "game_time_min": "tau / 60000 and s / 60000", "gold_difference": "F(start) cached gold, Blue - Red (WINDOWS gold_eng, gold_plc)",
    "start_frame_age_s": "start - F(start) (WINDOWS start_age_eng_ms, start_age_plc_ms) / 1000",
    "time_since_last_kill_s": "start - last kill before the start, pairs where both windows have an earlier kill",
    "objective_share": "windows whose OBJECTIVE_COUNTS sum at the end state exceeds the start state's (diagnostics.py:152-158); "
                       "engagement: projection OBJECTIVE_COUNTS.npy at query_pre and query_h90; placebo: the state parts' "
                       "feature_objective_counts at s and s + D",
    "bins": "per feature, engagements of the role per bin, matched engagements and the match rate (e1_candidates.role_summary)",
    "reuse": "placebo rows sharing (match, s) (e1_candidates.reuse_summary); de-duplicated r when the share > 5 %",
    "stratified_r": "|SMD| > 0.1 on V(start) or game time",
    "d_quartiles": "edges = numpy.quantile(D of the role's engagements, [0.25, 0.5, 0.75], linear); quartile = "
                   "searchsorted(edges, D, side='right')",
    "precision": "exact zeros and |W| < 1e-6 among TRAIN/Q_DEV/Q_CAL rows (placebo W from CPU V, engagement W from the "
                 "stored CUDA labels)"}


@dataclass(frozen=True)
class Paths:
    """Where the stages read and write; the defaults are the real run (the tests point them at tmp_path)."""

    source: Path = TOG_E1                  # Task 13/14 outputs (read only)
    out: Path = TOG_E1                     # this task's outputs
    q_run: Path = Q_RUN
    labels: Path = LABELS
    projection: Path = PROJECTION
    release: Path = RELEASE
    prespec: Path = PRESPEC
    amendment: Path = AMENDMENT

    @property
    def windows(self) -> Path:
        return self.source / "WINDOWS_h90.npz"

    @property
    def windows_record(self) -> Path:
        return self.source / "WINDOWS_h90.json"

    @property
    def pred(self) -> Path:
        return self.source / es.PRED_NAME

    @property
    def pred_sealed(self) -> Path:
        return self.source / es.SEALED_NAME

    @property
    def pred_record(self) -> Path:
        return self.source / es.PRED_RECORD

    @property
    def states_dir(self) -> Path:
        return self.source / es.STATES_DIR

    @property
    def states_record(self) -> Path:
        return self.states_dir / es.STATES_RECORD

    @property
    def plc(self) -> Path:
        return self.out / PLC

    @property
    def eng(self) -> Path:
        return self.out / ENG

    @property
    def check(self) -> Path:
        return self.out / CHECK_DIR

    @property
    def q_rows(self) -> Path:
        return self.q_run / "Q_ROWS_h90.npz"

    @property
    def q_store(self) -> Path:
        return self.q_run / "raw" / "store"

    @property
    def q_inputs(self) -> Path:
        return self.q_run / "inputs" / VARIANT

    @property
    def engagements(self) -> Path:
        return self.labels / "ENGAGEMENTS.npz"

    @property
    def diagnostics(self) -> Path:
        return self.labels / "DIAGNOSTICS_LAYER2.json"

    @property
    def objective_counts(self) -> Path:
        return self.projection / "OBJECTIVE_COUNTS.npy"


# ----------------------------------------------------------------------------- labels and keys
def placebo_labels(p_start: Any, p_end: Any) -> dict[str, np.ndarray]:
    """y, label_svi_delta, exact_zero and near_zero_bits of W = p_end - p_start by the main labels' rule (e4_run's mirror of
    assemble.label_table); probabilities that are not finite inside [0, 1] raise."""

    flags = er.zero_flags(p_start, p_end)
    return {"y": er.direction_labels(p_start, p_end),
            "label_svi_delta": np.asarray(p_end, dtype=np.float64) - np.asarray(p_start, dtype=np.float64),
            "exact_zero": flags["exact_zero"], "near_zero_bits": flags["near_zero_bits"]}


def placebo_keys(canonical_match_key: Any, component_key: Any) -> tuple[np.ndarray, np.ndarray]:
    """(component_key, engagement_key) of the placebo rows from their engagements' keys (KEY_FORM)."""

    cmk, ck = np.asarray(canonical_match_key), np.asarray(component_key)
    if cmk.dtype.kind != "S" or ck.dtype.kind != "S" or cmk.ndim != 1 or cmk.shape != ck.shape:
        raise Blocked("match and component keys must be aligned 1-D byte strings")
    component = np.char.add(np.asarray(PLC_COMPONENT_PREFIX), ck.astype(f"S{PLC_COMPONENT_KEEP}"))
    return component, np.char.add(np.char.add(cmk, b":"), component)


def key_problems(engagement_plc: Any, component_plc: Any, engagement_q: Any, component_q: Any) -> list[str]:
    """Placebo keys: unique, ASCII, and equal to no q_run_1 engagement or component key."""

    engagement, component = np.asarray(engagement_plc), np.asarray(component_plc)
    problems = []
    if len(np.unique(engagement)) != len(engagement):
        problems.append("placebo engagement keys are not unique")
    if len(engagement) and max(np.frombuffer(engagement.tobytes(), dtype=np.uint8)) >= 128:
        problems.append("a placebo engagement key is not ASCII")
    if np.isin(engagement, np.asarray(engagement_q)).any():
        problems.append("a placebo engagement key equals a q_run_1 engagement key")
    if np.isin(component, np.asarray(component_q)).any():
        problems.append("a placebo component key equals a q_run_1 component key")
    return problems


def inner_stop(canonical_match_key: Any, role: Any) -> np.ndarray:
    """rows.q_inner_stop of each TRAIN row's match (False on other rows), as rows.build_rows sets q_stop."""

    keys, role = np.asarray(canonical_match_key), np.asarray(role).astype(np.int64)
    stop = np.zeros(len(keys), dtype=bool)
    seen: dict[bytes, bool] = {}
    for i in np.flatnonzero(role == TRAIN).tolist():
        key = bytes(keys[i])
        if key not in seen:
            seen[key] = bool(q_rows_module.q_inner_stop(key.decode("ascii")))
        stop[i] = seen[key]
    return stop


def conform(values: Mapping[str, np.ndarray], like: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``values`` in the column order and dtypes of ``like`` (q_run_1's row table); a missing, extra or lossy column is
    BLOCKED."""

    if set(values) != set(like):
        raise Blocked(f"the placebo columns differ from q_run_1's: missing {sorted(set(like) - set(values))}, extra "
                      f"{sorted(set(values) - set(like))}")
    out = {}
    for name, reference in like.items():
        dtype, source = np.asarray(reference).dtype, np.asarray(values[name])
        cast = source.astype(dtype)
        if cast.shape != source.shape or not np.array_equal(cast, source, equal_nan=dtype.kind == "f"):
            raise Blocked(f"{name}: placebo values do not fit q_run_1's dtype {dtype}")
        out[name] = cast
    return out


def placebo_table(q_rows: Mapping[str, np.ndarray], windows: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray],
                  sealed: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """(placebo row table with labels on every row, matched window ids).  ``pred`` is PRED_PLACEBO_h90's public part,
    ``sealed`` the TEST rows' window_id, engagement_key and p_end (read in memory only).  The caller seals the TEST labels
    with rows.seal_test before anything is written.  Every alignment or validity failure is BLOCKED (PRESPEC_E1 section 7)."""

    es.check_alignment(windows, q_rows)
    matched = np.flatnonzero(np.asarray(windows["s_ms"]) >= 0).astype(np.int64)
    window_id = np.asarray(pred["window_id"]).astype(np.int64)
    if window_id.shape != matched.shape or not np.array_equal(window_id, matched):
        raise Blocked("PRED_PLACEBO_h90 rows are not the matched windows in WINDOWS row order")
    for name in ("engagement_key", "match_index", "role", "fold"):
        left, right = np.asarray(pred[name]), np.asarray(q_rows[name])[matched]
        if left.dtype.kind in "iu" and right.dtype.kind in "iu":
            left, right = left.astype(np.int64), right.astype(np.int64)
        if not np.array_equal(left, right):
            raise Blocked(f"PRED_PLACEBO_h90 {name} differs from its windows' q rows (row-table key mismatch)")
    s = np.asarray(windows["s_ms"])[matched].astype(np.int64)
    end = s + np.asarray(windows["D_ms"])[matched].astype(np.int64)
    if (not np.array_equal(np.asarray(pred["s_ms"]).astype(np.int64), s)
            or not np.array_equal(np.asarray(pred["end_ms"]).astype(np.int64), end)):
        raise Blocked("PRED_PLACEBO_h90 window times differ from WINDOWS_h90 s and s + D")
    role, fold = np.asarray(pred["role"]).astype(np.int64), np.asarray(pred["fold"]).astype(np.int64)
    try:
        codes = es.context_codes(role, fold)
    except ValueError as error:
        raise Blocked(f"a placebo row's V context: {error}") from error
    if not np.array_equal(np.asarray(pred["ctx"]).astype(np.int64), codes.astype(np.int64)):
        raise Blocked("a placebo row's V context differs from the role rule (PRESPEC_E1 section 7)")
    test = role == TEST
    p_start = np.asarray(pred["p_start"], dtype=np.float64)
    p_end = np.array(pred["p_end"], dtype=np.float64, copy=True)
    if not np.isnan(p_end[test]).all():
        raise Blocked("PRED_PLACEBO_h90.npz holds a TEST end value (it must sit only in the sealed file)")
    sealed_id = np.asarray(sealed["window_id"]).astype(np.int64)
    if (sealed_id.shape != window_id[test].shape or not np.array_equal(sealed_id, window_id[test])
            or not np.array_equal(np.asarray(sealed["engagement_key"]), np.asarray(pred["engagement_key"])[test])):
        raise Blocked("the sealed TEST end values do not align with the public TEST rows (window_id, engagement_key)")
    p_end[test] = np.asarray(sealed["p_end"], dtype=np.float64)
    valid = np.isfinite(p_start) & np.isfinite(p_end) & (p_start >= 0) & (p_start <= 1) & (p_end >= 0) & (p_end <= 1)
    if not valid.all():
        bad = {name: int((~valid & (role == code)).sum()) for name, code in PUBLIC_ROLES.items()}
        raise Blocked(f"a placebo V is not a finite probability: {bad}; TEST {'failed' if (~valid & test).any() else 'passed'}")
    labels = placebo_labels(p_start, p_end)
    cmk = np.asarray(q_rows["canonical_match_key"])[matched]
    component, engagement = placebo_keys(cmk, np.asarray(q_rows["component_key"])[matched])
    match_index = np.asarray(q_rows["match_index"])[matched].astype(np.int64)
    n = len(matched)
    zeros = np.zeros(n, dtype=np.int64)
    values = {"canonical_match_key": cmk, "component_key": component, "engagement_key": engagement, "match_index": match_index,
              "role": role, "fold": fold, "q_stop": inner_stop(cmk, role), "y": labels["y"],
              "weight_raw": q_rows_module.match_weights(match_index), "query_pre": np.arange(n, dtype=np.int64), "pre_ms": s,
              "elapsed_min": s / 60000.0, "frame_age_s": np.asarray(windows["start_age_plc_ms"])[matched].astype(np.int64) / 1000.0,
              "truncation_bits": zeros, "near_zero_bits": labels["near_zero_bits"], "exact_zero": labels["exact_zero"],
              "label_svi_delta": labels["label_svi_delta"], "pt_p_pre": p_start, "eval_n_kills": zeros,
              "eval_firstkill_credit_blue": zeros, "eval_firstkill_credit_red": zeros}
    return conform(values, q_rows), matched


def _placeholder_problems(table: Mapping[str, np.ndarray]) -> list[str]:
    test = np.asarray(table["role"]).astype(np.int64) == TEST
    return [f"{name}: the TEST rows do not hold the sealed placeholder {value}" for name, value in eq.TEST_PLACEHOLDERS.items()
            if name in table and not eq._placeholder_ok(np.asarray(table[name])[test], value)]


def _near_bits(delta: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(delta), dtype=np.uint8)
    for bit, threshold in enumerate(NEAR_ZERO):
        bits[np.abs(delta) < threshold] |= 1 << bit
    return bits


def placebo_problems(q_rows: Mapping[str, np.ndarray], public: Mapping[str, np.ndarray], sealed: Mapping[str, np.ndarray]) -> list[str]:
    """The placebo table after the seal: q_run_1's columns and dtypes, the sealed placeholders, the sealed file's layout and
    validity (e4_q.sealed_problems), consistent public labels, keys, weights, times and V (no count is reported)."""

    if list(public) != list(q_rows):
        return ["the placebo columns or their order differ from q_run_1's"]
    problems = [f"{name}: dtype differs from q_run_1's" for name in q_rows if public[name].dtype != np.asarray(q_rows[name]).dtype]
    problems += _placeholder_problems(public)
    problems += eq.sealed_problems(public, sealed)
    role = public["role"].astype(np.int64)
    known = role != TEST
    y, exact = public["y"][known].astype(np.int64), public["exact_zero"][known].astype(np.int64)
    delta = np.asarray(public["label_svi_delta"], dtype=np.float64)[known]
    if not np.isfinite(delta).all():
        problems.append("label_svi_delta is not finite on a TRAIN/Q_DEV/Q_CAL row")
    elif (not np.array_equal(y, (delta > 0).astype(np.int64)) or not np.array_equal(exact, (delta == 0).astype(np.int64))
          or not np.array_equal(public["near_zero_bits"][known], _near_bits(delta))):
        problems.append("y, exact_zero or near_zero_bits disagree with label_svi_delta on a TRAIN/Q_DEV/Q_CAL row")
    n = len(role)
    if not np.array_equal(public["query_pre"], np.arange(n)):
        problems.append("query_pre is not 0..n-1")
    if not np.array_equal(public["weight_raw"], q_rows_module.match_weights(public["match_index"])):
        problems.append("weight_raw is not rows.match_weights of the placebo table")
    if not np.array_equal(public["q_stop"], inner_stop(public["canonical_match_key"], role)):
        problems.append("q_stop is not rows.q_inner_stop of the placebo rows")
    age = np.asarray(public["frame_age_s"], dtype=np.float64)
    if np.any(~np.isfinite(age) | (age < 0) | (age > 10)):
        problems.append("a placebo start frame age lies outside [0, 10] s")
    if not np.array_equal(public["elapsed_min"], public["pre_ms"] / 60000.0) or np.any(public["pre_ms"] < 120_000):
        problems.append("elapsed_min is not pre_ms / 60000 or a window starts before 120 s")
    p = np.asarray(public["pt_p_pre"], dtype=np.float64)
    if np.any(~np.isfinite(p) | (p < 0) | (p > 1)):
        problems.append("pt_p_pre is not a probability")
    for name in ("truncation_bits", "eval_n_kills", "eval_firstkill_credit_blue", "eval_firstkill_credit_red"):
        if np.any(public[name] != 0):
            problems.append(f"{name} is not 0")
    return problems


# ----------------------------------------------------------------------------- the size-matched engagement table
def engagement_table(q_rows: Mapping[str, np.ndarray], matched: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The q_run_1 public rows of the matched engagements in q_run_1 order (all roles, TEST sealed as there) with weight_raw
    by rows.match_weights on the subset and q_stop by rows.q_inner_stop, which must equal q_run_1's (else BLOCKED)."""

    matched = np.asarray(matched, dtype=np.int64)
    eng = {name: np.array(np.asarray(value)[matched]) for name, value in q_rows.items()}
    stop = inner_stop(eng["canonical_match_key"], eng["role"])
    equal = bool(np.array_equal(stop, np.asarray(eng["q_stop"], dtype=bool)))
    if not equal:
        raise Blocked("rows.q_inner_stop differs from q_run_1's stored q_stop on a matched engagement")
    eng["q_stop"] = stop.astype(np.asarray(q_rows["q_stop"]).dtype)
    eng["weight_raw"] = q_rows_module.match_weights(eng["match_index"]).astype(np.asarray(q_rows["weight_raw"]).dtype)
    return eng, {"q_stop_recomputed_equals_stored": equal}


def engagement_problems(q_rows: Mapping[str, np.ndarray], matched: Any, eng: Mapping[str, np.ndarray]) -> list[str]:
    matched = np.asarray(matched, dtype=np.int64)
    if list(eng) != list(q_rows):
        return ["the engagement columns or their order differ from q_run_1's"]
    problems = []
    for name, value in q_rows.items():
        left, right = np.asarray(value)[matched], np.asarray(eng[name])
        if left.dtype != right.dtype or left.shape != right.shape:
            problems.append(f"{name}: dtype or shape differs from q_run_1's")
        elif name == "weight_raw":
            if not np.array_equal(right, q_rows_module.match_weights(eng["match_index"])):
                problems.append("weight_raw is not rows.match_weights of the engagement table")
        elif not np.array_equal(left, right, equal_nan=left.dtype.kind == "f"):
            problems.append(f"{name}: values differ from q_run_1's matched rows")
    problems += _placeholder_problems(eng)
    if len(np.unique(eng["engagement_key"])) != len(eng["engagement_key"]):
        problems.append("engagement keys are not unique")
    return problems


def pair_problems(plc: Mapping[str, np.ndarray], eng: Mapping[str, np.ndarray]) -> list[str]:
    """Placebo row i and engagement row i: the same match, role, fold and q STOP membership."""

    return [f"{name} differs between the placebo row and its engagement row" for name in
            ("canonical_match_key", "match_index", "role", "fold", "q_stop")
            if len(plc[name]) != len(eng[name]) or not np.array_equal(plc[name], eng[name])]


def key_map(matched: Any, plc: Mapping[str, np.ndarray], eng: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The recorded one-to-one pairing: placebo row i = engagement row i = q_run_1 row window_id[i]."""

    matched = np.asarray(matched, dtype=np.int64)
    n = len(matched)
    return {"plc_row": np.arange(n, dtype=np.int64), "eng_row": np.arange(n, dtype=np.int64), "window_id": matched,
            "engagement_key_plc": np.asarray(plc["engagement_key"]), "engagement_key_eng": np.asarray(eng["engagement_key"]),
            "query_pre_plc": np.asarray(plc["query_pre"]).astype(np.int64)}


def nontest_counts(table: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The counts block of rows.write_rows (rows.py:148-158) for TRAIN, Q_DEV and Q_CAL only: no TEST entry and no total
    row count (Ruling 48; PRESPEC_E1 section 6)."""

    role = np.asarray(table["role"]).astype(np.int64)
    match = np.asarray(table["match_index"])
    y, weight = np.asarray(table["y"]), np.asarray(table["weight_raw"], dtype=np.float64)
    exact, stop = np.asarray(table["exact_zero"]), np.asarray(table["q_stop"], dtype=bool)
    counts: dict[str, Any] = {}
    for name, code in PUBLIC_ROLES.items():
        mask = role == code
        values = y[mask].astype(np.float64)
        if not np.isin(values, (0.0, 1.0)).all():
            raise Blocked(f"{name} rows lack a binary label")
        counts[name] = {"rows": int(mask.sum()), "matches": int(np.unique(match[mask]).size), "positive": int(values.sum()),
                        "positive_rate_unit": float(values.mean()) if mask.any() else None,
                        "positive_rate_match_weighted": float((weight[mask] * values).sum() / weight[mask].sum()) if mask.any() else None,
                        "exact_zero": int((exact[mask] == 1).sum())}
    train = role == TRAIN
    counts["TRAIN"].update(stop_rows=int((train & stop).sum()), stop_matches=int(np.unique(match[train & stop]).size),
                           fit_rows=int((train & ~stop).sum()), fit_matches=int(np.unique(match[train & ~stop]).size))
    return counts


def row_record(arm: str, table: Mapping[str, np.ndarray], *, rows_sha256: str, sealed_sha256: str | None,
               inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The row-table record (rows.write_rows keys; deterministic, so a rerun reproduces its bytes)."""

    sealed_note = ("plc/Q_TEST_SEALED_h90.npz holds the TEST placebo labels, written at creation through rows.seal_test"
                   if sealed_sha256 else "no TEST label file: the TEST step joins q_run_1/Q_TEST_SEALED_h90.npz by "
                   "engagement_key after the E1 lock (PRESPEC_E1 section 3 실행 세부)")
    return {"version": ROWS_VERSION, "arm": arm, "horizon_s": HORIZON, "variant": VARIANT, "q_stop_salt": q_rows_module.Q_STOP_SALT,
            "restricted_columns": list(q_rows_module.RESTRICTED), "sealed_test_columns": list(SEALED_COLUMNS),
            "counts": nontest_counts(table), "rows_sha256": rows_sha256, "test_sealed_sha256": sealed_sha256,
            "test_sealed_note": sealed_note, "inputs": dict(inputs), "record_note": SEALED_SUMMARY,
            "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES}}


# ----------------------------------------------------------------------------- files and records
def npy_bytes(array: np.ndarray) -> bytes:
    """The bytes np.save writes for ``array`` (no pickle); fixed for fixed contents."""

    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _keep_or_write(path: Path, data: bytes) -> dict[str, str]:
    try:
        status, digest = es.write_or_keep(Path(path), data)
    except (ValueError, FileExistsError) as error:
        raise Blocked(str(error)) from error
    return {"path": str(path), "sha256": digest, "status": status}


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return _keep_or_write(path, es.npz_bytes(arrays, compress=False))


def _write_json(path: Path, payload: Any) -> dict[str, str]:
    return _keep_or_write(path, json.dumps(es._json_safe(payload), indent=1, sort_keys=True).encode("utf-8"))


def _load_fields(path: Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in fields}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _check_files(files: Mapping[str, Mapping[str, str]], stage: str) -> None:
    try:
        eq._check_files(files, stage)
    except RuntimeError as error:
        raise Blocked(str(error)) from error


def _stage_done(path: Path, stage: str) -> bool:
    if not Path(path).is_file():
        return False
    _check_files(es._read(path).get("files") or {}, stage)
    print(json.dumps({"stage": stage, "done": str(path)}), flush=True)
    return True


def _require(paths: Paths, name: str) -> dict[str, Any]:
    """An earlier stage's record: present, passed, and every file it lists unchanged, else BLOCKED."""

    path = paths.out / name
    if not path.is_file():
        raise Blocked(f"{name} is missing: run its stage first")
    record = es._read(path)
    if record.get("passed") is not True:
        raise Blocked(f"{name} did not pass")
    _check_files(record.get("files") or {}, name)
    return record


def _common(paths: Paths) -> dict[str, Any]:
    return {"code": document_record(Path(__file__)), "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
            "git_head": git_head(), "prespec": document_record(paths.prespec), "amendment": document_record(paths.amendment),
            "versions": es._versions()}


def _entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _blocked(out: Path, stage: str, reasons: Sequence[str], **extra: Any) -> int:
    """Write E1_ROWS_BLOCKED_<stage>_<n>.json once (first free n) and return 2."""

    path = eq._next_free(Path(out), f"E1_ROWS_BLOCKED_{stage}")
    write_once(path, es._json_safe({"version": VERSION, "experiment": "E1", "blocked_stage": stage, "blocked": True,
                                    "utc": utc(), "reasons": list(reasons), **extra}))
    print(json.dumps({"BLOCKED": stage, "file": str(path), "reasons": list(reasons)}, ensure_ascii=False), flush=True)
    return 2


def _sources(paths: Paths, expected: Mapping[str, str]) -> dict[str, Any]:
    """The Task 13/14 inputs and q_run_1's row table: each SHA-256 equal to its record (and to the dispatch value when
    given), the records consistent with each other, else BLOCKED.  The sealed file is hashed by its bytes only."""

    names = {"windows": paths.windows, "windows_record": paths.windows_record, "pred": paths.pred, "pred_sealed": paths.pred_sealed,
             "pred_record": paths.pred_record, "states_record": paths.states_record, "q_rows": paths.q_rows}
    missing = [str(path) for path in names.values() if not Path(path).is_file()]
    if missing:
        raise Blocked(f"inputs missing: {missing}")
    hashes = {key: sha256_file(path) for key, path in names.items()}
    windows_record, pred_record = es._read(paths.windows_record), es._read(paths.pred_record)
    states_record = es._read(paths.states_record)
    problems = []
    if hashes["windows"] != (windows_record.get("windows_npz") or {}).get("sha256"):
        problems.append("WINDOWS_h90.npz differs from WINDOWS_h90.json")
    if (windows_record.get("stage") != "final_windows" or windows_record.get("smoke") is not False
            or (windows_record.get("xy_check") or {}).get("passed") is not True
            or (windows_record.get("gate_record") or {}).get("passed") is not True):
        problems.append("WINDOWS_h90.json is not a passed final-window record")
    outputs = pred_record.get("outputs") or {}
    if pred_record.get("stage") != "placebo_v" or pred_record.get("smoke") is not False:
        problems.append("PRED_PLACEBO_h90.json is not a full placebo-V record")
    if hashes["pred"] != (outputs.get("public") or {}).get("sha256"):
        problems.append("PRED_PLACEBO_h90.npz differs from PRED_PLACEBO_h90.json")
    if hashes["pred_sealed"] != (outputs.get("sealed") or {}).get("sha256"):
        problems.append("PRED_PLACEBO_h90_TEST_SEALED.npz differs from PRED_PLACEBO_h90.json (compared by its bytes)")
    inputs = pred_record.get("inputs") or {}
    if inputs.get("windows_npz") != hashes["windows"] or inputs.get("q_rows") != hashes["q_rows"]:
        problems.append("PRED_PLACEBO_h90.json was made from other windows or q rows")
    if (pred_record.get("states") or {}).get("record_sha256") != hashes["states_record"]:
        problems.append("PRED_PLACEBO_h90.json was made from another STATES_h90.json")
    if (states_record.get("stage") != "states" or states_record.get("smoke") is not False
            or (states_record.get("windows") or {}).get("sha256") != hashes["windows"]
            or (states_record.get("q_rows") or {}).get("sha256") != hashes["q_rows"]):
        problems.append("STATES_h90.json is not the full states record of these windows and q rows")
    problems += [f"{key} SHA-256 differs from the dispatch value" for key, value in expected.items() if hashes.get(key) != value]
    if problems:
        raise Blocked("; ".join(problems))
    return {"hashes": hashes, "paths": {key: str(path) for key, path in names.items()}, "windows_record": windows_record,
            "pred_record": pred_record, "states_record": states_record}


def _input_entries(sources: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    entries = {key: {"path": sources["paths"][key], "sha256": value} for key, value in sources["hashes"].items()}
    entries["pred_sealed"]["note"] = "hashed by its bytes; read in memory by the rows stage only, for the TEST labels (Ruling 54)"
    return entries


# ----------------------------------------------------------------------------- stage: rows
def rows_stage(paths: Paths = Paths(), *, expected: Mapping[str, str] = EXPECTED) -> int:
    """Both row tables, the sealed TEST placebo labels, the key map and the row-table records; E1_ROWS.json."""

    target = paths.out / ROWS_STAGE
    if _stage_done(target, "rows"):
        return 0
    started, tick = utc(), time.perf_counter()
    sources = _sources(paths, expected)
    windows, windows_info = es.load_windows(paths.windows)
    q_rows = _load_npz(paths.q_rows)
    pred = _load_fields(paths.pred, PRED_COLUMNS)
    sealed_end = _load_fields(paths.pred_sealed, SEALED_PRED_COLUMNS)     # in memory only (Ruling 54)
    stages = [es._stage("load", started, tick)]
    s, t = utc(), time.perf_counter()
    full, matched = placebo_table(q_rows, windows, pred, sealed_end)
    del sealed_end
    public, sealed = q_rows_module.seal_test(full)
    del full
    eng, eng_info = engagement_table(q_rows, matched)
    problems = (placebo_problems(q_rows, public, sealed) + engagement_problems(q_rows, matched, eng)
                + key_problems(public["engagement_key"], public["component_key"], q_rows["engagement_key"], q_rows["component_key"])
                + pair_problems(public, eng))
    counts = {"plc": es.public_counts(public["role"], public["match_index"]), "eng": es.public_counts(eng["role"], eng["match_index"])}
    roles = sources["windows_record"].get("roles")
    if roles is not None:
        problems += [f"{name}: placebo rows differ from WINDOWS_h90.json matched windows" for name in PUBLIC_ROLES
                     if counts["plc"][name]["rows"] != (roles.get(name) or {}).get("matched")]
    if problems:
        raise Blocked("; ".join(problems))
    mapping = key_map(matched, public, eng)
    stages.append(es._stage("tables", s, t))
    s, t = utc(), time.perf_counter()
    inputs = _input_entries(sources)
    files = {"plc/Q_TEST_SEALED_h90.npz": _write_npz(paths.plc / SEALED_NAME, sealed)}
    del sealed
    files["plc/Q_ROWS_h90.npz"] = _write_npz(paths.plc / ROWS_NAME, public)
    files["plc/PLC_KEY_MAP_h90.npz"] = _write_npz(paths.plc / MAP_NAME, mapping)
    files["eng/Q_ROWS_h90.npz"] = _write_npz(paths.eng / ROWS_NAME, eng)
    files["plc/Q_ROWS_h90.json"] = _write_json(paths.plc / ROWS_RECORD, row_record(
        "placebo", public, rows_sha256=files["plc/Q_ROWS_h90.npz"]["sha256"],
        sealed_sha256=files["plc/Q_TEST_SEALED_h90.npz"]["sha256"], inputs=inputs))
    files["eng/Q_ROWS_h90.json"] = _write_json(paths.eng / ROWS_RECORD, row_record(
        "engagement", eng, rows_sha256=files["eng/Q_ROWS_h90.npz"]["sha256"], sealed_sha256=None,
        inputs={**inputs, "q_run_1_rows": {"path": str(paths.q_rows), "sha256": sources["hashes"]["q_rows"]}}))
    stages.append(es._stage("write", s, t))
    record = {
        "version": VERSION, "experiment": "E1", "stage": "rows", "passed": True, "started_utc": started, "utc": utc(),
        "seed": None, "files": files, "inputs": inputs, "windows": windows_info, "counts": counts,
        "row_order": "placebo rows: the matched windows in WINDOWS_h90 (= q_run_1) row order; engagement rows: the same "
                     "engagements in q_run_1 order, so placebo row i pairs with engagement row i",
        "q_stop": eng_info, "column_map": COLUMN_MAP, "key_form": KEY_FORM, "label_rule": LABEL_RULE,
        "weights": "weight_raw = rows.match_weights over each table (all roles)",
        "checks": {"passed": True, "applied_to": "every row of both tables (TEST rows included; a failure stops the run)",
                   "list": ["inputs equal their records and the dispatch hashes", "windows = q_run_1 rows (key, match, role)",
                            "PRED_PLACEBO rows = matched windows (keys, match, role, fold, s, s + D)", "V context = role rule",
                            "TEST end values only in the sealed file", "sealed rows align with the public TEST rows",
                            "V finite inside [0, 1]", "q_run_1 columns, order and dtypes, lossless", "sealed placeholders",
                            "sealed file layout and validity (e4_q.sealed_problems)", "labels consistent with W",
                            "keys unique, ASCII, disjoint from q_run_1's", "pairs share match, role, fold, q_stop",
                            "engagement rows equal q_run_1's except weight_raw", "q_stop recomputed = stored",
                            "placebo rows per role = WINDOWS_h90.json matched windows"]},
        "test_note": TEST_NOTE, "row_table_records": SEALED_SUMMARY, **_common(paths), "stages": stages,
        "peak_memory_bytes": {"driver": peak_memory_bytes()}}
    write_once(target, es._json_safe(record))
    print(json.dumps({"stage": "rows", "record": str(target), "record_sha256": sha256_file(target), "counts": counts}), flush=True)
    return 0


# ----------------------------------------------------------------------------- state parts and store chunks
def part_index(states_dir: Path, record: Mapping[str, Any], window_id: Any, match_index: Any, s_ms: Any, end_ms: Any,
               start_age_ms: Any) -> dict[str, Any]:
    """Per placebo row (window ids strictly increasing): its state part, the local start and end states, F(s) XY and
    support, and the start and end objective counts, read from the parts' small members.  A window in no part or in two,
    or a start/end state, held start frame or match that differs from the window, is BLOCKED."""

    window_id = np.asarray(window_id, dtype=np.int64)
    match_index, s_ms, end_ms, start_age_ms = (np.asarray(a, dtype=np.int64) for a in (match_index, s_ms, end_ms, start_age_ms))
    if window_id.ndim != 1 or np.any(np.diff(window_id) <= 0) or not (window_id.shape == match_index.shape == s_ms.shape
                                                                       == end_ms.shape == start_age_ms.shape):
        raise Blocked("placebo window ids must be strictly increasing and aligned with their match, times and start age")
    n = len(window_id)
    out: dict[str, Any] = {"part": np.full(n, -1, dtype=np.int64), "local_start": np.full(n, -1, dtype=np.int64),
                           "local_end": np.full(n, -1, dtype=np.int64), "xy": np.full((n, 10, 2), np.nan),
                           "xy_support": np.zeros(n, dtype=bool), "objective_start": np.zeros((n, 9), dtype=np.int64),
                           "objective_end": np.zeros((n, 9), dtype=np.int64), "files": []}
    parts = sorted(record["parts"].values(), key=lambda part: int(part["shard"]))
    for k, part in enumerate(parts):
        with np.load(Path(states_dir) / part["file"], allow_pickle=False) as stored:
            a = {name: stored[name] for name in PART_MEMBERS}
        size = len(a["state_id"])
        first = int(a["state_id"][0]) if size else 0
        if not np.array_equal(a["state_id"], np.arange(first, first + size)):
            raise Blocked(f"{part['file']}: state ids are not consecutive")
        ids = a["window_id"].astype(np.int64)
        rows = np.searchsorted(window_id, ids)
        if np.any(rows >= n) or not np.array_equal(window_id[np.minimum(rows, n - 1)], ids):
            raise Blocked(f"{part['file']}: a window of the part is not a placebo row")
        if np.any(out["part"][rows] >= 0):
            raise Blocked(f"{part['file']}: a window appears in two state parts")
        start, end = a["window_start_state"].astype(np.int64) - first, a["window_end_state"].astype(np.int64) - first
        if np.any((start < 0) | (start >= size) | (end < 0) | (end >= size)):
            raise Blocked(f"{part['file']}: a window's state lies outside the part")
        if (not np.array_equal(a["state_match"][start], match_index[rows]) or not np.array_equal(a["state_match"][end], match_index[rows])
                or not np.array_equal(a["state_ms"][start], s_ms[rows]) or not np.array_equal(a["state_ms"][end], end_ms[rows])):
            raise Blocked(f"{part['file']}: a window's start or end state differs from its window (match or time)")
        if not np.array_equal(a["frame_ms"][start], s_ms[rows] - start_age_ms[rows]):
            raise Blocked(f"{part['file']}: a window's held start frame differs from s - start_age_plc_ms")
        out["part"][rows], out["local_start"][rows], out["local_end"][rows] = k, start, end
        out["xy"][rows], out["xy_support"][rows] = a["xy"][start], a["xy_support"][start]
        out["objective_start"][rows] = a["feature_objective_counts"][start]
        out["objective_end"][rows] = a["feature_objective_counts"][end]
        out["files"].append({"file": part["file"], "sha256": part["sha256"], "shard": int(part["shard"])})
    if np.any(out["part"] < 0):
        raise Blocked("a placebo row's window is in no state part")
    return out


def chunk_plan(parts: Sequence[Mapping[str, Any]], *, parts_per_chunk: int = PARTS_PER_CHUNK) -> list[list[Mapping[str, Any]]]:
    """Groups of consecutive parts; their keys, in part order, must be 0..n-1 exactly (each part's keys non-empty and
    strictly increasing), else BLOCKED: the merged store then holds every row once in key order."""

    if int(parts_per_chunk) < 1:
        raise ValueError("a chunk holds at least one part")
    keys = [np.asarray(part["keys"], dtype=np.int64) for part in parts]
    if any(len(k) == 0 or np.any(np.diff(k) <= 0) for k in keys):
        raise Blocked("a part's rows are empty or not strictly increasing")
    joined = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64)
    if not np.array_equal(joined, np.arange(len(joined))):
        raise Blocked("the parts' rows are not one contiguous partition of the table rows in order")
    k = int(parts_per_chunk)
    return [list(parts[i:i + k]) for i in range(0, len(parts), k)]


def chunk_tasks(chunks: Sequence[Sequence[Mapping[str, Any]]], directory: Path, identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for k, chunk in enumerate(chunks):
        keys = np.concatenate([np.asarray(item["keys"], dtype=np.int64) for item in chunk])
        tasks.append({"chunk_dir": str(Path(directory) / f"chunk_{k:04d}"), "items": [dict(item) for item in chunk],
                      "identity": {**dict(identity), "chunk": k,
                                   "parts": [[Path(item["path"]).name, item["sha256"]] for item in chunk],
                                   "keys_sha256": hashlib.sha256(np.ascontiguousarray(keys).tobytes()).hexdigest()}})
    return tasks


def _chunk_done(directory: Path, keys: np.ndarray, identity: Mapping[str, Any]) -> bool:
    """True if ``directory`` holds this finished chunk; an unfinished one (no CHUNK.json) is moved aside, a different
    finished one is BLOCKED (never overwritten)."""

    record_path = directory / "CHUNK.json"
    if record_path.is_file():
        record = es._read(record_path)
        if (record.get("version") == qf.VERSION and record.get("identity") == identity
                and record.get("queries_sha256") == hashlib.sha256(np.ascontiguousarray(keys, dtype=np.int64).tobytes()).hexdigest()):
            return True
        raise Blocked(f"{directory} holds another chunk; it is never overwritten")
    if directory.exists():
        directory.rename(directory.with_name(f"{directory.name}.partial_{_stamp()}"))
    return False


def store_chunk(task: Mapping[str, Any]) -> dict[str, Any]:
    """One raw-store chunk (worker): for each item (a state part, its SHA-256, the local states in key order, their keys,
    expected match and time), controls_states.decode_states -> semantic_rows -> features.RawStoreWriter."""

    started, tick = utc(), time.perf_counter()
    directory = Path(task["chunk_dir"])
    keys = np.concatenate([np.asarray(item["keys"], dtype=np.int64) for item in task["items"]])
    identity = json.loads(json.dumps(es._json_safe(task["identity"]), sort_keys=True))
    result = {"chunk_dir": str(directory), "started_utc": started, "pid": os.getpid()}
    if _chunk_done(directory, keys, identity):
        return {**result, "reused": True, "chunk_json_sha256": sha256_file(directory / "CHUNK.json"),
                "seconds": time.perf_counter() - tick, "peak_memory_bytes": peak_memory_bytes()}
    for item in task["items"]:                         # every part is checked before anything is written
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise Blocked(f"{Path(item['path']).name} differs from its record")
    try:
        writer = qf.RawStoreWriter(directory, keys)
        for item in task["items"]:
            path = Path(item["path"])
            with np.load(path, allow_pickle=False) as stored:
                arrays = {name: stored[name] for name in stored.files}
            local = np.asarray(item["local"], dtype=np.int64)
            if (local.shape != np.shape(item["keys"]) or np.any(local < 0) or np.any(local >= len(arrays["time_ms"]))
                    or not np.array_equal(arrays["time_ms"][local], np.asarray(item["ms"], dtype=np.int64))
                    or not np.array_equal(arrays["state_match"][local], np.asarray(item["match"], dtype=np.int64))):
                raise Blocked(f"{path.name}: a requested state differs from its window (time or match)")
            states = cs.decode_states(arrays)
            del arrays
            rows = cs.semantic_rows([states[i] for i in local.tolist()])
            for key, row in zip(np.asarray(item["keys"], dtype=np.int64).tolist(), rows):
                writer.add(key, row)
        writer.close({"identity": identity})
    except Blocked:
        raise
    except (ValueError, KeyError) as error:        # the frozen decoder, row builder or writer refused a state or a key
        raise Blocked(f"{directory.name}: {type(error).__name__}: {error}") from error
    return {**result, "reused": False, "chunk_json_sha256": sha256_file(directory / "CHUNK.json"),
            "seconds": time.perf_counter() - tick, "peak_memory_bytes": peak_memory_bytes()}


def merge_store(chunk_dirs: Sequence[Path], store_dir: Path, extra: Mapping[str, Any]) -> dict[str, Any]:
    """features.merge_chunks into ``store_dir`` and BUILD.json last (features.build_raw_store's record plus ``extra``;
    no timestamp, so a rerun reproduces its bytes).  An existing BUILD.json is kept only if it names these chunks."""

    store_dir, chunk_dirs = Path(store_dir), [Path(d) for d in chunk_dirs]
    chunk_hashes = [sha256_file(d / "CHUNK.json") for d in chunk_dirs]
    build_path = store_dir / "BUILD.json"
    if build_path.is_file():
        build = es._read(build_path)
        if build.get("chunk_records_sha256") != chunk_hashes:
            raise Blocked(f"{build_path} was built from other chunks; it is never overwritten")
        return {"record": build, "entry": _entry(build_path), "status": "kept"}
    if store_dir.exists():
        store_dir.rename(store_dir.with_name(f"{store_dir.name}.partial_{_stamp()}"))
    try:
        record = qf.merge_chunks(chunk_dirs, store_dir)
    except ValueError as error:
        raise Blocked(f"merging the chunks: {error}") from error
    record["code_sha256"] = sha256_file(ROOT / "studies/q_production_20261005/features.py")
    record.update(extra)
    entry = _write_json(build_path, record)
    return {"record": record, "entry": entry, "status": "written"}


def store_digest(store_dir: Path) -> dict[str, Any]:
    entries = tree_hashes(store_dir)
    return {"files": len(entries), "sha256": tree_digest(entries), "rule": "sha256 of the sorted lines '<file>:<sha256>\\n'"}


# ----------------------------------------------------------------------------- stage: store path check
def check_sample(engagement_key: Any, role: Any, matched: Any, *, per_role: int = CHECK_PER_ROLE,
                 salt: str = CHECK_SALT) -> np.ndarray:
    """q_run_1 rows of the check: per public role the ``per_role`` matched engagements with the smallest
    SHA-256('<salt>:<engagement key>'), sorted.  TEST rows are never drawn."""

    keys, role, matched = np.asarray(engagement_key), np.asarray(role).astype(np.int64), np.asarray(matched, dtype=bool)
    picked = []
    for name, code in PUBLIC_ROLES.items():
        rows = np.flatnonzero((role == code) & matched)
        if len(rows) < per_role:
            raise Blocked(f"{name} has fewer than {per_role} matched engagements")
        digests = np.asarray([hashlib.sha256(f"{salt}:{bytes(k).decode('ascii')}".encode("utf-8")).hexdigest() for k in keys[rows]])
        picked.append(rows[np.argsort(digests, kind="stable")[:per_role]])
    return np.sort(np.concatenate(picked)).astype(np.int64)


def check_part(task: Mapping[str, Any]) -> dict[str, Any]:
    """States at tau of one shard's sampled engagements through Task 14's builder, stored as a state part (worker)."""

    started, tick = utc(), time.perf_counter()
    state_match, state_ms = np.asarray(task["state_match"], dtype=np.int64), np.asarray(task["state_ms"], dtype=np.int64)
    bank = es._open_bank(task)
    try:
        built = es._build_match_states(bank, task, state_match, state_ms)
    except Blocked:
        raise
    except (ValueError, KeyError) as error:        # e.g. the XY reader (e1_gate.xy_at_times) refusing a frame
        raise Blocked(f"{task['directory']}: {type(error).__name__}: {error}") from error
    del bank
    payload = cs.encode_states(built["states"])
    if not np.array_equal(payload["time_ms"], state_ms):
        raise Blocked(f"{task['directory']}: encoded state times differ from the requested times")
    payload.update(built["audit"])
    payload.update({"frame_ms": built["frame_ms"], "xy": built["xy"], "xy_support": built["xy_support"],
                    "state_id": np.arange(len(state_ms), dtype=np.int64), "state_match": state_match, "state_ms": state_ms,
                    "identity_json": es.identity_array(task["identity"])})
    entry = _keep_or_write(Path(task["path"]), es.npz_bytes(payload, compress=True))
    return {"shard": int(task["shard"]), "path": entry["path"], "sha256": entry["sha256"], "status": entry["status"],
            "started_utc": started, "seconds": time.perf_counter() - tick, "peak_memory_bytes": peak_memory_bytes(), "pid": os.getpid()}


def engagement_endpoints(engagements: Mapping[str, np.ndarray], q_rows: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """ENGAGEMENTS.npz columns of the q rows (the engagement rows of a q role with valid_h90, in order: rows.py:113-116);
    an identity difference is BLOCKED (row-table key mismatch).  No label column is read."""

    keep = np.isin(np.asarray(engagements["role"]).astype(np.int64), list(Q_ROLES.values())) & np.asarray(engagements["valid_h90"], dtype=bool)
    for name in ("canonical_match_key", "component_key", "query_pre", "pre_ms"):
        left, right = np.asarray(engagements[name])[keep], np.asarray(q_rows[name])
        if left.dtype.kind in "iu" and right.dtype.kind in "iu":
            left, right = left.astype(np.int64), right.astype(np.int64)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise Blocked(f"ENGAGEMENTS.npz differs from Q_ROWS_h90.npz in {name} (row-table key mismatch)")
    return {name: np.asarray(engagements[name])[keep].astype(np.int64) for name in ("query_pre", "query_h90", "post_ms_h90",
                                                                                     "pre_ms", "F_pre_ms")}


def _public_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """An e1_gate.mismatch_report of a TRAIN/Q_DEV/Q_CAL sample without its empty TEST entry."""

    roles = dict(report["roles"])
    if roles.pop("TEST", {"rows": 0}).get("rows"):
        raise Blocked("the check sample holds a TEST row")
    return {**report, "roles": roles}


def check_stage(paths: Paths = Paths(), *, workers: int = MAX_WORKERS, per_role: int = CHECK_PER_ROLE,
                expected: Mapping[str, str] = EXPECTED) -> int:
    """The store path check (non-TEST sample); E1_STORE_CHECK.json."""

    from studies.tog_experiments_20261012 import e1_gate as gate

    target = paths.out / CHECK_STAGE
    if _stage_done(target, "check"):
        return 0
    started, tick = utc(), time.perf_counter()
    _require(paths, ROWS_STAGE)
    sources = _sources(paths, expected)
    q = _load_fields(paths.q_rows, ("engagement_key", "canonical_match_key", "component_key", "match_index", "role", "query_pre",
                                    "pre_ms"))
    windows, _ = es.load_windows(paths.windows)
    sample = check_sample(q["engagement_key"], q["role"], np.asarray(windows["s_ms"]) >= 0, per_role=per_role)
    endpoints = engagement_endpoints(_load_fields(paths.engagements, ENGAGEMENT_COLUMNS), q)
    q_prepared = es._read(paths.q_inputs / "PREPARED.json")
    q_files = {"xy": _entry(paths.q_inputs / "XY.npy"), "store_build": _entry(paths.q_store / "BUILD.json"),
               "prepared": _entry(paths.q_inputs / "PREPARED.json"), "objective_counts": _entry(paths.objective_counts),
               "engagements": _entry(paths.engagements)}
    problems = []
    if q_prepared.get("rows_sha256") != sources["hashes"]["q_rows"]:
        problems.append("q_run_1 PREPARED.json was made from another row table")
    if q_prepared.get("xy_sha256") != q_files["xy"]["sha256"] or q_prepared.get("store_build_sha256") != q_files["store_build"]["sha256"]:
        problems.append("q_run_1 XY.npy or raw/store/BUILD.json differs from q_run_1 PREPARED.json (PRESPEC_E1 section 7)")
    if problems:
        raise Blocked("; ".join(problems))
    projection, projection_sha = es._projection()
    match, tau = q["match_index"][sample].astype(np.int64), q["pre_ms"][sample].astype(np.int64)
    if not np.array_equal(np.asarray(projection.matches["key"])[match], q["canonical_match_key"][sample]):
        raise Blocked("MATCH_TABLE keys differ from the sampled rows' match keys")
    shard = np.asarray(projection.matches["shard"])[match].astype(np.int64)
    tasks = []
    for value in np.unique(shard).tolist():
        own = np.flatnonzero(shard == value)
        state_match, state_ms, inverse = es.unique_states(match[own], tau[own])
        binding = dict(projection.binding(int(value)))
        tasks.append({"shard": int(value), "directory": binding["directory"], "binding": binding,
                      "source_root": str(projection.source_root), "path": str(paths.check / "parts" / f"part_{int(value):04d}.npz"),
                      "state_match": state_match, "state_ms": state_ms, "rows": own, "state_of_row": inverse,
                      "local": {int(m): int(projection.matches["local_match"][m]) for m in np.unique(state_match)},
                      "keys": {int(m): projection.key(int(m)) for m in np.unique(state_match)},
                      "identity": {"version": VERSION, "kind": "store_path_check_part", "shard": int(value),
                                   "bank_binding": binding, "plan_sha256": es.plan_digest({"state_match": state_match, "state_ms": state_ms})}})
    stages = [es._stage("load_and_sample", started, tick)]
    s, t = utc(), time.perf_counter()
    with es._executor(workers) as pool:
        part_results = es.run_tasks(check_part, tasks, pool, "check_parts")
        stages.append(es._stage("tau_states", s, t, **es._workers_record(part_results)))
        s, t = utc(), time.perf_counter()
        items = [{"path": result["path"], "sha256": result["sha256"], "local": task["state_of_row"], "keys": task["rows"],
                  "match": match[task["rows"]], "ms": tau[task["rows"]]} for task, result in zip(tasks, part_results)]
        chunks = chunk_tasks(chunk_plan(items), paths.check / "chunks",
                             {"version": VERSION, "kind": "store_path_check", "sample_sha256": hashlib.sha256(sample.tobytes()).hexdigest()})
        chunk_results = es.run_tasks(store_chunk, chunks, pool, "check_store")
    merged = merge_store([task["chunk_dir"] for task in chunks], paths.check / "store",
                         {"builder": f"{VERSION}: store path check (states at tau, Task 14's builder, this module's chunk writer)",
                          "keys": "check sample position", "sample_rule": f"smallest sha256('{CHECK_SALT}:<engagement key>')"})
    stages.append(es._stage("check_store", s, t, **es._workers_record(chunk_results)))
    s, t = utc(), time.perf_counter()
    rebuilt, stored = qf.RawStore(paths.check / "store"), qf.RawStore(paths.q_store)
    m = len(sample)
    if not np.array_equal(rebuilt.queries, np.arange(m)):
        raise Blocked("the check store's keys are not the sample positions")
    roles = q["role"][sample].astype(np.int64)
    store_report = _public_report(gate.mismatch_report(gate.field_mismatches(
        gate.store_fields(rebuilt, np.arange(m)), gate.store_fields(stored, stored.positions(q["query_pre"][sample]))), roles, sample))
    xy = np.full((m, 10, 2), np.nan)
    support, frame, objective = np.zeros(m, dtype=bool), np.full(m, -1, dtype=np.int64), np.zeros((m, 9), dtype=np.int64)
    for task, result in zip(tasks, part_results):
        with np.load(result["path"], allow_pickle=False) as part:
            local = task["state_of_row"]
            xy[task["rows"]], support[task["rows"]] = part["xy"][local], part["xy_support"][local]
            frame[task["rows"]] = part["frame_ms"][local]
            objective[task["rows"]] = part["feature_objective_counts"][local]
    source_xy = np.load(paths.q_inputs / "XY.npy", mmap_mode="r", allow_pickle=False)
    counts = np.load(paths.objective_counts, mmap_mode="r", allow_pickle=False)
    other_report = _public_report(gate.mismatch_report(gate.field_mismatches(
        {"xy": xy, "xy_support": support, "frame_ms": frame, "objective_counts": objective},
        {"xy": np.asarray(source_xy[sample], dtype=np.float64), "xy_support": np.ones(m, dtype=bool),
         "frame_ms": endpoints["F_pre_ms"][sample],
         "objective_counts": np.asarray(counts[endpoints["query_pre"][sample]], dtype=np.int64)}), roles, sample))
    stages.append(es._stage("compare", s, t))
    if store_report["mismatched_rows"] or other_report["mismatched_rows"]:
        raise Blocked(f"the store path check found mismatches: store {store_report['mismatched_fields']} "
                      f"({store_report['mismatched_rows']} rows); XY/frame/objective {other_report['mismatched_fields']} "
                      f"({other_report['mismatched_rows']} rows)")
    files = {"STORE_PATH_CHECK_h90/store/BUILD.json": merged["entry"],
             "STORE_PATH_CHECK_h90/store/CHUNK.json": _entry(paths.check / "store" / "CHUNK.json"),
             **{f"STORE_PATH_CHECK_h90/parts/{Path(r['path']).name}": {"path": r["path"], "sha256": r["sha256"]} for r in part_results}}
    record = {
        "version": VERSION, "experiment": "E1", "stage": "store_path_check", "passed": True, "started_utc": started, "utc": utc(),
        "seed": None, "files": files,
        "sample": {"rule": f"per TRAIN/Q_DEV/Q_CAL role the {per_role} matched engagements with the smallest "
                           f"sha256('{CHECK_SALT}:<engagement key>')", "salt": CHECK_SALT, "per_role": per_role,
                   "rows": {name: int((roles == code).sum()) for name, code in PUBLIC_ROLES.items()},
                   "matches": {name: int(len(np.unique(match[roles == code]))) for name, code in PUBLIC_ROLES.items()},
                   "q_run_1_rows": sample},
        "path": "states at tau by e1_states._build_match_states (Task 14's builder) -> controls_states.encode_states -> state "
                "part -> this module's store_chunk (decode_states -> semantic_rows -> features.RawStoreWriter) -> "
                "features.merge_chunks, the placebo store's path",
        "store": {"reference": str(paths.q_store), "comparison": "e1_gate.store_fields / field_mismatches (numbers bitwise, "
                  "bags by token text)", **store_report},
        "xy_frame_objective": {"reference": "q_run_1 inputs/main_h90/XY.npy, ENGAGEMENTS F_pre_ms, projection OBJECTIVE_COUNTS.npy "
                               "at query_pre", **other_report},
        "bank_shards": len(tasks), "inputs": {**_input_entries(sources), **q_files, "projection_files": projection_sha},
        **_common(paths), "stages": stages,
        "peak_memory_bytes": {"driver": peak_memory_bytes(),
                              "workers_max": max((r["peak_memory_bytes"] or 0 for r in [*part_results, *chunk_results]), default=None)}}
    write_once(target, es._json_safe(record))
    print(json.dumps({"stage": "check", "record": str(target), "rows": record["sample"]["rows"],
                      "mismatched_rows": {"store": store_report["mismatched_rows"], "xy_frame_objective": other_report["mismatched_rows"]},
                      "fields_compared": len(store_report["fields_compared"])}), flush=True)
    return 0


# ----------------------------------------------------------------------------- stage: placebo store and XY
def store_stage(paths: Paths = Paths(), *, workers: int = MAX_WORKERS, expected: Mapping[str, str] = EXPECTED) -> int:
    """The placebo raw store from the start states and the placebo XY; E1_STORE.json."""

    target = paths.out / STORE_STAGE
    if _stage_done(target, "store"):
        return 0
    started, tick = utc(), time.perf_counter()
    rows_record = _require(paths, ROWS_STAGE)
    _require(paths, CHECK_STAGE)
    sources = _sources(paths, expected)
    plc = _load_fields(paths.plc / ROWS_NAME, ("match_index", "role", "pre_ms", "query_pre"))
    mapping = _load_fields(paths.plc / MAP_NAME, ("window_id", "plc_row"))
    pred = _load_fields(paths.pred, ("window_id", "s_ms", "end_ms"))
    windows, _ = es.load_windows(paths.windows)
    n = len(plc["query_pre"])
    if (not np.array_equal(plc["query_pre"], np.arange(n)) or not np.array_equal(mapping["plc_row"], np.arange(n))
            or not np.array_equal(mapping["window_id"], pred["window_id"]) or not np.array_equal(pred["s_ms"], plc["pre_ms"])):
        raise Blocked("the placebo table, its key map and PRED_PLACEBO_h90 do not align")
    states_record = sources["states_record"]
    es._verify_parts(paths.states_dir, states_record)
    w = mapping["window_id"]
    index = part_index(paths.states_dir, states_record, w, plc["match_index"], pred["s_ms"], pred["end_ms"],
                       np.asarray(windows["start_age_plc_ms"])[w])
    role = plc["role"].astype(np.int64)
    finite = index["xy_support"] & np.isfinite(index["xy"]).all(axis=(1, 2))
    if not finite.all():
        bad = {name: int((~finite & (role == code)).sum()) for name, code in PUBLIC_ROLES.items()}
        raise Blocked(f"a matched window's F(s) XY is not finite: {bad}; TEST {'failed' if (~finite & (role == TEST)).any() else 'passed'}")
    stages = [es._stage("load_and_index", started, tick)]
    s, t = utc(), time.perf_counter()
    xy_entry = _keep_or_write(paths.plc / "inputs" / VARIANT / "XY.npy", npy_bytes(index["xy"].astype(np.float64)))
    items = []
    for k, part in enumerate(index["files"]):
        own = np.flatnonzero(index["part"] == k)
        items.append({"path": str(paths.states_dir / part["file"]), "sha256": part["sha256"], "local": index["local_start"][own],
                      "keys": own, "match": plc["match_index"][own], "ms": plc["pre_ms"][own]})
    identity = {"version": VERSION, "kind": "placebo_store", "plc_rows_sha256": rows_record["files"]["plc/Q_ROWS_h90.npz"]["sha256"],
                "states_record_sha256": sources["hashes"]["states_record"]}
    tasks = chunk_tasks(chunk_plan(items), paths.plc / "raw" / "chunks", identity)
    stages.append(es._stage("xy", s, t))
    s, t = utc(), time.perf_counter()
    with es._executor(workers) as pool:
        results = es.run_tasks(store_chunk, tasks, pool, "store")
    stages.append(es._stage("chunks", s, t, chunks_reused=sum(bool(r["reused"]) for r in results), **es._workers_record(results)))
    s, t = utc(), time.perf_counter()
    merged = merge_store([task["chunk_dir"] for task in tasks], paths.plc / "raw" / "store",
                         {"builder": f"{VERSION}: placebo start states (STATES_h90 parts) -> controls_states.decode_states -> "
                                     "semantic_rows -> features.RawStoreWriter chunks -> features.merge_chunks",
                          "keys": "placebo row index = plc/Q_ROWS_h90.npz query_pre", **identity})
    store = qf.RawStore(paths.plc / "raw" / "store")
    if store.n != n or not np.array_equal(store.queries, np.arange(n)):
        raise Blocked("the placebo raw store's keys are not 0..n-1")
    digest = store_digest(paths.plc / "raw" / "store")
    stages.append(es._stage("merge", s, t))
    files = {"plc/inputs/main_h90/XY.npy": {"path": xy_entry["path"], "sha256": xy_entry["sha256"]},
             "plc/raw/store/BUILD.json": merged["entry"], "plc/raw/store/CHUNK.json": _entry(paths.plc / "raw" / "store" / "CHUNK.json")}
    record = {
        "version": VERSION, "experiment": "E1", "stage": "placebo_store", "passed": True, "started_utc": started, "utc": utc(),
        "seed": None, "files": files, "store_tree": digest, "store_tree_sha256": digest["sha256"],
        "chunks": {"parts_per_chunk": PARTS_PER_CHUNK, "chunks": len(tasks), "dir": str(paths.plc / "raw" / "chunks")},
        "store_rule": "keys 0..n-1 in placebo row order; each row the start state's semantic row; features.merge_chunks; "
                      "BUILD.json = merge record + features.py SHA-256 + provenance (as features.build_raw_store)",
        "xy_rule": "F(s) XY of each placebo row's start state from its STATES_h90 part (e1_gate.xy_at_times at build, Task 14), "
                   "float64 [n, 10, 2], finite for every row of every role (checked; a failure is BLOCKED)",
        "checks": {"passed": True, "list": ["STATES_h90 parts equal their record", "every placebo row in exactly one part",
                                            "start and end states = (match, s) and (match, s + D)",
                                            "held start frame = s - start_age_plc_ms", "F(s) XY finite (all roles)",
                                            "chunk keys one contiguous partition", "merged store keys = 0..n-1"]},
        "inputs": {"plc_rows": rows_record["files"]["plc/Q_ROWS_h90.npz"], "key_map": rows_record["files"]["plc/PLC_KEY_MAP_h90.npz"],
                   "states_record": {"path": str(paths.states_record), "sha256": sources["hashes"]["states_record"]},
                   "states_parts_list_sha256": states_record.get("parts_list_sha256"),
                   "pred": {"path": str(paths.pred), "sha256": sources["hashes"]["pred"]},
                   "windows": {"path": str(paths.windows), "sha256": sources["hashes"]["windows"]}},
        "test_note": "the store, XY and BUILD.json hold every role's rows (q_prepare and the workers need them); BUILD.json and "
                     "CHUNK.json carry the frozen total row count, which nobody reads back into a TEST count",
        **_common(paths), "stages": stages,
        "peak_memory_bytes": {"driver": peak_memory_bytes(), "workers_max": es._workers_record(results)["worker_peak_memory_bytes_max"]}}
    write_once(target, es._json_safe(record))
    print(json.dumps({"stage": "store", "record": str(target), "build_sha256": merged["entry"]["sha256"], "xy_sha256": xy_entry["sha256"],
                      "store_tree_sha256": digest["sha256"]}), flush=True)
    return 0


# ----------------------------------------------------------------------------- stage: engagement inputs
def subset_xy(source_xy: Any, source_keys: Any, keys: Any, window_id: Any) -> np.ndarray:
    """q_run_1 XY rows of the engagement rows, found by engagement_key; the rows must be the engagement rows' q_run_1 rows
    (window_id) in q_run_1 order and every value finite, else BLOCKED."""

    source_keys, keys, window_id = np.asarray(source_keys), np.asarray(keys), np.asarray(window_id).astype(np.int64)
    shape = tuple(np.shape(source_xy))
    if len(shape) != 3 or shape[0] != len(source_keys) or shape[1:] != (10, 2):
        raise Blocked("the source XY rows do not align with the q_run_1 row table")
    if keys.ndim != 1 or keys.shape != window_id.shape or not len(source_keys):
        raise Blocked("one q_run_1 row per engagement row")
    order = np.argsort(source_keys, kind="stable")
    rows = order[np.minimum(np.searchsorted(source_keys[order], keys), len(order) - 1)]
    if not np.array_equal(source_keys[rows], keys):
        raise Blocked("an engagement row's key is not a q_run_1 row (XY rows by engagement_key)")
    if not np.array_equal(rows, window_id) or np.any(np.diff(rows) <= 0):
        raise Blocked("the XY rows found by engagement_key are not the engagement rows' q_run_1 rows in order")
    xy = np.ascontiguousarray(np.asarray(source_xy)[rows], dtype=np.float64)
    if not np.isfinite(xy).all():
        raise Blocked("an engagement row's XY is not finite")
    return xy


def eng_inputs_stage(paths: Paths = Paths()) -> int:
    """The junction to q_run_1's raw store and the engagement XY subset; E1_ENG_INPUTS.json."""

    target = paths.out / ENG_STAGE
    if _stage_done(target, "eng_inputs"):
        return 0
    started, tick = utc(), time.perf_counter()
    rows_record = _require(paths, ROWS_STAGE)
    q_prepared = es._read(paths.q_inputs / "PREPARED.json")
    q_rows_sha, build_sha, xy_sha = sha256_file(paths.q_rows), sha256_file(paths.q_store / "BUILD.json"), sha256_file(paths.q_inputs / "XY.npy")
    problems = []
    if q_prepared.get("rows_sha256") != q_rows_sha:
        problems.append("q_run_1 PREPARED.json was made from another row table")
    if build_sha != q_prepared.get("store_build_sha256"):
        problems.append("q_run_1 raw/store/BUILD.json differs from q_run_1 PREPARED.json store_build_sha256 (PRESPEC_E1 section 7)")
    if xy_sha != q_prepared.get("xy_sha256"):
        problems.append("q_run_1 inputs/main_h90/XY.npy differs from q_run_1 PREPARED.json xy_sha256 (PRESPEC_E1 section 7)")
    if problems:
        raise Blocked("; ".join(problems))
    try:
        link = eq.ensure_store_link(paths.eng / "raw" / "store", paths.q_store)
    except (OSError, ValueError) as error:
        raise Blocked(f"the eng raw/store junction: {error}") from error
    link["build_sha256_via_link"] = sha256_file(paths.eng / "raw" / "store" / "BUILD.json")
    if link["build_sha256_via_link"] != build_sha:
        raise Blocked("BUILD.json read through the junction differs from q_run_1's")
    eng = _load_fields(paths.eng / ROWS_NAME, ("engagement_key",))
    mapping = _load_fields(paths.plc / MAP_NAME, ("window_id", "engagement_key_eng"))
    if not np.array_equal(mapping["engagement_key_eng"], eng["engagement_key"]):
        raise Blocked("the key map's engagement keys are not the engagement table's")
    q_keys = _load_fields(paths.q_rows, ("engagement_key",))["engagement_key"]
    source = np.load(paths.q_inputs / "XY.npy", mmap_mode="r", allow_pickle=False)
    xy = subset_xy(source, q_keys, eng["engagement_key"], mapping["window_id"])
    del source
    xy_entry = _keep_or_write(paths.eng / "inputs" / VARIANT / "XY.npy", npy_bytes(xy))
    files = {"eng/inputs/main_h90/XY.npy": {"path": xy_entry["path"], "sha256": xy_entry["sha256"]}}
    record = {
        "version": VERSION, "experiment": "E1", "stage": "eng_inputs", "passed": True, "started_utc": started, "utc": utc(),
        "seed": None, "files": files, "store_link": link,
        "xy": {"source": str(paths.q_inputs / "XY.npy"), "source_sha256": xy_sha, "source_equals_prepared_xy_sha256": True,
               "subset_sha256": xy_entry["sha256"], "rule": "q_run_1 XY.npy rows of the engagement rows found by engagement_key "
               "(= key map window_id, q_run_1 order); PRESPEC_E1 section 3 실행 세부"},
        "q_run_1": {"prepared": _entry(paths.q_inputs / "PREPARED.json"), "rows_sha256": q_rows_sha, "store_build_sha256": build_sha},
        "inputs": {"eng_rows": rows_record["files"]["eng/Q_ROWS_h90.npz"], "key_map": rows_record["files"]["plc/PLC_KEY_MAP_h90.npz"]},
        "write_note": "nothing is written through the junction; q_prepare reads raw/store only (q_prepare.py:119-120)",
        **_common(paths), "stages": [es._stage("eng_inputs", started, tick)], "peak_memory_bytes": {"driver": peak_memory_bytes()}}
    write_once(target, es._json_safe(record))
    print(json.dumps({"stage": "eng_inputs", "record": str(target), "xy_sha256": xy_entry["sha256"], "junction": link["realpath"]}),
          flush=True)
    return 0


# ----------------------------------------------------------------------------- stage: q_prepare
def tree_hashes(root: Path) -> dict[str, dict[str, Any]]:
    """Every file under ``root`` (relative path -> size and SHA-256), sorted."""

    root = Path(root)
    return {path.relative_to(root).as_posix(): {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(p for p in root.rglob("*") if p.is_file())}


def tree_digest(entries: Mapping[str, Mapping[str, Any]]) -> str:
    return hashlib.sha256("".join(f"{name}:{entries[name]['sha256']}\n" for name in sorted(entries)).encode("utf-8")).hexdigest()


def tree_changes(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]) -> list[str]:
    changes = []
    for name in sorted(set(before) | set(after)):
        if name not in after:
            changes.append(f"removed {name}")
        elif name not in before:
            changes.append(f"added {name}")
        elif dict(before[name]) != dict(after[name]):
            changes.append(f"changed {name}")
    return changes


def q_prepare_command(run_dir: Path) -> list[str]:
    """The frozen CLI (q_prepare.py docstring) with the release this task names, run from the repository root."""

    return [sys.executable, "-B", PREPARE_SCRIPT, "--run", Path(run_dir).as_posix(), "--variant", VARIANT, "--release", RELEASE_ARGUMENT]


def prepare_env() -> dict[str, str]:
    """The q supervisor's worker environment (e4_q.worker_env, threads 2) with CUDA hidden by -1 (CPU only)."""

    env = dict(eq.worker_env())
    env["CUDA_VISIBLE_DEVICES"] = es.HIDE_CUDA
    return env


def run_q_prepare(run_dir: Path, *, poll_seconds: float = 0.5) -> dict[str, Any]:
    """q_prepare.py as a subprocess; stdout and stderr to <run>/logs (never printed: the stdout holds the frozen total row
    count); the child's peak working set sampled while it runs."""

    logs = Path(run_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    command = q_prepare_command(run_dir)
    stdout, stderr = logs / f"q_prepare_{stamp}.stdout.log", logs / f"q_prepare_{stamp}.stderr.log"
    started, tick, peak = utc(), time.perf_counter(), None
    with stdout.open("wb") as out_stream, stderr.open("wb") as err_stream:
        process = subprocess.Popen(command, cwd=str(ROOT), env=prepare_env(), stdout=out_stream, stderr=err_stream)
        try:
            import psutil

            handle = psutil.Process(process.pid)
        except Exception:  # noqa: BLE001  (a missing psutil only drops the memory figure)
            handle = None
        while process.poll() is None:
            if handle is not None:
                try:
                    info = handle.memory_info()
                    peak = max(peak or 0, int(getattr(info, "peak_wset", 0) or info.rss))
                except Exception:  # noqa: BLE001  (the child may have just exited)
                    pass
            time.sleep(poll_seconds)
    return {"command": command, "cwd": str(ROOT), "returncode": process.returncode, "stdout_log": str(stdout),
            "stderr_log": str(stderr), "stderr_bytes": stderr.stat().st_size, "started_utc": started, "finished_utc": utc(),
            "wall_seconds": time.perf_counter() - tick, "peak_memory_bytes_sampled": peak,
            "env": {name: prepare_env().get(name) for name in ("CUDA_VISIBLE_DEVICES", *eq.THREAD_VARIABLES)}}


def prepared_problems(prepared: Mapping[str, Any], expect: Mapping[str, Any]) -> list[str]:
    """PREPARED.json of one folder against what this task wrote: the row table, the XY, the raw store's BUILD.json, the
    release, and the two scopes (FIT = non-STOP TRAIN rows, REFIT_TRAIN = all TRAIN rows)."""

    problems = []
    if prepared.get("rows_file") != ROWS_NAME:
        problems.append("PREPARED.json rows_file is not Q_ROWS_h90.npz")
    if prepared.get("rows_sha256") != expect["rows_sha256"]:
        problems.append("PREPARED.json rows_sha256 is not the row table's")
    if prepared.get("xy_file") != f"inputs/{VARIANT}/XY.npy":
        problems.append("PREPARED.json xy_file is not inputs/main_h90/XY.npy")
    if prepared.get("xy_sha256") != expect["xy_sha256"]:
        problems.append("PREPARED.json xy_sha256 is not the XY written before")
    if prepared.get("store_build_sha256") != expect["store_build_sha256"]:
        problems.append("PREPARED.json store_build_sha256 is not the raw store's BUILD.json")
    if prepared.get("release_digest") != RELEASE_DIGEST:
        problems.append("PREPARED.json was made under another q release")
    if prepared.get("variant") != VARIANT:
        problems.append("PREPARED.json variant is not main_h90")
    contexts = prepared.get("contexts") or {}
    if sorted(contexts) != ["FIT", "REFIT_TRAIN"]:
        return problems + ["PREPARED.json does not list FIT and REFIT_TRAIN"]
    if (contexts["FIT"] or {}).get("scope_rows") != expect["fit_rows"]:
        problems.append("FIT scope rows are not the non-STOP TRAIN rows")
    if (contexts["REFIT_TRAIN"] or {}).get("scope_rows") != expect["train_rows"]:
        problems.append("REFIT_TRAIN scope rows are not the TRAIN rows")
    return problems


def context_record(inputs: Path, prepared: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Each scope's CONTEXT.json, CONTEXT.npz and bundle files with their hashes; CONTEXT.json must name its files and the
    same rows, XY and store as PREPARED.json."""

    record, problems = {}, []
    for scope in ("FIT", "REFIT_TRAIN"):
        directory = Path(inputs) / scope
        context = es._read(directory / "CONTEXT.json")
        files = {name: sha256_file(directory / name) for name in eq.CONTEXT_FILES}
        files["CONTEXT.json"] = sha256_file(directory / "CONTEXT.json")
        if (context.get("context_npz_sha256") != files["CONTEXT.npz"] or context.get("bundle_manifest_sha256") != files["bundle/MANIFEST.json"]
                or files["CONTEXT.json"] != prepared["contexts"][scope]["context_json_sha256"]):
            problems.append(f"{scope}: CONTEXT.json does not name its files or differs from PREPARED.json")
        problems += [f"{scope}: CONTEXT.json {key} differs from PREPARED.json" for key in ("rows_sha256", "xy_sha256", "store_build_sha256")
                     if context.get(key) != prepared.get(key)]
        record[scope] = {"files": files, "scope_rows": context.get("scope_rows"), "sigma": context.get("sigma"),
                         "scope_positions_sha256": context.get("scope_positions_sha256"), "fit_seconds": context.get("fit_seconds")}
    return record, problems


def prepare_stage(paths: Paths = Paths(), *, runner: Callable[[Path], Mapping[str, Any]] | None = None) -> int:
    """q_prepare for the placebo and the engagement folder with q_run_1/raw/store hashed before and after; E1_PREPARE.json."""

    target = paths.out / PREPARE_STAGE
    if _stage_done(target, "prepare"):
        return 0
    started, tick = utc(), time.perf_counter()
    rows_record, store_record, eng_record = (_require(paths, ROWS_STAGE), _require(paths, STORE_STAGE), _require(paths, ENG_STAGE))
    try:
        release = qrelease.verify(paths.release)
    except RuntimeError as error:
        raise Blocked(f"release: {error}") from error
    if release["digest"] != RELEASE_DIGEST:
        raise Blocked(f"release digest {release['digest']} is not {RELEASE_DIGEST}")
    try:
        link = eq.ensure_store_link(paths.eng / "raw" / "store", paths.q_store)
    except (OSError, ValueError) as error:
        raise Blocked(f"the eng raw/store junction: {error}") from error
    if link["created"]:
        raise Blocked("the eng raw/store junction was missing (the eng_inputs stage makes it)")
    expectations = {}
    for arm, folder, record, xy_label in ((PLC, paths.plc, store_record, "plc/inputs/main_h90/XY.npy"),
                                          (ENG, paths.eng, eng_record, "eng/inputs/main_h90/XY.npy")):
        table = _load_fields(folder / ROWS_NAME, ("role", "q_stop"))
        train = table["role"].astype(np.int64) == TRAIN
        build = (store_record["files"]["plc/raw/store/BUILD.json"]["sha256"] if arm == PLC
                 else eng_record["store_link"]["build_sha256_via_link"])
        expectations[arm] = {"rows_sha256": rows_record["files"][f"{arm}/Q_ROWS_h90.npz"]["sha256"],
                             "xy_sha256": record["files"][xy_label]["sha256"], "store_build_sha256": build,
                             "train_rows": int(train.sum()), "fit_rows": int((train & ~table["q_stop"].astype(bool)).sum())}
    stages = [es._stage("checks", started, tick)]
    s, t = utc(), time.perf_counter()
    protected = {"q_run_1/Q_ROWS_h90.npz": paths.q_rows, "q_run_1/inputs/main_h90/XY.npy": paths.q_inputs / "XY.npy",
                 "q_run_1/inputs/main_h90/PREPARED.json": paths.q_inputs / "PREPARED.json"}
    before = {"store": tree_hashes(paths.q_store), "files": {label: sha256_file(path) for label, path in protected.items()}}
    stages.append(es._stage("hash_q_run_1_before", s, t))
    runs, prepared, contexts, problems = {}, {}, {}, []
    for arm, folder in ((PLC, paths.plc), (ENG, paths.eng)):
        s, t = utc(), time.perf_counter()
        prepared_path = folder / "inputs" / VARIANT / "PREPARED.json"
        if not prepared_path.exists():
            run = dict((runner or run_q_prepare)(folder))
            runs[arm] = run
            if run["returncode"] != 0 or not prepared_path.exists():
                raise Blocked(f"{arm}: q_prepare.py exited with {run['returncode']} (logs: {run['stderr_log']})")
        prepared[arm] = es._read(prepared_path)
        problems += [f"{arm}: {p}" for p in prepared_problems(prepared[arm], expectations[arm])]
        if sha256_file(folder / "inputs" / VARIANT / "XY.npy") != expectations[arm]["xy_sha256"]:
            problems.append(f"{arm}: XY.npy changed during q_prepare")
        contexts[arm], context_problems = context_record(folder / "inputs" / VARIANT, prepared[arm])
        problems += [f"{arm}: {p}" for p in context_problems]
        stages.append(es._stage(f"q_prepare_{arm}", s, t, run=runs.get(arm)))
    s, t = utc(), time.perf_counter()
    after = {"store": tree_hashes(paths.q_store), "files": {label: sha256_file(path) for label, path in protected.items()}}
    changes = tree_changes(before["store"], after["store"]) + [f"changed {label}" for label in protected
                                                              if before["files"][label] != after["files"][label]]
    stages.append(es._stage("hash_q_run_1_after", s, t))
    if changes:
        raise Blocked(f"q_run_1 changed while q_prepare ran: {changes[:10]}")
    if problems:
        raise Blocked("; ".join(problems))
    files = {}
    for arm, folder in ((PLC, paths.plc), (ENG, paths.eng)):
        base = folder / "inputs" / VARIANT
        files[f"{arm}/inputs/main_h90/PREPARED.json"] = _entry(base / "PREPARED.json")
        for scope in ("FIT", "REFIT_TRAIN"):
            for name in ("CONTEXT.json", *eq.CONTEXT_FILES):
                files[f"{arm}/inputs/main_h90/{scope}/{name}"] = _entry(base / scope / name)
    shown = ("variant", "rows_file", "rows_sha256", "xy_file", "xy_sha256", "store_build_sha256", "release_digest", "prepare_version")
    record = {
        "version": VERSION, "experiment": "E1", "stage": "prepare", "passed": True, "started_utc": started, "utc": utc(),
        "seed": None, "files": files, "command": {arm: q_prepare_command(folder) for arm, folder in ((PLC, paths.plc), (ENG, paths.eng))},
        "q_prepare": runs, "release": {"path": str(paths.release), "digest": release["digest"]},
        "prepared": {arm: {key: prepared[arm].get(key) for key in shown} for arm in prepared},
        "prepared_note": "PREPARED.json 'rows' (frozen format: the total row count) is not copied here",
        "contexts": contexts, "expectations": {arm: {k: v for k, v in e.items()} for arm, e in expectations.items()},
        "scopes_rule": "FIT = TRAIN rows outside the q inner STOP matches, REFIT_TRAIN = all TRAIN rows (qbatch.ScopeContext), "
                       "fitted on each folder's own rows (PRESPEC_E1 section 3 입력과 전처리)",
        "q_run_1_untouched": {"store_files": len(after["store"]), "store_tree_sha256_before": tree_digest(before["store"]),
                              "store_tree_sha256_after": tree_digest(after["store"]), "build_json_sha256": after["store"]["BUILD.json"]["sha256"],
                              "other_files": after["files"], "changed": False,
                              "reading": "q_prepare.py writes only under <run>/inputs/<variant> (XY.npy when missing, the scope "
                                         "contexts, PREPARED.json); it opens raw/store read-only through features.RawStore "
                                         "(np.load mmap_mode='r', np.memmap mode='r')"},
        "store_link": link, **_common(paths), "stages": stages,
        "peak_memory_bytes": {"driver": peak_memory_bytes(),
                              "q_prepare_sampled": {arm: run.get("peak_memory_bytes_sampled") for arm, run in runs.items()}}}
    write_once(target, es._json_safe(record))
    print(json.dumps({"stage": "prepare", "record": str(target),
                      "prepared_sha256": {arm: files[f"{arm}/inputs/main_h90/PREPARED.json"]["sha256"] for arm in (PLC, ENG)},
                      "q_run_1_store_unchanged": True}), flush=True)
    return 0


# ----------------------------------------------------------------------------- balance
def smd_summary(engagement: Any, placebo: Any) -> dict[str, Any]:
    """Means, population SDs, pooled SD and SMD (e1_candidates.smd) of paired values."""

    a, b = np.asarray(engagement, dtype=np.float64), np.asarray(placebo, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise Blocked("SMD inputs must be paired finite 1-D values")
    if not len(a):
        return {"pairs": 0, "mean_engagement": None, "mean_placebo": None, "sd_engagement": None, "sd_placebo": None,
                "pooled_sd": None, "smd": None}
    return {"pairs": int(len(a)), "mean_engagement": float(a.mean()), "mean_placebo": float(b.mean()),
            "sd_engagement": float(a.std()), "sd_placebo": float(b.std()), "pooled_sd": math.sqrt((a.var() + b.var()) / 2),
            "smd": ec.smd(a, b)}


def objective_gain(start_counts: Any, end_counts: Any) -> np.ndarray:
    """OBJECTIVE_COUNTS summed at the end minus at the start (diagnostics.py:155-158); a window counts when it is > 0."""

    start, end = np.asarray(start_counts), np.asarray(end_counts)
    if start.shape != end.shape or start.ndim != 2:
        raise Blocked("objective counts must be aligned [n, k] arrays")
    return end.astype(np.int64).sum(axis=1) - start.astype(np.int64).sum(axis=1)


def _share(flags: np.ndarray) -> dict[str, Any]:
    flags = np.asarray(flags, dtype=bool)
    return {"windows": int(len(flags)), "with_objective_gain": int(flags.sum()), "share": float(flags.mean()) if len(flags) else None}


def triggers(smd: Mapping[str, Mapping[str, Any]], reuse: Mapping[str, Any]) -> dict[str, Any]:
    v, t = smd["v_start"]["smd"], smd["game_time_min"]["smd"]
    share = reuse.get("share_rows_in_reused_windows")
    return {"stratified_r": {"rule": "|SMD| > 0.1 on V(start) or game time (PRESPEC_E1 section 3 균형)",
                             "abs_smd_v_start": None if v is None else abs(v), "abs_smd_game_time": None if t is None else abs(t),
                             "flag": bool(any(value is not None and abs(value) > SMD_TRIGGER for value in (v, t)))},
            "deduplicated_r": {"rule": "rows in reused placebo windows > 5 % of the placebo rows (PRESPEC_E1 section 3 균형)",
                               "share_rows_in_reused_windows": share, "flag": bool(share is not None and share > REUSE_TRIGGER)}}


def d_quartiles(d_ms: Any, matched: Any) -> dict[str, Any]:
    d, matched = np.asarray(d_ms, dtype=np.float64) / 1000.0, np.asarray(matched, dtype=bool)
    edges = np.quantile(d, [0.25, 0.5, 0.75], method="linear")
    bins = np.searchsorted(edges, d, side="right")
    return {"edges_s": [float(e) for e in edges], "rule": DEFINITIONS["d_quartiles"],
            "quartiles": {str(k): {"engagements": int((bins == k).sum()), "matched": int(((bins == k) & matched).sum()),
                                   "match_rate": float(((bins == k) & matched).sum() / (bins == k).sum()) if (bins == k).any() else None}
                          for k in range(4)}}


def matched_medians(d_ms: Any, gap_ms: Any, matched: Any) -> dict[str, Any]:
    d, gap, matched = np.asarray(d_ms, dtype=np.float64) / 1000.0, np.asarray(gap_ms, dtype=np.int64), np.asarray(matched, dtype=bool)
    has = gap >= 0

    def median(values: np.ndarray) -> float | None:
        return float(np.median(values)) if len(values) else None

    return {"D_s": {"matched": median(d[matched]), "unmatched": median(d[~matched])},
            "time_since_last_kill_s": {"matched": median(gap[matched & has] / 1000.0), "unmatched": median(gap[~matched & has] / 1000.0),
                                       "matched_without_earlier_kill": int((matched & ~has).sum()),
                                       "unmatched_without_earlier_kill": int((~matched & ~has).sum())}}


def precision_counts(delta: Any) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise Blocked("precision counts need finite 1-D label values")
    size = np.abs(d)
    nonzero = size[size > 0]
    return {"rows": int(len(d)), "exact_zero": int((d == 0).sum()), "abs_lt_1e-6": int((size < TINY_W).sum()),
            "nonzero_abs_lt_1e-6": int(((size > 0) & (size < TINY_W)).sum()),
            "min_nonzero_abs": float(nonzero.min()) if len(nonzero) else None}


def role_balance(windows: Mapping[str, np.ndarray], pairs: Mapping[str, np.ndarray], code: int, *,
                 public: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """(balance block, e1_candidates.role_summary) of one role; ``public`` adds the extras and the precision disclosure."""

    role = np.asarray(windows["role"]).astype(np.int64)
    summary = ec.role_summary(windows, role == code)
    select = np.asarray(pairs["role"]).astype(np.int64) == code
    w = np.asarray(pairs["window_id"], dtype=np.int64)[select]
    if len(w) != summary["matched"] or np.any(np.asarray(windows["s_ms"])[w] < 0):
        raise Blocked(f"role {code}: the pairs are not the role's matched windows")
    gap_e, gap_p = np.asarray(windows["kill_gap_eng_ms"])[w], np.asarray(windows["kill_gap_plc_ms"])[w]
    both = (gap_e >= 0) & (gap_p >= 0)
    smd = {"v_start": smd_summary(np.asarray(pairs["v_eng"])[select], np.asarray(pairs["v_plc"])[select]),
           "game_time_min": smd_summary(np.asarray(windows["tau_ms"])[w] / 60000.0, np.asarray(windows["s_ms"])[w] / 60000.0),
           "gold_difference": smd_summary(np.asarray(windows["gold_eng"])[w], np.asarray(windows["gold_plc"])[w]),
           "start_frame_age_s": smd_summary(np.asarray(windows["start_age_eng_ms"])[w] / 1000.0,
                                            np.asarray(windows["start_age_plc_ms"])[w] / 1000.0),
           "time_since_last_kill_s": {**smd_summary(gap_e[both] / 1000, gap_p[both] / 1000),
                                      "pairs_dropped_without_earlier_kill": int((~both).sum())}}
    obj_e, obj_p = np.asarray(pairs["obj_eng"], dtype=bool)[select], np.asarray(pairs["obj_plc"], dtype=bool)[select]
    block = {"engagements": summary["engagements"], "matched": summary["matched"], "match_rate": summary["match_rate"],
             "pairs": int(select.sum()), "matches": int(np.unique(np.asarray(pairs["match_index"])[select]).size), "smd": smd,
             "objective_share": {"engagement": _share(obj_e), "placebo": _share(obj_p),
                                 "pairs_without_objective_in_either": int((~obj_e & ~obj_p).sum())},
             "bins": summary["bins"], "reuse": summary["reuse"], "triggers": triggers(smd, summary["reuse"])}
    if public:
        rows = role == code
        block["extras"] = {"d_quartiles": d_quartiles(np.asarray(windows["D_ms"])[rows], np.asarray(windows["s_ms"])[rows] >= 0),
                           "medians": matched_medians(np.asarray(windows["D_ms"])[rows], np.asarray(windows["kill_gap_eng_ms"])[rows],
                                                      np.asarray(windows["s_ms"])[rows] >= 0)}
        block["precision"] = {"placebo": precision_counts(np.asarray(pairs["delta_plc"])[select]),
                              "engagement": precision_counts(np.asarray(pairs["delta_eng"])[select])}
    return block, summary


def balance_tables(windows: Mapping[str, np.ndarray], pairs: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(public blocks of TRAIN, Q_DEV and Q_CAL, the sealed TEST block, the public roles' e1_candidates summaries)."""

    public, summaries = {}, {}
    for name, code in PUBLIC_ROLES.items():
        public[name], summaries[name] = role_balance(windows, pairs, code, public=True)
    sealed, _ = role_balance(windows, pairs, TEST, public=False)
    return public, {"TEST": sealed}, summaries


def balance_cross_checks(summaries: Mapping[str, Any], record_roles: Mapping[str, Any]) -> list[str]:
    """The public roles' recomputed window summaries against WINDOWS_h90.json (the same file, the same functions)."""

    problems = []
    for name in PUBLIC_ROLES:
        mine, theirs = json.loads(json.dumps(es._json_safe(summaries[name]))), record_roles.get(name) or {}
        problems += [f"{name}: {key} differs from WINDOWS_h90.json" for key in CROSS_CHECK_KEYS if mine.get(key) != theirs.get(key)]
    return problems


def headline(public: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {name: {"pairs": block["pairs"], "match_rate": block["match_rate"],
                   "smd": {key: value["smd"] for key, value in block["smd"].items()},
                   "objective_share": {arm: block["objective_share"][arm]["share"] for arm in ("engagement", "placebo")},
                   "reuse_share": block["reuse"]["share_rows_in_reused_windows"],
                   "stratified_r_flag": block["triggers"]["stratified_r"]["flag"],
                   "deduplicated_r_flag": block["triggers"]["deduplicated_r"]["flag"]} for name, block in public.items()}


def balance_stage(paths: Paths = Paths(), *, expected: Mapping[str, str] = EXPECTED) -> int:
    """BALANCE.json and BALANCE_TEST_SEALED.json."""

    target = paths.out / BALANCE_NAME
    if _stage_done(target, "balance"):
        return 0
    started, tick = utc(), time.perf_counter()
    rows_record = _require(paths, ROWS_STAGE)
    sources = _sources(paths, expected)
    es.load_windows(paths.windows)
    windows = _load_fields(paths.windows, WINDOW_BALANCE_COLUMNS)
    q = _load_fields(paths.q_rows, ("engagement_key", "canonical_match_key", "component_key", "match_index", "role", "query_pre",
                                    "pre_ms", "pt_p_pre"))
    plc = _load_fields(paths.plc / ROWS_NAME, ("role", "match_index", "label_svi_delta"))
    eng = _load_fields(paths.eng / ROWS_NAME, ("role", "match_index", "label_svi_delta", "engagement_key"))
    mapping = _load_fields(paths.plc / MAP_NAME, ("window_id", "engagement_key_eng"))
    pred = _load_fields(paths.pred, ("window_id", "p_start", "s_ms", "end_ms"))
    w = mapping["window_id"].astype(np.int64)
    if (not np.array_equal(w, pred["window_id"]) or not np.array_equal(mapping["engagement_key_eng"], eng["engagement_key"])
            or not np.array_equal(eng["engagement_key"], q["engagement_key"][w])
            or not np.array_equal(plc["role"].astype(np.int64), q["role"][w].astype(np.int64))):
        raise Blocked("the placebo table, the engagement table, the key map and PRED_PLACEBO_h90 do not align")
    endpoints = engagement_endpoints(_load_fields(paths.engagements, ENGAGEMENT_COLUMNS), q)
    if (not np.array_equal(np.asarray(windows["tau_ms"]).astype(np.int64), endpoints["pre_ms"])
            or not np.array_equal(np.asarray(windows["D_ms"]).astype(np.int64), endpoints["post_ms_h90"] - endpoints["pre_ms"])):
        raise Blocked("WINDOWS_h90 tau or D differs from ENGAGEMENTS pre_ms and post_ms_h90 - pre_ms")
    counts = np.load(paths.objective_counts, mmap_mode="r", allow_pickle=False)
    gain_eng = objective_gain(counts[endpoints["query_pre"]], counts[endpoints["query_h90"]])
    diagnostics = es._read(paths.diagnostics)["objective_events_in_window"]
    q_dev = q["role"].astype(np.int64) == Q_ROLES["Q_DEV"]
    diag_check = {"some": int((gain_eng[q_dev] > 0).sum()), "none": int((gain_eng[q_dev] <= 0).sum()),
                  "record_some": diagnostics["some"]["rows"], "record_none": diagnostics["none"]["rows"]}
    if diag_check["some"] != diag_check["record_some"] or diag_check["none"] != diag_check["record_none"]:
        raise Blocked("the engagement objective share of all Q_DEV engagements differs from labels_1/DIAGNOSTICS_LAYER2.json")
    states_record = sources["states_record"]
    es._verify_parts(paths.states_dir, states_record)
    index = part_index(paths.states_dir, states_record, w, q["match_index"][w], pred["s_ms"], pred["end_ms"],
                       np.asarray(windows["start_age_plc_ms"])[w])
    gain_plc = objective_gain(index["objective_start"], index["objective_end"])
    pairs = {"window_id": w, "role": plc["role"].astype(np.int64), "match_index": plc["match_index"], "v_eng": q["pt_p_pre"][w],
             "v_plc": pred["p_start"], "obj_eng": gain_eng[w] > 0, "obj_plc": gain_plc > 0, "delta_plc": plc["label_svi_delta"],
             "delta_eng": eng["label_svi_delta"]}
    stages = [es._stage("load_and_index", started, tick)]
    s, t = utc(), time.perf_counter()
    public, sealed, summaries = balance_tables(windows, pairs)
    problems = balance_cross_checks(summaries, sources["windows_record"].get("roles") or {})
    problems += [f"{name}: the kill-gap SMD differs from the window summary's" for name in PUBLIC_ROLES
                 if public[name]["smd"]["time_since_last_kill_s"]["smd"] != summaries[name]["kill_gap_balance"]["smd"]]
    if problems:
        raise Blocked("; ".join(problems))
    inputs = {**_input_entries(sources), "engagements": _entry(paths.engagements), "objective_counts": _entry(paths.objective_counts),
              "diagnostics": _entry(paths.diagnostics), "plc_rows": rows_record["files"]["plc/Q_ROWS_h90.npz"],
              "eng_rows": rows_record["files"]["eng/Q_ROWS_h90.npz"], "key_map": rows_record["files"]["plc/PLC_KEY_MAP_h90.npz"],
              "states_parts_list_sha256": states_record.get("parts_list_sha256")}
    sealed_entry = _write_json(paths.out / BALANCE_SEALED_NAME, {
        "version": VERSION, "experiment": "E1", "stage": "balance_test_sealed", "roles": sealed, "definitions": DEFINITIONS,
        "inputs": inputs, "note": "sealed summary (PRESPEC_E1 section 6): the TEST balance; nobody opens it before the E1 lock; "
                                  "no timestamp, so a rerun reproduces its bytes"})
    del sealed
    stages.append(es._stage("balance", s, t))
    record = {
        "version": VERSION, "experiment": "E1", "stage": "balance", "passed": True, "started_utc": started, "utc": utc(), "seed": None,
        "files": {"BALANCE_TEST_SEALED.json": {"path": sealed_entry["path"], "sha256": sealed_entry["sha256"]}},
        "headline": headline(public), "roles": public, "definitions": DEFINITIONS,
        "cross_checks": {"passed": True, "windows_record": f"{list(CROSS_CHECK_KEYS)} of every public role equal WINDOWS_h90.json",
                         "kill_gap_smd_equals_window_summary": True, "q_dev_objective_share_equals_diagnostics": diag_check,
                         "tau_and_D_equal_engagements": True},
        "not_here": "r_strat and the de-duplicated r are computed in Task 16 (they need the model losses)",
        "test_note": "TEST balance only in BALANCE_TEST_SEALED.json (hash above); no TEST figure here", "inputs": inputs,
        **_common(paths), "stages": stages, "peak_memory_bytes": {"driver": peak_memory_bytes()}}
    write_once(target, es._json_safe(record))
    print(json.dumps({"stage": "balance", "record": str(target), "sealed_sha256": sealed_entry["sha256"],
                      "headline": es._json_safe(record["headline"])}), flush=True)
    return 0


# ----------------------------------------------------------------------------- stage: run record
def record_stage(paths: Paths = Paths()) -> int:
    """E1_ROWS_RUN.json, after every stage passed and its files are unchanged."""

    target = paths.out / RUN_RECORD
    if target.exists():
        print(json.dumps({"stage": "record", "done": str(target)}), flush=True)
        return 0
    started, tick = utc(), time.perf_counter()
    names = (ROWS_STAGE, CHECK_STAGE, STORE_STAGE, ENG_STAGE, PREPARE_STAGE, BALANCE_NAME)
    records = {name: _require(paths, name) for name in names}
    digest = store_digest(paths.plc / "raw" / "store")
    if digest["sha256"] != records[STORE_STAGE]["store_tree_sha256"]:
        raise Blocked("the placebo raw store changed after the store stage")
    try:
        link = eq.ensure_store_link(paths.eng / "raw" / "store", paths.q_store)
    except (OSError, ValueError) as error:
        raise Blocked(f"the eng raw/store junction: {error}") from error
    run = {
        "version": VERSION, "experiment": "E1", "stage": "rows_run", "passed": True,
        "started_utc": records[ROWS_STAGE]["started_utc"], "utc": utc(), "seed": None,
        "seed_note": f"no randomness; the store path check sample is the SHA-256 rule with salt {CHECK_SALT}",
        "counts": records[ROWS_STAGE]["counts"], "counts_note": "TRAIN, Q_DEV and Q_CAL only (placebo rows = engagement rows "
                                                                "= matched windows)",
        "stage_records": {name: _entry(paths.out / name) for name in names},
        "outputs": {label: entry for name in names for label, entry in (records[name].get("files") or {}).items()},
        "placebo_store_tree": digest, "eng_store_link": link,
        "stages": {name: {"started_utc": records[name].get("started_utc"), "utc": records[name].get("utc"),
                          "wall_seconds": sum(float(stage.get("wall_seconds") or 0) for stage in records[name].get("stages") or []),
                          "steps": records[name].get("stages"), "peak_memory_bytes": records[name].get("peak_memory_bytes")}
                   for name in names},
        "inputs": records[ROWS_STAGE]["inputs"], "test_note": TEST_NOTE, **_common(paths),
        "record_stage": es._stage("record", started, tick)}
    write_once(target, es._json_safe(run))
    print(json.dumps({"stage": "record", "record": str(target), "record_sha256": sha256_file(target)}), flush=True)
    return 0


# ----------------------------------------------------------------------------- CLI
STAGE_ORDER = ("rows", "check", "store", "eng_inputs", "prepare", "balance", "record")
STAGE_FUNCTIONS: dict[str, Callable[..., int]] = {
    "rows": rows_stage, "check": check_stage, "store": store_stage, "eng_inputs": eng_inputs_stage, "prepare": prepare_stage,
    "balance": balance_stage, "record": record_stage}


def run_stage(name: str, paths: Paths = Paths(), **kwargs: Any) -> int:
    """One stage; a BLOCKED stop writes E1_ROWS_BLOCKED_<stage>_<n>.json and returns 2."""

    try:
        return int(STAGE_FUNCTIONS[name](paths, **kwargs) or 0)
    except Blocked as error:
        return _blocked(paths.out, name, [str(error)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=(*STAGE_ORDER, "run"))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1-{MAX_WORKERS}")
    os.environ["CUDA_VISIBLE_DEVICES"] = es.HIDE_CUDA     # CPU only; inherited by spawned workers and q_prepare
    es._below_normal_priority()
    try:
        from studies.v_production_20261004.worker import keep_awake

        keep_awake()
    except Exception:  # noqa: BLE001  (best effort, as score.main)
        pass
    paths = Paths()
    for stage in (STAGE_ORDER if args.stage == "run" else (args.stage,)):
        status = run_stage(stage, paths, **({"workers": args.workers} if stage in ("check", "store") else {}))
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
