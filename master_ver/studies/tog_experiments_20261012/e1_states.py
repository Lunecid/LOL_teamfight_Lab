"""T1.3 E1 start and end states of the matched placebo windows and their locked V (PRESPEC_E1 section 6 step 3).

Binding specification: reports/tog_experiments_20261012/PRESPEC_E1_KO.md as amended 2026-10-13 and re-approved (commit
7a841db): section 3 "라벨" (V contexts) and "실행 세부", section 6 step 3 and "잠금 전에 보지 않는 것", section 7
(BLOCKED); PLAN_E1_E5_KO.md Task 14; SDD Rulings 24, 27, 28, 49, 50, 51.

Input: tog_e1_1/WINDOWS_h90.npz (Task 13), one row per q_run_1/Q_ROWS_h90.npz row (all q roles, s_ms = -1 unmatched); its
SHA-256 must equal its record WINDOWS_h90.json, else BLOCKED.  Only matched rows (s_ms >= 0) get states and V.
window_id = the row index (= the q_run_1 row index); engagement_key travels with it.

V context (assemble.py:79-114; PRESPEC_E1 section 3 라벨): a TRAIN row the oof context of its fold, a Q_DEV, Q_CAL or
TEST row final, so a match's placebo windows use its engagements' context.  A role outside this rule, a TRAIN fold
outside 0-4, two contexts in one match, or a context that saw the match (CONTEXT_MEMBERSHIP bits 1-16, or a TRAIN match
that is not the context's held-out target, bit 32) is BLOCKED.

Stages (CLI); every file is written once under D:/LOL_Training_20261004/tog_e1_1:
  build  The unique (match, ms) states of the matched windows' starts s and ends s + D, built through the path the
         equality gate checked (controls_states.build_match; tog_e1_1/EQUALITY_GATE_TOG.json passed): one compressed
         part per bank shard, STATES_h90/part_<shard>.npz, holding the stored form controls_states.encode_states writes,
         the query audit arrays, F(t) and F(t) XY by spatial.py's rules (e1_gate.xy_at_times), the shard's windows
         with their start and end state ids, and an embedded identity (part version, code and input hashes, the
         shard's state plan).  Before a part is written, every window's held start frame must equal WINDOWS_h90's
         s - start_age_plc_ms, its frame count in (s, s + D] must equal n_bound, and its F(s) XY must be finite for all
         ten players; any failure is BLOCKED.  Restart: a part whose identity verifies is kept; a part made by other
         code or inputs stops the run (it is never reused and never overwritten).  Last, STATES_h90/STATES_h90.json.
  score  1. Main-label V reproduction (Ruling 51): 200 rows each of TRAIN folds 0-4, Q_DEV and Q_CAL (non-TEST;
            numpy default_rng(20261013)); the state at pre_ms through the same path, scored in the row's context with
            the production batching; it passes iff |V - pt_p_pre| <= 1e-5 on every row (the labels were scored on
            CUDA, this run on CPU; the tolerance of e5_figures.py V_REPRO_ATOL), else BLOCKED before any placebo V.
            Strength checks, reported: the same states in a wrong context (TRAIN rows final, Q_DEV/Q_CAL rows oof0) and
            the states at pre_ms + 1 s in the right context.
         2. V of every unique state: per state part and context, the part's states in (match, time) order in canonical
            1,024-row batches -> controls_states.semantic_rows -> queries.transform (the context's REFIT_TRAIN bundle)
            -> score.LockedV(run_3, context, device="cpu").score: logit, raw probability and calibrated V (float64).
            The batch layout does not depend on the worker count; each unique state is scored once.  The six V must
            be the main labels' V (labels_1/PRED_<ctx>_MANIFEST.json v, except the device).
         3. PRED_PLACEBO_h90.npz: one row per matched window (all roles, window_id order) with V(s) for every role
            (the PT_flex input) and V(s + D) for TRAIN, Q_DEV and Q_CAL.  A TEST row's V(s + D) is NaN there; it sits
            only in PRED_PLACEBO_h90_TEST_SEALED.npz (PRESPEC_E1 section 3 라벨), with a per-row flag for a TEST end
            state that is also a window's start state (whose V is public as that window's V(s)).  V_REPRO_h90.npz holds
            the reproduction rows (non-TEST); PRED_PLACEBO_h90.json is the run record.  TEST end values exist only in
            memory until the sealed file is written.  NPZ bytes are fixed (zip entries with a fixed timestamp), so a
            rerun after an interruption compares an existing output by SHA-256 only and never opens it.

Nothing about TEST is counted, printed, logged or recorded outside the sealed file (PRESPEC_E1 section 6 and "잠금 전에
보지 않는 것", Ruling 24): public records carry TRAIN, Q_DEV and Q_CAL counts only, and TEST checks enter them as
pass/fail.  At most 4 worker processes, CPU only (CUDA hidden before torch loads), below-normal priority.  No label, no
player identifier.

Usage: python -B -m studies.tog_experiments_20261012.e1_states build|score|run [--workers 4]
       Smoke check (never TEST shards, never into tog_e1_1):
       ... run --smoke --out <scratch>/STATES_h90 --shards train_0000,validation_0000 [--repro-per-group 20]
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Callable, Mapping, Sequence  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import controls_states as cs  # noqa: E402
from studies.label_production_20261005 import queries  # noqa: E402
from studies.label_production_20261005.assemble import Q_FINAL_ROLES  # noqa: E402
from studies.q_production_20261005.rows import Q_ROLES  # noqa: E402
from studies.tog_experiments_20261012.common import (  # noqa: E402
    LABELS, Q_RUN, TOG_E1, TRAINING, document_record, git_head, peak_memory_bytes, sha256_file, utc, write_once)

VERSION = "tog_e1_states_v1_20261013"
PART_VERSION = "tog_e1_states_part_v1"      # the build path of a state part; a part made by another path is never reused
RUN_3 = TRAINING / "run_3"
WINDOWS = TOG_E1 / "WINDOWS_h90.npz"
Q_ROWS = Q_RUN / "Q_ROWS_h90.npz"
MEMBERSHIP = cs.PROJECTION / "CONTEXT_MEMBERSHIP.npy"
GATE_RECORD = TOG_E1 / "EQUALITY_GATE_TOG.json"
PRESPEC = ROOT / "reports/tog_experiments_20261012/PRESPEC_E1_KO.md"
AMENDMENT = ROOT / "reports/tog_experiments_20261012/AMENDMENT_E1_KO.md"
STATES_DIR, STATES_RECORD = "STATES_h90", "STATES_h90.json"
PRED_NAME, SEALED_NAME, REPRO_NAME, PRED_RECORD = ("PRED_PLACEBO_h90.npz", "PRED_PLACEBO_h90_TEST_SEALED.npz",
                                                  "V_REPRO_h90.npz", "PRED_PLACEBO_h90.json")
CONTEXTS = tuple(queries.CONTEXTS)           # final, oof0..oof4: code = index (CONTEXT_MEMBERSHIP row order)
TRAIN, TEST = Q_ROLES["TRAIN"], Q_ROLES["TEST"]
ROLE_NAMES = {code: name for name, code in Q_ROLES.items()}
PUBLIC_ROLES = {name: code for name, code in Q_ROLES.items() if code != TEST}     # TRAIN, Q_DEV, Q_CAL
BATCH_ROWS = queries.BATCH_ROWS               # 1,024 (canonical scoring batches, queries.py:39)
MAX_WORKERS = 4
MIN_D_MS, MIN_START_MS = 10_000, 120_000      # PRESPEC_E1 section 3 (D reaches the first kill; s >= 120 s)
REPRO_SEED = 20261013
REPRO_PER_GROUP = 200
REPRO_ATOL = 1e-5                             # Ruling 51: e5_figures.py V_REPRO_ATOL (labels on CUDA, here CPU)
REPORT_ATOL = 1e-6                            # reported: rows over 1e-6
SHIFT_MS = 1_000                              # strength check: the state 1 s after pre_ms
WRONG_CONTEXT = {"TRAIN": "final", "Q_DEV": "oof0", "Q_CAL": "oof0"}    # strength check: a context that is not the row's
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
HIDE_CUDA = "-1"                              # CUDA_VISIBLE_DEVICES: no visible device.  Not "": on Windows an empty value
                                              # deletes the variable, so spawned workers would see every GPU
AUDIT = tuple(cs.AUDIT_FIELDS)                # query_frame_index, query_event_limit, query_item_limit, query_item_ledger_known
WINDOW_COLUMNS = ("engagement_key", "match_index", "role", "tau_ms", "D_ms", "s_ms", "start_age_plc_ms", "n_bound")
Q_COLUMNS = ("engagement_key", "canonical_match_key", "match_index", "role", "fold")
PUBLIC_COLUMNS = ("window_id", "engagement_key", "match_index", "role", "fold", "ctx", "s_ms", "end_ms", "start_state",
                  "end_state")
SCORE_KINDS = (("logit", "logit"), ("raw", "raw"), ("p", "v"))           # output prefix, LockedV.score value
SEALED_COLUMNS = ("window_id", "engagement_key", "ctx", "end_state", "logit_end", "raw_end", "p_end", "end_equals_a_start")
PLAN_STATE = ("state_match", "state_ms", "state_ctx", "state_role", "state_shard")
BUILD_CODE = tuple(dict.fromkeys(("studies/tog_experiments_20261012/e1_states.py", "studies/tog_experiments_20261012/common.py",
                                  "studies/tog_experiments_20261012/e1_gate.py", *cs.CODE_FILES)))
SCORE_CODE = tuple(dict.fromkeys((*BUILD_CODE, "studies/tog_experiments_20261012/e4_vscore.py",
                                  "studies/label_production_20261005/score.py", "studies/label_production_20261005/assemble.py",
                                  "studies/q_production_20261005/qtest.py", "studies/q_production_20261005/rows.py",
                                  "studies/v_full_preprocess_20261003/artifacts.py", "studies/v_model_fit_input_20261003/models.py",
                                  "studies/v_production_20261004/train.py", "studies/v_production_20261004/source.py",
                                  "studies/v_training_gate_20261003/guard.py")))
CONTEXT_RULE = ("context code = index in (final, oof0, oof1, oof2, oof3, oof4): Q_DEV/Q_CAL/TEST rows final (0), a TRAIN "
                "row oof<fold> (1 + fold) of its match's fold (assemble.py:82-85); one context per match; the context "
                "never saw the match: CONTEXT_MEMBERSHIP bits 1-16 are 0 and a TRAIN match carries bit 32 "
                "(assemble.py:109-114, e4_vscore.oof_leaks)")
BUILD_RULE = ("matched windows (WINDOWS_h90 s_ms >= 0); states = the unique (match, ms) among their starts s and ends "
              "s + D, in (match, time) order (state id = position); per bank shard controls_states.build_match at the "
              "match's state times -> controls_states.encode_states (the stored form; semantic rows by "
              "controls_states.semantic_rows); F(t) XY by e1_gate.xy_at_times; checks per window: held start frame = "
              "s - start_age_plc_ms, frame count in (s, s + D] = n_bound, F(s) XY finite")
SCORE_RULE = ("per state part and context, the part's states in (match, time) order cut into 1,024-row batches -> "
              "controls_states.semantic_rows -> queries.transform(<ctx>_REFIT_TRAIN bundle) -> "
              "score.LockedV(run_3, ctx, device='cpu').score -> (logit, raw probability, calibrated V); each unique "
              "state once; window V(s) = V of its start state, V(s + D) = V of its end state")
PART_COLUMNS = {
    "time_ms, feature_*, item_*, nonitem_utf8/offsets, item_utf8/offsets": "controls_states.encode_states (decode with "
                                                                          "controls_states.decode_states)",
    "state_id": "global state id (position in (match, time) order)", "state_match": "projection match index",
    "state_ms": "state time ms (= time_ms)", "state_ctx": "V context code", "state_role": "q role of the match",
    "state_shard": "bank shard index",
    "query_frame_index, query_event_limit, query_item_limit, query_item_ledger_known": "controls_states.build_match query arrays",
    "frame_ms": "held frame time F(t) (-1 none)", "xy, xy_support": "F(t) XY [n, 10, 2] float64 and support (e1_gate.xy_at_times)",
    "window_id, window_start_state, window_end_state": "the shard's matched windows (WINDOWS_h90 row) and their state ids",
    "identity_json": "the part identity (UTF-8 JSON)"}


class Blocked(ValueError):
    """A PRESPEC_E1 section 7 stop or an input that differs from its record; the CLI writes it as a BLOCKED record."""


# ----------------------------------------------------------------------------- V context rule
def context_code(role: int, fold: int) -> int:
    """The V context code (index in CONTEXTS) of a q row: final (0) for Q_DEV, Q_CAL and TEST, oof<fold> (1 + fold) for
    TRAIN (assemble.py:82-85).  A role outside the q roles or a TRAIN fold outside 0-4 raises (PRESPEC_E1 section 7)."""

    role_value, fold_value = int(role), int(fold)
    if role_value in Q_FINAL_ROLES:
        return 0
    if role_value == TRAIN:
        if not 0 <= fold_value < len(CONTEXTS) - 1:
            raise ValueError(f"a TRAIN row has fold {fold_value}, not an OOF fold 0-{len(CONTEXTS) - 2}")
        return 1 + fold_value
    raise ValueError(f"role {role_value} is not a q role (TRAIN 0, Q_DEV 3, Q_CAL 4, TEST 5)")


def context_for(role: int, fold: int) -> str:
    """The V context name of a q row ("final" or "oof<fold>"), by ``context_code``."""

    return CONTEXTS[context_code(role, fold)]


def context_codes(role: Any, fold: Any) -> np.ndarray:
    """``context_code`` per row (int8); any row outside the rule raises."""

    role, fold = np.asarray(role).astype(np.int64), np.asarray(fold).astype(np.int64)
    if role.ndim != 1 or role.shape != fold.shape:
        raise ValueError("one fold per role")
    outside = ~np.isin(role, (*Q_FINAL_ROLES, TRAIN))
    if outside.any():
        raise ValueError(f"role {int(role[outside][0])} is not a q role (TRAIN 0, Q_DEV 3, Q_CAL 4, TEST 5)")
    train = role == TRAIN
    if np.any(train & ((fold < 0) | (fold >= len(CONTEXTS) - 1))):
        raise ValueError(f"a TRAIN row has no OOF fold 0-{len(CONTEXTS) - 2}")
    return np.where(train, 1 + fold, 0).astype(np.int8)


def check_match_contexts(match_index: Any, codes: Any) -> None:
    """The windows of one match share one context (their engagements'), else BLOCKED."""

    match_index, codes = np.asarray(match_index).astype(np.int64), np.asarray(codes).astype(np.int64)
    if not len(match_index):
        return
    pairs = np.unique(np.stack((match_index, codes), axis=1), axis=0)
    if len(np.unique(pairs[:, 0])) != len(pairs):
        raise Blocked("a match's windows do not share one V context (PRESPEC_E1 section 3: one context per match)")


def oof_leaks(codes: Any, role: Any, match_index: Any, membership: np.ndarray) -> np.ndarray:
    """Per row, whether the scoring context saw the match (assemble.py:109-114, through e4_vscore.oof_leaks)."""

    from studies.tog_experiments_20261012.e4_vscore import oof_leaks as leaks

    return leaks(codes, role, match_index, membership)


def check_alignment(windows: Mapping[str, np.ndarray], rows: Mapping[str, np.ndarray]) -> None:
    """The window table is the q row table row for row (window_id = q_run_1 row index); a difference is BLOCKED
    (PRESPEC_E1 section 7, row-table key mismatch)."""

    for name in ("engagement_key", "match_index", "role"):
        left, right = np.asarray(windows[name]), np.asarray(rows[name])
        if left.dtype.kind in "iu" and right.dtype.kind in "iu":
            left, right = left.astype(np.int64), right.astype(np.int64)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise Blocked(f"WINDOWS_h90.npz differs from Q_ROWS_h90.npz in {name}")


# ----------------------------------------------------------------------------- states, batches, the seal
def unique_states(match_index: Any, times_ms: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(state match, state ms, state of each input) for the unique (match, ms) pairs in (match, time) order."""

    match_index, times = np.asarray(match_index).astype(np.int64), np.asarray(times_ms).astype(np.int64)
    if match_index.ndim != 1 or match_index.shape != times.shape:
        raise ValueError("one time per match index")
    if not len(match_index):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    unique, inverse = np.unique(np.stack((match_index, times), axis=1), axis=0, return_inverse=True)
    return unique[:, 0].astype(np.int64), unique[:, 1].astype(np.int64), np.asarray(inverse, dtype=np.int64).reshape(-1)


def state_plan(match_index: Any, s_ms: Any, end_ms: Any) -> dict[str, np.ndarray]:
    """The unique (match, ms) states of windows [s, end] in (match, time) order and each window's start and end state."""

    match_index, s, end = (np.asarray(a).astype(np.int64) for a in (match_index, s_ms, end_ms))
    if not (match_index.shape == s.shape == end.shape) or match_index.ndim != 1:
        raise ValueError("one match, start and end per window")
    if np.any(end <= s):
        raise ValueError("a window ends at or before its start")
    if np.any(s < 0) or np.any(end >= 1 << 31):
        raise ValueError("window times must be nonnegative int32 milliseconds")
    state_match, state_ms, inverse = unique_states(np.r_[match_index, match_index], np.r_[s, end])
    n = len(match_index)
    return {"state_match": state_match, "state_ms": state_ms, "start_state": inverse[:n], "end_state": inverse[n:]}


def end_is_a_start(start_state: Any, end_state: Any) -> np.ndarray:
    """Per window, whether its end state is the start state of a window (the same match and ms)."""

    return np.isin(np.asarray(end_state), np.asarray(start_state))


def canonical_batches(state_ctx: Any, batch_rows: int = BATCH_ROWS) -> list[tuple[int, np.ndarray]]:
    """(context code, positions) of the scoring batches of one state part: per context in code order, the part's states of
    that context in stored ((match, time)) order, cut into batches of ``batch_rows``.  It depends on the part only."""

    codes = np.asarray(state_ctx).astype(np.int64)
    if int(batch_rows) < 1:
        raise ValueError("a batch holds at least one row")
    batches = []
    for code in np.unique(codes).tolist():
        own = np.flatnonzero(codes == code)
        batches += [(int(code), own[start:start + int(batch_rows)]) for start in range(0, len(own), int(batch_rows))]
    return batches


def split_predictions(windows: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray],
                                                                                                    dict[str, np.ndarray]]:
    """(public, sealed) arrays of the matched windows.  Public: identity columns and V(s) (logit, raw, p) for every role,
    V(s + D) for TRAIN, Q_DEV and Q_CAL, NaN for TEST.  Sealed: the TEST rows' V(s + D) and whether their end state is a
    window's start state (PRESPEC_E1 section 3 라벨: TEST V(s + D) only in PRED_PLACEBO_h90_TEST_SEALED.npz)."""

    test = np.asarray(windows["role"]).astype(np.int64) == TEST
    start, end = np.asarray(windows["start_state"], dtype=np.int64), np.asarray(windows["end_state"], dtype=np.int64)
    public = {name: np.asarray(windows[name]) for name in PUBLIC_COLUMNS}
    sealed = {name: np.asarray(windows[name])[test] for name in ("window_id", "engagement_key", "ctx", "end_state")}
    for prefix, source in SCORE_KINDS:
        values = np.asarray(scores[source], dtype=np.float64)
        public[f"{prefix}_start"] = values[start]
        ends = values[end]
        sealed[f"{prefix}_end"] = ends[test]
        public[f"{prefix}_end"] = np.where(test, np.nan, ends)
    sealed["end_equals_a_start"] = end_is_a_start(start, end)[test]
    return public, {name: sealed[name] for name in SEALED_COLUMNS}


def public_counts(role: Any, match_index: Any) -> dict[str, dict[str, int]]:
    """Rows and matches per public role (TRAIN, Q_DEV, Q_CAL); TEST is never counted."""

    role, match_index = np.asarray(role).astype(np.int64), np.asarray(match_index).astype(np.int64)
    return {name: {"rows": int((role == code).sum()), "matches": int(len(np.unique(match_index[role == code])))}
            for name, code in PUBLIC_ROLES.items()}


# ----------------------------------------------------------------------------- files (fixed bytes, written once)
def npz_bytes(arrays: Mapping[str, np.ndarray], *, compress: bool) -> bytes:
    """The .npz bytes of ``arrays`` (np.load reads them) with every zip entry stamped 1980-01-01 00:00:00, so the same
    arrays always give the same bytes and SHA-256.  Object arrays are refused (no pickle)."""

    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=method, allowZip64=True) as archive:
        for name, value in arrays.items():
            array = np.asanyarray(value)
            if array.dtype.hasobject:
                raise ValueError(f"{name}: object arrays are not stored")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_ZIP_TIME)
            info.compress_type, info.create_system, info.external_attr = method, 0, 0o600 << 16
            with archive.open(info, "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
    return buffer.getvalue()


def write_bytes_once(path: Path, data: bytes) -> str:
    """Write ``data`` once (pending file, fsync, atomic rename) and return its SHA-256; an existing file is refused."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; it is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return hashlib.sha256(data).hexdigest()


def write_or_keep(path: Path, data: bytes) -> tuple[str, str]:
    """('written' | 'kept', SHA-256).  An existing file is compared with ``data`` by SHA-256 only (never opened as data:
    a sealed file stays unread); a different file raises and is never overwritten."""

    digest = hashlib.sha256(data).hexdigest()
    if Path(path).exists():
        if sha256_file(path) != digest:
            raise ValueError(f"{path} exists and differs from the recomputed bytes; it is never overwritten")
        return "kept", digest
    return "written", write_bytes_once(path, data)


def identity_array(identity: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(json.dumps(_json_safe(identity), sort_keys=True).encode("utf-8"), dtype=np.uint8).copy()


def part_path(states_dir: Path, shard: int) -> Path:
    return Path(states_dir) / f"part_{int(shard):04d}.npz"


def part_status(path: Path, identity: Mapping[str, Any]) -> str:
    """'done' if the part exists and its embedded identity equals ``identity``, 'missing' if absent.  A part with another
    identity (other code, inputs or plan) or an unreadable one raises: it is never reused and never overwritten."""

    path = Path(path)
    if not path.exists():
        return "missing"
    expected = json.loads(json.dumps(_json_safe(identity), sort_keys=True))
    try:
        with np.load(path, allow_pickle=False) as stored:
            found = json.loads(bytes(stored["identity_json"]).decode("utf-8"))
    except Exception as error:  # noqa: BLE001  (any unreadable part stops the run)
        raise ValueError(f"{path} is not a readable state part ({type(error).__name__}); it is never reused or "
                         "overwritten") from error
    if found != expected:
        keys = sorted(k for k in set(found) | set(expected) if found.get(k) != expected.get(k))
        raise ValueError(f"{path} was made by other code or inputs (differs in {keys}); it is never reused or overwritten")
    return "done"


def split_done(tasks: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], dict[int, str]]:
    """(tasks whose part is missing, {shard: SHA-256} of the parts that verify); a foreign part raises."""

    todo, done = [], {}
    for task in tasks:
        if part_status(Path(task["path"]), task["identity"]) == "done":
            done[int(task["shard"])] = sha256_file(Path(task["path"]))
        else:
            todo.append(task)
    return todo, done


def plan_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps([name, array.dtype.str, list(array.shape)]).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


# ----------------------------------------------------------------------------- reproduction of the main-label V
def repro_sample(role: Any, fold: Any, *, seed: int = REPRO_SEED, per_group: int = REPRO_PER_GROUP,
                 eligible: Any = None) -> dict[str, np.ndarray]:
    """Row indices of the reproduction sample: ``per_group`` rows each of TRAIN fold 0-4, Q_DEV and Q_CAL (in that order,
    one numpy default_rng(seed) generator, without replacement, sorted).  TEST rows are never drawn."""

    role, fold = np.asarray(role).astype(np.int64), np.asarray(fold).astype(np.int64)
    keep = np.ones(len(role), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    groups = {f"TRAIN_fold{f}": (role == TRAIN) & (fold == f) for f in range(len(CONTEXTS) - 1)}
    groups.update({name: role == code for name, code in PUBLIC_ROLES.items() if code != TRAIN})
    rng = np.random.default_rng(seed)
    sample = {}
    for name, mask in groups.items():
        candidates = np.flatnonzero(mask & keep & (role != TEST))
        if len(candidates) < per_group:
            raise ValueError(f"{name} has fewer than {per_group} rows to draw")
        sample[name] = np.sort(rng.choice(candidates, size=per_group, replace=False)).astype(np.int64)
    return sample


def repro_summary(v: Any, reference: Any, *, atol: float = REPRO_ATOL) -> dict[str, Any]:
    """Rescored V against the main labels' pt_p_pre: exact (float64) and float32 equality, rows over 1e-6 and over the
    tolerance, max |d|; passed iff every |d| <= tolerance (Ruling 51)."""

    v, reference = np.asarray(v, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if v.ndim != 1 or v.shape != reference.shape or not len(v):
        raise ValueError("one rescored value per reference row")
    d = np.abs(v - reference)
    finite = np.isfinite(d)
    return {"rows": int(len(v)), "exact_equal": int((v == reference).sum()),
            "float32_equal": int((v.astype(np.float32) == reference.astype(np.float32)).sum()),
            "over_1e-6": int((~finite | (d > REPORT_ATOL)).sum()), "over_tolerance": int((~finite | (d > atol)).sum()),
            "max_abs_diff": float(d.max()) if finite.all() else None,
            "median_abs_diff": float(np.median(d)) if finite.all() else None, "tolerance": float(atol),
            "passed": bool(finite.all() and np.all(d <= atol))}


def strength_summary(v: Any, reference: Any, *, atol: float = REPRO_ATOL) -> dict[str, Any]:
    """|d| of a deliberately wrong path against the reference: min, median, max and rows within the tolerance."""

    d = np.abs(np.asarray(v, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    return {"rows": int(len(d)), "min_abs_diff": float(d.min()), "median_abs_diff": float(np.median(d)),
            "max_abs_diff": float(d.max()), "within_tolerance": int((d <= atol).sum()),
            "within_1e-6": int((d <= REPORT_ATOL).sum()), "tolerance": float(atol)}


# ----------------------------------------------------------------------------- loading and the window plan
def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("ascii", "replace")
    return value


def load_windows(windows: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """WINDOWS_h90.npz columns after its SHA-256 matched its record WINDOWS_h90.json, a passed final-window record."""

    windows = Path(windows)
    record_path = windows.with_suffix(".json")
    record = _read(record_path)
    digest = sha256_file(windows)
    problems = []
    if digest != (record.get("windows_npz") or {}).get("sha256"):
        problems.append(f"{windows.name} SHA-256 {digest} differs from its record {record_path.name}")
    if (record.get("stage") != "final_windows" or record.get("smoke") is not False
            or (record.get("xy_check") or {}).get("passed") is not True or (record.get("gate_record") or {}).get("passed") is not True):
        problems.append(f"{record_path.name} is not a passed final-window record (stage, smoke, gate or XY check)")
    if problems:
        raise Blocked("; ".join(problems))
    with np.load(windows, allow_pickle=False) as table:
        columns = {name: table[name] for name in WINDOW_COLUMNS}
    return columns, {"path": str(windows), "sha256": digest, "record": str(record_path), "record_sha256": sha256_file(record_path),
                     "record_git_head": record.get("git_head"), "record_version": record.get("version")}


def load_q_rows(columns: Sequence[str] = Q_COLUMNS) -> dict[str, np.ndarray]:
    """q_run_1 row-table columns: identity columns (and pre_ms, pt_p_pre for the reproduction check)."""

    with np.load(Q_ROWS, allow_pickle=False) as table:
        return {name: table[name] for name in columns}


def window_plan(windows: Mapping[str, np.ndarray], rows: Mapping[str, np.ndarray], membership: np.ndarray,
                projection: Any) -> dict[str, np.ndarray | str]:
    """The matched windows with their context, end, state ids and the states' match, ms, context, role and shard, after
    the section 7 checks (row-table alignment, context rule, one context per match, OOF exclusion, MATCH_TABLE keys)."""

    check_alignment(windows, rows)
    matched = np.flatnonzero(np.asarray(windows["s_ms"]) >= 0)
    role = np.asarray(rows["role"])[matched].astype(np.int64)
    fold = np.asarray(rows["fold"])[matched].astype(np.int64)
    match_index = np.asarray(rows["match_index"])[matched].astype(np.int64)
    try:
        codes = context_codes(role, fold)
    except ValueError as error:
        raise Blocked(f"a matched window's V context: {error}") from error
    check_match_contexts(match_index, codes)
    if np.any(oof_leaks(codes, role, match_index, membership)):
        raise Blocked("a matched window's V context saw its match (CONTEXT_MEMBERSHIP, assemble.py:109-114)")
    s = np.asarray(windows["s_ms"])[matched].astype(np.int64)
    d = np.asarray(windows["D_ms"])[matched].astype(np.int64)
    start_age = np.asarray(windows["start_age_plc_ms"])[matched].astype(np.int64)
    if np.any(d < MIN_D_MS) or np.any(s < MIN_START_MS) or np.any(start_age < 0):
        raise Blocked("a matched window has D < 10 s, s < 120 s or no start age")
    matches = projection.matches
    if not np.array_equal(np.asarray(matches["key"])[match_index], np.asarray(rows["canonical_match_key"])[matched]):
        raise Blocked("MATCH_TABLE keys differ from the q rows' match keys")
    if not np.array_equal(np.asarray(matches["role"])[match_index].astype(np.int64), role):
        raise Blocked("MATCH_TABLE roles differ from the q rows' roles")
    plan = state_plan(match_index, s, s + d)
    n_states = len(plan["state_match"])
    state_ctx, state_role = np.full(n_states, -1, dtype=np.int8), np.full(n_states, -1, dtype=np.int8)
    for column in ("start_state", "end_state"):
        state_ctx[plan[column]], state_role[plan[column]] = codes, role.astype(np.int8)
    shard = np.asarray(matches["shard"])[plan["state_match"]].astype(np.int64)
    if np.any(np.diff(shard) < 0):
        raise Blocked("projection match order does not follow the shard order")
    out: dict[str, Any] = {
        "window_id": matched.astype(np.int64), "engagement_key": np.asarray(windows["engagement_key"])[matched],
        "match_index": match_index, "role": role.astype(np.int8), "fold": fold.astype(np.int8), "ctx": codes,
        "s_ms": s, "end_ms": s + d, "D_ms": d, "start_age_plc_ms": start_age,
        "n_bound": np.asarray(windows["n_bound"])[matched].astype(np.int64),
        "start_state": plan["start_state"], "end_state": plan["end_state"], "state_match": plan["state_match"],
        "state_ms": plan["state_ms"], "state_ctx": state_ctx, "state_role": state_role, "state_shard": shard,
        "state_local": np.asarray(matches["local_match"])[plan["state_match"]].astype(np.int64)}
    out["plan_sha256"] = plan_digest({name: out[name] for name in ("window_id", "ctx", "s_ms", "end_ms", "start_state", "end_state",
                                                                  *PLAN_STATE)})
    return out


def _code_hashes(files: Sequence[str]) -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in files}


def _versions(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, **dict(extra or {})}


def _stage(name: str, started: str, tick: float, **extra: Any) -> dict[str, Any]:
    return {"stage": name, "started_utc": started, "finished_utc": utc(), "wall_seconds": time.perf_counter() - tick,
            "peak_memory_bytes_driver": peak_memory_bytes(), **extra}


def _workers_record(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    peaks = [int(r["peak_memory_bytes"]) for r in results if r.get("peak_memory_bytes")]
    seconds = [float(r["seconds"]) for r in results]
    return {"tasks": len(results), "worker_processes": len({r["pid"] for r in results}), "task_seconds_sum": float(sum(seconds)),
            "task_seconds_max": max(seconds, default=0.0), "worker_peak_memory_bytes_max": max(peaks, default=None)}


def _check_destination(states_dir: Path, smoke: bool, only: Sequence[str] | None = None) -> None:
    resolved, main = Path(states_dir).resolve(), (TOG_E1 / STATES_DIR).resolve()
    if smoke:
        if resolved == TOG_E1.resolve() or resolved.is_relative_to(TOG_E1.resolve()):
            raise ValueError("a smoke run never writes into tog_e1_1")
        if only is not None and any(name.startswith("test_") for name in only):
            raise ValueError("a smoke run does not read TEST shards")
    elif resolved != main or only:
        raise ValueError("a full run writes only tog_e1_1/STATES_h90 and reads every shard (use --smoke otherwise)")


# ----------------------------------------------------------------------------- workers
def run_tasks(function: Callable[[Mapping[str, Any]], dict[str, Any]], tasks: Sequence[Mapping[str, Any]],
              pool: ProcessPoolExecutor | None, label: str) -> list[dict[str, Any]]:
    """Tasks in a process pool (or in this process without one); results in task order.  Progress shows no row count.
    A failing task cancels the pending ones and re-raises."""

    results: dict[int, dict[str, Any]] = {}
    tick = time.perf_counter()

    def progress(done: int) -> None:
        if done % 25 == 0 or done == len(tasks):
            print(json.dumps({"stage": label, "tasks_done": done, "of": len(tasks), "seconds": round(time.perf_counter() - tick, 1)}),
                  flush=True)

    if pool is None:
        for i, task in enumerate(tasks):
            results[i] = function(task)
            progress(i + 1)
        return [results[i] for i in range(len(tasks))]
    futures = {pool.submit(function, task): i for i, task in enumerate(tasks)}
    try:
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            progress(done)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    return [results[i] for i in range(len(tasks))]


def _executor(workers: int) -> Any:
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f"workers must be 1-{MAX_WORKERS}")
    if int(workers) == 1:
        return contextlib.nullcontext(None)
    return ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"))


def _build_match_states(bank: Any, task: Mapping[str, Any], state_match: np.ndarray, state_ms: np.ndarray) -> dict[str, Any]:
    """controls_states.build_match per match of one shard at the given (match, time)-ordered states; the query arrays,
    the held frame times and F(t) XY come along."""

    from studies.tog_experiments_20261012.e1_gate import xy_at_times

    states: list[Any] = []
    audit: dict[str, list[np.ndarray]] = {name: [] for name in AUDIT}
    frame_ms, xy, support = [], [], []
    bounds = np.flatnonzero(np.r_[True, state_match[1:] != state_match[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        match = int(state_match[a])
        times = [int(t) for t in state_ms[a:b]]
        try:
            source = bank.source(int(task["local"][match]), match, str(task["keys"][match]))   # ledger key = MATCH_TABLE key
            build = cs.build_match(source, times)
        except ValueError as error:
            raise Blocked(f"{task['directory']}: match {match}: {error}") from error
        if (len(build.states) != len(times) or [int(state.time_ms) for state in build.states] != times
                or not np.array_equal(np.asarray(build.query["query_ms"], dtype=np.int64), np.asarray(times, dtype=np.int64))):
            raise Blocked(f"{task['directory']}: match {match}: built state times differ from the requested times")
        states.extend(build.states)
        for name in AUDIT:
            audit[name].append(np.asarray(build.query[name]))
        index = np.asarray(build.query["query_frame_index"], dtype=np.int64)
        clock = np.asarray(source.view["minute_ts"], dtype=np.int64)
        frame_ms.append(np.where(index >= 0, clock[np.maximum(index, 0)], -1))
        at, ok = xy_at_times(source.view, times, index)
        xy.append(at)
        support.append(ok)
    return {"states": states, "audit": {name: np.concatenate(values) for name, values in audit.items()},
            "frame_ms": np.concatenate(frame_ms).astype(np.int64), "xy": np.concatenate(xy), "xy_support": np.concatenate(support)}


def _open_bank(task: Mapping[str, Any]) -> Any:
    """The task's static-preserved bank shard; its bound files must equal the projection's binding, else BLOCKED."""

    try:
        return cs.BankShard(int(task["shard"]), task["binding"], Path(task["source_root"]))
    except ValueError as error:
        raise Blocked(f"{task['directory']}: {error}") from error


def build_part(task: Mapping[str, Any]) -> dict[str, Any]:
    """One bank shard: the states of its matched windows, checked against WINDOWS_h90, written once as a part."""

    started, tick = utc(), time.perf_counter()
    plan, checks = task["plan"], task["checks"]
    bank = _open_bank(task)
    built = _build_match_states(bank, task, plan["state_match"], plan["state_ms"])
    del bank
    payload = cs.encode_states(built["states"])
    if not np.array_equal(payload["time_ms"], plan["state_ms"]):
        raise Blocked(f"{task['directory']}: encoded state times differ from the plan")
    payload.update(built["audit"])
    payload.update({"frame_ms": built["frame_ms"], "xy": built["xy"], "xy_support": built["xy_support"], **plan})
    first = int(plan["state_id"][0])
    start, end = plan["window_start_state"] - first, plan["window_end_state"] - first
    index = payload["query_frame_index"].astype(np.int64)
    problems = []
    if not np.array_equal(payload["frame_ms"][start], checks["s_ms"] - checks["start_age_plc_ms"]):
        problems.append("a window's held start frame differs from WINDOWS_h90 s - start_age_plc_ms")
    if not np.array_equal(index[end] - index[start], checks["n_bound"]):
        problems.append("a window's frame count in (s, s + D] differs from WINDOWS_h90 n_bound")
    if not np.all(payload["xy_support"][start]):
        problems.append("a window's F(s) XY is not finite for all ten players")
    if problems:
        raise Blocked(f"{task['directory']}: " + "; ".join(problems))
    payload["identity_json"] = identity_array(task["identity"])
    digest = write_bytes_once(Path(task["path"]), npz_bytes(payload, compress=True))
    return {"shard": int(task["shard"]), "directory": task["directory"], "sha256": digest, "started_utc": started,
            "seconds": time.perf_counter() - tick, "peak_memory_bytes": peak_memory_bytes(), "pid": os.getpid()}


_MODELS: dict[str, tuple[Any, Any]] = {}


def _model(ctx: str) -> tuple[Any, Any]:
    """The context's locked V on CPU and its REFIT_TRAIN bundle, loaded once per process.  CUDA must be hidden."""

    if ctx not in _MODELS:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != HIDE_CUDA:
            raise RuntimeError(f"CUDA must be hidden (CUDA_VISIBLE_DEVICES={HIDE_CUDA}) before the V loads: CPU only")
        from studies.label_production_20261005 import score
        from studies.v_full_preprocess_20261003.artifacts import load_bundle

        try:      # V_LOCK, checkpoint hash and bundle manifest are verified by the frozen loaders
            v = score.LockedV(RUN_3, ctx, device="cpu")
            bundle = load_bundle(Path(queries.PREPROCESS) / f"{ctx}_REFIT_TRAIN")
        except ValueError as error:
            raise Blocked(f"{ctx}: {error}") from error
        import torch

        if v.identity.get("device") != "cpu" or torch.cuda.is_available():
            raise RuntimeError(f"{ctx}: the V is not on the CPU or CUDA is visible")
        _MODELS[ctx] = (v, bundle)
    return _MODELS[ctx]


def _identities() -> dict[str, Any]:
    import torch

    return {"v_identity": {ctx: dict(model.identity) for ctx, (model, _bundle) in _MODELS.items()}, "torch": torch.__version__,
            "torch_threads": int(torch.get_num_threads()), "cuda_available": bool(torch.cuda.is_available())}


def score_in_batches(states: Sequence[Any], ctx: Any) -> dict[str, np.ndarray]:
    """(logit, raw, v) of stored-form states (controls_states.EndState) in ``canonical_batches`` order."""

    n, ctx = len(states), np.asarray(ctx)
    if ctx.shape != (n,) or np.any(ctx < 0) or np.any(ctx >= len(CONTEXTS)):
        raise ValueError("one context code 0-5 per state")
    out = {name: np.full(n, np.nan) for name in ("logit", "raw", "v")}
    for code, positions in canonical_batches(ctx):
        model, bundle = _model(CONTEXTS[code])
        values = model.score(queries.transform(cs.semantic_rows([states[i] for i in positions]), bundle))
        for name, value in zip(("logit", "raw", "v"), values):
            if len(value) != len(positions):
                raise ValueError(f"{CONTEXTS[code]}: {len(value)} scores for {len(positions)} rows")
            out[name][positions] = value
    return out


def repro_part(task: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduction rows of one shard: states at pre_ms (right and wrong context) and at pre_ms + 1 s (right context),
    each set scored with the production batching (per context, (match, time) order, 1,024-row batches)."""

    started, tick = utc(), time.perf_counter()
    bank = _open_bank(task)
    match, pre = np.asarray(task["match"], dtype=np.int64), np.asarray(task["pre_ms"], dtype=np.int64)
    out: dict[str, Any] = {"rows": np.asarray(task["rows"], dtype=np.int64)}
    for label, shift in (("v", 0), ("v_shift", SHIFT_MS)):
        state_match, state_ms, inverse = unique_states(match, pre + shift)
        built = _build_match_states(bank, task, state_match, state_ms)
        right = np.full(len(state_match), -1, dtype=np.int64)
        right[inverse] = task["ctx"]
        out[label] = score_in_batches(built["states"], right)["v"][inverse]
        if label == "v":
            wrong = np.full(len(state_match), -1, dtype=np.int64)
            wrong[inverse] = task["wrong_ctx"]
            out["v_wrong"] = score_in_batches(built["states"], wrong)["v"][inverse]
    return {**out, **_identities(), "started_utc": started, "seconds": time.perf_counter() - tick,
            "peak_memory_bytes": peak_memory_bytes(), "pid": os.getpid()}


def score_part(task: Mapping[str, Any]) -> dict[str, Any]:
    """V of every state of one state part (its file checked against STATES_h90.json first)."""

    started, tick = utc(), time.perf_counter()
    path = Path(task["path"])
    if sha256_file(path) != task["sha256"]:
        raise Blocked(f"{path.name} differs from STATES_h90.json")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if not np.array_equal(arrays["time_ms"], arrays["state_ms"]):
        raise Blocked(f"{path.name}: state times differ from the stored times")
    scores = score_in_batches(cs.decode_states(arrays), arrays["state_ctx"])
    return {"state_id": arrays["state_id"], "state_match": arrays["state_match"], "state_ms": arrays["state_ms"], **scores,
            **_identities(), "started_utc": started, "seconds": time.perf_counter() - tick, "peak_memory_bytes": peak_memory_bytes(),
            "pid": os.getpid()}


# ----------------------------------------------------------------------------- stage: build
def _build_tasks(plan: Mapping[str, Any], projection: Any, states_dir: Path, base_identity: Mapping[str, Any],
                 only: Sequence[str] | None) -> list[dict[str, Any]]:
    shards = plan["state_shard"]
    window_shard = shards[plan["start_state"]]
    if not np.array_equal(window_shard, shards[plan["end_state"]]):
        raise Blocked("a window's start and end states lie in different shards")
    bounds = np.flatnonzero(np.r_[True, shards[1:] != shards[:-1], True])
    tasks = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        shard = int(shards[a])
        binding = dict(projection.binding(shard))
        if only is not None and binding["directory"] not in only:
            continue
        own = np.flatnonzero(window_shard == shard)
        task_plan = {"state_id": np.arange(a, b, dtype=np.int64),
                     **{name: np.asarray(plan[name][a:b]) for name in PLAN_STATE},
                     "window_id": plan["window_id"][own], "window_start_state": plan["start_state"][own],
                     "window_end_state": plan["end_state"][own]}
        matches = np.unique(task_plan["state_match"])
        tasks.append({"shard": shard, "directory": binding["directory"], "binding": binding,
                      "source_root": str(projection.source_root), "path": str(part_path(states_dir, shard)),
                      "identity": {**base_identity, "shard": shard, "directory": binding["directory"], "bank_binding": binding,
                                   "plan_sha256": plan_digest(task_plan)},
                      "plan": task_plan, "checks": {name: plan[name][own] for name in ("s_ms", "start_age_plc_ms", "n_bound")},
                      "local": {int(m): int(projection.matches["local_match"][m]) for m in matches},
                      "keys": {int(m): projection.key(int(m)) for m in matches}})
    return tasks


def role_record(plan: Mapping[str, Any], windows: np.ndarray, states: np.ndarray) -> dict[str, Any]:
    """Per public role (TRAIN, Q_DEV, Q_CAL; never TEST): matched windows and matches (``public_counts``), unique states,
    windows per context and windows ending at a start state, over the selected windows and states."""

    windows, states = np.asarray(windows, dtype=bool), np.asarray(states, dtype=bool)
    role, state_role = plan["role"].astype(np.int64), plan["state_role"].astype(np.int64)
    counts = public_counts(role[windows], plan["match_index"][windows])
    flags = end_is_a_start(plan["start_state"], plan["end_state"])
    out = {}
    for name, code in PUBLIC_ROLES.items():
        own = windows & (role == code)
        out[name] = {"matched_windows": counts[name]["rows"], "matches": counts[name]["matches"],
                     "states": int((states & (state_role == code)).sum()),
                     "windows_by_context": {CONTEXTS[c]: int(((plan["ctx"] == c) & own).sum()) for c in np.unique(plan["ctx"][own]).tolist()},
                     "windows_ending_at_a_start_state": int((flags & own).sum())}
    return out


def build_states(windows: Path = WINDOWS, *, out: Path | None = None, workers: int = MAX_WORKERS, smoke: bool = False,
                 only: Sequence[str] | None = None) -> Path:
    """Build (or finish) the state parts of every matched window and STATES_h90.json; returns the states folder."""

    windows = Path(windows)
    states_dir = Path(out) if out is not None else windows.parent / STATES_DIR
    _check_destination(states_dir, smoke, only)
    if smoke and not only:
        raise ValueError("a smoke run names its shards (never TEST shards)")
    record_path = states_dir / STATES_RECORD
    if record_path.exists():
        _verify_parts(states_dir, _read(record_path))
        print(json.dumps({"stage": "build", "done": str(record_path)}), flush=True)
        return states_dir
    started, tick = utc(), time.perf_counter()
    gate = _read(GATE_RECORD)
    if gate.get("passed") is not True:
        raise Blocked("the equality gate (EQUALITY_GATE_TOG.json) did not pass")
    table, windows_info = load_windows(windows)
    rows = load_q_rows()
    projection, projection_sha = _projection()
    membership = np.load(MEMBERSHIP, allow_pickle=False)
    plan = window_plan(table, rows, membership, projection)
    del table, rows
    code = _code_hashes(BUILD_CODE)
    q_rows_sha = sha256_file(Q_ROWS)
    base_identity = {"version": PART_VERSION, "windows_sha256": windows_info["sha256"], "q_rows_sha256": q_rows_sha,
                     "projection_files_sha256": projection_sha, "code_sha256": code, "global_plan_sha256": plan["plan_sha256"]}
    tasks = _build_tasks(plan, projection, states_dir, base_identity, only)
    built = np.isin(plan["state_shard"], np.asarray([int(task["shard"]) for task in tasks], dtype=np.int64))
    stages = [_stage("load_and_plan", started, tick)]
    states_dir.mkdir(parents=True, exist_ok=True)
    todo, reused = split_done(tasks)
    s, t = utc(), time.perf_counter()
    with _executor(workers) as pool:
        results = run_tasks(build_part, todo, pool, "build")
    stages.append(_stage("parts", s, t, parts_written=len(results), parts_reused=len(reused), **_workers_record(results)))
    s, t = utc(), time.perf_counter()
    written = {int(r["shard"]): r["sha256"] for r in results}
    parts = {}
    for task in tasks:
        shard = int(task["shard"])
        if part_status(Path(task["path"]), task["identity"]) != "done":
            raise RuntimeError(f"{task['path']} is missing after the build")
        digest = sha256_file(Path(task["path"]))
        if digest != written.get(shard, reused.get(shard)):
            raise RuntimeError(f"{task['path']} changed after it was written")
        parts[task["directory"]] = {"file": Path(task["path"]).name, "shard": shard, "sha256": digest}
    lines = "".join(f"{part['file']}:{part['sha256']}\n" for part in sorted(parts.values(), key=lambda p: p["file"]))
    record = {
        "version": VERSION, "experiment": "E1", "stage": "states", "smoke": smoke, "smoke_shards": list(only) if only else None,
        "started_utc": started, "utc": utc(), "seed": None, "rule": BUILD_RULE,
        "windows": windows_info, "q_rows": {"path": str(Q_ROWS), "sha256": q_rows_sha},
        "equality_gate": {"path": str(GATE_RECORD), "sha256": sha256_file(GATE_RECORD), "passed": True},
        "plan_sha256": plan["plan_sha256"], "contexts": {"codes": dict(enumerate(CONTEXTS)), "rule": CONTEXT_RULE},
        "roles": role_record(plan, built[plan["start_state"]], built),
        "checks": {"passed": True, "applied_to": "every matched window of every role (failures stop the run as BLOCKED)",
                   "checks": ["row table alignment", "V context rule", "one context per match", "OOF exclusion",
                              "MATCH_TABLE keys and roles", "built times = requested times", "held start frame = s - start_age_plc_ms",
                              "frame count in (s, s + D] = n_bound", "F(s) XY finite"]},
        "test_note": "TEST windows and states are in the parts (PRESPEC_E1 section 6 step 3); no TEST count is recorded here",
        "parts": parts, "parts_list_sha256": hashlib.sha256(lines.encode("ascii")).hexdigest(),
        "parts_list_rule": "sha256 of the sorted lines '<file>:<sha256>\\n'", "part_columns": PART_COLUMNS,
        "part_identity": {key: value for key, value in base_identity.items() if key != "code_sha256"},
        "inputs": {"windows_npz": windows_info["sha256"], "windows_record": windows_info["record_sha256"], "q_rows": q_rows_sha,
                   "context_membership": sha256_file(MEMBERSHIP), "projection_files": projection_sha,
                   "projection_run_manifest": sha256_file(projection.root / "RUN_MANIFEST.json"),
                   "projection_bank_metadata": sha256_file(projection.root / "BANK_METADATA.json")},
        "prespec": document_record(PRESPEC), "amendment": document_record(AMENDMENT),
        "code_sha256": code, "versions": _versions(), "git_head": git_head(), "workers": int(workers), "stages": stages,
        "peak_memory_bytes": {"driver": peak_memory_bytes(), "workers_max": _workers_record(results)["worker_peak_memory_bytes_max"]}}
    stages.append(_stage("verify_parts_and_record", s, t))          # the record holds this list
    write_once(record_path, _json_safe(record))
    print(json.dumps({"stage": "build", "record": str(record_path), "record_sha256": sha256_file(record_path),
                      "parts_list_sha256": record["parts_list_sha256"],
                      "matched_windows": {name: r["matched_windows"] for name, r in record["roles"].items()}}), flush=True)
    return states_dir


def _projection() -> tuple[Any, dict[str, str]]:
    """The label projection (controls_states.Projection) with its files checked against its run manifest, else BLOCKED."""

    try:
        projection = cs.Projection()
        if projection.root.resolve() != Path(queries.PROJECTION).resolve():
            raise ValueError("the bank projection differs from the label projection")
        return projection, projection.file_hashes()
    except ValueError as error:
        raise Blocked(str(error)) from error


def _verify_parts(states_dir: Path, record: Mapping[str, Any]) -> None:
    for directory, part in record["parts"].items():
        if sha256_file(Path(states_dir) / part["file"]) != part["sha256"]:
            raise Blocked(f"{part['file']} ({directory}) differs from {STATES_RECORD}")


# ----------------------------------------------------------------------------- stage: score
def _repro_tasks(sample: Mapping[str, np.ndarray], rows: Mapping[str, np.ndarray], projection: Any) -> list[dict[str, Any]]:
    picked = np.sort(np.concatenate(list(sample.values())))
    match = np.asarray(rows["match_index"])[picked].astype(np.int64)
    role = np.asarray(rows["role"])[picked].astype(np.int64)
    ctx = context_codes(role, np.asarray(rows["fold"])[picked])
    wrong = np.asarray([CONTEXTS.index(WRONG_CONTEXT[ROLE_NAMES[int(r)]]) for r in role], dtype=np.int64)
    if np.any(wrong == ctx):
        raise ValueError("a wrong context equals the row's context")
    if not np.array_equal(np.asarray(projection.matches["key"])[match], np.asarray(rows["canonical_match_key"])[picked]):
        raise Blocked("MATCH_TABLE keys differ from the reproduction rows' match keys")
    shard = np.asarray(projection.matches["shard"])[match].astype(np.int64)
    tasks = []
    for value in np.unique(shard).tolist():
        own = shard == value
        binding = dict(projection.binding(value))
        tasks.append({"shard": int(value), "directory": binding["directory"], "binding": binding,
                      "source_root": str(projection.source_root), "rows": picked[own], "match": match[own],
                      "pre_ms": np.asarray(rows["pre_ms"])[picked[own]].astype(np.int64), "ctx": ctx[own], "wrong_ctx": wrong[own],
                      "local": {int(m): int(projection.matches["local_match"][m]) for m in np.unique(match[own])},
                      "keys": {int(m): projection.key(int(m)) for m in np.unique(match[own])}})
    return tasks


def _check_identities(results: Sequence[Mapping[str, Any]], label_v: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The V every worker loaded, per context: one identity, equal to the main labels' V apart from the device."""

    found: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result["cuda_available"]:
            raise RuntimeError("a worker saw CUDA; this run is CPU only")
        for ctx, identity in result["v_identity"].items():
            found.setdefault(ctx, []).append(identity)
    record = {}
    for ctx, identities in sorted(found.items()):
        if any(identity != identities[0] for identity in identities):
            raise Blocked(f"{ctx}: workers loaded different V")
        mine, label = identities[0], label_v[ctx]
        differs = sorted(k for k in set(mine) | set(label) if k != "device" and mine.get(k) != label.get(k))
        if differs or mine.get("device") != "cpu":
            raise Blocked(f"{ctx}: the V differs from the main labels' V in {differs or ['device']}")
        record[ctx] = {"identity": mine, "label_device": label.get("device"), "equals_label_v_except_device": True}
    return record


def score_states(states: Path = TOG_E1 / STATES_DIR, *, workers: int = MAX_WORKERS, smoke: bool = False,
                 repro_per_group: int = REPRO_PER_GROUP) -> Path:
    """The reproduction gate, then V of every state, then the public, sealed and reproduction files and the run record;
    returns the public prediction file."""

    os.environ["CUDA_VISIBLE_DEVICES"] = HIDE_CUDA   # CPU only: before torch loads here or in a spawned worker
    states_dir = Path(states)
    _check_destination(states_dir, smoke)
    out = states_dir.parent
    pred_path, sealed_path, repro_path, record_path = out / PRED_NAME, out / SEALED_NAME, out / REPRO_NAME, out / PRED_RECORD
    if record_path.exists():
        done = _read(record_path)
        for key, path in (("public", pred_path), ("sealed", sealed_path), ("repro", repro_path)):
            if sha256_file(path) != done["outputs"][key]["sha256"]:
                raise Blocked(f"{path.name} differs from {PRED_RECORD}")
        print(json.dumps({"stage": "score", "done": str(record_path)}), flush=True)
        return pred_path
    if repro_per_group != REPRO_PER_GROUP and not smoke:
        raise ValueError("the reproduction sample holds 200 rows per group (Ruling 51)")
    started, tick = utc(), time.perf_counter()
    manifest_path = states_dir / STATES_RECORD
    manifest = _read(manifest_path)
    if manifest.get("stage") != "states" or bool(manifest.get("smoke")) != smoke:
        raise Blocked(f"{manifest_path} is not a states record of this run kind")
    _verify_parts(states_dir, manifest)
    table, windows_info = load_windows(Path(manifest["windows"]["path"]))
    if windows_info["sha256"] != manifest["windows"]["sha256"]:
        raise Blocked("WINDOWS_h90.npz differs from the one the states were built from")
    rows = load_q_rows((*Q_COLUMNS, "pre_ms", "pt_p_pre"))     # pt_p_pre: the reproduction check only (non-TEST rows)
    projection, projection_sha = _projection()
    membership = np.load(MEMBERSHIP, allow_pickle=False)
    plan = window_plan(table, rows, membership, projection)
    del table
    if plan["plan_sha256"] != manifest["plan_sha256"]:
        raise Blocked("the window plan differs from the states' plan")
    label_v = {ctx: _read(LABELS / f"PRED_{ctx}_MANIFEST.json")["v"] for ctx in CONTEXTS}
    stages = [_stage("load_and_plan", started, tick)]

    eligible = None
    if smoke:
        directories = np.asarray([projection.binding(int(s))["directory"] for s in range(len(projection.metadata["source_shards"]))])
        eligible = np.isin(directories[np.asarray(projection.matches["shard"])[np.asarray(rows["match_index"], dtype=np.int64)]],
                           manifest["smoke_shards"])
    sample = repro_sample(rows["role"], rows["fold"], per_group=repro_per_group, eligible=eligible)
    with _executor(workers) as pool:
        s, t = utc(), time.perf_counter()
        repro_results = run_tasks(repro_part, _repro_tasks(sample, rows, projection), pool, "repro")
        stages.append(_stage("reproduction", s, t, **_workers_record(repro_results)))
        repro = _repro_record(sample, rows, repro_results)
        print(json.dumps({"stage": "reproduction", **{k: repro["summary"][k] for k in ("rows", "exact_equal", "float32_equal",
                                                                                        "over_1e-6", "max_abs_diff", "passed")}}),
              flush=True)
        if not repro["summary"]["passed"]:
            raise Blocked(f"the main-label V is not reproduced within {REPRO_ATOL}: {json.dumps(_json_safe(repro['summary']))}")
        s, t = utc(), time.perf_counter()
        tasks = [{"path": str(states_dir / part["file"]), "sha256": part["sha256"], "directory": directory}
                 for directory, part in sorted(manifest["parts"].items(), key=lambda item: item[1]["shard"])]
        score_results = run_tasks(score_part, tasks, pool, "score")
        stages.append(_stage("score", s, t, **_workers_record(score_results)))
    s, t = utc(), time.perf_counter()
    v_record = _check_identities([*repro_results, *score_results], label_v)
    scores, scored = _assemble_scores(plan, score_results, smoke)
    keep = scored[plan["start_state"]] & scored[plan["end_state"]]
    windows = {name: np.asarray(plan[name])[keep] for name in PUBLIC_COLUMNS}
    public, sealed = split_predictions(windows, scores)
    repro_arrays = repro.pop("arrays")
    outputs = {}
    for key, path, data in (("public", pred_path, npz_bytes(public, compress=False)),
                            ("sealed", sealed_path, npz_bytes(sealed, compress=False)),
                            ("repro", repro_path, npz_bytes(repro_arrays, compress=False))):
        status, digest = write_or_keep(path, data)
        outputs[key] = {"path": str(path), "sha256": digest, "status": status}
    del sealed
    outputs["public"]["columns"] = list(public)
    outputs["public"]["rows"] = "one per matched window of every role, window_id order; TEST V(s + D) columns are NaN"
    outputs["sealed"].update(columns=list(SEALED_COLUMNS), note="TEST rows only; nobody opens it before the E1 lock "
                             "(PRESPEC_E1 section 6 step 7); compared here by SHA-256 only")
    outputs["repro"]["columns"] = list(repro_arrays)
    stages.append(_stage("assemble_and_write", s, t))
    record = {
        "version": VERSION, "experiment": "E1", "stage": "placebo_v", "smoke": smoke, "started_utc": started, "utc": utc(),
        "seed": REPRO_SEED, "seed_use": "numpy default_rng for the reproduction sample only; scoring has no randomness",
        "device": "cpu", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "labels_device": {ctx: label_v[ctx].get("device") for ctx in CONTEXTS},
        "device_note": "the main labels were scored on CUDA (labels_1/PRED_<ctx>_MANIFEST.json v.device); this run scores "
                       "on CPU (score.LockedV device='cpu', CUDA hidden)",
        "v": v_record, "reproduction": repro, "scoring": {"rule": SCORE_RULE, "batch_rows": BATCH_ROWS},
        "outputs": outputs, "roles": role_record(plan, keep, scored),
        "validity": {"passed": True, "rule": "every scored state: logit finite, raw probability and V finite in [0, 1]; "
                                               "TEST enters as pass/fail only"},
        "test_note": "no TEST count or value is recorded here (PRESPEC_E1 section 6); TEST V(s + D) only in the sealed file",
        "states": {"record": str(manifest_path), "record_sha256": sha256_file(manifest_path),
                   "parts_list_sha256": manifest["parts_list_sha256"], "plan_sha256": manifest["plan_sha256"],
                   "build_git_head": manifest.get("git_head"), "build_code_sha256": manifest.get("code_sha256")},
        "inputs": {"windows_npz": windows_info["sha256"], "windows_record": windows_info["record_sha256"],
                   "q_rows": sha256_file(Q_ROWS), "context_membership": sha256_file(MEMBERSHIP), "projection_files": projection_sha,
                   "label_prediction_manifests": {ctx: sha256_file(LABELS / f"PRED_{ctx}_MANIFEST.json") for ctx in CONTEXTS},
                   "bundles": {ctx: {p.name: sha256_file(p) for p in sorted((Path(queries.PREPROCESS) / f"{ctx}_REFIT_TRAIN").iterdir())
                                     if p.is_file()} for ctx in CONTEXTS},
                   "v_locks": {ctx: sha256_file(RUN_3 / ctx / "V_LOCK.json") for ctx in CONTEXTS}},
        "prespec": document_record(PRESPEC), "amendment": document_record(AMENDMENT), "code_sha256": _code_hashes(SCORE_CODE),
        "versions": _versions({"torch": score_results[0]["torch"] if score_results else None,
                               "torch_threads": score_results[0]["torch_threads"] if score_results else None}),
        "git_head": git_head(), "workers": int(workers), "stages": stages,
        "peak_memory_bytes": {"driver": peak_memory_bytes(),
                              "workers_max": max((r["peak_memory_bytes"] or 0 for r in [*repro_results, *score_results]), default=None)}}
    write_once(record_path, _json_safe(record))
    print(json.dumps({"stage": "score", "record": str(record_path), "record_sha256": sha256_file(record_path),
                      "outputs": {key: value["sha256"] for key, value in outputs.items()}}), flush=True)
    return pred_path


def _repro_record(sample: Mapping[str, np.ndarray], rows: Mapping[str, np.ndarray], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    picked = np.sort(np.concatenate(list(sample.values())))
    position = {int(r): i for i, r in enumerate(picked.tolist())}
    values = {name: np.full(len(picked), np.nan) for name in ("v", "v_wrong", "v_shift")}
    seen = np.zeros(len(picked), dtype=bool)
    for result in results:
        at = np.asarray([position[int(r)] for r in result["rows"]], dtype=np.int64)
        if seen[at].any():
            raise RuntimeError("a reproduction row was scored twice")
        seen[at] = True
        for name in values:
            values[name][at] = result[name]
    if not seen.all():
        raise RuntimeError("a reproduction row was not scored")
    reference = np.asarray(rows["pt_p_pre"], dtype=np.float64)[picked]
    role = np.asarray(rows["role"])[picked].astype(np.int64)
    group = np.empty(len(picked), dtype="S12")
    for name, members in sample.items():
        group[[position[int(r)] for r in members]] = name.encode("ascii")
    ctx = context_codes(role, np.asarray(rows["fold"])[picked])
    arrays = {"q_row": picked, "group": group, "role": role.astype(np.int8), "fold": np.asarray(rows["fold"])[picked],
              "match_index": np.asarray(rows["match_index"])[picked].astype(np.int64),
              "pre_ms": np.asarray(rows["pre_ms"])[picked].astype(np.int64), "ctx": ctx,
              "wrong_ctx": np.asarray([CONTEXTS.index(WRONG_CONTEXT[ROLE_NAMES[int(r)]]) for r in role], dtype=np.int8),
              "pt_p_pre": reference, "v_cpu": values["v"], "v_wrong_ctx": values["v_wrong"], "v_shift_1s": values["v_shift"]}
    by_group = {}
    for name in sample:
        own = group == name.encode("ascii")
        by_group[name] = {**repro_summary(values["v"][own], reference[own]),
                          "matches": int(len(np.unique(arrays["match_index"][own])))}
    return {"rule": f"per group {len(next(iter(sample.values())))} rows (TRAIN folds 0-4, Q_DEV, Q_CAL; numpy "
                    f"default_rng({REPRO_SEED}) in that order, without replacement); the state at pre_ms through "
                    "controls_states.build_match, scored in the row's context; pass iff |V - pt_p_pre| <= 1e-5 on every "
                    "row (Ruling 51; labels on CUDA, this run on CPU)",
            "batching": "the production rule within each bank shard: per context, the shard's sampled states in (match, "
                        "time) order, 1,024-row batches",
            "seed": REPRO_SEED, "groups": list(sample), "summary": repro_summary(values["v"], reference), "by_group": by_group,
            "strength": {"wrong_context": {"rule": "the same states scored in a wrong context: TRAIN rows final, Q_DEV and "
                                                   "Q_CAL rows oof0", **strength_summary(values["v_wrong"], reference)},
                         "shift_1s": {"rule": "the state at pre_ms + 1 s in the row's context",
                                      **strength_summary(values["v_shift"], reference)}},
            "arrays": arrays}


def _assemble_scores(plan: Mapping[str, Any], results: Sequence[Mapping[str, Any]], smoke: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n = len(plan["state_match"])
    scores = {name: np.full(n, np.nan) for name in ("logit", "raw", "v")}
    scored = np.zeros(n, dtype=bool)
    for result in results:
        ids = np.asarray(result["state_id"], dtype=np.int64)
        if scored[ids].any():
            raise RuntimeError("a state was scored twice")
        if not (np.array_equal(result["state_match"], plan["state_match"][ids]) and np.array_equal(result["state_ms"], plan["state_ms"][ids])):
            raise Blocked("a part's states differ from the plan")
        scored[ids] = True
        for name in scores:
            scores[name][ids] = result[name]
    if not smoke and not scored.all():
        raise Blocked("not every state of a matched window was scored")
    valid = (np.isfinite(scores["logit"]) & np.isfinite(scores["raw"]) & np.isfinite(scores["v"])
             & (scores["raw"] >= 0) & (scores["raw"] <= 1) & (scores["v"] >= 0) & (scores["v"] <= 1))
    if not valid[scored].all():
        role = plan["state_role"].astype(np.int64)
        public = {name: int((scored & ~valid & (role == code)).sum()) for name, code in PUBLIC_ROLES.items()}
        raise Blocked(f"a state's V is not finite or outside [0, 1]: invalid states {public}; TEST "
                      f"{'failed' if (scored & ~valid & (role == TEST)).any() else 'passed'} (no TEST count is recorded)")
    return scores, scored


# ----------------------------------------------------------------------------- CLI
def _blocked(folder: Path, stage: str, reasons: Sequence[str], **extra: Any) -> int:
    """Write E1_STATES_BLOCKED_<stage>_<n>.json once (first free n) and return 2."""

    folder.mkdir(parents=True, exist_ok=True)
    k = 1
    while (folder / f"E1_STATES_BLOCKED_{stage}_{k}.json").exists():
        k += 1
    path = folder / f"E1_STATES_BLOCKED_{stage}_{k}.json"
    write_once(path, _json_safe({"version": VERSION, "experiment": "E1", "blocked_stage": stage, "blocked": True, "utc": utc(),
                                 "reasons": list(reasons), **extra}))
    print(json.dumps({"BLOCKED": stage, "file": str(path), "reasons": list(reasons)}, ensure_ascii=False), flush=True)
    return 2


def _below_normal_priority() -> int | None:
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        return int(process.nice())
    except Exception:  # noqa: BLE001  (priority is a courtesy, not a result)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("build", "score", "run"))
    parser.add_argument("--windows", default=str(WINDOWS))
    parser.add_argument("--out", default=None, help="states folder (default <windows folder>/STATES_h90)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--smoke", action="store_true", help="a check run on named non-TEST shards into another folder")
    parser.add_argument("--shards", default=None, help="comma-separated bank shard directories (smoke only)")
    parser.add_argument("--repro-per-group", type=int, default=REPRO_PER_GROUP, help="smoke only")
    args = parser.parse_args(argv)
    only = [name for name in args.shards.split(",") if name] if args.shards else None
    if only and not args.smoke:
        parser.error("--shards needs --smoke")
    if args.repro_per_group != REPRO_PER_GROUP and not args.smoke:
        parser.error("--repro-per-group needs --smoke")
    os.environ["CUDA_VISIBLE_DEVICES"] = HIDE_CUDA   # CPU only; set before torch loads, inherited by spawned workers
    _below_normal_priority()
    try:
        from studies.v_production_20261004.worker import keep_awake

        keep_awake()
    except Exception:  # noqa: BLE001  (best effort, as score.main)
        pass
    states_dir = Path(args.out) if args.out else Path(args.windows).parent / STATES_DIR
    stage = "build"
    try:
        if args.stage in ("build", "run"):
            build_states(Path(args.windows), out=states_dir, workers=args.workers, smoke=args.smoke, only=only)
        stage = "score"
        if args.stage in ("score", "run"):
            score_states(states_dir, workers=args.workers, smoke=args.smoke, repro_per_group=args.repro_per_group)
    except Blocked as error:
        return _blocked(states_dir.parent, stage, [str(error)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
