"""T1.2 E1 placebo windows: candidate grid, rules in order, per-rule rejection counts, deterministic hash choice.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E1_KO.md section 3 (matching features, exclusions,
selection, balance reporting), section 6 step 2 (order, the Q_DEV 30% gate), section 7 (BLOCKED), Appendix A (gold
edges), as amended 2026-10-13 (author's option 1, commit bd300bf); AMENDMENT_E1_KO.md; SDD Rulings 14-17, 24, 27, 49.
The features are e1_features.py's, one function for the engagement (start tau, D = h90 endpoint - tau) and for a
candidate (start s, the engagement's D).

Candidates of an engagement: s = tau + 1,000 k, k integer, |k| <= 600; points with s < 120,000 are dropped beforehand
(not counted).  Rules, each candidate counted once under the first it fails:
  start_age_out    start age (s - F(s)) outside [0, 10,000] or no frame at or before s
  end_range        s + D after the last frame, or s + D at or after the game end
  kill_inside      a CHAMPION_KILL in the closed window [s, s + D]
  outcome_overlap  [s, s + D] meets the closed outcome window [f - 10,000, e_h90] of a q engagement of the same match
                   (a row of q_run_1/Q_ROWS_h90.npz, the engagement's own included); each q row's component is the one
                   bank component with its component_key (exactly one, else BLOCKED), whose window equals [tau, tau + D].
                   Other bank components do not exclude (10-13 amendment; the 10-12 rule used every component)
  keys_differ      any of the six features differs from the engagement's (kill_gap_bin has five bins since 10-13)
No objective exclusion.  Among the survivors the chosen start has the smallest
hashlib.sha256((engagement_key.decode("ascii") + ":" + str(int(s))).encode("utf-8")).hexdigest(); a start may serve
several engagements (reuse).  An engagement without a survivor is unmatched (s = -1).

Stages (CLI), each writing once under D:/LOL_Training_20261004/tog_e1_1:
  gold   Before any window: F(tau) gold difference of the 268,101 TRAIN q rows -> per time_bin 20/40/60/80
         percentiles -> GOLD_TRAIN_h90.npz, GOLD_EDGES_h90.json.  The values then go into PRESPEC_E1 Appendix A,
         which is committed alone.  (Ran once on 10-13; the amendment keeps these edges.)
  gate   Q_DEV windows only (Ruling 27) -> WINDOWS_h90_QDEV_GATE_r1.npz/.json with the 30% gate on the Q_DEV match
         rate (matched / Q_DEV q engagements with valid h90).  Below 30% the run stops for the author (no relaxation
         here).  The edges used must equal Appendix A of the committed PRESPEC, else BLOCKED.  Attempt r0 (the 10-12
         rule, code commit fc10e84) is the superseded record WINDOWS_h90_QDEV_GATE_r0.*, never rewritten.
  final  After a passed gate: windows of every q role (TRAIN, Q_DEV, Q_CAL, TEST) once -> WINDOWS_h90.npz (one row per
         q engagement in q_run_1/Q_ROWS_h90.npz order; s_ms = -1 unmatched), WINDOWS_h90.json (TRAIN, Q_DEV, Q_CAL
         only) and WINDOWS_h90_TEST_SEALED.json (every TEST summary; nobody views it before the E1 lock, Ruling 24).
         The Q_DEV windows must equal the gate's.  A chosen window whose F(s) XY is not finite for all ten players is
         BLOCKED (counts written, the TEST count only in the sealed file).

Row identity comes from Q_ROWS_h90.npz (identity columns only; no label column is read), aligned with the locked
labels_1/ENGAGEMENTS.npz for each engagement's h90 endpoint and frames; bank shards are read only after their
BANK.npz and MATCH_LEDGER.jsonl hashes equal the projection's binding (BANK_METADATA.json).  At most 4 worker
processes, CPU only, below-normal priority.  No V score, no label, no player identifier.  Nothing about TEST is
printed.

Usage: python -B -m studies.tog_experiments_20261012.e1_candidates gold|gate|final [--workers 4]
       Smoke check: ... --out <scratch dir> --smoke --shards train_0000,validation_0000 [--gold <GOLD_EDGES json>]
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005.engagements import COMPONENT_COLUMNS  # noqa: E402
from studies.tog_experiments_20261012 import e1_features as ef  # noqa: E402
from studies.tog_experiments_20261012.common import (  # noqa: E402
    LABELS, Q_RUN, TOG_E1, document_record, git_head, peak_memory_bytes, sha256_file, utc, write_npz_once, write_once)

VERSION = "tog_e1_candidates_v2_20261013"
GRID_STEP_MS = 1_000
GRID_HALF_WIDTH = 600                 # |k| <= 600, so |s - tau| <= 10 min
GRID_OFFSETS_MS = GRID_STEP_MS * np.arange(-GRID_HALF_WIDTH, GRID_HALF_WIDTH + 1, dtype=np.int64)
GRID_POINTS = len(GRID_OFFSETS_MS)
MIN_START_MS = 120_000                # starts before 120 s are dropped from the grid beforehand (not counted)
ANCHOR_OFFSET_MS = 10_000             # an outcome window opens 10 s before the component's first kill
REJECT_KEYS = ("start_age_out", "end_range", "kill_inside", "outcome_overlap", "keys_differ")
GATE_ATTEMPT = "r1"                   # the 10-13 amended rule (q-engagement exclusion, five kill-gap bins)
SUPERSEDED_ATTEMPTS = ("r0",)         # the 10-12 rule (every bank component, three kill-gap bins), code commit fc10e84
GATE_THRESHOLD = 0.30
MAX_WORKERS = 4
CHAMPION_KILL = 1                     # FIELD_MANIFEST event_type_names index
ROLE_NAMES = {0: "TRAIN", 3: "Q_DEV", 4: "Q_CAL", 5: "TEST"}
Q_DEV, TEST = 3, 5
PUBLIC_ROLES = (0, 3, 4)

BANK = ROOT / "outputs/observable_state_20261002/full_static_preserved_20261002_1"
PROJECTION = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
PRESPEC = ROOT / "reports/tog_experiments_20261012/PRESPEC_E1_KO.md"
AMENDMENT = ROOT / "reports/tog_experiments_20261012/AMENDMENT_E1_KO.md"
Q_ROWS = Q_RUN / "Q_ROWS_h90.npz"
ENGAGEMENTS = LABELS / "ENGAGEMENTS.npz"
GOLD_JSON, GOLD_ROWS = "GOLD_EDGES_h90.json", "GOLD_TRAIN_h90.npz"
GATE_NAME = f"WINDOWS_h90_QDEV_GATE_{GATE_ATTEMPT}"
FINAL_NAME, SEALED_NAME = "WINDOWS_h90", "WINDOWS_h90_TEST_SEALED.json"
APPENDIX_HEADING = "## 부록 A"
SHARD_FIELDS = ("event_timestamp", "event_type_code", "component_values", "component_key", "minute_ts", "gold_team",
                "end_ms", "xy")
GOLD_FIELDS = ("minute_ts", "gold_team")
Q_ROW_COLUMNS = ("engagement_key", "canonical_match_key", "component_key", "match_index", "role", "fold", "query_pre",
                 "pre_ms")
ENGAGEMENT_COLUMNS = ("canonical_match_key", "component_key", "match_index", "role", "fold", "query_pre", "valid_h90",
                      "pre_ms", "post_ms_h90", "F_pre_ms", "F_post_ms_h90", "game_end_ms")
CODE_FILES = ("studies/tog_experiments_20261012/e1_candidates.py", "studies/tog_experiments_20261012/e1_features.py",
              "studies/tog_experiments_20261012/common.py", "studies/label_production_20261005/engagements.py",
              "studies/q_production_20261005/qtest.py")
RULE = ("attempt r1 (PRESPEC_E1 2026-10-13 amendment): candidates s = tau + 1000 k, |k| <= 600, s >= 120000 (dropped "
        "beforehand, not counted); rules in order start_age_out (s - F(s) outside [0, 10000]), end_range (s + D > last "
        "frame or s + D >= game end), kill_inside (CHAMPION_KILL in [s, s + D]), outcome_overlap ([s, s + D] meets "
        "[f - 10000, e_h90] of a q engagement of the same match, own included; other bank components do not exclude), "
        "keys_differ (start_age_bin, n_bound, end_age_bin, time_bin, kill_gap_bin with edges 30/60/90/180 s, gold_bin); "
        "choice: smallest sha256((engagement_key.decode('ascii') + ':' + str(int(s))).encode('utf-8')).hexdigest(); "
        "reuse allowed; no objective exclusion (PRESPEC_E1 section 3)")
COLUMN_NOTE = {
    "s_ms": "chosen placebo start, -1 unmatched", "D_ms": "engagement h90 endpoint - tau (the placebo window length)",
    "start_age_bin..gold_bin": "the engagement's six features (equal to the window's when matched; kill_gap_bin 0-4)",
    "*_eng / *_plc": "continuous values of the engagement at tau and of the chosen window at s (-1 / NaN without one): "
                     "start age ms, time since the last kill before the start ms (-1 none), F(start) gold difference",
    "n_candidates": "grid points with s >= 120 s", "n_survivors": "candidates passing every rule",
    "reject_counts": f"candidates per first failing rule, columns {list(REJECT_KEYS)}",
    "xy_finite_plc": "1 if all ten F(s) XY are finite, 0 if not, -1 without a window",
    "meets_component_window_plc": "1 if the chosen [s, s + D] meets the closed [f - 10 s, e_h90] of any bank component "
                                  "of the match (q engagements' windows are excluded by the rule, so another "
                                  "component's), 0 if not, -1 without a window (report only)"}


class Blocked(ValueError):
    """A PRESPEC_E1 section 7 stop (row-table key mismatch, inputs that differ from their records): written as a
    BLOCKED record by the stage, never silently skipped."""


# ----------------------------------------------------------------------------- rules on one match
def _ints(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size and array.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integers (ms)")
    return array.astype(np.int64, copy=False).reshape(-1)


def has_kill(kills_ms, start_ms, end_ms) -> np.ndarray:
    """Whether a kill lies in the closed window [start, end] (kills_ms sorted)."""

    kills, start, end = _ints(kills_ms, "kills_ms"), _ints(start_ms, "start_ms"), _ints(end_ms, "end_ms")
    if start.shape != end.shape:
        raise ValueError("one end per start")
    if np.any(np.diff(kills) < 0):
        raise ValueError("kill times must be sorted")
    if not len(kills):
        return np.zeros(len(start), dtype=bool)
    first = np.searchsorted(kills, start, side="left")
    return (first < len(kills)) & (kills[np.minimum(first, len(kills) - 1)] <= end)


def component_spans(first_kill_ms, e_h90_ms) -> np.ndarray:
    """Closed outcome windows [f - 10,000, e_h90] (K x 2); an end before the opening is refused."""

    f, e = _ints(first_kill_ms, "first_kill_ms"), _ints(e_h90_ms, "e_h90_ms")
    if f.shape != e.shape:
        raise ValueError("one endpoint per component")
    spans = np.stack((f - ANCHOR_OFFSET_MS, e), axis=1) if len(f) else np.zeros((0, 2), dtype=np.int64)
    if np.any(spans[:, 1] < spans[:, 0]):
        raise ValueError("an outcome window ends before it opens")
    return spans


def _component_table(component_values: np.ndarray) -> np.ndarray:
    values = np.asarray(component_values)
    if values.ndim != 2 or values.shape[1] != len(COMPONENT_COLUMNS):
        raise Blocked("component table width differs from FIELD_MANIFEST component_columns")
    return values


def all_component_spans(component_values: np.ndarray) -> np.ndarray:
    """Outcome windows [f - 10,000, e_h90] of every row of a bank component table (report only since 10-13)."""

    values = _component_table(component_values)
    return component_spans(values[:, COMPONENT_COLUMNS.index("f")], values[:, COMPONENT_COLUMNS.index("e_h90")])


def q_engagement_spans(component_keys, component_values, q_component_keys, tau_ms, D_ms) -> np.ndarray:
    """The outcome windows that exclude placebo candidates (PRESPEC_E1 section 3 as amended 2026-10-13): for each q
    engagement of the match (its q rows, in the given order) the closed [f - 10,000, e_h90] of the one bank component
    with its component_key.  That component must exist exactly once with f - 10,000 = tau, e_h90 = tau + D and
    valid_h90 1, else Blocked (section 7, row-table key mismatch).  Other components give no window."""

    values = _component_table(component_values)
    keys = np.asarray(component_keys)
    q_keys, tau, D = np.asarray(q_component_keys), _ints(tau_ms, "tau_ms"), _ints(D_ms, "D_ms")
    if len(keys) != len(values) or not (q_keys.shape == tau.shape == D.shape):
        raise Blocked("component keys and values, or q keys, tau and D, differ in length")
    col = {name: COMPONENT_COLUMNS.index(name) for name in ("f", "e_h90", "valid_h90")}
    rows = []
    for key, t, d in zip(q_keys.tolist(), tau.tolist(), D.tolist()):
        hit = np.flatnonzero(keys == key)
        if len(hit) != 1:
            raise Blocked(f"q component {bytes(key)!r} is in the bank component table not once ({len(hit)} rows)")
        v = values[hit[0]]
        if int(v[col["f"]]) - ANCHOR_OFFSET_MS != t or int(v[col["e_h90"]]) != t + d:
            raise Blocked(f"q component {bytes(key)!r}: [f - 10 s, e_h90] differs from the row's [tau, tau + D]")
        if int(v[col["valid_h90"]]) != 1:
            raise Blocked(f"q component {bytes(key)!r}: valid_h90 is not 1")
        rows.append(hit[0])
    return component_spans(values[rows, col["f"]], values[rows, col["e_h90"]])


def overlaps_any(spans, start_ms, end_ms) -> np.ndarray:
    """Whether [start, end] meets any closed span (ordered by opening; a running maximum of the closings)."""

    spans = _ints(spans, "spans").reshape(-1, 2)
    start, end = _ints(start_ms, "start_ms"), _ints(end_ms, "end_ms")
    if start.shape != end.shape:
        raise ValueError("one end per start")
    if not len(spans):
        return np.zeros(len(start), dtype=bool)
    if np.any(spans[:, 1] < spans[:, 0]):
        raise ValueError("a span ends before it opens")
    order = np.argsort(spans[:, 0], kind="stable")
    opens, closes = spans[order, 0], np.maximum.accumulate(spans[order, 1])
    last = np.searchsorted(opens, end, side="right") - 1
    return (last >= 0) & (closes[np.maximum(last, 0)] >= start)


def candidate_grid(tau_ms: int) -> np.ndarray:
    """tau + 1,000 k for |k| <= 600, without the starts before 120 s."""

    grid = int(tau_ms) + GRID_OFFSETS_MS
    return grid[grid >= MIN_START_MS]


def key_text(key: Any) -> str:
    """A row-table engagement key (bytes) decoded as ASCII; a str key must be ASCII too."""

    if isinstance(key, (bytes, np.bytes_)):
        return bytes(key).decode("ascii")
    text = str(key)
    text.encode("ascii")
    return text


def _digest(text: str, s: int) -> str:
    return hashlib.sha256((text + ":" + str(int(s))).encode("utf-8")).hexdigest()


def selection_digest(engagement_key: Any, s_ms: Any) -> str:
    """hashlib.sha256((engagement_key.decode("ascii") + ":" + str(int(s))).encode("utf-8")).hexdigest()."""

    if isinstance(s_ms, (float, np.floating)):
        raise TypeError("a start is an integer ms")
    return _digest(key_text(engagement_key), int(s_ms))


def evaluate_engagements(tau_ms, D_ms, kills_ms, spans, frames_ms, game_end_ms, gold_by_frame, gold_edges) -> dict[str, Any]:
    """Every candidate of the engagements of one match with its first failing rule (0 = survivor, i = REJECT_KEYS[i-1]).

    Returns owner (engagement index per candidate, candidates in engagement then grid order), s_ms, reason, keys and
    values of the candidates, and eng_keys and eng_values of the engagements (the same feature function).
    """

    tau, D = _ints(tau_ms, "tau_ms"), _ints(D_ms, "D_ms")
    if tau.shape != D.shape:
        raise ValueError("one D per engagement")
    frames, kills = ef._frames(frames_ms), ef._kills(kills_ms)
    game_end = int(game_end_ms)
    if game_end < 0:
        raise ValueError("the match has no game end")
    eng_values = ef.window_values(tau, D, frames, kills, gold_by_frame)
    eng_keys = ef.keys_from_values(eng_values, tau, gold_edges)
    grid = tau[:, None] + GRID_OFFSETS_MS[None, :]          # the same offsets as candidate_grid
    keep = grid >= MIN_START_MS
    owner = np.broadcast_to(np.arange(len(tau), dtype=np.int64)[:, None], grid.shape)[keep]
    s = grid[keep]
    d = D[owner]
    values = ef.window_values(s, d, frames, kills, gold_by_frame)
    keys = ef.keys_from_values(values, s, gold_edges)
    end = s + d
    differ = np.zeros(len(s), dtype=bool)
    for name in ef.FEATURES:
        differ |= keys[name] != eng_keys[name][owner]
    rules = (keys["start_age_bin"] < 0, (end > frames[-1]) | (end >= game_end), has_kill(kills, s, end),
             overlaps_any(spans, s, end), differ)
    reason = np.zeros(len(s), dtype=np.int8)
    undecided = np.ones(len(s), dtype=bool)
    for code, failed in enumerate(rules, start=1):
        reason[undecided & failed] = code
        undecided &= ~failed
    return {"owner": owner, "s_ms": s, "reason": reason, "keys": keys, "values": values, "eng_keys": eng_keys,
            "eng_values": eng_values}


def choose(owner: np.ndarray, s_ms: np.ndarray, reason: np.ndarray, key_texts: Sequence[str]) -> np.ndarray:
    """Per engagement the candidate position with the smallest selection digest among the survivors (-1 none)."""

    best = np.full(len(key_texts), -1, dtype=np.int64)
    best_digest: list[str | None] = [None] * len(key_texts)
    for position in np.flatnonzero(reason == 0).tolist():
        o = int(owner[position])
        digest = _digest(key_texts[o], int(s_ms[position]))
        if best_digest[o] is None or digest < best_digest[o]:
            best_digest[o], best[o] = digest, position
    return best


def match_windows(eng: Mapping[str, Any], kills_ms, spans, frames_ms, game_end_ms, gold_by_frame,
                  gold_edges) -> tuple[np.ndarray, dict[str, int]]:
    """Chosen placebo start per engagement of one match (-1 none) and the candidates counted per first failing rule.

    ``eng`` holds engagement_key, tau_ms and D_ms (one entry per engagement of the match).
    """

    texts = [key_text(key) for key in np.asarray(eng["engagement_key"]).reshape(-1)]
    c = evaluate_engagements(eng["tau_ms"], eng["D_ms"], kills_ms, spans, frames_ms, game_end_ms, gold_by_frame, gold_edges)
    if len(texts) != len(c["eng_keys"]["time_bin"]):
        raise ValueError("one key per engagement")
    best = choose(c["owner"], c["s_ms"], c["reason"], texts)
    s = np.full(len(texts), -1, dtype=np.int64)
    s[best >= 0] = c["s_ms"][best[best >= 0]]
    return s, {key: int((c["reason"] == code).sum()) for code, key in enumerate(REJECT_KEYS, start=1)}


# ----------------------------------------------------------------------------- summaries
def gate_passed(q_dev_match_rate: float) -> bool:
    return bool(q_dev_match_rate >= GATE_THRESHOLD)


def reuse_summary(match_index: np.ndarray, s_ms: np.ndarray) -> dict[str, Any]:
    """Placebo rows per chosen window (match, s): how many windows serve k engagements."""

    match_index, s = np.asarray(match_index, dtype=np.int64), np.asarray(s_ms, dtype=np.int64)
    rows = s >= 0
    if not rows.any():
        return {"rows": 0, "distinct_windows": 0, "distribution": {}, "rows_in_reused_windows": 0,
                "share_rows_in_reused_windows": None, "max_reuse": 0}
    _, counts = np.unique(np.stack((match_index[rows], s[rows]), axis=1), axis=0, return_counts=True)
    k, n = np.unique(counts, return_counts=True)
    reused = int(counts[counts > 1].sum())
    return {"rows": int(rows.sum()), "distinct_windows": int(len(counts)),
            "distribution": {str(int(a)): int(b) for a, b in zip(k, n)}, "rows_in_reused_windows": reused,
            "share_rows_in_reused_windows": reused / int(rows.sum()), "max_reuse": int(counts.max())}


def overlap_summary(match_index: np.ndarray, s_ms: np.ndarray, D_ms: np.ndarray) -> dict[str, Any]:
    """Placebo rows whose [s, s + D] meets the window of another placebo row of the same match with another start
    (rows sharing a start are reuse, counted by ``reuse_summary``)."""

    match_index, s, D = (np.asarray(a, dtype=np.int64) for a in (match_index, s_ms, D_ms))
    rows = np.flatnonzero(s >= 0)
    order = rows[np.lexsort((s[rows], match_index[rows]))]
    flagged = 0
    bounds = np.flatnonzero(np.r_[True, match_index[order][1:] != match_index[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        own = order[a:b]
        start, end = s[own], s[own] + D[own]
        meets = (start[:, None] <= end[None, :]) & (start[None, :] <= end[:, None]) & (start[:, None] != start[None, :])
        flagged += int(meets.any(axis=1).sum())
    return {"rows": int(len(rows)), "rows_overlapping_another_start": flagged,
            "share": flagged / len(rows) if len(rows) else None}


def smd(engagement: np.ndarray, placebo: np.ndarray) -> float | None:
    """(mean placebo - mean engagement) / sqrt((var_eng + var_plc) / 2), population variances, row weighted."""

    a, b = np.asarray(engagement, dtype=np.float64), np.asarray(placebo, dtype=np.float64)
    if not len(a) or not len(b):
        return None
    pooled = math.sqrt((a.var() + b.var()) / 2)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def role_summary(t: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    """Match rate, per-bin counts of every feature, rejections, reuse, overlap, kill-gap balance and XY of one role."""

    rows = np.flatnonzero(mask)
    s, match = t["s_ms"][rows], t["match_index"][rows]
    matched = s >= 0
    n, m = len(rows), int(matched.sum())
    bins = {}
    for name in ef.FEATURES:
        values = t[name][rows]
        bins[name] = {str(int(v)): {"engagements": int((values == v).sum()), "matched": int(((values == v) & matched).sum()),
                                    "match_rate": _rate(int(((values == v) & matched).sum()), int((values == v).sum()))}
                      for v in np.unique(values)}
    survivors = t["n_survivors"][rows]
    gap_eng, gap_plc = t["kill_gap_eng_ms"][rows], t["kill_gap_plc_ms"][rows]
    both = matched & (gap_eng >= 0) & (gap_plc >= 0)
    kill_bin = t["kill_gap_bin"][rows]
    by_gap = {str(b): {"engagements": int((kill_bin == b).sum()), "matched": int(((kill_bin == b) & matched).sum()),
                       "unmatched": int(((kill_bin == b) & ~matched).sum()),
                       "match_rate": _rate(int(((kill_bin == b) & matched).sum()), int((kill_bin == b).sum())),
                       "share_of_matched": _rate(int(((kill_bin == b) & matched).sum()), m),
                       "share_of_unmatched": _rate(int(((kill_bin == b) & ~matched).sum()), n - m)}
              for b in range(ef.N_KILL_GAP_BINS)}
    xy = t["xy_finite_plc"][rows]
    meets = t["meets_component_window_plc"][rows]
    return {
        "engagements": n, "matches": int(len(np.unique(match))), "matched": m, "unmatched": n - m,
        "match_rate": _rate(m, n), "matched_matches": int(len(np.unique(match[matched]))),
        "bins": bins,
        "candidates": {"grid_points_dropped_before_matching_s_below_120s": int((GRID_POINTS - t["n_candidates"][rows]).sum()),
                       "evaluated": int(t["n_candidates"][rows].sum()),
                       "rejected": {key: int(t["reject_counts"][rows, i].sum()) for i, key in enumerate(REJECT_KEYS)},
                       "survivors": int(survivors.sum()),
                       "survivors_per_matched_engagement": {
                           "min": int(survivors[matched].min()) if m else None,
                           "median": float(np.median(survivors[matched])) if m else None,
                           "mean": float(survivors[matched].mean()) if m else None,
                           "max": int(survivors[matched].max()) if m else None}},
        "reuse": reuse_summary(match, s),
        "overlap": overlap_summary(match, s, t["D_ms"][rows]),
        "kill_gap_balance": {
            "definition": "continuous time since the last kill before the start (s); pairs where both the engagement "
                          "and its placebo window have an earlier kill; smd = (mean placebo - mean engagement) / "
                          "sqrt((var engagement + var placebo) / 2), population variances, row weighted",
            "pairs": int(both.sum()), "pairs_dropped_without_earlier_kill": int(m - both.sum()),
            "mean_engagement_s": float(gap_eng[both].mean() / 1000) if both.any() else None,
            "mean_placebo_s": float(gap_plc[both].mean() / 1000) if both.any() else None,
            "smd": smd(gap_eng[both] / 1000, gap_plc[both] / 1000) if both.any() else None},
        "matched_vs_unmatched_by_kill_gap_bin": by_gap,
        "component_windows": {
            "definition": "matched placebo windows whose [s, s + D] meets the closed [f - 10 s, e_h90] of any bank "
                          "component of the match; the q engagements' windows are excluded by the rule, so these are "
                          "other components' windows (report only, PRESPEC_E1 section 3 as amended 2026-10-13)",
            "matched_windows_meeting_a_component_outcome_window": int(((meets == 1) & matched).sum()),
            "share": _rate(int(((meets == 1) & matched).sum()), m)},
        "xy": {"matched_windows": m, "nonfinite_f_s_xy": int(((xy != 1) & matched).sum())},
    }


def role_summaries(t: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Per-role summaries split for the seal (Ruling 24): TRAIN, Q_DEV and Q_CAL for the public record, the TEST summary
    alone for the sealed file (None without TEST rows)."""

    roles = np.asarray(t["role"]).astype(np.int64)
    public = {ROLE_NAMES[code]: role_summary(t, roles == code) for code in PUBLIC_ROLES if (roles == code).any()}
    return public, (role_summary(t, roles == TEST) if (roles == TEST).any() else None)


def windows_equal(gate: Mapping[str, np.ndarray], table: Mapping[str, np.ndarray]) -> tuple[bool, bool]:
    """(every gate row's engagement key is in the table, the start, D and six features agree on those rows)."""

    position = {bytes(key): i for i, key in enumerate(np.asarray(table["engagement_key"]).tolist())}
    mine = np.asarray([position.get(bytes(key), -1) for key in np.asarray(gate["engagement_key"]).tolist()], dtype=np.int64)
    present = mine >= 0
    same = bool(present.any()) and all(np.array_equal(np.asarray(gate[name])[present], np.asarray(table[name])[mine[present]])
                                       for name in ("s_ms", "D_ms", *ef.FEATURES))
    return bool(present.all()), same


# ----------------------------------------------------------------------------- PRESPEC Appendix A
def read_appendix_edges(text: str) -> np.ndarray:
    """The 3 x 4 gold edges from the Appendix A table of PRESPEC_E1 (rows "| 0 (", "| 1 (", "| 2 (": last four cells)."""

    at = text.find(APPENDIX_HEADING)
    if at < 0:
        raise ValueError("PRESPEC_E1 has no Appendix A")
    rows = {}
    for line in text[at:].splitlines()[1:]:
        if line.startswith("## "):
            break
        match = re.match(r"^\|\s*([012])\s*\(", line)
        if match:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 5:
                raise ValueError(f"Appendix A row has too few cells: {line}")
            rows[int(match.group(1))] = [float(cell) for cell in cells[-4:]]
    if sorted(rows) != [0, 1, 2]:
        raise ValueError("Appendix A holds no complete 3 x 4 gold-edge table")
    return np.asarray([rows[b] for b in range(3)], dtype=np.float64)


# ----------------------------------------------------------------------------- inputs
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


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_rows() -> dict[str, np.ndarray]:
    """q rows (Q_ROWS_h90.npz identity columns) with their tau, h90 endpoint, frames and game end from ENGAGEMENTS.npz.

    The row table's rows are the engagement rows of a q role with valid_h90, in order (rows.py:113-116); any identity
    difference is Blocked (row-table key mismatch).  No label column is read.
    """

    with np.load(Q_ROWS, allow_pickle=False) as table:
        q = {name: table[name] for name in Q_ROW_COLUMNS}
    with np.load(ENGAGEMENTS, allow_pickle=False) as table:
        e = {name: table[name] for name in ENGAGEMENT_COLUMNS}
    keep = np.isin(e["role"].astype(np.int64), list(ROLE_NAMES)) & e["valid_h90"].astype(bool)
    for name in ("canonical_match_key", "component_key", "match_index", "role", "fold", "query_pre", "pre_ms"):
        left, right = e[name][keep], q[name]
        if left.dtype.kind in "iu":
            left, right = left.astype(np.int64), right.astype(np.int64)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise Blocked(f"Q_ROWS_h90.npz differs from ENGAGEMENTS.npz in {name}")
    if not np.array_equal(q["engagement_key"], np.char.add(np.char.add(q["canonical_match_key"], b":"), q["component_key"])):
        raise Blocked("engagement_key is not canonical_match_key:component_key")
    if len(np.unique(q["engagement_key"])) != len(q["engagement_key"]):
        raise Blocked("engagement keys are not unique")
    rows = {"engagement_key": q["engagement_key"], "canonical_match_key": q["canonical_match_key"],
            "component_key": q["component_key"], "match_index": q["match_index"].astype(np.int64),
            "role": q["role"].astype(np.int64), "tau_ms": q["pre_ms"].astype(np.int64),
            "e_h90_ms": e["post_ms_h90"][keep].astype(np.int64), "F_pre_ms": e["F_pre_ms"][keep].astype(np.int64),
            "F_post_ms": e["F_post_ms_h90"][keep].astype(np.int64), "game_end_ms": e["game_end_ms"][keep].astype(np.int64)}
    rows["D_ms"] = rows["e_h90_ms"] - rows["tau_ms"]
    if np.any(rows["D_ms"] < ANCHOR_OFFSET_MS):
        raise Blocked("an engagement window does not reach its first kill (D < 10 s)")
    return rows


def load_bank_tables() -> dict[str, Any]:
    """MATCH_TABLE identity columns, shard bindings and the bank layout checks."""

    metadata = _read_json(PROJECTION / "BANK_METADATA.json")
    if Path(metadata["source_root"]).resolve() != BANK.resolve():
        raise Blocked("the projection's source is not the static-preserved bank")
    manifest = _read_json(BANK / "FIELD_MANIFEST.json")
    if tuple(manifest["component_columns"]) != COMPONENT_COLUMNS:
        raise Blocked("bank component columns differ from engagements.COMPONENT_COLUMNS")
    if list(manifest["team_order"]) != [100, 200] or manifest["event_type_names"][CHAMPION_KILL] != "CHAMPION_KILL":
        raise Blocked("bank team order or event codes differ from the PRESPEC reading")
    matches = np.load(PROJECTION / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    table = {name: np.asarray(matches[name]) for name in ("key", "shard", "local_match", "role")}
    return {"matches": table, "bindings": metadata["source_shards"]}


def shard_tasks(rows: Mapping[str, np.ndarray], select: np.ndarray, bank: Mapping[str, Any], kind: str,
                gold_edges: np.ndarray | None, only: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """One task per bank shard holding selected rows (shard order), with each row's match coordinates.

    A selection covers whole matches (every q row of a selected match), so a worker sees all the q engagements whose
    outcome windows exclude candidates in that match."""

    matches = bank["matches"]
    index = rows["match_index"]
    if not np.array_equal(matches["key"][index], rows["canonical_match_key"]):
        raise Blocked("MATCH_TABLE keys differ from the q rows' match keys")
    if not np.array_equal(matches["role"][index].astype(np.int64), rows["role"]):
        raise Blocked("MATCH_TABLE roles differ from the q rows' roles")
    select = np.asarray(select, dtype=bool)
    every = np.bincount(index, minlength=len(matches["key"]))
    chosen = np.bincount(index[select], minlength=len(matches["key"]))
    if np.any((chosen > 0) & (chosen != every)):
        raise ValueError("a selection must hold every q row of each selected match")
    picked = np.flatnonzero(select)
    shard = matches["shard"][index[picked]].astype(np.int64)
    tasks = []
    for value in np.unique(shard).tolist():
        binding = bank["bindings"][value]
        if only is not None and binding["directory"] not in only:
            continue
        own = picked[shard == value]
        tasks.append({"kind": kind, "shard": int(value), "directory": str(BANK / binding["directory"]),
                      "archive_sha256": binding["archive_sha256"], "match_ledger_sha256": binding["match_ledger_sha256"],
                      "rows": own, "local": matches["local_match"][index[own]].astype(np.int64),
                      **{name: rows[name][own] for name in ("match_index", "canonical_match_key", "component_key",
                                                             "engagement_key", "tau_ms", "D_ms", "F_pre_ms", "F_post_ms",
                                                             "game_end_ms")},
                      "gold_edges": gold_edges})
    return tasks


# ----------------------------------------------------------------------------- worker
def _slice(offsets: np.ndarray, local: int) -> slice:
    return slice(int(offsets[local]), int(offsets[local + 1]))


def shard_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """One bank shard: gold at F(tau) (kind "gold") or the windows of its rows (kind "windows")."""

    tick = time.perf_counter()
    directory = Path(task["directory"])
    if sha256_file(directory / "BANK.npz") != task["archive_sha256"]:
        raise Blocked(f"{directory.name}/BANK.npz differs from the projection binding")
    if sha256_file(directory / "MATCH_LEDGER.jsonl") != task["match_ledger_sha256"]:
        raise Blocked(f"{directory.name}/MATCH_LEDGER.jsonl differs from the projection binding")
    with (directory / "MATCH_LEDGER.jsonl").open(encoding="utf-8") as stream:
        ledger = [json.loads(line)["canonical_match_key"] for line in stream if line.strip()]
    fields = GOLD_FIELDS if task["kind"] == "gold" else SHARD_FIELDS
    with np.load(directory / "BANK.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in fields}
        offsets = {name: archive[f"offset_{name}"] for name in fields}
    for name in fields:
        if len(offsets[name]) != len(ledger) + 1:
            raise Blocked(f"{directory.name}: offsets of {name} do not cover the ledger")
    rows, local = np.asarray(task["rows"]), np.asarray(task["local"])
    n = len(rows)
    out: dict[str, np.ndarray] = {"rows": rows}
    if task["kind"] == "gold":
        out.update(F_tau_ms=np.full(n, -1, np.int64), gold=np.full(n, np.nan), time_bin=np.full(n, -1, np.int64))
    else:
        out.update(s_ms=np.full(n, -1, np.int64), n_candidates=np.zeros(n, np.int32), n_survivors=np.zeros(n, np.int32),
                   reject_counts=np.zeros((n, len(REJECT_KEYS)), np.int32), xy_finite_plc=np.full(n, -1, np.int8),
                   meets_component_window_plc=np.full(n, -1, np.int8),
                   **{name: np.full(n, -1, np.int64) for name in ef.FEATURES},
                   **{f"{name}_{arm}": np.full(n, -1, np.int64) for name in ("start_age", "kill_gap") for arm in ("eng", "plc")},
                   gold_eng=np.full(n, np.nan), gold_plc=np.full(n, np.nan))
    for value in np.unique(local).tolist():
        own = np.flatnonzero(local == value)          # every q row of this match (shard_tasks covers whole matches)
        key = bytes(task["canonical_match_key"][own[0]]).decode("ascii")
        if ledger[value] != key:
            raise Blocked(f"{directory.name}: ledger key of local match {value} differs from MATCH_TABLE")
        frames = arrays["minute_ts"][_slice(offsets["minute_ts"], value)].astype(np.int64)
        gold_by_frame = ef.team_gold_difference(arrays["gold_team"][_slice(offsets["gold_team"], value)])
        if len(gold_by_frame) != len(frames):
            raise Blocked(f"{key}: team gold and frame clock differ")
        tau, D = task["tau_ms"][own], task["D_ms"][own]
        first = ef.frame_at(frames, tau)
        if np.any(first < 0) or not np.array_equal(frames[np.maximum(first, 0)], task["F_pre_ms"][own]):
            raise Blocked(f"{key}: F(tau) differs from ENGAGEMENTS F_pre_ms")
        if task["kind"] == "gold":
            out["F_tau_ms"][own], out["gold"][own], out["time_bin"][own] = frames[first], gold_by_frame[first], ef.time_bin(tau)
            continue
        game_end = arrays["end_ms"][_slice(offsets["end_ms"], value)]
        if len(game_end) != 1 or int(game_end[0]) < 0 or np.any(task["game_end_ms"][own] != int(game_end[0])):
            raise Blocked(f"{key}: game end missing or different from ENGAGEMENTS")
        times = arrays["event_timestamp"][_slice(offsets["event_timestamp"], value)].astype(np.int64)
        kinds = arrays["event_type_code"][_slice(offsets["event_type_code"], value)]
        kills = np.sort(times[kinds == CHAMPION_KILL])
        components = arrays["component_values"][_slice(offsets["component_values"], value)]
        component_keys = arrays["component_key"][_slice(offsets["component_key"], value)]
        spans = q_engagement_spans(component_keys, components, task["component_key"][own], tau, D)   # q engagements only
        last = ef.frame_at(frames, tau + D)
        if not np.array_equal(frames[np.maximum(last, 0)], task["F_post_ms"][own]):
            raise Blocked(f"{key}: F(e_h90) differs from ENGAGEMENTS F_post_ms_h90")
        c = evaluate_engagements(tau, D, kills, spans, frames, int(game_end[0]), gold_by_frame, task["gold_edges"])
        if np.any(c["eng_keys"]["start_age_bin"] < 0):
            raise Blocked(f"{key}: an engagement start age lies outside [0, 10 s]")
        best = choose(c["owner"], c["s_ms"], c["reason"], [key_text(k) for k in task["engagement_key"][own]])
        counts = np.zeros((len(own), len(REJECT_KEYS) + 1), dtype=np.int64)
        np.add.at(counts, (c["owner"], c["reason"].astype(np.int64)), 1)
        out["n_candidates"][own] = counts.sum(axis=1)
        out["n_survivors"][own] = counts[:, 0]
        out["reject_counts"][own] = counts[:, 1:]
        for name in ef.FEATURES:
            out[name][own] = c["eng_keys"][name]
        out["start_age_eng"][own] = c["eng_values"]["start_age_ms"]
        out["kill_gap_eng"][own] = c["eng_values"]["kill_gap_ms"]
        out["gold_eng"][own] = c["eng_values"]["gold"]
        chosen = best >= 0
        if chosen.any():
            at, rows_chosen = best[chosen], own[chosen]
            out["s_ms"][rows_chosen] = c["s_ms"][at]
            out["start_age_plc"][rows_chosen] = c["values"]["start_age_ms"][at]
            out["kill_gap_plc"][rows_chosen] = c["values"]["kill_gap_ms"][at]
            out["gold_plc"][rows_chosen] = c["values"]["gold"][at]
            xy = arrays["xy"][_slice(offsets["xy"], value)]
            if len(xy) != len(frames):
                raise Blocked(f"{key}: xy frames and frame clock differ")
            out["xy_finite_plc"][rows_chosen] = np.isfinite(xy[c["values"]["start_frame"][at]]).all(axis=(1, 2)).astype(np.int8)
            out["meets_component_window_plc"][rows_chosen] = overlaps_any(
                all_component_spans(components), c["s_ms"][at], c["s_ms"][at] + D[chosen]).astype(np.int8)
    return {"arrays": out, "shard": int(task["shard"]), "directory": directory.name, "rows": n,
            "seconds": time.perf_counter() - tick, "peak_memory_bytes": peak_memory_bytes(), "pid": os.getpid()}


def run_tasks(tasks: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Shard tasks on at most ``workers`` processes (spawn); results in task order.  Progress shows no counts."""

    workers = max(1, min(int(workers), MAX_WORKERS))
    results: dict[int, dict[str, Any]] = {}
    tick = time.perf_counter()
    if workers == 1:
        for i, task in enumerate(tasks):
            results[i] = shard_task(task)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(shard_task, task): i for i, task in enumerate(tasks)}
            for done, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if done % 25 == 0 or done == len(tasks):
                    print(json.dumps({"shards_done": done, "of": len(tasks), "seconds": round(time.perf_counter() - tick, 1)}),
                          flush=True)
    return [results[i] for i in range(len(tasks))]


def _assemble(results: Sequence[Mapping[str, Any]], n: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-row arrays of all shard results placed by row index, and the mask of rows that were processed."""

    names = [name for name in results[0]["arrays"] if name != "rows"] if results else []
    out: dict[str, np.ndarray] = {}
    done = np.zeros(n, dtype=bool)
    for result in results:
        rows = result["arrays"]["rows"]
        if done[rows].any():
            raise ValueError("a row was processed twice")
        done[rows] = True
        for name in names:
            values = result["arrays"][name]
            if name not in out:
                out[name] = np.full((n, *values.shape[1:]), -1 if values.dtype.kind in "iu" else np.nan, dtype=values.dtype)
            out[name][rows] = values
    return out, done


def _shard_record(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seconds = [r["seconds"] for r in results]
    peaks = [r["peak_memory_bytes"] for r in results if r.get("peak_memory_bytes")]
    return {"shards": len(results), "directories": [r["directory"] for r in results],
            "seconds_sum": float(sum(seconds)), "seconds_max": float(max(seconds, default=0.0)),
            "worker_processes": len({r["pid"] for r in results}),
            "worker_peak_memory_bytes_max": max(peaks, default=None)}


def _shard_hashes(results: Sequence[Mapping[str, Any]], bank: Mapping[str, Any]) -> dict[str, Any]:
    """BANK.npz and MATCH_LEDGER.jsonl hashes of the shards read (checked against the binding by each worker)."""

    listed = {r["directory"]: {"BANK.npz": bank["bindings"][r["shard"]]["archive_sha256"],
                               "MATCH_LEDGER.jsonl": bank["bindings"][r["shard"]]["match_ledger_sha256"]} for r in results}
    lines = "".join(f"{name}:{h['BANK.npz']}\n" for name, h in sorted(listed.items()))
    return {"shards": listed, "bank_npz_list_sha256": hashlib.sha256(lines.encode("ascii")).hexdigest(),
            "list_rule": "sha256 of the sorted lines '<directory>:<BANK.npz sha256>\\n'"}


def _versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__}


def _code() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in CODE_FILES}


def _inputs() -> dict[str, str]:
    paths = (Q_ROWS, ENGAGEMENTS, PROJECTION / "MATCH_TABLE.npy", PROJECTION / "BANK_METADATA.json", BANK / "FIELD_MANIFEST.json")
    return {path.as_posix(): sha256_file(path) for path in paths}


def _stage(name: str, tick: float, **extra: Any) -> dict[str, Any]:
    return {"stage": name, "finished_utc": utc(), "wall_seconds": time.perf_counter() - tick,
            "peak_memory_bytes_driver": peak_memory_bytes(), **extra}


def _blocked(out: Path, which: str, reasons: Any, **extra: Any) -> int:
    """Write E1_WINDOWS_BLOCKED_<which>_<n>.json once (first free n) and return 2."""

    out.mkdir(parents=True, exist_ok=True)
    k = 1
    while (out / f"E1_WINDOWS_BLOCKED_{which}_{k}.json").exists():
        k += 1
    path = out / f"E1_WINDOWS_BLOCKED_{which}_{k}.json"
    write_once(path, _json_safe({"version": VERSION, "experiment": "E1", "blocked_stage": which, "blocked": True,
                                 "utc": utc(), "reasons": reasons, **extra}))
    print(json.dumps({"BLOCKED": which, "file": str(path), "reasons": _json_safe(reasons)}, ensure_ascii=False), flush=True)
    return 2


def _refuse_smoke_in_main(out: Path, smoke: bool) -> None:
    if smoke and Path(out).resolve() == TOG_E1.resolve():
        raise ValueError("a smoke run never writes into tog_e1_1")
    if not smoke and Path(out).resolve() != TOG_E1.resolve():
        raise ValueError("a full run writes only into tog_e1_1 (use --smoke for another folder)")


# ----------------------------------------------------------------------------- stages
def stage_gold(out: Path, workers: int, smoke: bool = False, only: Sequence[str] | None = None) -> int:
    """F(tau) gold difference of the TRAIN q rows and the 3 x 4 edges, before any window file exists."""

    started, tick = utc(), time.perf_counter()
    _refuse_smoke_in_main(out, smoke)
    out.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in out.glob("WINDOWS_h90*")] + [name for name in (GOLD_JSON, GOLD_ROWS) if (out / name).exists()]
    if existing:
        raise FileExistsError(f"gold edges come before any window and are written once; found {existing}")
    rows = load_rows()
    bank = load_bank_tables()
    stages = [_stage("load", tick)]
    tick_match = time.perf_counter()
    train = rows["role"] == 0
    results = run_tasks(shard_tasks(rows, train, bank, "gold", None, only), workers)
    arrays, done = _assemble(results, len(rows["role"]))
    stages.append(_stage("gold_at_F_tau", tick_match, **_shard_record(results)))
    if not smoke and not np.array_equal(done, train):
        raise ValueError("not every TRAIN q row was read")
    take = np.flatnonzero(done)
    gold, tbin = arrays["gold"][take], arrays["time_bin"][take]
    edges = ef.gold_edges(gold, tbin)
    gold_rows = {"engagement_key": rows["engagement_key"][take], "match_index": rows["match_index"][take].astype(np.int32),
                 "tau_ms": rows["tau_ms"][take], "F_tau_ms": arrays["F_tau_ms"][take], "time_bin": tbin.astype(np.int8),
                 "gold_diff": gold}
    rows_sha = write_npz_once(out / GOLD_ROWS, gold_rows)
    per_bin = {str(b): {"rows": int((tbin == b).sum()), "matches": int(len(np.unique(gold_rows["match_index"][tbin == b]))),
                        "min": float(gold[tbin == b].min()), "max": float(gold[tbin == b].max())} for b in range(3)}
    record = {"version": VERSION, "experiment": "E1", "stage": "gold", "smoke": smoke, "started_utc": started, "utc": utc(),
              "rule": "TRAIN q rows (Q_ROWS_h90 role 0); F(tau) = last bank frame at or before tau; gold = gold_team[F, 0] "
                      "- gold_team[F, 1] in float64 (team_order [100, 200], blue - red); time_bin of tau (< 15, [15, 25), "
                      ">= 25 min); per time_bin numpy.quantile(gold, [0.2, 0.4, 0.6, 0.8], method='linear') "
                      "(PRESPEC_E1 section 3 gold_bin)",
              "quantiles": list(ef.GOLD_QUANTILES), "time_edges_min": list(ef.TIME_EDGES_MIN),
              "edges": edges.tolist(), "edges_repr": [[repr(float(v)) for v in row] for row in edges],
              "rows": int(len(take)), "per_time_bin": per_bin,
              "gold_rows_file": {"path": str(out / GOLD_ROWS), "sha256": rows_sha, "rows": int(len(take))},
              "inputs": {**_inputs(), "bank_shards": _shard_hashes(results, bank)}, "code_sha256": _code(),
              "versions": _versions(), "git_head": git_head(), "stages": stages,
              "peak_memory_bytes": {"driver": peak_memory_bytes(), "workers_max": _shard_record(results)["worker_peak_memory_bytes_max"]}}
    write_once(out / GOLD_JSON, _json_safe(record))
    print(json.dumps({"stage": "gold", "rows": int(len(take)), "per_time_bin_rows": {b: v["rows"] for b, v in per_bin.items()},
                      "edges": record["edges_repr"], "gold_json_sha256": sha256_file(out / GOLD_JSON),
                      "gold_rows_sha256": rows_sha}, indent=1), flush=True)
    return 0


def _checked_edges(out: Path, gold_path: Path | None) -> tuple[np.ndarray | None, dict[str, Any], list[str]]:
    """Gold edges from the gold record, which must equal PRESPEC_E1 Appendix A in the committed PRESPEC."""

    path = Path(gold_path) if gold_path else out / GOLD_JSON
    problems: list[str] = []
    record = _read_json(path)
    edges = np.asarray([[float(v) for v in row] for row in record["edges_repr"]], dtype=np.float64)
    if record.get("smoke") or not np.array_equal(edges, np.asarray(record["edges"], dtype=np.float64)):
        problems.append("the gold record is a smoke record or its edges differ from their repr")
    prespec = document_record(PRESPEC)
    try:
        appendix = read_appendix_edges(PRESPEC.read_text(encoding="utf-8"))
        if not np.array_equal(appendix, edges):
            problems.append("gold edges differ from PRESPEC_E1 Appendix A")
    except ValueError as error:
        problems.append(f"PRESPEC_E1 Appendix A: {error}")
    gold_sha = sha256_file(path)
    if gold_sha not in PRESPEC.read_text(encoding="utf-8"):
        problems.append("PRESPEC_E1 Appendix A does not cite the gold record's SHA-256")
    if not prespec["commit"] or not prespec["unchanged_since_commit"]:
        problems.append("PRESPEC_E1 with Appendix A is not committed or changed since its commit")
    return (None if problems else edges), {"path": str(path), "sha256": gold_sha, "prespec": prespec,
                                           "amendment": document_record(AMENDMENT)}, problems


def _window_columns(rows: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray], take: np.ndarray) -> dict[str, np.ndarray]:
    small = {"start_age_bin": np.int8, "n_bound": np.int16, "end_age_bin": np.int8, "time_bin": np.int8,
             "kill_gap_bin": np.int8, "gold_bin": np.int8}
    columns = {"engagement_key": rows["engagement_key"][take], "match_index": rows["match_index"][take].astype(np.int32),
               "role": rows["role"][take].astype(np.int8), "tau_ms": rows["tau_ms"][take], "D_ms": rows["D_ms"][take],
               "s_ms": arrays["s_ms"][take]}
    columns.update({name: arrays[name][take].astype(dtype) for name, dtype in small.items()})
    columns.update({f"{name}_ms": arrays[name][take] for name in ("start_age_eng", "start_age_plc", "kill_gap_eng", "kill_gap_plc")})
    columns.update({name: arrays[name][take] for name in ("gold_eng", "gold_plc", "n_candidates", "n_survivors",
                                                           "reject_counts", "xy_finite_plc", "meets_component_window_plc")})
    return columns


def _superseded(out: Path) -> dict[str, Any]:
    """Records of the superseded gate attempts (kept unchanged; their hashes go into the E1 lock, PRESPEC_E1 section 6)."""

    records = {}
    for attempt in SUPERSEDED_ATTEMPTS:
        path, npz = out / f"WINDOWS_h90_QDEV_GATE_{attempt}.json", out / f"WINDOWS_h90_QDEV_GATE_{attempt}.npz"
        if path.exists():
            old = _read_json(path)
            records[attempt] = {"path": str(path), "sha256": sha256_file(path), "npz_sha256": sha256_file(npz) if npz.exists() else None,
                                "q_dev_match_rate": old["gate"]["q_dev_match_rate"], "passed": old["gate"]["passed"],
                                "git_head": old.get("git_head"), "rule": old.get("rule")}
    return records


def _summary_table(columns: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in columns.items()}


def _run_windows(rows, bank, select, edges, workers, only) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    results = run_tasks(shard_tasks(rows, select, bank, "windows", edges, only), workers)
    arrays, done = _assemble(results, len(rows["role"]))
    return arrays, done, results


def stage_gate(out: Path, workers: int, smoke: bool = False, only: Sequence[str] | None = None,
               gold_path: Path | None = None) -> int:
    """Q_DEV windows with the amended PRESPEC rule (attempt r1) and the 30% gate (Ruling 27); a section 7 stop is
    written as a BLOCKED record."""

    try:
        return _gate_run(Path(out), workers, smoke, only, gold_path)
    except Blocked as error:
        return _blocked(Path(out), "gate", [str(error)])


def _gate_run(out: Path, workers: int, smoke: bool, only: Sequence[str] | None, gold_path: Path | None) -> int:
    started, tick = utc(), time.perf_counter()
    _refuse_smoke_in_main(out, smoke)
    for name in (f"{GATE_NAME}.npz", f"{GATE_NAME}.json"):
        if (out / name).exists():
            raise FileExistsError(f"{out / name} exists; gate files are written once per attempt")
    edges, gold, problems = _checked_edges(out, gold_path)
    if problems and not smoke:
        return _blocked(out, "gate", problems, gold=gold)
    if edges is None:                                  # smoke runs may use an uncommitted record, recorded below
        record = _read_json(Path(gold_path) if gold_path else out / GOLD_JSON)
        edges = np.asarray(record["edges"], dtype=np.float64)
    rows = load_rows()
    bank = load_bank_tables()
    stages = [_stage("load", tick)]
    tick_match = time.perf_counter()
    q_dev = rows["role"] == Q_DEV
    arrays, done, results = _run_windows(rows, bank, q_dev, edges, workers, only)
    stages.append(_stage("windows_q_dev", tick_match, **_shard_record(results)))
    if not smoke and not np.array_equal(done, q_dev):
        return _blocked(out, "gate", ["not every Q_DEV row was processed"])
    take = np.flatnonzero(done)
    columns = _window_columns(rows, arrays, take)
    summary = role_summary(_summary_table(columns), np.ones(len(take), dtype=bool))
    rate = summary["match_rate"]
    passed = gate_passed(rate) if rate is not None else False
    xy_ok = summary["xy"]["nonfinite_f_s_xy"] == 0
    npz_sha = write_npz_once(out / f"{GATE_NAME}.npz", columns)
    record = {"version": VERSION, "experiment": "E1", "stage": "qdev_gate", "attempt": GATE_ATTEMPT, "smoke": smoke,
              "smoke_shards": list(only) if only else None, "started_utc": started, "utc": utc(), "rule": RULE,
              "superseded_gate_records": _superseded(out),
              "gold_edges": edges.tolist(), "gold_record": gold, "gold_problems": problems,
              "gate": {"rule": "matched Q_DEV engagements / Q_DEV q engagements with valid h90 >= 0.30 (PRESPEC_E1 "
                               "section 6 step 2); below: stop and ask the author (default relaxation order gold "
                               "terciles, then |k| <= 900)", "threshold": GATE_THRESHOLD, "q_dev_match_rate": rate,
                       "passed": passed},
              "xy_check": {"passed": xy_ok, "nonfinite_f_s_xy": summary["xy"]["nonfinite_f_s_xy"]},
              "Q_DEV": summary, "columns": COLUMN_NOTE,
              "windows_npz": {"path": str(out / f"{GATE_NAME}.npz"), "sha256": npz_sha, "rows": int(len(take))},
              "inputs": {**_inputs(), "bank_shards": _shard_hashes(results, bank)}, "code_sha256": _code(),
              "versions": _versions(), "git_head": git_head(), "workers": workers, "stages": stages,
              "peak_memory_bytes": {"driver": peak_memory_bytes(), "workers_max": _shard_record(results)["worker_peak_memory_bytes_max"]}}
    write_once(out / f"{GATE_NAME}.json", _json_safe(record))
    print(json.dumps({"stage": "qdev_gate", "attempt": GATE_ATTEMPT, "q_dev_engagements": summary["engagements"],
                      "q_dev_matched": summary["matched"], "q_dev_match_rate": rate, "gate_passed": passed,
                      "xy_nonfinite": summary["xy"]["nonfinite_f_s_xy"], "rejected": summary["candidates"]["rejected"],
                      "kill_gap_smd": summary["kill_gap_balance"]["smd"], "placebo_overlap_share": summary["overlap"]["share"],
                      "component_window_share": summary["component_windows"]["share"],
                      "gate_json_sha256": sha256_file(out / f"{GATE_NAME}.json"), "gate_npz_sha256": npz_sha}, indent=1),
          flush=True)
    if not xy_ok:
        return _blocked(out, "gate_xy", [f"{summary['xy']['nonfinite_f_s_xy']} chosen Q_DEV windows lack finite F(s) XY"])
    return 0 if passed else 3


def stage_final(out: Path, workers: int, smoke: bool = False, only: Sequence[str] | None = None,
                gold_path: Path | None = None) -> int:
    """Windows of every q role with the passed gate's rule, written once; TEST summaries only in the sealed file; a
    section 7 stop is written as a BLOCKED record."""

    try:
        return _final_run(Path(out), workers, smoke, only, gold_path)
    except Blocked as error:
        return _blocked(Path(out), "final", [str(error)])


def _final_run(out: Path, workers: int, smoke: bool, only: Sequence[str] | None, gold_path: Path | None) -> int:
    started, tick = utc(), time.perf_counter()
    _refuse_smoke_in_main(out, smoke)
    if only and any(name.startswith("test_") for name in only):
        raise ValueError("a smoke run does not read TEST shards")
    for name in (f"{FINAL_NAME}.npz", f"{FINAL_NAME}.json", SEALED_NAME):
        if (out / name).exists():
            raise FileExistsError(f"{out / name} exists; the final windows are written once")
    edges, gold, problems = _checked_edges(out, gold_path)
    gate_path = out / f"{GATE_NAME}.json"
    gate = _read_json(gate_path) if gate_path.exists() else None
    if gate is None:
        problems.append("no Q_DEV gate record")
    else:
        if not gate["gate"]["passed"] or not gate["xy_check"]["passed"]:
            problems.append("the Q_DEV gate did not pass")
        if sha256_file(out / f"{GATE_NAME}.npz") != gate["windows_npz"]["sha256"]:
            problems.append("the gate windows differ from their record")
        if gate.get("smoke") != smoke or gate["gold_edges"] != (edges.tolist() if edges is not None else gate["gold_edges"]):
            problems.append("the gate used another run kind or other gold edges")
    if problems and not smoke:
        return _blocked(out, "final", problems, gold=gold)
    if edges is None:
        edges = np.asarray(_read_json(Path(gold_path) if gold_path else out / GOLD_JSON)["edges"], dtype=np.float64)
    rows = load_rows()
    bank = load_bank_tables()
    stages = [_stage("load", tick)]
    tick_match = time.perf_counter()
    arrays, done, results = _run_windows(rows, bank, np.ones(len(rows["role"]), dtype=bool), edges, workers, only)
    stages.append(_stage("windows_all_roles", tick_match, **_shard_record(results)))
    if not smoke and not done.all():
        return _blocked(out, "final", ["not every q row was processed"])
    tick_write = time.perf_counter()
    take = np.flatnonzero(done)
    columns = _window_columns(rows, arrays, take)
    table = _summary_table(columns)
    # Q_DEV windows must equal the gate's (same rule, deterministic)
    same = None
    if gate is not None:
        with np.load(out / f"{GATE_NAME}.npz", allow_pickle=False) as archive:
            gate_cols = {name: archive[name] for name in ("engagement_key", "s_ms", "D_ms", *ef.FEATURES)}
        all_present, same = windows_equal(gate_cols, table)
        if not smoke and not (all_present and same):
            return _blocked(out, "final", ["the final Q_DEV windows differ from the gate windows"])
    public, sealed_summary = role_summaries(table)
    nonfinite_public = {name: s["xy"]["nonfinite_f_s_xy"] for name, s in public.items()}
    xy_ok = int(((table["xy_finite_plc"] != 1) & (table["s_ms"] >= 0)).sum()) == 0
    npz_sha = write_npz_once(out / f"{FINAL_NAME}.npz", columns)
    sealed = {"version": VERSION, "experiment": "E1", "stage": "windows_TEST_sealed", "smoke": smoke, "utc": utc(),
              "note": "every TEST window summary; nobody views this file before the E1 lock (PRESPEC_E1 section 6, "
                      "Ruling 24); its SHA-256 enters the lock", "windows_npz_sha256": npz_sha, "TEST": sealed_summary}
    write_once(out / SEALED_NAME, _json_safe(sealed))
    sealed_sha = sha256_file(out / SEALED_NAME)
    stages.append(_stage("summaries_and_writes", tick_write))
    record = {"version": VERSION, "experiment": "E1", "stage": "final_windows", "smoke": smoke,
              "smoke_shards": list(only) if only else None, "started_utc": started, "utc": utc(), "rule": RULE,
              "gold_edges": edges.tolist(), "gold_record": gold, "gold_problems": problems,
              "gate_record": {"path": str(gate_path), "sha256": sha256_file(gate_path) if gate is not None else None,
                              "npz_sha256": sha256_file(out / f"{GATE_NAME}.npz") if gate is not None else None,
                              "attempt": GATE_ATTEMPT, "q_dev_match_rate": gate["gate"]["q_dev_match_rate"] if gate else None,
                              "passed": gate["gate"]["passed"] if gate else None},
              "superseded_gate_records": _superseded(out),
              "qdev_equals_gate": same,
              "xy_check": {"passed": xy_ok, "nonfinite_f_s_xy_public_roles": nonfinite_public,
                           "note": "the TEST count is in the sealed summary"},
              "roles": public, "columns": COLUMN_NOTE,
              "windows_npz": {"path": str(out / f"{FINAL_NAME}.npz"), "sha256": npz_sha,
                              "row_order": "q_run_1/Q_ROWS_h90.npz rows (all q roles), s_ms = -1 unmatched"},
              "test_sealed_summary": {"path": str(out / SEALED_NAME), "sha256": sealed_sha},
              "inputs": {**_inputs(), "bank_shards": _shard_hashes(results, bank)}, "code_sha256": _code(),
              "versions": _versions(), "git_head": git_head(), "workers": workers, "stages": stages,
              "peak_memory_bytes": {"driver": peak_memory_bytes(), "workers_max": _shard_record(results)["worker_peak_memory_bytes_max"]}}
    write_once(out / f"{FINAL_NAME}.json", _json_safe(record))
    print(json.dumps({"stage": "final_windows", "match_rate": {name: s["match_rate"] for name, s in public.items()},
                      "matched": {name: s["matched"] for name, s in public.items()},
                      "engagements": {name: s["engagements"] for name, s in public.items()},
                      "kill_gap_smd": {name: s["kill_gap_balance"]["smd"] for name, s in public.items()},
                      "qdev_equals_gate": same, "xy_nonfinite_public": nonfinite_public, "xy_check_passed": xy_ok,
                      "windows_npz_sha256": npz_sha, "windows_json_sha256": sha256_file(out / f"{FINAL_NAME}.json"),
                      "test_sealed_sha256": sealed_sha}, indent=1), flush=True)
    if not xy_ok:
        return _blocked(out, "final_xy", ["a chosen window lacks finite F(s) XY (public counts in WINDOWS_h90.json, the "
                                          "TEST count in the sealed summary)"], nonfinite_public=nonfinite_public)
    return 0


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
    parser.add_argument("stage", choices=("gold", "gate", "final"))
    parser.add_argument("--out", default=str(TOG_E1))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--smoke", action="store_true", help="a check run into another folder on a few shards")
    parser.add_argument("--shards", default=None, help="comma-separated bank shard directories (smoke only)")
    parser.add_argument("--gold", default=None, help="gold record to use (default <out>/GOLD_EDGES_h90.json)")
    args = parser.parse_args(argv)
    only = [name for name in args.shards.split(",") if name] if args.shards else None
    if only and not args.smoke:
        parser.error("--shards needs --smoke")
    _below_normal_priority()
    out = Path(args.out)
    if args.stage == "gold":
        return stage_gold(out, args.workers, args.smoke, only)
    if args.stage == "gate":
        return stage_gate(out, args.workers, args.smoke, only, Path(args.gold) if args.gold else None)
    return stage_final(out, args.workers, args.smoke, only, Path(args.gold) if args.gold else None)


if __name__ == "__main__":
    raise SystemExit(main())
