"""T4.2 E4 labels and their agreement with the main labels (PLAN_E1_E5_KO.md Task 11; Task 12 extends this file).

Binding specification: reports/tog_experiments_20261012/PRESPEC_E4_KO.md (section 3: labels, TEST handling and
agreement; section 6 order, steps 2-3; SDD Rulings 19, 43, 44).

  labels     y4 = 1[p_post - p_pre > 0] (int8) on the stored raw probabilities of Task 10's PRED_<ctx>.npz (the LightGBM
             V's predict_proba, the quantity the V search scored; taken as stored, no calibration and no logit round
             trip: Ruling 43).  An exact zero is negative with exact_zero = 1.  near_zero_bits as the main labels
             (assemble.py:40 NEAR_ZERO = (0.005, 0.01, 0.02), :125-127): bit k is set when |p_post - p_pre| <
             NEAR_ZERO[k].  label_svi_delta4 = p_post - p_pre in float64 (assemble.py:121).  Rows: the TRAIN, Q_DEV and
             Q_CAL rows of q_run_1/Q_ROWS_h90.npz in its order, with Task 10's scoring context (TRAIN: oof<fold>;
             Q_DEV/Q_CAL: final) and both of its queries (query_pre; the h90 endpoint ENGAGEMENTS query_h90).
  agreement  Q_DEV rows: the share whose main label y (labels_1/LABELS_h90.npz direction_positive, matched to the row
             table by key and order) equals y4, overall and by |main SVI| < 0.01, [0.01, 0.05), >= 0.05.  Match weighted
             is primary: every match present weighs 1 (the mean within the match, then the mean over matches; weights
             recomputed inside each bin as qeval.metrics_by_stratum, D-Q13); row weighted is secondary.  Exact-zero rows
             stay in the denominator.  Cohen's kappa (secondary) and the exact-zero share in both weightings.

TEST rows (Ruling 44): nothing about them is computed, loaded into a label or summarized.  They are dropped from the row
list before any probability or label value is used.  PRED_final.npz and LABELS_h90.npz also hold TEST rows outside any
seal: their columns are loaded whole (an npz member has no partial read) and indexed only at TRAIN, Q_DEV and Q_CAL
positions.  The row table's own y and label_svi_delta hold the sealed placeholders (-1, NaN) at TEST rows.  Identity
columns (keys, roles, folds, query indices) of every row are compared for the alignment, as in Task 10.  Task 12 makes
the TEST labels, straight into the sealed file.  No player identifier is read or written.

Outputs, written once under D:/LOL_Training_20261004/tog_e4_1: E4_LABELS_nonTEST.npz (row keys, roles, contexts,
queries, y4, label_svi_delta4, exact_zero, near_zero_bits, p_pre, p_post) and E4_QDEV_AGREEMENT.json.  A BLOCKED condition
writes E4_LABELS_BLOCKED_<stage>_<n>.json before any label file and exits with 2.

Usage: python -B -m studies.tog_experiments_20261012.e4_run labels [--out DIR]   (inputs always from tog_e4_1)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005.assemble import NEAR_ZERO, PROJECTION  # noqa: E402
from studies.q_production_20261005 import rows as q_rows_module  # noqa: E402
from studies.q_production_20261005.qeval import match_weights  # noqa: E402
from studies.q_production_20261005.rows import Q_ROLES  # noqa: E402
from studies.tog_experiments_20261012 import e4_vscore as vs  # noqa: E402
from studies.tog_experiments_20261012.common import (  # noqa: E402
    LABELS, Q_RUN, document_record, git_head, peak_memory_bytes, sha256_file, utc, write_once)

VERSION = "tog_e4_labels_v1_20261013"
TOG_E4 = vs.TOG_E4
PRESPEC = vs.PRESPEC
CONTEXTS = vs.CONTEXTS                                  # final, oof0..oof4: context code = index
NON_TEST = {name: Q_ROLES[name] for name in ("TRAIN", "Q_DEV", "Q_CAL")}   # the rows Task 11 labels (Ruling 44)
BIN_EDGES = (0.01, 0.05)                                # |main SVI| bins (PRESPEC_E4 section 3)
BINS = ("abs_svi_lt_0.01", "abs_svi_0.01_to_0.05", "abs_svi_ge_0.05")
WEIGHTINGS = ("match_weighted", "row_weighted")         # primary first
TINY_DELTA = 1e-12                                      # Task 10 review (Opus) m2: 0 < |p_post - p_pre| < 1e-12
LABELS_NAME = "E4_LABELS_nonTEST.npz"
AGREEMENT_NAME = "E4_QDEV_AGREEMENT.json"
Q_ROWS = Q_RUN / "Q_ROWS_h90.npz"
ROW_COLUMNS = (*vs.ALIGN_COLUMNS, "engagement_key", "y", "label_svi_delta")   # y/label_svi_delta: TEST rows hold -1/NaN
ROW_INPUTS = (Q_ROWS, LABELS / "ENGAGEMENTS.npz", LABELS / "LABEL_LOCK.json", PROJECTION / "CONTEXT_MEMBERSHIP.npy")
CODE_FILES = ("studies/tog_experiments_20261012/e4_run.py", "studies/tog_experiments_20261012/e4_vscore.py",
              "studies/tog_experiments_20261012/common.py", "studies/label_production_20261005/assemble.py",
              "studies/q_production_20261005/rows.py", "studies/q_production_20261005/qeval.py",
              "studies/q_production_20261005/qtest.py", "studies/v_production_20261004/source.py")


# ----------------------------------------------------------------------------- labels
def _deltas(p_pre: Any, p_post: Any) -> np.ndarray:
    """p_post - p_pre in float64 for aligned 1-D arrays of finite probabilities inside [0, 1]; anything else raises
    (every E4 q row has both probabilities, so no row is invalid here)."""

    pre, post = np.asarray(p_pre, dtype=np.float64), np.asarray(p_post, dtype=np.float64)
    if pre.ndim != 1 or pre.shape != post.shape:
        raise ValueError("p_pre and p_post must be aligned 1-D arrays")
    for name, values in (("p_pre", pre), ("p_post", post)):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} must hold finite probabilities inside [0, 1]")
    return post - pre


def direction_labels(p_pre: Any, p_post: Any) -> np.ndarray:
    """y4 = 1[p_post - p_pre > 0] (int8); an exact zero is negative (assemble.py:123, no tolerance)."""

    return (_deltas(p_pre, p_post) > 0).astype(np.int8)


def zero_flags(p_pre: Any, p_post: Any) -> dict[str, np.ndarray]:
    """exact_zero (int8, p_post - p_pre == 0) and near_zero_bits (uint8, bit k when |p_post - p_pre| < NEAR_ZERO[k]),
    computed as the main labels' (assemble.py:124-127) on rows that are all valid."""

    delta = _deltas(p_pre, p_post)
    near = np.zeros(len(delta), dtype=np.uint8)
    for bit, threshold in enumerate(NEAR_ZERO):
        near[np.abs(delta) < threshold] |= 1 << bit
    return {"exact_zero": (delta == 0).astype(np.int8), "near_zero_bits": near}


def tiny_deltas(p_pre: Any, p_post: Any, bound: float = TINY_DELTA) -> dict[str, Any]:
    """Pairs with 0 < |p_post - p_pre| < bound (a last-digit difference of the probability path could flip only these)
    and the smallest nonzero |p_post - p_pre| (None when every pair ties)."""

    size = np.abs(_deltas(p_pre, p_post))
    nonzero = size[size > 0]
    return {"bound": bound, "rows": int(len(size)), "count": int(((size > 0) & (size < bound)).sum()),
            "min_nonzero_abs_delta": float(nonzero.min()) if len(nonzero) else None}


# ----------------------------------------------------------------------------- agreement
def svi_bins(svi_main: Any) -> np.ndarray:
    """Bin code per row by |main SVI|: 0 below 0.01, 1 in [0.01, 0.05), 2 from 0.05 on (PRESPEC_E4 section 3)."""

    size = np.abs(np.asarray(svi_main, dtype=np.float64))
    if not np.isfinite(size).all():
        raise ValueError("main SVI must be finite")
    return ((size >= BIN_EDGES[0]).astype(np.int8) + (size >= BIN_EDGES[1]).astype(np.int8)).astype(np.int8)


def _binary(values: Any, name: str, n: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (n is not None and array.shape != (n,)):
        raise ValueError(f"{name} must be a 1-D array aligned with the rows")
    if array.dtype.kind not in "biu" or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must hold 0/1 labels (the sealed TEST placeholder -1 is not a label)")
    return array.astype(np.int8)


def _share(values: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    return float((weights * values).sum() / total) if total > 0 else None


def _kappa(y_main: np.ndarray, y_new: np.ndarray, weights: np.ndarray) -> float | None:
    """Cohen's kappa (p_o - p_e) / (1 - p_e) of two binary labelings under row weights; None when p_e = 1."""

    total = float(weights.sum())
    if not total > 0:
        return None
    observed = float((weights * (y_main == y_new)).sum() / total)
    main, new = float((weights * y_main).sum() / total), float((weights * y_new).sum() / total)
    expected = main * new + (1.0 - main) * (1.0 - new)
    return None if expected >= 1.0 else (observed - expected) / (1.0 - expected)


def _entry(y_main: np.ndarray, y_new: np.ndarray, svi: np.ndarray, ids: np.ndarray | None,
           exact: np.ndarray | None) -> dict[str, Any]:
    n = len(y_main)
    weights = {"match_weighted": match_weights(ids) if ids is not None and n else None,
               "row_weighted": np.ones(n) if n else None}

    def both(value: Callable[[np.ndarray], float | None]) -> dict[str, float | None]:
        return {name: None if weights[name] is None else value(weights[name]) for name in WEIGHTINGS}

    equal = (y_main == y_new).astype(np.float64)
    main_zero = (svi == 0).astype(np.float64)
    return {"rows": int(n), "matches": None if ids is None else int(len(np.unique(ids))),
            "share_equal": both(lambda w: _share(equal, w)),
            "kappa": both(lambda w: _kappa(y_main, y_new, w)),
            "exact_zero_share_new": None if exact is None else both(lambda w: _share(exact, w)),
            "exact_zero_share_main": both(lambda w: _share(main_zero, w)),
            "positive_rate_main": both(lambda w: _share(y_main, w)),
            "positive_rate_new": both(lambda w: _share(y_new, w)),
            "confusion_rows": {"main1_new1": int(((y_main == 1) & (y_new == 1)).sum()),
                               "main1_new0": int(((y_main == 1) & (y_new == 0)).sum()),
                               "main0_new1": int(((y_main == 0) & (y_new == 1)).sum()),
                               "main0_new0": int(((y_main == 0) & (y_new == 0)).sum())}}


def agreement(y_main: Any, y_new: Any, svi_main: Any, match_index: Any = None, *,
              exact_zero_new: Any = None) -> dict[str, Any]:
    """Agreement of the main label y and the E4 label y4 on the same rows (PRESPEC_E4 section 3): overall and by
    |main SVI| bin, the share equal, Cohen's kappa, the exact-zero shares (E4 ties when ``exact_zero_new`` is given; main
    SVI == 0) and both positive rates, each match weighted (primary; None without ``match_index``) and row weighted.
    Every match present in a part weighs 1 there (weights 1/m recomputed inside each bin, qeval.metrics_by_stratum).
    Exact-zero rows stay in every denominator.  Labels must be 0/1 (a sealed TEST placeholder raises)."""

    y_main = _binary(y_main, "y_main")
    n = len(y_main)
    y_new = _binary(y_new, "y_new", n)
    svi = np.asarray(svi_main, dtype=np.float64)
    if svi.shape != (n,) or not np.isfinite(svi).all():
        raise ValueError("svi_main must be finite and aligned with the labels")
    ids = None if match_index is None else np.asarray(match_index)
    if ids is not None and ids.shape != (n,):
        raise ValueError("match_index must be aligned with the labels")
    exact = None if exact_zero_new is None else _binary(exact_zero_new, "exact_zero_new", n)
    if exact is not None and np.any((exact == 1) & (y_new == 1)):
        raise ValueError("an exact-zero E4 row must be negative")
    codes = svi_bins(svi)

    def part(rows: np.ndarray) -> dict[str, Any]:
        return _entry(y_main[rows], y_new[rows], svi[rows], None if ids is None else ids[rows],
                      None if exact is None else exact[rows])

    return {"overall": part(np.ones(n, dtype=bool)), "by_abs_main_svi": {name: part(codes == k) for k, name in enumerate(BINS)},
            "bin_edges": list(BIN_EDGES), "primary": "match_weighted",
            "weights": "match_weighted: 1/m per row, m = the match's rows in that part (every match present weighs 1); "
                       "row_weighted: 1 per row",
            "kappa": "Cohen: (p_o - p_e) / (1 - p_e), p_e = p_main p_new + (1 - p_main)(1 - p_new) under the same weights",
            "exact_zero_rows": "kept in every denominator; an exact zero is negative"}


# ----------------------------------------------------------------------------- rows
def _check_probabilities(values: np.ndarray, what: str) -> None:
    if values.dtype != np.float64 or values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{what}: probabilities must be finite float64 values inside [0, 1]")


def probabilities_at(pred: Mapping[str, np.ndarray], queries: Any, what: str) -> np.ndarray:
    """raw_p of one PRED_<ctx>.npz table (sorted query_index) at the given query indices.  Only those positions are read
    (PRED_final.npz also holds TEST rows' probabilities, outside any seal).  A query the table lacks, or a value read
    that is not a finite float64 inside [0, 1], raises."""

    keys = np.asarray(pred["query_index"], dtype=np.int64)
    queries = np.asarray(queries, dtype=np.int64)
    if keys.ndim != 1 or not len(keys):
        if len(queries):
            raise ValueError(f"{what}: the prediction table is empty")
        return np.zeros(0, dtype=np.float64)
    position = np.searchsorted(keys, queries)
    found = position < len(keys)
    found[found] = keys[position[found]] == queries[found]
    if not found.all():
        raise ValueError(f"{what}: {int((~found).sum())} queries are not in the prediction table")
    values = np.asarray(pred["raw_p"])[position]
    _check_probabilities(values, what)
    return values


def nontest_labels(rows: Mapping[str, np.ndarray], query_post: Any, preds: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """E4 labels of the row table's TRAIN, Q_DEV and Q_CAL rows, in row-table order; TEST rows are dropped before any
    probability is read (Ruling 44).  ``rows``: the row table's role, fold, query_pre, match_index and engagement_key;
    ``query_post``: every row's h90 endpoint query; ``preds``: {context: {query_index, raw_p}} (PRED_<ctx>.npz).
    Scoring context as Task 10 (e4_vscore.row_contexts): TRAIN rows oof<fold>, Q_DEV/Q_CAL rows final."""

    role = np.asarray(rows["role"]).astype(np.int64)
    if np.asarray(query_post).shape != role.shape:
        raise ValueError("query_post must give every row's endpoint query")
    q_row = np.flatnonzero(np.isin(role, list(NON_TEST.values())))
    codes = vs.row_contexts(role[q_row], np.asarray(rows["fold"])[q_row])
    query_pre = np.asarray(rows["query_pre"], dtype=np.int64)[q_row]
    query_end = np.asarray(query_post, dtype=np.int64)[q_row]
    p_pre, p_post = np.full(len(q_row), np.nan), np.full(len(q_row), np.nan)
    for code, ctx in enumerate(CONTEXTS):
        own = codes == code
        if not own.any():
            continue
        if ctx not in preds:
            raise ValueError(f"{ctx}: no prediction table for {int(own.sum())} rows")
        p_pre[own] = probabilities_at(preds[ctx], query_pre[own], f"{ctx} p_pre")
        p_post[own] = probabilities_at(preds[ctx], query_end[own], f"{ctx} p_post")
    flags = zero_flags(p_pre, p_post)                    # also refuses a row left without both probabilities
    return {"q_row": q_row.astype(np.int64), "engagement_key": np.asarray(rows["engagement_key"])[q_row],
            "match_index": np.asarray(rows["match_index"], dtype=np.int64)[q_row], "role": role[q_row].astype(np.int8),
            "fold": np.asarray(rows["fold"])[q_row].astype(np.int8), "context": codes.astype(np.int8),
            "query_pre": query_pre, "query_post": query_end, "y4": direction_labels(p_pre, p_post),
            "label_svi_delta4": p_post - p_pre, **flags, "p_pre": p_pre, "p_post": p_post}


def q_row_positions(labels: Mapping[str, np.ndarray], q_rows: Mapping[str, np.ndarray]) -> np.ndarray:
    """Positions in the main label table (LABELS_h90.npz, engagement order) of the row table's rows, in order: the q
    roles' valid rows (rows.build_rows, rows.py:115-117).  The identity columns and the engagement keys must agree one
    to one, else ValueError (a row-table key mismatch, PRESPEC_E4 section 7)."""

    role = np.asarray(labels["role"]).astype(np.int64)
    keep = np.flatnonzero(np.isin(role, list(Q_ROLES.values())) & np.asarray(labels["valid"], dtype=bool))
    for name in vs.ALIGN_COLUMNS:
        left, right = np.asarray(labels[name])[keep], np.asarray(q_rows[name])
        if left.dtype.kind in "iu":
            left, right = left.astype(np.int64), right.astype(np.int64)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"the row table differs from the main label table in {name}")
    key = np.char.add(np.char.add(np.asarray(labels["canonical_match_key"])[keep], b":"),
                      np.asarray(labels["component_key"])[keep])
    engagement_key = np.asarray(q_rows["engagement_key"])
    if key.shape != engagement_key.shape or not np.array_equal(key, engagement_key):
        raise ValueError("the row table's engagement keys differ from the main label table's")
    return keep


def main_labels(labels: Mapping[str, np.ndarray], q_rows: Mapping[str, np.ndarray], q_row: Any,
                position: Any = None) -> dict[str, np.ndarray]:
    """Main labels of the given row-table rows (TRAIN, Q_DEV and Q_CAL only; TEST labels are sealed): y =
    direction_positive and svi = svi_delta of LABELS_h90.npz at the rows' positions (``position``, else
    q_row_positions).  Where the row table carries its public copies (y, label_svi_delta), they must be equal."""

    q_row = np.asarray(q_row, dtype=np.int64)
    if not np.isin(np.asarray(q_rows["role"])[q_row], list(NON_TEST.values())).all():
        raise ValueError("main labels are read only for TRAIN, Q_DEV and Q_CAL rows (TEST labels are sealed)")
    where = (q_row_positions(labels, q_rows) if position is None else np.asarray(position, dtype=np.int64))[q_row]
    y = np.asarray(labels["direction_positive"])[where].astype(np.int8)
    svi = np.asarray(labels["svi_delta"], dtype=np.float64)[where]
    if not np.isin(y, (0, 1)).all() or not np.isfinite(svi).all():
        raise ValueError("a valid q row lacks its main label")
    if "y" in q_rows and not np.array_equal(np.asarray(q_rows["y"])[q_row].astype(np.int8), y):
        raise ValueError("the row table's y differs from LABELS_h90.npz direction_positive")
    if "label_svi_delta" in q_rows and not np.array_equal(np.asarray(q_rows["label_svi_delta"], dtype=np.float64)[q_row], svi):
        raise ValueError("the row table's label_svi_delta differs from LABELS_h90.npz svi_delta")
    return {"y": y, "svi": svi}


# ----------------------------------------------------------------------------- run
def _blocked(out: Path, which: str, reasons: Sequence[str], **extra: Any) -> int:
    """Write E4_LABELS_BLOCKED_<which>_<n>.json once (n = the first free number) and return 2."""

    out.mkdir(parents=True, exist_ok=True)
    k = 1
    while (out / f"E4_LABELS_BLOCKED_{which}_{k}.json").exists():
        k += 1
    path = out / f"E4_LABELS_BLOCKED_{which}_{k}.json"
    write_once(path, vs._json_safe({"version": VERSION, "experiment": "E4", "blocked_stage": which, "blocked": True,
                                    "utc": utc(), "reasons": list(reasons), **extra}))
    print(json.dumps({"BLOCKED": which, "file": str(path), "reasons": list(reasons)}, ensure_ascii=False), flush=True)
    return 2


def check_task10(pred_dir: Path, hashes: dict[str, str], row_inputs: Sequence[Path] = ROW_INPUTS,
                 role_rows: Mapping[str, int] = vs.Q_ROLE_ROWS) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    """Task 10's report and per-context records, and every disagreement: both of its steps passed (V_DEV in six
    contexts, OOF), it is not a smoke run, its role counts are PRESPEC_E4's, each PRED_<ctx>.npz and PRED_<ctx>.json
    equal the hashes it recorded, and the row-table inputs it scored (``row_inputs``) are unchanged."""

    problems: list[str] = []
    report = vs._read(pred_dir / "E4_VSCORE.json")
    if report.get("v_dev_passed") is not True:
        problems.append("E4_VSCORE.json: the V_DEV reproduction did not pass")
    if (report.get("oof_check") or {}).get("passed") is not True:
        problems.append("E4_VSCORE.json: the OOF exclusion check did not pass")
    if report.get("smoke"):
        problems.append("E4_VSCORE.json is a smoke run")
    if report.get("q_rows_by_role") != dict(role_rows):
        problems.append(f"E4_VSCORE.json q rows by role {report.get('q_rows_by_role')} are not PRESPEC_E4's {dict(role_rows)}")
    row_inputs = vs._hashes(row_inputs, hashes)
    records: dict[str, dict[str, Any]] = {}
    for ctx in CONTEXTS:
        entry = (report.get("contexts") or {}).get(ctx) or {}
        npz, record_path = pred_dir / f"PRED_{ctx}.npz", pred_dir / f"PRED_{ctx}.json"
        files = vs._hashes([npz, record_path], hashes)
        record = vs._read(record_path)
        if not (files[npz.as_posix()] == entry.get("pred_sha256") == record.get("pred_sha256")):
            problems.append(f"{ctx}: PRED_{ctx}.npz differs from E4_VSCORE.json / PRED_{ctx}.json")
        if files[record_path.as_posix()] != entry.get("record_sha256"):
            problems.append(f"{ctx}: PRED_{ctx}.json differs from E4_VSCORE.json")
        if record.get("context") != ctx or record.get("columns") != ["query_index", "raw_p"]:
            problems.append(f"{ctx}: PRED_{ctx}.json names context {record.get('context')!r}, columns {record.get('columns')}")
        for path, digest in row_inputs.items():
            if (record.get("inputs") or {}).get(path) != digest:
                problems.append(f"{ctx}: {path} differs from the file Task 10 scored")
        records[ctx] = record
    return report, records, problems


def _role_summary(table: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Counts of the E4 labels by non-TEST role (rows, matches, positives, exact zeros, near-zero bits, tiny deltas)."""

    out = {}
    for name, code in NON_TEST.items():
        own = table["role"] == code
        ids, y, exact = table["match_index"][own], table["y4"][own], table["exact_zero"][own]
        w = match_weights(ids) if own.any() else None
        out[name] = {"rows": int(own.sum()), "matches": int(len(np.unique(ids))), "positive": int(y.sum()),
                     "positive_rate": {"match_weighted": _share(y, w) if w is not None else None,
                                       "row_weighted": float(y.mean()) if own.any() else None},
                     "exact_zero": int(exact.sum()),
                     "exact_zero_share": {"match_weighted": _share(exact, w) if w is not None else None,
                                          "row_weighted": float(exact.mean()) if own.any() else None},
                     "near_zero_rows": {f"abs_delta_lt_{t}": int(((table["near_zero_bits"][own] & (1 << k)) != 0).sum())
                                        for k, t in enumerate(NEAR_ZERO)},
                     "tiny_delta": tiny_deltas(table["p_pre"][own], table["p_post"][own]),
                     "contexts": {CONTEXTS[c]: {"rows": int((table["context"][own] == c).sum()),
                                                "exact_zero": int(exact[table["context"][own] == c].sum())}
                                  for c in sorted(set(table["context"][own].tolist()))}}
    return out


def run(out: Path = TOG_E4, *, pred_dir: Path = TOG_E4, labels_dir: Path = LABELS, q_rows_path: Path = Q_ROWS,
        projection: Path = PROJECTION, role_rows: Mapping[str, int] = vs.Q_ROLE_ROWS) -> int:
    """PRESPEC_E4 section 6 steps 2-3 for TRAIN, Q_DEV and Q_CAL: the E4 labels and the Q_DEV agreement.  The keyword
    arguments name the inputs (defaults: tog_e4_1, labels_1, q_run_1/Q_ROWS_h90.npz, the projection) and the role counts
    (PRESPEC_E4 section 2); only a synthetic test passes others."""

    out, pred_dir, labels_dir = Path(out), Path(pred_dir), Path(labels_dir)
    q_rows_path, projection = Path(q_rows_path), Path(projection)
    main_path = labels_dir / "LABELS_h90.npz"
    row_inputs = (q_rows_path, labels_dir / "ENGAGEMENTS.npz", labels_dir / "LABEL_LOCK.json", projection / "CONTEXT_MEMBERSHIP.npy")
    label_inputs = (main_path, labels_dir / "LABELS_MANIFEST.json", labels_dir / "Q_DEV_H90_KEYS.npy")
    report_path = out / AGREEMENT_NAME
    if report_path.exists():
        print(json.dumps({"done": str(report_path)}), flush=True)
        return 0
    started, stages, hashes = utc(), [], {}

    # inputs: Task 10's records and tables, the label lock --------------------------------------------------------
    s, t = utc(), time.perf_counter()
    vscore, records, problems = check_task10(pred_dir, hashes, row_inputs, role_rows)
    try:
        lock = q_rows_module.verify_lock(labels_dir, 90)    # LABELS_h90 / ENGAGEMENTS / Q_DEV keys equal the label lock
    except (OSError, ValueError) as error:
        lock = {}
        problems.append(f"label lock: {error}")
    stages.append(vs._stage("inputs", s, t))
    if problems:
        return _blocked(out, "inputs", problems)

    # rows: the row table, its endpoint queries, its main labels; TEST rows dropped here ------------------------------
    s, t = utc(), time.perf_counter()
    with np.load(q_rows_path, allow_pickle=False) as stored:
        rows = {name: stored[name] for name in ROW_COLUMNS}
    with np.load(labels_dir / "ENGAGEMENTS.npz", allow_pickle=False) as stored:
        engagements = {name: stored[name] for name in (*vs.ALIGN_COLUMNS, "query_h90", "valid_h90")}
    try:
        keep = vs.align_q_rows(engagements, rows)            # Task 10's alignment and endpoint queries
    except ValueError as error:
        return _blocked(out, "rows", [f"row table key mismatch: {error}"])
    query_post = engagements["query_h90"][keep].astype(np.int64)
    del engagements
    role = rows["role"].astype(np.int64)
    by_role = {name: int((role == code).sum()) for name, code in Q_ROLES.items()}
    if by_role != dict(role_rows):
        problems.append(f"q rows by role {by_role} differ from PRESPEC_E4's {dict(role_rows)}")
    nontest = np.flatnonzero(np.isin(role, list(NON_TEST.values())))
    with np.load(main_path, allow_pickle=False) as stored:
        labels = {name: stored[name] for name in (*vs.ALIGN_COLUMNS, "valid", "query_post")}
        try:
            position = q_row_positions(labels, rows)
            main = main_labels({**labels, "direction_positive": stored["direction_positive"], "svi_delta": stored["svi_delta"]},
                               rows, nontest, position)
        except ValueError as error:
            return _blocked(out, "rows", [f"main labels: {error}"])
    query_post_equal = bool(np.array_equal(labels["query_post"][position[nontest]].astype(np.int64), query_post[nontest]))
    if not query_post_equal:
        problems.append("LABELS_h90.npz query_post differs from ENGAGEMENTS.npz query_h90 on TRAIN/Q_DEV/Q_CAL rows")
    del labels
    codes = vs.row_contexts(role[nontest], rows["fold"][nontest])
    membership = np.load(projection / "CONTEXT_MEMBERSHIP.npy", allow_pickle=False)
    leaked = vs.oof_leaks(codes, role[nontest], rows["match_index"][nontest], membership)
    del membership
    oof = {"rule": "assemble.py:109-114 (e4_vscore.oof_leaks), TRAIN/Q_DEV/Q_CAL rows (Task 10 checked all roles)",
           "rows_checked": int(len(nontest)), "leaked": int(leaked.sum()), "passed": bool(not leaked.any())}
    if leaked.any():
        problems.append(f"OOF exclusion failed on {int(leaked.sum())} rows")
    context_rows = {}
    for c, ctx in enumerate(CONTEXTS):
        mine = {name: int(((codes == c) & (role[nontest] == code)).sum()) for name, code in NON_TEST.items()}
        mine = {name: count for name, count in mine.items() if count}
        scored = {name: count for name, count in records[ctx]["rows"]["q_rows_by_role"].items() if name in NON_TEST}
        context_rows[ctx] = {"rows_by_role": mine, "task10_rows_by_role": scored, "equal": mine == scored}
        if mine != scored:
            problems.append(f"{ctx}: TRAIN/Q_DEV/Q_CAL rows {mine} differ from Task 10's {scored}")
    stages.append(vs._stage("rows", s, t))
    if problems:
        return _blocked(out, "rows", problems, oof_check=oof, context_rows=context_rows)

    # labels: TRAIN, Q_DEV, Q_CAL ------------------------------------------------------------------------------------
    s, t = utc(), time.perf_counter()
    preds = {}
    for ctx in CONTEXTS:
        with np.load(pred_dir / f"PRED_{ctx}.npz", allow_pickle=False) as stored:
            preds[ctx] = {"query_index": stored["query_index"], "raw_p": stored["raw_p"]}
    try:
        table = nontest_labels(rows, query_post, preds)
    except ValueError as error:
        return _blocked(out, "labels", [f"labels: {error}"])
    del preds
    if not np.array_equal(table["q_row"], nontest):
        return _blocked(out, "labels", ["the labelled rows are not the row table's TRAIN/Q_DEV/Q_CAL rows"])
    out.mkdir(parents=True, exist_ok=True)
    labels_sha = vs._write_npz_or_verify(out / LABELS_NAME, table)
    per_role = _role_summary(table)
    stages.append(vs._stage("labels", s, t))

    # Q_DEV agreement ---------------------------------------------------------------------------------------------------
    s, t = utc(), time.perf_counter()
    q_dev = table["role"] == Q_ROLES["Q_DEV"]
    result = agreement(main["y"][q_dev], table["y4"][q_dev], main["svi"][q_dev], table["match_index"][q_dev],
                       exact_zero_new=table["exact_zero"][q_dev])
    stages.append(vs._stage("agreement", s, t))

    inputs = {**vs._hashes([pred_dir / "E4_VSCORE.json"], hashes),
              **vs._hashes([pred_dir / f"PRED_{ctx}.{ext}" for ctx in CONTEXTS for ext in ("npz", "json")], hashes),
              **vs._hashes([*label_inputs, *row_inputs], hashes)}
    record = {
        "version": VERSION, "experiment": "E4",
        "step": "PRESPEC_E4 section 6 steps 2-3: E4 labels of TRAIN, Q_DEV and Q_CAL rows; Q_DEV agreement with the main labels",
        "definitions": {
            "probability": "raw_p of PRED_<ctx>.npz as stored (LightGBM V predict_proba, the quantity the V search scored; no "
                           "calibration, no logit round trip; Ruling 43); TRAIN rows: oof<fold>, Q_DEV/Q_CAL rows: final",
            "y4": "1[p_post - p_pre > 0]; an exact zero is negative (exact_zero = 1)",
            "label_svi_delta4": "p_post - p_pre (float64)",
            "near_zero_bits": f"bit k set when |p_post - p_pre| < {list(NEAR_ZERO)}[k] (assemble.py:40, :125-127)",
            "main_label": "labels_1/LABELS_h90.npz direction_positive and svi_delta, matched to q_run_1/Q_ROWS_h90.npz by "
                          "identity columns, engagement key and order (equal to the row table's y and label_svi_delta)",
            "agreement_bins": list(BINS), "tiny_delta_bound": TINY_DELTA},
        "labels": {"file": str(out / LABELS_NAME), "sha256": labels_sha, "columns": list(table), "rows": int(len(table["q_row"])),
                   "by_role": per_role,
                   "test_rows": "not labelled here (Ruling 44); Task 12 makes them straight into the sealed file"},
        "q_dev_agreement": result,
        "tiny_delta_by_role": {name: per_role[name]["tiny_delta"] for name in NON_TEST},
        "checks": {"task10": {"v_dev_passed": vscore.get("v_dev_passed"), "oof_passed": (vscore.get("oof_check") or {}).get("passed"),
                              "pred_and_record_hashes_equal": True, "row_inputs_unchanged": True},
                   "label_lock": {"verified": True, "locked_utc": lock.get("locked_utc")},
                   "row_alignment": {"q_rows": int(len(role)), "q_rows_by_role": by_role,
                                     "engagements": "e4_vscore.align_q_rows", "main_labels": "q_row_positions",
                                     "main_labels_equal_row_table": True, "query_post_equal": query_post_equal},
                   "oof_check": oof, "context_rows": context_rows},
        "inputs": inputs, "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "code": document_record(Path(__file__)), "prespec": document_record(PRESPEC), "git_head": git_head(),
        "versions": vs._versions(), "stages": stages, "peak_memory_bytes": peak_memory_bytes(),
        "notes": ["TEST rows were dropped from the row list before any probability or label was read; no TEST label, "
                  "agreement, rate or count beyond the public row count was computed (PRESPEC_E4 section 3, Ruling 44)",
                  "Q_DEV results are development evidence and decide nothing (PRESPEC_E4 section 6)",
                  "TEST agreement, kappa and exact-zero share come once, after the lock (Task 12)"],
        "run_started_utc": started, "utc": utc()}
    write_once(report_path, vs._json_safe(record))
    overall = result["overall"]
    print(json.dumps({"done": str(report_path), "labels_sha256": labels_sha,
                      "q_dev": {"rows": overall["rows"], "matches": overall["matches"], "share_equal": overall["share_equal"],
                                "kappa": overall["kappa"], "exact_zero_share_new": overall["exact_zero_share_new"]},
                      "tiny_delta": {name: per_role[name]["tiny_delta"]["count"] for name in NON_TEST},
                      "exact_zero": {name: per_role[name]["exact_zero"] for name in NON_TEST}}), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    labels = sub.add_parser("labels", help="E4 labels of TRAIN/Q_DEV/Q_CAL rows and the Q_DEV agreement (Task 11)")
    labels.add_argument("--out", default=str(TOG_E4))
    args = parser.parse_args(argv)
    return run(Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
