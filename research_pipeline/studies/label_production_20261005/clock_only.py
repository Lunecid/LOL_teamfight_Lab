"""Clock-only diagnostic sidecar (Q_DEV h90 keys): the pre observation with only the clock advanced.

PROTOCOL_KO.md §2 "시계만 변경": keep the pre state's non-time payload, move the
query time to the post endpoint, and score with the final V.  In the V input
only three blocks depend on the query time (studies/v_input_projection_20261002/
projection.py): ``elapsed_minutes = q/60000``, ``frame_age_seconds = (q -
frame_ms)/1000`` and ``objective_age_seconds = (q - last_ms)/1000`` (NaN when no
event yet).  The non-item history is a cumulative counter snapshot and the item
contexts are counts, so they stay as at the pre time.  Rows whose advanced frame
age exceeds the largest age in the V training rows are flagged as outside
support.  The difference is reported, never subtracted from the main SVI.

Usage: python -B studies/label_production_20261005/clock_only.py --run-root D:/LOL_Training_20261004/run_3 --labels D:/LOL_Training_20261004/labels_1
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import assemble, diagnostics, queries, score  # noqa: E402

VERSION = "label_clock_only_v1_20261005"
PROJECTION = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
TIME_FIELDS = ("elapsed_minutes", "frame_age_seconds", "objective_age_seconds")


def advance_clock(features: Mapping[str, Any], pre_ms: int, target_ms: int) -> dict[str, Any]:
    """Features of the pre observation with the query time moved to ``target_ms`` (projection formulas)."""

    out = dict(features)
    frame_ms = int(pre_ms) - int(np.rint(float(features["frame_age_seconds"]) * 1000.0))  # exact integer ms
    out["elapsed_minutes"] = np.float64(int(target_ms)) / 60000.0
    out["frame_age_seconds"] = np.float64(int(target_ms) - frame_ms) / 1000.0
    ages = np.asarray(features["objective_age_seconds"], dtype=np.float64)
    finite = np.isfinite(ages)
    last = int(pre_ms) - np.rint(np.where(finite, ages, 0.0) * 1000.0).astype(np.int64)
    advanced = np.full(ages.shape, np.nan, dtype=np.float64)
    advanced[finite] = (int(target_ms) - last[finite]).astype(np.float64) / 1000.0
    out["objective_age_seconds"] = advanced
    return out


def run(run_root: Path, labels: Path) -> dict[str, Any]:
    labels = Path(labels)
    keys = np.load(labels / "Q_DEV_H90_KEYS.npy", allow_pickle=False)
    h90 = dict(np.load(labels / "LABELS_h90.npz", allow_pickle=False))
    row_keys = np.char.add(np.char.add(h90["canonical_match_key"], b":"), h90["component_key"])
    rows = np.flatnonzero(np.isin(row_keys, keys) & h90["valid"])
    pre_q, post_ms = h90["query_pre"][rows].astype(np.int64), h90["post_ms"][rows].astype(np.int64)
    if len(np.unique(pre_q)) != len(pre_q):
        raise ValueError("two Q_DEV engagements share a pre query; clock targets would collide")
    order = np.argsort(pre_q, kind="stable")
    target_of = dict(zip(pre_q.tolist(), post_ms.tolist()))
    v = score.LockedV(run_root, "final")
    surface, bundle = queries.open_inputs("final")
    p_clock = np.full(len(rows), np.nan)
    hook = lambda query_index, query_ms, features: advance_clock(features, query_ms, target_of[int(query_index)])  # noqa: E731
    try:
        for start in range(0, len(order), queries.BATCH_ROWS):
            pick = order[start:start + queries.BATCH_ROWS]
            built = list(surface.iter_query_rows(pre_q[pick], features_hook=hook))
            if [q for q, _, _ in built] != pre_q[pick].tolist():
                raise ValueError("clock rows came back out of order")
            _logit, _raw, calibrated = v.score(queries.transform([row for _, row, _ in built], bundle))
            p_clock[pick] = calibrated
    finally:
        surface.close()
    ages = np.load(PROJECTION / "FRAME_AGE_SECONDS.npy", mmap_mode="r", allow_pickle=False)
    nominal = np.load(PROJECTION / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    matches = np.load(PROJECTION / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    first = np.load(PROJECTION / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)["first_query"]
    train = nominal[(np.asarray(matches["role"])[np.asarray(nominal["match_index"])] == 0) & np.asarray(nominal["supported"])]
    max_train_age = float(np.max(ages[np.asarray(first)[train["match_index"]] + train["query_ref"]]))
    advanced_age = (post_ms - (h90["pre_ms"][rows] - h90["pre_frame_age_ms"][rows])) / 1000.0
    outside = advanced_age > max_train_age
    clock_delta = p_clock - h90["p_pre"][rows]
    actual = h90["svi_delta"][rows]
    w = diagnostics._weights(h90["match_index"][rows])
    w = w / w.sum()
    report = {"version": VERSION, "utc": datetime.now(timezone.utc).isoformat(), "rows": int(len(rows)), "v": v.identity,
              "time_fields_advanced": list(TIME_FIELDS), "max_frame_age_in_V_training_s": max_train_age,
              "outside_training_support": int(outside.sum()), "clock_delta": diagnostics.summary(clock_delta, h90["match_index"][rows]),
              "direction_agreement_clock_vs_actual_match_equal": float(np.sum(w * ((clock_delta > 0) == (actual > 0)))),
              "mean_abs_actual_minus_clock_match_equal": float(np.sum(w * np.abs(actual - clock_delta))),
              "note": "measurement diagnostic: not subtracted from the main SVI, not a causal effect"}
    table = {"canonical_match_key": h90["canonical_match_key"][rows], "component_key": h90["component_key"][rows],
             "p_pre": h90["p_pre"][rows], "p_clock_only": p_clock, "p_post": h90["p_post"][rows],
             "advanced_frame_age_s": advanced_age, "outside_training_support": outside}
    report["table_sha256"] = assemble.write_npz(labels / "CLOCK_ONLY_h90.npz", table)
    (labels / "CLOCK_ONLY_h90.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    report = run(Path(args.run_root), Path(args.labels))
    print(json.dumps({k: report[k] for k in ("rows", "outside_training_support", "direction_agreement_clock_vs_actual_match_equal")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
