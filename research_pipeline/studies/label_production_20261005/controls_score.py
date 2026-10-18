"""Non-event controls, phase 3: final-V scores at control starts and ends, and the engagement-control comparison.

Protocol: reports/research_execution_20261003/PROTOCOL_KO.md line 32 (both engagement and control use the same
final V and calibrator; each pair's Delta difference is computed first, averaged within the match, then
aggregated with equal match weight; matched and unmatched rates use every eligible engagement as denominator;
a descriptive diagnostic, not a causal estimate) and LABEL_PRODUCER_DESIGN_KO.md section 6.

score    the final V_LOCK (``score.LockedV``) scores
           - the distinct control starts: stored nominal queries, read with ``QuerySurface`` as every label
             query is, in canonical 1,024-row batches of the sorted start queries;
           - the control ends: ``controls_states.iter_rows`` of CONTROL_END_ROWS_h90.npz, in end order, in
             1,024-row batches;
         -> CONTROL_V_h90.npz + CONTROL_V_h90.json (V identity, code and input hashes).
compare  needs LABELS_h90.npz.  Per matched Q_DEV h90 engagement j: W_j = V(end_j) - V(start_j) and the label's
         Delta_j = V(post_j) - V(pre_j) (same final V, same calibrator); d_j = Delta_j - W_j and
         a_j = |Delta_j| - |W_j|; per-match means, then the equal-weight mean over matches with a match-level
         bootstrap (2000, seed 20261011, 95 %); game-time strata of the engagement and a sensitivity without
         ends outside the V training frame-age support.
         -> CONTROLS_h90.npz (one row per eligible engagement, aligned with CONTROLS_MATCHING_h90.npz) +
            CONTROLS_h90.json (the optional files bound by lock.py).

CLI: python -B studies/label_production_20261005/controls_score.py score --labels D:/LOL_Training_20261004/labels_1
         --run-root D:/LOL_Training_20261004/run_3 [--end-rows <dir>/CONTROL_END_ROWS_h90.npz] [--device cuda]
     python -B studies/label_production_20261005/controls_score.py compare --labels D:/LOL_Training_20261004/labels_1
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
from typing import Any, Mapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from studies.label_production_20261005 import assemble, queries, score  # noqa: E402

VERSION = "label_controls_score_v1_20261005"
END_ROWS_DEFAULT = "CONTROL_END_ROWS_h90_1/CONTROL_END_ROWS_h90.npz"
GAME_TIME_BINS_MIN = (15, 25)
BOOTSTRAP, SEED, LEVEL = 2000, 20261011, 0.95


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    files = {**score.code_hashes()}
    for name in ("controls.py", "controls_states.py", "controls_score.py"):
        files[(here / name).relative_to(ROOT).as_posix()] = score.sha256_file(here / name)
    return files


# ----------------------------------------------------------------------------- score
def _batches(n: int, size: int = queries.BATCH_ROWS) -> list[tuple[int, int]]:
    return [(start, min(start + size, n)) for start in range(0, n, size)]


def score_controls(labels: Path, run_root: Path, end_rows: Path, device: str = "cuda") -> dict[str, Any]:
    from studies.label_production_20261005 import controls_states

    labels, run_root, end_rows = Path(labels), Path(run_root), Path(end_rows)
    target = labels / "CONTROL_V_h90.npz"
    if target.exists():
        raise FileExistsError(f"{target} exists; control scores are written once")
    matching = load_npz(labels / "CONTROLS_MATCHING_h90.npz")
    start_q = np.asarray(matching["control_start_query"], dtype=np.int64)
    starts = np.unique(start_q[start_q >= 0])
    v = score.LockedV(run_root, "final", device=device)
    surface, bundle = queries.open_inputs("final")
    out: dict[str, list[np.ndarray]] = {name: [] for name in ("start_logit", "start_raw", "start_v", "end_logit", "end_raw", "end_v")}
    try:
        for a, b in _batches(len(starts)):
            batch = starts[a:b]
            rows, seen = [], []
            for query_index, row, _meta in surface.iter_query_rows(batch):
                rows.append(row)
                seen.append(query_index)
            if not np.array_equal(np.asarray(seen, dtype=np.int64), batch):
                raise ValueError("control start rows came back out of order")
            for name, value in zip(("start_logit", "start_raw", "start_v"), v.score(queries.transform(rows, bundle))):
                out[name].append(value)
    finally:
        surface.close()
    meta_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        if rows:
            for name, value in zip(("end_logit", "end_raw", "end_v"), v.score(queries.transform(rows, bundle))):
                out[name].append(value)
            rows.clear()

    for expected, (end_index, row, meta) in enumerate(controls_states.iter_rows(end_rows)):
        if end_index != expected:
            raise ValueError("control end rows are not in end order")
        rows.append(row)
        meta_rows.append(meta)
        if len(rows) == queries.BATCH_ROWS:
            flush()
    flush()
    payload = {"start_query": starts, **{name: np.concatenate(values) if values else np.zeros(0) for name, values in out.items()},
               "end_index": np.asarray([m["end_index"] for m in meta_rows], dtype=np.int64),
               "end_match_index": np.asarray([m["match_index"] for m in meta_rows], dtype=np.int64),
               "end_ms": np.asarray([m["end_ms"] for m in meta_rows], dtype=np.int64),
               "end_after_last_stored_query": np.asarray([m["after_last_stored_query"] for m in meta_rows], dtype=bool),
               "end_outside_training_frame_age_support": np.asarray([m["outside_training_frame_age_support"] for m in meta_rows], dtype=bool)}
    for name in ("start_v", "end_v"):
        if not np.isfinite(payload[name]).all():
            raise ValueError(f"non-finite {name}")
    record = {"version": VERSION, "utc": utc(), "v": v.identity, "code_sha256": code_hashes(), "batch_rows": queries.BATCH_ROWS,
              "starts": int(len(starts)), "ends": int(len(meta_rows)),
              "inputs": {"CONTROLS_MATCHING_h90.npz": score.sha256_file(labels / "CONTROLS_MATCHING_h90.npz"), "end_rows": str(end_rows),
                         "end_rows_sha256": score.sha256_file(end_rows)}}
    record["sha256"] = assemble.write_npz(target, payload)
    (labels / "CONTROL_V_h90.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return record


# ----------------------------------------------------------------------------- compare
def _game_bin(pre_ms: np.ndarray) -> np.ndarray:
    return np.asarray(("lt15", "15_25", "ge25"))[np.digitize(np.asarray(pre_ms) / 60000.0, GAME_TIME_BINS_MIN)]


def match_means(match: np.ndarray, values: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(match ids, per-match mean of every column)."""

    ids, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
    return ids, {name: np.bincount(inverse, weights=np.asarray(v, dtype=np.float64), minlength=len(ids)) / counts for name, v in values.items()}


def aggregate(match: np.ndarray, values: Mapping[str, np.ndarray], *, resamples: int = BOOTSTRAP, seed: int = SEED) -> dict[str, Any]:
    """Equal-match-weight means of per-match means, with match-bootstrap percentile intervals."""

    if not len(match):
        return {"rows": 0, "matches": 0}
    ids, means = match_means(match, values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(resamples, len(ids)))
    lo, hi = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2
    out: dict[str, Any] = {"rows": int(len(match)), "matches": int(len(ids))}
    for name, per_match in means.items():
        boot = per_match[draws].mean(axis=1)
        out[name] = {"mean": float(per_match.mean()), "ci95": [float(np.quantile(boot, lo)), float(np.quantile(boot, hi))]}
    return out


def compare(labels: Path) -> dict[str, Any]:
    labels = Path(labels)
    matching = load_npz(labels / "CONTROLS_MATCHING_h90.npz")
    matching_record = json.loads((labels / "CONTROLS_MATCHING_h90.json").read_text(encoding="utf-8"))
    control_v = load_npz(labels / "CONTROL_V_h90.npz")
    control_record = json.loads((labels / "CONTROL_V_h90.json").read_text(encoding="utf-8"))
    end_rows = Path(control_record["inputs"]["end_rows"])
    end_table = load_npz(end_rows.parent / "END_TABLE.npz")
    table = load_npz(labels / "LABELS_h90.npz")
    n = len(matching["control_start_query"])
    label_keys = np.char.add(np.char.add(table["canonical_match_key"], b":"), table["component_key"])
    keys = np.char.add(np.char.add(matching["canonical_match_key"], b":"), matching["component_key"])
    order = np.argsort(label_keys, kind="stable")
    pos = np.searchsorted(label_keys[order], keys)
    pos = np.clip(pos, 0, len(order) - 1)
    found = label_keys[order][pos] == keys
    if not found.all():
        raise ValueError("an eligible Q_DEV engagement has no label row")
    label_row = order[pos]
    label_valid = np.asarray(table["valid"], dtype=bool)[label_row]
    start_q = np.asarray(matching["control_start_query"], dtype=np.int64)
    has_control = start_q >= 0
    engagement_end = np.asarray(end_table["engagement_end_index"], dtype=np.int64)
    if len(engagement_end) != n or not np.array_equal(engagement_end >= 0, has_control):
        raise ValueError("END_TABLE does not align with CONTROLS_MATCHING_h90.npz")
    start_pos = np.searchsorted(control_v["start_query"], start_q[has_control])
    if not np.array_equal(control_v["start_query"][start_pos], start_q[has_control]):
        raise ValueError("a control start was not scored")
    if not np.array_equal(control_v["end_index"], np.arange(len(control_v["end_index"]))):
        raise ValueError("control end scores are not in end order")
    delta = np.where(label_valid, np.asarray(table["svi_delta"], dtype=np.float64)[label_row], np.nan)
    w = np.full(n, np.nan)
    v_start, v_end = np.full(n, np.nan), np.full(n, np.nan)
    v_start[has_control] = control_v["start_v"][start_pos]
    v_end[has_control] = control_v["end_v"][engagement_end[has_control]]
    w[has_control] = v_end[has_control] - v_start[has_control]
    outside = np.zeros(n, dtype=bool)
    outside[has_control] = control_v["end_outside_training_frame_age_support"][engagement_end[has_control]]
    matched = has_control & label_valid                     # a pair needs both a control and a valid label
    match = np.asarray(table["match_index"], dtype=np.int64)[label_row]
    game_bin = _game_bin(np.asarray(table["pre_ms"], dtype=np.int64)[label_row])
    out_table = {"canonical_match_key": matching["canonical_match_key"], "component_key": matching["component_key"], "match_index": match,
                 "matched": matched, "control_start_query": start_q, "control_end_index": engagement_end, "v_start": v_start, "v_end": v_end,
                 "w_control": w, "delta_engagement": delta, "d_signed": delta - w, "d_abs": np.abs(delta) - np.abs(w),
                 "end_outside_training_frame_age_support": outside, "game_bin": game_bin.astype("S5")}

    def block(rows: np.ndarray) -> dict[str, Any]:
        values = {"delta_engagement": delta[rows], "w_control": w[rows], "d_signed": delta[rows] - w[rows], "d_abs": np.abs(delta[rows]) - np.abs(w[rows]),
                  "abs_delta_engagement": np.abs(delta[rows]), "abs_w_control": np.abs(w[rows]),
                  "delta_positive": (delta[rows] > 0).astype(float), "w_positive": (w[rows] > 0).astype(float)}
        return aggregate(match[rows], values)

    rows = np.flatnonzero(matched)
    report: dict[str, Any] = {
        "version": VERSION, "utc": utc(), "rule": "PROTOCOL_KO.md line 32: same final V and calibrator; per-pair difference, per-match mean, equal match weight",
        "eligible_q_dev_h90_engagements": int(n), "matched": int(matched.sum()), "unmatched": int((~has_control).sum()),
        "matched_rate": float(matched.mean()) if n else None, "unmatched_reasons": matching_record.get("unmatched_reasons"),
        "control_but_invalid_label": int((has_control & ~label_valid).sum()), "invalid_label_rows": int((~label_valid).sum()),
        "distinct_control_starts": int(len(control_v["start_query"])), "distinct_control_ends": int(len(control_v["end_index"])),
        "ends_outside_training_frame_age_support": int(control_v["end_outside_training_frame_age_support"].sum()),
        "ends_after_last_stored_query": int(control_v["end_after_last_stored_query"].sum()),
        "all_matched": block(rows),
        "excluding_ends_outside_support": block(rows[~outside[rows]]),
        "by_game_time": {name: block(rows[game_bin[rows] == name]) for name in ("lt15", "15_25", "ge25")},
        "unmatched_engagements": aggregate(match[~has_control & label_valid],
                                           {"delta_engagement": delta[~has_control & label_valid], "abs_delta_engagement": np.abs(delta[~has_control & label_valid]),
                                            "delta_positive": (delta[~has_control & label_valid] > 0).astype(float)}),
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "level": LEVEL, "unit": "match"},
        "note": "descriptive diagnostic of label behaviour in kill-free windows of the same length; not a causal estimate",
        "inputs": {"CONTROL_V_h90.npz": score.sha256_file(labels / "CONTROL_V_h90.npz"), "LABELS_h90.npz": score.sha256_file(labels / "LABELS_h90.npz"),
                   "CONTROLS_MATCHING_h90.npz": score.sha256_file(labels / "CONTROLS_MATCHING_h90.npz")},
        "code_sha256": code_hashes()}
    report["table_sha256"] = assemble.write_npz(labels / "CONTROLS_h90.npz", out_table)
    (labels / "CONTROLS_h90.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--labels", required=True)
    s.add_argument("--run-root", required=True)
    s.add_argument("--end-rows", default=None)
    s.add_argument("--device", default="cuda")
    c = sub.add_parser("compare")
    c.add_argument("--labels", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if args.command == "score":
        end_rows = Path(args.end_rows) if args.end_rows else Path(args.labels) / END_ROWS_DEFAULT
        record = score_controls(Path(args.labels), Path(args.run_root), end_rows, args.device)
        print(json.dumps({key: record[key] for key in ("starts", "ends", "sha256")}, indent=1))
    else:
        report = compare(Path(args.labels))
        print(json.dumps({key: report[key] for key in ("eligible_q_dev_h90_engagements", "matched", "matched_rate")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
