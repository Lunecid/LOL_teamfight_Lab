"""L5 of label production: measurement diagnostics on Q_DEV h90 keys (reported, never used to relabel).

reports/research_execution_20261003/PROTOCOL_KO.md §2 second layer and CONTRACT.json
``v_stability``.  ``freeze-keys`` writes the Q_DEV h90 key list before any
prediction exists; ``run`` computes the fixed tables over exactly those keys.
Match-equal weighting is primary (each match weighs 1, shared by its
engagements); row-equal is secondary.  There are no pass/fail thresholds.

Usage: python -B studies/label_production_20261005/diagnostics.py freeze-keys --labels D:/LOL_Training_20261004/labels_1
       python -B studies/label_production_20261005/diagnostics.py run --labels D:/LOL_Training_20261004/labels_1
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import assemble, score  # noqa: E402

VERSION = "label_diagnostics_v1_20261005"
PROJECTION = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
Q_DEV = assemble.ROLES.index("Q_DEV")
GAME_TIME_BINS_MIN = (15, 25)
PRE_AGE_BINS_S = (5,)
PROSPECT_BINS = (0.2, 0.4, 0.6, 0.8)


def freeze_keys(labels: Path) -> dict[str, Any]:
    """Freeze the Q_DEV h90 keys from the engagement table, before predictions exist."""

    labels = Path(labels)
    target = labels / "Q_DEV_H90_KEYS.npy"
    if target.is_file():
        raise FileExistsError("Q_DEV h90 keys are already frozen")
    if any((labels / f"PRED_{name}.npz").is_file() for name in assemble.CONTEXTS):
        raise RuntimeError("predictions already exist: keys must be frozen before any prediction")
    eng = dict(np.load(labels / "ENGAGEMENTS.npz", allow_pickle=False))
    rows = (eng["role"] == Q_DEV) & eng["valid_h90"]
    keys = np.char.add(np.char.add(eng["canonical_match_key"][rows], b":"), eng["component_key"][rows])
    with target.open("wb") as stream:
        np.save(stream, keys, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    record = {"version": VERSION, "frozen_utc": datetime.now(timezone.utc).isoformat(), "keys": int(len(keys)),
              "keys_sha256": score.sha256_file(target), "engagements_sha256": score.sha256_file(labels / "ENGAGEMENTS.npz"),
              "predictions_present_at_freeze": False}
    (labels / "Q_DEV_H90_KEYS.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def _weights(match_index: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(match_index, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def summary(svi: np.ndarray, match_index: np.ndarray) -> dict[str, Any]:
    """Distribution of valid SVI: match-equal (primary) and row-equal (secondary)."""

    ok = np.isfinite(svi)
    svi, match_index = svi[ok], match_index[ok]
    if not len(svi):
        return {"rows": 0}
    out: dict[str, Any] = {"rows": int(len(svi)), "matches": int(len(np.unique(match_index)))}
    for name, w in (("match_equal", _weights(match_index)), ("row_equal", np.ones(len(svi)))):
        w = w / w.sum()
        pos, neg, zero = svi > 0, svi < 0, svi == 0
        block = {"mean": float(np.sum(w * svi)), "mean_abs": float(np.sum(w * np.abs(svi))),
                 "positive": float(w[pos].sum()), "negative": float(w[neg].sum()), "exact_zero": float(w[zero].sum()),
                 "mean_given_positive": float(np.sum(w[pos] * svi[pos]) / w[pos].sum()) if pos.any() else None,
                 "mean_given_negative": float(np.sum(w[neg] * svi[neg]) / w[neg].sum()) if neg.any() else None}
        for threshold in assemble.NEAR_ZERO:
            block[f"abs_lt_{threshold}"] = float(w[np.abs(svi) < threshold].sum())
        out[name] = block
    return out


def strata(table: dict[str, np.ndarray], rows: np.ndarray, labels_for: np.ndarray, names: list[str]) -> dict[str, Any]:
    return {name: summary(table["svi_delta"][rows][labels_for == i], table["match_index"][rows][labels_for == i])
            for i, name in enumerate(names)}


def run(labels: Path) -> dict[str, Any]:
    labels = Path(labels)
    frozen = json.loads((labels / "Q_DEV_H90_KEYS.json").read_text(encoding="utf-8"))
    keys_frozen = np.load(labels / "Q_DEV_H90_KEYS.npy", allow_pickle=False)
    if score.sha256_file(labels / "Q_DEV_H90_KEYS.npy") != frozen["keys_sha256"]:
        raise ValueError("frozen Q_DEV keys changed")
    tables = {h: dict(np.load(labels / f"LABELS_h{h}.npz", allow_pickle=False)) for h in assemble.HORIZONS}
    h90 = tables[90]
    keys = np.char.add(np.char.add(h90["canonical_match_key"], b":"), h90["component_key"])
    rows = np.flatnonzero(np.isin(keys, keys_frozen))
    if len(rows) != len(keys_frozen):
        raise ValueError("label rows do not cover the frozen Q_DEV keys exactly")
    report: dict[str, Any] = {"version": VERSION, "utc": datetime.now(timezone.utc).isoformat(), "keys": frozen,
                              "missing_or_invalid": int((~h90["valid"][rows]).sum())}
    report["distribution"] = summary(h90["svi_delta"][rows], h90["match_index"][rows])
    same_frame = (h90["F_pre_ms"][rows] == h90["F_post_ms"][rows]).astype(int)
    report["frame"] = strata(h90, rows, same_frame, ["updated_frame", "same_frame"])
    minutes = h90["pre_ms"][rows] / 60000.0
    report["game_time"] = strata(h90, rows, np.digitize(minutes, GAME_TIME_BINS_MIN), ["lt15", "15_25", "ge25"])
    age = h90["pre_frame_age_ms"][rows] / 1000.0
    report["pre_frame_age"] = strata(h90, rows, (age > PRE_AGE_BINS_S[0]).astype(int), ["0_5s", "5_10s"])
    report["pre_prospect"] = strata(h90, rows, np.digitize(h90["p_pre"][rows], PROSPECT_BINS), ["0_.2", ".2_.4", ".4_.6", ".6_.8", ".8_1"])
    bits = h90["truncation_bits"][rows]
    report["truncation"] = {name: summary(h90["svi_delta"][rows][(bits >> bit) & 1 == 1], h90["match_index"][rows][(bits >> bit) & 1 == 1])
                            for bit, name in enumerate(("horizon", "next_kill_minus_1", "next_anchor_minus_1", "game_end_minus_1"))}
    follow = h90["actual_followup_ms"][rows] / 1000.0
    report["actual_followup_s"] = {"mean": float(np.mean(follow)), "quantiles": {str(q): float(np.quantile(follow, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
    # windows: intersection of keys valid in all three, plus each window's exclusion rate
    valid_all = np.ones(len(rows), dtype=bool)
    windows: dict[str, Any] = {}
    for h, table in tables.items():
        valid_all &= table["valid"][rows]
        windows[f"h{h}"] = {"excluded_rate": float(1 - table["valid"][rows].mean())}
    for h, table in tables.items():
        windows[f"h{h}"]["intersection"] = summary(np.where(valid_all, table["svi_delta"][rows], np.nan), table["match_index"][rows])
    for a, b in combinations(tables, 2):
        da, db = tables[a]["svi_delta"][rows][valid_all], tables[b]["svi_delta"][rows][valid_all]
        windows[f"h{a}_vs_h{b}"] = {"direction_agreement": float(np.mean((da > 0) == (db > 0))), "mean_abs_delta_difference": float(np.mean(np.abs(da - db)))}
    report["windows"] = windows
    # six V on the same keys (own raw probability, own calibrator)
    stability = dict(np.load(labels / "STABILITY_h90.npz", allow_pickle=False))
    s_keys = np.char.add(np.char.add(stability["canonical_match_key"], b":"), stability["component_key"])
    pick = np.isin(s_keys, keys_frozen)
    pairs: dict[str, Any] = {}
    w_all = _weights(stability["match_index"][pick])
    for a, b in combinations(assemble.CONTEXTS, 2):
        da, db = stability[f"svi_{a}"][pick], stability[f"svi_{b}"][pick]
        both = np.isfinite(da) & np.isfinite(db)
        w = w_all[both] / w_all[both].sum() if both.any() else w_all[both]
        pairs[f"{a}|{b}"] = {"rows": int(both.sum()), "missing": int((~both).sum()),
                             "direction_agreement_match_equal": float(np.sum(w * ((da[both] > 0) == (db[both] > 0)))) if both.any() else None,
                             "direction_agreement_row_equal": float(np.mean((da[both] > 0) == (db[both] > 0))) if both.any() else None,
                             "mean_signed_difference": float(np.sum(w * (da[both] - db[both]))) if both.any() else None,
                             "mean_abs_difference": float(np.sum(w * np.abs(da[both] - db[both]))) if both.any() else None,
                             "exact_zero_either": int(((da[both] == 0) | (db[both] == 0)).sum()),
                             "near_zero_either_lt_0.01": int(((np.abs(da[both]) < 0.01) | (np.abs(db[both]) < 0.01)).sum())}
    report["six_v"] = {"pairs": pairs, "final_reference": {name: pairs[f"final|{name}"] for name in assemble.CONTEXTS[1:]}}
    # descriptive event association (shared inputs make it mechanical, not an independent truth)
    kills = np.load(PROJECTION / "TEAM_KILLS.npy", mmap_mode="r", allow_pickle=False)
    objectives = np.load(PROJECTION / "OBJECTIVE_COUNTS.npy", mmap_mode="r", allow_pickle=False)
    pre_q, post_q = h90["query_pre"][rows], h90["query_post"][rows]
    kill_diff = (kills[post_q, 0] - kills[pre_q, 0]) - (kills[post_q, 1] - kills[pre_q, 1])  # Blue minus Red kills gained
    objective_gain = (objectives[post_q].sum(axis=1) - objectives[pre_q].sum(axis=1)).astype(np.int64)
    report["event_association_note"] = "kills/objectives share inputs with V: mechanical association, not an independent ground truth"
    report["kill_differential"] = strata(h90, rows, np.sign(kill_diff).astype(int) + 1, ["red_more", "even", "blue_more"])
    report["objective_events_in_window"] = strata(h90, rows, (objective_gain > 0).astype(int), ["none", "some"])
    pending = labels / f".DIAGNOSTICS_LAYER2.json.{os.getpid()}.pending"
    pending.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending, labels / "DIAGNOSTICS_LAYER2.json")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-keys", "run"):
        sub.add_parser(name).add_argument("--labels", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-keys":
        print(json.dumps(freeze_keys(Path(args.labels)), indent=1))
    else:
        report = run(Path(args.labels))
        print(json.dumps({"keys": report["keys"]["keys"], "distribution": report["distribution"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
