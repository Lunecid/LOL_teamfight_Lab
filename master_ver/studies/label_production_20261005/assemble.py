"""L3 of label production: label rows from the engagement table and the per-context predictions.

One file per horizon (h60/h90/h120) with every field of the contract's
``label_row_required`` (row-level fields here, run-level hashes in the
manifest), plus the six-V stability table for Q_DEV h90 keys.

Rules (reports/post_state_label_contract_20261002, CONTRACT.json ``label``):
SVI = p_post - p_pre in float64 with the same context's V at both ends;
direction 1 iff SVI > 0 (no tolerance); exact zero -> direction 0 and
exact_zero 1; invalid -> SVI/direction/exact_zero all NA (here NaN / -1) with
reasons kept; probabilities are never repaired by clipping.

Design: reports/label_production_20261005/LABEL_PRODUCER_DESIGN_KO.md (L3).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "label_rows_v1_20261005"
CONTRACT_VERSION = "post_state_label_v1_20261002"
CONTRACT = ROOT / "reports/post_state_label_contract_20261002/CONTRACT.json"
PROJECTION = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
CONTEXTS = ("final", "oof0", "oof1", "oof2", "oof3", "oof4")   # CONTEXT_MEMBERSHIP row order
ROLES = ("TRAIN", "V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "TEST")
Q_FINAL_ROLES = (3, 4, 5)
HORIZONS = (60, 90, 120)
NEAR_ZERO = (0.005, 0.01, 0.02)
CLIP = 1e-12
INVALID = {"v_role_game_not_a_q_row": 1, "missing_prediction": 2, "nonfinite_probability": 4, "probability_out_of_range": 8,
           "endpoint_invalid": 16, "oof_exclusion_failed": 32}
FLAGS = {"pre_raw_clipped": 1, "post_raw_clipped": 2, "pre_saturated": 4, "post_saturated": 8, "raw_tie": 16,
         "raw_sign_differs": 32, "calibrated_tie": 64}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(labels: Path, context: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = Path(labels) / f"PRED_{context}_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(labels) / f"PRED_{context}.npz"
    if sha256_file(path) != manifest["pred_sha256"]:
        raise ValueError(f"{context}: PRED file differs from its manifest")
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in ("query_index", "logit", "raw_p", "v")}, manifest


def lookup(pred: dict[str, np.ndarray], query_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions of query indices in a sorted prediction table, and which were found."""

    keys = pred["query_index"]
    position = np.clip(np.searchsorted(keys, query_index), 0, max(len(keys) - 1, 0))
    found = (query_index >= 0) & (len(keys) > 0) & (keys[position] == query_index)
    return position, found


def sign(value: np.ndarray) -> np.ndarray:
    return np.sign(value).astype(np.int8)


def label_table(eng: dict[str, np.ndarray], preds: dict[str, dict[str, np.ndarray]], membership: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    n = len(eng["role"])
    role, fold = eng["role"].astype(np.int64), eng["fold"].astype(np.int64)
    context = np.full(n, -1, dtype=np.int8)
    context[np.isin(role, Q_FINAL_ROLES)] = 0
    train = role == 0
    context[train] = 1 + fold[train]
    invalid = np.zeros(n, dtype=np.int32)
    invalid[context < 0] |= INVALID["v_role_game_not_a_q_row"]
    valid_endpoint = eng[f"valid_h{horizon}"].astype(bool)
    invalid[~valid_endpoint] |= INVALID["endpoint_invalid"]
    q_pre, q_post = eng["query_pre"].astype(np.int64), eng[f"query_h{horizon}"].astype(np.int64)
    out = {name: np.full(n, np.nan) for name in ("logit_pre", "logit_post", "raw_pre", "raw_post", "p_pre", "p_post")}
    bits = np.zeros(n, dtype=np.uint8)
    for code, name in enumerate(CONTEXTS):
        rows = np.flatnonzero(context == code)
        if not len(rows):
            continue
        bits[rows] = membership[code, eng["match_index"][rows]]
        if name not in preds:
            invalid[rows] |= INVALID["missing_prediction"]
            continue
        pred = preds[name]
        for end, queries in (("pre", q_pre[rows]), ("post", q_post[rows])):
            position, found = lookup(pred, queries)
            invalid[rows[~found]] |= INVALID["missing_prediction"]
            for target, source in ((f"logit_{end}", "logit"), (f"raw_{end}", "raw_p"), (f"p_{end}", "v")):
                values = np.full(len(rows), np.nan)
                values[found] = pred[source][position[found]]
                out[target][rows] = values
    # Whole-game exclusion proof: the scoring context never saw the game in fit/stop/refit/dev/cal (bits 1..16);
    # a TRAIN game must be the context's held-out target (bit 32).
    scored = context >= 0
    leaked = scored & ((bits & 31) != 0)
    leaked |= train & ((bits & 32) == 0)
    invalid[leaked] |= INVALID["oof_exclusion_failed"]
    probabilities = np.stack([out["p_pre"], out["p_post"]])
    finite = np.isfinite(probabilities).all(axis=0)
    invalid[scored & ~finite & ((invalid & INVALID["missing_prediction"]) == 0)] |= INVALID["nonfinite_probability"]
    in_range = finite & (probabilities >= 0).all(axis=0) & (probabilities <= 1).all(axis=0)
    invalid[scored & finite & ~in_range] |= INVALID["probability_out_of_range"]
    valid = invalid == 0
    svi = np.where(valid, out["p_post"] - out["p_pre"], np.nan)
    raw_delta = np.where(valid, out["raw_post"] - out["raw_pre"], np.nan)
    direction = np.where(valid, (svi > 0).astype(np.int8), -1).astype(np.int8)
    exact_zero = np.where(valid, (svi == 0).astype(np.int8), -1).astype(np.int8)
    near = np.zeros(n, dtype=np.uint8)
    for bit, threshold in enumerate(NEAR_ZERO):
        near[valid & (np.abs(np.nan_to_num(svi)) < threshold)] |= 1 << bit
    flags = np.zeros(n, dtype=np.uint8)
    for end, raw_bit, sat_bit in (("pre", FLAGS["pre_raw_clipped"], FLAGS["pre_saturated"]), ("post", FLAGS["post_raw_clipped"], FLAGS["post_saturated"])):
        raw, p = out[f"raw_{end}"], out[f"p_{end}"]
        flags[valid & ((raw <= CLIP) | (raw >= 1 - CLIP))] |= raw_bit
        flags[valid & ((p == 0) | (p == 1))] |= sat_bit
    flags[valid & (out["raw_pre"] == out["raw_post"])] |= FLAGS["raw_tie"]
    flags[valid & (sign(np.nan_to_num(raw_delta)) != sign(np.nan_to_num(svi)))] |= FLAGS["raw_sign_differs"]
    flags[valid & (svi == 0)] |= FLAGS["calibrated_tie"]
    table = {
        "canonical_match_key": eng["canonical_match_key"], "component_key": eng["component_key"], "match_index": eng["match_index"],
        "role": eng["role"], "fold": eng["fold"], "context": context, "horizon_s": np.full(n, horizon, dtype=np.int16),
        "first_kill_ms": eng["first_kill_ms"], "last_kill_ms": eng["last_kill_ms"], "pre_ms": eng["pre_ms"],
        "post_ms": eng[f"post_ms_h{horizon}"], "F_pre_ms": eng["F_pre_ms"], "F_post_ms": eng[f"F_post_ms_h{horizon}"],
        "pre_frame_age_ms": eng["pre_frame_age_ms"], "post_frame_age_ms": eng[f"post_frame_age_ms_h{horizon}"],
        "first_kill_frame_lag_ms": eng["first_kill_frame_lag_ms"], "actual_followup_ms": eng[f"actual_followup_ms_h{horizon}"],
        "comparison_elapsed_ms": eng[f"comparison_elapsed_ms_h{horizon}"], "truncation_bits": eng[f"truncation_bits_h{horizon}"],
        "global_isolation_verified": eng["global_isolation_verified"], "context_membership_bits": bits,
        "query_pre": q_pre, "query_post": q_post, **out, "svi_delta": svi, "raw_delta": raw_delta,
        "direction_positive": direction, "exact_zero": exact_zero, "near_zero_bits": near, "clip_saturation_tie_bits": flags,
        "valid": valid, "invalid_bits": invalid,
    }
    return table


def stability_table(eng: dict[str, np.ndarray], preds: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Q_DEV h90 keys under each of the six V (own raw probability, own calibrator)."""

    rows = np.flatnonzero(eng["role"] == ROLES.index("Q_DEV"))
    table: dict[str, np.ndarray] = {"canonical_match_key": eng["canonical_match_key"][rows], "component_key": eng["component_key"][rows],
                                    "match_index": eng["match_index"][rows]}
    for name in CONTEXTS:
        p = {}
        for end, column in (("pre", "query_pre"), ("post", "query_h90")):
            values = np.full(len(rows), np.nan)
            if name in preds:
                position, found = lookup(preds[name], eng[column][rows].astype(np.int64))
                values[found] = preds[name]["v"][position[found]]
            p[end] = values
        table[f"p_pre_{name}"], table[f"p_post_{name}"] = p["pre"], p["post"]
        table[f"svi_{name}"] = p["post"] - p["pre"]
    return table


def write_npz(path: Path, table: dict[str, np.ndarray]) -> str:
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.savez(stream, **table)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return sha256_file(path)


def assemble(labels: Path, *, require_all: bool = True) -> dict[str, Any]:
    labels = Path(labels)
    engagements_path = labels / "ENGAGEMENTS.npz"
    eng = dict(np.load(engagements_path, allow_pickle=False))
    preds, identities = {}, {}
    for name in CONTEXTS:
        if (labels / f"PRED_{name}.npz").is_file():
            preds[name], identities[name] = load_predictions(labels, name)
    if require_all and set(preds) != set(CONTEXTS):
        raise ValueError(f"predictions missing for {sorted(set(CONTEXTS) - set(preds))}")
    membership = np.load(PROJECTION / "CONTEXT_MEMBERSHIP.npy", allow_pickle=False)
    record: dict[str, Any] = {"version": VERSION, "contract_version": CONTRACT_VERSION, "contract_sha256": sha256_file(CONTRACT),
                              "code_sha256": sha256_file(Path(__file__)), "engagements_sha256": sha256_file(engagements_path),
                              "engagements_manifest_sha256": sha256_file(labels / "ENGAGEMENTS_MANIFEST.json"),
                              "v_sampling_ledger_sha256": sha256_file(PROJECTION / "V_ROWS.npy"),
                              "context_membership_sha256": sha256_file(PROJECTION / "CONTEXT_MEMBERSHIP.npy"),
                              "predictions": {name: {"pred_sha256": m["pred_sha256"], "queries_sha256": m["queries_sha256"], "v": m["v"]}
                                              for name, m in identities.items()},
                              "invalid_bits": INVALID, "flag_bits": FLAGS, "near_zero_thresholds": list(NEAR_ZERO),
                              "context_codes": {str(i): name for i, name in enumerate(CONTEXTS)}, "roles": list(ROLES),
                              "notes": ["q rows: TRAIN (OOF V of the fold), Q_DEV/Q_CAL/TEST (final V); V_DEV/V_CAL are V-role games, not q rows",
                                        "probabilities and SVI are float64; direction has no tolerance; NA = NaN (floats) / -1 (int8)"],
                              "files": {}, "counts": {}, "utc": datetime.now(timezone.utc).isoformat()}
    for horizon in HORIZONS:
        table = label_table(eng, preds, membership, horizon)
        name = f"LABELS_h{horizon}.npz"
        record["files"][name] = write_npz(labels / name, table)
        q_rows = table["context"] >= 0
        valid = table["valid"]
        record["counts"][f"h{horizon}"] = {
            "q_rows": int(q_rows.sum()), "valid": int(valid.sum()), "invalid_q_rows": int((q_rows & ~valid).sum()),
            "positive": int((table["direction_positive"] == 1).sum()), "exact_zero": int((table["exact_zero"] == 1).sum()),
            "by_role_valid": {ROLES[c]: int((valid & (table["role"] == c)).sum()) for c in range(len(ROLES))},
            "invalid_reasons": {reason: int(((table["invalid_bits"] & bit) != 0).sum()) for reason, bit in INVALID.items()},
        }
    record["files"]["STABILITY_h90.npz"] = write_npz(labels / "STABILITY_h90.npz", stability_table(eng, preds))
    pending = labels / f".LABELS_MANIFEST.json.{os.getpid()}.pending"
    pending.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending, labels / "LABELS_MANIFEST.json")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--partial", action="store_true", help="assemble with the contexts available so far (never for the lock)")
    args = parser.parse_args(argv)
    record = assemble(Path(args.labels), require_all=not args.partial)
    print(json.dumps(record["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
