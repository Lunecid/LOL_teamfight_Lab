"""q row table (reports/architecture_20261005/ARCHITECTURE_KO.md section 4.1).

One table per horizon, built only from a locked label version (LABEL_LOCK.json), so
every q cell shares the same keys, labels and weights:
  - rows: valid label rows of the q roles TRAIN, Q_DEV, Q_CAL, TEST (V_DEV/V_CAL games
    are not q rows); invalid q rows are excluded and counted by reason;
  - y = direction_positive (1 when SVI > 0, exact zero is 0);
  - q inner STOP: whole TRAIN matches with first8_u64(SHA256(q salt + key)) % 10 == 0
    (EXECUTION_CARDS.json inner_stop.q_salt; distinct from the V STOP salt);
  - raw weight 1/m with m = eligible rows of the match in this table; training scopes use
    n_rows / (n_matches * m) (EXECUTION_CARDS.json q.weighting).
Columns prefixed ``pt_`` or ``label_`` are restricted: ``pt_p_pre`` may be read only by the
p_only / PT_linear / PT_flex references, ``label_*`` never by a model.  ``eval_*`` columns (kill
count, first-kill credit per team, from ENGAGEMENTS.npz) define reporting strata only (D-Q17).
TEST labels are sealed: in the row table TEST rows carry y = -1 and NaN label columns, and
their labels go to Q_TEST_SEALED_h*.npz (sha256 in the manifest), read only after SCORE_LOCK.

CLI: python -B -m studies.q_production_20261005.rows --labels D:/LOL_Training_20261004/labels_1
     --out D:/LOL_Training_20261004/q_run_1 [--horizon 90]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import assemble  # noqa: E402
from studies.v_preparation_20261002 import partition  # noqa: E402

VERSION = "q_rows_v1_20261005"
Q_STOP_SALT = "full_study_20261001_q_inner_v2:"
Q_ROLES = {"TRAIN": 0, "Q_DEV": 3, "Q_CAL": 4, "TEST": 5}
HORIZONS = (60, 90, 120)
RESTRICTED = ("pt_p_pre", "label_svi_delta")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def q_inner_stop(match_key: str) -> bool:
    """Whether a TRAIN match is in the q inner STOP set (whole match moves together)."""

    return partition._hash_u64(Q_STOP_SALT, match_key) % 10 == 0


def verify_lock(labels: Path, horizon: int) -> dict[str, Any]:
    """Return LABEL_LOCK.json after checking that q may start and the files it binds are unchanged."""

    labels = Path(labels)
    path = labels / "LABEL_LOCK.json"
    if not path.is_file():
        raise FileNotFoundError("labels are not locked; q may not start before LABEL_LOCK.json")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("q_may_start") is not True:
        raise ValueError("the label lock does not allow q to start")
    for name in (f"LABELS_h{horizon}.npz", "ENGAGEMENTS.npz", "Q_DEV_H90_KEYS.npy"):
        recorded = lock.get("files", {}).get(name)
        if recorded is None or sha256_file(labels / name) != recorded:
            raise ValueError(f"{name} differs from the label lock")
    return lock


def match_weights(match_index: np.ndarray) -> np.ndarray:
    """Raw weight 1/m: every match contributes total weight 1."""

    _, inverse, counts = np.unique(np.asarray(match_index), return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def normalized_weights(match_index: np.ndarray) -> np.ndarray:
    """Training weight n_rows / (n_matches * m) over the given scope rows (mean 1)."""

    match_index = np.asarray(match_index)
    if not len(match_index):
        return np.zeros(0)
    n_matches = len(np.unique(match_index))
    return len(match_index) / n_matches * match_weights(match_index)


EVAL_COLUMNS = {"eval_n_kills": "n_members", "eval_firstkill_credit_blue": "firstkill_credit_blue", "eval_firstkill_credit_red": "firstkill_credit_red"}


def evaluation_columns(rows: np.ndarray, table: dict[str, np.ndarray], engagements: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluation-only engagement columns (kill count, first-kill credit per team) for the selected rows.

    They define reporting strata added before any q result (DECISIONS D-Q17) and are never model inputs:
    q inputs come only from the raw store."""

    for key in ("canonical_match_key", "component_key"):
        if len(engagements[key]) != len(table[key]) or not np.array_equal(engagements[key][rows], table[key][rows]):
            raise ValueError("ENGAGEMENTS.npz does not align with the label table")
    return {name: np.asarray(engagements[source])[rows].astype(np.int16) for name, source in EVAL_COLUMNS.items()}


def build_rows(table: dict[str, np.ndarray], frozen_q_dev_keys: np.ndarray | None = None,
               engagements: dict[str, np.ndarray] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select q rows from one LABELS_h*.npz table; return the row table and its counts."""

    role = table["role"].astype(np.int64)
    q_role = np.isin(role, list(Q_ROLES.values()))
    valid = table["valid"].astype(bool)
    keep = q_role & valid
    direction = table["direction_positive"].astype(np.int64)
    if not np.isin(direction[keep], (0, 1)).all():
        raise ValueError("a valid q row has no binary direction label")
    excluded = q_role & ~valid
    invalid = table["invalid_bits"].astype(np.int64)
    rows = np.flatnonzero(keep)
    match_key = table["canonical_match_key"][rows]
    match_index = table["match_index"][rows].astype(np.int64)
    out_role = role[rows]
    stop = np.zeros(len(rows), dtype=bool)
    train = out_role == Q_ROLES["TRAIN"]
    stop[train] = [q_inner_stop(bytes(key).decode("ascii")) for key in match_key[train]]
    pre_ms = table["pre_ms"][rows].astype(np.int64)
    out = {
        "canonical_match_key": match_key, "component_key": table["component_key"][rows],
        "engagement_key": np.char.add(np.char.add(match_key, b":"), table["component_key"][rows]),
        "match_index": match_index, "role": out_role.astype(np.int8), "fold": table["fold"][rows].astype(np.int8),
        "q_stop": stop, "y": direction[rows].astype(np.int8), "weight_raw": match_weights(match_index),
        "query_pre": table["query_pre"][rows].astype(np.int64), "pre_ms": pre_ms, "elapsed_min": pre_ms / 60000.0,
        "frame_age_s": (pre_ms - table["F_pre_ms"][rows].astype(np.int64)) / 1000.0,
        "truncation_bits": table["truncation_bits"][rows], "near_zero_bits": table["near_zero_bits"][rows],
        "exact_zero": table["exact_zero"][rows], "label_svi_delta": table["svi_delta"][rows],
        "pt_p_pre": table["p_pre"][rows],
    }
    if engagements is not None:
        out.update(evaluation_columns(rows, table, engagements))
    if len(np.unique(out["engagement_key"])) != len(rows):
        raise ValueError("engagement keys are not unique")
    if not np.isfinite(out["pt_p_pre"]).all():
        raise ValueError("a valid q row lacks its pre probability")
    counts: dict[str, Any] = {"rows": int(len(rows)), "excluded_invalid_q_rows": int(excluded.sum()),
                              "excluded_reasons": {reason: int(((invalid[excluded] & bit) != 0).sum()) for reason, bit in assemble.INVALID.items()}}
    for name, code in Q_ROLES.items():
        mask = out_role == code
        y, w = out["y"][mask], out["weight_raw"][mask]
        counts[name] = {"rows": int(mask.sum()), "matches": int(len(np.unique(match_index[mask]))), "positive": int(y.sum()),
                        "positive_rate_unit": float(y.mean()) if mask.any() else None,
                        "positive_rate_match_weighted": float((w * y).sum() / w.sum()) if mask.any() else None}
    train_rows = out_role == Q_ROLES["TRAIN"]
    counts["TRAIN"].update({"stop_rows": int(stop.sum()), "stop_matches": int(len(np.unique(match_index[stop]))),
                            "fit_rows": int((train_rows & ~stop).sum()), "fit_matches": int(len(np.unique(match_index[train_rows & ~stop])))})
    if frozen_q_dev_keys is not None:
        q_dev = out["engagement_key"][out_role == Q_ROLES["Q_DEV"]]
        frozen = np.asarray(frozen_q_dev_keys)
        counts["q_dev_frozen_keys"] = {"frozen": int(len(frozen)), "q_rows": int(len(q_dev)),
                                       "q_rows_outside_frozen": int((~np.isin(q_dev, frozen)).sum()),
                                       "frozen_not_q_rows": int((~np.isin(frozen, q_dev)).sum())}
        if counts["q_dev_frozen_keys"]["q_rows_outside_frozen"]:
            raise ValueError("Q_DEV rows outside the keys frozen before predictions")
    return out, counts


VARIANT_HORIZON = {"main_h90": 90, "new_V_h60": 60, "new_V_h120": 120, "exclude_near_zero_h90": 90, "exclude_zero_coordinate_h90": 90}
NEAR_ZERO = 0.01


def derive_variant(rows: dict[str, np.ndarray], variant: str, zero_coordinate: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Row table of a filtered sensitivity variant from the public h90 table (ARCHITECTURE_KO.md section 5.2).

    exclude_near_zero_h90 keeps rows with |delta| >= 0.01 and delta != 0.  TEST deltas are sealed,
    so TEST rows stay here and the same filter is applied to them when the sealed labels are
    opened after SCORE_LOCK (``test_filter``).  exclude_zero_coordinate_h90 drops, in every role,
    rows in which any player's pre-frame XY is exactly (0, 0).  Raw weights 1/m are recomputed on
    the kept rows; q STOP membership is per match and unchanged."""

    role = np.asarray(rows["role"])
    test = role == Q_ROLES["TEST"]
    if variant == "exclude_near_zero_h90":
        delta = np.asarray(rows["label_svi_delta"], dtype=np.float64)
        keep = test | ((np.abs(delta) >= NEAR_ZERO) & (delta != 0))
        test_filter = "abs(label_svi_delta) >= 0.01 and label_svi_delta != 0, applied to TEST after SCORE_LOCK"
    elif variant == "exclude_zero_coordinate_h90":
        if zero_coordinate is None or len(zero_coordinate) != len(role):
            raise ValueError("exclude_zero_coordinate_h90 needs one zero-coordinate flag per row")
        keep = ~np.asarray(zero_coordinate, dtype=bool)
        test_filter = None
    else:
        raise ValueError(f"{variant} is not a filtered variant of the h90 table")
    out = {name: np.asarray(value)[keep] for name, value in rows.items()}
    out["weight_raw"] = match_weights(out["match_index"])
    by_role = {name: {"rows": int((role[keep] == code).sum()), "excluded": int(((role == code) & ~keep).sum())} for name, code in Q_ROLES.items()}
    return out, {"variant": variant, "rows": int(keep.sum()), "excluded": int((~keep).sum()), "test_filter": test_filter, "by_role": by_role}


SEALED_COLUMNS = ("y", "label_svi_delta", "exact_zero", "near_zero_bits")


def seal_test(rows: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split TEST labels out of the row table (TEST is scored once, after SCORE_LOCK)."""

    test = rows["role"] == Q_ROLES["TEST"]
    sealed = {"engagement_key": rows["engagement_key"][test], **{name: rows[name][test] for name in SEALED_COLUMNS}}
    public = {name: value.copy() for name, value in rows.items()}
    public["y"][test] = -1
    public["label_svi_delta"][test] = np.nan
    public["exact_zero"][test] = -1
    public["near_zero_bits"][test] = 0
    return public, sealed


def write_rows(labels: Path, out_dir: Path, horizon: int = 90) -> dict[str, Any]:
    labels, out_dir = Path(labels), Path(out_dir)
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    lock = verify_lock(labels, horizon)
    table = dict(np.load(labels / f"LABELS_h{horizon}.npz", allow_pickle=False))
    frozen = np.load(labels / "Q_DEV_H90_KEYS.npy", allow_pickle=False) if horizon == 90 else None
    with np.load(labels / "ENGAGEMENTS.npz", allow_pickle=False) as eng:
        engagements = {name: eng[name] for name in ("canonical_match_key", "component_key", *EVAL_COLUMNS.values())}
    rows, counts = build_rows(table, frozen, engagements)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"Q_ROWS_h{horizon}.npz"
    if target.exists():
        raise FileExistsError(f"{target} exists; q rows are written once per label lock")
    public, sealed = seal_test(rows)
    record = {"version": VERSION, "utc": datetime.now(timezone.utc).isoformat(), "horizon_s": horizon,
              "label_lock_sha256": sha256_file(labels / "LABEL_LOCK.json"), "label_lock_utc": lock.get("locked_utc"),
              "labels_sha256": lock["files"][f"LABELS_h{horizon}.npz"], "q_stop_salt": Q_STOP_SALT, "restricted_columns": list(RESTRICTED),
              "sealed_test_columns": list(SEALED_COLUMNS), "code_sha256": sha256_file(Path(__file__)), "counts": counts}
    record["test_sealed_sha256"] = assemble.write_npz(out_dir / f"Q_TEST_SEALED_h{horizon}.npz", sealed)
    record["rows_sha256"] = assemble.write_npz(target, public)
    (out_dir / f"Q_ROWS_h{horizon}.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon", type=int, default=90)
    args = parser.parse_args(argv)
    print(json.dumps(write_rows(Path(args.labels), Path(args.out), args.horizon)["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
