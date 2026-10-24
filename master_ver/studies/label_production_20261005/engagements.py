"""L0 of label production: the primary-supported engagement table (no V model needed).

Reads the observable-state bank (per-shard component arrays) and the input
projection (MATCH_TABLE, MATCH_QUERIES, QUERY_TABLE) and writes one row per
primary-supported engagement: times, endpoints and their truncation reasons,
frame ages, role/fold, and the global QUERY_TABLE index of the pre state and of
each post state (h60/h90/h120).  Every endpoint is recomputed with the
population rule and every query time is checked against both the bank and the
projection; any disagreement stops the run.

Definitions: reports/post_state_label_contract_20261002 (post_state_label_v1_20261002),
reports/population_adoption_20261002/CONTRACT.json.  Design:
reports/label_production_20261005/LABEL_PRODUCER_DESIGN_KO.md (L0).

Usage: python -B studies/label_production_20261005/engagements.py --output D:/LOL_Training_20261004/labels_1
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

VERSION = "label_engagements_v1_20261005"
BANK = ROOT / "outputs/observable_state_20261002/full_static_preserved_20261002_1"
PROJECTION = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
LABEL_CONTRACT = ROOT / "reports/post_state_label_contract_20261002/CONTRACT.json"
ROLES = ("TRAIN", "V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "TEST")      # MATCH_TABLE.role codes (v_preparation run.py)
HORIZONS = (60, 90, 120)
REFS = ("pre", "h60", "h90", "h120")                                 # FIELD_MANIFEST component_query_refs
TRUNCATION = ("horizon", "next_kill_minus_1", "next_anchor_minus_1", "game_end_minus_1")  # bit i = 1 << i
COMPONENT_COLUMNS = (
    "component_id", "n_members", "same_ms_first_n", "f", "L", "pre", "F_pre_ms", "F_pre_index", "pre_frame_age_ms",
    "first_kill_frame_lag_ms", "firstkill_credit_bits", "firstkill_credit_blue", "firstkill_credit_red", "c0_n_env_killer",
    "next_kill_after_L", "next_component_anchor", "foreign_kills_in_pre_L", "other_component_anchors_in_pre_L", "game_end_ms",
    "winning_team", "support_ok_tau_r1800", "p_lower_bits_tau_r1800", "p_upper_bits_tau_r1800", "support_ok_tau_r2000",
    "p_lower_bits_tau_r2000", "p_upper_bits_tau_r2000", "pre_frame_support", "team_map_ok", "events_support_ok",
    "terminal_support_authoritative", "terminal_component_order_ok", "firstkill_direct_opposition", "global_isolation_verified",
    "overlap_next_anchor_le_L", "candidate_eligible", "supported_candidate_eligible", "primary_candidate_eligible",
    "early_candidate_eligible", "e_h60", "valid_h60", "post_frame_index_h60", "F_post_h60", "e_h90", "valid_h90",
    "post_frame_index_h90", "F_post_h90", "e_h120", "valid_h120", "post_frame_index_h120", "F_post_h120",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def endpoint(last_kill: int, horizon_s: int, next_kill: int, next_anchor: int, game_end: int) -> tuple[int, int]:
    """population.py `_endpoint` with -1 meaning "none": (endpoint, truncation bits)."""

    candidates = [(0, int(last_kill) + 1000 * int(horizon_s))]
    if next_kill >= 0:
        candidates.append((1, int(next_kill) - 1))
    if next_anchor >= 0:
        candidates.append((2, int(next_anchor) - 1))
    if game_end >= 0:
        candidates.append((3, int(game_end) - 1))
    value = min(v for _, v in candidates)
    return value, sum(1 << bit for bit, v in candidates if v == value)


def shard_dirs(bank: Path = BANK) -> list[Path]:
    return sorted(path for path in Path(bank).iterdir() if (path / "BANK.npz").is_file())


def read_shard(directory: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    names = ("component_values", "component_key", "component_query_refs", "query_ms")
    with np.load(directory / "BANK.npz", allow_pickle=False) as bank:
        arrays = {name: bank[name] for name in names}
        arrays.update({f"offset_{name}": bank[f"offset_{name}"] for name in names})
    with (directory / "MATCH_LEDGER.jsonl").open(encoding="utf-8") as stream:
        ledger = [json.loads(line) for line in stream if line.strip()]
    for name in names:
        if len(arrays[f"offset_{name}"]) != len(ledger) + 1:
            raise ValueError(f"{directory.name}: offsets of {name} do not cover the ledger")
    if arrays["component_values"].shape[1] != len(COMPONENT_COLUMNS):
        raise ValueError(f"{directory.name}: component_values width differs from the manifest")
    return ledger, arrays


def build(output: Path, bank: Path = BANK, projection: Path = PROJECTION) -> dict[str, Any]:
    manifest = json.loads((Path(bank) / "FIELD_MANIFEST.json").read_text(encoding="utf-8"))
    if tuple(manifest["component_columns"]) != COMPONENT_COLUMNS or tuple(manifest["component_query_refs"]) != REFS:
        raise ValueError("bank FIELD_MANIFEST component layout differs from this code")
    if manifest.get("query_refs_local_to_match") is not True:
        raise ValueError("bank query references are not match-local")
    col = {name: index for index, name in enumerate(COMPONENT_COLUMNS)}
    matches = np.load(Path(projection) / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    match_queries = np.load(Path(projection) / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)
    queries = np.load(Path(projection) / "QUERY_TABLE.npy", mmap_mode="r", allow_pickle=False)
    index_of = {bytes(key).decode("ascii"): position for position, key in enumerate(matches["key"])}
    if len(index_of) != len(matches):
        raise ValueError("MATCH_TABLE keys are not unique")

    rows: dict[str, list] = {name: [] for name in (
        "match_index", "canonical_match_key", "component_key", "component_id", "role", "fold", "patch_role", "target_Blue_win",
        "n_members", "first_kill_ms", "last_kill_ms", "pre_ms", "F_pre_ms", "pre_frame_age_ms", "first_kill_frame_lag_ms",
        "next_kill_after_L", "next_component_anchor", "game_end_ms", "global_isolation_verified",
        "firstkill_credit_blue", "firstkill_credit_red", "query_pre")}
    for h in HORIZONS:
        for name in ("post_ms", "valid", "F_post_ms", "post_frame_age_ms", "actual_followup_ms", "comparison_elapsed_ms",
                     "truncation_bits", "query"):
            rows[f"{name}_h{h}"] = []
    shard_hashes: dict[str, str] = {}
    seen_matches: set[int] = set()
    for directory in shard_dirs(bank):
        ledger, arrays = read_shard(directory)
        shard_hashes[directory.name] = sha256_file(directory / "BANK.npz")
        for local, entry in enumerate(ledger):
            key = str(entry["canonical_match_key"])
            match_index = index_of.get(key)
            if match_index is None or match_index in seen_matches:
                raise ValueError(f"{directory.name}: match {key} is missing from MATCH_TABLE or repeated")
            seen_matches.add(match_index)
            match = matches[match_index]
            lo, hi = (int(v) for v in arrays["offset_component_values"][local:local + 2])
            values = arrays["component_values"][lo:hi]
            k_lo, k_hi = (int(v) for v in arrays["offset_component_key"][local:local + 2])
            r_lo, r_hi = (int(v) for v in arrays["offset_component_query_refs"][local:local + 2])
            q_lo, q_hi = (int(v) for v in arrays["offset_query_ms"][local:local + 2])
            keys, refs, local_ms = arrays["component_key"][k_lo:k_hi], arrays["component_query_refs"][r_lo:r_hi], arrays["query_ms"][q_lo:q_hi]
            if not (len(values) == len(keys) == len(refs)):
                raise ValueError(f"{directory.name}: component arrays disagree for match {key}")
            first_query, n_queries = int(match_queries[match_index]["first_query"]), int(match_queries[match_index]["n_queries"])
            if n_queries != len(local_ms):
                raise ValueError(f"{directory.name}: bank and projection query counts differ for match {key}")
            for c in range(len(values)):
                v = values[c]
                if not (v[col["primary_candidate_eligible"]] and v[col["supported_candidate_eligible"]]):
                    if (refs[c] >= 0).any():
                        raise ValueError(f"{key}: a non-primary component carries query references")
                    continue
                f, last, pre = int(v[col["f"]]), int(v[col["L"]]), int(v[col["pre"]])
                if pre != f - 10_000 or not v[col["global_isolation_verified"]]:
                    raise ValueError(f"{key}: primary component breaks the anchor or isolation rule")

                def global_query(ref: int, expected_ms: int) -> int:
                    if ref < 0:
                        return -1
                    index = first_query + int(ref)
                    row = queries[index]
                    if (int(row["match_index"]) != match_index or int(row["local_query"]) != int(ref)
                            or int(row["query_ms"]) != expected_ms or int(local_ms[ref]) != expected_ms):
                        raise ValueError(f"{key}: query {index} does not match its engagement time {expected_ms}")
                    return index

                rows["match_index"].append(match_index)
                rows["canonical_match_key"].append(key.encode("ascii"))
                rows["component_key"].append(bytes(keys[c]))
                rows["component_id"].append(int(v[col["component_id"]]))
                rows["role"].append(int(match["role"]))
                rows["fold"].append(int(match["fold"]))
                rows["patch_role"].append(int(match["patch_role"]))
                rows["target_Blue_win"].append(bool(match["target_Blue_win"]))
                for name, column in (("n_members", "n_members"), ("first_kill_ms", "f"), ("last_kill_ms", "L"), ("pre_ms", "pre"),
                                     ("F_pre_ms", "F_pre_ms"), ("pre_frame_age_ms", "pre_frame_age_ms"),
                                     ("first_kill_frame_lag_ms", "first_kill_frame_lag_ms"), ("next_kill_after_L", "next_kill_after_L"),
                                     ("next_component_anchor", "next_component_anchor"), ("game_end_ms", "game_end_ms"),
                                     ("firstkill_credit_blue", "firstkill_credit_blue"), ("firstkill_credit_red", "firstkill_credit_red")):
                    rows[name].append(int(v[col[column]]))
                rows["global_isolation_verified"].append(bool(v[col["global_isolation_verified"]]))
                rows["query_pre"].append(global_query(int(refs[c][0]), pre))
                for slot, h in enumerate(HORIZONS, start=1):
                    valid = bool(v[col[f"valid_h{h}"]])
                    stored = int(v[col[f"e_h{h}"]])
                    recomputed, bits = endpoint(last, h, int(v[col["next_kill_after_L"]]), int(v[col["next_component_anchor"]]),
                                                int(v[col["game_end_ms"]]))
                    if valid and stored != recomputed:
                        raise ValueError(f"{key}: stored e_h{h}={stored} differs from the rule ({recomputed})")
                    post_frame = int(v[col[f"F_post_h{h}"]])
                    rows[f"post_ms_h{h}"].append(stored if valid else -1)
                    rows[f"valid_h{h}"].append(valid)
                    rows[f"F_post_ms_h{h}"].append(post_frame if valid else -1)
                    rows[f"post_frame_age_ms_h{h}"].append(stored - post_frame if valid else -1)
                    rows[f"actual_followup_ms_h{h}"].append(stored - last if valid else -1)
                    rows[f"comparison_elapsed_ms_h{h}"].append(stored - pre if valid else -1)
                    rows[f"truncation_bits_h{h}"].append(bits if valid else 0)
                    rows[f"query_h{h}"].append(global_query(int(refs[c][slot]), stored) if valid else -1)
                    if valid != (int(refs[c][slot]) >= 0):
                        raise ValueError(f"{key}: h{h} validity and query reference disagree")
    if len(seen_matches) != len(matches):
        raise ValueError(f"bank covers {len(seen_matches)} of {len(matches)} projection matches")

    dtypes = {"canonical_match_key": "S64", "component_key": "S64", "role": np.int8, "fold": np.int8, "patch_role": np.int8,
              "target_Blue_win": bool, "global_isolation_verified": bool, "match_index": np.int32, "component_id": np.int32}
    table = {name: np.asarray(values, dtype=dtypes.get(name, bool if name.startswith("valid_") else np.int64))
             for name, values in rows.items()}
    keys = np.char.add(np.char.add(table["canonical_match_key"], b":"), table["component_key"])
    if len(np.unique(keys)) != len(keys):
        raise ValueError("engagement keys are not unique")
    if (table["query_pre"] < 0).any() or (table["query_h90"] < 0).any() or not table["valid_h90"].all():
        raise ValueError("a primary-supported engagement lacks its pre or h90 query")
    counts = {ROLES[code]: int((table["role"] == code).sum()) for code in range(len(ROLES))}
    folds = {str(k): int(((table["role"] == 0) & (table["fold"] == k)).sum()) for k in range(5)}
    if counts["TRAIN"] != sum(folds.values()):
        raise ValueError("TRAIN engagements without a fold")
    truncation = {f"h{h}": {name: int(((table[f"truncation_bits_h{h}"] >> bit) & 1)[table[f"valid_h{h}"]].sum())
                             for bit, name in enumerate(TRUNCATION)} for h in HORIZONS}

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "ENGAGEMENTS.npz"
    pending = output / f".ENGAGEMENTS.npz.{os.getpid()}.pending"
    with pending.open("wb") as stream:
        np.savez(stream, **table)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, target)
    record = {
        "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_contract_sha256": sha256_file(LABEL_CONTRACT), "code_sha256": sha256_file(Path(__file__)),
        "inputs": {"bank_field_manifest_sha256": sha256_file(Path(bank) / "FIELD_MANIFEST.json"),
                   "bank_shards_sha256": shard_hashes,
                   "projection": {name: sha256_file(Path(projection) / name) for name in ("MATCH_TABLE.npy", "MATCH_QUERIES.npy", "QUERY_TABLE.npy")}},
        "rows": int(len(keys)), "role_counts": counts, "train_fold_counts": folds,
        "valid_counts": {f"h{h}": int(table[f"valid_h{h}"].sum()) for h in HORIZONS},
        "truncation_counts_valid_rows": truncation, "truncation_bit_order": list(TRUNCATION),
        "roles": list(ROLES), "columns": sorted(table), "engagements_sha256": sha256_file(target),
        "checks": ["primary&supported components only; non-primary carry no query refs", "pre = f - 10000 and global isolation verified",
                   "stored e_h equals the population endpoint rule for every valid horizon",
                   "every query index matches match, local ref and time in QUERY_TABLE and in the bank",
                   "h validity agrees with the presence of its query ref", "engagement keys unique",
                   "every row has pre and h90 queries", "bank covers every projection match"],
    }
    pending_manifest = output / f".ENGAGEMENTS_MANIFEST.json.{os.getpid()}.pending"
    pending_manifest.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending_manifest, output / "ENGAGEMENTS_MANIFEST.json")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    record = build(Path(args.output))
    print(json.dumps({k: record[k] for k in ("rows", "role_counts", "train_fold_counts", "valid_counts")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
