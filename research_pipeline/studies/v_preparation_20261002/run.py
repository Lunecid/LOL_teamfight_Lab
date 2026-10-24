"""Prepare immutable V-only row references and match-context membership."""
from __future__ import annotations

import json
import hashlib
import os
import platform
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from studies.observable_state_20261002.bank import match_view
from studies.observable_state_20261002.run import bindings as observable_bindings
from studies.observable_state_20261002.source import EXCLUDED_KEY, ROOT, file_sha256, load_frozen_entries
from studies.observable_state_20261002.sampling import query_support, SAMPLER_FORMULA_SHA256
from studies.training_runtime_20261002.preflight import _atomic_write_json
from .partition import assigned_role, context_membership, inner_stop, nominal_reference_rows, train_fold

REPORT = ROOT / "reports/v_preparation_20261002"
ROLES = ("TRAIN", "V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "TEST")
SCOPES = ("fit", "stop", "refit", "dev", "cal", "target")
PATCH_ROLES = ("TRAIN", "VALIDATION", "TEST")
SUPPORT_REASONS = ("frame_clock_not_integer_vector", "empty_frame_clock",
                   "frame_clock_not_nonnegative_strictly_increasing", "negative_query",
                   "no_preceding_frame", "query_after_last_supported_frame")
MATCH_DTYPE = np.dtype([("key", "S64"), ("shard", "u2"), ("local_match", "u2"),
                       ("original_index", "u4"), ("patch_role", "u1"),
                       ("role", "u1"), ("fold", "i1"), ("stop", "?"), ("target_Blue_win", "?"),
                       ("first_nominal_row", "i8"), ("n_nominal_rows", "i4")])
ROW_DTYPE = np.dtype([("match_index", "u4"), ("shard", "u2"), ("local_match", "u2"),
                     ("nominal_index", "u4"), ("bucket_id", "i4"), ("bucket_lo_ms", "i4"),
                     ("bucket_hi_ms", "i4"), ("nominal_ms", "i4"), ("actual_ms", "i4"), ("query_ref", "i4"),
                     ("frame_index", "i4"), ("event_limit", "i4"), ("supported", "?"), ("support_reason_mask", "u2")])
SOURCE_FIELDS = ("v_nominal", "query_ms", "query_frame_index", "query_event_limit",
                 "query_item_ledger_known", "minute_ts", "event_timestamp", "end_ms", "winning_team")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def code_bindings() -> dict:
    result = observable_bindings()
    paths = list(Path(__file__).parent.glob("*.py")) + [REPORT / "CONTRACT.json",
             REPORT / "runtime_preflight_1/manifest.json", ROOT / "reports/observable_state_20261002/ACCEPTANCE.json"]
    for path in paths:
        result[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    return result


def verify_nominal_sampler(entry: dict, bank: dict, refs: dict) -> np.ndarray:
    """Regenerate complete buckets from the original local ID, never the key."""
    original_id, end = entry["match_id"], int(bank["end_ms"])
    expected = []
    masks = []
    for bucket, lo in enumerate(range(120000, end, 120000)):
        hi = min(lo + 119999, end - 1)
        h = int.from_bytes(hashlib.sha256(f"{original_id}:{bucket}".encode()).digest()[:8], "big")
        nominal = lo + h % (hi - lo + 1)
        frame, reasons = query_support(bank["minute_ts"], nominal)
        expected.append((bucket, lo, hi, nominal, nominal if not reasons else -1,
                         frame if frame is not None else -1))
        masks.append(sum(1 << SUPPORT_REASONS.index(reason) for reason in reasons))
    if not np.array_equal(np.asarray(expected, dtype=np.int64).reshape(-1, 6), bank["v_nominal"][:, :6]):
        raise ValueError("nominal sampler differs from original ID or complete bucket coverage")
    masks = np.asarray(masks, dtype=np.uint16)
    if not np.array_equal(masks == 0, refs["supported"]):
        raise ValueError("nominal support reason differs from retained support")
    return masks


def update_contexts(matrix: np.ndarray, index: int, patch: str, key: str, rows: int,
                    supported: int, winner: int, counters: dict) -> None:
    for context in range(6):
        name = "final" if context == 0 else f"oof{context - 1}"
        flags = context_membership(patch, key, excluded_fold=None if context == 0 else context - 1)
        mask = sum(1 << bit for bit, scope in enumerate(SCOPES) if flags[scope])
        matrix[context, index] = mask
        for scope in SCOPES:
            if flags[scope]:
                counters[name].update({f"{scope}:matches": 1, f"{scope}:nominal_rows": rows,
                                      f"{scope}:supported_rows": supported,
                                      f"{scope}:class{winner}_supported_rows": supported})
        if patch == "15.14" and context and train_fold(key) == context - 1 and mask != 32:
            raise ValueError("own OOF match admitted to another learning scope")
        if patch != "15.14" and mask & 7:
            raise ValueError("non-TRAIN match admitted to FIT/STOP/refit")


def verify_saved(output: Path, entries: dict, expected_rows: int, expected_row_digest: str | None = None) -> dict:
    matches = np.load(output / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    rows = np.load(output / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    contexts = np.load(output / "CONTEXT_MEMBERSHIP.npy", mmap_mode="r", allow_pickle=False)
    try:
        if len(matches) != len(entries) or len(rows) != expected_rows or contexts.shape != (6, len(matches)):
            raise ValueError("saved preparation coverage changed")
        keys = [value.decode("ascii") for value in matches["key"]]
        if len(set(keys)) != len(keys) or set(keys) != set(entries) or EXCLUDED_KEY in keys:
            raise ValueError("saved match coverage or exclusion changed")
        for index, key in enumerate(keys):
            patch = entries[key]["patch"]
            row = matches[index]
            if (int(row["original_index"]) != entries[key]["original_index"] or
                    int(row["patch_role"]) != PATCH_ROLES.index(entries[key]["role"]) or
                    int(row["role"]) != ROLES.index(assigned_role(patch, key)) or
                    int(row["fold"]) != (train_fold(key) if patch == "15.14" else -1) or
                    bool(row["stop"]) != (inner_stop(key) if patch == "15.14" else False) or
                    bool(row["target_Blue_win"]) != (entries[key]["audited_numbers"]["game_end_winning_team"] == 100)):
                raise ValueError("saved role, fold or target binding mismatch")
            for context in range(6):
                flags = context_membership(patch, key, excluded_fold=None if context == 0 else context - 1)
                if int(contexts[context, index]) != sum(1 << bit for bit, scope in enumerate(SCOPES) if flags[scope]):
                    raise ValueError("saved OOF dependency membership mismatch")
        if np.any(rows["match_index"] >= len(matches)):
            raise ValueError("nominal row outside fixed match table")
        per_match = np.bincount(rows["match_index"].astype(np.int64), minlength=len(matches))
        if not np.array_equal(per_match, matches["n_nominal_rows"]):
            raise ValueError("nominal rows dropped, duplicated or reassigned")
        row_digest = hashlib.sha256()
        for start in range(0, len(rows), 65536):
            part = rows[start:start + 65536]
            row_digest.update(part.tobytes())
            owner = matches[part["match_index"]]
            if (np.any(part["shard"] != owner["shard"]) or np.any(part["local_match"] != owner["local_match"]) or
                    np.any(part["nominal_index"] != np.arange(start, start + len(part)) - owner["first_nominal_row"])):
                raise ValueError("nominal source reference identity changed")
            if (np.any(part["supported"] != (part["query_ref"] >= 0)) or
                    np.any(part["supported"] != (part["support_reason_mask"] == 0)) or
                    np.any(part["actual_ms"][part["supported"]] != part["nominal_ms"][part["supported"]]) or
                    np.any(part["actual_ms"][~part["supported"]] != -1) or
                    np.any(part["query_ref"][~part["supported"]] != -1) or
                    np.any(part["frame_index"][~part["supported"]] != -1) or
                    np.any(part["event_limit"][~part["supported"]] != -1)):
                raise ValueError("unsupported nominal state was converted into a valid input")
        if expected_row_digest is not None and row_digest.hexdigest() != expected_row_digest:
            raise ValueError("reloaded nominal rows differ from validated source references")
        return {"complete_saved_match_and_nominal_coverage": True, "whole_match_context_membership": True,
                "saved_reference_bytes_match_validated_source": expected_row_digest is not None,
                "validated_reference_row_sha256": row_digest.hexdigest(),
                "actual_model_consumption_verified": False, "n_matches": len(matches), "n_nominal_rows": len(rows)}
    finally:
        for value in (matches, rows, contexts):
            value._mmap.close()


def run(output: Path) -> dict:
    source_acceptance_path = ROOT / "reports/observable_state_20261002/ACCEPTANCE.json"
    accepted = read(source_acceptance_path)
    source = Path(accepted["accepted_run"])
    original = read(source / "RUN_MANIFEST.json")
    if not accepted["phase_accepted"] or not original["complete"] or not original["accepted"]:
        raise ValueError("source phase was not accepted")
    if file_sha256(source / "RUN_MANIFEST.json") != accepted["run_manifest_sha256"]:
        raise ValueError("accepted source manifest changed")
    source_contracts = {name: file_sha256(source / name) for name in
                        ("FIELD_MANIFEST.json", "FIELD_SUPPORT_COUNTS.json")}
    if not read(REPORT / "runtime_preflight_1/manifest.json")["all_ok"]:
        raise ValueError("CPU/GPU runtime preflight failed")
    entries, cohort = load_frozen_entries()
    expected = {entry["canonical_match_key"]: entry for entry in entries}
    if cohort != original["cohort"]:
        raise ValueError("source cohort and current fixed cohort differ")
    n_rows = accepted["counts"]["n_nominal_V_queries"]
    projected = len(entries) * MATCH_DTYPE.itemsize + n_rows * ROW_DTYPE.itemsize + 6 * len(entries)
    if shutil.disk_usage(ROOT).free < projected + (2 << 30):
        raise OSError("reference preparation violates 2GiB disk reserve")
    output.mkdir(parents=True, exist_ok=False)
    binding = code_bindings()
    manifest = {"complete": False, "accepted_preparation": False, "training_ready": False,
                "new_models_fitted": False, "new_SVI_labels_produced": False, "source": str(source),
                "bindings": binding, "cohort": cohort, "source_acceptance_sha256": file_sha256(source_acceptance_path),
                "source_field_contract_hashes": source_contracts,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "CPU_budget": 4,
                                "preparation_threads": 1, "GPU_used": False,
                                "numeric_threads": {name: os.environ.get(name) for name in
                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}}
    matches = rows = contexts = None
    start = time.perf_counter()
    counts, role_counts, seen, links = Counter(), {role: Counter() for role in ROLES}, set(), []
    context_counts = {"final": Counter(), **{f"oof{k}": Counter() for k in range(5)}}
    row_digest = hashlib.sha256()
    try:
        matches = np.lib.format.open_memmap(output / "MATCH_TABLE.npy", mode="w+", dtype=MATCH_DTYPE, shape=(len(entries),))
        rows = np.lib.format.open_memmap(output / "V_ROWS.npy", mode="w+", dtype=ROW_DTYPE, shape=(n_rows,))
        contexts = np.lib.format.open_memmap(output / "CONTEXT_MEMBERSHIP.npy", mode="w+", dtype="u1", shape=(6, len(entries)))
        match_cursor = row_cursor = 0
        records = sorted(accepted["shard_artifact_hashes"], key=lambda value: value["directory"])
        for shard_index, record in enumerate(records):
            directory = source / record["directory"]
            for filename, key in (("BANK.npz", "archive_sha256"), ("SHARD_MANIFEST.json", "shard_manifest_sha256"),
                                  ("MATCH_LEDGER.jsonl", "match_ledger_sha256")):
                if file_sha256(directory / filename) != record[key]:
                    raise ValueError("accepted observable shard changed")
            layout = read(directory / "LAYOUT.json")
            local_layout = {"scalar_fields": [key for key in layout["scalar_fields"] if key in SOURCE_FIELDS],
                            "match_fields": list(SOURCE_FIELDS)}
            ledgers = [json.loads(line) for line in (directory / "MATCH_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
            with np.load(directory / "BANK.npz", allow_pickle=False) as archive:
                selected = {key: archive[key] for name in SOURCE_FIELDS for key in (name, "offset_" + name)}
            for local, ledger in enumerate(ledgers):
                key = ledger["canonical_match_key"]
                if key not in expected or key in seen or key == EXCLUDED_KEY:
                    raise ValueError("duplicate, foreign or excluded source match")
                entry = expected[key]
                if ledger["source_hashes"] != entry["source_hashes"] or ledger["patch"] != entry["patch"] or ledger["role"] != entry["role"]:
                    raise ValueError("match source or patch role changed")
                bank = match_view(selected, local_layout, local)
                if int(bank["end_ms"]) != entry["audited_numbers"]["game_end_ms"] or int(bank["winning_team"]) != entry["audited_numbers"]["game_end_winning_team"]:
                    raise ValueError("verified target or terminal changed")
                refs = nominal_reference_rows(bank)
                reason_masks = verify_nominal_sampler(entry, bank, refs)
                size, supported = len(refs["nominal_ms"]), int(refs["supported"].sum())
                item_known = bank["query_item_ledger_known"]
                if item_known.dtype != np.dtype(bool) or item_known.shape != bank["query_ms"].shape:
                    raise ValueError("item replay support query coverage changed")
                nominal_known = int(item_known[refs["query_ref"][refs["supported"]]].sum())
                if size != ledger["counts"]["n_nominal_V_queries"] or supported != ledger["counts"]["n_supported_V_queries"]:
                    raise ValueError("nominal-only V reference cardinality mismatch")
                patch, role = entry["patch"], assigned_role(entry["patch"], key)
                winner = int(int(bank["winning_team"]) == 100)
                matches[match_cursor] = (key.encode("ascii"), shard_index, local, entry["original_index"],
                                         PATCH_ROLES.index(entry["role"]), ROLES.index(role),
                                         train_fold(key) if patch == "15.14" else -1,
                                         inner_stop(key) if patch == "15.14" else False, winner, row_cursor, size)
                target = rows[row_cursor:row_cursor + size]
                for name, value in (("match_index", match_cursor), ("shard", shard_index), ("local_match", local)):
                    target[name] = value
                target["nominal_index"] = np.arange(size, dtype=np.uint32)
                for name in ("bucket_id", "bucket_lo_ms", "bucket_hi_ms", "nominal_ms", "actual_ms",
                             "query_ref", "frame_index", "event_limit", "supported"):
                    target[name] = refs[name]
                    if not np.array_equal(target[name], refs[name]):
                        raise ValueError("reference dtype cast lost source values")
                target["support_reason_mask"] = reason_masks
                row_digest.update(target.tobytes())
                update_contexts(contexts, match_cursor, patch, key, size, supported, winner, context_counts)
                role_counts[role].update(n_matches=1, n_nominal_rows=size, n_supported_rows=supported,
                                        n_nominal_item_ledger_known=nominal_known,
                                        **{f"class{winner}_supported_rows": supported})
                counts.update(n_matches=1, n_nominal_rows=size, n_supported_rows=supported,
                              n_all_source_queries=len(item_known), n_all_item_ledger_known=int(item_known.sum()),
                              n_nominal_item_ledger_known=nominal_known)
                seen.add(key)
                row_cursor += size
                match_cursor += 1
            links.append({**record, "layout_sha256": file_sha256(directory / "LAYOUT.json")})
            manifest["counts"] = dict(counts)
            _atomic_write_json(output / "RUN_MANIFEST.json", manifest)
            print(json.dumps({"phase": "V_reference_preparation", "matches": match_cursor, "nominal_rows": row_cursor}), flush=True)
        if seen != set(expected) or row_cursor != n_rows or counts["n_supported_rows"] != accepted["counts"]["n_supported_V_queries"]:
            raise ValueError("full nominal V preparation omitted source matches or rows")
        del target
        for value in (matches, rows, contexts):
            value.flush()
            value._mmap.close()
        matches = rows = contexts = None
        verified = verify_saved(output, expected, n_rows, expected_row_digest=row_digest.hexdigest())
        for name, values in context_counts.items():
            for scope in ("fit", "stop", "refit", "dev", "cal"):
                if not values[f"{scope}:class0_supported_rows"] or not values[f"{scope}:class1_supported_rows"]:
                    raise ValueError(f"{name}/{scope} lacks both outcome classes")
        if code_bindings() != binding:
            raise ValueError("code or accepted input definitions changed during preparation")
        _, final_cohort = load_frozen_entries()
        if cohort != final_cohort:
            raise ValueError("fixed cohort changed during preparation")
        if source_contracts != {name: file_sha256(source / name) for name in source_contracts}:
            raise ValueError("accepted source field definitions changed during preparation")
        _atomic_write_json(output / "SOURCE_SHARD_LINKS.json", {"source": str(source), "shards": links})
        _atomic_write_json(output / "SPLIT_AND_CONTEXT_COUNTS.json", {"roles": {k: dict(v) for k, v in role_counts.items()},
                          "contexts": {k: dict(v) for k, v in context_counts.items()}, "role_rebalancing": False})
        verified.update(original_id_sampler_regenerated_all_rows=True,
                        sampler_formula_sha256=SAMPLER_FORMULA_SHA256,
                        support_reason_bits={str(bit): reason for bit, reason in enumerate(SUPPORT_REASONS)})
        _atomic_write_json(output / "INDEPENDENT_CHECKS.json", verified)
        manifest.update(complete=True, accepted_preparation=True, elapsed_seconds=time.perf_counter() - start,
                        completed_utc=datetime.now(timezone.utc).isoformat(),
                        saved_files={name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size} for name in
                        ("MATCH_TABLE.npy", "V_ROWS.npy", "CONTEXT_MEMBERSHIP.npy", "SOURCE_SHARD_LINKS.json", "SPLIT_AND_CONTEXT_COUNTS.json", "INDEPENDENT_CHECKS.json")})
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for value in (matches, rows, contexts):
            if value is not None:
                value.flush()
                value._mmap.close()
        _atomic_write_json(output / "RUN_MANIFEST.json", manifest)
    return manifest
