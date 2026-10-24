"""E2a stage 'preparation': V reference preparation (MATCH_TABLE, V_ROWS, CONTEXT_MEMBERSHIP) for an EXTERNAL bank.

README
  Transcription of studies/v_preparation_20261002/run.run (frozen; not callable here because it reads the accepted
  2025 bank through the acceptance certificate and assigns roles through partition.assigned_role, which accepts only
  15.14-15.16).  Unchanged frozen calls per match: partition.nominal_reference_rows (nominal V references, support and
  event limits) and run.verify_nominal_sampler (regenerates every nominal bucket from the original match id).
  MATCH_TABLE / V_ROWS dtypes are the frozen run.MATCH_DTYPE / ROW_DTYPE; rows are filled field by field in the same
  order.  EXTERNAL coding: patch_role 3, role 6, fold -1, stop False; CONTEXT_MEMBERSHIP is 0 in all six contexts
  (an external match is never in FIT, STOP, REFIT, DEV, CAL or an OOF target).  Shards are read in sorted directory
  order (shard index = position), as the frozen run does.

Outputs (stage_dir(dataset,'preparation')): MATCH_TABLE.npy, V_ROWS.npy, CONTEXT_MEMBERSHIP.npy,
SOURCE_SHARD_LINKS.json, SPLIT_AND_CONTEXT_COUNTS.json, INDEPENDENT_CHECKS.json, RUN_MANIFEST.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402

SHARD_FILES = (("BANK.npz", "archive_sha256"), ("SHARD_MANIFEST.json", "shard_manifest_sha256"),
               ("MATCH_LEDGER.jsonl", "match_ledger_sha256"), ("STATIC_REPAIR.json", "static_repair_sha256"),
               ("LAYOUT.json", "layout_sha256"))


def bank_shard_records(bank_root: Path) -> list[dict]:
    names = sorted(e.name for e in bank_root.iterdir() if e.is_dir() and (e / "BANK.npz").is_file())
    out = []
    for name in names:
        record = {"directory": name}
        for filename, field in SHARD_FILES:
            record[field] = C.sha256_file(bank_root / name / filename)
        out.append(record)
    return out


def verify_saved(output: Path, entries: dict, expected_rows: int, expected_row_digest: str) -> dict:
    """run.verify_saved with the EXTERNAL codes in place of assigned_role/train_fold/inner_stop."""
    import numpy as np
    from studies.v_preparation_20261002.run import ROW_DTYPE, MATCH_DTYPE

    matches = np.load(output / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    rows = np.load(output / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    contexts = np.load(output / "CONTEXT_MEMBERSHIP.npy", mmap_mode="r", allow_pickle=False)
    try:
        if matches.dtype != MATCH_DTYPE or rows.dtype != ROW_DTYPE:
            raise ValueError("saved preparation dtypes differ from the frozen dtypes")
        if len(matches) != len(entries) or len(rows) != expected_rows or contexts.shape != (6, len(matches)):
            raise ValueError("saved preparation coverage changed")
        keys = [value.decode("ascii") for value in matches["key"]]
        if len(set(keys)) != len(keys) or set(keys) != set(entries):
            raise ValueError("saved match coverage changed")
        for index, key in enumerate(keys):
            row = matches[index]
            if (int(row["original_index"]) != entries[key]["original_index"] or
                    int(row["patch_role"]) != C.EXTERNAL_PATCH_ROLE_CODE or int(row["role"]) != C.EXTERNAL_ROLE_CODE or
                    int(row["fold"]) != -1 or bool(row["stop"]) or
                    bool(row["target_Blue_win"]) != (entries[key]["audited_numbers"]["game_end_winning_team"] == 100)):
                raise ValueError("saved role, fold or target binding mismatch")
        if np.any(contexts != 0):
            raise ValueError("an EXTERNAL match entered a learning context")
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
        if row_digest.hexdigest() != expected_row_digest:
            raise ValueError("reloaded nominal rows differ from validated source references")
        return {"complete_saved_match_and_nominal_coverage": True, "whole_match_context_membership": True,
                "saved_reference_bytes_match_validated_source": True,
                "validated_reference_row_sha256": row_digest.hexdigest(),
                "actual_model_consumption_verified": False, "n_matches": len(matches), "n_nominal_rows": len(rows)}
    finally:
        for value in (matches, rows, contexts):
            value._mmap.close()


def run_preparation(dataset: str, log) -> dict:
    C.runtime_hygiene()
    import numpy as np
    from studies.observable_state_20261002.bank import match_view
    from studies.observable_state_20261002.sampling import SAMPLER_FORMULA_SHA256
    from studies.training_runtime_20261002.preflight import _atomic_write_json
    from studies.v_preparation_20261002.partition import nominal_reference_rows
    from studies.v_preparation_20261002.run import (MATCH_DTYPE, ROW_DTYPE, SOURCE_FIELDS, SUPPORT_REASONS,
                                                     verify_nominal_sampler)
    from studies.external_validation_20261009.e2a_audit import load_external_entries

    bank_root, output = C.stage_dir(dataset, "bank"), C.stage_dir(dataset, "preparation")
    if (output / "RUN_MANIFEST.json").exists() and C.read_json(output / "RUN_MANIFEST.json").get("complete"):
        log(f"[preparation] {dataset}: complete preparation exists, reused")
        return C.read_json(output / "RUN_MANIFEST.json")
    bank_manifest = C.read_json(bank_root / "RUN_MANIFEST.json")
    if not bank_manifest.get("complete"):
        raise ValueError("bank is incomplete")
    all_entries, _excluded, binding = load_external_entries(dataset)
    by_key = {e["canonical_match_key"]: e for e in all_entries}
    records = bank_shard_records(bank_root)
    ledgers_by_shard = {r["directory"]: [json.loads(line) for line in (bank_root / r["directory"] / "MATCH_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
                        for r in records}
    expected = {json_row["canonical_match_key"]: by_key[json_row["canonical_match_key"]]
                for rows in ledgers_by_shard.values() for json_row in rows}
    n_matches = len(expected)
    n_rows = sum(int(row["counts"]["n_nominal_V_queries"]) for rows in ledgers_by_shard.values() for row in rows)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"complete": False, "accepted_preparation": False, "training_ready": False, "new_models_fitted": False,
                "new_SVI_labels_produced": False, "source": str(bank_root), "dataset": dataset, "role": C.EXTERNAL_ROLE,
                "entries_binding": binding, "bank_manifest_sha256": C.sha256_file(bank_root / "RUN_MANIFEST.json"),
                "source_field_contract_hashes": {name: C.sha256_file(bank_root / name) for name in ("FIELD_MANIFEST.json", "FIELD_SUPPORT_COUNTS.json")},
                "started_utc": C.utc_now(), "external_codes": {"patch_role": C.EXTERNAL_PATCH_ROLE_CODE, "role": C.EXTERNAL_ROLE_CODE,
                                                                 "fold": -1, "stop": False, "context_membership": 0},
                "hook": C.FROZEN_HOOKS["preparation"]}
    counts, seen, links = Counter(), set(), []
    row_digest = hashlib.sha256()
    matches = np.lib.format.open_memmap(output / "MATCH_TABLE.npy", mode="w+", dtype=MATCH_DTYPE, shape=(n_matches,))
    rows = np.lib.format.open_memmap(output / "V_ROWS.npy", mode="w+", dtype=ROW_DTYPE, shape=(n_rows,))
    contexts = np.lib.format.open_memmap(output / "CONTEXT_MEMBERSHIP.npy", mode="w+", dtype="u1", shape=(6, n_matches))
    try:
        contexts[:] = 0
        match_cursor = row_cursor = 0
        for shard_index, record in enumerate(records):
            directory = bank_root / record["directory"]
            layout = json.loads((directory / "LAYOUT.json").read_text(encoding="utf-8"))
            local_layout = {"scalar_fields": [key for key in layout["scalar_fields"] if key in SOURCE_FIELDS],
                            "match_fields": list(SOURCE_FIELDS)}
            ledgers = ledgers_by_shard[record["directory"]]
            with np.load(directory / "BANK.npz", allow_pickle=False) as archive:
                selected = {key: archive[key] for name in SOURCE_FIELDS for key in (name, "offset_" + name)}
            for local, ledger in enumerate(ledgers):
                key = ledger["canonical_match_key"]
                if key not in expected or key in seen:
                    raise ValueError("duplicate or foreign source match")
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
                winner = int(int(bank["winning_team"]) == 100)
                matches[match_cursor] = (key.encode("ascii"), shard_index, local, entry["original_index"],
                                         C.EXTERNAL_PATCH_ROLE_CODE, C.EXTERNAL_ROLE_CODE, -1, False, winner, row_cursor, size)
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
                counts.update(n_matches=1, n_nominal_rows=size, n_supported_rows=supported,
                              n_all_source_queries=len(item_known), n_all_item_ledger_known=int(item_known.sum()),
                              n_nominal_item_ledger_known=nominal_known,
                              **{f"class{winner}_supported_rows": supported})
                seen.add(key)
                row_cursor += size
                match_cursor += 1
            links.append(dict(record))
        if seen != set(expected) or row_cursor != n_rows or match_cursor != n_matches:
            raise ValueError("nominal V preparation omitted source matches or rows")
        del target
    finally:
        for value in (matches, rows, contexts):
            value.flush()
            value._mmap.close()
    verified = verify_saved(output, expected, n_rows, row_digest.hexdigest())
    _atomic_write_json(output / "SOURCE_SHARD_LINKS.json", {"source": str(bank_root), "shards": links})
    _atomic_write_json(output / "SPLIT_AND_CONTEXT_COUNTS.json", {"roles": {C.EXTERNAL_ROLE: dict(counts)},
                       "contexts": {name: {} for name in ("final", "oof0", "oof1", "oof2", "oof3", "oof4")},
                       "role_rebalancing": False})
    verified.update(original_id_sampler_regenerated_all_rows=True, sampler_formula_sha256=SAMPLER_FORMULA_SHA256,
                    support_reason_bits={str(bit): reason for bit, reason in enumerate(SUPPORT_REASONS)})
    _atomic_write_json(output / "INDEPENDENT_CHECKS.json", verified)
    manifest.update(complete=True, accepted_preparation=True, counts=dict(counts),
                    elapsed_seconds=round(time.perf_counter() - started, 2), completed_utc=C.utc_now(),
                    saved_files={name: {"sha256": C.sha256_file(output / name), "bytes": (output / name).stat().st_size} for name in
                                 ("MATCH_TABLE.npy", "V_ROWS.npy", "CONTEXT_MEMBERSHIP.npy", "SOURCE_SHARD_LINKS.json",
                                  "SPLIT_AND_CONTEXT_COUNTS.json", "INDEPENDENT_CHECKS.json")})
    _atomic_write_json(output / "RUN_MANIFEST.json", manifest)
    log(f"[preparation] {dataset}: {n_matches} matches, {n_rows} nominal rows ({counts['n_supported_rows']} supported) "
        f"in {manifest['elapsed_seconds']} s")
    return manifest
