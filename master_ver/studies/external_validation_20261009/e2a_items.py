"""E2a stage 'items': recipient-confirmed observed item-node bank (the item ledger the V projection depends on).

README
  Transcription of studies/item_node_observed_20261002/audit.run_bank (frozen) without its three 2025-specific parts:
    - the acceptance gates (_accepted_source / verify_preparation_bindings require the 2025 certificates),
    - CPU pinning (reader on CPU 0, replay workers on CPUs 2/4/6; affinity never changes a value),
    - vocabulary fitting (FIT / REFIT_TRAIN scopes are empty for EXTERNAL matches; external rows are encoded with the
      frozen V/q vocabularies downstream, so unseen item IDs stay raw here and map to UNK there).
  Unchanged frozen pieces: item_node_encoding_20261002.source.AcceptedSourceReader (source shard reader),
  audit.PreparationIndex (V preparation references), audit._source_task (= observe_match + _compact_for_writer),
  audit._BankWriter (every persisted array), audit._merge_stats / _jsonable / _saved_file_records,
  audit.independent_reload_check, bank.FORMAT_VERSION / REASON_BITS.  Replay runs in <= 6 spawned processes with a
  bounded in-flight queue; results are written strictly in source order, as in the frozen run.

Outputs (stage_dir(dataset,'items')): the frozen file set (MATCH_TABLE, QUERY_TABLE, NODE_OFFSETS, NODE_SUPPORT,
NODE_REASON_MASK, NOMINAL_TABLE, CONTEXT_MEMBERSHIP, ITEM_IDS.bin, ITEM_COUNTS.bin, EVENT_SOURCE_INDEX_*.bin,
MATCH_STATS.jsonl, PROGRESS.jsonl, SUMMARY.json, SOURCE_BINDINGS.json, PREPARATION_BINDINGS.json, BANK_METADATA.json,
RUN_MANIFEST.json) minus the VOCABULARY_* files.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402

INFLIGHT_PER_WORKER = 4


def _init_worker() -> None:
    C.runtime_hygiene()


def source_bindings(source_root: Path, preparation_root: Path) -> dict:
    """audit.verify_source_bindings without the 2025 acceptance certificate (all five shard hashes re-read)."""
    from studies.item_node_encoding_20261002.source import AcceptedSourceReader
    from studies.item_node_observed_20261002.audit import _json_digest, _required_source_links

    links = _required_source_links(preparation_root)
    reader = AcceptedSourceReader(source_root, preparation_root, verify_hashes=True)
    verified = reader.verify_source_bindings()
    records = verified["records"]
    link_map = {row["directory"]: row for row in links}
    for record in records:
        link = link_map[record["directory"]]
        for name in ("archive_sha256", "layout_sha256", "shard_manifest_sha256", "match_ledger_sha256", "static_repair_sha256"):
            if record[name] != link[name]:
                raise ValueError(f"source shard {record['directory']} {name} differs from SOURCE_SHARD_LINKS")
    return {"source_root": str(source_root), "n_shards": len(records), "records": records,
            "records_sha256": _json_digest(records),
            "acceptance": {"kind": "E2a external bank", "bank_manifest_sha256": C.sha256_file(source_root / "RUN_MANIFEST.json")}}


def preparation_bindings(preparation_root: Path) -> dict:
    """audit.verify_preparation_bindings without the 2025 certificate; context masks must all be 0."""
    import numpy as np
    from studies.item_node_observed_20261002.audit import PreparationIndex

    manifest_path = preparation_root / "RUN_MANIFEST.json"
    manifest = C.read_json(manifest_path)
    if manifest.get("complete") is not True or manifest.get("accepted_preparation") is not True:
        raise ValueError("preparation manifest is not complete")
    records = {}
    for name in ("RUN_MANIFEST.json", "SOURCE_SHARD_LINKS.json", "MATCH_TABLE.npy", "V_ROWS.npy", "CONTEXT_MEMBERSHIP.npy"):
        digest = C.sha256_file(preparation_root / name)
        if name != "RUN_MANIFEST.json" and digest != manifest["saved_files"][name]["sha256"]:
            raise ValueError(f"preparation saved hash differs: {name}")
        records[name] = {"sha256": digest, "bytes": (preparation_root / name).stat().st_size}
    index = PreparationIndex(preparation_root, strict_full=False)
    contexts = np.asarray(index.contexts)
    if np.any(contexts != 0):
        raise ValueError("EXTERNAL preparation has a nonzero context membership")
    return {"acceptance_path": None, "manifest_sha256": records["RUN_MANIFEST.json"]["sha256"], "files": records,
            "metadata": index.metadata(), "exact_context_membership": True,
            "n_match_context_cells": int(contexts.size), "all_context_masks_zero": True}


def run_items(dataset: str, workers: int, log) -> dict:
    C.runtime_hygiene()
    import numpy as np
    from studies.item_node_encoding_20261002.source import AcceptedSourceReader
    from studies.item_node_observed_20261002.audit import (PARTICIPANTS, PreparationIndex, _BankWriter, _code_bindings,
                                                           _decode_key, _jsonable, _merge_stats, _saved_file_records,
                                                           _source_task, independent_reload_check)
    from studies.item_node_observed_20261002.bank import FORMAT_VERSION, REASON_BITS, UNASSIGNED_REASON_BITS

    source_root, preparation_root = C.stage_dir(dataset, "bank"), C.stage_dir(dataset, "preparation")
    output = C.stage_dir(dataset, "items")
    if (output / "RUN_MANIFEST.json").exists() and C.read_json(output / "RUN_MANIFEST.json").get("complete"):
        log(f"[items] {dataset}: complete item bank exists, reused")
        return C.read_json(output / "RUN_MANIFEST.json")
    if output.exists():
        shutil.rmtree(output)
    source_before = source_bindings(source_root, preparation_root)
    preparation_before = preparation_bindings(preparation_root)
    index = PreparationIndex(preparation_root, strict_full=False)
    n_matches = len(index.match_table)
    n_nominal = len(index.v_rows)
    n_queries = 0
    for directory in sorted(row["directory"] for row in source_before["records"]):
        for line in (source_root / directory / "MATCH_LEDGER.jsonl").read_text(encoding="utf-8").splitlines():
            n_queries += int(json.loads(line)["counts"]["n_unique_queries"])
    output.mkdir(parents=True, exist_ok=False)
    started_perf = time.perf_counter()
    manifest = {
        "format_version": FORMAT_VERSION, "complete": False, "accepted_bank": False, "full_corpus_verified": False,
        "training_ready": False, "new_models_fitted": False, "new_SVI_labels_produced": False,
        "actual_inventory_complete": "unknown", "dataset": dataset, "role": C.EXTERNAL_ROLE,
        "source_root": str(source_root), "preparation_root": str(preparation_root), "started_utc": C.utc_now(),
        "environment": {"numpy": np.__version__, "reader_writer_processes": 1, "replay_workers": int(workers),
                        "numeric_threads": 1, "bounded_inflight": INFLIGHT_PER_WORKER * int(workers), "GPU_used": False,
                        "cpu_pinning": False, "priority": "below_normal"},
        "code_bindings_before": _code_bindings(),
        "build_gates": {"verify_hashes": True, "strict_full": False, "full_gate": False,
                        "capacities": [n_matches, n_queries, n_nominal]},
        "source_bindings_before": source_before, "preparation_bindings_before": preparation_before,
        "hook": C.FROZEN_HOOKS["items"],
    }
    (output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    aggregate = {
        "n_matches": 0, "n_unique_queries": 0, "n_nominal_rows": 0, "n_item_events": 0,
        "n_assigned_events": 0, "n_unassigned_events": 0, "n_supported_player_queries": 0,
        "n_unassigned_observed_queries": 0, "support_by_participant": np.zeros(PARTICIPANTS, dtype=np.int64),
        "reason_counts": Counter(), "unassigned_reason_counts": Counter(), "item_event_counts": Counter(),
        "item_id_event_counts": Counter(), "by_role": {}, "by_patch": {}, "worker_affinity_masks": Counter(),
    }
    writer = _BankWriter(output, capacities=(n_matches, n_queries, n_nominal))
    reader = AcceptedSourceReader(source_root, preparation_root, verify_hashes=False)

    def tasks():
        for match_index, match in enumerate(reader.iter_matches()):
            prep_index, nominal_rows, context_masks = index.rows_for(match.shard_index, match.local_index)
            if len(context_masks) != 6:
                raise ValueError("observed bank requires all six context memberships")
            if match.canonical_match_key != _decode_key(index.match_table[prep_index]["key"]):
                raise ValueError("source canonical match key differs from V preparation")
            if prep_index != match_index:
                raise ValueError("source order differs from V preparation order")
            yield match_index, match, nominal_rows, context_masks

    try:
        bound = INFLIGHT_PER_WORKER * max(1, int(workers))
        with ProcessPoolExecutor(max_workers=max(1, int(workers)), initializer=_init_worker) as pool:
            pending = deque()
            iterator = iter(tasks())
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < bound:
                    try:
                        match_index, match, nominal_rows, context_masks = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append((match_index, nominal_rows, pool.submit(_source_task, match, nominal_rows, context_masks)))
                if pending:
                    match_index, nominal_rows, future = pending.popleft()
                    observed = future.result()
                    writer.write(match_index, observed, nominal_rows)
                    if writer._match_count % 128 == 0:
                        writer.write_progress()
                    if writer._match_count % 5000 == 0:
                        log(f"[items] {dataset}: {writer._match_count}/{n_matches} matches")
                    _merge_stats(aggregate, observed)
        writer_counts = writer.finalize()
    except BaseException as exc:
        for name in ("_ids", "_counts", "_assigned_sources", "_unassigned_sources", "_match_log", "_progress"):
            handle = getattr(writer, name, None)
            if handle is not None and not handle.closed:
                handle.close()
        manifest.update(error=f"{type(exc).__name__}: {exc}")
        (output / "FAILED.json").write_text(json.dumps({"failed": True, "error": manifest["error"]}) + "\n", encoding="utf-8")
        raise
    aggregate["n_matches"] = writer_counts["n_matches"]
    aggregate["n_unique_queries"] = writer_counts["n_unique_queries"]
    aggregate["n_nominal_rows"] = writer_counts["n_nominal_rows"]
    if (aggregate["n_matches"], aggregate["n_unique_queries"], aggregate["n_nominal_rows"]) != (n_matches, n_queries, n_nominal):
        raise ValueError("observed bank coverage differs from the preparation and bank ledgers")
    for name, value in (("SOURCE_BINDINGS.json", source_before), ("PREPARATION_BINDINGS.json", preparation_before)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = _jsonable(aggregate)
    summary["complete_full_corpus_coverage"] = True
    summary["scope_membership"] = {}
    summary["vocabulary_scope_audit"] = None
    summary["vocabulary_note"] = "no vocabulary fitted for EXTERNAL rows; downstream uses the frozen V/q vocabularies"
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {
        "format_version": FORMAT_VERSION,
        "node_order": "participant IDs 1..10",
        "binary_members": {"ITEM_IDS.bin": "little-endian signed int64", "ITEM_COUNTS.bin": "little-endian signed int64"},
        "ragged_offsets": "NODE_OFFSETS.npy rows are per-query ten-node global element offsets",
        "support": "NODE_SUPPORT.npy is recipient-confirmed observed-ledger support; actual inventory completeness remains unknown",
        "reason_bits": {
            "assigned": {str(key): int(value) for key, value in sorted(REASON_BITS.items(), key=lambda row: row[1])},
            "unassigned": {str(key): int(value) for key, value in sorted(UNASSIGNED_REASON_BITS.items(), key=lambda row: row[1])},
        },
        "history_framing": "EVENT_SOURCE_INDEX_* are little-endian int32 source event indices; accepted source banks retain typed fields",
        "history_refs": "MATCH_TABLE first/n fields point into source-index arrays; QUERY_TABLE ends are lazy per-match prefix references",
        "outcome_fields_in_tensors": False,
        "source_root": str(source_root),
        "preparation_root": str(preparation_root),
        "vocabulary_fit_performed": False,
        "vocabulary_fit_models": False,
        "vocabulary_scope_metadata": None,
        "vocabularies": {},
        "dataset": dataset,
        "role": C.EXTERNAL_ROLE,
    }
    (output / "BANK_METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    reload = independent_reload_check(output)
    source_after = source_bindings(source_root, preparation_root)
    preparation_after = preparation_bindings(preparation_root)
    if source_after["records_sha256"] != source_before["records_sha256"] or preparation_after != preparation_before:
        raise ValueError("item bank inputs changed during the run")
    if _code_bindings() != manifest["code_bindings_before"]:
        raise ValueError("code bindings changed during observed bank run")
    manifest.update({
        "complete": True, "accepted_bank": False, "accepted_by_e2a_checks": True, "full_corpus_verified": False,
        "vocabulary_fit_performed": False, "vocabulary_fit_models": False,
        "counts": _jsonable({**writer_counts, "n_shards": source_before["n_shards"]}),
        "elapsed_seconds": round(time.perf_counter() - started_perf, 1), "completed_utc": C.utc_now(),
        "source_bindings_after": source_after, "preparation_bindings_after": preparation_after,
        "code_bindings_after": _code_bindings(), "independent_reload": reload,
        "saved_files": _saved_file_records(output), "independent_reload_required": True,
    })
    (output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"[items] {dataset}: {writer_counts} in {manifest['elapsed_seconds']} s")
    return manifest
