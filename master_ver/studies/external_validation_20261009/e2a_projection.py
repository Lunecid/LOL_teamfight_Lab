"""E2a stage 'projection': V input raw projection (QUERY_TABLE, MATCH_QUERIES, feature blocks) for an EXTERNAL dataset.

README
  Transcription of studies/v_input_projection_20261002/run.build (frozen) with two changes only:
    - input_bindings() requires the 2025 acceptance certificates; it is replaced by the E2a bindings of the dataset's
      bank / preparation / item bank (all saved hashes re-read before and after),
    - role names are the frozen v_preparation ROLES plus EXTERNAL (index 6), so the per-role summary can name the
      EXTERNAL code (frozen ROLES[6] would raise).
  CPU pinning to CPU 0 and the 10 GiB free-RAM precondition of the frozen run are not applied (no value depends on
  them; a 4 GiB guard is kept).  Unchanged frozen calls: projection.project_match (every feature block),
  bank.load_source_shard / local_source, schema.field_manifest, run._support_add / _serial_support / code_bindings,
  item_node_observed_20261002.bank.ObservedBankLoader, and the frozen block specs/dtypes (QUERY_SPECS, STATIC_SPECS,
  QUERY_DTYPE, MATCH_QUERY_DTYPE, FORMAT).  The output therefore loads with the frozen RawProjectionLoader.

Outputs (stage_dir(dataset,'projection')): the frozen file set of full_raw_20261002_1.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402

RESERVE = 8 * (1 << 30)


def input_bindings(source: Path, preparation: Path, items: Path) -> dict:
    from studies.external_validation_20261009.e2a_items import preparation_bindings, source_bindings

    item_manifest = C.read_json(items / "RUN_MANIFEST.json")
    if item_manifest.get("complete") is not True or item_manifest.get("accepted_by_e2a_checks") is not True:
        raise ValueError("item bank is incomplete")
    prep_manifest = C.read_json(preparation / "RUN_MANIFEST.json")
    for name, saved in prep_manifest["saved_files"].items():
        if C.sha256_file(preparation / name) != saved["sha256"]:
            raise ValueError(f"preparation saved artifact changed: {name}")
    for name, saved in item_manifest["saved_files"].items():
        if C.sha256_file(items / name) != saved["sha256"]:
            raise ValueError(f"item saved artifact changed: {name}")
    return {"source": source_bindings(source, preparation), "preparation": preparation_bindings(preparation),
            "preparation_manifest_sha256": C.sha256_file(preparation / "RUN_MANIFEST.json"),
            "item_manifest_sha256": C.sha256_file(items / "RUN_MANIFEST.json"),
            "observable_manifest_sha256": C.sha256_file(source / "RUN_MANIFEST.json")}


def run_projection(dataset: str, log) -> dict:
    C.runtime_hygiene()
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits
    from studies.item_node_observed_20261002.bank import ObservedBankLoader
    from studies.v_input_projection_20261002.bank import (FORMAT, MATCH_QUERY_DTYPE, QUERY_DTYPE, QUERY_SPECS,
                                                          STATIC_SPECS, load_source_shard, local_source)
    from studies.v_input_projection_20261002.projection import project_match
    from studies.v_input_projection_20261002.run import _close, _serial_support, _support_add, code_bindings
    from studies.v_input_projection_20261002.schema import field_manifest
    from studies.v_preparation_20261002.run import ROLES as FROZEN_ROLES

    roles_names = tuple(FROZEN_ROLES) + (C.EXTERNAL_ROLE,)
    if roles_names.index(C.EXTERNAL_ROLE) != C.EXTERNAL_ROLE_CODE:
        raise ValueError("EXTERNAL role code differs from its position after the frozen roles")
    source, preparation, items = (C.stage_dir(dataset, s) for s in ("bank", "preparation", "items"))
    output = C.stage_dir(dataset, "projection")
    if (output / "RUN_MANIFEST.json").exists() and C.read_json(output / "RUN_MANIFEST.json").get("complete"):
        log(f"[projection] {dataset}: complete projection exists, reused")
        return C.read_json(output / "RUN_MANIFEST.json")
    if output.exists():
        shutil.rmtree(output)
    pool_limit = threadpool_limits(limits=1)
    actual_pools = threadpool_info()
    if any(pool["num_threads"] != 1 for pool in actual_pools):
        pool_limit.restore_original_limits()
        raise RuntimeError("numerical thread pool limit was not applied")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    (output / "RUN_START.json").write_text(json.dumps({"started_utc": C.utc_now(), "status": "running", "max_matches": None}, indent=2) + "\n", encoding="utf-8")
    mapped = []
    item_loader = None
    try:
        before = input_bindings(source, preparation, items)
        code_before = code_bindings()
        prep_matches = np.load(preparation / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        prep_rows = np.load(preparation / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
        prep_contexts = np.load(preparation / "CONTEXT_MEMBERSHIP.npy", mmap_mode="r", allow_pickle=False)
        mapped.extend((prep_matches, prep_rows, prep_contexts))
        item_loader = ObservedBankLoader(items, verify_hashes=False)  # all saved bytes verified above
        n_matches = len(prep_matches)
        if len(item_loader.matches) != len(prep_matches) or not np.array_equal(prep_matches["key"], item_loader.matches["canonical_key"]):
            raise ValueError("item/preparation match order differs")
        last_match = prep_matches[n_matches - 1]
        n_rows = int(last_match["first_nominal_row"]) + int(last_match["n_nominal_rows"])
        n_queries = int(item_loader.matches[n_matches - 1]["first_query"]) + int(item_loader.matches[n_matches - 1]["n_queries"])
        payload = sum(n_queries * int(np.prod(tail)) * np.dtype(dtype).itemsize for dtype, tail in QUERY_SPECS.values())
        payload += sum(n_matches * int(np.prod(tail)) * np.dtype(dtype).itemsize for dtype, tail in STATIC_SPECS.values())
        payload += n_rows * prep_rows.dtype.itemsize + n_matches * prep_matches.dtype.itemsize + n_queries * QUERY_DTYPE.itemsize
        if shutil.disk_usage(output).free < RESERVE + payload + (1 << 30):
            raise RuntimeError("projected output and reserve exceed current free disk")
        if psutil.virtual_memory().available < (4 << 30):
            raise RuntimeError("less than 4GiB available RAM before projection")
        blocks = {}
        for name, (dtype, tail) in {**QUERY_SPECS, **STATIC_SPECS}.items():
            size = n_queries if name in QUERY_SPECS else n_matches
            blocks[name] = np.lib.format.open_memmap(output / (name.upper() + ".npy"), mode="w+", dtype=dtype, shape=(size, *tail))
        mapped.extend(blocks.values())
        query_table = np.lib.format.open_memmap(output / "QUERY_TABLE.npy", mode="w+", dtype=QUERY_DTYPE, shape=(n_queries,))
        match_queries = np.lib.format.open_memmap(output / "MATCH_QUERIES.npy", mode="w+", dtype=MATCH_QUERY_DTYPE, shape=(n_matches,))
        mapped.extend((query_table, match_queries))
        np.save(output / "MATCH_TABLE.npy", np.asarray(prep_matches[:n_matches]), allow_pickle=False)
        np.save(output / "V_ROWS.npy", np.asarray(prep_rows[:n_rows]), allow_pickle=False)
        np.save(output / "CONTEXT_MEMBERSHIP.npy", np.asarray(prep_contexts[:, :n_matches]), allow_pickle=False)
        (output / "FIELD_MANIFEST.json").write_text(json.dumps(field_manifest(), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        counts = Counter()
        roles = {role: Counter() for role in roles_names}
        support = {role: {"all_queries": {}, "nominal_supported": {}, "static": {}} for role in roles_names}
        expected_fields = set(QUERY_SPECS) | set(STATIC_SPECS)
        query_cursor = 0
        source_records = before["source"]["records"]
        with (output / "PROGRESS.jsonl").open("w", encoding="utf-8") as progress:
            for shard, binding in enumerate(source_records):
                selected = np.flatnonzero(prep_matches[:n_matches]["shard"] == shard)
                if not len(selected):
                    continue
                arrays, layout = load_source_shard(source / binding["directory"])
                for match_index_value in selected:
                    match_index = int(match_index_value)
                    match = prep_matches[match_index]
                    raw = local_source(arrays, layout, int(match["local_match"]))
                    projected = project_match(raw)
                    if set(projected) != expected_fields:
                        raise ValueError(f"raw projection fields differ: {set(projected) ^ expected_fields}")
                    size = len(raw["query_ms"])
                    item_match = item_loader.matches[match_index]
                    if int(item_match["first_query"]) != query_cursor or int(item_match["n_queries"]) != size:
                        raise ValueError("item query offset differs from source order")
                    item_queries = item_loader.queries[query_cursor:query_cursor + size]
                    if not np.array_equal(item_queries["query_ms"], raw["query_ms"]) or not np.array_equal(item_queries["event_limit"], raw["query_event_limit"]):
                        raise ValueError("item/source inclusive query references differ")
                    for name, (dtype, tail) in QUERY_SPECS.items():
                        values = np.asarray(projected[name])
                        if values.shape != (size, *tail) or values.dtype != np.dtype(dtype):
                            raise ValueError(f"invalid query block {name}: {values.shape}/{values.dtype}")
                        blocks[name][query_cursor:query_cursor + size] = values
                    for name, (dtype, tail) in STATIC_SPECS.items():
                        values = np.asarray(projected[name])
                        if values.shape != tail or values.dtype != np.dtype(dtype):
                            raise ValueError(f"invalid static block {name}: {values.shape}/{values.dtype}")
                        blocks[name][match_index] = values
                    qrows = query_table[query_cursor:query_cursor + size]
                    qrows["match_index"], qrows["local_query"] = match_index, np.arange(size)
                    qrows["query_ms"] = raw["query_ms"]
                    qrows["frame_index"] = raw["query_frame_index"]
                    qrows["event_limit"] = raw["query_event_limit"]
                    qrows["item_query"] = np.arange(query_cursor, query_cursor + size)
                    match_queries[match_index] = (query_cursor, size)
                    first, number = int(match["first_nominal_row"]), int(match["n_nominal_rows"])
                    nominal = prep_rows[first:first + number]
                    source_nominal = np.asarray(raw["v_nominal"])
                    if len(source_nominal) != number or not np.array_equal(nominal["query_ref"], source_nominal[:, 6]):
                        raise ValueError("nominal raw/source reference mismatch")
                    supported = np.asarray(nominal["supported"], dtype=bool)
                    refs = nominal["query_ref"][supported]
                    if np.any(refs < 0) or np.any(refs >= size) or not np.array_equal(raw["query_ms"][refs], nominal["actual_ms"][supported]):
                        raise ValueError("supported nominal references differ")
                    if np.any(nominal["query_ref"][~supported] != -1):
                        raise ValueError("unsupported nominal row retains a model query")
                    role = roles_names[int(match["role"])]
                    update = dict(matches=1, unique_queries=size, nominal_rows=number, supported_nominal_rows=int(supported.sum()),
                                  recorded_events=len(raw["event_timestamp"]))
                    counts.update(update)
                    roles[role].update(update)
                    roles[role]["observed_supported_player_queries"] += int(item_loader.support[query_cursor:query_cursor + size].sum())
                    for name in QUERY_SPECS:
                        _support_add(support[role]["all_queries"], name, projected[name])
                        _support_add(support[role]["nominal_supported"], name, projected[name][refs])
                    for name in STATIC_SPECS:
                        _support_add(support[role]["static"], name, projected[name][None, ...])
                    query_cursor += size
                counts["source_shards"] += 1
                if shutil.disk_usage(output).free < RESERVE:
                    raise RuntimeError("8GiB disk reserve reached")
                if psutil.Process().memory_info().rss > (16 << 30):
                    raise RuntimeError("16GiB process RSS budget exceeded")
                progress_row = {"phase": "raw_projection", **dict(counts), "elapsed_seconds": time.perf_counter() - started,
                                "rss_bytes": psutil.Process().memory_info().rss, "free_bytes": shutil.disk_usage(output).free}
                progress.write(json.dumps(progress_row) + "\n")
                progress.flush()
                del arrays
        if query_cursor != n_queries or counts["matches"] != n_matches or counts["nominal_rows"] != n_rows:
            raise ValueError("projection cardinalities differ from the preparation references")
        _close(mapped)
        mapped.clear()
        item_loader.close()
        item_loader = None
        metadata = {"format_version": FORMAT, "source_root": str(source), "preparation_root": str(preparation), "item_root": str(items),
                    "item_manifest_sha256": before["item_manifest_sha256"], "source_shards": source_records,
                    "query_block_specs": {name: {"dtype": dtype, "tail_shape": list(tail)} for name, (dtype, tail) in QUERY_SPECS.items()},
                    "static_block_specs": {name: {"dtype": dtype, "tail_shape": list(tail)} for name, (dtype, tail) in STATIC_SPECS.items()},
                    "history_storage": "Accepted source references with inclusive cutoff; no duplicated query prefixes",
                    "nominal_only_for_V_fit": True, "metadata_is_model_input": False,
                    "whole_actual_inventory_complete": "unknown", "training_ready": False,
                    "dataset": dataset, "role": C.EXTERNAL_ROLE}
        (output / "BANK_METADATA.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        nonempty = {role: value for role, value in support.items() if any(value.values())}
        (output / "FIELD_SUPPORT_COUNTS.json").write_text(json.dumps({"counts_are_observed_values_not_raw_presence": True, "roles": _serial_support(nonempty)},
                                                                     indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (output / "SUMMARY.json").write_text(json.dumps({"counts": dict(counts), "roles": {key: dict(value) for key, value in roles.items() if value},
                                                         "full_scope": True, "new_models_fitted": False, "training_ready": False,
                                                         "dataset": dataset, "role": C.EXTERNAL_ROLE}, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        after = input_bindings(source, preparation, items)
        code_after = code_bindings()
        if before != after or code_before != code_after:
            raise ValueError("input or production code changed during projection")
        saved = {path.name: {"bytes": path.stat().st_size, "sha256": C.sha256_file(path)}
                 for path in sorted(output.iterdir()) if path.is_file() and path.name not in {"RUN_MANIFEST.json", "FAILED.json"}}
        result = {"complete": True, "accepted_projection": False, "accepted_by_e2a_checks": True, "full_scope": True,
                  "full_corpus_verified": False, "training_ready": False, "new_models_fitted": False, "new_SVI_labels_produced": False,
                  "dataset": dataset, "role": C.EXTERNAL_ROLE, "counts": dict(counts), "bindings_before": before, "bindings_after": after,
                  "code_bindings_before": code_before, "code_bindings_after": code_after, "saved_files": saved,
                  "build_gates": {"verify_hashes": True, "max_matches": None, "full_gate": True},
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__, "CPU_affinity": "not pinned",
                              "actual_thread_pools": actual_pools,
                              "numeric_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                              "GPU_used": False}, "elapsed_seconds": round(time.perf_counter() - started, 1),
                  "completed_utc": C.utc_now(), "hook": C.FROZEN_HOOKS["projection"]}
        (output / "RUN_MANIFEST.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        log(f"[projection] {dataset}: {dict(counts)} in {result['elapsed_seconds']} s")
        return result
    except Exception as error:
        (output / "FAILED.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error),
                                                        "elapsed_seconds": time.perf_counter() - started}) + "\n", encoding="utf-8")
        raise
    finally:
        _close(mapped)
        if item_loader is not None:
            item_loader.close()
        pool_limit.restore_original_limits()
