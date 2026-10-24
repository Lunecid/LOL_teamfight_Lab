"""Materialize shared raw numeric blocks without learning or changing populations."""
from __future__ import annotations

import json
import os
import platform
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from studies.item_node_encoding_20261002.audit import verify_preparation_bindings
from studies.item_node_encoding_20261002.source import AcceptedSourceReader, file_sha256
from studies.item_node_observed_20261002.bank import ObservedBankLoader
from studies.observable_state_20261002.run import bindings as observable_bindings
from studies.v_preparation_20261002.run import ROLES
from .bank import (ROOT, FORMAT, QUERY_SPECS, STATIC_SPECS, QUERY_DTYPE, MATCH_QUERY_DTYPE,
                   read_json, load_source_shard, local_source)

REPORT = ROOT / "reports/v_input_projection_20261002"
SOURCE = ROOT / "outputs/observable_state_20261002/full_static_preserved_20261002_1"
PREPARATION = ROOT / "outputs/v_preparation_20261002/full_nominal_refs_20261002_1"
ITEMS = ROOT / "outputs/item_node_observed_20261002/full_bank_v3_20261002_1"
EXPECTED = {"matches": 209999, "unique_queries": 4384026, "nominal_rows": 2618905, "source_shards": 411}
RESERVE = 8 * (1 << 30)


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def code_bindings() -> dict:
    result = observable_bindings()
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((ROOT / "studies/item_node_observed_20261002").glob("*.py"))
    paths += list((ROOT / "studies/item_node_encoding_20261002").glob("*.py"))
    paths += list((ROOT / "studies/v_preparation_20261002").glob("*.py"))
    paths += [REPORT / "CONTRACT.json", ROOT / "reports/v_input_adapter_20261002/CONTRACT.json",
              ROOT / "reports/item_node_observed_20261002/CONTRACT.json",
              ROOT / "reports/v_preparation_20261002/CONTRACT.json"]
    result.update({str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths})
    return result


def input_bindings(source: Path, preparation: Path, items: Path) -> dict:
    record = read_json(ROOT / "reports/item_node_observed_20261002/ACCEPTANCE.json")
    if record.get("input_preparation_accepted") is not True or Path(record["accepted_run"]).resolve() != items.resolve():
        raise ValueError("requested item bank is not the accepted v3 run")
    for relative, expected in record["evidence"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"item acceptance evidence changed: {relative}")
    item_manifest = read_json(items / "RUN_MANIFEST.json")
    for relative, expected in item_manifest["code_bindings_after"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"item dependency changed: {relative}")
    source_record = read_json(ROOT / "reports/observable_state_20261002/ACCEPTANCE.json")
    if not source_record.get("phase_accepted") or Path(source_record["accepted_run"]).resolve() != source.resolve():
        raise ValueError("requested observable source is not accepted")
    if file_sha256(source / "RUN_MANIFEST.json") != source_record["run_manifest_sha256"]:
        raise ValueError("accepted observable manifest changed")
    source_binding = AcceptedSourceReader(source, preparation, verify_hashes=True).verify_source_bindings()
    preparation_binding = verify_preparation_bindings(preparation)
    prep_manifest = read_json(preparation / "RUN_MANIFEST.json")
    for name, saved in prep_manifest["saved_files"].items():
        if file_sha256(preparation / name) != saved["sha256"]:
            raise ValueError(f"preparation saved artifact changed: {name}")
    for name, saved in item_manifest["saved_files"].items():
        if file_sha256(items / name) != saved["sha256"]:
            raise ValueError(f"item saved artifact changed: {name}")
    return {"source": source_binding, "preparation": preparation_binding,
            "preparation_manifest_sha256": file_sha256(preparation / "RUN_MANIFEST.json"),
            "item_manifest_sha256": file_sha256(items / "RUN_MANIFEST.json"),
            "item_acceptance_sha256": file_sha256(ROOT / "reports/item_node_observed_20261002/ACCEPTANCE.json"),
            "observable_manifest_sha256": file_sha256(source / "RUN_MANIFEST.json")}


def _close(arrays):
    for array in arrays:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            if getattr(array, "mode", None) != "r":
                array.flush()
            mmap.close()


def _support_add(summary: dict, name: str, values: np.ndarray):
    data = np.asarray(values)
    if name not in summary:
        summary[name] = {"rows": 0, "finite_per_position": np.zeros(data.shape[1:], dtype=np.int64),
                         "zero_per_position": np.zeros(data.shape[1:], dtype=np.int64)}
    row = summary[name]
    row["rows"] += len(data)
    row["finite_per_position"] += np.isfinite(data).sum(axis=0, dtype=np.int64)
    row["zero_per_position"] += (data == 0).sum(axis=0, dtype=np.int64)


def _serial_support(value):
    if isinstance(value, dict):
        return {key: _serial_support(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build(output: Path, *, source: Path = SOURCE, preparation: Path = PREPARATION,
          items: Path = ITEMS, max_matches: int | None = None) -> dict:
    from .projection import project_match
    from .schema import field_manifest
    output, source, preparation, items = map(Path, (output, source, preparation, items))
    if max_matches is not None and (isinstance(max_matches, bool) or max_matches < 1):
        raise ValueError("max_matches must be positive or omitted")
    if output.exists():
        raise FileExistsError(output)
    pool_limit = threadpool_limits(limits=1)
    actual_pools = threadpool_info()
    if any(pool["num_threads"] != 1 for pool in actual_pools):
        pool_limit.restore_original_limits()
        raise RuntimeError("numerical thread pool limit was not applied")
    original_affinity = psutil.Process().cpu_affinity()
    psutil.Process().cpu_affinity([0])
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "RUN_START.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                          "status": "running", "max_matches": max_matches})
    mapped = []
    item_loader = None
    try:
        before = input_bindings(source, preparation, items)
        code_before = code_bindings()
        prep_matches = np.load(preparation / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        prep_rows = np.load(preparation / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
        prep_contexts = np.load(preparation / "CONTEXT_MEMBERSHIP.npy", mmap_mode="r", allow_pickle=False)
        mapped.extend((prep_matches, prep_rows, prep_contexts))
        item_loader = ObservedBankLoader(items, verify_hashes=False)  # All saved bytes verified above.
        n_matches = len(prep_matches) if max_matches is None else min(int(max_matches), len(prep_matches))
        if len(item_loader.matches) != len(prep_matches) or not np.array_equal(prep_matches["key"], item_loader.matches["canonical_key"]):
            raise ValueError("accepted item/preparation match order differs")
        last_match = prep_matches[n_matches - 1]
        n_rows = int(last_match["first_nominal_row"]) + int(last_match["n_nominal_rows"])
        n_queries = int(item_loader.matches[n_matches - 1]["first_query"]) + int(item_loader.matches[n_matches - 1]["n_queries"])
        payload = sum(n_queries * int(np.prod(tail)) * np.dtype(dtype).itemsize for dtype, tail in QUERY_SPECS.values())
        payload += sum(n_matches * int(np.prod(tail)) * np.dtype(dtype).itemsize for dtype, tail in STATIC_SPECS.values())
        payload += n_rows * prep_rows.dtype.itemsize + n_matches * prep_matches.dtype.itemsize + n_queries * QUERY_DTYPE.itemsize
        if shutil.disk_usage(output).free < RESERVE + payload + (1 << 30):
            raise RuntimeError("projected output and reserve exceed current free disk")
        if psutil.virtual_memory().available < (10 << 30):
            raise RuntimeError("less than 10GiB available RAM before projection")
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
        write_json(output / "FIELD_MANIFEST.json", field_manifest())
        counts = Counter()
        roles = {role: Counter() for role in ROLES}
        support = {role: {"all_queries": {}, "nominal_supported": {}, "static": {}} for role in ROLES}
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
                    role = ROLES[int(match["role"])]
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
                print(json.dumps(progress_row), flush=True)
                del arrays
        if query_cursor != n_queries or counts["matches"] != n_matches or counts["nominal_rows"] != n_rows:
            raise ValueError("projection cardinalities differ from accepted references")
        full = max_matches is None and all(counts[name] == value for name, value in EXPECTED.items())
        if max_matches is None and not full:
            raise ValueError("full projection did not consume all accepted rows")
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
                    "whole_actual_inventory_complete": "unknown", "training_ready": False}
        write_json(output / "BANK_METADATA.json", metadata)
        write_json(output / "FIELD_SUPPORT_COUNTS.json", {"counts_are_observed_values_not_raw_presence": True, "roles": _serial_support(support)})
        write_json(output / "SUMMARY.json", {"counts": dict(counts), "roles": {key: dict(value) for key, value in roles.items()},
                                             "full_scope": full, "new_models_fitted": False, "training_ready": False})
        after = input_bindings(source, preparation, items)
        code_after = code_bindings()
        if before != after or code_before != code_after:
            raise ValueError("input or production code changed during projection")
        saved = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                 for path in sorted(output.iterdir()) if path.is_file() and path.name not in {"RUN_MANIFEST.json", "FAILED.json"}}
        result = {"complete": True, "accepted_projection": False, "full_scope": full, "full_corpus_verified": False,
                  "training_ready": False, "new_models_fitted": False, "new_SVI_labels_produced": False,
                  "counts": dict(counts), "bindings_before": before, "bindings_after": after,
                  "code_bindings_before": code_before, "code_bindings_after": code_after, "saved_files": saved,
                  "build_gates": {"verify_hashes": True, "max_matches": max_matches, "full_gate": full},
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__, "CPU_affinity": psutil.Process().cpu_affinity(),
                              "actual_thread_pools": actual_pools,
                              "numeric_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                              "GPU_used": False}, "elapsed_seconds": time.perf_counter() - started,
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "RUN_MANIFEST.json", result)
        return result
    except Exception as error:
        write_json(output / "FAILED.json", {"type": type(error).__name__, "message": str(error), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        _close(mapped)
        if item_loader is not None:
            item_loader.close()
        pool_limit.restore_original_limits()
        psutil.Process().cpu_affinity(original_affinity)
