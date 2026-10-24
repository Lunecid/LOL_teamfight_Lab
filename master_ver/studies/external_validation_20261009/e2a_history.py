"""E2a stage 'history': shared-history store for an EXTERNAL projection.

README
  Semantics are those of reports/shared_history_eight_cpu_20261003/run.py (frozen): per 256-match shard a worker opens
  the frozen RawProjectionLoader, streams studies/v_shared_history_20261003/bank.iter_match_views, requires every
  query's semantic_row(view) to equal semantic_row(common.query_view(loader, q)) (the independent reference path),
  writes the shard with v_shared_history_production_20261003/store.write_shard, reloads it with store.ShardReader,
  checks coverage, and re-checks every query's semantic fingerprint after reload.  The parent re-validates each
  receipt (ShardReader hashes + coverage + counts) and commits accepted_<first>.json; ACCEPTANCE.json and PLAN.json
  follow the frozen layout so the store opens with the frozen BankSurface-style readers.
  Changed: scheduling only - a <= 6-process spawn pool at below-normal priority instead of the eight leased, pinned
  CPUs (the frozen runner refuses anything but eight CPU ids); the identity (PLAN.json) records the E2a dataset, role and
  code instead of the 2025 research/resource contracts.  The three helpers equal / fingerprint / validate_coverage are
  verbatim copies (their frozen modules import torch through reports/v_model_fit_input_20261003/integration.py).

Outputs (stage_dir(dataset,'history')): PLAN.json, accepted_<first:06d>.json, shards/shard_<first>_<uuid>/{shard.bin,
index.json, MANIFEST.json}, ACCEPTANCE.json.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402

VERBATIM_SOURCES = {
    "equal": "reports/v_stream_preparation_20261003/history_probe.py:20-32",
    "fingerprint": "reports/shared_history_production_20261003/run.py:40-41",
    "validate_coverage": "reports/shared_history_production_20261003/run.py:44-56",
}


def equal(actual, expected):
    import numpy as np

    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("semantic keys differ")
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError("semantic lengths differ")
        for a, b in zip(actual, expected):
            equal(a, b)
    else:
        np.testing.assert_equal(actual, expected)


def fingerprint(value):
    from studies.v_shared_history_20261003.bank import _plain

    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=True).encode()).hexdigest()


def validate_coverage(reader, loader, first, last):
    import numpy as np

    if reader.match_indices != list(range(first, last)):
        raise ValueError('Shard actual match coverage differs')
    for match in range(first, last):
        start = int(loader.match_queries[match]['first_query'])
        count = int(loader.match_queries[match]['n_queries'])
        refs = loader.queries[start:start+count]
        if not np.all(refs['match_index'] == match):
            raise ValueError('Source query ownership differs')
        entry = reader.index[match]
        if entry['query_ids'] != list(range(start, start+count)) or entry['query_ms'] != refs['query_ms'].tolist():
            raise ValueError('Shard actual query coverage/time differs')


def atomic(path: Path, value) -> None:
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _code_paths() -> list[Path]:
    root = C.ROOT
    paths = [root / "reports/shared_history_eight_cpu_20261003/run.py",
             root / "reports/shared_history_production_20261003/run.py",
             root / "reports/v_stream_preparation_20261003/history_probe.py"]
    for folder in ("v_shared_history_20261003", "v_shared_history_production_20261003", "v_model_input_20261003",
                   "v_model_fit_input_20261003", "v_input_projection_20261002", "item_node_observed_20261002",
                   "observable_state_20261002"):
        paths.extend((root / "studies" / folder).glob("*.py"))
    paths.extend((C.PACKAGE / "e2a_common.py", C.PACKAGE / "e2a_history.py"))
    return sorted({p.resolve() for p in paths if p.is_file()})


def _init_worker() -> None:
    C.runtime_hygiene()


def history_shard_task(payload: dict) -> dict:
    """One shard: generate, compare with the reference path, write, reload, re-compare."""
    from studies.v_input_projection_20261002.bank import RawProjectionLoader
    from studies.v_model_fit_input_20261003.common import query_view, semantic_row
    from studies.v_shared_history_20261003.bank import iter_match_views, loader_fingerprint
    from studies.v_shared_history_production_20261003.store import ShardReader, write_shard

    identity = payload["identity"]
    first, last = int(payload["first"]), int(payload["last"])
    output = Path(payload["output_root"])
    directory = output / "shards" / f"shard_{first:06d}_{uuid.uuid4().hex}"
    expected: dict[int, str] = {}
    started = time.perf_counter()
    with RawProjectionLoader(Path(payload["projection"]), verify_hashes=False) as loader:
        if loader_fingerprint(loader) != identity["loader_manifest_sha256"]:
            raise ValueError("worker loader manifest differs from identity")

        def matches():
            for match in range(first, last):
                def records(match: int = match):
                    count = int(loader.match_queries[match]["n_queries"])
                    if count == 0:
                        return
                    for record in iter_match_views(loader, match):
                        actual = semantic_row(record["view"])
                        reference = semantic_row(query_view(loader, record["query_id"]))
                        equal(actual, reference)
                        expected[int(record["query_id"])] = fingerprint(reference)
                        yield record

                yield match, records()

        write_shard(directory, matches(), identity=identity, max_matches=int(payload["shard_matches"]))
        reader = ShardReader(directory, identity=identity)
        validate_coverage(reader, loader, first, last)
        expected_queries = int(sum(loader.match_queries["n_queries"][first:last]))
        if len(expected) != expected_queries:
            raise ValueError("full shard query coverage differs")
        for query_id, expected_hash in expected.items():
            actual_hash = fingerprint(semantic_row(reader.query_view(loader, query_id)))
            if actual_hash != expected_hash:
                raise ValueError("reload semantic comparison differs")
    relative = str(directory.relative_to(output)).replace("\\", "/")
    return {
        "directory": relative,
        "match_range": [first, last],
        "queries": expected_queries,
        "identity_sha256": fingerprint(identity),
        "manifest_sha256": C.sha256_file(directory / "MANIFEST.json"),
        "all_queries_reference_and_reload_equal": True,
        "worker_loader_verify_hashes": False,
        "worker_pid": os.getpid(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def _validate_receipt(receipt: dict, loader, identity: dict, output: Path, first: int, last: int) -> None:
    from studies.v_shared_history_production_20261003.store import ShardReader

    expected_queries = int(sum(loader.match_queries["n_queries"][first:last]))
    if (receipt.get("identity_sha256") != fingerprint(identity) or receipt.get("match_range") != [first, last]
            or int(receipt.get("queries", -1)) != expected_queries
            or receipt.get("all_queries_reference_and_reload_equal") is not True):
        raise ValueError("worker receipt identity or range differs")
    directory = (output / receipt["directory"]).resolve()
    directory.relative_to(output.resolve())
    if not directory.is_dir() or C.sha256_file(directory / "MANIFEST.json") != receipt.get("manifest_sha256"):
        raise ValueError("worker shard manifest missing or hash differs")
    reader = ShardReader(directory, identity=identity)
    validate_coverage(reader, loader, first, last)
    if reader.manifest.get("n_queries") != expected_queries:
        raise ValueError("worker shard query count differs")


def run_history(dataset: str, workers: int, log) -> dict:
    C.runtime_hygiene()
    import numpy as np
    from studies.v_input_projection_20261002.bank import RawProjectionLoader
    from studies.v_model_fit_input_20261003.common import semantic_fingerprint
    from studies.v_shared_history_20261003.bank import loader_fingerprint

    projection, output = C.stage_dir(dataset, "projection"), C.stage_dir(dataset, "history")
    if (output / "ACCEPTANCE.json").exists():
        log(f"[history] {dataset}: accepted history exists, reused")
        return C.read_json(output / "ACCEPTANCE.json")
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    started = time.monotonic()
    projection_manifest = C.read_json(projection / "RUN_MANIFEST.json")
    with RawProjectionLoader(projection, verify_hashes=True) as loader:
        total_matches = len(loader.matches)
        total_queries = int(sum(loader.match_queries["n_queries"][:total_matches]))
        empty_query_matches = int(np.count_nonzero(loader.match_queries["n_queries"][:total_matches] == 0))
        identity = {
            "source_bindings": {"projection_root": str(projection),
                                "projection_manifest_sha256": C.sha256_file(projection / "RUN_MANIFEST.json"),
                                "projection_saved_files": projection_manifest["saved_files"],
                                "item_manifest_sha256": loader.metadata["item_manifest_sha256"],
                                "bank_manifest_sha256": C.sha256_file(C.stage_dir(dataset, "bank") / "RUN_MANIFEST.json")},
            "loader_manifest_sha256": loader_fingerprint(loader),
            "semantic_fingerprint": semantic_fingerprint(),
            "code": C.code_hashes(_code_paths()),
            "verbatim_helpers": VERBATIM_SOURCES,
            "dataset": dataset,
            "role": C.EXTERNAL_ROLE,
            "match_start": 0,
            "match_end": int(total_matches),
            "query_count": int(total_queries),
            "empty_query_matches": int(empty_query_matches),
            "shard_matches": int(C.HISTORY_SHARD_MATCHES),
            "workers_max": int(C.MAX_WORKERS),
            "numeric_threads": 1,
        }
        plan = output / "PLAN.json"
        if plan.exists():
            if json.loads(plan.read_text(encoding="utf-8")) != identity:
                raise ValueError("resume identity differs; use a new output directory")
        else:
            atomic(plan, identity)
        ranges = [(first, min(first + C.HISTORY_SHARD_MATCHES, total_matches))
                  for first in range(0, total_matches, C.HISTORY_SHARD_MATCHES)]
        receipts = {}
        for path in sorted(output.glob("accepted_*.json")):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            span = tuple(receipt["match_range"])
            if span not in ranges:
                raise ValueError("accepted receipt range is outside the current plan")
            _validate_receipt(receipt, loader, identity, output, *span)
            receipts[span] = receipt
        missing = [span for span in ranges if span not in receipts]
        log(f"[history] {dataset}: {total_matches} matches, {total_queries} queries, {len(ranges)} shards "
            f"({len(receipts)} reused, {len(missing)} to build)")
        context = multiprocessing.get_context("spawn")
        n_workers = max(1, min(int(workers), len(missing))) if missing else 0
        done_queries = 0
        if missing:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=context, initializer=_init_worker) as pool:
                pending, iterator = {}, iter(missing)

                def submit(span):
                    payload = {"projection": str(projection), "output_root": str(output), "first": span[0],
                               "last": span[1], "identity": identity, "shard_matches": C.HISTORY_SHARD_MATCHES}
                    pending[pool.submit(history_shard_task, payload)] = span

                for _ in range(n_workers):
                    span = next(iterator, None)
                    if span:
                        submit(span)
                while pending:
                    done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        span = pending.pop(future)
                        receipt = future.result()
                        _validate_receipt(receipt, loader, identity, output, *span)
                        path = output / f"accepted_{span[0]:06d}.json"
                        if path.exists():
                            raise ValueError(f"accepted receipt already exists: {path.name}")
                        atomic(path, receipt)
                        receipts[span] = receipt
                        done_queries += int(receipt["queries"])
                        elapsed = time.monotonic() - started
                        log(f"[history] {dataset}: shard {span} queries={receipt['queries']} "
                            f"accepted={len(receipts)}/{len(ranges)} rate={done_queries / max(elapsed, 1e-9):.1f} q/s")
                        span = next(iterator, None)
                        if span:
                            submit(span)
        if set(receipts) != set(ranges):
            raise ValueError("accepted shard ranges contain holes")
        touched = sorted({int(value) for value in loader.matches["shard"][:total_matches]})
        source_metadata = loader.metadata
    with RawProjectionLoader(projection, verify_hashes=True) as ending:
        if loader_fingerprint(ending) != identity["loader_manifest_sha256"]:
            raise ValueError("source loader manifest changed")
        ending._item_loader()
    for index in touched:
        bound = source_metadata["source_shards"][index]
        directory = Path(source_metadata["source_root"]) / bound["directory"]
        for key, name in (("archive_sha256", "BANK.npz"), ("layout_sha256", "LAYOUT.json"),
                          ("shard_manifest_sha256", "SHARD_MANIFEST.json"), ("match_ledger_sha256", "MATCH_LEDGER.jsonl"),
                          ("static_repair_sha256", "STATIC_REPAIR.json")):
            if C.sha256_file(directory / name) != bound[key]:
                raise ValueError("source history changed during production")
    if C.code_hashes(_code_paths()) != identity["code"]:
        raise ValueError("code changed during shared-history production")
    result = {
        "complete": True,
        "training_ready": False,
        "full_bank_produced": True,
        "dataset": dataset,
        "role": C.EXTERNAL_ROLE,
        "matches": total_matches,
        "queries": total_queries,
        "shards": [receipts[span] for span in ranges],
        "empty_query_matches": empty_query_matches,
        "source_before_after_equal": True,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "identity_sha256": fingerprint(identity),
        "workers": n_workers,
        "gpu_used": False,
        "hook": C.FROZEN_HOOKS["history"],
    }
    atomic(output / "ACCEPTANCE.json", result)
    log(f"[history] {dataset}: complete, {total_matches} matches, {total_queries} queries in {result['elapsed_seconds']} s")
    return result
