"""E2a stage 'bank': observable-state bank with the static-preservation repair, for an EXTERNAL dataset.

README
  Frozen code, called unchanged per match / per shard:
    studies/observable_state_20261002/producer.produce_match        (bank arrays, components, queries, ledger)
    studies/observable_state_20261002/bank.ShardBuilder, storage.write_shard / configure_write_lock
    studies/observable_static_preservation_20261002/repair.repair_shard  (literal cached static IDs, unchanged payloads)
    studies/observable_state_20261002/run._verify_shard, run.executable_schema
  Order per shard (same as the accepted two-step production full_20261002_1 -> full_static_preserved_20261002_1):
    produce -> bank_raw/<shard>  ->  repair -> bank/<shard>  ->  frozen independent verify (_verify_shard).
  The accepted parent was produced with an older kernel.py (negative cached selection IDs coalesced to 0) and repaired;
  the current kernel already preserves them, so the repair re-derives the literal IDs, checks them against the kernel
  and reports n_static_values_changed / n_bans_changed (0 expected).

  Hooks (worker-local rebinding, no file edited; see e2a_common.FROZEN_HOOKS):
    producer.read_cached_triplet -> source.read_cached_triplet(cache_dir=<dataset cache>)
    cache_source_audit_20261002.config.CACHE_DIR -> <dataset cache>     (read by repair_shard)
    producer.static_lookup_counts, repair.static_lookup_counts -> external_static_lookup_counts (ledger diagnostic)
  Shard loop = producer.produce_shard with per-match exclusion: a match whose frozen produce_match raises (e.g.
  counts differing from its audit) is excluded with the exception text instead of failing its shard; the ledger line
  gains the key 'dataset'.  Shards hold <= 512 matches in patch-index order, named external_0000, external_0001, ...

Outputs: stage_dir(dataset,'bank_raw')/<shard>, stage_dir(dataset,'bank')/{<shard>, RUN_MANIFEST.json,
FIELD_MANIFEST.json, FIELD_SUPPORT_COUNTS.json, BANK_ENTRIES.jsonl, EXCLUDED.json}.
"""
from __future__ import annotations

import functools
import json
import multiprocessing
import os
import platform
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402

FROZEN_LOOKUP_PATCHES = frozenset(("15.14", "15.15", "15.16"))
DATADRAGON = C.ROOT / "outputs" / "reest_exact_v4_20260925" / "datadragon_raw"


# --------------------------------------------------------------------------------------------- static diagnostics
@functools.lru_cache(maxsize=None)
def external_lookups(patch: str) -> dict:
    """Champion IDs of the patch's local Data Dragon championFull (same key parsing as champion_map._int_id).

    No summoner-spell or runesReforged table exists offline for patches after 15.16; those positions are reported as
    'recorded_ID_map_unavailable' rather than guessed.  Raw IDs are never changed.
    """
    path = DATADRAGON / f"championFull_{patch}.1.json"
    if not path.is_file():
        return {"champions": None, "source_hashes": {}, "champion_table": None}
    payload = json.loads(path.read_bytes())
    champions = set()
    for record in payload.get("data", {}).values():
        text = str(record.get("key", "")).strip()
        if text.isdigit() and int(text) > 0:
            champions.add(int(text))
    return {"champions": frozenset(champions), "source_hashes": {path.name: C.sha256_file(path)},
            "champion_table": str(path)}


def external_static_lookup_counts(static_ids, bans, patch: str) -> dict:
    """Frozen categorical.static_lookup_counts for 15.14-15.16; the same status vocabulary otherwise."""
    if patch in FROZEN_LOOKUP_PATCHES:
        from studies.observable_state_20261002.categorical import static_lookup_counts

        return static_lookup_counts(static_ids, bans, patch)
    from studies.observable_state_20261002.schema import STATIC_NAMES

    if static_ids.shape != (10, 14) or static_ids.dtype.kind not in "iu" or bans.shape != (2, 5):
        raise ValueError("static selection shape or dtype invalid")
    source = external_lookups(patch)
    champions = source["champions"]
    result = {}
    for index, name in enumerate(STATIC_NAMES):
        allowed = champions if name == "champion_id" else None
        counts = Counter()
        for value in static_ids[:, index].tolist():
            status = "nonpositive_cache_ID_presence_unknown" if value <= 0 else (
                "recorded_ID_map_unavailable" if allowed is None else
                "canonical_ID_known" if value in allowed else "positive_ID_lookup_missing")
            counts[status] += 1
        result[name] = dict(counts)
    counts = Counter()
    for value in bans.reshape(-1).tolist():
        value = int(value)
        counts["recorded_ID_map_unavailable" if champions is None and value > 0 else
               "canonical_ID_known" if champions is not None and value in champions else
               "nonpositive_cache_ID_presence_unknown" if value <= 0 else "positive_ID_lookup_missing"] += 1
    result["bans"] = dict(counts)
    return result


# --------------------------------------------------------------------------------------------- worker hooks
_HOOKS: dict = {}


def install_hooks(cache_dir: str, lock=None) -> dict:
    """Worker-local rebinding of the three frozen path/table hooks (documented in FROZEN_HOOKS)."""
    C.runtime_hygiene()
    from studies.cache_source_audit_20261002 import config as AC
    from studies.observable_state_20261002 import producer, source, storage
    from studies.observable_static_preservation_20261002 import repair

    if lock is not None:
        storage.configure_write_lock(lock)
    producer.read_cached_triplet = functools.partial(source.read_cached_triplet, cache_dir=Path(cache_dir))
    producer.static_lookup_counts = external_static_lookup_counts
    repair.static_lookup_counts = external_static_lookup_counts
    AC.CACHE_DIR = Path(cache_dir)
    _HOOKS.update(cache_dir=str(cache_dir), installed_pid=os.getpid())
    return dict(_HOOKS)


def _check_appendable(builder, arrays: dict) -> None:
    """ShardBuilder.append's checks without mutating the builder (append validates while it appends)."""
    import numpy as np

    if builder.keys is not None and set(arrays) != builder.keys:
        raise ValueError("match bank schema changed inside shard")
    for key, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise ValueError("object payload is forbidden")
        if value.ndim == 0:
            value = value.reshape(1)
        parts = builder.parts.get(key)
        if parts and value.shape[1:] != parts[0].shape[1:]:
            raise ValueError("trailing bank shape changed")


def produce_external_shard(directory: Path, entries: list[dict], dataset: str) -> dict:
    """producer.produce_shard with per-match exclusion and the 'dataset' ledger key."""
    import numpy as np  # noqa: F401  (numpy imported after thread limits)
    from studies.observable_state_20261002 import producer
    from studies.observable_state_20261002.bank import ShardBuilder
    from studies.observable_state_20261002.storage import write_shard

    start = time.perf_counter()
    builder = ShardBuilder()
    ledgers, produced, excluded = [], [], []
    counts = Counter()
    for entry in entries:
        try:
            bank, ledger = producer.produce_match(entry)
            _check_appendable(builder, bank)
        except Exception as exc:  # noqa: BLE001 - excluded match, reason preserved
            excluded.append({"match_id": entry["match_id"], "canonical_match_key": entry["canonical_match_key"],
                             "patch": entry["patch"], "stage": "bank_producer",
                             "reason": f"{type(exc).__name__}: {exc}"[:500]})
            continue
        ledger["dataset"] = dataset
        builder.append(bank)
        ledgers.append(ledger)
        produced.append(entry)
        counts.update(ledger["counts"])
    if not ledgers:
        return {"complete": True, "directory": directory.name, "empty": True, "produced": [], "excluded": excluded,
                "counts": {}, "elapsed_seconds": time.perf_counter() - start}
    arrays, layout = builder.finish()
    manifest = write_shard(directory, arrays, compressed=True)
    (directory / "LAYOUT.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    with (directory / "MATCH_LEDGER.jsonl").open("x", encoding="utf-8") as handle:
        for row in ledgers:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result = {"complete": True, "directory": directory.name, "role": C.EXTERNAL_ROLE, "dataset": dataset,
              "counts": dict(counts), "payload_bytes": manifest["logical_bytes"],
              "stored_bytes": manifest["archive_bytes"], "elapsed_seconds": time.perf_counter() - start}
    (directory / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {**result, "empty": False, "produced": produced, "excluded": excluded}


def _shard_complete(directory: Path) -> bool:
    manifest = directory / "SHARD_MANIFEST.json"
    required = ("STATIC_REPAIR.json", "MATCH_LEDGER.jsonl", "LAYOUT.json", "E2A_SHARD_RESULT.json")
    if not manifest.is_file() or not all((directory / name).is_file() for name in required):
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("complete") is True
    except ValueError:
        return False


def shard_task(task: tuple) -> dict:
    """produce -> repair -> frozen verify for one shard (runs in a hooked worker)."""
    from studies.observable_state_20261002.run import _verify_shard
    from studies.observable_static_preservation_20261002.repair import repair_shard

    raw_dir, final_dir, entries, dataset = Path(task[0]), Path(task[1]), task[2], task[3]
    started = time.perf_counter()
    if _shard_complete(final_dir):
        keys = [json.loads(line)["canonical_match_key"] for line in (final_dir / "MATCH_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
        by_key = {e["canonical_match_key"]: e for e in entries}
        produced = [by_key[k] for k in keys]
        excluded = [e for e in entries if e["canonical_match_key"] not in set(keys)]
        prior = json.loads((final_dir / "E2A_SHARD_RESULT.json").read_text(encoding="utf-8"))
        if sorted(e["canonical_match_key"] for e in prior["excluded"]) != sorted(e["canonical_match_key"] for e in excluded):
            raise ValueError(f"reused shard {final_dir.name} exclusions differ from its recorded result")
        verified = _verify_shard(final_dir, produced)
        return {**prior, "reused": True, "verified_counts": verified["counts"], "support": verified["support"],
                "produced_keys": keys}
    for stale in (raw_dir, final_dir):
        if stale.exists():
            shutil.rmtree(stale)
    produced = produce_external_shard(raw_dir, entries, dataset)
    if produced["empty"]:
        return {"directory": final_dir.name, "empty": True, "excluded": produced["excluded"], "produced_keys": []}
    entries_ok = produced["produced"]
    raw_verified = _verify_shard(raw_dir, entries_ok)
    repair = repair_shard(raw_dir, final_dir, entries_ok)
    verified = _verify_shard(final_dir, entries_ok)
    if raw_verified["counts"] != verified["counts"]:
        raise ValueError("static repair changed verified bank counts")
    result = {"directory": final_dir.name, "empty": False, "dataset": dataset, "role": C.EXTERNAL_ROLE,
              "n_matches": len(entries_ok), "producer_counts": produced["counts"],
              "repair_counts": repair["counts"], "stored_bytes": repair["stored_bytes"],
              "payload_bytes": produced["payload_bytes"], "excluded": produced["excluded"],
              "elapsed_seconds": round(time.perf_counter() - started, 2), "worker_pid": os.getpid()}
    (final_dir / "E2A_SHARD_RESULT.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return {**result, "verified_counts": verified["counts"], "support": verified["support"],
            "produced_keys": [e["canonical_match_key"] for e in entries_ok]}


def _init_worker(cache_dir: str, lock) -> None:
    install_hooks(cache_dir, lock)


def run_bank(dataset: str, workers: int, log) -> dict:
    from studies.external_validation_20261009.e2a_audit import load_external_entries

    C.runtime_hygiene()
    raw_root, bank_root = C.stage_dir(dataset, "bank_raw"), C.stage_dir(dataset, "bank")
    manifest_path = bank_root / "RUN_MANIFEST.json"
    if manifest_path.exists() and C.read_json(manifest_path).get("complete"):
        log(f"[bank] {dataset}: complete bank exists, reused")
        return C.read_json(manifest_path)
    import numpy as np
    from studies.observable_state_20261002.run import executable_schema

    entries, excluded_audit, binding = load_external_entries(dataset)
    raw_root.mkdir(parents=True, exist_ok=True)
    bank_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for number, start in enumerate(range(0, len(entries), C.SHARD_MATCHES)):
        name = f"external_{number:04d}"
        tasks.append((str(raw_root / name), str(bank_root / name), entries[start:start + C.SHARD_MATCHES], dataset))
    started = time.perf_counter()
    results, counts, verified_counts, repair_counts = [], Counter(), Counter(), Counter()
    support = Counter()
    excluded = list(excluded_audit)
    produced_keys = {}
    context = multiprocessing.get_context("spawn")
    n_workers = max(1, min(int(workers), len(tasks)))
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=context, initializer=_init_worker,
                             initargs=(str(C.CACHE[dataset]), context.Lock())) as pool:
        pending, iterator = {}, iter(tasks)
        for _ in range(n_workers):
            task = next(iterator, None)
            if task:
                pending[pool.submit(shard_task, task)] = task
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                result = future.result()
                results.append({k: v for k, v in result.items() if k not in {"support", "produced_keys"}})
                excluded.extend(result.get("excluded", []))
                if not result.get("empty"):
                    counts.update(result["producer_counts"])
                    verified_counts.update(result["verified_counts"])
                    repair_counts.update(result["repair_counts"])
                    support.update(result["support"])
                    produced_keys[result["directory"]] = result["produced_keys"]
                log(f"[bank] {dataset}: shard {result['directory']} matches={result.get('n_matches', 0)} "
                    f"excluded={len(result.get('excluded', []))} done={len(results)}/{len(tasks)} "
                    f"elapsed={time.perf_counter() - started:.0f}s")
                task = next(iterator, None)
                if task:
                    pending[pool.submit(shard_task, task)] = task
    if any(counts[k] != v for k, v in verified_counts.items()):
        raise ValueError("independent verified bank counts differ from producer counts")
    by_key = {e["canonical_match_key"]: e for e in entries}
    rows = []
    for directory in sorted(produced_keys):
        for local, key in enumerate(produced_keys[directory]):
            entry = by_key[key]
            rows.append({"shard": directory, "local_match": local, "match_id": entry["match_id"],
                         "canonical_match_key": key, "patch": entry["patch"], "dataset": dataset,
                         "original_index": entry["original_index"]})
    with (bank_root / "BANK_ENTRIES.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    schema = executable_schema()
    schema["state_generator_complete"] = True
    (bank_root / "FIELD_MANIFEST.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    (bank_root / "FIELD_SUPPORT_COUNTS.json").write_text(json.dumps(
        {"full_corpus": True, "role_counts": {C.EXTERNAL_ROLE: dict(support)}, "raw_field_presence_verified": False,
         "negative_cached_IDs_preserved": True}, indent=2) + "\n", encoding="utf-8")
    C.write_json(bank_root / "EXCLUDED.json", {"dataset": dataset, "n_excluded": len(excluded), "excluded": excluded,
                                               "reason_counts": dict(Counter(row["stage"] + ":" + row["reason"].split(":")[0] for row in excluded))})
    patches = sorted({e["patch"] for e in entries})
    manifest = {
        "version": C.VERSION, "stage": "bank", "dataset": dataset, "role": C.EXTERNAL_ROLE, "complete": True,
        "accepted_by_e2a_checks": True, "cache_dir": str(C.CACHE[dataset]), "entries_binding": binding,
        "n_indexed": binding["n_indexed"], "n_entries_after_audit": len(entries), "n_bank_matches": len(rows),
        "n_excluded": len(excluded), "counts": dict(counts), "independent_counts": dict(verified_counts),
        "static_repair_counts": dict(repair_counts),
        "shards": sorted(results, key=lambda r: r["directory"]),
        "static_lookup_tables": {p: (external_lookups(p)["source_hashes"] if p not in FROZEN_LOOKUP_PATCHES else "frozen categorical.lookups") for p in patches},
        "field_manifest_sha256": C.sha256_file(bank_root / "FIELD_MANIFEST.json"),
        "field_support_counts_sha256": C.sha256_file(bank_root / "FIELD_SUPPORT_COUNTS.json"),
        "bank_entries_sha256": C.sha256_file(bank_root / "BANK_ENTRIES.jsonl"),
        "hooks": {k: C.FROZEN_HOOKS[k] for k in ("bank_source_read", "bank_static_lookup", "bank_shard_loop", "static_repair_cache")},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "workers": n_workers,
                        "start_method": "spawn", "numeric_threads": 1, "GPU_used": False, "priority": "below_normal"},
        "elapsed_seconds": round(time.perf_counter() - started, 1), "completed_utc": C.utc_now(),
    }
    C.write_json(manifest_path, manifest)
    log(f"[bank] {dataset}: {len(rows)} matches in {len(produced_keys)} shards, excluded {len(excluded)}, "
        f"repair changes static={repair_counts['n_static_values_changed']} bans={repair_counts['n_bans_changed']}, "
        f"{manifest['elapsed_seconds']} s")
    return manifest
