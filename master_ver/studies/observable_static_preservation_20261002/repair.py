"""Reuse verified observable arrays and independently preserve cached static IDs."""
from __future__ import annotations

import json
import multiprocessing
import os
import platform
import shutil
import time
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from studies.observable_state_20261002.bank import match_view
from studies.observable_state_20261002.categorical import static_lookup_counts
from studies.observable_state_20261002.kernel import STATIC_NAMES, _static_arrays
from studies.observable_state_20261002.run import _reconcile_train, _verify_shard, bindings
from studies.observable_state_20261002.source import EXCLUDED_KEY, ROOT, file_sha256, load_frozen_entries
from studies.observable_state_20261002.storage import array_digest, close_shard, configure_write_lock, open_shard, write_shard
from studies.cache_source_audit_20261002 import config as AC
from studies.cache_source_audit_20261002.audit import _read_once


def direct_cached_static(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Independent literal projection: preserve negative native IDs as recorded."""
    source = meta["static_meta"]
    values = np.zeros((10, 14), dtype=np.int64)
    for pid in range(1, 11):
        for index, name in enumerate(STATIC_NAMES):
            if index == 0:
                raw = source.get("champion_by_pid", {}).get(str(pid))
            else:
                container = "summoner_spells_by_pid" if index in (1, 2) else "runes_by_pid"
                raw = source.get(container, {}).get(str(pid), {}).get(name)
            if type(raw) is int:
                values[pid - 1, index] = raw
            elif raw is not None:
                raise ValueError("cached selection field is not a native integer")
    bans = np.zeros((2, 5), dtype=np.int64)
    for team, side in enumerate(("blue", "red")):
        for index, value in enumerate(source.get("bans", {}).get(side, [])[:5]):
            if type(value) is not int:
                raise ValueError("cached ban field is not a native integer")
            bans[team, index] = value
    return values, bans


def repair_shard(parent: Path, destination: Path, entries: list[dict]) -> dict:
    layout = json.loads((parent / "LAYOUT.json").read_text(encoding="utf-8"))
    ledgers = [json.loads(line) for line in (parent / "MATCH_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(ledgers) != len(entries):
        raise ValueError("parent static repair coverage mismatch")
    arrays = open_shard(parent)
    corrected = dict(arrays)
    corrected["static_ids"] = arrays["static_ids"].copy()
    corrected["bans"] = arrays["bans"].copy()
    counts = Counter()
    counts["n_explicit_source_identity_rows"] = len(arrays["event_explicit_rows"])
    try:
        for index, (entry, ledger) in enumerate(zip(entries, ledgers)):
            if entry["canonical_match_key"] == EXCLUDED_KEY or entry["canonical_match_key"] != ledger["canonical_match_key"]:
                raise ValueError("excluded or mismatched match in static derivation")
            payload, details = _read_once(AC.CACHE_DIR / (entry["match_id"] + ".meta.json"))
            if payload is None or not details.get("stable") or details["sha256"] != entry["source_hashes"]["meta"]:
                raise ValueError("accepted original metadata hash changed")
            meta = json.loads(payload)
            literal, bans = direct_cached_static(meta)
            executable, executable_bans, _ = _static_arrays(meta)
            if not np.array_equal(literal, executable) or not np.array_equal(bans, executable_bans):
                raise ValueError("corrected executable static kernel differs from literal cached selection")
            before = match_view(arrays, layout, index)
            a, b = arrays["offset_static_ids"][index:index + 2]
            corrected["static_ids"][a:b] = literal
            a, b = arrays["offset_bans"][index:index + 2]
            corrected["bans"][a:b] = bans
            counts.update(n_matches=1, n_static_values_changed=int(np.sum(before["static_ids"] != literal)),
                          n_bans_changed=int(np.sum(before["bans"] != bans)),
                          n_negative_cached_static_ids=int(np.sum(literal < 0)),
                          n_negative_cached_ban_ids=int(np.sum(bans < 0)))
            ledger["static_preservation_verified"] = True
            ledger["support"]["static_lookup_counts"] = static_lookup_counts(literal, bans, entry["patch"])
        # No frame, event, query, component, exclusion or endpoint is recomputed.
        unchanged = {key: array_digest(value) for key, value in arrays.items() if key not in {"static_ids", "bans"}}
        saved = write_shard(destination, corrected, compressed=True)
        for key, digest in unchanged.items():
            if saved["arrays"][key]["payload_sha256"] != digest:
                raise ValueError("static derivation altered another observable array")
        shutil.copyfile(parent / "LAYOUT.json", destination / "LAYOUT.json")
        with (destination / "MATCH_LEDGER.jsonl").open("x", encoding="utf-8") as handle:
            for ledger in ledgers:
                handle.write(json.dumps(ledger, separators=(",", ":")) + "\n")
        result = {"directory": destination.name, "role": entries[0]["role"], "counts": dict(counts),
                  "source_archive_sha256": file_sha256(parent / "BANK.npz"),
                  "unchanged_payload_sha256": unchanged, "stored_bytes": saved["archive_bytes"]}
        (destination / "STATIC_REPAIR.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        close_shard(arrays)


def _initialize(lock) -> None:
    configure_write_lock(lock)


def _repair_task(task: tuple[Path, Path, list[dict]]) -> tuple[dict, dict]:
    parent, destination, entries = task
    result = repair_shard(parent, destination, entries)
    return result, _verify_shard(destination, entries)


def repair_bindings() -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
            for path in Path(__file__).parent.glob("*.py")}


def repair_run(parent: Path, output: Path) -> dict:
    parent_hash = file_sha256(parent / "RUN_MANIFEST.json")
    original = json.loads((parent / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if not original["complete"] or original["pilot"] or original.get("error"):
        raise ValueError("parent full observable computation is incomplete")
    checks_hash = file_sha256(parent / "INDEPENDENT_CHECKS.json")
    parent_checks = json.loads((parent / "INDEPENDENT_CHECKS.json").read_text(encoding="utf-8"))
    if not parent_checks["complete_full_coverage"] or not parent_checks["all_saved_arrays_reloaded_and_hashed"] or parent_checks["excluded_source_read"]:
        raise ValueError("parent independent full coverage was not verified")
    entries, cohort = load_frozen_entries()
    if cohort != original["cohort"]:
        raise ValueError("frozen population changed before static correction")
    output.mkdir(parents=True, exist_ok=False)
    code = bindings()
    repair_code = repair_bindings()
    manifest = {"complete": False, "accepted": False, "parent": str(parent.resolve()), "cohort": cohort,
                "parent_manifest_sha256": parent_hash, "parent_independent_checks_sha256": checks_hash, "bindings": code,
                "repair_bindings": repair_code, "new_models_fitted": False, "new_SVI_labels_produced": False,
                "training_ready": False, "shards": [], "started_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "workers": 3,
                                "CPU_budget": 4, "GPU_used": False, "numeric_threads": {name: os.environ.get(name) for name in
                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}}
    snapshot_paths = list((ROOT / "studies/observable_state_20261002").glob("*.py"))
    snapshot_paths += list(Path(__file__).parent.glob("*.py"))
    snapshot = output / "OBSERVABLE_CODE_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(snapshot_paths):
            archive.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
    manifest["code_snapshot_sha256"] = file_sha256(snapshot)
    groups = {role: [entry for entry in entries if entry["role"] == role] for role in ("TRAIN", "VALIDATION", "TEST")}
    counts, independent = Counter(), Counter()
    train_rows, supports = [], {role: Counter() for role in groups}
    start = time.perf_counter()
    try:
        tasks, offsets = [], {role: 0 for role in groups}
        for record in sorted(original["shards"], key=lambda r: r["directory"]):
            role = record["role"]
            begin = offsets[role]
            selected = groups[role][begin:begin + record["counts"]["n_matches"]]
            offsets[role] += len(selected)
            tasks.append((parent / record["directory"], output / record["directory"], selected))
        if any(offsets[role] != len(entries) for role, entries in groups.items()):
            raise ValueError("parent shards do not cover each frozen role")
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=3, mp_context=context, initializer=_initialize, initargs=(context.Lock(),)) as pool:
            iterator, pending = iter(tasks), set()
            for _ in range(3):
                task = next(iterator, None)
                if task:
                    pending.add(pool.submit(_repair_task, task))
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    result, verified = future.result()
                    manifest["shards"].append(result)
                    counts.update(result["counts"])
                    independent.update(verified["counts"])
                    train_rows.extend(verified["train_rows"])
                    supports[result["role"]].update(verified["support"])
                    manifest["counts"] = dict(counts)
                    (output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({"phase": "static_preservation", "matches": counts["n_matches"],
                                      "negative_ban_ids_preserved": counts["n_negative_cached_ban_ids"]}), flush=True)
                    task = next(iterator, None)
                    if task:
                        pending.add(pool.submit(_repair_task, task))
        if (counts["n_matches"] != len(entries) or dict(independent) != parent_checks["counts"] or
                any(original["counts"][key] != value for key, value in independent.items())):
            raise ValueError("corrected full-corpus cardinalities changed")
        reconciliation = _reconcile_train(train_rows)
        if reconciliation["counts"]["n_candidate_changed"]:
            raise ValueError("corrected static preservation changed TRAIN population")
        (output / "TRAIN_RECONCILIATION.json").write_text(json.dumps(reconciliation, indent=2) + "\n", encoding="utf-8")
        (output / "FIELD_SUPPORT_COUNTS.json").write_text(json.dumps({"full_corpus": True,
             "role_counts": {role: dict(value) for role, value in supports.items()}, "raw_field_presence_verified": False,
             "negative_cached_IDs_preserved": True}, indent=2) + "\n", encoding="utf-8")
        if code != bindings() or repair_code != repair_bindings() or manifest["code_snapshot_sha256"] != file_sha256(snapshot):
            raise ValueError("corrected implementation changed during execution")
        if (parent_hash != file_sha256(parent / "RUN_MANIFEST.json") or
                checks_hash != file_sha256(parent / "INDEPENDENT_CHECKS.json")):
            raise ValueError("completed parent manifest changed during correction")
        _, final_cohort = load_frozen_entries()
        if cohort != final_cohort:
            raise ValueError("final frozen cohort binding changed")
        schema = json.loads((parent / "FIELD_MANIFEST.json").read_text(encoding="utf-8"))
        schema["state_generator_complete"] = True
        (output / "FIELD_MANIFEST.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        manifest.update(complete=True, accepted=True, independent_counts=dict(independent),
                        field_manifest_sha256=file_sha256(output / "FIELD_MANIFEST.json"),
                        field_support_counts_sha256=file_sha256(output / "FIELD_SUPPORT_COUNTS.json"),
                        train_reconciliation_sha256=file_sha256(output / "TRAIN_RECONCILIATION.json"),
                        elapsed_seconds=time.perf_counter() - start,
                        completed_utc=datetime.now(timezone.utc).isoformat(),
                        static_source_and_complete_payload_reload_verified=True,
                        non_static_observable_payloads_unchanged=True)
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
