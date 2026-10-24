"""E2a stage 'audit': frozen cache source audit per dataset, then the external entry list the bank consumes.

README
  Frozen code: studies/cache_source_audit_20261002/run_audit._worker (= audit.audit_match_files with the run's
  failed-row fallback) and run_audit._ledger_record, unchanged.  The audit reads the three cache files of a match once,
  hashes them, and records the numbers the observable producer re-checks (n_frames, n_events, n_kills, game_end_ms,
  game_end_winning_team).  Only adaptation: the audit row's role, which the frozen code derives from the 2025
  PATCH_ROLES table (TEST for 15.16, UNKNOWN otherwise), and its canonical_key ("<role>:<key>") are relabelled to
  EXTERNAL before _ledger_record.  The frozen champion-consistency diagnostic has maps only for 15.14-15.16, so for later
  patches it reports 'map unavailable' (status 'unknown', never critical).

  Entry rule (mirrors observable_state_20261002.source.load_frozen_entries without the 2025 cohort/role gates):
  every audited match whose status is not 'fail' becomes an entry {match_id, canonical_match_key, patch,
  role EXTERNAL, dataset, original_index, source_hashes, audited_numbers}; 'fail' rows (critical source issues, e.g.
  missing or ambiguous GAME_END, broken timestamps) are excluded and listed with their reasons.

Outputs (stage_dir(dataset, 'audit')): row_ledger.jsonl (index order, row_no = position), AUDIT_MANIFEST.json.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402

LEDGER_NUMBERS = ("n_frames", "n_events", "n_kills", "game_end_ms", "game_end_winning_team")


def dataset_matches(dataset: str) -> list[tuple[str, str, int]]:
    """(match_id, patch, original_index) in patch-index order."""
    if dataset == C.DATASET_PARITY:
        subset = C.read_json(C.PARITY_SUBSET)
        index = C.read_json(C.REF_PATCH_INDEX_2025)
        pairs = list(index.items())
        out = []
        for row in sorted(subset["matches"], key=lambda r: r["original_index"]):
            mid, patch = pairs[row["original_index"]]
            if mid != row["match_id"] or patch != C.PARITY_PATCH or C.canonical_key(mid) != row["canonical_match_key"]:
                raise ValueError("parity subset row differs from the 2025 patch index")
            out.append((mid, patch, int(row["original_index"])))
        return out
    index = C.read_json(C.PATCH_INDEX[dataset])
    out = [(mid, patch, position) for position, (mid, patch) in enumerate(index.items())]
    bad = sorted({patch for _, patch, _ in out} - set(C.EXPECTED_PATCHES[dataset]))
    if bad:
        raise ValueError(f"{dataset}: unexpected patches in the patch index {bad}")
    return out


def _init_worker() -> None:
    C.runtime_hygiene()


def _audit_task(task: tuple[str, str, str, int]) -> dict:
    from studies.cache_source_audit_20261002.run_audit import _ledger_record, _worker

    match_id, patch, cache_dir, row_no = task
    row = _worker((match_id, patch, cache_dir))
    frozen_role = row.get("role")
    row["role"] = C.EXTERNAL_ROLE
    row["canonical_key"] = f"{C.EXTERNAL_ROLE}:{row['match_key_sha256']}"
    record = _ledger_record(row, row_no)
    return {"record": record, "frozen_role": frozen_role, "critical_reason_count": row.get("critical_reason_count"),
            "worker_error": row.get("worker_error")}


def run_audit(dataset: str, workers: int, log) -> dict:
    out_dir = C.stage_dir(dataset, "audit")
    manifest_path = out_dir / "AUDIT_MANIFEST.json"
    if manifest_path.exists():
        manifest = C.read_json(manifest_path)
        if manifest.get("complete"):
            log(f"[audit] {dataset}: complete audit exists, reused")
            return manifest
        raise FileExistsError(f"incomplete audit at {out_dir}; remove it explicitly before re-running")
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = dataset_matches(dataset)
    cache = str(C.CACHE[dataset])
    tasks = [(mid, patch, cache, row_no) for row_no, (mid, patch, _oi) in enumerate(matches)]
    started = time.perf_counter()
    status, reasons, frozen_roles = Counter(), Counter(), Counter()
    ledger_tmp = out_dir / "row_ledger.jsonl.partial"
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool, \
            ledger_tmp.open("w", encoding="utf-8", newline="\n") as ledger:
        for done, result in enumerate(pool.map(_audit_task, tasks, chunksize=64), 1):
            record = result["record"]
            status[record["status"]] += 1
            reasons.update(record["reasons"])
            frozen_roles[str(result["frozen_role"])] += 1
            ledger.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if done % 5000 == 0:
                log(f"[audit] {dataset}: {done}/{len(tasks)} status={dict(status)}")
    ledger_path = out_dir / "row_ledger.jsonl"
    ledger_tmp.replace(ledger_path)
    manifest = {
        "version": C.VERSION, "stage": "audit", "dataset": dataset, "complete": True,
        "cache_dir": cache, "patch_index": str(C.PATCH_INDEX[dataset]),
        "patch_index_sha256": C.sha256_file(C.PATCH_INDEX[dataset]),
        "parity_subset_sha256": C.sha256_file(C.PARITY_SUBSET) if dataset == C.DATASET_PARITY else None,
        "n_matches": len(tasks), "status_counts": dict(status), "reason_counts": dict(sorted(reasons.items())),
        "frozen_audit_role_before_relabel": dict(frozen_roles), "role": C.EXTERNAL_ROLE,
        "row_ledger_sha256": C.sha256_file(ledger_path), "workers": workers,
        "elapsed_seconds": round(time.perf_counter() - started, 1), "completed_utc": C.utc_now(),
        "frozen_calls": ["studies.cache_source_audit_20261002.run_audit._worker((match_id, patch, cache_dir))",
                         "studies.cache_source_audit_20261002.run_audit._ledger_record(row, row_no)"],
        "hook": C.FROZEN_HOOKS["cache_source_audit"],
    }
    C.write_json(manifest_path, manifest)
    log(f"[audit] {dataset}: {len(tasks)} matches status={dict(status)} in {manifest['elapsed_seconds']} s")
    return manifest


def load_external_entries(dataset: str) -> tuple[list[dict], list[dict], dict]:
    """(entries, excluded, binding) from the dataset audit; entries keep patch-index order."""
    out_dir = C.stage_dir(dataset, "audit")
    manifest = C.read_json(out_dir / "AUDIT_MANIFEST.json")
    ledger_path = out_dir / "row_ledger.jsonl"
    if not manifest.get("complete") or C.sha256_file(ledger_path) != manifest["row_ledger_sha256"]:
        raise ValueError(f"{dataset}: audit ledger incomplete or changed")
    matches = dataset_matches(dataset)
    entries, excluded = [], []
    with ledger_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != len(matches):
        raise ValueError(f"{dataset}: audit ledger covers {len(rows)} of {len(matches)} matches")
    seen = set()
    for (mid, patch, original_index), row in zip(matches, rows):
        key = C.canonical_key(mid)
        if row["match_key_sha256"] != key or key in seen or row["role"] != C.EXTERNAL_ROLE:
            raise ValueError(f"{dataset}: audit ledger identity mismatch at {mid}")
        seen.add(key)
        if row["status"] == "fail":
            excluded.append({"match_id": mid, "canonical_match_key": key, "patch": patch, "stage": "audit",
                             "reason": "audit_status_fail", "audit_reasons": row["reasons"]})
            continue
        if set(row["source_hashes"]) != {"npz", "meta", "events"} or not all(row["source_hashes"].values()):
            raise ValueError(f"{dataset}: audit row without three source hashes {mid}")
        entries.append({"match_id": mid, "canonical_match_key": key, "patch": patch, "role": C.EXTERNAL_ROLE,
                        "dataset": dataset, "original_index": int(original_index),
                        "source_hashes": row["source_hashes"],
                        "audited_numbers": {name: row["numbers"][name] for name in LEDGER_NUMBERS}})
    binding = {"dataset": dataset, "role": C.EXTERNAL_ROLE, "audit_manifest_sha256": C.sha256_file(out_dir / "AUDIT_MANIFEST.json"),
               "audit_ledger_sha256": manifest["row_ledger_sha256"], "n_indexed": len(matches),
               "n_entries": len(entries), "n_excluded_audit": len(excluded),
               "patch_counts": dict(Counter(e["patch"] for e in entries))}
    return entries, excluded, binding
