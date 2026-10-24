"""Supervisor and bounded worker runner for the immutable cache audit."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
from io import StringIO
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

from . import config as C
from .audit import AggregateAccumulator, audit_match_files
from .champion_map import source_map_binding


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_contract_binding() -> dict[str, Any]:
    """Load the immutable declaration order without importing project code globally."""
    config_path = C.ROOT / "worktrees" / "engagement-state-value" / "core" / "config.py"
    feature_path = config_path.with_name("feature_contract.py")
    base = {
        "ok": False,
        "config_path": config_path.relative_to(C.ROOT).as_posix(),
        "feature_contract_path": feature_path.relative_to(C.ROOT).as_posix(),
        "config_sha256": _sha256_file(config_path) if config_path.is_file() else None,
        "feature_contract_sha256": _sha256_file(feature_path) if feature_path.is_file() else None,
        "name_lists": {},
        "index_to_name": {},
        "name_to_index": {},
    }
    if not config_path.is_file() or not feature_path.is_file():
        base["error"] = "feature_declaration_source_missing"
        return base
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.get(name) for name in ("core", "core.config", "core.feature_contract")}
    try:
        sys.path.insert(0, str(config_path.parent.parent))
        importlib.invalidate_caches()
        for name in saved_modules:
            sys.modules.pop(name, None)
        module = importlib.import_module("core.config")
        names = {
            "node_minute": [str(name) for name in module.NODE_FEATURE_NAMES],
            "global_minute": [str(name) for name in module.GLOBAL_FEATURE_NAMES],
        }
        valid = len(names["node_minute"]) == 76 and len(names["global_minute"]) == 26
        valid = valid and all(len(set(values)) == len(values) for values in names.values())
        base["name_lists"] = names
        base["index_to_name"] = {
            field: {str(index): name for index, name in enumerate(values)}
            for field, values in names.items()
        }
        base["name_to_index"] = {
            field: {name: index for index, name in enumerate(values)}
            for field, values in names.items()
        }
        base["ok"] = bool(valid)
        if not valid:
            base["error"] = "feature_declaration_dimension_or_duplicate_mismatch"
    except Exception as exc:  # noqa: BLE001 - preserve binding evidence in manifest
        base["error"] = f"feature_declaration_import_error:{type(exc).__name__}"
    finally:
        sys.path[:] = saved_path
        for name, value in saved_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return base


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _code_binding() -> dict[str, Any]:
    files = sorted(path for path in C.STUDY.rglob("*.py") if "__pycache__" not in path.parts)
    implementation = {
        path.relative_to(C.ROOT).as_posix(): _sha256_file(path)
        for path in files
    }
    helper = C.ROOT / "studies" / "anchor_frame_audit_20261002" / "contracts.py"
    helper_hashes = {
        helper.relative_to(C.ROOT).as_posix(): _sha256_file(helper) if helper.is_file() else None
    }
    feature_binding = _feature_contract_binding()
    digest = hashlib.sha256()
    source_hashes = {
        **implementation,
        **helper_hashes,
        feature_binding["config_path"]: feature_binding["config_sha256"],
        feature_binding["feature_contract_path"]: feature_binding["feature_contract_sha256"],
    }
    for name, value in sorted(source_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value).encode("ascii"))
    return {
        "implementation_sha256": implementation,
        "helper_sha256": helper_hashes,
        "feature_contract": feature_binding,
        "code_binding_sha256": digest.hexdigest(),
    }


def _load_index(path: Path) -> tuple[bytes, dict[str, Any], list[tuple[str, str]]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("patch index must be a flat JSON object")
    entries: list[tuple[str, str]] = []
    for match_id, patch in payload.items():
        if not isinstance(match_id, str) or not match_id.strip() or not isinstance(patch, str):
            raise ValueError("patch index contains a non-string match key or patch")
        entries.append((match_id, patch))
    patch_counts: dict[str, int] = {}
    for _, patch in entries:
        patch_counts[patch] = patch_counts.get(patch, 0) + 1
    info = {
        "file_name": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_entries": len(entries),
        "patch_counts": dict(sorted(patch_counts.items())),
        "expected_sha256": C.EXPECTED_INDEX_SHA256,
        "expected_patch_counts": dict(C.EXPECTED_PATCH_COUNTS),
    }
    return raw, info, entries


def _load_fixed_cohort_manifest(path: Path, entries: list[tuple[str, str]]) -> dict[str, Any]:
    """Verify the fixed split manifest without retaining its raw rows."""
    required = {
        "match_id", "patch", "patch_full", "role", "has_npz", "has_meta", "has_events",
        "checked", "meta_ok", "meta_id_matches_stem", "npz_zip_ok", "events_brackets_ok",
    }
    info: dict[str, Any] = {
        "file_name": path.name,
        "expected_sha256": C.EXPECTED_COHORT_MANIFEST_SHA256,
        "sha256": None,
        "present": path.is_file(),
        "required_columns_ok": False,
        "n_rows": 0,
        "identity": {
            "missing_index_rows": None,
            "extra_csv_rows": None,
            "duplicate_csv_ids": 0,
            "patch_mismatches": 0,
            "role_mismatches": 0,
            "patch_full_mismatches": 0,
        },
        "triplet_diagnostic_counts": {},
        "production_identity_ok": False,
    }
    if not path.is_file():
        return info
    raw = path.read_bytes()
    info["sha256"] = hashlib.sha256(raw).hexdigest()
    reader = csv.DictReader(StringIO(raw.decode("utf-8-sig")))
    fields = set(reader.fieldnames or [])
    info["required_columns_ok"] = required <= fields
    if not info["required_columns_ok"]:
        info["missing_columns"] = sorted(required - fields)
        return info
    expected = {match_id: patch for match_id, patch in entries}
    seen: set[str] = set()
    bad_triplet: Counter[str] = Counter()
    for record in reader:
        info["n_rows"] += 1
        match_id = str(record.get("match_id") or "")
        patch = str(record.get("patch") or "")
        role = str(record.get("role") or "")
        expected_patch = expected.get(match_id)
        if match_id in seen:
            info["identity"]["duplicate_csv_ids"] += 1
        seen.add(match_id)
        if expected_patch is None:
            continue
        if patch != expected_patch:
            info["identity"]["patch_mismatches"] += 1
        if role != C.PATCH_ROLES.get(patch, "UNKNOWN"):
            info["identity"]["role_mismatches"] += 1
        patch_full = str(record.get("patch_full") or "")
        if not patch_full.startswith(patch + "."):
            info["identity"]["patch_full_mismatches"] += 1
        for key in ("has_npz", "has_meta", "has_events", "checked", "meta_ok", "meta_id_matches_stem", "npz_zip_ok", "events_brackets_ok"):
            if str(record.get(key) or "") != "1":
                bad_triplet[key] += 1
    info["identity"]["missing_index_rows"] = len(set(expected) - seen)
    info["identity"]["extra_csv_rows"] = len(seen - set(expected))
    info["triplet_diagnostic_counts"] = {key: int(value) for key, value in sorted(bad_triplet.items())}
    info["production_identity_ok"] = bool(
        info["sha256"] == C.EXPECTED_COHORT_MANIFEST_SHA256
        and info["n_rows"] == len(entries)
        and info["identity"]["missing_index_rows"] == 0
        and info["identity"]["extra_csv_rows"] == 0
        and info["identity"]["duplicate_csv_ids"] == 0
        and info["identity"]["patch_mismatches"] == 0
        and info["identity"]["role_mismatches"] == 0
        and info["identity"]["patch_full_mismatches"] == 0
        and not bad_triplet
    )
    return info


def _failed_worker_row(match_id: str, patch: str, error: Exception) -> dict[str, Any]:
    from .audit import audit_match_bytes

    row = audit_match_bytes(
        match_id,
        patch,
        npz_bytes=None,
        meta_bytes=None,
        events_bytes=None,
        source_stability=None,
    )
    row["status"] = "fail"
    row["reasons"] = sorted(set(row.get("reasons", [])) | {f"worker_exception:{type(error).__name__}"})
    row["worker_error"] = type(error).__name__
    return row


def _worker(payload: tuple[str, str, str]) -> dict[str, Any]:
    match_id, patch, cache_dir = payload
    try:
        return audit_match_files(match_id, patch, Path(cache_dir))
    except Exception as exc:  # noqa: BLE001 - keep one failed row and continue the batch
        return _failed_worker_row(match_id, patch, exc)


def _ledger_record(row: Mapping[str, Any], row_no: int) -> dict[str, Any]:
    npz = row.get("npz", {}) or {}
    frame = npz.get("frame_clock", {}) or {}
    events = row.get("events", {}) or {}
    ge = events.get("game_end", {}) or {}
    terminal = events.get("terminal_frame_order", {}) or {}
    champion = row.get("static_champion_consistency", {}) or {}
    kill = events.get("kill_support", {}) or {}
    kill_numeric = {
        str(key): int(value)
        for key, value in (kill.get("counts", {}) or {}).items()
        if type(value) is int
    }
    for key in ("n_kills",):
        if type(kill.get(key)) is int:
            kill_numeric[key] = int(kill[key])
    return {
        "row_no": int(row_no),
        "match_key_sha256": row.get("match_key_sha256", ""),
        "role": row.get("role", "UNKNOWN"),
        "status": row.get("status", "unknown"),
        "reasons": list(row.get("reasons", [])),
        "source_hashes": dict(row.get("source_hashes", {})),
        "numbers": {
            "n_frames": frame.get("n_frames"),
            "first_frame_ms": frame.get("min_ms"),
            "last_frame_ms": frame.get("max_ms"),
            "n_events": events.get("n_events"),
            "n_kills": (events.get("kill_support", {}) or {}).get("n_kills"),
            "n_game_end_valid": ge.get("n_valid"),
            "game_end_ms": ge.get("timestamp_ms"),
            "game_end_winning_team": ge.get("winning_team"),
            "terminal_gap_ms": terminal.get("terminal_gap_ms"),
            "n_game_end_objects": ge.get("n_objects"),
            "n_game_end_malformed_timestamp": ge.get("n_malformed_timestamp"),
            "n_game_end_malformed_winner": ge.get("n_malformed_winner"),
            "n_position_pairs_finite": (npz.get("stored_positions", {}) or {}).get("n_pairs_finite"),
            "n_gold_values_finite": (npz.get("gold_values", {}) or {}).get("n_values_finite"),
            "champion_consistency": {
                str(key): int(value)
                for key, value in champion.items()
                if key.startswith("n_") and type(value) is int
            },
            "kill_support": kill_numeric,
        },
    }


def _runtime_info(workers: int) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "workers": int(workers),
        "max_workers_policy": C.MAX_WORKERS,
        "threading_policy": "one supervisor plus at most three process workers; no nested worker pool",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def run_audit(
    *,
    index_path: Path = C.INDEX_PATH,
    cache_dir: Path = C.CACHE_DIR,
    output_dir: Path | None = None,
    workers: int = C.MAX_WORKERS,
    max_matches: int | None = None,
    strict_index: bool = True,
) -> dict[str, Any]:
    """Run a full or explicitly bounded audit into a new workspace directory."""
    output_dir = Path(output_dir) if output_dir is not None else C.new_run_dir()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    log_path = output_dir / "run.log"
    status_path = output_dir / "STATUS.json"

    def log(message: str) -> None:
        line = f"[{_utc()}] {message}"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    worker_count = max(1, min(C.MAX_WORKERS, int(workers)))
    status: dict[str, Any] = {
        "run_status": "running",
        "started_utc": _utc(),
        "output_dir": str(output_dir),
        "training_ready": False,
        "final_ready": False,
        "workers": worker_count,
        "max_matches": max_matches,
    }
    _write_json(status_path, status)
    log(f"start workers={worker_count} max_matches={max_matches}")

    binding = _code_binding()
    source_map = source_map_binding()
    accumulator = AggregateAccumulator()
    support = {
        "rows_target": 0,
        "rows_processed": 0,
        "rows_with_three_source_hashes": 0,
        "rows_source_unstable": 0,
        "rows_with_fail_status": 0,
        "full_corpus_target": False,
    }
    ledger_path = output_dir / "row_ledger.jsonl"
    try:
        _, index_info, entries = _load_index(Path(index_path))
        cohort_info = (
            _load_fixed_cohort_manifest(C.FIXED_COHORT_MANIFEST_PATH, entries)
            if strict_index
            else {"present": False, "production_identity_ok": False, "skipped": "nonproduction_index"}
        )
        index_ok = (
            index_info["sha256"] == C.EXPECTED_INDEX_SHA256
            and index_info["n_entries"] == C.EXPECTED_INDEX_COUNT
            and index_info["patch_counts"] == C.EXPECTED_PATCH_COUNTS
            and cohort_info["production_identity_ok"]
        )
        index_info["production_identity_ok"] = bool(index_ok)
        if strict_index and (
            not index_ok
            or not binding["feature_contract"].get("ok")
            or not source_map.get("production_identity_ok")
        ):
            raise RuntimeError("production input or feature declaration identity check failed")
        entries = sorted(entries, key=lambda pair: (C.PATCH_ROLES.get(pair[1], "UNKNOWN"), pair[0].strip()))
        target_entries = entries if max_matches is None else entries[: int(max_matches)]
        if max_matches is not None and (isinstance(max_matches, bool) or int(max_matches) <= 0):
            raise ValueError("max_matches must be a positive integer")
        support["rows_target"] = len(target_entries)
        support["full_corpus_target"] = max_matches is None
        _write_json(output_dir / "input_identity.json", {"patch_index": index_info, "fixed_cohort_manifest": cohort_info})
        log(f"index entries={len(entries)} target={len(target_entries)} production_identity_ok={index_ok}")

        pool = ProcessPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
        try:
            with ledger_path.open("w", encoding="utf-8", newline="\n") as ledger:
                for begin in range(0, len(target_entries), C.BATCH_SIZE):
                    batch = target_entries[begin : begin + C.BATCH_SIZE]
                    payloads = [(match_id, patch, str(cache_dir)) for match_id, patch in batch]
                    if pool is None:
                        rows = [_worker(payload) for payload in payloads]
                    else:
                        rows = list(pool.map(_worker, payloads))
                    for offset, row in enumerate(rows):
                        accumulator.add(row)
                        support["rows_processed"] += 1
                        if all(row.get("source_hashes", {}).get(name) for name in ("npz", "meta", "events")):
                            support["rows_with_three_source_hashes"] += 1
                        if row.get("source_files_stable") is False:
                            support["rows_source_unstable"] += 1
                        if row.get("status") == "fail":
                            support["rows_with_fail_status"] += 1
                        ledger.write(json.dumps(_ledger_record(row, begin + offset), ensure_ascii=False, sort_keys=True) + "\n")
                    ledger.flush()
                    progress = {
                        "rows_processed": support["rows_processed"],
                        "rows_target": support["rows_target"],
                        "batch_start": begin,
                        "batch_size": len(batch),
                        "status_counts": dict(sorted(accumulator.status_counts.items())),
                        "utc": _utc(),
                    }
                    _write_json(output_dir / "progress.json", progress)
                    log(f"progress {support['rows_processed']}/{support['rows_target']} status_counts={progress['status_counts']}")
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=False)

        aggregate = accumulator.finish(
            expected_index_counts=C.EXPECTED_PATCH_COUNTS,
            input_support=support,
            feature_binding=binding["feature_contract"],
        )
        _write_json(output_dir / "aggregate.json", aggregate)
        contract_hash = _sha256_file(C.POST_STATE_CONTRACT_PATH) if C.POST_STATE_CONTRACT_PATH.is_file() else None
        scope_hash = _sha256_file(C.AUDIT_SCOPE_PATH) if C.AUDIT_SCOPE_PATH.is_file() else None
        manifest: dict[str, Any] = {
            "run_status": "completed",
            "completed_utc": _utc(),
            "output_dir": str(output_dir),
            "input_index": index_info,
            "fixed_cohort_manifest": cohort_info,
            "source_map": source_map,
            "cache_source": "private immutable cache path configured by this study",
            "source_file_read_policy": "each indexed row reads npz, meta.json and events.json once; parse uses the read bytes",
            "code_binding": binding,
            "post_state_contract_sha256": contract_hash,
            "audit_scope_sha256": scope_hash,
            "runtime": _runtime_info(worker_count),
            "aggregate": aggregate,
            "training_ready": False,
            "final_ready": False,
            "full_corpus_scanned": bool(max_matches is None and support["rows_processed"] == len(entries)),
            "max_matches": max_matches,
            "artifacts": ["STATUS.json", "RUN_MANIFEST.json", "aggregate.json", "input_identity.json", "progress.json", "row_ledger.jsonl", "run.log"],
        }
        _write_json(output_dir / "RUN_MANIFEST.json", manifest)
        status.update({
            "run_status": "completed",
            "finished_utc": _utc(),
            "rows_processed": support["rows_processed"],
            "full_corpus_scanned": manifest["full_corpus_scanned"],
            "manifest_sha256": _sha256_file(output_dir / "RUN_MANIFEST.json"),
        })
        _write_json(status_path, status)
        log(f"finished rows={support['rows_processed']} full_corpus_scanned={manifest['full_corpus_scanned']}")
        return manifest
    except Exception as exc:  # noqa: BLE001 - preserve run status and partial ledger
        status.update({
            "run_status": "failed",
            "finished_utc": _utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rows_processed": support["rows_processed"],
        })
        _write_json(status_path, status)
        failure = {
            "run_status": "failed",
            "failed_utc": _utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "code_binding": binding,
            "source_map": source_map,
            "training_ready": False,
            "final_ready": False,
        }
        _write_json(output_dir / "RUN_MANIFEST.json", failure)
        log(f"failed error_type={type(exc).__name__}")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=C.INDEX_PATH)
    parser.add_argument("--cache-dir", type=Path, default=C.CACHE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=C.MAX_WORKERS)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--allow-nonproduction-index", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = run_audit(
            index_path=args.index,
            cache_dir=args.cache_dir,
            output_dir=args.output,
            workers=args.workers,
            max_matches=args.max_matches,
            strict_index=not args.allow_nonproduction_index,
        )
    except Exception as exc:  # noqa: BLE001 - status file is already preserved
        print(f"cache source audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_status": manifest["run_status"], "output_dir": manifest["output_dir"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
