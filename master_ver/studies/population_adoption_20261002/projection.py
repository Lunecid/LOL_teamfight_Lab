"""Hash-checked full-partition projection over the accepted observation run."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import config as C
from .contracts import PASS, UNKNOWN, evaluate_row, row_reason_text, validate_contract


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _execution_code_hashes() -> tuple[dict[str, str], dict[str, str]]:
    implementation: dict[str, str] = {}
    for path in sorted(C.STUDY.rglob("*.py")):
        implementation[str(path.relative_to(C.ROOT)).replace("\\", "/")] = file_sha256(path)
    dependency = {
        "numpy_version": np.__version__,
        "observation_support_io_source_sha256": file_sha256(
            C.ROOT / "studies" / "observation_support_20261002" / "io_source.py"
        ),
    }
    return implementation, dependency


def _status(value: Any) -> str:
    if value is None:
        return UNKNOWN
    if isinstance(value, (bool, np.bool_)):
        return PASS if bool(value) else "fail"
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return PASS if int(value) else "fail"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"pass", "passed", "true", "valid", "ok", "supported"}:
            return PASS
        if lowered in {"fail", "failed", "false", "invalid", "rejected"}:
            return "fail"
    return UNKNOWN


def _status_array(values: Any, n: int | None = None) -> np.ndarray:
    """Normalize scalar status fields without truthiness or float coercion."""
    array = np.asarray(values)
    if n is not None and array.shape != (n,):
        raise ValueError(f"expected one-dimensional status field of length {n}, got {array.shape}")
    result = np.full(array.shape, UNKNOWN, dtype="U7")
    if array.dtype.kind == "b":
        result[array] = PASS
        result[~array] = "fail"
    elif array.dtype.kind in "iu":
        result[array == 1] = PASS
        result[array == 0] = "fail"
    elif array.dtype.kind in "SUO":
        flat = array.reshape(-1)
        out = result.reshape(-1)
        for index, value in enumerate(flat.tolist()):
            if isinstance(value, str):
                text = value.strip().lower()
                if text in {"pass", "passed", "confirmed_opposition"}:
                    out[index] = PASS
                elif text in {"fail", "failed", "not_confirmed", "unconfirmed"}:
                    out[index] = "fail"
                elif text in {"unknown", "pending", "unavailable", "na", "null"}:
                    out[index] = UNKNOWN
    return result


def _integer_array(values: Any, n: int, *, nonnegative: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return strict integer values and a validity mask.

    Source NPZ fields are integer arrays.  Object/float values are kept
    unknown instead of being truncated or parsed permissively.
    """
    array = np.asarray(values)
    if array.shape != (n,):
        raise ValueError(f"expected one-dimensional integer field of length {n}, got {array.shape}")
    result = np.full(n, -1, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    if array.dtype.kind in "iu":
        result[:] = array.astype(np.int64, copy=False)
        valid[:] = True
        if nonnegative:
            valid &= result >= 0
    return result, valid


def _bool_array(values: Any) -> np.ndarray:
    return _status_array(values).reshape(-1) == PASS


def _as_text(values: Any, n: int, *, default: str = "") -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (n,):
        raise ValueError(f"expected one-dimensional field of length {n}, got {array.shape}")
    return array.astype(f"U{max(1, min(256, array.dtype.itemsize // 4 if array.dtype.kind == 'U' else 256))}")


def load_contract(path: Path = C.CONTRACT_PATH) -> tuple[dict[str, Any], str, dict[str, Any]]:
    contract_path = Path(path).resolve()
    contract = _read_json(contract_path)
    contract_hash = file_sha256(contract_path)
    validation = validate_contract(contract)
    if not validation["ok"]:
        raise ValueError("invalid population adoption contract: " + ", ".join(validation["errors"]))
    return contract, contract_hash, validation


def _expected_run_from_acceptance(acceptance_path: Path) -> Path:
    acceptance = _read_json(acceptance_path)
    run_dir = acceptance.get("run_dir")
    if not run_dir:
        raise ValueError("full diagnostic acceptance has no run_dir")
    return Path(str(run_dir)).resolve()


def validate_source_run(
    source_run: Path = C.DEFAULT_SOURCE_RUN,
    *,
    acceptance_path: Path = C.DIAGNOSTIC_ACCEPTANCE_PATH,
    hash_artifacts: bool = True,
) -> dict[str, Any]:
    """Verify the immutable diagnostic run and return its source hash ledger."""
    run = Path(source_run).resolve()
    acceptance_file = Path(acceptance_path).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"source run does not exist: {run}")
    acceptance = _read_json(acceptance_file)
    if acceptance.get("full_diagnostic_accepted") is not True:
        raise ValueError("full diagnostic acceptance is false")
    artifact_checks = acceptance.get("artifact_hash_checks") or {}
    if not artifact_checks or not all(v is True for v in artifact_checks.values()):
        raise ValueError("full diagnostic artifact hash checks are incomplete")
    accepted_run = _expected_run_from_acceptance(acceptance_file)
    if accepted_run != run:
        raise ValueError(f"source run differs from accepted run: {run} != {accepted_run}")
    manifest_path = run / "RUN_MANIFEST.json"
    manifest = _read_json(manifest_path)
    identity = manifest.get("identity") or {}
    if manifest.get("complete") is not True or identity.get("all_ok") is not True:
        raise ValueError("source diagnostic manifest is incomplete")
    if int(manifest.get("n_components", -1)) != C.EXPECTED_COMPONENTS:
        raise ValueError("source component count does not match the accepted TRAIN partition")
    if int(acceptance.get("components", -1)) != C.EXPECTED_COMPONENTS:
        raise ValueError("acceptance component count does not match the accepted TRAIN partition")
    paths = {name: run / name for name in C.SOURCE_ARTIFACTS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("source run missing artifacts: " + ", ".join(missing))

    actual_manifest_hash = file_sha256(manifest_path)
    acceptance_manifest_hash = str(acceptance.get("manifest_sha256") or "")
    if not acceptance_manifest_hash or actual_manifest_hash != acceptance_manifest_hash:
        raise ValueError("acceptance manifest_sha256 does not match RUN_MANIFEST.json")
    hashes: dict[str, str] = {
        "run_manifest_sha256": actual_manifest_hash,
        "full_diagnostic_acceptance_sha256": file_sha256(acceptance_file),
    }
    recorded = {
        "components": str(manifest.get("components_sha256") or ""),
        "evidence": str(manifest.get("evidence_sha256") or ""),
        "reference_fields": str(manifest.get("reference_fields_sha256") or ""),
    }
    validation_errors: list[str] = []
    if hash_artifacts:
        for key, filename in (("components", "components.npz"), ("evidence", "evidence.npz"), ("reference_fields", "reference_fields.npz")):
            if not recorded[key]:
                raise ValueError(f"source {filename} has no recorded sha256")
            actual = file_sha256(paths[filename])
            if actual != recorded[key]:
                raise ValueError(f"source {filename} sha256 mismatch")
            hashes[f"{key}_sha256"] = actual
    else:
        hashes.update({f"{key}_sha256": value for key, value in recorded.items()})
        validation_errors.append("artifact_hashes_not_recomputed_debug_only")
    source_hashes = manifest.get("source_hashes") or {}
    hashes["anchor_manifest_sha256"] = str(identity.get("manifest_sha256") or "")
    hashes["source_matches"] = str(acceptance.get("source_matches", ""))
    hashes["source_kills"] = str(acceptance.get("source_kills", ""))
    ledger_expected = {
        "ledger_sha256": str(identity.get("ledger_sha256") or ""),
        "ranges_sha256": str(identity.get("ranges_sha256") or ""),
        "match_meta_sha256": str(identity.get("match_meta_sha256") or ""),
    }
    ledger_files = {
        "ledger_sha256": C.SOURCE_LEDGER_DIR / "event_ledger_train.npz",
        "ranges_sha256": C.SOURCE_LEDGER_DIR / "ledger_match_ranges.npz",
        "match_meta_sha256": C.SOURCE_LEDGER_DIR / "train_match_meta.npz",
    }
    for key, path in ledger_files.items():
        expected = ledger_expected[key]
        if not expected or not path.is_file():
            raise ValueError(f"missing frozen source hash/file for {key}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"frozen source {path.name} sha256 mismatch")
        hashes[key] = actual
    definition_paths = {
        "post_state_contract_sha256": C.POST_STATE_CONTRACT_PATH,
        "radius_source_evidence_sha256": C.RADIUS_SOURCE_EVIDENCE_PATH,
        "input_policy_sha256": C.INPUT_POLICY_PATH,
        "population_policy_sha256": C.POPULATION_POLICY_PATH,
    }
    definition_hashes: dict[str, str] = {}
    for key, path in definition_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required parent definition is missing: {path}")
        definition_hashes[key] = file_sha256(path)
    ok = not validation_errors
    return {
        "ok": ok,
        "validation_errors": validation_errors,
        "hash_artifacts": bool(hash_artifacts),
        "source_run": str(run),
        "acceptance_path": str(acceptance_file),
        "manifest": manifest,
        "acceptance": acceptance,
        "source_manifest_hashes": hashes,
        "source_hashes": source_hashes,
        "definition_hashes": definition_hashes,
        "source_seed": manifest.get("seed"),
        "source_environment": manifest.get("machine") or manifest.get("environment"),
        "source_thread_environment": manifest.get("thread_limits") or {},
        "source_code_hashes": manifest.get("code_hashes") or {
            "run_code_sha256": str(manifest.get("code_sha256") or ""),
            "protocol_sha256": str(manifest.get("protocol_sha256") or ""),
            "source_census_sha256": str(manifest.get("source_census_sha256") or ""),
        },
        "n_matches": int(acceptance.get("source_matches", manifest.get("n_matches_processed", -1))),
        "n_components": int(manifest.get("n_components", -1)),
    }


def _required_component_keys() -> tuple[str, ...]:
    return (
        "match_local", "component_id", "anon_match_key", "member_key", "f", "L", "tau", "F_tau",
        "team_map_ok", "c0_bits", "c0_n_blue", "c0_n_red", "c0_n_unknown_team", "c0_n_credited",
        "c0_n_env_killer", "participation_mode", "raw_presence_verified",
        "cand_10_20_tau", "lag_f_F_tau", "foreign_kills_in_tau_L", "other_anchors_in_tau_L",
        "frame_support_ok", "terminal_support_authoritative", "events_support_ok", "valid_h90",
    ) + tuple(
        f"{field}_tau_r{radius}"
        for radius in (C.PRIMARY_RADIUS, *C.SENSITIVITY_RADII)
        for field in ("support_ok", "p_lower_n", "p_upper_n")
    )


def _load_projection_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in _required_component_keys() if key not in archive.files]
        if missing:
            raise ValueError("source components.npz missing fields: " + ", ".join(missing))
        return {key: np.asarray(archive[key]) for key in _required_component_keys()}


def _participant_team(pid: Any, blue_pids: np.ndarray) -> int | None:
    """Return an actual team for a valid player ID; actor 0 is not a player."""
    try:
        participant = int(pid)
    except (TypeError, ValueError, OverflowError):
        return None
    if not 1 <= participant <= 10:
        return None
    return 100 if bool(np.any(blue_pids == participant)) else 200


def _bounded_sidecar_header(path: Path) -> dict[str, Any] | None:
    """Read the small sidecar identity fields without loading its huge body."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(256 << 10).decode("utf-8", errors="ignore")
    except OSError:
        return None

    def grab_text(key: str) -> str | None:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', prefix)
        return match.group(1) if match else None

    def grab_int(key: str) -> int | None:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+)', prefix)
        return int(match.group(1)) if match else None

    npz_hash = grab_text("npz_sha256")
    event_rows = grab_int("n_event_rows")
    source_kills = grab_text("source_kills_sha256")
    source_manifest = grab_text("source_manifest_sha256")
    if npz_hash is None or event_rows is None or source_kills is None or source_manifest is None:
        return None
    return {
        "npz_sha256": npz_hash,
        "n_event_rows": event_rows,
        "source_kills_sha256": source_kills,
        "source_manifest_sha256": source_manifest,
    }


def _derive_direct_opposition(
    component_arrays: Mapping[str, np.ndarray],
    *,
    ledger_dir: Path = C.SOURCE_LEDGER_DIR,
    batch_dir: Path = C.SOURCE_BATCH_DIR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Join frozen component labels to every same-time first-kill member event.

    C0 team counts are necessary but do not establish an opposing interaction:
    two environment deaths can produce credits on both teams.  The join below
    checks the killer/victim relation and every recorded assist in the complete
    first-kill block.  It uses only the already hashed kill ledger, labels, and
    actual blue-pid metadata; cache frames are never reread.
    """
    n = int(component_arrays["f"].shape[0])
    unknown = np.full(n, UNKNOWN, dtype="U7")
    relation_kind = np.full(n, "unknown", dtype="U9")
    credit_parse_status = np.full(n, UNKNOWN, dtype="U7")
    required = (
        ledger_dir / "event_ledger_train.npz",
        ledger_dir / "ledger_match_ranges.npz",
        ledger_dir / "train_match_meta.npz",
    )
    batch_paths = sorted(batch_dir.glob("batch_*.npz")) if batch_dir.is_dir() else []
    if not all(path.is_file() for path in required) or not batch_paths:
        return unknown, {
            "status": UNKNOWN,
            "reason": "direct_firstkill_source_unavailable",
            "n_covered": 0,
            "relation_kind": relation_kind,
            "credit_parse_status": credit_parse_status,
        }
    try:
        with np.load(required[0], allow_pickle=False) as ledger_archive:
            ledger_fields = {key: np.asarray(ledger_archive[key]) for key in ("t_ms", "killer", "victim", "assist_mask")}
        with np.load(required[1], allow_pickle=False) as ranges_archive:
            ranges_fields = {key: np.asarray(ranges_archive[key]) for key in ("starts", "ends")}
        with np.load(required[2], allow_pickle=False) as meta_archive:
            blue_pids = np.asarray(meta_archive["blue_pids"])
    except (KeyError, OSError, ValueError) as exc:
        return unknown, {
            "status": UNKNOWN,
            "reason": f"direct_firstkill_source_invalid:{type(exc).__name__}",
            "n_covered": 0,
            "relation_kind": relation_kind,
            "credit_parse_status": credit_parse_status,
        }

    if any(array.dtype.kind not in "iu" or array.dtype.kind == "b" for array in ledger_fields.values()):
        return unknown, {
            "status": UNKNOWN,
            "reason": "direct_firstkill_source_noninteger_ledger",
            "n_covered": 0,
            "relation_kind": relation_kind,
            "credit_parse_status": credit_parse_status,
        }
    if any(array.dtype.kind not in "iu" or array.dtype.kind == "b" for array in ranges_fields.values()):
        return unknown, {
            "status": UNKNOWN,
            "reason": "direct_firstkill_source_noninteger_ranges",
            "n_covered": 0,
            "relation_kind": relation_kind,
            "credit_parse_status": credit_parse_status,
        }
    if blue_pids.dtype.kind not in "iu" or blue_pids.dtype.kind == "b":
        return unknown, {
            "status": UNKNOWN,
            "reason": "direct_firstkill_source_noninteger_team_metadata",
            "n_covered": 0,
            "relation_kind": relation_kind,
            "credit_parse_status": credit_parse_status,
        }
    t_ms = ledger_fields["t_ms"].astype(np.int64, copy=False)
    killer = ledger_fields["killer"].astype(np.int64, copy=False)
    victim = ledger_fields["victim"].astype(np.int64, copy=False)
    assist = ledger_fields["assist_mask"].astype(np.int64, copy=False)
    starts = ranges_fields["starts"].astype(np.int64, copy=False)
    ends = ranges_fields["ends"].astype(np.int64, copy=False)
    if not (t_ms.shape == killer.shape == victim.shape == assist.shape):
        return unknown, {
            "status": UNKNOWN,
            "reason": "direct_firstkill_source_ledger_shape_mismatch",
            "n_covered": 0,
            "relation_kind": relation_kind,
            "credit_parse_status": credit_parse_status,
        }
    if starts.shape != ends.shape or blue_pids.ndim != 2 or blue_pids.shape[0] != starts.size:
        return unknown, {
            "status": UNKNOWN,
            "reason": "direct_firstkill_source_range_team_shape_mismatch",
            "n_covered": 0,
            "relation_kind": relation_kind,
            "credit_parse_status": credit_parse_status,
        }
    team_metadata_ok = (
        blue_pids.shape[1] == 5
        and np.all((blue_pids >= 1) & (blue_pids <= 10))
        and np.all(np.diff(np.sort(blue_pids.astype(np.int64), axis=1), axis=1) != 0)
    )
    if not team_metadata_ok:
        return unknown, {
            "status": UNKNOWN,
            "reason": "direct_firstkill_source_team_metadata_invalid",
            "n_covered": 0,
            "relation_kind": relation_kind,
            "credit_parse_status": credit_parse_status,
        }

    production_sidecars = batch_dir.resolve() == C.SOURCE_BATCH_DIR.resolve()
    batch_records: list[dict[str, Any]] = []
    sidecar_identity_ok = True
    sidecar_hashes: list[str] = []
    label_identity_hashes: dict[str, str] = {}
    if production_sidecars:
        try:
            expected_ids = [int(path.stem.split("_")[1]) for path in batch_paths]
            sidecar_paths = [batch_dir / f"batch_{batch_id:04d}.json" for batch_id in expected_ids]
            if not all(path.is_file() for path in sidecar_paths):
                sidecar_identity_ok = False
            identity_path = batch_dir.parent / "IDENTITY.json"
            status_path = batch_dir.parent / "RUN_STATUS.json"
            if not identity_path.is_file() or not status_path.is_file():
                sidecar_identity_ok = False
            else:
                identity = _read_json(identity_path)
                run_status = _read_json(status_path)
                sidecar_identity_ok &= (
                    identity.get("ledger_sha256") == C.SOURCE_LEDGER_SHA256
                    and identity.get("ranges_sha256") == C.SOURCE_RANGES_SHA256
                    and identity.get("match_meta_sha256") == C.SOURCE_MATCH_META_SHA256
                    and identity.get("source_manifest_sha256") == C.EXPECTED_LABEL_SOURCE_MANIFEST_SHA256
                    and run_status.get("complete") is True
                    and int(run_status.get("n_batches_total", -1)) == len(expected_ids)
                    and int(run_status.get("n_batches_committed_valid", -1)) == len(expected_ids)
                )
                label_identity_hashes = {
                    "identity_sha256": file_sha256(identity_path),
                    "run_status_sha256": file_sha256(status_path),
                }
            for batch_id, sidecar_path, batch_path in zip(expected_ids, sidecar_paths, batch_paths):
                header = _bounded_sidecar_header(sidecar_path)
                if header is None:
                    sidecar_identity_ok = False
                    continue
                actual_hash = file_sha256(batch_path)
                sidecar_hashes.append(f"{batch_id}:{actual_hash}")
                sidecar_identity_ok &= (
                    header["npz_sha256"] == actual_hash
                    and header["source_kills_sha256"] == C.EXPECTED_KILLS_SHA256
                    and header["source_manifest_sha256"] == C.EXPECTED_LABEL_SOURCE_MANIFEST_SHA256
                )
                with np.load(batch_path, allow_pickle=False) as batch:
                    if "match_locals" not in batch.files or "labels_t14000_s2000_m1" not in batch.files:
                        sidecar_identity_ok = False
                        continue
                    batch_records.append({
                        "batch_id": batch_id,
                        "match_locals": np.asarray(batch["match_locals"]),
                        "labels": np.asarray(batch["labels_t14000_s2000_m1"]),
                        "n_event_rows": header["n_event_rows"],
                    })
        except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError):
            sidecar_identity_ok = False
    else:
        for batch_path in batch_paths:
            with np.load(batch_path, allow_pickle=False) as batch:
                if "match_locals" not in batch.files or "labels_t14000_s2000_m1" not in batch.files:
                    continue
                batch_records.append({
                    "batch_id": int(batch_path.stem.split("_")[1]),
                    "match_locals": np.asarray(batch["match_locals"]),
                    "labels": np.asarray(batch["labels_t14000_s2000_m1"]),
                    "n_event_rows": None,
                    "identity_source_kills": None,
                })
    if not batch_records:
        return unknown, {
            "status": UNKNOWN,
            "reason": "direct_firstkill_label_batches_unavailable",
            "n_covered": 0,
            "relation_kind": relation_kind,
            "credit_parse_status": credit_parse_status,
        }

    by_key = {
        (int(match), int(component)): index
        for index, (match, component) in enumerate(
            zip(component_arrays["match_local"], component_arrays["component_id"])
        )
    }
    duplicate_component_keys = n - len(by_key)
    member_key_ok = True
    if "member_key" in component_arrays:
        member_keys = np.asarray(component_arrays["member_key"]).astype("U256")
        match_values = np.asarray(component_arrays["match_local"])
        component_values = np.asarray(component_arrays["component_id"])
        expected_member_keys = np.asarray(
            [f"m{int(match):05d}:c{int(component)}" for match, component in zip(match_values, component_values)],
            dtype="U256",
        )
        member_key_ok = np.array_equal(member_keys, expected_member_keys)
    c0_bits = np.asarray(component_arrays["c0_bits"]) if "c0_bits" in component_arrays else None
    c0_bits_ok = c0_bits is None or (c0_bits.shape == (n,) and c0_bits.dtype.kind in "iu" and c0_bits.dtype.kind != "b")
    covered = np.zeros(n, dtype=bool)
    seen_matches = np.zeros(starts.size, dtype=bool)
    labels_coverage_ok = True
    batch_cursor_ok = True
    unmapped_labels = 0
    component_time_mismatches = 0
    c0_reconciliation_failures = 0
    batches_seen = len(batch_records)
    for record in batch_records:
        match_locals = np.asarray(record["match_locals"])
        labels_flat = np.asarray(record["labels"])
        if match_locals.dtype.kind not in "iu" or labels_flat.dtype.kind not in "iu":
            labels_coverage_ok = False
            continue
        match_locals = match_locals.astype(np.int64, copy=False)
        labels_flat = labels_flat.astype(np.int64, copy=False)
        offset = 0
        expected_batch_rows = 0
        for match_local in match_locals.tolist():
            match = int(match_local)
            if match < 0 or match >= len(starts) or seen_matches[match]:
                labels_coverage_ok = False
                continue
            seen_matches[match] = True
            start, end = int(starts[match]), int(ends[match])
            if start == -1 and end == -1:
                length = 0
            elif start < 0 or end < start or end > t_ms.size:
                labels_coverage_ok = False
                continue
            else:
                length = end - start
            expected_batch_rows += length
            if offset + length > labels_flat.size:
                labels_coverage_ok = False
                continue
            labels = labels_flat[offset:offset + length]
            offset += length
            if length == 0:
                continue
            times = t_ms[start:end]
            unique_labels = np.unique(labels[labels >= 0])
            team_pids = blue_pids[match] if match < blue_pids.shape[0] else np.asarray([], dtype=np.int64)
            for label in unique_labels.tolist():
                index = by_key.get((match, int(label)))
                if index is None:
                    unmapped_labels += 1
                    continue
                member = labels == int(label)
                if not np.any(member):
                    unmapped_labels += 1
                    continue
                first = int(times[member].min())
                last = int(times[member].max())
                if first != int(component_arrays["f"][index]) or (
                    "L" in component_arrays and last != int(component_arrays["L"][index])
                ):
                    # A label/ledger first timestamp mismatch cannot be
                    # repaired by aggregate C0 counts.
                    component_time_mismatches += 1
                    continue
                first_rows = np.flatnonzero(member & (times == first))
                direct = False
                relation = "none"
                block_parse = True
                block_credit = 0
                for row_index in first_rows.tolist():
                    source_index = start + int(row_index)
                    killer_id = int(killer[source_index])
                    victim_id = int(victim[source_index])
                    assist_mask = int(assist[source_index])
                    valid_event = (
                        0 <= killer_id <= 10
                        and 1 <= victim_id <= 10
                        and 0 <= assist_mask <= 1023
                    )
                    if not valid_event:
                        block_parse = False
                        continue
                    block_credit |= assist_mask
                    block_credit |= 1 << (victim_id - 1)
                    if killer_id:
                        block_credit |= 1 << (killer_id - 1)
                    victim_team = _participant_team(victim_id, team_pids)
                    killer_team = _participant_team(killer_id, team_pids)
                    if victim_team is not None and killer_team is not None and victim_team != killer_team:
                        direct = True
                        relation = "killer"
                    if victim_team is None:
                        continue
                    for pid in range(1, 11):
                        if assist_mask & (1 << (pid - 1)):
                            assist_team = _participant_team(pid, team_pids)
                            if assist_team is not None and assist_team != victim_team:
                                direct = True
                                relation = "assist"
                row_is_isolated = True
                if "foreign_kills_in_tau_L" in component_arrays and "other_anchors_in_tau_L" in component_arrays:
                    row_is_isolated = (
                        int(component_arrays["foreign_kills_in_tau_L"][index]) == 0
                        and int(component_arrays["other_anchors_in_tau_L"][index]) == 0
                    )
                c0_mismatch = (
                    row_is_isolated
                    and c0_bits_ok
                    and c0_bits is not None
                    and int(c0_bits[index]) != block_credit
                )
                if not block_parse or c0_mismatch:
                    if c0_mismatch:
                        c0_reconciliation_failures += 1
                    continue
                unknown[index] = "pass" if direct else "fail"
                relation_kind[index] = relation if direct else "none"
                credit_parse_status[index] = PASS
                covered[index] = True
        if offset != labels_flat.size:
            batch_cursor_ok = False
        if record.get("n_event_rows") is not None and int(record["n_event_rows"]) != expected_batch_rows:
            labels_coverage_ok = False
    labels_coverage_ok &= bool(np.all(seen_matches))
    direct_status = PASS if bool(
        sidecar_identity_ok
        and labels_coverage_ok
        and batch_cursor_ok
        and member_key_ok
        and c0_bits_ok
        and duplicate_component_keys == 0
        and unmapped_labels == 0
        and component_time_mismatches == 0
        and c0_reconciliation_failures == 0
        and np.all(covered)
    ) else UNKNOWN
    return unknown, {
        "status": direct_status,
        "reason": "frozen_same_timestamp_member_block_join",
        "n_covered": int(covered.sum()),
        "n_rows": n,
        "n_batches_seen": batches_seen,
        "n_batches_sidecar_verified": int(batches_seen if production_sidecars and sidecar_identity_ok else 0),
        "batch_labels_sha256": hashlib.sha256("\n".join(sidecar_hashes).encode("utf-8")).hexdigest() if sidecar_hashes else "",
        "label_identity_hashes": label_identity_hashes,
        "n_unmapped_labels": int(unmapped_labels),
        "n_duplicate_component_keys": int(duplicate_component_keys),
        "n_component_time_mismatches": int(component_time_mismatches),
        "n_c0_reconciliation_failures": int(c0_reconciliation_failures),
        "labels_coverage_ok": bool(labels_coverage_ok),
        "batch_cursor_ok": bool(batch_cursor_ok),
        "member_key_reconciliation_ok": bool(member_key_ok),
        "c0_reconciliation_available": c0_bits is not None,
        "relation_kind": relation_kind,
        "credit_parse_status": credit_parse_status,
    }


def _reason_dictionary() -> dict[int, str]:
    return {
        8388608: "firstkill_credit_parse_not_verified",
        1: "pre_support_unknown_or_malformed",
        2: "pre_does_not_equal_firstkill_minus_10000",
        4: "lastkill_before_firstkill",
        8: "frame_lag_support_unknown",
        16: "frame_lag_inconsistent_with_firstkill_and_frame",
        32: "candidate_flag_inconsistent_with_inclusive_frame_lag",
        64: "firstkill_frame_lag_outside_inclusive_10000_20000",
        128: "global_isolation_support_unknown",
        256: "global_isolation_failed_closed_pre_lastkill",
        512: "critical_source_or_team_credit_unknown",
        1024: "actual_team_mapping_or_credit_invalid",
        2048: "firstkill_credit_block_association_unverified",
        4096: "opposing_player_credit_unconfirmed",
        8192: "no_confirmed_opposing_player_credit",
        16384: "environment_killer_zero_not_player",
        32768: "h90_endpoint_or_frame_support_invalid",
        65536: "h90_endpoint_or_frame_support_unknown",
        131072: "event_support_not_verified",
        262144: "actual_frame_support_not_verified",
        524288: "game_end_support_not_verified",
        1048576: "starting_scale_is_cache_proxy_or_unknown",
        2097152: "early_pre_cohort_preserved_separately",
        4194304: "label_invalid",
        8388608: "no_valid_firstkill_credit",
        16777216: "unknown_starting_scale_retained_for_overall_candidate",
    }


def _rows_from_arrays(arrays: Mapping[str, np.ndarray], source_status: str) -> dict[str, np.ndarray]:
    """Project the complete partition with typed vector operations.

    ``evaluate_row`` remains the readable scalar policy oracle used by unit
    tests.  This path avoids constructing millions of Python row dictionaries
    and keeps every eligibility flag as a real boolean array.
    """
    n = int(arrays["f"].shape[0])
    for key, value in arrays.items():
        if np.asarray(value).shape != (n,):
            raise ValueError(f"component field {key!r} has unexpected shape {np.asarray(value).shape}")
    f, f_valid = _integer_array(arrays["f"], n, nonnegative=True)
    last, last_valid = _integer_array(arrays["L"], n, nonnegative=True)
    supplied_pre, tau_valid = _integer_array(arrays["tau"], n)
    frame, frame_valid = _integer_array(arrays["F_tau"], n, nonnegative=True)
    lag, lag_valid = _integer_array(arrays["lag_f_F_tau"], n, nonnegative=True)
    foreign, foreign_valid = _integer_array(arrays["foreign_kills_in_tau_L"], n)
    anchors, anchors_valid = _integer_array(arrays["other_anchors_in_tau_L"], n)
    blue, blue_valid = _integer_array(arrays["c0_n_blue"], n, nonnegative=True)
    red, red_valid = _integer_array(arrays["c0_n_red"], n, nonnegative=True)
    unknown_team_count, unknown_team_valid = _integer_array(arrays["c0_n_unknown_team"], n, nonnegative=True)
    credited, credited_valid = _integer_array(arrays["c0_n_credited"], n, nonnegative=True)
    env_killer, env_valid = _integer_array(arrays["c0_n_env_killer"], n, nonnegative=True)

    pre = np.where(f_valid, f - C.ANCHOR_OFFSET_MS, -1).astype(np.int64, copy=False)
    time_role = np.full(n, "unknown_time", dtype="U32")
    has_time = f_valid & last_valid
    time_role[has_time & (pre >= C.PRIMARY_PRE_MIN_MS)] = "primary_pre_ge_120000"
    time_role[has_time & (pre < C.PRIMARY_PRE_MIN_MS)] = "early_pre_lt_120000"
    pre_consistent = f_valid & tau_valid & (supplied_pre == pre)
    pre_mismatch = f_valid & tau_valid & ~pre_consistent
    last_before_first = f_valid & last_valid & (last < f)

    candidate_state = _status_array(arrays["cand_10_20_tau"], n)
    candidate_pass = candidate_state == PASS
    candidate_fail = candidate_state == "fail"
    expected_candidate = lag_valid & (lag >= C.FRAME_LAG_MIN_MS) & (lag <= C.FRAME_LAG_MAX_MS)
    frame_relation = frame_valid & lag_valid & f_valid & ((f - frame) == lag)
    isolation_known = foreign_valid & anchors_valid
    isolation_bad = isolation_known & ((foreign != 0) | (anchors != 0))
    hard_timing_fail = pre_mismatch | last_before_first | candidate_fail | isolation_bad
    timing_state = np.full(n, UNKNOWN, dtype="U7")
    timing_pass = (
        f_valid & last_valid & tau_valid & pre_consistent & frame_relation &
        candidate_pass & expected_candidate & isolation_known & ~isolation_bad
    )
    timing_state[timing_pass] = PASS
    timing_state[hard_timing_fail] = "fail"

    team_map_state = _status_array(arrays["team_map_ok"], n)
    source_state = _status_array(np.full(n, source_status, dtype="U16"), n)
    parse_state = _status_array(arrays.get("firstkill_credit_parse_status", np.full(n, UNKNOWN, dtype="U7")), n)
    critical_unknown = (
        (source_state == UNKNOWN) | (parse_state == UNKNOWN) | (team_map_state == UNKNOWN) |
        ~blue_valid | ~red_valid | ~unknown_team_valid
    )
    critical_fail = (
        (source_state == "fail") | (parse_state == "fail") | (team_map_state == "fail") |
        (unknown_team_valid & (unknown_team_count != 0))
    )
    critical_pass = ~(critical_unknown | critical_fail)

    direct_state = _status_array(arrays.get("firstkill_direct_opposition", np.full(n, UNKNOWN, dtype="U7")), n)
    both_teams = blue_valid & red_valid & (blue >= 1) & (red >= 1)
    opposition = np.full(n, UNKNOWN, dtype="U48")
    minimum_state = np.full(n, UNKNOWN, dtype="U7")
    confirmed = critical_pass & (direct_state == PASS) & both_teams
    unverified = critical_pass & (direct_state == UNKNOWN)
    no_credit = critical_pass & (direct_state == "fail") & (blue_valid & red_valid) & (blue == 0) & (red == 0)
    unconfirmed = critical_pass & (direct_state == "fail") & ~no_credit
    opposition[confirmed] = "confirmed_two_team_firstkill_credit"
    opposition[unverified] = "opposing_player_credit_unverified"
    opposition[no_credit] = "no_confirmed_player_credit"
    opposition[unconfirmed] = "opposing_player_credit_unconfirmed"
    minimum_state[confirmed] = PASS
    minimum_state[no_credit | unconfirmed] = "fail"

    support_inputs = {
        key: _status_array(arrays[key], n)
        for key in ("valid_h90", "events_support_ok", "frame_support_ok", "terminal_support_authoritative")
    }
    support_stack = np.stack(tuple(support_inputs.values()), axis=0)
    support_status = np.full(n, UNKNOWN, dtype="U7")
    support_status[np.all(support_stack == PASS, axis=0)] = PASS
    support_status[np.any(support_stack == "fail", axis=0)] = "fail"

    raw_state = _status_array(arrays["raw_presence_verified"], n)
    mode = np.asarray(arrays["participation_mode"])
    mode_text = mode.astype("U64") if mode.shape == (n,) else np.full(n, "", dtype="U64")
    proof_values = arrays.get(
        "actual_participation_proof",
        np.full(n, UNKNOWN, dtype="U7"),
    )
    proof_state = _status_array(proof_values, n)
    actual = (raw_state == PASS) & (proof_state == PASS) & ~np.isin(mode_text, ("cache_only_proxy", "", "None"))
    starting_scale = np.full(n, "unknown_scale", dtype="U32")
    starting_scale[actual] = "verified"
    primary_support, primary_lower, primary_upper, primary_exact = {}, {}, {}, {}
    proxy_status: dict[int, np.ndarray] = {}
    for radius in (C.PRIMARY_RADIUS, *C.SENSITIVITY_RADII):
        support = _status_array(arrays[f"support_ok_tau_r{radius}"], n)
        lower, lower_valid = _integer_array(arrays[f"p_lower_n_tau_r{radius}"], n)
        upper, upper_valid = _integer_array(arrays[f"p_upper_n_tau_r{radius}"], n)
        lower_valid &= lower >= 0
        upper_valid &= upper >= 0
        status = np.full(n, UNKNOWN, dtype="U7")
        status[support == "fail"] = "fail"
        supported = (support == PASS) & lower_valid & upper_valid
        status[supported] = PASS
        proxy_status[radius] = status
        primary_lower[radius] = np.where(supported, lower, -1).astype(np.int64, copy=False)
        primary_upper[radius] = np.where(supported, upper, -1).astype(np.int64, copy=False)
        primary_exact[radius] = supported & (lower == upper)

    critical_or_time_candidate = critical_pass & (timing_state == PASS) & (minimum_state == PASS)
    supported_candidate = critical_or_time_candidate & (support_status == PASS)
    review_candidate = critical_pass & (timing_state == PASS) & (minimum_state == "fail")
    label_state = np.full(n, UNKNOWN, dtype="U7")
    eligibility = np.full(n, "excluded_minimum_team_credit", dtype="U48")
    eligibility[critical_fail | critical_unknown] = "unverified_critical_source"
    eligibility[(~(critical_fail | critical_unknown)) & (timing_state == "fail")] = "excluded_timing_or_isolation"
    eligibility[(~(critical_fail | critical_unknown)) & (timing_state == UNKNOWN)] = "unverified_timing"
    eligibility[(critical_pass & (timing_state == PASS) & (minimum_state == UNKNOWN))] = "unverified_critical_source"
    eligibility[review_candidate] = "review_candidate_no_confirmed_opposition"
    eligibility[critical_or_time_candidate & (support_status != PASS)] = "candidate_pending_source_support"
    eligibility[supported_candidate & (time_role == "primary_pre_ge_120000")] = "candidate_primary_pending_label"
    eligibility[supported_candidate & (time_role == "early_pre_lt_120000")] = "candidate_early_pending_label"

    overall_retained = supported_candidate
    size_eligible = supported_candidate & actual & (label_state == PASS)
    analysis_state = np.full(n, UNKNOWN, dtype="U7")
    final_ready = np.zeros(n, dtype=bool)

    # A compact reason mask keeps the full reason ledger while avoiding a
    # Python list per source row.  The manifest stores the dictionary.
    reason_dictionary = _reason_dictionary()
    reason_mask = np.zeros(n, dtype=np.uint32)
    def add(mask: np.ndarray, bit: int) -> None:
        reason_mask[mask] |= np.uint32(bit)

    add(~tau_valid, 1)
    add(pre_mismatch, 2)
    add(last_before_first, 4)
    add(~(frame_valid & lag_valid), 8)
    add(f_valid & frame_valid & lag_valid & ~frame_relation, 16)
    add((candidate_state != UNKNOWN) & (lag_valid & (candidate_pass != expected_candidate)), 32)
    add(candidate_fail, 64)
    add(~isolation_known, 128)
    add(isolation_bad, 256)
    add(critical_unknown, 512)
    add(critical_fail, 1024)
    add(parse_state != PASS, 8388608)
    add(unverified, 2048)
    add(unconfirmed, 4096)
    add(no_credit, 8192)
    add(env_valid & (env_killer > 0), 16384)
    add(support_inputs["valid_h90"] == "fail", 32768)
    add(support_inputs["valid_h90"] == UNKNOWN, 65536)
    add(support_inputs["events_support_ok"] != PASS, 131072)
    add(support_inputs["frame_support_ok"] != PASS, 262144)
    add(support_inputs["terminal_support_authoritative"] != PASS, 524288)
    add(~actual, 1048576)
    add(time_role == "early_pre_lt_120000", 2097152)
    add(eligibility == "invalid_label", 4194304)
    add(critical_pass & credited_valid & (credited == 0), 8388608)
    add(supported_candidate & ~actual, 16777216)
    reason_text_values = [
        row_reason_text([reason_dictionary[bit] for bit in sorted(reason_dictionary) if mask & bit])
        for mask in reason_mask
    ]
    reason_text_width = max((len(value) for value in reason_text_values), default=1)
    reason_text = np.asarray(reason_text_values, dtype=f"U{max(1, reason_text_width)}")

    typed: dict[str, np.ndarray] = {
        "canonical_match_key": np.asarray(arrays["anon_match_key"]).astype("U256"),
        "component_key": np.asarray(arrays["member_key"]).astype("U256"),
        "firstkill_ms": np.where(f_valid, f, -1),
        "lastkill_ms": np.where(last_valid, last, -1),
        "pre_ms": pre,
        "cohort_time_role": time_role,
        "firstkill_credit_blue": np.where(blue_valid, blue, -1),
        "firstkill_credit_red": np.where(red_valid, red, -1),
        "opposition_evidence": opposition,
        "firstkill_direct_opposition_status": direct_state,
        "firstkill_credit_parse_status": _status_array(
            arrays.get("firstkill_credit_parse_status", np.full(n, UNKNOWN, dtype="U7")), n
        ),
        "firstkill_relation_kind": np.asarray(arrays.get("firstkill_relation_kind", np.full(n, "unknown", dtype="U9"))).astype("U9"),
        "critical_source_status": np.where(critical_pass, PASS, np.where(critical_fail, "fail", UNKNOWN)).astype("U7"),
        "timing_status": timing_state,
        "starting_scale_group": starting_scale,
        "actual_participation_verified": actual.astype(bool, copy=False),
        "minimum_team_status": minimum_state,
        "support_status": support_status,
        "label_status": label_state,
        "analysis_status": analysis_state,
        "candidate_eligible": critical_or_time_candidate.astype(bool, copy=False),
        "supported_candidate_eligible": supported_candidate.astype(bool, copy=False),
        "review_candidate": review_candidate.astype(bool, copy=False),
        "overall_retained_pending_label": overall_retained.astype(bool, copy=False),
        "size_analysis_eligible": size_eligible.astype(bool, copy=False),
        "eligibility_status": eligibility,
        "all_reasons": reason_text,
        "reason_mask": reason_mask,
        "final_population_ready": final_ready,
        "proxy_P0_exact_r1800": primary_exact[C.PRIMARY_RADIUS].astype(bool, copy=False),
        "proxy_P0_exact_r2000": primary_exact[2000].astype(bool, copy=False),
        "source_row_index": np.arange(n, dtype=np.int64),
    }
    for radius in (C.PRIMARY_RADIUS, *C.SENSITIVITY_RADII):
        typed[f"proxy_P0_lower_r{radius}"] = primary_lower[radius]
        typed[f"proxy_P0_upper_r{radius}"] = primary_upper[radius]
        typed[f"proxy_P0_support_r{radius}"] = proxy_status[radius]
    return typed


def _independent_checks(
    source: Mapping[str, np.ndarray],
    projection: Mapping[str, np.ndarray],
    counts: Mapping[str, Any],
) -> dict[str, Any]:
    n = int(source["f"].shape[0])
    checks: dict[str, Any] = {}

    def check(name: str, value: bool) -> None:
        checks[name] = bool(value)

    check("row_count_preserved", int(projection["source_row_index"].size) == n)
    check("source_row_index_unique", np.array_equal(projection["source_row_index"], np.arange(n, dtype=np.int64)))
    for field in (
        "actual_participation_verified", "candidate_eligible", "supported_candidate_eligible",
        "review_candidate", "overall_retained_pending_label", "size_analysis_eligible",
        "final_population_ready", "proxy_P0_exact_r1800", "proxy_P0_exact_r2000",
    ):
        check(f"{field}_is_bool", projection[field].dtype.kind == "b")
    check("pre_is_firstkill_minus_10000", np.array_equal(projection["pre_ms"], source["f"] - C.ANCHOR_OFFSET_MS))
    expected_candidate = (source["lag_f_F_tau"] >= C.FRAME_LAG_MIN_MS) & (source["lag_f_F_tau"] <= C.FRAME_LAG_MAX_MS)
    check("inclusive_frame_lag_reconciles", np.array_equal(_bool_array(source["cand_10_20_tau"]), expected_candidate))
    expected_isolated = expected_candidate & (source["foreign_kills_in_tau_L"] == 0) & (source["other_anchors_in_tau_L"] == 0)
    check("closed_isolation_reconciles", int(expected_isolated.sum()) == int(counts["n_isolated_candidates"]))
    check("minimum_team_never_fabricated", np.all(
        (~_bool_array(projection["candidate_eligible"])) |
        ((source["c0_n_blue"] >= 1) & (source["c0_n_red"] >= 1) &
         (projection["firstkill_direct_opposition_status"] == PASS))
    ))
    for radius in (C.PRIMARY_RADIUS, *C.SENSITIVITY_RADII):
        lower = projection[f"proxy_P0_lower_r{radius}"]
        upper = projection[f"proxy_P0_upper_r{radius}"]
        supported = projection[f"proxy_P0_support_r{radius}"] == PASS
        check(f"proxy_bounds_ordered_r{radius}", bool(np.all((~supported) | ((lower >= 0) & (lower <= upper)))))
    check("final_population_not_ready", not bool(np.any(_bool_array(projection["final_population_ready"]))))
    checks["all_ok"] = bool(all(checks.values()))
    checks["n_checks"] = len(checks) - 2
    checks["n_mismatches"] = int(sum(1 for key, value in checks.items() if key not in {"all_ok", "n_checks", "n_mismatches"} and not value))
    return checks


def _counts(source: Mapping[str, np.ndarray], projection: Mapping[str, np.ndarray]) -> dict[str, Any]:
    n = int(source["f"].shape[0])
    candidate = _bool_array(source["cand_10_20_tau"])
    isolated = candidate & (source["foreign_kills_in_tau_L"] == 0) & (source["other_anchors_in_tau_L"] == 0)
    primary = source["f"] - C.ANCHOR_OFFSET_MS >= C.PRIMARY_PRE_MIN_MS
    early = ~primary
    both = (source["c0_n_blue"] >= 1) & (source["c0_n_red"] >= 1)
    candidate_eligible = _bool_array(projection["candidate_eligible"])
    supported_candidate = _bool_array(projection["supported_candidate_eligible"])
    return {
        "n_rows": n,
        "n_candidate_10_20_inclusive": int(candidate.sum()),
        "n_isolated_candidates": int(isolated.sum()),
        "n_primary_isolated_candidates": int((isolated & primary).sum()),
        "n_early_isolated_candidates": int((isolated & early).sum()),
        "n_minimum_two_team_firstkill_credit": int((isolated & both).sum()),
        "n_direct_opposition_confirmed": int(np.sum(projection["firstkill_direct_opposition_status"] == PASS)),
        "n_opposition_unverified": int(np.sum(projection["firstkill_direct_opposition_status"] == UNKNOWN)),
        "n_primary_minimum_two_team_candidates": int((isolated & primary & both).sum()),
        "n_early_minimum_two_team_candidates": int((isolated & early & both).sum()),
        "n_candidate_eligible_pending_label": int(candidate_eligible.sum()),
        "n_supported_candidate_eligible_pending_label": int(supported_candidate.sum()),
        "n_primary_candidate_eligible_pending_label": int((candidate_eligible & primary).sum()),
        "n_early_candidate_eligible_pending_label": int((candidate_eligible & early).sum()),
        "n_primary_supported_candidate_eligible_pending_label": int((supported_candidate & primary).sum()),
        "n_early_supported_candidate_eligible_pending_label": int((supported_candidate & early).sum()),
        "n_review_no_confirmed_opposition": int(np.sum(projection["review_candidate"])),
        "n_unknown_starting_scale": int(np.sum(projection["starting_scale_group"] == "unknown_scale")),
        "n_proxy_exact_r1800": int(np.sum(projection["proxy_P0_exact_r1800"])),
        "n_proxy_ambiguous_r1800": int(np.sum(projection["proxy_P0_support_r1800"] == PASS) - np.sum(projection["proxy_P0_exact_r1800"])),
        "n_actual_participation_verified": int(np.sum(projection["actual_participation_verified"])),
        "n_size_analysis_eligible": int(np.sum(projection["size_analysis_eligible"])),
        "n_overall_retained_pending_label": int(np.sum(projection["overall_retained_pending_label"])),
        "n_final_population_ready": 0,
    }


def project_partition(
    source_run: Path = C.DEFAULT_SOURCE_RUN,
    *,
    contract_path: Path = C.CONTRACT_PATH,
    output_dir: Path | None = None,
    hash_artifacts: bool = True,
) -> dict[str, Any]:
    """Project every component row into the adopted policy output."""
    contract, contract_hash, contract_validation = load_contract(contract_path)
    source_info = validate_source_run(source_run, hash_artifacts=hash_artifacts)
    if not source_info["ok"]:
        raise ValueError(
            "source validation is debug-only and cannot produce an accepted projection: "
            + ", ".join(source_info.get("validation_errors", ()))
        )
    source_path = Path(source_info["source_run"]) / "components.npz"
    arrays = _load_projection_arrays(source_path)
    direct_opposition, direct_meta = _derive_direct_opposition(arrays)
    if direct_meta.get("status") != PASS:
        raise RuntimeError(
            "direct first-kill source join is incomplete or unverified: "
            + str(direct_meta.get("reason", "unknown"))
        )
    arrays["firstkill_direct_opposition"] = direct_opposition
    relation_kind = direct_meta.pop("relation_kind", None)
    if relation_kind is None:
        relation_kind = np.full(arrays["f"].shape[0], UNKNOWN, dtype="U9")
    arrays["firstkill_relation_kind"] = np.asarray(relation_kind)
    credit_parse_status = direct_meta.pop("credit_parse_status", None)
    if credit_parse_status is None:
        credit_parse_status = np.full(arrays["f"].shape[0], UNKNOWN, dtype="U7")
    arrays["firstkill_credit_parse_status"] = np.asarray(credit_parse_status)
    source_status = PASS if source_info["ok"] else UNKNOWN
    projection = _rows_from_arrays(arrays, source_status)
    counts = _counts(arrays, projection)
    checks = _independent_checks(arrays, projection, counts)
    if not checks["all_ok"]:
        raise RuntimeError("projection independent checks failed: " + ", ".join(k for k, v in checks.items() if k not in {"all_ok", "n_checks", "n_mismatches"} and not v))
    if output_dir is None:
        raise ValueError("output_dir is required for a fresh named projection run")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    projection_path = destination / "population_projection.npz"
    np.savez_compressed(projection_path, **projection)
    projection_hash = file_sha256(projection_path)
    source_hashes = source_info["source_manifest_hashes"]
    implementation_code_hashes, dependency_code_hashes = _execution_code_hashes()
    thread_environment = {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
        if os.environ.get(key) is not None
    }
    manifest = {
        "study": "population_adoption_20261002",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_projection",
        "contract_version": contract["version"],
        "contract_sha256": contract_hash,
        "contract_validation": contract_validation,
        "source_run": source_info["source_run"],
        "source_manifest_hashes": source_hashes,
        "definition_hashes": source_info["definition_hashes"],
        "source_seed": source_info["source_seed"],
        "source_environment": source_info["source_environment"],
        "source_thread_environment": source_info["source_thread_environment"],
        "source_code_hashes": source_info["source_code_hashes"],
        "implementation_code_hashes": implementation_code_hashes,
        "dependency_code_hashes": dependency_code_hashes,
        "thread_environment": thread_environment,
        "deterministic_seed": None,
        "deterministic_seed_status": "not_applicable_no_sampling",
        "source_seed_not_applicable": source_info["source_seed"] is None,
        "source_files": {
            "components.npz": source_hashes.get("components_sha256", ""),
            "evidence.npz": source_hashes.get("evidence_sha256", ""),
            "reference_fields.npz": source_hashes.get("reference_fields_sha256", ""),
            "event_ledger_train.npz": source_hashes.get("ledger_sha256", ""),
            "ledger_match_ranges.npz": source_hashes.get("ranges_sha256", ""),
            "train_match_meta.npz": source_hashes.get("match_meta_sha256", ""),
        },
        "source_artifact_hashes_verified": bool(hash_artifacts),
        "direct_opposition_join": direct_meta,
        "projection_sha256": projection_hash,
        "projection_file": projection_path.name,
        "storage": "lossless_deflate_npz; decoded dtypes and arrays preserved",
        "required_result_fields": list(C.REQUIRED_RESULT_FIELDS),
        "field_contract": {
            "proxy_P0_lower_upper_r1800": "cache-only lower/upper proxy object; primary radius 1800",
            "proxy_P0_lower_upper_r2000": "cache-only lower/upper proxy object; fixed sensitivity radius 2000",
            "all_reasons": "JSON array encoded as a UTF-8-safe string per row",
            "final_population_ready": "always false until remaining source/state/V/label/q gates pass",
        },
        "counts": counts,
        "independent_checks": checks,
        "reason_dictionary": {
            str(bit): reason
            for bit, reason in sorted(
                ((int(bit), reason) for bit, reason in _reason_dictionary().items()),
                key=lambda item: item[0],
            )
        },
        "training_ready": False,
        "final_population_ready": False,
        "no_sampling": True,
        "raw_cache_reread": False,
        "model_fitting": False,
        "cpu_workers": 1,
        "machine": {"platform": platform.platform(), "python": sys.version},
    }
    _write_json(destination / "RUN_MANIFEST.json", manifest)
    _write_json(destination / "COUNTS.json", counts)
    _write_json(destination / "INDEPENDENT_CHECKS.json", checks)
    _write_json(destination / "CONTRACT_POINTER.json", {
        "contract_path": str(Path(contract_path).resolve()),
        "contract_version": contract["version"],
        "contract_sha256": contract_hash,
        "source_manifest_hashes": source_hashes,
    })
    return manifest
