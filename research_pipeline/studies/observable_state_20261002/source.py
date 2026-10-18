"""Bind the frozen cohort to the accepted audit and read immutable triplets."""
from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from studies.cache_source_audit_20261002 import config as AC
from studies.cache_source_audit_20261002.audit import _event_array, _read_once
from studies.cache_source_audit_20261002.run_audit import (
    _load_fixed_cohort_manifest,
    _load_index,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "observable_state_20261002"
AUDIT_ACCEPTANCE = ROOT / "reports" / "cache_source_audit_20261002" / "ACCEPTANCE.json"
EXCLUDED_KEY = os.environ.get("LOL_EXCLUDED_MATCH_KEY_SHA256")
ACTIVE_ROLES = json.loads(os.environ.get("LOL_EXPECTED_ACTIVE_ROLES_JSON", "{}"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_key(match_id: str) -> str:
    return hashlib.sha256(match_id.strip().encode("utf-8")).hexdigest()


def load_frozen_entries() -> tuple[list[dict], dict]:
    """Exclude the user-selected match before opening any of its source files."""
    if not EXCLUDED_KEY or not ACTIVE_ROLES:
        raise ValueError("Set local cohort lock environment variables before loading the frozen cohort")
    _, _, pairs = _load_index(AC.INDEX_PATH)
    cohort = _load_fixed_cohort_manifest(AC.FIXED_COHORT_MANIFEST_PATH, pairs)
    if not cohort["production_identity_ok"]:
        raise ValueError("frozen cohort identity failed")
    amendment_path = REPORT / "USER_EXCLUSION.json"
    amendment_hash = file_sha256(amendment_path)
    acceptance_hash = file_sha256(AUDIT_ACCEPTANCE)
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment["excluded_match_key_sha256"] != EXCLUDED_KEY or amendment["expected_active_roles"] != ACTIVE_ROLES:
        raise ValueError("user exclusion amendment changed")
    acceptance = json.loads(AUDIT_ACCEPTANCE.read_text(encoding="utf-8"))
    if acceptance.get("audit_accepted") is not True or acceptance.get("n_matches") != len(pairs):
        raise ValueError("full source audit is not accepted")
    ledger_rel = next(key for key in acceptance["artifacts"] if key.endswith("/row_ledger.jsonl"))
    ledger = ROOT / ledger_rel
    ledger_hash = file_sha256(ledger)
    if ledger_hash != acceptance["artifacts"][ledger_rel]["sha256"]:
        raise ValueError("accepted source ledger hash changed")
    expected = {canonical_key(mid): (mid, patch) for mid, patch in pairs}
    if len(expected) != len(pairs):
        raise ValueError("canonical cohort key collision")
    rows = {}
    with ledger.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["match_key_sha256"]
            if key in rows or key not in expected:
                raise ValueError("source audit duplicate or foreign match key")
            mid, patch = expected[key]
            if row["role"] != AC.PATCH_ROLES[patch] or set(row["source_hashes"]) != {"npz", "meta", "events"}:
                raise ValueError("source audit role or hash fields mismatch")
            if key != EXCLUDED_KEY and row["status"] == "fail":
                raise ValueError("unapproved source failure in active cohort")
            rows[key] = row
    if set(rows) != set(expected) or rows[EXCLUDED_KEY]["role"] != "VALIDATION":
        raise ValueError("source ledger coverage or exclusion identity mismatch")
    entries = []
    for original_index, (mid, patch) in enumerate(pairs):
        key = canonical_key(mid)
        if key == EXCLUDED_KEY:
            continue
        entries.append({"match_id": mid, "canonical_match_key": key, "patch": patch,
                        "role": AC.PATCH_ROLES[patch], "original_index": original_index,
                        "source_hashes": rows[key]["source_hashes"],
                        "audited_numbers": {name: rows[key]["numbers"][name] for name in (
                            "n_frames", "n_events", "n_kills", "game_end_ms", "game_end_winning_team")}})
    if dict(Counter(row["role"] for row in entries)) != ACTIVE_ROLES:
        raise ValueError("active patch roles do not match adopted exclusion")
    if (file_sha256(ledger) != ledger_hash or file_sha256(AUDIT_ACCEPTANCE) != acceptance_hash
            or file_sha256(amendment_path) != amendment_hash
            or file_sha256(AC.INDEX_PATH) != AC.EXPECTED_INDEX_SHA256
            or file_sha256(AC.FIXED_COHORT_MANIFEST_PATH) != AC.EXPECTED_COHORT_MANIFEST_SHA256):
        raise ValueError("cohort or accepted source binding changed during preparation")
    binding = {"cohort_sha256": cohort["sha256"], "index_sha256": AC.EXPECTED_INDEX_SHA256,
               "source_ledger_sha256": ledger_hash,
               "source_audit_acceptance_sha256": acceptance_hash,
               "user_exclusion_sha256": amendment_hash,
               "n_active": len(entries), "role_counts": ACTIVE_ROLES,
               "excluded_keys": [EXCLUDED_KEY]}
    return entries, binding


def read_cached_triplet(entry: dict, cache_dir: Path = AC.CACHE_DIR) -> tuple[dict, dict, list, dict]:
    """Read once, compare full byte hashes, then decode the validated bytes."""
    if entry["canonical_match_key"] == EXCLUDED_KEY:
        raise ValueError("user-excluded source must never be read for generation")
    payloads, info = {}, {}
    for name, suffix in (("npz", ".npz"), ("meta", ".meta.json"), ("events", ".events.json")):
        data, details = _read_once(cache_dir / (entry["match_id"] + suffix))
        if data is None or details.get("stable") is not True:
            raise ValueError(f"source read unstable: {name}")
        if details["sha256"] != entry["source_hashes"][name]:
            raise ValueError(f"accepted source hash changed: {name}")
        payloads[name], info[name] = data, details
    with np.load(io.BytesIO(payloads["npz"]), allow_pickle=False) as bank:
        arrays = {name: np.array(bank[name], copy=True) for name in AC.EXPECTED_NPZ_KEYS}
    meta = json.loads(payloads["meta"])
    events, wrapper, reasons = _event_array(json.loads(payloads["events"]))
    if not isinstance(meta, dict) or events is None or reasons:
        raise ValueError("source metadata or event wrapper invalid")
    info["event_wrapper"] = wrapper
    return arrays, meta, events, info
