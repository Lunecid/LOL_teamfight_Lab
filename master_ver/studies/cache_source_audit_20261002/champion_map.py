"""Read-only Data Dragon champion ID/name maps for cache diagnostics."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import config as C


def _casefold_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value.casefold() if value else None


def _int_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001 - malformed source row stays diagnostic
        return None
    if not text.isdigit():
        return None
    number = int(text)
    return number if number > 0 else None


def _empty_map(patch: str, *, path: Path, error: str, sha256: str | None = None) -> dict[str, Any]:
    return {
        "patch": str(patch),
        "path": path,
        "sha256": sha256,
        "load_status": "unknown",
        "error": error,
        "n_entries": 0,
        "n_valid_id_entries": 0,
        "n_invalid_id_entries": 0,
        "id_to_names": {},
        "name_to_ids": {},
    }


@lru_cache(maxsize=None)
def load_champion_map(patch: str) -> dict[str, Any]:
    """Load one patch map once per Python process from one byte read."""
    patch = str(patch)
    path = C.DATADRAGON_PATHS.get(patch)
    if path is None:
        return _empty_map(patch, path=Path(), error="patch_map_not_declared")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return _empty_map(patch, path=path, error=f"map_read_error:{type(exc).__name__}")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - preserve source support status
        return _empty_map(patch, path=path, error=f"map_json_error:{type(exc).__name__}", sha256=digest)
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, Mapping):
        return _empty_map(patch, path=path, error="map_data_not_object", sha256=digest)

    id_to_names: dict[int, set[str]] = {}
    invalid_ids = 0
    for data_key, record in data.items():
        if not isinstance(record, Mapping):
            invalid_ids += 1
            continue
        champion_id = _int_id(record.get("key"))
        if champion_id is None:
            invalid_ids += 1
            continue
        names = {name for name in (_casefold_name(data_key), _casefold_name(record.get("id"))) if name}
        if not names:
            invalid_ids += 1
            continue
        id_to_names.setdefault(champion_id, set()).update(names)

    name_to_ids: dict[str, set[int]] = {}
    for champion_id, names in id_to_names.items():
        for name in names:
            name_to_ids.setdefault(name, set()).add(champion_id)
    return {
        "patch": patch,
        "path": path,
        "sha256": digest,
        "load_status": "pass" if id_to_names else "unknown",
        "error": None if id_to_names else "map_contains_no_valid_entries",
        "n_entries": len(data),
        "n_valid_id_entries": len(id_to_names),
        "n_invalid_id_entries": int(invalid_ids),
        "id_to_names": {int(key): tuple(sorted(values)) for key, values in sorted(id_to_names.items())},
        "name_to_ids": {key: tuple(sorted(values)) for key, values in sorted(name_to_ids.items())},
    }


def source_map_binding() -> dict[str, Any]:
    """Return bounded source identity and policy metadata without map contents."""
    patches: dict[str, Any] = {}
    for patch in sorted(C.DATADRAGON_PATHS):
        mapping = load_champion_map(patch)
        actual = mapping.get("sha256")
        expected = C.DATADRAGON_EXPECTED_SHA256.get(patch)
        patches[patch] = {
            "path": str(C.DATADRAGON_PATHS[patch].relative_to(C.ROOT)).replace("\\", "/"),
            "sha256": actual,
            "expected_sha256": expected,
            "sha256_matches_expected": bool(actual and expected and actual == expected),
            "load_status": mapping.get("load_status"),
            "n_entries": int(mapping.get("n_entries", 0)),
            "n_valid_id_entries": int(mapping.get("n_valid_id_entries", 0)),
            "n_invalid_id_entries": int(mapping.get("n_invalid_id_entries", 0)),
            "error": mapping.get("error"),
        }
    return {
        "policy": "current API ID/name consistency only; data key and data.id casefolded; human display name excluded",
        "patches": patches,
        "production_identity_ok": bool(all(
            item["sha256_matches_expected"] and item["load_status"] == "pass"
            for item in patches.values()
        )),
    }
