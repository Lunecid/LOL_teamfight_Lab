"""Read-only loaders and identity checks against the completed ST study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config as C


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def anonymous_match_key(match_local: int) -> str:
    return f"m{int(match_local):05d}"


def load_source_bundle() -> dict[str, Any]:
    src_man = load_json(C.ST_SOURCE / "SOURCE_MANIFEST.json")
    run_man = load_json(C.ST_OUT / "RUN_MANIFEST.json")
    agg = load_json(C.ST_AGG / "summary.json")
    match_ids = load_json(C.ST_SOURCE / "train_match_ids.json")
    ledger = np.load(C.ST_SOURCE / "event_ledger_train.npz", allow_pickle=False)
    ranges = np.load(C.ST_SOURCE / "ledger_match_ranges.npz", allow_pickle=False)
    meta = np.load(C.ST_SOURCE / "train_match_meta.npz", allow_pickle=False)
    return {
        "source_manifest": src_man,
        "run_manifest": run_man,
        "aggregate": agg,
        "match_ids": match_ids,
        "ledger": {k: ledger[k] for k in ledger.files},
        "ranges": {k: ranges[k] for k in ranges.files},
        "meta": {k: meta[k] for k in meta.files},
    }


def verify_identity(bundle: dict[str, Any]) -> dict[str, Any]:
    src = bundle["source_manifest"]
    run = bundle["run_manifest"]
    agg = bundle["aggregate"]
    ids = bundle["match_ids"]
    led = bundle["ledger"]
    rng = bundle["ranges"]
    meta = bundle["meta"]
    checks: dict[str, Any] = {}
    checks["kills_sha256_match"] = src["kills_npz"]["sha256"] == C.EXPECTED_KILLS_SHA256
    checks["n_matches"] = len(ids) == C.EXPECTED_TRAIN_MATCHES == int(rng["starts"].shape[0])
    checks["n_kills"] = int(led["t_ms"].shape[0]) == C.EXPECTED_TRAIN_KILLS
    checks["zero_kill"] = int(src["train"]["zero_kill_matches"]) == C.EXPECTED_ZERO_KILL
    checks["ledger_sha256"] = file_sha256(C.ST_SOURCE / "event_ledger_train.npz") == run["identity"]["ledger_sha256"]
    checks["ranges_sha256"] = file_sha256(C.ST_SOURCE / "ledger_match_ranges.npz") == run["identity"]["ranges_sha256"]
    checks["match_ids_sha256"] = file_sha256(C.ST_SOURCE / "train_match_ids.json") == run["identity"]["match_ids_sha256"]
    checks["meta_sha256"] = file_sha256(C.ST_SOURCE / "train_match_meta.npz") == run["identity"]["match_meta_sha256"]
    # Adopted config conservation / span / diameter from aggregate
    row = next(r for r in agg["rows"] if r["config_id"] == C.ADOPTED_CFG)
    checks["n_components_expected"] = int(row["n_clusters"]) == C.EXPECTED_MIN1_COMPONENTS
    checks["span_max"] = int(row["span_ms_max"]) == C.EXPECTED_SPAN_MAX
    checks["diameter_max_close"] = abs(float(row["diameter_max"]) - C.EXPECTED_DIAMETER_MAX) < 1e-6
    checks["conservation_ok"] = bool(agg["conservation"][C.ADOPTED_CFG]["ok"])
    checks["all_ok"] = all(bool(v) for v in checks.values() if isinstance(v, bool))
    checks["adopted_row"] = {
        "n_clusters": row["n_clusters"],
        "span_ms_max": row["span_ms_max"],
        "diameter_max": row["diameter_max"],
        "n_zero_kill_matches": row["n_matches_zero_clusters"],
    }
    return checks


def list_batch_ids() -> list[int]:
    ids = []
    for p in C.ST_BATCHES.glob("batch_*.json"):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def _read_batch_sidecar_header(js_path: Path) -> dict[str, Any]:
    """Read only a bounded prefix of batch_*.json (avoid loading ~GB of sidecars)."""
    import re

    # Sidecar layout: scalars and identity precede the huge 'clusters' payload.
    with open(js_path, "r", encoding="utf-8") as f:
        head = f.read(8192)

    def _grab(key: str) -> str | None:
        m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', head)
        return m.group(1) if m else None

    def _grab_int(key: str) -> int | None:
        m = re.search(rf'"{key}"\s*:\s*(-?\d+)', head)
        return int(m.group(1)) if m else None

    out = {
        "npz_sha256": _grab("npz_sha256"),
        "n_event_rows": _grab_int("n_event_rows"),
        "identity_source_kills": _grab("source_kills_sha256"),
    }
    if out["npz_sha256"] is None or out["n_event_rows"] is None or out["identity_source_kills"] is None:
        # Fallback: full parse only if header incomplete
        js = load_json(js_path)
        return {
            "npz_sha256": js["npz_sha256"],
            "n_event_rows": int(js["n_event_rows"]),
            "identity_source_kills": js["identity"]["source_kills_sha256"],
        }
    return out


def load_adopted_labels_for_batch(batch_id: int) -> dict[str, Any]:
    npz_path = C.ST_BATCHES / f"batch_{batch_id:04d}.npz"
    js_path = C.ST_BATCHES / f"batch_{batch_id:04d}.json"
    js = _read_batch_sidecar_header(js_path)
    with np.load(npz_path, allow_pickle=False) as z:
        match_locals = np.asarray(z["match_locals"], dtype=np.int32)
        labels = np.asarray(z[f"labels_{C.ADOPTED_CFG}"], dtype=np.int64)
    return {
        "batch_id": batch_id,
        "match_locals": match_locals,
        "labels": labels,
        "n_event_rows": int(js["n_event_rows"]),
        "npz_sha256": js["npz_sha256"],
        "identity_source_kills": js["identity_source_kills"],
    }


def load_minute_ts(match_id: str) -> tuple[np.ndarray | None, str]:
    path = C.CACHE_DIR / f"{match_id}.npz"
    if not path.is_file():
        return None, "missing_npz"
    try:
        with np.load(path, allow_pickle=False) as z:
            if "minute_ts" not in z.files:
                return None, "missing_minute_ts_member"
            ts = np.asarray(z["minute_ts"], dtype=np.int64)
        return ts, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"npz_read_error:{type(exc).__name__}"


def _events_array_from_json(data: Any) -> list[Any]:
    """Normalize events.json payload to a top-level event object list (authority)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("events"), list):
            return list(data["events"])
        frames = None
        if isinstance(data.get("info"), dict) and isinstance(data["info"].get("frames"), list):
            frames = data["info"]["frames"]
        elif isinstance(data.get("frames"), list):
            frames = data["frames"]
        if frames is not None:
            out: list[Any] = []
            for fr in frames:
                if isinstance(fr, dict) and isinstance(fr.get("events"), list):
                    out.extend(fr["events"])
            return out
    return []


def load_events_game_end(match_id: str) -> tuple[dict[str, Any], str]:
    """Extract GAME_END via object-local JSON parsing of the original event array.

    The full original event array is authoritative. Nearest-string heuristics are not used.
    Also returns an independent reference parse for same-pass cross-check fields.
    """
    from . import contracts as CT

    path = C.CACHE_DIR / f"{match_id}.events.json"
    empty = {
        "n_game_end": 0,
        "game_end_ms": C.SENTINEL_MISSING,
        "game_end_all": [],
        "game_end_unique": [],
        "game_end_multiple_events": False,
        "game_end_duplicate_ts": False,
        "game_end_ambiguous_distinct": False,
        "game_end_malformed": 0,
        "game_end_malformed_reasons": {},
        "game_end_known": False,
        "terminal_crosscheck_match": True,
        "terminal_crosscheck_detail": {},
    }
    if not path.is_file():
        return dict(empty), "missing_events_json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        out = dict(empty)
        out["terminal_crosscheck_match"] = False
        out["terminal_crosscheck_detail"] = {"error": type(exc).__name__}
        return out, f"events_json_parse_error:{type(exc).__name__}"

    events = _events_array_from_json(data)
    primary = CT.parse_game_end_events(events)
    reference = CT.parse_game_end_events_reference(events)
    cross_keys = (
        "n_game_end",
        "game_end_ms",
        "game_end_all",
        "game_end_unique",
        "game_end_multiple_events",
        "game_end_duplicate_ts",
        "game_end_ambiguous_distinct",
        "game_end_malformed",
        "game_end_known",
    )
    mismatch = {k: {"primary": primary.get(k), "reference": reference.get(k)} for k in cross_keys if primary.get(k) != reference.get(k)}
    out = dict(primary)
    out["terminal_crosscheck_match"] = len(mismatch) == 0
    out["terminal_crosscheck_detail"] = mismatch
    return out, ""
