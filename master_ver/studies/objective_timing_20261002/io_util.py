"""Read-only source loaders and hash helpers."""
from __future__ import annotations

import hashlib
import json
import sys
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


def write_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return file_sha256(path)


def verify_anchor_manifest() -> dict[str, Any]:
    """Verify anchor manifest and all four frozen file identities before processing."""
    path = C.ANCHOR_MANIFEST
    sha = file_sha256(path)
    man = load_json(path)
    checks: dict[str, Any] = {
        "manifest_path": str(path),
        "sha256": sha,
        "sha256_ok": sha == C.EXPECTED_ANCHOR_MANIFEST_SHA256,
        "complete": bool(man.get("complete")),
        "gates_all_ok": bool(man.get("identity", {}).get("all_ok")),
        "n_components": int(man.get("aggregates", {}).get("n_components", -1)),
        "n_components_ok": int(man.get("aggregates", {}).get("n_components", -1))
        == C.EXPECTED_MIN1_COMPONENTS,
    }
    src = man.get("source_hashes", {})
    checks["source_hashes"] = src
    checks["components_exists"] = C.COMPONENTS_NPZ.is_file()
    checks["terminals_exists"] = C.TERMINALS_JSON.is_file()
    checks["components_sha256"] = (
        file_sha256(C.COMPONENTS_NPZ) if checks["components_exists"] else ""
    )
    checks["terminals_sha256"] = (
        file_sha256(C.TERMINALS_JSON) if checks["terminals_exists"] else ""
    )

    # Compare against frozen constants AND values recorded on the anchor manifest.
    man_comp = str(man.get("components_sha256") or "")
    man_term = str(man.get("terminals_sha256") or "")
    checks["components_sha256_expected"] = C.EXPECTED_COMPONENTS_SHA256
    checks["terminals_sha256_expected"] = C.EXPECTED_TERMINALS_SHA256
    checks["components_sha256_manifest"] = man_comp
    checks["terminals_sha256_manifest"] = man_term
    checks["components_sha256_ok"] = (
        checks["components_sha256"] == C.EXPECTED_COMPONENTS_SHA256
        and checks["components_sha256"] == man_comp
        and bool(man_comp)
    )
    checks["terminals_sha256_ok"] = (
        checks["terminals_sha256"] == C.EXPECTED_TERMINALS_SHA256
        and checks["terminals_sha256"] == man_term
        and bool(man_term)
    )

    match_ids_path = C.ST_SOURCE / "train_match_ids.json"
    match_meta_path = C.ST_SOURCE / "train_match_meta.npz"
    checks["match_ids_exists"] = match_ids_path.is_file()
    checks["match_meta_exists"] = match_meta_path.is_file()
    checks["match_ids_sha256"] = file_sha256(match_ids_path) if checks["match_ids_exists"] else ""
    checks["match_meta_sha256"] = (
        file_sha256(match_meta_path) if checks["match_meta_exists"] else ""
    )
    src_mid = str(src.get("match_ids_sha256") or "")
    src_meta = str(src.get("match_meta_sha256") or "")
    checks["match_ids_sha256_expected"] = C.EXPECTED_MATCH_IDS_SHA256
    checks["match_meta_sha256_expected"] = C.EXPECTED_MATCH_META_SHA256
    checks["match_ids_sha256_ok"] = (
        checks["match_ids_sha256"] == C.EXPECTED_MATCH_IDS_SHA256
        and checks["match_ids_sha256"] == src_mid
        and bool(src_mid)
    )
    checks["match_meta_sha256_ok"] = (
        checks["match_meta_sha256"] == C.EXPECTED_MATCH_META_SHA256
        and checks["match_meta_sha256"] == src_meta
        and bool(src_meta)
    )

    checks["all_ok"] = bool(
        checks["sha256_ok"]
        and checks["complete"]
        and checks["gates_all_ok"]
        and checks["n_components_ok"]
        and checks["components_exists"]
        and checks["terminals_exists"]
        and checks["components_sha256_ok"]
        and checks["terminals_sha256_ok"]
        and checks["match_ids_exists"]
        and checks["match_meta_exists"]
        and checks["match_ids_sha256_ok"]
        and checks["match_meta_sha256_ok"]
    )
    return checks


def load_match_ids() -> list[str]:
    path = C.ST_SOURCE / "train_match_ids.json"
    actual = file_sha256(path)
    man = load_json(C.ANCHOR_MANIFEST)
    expected_src = str((man.get("source_hashes") or {}).get("match_ids_sha256") or "")
    if actual != C.EXPECTED_MATCH_IDS_SHA256:
        raise RuntimeError(
            f"match_ids sha256 mismatch: {actual} != {C.EXPECTED_MATCH_IDS_SHA256}"
        )
    if actual != expected_src:
        raise RuntimeError(
            f"match_ids sha256 != anchor source_hashes.match_ids_sha256: {actual} != {expected_src}"
        )
    ids = load_json(path)
    if len(ids) != C.EXPECTED_TRAIN_MATCHES:
        raise RuntimeError(f"match id count {len(ids)} != {C.EXPECTED_TRAIN_MATCHES}")
    return [str(x) for x in ids]


def load_components() -> dict[str, np.ndarray]:
    """Load numeric component fields only (no unrestricted pickle of object arrays)."""
    keys = [
        "match_local",
        "component_id",
        "L",
        "f",
        "delta",
        "game_end_ms",
        "game_end_known",
        "last_frame_ms",
        "next_kill_after_L",
        "next_component_tau",
        "foreign_kills_in_tau_L",
        "other_anchors_in_tau_L",
        "strict_eligible",
        "overlap_next_tau_le_L",
    ]
    actual = file_sha256(C.COMPONENTS_NPZ)
    if actual != C.EXPECTED_COMPONENTS_SHA256:
        raise RuntimeError(
            f"components.npz sha256 mismatch: {actual} != {C.EXPECTED_COMPONENTS_SHA256}"
        )
    man = load_json(C.ANCHOR_MANIFEST)
    man_comp = str(man.get("components_sha256") or "")
    if actual != man_comp:
        raise RuntimeError(
            f"components.npz sha256 != anchor manifest: {actual} != {man_comp}"
        )
    with np.load(C.COMPONENTS_NPZ, allow_pickle=False) as z:
        out = {k: np.asarray(z[k]) for k in keys}
    if int(out["L"].shape[0]) != C.EXPECTED_MIN1_COMPONENTS:
        raise RuntimeError("component count mismatch")
    if not bool(np.all(np.diff(out["match_local"]) >= 0)):
        raise RuntimeError("components not sorted by match_local")
    return out


def load_meta_blue_pids() -> np.ndarray:
    path = C.ST_SOURCE / "train_match_meta.npz"
    actual = file_sha256(path)
    man = load_json(C.ANCHOR_MANIFEST)
    expected_src = str((man.get("source_hashes") or {}).get("match_meta_sha256") or "")
    if actual != C.EXPECTED_MATCH_META_SHA256:
        raise RuntimeError(
            f"match_meta sha256 mismatch: {actual} != {C.EXPECTED_MATCH_META_SHA256}"
        )
    if actual != expected_src:
        raise RuntimeError(
            f"match_meta sha256 != anchor source_hashes.match_meta_sha256: {actual} != {expected_src}"
        )
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["blue_pids"], dtype=np.int16)


def events_array_from_json(data: Any) -> tuple[list[Any], str]:
    """Normalize events.json payload to a top-level event object list.

    Returns (events, error). Unsupported or malformed payloads return error and
    an empty list — callers must NOT treat that as a valid empty match timeline.
    Neutral acquisition events are unrelated: they appear as ordinary event
    objects once the payload schema is valid.
    """
    if isinstance(data, list):
        return data, ""
    if isinstance(data, dict):
        if "events" in data:
            if isinstance(data.get("events"), list):
                return list(data["events"]), ""
            return [], "malformed_events_field_not_list"
        frames = None
        if isinstance(data.get("info"), dict) and "frames" in data["info"]:
            if not isinstance(data["info"].get("frames"), list):
                return [], "malformed_info_frames_not_list"
            frames = data["info"]["frames"]
        elif "frames" in data:
            if not isinstance(data.get("frames"), list):
                return [], "malformed_frames_not_list"
            frames = data["frames"]
        if frames is not None:
            out: list[Any] = []
            for i, fr in enumerate(frames):
                if not isinstance(fr, dict):
                    return [], f"malformed_frame_not_dict:{i}"
                if "events" not in fr:
                    continue
                evs = fr.get("events")
                if not isinstance(evs, list):
                    return [], f"malformed_frame_events_not_list:{i}"
                out.extend(evs)
            return out, ""
        return [], "unsupported_events_payload"
    return [], "unsupported_events_payload"


def load_events(match_id: str) -> tuple[list[Any], str]:
    """Load match events. On schema/parse failure return ([], error).

    Known source-error handling:
    - missing file → missing_events_json (gate fails on full corpus)
    - JSON decode error → events_json_parse_error:*
    - unsupported/malformed payload → error from events_array_from_json
    - empty list with empty error means a valid payload with zero events
    Callers must mark observation support unknown / fail completion on error;
    never treat schema failure as observed NO_ACQ.
    """
    path = C.CACHE_DIR / f"{match_id}.events.json"
    if not path.is_file():
        return [], "missing_events_json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return [], f"events_json_parse_error:{type(exc).__name__}"
    events, err = events_array_from_json(data)
    if err:
        return [], err
    return events, ""


def environment_record() -> dict[str, Any]:
    import platform

    from . import thread_limits as TL

    try:
        import numpy as np  # noqa: F401

        np_ver = np.__version__
    except Exception:
        np_ver = ""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np_ver,
        "seed": C.SEED,
        "thread_limits": TL.applied_settings(),
        "cpu_budget": C.CPU_BUDGET,
        "max_worker_processes": C.MAX_WORKER_PROCESSES,
    }


def verify_reusable_source_run(source_run: Path) -> dict[str, Any]:
    """Gate ``reaggregate_from`` before copy/requery.

    Requires a complete full-corpus source manifest, frozen baseline identities,
    expected match/component counts, and ``objective_events.npz`` SHA equal to the
    hash recorded on that source manifest. Smoke/partial/incomplete/tampered
    sources must fail here (equal row count alone is insufficient).
    """
    source_run = Path(source_run)
    checks: dict[str, Any] = {
        "source_run": str(source_run),
        "manifest_exists": False,
        "events_exists": False,
        "complete": False,
        "not_smoke": False,
        "full_corpus": False,
        "n_matches_ok": False,
        "n_components_ok": False,
        "events_sha_matches_manifest": False,
        "baseline_refs_ok": False,
        "study_ok": False,
        "failures": [],
    }
    man_path = source_run / "RUN_MANIFEST.json"
    ev_path = source_run / "objective_events.npz"
    checks["manifest_exists"] = man_path.is_file()
    checks["events_exists"] = ev_path.is_file()
    if not checks["manifest_exists"]:
        checks["failures"].append("missing_source_manifest")
        checks["all_ok"] = False
        return checks
    if not checks["events_exists"]:
        checks["failures"].append("missing_objective_events")
        checks["all_ok"] = False
        return checks

    man = load_json(man_path)
    checks["study"] = man.get("study")
    checks["study_ok"] = str(man.get("study") or "") == "objective_timing_20261002"
    if not checks["study_ok"]:
        checks["failures"].append("incompatible_study")

    checks["complete"] = bool(man.get("complete"))
    if not checks["complete"]:
        checks["failures"].append("source_manifest_incomplete")

    max_matches = man.get("max_matches")
    gates = dict(man.get("gates") or {})
    smoke = max_matches is not None or gates.get("full_corpus") is False
    checks["max_matches"] = max_matches
    checks["not_smoke"] = not smoke
    if smoke:
        checks["failures"].append("smoke_or_partial_source")

    n_matches = int(man.get("n_matches") or -1)
    n_components = int(man.get("n_components") or -1)
    checks["n_matches"] = n_matches
    checks["n_components"] = n_components
    checks["n_matches_ok"] = n_matches == C.EXPECTED_TRAIN_MATCHES
    checks["n_components_ok"] = n_components == C.EXPECTED_MIN1_COMPONENTS
    if not checks["n_matches_ok"]:
        checks["failures"].append("n_matches_mismatch")
    if not checks["n_components_ok"]:
        checks["failures"].append("n_components_mismatch")

    full_corpus = bool(gates.get("full_corpus", False)) or (
        checks["n_matches_ok"]
        and checks["n_components_ok"]
        and checks["complete"]
        and checks["not_smoke"]
    )
    # Require explicit full_corpus gate when present; otherwise counts+complete+not_smoke
    if "full_corpus" in gates:
        full_corpus = bool(gates.get("full_corpus")) and checks["n_matches_ok"] and checks["n_components_ok"]
    checks["full_corpus"] = full_corpus
    if not full_corpus:
        checks["failures"].append("not_full_corpus")

    arts = dict(man.get("artifacts") or {})
    recorded = str(arts.get("objective_events_sha256") or "")
    actual = file_sha256(ev_path)
    checks["objective_events_sha256_actual"] = actual
    checks["objective_events_sha256_manifest"] = recorded
    checks["events_sha_matches_manifest"] = bool(recorded) and actual == recorded
    if not recorded:
        checks["failures"].append("manifest_missing_objective_events_sha256")
    elif actual != recorded:
        checks["failures"].append("objective_events_sha_mismatch")

    src = dict(man.get("source") or {})
    baseline = {
        "anchor_manifest_sha256": str(src.get("anchor_manifest_sha256") or ""),
        "components_sha256": str(src.get("components_sha256") or ""),
        "terminals_sha256": str(src.get("terminals_sha256") or ""),
    }
    checks["source_baseline_refs"] = baseline
    baseline_ok = (
        baseline["anchor_manifest_sha256"] == C.EXPECTED_ANCHOR_MANIFEST_SHA256
        and baseline["components_sha256"] == C.EXPECTED_COMPONENTS_SHA256
        and baseline["terminals_sha256"] == C.EXPECTED_TERMINALS_SHA256
    )
    # Optional match_ids/meta when recorded on newer manifests
    mid = str(src.get("match_ids_sha256") or "")
    meta = str(src.get("match_meta_sha256") or "")
    if mid:
        baseline_ok = baseline_ok and mid == C.EXPECTED_MATCH_IDS_SHA256
    if meta:
        baseline_ok = baseline_ok and meta == C.EXPECTED_MATCH_META_SHA256
    checks["baseline_refs_ok"] = baseline_ok
    if not baseline_ok:
        checks["failures"].append("baseline_refs_mismatch")

    checks["all_ok"] = bool(
        checks["manifest_exists"]
        and checks["events_exists"]
        and checks["study_ok"]
        and checks["complete"]
        and checks["not_smoke"]
        and checks["full_corpus"]
        and checks["n_matches_ok"]
        and checks["n_components_ok"]
        and checks["events_sha_matches_manifest"]
        and checks["baseline_refs_ok"]
        and not checks["failures"]
    )
    # Deduplicate failures while preserving all_ok logic
    if checks["all_ok"]:
        checks["failures"] = []
    return checks
