"""External (later-patch) validation, step E1: sample freeze and raw -> per-match cache build.

Pre-registered rule (DECISION_LOG_KO.md 10-09 12:10, METHODS_MEMO_KO.md section 6.12): primary set KR 15.18-15.22,
per patch the 8,000 matches with the smallest SHA256("external_20261009:" + match_id); secondary set KR 16.13, all
matches.  Filters: queueId 420, mapId 11, gameMode CLASSIC, timeline file present; patch = major.minor of
info.gameVersion.  Nothing but the hash decides membership.

Modules: sample (scan + freeze, writes SAMPLE_MANIFEST.json before any cache is built), build_cache (builder parity on
200 KR 16.13 matches, full build of the 15.18-15.22 sample, E1_REPORT.json).  The cache builder is the old repository's
code (data.cache_io / gameplay.pipeline_cache), imported read-only; this package never writes into that repository.

Importing this module has no side effects and does not import the old repository.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

VERSION = "external_validation_e1_20261009"
HASH_TAG = "external_20261009:"
PATCHES_2025 = ("15.18", "15.19", "15.20", "15.21", "15.22")
PATCH_2026 = "16.13"
PER_PATCH = 8000
PARITY_N = 200
DATASET_2025 = "kr_15_18_22"
DATASET_2026 = "kr_16_13"
FILTERS = {"queueId": 420, "mapId": 11, "gameMode": "CLASSIC"}

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "external_validation_20261009"
LOGS = REPORTS / "logs"
DATA_ROOT = Path(r"D:\LOL_External_20261009")
OLD_REPO = Path(r"C:\Users\todtj\PycharmProjects\LOL_teamfight")
RAW = {
    DATASET_2025: Path(r"C:\Users\todtj\PycharmProjects\Lol_project\data\raw\matches\kr"),
    DATASET_2026: Path(r"D:\LOL_Project\data\raw\2026_current\kr"),
}
REF_CACHE_2025 = Path(r"D:\LOL_Project\cache\match_cache_fresh_v3_engage_status13")
REF_PATCH_INDEX_2025 = Path(r"D:\LOL_Project\cache\match_cache_fresh_v3_engage_status13_patch_index.json")
# Old-lineage KR 16.13 caches: the one named by EXTERNAL_DATA_INVENTORY_20261009.json (written 2026-09-15 by
# scripts/fc20260915_external_prepare.py) and an earlier copy (2026-08-09) that also holds KR 16.15.
OLD_CACHE_16_13 = ROOT / "outputs" / "full_corpus_training_20260915" / "external" / "KR_16.13" / "cache"
OLD_CACHE_16_13_PREPARED = OLD_CACHE_16_13.parent / "prepared_manifest.json"
OLD_PREFLIGHT_16_13 = ROOT / "outputs" / "full_corpus_preflight_20260915" / "external" / "external_KR_16.13.csv"
OLD_CACHE_16_13_ALT = Path(r"D:\LOL_Project\fusion_2615\cache\match_cache_fresh_v3_engage_status13")
DRIVER_2026 = ROOT / "scripts" / "fc20260915_external_prepare.py"
PARENT_SOURCE_HASHES = ROOT / "outputs" / "postkill_objective_delay_full" / "run.json"

SAMPLE_MANIFEST = REPORTS / "SAMPLE_MANIFEST.json"
SAMPLE_MANIFEST_SHA = REPORTS / "SAMPLE_MANIFEST.sha256"
PARITY_REPORT = REPORTS / "PARITY_16_13.json"
E1_REPORT = REPORTS / "E1_REPORT.json"
CACHE_DIR = {DATASET_2025: DATA_ROOT / "cache_kr_15_18_22", DATASET_2026: DATA_ROOT / "cache_kr_16_13"}
PATCH_INDEX = {DATASET_2025: DATA_ROOT / "match_cache_kr_15_18_22_patch_index.json",
               DATASET_2026: DATA_ROOT / "match_cache_kr_16_13_patch_index.json"}
PARITY_DIR = DATA_ROOT / "parity_16_13" / "cache"
WORK = DATA_ROOT / "work"
RUNTIME_REPO = DATA_ROOT / "runtime_repo"
MAX_WORKERS = 4


def match_hash(match_id: str) -> str:
    """h = SHA256("external_20261009:" + match_id).hexdigest(); match_id is the file-name stem."""
    return hashlib.sha256((HASH_TAG + str(match_id)).encode("utf-8")).hexdigest()


def select_smallest(match_ids, n: int) -> list[str]:
    """The n ids with the smallest h (ascending h; equal-length lowercase hex compares like the integer)."""
    ranked = sorted({str(m) for m in match_ids}, key=lambda m: (match_hash(m), m))
    return ranked[: int(n)]


def major_minor(game_version) -> str:
    """Patch = major.minor of info.gameVersion ('' when the version has fewer than two parts)."""
    parts = str(game_version or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 and parts[0] and parts[1] else ""


def sha256_file(path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_bytes(path, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    else:
        raise PermissionError(f"could not replace {path}")
    return sha256_bytes(data)


def json_bytes(obj, list_key: str | None = None) -> bytes:
    """indent=1 JSON; when list_key is given, that top-level list is written one compact record per line."""
    if list_key is None or list_key not in obj:
        return json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False).encode("utf-8")
    head = {k: v for k, v in obj.items() if k != list_key}
    text = json.dumps(head, indent=1, ensure_ascii=False, allow_nan=False)
    rows = ",\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",", ":")) for r in obj[list_key])
    body = "[\n" + rows + "\n]" if rows else "[]"
    text = text[:-1].rstrip() + (",\n" if head else "\n") + f' {json.dumps(list_key)}: ' + body + "\n}"
    return text.encode("utf-8")


def write_json(path, obj, list_key: str | None = None) -> str:
    return atomic_write_bytes(path, json_bytes(obj, list_key))


def read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def set_below_normal_priority() -> bool:
    try:
        import psutil

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return True
    except Exception:
        return False


class Log:
    """Append-only text log with UTC timestamps (flushed per line)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{utc_now()} {msg}"
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


def dir_bytes(path) -> tuple[int, int]:
    """(files, bytes) under path, recursive, via os.scandir."""
    files = total = 0
    stack = [str(path)]
    while stack:
        top = stack.pop()
        try:
            it = os.scandir(top)
        except OSError:
            continue
        with it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    files += 1
                    total += e.stat().st_size
    return files, total
