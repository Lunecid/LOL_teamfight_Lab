"""T3.1 E3 player matrix: the external KR 15.18-15.22 matches as dense player numbers (no identifier is kept).

Binding specification: reports/tog_experiments_20261012/PRESPEC_E3_KO.md sections 2, 3, 6 and 9 (PLAN_E1_E5_KO.md
Task 7).  The variance functions (Task 8) and the E3 run (Task 9, which re-opens the sealed external labels after the
lock) are not here: this module reads no label and no score.

Match set and order
  One matrix row per match of the external row table D:/LOL_External_20261009/q_inputs/kr_15_18_22/Q_ROWS_h90.npz
  (146,845 rows, 38,424 matches), in ascending match_index (np.unique): the order in which e3_evaluate.py:172 and
  qeval.per_match_loss / align_cells ("ascending match id") lay out the per-match losses D_m (PRESPEC_E3 section 3).
  Every match_index must carry one canonical_match_key and one patch in the row table (checked).  Task 9 aligns with
  np.array_equal(PLAYERS.match_index, np.unique(Q_ROWS.match_index)) and the canonical keys.

Mapping (PRESPEC_E3 section 2)
  canonical_match_key = e2a_common.canonical_key(match_id) = SHA-256(match_id) over the match ids of the patch index
  D:/LOL_External_20261009/match_cache_kr_15_18_22_patch_index.json (match_id -> patch).  The detail file is
  C:/Users/todtj/PycharmProjects/Lol_project/data/raw/matches/kr/detail/<match_id>.json; its players are
  info.participants[].puuid placed by participantId, so column j holds participantId j + 1.  A participant without a
  puuid (field absent, not a string, or blank) is -1.

Privacy
  puuid strings exist only inside _number_players, whose local dict maps puuid -> dense number (numbers by first
  appearance: matrix order, then participantId 1-10; a failed match numbers nobody).  No identifier is printed, logged,
  raised or written; the outputs hold integers, match keys and aggregate counts.  One process and no worker, so the
  strings never cross a process boundary.

Failures, per match, counted and never dropped (map_status codes 1-7; 0 = mapped)
  key_not_in_patch_index; patch_mismatch (the patch index's patch differs from the row table's); detail_file_missing;
  detail_unreadable (OS error, invalid JSON, no info.participants list); match_id_mismatch (metadata.matchId present and
  different from the file's match id); participants_malformed (not ten participant objects with participantIds 1-10);
  duplicate_puuid (one puuid on two participants).  A failed match keeps its row with all ten entries -1 (k_m = 0).
  More than 1 % failed matches is BLOCKED (PRESPEC_E3 section 6), and so is a missing input:
  E3_BLOCKED_players_<n>.json (counts and paths only) is written instead of the matrix and the exit status is 2.

Outputs, each written once under D:/LOL_Training_20261004/tog_e3_1
  PLAYERS_kr_15_18_22.npz   match_index int64 (M,), canonical_match_key S64 (M,), player_idx int64 (M, 10),
                            map_status int8 (M,), players_by_match_count int64 (K + 1,) (entry k = number of players
                            in exactly k matches)
  PLAYERS_kr_15_18_22.json  rows and matches (by patch), mapping result, player counts, the matches-per-player
                            distribution, a digest of the detail files read, input and code SHA-256, UTC times, stage
                            wall times and peak memory.

Usage: python -B -m studies.tog_experiments_20261012.e3_players
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.external_validation_20261009 import e2a_common  # noqa: E402
from studies.tog_experiments_20261012.common import (  # noqa: E402
    TRAINING, git_head, peak_memory_bytes, run_record, utc, write_npz_once, write_once)

VERSION = "tog_e3_players_v1_20261012"
DATASET = e2a_common.DATASET_2025                    # "kr_15_18_22", the only set with Riot detail files (PRESPEC_E3 s2)
TOG_E3 = TRAINING / "tog_e3_1"
SOURCES = {DATASET: {
    "rows_path": e2a_common.DATA_ROOT / "q_inputs" / DATASET / "Q_ROWS_h90.npz",
    "patch_index_path": e2a_common.PATCH_INDEX[DATASET],
    "detail_dir": Path(r"C:\Users\todtj\PycharmProjects\Lol_project\data\raw\matches\kr\detail"),
}}
PLAYERS = 10
MAX_FAILED_SHARE = 0.01                              # PRESPEC_E3 section 6: more than 1 % of the matches is BLOCKED
MAPPED = 0
FAILURES = ("key_not_in_patch_index", "patch_mismatch", "detail_file_missing", "detail_unreadable",
            "match_id_mismatch", "participants_malformed", "duplicate_puuid")
STATUS = {name: code for code, name in enumerate(FAILURES, start=1)}
CODE_FILES = (Path(__file__).resolve(), ROOT / "studies/tog_experiments_20261012/common.py",
              ROOT / "studies/external_validation_20261009/e2a_common.py", ROOT / "studies/q_production_20261005/qtest.py")
ORDER = ("one row per match of the row table in ascending match_index (np.unique), the order of the per-match losses "
         "D_m in e3_evaluate.py:172 and qeval.per_match_loss / align_cells (ascending match id); align with "
         "np.array_equal(match_index, np.unique(Q_ROWS.match_index)) and the canonical keys")
COLUMNS = "player_idx[:, j] is participantId j + 1; -1 = no puuid, and all ten entries of a failed match"
PRIVACY = ("puuid strings existed only in memory, inside _number_players (dict puuid -> dense number, discarded on "
           "return); numbers follow first appearance in matrix order, then participantId; no identifier is written")
DIGEST_RULE = "SHA-256 of the per-file SHA-256 hex digests, one per line, of the detail files read, in matrix order"


@dataclass(frozen=True, eq=False)
class PlayerMatrix:
    """The matrix and the counts it was built with; no identifier."""

    match_index: np.ndarray        # int64 (M,), ascending
    match_keys: np.ndarray         # S64 (M,), canonical_match_key of each row
    player_idx: np.ndarray         # int64 (M, 10), column j = participantId j + 1, -1 = none
    map_status: np.ndarray         # int8 (M,), 0 mapped, else a STATUS code
    patch: np.ndarray              # S8 (M,), the row table's patch of each match
    rows: int                      # rows of the row table
    rows_by_patch: dict[str, int]
    slots_without_puuid: int       # participants of mapped matches without a puuid
    files_read: int                # detail files whose bytes were read
    files_digest: str              # DIGEST_RULE
    seconds: dict[str, float]      # stage wall times


def _sources(dataset: str, rows_path: Any = None, patch_index_path: Any = None, detail_dir: Any = None) -> dict[str, Path]:
    if dataset not in SOURCES:
        raise ValueError(f"unknown dataset {dataset!r}: the player matrix covers {sorted(SOURCES)} (PRESPEC_E3 section 2)")
    given = {"rows_path": rows_path, "patch_index_path": patch_index_path, "detail_dir": detail_dir}
    return {name: Path(given[name]) if given[name] is not None else path for name, path in SOURCES[dataset].items()}


def match_table(rows_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, int]]:
    """(ascending match_index, canonical key and patch of each match, row count, rows by patch) of a row table."""

    with np.load(rows_path, allow_pickle=False) as data:
        match_index = np.asarray(data["match_index"], dtype=np.int64)
        keys = np.asarray(data["canonical_match_key"])
        patch = np.asarray(data["patch"])
    if match_index.size == 0:
        raise ValueError("the row table has no rows")
    unique, first, inverse = np.unique(match_index, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    if not (np.array_equal(keys, keys[first][inverse]) and np.array_equal(patch, patch[first][inverse])):
        raise ValueError("a match_index of the row table carries two canonical keys or two patches")
    if np.unique(keys).size != unique.size:
        raise ValueError("a canonical key of the row table carries two match_index values")
    rows_by_patch = {p.decode("ascii"): int((patch == p).sum()) for p in np.unique(patch).tolist()}
    return unique, keys[first], patch[first], int(match_index.size), rows_by_patch


def key_index(patch_index_path: Path) -> dict[bytes, tuple[str, str]]:
    """canonical_match_key (ASCII bytes) -> (match_id, patch) for every match id of the patch index."""

    index = e2a_common.read_json(patch_index_path)
    out = {e2a_common.canonical_key(match_id).encode("ascii"): (str(match_id), str(patch)) for match_id, patch in index.items()}
    if len(out) != len(index):
        raise ValueError("two match ids of the patch index share a canonical key")
    return out


def _number_players(match_ids: list[str | None], detail_dir: Path) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    """Read each match's detail file and number its players; ``match_ids[i]`` is None for a row that failed before.

    Returns (player_idx, file-stage status, slots without puuid, files read, digest of the files read).  The dict
    ``numbers`` (puuid -> dense number) is the only holder of puuid strings and ends with this call.
    """

    player_idx = np.full((len(match_ids), PLAYERS), -1, dtype=np.int64)
    status = np.zeros(len(match_ids), dtype=np.int8)
    numbers: dict[str, int] = {}
    digest = hashlib.sha256()
    slots_without = files_read = 0
    for row, match_id in enumerate(match_ids):
        if match_id is None:
            continue
        try:
            raw = (detail_dir / f"{match_id}.json").read_bytes()
        except FileNotFoundError:
            status[row] = STATUS["detail_file_missing"]
            continue
        except OSError:
            status[row] = STATUS["detail_unreadable"]
            continue
        files_read += 1
        digest.update(hashlib.sha256(raw).hexdigest().encode("ascii") + b"\n")
        try:
            doc = json.loads(raw)
            participants = doc["info"]["participants"]
        except (ValueError, TypeError, KeyError):        # invalid JSON or UTF-8, not an object, no info.participants
            status[row] = STATUS["detail_unreadable"]
            continue
        if not isinstance(participants, list):
            status[row] = STATUS["detail_unreadable"]
            continue
        metadata = doc.get("metadata")
        if isinstance(metadata, dict) and "matchId" in metadata and metadata["matchId"] != match_id:
            status[row] = STATUS["match_id_mismatch"]
            continue
        if len(participants) != PLAYERS or not all(isinstance(p, dict) for p in participants):
            status[row] = STATUS["participants_malformed"]
            continue
        ids = [p.get("participantId") for p in participants]
        if not all(type(i) is int for i in ids) or sorted(ids) != list(range(1, PLAYERS + 1)):
            status[row] = STATUS["participants_malformed"]
            continue
        slots = [""] * PLAYERS
        for p in participants:
            puuid = p.get("puuid")
            slots[p["participantId"] - 1] = puuid.strip() if isinstance(puuid, str) else ""
        present = [s for s in slots if s]
        if len(set(present)) != len(present):
            status[row] = STATUS["duplicate_puuid"]
            continue
        for column, puuid in enumerate(slots):
            if puuid:
                player_idx[row, column] = numbers.setdefault(puuid, len(numbers))
            else:
                slots_without += 1
    return player_idx, status, slots_without, files_read, digest.hexdigest()


def build(dataset: str = DATASET, *, rows_path: Any = None, patch_index_path: Any = None,
          detail_dir: Any = None) -> PlayerMatrix:
    """The player matrix of ``dataset``; the path arguments replace the PRESPEC sources (tests use fake sets)."""

    src = _sources(dataset, rows_path, patch_index_path, detail_dir)
    tick = time.perf_counter()
    match_index, match_keys, patch, rows, rows_by_patch = match_table(src["rows_path"])
    index = key_index(src["patch_index_path"])
    pre_status = np.zeros(match_index.size, dtype=np.int8)
    match_ids: list[str | None] = []
    for row, (match_key, match_patch) in enumerate(zip(match_keys.tolist(), patch.tolist())):
        entry = index.get(bytes(match_key))
        if entry is None:
            pre_status[row] = STATUS["key_not_in_patch_index"]
        elif entry[1] != bytes(match_patch).decode("ascii"):
            pre_status[row] = STATUS["patch_mismatch"]
        match_ids.append(entry[0] if pre_status[row] == MAPPED else None)
    tick_index = time.perf_counter()
    player_idx, file_status, slots_without, files_read, digest = _number_players(match_ids, Path(src["detail_dir"]))
    status = np.where(pre_status != MAPPED, pre_status, file_status).astype(np.int8)
    return PlayerMatrix(match_index=match_index, match_keys=match_keys, player_idx=player_idx, map_status=status,
                        patch=patch, rows=rows, rows_by_patch=rows_by_patch, slots_without_puuid=slots_without,
                        files_read=files_read, files_digest=digest,
                        seconds={"rows_and_patch_index": tick_index - tick, "detail_files": time.perf_counter() - tick_index})


def player_matrix(dataset: str, *, rows_path: Any = None, patch_index_path: Any = None,
                  detail_dir: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """(canonical match keys S64 (M,), player numbers int64 (M, 10)) in ascending match_index; -1 = no player."""

    built = build(dataset, rows_path=rows_path, patch_index_path=patch_index_path, detail_dir=detail_dir)
    return built.match_keys, built.player_idx


def failure_counts(map_status: Any) -> dict[str, int]:
    """Failed matches by reason (every reason, zeros included)."""

    status = np.asarray(map_status)
    return {name: int((status == code).sum()) for name, code in STATUS.items()}


def players_by_match_count(player_idx: Any) -> np.ndarray:
    """Entry k = number of players in exactly k matches (entry 0 is 0); a player is at most once per match."""

    idx = np.asarray(player_idx)
    return np.bincount(np.bincount(idx[idx >= 0]), minlength=1).astype(np.int64)


def summarize(built: PlayerMatrix) -> dict[str, Any]:
    """Aggregate counts of a player matrix (no identifier, no per-player row)."""

    m = int(built.match_index.size)
    failed = int((built.map_status != MAPPED).sum())
    share = failed / m
    identified = built.player_idx >= 0
    per_player = np.bincount(built.player_idx[identified])
    histogram = players_by_match_count(built.player_idx)
    k_m = np.bincount(identified.sum(axis=1), minlength=PLAYERS + 1)
    distribution: dict[str, Any] = {"players": int(per_player.size),
                                    "histogram": {str(k): int(n) for k, n in enumerate(histogram) if n}}
    if per_player.size:
        q50, q90, q99 = np.quantile(per_player, [0.5, 0.9, 0.99]).tolist()
        distribution.update(min=int(per_player.min()), median=q50, p90=q90, p99=q99, max=int(per_player.max()),
                            mean=float(per_player.mean()))
    patches = np.unique(built.patch).tolist()
    return {
        "rows": built.rows, "matches": m, "rows_by_patch": built.rows_by_patch,
        "matches_by_patch": {p.decode("ascii"): int((built.patch == p).sum()) for p in patches},
        "mapping": {"mapped": m - failed, "failed": failed, "failed_share": share, "max_failed_share": MAX_FAILED_SHARE,
                    "blocked": share > MAX_FAILED_SHARE, "by_reason": failure_counts(built.map_status),
                    "failed_by_patch": {p.decode("ascii"): int(((built.patch == p) & (built.map_status != MAPPED)).sum())
                                        for p in patches},
                    "status_codes": {str(MAPPED): "mapped", **{str(code): name for name, code in STATUS.items()}},
                    "rule": "PRESPEC_E3 section 6: more than 1 % failed matches is BLOCKED; otherwise a failed match "
                            "keeps its row with all ten entries -1 (k_m = 0)"},
        "players": {"distinct": int(per_player.size), "identified_slots": int(identified.sum()),
                    "slots_without_puuid": built.slots_without_puuid,
                    "matches_by_identified_players": {str(k): int(n) for k, n in enumerate(k_m) if n}},
        "matches_per_player": distribution,
    }


def run(out_dir: Path = TOG_E3, dataset: str = DATASET, *, rows_path: Any = None, patch_index_path: Any = None,
        detail_dir: Any = None) -> int:
    """Build the matrix and write PLAYERS_<dataset>.npz and .json once; 0 on success, 2 when BLOCKED."""

    started, tick = utc(), time.perf_counter()
    src = _sources(dataset, rows_path, patch_index_path, detail_dir)
    out_dir = Path(out_dir)
    npz_path, json_path = out_dir / f"PLAYERS_{dataset}.npz", out_dir / f"PLAYERS_{dataset}.json"
    for path in (npz_path, json_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; it is written once")
    sources = {name: str(path) for name, path in src.items()}
    missing = {name: str(path) for name, path in src.items() if not path.exists()}
    if missing:
        return _blocked(out_dir, dataset, {"input_missing": missing}, sources=sources)
    built = build(dataset, **src)
    summary = summarize(built)
    provenance = {"sources": sources, "detail_files": {"read": built.files_read, "sha256_of_file_digests": built.files_digest,
                                                       "digest_rule": DIGEST_RULE},
                  "run": run_record(rows=built.rows, matches=summary["matches"],
                                    inputs=[src["rows_path"], src["patch_index_path"]], code=CODE_FILES,
                                    started_utc=started)}
    if summary["mapping"]["blocked"]:
        return _blocked(out_dir, dataset, {"mapping_failed_share_over_max": {
            "failed": summary["mapping"]["failed"], "matches": summary["matches"],
            "failed_share": summary["mapping"]["failed_share"], "max_failed_share": MAX_FAILED_SHARE}},
            **summary, **provenance)
    tick_write = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"match_index": built.match_index, "canonical_match_key": built.match_keys, "player_idx": built.player_idx,
              "map_status": built.map_status, "players_by_match_count": players_by_match_count(built.player_idx)}
    npz_sha = write_npz_once(npz_path, arrays)
    record = {"version": VERSION, "experiment": "E3", "task": "T3.1 player matrix", "dataset": dataset,
              "prespec": "reports/tog_experiments_20261012/PRESPEC_E3_KO.md (sections 2, 3, 6, 9)",
              "order": ORDER, "columns": COLUMNS, "privacy": PRIVACY,
              "npz": str(npz_path), "npz_sha256": npz_sha,
              "npz_arrays": {name: {"dtype": str(a.dtype), "shape": list(a.shape)} for name, a in arrays.items()},
              **summary, **provenance,
              "processes": "main process only, no worker (puuid strings never leave _number_players)",
              "stage_wall_seconds": {**built.seconds, "write_npz": time.perf_counter() - tick_write},
              "wall_seconds": time.perf_counter() - tick, "peak_memory_bytes": peak_memory_bytes(),
              "git_head": git_head(), "finished_utc": utc()}
    write_once(json_path, _jsonable(record))
    mapping, spread = summary["mapping"], summary["matches_per_player"]
    print(f"PLAYERS {dataset}: rows={built.rows} matches={summary['matches']} mapped={mapping['mapped']} "
          f"failed={mapping['failed']} ({mapping['failed_share']:.4%}) players={spread['players']} "
          f"matches/player median={spread.get('median')} max={spread.get('max')} -> {npz_path}")
    return 0


def _blocked(out_dir: Path, dataset: str, reasons: Mapping[str, Any], **extra: Any) -> int:
    """Write E3_BLOCKED_players_<n>.json once (n = the first free number, so earlier attempts stay) and return 2."""

    out_dir.mkdir(parents=True, exist_ok=True)
    k = 1
    while (out_dir / f"E3_BLOCKED_players_{k}.json").exists():
        k += 1
    path = out_dir / f"E3_BLOCKED_players_{k}.json"
    write_once(path, _jsonable({"version": VERSION, "experiment": "E3", "task": "T3.1 player matrix", "dataset": dataset,
                                "blocked": True, "utc": utc(), "reasons": reasons, **extra}))
    print(json.dumps({"BLOCKED": "players", "file": str(path), "reasons": sorted(reasons)}))
    return 2


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dataset", default=DATASET, choices=sorted(SOURCES))
    args = parser.parse_args(argv)
    return run(TOG_E3, args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
