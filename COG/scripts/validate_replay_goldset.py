from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


MATCH_ID_RE = re.compile(r"^[A-Z0-9]+_\d+$")
KNOWN_MONSTERS = {
    "DRAGON",
    "BARON_NASHOR",
    "RIFTHERALD",
    "ATAKHAN",
    "HORDE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a replay archive, select a time-stratified full-match gold set, "
            "and run technical checks on teamfight_v2 without exposing detector windows."
        )
    )
    parser.add_argument("--replay-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-matches", type=int, default=100)
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--min-duration-sec", type=int, default=600)
    parser.add_argument("--skip-file-hashes", action="store_true")
    parser.add_argument("--skip-detector", action="store_true")
    return parser.parse_args()


def open_immutable(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON is not an object: {path.name}")
    return value


def utc_iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc).isoformat()


def quantiles(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(p: float) -> float:
        index = (len(ordered) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        if lo == hi:
            return float(ordered[lo])
        return float(ordered[lo] * (hi - index) + ordered[hi] * (index - lo))

    return {
        "min": float(ordered[0]),
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "max": float(ordered[-1]),
    }


def stable_key(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def largest_remainder(total: int, capacities: Mapping[Any, int]) -> Dict[Any, int]:
    keys = [key for key, capacity in capacities.items() if capacity > 0]
    if not keys or total <= 0:
        return {key: 0 for key in capacities}
    total = min(total, sum(capacities[key] for key in keys))
    capacity_total = sum(capacities[key] for key in keys)
    raw = {key: total * capacities[key] / capacity_total for key in keys}
    allocation = {key: min(capacities[key], int(math.floor(raw[key]))) for key in keys}
    remaining = total - sum(allocation.values())
    order = sorted(
        keys,
        key=lambda key: (raw[key] - math.floor(raw[key]), capacities[key], str(key)),
        reverse=True,
    )
    while remaining:
        changed = False
        for key in order:
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:
            break
    return {key: allocation.get(key, 0) for key in capacities}


def add_issue(
    issues: List[dict], severity: str, code: str, message: str, examples: Iterable[str]
) -> None:
    examples = list(examples)
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "count": len(examples),
            "examples": examples[:5],
        }
    )


def validate_archive(args: argparse.Namespace) -> Tuple[List[dict], dict]:
    replay_db = Path(args.replay_db).resolve()
    source_db = Path(args.source_db).resolve()
    raw_root = Path(args.raw_root).resolve()
    replay_conn = open_immutable(replay_db)
    source_conn = open_immutable(source_db)
    try:
        replay_integrity = replay_conn.execute("PRAGMA integrity_check").fetchone()[0]
        source_integrity = source_conn.execute("PRAGMA integrity_check").fetchone()[0]
        rows = [
            dict(row)
            for row in replay_conn.execute(
                """
                SELECT match_id, game_id, platform, public_patch, api_patch,
                       game_creation, detail_sha256, timeline_sha256, status,
                       replay_path, replay_bytes, replay_sha256, last_error
                FROM replays WHERE status='complete'
                ORDER BY game_creation, match_id
                """
            )
        ]
        source_rows = {
            row["match_id"]: dict(row)
            for row in source_conn.execute(
                """
                SELECT match_id, status, public_patch, api_patch, game_creation,
                       detail_sha256, timeline_sha256, detail_bytes, timeline_bytes
                FROM matches
                """
            )
        }
    finally:
        replay_conn.close()
        source_conn.close()

    issues: List[dict] = []
    failures: Dict[str, List[str]] = defaultdict(list)
    records: List[dict] = []
    replay_hashes: List[str] = []

    for index, row in enumerate(rows, start=1):
        match_id = str(row["match_id"])
        platform = str(row["platform"]).lower()
        replay_path = Path(str(row.get("replay_path") or ""))
        detail_path = raw_root / platform / "detail" / f"{match_id}.json"
        timeline_path = raw_root / platform / "timeline" / f"{match_id}.json"
        source = source_rows.get(match_id)
        record = dict(row)
        record.update(
            {
                "detail_path": str(detail_path),
                "timeline_path": str(timeline_path),
                "duration_sec": None,
                "eligible": False,
            }
        )

        if not MATCH_ID_RE.fullmatch(match_id):
            failures["invalid_match_id"].append(match_id)
        if source is None:
            failures["missing_source_row"].append(match_id)
        else:
            if source["status"] != "complete":
                failures["source_not_complete"].append(match_id)
            for field in ("detail_sha256", "timeline_sha256", "public_patch", "api_patch"):
                if str(source.get(field) or "") != str(row.get(field) or ""):
                    failures[f"source_{field}_mismatch"].append(match_id)
            if int(source.get("game_creation") or 0) != int(row.get("game_creation") or 0):
                failures["source_game_creation_mismatch"].append(match_id)

        if not replay_path.is_file():
            failures["missing_replay_file"].append(match_id)
        else:
            actual_size = replay_path.stat().st_size
            if actual_size != int(row.get("replay_bytes") or -1):
                failures["replay_size_mismatch"].append(match_id)
            if not args.skip_file_hashes:
                actual_hash = sha256_file(replay_path)
                replay_hashes.append(actual_hash)
                if actual_hash != str(row.get("replay_sha256") or ""):
                    failures["replay_hash_mismatch"].append(match_id)

        detail = None
        if not detail_path.is_file():
            failures["missing_detail_file"].append(match_id)
        else:
            if not args.skip_file_hashes and sha256_file(detail_path) != str(
                row.get("detail_sha256") or ""
            ):
                failures["detail_hash_mismatch"].append(match_id)
            try:
                detail = load_json(detail_path)
                metadata_match = str(detail.get("metadata", {}).get("matchId", ""))
                if metadata_match and metadata_match != match_id:
                    failures["detail_match_id_mismatch"].append(match_id)
                info = detail.get("info", {}) or {}
                duration = int(info.get("gameDuration", 0) or 0)
                record["duration_sec"] = duration
                detail_creation = int(info.get("gameCreation", 0) or 0)
                if detail_creation and detail_creation != int(row.get("game_creation") or 0):
                    failures["detail_game_creation_mismatch"].append(match_id)
                api_patch = ".".join(str(info.get("gameVersion", "")).split(".")[:2])
                if api_patch and api_patch != str(row.get("api_patch") or ""):
                    failures["detail_api_patch_mismatch"].append(match_id)
            except Exception:
                failures["detail_json_invalid"].append(match_id)

        if not timeline_path.is_file():
            failures["missing_timeline_file"].append(match_id)
        else:
            if not args.skip_file_hashes and sha256_file(timeline_path) != str(
                row.get("timeline_sha256") or ""
            ):
                failures["timeline_hash_mismatch"].append(match_id)

        record["eligible"] = (
            replay_path.is_file()
            and detail_path.is_file()
            and timeline_path.is_file()
            and int(record.get("duration_sec") or 0) >= args.min_duration_sec
            and not any(match_id in values for values in failures.values())
        )
        records.append(record)
        if index % 25 == 0 or index == len(rows):
            print(f"[integrity {index:3d}/{len(rows)}]", flush=True)

    for code, examples in sorted(failures.items()):
        add_issue(issues, "critical", code, code.replace("_", " "), examples)

    duplicate_match_ids = [key for key, count in Counter(r["match_id"] for r in rows).items() if count > 1]
    if duplicate_match_ids:
        add_issue(issues, "critical", "duplicate_match_id", "duplicate replay grain", duplicate_match_ids)
    duplicate_replay_hashes = [key for key, count in Counter(replay_hashes).items() if count > 1]
    if duplicate_replay_hashes:
        examples = []
        for digest in duplicate_replay_hashes:
            examples.extend(r["match_id"] for r in records if r.get("replay_sha256") == digest)
        add_issue(
            issues,
            "high",
            "duplicate_replay_hash",
            "different match rows share an identical replay payload",
            examples,
        )

    small = [r["match_id"] for r in records if int(r.get("replay_bytes") or 0) < 1024 * 1024]
    if small:
        add_issue(
            issues,
            "medium",
            "small_replay_file",
            "replay is below 1 MiB; retain only after manual playback succeeds",
            small,
        )

    if replay_integrity != "ok":
        add_issue(issues, "critical", "replay_db_integrity", str(replay_integrity), [str(replay_db)])
    if source_integrity != "ok":
        add_issue(issues, "critical", "source_db_integrity", str(source_integrity), [str(source_db)])

    sizes = [int(r.get("replay_bytes") or 0) for r in records]
    durations = [int(r["duration_sec"]) for r in records if r.get("duration_sec") is not None]
    patches = Counter((str(r.get("public_patch")), str(r.get("api_patch"))) for r in records)
    severities = Counter(issue["severity"] for issue in issues)
    overall = "fail" if severities["critical"] else ("warning" if issues else "pass")
    report = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "grain": "one archived replay per Match-V5 match_id",
        "counts": {
            "complete_rows": len(records),
            "unique_match_ids": len({r["match_id"] for r in records}),
            "eligible_for_sampling": sum(bool(r["eligible"]) for r in records),
            "total_bytes": sum(sizes),
        },
        "database_integrity": {"replay_manifest": replay_integrity, "source_collector": source_integrity},
        "distribution": {
            "patches": [
                {"public_patch": public, "api_patch": api, "count": count}
                for (public, api), count in sorted(patches.items())
            ],
            "game_creation_utc": {
                "min": utc_iso(min(int(r["game_creation"]) for r in records)),
                "max": utc_iso(max(int(r["game_creation"]) for r in records)),
            },
            "replay_bytes": quantiles(sizes),
            "duration_sec": quantiles(durations),
        },
        "issues": issues,
    }
    return records, report


def select_goldset(records: Sequence[dict], args: argparse.Namespace) -> List[dict]:
    eligible = [dict(row) for row in records if row.get("eligible")]
    if len(eligible) < args.n_matches:
        raise RuntimeError(
            f"only {len(eligible)} eligible matches; need {args.n_matches} after integrity filters"
        )
    by_patch: Dict[str, List[dict]] = defaultdict(list)
    for row in eligible:
        by_patch[str(row.get("public_patch") or "unknown")].append(row)
    patch_alloc = largest_remainder(args.n_matches, {key: len(value) for key, value in by_patch.items()})
    selected: List[dict] = []
    for patch, patch_rows in sorted(by_patch.items()):
        patch_rows.sort(key=lambda row: (int(row["game_creation"]), row["match_id"]))
        bins: Dict[int, List[dict]] = defaultdict(list)
        for index, row in enumerate(patch_rows):
            bin_index = min(args.time_bins - 1, index * args.time_bins // len(patch_rows))
            row["time_bin"] = bin_index
            bins[bin_index].append(row)
        bin_alloc = largest_remainder(patch_alloc[patch], {key: len(value) for key, value in bins.items()})
        for bin_index, bin_rows in sorted(bins.items()):
            ranked = sorted(
                bin_rows,
                key=lambda row: stable_key(args.seed, f"{patch}:{bin_index}:{row['match_id']}"),
            )
            for row in ranked[: bin_alloc[bin_index]]:
                row["stratum"] = f"{patch}/t{bin_index + 1:02d}"
                selected.append(row)

    selected.sort(key=lambda row: stable_key(args.seed, f"queue:{row['match_id']}"))
    for index, row in enumerate(selected, start=1):
        row["match_code"] = f"R{index:03d}"
        row["queue_order"] = index
    if len(selected) != args.n_matches:
        raise AssertionError(f"sampling returned {len(selected)}, expected {args.n_matches}")
    return selected


def team_map(detail: dict) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for participant in detail.get("info", {}).get("participants", []) or []:
        pid = int(participant.get("participantId", 0) or 0)
        if 1 <= pid <= 10:
            mapping[pid] = int(participant.get("teamId", 100 if pid <= 5 else 200))
    for pid in range(1, 11):
        mapping.setdefault(pid, 100 if pid <= 5 else 200)
    return mapping


def digest_fights(fights: Sequence[dict]) -> List[dict]:
    output = []
    for fight in fights:
        engage = int(fight.get("engage_ts", -1))
        if engage < 0:
            continue
        output.append(
            {
                "engage_ts": engage,
                "first_kill_ts": int(fight.get("first_kill_ts", engage)),
                "last_kill_ts": int(fight.get("last_kill_ts", engage)),
                "fight_type": str(fight.get("fight_type", "unknown")),
                "fight_scale": str(fight.get("fight_scale", fight.get("fight_type", "unknown"))),
                "fight_context": str(fight.get("fight_context", "unknown")),
                "fight_label": str(fight.get("fight_label", fight.get("fight_type", "unknown"))),
                "centroid_x": round(float(fight.get("centroid_x", 0.0)), 6),
                "centroid_y": round(float(fight.get("centroid_y", 0.0)), 6),
                "blue_participants": int(fight.get("det_cluster_blue", 0)),
                "red_participants": int(fight.get("det_cluster_red", 0)),
                "n_kills": int(fight.get("det_kill_count_window", 0)),
            }
        )
    return output


def run_detector(selected: Sequence[dict], args: argparse.Namespace, out_dir: Path) -> dict:
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root))
    from analysis.annotation_study.prepare import build_viewer_payload
    from core.config import cfg
    from gameplay.fights import detect_fights
    from gameplay.pipeline_cache import parse_timeline_to_minute_cache

    matches_dir = out_dir / "matches"
    detector_dir = out_dir / "detector"
    matches_dir.mkdir(parents=True, exist_ok=True)
    detector_dir.mkdir(parents=True, exist_ok=True)

    exceptions: List[dict] = []
    boundary_failures: List[str] = []
    nondeterministic: List[str] = []
    overlaps: List[str] = []
    diag_errors: List[str] = []
    unknown_monsters: Counter[str] = Counter()
    fight_types: Counter[str] = Counter()
    fight_scales: Counter[str] = Counter()
    fight_contexts: Counter[str] = Counter()
    counts: List[int] = []
    zero_fight_matches: List[str] = []
    candidate_rows: List[dict] = []

    for index, row in enumerate(selected, start=1):
        match_id = row["match_id"]
        try:
            detail = load_json(Path(row["detail_path"]))
            timeline = load_json(Path(row["timeline_path"]))
            mapping = team_map(detail)
            cache = parse_timeline_to_minute_cache(timeline, mapping, detail=detail, cfg_in=cfg)
            if not cache:
                raise RuntimeError("timeline parser returned an empty cache")
            first = digest_fights(detect_fights(cache, mapping) or [])
            first_diag = dict(cache.get("fight_detect_diag", {}) or {})
            second = digest_fights(detect_fights(cache, mapping) or [])
            if first != second:
                nondeterministic.append(match_id)

            duration_ms = int(row.get("duration_sec") or 0) * 1000
            for fight_index, fight in enumerate(first):
                engage = fight["engage_ts"]
                first_kill = fight["first_kill_ts"]
                last_kill = fight["last_kill_ts"]
                if not (0 <= engage < first_kill <= last_kill <= duration_ms + 60000):
                    boundary_failures.append(f"{match_id}:{fight_index}")
                if fight["n_kills"] < 1:
                    boundary_failures.append(f"{match_id}:{fight_index}:zero_kills")
                if fight_index and engage < first[fight_index - 1]["last_kill_ts"]:
                    overlaps.append(f"{match_id}:{fight_index - 1}-{fight_index}")
                fight_types[fight["fight_type"]] += 1
                fight_scales[fight["fight_scale"]] += 1
                fight_contexts[fight["fight_context"]] += 1
                candidate_rows.append(
                    {
                        "match_id": match_id,
                        "match_code": row["match_code"],
                        "fight_index": fight_index,
                        **fight,
                    }
                )
            counts.append(len(first))
            if not first:
                zero_fight_matches.append(match_id)
            if first_diag.get("errors"):
                diag_errors.append(match_id)

            for frame in timeline.get("info", {}).get("frames", []) or []:
                for event in frame.get("events", []) or []:
                    if str(event.get("type", "")) == "ELITE_MONSTER_KILL":
                        monster = str(event.get("monsterType", "") or "UNKNOWN").upper()
                        if monster not in KNOWN_MONSTERS:
                            unknown_monsters[monster] += 1

            payload = build_viewer_payload(match_id, timeline, detail)
            if payload is None:
                raise RuntimeError("could not build internal telemetry payload")
            (matches_dir / f"{match_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            (detector_dir / f"{match_id}.json").write_text(
                json.dumps({"match_id": match_id, "fights": first}, indent=2),
                encoding="utf-8",
            )
            row["n_detected_fights"] = len(first)
            print(
                f"[detector {index:3d}/{len(selected)}] {row['match_code']} fights={len(first)}",
                flush=True,
            )
        except Exception as exc:
            exceptions.append({"match_id": match_id, "error": f"{type(exc).__name__}: {exc}"})

    with (out_dir / "candidate_intervals.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "match_id",
            "match_code",
            "fight_index",
            "engage_ts",
            "first_kill_ts",
            "last_kill_ts",
            "fight_type",
            "fight_scale",
            "fight_context",
            "fight_label",
            "centroid_x",
            "centroid_y",
            "blue_participants",
            "red_participants",
            "n_kills",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    report = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "detector": "teamfight_v2",
        "sample_matches": len(selected),
        "matches_processed": len(selected) - len(exceptions),
        "candidate_fights": len(candidate_rows),
        "fights_per_match": {
            "mean": statistics.mean(counts) if counts else 0.0,
            "median": statistics.median(counts) if counts else 0.0,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
        },
        "fight_types": dict(sorted(fight_types.items())),
        "fight_scales": dict(sorted(fight_scales.items())),
        "fight_contexts": dict(sorted(fight_contexts.items())),
        "zero_fight_matches": len(zero_fight_matches),
        "checks": {
            "exceptions": exceptions,
            "nondeterministic_matches": nondeterministic,
            "boundary_failures": boundary_failures,
            "overlapping_detection_intervals": overlaps,
            "diagnostic_error_matches": diag_errors,
            "unknown_monster_events": dict(sorted(unknown_monsters.items())),
        },
        "technical_status": (
            "pass"
            if not (exceptions or nondeterministic or boundary_failures or diag_errors)
            else "fail"
        ),
        "performance_metrics_status": "pending_human_annotations",
        "performance_metrics_note": (
            "Precision, recall, and F1 are intentionally not computed until at least "
            "two annotators complete full-ROFL blind labels."
        ),
    }
    return report


def render_form(template: Path, queue: Sequence[dict], destination: Path) -> None:
    html = template.read_text(encoding="utf-8")
    marker = "__QUEUE_JSON__"
    if marker not in html:
        raise ValueError(f"annotation template does not contain {marker}")
    safe_json = json.dumps(list(queue), ensure_ascii=False).replace("</", "<\\/")
    destination.write_text(html.replace(marker, safe_json), encoding="utf-8")


def write_package(
    selected: Sequence[dict], data_report: dict, technical_report: dict, args: argparse.Namespace
) -> None:
    out_dir = Path(args.out).resolve()
    annotator_dir = out_dir / "annotator"
    annotator_dir.mkdir(parents=True, exist_ok=True)

    queue = [
        {
            "queue_order": row["queue_order"],
            "match_code": row["match_code"],
            "match_id": row["match_id"],
            "replay_path": row["replay_path"],
            "patch": row["public_patch"],
            "game_creation_utc": utc_iso(int(row["game_creation"])),
            "duration_ms": int(row.get("duration_sec") or 0) * 1000,
        }
        for row in selected
    ]
    (annotator_dir / "queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (annotator_dir / "replay_queue.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    blank = {
        "annotator": "REPLACE_WITH_UNIQUE_ID",
        "protocol_version": "rofl_full_match_v1",
        "matches": {row["match_id"]: [] for row in queue},
    }
    (annotator_dir / "annotations_blank.json").write_text(
        json.dumps(blank, indent=2), encoding="utf-8"
    )
    template = Path(__file__).resolve().parents[1] / "analysis" / "annotation_study" / "replay_annotation_form.html"
    render_form(template, queue, annotator_dir / "annotation_form.html")

    manifest = {
        "protocol_version": "rofl_full_match_v1",
        "seed": args.seed,
        "sampling": {
            "n_matches": args.n_matches,
            "time_bins_per_patch": args.time_bins,
            "minimum_duration_sec": args.min_duration_sec,
            "unit": "full match",
            "source_population": "integrity-verified complete replay archive",
        },
        "matches": [
            {
                "match_id": row["match_id"],
                "match_code": row["match_code"],
                "queue_order": row["queue_order"],
                "stratum": row["stratum"],
                "public_patch": row["public_patch"],
                "api_patch": row["api_patch"],
                "game_creation": row["game_creation"],
                "duration_sec": row["duration_sec"],
                "replay_path": row["replay_path"],
                "replay_bytes": row["replay_bytes"],
                "replay_sha256": row["replay_sha256"],
                "detail_sha256": row["detail_sha256"],
                "timeline_sha256": row["timeline_sha256"],
                "n_detected_fights": row.get("n_detected_fights"),
            }
            for row in selected
        ],
    }
    (out_dir / "goldset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "data_quality_report.json").write_text(
        json.dumps(data_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "technical_test_report.json").write_text(
        json.dumps(technical_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    protocol = """# Full-ROFL blind annotation protocol

Give each annotator only this `annotator` directory and read-only access to the replay paths in `replay_queue.csv`. Never give annotators the parent `detector`, `matches`, `candidate_intervals.csv`, or `technical_test_report.json` files.

1. Use a unique annotator ID. Annotators work independently and do not discuss intervals until scoring is complete.
2. Open each full `.rofl` in the League Client and watch the entire match. Do not navigate using detector-selected clips or candidate timestamps.
3. Mark every combat engagement from initiation to disengage/wipe, including engagements with no kill.
4. Use `teamfight` for 3v3 or larger, `skirmish` for roughly 2v2, and `uncertain` when the boundary/type is ambiguous.
5. Record League Client clock times in `annotation_form.html`. Export JSON after every session; the browser also keeps a local autosave.
6. Use two annotators for all 100 matches. Resolve disagreements only after the independent exports are frozen.

After both exports exist, score them with `analysis.annotation_study.score` against the parent package. Report detector-vs-human results beside inter-annotator agreement. Precision/recall/F1 before this step are not valid performance claims.
"""
    (annotator_dir / "PROTOCOL.md").write_text(protocol, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.n_matches <= 0 or args.time_bins <= 0:
        raise SystemExit("--n-matches and --time-bins must be positive")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records, data_report = validate_archive(args)
    if data_report["overall_status"] == "fail":
        (out_dir / "data_quality_report.json").write_text(
            json.dumps(data_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("critical data-quality failures found; sampling and detector run stopped")
        return 2

    selected = select_goldset(records, args)
    technical_report: dict
    if args.skip_detector:
        technical_report = {
            "technical_status": "not_run",
            "performance_metrics_status": "pending_human_annotations",
        }
    else:
        technical_report = run_detector(selected, args, out_dir)
    write_package(selected, data_report, technical_report, args)
    print(json.dumps({
        "data_quality": data_report["overall_status"],
        "selected": len(selected),
        "detector": technical_report.get("technical_status"),
        "out": str(out_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
