"""prepare.py — Build the annotation-study dataset (Track A).

Samples matches (seeded, patch-filtered), converts each match's Match-V5
timeline into a compact minimap-replay payload for the human annotation
viewer, and runs the teamfight_v2 detector on the same telemetry. Detector
intervals are written to a separate subdirectory so annotators never see
them (blind annotation).

Output layout (default: outputs/annotation_study/):
    manifest.json                 sampling parameters + match list
    viewer.html                   the annotation UI (copied from this package)
    matches/<match_id>.json       viewer payloads (safe to give annotators)
    detector/<match_id>.json      detector intervals (do NOT give annotators)

Two data sources:
  --source raw    read Match-V5 detail/timeline JSONs (--timeline_dir/--detail_dir)
  --source cache  read the pipeline's match cache (npz + events + meta) — use this
                  for the paper corpus when the raw JSONs are no longer available.
                  Point LOL_OUTPUT_ROOT at the output root containing cache/<dirname>.

Usage:
    # paper corpus from the match cache
    LOL_OUTPUT_ROOT=D:/LOL_Project python -m analysis.annotation_study.prepare \
        --source cache --n_matches 50 --seed 7
    # raw JSONs (e.g. a fresh collection)
    python -m analysis.annotation_study.prepare --n_matches 50 --seed 7 \
        --timeline_dir <dir> --detail_dir <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import cfg
from gameplay.fights import detect_fights
from gameplay.pipeline_cache import parse_timeline_to_minute_cache

# Objective monster/building kinds surfaced in the viewer.
_MONSTER_KINDS = {
    "DRAGON": "dragon",
    "ELDER_DRAGON": "elder",
    "BARON_NASHOR": "baron",
    "RIFTHERALD": "herald",
    "ATAKHAN": "atakhan",
    "HORDE": "horde",
}


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _patch_of(detail: dict) -> str:
    ver = str(detail.get("info", {}).get("gameVersion", ""))
    parts = ver.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else ""


def _team_map(detail: dict) -> Dict[int, int]:
    tm: Dict[int, int] = {}
    for p in detail.get("info", {}).get("participants", []) or []:
        pid = int(p.get("participantId", 0))
        if 1 <= pid <= 10:
            tm[pid] = int(p.get("teamId", 100 if pid <= 5 else 200))
    for pid in range(1, 11):
        tm.setdefault(pid, 100 if pid <= 5 else 200)
    return tm


def build_viewer_payload(match_id: str, timeline: dict, detail: dict) -> Optional[dict]:
    """Compact minimap-replay payload: 60s position frames + positioned events."""
    frames = timeline.get("info", {}).get("frames", []) or []
    if len(frames) < 10:  # skip remakes / very short games
        return None

    participants = []
    for p in detail.get("info", {}).get("participants", []) or []:
        participants.append({
            "pid": int(p.get("participantId", 0)),
            "team": int(p.get("teamId", 0)),
            "champ": str(p.get("championName", "?")),
        })
    if len(participants) != 10:
        return None

    pos_frames: List[dict] = []
    kills: List[dict] = []
    objectives: List[dict] = []
    buildings: List[dict] = []

    for fr in frames:
        ts = int(fr.get("timestamp", 0))
        pf = fr.get("participantFrames", {}) or {}
        pos = []
        levels = []
        for pid in range(1, 11):
            entry = pf.get(str(pid), pf.get(pid, {})) or {}
            p = entry.get("position", {}) or {}
            pos.append([int(p.get("x", 0)), int(p.get("y", 0))])
            levels.append(int(entry.get("level", 1)))
        pos_frames.append({"ts": ts, "pos": pos, "lvl": levels})

        for ev in fr.get("events", []) or []:
            et = str(ev.get("type", ""))
            ets = int(ev.get("timestamp", 0))
            epos = ev.get("position", {}) or {}
            ex, ey = int(epos.get("x", 0)), int(epos.get("y", 0))
            if et == "CHAMPION_KILL":
                kills.append({
                    "ts": ets, "x": ex, "y": ey,
                    "killer": int(ev.get("killerId", 0)),
                    "victim": int(ev.get("victimId", 0)),
                    "assists": [int(a) for a in ev.get("assistingParticipantIds", []) or []],
                })
            elif et == "ELITE_MONSTER_KILL":
                kind = _MONSTER_KINDS.get(str(ev.get("monsterType", "")), "monster")
                objectives.append({
                    "ts": ets, "x": ex, "y": ey, "kind": kind,
                    "team": int(ev.get("killerTeamId", 0)),
                })
            elif et == "BUILDING_KILL":
                kind = "inhib" if str(ev.get("buildingType", "")) == "INHIBITOR_BUILDING" else "tower"
                buildings.append({
                    "ts": ets, "x": ex, "y": ey, "kind": kind,
                    # teamId on BUILDING_KILL is the team that LOST the building
                    "lost_team": int(ev.get("teamId", 0)),
                })

    duration_ms = int(detail.get("info", {}).get("gameDuration", 0)) * 1000
    if duration_ms <= 0:
        duration_ms = pos_frames[-1]["ts"]

    return {
        "match_id": match_id,
        "patch": _patch_of(detail),
        "duration_ms": max(duration_ms, pos_frames[-1]["ts"]),
        "participants": participants,
        "frames": pos_frames,
        "kills": kills,
        "objectives": objectives,
        "buildings": buildings,
    }


def build_viewer_payload_from_pack(match_id: str, pack: dict) -> Optional[dict]:
    """Viewer payload from a pipeline cache pack (npz arrays + events + meta)."""
    minute_ts = pack.get("minute_ts")
    xy = pack.get("xy_raw_minute")
    events = pack.get("events", []) or []
    meta = pack.get("meta", {}) or {}
    if minute_ts is None or xy is None or len(minute_ts) < 10:
        return None

    tm = {int(k): int(v) for k, v in (meta.get("team_map") or {}).items()}
    names = (meta.get("static_meta") or {}).get("champion_name_by_pid", {}) or {}
    participants = [{
        "pid": pid,
        "team": tm.get(pid, 100 if pid <= 5 else 200),
        "champ": str(names.get(str(pid), names.get(pid, "?"))),
    } for pid in range(1, 11)]

    pos_frames = [{
        "ts": int(minute_ts[i]),
        "pos": [[int(xy[i, k, 0]), int(xy[i, k, 1])] for k in range(10)],
        "lvl": [0] * 10,
    } for i in range(len(minute_ts))]

    kills: List[dict] = []
    objectives: List[dict] = []
    buildings: List[dict] = []
    for ev in events:
        et = str(ev.get("type", ""))
        ets = int(ev.get("timestamp", 0))
        epos = ev.get("position", {}) or {}
        ex, ey = int(epos.get("x", 0)), int(epos.get("y", 0))
        if et == "CHAMPION_KILL":
            kills.append({
                "ts": ets, "x": ex, "y": ey,
                "killer": int(ev.get("killerId", 0)),
                "victim": int(ev.get("victimId", 0)),
                "assists": [int(a) for a in ev.get("assistingParticipantIds", []) or []],
            })
        elif et == "ELITE_MONSTER_KILL":
            kind = _MONSTER_KINDS.get(str(ev.get("monsterType", "")), "monster")
            objectives.append({"ts": ets, "x": ex, "y": ey, "kind": kind,
                               "team": int(ev.get("killerTeamId", 0))})
        elif et == "BUILDING_KILL":
            kind = "inhib" if str(ev.get("buildingType", "")) == "INHIBITOR_BUILDING" else "tower"
            buildings.append({"ts": ets, "x": ex, "y": ey, "kind": kind,
                              "lost_team": int(ev.get("teamId", 0))})

    duration_ms = int(minute_ts[-1]) + 60000
    if kills:
        duration_ms = max(duration_ms, max(k["ts"] for k in kills) + 5000)

    return {
        "match_id": match_id,
        "patch": str(meta.get("patch", "")),
        "duration_ms": duration_ms,
        "participants": participants,
        "frames": pos_frames,
        "kills": kills,
        "objectives": objectives,
        "buildings": buildings,
    }


def _digest_fights(fights: List[dict]) -> List[dict]:
    out = []
    for f in fights:
        if not isinstance(f, dict):
            continue
        engage = int(f.get("engage_ts", -1))
        if engage < 0:
            continue
        out.append({
            "engage_ts": engage,
            "first_kill_ts": int(f.get("first_kill_ts", engage)),
            "last_kill_ts": int(f.get("last_kill_ts", engage)),
            "fight_type": str(f.get("fight_type", "unknown")),
            "centroid_x": float(f.get("centroid_x", 0.0)),
            "centroid_y": float(f.get("centroid_y", 0.0)),
            "n_kills": int(f.get("det_kill_count_window", 0)),
        })
    return out


def run_detector(timeline: dict, detail: dict) -> List[dict]:
    """Run teamfight_v2 on raw JSON telemetry; return interval digests."""
    tm = _team_map(detail)
    cache = parse_timeline_to_minute_cache(timeline, tm, detail=detail, cfg_in=cfg)
    if not cache:
        return []
    return _digest_fights(detect_fights(cache, tm) or [])


def _iter_raw(args, allowed, rng):
    """Yield (match_id, payload, detector_digests) from raw Match-V5 JSONs."""
    timeline_dir = Path(args.timeline_dir)
    detail_dir = Path(args.detail_dir)
    files = sorted(p.name for p in timeline_dir.glob("*.json"))
    rng.shuffle(files)
    for name in files:
        match_id = name[:-5]
        detail = _load_json(detail_dir / name)
        if detail is None:
            continue
        if allowed and _patch_of(detail) not in allowed:
            continue
        timeline = _load_json(timeline_dir / name)
        if timeline is None:
            continue
        payload = build_viewer_payload(match_id, timeline, detail)
        if payload is None:
            continue
        yield match_id, payload, run_detector(timeline, detail)


def _iter_cache(args, allowed, rng):
    """Yield (match_id, payload, detector_digests) from the pipeline match cache."""
    from core.config import CACHE_DIR
    from data.cache_io import load_match_cache

    ids = sorted(p.name[:-len(".meta.json")] for p in CACHE_DIR.glob("*.meta.json"))
    if not ids:
        raise SystemExit(f"no cached matches under {CACHE_DIR} "
                         f"(set LOL_OUTPUT_ROOT to the output root containing cache/)")
    rng.shuffle(ids)
    for match_id in ids:
        pack = load_match_cache(match_id)
        if pack is None:
            continue
        patch = str(pack.get("meta", {}).get("patch", ""))
        if allowed and patch not in allowed:
            continue
        payload = build_viewer_payload_from_pack(match_id, pack)
        if payload is None:
            continue
        tm = {int(k): int(v) for k, v in (pack.get("meta", {}).get("team_map") or {}).items()}
        for pid in range(1, 11):
            tm.setdefault(pid, 100 if pid <= 5 else 200)
        detector = _digest_fights(detect_fights(pack, tm) or [])
        yield match_id, payload, detector


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the annotation-study dataset")
    ap.add_argument("--source", choices=("raw", "cache"), default="raw")
    ap.add_argument("--n_matches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="outputs/annotation_study")
    ap.add_argument("--timeline_dir", type=str, default=str(cfg.TIMELINE_DIR))
    ap.add_argument("--detail_dir", type=str, default=str(cfg.DETAIL_DIR))
    ap.add_argument("--patches", type=str, default="15.14,15.15,15.16",
                    help="comma-separated patch allowlist; empty = accept all")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "matches").mkdir(parents=True, exist_ok=True)
    (out_dir / "detector").mkdir(parents=True, exist_ok=True)

    allowed = {p.strip() for p in args.patches.split(",") if p.strip()}
    rng = random.Random(args.seed)
    source = _iter_cache(args, allowed, rng) if args.source == "cache" \
        else _iter_raw(args, allowed, rng)

    picked: List[dict] = []
    for match_id, payload, detector in source:
        with open(out_dir / "matches" / f"{match_id}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        with open(out_dir / "detector" / f"{match_id}.json", "w", encoding="utf-8") as f:
            json.dump({"match_id": match_id, "fights": detector}, f, indent=2)

        picked.append({
            "match_id": match_id,
            "patch": payload["patch"],
            "duration_ms": payload["duration_ms"],
            "n_kills": len(payload["kills"]),
            "n_detected_fights": len(detector),
        })
        print(f"[{len(picked):3d}/{args.n_matches}] {match_id} "
              f"patch={payload['patch']} kills={len(payload['kills'])} "
              f"detected={len(detector)}")
        if len(picked) >= args.n_matches:
            break

    manifest = {
        "source": args.source,
        "seed": args.seed,
        "n_matches": len(picked),
        "patches": sorted(allowed) if allowed else "all",
        "matches": picked,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    viewer_src = Path(__file__).parent / "viewer.html"
    if viewer_src.exists():
        shutil.copy(viewer_src, out_dir / "viewer.html")

    print(f"\nDone: {len(picked)} matches -> {out_dir}")
    print("Give annotators: viewer.html + matches/   (NOT detector/)")


if __name__ == "__main__":
    main()
