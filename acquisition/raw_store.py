from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


MATCH_ID_RE = re.compile(r"^[A-Z0-9]+_[0-9]+$")


class RawValidationError(ValueError):
    pass


@dataclass(frozen=True)
class StoredPair:
    detail_path: Path
    timeline_path: Path
    detail_sha256: str
    timeline_sha256: str
    detail_bytes: int
    timeline_bytes: int


def _match_id(payload: Dict[str, Any]) -> Optional[str]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("matchId")
    return str(value) if value is not None else None


def validate_detail(
    match_id: str,
    payload: Dict[str, Any],
    *,
    queue_id: int,
    map_id: int,
    game_mode: str,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RawValidationError("detail payload must be an object")
    if _match_id(payload) != match_id:
        raise RawValidationError("detail metadata.matchId does not match filename")
    info = payload.get("info")
    if not isinstance(info, dict):
        raise RawValidationError("detail info is missing")
    if info.get("queueId") != queue_id:
        raise RawValidationError(f"unexpected queueId: {info.get('queueId')}")
    if info.get("mapId") != map_id:
        raise RawValidationError(f"unexpected mapId: {info.get('mapId')}")
    if info.get("gameMode") != game_mode:
        raise RawValidationError(f"unexpected gameMode: {info.get('gameMode')}")
    participants = info.get("participants")
    if not isinstance(participants, list) or len(participants) != 10:
        raise RawValidationError("detail must contain exactly 10 participants")
    if any(not isinstance(participant, dict) for participant in participants):
        raise RawValidationError("detail participants must be objects")
    participant_ids = [participant.get("participantId") for participant in participants]
    if set(participant_ids) != set(range(1, 11)):
        raise RawValidationError("detail participant IDs must be 1..10")
    teams = info.get("teams")
    if not isinstance(teams, list) or len(teams) != 2:
        raise RawValidationError("detail must contain exactly 2 teams")
    if any(not isinstance(team, dict) for team in teams):
        raise RawValidationError("detail teams must be objects")
    if {team.get("teamId") for team in teams} != {100, 200}:
        raise RawValidationError("detail team IDs must be 100 and 200")
    game_version = info.get("gameVersion")
    if not isinstance(game_version, str) or "." not in game_version:
        raise RawValidationError("detail gameVersion is missing or malformed")
    return info


def validate_timeline(match_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RawValidationError("timeline payload must be an object")
    if _match_id(payload) != match_id:
        raise RawValidationError("timeline metadata.matchId does not match filename")
    info = payload.get("info")
    if not isinstance(info, dict):
        raise RawValidationError("timeline info is missing")
    frames = info.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RawValidationError("timeline has no frames")
    if any(not isinstance(frame, dict) for frame in frames):
        raise RawValidationError("timeline frames must be objects")
    previous_timestamp = -1
    for frame in frames:
        timestamp = frame.get("timestamp")
        if not isinstance(timestamp, int) or timestamp < previous_timestamp:
            raise RawValidationError("timeline frame timestamps must be monotonic integers")
        if not isinstance(frame.get("events"), list):
            raise RawValidationError("timeline frame events must be an array")
        previous_timestamp = timestamp
    participants = info.get("participants")
    if not isinstance(participants, list) or len(participants) != 10:
        raise RawValidationError("timeline must contain exactly 10 participants")
    if any(not isinstance(participant, dict) for participant in participants):
        raise RawValidationError("timeline participants must be objects")
    participant_ids = [participant.get("participantId") for participant in participants]
    if set(participant_ids) != set(range(1, 11)):
        raise RawValidationError("timeline participant IDs must be 1..10")
    return info


class RawStore:
    def __init__(self, root: Path, platform: str) -> None:
        self.root = Path(root).resolve()
        self.platform = platform.lower()
        self.detail_dir = self.root / self.platform / "detail"
        self.timeline_dir = self.root / self.platform / "timeline"
        self.quarantine_dir = self.root / self.platform / "quarantine"
        self.detail_dir.mkdir(parents=True, exist_ok=True)
        self.timeline_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_match_id(match_id: str) -> str:
        value = str(match_id)
        if not MATCH_ID_RE.fullmatch(value):
            raise RawValidationError("malformed match ID")
        return value

    @staticmethod
    def _serialize(payload: Dict[str, Any]) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _write_temp(final_path: Path, data: bytes) -> Path:
        temp = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.part"
        )
        with temp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return temp

    def paths(self, match_id: str) -> Tuple[Path, Path]:
        safe_id = self._safe_match_id(match_id)
        return (
            self.detail_dir / f"{safe_id}.json",
            self.timeline_dir / f"{safe_id}.json",
        )

    def write_pair(
        self,
        match_id: str,
        detail: Dict[str, Any],
        timeline: Dict[str, Any],
        *,
        queue_id: int,
        map_id: int,
        game_mode: str,
    ) -> StoredPair:
        detail_info = validate_detail(
            match_id,
            detail,
            queue_id=queue_id,
            map_id=map_id,
            game_mode=game_mode,
        )
        timeline_info = validate_timeline(match_id, timeline)
        detail_ids = {
            participant["participantId"] for participant in detail_info["participants"]
        }
        timeline_ids = {
            participant["participantId"] for participant in timeline_info["participants"]
        }
        if detail_ids != timeline_ids:
            raise RawValidationError("detail/timeline participant IDs disagree")
        detail_bytes = self._serialize(detail)
        timeline_bytes = self._serialize(timeline)
        detail_path, timeline_path = self.paths(match_id)
        detail_temp: Optional[Path] = None
        timeline_temp: Optional[Path] = None
        try:
            detail_temp = self._write_temp(detail_path, detail_bytes)
            timeline_temp = self._write_temp(timeline_path, timeline_bytes)
            # A crash between replacements leaves an unpaired detail, which the
            # downstream pairer safely ignores. A later retry completes it.
            os.replace(detail_temp, detail_path)
            detail_temp = None
            os.replace(timeline_temp, timeline_path)
            timeline_temp = None
        finally:
            for temp in (detail_temp, timeline_temp):
                if temp is not None:
                    try:
                        temp.unlink()
                    except FileNotFoundError:
                        pass

        return StoredPair(
            detail_path=detail_path,
            timeline_path=timeline_path,
            detail_sha256=self._sha256(detail_bytes),
            timeline_sha256=self._sha256(timeline_bytes),
            detail_bytes=len(detail_bytes),
            timeline_bytes=len(timeline_bytes),
        )

    def read_valid_pair(
        self,
        match_id: str,
        *,
        queue_id: int,
        map_id: int,
        game_mode: str,
    ) -> Optional[StoredPair]:
        detail_path, timeline_path = self.paths(match_id)
        if not detail_path.exists() or not timeline_path.exists():
            return None
        try:
            detail_bytes = detail_path.read_bytes()
            timeline_bytes = timeline_path.read_bytes()
            detail = json.loads(detail_bytes)
            timeline = json.loads(timeline_bytes)
            detail_info = validate_detail(
                match_id,
                detail,
                queue_id=queue_id,
                map_id=map_id,
                game_mode=game_mode,
            )
            timeline_info = validate_timeline(match_id, timeline)
            if {
                participant["participantId"]
                for participant in detail_info["participants"]
            } != {
                participant["participantId"]
                for participant in timeline_info["participants"]
            }:
                raise RawValidationError("detail/timeline participant IDs disagree")
        except (OSError, ValueError, json.JSONDecodeError, RawValidationError):
            self.quarantine(match_id, "existing_pair_failed_validation")
            return None
        return StoredPair(
            detail_path=detail_path,
            timeline_path=timeline_path,
            detail_sha256=self._sha256(detail_bytes),
            timeline_sha256=self._sha256(timeline_bytes),
            detail_bytes=len(detail_bytes),
            timeline_bytes=len(timeline_bytes),
        )

    def quarantine(self, match_id: str, reason: str) -> None:
        detail_path, timeline_path = self.paths(match_id)
        safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason)[:80]
        target = self.quarantine_dir / safe_reason
        target.mkdir(parents=True, exist_ok=True)
        for source in (detail_path, timeline_path):
            if source.exists():
                destination = target / source.name
                if destination.exists():
                    destination = target / f"{source.stem}.{uuid.uuid4().hex}.json"
                os.replace(source, destination)
