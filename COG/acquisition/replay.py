from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sqlite3
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPLAY_SCHEMA_VERSION = 1


class LeagueClientUnavailable(RuntimeError):
    """Raised when the local League Client cannot serve replay requests."""


class ReplayAPIError(RuntimeError):
    def __init__(self, status_code: int, path: str) -> None:
        super().__init__(f"League replay API returned HTTP {status_code} for {path}")
        self.status_code = int(status_code)
        self.path = path


class ReplayDownloadError(RuntimeError):
    """Raised when a replay exists but cannot be downloaded safely."""


@dataclass(frozen=True)
class ReplayCandidate:
    match_id: str
    game_id: str
    platform: str
    public_patch: Optional[str]
    api_patch: Optional[str]
    game_creation: Optional[int]
    detail_sha256: Optional[str]
    timeline_sha256: Optional[str]


@dataclass(frozen=True)
class ReplayCollectorConfig:
    source_database: Path
    output_root: Path
    league_lockfile: Path
    platform: str = "kr"
    max_replays: int = 500
    max_candidates: int = 2000
    max_storage_bytes: int = 8 * 1024**3
    min_free_bytes: int = 10 * 1024**3
    poll_seconds: float = 1.0
    download_timeout_seconds: float = 180.0
    request_delay_seconds: float = 0.5
    delete_client_copy: bool = True
    oldest_first: bool = True
    min_replay_bytes: int = 1024

    def __post_init__(self) -> None:
        if self.max_replays < 1 or self.max_candidates < 1:
            raise ValueError("replay and candidate limits must be positive")
        if self.max_storage_bytes < 1:
            raise ValueError("max_storage_bytes must be positive")
        if self.min_free_bytes < 0:
            raise ValueError("min_free_bytes must be non-negative")
        if self.poll_seconds <= 0 or self.download_timeout_seconds <= 0:
            raise ValueError("poll and timeout values must be positive")
        if self.request_delay_seconds < 0:
            raise ValueError("request_delay_seconds must be non-negative")
        if self.min_replay_bytes < 1:
            raise ValueError("min_replay_bytes must be positive")

    @property
    def archive_dir(self) -> Path:
        return self.output_root / self.platform.lower() / "rofl"

    @property
    def state_dir(self) -> Path:
        return self.output_root / "_replay_state"

    @property
    def state_database(self) -> Path:
        return self.state_dir / "replay_manifest.sqlite3"

    @property
    def manifest_jsonl(self) -> Path:
        return self.state_dir / "manifest.jsonl"


@dataclass
class ReplayCollectionSummary:
    considered: int = 0
    downloaded: int = 0
    already_complete: int = 0
    unavailable: int = 0
    failed: int = 0
    bytes_archived: int = 0
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_game_id(match_id: str, platform: str) -> str:
    prefix = platform.upper() + "_"
    if not match_id.startswith(prefix):
        raise ValueError(f"match_id must start with {prefix}")
    game_id = match_id[len(prefix) :]
    if not game_id.isdigit():
        raise ValueError("match_id game component must be numeric")
    return game_id


def load_replay_candidates(
    database: Path,
    *,
    platform: str = "kr",
    oldest_first: bool = True,
) -> List[ReplayCandidate]:
    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    uri = f"file:{database.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    direction = "ASC" if oldest_first else "DESC"
    try:
        rows = conn.execute(
            f"""
            SELECT match_id, platform, public_patch, api_patch, game_creation,
                   detail_sha256, timeline_sha256
            FROM matches
            WHERE status='complete' AND lower(platform)=?
            ORDER BY game_creation {direction}, match_id {direction}
            """,
            (platform.lower(),),
        ).fetchall()
    finally:
        conn.close()
    return [
        ReplayCandidate(
            match_id=str(row["match_id"]),
            game_id=_parse_game_id(str(row["match_id"]), platform),
            platform=str(row["platform"]),
            public_patch=row["public_patch"],
            api_patch=row["api_patch"],
            game_creation=row["game_creation"],
            detail_sha256=row["detail_sha256"],
            timeline_sha256=row["timeline_sha256"],
        )
        for row in rows
    ]


class LeagueReplayClient:
    """Minimal authenticated client for the local League replay service.

    The League lockfile password is read only in memory and is never returned,
    persisted, or included in exception messages.
    """

    def __init__(self, lockfile: Path) -> None:
        self.lockfile = Path(lockfile)
        self._ssl_context = ssl._create_unverified_context()

    def _connection(self) -> Tuple[str, Dict[str, str]]:
        try:
            parts = self.lockfile.read_text(encoding="utf-8").strip().split(":")
        except OSError as exc:
            raise LeagueClientUnavailable(
                "League Client lockfile is unavailable; open and log into the client"
            ) from exc
        if len(parts) != 5:
            raise LeagueClientUnavailable("League Client lockfile has an invalid format")
        _, _, port, password, protocol = parts
        token = base64.b64encode(f"riot:{password}".encode("utf-8")).decode("ascii")
        return f"{protocol}://127.0.0.1:{port}", {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Any] = None,
        timeout: float = 20.0,
    ) -> Any:
        base_url, headers = self._connection()
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                context=self._ssl_context,
                timeout=timeout,
            ) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise ReplayAPIError(exc.code, path) from None
        except (OSError, urllib.error.URLError) as exc:
            raise LeagueClientUnavailable(
                "League Client replay service is not reachable"
            ) from exc
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReplayDownloadError(
                f"League replay API returned invalid JSON for {path}"
            ) from exc

    def configuration(self) -> Dict[str, Any]:
        data = self._request("GET", "/lol-replays/v1/configuration")
        if not isinstance(data, dict):
            raise ReplayDownloadError("invalid replay configuration response")
        return data

    def replay_directory(self) -> Path:
        data = self._request("GET", "/lol-replays/v1/rofls/path")
        if not isinstance(data, str) or not data.strip():
            raise ReplayDownloadError("League Client returned no replay directory")
        return Path(data)

    def create_metadata(self, game_id: str) -> None:
        self._request("POST", f"/lol-replays/v2/metadata/{game_id}/create", {})

    def metadata(self, game_id: str) -> Dict[str, Any]:
        data = self._request("GET", f"/lol-replays/v1/metadata/{game_id}")
        if not isinstance(data, dict):
            raise ReplayDownloadError("invalid replay metadata response")
        return data

    def start_download(self, game_id: str) -> None:
        self._request("POST", f"/lol-replays/v1/rofls/{game_id}/download", {})

    def download(self, candidate: ReplayCandidate, config: ReplayCollectorConfig) -> Path:
        client_config = self.configuration()
        if not bool(client_config.get("isLoggedIn")):
            raise LeagueClientUnavailable("League Client is not logged in")
        if not bool(client_config.get("isReplaysEnabled")):
            raise LeagueClientUnavailable("League Client replays are disabled")
        if bool(client_config.get("isPatching")):
            raise LeagueClientUnavailable("League Client is patching")
        if bool(client_config.get("isPlayingGame")) or bool(
            client_config.get("isPlayingReplay")
        ):
            raise LeagueClientUnavailable("League Client is currently busy")

        replay_dir = self.replay_directory()
        source = replay_dir / f"{candidate.platform.upper()}-{candidate.game_id}.rofl"
        if source.is_file() and source.stat().st_size >= config.min_replay_bytes:
            return source

        self.create_metadata(candidate.game_id)
        metadata = self.metadata(candidate.game_id)
        state = str(metadata.get("state") or "")
        if state != "watch" or not source.is_file():
            self.start_download(candidate.game_id)

        deadline = time.monotonic() + config.download_timeout_seconds
        while time.monotonic() < deadline:
            metadata = self.metadata(candidate.game_id)
            state = str(metadata.get("state") or "")
            if state == "watch" and source.is_file():
                if source.stat().st_size < config.min_replay_bytes:
                    raise ReplayDownloadError(
                        f"replay file is unexpectedly small for {candidate.match_id}"
                    )
                return source
            if state in {"expired", "lost", "unavailable", "error"}:
                raise ReplayDownloadError(
                    f"replay entered terminal state {state!r} for {candidate.match_id}"
                )
            time.sleep(config.poll_seconds)
        raise ReplayDownloadError(f"replay download timed out for {candidate.match_id}")


class ReplayManifest:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _initialize(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS replays (
                match_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                public_patch TEXT,
                api_patch TEXT,
                game_creation INTEGER,
                detail_sha256 TEXT,
                timeline_sha256 TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt_at INTEGER,
                completed_at INTEGER,
                replay_path TEXT,
                replay_bytes INTEGER,
                replay_sha256 TEXT,
                last_error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_replays_status
                ON replays(status, game_creation);
            """
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO metadata(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(REPLAY_SCHEMA_VERSION),),
            )

    def close(self) -> None:
        self.conn.close()

    def register(self, candidates: Iterable[ReplayCandidate]) -> None:
        rows = [
            (
                c.match_id,
                c.game_id,
                c.platform,
                c.public_patch,
                c.api_patch,
                c.game_creation,
                c.detail_sha256,
                c.timeline_sha256,
            )
            for c in candidates
        ]
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO replays(
                    match_id, game_id, platform, public_patch, api_patch,
                    game_creation, detail_sha256, timeline_sha256
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id) DO UPDATE SET
                    public_patch=excluded.public_patch,
                    api_patch=excluded.api_patch,
                    game_creation=excluded.game_creation,
                    detail_sha256=excluded.detail_sha256,
                    timeline_sha256=excluded.timeline_sha256
                """,
                rows,
            )

    def status(self, match_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT status FROM replays WHERE match_id=?", (match_id,)
        ).fetchone()
        return str(row["status"]) if row is not None else None

    def mark_attempt(self, match_id: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE replays
                SET status='downloading', attempts=attempts+1,
                    last_attempt_at=?, last_error=NULL
                WHERE match_id=?
                """,
                (int(time.time()), match_id),
            )

    def mark_complete(
        self,
        match_id: str,
        path: Path,
        replay_bytes: int,
        replay_sha256: str,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE replays
                SET status='complete', completed_at=?, replay_path=?,
                    replay_bytes=?, replay_sha256=?, last_error=NULL
                WHERE match_id=?
                """,
                (
                    int(time.time()),
                    str(path),
                    int(replay_bytes),
                    replay_sha256,
                    match_id,
                ),
            )

    def mark_error(self, match_id: str, status: str, error: str) -> None:
        if status not in {"unavailable", "failed"}:
            raise ValueError("invalid replay error status")
        with self.conn:
            self.conn.execute(
                "UPDATE replays SET status=?, last_error=? WHERE match_id=?",
                (status, str(error)[:500], match_id),
            )

    def complete_totals(self) -> Tuple[int, int]:
        row = self.conn.execute(
            """
            SELECT count(*) AS n, coalesce(sum(replay_bytes), 0) AS total_bytes
            FROM replays WHERE status='complete'
            """
        ).fetchone()
        return int(row["n"]), int(row["total_bytes"])

    def export_jsonl(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        rows = self.conn.execute(
            """
            SELECT match_id, game_id, platform, public_patch, api_patch,
                   game_creation, detail_sha256, timeline_sha256, status,
                   attempts, last_attempt_at, completed_at, replay_path,
                   replay_bytes, replay_sha256, last_error
            FROM replays ORDER BY game_creation, match_id
            """
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def archive_replay(
    source: Path,
    destination: Path,
    *,
    delete_source: bool,
    min_replay_bytes: int,
) -> Tuple[int, str]:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise ReplayDownloadError(f"downloaded replay is missing: {source.name}")
    if source.stat().st_size < min_replay_bytes:
        raise ReplayDownloadError(f"downloaded replay is too small: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
            while True:
                chunk = input_handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        size = temporary.stat().st_size
        if size < min_replay_bytes:
            raise ReplayDownloadError(f"archived replay is too small: {source.name}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if delete_source and source.resolve() != destination.resolve():
        source.unlink()
    return size, digest.hexdigest()


class ReplayCollector:
    def __init__(
        self,
        config: ReplayCollectorConfig,
        client: Optional[LeagueReplayClient] = None,
        manifest: Optional[ReplayManifest] = None,
    ) -> None:
        self.config = config
        self.client = client or LeagueReplayClient(config.league_lockfile)
        self.manifest = manifest or ReplayManifest(config.state_database)
        self._owns_manifest = manifest is None

    def close(self) -> None:
        if self._owns_manifest:
            self.manifest.close()

    def _destination(self, candidate: ReplayCandidate) -> Path:
        return self.config.archive_dir / (
            f"{candidate.platform.upper()}-{candidate.game_id}.rofl"
        )

    def collect(self, *, dry_run: bool = False) -> ReplayCollectionSummary:
        summary = ReplayCollectionSummary()
        candidates = load_replay_candidates(
            self.config.source_database,
            platform=self.config.platform,
            oldest_first=self.config.oldest_first,
        )
        self.manifest.register(candidates)
        complete_count, complete_bytes = self.manifest.complete_totals()
        summary.already_complete = complete_count
        summary.bytes_archived = complete_bytes

        if dry_run:
            summary.considered = min(len(candidates), self.config.max_candidates)
            summary.already_complete = complete_count
            summary.stop_reason = "dry_run"
            self.manifest.export_jsonl(self.config.manifest_jsonl)
            return summary

        attempted = 0
        try:
            for candidate in candidates:
                if complete_count >= self.config.max_replays:
                    summary.stop_reason = "max_replays"
                    break
                if attempted >= self.config.max_candidates:
                    summary.stop_reason = "max_candidates"
                    break
                if complete_bytes >= self.config.max_storage_bytes:
                    summary.stop_reason = "max_storage_bytes"
                    break
                self.config.output_root.mkdir(parents=True, exist_ok=True)
                free_bytes = shutil.disk_usage(self.config.output_root).free
                if free_bytes < self.config.min_free_bytes:
                    summary.stop_reason = "min_free_bytes"
                    break

                destination = self._destination(candidate)
                status = self.manifest.status(candidate.match_id)
                if status == "complete" and destination.is_file():
                    continue
                if status == "unavailable":
                    continue
                if destination.is_file() and destination.stat().st_size >= self.config.min_replay_bytes:
                    size = destination.stat().st_size
                    replay_hash = sha256_file(destination)
                    self.manifest.mark_complete(
                        candidate.match_id, destination, size, replay_hash
                    )
                    complete_count += 1
                    complete_bytes += size
                    summary.already_complete += 1
                    summary.bytes_archived = complete_bytes
                    continue

                attempted += 1
                summary.considered += 1
                self.manifest.mark_attempt(candidate.match_id)
                try:
                    source = self.client.download(candidate, self.config)
                    size, replay_hash = archive_replay(
                        source,
                        destination,
                        delete_source=self.config.delete_client_copy,
                        min_replay_bytes=self.config.min_replay_bytes,
                    )
                    self.manifest.mark_complete(
                        candidate.match_id, destination, size, replay_hash
                    )
                    summary.downloaded += 1
                    complete_count += 1
                    complete_bytes += size
                    summary.bytes_archived = complete_bytes
                except ReplayAPIError as exc:
                    if exc.status_code == 404:
                        self.manifest.mark_error(
                            candidate.match_id, "unavailable", str(exc)
                        )
                        summary.unavailable += 1
                    else:
                        self.manifest.mark_error(candidate.match_id, "failed", str(exc))
                        summary.failed += 1
                except ReplayDownloadError as exc:
                    self.manifest.mark_error(candidate.match_id, "failed", str(exc))
                    summary.failed += 1
                if self.config.request_delay_seconds:
                    time.sleep(self.config.request_delay_seconds)
        finally:
            self.manifest.export_jsonl(self.config.manifest_jsonl)
        return summary
