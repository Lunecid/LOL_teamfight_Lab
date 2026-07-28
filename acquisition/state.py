from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 3
MAX_MATCH_ATTEMPTS = 8


@dataclass(frozen=True)
class PendingMatch:
    match_id: str
    platform: str
    source_puuid: Optional[str]
    attempts: int


@dataclass(frozen=True)
class PlayerCursor:
    puuid: str
    last_scan_end: Optional[int]
    scan_window_start: Optional[int]
    scan_window_end: Optional[int]
    scan_next_start: int


class CollectionState:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def close(self) -> None:
        self.conn.close()

    def _initialize(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS players (
                puuid TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                first_seen INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                current_tier TEXT NOT NULL,
                current_league_points INTEGER,
                active INTEGER NOT NULL DEFAULT 1,
                last_scan_end INTEGER,
                last_scan_attempt INTEGER,
                scan_window_start INTEGER,
                scan_window_end INTEGER,
                scan_next_start INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS rank_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at INTEGER NOT NULL,
                platform TEXT NOT NULL,
                queue TEXT NOT NULL,
                tier TEXT NOT NULL,
                puuid TEXT NOT NULL,
                league_points INTEGER,
                wins INTEGER,
                losses INTEGER,
                veteran INTEGER,
                inactive INTEGER,
                fresh_blood INTEGER,
                hot_streak INTEGER,
                UNIQUE(captured_at, platform, queue, tier, puuid)
            );
            CREATE INDEX IF NOT EXISTS idx_rank_snapshot_time
                ON rank_snapshots(captured_at, tier);

            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                source_puuid TEXT,
                discovered_at INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt_at INTEGER,
                last_http_status INTEGER,
                last_error TEXT,
                completed_at INTEGER,
                retrieved_at INTEGER,
                queue_id INTEGER,
                map_id INTEGER,
                game_mode TEXT,
                game_version TEXT,
                api_patch TEXT,
                public_patch TEXT,
                game_creation INTEGER,
                detail_sha256 TEXT,
                timeline_sha256 TEXT,
                detail_bytes INTEGER,
                timeline_bytes INTEGER,
                exclusion_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_matches_status
                ON matches(status, attempts, discovered_at);

            CREATE TABLE IF NOT EXISTS collection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        current = self.get_metadata("schema_version")
        if current is None:
            self.set_metadata("schema_version", str(SCHEMA_VERSION))
        elif int(current) != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported collector database schema {current}; expected {SCHEMA_VERSION}"
            )

    def set_metadata(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO metadata(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, str(value)),
            )

    def get_metadata(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def metadata_int(self, key: str) -> Optional[int]:
        value = self.get_metadata(key)
        return int(value) if value is not None else None

    def record_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        safe_payload = {
            key: value
            for key, value in payload.items()
            if "key" not in key.lower() and "token" not in key.lower()
        }
        with self.conn:
            self.conn.execute(
                "INSERT INTO collection_events(occurred_at, event_type, payload_json) VALUES(?, ?, ?)",
                (int(time.time()), event_type, json.dumps(safe_payload, sort_keys=True)),
            )

    @staticmethod
    def _bool_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return 1 if bool(value) else 0

    def record_rank_entries(
        self,
        *,
        captured_at: int,
        platform: str,
        queue: str,
        tier: str,
        entries: Sequence[Dict[str, Any]],
    ) -> Tuple[int, int]:
        snapshot_rows: List[Tuple[Any, ...]] = []
        player_rows: List[Tuple[Any, ...]] = []
        skipped = 0
        for entry in entries:
            puuid = entry.get("puuid")
            if not puuid:
                skipped += 1
                continue
            normalized_puuid = str(puuid)
            lp = entry.get("leaguePoints")
            snapshot_rows.append(
                (
                    int(captured_at),
                    platform,
                    queue,
                    tier.upper(),
                    normalized_puuid,
                    int(lp) if lp is not None else None,
                    int(entry["wins"]) if entry.get("wins") is not None else None,
                    int(entry["losses"]) if entry.get("losses") is not None else None,
                    self._bool_int(entry.get("veteran")),
                    self._bool_int(entry.get("inactive")),
                    self._bool_int(entry.get("freshBlood")),
                    self._bool_int(entry.get("hotStreak")),
                )
            )
            player_rows.append(
                (
                    normalized_puuid,
                    platform,
                    int(captured_at),
                    int(captured_at),
                    tier.upper(),
                    int(lp) if lp is not None else None,
                )
            )

        with self.conn:
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO rank_snapshots(
                    captured_at, platform, queue, tier, puuid, league_points,
                    wins, losses, veteran, inactive, fresh_blood, hot_streak
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                snapshot_rows,
            )
            self.conn.executemany(
                """
                INSERT INTO players(
                    puuid, platform, first_seen, last_seen, current_tier,
                    current_league_points, active
                ) VALUES(?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(puuid) DO UPDATE SET
                    platform=excluded.platform,
                    last_seen=excluded.last_seen,
                    current_tier=excluded.current_tier,
                    current_league_points=excluded.current_league_points,
                    active=1
                """,
                player_rows,
            )
        return len(player_rows), skipped

    def finalize_rank_snapshot(self, platform: str, captured_at: int) -> None:
        """Deactivate players absent from the newly completed roster snapshot."""

        with self.conn:
            self.conn.execute(
                """
                UPDATE players
                SET active=CASE WHEN last_seen=? THEN 1 ELSE 0 END
                WHERE platform=?
                """,
                (int(captured_at), platform),
            )

    def players_due(self, platform: str, limit: int) -> List[PlayerCursor]:
        rows = self.conn.execute(
            """
            SELECT puuid, last_scan_end, scan_window_start, scan_window_end,
                   scan_next_start
            FROM players
            WHERE platform=? AND (active=1 OR scan_window_end IS NOT NULL)
            ORDER BY
                CASE WHEN scan_window_end IS NOT NULL THEN 0
                     WHEN last_scan_end IS NULL THEN 1 ELSE 2 END,
                COALESCE(last_scan_end, 0),
                puuid
            LIMIT ?
            """,
            (platform, int(limit)),
        ).fetchall()
        return [
            PlayerCursor(
                puuid=str(row["puuid"]),
                last_scan_end=(
                    int(row["last_scan_end"])
                    if row["last_scan_end"] is not None
                    else None
                ),
                scan_window_start=(
                    int(row["scan_window_start"])
                    if row["scan_window_start"] is not None
                    else None
                ),
                scan_window_end=(
                    int(row["scan_window_end"])
                    if row["scan_window_end"] is not None
                    else None
                ),
                scan_next_start=int(row["scan_next_start"] or 0),
            )
            for row in rows
        ]

    def mark_player_scan_attempt(self, puuid: str, attempted_at: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE players SET last_scan_attempt=? WHERE puuid=?",
                (int(attempted_at), puuid),
            )

    def commit_player_scan_page(
        self,
        *,
        puuid: str,
        platform: str,
        window_start: int,
        window_end: int,
        next_start: int,
        window_complete: bool,
        match_ids: Iterable[str],
        discovered_at: int,
    ) -> int:
        """Atomically enqueue a page and advance its player scan cursor.

        If the process dies before this transaction commits, neither the IDs nor
        the cursor advance. The same page is then safely re-requested.
        """

        rows = [
            (str(match_id), platform, puuid, int(discovered_at))
            for match_id in dict.fromkeys(match_ids)
            if match_id
        ]
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO matches(
                    match_id, platform, source_puuid, discovered_at, status
                ) VALUES(?, ?, ?, ?, 'pending')
                """,
                rows,
            )
            inserted = self.conn.total_changes - before
            if window_complete:
                self.conn.execute(
                    """
                    UPDATE players SET
                        last_scan_end=?, last_scan_attempt=?,
                        scan_window_start=NULL, scan_window_end=NULL,
                        scan_next_start=0
                    WHERE puuid=?
                    """,
                    (int(window_end), int(time.time()), puuid),
                )
            else:
                self.conn.execute(
                    """
                    UPDATE players SET
                        last_scan_attempt=?, scan_window_start=?,
                        scan_window_end=?, scan_next_start=?
                    WHERE puuid=?
                    """,
                    (
                        int(time.time()),
                        int(window_start),
                        int(window_end),
                        int(next_start),
                        puuid,
                    ),
                )
        return int(inserted)

    def enqueue_matches(
        self,
        match_ids: Iterable[str],
        *,
        platform: str,
        source_puuid: str,
        discovered_at: int,
    ) -> int:
        before = self.conn.total_changes
        rows = [
            (str(match_id), platform, source_puuid, int(discovered_at))
            for match_id in dict.fromkeys(match_ids)
            if match_id
        ]
        with self.conn:
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO matches(
                    match_id, platform, source_puuid, discovered_at, status
                ) VALUES(?, ?, ?, ?, 'pending')
                """,
                rows,
            )
        return self.conn.total_changes - before

    def pending_matches(
        self,
        platform: str,
        limit: int,
        max_attempts: int = MAX_MATCH_ATTEMPTS,
    ) -> List[PendingMatch]:
        with self.conn:
            self.conn.execute(
                """
                UPDATE matches SET status='exhausted', completed_at=?
                WHERE platform=? AND status IN ('pending', 'retry') AND attempts >= ?
                """,
                (int(time.time()), platform, int(max_attempts)),
            )
        rows = self.conn.execute(
            """
            SELECT match_id, platform, source_puuid, attempts
            FROM matches
            WHERE platform=? AND status IN ('pending', 'retry') AND attempts < ?
            ORDER BY discovered_at, match_id
            LIMIT ?
            """,
            (platform, int(max_attempts), int(limit)),
        ).fetchall()
        return [
            PendingMatch(
                match_id=str(row["match_id"]),
                platform=str(row["platform"]),
                source_puuid=(
                    str(row["source_puuid"])
                    if row["source_puuid"] is not None
                    else None
                ),
                attempts=int(row["attempts"]),
            )
            for row in rows
        ]

    def mark_attempt(self, match_id: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE matches
                SET attempts=attempts+1, last_attempt_at=?, last_error=NULL,
                    last_http_status=NULL
                WHERE match_id=?
                """,
                (int(time.time()), match_id),
            )

    def mark_retry(
        self,
        match_id: str,
        error: str,
        http_status: Optional[int] = None,
        max_attempts: int = MAX_MATCH_ATTEMPTS,
    ) -> None:
        cleaned = str(error).replace("\n", " ")[:500]
        row = self.conn.execute(
            "SELECT attempts FROM matches WHERE match_id=?", (match_id,)
        ).fetchone()
        attempts = int(row["attempts"]) if row is not None else 0
        terminal = attempts >= int(max_attempts)
        status = "exhausted" if terminal else "retry"
        with self.conn:
            self.conn.execute(
                """
                UPDATE matches
                SET status=?, last_error=?, last_http_status=?,
                    completed_at=CASE WHEN ? THEN ? ELSE completed_at END
                WHERE match_id=?
                """,
                (
                    status,
                    cleaned,
                    http_status,
                    1 if terminal else 0,
                    int(time.time()),
                    match_id,
                ),
            )
            if terminal:
                self.conn.execute(
                    """
                    INSERT INTO collection_events(occurred_at, event_type, payload_json)
                    VALUES(?, 'match_attempts_exhausted', ?)
                    """,
                    (
                        int(time.time()),
                        json.dumps(
                            {"match_id": match_id, "attempts": attempts},
                            sort_keys=True,
                        ),
                    ),
                )

    def mark_excluded(
        self,
        match_id: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = metadata or {}
        with self.conn:
            self.conn.execute(
                """
                UPDATE matches SET
                    status='excluded', completed_at=?, exclusion_reason=?,
                    queue_id=?, map_id=?, game_mode=?, game_version=?,
                    api_patch=?, public_patch=?, game_creation=?
                WHERE match_id=?
                """,
                (
                    int(time.time()),
                    reason[:200],
                    metadata.get("queue_id"),
                    metadata.get("map_id"),
                    metadata.get("game_mode"),
                    metadata.get("game_version"),
                    metadata.get("api_patch"),
                    metadata.get("public_patch"),
                    metadata.get("game_creation"),
                    match_id,
                ),
            )

    def mark_complete(
        self,
        match_id: str,
        *,
        metadata: Dict[str, Any],
        detail_sha256: str,
        timeline_sha256: str,
        detail_bytes: int,
        timeline_bytes: int,
    ) -> None:
        now = int(time.time())
        with self.conn:
            self.conn.execute(
                """
                UPDATE matches SET
                    status='complete', completed_at=?, retrieved_at=?,
                    last_error=NULL, last_http_status=NULL,
                    queue_id=?, map_id=?, game_mode=?, game_version=?,
                    api_patch=?, public_patch=?, game_creation=?,
                    detail_sha256=?, timeline_sha256=?,
                    detail_bytes=?, timeline_bytes=?
                WHERE match_id=?
                """,
                (
                    now,
                    now,
                    metadata.get("queue_id"),
                    metadata.get("map_id"),
                    metadata.get("game_mode"),
                    metadata.get("game_version"),
                    metadata.get("api_patch"),
                    metadata.get("public_patch"),
                    metadata.get("game_creation"),
                    detail_sha256,
                    timeline_sha256,
                    int(detail_bytes),
                    int(timeline_bytes),
                    match_id,
                ),
            )

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM matches GROUP BY status"
        ).fetchall():
            counts[str(row["status"])] = int(row["n"])
        counts["players"] = int(
            self.conn.execute("SELECT COUNT(*) AS n FROM players").fetchone()["n"]
        )
        counts["active_players"] = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM players WHERE active=1"
            ).fetchone()["n"]
        )
        counts["rank_snapshots"] = int(
            self.conn.execute("SELECT COUNT(*) AS n FROM rank_snapshots").fetchone()["n"]
        )
        return counts

    def complete_count(
        self,
        *,
        platform: Optional[str] = None,
        api_patch: Optional[str] = None,
    ) -> int:
        clauses = ["status='complete'"]
        params: List[Any] = []
        if platform is not None:
            clauses.append("lower(platform)=?")
            params.append(str(platform).lower())
        if api_patch is not None:
            clauses.append("api_patch=?")
            params.append(str(api_patch))
        row = self.conn.execute(
            f"SELECT COUNT(*) AS n FROM matches WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        return int(row["n"] or 0)

    def raw_bytes(self) -> int:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(detail_bytes + timeline_bytes), 0) AS n
            FROM matches WHERE status='complete'
            """
        ).fetchone()
        return int(row["n"] or 0)

    def patch_counts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT COALESCE(public_patch, api_patch, 'unknown') AS patch,
                   COUNT(*) AS matches,
                   COALESCE(SUM(detail_bytes + timeline_bytes), 0) AS bytes
            FROM matches
            WHERE status='complete'
            GROUP BY COALESCE(public_patch, api_patch, 'unknown')
            ORDER BY patch
            """
        ).fetchall()
        return [
            {
                "patch": str(row["patch"]),
                "matches": int(row["matches"]),
                "bytes": int(row["bytes"] or 0),
            }
            for row in rows
        ]
