from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .client import (
    AuthenticationRejected,
    ResponseValidationError,
    RiotAPIError,
    RiotClient,
)
from .config import (
    CollectorConfig,
    patch_from_game_version,
    public_patch_label,
)
from .key_provider import KeyUnavailable
from .raw_store import RawStore, RawValidationError, validate_detail, validate_timeline
from .state import CollectionState, PendingMatch, PlayerCursor


LOGGER = logging.getLogger(__name__)


@dataclass
class CycleSummary:
    started_at: int
    finished_at: Optional[int] = None
    rank_entries: int = 0
    rank_entries_skipped: int = 0
    players_scanned: int = 0
    match_ids_discovered: int = 0
    matches_completed: int = 0
    matches_excluded: int = 0
    matches_missing: int = 0
    matches_retried: int = 0
    raw_bytes_added: int = 0
    storage_pause_reason: Optional[str] = None
    quota_reached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CollectionAgent:
    def __init__(
        self,
        config: CollectorConfig,
        client: RiotClient,
        state: CollectionState,
        store: RawStore,
        monotonic=time.monotonic,
        disk_usage=shutil.disk_usage,
    ) -> None:
        self.config = config
        self.client = client
        self.state = state
        self.store = store
        self._monotonic = monotonic
        self._disk_usage = disk_usage

    def rank_refresh_due(self, now: int) -> bool:
        previous = self.state.metadata_int("last_rank_refresh")
        return previous is None or now - previous >= self.config.rank_refresh_seconds

    def _target_patch_label(self) -> Optional[str]:
        patch = self.config.exact_api_patch
        return f"{patch[0]}.{patch[1]}" if patch is not None else None

    def complete_count(self) -> int:
        return self.state.complete_count(
            platform=self.config.platform,
            api_patch=self._target_patch_label(),
        )

    def quota_reached(self) -> bool:
        target = self.config.max_complete_matches
        return target is not None and self.complete_count() >= target

    async def refresh_rank_snapshot(self, now: int, summary: CycleSummary) -> None:
        captured_at = int(now)
        tier_payloads: List[Tuple[str, List[Dict[str, Any]]]] = []
        for tier in self.config.tiers:
            entries = await self.client.high_tier_entries(tier)
            missing_puuid = sum(1 for entry in entries if not entry.get("puuid"))
            if entries and (
                missing_puuid / len(entries) > self.config.max_missing_puuid_fraction
            ):
                raise ResponseValidationError(
                    f"{tier} roster omitted puuid for {missing_puuid}/{len(entries)} entries"
                )
            tier_payloads.append((tier, entries))

        for tier, entries in tier_payloads:
            saved, skipped = self.state.record_rank_entries(
                captured_at=captured_at,
                platform=self.config.platform,
                queue=self.config.queue,
                tier=tier,
                entries=entries,
            )
            summary.rank_entries += saved
            summary.rank_entries_skipped += skipped
        self.state.finalize_rank_snapshot(self.config.platform, captured_at)
        self.state.set_metadata("last_rank_refresh", str(captured_at))
        self.state.record_event(
            "rank_snapshot",
            {
                "captured_at": captured_at,
                "platform": self.config.platform,
                "entries": summary.rank_entries,
                "skipped_without_puuid": summary.rank_entries_skipped,
            },
        )

    async def _discover_for_player(
        self,
        cursor: PlayerCursor,
        now: int,
    ) -> Tuple[int, bool]:
        if cursor.scan_window_start is not None and cursor.scan_window_end is not None:
            start_time = cursor.scan_window_start
            end_time = cursor.scan_window_end
            next_start = cursor.scan_next_start
        elif cursor.last_scan_end is None:
            start_time = now - self.config.initial_lookback_seconds
            end_time = now
            next_start = 0
        else:
            start_time = max(0, cursor.last_scan_end - self.config.scan_overlap_seconds)
            end_time = now
            next_start = 0
        self.state.mark_player_scan_attempt(cursor.puuid, now)
        inserted = 0
        window_complete = False
        for _ in range(self.config.match_pages_per_player):
            batch = await self.client.match_ids_by_puuid(
                cursor.puuid,
                start_time=start_time,
                end_time=end_time,
                start=next_start,
                count=100,
            )
            window_complete = len(batch) < 100
            next_start += len(batch)
            inserted += self.state.commit_player_scan_page(
                puuid=cursor.puuid,
                platform=self.config.platform,
                window_start=start_time,
                window_end=end_time,
                next_start=next_start,
                window_complete=window_complete,
                match_ids=batch,
                discovered_at=now,
            )
            if window_complete:
                break
        return inserted, window_complete

    async def discover_matches(
        self,
        now: int,
        summary: CycleSummary,
        deadline: Optional[float] = None,
    ) -> None:
        cursors = self.state.players_due(
            self.config.platform,
            self.config.players_per_cycle,
        )
        for cursor in cursors:
            if deadline is not None and self._monotonic() >= deadline:
                break
            inserted, _ = await self._discover_for_player(cursor, now)
            summary.players_scanned += 1
            summary.match_ids_discovered += inserted

    def _metadata_from_detail(self, match_id: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        metadata = detail.get("metadata")
        info = detail.get("info")
        if not isinstance(metadata, dict) or metadata.get("matchId") != match_id:
            raise RawValidationError("detail metadata.matchId mismatch")
        if not isinstance(info, dict):
            raise RawValidationError("detail info is missing")
        game_version = str(info.get("gameVersion") or "")
        api_patch_tuple = patch_from_game_version(game_version)
        return {
            "queue_id": info.get("queueId"),
            "map_id": info.get("mapId"),
            "game_mode": info.get("gameMode"),
            "game_version": game_version,
            "api_patch": (
                f"{api_patch_tuple[0]}.{api_patch_tuple[1]}"
                if api_patch_tuple is not None
                else None
            ),
            "public_patch": public_patch_label(api_patch_tuple),
            "game_creation": info.get("gameCreation"),
            "api_patch_tuple": api_patch_tuple,
        }

    def _exclusion_reason(self, metadata: Dict[str, Any]) -> Optional[str]:
        if metadata.get("queue_id") != self.config.queue_id:
            return "queue_id"
        if metadata.get("map_id") != self.config.map_id:
            return "map_id"
        if metadata.get("game_mode") != self.config.game_mode:
            return "game_mode"
        patch = metadata.get("api_patch_tuple")
        if patch is None:
            return "malformed_game_version"
        if self.config.min_api_patch is not None and patch < self.config.min_api_patch:
            return "before_min_api_patch"
        if self.config.exact_api_patch is not None and patch != self.config.exact_api_patch:
            return "outside_exact_api_patch"
        return None

    def _recover_existing_pair(
        self, pending: PendingMatch
    ) -> Optional[Tuple[str, int]]:
        stored = self.store.read_valid_pair(
            pending.match_id,
            queue_id=self.config.queue_id,
            map_id=self.config.map_id,
            game_mode=self.config.game_mode,
        )
        if stored is None:
            return None
        detail = json.loads(stored.detail_path.read_text(encoding="utf-8"))
        metadata = self._metadata_from_detail(pending.match_id, detail)
        exclusion = self._exclusion_reason(metadata)
        serializable_metadata = dict(metadata)
        serializable_metadata.pop("api_patch_tuple", None)
        if exclusion is not None:
            self.store.quarantine(pending.match_id, f"excluded_{exclusion}")
            self.state.mark_excluded(
                pending.match_id,
                exclusion,
                serializable_metadata,
            )
            return "excluded", 0
        self.state.mark_complete(
            pending.match_id,
            metadata=serializable_metadata,
            detail_sha256=stored.detail_sha256,
            timeline_sha256=stored.timeline_sha256,
            detail_bytes=stored.detail_bytes,
            timeline_bytes=stored.timeline_bytes,
        )
        return "complete", stored.detail_bytes + stored.timeline_bytes

    async def _download_one(self, pending: PendingMatch, summary: CycleSummary) -> None:
        recovered = self._recover_existing_pair(pending)
        if recovered is not None and recovered[0] == "complete":
            summary.matches_completed += 1
            summary.raw_bytes_added += recovered[1]
            return
        if recovered is not None and recovered[0] == "excluded":
            summary.matches_excluded += 1
            return

        self.state.mark_attempt(pending.match_id)
        detail = await self.client.match_detail(pending.match_id)
        if detail is None:
            self.state.mark_excluded(pending.match_id, "detail_not_found")
            summary.matches_missing += 1
            return

        metadata = self._metadata_from_detail(pending.match_id, detail)
        exclusion = self._exclusion_reason(metadata)
        serializable_metadata = dict(metadata)
        serializable_metadata.pop("api_patch_tuple", None)
        if exclusion is not None:
            self.state.mark_excluded(
                pending.match_id,
                exclusion,
                serializable_metadata,
            )
            summary.matches_excluded += 1
            return

        validate_detail(
            pending.match_id,
            detail,
            queue_id=self.config.queue_id,
            map_id=self.config.map_id,
            game_mode=self.config.game_mode,
        )
        timeline = await self.client.match_timeline(pending.match_id)
        if timeline is None:
            self.state.mark_retry(pending.match_id, "timeline_not_found", 404)
            summary.matches_missing += 1
            return
        validate_timeline(pending.match_id, timeline)

        stored = self.store.write_pair(
            pending.match_id,
            detail,
            timeline,
            queue_id=self.config.queue_id,
            map_id=self.config.map_id,
            game_mode=self.config.game_mode,
        )
        self.state.mark_complete(
            pending.match_id,
            metadata=serializable_metadata,
            detail_sha256=stored.detail_sha256,
            timeline_sha256=stored.timeline_sha256,
            detail_bytes=stored.detail_bytes,
            timeline_bytes=stored.timeline_bytes,
        )
        summary.matches_completed += 1
        summary.raw_bytes_added += stored.detail_bytes + stored.timeline_bytes

    def _storage_pause_reason(self, accounted_bytes: int) -> Optional[str]:
        limit = self.config.max_storage_bytes
        if limit is not None and accounted_bytes >= limit:
            return f"max_storage_bytes reached ({accounted_bytes} >= {limit})"
        free_bytes = int(self._disk_usage(self.store.root).free)
        if free_bytes <= self.config.min_free_bytes:
            return (
                "minimum free-space reserve reached "
                f"({free_bytes} <= {self.config.min_free_bytes})"
            )
        return None

    async def download_pending(
        self,
        summary: CycleSummary,
        deadline: Optional[float] = None,
    ) -> None:
        pending = self.state.pending_matches(
            self.config.platform,
            self.config.matches_per_cycle,
        )
        initial_raw_bytes = self.state.raw_bytes()
        for item in pending:
            if self.quota_reached():
                summary.quota_reached = True
                break
            if deadline is not None and self._monotonic() >= deadline:
                break
            storage_reason = self._storage_pause_reason(
                initial_raw_bytes + summary.raw_bytes_added
            )
            if storage_reason is not None:
                summary.storage_pause_reason = storage_reason
                self.state.record_event(
                    "storage_paused",
                    {
                        "reason": storage_reason,
                        "raw_bytes": initial_raw_bytes + summary.raw_bytes_added,
                    },
                )
                break
            try:
                await self._download_one(item, summary)
            except (AuthenticationRejected, KeyUnavailable):
                raise
            except RawValidationError as exc:
                self.state.mark_excluded(item.match_id, f"invalid_payload:{exc}")
                summary.matches_excluded += 1
            except RiotAPIError as exc:
                self.state.mark_retry(item.match_id, str(exc))
                summary.matches_retried += 1
                # Persistent API/network failures are global. Let the supervisor
                # back off instead of hammering the next thousands of matches.
                raise

    async def run_cycle(self) -> CycleSummary:
        now = int(time.time())
        deadline = self._monotonic() + self.config.cycle_work_budget_seconds
        summary = CycleSummary(started_at=now)
        if self.quota_reached():
            summary.quota_reached = True
            summary.finished_at = int(time.time())
            self.state.record_event("cycle_complete", summary.to_dict())
            return summary
        if self.rank_refresh_due(now):
            await self.refresh_rank_snapshot(now, summary)
        await self.discover_matches(now, summary, deadline)
        await self.download_pending(summary, deadline)
        summary.quota_reached = self.quota_reached()
        summary.finished_at = int(time.time())
        self.state.record_event("cycle_complete", summary.to_dict())
        return summary
