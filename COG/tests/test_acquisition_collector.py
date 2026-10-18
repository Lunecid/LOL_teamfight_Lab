import asyncio

import pytest

from acquisition.collector import CollectionAgent
from acquisition.client import ResponseValidationError
from acquisition.config import CollectorConfig
from acquisition.raw_store import RawStore
from acquisition.state import CollectionState


def make_detail(match_id, patch="16.13.1"):
    return {
        "metadata": {"matchId": match_id, "participants": [str(i) for i in range(10)]},
        "info": {
            "gameVersion": patch,
            "queueId": 420,
            "mapId": 11,
            "gameMode": "CLASSIC",
            "gameCreation": 123456,
            "participants": [{"participantId": i + 1} for i in range(10)],
            "teams": [{"teamId": 100}, {"teamId": 200}],
        },
    }


def make_timeline(match_id):
    return {
        "metadata": {"matchId": match_id, "participants": [str(i) for i in range(10)]},
        "info": {
            "participants": [{"participantId": i + 1} for i in range(10)],
            "frames": [{"timestamp": 0, "events": []}],
        },
    }


class FakeClient:
    def __init__(self, patch="16.13.1"):
        self.patch = patch
        self.match_id_calls = []
        self.detail_calls = 0

    async def high_tier_entries(self, tier):
        return [{"puuid": "player-1", "leaguePoints": 500, "wins": 10, "losses": 5}]

    async def match_ids_by_puuid(self, puuid, **kwargs):
        self.match_id_calls.append((puuid, kwargs))
        return ["KR_123"]

    async def match_detail(self, match_id):
        self.detail_calls += 1
        return make_detail(match_id, self.patch)

    async def match_timeline(self, match_id):
        return make_timeline(match_id)


def make_config(tmp_path):
    return CollectorConfig(
        key_file=tmp_path / "riot.env",
        output_root=tmp_path / "raw",
        state_dir=tmp_path / "state",
        players_per_cycle=10,
        matches_per_cycle=10,
    )


def test_cycle_collects_rank_provenance_and_atomic_pair(tmp_path):
    async def exercise():
        config = make_config(tmp_path)
        state = CollectionState(config.database_path)
        store = RawStore(config.output_root, config.platform)
        fake = FakeClient()
        agent = CollectionAgent(config, fake, state, store)
        try:
            summary = await agent.run_cycle()
            assert summary.rank_entries == 3
            assert summary.players_scanned == 1
            assert summary.match_ids_discovered == 1
            assert summary.matches_completed == 1
            assert state.counts()["complete"] == 1
            detail_path, timeline_path = store.paths("KR_123")
            assert detail_path.exists() and timeline_path.exists()
            assert fake.match_id_calls[0][1]["start_time"] < fake.match_id_calls[0][1]["end_time"]
        finally:
            state.close()

    asyncio.run(exercise())


def test_cycle_excludes_pre_2026_patch_without_writing_raw(tmp_path):
    async def exercise():
        config = make_config(tmp_path)
        state = CollectionState(config.database_path)
        store = RawStore(config.output_root, config.platform)
        agent = CollectionAgent(config, FakeClient("15.24.1"), state, store)
        try:
            summary = await agent.run_cycle()
            assert summary.matches_excluded == 1
            assert state.counts()["excluded"] == 1
            detail_path, timeline_path = store.paths("KR_123")
            assert not detail_path.exists()
            assert not timeline_path.exists()
        finally:
            state.close()

    asyncio.run(exercise())


def test_exact_patch_excludes_later_patch(tmp_path):
    async def exercise():
        base = make_config(tmp_path)
        config = CollectorConfig(
            **{
                **base.__dict__,
                "exact_api_patch": (16, 13),
            }
        )
        state = CollectionState(config.database_path)
        store = RawStore(config.output_root, config.platform)
        agent = CollectionAgent(config, FakeClient("16.14.1"), state, store)
        try:
            summary = await agent.run_cycle()
            assert summary.matches_excluded == 1
            assert state.counts()["excluded"] == 1
        finally:
            state.close()

    asyncio.run(exercise())


def test_complete_quota_stops_without_overshoot(tmp_path):
    async def exercise():
        base = make_config(tmp_path)
        config = CollectorConfig(
            **{
                **base.__dict__,
                "exact_api_patch": (16, 13),
                "max_complete_matches": 1,
            }
        )
        state = CollectionState(config.database_path)
        store = RawStore(config.output_root, config.platform)
        state.enqueue_matches(
            ["KR_123", "KR_124"],
            platform="kr",
            source_puuid="test-player",
            discovered_at=1000,
        )
        agent = CollectionAgent(config, FakeClient("16.13.1"), state, store)
        from acquisition.collector import CycleSummary

        summary = CycleSummary(started_at=1000)
        try:
            await agent.download_pending(summary)
            assert summary.matches_completed == 1
            assert summary.quota_reached is True
            assert state.complete_count(platform="kr", api_patch="16.13") == 1
            assert len(state.pending_matches("kr", 10)) == 1
        finally:
            state.close()

    asyncio.run(exercise())


def test_existing_old_patch_pair_is_quarantined_not_recovered(tmp_path):
    async def exercise():
        config = make_config(tmp_path)
        state = CollectionState(config.database_path)
        store = RawStore(config.output_root, config.platform)
        store.write_pair(
            "KR_123",
            make_detail("KR_123", "15.24.1"),
            make_timeline("KR_123"),
            queue_id=420,
            map_id=11,
            game_mode="CLASSIC",
        )
        state.enqueue_matches(
            ["KR_123"],
            platform="kr",
            source_puuid="player-1",
            discovered_at=1000,
        )
        agent = CollectionAgent(config, FakeClient(), state, store)
        try:
            summary = await agent.run_cycle()
            assert summary.matches_excluded == 1
            assert state.counts()["excluded"] == 1
            detail_path, timeline_path = store.paths("KR_123")
            assert not detail_path.exists() and not timeline_path.exists()
            assert len(list(store.quarantine_dir.rglob("*.json"))) == 2
        finally:
            state.close()

    asyncio.run(exercise())


def test_full_pages_resume_from_persisted_offset(tmp_path):
    class FullPageClient(FakeClient):
        async def match_ids_by_puuid(self, puuid, **kwargs):
            self.match_id_calls.append((puuid, kwargs))
            start = kwargs["start"]
            return [f"KR_{start + i + 1}" for i in range(100)]

    async def exercise():
        config = CollectorConfig(
            key_file=tmp_path / "riot.env",
            output_root=tmp_path / "raw",
            state_dir=tmp_path / "state",
            players_per_cycle=1,
            matches_per_cycle=1,
            match_pages_per_player=2,
        )
        state = CollectionState(config.database_path)
        store = RawStore(config.output_root, config.platform)
        fake = FullPageClient()
        agent = CollectionAgent(config, fake, state, store)
        try:
            await agent.refresh_rank_snapshot(1000, type("S", (), {
                "rank_entries": 0,
                "rank_entries_skipped": 0,
            })())
            from acquisition.collector import CycleSummary

            await agent.discover_matches(2000, CycleSummary(started_at=2000))
            cursor = state.players_due("kr", 1)[0]
            assert cursor.scan_next_start == 200
            await agent.discover_matches(3000, CycleSummary(started_at=3000))
            assert [call[1]["start"] for call in fake.match_id_calls] == [0, 100, 200, 300]
            assert state.players_due("kr", 1)[0].scan_next_start == 400
        finally:
            state.close()

    asyncio.run(exercise())


def test_cycle_deadline_does_not_starve_rank_refresh(tmp_path):
    async def exercise():
        config = make_config(tmp_path)
        state = CollectionState(config.database_path)
        store = RawStore(config.output_root, config.platform)
        clock = iter([0.0, float(config.cycle_work_budget_seconds) + 1.0])
        agent = CollectionAgent(
            config,
            FakeClient(),
            state,
            store,
            monotonic=lambda: next(clock),
        )
        try:
            summary = await agent.run_cycle()
            assert summary.rank_entries == 3
            assert summary.players_scanned == 0
            assert state.counts()["rank_snapshots"] == 3
        finally:
            state.close()

    asyncio.run(exercise())


def test_rank_schema_change_with_missing_puuid_fails_closed(tmp_path):
    class MissingPuuidClient(FakeClient):
        async def high_tier_entries(self, tier):
            return [{"leaguePoints": 500, "wins": 10, "losses": 5}]

    async def exercise():
        config = make_config(tmp_path)
        state = CollectionState(config.database_path)
        store = RawStore(config.output_root, config.platform)
        agent = CollectionAgent(config, MissingPuuidClient(), state, store)
        try:
            with pytest.raises(ResponseValidationError):
                await agent.run_cycle()
            assert state.get_metadata("last_rank_refresh") is None
            assert state.counts()["players"] == 0
        finally:
            state.close()

    asyncio.run(exercise())


def test_storage_quota_pauses_download_without_losing_pending_match(tmp_path):
    async def exercise():
        config = CollectorConfig(
            key_file=tmp_path / "riot.env",
            output_root=tmp_path / "raw",
            state_dir=tmp_path / "state",
            max_storage_bytes=0,
            min_free_bytes=0,
        )
        state = CollectionState(config.database_path)
        store = RawStore(config.output_root, config.platform)
        fake = FakeClient()
        agent = CollectionAgent(config, fake, state, store)
        state.enqueue_matches(
            ["KR_123"],
            platform="kr",
            source_puuid="player-1",
            discovered_at=1000,
        )
        from acquisition.collector import CycleSummary

        summary = CycleSummary(started_at=1000)
        try:
            await agent.download_pending(summary)
            assert summary.storage_pause_reason is not None
            assert fake.detail_calls == 0
            assert [item.match_id for item in state.pending_matches("kr", 10)] == [
                "KR_123"
            ]
        finally:
            state.close()

    asyncio.run(exercise())
