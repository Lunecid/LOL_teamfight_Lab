from acquisition.state import CollectionState


def rank_entry(puuid="p1", lp=500):
    return {
        "puuid": puuid,
        "leaguePoints": lp,
        "wins": 10,
        "losses": 5,
        "veteran": False,
        "inactive": False,
        "freshBlood": True,
        "hotStreak": False,
    }


def test_rank_snapshot_preserves_provenance_and_player_cursor(tmp_path):
    state = CollectionState(tmp_path / "state.sqlite3")
    try:
        saved, skipped = state.record_rank_entries(
            captured_at=1000,
            platform="kr",
            queue="RANKED_SOLO_5x5",
            tier="MASTER",
            entries=[rank_entry(), {"leaguePoints": 1}],
        )
        assert (saved, skipped) == (1, 1)
        due = state.players_due("kr", 10)
        assert len(due) == 1
        assert due[0].puuid == "p1"
        assert due[0].last_scan_end is None

        state.commit_player_scan_page(
            puuid="p1",
            platform="kr",
            window_start=900,
            window_end=1200,
            next_start=0,
            window_complete=True,
            match_ids=[],
            discovered_at=1200,
        )
        assert state.players_due("kr", 10)[0].last_scan_end == 1200
        counts = state.counts()
        assert counts["players"] == 1
        assert counts["rank_snapshots"] == 1
    finally:
        state.close()


def test_full_page_persists_backfill_cursor_with_ids_in_one_transaction(tmp_path):
    state = CollectionState(tmp_path / "state.sqlite3")
    try:
        state.record_rank_entries(
            captured_at=1000,
            platform="kr",
            queue="RANKED_SOLO_5x5",
            tier="MASTER",
            entries=[rank_entry()],
        )
        ids = [f"KR_{i}" for i in range(100)]
        inserted = state.commit_player_scan_page(
            puuid="p1",
            platform="kr",
            window_start=900,
            window_end=1200,
            next_start=100,
            window_complete=False,
            match_ids=ids,
            discovered_at=1200,
        )
        cursor = state.players_due("kr", 1)[0]
        assert inserted == 100
        assert cursor.last_scan_end is None
        assert cursor.scan_window_start == 900
        assert cursor.scan_window_end == 1200
        assert cursor.scan_next_start == 100
        assert len(state.pending_matches("kr", 200)) == 100

        state.commit_player_scan_page(
            puuid="p1",
            platform="kr",
            window_start=900,
            window_end=1200,
            next_start=100,
            window_complete=True,
            match_ids=[],
            discovered_at=1201,
        )
        completed = state.players_due("kr", 1)[0]
        assert completed.last_scan_end == 1200
        assert completed.scan_window_end is None
        assert completed.scan_next_start == 0
    finally:
        state.close()


def test_absent_players_are_deactivated_after_complete_snapshot(tmp_path):
    state = CollectionState(tmp_path / "state.sqlite3")
    try:
        state.record_rank_entries(
            captured_at=1000,
            platform="kr",
            queue="RANKED_SOLO_5x5",
            tier="MASTER",
            entries=[rank_entry("p1")],
        )
        state.finalize_rank_snapshot("kr", 1000)
        state.record_rank_entries(
            captured_at=2000,
            platform="kr",
            queue="RANKED_SOLO_5x5",
            tier="MASTER",
            entries=[rank_entry("p2")],
        )
        state.finalize_rank_snapshot("kr", 2000)

        assert [cursor.puuid for cursor in state.players_due("kr", 10)] == ["p2"]
        assert state.counts()["players"] == 2
        assert state.counts()["active_players"] == 1
    finally:
        state.close()


def test_retry_budget_moves_match_to_exhausted_attrition_state(tmp_path):
    state = CollectionState(tmp_path / "state.sqlite3")
    try:
        state.enqueue_matches(
            ["KR_1"], platform="kr", source_puuid="p1", discovered_at=1000
        )
        for _ in range(8):
            state.mark_attempt("KR_1")
            state.mark_retry("KR_1", "transport failed")

        assert state.pending_matches("kr", 10) == []
        assert state.counts()["exhausted"] == 1
        event = state.conn.execute(
            "SELECT event_type FROM collection_events ORDER BY id DESC LIMIT 1"
        ).fetchone()["event_type"]
        assert event == "match_attempts_exhausted"
    finally:
        state.close()


def test_match_queue_is_idempotent_and_records_completion(tmp_path):
    state = CollectionState(tmp_path / "state.sqlite3")
    try:
        assert state.enqueue_matches(
            ["KR_1", "KR_1", "KR_2"],
            platform="kr",
            source_puuid="p1",
            discovered_at=1000,
        ) == 2
        assert state.enqueue_matches(
            ["KR_1"], platform="kr", source_puuid="p1", discovered_at=1001
        ) == 0
        pending = state.pending_matches("kr", 10)
        assert [item.match_id for item in pending] == ["KR_1", "KR_2"]

        state.mark_attempt("KR_1")
        state.mark_complete(
            "KR_1",
            metadata={
                "queue_id": 420,
                "map_id": 11,
                "game_mode": "CLASSIC",
                "game_version": "16.13.1",
                "api_patch": "16.13",
                "public_patch": "26.13",
                "game_creation": 123,
            },
            detail_sha256="a" * 64,
            timeline_sha256="b" * 64,
            detail_bytes=100,
            timeline_bytes=200,
        )
        assert state.counts()["complete"] == 1
        assert state.raw_bytes() == 300
        assert state.patch_counts() == [
            {"patch": "26.13", "matches": 1, "bytes": 300}
        ]
        assert [item.match_id for item in state.pending_matches("kr", 10)] == ["KR_2"]
    finally:
        state.close()


def test_event_payload_drops_key_and_token_fields(tmp_path):
    state = CollectionState(tmp_path / "state.sqlite3")
    try:
        state.record_event(
            "safe",
            {"api_key": "secret", "tokenValue": "secret", "status": "ok"},
        )
        payload = state.conn.execute(
            "SELECT payload_json FROM collection_events"
        ).fetchone()["payload_json"]
        assert "secret" not in payload
        assert "status" in payload
    finally:
        state.close()
