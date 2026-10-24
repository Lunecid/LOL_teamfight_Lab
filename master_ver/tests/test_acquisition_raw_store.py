import json

import pytest

from acquisition.raw_store import RawStore, RawValidationError


def detail(match_id="KR_123", patch="16.13.1"):
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


def timeline(match_id="KR_123"):
    return {
        "metadata": {"matchId": match_id, "participants": [str(i) for i in range(10)]},
        "info": {
            "participants": [{"participantId": i + 1} for i in range(10)],
            "frames": [{"timestamp": 0, "events": []}],
            "frameInterval": 60000,
        },
    }


def test_write_pair_is_valid_and_hashes_are_stable(tmp_path):
    store = RawStore(tmp_path / "raw", "kr")
    stored = store.write_pair(
        "KR_123", detail(), timeline(), queue_id=420, map_id=11, game_mode="CLASSIC"
    )

    assert stored.detail_path.exists()
    assert stored.timeline_path.exists()
    assert len(stored.detail_sha256) == 64
    assert len(stored.timeline_sha256) == 64
    assert list((tmp_path / "raw").rglob("*.part")) == []

    recovered = store.read_valid_pair(
        "KR_123", queue_id=420, map_id=11, game_mode="CLASSIC"
    )
    assert recovered is not None
    assert recovered.detail_sha256 == stored.detail_sha256


def test_write_pair_rejects_mismatched_timeline_without_final_files(tmp_path):
    store = RawStore(tmp_path / "raw", "kr")
    with pytest.raises(RawValidationError):
        store.write_pair(
            "KR_123",
            detail(),
            timeline("KR_999"),
            queue_id=420,
            map_id=11,
            game_mode="CLASSIC",
        )
    detail_path, timeline_path = store.paths("KR_123")
    assert not detail_path.exists()
    assert not timeline_path.exists()


def test_corrupt_existing_pair_is_quarantined(tmp_path):
    store = RawStore(tmp_path / "raw", "kr")
    detail_path, timeline_path = store.paths("KR_123")
    detail_path.write_text("not json", encoding="utf-8")
    timeline_path.write_text(json.dumps(timeline()), encoding="utf-8")

    recovered = store.read_valid_pair(
        "KR_123", queue_id=420, map_id=11, game_mode="CLASSIC"
    )

    assert recovered is None
    assert not detail_path.exists()
    assert not timeline_path.exists()
    assert len(list(store.quarantine_dir.rglob("*.json"))) == 2


def test_path_traversal_match_id_is_rejected(tmp_path):
    store = RawStore(tmp_path / "raw", "kr")
    with pytest.raises(RawValidationError):
        store.paths("../secret")


def test_structurally_empty_participant_objects_are_rejected(tmp_path):
    store = RawStore(tmp_path / "raw", "kr")
    broken = detail()
    broken["info"]["participants"] = [None] * 10
    with pytest.raises(RawValidationError):
        store.write_pair(
            "KR_123",
            broken,
            timeline(),
            queue_id=420,
            map_id=11,
            game_mode="CLASSIC",
        )


def test_non_object_timeline_frame_is_rejected(tmp_path):
    store = RawStore(tmp_path / "raw", "kr")
    broken = timeline()
    broken["info"]["frames"] = [None]
    with pytest.raises(RawValidationError):
        store.write_pair(
            "KR_123",
            detail(),
            broken,
            queue_id=420,
            map_id=11,
            game_mode="CLASSIC",
        )
