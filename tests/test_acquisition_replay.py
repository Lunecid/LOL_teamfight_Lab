from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from acquisition.replay import (
    ReplayCandidate,
    ReplayCollector,
    ReplayCollectorConfig,
    ReplayManifest,
    archive_replay,
    load_replay_candidates,
    sha256_file,
)


def _source_database(path: Path) -> Path:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE matches (
            match_id TEXT PRIMARY KEY,
            platform TEXT,
            public_patch TEXT,
            api_patch TEXT,
            game_creation INTEGER,
            detail_sha256 TEXT,
            timeline_sha256 TEXT,
            status TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO matches VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("KR_20", "kr", "26.13", "16.13", 200, "d20", "t20", "complete"),
            ("KR_10", "kr", "26.13", "16.13", 100, "d10", "t10", "complete"),
            ("KR_30", "kr", "26.13", "16.13", 300, "d30", "t30", "pending"),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _config(tmp_path: Path, source_db: Path, **overrides) -> ReplayCollectorConfig:
    values = dict(
        source_database=source_db,
        output_root=tmp_path / "replays",
        league_lockfile=tmp_path / "lockfile",
        max_replays=10,
        max_candidates=10,
        max_storage_bytes=10 * 1024**2,
        min_free_bytes=0,
        request_delay_seconds=0,
        min_replay_bytes=16,
    )
    values.update(overrides)
    return ReplayCollectorConfig(**values)


def test_load_candidates_is_oldest_first_and_complete_only(tmp_path: Path) -> None:
    database = _source_database(tmp_path / "collector.sqlite3")
    candidates = load_replay_candidates(database, platform="kr", oldest_first=True)
    assert [item.match_id for item in candidates] == ["KR_10", "KR_20"]
    assert candidates[0].detail_sha256 == "d10"


def test_archive_replay_is_atomic_and_hashes_bytes(tmp_path: Path) -> None:
    source = tmp_path / "KR-10.rofl"
    destination = tmp_path / "archive" / source.name
    source.write_bytes(b"replay-payload-that-is-large-enough")
    size, digest = archive_replay(
        source, destination, delete_source=True, min_replay_bytes=16
    )
    assert size == destination.stat().st_size
    assert digest == sha256_file(destination)
    assert not source.exists()
    assert not destination.with_suffix(".rofl.part").exists()


class _FakeClient:
    def __init__(self, download_dir: Path) -> None:
        self.download_dir = download_dir
        self.calls = []

    def download(self, candidate: ReplayCandidate, config: ReplayCollectorConfig) -> Path:
        self.calls.append(candidate.match_id)
        path = self.download_dir / f"KR-{candidate.game_id}.rofl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((candidate.match_id * 8).encode("ascii"))
        return path


def test_collector_resumes_and_exports_linked_manifest(tmp_path: Path) -> None:
    database = _source_database(tmp_path / "collector.sqlite3")
    config = _config(tmp_path, database, max_replays=2)
    client = _FakeClient(tmp_path / "client")
    manifest = ReplayManifest(config.state_database)
    collector = ReplayCollector(config, client=client, manifest=manifest)
    first = collector.collect()
    second = collector.collect()
    assert first.downloaded == 2
    assert second.downloaded == 0
    assert second.already_complete == 2
    assert client.calls == ["KR_10", "KR_20"]
    lines = [json.loads(line) for line in config.manifest_jsonl.read_text().splitlines()]
    complete = [row for row in lines if row["status"] == "complete"]
    assert len(complete) == 2
    assert complete[0]["detail_sha256"] == "d10"
    assert complete[0]["timeline_sha256"] == "t10"
    assert len(complete[0]["replay_sha256"]) == 64
    manifest.close()


def test_config_rejects_nonpositive_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ReplayCollectorConfig(
            source_database=tmp_path / "db",
            output_root=tmp_path / "out",
            league_lockfile=tmp_path / "lock",
            max_replays=0,
        )

