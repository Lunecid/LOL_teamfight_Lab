"""Integration evidence for lossless typed banks and complete shard replay."""
import numpy as np

from studies.observable_state_20261002 import producer
from studies.observable_state_20261002.bank import ShardBuilder, match_view, pack_bank, unpack_bank
from studies.observable_state_20261002.kernel import ObservableMatch
from studies.observable_state_20261002.storage import close_shard, open_shard, write_shard
from studies.observable_state_20261002.tests.test_kernel import _meta


def test_producer_complete_disk_roundtrip_and_empty_query_shapes(tmp_path, monkeypatch):
    def sources(entry):
        end = entry["audited_numbers"]["game_end_ms"]
        ts = np.array([0, 120000, end], dtype=np.int64) if end > 120000 else np.array([0, end], dtype=np.int64)
        arrays = {"minute_ts": ts, "node_minute": np.zeros((len(ts), 10, 76), dtype=np.float32),
                  "xy_raw_minute": np.ones((len(ts), 10, 2), dtype=np.float32),
                  "gold_team_minute": np.zeros((len(ts), 2), dtype=np.float32)}
        events = [] if end < 120000 else [{"type": "CHAMPION_KILL", "timestamp": 130000,
                                          "killerId": 1, "victimId": 6, "position": {"x": 100, "y": 200}}]
        events.append({"type": "GAME_END", "timestamp": end, "winningTeam": 100})
        return arrays, _meta(), events, {k: {"stable": True} for k in ("npz", "meta", "events")}

    monkeypatch.setattr(producer, "read_cached_triplet", sources)
    builder = ShardBuilder()
    for end, frames, events, kills in [(300000, 3, 2, 1), (80000, 2, 1, 0)]:
        entry = {"match_id": "fixture", "canonical_match_key": str(end), "patch": "15.14", "role": "TRAIN",
                 "original_index": 0, "source_hashes": {}, "audited_numbers": {
                     "n_frames": frames, "n_events": events, "n_kills": kills,
                     "game_end_ms": end, "game_end_winning_team": 100}}
        bank, ledger = producer.produce_match(entry)
        assert ledger["counts"]["n_match_roundtrips"] == 1
        builder.append(bank)
    arrays, layout = builder.finish()
    directory = tmp_path / "state_shard"
    write_shard(directory, arrays, compressed=True)
    loaded = open_shard(directory)
    try:
        first = match_view(loaded, layout, 0)
        model_bank = {k: v for k, v in first.items() if not k.startswith(("query_", "component_", "v_nominal"))}
        match = ObservableMatch.from_bank(unpack_bank(model_bank), "15.14")
        assert match.query_numeric(130000)["team_kills"].tolist() == [1, 0]
        assert first["component_reason_codes"].max() >= 131072
        empty = match_view(loaded, layout, 1)
        assert empty["query_ms"].shape == (0,)
        assert empty["query_objective_counts"].shape == (0, 9, 3)
        assert empty["component_values"].shape == (0, len(producer.COMPONENT_COLUMNS))
    finally:
        close_shard(loaded)
