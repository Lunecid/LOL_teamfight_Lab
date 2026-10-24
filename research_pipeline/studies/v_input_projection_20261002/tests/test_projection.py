from __future__ import annotations

import json

import numpy as np

from studies.v_input_projection_20261002 import (
    QUERY_BLOCK_SPECS,
    SemanticHistory,
    field_manifest,
    project_match,
)


def _bank() -> dict[str, np.ndarray]:
    frames = np.zeros((3, 10, 46), dtype=np.float32)
    # The participant rows alternate teams.  Growth must follow this actual
    # map, rather than assuming rows 1-5 are Blue.
    for frame in range(3):
        for participant in range(10):
            frames[frame, participant, :6] = np.asarray(
                [participant + 1, participant + 1, participant + 1,
                 participant + 1, participant + 1, participant + 1],
                dtype=np.float32,
            )
    team_ids = np.asarray([200, 100, 200, 100, 200, 100, 200, 100, 200, 100], dtype=np.int64)
    minute_ts = np.asarray([0, 60_000, 180_000], dtype=np.int64)
    event_types = ["CHAMPION_KILL", "BUILDING_KILL", "ITEM_PURCHASED", "LEVEL_UP", "GAME_END"]
    timestamps = np.asarray([60_000, 60_000, 60_000, 90_000, 150_000], dtype=np.int64)
    values = np.zeros((len(event_types), 15), dtype=np.int64)
    valid = np.zeros_like(values, dtype=bool)

    def put(row: int, name: str, value: int) -> None:
        column = {"participantId": 0, "itemId": 1, "beforeId": 2, "afterId": 3,
                  "killerId": 4, "victimId": 5, "killerTeamId": 6, "teamId": 7,
                  "level": 8, "skillSlot": 9}[name]
        values[row, column] = value
        valid[row, column] = True

    put(0, "killerId", 1)
    put(0, "victimId", 2)
    put(1, "killerTeamId", 100)
    put(1, "teamId", 100)
    put(2, "participantId", 1)
    put(3, "participantId", 4)
    put(3, "level", 0)  # explicit zero is present and valid
    strings = np.zeros((len(event_types), 9), dtype=np.int16)
    string_validity = np.zeros_like(strings, dtype=bool)
    # buildingType code 1 is TOWER_BUILDING in the accepted enum.
    strings[1, 2] = 1
    string_validity[1, 2] = True
    raw_value = json.dumps("bad", separators=(",", ":")).encode("utf-8")
    raw_rows = np.asarray([2], dtype=np.int32)
    raw_cols = np.asarray([1], dtype=np.int16)
    raw_offsets = np.asarray([0, len(raw_value)], dtype=np.int64)
    raw_values = np.frombuffer(raw_value, dtype=np.uint8).copy()
    assists = np.asarray([3], dtype=np.int64)
    assist_offsets = np.asarray([0, 1, 1, 1, 1, 1], dtype=np.int64)
    assist_validity = np.asarray([True, False, False, False, False], dtype=bool)
    assist_presence = np.asarray([True, False, False, False, False], dtype=bool)
    return {
        "minute_ts": minute_ts,
        "node_frame": frames,
        "xy": np.full((3, 10, 2), 123.0, dtype=np.float32),
        "gold_team": np.asarray([[0.0, 0.0], [100.0, 200.0], [300.0, 400.0]], dtype=np.float32),
        "team_ids": team_ids,
        "static_ids": np.zeros((10, 14), dtype=np.int64),
        "bans": np.zeros((2, 5), dtype=np.int64),
        "event_timestamp": timestamps,
        "event_type": np.asarray(event_types, dtype="U24"),
        "event_category_code": np.asarray([0, 6, 0, 0, 0], dtype=np.int16),
        "event_field_values": values,
        "event_field_validity": valid,
        "event_assist_offsets": assist_offsets,
        "event_assist_values": assists,
        "event_assist_validity": assist_validity,
        "event_assist_presence": assist_presence,
        "event_string_codes": strings,
        "event_string_validity": string_validity,
        "event_raw_rows": raw_rows,
        "event_raw_cols": raw_cols,
        "event_raw_offsets": raw_offsets,
        "event_raw_values": raw_values,
        "item_event_indices": np.asarray([2], dtype=np.int32),
        "query_ms": np.asarray([60_000, 100_000], dtype=np.int64),
        "query_frame_index": np.asarray([1, 1], dtype=np.int32),
        "query_event_limit": np.asarray([3, 4], dtype=np.int32),
        "query_item_limit": np.asarray([1, 1], dtype=np.int32),
        "query_team_kills": np.asarray([[0, 1], [0, 1]], dtype=np.int32),
        "query_kda": np.asarray([
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
        ], dtype=np.int32),
        "query_objective_counts": np.zeros((2, 9, 3), dtype=np.int32),
        "query_objective_last_ms": np.full((2, 9), np.nan, dtype=np.float64),
        "query_item_ledger_known": np.asarray([True, False], dtype=bool),
    }


def test_project_match_uses_actual_team_map_and_preserves_zero_static_values():
    bank = _bank()
    bank["query_objective_counts"][:, 5, 0] = 1
    bank["query_objective_counts"][:, 5, 1] = 1
    bank["query_objective_last_ms"][:, 5] = 60_000
    projected = project_match(bank)
    assert projected["node_frame"].shape == (2, 10, 46)
    assert projected["static_ids"].shape == (10, 14)
    assert projected["bans"].shape == (2, 5)
    assert np.all(projected["static_ids"] == 0)
    # Blue rows are participant IDs 2,4,6,8,10; red rows are 1,3,5,7,9.
    np.testing.assert_allclose(projected["team_growth"][0, 0, 0], 6.0)
    np.testing.assert_allclose(projected["team_growth"][0, 0, 1:], [30.0] * 5)
    np.testing.assert_allclose(projected["team_growth"][0, 1, 0], 5.0)
    np.testing.assert_allclose(projected["team_growth_delta"][0], [1.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_equal(projected["objective_counts"][0], np.eye(9, dtype=np.int32)[5])
    np.testing.assert_allclose(projected["objective_age_seconds"][0, 5], 0.0)
    assert "xy" not in projected
    assert set(projected) == {
        "node_frame", "elapsed_minutes", "frame_age_seconds", "team_growth", "team_growth_delta",
        "cached_gold", "team_kills", "player_kda", "objective_counts", "objective_age_seconds",
        "static_ids", "bans", "team_ids",
    }


def test_future_suffix_does_not_change_fixed_query_projection():
    bank = _bank()
    before = project_match(bank)
    after = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in bank.items()}
    after["node_frame"][2, :, :] = 999.0
    after["event_field_values"][4, 4] = 10
    after["event_field_validity"][4, 4] = True
    after["winning_team"] = np.asarray(100, dtype=np.int16)
    after_projected = project_match(after)
    for name in QUERY_BLOCK_SPECS:
        assert np.array_equal(before[name], after_projected[name], equal_nan=True)


def test_nonfinite_growth_is_unknown_only_for_the_affected_channel():
    bank = _bank()
    bank["node_frame"][1, 0, 1] = np.nan  # red xp contributor only
    projected = project_match(bank)
    assert np.isnan(projected["team_growth"][0, 1, 1])
    np.testing.assert_allclose(projected["team_growth"][0, 1, 0], 5.0)
    np.testing.assert_allclose(projected["team_growth"][0, 0, 1], 30.0)
    assert np.isnan(projected["team_growth_delta"][0, 1])
    np.testing.assert_allclose(projected["team_growth_delta"][0, 2:], [5.0] * 4)


def test_semantic_history_is_inclusive_and_keeps_missing_vs_invalid_without_coordinates():
    bank = _bank()
    history = SemanticHistory(bank)
    prefix = history.prefix(60_000)
    assert len(prefix["B.team_kills"]) == 1
    assert len(prefix["B.objective_structure_history"]) == 1
    item = prefix["C.item_transaction_history"][0]
    assert item["payload"]["itemId"] == "bad"
    assert item["field_presence"]["itemId"] is True
    assert item["field_validity"]["itemId"] is False
    assert "xy" not in item and "position" not in item
    assert all(record["event_type"] != "GAME_END" for records in prefix.values() for record in records)
    assert [record["event_type"] for record in history.iter_prefix(60_000)] == [
        "CHAMPION_KILL", "BUILDING_KILL", "ITEM_PURCHASED",
    ]
    level = history.prefix(100_000)["E.confirmed_level"][0]
    assert level["payload"]["level"] == 0
    assert level["field_presence"]["level"] is True
    assert level["field_validity"]["level"] is True
    columnar = history.prefix_columnar(60_000)
    assert columnar["C.item_transaction_history"]["timestamp_ms"].tolist() == [60_000]


def test_manifest_is_json_serializable_and_declares_total_only_objectives():
    manifest = field_manifest()
    json.dumps(manifest)
    assert manifest["objective"]["count_team_axis_source"] == ["total", "blue", "red"]
    assert manifest["objective"]["counts_shape"] == [9]
    assert manifest["items"]["full_inventory_complete"] == "unknown"
    assert manifest["items"]["item_reference_required"] is True
