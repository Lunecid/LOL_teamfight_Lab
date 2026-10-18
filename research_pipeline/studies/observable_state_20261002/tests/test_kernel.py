"""Focused chronology and uncertainty checks for the observable-state kernel."""
from __future__ import annotations

import numpy as np
import pytest

from studies.observable_state_20261002.kernel import ObservableMatch, ordered_feature_description


def _meta(*, zeros: bool = False, bad_team: bool = False) -> dict:
    team_map = {str(pid): (100 if pid <= 5 else 200) for pid in range(1, 11)}
    if bad_team:
        team_map["5"] = 200
    value = 0 if zeros else 1
    return {
        "team_map": team_map,
        "static_meta": {
            "champion_by_pid": {str(pid): value * pid for pid in range(1, 11)},
            "summoner_spells_by_pid": {
                str(pid): {"summoner_spell_1_id": value, "summoner_spell_2_id": value}
                for pid in range(1, 11)
            },
            "runes_by_pid": {
                str(pid): {
                    "primary_style_id": value,
                    "sub_style_id": value,
                    "primary_rune_1": value,
                    "primary_rune_2": value,
                    "primary_rune_3": value,
                    "primary_rune_4": value,
                    "sub_rune_1": value,
                    "sub_rune_2": value,
                    "stat_perk_offense": value,
                    "stat_perk_flex": value,
                    "stat_perk_defense": value,
                }
                for pid in range(1, 11)
            },
            "bans": {"blue": [value] * 5, "red": [value] * 5},
        },
    }


def _match(events=None, *, zeros: bool = False) -> ObservableMatch:
    ts = np.asarray([0, 60_000, 120_000], dtype=np.int64)
    node = np.zeros((3, 10, 46), dtype=np.float32)
    node[0, :, 0] = 1.0
    node[1, :, 0] = 2.0
    node[2, :, 0] = 3.0
    node[1, :, 1] = np.arange(10, dtype=np.float32)
    xy = np.zeros((3, 10, 2), dtype=np.float32)
    xy[1, :, 0] = 11.0
    gold = np.asarray([[0, 0], [100, 200], [300, 400]], dtype=np.float32)
    arrays = {"minute_ts": ts, "node_frame": node, "xy": xy, "gold_team": gold}
    if events is None:
        events = [{"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100}]
    return ObservableMatch(arrays, _meta(zeros=zeros), events, "15.14")


def test_chronology_includes_same_timestamp_records_and_source_indices():
    events = [
        {"type": "LEVEL_UP", "timestamp": 60_000, "participantId": 1, "level": 2},
        {"type": "CHAMPION_KILL", "timestamp": 60_000, "killerId": 1, "victimId": 6, "assistingParticipantIds": []},
        {"type": "SKILL_LEVEL_UP", "timestamp": 60_000, "participantId": 1, "skillSlot": 2},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    state = _match(events).query(60_000)
    assert state["audit"]["event_cutoff_count"] == 3
    assert state["team_kills"].tolist() == [1, 0]
    assert [row["source_index"] for row in state["histories"]["level_ups"]] == [0]
    assert [row["source_index"] for row in state["histories"]["skill_level_ups"]] == [2]


def test_held_frame_and_future_frame_mutation_are_invariant():
    match = _match()
    before = match.query(60_000)
    assert before["frame_index"] == 1
    assert np.all(before["node_frame"][:, 0] == 2.0)
    assert np.all(before["xy"][:, 0] == 11.0)
    # The future frame is immutable and cannot affect a prior query.
    with pytest.raises(ValueError):
        match.node_frame[2, :, 0] = 99.0
    after = match.query(60_000)
    np.testing.assert_array_equal(before["node_frame"], after["node_frame"])
    np.testing.assert_array_equal(before["gold_team"], after["gold_team"])


def test_same_ms_item_order_is_unknown_but_history_is_retained():
    events = [
        {"type": "ITEM_PURCHASED", "timestamp": 60_000, "participantId": 1, "itemId": 1001},
        {"type": "ITEM_SOLD", "timestamp": 60_000, "participantId": 1, "itemId": 1001},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    state = _match(events).query(60_000)
    ledger = state["item_ledger"]
    assert ledger["known"] is False
    assert ledger["reason"] == "same_timestamp_item_order_ambiguous"
    assert np.isnan(ledger["payload"]).all()
    assert [row["source_index"] for row in ledger["transactions"]] == [0, 1]


def test_item_undo_can_restore_after_without_item_id_and_replay_stays_known():
    events = [
        {"type": "ITEM_UNDO", "timestamp": 60_000, "participantId": 1, "beforeId": 0, "afterId": 1001},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    ledger = _match(events).query(60_000)["item_ledger"]
    assert ledger["known"] is True
    assert ledger["item_ids"].tolist() == [1001]
    assert ledger["counts"].tolist() == [1]


def test_replay_underflow_is_unknown_without_erasing_transaction_history():
    events = [
        {"type": "ITEM_SOLD", "timestamp": 60_000, "participantId": 1, "itemId": 1001},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    ledger = _match(events).query(60_000)["item_ledger"]
    assert ledger["known"] is False
    assert ledger["reason"] == "item_replay_underflow"
    assert len(ledger["transactions"]) == 1


def test_environmental_kill_adds_victim_death_without_player_kill_credit():
    events = [
        {"type": "CHAMPION_KILL", "timestamp": 60_000, "killerId": 0, "victimId": 6, "assistingParticipantIds": []},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    state = _match(events).query(60_000)
    assert state["team_kills"].tolist() == [0, 0]
    assert state["kda"]["kills"].sum() == 0
    assert state["kda"]["deaths"][5] == 1
    assert state["kda"]["environmental_deaths"][0] == 1


def test_team_mapping_requires_exact_five_each_and_zero_static_ids_keep_unknown_provenance():
    with pytest.raises(ValueError, match="exact 5v5"):
        ObservableMatch(
            {"minute_ts": np.asarray([0], dtype=np.int64), "node_frame": np.zeros((1, 10, 46), np.float32), "xy": np.zeros((1, 10, 2), np.float32), "gold_team": np.zeros((1, 2), np.float32)},
            _meta(bad_team=True), [{"type": "GAME_END", "timestamp": 0, "winningTeam": 100}], "15.14",
        )
    match = _match(zeros=True)
    assert match.static_ids.shape == (10, 14)
    assert np.all(match.static_ids == 0)
    assert any(reason.startswith("C.champion_id") for reason in match.typed_projection()["event_reasons"])


def test_feature_description_has_adopted_order_and_excludes_legacy_channels():
    description = ordered_feature_description()
    fields = description["ordered_fields"]
    assert description["frame_width"] == 46
    assert [field["group"] for field in fields[:46]] == ["E"] * 6 + ["R"] * 27 + ["D"] * 13
    ids = {field["id"] for field in fields}
    assert "E.alive" not in ids
    assert "E.gps_norm" not in ids
    assert "champion_name_id" not in " ".join(ids)
    assert len(ids) == len(fields)


def test_numeric_bank_roundtrip_preserves_query_state():
    events = [
        {"type": "CHAMPION_KILL", "timestamp": 60_000, "killerId": 1, "victimId": 6, "assistingParticipantIds": [2]},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    original = _match(events)
    bank = original.export_bank()
    assert all(isinstance(value, np.ndarray) and not value.dtype.hasobject for value in bank.values())
    reopened = ObservableMatch.from_bank(bank, patch="15.14")
    left = original.query(60_000)
    right = reopened.query(60_000)
    np.testing.assert_array_equal(left["node_frame"], right["node_frame"])
    np.testing.assert_array_equal(left["team_kills"], right["team_kills"])
    assert left["kills"] == right["kills"]


def test_terminal_duplicate_markers_are_ambiguous_and_bank_keeps_original_ordinals():
    duplicate_terminal = _match([
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ])
    assert duplicate_terminal.end_ms is None
    unsorted = _match([
        {"type": "CHAMPION_KILL", "timestamp": 60_000, "killerId": 1, "victimId": 6, "assistingParticipantIds": []},
        {"type": "LEVEL_UP", "timestamp": 0, "participantId": 1, "level": 2},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ])
    reopened = ObservableMatch.from_bank(unsorted.export_bank(), patch="15.14")
    assert [event.source_index for event in reopened.typed_events if event.timestamp_valid] == [1, 0, 2]


def test_missing_game_end_does_not_use_unverified_metadata_fallback():
    arrays = {
        "minute_ts": np.asarray([0, 60_000], dtype=np.int64),
        "node_frame": np.zeros((2, 10, 46), dtype=np.float32),
        "xy": np.zeros((2, 10, 2), dtype=np.float32),
        "gold_team": np.zeros((2, 2), dtype=np.float32),
    }
    meta = _meta()
    meta["end_ms"] = 60_000
    meta["winning_team"] = 100
    match = ObservableMatch(arrays, meta, [], "15.14")
    assert match.end_ms is None and match.winning_team is None


def test_numeric_queries_are_compact_and_supported_range_is_enforced():
    match = _match()
    numeric = match.query_numeric(60_000)
    assert numeric["audit"]["history_materialized"] is False
    assert numeric["event_cutoff_index"] == 0
    batch = match.validate_queries([0, 60_000, 120_000])
    assert batch["frame_index"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="outside supported"):
        match.query_numeric(120_001)
    with pytest.raises(ValueError, match="nonnegative"):
        match.query_numeric(-1)


def test_objective_categories_keep_verified_subtype_and_unknown_diagnostics():
    events = [
        {"type": "ELITE_MONSTER_KILL", "timestamp": 60_000, "monsterType": "DRAGON", "monsterSubType": "INFERNAL_DRAGON", "killerTeamId": 100, "position": {"x": 10, "y": 20}},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 60_000, "monsterType": "BARON_NASHOR", "killerTeamId": 200},
        {"type": "BUILDING_KILL", "timestamp": 60_000, "buildingType": "INHIBITOR_BUILDING", "teamId": 100},
        {"type": "TURRET_PLATE_DESTROYED", "timestamp": 60_000, "teamId": 200},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 60_000, "monsterType": "UNVERIFIED_MONSTER", "killerTeamId": 100},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    state = _match(events).query(60_000)
    counts = state["objectives"]["counts"]
    assert counts["dragon"]["blue"] == 1
    assert counts["baron"]["red"] == 1
    assert counts["inhibitor"]["red"] == 1
    assert counts["plate"]["blue"] == 1
    assert counts["tower"]["total"] == 0
    assert state["objectives"]["unknown_category_count"] == 1
    assert any("unknown_objective_category" in reason for reason in state["audit"]["event_reasons"])
    dragon = next(row for row in state["objectives"]["history"] if row["objective_category"] == "dragon")
    assert dragon["monster_subtype"] == "INFERNAL_DRAGON"
    assert dragon["position"] == {"x": 10.0, "y": 20.0}


def test_typed_bank_keeps_objective_strings_position_skill_type_and_undo_gold_gain():
    events = [
        {"type": "ELITE_MONSTER_KILL", "timestamp": 60_000, "monsterType": "DRAGON", "monsterSubType": "OCEAN_DRAGON", "killerTeamId": 100, "position": {"x": 1, "y": 2}},
        {"type": "SKILL_LEVEL_UP", "timestamp": 60_000, "participantId": 1, "skillSlot": 2, "levelUpType": "NORMAL"},
        {"type": "ITEM_UNDO", "timestamp": 60_000, "participantId": 1, "beforeId": 0, "afterId": 1001, "goldGain": 50},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    original = _match(events)
    bank = original.export_bank()
    assert set(("event_category_code", "event_string_codes", "event_position_xy", "event_position_validity")) <= set(bank)
    assert bank["event_position_xy"].dtype == np.dtype("float64")
    assert all(isinstance(value, np.ndarray) and not value.dtype.hasobject for value in bank.values())
    reopened = ObservableMatch.from_bank(bank, patch="15.14")
    state = reopened.query(60_000)
    assert state["skill_level_ups"][0]["payload"]["levelUpType"] == "NORMAL"
    assert state["item_transactions"][0]["gold_gain"] == 50
    dragon = next(row for row in state["objectives"]["history"] if row["objective_category"] == "dragon")
    assert dragon["monster_subtype"] == "OCEAN_DRAGON"
    assert dragon["position"] == {"x": 1.0, "y": 2.0}


def test_unknown_string_tokens_survive_bank_reload_without_objective_credit():
    events = [
        {"type": "ELITE_MONSTER_KILL", "timestamp": 60_000, "monsterType": "FUTURE_MONSTER", "monsterSubType": "FUTURE_SUBTYPE", "killerTeamId": 100},
        {"type": "SKILL_LEVEL_UP", "timestamp": 60_000, "participantId": 1, "skillSlot": 2, "levelUpType": "FUTURE_LEVEL_KIND"},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    original = _match(events)
    reopened = ObservableMatch.from_bank(original.export_bank(), patch="15.14")
    state = reopened.query(60_000)
    assert state["objectives"]["counts"]["dragon"]["total"] == 0
    objective = state["objectives"]["history"][0]
    assert objective["objective_category"] is None
    assert objective["monster_type"] == "FUTURE_MONSTER"
    assert objective["monster_subtype"] == "FUTURE_SUBTYPE"
    typed = next(event for event in reopened.typed_events if event.event_type == "ELITE_MONSTER_KILL")
    assert typed.payload["monsterType"] == "FUTURE_MONSTER"
    assert typed.payload["monsterSubType"] == "FUTURE_SUBTYPE"
    assert state["skill_level_ups"][0]["payload"]["levelUpType"] == "FUTURE_LEVEL_KIND"


def test_explicit_source_identity_is_retained_without_hash_only_reload():
    events = [
        {"type": "CHAMPION_KILL", "timestamp": 60_000, "source_id": "evt-α", "killerId": 1, "victimId": 6},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    reopened = ObservableMatch.from_bank(_match(events).export_bank(), patch="15.14")
    event = next(event for event in reopened.typed_events if event.event_type == "CHAMPION_KILL")
    assert event.source_identity == "source_id:evt-α"


def test_invalid_recorded_position_keeps_presence_unknown_through_bank():
    events = [
        {"type": "CHAMPION_KILL", "timestamp": 60_000, "killerId": 1, "victimId": 6, "position": {"x": "bad"}},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    reopened = ObservableMatch.from_bank(_match(events).export_bank(), patch="15.14")
    typed = next(event for event in reopened.typed_events if event.event_type == "CHAMPION_KILL")
    assert "position" in typed.payload and typed.payload["position"] == {}


def test_invalid_assist_container_keeps_presence_unknown_through_bank():
    events = [
        {"type": "CHAMPION_KILL", "timestamp": 60_000, "killerId": 1, "victimId": 6, "assistingParticipantIds": "bad"},
        {"type": "GAME_END", "timestamp": 120_000, "winningTeam": 100},
    ]
    reopened = ObservableMatch.from_bank(_match(events).export_bank(), patch="15.14")
    typed = next(event for event in reopened.typed_events if event.event_type == "CHAMPION_KILL")
    assert "assistingParticipantIds" in typed.payload and typed.payload["assistingParticipantIds"] is None
