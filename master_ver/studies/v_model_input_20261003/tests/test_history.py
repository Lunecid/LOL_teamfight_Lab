from __future__ import annotations

import json

import pytest

from studies.v_model_input_20261003.history import (
    ADD_ACCEPTED,
    ADD_SKIPPED_ITEM,
    ADD_SKIPPED_UNSUPPORTED,
    HistoryAccumulator,
    event_delta,
)


def _record(timestamp: int, event_type: str, payload: dict, *, presence=None, validity=None, **extra):
    fields = tuple(payload)
    return {
        "timestamp_ms": timestamp,
        "event_type": event_type,
        "payload": payload,
        "field_presence": {name: True for name in fields} if presence is None else presence,
        "field_validity": {name: True for name in fields} if validity is None else validity,
        **extra,
    }


def _decoded(snapshot):
    return {
        lane: {
            json.loads(token): count
            for token, count in counters.items()
        }
        for lane, counters in snapshot.items()
        if lane in {"global"}
    }


def _has(counter, event_type, field_role, measure, typed_value):
    token = json.dumps(
        [event_type, field_role, measure, typed_value],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return counter.get(token, 0)


def test_assigned_level_is_node_local_and_snapshot_is_sorted_immutable_json():
    acc = HistoryAccumulator()
    assert acc.add(
        _record(
            100,
            "LEVEL_UP",
            {"participantId": 1, "level": 2, "levelUpType": "NORMAL"},
            source_id="ignored",
        )
    ) == ADD_ACCEPTED

    snapshot = acc.snapshot()
    assert len(snapshot["node"]) == 10
    assert len(snapshot["team"]) == 2
    assert snapshot["global"] == {}
    node = snapshot["node"][0]
    assert _has(node, "LEVEL_UP", "event", "count", ["int", 1]) == 1
    assert _has(node, "LEVEL_UP", "participantId", "status", ["str", "valid"]) == 1
    assert _has(node, "LEVEL_UP", "participantId", "assignment", ["str", "assigned"]) == 1
    assert _has(node, "LEVEL_UP", "level", "value", ["int", 2]) == 1
    assert _has(node, "LEVEL_UP", "levelUpType", "value", ["str", "NORMAL"]) == 1
    assert list(node) == sorted(node)
    json.dumps(snapshot, ensure_ascii=False)
    with pytest.raises(TypeError):
        snapshot["global"]["new"] = 1
    with pytest.raises(TypeError):
        snapshot["node"].append({})


def test_kill_credits_preserve_assist_multiplicity_and_drop_invalid_ids():
    record = _record(
        200,
        "CHAMPION_KILL",
        {
            "killerId": 1,
            "victimId": 6,
            "assistingParticipantIds": [2, 2, 0, 11, "3", True],
        },
    )
    acc = HistoryAccumulator()
    assert acc.add(record) == ADD_ACCEPTED
    snapshot = acc.snapshot()
    global_counter = snapshot["global"]
    assert _has(global_counter, "CHAMPION_KILL", "event", "count", ["int", 1]) == 1
    assert _has(global_counter, "CHAMPION_KILL", "killerId", "status", ["str", "valid"]) == 1
    assert _has(global_counter, "CHAMPION_KILL", "victimId", "status", ["str", "valid"]) == 1
    assert _has(global_counter, "CHAMPION_KILL", "assistingParticipantIds", "status", ["str", "valid"]) == 1
    assert _has(snapshot["node"][0], "CHAMPION_KILL", "killer", "credit", ["int", 1]) == 1
    assert _has(snapshot["node"][5], "CHAMPION_KILL", "victim", "credit", ["int", 1]) == 1
    assert _has(snapshot["node"][1], "CHAMPION_KILL", "assist", "credit", ["int", 1]) == 2
    assert all("0" not in token and "11" not in token for token in snapshot["node"][0])
    assert event_delta(record)


def test_explicit_team_lane_does_not_invert_building_owner_when_killer_is_zero():
    acc = HistoryAccumulator()
    record = _record(
        300,
        "BUILDING_KILL",
        {"killerId": 0, "teamId": 100, "buildingType": "TOWER_BUILDING"},
        objective_category="tower",
    )
    assert acc.add(record) == ADD_ACCEPTED
    snapshot = acc.snapshot()
    assert _has(snapshot["team"][0], "BUILDING_KILL", "teamId", "event", ["int", 1]) == 1
    assert _has(snapshot["team"][0], "BUILDING_KILL", "teamId", "category", ["str", "tower"]) == 1
    assert snapshot["team"][1] == {}
    assert not any(_has(counter, "BUILDING_KILL", "killer", "credit", ["int", 1]) for counter in snapshot["node"])
    assert _has(snapshot["global"], "BUILDING_KILL", "killerId", "status", ["str", "valid"]) == 1


def test_field_status_distinguishes_missing_malformed_and_valid_empty_assists():
    acc = HistoryAccumulator()
    acc.add(
        _record(
            10,
            "CHAMPION_KILL",
            {"killerId": "bad", "assistingParticipantIds": []},
            presence={"killerId": True, "assistingParticipantIds": True},
            validity={"killerId": False, "assistingParticipantIds": True},
        )
    )
    counter = acc.snapshot()["global"]
    assert _has(counter, "CHAMPION_KILL", "killerId", "status", ["str", "malformed"]) == 1
    assert _has(counter, "CHAMPION_KILL", "victimId", "status", ["str", "absent"]) == 1
    assert _has(counter, "CHAMPION_KILL", "assistingParticipantIds", "status", ["str", "valid_empty"]) == 1
    assert not any("bad" in token for token in counter)


def test_unassigned_level_is_global_and_ids_are_never_value_tokens():
    acc = HistoryAccumulator()
    acc.add(
        _record(
            10,
            "SKILL_LEVEL_UP",
            {"participantId": 0, "level": 3, "skillSlot": 2, "levelUpType": "Q"},
        )
    )
    counter = acc.snapshot()["global"]
    assert _has(counter, "SKILL_LEVEL_UP", "event", "count", ["int", 1]) == 1
    assert _has(counter, "SKILL_LEVEL_UP", "participantId", "assignment", ["str", "unassigned"]) == 1
    assert _has(counter, "SKILL_LEVEL_UP", "level", "value", ["int", 3]) == 1
    assert _has(counter, "SKILL_LEVEL_UP", "skillSlot", "value", ["int", 2]) == 1
    assert not any('"participantId","value"' in token for token in counter)


def test_item_and_unsupported_records_are_explicitly_skipped_without_counters():
    acc = HistoryAccumulator()
    assert acc.add(_record(10, "ITEM_PURCHASED", {"participantId": 1, "itemId": 1001})) == ADD_SKIPPED_ITEM
    assert acc.add(_record(20, "GAME_END", {"winningTeam": 100})) == ADD_SKIPPED_UNSUPPORTED
    assert acc.snapshot()["global"] == {}
    assert event_delta(_record(10, "ITEM_SOLD", {"participantId": 1, "itemId": 1001})) == ()


def test_timestamp_order_and_inclusive_monotonic_snapshot_cutoffs_are_guarded():
    acc = HistoryAccumulator()
    acc.add(_record(10, "LEVEL_UP", {"participantId": 1, "level": 2}))
    acc.add(_record(10, "LEVEL_UP", {"participantId": 2, "level": 2}))
    with pytest.raises(ValueError):
        acc.add(_record(9, "LEVEL_UP", {"participantId": 3, "level": 2}))
    at_ten = acc.snapshot_at(10)
    assert _has(at_ten["node"][0], "LEVEL_UP", "event", "count", ["int", 1]) == 1
    assert _has(at_ten["node"][1], "LEVEL_UP", "event", "count", ["int", 1]) == 1
    after_last = acc.snapshot_at(11)
    assert after_last == at_ten
    with pytest.raises(ValueError):
        acc.snapshot_at(9)
    with pytest.raises(ValueError):
        acc.snapshot_at(1.0)


def test_empty_or_post_event_cutoffs_are_valid_but_query_phase_freezes_adds():
    empty = HistoryAccumulator()
    before_any = empty.snapshot_at(500)
    assert before_any["global"] == {} and all(not lane for lane in before_any["node"])
    acc = HistoryAccumulator()
    acc.add(_record(10, "LEVEL_UP", {"participantId": 1, "level": 2}))
    acc.snapshot_at(1000)
    with pytest.raises(ValueError):
        acc.add(_record(1001, "LEVEL_UP", {"participantId": 1, "level": 3}))


def test_extraneous_source_and_coordinate_fields_never_become_tokens():
    acc = HistoryAccumulator()
    acc.add(
        _record(
            10,
            "ELITE_MONSTER_KILL",
            {
                "killerId": 1,
                "monsterType": "DRAGON",
                "position": {"x": 1, "y": 2},
                "winningTeam": 100,
                "source_identity": "event-1",
            },
            source_id="event-1",
        )
    )
    serialized = json.dumps(acc.snapshot(), ensure_ascii=False)
    assert "position" not in serialized
    assert "winningTeam" not in serialized
    assert "source_identity" not in serialized
    assert "event-1" not in serialized


def test_presence_and_validity_flags_do_not_use_string_truthiness():
    acc = HistoryAccumulator()
    acc.add(
        _record(
            10,
            "LEVEL_UP",
            {"participantId": 1, "level": 2},
            presence={"participantId": "false", "level": "true"},
            validity={"participantId": "true", "level": "true"},
        )
    )
    snapshot = acc.snapshot()
    assert _has(snapshot["global"], "LEVEL_UP", "event", "count", ["int", 1]) == 1
    assert _has(snapshot["global"], "LEVEL_UP", "participantId", "status", ["str", "absent"]) == 1
    assert _has(snapshot["global"], "LEVEL_UP", "level", "status", ["str", "absent"]) == 1
    assert all(not counter for counter in acc.snapshot()["node"])


def test_source_status_preserves_raw_flags_when_strict_payload_type_is_unroutable():
    acc = HistoryAccumulator()
    acc.add(
        _record(
            10,
            "LEVEL_UP",
            {"participantId": 1.0, "level": 1.0},
            presence={"participantId": True, "level": True},
            validity={"participantId": True, "level": True},
        )
    )
    acc.add(
        _record(
            20,
            "LEVEL_UP",
            {"participantId": 1.0, "level": 1.0},
            presence={"participantId": True, "level": True},
            validity={"participantId": False, "level": False},
        )
    )
    snapshot = acc.snapshot()
    global_counter = snapshot["global"]
    assert _has(global_counter, "LEVEL_UP", "participantId", "source_status", ["str", "present_valid"]) == 1
    assert _has(global_counter, "LEVEL_UP", "participantId", "source_status", ["str", "present_invalid"]) == 1
    assert _has(global_counter, "LEVEL_UP", "level", "source_status", ["str", "present_valid"]) == 1
    assert _has(global_counter, "LEVEL_UP", "level", "source_status", ["str", "present_invalid"]) == 1
    assert _has(global_counter, "LEVEL_UP", "participantId", "status", ["str", "malformed"]) == 2
    assert _has(global_counter, "LEVEL_UP", "level", "status", ["str", "malformed"]) == 2
    assert all(not counter for counter in snapshot["node"])
