from __future__ import annotations

import json

import pytest

from reports.v_model_input_20261003.independent import (
    HistoryAuditMismatch,
    audit_match,
    event_delta,
    literal_delta,
)


def _record(timestamp: int, event_type: str, payload: dict, *, objective_category=None, presence=None, validity=None):
    fields = tuple(payload)
    return {
        "timestamp_ms": timestamp,
        "event_type": event_type,
        "payload": payload,
        "field_presence": {name: True for name in fields} if presence is None else presence,
        "field_validity": {name: True for name in fields} if validity is None else validity,
        **({"objective_category": objective_category} if objective_category is not None else {}),
    }


def test_audit_match_compares_all_seven_types_items_and_inclusive_boundaries():
    records = [
        _record(10, "LEVEL_UP", {"participantId": 1, "level": 2}),
        _record(100, "CHAMPION_KILL", {"killerId": 1, "victimId": 6, "assistingParticipantIds": [2, 2]}),
        _record(100, "ELITE_MONSTER_KILL", {"killerTeamId": 100, "monsterType": "DRAGON"}, objective_category="dragon"),
        _record(200, "BUILDING_KILL", {"killerId": 0, "teamId": 100, "buildingType": "TOWER_BUILDING"}, objective_category="tower"),
        _record(300, "TURRET_PLATE_DESTROYED", {"teamId": 200}, objective_category="plate"),
        _record(400, "DRAGON_SOUL_GIVEN", {"teamId": 100, "dragonSoul": "OCEAN"}, objective_category="dragon_soul"),
        _record(500, "SKILL_LEVEL_UP", {"participantId": 6, "skillSlot": 2, "levelUpType": "Q"}),
        _record(600, "ITEM_PURCHASED", {"participantId": 1, "itemId": 1001}),
        _record(700, "GAME_END", {"winningTeam": 100}),
    ]
    result = audit_match(records, [0, 100, 100, 350, 999])
    assert result == {
        "events_checked": 9,
        "nonitem_events": 7,
        "item_events_skipped": 1,
        "unsupported_events_skipped": 1,
        "query_boundaries": 5,
        "prefix_events_checked": 3 + 3 + 5 + 9,
    }


def test_literal_delta_retains_assist_multiplicity_and_explicit_team_only():
    record = _record(
        100,
        "CHAMPION_KILL",
        {"killerId": 1, "victimId": 6, "assistingParticipantIds": [2, 2, 0, 11, "3"], "teamId": 200},
    )
    delta = event_delta(record)
    node_two = [count for lane, token, count in delta if lane == ("node", 1) and '"assist","credit"' in token]
    assert node_two == [2]
    assert not any(lane == ("node", 2) and '"assist","credit"' in token for lane, token, _ in delta)
    assert any(lane == ("team", 1) and '"teamId","event"' in token for lane, token, _ in delta)

    building = _record(200, "BUILDING_KILL", {"killerId": 0, "teamId": 100, "buildingType": "TOWER_BUILDING"}, objective_category="tower")
    building_delta = event_delta(building)
    assert any(lane == ("team", 0) and '"teamId","event"' in token for lane, token, _ in building_delta)
    assert not any(lane[0] == "node" and '"killer","credit"' in token for lane, token, _ in building_delta)


def test_source_flags_are_separate_from_routing_and_forbidden_raw_values_do_not_tokenize():
    record = _record(
        100,
        "SKILL_LEVEL_UP",
        {"participantId": 1, "skillSlot": 2, "levelUpType": "Q", "position": {"x": 1, "y": 2}},
        presence={"participantId": "false", "skillSlot": True, "levelUpType": True},
        validity={"participantId": True, "skillSlot": True, "levelUpType": True},
    )
    status, delta = literal_delta(record)
    assert status == "accepted"
    serialized = json.dumps(delta, ensure_ascii=False)
    assert "position" not in serialized
    assert any(
        lane == ("global", -1)
        and '"participantId","source_status"' in token
        and '"absent_valid"' in token
        for lane, token, _count in delta
    )
    assert not any(lane[0] == "node" for lane, _token, _count in delta)


def test_strict_timestamps_and_delta_mismatch_fail_closed():
    record = _record(100, "LEVEL_UP", {"participantId": 1, "level": 2})
    with pytest.raises(ValueError):
        audit_match([record], [100.0])
    with pytest.raises(ValueError):
        audit_match([record, _record(99, "LEVEL_UP", {"participantId": 1, "level": 2})], [100])

    # The driver must raise on a worker/oracle mismatch rather than returning a
    # partial or accepted report.  Injecting malformed source maps exercises
    # the strict record-shape boundary before any aggregate is accepted.
    malformed = dict(record)
    malformed.pop("field_validity")
    with pytest.raises(ValueError, match="require payload"):
        audit_match([malformed], [100])


@pytest.mark.parametrize("mutation,location", [
    ("delta", "event_delta"),
    ("timestamp", "worker_timestamps"),
])
def test_audit_driver_rejects_mutated_worker_state(monkeypatch, mutation, location):
    from studies.v_model_input_20261003 import history as production

    original_add = production.HistoryAccumulator.add

    def broken_add(self, row):
        status = original_add(self, row)
        if mutation == "delta" and status == "accepted":
            lane, token, count = self._deltas[-1][0]
            self._deltas[-1] = ((lane, token, count + 1),) + tuple(self._deltas[-1][1:])
        if mutation == "timestamp":
            self._timestamps[-1] += 1
        return status

    monkeypatch.setattr(production.HistoryAccumulator, "add", broken_add)
    with pytest.raises(HistoryAuditMismatch, match=location):
        audit_match([_record(100, "LEVEL_UP", {"participantId": 1, "level": 2})], [100])


def test_source_valid_noncanonical_ids_are_checked_against_production():
    records = []
    for value in (1, 0, 11, 1.0, True, "1", None):
        for valid in (False, True):
            records.append(_record(len(records), "LEVEL_UP", {"participantId": value, "level": 2.0},
                                   validity={"participantId": valid, "level": valid}))
    result = audit_match(records, [0, 3, 8, 99])
    assert result["events_checked"] == len(records)
    assert result["nonitem_events"] == len(records)
