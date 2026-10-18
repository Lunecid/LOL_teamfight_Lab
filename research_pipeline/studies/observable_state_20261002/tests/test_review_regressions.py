"""Independent reviewer reproductions retained as scientific boundary tests."""
import numpy as np
import pytest

from studies.observable_state_20261002.bank import pack_bank, unpack_bank
from studies.observable_state_20261002.kernel import ObservableMatch
from studies.observable_state_20261002.population import normalize_champion_kills
from studies.observable_state_20261002.tests.test_kernel import _match, _meta


def _events(record):
    return [record, {"type": "GAME_END", "timestamp": 120000, "winningTeam": 100}]


@pytest.mark.parametrize("team", [None, True, {}, "100"])
def test_explicit_invalid_objective_team_cannot_fall_back_or_become_known_on_reload(team):
    match = _match(_events({"type": "ELITE_MONSTER_KILL", "timestamp": 60000, "monsterType": "DRAGON",
                           "killerId": 1, "killerTeamId": team}))
    bank = unpack_bank(pack_bank(match.export_bank()))
    loaded = ObservableMatch.from_bank(bank, "15.14")
    for value in (match, loaded):
        assert value.query_numeric(60000)["objective_counts"][0].tolist() == [1, 0, 0]
        assert "killerTeamId" in value.typed_events[0].payload


def test_future_unknown_objective_diagnostic_respects_query_cutoff():
    match = _match(_events({"type": "ELITE_MONSTER_KILL", "timestamp": 90000, "monsterType": "FUTURE_VARIANT"}))
    assert match.query_numeric(0)["unknown_objective_category_count"] == 0
    assert match.query(0)["objectives"]["unknown_category_count"] == 0
    assert match.query_numeric(90000)["unknown_objective_category_count"] == 1


def test_malformed_source_ids_and_assists_survive_bank_without_eligibility_upgrade():
    match = _match(_events({"type": "CHAMPION_KILL", "timestamp": 60000, "killerId": "1", "victimId": 6,
                           "assistingParticipantIds": [2, "3", True], "position": {"x": 1234.123456789, "y": 2000.1}}))
    loaded = ObservableMatch.from_bank(unpack_bank(pack_bank(match.export_bank())), "15.14")
    assert normalize_champion_kills(match) == normalize_champion_kills(loaded)
    assert match.query_numeric(60000)["team_kills"].tolist() == [0, 0]
    assert np.array_equal(match.query_numeric(60000)["kda"], loaded.query_numeric(60000)["kda"])


def test_prefix_kill_timing_is_inclusive_and_duplicate_identity_does_not_advance_age():
    source = {"type": "CHAMPION_KILL", "timestamp": 60000, "killerId": 1, "victimId": 6,
              "assistingParticipantIds": [2], "source_id": "kill:one", "position": {"x": 100, "y": 100}}
    match = _match([source, dict(source, timestamp=90000), {"type": "GAME_END", "timestamp": 120000, "winningTeam": 100}])
    assert np.isnan(match.query_numeric(59999)["kill_timing"]["age_seconds"])
    assert match.query_numeric(60000)["kill_timing"]["age_seconds"] == 0
    assert match.query_numeric(90000)["kill_timing"]["age_seconds"] == 30


def test_negative_cached_selection_ids_are_preserved_and_unknown_after_reload():
    template = _match()
    meta = _meta()
    meta["static_meta"]["champion_by_pid"]["1"] = -7
    meta["static_meta"]["summoner_spells_by_pid"]["2"]["summoner_spell_1_id"] = -3
    meta["static_meta"]["runes_by_pid"]["3"]["primary_rune_1"] = -5
    meta["static_meta"]["bans"]["blue"][0] = -1
    arrays = {"minute_ts": template.minute_ts, "node_frame": template.node_frame,
              "xy": template.xy, "gold_team": template.gold_team}
    original = ObservableMatch(arrays, meta, _events({"type": "LEVEL_UP", "timestamp": 60000,
                                                    "participantId": 1, "level": 2}), "15.14")
    loaded = ObservableMatch.from_bank(unpack_bank(pack_bank(original.export_bank())), "15.14")
    for match in (original, loaded):
        assert match.static_ids[0, 0] == -7
        assert match.static_ids[1, 1] == -3
        assert match.static_ids[2, 5] == -5
        assert match.bans[0, 0] == -1
        assert {"C.champion_id:pid=1", "C.summoner_spell_1_id:pid=2",
                "C.primary_rune_1:pid=3", "C.bans.blue[0]"} <= set(match.static_unknown_fields)


def test_independent_credit_oracle_deduplicates_identity_across_event_types():
    from studies.observable_state_20261002.producer import _independent_credit_check
    events = _events({"type": "LEVEL_UP", "timestamp": 30000, "participantId": 1,
                      "level": 2, "source_id": "shared:event"})
    events.insert(1, {"type": "CHAMPION_KILL", "timestamp": 60000, "killerId": 1, "victimId": 6,
                      "assistingParticipantIds": [2], "source_id": "shared:event"})
    match = _match(events)
    times = [30000, 60000, 90000]
    queries = {"query_ms": np.asarray(times, dtype=np.int32),
               "query_team_kills": np.asarray([match.query_numeric(t)["team_kills"] for t in times]),
               "query_kda": np.asarray([match.query_numeric(t)["kda"] for t in times])}
    assert not queries["query_team_kills"].any()
    _independent_credit_check(events, match, queries)
