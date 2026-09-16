from copy import deepcopy

import numpy as np
import pytest

from gameplay.state_value import StateBuilder, SNAPSHOT_FIELDS, final_outcome, value_labels
from train.state_value_experiment import assert_disjoint, fit_engagement_oof, match_weights, select_value_regularization


def fixture_pack():
    names = [*SNAPSHOT_FIELDS, "champion_id"]
    node = np.ones((4, 10, len(names)), dtype=np.float32)
    return names, {"minute_ts": np.array([0, 60000, 120000, 180000]), "node_minute": node,
                   "meta": {"team_map": {p: 100 if p <= 5 else 200 for p in range(1, 11)},
                            "role_slots": {p: p-1 for p in range(1, 11)}}, "events": []}


def test_future_frames_events_and_winner_cannot_change_pre_state():
    names, pack = fixture_pack()
    expected = StateBuilder(pack, names).at(70000)
    altered = deepcopy(pack)
    altered["node_minute"][2:] = 999
    altered["events"] = [{"type": "ELITE_MONSTER_KILL", "timestamp": 70001, "monsterType": "BARON_NASHOR", "killerTeamId": 100},
                         {"type": "GAME_END", "timestamp": 180000, "winningTeam": 200}]
    actual = StateBuilder(altered, names).at(70000)
    assert actual == expected
    assert actual.snapshot_ms == 60000


def test_objective_acquisition_and_death_history_have_precise_boundaries():
    names, pack = fixture_pack()
    pack["events"] = [
        {"type": "ELITE_MONSTER_KILL", "timestamp": 65000, "monsterType": "BARON_NASHOR", "killerTeamId": 100},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 70000, "monsterType": "DRAGON", "monsterSubType": "ELDER_DRAGON", "killerTeamId": 200},
        {"type": "CHAMPION_KILL", "timestamp": 71000, "victimId": 1, "killerId": 6}]
    b = StateBuilder(pack, names)
    assert b.at(64999).values["blue_baron"] == 0
    assert b.at(65000).values["blue_baron"] == 1
    assert b.at(70000).values["red_elder"] == 1
    assert b.at(70000).values["red_dragons"] == 0
    assert b.at(71000).values["slot0_baron_death_since_acquisition"] == 1
    assert b.at(71000).values["slot1_baron_death_since_acquisition"] == 0
    assert b.at(125000).values["blue_baron_acquired_last_60s"] == 0


def test_dragon_stack_soul_and_structure_owner_are_distinct():
    names, pack = fixture_pack()
    pack["events"] = [
        {"type": "ELITE_MONSTER_KILL", "timestamp": 10000, "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON", "killerTeamId": 200},
        {"type": "DRAGON_SOUL_GIVEN", "timestamp": 20000, "teamId": 200, "dragonSoul": "Infernal"},
        {"type": "BUILDING_KILL", "timestamp": 30000, "teamId": 100, "buildingType": "TOWER_BUILDING", "towerType": "OUTER_TURRET"}]
    s = StateBuilder(pack, names).at(30000).values
    assert s["red_dragons"] == 1 and s["red_dragon_FIRE"] == 1
    assert s["red_soul_event_recorded"] == 1
    assert s["red_soul_FIRE"] == 1
    assert s["red_tower_OUTER_TURRET"] == 1 and s["blue_tower_OUTER_TURRET"] == 0


def test_missing_outcome_and_invalid_state_are_not_silently_labelled():
    with pytest.raises(ValueError):
        final_outcome([])
    with pytest.raises(ValueError):
        final_outcome([{"type": "GAME_END", "timestamp": 1, "winningTeam": 100}, {"type": "GAME_END", "timestamp": 2, "winningTeam": 200}])
    names, pack = fixture_pack()
    with pytest.raises(ValueError):
        StateBuilder(pack, names).at(-1)
    y, delta, reason = value_labels(np.array([.5, .5, .5, np.nan]), np.array([.6, .4, .5, .7]))
    assert y.tolist() == [1, 0, -1, -1]
    assert reason.tolist() == ["labelled", "labelled", "tie", "invalid_state_or_value"]


def test_role_order_is_explicit_and_pre_frame_not_next_frame():
    names, pack = fixture_pack()
    pack["meta"]["role_slots"][1], pack["meta"]["role_slots"][2] = 1, 0
    pack["node_minute"][1, 0, names.index("totalGold_norm")] = 2
    pack["node_minute"][2, 0, names.index("totalGold_norm")] = 99
    s = StateBuilder(pack, names).at(119999)
    assert s.values["slot1_totalGold_norm"] == 2
    assert s.values["slot0_totalGold_norm"] == 1


def test_oof_stacking_never_trains_on_its_validation_matches():
    groups = np.repeat([f"m{i}" for i in range(12)], 4)
    y = np.tile([0, 1, 0, 1], 12)
    X = np.random.default_rng(1).normal(size=(len(y), 3))
    p, ids, audits = fit_engagement_oof(X, y, groups, folds=3, trees=2)
    assert np.isfinite(p).all() and set(ids) == {0, 1, 2}
    for audit in audits:
        assert not set(audit["train_matches"]) & set(audit["validation_matches"])
    with pytest.raises(ValueError):
        assert_disjoint(["m1"], ["m1"])
    w = match_weights(np.array(["a", "a", "b"]))
    assert w[:2].sum() == w[2]


def test_value_regularization_selection_preserves_match_groups():
    groups = np.repeat([f"m{i}" for i in range(12)], 3)
    y = np.repeat(np.arange(12) % 2, 3)
    X = np.random.default_rng(17).normal(size=(len(y), 2))
    C, audit = select_value_regularization(X, y, groups, ["gold", "time"], candidates=(.001, .1))
    assert C == min(audit["scores"], key=lambda row: row["log_loss"])["C"]
    held = []
    for fold in audit["folds"]:
        assert not set(fold["train_matches"]) & set(fold["validation_matches"])
        held.extend(fold["validation_matches"])
    assert sorted(held) == sorted(set(groups))
