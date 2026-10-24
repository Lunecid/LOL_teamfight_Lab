import hashlib

import numpy as np
import pytest

from studies.v_input_projection_20261002.schema import QUERY_BLOCK_SPECS, STATIC_BLOCK_SPECS
from studies.v_model_input_20261003.view import make_view, permute_players, replacement_swap, swap_target, swap_view


def fixture_view():
    features = {name: np.arange(max(1, int(np.prod(shape))), dtype=dtype).reshape(shape)
                for name, (dtype, shape) in (*QUERY_BLOCK_SPECS.items(), *STATIC_BLOCK_SPECS.items())}
    features["team_ids"] = np.array([100] * 5 + [200] * 5)
    features["node_frame"][2, 3] = np.nan
    features["items"] = {
        "item_ids": np.array([1001, 1004, 1004, 1006]), "counts": np.array([1, 2, 1, 3]),
        "offsets": np.array([0, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]),
        "recorded_ledger_supported": np.array([True, False] * 5), "reason_mask": np.arange(10),
        "unassigned_observed": True, "whole_inventory_verified": False,
    }
    history = {"version": "fixture", "node": [{str(i): i + 1} for i in range(10)],
               "team": [{"raw_teamId": 2}, {"raw_killerTeamId": 3}], "global": {"unknown": 4}}
    items = {"node": [{"typed_item_context": i} for i in range(10)], "unassigned": {"unknown": 5}}
    return make_view(features, history, item_history=items)


def equal(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_equal(a, b)
    else:
        assert a == b


def test_swap_map_and_involution():
    view = fixture_view()
    swapped = swap_view(view)
    equal(view, swap_view(swapped))
    assert swapped["focal_is_blue"] is False
    for name in ("team_growth", "cached_gold", "team_kills", "bans"):
        np.testing.assert_equal(swapped["features"][name], view["features"][name][::-1])
    np.testing.assert_equal(swapped["features"]["team_growth_delta"], -view["features"]["team_growth_delta"])
    np.testing.assert_equal(swapped["features"]["team_ids"], 300 - view["features"]["team_ids"])
    for name in ("node_frame", "player_kda", "static_ids", "objective_counts", "objective_age_seconds", "items"):
        equal(swapped["features"][name], view["features"][name])
    equal(swapped["item_history"], view["item_history"])
    equal(swapped["nonitem_history"]["global"], view["nonitem_history"]["global"])
    equal(swapped["nonitem_history"]["team"], view["nonitem_history"]["team"][::-1])
    assert swap_target(swap_target(1)) == 1
    swapped["features"]["node_frame"][0, 0] = 99
    assert view["features"]["node_frame"][0, 0] == 0


def test_permutation_routes_ragged_and_histories_and_commutes_with_swap():
    view = fixture_view()
    order = np.array([6, 2, 1, 0, 5, 4, 3, 7, 9, 8])
    changed = permute_players(view, order)
    equal(permute_players(changed, np.argsort(order)), view)
    equal(swap_view(changed), permute_players(swap_view(view), order))
    assert changed["features"]["items"]["item_ids"][:4].tolist() == [1006, 1004, 1001, 1004]
    assert changed["features"]["items"]["offsets"][:5].tolist() == [0, 1, 2, 2, 4]
    assert changed["item_history"]["node"][0] == {"typed_item_context": 6}
    equal(changed["nonitem_history"]["global"], view["nonitem_history"]["global"])


def test_hash_is_literal_seeded_replacement_not_feature():
    raw = b"full_study_20261001_v_swap_v2:fixture|10000|20261001"
    expected = bool(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 2)
    assert replacement_swap("fixture", 10000, 20261001) == expected
    assert "target" not in fixture_view() and "audit" not in fixture_view()
    with pytest.raises(ValueError):
        replacement_swap("fixture", True, 1)


def test_unknown_fields_and_bad_routing_fail_closed():
    view = fixture_view()
    view["features"]["target"] = 1
    with pytest.raises(ValueError):
        swap_view(view)
    view = fixture_view()
    view["features"]["team_ids"][0] = 0
    with pytest.raises(ValueError):
        swap_view(view)
    with pytest.raises(ValueError):
        permute_players(fixture_view(), [0] * 10)
    with pytest.raises(ValueError):
        swap_target(0.5)


def test_actual_immutable_reducer_snapshot_integrates():
    from studies.v_model_input_20261003.history import HistoryAccumulator
    accumulator = HistoryAccumulator()
    accumulator.add({"timestamp_ms": 100, "event_type": "LEVEL_UP",
                     "payload": {"participantId": 1, "level": 2},
                     "field_presence": {"participantId": True, "level": True},
                     "field_validity": {"participantId": True, "level": True}})
    snapshot = accumulator.snapshot()
    view = make_view(fixture_view()["features"], snapshot)
    equal(swap_view(swap_view(view)), view)
    order = np.arange(9, -1, -1)
    equal(permute_players(permute_players(view, order), order), view)
    view["nonitem_history"]["node"][0]["injected"] = 1
    assert "injected" not in snapshot["node"][0]
