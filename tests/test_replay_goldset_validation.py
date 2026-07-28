from __future__ import annotations

from argparse import Namespace

from scripts.validate_replay_goldset import (
    digest_fights,
    largest_remainder,
    quantiles,
    select_goldset,
)


def test_largest_remainder_respects_capacity_and_total() -> None:
    allocation = largest_remainder(7, {"a": 5, "b": 3, "c": 2})
    assert sum(allocation.values()) == 7
    assert all(allocation[key] <= capacity for key, capacity in {"a": 5, "b": 3, "c": 2}.items())


def test_select_goldset_is_deterministic_and_balanced_across_time_bins() -> None:
    records = []
    for index in range(40):
        records.append(
            {
                "match_id": f"KR_{index:04d}",
                "public_patch": "26.13",
                "game_creation": 1_000_000 + index,
                "eligible": True,
            }
        )
    args = Namespace(n_matches=20, time_bins=4, seed=7)
    first = select_goldset(records, args)
    second = select_goldset(records, args)
    assert [row["match_id"] for row in first] == [row["match_id"] for row in second]
    counts = {}
    for row in first:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    assert counts == {"26.13/t01": 5, "26.13/t02": 5, "26.13/t03": 5, "26.13/t04": 5}


def test_digest_fights_keeps_only_stable_detector_fields() -> None:
    result = digest_fights(
        [
            {
                "engage_ts": 10_000,
                "first_kill_ts": 20_000,
                "last_kill_ts": 25_000,
                "fight_type": "teamfight",
                "fight_scale": "teamfight",
                "fight_context": "objective_baron",
                "fight_label": "teamfight · objective_baron",
                "centroid_x": 123.4567894,
                "centroid_y": 456.1234567,
                "det_cluster_blue": 3,
                "det_cluster_red": 4,
                "det_kill_count_window": 2,
                "volatile": object(),
            }
        ]
    )
    assert result == [
        {
            "engage_ts": 10_000,
            "first_kill_ts": 20_000,
            "last_kill_ts": 25_000,
            "fight_type": "teamfight",
            "fight_scale": "teamfight",
            "fight_context": "objective_baron",
            "fight_label": "teamfight · objective_baron",
            "centroid_x": 123.456789,
            "centroid_y": 456.123457,
            "blue_participants": 3,
            "red_participants": 4,
            "n_kills": 2,
        }
    ]


def test_quantiles_reports_expected_endpoints_and_median() -> None:
    result = quantiles([1, 2, 3, 4, 5])
    assert result["min"] == 1
    assert result["median"] == 3
    assert result["max"] == 5
