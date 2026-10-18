from __future__ import annotations

from studies.input_specification_20261002.schema_extract import build_inventory, parse_config_literals
from studies.input_specification_20261002.timing import TimingInputs, evaluate_timing


def _inputs(
    *,
    first: int = 30_000,
    last: int = 35_000,
    frames: tuple[int, ...] = (0, 20_000, 40_000, 95_000, 125_000, 155_000, 200_000),
    kills: tuple[int, ...] | None = None,
    anchors: tuple[tuple[int, int], ...] = (),
    foreign: tuple[int, ...] = (),
    ends: tuple[int, ...] = (200_000,),
    frame_verified: bool = True,
    component_verified: bool = True,
    kill_verified: bool = True,
) -> TimingInputs:
    return TimingInputs(
        first_kill_ms=first,
        last_kill_ms=last,
        actual_frame_timestamps=frames,
        same_match_kill_timestamps=kills if kills is not None else (first, last),
        other_component_anchors=anchors,
        foreign_kill_timestamps=foreign,
        game_end_markers=ends,
        frame_source_verified=frame_verified,
        component_coverage_verified=component_verified,
        kill_timeline_verified=kill_verified,
    )


def test_declared_schema_counts_and_order_are_read_without_importing_production_modules():
    bindings, _ = parse_config_literals()
    assert len(bindings["NODE_FEATURE_NAMES"]) == 76
    assert len(bindings["GLOBAL_FEATURE_NAMES"]) == 26
    assert len(bindings["EVENT_FEATURE_NAMES"]) == 44
    assert len(bindings["CHAMPION_STATS_KEYS"]) == 25
    assert len(bindings["DAMAGE_STATS_KEYS"]) == 12
    assert bindings["NODE_FEATURE_NAMES"][0] == "champion_id"
    assert bindings["GLOBAL_FEATURE_NAMES"][0] == "time_norm"
    assert bindings["EVENT_FEATURE_NAMES"][0] == "kills_t100"


def test_inventory_keeps_cache_vs_api_parent_and_unknown_evidence_separate():
    inventory = build_inventory()
    fields = {row["actual_name"]: row for row in inventory["fields"]}
    assert fields["champion_id"]["cache_declared"] is True
    assert fields["champion_id"]["api_declared"] is False
    assert fields["champion_id"]["raw_parent_api_declared"] is True
    assert fields["champion_id"]["evidence"]["raw_presence_unknown"] is True
    assert fields["champion_id"]["evidence"]["per_field_actual_support"] == "unverified"
    assert fields["cs_abilityHaste"]["numeric_unit"] == "unitless normalized feature"
    assert fields["time_norm"]["query_safety"] == "future_dependent_full_match_length"
    assert inventory["normalizers"]["CS_DENOM"]["health"] == 8500
    assert inventory["normalizers"]["DS_DENOM"]["totalDamageTaken"] == 200000


def test_every_emitted_local_source_pointer_has_a_source_hash():
    inventory = build_inventory()
    hashed = set(inventory["source_hashes"])
    pointers = [pointer for row in inventory["fields"] for pointer in row["source_pointers"]]
    pointers += [row["source_pointer"] for row in inventory["static_metadata"]]
    pointers += [row["source_pointer"] for row in inventory["event_payloads"]]
    for pointer in pointers:
        source_path = pointer.rsplit(":", 1)[0]
        assert source_path in hashed, pointer


def test_lag_boundaries_are_inclusive_and_future_frame_is_not_used():
    lower = evaluate_timing(_inputs())
    assert lower["rule_statuses"]["T05_FIRST_KILL_FRAME_LAG_INCLUSIVE"]["status"] == "pass"
    assert lower["first_kill_frame_lag_ms"] == 10_000

    upper = evaluate_timing(_inputs(frames=(0, 10_000, 40_000, 95_000, 125_000, 155_000, 200_000)))
    assert upper["rule_statuses"]["T05_FIRST_KILL_FRAME_LAG_INCLUSIVE"]["status"] == "pass"
    assert upper["first_kill_frame_lag_ms"] == 20_000

    future = evaluate_timing(_inputs(frames=(0, 19_000, 20_001, 40_000, 95_000, 125_000, 155_000, 200_000)))
    assert future["rule_statuses"]["T04_PRECEDING_ACTUAL_FRAME"]["preceding_ms"] == 19_000
    assert future["first_kill_frame_lag_ms"] == 11_000


def test_lag_over_20_seconds_fails():
    result = evaluate_timing(_inputs(frames=(0, 9_000, 40_000, 95_000, 125_000, 155_000, 200_000)))
    assert result["rule_statuses"]["T05_FIRST_KILL_FRAME_LAG_INCLUSIVE"]["status"] == "fail"
    assert result["timing_gate_passed"] is False


def test_global_isolation_is_closed_and_same_timestamp_conflicts_fail():
    foreign_at_l = evaluate_timing(_inputs(foreign=(35_000,)))
    assert foreign_at_l["rule_statuses"]["T08_GLOBAL_ISOLATION_INCLUSIVE"]["status"] == "fail"

    same_f_component = evaluate_timing(_inputs(anchors=((30_000, 30_000),)))
    assert same_f_component["rule_statuses"]["T08_GLOBAL_ISOLATION_INCLUSIVE"]["status"] == "fail"

    overlap = evaluate_timing(_inputs(anchors=((40_000, 34_000),)))
    assert overlap["rule_statuses"]["T11_ANCHOR_OVERLAP_REJECTED"]["status"] == "fail"
    assert overlap["post_by_horizon"]["h90"]["post_ms"] is None


def test_next_anchor_is_ordered_by_component_first_kill_and_truncates_after_l():
    result = evaluate_timing(_inputs(anchors=((80_000, 70_000),)))
    assert result["rule_statuses"]["T10_NEXT_ANCHOR_STRICT_GT_F"]["next_anchor_ms"] == 70_000
    assert result["rule_statuses"]["T10_NEXT_ANCHOR_STRICT_GT_F"]["next_component_first_kill_ms"] == 80_000
    assert result["post_by_horizon"]["h90"]["post_ms"] == 69_999
    assert result["post_by_horizon"]["h90"]["truncation_reasons"] == ["next_anchor_minus_1"]


def test_missing_end_and_past_last_frame_are_invalid_without_clamp():
    missing_end = evaluate_timing(_inputs(ends=()))
    assert missing_end["rule_statuses"]["T07_GAME_END_UNIQUE_VERIFIED"]["status"] == "fail"
    assert missing_end["post_by_horizon"]["h90"]["post_ms"] is None

    past_last_frame = evaluate_timing(_inputs(frames=(0, 20_000, 40_000, 95_000, 120_000)))
    assert past_last_frame["post_by_horizon"]["h90"]["post_ms"] is None
    assert "post_after_last_actual_frame" in past_last_frame["post_by_horizon"]["h90"]["reasons"]
    assert past_last_frame["rule_statuses"]["T14_NO_ENDPOINT_CLAMP"]["status"] == "fail"


def test_frame_duplicates_fractional_values_and_unverified_source_fail():
    duplicate = evaluate_timing(_inputs(frames=(0, 20_000, 20_000, 40_000, 95_000, 125_000, 155_000, 200_000)))
    assert duplicate["rule_statuses"]["T02_TIMESTAMP_INTEGRITY"]["status"] == "fail"

    fractional = evaluate_timing(_inputs(frames=(0, 20_000.5, 40_000, 95_000, 125_000, 155_000, 200_000)))
    assert fractional["rule_statuses"]["T02_TIMESTAMP_INTEGRITY"]["status"] == "fail"

    unverified = evaluate_timing(_inputs(frame_verified=False))
    assert unverified["rule_statuses"]["T01_FRAME_SOURCE_VERIFIED"]["status"] == "fail"
    assert unverified["timing_gate_passed"] is False


def test_proof_flags_are_strict_booleans_and_GAME_END_must_be_unique():
    unverified_text = evaluate_timing(_inputs(frame_verified="false"))
    assert unverified_text["rule_statuses"]["T01_FRAME_SOURCE_VERIFIED"]["status"] == "fail"

    unverified_integer = evaluate_timing(_inputs(component_verified=1))
    assert unverified_integer["rule_statuses"]["T06_COMPLETE_COVERAGE_PROOF"]["status"] == "fail"

    duplicate_end = evaluate_timing(_inputs(ends=(200_000, 200_000)))
    assert duplicate_end["rule_statuses"]["T07_GAME_END_UNIQUE_VERIFIED"]["status"] == "fail"


def test_endpoint_equal_to_last_kill_is_valid_after_next_kill_minus_one():
    result = evaluate_timing(
        _inputs(
            first=30_000,
            last=30_000,
            frames=(0, 20_000, 40_000, 200_000),
            kills=(19_000, 30_000, 30_001),
            anchors=((19_000, 9_000),),
            foreign=(19_000, 30_001),
            ends=(200_000,),
        )
    )
    assert result["post_by_horizon"]["h90"]["post_ms"] == 30_000
    assert result["post_by_horizon"]["h90"]["valid"] is True
    assert result["timing_gate_passed"] is True


def test_tied_next_kill_and_next_anchor_reasons_are_retained():
    result = evaluate_timing(
        _inputs(
            first=30_000,
            last=30_000,
            frames=(0, 20_000, 40_000, 200_000),
            kills=(19_000, 30_000, 33_000, 43_000),
            anchors=((19_000, 9_000), (43_000, 33_000)),
            foreign=(19_000, 33_000, 43_000),
            ends=(200_000,),
        )
    )
    assert result["post_by_horizon"]["h90"]["post_ms"] == 32_999
    assert result["post_by_horizon"]["h90"]["truncation_reasons"] == [
        "next_anchor_minus_1",
        "next_kill_minus_1",
    ]


def test_main_timing_gate_is_h90_only_and_final_population_stays_pending():
    result = evaluate_timing(
        _inputs(
            first=30_000,
            last=70_000,
            frames=(0, 20_000, 100_000, 160_000, 180_000),
            ends=(300_001,),
        )
    )
    assert result["post_by_horizon"]["h90"]["valid"] is True
    assert result["post_by_horizon"]["h120"]["valid"] is False
    assert result["all_horizons_valid"] is False
    assert result["timing_gate_passed"] is True
    assert result["final_population_ready"] is False
    main = result["post_by_horizon"]["h90"]
    assert main["F_pre_ms"] == 20_000
    assert main["F_post_ms"] == 160_000
    assert main["actual_followup_ms"] == 90_000
    assert main["comparison_elapsed_ms"] == 140_000


def test_coverage_proof_is_required_before_isolation_and_boundaries_must_be_in_timeline():
    blocked = evaluate_timing(_inputs(component_verified=False))
    assert blocked["rule_statuses"]["T06_COMPLETE_COVERAGE_PROOF"]["status"] == "fail"
    assert blocked["rule_statuses"]["T08_GLOBAL_ISOLATION_INCLUSIVE"]["status"] == "blocked"

    missing_boundary = evaluate_timing(_inputs(kills=(40_000,)))
    assert missing_boundary["rule_statuses"]["T06_COMPLETE_COVERAGE_PROOF"]["status"] == "fail"
    assert "kill_timeline_missing_first_kill_timestamp" in missing_boundary["rule_statuses"]["T06_COMPLETE_COVERAGE_PROOF"]["reasons"]
