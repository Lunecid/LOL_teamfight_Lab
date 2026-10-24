"""Focused all-role population and endpoint checks."""
from __future__ import annotations

import numpy as np

from studies.observable_state_20261002.kernel import ObservableMatch
from studies.observable_state_20261002.population import build_population, normalize_kills


def _match(events, *, frame_ts=(0, 120_000, 200_000), end=300_000):
    team_map = {str(pid): (100 if pid <= 5 else 200) for pid in range(1, 11)}
    arrays = {
        "minute_ts": np.asarray(frame_ts, dtype=np.int64),
        "node_frame": np.zeros((len(frame_ts), 10, 46), dtype=np.float32),
        "xy": np.ones((len(frame_ts), 10, 2), dtype=np.float32),
        "gold_team": np.zeros((len(frame_ts), 2), dtype=np.float32),
    }
    meta = {"team_map": team_map, "static_meta": {}}
    terminal = {"type": "GAME_END", "timestamp": end, "winningTeam": 100}
    return ObservableMatch(arrays, meta, [*events, terminal], "15.14")


def _kill(timestamp, killer=1, victim=6, x=0.0, y=0.0, assists=()):
    return {
        "type": "CHAMPION_KILL",
        "timestamp": timestamp,
        "killerId": killer,
        "victimId": victim,
        "assistingParticipantIds": list(assists),
        "position": {"x": x, "y": y},
    }


def test_transitive_st_components_have_no_diameter_cap():
    match = _match([
        _kill(130_000, killer=1, victim=6, x=0, y=0),
        _kill(143_000, killer=2, victim=7, x=2_000, y=0),
        _kill(156_000, killer=3, victim=8, x=4_000, y=0),
    ])
    rows = build_population(match)
    assert len(rows) == 1
    assert rows[0]["n_members"] == 3
    assert rows[0]["event_member_list"] == [0, 1, 2]


def test_first_timestamp_block_uses_all_kills_for_opposition():
    match = _match([
        _kill(130_000, killer=1, victim=2, x=0, y=0),
        _kill(130_000, killer=6, victim=3, x=100, y=0),
    ])
    row = build_population(match)[0]
    assert row["same_ms_first_n"] == 2
    assert row["firstkill_direct_opposition_status"] == "pass"
    assert row["firstkill_relation_kind"] == "killer"
    assert row["firstkill_credit_blue"] >= 1
    assert row["firstkill_credit_red"] >= 1


def test_environmental_killer_zero_is_not_player_credit():
    row = build_population(_match([_kill(130_000, killer=0, victim=6)]))[0]
    assert row["c0_n_env_killer"] == 1
    assert row["c0_n_credited"] == 1
    assert row["firstkill_credit_blue"] == 0
    assert row["firstkill_direct_opposition_status"] == "fail"
    assert row["candidate_eligible"] is False
    assert "environment_killer_zero_not_player" in row["all_reasons"]


def test_inclusive_pre_frame_lag_boundaries_are_candidates():
    exact_10 = build_population(_match([_kill(130_000)], frame_ts=(0, 120_000), end=300_000))[0]
    exact_20 = build_population(_match([_kill(140_000)], frame_ts=(0, 120_000), end=300_000))[0]
    assert exact_10["lag_f_F_tau"] == 10_000
    assert exact_20["lag_f_F_tau"] == 20_000
    assert exact_10["cand_10_20_tau"] is True
    assert exact_20["cand_10_20_tau"] is True


def test_closed_isolation_counts_foreign_kill_at_pre_and_same_anchor():
    row = build_population(_match([
        _kill(120_000, killer=1, victim=6, x=0, y=0),
        _kill(130_000, killer=1, victim=6, x=10_000, y=0),
    ]))[1]
    assert row["foreign_kills_in_pre_L"] == 1
    assert row["isolated"] is False
    assert "global_isolation_failed_closed_pre_lastkill" in row["all_reasons"]

    same_f = build_population(_match([
        _kill(130_000, killer=1, victim=6, x=0, y=0),
        _kill(130_000, killer=6, victim=1, x=10_000, y=0),
    ]))
    assert len(same_f) == 2
    assert all(row["isolated"] is False for row in same_f)


def test_next_anchor_uses_later_first_kill_minus_ten_seconds_and_invalidates_overlap():
    rows = build_population(_match([
        _kill(130_000, killer=1, victim=6, x=0, y=0),
        _kill(143_000, killer=2, victim=7, x=2_000, y=0),
        _kill(156_000, killer=3, victim=8, x=4_000, y=0),
        _kill(161_000, killer=6, victim=1, x=10_000, y=0),
    ], frame_ts=(0, 120_000, 300_000), end=400_000))
    first = rows[0]
    assert first["next_component_anchor"] == 151_000
    assert first["next_component_anchor"] <= first["L"]
    assert first["valid_h90"] is False
    assert first["candidate_eligible"] is False


def test_endpoint_ties_and_support_are_preserved_without_clamping():
    tied = build_population(_match(
        [_kill(130_000)], frame_ts=(0, 120_000, 220_000), end=220_001
    ))[0]
    assert tied["e_h90"] == 220_000
    assert set(tied["truncation_reasons_h90"]) == {"horizon", "game_end_minus_1"}
    assert tied["valid_h90"] is True

    unsupported = build_population(_match(
        [_kill(130_000)], frame_ts=(0, 120_000, 150_000), end=300_000
    ))[0]
    assert unsupported["e_h90"] > unsupported["F_post_h90"]
    assert unsupported["frame_support_h90"] is False
    assert unsupported["valid_h90"] is False
    assert unsupported["candidate_eligible"] is True
    assert unsupported["supported_candidate_eligible"] is False


def test_early_cohort_and_zero_kill_match_are_retained_separately():
    early = build_population(_match([_kill(110_000)], frame_ts=(0, 100_000), end=300_000))[0]
    assert early["cohort_time_role"] == "early_pre_lt_120000"
    assert early["candidate_eligible"] is True
    assert "early_pre_cohort_preserved_separately" in early["all_reasons"]
    assert build_population(_match([])) == []


def test_normalizer_keeps_source_ordinal_and_missing_assist_unknown():
    match = _match([{
        "type": "CHAMPION_KILL",
        "timestamp": 130_000,
        "killerId": 1,
        "victimId": 6,
        "position": {"x": 0, "y": 0},
    }])
    normalized = normalize_kills(match)
    assert normalized[0]["source_ordinal"] == 0
    assert normalized[0]["assist_field_present"] is False
    assert normalized[0]["assist_status"] == "unknown"
    assert "assist_field_missing" in normalized[0]["issues"]
    row = build_population(match)[0]
    assert row["critical_source_status"] == "pass"
    assert row["candidate_eligible"] is True
    assert row["optional_issue_kill_ordinals"] == [0]
