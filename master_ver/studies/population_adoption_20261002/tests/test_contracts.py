from __future__ import annotations

import json
from pathlib import Path

from studies.population_adoption_20261002 import config as C
from studies.population_adoption_20261002.contracts import evaluate_row, validate_contract


def _row(**updates):
    row = {
        "anon_match_key": "m00001",
        "member_key": "m00001:c0",
        "f": 130_000,
        "L": 130_000,
        "tau": 120_000,
        "F_tau": 120_000,
        "lag_f_F_tau": 10_000,
        "cand_10_20_tau": 1,
        "foreign_kills_in_tau_L": 0,
        "other_anchors_in_tau_L": 0,
        "team_map_ok": 1,
        "c0_n_blue": 1,
        "c0_n_red": 1,
        "c0_n_unknown_team": 0,
        "c0_n_credited": 2,
        "c0_n_env_killer": 0,
        "firstkill_direct_opposition": "pass",
        "firstkill_credit_parse_status": "pass",
        "participation_mode": "cache_only_proxy",
        "raw_presence_verified": 0,
        "valid_h90": 1,
        "events_support_ok": 1,
        "frame_support_ok": 1,
        "terminal_support_authoritative": 1,
    }
    for radius in (1800, 2000):
        row.update({
            f"support_ok_tau_r{radius}": 1,
            f"p_lower_n_tau_r{radius}": 2,
            f"p_upper_n_tau_r{radius}": 3,
        })
    row.update(updates)
    if "F_tau" not in updates and "f" in updates:
        row["F_tau"] = row["f"] - 10_000
    return row


def test_contract_version_and_required_fields():
    contract_path = Path(C.CONTRACT_PATH)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    check = validate_contract(contract)
    assert check["ok"] is True
    assert tuple(check["required_result_fields"]) == C.REQUIRED_RESULT_FIELDS


def test_primary_boundary_is_inclusive_and_unknown_proxy_is_retained():
    result = evaluate_row(_row())
    assert result["cohort_time_role"] == "primary_pre_ge_120000"
    assert result["timing_status"] == "pass"
    assert result["candidate_eligible"] is True
    assert result["overall_retained_pending_label"] is True
    assert result["starting_scale_group"] == "unknown_scale"
    assert result["actual_participation_verified"] is False
    assert result["final_population_ready"] is False


def test_early_boundary_is_preserved_as_separate_cohort():
    result = evaluate_row(_row(f=129_999, L=129_999, tau=119_999))
    assert result["cohort_time_role"] == "early_pre_lt_120000"
    assert result["candidate_eligible"] is True
    assert "early_pre_cohort_preserved_separately" in result["all_reasons"]


def test_two_team_c0_counts_do_not_fabricate_opposition():
    result = evaluate_row(_row(firstkill_direct_opposition=None))
    assert result["opposition_evidence"] == "opposing_player_credit_unverified"
    assert result["candidate_eligible"] is False
    assert result["review_candidate"] is False
    assert result["eligibility_status"] == "unverified_critical_source"


def test_environment_killer_and_victim_only_are_review_candidate():
    result = evaluate_row(
        _row(
            c0_n_blue=1,
            c0_n_red=0,
            c0_n_credited=1,
            c0_n_env_killer=1,
            firstkill_direct_opposition="fail",
        )
    )
    assert result["opposition_evidence"] == "opposing_player_credit_unconfirmed"
    assert result["candidate_eligible"] is False
    assert result["review_candidate"] is True
    assert "environment_killer_zero_not_player" in result["all_reasons"]


def test_unknown_team_or_time_does_not_become_false():
    result = evaluate_row(_row(team_map_ok=None, c0_n_blue=None, c0_n_red=None, cand_10_20_tau=None, lag_f_F_tau=None))
    assert result["critical_source_status"] == "unknown"
    assert result["timing_status"] == "unknown"
    assert result["eligibility_status"] in {"unverified_critical_source", "unverified_timing"}


def test_label_failure_is_not_a_negative_label():
    result = evaluate_row(_row(), label_status="fail")
    assert result["candidate_eligible"] is True
    assert result["overall_retained_pending_label"] is False
    assert result["eligibility_status"] == "invalid_label"
    assert result["final_population_ready"] is False


def test_malformed_or_missing_pre_cannot_pass_timing():
    malformed = evaluate_row(_row(tau=120000.0))
    missing = evaluate_row(_row(tau=None))
    assert malformed["timing_status"] == "unknown"
    assert missing["timing_status"] == "unknown"
    assert malformed["candidate_eligible"] is False
    assert missing["candidate_eligible"] is False


def test_string_boolean_status_is_unknown_until_verified():
    result = evaluate_row(_row(cand_10_20_tau="true"))
    assert result["timing_status"] == "unknown"
    assert result["candidate_eligible"] is False


def test_pre_mismatch_and_last_before_first_are_failures():
    mismatch = evaluate_row(_row(tau=119999))
    reversed_time = evaluate_row(_row(L=129999))
    assert mismatch["timing_status"] == "fail"
    assert "pre_does_not_equal_firstkill_minus_10000" in mismatch["all_reasons"]
    assert reversed_time["timing_status"] == "fail"
    assert "lastkill_before_firstkill" in reversed_time["all_reasons"]


def test_actual_participation_requires_strict_operational_proof():
    result = evaluate_row(
        _row(
            raw_presence_verified=1,
            participation_mode="participant_frames",
            actual_participation_proof=1,
        ),
        label_status="pass",
    )
    assert result["actual_participation_verified"] is True
    assert result["size_analysis_eligible"] is True


def test_credit_parse_must_pass_even_with_positive_relation():
    import numpy as np
    from studies.population_adoption_20261002.projection import _rows_from_arrays
    rows = [_row(firstkill_credit_parse_status=s) for s in ("pass", "unknown", "fail")]
    arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    result = _rows_from_arrays(arrays, "pass")
    assert result["candidate_eligible"].tolist() == [True, False, False]
    for row, expected in zip(rows, (True, False, False)):
        assert evaluate_row(row)["candidate_eligible"] is expected


def test_size_analysis_requires_valid_label():
    row = _row(raw_presence_verified=1, participation_mode="participant_frames", actual_participation_proof=1)
    assert evaluate_row(row, label_status="fail")["size_analysis_eligible"] is False
    assert evaluate_row(row, label_status="unknown")["size_analysis_eligible"] is False
    assert evaluate_row(row, label_status="pass")["size_analysis_eligible"] is True


def test_vector_pending_label_size_matches_scalar():
    import numpy as np
    from studies.population_adoption_20261002.projection import _rows_from_arrays
    row = _row(raw_presence_verified=1, participation_mode="participant_frames", actual_participation_proof=1)
    arrays = {k: np.asarray([v]) for k, v in row.items()}
    projected = _rows_from_arrays(arrays, "pass")
    assert projected["actual_participation_verified"].tolist() == [True]
    assert projected["size_analysis_eligible"].tolist() == [False]
    assert evaluate_row(row)["size_analysis_eligible"] is False
