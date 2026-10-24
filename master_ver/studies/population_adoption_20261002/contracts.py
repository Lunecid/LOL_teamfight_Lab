"""Pure policy contracts for the adopted population decisions.

This module accepts already validated source values.  Missing or malformed
values remain ``unknown``; they are never silently treated as a negative
observation or as a zero-sized engagement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import config as C

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"


def _status(value: Any, *, missing: str = UNKNOWN) -> str:
    """Normalize a three-valued status without Python truthiness traps."""
    if value is None:
        return missing
    if isinstance(value, (bool, np.bool_)):
        return PASS if bool(value) else FAIL
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return PASS if int(value) else FAIL
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"pass", "passed", "confirmed_opposition"}:
            return PASS
        if text in {"fail", "failed", "not_confirmed", "unconfirmed"}:
            return FAIL
        if text in {"unknown", "pending", "unavailable", "na", "null"}:
            return UNKNOWN
    return UNKNOWN


def _bool_value(value: Any) -> bool | None:
    state = _status(value)
    return True if state == PASS else False if state == FAIL else None


def _int_value(value: Any) -> int | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _finite_nonnegative(value: Any) -> int | None:
    number = _int_value(value)
    return number if number is not None and number >= 0 else None


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _proxy_bundle(row: Mapping[str, Any], radius: int) -> dict[str, Any]:
    tag = f"tau_r{radius}"
    support = _bool_value(row.get(f"support_ok_{tag}"))
    lower = _finite_nonnegative(row.get(f"p_lower_n_{tag}"))
    upper = _finite_nonnegative(row.get(f"p_upper_n_{tag}"))
    if support is not True or lower is None or upper is None:
        return {
            "radius_units": radius,
            "lower": None,
            "upper": None,
            "support_status": UNKNOWN if support is None else (PASS if support else FAIL),
            "provenance": "cache_only_proxy_or_unknown",
            "exact_proxy": None,
        }
    return {
        "radius_units": radius,
        "lower": lower,
        "upper": upper,
        "support_status": PASS,
        "provenance": "cache_only_proxy",
        "exact_proxy": bool(lower == upper),
    }


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the adoption contract fields needed by this package."""
    errors: list[str] = []
    version = contract.get("version")
    if version != "population_adopted_v1_20261002":
        errors.append("unexpected_contract_version")
    adopted = contract.get("adopted")
    if not isinstance(adopted, Mapping):
        errors.append("missing_adopted_section")
        adopted = {}
    radius = adopted.get("participation_radius", {})
    if radius.get("primary_units") != C.PRIMARY_RADIUS:
        errors.append("primary_radius_mismatch")
    if list(radius.get("sensitivity_units", [])) != list(C.SENSITIVITY_RADII):
        errors.append("sensitivity_radius_mismatch")
    minimum = adopted.get("minimum_scale", {})
    if minimum.get("blue_minimum") != 1 or minimum.get("red_minimum") != 1:
        errors.append("minimum_team_mismatch")
    primary_time = adopted.get("primary_time", {})
    if primary_time.get("minimum_pre_ms") != C.PRIMARY_PRE_MIN_MS:
        errors.append("primary_pre_mismatch")
    required = tuple(contract.get("required_result_fields", ()))
    if required != C.REQUIRED_RESULT_FIELDS:
        errors.append("required_result_fields_mismatch")
    return {
        "ok": not errors,
        "errors": errors,
        "version": version,
        "required_result_fields": list(required),
    }


def evaluate_row(
    row: Mapping[str, Any],
    *,
    source_status: str = PASS,
    label_status: str = UNKNOWN,
) -> dict[str, Any]:
    """Apply the adopted population policy to one diagnostic component row.

    ``source_status`` describes the hash-verified source run.  ``label_status``
    is intentionally independent: this phase has no V/SVI producer, so the
    projection defaults it to ``unknown`` and keeps final readiness false.
    """
    reasons: list[str] = []
    source_state = _status(source_status)
    label_state = _status(label_status)

    f = _finite_nonnegative(row.get("f"))
    last = _finite_nonnegative(row.get("L"))
    supplied_pre = _int_value(row.get("tau"))
    pre = f - C.ANCHOR_OFFSET_MS if f is not None else None
    tau_supplied = supplied_pre is not None
    pre_inconsistent = False
    if not tau_supplied:
        _add_reason(reasons, "pre_support_unknown_or_malformed")
    elif pre is not None and supplied_pre != pre:
        pre_inconsistent = True
        _add_reason(reasons, "pre_does_not_equal_firstkill_minus_10000")
    if f is None or last is None or pre is None:
        time_role = "unknown_time"
        timing_state = UNKNOWN
        _add_reason(reasons, "missing_component_time")
    else:
        time_role = "primary_pre_ge_120000" if pre >= C.PRIMARY_PRE_MIN_MS else "early_pre_lt_120000"
        frame = _finite_nonnegative(row.get("F_tau"))
        lag = _finite_nonnegative(row.get("lag_f_F_tau"))
        time_order_invalid = last < f
        if time_order_invalid:
            _add_reason(reasons, "lastkill_before_firstkill")
        if frame is None or lag is None:
            _add_reason(reasons, "frame_lag_support_unknown")
            timing_state = UNKNOWN
        elif f - frame != lag:
            _add_reason(reasons, "frame_lag_inconsistent_with_firstkill_and_frame")
            timing_state = UNKNOWN
        else:
            timing_state = FAIL if pre_inconsistent else PASS
        candidate = _bool_value(row.get("cand_10_20_tau"))
        expected_candidate = None if lag is None else C.FRAME_LAG_MIN_MS <= lag <= C.FRAME_LAG_MAX_MS
        if candidate is not None and expected_candidate is not None and candidate != expected_candidate:
            _add_reason(reasons, "candidate_flag_inconsistent_with_inclusive_frame_lag")
            timing_state = UNKNOWN
        if candidate is None:
            timing_state = UNKNOWN
            _add_reason(reasons, "frame_lag_support_unknown")
        elif not candidate:
            timing_state = FAIL
            _add_reason(reasons, "firstkill_frame_lag_outside_inclusive_10000_20000")
        foreign = _int_value(row.get("foreign_kills_in_tau_L"))
        anchors = _int_value(row.get("other_anchors_in_tau_L"))
        if foreign is None or anchors is None:
            if timing_state == PASS:
                timing_state = UNKNOWN
            _add_reason(reasons, "global_isolation_support_unknown")
        elif foreign != 0 or anchors != 0:
            timing_state = FAIL
            _add_reason(reasons, "global_isolation_failed_closed_pre_lastkill")
        if not tau_supplied:
            timing_state = UNKNOWN
        if pre_inconsistent or time_order_invalid:
            timing_state = FAIL

    team_map = _bool_value(row.get("team_map_ok"))
    blue = _finite_nonnegative(row.get("c0_n_blue"))
    red = _finite_nonnegative(row.get("c0_n_red"))
    unknown_team = _finite_nonnegative(row.get("c0_n_unknown_team"))
    credited = _finite_nonnegative(row.get("c0_n_credited", row.get("c0_n")))
    env_killer = _finite_nonnegative(row.get("c0_n_env_killer"))
    direct_opposition = _status(row.get("firstkill_direct_opposition"))
    credit_parse_status = _status(row.get("firstkill_credit_parse_status"))
    if credit_parse_status != PASS:
        _add_reason(reasons, "firstkill_credit_parse_not_verified")
    if env_killer is not None and env_killer > 0:
        _add_reason(reasons, "environment_killer_zero_not_player")

    if source_state == FAIL:
        critical_source = FAIL
        opposition = UNKNOWN
        minimum_state = UNKNOWN
        _add_reason(reasons, "critical_source_failed")
    elif source_state == UNKNOWN or credit_parse_status == UNKNOWN or team_map is None or blue is None or red is None or unknown_team is None:
        critical_source = UNKNOWN
        opposition = UNKNOWN
        minimum_state = UNKNOWN
        _add_reason(reasons, "critical_source_or_team_credit_unknown")
    elif credit_parse_status == FAIL or team_map is False or unknown_team != 0:
        critical_source = FAIL
        opposition = UNKNOWN
        minimum_state = UNKNOWN
        _add_reason(reasons, "actual_team_mapping_or_credit_invalid")
    else:
        critical_source = PASS
        if direct_opposition == PASS and blue >= 1 and red >= 1:
            opposition = "confirmed_two_team_firstkill_credit"
            minimum_state = PASS
        elif direct_opposition == UNKNOWN:
            opposition = "opposing_player_credit_unverified"
            minimum_state = UNKNOWN
            _add_reason(reasons, "firstkill_credit_block_association_unverified")
        elif blue == 0 and red == 0:
            opposition = "no_confirmed_player_credit"
            minimum_state = FAIL
            _add_reason(reasons, "no_confirmed_opposing_player_credit")
        else:
            opposition = "opposing_player_credit_unconfirmed"
            minimum_state = FAIL
            _add_reason(reasons, "opposing_player_credit_unconfirmed")
        if credited == 0:
            _add_reason(reasons, "no_valid_firstkill_credit")

    h90 = _bool_value(row.get("valid_h90"))
    if h90 is False:
        _add_reason(reasons, "h90_endpoint_or_frame_support_invalid")
    elif h90 is None:
        _add_reason(reasons, "h90_endpoint_or_frame_support_unknown")
    support_states = [_status(row.get(key)) for key in ("valid_h90", "events_support_ok", "frame_support_ok", "terminal_support_authoritative")]
    support_status = FAIL if FAIL in support_states else UNKNOWN if UNKNOWN in support_states else PASS
    if _bool_value(row.get("events_support_ok")) is not True:
        _add_reason(reasons, "event_support_not_verified")
    if _bool_value(row.get("frame_support_ok")) is not True:
        _add_reason(reasons, "actual_frame_support_not_verified")
    if _bool_value(row.get("terminal_support_authoritative")) is not True:
        _add_reason(reasons, "game_end_support_not_verified")

    proxies = {radius: _proxy_bundle(row, radius) for radius in (C.PRIMARY_RADIUS, *C.SENSITIVITY_RADII)}
    mode = row.get("participation_mode")
    raw = _bool_value(row.get("raw_presence_verified"))
    # Cache presence and a non-cache mode describe a possible measurement
    # source.  They do not prove operational membership without a separate
    # producer flag tying the measurement to actual participation.
    strict_participation_proof = _bool_value(
        row.get("actual_participation_proof")
    )
    actual = bool(
        raw is True
        and strict_participation_proof is True
        and str(mode) not in {"cache_only_proxy", "", "None"}
    )
    if not actual:
        _add_reason(reasons, "starting_scale_is_cache_proxy_or_unknown")
    primary_proxy = proxies[C.PRIMARY_RADIUS]
    if primary_proxy["support_status"] == PASS:
        auxiliary_scale_group = (
            "cache_proxy_exact" if primary_proxy["exact_proxy"] is True else "cache_proxy_ambiguous"
        )
    else:
        auxiliary_scale_group = "unknown_proxy_support"
    starting_scale_group = "verified" if actual else "unknown_scale"

    candidate_eligible = bool(critical_source == PASS and timing_state == PASS and minimum_state == PASS)
    supported_candidate_eligible = bool(candidate_eligible and support_status == PASS)
    # A review cohort is reserved for a known, recorded one-sided credit case.
    # Missing direct relation evidence remains unverified and cannot become a
    # review-confirmed engagement from aggregate team counts alone.
    review_candidate = bool(critical_source == PASS and timing_state == PASS and minimum_state == FAIL)
    if candidate_eligible:
        if support_status != PASS:
            eligibility = "candidate_pending_source_support"
        elif label_state == FAIL:
            eligibility = "invalid_label"
            _add_reason(reasons, "label_invalid")
        elif time_role == "primary_pre_ge_120000":
            eligibility = "candidate_primary_pending_label"
        else:
            eligibility = "candidate_early_pending_label"
            _add_reason(reasons, "early_pre_cohort_preserved_separately")
        if starting_scale_group == "unknown_scale":
            _add_reason(reasons, "unknown_starting_scale_retained_for_overall_candidate")
    elif review_candidate:
        eligibility = "review_candidate_no_confirmed_opposition"
    elif critical_source in {FAIL, UNKNOWN}:
        eligibility = "unverified_critical_source"
    elif timing_state == FAIL:
        eligibility = "excluded_timing_or_isolation"
    elif timing_state == UNKNOWN:
        eligibility = "unverified_timing"
    elif minimum_state == UNKNOWN:
        eligibility = "unverified_critical_source"
    else:
        eligibility = "excluded_minimum_team_credit"

    # Unknown label support is pending, not a negative label.  Keep a separate
    # retention flag for the eventual full label gate.
    overall_retained = bool(supported_candidate_eligible and label_state != FAIL)
    analysis_state = PASS if supported_candidate_eligible and label_state == PASS else FAIL if label_state == FAIL else UNKNOWN
    if not overall_retained and supported_candidate_eligible and label_state == FAIL:
        _add_reason(reasons, "invalid_label_not_retained")
    if not actual:
        size_analysis_eligible = False
    else:
        size_analysis_eligible = bool(supported_candidate_eligible and label_state == PASS)
    reasons = sorted(set(reasons))
    return {
        "canonical_match_key": str(row.get("anon_match_key", row.get("match_local", ""))),
        "component_key": str(row.get("member_key", "")),
        "firstkill_ms": f,
        "lastkill_ms": last,
        "pre_ms": pre,
        "cohort_time_role": time_role,
        "firstkill_credit_blue": blue,
        "firstkill_credit_red": red,
        "opposition_evidence": opposition,
        "firstkill_direct_opposition_status": direct_opposition,
        "firstkill_credit_parse_status": credit_parse_status,
        "critical_source_status": critical_source,
        "timing_status": timing_state,
        "proxy_P0_lower_upper_r1800": proxies[C.PRIMARY_RADIUS],
        "proxy_P0_lower_upper_r2000": proxies[2000],
        "starting_scale_group": starting_scale_group,
        "auxiliary_proxy_scale_group": auxiliary_scale_group,
        "actual_participation_verified": actual,
        "minimum_team_status": minimum_state,
        "support_status": support_status,
        "label_status": label_state,
        "analysis_status": analysis_state,
        "candidate_eligible": candidate_eligible,
        "supported_candidate_eligible": supported_candidate_eligible,
        "review_candidate": review_candidate,
        "overall_retained_pending_label": overall_retained,
        "size_analysis_eligible": size_analysis_eligible,
        "eligibility_status": eligibility,
        "all_reasons": reasons,
        "final_population_ready": False,
    }


def row_reason_text(reasons: Any) -> str:
    """Stable compact encoding for the typed projection archive."""
    if isinstance(reasons, str):
        return reasons
    try:
        return json.dumps(list(reasons or ()), ensure_ascii=True, separators=(",", ":"))
    except TypeError:
        return json.dumps([str(reasons)], ensure_ascii=True, separators=(",", ":"))
