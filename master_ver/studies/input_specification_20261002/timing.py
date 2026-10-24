"""Pure temporal eligibility checks for the adopted post-state contract.

This module accepts caller-verified actual timestamps and complete coverage
proofs.  It does not read cache files, infer missing events, clamp endpoints,
or import production gameplay/model code.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Mapping, Sequence


TIMING_RULES = (
    "T01_FRAME_SOURCE_VERIFIED",
    "T02_TIMESTAMP_INTEGRITY",
    "T03_PRE_BOUNDARY",
    "T04_PRECEDING_ACTUAL_FRAME",
    "T05_FIRST_KILL_FRAME_LAG_INCLUSIVE",
    "T06_COMPLETE_COVERAGE_PROOF",
    "T07_GAME_END_UNIQUE_VERIFIED",
    "T08_GLOBAL_ISOLATION_INCLUSIVE",
    "T09_NEXT_KILL_STRICT_GT_L",
    "T10_NEXT_ANCHOR_STRICT_GT_F",
    "T11_ANCHOR_OVERLAP_REJECTED",
    "T12_ENDPOINT_MINIMUM_AND_TIES",
    "T13_ENDPOINT_FRAME_SUPPORT",
    "T14_NO_ENDPOINT_CLAMP",
    "T15_NO_SYNTHETIC_FRAME_SUPPORT",
)

HORIZONS_S = (60, 90, 120)
PENDING_DECISIONS = (
    "P0_participation_radius_position_alive_uncertainty",
    "missing_scale",
    "minimum_team_size",
    "start_at_or_after_120s_support",
)


@dataclass(frozen=True)
class TimingInputs:
    """Inputs required by :func:`evaluate_timing`.

    ``actual_frame_timestamps`` must be the source-verified frame timestamps,
    not a reconstructed sequence from ``FRAME_MS``.  The two coverage flags
    are independent proof obligations and must be true before global
    isolation is evaluated.  Each ``other_component_anchors`` entry is a
    ``(component_first_kill_ms, component_anchor_ms)`` pair or a mapping with
    those two keys.  This preserves component first-kill ordering separately
    from the anchor used for truncation.
    """

    first_kill_ms: Any
    last_kill_ms: Any
    actual_frame_timestamps: Sequence[Any]
    same_match_kill_timestamps: Sequence[Any]
    other_component_anchors: Sequence[Any]
    foreign_kill_timestamps: Sequence[Any]
    game_end_markers: Sequence[Any]
    frame_source_verified: bool
    component_coverage_verified: bool
    kill_timeline_verified: bool


def _int_timestamp(value: Any, label: str) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, f"{label}:boolean_timestamp"
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or value != math.trunc(value):
            return None, f"{label}:fractional_or_nonfinite_timestamp"
        return None, f"{label}:non_integer_timestamp"
    else:
        return None, f"{label}:non_integer_timestamp"
    if out < 0:
        return None, f"{label}:negative_timestamp"
    return out, None


def _timestamps(values: Sequence[Any], label: str) -> tuple[list[int], list[str]]:
    if values is None:
        return [], [f"{label}:missing_timestamp_sequence"]
    out: list[int] = []
    reasons: list[str] = []
    for idx, value in enumerate(values):
        parsed, reason = _int_timestamp(value, f"{label}[{idx}]")
        if reason:
            reasons.append(reason)
        elif parsed is not None:
            out.append(parsed)
    return out, reasons


def _component_records(values: Sequence[Any]) -> tuple[list[tuple[int, int]], list[str]]:
    records: list[tuple[int, int]] = []
    reasons: list[str] = []
    if values is None:
        return [], ["other_component_anchors:missing_component_records"]
    for idx, value in enumerate(values):
        if isinstance(value, Mapping):
            first_raw = value.get("component_first_kill_ms")
            anchor_raw = value.get("component_anchor_ms")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            first_raw, anchor_raw = value[0], value[1]
        else:
            reasons.append(f"other_component_anchors[{idx}]:first_kill_anchor_pair_required")
            continue
        first, first_reason = _int_timestamp(first_raw, f"other_component_anchors[{idx}].component_first_kill_ms")
        anchor, anchor_reason = _int_timestamp(anchor_raw, f"other_component_anchors[{idx}].component_anchor_ms")
        if first_reason:
            reasons.append(first_reason)
        if anchor_reason:
            reasons.append(anchor_reason)
        if first is not None and anchor is not None:
            records.append((first, anchor))
    return records, reasons


def _status(status: str, *reasons: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "reasons": sorted({reason for reason in reasons if reason})}
    result.update(details)
    return result


def _empty_rule_map() -> dict[str, dict[str, Any]]:
    return {rule: _status("blocked", "not_evaluated") for rule in TIMING_RULES}


def _preceding_frame(frames: Sequence[int], query_ms: int) -> tuple[int | None, int | None]:
    idx = bisect_right(frames, query_ms) - 1
    if idx < 0:
        return None, None
    return int(frames[idx]), idx


def _endpoint_candidates(
    last_kill_ms: int,
    horizon_s: int,
    next_kill_ms: int | None,
    next_anchor_ms: int | None,
    game_end_ms: int,
) -> tuple[int, list[str], list[dict[str, Any]]]:
    candidates: list[tuple[str, int]] = [("horizon", int(last_kill_ms) + 1000 * int(horizon_s))]
    if next_kill_ms is not None:
        candidates.append(("next_kill_minus_1", int(next_kill_ms) - 1))
    if next_anchor_ms is not None:
        candidates.append(("next_anchor_minus_1", int(next_anchor_ms) - 1))
    candidates.append(("game_end_minus_1", int(game_end_ms) - 1))
    post = min(value for _, value in candidates)
    ties = sorted(name for name, value in candidates if value == post)
    return post, ties, [{"reason": name, "candidate_ms": value} for name, value in sorted(candidates)]


def evaluate_timing(inputs: TimingInputs) -> dict[str, Any]:
    """Evaluate inclusive pre/lag rules and h60/h90/h120 endpoint support.

    All reasons are retained in deterministic lexical order.  ``post_by_horizon``
    uses ``null`` for invalid endpoints and never substitutes the last frame.
    ``final_population_ready`` remains false while the adopted pending policy
    decisions are unresolved, even when every timing rule passes.
    """
    rules = _empty_rule_map()
    reasons: list[str] = []

    first_kill, first_reason = _int_timestamp(inputs.first_kill_ms, "first_kill_ms")
    last_kill, last_reason = _int_timestamp(inputs.last_kill_ms, "last_kill_ms")
    frame_ts, frame_reasons = _timestamps(inputs.actual_frame_timestamps, "actual_frame_timestamps")
    kill_ts, kill_reasons = _timestamps(inputs.same_match_kill_timestamps, "same_match_kill_timestamps")
    anchor_records, anchor_reasons = _component_records(inputs.other_component_anchors)
    anchor_ts = [anchor for _, anchor in anchor_records]
    foreign_ts, foreign_reasons = _timestamps(inputs.foreign_kill_timestamps, "foreign_kill_timestamps")
    end_markers, end_reasons = _timestamps(inputs.game_end_markers, "game_end_markers")
    timestamp_reasons = [
        reason
        for reason in [first_reason, last_reason]
        if reason
    ] + frame_reasons + kill_reasons + anchor_reasons + foreign_reasons + end_reasons

    frame_verified = inputs.frame_source_verified is True
    component_verified = inputs.component_coverage_verified is True
    kill_verified = inputs.kill_timeline_verified is True
    rules["T01_FRAME_SOURCE_VERIFIED"] = _status(
        "pass" if frame_verified else "fail",
        *( [] if frame_verified else ["actual_frame_source_unverified"] ),
    )
    if timestamp_reasons:
        rules["T02_TIMESTAMP_INTEGRITY"] = _status("fail", *timestamp_reasons)
    elif len(frame_ts) != len(set(frame_ts)):
        rules["T02_TIMESTAMP_INTEGRITY"] = _status("fail", "duplicate_actual_frame_timestamp")
    elif any(left >= right for left, right in zip(frame_ts, frame_ts[1:])):
        rules["T02_TIMESTAMP_INTEGRITY"] = _status("fail", "actual_frame_timestamps_not_strictly_increasing")
    else:
        rules["T02_TIMESTAMP_INTEGRITY"] = _status("pass")

    if first_kill is None or last_kill is None:
        rules["T03_PRE_BOUNDARY"] = _status("blocked", "kill_boundary_timestamp_invalid")
        rules["T04_PRECEDING_ACTUAL_FRAME"] = _status("blocked", "kill_boundary_timestamp_invalid")
        rules["T05_FIRST_KILL_FRAME_LAG_INCLUSIVE"] = _status("blocked", "kill_boundary_timestamp_invalid")
        pre_ms = None
    else:
        pre_ms = first_kill - 10_000
        pre_reasons: list[str] = []
        if first_kill > last_kill:
            pre_reasons.append("first_kill_after_last_kill")
        if pre_ms < 0:
            pre_reasons.append("pre_negative")
        rules["T03_PRE_BOUNDARY"] = _status("pass" if not pre_reasons else "fail", *pre_reasons, pre_ms=pre_ms)
        if not frame_ts or pre_ms < 0:
            rules["T04_PRECEDING_ACTUAL_FRAME"] = _status("fail", "no_preceding_actual_frame")
            preceding_ms = None
        else:
            preceding_ms, preceding_idx = _preceding_frame(frame_ts, pre_ms)
            if preceding_ms is None:
                rules["T04_PRECEDING_ACTUAL_FRAME"] = _status("fail", "no_preceding_actual_frame")
            else:
                rules["T04_PRECEDING_ACTUAL_FRAME"] = _status(
                    "pass", preceding_ms=preceding_ms, preceding_index=preceding_idx
                )
        if preceding_ms is None:
            rules["T05_FIRST_KILL_FRAME_LAG_INCLUSIVE"] = _status("blocked", "no_preceding_actual_frame")
        else:
            lag_ms = first_kill - preceding_ms
            ok = 10_000 <= lag_ms <= 20_000
            rules["T05_FIRST_KILL_FRAME_LAG_INCLUSIVE"] = _status(
                "pass" if ok else "fail",
                *( [] if ok else ["first_kill_frame_lag_outside_inclusive_10000_20000"] ),
                lag_ms=lag_ms,
                min_ms=10_000,
                max_ms=20_000,
                inclusive=True,
            )

    coverage_ok = component_verified and kill_verified
    coverage_reasons = []
    if not component_verified:
        coverage_reasons.append("complete_component_coverage_proof_required")
    if not kill_verified:
        coverage_reasons.append("complete_kill_timeline_proof_required")
    if coverage_ok and first_kill is not None and first_kill not in kill_ts:
        coverage_reasons.append("kill_timeline_missing_first_kill_timestamp")
    if coverage_ok and last_kill is not None and last_kill not in kill_ts:
        coverage_reasons.append("kill_timeline_missing_last_kill_timestamp")
    if coverage_ok and Counter(foreign_ts) - Counter(kill_ts):
        coverage_reasons.append("foreign_kill_not_in_same_match_kill_timeline")
    if anchor_reasons:
        coverage_reasons.extend(anchor_reasons)
    if coverage_reasons:
        coverage_ok = False
    rules["T06_COMPLETE_COVERAGE_PROOF"] = _status("pass" if coverage_ok else "fail", *coverage_reasons)

    if len(end_markers) != 1:
        rules["T07_GAME_END_UNIQUE_VERIFIED"] = _status(
            "fail", "exactly_one_verified_GAME_END_marker_required", marker_count=len(end_markers)
        )
        game_end_ms = None
    else:
        game_end_ms = end_markers[0]
        end_reasons = []
        if last_kill is not None and last_kill >= game_end_ms:
            end_reasons.append("last_kill_not_before_GAME_END")
        rules["T07_GAME_END_UNIQUE_VERIFIED"] = _status(
            "pass" if not end_reasons else "fail", *end_reasons, game_end_ms=game_end_ms
        )

    isolation_reasons: list[str] = []
    overlap_reasons: list[str] = []
    next_kill_ms: int | None = None
    next_anchor_ms: int | None = None
    if not coverage_ok:
        rules["T08_GLOBAL_ISOLATION_INCLUSIVE"] = _status("blocked", "coverage_proof_required_before_global_isolation")
        rules["T09_NEXT_KILL_STRICT_GT_L"] = _status("blocked", "kill_timeline_proof_required")
        rules["T10_NEXT_ANCHOR_STRICT_GT_F"] = _status("blocked", "component_coverage_proof_required")
        rules["T11_ANCHOR_OVERLAP_REJECTED"] = _status("blocked", "component_coverage_proof_required")
    elif first_kill is None or last_kill is None or pre_ms is None:
        rules["T08_GLOBAL_ISOLATION_INCLUSIVE"] = _status("blocked", "kill_boundary_timestamp_invalid")
        rules["T09_NEXT_KILL_STRICT_GT_L"] = _status("blocked", "kill_boundary_timestamp_invalid")
        rules["T10_NEXT_ANCHOR_STRICT_GT_F"] = _status("blocked", "kill_boundary_timestamp_invalid")
        rules["T11_ANCHOR_OVERLAP_REJECTED"] = _status("blocked", "kill_boundary_timestamp_invalid")
    else:
        isolation_reasons.extend(
            f"foreign_kill_in_closed_isolation[{ts}]"
            for ts in foreign_ts
            if pre_ms <= ts <= last_kill
        )
        isolation_reasons.extend(
            f"other_component_anchor_in_closed_isolation[{ts}]"
            for ts in anchor_ts
            if pre_ms <= ts <= last_kill
        )
        rules["T08_GLOBAL_ISOLATION_INCLUSIVE"] = _status(
            "pass" if not isolation_reasons else "fail", *isolation_reasons,
            interval=[pre_ms, last_kill], inclusive=True,
        )
        next_kills = sorted(ts for ts in kill_ts if ts > last_kill)
        next_kill_ms = next_kills[0] if next_kills else None
        rules["T09_NEXT_KILL_STRICT_GT_L"] = _status(
            "pass",
            "no_next_same_match_kill_after_complete_coverage" if next_kill_ms is None else "",
            next_kill_ms=next_kill_ms,
            strict_gt_last_kill=True,
        )
        next_records = sorted(
            (anchor, component_first)
            for component_first, anchor in anchor_records
            if component_first > first_kill
        )
        next_anchor_ms = next_records[0][0] if next_records else None
        next_anchor_first_kill_ms = next_records[0][1] if next_records else None
        if next_anchor_ms is not None and next_anchor_ms <= last_kill:
            overlap_reasons.append(f"next_component_anchor_le_L[{next_anchor_ms}]")
        rules["T10_NEXT_ANCHOR_STRICT_GT_F"] = _status(
            "pass",
            "no_next_component_anchor_after_complete_coverage" if next_anchor_ms is None else "",
            next_anchor_ms=next_anchor_ms,
            next_component_first_kill_ms=next_anchor_first_kill_ms,
            strict_gt_first_kill=True,
        )
        rules["T11_ANCHOR_OVERLAP_REJECTED"] = _status(
            "pass" if not overlap_reasons else "fail", *overlap_reasons,
            overlap_anchor_le_L=bool(overlap_reasons),
        )

    base_ok = all(
        rules[rule]["status"] == "pass"
        for rule in (
            "T01_FRAME_SOURCE_VERIFIED", "T02_TIMESTAMP_INTEGRITY", "T03_PRE_BOUNDARY",
            "T04_PRECEDING_ACTUAL_FRAME", "T05_FIRST_KILL_FRAME_LAG_INCLUSIVE",
            "T06_COMPLETE_COVERAGE_PROOF", "T07_GAME_END_UNIQUE_VERIFIED",
            "T08_GLOBAL_ISOLATION_INCLUSIVE", "T11_ANCHOR_OVERLAP_REJECTED",
        )
    )

    post_by_horizon: dict[str, Any] = {}
    all_horizon_ok = True
    h90_valid = False
    for horizon_s in HORIZONS_S:
        key = f"h{horizon_s}"
        endpoint_reasons: list[str] = []
        post_ms: int | None = None
        truncation_reasons: list[str] = []
        candidates: list[dict[str, Any]] = []
        if not base_ok or game_end_ms is None or last_kill is None:
            endpoint_reasons.append("timing_prerequisite_failed")
        else:
            post_ms, truncation_reasons, candidates = _endpoint_candidates(
                last_kill, horizon_s, next_kill_ms, next_anchor_ms, game_end_ms
            )
            if post_ms < last_kill:
                endpoint_reasons.append("post_before_last_kill")
            if pre_ms is not None and post_ms <= pre_ms:
                endpoint_reasons.append("post_not_after_pre")
            if not frame_ts:
                endpoint_reasons.append("no_actual_frame_support")
            elif post_ms > frame_ts[-1]:
                endpoint_reasons.append("post_after_last_actual_frame")
            else:
                post_frame_ms, post_idx = _preceding_frame(frame_ts, post_ms)
                if post_frame_ms is None:
                    endpoint_reasons.append("no_preceding_actual_frame_at_post")
                else:
                    candidates_post_frame = {"post_frame_ms": post_frame_ms, "post_frame_index": post_idx}
            if frame_ts and post_ms is not None and post_ms > frame_ts[-1]:
                endpoint_reasons.append("endpoint_clamp_forbidden")
        valid = not endpoint_reasons
        all_horizon_ok = all_horizon_ok and valid
        if horizon_s == 90:
            h90_valid = valid
        pre_frame_ms = None
        post_frame_ms_out = None
        pre_age_ms = None
        post_age_ms = None
        actual_followup_ms = None
        comparison_elapsed_ms = None
        if valid and pre_ms is not None and first_kill is not None and last_kill is not None and post_ms is not None:
            pre_frame_ms, _ = _preceding_frame(frame_ts, pre_ms)
            post_frame_ms_out, _ = _preceding_frame(frame_ts, post_ms)
            pre_age_ms = pre_ms - pre_frame_ms if pre_frame_ms is not None else None
            post_age_ms = post_ms - post_frame_ms_out if post_frame_ms_out is not None else None
            actual_followup_ms = post_ms - last_kill
            comparison_elapsed_ms = post_ms - pre_ms
        post_by_horizon[key] = {
            "post_ms": post_ms if valid else None,
            "valid": valid,
            "reasons": sorted(set(endpoint_reasons)),
            "truncation_reasons": sorted(set(truncation_reasons)),
            "candidates": candidates,
            "F_pre_ms": pre_frame_ms,
            "F_post_ms": post_frame_ms_out,
            "pre_frame_age_ms": pre_age_ms,
            "post_frame_age_ms": post_age_ms,
            "actual_followup_ms": actual_followup_ms,
            "comparison_elapsed_ms": comparison_elapsed_ms,
        }

    rules["T12_ENDPOINT_MINIMUM_AND_TIES"] = _status(
        "pass" if h90_valid else "fail",
        *( [] if h90_valid else ["h90_endpoint_invalid"] ),
        horizons=list(HORIZONS_S),
        main_horizon_s=90,
        by_horizon={key: value["valid"] for key, value in post_by_horizon.items()},
    )
    support_reasons = sorted(
        set(
            reason
            for entry in [post_by_horizon["h90"]]
            for reason in entry["reasons"]
            if reason in {"post_after_last_actual_frame", "no_actual_frame_support", "no_preceding_actual_frame_at_post"}
        )
    )
    rules["T13_ENDPOINT_FRAME_SUPPORT"] = _status(
        "pass" if h90_valid and not support_reasons else "fail",
        *(support_reasons if support_reasons else ([] if h90_valid else ["h90_endpoint_invalid"])),
        main_horizon_s=90,
    )
    rules["T14_NO_ENDPOINT_CLAMP"] = _status(
        "pass" if h90_valid else "fail",
        *( [] if h90_valid else ["invalid_endpoint_is_null_no_clamp"] ),
        main_horizon_s=90,
    )
    rules["T15_NO_SYNTHETIC_FRAME_SUPPORT"] = _status(
        "pass" if frame_verified and bool(frame_ts) and not timestamp_reasons else "fail",
        *( [] if frame_verified and bool(frame_ts) and not timestamp_reasons else ["synthetic_or_unverified_frame_support_forbidden"] ),
    )

    timing_gate_passed = bool(base_ok and h90_valid and rules["T13_ENDPOINT_FRAME_SUPPORT"]["status"] == "pass")
    reasons.extend(
        reason
        for rule in rules.values()
        for reason in rule.get("reasons", [])
        if reason and reason != "not_evaluated"
    )
    reasons.extend(
        reason
        for entry in post_by_horizon.values()
        for reason in entry.get("reasons", [])
        if reason
    )
    final_population_ready = bool(timing_gate_passed and not PENDING_DECISIONS)
    return {
        "rule_ids": list(TIMING_RULES),
        "rule_statuses": rules,
        "pre_ms": pre_ms,
        "first_kill_ms": first_kill,
        "last_kill_ms": last_kill,
        "first_kill_frame_lag_ms": rules["T05_FIRST_KILL_FRAME_LAG_INCLUSIVE"].get("lag_ms"),
        "next_kill_ms": next_kill_ms,
        "next_anchor_ms": next_anchor_ms,
        "game_end_ms": game_end_ms,
        "post_by_horizon": post_by_horizon,
        "all_horizons_valid": all_horizon_ok,
        "timing_gate_passed": timing_gate_passed,
        "pending_decisions": [{"id": decision, "status": "pending"} for decision in PENDING_DECISIONS],
        "final_population_ready": final_population_ready,
        "reasons": sorted(set(reasons)),
        "policy": {
            "cutoff_inclusive": True,
            "pre_formula": "f - 10000",
            "lag_ms": {"min": 10000, "max": 20000, "inclusive": True},
            "global_isolation_interval": "[pre,L]",
            "next_kill_relation": "> L",
            "next_anchor_relation": "> f",
            "endpoint_formula": "min(L+1000*h, next_kill-1, next_anchor-1, GAME_END-1)",
            "missing_next_terms": "omitted only after complete coverage proof",
            "endpoint_clamp": False,
            "synthetic_frames": False,
        },
    }


def rule_table() -> list[dict[str, Any]]:
    """Return stable rule metadata for report integration."""
    descriptions = {
        "T01_FRAME_SOURCE_VERIFIED": "Caller supplied source-verified actual frame timestamps.",
        "T02_TIMESTAMP_INTEGRITY": "Timestamps are nonnegative integers; frames are unique and strictly increasing.",
        "T03_PRE_BOUNDARY": "pre=f-10000 is nonnegative and f<=L.",
        "T04_PRECEDING_ACTUAL_FRAME": "Use max(actual frame timestamp <= pre); never read a future frame.",
        "T05_FIRST_KILL_FRAME_LAG_INCLUSIVE": "first-kill-to-preceding-frame lag is [10000,20000] inclusive.",
        "T06_COMPLETE_COVERAGE_PROOF": "Full component and same-match kill timeline coverage is verified first.",
        "T07_GAME_END_UNIQUE_VERIFIED": "Exactly one independently verified GAME_END marker is present.",
        "T08_GLOBAL_ISOLATION_INCLUSIVE": "No foreign kill or other component anchor occurs in [pre,L].",
        "T09_NEXT_KILL_STRICT_GT_L": "next kill is the earliest same-match kill with timestamp > L.",
        "T10_NEXT_ANCHOR_STRICT_GT_F": "next anchor is the minimum anchor among components whose first kill is strictly > f.",
        "T11_ANCHOR_OVERLAP_REJECTED": "A next anchor <=L invalidates eligibility.",
        "T12_ENDPOINT_MINIMUM_AND_TIES": "h60/h90/h120 endpoints use the adopted minimum and retain tied reasons.",
        "T13_ENDPOINT_FRAME_SUPPORT": "Each valid endpoint has actual frame support and is <= the last actual frame.",
        "T14_NO_ENDPOINT_CLAMP": "Unsupported endpoints remain null; no last-frame clamp.",
        "T15_NO_SYNTHETIC_FRAME_SUPPORT": "Synthetic or fallback frame support cannot pass the guard.",
    }
    return [{"rule_id": rule, "description": descriptions[rule]} for rule in TIMING_RULES]
