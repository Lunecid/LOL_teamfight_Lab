"""Focused contract fixtures for observation_support_20261002 corrections."""
from __future__ import annotations

import math

import numpy as np
import pytest

from studies.observation_support_20261002 import config as C
from studies.observation_support_20261002 import contracts as CT


def test_cutoff_equality_and_frame_differs():
    ts = np.asarray([0, 60_000, 120_000], dtype=np.int64)
    # f such that tau and tm1 land in different minute frames
    f = 130_000  # tau=120000 -> F=120000; tm1=119999 -> F=60000
    d = CT.cutoff_diagnostics(ts, f)
    assert d["frame_differs"] is True
    assert d["F_tau"] == 120_000
    assert d["F_tm1"] == 60_000
    assert d["F_tau_idx"] != d["F_tm1_idx"]


def test_cutoff_same_frame():
    ts = np.asarray([0, 60_000, 120_000], dtype=np.int64)
    f = 125_000  # tau=115000 -> F=60000; tm1=114999 -> F=60000
    d = CT.cutoff_diagnostics(ts, f)
    assert d["same_F"] is True
    assert d["frame_differs"] is False


def test_no_prior_frame():
    ts = np.asarray([100_000], dtype=np.int64)
    d = CT.cutoff_diagnostics(ts, 50_000)
    assert d["tau_reason"] == "no_preceding_frame"
    assert d["F_tau"] == C.SENTINEL_MISSING


def test_irregular_duplicate_nonmonotone_frames():
    assert CT.preceding_frame(np.asarray([], dtype=np.int64), 1)[3] == "empty_minute_ts"
    assert CT.preceding_frame(np.asarray([-1, 0], dtype=np.int64), 0)[3] == "negative_minute_ts"
    assert CT.preceding_frame(np.asarray([10, 5], dtype=np.int64), 10)[3] == "nonmonotone_minute_ts"
    # duplicates allowed if non-decreasing
    F, i, age, reason = CT.preceding_frame(np.asarray([0, 60_000, 60_000], dtype=np.int64), 60_000)
    assert reason == ""
    assert F == 60_000


def test_event_tie_all_inclusive_original_order():
    ts = np.asarray([100, 200, 200, 300], dtype=np.int64)
    idx = CT.events_at_or_before(ts, 200)
    assert idx.tolist() == [0, 1, 2]


def test_count_events_between_and_unknown_sentinel():
    ts = np.asarray([10, 20, 30, 40], dtype=np.int64)
    assert CT.count_events_between(ts, 20, 40) == 2  # 30,40
    assert CT.count_events_at(ts, 20) == 1
    assert CT.unknown_event_count() == C.SENTINEL_MISSING


def test_same_f_disjoint_member_kills_union():
    killers = np.asarray([1, 2], dtype=np.int64)
    victims = np.asarray([6, 7], dtype=np.int64)
    assists = np.asarray([0, 0], dtype=np.int64)
    bits, n_env = CT.union_credit_bitset(killers, victims, assists, np.asarray([0, 1]))
    assert n_env == 0
    assert bits == (1 << 0) | (1 << 1) | (1 << 5) | (1 << 6)


def test_actor0_env_killer():
    bits, n_env = CT.union_credit_bitset(
        np.asarray([0], dtype=np.int64),
        np.asarray([3], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0]),
    )
    assert n_env == 1
    assert bits == (1 << 2)


def test_exact_radius_boundary_and_outside():
    # Kill origin must be nonzero-safe for player XY; cache treats any 0 coord as ambiguous.
    locs_x = np.asarray([100.0])
    locs_y = np.asarray([100.0])
    xy = np.zeros((10, 2), dtype=np.float64)
    hp = np.full(10, 0.5, dtype=np.float64)
    xy[0] = [1900.0, 100.0]  # pid1 exactly on 1800
    xy[1] = [1900.1, 100.0]  # outside
    # nonzero coords for others unused
    for j in range(2, 10):
        xy[j] = [5000.0, 5000.0]
    p = CT.participation_proxy_cache(xy, hp, locs_x, locs_y, 1800.0, c0_bits=0)
    assert p["near_bits"] & (1 << 0)
    assert not (p["near_bits"] & (1 << 1))


def test_raw_position_rejects_string_bool_before_conversion():
    assert CT.raw_position_state("100", "200") == C.RAW_POS_UNKNOWN
    assert CT.raw_position_state(True, 1.0) == C.RAW_POS_UNKNOWN
    assert CT.raw_position_state(1.0, False) == C.RAW_POS_UNKNOWN
    assert CT.raw_position_state(0.0, 0.0) == C.RAW_POS_KNOWN  # numeric 0 allowed
    assert CT.raw_position_state(1, 2) == C.RAW_POS_KNOWN
    assert CT.raw_position_state(1.5, float("nan")) == C.RAW_POS_UNKNOWN


def test_raw_health_rejects_string_bool():
    assert CT.raw_health_state("1") == C.RAW_HP_UNKNOWN
    assert CT.raw_health_state(True) == C.RAW_HP_UNKNOWN
    assert CT.raw_health_state(0) == C.RAW_HP_OBSERVED_ZERO
    assert CT.raw_health_state(10.0) == C.RAW_HP_POSITIVE
    assert CT.raw_health_state(-1) == C.RAW_HP_UNKNOWN


def test_cache_ambiguous_zero_and_bounds():
    locs_x = np.asarray([100.0])
    locs_y = np.asarray([100.0])
    xy = np.zeros((10, 2), dtype=np.float64)
    hp = np.zeros(10, dtype=np.float64)
    # pid1 near with positive hp
    xy[0] = [100.0, 100.0]
    hp[0] = 0.8
    # pid2 near with ambiguous hp=0 — geometric near, not N_lower
    xy[1] = [150.0, 100.0]
    hp[1] = 0.0
    # pid3 missing coords
    xy[2] = [0.0, 0.0]
    hp[2] = 0.5
    p = CT.participation_proxy_cache(xy, hp, locs_x, locs_y, 1800.0, c0_bits=0)
    assert p["lower_le_upper"]
    assert p["n_lower"] == 1
    assert p["n_upper"] >= p["n_lower"]
    # geometric near includes pid1 and pid2
    assert p["geom_near_bits"] & (1 << 0)
    assert p["geom_near_bits"] & (1 << 1)
    # participation near-only lower must NOT include ambiguous-hp nearby player
    assert not (p["part_near_only_lower_bits"] & (1 << 1))
    assert p["part_near_only_lower_bits"] & (1 << 0)
    # ambiguous nearby is in upper, not definitely near-only participation
    assert p["part_near_only_upper_bits"] & (1 << 1)


def test_nearby_ambiguous_hp_absent_from_c0():
    locs_x = np.asarray([100.0])
    locs_y = np.asarray([100.0])
    xy = np.full((10, 2), 9000.0, dtype=np.float64)
    hp = np.full(10, 0.5, dtype=np.float64)
    xy[3] = [110.0, 100.0]  # pid4 nearby (both coords nonzero)
    hp[3] = 0.0  # ambiguous
    c0 = 1 << 0  # pid1 credited only
    p = CT.participation_proxy_cache(xy, hp, locs_x, locs_y, 1800.0, c0_bits=c0)
    assert p["geom_near_only_bits"] & (1 << 3)
    assert not (p["part_near_only_lower_bits"] & (1 << 3))
    assert p["c0_subset_lower"]
    assert p["c0_subset_upper"]


def test_c0_preserved_regardless_of_frame_health():
    locs_x = np.asarray([0.0])
    locs_y = np.asarray([0.0])
    xy = np.full((10, 2), 9000.0, dtype=np.float64)
    hp = np.zeros(10, dtype=np.float64)
    c0 = (1 << 4) | (1 << 5)
    p = CT.participation_proxy_cache(xy, hp, locs_x, locs_y, 1800.0, c0_bits=c0)
    assert (p["p_lower_bits"] & c0) == c0
    assert (p["p_upper_bits"] & c0) == c0


def test_radius_monotonicity():
    assert CT.radius_monotonicity(3, 5)
    assert not CT.radius_monotonicity(5, 3)


def test_recompute_endpoint_ms():
    # horizon only
    e = CT.recompute_endpoint_ms(1000, 60, -1, -1, -1, False)
    assert e == 1000 + 60_000
    # clipped by next kill
    e2 = CT.recompute_endpoint_ms(1000, 60, 5000, -1, -1, False)
    assert e2 == 4999


def test_parse_timestamp_rejects_bool():
    assert CT.parse_timestamp_ms(True)[1] == "boolean_timestamp"
    assert CT.parse_timestamp_ms(1.5)[1] == "fractional_timestamp"
