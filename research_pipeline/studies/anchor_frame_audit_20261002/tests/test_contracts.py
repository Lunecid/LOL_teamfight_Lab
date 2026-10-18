"""Synthetic scientific tests for anchor-frame contracts (no corpus substitute)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.anchor_frame_audit_20261002 import contracts as CT  # noqa: E402
from studies.anchor_frame_audit_20261002 import config as C  # noqa: E402


def test_inclusive_includes_tau_exact_10000():
    minute_ts = np.array([0, 60_000, 120_000, 180_000], dtype=np.int64)
    f = 190_000
    # tau = 180000 == frame -> F_inclusive=180000, delta=10000 exact
    r = CT.inclusive_alignment(minute_ts, f)
    assert r["tau"] == 180_000
    assert r["F_inclusive"] == 180_000
    assert r["delta"] == 10_000
    assert r["exact_10000"] is True


def test_strict_excludes_tau_frame():
    minute_ts = np.array([0, 60_000, 120_000, 180_000], dtype=np.int64)
    f = 190_000
    # t_in = 179999 -> F_strict=120000, a=59999 >= 10000 -> not eligible
    r = CT.strict_alignment(minute_ts, f)
    assert r["t_in"] == 179_999
    assert r["F_strict"] == 120_000
    assert r["a"] == 59_999
    assert r["strict_eligible"] is False


def test_inclusive_20000_boundary():
    minute_ts = np.array([0, 60_000, 120_000, 170_000], dtype=np.int64)
    f = 190_000  # tau=180000; F=170000; delta=20000
    r = CT.inclusive_alignment(minute_ts, f)
    assert r["delta"] == 20_000
    assert CT.cumulative_inclusive_eligible(r["delta"], 10_000) is True
    assert CT.cumulative_inclusive_eligible(r["delta"], 9_000) is False


def test_missing_prior_frame():
    minute_ts = np.array([100_000, 160_000], dtype=np.int64)
    f = 50_000  # tau=-negative? 50000-10000=40000 < first frame
    r = CT.inclusive_alignment(minute_ts, f)
    assert r["inclusive_reason"] == "no_preceding_frame"
    assert r["F_inclusive"] == C.SENTINEL_MISSING


def test_irregular_actual_frames_no_rounding():
    minute_ts = np.array([0, 60_014, 120_028, 180_039], dtype=np.int64)
    f = 190_039  # tau=180039 exact on irregular frame
    r = CT.inclusive_alignment(minute_ts, f)
    assert r["F_inclusive"] == 180_039
    assert r["exact_10000"] is True
    # nearest-future must not be used: query just before first usable
    r2 = CT.inclusive_alignment(minute_ts, 60_013 + 10_000)
    # f=70013, tau=60013 -> F=0? 60013 < 60014 so F=0, delta=70013
    assert r2["F_inclusive"] == 0
    assert r2["exact_10000"] is False


def test_equal_ms_first_members_credit_not_auto_1v1():
    # Environmental killer 0 + victim 1 should not invent a 1v1 scale by itself
    killers = np.array([0, 6], dtype=np.int64)
    victims = np.array([1, 2], dtype=np.int64)
    assists = np.array([0, 0], dtype=np.int64)
    blue = [1, 2, 3, 4, 5]
    c0 = CT.team_credit_counts(killers, victims, assists, np.array([0]), blue)
    assert c0["n_env_killer"] == 1
    assert c0["n_blue"] == 1
    assert c0["n_red"] == 0
    assert c0["scale_max_side"] == 1  # not automatic "1v1" label; just max side count
    # Two-sided credits
    c_all = CT.team_credit_counts(killers, victims, assists, np.array([0, 1]), blue)
    assert c_all["n_blue"] == 2 and c_all["n_red"] == 1


def test_next_anchor_overlap_and_endpoints():
    L = 200_000
    kills = [100_000, 200_000, 250_000]
    # next component tau <= L => overlap
    ep = CT.endpoint_bundle(L, kills, next_component_tau=190_000, game_end_ms=400_000)
    assert ep["overlap_next_tau_le_L"] is True
    val = CT.endpoint_validity_flags(ep["e_h90"], L, tau=L - 10_000, support_start=0, last_frame=399_000, overlap_next_tau_le_L=True)
    assert val["valid"] is False
    assert "same_match_overlap_next_start_le_L" in val["invalid_reasons"]


def test_absent_and_duplicate_game_end():
    absent = CT.parse_game_end_events([{"type": "CHAMPION_KILL", "timestamp": 1}])
    assert absent["game_end_known"] is False
    assert absent["game_end_ms"] == C.SENTINEL_MISSING
    # Multiple events with distinct timestamps => ambiguous, not authoritatively known
    dup = CT.parse_game_end_events([
        {"type": "GAME_END", "timestamp": 100},
        {"type": "GAME_END", "timestamp": 200},
    ])
    assert dup["n_game_end"] == 2
    assert dup["game_end_multiple_events"] is True
    assert dup["game_end_duplicate_ts"] is False
    assert dup["game_end_ambiguous_distinct"] is True
    assert dup["game_end_known"] is False
    # Duplicate identical timestamps: multiple events, but not distinct-ambiguous
    same = CT.parse_game_end_events([
        {"type": "GAME_END", "timestamp": 100},
        {"type": "GAME_END", "timestamp": 100},
    ])
    assert same["n_game_end"] == 2
    assert same["game_end_multiple_events"] is True
    assert same["game_end_duplicate_ts"] is True
    assert same["game_end_ambiguous_distinct"] is False
    assert same["game_end_known"] is True
    assert same["game_end_ms"] == 100


def test_game_end_adjacent_previous_object_timestamp_not_stolen():
    # Reproduces the ordinal-4 neighborhood failure mode: LEVEL_UP ts precedes GAME_END.
    events = [
        {"level": 17, "participantId": 2, "timestamp": 1_644_432, "type": "LEVEL_UP"},
        {"gameId": 1, "timestamp": 1_647_338, "type": "GAME_END", "winningTeam": 100},
    ]
    r = CT.parse_game_end_events(events)
    assert r["game_end_known"] is True
    assert r["game_end_ms"] == 1_647_338
    assert r["game_end_all"] == [1_647_338]
    ref = CT.parse_game_end_events_reference(events)
    assert ref["game_end_ms"] == 1_647_338
    assert ref["game_end_known"] is True


def test_game_end_both_key_orders_and_nested_ignored():
    # type-first and timestamp-first objects both parse; nested values ignored.
    events = [
        {"type": "GAME_END", "timestamp": 5000, "meta": {"timestamp": 999}},
        {"timestamp": 5000, "type": "GAME_END", "nested": {"type": "GAME_END", "timestamp": 123}},
    ]
    # Second object is also GAME_END at same time; nested dict is not a top-level event.
    r = CT.parse_game_end_events(events)
    assert r["n_game_end"] == 2
    assert r["game_end_ms"] == 5000
    assert r["game_end_known"] is True
    # Nested GAME_END-looking mapping only counts if it is a top-level array element
    nested_only = CT.parse_game_end_events([
        {"type": "CHAMPION_KILL", "timestamp": 1, "extra": {"type": "GAME_END", "timestamp": 9}},
    ])
    assert nested_only["game_end_known"] is False
    assert nested_only["n_game_end"] == 0


def test_game_end_absent_malformed_and_multiple_terminals():
    absent = CT.parse_game_end_events([])
    assert absent["game_end_known"] is False
    malformed = CT.parse_game_end_events([
        {"type": "GAME_END"},  # missing
        {"type": "GAME_END", "timestamp": True},  # boolean not ms
        {"type": "GAME_END", "timestamp": 12.5},  # fractional — no truncate
        {"type": "GAME_END", "timestamp": -3},
        {"type": "GAME_END", "timestamp": 100},
    ])
    assert malformed["game_end_malformed"] == 4
    assert malformed["game_end_known"] is True
    assert malformed["game_end_ms"] == 100
    multi = CT.parse_game_end_events([
        {"type": "GAME_END", "timestamp": 10},
        {"type": "GAME_END", "timestamp": 20},
        {"type": "GAME_END", "timestamp": 20},
    ])
    assert multi["game_end_multiple_events"] is True
    assert multi["game_end_duplicate_ts"] is True
    assert multi["game_end_ambiguous_distinct"] is True
    assert multi["game_end_known"] is False


def test_parse_timestamp_ms_rejects_bool_and_fraction():
    assert CT.parse_timestamp_ms(True)[0] is None
    assert CT.parse_timestamp_ms(12.7)[0] is None
    assert CT.parse_timestamp_ms(12.0)[0] == 12
    assert CT.parse_timestamp_ms(12)[0] == 12


def test_object_local_loader_matches_full_json_on_adjacent_case(tmp_path, monkeypatch):
    from studies.anchor_frame_audit_20261002 import io_source as IO
    from studies.anchor_frame_audit_20261002 import config as CFG

    payload = [
        {"level": 17, "participantId": 2, "timestamp": 1_644_432, "type": "LEVEL_UP"},
        {"gameId": 1, "realTimestamp": 9, "timestamp": 1_647_338, "type": "GAME_END", "winningTeam": 100},
    ]
    mid = "synthetic_match_adjacent"
    path = tmp_path / f"{mid}.events.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    monkeypatch.setattr(CFG, "CACHE_DIR", tmp_path)
    info, reason = IO.load_events_game_end(mid)
    assert reason == ""
    assert info["game_end_ms"] == 1_647_338
    assert info["terminal_crosscheck_match"] is True
    assert info["game_end_known"] is True


def test_endpoint_e_equals_L_and_last_frame_boundary():
    L = 300_000
    kills = [300_000]  # no next kill
    # next eng soon, GE large -> may clip
    ep = CT.endpoint_bundle(L, kills, next_component_tau=300_001, game_end_ms=500_000)
    # horizon 60 -> 360000; next_eng-1=300000; so e can equal L
    assert ep["e_h60"] == L
    val_eq = CT.endpoint_validity_flags(ep["e_h60"], L, tau=L - 10_000, support_start=0, last_frame=500_000, overlap_next_tau_le_L=False)
    assert val_eq["flags"]["endpoint_ge_L"] is True
    # e < L invalid
    val_lt = CT.endpoint_validity_flags(L - 1, L, tau=L - 10_000, support_start=0, last_frame=500_000, overlap_next_tau_le_L=False)
    assert val_lt["valid"] is False
    assert "endpoint_ge_L" in val_lt["invalid_reasons"]
    # last-frame boundary: e > last_frame invalid
    val_lf = CT.endpoint_validity_flags(L + 1000, L, tau=L - 10_000, support_start=0, last_frame=L + 500, overlap_next_tau_le_L=False)
    assert val_lf["valid"] is False
    assert "endpoint_le_last_frame" in val_lf["invalid_reasons"]


def test_missing_game_end_not_replaced_by_last_frame():
    ep = CT.endpoint_bundle(200_000, [200_000], next_component_tau=None, game_end_ms=None)
    assert ep["missing_terminal_game_end"] is True
    assert ep["e_h90_cand_game_end"] == C.SENTINEL_INF
    # endpoint uses horizon (or other), not last_frame
    assert ep["e_h90"] == 200_000 + 90_000


def test_negative_tau_flag():
    minute_ts = np.array([0, 60_000], dtype=np.int64)
    r = CT.inclusive_alignment(minute_ts, f=5_000)
    assert r["negative_tau"] is True
    assert r["tau"] == -5_000


def test_zero_kill_match_conservation_shape():
    # Structural: zero-kill match yields empty components list contract
    assert C.EXPECTED_ZERO_KILL == 571
    assert C.EXPECTED_TRAIN_MATCHES - C.EXPECTED_ZERO_KILL == 74_102
