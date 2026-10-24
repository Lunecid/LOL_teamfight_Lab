"""T1.2 E1 matching features and placebo windows (PRESPEC_E1_KO.md section 3 as amended 2026-10-13; Rulings 14-17).

The plan's six tests (PLAN_E1_E5_KO.md Task 13) adapted to the binding rules: gold edges are a 3 x 4 array (four
edges per time bin), the reject keys are start_age_out, end_range, kill_inside, outcome_overlap and keys_differ,
the grid is tau + 1,000 k (|k| <= 600, starts before 120 s dropped beforehand), kill-gap edges 30, 60, 90 and 180 s
(five bins, 10-13 amendment), end_age open above 40 s, outcome windows of the match's q engagements only (10-13).
Hand checks for the plan's case (tau = 603 s, D = 40 s, one kill at 613 s, outcome window [603, 643] s, frames every
60 s, game end 30 min): a survivor needs start age <= 5 s, no frame in (s, s + 40 s], end age >= 40 s, s < 15 min and
no kill before s within 180 s (kill_gap_bin 4), so s = 60 m + a s with a in 0..5 and m in 2..9 or 14.  The plan
note's r1 tests are marked (a)-(f).
"""
import hashlib

import numpy as np
import pytest

from studies.label_production_20261005.engagements import COMPONENT_COLUMNS
from studies.tog_experiments_20261012 import e1_candidates as ec
from studies.tog_experiments_20261012 import e1_features as ef
from studies.tog_experiments_20261012.e1_candidates import has_kill, match_windows

FRAMES = np.arange(0, 40 * 60_000 + 1, 60_000)
GOLD = np.zeros(len(FRAMES))
EDGES = np.tile(np.array([-3.0, -1.0, 1.0, 3.0]), (3, 1))     # gold 0 falls in bin 2 of every time bin
GAME_END = 30 * 60_000
NO_SPANS = np.zeros((0, 2), np.int64)


def _eng(key, tau, D):
    return {"engagement_key": np.array([key]), "tau_ms": np.array([tau]), "D_ms": np.array([D])}


def _run(eng, kills, spans, frames=FRAMES, game_end=GAME_END, gold=GOLD, edges=EDGES):
    return match_windows(eng, kills, spans, frames, game_end, gold, edges)


def _candidates(tau, D, kills, spans, frames=FRAMES, game_end=GAME_END, gold=GOLD, edges=EDGES):
    return ec.evaluate_engagements(np.array([tau]), np.array([D]), np.asarray(kills), np.asarray(spans), frames,
                                   game_end, gold, edges)


def _keys(start, D, kills, frames=FRAMES, gold=GOLD, edges=EDGES):
    return ef.window_keys(np.array([start]), np.array([D]), frames, np.asarray(kills), gold, edges)


def _names(c):
    """Candidate start -> its first failing rule ("kept" for a survivor)."""
    return {s: ("kept" if r == 0 else ec.REJECT_KEYS[r - 1]) for s, r in zip(c["s_ms"].tolist(), c["reason"].tolist())}


# ----------------------------------------------------------------------------- the plan's six tests, adapted
def test_kill_interval_is_closed():
    got = has_kill(np.array([100, 500]), np.array([0, 101, 500, 501]), np.array([100, 499, 600, 900]))
    assert got.tolist() == [True, False, True, False]


def test_rejects_windows_with_kills():
    kills = np.arange(120_000, 30 * 60_000, 10_000)          # every 30 s window holds a kill
    s, rej = _run(_eng(b"e", 5 * 60_000 + 2_000, 30_000), kills, NO_SPANS)
    assert s[0] == -1 and rej["kill_inside"] > 0
    assert set(rej) == set(ec.REJECT_KEYS)


def test_rejects_outcome_window_overlap():
    # first kill f = 613 s, tau = 603 s, h90 endpoint 643 s: the whole outcome window is excluded
    eng = _eng(b"e", 603_000, 40_000)
    s, rej = _run(eng, np.array([613_000]), np.array([[603_000, 643_000]]))
    assert s[0] != -1 and not (603_000 - 40_000 <= s[0] <= 643_000)


def test_kill_gap_bin_matches():
    # tau has no earlier kill (gap >= 180 s); starts 0-180 s after the 613 s kill must be rejected
    eng = _eng(b"e", 603_000, 40_000)
    s, _ = _run(eng, np.array([613_000]), np.array([[603_000, 643_000]]))
    assert not (613_000 < s[0] < 613_000 + 180_000)


def test_frame_boundary_count_matches():
    eng = _eng(b"e", 603_000, 40_000)                        # (603, 643] holds no frame boundary
    s, _ = _run(eng, np.array([613_000]), np.array([[603_000, 643_000]]))
    assert not np.any((FRAMES > s[0]) & (FRAMES <= s[0] + 40_000))


def test_choice_is_hash_not_nearest():
    a, _ = _run(_eng(b"a", 603_000, 40_000), np.array([613_000]), np.array([[603_000, 643_000]]))
    b, _ = _run(_eng(b"a", 603_000, 40_000), np.array([613_000]), np.array([[603_000, 643_000]]))
    assert a[0] == b[0]                                     # deterministic
    keys = [_run(_eng(k, 603_000, 40_000), np.array([613_000]), np.array([[603_000, 643_000]]))[0][0]
            for k in (b"a", b"b", b"c", b"d", b"e", b"f")]
    assert len(set(keys)) > 1                               # depends on the engagement key
    survivors = _candidates(603_000, 40_000, [613_000], [[603_000, 643_000]])
    feasible = survivors["s_ms"][survivors["reason"] == 0]
    assert len(feasible) == 54                              # m in 2..9 or 14, a in 0..5 (module docstring)
    nearest = feasible[np.argmin(np.abs(feasible - 603_000))]
    assert any(k != nearest for k in keys)                  # not the nearest survivor


# ----------------------------------------------------------------------------- added tests
def test_grid_steps_whole_seconds_from_a_tau_that_is_not_a_multiple_of_1000():
    tau = 603_500
    grid = ec.candidate_grid(tau)
    assert grid.dtype == np.int64 and len(grid) == 1084     # k = -483 .. 600 (starts before 120 s dropped)
    assert grid[0] == 120_500 and grid[-1] == tau + 600_000 and np.all(np.diff(grid) == 1_000)
    assert ec.candidate_grid(603_000)[0] == 120_000         # a start at exactly 120 s stays
    s, _ = _run(_eng(b"e", tau, 40_000), np.array([613_500]), np.array([[603_500, 643_500]]))
    assert s[0] != -1 and (s[0] - tau) % 1_000 == 0 and s[0] % 1_000 == 500
    got, want = _keys(s[0], 40_000, [613_500]), _keys(tau, 40_000, [613_500])
    assert {name: int(v[0]) for name, v in got.items()} == {name: int(v[0]) for name, v in want.items()}
    c = _candidates(tau, 40_000, [613_500], [[603_500, 643_500]])
    np.testing.assert_array_equal(c["s_ms"], grid)


# A match with interleaved components (Opus review M6); frames every 60 s, game end 30 min.  Population endpoints
# e = min(L + 90 s, next kill - 1 ms, next anchor - 1 ms, game end - 1 ms), anchor = first kill - 10 s:
#   B   5 kills 1,040 / 1,050 / 1,060 / 1,070 / 1,075 s, not a q engagement; the nested C (first kill 1,055 s) cuts its
#       endpoint at C's anchor: e_B = 1,044.999 s, before its own last kill (valid_h90 0)
#   C   2 kills 1,055 / 1,056 s nested in B, not a q engagement, e_C = 1,059.999 s (next kill 1,060 s)
#   P   1 kill 1,190 s, not a q engagement, e_P = 1,199.999 s (E's anchor)
#   E   q engagement, 1 kill 1,210 s, tau 1,200 s, e_E = 1,289.999 s (Q2's anchor), D = 89.999 s; kill gap 10 s (bin 0)
#   Q2  q engagement, 1 kill 1,300 s, tau 1,290 s, e = 1,390 s, D = 100 s
#   N   1 kill 1,440 s, not a q engagement, e_N = 1,530 s
COMPONENTS = ((b"B", 1_040_000, 1_075_000, 1_044_999, 0), (b"C", 1_055_000, 1_056_000, 1_059_999, 0),
              (b"P", 1_190_000, 1_190_000, 1_199_999, 1), (b"E", 1_210_000, 1_210_000, 1_289_999, 1),
              (b"Q2", 1_300_000, 1_300_000, 1_390_000, 1), (b"N", 1_440_000, 1_440_000, 1_530_000, 1))
SCENE_KILLS = np.array([1_040_000, 1_050_000, 1_055_000, 1_056_000, 1_060_000, 1_070_000, 1_075_000, 1_190_000,
                        1_210_000, 1_300_000, 1_440_000])


def _component_table(rows):
    col = {name: i for i, name in enumerate(COMPONENT_COLUMNS)}
    values = np.zeros((len(rows), len(COMPONENT_COLUMNS)), dtype=np.int32)
    for i, (_, f, last, e, valid) in enumerate(rows):
        values[i, [col["f"], col["L"], col["e_h90"], col["valid_h90"]]] = (f, last, e, valid)
    return np.array([row[0] for row in rows], dtype="S64"), values


def _scene_spans(q_keys=(b"E", b"Q2"), taus=(1_200_000, 1_290_000), Ds=(89_999, 100_000), rows=COMPONENTS):
    keys, values = _component_table(rows)
    return ec.q_engagement_spans(keys, values, np.array(q_keys, dtype="S64"), np.array(taus), np.array(Ds))


def test_outcome_exclusion_uses_the_q_engagements_of_the_match_only():
    spans = _scene_spans()
    assert spans.tolist() == [[1_200_000, 1_289_999], [1_290_000, 1_390_000]]   # E and Q2, nothing from B, C, P, N
    every = ec.component_spans(np.array([r[1] for r in COMPONENTS]), np.array([r[3] for r in COMPONENTS]))
    new, old = (_names(_candidates(1_200_000, 89_999, SCENE_KILLS, sp)) for sp in (spans, every))
    # (a) kill-free, inside N's outcome window [1,430, 1,530] s, 1 s after N's kill, with every feature of E: kept now,
    #     outcome_overlap under the superseded all-component rule
    assert new[1_441_000] == "kept" and old[1_441_000] == "outcome_overlap"
    # (b) kill-free inside the other q engagement Q2's outcome window [1,290, 1,390] s: outcome_overlap either way
    assert new[1_320_000] == old[1_320_000] == "outcome_overlap"
    # own outcome window: E alone (no Q2, no N), e_E = 1,300 s; the kill-free start 1,260 s lies inside [1,200, 1,300] s
    alone = [r for r in COMPONENTS if r[0] not in (b"Q2", b"N")]
    alone[3] = (b"E", 1_210_000, 1_210_000, 1_300_000, 1)
    own = _scene_spans((b"E",), (1_200_000,), (100_000,), alone)
    assert own.tolist() == [[1_200_000, 1_300_000]]
    assert _names(_candidates(1_200_000, 100_000, SCENE_KILLS[:-2], own))[1_260_000] == "outcome_overlap"


def test_interleaved_components_follow_the_rule():
    # (d) B's endpoint is cut at C's anchor (1,044.999 s < its last kill 1,075 s).  The kill-free start 1,080 s, 5 s
    # after B's last kill, is not excluded (B is not a q engagement, and its window ends before 1,075 s anyway); its kill
    # gap is measured from B's last kill (5 s, bin 0) and matches E's (10 s after P's kill, bin 0).  A start whose window
    # holds B's kills is kill_inside.
    c = _candidates(1_200_000, 89_999, SCENE_KILLS, _scene_spans())
    name = _names(c)
    at = int(np.flatnonzero(c["s_ms"] == 1_080_000)[0])
    assert name[1_080_000] == name[1_085_000] == "kept"
    assert int(c["values"]["kill_gap_ms"][at]) == 5_000 and int(c["keys"]["kill_gap_bin"][at]) == 0
    assert int(c["eng_values"]["kill_gap_ms"][0]) == 10_000 and int(c["eng_keys"]["kill_gap_bin"][0]) == 0
    assert name[1_020_000] == "kill_inside"
    s, rej = _run(_eng(b"m:E", 1_200_000, 89_999), SCENE_KILLS, _scene_spans())
    survivors = c["s_ms"][c["reason"] == 0].tolist()
    assert s[0] == min(survivors, key=lambda v: hashlib.sha256(f"m:E:{v}".encode("utf-8")).hexdigest())


def test_each_candidate_counts_once_under_its_first_failing_rule():
    c = _candidates(603_000, 40_000, [613_000], [[603_000, 643_000]])
    name = _names(c)
    assert name[615_000] == "start_age_out"                 # start age 15 s (also overlaps the outcome window)
    assert name[606_000] == "kill_inside"                   # kill at 613 s inside (also overlapping)
    assert name[600_000] == "kill_inside"
    assert name[546_000] == "keys_differ"                   # start age 6 s: start_age_bin 1, not 0
    assert name[660_000] == "keys_differ"                   # 47 s after the kill: kill_gap_bin 1, not 4
    assert name[541_000] == name[545_000] == "kept"
    s, rej = _run(_eng(b"e", 603_000, 40_000), np.array([613_000]), np.array([[603_000, 643_000]]))
    assert sum(rej.values()) + int((c["reason"] == 0).sum()) == len(c["s_ms"]) == 1084   # s < 120 s not counted
    assert rej == {key: int((c["reason"] == code).sum()) for code, key in enumerate(ec.REJECT_KEYS, start=1)}


def test_end_range_needs_the_end_inside_the_frames_and_before_the_game_end():
    frames = np.arange(0, 20 * 60_000 + 1, 60_000)          # last frame 20 min, game end 30 min
    name = _names(_candidates(1_080_000, 60_000, [], NO_SPANS, frames=frames, gold=np.zeros(len(frames))))
    assert name[1_140_000] != "end_range"                   # end 1,200 s = the last frame
    assert name[1_141_000] == "end_range"                   # end 1,201 s, after the last frame
    name = _names(_candidates(1_681_000, 59_000, [], NO_SPANS))   # frames to 40 min, game end 1,800 s
    assert name[1_740_000] != "end_range"                   # end 1,799 s
    assert name[1_741_000] == "end_range"                   # end 1,800 s = the game end


def test_a_window_differing_only_in_frame_boundaries_is_rejected():
    # frames every 60 s with one 10 s step (600 -> 610 s); tau = 300 s, D = 55 s: no frame in (300, 355] s, end age
    # 55 s.  The start 600 s holds the 610 s frame (n_bound 1) and still has end age 45 s (bin 2) and every other
    # feature of the engagement.
    frames = np.r_[np.arange(0, 600_001, 60_000), np.arange(610_000, 40 * 60_000, 60_000)]
    gold = np.zeros(len(frames))
    c = _candidates(300_000, 55_000, [], NO_SPANS, frames=frames, gold=gold)
    at = int(np.flatnonzero(c["s_ms"] == 600_000)[0])
    assert int(c["keys"]["n_bound"][at]) == 1 and int(c["eng_keys"]["n_bound"][0]) == 0
    assert all(int(c["keys"][name][at]) == int(c["eng_keys"][name][0]) for name in ef.FEATURES if name != "n_bound")
    assert _names(c)[600_000] == "keys_differ"


def test_feature_bins_at_their_edges():
    frames = np.array([0, 60_000, 120_000, 180_000, 244_424, 300_000, 900_000, 1_500_000, 2_000_000])
    gold = np.arange(len(frames), dtype=np.float64)
    edges = np.array([[1.0, 2.0, 3.0, 4.0]] * 3)
    # start ages 0, 5.000, 5.001, 10.000, 10.001 s
    keys = ef.window_keys(np.array([120_000, 125_000, 125_001, 130_000, 130_001]), np.full(5, 20_000), frames,
                          np.array([], dtype=np.int64), gold, edges)
    assert keys["start_age_bin"].tolist() == [0, 0, 1, 1, -1]
    # end ages 19.999, 20.000, 39.999, 40.000 s, and 64.423 s inside a frame gap above 60 s (180 -> 244.424 s)
    starts = np.array([120_000, 120_000, 120_000, 120_000, 180_000])
    keys = ef.window_keys(starts, np.array([19_999, 20_000, 39_999, 40_000, 64_423]), frames,
                          np.array([], dtype=np.int64), gold, edges)
    assert keys["end_age_bin"].tolist() == [0, 1, 1, 2, 2] and keys["n_bound"].tolist() == [0, 0, 0, 0, 0]
    # n_bound counts frames f with start < f <= start + D
    keys = ef.window_keys(np.array([120_000, 119_999]), np.array([60_000, 60_001]), frames, np.array([0]), gold, edges)
    assert keys["n_bound"].tolist() == [1, 2]
    # time bins: < 15 min, [15, 25) min, >= 25 min
    assert ef.time_bin(np.array([899_999, 900_000, 1_499_999, 1_500_000])).tolist() == [0, 1, 1, 2]
    # (e) kill gaps at the five-bin edges 30, 60, 90 and 180 s; no earlier kill; a kill exactly at the start is not earlier
    kills = np.array([100_000])
    gaps = [29_999, 30_000, 59_999, 60_000, 89_999, 90_000, 179_999, 180_000]
    values = ef.window_values(np.array([100_000 + g for g in gaps] + [50_000, 100_000]), np.full(10, 10_000), frames,
                              kills, gold)
    assert values["kill_gap_ms"].tolist() == gaps + [-1, -1]
    assert ef.kill_gap_bin(values["kill_gap_ms"]).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 4]
    # gold at F(start); a value equal to an edge goes to the upper bin (side="right")
    keys = ef.window_keys(np.array([120_000, 180_000, 300_000, 305_000]), np.full(4, 10_000), frames,
                          np.array([], dtype=np.int64), gold, edges)
    assert keys["gold_bin"].tolist() == [2, 3, 4, 4]       # gold 2, 3, 5, 5 against edges 1, 2, 3, 4


def test_gold_edges_are_linear_quantiles_of_the_rows_per_time_bin():
    gold = np.array([0.0, 10.0, 20.0, 30.0, 40.0, -5.0, 5.0, 100.0, 7.0, 8.0])
    tbin = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])
    edges = ef.gold_edges(gold, tbin)
    assert edges.shape == (3, 4) and edges.dtype == np.float64
    assert edges[0].tolist() == [8.0, 16.0, 24.0, 32.0]     # positions 0.8, 1.6, 2.4, 3.2 of 0..40 step 10
    for b in range(3):
        np.testing.assert_array_equal(edges[b], np.quantile(gold[tbin == b], [0.2, 0.4, 0.6, 0.8], method="linear"))
    with pytest.raises(ValueError):
        ef.gold_edges(gold[tbin < 2], tbin[tbin < 2])       # a time bin without rows
    with pytest.raises(ValueError):
        ef.gold_edges(np.r_[gold, np.nan], np.r_[tbin, 0])
    assert ef.team_gold_difference(np.array([[2500, 2400], [3000, 3100.5]], np.float32)).tolist() == [100.0, -100.5]
    assert ef.gold_bin(np.array([8.0, 7.999, 32.0, 50.0]), np.zeros(4, np.int64), edges).tolist() == [1, 0, 4, 4]


def test_selection_hash_is_sha256_of_the_ascii_key_and_the_integer_start():
    key = b"95ad19d3:4f2a"
    assert ec.selection_digest(key, np.int64(603_000)) == hashlib.sha256(b"95ad19d3:4f2a:603000").hexdigest()
    assert ec.selection_digest(np.bytes_(key), 603_000) == ec.selection_digest(key, 603_000)
    with pytest.raises(UnicodeDecodeError):
        ec.selection_digest(b"\xff", 1)
    c = _candidates(603_000, 40_000, [613_000], [[603_000, 643_000]])
    survivors = c["s_ms"][c["reason"] == 0].tolist()
    expected = min(survivors, key=lambda s: hashlib.sha256(f"95ad19d3:4f2a:{s}".encode("utf-8")).hexdigest())
    s, _ = _run(_eng(key, 603_000, 40_000), np.array([613_000]), np.array([[603_000, 643_000]]))
    assert s[0] == expected


def test_several_engagements_of_one_match_equal_separate_calls():
    kills, spans = np.array([613_000, 1_330_000]), np.array([[603_000, 643_000], [1_320_000, 1_420_000]])
    eng = {"engagement_key": np.array([b"m:a", b"m:b"]), "tau_ms": np.array([603_000, 1_320_000]),
           "D_ms": np.array([40_000, 100_000])}
    s, rej = _run(eng, kills, spans)
    one = [_run(_eng(eng["engagement_key"][i], int(eng["tau_ms"][i]), int(eng["D_ms"][i])), kills, spans) for i in range(2)]
    assert s.tolist() == [one[0][0][0], one[1][0][0]]
    assert rej == {key: one[0][1][key] + one[1][1][key] for key in ec.REJECT_KEYS}


def test_a_q_row_needs_exactly_one_matching_bank_component():
    # (c) PRESPEC_E1 section 7, row-table key mismatch: stop unless each q row's component_key is in the match's bank
    # component table exactly once with f - 10 s = tau, e_h90 = tau + D and valid_h90 1
    with pytest.raises(ec.Blocked, match="not once"):
        _scene_spans(q_keys=(b"E", b"X"))                                          # absent
    with pytest.raises(ec.Blocked, match="not once"):
        _scene_spans(rows=COMPONENTS + (COMPONENTS[3],))                            # twice
    with pytest.raises(ec.Blocked, match="tau"):
        _scene_spans(taus=(1_200_001, 1_290_000))
    with pytest.raises(ec.Blocked, match="tau"):
        _scene_spans(Ds=(89_998, 100_000))
    with pytest.raises(ec.Blocked, match="valid_h90"):
        _scene_spans(rows=COMPONENTS[:3] + ((b"E", 1_210_000, 1_210_000, 1_289_999, 0),) + COMPONENTS[4:])
    with pytest.raises(ValueError):
        ec.component_spans(np.array([100_000]), np.array([89_999]))   # an outcome window cannot end before it opens
    assert ec.overlaps_any(np.array([[90_000, 150_000]]), np.array([150_000, 150_001, 0]),
                           np.array([160_000, 170_000, 90_000])).tolist() == [True, False, True]


def _reference(key, tau, D, kills, spans, frames, game_end, gold, edges):
    """PRESPEC_E1 section 3 written out per candidate with plain Python (an independent restatement)."""

    def held(t):
        return max((i for i, f in enumerate(frames) if f <= t), default=-1)

    def features(start):
        i = held(start)
        age = start - frames[i] if i >= 0 else None
        sab = -1 if age is None or age > 10_000 else (0 if age <= 5_000 else 1)
        nb = sum(1 for f in frames if start < f <= start + D)
        j = held(start + D)
        ea = start + D - frames[j] if j >= 0 else None
        eab = -1 if ea is None else (0 if ea < 20_000 else (1 if ea < 40_000 else 2))
        tb = 0 if start / 60_000 < 15 else (1 if start / 60_000 < 25 else 2)
        before = [k for k in kills if k < start]
        gap = start - max(before) if before else None
        kb = 4 if gap is None else sum(1 for edge in (30_000, 60_000, 90_000, 180_000) if gap >= edge)
        gb = sum(1 for e in edges[tb] if e <= gold[i]) if i >= 0 else -1
        return (sab, nb, eab, tb, kb, gb)

    own = features(tau)
    reasons = {}
    for k in range(-600, 601):
        s = tau + 1_000 * k
        if s < 120_000:
            continue
        f = features(s)
        if f[0] == -1:
            reasons[s] = 1
        elif s + D > frames[-1] or s + D >= game_end:
            reasons[s] = 2
        elif any(s <= t <= s + D for t in kills):
            reasons[s] = 3
        elif any(s <= e and a <= s + D for a, e in spans):
            reasons[s] = 4
        elif f != own:
            reasons[s] = 5
        else:
            reasons[s] = 0
    survivors = [s for s, r in reasons.items() if r == 0]
    chosen = min(survivors, key=lambda s: hashlib.sha256(f"{key}:{s}".encode("utf-8")).hexdigest()) if survivors else -1
    return reasons, chosen, own


@pytest.mark.parametrize("seed", range(6))
def test_rules_equal_a_plain_restatement_on_random_matches(seed):
    # (f) five kill-gap bins; the outcome windows are those of the match's q engagements only, built from a bank-style
    # component table that also holds other components (valid or not) whose windows must not exclude anything
    rng = np.random.default_rng(seed)
    steps = 60_000 + rng.integers(0, 80, size=34)
    steps[rng.integers(5, 30)] = 64_424                     # one frame gap above 60 s
    frames = np.r_[0, np.cumsum(steps)].astype(np.int64)
    game_end = int(frames[-1]) + int(rng.integers(0, 2))     # the last frame sits at (or 1 ms before) the game end
    taus, Ds = [], []
    while len(taus) < 4:
        i = int(rng.integers(2, len(frames) - 4))
        tau, D = int(frames[i] + rng.integers(0, 10_001)), int(rng.integers(10_000, 150_000))
        if tau + D < min(frames[-1], game_end):
            taus.append(tau)
            Ds.append(D)
    other = rng.choice(np.arange(60_000, game_end, 1_000), size=int(rng.integers(8, 40)), replace=False)
    kills = np.unique(np.r_[other, np.array(taus) + 10_000]).astype(np.int64)    # each q engagement's first kill
    f = other[rng.choice(len(other), size=min(10, len(other)), replace=False)]
    rows = [(f"q{j}".encode("ascii"), taus[j] + 10_000, taus[j] + 10_000, taus[j] + Ds[j], 1) for j in range(4)]
    rows += [(f"n{i}".encode("ascii"), int(v), int(v), int(v) + int(rng.integers(0, 90_001)), int(rng.integers(0, 2)))
             for i, v in enumerate(f)]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    table_keys, table_values = _component_table(rows)
    spans = ec.q_engagement_spans(table_keys, table_values, np.array([f"q{j}".encode("ascii") for j in range(4)], dtype="S64"),
                                  np.array(taus), np.array(Ds))
    q_windows = [(taus[j], taus[j] + Ds[j]) for j in range(4)]
    assert spans.tolist() == [list(w) for w in q_windows]
    gold = rng.integers(-6_000, 6_000, size=len(frames)).astype(np.float64)
    edges = np.sort(rng.integers(-3_000, 3_000, size=(3, 4)), axis=1).astype(np.float64)
    keys = np.array([f"m{seed}:c{j}".encode("ascii") for j in range(4)])
    c = ec.evaluate_engagements(np.array(taus), np.array(Ds), kills, spans, frames, game_end, gold, edges)
    s, rej = match_windows({"engagement_key": keys, "tau_ms": np.array(taus), "D_ms": np.array(Ds)}, kills, spans,
                           frames, game_end, gold, edges)
    total = dict.fromkeys(ec.REJECT_KEYS, 0)
    for j in range(4):
        reasons, chosen, own = _reference(keys[j].decode("ascii"), taus[j], Ds[j], kills.tolist(), q_windows,
                                          frames.tolist(), game_end, gold.tolist(), edges.tolist())
        mine = c["owner"] == j
        assert dict(zip(c["s_ms"][mine].tolist(), c["reason"][mine].tolist())) == reasons
        assert tuple(int(c["eng_keys"][name][j]) for name in ef.FEATURES) == own
        assert int(s[j]) == chosen
        for r in reasons.values():
            if r:
                total[ec.REJECT_KEYS[r - 1]] += 1
    assert rej == total


def test_window_summaries():
    match = np.array([1, 1, 1, 2, 2, 3])
    s = np.array([500_000, 500_000, 530_000, 700_000, -1, 900_000])
    D = np.array([40_000, 35_000, 40_000, 40_000, 40_000, 30_000])
    reuse = ec.reuse_summary(match, s)
    assert reuse["distribution"] == {"1": 3, "2": 1} and reuse["rows_in_reused_windows"] == 2
    assert reuse["distinct_windows"] == 4 and reuse["rows"] == 5
    overlap = ec.overlap_summary(match, s, D)               # [500, 540] and [500, 535] both overlap [530, 570]
    assert overlap["rows_overlapping_another_start"] == 3 and overlap["rows"] == 5
    eng, plc = np.array([100.0, 200.0, 300.0]), np.array([110.0, 190.0, 330.0])
    pooled = np.sqrt((eng.var() + plc.var()) / 2)
    assert ec.smd(eng, plc) == pytest.approx((plc.mean() - eng.mean()) / pooled)
    assert ec.gate_passed(0.30) and not ec.gate_passed(0.2999)


def test_role_summary_by_hand():
    n_cand = np.array([1084, 1084, 1201, 1201, 1201, 1201], dtype=np.int32)
    n_surv = np.array([3, 0, 5, 2, 0, 1], dtype=np.int32)
    table = {"match_index": np.array([1, 1, 2, 2, 3, 3]), "s_ms": np.array([500_000, -1, 700_000, 700_000, -1, 900_000]),
             "D_ms": np.full(6, 40_000), "start_age_bin": np.array([0, 0, 1, 1, 0, 1]), "n_bound": np.array([0, 0, 0, 0, 1, 1]),
             "end_age_bin": np.array([2, 2, 2, 2, 1, 1]), "time_bin": np.array([0, 0, 0, 0, 1, 1]),
             "kill_gap_bin": np.array([2, 0, 1, 1, 0, 2]), "gold_bin": np.array([0, 1, 2, 3, 4, 0]),
             "n_candidates": n_cand, "n_survivors": n_surv,
             "reject_counts": np.stack([n_cand - n_surv] + [np.zeros(6, np.int32)] * 4, axis=1),
             "kill_gap_eng_ms": np.array([-1, 50_000, 100_000, 120_000, 30_000, -1]),
             "kill_gap_plc_ms": np.array([300_000, -1, 110_000, 150_000, -1, 400_000]),
             "xy_finite_plc": np.array([1, -1, 1, 0, -1, 1], dtype=np.int8),
             "meets_component_window_plc": np.array([1, -1, 0, 1, -1, 1], dtype=np.int8)}
    s = ec.role_summary(table, np.ones(6, dtype=bool))
    assert s["component_windows"]["matched_windows_meeting_a_component_outcome_window"] == 3
    assert s["component_windows"]["share"] == 0.75
    assert (s["engagements"], s["matched"], s["matches"], s["matched_matches"]) == (6, 4, 3, 3)
    assert s["match_rate"] == pytest.approx(4 / 6)
    assert s["bins"]["kill_gap_bin"] == {"0": {"engagements": 2, "matched": 0, "match_rate": 0.0},
                                         "1": {"engagements": 2, "matched": 2, "match_rate": 1.0},
                                         "2": {"engagements": 2, "matched": 2, "match_rate": 1.0}}
    by_gap = s["matched_vs_unmatched_by_kill_gap_bin"]
    assert sorted(by_gap) == ["0", "1", "2", "3", "4"] and by_gap["4"]["engagements"] == 0
    assert by_gap["0"]["share_of_unmatched"] == 1.0 and by_gap["1"]["share_of_matched"] == 0.5
    balance = s["kill_gap_balance"]
    assert (balance["pairs"], balance["pairs_dropped_without_earlier_kill"]) == (2, 2)
    assert (balance["mean_engagement_s"], balance["mean_placebo_s"]) == (110.0, 130.0)
    assert balance["smd"] == pytest.approx(20 / np.sqrt(250))
    assert s["reuse"]["distribution"] == {"1": 2, "2": 1} and s["xy"]["nonfinite_f_s_xy"] == 1
    cand = s["candidates"]
    assert (cand["evaluated"], cand["grid_points_dropped_before_matching_s_below_120s"], cand["survivors"]) == (6972, 234, 11)
    assert cand["rejected"]["start_age_out"] == 6961 and cand["rejected"]["keys_differ"] == 0
    assert cand["survivors_per_matched_engagement"] == {"min": 1, "median": 2.5, "mean": 2.75, "max": 5}


def _summary_rows(roles, s_ms):
    n = len(roles)
    table = {"role": np.asarray(roles, dtype=np.int8), "match_index": np.arange(n), "s_ms": np.asarray(s_ms),
             "D_ms": np.full(n, 30_000), "n_candidates": np.full(n, 1201, np.int32), "n_survivors": np.ones(n, np.int32),
             "reject_counts": np.zeros((n, 5), np.int32), "kill_gap_eng_ms": np.full(n, 40_000),
             "kill_gap_plc_ms": np.full(n, 41_000), "xy_finite_plc": np.ones(n, np.int8),
             "meets_component_window_plc": np.ones(n, np.int8), "engagement_key": np.array([f"k{i}".encode() for i in range(n)])}
    table.update({name: np.zeros(n, np.int64) for name in ef.FEATURES})
    return table


def test_test_rows_go_only_to_the_sealed_summary():
    table = _summary_rows([0, 0, 3, 3, 3, 4, 5, 5, 5, 5], [1, -1, 1, 1, -1, 1, 1, -1, -1, -1])
    public, sealed = ec.role_summaries(table)
    assert set(public) == {"TRAIN", "Q_DEV", "Q_CAL"}
    assert [public[r]["engagements"] for r in ("TRAIN", "Q_DEV", "Q_CAL")] == [2, 3, 1]
    assert (sealed["engagements"], sealed["matched"]) == (4, 1)
    no_test = _summary_rows([0, 3, 4], [1, 1, -1])
    assert ec.role_summaries(no_test)[1] is None


def test_final_q_dev_windows_are_compared_with_the_gate_by_key():
    table = _summary_rows([0, 3, 3], [5_000, 7_000, -1])
    gate = {name: np.asarray(value)[[2, 1]] for name, value in table.items()}      # Q_DEV rows, another order
    assert ec.windows_equal(gate, table) == (True, True)
    table["s_ms"] = np.array([5_000, 8_000, -1])
    assert ec.windows_equal(gate, table) == (True, False)
    table["s_ms"] = np.array([5_000, 7_000, -1])
    table["gold_bin"] = np.array([0, 0, 3])
    assert ec.windows_equal(gate, table) == (True, False)
    gate["engagement_key"] = np.array([b"k2", b"zz"])
    assert ec.windows_equal(gate, table)[0] is False


def test_appendix_edges_are_read_from_the_prespec_table():
    text = ("# x\n\n## 부록 A. 골드 차 경계 (Task 13이 채움, 창 생성 전)\n\n"
            "| time_bin | TRAIN 행 | 20% | 40% | 60% | 80% |\n|---|---|---|---|---|---|\n"
            "| 0 (< 15분) | 10 | -1500.0 | -500.5 | 400.0 | 1400.25 |\n"
            "| 1 (15–25분) | 11 | -2.0 | -1.0 | 1.0 | 2.0 |\n"
            "| 2 (≥ 25분) | 12 | -3.0 | -1.5 | 1.5 | 3.0 |\n")
    edges = ec.read_appendix_edges(text)
    assert edges.tolist() == [[-1500.0, -500.5, 400.0, 1400.25], [-2.0, -1.0, 1.0, 2.0], [-3.0, -1.5, 1.5, 3.0]]
    with pytest.raises(ValueError):
        ec.read_appendix_edges("## 부록 A. 골드 차 경계 (Task 13이 채움, 창 생성 전)\n\n(비어 있음)\n")
