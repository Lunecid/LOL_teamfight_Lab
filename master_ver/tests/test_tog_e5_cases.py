"""E5 case selection (PLAN_E1_E5_KO.md Task 5, PRESPEC_E5_KO.md section 3, SDD rulings 25 and 32).

The first two tests are the plan's, verbatim, on the plan's table T with the column ka_kills added (equal to n_kills,
Ruling 32), so their expectations stay.
"""
import hashlib

import numpy as np
import pytest
from studies.tog_experiments_20261012.e5_cases import select_cases

T = {"engagement_key": np.array(["k1", "k2", "k3", "k4", "k5", "k6"]),
     "p_q": np.array([0.85, 0.9, 0.5, 0.15, 0.6, 0.7]),
     "p_pt": np.array([0.6, 0.6, 0.1, 0.5, 0.6, 0.65]),
     "y": np.array([0, 0, 1, 1, 1, 1]),
     "svi": np.array([-0.05, -0.08, 0.02, 0.03, 0.01, 0.04]),
     "n_kills": np.array([2, 3, 1, 1, 2, 1]),
     "ka_kills": np.array([2, 3, 1, 1, 2, 1]),
     "same_frame": np.array([0, 0, 1, 1, 0, 1]),
     "objective": np.array([0, 0, 0, 0, 0, 0]),
     "kill_side_blue": np.array([0, 1, 0, 0, 1, 0])}

def test_rules_pick_expected_keys():
    out = select_cases(T)
    assert out["a"] == "k2"   # |0.9 - 0| = 0.9 >= 0.8, 3 kills, largest |svi|
    assert out["b"] == "k3"   # |0.5 - 1| < |0.1 - 1| and |0.5 - 0.1| = 0.4 is the largest gap
    assert out["c"] == "k6"   # single kill by red, same frame, svi > 0 (blue up), largest |svi| among k3, k4, k6

def test_ties_break_by_key_hash():
    t = {k: v.copy() for k, v in T.items()}
    t["svi"][1] = -0.05; t["n_kills"][0] = 3          # k1 and k2 now tie under rule (a)
    import hashlib
    expect = min(["k1", "k2"], key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    assert select_cases(t)["a"] == expect


def _copy(table):
    return {k: v.copy() for k, v in table.items()}


def test_cases_are_distinct():
    # Ruling 25.  With p_pt of k3 at 0.6 only k6 meets rule (b) (|0.7 - 1| = 0.3 < |0.65 - 1| = 0.35), so (b) = k6.
    # Rule (c) would take k6 again (largest |svi| 0.04 among k3, k4, k6); without k6 it takes k4 (0.03 > 0.02 of k3).
    t = _copy(T)
    t["p_pt"][2] = 0.6
    out = select_cases(t)
    assert [out[r] for r in "abc"] == ["k2", "k6", "k4"]


def test_rule_a_reports_its_threshold():
    assert select_cases(T)["a_threshold"] == 0.8


def test_rule_a_threshold_is_inclusive():
    t = _copy(T)
    t["p_q"][1] = 0.8              # k2: |0.8 - 0| = 0.8 exactly; with a strict ">" (a) would fall to k1
    out = select_cases(t)
    assert out["a"] == "k2" and out["a_threshold"] == 0.8


@pytest.mark.parametrize("k5_p_q, k5_svi, k3_kills, a, threshold", [
    (0.25, 0.01, 2, "k5", 0.7),    # k5 (|0.25 - 1| = 0.75) is alone at 0.7; k3 (0.65, larger |svi| 0.02) waits for 0.6
    (0.45, 0.09, 2, "k3", 0.6),    # none at 0.7; k3 at 0.6; k5 (0.55, larger |svi| 0.09) stays below the last threshold
    (0.45, 0.09, 1, None, None),   # none at 0.6 either: no case (a)
])
def test_rule_a_lowers_its_threshold(k5_p_q, k5_svi, k3_kills, a, threshold):
    # k1 and k2 drop to one kill (k4 has one), so no 2-kill engagement has |p_q - y| >= 0.8.
    # k3 gets |0.35 - 1| = 0.65 with k3_kills kills; k5 has two kills.
    t = _copy(T)
    t["n_kills"][[0, 1]] = 1
    t["p_q"][2] = 0.35; t["n_kills"][2] = k3_kills
    t["p_q"][4] = k5_p_q; t["svi"][4] = k5_svi
    out = select_cases(t)
    assert out["a"] == a and out["a_threshold"] == threshold
    if a is None:                  # no case (a) takes nothing away from the later rules
        assert [out["b"], out["c"]] == ["k3", "k6"]


@pytest.mark.parametrize("as_bytes", [False, True])
def test_tie_hashes_the_ascii_key_text(as_bytes):
    # k4 joins rule (a) (two kills, |0.15 - 1| = 0.85) and ties k2 at |svi| 0.08.  The smaller hash is k2's, which comes
    # before k4 in the table (in the plan's tie test the winner comes after), and hashing str(b"k2") would pick k4.
    t = _copy(T)
    t["n_kills"][3] = 2; t["svi"][3] = 0.08
    expect = min(["k2", "k4"], key=lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
    if as_bytes:                   # real engagement keys are byte strings; the case keeps the table's key type
        t["engagement_key"] = np.array([k.encode("ascii") for k in T["engagement_key"].tolist()])
        expect = expect.encode("ascii")
    out = select_cases(t)
    assert out["a"] == expect and type(out["a"]) is type(expect)


def test_rule_without_a_remaining_candidate_gives_none():
    # (b)'s only candidate is (a)'s case: k2 meets (b) (|0.9 - 0| < |0.95 - 0|); k3 and k6 now have p_pt as far from
    # y as p_q, which is not closer.  (c) is unaffected.
    t = _copy(T)
    t["p_pt"][[1, 2, 5]] = [0.95, 0.5, 0.7]
    out = select_cases(t)
    assert [out[r] for r in "abc"] == ["k2", None, "k6"]
    # (c)'s only candidate is (b)'s case: (b) = k6 as in test_cases_are_distinct, and k3, k4 leave the same frame.
    t = _copy(T)
    t["p_pt"][2] = 0.6
    t["same_frame"][[2, 3]] = 0
    out = select_cases(t)
    assert [out[r] for r in "abc"] == ["k2", "k6", None]


def test_each_rule_counts_its_own_kills():
    # Ruling 32: rule (a) reads n_kills (eval_n_kills), rule (c) reads ka_kills (KA blue_kills + red_kills).
    t = _copy(T)
    t["n_kills"][1] = 1            # k2: one kill by eval_n_kills, still three team-credited kills
    t["ka_kills"][5] = 2           # k6: two team-credited kills, still one by eval_n_kills
    out = select_cases(t)
    assert out["a"] == "k1"        # k2 leaves (a); k1 (|0.85 - 0|, two kills) remains
    assert out["c"] == "k4"        # k6 leaves (c); k4 (|svi| 0.03) beats k3 (0.02)


@pytest.mark.parametrize("change, c", [
    ({}, "k5"),                                                  # blue kill, svi < 0 (V moved to red): a violation
    ({"svi": {4: 0.06}}, "k6"),                                  # blue kill, svi > 0
    ({"kill_side_blue": {4: 0}}, "k6"),                          # red kill, svi < 0
    ({"objective": {4: 1}}, "k6"),                               # objective counts changed
    ({"same_frame": {4: 0}}, "k6"),                              # F_pre_ms != F_post_ms
    ({"ka_kills": {4: 2}}, "k6"),                                # two team-credited kills
    ({"svi": {4: 0.0}, "same_frame": {2: 0, 3: 0, 5: 0}}, None),  # svi = 0 is no violation; k3, k4, k6 left out
])
def test_rule_c_conditions(change, c):
    # k5 becomes a same-frame single blue kill with svi = -0.06, the largest |svi| among the (c) candidates; each change
    # removes it and (c) falls back to k6 (|svi| 0.04 > 0.03 of k4; k3 is (b)'s case).
    t = _copy(T)
    t["same_frame"][4] = 1; t["ka_kills"][4] = 1; t["svi"][4] = -0.06
    for name, cells in change.items():
        for row, value in cells.items():
            t[name][row] = value
    assert select_cases(t)["c"] == c


@pytest.mark.parametrize("name, values, error", [
    ("engagement_key", np.array(["k1", "k2", "k3", "k4", "k5", "k1"]), ValueError),   # duplicate key
    ("engagement_key", np.array([1, 2, 3, 4, 5, 6]), TypeError),                       # neither str nor bytes
    ("svi", np.array([-0.05, np.nan, 0.02, 0.03, 0.01, 0.04]), ValueError),           # not finite
    ("y", np.array([0, 0, 1, 1, 1, 2]), ValueError),                                  # not 0 or 1
    ("p_q", np.array([0.85, 0.9, 0.5, 0.15, 0.6, 1.5]), ValueError),                  # not a probability
    ("objective", np.array([0, 0, 0]), ValueError),                                   # one value per engagement
])
def test_malformed_table_is_refused(name, values, error):
    t = _copy(T)
    t[name] = values
    with pytest.raises(error):
        select_cases(t)


def test_table_without_ka_kills_is_refused():
    t = _copy(T)
    del t["ka_kills"]
    with pytest.raises(KeyError):
        select_cases(t)


# ---------------------------------------------------------------- Ruling 33: the Task 5 review additions

def _changed(table, change):
    t = _copy(table)
    for name, cells in change.items():
        for row, value in cells.items():
            t[name][row] = value
    return t


def test_rule_c_drops_rule_a_case():
    # The (a)/(c) overlap behind Ruling 25.  k2 stays (a) (three kills by eval_n_kills, |0.9 - 0| = 0.9) and becomes the
    # top (c) candidate (one KA kill, by blue, same frame, svi -0.08 against blue).  (c) drops it together with (b)'s k3
    # and takes k6 (|svi| 0.04 > 0.03 of k4); a rule that dropped only the previous rule's case would give k2 again.
    out = select_cases(_changed(T, {"same_frame": {1: 1}, "ka_kills": {1: 1}}))
    assert [out[r] for r in "abc"] == ["k2", "k3", "k6"] and out["a_threshold"] == 0.8


@pytest.mark.parametrize("change, b, tied, c", [
    ({"svi": {3: 0.04}}, "k3", ("k4", "k6"), "k6"),                    # k4 ties k6 at |svi| 0.04; the later row wins
    ({"p_pt": {2: 0.5}, "svi": {2: 0.03}}, "k6", ("k3", "k4"), "k3"),  # k3 leaves (b) (0.5 is not < 0.5), so (b) = k6;
])                                                                     # k3 ties k4 at 0.03; the earlier row wins
def test_rule_c_tie_breaks_by_key_hash(change, b, tied, c):
    # Same tie rule in (c) as in (a): the smallest SHA-256 of the key text, whichever row comes first.
    assert c == min(tied, key=lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
    out = select_cases(_changed(T, change))
    assert [out[r] for r in "abc"] == ["k2", b, c]


def test_rule_b_drops_rule_a_case_with_the_largest_gap():
    # k2, chosen by (a), is also the (b) candidate with the largest gap: |0.9 - 0| < |1.0 - 0| and |0.9 - 1.0| = 0.1.
    # k3's gap shrinks to |0.5 - 0.45| = 0.05 (still closer: 0.5 < 0.55) and k6 leaves (b) (0.3 is not < 0.3).
    # (b) skips k2 and takes k3; (c) takes k6 as in the plan's test.
    out = select_cases(_changed(T, {"p_pt": {1: 1.0, 2: 0.45, 5: 0.7}}))
    assert [out[r] for r in "abc"] == ["k2", "k3", "k6"]


@pytest.mark.parametrize("change", [
    {},                                             # the plan's T
    {"svi": {1: -0.05}, "n_kills": {0: 3}},         # the plan's (a) tie
    {"same_frame": {1: 1}, "ka_kills": {1: 1}},     # the (a)/(c) overlap
    {"svi": {3: 0.04}},                             # (c) tie won by the later row
    {"p_pt": {2: 0.5}, "svi": {2: 0.03}},           # (c) tie won by the earlier row
    {"p_pt": {1: 1.0, 2: 0.45, 5: 0.7}},            # (b) skips (a)'s case
])
def test_row_order_does_not_matter(change):
    # Every column reversed: the same three cases and the same threshold.
    t = _changed(T, change)
    assert select_cases({k: v[::-1].copy() for k, v in t.items()}) == select_cases(t)
