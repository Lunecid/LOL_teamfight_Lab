"""Tests of ``c_ept.checks.online_onsets.absorbed_share`` (S4b preparation M15, decision B15): the share of the engagement onsets decided
online that a later kill merges into another unit. Synthetic kills only, written by hand in seconds and map units; nothing is read.

The rule is the thesis connection rule (C-9): two kills are linked when ``|dt| <= G_ms`` (15 s) and the distance is ``<= D`` (2,000 u).
"""
import itertools

import numpy as np
import pytest

from c_ept.checks import online_onsets
from c_ept.checks.online_onsets import absorbed_share

S = 1000                      # ms in a second


def run(kills, **kw):
    """``absorbed_share`` of ``kills`` = [(t in s, x, y), ...]."""
    t = np.array([k[0] * S for k in kills], dtype=np.int64)
    xy = np.array([[k[1], k[2]] for k in kills], dtype=np.float64).reshape(-1, 2)
    return absorbed_share(t, xy, **kw)


def counts(got):
    return got['n_kills'], got['n_onsets'], got['n_units'], got['n_absorbed']


# ---- the review example A1-F4: a bridge kill merges an engagement that had already started ------------------------------------------------
def test_absorbed_share_on_the_review_example():
    """A open at the origin (t = 0). B at 10 s and 3,000 u: inside 15 s of A but outside 2,000 u, so a new unit, and its onset is decided
    online at 10 s. The bridge C at 14 s and 1,500 u from both links to A (14 s, 1,500 u) and to B (4 s, 1,500 u): the two units become
    one and B, whose onset was decided first as a separate engagement, is absorbed. 3 kills, 2 onsets decided online, 1 unit at the
    end, 1 absorbed, share 1/2."""
    got = run([(0, 0, 0), (10, 3000, 0), (14, 1500, 0)])
    assert counts(got) == (3, 2, 1, 1)
    assert got['absorbed_share'] == 0.5 and got['G_ms'] == 15_000 and got['D'] == 2_000


def test_no_bridge_means_zero_absorption():
    """Without the bridge no unit merges: C too late for A and B (30 s), C too far from A and B, and a kill that only extends one unit."""
    assert counts(run([(0, 0, 0), (10, 3000, 0), (30, 1500, 0)])) == (3, 3, 3, 0)         # C is 30 s from A and 20 s from B: its own onset
    assert counts(run([(0, 0, 0), (10, 3000, 0), (14, 1500, 3000)])) == (3, 3, 3, 0)       # C is 3,354 u from both
    assert counts(run([(0, 0, 0), (10, 3000, 0), (12, 4000, 0)])) == (3, 2, 2, 0)          # C extends B's unit only
    got = run([(0, 0, 0), (10, 3000, 0), (30, 1500, 0)])
    assert got['absorbed_share'] == 0.0
    assert counts(run([(0, 0, 0), (5, 500, 0), (9, 1000, 0)])) == (3, 1, 1, 0)             # one chain of one unit: one onset, nothing absorbed
    assert run([(0, 0, 0), (5, 500, 0), (9, 1000, 0)])['absorbed_share'] == 0.0


def test_a_kill_between_the_onsets_in_time_is_not_a_bridge_and_a_late_one_that_reaches_only_one_unit_is_not_either():
    """Only a kill that links to two different units merges them. Here the units of A (0 s) and B (10 s) are far apart; D at 25 s is
    15 s from B and 25 s from A, so it joins the unit of B and the unit of A stays closed in time: nothing is absorbed. A kill C that
    comes at 9.9 s, before B, links to A and B then links to C: B is not an onset at all (it has an earlier neighbour), so there is
    one onset and nothing is absorbed either."""
    assert counts(run([(0, 0, 0), (10, 3000, 0), (25, 1500, 0)])) == (3, 2, 2, 0)
    assert counts(run([(0, 0, 0), (10, 3000, 0), (9, 1500, 0)])) == (3, 1, 1, 0)


def test_three_units_merged_by_two_bridges_absorb_two_onsets():
    """A (0 s, 0 u), B (3 s, 3,000 u), C (6 s, 6,000 u): three units. D (9 s, 1,500 u) bridges A and B; E (12 s, 4,500 u) bridges B and C
    (and not D, 3,000 u away). 5 kills, 3 onsets, 1 unit, 2 absorbed (B at D, C at E), share 2/3."""
    got = run([(0, 0, 0), (3, 3000, 0), (6, 6000, 0), (9, 1500, 0), (12, 4500, 0)])
    assert counts(got) == (5, 3, 1, 2) and got['absorbed_share'] == pytest.approx(2 / 3)


def test_one_kill_that_links_three_units_absorbs_two_onsets():
    """A, B, C lie on a circle of radius 1,900 u around the origin, more than 3,000 u apart, so they are three units; the kill D at the
    centre (5 s later) is within 2,000 u of all three: m = 3, two onsets are absorbed at once (m - 1, not 1)."""
    got = run([(0, 1900, 0), (1, -950, 1645), (2, -950, -1645), (5, 0, 0)])
    assert counts(got) == (4, 3, 1, 2) and got['absorbed_share'] == pytest.approx(2 / 3)


# ---- the boundaries: equal to G, equal to D, equal times ----------------------------------------------------------------------------------
def test_a_gap_of_exactly_G_links_and_one_ms_more_does_not():
    """The rule is ``|dt| <= G_ms``. C is within reach of A only (B is near C in time too); the bridge needs the link C-A."""
    def case(c_ms):
        t = np.array([0, 10 * S, c_ms], dtype=np.int64)
        xy = np.array([[0, 0], [3000, 0], [1500, 0]], dtype=np.float64)
        return counts(absorbed_share(t, xy))
    assert case(15_000) == (3, 2, 1, 1)                  # |C - A| = 15,000 ms = G: linked, B is absorbed
    assert case(15_001) == (3, 2, 2, 0)                  # 15,001 ms: C links to B only
    assert case(14_999) == (3, 2, 1, 1)
    t, xy = np.array([0, 10 * S, 15_000]), np.array([[0, 0], [3000, 0], [1500, 0]], dtype=float)
    assert counts(absorbed_share(t, xy, G_ms=14_999)) == (3, 2, 2, 0)                       # the threshold is the argument
    assert counts(absorbed_share(t, xy, G_ms=15_000)) == (3, 2, 1, 1)


def test_a_distance_of_exactly_D_links_and_a_little_more_does_not():
    """The rule is ``distance <= D``: A at the origin, B at 4,000 u, the bridge C 2,000 u from both (a 3-4-5 triangle keeps the squares exact)."""
    assert counts(run([(0, 0, 0), (5, 4000, 0), (10, 2000, 0)])) == (3, 2, 1, 1)           # 2,000 u from A and from B: linked to both
    assert counts(run([(0, 0, 0), (5, 4001, 0), (10, 2000, 0)])) == (3, 2, 2, 0)           # 2,001 u from B: only A
    assert counts(run([(0, 0, 0), (5, 3200, 2400), (10, 1600, 1200)])) == (3, 2, 1, 1)     # 2,000 u from both along a diagonal: exact squares
    assert counts(run([(0, 0, 0), (5, 4000, 0), (10, 2000, 0)], D=1999)) == (3, 3, 3, 0)   # the threshold is the argument
    assert counts(run([(0, 0, 0), (5, 4000, 0), (10, 2000, 0)], D=2000)) == (3, 2, 1, 1)


def test_kills_at_the_same_ms_are_one_batch_and_never_absorb_each_other():
    """I1 of the M15 review. A model at one ms sees all kills of that ms together, so there is no online order among them and a kill of the
    ms cannot absorb an onset decided at the same ms. Three kills at 7 s at (0, 0), (100, 3,000) and (200, 1,500): the first two are
    3,000 u apart and both within 1,500 u of the third, so they are one component with no earlier kill: one onset, nothing absorbed
    (the old order by (x, y) gave 2 onsets and 1 absorbed). The result holds in every order of the list."""
    kills = [(7, 0, 0), (7, 100, 3000), (7, 200, 1500)]
    for order in itertools.permutations(range(3)):
        got = run([kills[i] for i in order])
        assert counts(got) == (3, 1, 1, 0) and got['absorbed_share'] == 0.0, order
    assert counts(run([(7, 0, 0), (7, 0, 0)])) == (2, 1, 1, 0)                              # two kills at one point and one time: one unit


def test_a_batch_that_joins_one_earlier_unit_is_not_an_absorption_whatever_its_coordinates():
    """The second example of the review: an open unit at the origin (0 s); at 5 s P (1,900, 0) links to it and Q (1,800, 1,900) links only
    to P (1,903 u). Q is no onset: P and Q are one component that joins one earlier unit, so one onset, nothing absorbed (the old order
    made Q an onset first and let P absorb it)."""
    kills = [(0, 0, 0), (5, 1900, 0), (5, 1800, 1900)]
    for order in itertools.permutations(range(3)):
        assert counts(run([kills[i] for i in order])) == (3, 1, 1, 0), order


def test_a_batch_component_that_joins_two_earlier_units_absorbs_one_onset():
    """Two earlier units A (0 s, origin) and B (1 s, 3,000 u) are two onsets. At 5 s the batch P (1,500, 0) and R (1,500, 1,000) is one
    component (the two are 1,000 u apart) that links A and B through P: it joins two units and absorbs one onset. 4 kills, 2 onsets,
    1 unit, 1 absorbed."""
    kills = [(0, 0, 0), (1, 3000, 0), (5, 1500, 0), (5, 1500, 1000)]
    for order in itertools.permutations(range(4)):
        assert counts(run([kills[i] for i in order])) == (4, 2, 1, 1), order


def random_match(rng):
    """A random match of up to 40 kills: times on a 5 s grid (many kills share a ms), positions in a 6,000 u square."""
    n = int(rng.integers(0, 41))
    t = rng.integers(0, 9, size=n) * 5 * S
    xy = rng.uniform(0, 6000, size=(n, 2))
    return t.astype(np.int64), xy


def test_the_result_does_not_depend_on_the_order_of_the_kills_of_one_ms():
    """Property test, seeded: 400 random matches with many kills at the same ms; the whole result is the same for the listed order and for
    5 random shuffles of the kills (a shuffle also reorders the kills of one ms among themselves)."""
    rng = np.random.default_rng(20261026)
    for _ in range(400):
        t, xy = random_match(rng)
        want = absorbed_share(t, xy)
        for _ in range(5):
            p = rng.permutation(len(t))
            assert absorbed_share(t[p], xy[p]) == want


def test_the_final_unit_count_is_the_number_of_components_of_the_offline_assembly():
    """m5 of the review: ``n_units`` equals ``len(units.link_components(...))`` (the offline connected components of the thesis rule, the
    kills in time order) on 400 seeded random matches, for the default thresholds and for others; onsets and absorbed satisfy
    ``n_onsets - n_absorbed == n_units``."""
    from c_ept.domain import units
    rng = np.random.default_rng(7)
    for k in range(400):
        t, xy = random_match(rng)
        g, d = ((15_000, 2_000), (5_000, 1_000), (0, 2_500), (20_000, 3_000))[k % 4]
        got = absorbed_share(t, xy, G_ms=g, D=d)
        order = np.argsort(t, kind='stable')
        want = len(units.link_components(t[order], xy[order, 0], xy[order, 1], gap_ms=g, link_u=d))
        assert got['n_units'] == want and got['n_onsets'] - got['n_absorbed'] == want
        assert got['n_onsets'] >= want and (got['n_kills'] == 0 or want >= 1)


def test_the_input_is_not_changed_and_the_result_does_not_depend_on_the_order_of_kills_at_different_times():
    t = np.array([14 * S, 0, 10 * S], dtype=np.int64)
    xy = np.array([[1500, 0], [0, 0], [3000, 0]], dtype=np.float64)
    t0, xy0 = t.copy(), xy.copy()
    got = absorbed_share(t, xy)
    assert counts(got) == (3, 2, 1, 1) and np.array_equal(t, t0) and np.array_equal(xy, xy0)
    assert got == absorbed_share(t[::-1].copy(), xy[::-1].copy())


# ---- empty input, types, validation ----------------------------------------------------------------------------------------------------------
def test_no_kill_has_no_share_and_one_kill_has_none_absorbed():
    got = run([])
    assert counts(got) == (0, 0, 0, 0) and got['absorbed_share'] is None                  # a share of nothing is not 0
    got = run([(3, 10, 10)])
    assert counts(got) == (1, 1, 1, 0) and got['absorbed_share'] == 0.0


def test_the_result_holds_plain_python_numbers():
    got = run([(0, 0, 0), (10, 3000, 0), (14, 1500, 0)])
    assert all(type(got[k]) is int for k in ('n_kills', 'n_onsets', 'n_units', 'n_absorbed', 'G_ms', 'D')) and type(got['absorbed_share']) is float
    assert counts(got) == (3, 2, 1, 1) and got['n_onsets'] - got['n_units'] == got['n_absorbed']       # the identity of the count
    float_times = absorbed_share(np.array([0.0, 10_000.0, 14_000.0]), np.array([[0, 0], [3000, 0], [1500, 0]], dtype=np.float32))
    assert counts(float_times) == (3, 2, 1, 1)                                             # float times and float32 positions are accepted


@pytest.mark.parametrize('t, xy, kw, text', [
    (np.zeros((2, 1)), np.zeros((2, 2)), {}, 'one-dimensional'),
    (np.zeros(3), np.zeros((2, 2)), {}, 'one row'),
    (np.zeros(2), np.zeros((2, 3)), {}, 'shape'),
    (np.zeros(2), np.zeros(2), {}, 'shape'),
    (np.array([0.0, np.nan]), np.zeros((2, 2)), {}, 'finite'),
    (np.zeros(2), np.array([[0.0, 0.0], [np.inf, 0.0]]), {}, 'finite'),
    (np.array([True, False]), np.zeros((2, 2)), {}, 'bool'),
    (np.array(['a', 'b']), np.zeros((2, 2)), {}, 'numeric'),
    (np.zeros(2), np.zeros((2, 2)), {'G_ms': -1}, 'G_ms'),
    (np.zeros(2), np.zeros((2, 2)), {'D': -0.5}, r'\bD\b'),
    (np.zeros(2), np.zeros((2, 2)), {'G_ms': float('nan')}, 'G_ms'),
    (np.zeros(2), np.zeros((2, 2)), {'D': float('inf')}, r'\bD\b'),
    (np.zeros(2), np.zeros((2, 2)), {'G_ms': True}, 'G_ms'),
    (np.zeros(2), np.zeros((2, 2)), {'D': 'far'}, r'\bD\b'),
])
def test_bad_input_is_refused_with_a_message_that_holds_no_value(t, xy, kw, text):
    """The inputs are arrays the caller read: wrong shape, a length that differs, NaN or infinity (a kill without a position is the
    caller's to drop, it is not guessed here), a bool or text array, a negative or non-finite threshold. The message names the argument
    and the problem, not the numbers (they come from matches)."""
    with pytest.raises((ValueError, TypeError), match=text):
        absorbed_share(t, xy, **kw)


def test_the_module_imports_numpy_only_and_reads_no_data():
    """The function takes arrays and returns a dict: the module imports no reader, store, cohort or config (nothing of the data)."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(online_onsets.__file__).read_text(encoding='utf-8'))
    imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or '').split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {'__future__', 'numpy'}, imported
