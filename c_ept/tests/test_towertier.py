"""Tests of ``c_ept.towertier``: the tower state before t0, the tier of a taken tower and its expected price (stage S3 Task 8, S1-5).

No run folder, raw file or catalog is read. Every S1 store here is written by this file with ``store_core.Store`` under a pytest folder
(ruling R24: the toy data of this task stay local to the test file). A store holds the arrays ``towertier`` may read, plus
``matches/key`` with sentinel text that no code may read, and is wrapped in a ``SpyStore`` where a test needs to know what was read.
Truth comes from six places:

    brief tests    the three tests of the task brief: probabilities add up to 1, ``destroyed_before`` does not count the event on the
                   t0 millisecond (``test_destroyed_counts_use_only_events_before_t0``), the expected price is sum p x tier price
    hand store     five small matches written out by hand, with the state of every team and tier at chosen times worked out below them
    oracles        plain loops over the events, written from the text of the rules, that the code is compared with on random stores
    lawful world   toy matches in which the towers of a team fall in the fixed order outer, outer, outer, inner, ..., nexus, so the tier
                   is a function of the destroyed counts and a fit has to find it
    closed form    models with chosen parameters, whose probabilities are worked out by hand (softmax, the standardisation, the mixing of
                   ruling R13), and scikit-learn fitted on the same standardised design for the fit itself
    rulings        R13 (every categorical output is (1 - 1e-6) p + 1e-6 / K over the full class set), R20 (tolerance 1e-8, an honest
                   convergence flag), R24 (the files this task may touch) and the global constraints (E6 through ``check``, holdout roles
                   refused before any read, the state before t0 from ``destroyed_before`` alone, no match key anywhere)
"""
import ast
import hashlib
import json
import math
import warnings
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from c_ept import config as C
from c_ept import flowprice, towertier
from c_ept.store_core import Store
from c_ept.tests.conftest import all_imports, module_level_imports

SOURCE = Path(towertier.__file__)
MIX = 1e-6
BLUE, RED = 0, 1                                                      # objectives/taker: the team that took the tower
OUTER, INNER, BASE, NEXUS = 0, 1, 2, 3                                # objectives/sub of a tower
KIND = {name: i for i, name in enumerate(C.O_KINDS)}
TOWER = KIND['tower']
KEY_PREFIX = 'TOYKEY'                                                  # the sentinel text of matches/key: it must never show up
LAW = (OUTER,) * 3 + (INNER,) * 3 + (BASE,) * 3 + (NEXUS,) * 2         # the towers of a team fall in this order of tiers
ARRAYS_READ = {'matches/obj_off', 'matches/role', 'objectives/t', 'objectives/kind', 'objectives/sub', 'objectives/taker'}
PRICES = {'dragon': 410.0, 'elder': 1520.0, 'grubs': 260.0, 'herald': 630.0, 'baron': 1830.0, 'atakhan': 710.0, 'tower_outer': 520.0,
          'tower_inner': 610.0, 'tower_base': 400.0, 'tower_nexus': 900.0, 'inhibitor': 505.0, 'plate': 125.0}   # tower prices not monotone


# =====================================================================================================================================
# a toy S1 store written here: the arrays of the matches and the objectives, and the frame-clock rows
# =====================================================================================================================================
class Toy(NamedTuple):
    """One toy match: the role, its objectives ``(t ms, kind, sub, taker)`` in store order, and the t0 of its frame-clock rows."""
    role: str
    objectives: tuple = ()
    t0s: tuple = ()


def tower(t, tier, taker):
    return int(t), TOWER, int(tier), int(taker)


def other(t, name, taker):
    """An objective that is not a tower: ``name`` is a kind of ``config.O_KINDS`` or 'unknown' (kind -1, another monster or building)."""
    return int(t), (-1 if name == 'unknown' else KIND[name]), -1, int(taker)


def _put(store, name, values, dtype):
    values = np.asarray(values, dtype=dtype)
    a = store.allocate(name, values.shape, dtype)
    a[...] = values
    a.flush()
    del a


def build_store(root, toys) -> Store:
    """The S1 store of ``toys`` under ``root``: matches/{key, role, T_ms, obj_off}, objectives/{t, kind, sub, taker} and
    frame_rows/{match, t0}, in the dtypes of ``run_s1.SPEC``."""
    toys = list(toys)
    flat = [e for toy in toys for e in toy.objectives]
    rows = [(m, t0) for m, toy in enumerate(toys) for t0 in toy.t0s]
    s = Store(root)
    _put(s, 'matches/key', [f'{KEY_PREFIX}_{m:04d}'.encode('ascii') for m in range(len(toys))], 'S24')
    _put(s, 'matches/role', [C.ROLES.index(toy.role) for toy in toys], 'i1')
    _put(s, 'matches/T_ms', [3_000_000] * len(toys), 'i8')
    _put(s, 'matches/obj_off', np.concatenate([[0], np.cumsum([len(toy.objectives) for toy in toys])]), 'i8')
    for name, column, dtype in (('t', 0, 'i8'), ('kind', 1, 'i1'), ('sub', 2, 'i1'), ('taker', 3, 'i1')):
        _put(s, f'objectives/{name}', [e[column] for e in flat], dtype)
    _put(s, 'frame_rows/match', [m for m, _ in rows], 'i4')
    _put(s, 'frame_rows/t0', [t0 for _, t0 in rows], 'i8')
    s.finalize()
    return s


class SpyStore:
    """A store that records the name of every array read, in order, in ``reads`` (a list that can be shared with other recorders)."""

    def __init__(self, store, log=None):
        self._store = store
        self.reads = [] if log is None else log

    def array(self, name):
        self.reads.append(name)
        return self._store.array(name)


class Untouchable:
    """A store that fails on any use: proof that a refusal came before the first read."""

    def __getattr__(self, name):
        raise AssertionError(f'the store was used ({name})')


# =====================================================================================================================================
# the hand store
# =====================================================================================================================================
# Teams are 0 blue and 1 red. A tower taken by red is lost by blue (the victim is 1 - taker). The tiers are 0 outer, 1 inner, 2 base,
# 3 nexus. The state of a team is its destroyed counts [outer, inner, base, nexus], and the state at t counts the events with time < t.
HAND = (
    Toy('FIT', (
        tower(100_000, OUTER, BLUE),             # red loses an outer tower
        tower(200_000, OUTER, RED),              # blue loses an outer tower
        tower(200_000, INNER, BLUE),             # red loses an inner tower, in the same ms as the line above
        other(300_000, 'baron', BLUE),           # not a tower
        tower(350_000, -1, BLUE),                # a tower whose tier the file does not name: not counted
        tower(360_000, INNER, -1),               # a tower whose taker the file does not name: not counted
        other(370_000, 'unknown', BLUE),         # another monster or building (kind -1)
        tower(380_000, -1, -1),                  # a tower with neither a tier nor a taker: counted once, as one without a tier
        tower(400_000, BASE, RED),               # blue loses a base tower
        tower(500_000, NEXUS, BLUE),             # red loses a nexus tower
        other(500_000, 'inhibitor', BLUE),       # not a tower, in the same ms as the line above
        other(600_000, 'plate', RED),            # not a tower
    ), (120_300, 180_300, 240_300, 420_300, 520_300)),
    Toy('FIT', (
        tower(150_000, OUTER, RED),              # blue loses three outer towers and then an inner one, all taken by red
        tower(250_000, OUTER, RED),
        tower(350_000, OUTER, RED),
        tower(450_000, INNER, RED),
    ), (200_300, 400_300)),
    Toy('STOP', (tower(99_000, NEXUS, RED),), (120_300,)),
    Toy('DEV', (), (120_300, 180_300)),
    Toy('CAL', (tower(10_000, OUTER, BLUE), other(20_000, 'dragon', RED)), ()),
)
Z4 = [0, 0, 0, 0]
# (match, t_ms, state of blue [outer, inner, base, nexus], state of red) worked out from the lists above
HAND_STATES = (
    (0, 0, Z4, Z4),
    (0, 100_000, Z4, Z4),                        # the outer tower of red falls at exactly this ms: not counted
    (0, 100_001, Z4, [1, 0, 0, 0]),
    (0, 200_000, Z4, [1, 0, 0, 0]),              # nor are the two events at exactly 200,000
    (0, 200_001, [1, 0, 0, 0], [1, 1, 0, 0]),
    (0, 400_000, [1, 0, 0, 0], [1, 1, 0, 0]),    # the four events from 350,000 to 380,000 are never counted; the one at 400,000 is not yet
    (0, 400_001, [1, 0, 1, 0], [1, 1, 0, 0]),
    (0, 500_000, [1, 0, 1, 0], [1, 1, 0, 0]),
    (0, 500_001, [1, 0, 1, 0], [1, 1, 0, 1]),
    (0, 10_000_000, [1, 0, 1, 0], [1, 1, 0, 1]),
    (1, 150_000, Z4, Z4),
    (1, 150_001, [1, 0, 0, 0], Z4),
    (1, 350_001, [3, 0, 0, 0], Z4),
    (1, 450_000, [3, 0, 0, 0], Z4),
    (1, 450_001, [3, 1, 0, 0], Z4),
    (2, 99_000, Z4, Z4),
    (2, 99_001, [0, 0, 0, 1], Z4),
    (3, 0, Z4, Z4),
    (3, 10_000_000, Z4, Z4),                     # a match without any objective
    (4, 10_000, Z4, Z4),
    (4, 10_001, Z4, [1, 0, 0, 0]),
)


@pytest.fixture(scope='module')
def hand(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp('hand'), HAND)


def hand_query():
    match = np.array([c[0] for c in HAND_STATES])
    t_ms = np.array([c[1] for c in HAND_STATES])
    want = np.array([[c[2], c[3]] for c in HAND_STATES])
    return match, t_ms, want


# =====================================================================================================================================
# destroyed_before
# =====================================================================================================================================
def test_destroyed_counts_use_only_events_before_t0(hand):
    """The brief's test: an event on the same millisecond as t0 is not counted, one millisecond earlier it is."""
    for match, t_ms, blue, red in HAND_STATES:
        got = towertier.destroyed_before(hand, [match], [t_ms])
        assert got.shape == (1, 2, 4), (match, t_ms)
        assert got[0].tolist() == [blue, red], (match, t_ms)
    exact = towertier.destroyed_before(hand, [0, 0], [100_000, 100_001])
    assert exact[0, RED, OUTER] == 0 and exact[1, RED, OUTER] == 1


def test_all_the_hand_queries_in_one_call_in_any_order(hand):
    match, t_ms, want = hand_query()
    got = towertier.destroyed_before(hand, match, t_ms)
    assert got.shape == want.shape == (len(HAND_STATES), 2, 4) and got.dtype.kind == 'i' and np.array_equal(got, want)
    order = np.random.default_rng(3).permutation(len(match))                                  # the queries of a match are not together
    assert np.array_equal(towertier.destroyed_before(hand, match[order], t_ms[order]), want[order])
    twice = np.concatenate([np.arange(len(match)), np.arange(len(match))])                    # and a query may be asked more than once
    assert np.array_equal(towertier.destroyed_before(hand, match[twice], t_ms[twice]), want[twice])


def test_the_victim_is_the_team_that_did_not_take_the_tower(tmp_path):
    s = build_store(tmp_path / 'one', [Toy('FIT', (tower(10, INNER, BLUE), tower(20, BASE, RED)))])
    got = towertier.destroyed_before(s, [0], [1_000])[0]
    assert got[RED].tolist() == [0, 1, 0, 0]                  # blue took the inner tower: red lost it
    assert got[BLUE].tolist() == [0, 0, 1, 0]                 # red took the base tower: blue lost it


def test_what_is_not_a_tower_with_a_tier_and_a_taker_is_not_counted(tmp_path):
    objectives = [other(10, name, taker) for name in C.O_KINDS if name != 'tower' for taker in (BLUE, RED)]
    objectives += [(20, -1, 2, BLUE),                                   # kind -1 with a tier column
                   (30, len(C.O_KINDS), 1, BLUE),                       # a kind that is not in the config
                   tower(40, -1, BLUE), tower(50, 4, BLUE),             # a tower without a tier, a tier that does not exist
                   tower(60, OUTER, -1), tower(70, OUTER, 2),           # a tower without a taker, a taker that is neither team
                   (80, KIND['inhibitor'], 2, BLUE)]                    # an inhibitor with a tier column is still an inhibitor
    s = build_store(tmp_path / 'junk', [Toy('FIT', tuple(objectives))])
    assert not towertier.destroyed_before(s, [0], [10_000_000]).any()


def test_each_match_is_counted_apart_and_a_match_without_objectives_is_zero(hand):
    both = towertier.destroyed_before(hand, [0, 2, 3], [10_000_000, 10_000_000, 10_000_000])
    assert both[0].tolist() == [[1, 0, 1, 0], [1, 1, 0, 1]] and both[1].tolist() == [[0, 0, 0, 1], Z4] and not both[2].any()
    empty = towertier.destroyed_before(hand, [], [])
    assert empty.shape == (0, 2, 4) and empty.dtype.kind == 'i'


def test_the_state_at_every_frame_row_of_the_store(hand):
    """How the composer asks: the match and the t0 of every frame-clock row (``frame_rows/match`` and ``frame_rows/t0``)."""
    match, t0 = np.asarray(hand.array('frame_rows/match')), np.asarray(hand.array('frame_rows/t0'))
    got = towertier.destroyed_before(hand, match, t0)
    want = [oracle_destroyed(HAND, m, t) for m, t in zip(match.tolist(), t0.tolist())]
    assert got.shape == (len(match), 2, 4) and len(match) == 10 and np.array_equal(got, np.array(want))
    assert got[0, RED].tolist() == [1, 0, 0, 0] and got[4, RED].tolist() == [1, 1, 0, 1] and got[4, BLUE].tolist() == [1, 0, 1, 0]


def test_the_state_never_goes_down_as_time_goes_on(hand):
    t = np.arange(0, 700_000, 997)
    for match in range(len(HAND)):
        got = towertier.destroyed_before(hand, np.full(len(t), match), t)
        assert (np.diff(got, axis=0) >= 0).all(), match
    assert towertier.destroyed_before(hand, [0], [699_999]).sum() == 5       # the five towers of match 0 that have a tier and a taker


def test_events_after_t0_do_not_change_the_state(tmp_path):
    """Two stores with the same past and different futures give the same state at t0 (nothing after t0 enters, rule A3)."""
    t0 = 300_000
    past = [tower(50_000, OUTER, BLUE), tower(120_000, OUTER, RED), other(200_000, 'dragon', BLUE), tower(299_999, INNER, BLUE)]
    futures = [[], [tower(t0, OUTER, RED), tower(t0 + 1, INNER, RED), tower(900_000, NEXUS, BLUE)],
               [tower(t0, BASE, BLUE), tower(t0, NEXUS, RED), other(t0 + 5, 'baron', RED)],
               [tower(t0 + k, k % 4, k % 2) for k in range(1, 40)]]
    results = []
    for i, future in enumerate(futures):
        s = build_store(tmp_path / f'future{i}', [Toy('FIT', tuple(past + future))])
        results.append(towertier.destroyed_before(s, [0], [t0])[0])
    assert results[0].tolist() == [[1, 0, 0, 0], [1, 1, 0, 0]]               # blue lost an outer tower (120,000), red an outer and an inner
    assert all(np.array_equal(results[0], r) for r in results[1:])


def test_destroyed_before_reads_only_the_arrays_it_needs(hand):
    spy = SpyStore(hand)
    towertier.destroyed_before(spy, [0, 1, 3], [150_000, 300_000, 999])
    assert spy.reads and set(spy.reads) <= {'matches/obj_off', 'objectives/t', 'objectives/kind', 'objectives/sub', 'objectives/taker'}
    assert 'matches/key' not in spy.reads


def test_the_order_of_the_events_in_a_match_does_not_matter(tmp_path):
    rng = np.random.default_rng(11)
    objectives = [tower(int(t), int(rng.integers(0, 4)), int(rng.integers(0, 2))) for t in rng.integers(0, 30, 40) * 1_000]
    shuffled = [objectives[i] for i in rng.permutation(len(objectives))]
    a = build_store(tmp_path / 'a', [Toy('FIT', tuple(sorted(objectives)))])
    b = build_store(tmp_path / 'b', [Toy('FIT', tuple(shuffled))])
    t = np.arange(0, 32_000, 500)
    assert np.array_equal(towertier.destroyed_before(a, np.zeros(len(t), int), t), towertier.destroyed_before(b, np.zeros(len(t), int), t))


def test_a_store_whose_offsets_do_not_cut_the_objective_table_is_refused(tmp_path):
    def store(name, off, off_dtype='i8', n_roles=1):
        s = Store(tmp_path / name)
        _put(s, 'matches/role', [0] * n_roles, 'i1')
        _put(s, 'matches/obj_off', off, off_dtype)
        for column, dtype in (('t', 'i8'), ('kind', 'i1'), ('sub', 'i1'), ('taker', 'i1')):
            _put(s, f'objectives/{column}', [0, 0], dtype)                       # two objective rows
        s.finalize()
        return s
    bad = (([0, 3], 'i8'),           # ends beyond the rows
           ([1, 2], 'i8'),           # does not start at 0
           ([0, 2, 1, 2], 'i8'),     # goes down
           ([[0, 2]], 'i8'),         # 2-d
           ([0.0, 2.0], 'f8'),       # not integers
           ([], 'i8'))               # no offsets at all
    for i, (off, dtype) in enumerate(bad):
        with pytest.raises(ValueError) as raised:
            towertier.destroyed_before(store(f'bad{i}', off, dtype), [0], [0])
        assert 'obj_off' in str(raised.value)
    assert towertier.destroyed_before(store('good', [0, 2]), [0], [0]).shape == (1, 2, 4)
    with pytest.raises(ValueError, match='matches/role'):
        towertier.tier_rows(store('roles', [0, 1, 2]), ('FIT',))                 # the offsets cut two matches, the role array has one


def test_destroyed_before_refuses_what_it_cannot_count(hand):
    for match, t_ms in (([5], [0]), ([-1], [0]), ([0, 1], [0]), ([[0]], [[0]]), ([0], [-1])):
        with pytest.raises(ValueError) as raised:
            towertier.destroyed_before(hand, match, t_ms)
        assert KEY_PREFIX not in str(raised.value)
    for match, t_ms in (([0.0], [0]), ([0], [0.5]), ([True], [0]), (['0'], [0])):
        with pytest.raises(TypeError):
            towertier.destroyed_before(hand, match, t_ms)


# ---- the plain-loop oracle, and random stores ---------------------------------------------------------------------------------------
def oracle_destroyed(toys, m, t_ms):
    """(2, 4) counts of the towers with a tier and a taker lost by each team strictly before t_ms in toy match m, by a plain loop."""
    out = np.zeros((2, 4), dtype=np.int64)
    for t, kind, sub, taker in toys[m].objectives:
        if kind == TOWER and 0 <= sub <= 3 and taker in (BLUE, RED) and t < t_ms:
            out[1 - taker, sub] += 1
    return out


def random_toys(seed, n_matches=10, ordered=False):
    rng = np.random.default_rng(seed)
    roles = ('FIT', 'STOP', 'DEV', 'CAL', 'AUDIT', 'UNUSED')
    toys = []
    for _ in range(n_matches):
        objectives = []
        for _ in range(int(rng.integers(0, 40))):
            if rng.random() < 0.6:
                kind, sub = TOWER, int(rng.integers(-1, 4))
            else:
                kind, sub = int(rng.integers(-1, len(C.O_KINDS))), int(rng.integers(-1, 4))     # the tier is carried by any kind
            taker = int(rng.choice((-1, BLUE, RED), p=(0.1, 0.45, 0.45)))
            objectives.append((int(rng.integers(0, 40)) * 15_000, kind, sub, taker))            # a coarse grid: many ties
        if ordered:
            objectives.sort(key=lambda e: e[0])
        toys.append(Toy(roles[int(rng.integers(0, len(roles)))], tuple(objectives)))
    return toys


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('ordered', [False, True])
def test_destroyed_before_agrees_with_a_plain_loop_on_random_stores(tmp_path, seed, ordered):
    toys = random_toys(seed, ordered=ordered)
    s = build_store(tmp_path / 'random', toys)
    rng = np.random.default_rng(100 + seed)
    match, t_ms = [], []
    for m, toy in enumerate(toys):
        times = {0, 10_000_000}
        for t, *_ in toy.objectives:
            times.update((t - 1, t, t + 1))                              # the two sides of every event and the event itself
        times.update(int(v) for v in rng.integers(0, 700_000, 6))
        for t in sorted(times):
            match.append(m)
            t_ms.append(max(t, 0))
    order = rng.permutation(len(match))
    match, t_ms = np.array(match)[order], np.array(t_ms)[order]
    got = towertier.destroyed_before(s, match, t_ms)
    want = np.array([oracle_destroyed(toys, m, t) for m, t in zip(match.tolist(), t_ms.tolist())])
    assert got.shape == want.shape and len(match) > 100 and np.array_equal(got, want)


# =====================================================================================================================================
# victim_counts and tier_rows
# =====================================================================================================================================
def test_victim_counts_pick_the_team_that_lost_the_tower():
    D = np.arange(3 * 2 * 4).reshape(3, 2, 4)
    got = towertier.victim_counts(D, [BLUE, RED, BLUE])
    assert got.tolist() == [D[0, RED].tolist(), D[1, BLUE].tolist(), D[2, RED].tolist()]       # blue took it: red is the victim
    for taker in ([0, 1], [0, 1, 2], [0, 1, -1], [[0], [1], [0]]):
        with pytest.raises(ValueError):
            towertier.victim_counts(D, taker)
    with pytest.raises(ValueError):
        towertier.victim_counts(D[:, :, :3], [0, 1, 0])


HAND_FIT_ROWS = (        # (match, t ms, taker, tier, destroyed counts of the victim before the event) of the FIT matches of the hand store
    (0, 100_000, BLUE, OUTER, [0, 0, 0, 0]),     # red is the victim
    (0, 200_000, RED, OUTER, [0, 0, 0, 0]),      # blue is the victim; its own state, not that of red
    (0, 200_000, BLUE, INNER, [1, 0, 0, 0]),     # red lost an outer tower at 100,000
    (0, 400_000, RED, BASE, [1, 0, 0, 0]),       # blue lost an outer tower at 200,000
    (0, 500_000, BLUE, NEXUS, [1, 1, 0, 0]),     # red lost an outer and an inner tower
    (1, 150_000, RED, OUTER, [0, 0, 0, 0]),
    (1, 250_000, RED, OUTER, [1, 0, 0, 0]),
    (1, 350_000, RED, OUTER, [2, 0, 0, 0]),
    (1, 450_000, RED, INNER, [3, 0, 0, 0]),
)


def test_tier_rows_of_the_hand_store(hand):
    rows = towertier.tier_rows(hand, ('FIT',))
    assert rows.match.tolist() == [r[0] for r in HAND_FIT_ROWS] and rows.t_ms.tolist() == [r[1] for r in HAND_FIT_ROWS]
    assert rows.taker.tolist() == [r[2] for r in HAND_FIT_ROWS] and rows.tier.tolist() == [r[3] for r in HAND_FIT_ROWS]
    assert rows.destroyed.tolist() == [r[4] for r in HAND_FIT_ROWS]
    # 8 + 4 tower events in the two FIT matches; two have no tier (350,000, and 380,000 which has no taker either), one has a tier but
    # no taker (360,000): each left-out event is counted once
    assert (rows.n_matches, rows.n_tower_events, rows.n_no_tier, rows.n_no_taker) == (2, 12, 2, 1)
    assert rows.destroyed.shape == (9, 4)


def test_tier_rows_take_the_matches_of_the_asked_roles_only(hand):
    assert towertier.tier_rows(hand, 'STOP').match.tolist() == [2] and towertier.tier_rows(hand, ('STOP',)).tier.tolist() == [NEXUS]
    assert len(towertier.tier_rows(hand, ('FIT', 'STOP')).tier) == 10 and len(towertier.tier_rows(hand, ('DEV',)).tier) == 0
    dev = towertier.tier_rows(hand, ('DEV',))
    assert dev.n_matches == 1 and dev.n_tower_events == 0 and dev.destroyed.shape == (0, 4)


def test_tier_rows_refuse_a_holdout_role_and_a_role_s3_does_not_use_before_any_read():
    for roles in ('TEST', ('EXTERNAL',), ('FIT', 'TEST'), ('CAL',), ('AUDIT',), ('UNUSED',), ('NOT_A_ROLE',), ()):
        with pytest.raises(ValueError):
            towertier.tier_rows(Untouchable(), roles)


# =====================================================================================================================================
# the lawful world: the towers of a team fall in the order of LAW, so the tier follows from the destroyed counts
# =====================================================================================================================================
def lawful_toys(seed, n_matches, role='FIT', law=LAW):
    rng = np.random.default_rng(seed)
    toys = []
    for _ in range(n_matches):
        events = []
        for victim in (BLUE, RED):
            k = int(rng.integers(0, len(law) + 1))
            times = np.sort(rng.choice(np.arange(10, 240), size=k, replace=False)) * 10_000    # distinct times inside a team
            events += [(int(t), victim, law[j]) for j, t in enumerate(times)]
        events.sort()
        objectives = [tower(t, tier, 1 - victim) for t, victim, tier in events]
        for _ in range(int(rng.integers(0, 4))):                                               # objectives that are not towers
            objectives.append(other(int(rng.integers(10, 240)) * 10_000, str(rng.choice(('dragon', 'baron', 'plate', 'inhibitor'))),
                                    int(rng.integers(0, 2))))
        objectives.sort(key=lambda e: e[0])
        toys.append(Toy(role, tuple(objectives)))
    return toys


def lawful_state(j, law=LAW):
    """The destroyed counts of a team after j towers have fallen in the order of ``law``."""
    counts = [0, 0, 0, 0]
    for tier in law[:j]:
        counts[tier] += 1
    return counts


# FIT follows LAW; STOP and DEV follow the opposite order, so a fit that read them would get another answer
LAWFUL = lawful_toys(1, 300) + lawful_toys(2, 40, 'STOP', LAW[::-1]) + lawful_toys(3, 40, 'DEV', LAW[::-1])


@pytest.fixture(scope='module')
def lawful(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp('lawful'), LAWFUL)


@pytest.fixture(scope='module')
def lawful_fit_only(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp('lawful_fit_only'), LAWFUL[:300])


@pytest.fixture(scope='module')
def model(lawful):
    return towertier.TierModel.fit(lawful, roles=('FIT',))


@pytest.fixture(scope='module')
def small(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp('small'), lawful_toys(5, 40))


def test_probabilities_sum_to_one(model):
    """The brief's test: a probability for each of the four tiers, adding up to 1, on a grid of states, takers and times."""
    rng = np.random.default_rng(0)
    n = 5_000
    D = rng.integers(0, 4, size=(n, 4))
    D[:, 3] = np.minimum(D[:, 3], 2)
    taker, t_s = rng.integers(0, 2, n), rng.uniform(0, 4_000, n)
    p = model.prob(D, taker, t_s)
    assert p.shape == (n, 4) and p.dtype == np.float64 and np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert (p >= MIX / 4 - 1e-18).all() and (p <= 1.0).all()                  # ruling R13: no tier has less than 1e-6 / 4
    far = model.prob(np.full((3, 4), 1e6), [0, 1, 1], [1e9, 0.0, 5.0])        # far outside the training range: no overflow
    assert np.isfinite(far).all() and np.allclose(far.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_the_tier_follows_the_destroyed_towers_of_the_victim(model):
    for j, tier in enumerate(LAW):                                            # the eleven states a team can be in before a tower falls
        for taker in (BLUE, RED):
            for minute in (12.0, 30.0):
                p = model.prob([lawful_state(j)], [taker], [minute * 60.0])[0]
                assert int(p.argmax()) == tier and p[tier] > 0.9, (j, taker, minute, p.round(4).tolist())


def test_a_victim_with_every_outer_tower_standing_gets_no_base_or_nexus_tier(model):
    p = model.prob(np.zeros((6, 4)), [0, 1] * 3, [300.0, 300.0, 900.0, 900.0, 2_400.0, 2_400.0])
    assert (p[:, OUTER] > 0.99).all() and (p[:, BASE] + p[:, NEXUS] < 1e-3).all()
    full = model.prob([[3, 3, 3, 1]], [BLUE], [1_800.0])[0]                   # one nexus tower is down and the other one is next
    assert full[NEXUS] > 0.99 and full[OUTER] + full[INNER] < 1e-3


def test_the_destroyed_counts_of_a_training_row_are_those_of_the_victim_team(lawful_fit_only):
    """In the lawful world only the victim's own destroyed towers say what falls next: the tier of every row is a function of the victim's
    counts, which pins the orientation of the training rows (the counts of the taker would be independent of the tier)."""
    rows = towertier.tier_rows(lawful_fit_only, ('FIT',))
    states = np.array([lawful_state(j) for j in range(len(LAW))])
    victim_total = rows.destroyed.sum(axis=1)
    assert np.array_equal(rows.tier, np.array(LAW)[victim_total])             # the tier is a function of the victim's counts
    assert np.array_equal(rows.destroyed, states[victim_total])


def test_fit_uses_only_the_asked_roles(lawful, lawful_fit_only, model):
    same = towertier.TierModel.fit(lawful_fit_only, roles=('FIT',))           # the same FIT matches in the same order, no other role
    for name in ('classes_', 'coef_std_', 'intercept_std_', 'mean_', 'scale_'):
        assert np.array_equal(getattr(same, name), getattr(model, name)), name
    dev = towertier.TierModel.fit(lawful, roles=('DEV',))                     # the opposite order: the first tower to fall is a nexus
    assert dev.prob([lawful_state(0)], [BLUE], [900.0])[0, NEXUS] > 0.9 and model.prob([lawful_state(0)], [BLUE], [900.0])[0, NEXUS] < 1e-3
    both = towertier.TierModel.fit(lawful, roles=('FIT', 'DEV'))
    assert not np.array_equal(both.coef_std_, model.coef_std_) and both.summary()['fit_roles'] == ['FIT', 'DEV']
    assert model.summary()['fit_roles'] == ['FIT'] and model.summary()['n_matches'] == 300


def test_fit_is_scikit_learn_on_the_standardised_design(lawful_fit_only, model):
    rows = towertier.tier_rows(lawful_fit_only, ('FIT',))
    X = np.column_stack([rows.destroyed, rows.taker, rows.t_ms / 60_000.0]).astype(np.float64)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    ref = LogisticRegression(C=1.0, max_iter=2000, tol=1e-8).fit((X - mean) / scale, rows.tier)
    assert model.classes_.tolist() == ref.classes_.tolist() == [0, 1, 2, 3]
    assert np.allclose(model.mean_, mean, rtol=1e-12, atol=0) and np.allclose(model.scale_, scale, rtol=1e-12, atol=0)
    mine = model.prob(rows.destroyed, rows.taker, rows.t_ms / 1000.0)
    unmixed = (mine - MIX / 4) / (1.0 - MIX)                                  # take the mixing of ruling R13 off again
    assert np.abs(unmixed - ref.predict_proba((X - mean) / scale)).max() < 1e-7
    assert np.allclose(model.coef_std_, ref.coef_, atol=1e-6) and np.allclose(model.intercept_std_, ref.intercept_, atol=1e-6)


def test_the_estimator_is_the_one_of_ruling_r20_on_a_standardised_design(small, monkeypatch):
    seen = []
    real = LogisticRegression.fit

    def fit(self, X, y, sample_weight=None):
        seen.append((self.get_params(), np.array(X), np.array(y)))
        return real(self, X, y, sample_weight)
    monkeypatch.setattr(LogisticRegression, 'fit', fit)
    towertier.TierModel.fit(small)
    (params, X, y), = seen
    assert (params['C'], params['max_iter'], params['tol'], params['solver'], params['penalty']) == (1.0, 2000, 1e-8, 'lbfgs', 'l2')
    assert params['class_weight'] is None and params['fit_intercept'] is True
    assert X.shape == (len(y), 6) and np.allclose(X.mean(axis=0), 0, atol=1e-12) and np.allclose(X.std(axis=0), 1, atol=1e-12)
    assert (towertier.C_LOGISTIC, towertier.MAX_ITER, towertier.TOL, towertier.MIX) == (1.0, 2000, 1e-8, 1e-6)


def test_a_tier_that_never_occurred_gets_the_uniform_share_of_ruling_r13(tmp_path):
    law = (OUTER,) * 3 + (INNER,) * 3 + (NEXUS,) * 2                         # no base tower ever falls
    s = build_store(tmp_path / 'law', lawful_toys(8, 120, law=law))
    m = towertier.TierModel.fit(s)
    assert m.classes_.tolist() == [OUTER, INNER, NEXUS]
    rng = np.random.default_rng(1)
    p = m.prob(rng.integers(0, 4, (50, 4)), rng.integers(0, 2, 50), rng.uniform(0, 3_000, 50))
    assert (p[:, BASE] == MIX / 4).all() and np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12) and (p[:, [0, 1, 3]] > MIX / 4).all()
    assert m.summary()['n_by_tier'][BASE] == 0


def test_one_tier_gives_a_constant_model_and_two_tiers_a_binary_one(tmp_path):
    one = towertier.TierModel.fit(build_store(tmp_path / 'one', lawful_toys(9, 20, law=(OUTER,) * 3)))
    assert one.classes_.tolist() == [OUTER] and one.summary()['converged'] is True and one.summary()['n_iter'] == 0
    assert one.converged_ is True and one.n_iter_ == 0                           # a constant model is not iterated and counts as converged
    p = one.prob([[0, 0, 0, 0], [3, 2, 1, 0]], [BLUE, RED], [60.0, 3_000.0])
    assert np.allclose(p, [[1.0 - MIX + MIX / 4, MIX / 4, MIX / 4, MIX / 4]] * 2, rtol=0, atol=1e-15)
    two_store = build_store(tmp_path / 'two', lawful_toys(10, 80, law=(OUTER,) * 3 + (INNER,) * 3))
    two = towertier.TierModel.fit(two_store)
    assert two.classes_.tolist() == [OUTER, INNER] and two.coef_std_.shape == (2, 6) and not two.coef_std_[0].any()      # logits [0, z]
    q = two.prob([lawful_state(0, (OUTER,) * 3 + (INNER,) * 3), lawful_state(5, (OUTER,) * 3 + (INNER,) * 3)], [BLUE, BLUE], [600.0, 600.0])
    assert q[0, OUTER] > 0.9 and q[1, INNER] > 0.9 and (q[:, [BASE, NEXUS]] == MIX / 4).all()


def test_fit_with_no_tower_event_in_the_roles_is_refused_without_a_key(hand):
    with pytest.raises(ValueError, match='tower') as raised:
        towertier.TierModel.fit(hand, roles=('DEV',))
    assert KEY_PREFIX not in str(raised.value) and 'DEV' in str(raised.value)


def test_fit_is_deterministic_bit_for_bit(lawful, model, tmp_path):
    again = towertier.TierModel.fit(lawful, roles=('FIT',))
    for name in ('classes_', 'coef_std_', 'intercept_std_', 'mean_', 'scale_'):
        assert np.array_equal(getattr(again, name), getattr(model, name)), name
    model.save(tmp_path / 'a')
    again.save(tmp_path / 'b')
    for name in (towertier.NPZ_NAME, towertier.JSON_NAME):
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()


# ---- convergence (ruling R20) -----------------------------------------------------------------------------------------------------
def test_a_converged_fit_says_so_and_raises_no_convergence_warning(small):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        m = towertier.TierModel.fit(small)
    summary = m.summary()
    assert summary['converged'] is True and 0 < summary['n_iter'] < towertier.MAX_ITER
    assert m.converged_ is True and m.n_iter_ == summary['n_iter'] and isinstance(m.n_iter_, int)          # the attributes of ruling R20


def test_a_convergence_warning_marks_the_fit_not_converged_also_below_the_iteration_limit(small, monkeypatch):
    real = LogisticRegression.fit

    def fit(self, X, y, sample_weight=None):
        out = real(self, X, y, sample_weight)
        warnings.warn('lbfgs failed to converge (status=2): ABNORMAL', ConvergenceWarning)
        return out
    monkeypatch.setattr(LogisticRegression, 'fit', fit)
    with pytest.warns(ConvergenceWarning, match='TierModel') as caught:
        m = towertier.TierModel.fit(small)
    assert len([w for w in caught if issubclass(w.category, ConvergenceWarning)]) == 1                 # ours replaces scikit-learn's
    assert 0 < m.summary()['n_iter'] < towertier.MAX_ITER and m.summary()['converged'] is False
    assert m.converged_ is False and m.n_iter_ == m.summary()['n_iter']


def test_reaching_the_iteration_limit_marks_the_fit_not_converged_even_when_no_warning_came(small, monkeypatch):
    real = LogisticRegression.fit

    def fit(self, X, y, sample_weight=None):
        out = real(self, X, y, sample_weight)
        self.n_iter_ = np.array([self.max_iter])
        return out
    monkeypatch.setattr(LogisticRegression, 'fit', fit)
    with pytest.warns(ConvergenceWarning, match='TierModel'):
        m = towertier.TierModel.fit(small)
    assert m.summary()['converged'] is False and m.summary()['n_iter'] == towertier.MAX_ITER
    assert m.converged_ is False and m.n_iter_ == towertier.MAX_ITER


def test_a_warning_that_is_not_a_convergence_warning_goes_on_to_the_caller(small, monkeypatch):
    real = LogisticRegression.fit

    def fit(self, X, y, sample_weight=None):
        out = real(self, X, y, sample_weight)
        warnings.warn('something else about the estimator', UserWarning)
        return out
    monkeypatch.setattr(LogisticRegression, 'fit', fit)
    with pytest.warns(UserWarning, match='something else about the estimator'):
        m = towertier.TierModel.fit(small)
    assert m.summary()['converged'] is True                                    # it is no ConvergenceWarning: the flag does not change


# ---- E6, roles, what is read ------------------------------------------------------------------------------------------------------
def test_fit_calls_the_check_once_before_the_first_read(hand):
    log = []
    towertier.TierModel.fit(SpyStore(hand, log), roles=('FIT',), check=lambda: log.append('check'))
    assert log.count('check') == 1 and log[0] == 'check' and len(log) > 1


def test_an_e6_refusal_stops_the_fit_before_anything_is_read(hand):
    log = []

    def refuse():
        log.append('check')
        raise RuntimeError('E6: stop')
    with pytest.raises(RuntimeError, match='E6'):
        towertier.TierModel.fit(SpyStore(hand, log), check=refuse)
    assert log == ['check']


def test_the_real_e6_guard_refuses_through_check_while_a_model_server_runs(hand, fake_processes):
    pytest.importorskip('psutil')                                               # without it the guard cannot see any process
    from c_ept import run_s1
    fake_processes([(4242, 'llama-server.exe', ['llama-server.exe', '--model', 'x'])])
    spy = SpyStore(hand)
    with pytest.raises(run_s1.StrataRunning):
        towertier.TierModel.fit(spy, check=run_s1.require_no_strata)
    assert spy.reads == []
    fake_processes([])
    assert towertier.TierModel.fit(spy, check=run_s1.require_no_strata).summary()['n_rows'] == 9


def test_fit_refuses_a_holdout_role_and_a_role_s3_does_not_use_before_any_read():
    calls = []
    for roles in ('TEST', ('EXTERNAL',), ('FIT', 'TEST'), ('CAL',), ('AUDIT',), ('UNUSED',), ('NOT_A_ROLE',), ()):
        with pytest.raises(ValueError):
            towertier.TierModel.fit(Untouchable(), roles=roles, check=lambda: calls.append(1))
    assert calls == []                                                          # a refused role stops the call before the E6 check too


def test_fit_reads_only_the_arrays_it_is_documented_to_read(hand):
    spy = SpyStore(hand)
    towertier.TierModel.fit(spy)
    assert spy.reads and set(spy.reads) <= ARRAYS_READ and 'matches/key' not in spy.reads


def test_the_summary_reports_the_counts_of_the_fit(hand):
    m = towertier.TierModel.fit(hand)
    summary = m.summary()
    assert summary['classes'] == [0, 1, 2, 3] and summary['fit_roles'] == ['FIT'] and summary['converged'] is True
    assert (summary['n_matches'], summary['n_rows'], summary['n_by_tier']) == (2, 9, [5, 2, 1, 1])
    assert (summary['n_tower_events'], summary['n_no_tier'], summary['n_no_taker']) == (12, 2, 1)
    assert json.loads(json.dumps(summary)) == summary                           # plain JSON values
    summary['n_rows'] = 0                                                       # a copy: changing it, also a list in it, does not
    summary['n_by_tier'][0] = 99                                                # change the model
    summary['fit_roles'].append('DEV')
    summary['classes'].clear()
    again = m.summary()
    assert (again['n_rows'], again['n_by_tier'], again['fit_roles'], again['classes']) == (9, [5, 2, 1, 1], ['FIT'], [0, 1, 2, 3])


def test_roles_may_be_one_name_or_a_tuple_of_names(hand):
    one, many = towertier.TierModel.fit(hand, roles='FIT'), towertier.TierModel.fit(hand, roles=('FIT',))
    assert one.summary() == many.summary() and np.array_equal(one.coef_std_, many.coef_std_)
    both = towertier.TierModel.fit(hand, roles=['STOP', 'FIT'])
    assert both.summary()['fit_roles'] == ['FIT', 'STOP'] and both.summary()['n_rows'] == 10        # in the order of config.ROLES


# =====================================================================================================================================
# prob: the closed form worked out by hand
# =====================================================================================================================================
def softmax(z):
    e = [math.exp(v - max(z)) for v in z]
    return [v / sum(e) for v in e]


def mixed(full):
    return [(1.0 - MIX) * v + MIX / 4 for v in full]


def test_a_binary_model_with_a_standardised_minute_by_hand():
    classes = [OUTER, INNER]
    coef = [[0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]                          # logits: class 0 gets 0, class 1 gets 2 z_minute - 1
    m = towertier.TierModel(classes, coef, [0.0, -1.0], [0, 0, 0, 0, 0, 10.0], [1, 1, 1, 1, 1, 5.0])
    p = m.prob([[0, 0, 0, 0]], [BLUE], [900.0])[0]                              # 900 s = 15 min, z = (15 - 10) / 5 = 1, logit = 1
    one = 1.0 / (1.0 + math.exp(-1.0))
    assert p.tolist() == pytest.approx(mixed([1.0 - one, one, 0.0, 0.0]), rel=1e-13, abs=0)
    assert one == pytest.approx(0.7310585786300049)


def test_a_three_tier_model_with_standardised_counts_and_taker_by_hand():
    classes = [OUTER, BASE, NEXUS]
    coef = [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
    intercept = [0.0, 0.5, -0.5]
    mean, scale = [1, 1, 1, 1, 0.5, 10], [2, 2, 2, 2, 0.5, 5]
    m = towertier.TierModel(classes, coef, intercept, mean, scale)
    # counts (3, 0, 5, 0): z = (1, -0.5, 2, -0.5); taker 1: z = 1; t = 0
    p = m.prob([[3, 0, 5, 0]], [RED], [0.0])[0]
    z = [1.0 * 1 + 0.0, 2.0 + 0.5, 1.0 - 0.5]
    s = softmax(z)
    assert p.tolist() == pytest.approx(mixed([s[0], 0.0, s[1], s[2]]), rel=1e-13, abs=0)
    assert p[INNER] == MIX / 4 and p.sum() == pytest.approx(1.0, abs=1e-15)


def test_a_model_with_zero_coefficients_is_uniform():
    m = towertier.TierModel([0, 1, 2, 3], np.zeros((4, 6)), np.zeros(4), np.zeros(6), np.ones(6))
    p = m.prob(np.zeros((2, 4)), [0, 1], [10.0, 20.0])
    assert p.shape == (2, 4) and np.allclose(p, 0.25, rtol=1e-14, atol=0)


def test_prob_is_row_wise(model):
    rng = np.random.default_rng(4)
    D = rng.integers(0, 4, (30, 4))
    taker, t_s = rng.integers(0, 2, 30), rng.uniform(0, 3_000, 30)
    batch = model.prob(D, taker, t_s)
    single = np.vstack([model.prob(D[i:i + 1], taker[i:i + 1], t_s[i:i + 1]) for i in range(30)])
    assert np.allclose(batch, single, rtol=1e-12, atol=1e-15)
    assert model.prob(np.zeros((0, 4)), np.zeros(0), np.zeros(0)).shape == (0, 4)


def test_prob_accepts_lists_and_other_numeric_types_and_leaves_its_inputs_alone(model):
    D, taker, t_s = np.array([[1, 0, 0, 0], [3, 1, 0, 0]]), np.array([0, 1]), np.array([600.0, 1_200.0])
    keep = (D.copy(), taker.copy(), t_s.copy())
    a = model.prob(D, taker, t_s)
    assert all(np.array_equal(x, y) for x, y in zip((D, taker, t_s), keep))
    assert np.array_equal(a, model.prob(D.tolist(), taker.tolist(), t_s.tolist()))
    assert np.array_equal(a, model.prob(D.astype(np.int8), taker.astype(bool), t_s.astype(np.float32)))
    assert np.array_equal(a, model.prob(D.astype(float), taker.astype(float), np.array([600, 1_200])))


def test_prob_refuses_inputs_that_are_not_states_takers_and_times(model):
    ok = ([[0, 0, 0, 0]], [0], [60.0])
    bad = [([[0, 0, 0]], [0], [60.0]), ([0, 0, 0, 0], [0], [60.0]), ([[0, 0, 0, 0]], [0, 1], [60.0]), ([[0, 0, 0, 0]], [0], [60.0, 70.0]),
           ([[0, 0, 0, 0]], [2], [60.0]), ([[0, 0, 0, 0]], [-1], [60.0]), ([[0, 0, 0, 0]], [0.5], [60.0]),
           ([[0, 0, 0, 0]], [np.nan], [60.0]),
           ([[-1, 0, 0, 0]], [0], [60.0]), ([[np.nan, 0, 0, 0]], [0], [60.0]), ([[np.inf, 0, 0, 0]], [0], [60.0]),
           ([[0, 0, 0, 0]], [0], [-1.0]), ([[0, 0, 0, 0]], [0], [np.nan]), ([[0, 0, 0, 0]], [0], [np.inf]), ([[0, 0, 0, 0]], [[0]], [60.0])]
    model.prob(*ok)
    for args in bad:
        with pytest.raises(ValueError):
            model.prob(*args)
    for args in bad[2:4]:                                                       # a taker or a time for every row, said in so many words
        with pytest.raises(ValueError, match='one entry for each'):
            model.prob(*args)
    for args in ([['a', 'b', 'c', 'd']], [0], [60.0]), ([[0, 0, 0, 0]], [0], ['x']):
        with pytest.raises((TypeError, ValueError)):
            model.prob(*args)


def test_the_parameters_of_a_model_are_copies_and_read_only(model, tmp_path):
    coef = np.zeros((2, 6))
    made = towertier.TierModel([0, 1], coef, np.zeros(2), np.zeros(6), np.ones(6))
    coef[0, 0] = 5.0                                                            # the model has its own copy
    assert not made.coef_std_.any()
    model.save(tmp_path / 'tier')
    for m in (made, model, towertier.TierModel.load(tmp_path / 'tier')):
        for name in ('classes_', 'coef_std_', 'intercept_std_', 'mean_', 'scale_'):
            with pytest.raises(ValueError, match='read-only'):
                getattr(m, name)[0] = 1


def test_a_model_is_built_from_consistent_parameters_only():
    good = dict(classes=[0, 1], coef=np.zeros((2, 6)), intercept=np.zeros(2), mean=np.zeros(6), scale=np.ones(6))
    towertier.TierModel(**good)
    for change in ({'classes': [1, 0]}, {'classes': [0, 0]}, {'classes': [0, 4]}, {'classes': [-1, 0]}, {'classes': []},
                   {'coef': np.zeros((2, 5))}, {'coef': np.zeros((3, 6))}, {'intercept': np.zeros(3)}, {'mean': np.zeros(5)},
                   {'scale': np.ones(5)}, {'scale': np.zeros(6)}, {'scale': np.full(6, -1.0)}, {'coef': np.full((2, 6), np.nan)},
                   {'mean': np.full(6, np.inf)}):
        with pytest.raises(ValueError):
            towertier.TierModel(**{**good, **change})
    with pytest.raises(TypeError):
        towertier.TierModel(**{**good, 'classes': [0.0, 1.0]})
    blank = towertier.TierModel(**good).summary()                              # a model made by hand has a blank record of the fit
    assert blank['classes'] == [0, 1] and blank['n_rows'] == 0 and blank['fit_roles'] == [] and blank['converged'] is True
    info = {k: v for k, v in blank.items() if k != 'classes'}
    assert towertier.TierModel(**good, info=info).summary() == blank
    for bad in ({'n_rows': 1}, {**info, 'extra': 1}, [], 'info', {k: v for k, v in info.items() if k != 'sklearn'}):
        with pytest.raises(ValueError):
            towertier.TierModel(**good, info=bad)


# =====================================================================================================================================
# expected_tower_price
# =====================================================================================================================================
def test_expected_price_is_sum_of_prob_times_tier_price():
    prob = np.array([[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    got = towertier.expected_tower_price(prob, PRICES)
    want = [0.1 * 520.0 + 0.2 * 610.0 + 0.3 * 400.0 + 0.4 * 900.0, 520.0, 0.25 * (520.0 + 610.0 + 400.0 + 900.0)]
    assert got.shape == (3,) and got.dtype == np.float64 and got.tolist() == pytest.approx(want, rel=1e-14)
    assert want[0] == pytest.approx(52 + 122 + 120 + 360)                                  # 654, worked out by hand


def test_a_certain_tier_has_the_price_of_its_group_and_the_order_of_the_tiers_is_outer_inner_base_nexus():
    for tier, group in enumerate(('tower_outer', 'tower_inner', 'tower_base', 'tower_nexus')):
        sure = np.eye(4)[[tier]]
        assert towertier.expected_tower_price(sure, PRICES).tolist() == [PRICES[group]]
    assert towertier.TIER_GROUPS == ('tower_outer', 'tower_inner', 'tower_base', 'tower_nexus') == flowprice.GROUPS[6:10]


def test_expected_price_ignores_the_other_groups_and_is_linear_in_the_weights():
    prices = {g: PRICES[g] for g in towertier.TIER_GROUPS}                                  # only the four tower groups are needed
    w = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert towertier.expected_tower_price(w, prices).tolist() == [520.0 + 1220.0 + 1200.0 + 3600.0]
    assert towertier.expected_tower_price(2 * w, prices).tolist() == [2 * (520.0 + 1220.0 + 1200.0 + 3600.0)]
    assert towertier.expected_tower_price(np.zeros((0, 4)), prices).shape == (0,)


def test_expected_price_refuses_a_missing_group_and_a_price_that_is_not_a_finite_number():
    prob = np.full((2, 4), 0.25)
    for group in towertier.TIER_GROUPS:
        with pytest.raises(ValueError, match=group):
            towertier.expected_tower_price(prob, {k: v for k, v in PRICES.items() if k != group})
        for value in (None, float('nan'), float('inf'), True, 'abc'):
            with pytest.raises(ValueError, match=group):
                towertier.expected_tower_price(prob, {**PRICES, group: value})
    with pytest.raises(TypeError):
        towertier.expected_tower_price(prob, [520.0, 610.0, 400.0, 900.0])


def test_expected_price_refuses_probabilities_that_are_not_four_non_negative_numbers():
    for prob in (np.ones((2, 3)), np.ones(4), np.full((2, 4), -0.1), np.full((2, 4), np.nan), np.full((2, 4), np.inf), np.ones((2, 4, 1))):
        with pytest.raises(ValueError):
            towertier.expected_tower_price(prob, PRICES)


def test_the_expected_price_of_a_fitted_state_is_the_price_of_the_tier_that_falls_next(model):
    prices = {**PRICES}
    for j, tier in enumerate(LAW):
        p = model.prob([lawful_state(j)], [BLUE], [1_200.0])
        price = towertier.expected_tower_price(p, prices)[0]
        assert price == pytest.approx(prices[towertier.TIER_GROUPS[tier]], rel=0.1), (j, tier, price)
    nexus_next = model.prob([[3, 3, 3, 0]], [RED], [1_800.0])
    assert towertier.expected_tower_price(nexus_next, prices)[0] == pytest.approx(900.0, rel=0.05)


# =====================================================================================================================================
# save and load
# =====================================================================================================================================
def same_model(a, b):
    return all(np.array_equal(getattr(a, n), getattr(b, n)) and getattr(a, n).dtype == getattr(b, n).dtype
               for n in ('classes_', 'coef_std_', 'intercept_std_', 'mean_', 'scale_'))


def test_save_writes_an_npz_and_a_json_and_nothing_else(model, tmp_path):
    model.save(tmp_path / 'tier')
    assert sorted(p.name for p in (tmp_path / 'tier').iterdir()) == ['tower_tier.json', 'tower_tier.npz']
    with np.load(tmp_path / 'tier' / 'tower_tier.npz', allow_pickle=False) as z:                    # no pickle: it loads without it
        assert set(z.files) == {'classes', 'coef', 'intercept', 'mean', 'scale'} and all(z[k].dtype.kind in 'fi' for k in z.files)
    doc = json.loads((tmp_path / 'tier' / 'tower_tier.json').read_text(encoding='utf-8'))
    assert doc['schema'] == 'tower_tier_v1' and doc['features'] == list(towertier.FEATURE_NAMES)
    assert doc['groups'] == list(towertier.TIER_GROUPS)
    assert doc['npz_sha256'] == hashlib.sha256((tmp_path / 'tier' / 'tower_tier.npz').read_bytes()).hexdigest()


def test_load_gives_back_the_same_model_bit_for_bit(model, tmp_path):
    model.save(tmp_path / 'tier')
    again = towertier.TierModel.load(tmp_path / 'tier')
    assert same_model(again, model) and again.summary() == model.summary()
    rng = np.random.default_rng(5)
    D, taker, t_s = rng.integers(0, 4, (400, 4)), rng.integers(0, 2, 400), rng.uniform(0, 4_000, 400)
    assert np.array_equal(again.prob(D, taker, t_s), model.prob(D, taker, t_s))                    # the same bits, not just close


def test_a_loaded_model_saves_the_same_bytes_again(model, tmp_path):
    model.save(tmp_path / 'a')
    towertier.TierModel.load(tmp_path / 'a').save(tmp_path / 'b')
    for name in (towertier.NPZ_NAME, towertier.JSON_NAME):
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()


def test_the_saved_files_do_not_depend_on_the_time_of_saving(model, tmp_path):
    model.save(tmp_path / 'a')
    model.save(tmp_path / 'b')
    for name in (towertier.NPZ_NAME, towertier.JSON_NAME):
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()


def test_the_npz_is_stored_without_compression_and_without_time_stamps(model, tmp_path):
    model.save(tmp_path / 'tier')
    with zipfile.ZipFile(tmp_path / 'tier' / 'tower_tier.npz') as zf:
        infos = zf.infolist()
    assert [i.filename for i in infos] == ['classes.npy', 'coef.npy', 'intercept.npy', 'mean.npy', 'scale.npy']
    assert all(i.date_time == (1980, 1, 1, 0, 0, 0) and i.compress_type == zipfile.ZIP_STORED for i in infos)


def test_save_never_writes_over_a_saved_model(model, tmp_path):
    model.save(tmp_path / 'tier')
    with pytest.raises(FileExistsError):
        model.save(tmp_path / 'tier')


def test_load_refuses_a_changed_array_file(model, tmp_path):
    model.save(tmp_path / 'tier')
    path = tmp_path / 'tier' / 'tower_tier.npz'
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='sha256|changed|hash'):
        towertier.TierModel.load(tmp_path / 'tier')


def test_load_refuses_a_missing_file_and_a_wrong_schema(model, tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        towertier.TierModel.load(tmp_path / 'nothing')
    model.save(tmp_path / 'tier')
    path = tmp_path / 'tier' / 'tower_tier.json'
    doc = json.loads(path.read_text(encoding='utf-8'))
    path.write_text(json.dumps({**doc, 'schema': 'something_else'}), encoding='utf-8')
    with pytest.raises(ValueError, match='schema'):
        towertier.TierModel.load(tmp_path / 'tier')
    path.write_text(json.dumps({**doc, 'features': doc['features'][::-1]}), encoding='utf-8')
    with pytest.raises(ValueError, match='features'):
        towertier.TierModel.load(tmp_path / 'tier')
    path.write_text(json.dumps({**doc, 'groups': doc['groups'][::-1]}), encoding='utf-8')
    with pytest.raises(ValueError, match='groups'):
        towertier.TierModel.load(tmp_path / 'tier')
    path.write_text(json.dumps(doc), encoding='utf-8')
    assert same_model(towertier.TierModel.load(tmp_path / 'tier'), model)                          # the untouched JSON still loads


def test_load_refuses_a_json_whose_facts_are_malformed_or_disagree_with_the_arrays(model, tmp_path):
    model.save(tmp_path / 'tier')
    path = tmp_path / 'tier' / 'tower_tier.json'
    original = json.loads(path.read_text(encoding='utf-8'))
    changes = (lambda d: d['counts'].__setitem__('n_rows', 'many'), lambda d: d['counts'].__setitem__('n_rows', True),
               lambda d: d['counts'].__setitem__('n_by_tier', [1, 2, 3]), lambda d: d['counts'].pop('n_matches'),
               lambda d: d['fit'].__setitem__('converged', 'yes'), lambda d: d['fit'].__setitem__('n_iter', 1.5),
               lambda d: d.__setitem__('fit_roles', ['TEAM']), lambda d: d.__setitem__('fit_roles', 'FIT'),
               lambda d: d.__setitem__('counts', [1]), lambda d: d.__setitem__('classes', [0, 1]),
               # made with other settings: ``prob`` would not give the same answer, so the model is not loaded
               lambda d: d['fit'].__setitem__('mix', 1e-5), lambda d: d['fit'].__setitem__('C', 2.0),
               lambda d: d['fit'].__setitem__('tol', 1e-4), lambda d: d['fit'].__setitem__('max_iter', 100), lambda d: d['fit'].pop('mix'),
               lambda d: d['fit'].__setitem__('C', True))                                # true == 1.0 in Python, but it is no number here
    for change in changes:
        doc = json.loads(json.dumps(original))
        change(doc)
        path.write_text(json.dumps(doc), encoding='utf-8')
        with pytest.raises(ValueError):
            towertier.TierModel.load(tmp_path / 'tier')
    path.write_text(json.dumps(original), encoding='utf-8')
    assert towertier.TierModel.load(tmp_path / 'tier').summary() == model.summary()


def test_load_refuses_arrays_that_do_not_fit_the_features_even_with_a_matching_hash(model, tmp_path):
    model.save(tmp_path / 'tier')
    npz, js = tmp_path / 'tier' / 'tower_tier.npz', tmp_path / 'tier' / 'tower_tier.json'
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays['coef'] = arrays['coef'][:, :5]                                                         # five columns, not six
    np.savez(npz, **arrays)
    doc = json.loads(js.read_text(encoding='utf-8'))
    doc['npz_sha256'] = hashlib.sha256(npz.read_bytes()).hexdigest()
    js.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='shape|columns|features'):
        towertier.TierModel.load(tmp_path / 'tier')


def test_the_saved_json_holds_no_key_and_the_fit_facts(model, tmp_path):
    model.save(tmp_path / 'tier')
    text = (tmp_path / 'tier' / 'tower_tier.json').read_text(encoding='utf-8')
    assert KEY_PREFIX not in text
    doc = json.loads(text)
    assert doc['fit_roles'] == ['FIT'] and doc['classes'] == [0, 1, 2, 3] and doc['counts']['n_matches'] == 300
    assert doc['fit']['C'] == 1.0 and doc['fit']['max_iter'] == 2000 and doc['fit']['tol'] == 1e-8 and doc['fit']['mix'] == 1e-6
    assert doc['fit']['converged'] is True and doc['fit']['n_iter'] == model.summary()['n_iter']


# =====================================================================================================================================
# structure
# =====================================================================================================================================
def test_the_module_imports_no_runner_no_raw_reader_no_value_no_model_and_no_pickle():
    imported = module_level_imports(SOURCE)
    forbidden = {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.cohort', 'c_ept.runlog', 'c_ept.terrain', 'c_ept.store_build', 'c_ept.reader',
                 'c_ept.tokens', 'c_ept.value', 'c_ept.compose', 'c_ept.units', 'c_ept.targets', 'c_ept.baselines_b0', 'c_ept.baselines_b1',
                 'c_ept.heads', 'c_ept.head_rows', 'c_ept.features', 'sklearn', 'lightgbm'}
    assert not imported & forbidden, imported & forbidden
    # the E6 guard is the ``check`` argument: the runner owns ``run_s1``, this module never imports it, not even inside a function
    assert not all_imports(SOURCE) & {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.reader', 'c_ept.cohort', 'pickle'}


def test_the_module_names_only_the_arrays_it_is_documented_to_read():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.count('/') == 1 and ' ' not in node.value and node.value.split('/')[0].isidentifier()}
    assert names <= ARRAYS_READ, names - ARRAYS_READ
    assert names == ARRAYS_READ                                                # and every one of them is read somewhere


def test_the_tier_groups_and_constants_are_those_of_the_plan():
    assert towertier.N_TIERS == 4 and towertier.MIX == 1e-6 and towertier.FEATURE_NAMES[-2:] == ('taker', 'minute')
    assert len(towertier.FEATURE_NAMES) == 6 and towertier.TIER_GROUPS == tuple(flowprice.GROUPS[i] for i in range(6, 10))


def test_no_message_names_a_match_key(hand):
    messages = []
    for call in (lambda: towertier.destroyed_before(hand, [99], [0]), lambda: towertier.tier_rows(hand, ('TEST',)),
                 lambda: towertier.TierModel.fit(hand, roles=('DEV',)), lambda: towertier.TierModel.fit(hand, roles=('NOPE',))):
        with pytest.raises(ValueError) as raised:
            call()
        messages.append(str(raised.value))
    assert messages and not any(KEY_PREFIX in m for m in messages)
