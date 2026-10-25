"""Tests of c_ept.units: the link rule, the clocks (tau, delta, scope), size classes and first-kill marks. Synthetic inputs only."""
import dataclasses
import runpy

import numpy as np
import pytest

from c_ept import config as C
from c_ept.data import reader
from c_ept.domain import units
from c_ept.tests import synth
from c_ept.tests.synth import TEAM, kt


def comps(t, x, y):
    return [list(c) for c in units.link_components(np.array(t), np.array(x, float), np.array(y, float))]


def as_sets(parts):
    return {frozenset(int(i) for i in c) for c in parts}


def brute_force(t, x, y, gap_ms=C.G_MS, link_u=C.L_U):
    """Components of the graph over all pairs (no sweep window, no union-find), as a set of frozensets of indices."""
    n = len(t)
    near = [[j for j in range(n) if j != i and abs(int(t[i]) - int(t[j])) <= gap_ms
             and (x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]) <= link_u * link_u] for i in range(n)]
    seen, out = set(), set()
    for start in range(n):
        if start in seen:
            continue
        part, stack = {start}, [start]
        seen.add(start)
        while stack:
            for j in near[stack.pop()]:
                if j not in seen:
                    seen.add(j)
                    part.add(j)
                    stack.append(j)
        out.add(frozenset(part))
    return out


def test_link_boundaries_inclusive():
    assert comps([0, 15_000], [0, 2_000], [0, 0]) == [[0, 1]]
    assert comps([0, 15_001], [0, 0], [0, 0]) == [[0], [1]]
    assert comps([0, 1], [0, 2_000.001], [0, 0]) == [[0], [1]]


def test_link_transitive_chain():
    assert comps([0, 10_000, 20_000], [0, 1_500, 3_000], [0, 0, 0]) == [[0, 1, 2]]


def test_nan_position_is_singleton():
    assert comps([0, 1_000], [0, np.nan], [0, np.nan]) == [[0], [1]]


def test_link_matches_bruteforce_and_prototype(proto):
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(1, 40))
        t = np.sort(rng.integers(0, 600_000, n))
        x, y = rng.uniform(0, 15_000, (2, n))
        assert as_sets(units.link_components(t, x, y)) == brute_force(t, x, y)
        if proto is not None:
            assert [list(c) for c in units.link_components(t, x, y)] == proto.link_components(t, x, y, 15_000)


def test_tau_delta_scope():
    u = units.build_units(kt([(200_000, 0), (215_000, 500), (230_000, 1_000)]), TEAM, 600_000)     # a chain of 15 s steps
    assert (u.f[0], u.last[0], u.tau[0], u.delta[0]) == (200_000, 230_000, 185_000, 230_001)
    u = units.build_units(kt([(200_000, 0)]), TEAM, 600_000)
    assert u.delta[0] == 220_000
    s = units.build_units(kt([(119_999, 0), (300_000, 9_000), (570_000, 0), (580_000, 9_000)]), TEAM, 600_000)
    assert list(s.in_scope) == [False, True, True, False]       # 570,000 = T - 30,000 is in, 580,000 is out


def test_size_class_table():
    cases = {(4, 4): 0, (5, 4): 0, (2, 3): 1, (3, 3): 1, (2, 4): 2, (4, 3): 2, (1, 5): 3, (0, 0): 3}
    assert {k: units.size_class(*k) for k in cases} == cases


def test_mark_codes():
    assert [units.first_kill_mark(k, v) for k, v in ((1, 6), (5, 10), (6, 1), (10, 5), (0, 3), (0, 8))] == [0, 24, 25, 49, 50, 51]


def test_first_kill_tie_uses_event_order():
    u = units.build_units(kt([(200_000, 0, 1, 6), (200_000, 100, 7, 2)]), TEAM, 600_000)
    assert u.mark[0] == units.first_kill_mark(1, 6)


def test_participants_counted_per_team():
    u = units.build_units(kt([(200_000, 0, 1, 6, (2, 3)), (205_000, 0, 7, 2, (8,))]), TEAM, 600_000)
    assert (u.nb[0], u.nr[0], u.size[0]) == (3, 3, 1)       # blue {1, 2, 3}, red {6, 7, 8}


def test_same_team_kill_rejected():
    with pytest.raises(reader.MatchRejected, match='same_team_kill'):
        units.build_units(kt([(200_000, 0, 1, 2)]), TEAM, 600_000)


# ---- beyond the brief ---------------------------------------------------------------------------------------------------------
# A mutation check of units.py against the ten tests above left three changes alive: the lower bound of the scope (f = 120,000),
# the order of killer slot and victim slot in a mark (every pair in test_mark_codes has equal slots) and blue <-> red in (nb, nr)
# (the example above is 3 : 3). The tests below close those gaps and cover what the ten do not reach: spread, n_kills, the column
# types and order, executions that start a unit, and kill_table.
def test_unit_columns_and_dtypes():
    # unit 0 starts at the lower bound of the scope; NaN position and a distance of 3,000 link to nothing
    u = units.build_units(kt([(120_000, 0), (125_000, 1_500), (126_000, np.nan), (400_000, 0), (401_000, 3_000)]), TEAM, 600_000)
    assert [list(g) for g in u.comp] == [[0, 1], [2], [3], [4]]
    assert list(u.f) == [120_000, 126_000, 400_000, 401_000] and list(u.n_kills) == [2, 1, 1, 1]
    assert list(u.in_scope) == [True, True, True, True] and list(u.spread) == [1_500.0, 0.0, 0.0, 0.0]
    # the widest pair, not the first and last kill, and in the plane: (0, 0), (1200, 1600) at exactly 2,000, (600, 0)
    k = dataclasses.replace(kt([(200_000, 0), (201_000, 1_200), (202_000, 600)]), y=np.array([0.0, 1_600.0, 0.0]))
    w = units.build_units(k, TEAM, 600_000)
    assert [list(g) for g in w.comp] == [[0, 1, 2]] and w.spread[0] == 2_000.0
    dtypes = dict(f='i8', last='i8', tau='i8', delta='i8', nb='i1', nr='i1', size='i1', mark='i2', in_scope='b1', spread='f4', n_kills='i2')
    assert [f.name for f in dataclasses.fields(units.UnitTable)] == ['comp', *dtypes]
    assert {c: getattr(u, c).dtype.str[1:] for c in dtypes} == dtypes
    e = units.build_units(kt([]), TEAM, 600_000)                                   # no kills: no units, same column types
    assert e.comp == [] and {c: getattr(e, c).dtype.str[1:] for c in dtypes} == dtypes and all(len(getattr(e, c)) == 0 for c in dtypes)


def test_mark_and_team_orientation():
    # mark = 25 d + 5 i + j with the killer's slot i and the victim's slot j
    assert [units.first_kill_mark(k, v) for k, v in ((1, 7), (2, 6), (3, 9), (8, 4), (9, 3))] == [1, 5, 13, 38, 42]
    # (nb, nr) is (blue, red): blue {1, 2}, red {6, 7, 8, 9}
    u = units.build_units(kt([(200_000, 0, 1, 6, (2,)), (205_000, 0, 7, 2, (8, 9))]), TEAM, 600_000)
    assert (u.nb[0], u.nr[0], u.size[0]) == (2, 4, 2)
    # an execution starts a unit: mark 50 / 51 by the victim's team; killer 0 counts nobody, the victim and the assisters count
    u = units.build_units(kt([(200_000, 0, 0, 3, (6, 7)), (400_000, 0, 0, 8, (1,))]), TEAM, 600_000)
    assert list(u.mark) == [50, 51] and [(a, b) for a, b in zip(u.nb, u.nr)] == [(1, 2), (1, 1)]


def test_kill_table_columns():
    nan = float('nan')
    rec = synth.make_record(T_ms=900_000, kills=[
        dict(t=130_000, x=1_000, y=2_000, killer=1, victim=6, assists=(2, 3), bounty=300, shutdown=150),
        dict(t=131_000, x=None, y=None, killer=0, victim=3),                           # an execution, no position
    ], events=[
        {'type': 'CHAMPION_KILL', 'timestamp': 131_000, 'victimId': 4},                # a bare event: no killer, position, assists, bounty
        {'type': 'ELITE_MONSTER_KILL', 'timestamp': 132_000, 'killerId': 1},           # not a kill of a champion
        {'type': 'CHAMPION_KILL', 'timestamp': 140_000, 'victimId': 9, 'killerId': 2, 'position': {'x': nan, 'y': 5},
         'assistingParticipantIds': [0, 1, 11, 3, 3], 'bounty': None, 'shutdownBounty': None},
        {'type': 'CHAMPION_KILL', 'timestamp': 150_000, 'victimId': 7, 'killerId': 0, 'position': {'x': 100, 'y': 200}, 'bounty': 100},
    ])
    k = units.kill_table(rec)
    assert k.t.tolist() == [130_000, 131_000, 131_000, 140_000, 150_000]            # record order, also within the same ms
    assert k.killer.tolist() == [1, 0, 0, 2, 0] and k.victim.tolist() == [6, 3, 4, 9, 7]
    assert k.pos_missing.tolist() == [False, True, True, True, False]
    assert (k.x[[0, 4]].tolist(), k.y[[0, 4]].tolist()) == ([1_000.0, 100.0], [2_000.0, 200.0])
    assert np.isnan(k.x[[1, 2, 3]]).all() and np.isnan(k.y[[1, 2, 3]]).all()      # x or y missing or NaN: both NaN
    assert k.amask.tolist() == [0b110, 0, 0, 0b101, 0]          # assisters 2, 3 / none / 1, 3 (the ids 0 and 11 and the repeat are ignored)
    assert k.bounty.tolist() == [300, 300, 0, 0, 100] and k.shutdown.tolist() == [150, 0, 0, 0, 0]
    dtypes = dict(t='i8', x='f8', y='f8', killer='i1', victim='i1', amask='i2', bounty='i4', shutdown='i4', pos_missing='b1')
    assert [f.name for f in dataclasses.fields(units.KillTable)] == list(dtypes)
    assert {c: getattr(k, c).dtype.str[1:] for c in dtypes} == dtypes


# ---- fix round 1: one error type for malformed ids, a dense link case, the helpers Task 4 uses -------------------------------------
def kill_event(**changes):
    """A valid CHAMPION_KILL event (killer 1, victim 6) with the given fields replaced; a value of ``...`` leaves the field out."""
    e = {'type': 'CHAMPION_KILL', 'timestamp': 140_000, 'killerId': 1, 'victimId': 6, **changes}
    return {name: value for name, value in e.items() if value is not ...}


BAD_KILLS = {
    'killerId 200': kill_event(killerId=200), 'killerId -1': kill_event(killerId=-1), 'killerId 11': kill_event(killerId=11),
    'killerId None': kill_event(killerId=None), 'killerId True': kill_event(killerId=True), 'killerId 1.0': kill_event(killerId=1.0),
    'victimId missing': kill_event(victimId=...), 'victimId None': kill_event(victimId=None), 'victimId 0': kill_event(victimId=0),
    'victimId 11': kill_event(victimId=11), 'victimId True': kill_event(victimId=True), 'victimId "6"': kill_event(victimId='6'),
    'assist "3"': kill_event(assistingParticipantIds=['3']), 'assist 3.0': kill_event(assistingParticipantIds=[2, 3.0]),
    'assist None': kill_event(assistingParticipantIds=[2, None]), 'assist True': kill_event(assistingParticipantIds=[True]),
    'assists not a list': kill_event(assistingParticipantIds=3),
}


@pytest.mark.parametrize('event', list(BAD_KILLS.values()), ids=list(BAD_KILLS))
def test_malformed_ids_raise_match_format_error(event):
    rec = synth.make_record(T_ms=900_000, events=[event])
    with pytest.raises(reader.MatchFormatError) as raised:
        units.kill_table(rec)
    assert f'(patch {rec.patch})' in str(raised.value) and rec.key not in str(raised.value)       # the patch is named, the key never
    assert (raised.value.key, raised.value.patch) == (rec.key, rec.patch)


def test_ids_at_the_edges_of_their_ranges_are_accepted():
    rec = synth.make_record(T_ms=900_000, events=[
        kill_event(killerId=0, victimId=1), kill_event(killerId=10, victimId=1), kill_event(killerId=1, victimId=10),
        kill_event(killerId=...),                                                  # no killerId: an execution, as in the legacy code
        kill_event(assistingParticipantIds=None), kill_event(assistingParticipantIds=[]),
        kill_event(assistingParticipantIds=[0, 10, 11, -4, 99]),                   # integers outside 1-10 are ignored, 10 is a participant
    ])
    k = units.kill_table(rec)
    assert k.killer.tolist() == [0, 10, 1, 0, 1, 1, 1] and k.victim.tolist() == [1, 1, 10, 6, 6, 6, 6]
    assert k.amask.tolist() == [0, 0, 0, 0, 0, 0, 1 << 9]


def test_id_messages_default_to_a_champion_kill_and_take_a_label():
    # the id rules are shared with the objective table of c_ept.targets, which names its own event: the wording of kill_table stays
    rec = synth.make_record(T_ms=900_000)
    cases = [(units._event_id, ({}, 'victimId', 1, None), 'has no victimId'),
             (units._event_id, ({'killerId': 'x'}, 'killerId', 0, 0), 'has a killerId of type str, not an integer'),
             (units._event_id, ({'killerId': 11}, 'killerId', 0, 0), 'has killerId 11, not in 0-10'),
             (units._assist_mask, ({'assistingParticipantIds': 3},), 'has assistingParticipantIds of type int, not a list'),
             (units._assist_mask, ({'assistingParticipantIds': ['3']},),
              'has an assistingParticipantIds entry of type str, not an integer')]
    for fn, args, rest in cases:
        for extra, label in (((), 'a CHAMPION_KILL'), (('an ELITE_MONSTER_KILL event',), 'an ELITE_MONSTER_KILL event')):
            with pytest.raises(reader.MatchFormatError) as raised:
                fn(rec, *args, *extra)
            assert raised.value.reason == f'{label} {rest}' and (raised.value.key, raised.value.patch) == (rec.key, rec.patch)
            assert str(raised.value) == f'{label} {rest} (patch {rec.patch})'


def test_link_dense_boundary_cases_match_bruteforce_and_prototype(proto):
    steps, offsets = (0, 15_000, 15_001), ((2_000, 0), (1_200, 1_600), (2_001, 0))
    # four kills held together by steps of exactly 15,000 ms and offsets of exactly 2,000 units; the fifth is 15,001 ms and
    # 2,001 units from the fourth, the sixth has no position
    cases = [(np.array([0, 0, 15_000, 30_000, 45_001, 45_001]), np.array([0, 2_000, 3_200, 5_200, 7_201, np.nan]),
              np.array([0, 0, 1_600, 1_600, 1_600, np.nan]))]
    rng = np.random.default_rng(7)
    for _ in range(300):
        n = int(rng.integers(6, 24))
        t = np.cumsum(rng.choice(steps, n))
        xy = np.zeros((n, 2))
        for i in range(1, n):                                    # every kill sits one boundary offset (either sign) from an earlier kill
            dx, dy = offsets[int(rng.integers(len(offsets)))]
            xy[i] = xy[int(rng.integers(i))] + (dx * rng.choice((-1, 1)), dy * rng.choice((-1, 1)))
        xy[int(rng.integers(n))] = np.nan                        # one kill without a position
        cases.append((t, xy[:, 0], xy[:, 1]))
    largest = []
    for t, x, y in cases:
        got = units.link_components(t, x, y)
        assert as_sets(got) == brute_force(t, x, y)
        if proto is not None:
            assert [list(c) for c in got] == proto.link_components(t, x, y, 15_000)
        largest.append(max(len(c) for c in got))
    assert largest[0] == 4 and max(largest) >= 4


def test_kill_people_mask_bits():
    k = kt([(200_000, 0, 1, 6, (2, 3)), (201_000, 0, 0, 3, ()), (202_000, 0, 7, 2, (8,))])
    mask = units.kill_people_mask(k)
    assert mask.dtype == np.int16 and mask.tolist() == [0b10_0111, 0b100, 0b1100_0010]       # {1, 2, 3, 6}, {3}, {2, 7, 8}
    odd = dataclasses.replace(kt([(200_000, 0, 11, 6, (2,))]), amask=np.array([2 | (1 << 12)], dtype=np.int16))
    assert units.kill_people_mask(odd).tolist() == [(1 << 5) | (1 << 1)]       # killer 11 and a bit above participant 10 credit nobody
    empty = units.kill_people_mask(kt([]))
    assert empty.shape == (0,) and empty.dtype == np.int16


def test_unit_people_equal_popcounts_of_kill_people_mask():
    n_units = 0
    for seed in range(120):
        rec = synth.records_from_seed(f'SYN_{seed}', '15.18')
        k = units.kill_table(rec)
        u = units.build_units(k, rec.team, rec.T_ms)
        mask = units.kill_people_mask(k)
        blue = sum(1 << i for i in range(10) if rec.team[i] == 100)
        red = sum(1 << i for i in range(10) if rec.team[i] == 200)
        for a, g in enumerate(u.comp):
            people = int(np.bitwise_or.reduce(mask[g]))
            assert (u.nb[a], u.nr[a]) == (bin(people & blue).count('1'), bin(people & red).count('1'))
        n_units += len(u.comp)
    assert n_units > 1_000


def test_event_xy_rule():
    assert units.event_xy({'position': {'x': 3, 'y': 4.5}}) == (3.0, 4.5)
    for e in ({}, {'position': None}, {'position': [1, 2]}, {'position': {'x': 1}}, {'position': {'y': 1}},
              {'position': {'x': float('nan'), 'y': 1}}, {'position': {'x': 1, 'y': float('inf')}}, {'position': {'x': 'a', 'y': 1}},
              {'position': {'x': None, 'y': 1}}):
        x, y = units.event_xy(e)
        assert np.isnan(x) and np.isnan(y)                                              # missing, not a dict or not finite: both NaN


def test_mark_layout_is_checked_against_config(monkeypatch):
    monkeypatch.setattr(C, 'N_MARKS', 53)
    with pytest.raises(RuntimeError, match='N_MARKS'):
        runpy.run_path(units.__file__)                      # the module text in a fresh namespace: the module under test is not touched
