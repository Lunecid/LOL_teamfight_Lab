"""Tests of c_ept.targets: the kill gold K, the attribution of objectives to units (G3) and their conversions (G4). Synthetic data only."""
import dataclasses

import numpy as np
import pytest

from c_ept import config as C
from c_ept import reader, targets, units
from c_ept.tests import synth
from c_ept.tests.synth import TEAM, kt

F_A = 200_000                                                      # first kill of the single unit "A" of the window tests
TAU_A, DELTA_A = F_A - C.TAU_LEAD_MS, F_A + C.DELTA_AFTER_F_MS     # 185,000 and 220,000
OBJECTIVE_TYPES = ('ELITE_MONSTER_KILL', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED')
N_CELLS = 2 * len(C.O_KINDS)                                       # 18


# ---- helpers ----------------------------------------------------------------------------------------------------------------
def pids(side):
    """Participant ids of a team: side 0 blue (1-5), side 1 red (6-10)."""
    return list(range(1 + 5 * side, 6 + 5 * side))


def kill(t, x, killer, victim, assists=(), y=0, bounty=300, shutdown=0):
    """A kill for ``synth.make_record`` (x or y None: no position)."""
    return dict(t=t, x=x, y=y, killer=killer, victim=victim, assists=assists, bounty=bounty, shutdown=shutdown)


def objective(t, typ='ELITE_MONSTER_KILL', *, cred=(), xy=(0, 0), team=100, monster=None, sub=None, building=None, tower=None):
    """An objective for ``synth.make_record``. ``cred``: the credited participants, the first is the killer, the others assist.

    ``team`` is the killerTeamId of an ELITE_MONSTER_KILL and the teamId (the owner who lost it) of a building or a plate; None leaves
    it out. An ELITE_MONSTER_KILL without ``monster`` is a DRAGON (FIRE_DRAGON), a BUILDING_KILL without ``building`` an outer tower.
    """
    if typ == 'ELITE_MONSTER_KILL' and monster is None:
        monster, sub = 'DRAGON', 'FIRE_DRAGON'
    if typ == 'BUILDING_KILL' and building is None:
        building, tower = 'TOWER_BUILDING', 'OUTER_TURRET'
    cred = list(cred)
    return dict(t=t, type=typ, x=xy[0], y=xy[1], killer=cred[0] if cred else 0, assists=tuple(cred[1:]), team=team, monster=monster,
                sub=sub, building=building, tower=tower)


def pipeline(rec):
    """(KillTable, UnitTable, ObjectiveTable) of a record."""
    k = units.kill_table(rec)
    u = units.build_units(k, rec.team, rec.T_ms)
    return k, u, targets.objective_table(rec, u, k)


def unit_name(o, i, names='AB'):
    """The name of the unit the objective ``i`` is attached to, None when it is attached to none."""
    return None if o.unit[i] < 0 else names[int(o.unit[i])]


def attach(cred, xy, t=210_000):
    """Two units active at ``t``: A (people {1, 6}, its kill at (0, 0)) and B (people {2, 3, 7}, its kill at (5,000, 0)). Which one
    gets an objective that credits ``cred`` at ``xy``?"""
    rec = synth.make_record(T_ms=600_000, kills=[kill(200_000, 0, 1, 6), kill(203_000, 5_000, 2, 7, (3,))],
                            objectives=[objective(t, cred=sorted(cred), xy=xy)])
    _, u, o = pipeline(rec)
    assert len(u.f) == 2 and u.tau[0] < t < u.delta[0] and u.tau[1] < t < u.delta[1]
    return unit_name(o, 0)


def attach_at(t):
    """One unit A (first kill F_A, people {1, 6}) and an objective of participant 1 at its position at time ``t``."""
    _, u, o = pipeline(synth.make_record(T_ms=600_000, kills=[kill(F_A, 0, 1, 6)], objectives=[objective(t, cred=[1])]))
    assert (u.tau[0], u.delta[0]) == (TAU_A, DELTA_A)
    return unit_name(o, 0)


def taker(*, elite_team=None, building_team=None, plate_team=None):
    """The taker of one objective: an ELITE_MONSTER_KILL with this killerTeamId, a tower or a plate with this teamId."""
    given = [(typ, team) for typ, team in zip(OBJECTIVE_TYPES, (elite_team, building_team, plate_team)) if team is not None]
    assert len(given) == 1
    _, _, o = pipeline(synth.make_record(objectives=[objective(150_000, given[0][0], team=given[0][1])]))
    return int(o.taker[0])


def conv_case(delta, T, t_obj=None):
    """One unit that ends at ``delta`` (one kill at delta - 20,000) and one objective of nobody (no champion credit) in a match of
    length ``T``, 5,000 ms after delta or at T. Returns what ``targets.conversions`` returns."""
    t_obj = min(delta + 5_000, T) if t_obj is None else t_obj
    rec = synth.make_record(T_ms=T, kills=[kill(delta - C.DELTA_AFTER_F_MS, 0, 1, 6)], objectives=[objective(t_obj)])
    _, u, o = pipeline(rec)
    assert u.delta[0] == delta and o.unit[0] == -1
    return targets.conversions(o, u, rec.T_ms)


def conv_case_two_objectives_one_attached():
    """Unit A (index 0) ends at 220,000. Unit B (index 1, far from A) lasts from 235,000 to 270,000 and claims the first objective
    (t = 240,000, which also lies in A's conversion window); the second objective (t = 241,000) credits nobody."""
    rec = synth.make_record(T_ms=600_000, kills=[kill(200_000, 0, 1, 6), kill(250_000, 6_000, 2, 7)], objectives=[
        objective(240_000, cred=[2], xy=(6_000, 0)), objective(241_000)])
    _, u, o = pipeline(rec)
    assert list(u.delta) == [220_000, 270_000] and o.unit.tolist() == [1, -1]
    return targets.conversions(o, u, rec.T_ms)


def conv_case_two_units(delta_a, delta_b, t_obj):
    """Units A and B (6,000 u apart, ends delta_a < delta_b) and one objective of nobody at ``t_obj``."""
    rec = synth.make_record(T_ms=900_000, kills=[kill(delta_a - 20_000, 0, 1, 6), kill(delta_b - 20_000, 6_000, 2, 7)],
                            objectives=[objective(t_obj)])
    _, u, o = pipeline(rec)
    assert list(u.delta) == [delta_a, delta_b] and o.unit[0] == -1
    return targets.conversions(o, u, rec.T_ms)


_MIX = (    # (type, monster, sub, building, tower): the objectives of a random match; the last two are unknown to the kind table
    ('ELITE_MONSTER_KILL', 'DRAGON', 'FIRE_DRAGON', None, None), ('ELITE_MONSTER_KILL', 'DRAGON', 'ELDER_DRAGON', None, None),
    ('ELITE_MONSTER_KILL', 'HORDE', None, None, None), ('ELITE_MONSTER_KILL', 'RIFTHERALD', None, None, None),
    ('ELITE_MONSTER_KILL', 'BARON_NASHOR', None, None, None), ('ELITE_MONSTER_KILL', 'ATAKHAN', None, None, None),
    ('BUILDING_KILL', None, None, 'TOWER_BUILDING', 'OUTER_TURRET'), ('BUILDING_KILL', None, None, 'TOWER_BUILDING', 'BASE_TURRET'),
    ('BUILDING_KILL', None, None, 'INHIBITOR_BUILDING', None), ('TURRET_PLATE_DESTROYED', None, None, None, None),
    ('ELITE_MONSTER_KILL', 'SOMETHING_NEW', None, None, None), ('BUILDING_KILL', None, None, 'SOMETHING_NEW', None))


def synth_many(seed, n_units=30, n_obj=80):
    """A random match of ``n_units`` engagements, 30 s apart (so each is one unit), and ``n_obj`` objectives. Returns (record, units,
    kills). Most objectives are close to an engagement in time and place and credit players of the team that took them, so that some
    attach, some fall into a conversion window and some are left over; a few have no champion, no position, no team or a kind that
    the kind table does not know."""
    rng = np.random.default_rng(seed)
    kills, spots = [], []
    for i in range(n_units):
        f, cx, cy = 100_000 + 30_000 * i, *rng.uniform(1_500.0, 13_000.0, 2)
        spots.append((f, cx, cy))
        side, t = int(rng.integers(0, 2)), f
        for _ in range(int(rng.integers(1, 4))):
            killer = int(rng.choice(pids(side)))
            helpers = [p for p in pids(side) if p != killer]
            assists = tuple(sorted(int(a) for a in rng.choice(helpers, int(rng.integers(0, 3)), replace=False)))
            kills.append(kill(t, int(cx + rng.integers(-300, 300)), killer, int(rng.choice(pids(1 - side))), assists,
                              y=int(cy + rng.integers(-300, 300))))
            t += int(rng.integers(1_000, 5_000))
    objectives = []
    for _ in range(n_obj):
        typ, monster, sub, building, tower = _MIX[int(rng.integers(len(_MIX)))]
        side = int(rng.integers(0, 2))                                             # the team that takes the objective
        if rng.random() < 0.8:
            f, cx, cy = spots[int(rng.integers(n_units))]
            t, xy = f + int(rng.integers(-20_000, 90_000)), (int(cx + rng.normal(0, 700)), int(cy + rng.normal(0, 700)))
        else:
            t, xy = int(rng.integers(30_000, 1_190_000)), (int(rng.integers(0, 14_875)), int(rng.integers(0, 14_875)))
        cred = () if rng.random() < 0.2 else tuple(int(p) for p in rng.choice(pids(side), int(rng.integers(1, 4)), replace=False))
        team = 100 + 100 * side if typ == 'ELITE_MONSTER_KILL' else 200 - 100 * side
        objectives.append(objective(t, typ, cred=cred, xy=(None, None) if rng.random() < 0.05 else xy,
                                    team=None if rng.random() < 0.05 else team, monster=monster, sub=sub, building=building, tower=tower))
    rec = synth.make_record(kills=kills, objectives=objectives)
    k = units.kill_table(rec)
    u = units.build_units(k, rec.team, rec.T_ms)
    assert len(u.f) == n_units
    return rec, u, k


def assert_accounting(o, u, T_ms):
    """The accounting identity of every objective with a known taker; returns (O summed over the units, conv, conv_unit)."""
    conv, _, cu = targets.conversions(o, u, T_ms)
    assert not np.any((o.unit >= 0) & (cu >= 0))                                  # no objective is both attached and converted
    cells = sum((targets.unit_O(o, i) for i in range(len(u.f))), np.zeros(N_CELLS, np.int64))
    known = o.taker >= 0
    assert cells.sum() + conv.sum() + int(np.sum(known & (o.unit < 0) & (cu < 0))) == int(known.sum())
    assert cells.sum() == int(np.sum(o.unit >= 0)) and conv.sum() == int(np.sum(cu >= 0))      # the tables count exactly those objects
    return cells, conv, cu


# DRAGON, ELDER (monsterType), ELDER (monsterSubType), HORDE, RIFTHERALD, BARON_NASHOR, ATAKHAN, TOWER, INHIBITOR, plate, none of these
KIND_CASES = [
    {'type': 'ELITE_MONSTER_KILL', 'monsterType': 'DRAGON', 'monsterSubType': 'FIRE_DRAGON'},
    {'type': 'ELITE_MONSTER_KILL', 'monsterType': 'ELDER_DRAGON'},
    {'type': 'ELITE_MONSTER_KILL', 'monsterType': 'DRAGON', 'monsterSubType': 'ELDER_DRAGON'},
    {'type': 'ELITE_MONSTER_KILL', 'monsterType': 'HORDE'},
    {'type': 'ELITE_MONSTER_KILL', 'monsterType': 'RIFTHERALD'},
    {'type': 'ELITE_MONSTER_KILL', 'monsterType': 'BARON_NASHOR'},
    {'type': 'ELITE_MONSTER_KILL', 'monsterType': 'ATAKHAN'},
    {'type': 'BUILDING_KILL', 'buildingType': 'TOWER_BUILDING', 'towerType': 'OUTER_TURRET'},
    {'type': 'BUILDING_KILL', 'buildingType': 'INHIBITOR_BUILDING'},
    {'type': 'TURRET_PLATE_DESTROYED', 'laneType': 'TOP_LANE'},
    {'type': 'ELITE_MONSTER_KILL', 'monsterType': 'SOMETHING_NEW'},
]


# ---- the nine tests of the brief ----------------------------------------------------------------------------------------------
def test_K_counts_and_gold():
    k = kt([(200_000, 0, 1, 6, (2, 3), 300, 150), (205_000, 0, 7, 2, (), 400, 0), (208_000, 0, 0, 4, (), 0, 0)])
    comp = np.arange(3)
    assert list(targets.unit_K(k, comp, TEAM)) == [1, 1, 300, 400, 150, 0, 2, 0]
    assert list(targets.unit_exec(k, comp, TEAM)) == [1, 0]
    assert list(targets.unit_K(k, comp, TEAM, upto_ms=200_000)) == [1, 0, 300, 0, 150, 0, 2, 0]


def test_attribution_overlap_then_distance_then_order():
    assert attach(cred={1, 2, 3}, xy=(0, 0)) == 'B'          # overlap 2 > 1, although A is much closer
    assert attach(cred={1, 2}, xy=(4_000, 0)) == 'B'         # equal overlap, B is closer
    assert attach(cred={1, 2}, xy=(2_500, 0)) == 'A'         # equal overlap and distance: the earlier unit
    assert attach(cred={4, 9}, xy=(0, 0)) is None            # no overlap


def test_attribution_window_half_open():
    assert attach_at(t=TAU_A) == 'A' and attach_at(t=DELTA_A) is None
    assert attach_at(t=TAU_A - 1) is None and attach_at(t=DELTA_A - 1) == 'A'


def test_objective_kind_mapping():
    assert [targets.objective_kind(e) for e in KIND_CASES] == [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, -1]
    names = [C.O_KINDS[i] if i >= 0 else None for i in map(targets.objective_kind, KIND_CASES)]       # the codes are indices of O_KINDS
    assert names == ['dragon', 'elder', 'elder', 'grubs', 'herald', 'baron', 'atakhan', 'tower', 'inhibitor', 'plate', None]


def test_taker_rules():
    assert taker(elite_team=200) == 1 and taker(building_team=100) == 1 and taker(plate_team=200) == 0
    assert taker(elite_team=100) == 0 and taker(building_team=200) == 0 and taker(plate_team=100) == 1


def test_conversion_exposure_cut_at_game_end():
    conv, expo, _ = conv_case(delta=590_000, T=600_000)
    assert expo[0] == 10_000 and conv[0].sum() == 1          # the objective 5,000 ms after delta is inside the shortened window
    conv, expo, _ = conv_case(delta=600_000, T=600_000)
    assert expo[0] == 0 and conv[0].sum() == 0


def test_conversion_counts_only_unattributed():
    conv, _, cu = conv_case_two_objectives_one_attached()
    assert conv[0].sum() == 1 and conv[1].sum() == 0 and cu.tolist() == [-1, 0]


def test_conversion_assigned_to_latest_preceding_delta():
    # unit A (delta 300,000) and unit B (delta 320,000), an objective of nobody at t = 340,000: B only
    conv, _, cu = conv_case_two_units(delta_a=300_000, delta_b=320_000, t_obj=340_000)
    assert (conv[0].sum(), conv[1].sum(), cu[0]) == (0, 1, 1)
    conv, _, cu = conv_case_two_units(delta_a=300_000, delta_b=320_000, t_obj=310_000)       # B has not ended yet: A
    assert (conv[0].sum(), conv[1].sum(), cu[0]) == (1, 0, 0)


def test_accounting_identity():
    rec, u, k = synth_many(seed=3)                           # 30 engagements, 80 random objectives
    obj = targets.objective_table(rec, u, k)
    conv, _, cu = targets.conversions(obj, u, rec.T_ms)
    assert not np.any((obj.unit >= 0) & (cu >= 0))
    cells = sum(targets.unit_O(obj, i) for i in range(len(u.f)))
    known = obj.taker >= 0
    assert cells.sum() + conv.sum() + int(np.sum(known & (obj.unit < 0) & (cu < 0))) == int(known.sum())
    # the three outcomes all occur, so that the identity is not satisfied trivially
    assert int(np.sum(obj.unit >= 0)) > 5 and int(np.sum(cu >= 0)) > 5 and int(np.sum(known & (obj.unit < 0) & (cu < 0))) > 5


# ---- beyond the brief ---------------------------------------------------------------------------------------------------------
# K: only the rows of the unit, only champion killers, inclusive upto_ms, executions pay no team (not even their assisters).
def test_K_rows_of_the_unit_champion_killers_and_upto():
    k = kt([(200_000, 0, 1, 6, (2, 3), 300, 150),          # blue kill, two assisters
            (205_000, 100, 0, 3, (7, 8), 500, 100),         # execution of blue victim 3 with two red assisters: pays nobody
            (210_000, 200, 7, 2, (8,), 400, 50),            # red kill, one assister
            (215_000, 300, 2, 9, (1, 3, 4), 250, 0)])       # blue kill, three assisters
    assert list(targets.unit_K(k, np.array([0, 3]), TEAM)) == [2, 0, 550, 0, 150, 0, 5, 0]       # only the rows of this unit
    assert list(targets.unit_K(k, np.arange(4), TEAM)) == [2, 1, 550, 400, 150, 50, 5, 1]
    assert list(targets.unit_K(k, np.arange(4), TEAM, upto_ms=209_999)) == [1, 0, 300, 0, 150, 0, 2, 0]
    assert list(targets.unit_K(k, np.arange(4), TEAM, upto_ms=210_000)) == [1, 1, 300, 400, 150, 50, 2, 1]      # upto_ms is inclusive
    assert list(targets.unit_K(k, np.arange(4), TEAM, upto_ms=None)) == list(targets.unit_K(k, np.arange(4), TEAM))
    assert list(targets.unit_K(k, np.arange(4), TEAM, upto_ms=199_999)) == [0] * 8
    assert list(targets.unit_exec(k, np.arange(4), TEAM)) == [1, 0]                              # victim 3 is on blue
    assert list(targets.unit_exec(k, np.array([0, 2]), TEAM)) == [0, 0]
    e = kt([(200_000, 0, 0, 8), (201_000, 0, 0, 1), (202_000, 0, 0, 9)])
    assert list(targets.unit_exec(e, np.arange(3), TEAM)) == [1, 2]                              # (blue victims, red victims)
    assert targets.K_FIELDS == ('kills_b', 'kills_r', 'bounty_b', 'bounty_r', 'shutdown_b', 'shutdown_r', 'assists_b', 'assists_r')


def test_K_bits_above_participant_10_are_not_assisters():
    # 0x1002 (participant 2 and bit 12) and 0x8002 (participant 2 and the sign bit): one assister each
    k = dataclasses.replace(kt([(200_000, 0, 1, 6), (201_000, 0, 1, 7)]), amask=np.array([2 | (1 << 12), -32766], dtype=np.int16))
    assert list(targets.unit_K(k, np.arange(2), TEAM)) == [2, 0, 600, 0, 0, 0, 2, 0]


def test_K_needs_ten_teams_of_blue_and_red():
    k = kt([(200_000, 0, 1, 6)])
    for bad in (np.array([100] * 9), np.array([100] * 5 + [300] * 5), np.array([100] * 5 + [0] * 5)):
        with pytest.raises(ValueError, match='team'):
            targets.unit_K(k, np.arange(1), bad)
        with pytest.raises(ValueError, match='team'):
            targets.unit_exec(k, np.arange(1), bad)


# objective_table: columns, order, every time of the match, the unit rule of ids
def test_objective_table_columns_and_order():
    rec = synth.make_record(T_ms=900_000, objectives=[
        objective(10_000, cred=[2, 1, 5], xy=(9_837, 4_397), team=100, monster='DRAGON', sub='FIRE_DRAGON'),            # blue dragon
        objective(20_000, 'BUILDING_KILL', cred=[3], xy=(100, 200), team=200, building='TOWER_BUILDING',
                  tower='INNER_TURRET'),                                                                      # red tower, blue takes it
        objective(30_000, 'TURRET_PLATE_DESTROYED', cred=[7, 8], xy=(5, 6), team=100),                                  # plate of blue lost
        objective(40_000, xy=(None, None), team=None, monster='HORDE'),                                                 # grubs: no position
        objective(900_000, 'BUILDING_KILL', cred=[10], xy=(7, 7), team=100, building='INHIBITOR_BUILDING'),             # at the game end
    ], events=[{'type': 'ITEM_PURCHASED', 'timestamp': 15_000, 'participantId': 1, 'itemId': 1055, 'killerId': 200},
               {'type': 'WARD_PLACED', 'timestamp': 16_000, 'creatorId': 3}, {'type': 'LEVEL_UP', 'timestamp': 17_000, 'level': 2}])
    k, u, o = pipeline(rec)
    assert len(u.f) == 0
    assert o.t.tolist() == [10_000, 20_000, 30_000, 40_000, 900_000]                # no 120 s filter; other event types are left out
    assert o.kind.tolist() == [0, 6, 8, 2, 7] and o.taker.tolist() == [0, 0, 1, -1, 1]
    assert o.has_pos.tolist() == [True, True, True, False, True]
    assert o.x[[0, 1, 2, 4]].tolist() == [9_837.0, 100.0, 5.0, 7.0] and o.y[[0, 1, 2, 4]].tolist() == [4_397.0, 200.0, 6.0, 7.0]
    assert np.isnan(o.x[3]) and np.isnan(o.y[3])
    assert o.cred.tolist() == [0b10011, 0b100, 1 << 6, 0, 1 << 9]                  # killer + assisters; the plate only the killer
    assert o.unit.tolist() == [-1] * 5 and o.sub.tolist() == [-1, 1, -1, -1, -1]
    dtypes = dict(t='i8', kind='i1', taker='i1', x='f8', y='f8', has_pos='b1', cred='i2', unit='i4', sub='i1')
    assert [f.name for f in dataclasses.fields(targets.ObjectiveTable)] == list(dtypes)
    assert {c: getattr(o, c).dtype.str[1:] for c in dtypes} == dtypes


def test_unusable_position_is_no_position():
    ok = {'x': 3, 'y': 4}
    positions = [ok, None, [1, 2], {'x': 1}, {'y': 1}, {'x': float('nan'), 'y': 1}, {'x': 1, 'y': float('inf')}, {'x': 'a', 'y': 1}]
    events = [{'type': 'ELITE_MONSTER_KILL', 'timestamp': 150_000 + i, 'killerId': 1, 'killerTeamId': 100, 'monsterType': 'DRAGON',
               'position': p} for i, p in enumerate(positions)]
    events.append({'type': 'ELITE_MONSTER_KILL', 'timestamp': 150_100, 'killerId': 1, 'killerTeamId': 100, 'monsterType': 'DRAGON'})
    _, _, o = pipeline(synth.make_record(T_ms=900_000, events=events))
    assert o.has_pos.tolist() == [True] + [False] * 8
    assert np.isnan(o.x[1:]).all() and np.isnan(o.y[1:]).all() and (o.x[0], o.y[0]) == (3.0, 4.0)


def obj_event(typ, **changes):
    """A valid raw objective event of type ``typ`` with the given fields replaced; a value of ``...`` leaves the field out."""
    e = {'type': typ, 'timestamp': 140_000, 'killerId': 1, 'position': {'x': 100, 'y': 100}}
    e |= {'ELITE_MONSTER_KILL': {'killerTeamId': 100, 'monsterType': 'DRAGON'},
          'BUILDING_KILL': {'teamId': 200, 'buildingType': 'TOWER_BUILDING'}, 'TURRET_PLATE_DESTROYED': {'teamId': 200}}[typ]
    e |= changes
    return {name: value for name, value in e.items() if value is not ...}


BAD_KILLER = {
    'killerId 200': dict(killerId=200), 'killerId -1': dict(killerId=-1), 'killerId 11': dict(killerId=11),
    'killerId None': dict(killerId=None), 'killerId True': dict(killerId=True), 'killerId 1.0': dict(killerId=1.0),
    'killerId "3"': dict(killerId='3'),
}
BAD_ASSISTS = {
    'assist "3"': dict(assistingParticipantIds=['3']), 'assist 3.0': dict(assistingParticipantIds=[2, 3.0]),
    'assist None': dict(assistingParticipantIds=[2, None]), 'assist True': dict(assistingParticipantIds=[True]),
    'assists not a list': dict(assistingParticipantIds=3), 'assists a string': dict(assistingParticipantIds='12'),
}


def assert_format_error(rec, typ):
    with pytest.raises(reader.MatchFormatError) as raised:
        pipeline(rec)
    message = str(raised.value)
    assert f'(patch {rec.patch})' in message and rec.key not in message and typ in message      # the patch and the event, never the key
    assert (raised.value.key, raised.value.patch) == (rec.key, rec.patch)


@pytest.mark.parametrize('typ', OBJECTIVE_TYPES)
@pytest.mark.parametrize('changes', list(BAD_KILLER.values()), ids=list(BAD_KILLER))
def test_malformed_killer_id_raises_match_format_error(typ, changes):
    assert_format_error(synth.make_record(T_ms=900_000, events=[obj_event(typ, **changes)]), typ)


@pytest.mark.parametrize('typ', OBJECTIVE_TYPES)       # the assists of a plate are parsed too, although they are not credited
@pytest.mark.parametrize('changes', list(BAD_ASSISTS.values()), ids=list(BAD_ASSISTS))
def test_malformed_assists_raise_match_format_error(typ, changes):
    assert_format_error(synth.make_record(T_ms=900_000, events=[obj_event(typ, **changes)]), typ)


def test_ids_at_the_edges_are_accepted_and_plates_carry_only_the_killer():
    rec = synth.make_record(T_ms=900_000, events=[
        obj_event('ELITE_MONSTER_KILL', killerId=0), obj_event('ELITE_MONSTER_KILL', killerId=10),
        obj_event('ELITE_MONSTER_KILL', killerId=...),                                   # no killerId: no champion
        obj_event('BUILDING_KILL', killerId=3, assistingParticipantIds=None), obj_event('BUILDING_KILL', assistingParticipantIds=[]),
        obj_event('BUILDING_KILL', killerId=0, assistingParticipantIds=[0, 10, 11, -4, 99, 2]),   # integers outside 1-10 are ignored
        obj_event('TURRET_PLATE_DESTROYED', killerId=4, assistingParticipantIds=[1, 2]),          # a plate is credited to its killer only
        obj_event('TURRET_PLATE_DESTROYED', killerId=5, assistingParticipantIds=[0, 11, 99]),     # ... whatever valid assists it carries
        obj_event('TURRET_PLATE_DESTROYED', killerId=6, assistingParticipantIds=None),
        obj_event('TURRET_PLATE_DESTROYED', killerId=...),
    ])
    _, _, o = pipeline(rec)
    assert o.cred.tolist() == [0, 1 << 9, 0, 1 << 2, 1 << 0, (1 << 9) | (1 << 1), 1 << 3, 1 << 4, 1 << 5, 0]


def test_other_event_types_are_not_looked_at():
    rec = synth.make_record(T_ms=900_000, events=[
        {'type': 'ITEM_PURCHASED', 'timestamp': 100_000, 'killerId': 'x', 'assistingParticipantIds': 7, 'monsterType': 'DRAGON'},
        {'type': 'WARD_KILL', 'timestamp': 100_001, 'killerId': 99, 'teamId': 100}, {'timestamp': 100_002},
        {'type': None, 'timestamp': 100_003}, {'type': ['ELITE_MONSTER_KILL'], 'timestamp': 100_004}])
    _, _, o = pipeline(rec)
    assert len(o.t) == 0


def test_kind_minus_one_cases():
    odd = [
        {'type': 'ELITE_MONSTER_KILL'}, {'type': 'ELITE_MONSTER_KILL', 'monsterType': None},
        {'type': 'ELITE_MONSTER_KILL', 'monsterType': 7}, {'type': 'ELITE_MONSTER_KILL', 'monsterType': ['DRAGON']},
        {'type': 'ELITE_MONSTER_KILL', 'monsterType': 'dragon'}, {'type': 'BUILDING_KILL'},
        {'type': 'BUILDING_KILL', 'buildingType': 'NEXUS_BUILDING'}, {'type': 'BUILDING_KILL', 'buildingType': {}},
        {'type': 'CHAMPION_KILL', 'monsterType': 'DRAGON'}, {'type': 'ITEM_PURCHASED', 'buildingType': 'TOWER_BUILDING'},
        {}, {'type': None},
    ]
    assert [targets.objective_kind(e) for e in odd] == [-1] * len(odd)
    # a DRAGON with another or no sub type is a dragon; the elder rule needs the sub type on a DRAGON, or the monster type itself
    assert targets.objective_kind({'type': 'ELITE_MONSTER_KILL', 'monsterType': 'DRAGON'}) == 0
    assert targets.objective_kind({'type': 'ELITE_MONSTER_KILL', 'monsterType': 'DRAGON', 'monsterSubType': None}) == 0
    assert targets.objective_kind({'type': 'ELITE_MONSTER_KILL', 'monsterType': 'HORDE', 'monsterSubType': 'ELDER_DRAGON'}) == 2
    # a monster type on a building event, or a building type on a monster event, does not count
    assert targets.objective_kind({'type': 'BUILDING_KILL', 'monsterType': 'DRAGON'}) == -1
    assert targets.objective_kind({'type': 'ELITE_MONSTER_KILL', 'buildingType': 'TOWER_BUILDING'}) == -1


@pytest.mark.parametrize('typ, field', zip(OBJECTIVE_TYPES, ('killerTeamId', 'teamId', 'teamId')))
@pytest.mark.parametrize('value', [..., None, 0, 1, 300, -100, '100', '200', True, False, 100.0, 200.0, [100], {}])
def test_unknown_team_gives_taker_minus_one(typ, field, value):
    # a team id is the integer 100 or 200: a float or a bool is not one, even where it compares equal to it
    _, _, o = pipeline(synth.make_record(T_ms=900_000, events=[obj_event(typ, **{field: value})]))
    assert o.taker.tolist() == [-1]


def test_tower_tier():
    tiers = [('OUTER_TURRET', 0), ('INNER_TURRET', 1), ('BASE_TURRET', 2), ('NEXUS_TURRET', 3), ('UNKNOWN_TURRET', -1), (None, -1)]
    rec = synth.make_record(T_ms=900_000, objectives=[objective(150_000 + i, 'BUILDING_KILL', building='TOWER_BUILDING', tower=tw)
                                                      for i, (tw, _) in enumerate(tiers)] + [
        objective(160_000, 'BUILDING_KILL', building='INHIBITOR_BUILDING'), objective(160_001, 'TURRET_PLATE_DESTROYED'),
        objective(160_002, monster='HORDE')], events=[
        {'type': 'ELITE_MONSTER_KILL', 'timestamp': 160_003, 'killerId': 1, 'monsterType': 'DRAGON', 'towerType': 'INNER_TURRET'},
        {'type': 'TURRET_PLATE_DESTROYED', 'timestamp': 160_004, 'killerId': 1, 'towerType': 'BASE_TURRET'},
        {'type': 'BUILDING_KILL', 'timestamp': 160_005, 'killerId': 1, 'towerType': 3}])
    _, _, o = pipeline(rec)
    assert o.sub.tolist() == [t for _, t in tiers] + [-1] * 6
    assert o.kind.tolist() == [6] * 6 + [7, 8, 2, 0, 8, -1]                       # the tier never changes the kind, and so not the cell


def test_tier_only_for_a_tower():
    # a tier needs buildingType TOWER_BUILDING: an inhibitor, a building of no known kind and a building without buildingType get -1 even
    # where they carry a towerType
    rec = synth.make_record(T_ms=900_000, objectives=[
        objective(150_000, 'BUILDING_KILL', building='INHIBITOR_BUILDING', tower='OUTER_TURRET'),
        objective(150_001, 'BUILDING_KILL', building='NEXUS_BUILDING', tower='INNER_TURRET'),
        objective(150_002, 'BUILDING_KILL', building='TOWER_BUILDING', tower='NEXUS_TURRET'),
    ], events=[{'type': 'BUILDING_KILL', 'timestamp': 150_003, 'killerId': 1, 'towerType': 'BASE_TURRET'}])
    _, _, o = pipeline(rec)
    assert o.kind.tolist() == [7, -1, 6, -1] and o.sub.tolist() == [-1, -1, 3, -1]


def test_unit_O_cells_and_upto_inclusive():
    cred = [1]                                           # participant 1 is in unit A
    rec = synth.make_record(T_ms=600_000, kills=[kill(F_A, 0, 1, 6)], objectives=[
        objective(205_000, cred=cred),                                                                             # dragon, blue
        objective(210_000, 'BUILDING_KILL', cred=cred, team=100, building='TOWER_BUILDING', tower='OUTER_TURRET'),   # tower, red takes it
        objective(212_000, 'BUILDING_KILL', cred=cred, team=100, building='TOWER_BUILDING', tower='BASE_TURRET'),    # other tier, same cell
        objective(214_000, 'TURRET_PLATE_DESTROYED', cred=cred, team=200),                                         # plate, blue
        objective(216_000, cred=cred, team=200, monster='DRAGON', sub='ELDER_DRAGON'),                              # elder, red
        objective(217_000, cred=[4]),                                                                              # nobody of A
    ])
    _, u, o = pipeline(rec)
    assert o.unit.tolist() == [0, 0, 0, 0, 0, -1]
    want = np.zeros(N_CELLS, np.int64)
    want[[0, 3, 13, 16]] = [1, 1, 2, 1]                  # cell = 2 * kind + taker: dragon blue 0, elder red 3, tower red 13, plate blue 16
    got = targets.unit_O(o, 0)
    assert got.dtype == np.int64 and got.shape == (N_CELLS,) and got.tolist() == want.tolist()
    assert targets.unit_O(o, 1).tolist() == [0] * N_CELLS                                      # no such unit: nothing attached
    cut = {205_000: {0: 1}, 209_999: {0: 1}, 210_000: {0: 1, 13: 1}, 212_000: {0: 1, 13: 2}, 216_000: {0: 1, 3: 1, 13: 2, 16: 1}}
    for upto, cells in cut.items():
        v = np.zeros(N_CELLS, np.int64)
        for c, n in cells.items():
            v[c] = n
        assert targets.unit_O(o, 0, upto_ms=upto).tolist() == v.tolist(), upto            # upto_ms is inclusive
    assert targets.unit_O(o, 0, upto_ms=204_999).tolist() == [0] * N_CELLS


# attribution: the rules behind the three-line test of the brief
def test_overlap_counts_all_people_of_the_unit():
    # A has two kills: people {1, 6} and {2, 7, 3}, so {1, 2, 3, 6, 7}; B has {4, 9}. The objective credits 3, 7 and 4.
    rec = synth.make_record(T_ms=600_000, kills=[kill(200_000, 0, 1, 6), kill(204_000, 300, 2, 7, (3,)), kill(203_000, 6_000, 4, 9)],
                            objectives=[objective(210_000, cred=[3, 7, 4], xy=(6_000, 0))])
    _, u, o = pipeline(rec)
    assert u.n_kills.tolist() == [2, 1] and u.f.tolist() == [200_000, 203_000]
    assert o.unit.tolist() == [0]                        # overlap 2 with A (3 and 7), 1 with B (4), although B is next to the objective


@pytest.mark.parametrize('kills, xy, want', [
    # the distance is the minimum over all kills of the unit: A = (0, 0) and (1,800, 0), B = (4,000, 0); at (2,500, 0) A is 700 away
    ([kill(200_000, 0, 1, 6), kill(206_000, 1_800, 1, 6), kill(203_000, 4_000, 2, 7)], (2_500, 0), 'A'),
    # ... not the distance to the last kill: A = (0, 0) and (1,800, 0), B = (-2,600, 0); at (-500, 0) A is 500 away, B 2,100
    ([kill(200_000, 0, 1, 6), kill(206_000, 1_800, 1, 6), kill(203_000, -2_600, 2, 7)], (-500, 0), 'A'),
    # ... and in the plane: A = (0, 0), B = (2,000, 3,000); at (0, 3,000) A is 3,000 away, B 2,000
    ([kill(200_000, 0, 1, 6), kill(203_000, 2_000, 2, 7, y=3_000)], (0, 3_000), 'B'),
])
def test_distance_is_minimum_euclidean_over_the_unit_kills(kills, xy, want):
    rec = synth.make_record(T_ms=600_000, kills=kills, objectives=[objective(210_000, cred=[1, 2], xy=xy)])    # overlap 1 with A and B
    _, u, o = pipeline(rec)
    assert len(u.f) == 2 and unit_name(o, 0) == want


def test_unit_without_positioned_kill_wins_only_on_overlap():
    def attached(cred):
        # P (people {1, 6}) has a kill without position, Q (people {2, 7}) one at (5,000, 0)
        rec = synth.make_record(T_ms=600_000, kills=[kill(200_000, None, 1, 6, y=None), kill(203_000, 5_000, 2, 7)],
                                objectives=[objective(210_000, cred=cred, xy=(4_000, 0))])
        k, u, o = pipeline(rec)
        assert len(u.f) == 2 and k.pos_missing.tolist() == [True, False]
        return int(o.unit[0])
    assert attached([1]) == 0                            # only P overlaps: it gets the object although its distance is infinite
    assert attached([1, 2]) == 1                         # equal overlap: Q at 1,000 u beats P at infinity
    assert attached([1, 6, 2]) == 0                      # P has the larger overlap (two against one)
    assert attached([4]) == -1


def test_out_of_scope_units_are_candidates():
    rec = synth.make_record(T_ms=600_000, kills=[kill(60_000, 0, 1, 6), kill(585_000, 6_000, 2, 7)], objectives=[
        objective(70_000, cred=[1]), objective(100_000, cred=[2], xy=(6_000, 0)), objective(590_000, cred=[2], xy=(6_000, 0))])
    _, u, o = pipeline(rec)
    assert u.in_scope.tolist() == [False, False]         # before 120 s, and after T - 30 s
    assert o.unit.tolist() == [0, -1, 1]                 # the object at 100,000 is in the window of neither unit


def test_object_without_position_is_never_attributed():
    rec = synth.make_record(T_ms=600_000, kills=[kill(F_A, 0, 1, 6)], objectives=[
        objective(210_000, cred=[1], xy=(None, None)), objective(210_001, cred=[1])])
    _, _, o = pipeline(rec)
    assert o.has_pos.tolist() == [False, True] and o.unit.tolist() == [-1, 0]


def test_attribution_needs_a_known_kind_and_taker():
    # four objectives inside unit A's window that credit participant 1 at A's kill: only the first has a known kind and taker
    rec = synth.make_record(T_ms=600_000, kills=[kill(F_A, 0, 1, 6)], objectives=[
        objective(210_000, cred=[1]), objective(210_001, cred=[1], monster='SOMETHING_NEW'), objective(210_002, cred=[1], team=None),
        objective(210_003, cred=[1], team=300, monster='SOMETHING_NEW')])
    _, _, o = pipeline(rec)
    assert o.kind.tolist() == [0, -1, 0, -1] and o.taker.tolist() == [0, 0, -1, -1]
    assert o.unit.tolist() == [0, -1, -1, -1]


# conversions
def test_conversion_window_edges_and_cell():
    # one unit A (delta 220,000); objectives of nobody at the edges of (delta, delta + 60,000]; the cell is 2 * kind + taker
    rec = synth.make_record(T_ms=900_000, kills=[kill(F_A, 0, 1, 6)], objectives=[
        objective(DELTA_A), objective(DELTA_A + 1, 'BUILDING_KILL', team=100, building='INHIBITOR_BUILDING'),
        objective(DELTA_A + C.CONV_MS, 'TURRET_PLATE_DESTROYED', team=200), objective(DELTA_A + C.CONV_MS + 1)])
    _, u, o = pipeline(rec)
    conv, expo, cu = targets.conversions(o, u, rec.T_ms)
    assert cu.tolist() == [-1, 0, 0, -1] and expo.tolist() == [C.CONV_MS]
    want = np.zeros((1, N_CELLS), np.int64)
    want[0, [2 * 7 + 1, 2 * 8 + 0]] = 1                  # inhibitor taken by red, plate taken by blue
    assert conv.tolist() == want.tolist()


def test_conversion_tie_goes_to_the_earlier_unit_and_latest_delta_is_by_value():
    # two units with the same delta (both first kills at 200,000, far apart): the earlier index
    rec = synth.make_record(T_ms=900_000, kills=[kill(F_A, 0, 1, 6), kill(F_A, 6_000, 2, 7)], objectives=[objective(230_000)])
    _, u, o = pipeline(rec)
    conv, _, cu = targets.conversions(o, u, rec.T_ms)
    assert u.delta.tolist() == [DELTA_A, DELTA_A] and cu.tolist() == [0] and conv.sum(axis=1).tolist() == [1, 0]
    # unit 0 is a chain of kills 100,000 ... 200,000 (delta 200,001), unit 1 one kill at 150,000 (delta 170,000): the latest delta is
    # unit 0's although it is the first; an objective of nobody at 205,000 is in both conversion windows
    chain = [kill(100_000 + 10_000 * i, 0, 1, 6) for i in range(11)]
    rec = synth.make_record(T_ms=900_000, kills=chain + [kill(150_000, 6_000, 2, 7)], objectives=[objective(205_000)])
    _, u, o = pipeline(rec)
    conv, _, cu = targets.conversions(o, u, rec.T_ms)
    assert u.n_kills.tolist() == [11, 1] and u.delta.tolist() == [200_001, 170_000] and cu.tolist() == [0]
    assert conv.sum(axis=1).tolist() == [1, 0]


def test_conversion_exposure_values():
    T = 600_000
    cases = ((300_000, C.CONV_MS), (540_000, C.CONV_MS), (540_001, 59_999), (590_000, 10_000), (599_999, 1), (T, 0), (T + 5_000, 0))
    for delta, want in cases:
        _, expo, _ = conv_case(delta=delta, T=T, t_obj=T)
        assert expo.tolist() == [want], delta
        assert expo.dtype == np.int64


def test_unknown_kind_or_taker_is_never_attributed_or_converted():
    # one unit A (people {1, 6}, delta 220,000); every objective credits participant 1 at A's kill, so a valid one would attach
    # (t = 210,000, inside A's window) or convert (t = 230,000, inside its conversion window)
    rec = synth.make_record(T_ms=600_000, kills=[kill(F_A, 0, 1, 6)], objectives=[
        objective(210_000, cred=[1]),                                                    # 0 valid: attached
        objective(210_001, cred=[1], monster='SOMETHING_NEW'),                           # 1 kind -1, taker known: rest
        objective(210_002, cred=[1], team=None),                                         # 2 taker -1: not counted anywhere
        objective(230_000, cred=[1], monster='SOMETHING_NEW'),                           # 3 kind -1, taker known: rest
        objective(230_001, cred=[1], team=300),                                          # 4 taker -1
        objective(230_002, cred=[1], team=None, monster='SOMETHING_NEW'),                # 5 both unknown
        objective(230_003, cred=[1]),                                                    # 6 valid: converted
        objective(230_004, cred=[1], xy=(None, None)),                                   # 7 valid, no position: converted
    ])
    _, u, o = pipeline(rec)
    assert o.kind.tolist() == [0, -1, 0, -1, 0, -1, 0, 0] and o.taker.tolist() == [0, 0, -1, 0, -1, -1, 0, 0]
    cells, conv, cu = assert_accounting(o, u, rec.T_ms)
    assert o.unit.tolist() == [0, -1, -1, -1, -1, -1, -1, -1] and cu.tolist() == [-1, -1, -1, -1, -1, -1, 0, 0]
    assert cells.sum() == 1 and conv.sum() == 2
    known = o.taker >= 0
    assert int(known.sum()) == 5 and int(np.sum(known & (o.unit < 0) & (cu < 0))) == 2          # 1 + 2 + 2 = 5


def test_unit_O_and_conversions_count_only_valid_cells():
    # A table that was edited by hand: of five objects of unit 0 (t = 210,000) and five of nobody (t = 230,000, in the conversion window
    # of unit 0) only the first of each five has a cell: the others have an unknown kind or taker, a kind past the nine or a taker past
    # red, and the cell 2 * kind + taker would be wrong or out of range.
    n = 10
    kind, taker_ = np.array([3, -1, 3, len(C.O_KINDS), 3] * 2, np.int8), np.array([1, 1, -1, 0, 2] * 2, np.int8)
    o = targets.ObjectiveTable(t=np.array([210_000] * 5 + [230_000] * 5), kind=kind, taker=taker_, x=np.zeros(n), y=np.zeros(n),
                               has_pos=np.ones(n, bool), cred=np.zeros(n, np.int16), unit=np.array([0] * 5 + [-1] * 5, np.int32),
                               sub=np.full(n, -1, np.int8))
    want = np.zeros(N_CELLS, np.int64)
    want[2 * 3 + 1] = 1
    assert targets.unit_O(o, 0).tolist() == want.tolist()
    _, u, _ = pipeline(synth.make_record(T_ms=600_000, kills=[kill(F_A, 0, 1, 6)]))
    conv, _, cu = targets.conversions(o, u, 600_000)
    assert cu.tolist() == [-1] * 5 + [0, -1, -1, -1, -1] and conv[0].tolist() == want.tolist()


def test_accounting_identity_on_random_matches():
    n_attached = n_converted = n_rest = n_unknown = 0
    for seed in range(100):
        rec = synth.records_from_seed(f'SYN_{seed}', '15.18')
        _, u, o = pipeline(rec)
        cells, conv, cu = assert_accounting(o, u, rec.T_ms)
        n_attached += int(cells.sum())
        n_converted += int(conv.sum())
        n_rest += int(np.sum((o.taker >= 0) & (o.unit < 0) & (cu < 0)))
        n_unknown += int(np.sum(o.taker < 0))
    assert min(n_attached, n_converted, n_rest, n_unknown) > 20          # all four outcomes occur, so the identity is not trivial


def test_accounting_identity_with_unknown_kinds_and_takers_in_a_big_match():
    for seed in range(8):
        rec, u, k = synth_many(seed)
        o = targets.objective_table(rec, u, k)
        cells, conv, cu = assert_accounting(o, u, rec.T_ms)
        bad = (o.kind < 0) | (o.taker < 0)
        assert bad.any() and np.all(o.unit[bad] == -1) and np.all(cu[bad] == -1)           # unknown ones are never attached or converted
        assert cells.sum() > 0 and conv.sum() > 0


def test_dtypes_and_empty_inputs():
    empty_k = kt([])
    comp = np.array([], dtype=np.intp)
    K, E = targets.unit_K(empty_k, comp, TEAM), targets.unit_exec(empty_k, comp, TEAM)
    assert (K.dtype, K.shape, E.dtype, E.shape) == (np.int64, (8,), np.int64, (2,)) and not K.any() and not E.any()
    # no kills, one objective
    rec = synth.make_record(T_ms=600_000, objectives=[objective(150_000)])
    k, u, o = pipeline(rec)
    assert len(u.f) == 0 and o.unit.tolist() == [-1] and o.unit.dtype == np.int32
    conv, expo, cu = targets.conversions(o, u, rec.T_ms)
    assert conv.shape == (0, N_CELLS) and conv.dtype == np.int64 and expo.shape == (0,) and expo.dtype == np.int64
    assert cu.tolist() == [-1] and cu.dtype == np.int32
    # kills, no objective
    rec = synth.make_record(T_ms=600_000, kills=[kill(F_A, 0, 1, 6), kill(300_000, 6_000, 2, 7)])
    k, u, o = pipeline(rec)
    assert len(u.f) == 2 and len(o.t) == 0
    assert {c: getattr(o, c).dtype.str[1:] for c in ('t', 'kind', 'taker', 'x', 'y', 'has_pos', 'cred', 'unit', 'sub')} == dict(
        t='i8', kind='i1', taker='i1', x='f8', y='f8', has_pos='b1', cred='i2', unit='i4', sub='i1')
    conv, expo, cu = targets.conversions(o, u, rec.T_ms)
    assert conv.shape == (2, N_CELLS) and conv.dtype == np.int64 and not conv.any() and expo.tolist() == [C.CONV_MS, C.CONV_MS]
    assert cu.shape == (0,) and cu.dtype == np.int32
    cells = targets.unit_O(o, 0)
    assert cells.dtype == np.int64 and cells.shape == (N_CELLS,) and not cells.any() and not targets.unit_O(o, 0, upto_ms=0).any()
    # neither
    rec = synth.make_record(T_ms=600_000)
    k, u, o = pipeline(rec)
    conv, expo, cu = targets.conversions(o, u, rec.T_ms)
    assert conv.shape == (0, N_CELLS) and expo.shape == (0,) and cu.shape == (0,) and len(o.t) == 0
    # filled tables
    k, u, o = pipeline(synth.records_from_seed('SYN_3', '15.18'))
    conv, expo, cu = targets.conversions(o, u, 10 ** 7)
    assert (conv.dtype, expo.dtype, cu.dtype) == (np.int64, np.int64, np.int32) and conv.shape == (len(u.f), N_CELLS)
    for i in range(len(u.f)):
        K = targets.unit_K(k, u.comp[i], TEAM, upto_ms=int(u.f[i]))
        assert K.dtype == np.int64 and K.shape == (8,) and targets.unit_O(o, i, upto_ms=int(u.f[i])).dtype == np.int64
