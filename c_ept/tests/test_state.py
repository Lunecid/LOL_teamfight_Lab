"""Tests of ``c_ept.state``: who is dead and for how long, the objective timers and the state of the 34 landmarks at a reference time.

The synthetic tests build matches with ``synth.make_record`` and check hand-made numbers (respawn timers, spawn timers and the rule delays
of LEGACY's tables for the patches 15.18 and 16.13). The golden tests (``golden_records``, the 20 real matches of the S1 golden lock) run
only with ``C_EPT_S1_RUN`` set; their messages carry the position of the match in the golden list, never its key. Run with ``--tb=short``.
"""
import ast
import dataclasses
import functools
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from c_ept import casts
from c_ept import config as C
from c_ept import legacy, state
from c_ept.checks import common, sweep_state
from c_ept.tests import synth
from c_ept.tests.conftest import read_json

REPO = Path(C.__file__).parents[1]
COL = {name: i for i, name in enumerate(state.OBJ_STATE_NAMES)}
BUILDINGS = 28                       # the first 28 landmarks are the buildings of the map file, the last 6 the nests
NEST_KINDS = ('dragon', 'baron', 'herald', 'grubs', 'atakhan', 'atakhan')
S1_MODULES = ('c_ept.targets', 'c_ept.event_rows', 'c_ept.frame_rows', 'c_ept.store_core', 'c_ept.run_s1')    # what a view must not reach


# ---- helpers: the anchors of the map file, building and monster events, the landmark state ------------------------------------------
@functools.lru_cache(maxsize=None)
def anchors() -> dict:
    return read_json(Path(C.LEGACY_ROOT) / 'config' / 'game_rules' / 'map_anchors.json', 'map_anchors.json')


def anchor(team, lane, building, tower=None, slot=0):
    """(x, y) of a building in the map file (read here again, apart from ``state``)."""
    rows = [a for a in anchors()['towers'] if (a['team'], a['lane'], a['building'], a['tower']) == (team, lane, building, tower)]
    return rows[slot]['x'], rows[slot]['y']


def building_kill(t, team, lane, building, tower=None, slot=0, at='anchor'):
    """BUILDING_KILL of a building of ``team`` (the owner that loses it) at its anchor (``at``: another (x, y), or None for no position)."""
    x, y = anchor(team, lane, building, tower, slot) if at == 'anchor' else (at or (None, None))
    return dict(t=t, type='BUILDING_KILL', x=x, y=y, team=team, building=building, lane=lane, tower=tower)


def inhib_kill(t, team, lane):
    return building_kill(t, team, lane, 'INHIBITOR_BUILDING')


def tower_kill(t, team, lane, tower, slot=0, at='anchor'):
    return building_kill(t, team, lane, 'TOWER_BUILDING', tower, slot, at)


def monster_kill(t, monster, team, sub=None, killer=1):
    return dict(t=t, type='ELITE_MONSTER_KILL', x=5000, y=10000, killer=killer, team=team, monster=monster, sub=sub)


def plate_kill(t, team, lane):
    return dict(t=t, type='TURRET_PLATE_DESTROYED', x=5000, y=5000, killer=1, team=team, lane=lane)


def ls(rec, t):
    """The landmark state of ``rec`` at ``t`` as the brief writes it."""
    return state.landmark_state(rec, t, state.objective_state(rec, t))


def nest(kind, slot=0):
    return next(i for i, lm in enumerate(state.LANDMARKS) if lm['kind'] == kind and lm['slot'] == slot)


def what_it_hit(k):
    """The kind of landmark a ``BuildingKill`` hit: ``'inhibitor'``, ``'nexus'`` (a nexus turret) or ``'lane'`` (a lane turret); ``'none'``
    for a kill that fits no landmark (``landmark`` is -1 there, which Python would read as ``LANDMARKS[-1]``, an atakhan nest)."""
    if k.landmark < 0:
        return 'none'
    lm = state.LANDMARKS[k.landmark]
    return 'inhibitor' if lm['kind'] == 'inhibitor' else 'nexus' if lm['tower'] == 'NEXUS_TURRET' else 'lane'


def same(a, b):
    """Two arrays (or tuples of arrays) equal bit for bit, NaN equal to NaN."""
    if isinstance(a, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def at_ref(rec, t):
    """Everything ``state`` gives at ``t`` for ``rec`` through its functions: (alive, respawn seconds, objective state, landmark state)."""
    obj = state.objective_state(rec, t)
    return (*state.alive_respawn(rec, t), obj, state.landmark_state(rec, t, obj))


def at_ref_index(index, t):
    obj = index.objective_state(t)
    return (*index.alive_respawn(t), obj, index.landmark_state(t, obj))


def truncated(rec, t):
    """``rec`` with the events after ``t`` taken away (the arrays are shared; the record is frozen)."""
    return dataclasses.replace(rec, events=tuple(e for e in rec.events if e['timestamp'] <= t))


# ---- the five tests of the brief ----------------------------------------------------------------------------------------------------
def test_alive_respawn_ignores_later_kills():
    rec = synth.make_record(kills=[dict(t=300_000, x=0, y=0, killer=1, victim=6)])
    a, r = state.alive_respawn(rec, 299_999)
    assert a[5] == 1 and r[5] == 0
    a, r = state.alive_respawn(rec, 300_000)                       # R10-a: the kill on the reference ms is visible
    assert a[5] == 0
    a, r = state.alive_respawn(rec, 300_001)
    assert a[5] == 0 and r[5] > 0


def test_objective_state_names_equal_across_patches():
    names = [tuple(legacy.load().objective_timers.feature_names(p)) for p in C.PATCHES]
    assert len(set(names)) == 1 and len(names[0]) == 89


def test_landmarks_fixed_34():
    assert len(state.LANDMARKS) == 34
    assert [lm['kind'] for lm in state.LANDMARKS[-6:]] == ['dragon', 'baron', 'herald', 'grubs', 'atakhan', 'atakhan']


def test_tower_destroyed_after_building_kill():
    rec = synth.make_record(objectives=[dict(t=500_000, type='BUILDING_KILL', x=5846, y=6396, team=100, building='TOWER_BUILDING',
                                             tower='OUTER_TURRET', lane='MID_LANE')])             # (5846, 6396): the anchor of that turret
    i = state.landmark_index(team=100, lane='MID_LANE', building='TOWER_BUILDING', tower='OUTER_TURRET')
    assert ls(rec, 499_999)[i, 0] == 1 and ls(rec, 500_000)[i, 0] == 0


def test_inhibitor_respawn_counts_down():
    rec = synth.make_record(objectives=[inhib_kill(t=900_000, team=200, lane='BOT_LANE')])
    j = state.landmark_index(team=200, lane='BOT_LANE', building='INHIBITOR_BUILDING')
    s1, s2 = ls(rec, 960_000)[j], ls(rec, 1_020_000)[j]
    assert s1[0] == 0 and s2[0] == 0 and s1[1] - s2[1] == pytest.approx(60.0)


# ---- alive and respawn: the instants, the level, the early revive --------------------------------------------------------------------
def test_the_kill_instant_the_respawn_instant_and_the_timer_of_a_level_one_victim():
    """R10-a. LEGACY's interval is [kill ms, kill ms + timer) (measured 2026-10-21: dead on the kill ms with the whole timer left, alive
    again on the respawn ms with 0 left), so the inclusive rule needs no special case. Level 1 before 15 min: 10 s."""
    rec = synth.make_record(kills=[dict(t=300_000, x=0, y=0, killer=1, victim=6)])
    others = [i for i in range(10) if i != 5]
    for ref, alive, left in ((299_999, 1, 0.0), (300_000, 0, 10.0), (309_999, 0, 0.001), (310_000, 1, 0.0), (310_001, 1, 0.0)):
        a, r = state.alive_respawn(rec, ref)
        assert a[5] == alive and float(r[5]) == pytest.approx(left, abs=1e-6), ref
        assert a[others].tolist() == [1.0] * 9 and r[others].tolist() == [0.0] * 9


def test_the_death_timer_uses_the_level_of_the_victim_at_the_kill_ms():
    """Level 6 gives 16 s; a LEVEL_UP on the kill ms counts (LEGACY: level of the events at or before the kill), one ms later does not."""
    def up(t, level, pid=6):
        return {'type': 'LEVEL_UP', 'timestamp': t, 'participantId': pid, 'level': level}
    kill = [dict(t=300_000, x=0, y=0, killer=1, victim=6)]
    left = lambda events, ref: float(state.alive_respawn(synth.make_record(kills=kill, events=events), ref)[1][5])     # noqa: E731
    assert left([up(200_000, 6)], 305_000) == 11.0                     # 16 s from 300,000
    assert left([up(300_000, 7)], 300_000) == 20.0                     # level 7: 20 s, the LEVEL_UP on the kill ms counts
    assert left([up(300_001, 7)], 300_000) == 10.0
    assert left([up(100_000, 9, pid=7)], 300_000) == 10.0              # the level of another player does not count


def test_the_death_timer_grows_after_15_minutes():
    """Time factor 0.0085 per minute after 15:00: a kill at 16:00 of a level 1 victim lasts 10 (1 + 0.0085) = 10.085 s."""
    rec = synth.make_record(kills=[dict(t=960_000, x=0, y=0, killer=1, victim=6)])
    assert float(state.alive_respawn(rec, 960_000)[1][5]) == pytest.approx(10.085, abs=1e-5)


def test_a_second_death_before_the_respawn_is_not_seen_before_it_happens():
    """LEGACY cuts an interval at the next death of the same player. At 304,000 the second kill (305,000) is not visible, so 6 s are left
    (a cut by the future would show 1 s); at 305,000 the second timer starts with its whole 10 s."""
    rec = synth.make_record(kills=[dict(t=300_000, x=0, y=0, killer=1, victim=6), dict(t=305_000, x=0, y=0, killer=2, victim=6)])
    assert [float(state.alive_respawn(rec, t)[1][5]) for t in (304_000, 305_000, 306_000, 315_000)] == pytest.approx([6.0, 10.0, 9.0, 0.0])
    assert [float(state.alive_respawn(rec, t)[0][5]) for t in (304_000, 314_999, 315_000)] == [0.0, 0.0, 1.0]


# ---- the objective state: names, hand values, the visible window -------------------------------------------------------------------
def test_the_names_are_those_of_legacy_in_its_order_and_cover_every_patch():
    names = tuple(legacy.load().objective_timers.feature_names('15.18'))
    assert isinstance(state.OBJ_STATE_NAMES, tuple) and state.OBJ_STATE_NAMES == names and len(set(names)) == 89
    assert names[0] == 't_s' and {'dragon_up', 'blue_inhibs_down', 'red_inhibs_down'} <= set(names)
    for patch in C.PATCHES:
        obj = state.objective_state(synth.make_record(patch=patch), 100_000)
        assert obj.shape == (89,) and obj.dtype == np.float32 and obj[0] == 100.0


def test_the_spawn_timers_of_a_quiet_match_are_the_rules_of_the_patch():
    """15.18 (hand values of the rule table): dragon 300 s, grubs 480 s, herald 900 s, atakhan 1,200 s, baron 1,500 s; 16.13: baron 1,200 s
    and no Atakhan (NaN: it will not spawn)."""
    old = synth.make_record(T_ms=2_000_000)
    o = state.objective_state(old, 200_000)
    want = {'t_s': 200.0, 'dragon_next_spawn_s': 100.0, 'voidgrub_next_spawn_s': 280.0, 'herald_next_spawn_s': 700.0,
            'atakhan_next_spawn_s': 1000.0, 'baron_next_spawn_s': 1300.0, 'dragon_up': 0.0, 'baron_up': 0.0, 'blue_towers_outer': 3.0,
            'red_towers_nexus': 2.0, 'blue_inhibs_down': 0.0, 'rule_uncertain_herald': 0.0}
    assert {n: float(o[COL[n]]) for n in want} == want
    up = state.objective_state(old, 300_000)
    assert (up[COL['dragon_up']], up[COL['dragon_next_spawn_s']]) == (1.0, 0.0)
    new = state.objective_state(synth.make_record(T_ms=2_000_000, patch='16.13'), 200_000)
    assert new[COL['baron_next_spawn_s']] == 1000.0 and np.isnan(new[COL['atakhan_next_spawn_s']]) and new[COL['atakhan_up']] == 0.0
    assert new[COL['rule_uncertain_herald']] == 1.0                    # LEGACY flags the 16.x herald despawn as uncertain


def test_monster_kills_move_the_timers_and_the_counts_on_the_reference_ms_itself():
    rec = synth.make_record(T_ms=2_000_000, objectives=[monster_kill(400_000, 'DRAGON', 100, 'FIRE_DRAGON'),
                                                        monster_kill(1_600_000, 'BARON_NASHOR', 100)])
    before, on = state.objective_state(rec, 399_999), state.objective_state(rec, 400_000)
    assert (before[COL['blue_dragons']], on[COL['blue_dragons']], on[COL['blue_dragon_fire']]) == (0.0, 1.0, 1.0)
    assert state.objective_state(rec, 450_000)[COL['dragon_next_spawn_s']] == 250.0          # 300 s after the kill
    b = state.objective_state(rec, 1_700_000)
    assert (b[COL['blue_barons']], b[COL['baron_next_spawn_s']], b[COL['blue_baron_buff_s']], b[COL['blue_baron_buff_n']]) == (
        1.0, 260.0, 80.0, 5.0)                                         # 360 s respawn; 180 s buff for the five living blue players
    assert b[COL['red_baron_buff_n']] == 0.0


EVENT_CASES = {                      # the events the state reads, as the keywords of make_record, for a given time
    'kill': lambda t: dict(kills=[dict(t=t, x=0, y=0, killer=1, victim=6)]),
    'tower': lambda t: dict(objectives=[tower_kill(t, 100, 'TOP_LANE', 'OUTER_TURRET')]),
    'inhibitor': lambda t: dict(objectives=[inhib_kill(t, 200, 'MID_LANE')]),
    'dragon': lambda t: dict(objectives=[monster_kill(t, 'DRAGON', 200, 'EARTH_DRAGON')]),
    'grubs': lambda t: dict(objectives=[monster_kill(t, 'HORDE', 100)]),
    'plate': lambda t: dict(objectives=[plate_kill(t, 200, 'MID_LANE')]),
}


def test_an_event_after_the_reference_time_does_not_change_the_state_and_one_on_it_does():
    """R10-a and R10-g for each kind of event that the state reads: one event at ref + 1 (hidden) and the same at ref (visible)."""
    ref = 700_000
    base = at_ref(synth.make_record(), ref)
    for name, make in EVENT_CASES.items():
        assert same(at_ref(synth.make_record(**make(ref + 1)), ref), base), name
        assert not same(at_ref(synth.make_record(**make(ref)), ref), base), name


# ---- the landmarks: structure, indexing, buildings, nests ----------------------------------------------------------------------------
def test_the_landmarks_are_the_28_buildings_of_the_map_file_in_its_order_then_the_six_nests():
    lms = state.LANDMARKS
    assert len(lms) == 34 and all(set(lm) == {'kind', 'team', 'lane', 'building', 'tower', 'slot', 'x', 'y'} for lm in lms)
    rows = anchors()['towers']
    assert [(lm['team'], lm['lane'], lm['building'], lm['tower'], lm['slot'], lm['x'], lm['y']) for lm in lms[:BUILDINGS]] == [
        (a['team'], a['lane'], a['building'], a['tower'], a['slot'], a['x'], a['y']) for a in rows]
    assert [lm['kind'] for lm in lms[:BUILDINGS]] == ['inhibitor' if a['building'] == 'INHIBITOR_BUILDING' else 'tower' for a in rows]
    for team in (100, 200):                                            # per team: 3 inhibitors, 3 outer, 3 inner, 3 base, 2 nexus turrets
        mine = [lm for lm in lms[:BUILDINGS] if lm['team'] == team]
        tiers = ('OUTER_TURRET', 'INNER_TURRET', 'BASE_TURRET', 'NEXUS_TURRET')
        assert len(mine) == 14 and [sum(lm['tower'] == t for lm in mine) for t in tiers] == [3, 3, 3, 2]
        assert sum(lm['kind'] == 'inhibitor' for lm in mine) == 3
    objs = anchors()['objectives']
    got = [(lm['x'], lm['y']) for lm in lms[BUILDINGS:]]
    places = (('DRAGON', 0), ('BARON_NASHOR', 0), ('RIFTHERALD', 0), ('HORDE', 0), ('ATAKHAN', 0), ('ATAKHAN', 1))
    assert got == [(objs[k][s]['x'], objs[k][s]['y']) for k, s in places]
    assert [lm['slot'] for lm in lms[BUILDINGS:]] == [0, 0, 0, 0, 0, 1]
    assert all(lm[k] is None for lm in lms[BUILDINGS:] for k in ('team', 'lane', 'building', 'tower'))
    assert (anchor(100, 'MID_LANE', 'TOWER_BUILDING', 'OUTER_TURRET'), anchor(200, 'BOT_LANE', 'INHIBITOR_BUILDING')) == (
        (5846, 6396), (13594, 11319))                                  # the hand values used by the brief's tests (R10-c)


def test_landmark_index_finds_every_building_and_refuses_everything_else():
    for i, lm in enumerate(state.LANDMARKS[:BUILDINGS]):
        assert state.landmark_index(lm['team'], lm['lane'], lm['building'], lm['tower'], lm['slot']) == i
    assert state.landmark_index(200, 'BOT_LANE', 'INHIBITOR_BUILDING') == state.landmark_index(200, 'BOT_LANE', 'INHIBITOR_BUILDING', None)
    for bad in ((100, 'MID_LANE', 'TOWER_BUILDING', 'OUTER_TURRET', 1),       # one outer turret per lane
                (300, 'MID_LANE', 'TOWER_BUILDING', 'OUTER_TURRET', 0), (100, 'JUNGLE', 'TOWER_BUILDING', 'OUTER_TURRET', 0),
                (100, 'MID_LANE', 'NEXUS_BUILDING', None, 0), (100, 'TOP_LANE', 'TOWER_BUILDING', 'NEXUS_TURRET', 0)):
        with pytest.raises(KeyError):
            state.landmark_index(*bad)


def test_the_landmark_entries_cannot_be_changed():
    """An entry that a consumer changes (to mirror positions for a team swap, say) would change the nexus matching of the whole process:
    the entries are read-only mappings, and the tuple of them is read-only too. If an entry were a plain dict, this test puts its values
    back, so that a failure here does not show up in other tests."""
    lms = state.LANDMARKS
    saved = [dict(lm) for lm in lms]
    try:
        for lm in lms:
            with pytest.raises(TypeError):
                lm['x'] = 0.0                                          # a key that is there
            with pytest.raises(TypeError):
                lm['mirrored'] = True                                  # a new key
            with pytest.raises(TypeError):
                del lm['y']
        with pytest.raises(TypeError):
            lms[0] = lms[1]                                            # the tuple of entries
    finally:
        for lm, old in zip(lms, saved):
            if isinstance(lm, dict):
                lm.clear()
                lm.update(old)
    assert [dict(lm) for lm in lms] == saved and all(isinstance(lm, Mapping) for lm in lms)
    copy = dict(lms[0])                                                # a copy of an entry is an ordinary dict; the entry stays as it was
    copy['x'] = -1.0
    assert lms[0]['x'] != -1.0


def test_a_kill_at_the_anchor_of_each_of_the_28_buildings_takes_down_exactly_that_landmark():
    """Tower: (0, 0), the turret never comes back; inhibitor: (0, 300), the rule delay; nexus turret: (0, 180)."""
    seconds = {'tower': 0.0, 'inhibitor': 300.0}
    for i, lm in enumerate(state.LANDMARKS[:BUILDINGS]):
        rec = synth.make_record(objectives=[building_kill(500_000, lm['team'], lm['lane'], lm['building'], lm['tower'], lm['slot'])])
        before, after = ls(rec, 499_999)[:BUILDINGS], ls(rec, 500_000)[:BUILDINGS]
        assert before.tolist() == [[1.0, 0.0]] * BUILDINGS, i
        want = before.copy()
        want[i] = (0.0, 180.0 if lm['tower'] == 'NEXUS_TURRET' else seconds[lm['kind']])
        assert after.tolist() == want.tolist(), i


def test_a_destroyed_tower_stays_down_and_an_inhibitor_comes_back_after_300_seconds():
    rec = synth.make_record(T_ms=2_000_000, objectives=[tower_kill(500_000, 100, 'MID_LANE', 'OUTER_TURRET'),
                                                        inhib_kill(900_000, 200, 'BOT_LANE')])
    tw = state.landmark_index(100, 'MID_LANE', 'TOWER_BUILDING', 'OUTER_TURRET')
    ih = state.landmark_index(200, 'BOT_LANE', 'INHIBITOR_BUILDING')
    assert ls(rec, 1_900_000)[tw].tolist() == [0.0, 0.0]
    rows = [ls(rec, t)[ih] for t in (899_999, 900_000, 1_199_999, 1_200_000)]
    assert [r[0] for r in rows] == [1.0, 0.0, 0.0, 1.0] and [float(r[1]) for r in rows] == pytest.approx([0.0, 300.0, 0.001, 0.0], abs=1e-6)


def test_nexus_turrets_take_the_nearest_standing_turret_and_stand_again_after_the_rule_delay():
    """Rule nexus_turret_respawn_ms = 180 s. Team 100's two nexus turrets; each event comes at the anchor of a turret. e3 is at the anchor
    of turret 0, but turret 0 is down and turret 1 has just stood again, so e3 is turret 1; e4 finds none standing: nearest, flagged."""
    s0, s1 = (state.landmark_index(100, 'MID_LANE', 'TOWER_BUILDING', 'NEXUS_TURRET', k) for k in (0, 1))
    rec = synth.make_record(T_ms=2_000_000, objectives=[tower_kill(600_000, 100, 'MID_LANE', 'NEXUS_TURRET', 1),      # e1: turret 1
                                                        tower_kill(650_000, 100, 'MID_LANE', 'NEXUS_TURRET', 0),      # e2: turret 0
                                                        tower_kill(790_000, 100, 'MID_LANE', 'NEXUS_TURRET', 0),      # e3: at 0, takes 1
                                                        tower_kill(800_000, 100, 'MID_LANE', 'NEXUS_TURRET', 1)])     # e4: none standing

    def got(t):
        s = ls(rec, t)
        return s[s0].tolist(), s[s1].tolist()
    assert got(599_999) == ([1.0, 0.0], [1.0, 0.0])
    assert got(700_000) == ([0.0, 130.0], [0.0, 80.0])                 # 650,000 + 180,000 - 700,000; 600,000 + 180,000 - 700,000
    assert got(780_000) == ([0.0, 50.0], [1.0, 0.0])                   # turret 1 stands again on the ms of its respawn
    assert got(790_000) == ([0.0, 40.0], [0.0, 180.0])                 # e3 took turret 1, not the nearer, still dead turret 0
    assert got(800_000) == ([0.0, 30.0], [0.0, 180.0])                 # e4: the timer of the nearest one starts again
    assert [(k.landmark, k.status) for k in state.building_kills(rec)] == [(s1, 'ok'), (s0, 'ok'), (s1, 'after_respawn'),
                                                                           (s1, 'already_down')]


def test_a_nexus_turret_kill_without_a_position_takes_the_standing_turret_of_the_lowest_slot():
    rec = synth.make_record(objectives=[tower_kill(500_000, 200, 'MID_LANE', 'NEXUS_TURRET', at=None),
                                        tower_kill(510_000, 200, 'MID_LANE', 'NEXUS_TURRET', at=None)])
    s0, s1 = (state.landmark_index(200, 'MID_LANE', 'TOWER_BUILDING', 'NEXUS_TURRET', k) for k in (0, 1))
    assert [(k.landmark, k.status) for k in state.building_kills(rec)] == [(s0, 'ok'), (s1, 'ok')]


def test_an_inhibitor_killed_again_is_flagged_only_when_it_had_not_come_back():
    rec = synth.make_record(T_ms=2_000_000, objectives=[inhib_kill(500_000, 100, 'TOP_LANE'), inhib_kill(700_000, 100, 'TOP_LANE'),
                                                        inhib_kill(1_000_001, 100, 'TOP_LANE')])
    assert [k.status for k in state.building_kills(rec)] == ['ok', 'already_down', 'after_respawn']         # 700,000 + 300,000 = 1,000,000
    i = state.landmark_index(100, 'TOP_LANE', 'INHIBITOR_BUILDING')
    assert ls(rec, 1_000_001)[i].tolist() == [0.0, 300.0]                  # LEGACY: the last kill sets the countdown, as the state does


def test_a_tower_killed_twice_and_events_that_fit_no_landmark_are_reported_and_change_nothing():
    """Seven BUILDING_KILLs: two that fit (the second is a kill of a tower that was down), then five that no landmark fits."""
    events = [tower_kill(500_000, 100, 'MID_LANE', 'OUTER_TURRET'), tower_kill(510_000, 100, 'MID_LANE', 'OUTER_TURRET'),
              dict(tower_kill(520_000, 100, 'MID_LANE', 'INNER_TURRET'), lane='JUNGLE'),
              dict(tower_kill(530_000, 100, 'MID_LANE', 'INNER_TURRET'), team=None),
              dict(tower_kill(540_000, 100, 'MID_LANE', 'INNER_TURRET'), team=150),
              dict(inhib_kill(550_000, 200, 'MID_LANE'), building='NEXUS_BUILDING'),
              dict(tower_kill(560_000, 100, 'MID_LANE', 'NEXUS_TURRET'), lane='TOP_LANE')]               # no nexus turret in that lane
    rec = synth.make_record(objectives=events)
    kills = state.building_kills(rec)
    assert [k.status for k in kills] == ['ok', 'already_down'] + ['unmatched'] * 5 and [k.landmark for k in kills[2:]] == [-1] * 5
    assert [k.t for k in kills] == [500_000 + 10_000 * i for i in range(7)]
    only_first = synth.make_record(objectives=events[:1])
    assert same(ls(rec, 1_000_000)[:BUILDINGS], ls(only_first, 1_000_000)[:BUILDINGS])
    assert [k.status for k in state.building_kills(rec, 510_000)] == ['ok', 'already_down']                # until_ms: at or before


def test_a_tower_type_that_does_not_exist_is_unmatched_and_a_text_that_is_not_a_string_is_not_read():
    weird = [dict(tower_kill(500_000, 100, 'MID_LANE', 'INNER_TURRET'), tower='MAGIC_TURRET'),
             dict(tower_kill(510_000, 100, 'MID_LANE', 'INNER_TURRET'), tower=7)]
    raw = {'type': 'BUILDING_KILL', 'timestamp': 520_000, 'teamId': '100', 'buildingType': 'TOWER_BUILDING', 'towerType': 'INNER_TURRET',
           'laneType': 'MID_LANE', 'killerId': 1, 'position': {'x': 5048, 'y': 4812}}               # a team id that is a string is no team
    assert [k.status for k in state.building_kills(synth.make_record(objectives=weird, events=[raw]))] == ['unmatched'] * 3
    lower = tower_kill(500_000, 100, 'MID_LANE', 'OUTER_TURRET')
    rec = synth.make_record(objectives=[dict(lower, tower='outer_turret', lane='mid_lane')])         # LEGACY upper-cases the three texts
    assert [k.status for k in state.building_kills(rec)] == ['ok']


@pytest.mark.parametrize('bad', [1.5, 500_000.0, True, False, '500000', np.float64(3.0), np.bool_(True)])
def test_building_kills_checks_its_time_argument_like_the_other_functions(bad):
    rec = synth.make_record(objectives=[tower_kill(500_000, 100, 'MID_LANE', 'OUTER_TURRET')])
    with pytest.raises(TypeError, match='until_ms'):
        state.building_kills(rec, bad)


def test_building_kills_takes_no_limit_a_python_int_or_a_numpy_integer():
    rec = synth.make_record(objectives=[tower_kill(500_000, 100, 'MID_LANE', 'OUTER_TURRET')])
    got = [len(state.building_kills(rec, until)) for until in (None, 500_000, np.int64(500_000), np.int32(499_999), 499_999)]
    assert got == [1, 1, 1, 0, 0]


def test_a_kill_that_fits_no_landmark_gets_a_label_of_its_own():
    """``landmark`` -1 is not an index: read as ``LANDMARKS[-1]`` (an atakhan nest) it was labelled a lane turret."""
    rec = synth.make_record(objectives=[tower_kill(100_000, 100, 'MID_LANE', 'OUTER_TURRET'),
                                        dict(tower_kill(110_000, 100, 'MID_LANE', 'INNER_TURRET'), team=None),
                                        inhib_kill(120_000, 200, 'TOP_LANE'), tower_kill(130_000, 100, 'MID_LANE', 'NEXUS_TURRET', 0)])
    kills = state.building_kills(rec)
    assert [k.landmark for k in kills][1] == -1 and [what_it_hit(k) for k in kills] == ['lane', 'none', 'inhibitor', 'nexus']


def test_the_landmark_rows_and_legacys_turret_counts_can_disagree_on_malformed_events():
    """The matching is strict (the teamId an int, the lane must fit, the towerType known); LEGACY's own counters are lenient
    (``int(teamId)``, a tier counted whatever the lane, a nexus turret keyed by its position whatever the lane) and LEGACY raises a bare
    KeyError on an unknown towerType. On such events the landmark rows and LEGACY's counts differ, as the docstring of ``building_kills``
    says."""
    def standing_and_counted(rec, side, short, tier):
        obj = state.objective_state(rec, 600_000)
        rows = [i for i, x in enumerate(state.LANDMARKS[:BUILDINGS]) if (x['team'], x['tower']) == (100, tier)]
        return float(state.landmark_state(rec, 600_000, obj)[rows, 0].sum()), float(obj[COL[f'{side}_towers_{short}']])
    lane = synth.make_record(objectives=[dict(tower_kill(500_000, 100, 'MID_LANE', 'INNER_TURRET'), lane='JUNGLE')])
    assert [k.status for k in state.building_kills(lane)] == ['unmatched']
    assert standing_and_counted(lane, 'blue', 'inner', 'INNER_TURRET') == (3.0, 2.0)                  # LEGACY counts the tier in any lane
    text = {'type': 'BUILDING_KILL', 'timestamp': 500_000, 'teamId': '100', 'buildingType': 'TOWER_BUILDING', 'towerType': 'OUTER_TURRET',
            'laneType': 'MID_LANE', 'killerId': 1, 'position': {'x': 5846, 'y': 6396}}
    team = synth.make_record(events=[text])
    assert [k.status for k in state.building_kills(team)] == ['unmatched']
    assert standing_and_counted(team, 'blue', 'outer', 'OUTER_TURRET') == (3.0, 2.0)                  # LEGACY reads int('100')
    nexus = synth.make_record(objectives=[dict(tower_kill(500_000, 100, 'MID_LANE', 'NEXUS_TURRET', 0), lane='TOP_LANE')])
    assert [k.status for k in state.building_kills(nexus)] == ['unmatched']
    assert standing_and_counted(nexus, 'blue', 'nexus', 'NEXUS_TURRET') == (2.0, 1.0)                 # LEGACY keys it by its position
    magic = synth.make_record(objectives=[dict(tower_kill(500_000, 100, 'MID_LANE', 'INNER_TURRET'), tower='MAGIC_TURRET')])
    assert [k.status for k in state.building_kills(magic)] == ['unmatched']
    with pytest.raises(KeyError, match='MAGIC_TURRET'):
        state.objective_state(magic, 600_000)


def test_a_second_kill_of_the_same_nexus_turret_goes_to_its_partner_and_only_the_sweep_counts_show_it():
    """While the partner stands, a second kill of the turret that is down is reassigned to the partner with the status ``ok`` (the rule
    takes the nearest turret among those that stand). It shows only as a kill far from its landmark's anchor (the partner is 631 units
    away) and as a mismatch with LEGACY's count, which keys a nexus turret by position: one turret down, where two landmarks are. So
    'no kill on a destroyed landmark' means something for nexus turrets only together with those two counts."""
    rec = synth.make_record(objectives=[tower_kill(600_000, 100, 'MID_LANE', 'NEXUS_TURRET', 1),
                                        tower_kill(610_000, 100, 'MID_LANE', 'NEXUS_TURRET', 1)])
    s0, s1 = (state.landmark_index(100, 'MID_LANE', 'TOWER_BUILDING', 'NEXUS_TURRET', k) for k in (0, 1))
    assert [(k.landmark, k.status) for k in state.building_kills(rec)] == [(s1, 'ok'), (s0, 'ok')]
    got = sweep_state.count_record(rec)
    assert got['building_kills_on_destroyed_landmark'] == 0 and got['building_kills_far_from_their_landmark'] == 1
    assert got['tower_count_mismatch'] == 2                            # at the frames 660,300 and 720,300, while both landmarks are down


def test_nests_follow_the_timers_of_the_patch_and_are_nan_where_legacy_says_never_again():
    rec = synth.make_record(T_ms=2_000_000, objectives=[monster_kill(400_000, 'DRAGON', 100, 'FIRE_DRAGON'),
                                                        monster_kill(1_000_000, 'RIFTHERALD', 200)])
    s = ls(rec, 200_000)
    quiet = [s[nest(k, slot)].tolist() for k, slot in (('dragon', 0), ('baron', 0), ('herald', 0), ('grubs', 0), ('atakhan', 0),
                                                       ('atakhan', 1))]
    assert quiet == [[0.0, 100.0], [0.0, 1300.0], [0.0, 700.0], [0.0, 280.0], [0.0, 1000.0], [0.0, 1000.0]]
    s = ls(rec, 500_000)                                               # the dragon was taken at 400,000; the grubs are up since 480,000
    assert [s[nest(k)].tolist() for k in ('dragon', 'grubs', 'herald')] == [[0.0, 200.0], [1.0, 0.0], [0.0, 400.0]]
    s = ls(rec, 1_100_000)                                             # the herald is taken: it will not spawn again
    assert s[nest('herald')][0] == 0.0 and np.isnan(s[nest('herald')][1])
    assert s[nest('baron')].tolist() == [0.0, 400.0] and s[nest('atakhan')].tolist() == [0.0, 100.0]
    new = ls(synth.make_record(T_ms=2_000_000, patch='16.13'), 200_000)                  # no Atakhan in 16.x
    assert [bool(np.isnan(new[nest('atakhan', k)][1])) for k in (0, 1)] == [True, True] and new[nest('atakhan')][0] == 0.0


def test_three_grub_kills_end_the_grub_nest_and_the_inhibitor_row_copies_the_objective_vector():
    grubs = [monster_kill(500_000 + 1_000 * i, 'HORDE', 100) for i in range(3)]
    rec = synth.make_record(T_ms=2_000_000, objectives=grubs + [inhib_kill(900_000, 100, 'MID_LANE')])
    obj = state.objective_state(rec, 1_000_000)
    s = state.landmark_state(rec, 1_000_000, obj)
    assert s[nest('grubs')][0] == 0.0 and np.isnan(s[nest('grubs')][1]) and np.isnan(obj[COL['voidgrub_next_spawn_s']])
    j = state.landmark_index(100, 'MID_LANE', 'INHIBITOR_BUILDING')
    assert s[j].tolist() == [0.0, obj[COL['blue_inhib_mid_respawn_s']]] and obj[COL['blue_inhib_mid_respawn_s']] == 200.0


def test_landmark_state_refuses_an_objective_vector_of_another_time_or_shape():
    rec = synth.make_record()
    obj = state.objective_state(rec, 100_000)
    assert state.landmark_state(rec, 100_000, obj).shape == (34, 2)
    for bad in (state.objective_state(rec, 100_001), obj[:88], obj.reshape(1, 89), np.zeros(89, dtype=np.float32)):
        with pytest.raises(ValueError):
            state.landmark_state(rec, 100_000, bad)
    assert same(state.landmark_state(rec, 100_000, obj.astype(np.float64)), state.landmark_state(rec, 100_000, obj))


@pytest.mark.parametrize('bad', [1.5, 100_000.0, True, '100000', None, np.float64(3.0)])
def test_the_reference_time_is_a_whole_number_of_milliseconds(bad):
    rec = synth.make_record()
    obj = state.objective_state(rec, 100_000)
    index = state.StateIndex(rec)
    calls = (lambda: state.alive_respawn(rec, bad), lambda: state.objective_state(rec, bad), lambda: state.landmark_state(rec, bad, obj),
             lambda: index.alive_respawn(bad), lambda: index.objective_state(bad), lambda: index.landmark_state(bad, obj))
    for call in calls:
        with pytest.raises(TypeError):
            call()
    assert state.alive_respawn(rec, np.int64(100_000))[0].dtype == np.float32           # a numpy integer is a whole number of ms


# ---- dtype, range, NaN (R10-f) -------------------------------------------------------------------------------------------------------
def check_policy(a, r, obj, lm):
    """The rules of the outputs: float32; flags exactly 0 or 1; seconds not negative; NaN only in the three LEGACY features and in the
    seconds of the nests that use them."""
    assert (a.dtype, r.dtype, obj.dtype, lm.dtype) == (np.float32,) * 4
    assert a.shape == r.shape == (10,) and obj.shape == (89,) and lm.shape == (34, 2)
    assert set(a.tolist()) <= {0.0, 1.0} and np.isfinite(r).all() and (r >= 0).all() and ((a == 0) == (r > 0)).all()
    nan_cols = {COL[n] for n in state.NAN_FEATURES}
    assert {int(i) for i in np.flatnonzero(np.isnan(obj))} <= nan_cols and not np.isinf(obj).any()
    assert set(lm[:, 0].tolist()) <= {0.0, 1.0} and not np.isinf(lm).any()
    nan_rows = {int(i) for i in np.flatnonzero(np.isnan(lm[:, 1]))}
    assert nan_rows <= {i for i, k in enumerate(NEST_KINDS, BUILDINGS) if k in ('herald', 'grubs', 'atakhan')}
    assert not np.isnan(lm[:, 0]).any()
    assert (lm[~np.isnan(lm[:, 1]), 1] >= 0).all()


def test_the_nan_features_are_named_and_the_outputs_follow_the_dtype_and_range_rules():
    assert state.NAN_FEATURES == ('herald_next_spawn_s', 'atakhan_next_spawn_s', 'voidgrub_next_spawn_s')
    for seed in range(14):
        for patch in C.PATCHES:
            rec = synth.records_from_seed(f'SYN_{seed}', patch)
            for ref in (0, 59_999, 300_000, 600_000, 900_000, rec.T_ms // 2, rec.T_ms):
                check_policy(*at_ref(rec, ref))


def test_a_value_that_legacy_does_not_define_as_nan_is_refused(monkeypatch):
    """A NaN in a feature that is not one of the three is a change of LEGACY's tables or code: the state stops instead of passing it on."""
    rec = synth.make_record()
    objective_timers = legacy.load().objective_timers
    real = objective_timers.objective_state_at

    def nan_dragon(*args, **kwargs):
        d = real(*args, **kwargs)
        d['dragon_next_spawn_s'] = float('nan')
        return d
    monkeypatch.setattr(objective_timers, 'objective_state_at', nan_dragon)
    with pytest.raises(ValueError, match='dragon_next_spawn_s'):
        state.objective_state(rec, 100_000)


def test_the_cast_to_float32_is_the_range_checked_one_of_casts():
    assert state.narrow is casts.narrow


# ---- no hindsight and the per-match index (R10-g) --------------------------------------------------------------------------------------
def future_events(t):
    """One event of every kind the state reads, all at ``t``."""
    return [{'type': 'CHAMPION_KILL', 'timestamp': t, 'killerId': 1, 'victimId': 7, 'position': {'x': 1, 'y': 1}},
            {'type': 'LEVEL_UP', 'timestamp': t, 'participantId': 7, 'level': 12},
            {'type': 'BUILDING_KILL', 'timestamp': t, 'teamId': 200, 'buildingType': 'TOWER_BUILDING', 'towerType': 'BASE_TURRET',
             'laneType': 'TOP_LANE', 'killerId': 1, 'position': {'x': 10481, 'y': 13650}},
            {'type': 'BUILDING_KILL', 'timestamp': t, 'teamId': 100, 'buildingType': 'INHIBITOR_BUILDING', 'laneType': 'MID_LANE',
             'killerId': 6, 'position': {'x': 3210, 'y': 3217}},
            {'type': 'ELITE_MONSTER_KILL', 'timestamp': t, 'killerTeamId': 100, 'killerId': 1, 'monsterType': 'BARON_NASHOR'},
            {'type': 'ELITE_MONSTER_KILL', 'timestamp': t, 'killerTeamId': 200, 'killerId': 6, 'monsterType': 'DRAGON',
             'monsterSubType': 'WATER_DRAGON'},
            {'type': 'DRAGON_SOUL_GIVEN', 'timestamp': t, 'teamId': 200, 'name': 'ocean'},
            {'type': 'TURRET_PLATE_DESTROYED', 'timestamp': t, 'teamId': 100, 'laneType': 'BOT_LANE', 'killerId': 7}]


def with_events(rec, extra):
    events = sorted(list(rec.events) + extra, key=lambda e: e['timestamp'])          # stable
    return dataclasses.replace(rec, events=tuple(events))


def test_the_state_of_a_match_cut_at_the_reference_time_is_the_same_as_that_of_the_whole_match():
    """R10-g on random synthetic matches: what is after ``ref`` can neither be seen by the functions nor by the index."""
    for seed in range(12):
        for patch in ('15.18', '16.13'):
            rec = synth.records_from_seed(f'SYN_{seed}', patch)
            whole = state.StateIndex(rec)
            for ref in (0, 120_000, 300_000, 480_000, 900_000, rec.T_ms // 2, rec.T_ms - 1):
                cut = truncated(rec, ref)
                want = at_ref(cut, ref)
                assert same(at_ref(rec, ref), want), (seed, patch, ref)
                assert same(at_ref_index(whole, ref), want), (seed, patch, ref)
                assert same(at_ref_index(state.StateIndex(cut), ref), want), (seed, patch, ref)


def test_events_of_every_kind_one_ms_after_the_reference_time_change_nothing_and_on_it_they_do():
    rec = synth.make_record(T_ms=2_000_000, objectives=[tower_kill(300_000, 100, 'TOP_LANE', 'OUTER_TURRET'),
                                                        monster_kill(350_000, 'DRAGON', 100, 'FIRE_DRAGON')])
    ref = 1_000_000
    base = at_ref(rec, ref)
    assert same(at_ref(with_events(rec, future_events(ref + 1)), ref), base)
    assert same(at_ref_index(state.StateIndex(with_events(rec, future_events(ref + 1))), ref), base)
    on = at_ref(with_events(rec, future_events(ref)), ref)
    assert all(not same(on[k], base[k]) for k in range(4))                                           # each of the four outputs reads them


def test_the_index_gives_what_the_functions_give():
    for seed in range(8):
        for patch in C.PATCHES:
            rec = synth.records_from_seed(f'SYN_{seed}', patch)
            index = state.StateIndex(rec)
            for ref in (0, 59_999, 300_000, 777_777, rec.T_ms):
                assert same(at_ref_index(index, ref), at_ref(rec, ref)), (seed, patch, ref)


# ---- what the module reaches (R10-g, R10-h) ------------------------------------------------------------------------------------------
def module_level_imports(path: Path) -> set:
    """Every module a statement at module level (also inside ``if``/``try``, not inside a function) imports, and ``module.name`` for
    ``from module import name`` (``from c_ept import run_s1`` is ``c_ept.run_s1``)."""
    found = set()

    def visit(body):
        for node in body:
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.add(node.module or '')
                found.update(f'{node.module}.{a.name}' for a in node.names)
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for part in ('body', 'orelse', 'finalbody'):
                    visit(getattr(node, part, []))
                for handler in getattr(node, 'handlers', []):
                    visit(handler.body)
    visit(ast.parse(path.read_text(encoding='utf-8')).body)
    return found


def test_the_import_scan_sees_a_module_level_import_and_not_one_inside_a_function(tmp_path):
    src = ('import sys\nfrom c_ept import run_s1\nif True:\n    from c_ept.targets import x\n\n'
           'def main():\n    from c_ept import store_core\n')
    f = tmp_path / 'm.py'
    f.write_text(src, encoding='utf-8')
    got = module_level_imports(f)
    assert {'c_ept.run_s1', 'c_ept.targets'} <= got and 'c_ept.store_core' not in got
    assert got & set(S1_MODULES) == {'c_ept.run_s1', 'c_ept.targets'}


@pytest.mark.parametrize('module', ['state.py', 'checks/sweep_state.py'])
def test_no_module_level_import_reaches_an_s1_target_store_or_the_s1_runner(module):
    assert not module_level_imports(Path(state.__file__) if module == 'state.py' else REPO / 'c_ept' / module) & set(S1_MODULES)


def test_importing_the_state_module_loads_no_s1_module():
    """A fresh interpreter: ``c_ept.state`` and the sweep script load none of the S1 modules (not even through another module), so a
    view-builder worker never pulls in the S1 runner chain (R10-h) and the state has no way to the S1 targets (R10-g)."""
    for name in ('c_ept.state', 'c_ept.checks.sweep_state'):
        code = f'import sys, {name}; sys.stdout.write(",".join(m for m in {S1_MODULES!r} if m in sys.modules))'
        r = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, encoding='utf-8')
        assert (r.returncode, r.stdout, r.stderr) == (0, '', ''), name


# ---- the golden matches -------------------------------------------------------------------------------------------------------------
def probe_refs(rec, per_match=24):
    """Reference times of one match: every regular frame, the game end, and both sides of the ms of kills and objectives (at most
    ``per_match`` of those event times, spread evenly)."""
    kinds = ('CHAMPION_KILL', 'BUILDING_KILL', 'ELITE_MONSTER_KILL', 'DRAGON_SOUL_GIVEN', 'TURRET_PLATE_DESTROYED')
    times = sorted({e['timestamp'] for e in rec.events if e['type'] in kinds})
    times = [times[int(k)] for k in np.linspace(0, len(times) - 1, min(per_match, len(times)))] if times else []
    return sorted({int(t) for t in rec.frame_ts} | {rec.T_ms} | set(times) | {t - 1 for t in times})


@pytest.fixture(scope='module')
def golden_states(golden_records):
    """For each golden match, ``[(ref, at_ref(rec, ref)), ...]`` at its probe times (made once: the functions cost a few ms a call)."""
    return [[(ref, at_ref(rec, ref)) for ref in probe_refs(rec)] for rec in golden_records]


def test_golden_every_building_kill_matches_a_landmark_and_none_hits_a_building_that_was_down(golden_records):
    """Measured 2026-10-21 on the 20 golden matches (all 15.18): 249 BUILDING_KILLs = 193 lane turrets + 23 nexus turrets + 33 inhibitors;
    every one fits a landmark, none was down already; four inhibitors are killed again after their 300 s (304 to 383 s later)."""
    counts = {}
    for rec in golden_records:
        for k in state.building_kills(rec):
            key = (what_it_hit(k), k.status)
            counts[key] = counts.get(key, 0) + 1
    assert counts == {('lane', 'ok'): 193, ('nexus', 'ok'): 23, ('inhibitor', 'ok'): 29, ('inhibitor', 'after_respawn'): 4}


def test_golden_the_index_equals_the_functions_at_every_probe_time_and_nothing_after_the_time_is_seen(golden_records, golden_states):
    n_refs = 0
    for m, (rec, states) in enumerate(zip(golden_records, golden_states)):
        index = state.StateIndex(rec)
        for ref, full in states:
            n_refs += 1
            assert same(at_ref_index(index, ref), full), (m, ref)
            if n_refs % 7 == 0:                                         # the cut match is rebuilt: sample it
                assert same(at_ref(truncated(rec, ref), ref), full), (m, ref)
    assert n_refs > 1_000


def test_golden_outputs_follow_the_dtype_range_and_nan_rules(golden_states):
    for m, states in enumerate(golden_states):
        for ref, outputs in states:
            try:
                check_policy(*outputs)
            except AssertionError as exc:
                raise AssertionError(f'match {m}, ref {ref}: {exc}') from None


def test_golden_the_standing_towers_agree_with_the_tower_counts_of_legacy(golden_states):
    """The landmark rows (the matching of kills to turrets, made here) against LEGACY's own count of standing towers per team and
    tier (made from the same events, keyed otherwise): they must agree at every probe time."""
    tiers = (('outer', 'OUTER_TURRET'), ('inner', 'INNER_TURRET'), ('base', 'BASE_TURRET'), ('nexus', 'NEXUS_TURRET'))
    for m, states in enumerate(golden_states):
        for ref, (_, _, obj, lm) in states:
            for team, side in ((100, 'blue'), (200, 'red')):
                for short, tier in tiers:
                    rows = [i for i, x in enumerate(state.LANDMARKS[:BUILDINGS]) if (x['team'], x['tower']) == (team, tier)]
                    assert lm[rows, 0].sum() == obj[COL[f'{side}_towers_{short}']], (m, ref, side, short)
                rows = [i for i, x in enumerate(state.LANDMARKS[:BUILDINGS]) if x['team'] == team and x['kind'] == 'inhibitor']
                assert (lm[rows, 0] == 0).sum() == obj[COL[f'{side}_inhibs_down']], (m, ref, side)


# ---- the sweep script ----------------------------------------------------------------------------------------------------------------
def write_store(folder, rows):
    d = folder / 's1' / 'matches'
    d.mkdir(parents=True)
    np.save(d / 'key.npy', np.array([r[0].encode('ascii') for r in rows], dtype='S24'))
    np.save(d / 'patch.npy', np.array([C.PATCHES.index(r[1]) for r in rows], dtype=np.int8))
    np.save(d / 'role.npy', np.array([C.ROLES.index(r[2]) for r in rows], dtype=np.int8))


def test_the_sweep_module_does_nothing_when_it_is_imported_and_explains_itself():
    r = subprocess.run([sys.executable, '-B', '-c', 'import c_ept.checks.sweep_state as m; assert callable(m.main)'], cwd=REPO,
                       capture_output=True, text=True, encoding='utf-8')
    assert (r.returncode, r.stdout, r.stderr) == (0, '', '')
    r = subprocess.run([sys.executable, '-B', '-m', 'c_ept.checks.sweep_state', '--help'], cwd=REPO, capture_output=True, text=True,
                       encoding='utf-8')
    text = ' '.join(r.stdout.split())
    assert r.returncode == 0 and 'S1 run folder' in text and '--out' in text and 'counts' in text and '--workers' in text, r.stderr


def test_the_sweep_stops_on_a_running_strata_server_before_it_reads_anything(tmp_path, fake_processes, capsys):
    fake_processes([synth.STRATA_SERVER])
    assert sweep_state.main([str(tmp_path / 'no_run_here')]) == 2
    err = capsys.readouterr().err
    assert 'E6' in err and 'does not stop' in err and 'no S1 store' not in err
    fake_processes([])
    assert sweep_state.main([str(tmp_path / 'no_run_here')]) == 2
    err = capsys.readouterr().err
    assert 'no S1 store' in err and 'E6' not in err
    assert list(tmp_path.iterdir()) == []


def test_the_sweep_does_not_write_into_the_s1_run_folder(tmp_path, capsys):
    write_store(tmp_path / 'run', [('SYN_a', '15.18', 'FIT')])
    (tmp_path / 'run' / 's1' / 'STORE.json').write_text('{}', encoding='utf-8')
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*'))
    for run in (tmp_path / 'run', tmp_path / 'run' / 's1'):
        with pytest.raises(SystemExit) as err:
            sweep_state.main([str(run), '--out', str(tmp_path / 'run' / 'counts.json')])
        assert err.value.code == 2 and 'S1 run folder' in capsys.readouterr().err
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*')) == before


def test_one_synthetic_match_is_counted_by_the_sweep():
    """``count_record`` on a match whose numbers are known by hand: 5 building kills (a tower, the same tower again, one that fits no
    landmark, one far from its anchor, one without a position); the states are taken at the regular frames and at the game end."""
    rec = synth.make_record(T_ms=300_000, objectives=[tower_kill(100_000, 100, 'MID_LANE', 'OUTER_TURRET'),
                                                      tower_kill(110_000, 100, 'MID_LANE', 'OUTER_TURRET'),
                                                      dict(tower_kill(120_000, 100, 'MID_LANE', 'INNER_TURRET'), team=None),
                                                      tower_kill(130_000, 200, 'TOP_LANE', 'OUTER_TURRET', at=(0, 0)),
                                                      tower_kill(140_000, 200, 'MID_LANE', 'OUTER_TURRET', at=None)])
    got = sweep_state.count_record(rec)
    assert got['matches'] == 1 and got['building_kill_events'] == 5 and got['building_kills_unmatched'] == 1
    assert got['building_kills_far_from_their_landmark'] == 1 and got['building_kills_without_position'] == 1
    assert got['building_kills_on_destroyed_landmark'] == 1 and got['nexus_rekills_after_respawn'] == 0
    assert got['state_refs'] == len(rec.frame_ts) + 1 == 6 and got['landmark_cells'] == 68 * 6
    assert got['landmark_cells_not_finite'] == got['landmark_cells_not_finite_other'] == 0
    assert got['index_vs_function_mismatch'] == got['tower_count_mismatch'] == got['policy_violations'] == 0


def test_golden_the_sweep_runs_on_a_few_matches_and_shows_no_key(golden_records, tmp_path, capsys):
    run = os.environ['C_EPT_S1_RUN']
    out = tmp_path / 'counts.json'
    assert sweep_state.main([run, '--limit', '3', '--workers', '1', '--out', str(out)]) == 0
    printed = capsys.readouterr().out
    written = out.read_text(encoding='utf-8')
    got = json.loads(written)
    counts = got['counts']
    assert got['script'] == sweep_state.SCRIPT and got['run'] == common.run_folder(run).name and got.get('errors', {}) == {}
    assert counts['matches'] == 3 and counts['building_kill_events'] > 0 and counts['state_refs'] > 0
    assert counts['matches_with_exception'] == 0
    assert counts['building_kills_unmatched'] == counts['building_kills_on_destroyed_landmark'] == counts['policy_violations'] == 0
    keys = [rec.key for rec in golden_records] + [key for key, _, _ in common.s1_matches(run)[:3]]    # the golden and the three read
    assert not any(key in printed or key in written for key in keys)
    sample = tmp_path / 'sample.json'                                          # --sample N --seed S draws N matches of the store
    assert sweep_state.main([run, '--sample', '4', '--seed', '7', '--workers', '1', '--out', str(sample)]) == 0
    again = tmp_path / 'again.json'
    assert sweep_state.main([run, '--sample', '4', '--seed', '7', '--workers', '1', '--out', str(again)]) == 0
    assert sample.read_text(encoding='utf-8') == again.read_text(encoding='utf-8')
    assert json.loads(sample.read_text(encoding='utf-8'))['counts']['matches'] == 4
