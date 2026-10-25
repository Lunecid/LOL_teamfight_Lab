"""Tests of ``c_ept.tokens``: the layered view of one match at a reference time.

The synthetic tests build matches with ``synth.make_record`` plus raw events written here, on the open-grid terrain of ``conftest.py``
(every cell walkable; its all-pairs table is built once per session). They check hand-made numbers, and a plain-loop oracle that reads
the raw event dicts and knows nothing of the module's arrays. The golden tests (``golden_records``, the 20 real matches of the S1 golden
lock) run only with ``C_EPT_S1_RUN`` set; their messages carry the position of the match in the golden list, never its key. Nothing here
calls a CLI ``main()`` or a builder with default roots (S2-6). Run with ``--tb=short``.
"""
import ast
import dataclasses
import inspect
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from c_ept import config as C
from c_ept.domain import state, targets
from c_ept.preprocessing import tokens, traits
from c_ept.data.reader import MatchFormatError
from c_ept.tests import synth
from c_ept.tests.conftest import (PROTECTED_IMPORT_PRELUDE, all_imports, import_watch_names,
                                  module_level_imports, package_of)
from c_ept.tests.view_helpers import (K, LM_XY, NAN, P, REF, S, TYPE, StubTerrain, cut_at, damage, differing, extra_event, frame_at,
                                      frames_with, make_ctx, nan_cells, nan_conditions_broken, nan_outside_allowed, old_total, raw_building,
                                      raw_elite, raw_item, raw_kill, raw_level, raw_plate, raw_ward_kill, raw_ward_placed, record,
                                      rich_events, views_equal)

REPO = Path(C.__file__).parents[1]
S1_MODULES = import_watch_names(('c_ept.targets', 'c_ept.event_rows', 'c_ept.frame_rows', 'c_ept.store_core', 'c_ept.run_s1'))


# ---- the oracle: the raw events read in plain loops -------------------------------------------------------------------------------
def pid_side(pid):
    """0 blue (1-5), 1 red (6-10), 2 no champion."""
    return 0 if 1 <= pid <= 5 else 1 if 6 <= pid <= 10 else 2


def bit(pid):
    return 1 << (pid - 1) if 1 <= pid <= 10 else 0


def assist_ids(e):
    """The distinct participants 1-10 of the assist list, ascending (integers outside 1-10 are not players and are ignored)."""
    return sorted({a for a in (e.get('assistingParticipantIds') or ()) if type(a) is int and 1 <= a <= 10})


def oracle_event(e):
    """(type index, team, actor mask, gold, x, y) of a raw event of the 8 types, by the rules of the module text."""
    typ = e['type']
    killer = e.get('killerId', 0)
    taker = {100: 1, 200: 0}.get(e.get('teamId'), 2)                     # a building or plate: the owner lost it, the other team took it
    if typ == 'CHAMPION_KILL':
        team, actors, gold = pid_side(killer), bit(killer) | bit(e['victimId']) | sum(bit(a) for a in assist_ids(e)), \
            e.get('bounty', 0) + e.get('shutdownBounty', 0)
    elif typ == 'ELITE_MONSTER_KILL':
        team = {100: 0, 200: 1}.get(e.get('killerTeamId'), 2)
        actors, gold = bit(killer) | sum(bit(a) for a in assist_ids(e)), e.get('bounty', 0)
    elif typ == 'BUILDING_KILL':
        team, actors, gold = taker, bit(killer) | sum(bit(a) for a in assist_ids(e)), e.get('bounty', 0)
    elif typ == 'TURRET_PLATE_DESTROYED':
        team, actors, gold = taker, bit(killer), 0
    else:
        actor_field = {'ITEM_PURCHASED': 'participantId', 'LEVEL_UP': 'participantId', 'WARD_PLACED': 'creatorId', 'WARD_KILL': 'killerId'}
        pid = e.get(actor_field[typ], 0)
        team, actors, gold = pid_side(pid), bit(pid), 0
    pos = e.get('position')
    return TYPE[typ], team, actors, float(gold), (pos['x'], pos['y']) if pos else (NAN, NAN)


def ordered(pids, ref):
    """The participants on the team ``ref`` first, then the others; ascending within each group (the order a team swap keeps)."""
    return sorted(pids, key=lambda p: (pid_side(p) != ref, p))


def pairs_of(members):
    return [(members[a] - 1, members[b] - 1) for a in range(len(members)) for b in range(a + 1, len(members))]


def oracle_edges_of(e):
    """The (i, j, kind) edges of one raw kill or objective event, in the order of the module text; [] for any other event."""
    typ = e['type']
    killer = e.get('killerId', 0)
    assists = assist_ids(e)
    if typ == 'CHAMPION_KILL':
        victim = e['victimId']
        ref = pid_side(killer) if killer else 1 - pid_side(victim)                       # the side of the kill
        helpers = ordered([a for a in assists if a not in (killer, victim)], ref)       # neither the killer (kind 0) nor the victim is one
        out = [(killer - 1, victim - 1, 0)] if killer else []
        out += [(a - 1, victim - 1, 1) for a in helpers]
        out += [(i, j, 2) for i, j in pairs_of(([killer] if killer else []) + helpers)]
        dealers = sorted({d['participantId'] for d in e.get('victimDamageReceived') or ()
                          if isinstance(d, dict) and type(d.get('participantId')) is int and 1 <= d['participantId'] <= 10
                          and pid_side(d['participantId']) != pid_side(victim)})
        return out + [(d - 1, victim - 1, 3) for d in dealers]
    if typ in ('ELITE_MONSTER_KILL', 'BUILDING_KILL'):
        team = oracle_event(e)[1]
        ref = team if team != 2 else pid_side(killer)
        return [(i, j, 4) for i, j in pairs_of(([killer] if killer else []) + ordered([a for a in assists if a != killer], ref))]
    return []


def oracle_arrays(rec, ref):
    """{ev_*, ed_*, old} of a view at ``ref`` from the raw events of ``rec``, written as plain loops."""
    rows = [(e['timestamp'], e) for e in rec.events if e['type'] in C.EVENT_TYPES]
    window = [(t, e) for t, e in rows if ref - C.RECENT_MS < t <= ref][::-1]            # the newest first; the later of two in a ms first
    kept = window[:C.MAX_EVENTS]
    cap = C.MAX_EVENTS
    ev = dict(ev_type=np.full(cap, -1), ev_dt=np.zeros(cap), ev_team=np.full(cap, 2), ev_xy=np.full((cap, 2), NAN),
              ev_actors=np.zeros(cap, int), ev_gold=np.zeros(cap))
    for r, (t, e) in enumerate(kept):
        ev['ev_type'][r], ev['ev_team'][r], ev['ev_actors'][r], ev['ev_gold'][r], ev['ev_xy'][r] = oracle_event(e)
        ev['ev_dt'][r] = (ref - t) / 1000.0
    window_edges = [(i, j, (ref - t) / 1000.0, kind) for t, e in window for i, j, kind in oracle_edges_of(e)]
    edges = window_edges[:C.MAX_EDGES]
    ed = dict(ed_i=np.full(128, -1), ed_j=np.full(128, -1), ed_dt=np.zeros(128), ed_kind=np.full(128, -1))
    for r, (i, j, dt, kind) in enumerate(edges):
        ed['ed_i'][r], ed['ed_j'][r], ed['ed_dt'][r], ed['ed_kind'][r] = i, j, dt, kind
    old = np.zeros(16)
    for t, e in rows:
        if ref - C.OLD_MS < t <= ref - C.RECENT_MS:
            ti, team, *_ = oracle_event(e)
            if team < 2:
                old[2 * ti + team] += 1
    return dict(ev, **ed, ev_n=len(kept), ev_dropped=len(window) - len(kept), ed_n=len(edges), ed_dropped=len(window_edges) - len(edges),
                old=old)


def check_against_oracle(v, rec, ref):
    """Assert that the token, edge and old-count arrays of the view ``v`` are the oracle's. Returns the oracle."""
    o = oracle_arrays(rec, ref)
    for key in ('ev_type', 'ev_team', 'ev_actors', 'ed_i', 'ed_j', 'ed_kind'):
        assert np.array_equal(v[key], o[key]), key
    for key in ('ev_dt', 'ev_gold', 'ed_dt'):
        assert np.array_equal(v[key], o[key].astype(np.float32)), key
    assert np.array_equal(v['ev_xy'], o['ev_xy'].astype(np.float32), equal_nan=True), 'ev_xy'
    counts = ('ev_n', 'ev_dropped', 'ed_n', 'ed_dropped')
    assert [int(v[key]) for key in counts] == [o[key] for key in counts]
    assert np.array_equal(v['summary'][2:18], o['old'].astype(np.float32)), 'old counts'
    return o


# ---- fixtures: the helpers of the brief -----------------------------------------------------------------------------------------------
@pytest.fixture
def ctx(open_terrain):
    return make_ctx(open_terrain, rich_events())


@pytest.fixture
def ctx_with_future_event(open_terrain):
    """``.base``: a context; ``.more``: the same match with one more kill (with a damage list) one millisecond after ``REF``."""
    return types.SimpleNamespace(base=make_ctx(open_terrain, rich_events()),
                                 more=make_ctx(open_terrain, rich_events() + [extra_event('CHAMPION_KILL', REF + 1)]))


@pytest.fixture
def ctx_with_n_events(open_terrain):
    """``f(n, step_ms=1000)``: a context with n ward events, one every ``step_ms``, the newest on ``REF``."""
    return lambda n, step_ms=1000: make_ctx(open_terrain, [raw_ward_placed(REF - step_ms * i, 1) for i in range(n)])


@pytest.fixture
def ctx_events_at(open_terrain):
    """``f(times)``: a context with one ward event of the blue player 1 at each of the times."""
    return lambda times: make_ctx(open_terrain, [raw_ward_placed(t, 1) for t in times])


@pytest.fixture
def ctx_kill_with_damage(open_terrain):
    """``f(victim, dealers)``: a context with a kill of ``victim`` 5 s before ``REF`` by player 1 (or 6), with the damage of ``dealers``."""
    return lambda victim, dealers: make_ctx(open_terrain, [raw_kill(REF - 5_000, 1 if victim > 5 else 6, victim, dealers=dealers)])


# ---- the brief's tests ------------------------------------------------------------------------------------------------------------
def test_no_future_events_in_view(ctx_with_future_event):                 # an event at ref + 1 leaves the view the same
    a = tokens.build_view(ctx_with_future_event.base, REF, K)
    b = tokens.build_view(ctx_with_future_event.more, REF, K)
    assert views_equal(a, b)


def test_frame_row_view_hides_events_after_t0(ctx, open_terrain):
    t0 = int(ctx.rec.frame_ts[5])
    v = tokens.build_view(ctx, ctx.rec.frame_ts[5], 5)
    assert np.all(v['ev_dt'][:v['ev_n']] >= 0)
    cut = make_ctx(open_terrain, [e for e in rich_events() if e['timestamp'] <= t0])             # the same match without anything after t0
    assert max(e['timestamp'] for e in rich_events()) > t0 and views_equal(v, tokens.build_view(cut, t0, 5))


def test_event_cap_keeps_most_recent(ctx_with_n_events):
    cap, extra = C.MAX_EVENTS, 5
    v = tokens.build_view(ctx_with_n_events(cap + extra, step_ms=100), REF, K)        # an event every 0.1 s: all inside the 120 s window
    assert (v['ev_n'], v['ev_dropped']) == (cap, extra) and v['ev_dt'][0] == 0
    assert np.all(np.diff(v['ev_dt']) > 0) and np.allclose(v['ev_dt'], np.arange(cap) / 10)      # the cap newest; the oldest five dropped


def test_old_counts_window(ctx_events_at):
    v = tokens.build_view(ctx_events_at([REF - 600_000, REF - 599_999, REF - 120_000, REF - 119_999]), REF, K)
    assert old_total(v) == 2 and v['ev_n'] == 1


def test_edges_drop_same_team_damage_entries(ctx_kill_with_damage):
    v = tokens.build_view(ctx_kill_with_damage(victim=6, dealers=[1, 2, 7]), REF, K)
    assert set(zip(v['ed_i'][:v['ed_n']], v['ed_kind'][:v['ed_n']])) >= {(0, 3), (1, 3)}
    assert (6, 3) not in set(zip(v['ed_i'], v['ed_kind']))


def test_view_columns_have_no_ids():
    assert not [n for n in tokens.PLAYER_NUM + tokens.SUMMARY if n.endswith('_id') or n.endswith('Id')]


def test_frame_age(ctx):
    v = tokens.build_view(ctx, ctx.rec.frame_ts[5] + 25_000, 5)
    assert v['summary'][tokens.SUMMARY.index('frame_age_s')] == pytest.approx(25.0)


# ---- the layout: names, shapes, dtypes, bytes --------------------------------------------------------------------------------------
def test_player_num_and_summary_layout():
    assert len(tokens.PLAYER_NUM) == 81 and len(set(tokens.PLAYER_NUM)) == 81
    assert tokens.PLAYER_NUM[:33] == C.FRAME_FIELDS
    assert tokens.PLAYER_NUM[33:40] == ('prev_x', 'prev_y', 'has_prev', 'level_now', 'alive', 'respawn_s', 'wall_fraction')
    assert tokens.PLAYER_NUM[40:59] == traits.TRAIT_NAMES and tokens.PLAYER_NUM[59:] == traits.item_vector_names()
    assert len(tokens.SUMMARY) == 107 and len(set(tokens.SUMMARY)) == 107
    assert tokens.SUMMARY[:2] == ('game_time_s', 'frame_age_s')
    assert tokens.SUMMARY[2:18] == tuple(f'old_{t.lower()}_{s}' for t in C.EVENT_TYPES for s in 'br')
    assert tokens.SUMMARY[2] == 'old_champion_kill_b' and tokens.SUMMARY[17] == 'old_ward_kill_r'
    assert tokens.SUMMARY[18:] == state.OBJ_STATE_NAMES


EV_CAP, ED_CAP = C.MAX_EVENTS, C.MAX_EDGES        # the caps (event cap 128 since the S2 gate S2G-1, 2026-10-21; it was 80)
VIEW_SHAPES = {'players_num': ((10, 81), 'f4'), 'players_zone': ((10,), 'i1'), 'landmarks': ((34, 2), 'f4'),
               'ev_type': ((EV_CAP,), 'i1'), 'ev_dt': ((EV_CAP,), 'f4'), 'ev_team': ((EV_CAP,), 'i1'), 'ev_xy': ((EV_CAP, 2), 'f4'),
               'ev_actors': ((EV_CAP,), 'i2'), 'ev_gold': ((EV_CAP,), 'f4'), 'ev_n': ((), 'i2'), 'ev_dropped': ((), 'i2'),
               'ed_i': ((ED_CAP,), 'i1'), 'ed_j': ((ED_CAP,), 'i1'), 'ed_dt': ((ED_CAP,), 'f4'), 'ed_kind': ((ED_CAP,), 'i1'),
               'ed_n': ((), 'i2'), 'ed_dropped': ((), 'i2'), 'summary': ((107,), 'f4'), 'geo_pp': ((10, 10), 'f2'),
               'geo_pl': ((10, 34), 'f2')}


def test_view_keys_shapes_and_dtypes_are_the_briefs(ctx):
    v = tokens.build_view(ctx, REF, K)
    assert list(v) == list(VIEW_SHAPES)
    for key, (shape, dtype) in VIEW_SHAPES.items():
        assert isinstance(v[key], np.ndarray) and v[key].shape == shape and v[key].dtype == np.dtype(dtype), key
        assert v[key].flags['C_CONTIGUOUS'] and v[key].flags['OWNDATA']                  # a new array, not a view into the context
    assert {k: (a.shape, a.dtype.str[1:]) for k, a in v.items()} == {k: (s, d) for k, (s, d) in VIEW_SHAPES.items()}


def test_view_spec_and_bytes_per_view_are_the_actual_dtypes(ctx):
    v = tokens.build_view(ctx, REF, K)
    assert list(tokens.VIEW_SPEC) == list(v) and all(tokens.VIEW_SPEC[k] == (a.shape, a.dtype) for k, a in v.items())
    # worked out by hand from the dtypes: 8,294 bytes at the caps 128 and 128 (it was 7,334 at the old event cap 80)
    by_hand = (10 * 81 * 4 + 10 * 1 + 34 * 2 * 4                               # players_num, players_zone, landmarks
               + EV_CAP * (1 + 4 + 1 + 2 * 4 + 2 + 4) + 2 + 2                  # ev_type, _dt, _team, _xy, _actors, _gold; ev_n, ev_dropped
               + ED_CAP * (1 + 1 + 4 + 1) + 2 + 2                              # ed_i, ed_j, ed_dt, ed_kind; ed_n, ed_dropped
               + 107 * 4 + 10 * 10 * 2 + 10 * 34 * 2)                          # summary, geo_pp, geo_pl
    assert tokens.bytes_per_view() == sum(a.nbytes for a in v.values()) == by_hand


def test_nan_allowed_names_real_columns_and_every_entry_says_when():
    nan_allowed = tokens.NAN_ALLOWED
    assert set(nan_allowed) == {'players_num', 'summary', 'landmarks', 'ev_xy', 'geo_pp', 'geo_pl'}
    assert all(set(entry) == {'cells', 'when'} and isinstance(entry['when'], str) and entry['when'] for entry in nan_allowed.values())
    assert set(nan_allowed['summary']['cells']) == set(state.NAN_FEATURES) <= set(tokens.SUMMARY)
    assert set(nan_allowed['players_num']['cells']) <= set(tokens.PLAYER_NUM)
    assert 'champ_unknown' not in nan_allowed['players_num']['cells'] and 'x' in nan_allowed['players_num']['cells']
    cells = nan_allowed['landmarks']['cells']                                                   # (row, column 1) cells, not whole rows
    assert all(isinstance(c, tuple) and len(c) == 2 and c[1] == 1 for c in cells)
    assert [state.LANDMARKS[row]['kind'] for row, _ in cells] == ['herald', 'grubs', 'atakhan', 'atakhan']
    assert nan_allowed['ev_xy']['cells'] == nan_allowed['geo_pp']['cells'] == nan_allowed['geo_pl']['cells'] == 'all'


def test_nan_allowed_is_exhaustive_over_the_float_arrays_of_a_view():
    """Every float array of a view is in the map, except the three that are never NaN; and the module text says the map is exhaustive."""
    floats = {key for key, (shape, dtype) in tokens.VIEW_SPEC.items() if dtype.kind == 'f'}
    assert floats - set(tokens.NAN_ALLOWED) == {'ev_dt', 'ev_gold', 'ed_dt'}
    assert set(tokens.NAN_ALLOWED) <= floats
    assert 'exhaustive' in tokens.__doc__


def test_landmark_positions_are_those_of_the_landmarks():
    assert np.array_equal(tokens.LANDMARK_XY, LM_XY) and tokens.LANDMARK_XY.shape == (34, 2)
    assert not tokens.LANDMARK_XY.flags.writeable


# ---- tokens ---------------------------------------------------------------------------------------------------------------------
def test_token_fields_by_event_type(open_terrain):
    """One event of each type, hand-computed: the type index, the age in s, the team, the actor bits, the gold, the position."""
    events = [raw_kill(REF - 9_000, 1, 6, assists=[2, 3], x=6000, y=6100, bounty=300, shutdown=150),
              raw_elite(REF - 8_000, 7, 200, assists=[8, 9], x=9837, y=4401, bounty=0),
              raw_building(REF - 7_000, 100, 'TOP_LANE', killer=7, x=981, y=10441, bounty=250),
              raw_plate(REF - 6_000, 200, 'TOP_LANE', killer=2, x=4318, y=13875),
              raw_item(REF - 5_000, 8), raw_level(REF - 4_000, 5, 6), raw_ward_placed(REF - 3_000, 10), raw_ward_kill(REF - 2_000, 4),
              raw_kill(REF - 1_000, 0, 3, x=4000, y=5000, bounty=100)]
    v = tokens.build_view(make_ctx(open_terrain, events), REF, K)
    assert (v['ev_n'], v['ev_dropped']) == (9, 0)
    rows = [  # type, dt, team, actors, gold, x, y (the newest first)
        ('CHAMPION_KILL', 1.0, 2, bit(3), 100.0, 4000, 5000),                               # an execution: no team; the victim is the actor
        ('WARD_KILL', 2.0, 0, bit(4), 0.0, NAN, NAN),
        ('WARD_PLACED', 3.0, 1, bit(10), 0.0, NAN, NAN),
        ('LEVEL_UP', 4.0, 0, bit(5), 0.0, NAN, NAN),
        ('ITEM_PURCHASED', 5.0, 1, bit(8), 0.0, NAN, NAN),
        ('TURRET_PLATE_DESTROYED', 6.0, 0, bit(2), 0.0, 4318, 13875),                           # red lost the plate: blue took it
        ('BUILDING_KILL', 7.0, 1, bit(7), 250.0, 981, 10441),                                   # blue lost the turret: red took it
        ('ELITE_MONSTER_KILL', 8.0, 1, bit(7) | bit(8) | bit(9), 0.0, 9837, 4401),             # killerTeamId 200
        ('CHAMPION_KILL', 9.0, 0, bit(1) | bit(6) | bit(2) | bit(3), 450.0, 6000, 6100)]     # killer, victim and assists; bounty + shutdown
    for r, (typ, dt, team, actors, gold, x, y) in enumerate(rows):
        got = (v['ev_type'][r], v['ev_dt'][r], v['ev_team'][r], v['ev_actors'][r], v['ev_gold'][r])
        assert got == (TYPE[typ], dt, team, actors, gold), r
        assert np.array_equal(v['ev_xy'][r], np.array([x, y], dtype=np.float32), equal_nan=True), r
    assert bit(1) | bit(6) | bit(2) | bit(3) == 39 and bit(10) == 512
    pad = slice(9, C.MAX_EVENTS)                                                               # the rows after the last token are padding
    assert np.all(v['ev_type'][pad] == -1) and np.all(v['ev_team'][pad] == 2) and np.all(v['ev_actors'][pad] == 0)
    assert np.all(v['ev_dt'][pad] == 0) and np.all(v['ev_gold'][pad] == 0) and np.all(np.isnan(v['ev_xy'][pad]))


def test_unknown_team_ids_and_players_without_a_champion_give_no_team(open_terrain):
    """A team id that is not 100 or 200, and an actor that is no champion (participant 0), give team 2 and no actor bit."""
    events = [raw_elite(REF - 5_000, 1, 300), raw_building(REF - 4_000, 200, 'MID_LANE', killer=1), raw_item(REF - 3_000, 0),
              raw_ward_placed(REF - 2_000, 0), raw_plate(REF - 1_000, 100, 'MID_LANE', killer=0)]
    events[1]['teamId'] = 300                                                                  # not a team id (LEGACY reads int(teamId))
    c = make_ctx(open_terrain, events)
    v = tokens.build_view(c, REF, K)
    o = check_against_oracle(v, c.rec, REF)
    assert list(v['ev_team'][:5]) == [1, 2, 2, 2, 2] and list(o['ev_team'][:5]) == [1, 2, 2, 2, 2]   # only the plate (taker red) has a team
    assert list(v['ev_actors'][:5]) == [0, 0, 0, bit(1), bit(1)]                      # the newest first: plate, ward, item, building, elite
    later = REF + 200_000                                                                  # they are old by then: only the plate is counted
    assert old_total(tokens.build_view(c, later, frame_at(c.rec, later))) == 1


def test_other_event_types_make_no_token(open_terrain):
    events = [raw_ward_placed(REF - 3_000, 1), {'type': 'SKILL_LEVEL_UP', 'timestamp': REF - 2_000, 'participantId': 1, 'skillSlot': 1},
              {'type': 'ITEM_DESTROYED', 'timestamp': REF - 1_500, 'participantId': 1, 'itemId': 1055}]
    v = tokens.build_view(make_ctx(open_terrain, events), REF, K)
    assert v['ev_n'] == 1 and v['ev_type'][0] == TYPE['WARD_PLACED']


def test_a_match_without_any_token_event_has_empty_token_and_edge_arrays(open_terrain):
    ctx = make_ctx(open_terrain)
    for ref, k in ((0, 0), (120_300, 2), (REF, K)):
        v = tokens.build_view(ctx, ref, k)
        assert (v['ev_n'], v['ev_dropped'], v['ed_n']) == (0, 0, 0) and old_total(v) == 0
        assert np.all(v['ev_type'] == -1) and np.all(v['ed_i'] == -1) and np.all(v['ed_kind'] == -1)
        check_against_oracle(v, ctx.rec, ref)


MALFORMED = [  # (what is wrong, the event, the word the error names)
    ('killer id is text', lambda t: {**raw_kill(t, 1, 6), 'killerId': 'x'}, 'killerId'),
    ('killer id above 10', lambda t: {**raw_kill(t, 1, 6), 'killerId': 11}, 'killerId'),
    ('victim id missing', lambda t: {k: v for k, v in raw_kill(t, 1, 6).items() if k != 'victimId'}, 'victimId'),
    ('assists not a list', lambda t: {**raw_kill(t, 1, 6), 'assistingParticipantIds': 3}, 'assistingParticipantIds'),
    ('assist entry is a bool', lambda t: {**raw_kill(t, 1, 6), 'assistingParticipantIds': [True]}, 'assistingParticipantIds'),
    ('bounty is text', lambda t: {**raw_kill(t, 1, 6), 'bounty': 'much'}, 'bounty'),
    ('bounty is a bool', lambda t: {**raw_elite(t, 1, 100), 'bounty': True}, 'bounty'),
    ('creator id is null', lambda t: {**raw_ward_placed(t, 1), 'creatorId': None}, 'creatorId'),
    ('item participant missing', lambda t: {k: v for k, v in raw_item(t, 1).items() if k != 'participantId'}, 'participantId'),
    ('ward killer is a float', lambda t: {**raw_ward_kill(t, 1), 'killerId': 1.0}, 'killerId'),
]


@pytest.mark.parametrize('what, make, word', MALFORMED, ids=[m[0] for m in MALFORMED])
def test_a_malformed_event_stops_the_context_with_a_key_free_format_error(open_terrain, what, make, word):
    with pytest.raises(MatchFormatError, match=word) as err:
        make_ctx(open_terrain, [make(REF - 5_000)])
    assert 'SYN_0' not in str(err.value) and '15.18' in str(err.value)                          # the patch is named, the match key is not


def test_the_context_needs_the_usual_team_layout(open_terrain):
    rec = record(rich_events())
    odd = dataclasses.replace(rec, team=np.array([100, 100, 100, 100, 200, 200, 200, 200, 200, 200], dtype=np.int16))
    with pytest.raises(ValueError, match='participants 1-5'):
        tokens.MatchContext(odd, open_terrain)
    with pytest.raises(TypeError):
        tokens.MatchContext('not a record', open_terrain)


def test_events_out_of_time_order_are_a_format_error_without_the_key(open_terrain):
    rec = record([raw_ward_placed(10_000, 1), raw_ward_placed(20_000, 2)])
    wards = [e for e in rec.events if e['type'] == 'WARD_PLACED']
    bad = dataclasses.replace(rec, events=(*reversed(wards), rec.events[-1]))                   # 20,000 ms before 10,000 ms
    with pytest.raises(MatchFormatError, match='time order') as err:
        tokens.MatchContext(bad, open_terrain)
    assert 'SYN_0' not in str(err.value) and '15.18' in str(err.value)


def edge_list(v):
    """The (i, j, kind) edges of a view, in order."""
    n = int(v['ed_n'])
    return list(zip(v['ed_i'][:n].tolist(), v['ed_j'][:n].tolist(), v['ed_kind'][:n].tolist()))


def test_an_assist_list_with_the_killer_the_victim_repeats_or_non_players_gives_the_edges_of_the_clean_list(open_terrain):
    """Kind 1 must not repeat the killer's kind-0 edge and kind 2 must not pair the victim; ids that are no player are ignored."""
    clean = raw_kill(REF - 5_000, 1, 6, assists=[2, 3], dealers=[1, 2])
    dirty = raw_kill(REF - 5_000, 1, 6, assists=[1, 6, 3, 2, 2, 11, 0, -3], dealers=[1, 2])
    a = tokens.build_view(make_ctx(open_terrain, [clean]), REF, K)
    c = make_ctx(open_terrain, [dirty])
    b = tokens.build_view(c, REF, K)
    assert differing(a, b) == []
    edges = edge_list(b)
    assert edges.count((0, 5, 0)) == 1 and (0, 5, 1) not in edges                               # the killer is the kind-0 edge only
    assert not [e for e in edges if e[2] == 2 and 5 in e[:2]]                                   # the victim pairs with nobody
    assert [e[:2] for e in edges if e[2] == 1] == [(1, 5), (2, 5)]
    check_against_oracle(b, c.rec, REF)


def test_the_victim_in_the_assist_list_of_an_execution_is_no_assister_and_no_pair(open_terrain):
    c = make_ctx(open_terrain, [raw_kill(REF - 5_000, 0, 3, assists=[3, 7], x=4000, y=5000)])
    v = tokens.build_view(c, REF, K)
    assert edge_list(v) == [(6, 2, 1)]                                                      # player 7 assisted; nobody is left to pair with
    check_against_oracle(v, c.rec, REF)


def test_events_of_one_millisecond_come_the_later_of_the_file_first(open_terrain):
    events = [raw_ward_placed(REF - 1000, 1), raw_ward_placed(REF - 1000, 2), raw_ward_placed(REF - 1000, 3),
              raw_ward_placed(REF - 2000, 4)]
    v = tokens.build_view(make_ctx(open_terrain, events), REF, K)
    assert list(v['ev_actors'][:4]) == [bit(3), bit(2), bit(1), bit(4)] and list(v['ev_dt'][:4]) == [1.0, 1.0, 1.0, 2.0]


def test_the_recent_window_is_open_below_and_closed_above(open_terrain):
    events = [raw_ward_placed(REF - 120_000, 1), raw_ward_placed(REF - 119_999, 2), raw_ward_placed(REF, 3), raw_ward_placed(REF + 1, 4)]
    v = tokens.build_view(make_ctx(open_terrain, events), REF, K)
    assert v['ev_n'] == 2 and list(v['ev_actors'][:2]) == [bit(3), bit(2)] and v['ev_dt'][1] == pytest.approx(119.999, abs=1e-4)
    assert old_total(v) == 1                                                                 # REF - 120,000 is the newest of the old events


def test_the_token_cap_drops_the_oldest_and_counts_them(open_terrain):
    cap, extra = C.MAX_EVENTS, 20
    ctx = make_ctx(open_terrain, [raw_ward_placed(REF - 10 * i, 1 + i % 10) for i in range(cap + extra)])      # one event every 10 ms
    v = tokens.build_view(ctx, REF, K)
    o = check_against_oracle(v, ctx.rec, REF)
    assert (v['ev_n'], v['ev_dropped']) == (cap, extra) and o['ev_dropped'] == extra
    assert v['ev_dt'][cap - 1] == pytest.approx((cap - 1) / 100)


# ---- the team of a building, a plate and a monster is the one the objective table gives ----------------------------------------------
def test_team_column_agrees_with_the_objective_table_on_the_synthetic_objectives(open_terrain):
    rec = synth.records_from_seed('SYN_7', '15.19')
    ctx = tokens.MatchContext(rec, open_terrain)
    taker = {id(e): targets._taker(e) for e in rec.events if e['type'] in targets.OBJECTIVE_TYPES}
    assert taker
    ev = ctx.events
    rows = [e for e in rec.events if e['type'] in C.EVENT_TYPES]
    for j, e in enumerate(rows):
        if id(e) in taker:
            assert int(ev.team[j]) == (taker[id(e)] if taker[id(e)] >= 0 else 2), j


# ---- edges ----------------------------------------------------------------------------------------------------------------------
def test_edges_by_hand(open_terrain):
    """A kill with assists and a damage list, a dragon, a turret and a plate: the five kinds, their order and their ages."""
    events = [raw_kill(REF - 10_000, 1, 6, assists=[3, 2], x=6000, y=6000, dealers=[4, 1, 7, 2, 4], extra_damage=[damage(0), damage(6)]),
              raw_elite(REF - 4_000, 7, 200, assists=[9, 8]),
              raw_building(REF - 3_000, 100, 'MID_LANE', 'TOWER_BUILDING', 'OUTER_TURRET', killer=7, assists=[8], x=5846, y=6396),
              raw_plate(REF - 2_000, 100, 'TOP_LANE', killer=7, x=981, y=10441)]
    v = tokens.build_view(make_ctx(open_terrain, events), REF, K)
    expected = [(6, 7, 3.0, 4),                                                     # the turret (the newest with edges): killer 7, assist 8
                (6, 7, 4.0, 4), (6, 8, 4.0, 4), (7, 8, 4.0, 4),                            # the dragon: killer 7, assists 8 and 9, in pairs
                (0, 5, 10.0, 0),                                                                # killer 1 - victim 6
                (1, 5, 10.0, 1), (2, 5, 10.0, 1),                                             # assists 2 and 3 - victim, in ascending order
                (0, 1, 10.0, 2), (0, 2, 10.0, 2), (1, 2, 10.0, 2),                              # the killing side: 1, 2, 3 in pairs
                (0, 5, 10.0, 3), (1, 5, 10.0, 3), (3, 5, 10.0, 3)]                # dealers 1, 2, 4 (7 is on the victim's team, 0 no player)
    assert v['ed_n'] == len(expected)
    n = int(v['ed_n'])
    got = list(zip(v['ed_i'][:n].tolist(), v['ed_j'][:n].tolist(), v['ed_dt'][:n].tolist(), v['ed_kind'][:n].tolist()))
    assert got == expected
    assert np.all(v['ed_i'][v['ed_n']:] == -1) and np.all(v['ed_j'][v['ed_n']:] == -1) and np.all(v['ed_kind'][v['ed_n']:] == -1)
    assert np.all(v['ed_dt'][v['ed_n']:] == 0)
    assert v['ed_dropped'] == 0                                                                  # 13 edges, far below the cap


def test_assisters_on_both_teams_come_the_team_of_the_event_first(open_terrain):
    """Real objective events list assisters of both teams. The team of the event first, each team ascending, whatever the ids: the
    order is the one a team swap keeps (test_symmetry checks that against a mirrored match)."""
    def pairs(ctx):
        v = tokens.build_view(ctx, REF, K)
        return list(zip(v['ed_i'][:v['ed_n']].tolist(), v['ed_j'][:v['ed_n']].tolist()))

    blue = pairs(make_ctx(open_terrain, [raw_elite(REF - 5_000, 1, 100, assists=[7, 3, 8, 2])]))   # killer 1; blue assisters 2, 3; red 7, 8
    red = pairs(make_ctx(open_terrain, [raw_elite(REF - 5_000, 6, 200, assists=[2, 8, 3, 7])]))    # the same event seen from the other side
    members = [0, 1, 2, 6, 7]                                                                      # participants 1, 2, 3 (blue), 7, 8 (red)
    assert blue == [(members[a], members[b]) for a in range(5) for b in range(a + 1, 5)]
    assert red == [((i + 5) % 10, (j + 5) % 10) for i, j in blue]
    kill = make_ctx(open_terrain, [raw_kill(REF - 5_000, 1, 6, assists=[8, 3, 2], x=6000, y=6000)])
    v = tokens.build_view(kill, REF, K)
    assert [(i, j) for i, j, k in zip(v['ed_i'], v['ed_j'], v['ed_kind']) if k == 1] == [(1, 5), (2, 5), (7, 5)]     # assisters 2, 3 then 8
    check_against_oracle(v, kill.rec, REF)


def test_an_execution_has_assist_and_damage_edges_but_no_killer_edge(open_terrain):
    ctx = make_ctx(open_terrain, [raw_kill(REF - 5_000, 0, 3, assists=[7, 8], x=4000, y=5000, dealers=[6, 9])])
    v = tokens.build_view(ctx, REF, K)
    got = list(zip(v['ed_i'][:v['ed_n']].tolist(), v['ed_j'][:v['ed_n']].tolist(), v['ed_kind'][:v['ed_n']].tolist()))
    assert got == [(6, 2, 1), (7, 2, 1), (6, 7, 2), (5, 2, 3), (8, 2, 3)]
    check_against_oracle(v, ctx.rec, REF)


def test_damage_entries_without_a_participant_or_on_the_victims_team_are_dropped_and_each_dealer_counts_once(open_terrain):
    junk = [damage(0), damage(11), damage(-1), damage(True), damage('3'), damage(None), damage(3.0), {'name': 'no id'}, 'text', None,
            damage(6), damage(8), damage(10)]                                           # minions and towers (0), bad ids, the victim's team
    ctx = make_ctx(open_terrain, [raw_kill(REF - 5_000, 2, 9, dealers=[2, 2, 1], extra_damage=junk)])
    v = tokens.build_view(ctx, REF, K)
    kind3 = [(i, j) for i, j, k in zip(v['ed_i'], v['ed_j'], v['ed_kind']) if k == 3]
    assert kind3 == [(0, 8), (1, 8)]                                                        # players 1 and 2 (indices 0 and 1) hit player 9
    check_against_oracle(v, ctx.rec, REF)


def test_a_kill_without_a_damage_list_has_no_damage_edges_and_a_bad_list_is_a_format_error(open_terrain):
    e = raw_kill(REF - 5_000, 1, 6)
    del e['victimDamageReceived']
    assert 3 not in tokens.build_view(make_ctx(open_terrain, [e]), REF, K)['ed_kind']
    e['victimDamageReceived'] = None
    assert 3 not in tokens.build_view(make_ctx(open_terrain, [e]), REF, K)['ed_kind']
    e['victimDamageReceived'] = {'participantId': 1}
    with pytest.raises(MatchFormatError, match='victimDamageReceived') as err:
        make_ctx(open_terrain, [e])
    assert 'SYN_0' not in str(err.value)                                                         # the message never names the match key


def test_the_edge_cap_keeps_the_most_recent_128_even_inside_one_event(open_terrain):
    events = [raw_kill(REF - 1000 * (i + 1), 1, 6, assists=[2, 3, 4, 5], x=6000, y=6000) for i in range(20)]       # 1 + 4 + 10 edges each
    ctx = make_ctx(open_terrain, events)
    v = tokens.build_view(ctx, REF, K)
    assert v['ed_n'] == 128 and np.all(np.diff(v['ed_dt']) >= 0)
    ages = v['ed_dt'].tolist()
    assert [ages.count(float(s)) for s in range(1, 10)] == [15] * 8 + [8]                   # 8 whole kills and the first 8 edges of the 9th
    o = check_against_oracle(v, ctx.rec, REF)
    assert o['ed_n'] == 128 and v['ed_dropped'] == 20 * 15 - 128 == o['ed_dropped']        # 300 edges in the window, 172 cut


def test_ed_dropped_counts_the_edges_the_cap_cuts_like_ev_dropped_counts_events(open_terrain):
    """128 edges fit exactly (nothing cut), the 129th is cut; edges older than the window are not counted."""
    kills = [raw_kill(REF - 1000 * (i + 1), 1, 6, assists=[2, 3, 4, 5], x=6000, y=6000) for i in range(8)]    # 8 x (1 + 4 + 10) = 120 edges
    small = raw_kill(REF - 9_500, 1, 6, assists=[2, 3], x=6000, y=6000, dealers=[1, 2])                           # 1 + 2 + 3 + 2 = 8 edges
    one = raw_elite(REF - 10_000, 7, 200, assists=[8])                                                              # 1 edge, the oldest
    old = raw_kill(REF - 125_000, 1, 6, assists=[2, 3, 4, 5])                                                      # outside the window
    fits = tokens.build_view(make_ctx(open_terrain, kills + [small, old]), REF, K)
    over_ctx = make_ctx(open_terrain, kills + [small, one, old])
    over = tokens.build_view(over_ctx, REF, K)
    assert (fits['ed_n'], fits['ed_dropped']) == (128, 0)
    assert (over['ed_n'], over['ed_dropped']) == (128, 1)
    assert over['ed_dropped'].dtype == np.int16 and over['ed_dropped'].shape == ()
    check_against_oracle(over, over_ctx.rec, REF)
    later = tokens.build_view(over_ctx, REF + 3_000, K)                                         # the window moves on: the same edges, older
    assert (later['ed_n'], later['ed_dropped']) == (128, 1)
    gone = tokens.build_view(over_ctx, REF + 119_000, frame_at(over_ctx.rec, REF + 119_000))            # all but the oldest left the window
    assert (gone['ed_n'], gone['ed_dropped']) == (0, 0)


def test_edges_come_from_the_recent_window_only(open_terrain):
    events = [raw_kill(REF - 120_000, 1, 6, x=6000, y=6000), raw_kill(REF - 119_999, 2, 7, x=6000, y=6000),
              raw_kill(REF + 1, 3, 8, x=6000, y=6000)]
    v = tokens.build_view(make_ctx(open_terrain, events), REF, K)
    assert [(i, j) for i, j, k in zip(v['ed_i'], v['ed_j'], v['ed_kind']) if k == 0] == [(1, 6)]


# ---- old counts and the summary ---------------------------------------------------------------------------------------------------
def test_old_counts_by_type_and_team(open_terrain):
    base = REF - 300_000
    events = [raw_kill(base, 1, 6), raw_kill(base + 1, 1, 7), raw_kill(base + 2, 7, 2), raw_kill(base + 3, 0, 3),   # blue 2, red 1, no team
              raw_elite(base + 4, 7, 200, assists=[8]), raw_building(base + 5, 200, 'MID_LANE', killer=1, x=8955, y=8510),  # red, blue
              raw_plate(base + 6, 100, 'MID_LANE'), raw_item(base + 7, 6), raw_item(base + 8, 1), raw_item(base + 9, 0),   # red, blue, none
              raw_level(base + 10, 2, 3), raw_ward_placed(base + 11, 9), raw_ward_kill(base + 12, 5)]
    v = tokens.build_view(make_ctx(open_terrain, events), REF, K)
    old = {n: int(v['summary'][S[n]]) for n in tokens.SUMMARY if n.startswith('old_')}
    expect = {n: 0 for n in old}
    expect.update(old_champion_kill_b=2, old_champion_kill_r=1, old_elite_monster_kill_r=1, old_building_kill_b=1,
                  old_turret_plate_destroyed_r=1, old_item_purchased_b=1, old_item_purchased_r=1, old_level_up_b=1, old_ward_placed_r=1,
                  old_ward_kill_b=1)
    assert old == expect and v['ev_n'] == 0
    check_against_oracle(v, record(events), REF)


def test_the_summary_holds_the_clock_the_age_the_old_counts_and_the_objective_state(ctx):
    v = tokens.build_view(ctx, REF, K)
    obj = state.objective_state(ctx.rec, REF)
    assert v['summary'][S['game_time_s']] == np.float32(REF / 1000.0)
    assert v['summary'][S['frame_age_s']] == np.float32((REF - 600_300) / 1000.0)
    assert np.array_equal(v['summary'][18:], obj, equal_nan=True) and v['summary'][S['t_s']] == v['summary'][S['game_time_s']]
    check_against_oracle(v, ctx.rec, REF)


# ---- the players block ----------------------------------------------------------------------------------------------------------------
def direct_blocks(rec, terr, ref, k):
    """Every piece of the players block, the zone, the geodesics and the landmarks computed straight from the module functions at ``ref``
    (no MatchContext, no index, a fresh item replay): the arrays a view must equal [R12-b]."""
    frame = rec.frame_num[k]
    levels = frame[:, P['level']] if int(rec.frame_ts[k]) == ref else traits.level_at(rec, ref)
    alive, respawn = state.alive_respawn(rec, ref)
    obj = state.objective_state(rec, ref)
    px, py = frame[:, P['x']], frame[:, P['y']]
    out = np.full((10, len(tokens.PLAYER_NUM)), np.nan, dtype=np.float32)
    out[:, :33] = frame
    out[:, P['has_prev']] = 0
    if k:
        prev = rec.frame_num[k - 1][:, :2]
        ok = np.isfinite(prev).all(axis=1)
        out[ok, P['prev_x']], out[ok, P['prev_y']], out[ok, P['has_prev']] = prev[ok, 0], prev[ok, 1], 1
    out[:, P['level_now']] = levels
    out[:, P['alive']], out[:, P['respawn_s']] = alive, respawn
    out[:, P['wall_fraction']] = terr.wall_fraction(px, py)
    out[:, 40:59] = traits.champion_traits(rec, levels)
    out[:, 59:] = traits.ItemIndex(rec).vectors(ref)
    return dict(players_num=out, players_zone=terr.zone(px, py), landmarks=state.landmark_state(rec, ref, obj),
                geo_pp=terr.geo(px[:, None], py[:, None], px[None, :], py[None, :]).astype(np.float16),
                geo_pl=terr.geo(px[:, None], py[:, None], LM_XY[None, :, 0], LM_XY[None, :, 1]).astype(np.float16),
                summary_obj=obj)


def assert_equals_direct(v, rec, terr, ref, k):
    d = direct_blocks(rec, terr, ref, k)
    for key in ('players_num', 'players_zone', 'landmarks', 'geo_pp', 'geo_pl'):
        assert v[key].dtype == d[key].dtype and np.array_equal(v[key], d[key], equal_nan=True), key
    assert np.array_equal(v['summary'][18:], d['summary_obj'], equal_nan=True), 'objective state'
    check_against_oracle(v, rec, ref)


@pytest.mark.parametrize('ref, k', [(120_300, 2), (300_300, 5), (300_301, 5), (REF, K), (REF + 40_000, K), (1_140_300, 19)])
def test_view_equals_direct_computation_on_the_synthetic_match(open_terrain, ref, k):
    """R12-b on frame-clock refs (the frame's own ms) and event-clock refs; each piece is the module function's result."""
    kills = [raw_kill(180_000, 1, 6, assists=[2], x=6000, y=6000),
             raw_kill(300_300, 2, 7, assists=[1, 3], x=6100, y=6050),                               # on the ms of frame 5
             raw_kill(301_000, 8, 3, x=6300, y=6200, dealers=[8, 9]), raw_kill(1_000_000, 4, 9, assists=[5], x=6200, y=6400)]
    ctx = make_ctx(open_terrain, rich_events() + kills)
    v = tokens.build_view(ctx, ref, k)
    assert_equals_direct(v, ctx.rec, open_terrain, ref, k)


@pytest.mark.parametrize('ref, k', [(120_300, 2), (300_300, 5), (300_301, 5), (REF, K), (REF + 40_000, K), (1_140_300, 19)])
def test_view_is_the_view_of_the_synthetic_match_cut_at_the_reference_time(open_terrain, ref, k):
    """No hindsight: events and frames after ``ref`` are not in the view, whatever they are (kills, objectives, levels, items)."""
    times = [t for t in (ref + 1, ref + 3_000, ref + 7_000, ref + 40_000) if t < 1_200_000]          # before the game end of the record
    later = [raw_kill(t, 1 + t % 5, 6 + t % 5, assists=[2], x=6000, y=6000, dealers=[1, 2]) for t in times]
    later += [extra_event(kind, times[1]) for kind in C.EVENT_TYPES]
    whole = make_ctx(open_terrain, rich_events() + later)
    cut = tokens.MatchContext(cut_at(whole.rec, ref), open_terrain)
    assert differing(tokens.build_view(whole, ref, k), tokens.build_view(cut, ref, k)) == []


def test_views_do_not_depend_on_the_order_of_the_calls(open_terrain):
    ctx = make_ctx(open_terrain, rich_events())
    calls = [(REF, K), (300_300, 5), (REF - 5_000, K), (1_140_300, 19), (120_300, 2), (REF, K)]
    first = [tokens.build_view(ctx, *c) for c in calls]
    again = [tokens.build_view(ctx, *c) for c in reversed(calls)][::-1]
    fresh = [tokens.build_view(make_ctx(open_terrain, rich_events()), *c) for c in calls]
    assert all(views_equal(a, b) and views_equal(a, c) for a, b, c in zip(first, again, fresh))


def test_a_view_is_new_arrays_that_do_not_share_memory_with_the_context(ctx):
    a = tokens.build_view(ctx, REF, K)
    for arr in a.values():
        arr[...] = 0
    b = tokens.build_view(ctx, REF, K)
    assert not views_equal(a, b) and b['ev_n'] > 0 and b['players_num'].any()


def test_prev_position_is_the_frame_before_and_nan_at_frame_zero(ctx):
    v = tokens.build_view(ctx, REF, K)
    prev = ctx.rec.frame_num[K - 1][:, :2]
    assert np.array_equal(v['players_num'][:, [P['prev_x'], P['prev_y']]], prev.astype(np.float32))
    assert np.all(v['players_num'][:, P['has_prev']] == 1)
    v0 = tokens.build_view(ctx, 0, 0)
    assert np.all(v0['players_num'][:, P['has_prev']] == 0) and v0['ev_n'] == 0 and v0['summary'][S['frame_age_s']] == 0
    expect = np.zeros((10, 81), dtype=bool)
    expect[:, [P['prev_x'], P['prev_y']]] = True                                              # the only NaN of the first frame
    assert np.array_equal(np.isnan(v0['players_num']), expect)


def test_level_rule_the_frame_row_uses_the_frames_level_and_the_event_view_the_level_clock(open_terrain):
    """R12-a. Player 1 reaches level 4 on exactly the ms of frame 3 (180,300 ms) while the frame still shows level 3: the frame-clock row
    keeps the frame's snapshot, an event-clock view one ms later has the level clock. The LEVEL_UP is a token with age 0 in the row."""
    ctx = make_ctx(open_terrain, [raw_level(180_300, 1, 4), raw_level(100_000, 2, 3)])
    assert ctx.rec.frame_num[3, 0, P['level']] == 3 and int(ctx.rec.frame_ts[3]) == 180_300
    row = tokens.build_view(ctx, 180_300, 3)
    after = tokens.build_view(ctx, 180_301, 3)
    health = P['lvl_healthMax']
    assert list(row['players_num'][:3, P['level_now']]) == [3, 3, 3] and list(row['players_num'][:3, P['level']]) == [3, 3, 3]
    assert list(after['players_num'][:3, P['level_now']]) == [4, 3, 1] and list(after['players_num'][:3, P['level']]) == [3, 3, 3]
    assert row['ev_type'][0] == TYPE['LEVEL_UP'] and row['ev_dt'][0] == 0 and row['ev_actors'][0] == bit(1)
    h = traits.TRAIT_NAMES.index('lvl_healthMax')                                              # the column in a trait row, not in the view
    at3, at4 = traits.champion_traits(ctx.rec, np.full(10, 3))[0, h], traits.champion_traits(ctx.rec, np.full(10, 4))[0, h]
    assert at3 < at4 and row['players_num'][0, health] == at3 and after['players_num'][0, health] == at4
    # the frame row carries the level of the frame for every player, whatever the clock says
    frame_levels = ctx.rec.frame_num[3, :, P['level']]
    assert np.array_equal(row['players_num'][:, 40:59], traits.champion_traits(ctx.rec, frame_levels), equal_nan=True)
    # with no LEVEL_UP on the frame's ms, and level events that say what the frame says, the two clocks agree
    plain = make_ctx(open_terrain, [raw_level(150_000, p + 1, 3) for p in range(10)])
    a, b = tokens.build_view(plain, 180_300, 3), tokens.build_view(plain, 180_301, 3)
    clock_columns = [P['level_now']] + list(range(40, 59))                                          # level_now and the trait columns
    assert np.array_equal(a['players_num'][:, P['level_now']], np.full(10, 3, np.float32))
    assert np.array_equal(a['players_num'][:, clock_columns], b['players_num'][:, clock_columns])


# ---- missing values (R12-d) -----------------------------------------------------------------------------------------------------------
def test_a_missing_position_gives_zone_minus_one_and_nan_distances_and_wall_fraction(open_terrain):
    rec = frames_with(record(rich_events()), {(K, 2, 'x'): NAN, (K, 2, 'y'): NAN, (K - 1, 4, 'x'): NAN, (K - 1, 4, 'y'): NAN})
    v = tokens.build_view(tokens.MatchContext(rec, open_terrain), REF, K)
    assert v['players_zone'][2] == -1 and np.all(np.delete(v['players_zone'], 2) >= 0)
    assert np.isnan(v['players_num'][2, [P['x'], P['y'], P['wall_fraction']]]).all()
    assert np.isnan(v['geo_pp'][2]).all() and np.isnan(v['geo_pp'][:, 2]).all() and np.isnan(v['geo_pl'][2]).all()
    others = np.delete(np.arange(10), 2)
    assert np.isfinite(v['geo_pp'][np.ix_(others, others)]).all() and np.isfinite(v['geo_pl'][others]).all()
    assert np.isfinite(v['players_num'][others][:, [P['wall_fraction']]]).all()
    # the previous frame's position of player 5 is missing: prev is NaN and has_prev is 0, for that player only
    assert np.isnan(v['players_num'][4, [P['prev_x'], P['prev_y']]]).all() and v['players_num'][4, P['has_prev']] == 0
    assert list(v['players_num'][:, P['has_prev']]) == [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    # NaN only in the cells that belong to a missing position
    nan = np.isnan(v['players_num'])
    expect = np.zeros_like(nan)
    expect[2, [P['x'], P['y'], P['wall_fraction']]] = True
    expect[4, [P['prev_x'], P['prev_y']]] = True
    assert np.array_equal(nan, expect)


def test_nan_stays_inside_nan_allowed_on_views_that_hold_every_kind_of_missing_value(open_terrain):
    """A missing position (frame K) and a missing previous position (frame K - 1), an unknown champion, events without a position, the
    first frame, and late refs where the legacy rules say "will not spawn again": no NaN outside the map, each where its condition holds."""
    rec = frames_with(record(rich_events()), {(K, 2, 'x'): NAN, (K, 2, 'y'): NAN, (K - 1, 4, 'x'): NAN, (K - 1, 4, 'y'): NAN})
    champ = np.array(rec.champ)
    champ[7] = 99_999                                                                           # not in any champion table
    ctx = tokens.MatchContext(dataclasses.replace(rec, champ=champ), open_terrain)
    seen = set()
    for ref, k in [(0, 0), (120_300, 2), (300_300, 5), (REF, K), (REF + 40_000, K), (1_140_300, 19)]:
        v = tokens.build_view(ctx, ref, k)
        assert nan_outside_allowed(v) == {}, (ref, nan_outside_allowed(v))
        assert nan_conditions_broken(v) == [], (ref, nan_conditions_broken(v))
        seen |= set(nan_cells(v))
    assert seen == set(tokens.NAN_ALLOWED)                                                   # every entry of the map is used by these views


def test_the_nan_check_of_the_tests_sees_a_nan_where_the_map_does_not_allow_one(ctx):
    v = tokens.build_view(ctx, REF, K)
    assert nan_outside_allowed(v) == {} and nan_conditions_broken(v) == []
    for key, cell in (('players_num', (3, P['alive'])), ('landmarks', (0, 0)), ('landmarks', (30, 0)), ('summary', (S['dragon_up'],)),
                      ('ev_dt', (0,)), ('ed_dt', (0,)), ('ev_gold', (0,))):
        w = {k: a.copy() for k, a in v.items()}
        w[key][cell] = np.nan
        assert nan_outside_allowed(w) == {key: 1}, (key, cell)
    w = {k: a.copy() for k, a in v.items()}
    w['players_num'][3, P['wall_fraction']] = np.nan                                           # allowed column, but the position is there
    assert nan_outside_allowed(w) == {} and nan_conditions_broken(w) == ['wall_fraction']


def test_an_unknown_champion_has_nan_traits_and_the_unknown_flag_and_no_other_nan(open_terrain):
    rec = record(rich_events())
    champ = np.array(rec.champ)
    champ[3] = 99_999                                                                           # not in any champion table
    v = tokens.build_view(tokens.MatchContext(dataclasses.replace(rec, champ=champ), open_terrain), REF, K)
    traits_block = v['players_num'][:, 40:59]
    assert np.isnan(traits_block[3, :18]).all() and traits_block[3, 18] == 1
    assert np.isfinite(np.delete(traits_block, 3, axis=0)).all() and np.all(np.delete(traits_block, 3, axis=0)[:, 18] == 0)
    assert np.isfinite(np.delete(v['players_num'], slice(40, 58), axis=1)).all()


@pytest.mark.parametrize('name', ['currentGold', 'cs_armor', 'level', 'xp'])
def test_nan_in_another_frame_column_stops_the_view_and_names_the_column(open_terrain, name):
    rec = frames_with(record(rich_events()), {(K, 6, name): NAN})
    ctx = tokens.MatchContext(rec, open_terrain)
    with pytest.raises(ValueError, match=rf'frame column.*{name}') as err:                       # named as a column of the frame
        tokens.build_view(ctx, 600_300, K)
    assert 'SYN_0' not in str(err.value)
    tokens.build_view(ctx, int(rec.frame_ts[K - 1]), K - 1)                                 # the frame before has no hole: its view is fine


def test_an_infinite_frame_value_stops_the_view(open_terrain):
    ctx = tokens.MatchContext(frames_with(record(rich_events()), {(K, 6, 'totalGold'): float('inf')}), open_terrain)
    with pytest.raises(ValueError):
        tokens.build_view(ctx, REF, K)


# ---- geodesics in float16 (R12-e) ----------------------------------------------------------------------------------------------------
def test_geodesic_arrays_are_float16_game_units_from_the_terrain(ctx, open_terrain):
    v = tokens.build_view(ctx, REF, K)
    px, py = ctx.rec.frame_num[K][:, 0], ctx.rec.frame_num[K][:, 1]
    pp = open_terrain.geo(px[:, None], py[:, None], px[None, :], py[None, :])
    pl = open_terrain.geo(px[:, None], py[:, None], LM_XY[None, :, 0], LM_XY[None, :, 1])
    assert v['geo_pp'].dtype == np.float16 and v['geo_pl'].dtype == np.float16
    assert np.array_equal(v['geo_pp'], pp.astype(np.float16)) and np.array_equal(v['geo_pl'], pl.astype(np.float16))
    assert np.all(np.diag(v['geo_pp']) == 0) and np.array_equal(v['geo_pp'], v['geo_pp'].T)
    assert np.abs(v['geo_pp'].astype(np.float32) - pp).max() <= 8 and np.abs(v['geo_pl'].astype(np.float32) - pl).max() <= 8
    assert pl.max() < 21_214 and pp.max() < 21_214                                              # the longest snapped path on the grid


def test_float16_holds_every_geodesic_distance_with_a_step_of_at_most_16_units():
    d = np.arange(0, 21_214, dtype=np.float32)
    assert np.abs(d.astype(np.float16).astype(np.float32) - d).max() <= 8                       # half a step of 16 in [16,384, 32,768)
    assert np.isfinite(np.float16(21_213))


@pytest.mark.parametrize('distance', [70_000.0, float('inf')])
def test_a_geodesic_that_float16_cannot_hold_is_refused_by_name(distance):
    ctx = tokens.MatchContext(record(rich_events()), StubTerrain(distance))
    with pytest.raises(ValueError, match='geo_p'):
        tokens.build_view(ctx, REF, K)
    tokens.build_view(tokens.MatchContext(record(rich_events()), StubTerrain(21_213.0)), REF, K)       # the longest real distance fits


class OddWallTerrain(StubTerrain):
    """A terrain whose wall fraction is one number for every position (NaN or infinity here)."""

    def __init__(self, wall):
        super().__init__(100.0)
        self.wall = wall

    def wall_fraction(self, x, y):
        return np.full(np.shape(x), self.wall, dtype=np.float32)


def test_a_nan_wall_fraction_at_a_known_position_stops_the_view_and_names_the_column():
    """NaN is allowed in wall_fraction only where the position is missing [S2-4]; the check also covers derived columns."""
    ctx = tokens.MatchContext(record(rich_events()), OddWallTerrain(float('nan')))
    with pytest.raises(ValueError, match='NaN in wall_fraction'):
        tokens.build_view(ctx, REF, K)
    tokens.build_view(tokens.MatchContext(record(rich_events()), OddWallTerrain(0.5)), REF, K)


def test_an_infinite_value_in_a_derived_column_stops_the_view_and_names_the_column():
    ctx = tokens.MatchContext(record(rich_events()), OddWallTerrain(float('inf')))
    with pytest.raises(ValueError, match='infinite value in wall_fraction'):
        tokens.build_view(ctx, REF, K)


def test_the_context_needs_a_terrain_that_answers_geo_zone_and_wall_fraction():
    with pytest.raises(TypeError):
        tokens.MatchContext(record(), object())


# ---- visibility: tokens, edges, old counts, levels, items, state and landmarks (R12-c) ---------------------------------------------------
def effect_on_ref(kind, base, on):
    """What an event of ``kind`` on the reference ms changes in the other components of the view (the token is checked by the caller)."""
    names = {n: i for i, n in enumerate(state.OBJ_STATE_NAMES)}
    if kind == 'CHAMPION_KILL':
        assert on['players_num'][5, P['alive']] == 0 and base['players_num'][5, P['alive']] == 1
        assert on['players_num'][5, P['respawn_s']] > 0
        assert on['ed_n'] > base['ed_n'] and on['ed_dt'][0] == 0 and on['ed_kind'][0] == 0                  # the newest edge is the kill's
    elif kind == 'ELITE_MONSTER_KILL':
        assert on['summary'][18 + names['blue_dragons']] == base['summary'][18 + names['blue_dragons']] + 1
        assert on['ed_n'] == base['ed_n'] + 3 and on['ed_kind'][0] == 4
    elif kind == 'BUILDING_KILL':
        i = state.landmark_index(200, 'MID_LANE', 'TOWER_BUILDING', 'INNER_TURRET')
        assert base['landmarks'][i, 0] == 1 and on['landmarks'][i, 0] == 0
        assert on['summary'][18 + names['red_towers_inner']] == base['summary'][18 + names['red_towers_inner']] - 1
    elif kind == 'TURRET_PLATE_DESTROYED':
        assert on['summary'][18 + names['red_plates_top']] == base['summary'][18 + names['red_plates_top']] - 1
    elif kind == 'ITEM_PURCHASED':
        assert not np.array_equal(on['players_num'][0, 59:], base['players_num'][0, 59:])
    elif kind == 'LEVEL_UP':
        assert on['players_num'][0, P['level_now']] == 5 and base['players_num'][0, P['level_now']] == 1
        assert on['players_num'][0, P['lvl_healthMax']] > base['players_num'][0, P['lvl_healthMax']]


@pytest.mark.parametrize('kind', C.EVENT_TYPES)
def test_an_event_of_every_type_is_seen_on_the_reference_ms_and_unseen_one_ms_later(open_terrain, kind):
    """R12-c. The inclusive rule for tokens, edges, levels, items, state and landmarks: ``t <= ref`` is seen, ``ref + 1`` is not."""
    events = rich_events()
    base = tokens.build_view(make_ctx(open_terrain, events), REF, K)
    late = tokens.build_view(make_ctx(open_terrain, events + [extra_event(kind, REF + 1)]), REF, K)
    on = tokens.build_view(make_ctx(open_terrain, events + [extra_event(kind, REF)]), REF, K)
    assert differing(base, late) == []
    assert on['ev_n'] == base['ev_n'] + 1 and on['ev_dt'][0] == 0 and on['ev_type'][0] == TYPE[kind]
    effect_on_ref(kind, base, on)
    # and the same event is seen by a view one ms later
    after = tokens.build_view(make_ctx(open_terrain, events + [extra_event(kind, REF + 1)]), REF + 1, K)
    assert after['ev_n'] == base['ev_n'] + 1 and after['ev_dt'][0] == 0


def test_an_old_event_enters_the_old_counts_exactly_when_it_leaves_the_recent_window(open_terrain):
    events = [raw_kill(REF - 120_000, 1, 6, x=6000, y=6000)]
    ctx = make_ctx(open_terrain, events)
    here, an_ms_earlier = tokens.build_view(ctx, REF, K), tokens.build_view(ctx, REF - 1, K)
    assert (here['ev_n'], old_total(here)) == (0, 1) and (an_ms_earlier['ev_n'], old_total(an_ms_earlier)) == (1, 0)
    assert an_ms_earlier['ed_n'] > 0 and here['ed_n'] == 0


def test_a_frame_after_the_reference_time_is_refused_and_the_arguments_are_checked(ctx):
    with pytest.raises(ValueError, match='after the reference'):
        tokens.build_view(ctx, 300_299, 5)                                                       # frame 5 is at 300,300 ms
    tokens.build_view(ctx, 300_300, 5)
    for k in (-1, 20, 1000):
        with pytest.raises(ValueError, match='frame_k'):
            tokens.build_view(ctx, REF, k)
    for bad in (REF + 0.5, float(REF), True, '617000', None):
        with pytest.raises(TypeError):
            tokens.build_view(ctx, bad, K)
    for bad in (1.0, True, '3', None):
        with pytest.raises(TypeError):
            tokens.build_view(ctx, REF, bad)
    tokens.build_view(ctx, np.int64(REF), np.int16(K))                                          # numpy integers are fine


def test_a_stale_frame_k_is_refused_and_the_valid_ends_and_the_same_ms_case_are_not(ctx):
    """A view takes the last regular frame before its reference time: a frame strictly between frame_k and ref makes frame_k stale."""
    ts = [int(t) for t in ctx.rec.frame_ts]
    for ref, k in ((REF + 59_000, K), (REF, 0), (ts[11] + 1, 10)):
        with pytest.raises(ValueError, match='stale') as err:
            tokens.build_view(ctx, ref, k)
        assert 'SYN_0' not in str(err.value)
    tokens.build_view(ctx, ts[K + 1], K)                                                    # the next frame is at the reference time: valid
    tokens.build_view(ctx, ts[K + 1] - 1, K)
    last = len(ts) - 1
    tokens.build_view(ctx, ts[last] + 50_000, last)                                             # the last frame of the match, any later ref
    row = tokens.build_view(ctx, ts[5], 5)                                                       # the frame-clock row of frame 5 ...
    event = tokens.build_view(ctx, ts[5], 4)                                              # ... and an event row on its ms (S1: frame k - 1)
    assert row['summary'][S['frame_age_s']] == 0 and event['summary'][S['frame_age_s']] == pytest.approx(60.0)
    assert differing(row, event) != []


def test_the_level_of_a_frame_row_and_of_an_event_view_goes_through_narrow(open_terrain, monkeypatch):
    """S2-4, R12-h: the level enters the float32 buffer through the checked cast, whether it is the frame's float or the clock's int."""
    names = []
    real = tokens.narrow

    def spy(name, a, dtype):
        names.append(name)
        return real(name, a, dtype)

    ctx = make_ctx(open_terrain, rich_events())
    monkeypatch.setattr(tokens, 'narrow', spy)
    for ref, k in ((600_300, K), (REF, K)):                                                      # a frame-clock row, an event-clock view
        names.clear()
        tokens.build_view(ctx, ref, k)
        assert 'level_now' in names, (ref, names)


def test_landmarks_are_the_landmark_state_of_the_module_at_ref(ctx):
    v = tokens.build_view(ctx, REF, K)
    obj = state.objective_state(ctx.rec, REF)
    assert np.array_equal(v['landmarks'], state.landmark_state(ctx.rec, REF, obj), equal_nan=True)
    i = state.landmark_index(200, 'MID_LANE', 'TOWER_BUILDING', 'OUTER_TURRET')
    assert list(v['landmarks'][i]) == [0, 0]                                                      # killed 30 s before REF in rich_events


# ---- structure: no hindsight [S2-11, R12-g], casts [R12-h] -----------------------------------------------------------------------------
# ``module_level_imports(path)`` and ``all_imports(path)`` (conftest.py) resolve a relative import against the package of the file; a file
# outside the repository needs its package given, as the self-tests below do.
def test_the_import_scans_see_what_they_claim_to_see(tmp_path):
    src = tmp_path / 'm.py'
    src.write_text('import numpy\n\n\ndef f():\n    from c_ept import run_s1\n    import c_ept.targets\n\n\nif True:\n'
                   '    from c_ept.store_core import Store\n', encoding='utf-8')
    assert {'numpy', 'c_ept.run_s1', 'c_ept.targets', 'c_ept.store_core', 'c_ept.store_core.Store'} <= all_imports(src, 'c_ept')
    assert module_level_imports(src, 'c_ept') == {'numpy', 'c_ept.store_core', 'c_ept.store_core.Store'}   # the two inside f are missed


def banned(names):
    """The names among ``names`` that are an S1 module or lie below one (what the real scan tests refuse)."""
    return sorted(m for m in names for s in S1_MODULES if m == s or m.startswith(s + '.'))


def test_the_import_scans_resolve_a_relative_import_against_the_package(tmp_path):
    """``from . import run_s1`` in ``c_ept/tokens.py`` is ``c_ept.run_s1``. Read as an absolute name it was ``None.run_s1`` (and
    ``from .targets import x`` was ``targets``), so a view module could have imported an S1 module unseen (final review, Task 12)."""
    src = tmp_path / 'm.py'
    src.write_text('import numpy\nfrom . import run_s1\nfrom .targets import Targets\n\n\ndef f():\n    from .store_core import Store\n'
                   '    from . import event_rows as rows\n', encoding='utf-8')
    top = all_imports(src, 'c_ept')                                                         # as c_ept/tokens.py
    assert {'numpy', 'c_ept', 'c_ept.run_s1', 'c_ept.targets', 'c_ept.targets.Targets', 'c_ept.store_core', 'c_ept.store_core.Store',
            'c_ept.event_rows'} == top
    assert banned(top) == ['c_ept.event_rows', 'c_ept.run_s1', 'c_ept.store_core', 'c_ept.store_core.Store', 'c_ept.targets',
                           'c_ept.targets.Targets']
    assert module_level_imports(src, 'c_ept') == {'numpy', 'c_ept', 'c_ept.run_s1', 'c_ept.targets', 'c_ept.targets.Targets'}
    assert banned(module_level_imports(src, 'c_ept')) == ['c_ept.run_s1', 'c_ept.targets', 'c_ept.targets.Targets']       # f() is missed
    assert 'None.run_s1' not in top and 'targets' not in top                                # the absolute reading is gone
    sub = tmp_path / 'checks_m.py'
    sub.write_text('from .. import run_s1\nfrom ..targets import Targets\nfrom . import common\nfrom .common import helper\n'
                   'from c_ept import config\nimport c_ept.units\n', encoding='utf-8')
    assert all_imports(sub, 'c_ept.checks') == {                                           # as c_ept/checks/x.py: ".." is c_ept
        'c_ept', 'c_ept.run_s1', 'c_ept.targets', 'c_ept.targets.Targets', 'c_ept.checks', 'c_ept.checks.common',
        'c_ept.checks.common.helper', 'c_ept.config', 'c_ept.units'}
    assert banned(all_imports(sub, 'c_ept.checks')) == ['c_ept.run_s1', 'c_ept.targets', 'c_ept.targets.Targets']
    too_high = tmp_path / 'high.py'
    too_high.write_text('from ... import x\n', encoding='utf-8')
    with pytest.raises(ValueError, match='climbs above'):
        all_imports(too_high, 'c_ept.checks')                                              # level 3 in a package of two names
    with pytest.raises(ValueError, match='climbs above'):
        module_level_imports(src, '')                                                      # a relative import in a top-level script


def test_the_package_of_a_file_is_the_folder_it_stands_in():
    assert package_of(REPO / 'c_ept' / 'tokens.py') == 'c_ept' == package_of(REPO / 'c_ept' / '__init__.py')
    assert package_of(REPO / 'c_ept' / 'checks' / 'common.py') == 'c_ept.checks'
    with pytest.raises(ValueError, match='not in the repository'):
        package_of(REPO.parent / 'no_such_sibling' / 'm.py')                               # outside: the caller gives the package then


@pytest.mark.parametrize('name, scan', [('tokens', all_imports), ('symmetry', all_imports), ('state', module_level_imports),
                                        ('traits', module_level_imports)])
def test_the_view_modules_import_no_s1_module(name, scan):
    """R12-g: ``tokens`` and ``symmetry`` name no S1 module anywhere, also inside a function; ``state`` and ``traits`` at module level.
    ``import c_ept.run_s1``, ``from c_ept import run_s1`` and ``from c_ept.run_s1 import x`` all show as ``c_ept.run_s1``."""
    from importlib import import_module
    imported = scan(Path(import_module(f'c_ept.{name}').__file__))
    assert 'numpy' in imported                                                                  # the scan does see the imports
    assert not banned(imported), banned(imported)


@pytest.mark.parametrize('statement', ['import c_ept.tokens, c_ept.symmetry',
                                       'import c_ept.preprocessing.tokens, c_ept.domain.symmetry'])
def test_a_fresh_interpreter_that_imports_the_view_modules_loads_no_s1_module(statement):
    code = (PROTECTED_IMPORT_PRELUDE + f'import sys\n{statement}\n'
            f'print([m for m in {S1_MODULES!r} if m in sys.modules])')
    out = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert out.returncode == 0, out.stderr[-300:]
    assert out.stdout.strip() == '[]'


def test_the_context_and_the_view_builder_take_no_s1_store():
    assert list(inspect.signature(tokens.MatchContext).parameters) == ['rec', 'terr']
    assert list(inspect.signature(tokens.build_view).parameters) == ['ctx', 'ref_ms', 'frame_k']


NARROW_DTYPES = {'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32', 'float16', 'float32'}


@pytest.mark.parametrize('name', ['tokens', 'symmetry'])
def test_no_cast_to_a_narrower_dtype_bypasses_the_checked_narrow(name):
    """R12-h: ``.astype(<narrower dtype>)`` is not used in the view modules (a widening such as astype(np.int64) is fine)."""
    from importlib import import_module
    tree = ast.parse(Path(import_module(f'c_ept.{name}').__file__).read_text(encoding='utf-8'))
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'astype'
           and any(isinstance(a, ast.Attribute) and a.attr in NARROW_DTYPES for a in n.args)]
    assert bad == []


def test_the_citation_of_the_masking_and_elapsed_time_inputs_is_in_the_module():
    text = Path(tokens.__file__).read_text(encoding='utf-8')
    assert 'Che et al. 2018' in text and 's41598-018-24271-9' in text and '8:6085' in text


# ---- golden matches ---------------------------------------------------------------------------------------------------------------
def golden_refs(rec):
    """Frame-clock refs (several frames) and event-clock refs (after a frame, and at kills) of one record: [(ref, frame_k), ...]."""
    ts = rec.frame_ts
    ks = sorted({2, 3, len(ts) // 2, len(ts) - 2, len(ts) - 1} & set(range(len(ts))))
    out = [(int(ts[k]), k) for k in ks] + [(int(ts[k]) + 13_579, k) for k in ks]
    for e in [e for e in rec.events if e['type'] == 'CHAMPION_KILL' and e['timestamp'] >= 120_000][::7]:
        k = int(np.searchsorted(ts, e['timestamp'], side='left')) - 1                          # the frame strictly before the kill
        out.append((e['timestamp'], k))
    return sorted(set(out))


def test_golden_views_equal_the_direct_computation(golden_records, open_terrain):
    """R12-b on real matches: the frame-clock rows, event-clock views and the views on a kill's ms equal the module functions' results."""
    count = 0
    for m, rec in enumerate(golden_records):
        ctx = tokens.MatchContext(rec, open_terrain)
        for ref, k in golden_refs(rec):
            try:
                assert_equals_direct(tokens.build_view(ctx, ref, k), rec, open_terrain, ref, k)
            except AssertionError as err:
                raise AssertionError(f'match {m} at ref {ref} (frame {k}): {str(err)[:200]}') from None
            count += 1
    assert count >= 200


def test_golden_view_is_the_view_of_the_match_cut_at_the_reference_time(golden_records, open_terrain):
    """No hindsight [S2-11, R12-g] in its strongest form on real matches: what happens after ``ref`` cannot be in the view, because the
    view of the whole match equals the view of a match that ends at ``ref``."""
    count = 0
    for m, rec in enumerate(golden_records[:10]):
        ctx = tokens.MatchContext(rec, open_terrain)
        for ref, k in golden_refs(rec)[::2]:
            a, b = tokens.build_view(ctx, ref, k), tokens.build_view(tokens.MatchContext(cut_at(rec, ref), open_terrain), ref, k)
            assert differing(a, b) == [], (m, ref, k)
            count += 1
    assert count >= 60


def test_golden_nan_rules_on_every_regular_frame(golden_records, open_terrain):
    """R12-d and R12-k: no NaN in players_num, in the summary except the three LEGACY columns, in the landmarks except the seconds of
    the herald, grubs and atakhan rows, in the edges or the geodesics; all positions are present in S1."""
    frames, nan_summary, nan_landmark = 0, set(), set()
    allowed_cols = {S[n] for n in tokens.NAN_ALLOWED['summary']['cells']}
    for m, rec in enumerate(golden_records):
        ctx = tokens.MatchContext(rec, open_terrain)
        for k in range(len(rec.frame_ts)):
            v = tokens.build_view(ctx, int(rec.frame_ts[k]), k)
            cells = nan_cells(v)
            if k == 0:                                                                  # no frame before: prev_x, prev_y of the ten players
                assert cells.pop('players_num') == 20 and np.isnan(v['players_num'][:, [P['prev_x'], P['prev_y']]]).all()
            assert set(cells) <= {'summary', 'landmarks', 'ev_xy'}, (m, k, cells)
            nan_summary |= {int(i) for i in np.flatnonzero(np.isnan(v['summary']))}
            nan_landmark |= {(int(i), int(j)) for i, j in zip(*np.nonzero(np.isnan(v['landmarks'])))}
            assert np.all(v['players_zone'] >= 0)
            frames += 1
    assert frames == 582
    assert nan_summary <= allowed_cols
    assert nan_landmark <= set(tokens.NAN_ALLOWED['landmarks']['cells'])
    assert nan_summary and nan_landmark                                                            # the NaN of the legacy rules do occur


def test_golden_tokens_and_edges_equal_the_oracle_at_every_regular_frame(golden_records, open_terrain):
    windows, dropped, edges_dropped, busiest = 0, 0, 0, 0
    for m, rec in enumerate(golden_records):
        ctx = tokens.MatchContext(rec, open_terrain)
        for k in range(2, len(rec.frame_ts)):
            ref = int(rec.frame_ts[k])
            v = tokens.build_view(ctx, ref, k)
            try:
                check_against_oracle(v, rec, ref)
            except AssertionError as err:
                raise AssertionError(f'match {m} frame {k}: {str(err)[:200]}') from None
            windows += 1
            dropped += v['ev_dropped'] > 0
            edges_dropped += v['ed_dropped'] > 0
            busiest = max(busiest, int(v['ev_n']) + int(v['ev_dropped']))
    # the frames 2 ... of the 20 matches (re-measured 2026-10-21 at the event cap 128, S2 gate S2G-1): the busiest of the 542 windows holds
    # 125 events, so none holds more than the cap (at the cap 80 it was 33 windows, 6.1%); 3 (0.55%) hold more than 128 edges
    assert (windows, dropped, edges_dropped, busiest) == (542, 0, 3, 125)


def test_golden_team_column_agrees_with_the_objective_table(golden_records, open_terrain):
    n = 0
    for rec in golden_records:
        ctx = tokens.MatchContext(rec, open_terrain)
        rows = [e for e in rec.events if e['type'] in C.EVENT_TYPES]
        for j, e in enumerate(rows):
            if e['type'] in targets.OBJECTIVE_TYPES:
                want = targets._taker(e)
                assert int(ctx.events.team[j]) == (want if want >= 0 else 2)
                n += 1
    assert n == 187 + 249 + 220


def test_golden_item_purchases_of_participant_zero_have_no_team(golden_records, open_terrain):
    zero = 0
    for rec in golden_records:
        ctx = tokens.MatchContext(rec, open_terrain)
        rows = [e for e in rec.events if e['type'] in C.EVENT_TYPES]
        for j, e in enumerate(rows):
            if e['type'] == 'ITEM_PURCHASED' and e['participantId'] == 0:
                assert ctx.events.team[j] == 2 and ctx.events.actors[j] == 0
                zero += 1
    assert zero == 40


def test_golden_nan_only_inside_nan_allowed_for_frame_rows_and_for_event_refs(golden_records, open_terrain):
    """NAN_ALLOWED is exhaustive: on the 20 golden matches no NaN stands outside the map and each NaN stands where its condition says,
    in every frame-clock row (582) and in the event-clock views (after a frame, and at kills). game_time_s and t_s are always equal."""
    rows = events = 0
    for m, rec in enumerate(golden_records):
        ctx = tokens.MatchContext(rec, open_terrain)
        frame_refs = [(int(rec.frame_ts[k]), k) for k in range(len(rec.frame_ts))]
        event_refs = [(ref, k) for ref, k in golden_refs(rec) if ref != int(rec.frame_ts[k])]
        for kind, refs in (('frame', frame_refs), ('event', event_refs)):
            for ref, k in refs:
                v = tokens.build_view(ctx, ref, k)
                where = (m, kind, ref, k)
                assert nan_outside_allowed(v) == {}, (where, nan_outside_allowed(v))
                assert nan_conditions_broken(v) == [], (where, nan_conditions_broken(v))
                assert v['summary'][S['game_time_s']] == v['summary'][S['t_s']], where
                rows += kind == 'frame'
                events += kind == 'event'
    assert rows == 582 and events >= 200
