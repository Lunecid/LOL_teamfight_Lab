"""Helpers shared by the tests of the view builder (``test_tokens.py`` and ``test_symmetry.py``) and by the tests that build views.

Raw events with the field names of the Match-V5 files, records and contexts of synthetic matches, the comparison of two views, the check
of ``tokens.NAN_ALLOWED`` and a terrain stub. There is no test here (the file name does not start with ``test_``), and nothing in it reads
raw data, a run folder or the catalog. Moved out of ``test_tokens.py`` in the fix round of Task 12 (it was imported from there).
"""
import dataclasses

import numpy as np

from c_ept import config as C
from c_ept.domain import state
from c_ept.preprocessing import tokens, traits
from c_ept.tests import synth

K = 10                          # frame index of the event-clock tests: frame 10 of the synthetic grid is at 600,300 ms
REF = 617_000                   # 16.7 s after it
P = {name: i for i, name in enumerate(tokens.PLAYER_NUM)}
S = {name: i for i, name in enumerate(tokens.SUMMARY)}
LM_XY = np.array([[lm['x'], lm['y']] for lm in state.LANDMARKS])          # new numeric array; LANDMARKS itself is never touched (R12-m)
TYPE = {name: i for i, name in enumerate(C.EVENT_TYPES)}
NAN = float('nan')


# ---- raw events -------------------------------------------------------------------------------------------------------------------
def damage(pid, spell='spell'):
    """One entry of a victimDamageReceived list (the fields of a real one; no name of a real player)."""
    return {'participantId': pid, 'name': 'Name', 'spellName': spell, 'spellSlot': 0, 'basic': False, 'type': 'OTHER',
            'magicDamage': 10, 'physicalDamage': 0, 'trueDamage': 0}


def raw_kill(t, killer, victim, assists=(), x=6000, y=6000, bounty=300, shutdown=0, dealers=(), extra_damage=()):
    """A CHAMPION_KILL. ``dealers``: one participant id per damage source, two entries each (a real list has one per spell);
    ``extra_damage``: entries put in as they are. ``x=None``: no position."""
    e = {'type': 'CHAMPION_KILL', 'timestamp': t, 'killerId': killer, 'victimId': victim, 'bounty': bounty, 'shutdownBounty': shutdown,
         'killStreakLength': 0, 'victimDamageDealt': [],
         'victimDamageReceived': [damage(d, s) for d in dealers for s in ('a', 'b')] + list(extra_damage)}
    if x is not None:
        e['position'] = {'x': x, 'y': y}
    if assists:
        e['assistingParticipantIds'] = list(assists)
    return e


def raw_elite(t, killer, team, assists=(), monster='DRAGON', sub='FIRE_DRAGON', x=9837, y=4401, bounty=0):
    e = {'type': 'ELITE_MONSTER_KILL', 'timestamp': t, 'killerId': killer, 'killerTeamId': team, 'monsterType': monster, 'bounty': bounty}
    if sub:
        e['monsterSubType'] = sub
    if x is not None:
        e['position'] = {'x': x, 'y': y}
    if assists:
        e['assistingParticipantIds'] = list(assists)
    return e


def raw_building(t, owner, lane, building='TOWER_BUILDING', tower='OUTER_TURRET', killer=0, assists=(), x=None, y=None, bounty=0):
    """A BUILDING_KILL of a building of ``owner`` (the team that loses it)."""
    e = {'type': 'BUILDING_KILL', 'timestamp': t, 'killerId': killer, 'teamId': owner, 'buildingType': building, 'laneType': lane,
         'bounty': bounty}
    if tower:
        e['towerType'] = tower
    if x is not None:
        e['position'] = {'x': x, 'y': y}
    if assists:
        e['assistingParticipantIds'] = list(assists)
    return e


def raw_plate(t, owner, lane, killer=0, x=5000, y=5000):
    e = {'type': 'TURRET_PLATE_DESTROYED', 'timestamp': t, 'killerId': killer, 'teamId': owner, 'laneType': lane}
    if x is not None:
        e['position'] = {'x': x, 'y': y}
    return e


def raw_item(t, pid, item=1055):
    return {'type': 'ITEM_PURCHASED', 'timestamp': t, 'participantId': pid, 'itemId': item}


def raw_level(t, pid, level):
    return {'type': 'LEVEL_UP', 'timestamp': t, 'participantId': pid, 'level': level}


def raw_ward_placed(t, creator):
    return {'type': 'WARD_PLACED', 'timestamp': t, 'creatorId': creator, 'wardType': 'YELLOW_TRINKET'}


def raw_ward_kill(t, killer):
    return {'type': 'WARD_KILL', 'timestamp': t, 'killerId': killer, 'wardType': 'CONTROL_WARD'}


def rich_events(ref=REF):
    """Events of every kind around ``ref``: some old (2-10 minutes before it), most in the last 120 s, none at or after ``ref``."""
    return [
        raw_ward_placed(ref - 400_000, 1), raw_item(ref - 300_000, 6, 1055), raw_kill(ref - 250_000, 7, 2, assists=[8], x=7000, y=7000),
        raw_level(ref - 200_000, 3, 5), raw_elite(ref - 150_000, 1, 100, assists=[2], monster='HORDE', sub=None, x=4791, y=10182),
        raw_kill(ref - 60_000, 1, 6, assists=[2, 3], x=6000, y=6000, shutdown=150, dealers=[1, 2, 4, 7]),
        raw_elite(ref - 50_000, 7, 200, assists=[8, 9]),
        raw_kill(ref - 40_000, 0, 3, x=4000, y=5000),                                    # an execution: the killer is no champion
        raw_building(ref - 30_000, 200, 'MID_LANE', killer=2, assists=[3], x=8955, y=8510),
        raw_plate(ref - 25_000, 100, 'TOP_LANE', killer=0, x=981, y=10441),
        raw_item(ref - 20_000, 8, 3006), raw_level(ref - 15_000, 5, 6), raw_ward_placed(ref - 10_000, 10), raw_ward_kill(ref - 5_000, 4),
        raw_kill(ref - 2_000, 4, 9, x=None),                                             # a kill without a position
    ]


def extra_event(kind, t):
    """One event of the given type at ``t`` that changes something in the view if it is seen."""
    return {'CHAMPION_KILL': lambda: raw_kill(t, 1, 6, assists=[2], x=6200, y=6100, dealers=[1, 2, 3]),
            'ELITE_MONSTER_KILL': lambda: raw_elite(t, 1, 100, assists=[2, 3]),
            'BUILDING_KILL': lambda: raw_building(t, 200, 'MID_LANE', 'TOWER_BUILDING', 'INNER_TURRET', killer=1, assists=[2],
                                                  x=9767, y=10113),
            'TURRET_PLATE_DESTROYED': lambda: raw_plate(t, 200, 'TOP_LANE', killer=1, x=4318, y=13875),
            'ITEM_PURCHASED': lambda: raw_item(t, 1, 1055),
            'LEVEL_UP': lambda: raw_level(t, 1, 5),
            'WARD_PLACED': lambda: raw_ward_placed(t, 1),
            'WARD_KILL': lambda: raw_ward_kill(t, 6)}[kind]()


# ---- records, contexts, views -------------------------------------------------------------------------------------------------------
def record(events=(), **kw):
    return synth.make_record(T_ms=1_200_000, events=list(events), **kw)


def make_ctx(terr, events=(), **kw):
    return tokens.MatchContext(record(events, **kw), terr)


def with_frames(rec, frame_num):
    """``rec`` with other frame values (the constructor freezes what it is given: pass a copy)."""
    return dataclasses.replace(rec, frame_num=np.array(frame_num))


def frames_with(rec, edits):
    """A copy of the record's frames with ``edits`` ({(frame, player, column name): value}) applied."""
    fn = np.array(rec.frame_num)
    for (k, p, name), value in edits.items():
        fn[k, p, P[name]] = value
    return with_frames(rec, fn)


def cut_at(rec, ref):
    """The record as it stood at ``ref``: no event after it (and so no GAME_END). The frames are kept; a view reads up to ``frame_k``."""
    return dataclasses.replace(rec, events=tuple(e for e in rec.events if e['timestamp'] <= ref))


def frame_at(rec, ref):
    """The ``frame_k`` that a view at ``ref`` takes: the last regular frame at or before it (the frame itself when ``ref`` is its ms)."""
    return int(np.searchsorted(rec.frame_ts, ref, side='right')) - 1


def differing(a, b):
    """The keys in which two views differ (dtype, shape or values; NaN equals NaN); the keys themselves must be the same."""
    assert a.keys() == b.keys()
    out = []
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == 'f'):
            out.append(key)
    return out


def views_equal(a, b):
    return not differing(a, b)


def old_total(v):
    """The events counted in the 'old' columns of a view's summary."""
    return int(sum(v['summary'][S[name]] for name in tokens.SUMMARY if name.startswith('old_')))


def nan_cells(v):
    """{key: number of NaN cells} of the float arrays of a view that hold any."""
    return {k: int(np.isnan(a).sum()) for k, a in v.items() if a.dtype.kind == 'f' and np.isnan(a).any()}


# ---- tokens.NAN_ALLOWED -------------------------------------------------------------------------------------------------------------
def nan_outside_allowed(v):
    """{key: number of NaN cells that ``tokens.NAN_ALLOWED`` does not allow} for a view; ``{}`` when the view keeps to the map.

    Reads the ``cells`` of each entry: column names for ``players_num`` and ``summary``, (row, column) cells for ``landmarks``, ``'all'``
    for an array where any cell may be NaN under the entry's ``when``. A float array that has no entry may hold no NaN at all.
    """
    out = {}
    for key, a in v.items():
        if a.dtype.kind != 'f':
            continue
        nan = np.isnan(a)
        if not nan.any():
            continue
        entry = tokens.NAN_ALLOWED.get(key)
        if entry is None:
            out[key] = int(nan.sum())
            continue
        cells = entry['cells']
        if cells == 'all':
            continue
        ok = np.zeros(a.shape, dtype=bool)
        if key == 'players_num':
            ok[:, [P[name] for name in cells]] = True
        elif key == 'summary':
            ok[[S[name] for name in cells]] = True
        elif key == 'landmarks':
            for row, col in cells:
                ok[row, col] = True
        else:
            raise AssertionError(f'NAN_ALLOWED[{key!r}] has cells {cells!r} that this helper does not read')
        stray = int((nan & ~ok).sum())
        if stray:
            out[key] = stray
    return out


def nan_conditions_broken(v):
    """The conditions of ``tokens.NAN_ALLOWED[...]['when']`` that a view breaks, as a list of names (``[]``: all hold). It reads only the
    view: NaN exactly where the entry says (a NaN in an allowed column elsewhere is also a break)."""
    p = v['players_num']
    missing = np.isnan(p[:, P['x']]) | np.isnan(p[:, P['y']])                       # the player's position is missing in the frame
    no_prev = p[:, P['has_prev']] == 0
    trait_nan = np.isnan(p[:, [P[n] for n in traits.TRAIT_NAMES if n != 'champ_unknown']])
    unknown = p[:, P['champ_unknown']] == 1
    broken = []
    if not (np.array_equal(np.isnan(p[:, P['prev_x']]), no_prev) and np.array_equal(np.isnan(p[:, P['prev_y']]), no_prev)):
        broken.append('prev_x, prev_y')
    if not np.array_equal(np.isnan(p[:, P['wall_fraction']]), missing):
        broken.append('wall_fraction')
    if not (np.array_equal(trait_nan.all(axis=1), unknown) and np.array_equal(trait_nan.any(axis=1), unknown)):
        broken.append('traits')
    if not np.array_equal(v['players_zone'] == -1, missing):
        broken.append('players_zone')
    if not np.array_equal(np.isnan(v['geo_pp']), missing[:, None] | missing[None, :]):
        broken.append('geo_pp')
    if not np.array_equal(np.isnan(v['geo_pl']), np.repeat(missing[:, None], v['geo_pl'].shape[1], axis=1)):
        broken.append('geo_pl')
    pad = v['ev_type'] == -1
    if not np.isnan(v['ev_xy'][pad]).all():
        broken.append('ev_xy padding')
    return broken


# ---- a terrain stub -----------------------------------------------------------------------------------------------------------------
class StubTerrain:
    """A terrain object that answers every geodesic with one number: the view must range-check the float16 cast [S2-4]."""

    def __init__(self, distance):
        self.distance = distance

    def geo(self, xa, ya, xb, yb):
        return np.full(np.broadcast_shapes(*(np.shape(a) for a in (xa, ya, xb, yb))), self.distance, dtype=np.float32)

    def zone(self, x, y):
        return np.zeros(np.shape(x), dtype=np.int8)

    def wall_fraction(self, x, y):
        return np.zeros(np.shape(x), dtype=np.float32)
