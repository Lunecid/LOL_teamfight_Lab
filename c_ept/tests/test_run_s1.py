"""Tests of the S1 runner (c_ept.run_s1) and of runlog.amend_manifest on synthetic matches.

No raw file is read: every match comes from ``synth`` (random ones by key, a few hand-made ones with worked-out values). Every store and
run folder lives under pytest's tmp_path, and the run catalog is never touched. The ``records_*`` functions at module level are the
``records`` argument of the runs: worker processes import them by name, so they must be picklable.
"""
import dataclasses
import inspect
import json
import re

import numpy as np
import pytest

from c_ept import cohort
from c_ept import config as C
from c_ept.data import event_rows, frame_rows, reader
from c_ept.workflows import run_s1
from c_ept import runlog
from c_ept.data import store_core
from c_ept.domain import units
from c_ept.tests import synth

# The store of the 8a rulings (name -> (dtype, shape after the row axis)), written out here and not read from run_s1.
LAYOUT = {
    'matches/key': ('S24', ()), 'matches/patch': ('i1', ()), 'matches/role': ('i1', ()), 'matches/T_ms': ('i8', ()),
    'matches/team': ('i2', (10,)), 'matches/champ': ('i4', (10,)), 'matches/end_xy': ('f4', (10, 2)),
    'matches/kill_off': ('i8', ()), 'matches/obj_off': ('i8', ()), 'matches/frame_off': ('i8', ()), 'matches/unit_off': ('i8', ()),
    'matches/row_off': ('i8', ()), 'matches/onset_off': ('i8', ()), 'matches/event_off': ('i8', ()),
    'frames/ts': ('i8', ()), 'frames/num': ('f4', (10, 33)),
    'kills/t': ('i8', ()), 'kills/x': ('f8', ()), 'kills/y': ('f8', ()), 'kills/killer': ('i1', ()), 'kills/victim': ('i1', ()),
    'kills/amask': ('i2', ()), 'kills/bounty': ('i4', ()), 'kills/shutdown': ('i4', ()), 'kills/pos_missing': ('?', ()),
    'kills/unit': ('i4', ()),
    'units/match': ('i4', ()), 'units/kill0': ('i4', ()), 'units/n_kills': ('i2', ()), 'units/f': ('i8', ()), 'units/last': ('i8', ()),
    'units/tau': ('i8', ()), 'units/delta': ('i8', ()), 'units/nb': ('i1', ()), 'units/nr': ('i1', ()), 'units/size': ('i1', ()),
    'units/mark': ('i2', ()), 'units/in_scope': ('?', ()), 'units/spread': ('f4', ()), 'units/K': ('i4', (8,)),
    'units/K_at_f': ('i4', (8,)), 'units/exec': ('i2', (2,)), 'units/O': ('i2', (18,)), 'units/O_at_f': ('i2', (18,)),
    'units/conv': ('i2', (18,)), 'units/conv_exposure_ms': ('i4', ()),
    'objectives/t': ('i8', ()), 'objectives/kind': ('i1', ()), 'objectives/taker': ('i1', ()), 'objectives/x': ('f4', ()),
    'objectives/y': ('f4', ()), 'objectives/has_pos': ('?', ()), 'objectives/cred': ('i2', ()), 'objectives/unit': ('i4', ()),
    'objectives/conv_unit': ('i4', ()), 'objectives/sub': ('i1', ()),
    'frame_rows/match': ('i4', ()), 'frame_rows/k': ('i2', ()), 'frame_rows/t0': ('i8', ()), 'frame_rows/hi': ('i8', ()),
    'frame_rows/exposure_ms': ('i4', (18,)),
    'onsets/row': ('i4', ()), 'onsets/bin': ('i1', ()), 'onsets/mark': ('i2', ()), 'onsets/unit': ('i4', ()),
    'event_rows/unit': ('i4', ()), 'event_rows/kind': ('i1', ()), 'event_rows/ref': ('i8', ()), 'event_rows/frame_k': ('i2', ()),
    'event_rows/t0': ('i8', ()), 'event_rows/age_ms': ('i8', ()), 'event_rows/frame_row': ('i4', ()),
}
OFFSET_OF = {'frames': 'frame_off', 'kills': 'kill_off', 'units': 'unit_off', 'objectives': 'obj_off', 'frame_rows': 'row_off',
             'onsets': 'onset_off', 'event_rows': 'event_off'}      # table -> its offset array in matches/
MACROS = ('FIT', 'STOP', 'DEV', 'CAL', 'AUDIT')


# ---- records of the runs (module level: workers import them) ---------------------------------------------------------------------
def hand_k(key, patch):
    """T = 900,000 ms, frames at 0 and 60,000 k + 300 (k = 1 .. 14); one engagement of two kills and four objectives, worked out in
    ``test_hand_worked_targets_and_rows``.

    kill A  300,000 ms  (5000, 5000)  blue 1 kills red 6, assists 2 and 3, bounty 300, shutdown 100
    kill B  305,000 ms  (5200, 5100)  red 7 kills blue 2, assist 8, bounty 250
    dragon  290,000     by blue 1 (assist 2), at the fight             -> attributed to the unit before its first kill
    tower   310,000     red 7 kills an outer turret of blue            -> attributed
    plate   330,000     no champion credit (blue gets it)              -> converted (delta = 320,000)
    inhib.  400,000     no position, blue 1 kills it                   -> rest (outside the conversion window, no position)
    """
    kills = [dict(t=300_000, x=5000, y=5000, killer=1, victim=6, assists=(2, 3), bounty=300, shutdown=100),
             dict(t=305_000, x=5200, y=5100, killer=7, victim=2, assists=(8,), bounty=250, shutdown=0)]
    objectives = [
        dict(t=290_000, type='ELITE_MONSTER_KILL', x=5100, y=5050, killer=1, assists=(2,), team=100, monster='DRAGON', sub='FIRE_DRAGON'),
        dict(t=310_000, type='BUILDING_KILL', x=5300, y=5000, killer=7, team=100, building='TOWER_BUILDING', tower='OUTER_TURRET'),
        dict(t=330_000, type='TURRET_PLATE_DESTROYED', x=9000, y=9000, killer=0, team=200),
        dict(t=400_000, type='BUILDING_KILL', x=None, y=None, killer=1, team=200, building='INHIBITOR_BUILDING')]
    return synth.make_record(T_ms=900_000, kills=kills, objectives=objectives, key=key, patch=patch)


def hand_neg(key, patch):
    """The Task 6 case with frame-row sentinels: T = 440,000, eight kills 20+ s apart; the M3 view of the unit at 120,000 is frame 1
    and the M4 view of the unit at 405,000 is a frame after T - 30 s, so two event rows have frame_row = -1."""
    times = (100_000, 120_000, 180_300, 262_000, 262_500, 300_300, 355_000, 405_000)
    kills = [dict(t=t, x=1_000.0, y=1_000.0, killer=1, victim=6) for t in times]
    return synth.make_record(T_ms=440_000, kills=kills, key=key, patch=patch)


def hand_big(key, patch):
    """Two kills of one blue killer in one unit with a bounty of 2e9 each: the summed bounty does not fit an int32."""
    kills = [dict(t=200_000, x=1000, y=1000, killer=1, victim=6, bounty=2_000_000_000),
             dict(t=205_000, x=1000, y=1000, killer=1, victim=7, bounty=2_000_000_000)]
    return synth.make_record(T_ms=900_000, kills=kills, key=key, patch=patch)


def hand_same_team(key, patch):
    return synth.make_record(T_ms=900_000, kills=[dict(t=200_000, x=1000, y=1000, killer=1, victim=2)], key=key, patch=patch)


def hand_empty(key, patch):
    """Frames only: no kill, no objective, so no unit, no onset and no event row (a few such matches are in the corpus)."""
    return synth.make_record(T_ms=600_000, key=key, patch=patch)


def hand_objectives_only(key, patch):
    """An objective but no kill: there is no unit to attribute it to or to convert it for."""
    objectives = [dict(t=200_000, type='ELITE_MONSTER_KILL', x=5000, y=5000, killer=1, team=100, monster='BARON_NASHOR')]
    return synth.make_record(T_ms=600_000, objectives=objectives, key=key, patch=patch)


def hand_short(key, patch):
    """Two regular frames only: no frame-clock row; the unit at 150,000 is in scope and its M3 and M4 views are frame 1."""
    kills = [dict(t=150_000, x=1000, y=1000, killer=1, victim=6)]
    return synth.make_record(T_ms=300_000, n_frames=2, kills=kills, key=key, patch=patch)


def records_hand(key, patch):
    """The hand-made matches by their keys, a random synthetic match otherwise."""
    hand = {'HAND_K': hand_k, 'HAND_NEG': hand_neg, 'HAND_BIG': hand_big, 'HAND_SAME': hand_same_team, 'HAND_EMPTY': hand_empty,
            'HAND_OBJ': hand_objectives_only, 'HAND_SHORT': hand_short}
    return hand[key](key, patch) if key in hand else synth.records_from_seed(key, patch)


def _number(key):
    return int(key[4:])        # 'SYN_0022' -> 22


def records_failing(key, patch):
    """A malformed file (not a rejection) from match 22 on: the run must stop."""
    if _number(key) >= 22:
        raise reader.MatchFormatError(key, patch, 'planted format error')
    return synth.records_from_seed(key, patch)


def records_new_only(key, patch):
    """Fails when a match of the blocks 0-2 (matches 0-20 at block size 7) is read again."""
    if _number(key) < 21:
        raise AssertionError('a block that is marked done was extracted again')
    return synth.records_from_seed(key, patch)


def records_rejecting(key, patch):
    n = _number(key)
    if n % 5 == 0:
        raise reader.MatchRejected('missing_file')
    if n % 7 == 3:
        raise reader.MatchRejected('no_game_end')
    return synth.records_from_seed(key, patch)


def records_reject_all(key, patch):
    raise reader.MatchRejected('missing_file')


def records_forbidden(key, patch):
    raise AssertionError('records must not be called')


def records_leaky(key, patch):
    raise OSError(f'cannot read {key}.json')


# ---- fixtures and helpers ---------------------------------------------------------------------------------------------------------
@pytest.fixture
def synth_matches():
    """30 synthetic matches (patch, key, macro): the patches 15.18-15.21 and the five S1 macros in turn; the key digits are the seed."""
    return [(C.PATCHES[i % 4], f'SYN_{i:04d}', MACROS[i % 5]) for i in range(30)]


@pytest.fixture
def mixed(synth_matches):
    """Random matches around the hand-made ones, so that the hand-made matches sit at non-trivial offsets."""
    hand = [('15.19', 'HAND_K', 'DEV'), ('15.20', 'HAND_NEG', 'CAL'), ('15.21', 'HAND_EMPTY', 'AUDIT'), ('15.18', 'HAND_OBJ', 'FIT'),
            ('15.19', 'HAND_SHORT', 'STOP')]
    return synth_matches[:8] + hand + synth_matches[8:12]


def files_hash(root):
    """{array file: SHA-256} of an assembled store; STORE.json (it records the block indices) and temporary files are left out."""
    s = store_core.Store(root)
    return s.hashes(s.stored_files())


def build(matches, run, **kw):
    """run_s1 on synthetic matches, loaded again."""
    kw.setdefault('workers', 1)
    kw.setdefault('records', records_hand)
    return run_s1.load_s1(s1_run(matches, run, **kw))


def rows_of(s, name, m):
    """The rows of array ``name`` that belong to match ``m`` (the one row of the match table itself)."""
    table = name.split('/')[0]
    if table == 'matches':
        return np.array(s.array(name)[m:m + 1])
    off = s.array('matches/' + OFFSET_OF[table])
    return np.array(s.array(name)[off[m]:off[m + 1]])


def start_of(s, table, m):
    return int(s.array('matches/' + OFFSET_OF[table])[m])


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')


roles_of = synth.roles_of       # the stand-in for cohort.role_of that synthetic keys need (see there); the real split: see the last test


def s1_run(matches, run, **kw):
    """``run_s1.run_s1`` on synthetic matches: ``roles_of`` answers for the cohort split unless the test gives another ``role_of``."""
    kw.setdefault('role_of', roles_of(matches))
    return run_s1.run_s1(matches, run, **kw)


# ---- the two tests of the task brief ----------------------------------------------------------------------------------------------
def test_s1_independent_of_workers_and_blocks(tmp_path, synth_matches):
    a = s1_run(synth_matches, tmp_path / 'a', workers=1, block_matches=7, records=synth.records_from_seed)
    b = s1_run(synth_matches, tmp_path / 'b', workers=2, block_matches=11, records=synth.records_from_seed)
    assert files_hash(a) == files_hash(b)
    assert set(files_hash(a)) == {name + '.npy' for name in LAYOUT}              # the comparison covers every array: it is not vacuous


def test_s1_offsets_consistent(tmp_path, synth_matches):
    s = run_s1.load_s1(s1_run(synth_matches, tmp_path, workers=1, records=synth.records_from_seed))
    assert s.array('matches/kill_off')[-1] == len(s.array('kills/t')) and np.all(s.array('kills/unit') >= 0)


# ---- layout -----------------------------------------------------------------------------------------------------------------------
def test_store_layout_is_the_ruling_table(tmp_path, synth_matches):
    s = build(synth_matches, tmp_path, records=synth.records_from_seed, block_matches=7)
    assert set(s.manifest['arrays']) == set(LAYOUT) and s.manifest['complete'] is True
    n = len(synth_matches)
    for name, (dtype, shape) in LAYOUT.items():
        a = s.array(name)
        assert a.dtype == np.dtype(dtype) and a.shape[1:] == shape, name
        table = name.split('/')[0]
        rows = n + 1 if name.endswith('_off') else n if table == 'matches' else int(s.array('matches/' + OFFSET_OF[table])[-1])
        assert len(a) == rows, name
    for table, off in OFFSET_OF.items():
        o = s.array('matches/' + off)
        assert o[0] == 0 and np.all(np.diff(o) >= 0), table                  # offsets start at 0 and never go back
    # the match table: rows in the order of the list, patch and role as indices of the config lists
    assert [k.decode() for k in s.array('matches/key')] == [k for _, k, _ in synth_matches]
    assert s.array('matches/patch').tolist() == [C.PATCHES.index(p) for p, _, _ in synth_matches]
    assert s.array('matches/role').tolist() == [C.ROLES.index(m) for _, _, m in synth_matches]
    # the extract stage holds the plan and the finished blocks, and no timing field
    stage = s.manifest['stages']['extract']
    assert set(stage) == {'blocks', 'block_matches', 'plan_sha256', 'n_matches', 'n_blocks'}
    assert stage['block_matches'] == 7 and stage['n_matches'] == 30 and stage['n_blocks'] == 5 and sorted(stage['blocks']) == list(range(5))
    assert len(stage['plan_sha256']) == 64 and 'assemble' in s.manifest['stages']


# array -> (table whose global start is added to its local value, only where the value is >= 0); the rule of the assembly, written out
SHIFTS = {'kills/unit': ('units', False), 'units/kill0': ('kills', False), 'objectives/unit': ('units', True),
          'objectives/conv_unit': ('units', True), 'onsets/row': ('frame_rows', False), 'onsets/unit': ('units', False),
          'event_rows/unit': ('units', False), 'event_rows/frame_row': ('frame_rows', True)}


def test_assembled_store_equals_the_per_match_extraction(tmp_path, mixed):
    """Every match in the store is what ``extract_match`` gives for it, with the global offsets added to the index columns (-1 stays -1)
    and the match index in ``units/match`` and ``frame_rows/match``; the other columns are copied as they are."""
    s = build(mixed, tmp_path, block_matches=5)
    expect_names = set(LAYOUT) - {n for n in LAYOUT if n.endswith('_off')}
    for m, (patch, key, macro) in enumerate(mixed):
        want = run_s1.extract_match(patch, key, macro, records=records_hand)
        lo = {t: start_of(s, t, m) for t in OFFSET_OF}
        for name, (table, only_valid) in SHIFTS.items():
            a = want[name].astype(np.int64)
            want[name] = (np.where(a >= 0, a + lo[table], a) if only_valid else a + lo[table]).astype(want[name].dtype)
        want['units/match'] = np.full(len(want['units/f']), m, dtype=np.int32)
        want['frame_rows/match'] = np.full(len(want['frame_rows/k']), m, dtype=np.int32)
        assert set(want) == expect_names
        assert [n for n in sorted(want) if not same(want[n], rows_of(s, n, m))] == [], m


def test_offsets_keep_minus_one(tmp_path, mixed):
    """-1 means none (no unit, no conversion, no frame-clock row) and must stay -1; every other index is global and points into the
    rows of its own match."""
    s = build(mixed, tmp_path, block_matches=3)
    for name, target in (('objectives/unit', 'units'), ('objectives/conv_unit', 'units'), ('event_rows/frame_row', 'frame_rows')):
        a = np.array(s.array(name))
        assert a.min() == -1, name                                           # the sentinel is there: the test is not vacuous
        for m in range(len(mixed)):
            v = rows_of(s, name, m)
            ok = v[v >= 0]
            assert np.all(v >= -1) and np.all((ok >= start_of(s, target, m)) & (ok < start_of(s, target, m + 1))), (name, m)
    # the other index columns have no sentinel and point into their own match too
    for name, target in (('kills/unit', 'units'), ('units/kill0', 'kills'), ('onsets/row', 'frame_rows'), ('onsets/unit', 'units'),
                         ('event_rows/unit', 'units')):
        for m in range(len(mixed)):
            v = rows_of(s, name, m)
            assert np.all((v >= start_of(s, target, m)) & (v < start_of(s, target, m + 1))), (name, m)
    assert np.all(np.array(s.array('units/in_scope'))[s.array('onsets/unit')])      # every onset is a unit in scope


def test_hand_worked_targets_and_rows(tmp_path):
    ms = [('15.18', 'SYN_0001', 'FIT'), ('15.19', 'HAND_K', 'DEV'), ('15.20', 'SYN_0002', 'STOP')]
    s = build(ms, tmp_path)
    u0, k0, r0 = (start_of(s, t, 1) for t in ('units', 'kills', 'frame_rows'))
    assert u0 > 0 and k0 > 0 and r0 > 0 and start_of(s, 'objectives', 1) > 0           # the hand-made match is not at offset 0

    def at(name):
        return rows_of(s, name, 1).tolist()
    # kills: two of one unit (local unit 0 -> global u0)
    assert at('kills/t') == [300_000, 305_000] and at('kills/killer') == [1, 7] and at('kills/victim') == [6, 2]
    assert at('kills/amask') == [0b110, 0b10000000] and at('kills/bounty') == [300, 250] and at('kills/shutdown') == [100, 0]
    assert at('kills/unit') == [u0, u0] and at('kills/pos_missing') == [False, False]
    # the unit: f, last, tau, delta, people 3 + 3 (size S), first-kill mark 0, K over the whole unit and up to f
    assert at('units/match') == [1] and at('units/kill0') == [k0] and at('units/n_kills') == [2]
    assert (at('units/f'), at('units/last'), at('units/tau'), at('units/delta')) == ([300_000], [305_000], [285_000], [320_000])
    assert (at('units/nb'), at('units/nr'), at('units/size'), at('units/mark'), at('units/in_scope')) == ([3], [3], [1], [0], [True])
    assert at('units/K') == [[1, 1, 300, 250, 100, 0, 2, 1]] and at('units/K_at_f') == [[1, 0, 300, 0, 100, 0, 2, 0]]
    assert at('units/exec') == [[0, 0]] and at('units/conv_exposure_ms') == [60_000]
    assert at('units/spread')[0] == pytest.approx(np.hypot(200, 100), rel=1e-6)
    cells = lambda *idx: [[1 if i in idx else 0 for i in range(18)]]            # noqa: E731
    # dragon of blue (cell 0) and tower of red (cell 13) are the unit's; the plate of blue (cell 16) is converted to it
    assert at('units/O') == cells(0, 13) and at('units/O_at_f') == cells(0) and at('units/conv') == cells(16)
    # objectives: attributed (unit), converted (conv_unit) or rest, in event order; the inhibitor has no position
    assert at('objectives/t') == [290_000, 310_000, 330_000, 400_000] and at('objectives/kind') == [0, 6, 8, 7]
    assert at('objectives/taker') == [0, 1, 0, 0] and at('objectives/has_pos') == [True, True, True, False]
    assert at('objectives/cred') == [0b11, 0b1000000, 0, 0b1] and at('objectives/sub') == [-1, 0, -1, -1]
    assert at('objectives/unit') == [u0, u0, -1, -1] and at('objectives/conv_unit') == [-1, -1, u0, -1]
    x, y = np.array(at('objectives/x')), np.array(at('objectives/y'))
    assert x[:3].tolist() == [5100.0, 5300.0, 9000.0] and y[:3].tolist() == [5050.0, 5000.0, 9000.0] and np.isnan(x[3]) and np.isnan(y[3])
    # frame rows k = 2 .. 14, onsets of the unit in the three rows whose window holds f, and its two event rows
    assert at('frame_rows/match') == [1] * 13 and at('frame_rows/k') == list(range(2, 15))
    assert at('frame_rows/t0')[:2] == [120_300, 180_300] and at('frame_rows/hi')[0] == 300_300 and at('frame_rows/hi')[-1] == 870_000
    assert at('frame_rows/exposure_ms')[-1][:4] == [10_000, 10_000, 9_700, 0]
    assert at('onsets/row') == [r0, r0 + 1, r0 + 2] and at('onsets/bin') == [17, 11, 5] and at('onsets/mark') == [0, 0, 0]
    assert at('onsets/unit') == [u0, u0, u0]
    assert at('event_rows/unit') == [u0, u0] and at('event_rows/kind') == [1, 2] and at('event_rows/ref') == [300_000, 320_000]
    assert at('event_rows/frame_k') == [4, 5] and at('event_rows/t0') == [240_300, 300_300] and at('event_rows/age_ms') == [59_700, 19_700]
    assert at('event_rows/frame_row') == [r0 + 2, r0 + 3]
    # the match itself
    rec = hand_k('HAND_K', '15.19')
    assert at('matches/T_ms') == [900_000] and at('matches/patch') == [1] and at('matches/role') == [C.ROLES.index('DEV')]
    assert at('matches/team') == [rec.team.tolist()] and at('matches/champ') == [rec.champ.tolist()]
    assert at('frames/ts') == rec.frame_ts.tolist() and same(rows_of(s, 'frames/num', 1), rec.frame_num.astype(np.float32))
    assert same(rows_of(s, 'matches/end_xy', 1), rec.end_xy.astype(np.float32)[None])


def test_matches_without_units_or_frame_rows_have_empty_slices(tmp_path):
    ms = [('15.18', 'SYN_0001', 'FIT'), ('15.21', 'HAND_EMPTY', 'AUDIT'), ('15.18', 'HAND_OBJ', 'FIT'), ('15.19', 'HAND_SHORT', 'STOP')]
    s = build(ms, tmp_path)
    n_empty = {t: [len(rows_of(s, f'{t}/{c}', m)) for m in (1, 2, 3)] for t, c in
               (('frames', 'ts'), ('kills', 't'), ('units', 'f'), ('objectives', 't'), ('frame_rows', 'k'), ('onsets', 'row'),
                ('event_rows', 'ref'))}
    # no kill: no unit, no onset, no event row; frames and rows are there (T = 600,000: 10 frames, rows k = 2 .. 9)
    assert n_empty['kills'] == [0, 0, 1] and n_empty['units'] == [0, 0, 1] and n_empty['onsets'] == [0, 0, 0]
    assert n_empty['event_rows'] == [0, 0, 2] and n_empty['frames'] == [10, 10, 2] and n_empty['frame_rows'] == [8, 8, 0]
    # an objective without a unit is not attributed and not converted
    assert rows_of(s, 'objectives/kind', 2).tolist() == [4] and rows_of(s, 'objectives/unit', 2).tolist() == [-1]
    assert rows_of(s, 'objectives/conv_unit', 2).tolist() == [-1] and rows_of(s, 'objectives/taker', 2).tolist() == [0]
    # two frames: the views of the one unit use frame 1, which is no row
    assert rows_of(s, 'event_rows/frame_k', 3).tolist() == [1, 1] and rows_of(s, 'event_rows/frame_row', 3).tolist() == [-1, -1]
    assert rows_of(s, 'event_rows/age_ms', 3).tolist() == [89_700, 109_700] and rows_of(s, 'units/in_scope', 3).tolist() == [True]
    assert rows_of(s, 'frame_rows/exposure_ms', 3).shape == (0, 18)


def test_hand_frame_row_sentinels_are_kept(tmp_path):
    s = build([('15.18', 'SYN_0003', 'FIT'), ('15.18', 'HAND_NEG', 'FIT')], tmp_path)
    r0 = start_of(s, 'frame_rows', 1)
    local = [-1, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, -1]                                  # the frame_row column worked out in test_event_rows
    assert r0 > 0 and rows_of(s, 'event_rows/frame_row', 1).tolist() == [v if v == -1 else v + r0 for v in local]


# ---- narrowing casts --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('value, dtype', [(40_000, 'i2'), (-32_769, 'i2'), (2 ** 31, 'i4'), (-(2 ** 31) - 1, 'i4'), (128, 'i1'),
                                          (-129, 'i1'), (2 ** 63 - 1, 'i4')])
def test_narrowing_integer_cast_raises_on_overflow(value, dtype):
    with pytest.raises(ValueError, match='units/K'):
        run_s1.narrow('units/K', np.array([1, value], dtype=np.int64), np.dtype(dtype))


@pytest.mark.parametrize('value, dtype', [(32_767, 'i2'), (-32_768, 'i2'), (2 ** 31 - 1, 'i4'), (127, 'i1'), (-128, 'i1')])
def test_narrowing_integer_cast_keeps_values_that_fit(value, dtype):
    out = run_s1.narrow('units/K', np.array([0, value], dtype=np.int64), np.dtype(dtype))
    assert out.dtype == np.dtype(dtype) and out.tolist() == [0, value] and out.flags['C_CONTIGUOUS']


@pytest.mark.parametrize('value', [1e39, -1e39, np.inf, -np.inf])
def test_narrowing_float_cast_raises_unless_finite(value):
    with pytest.raises(ValueError, match='objectives/x'):
        run_s1.narrow('objectives/x', np.array([1.0, value]), np.dtype('f4'))


def test_narrowing_float_cast_keeps_nan_and_finite_values():
    out = run_s1.narrow('objectives/x', np.array([1.5, np.nan, 14_875.0]), np.dtype('f4'))
    assert out.dtype == np.float32 and out[0] == 1.5 and np.isnan(out[1]) and out[2] == 14_875.0


def test_narrowing_refuses_other_kind_changes():
    with pytest.raises(TypeError):
        run_s1.narrow('kills/pos_missing', np.array([0, 1]), np.dtype('?'))          # an integer is not a bool
    with pytest.raises(TypeError):
        run_s1.narrow('units/K', np.array([0.5]), np.dtype('i4'))                    # a float is never narrowed to an integer
    assert run_s1.narrow('kills/pos_missing', np.array([True, False]), np.dtype('?')).tolist() == [True, False]


def test_a_summed_target_that_does_not_fit_stops_the_match():
    with pytest.raises(ValueError, match='units/K'):
        run_s1.extract_match('15.18', 'HAND_BIG', 'FIT', records=records_hand)


# ---- keys, refusals, rejections ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('key', ['K' * 25, 'KR_' + 'é' * 3])
def test_key_that_does_not_fit_24_ascii_bytes_raises(tmp_path, key):
    with pytest.raises(ValueError, match='24') as e:
        run_s1.extract_match('15.18', key, 'FIT', records=records_forbidden)
    assert key not in str(e.value)
    with pytest.raises(ValueError, match='24'):
        run_s1.run_s1([('15.18', key, 'FIT')], tmp_path / 'r', workers=1, records=records_forbidden)
    assert not (tmp_path / 'r').exists()                                            # refused before anything is written


def test_key_of_24_bytes_is_kept_whole(tmp_path):
    key = 'SYN_' + '9' * 20                                                          # 24 bytes
    assert len(key) == 24
    s = build([('15.18', key, 'FIT')], tmp_path, records=synth.records_from_seed)
    assert s.array('matches/key')[0] == key.encode()


@pytest.mark.parametrize('match', [('15.22', 'SYN_0001', 'FIT'), ('16.13', 'SYN_0001', 'FIT'), ('15.18', 'SYN_0001', 'TEST'),
                                   ('15.18', 'SYN_0001', 'EXTERNAL'), ('15.18', 'SYN_0001', 'NOPE'), ('14.01', 'SYN_0001', 'FIT')])
def test_runs_refuse_holdout_patches_and_roles_before_reading_anything(tmp_path, match):
    with pytest.raises(ValueError):
        run_s1.run_s1([('15.18', 'SYN_0000', 'FIT'), match], tmp_path / 'r', workers=1, records=records_forbidden)
    assert not (tmp_path / 'r').exists()
    with pytest.raises(ValueError):
        run_s1.golden(1, tmp_path / 'g', records=records_forbidden, matches=[match])


@pytest.mark.parametrize('block_matches, error', [(0, ValueError), (-3, ValueError), (2.0, TypeError), (True, TypeError), ('7', TypeError)])
def test_bad_block_sizes_and_empty_lists_are_refused_before_anything_is_written(tmp_path, synth_matches, block_matches, error):
    with pytest.raises(error):
        run_s1.run_s1(synth_matches, tmp_path / 'r', workers=1, block_matches=block_matches, records=records_forbidden)
    with pytest.raises(ValueError, match='no matches'):
        run_s1.run_s1([], tmp_path / 'r', workers=1, records=records_forbidden)
    with pytest.raises(ValueError, match='workers'):
        run_s1.run_s1(synth_matches, tmp_path / 'r', workers=0, records=records_forbidden)
    assert not (tmp_path / 'r').exists()


def test_rejected_matches_are_counted_by_patch_and_code_and_left_out(tmp_path, synth_matches):
    run = tmp_path / 'r'
    s = build(synth_matches, run, records=records_rejecting, block_matches=4)
    want = {}
    for p, k, _ in synth_matches:
        n = _number(k)
        code = 'missing_file' if n % 5 == 0 else 'no_game_end' if n % 7 == 3 else None
        if code:
            want.setdefault(p, {}).setdefault(code, 0)
            want[p][code] += 1
    text = (run / 'REJECTS.json').read_text(encoding='utf-8')
    assert json.loads(text) == want and sum(sum(v.values()) for v in want.values()) == 9
    assert not [k for _, k, _ in synth_matches if k in text]                       # no match key in the file
    kept = [k for _, k, _ in synth_matches if _number(k) % 5 and _number(k) % 7 != 3]
    assert [k.decode() for k in s.array('matches/key')] == kept
    res = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))['s1_result']
    assert res['rejects_total'] == 9 and res['rejects'] == want and res['matches_stored'] == len(kept)
    assert res['matches_requested'] == 30
    # the rejects do not depend on the block plan, and neither does the store
    other = s1_run(synth_matches, tmp_path / 'o', workers=1, block_matches=9, records=records_rejecting)
    assert files_hash(other) == files_hash(s.root) and json.loads((tmp_path / 'o' / 'REJECTS.json').read_text(encoding='utf-8')) == want


def test_a_run_without_rejects_writes_an_empty_rejects_file(tmp_path, synth_matches):
    build(synth_matches, tmp_path, records=synth.records_from_seed)
    assert json.loads((tmp_path / 'REJECTS.json').read_text(encoding='utf-8')) == {}


def test_all_matches_rejected_gives_an_empty_complete_store(tmp_path, synth_matches):
    s = build(synth_matches[:6], tmp_path, records=records_reject_all, block_matches=4)
    assert s.manifest['complete'] and len(s.array('matches/key')) == 0 and s.array('matches/kill_off').tolist() == [0]
    assert all(len(s.array(n)) == 0 for n in LAYOUT if not n.endswith('_off'))
    want = {'15.18': {'missing_file': 2}, '15.19': {'missing_file': 2}, '15.20': {'missing_file': 1}, '15.21': {'missing_file': 1}}
    assert json.loads((tmp_path / 'REJECTS.json').read_text(encoding='utf-8')) == want


def test_extract_match_returns_rejections_and_raises_format_errors():
    assert run_s1.extract_match('15.18', 'SYN_0000', 'FIT', records=records_reject_all) == {'reject': 'missing_file'}
    assert run_s1.extract_match('15.18', 'HAND_SAME', 'FIT', records=records_hand) == {'reject': 'same_team_kill'}    # from build_units
    with pytest.raises(reader.MatchFormatError):
        run_s1.extract_match('15.18', 'SYN_0023', 'FIT', records=records_failing)


def test_extract_match_schema():
    got = run_s1.extract_match('15.18', 'HAND_K', 'FIT', records=records_hand)
    want = set(LAYOUT) - {'units/match', 'frame_rows/match'} - {n for n in LAYOUT if n.endswith('_off')}
    assert set(got) == want
    for name in want:
        dtype, shape = LAYOUT[name]
        assert got[name].dtype == np.dtype(dtype) and got[name].shape[1:] == shape, name


def test_extract_match_refuses_a_record_of_another_match():
    with pytest.raises(run_s1.S1CheckError, match='another match'):
        run_s1.extract_match('15.18', 'SYN_0001', 'FIT', records=lambda key, patch: synth.records_from_seed('SYN_0002', patch))


# ---- the loud per-match checks ----------------------------------------------------------------------------------------------------
def test_loud_check_m3_rows_equal_the_in_scope_units(monkeypatch):
    real = event_rows.event_rows

    def drop_first(rec, u, rows):
        e = real(rec, u, rows)
        keep = np.arange(len(e.unit)) > 0
        return dataclasses.replace(e, **{f.name: getattr(e, f.name)[keep] for f in dataclasses.fields(e)})
    monkeypatch.setattr(event_rows, 'event_rows', drop_first)
    with pytest.raises(run_s1.S1CheckError, match='M3') as e:
        run_s1.extract_match('15.18', 'HAND_K', 'FIT', records=records_hand)
    assert 'HAND_K' not in str(e.value) and '15.18' in str(e.value)                  # the patch is named, the key is not


def test_loud_check_kill_times_do_not_decrease(monkeypatch):
    real = units.kill_table

    def reversed_times(rec):
        k = real(rec)
        return dataclasses.replace(k, t=k.t[::-1].copy())
    monkeypatch.setattr(units, 'kill_table', reversed_times)
    with pytest.raises(run_s1.S1CheckError, match='kill times'):
        run_s1.extract_match('15.18', 'HAND_K', 'FIT', records=records_hand)


def test_loud_check_every_kill_has_a_unit(monkeypatch):
    real = units.build_units

    def lose_a_kill(k, team, T_ms):
        u = real(k, team, T_ms)
        return dataclasses.replace(u, comp=[u.comp[0][:-1]] + u.comp[1:])
    monkeypatch.setattr(units, 'build_units', lose_a_kill)
    with pytest.raises(run_s1.S1CheckError, match='no unit'):
        run_s1.extract_match('15.18', 'HAND_K', 'FIT', records=records_hand)


def test_loud_check_every_onset_is_a_unit_in_scope(monkeypatch):
    """HAND_NEG's first unit (f = 100,000) is before 2:00: an onset on it must stop the run."""
    def onset_on_unit_0(rows, u):
        return frame_rows.Onsets(row=np.array([0], np.int32), bin=np.array([0], np.int8), mark=np.array([0], np.int16),
                                 unit=np.array([0], np.int32))
    monkeypatch.setattr(frame_rows, 'onsets', onset_on_unit_0)
    with pytest.raises(run_s1.S1CheckError, match='onset'):
        run_s1.extract_match('15.18', 'HAND_NEG', 'FIT', records=records_hand)


def test_checks_pass_on_the_hand_made_and_random_matches():
    for key in ('HAND_K', 'HAND_NEG', 'SYN_0000', 'SYN_0007'):
        assert 'reject' not in run_s1.extract_match('15.18', key, 'FIT', records=records_hand)


def test_unexpected_errors_stop_the_run_without_the_key_in_the_message(tmp_path):
    with pytest.raises(run_s1.S1WorkerError) as e:
        s1_run([('15.18', 'SYN_0001', 'FIT')], tmp_path / 'r', workers=1, records=records_leaky)
    assert 'SYN_0001' not in str(e.value) and 'OSError' in str(e.value) and '15.18' in str(e.value)
    assert not store_core.Store(tmp_path / 'r' / 's1').manifest['stages']['extract']['blocks']


# ---- resume and the plan ----------------------------------------------------------------------------------------------------------
def test_interrupted_run_resumes_with_the_same_plan_only(tmp_path, synth_matches):
    ref = s1_run(synth_matches, tmp_path / 'ref', workers=1, block_matches=7, records=synth.records_from_seed)
    run = tmp_path / 'r'
    with pytest.raises(reader.MatchFormatError):                                    # a malformed file from match 22 on stops the run
        s1_run(synth_matches, run, workers=2, block_matches=7, records=records_failing)
    store = store_core.Store(run / 's1')
    assert store.done('extract') == {0, 1, 2} and not store.manifest['complete']
    run_json = (run / 'RUN.json').read_bytes()
    # another block size or another match list: refused, and the saved run is not changed
    with pytest.raises(ValueError, match='block_matches'):
        s1_run(synth_matches, run, workers=1, block_matches=11, records=synth.records_from_seed)
    one_other = synth_matches[:5] + [('15.18', 'SYN_0099', 'FIT')] + synth_matches[6:]
    for other in (synth_matches[:-1], list(reversed(synth_matches)), one_other):
        with pytest.raises(ValueError, match='match list'):
            s1_run(other, run, workers=1, block_matches=7, records=synth.records_from_seed)
    assert store_core.Store(run / 's1').done('extract') == {0, 1, 2} and (run / 'RUN.json').read_bytes() == run_json
    # the same plan resumes (with another worker count); blocks 0-2 are not read again; the store is the uninterrupted one
    done = s1_run(synth_matches, run, workers=1, block_matches=7, records=records_new_only)
    assert files_hash(done) == files_hash(ref)
    man = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    assert man['created'] == json.loads(run_json.decode('utf-8'))['created'] and man['s1_result']['blocks_resumed'] == 3
    run_s1.load_s1(run)                                                              # verifies the store


def test_a_complete_run_is_returned_untouched_and_the_plan_is_still_checked(tmp_path, synth_matches):
    run = tmp_path / 'r'
    a = s1_run(synth_matches, run, workers=1, block_matches=7, records=synth.records_from_seed)
    before = {p: p.read_bytes() for p in (run / 'RUN.json', a / 'STORE.json', run / 'REJECTS.json')}
    assert s1_run(synth_matches, run, workers=1, block_matches=7, records=records_forbidden) == a
    assert all(p.read_bytes() == v for p, v in before.items())
    with pytest.raises(ValueError, match='match list'):
        s1_run(synth_matches[:-1], run, workers=1, block_matches=7, records=records_forbidden)
    with pytest.raises(ValueError, match='block_matches'):
        s1_run(synth_matches, run, workers=1, block_matches=8, records=records_forbidden)


def test_plan_hash_is_over_the_ordered_list(synth_matches):
    h = run_s1.plan_sha256(synth_matches)
    assert len(h) == 64 and h == run_s1.plan_sha256(list(map(tuple, synth_matches)))
    assert h != run_s1.plan_sha256(synth_matches[::-1]) and h != run_s1.plan_sha256(synth_matches[:-1])
    changed = list(synth_matches)
    changed[3] = (changed[3][0], changed[3][1], 'DEV' if changed[3][2] != 'DEV' else 'FIT')
    assert h != run_s1.plan_sha256(changed)


def test_files_hash_leaves_out_store_json_and_block_files(tmp_path, synth_matches):
    a = s1_run(synth_matches, tmp_path / 'a', workers=1, block_matches=7, records=synth.records_from_seed)
    b = s1_run(synth_matches, tmp_path / 'b', workers=1, block_matches=11, records=synth.records_from_seed)
    assert (a / 'STORE.json').read_bytes() != (b / 'STORE.json').read_bytes()       # it records the block indices: it must not be hashed
    assert files_hash(a) == files_hash(b) and 'STORE.json' not in files_hash(a)
    assert all(name.endswith('.npy') for name in files_hash(a)) and not list(a.rglob('*.npz'))
    blocks = tmp_path / 'a' / 's1_blocks'                                            # block files live next to the store, not in it
    assert sorted(p.name for p in blocks.iterdir()) == [f'b{i:04d}.npz' for i in range(5)]
    assert a == tmp_path / 'a' / 's1' and not list(tmp_path.rglob('*.tmp'))


# ---- run record -------------------------------------------------------------------------------------------------------------------
def test_run_record_in_run_json(tmp_path, synth_matches):
    run = tmp_path / '20261020_c_ept_s1_1'
    root = s1_run(synth_matches, run, workers=1, block_matches=7, records=synth.records_from_seed, info={'macros': list(MACROS)})
    man = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    assert man['stage'] == 's1' and len(man['code_commit']) == 40 and 'strata_running' in man and 'split_policy_sha256' in man
    assert man['args']['workers'] == 1 and man['args']['block_matches'] == 7 and man['args']['macros'] == list(MACROS)
    assert man['args']['n_matches'] == 30 and len(man['args']['plan_sha256']) == 64
    r = man['s1_result']
    assert r['complete'] is True and r['wall_s'] > 0 and set(r['stage_s']) == {'extract', 'assemble', 'finalize'}
    assert all(v >= 0 for v in r['stage_s'].values()) and r['stage_s']['extract'] <= r['wall_s']
    assert r['peak_private_gib'] > 0 and r['memory_samples'] >= 1                    # psutil is installed in this environment
    assert (r['workers'], r['block_matches'], r['n_blocks'], r['blocks_resumed']) == (1, 7, 5, 0)
    assert r['matches_requested'] == r['matches_stored'] == 30 and r['rejects_total'] == 0 and r['rejects'] == {}
    assert r['matches_by_macro'] == {m: 6 for m in MACROS} and r['matches_by_patch'] == {'15.18': 8, '15.19': 8, '15.20': 7, '15.21': 7}
    assert sum(r['matches_by_macro_patch'][m][p] for m in MACROS for p in r['matches_by_macro_patch'][m]) == 30
    s = store_core.Store(root)
    assert r['rows'] == {'matches': 30, 'frames': len(s.array('frames/ts')), 'kills': len(s.array('kills/t')),
                         'units': len(s.array('units/f')), 'objectives': len(s.array('objectives/t')),
                         'frame_rows': len(s.array('frame_rows/k')), 'onsets': len(s.array('onsets/row')),
                         'event_rows': len(s.array('event_rows/ref'))}
    assert r['store_files'] == len(LAYOUT) + 1
    assert r['store_bytes'] == sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    del s


def test_run_record_of_a_folder_without_a_run_name_has_no_stage(tmp_path, synth_matches):
    build(synth_matches[:4], tmp_path / 'x', records=synth.records_from_seed)
    assert json.loads((tmp_path / 'x' / 'RUN.json').read_text(encoding='utf-8'))['stage'] is None


# ---- amend_manifest ---------------------------------------------------------------------------------------------------------------
def test_amend_manifest_adds_keys_and_never_overwrites(tmp_path):
    m = runlog.write_manifest(tmp_path, {'args': {'a': 1}})
    out = runlog.amend_manifest(tmp_path, {'timing': {'wall_s': 1.5}, 'extra': [1, 2]})
    assert out == {**m, 'timing': {'wall_s': 1.5}, 'extra': [1, 2]}
    assert json.loads((tmp_path / 'RUN.json').read_text(encoding='utf-8')) == out
    assert not list(tmp_path.glob('*.tmp'))                                          # written atomically: no leftover temporary file
    before = (tmp_path / 'RUN.json').read_bytes()
    for clash in ({'timing': 1}, {'new': 1, 'stage': 'x'}, {'code_commit': 'x'}, {'args': {}}):
        with pytest.raises(ValueError, match='already'):
            runlog.amend_manifest(tmp_path, clash)
    assert (tmp_path / 'RUN.json').read_bytes() == before                           # a refused amendment writes nothing
    with pytest.raises(TypeError):
        runlog.amend_manifest(tmp_path, {1: 'a'})
    with pytest.raises(FileNotFoundError):
        runlog.amend_manifest(tmp_path / 'nowhere', {'a': 1})


def test_amend_manifest_handles_numpy_and_paths_like_write_manifest(tmp_path):
    runlog.write_manifest(tmp_path, {})
    out = runlog.amend_manifest(tmp_path, {'n': np.int64(5), 'where': tmp_path})
    assert out['n'] == 5 and out['where'] == str(tmp_path)


# ---- load_s1 ----------------------------------------------------------------------------------------------------------------------
def test_load_s1_takes_the_run_or_the_store_and_verifies(tmp_path, synth_matches):
    root = s1_run(synth_matches, tmp_path / 'r', workers=1, records=synth.records_from_seed)
    a, b = run_s1.load_s1(tmp_path / 'r'), run_s1.load_s1(root)
    assert a.root == b.root == root and a.manifest['complete']
    del a, b
    arr = np.load(root / 'kills' / 't.npy', mmap_mode='r+')                          # damage one array
    arr[0] += 1
    arr.flush()
    del arr
    with pytest.raises(ValueError, match='t.npy'):
        run_s1.load_s1(tmp_path / 'r')
    assert run_s1.load_s1(tmp_path / 'r', verify=False).manifest['complete']         # the check can be skipped, the damage stays


def test_load_s1_refuses_an_incomplete_or_missing_store(tmp_path, synth_matches):
    with pytest.raises(FileNotFoundError):
        run_s1.load_s1(tmp_path / 'nowhere')
    with pytest.raises(reader.MatchFormatError):
        s1_run(synth_matches, tmp_path / 'r', workers=1, block_matches=7, records=records_failing)
    for verify in (True, False):
        with pytest.raises(ValueError):
            run_s1.load_s1(tmp_path / 'r', verify=verify)


# ---- golden (written here, run on the cached matches in task 8b) ------------------------------------------------------------------
def test_golden_writes_array_hashes_of_the_first_n_matches_and_no_keys(tmp_path, synth_matches):
    out = tmp_path / 'golden'
    info = run_s1.golden(5, out, records=synth.records_from_seed, matches=synth_matches,
                         role_of=roles_of(synth_matches))
    text = (out / 'GOLDEN.json').read_text(encoding='utf-8')
    g = json.loads(text)
    assert g == info and g['n_matches'] == 5 and not [k for _, k, _ in synth_matches if k in text]
    # the hashes are those of an ordinary run over the same five matches, whatever its blocks and workers
    ref = s1_run(synth_matches[:5], tmp_path / 'ref', workers=2, block_matches=2, records=synth.records_from_seed)
    assert g['files'] == files_hash(ref) and set(g['files']) == {n + '.npy' for n in LAYOUT}
    assert g['plan_sha256'] == run_s1.plan_sha256(synth_matches[:5]) and len(g['code_commit']) == 40
    with pytest.raises(FileExistsError):                                             # a lock file is not overwritten
        run_s1.golden(5, out, records=synth.records_from_seed, matches=synth_matches,
                      role_of=roles_of(synth_matches))
    assert json.loads((out / 'GOLDEN.json').read_text(encoding='utf-8')) == g


# ---- the state of the code in a golden lock (task 8b) -----------------------------------------------------------------------------
EXTRACTION_PATHS = [
    'c_ept/__init__.py',
    'c_ept/_compat/__init__.py',
    'c_ept/config.py',
    'c_ept/data/reader.py',
    'c_ept/_compat/reader.py',
    'c_ept/domain/units.py',
    'c_ept/_compat/units.py',
    'c_ept/domain/targets.py',
    'c_ept/_compat/targets.py',
    'c_ept/data/frame_rows.py',
    'c_ept/_compat/frame_rows.py',
    'c_ept/data/event_rows.py',
    'c_ept/_compat/event_rows.py',
    'c_ept/data/store_core.py',
    'c_ept/_compat/store_core.py',
    'c_ept/workflows/run_s1.py',
    'c_ept/_compat/run_s1.py',
    'c_ept/legacy.py',
    'c_ept/cohort.py',
    'c_ept/data/casts.py',
    'c_ept/_compat/casts.py',
    'c_ept/data/__init__.py',
    'c_ept/domain/__init__.py',
    'c_ept/workflows/__init__.py',
]


def fake_git(status, head='f' * 40):
    """A stand-in for ``runlog._git``: ``rev-parse`` gives ``head``, ``status`` gives the text of ``git status --porcelain`` (None: git
    fails). Returns the function and the list of calls it received."""
    calls = []

    def git(*args):
        calls.append(args)
        return head + '\n' if args[0] == 'rev-parse' else status
    return git, calls


@pytest.mark.parametrize('status, clean, dirty', [
    ('', True, []),
    (' M c_ept/units.py\n?? c_ept/targets.py\nM  c_ept/run_s1.py\n', False, ['c_ept/run_s1.py', 'c_ept/targets.py', 'c_ept/units.py']),
    ('R  c_ept/old.py -> c_ept/reader.py\n D c_ept/cohort.py\n', False, ['c_ept/cohort.py', 'c_ept/reader.py']),
    (None, None, None)], ids=['clean', 'modified-staged-untracked', 'renamed-deleted', 'git-unavailable'])
def test_extraction_state_is_read_from_git_status(monkeypatch, status, clean, dirty):
    git, calls = fake_git(status)
    monkeypatch.setattr(runlog, '_git', git)
    state = run_s1.extraction_state()
    assert (state['extraction_clean'], state['dirty_paths']) == (clean, dirty) and state['code_commit'] == 'f' * 40
    assert state['extraction_modules'] == EXTRACTION_PATHS
    assert [c for c in calls if c[0] == 'status'] == [('status', '--porcelain', '--', *EXTRACTION_PATHS)]     # the fixed implementation and compatibility inventory


def test_golden_records_the_extraction_state_and_the_record_is_no_input_of_the_arrays(tmp_path, synth_matches, monkeypatch):
    got = {}
    for name, status in (('clean', ''), ('dirty', ' M c_ept/units.py\n')):
        monkeypatch.setattr(runlog, '_git', fake_git(status)[0])
        got[name] = run_s1.golden(5, tmp_path / name, records=synth.records_from_seed, matches=synth_matches,
                                  role_of=roles_of(synth_matches))
        assert json.loads((tmp_path / name / 'GOLDEN.json').read_text(encoding='utf-8')) == got[name]
    assert (got['clean']['extraction_clean'], got['clean']['dirty_paths']) == (True, [])
    assert (got['dirty']['extraction_clean'], got['dirty']['dirty_paths']) == (False, ['c_ept/units.py'])
    assert got['clean']['code_commit'] == got['dirty']['code_commit'] == 'f' * 40 and got['clean']['extraction_modules'] == EXTRACTION_PATHS
    assert got['clean']['files'] == got['dirty']['files']
    text = (tmp_path / 'dirty' / 'GOLDEN.json').read_text(encoding='utf-8')
    assert not [k for _, k, _ in synth_matches if k in text]                    # paths only: no key got in with the state


# ---- E6: no Strata server during an S1 run (Ruling 33) ------------------------------------------------------------------------------
def test_e6_refuses_to_start_while_strata_runs(tmp_path, monkeypatch, fake_processes, capsys):
    fake_processes([synth.STRATA_SERVER, synth.STRATA_BRIDGE])

    def started(*args, **kwargs):             # if the refusal fails, the run would extract the real cached matches and register itself
        raise AssertionError('S1 started although the Strata server runs')
    monkeypatch.setattr(run_s1, 'run_s1', started)
    rc = run_s1.main(['--macros', 'FIT', '--runs-root', str(tmp_path / 'runs'), '--catalog-db', str(tmp_path / 'catalog.sqlite')])
    err = capsys.readouterr().err
    assert not (tmp_path / 'catalog.sqlite').exists()
    assert rc != 0 and 'E6' in err and 'Strata' in err and 'does not stop' in err
    assert f'pid {synth.STRATA_SERVER[0]}' in err and 'strata_worker_mcp' not in err        # it names the server it found, not the bridge
    assert not (tmp_path / 'runs').exists()                                         # no run folder, nothing extracted, nobody killed


def test_the_library_entry_points_refuse_while_the_strata_server_runs(tmp_path, synth_matches, fake_processes):
    """run_s1, build_store and golden each stop with StrataRunning before anything is written or read; a caller that does not go
    through main() is protected as well."""
    fake_processes([synth.STRATA_SERVER])
    with pytest.raises(run_s1.StrataRunning, match=r'E6.*does not stop.*pid 4101'):
        s1_run(synth_matches, tmp_path / 'run', workers=1, records=records_forbidden)
    assert not (tmp_path / 'run').exists()
    with pytest.raises(run_s1.StrataRunning, match='E6'):
        run_s1.build_store(synth_matches, tmp_path / 's1', tmp_path / 'blocks', workers=1, records=records_forbidden,
                           role_of=roles_of(synth_matches))
    assert not (tmp_path / 's1').exists() and not (tmp_path / 'blocks').exists()
    with pytest.raises(run_s1.StrataRunning, match='E6'):
        run_s1.golden(5, tmp_path / 'g', records=records_forbidden, matches=synth_matches,
                      role_of=roles_of(synth_matches))
    assert not (tmp_path / 'g').exists()


def test_the_guard_comes_after_the_argument_checks_and_names_no_command_line(tmp_path, synth_matches, fake_processes):
    fake_processes([synth.STRATA_SERVER])
    with pytest.raises(ValueError, match='no matches'):                              # a bad call is a bad call, whatever runs
        run_s1.run_s1([], tmp_path / 'run', workers=1, records=records_forbidden)
    with pytest.raises(run_s1.StrataRunning) as raised:
        s1_run(synth_matches, tmp_path / 'run', workers=1, records=records_forbidden)
    message = str(raised.value)
    assert 'python.exe (pid 4101)' in message                                        # the name and the pid ...
    assert not [t for t in synth.STRATA_SERVER[2] if t != 'strata' and t in message]      # ... and no part of the command line


def test_the_idle_bridges_do_not_stop_a_run_and_the_manifest_says_what_was_seen(tmp_path, synth_matches, fake_processes):
    fake_processes([synth.STRATA_BRIDGE, (4106, 'python.exe', synth.STRATA_BRIDGE[2]), synth.UNRELATED_PYTHON])
    run = tmp_path / '20261020_c_ept_s1_1'
    s1_run(synth_matches, run, workers=1, block_matches=7, records=synth.records_from_seed)
    man = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    assert man['strata_running'] is False and man['strata_detail']['matched'] == [] and man['strata_detail']['bridges_ignored'] == 2
    assert man['s1_result']['complete'] is True


def test_without_psutil_the_guard_warns_and_lets_the_run_go_on(monkeypatch):
    monkeypatch.setattr(runlog, 'psutil', None)
    with pytest.warns(RuntimeWarning, match='psutil'):
        assert run_s1.require_no_strata() is None


# ---- command line -----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('macros', [['TEST'], ['FIT', 'EXTERNAL'], ['NOPE']])
def test_cli_refuses_test_external_and_unknown_macros(tmp_path, macros, capsys, monkeypatch):
    def started(*args, **kwargs):             # if the refusal fails, the run would start here instead of asserting
        raise AssertionError('S1 started with a refused macro role')
    monkeypatch.setattr(run_s1, 'run_s1', started)
    rc = run_s1.main(['--macros', *macros, '--runs-root', str(tmp_path / 'runs'), '--catalog-db', str(tmp_path / 'catalog.sqlite')])
    assert rc != 0 and macros[-1] in capsys.readouterr().err and not (tmp_path / 'runs').exists()
    assert not (tmp_path / 'catalog.sqlite').exists()


def test_cli_needs_macros_or_golden_with_out(tmp_path, monkeypatch):
    def started(*args, **kwargs):             # if a refusal fails, the run or the golden lock starts here instead of asserting
        raise AssertionError('S1 or the golden lock started on a refused command line')
    monkeypatch.setattr(run_s1, 'run_s1', started)
    monkeypatch.setattr(run_s1, 'golden', started)
    tmp_args = ['--runs-root', str(tmp_path / 'runs'), '--catalog-db', str(tmp_path / 'catalog.sqlite')]
    assert run_s1.main(tmp_args) != 0
    assert run_s1.main(['--golden', '3', *tmp_args]) != 0
    assert run_s1.main(['--golden', '3', '--out', str(tmp_path / 'g'), '--macros', 'FIT', *tmp_args]) != 0
    assert not (tmp_path / 'runs').exists() and not (tmp_path / 'g').exists()
    assert not (tmp_path / 'catalog.sqlite').exists()


# ---- role verification (pre-S2 guard 3): the roles of the store are recomputed, never taken from the caller -----------------------
# The macro role of a match comes from the cohort split alone. The runner recomputes it for every listed (patch, key) before it extracts
# anything, and refuses a list whose role differs or that names a match twice. The synthetic keys of these tests are not in the split,
# so ``roles_of`` stands in for ``cohort.role_of`` (the way ``records`` stands in for the reader); the real split has its own test.
def test_the_role_function_is_a_parameter_that_defaults_to_the_cohort_split():
    for fn in (run_s1.verify_roles, run_s1.build_store, run_s1.run_s1, run_s1.golden):
        assert inspect.signature(fn).parameters['role_of'].default is cohort.role_of, fn.__name__
    assert issubclass(run_s1.RoleMismatch, run_s1.MatchListError) and issubclass(run_s1.DuplicateMatch, run_s1.MatchListError)
    assert issubclass(run_s1.MatchListError, ValueError)


def test_verify_roles_asks_the_split_once_for_every_listed_match(synth_matches):
    asked, inner = [], roles_of(synth_matches)

    def spy(key, patch):
        asked.append((patch, key))
        return inner(key, patch)
    assert run_s1.verify_roles(synth_matches, spy) is None
    assert asked == [(patch, key) for patch, key, _ in synth_matches]


def test_a_role_that_differs_from_the_split_is_refused_before_anything_is_written(tmp_path, synth_matches, fake_processes):
    fake_processes([synth.STRATA_SERVER])                      # a bad list is refused whatever runs, like the other argument checks
    one, three = [synth_matches[7][1]], [synth_matches[2][1], synth_matches[11][1], synth_matches[29][1]]
    for flip, n, first in ((one, 1, 7), (three, 3, 2)):
        role_of = roles_of(synth_matches, flip=flip)
        with pytest.raises(run_s1.RoleMismatch) as e:
            run_s1.run_s1(synth_matches, tmp_path / 'run', workers=1, records=records_forbidden, role_of=role_of)
        text = str(e.value)
        assert isinstance(e.value, ValueError) and f'{n} of the 30' in text and f'position {first} (0-based)' in text
        assert not [k for _, k, _ in synth_matches if k in text]                       # the message holds no match key
        assert not (tmp_path / 'run').exists()
        with pytest.raises(run_s1.RoleMismatch, match=f'{n} of the 30'):
            run_s1.build_store(synth_matches, tmp_path / 's1', tmp_path / 'blocks', workers=1, records=records_forbidden, role_of=role_of)
        assert not (tmp_path / 's1').exists() and not (tmp_path / 'blocks').exists()
    with pytest.raises(run_s1.RoleMismatch, match='1 of the 5'):           # the golden lock checks the 5 matches it selects
        run_s1.golden(5, tmp_path / 'g', records=records_forbidden, matches=synth_matches,
                      role_of=roles_of(synth_matches, flip=[synth_matches[2][1]]))
    assert not (tmp_path / 'g').exists()


def test_the_message_says_which_roles_disagree_without_naming_a_match(synth_matches):
    flip = {synth_matches[0][1], synth_matches[5][1], synth_matches[1][1]}               # FIT, FIT (index 5 has FIT again) and STOP
    with pytest.raises(run_s1.RoleMismatch) as e:
        run_s1.verify_roles(synth_matches, roles_of(synth_matches, flip=flip))
    text = str(e.value)
    assert '3 of the 30' in text and 'FIT listed, STOP in the split: 2' in text and 'STOP listed, DEV in the split: 1' in text
    assert 'position 0 (0-based)' in text and 'patch 15.18' in text and not any(k in text for _, k, _ in synth_matches)


def test_a_match_listed_twice_is_refused_before_anything_is_written(tmp_path, synth_matches):
    same_entry = synth_matches[3]
    other_macro = (*synth_matches[3][:2], 'DEV' if synth_matches[3][2] != 'DEV' else 'FIT')
    for repeat, position in ((same_entry, 30), (other_macro, 30)):         # the second occurrence is the repeat; repeats are found first
        matches = [*synth_matches, repeat]
        with pytest.raises(run_s1.DuplicateMatch) as e:
            run_s1.run_s1(matches, tmp_path / 'run', workers=1, records=records_forbidden, role_of=roles_of(matches))
        text = str(e.value)
        assert isinstance(e.value, ValueError) and '1 of the 31' in text and f'position {position} (0-based)' in text
        assert not [k for _, k, _ in matches if k in text] and not (tmp_path / 'run').exists()
        with pytest.raises(run_s1.DuplicateMatch):
            run_s1.build_store(matches, tmp_path / 's1', tmp_path / 'blocks', workers=1, records=records_forbidden,
                               role_of=roles_of(matches))
        assert not (tmp_path / 's1').exists() and not (tmp_path / 'blocks').exists()
    early = [synth_matches[0], synth_matches[1], synth_matches[0], synth_matches[1], synth_matches[2]]      # two repeats, the first at 2
    with pytest.raises(run_s1.DuplicateMatch, match=r'2 of the 5.*position 2 \(0-based\)'):
        run_s1.verify_roles(early, roles_of(early))
    with pytest.raises(run_s1.DuplicateMatch):
        run_s1.golden(3, tmp_path / 'g', records=records_forbidden, matches=early, role_of=roles_of(early))
    assert not (tmp_path / 'g').exists()


def test_a_key_in_two_patches_is_two_matches_not_a_repeat():
    matches = [('15.18', 'SYN_0001', 'FIT'), ('15.19', 'SYN_0001', 'FIT'), ('15.19', 'SYN_0002', 'FIT')]
    assert run_s1.verify_roles(matches, roles_of(matches)) is None


def test_the_default_asks_the_real_cohort_split():
    """The real ``cohort.role_of`` on real cache keys (computed here, no key is written into the repository): the roles it gives pass,
    and every one of them is refused once the list says another macro. Every assert below compares booleans that were computed before
    it: pytest writes out the operands of a failed assert (``len(matches)``, ``verify_roles(matches)``, ``in text``), and a list of
    matches or an error text that holds real keys would reach the test report. So a failure prints no key."""
    matches = [(patch, key, cohort.role_of(key, patch)[1]) for patch in ('15.18', '15.19') for key in cohort.cache_keys(patch)[:40]]
    n_ok = len(matches) == 80
    passes = run_s1.verify_roles(matches) is None          # a role that differs raises RoleMismatch instead: its text holds counts only
    assert n_ok and passes
    other = {macro: 'AUDIT' if macro != 'AUDIT' else 'FIT' for macro in C.ROLES}
    wrong = [(patch, key, other[macro]) for patch, key, macro in matches]
    with pytest.raises(run_s1.RoleMismatch) as e:
        run_s1.verify_roles(wrong)
    text = str(e.value)
    leaked = any(key in text for _, key, _ in matches)
    counts_all = '80 of the 80' in text
    assert counts_all and not leaked
    one = [*matches[:17], (matches[17][0], matches[17][1], other[matches[17][2]]), *matches[18:]]
    with pytest.raises(run_s1.RoleMismatch) as e_one:
        run_s1.verify_roles(one)
    names_position_17 = re.search(r'1 of the 80.*position 17 \(0-based\)', str(e_one.value)) is not None
    assert names_position_17
    # why the synthetic tests inject their own function: the real split gives a synthetic key some role of its own, not the one a test
    # list names, so the unmodified default refuses such a list (as it must)
    synthetic = [('15.18', f'SYN_{i:04d}', MACROS[i % 5]) for i in range(30)]
    with pytest.raises(run_s1.RoleMismatch):
        run_s1.verify_roles(synthetic)
