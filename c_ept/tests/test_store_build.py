r"""Tests of the S2 view store builder (``c_ept.store_build``) and its command line (``c_ept.run_s2``) on a synthetic S1 store.

No raw file is read. The S1 store (30 matches) is made by ``run_s1.build_store`` from ``synth.records_from_seed``; the builds take
``records=synth.records_from_seed`` (a synthetic key is no raw file). The terrain is the open 75 x 75 grid of ``conftest.py`` (the all-pairs
table is built once per session, R13-j) with a hand-made TERRAIN.json, because R13-m wants provenance and ``terrain.build_terrain`` would
cost a second all-pairs table. Every store, run folder and catalog lives under pytest's temporary folders; ``D:\LOL_Data\runs`` and the
default run index are never touched (S2-6): the command-line tests stub the heavy entry points and pass a temporary ``--runs-root`` and
``--catalog-db``. Module-scoped fixtures that reach the E6 check build inside ``conftest.empty_process_list()``. Run with ``--tb=short``.
The ``records_*`` functions at module level are the ``records`` argument of the builds with a pool: worker processes import them by name.
"""
import ast
import concurrent.futures
import dataclasses
import gc
import hashlib
import inspect
import json
import os
import pickle
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import pytest

from c_ept import config as C
from c_ept import reader, run_s1, run_s2, runlog, store_build, store_core, terrain, tokens
from c_ept.tests import synth
from c_ept.tests.conftest import empty_process_list

REPO = Path(C.__file__).parents[1]
MACROS = synth.S1_MACROS
N_MATCHES = 30
ALL = list(range(N_MATCHES))
VIEWS_COLUMNS = ('src', 'src_index', 'match', 'ref', 't0', 'frame_k')
VIEWS_DTYPES = ('i1', 'i4', 'i4', 'i8', 'i8', 'i2')
S1_READ_BY_S2 = {'matches/key', 'matches/patch', 'matches/role', 'matches/row_off', 'matches/event_off', 'frame_rows/k', 'frame_rows/t0',
                 'frame_rows/match', 'event_rows/kind', 'event_rows/ref', 'event_rows/frame_k', 'event_rows/t0'}


# ---- records of the builds (module level: workers import them) ----------------------------------------------------------------------
def records_forbidden(key, patch):
    raise AssertionError('records was called: this call may read nothing')


def records_leaky(key, patch):
    """An unexpected error whose text holds the match key (the way an OSError names its file); the match is S1 row 7."""
    if key == 'SYN_0007':
        raise OSError(f'cannot read {key}.json')
    return synth.records_from_seed(key, patch)


def records_leaky_id(key, patch):
    """An unexpected error that names the number inside the key (the game id) and not the whole key, the way a message quotes a value that
    it read from the file; the match is S1 row 7."""
    if key == 'SYN_0007':
        raise OSError(f'game {key.split("_")[1]} has no participants')
    return synth.records_from_seed(key, patch)


def records_leaky_both(key, patch):
    """An error whose text holds the whole key once and the game id on its own once more."""
    if key == 'SYN_0007':
        raise OSError(f'cannot read {key}.json: game {key.split("_")[1]} is damaged')
    return synth.records_from_seed(key, patch)


def records_bad_format(key, patch):
    if key == 'SYN_0007':
        raise reader.MatchFormatError(key, patch, 'a clock that breaks what every real match shows')
    return synth.records_from_seed(key, patch)


def records_rejecting(key, patch):
    if key == 'SYN_0007':
        raise reader.MatchRejected('missing_file')
    return synth.records_from_seed(key, patch)


def records_other_match(key, patch):
    return synth.records_from_seed('SYN_0099' if key == 'SYN_0003' else key, patch)


def records_shifted_frames(key, patch):
    """The record of S1 match 3 with every regular frame after the first 1 ms early: S1 stored other frame times than the files give now."""
    rec = synth.records_from_seed(key, patch)
    if key == 'SYN_0003':
        ts = np.array(rec.frame_ts)
        ts[1:] -= 1
        rec = dataclasses.replace(rec, frame_ts=ts)
    return rec


def records_slow_first_block(key, patch):
    """The matches 0-4 (block 0 of a plan in blocks of 5) take longer than the others: a pool finishes block 1 first."""
    if key < 'SYN_0005':
        time.sleep(0.4)
    return synth.records_from_seed(key, patch)


# ---- fixtures and helpers ---------------------------------------------------------------------------------------------------------
def synth_matches():
    """The 30 synthetic matches (patch, key, macro) of the S1 tests: the patches 15.18-15.21 and the five macros in turn."""
    return [(C.PATCHES[i % 4], f'SYN_{i:04d}', MACROS[i % 5]) for i in range(N_MATCHES)]


@pytest.fixture(scope='module')
def s1_small(tmp_path_factory):
    """The run folder of a synthetic S1 run of 30 matches (rows 0, 5, 10, ... are FIT, 1, 6, ... STOP, then DEV, CAL, AUDIT)."""
    run = tmp_path_factory.mktemp('s1_small') / '20261021_c_ept_s1_1'
    matches = synth_matches()
    with empty_process_list():
        run_s1.build_store(matches, run / run_s1.STORE_DIR, run / run_s1.BLOCK_DIR, workers=1, block_matches=7,
                           records=synth.records_from_seed, role_of=synth.roles_of(matches))
    yield run
    gc.collect()
    shutil.rmtree(run.parent, ignore_errors=True)


def current_terrain_parameters():
    """The ``grid`` record and the ``zone_rules`` that ``terrain.build_terrain`` writes into TERRAIN.json, written out here from the
    constants of the terrain module (the zone rules as JSON holds them: a list where the rules have a tuple)."""
    return {'grid': {'cells': terrain.GRID, 'cell_u': int(terrain.CELL), 'histogram_cells': terrain.GRID100,
                     'histogram_u': int(terrain.MAP_LIMIT // terrain.GRID100), 'min_count': terrain.MIN_COUNT},
            'zone_rules': {k: (list(v) if isinstance(v, tuple) else v) for k, v in terrain.ZONE_RULES.items()}}


def write_terrain_json(folder, s1_run, **extra):
    """TERRAIN.json for a folder that holds terrain.npz and geodesic.npy: what ``Terrain.verify``, the parameter guard (final review I1)
    and the S2 record need."""
    s1_files = run_s1.load_s1(s1_run, verify=False).manifest['files']
    info = {'version': 1, 's1_run': Path(s1_run).name, 's1_files_sha256': runlog.sha256_json(s1_files),
            'sha256': {name: store_core.sha(Path(folder) / name) for name in ('terrain.npz', 'geodesic.npy')},
            **current_terrain_parameters(), **runlog.code_state(), **extra}
    (Path(folder) / 'TERRAIN.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return info


@pytest.fixture(scope='module')
def terr(open_dir, s1_small, tmp_path_factory):
    """A terrain folder with provenance: the open grid of the session (copied, so that no test holds a map on the shared files)."""
    out = tmp_path_factory.mktemp('s2_terrain') / 'terrain'
    out.mkdir()
    for name in ('terrain.npz', 'geodesic.npy'):
        shutil.copy2(open_dir / name, out / name)
    write_terrain_json(out, s1_small)
    yield out
    gc.collect()
    shutil.rmtree(out.parent, ignore_errors=True)


def build(s1, terr_dir, matches, run, **kw):
    """``store_build.build_s2`` on the synthetic S1 run: one process and the synthetic records unless the test says otherwise."""
    kw.setdefault('workers', 1)
    kw.setdefault('records', synth.records_from_seed)
    return store_build.build_s2(s1, terr_dir, matches, run, **kw)


@pytest.fixture(scope='module')
def full(s1_small, terr, tmp_path_factory):
    """The S2 store of all 30 matches (one process, blocks of 5): the root ``<run>/s2``."""
    run = tmp_path_factory.mktemp('s2_full') / '20261021_c_ept_s2_1'
    with empty_process_list():
        root = build(s1_small, terr, ALL, run, workers=1, block_matches=5)
    yield root
    gc.collect()
    shutil.rmtree(run.parent, ignore_errors=True)


def load(root):
    return store_core.Store(root)


def store_hashes(root):
    """{array file: SHA-256}; STORE.json (it records the plan) is left out (S2-3, R13-c)."""
    s = store_core.Store(root)
    return s.hashes(s.stored_files())


def s1_store(s1_run):
    return run_s1.load_s1(s1_run, verify=False)


def expected_arrays(n):
    """{array name: (shape, dtype)} of an S2 store with n views, from the views table of the brief and ``tokens.VIEW_SPEC``."""
    out = {f'views/{c}': ((n,), np.dtype(d)) for c, d in zip(VIEWS_COLUMNS, VIEWS_DTYPES)}
    out.update({f'x/{k}': ((n, *shape), np.dtype(dtype)) for k, (shape, dtype) in tokens.VIEW_SPEC.items()})
    return out


def snapshot(folder):
    """{relative path: SHA-256} of every file under ``folder``: two equal snapshots mean that nothing was written."""
    return {p.relative_to(folder).as_posix(): store_core.sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def run_json(run):
    return json.loads((Path(run) / 'RUN.json').read_text(encoding='utf-8'))


def text_files(folder):
    """The JSON files under ``folder`` (the arrays are binary and hold no text)."""
    return sorted(Path(folder).rglob('*.json'))


def oracle_table(s1, rows):
    """The views table by plain loops over the S1 arrays (R13-d): per match the frame-clock rows, then the event rows as stored."""
    cols = {c: [] for c in VIEWS_COLUMNS}
    row_off, event_off = np.asarray(s1.array('matches/row_off')), np.asarray(s1.array('matches/event_off'))
    fk, ft0 = np.asarray(s1.array('frame_rows/k')), np.asarray(s1.array('frame_rows/t0'))
    kind, eref = np.asarray(s1.array('event_rows/kind')), np.asarray(s1.array('event_rows/ref'))
    efk, et0 = np.asarray(s1.array('event_rows/frame_k')), np.asarray(s1.array('event_rows/t0'))
    for m in rows:
        for r in range(row_off[m], row_off[m + 1]):
            for c, v in zip(VIEWS_COLUMNS, (0, r, m, ft0[r], ft0[r], fk[r])):
                cols[c].append(v)
        for e in range(event_off[m], event_off[m + 1]):
            for c, v in zip(VIEWS_COLUMNS, (kind[e], e, m, eref[e], et0[e], efk[e])):
                cols[c].append(v)
    return {c: np.array(cols[c], dtype=np.dtype(d)) for c, d in zip(VIEWS_COLUMNS, VIEWS_DTYPES)}


def match_of(s1, m):
    """(key, patch) of the S1 match row m (synthetic keys)."""
    return s1.array('matches/key')[m].decode('ascii'), C.PATCHES[int(s1.array('matches/patch')[m])]


def same_view(row, view):
    """True when every array of the view is bit-identical to the stored row (dtype, shape and bytes; NaN payloads included)."""
    return set(row) == set(view) and all(row[k].dtype == view[k].dtype and row[k].shape == view[k].shape
                                         and row[k].tobytes() == view[k].tobytes() for k in view)


def row_of(s, i):
    return {k: np.asarray(s.array(f'x/{k}')[i]) for k in tokens.VIEW_SPEC}


def write_mini_s1(run, rows):
    """A complete hand-made S1 store under ``run/s1`` with only the arrays that S2 reads (the dtypes of ``run_s1.SPEC``).

    ``rows``: dicts ``key``, ``patch``, ``role``, ``frames`` ([(k, t0), ...]) and ``events`` ([(kind, ref, frame_k, t0), ...]).
    """
    spec, root = run_s1.SPEC, Path(run) / run_s1.STORE_DIR
    arrays = {
        'matches/key': np.array([r['key'].encode('ascii') for r in rows], dtype=spec['matches/key'][0]),
        'matches/patch': np.array([C.PATCHES.index(r['patch']) for r in rows], dtype='i1'),
        'matches/role': np.array([C.ROLES.index(r['role']) for r in rows], dtype='i1'),
        'matches/row_off': np.concatenate([[0], np.cumsum([len(r['frames']) for r in rows])]).astype('i8'),
        'matches/event_off': np.concatenate([[0], np.cumsum([len(r['events']) for r in rows])]).astype('i8'),
        'frame_rows/k': np.array([f[0] for r in rows for f in r['frames']], dtype='i2'),
        'frame_rows/t0': np.array([f[1] for r in rows for f in r['frames']], dtype='i8'),
        'frame_rows/match': np.array([i for i, r in enumerate(rows) for _ in r['frames']], dtype='i4'),
        'event_rows/kind': np.array([e[0] for r in rows for e in r['events']], dtype='i1'),
        'event_rows/ref': np.array([e[1] for r in rows for e in r['events']], dtype='i8'),
        'event_rows/frame_k': np.array([e[2] for r in rows for e in r['events']], dtype='i2'),
        'event_rows/t0': np.array([e[3] for r in rows for e in r['events']], dtype='i8'),
    }
    s = store_core.Store(root)
    for name, a in arrays.items():
        p = s.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, a)
    s.finalize()
    return Path(run)


def mini_row(role='FIT', patch='15.18', key='MINI_0', frames=((2, 120_300),), events=((1, 130_000, 2, 120_300),)):
    return dict(key=key, patch=patch, role=role, frames=list(frames), events=list(events))


def terrain_copy(src, dst, mutate=None):
    """A copy of a terrain folder with provenance; ``mutate(info)`` edits TERRAIN.json on the way."""
    dst = Path(dst)
    dst.mkdir(parents=True)
    for name in ('terrain.npz', 'geodesic.npy'):
        shutil.copy2(Path(src) / name, dst / name)
    info = json.loads((Path(src) / 'TERRAIN.json').read_text(encoding='utf-8'))
    if mutate is not None:
        mutate(info)
    (dst / 'TERRAIN.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return dst


def damage_block(run, s1_run, matches, block_matches, block):
    """Fill the rows of one block, in every array of the S2 store under ``run``, with a sentinel: what a power cut that left a block half
    written would leave. A resume that skips the block (it is not marked) then cannot give the bytes of an uninterrupted build. Returns
    the row range (r0, r1)."""
    s1, lo = s1_store(s1_run), block * block_matches
    r0 = len(store_build.view_table(s1, matches[:lo])['src']) if lo else 0
    r1 = len(store_build.view_table(s1, matches[:lo + block_matches])['src'])
    assert r1 > r0
    s = store_core.Store(Path(run) / 's2')
    for name in s.manifest['arrays']:
        a = np.load(s.path(name), mmap_mode='r+')
        a[r0:r1] = 1
        a.flush()
        del a
    return r0, r1


def git_as(letter, dirty):
    """A stand-in for ``runlog._git``: a repository at the commit ``letter * 40`` with or without changes in c_ept/."""
    def git(*args):
        if args[0] == 'rev-parse':
            return letter * 40 + '\n'
        return ' M c_ept/x.py\n' if dirty else ''
    return git


# ---- the three tests of the task brief ----------------------------------------------------------------------------------------------
def test_store_resume_identical(tmp_path, s1_small, terr, full):
    """30 matches: one process in blocks of 5 against two processes in blocks of 10 that are stopped after block 1 and resumed. The rows
    of the unmarked block 1 are damaged before the resume, so that a resume which skips that block cannot match."""
    run = tmp_path / '20261021_c_ept_s2_2'
    with pytest.raises(store_build.StopRequested):
        build(s1_small, terr, ALL, run, workers=2, block_matches=10, stop_after_block=1)
    assert load(run / 's2').done('views') == {0} and not load(run / 's2').manifest['complete']      # block 1 was written, not marked
    r0, r1 = damage_block(run, s1_small, ALL, 10, 1)
    rows = (np.asarray(load(run / 's2').array('x/summary'))[r0:r1], np.asarray(load(full).array('x/summary'))[r0:r1])
    assert not np.array_equal(*rows)                                                                # the damage is a real difference
    del rows
    resumed = build(s1_small, terr, ALL, run, workers=2, block_matches=10)
    assert load(resumed).done('views') == {0, 1, 2}                                                # every block is marked: none was skipped
    assert store_hashes(full) == store_hashes(resumed)
    n = len(load(full).array('views/src'))
    assert set(store_hashes(full)) == {name + '.npy' for name in expected_arrays(n)}               # the comparison is not vacuous
    man = run_json(run)['s2_result']
    assert man['complete'] is True and man['blocks_resumed'] == 1 and man['workers'] == 2 and man['n_blocks'] == 3
    assert man['peak_private_worker_gib'] > 0                                                      # a pool ran and was seen


def test_views_cover_rows(s1_small, full):
    s1, s = s1_store(s1_small), load(full)
    n_frame, n_event = len(s1.array('frame_rows/k')), len(s1.array('event_rows/ref'))
    assert len(s.array('views/src')) == n_frame + n_event
    src = np.asarray(s.array('views/src'))
    kind = np.asarray(s1.array('event_rows/kind'))
    assert [int((src == c).sum()) for c in (0, 1, 2)] == [n_frame, int((kind == 1).sum()), int((kind == 2).sum())]
    assert n_frame > 0 and (src == 1).any() and (src == 2).any()                                    # all three kinds are present


def test_view_row_equals_direct_build(s1_small, terr, full):
    s1, s = s1_store(s1_small), load(full)
    n = len(s.array('views/src'))
    t, contexts = terrain.Terrain(terr), {}
    try:
        for i in [17, 0, n - 1, *range(5, n, 83)]:
            m = int(s.array('views/match')[i])
            if m not in contexts:
                contexts[m] = tokens.MatchContext(synth.records_from_seed(*match_of(s1, m)), t)
            ref, frame_k = int(s.array('views/ref')[i]), int(s.array('views/frame_k')[i])
            assert same_view(row_of(s, i), tokens.build_view(contexts[m], ref, frame_k)), f'view {i}'
    finally:
        del contexts, t
        gc.collect()


# ---- the store layout (R13-c, R13-r, R13-o) -------------------------------------------------------------------------------------------
def test_store_layout_follows_the_brief_and_view_spec(full):
    s = load(full)
    n = len(s.array('views/src'))
    want = expected_arrays(n)
    assert sorted(s.manifest['arrays']) == sorted(want)
    for name, (shape, dtype) in want.items():
        assert s.manifest['arrays'][name] == {'shape': list(shape), 'dtype': dtype.str}              # plain form (R13-r)
        a = s.array(name)
        assert a.shape == shape and a.dtype == dtype
    assert [s.array(f'views/{c}').dtype.str for c in VIEWS_COLUMNS] == [np.dtype(d).str for d in VIEWS_DTYPES]
    assert sum(np.dtype(d).itemsize for d in VIEWS_DTYPES) == 27
    assert s.manifest['complete'] is True and not [p for p in Path(full).rglob('*') if p.is_file() and p.suffix not in ('.npy', '.json')]
    assert sorted(p.name for p in Path(full).iterdir()) == ['STORE.json', 'views', 'x']             # no log or block file in the store root


def test_views_table_order_and_columns_follow_s1(s1_small, full):
    """R13-d: match, then its frame-clock rows, then its event rows in the order S1 stored them (unit, kind): M3 and M4 interleave."""
    s1, s = s1_store(s1_small), load(full)
    want = oracle_table(s1, ALL)
    for c in VIEWS_COLUMNS:
        got = np.asarray(s.array(f'views/{c}'))
        assert got.dtype == want[c].dtype and np.array_equal(got, want[c]), c
    src, match = np.asarray(s.array('views/src')), np.asarray(s.array('views/match'))
    assert np.array_equal(np.unique(match), np.arange(N_MATCHES)) and bool(np.all(np.diff(match) >= 0))   # matches in the order of the list
    interleaved = 0
    for m in ALL:
        part = src[match == m]
        n_frame = int(np.diff(np.asarray(s1.array('matches/row_off'))[[m, m + 1]])[0])
        assert not part[:n_frame].any() and part[n_frame:].all()                                    # frame-clock rows (0) first, then 1 / 2
        events = part[n_frame:]
        interleaved += int(bool(np.any(events[:-1] > events[1:])))                                   # a 2 before a later 1: not regrouped
    assert interleaved > 0                                                                           # the check on the order is not vacuous


def test_frame_k_of_the_views_is_the_frame_of_the_clock_that_s1_stored(s1_small, full):
    """R13-r: a frame-clock row is (frame_ts[k], k); an event row has the last regular frame strictly before ref."""
    s1, s = s1_store(s1_small), load(full)
    src, match = np.asarray(s.array('views/src')), np.asarray(s.array('views/match'))
    ref, t0, k = (np.asarray(s.array(f'views/{c}')).astype(np.int64) for c in ('ref', 't0', 'frame_k'))
    seen = {0: 0, 1: 0, 2: 0}
    for m in ALL:
        ts = synth.records_from_seed(*match_of(s1, m)).frame_ts
        sel = np.flatnonzero(match == m)
        for i in sel:
            seen[int(src[i])] += 1
            assert t0[i] == ts[k[i]]
            if src[i] == 0:
                assert ref[i] == ts[k[i]]                                                           # the frame is the present
            else:
                assert ts[k[i]] < ref[i] and (k[i] + 1 == len(ts) or ts[k[i] + 1] >= ref[i])         # the last frame strictly before ref
                assert k[i] == np.searchsorted(ts, ref[i], side='left') - 1             # S1's rule (event_rows.py): 'left', not 'right'
    assert all(seen.values())


def test_an_event_row_on_the_ms_of_a_frame_keeps_the_frame_before_it(terr, tmp_path):
    """S1 takes ``searchsorted(frame_ts, ref, 'left') - 1`` for an event row, so a ref that falls exactly on a frame's ms has the frame
    BEFORE that frame (a helper that returns the frame itself there would give another view). S2 passes S1's frame_k on unchanged."""
    ts = synth.records_from_seed('MINI_0', '15.18').frame_ts
    t2, t3 = int(ts[2]), int(ts[3])
    s1_run = write_mini_s1(tmp_path / '20261021_c_ept_s1_9', [mini_row(key='MINI_0', frames=[(3, t3)], events=[(1, t3, 2, t2)])])
    root = build(s1_run, terr, [0], tmp_path / 'r')
    s = load(root)
    assert np.asarray(s.array('views/src')).tolist() == [0, 1]
    assert [int(np.asarray(s.array(f'views/{c}'))[1]) for c in ('ref', 't0', 'frame_k')] == [t3, t2, 2]     # the event row: frame 2
    assert [int(np.asarray(s.array(f'views/{c}'))[0]) for c in ('ref', 't0', 'frame_k')] == [t3, t3, 3]     # the frame-clock row: frame 3
    t, ctx = terrain.Terrain(terr), None
    try:
        ctx = tokens.MatchContext(synth.records_from_seed('MINI_0', '15.18'), t)
        assert same_view(row_of(s, 1), tokens.build_view(ctx, t3, 2))
        assert same_view(row_of(s, 0), tokens.build_view(ctx, t3, 3))
        assert not same_view(row_of(s, 1), tokens.build_view(ctx, t3, 3))                                   # the two frames give two views
    finally:
        del ctx, t
        gc.collect()
    age = np.asarray(s.array('x/summary'))[:, tokens.SUMMARY.index('frame_age_s')]
    assert age.tolist() == [0.0, (t3 - t2) / 1000.0]                                                         # frame age 0 s against 60 s


def test_a_subset_in_any_order_is_taken_as_listed(s1_small, terr, tmp_path):
    rows = [7, 2, 3]
    s1 = s1_store(s1_small)
    root = build(s1_small, terr, rows, tmp_path / 'r', block_matches=2)
    s = load(root)
    want = oracle_table(s1, rows)
    for c in VIEWS_COLUMNS:
        assert np.array_equal(np.asarray(s.array(f'views/{c}')), want[c]), c
    match = np.asarray(s.array('views/match'))
    assert [int(match[0]), int(match[-1])] == [7, 3] and list(dict.fromkeys(match.tolist())) == rows
    table = store_build.view_table(s1, rows)
    assert {c: a.dtype.str for c, a in table.items()} == {c: np.dtype(d).str for c, d in zip(VIEWS_COLUMNS, VIEWS_DTYPES)}
    assert all(np.array_equal(table[c], want[c]) for c in VIEWS_COLUMNS)


def test_view_table_by_hand_on_a_two_match_store(tmp_path):
    """R13-d by hand: match 0 has two frame-clock rows and three event rows (M3, M4 of one unit, then M3 of the next), match 1 one of each.
    The list [1, 0] puts match 1 first; inside a match the frame-clock rows come first and the event rows keep their stored order."""
    s1_run = write_mini_s1(tmp_path / '20261021_c_ept_s1_9', [
        mini_row(key='MINI_0', frames=[(2, 120_300), (3, 180_300)],
                 events=[(1, 130_000, 2, 120_300), (2, 150_000, 2, 120_300), (1, 185_000, 3, 180_300)]),
        mini_row(key='MINI_1', frames=[(2, 121_000)], events=[(1, 125_000, 2, 121_000)])])
    s1 = run_s1.load_s1(s1_run)
    got = store_build.view_table(s1, [1, 0])
    want = {'src': [0, 1, 0, 0, 1, 2, 1], 'src_index': [2, 3, 0, 1, 0, 1, 2], 'match': [1, 1, 0, 0, 0, 0, 0],
            'ref': [121_000, 125_000, 120_300, 180_300, 130_000, 150_000, 185_000],
            't0': [121_000, 121_000, 120_300, 180_300, 120_300, 120_300, 180_300], 'frame_k': [2, 2, 2, 3, 2, 2, 3]}
    assert {c: a.tolist() for c, a in got.items()} == want
    assert {c: a.dtype.str for c, a in got.items()} == {c: np.dtype(d).str for c, d in zip(VIEWS_COLUMNS, VIEWS_DTYPES)}
    assert store_build.view_table(s1, [0])['src'].tolist() == [0, 0, 1, 2, 1]                    # the stored order, M3 and M4 not regrouped


def test_view_table_refuses_matches_that_the_s1_store_does_not_hold(s1_small):
    s1 = s1_store(s1_small)
    for rows in ([0, N_MATCHES], [-1], [0, 10 ** 9]):
        with pytest.raises(ValueError, match='S1 store'):
            store_build.view_table(s1, rows)


def test_masked_views_are_not_stored_and_each_onset_row_has_its_frame_clock_view(s1_small, full):
    """R13-e / D1: no view at t0 for M3 or M4; the input of an onset is the frame-clock view of the row ``onsets/row``."""
    s1, s = s1_store(s1_small), load(full)
    src, idx = np.asarray(s.array('views/src')), np.asarray(s.array('views/src_index'))
    n_event = len(s1.array('event_rows/ref'))
    assert int((src > 0).sum()) == n_event and set(np.unique(src).tolist()) == {0, 1, 2}             # open views only: one per event row
    first = {int(r): i for i, r in zip(np.flatnonzero(src == 0), idx[src == 0])}                      # S1 frame-clock row -> view
    assert len(first) == int((src == 0).sum())                                                       # one view per row
    onset_rows = np.asarray(s1.array('onsets/row'))
    assert onset_rows.size > 0 and all(int(r) in first for r in onset_rows)                           # every onset finds its view
    for r in onset_rows[:50]:
        v = first[int(r)]
        assert src[v] == 0 and idx[v] == r and np.asarray(s.array('views/t0'))[v] == np.asarray(s1.array('frame_rows/t0'))[r]
    text = run_json(Path(full).parent)['masked_views']
    assert all(word in text for word in ('onsets/row', 'views/src', 'views/src_index'))


def test_bytes_per_view_follow_the_dtypes(full):
    s = load(full)
    man = run_json(Path(full).parent)['s2_result']
    x_bytes = sum(s.array(name).dtype.itemsize * int(np.prod(s.array(name).shape[1:], dtype=np.int64)) for name in s.manifest['arrays']
                  if name.startswith('x/'))
    views_bytes = sum(s.array(name).dtype.itemsize for name in s.manifest['arrays'] if name.startswith('views/'))
    assert man['x_bytes_per_view'] == x_bytes == tokens.bytes_per_view()
    assert man['views_bytes_per_view'] == views_bytes == 27
    assert man['bytes_per_view'] == x_bytes + views_bytes
    files = [p for p in Path(full).rglob('*.npy')]
    assert sum(p.stat().st_size for p in files) >= man['views'] * man['bytes_per_view']


# ---- run record (R13-f, R13-o) ---------------------------------------------------------------------------------------------------------
def test_run_json_records_the_provenance(s1_small, terr, full):
    run = Path(full).parent
    man = run_json(run)
    files = json.loads((s1_small / 's1' / 'STORE.json').read_text(encoding='utf-8'))['files']
    want_hash = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    assert man['s1'] == {'run': s1_small.name, 'files_sha256': want_hash}
    assert man['terrain']['sha256'] == {name: store_core.sha(terr / name) for name in ('TERRAIN.json', 'geodesic.npy', 'terrain.npz')}
    assert man['terrain']['built_from_this_s1'] is True and man['terrain']['s1_files_sha256'] == want_hash
    # the commit is the HEAD of the moment the fixture ran: other commits may land in this worktree, so only its form is checked here
    # (test_run_json_takes_the_git_values_from_runlog_code_state pins the values with a fixed git)
    assert re.fullmatch(r'[0-9a-f]{40}', man['code_commit']) and isinstance(man['code_dirty'], bool)
    end = man['s2_result']['code_state_at_end']
    assert set(end) == {'code_commit', 'code_dirty'} and isinstance(end['code_dirty'], bool)
    assert re.fullmatch(r'[0-9a-f]{40}', end['code_commit'])
    assert man['stage'] == 's2' and man['strata_running'] is False and 'gpu' in man
    plan = runlog.sha256_json({'s1_files_sha256': want_hash, 'matches': ALL})
    assert man['args']['plan_sha256'] == plan and man['args']['n_matches'] == N_MATCHES
    assert (man['args']['workers'], man['args']['block_matches']) == (1, 5)
    stage = load(full).manifest['stages']['views']
    assert (stage['plan_sha256'], stage['block_matches'], stage['n_matches'], stage['n_blocks']) == (plan, 5, N_MATCHES, 6)
    assert stage['terrain_sha256'] == runlog.sha256_json(man['terrain']['sha256']) and re.fullmatch(r'[0-9a-f]{64}', stage['layout_sha256'])
    assert stage['n_views'] == len(load(full).array('views/src'))
    r = man['s2_result']
    s = load(full)
    src = np.asarray(s.array('views/src'))
    assert r['views'] == len(src) and r['views_by_src'] == {'frame_clock': int((src == 0).sum()), 'M3': int((src == 1).sum()),
                                                            'M4': int((src == 2).sum())}
    assert r['complete'] is True and r['matches'] == N_MATCHES and r['wall_s'] > 0 and r['blocks_resumed'] == 0 and r['n_blocks'] == 6
    assert set(r['stage_s']) == {'allocate', 'views', 'finalize'} and all(v >= 0 for v in r['stage_s'].values())
    assert r['stage_s']['views'] <= r['wall_s'] and r['wall_s'] <= sum(r['stage_s'].values()) + 0.3
    assert r['matches_built'] == N_MATCHES and r['seconds_per_match'] > 0
    assert r['calls'] == 1 and r['recovered'] is False and r['warnings'] == [] and man['args']['info'] == {}
    only = r['per_call'][0]
    assert (only['workers'], only['blocks_built'], only['matches_built']) == (1, 6, N_MATCHES) and only['warnings'] == []
    assert (only['peak_private_gib'], only['peak_private_main_gib']) == (r['peak_private_gib'], r['peak_private_main_gib'])
    assert re.fullmatch(r'[0-9a-f]{40}', only['code_commit']) and isinstance(only['code_dirty'], bool)
    store_files = s.stored_files()                       # the arrays: STORE.json records the plan and the timings of the calls, it is not counted
    assert r['store_files'] == len(expected_arrays(1)) == len(store_files) == len(list(Path(full).rglob('*.npy')))
    assert r['store_bytes'] == sum(p.stat().st_size for p in store_files)
    assert r['store_bytes'] == sum(p.stat().st_size for p in Path(full).rglob('*') if p.is_file()) - (Path(full) / 'STORE.json').stat().st_size


def test_run_json_records_the_peak_private_memory(full):
    r = run_json(Path(full).parent)['s2_result']
    assert r['peak_private_gib'] > 0 and r['memory_samples'] >= 1                                  # psutil is installed in this environment
    assert 0 < r['peak_private_main_gib'] <= r['peak_private_gib'] + 0.01
    assert r['peak_private_worker_gib'] is None                                                      # one process: no worker exists


def test_run_json_takes_the_git_values_from_runlog_code_state(s1_small, terr, tmp_path, monkeypatch):
    """R13-f: the commit and the dirty flag are those of ``runlog.code_state`` (git is a stand-in here, so the answer is fixed)."""
    def git(*args):
        return 'a' * 40 + '\n' if args[0] == 'rev-parse' else ' M c_ept/x.py\n'
    monkeypatch.setattr(runlog, '_git', git)
    build(s1_small, terr, [0, 1], tmp_path / 'r')
    man = run_json(tmp_path / 'r')
    assert (man['code_commit'], man['code_dirty']) == ('a' * 40, True)
    assert {k: man[k] for k in ('code_commit', 'code_dirty')} == runlog.code_state() == man['s2_result']['code_state_at_end']


def test_nothing_that_names_a_match_is_written(full):
    run = Path(full).parent
    for p in text_files(run):
        text = p.read_text(encoding='utf-8')
        assert 'SYN_' not in text and 'puuid' not in text and 'riotId' not in text, p.name


def test_the_s1_store_is_left_as_it_was(s1_small, full):
    run_s1.load_s1(s1_small)                                                                        # verifies every file against STORE.json


def test_run_info_is_kept_apart_from_the_computed_args(s1_small, terr, tmp_path):
    """Review Minor 9: the caller's ``info`` is one key of ``args``; a key of it can never overwrite ``plan_sha256`` or ``workers``."""
    info = {'macros': ['FIT'], 'limit': 2, 'plan_sha256': 'not the plan', 'workers': 99, 'n_matches': 0, 'info': 'x'}
    build(s1_small, terr, [0, 1], tmp_path / 'r', info=info)
    args = run_json(tmp_path / 'r')['args']
    assert args['info'] == info and set(args) == {'workers', 'block_matches', 'n_matches', 'plan_sha256', 'info'}
    assert (args['workers'], args['n_matches'], len(args['plan_sha256'])) == (1, 2, 64) and args['plan_sha256'] != info['plan_sha256']
    build(s1_small, terr, [0, 1], tmp_path / 'none')
    assert run_json(tmp_path / 'none')['args']['info'] == {}


def test_the_s1_store_folder_is_accepted_like_its_run_folder(s1_small, terr, tmp_path):
    a = build(s1_small, terr, [0, 1], tmp_path / 'a')
    b = build(s1_small / run_s1.STORE_DIR, terr, [0, 1], tmp_path / 'b')
    assert store_hashes(a) == store_hashes(b) and run_json(tmp_path / 'b')['s1']['run'] == s1_small.name


def test_a_run_inside_the_s1_run_is_refused(s1_small, terr):
    before = snapshot(s1_small)
    for inside in (s1_small / 'later', s1_small / run_s1.STORE_DIR / 'x', s1_small):
        with pytest.raises(ValueError, match='S1 run'):
            build(s1_small, terr, [0], inside, records=records_forbidden)
    assert snapshot(s1_small) == before


# ---- resume and the plan (R13-b, R13-c, R13-n) ------------------------------------------------------------------------------------------
def interrupted(tmp_path, s1, terr_dir, matches=None, block_matches=4, damage=True):
    """A store of 12 matches in blocks of 4 that stopped after the write of block 1 (before its mark): block 0 is marked done. With
    ``damage`` (the default) the rows of the unmarked block 1 are then filled with a sentinel (``damage_block``), so that a resume has to
    build that block again to give the bytes of an uninterrupted build; a resume that skips it leaves the sentinel and is caught."""
    run = tmp_path / '20261021_c_ept_s2_3'
    matches = matches or ALL[:12]
    with pytest.raises(store_build.StopRequested):
        build(s1, terr_dir, matches, run, block_matches=block_matches, stop_after_block=1)
    if damage:
        damage_block(run, s1, matches, block_matches, 1)
    return run


def test_stop_after_block_stops_between_the_write_and_the_mark(s1_small, terr, tmp_path):
    run = interrupted(tmp_path, s1_small, terr, damage=False)
    s = load(run / 's2')
    assert s.done('views') == {0} and not s.manifest['complete']
    assert s.manifest['stages']['views']['n_blocks'] == 3
    s1 = s1_store(s1_small)
    table = store_build.view_table(s1, ALL[:12])
    two_blocks = len(store_build.view_table(s1, ALL[:8])['src'])
    ref = np.asarray(s.array('views/ref'))
    assert np.array_equal(ref[:two_blocks], table['ref'][:two_blocks])                               # blocks 0 and 1 are on disk ...
    assert not ref[two_blocks:].any()                                                                # ... and nothing of block 2
    x = np.asarray(s.array('x/summary'))
    assert x[:two_blocks].any(axis=1).all() and not x[two_blocks:].any()                        # the views of block 1 are written, unmarked
    json.loads((run / 's2' / 'STORE.json').read_text(encoding='utf-8'))
    assert not list(run.rglob('*.tmp'))


def test_resume_refuses_another_plan_before_any_write_and_the_same_plan_completes(s1_small, terr, tmp_path):
    run = interrupted(tmp_path, s1_small, terr)
    before = snapshot(run)
    with pytest.raises(ValueError, match='block_matches'):
        build(s1_small, terr, ALL[:12], run, block_matches=6, records=records_forbidden)
    for other in (ALL[:11], list(reversed(ALL[:12])), ALL[:11] + [12], ALL[:12] + [12]):
        with pytest.raises(ValueError, match='plan_sha256'):
            build(s1_small, terr, other, run, block_matches=4, records=records_forbidden)
    assert snapshot(run) == before                                                                   # nothing was written by a refusal
    done = build(s1_small, terr, ALL[:12], run, block_matches=4, workers=2)                          # same plan, another worker count
    ref = build(s1_small, terr, ALL[:12], tmp_path / 'ref', block_matches=3)                         # and another block size, uninterrupted
    assert store_hashes(done) == store_hashes(ref)                                                   # the damaged block 1 was built again
    assert load(done).manifest['complete'] is True and load(done).done('views') == {0, 1, 2}      # and every block is marked
    assert run_json(run)['s2_result']['blocks_resumed'] == 1
    assert run_json(run)['args']['workers'] == 1 and run_json(run)['s2_result']['workers'] == 2  # the first call's args, this call's result


def test_the_store_is_not_finalized_while_a_block_is_not_marked(s1_small, terr, tmp_path, monkeypatch):
    """Review Minor 1: a loop that ends with a block unmarked (a skipped block) must stop before ``finalize``, not call the store done."""
    original, run = store_core.Store.mark_done, tmp_path / '20261021_c_ept_s2_4'

    def lose_the_mark_of_block_1(self, stage, block, names):
        if block != 1:
            original(self, stage, block, names)
    with monkeypatch.context() as patch:
        patch.setattr(store_core.Store, 'mark_done', lose_the_mark_of_block_1)
        with pytest.raises(RuntimeError, match='not marked'):
            build(s1_small, terr, ALL[:12], run, block_matches=4)
    s = load(run / 's2')
    assert s.done('views') == {0, 2} and not s.manifest['complete'] and not s.manifest['files']     # no hash was taken, nothing is complete
    again = build(s1_small, terr, ALL[:12], run, block_matches=4)                                # the next call builds block 1 and finishes
    assert load(again).done('views') == {0, 1, 2} and load(again).manifest['complete'] is True
    assert store_hashes(again) == store_hashes(build(s1_small, terr, ALL[:12], tmp_path / 'ref', block_matches=4))


def test_rows_are_written_at_fixed_offsets_whatever_the_completion_order(s1_small, terr, tmp_path):
    """R13-c: block 0 is slow, so a pool finishes block 1 first; the stored arrays are those of the in-order build."""
    rows = ALL[:10]
    ref = build(s1_small, terr, rows, tmp_path / 'ref', block_matches=5)
    slow = build(s1_small, terr, rows, tmp_path / 'slow', workers=2, block_matches=5, records=records_slow_first_block)
    assert store_hashes(slow) == store_hashes(ref)


def test_peaks_wall_time_and_matches_are_kept_across_calls(s1_small, terr, tmp_path):
    """Review Minor 2: the first call (two workers, stopped after block 1) is not forgotten by the call that finishes the store (one
    process, so it has no worker of its own): the maxima, the times and the match counts of RUN.json are those of all calls."""
    run = tmp_path / '20261021_c_ept_s2_5'
    with pytest.raises(store_build.StopRequested):
        build(s1_small, terr, ALL, run, workers=2, block_matches=10, stop_after_block=1)
    first = load(run / 's2').manifest['stages']['views']['calls']
    assert len(first) == 1 and first[0]['workers'] == 2 and (first[0]['blocks_built'], first[0]['matches_built']) == (1, 10)
    assert first[0]['peak_private_bytes']['worker'] > 0 and first[0]['views_s'] > 0 and first[0]['memory_samples'] >= 1
    build(s1_small, terr, ALL, run, workers=1, block_matches=10)
    calls = load(run / 's2').manifest['stages']['views']['calls']
    r = run_json(run)['s2_result']
    assert len(calls) == 2 and calls[0] == first[0] and (calls[1]['workers'], calls[1]['peak_private_bytes']['worker']) == (1, 0)
    assert r['calls'] == 2 and [c['workers'] for c in r['per_call']] == [2, 1] and [c['matches_built'] for c in r['per_call']] == [10, 20]
    assert r['matches_built'] == 30 and r['blocks_resumed'] == 1 and r['workers'] == 1 and r['n_blocks'] == 3
    per_call = r['per_call']
    assert r['peak_private_worker_gib'] == per_call[0]['peak_private_worker_gib'] > 0 and per_call[1]['peak_private_worker_gib'] is None
    assert r['peak_private_gib'] == max(c['peak_private_gib'] for c in per_call)
    assert r['peak_private_main_gib'] == max(c['peak_private_main_gib'] for c in per_call)
    assert r['memory_samples'] == sum(c['memory_samples'] for c in per_call)
    assert abs(r['wall_s'] - (sum(c['wall_s'] for c in per_call) + r['stage_s']['finalize'])) <= 0.35      # rounded to 0.1 s each
    assert r['stage_s']['views'] >= per_call[0]['views_s'] and r['stage_s']['allocate'] >= per_call[0]['allocate_s']
    assert abs(r['seconds_per_match'] - sum(c['views_s'] for c in per_call) / 30) < 0.01                    # over both calls, not the last


def test_the_call_record_is_saved_after_every_block(s1_small, terr, tmp_path, monkeypatch):
    """A power cut skips every ``except`` clause, so the record of the call must already be in STORE.json after each finished block."""
    original, seen = store_core.Store.set_stage, []

    def spy(self, stage, **info):
        if 'calls' in info and info['calls']:
            seen.append(info['calls'][-1]['blocks_built'])
        original(self, stage, **info)
    with monkeypatch.context() as patch:
        patch.setattr(store_core.Store, 'set_stage', spy)
        build(s1_small, terr, ALL[:6], tmp_path / 'r', block_matches=2)
    assert seen == [0, 1, 2, 3, 3]                                      # at the start, after each of the 3 blocks, and before finalize


def test_a_call_that_fails_in_its_first_block_leaves_its_record(s1_small, terr, tmp_path):
    run = tmp_path / '20261021_c_ept_s2_10'
    with pytest.raises(store_build.S2WorkerError):
        build(s1_small, terr, ALL[6:12], run, block_matches=5, records=records_leaky)                 # S1 row 7 is in the first block
    s = load(run / 's2')
    (call,) = s.manifest['stages']['views']['calls']
    assert s.done('views') == set() and (call['blocks_built'], call['matches_built']) == (0, 0)
    assert call['views_s'] > 0 and call['memory_samples'] >= 1 and call['peak_private_bytes']['main'] > 0      # what it did is on record
    build(s1_small, terr, ALL[6:12], run, block_matches=5)                                           # the next call finishes the store
    r = run_json(run)['s2_result']
    assert r['calls'] == 2 and [c['matches_built'] for c in r['per_call']] == [0, 6] and r['matches_built'] == 6
    assert r['blocks_resumed'] == 0                                      # the second call built both blocks


def test_a_crash_between_finalize_and_the_run_record_is_repaired_by_the_next_call(s1_small, terr, tmp_path, monkeypatch):
    """Review Minor 3: a complete store whose RUN.json has no ``s2_result`` (power cut after ``finalize``) is not left as it is for good:
    the next call adds the record from what the store holds, reads no match, and a call after that changes nothing."""
    run = tmp_path / '20261021_c_ept_s2_6'

    def cut(*args, **kwargs):
        raise OSError('power cut between finalize and the run record')
    with monkeypatch.context() as patch:
        patch.setattr(runlog, 'amend_manifest', cut)
        with pytest.raises(OSError, match='power cut'):
            build(s1_small, terr, ALL[:6], run, block_matches=3)
    assert load(run / 's2').manifest['complete'] is True and 's2_result' not in run_json(run)
    root = build(s1_small, terr, ALL[:6], run, block_matches=3, records=records_forbidden)
    ref = build(s1_small, terr, ALL[:6], tmp_path / 'ref', block_matches=3)
    r, want = run_json(run)['s2_result'], run_json(tmp_path / 'ref')['s2_result']
    assert root == run / 's2' and r['complete'] is True and r['recovered'] is True and want['recovered'] is False
    for name in ('views', 'views_by_src', 'matches', 'n_blocks', 'x_bytes_per_view', 'views_bytes_per_view', 'bytes_per_view',
                 'store_files', 'store_bytes', 'workers', 'block_matches'):
        assert r[name] == want[name], name
    assert r['matches_built'] == 6 and r['calls'] == 1 and r['wall_s'] > 0 and r['peak_private_gib'] > 0
    assert r['stage_s']['finalize'] is None                                                          # this call did not finalize
    assert load(root).done('views') == {0, 1} and store_hashes(root) == store_hashes(ref)
    before = snapshot(run)
    build(s1_small, terr, ALL[:6], run, block_matches=3, records=records_forbidden)
    assert snapshot(run) == before                                                                   # the repaired run is left alone


def test_store_bytes_counts_the_stored_arrays_only_so_a_repaired_record_equals_a_fresh_one(s1_small, terr, tmp_path, monkeypatch):
    """The cause of a flake (final fix pass A). ``store_bytes`` summed every file under ``s2\\``, STORE.json too, and STORE.json holds the
    times and the memory peaks of the calls; they print with a different number of digits from one build to the next, so two builds of
    one plan differed by a byte or two (the repair test above failed in 2 of 20 runs, the sign of the difference flipping). The figures
    are now those of the stored arrays (``Store.stored_files()``, as the store hashes of S2-3): a function of the plan alone. Here the
    metadata is made to differ by a kilobyte, which no luck of the digits can hide."""
    run, fresh_run = tmp_path / '20261021_c_ept_s2_30', tmp_path / '20261021_c_ept_s2_31'

    def cut(*args, **kwargs):
        raise OSError('power cut between finalize and the run record')
    with monkeypatch.context() as patch:
        patch.setattr(runlog, 'amend_manifest', cut)
        with pytest.raises(OSError, match='power cut'):
            build(s1_small, terr, ALL[:6], run, block_matches=3)
    load(run / 's2').set_stage('views', note='x' * 1000)                      # STORE.json grows by a kilobyte; no array changes
    build(s1_small, terr, ALL[:6], run, block_matches=3, records=records_forbidden)                   # the repair
    build(s1_small, terr, ALL[:6], fresh_run, block_matches=3)
    repaired, fresh = run_json(run)['s2_result'], run_json(fresh_run)['s2_result']
    assert repaired['recovered'] is True and fresh['recovered'] is False
    assert (run / 's2' / 'STORE.json').stat().st_size >= (fresh_run / 's2' / 'STORE.json').stat().st_size + 900     # the metadata differs ...
    assert (repaired['store_files'], repaired['store_bytes']) == (fresh['store_files'], fresh['store_bytes'])        # ... the figures do not
    s = load(run / 's2')
    assert repaired['store_files'] == len(expected_arrays(1)) == len(s.stored_files())
    assert repaired['store_bytes'] == sum(p.stat().st_size for p in s.stored_files())
    assert store_hashes(run / 's2') == store_hashes(fresh_run / 's2')


def test_a_crash_in_the_allocation_loop_does_not_leave_an_array_out_of_the_manifest(s1_small, terr, tmp_path, monkeypatch):
    """Review Minor 4: a cut between the creation of an array file and the save of its STORE.json entry leaves a file that the manifest
    does not list, and ``Store.allocate`` opens such a file without listing it. The build lists it again; the store is the same as an
    uninterrupted one, and its manifest has every array in plain form."""
    run, original, saves = tmp_path / '20261021_c_ept_s2_7', store_core.Store.save, {'n': 0}

    def cut_in_the_third_array(self):
        saves['n'] += 1
        if saves['n'] == 4:                                                       # 1: the stage, 2 and 3: two arrays, 4: the third array
            raise OSError('power cut between an array file and its entry')
        original(self)
    with monkeypatch.context() as patch:
        patch.setattr(store_core.Store, 'save', cut_in_the_third_array)
        with pytest.raises(OSError, match='power cut'):
            build(s1_small, terr, ALL[:6], run, block_matches=3, records=records_forbidden)
    names, on_disk = list(expected_arrays(1)), store_core.Store(run / 's2')
    assert sorted(on_disk.manifest['arrays']) == sorted(names[:2])                               # the entry of the third array was lost ...
    assert on_disk.path(names[2]).is_file() and not on_disk.path(names[3]).exists()                  # ... its file was not
    resumed = build(s1_small, terr, ALL[:6], run, block_matches=3)
    s = load(resumed)
    n = len(s.array('views/src'))
    assert s.manifest['arrays'] == {name: {'shape': list(shape), 'dtype': dtype.str} for name, (shape, dtype) in expected_arrays(n).items()}
    ref = build(s1_small, terr, ALL[:6], tmp_path / 'ref', block_matches=3)
    assert store_hashes(resumed) == store_hashes(ref)
    s.verify()


def test_a_manifest_entry_that_does_not_match_the_layout_is_refused_before_a_block_is_built(s1_small, terr, tmp_path):
    run = interrupted(tmp_path, s1_small, terr, damage=False)
    path = run / 's2' / 'STORE.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['arrays']['x/ev_n']['dtype'] = '<i4'                                          # edited: the file and the layout say int16
    path.write_text(json.dumps(manifest), encoding='utf-8')
    before = snapshot(run)
    with pytest.raises(ValueError, match='do not match the layout'):
        build(s1_small, terr, ALL[:12], run, block_matches=4, records=records_forbidden)
    assert snapshot(run) == before


def test_a_resume_on_other_code_warns_in_one_line_and_records_both_states(s1_small, terr, tmp_path, monkeypatch, capsys):
    """Review Minor 6: RUN.json holds the commit and dirty flag of the first call; a resume on other code says so and records its own."""
    run = tmp_path / '20261021_c_ept_s2_8'
    with monkeypatch.context() as patch:
        patch.setattr(runlog, '_git', git_as('a', False))
        with pytest.raises(store_build.StopRequested):
            build(s1_small, terr, ALL[:12], run, block_matches=4, stop_after_block=1)
    assert capsys.readouterr().err == ''                                                        # the first call has nothing to compare with
    with monkeypatch.context() as patch:
        patch.setattr(runlog, '_git', git_as('b', True))
        build(s1_small, terr, ALL[:12], run, block_matches=4)
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1 and err.startswith('WARNING: ') and 'a' * 12 in err and 'b' * 12 in err and 'resum' in err
    man = run_json(run)
    assert (man['code_commit'], man['code_dirty']) == ('a' * 40, False)                     # RUN.json still says what the first call ran on
    r = man['s2_result']
    assert [(c['code_commit'], c['code_dirty']) for c in r['per_call']] == [('a' * 40, False), ('b' * 40, True)]
    assert r['code_state_at_end'] == {'code_commit': 'b' * 40, 'code_dirty': True} and r['complete'] is True
    assert err.strip() == 'WARNING: ' + r['warnings'][0] and len(r['warnings']) == 1 and r['per_call'][1]['warnings'] == r['warnings']


def test_a_resume_on_the_same_code_prints_no_warning(s1_small, terr, tmp_path, monkeypatch, capsys):
    run = tmp_path / '20261021_c_ept_s2_9'
    with monkeypatch.context() as patch:
        patch.setattr(runlog, '_git', git_as('c', True))
        with pytest.raises(store_build.StopRequested):
            build(s1_small, terr, ALL[:12], run, block_matches=4, stop_after_block=1)
        build(s1_small, terr, ALL[:12], run, block_matches=4)
    assert capsys.readouterr().err == '' and run_json(run)['s2_result']['warnings'] == []


def test_resume_refuses_another_terrain_before_any_write(s1_small, terr, tmp_path):
    run = interrupted(tmp_path, s1_small, terr)
    other = terrain_copy(terr, tmp_path / 'terrain2', mutate=lambda info: info.update(note='another build of the same grid'))
    before = snapshot(run)
    with pytest.raises(ValueError, match='terrain'):
        build(s1_small, other, ALL[:12], run, block_matches=4, records=records_forbidden)
    assert snapshot(run) == before
    gc.collect()
    shutil.rmtree(other, ignore_errors=True)


def test_resume_refuses_another_view_layout_before_any_write(s1_small, terr, tmp_path, monkeypatch):
    """A key added to ``tokens.VIEW_SPEC`` after the first run would leave the finished blocks without that column."""
    run = interrupted(tmp_path, s1_small, terr)
    before = snapshot(run)
    extra = types.MappingProxyType({**tokens.VIEW_SPEC, 'extra_counter': ((), np.dtype('i2'))})
    monkeypatch.setattr(tokens, 'VIEW_SPEC', extra)
    with pytest.raises(ValueError, match='layout'):
        build(s1_small, terr, ALL[:12], run, block_matches=4, records=records_forbidden)
    assert snapshot(run) == before


def test_a_complete_store_is_returned_untouched_and_the_plan_is_still_checked(s1_small, terr, full):
    run = Path(full).parent
    before = snapshot(run)
    assert build(s1_small, terr, ALL, run, block_matches=5, records=records_forbidden) == Path(full)
    assert snapshot(run) == before
    with pytest.raises(ValueError, match='block_matches'):
        build(s1_small, terr, ALL, run, block_matches=6, records=records_forbidden)
    with pytest.raises(ValueError, match='plan_sha256'):
        build(s1_small, terr, ALL[:-1], run, block_matches=5, records=records_forbidden)
    assert snapshot(run) == before


def test_the_plan_hash_ties_the_list_to_one_s1_store(s1_small, terr, tmp_path):
    """R13-n: the hash is over the S1 files hash and the ordered row list (no key); another S1 store gives another plan."""
    build(s1_small, terr, [0, 1, 2], tmp_path / 'a', block_matches=2)
    build(s1_small, terr, [2, 1, 0], tmp_path / 'b', block_matches=2)
    ha, hb = (run_json(tmp_path / n)['args']['plan_sha256'] for n in 'ab')
    assert len(ha) == 64 and ha != hb
    files = json.loads((s1_small / 's1' / 'STORE.json').read_text(encoding='utf-8'))['files']
    assert ha == runlog.sha256_json({'s1_files_sha256': runlog.sha256_json(files), 'matches': [0, 1, 2]})
    assert ha != runlog.sha256_json({'s1_files_sha256': runlog.sha256_json({}), 'matches': [0, 1, 2]})


def test_progress_is_reported_after_every_block(s1_small, terr, tmp_path):
    seen = []
    build(s1_small, terr, ALL[:6], tmp_path / 'r', block_matches=2, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def fake_pool_class(state):
    """A stand-in for ``ProcessPoolExecutor`` that runs each job at once, in this process, and keeps the error of a failed job in its
    future, as the real one does. ``state`` counts the futures that were submitted and not yet read, and lists the blocks that were
    cancelled."""
    class FakePool:
        def __init__(self, max_workers, mp_context=None, initializer=None, initargs=()):
            state['workers'] = max_workers
            initializer(*initargs)                                                                  # sets the builder of this "worker"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            store_build._BUILDER = None
            return False

        def submit(self, fn, job):
            future = concurrent.futures.Future()
            try:
                future.set_result(fn(job))
            except Exception as error:
                future.set_exception(error)
            state['outstanding'] += 1
            state['peak'] = max(state['peak'], state['outstanding'])
            state['order'].append(job[0])
            read, cancel = future.result, future.cancel

            def result(*args, **kwargs):
                state['outstanding'] -= 1
                return read(*args, **kwargs)

            def cancelled():
                state['cancelled'].append(job[0])
                return cancel()
            future.result, future.cancel = result, cancelled
            return future
    return FakePool


def new_pool_state():
    return {'outstanding': 0, 'peak': 0, 'order': [], 'cancelled': []}


def test_at_most_workers_blocks_are_in_flight_and_they_are_written_in_block_order(s1_small, terr, tmp_path, monkeypatch):
    """R13-k: at most ``workers`` blocks are submitted and not yet read; the blocks are marked in block order."""
    state, rows = new_pool_state(), ALL[:8]
    with monkeypatch.context() as patch:                                     # not monkeypatch.undo(): that would undo the fake process list
        patch.setattr(store_build.cf, 'ProcessPoolExecutor', fake_pool_class(state))
        pooled = build(s1_small, terr, rows, tmp_path / 'pooled', workers=3, block_matches=1)
    assert state['workers'] == 3 and state['peak'] == 3 and state['outstanding'] == 0 and state['order'] == list(range(8))
    single = build(s1_small, terr, rows, tmp_path / 'single', workers=1, block_matches=2)
    assert store_hashes(pooled) == store_hashes(single)
    assert load(pooled).manifest['stages']['views']['blocks'] == list(range(8))                      # marked in block order


def test_cancel_is_attempted_on_the_blocks_in_flight_when_a_block_fails(s1_small, terr, tmp_path, monkeypatch):
    """Review Minor 7: this checks that ``cancel`` is called on every block in flight, in block order, and that nothing after the failed
    block is written. It cannot say what a real ``cancel`` does: it stops a block that has not started, and a block that is running
    finishes before the pool shuts down (its result is dropped); the stand-in pool has already run every job."""
    state = new_pool_state()
    with monkeypatch.context() as patch:
        patch.setattr(store_build.cf, 'ProcessPoolExecutor', fake_pool_class(state))
        with pytest.raises(store_build.S2WorkerError):
            build(s1_small, terr, ALL[:12], tmp_path / 'r', workers=3, block_matches=1, records=records_leaky)
    assert state['cancelled'] == [8, 9]                                      # block 7 (match 7) failed; blocks 8 and 9 were in flight
    assert load(tmp_path / 'r' / 's2').done('views') == set(range(7))


@pytest.mark.parametrize('fault', ['another block', 'a view missing'])
def test_a_worker_result_that_does_not_fit_its_block_is_refused_before_it_is_written(s1_small, terr, tmp_path, monkeypatch, fault):
    original = store_build._BlockBuilder.build

    def wrong(self, job):
        b, arrays, t0_seen = original(self, job)
        return (b + 1, arrays, t0_seen) if fault == 'another block' else (b, arrays, t0_seen[:-1])
    monkeypatch.setattr(store_build._BlockBuilder, 'build', wrong)
    with pytest.raises(ValueError, match='block 0'):
        build(s1_small, terr, ALL[:4], tmp_path / 'r', block_matches=2)
    s = load(tmp_path / 'r' / 's2')
    assert s.done('views') == set() and not np.asarray(s.array('views/ref')).any()


# ---- one build per run folder (final review M7) --------------------------------------------------------------------------------------------
def test_a_second_build_on_the_same_run_is_refused_while_the_first_one_runs(s1_small, terr, tmp_path):
    """Two builds on one run folder would write the same rows and the same STORE.json. The second one, a build and a resume alike, is
    refused at the lock; the first goes on and finishes. The lock lies outside the store root and is gone at the end."""
    run, attempts, held = tmp_path / '20261021_c_ept_s2_11', [], []

    def second(done, total):
        held.append((run / store_build.LOCK_NAME).is_file())
        for kw in ({}, {'workers': 2}):
            try:
                build(s1_small, terr, ALL[:6], run, block_matches=2, records=records_forbidden, **kw)
            except Exception as error:
                attempts.append(error)
            else:
                attempts.append(None)
        held.append((run / store_build.LOCK_NAME).is_file())                      # a refused call did not take the first one's lock away
    root = build(s1_small, terr, ALL[:6], run, block_matches=2, progress=second)
    assert held == [True] * 6 and len(attempts) == 6
    assert all(isinstance(e, store_build.RunLocked) and store_build.LOCK_NAME in str(e) for e in attempts)
    assert not (run / store_build.LOCK_NAME).exists() and not list(Path(root).rglob('*.lock'))
    assert store_hashes(root) == store_hashes(build(s1_small, terr, ALL[:6], tmp_path / 'ref', block_matches=2))


def test_a_stale_lock_refuses_a_build_and_a_resume_with_a_message_and_the_builder_never_removes_it(s1_small, terr, tmp_path):
    """A power cut leaves the lock behind. The build says so and tells the user to delete the file; it never deletes it itself (it cannot
    know that the process that made it is dead)."""
    run = interrupted(tmp_path, s1_small, terr, damage=False)                       # a resumable store; the cut left the lock as well
    lock = run / store_build.LOCK_NAME
    lock.write_text(json.dumps({'pid': 4_000_000_000, 'created': '2026-10-21T01:02:03+00:00'}), encoding='utf-8')
    before = snapshot(run)
    with pytest.raises(store_build.RunLocked) as raised:
        build(s1_small, terr, ALL[:12], run, block_matches=4, records=records_forbidden)
    text = str(raised.value)
    assert store_build.LOCK_NAME in text and str(lock) in text and 'delete' in text and 'resume' in text
    assert '2026-10-21T01:02:03' in text and '4000000000' in text and 'SYN_' not in text
    assert snapshot(run) == before and lock.is_file()                                # nothing written, the lock is still there
    lock.unlink()                                                                     # the deliberate act of the user
    done = build(s1_small, terr, ALL[:12], run, block_matches=4)
    assert load(done).manifest['complete'] is True and not lock.exists()
    assert store_hashes(done) == store_hashes(build(s1_small, terr, ALL[:12], tmp_path / 'ref', block_matches=4))


def test_a_lock_without_readable_content_refuses_too_and_nothing_else_is_written(s1_small, terr, tmp_path):
    run = tmp_path / '20261021_c_ept_s2_15'
    run.mkdir()
    (run / store_build.LOCK_NAME).write_bytes(b'')
    with pytest.raises(store_build.RunLocked, match='delete'):
        build(s1_small, terr, ALL[:4], run, records=records_forbidden)
    assert [p.name for p in run.iterdir()] == [store_build.LOCK_NAME]


def test_the_lock_does_not_outlive_a_build_that_stops_fails_or_is_refused(s1_small, terr, tmp_path):
    stopped, failed, refused = (tmp_path / f'20261021_c_ept_s2_{n}' for n in (21, 22, 23))
    with pytest.raises(store_build.StopRequested):
        build(s1_small, terr, ALL[:6], stopped, block_matches=2, stop_after_block=1)
    with pytest.raises(store_build.S2WorkerError):
        build(s1_small, terr, ALL[6:12], failed, block_matches=5, records=records_leaky)
    build(s1_small, terr, ALL[:6], refused, block_matches=2)
    before = snapshot(refused)
    with pytest.raises(ValueError, match='block_matches'):                           # refused while the lock is held: a plan that differs
        build(s1_small, terr, ALL[:6], refused, block_matches=3, records=records_forbidden)
    assert snapshot(refused) == before
    for run in (stopped, failed, refused):
        assert not (run / store_build.LOCK_NAME).exists(), run.name
    build(s1_small, terr, ALL[:6], stopped, block_matches=2)                          # and a resume is not held up by the stopped call
    assert load(stopped / 's2').manifest['complete'] is True


# ---- a store without its RUN.json (final review M6) ----------------------------------------------------------------------------------------
def test_a_run_whose_store_exists_but_whose_run_json_is_missing_is_refused_and_nothing_is_written(s1_small, terr, tmp_path):
    """RUN.json is written before STORE.json exists, so a store without it means the file was deleted or the run folder was copied without
    it. A new RUN.json would claim this call's commit, GPU and plan for rows that another call built."""
    done, cut = tmp_path / '20261021_c_ept_s2_24', interrupted(tmp_path, s1_small, terr, damage=False)
    build(s1_small, terr, ALL[:6], done, block_matches=3)
    for run, matches, block_matches in ((done, ALL[:6], 3), (cut, ALL[:12], 4)):                   # a complete store, a store that was cut
        (run / 'RUN.json').unlink()
        before = snapshot(run)
        with pytest.raises(ValueError, match='no RUN.json') as raised:
            build(s1_small, terr, matches, run, block_matches=block_matches, records=records_forbidden)
        assert 'SYN_' not in str(raised.value) and run.name in str(raised.value)
        assert snapshot(run) == before and not (run / 'RUN.json').exists()
    fresh = build(s1_small, terr, ALL[:6], tmp_path / 'fresh', block_matches=3)                     # a new run still gets its RUN.json
    assert run_json(Path(fresh).parent)['s2_result']['complete'] is True


# ---- the counters of the call are saved with the mark (final review M3) ---------------------------------------------------------------------
def test_the_counters_of_the_call_are_in_the_same_save_as_the_mark_of_the_block(s1_small, terr, tmp_path, monkeypatch):
    """``mark_done`` saves STORE.json. If the counters were raised only after it, a cut between that save and the next one would leave a
    block that is marked and that the call record does not count. At every save of the build the marked blocks and the counted blocks
    (and the counted matches, 2 per block here) agree."""
    original, observed = store_core.Store.save, []

    def spy(self):
        stage = self.manifest['stages'].get('views')
        if stage and stage.get('calls'):
            observed.append((len(stage['blocks']), sum(c['blocks_built'] for c in stage['calls']),
                             sum(c['matches_built'] for c in stage['calls'])))
        original(self)
    with monkeypatch.context() as patch:
        patch.setattr(store_core.Store, 'save', spy)
        build(s1_small, terr, ALL[:6], tmp_path / 'r', block_matches=2)
    assert len(observed) >= 4 and observed[-1][:2] == (3, 3)                               # the marks of all three blocks were saved
    assert all(marked == counted and matches == 2 * counted for marked, counted, matches in observed), observed


def test_a_mark_that_fails_leaves_the_counters_at_the_marked_blocks(s1_small, terr, tmp_path, monkeypatch):
    original, run = store_core.Store.mark_done, tmp_path / '20261021_c_ept_s2_25'

    def failing(self, stage, block, names):
        if block == 1:
            raise OSError('disk error at the mark of block 1')
        original(self, stage, block, names)
    with monkeypatch.context() as patch:
        patch.setattr(store_core.Store, 'mark_done', failing)
        with pytest.raises(OSError, match='disk error'):
            build(s1_small, terr, ALL[:6], run, block_matches=2)
    stage = load(run / 's2').manifest['stages']['views']
    (call,) = stage['calls']
    assert stage['blocks'] == [0] and (call['blocks_built'], call['matches_built']) == (1, 2)       # block 1 is not counted: it is not marked


# ---- the code state at the end of a repaired record (final review M4) ------------------------------------------------------------------------
def test_a_repaired_record_takes_the_code_state_at_the_end_from_the_last_call_that_built(s1_small, terr, tmp_path, monkeypatch):
    """A call that only repairs the record (the store is complete, ``s2_result`` is missing) builds nothing: its own commit is not the
    code that made the store."""
    run = tmp_path / '20261021_c_ept_s2_26'

    def cut(*args, **kwargs):
        raise OSError('power cut between finalize and the run record')
    with monkeypatch.context() as patch:
        patch.setattr(runlog, '_git', git_as('a', False))
        patch.setattr(runlog, 'amend_manifest', cut)
        with pytest.raises(OSError, match='power cut'):
            build(s1_small, terr, ALL[:6], run, block_matches=3)
    with monkeypatch.context() as patch:
        patch.setattr(runlog, '_git', git_as('b', True))
        build(s1_small, terr, ALL[:6], run, block_matches=3, records=records_forbidden)
    r = run_json(run)['s2_result']
    assert r['recovered'] is True and r['calls'] == 1
    assert r['code_state_at_end'] == {'code_commit': 'a' * 40, 'code_dirty': False} == {k: r['per_call'][-1][k] for k in ('code_commit', 'code_dirty')}
    assert run_json(run)['code_commit'] == 'a' * 40                                              # and the first call's, as always


# ---- an array file that a cut left unreadable (final review M5) -----------------------------------------------------------------------------
def cut_in_the_third_array(tmp_path, s1, terr_dir, monkeypatch, name):
    """The build of ALL[:6] is cut when the entry of its third array is saved: the file of that array exists, the manifest does not list it.
    Returns (run, the names of the arrays in layout order)."""
    run, original, saves = tmp_path / name, store_core.Store.save, {'n': 0}

    def cut_there(self):
        saves['n'] += 1
        if saves['n'] == 4:                                                       # 1: the stage, 2 and 3: two arrays, 4: the third array
            raise OSError('power cut between an array file and its entry')
        original(self)
    with monkeypatch.context() as patch:
        patch.setattr(store_core.Store, 'save', cut_there)
        with pytest.raises(OSError, match='power cut'):
            build(s1, terr_dir, ALL[:6], run, block_matches=3, records=records_forbidden)
    names = list(expected_arrays(1))
    on_disk = store_core.Store(run / 's2')
    assert names[2] not in on_disk.manifest['arrays'] and on_disk.path(names[2]).is_file()
    return run, names


@pytest.mark.parametrize('kept', [0, 10, 40], ids=['empty-file', 'cut-in-the-magic-string', 'cut-in-the-header'])
def test_an_unlisted_array_file_that_cannot_be_opened_is_created_again(s1_small, terr, tmp_path, monkeypatch, capsys, kept):
    """A cut during the creation of an array can leave 0 bytes or half a header, and STORE.json does not list the array yet (nothing can
    have been written into it: a block is marked only after every array is listed), so ``allocate`` would fail on every resume."""
    monkeypatch.setattr(runlog, '_git', git_as('c', False))                         # other commits land in this worktree: no "other code" warning
    run, names = cut_in_the_third_array(tmp_path, s1_small, terr, monkeypatch, '20261021_c_ept_s2_27')
    victim = store_core.Store(run / 's2').path(names[2])
    victim.write_bytes(victim.read_bytes()[:kept])
    capsys.readouterr()
    resumed = build(s1_small, terr, ALL[:6], run, block_matches=3)
    assert store_hashes(resumed) == store_hashes(build(s1_small, terr, ALL[:6], tmp_path / 'ref', block_matches=3))
    s = load(resumed)
    s.verify()
    assert s.manifest['arrays'][names[2]]['shape'][0] == len(s.array('views/src'))
    err = capsys.readouterr().err
    r = run_json(run)['s2_result']
    assert len(err.splitlines()) == 1 and err.startswith('WARNING: ') and names[2] in err                 # one line, and the fact is on record
    assert r['warnings'] == [err.strip()[len('WARNING: '):]] and 'created again' in r['warnings'][0]


def test_an_unlisted_array_file_that_opens_is_kept_and_nothing_is_said(s1_small, terr, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runlog, '_git', git_as('c', False))
    run, names = cut_in_the_third_array(tmp_path, s1_small, terr, monkeypatch, '20261021_c_ept_s2_28')
    capsys.readouterr()
    resumed = build(s1_small, terr, ALL[:6], run, block_matches=3)
    assert capsys.readouterr().err == '' and run_json(run)['s2_result']['warnings'] == []
    assert store_hashes(resumed) == store_hashes(build(s1_small, terr, ALL[:6], tmp_path / 'ref', block_matches=3))


def test_a_listed_array_file_that_cannot_be_opened_still_raises_and_is_left_alone(s1_small, terr, tmp_path):
    """A listed array may hold the rows of marked blocks: an unreadable one is damage, not a leftover, and is never replaced."""
    run = interrupted(tmp_path, s1_small, terr, damage=False)                          # every array is listed, block 0 is marked
    victim = store_core.Store(run / 's2').path('x/summary')
    victim.write_bytes(b'')
    before = snapshot(run)
    with pytest.raises((ValueError, EOFError, OSError)):
        build(s1_small, terr, ALL[:12], run, block_matches=4, records=records_forbidden)
    assert victim.read_bytes() == b'' and snapshot(run) == before


# ---- a model server that starts while the build runs (final review, recommendation 3) -------------------------------------------------------
@pytest.mark.parametrize('workers', [1, 3])
def test_a_model_server_that_starts_during_the_build_stops_it_cleanly_and_the_run_resumes(s1_small, terr, tmp_path, monkeypatch,
                                                                                         fake_processes, workers):
    """The author runs Strata for coding: if it starts while S2 builds, the timing and memory record would be wrong. E6 is checked again
    after every finished block; the build stops after a marked block (nothing partial), says to stop the server and resume, and the run
    resumes to the bytes of an uninterrupted build."""
    monkeypatch.setattr(runlog, '_git', git_as('c', False))                         # a fixed code state: no "other code" warning on the resume
    run, state, rows = tmp_path / '20261021_c_ept_s2_29', new_pool_state(), ALL[:12]

    def the_author_starts_strata(done, total):
        if done == 2:
            fake_processes([synth.STRATA_SERVER])
    with monkeypatch.context() as patch:
        if workers > 1:
            patch.setattr(store_build.cf, 'ProcessPoolExecutor', fake_pool_class(state))
        with pytest.raises(run_s1.StrataRunning, match=r'E6.*pid 4101.*Stop the model server.*resume') as raised:
            build(s1_small, terr, rows, run, workers=workers, block_matches=2, progress=the_author_starts_strata)
    text = str(raised.value)
    assert 'after block 2 of 6' in text and run.name in text and 'SYN_' not in text
    s = load(run / 's2')
    assert s.done('views') == {0, 1} and not s.manifest['complete']                           # the two blocks that were done are marked ...
    (call,) = s.manifest['stages']['views']['calls']
    assert (call['blocks_built'], call['matches_built']) == (2, 4)                            # ... and counted, and nothing else
    first = len(store_build.view_table(s1_store(s1_small), rows[:4])['src'])
    assert not np.asarray(s.array('views/ref'))[first:].any()                                 # no row of any later block was written
    assert not (run / store_build.LOCK_NAME).exists()
    before = snapshot(run)
    with pytest.raises(run_s1.StrataRunning):                                                 # while it runs, a resume is refused at the start
        build(s1_small, terr, rows, run, workers=workers, block_matches=2, records=records_forbidden)
    assert snapshot(run) == before
    fake_processes([])                                                                        # the author stopped it
    done = build(s1_small, terr, rows, run, block_matches=2)
    assert load(done).done('views') == set(range(6)) and load(done).manifest['complete'] is True
    assert store_hashes(done) == store_hashes(build(s1_small, terr, rows, tmp_path / 'ref', block_matches=2))
    r = run_json(run)['s2_result']
    assert r['calls'] == 2 and r['matches_built'] == 12 and r['blocks_resumed'] == 2 and r['warnings'] == []
    if workers > 1:
        assert state['cancelled'] == [2, 3]                  # blocks 2 and 3 were in flight when the server was seen: cancelled, not written


def test_e6_is_checked_again_after_every_finished_block_but_the_last(s1_small, terr, tmp_path, monkeypatch):
    calls, original = [], run_s1.require_no_strata

    def counting():
        calls.append(1)
        return original()
    monkeypatch.setattr(run_s1, 'require_no_strata', counting)
    build(s1_small, terr, ALL[:6], tmp_path / 'r', block_matches=2)
    assert len(calls) == 1 + 2                    # at the start, then after block 0 and block 1; after the last block nothing is left to protect


# ---- the game id inside a key is scrubbed from worker errors too (final review M10) ------------------------------------------------------------
@pytest.mark.parametrize('records, workers', [(records_leaky_id, 1), (records_leaky_id, 2), (records_leaky_both, 1), (records_leaky_both, 2)],
                         ids=['id-one-process', 'id-pool', 'key-and-id-one-process', 'key-and-id-pool'])
def test_the_game_id_inside_the_key_is_scrubbed_from_an_error_of_a_worker(s1_small, terr, tmp_path, records, workers):
    """A message that quotes a value of the file names the match by its number alone (the key is ``<platform>_<game id>``). Both the key
    and the number are replaced, and an error that holds only the number is recognised as one that names the match."""
    run = tmp_path / 'r'
    with pytest.raises(store_build.S2WorkerError) as raised:
        build(s1_small, terr, ALL[:12], run, workers=workers, block_matches=5, records=records)
    text, shown = str(raised.value), ''.join(traceback.format_exception(raised.value))
    assert '0007' not in shown and 'SYN_0007' not in shown                                  # neither the number nor the key is shown
    assert '<id>' in text and 'OSError' in text and 'match #7' in text and (records is not records_leaky_both or '<key>' in text)
    s = load(run / 's2')
    assert s.done('views') == {0} and not s.manifest['complete']                              # the failed block was not written


def test_the_key_parts_that_name_a_match_are_the_key_and_its_game_id():
    parts = store_build._key_parts
    assert parts('SYN_0007') == ('SYN_0007', '0007') and parts('EUW1_6543210987') == ('EUW1_6543210987', '6543210987')
    assert parts('6543210987') == ('6543210987',) and parts('EUW1') == ('EUW1',) and parts('') == ()
    assert store_build._scrub('read EUW1_6543210987.json, game 6543210987', 'EUW1_6543210987') == 'read <key>.json, game <id>'


# ---- arguments, S1-only matches and holdouts (R13-h) -------------------------------------------------------------------------------------
@pytest.mark.parametrize('kw, error', [({'workers': 0}, ValueError), ({'workers': -1}, ValueError), ({'workers': True}, ValueError),
                                       ({'workers': 2.0}, ValueError), ({'block_matches': 0}, ValueError),
                                       ({'block_matches': -3}, ValueError), ({'block_matches': 2.0}, TypeError),
                                       ({'block_matches': True}, TypeError), ({'block_matches': '7'}, TypeError),
                                       ({'stop_after_block': -1}, ValueError), ({'stop_after_block': True}, ValueError),
                                       ({'stop_after_block': 1.5}, ValueError), ({'workers': 62}, ValueError),
                                       ({'workers': 1000}, ValueError), ({'info': 'macros'}, TypeError), ({'info': ['FIT']}, TypeError),
                                       ({'info': {'path': object()}}, TypeError)])
def test_bad_arguments_are_refused_before_anything_is_written(s1_small, terr, tmp_path, kw, error):
    with pytest.raises(error):
        build(s1_small, terr, [0, 1], tmp_path / 'r', records=records_forbidden, **kw)
    assert not (tmp_path / 'r').exists()


def test_workers_above_the_limit_of_the_process_pool_on_windows_are_refused_before_anything_is_written(s1_small, terr, tmp_path):
    """Review Minor 8: ``ProcessPoolExecutor`` refuses more than 61 workers on Windows, but only after RUN.json and the store are made."""
    assert store_build.MAX_WORKERS == 61
    store_build._check_workers(61)                                                                   # the limit itself is allowed
    for workers in (62, 1000):
        with pytest.raises(ValueError, match='at most 61'):
            build(s1_small, terr, [0, 1], tmp_path / 'r', workers=workers, records=records_forbidden)
    assert not (tmp_path / 'r').exists()


@pytest.mark.parametrize('matches', [[], [0, 0], [0, 1, 0], [True], [0.0], ['0'], [None], 7, 'abc'],
                         ids=['empty', 'twice', 'repeat', 'bool', 'float', 'text', 'none', 'int', 'str'])
def test_a_match_list_that_is_no_list_of_distinct_rows_is_refused(s1_small, terr, tmp_path, matches):
    with pytest.raises((ValueError, TypeError)):
        build(s1_small, terr, matches, tmp_path / 'r', records=records_forbidden)
    assert not (tmp_path / 'r').exists()


@pytest.mark.parametrize('matches', [[0, N_MATCHES], [-1], [N_MATCHES + 100]])
def test_only_matches_of_the_s1_store_are_accepted(s1_small, terr, tmp_path, matches):
    with pytest.raises(ValueError, match='S1 store'):
        build(s1_small, terr, matches, tmp_path / 'r', records=records_forbidden)
    assert not (tmp_path / 'r').exists()


@pytest.mark.parametrize('role, patch', [('TEST', '15.22'), ('EXTERNAL', '16.13'), ('TEST', '15.18'), ('FIT', '15.22'), ('FIT', '16.13')])
def test_holdout_matches_are_refused_before_anything_is_read(terr, tmp_path, role, patch):
    """A store that lists a TEST or EXTERNAL role, or a holdout patch, is refused whatever the match list; ``records`` is never asked."""
    s1_run = write_mini_s1(tmp_path / '20261021_c_ept_s1_9', [mini_row(role=role, patch=patch)])
    with pytest.raises(ValueError, match='holdout|TEST|EXTERNAL'):
        build(s1_run, terr, [0], tmp_path / 'r', records=records_forbidden)
    assert not (tmp_path / 'r').exists()


def test_a_match_without_views_is_not_read_and_leaves_no_row(terr, tmp_path):
    """The middle match has no frame-clock row and no event row (a short match): it is neither read nor represented; the others are."""
    ts0, ts2 = (int(synth.records_from_seed(key, '15.18').frame_ts[k]) for key, k in (('MINI_0', 2), ('MINI_2', 3)))
    s1_run = write_mini_s1(tmp_path / '20261021_c_ept_s1_9', [
        mini_row(key='MINI_0', frames=[(2, ts0)], events=[(1, ts0 + 5000, 2, ts0)]),
        mini_row(key='MINI_1', frames=[], events=[]),
        mini_row(key='MINI_2', frames=[(3, ts2)], events=[])])
    asked = []

    def spy(key, patch):
        asked.append(key)
        return synth.records_from_seed(key, patch)
    root = build(s1_run, terr, [0, 1, 2], tmp_path / 'r', block_matches=2, records=spy)
    s = load(root)
    assert asked == ['MINI_0', 'MINI_2'] and np.asarray(s.array('views/match')).tolist() == [0, 0, 2]
    assert np.asarray(s.array('views/src')).tolist() == [0, 1, 0] and s.manifest['complete'] is True
    assert run_json(tmp_path / 'r')['s2_result']['views'] == 3 and s.manifest['stages']['views']['n_blocks'] == 2


def test_a_store_without_any_view_is_refused(terr, tmp_path):
    s1_run = write_mini_s1(tmp_path / '20261021_c_ept_s1_9', [mini_row(frames=(), events=())])
    with pytest.raises(ValueError, match='no view'):
        build(s1_run, terr, [0], tmp_path / 'r', records=records_forbidden)
    assert not (tmp_path / 'r').exists()


def test_records_are_asked_for_key_and_patch_only_and_no_holdout_switch_is_passed(s1_small, terr, tmp_path):
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return synth.records_from_seed(*args, **kwargs)
    build(s1_small, terr, [0, 1, 2], tmp_path / 'r', records=spy)
    assert len(calls) == 3 and all(len(args) == 2 and not kwargs for args, kwargs in calls)
    for name in ('store_build.py', 'run_s2.py'):
        tree = ast.parse((Path(store_build.__file__) if name == 'store_build.py' else REPO / 'c_ept' / 'workflows' / name).read_text(encoding='utf-8'))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == 'allow_holdout'], name


# ---- E6 (R13-a, S2-1) --------------------------------------------------------------------------------------------------------------------
def test_build_s2_signature_is_the_briefs_with_two_optional_additions():
    """The call that ``run_s2.main`` makes fits the real function (the command-line tests stub it)."""
    sig = inspect.signature(store_build.build_s2)
    names = list(sig.parameters)
    assert names[:8] == ['s1_run', 'terrain_path', 'matches', 'run', 'workers', 'block_matches', 'stop_after_block', 'records']
    assert [sig.parameters[n].default for n in names[4:8]] == [8, 250, None, reader.read_match]
    assert names[8:] == ['progress', 'info'] and all(sig.parameters[n].default is None for n in names[8:])
    sig.bind(Path('s1'), Path('terrain'), [0], Path('run'), 8, 250, info={}, progress=None)
    assert inspect.signature(store_build.view_table).bind(None, [0]) and store_build.StopRequested.__mro__[1] is RuntimeError


def test_e6_stops_the_build_before_anything_is_read_or_written(tmp_path, fake_processes):
    fake_processes([synth.STRATA_SERVER])
    with pytest.raises(run_s1.StrataRunning, match=r'E6.*does not stop.*pid 4101'):
        store_build.build_s2(tmp_path / 'no_s1', tmp_path / 'no_terrain', [0], tmp_path / 'run', workers=1, records=records_forbidden)
    assert not (tmp_path / 'run').exists()


def test_the_idle_bridges_do_not_stop_a_build_and_the_record_says_what_was_seen(s1_small, terr, tmp_path, fake_processes):
    fake_processes([synth.STRATA_BRIDGE, (4106, 'python.exe', synth.STRATA_BRIDGE[2]), synth.UNRELATED_PYTHON])
    build(s1_small, terr, [0], tmp_path / 'r')
    man = run_json(tmp_path / 'r')
    assert man['strata_running'] is False and man['strata_detail']['bridges_ignored'] == 2 and man['s2_result']['complete'] is True


def test_the_e6_check_comes_after_the_argument_checks(tmp_path, fake_processes):
    fake_processes([synth.STRATA_SERVER])
    with pytest.raises(ValueError, match='workers'):
        store_build.build_s2(tmp_path / 'no_s1', tmp_path / 'no_terrain', [0], tmp_path / 'run', workers=0, records=records_forbidden)


# ---- the terrain needs provenance (R13-m) ------------------------------------------------------------------------------------------------
def test_a_terrain_folder_without_terrain_json_is_refused(open_dir, s1_small, tmp_path):
    with pytest.raises(ValueError, match='TERRAIN.json'):
        build(s1_small, open_dir, [0], tmp_path / 'r', records=records_forbidden)
    assert not (tmp_path / 'r').exists()


def test_a_terrain_whose_files_differ_from_terrain_json_is_refused(s1_small, terr, tmp_path):
    def wrong_hash(info):
        info['sha256']['terrain.npz'] = '0' * 64
    bad = terrain_copy(terr, tmp_path / 'terrain_bad', mutate=wrong_hash)
    with pytest.raises(ValueError, match='SHA-256'):
        build(s1_small, bad, [0], tmp_path / 'r', records=records_forbidden)
    assert not (tmp_path / 'r').exists()
    gc.collect()
    shutil.rmtree(bad, ignore_errors=True)


def test_a_terrain_json_that_does_not_list_both_files_is_refused(s1_small, terr, tmp_path):
    def one_file(info):
        del info['sha256']['geodesic.npy']
    bad = terrain_copy(terr, tmp_path / 'terrain_bad', mutate=one_file)
    with pytest.raises(ValueError, match='geodesic.npy'):
        build(s1_small, bad, [0], tmp_path / 'r', records=records_forbidden)
    assert not (tmp_path / 'r').exists()
    gc.collect()
    shutil.rmtree(bad, ignore_errors=True)


def test_a_terrain_of_another_s1_run_is_recorded_and_warned_about_not_refused(s1_small, terr, tmp_path, capsys):
    """Review Minor 5 (Ruling 57): one WARNING line on stderr, the fact stays in RUN.json, and the build goes on."""
    def another_s1(info):
        info.update(s1_files_sha256='f' * 64, s1_run='20261020_c_ept_s1_7')
    other = terrain_copy(terr, tmp_path / 'terrain2', mutate=another_s1)
    build(s1_small, other, [0, 1], tmp_path / 'r')
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1 and err.startswith('WARNING: ') and 'terrain' in err and 'another S1 run' in err
    t = run_json(tmp_path / 'r')['terrain']
    assert t['built_from_this_s1'] is False and t['s1_files_sha256'] == 'f' * 64 and t['s1_run'] == '20261020_c_ept_s1_7'
    assert t['sha256']['TERRAIN.json'] == store_core.sha(other / 'TERRAIN.json')
    assert run_json(tmp_path / 'r')['s2_result']['warnings'] == [err.strip()[len('WARNING: '):]]
    build(s1_small, terr, [0, 1], tmp_path / 'matching')                                             # the terrain of this S1 run: silent
    assert capsys.readouterr().err == '' and run_json(tmp_path / 'matching')['s2_result']['warnings'] == []
    gc.collect()
    shutil.rmtree(other, ignore_errors=True)


def test_the_terrain_folder_is_read_only(s1_small, terr, tmp_path):
    copy = terrain_copy(terr, tmp_path / 'terrain_ro')
    before = snapshot(copy)
    build(s1_small, copy, [0, 1], tmp_path / 'r')
    assert snapshot(copy) == before and sorted(p.name for p in copy.iterdir()) == ['TERRAIN.json', 'geodesic.npy', 'terrain.npz']
    gc.collect()
    shutil.rmtree(copy, ignore_errors=True)


# ---- the terrain must come from the current parameters (final review I1) ---------------------------------------------------------------
# A terrain whose files match its own TERRAIN.json passed every check, so a full build given the terrain of the first smoke run (walkable
# threshold min_count 5, before the S2 gate S2G-2 made it 150) would have used it without a word. The grid record and the zone rules of
# TERRAIN.json must be those of this code. Literal copies of the two real TERRAIN.json records (the first and the final smoke run):
REJECTED_SMOKE_GRID = {'cells': 75, 'cell_u': 200, 'histogram_cells': 150, 'histogram_u': 100, 'min_count': 5}
FINAL_SMOKE_GRID = {**REJECTED_SMOKE_GRID, 'min_count': 150}
SMOKE_ZONE_RULES = {'base_sum_above': 1.64, 'base_sum_below': 0.36, 'lane': 0.13, 'mid_band': 0.07, 'norm': 14875.0,
                    'priority': ['base', 'lane', 'mid', 'river', 'jungle'], 'river_band': 0.07}


def terrain_variant(src, dst, mutate):
    """A terrain folder like ``src`` (the two data files are hard links, a copy where links are not possible) with TERRAIN.json edited by
    ``mutate(info)``. Its files match its own TERRAIN.json whatever the edit does to the parameters."""
    dst = Path(dst)
    dst.mkdir(parents=True)
    for name in ('terrain.npz', 'geodesic.npy'):
        try:
            os.link(Path(src) / name, dst / name)
        except OSError:
            shutil.copy2(Path(src) / name, dst / name)
    info = json.loads((Path(src) / 'TERRAIN.json').read_text(encoding='utf-8'))
    mutate(info)
    (dst / 'TERRAIN.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return dst


TERRAIN_REFUSALS = [
    pytest.param(lambda i: i['grid'].update(REJECTED_SMOKE_GRID), rf'grid\.min_count is 5 .*terrain\.MIN_COUNT is {terrain.MIN_COUNT}',
                 id='min-count-5-of-the-first-smoke-run'),
    pytest.param(lambda i: i['grid'].update(cells=terrain.GRID + 1), r'grid\.cells', id='cells'),
    pytest.param(lambda i: i['grid'].update(cell_u=100), r'grid\.cell_u', id='cell-size'),
    pytest.param(lambda i: i['grid'].update(histogram_cells=100), r'grid\.histogram_cells', id='histogram-cells'),
    pytest.param(lambda i: i['grid'].update(histogram_u=200), r'grid\.histogram_u', id='histogram-cell-size'),
    pytest.param(lambda i: i['grid'].pop('min_count'), r'grid\.min_count', id='no-min-count'),
    pytest.param(lambda i: i.pop('grid'), r'no grid record', id='no-grid'),
    pytest.param(lambda i: i.update(grid='75 x 75'), r'no grid record', id='grid-is-not-a-record'),
    pytest.param(lambda i: i['zone_rules'].update(lane=0.12), r'zone_rules\.lane', id='zone-lane'),
    pytest.param(lambda i: i['zone_rules'].update(priority=['lane', 'base', 'mid', 'river', 'jungle']), r'zone_rules\.priority',
                 id='zone-priority'),
    pytest.param(lambda i: i['zone_rules'].pop('mid_band'), r'zone_rules\.mid_band', id='zone-rule-missing'),
    pytest.param(lambda i: i['zone_rules'].update(extra_rule=1), r'zone_rules\.extra_rule', id='zone-rule-added'),
    pytest.param(lambda i: i.pop('zone_rules'), r'no zone_rules record', id='no-zone-rules'),
]


@pytest.mark.parametrize('edit, pattern', TERRAIN_REFUSALS)
def test_a_terrain_built_with_other_parameters_is_refused_before_anything_is_written(s1_small, terr, tmp_path, edit, pattern):
    bad = terrain_variant(terr, tmp_path / 'terrain_bad', edit)
    with pytest.raises(ValueError, match=pattern) as raised:
        build(s1_small, bad, [0], tmp_path / 'r', records=records_forbidden)
    assert 'SYN_' not in str(raised.value)                                                          # the message names fields, never a key
    assert not (tmp_path / 'r').exists()


def test_the_terrain_of_the_first_smoke_run_is_refused_although_its_files_match_its_own_terrain_json(s1_small, terr, tmp_path):
    """The case of the review: ``Terrain.verify`` accepts the folder (files and TERRAIN.json agree), only the parameters tell."""
    old = terrain_variant(terr, tmp_path / 'terrain_old', lambda info: info['grid'].update(REJECTED_SMOKE_GRID))
    t = terrain.Terrain(old)
    t.verify()                                                                                     # the folder is self-consistent
    del t
    gc.collect()
    with pytest.raises(ValueError, match='min_count'):
        build(s1_small, old, ALL[:2], tmp_path / 'r', records=records_forbidden)
    assert not (tmp_path / 'r').exists()
    gc.collect()
    shutil.rmtree(old, ignore_errors=True)


def test_a_terrain_with_the_current_parameters_is_accepted_and_its_grid_is_recorded(terr, full):
    man = run_json(Path(full).parent)
    assert man['terrain']['grid'] == current_terrain_parameters()['grid'] == json.loads((terr / 'TERRAIN.json').read_text())['grid']
    assert man['terrain']['grid'] == FINAL_SMOKE_GRID                                              # the walkable threshold of gate S2G-2
    assert man['s2_result']['complete'] is True


def test_terrain_parameter_problems_names_every_difference_and_accepts_the_two_real_records():
    final = {'grid': dict(FINAL_SMOKE_GRID), 'zone_rules': json.loads(json.dumps(SMOKE_ZONE_RULES))}      # TERRAIN.json of the final smoke run
    assert store_build.terrain_parameter_problems(final) == []
    first = {'grid': dict(REJECTED_SMOKE_GRID), 'zone_rules': json.loads(json.dumps(SMOKE_ZONE_RULES))}  # the first smoke run's
    (only,) = store_build.terrain_parameter_problems(first)
    assert 'grid.min_count' in only and ' 5 ' in only and f'terrain.MIN_COUNT is {terrain.MIN_COUNT}' in only
    both = {'grid': {**FINAL_SMOKE_GRID, 'cells': 80, 'min_count': 5}, 'zone_rules': {**SMOKE_ZONE_RULES, 'lane': 0.2}}
    problems = store_build.terrain_parameter_problems(both)
    assert len(problems) == 3 and any('grid.cells' in p for p in problems) and any('zone_rules.lane' in p for p in problems)
    assert store_build.terrain_parameter_problems({}) == ['TERRAIN.json has no grid record', 'TERRAIN.json has no zone_rules record']
    assert store_build.terrain_parameters() == current_terrain_parameters()                          # what the guard compares with


def test_terrain_problems_reads_the_folder_and_says_what_is_wrong_with_the_file(terr, tmp_path):
    assert store_build.terrain_problems(terr) == []
    assert 'no TERRAIN.json' in store_build.terrain_problems(tmp_path / 'nowhere')[0]
    broken = tmp_path / 'broken'
    broken.mkdir()
    (broken / 'TERRAIN.json').write_text('{"grid": ', encoding='utf-8')
    assert 'cannot be read' in store_build.terrain_problems(broken)[0]
    (broken / 'TERRAIN.json').write_text('[1, 2]', encoding='utf-8')
    assert 'not a record' in store_build.terrain_problems(broken)[0]


# ---- space (R13-k) -----------------------------------------------------------------------------------------------------------------------
def test_free_space_is_checked_for_all_views_plus_ten_percent_before_anything_is_allocated(s1_small, terr, tmp_path, monkeypatch):
    calls = []

    def spy(path, nbytes, margin=1.1):
        calls.append((Path(path), nbytes, margin))
    monkeypatch.setattr(store_core, 'require_free_bytes', spy)
    s1 = s1_store(s1_small)
    rows = [0, 1, 2]
    n = len(store_build.view_table(s1, rows)['src'])
    build(s1_small, terr, rows, tmp_path / 'r')
    assert len(calls) == 1 and calls[0][0] == tmp_path / 'r' / 's2' and calls[0][2] == 1.1
    assert calls[0][1] == n * (tokens.bytes_per_view() + 27)


def test_a_full_disk_stops_the_build_before_a_run_folder_is_made(s1_small, terr, tmp_path, monkeypatch):
    def full_disk(path, nbytes, margin=1.1):
        raise OSError('need 9.9 GiB, free 0.1 GiB')
    monkeypatch.setattr(store_core, 'require_free_bytes', full_disk)
    with pytest.raises(OSError, match='GiB'):
        build(s1_small, terr, [0, 1], tmp_path / 'r', records=records_forbidden)
    assert not (tmp_path / 'r').exists()


# ---- no hindsight (R13-g, S2-11) and errors (R13-r) --------------------------------------------------------------------------------------
def test_the_main_process_reads_only_the_view_table_columns_of_s1(s1_small, terr, tmp_path, monkeypatch):
    names = []
    original = store_core.Store.array

    def spy(self, name):
        names.append(name)
        return original(self, name)
    monkeypatch.setattr(store_core.Store, 'array', spy)
    build(s1_small, terr, [0, 1, 2], tmp_path / 'r')
    assert names and set(names) <= S1_READ_BY_S2, sorted(set(names) - S1_READ_BY_S2)
    assert not [n for n in names if n.split('/')[0] in ('units', 'kills', 'objectives', 'onsets', 'frames')]       # no target, no raw frame


def test_workers_get_ref_and_frame_k_pairs_and_return_plain_arrays(s1_small, terr, tmp_path, monkeypatch):
    jobs, results = [], []
    original = store_build._BlockBuilder.build

    def spy(self, job):
        jobs.append(job)
        result = original(self, job)
        results.append(result)
        return result
    monkeypatch.setattr(store_build._BlockBuilder, 'build', spy)
    build(s1_small, terr, [0, 1, 2, 3], tmp_path / 'r', block_matches=2)
    assert len(jobs) == 2 and len(results) == 2
    s1 = s1_store(s1_small)
    n_pairs = 0
    for b, items in jobs:
        for match, key, patch, refs, ks in items:
            assert isinstance(match, int) and isinstance(key, str) and isinstance(patch, str)
            assert isinstance(refs, np.ndarray) and isinstance(ks, np.ndarray) and refs.ndim == ks.ndim == 1 and len(refs) == len(ks)
            assert refs.dtype == ks.dtype == np.int64 and (key, patch) == match_of(s1, match)
            n_pairs += len(refs)
        pickle.loads(pickle.dumps((b, items)))                                                       # a job crosses a process boundary
    assert n_pairs == len(store_build.view_table(s1, [0, 1, 2, 3])['src'])
    for b, arrays, t0_seen in results:
        assert isinstance(b, int) and isinstance(t0_seen, np.ndarray)
        assert set(arrays) == {f'x/{k}' for k in tokens.VIEW_SPEC} and all(type(a) is np.ndarray for a in arrays.values())
        pickle.loads(pickle.dumps((b, arrays, t0_seen)))                                             # no read-only mapping travels back


def test_s1_targets_cannot_change_a_view(s1_small, terr, tmp_path):
    """S2-11: garble every S1 array except the ones the view table reads; the views must not change (the build has no access to them)."""
    copy = tmp_path / '20261021_c_ept_s1_2'
    shutil.copytree(s1_small / run_s1.STORE_DIR, copy / run_s1.STORE_DIR)
    s = store_core.Store(copy / run_s1.STORE_DIR)
    rng = np.random.default_rng(7)
    garbled = []
    for name in s.manifest['arrays']:
        if name in S1_READ_BY_S2:
            continue
        a = np.load(s.path(name), mmap_mode='r+')
        if a.dtype.kind == 'b':
            a[...] = rng.random(a.shape) < 0.5
        elif a.dtype.kind == 'f':
            a[...] = rng.random(a.shape)
        else:
            a[...] = rng.integers(0, 5, a.shape)
        a.flush()
        del a
        garbled.append(name)
    s.finalize()
    assert {'units/K', 'units/K_at_f', 'units/O', 'units/O_at_f', 'units/conv', 'units/delta', 'objectives/conv_unit'} <= set(garbled)
    rows = [0, 1, 2, 3, 4]
    a = build(s1_small, terr, rows, tmp_path / 'a')
    b = build(copy, terr, rows, tmp_path / 'b')
    assert run_json(tmp_path / 'a')['s1']['files_sha256'] != run_json(tmp_path / 'b')['s1']['files_sha256']       # S2 saw another S1
    assert store_hashes(a) == store_hashes(b)


def test_the_unpicklable_constants_are_read_in_the_worker_not_sent(s1_small, terr, tmp_path):
    with pytest.raises(TypeError):
        pickle.dumps(tokens.VIEW_SPEC)                                                          # the premise: a mapping proxy cannot travel
    root = build(s1_small, terr, ALL[:6], tmp_path / 'r', workers=2, block_matches=3)
    assert load(root).manifest['complete'] is True


@pytest.mark.parametrize('workers', [1, 2])
def test_an_error_with_the_match_key_in_its_text_is_raised_without_the_key_and_the_block_is_not_written(s1_small, terr, tmp_path, workers):
    run = tmp_path / 'r'
    with pytest.raises(store_build.S2WorkerError) as raised:
        build(s1_small, terr, ALL[:12], run, workers=workers, block_matches=5, records=records_leaky)
    shown = ''.join(traceback.format_exception(raised.value))
    assert 'SYN_0007' not in shown and '<key>' in str(raised.value) and 'OSError' in str(raised.value)
    s = load(run / 's2')
    assert s.done('views') == {0} and not s.manifest['complete']                             # block 0 (matches 0-4) is done, block 1 is not
    first = len(store_build.view_table(s1_store(s1_small), ALL[:5])['src'])
    assert not np.asarray(s.array('views/ref'))[first:].any()                                  # nothing of the failed block (or later ones)
    assert not [p for p in text_files(run) if 'SYN_0007' in p.read_text(encoding='utf-8')]


@pytest.mark.parametrize('workers', [1, 2])
def test_a_malformed_match_stops_the_build_and_is_never_skipped(s1_small, terr, tmp_path, workers):
    run = tmp_path / 'r'
    with pytest.raises(reader.MatchFormatError) as raised:
        build(s1_small, terr, ALL[:12], run, workers=workers, block_matches=5, records=records_bad_format)
    assert 'SYN_0007' not in ''.join(traceback.format_exception(raised.value))
    s = load(run / 's2')
    assert s.done('views') == {0} and not s.manifest['complete']


def test_a_match_that_the_reader_now_rejects_stops_the_build_without_the_key(s1_small, terr, tmp_path):
    with pytest.raises(store_build.S2CheckError, match='missing_file') as raised:
        build(s1_small, terr, ALL[:12], tmp_path / 'r', block_matches=5, records=records_rejecting)
    assert 'SYN_0007' not in ''.join(traceback.format_exception(raised.value)) and 'match #7' in str(raised.value)


def test_a_record_of_another_match_stops_the_build(s1_small, terr, tmp_path):
    with pytest.raises(store_build.S2CheckError, match='another match') as raised:
        build(s1_small, terr, ALL[:6], tmp_path / 'r', block_matches=3, records=records_other_match)
    assert 'SYN_0003' not in str(raised.value) and 'SYN_0099' not in str(raised.value)
    assert load(tmp_path / 'r' / 's2').done('views') == {0}                                          # matches 0-2 are fine, 3 is in block 1


def test_frame_times_that_differ_from_s1_stop_the_block_before_it_is_written(s1_small, terr, tmp_path):
    """The record is read again from the files: S1's frame time of every view must be the record's (else the files changed)."""
    run = tmp_path / 'r'
    with pytest.raises(store_build.S2CheckError, match='frame') as raised:
        build(s1_small, terr, ALL[:6], run, block_matches=3, records=records_shifted_frames)
    assert 'SYN_0003' not in str(raised.value) and 'match #3' in str(raised.value)
    s = load(run / 's2')
    first = len(store_build.view_table(s1_store(s1_small), ALL[:3])['src'])
    assert s.done('views') == {0} and not np.asarray(s.array('views/ref'))[first:].any()             # nothing of the block that failed


# ---- import hygiene (R13-a, R13-g) -------------------------------------------------------------------------------------------------------
def module_level_imports(path: Path) -> set:
    """Every module that a statement at module level imports (also inside ``if``/``try``/``with``, not inside a function)."""
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


S1_CHAIN = ('c_ept.run_s1', 'c_ept.targets', 'c_ept.event_rows', 'c_ept.frame_rows', 'c_ept.cohort')


@pytest.mark.parametrize('name', ['store_build.py', 'run_s2.py'])
def test_no_module_level_import_reaches_the_s1_runner_or_the_s1_targets(name):
    imports = module_level_imports(Path(store_build.__file__) if name == 'store_build.py' else REPO / 'c_ept' / 'workflows' / name)
    assert not [i for i in imports if i in S1_CHAIN or i.startswith(S1_CHAIN)], sorted(imports)       # a spawned worker imports this module


def test_importing_the_builder_and_the_runner_loads_no_s1_module():
    code = ('import sys, c_ept.store_build, c_ept.run_s2\n'
            f'print(sorted(m for m in {S1_CHAIN!r} if m in sys.modules))')
    done = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and done.stdout.strip() == '[]', done.stderr[-500:]


def test_the_e6_guard_is_imported_inside_the_entry_points():
    tree = ast.parse(Path(store_build.__file__).read_text(encoding='utf-8'))
    build_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'build_s2')
    inner = [n for n in ast.walk(build_fn) if isinstance(n, ast.ImportFrom) and n.module == 'c_ept.run_s1']
    assert inner and any(a.name == 'require_no_strata' for n in inner for a in n.names)


# ---- command line (R13-j, R13-l, R13-h) --------------------------------------------------------------------------------------------------
@pytest.fixture
def stubs(monkeypatch):
    """The heavy entry points of the command line stubbed (S2-6): ``calls.build`` and ``calls.terrain`` record what they were given."""
    calls = types.SimpleNamespace(build=[], terrain=[])

    def fake_build(s1_run, terrain_path, matches, run, workers=8, block_matches=250, stop_after_block=None, records=None, **kw):
        calls.build.append(dict(s1_run=Path(s1_run), terrain_path=Path(terrain_path), matches=list(matches), run=Path(run),
                                workers=workers, block_matches=block_matches, kw=kw))
        root = Path(run) / 's2'
        root.mkdir(parents=True, exist_ok=True)
        (Path(run) / 'RUN.json').write_text(json.dumps({'stage': 's2', 'created': '2026-10-21T00:00:00+00:00', 's2_result': {
            'complete': True, 'views': 12, 'bytes_per_view': 7359, 'wall_s': 1.5, 'peak_private_gib': 0.5, 'matches': len(matches)}}),
            encoding='utf-8')
        if 'progress' in kw and kw['progress'] is not None:
            kw['progress'](1, 1)
        return root

    def fake_terrain(s1_run, run):
        calls.terrain.append((Path(s1_run), Path(run)))
        out = Path(run) / 'terrain'
        out.mkdir(parents=True)
        return out
    monkeypatch.setattr(store_build, 'build_s2', fake_build)
    monkeypatch.setattr(terrain, 'build_terrain', fake_terrain)
    return calls


def cli(tmp_path, *args):
    return run_s2.main([*args, '--runs-root', str(tmp_path / 'runs'), '--catalog-db', str(tmp_path / 'catalog.sqlite')])


def catalog_rows(tmp_path):
    db = tmp_path / 'catalog.sqlite'
    if not db.exists():
        return []
    con = sqlite3.connect(db)
    try:
        return con.execute('SELECT run, stage FROM runs').fetchall()
    finally:
        con.close()


def test_cli_selects_the_s1_matches_of_the_macros_in_s1_order_and_registers_at_the_end(s1_small, terr, tmp_path, stubs, capsys):
    rc = cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT', 'STOP', '--limit', '5', '--workers', '3',
             '--block-matches', '4')
    assert rc == 0 and len(stubs.build) == 1 and not stubs.terrain
    call = stubs.build[0]
    assert call['matches'] == [0, 1, 5, 6, 10] and all(type(m) is int for m in call['matches'])      # FIT and STOP rows, first 5
    assert (call['workers'], call['block_matches'], call['s1_run'], call['terrain_path']) == (3, 4, s1_small, terr)
    run = call['run']
    assert run.parent == tmp_path / 'runs' and run.name.endswith('_c_ept_s2_1') and run.is_dir()
    assert catalog_rows(tmp_path) == [(run.name, 's2')]
    out = capsys.readouterr().out
    assert str(run) in out and 'SYN_' not in out
    assert call['kw']['info'] == {'macros': ['FIT', 'STOP'], 'limit': 5} and callable(call['kw']['progress'])


def test_cli_defaults_and_the_whole_role_list(s1_small, terr, tmp_path, stubs):
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', *MACROS) == 0
    call = stubs.build[0]
    assert (call['workers'], call['block_matches'], call['matches']) == (8, 250, ALL)


def test_cli_registers_nothing_when_the_build_fails(s1_small, terr, tmp_path, monkeypatch):
    def failing(*args, **kwargs):
        raise RuntimeError('the build stopped')
    monkeypatch.setattr(store_build, 'build_s2', failing)
    with pytest.raises(RuntimeError, match='stopped'):
        cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT')
    assert catalog_rows(tmp_path) == []                                                              # only a complete build is registered


@pytest.mark.parametrize('macros', [['TEST'], ['FIT', 'EXTERNAL'], ['NOPE']])
def test_cli_refuses_test_external_and_unknown_macros(s1_small, terr, tmp_path, stubs, macros, capsys):
    rc = cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', *macros)
    assert rc != 0 and macros[-1] in capsys.readouterr().err
    assert not stubs.build and not stubs.terrain and not (tmp_path / 'runs').exists() and catalog_rows(tmp_path) == []


@pytest.mark.parametrize('extra', [[], ['--terrain', 'T', '--build-terrain']], ids=['neither', 'both'])
def test_cli_needs_exactly_one_terrain_option(s1_small, terr, tmp_path, stubs, extra, capsys):
    extra = [str(terr) if x == 'T' else x for x in extra]
    assert cli(tmp_path, '--s1', str(s1_small), '--macros', 'FIT', *extra) != 0
    assert 'terrain' in capsys.readouterr().err
    assert not stubs.build and not stubs.terrain and not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('flag, value', [('--workers', '0'), ('--workers', '62'), ('--block-matches', '0'), ('--limit', '0')])
def test_cli_refuses_bad_numbers(s1_small, terr, tmp_path, stubs, flag, value, capsys):
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT', flag, value) != 0
    assert flag in capsys.readouterr().err and not stubs.build and not (tmp_path / 'runs').exists()


def test_cli_stops_at_e6_before_a_run_folder_is_made(s1_small, terr, tmp_path, stubs, fake_processes, capsys):
    fake_processes([synth.STRATA_SERVER, synth.STRATA_BRIDGE])
    rc = cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT')
    err = capsys.readouterr().err
    assert rc != 0 and 'E6' in err and 'does not stop' in err and f'pid {synth.STRATA_SERVER[0]}' in err and 'strata_worker_mcp' not in err
    assert not stubs.build and not (tmp_path / 'runs').exists() and catalog_rows(tmp_path) == []


def test_cli_refuses_a_terrain_folder_without_terrain_json_before_a_run_folder_is_made(s1_small, open_dir, tmp_path, stubs, capsys):
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(open_dir), '--macros', 'FIT') != 0
    assert 'TERRAIN.json' in capsys.readouterr().err and not stubs.build and not (tmp_path / 'runs').exists()
    assert catalog_rows(tmp_path) == []


def test_cli_does_not_register_a_build_that_is_not_complete(s1_small, terr, tmp_path, monkeypatch, capsys):
    def unfinished(s1_run, terrain_path, matches, run, *args, **kwargs):
        (Path(run) / 'RUN.json').write_text(json.dumps({'stage': 's2', 's2_result': {'complete': False}}), encoding='utf-8')
        return Path(run) / 's2'
    monkeypatch.setattr(store_build, 'build_s2', unfinished)
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT') != 0
    assert 'did not finish' in capsys.readouterr().err and catalog_rows(tmp_path) == []


def test_cli_resume_continues_in_the_given_run_folder_and_makes_no_new_one(s1_small, terr, tmp_path, stubs):
    run = tmp_path / 'runs' / '20261021_c_ept_s2_1'
    (run / 's2').mkdir(parents=True)
    (run / 's2' / 'STORE.json').write_text('{}', encoding='utf-8')                                  # a store that an earlier call began
    (run / 'RUN.json').write_text('{}', encoding='utf-8')                                           # and the record that the first call wrote
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT', '--limit', '4', '--resume', str(run)) == 0
    assert [c['run'] for c in stubs.build] == [run] and stubs.build[0]['matches'] == [0, 5, 10, 15] and not stubs.terrain
    assert sorted(p.name for p in (tmp_path / 'runs').iterdir()) == [run.name]                       # no second run folder
    assert catalog_rows(tmp_path) == [(run.name, 's2')]


def test_cli_resume_refuses_build_terrain_and_a_folder_without_a_store(s1_small, terr, tmp_path, stubs, capsys):
    run = tmp_path / 'runs' / '20261021_c_ept_s2_1'
    run.mkdir(parents=True)
    assert cli(tmp_path, '--s1', str(s1_small), '--build-terrain', '--macros', 'FIT', '--resume', str(run)) != 0
    assert '--resume' in capsys.readouterr().err
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT', '--resume', str(run)) != 0
    assert 'no S2 store' in capsys.readouterr().err
    assert not stubs.build and not stubs.terrain and catalog_rows(tmp_path) == [] and list(run.iterdir()) == []


def test_cli_refuses_a_selection_without_matches(s1_small, terr, tmp_path, stubs, capsys):
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'UNUSED') != 0
    assert 'no ' in capsys.readouterr().err and not stubs.build and not (tmp_path / 'runs').exists()


def test_cli_build_terrain_writes_into_the_new_run_folder_and_a_given_terrain_stays_read_only(s1_small, terr, tmp_path, stubs):
    assert cli(tmp_path, '--s1', str(s1_small), '--build-terrain', '--macros', 'FIT', '--limit', '3') == 0
    call = stubs.build[0]
    assert stubs.terrain == [(s1_small, call['run'])] and call['terrain_path'] == call['run'] / 'terrain'
    assert call['matches'] == [0, 5, 10]
    before = snapshot(terr)
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT', '--limit', '3') == 0
    assert len(stubs.terrain) == 1 and stubs.build[1]['terrain_path'] == terr and snapshot(terr) == before
    assert sorted(catalog_rows(tmp_path)) == sorted((c['run'].name, 's2') for c in stubs.build)
    assert stubs.build[0]['run'] != stubs.build[1]['run'] and stubs.build[0]['run'].parent == stubs.build[1]['run'].parent


def test_cli_refuses_a_missing_s1_run_before_a_run_folder_is_made(tmp_path, stubs, terr, capsys):
    assert cli(tmp_path, '--s1', str(tmp_path / 'no_s1'), '--terrain', str(terr), '--macros', 'FIT') != 0
    assert 'S1' in capsys.readouterr().err and not (tmp_path / 'runs').exists() and not stubs.build


# ---- the command line and the final fix pass (A) -------------------------------------------------------------------------------------------
def test_cli_refuses_a_terrain_built_with_other_parameters_before_a_run_folder_is_made(s1_small, terr, tmp_path, stubs, capsys):
    """Final review I1: the terrain of the first smoke run (min_count 5) is refused at the command line, with the field named."""
    old = terrain_variant(terr, tmp_path / 'terrain_old', lambda info: info['grid'].update(REJECTED_SMOKE_GRID))
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(old), '--macros', 'FIT') == 2
    err = capsys.readouterr().err
    assert 'grid.min_count' in err and 'SYN_' not in err and f'{tmp_path.name}{os.sep}terrain_old' in err       # the folder is named with its parent
    assert not stubs.build and not stubs.terrain and not (tmp_path / 'runs').exists() and catalog_rows(tmp_path) == []
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT') == 0           # the terrain of the current parameters


def test_cli_resume_needs_the_run_json_of_the_run(s1_small, terr, tmp_path, stubs, capsys):
    """Final review M6: a run folder with a store and without RUN.json is refused before build_s2 is called."""
    run = tmp_path / 'runs' / '20261021_c_ept_s2_1'
    (run / 's2').mkdir(parents=True)
    (run / 's2' / 'STORE.json').write_text('{}', encoding='utf-8')
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT', '--resume', str(run)) == 2
    assert 'RUN.json' in capsys.readouterr().err and not stubs.build and catalog_rows(tmp_path) == []


def test_cli_tells_the_user_what_to_delete_when_the_run_is_locked(s1_small, terr, tmp_path, monkeypatch, capsys):
    """Final review M7: build_s2 refuses a locked run; the command line prints the message (no traceback) and registers nothing."""
    def locked(*args, **kwargs):
        raise store_build.RunLocked(f'another S2 build holds the lock {store_build.LOCK_NAME}: delete it if no build runs')
    monkeypatch.setattr(store_build, 'build_s2', locked)
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT') == 2
    assert store_build.LOCK_NAME in capsys.readouterr().err and catalog_rows(tmp_path) == []


def test_cli_tells_how_to_resume_after_a_model_server_stopped_the_build(s1_small, terr, tmp_path, monkeypatch, capsys):
    """Review recommendation 3: E6 stopped the build after a block; the command line says so, names the run folder and the resume
    command, and registers nothing."""
    def stopped(s1_run, terrain_path, matches, run, *args, **kwargs):
        (Path(run) / store_build.STORE_DIR).mkdir(parents=True)                 # the blocks done before the stop: a store was begun
        (Path(run) / store_build.STORE_DIR / 'STORE.json').write_text('{}', encoding='utf-8')
        raise run_s1.StrataRunning('E6: a Strata process is running. Found: python.exe (pid 4101). A model server started while the S2 '
                                   'build ran.')
    monkeypatch.setattr(store_build, 'build_s2', stopped)
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT', 'STOP', '--limit', '6',
               '--block-matches', '3', '--workers', '2') == 2
    err = capsys.readouterr().err
    run = next((tmp_path / 'runs').iterdir())
    assert 'E6' in err and 'pid 4101' in err and f'--resume {run}' in err and f'--terrain {terr}' in err
    assert '--macros FIT STOP' in err and '--limit 6' in err and '--block-matches 3' in err and '--workers 2' in err
    assert '--catalog-db' in err                                                  # the test uses another run index than the default
    assert catalog_rows(tmp_path) == []


def test_cli_says_to_run_the_command_again_when_a_model_server_stopped_the_build_before_any_store_was_begun(s1_small, terr, tmp_path,
                                                                                                        monkeypatch, capsys):
    def stopped(*args, **kwargs):
        raise run_s1.StrataRunning('E6: a Strata process is running. Found: python.exe (pid 4101).')
    monkeypatch.setattr(store_build, 'build_s2', stopped)
    assert cli(tmp_path, '--s1', str(s1_small), '--terrain', str(terr), '--macros', 'FIT') == 2
    err = capsys.readouterr().err
    assert 'pid 4101' in err and 'run the command again' in err and '--resume' not in err and catalog_rows(tmp_path) == []
