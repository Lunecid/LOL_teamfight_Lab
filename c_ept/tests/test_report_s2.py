r"""Tests of ``c_ept.report_s2`` on a small S2 run that these tests build by hand (R14-g).

The run has the layout that ``store_build`` writes -- ``<run>\s2`` with the arrays ``views/<column>`` (six columns, 27 bytes per view)
and ``x/<key>`` (every key of ``tokens.VIEW_SPEC``), RUN.json with ``s1``, ``terrain`` and ``s2_result``, and a terrain folder with
TERRAIN.json -- but it is made here, from synthetic values, by ``store_core.Store`` (the layout is written out below from the rulings R13-d,
R13-f, R13-o); only the integration test at the end of the file imports ``store_build`` and feeds a real ``build_s2`` output to the
report. Three smoke matches (S1 rows 0-2 of an eight-match synthetic S1 store), 42 views (24
frame-clock, 9 M3, 9 M4), with the cases the report must find planted in known places: events and edges that overflow, positions that are
missing, at exactly (0, 0) or off the map, unknown champions and items, NaN where ``tokens.NAN_ALLOWED`` allows it, constant columns, and a
level formula with known differences. Each expected number is worked out in the comment next to the planting, or by a short loop over the
planted cells.

No raw file, no run folder under ``D:\LOL_Data\runs`` and no catalog is read or written: the match keys are sentinel strings, the match
records of the raw-measurement tests are synthetic (``synth.make_record``), and every folder lives under pytest's temporary folders. The
terrain is the open 75 x 75 grid of ``conftest.py`` (the all-pairs table is built once per session) with a hand-made TERRAIN.json; the path
tests add a corridor map and a staircase map (cheap tables: few walkable cells). Run with ``--tb=short``.
"""
import gc
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from c_ept import config as C
from c_ept import reader, report_s2, run_s1, runlog, state, store_core, terrain, tokens, traits
from c_ept.checks import sweep_state
from c_ept.store_core import Store
from c_ept.tests import synth
from c_ept.tests import view_helpers as vh
from c_ept.tests.conftest import empty_process_list

P, S = vh.P, vh.S                                                # column name -> index in players_num / summary
RUN_NAME, S1_NAME = '20261021_c_ept_s2_1', '20261020_c_ept_s1_1'
VIEWS_COLUMNS = (('src', 'i1'), ('src_index', 'i4'), ('match', 'i4'), ('ref', 'i8'), ('t0', 'i8'), ('frame_k', 'i2'))      # R13-d, 27 bytes
GIB = 2 ** 30

# The synthetic S1 store: eight matches, and only the arrays that the report reads.
N_S1 = 8
S1_ROLES = ('FIT', 'FIT', 'FIT', 'STOP', 'DEV', 'AUDIT', 'CAL', 'FIT')       # the smoke run covers the first three; match 7 is FIT too
S1_PATCHES = ('15.18', '15.18', '15.19', '15.19', '15.20', '15.21', '15.21', '15.21')
S1_FRAME_ROWS = (8, 6, 10, 5, 4, 3, 2, 4)           # frame-clock rows of each match
S1_EVENT_ROWS = (6, 4, 8, 2, 0, 4, 2, 2)            # event rows (M3 and M4 interleaved) of each match
KEYS = [f'SENTINEL-KEY-{i:04d}' for i in range(N_S1)]            # the "match keys": they must never appear in any output
SMOKE = (0, 1, 2)                                   # the S1 matches of the smoke run: 8 + 6 + 10 frame rows, 6 + 4 + 8 event rows
N = 42                                              # views: 24 frame-clock + 18 event rows
PEAK = {'tree': 6.5, 'main': 1.25, 'worker': 0.9}   # GiB, as RUN.json records them

# planted cases (view numbers are positions in the views table; player numbers are rows 0-9 of players_num)
MISSING = [(1, 3), (5, 3), (20, 7), (41, 0)]                       # position missing: x, y NaN (4 of 420 player rows)
ZERO_ZERO = [(3, 2), (11, 0)]                                      # exactly (0, 0)
OFF_MAP = {(7, 4): (-50.0, 8000.0), (9, 1): (9000.0, 15000.5), (11, 5): (14875.5, 100.0)}         # outside [0, 14,875]^2
ON_EDGE = {(12, 9): (0.0, 5.0), (13, 8): (14875.0, 0.0)}           # inside: only one coordinate is 0, and 14,875 is on the map
UNKNOWN_CHAMP = [(v, 9) for v in range(24, 34)] + [(34, 9), (35, 9)]    # 12 rows: players_num traits NaN, champ_unknown 1
UNKNOWN_ITEMS = {(v, 4): 1 for v in range(20)} | {(40, 4): 2}      # 21 rows with an unknown item (22 items in all)
EV_OVER = {2: 5, 9: 12, 30: 3}                                     # views whose MAX_EVENTS slots overflowed -> ev_dropped (3 of 42 = 7.1 %)
ED_OVER = {9: 7, 33: 20}                                           # views whose 128 edge slots overflowed -> ed_dropped (2 of 42 = 4.8 %)
PATTERN = (-5, -3, -2, -1, 0, 0, 1, 2, 3, 5)                       # lvl + item - cs of frame-clock view f, player p: [(f + p) % 10]
CONST_SUMMARY = {'atakhan_in_patch': 1.0, **{n: 0.0 for n in state.OBJ_STATE_NAMES if n.startswith('rule_uncertain_')}}  # + voidgrub: NaN
ITEM_AD, LVL_AD = 20.25, 60.5                       # the item and level terms of the formula, the same in every row (level: NaN in some)
CONST_PLAYERS = {'cs_spellVamp': 0.0, 'cs_omnivamp': 0.0, 'item_attackDamage': ITEM_AD}


GRID_RECORD = {'cells': 75, 'cell_u': 200, 'histogram_cells': 150, 'histogram_u': 100, 'min_count': terrain.MIN_COUNT}     # as build_terrain writes it


def current_zone_rules() -> dict:
    """``terrain.ZONE_RULES`` as TERRAIN.json holds them (a list where the rules have a tuple)."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in terrain.ZONE_RULES.items()}


def ref_of(m: int, k: int) -> int:
    """The time of regular frame k of smoke match m: a little late, like a real timeline (the 11 ms per match tell the matches apart)."""
    return 60_000 * k + 300 + 11 * m


def make_table(order=SMOKE) -> dict:
    """The views table of the three smoke matches, in the order ``order``: per match its frame-clock rows (frames 2 ... ), then its event
    rows (M3, M4, M3, ...)."""
    row_off = np.concatenate([[0], np.cumsum(S1_FRAME_ROWS)])
    event_off = np.concatenate([[0], np.cumsum(S1_EVENT_ROWS)])
    rows = []
    for m in order:
        for j in range(S1_FRAME_ROWS[m]):
            t = ref_of(m, j + 2)
            rows.append((0, int(row_off[m]) + j, m, t, t, j + 2))
        for e in range(S1_EVENT_ROWS[m]):
            t0 = ref_of(m, e + 3)
            rows.append((1 + e % 2, int(event_off[m]) + e, m, t0 + 4_000 + 5_000 * e, t0, e + 3))        # the frame is 4 + 5 e seconds old
    return {c: np.array(v, dtype=d) for (c, d), v in zip(VIEWS_COLUMNS, zip(*rows))}


def drop_position(x: dict, v: int, p: int) -> None:
    """Player p of view v has no position, with everything that follows from it (tokens.NAN_ALLOWED, 'when')."""
    pl = x['players_num']
    pl[v, p, [P['x'], P['y'], P['wall_fraction']]] = np.nan
    x['players_zone'][v, p] = -1
    x['geo_pp'][v, p, :] = x['geo_pp'][v, :, p] = np.nan
    x['geo_pl'][v, p, :] = np.nan


def make_x(table: dict) -> dict:
    """The 20 x/ arrays of the 42 views, filled so that every number the report gives is known (see the planted cases above)."""
    rng = np.random.default_rng(14)
    x = {}
    for key, (shape, dtype) in tokens.VIEW_SPEC.items():
        x[key] = (rng.uniform(1.0, 100.0, (N, *shape)).astype(dtype) if dtype.kind == 'f' else np.zeros((N, *shape), dtype=dtype))
    ref, t0, src, frame_k = (table[c] for c in ('ref', 't0', 'src', 'frame_k'))
    pl = x['players_num']
    pl[:, :, P['x']] = rng.uniform(500.0, 14000.0, (N, 10))
    pl[:, :, P['y']] = rng.uniform(500.0, 14000.0, (N, 10))
    pl[:, :, P['level']] = (1 + frame_k // 3)[:, None]                               # the level of the frame: 1 + k // 3 for every player
    pl[:, :, P['has_prev']] = 1
    pl[:, :, [P['prev_x'], P['prev_y']]] = pl[:, :, [P['x'], P['y']]] - 20.0
    pl[:, :, P['wall_fraction']] = rng.uniform(0.0, 1.0, (N, 10))
    pl[:, :, P['champ_unknown']] = 0
    pl[:, :, P['unknown_item_count']] = 0
    for name, value in CONST_PLAYERS.items():
        pl[:, :, P[name]] = value
    x['players_zone'][:] = rng.integers(0, 10, (N, 10))
    x['landmarks'][:, :, 0] = rng.integers(0, 2, (N, 34))
    # level formula: lvl + item - cs = PATTERN[(f + p) % 10] on the frame-clock views (f = 0 ... 23 among them), 100 on the others
    frame_views = np.flatnonzero(src == 0)
    diff = np.full((N, 10), 100.0)
    diff[frame_views] = np.array(PATTERN)[(np.arange(len(frame_views))[:, None] + np.arange(10)[None, :]) % 10]
    pl[:, :, P['lvl_attackDamage']] = LVL_AD                                       # (the item term is one of the constants set above)
    pl[:, :, P['cs_attackDamage']] = LVL_AD + ITEM_AD - diff                       # multiples of 0.25: exact in float32
    # positions: (0, 0), off the map, on the edge, missing
    for v, p in ZERO_ZERO:
        pl[v, p, [P['x'], P['y']]] = 0.0
    for (v, p), (px, py) in {**OFF_MAP, **ON_EDGE}.items():
        pl[v, p, [P['x'], P['y']]] = (px, py)
    for v, p in MISSING:
        drop_position(x, v, p)
    pl[0, :, P['has_prev']] = 0                                                       # the first view has no previous frame: 10 rows
    pl[0, :, P['prev_x']] = np.nan
    pl[0, :, P['prev_y']] = np.nan
    # unknown champions and items
    trait_columns = [P[n] for n in traits.TRAIT_NAMES if n != 'champ_unknown']
    for v, p in UNKNOWN_CHAMP:
        pl[v, p, trait_columns] = np.nan
        pl[v, p, P['champ_unknown']] = 1
    for (v, p), count in UNKNOWN_ITEMS.items():
        pl[v, p, P['unknown_item_count']] = count
    # tokens and edges: view v has 3 + 7 v mod 40 tokens and 5 v mod 100 edges, unless it overflowed
    ev_n = 3 + (np.arange(N) * 7) % 40
    ed_n = (np.arange(N) * 5) % 100
    for v, dropped in EV_OVER.items():
        ev_n[v] = C.MAX_EVENTS
        x['ev_dropped'][v] = dropped
    for v, dropped in ED_OVER.items():
        ed_n[v] = C.MAX_EDGES
        x['ed_dropped'][v] = dropped
    x['ev_n'][:], x['ed_n'][:] = ev_n, ed_n
    for v in range(N):
        n, m = int(ev_n[v]), int(ed_n[v])
        x['ev_type'][v, :] = -1
        x['ev_type'][v, :n] = (v + np.arange(n)) % 8
        x['ev_team'][v, :] = 2
        x['ev_team'][v, :n] = np.arange(n) % 2
        x['ev_dt'][v, :] = 0.0
        x['ev_dt'][v, :n] = np.arange(n) * 1.5
        x['ev_gold'][v, n:] = 0.0
        x['ev_xy'][v, n:, :] = np.nan
        x['ed_i'][v, :], x['ed_j'][v, :], x['ed_kind'][v, :] = -1, -1, -1
        x['ed_i'][v, :m], x['ed_j'][v, :m], x['ed_kind'][v, :m] = 0, 1, np.arange(m) % 5
        x['ed_dt'][v, :] = 0.0
        x['ed_dt'][v, :m] = np.arange(m) * 0.5
    # summary: the clocks, the constants, the NaN cells of the objective state, and random values elsewhere
    sm = x['summary']
    sm[:, S['game_time_s']] = sm[:, S['t_s']] = ref / 1000.0
    sm[:, S['frame_age_s']] = (ref - t0) / 1000.0
    for name, value in CONST_SUMMARY.items():
        sm[:, S[name]] = value
    sm[:10, S['herald_next_spawn_s']] = np.nan                                        # 10 of 42 views
    sm[:, S['voidgrub_next_spawn_s']] = np.nan                                        # all 42: a constant column
    x['landmarks'][:10, 30, 1] = np.nan                                               # the herald seconds cell: 10 of 42 views
    x['landmarks'][:21, 31, 1] = np.nan                                               # the grubs seconds cell: 21 of 42 views
    return x


def sha_of(path) -> str:
    return store_core.sha(path)


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


@pytest.fixture(scope='module')
def s1_dir(tmp_path_factory):
    """The synthetic S1 run folder ``<tmp>/20261020_c_ept_s1_1`` (the store is ``s1`` below it): the arrays the report reads, complete."""
    run = tmp_path_factory.mktemp('report_s2_s1') / S1_NAME
    s = Store(run / 's1')
    row_off = np.concatenate([[0], np.cumsum(S1_FRAME_ROWS)])
    event_off = np.concatenate([[0], np.cumsum(S1_EVENT_ROWS)])
    values = {'matches/key': (np.array([k.encode('ascii') for k in KEYS], dtype='S24'), 'S24'),
              'matches/patch': (np.array([C.PATCHES.index(p) for p in S1_PATCHES]), 'i1'),
              'matches/role': (np.array([C.ROLES.index(r) for r in S1_ROLES]), 'i1'),
              'matches/row_off': (row_off, 'i8'), 'matches/event_off': (event_off, 'i8')}
    for name, (v, dtype) in values.items():
        a = s.allocate(name, v.shape, dtype)
        a[...] = v
        a.flush()
        del a
    s.finalize()
    yield run
    gc.collect()
    shutil.rmtree(run.parent, ignore_errors=True)


def write_terrain_folder(out, open_table, s1_dir, fit_matches) -> Path:
    """A terrain folder with provenance: the open grid, a made-up occupancy, a hand-made TERRAIN.json whose hashes are right and whose
    parameters (grid record and zone rules) are those of the current code, as ``terrain.build_terrain`` writes them. ``s1_dir`` is the
    S1 run folder the terrain claims to come from."""
    rng = np.random.default_rng(3)
    frames100, kills100 = rng.poisson(40.0, (150, 150)).astype(float), rng.poisson(2.0, (150, 150)).astype(float)
    hashes = terrain.write_terrain_files(out, np.ones((75, 75), dtype=bool), frames100, kills100, open_table)
    census = lambda n, nan, zz, off, nb: dict(n=n, nan=nan, zero_zero=zz, outside_map=off, not_binned=nb)          # noqa: E731
    info = {'version': 1, 's1_run': Path(s1_dir).name, 's1_files_sha256': runlog.sha256_json(Store(Path(s1_dir) / 's1').manifest['files']),
            'fit_matches': fit_matches, 'grid': dict(GRID_RECORD), 'zone_rules': current_zone_rules(),
            'walkable_cells': 5625,
            'components': {'count': 1, 'largest_cells': 5625, 'outside_largest': 0},
            'positions': {'regular_frames': census(1000, 4, 1, 2, 6), 'game_end_frame': census(30, 0, 0, 0, 0),
                          'kills': census(77, 3, 0, 1, 4)},
            'code_commit': 'a' * 40, 'code_dirty': False, 'sha256': hashes}
    (Path(out) / 'TERRAIN.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return Path(out)


@pytest.fixture(scope='module')
def terrain_base(open_table, s1_dir, tmp_path_factory):
    """The terrain folder of the hand-made runs (see ``write_terrain_folder``)."""
    out = write_terrain_folder(tmp_path_factory.mktemp('report_s2_terrain') / 'terrain', open_table, s1_dir, 3)
    yield out
    gc.collect()
    shutil.rmtree(out.parent, ignore_errors=True)


class Run:
    """A built S2 run: ``path`` (the run folder), ``x`` and ``table`` (the arrays that were written) and the S1 run folder."""

    def __init__(self, path, x, table, s1):
        self.path, self.x, self.table, self.s1 = path, x, table, s1


def run_json_for(s1_dir, table, terrain_dir, edit=None) -> dict:
    """RUN.json as ``store_build`` writes it: provenance fields, ``args``, ``s1``, ``terrain`` and (at the end) ``s2_result``."""
    by_src = np.bincount(table['src'], minlength=3)
    x_bytes = tokens.bytes_per_view()
    s1_files = runlog.sha256_json(Store(s1_dir / 's1').manifest['files'])
    doc = {
        'stage': 's2', 'run': RUN_NAME, 'created': '2026-10-21T09:00:00+00:00', 'code_commit': 'a' * 40, 'code_dirty': False,
        'gpu': 'NVIDIA GeForce RTX 5070 Ti, 1431 MiB, 16303 MiB', 'strata_running': False,
        'args': {'workers': 2, 'block_matches': 250, 'n_matches': 3, 'plan_sha256': 'b' * 64, 'macros': ['FIT'], 'limit': 3},
        's1': {'run': S1_NAME, 'files_sha256': s1_files},
        'terrain': {'sha256': {**{n: sha_of(terrain_dir / n) for n in ('terrain.npz', 'geodesic.npy')},
                               'TERRAIN.json': sha_of(terrain_dir / 'TERRAIN.json')},
                    's1_run': S1_NAME, 's1_files_sha256': s1_files, 'built_from_this_s1': True, 'grid': dict(GRID_RECORD)},
        's2_result': {'complete': True, 'wall_s': 12.5, 'stage_s': {'allocate': 0.1, 'views': 12.0, 'finalize': 0.4},
                      'peak_private_gib': PEAK['tree'], 'peak_private_main_gib': PEAK['main'], 'peak_private_worker_gib': PEAK['worker'],
                      'memory_samples': 9, 'workers': 2, 'block_matches': 250, 'n_blocks': 1, 'blocks_resumed': 0, 'matches': 3,
                      'matches_built': 3, 'seconds_per_match': 4.0, 'views': N,
                      'views_by_src': {'frame_clock': int(by_src[0]), 'M3': int(by_src[1]), 'M4': int(by_src[2])},
                      'x_bytes_per_view': x_bytes, 'views_bytes_per_view': 27, 'bytes_per_view': x_bytes + 27,
                      'store_files': 26, 'store_bytes': 0}}                       # 26 stored arrays (STORE.json is not counted)
    if edit is not None:
        edit(doc)
    return doc


def build_run(tmp_path, s1_dir, terrain_base, edit_x=None, edit_run=None, name=RUN_NAME, order=SMOKE, edit_terrain=None) -> Run:
    """Write the small S2 run under ``tmp_path``: the store, RUN.json and a terrain folder (hard links to the module's files).
    ``edit_x(x, table)`` changes the arrays before they are written, ``edit_run(doc)`` RUN.json before it is, ``edit_terrain(info)`` the
    parsed TERRAIN.json of the run's own terrain folder (the file is replaced, so the shared one stays as it is; RUN.json then records
    the hash of the edited file, as a build of that terrain would have)."""
    run = tmp_path / name
    table = make_table(order)
    x = make_x(table)
    if edit_x is not None:
        edit_x(x, table)
    s = Store(run / 's2')
    layout = {f'views/{c}': ((N,), np.dtype(d)) for c, d in VIEWS_COLUMNS}
    layout.update({f'x/{k}': ((N, *shape), np.dtype(dt)) for k, (shape, dt) in tokens.VIEW_SPEC.items()})
    for array, (shape, dtype) in layout.items():
        a = s.allocate(array, shape, dtype)
        group, column = array.split('/')
        a[...] = table[column] if group == 'views' else x[column]
        a.flush()
        del a
    s.set_stage('views', block_matches=250, n_matches=3, n_views=N)
    s.finalize()
    (run / 'terrain').mkdir()
    for f in terrain_base.iterdir():
        link_or_copy(f, run / 'terrain' / f.name)
    if edit_terrain is not None:
        path = run / 'terrain' / 'TERRAIN.json'
        info = json.loads(path.read_text(encoding='utf-8'))
        edit_terrain(info)
        path.unlink()                                                      # the name was a hard link to the shared file: write a new one
        path.write_text(json.dumps(info, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    doc = run_json_for(s1_dir, table, run / 'terrain', edit_run)
    (run / 'RUN.json').write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
    return Run(run, x, table, s1_dir)


def rep(r: Run, **kw) -> dict:
    """Run the report of a built run and return the parsed S2_REPORT.json."""
    path = report_s2.report(r.path, s1=r.s1, **kw)
    assert path == r.path / 'S2_REPORT.json'
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture
def small_s2_run(tmp_path, s1_dir, terrain_base):
    return build_run(tmp_path, s1_dir, terrain_base)


@pytest.fixture(scope='module')
def nominal(tmp_path_factory, s1_dir, terrain_base):
    """The nominal run and its report, made once: the tests that only read the report share it."""
    r = build_run(tmp_path_factory.mktemp('report_s2_nominal'), s1_dir, terrain_base)
    doc = rep(r)
    yield r, doc
    gc.collect()
    shutil.rmtree(r.path.parent, ignore_errors=True)


# ---- the brief's test ---------------------------------------------------------------------------------------------------------------
def test_report_fields(tmp_path, small_s2_run):
    path = report_s2.report(small_s2_run.path, s1=small_s2_run.s1)
    r = json.loads(path.read_text(encoding='utf-8'))
    assert {'views', 'bytes_per_view', 'seconds_per_match', 'peak_private_gib', 'event_overflow_share', 'edge_overflow_share',
            'unknown_item_share', 'unknown_champion_share', 'level_formula_check', 'extrapolated_gib'} <= set(r)
    assert {'suspect_positions', 'nan_share', 'constant_columns', 'level_lag', 'building_kill_matching', 'views_by_src'} <= set(r)


def flag_codes(doc: dict) -> list:
    return [f['code'] for f in doc['flags']]


# the overflow that ``make_x`` plants: events in 3 of 42 views (7.1 %, over 5 %), edges in 2 (4.8 %), either in 4 (9.5 %, over 5 %)
OVERFLOW_FLAGS = ['EVENT_OVERFLOW_OVER_5_PERCENT', 'EVENTS_OR_EDGES_OVERFLOW_OVER_5_PERCENT']


# ---- sizes, memory, overflow ----------------------------------------------------------------------------------------------------------
def test_views_matches_and_bytes_per_view(nominal):
    r, doc = nominal
    assert doc['views'] == N == 42
    assert doc['views_by_src'] == {'frame_clock': 24, 'M3': 9, 'M4': 9}              # per match the event rows alternate M3, M4, M3, ...
    assert doc['matches'] == 3 and doc['matches_with_views'] == 3 and doc['views_per_match'] == 14.0
    x_bytes = tokens.bytes_per_view()
    assert doc['bytes_per_view'] == x_bytes + 27                                      # R14-i: the total ...
    assert doc['bytes_per_view_parts'] == {'x': x_bytes, 'views_table': 27}           # ... and the two parts
    store = Store(r.path / 's2')
    assert doc['store'] == {'complete': True, 'verified': True, 'files': 26, 'bytes': sum(p.stat().st_size for p in store.stored_files())}


def test_the_run_record_is_carried_into_the_report(nominal):
    r, doc = nominal
    assert doc['seconds_per_match'] == 4.0 and doc['wall_s'] == 12.5
    assert (doc['peak_private_gib'], doc['peak_private_main_gib'], doc['peak_private_worker_gib']) == (6.5, 1.25, 0.9)
    assert doc['memory_gate'] == {'limit_gib': 48.0, 'peak_private_gib': 6.5, 'within_limit': True}
    assert doc['run']['name'] == RUN_NAME and doc['run']['strata_running'] is False and 'RTX' in doc['run']['gpu_at_start']
    assert doc['run']['code_commit'] == 'a' * 40 and doc['run']['args'] == {'workers': 2, 'block_matches': 250, 'n_matches': 3,
                                                                          'macros': ['FIT'], 'limit': 3}
    assert doc['s1_run'] == S1_NAME
    assert doc['smoke'] == {'matches_by_patch': {'15.18': 2, '15.19': 1}, 'matches_by_macro': {'FIT': 3}}
    assert (doc['run']['calls'], doc['run']['recovered'], doc['run']['warnings'], doc['run']['per_call']) == (None, None, [], None)
    assert doc['run']['code_state_at_end'] is None                                # the committed layout has none of the fix round's fields


def later_layout(doc):
    """RUN.json as the Task 13 fix round writes it: the command line's info in ``args.info`` (next to computed keys that it can never
    overwrite), and ``calls``, ``recovered``, ``warnings`` and ``per_call`` in ``s2_result`` (a build of two calls here)."""
    args = doc['args']
    args['info'] = {'macros': args.pop('macros'), 'limit': args.pop('limit'), 'workers': 99, 'plan_sha256': 'not the plan'}
    per_call = [{'workers': 2, 'blocks_built': 1, 'matches_built': 1, 'peak_private_gib': 6.5, 'peak_private_main_gib': 1.25,
                 'peak_private_worker_gib': 0.9, 'code_commit': 'a' * 40, 'code_dirty': False, 'warnings': []},
                {'workers': 1, 'blocks_built': 1, 'matches_built': 2, 'peak_private_gib': 3.0, 'peak_private_main_gib': 3.0,
                 'peak_private_worker_gib': None, 'code_commit': 'a' * 40, 'code_dirty': False, 'warnings': []}]
    doc['s2_result'].update(calls=2, recovered=False, warnings=[], per_call=per_call,
                            code_state_at_end={'code_commit': 'a' * 40, 'code_dirty': False})


def test_the_run_record_is_read_in_the_layout_of_the_fix_round_too(tmp_path, s1_dir, terrain_base):
    doc = rep(build_run(tmp_path, s1_dir, terrain_base, edit_run=later_layout))
    assert doc['run']['args'] == {'workers': 2, 'block_matches': 250, 'n_matches': 3, 'macros': ['FIT'], 'limit': 3}   # info never wins
    assert (doc['run']['calls'], doc['run']['recovered'], doc['run']['warnings']) == (2, False, [])
    assert doc['run']['code_state_at_end'] == {'code_commit': 'a' * 40, 'code_dirty': False}
    assert [c['workers'] for c in doc['run']['per_call']] == [2, 1] and doc['run']['per_call'][1]['peak_private_worker_gib'] is None
    assert flag_codes(doc) == OVERFLOW_FLAGS                                                      # a clean two-call build adds no flag


def test_warnings_of_the_build_and_a_repaired_run_record_are_flagged(tmp_path, s1_dir, terrain_base):
    def edit(doc):
        later_layout(doc)
        doc['s2_result'].update(recovered=True, warnings=['a resume on other code (aaaaaaaaaaaa -> bbbbbbbbbbbb)'])
    doc = rep(build_run(tmp_path, s1_dir, terrain_base, edit_run=edit))
    assert {'BUILD_WARNINGS', 'RUN_RECORD_RECOVERED'} <= set(flag_codes(doc)) and doc['run']['recovered'] is True
    warning = next(f for f in doc['flags'] if f['code'] == 'BUILD_WARNINGS')
    assert 'a resume on other code' in warning['message'] and '1 warning(s)' in warning['message']


@pytest.mark.parametrize('peak, over', [(48.0, False), (48.001, True), (49.5, True)])
def test_the_48_gib_gate_is_a_flag_above_the_limit_only(tmp_path, s1_dir, terrain_base, peak, over):
    r = build_run(tmp_path, s1_dir, terrain_base, edit_run=lambda d: d['s2_result'].update(peak_private_gib=peak))
    doc = rep(r)
    assert doc['memory_gate'] == {'limit_gib': 48.0, 'peak_private_gib': peak, 'within_limit': not over}
    assert ('PEAK_MEMORY_OVER_LIMIT' in flag_codes(doc)) is over


def test_an_unrecorded_peak_cannot_pass_the_gate(tmp_path, s1_dir, terrain_base):
    r = build_run(tmp_path, s1_dir, terrain_base, edit_run=lambda d: d['s2_result'].update(peak_private_gib=None))
    doc = rep(r)
    assert doc['memory_gate']['within_limit'] is None and 'PEAK_MEMORY_NOT_RECORDED' in flag_codes(doc)


def test_a_build_with_a_running_model_server_is_flagged(tmp_path, s1_dir, terrain_base):
    r = build_run(tmp_path, s1_dir, terrain_base, edit_run=lambda d: d.update(strata_running=True))
    assert 'STRATA_WAS_RUNNING_AT_BUILD' in flag_codes(rep(r))


def test_overflow_shares_and_the_five_percent_flags(nominal):
    r, doc = nominal
    assert doc['event_overflow_share'] == pytest.approx(3 / 42)                         # views 2, 9, 30
    assert doc['edge_overflow_share'] == pytest.approx(2 / 42)                          # views 9, 33
    assert flag_codes(doc) == OVERFLOW_FLAGS            # events 7.1 % is more than 5 %, edges 4.8 % is not; either of them (9.5 %) is
    o = doc['overflow']
    assert o['caps'] == {'events': C.MAX_EVENTS, 'edges': C.MAX_EDGES} and o['limit_share'] == 0.05
    ev, ed = o['events'], o['edges']
    assert (ev['views_with_dropped'], ev['dropped_total'], ev['max_stored']) == (3, 20, C.MAX_EVENTS)    # 5 + 12 + 3 events did not fit
    assert (ed['views_with_dropped'], ed['dropped_total'], ed['max_stored']) == (2, 27, C.MAX_EDGES)     # 7 + 20 edges
    assert ev['window_total']['max'] == C.MAX_EVENTS + 12 and ed['window_total']['max'] == C.MAX_EDGES + 20      # stored + dropped
    assert ev['by_src'] == {'frame_clock': 2, 'M3': 0, 'M4': 1} and ed['by_src'] == {'frame_clock': 1, 'M3': 0, 'M4': 1}
    # the smallest cap that keeps the share at or under 5 % (2 of 42 views may overflow): the third largest window total. It is written
    # only where the share is over 5 % (events, 7.1 %); where it is not (edges, 4.8 %) a smaller cap than the present one would be a
    # proposal for nothing: None, with a note that says so (final review, Task 14 minor)
    assert ev['cap_for_5_percent'] == C.MAX_EVENTS + 3 and ed['cap_for_5_percent'] is None
    assert 'smallest cap' in ev['cap_for_5_percent_note'] and 'no cap is changed' in ev['cap_for_5_percent_note']
    assert 'None' in ed['cap_for_5_percent_note'] and 'at most 5 %' in ed['cap_for_5_percent_note']
    flag = next(f for f in doc['flags'] if f['code'] == 'EVENT_OVERFLOW_OVER_5_PERCENT')
    assert str(C.MAX_EVENTS + 3) in flag['message'] and 'not changed' in flag['message']       # a proposal, not a change
    assert o['events_or_edges'] == {'views_with_dropped': 4, 'share': pytest.approx(4 / 42)}      # views 2, 9, 30, 33 (view 9 drops both)
    either = next(f for f in doc['flags'] if f['code'] == 'EVENTS_OR_EDGES_OVERFLOW_OVER_5_PERCENT')
    assert '9.5%' in either['message'] and '4 of 42' in either['message']


def test_edge_overflow_over_five_percent_is_flagged_with_a_cap(tmp_path, s1_dir, terrain_base):
    def edit(x, table):
        for v in (0, 1, 2):                                      # 3 more views whose 128 edge slots overflowed: 5 of 42 = 11.9 %
            x['ed_n'][v], x['ed_dropped'][v] = C.MAX_EDGES, 9
    doc = rep(build_run(tmp_path, s1_dir, terrain_base, edit_x=edit))
    assert doc['edge_overflow_share'] == pytest.approx(5 / 42) and 'EDGE_OVERFLOW_OVER_5_PERCENT' in flag_codes(doc)
    flag = next(f for f in doc['flags'] if f['code'] == 'EDGE_OVERFLOW_OVER_5_PERCENT')
    assert f'{doc["overflow"]["edges"]["cap_for_5_percent"]}' in flag['message'] and doc['overflow']['edges']['cap_for_5_percent'] == 137
    assert doc['overflow']['edges']['cap_for_5_percent_note'].startswith('the smallest cap')            # the flag fired: a proposal is written


def test_events_and_edges_that_overflow_in_different_views_are_flagged_together_above_five_percent(tmp_path, s1_dir, terrain_base, capsys):
    """Review minor (Task 14): R14-e asks for a report when more than 5 % of the views drop events OR edges. Here 2 of 42 views drop events
    (4.8 %) and 2 other views drop edges (4.8 %): neither share is over 5 %, but 4 of 42 views (9.5 %) are."""
    def edit(x, table):
        for v in EV_OVER:
            x['ev_dropped'][v], x['ev_n'][v] = 0, 10
        for v in ED_OVER:
            x['ed_dropped'][v], x['ed_n'][v] = 0, 10
        for v in (2, 9):
            x['ev_n'][v], x['ev_dropped'][v] = C.MAX_EVENTS, 4
        for v in (20, 21):
            x['ed_n'][v], x['ed_dropped'][v] = C.MAX_EDGES, 6
    r = build_run(tmp_path, s1_dir, terrain_base, edit_x=edit)
    doc = rep(r)
    assert flag_codes(doc) == ['EVENTS_OR_EDGES_OVERFLOW_OVER_5_PERCENT']
    assert (doc['event_overflow_share'], doc['edge_overflow_share']) == (pytest.approx(2 / 42), pytest.approx(2 / 42))
    assert doc['overflow']['events_or_edges'] == {'views_with_dropped': 4, 'share': pytest.approx(4 / 42)}
    assert doc['overflow']['events']['cap_for_5_percent'] is None and doc['overflow']['edges']['cap_for_5_percent'] is None
    message = doc['flags'][0]['message']
    assert '9.5%' in message and '4 of 42' in message and '2 drop events' in message and '2 drop edges' in message
    assert report_s2.main([str(r.path), '--s1', str(r.s1), '--no-matches']) == 0                  # and the command line prints the share
    out = capsys.readouterr().out
    assert 'events or edges 9.52% (4 of 42 views)' in out and 'FLAG EVENTS_OR_EDGES_OVERFLOW_OVER_5_PERCENT' in out


def test_a_terrain_that_was_not_built_from_this_s1_is_flagged(tmp_path, s1_dir, terrain_base):
    doc = rep(build_run(tmp_path, s1_dir, terrain_base, edit_run=put('terrain', 'built_from_this_s1', value=False)))
    assert 'TERRAIN_NOT_FROM_THIS_S1' in flag_codes(doc) and doc['terrain']['built_from_this_s1'] is False
    assert rep(build_run(tmp_path / 'again', s1_dir, terrain_base))['terrain']['built_from_this_s1'] is True


def test_a_run_without_findings_has_no_flags(tmp_path, s1_dir, terrain_base):
    def clean(x, table):
        for v in EV_OVER:
            x['ev_dropped'][v] = 0
            x['ev_n'][v] = 10
    doc = rep(build_run(tmp_path, s1_dir, terrain_base, edit_x=clean))
    assert doc['flags'] == [] and doc['event_overflow_share'] == 0.0


def test_the_limit_is_compared_with_integers_and_means_more_than_five_percent():
    assert report_s2.over_share(2, 40) is False and report_s2.over_share(3, 41) is True and report_s2.over_share(0, 0) is False
    assert report_s2.over_share(3, 60) is False and report_s2.over_share(4, 60) is True


# ---- unknown champions and items; the level formula -------------------------------------------------------------------------------------
def test_unknown_champion_and_item_shares(nominal):
    r, doc = nominal
    assert doc['unknown_champion_share'] == pytest.approx(12 / 420)
    assert doc['unknown_item_share'] == pytest.approx(21 / 420) == pytest.approx(0.05)
    assert doc['unknown'] == {'player_rows': 420, 'rows_with_unknown_champion': 12, 'rows_with_unknown_item': 21,
                              'unknown_items_total': 22, 'views_with_unknown_champion': 12, 'views_with_unknown_item': 21}


def test_level_formula_check_uses_frame_clock_views_only_and_skips_unknown_champions(nominal):
    r, doc = nominal
    check = doc['level_formula_check']
    # 240 player-frames. Out: the 10 of an unknown champion (frame-clock views 14-23, player 9: NaN traits) and the 14 whose player owns an
    # item that the item table lacks (frame-clock views 0-13, player 4: the item attack damage is short by that item). Final review minor.
    diffs = np.array([PATTERN[(f + p) % 10] for f in range(24) for p in range(10) if not (f >= 14 and p == 9) and not (f < 14 and p == 4)],
                     dtype=float)
    assert len(diffs) == 216
    assert check['player_frames'] == 216 and check['nan_excluded'] == 10 and check['unknown_item_excluded'] == 14 and check['views'] == 24
    assert check['player_frames'] + check['nan_excluded'] + check['unknown_item_excluded'] == 10 * check['views']      # nothing counted twice
    assert check['median'] == pytest.approx(np.median(diffs))
    assert check['p10'] == pytest.approx(np.percentile(diffs, 10)) and check['p90'] == pytest.approx(np.percentile(diffs, 90))
    assert check['share_within_1'] == pytest.approx(np.mean(np.abs(diffs) <= 1))
    assert check['formula'] == 'lvl_attackDamage + item_attackDamage - cs_attackDamage'
    assert set(check['fields_used']) == {'lvl_attackDamage', 'item_attackDamage', 'cs_attackDamage'}
    assert 'championStats' in check['fields_used']['cs_attackDamage'] and 'frame' in check['fields_used']['cs_attackDamage']
    assert check['t0_offset_from_whole_minute_ms'] == {'median': 311.0, 'max': 322}   # 300 + 11 m ms after the minute
    assert 'sanity check' in check['note']


def test_the_level_formula_difference_has_the_sign_of_the_formula(tmp_path, s1_dir, terrain_base):
    """cs lower by 3 than lvl + item in every row -> a difference of +3 everywhere (so the sign is lvl + item - cs, not the reverse)."""
    def edit(x, table):
        pl = x['players_num']
        pl[:, :, P['cs_attackDamage']] = LVL_AD + ITEM_AD - 3.0
        pl[:, :, P['lvl_attackDamage']] = LVL_AD
    check = rep(build_run(tmp_path, s1_dir, terrain_base, edit_x=edit))['level_formula_check']
    assert (check['median'], check['p10'], check['p90'], check['share_within_1']) == (3.0, 3.0, 3.0, 0.0)


def test_rows_with_an_unknown_item_are_left_out_of_the_level_formula_and_counted(tmp_path, s1_dir, terrain_base):
    """An item that the table lacks adds nothing to ``item_attackDamage``, so the difference of such a row is wrong by that item's attack
    damage: the row is left out and counted. Here two rows have an unknown item and a difference of +250 (every other row: +3)."""
    def edit(x, table):
        pl = x['players_num']
        pl[:, :, P['cs_attackDamage']] = LVL_AD + ITEM_AD - 3.0
        pl[:, :, P['unknown_item_count']] = 0
        for v, p, count in ((3, 2, 1), (5, 7, 2)):                                          # frame-clock views of the first match
            pl[v, p, P['unknown_item_count']] = count
            pl[v, p, P['cs_attackDamage']] = LVL_AD + ITEM_AD - 250.0
    check = rep(build_run(tmp_path, s1_dir, terrain_base, edit_x=edit))['level_formula_check']
    assert check['unknown_item_excluded'] == 2 and check['nan_excluded'] == 10 and check['player_frames'] == 240 - 10 - 2
    assert (check['median'], check['p10'], check['p90']) == (3.0, 3.0, 3.0) and 'unknown item' in check['note']


# ---- extrapolation ------------------------------------------------------------------------------------------------------------------
def test_extrapolated_gib_counts_the_s1_store_and_shows_the_arithmetic(nominal):
    r, doc = nominal
    e = doc['extrapolated_gib']
    per_view = tokens.bytes_per_view() + 27
    # S1 rows per macro (matches, frame rows, event rows): FIT = matches 0-2 and 7, STOP = 3, DEV = 4, AUDIT = 5, CAL = 6
    want = {'FIT': (4, 28, 20), 'STOP': (1, 5, 2), 'DEV': (1, 4, 0), 'CAL': (1, 2, 2), 'AUDIT': (1, 3, 4)}
    full = e['detail']['cache_full_build']
    assert full['macros'] == ['FIT', 'STOP', 'DEV', 'CAL', 'AUDIT']
    for macro, (m, f, ev) in want.items():
        assert full['by_macro'][macro] == {'matches': m, 'frame_rows': f, 'event_rows': ev, 'views': f + ev,
                                           'gib': pytest.approx((f + ev) * per_view / GIB)}
    assert full['views'] == 48 + 7 + 4 + 4 + 7 == 70
    assert e['cache_full_build'] == pytest.approx(70 * per_view / GIB) == pytest.approx(full['gib'])
    assert f'70 views x {per_view:,} bytes' in full['arithmetic'] and '2^30' in full['arithmetic']
    assert 'views_of_other_roles_not_in_the_gate_command' not in full                      # the S1 store holds the five roles only
    est = e['detail']['raw_fit_estimate']
    # R14-c: the raw FIT cohort x the views per match of THIS (smoke) run: 42 views / 3 matches = 14
    assert est['is_estimate'] is True and est['matches'] == report_s2.RAW_FIT_MATCHES == 162_000
    assert est['views_per_match'] == 14.0 and est['views'] == pytest.approx(162_000 * 14.0)
    assert e['raw_fit_estimate'] == pytest.approx(162_000 * 14.0 * per_view / GIB) == pytest.approx(est['gib'])
    assert f'162,000 matches x 14.00 views/match x {per_view:,} bytes' in est['arithmetic']
    assert 'estimate' in est['arithmetic'] and 'design' in est['matches_source'] and '42 views / 3 matches' in est['views_per_match_source']
    # beside it, the same with the mean of all 4 FIT matches of the S1 store: 48 views / 4 matches = 12
    cross = est['cross_check_mean_of_all_fit_matches_of_s1']
    assert cross['views_per_match'] == 12.0 and cross['gib'] == pytest.approx(162_000 * 12.0 * per_view / GIB)
    assert '48 views / 4 FIT matches' in cross['arithmetic']
    assert e['bytes_per_view'] == per_view and e['detail']['drive_free_gib_at_report_time'] > 0


def test_the_raw_fit_match_count_can_be_set_and_must_be_a_positive_int(tmp_path, s1_dir, terrain_base):
    r = build_run(tmp_path, s1_dir, terrain_base)
    doc = rep(r, raw_fit_matches=1000)
    assert doc['extrapolated_gib']['detail']['raw_fit_estimate']['matches'] == 1000
    assert doc['extrapolated_gib']['raw_fit_estimate'] == pytest.approx(1000 * 14.0 * (tokens.bytes_per_view() + 27) / GIB)
    for bad in (0, -5, 1.5, True, '162000'):
        with pytest.raises(ValueError, match='raw_fit_matches'):
            report_s2.report(r.path, s1=r.s1, raw_fit_matches=bad)
    with pytest.raises(SystemExit) as exit_info:                                              # the command line says so too (argparse: 2)
        report_s2.main([str(r.path), '--no-matches', '--raw-fit-matches', '0'])
    assert exit_info.value.code == 2


# ---- positions, NaN, constants, ages -----------------------------------------------------------------------------------------------------
def test_suspect_positions(nominal):
    r, doc = nominal
    sp = doc['suspect_positions']
    census = lambda n, nan, zz, off, nb: dict(n=n, nan=nan, zero_zero=zz, outside_map=off, not_binned=nb)          # noqa: E731
    assert sp['terrain_fit'] == {'regular_frames': census(1000, 4, 1, 2, 6), 'game_end_frame': census(30, 0, 0, 0, 0),
                                 'kills': census(77, 3, 0, 1, 4)}                       # copied from TERRAIN.json, by source (S2-9)
    v = sp['smoke_views']
    assert (v['player_rows'], v['nan_rows'], v['zero_zero_rows'], v['off_map_rows']) == (420, 4, 2, 3)
    assert (v['views_with_zero_zero'], v['views_with_off_map'], v['views_with_any']) == (2, 3, 4)     # views 3, 11 / 7, 9, 11 / 3, 7, 9, 11
    assert v['views_with_any_by_src'] == {'frame_clock': 2, 'M3': 0, 'M4': 2}
    assert v['map_side'] == 14875.0                                                     # (0, 5) and (14875, 0) are on the map: not counted


def test_nan_share_per_column_and_cell(nominal):
    r, doc = nominal
    ns = doc['nan_share']
    assert (ns['views'], ns['player_rows']) == (42, 420) and ns['outside_allowed'] == [] and ns['ok'] is True
    allowed = tokens.NAN_ALLOWED
    assert set(ns['players_num']) == set(allowed['players_num']['cells'])             # every allowed column is listed, 0.0 if it has none
    assert set(ns['summary']) == set(allowed['summary']['cells'])
    for column in ('x', 'y', 'wall_fraction'):
        assert ns['players_num'][column] == pytest.approx(4 / 420)                      # the 4 players without a position
    assert ns['players_num']['prev_x'] == ns['players_num']['prev_y'] == pytest.approx(10 / 420)     # the first view has no previous frame
    for column in traits.TRAIT_NAMES[:-1]:
        assert ns['players_num'][column] == pytest.approx(12 / 420)                     # the 12 players of an unknown champion
    assert ns['summary'] == {'herald_next_spawn_s': pytest.approx(10 / 42), 'atakhan_next_spawn_s': 0.0, 'voidgrub_next_spawn_s': 1.0}
    assert ns['landmark_seconds'] == {'herald[0]': pytest.approx(10 / 42), 'grubs[0]': pytest.approx(0.5), 'atakhan[0]': 0.0,
                                      'atakhan[1]': 0.0}
    assert ns['ev_xy'] == pytest.approx(np.isnan(r.x['ev_xy']).mean())                  # the padding rows
    assert ns['geo_pp'] == pytest.approx(4 * 19 / (42 * 100)) and ns['geo_pl'] == pytest.approx(4 * 34 / (42 * 340))


def test_nan_and_infinity_outside_the_allowed_map_are_found_and_flagged(tmp_path, s1_dir, terrain_base):
    def edit(x, table):
        x['ev_dt'][4, 0] = np.nan                                  # no entry in NAN_ALLOWED: no NaN at all
        x['players_num'][6, 2, P['level']] = np.nan                # a frame column other than the position
        x['summary'][3, S['dragon_next_spawn_s']] = np.nan         # not one of the three NAN_FEATURES
        x['landmarks'][2, 5, 1] = np.nan                           # a landmark seconds cell that is not a nest
        x['geo_pp'][1, 0, 1] = np.inf                              # infinity is allowed nowhere
    doc = rep(build_run(tmp_path, s1_dir, terrain_base, edit_x=edit))
    ns = doc['nan_share']
    found = {(e['array'], e['cell'], e['kind']): e['count'] for e in ns['outside_allowed']}
    assert found == {('ev_dt', 'all', 'nan'): 1, ('players_num', 'level', 'nan'): 1, ('summary', 'dragon_next_spawn_s', 'nan'): 1,
                     ('landmarks', 'row 5, column 1', 'nan'): 1, ('geo_pp', 'all', 'inf'): 1}
    assert ns['ok'] is False and 'NAN_OUTSIDE_ALLOWED' in flag_codes(doc)


def test_constant_columns(nominal):
    r, doc = nominal
    c = doc['constant_columns']
    assert c['views'] == 42
    assert c['summary'] == {**CONST_SUMMARY, 'voidgrub_next_spawn_s': None}            # all NaN is one value; atakhan_in_patch, 8 flags
    assert len([n for n in c['summary'] if n.startswith('rule_uncertain_')]) == 8
    assert c['players_num'] == CONST_PLAYERS                                           # a column that is NaN in some rows is not constant
    assert c['patches_in_smoke'] == ['15.18', '15.19'] and 'smoke' in c['note']


def test_frame_age_by_view_kind(nominal):
    r, doc = nominal
    t = r.table
    ages = (t['ref'] - t['t0']) / 1000.0
    fa = doc['frame_age_s']
    assert fa['frame_clock'] == {'views': 24, 'min': 0.0, 'median': 0.0, 'p99': 0.0, 'max': 0.0}      # a frame-clock row is on its frame
    for name, code in (('M3', 1), ('M4', 2)):
        a = ages[t['src'] == code]
        assert fa[name]['views'] == 9 and fa[name]['min'] == a.min() and fa[name]['max'] == a.max()
        assert fa[name]['median'] == pytest.approx(np.median(a)) and fa[name]['p99'] == pytest.approx(np.percentile(a, 99))
    assert doc['checks'] == {'views_agree_with_run_json': True, 'frame_clock_rows_on_their_frame': True,
                             'frame_age_matches_summary': True, 'game_time_matches_ref': True}


# ---- the figures ---------------------------------------------------------------------------------------------------------------------
PNG = b'\x89PNG\r\n\x1a\n'


def test_four_figures_are_written(nominal):
    r, doc = nominal
    assert sorted(doc['figures']) == ['fig1_occupancy_walkable', 'fig2_geodesic_path', 'fig3_event_edge_counts', 'fig4_frame_age']
    for name, fig in doc['figures'].items():
        path = r.path / fig['file']
        assert path == r.path / 'report' / f'{name}.png'
        data = path.read_bytes()
        assert data.startswith(PNG) and len(data) > 8_000                                # a real picture, not an empty canvas
    assert sorted(p.name for p in (r.path / 'report').iterdir()) == sorted(f'{n}.png' for n in doc['figures'])


def test_the_numbers_behind_the_figures(nominal):
    r, doc = nominal
    f = {name: fig['numbers'] for name, fig in doc['figures'].items()}
    assert f['fig1_occupancy_walkable'] == {'walkable_cells': 5625, 'largest_component_cells': 5625, 'cells_outside_largest': 0,
                                            'zones': list(terrain.ZONES)}
    assert f['fig3_event_edge_counts']['caps'] == {'events': C.MAX_EVENTS, 'edges': C.MAX_EDGES}
    assert f['fig3_event_edge_counts']['views_with_dropped'] == {'events': 3, 'edges': 2}
    assert f['fig3_event_edge_counts']['max_window'] == {'events': C.MAX_EVENTS + 12, 'edges': C.MAX_EDGES + 20}     # stored + dropped
    path = f['fig2_geodesic_path']
    assert path['length_u'] == pytest.approx(path['table_distance_u'], abs=1.0) and path['length_u'] >= path['straight_u']
    assert doc['terrain']['geodesic_path'] == path and 'cells' not in path and path['n_cells'] > 50
    assert f['fig4_frame_age']['views'] == {'frame_clock': 24, 'M3': 9, 'M4': 9}


@pytest.fixture
def figures(monkeypatch):
    """The matplotlib figures of the next reports by file stem (``fig1_occupancy_walkable`` ...): every figure goes through ``_save``."""
    seen, real_save = {}, report_s2._save

    def spy(fig, path):
        seen[Path(path).stem] = fig
        real_save(fig, path)
    monkeypatch.setattr(report_s2, '_save', spy)
    return seen


def test_figure_3_plots_the_window_totals_so_that_the_tail_beyond_the_cap_shows(tmp_path, s1_dir, terrain_base, figures):
    """Review minor (Task 14): the stored counts stop at the cap, so a histogram of them could never show how far the windows run past it.
    The histograms hold stored + dropped, and their range reaches the largest window total (92 events and 148 edges here)."""
    rep(build_run(tmp_path, s1_dir, terrain_base))
    events_ax, edges_ax = figures['fig3_event_edge_counts'].axes
    for ax, cap, largest in ((events_ax, C.MAX_EVENTS, C.MAX_EVENTS + 12), (edges_ax, C.MAX_EDGES, C.MAX_EDGES + 20)):
        reach = max(float(patch.get_xy()[:, 0].max()) for patch in ax.patches)
        assert reach >= largest + 0.5 and ax.get_xlim()[1] >= largest             # the bars, and the axis, run past the cap to the longest window
        assert ax.get_xlim()[1] > cap + 1.5                                       # (the range of the stored counts ended at cap + 1.5)
        lines = [line.get_xdata()[0] for line in ax.lines]
        assert cap in lines                                                       # the cap stays marked
    assert 'window total' in events_ax.get_xlabel() and 'window total' in edges_ax.get_xlabel()


def test_figure_1_legend_gives_the_walkable_threshold_of_the_terrain_that_was_built(tmp_path, s1_dir, terrain_base, figures):
    """Final review M1: the legend said 'at least 5 positions' whatever the terrain. It now takes the number from TERRAIN.json."""
    def legend():
        return [t.get_text() for t in figures['fig1_occupancy_walkable'].axes[0].get_legend().get_texts()]
    doc = rep(build_run(tmp_path / 'final', s1_dir, terrain_base))
    assert f'walkable area (at least {terrain.MIN_COUNT} positions per 200 u cell)' in legend()
    assert doc['terrain']['grid'] == GRID_RECORD                                                      # and the report shows the grid record
    figures.clear()
    doc = rep(build_run(tmp_path / 'first', s1_dir, terrain_base, edit_terrain=lambda info: info['grid'].update(min_count=5)))
    assert 'walkable area (at least 5 positions per 200 u cell)' in legend()
    assert doc['terrain']['grid'] == {**GRID_RECORD, 'min_count': 5}                                  # the terrain of the first smoke run, as it is


@pytest.fixture(scope='module')
def corridor(tmp_path_factory):
    """A terrain folder whose walkable cells are two corridors (bottom: iy 8-11, ix 5-60; right: ix 57-60, iy 8-66): the two base anchors
    are joined only through the bend."""
    walk = np.zeros((75, 75), dtype=bool)
    walk[8:12, 5:61] = True
    walk[8:67, 57:61] = True
    out = tmp_path_factory.mktemp('report_s2_corridor') / 'terrain'
    zero = np.zeros((150, 150))
    terrain.write_terrain_files(out, walk, zero, zero, terrain.geodesic_table(walk))
    yield out
    gc.collect()
    shutil.rmtree(out.parent, ignore_errors=True)


def test_the_base_path_follows_a_corridor_and_has_the_length_of_the_table(corridor):
    terr = terrain.Terrain(corridor)
    p = report_s2.base_path(terr)
    cells, walk = p['cells'], terr.walk
    assert len(cells) > 60 and all(walk[iy, ix] for ix, iy in cells)                   # every cell of the path is walkable
    steps = np.diff(np.array(cells), axis=0)
    assert (np.abs(steps).max(axis=1) == 1).all()                                       # neighbours, one step at a time
    for (ix, iy), (dx, dy) in zip(cells[:-1], steps.tolist()):
        if dx and dy:
            assert walk[iy, ix + dx] and walk[iy + dy, ix]                              # a diagonal step never cuts a corner
    assert abs(p['length_u'] - p['table_distance_u']) <= 1.0                            # the path is a shortest path of the table
    assert p['length_u'] > 1.25 * p['straight_u']                                       # the corridor forces a detour
    assert any(ix >= 57 and iy >= 60 for ix, iy in cells) and any(ix <= 12 and iy <= 12 for ix, iy in cells)
    start, end = int(terr.snap(*p['start_xy'])), int(terr.snap(*p['end_xy']))
    assert cells[0] == (start % 75, start // 75) and cells[-1] == (end % 75, end // 75)
    del terr


@pytest.fixture(scope='module')
def staircase(tmp_path_factory):
    """A terrain folder whose only walkable cells are a one-cell-wide staircase from the blue base cell (9, 10) to the red base cell
    (64, 64): +x, +y, +x, ... Its cells touch diagonally at every step, but a diagonal needs both orthogonal cells, so the only route is
    along the steps (109 of 200 u); a path that cut the corners would be 15,473 u."""
    cells = [(9, 10)]
    while cells[-1] != (64, 64):
        ix, iy = cells[-1]
        cells.append((ix + 1, iy) if (ix - 9) == (iy - 10) else (ix, iy + 1))
    walk = np.zeros((75, 75), dtype=bool)
    for ix, iy in cells:
        walk[iy, ix] = True
    out = tmp_path_factory.mktemp('report_s2_staircase') / 'terrain'
    zero = np.zeros((150, 150))
    terrain.write_terrain_files(out, walk, zero, zero, terrain.geodesic_table(walk))
    yield out, cells
    gc.collect()
    shutil.rmtree(out.parent, ignore_errors=True)


def test_the_base_path_never_cuts_a_corner_that_the_graph_of_the_table_does_not_have(staircase):
    folder, cells = staircase
    terr = terrain.Terrain(folder)
    p = report_s2.base_path(terr)
    assert p['cells'] == cells and p['n_cells'] == len(cells) == 110                  # exactly the steps of the staircase
    assert p['length_u'] == p['table_distance_u'] == 109 * 200                         # 21,800 u, not the 15,473 u of a diagonal line
    assert p['ratio'] == pytest.approx(21_800 / p['straight_u'])
    del terr


def test_the_base_path_on_the_open_grid_is_close_to_a_straight_line(open_terrain):
    p = report_s2.base_path(open_terrain)
    assert abs(p['length_u'] - p['table_distance_u']) <= 1.0
    assert p['straight_u'] <= p['length_u'] <= 1.05 * p['straight_u']
    (x0, y0), (x1, y1) = p['start_xy'], p['end_xy']
    assert x0 < 3000 and y0 < 3000 and x1 > 11000 and y1 > 11000                       # the nexus turrets of the blue and the red base


def test_the_base_anchors_are_the_midpoints_of_the_two_nexus_turrets_of_each_team():
    blue, red = report_s2.base_anchors()
    turrets = {team: [lm for lm in state.LANDMARKS if lm['tower'] == 'NEXUS_TURRET' and lm['team'] == team] for team in (100, 200)}
    assert [len(turrets[100]), len(turrets[200])] == [2, 2]
    for anchor, team in ((blue, 100), (red, 200)):
        assert anchor == (sum(lm['x'] for lm in turrets[team]) / 2, sum(lm['y'] for lm in turrets[team]) / 2)
    assert blue[0] < 3000 and blue[1] < 3000 and red[0] > 11000 and red[1] > 11000    # one in each corner of the map


# ---- measurements on the raw matches (only with records) ---------------------------------------------------------------------------------
def lag_records(deviations=()):
    """The three smoke matches with LEVEL_UP events that give every player the clock level 1 + k // 3 at the frame times of the stored views
    (the level the views store), plus ``deviations`` [(match, event), ...]."""
    recs = {}
    for m in SMOKE:
        events = [vh.raw_level(ref_of(m, 3 * (level - 1)) - 20_000, pid, level) for pid in range(1, 11) for level in (2, 3, 4)]
        events += [e for mm, e in deviations if mm == m]
        recs[KEYS[m]] = synth.make_record(T_ms=1_200_000, events=events, patch=S1_PATCHES[m], key=KEYS[m])
    return recs


def lag_deviations():
    """(a) match 0, frame 5, player 4: a LEVEL_UP to 3 on the ms of the frame, which still shows 2 (the frame lags by one on the LEVEL_UP);
    (b) match 1, frame 7, player 2: a LEVEL_UP to 5 on the ms of the frame, which shows 3 (two levels, on the LEVEL_UP);
    (c) match 2, frame 11, player 7: a LEVEL_UP to 5 one second before the frame, which shows 4 (no LEVEL_UP on the ms)."""
    return [(0, vh.raw_level(ref_of(0, 5), 4, 3)), (1, vh.raw_level(ref_of(1, 7), 2, 5)), (2, vh.raw_level(ref_of(2, 11) - 1000, 7, 5))]


def hook(recs):
    """The ``records`` argument: key and patch in, the synthetic record out (it asserts that the patch is the S1 patch of the match)."""
    def read(key, patch):
        assert patch == S1_PATCHES[KEYS.index(key)]
        return recs[key]
    return read


def test_nothing_is_read_from_raw_matches_unless_records_are_given(nominal):
    r, doc = nominal
    for name in ('level_lag', 'building_kill_matching'):
        assert doc[name]['measured'] is False and 'records' in doc[name]['reason']
    assert doc['raw_measurement'] == {'measured': False, 'matches_requested': 3, 'matches_measured': 0, 'errors': {}}


def test_level_lag_counts_the_frame_clock_cells_that_differ_from_the_level_clock(tmp_path, s1_dir, terrain_base):
    doc = rep(build_run(tmp_path, s1_dir, terrain_base), records=hook(lag_records(lag_deviations())))
    lag = doc['level_lag']
    assert lag['measured'] is True and lag['matches'] == 3 and lag['frame_clock_cells'] == 240            # 24 frame-clock views x 10
    assert lag['differ'] == 3 and lag['share_differ'] == pytest.approx(3 / 240)
    assert lag['differ_on_same_ms_level_up'] == 2 and lag['differ_lag_by_one_at_level_up_ms'] == 1 and lag['differ_unexplained'] == 2
    assert doc['raw_measurement'] == {'measured': True, 'matches_requested': 3, 'matches_measured': 3, 'errors': {}}
    assert 'LEVEL_LAG_UNEXPLAINED' in flag_codes(doc)                                  # (b) and (c) are not the S2-8 case


def test_no_level_lag_when_every_frame_agrees_with_the_clock(tmp_path, s1_dir, terrain_base):
    doc = rep(build_run(tmp_path, s1_dir, terrain_base), records=hook(lag_records()))
    assert doc['level_lag']['differ'] == 0 and doc['level_lag']['differ_unexplained'] == 0
    assert 'LEVEL_LAG_UNEXPLAINED' not in flag_codes(doc)


def test_the_level_lag_does_not_depend_on_the_order_of_the_matches_in_the_views_table(tmp_path, s1_dir, terrain_base):
    r = build_run(tmp_path, s1_dir, terrain_base, order=(2, 0, 1))
    assert r.table['match'].tolist()[:1] == [2]
    lag = rep(r, records=hook(lag_records(lag_deviations())))['level_lag']
    assert (lag['frame_clock_cells'], lag['differ'], lag['differ_on_same_ms_level_up'], lag['differ_unexplained']) == (240, 3, 2, 2)


def building_records():
    """Match 0: four BUILDING_KILLs of the blue team's buildings: the outer top turret (ok, at its landmark), the same turret again
    (already down), a kill in 'JUNGLE' (no landmark fits, no position) and an inner mid turret without a position."""
    lm = state.LANDMARKS[state.landmark_index(100, 'TOP_LANE', 'TOWER_BUILDING', 'OUTER_TURRET')]
    events = [vh.raw_building(200_000, 100, 'TOP_LANE', x=lm['x'], y=lm['y']),
              vh.raw_building(260_000, 100, 'TOP_LANE', x=lm['x'], y=lm['y']),
              vh.raw_building(300_000, 100, 'JUNGLE'), vh.raw_building(320_000, 100, 'MID_LANE', 'TOWER_BUILDING', 'INNER_TURRET')]
    return {KEYS[m]: synth.make_record(T_ms=1_200_000, events=events if m == 0 else [], patch=S1_PATCHES[m], key=KEYS[m]) for m in SMOKE}


def test_building_kill_matching_is_the_sweep_state_counters_of_the_smoke_matches(tmp_path, s1_dir, terrain_base):
    recs = building_records()
    doc = rep(build_run(tmp_path, s1_dir, terrain_base), records=hook(recs))
    got = doc['building_kill_matching']
    expected = Counter({k: 0 for k in sweep_state.COUNT_KEYS})
    for key in KEYS[:3]:
        sweep_state.count_record(recs[key], expected)                                   # the same function, on the same records
    assert got['measured'] is True and got['counts'] == {k: expected[k] for k in sweep_state.COUNT_KEYS}
    c = got['counts']
    assert (c['matches'], c['building_kill_events'], c['building_kills_unmatched'], c['building_kills_on_destroyed_landmark'],
            c['building_kills_on_destroyed_lane_turret'], c['building_kills_without_position'],
            c['building_kills_far_from_their_landmark']) == (3, 4, 1, 1, 1, 2, 0)
    assert set(got['objective_nan_cells_by_feature']) == set(state.NAN_FEATURES)


@pytest.mark.parametrize('make, label', [
    (lambda key, patch: reader.MatchRejected('no_game_end'), 'MatchRejected'),
    (lambda key, patch: reader.MatchFormatError(key, patch, 'a clock that breaks what every real match shows'), 'MatchFormatError'),
    (lambda key, patch: reader.MatchReadError('timeline', 'PermissionError', 13, patch), 'MatchReadError'),
    (lambda key, patch: ValueError(f'cannot use {key}'), 'ValueError'),                       # a record-format error whose text holds the key
], ids=['rejected', 'format-error', 'read-error', 'value-error'])
def test_a_match_that_cannot_be_measured_is_counted_by_class_and_no_key_is_written(tmp_path, s1_dir, terrain_base, make, label):
    """The reader's errors and the record-format errors are counted by class and place (never by message) and the measurement goes on."""
    recs = lag_records()

    def read(key, patch):
        if key == KEYS[1]:
            raise make(key, patch)
        return recs[key]
    r = build_run(tmp_path, s1_dir, terrain_base)
    doc = rep(r, records=read)
    raw = doc['raw_measurement']
    assert raw['matches_requested'] == 3 and raw['matches_measured'] == 2 and list(raw['errors'].values()) == [1]
    assert next(iter(raw['errors'])).startswith(label) and 'RAW_MEASUREMENT_ERRORS' in flag_codes(doc)
    assert doc['level_lag']['matches'] == 2 and doc['level_lag']['frame_clock_cells'] == 180        # (8 + 10) frame-clock views x 10
    assert not any(k in (r.path / 'S2_REPORT.json').read_text(encoding='utf-8') for k in KEYS)


@pytest.mark.parametrize('error', [OSError('the disk is gone'), KeyError('frame'), RuntimeError('a bug')],
                         ids=['OSError', 'KeyError', 'RuntimeError'])
def test_an_unexpected_error_in_the_measurement_is_not_swallowed_and_nothing_is_written(tmp_path, s1_dir, terrain_base, error):
    """Review minor (Task 14): only the errors of the reader and of the record format are counted. Anything else is a bug or a broken
    machine, and a report that counted it as one unmeasured match would hide it."""
    recs = lag_records()

    def read(key, patch):
        if key == KEYS[1]:
            raise error
        return recs[key]
    r = build_run(tmp_path, s1_dir, terrain_base)
    with pytest.raises(type(error)):
        report_s2.report(r.path, s1=r.s1, records=read)
    assert not (r.path / 'S2_REPORT.json').exists() and not (r.path / 'report').exists()


def test_the_library_path_checks_e6_before_it_reads_a_raw_match(small_s2_run, fake_processes):
    """Final review M9: ``report(records=...)`` reads raw matches, and only the command line asked for the E6 check. The check is made
    inside the measurement, and the measurement runs before the first figure is drawn, so a refused report has written nothing."""
    r, asked = small_s2_run, []

    def records(key, patch):
        asked.append(key)
        raise AssertionError('a raw match was read while a model server runs')
    fake_processes([synth.STRATA_SERVER])
    with pytest.raises(run_s1.StrataRunning, match=r'E6.*pid 4101'):
        report_s2.report(r.path, s1=r.s1, records=records)
    assert not asked and not (r.path / 'S2_REPORT.json').exists() and not (r.path / 'report').exists()
    fake_processes([synth.STRATA_BRIDGE])                                                         # an idle MCP bridge is no server
    doc = json.loads(report_s2.report(r.path, s1=r.s1, records=hook(lag_records())).read_text(encoding='utf-8'))
    assert doc['raw_measurement']['measured'] is True
    fake_processes([synth.STRATA_SERVER])
    assert report_s2.report(r.path, s1=r.s1).is_file()                                            # files only: no raw match, no E6 check


# ---- privacy -------------------------------------------------------------------------------------------------------------------------
def test_no_match_key_is_written_anywhere(nominal):
    r, doc = nominal
    blobs = [(r.path / 'S2_REPORT.json').read_bytes()] + [p.read_bytes() for p in (r.path / 'report').iterdir()]
    assert len(blobs) == 5
    assert not any(key.encode('ascii') in blob for key in KEYS for blob in blobs)


def test_the_key_guard_names_the_number_of_hits_and_never_a_key(s1_dir):
    s1 = Store(s1_dir / 's1')
    report_s2.assert_no_keys(s1, 'counts 1 2 3 {"a": 4}')
    with pytest.raises(RuntimeError) as err:
        report_s2.assert_no_keys(s1, f'text {KEYS[2]} more {KEYS[5]}')
    assert '2 match keys' in str(err.value) and not any(k in str(err.value) for k in KEYS)


# ---- refusals and guards -------------------------------------------------------------------------------------------------------------
def pop(*path):
    def edit(doc):
        d = doc
        for step in path[:-1]:
            d = d[step]
        del d[path[-1]]
    return edit


def put(*path, value):
    def edit(doc):
        d = doc
        for step in path[:-1]:
            d = d[step]
        d[path[-1]] = value
    return edit


@pytest.mark.parametrize('edit, message', [
    (put('s1', 'files_sha256', value='0' * 64), 'S1 store'),                                   # another S1 store than the build read
    (put('terrain', 'sha256', 'geodesic.npy', value='0' * 64), 'terrain'),                      # another terrain than the build recorded
    (put('s2_result', 'views', value=41), 'views'),                                            # RUN.json and the store disagree
    (put('s2_result', 'views_by_src', value={'frame_clock': 24, 'M3': 10, 'M4': 8}), 'views_by_src'),
    (put('s2_result', 'bytes_per_view', value=7000), 'bytes_per_view'),
    (put('s2_result', 'matches', value=2), 'matches'),                                         # the table holds views of 3 matches
    (put('s2_result', 'complete', value=False), 'complete'),
    (pop('s2_result'), 's2_result'),                                                           # the build did not finish
    (pop('s2_result', 'peak_private_gib'), 'peak_private_gib'),                                # a field the gate needs is missing: named
    (pop('s2_result', 'seconds_per_match'), 'seconds_per_match'),
    (pop('s1'), 's1'),
    (pop('terrain'), 'terrain'),
])
def test_inputs_that_do_not_belong_together_are_refused_and_the_message_names_the_field(tmp_path, s1_dir, terrain_base, edit, message):
    r = build_run(tmp_path, s1_dir, terrain_base, edit_run=edit)
    with pytest.raises(report_s2.ReportInputError, match=message):
        report_s2.report(r.path, s1=r.s1)
    assert not (r.path / 'S2_REPORT.json').exists()                                            # nothing was written


def test_a_listed_match_without_a_view_counts_in_the_views_per_match(tmp_path, s1_dir, terrain_base):
    """A very short match that is listed has no frame-clock row and no event row, so the views table holds fewer matches than RUN.json
    lists: the report takes the listed number for the views per match (the raw FIT cohort holds such matches too)."""
    doc = rep(build_run(tmp_path, s1_dir, terrain_base, edit_run=put('s2_result', 'matches', value=4)))
    assert doc['matches'] == 4 and doc['matches_with_views'] == 3 and doc['views_per_match'] == pytest.approx(10.5)
    est = doc['extrapolated_gib']['detail']['raw_fit_estimate']
    assert est['views_per_match'] == pytest.approx(10.5) and '42 views / 4 matches listed' in est['views_per_match_source']
    assert flag_codes(doc) == OVERFLOW_FLAGS                                                   # nothing else comes of the missing match


def test_a_store_that_is_not_complete_is_refused(small_s2_run):
    store = Store(small_s2_run.path / 's2')
    store.manifest['complete'] = False
    store.save()
    with pytest.raises(ValueError, match='complete'):
        report_s2.report(small_s2_run.path, s1=small_s2_run.s1)


def test_a_store_with_another_layout_than_view_spec_is_refused(small_s2_run):
    store = Store(small_s2_run.path / 's2')
    store.manifest['arrays']['x/summary']['dtype'] = '<f8'
    store.save()
    with pytest.raises(report_s2.ReportInputError, match=r'x/summary'):
        report_s2.report(small_s2_run.path, s1=small_s2_run.s1)


def test_a_changed_byte_is_found_by_the_verification_and_no_verify_skips_it(small_s2_run):
    path = small_s2_run.path / 's2' / 'x' / 'ev_n.npy'
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='STORE.json'):
        report_s2.report(small_s2_run.path, s1=small_s2_run.s1)
    doc = json.loads(report_s2.report(small_s2_run.path, s1=small_s2_run.s1, verify=False).read_text(encoding='utf-8'))
    assert doc['store']['verified'] is False


def test_the_report_does_not_touch_the_store(nominal):
    r, doc = nominal
    Store(r.path / 's2').verify()                                                              # still complete and every hash right
    assert len(Store(r.path / 's2').stored_files()) == 26


def test_outputs_may_go_neither_into_the_store_nor_into_the_s1_run(small_s2_run):
    r = small_s2_run
    for kw in ({'json_path': r.path / 's2' / 'S2_REPORT.json'}, {'report_dir': r.path / 's2' / 'report'}):
        with pytest.raises(ValueError, match='store'):
            report_s2.report(r.path, s1=r.s1, **kw)
    for kw in ({'json_path': r.s1 / 'S2_REPORT.json'}, {'report_dir': r.s1 / 'report'}):
        with pytest.raises(ValueError, match='S1'):
            report_s2.report(r.path, s1=r.s1, **kw)
    assert not (r.s1 / 'S2_REPORT.json').exists() and not (r.s1 / 'report').exists()


def test_the_s1_run_is_found_next_to_the_s2_run_by_the_name_in_run_json(tmp_path, s1_dir, terrain_base):
    r = build_run(s1_dir.parent, s1_dir, terrain_base, name='20261021_c_ept_s2_7')            # a sibling of the S1 run folder
    try:
        assert report_s2.report(r.path).is_file()                                              # no s1=
    finally:
        shutil.rmtree(r.path, ignore_errors=True)
    lonely = build_run(tmp_path, s1_dir, terrain_base)
    with pytest.raises(FileNotFoundError, match=S1_NAME):
        report_s2.report(lonely.path)
    assert report_s2.report(lonely.path, s1=s1_dir / 's1').is_file()                           # the store folder does as the run folder


def test_the_terrain_folder_can_lie_outside_the_run_and_must_be_there(tmp_path, s1_dir, terrain_base):
    r = build_run(tmp_path, s1_dir, terrain_base)
    shutil.move(str(r.path / 'terrain'), str(tmp_path / 'shared_terrain'))                     # a full build uses the smoke run's terrain
    with pytest.raises(FileNotFoundError, match='--terrain'):
        report_s2.report(r.path, s1=r.s1)
    doc = json.loads(report_s2.report(r.path, s1=r.s1, terrain_dir=tmp_path / 'shared_terrain').read_text(encoding='utf-8'))
    assert doc['terrain']['folder'] == 'shared_terrain' and doc['terrain']['hashes_match_run_json'] is True


def test_a_terrain_without_provenance_is_refused(tmp_path, s1_dir, terrain_base):
    r = build_run(tmp_path, s1_dir, terrain_base)
    (r.path / 'terrain' / 'TERRAIN.json').unlink()
    with pytest.raises(report_s2.ReportInputError, match='TERRAIN.json'):
        report_s2.report(r.path, s1=r.s1)


@pytest.mark.parametrize('edit, field', [
    (lambda info: info['positions'].pop('kills'), 'positions.kills'),
    (lambda info: info['positions'].pop('regular_frames'), 'positions.regular_frames'),
    (lambda info: info['positions'].pop('game_end_frame'), 'positions.game_end_frame'),
    (lambda info: info.pop('positions'), 'positions'),
    (lambda info: info['grid'].pop('min_count'), 'grid.min_count'),
    (lambda info: info.pop('grid'), 'grid'),
    (lambda info: info.pop('sha256'), 'sha256'),
], ids=['no-kills', 'no-regular-frames', 'no-game-end-frame', 'no-census', 'no-min-count', 'no-grid', 'no-hashes'])
def test_a_terrain_json_that_lacks_a_field_the_report_reads_is_refused_before_any_figure_is_written(tmp_path, s1_dir, terrain_base, edit, field):
    """Review minor (Task 14): the position census (``suspect_positions``) and the walkable threshold (the legend of figure 1) were read
    after the figures were drawn, so a TERRAIN.json without them stopped the report with four figures already on disk."""
    r = build_run(tmp_path, s1_dir, terrain_base, edit_terrain=edit)
    with pytest.raises(report_s2.ReportInputError, match=field):
        report_s2.report(r.path, s1=r.s1)
    assert not (r.path / 'report').exists() and not (r.path / 'S2_REPORT.json').exists()


def test_a_terrain_whose_files_differ_from_their_terrain_json_is_a_report_input_error(tmp_path, s1_dir, terrain_base, capsys):
    """``Terrain.verify`` raised a plain ValueError, which the command line did not catch."""
    r = build_run(tmp_path, s1_dir, terrain_base, edit_terrain=lambda info: info['sha256'].update({'terrain.npz': '0' * 64}))
    with pytest.raises(report_s2.ReportInputError, match=r'terrain\.npz.*differ from the SHA-256 in TERRAIN\.json'):
        report_s2.report(r.path, s1=r.s1)
    assert not (r.path / 'report').exists() and not (r.path / 'S2_REPORT.json').exists()
    assert report_s2.main([str(r.path), '--s1', str(r.s1), '--no-matches']) == 2                     # the command line says why and writes nothing
    err = capsys.readouterr().err
    assert 'report refused' in err and 'terrain.npz' in err and not (r.path / 'report').exists()


def test_a_terrain_whose_two_files_do_not_belong_together_is_a_report_input_error(tmp_path, s1_dir, terrain_base, corridor, capsys):
    """The walkable grid of one terrain with the geodesic table of another: ``Terrain`` itself refuses it (a ValueError), the report says
    why and writes nothing."""
    bad, zero = tmp_path / 'mismatch', np.zeros((150, 150))
    hashes = terrain.write_terrain_files(bad, np.ones((75, 75), dtype=bool), zero, zero, np.load(corridor / 'geodesic.npy'))
    info = json.loads((terrain_base / 'TERRAIN.json').read_text(encoding='utf-8'))
    info['sha256'] = hashes
    (bad / 'TERRAIN.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    r = build_run(tmp_path / 'run', s1_dir, bad)
    with pytest.raises(report_s2.ReportInputError, match='does not belong'):
        report_s2.report(r.path, s1=r.s1)
    assert not (r.path / 'report').exists() and not (r.path / 'S2_REPORT.json').exists()
    assert report_s2.main([str(r.path), '--s1', str(r.s1), '--no-matches']) == 2 and 'report refused' in capsys.readouterr().err


def test_a_views_table_that_does_not_fit_the_s1_store_or_the_sources_is_refused(tmp_path, s1_dir, terrain_base):
    def far_match(x, table):
        table['match'][0] = 99                                                                 # an S1 row that the store does not hold

    def odd_source(x, table):
        table['src'][3] = 3                                                                    # neither frame clock, M3 nor M4
    r = build_run(tmp_path / 'a', s1_dir, terrain_base, edit_x=far_match)
    with pytest.raises(report_s2.ReportInputError, match='views/match'):
        report_s2.report(r.path, s1=r.s1)
    r = build_run(tmp_path / 'b', s1_dir, terrain_base, edit_x=odd_source)
    with pytest.raises(report_s2.ReportInputError, match='views/src'):
        report_s2.report(r.path, s1=r.s1)


def test_the_chunk_size_changes_no_number(nominal, tmp_path, s1_dir, terrain_base, monkeypatch):
    monkeypatch.setattr(report_s2, 'CHUNK_VIEWS', 5)
    doc = rep(build_run(tmp_path, s1_dir, terrain_base))
    a, b = json.loads(json.dumps(nominal[1])), json.loads(json.dumps(doc))
    for d in (a, b):
        d['extrapolated_gib']['detail'].pop('drive_free_gib_at_report_time')               # the free space moves between two calls
        d.pop('report_code')                                                               # and so can the commit of a repository in use
    assert a == b


def test_report_s2_does_not_import_the_runner_chain_or_the_builder_at_module_level():
    import ast
    tree = ast.parse(Path(report_s2.__file__).read_text(encoding='utf-8'))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {node.module} | {f'{node.module}.{a.name}' for a in node.names}
    assert not names & {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.store_build', 'c_ept.report_s1'}


# ---- the command line ---------------------------------------------------------------------------------------------------------------
def test_cli_writes_the_report_from_files_only_even_while_a_model_server_runs(small_s2_run, capsys, fake_processes):
    r = small_s2_run
    fake_processes([synth.STRATA_SERVER])                                                      # E6 would refuse a run with raw matches
    assert report_s2.main([str(r.path), '--s1', str(r.s1), '--no-matches']) == 0
    out = capsys.readouterr().out
    assert (r.path / 'S2_REPORT.json').is_file() and 'views 42' in out and 'FLAG EVENT_OVERFLOW_OVER_5_PERCENT' in out
    assert not any(k in out for k in KEYS)
    for name in ('fig1_occupancy_walkable', 'fig2_geodesic_path', 'fig3_event_edge_counts', 'fig4_frame_age'):
        assert f'figure: {r.path / "report" / (name + ".png")}' in out                           # the four figure paths, in full
    assert 'peak private 6.5 GiB (main 1.25, largest worker 0.9)' in out
    assert f'{tokens.bytes_per_view() + 27:,} bytes per view' in out            # x arrays + 27-byte table (8,321 at the caps 128, 128)
    assert '0 cell group(s) outside tokens.NAN_ALLOWED; largest shares summary/voidgrub_next_spawn_s 100.0%' in out
    assert 'constant columns: summary 10, players_num 3' in out and 'extrapolated: full cache build' in out


def test_cli_says_why_it_refuses_inputs_that_do_not_fit_and_writes_nothing(tmp_path, s1_dir, terrain_base, capsys):
    r = build_run(tmp_path, s1_dir, terrain_base, edit_run=put('s1', 'files_sha256', value='0' * 64))
    assert report_s2.main([str(r.path), '--s1', str(r.s1), '--no-matches']) == 2
    captured = capsys.readouterr()
    assert 'report refused' in captured.err and 'S1 store' in captured.err and captured.out == ''
    assert not (r.path / 'S2_REPORT.json').exists() and not (r.path / 'report').exists()
    assert report_s2.main([str(r.path / 'nowhere'), '--no-matches']) == 2 and 'report refused' in capsys.readouterr().err


def test_cli_refuses_to_read_raw_matches_while_a_model_server_runs(small_s2_run, capsys, fake_processes):
    r = small_s2_run
    fake_processes([synth.STRATA_SERVER])
    assert report_s2.main([str(r.path), '--s1', str(r.s1)]) == 2
    assert 'E6' in capsys.readouterr().err and not (r.path / 'S2_REPORT.json').exists()       # nothing was read, nothing written


def test_cli_measures_the_smoke_matches_with_the_reader(small_s2_run, monkeypatch, capsys):
    r = small_s2_run
    monkeypatch.setattr(reader, 'read_match', hook(lag_records(lag_deviations())))              # the reader's default root is never opened
    assert report_s2.main([str(r.path), '--s1', str(r.s1)]) == 0
    doc = json.loads((r.path / 'S2_REPORT.json').read_text(encoding='utf-8'))
    assert doc['level_lag']['measured'] is True and doc['level_lag']['differ'] == 3
    assert 'FLAG LEVEL_LAG_UNEXPLAINED' in capsys.readouterr().out


def test_cli_stops_with_exit_code_2_when_a_model_server_starts_between_its_check_and_the_measurement(small_s2_run, monkeypatch, capsys):
    """The command line checks E6 first; the measurement checks it again (final review M9). If the server appears in between, the report
    raises ``StrataRunning`` and the command line prints it like the first refusal (no traceback), with nothing written."""
    r = small_s2_run

    def late(*args, **kwargs):
        raise run_s1.StrataRunning('E6: a Strata process is running. Found: python.exe (pid 4101).')
    monkeypatch.setattr(report_s2, 'report', late)
    assert report_s2.main([str(r.path), '--s1', str(r.s1)]) == 2
    err = capsys.readouterr().err
    assert 'E6' in err and 'pid 4101' in err and '--no-matches' in err and not (r.path / 'S2_REPORT.json').exists()


# ---- a real build_s2 output goes into report() (final fix pass A) -------------------------------------------------------------------------
# The tests above give the report a run that they wrote by hand, so they cannot tell whether store_build and report_s2 still agree on the
# layout. These tests build a small S2 store with the real ``store_build.build_s2`` on a synthetic S1 store (``run_s1.build_store`` on
# synthetic records: no raw file) and report it. ``build_s2`` writes RUN.json through ``runlog.write_manifest``, which reads the decision
# record and the LEGACY worktree. Those reads are read only and the rest of this suite makes them as well (test_runlog, test_store_build),
# so they are accepted here and not stubbed.
REAL_MATCHES = [(C.PATCHES[i % 4], f'SYN_{i:04d}', synth.S1_MACROS[i % 5]) for i in range(10)]


@pytest.fixture(scope='module')
def real_s1(tmp_path_factory):
    """The run folder of a synthetic S1 run of ten matches made by ``run_s1.build_store`` (FIT, STOP, DEV, CAL, AUDIT, FIT, ...)."""
    run = tmp_path_factory.mktemp('report_s2_real_s1') / '20261021_c_ept_s1_2'
    with empty_process_list():            # a module-scoped fixture is set up before the autouse one: E6 would see the real process list
        run_s1.build_store(REAL_MATCHES, run / run_s1.STORE_DIR, run / run_s1.BLOCK_DIR, workers=1, block_matches=4,
                           records=synth.records_from_seed, role_of=synth.roles_of(REAL_MATCHES))
    yield run
    gc.collect()
    shutil.rmtree(run.parent, ignore_errors=True)


@pytest.fixture(scope='module')
def real_run(real_s1, open_table, tmp_path_factory):
    """(run folder, the parsed S2_REPORT.json): ``build_s2`` on the synthetic S1 store, then ``report`` with the synthetic records."""
    from c_ept import store_build            # only these tests need the builder
    base = tmp_path_factory.mktemp('report_s2_real_run')
    terr = write_terrain_folder(base / 'terrain', open_table, real_s1, 2)
    run = base / '20261021_c_ept_s2_1'
    with empty_process_list():
        store_build.build_s2(real_s1, terr, list(range(len(REAL_MATCHES))), run, workers=1, block_matches=4,
                             records=synth.records_from_seed, info={'macros': list(synth.S1_MACROS), 'limit': None})
        path = report_s2.report(run, s1=real_s1, terrain_dir=terr, records=synth.records_from_seed)
    yield run, json.loads(path.read_text(encoding='utf-8'))
    gc.collect()
    shutil.rmtree(base, ignore_errors=True)


def test_the_report_of_a_real_build_s2_run_agrees_with_the_run_json_that_the_builder_wrote(real_run, real_s1):
    run, doc = real_run
    man = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    r = man['s2_result']
    assert r['complete'] is True and doc['views'] == r['views'] > 0 and doc['views_by_src'] == r['views_by_src']
    assert doc['bytes_per_view'] == r['bytes_per_view']
    assert doc['bytes_per_view_parts'] == {'x': r['x_bytes_per_view'], 'views_table': r['views_bytes_per_view']}
    assert doc['matches'] == r['matches'] == len(REAL_MATCHES) and doc['matches_with_views'] <= len(REAL_MATCHES)
    assert (doc['peak_private_gib'], doc['peak_private_main_gib'], doc['peak_private_worker_gib']) == (
        r['peak_private_gib'], r['peak_private_main_gib'], r['peak_private_worker_gib'])
    assert (doc['seconds_per_match'], doc['wall_s'], doc['workers']) == (r['seconds_per_match'], r['wall_s'], 1)
    assert doc['run']['args'] == {'workers': 1, 'block_matches': 4, 'n_matches': len(REAL_MATCHES), 'macros': list(synth.S1_MACROS),
                                  'limit': None}                                      # args.info of the builder is read like the old layout
    assert (doc['run']['calls'], doc['run']['recovered'], doc['run']['warnings']) == (1, False, [])
    assert doc['run']['code_state_at_end'] == r['code_state_at_end'] and len(doc['run']['per_call']) == 1
    assert doc['s1_run'] == real_s1.name == man['s1']['run']
    # the terrain section and RUN.json carry the same grid record, the one of the current code
    assert doc['terrain']['grid'] == man['terrain']['grid'] == GRID_RECORD and doc['terrain']['built_from_this_s1'] is True
    assert doc['terrain']['hashes_match_run_json'] is True


def test_the_report_of_a_real_build_s2_run_finds_no_inconsistency_and_counts_the_s1_rows(real_run, real_s1):
    run, doc = real_run
    r = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))['s2_result']
    assert doc['checks'] == {'views_agree_with_run_json': True, 'frame_clock_rows_on_their_frame': True,
                             'frame_age_matches_summary': True, 'game_time_matches_ref': True}
    codes = set(flag_codes(doc))
    assert not codes & {'VIEW_TABLE_INCONSISTENT', 'NAN_OUTSIDE_ALLOWED', 'RUN_RECORD_RECOVERED', 'BUILD_WARNINGS',
                        'STRATA_WAS_RUNNING_AT_BUILD', 'TERRAIN_NOT_FROM_THIS_S1', 'PEAK_MEMORY_NOT_RECORDED'}, doc['flags']
    # the store: the report and the builder's record count the same files, the stored arrays (STORE.json is left out of both), and so
    # agree to the byte
    assert doc['store']['verified'] is True and doc['store']['files'] == r['store_files'] == len(Store(run / 's2').stored_files())
    assert doc['store']['bytes'] == r['store_bytes']
    # every S1 row of the ten matches has a view: the exact count of the extrapolation is the number of views of the store
    assert doc['extrapolated_gib']['detail']['cache_full_build']['views'] == doc['views']
    # the measurements on the (synthetic) records ran on every match with views; no error
    assert doc['raw_measurement']['measured'] is True and doc['raw_measurement']['errors'] == {}
    assert doc['raw_measurement']['matches_measured'] == doc['matches_with_views']
    assert sorted(p.name for p in (run / 'report').iterdir()) == sorted(f'{n}.png' for n in doc['figures'])


def test_a_real_build_s2_run_is_reported_without_keys_and_without_raw_matches_when_no_records_are_given(real_run, real_s1):
    run, doc = real_run
    assert 'SYN_' not in (run / 'S2_REPORT.json').read_text(encoding='utf-8')
    again = json.loads(report_s2.report(run, s1=real_s1, terrain_dir=run.parent / 'terrain', json_path=run.parent / 'again.json',
                                        report_dir=run.parent / 'again').read_text(encoding='utf-8'))
    assert again['raw_measurement']['measured'] is False and again['views'] == doc['views']             # records=None: files only
    same = ('views', 'views_by_src', 'bytes_per_view', 'overflow', 'nan_share', 'constant_columns', 'unknown', 'suspect_positions')
    assert {k: again[k] for k in same} == {k: doc[k] for k in same}
