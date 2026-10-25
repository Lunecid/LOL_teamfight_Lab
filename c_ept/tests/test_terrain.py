"""Tests of ``c_ept.terrain``: the walkable grid that the FIT positions make, its all-pairs geodesic table, snapping, zones, wall fraction.

The always-on tests use hand-made grids and a tiny hand-made S1 store (arrays with the dtypes of ``run_s1.SPEC``, so no match key of the
real data is involved). The probe-equality test and the pinned counts of the real grid run only with ``C_EPT_S1_RUN`` set and read the S1
store and the prototype file read only. An all-pairs table costs about 4 s and 63 MB, so the open grid is built once per session (R11-g).
No test calls ``build_terrain`` on a real S1 run or writes outside ``tmp_path`` (S2-6, R11-h). Run with ``--tb=short``.
"""
import gc
import hashlib
import heapq
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from c_ept import config as C
from c_ept.workflows import run_s1
from c_ept import runlog
from c_ept.data import store_core
from c_ept.preprocessing import terrain
from c_ept.tests.conftest import PROTECTED_IMPORT_PRELUDE, PROTO, import_watch_names, module_level_imports

REPO = Path(C.__file__).parents[1]
N = 75
OPEN = np.ones((N, N), bool)
ZERO100 = np.zeros((150, 150))
S1_MODULES = import_watch_names(('c_ept.targets', 'c_ept.event_rows', 'c_ept.frame_rows', 'c_ept.store_core', 'c_ept.run_s1'))
NAN = float('nan')
DIAG = 200 * 2 ** 0.5                                                                                         # one diagonal step


# ---- helpers --------------------------------------------------------------------------------------------------------------------
def centre(ix, iy):
    """(x, y) of the centre of the 200 u cell (ix, iy)."""
    return 200.0 * ix + 100.0, 200.0 * iy + 100.0


def fake_table(walk):
    """A table with the right diagonal (0 on walkable cells, 65,535 on walls) and 65,535 everywhere else: enough for a ``Terrain`` that
    is never asked for a distance, without the 4 s of an all-pairs run."""
    t = np.full((N * N, N * N), 65_535, np.uint16)
    t[np.diag_indices(N * N)] = np.where(walk.ravel(), 0, 65_535)
    return t


def walled_grid():
    """A hand-made grid: a 55 x 55 room with a one-cell hole in its middle, an island of 2 x 2 cells, and an island of one cell that
    touches the room's corner cell (59, 59) only diagonally (a diagonal step needs both orthogonal neighbours, so it is cut off)."""
    w = np.zeros((N, N), bool)
    w[5:60, 5:60] = True
    w[30, 30] = False
    w[66:68, 66:68] = True
    w[60, 60] = True
    return w


def room_cells():
    """The cell numbers (75 iy + ix) of the room of ``walled_grid``, ascending, from the room's own shape."""
    m = np.zeros((N, N), bool)
    m[5:60, 5:60] = True
    m[30, 30] = False
    return np.flatnonzero(m.ravel())


def snap_reference(cells, x, y):
    """Brute force: the cell of ``cells`` whose centre is nearest to (x, y); a tie goes to the smaller cell number."""
    cx, cy = 200.0 * (cells % N) + 100.0, 200.0 * (cells // N) + 100.0
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return int(cells[np.lexsort((cells, d2))[0]])


def dijkstra_row(walk, src):
    """The row of the geodesic table for the source cell ``src``, from a plain Dijkstra with the rules written out: eight neighbours,
    200 straight and 200 sqrt(2) diagonal, a diagonal step only if both orthogonal neighbours are walkable; each path length is
    rounded once; a wall cell and a cell without a path are 65,535."""
    sx, sy = src % N, src // N
    if not walk[sy, sx]:
        return np.full(N * N, 65_535, dtype=np.uint16)
    dist = {(sx, sy): 0.0}
    heap = [(0.0, sx, sy)]
    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist[(x, y)]:
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if (dx == dy == 0) or not (0 <= nx < N and 0 <= ny < N) or not walk[ny, nx]:
                    continue
                if dx and dy and not (walk[y, nx] and walk[ny, x]):
                    continue
                nd = d + (DIAG if dx and dy else 200.0)
                if nd < dist.get((nx, ny), math.inf):
                    dist[(nx, ny)] = nd
                    heapq.heappush(heap, (nd, nx, ny))
    row = np.full(N * N, 65_535, dtype=np.uint16)
    for (x, y), d in dist.items():
        row[y * N + x] = round(d)
    return row


def wall_fraction_reference(walk, x, y):
    """Share of the 5 x 5 window around the cell that holds (x, y) that is not walkable; cells outside the grid are walls (loops)."""
    cx, cy = math.floor(x / 200.0), math.floor(y / 200.0)
    free = sum(1 for dy in range(-2, 3) for dx in range(-2, 3) if 0 <= cx + dx < N and 0 <= cy + dy < N and walk[cy + dy, cx + dx])
    return 1.0 - free / 25.0


def zone_reference(x, y):
    """The rule of the zone probe (D:\\LOL_Data\\runs\\20261020_terrain_spatial_probe_1, zone_abs) in its literal u, v form."""
    u, v = np.asarray(x, dtype=float) / 14_875.0, np.asarray(y, dtype=float) / 14_875.0
    z = np.full(u.shape, -1, dtype=int)
    s = u + v
    z[s < 0.36] = 0
    z[(z < 0) & (s > 1.64)] = 1
    z[(z < 0) & ((u < 0.13) | (v > 0.87))] = 2
    z[(z < 0) & ((v < 0.13) | (u > 0.87))] = 4
    z[(z < 0) & (np.abs(u - v) < 0.07)] = 3
    z[(z < 0) & (np.abs(s - 1) < 0.07)] = 5
    jg = z < 0
    z[jg & (u < v) & (s < 1)] = 6
    z[jg & (u > v) & (s < 1)] = 7
    z[jg & (u < v) & (s > 1)] = 8
    z[jg & (u > v) & (s > 1)] = 9
    return z


def write_mini_s1(run, matches):
    """A complete S1 store under ``run / 's1'`` that holds only the arrays terrain reads, with the dtypes of ``run_s1.SPEC``.

    ``matches``: a list of dicts ``role`` (a name of ``config.ROLES``), ``frames`` ((F, 10, 2): x, y of the regular frames), ``end``
    ((10, 2): the game-end frame) and ``kills`` ((K, 2)). The key column holds a sentinel text per match, so that a test can check that
    no key reaches an output. Returns ``run``.
    """
    spec = run_s1.SPEC
    root = Path(run) / run_s1.STORE_DIR
    frames = [np.asarray(m['frames'], dtype=float).reshape(-1, 10, 2) for m in matches]
    kills = [np.asarray(m['kills'], dtype=float).reshape(-1, 2) for m in matches]
    num = np.zeros((sum(len(f) for f in frames), 10, len(C.FRAME_FIELDS)), dtype=spec['frames/num'][0])
    num[:, :, :2] = np.concatenate(frames)
    kxy = np.concatenate(kills)
    arrays = {
        'matches/key': np.array([f'SENTINEL_KEY_{i}'.encode('ascii') for i in range(len(matches))], dtype=spec['matches/key'][0]),
        'matches/role': np.array([C.ROLES.index(m['role']) for m in matches], dtype=spec['matches/role'][0]),
        'matches/end_xy': np.stack([np.asarray(m['end'], dtype=float) for m in matches]).astype(spec['matches/end_xy'][0]),
        'matches/frame_off': np.concatenate([[0], np.cumsum([len(f) for f in frames])]).astype('i8'),
        'matches/kill_off': np.concatenate([[0], np.cumsum([len(k) for k in kills])]).astype('i8'),
        'frames/num': num,
        'kills/x': kxy[:, 0].astype(spec['kills/x'][0]), 'kills/y': kxy[:, 1].astype(spec['kills/y'][0]),
    }
    s = store_core.Store(root)
    for name, a in arrays.items():
        p = s.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, a)
    s.finalize()
    return Path(run)


HITS = terrain.MIN_COUNT      # hits that make a cell walkable (S2 gate S2G-2, 2026-10-21: 150, was 5); the mini matches below scale with it
BLOCK = [(ix, iy) for ix in (10, 11, 12) for iy in (10, 11, 12)]       # nine cells that become walkable and form the main component
ISLAND = (30, 30)                                                      # one cell with exactly MIN_COUNT hits: a component of its own


def mini_matches():
    """Four matches: FIT (rows 0, 2 and 3; the last has no kill) and DEV (row 1, positions in a cell that must stay a wall). The special
    positions are the ones that ``MINI_CENSUS`` counts. Each block cell and the island get ``HITS`` hits in match 0, so they are walkable
    at ``terrain.MIN_COUNT``; the cells of the special positions hold at most 3 hits each (``HITS`` must stay above 3); the DEV cell holds
    ``10 * HITS + 14`` hits, so it would be walkable were the DEV match read."""
    a_frames = [centre(*c) for c in BLOCK for _ in range(HITS)] + [centre(*ISLAND)] * HITS
    a_frames += [(0.0, 0.0), (NAN, NAN), (NAN, 100.0), (15_000.0, 15_000.0), (16_000.0, 500.0), (14_875.0, 14_875.0)]
    a_frames += [centre(10, 10)] * 4                                                    # 10 HITS + 10 positions: whole frames of 10 players
    a_end = [centre(10, 10)] * 5 + [(NAN, NAN)] * 2 + [(0.0, 0.0), (-5.0, 100.0), (14_900.0, 100.0)]
    a_kills = [centre(12, 12)] * 3 + [(NAN, NAN), (0.0, 0.0), (-1.0, -1.0), (15_000.0, 100.0), (14_876.0, 2_000.0)]
    a_kills += [(0.0, 3_000.0), (3_000.0, 0.0)]                                         # one zero coordinate only: not (0, 0)
    c_frames = [centre(11, 10)] * 10 + [centre(11, 11)] * 10                            # 2 frames
    dev = centre(60, 60)
    return [dict(role='FIT', frames=a_frames, end=a_end, kills=a_kills),
            dict(role='DEV', frames=[dev] * (10 * HITS), end=[dev] * 10, kills=[dev] * 4),
            dict(role='FIT', frames=c_frames, end=[centre(11, 11)] * 10, kills=[centre(10, 12)] * 2),
            dict(role='FIT', frames=[centre(12, 10)] * 10, end=[centre(12, 10)] * 10, kills=[])]


FIT_ROWS = [0, 2, 3]
MINI_CENSUS = {      # counted by hand from the FIT matches of mini_matches: n, NaN, exactly (0, 0), outside [0, 14,875]^2, not binned
    'regular_frames': dict(n=10 * HITS + 40, nan=2, zero_zero=1, outside_map=2, not_binned=3),     # 10 HITS + 10, 20 and 10 positions
    'game_end_frame': dict(n=30, nan=2, zero_zero=1, outside_map=2, not_binned=3),
    'kills': dict(n=12, nan=1, zero_zero=1, outside_map=3, not_binned=2),
}
MINI_BINNED = {'frames': 10 * HITS + 64, 'kills': 10}       # (10 HITS + 40 - 3) + (30 - 3) frame positions, 12 - 2 kills: in the histogram


def points_of(m, key):
    """The (x, y) pairs of one mini match: ``'frames'`` (regular and game-end frame), or ``'kills'``."""
    if key == 'kills':
        return [tuple(p) for p in np.asarray(m['kills'], dtype=float).reshape(-1, 2)]
    return [tuple(p) for p in np.asarray(m['frames'], dtype=float).reshape(-1, 2)] + [tuple(p) for p in np.asarray(m['end'], dtype=float)]


def histogram_reference(points):
    """The (150, 150) counts [ix, iy] of the positions that lie in [0, 15,000]^2 (both ends included), counted one point at a time."""
    h = np.zeros((150, 150))
    for x, y in points:
        if 0.0 <= x <= 15_000.0 and 0.0 <= y <= 15_000.0:
            h[min(int(x // 100), 149), min(int(y // 100), 149)] += 1
    return h


# ---- fixtures -------------------------------------------------------------------------------------------------------------------
# ``open_table``, ``open_dir`` and ``open_terrain`` (the open grid: its all-pairs table is built once per session, R11-g, R13-j) are in
# conftest.py since Task 12, where the view tests share them.
@pytest.fixture(scope='module')
def walled_dir(tmp_path_factory):
    w = walled_grid()
    out = tmp_path_factory.mktemp('walled_terrain')
    terrain.write_terrain_files(out, w, ZERO100, ZERO100, terrain.geodesic_table(w))
    yield out
    gc.collect()
    shutil.rmtree(out, ignore_errors=True)


@pytest.fixture
def walled_terrain(walled_dir):
    t = terrain.Terrain(walled_dir)
    yield t
    del t


@pytest.fixture(autouse=True)
def _remove_big_tmp_files(tmp_path):
    """A test that writes a 63 MB table removes it afterwards: pytest keeps the temporary folders of the last three runs."""
    yield
    gc.collect()
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture(scope='module')
def s1_run():
    run = os.environ.get('C_EPT_S1_RUN')
    if not run:
        pytest.skip('C_EPT_S1_RUN is not set: these tests read the S1 store of the S1 run')
    return Path(run)


def load_s1(run):
    """The S1 store of the run, read only. (The SHA-256 of every file is checked by the golden tests, not repeated here.)"""
    return run_s1.load_s1(run, verify=False)


def fit_rows(run):
    return np.flatnonzero(np.asarray(load_s1(run).array('matches/role')) == C.ROLES.index('FIT'))


@pytest.fixture(scope='module')
def real_scan(s1_run):
    return terrain.scan_positions(load_s1(s1_run), fit_rows(s1_run))


# ---- the walkable grid (spec section 11, item 12) --------------------------------------------------------------------------------
def test_pool_2x2():
    """The 2 x 2 histogram cells below a walkable cell add up: MIN_COUNT hits in all make it walkable, one hit less does not."""
    f = np.zeros((150, 150)); f[0, 0] = HITS - 2; f[1, 1] = 2; assert terrain.walkable(f, np.zeros_like(f))[0, 0]
    f[1, 1] = 1                                       # one hit short
    assert not terrain.walkable(f, np.zeros_like(f))[0, 0]


def test_walkable_threshold_orientation_and_shape():
    """MIN_COUNT hits make a cell walkable, one less does not; frames and kills add up; the result is [iy, ix] while histogram2d gives
    [ix, iy]."""
    f, k = np.zeros((150, 150)), np.zeros((150, 150))
    f[20, 40] = HITS - 1                              # x in [2000, 2100), y in [4000, 4100): cell (ix 10, iy 20) is one hit short
    f[60, 80], k[61, 80] = HITS - 3, 3                # cell (ix 30, iy 40): (HITS - 3) + 3 = MIN_COUNT hits
    f[10, 0] = HITS                                   # x in [1000, 1100), y in [0, 100): cell ix 5, iy 0
    w = terrain.walkable(f, k)
    assert w.shape == (N, N) and w.dtype == bool
    assert not w[20, 10] and w[40, 30] and w[0, 5]
    assert not w[5, 0] and w.sum() == 2               # the transposed cell is not walkable
    with pytest.raises(ValueError):
        terrain.walkable(np.zeros((75, 75)), np.zeros((75, 75)))


def test_constants_are_the_briefs():
    assert (terrain.GRID100, terrain.GRID, terrain.MIN_COUNT) == (150, 75, 150)      # S2 gate S2G-2, 2026-10-21: MIN_COUNT was 5
    assert terrain.ZONES == ('base_blue', 'base_red', 'top', 'mid', 'bot', 'river',
                             'jg_blue_top', 'jg_blue_bot', 'jg_red_top', 'jg_red_bot')


# ---- the geodesic table ----------------------------------------------------------------------------------------------------------
def test_geodesic_straight_line(open_table):
    g = open_table; assert g[0, 10] == 2_000          # the same row, ten cells


def test_geodesic_wall_detour():
    w = np.ones((75, 75), bool); w[10, :70] = False   # a wall of 70 cells with a gap at the end
    g = terrain.geodesic_table(w); a, b = 9 * 75 + 0, 11 * 75 + 0
    assert g[a, b] > 400 and g[a, b] != 65_535 and g[a, b] == g[b, a]
    # 70 cells to the gap and 70 back, and two steps through it; no diagonal can be used at the wall's end (no corner cutting)
    assert g[a, b] == 142 * 200


def test_geodesic_no_corner_cutting():
    w = np.ones((75, 75), bool); w[0, 1] = w[1, 0] = False; g = terrain.geodesic_table(w)
    assert g[0, 76] == 65_535                         # no diagonal step out of (0, 0), so it is cut off
    assert g[0, 0] == 0 and g[0, 2] == 65_535         # it reaches only itself
    assert (g[1] == 65_535).all() and (g[:, 1] == 65_535).all() and (g[75] == 65_535).all()      # a wall cell: 65,535, also against itself


def test_geodesic_open_grid_values(open_table):
    g = open_table
    assert g.shape == (5625, 5625) and g.dtype == np.uint16
    assert np.array_equal(g, g.T) and not g.diagonal().any()
    assert g[0, 76] == round(DIAG)                                               # one diagonal step
    assert g[0, 5 * 75 + 3] == round(3 * DIAG + 2 * 200)                         # three diagonal steps and two straight ones: rounded once
    assert g.max() == g[0, 74 * 75 + 74] == round(74 * DIAG) == 20_930           # the longest line of the open grid, far below 65,535
    assert 65_535 not in g                                                       # nothing is unreachable on the open grid


def test_geodesic_table_equals_a_plain_dijkstra_on_a_grid_with_islands_holes_and_diagonal_contacts(walled_terrain):
    w = walled_grid()
    rng = np.random.default_rng(4)
    sources = [int(c) for c in rng.choice(np.flatnonzero(w.ravel()), 12, replace=False)]
    sources += [59 * 75 + 59, 60 * 75 + 60, 66 * 75 + 66, 29 * 75 + 30, 30 * 75 + 30, 0]       # corner, islands, beside the hole, two walls
    for src in sources:
        assert np.array_equal(dijkstra_row(w, src), walled_terrain.geodesic[src]), src
    g = np.asarray(walled_terrain.geodesic)
    assert np.array_equal(g, g.T)
    assert (g[60 * 75 + 60, np.flatnonzero(w.ravel())] == 65_535).sum() == w.sum() - 1                 # an island reaches only itself


def test_geodesic_range_check_raises_when_a_reachable_distance_is_65535_or_more():
    """R11-g / S2-4: a serpentine corridor of 38 rows is far longer than a uint16 holds; the cast must raise, not wrap or clip."""
    w = np.zeros((75, 75), bool)
    w[0::2, :] = True                                 # corridors on the even rows
    w[1::4, 74] = True                                # connected at the right end after rows 0, 4, 8, ...
    w[3::4, 0] = True                                 # and at the left end after rows 2, 6, 10, ...
    with pytest.raises(ValueError, match='geodesic'):
        terrain.geodesic_table(w)


def test_geodesic_table_checks_its_input():
    with pytest.raises(ValueError):
        terrain.geodesic_table(np.ones((74, 75), bool))
    with pytest.raises(TypeError):
        terrain.geodesic_table(np.ones((75, 75), int))


# ---- snapping and the distance of two positions ------------------------------------------------------------------------------------
def test_geodesic_snaps_offgrid_points(open_terrain):
    d = open_terrain.geo(np.array([-50.0]), np.array([15_100.0]), np.array([7_000.0]), np.array([7_000.0]))
    assert np.isfinite(d).all() and d[0] > 0


def test_snap_off_grid_and_ties_go_to_the_smaller_cell_number(open_terrain):
    t = open_terrain
    assert t.snap(np.array([-50.0]), np.array([15_100.0]))[0] == 74 * 75 + 0                # the nearest centre of the grid
    assert t.snap(np.array([7_000.0]), np.array([7_000.0]))[0] == 34 * 75 + 34             # a corner of four cells: the smallest number
    d = t.geo(np.array([-50.0]), np.array([15_100.0]), np.array([7_000.0]), np.array([7_000.0]))
    assert d.dtype == np.float32 and d[0] == round(34 * DIAG + 6 * 200)                    # (0, 74) to (34, 34): 34 diagonals, 6 straight
    assert t.snap(np.array([6_000.0]), np.array([5_100.0]))[0] == 25 * 75 + 29            # on an edge between two cells: the smaller number
    assert t.snap(np.array([15_000.0]), np.array([15_000.0]))[0] == 74 * 75 + 74


def test_snap_inside_a_walkable_cell_is_that_cell(walled_terrain):
    rng = np.random.default_rng(3)
    cells = room_cells()
    x = 200.0 * (cells % N) + rng.uniform(1, 199, cells.size)
    y = 200.0 * (cells // N) + rng.uniform(1, 199, cells.size)
    assert np.array_equal(walled_terrain.snap(x, y), cells)


def test_snap_goes_only_to_the_largest_component(walled_terrain):
    t, cells = walled_terrain, room_cells()
    assert t.snap(*centre(66, 66)) == snap_reference(cells, *centre(66, 66)) and t.snap(*centre(66, 66)) in cells    # the 2 x 2 island
    assert t.snap(*centre(67, 67)) in cells
    assert t.snap(*centre(60, 60)) == 59 * 75 + 59                                         # the diagonal island: the corner of the room
    assert not t.main[60, 60] and not t.main[66, 66] and t.main[59, 59] and t.main.sum() == cells.size


def test_snap_a_hole_is_a_four_way_tie_and_the_smallest_number_wins(walled_terrain):
    assert walled_terrain.snap(*centre(30, 30)) == 29 * 75 + 30                            # the four neighbours are 200 u away
    assert walled_terrain.snap(6_200.0, 6_100.0) == 30 * 75 + 31                           # on the edge to the right neighbour: nearest


def test_snap_equals_brute_force_on_random_integer_and_boundary_points(walled_terrain):
    rng = np.random.default_rng(11)
    cells = room_cells()
    x = np.concatenate([rng.integers(-600, 15_600, 700).astype(float), 200.0 * rng.integers(-2, 78, 300), rng.uniform(-300, 15_300, 300)])
    y = np.concatenate([rng.integers(-600, 15_600, 700).astype(float), 200.0 * rng.integers(-2, 78, 300), rng.uniform(-300, 15_300, 300)])
    got = walled_terrain.snap(x, y)
    want = np.array([snap_reference(cells, a, b) for a, b in zip(x, y)])
    assert got.dtype.kind == 'i' and got.shape == x.shape and np.array_equal(got, want)


def test_missing_and_infinite_positions(open_terrain):
    t = open_terrain
    nan = np.array([np.nan, 100.0, np.nan]), np.array([100.0, np.nan, np.nan])
    assert (t.snap(*nan) == -1).all() and (t.zone(*nan) == -1).all()
    assert np.isnan(t.wall_fraction(*nan)).all() and np.isnan(t.geo(*nan, *nan)).all()
    d = t.geo(np.array([100.0, np.nan]), np.array([100.0, 5.0]), np.array([300.0, 300.0]), np.array([100.0, 100.0]))
    assert d[0] == 200 and np.isnan(d[1])                                                   # a missing position spoils its own pair only
    for bad in (np.inf, -np.inf):
        with pytest.raises(ValueError):
            t.snap(np.array([bad]), np.array([0.0]))
        with pytest.raises(ValueError):
            t.wall_fraction(np.array([0.0]), np.array([bad]))


def test_geo_broadcasts_is_symmetric_and_float32(walled_terrain):
    t = walled_terrain
    xs, ys = np.array([[1_100.0], [5_000.0], [-300.0]]), np.array([[1_100.0], [7_300.0], [900.0]])
    d = t.geo(xs, ys, xs.T, ys.T)
    assert d.shape == (3, 3) and d.dtype == np.float32 and np.isfinite(d).all()
    assert np.array_equal(d, d.T) and not np.diag(d).any()


def test_geo_raises_when_the_table_holds_the_unreachable_marker(open_table, tmp_path):
    """R11-b: a looked-up 65,535 is never returned as a distance."""
    bad = open_table.copy()
    bad[0, 74 * 75 + 74] = bad[74 * 75 + 74, 0] = 65_535
    terrain.write_terrain_files(tmp_path, OPEN, ZERO100, ZERO100, bad)
    t = terrain.Terrain(tmp_path)
    try:
        with pytest.raises(ValueError, match='65,535'):
            t.geo(np.array([100.0]), np.array([100.0]), np.array([14_900.0]), np.array([14_900.0]))
        assert t.geo(np.array([100.0]), np.array([100.0]), np.array([14_700.0]), np.array([14_900.0]))[0] > 0      # other pairs are fine
    finally:
        del t
        gc.collect()


# ---- islands (R11-b) ----------------------------------------------------------------------------------------------------------
def test_components_count_the_islands():
    w = walled_grid()
    c = terrain.components(w)
    assert sorted(c.sizes.tolist()) == [1, 4, 3024]
    assert c.main.shape == (N, N) and int(c.main.sum()) == 3024 and c.main[5, 5] and not c.main[30, 30]
    assert (c.labels[~w] == -1).all() and (c.labels[w] >= 0).all()
    assert c.labels[60, 60] != c.labels[59, 59]                                     # a diagonal contact is no connection
    assert c.labels[66, 66] == c.labels[67, 67] != c.labels[59, 59]


def test_components_pick_the_largest_and_break_a_size_tie_by_the_smaller_cell_number():
    w = np.zeros((N, N), bool)
    w[50, 50:52] = True                                  # two cells, high cell numbers
    w[3, 3:5] = True                                     # two cells, low cell numbers
    c = terrain.components(w)
    assert c.main[3, 3] and c.main[3, 4] and not c.main[50, 50]
    empty = terrain.components(np.zeros((N, N), bool))
    assert len(empty.sizes) == 0 and not empty.main.any()


def test_terrain_without_walkable_cells_is_refused(tmp_path):
    w = np.zeros((N, N), bool)
    terrain.write_terrain_files(tmp_path, w, ZERO100, ZERO100, fake_table(w))
    with pytest.raises(ValueError, match='walkable'):
        terrain.Terrain(tmp_path)


# ---- wall fraction (R11-c) -----------------------------------------------------------------------------------------------------
def test_wall_fraction_counts_cells_outside_the_grid_as_walls(open_terrain):
    f = open_terrain.wall_fraction(np.array([2_100.0, 100.0, 300.0, -50.0, -5_000.0, 15_000.0, 14_900.0]),
                                   np.array([7_100.0, 100.0, 300.0, 1_100.0, 7_000.0, 1_100.0, 14_900.0]))
    assert f.dtype == np.float32
    np.testing.assert_allclose(f, [0.0, 16 / 25, 9 / 25, 15 / 25, 1.0, 15 / 25, 16 / 25], atol=1e-7)


def test_wall_fraction_matches_a_loop_on_a_random_grid(tmp_path):
    rng = np.random.default_rng(5)
    w = rng.random((N, N)) < 0.7
    terrain.write_terrain_files(tmp_path, w, ZERO100, ZERO100, fake_table(w))
    t = terrain.Terrain(tmp_path)
    try:
        x = np.concatenate([rng.uniform(-1_500, 16_500, 400), 200.0 * rng.integers(-4, 80, 100)])
        y = np.concatenate([rng.uniform(-1_500, 16_500, 400), 200.0 * rng.integers(-4, 80, 100)])
        got = t.wall_fraction(x, y)
        want = np.array([wall_fraction_reference(w, a, b) for a, b in zip(x, y)])
        assert got.dtype == np.float32 and ((got >= 0) & (got <= 1)).all()
        np.testing.assert_allclose(got, want, atol=1e-7)
        assert t.wall_fraction(*centre(10, 10)).shape == ()                       # a scalar position gives a scalar result
    finally:
        del t
        gc.collect()


def test_wall_fraction_uses_the_containing_cell_and_does_not_snap(walled_terrain):
    t, w = walled_terrain, walled_grid()
    # a point in the cut-off cell (60, 60): its own window holds 5 walkable cells (the room's corner 2 x 2 and itself); the window of the
    # cell it would snap to, (59, 59), holds 10
    assert t.wall_fraction(*centre(60, 60)) == pytest.approx(20 / 25, abs=1e-7)
    assert t.wall_fraction(*centre(65, 65)) == pytest.approx(21 / 25, abs=1e-7)           # a wall cell beside the 2 x 2 island: 4 walkable
    assert t.wall_fraction(*centre(30, 30)) == pytest.approx(1 / 25, abs=1e-7)            # the hole in the room
    for x, y in ((6_000.0, 6_000.0), (12_100.0, 12_100.0), (13_000.0, 13_000.0)):
        assert t.wall_fraction(x, y) == pytest.approx(wall_fraction_reference(w, x, y), abs=1e-7)


# ---- zones (R11-a) -------------------------------------------------------------------------------------------------------------
def test_zone_rules_and_swap():
    pts = {(500, 500): 'base_blue', (14_375, 14_375): 'base_red', (7_437, 7_437): 'mid',
           (800, 7_000): 'top', (7_000, 800): 'bot', (3_000, 11_875): 'river'}
    assert {p: terrain.ZONES[terrain.zone_of(*p)] for p in pts} == pts
    rng = np.random.default_rng(1); x, y = rng.uniform(0, 14_875, (2, 10_000))
    assert np.array_equal(terrain.zone_of(14_875 - x, 14_875 - y), terrain.ZONE_SWAP[terrain.zone_of(x, y)])


def test_terrain_zone_is_zone_of(open_terrain):
    rng = np.random.default_rng(6)
    x, y = rng.uniform(-500, 15_500, (2, 1_000))
    assert np.array_equal(open_terrain.zone(x, y), terrain.zone_of(x, y))


def test_zone_swap_is_the_expected_involution_and_comes_from_the_rules():
    names, swap = terrain.ZONES, terrain.ZONE_SWAP
    assert swap.dtype == np.int8 and sorted(swap.tolist()) == list(range(10))              # a permutation of the ten zones
    assert np.array_equal(swap[swap], np.arange(10))                                       # swapping twice is the identity
    assert {names[i]: names[swap[i]] for i in range(10)} == {
        'base_blue': 'base_red', 'base_red': 'base_blue', 'top': 'bot', 'bot': 'top', 'mid': 'mid', 'river': 'river',
        'jg_blue_top': 'jg_red_bot', 'jg_red_bot': 'jg_blue_top', 'jg_blue_bot': 'jg_red_top', 'jg_red_top': 'jg_blue_bot'}
    assert not swap.flags.writeable


def test_zone_of_equals_the_literal_u_v_rule_of_the_probe_also_off_the_map():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-1_000, 16_000, (2, 200_000))
    got = terrain.zone_of(x, y)
    assert got.dtype == np.int8 and got.shape == x.shape
    assert np.array_equal(got, zone_reference(x, y))
    assert set(np.unique(got).tolist()) == set(range(10))                                  # every zone occurs


def test_zone_thresholds_are_strict_on_whole_number_positions():
    """Positions are whole numbers, so the line x + y = 5,355 (u + v = 0.36) carries real positions: strict means the line itself is not
    base_blue, and the line below it is."""
    for total, is_base in ((5_354, True), (5_355, False), (5_356, False)):
        px = np.arange(0.0, total + 1)
        z = terrain.zone_of(px, total - px)
        assert (z == terrain.ZONES.index('base_blue')).all() == is_base
        assert (z == terrain.ZONES.index('base_blue')).any() == is_base
    for total, is_base in ((24_396, True), (24_395, False), (24_394, False)):               # the mirror: u + v = 1.64
        px = np.arange(total - 14_875.0, 14_876.0)
        z = terrain.zone_of(px, total - px)
        assert (z == terrain.ZONES.index('base_red')).all() == is_base
        assert (z == terrain.ZONES.index('base_red')).any() == is_base


def test_zone_swap_is_exact_on_a_lattice_and_on_every_threshold_line():
    """Mirroring a position (x -> 14,875 - x) must give exactly the swapped zone also where a threshold is crossed: on whole-number
    lines and on the half-integer lines of the lane (0.13 W = 1,933.75) and the mid and river bands (0.07 W = 1,041.25)."""
    k = np.unique(np.concatenate([np.arange(0.0, 14_876.0, 25.0), [1_933.75, 12_941.25, 1_041.25, 13_833.75, 5_503.75, 9_371.25]]))
    gx, gy = np.meshgrid(k, k)
    px, py = gx.ravel(), gy.ravel()
    assert np.array_equal(terrain.zone_of(14_875 - px, 14_875 - py), terrain.ZONE_SWAP[terrain.zone_of(px, py)])
    t = np.arange(0.0, 14_876.0)
    for total in (5_354.0, 5_355.0, 5_356.0, 14_875.0 - 1_041.25, 14_875.0 + 1_041.25, 24_394.0, 24_395.0, 24_396.0):     # x + y = total
        ok = (total - t >= 0) & (total - t <= 14_875)
        px, py = t[ok], total - t[ok]
        assert np.array_equal(terrain.zone_of(14_875 - px, 14_875 - py), terrain.ZONE_SWAP[terrain.zone_of(px, py)])
    for diff in (1_041.25, -1_041.25, 0.0):                                                                              # x - y = diff
        ok = (t - diff >= 0) & (t - diff <= 14_875)
        px, py = t[ok], t[ok] - diff
        assert np.array_equal(terrain.zone_of(14_875 - px, 14_875 - py), terrain.ZONE_SWAP[terrain.zone_of(px, py)])


def test_zone_rules_hold_the_briefs_numbers_and_the_swap_constant():
    r = terrain.ZONE_RULES
    assert r['norm'] == C.W == 14_875.0
    assert (r['base_sum_below'], r['base_sum_above'], r['lane'], r['mid_band'], r['river_band']) == (0.36, 1.64, 0.13, 0.07, 0.07)
    assert r['priority'] == ('base', 'lane', 'mid', 'river', 'jungle')
    with pytest.raises(TypeError):
        r['lane'] = 0.2                                                                     # read only


# ---- occupancy from the S1 store (synthetic store) ---------------------------------------------------------------------------------
def test_occupancy_of_a_synthetic_store_counts_the_selected_rows_only(tmp_path):
    ms = mini_matches()
    s = load_s1(write_mini_s1(tmp_path / 'run', ms))
    frames, kills = terrain.occupancy_100(s, FIT_ROWS)
    assert frames.shape == kills.shape == (150, 150)
    assert np.array_equal(frames, histogram_reference([p for i in FIT_ROWS for p in points_of(ms[i], 'frames')]))
    assert np.array_equal(kills, histogram_reference([p for i in FIT_ROWS for p in points_of(ms[i], 'kills')]))
    assert frames[23, 21] == HITS + 10 and frames[21, 23] == HITS    # [ix, iy] as histogram2d gives it: centre (11, 10) is x 2,300, y 2,100
    dev_f, dev_k = terrain.occupancy_100(s, [1])                             # another selection: the DEV match only
    dev_points = 10 * HITS + 10                                              # its frames and its game-end frame
    assert dev_f.sum() == dev_points and dev_k.sum() == 4 and dev_f[121, 121] == dev_points and dev_k[121, 121] == 4
    assert np.array_equal(terrain.occupancy_100(s, np.array([3, 0, 2]))[0], frames)            # the order of the rows does not matter
    assert terrain.occupancy_100(s, np.array([], dtype=np.int64))[0].sum() == 0               # an empty selection counts nothing


def test_occupancy_refuses_bad_row_selections(tmp_path):
    s = load_s1(write_mini_s1(tmp_path / 'run', mini_matches()))
    for rows in ([0, 0], [4], [-1], [[0, 1]], [0.5]):
        with pytest.raises(ValueError):
            terrain.occupancy_100(s, rows)


def test_position_census_counts_suspect_positions_per_kind(tmp_path):
    """R11-d: exactly (0, 0), outside [0, 14,875]^2, NaN, and the positions that the histogram does not bin; FIT rows only."""
    s = load_s1(write_mini_s1(tmp_path / 'run', mini_matches()))
    frames, kills, census = terrain.scan_positions(s, FIT_ROWS)
    assert census == MINI_CENSUS
    assert (int(frames.sum()), int(kills.sum())) == (MINI_BINNED['frames'], MINI_BINNED['kills'])


def test_results_do_not_depend_on_the_chunk_sizes(tmp_path, monkeypatch):
    """The store is read in chunks of matches, Dijkstra runs in chunks of sources, and ``snap`` measures in chunks of positions: a chunk
    boundary must change nothing."""
    s = load_s1(write_mini_s1(tmp_path / 'run', mini_matches()))
    want_scan = terrain.scan_positions(s, FIT_ROWS)
    rng = np.random.default_rng(8)
    w = np.zeros((N, N), bool)
    w[rng.integers(0, N, 160), rng.integers(0, N, 160)] = True                       # a few dozen scattered cells with some neighbours
    w[20:26, 20:26] = True
    want_table = terrain.geodesic_table(w)
    x, y = rng.uniform(-500, 15_500, (2, 400))
    terrain.write_terrain_files(tmp_path / 't', w, ZERO100, ZERO100, want_table)
    t = terrain.Terrain(tmp_path / 't')
    want_snap = t.snap(x, y)
    monkeypatch.setattr(terrain, '_CHUNK_MATCHES', 1)
    monkeypatch.setattr(terrain, '_CHUNK_SOURCES', 7)
    monkeypatch.setattr(terrain, '_CHUNK_SNAP', 3)
    got_scan = terrain.scan_positions(s, FIT_ROWS)
    try:
        assert np.array_equal(got_scan[0], want_scan[0]) and np.array_equal(got_scan[1], want_scan[1]) and got_scan[2] == want_scan[2]
        assert np.array_equal(terrain.geodesic_table(w), want_table)
        assert np.array_equal(t.snap(x, y), want_snap)
    finally:
        del t
        gc.collect()


# ---- build_terrain (R11-d, R11-e, R11-f, R11-h, R11-i) ------------------------------------------------------------------------------
def build(tmp_path, name='run'):
    """A mini S1 run in ``tmp_path`` and the terrain built from it; returns (run folder of the terrain, terrain directory)."""
    s1 = write_mini_s1(tmp_path / f's1_{name}' / '20261020_c_ept_s1_9', mini_matches())
    run = tmp_path / name
    return run, terrain.build_terrain(s1, run)


def read_terrain_json(out):
    return json.loads((out / 'TERRAIN.json').read_text(encoding='utf-8'))


def test_build_terrain_writes_the_three_files_and_the_provenance(tmp_path, monkeypatch):
    states = []
    real_code_state = runlog.code_state

    def code_state_once():
        """The one answer of git for this test: the build asks once, and the check below compares TERRAIN.json with that answer. A second
        ``git rev-parse`` about 0.5 s later saw a commit that another session had made in between, and the test failed (final review)."""
        states.append(real_code_state())
        return states[-1]
    monkeypatch.setattr(runlog, 'code_state', code_state_once)
    run, out = build(tmp_path)
    try:
        assert out == run / 'terrain' and sorted(p.name for p in out.iterdir()) == ['TERRAIN.json', 'geodesic.npy', 'terrain.npz']
        assert not (run / 'terrain.tmp').exists()
        info = json.loads((out / 'TERRAIN.json').read_text(encoding='utf-8'))
        s1_files = json.loads((tmp_path / 's1_run' / '20261020_c_ept_s1_9' / 's1' / 'STORE.json').read_text(encoding='utf-8'))['files']
        assert info['s1_run'] == '20261020_c_ept_s1_9' and info['fit_matches'] == 3
        want_hash = hashlib.sha256(json.dumps(s1_files, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        assert info['s1_files_sha256'] == want_hash == runlog.sha256_json(s1_files)
        assert info['walkable_cells'] == 10 and info['components'] == {'count': 2, 'largest_cells': 9, 'outside_largest': 1}
        assert info['positions'] == MINI_CENSUS
        assert info['sha256'] == {'terrain.npz': store_core.sha(out / 'terrain.npz'), 'geodesic.npy': store_core.sha(out / 'geodesic.npy')}
        assert len(states) == 1 and {k: info[k] for k in ('code_commit', 'code_dirty')} == states[0]       # git was asked once: its answer
        if (REPO / '.git').exists() and shutil.which('git'):                       # None is the answer of "no git" only
            assert re.fullmatch(r'[0-9a-f]{40}', info['code_commit']) and isinstance(info['code_dirty'], bool)
        else:
            assert info['code_commit'] is None and info['code_dirty'] is None
        assert info['zone_rules']['base_sum_below'] == 0.36
        with np.load(out / 'terrain.npz') as z:
            want = np.zeros((N, N), bool)
            for ix, iy in BLOCK + [ISLAND]:
                want[iy, ix] = True
            assert sorted(z.files) == ['frames100', 'kills100', 'walk'] and np.array_equal(z['walk'], want)     # the DEV match left no cell
            assert (z['frames100'].sum(), z['kills100'].sum()) == (MINI_BINNED['frames'], MINI_BINNED['kills'])
        t = terrain.Terrain(out)
        try:
            assert t.manifest == info and t.verify() == info['sha256']
            assert t.geodesic.dtype == np.uint16 and t.geodesic.shape == (5625, 5625)
            assert t.geo(*centre(10, 10), *centre(12, 12))[()] == round(2 * DIAG)
            assert t.main.sum() == 9 and t.snap(*centre(*ISLAND)) == 12 * 75 + 12           # the island point snaps to a block cell
        finally:
            del t
            gc.collect()
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_build_terrain_carries_no_key_and_is_reproducible(tmp_path):
    run_a, out_a = build(tmp_path, 'a')
    run_b, out_b = build(tmp_path, 'b')
    try:
        for out in (out_a, out_b):
            for f in out.iterdir():
                data = f.read_bytes()
                assert b'SENTINEL_KEY' not in data and b'puuid' not in data and b'riotId' not in data, f.name
        assert read_terrain_json(out_a) == read_terrain_json(out_b)
        assert store_core.sha(out_a / 'terrain.npz') == store_core.sha(out_b / 'terrain.npz')                 # no time stamp in the archive
        assert store_core.sha(out_a / 'geodesic.npy') == store_core.sha(out_b / 'geodesic.npy')
        with zipfile.ZipFile(out_a / 'terrain.npz') as zf:
            assert {i.date_time for i in zf.infolist()} == {(1980, 1, 1, 0, 0, 0)}         # a fixed date, so equal inputs give equal bytes
    finally:
        shutil.rmtree(run_a, ignore_errors=True)
        shutil.rmtree(run_b, ignore_errors=True)


def test_build_terrain_refuses_to_overwrite_and_leaves_the_existing_folder_alone(tmp_path):
    s1 = write_mini_s1(tmp_path / 's1_run' / '20261020_c_ept_s1_9', mini_matches())
    run = tmp_path / 'run'
    (run / 'terrain').mkdir(parents=True)
    (run / 'terrain' / 'marker.txt').write_text('kept', encoding='utf-8')
    with pytest.raises(FileExistsError):
        terrain.build_terrain(s1, run)
    assert [p.name for p in (run / 'terrain').iterdir()] == ['marker.txt'] and not (run / 'terrain.tmp').exists()
    with pytest.raises(FileExistsError):                                              # refused before the S1 run is even looked at
        terrain.build_terrain(tmp_path / 'no_such_s1_run', run)
    shutil.rmtree(run / 'terrain')
    (run / 'terrain.tmp').mkdir()                                                    # the leftover of an interrupted build is refused too
    with pytest.raises(FileExistsError):
        terrain.build_terrain(s1, run)
    assert not (run / 'terrain').exists()


def test_build_terrain_stops_at_the_strata_check_before_reading_or_writing(tmp_path, fake_processes):
    """S2-1 / R11-e: the E6 check comes first. The S1 path does not exist, so a read of the store would be a different error."""
    fake_processes([(4242, 'python.exe', ['python.exe', 'D:/Strata/serve/server.py'])])
    with pytest.raises(run_s1.StrataRunning):
        terrain.build_terrain(tmp_path / 'no_such_s1_run', tmp_path / 'run')
    assert not (tmp_path / 'run').exists()


def test_build_terrain_needs_fit_matches(tmp_path):
    only_dev = [dict(m, role='DEV') for m in mini_matches()]
    s1 = write_mini_s1(tmp_path / 's1_run' / '20261020_c_ept_s1_9', only_dev)
    with pytest.raises(ValueError, match='FIT'):
        terrain.build_terrain(s1, tmp_path / 'run')
    assert not (tmp_path / 'run' / 'terrain').exists() and not (tmp_path / 'run' / 'terrain.tmp').exists()


def test_build_terrain_takes_the_store_folder_as_well_as_the_run_folder(tmp_path):
    s1 = write_mini_s1(tmp_path / 's1_run' / '20261020_c_ept_s1_9', mini_matches())
    out = terrain.build_terrain(s1 / run_s1.STORE_DIR, tmp_path / 'run')
    try:
        assert json.loads((out / 'TERRAIN.json').read_text(encoding='utf-8'))['s1_run'] == '20261020_c_ept_s1_9'
    finally:
        shutil.rmtree(tmp_path / 'run', ignore_errors=True)


def test_build_terrain_records_the_git_values_that_runlog_computes(tmp_path, monkeypatch):
    """TERRAIN.json takes the commit and the dirty flag from ``runlog.code_state`` (git is a stand-in here, so the answer is fixed)."""
    def git(*args):
        return 'a' * 40 + '\n' if args[0] == 'rev-parse' else ' M c_ept/x.py\n'
    monkeypatch.setattr(runlog, '_git', git)
    run, out = build(tmp_path)
    try:
        info = read_terrain_json(out)
        assert (info['code_commit'], info['code_dirty']) == ('a' * 40, True)
        assert {k: info[k] for k in ('code_commit', 'code_dirty')} == runlog.code_state()
    finally:
        shutil.rmtree(run, ignore_errors=True)


def tree(path):
    return sorted(p.relative_to(path).as_posix() for p in path.rglob('*'))


def test_build_terrain_refuses_a_run_that_is_or_lies_inside_the_s1_run(tmp_path):
    """The S1 run is read only: however the S1 run and the run folder are given, nothing is written into it."""
    s1 = write_mini_s1(tmp_path / 's1_run' / '20261020_c_ept_s1_9', mini_matches())
    before = tree(tmp_path)
    for given in (s1, s1 / run_s1.STORE_DIR):                                       # the run folder, or its store folder
        for run in (s1, s1 / run_s1.STORE_DIR, s1 / 'later', s1 / 'a' / 'b', s1 / '..' / s1.name, s1 / run_s1.STORE_DIR / 'matches'):
            with pytest.raises(ValueError, match='S1 run'):
                terrain.build_terrain(given, run)
    assert tree(tmp_path) == before


def test_build_terrain_accepts_a_folder_beside_the_s1_run_even_with_the_same_start_of_name(tmp_path):
    s1 = write_mini_s1(tmp_path / 's1_run' / '20261020_c_ept_s1_9', mini_matches())
    run = s1.with_name(s1.name + '_other')                                          # a string prefix of the S1 run's path is not "inside"
    out = terrain.build_terrain(s1, run)
    try:
        assert out == run / 'terrain' and (out / 'TERRAIN.json').is_file()
    finally:
        shutil.rmtree(run, ignore_errors=True)


# ---- the same guard, whatever the spelling of the folders (final review, Task 11 deferred) --------------------------------------------
# The textual test above cannot fail when ``.resolve()`` is taken out of the guard: ``s1 / '..' / s1.name`` is inside by its first parts.
# These tests give the S1 run and the run folder in spellings that only a resolved path (a relative name, a junction) or the identity of
# the folders (the extended-length prefix and the administrative share of a drive, which ``Path.resolve`` leaves as they are) can see
# through. Every junction is made under ``tmp_path`` and removed with ``os.rmdir`` (which removes the link and not its target) before the
# test ends.
windows_only = pytest.mark.skipif(sys.platform != 'win32', reason='junctions, the \\\\?\\ prefix and administrative shares are Windows')


def make_junction(link: Path, target: Path) -> None:
    """A directory junction ``link`` that points at ``target`` (``mklink /J`` needs no privilege); the test removes it with ``os.rmdir``."""
    done = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip(f'no junction can be made here: {done.stdout.strip()[:100]}')


def extended(path: Path) -> Path:
    r"""``path`` with the extended-length prefix: ``C:\a`` as ``\\?\C:\a``."""
    return Path('\\\\?\\' + str(path))


def admin_share(path: Path, host: str) -> Path:
    r"""``path`` on the administrative share of its drive on ``host``: ``C:\a`` as ``\\localhost\C$\a``."""
    drive, rest = os.path.splitdrive(str(path))
    return Path(f'\\\\{host}\\{drive[0]}${rest}')


def assert_refused(s1_given, runs) -> None:
    for run in runs:
        with pytest.raises(ValueError, match='S1 run'):
            terrain.build_terrain(s1_given, run)


def test_build_terrain_reads_a_relative_run_folder_against_the_working_directory(tmp_path, monkeypatch):
    s1 = write_mini_s1(tmp_path / 's1_run' / '20261020_c_ept_s1_9', mini_matches())
    store, before = s1 / run_s1.STORE_DIR, tree(tmp_path)
    monkeypatch.chdir(store)                                                   # a relative name from here lies inside the S1 run
    for given in (s1, store):
        assert_refused(given, [Path('later'), Path('.'), Path('..'), Path('..') / 'later', Path('a') / '..' / 'b'])
    monkeypatch.chdir(tmp_path)                                               # the S1 run and the run folder both relative
    rel = Path('s1_run') / s1.name
    for given in (rel, rel / run_s1.STORE_DIR):
        assert_refused(given, [rel, rel / 'later', rel / run_s1.STORE_DIR / 'matches', Path('s1_run') / '..' / 's1_run' / s1.name / 'x'])
    assert tree(tmp_path) == before
    monkeypatch.chdir(s1)                                                      # the S1 run as "." and a relative folder beside it
    out = terrain.build_terrain(Path('.'), Path('..') / 'beside')              # ... is not refused, and the S1 run keeps its own name
    try:
        assert out.resolve() == (tmp_path / 's1_run' / 'beside' / 'terrain').resolve()
        assert read_terrain_json(out)['s1_run'] == s1.name != Path('.').name
    finally:
        shutil.rmtree(tmp_path / 's1_run' / 'beside', ignore_errors=True)


@windows_only
def test_build_terrain_follows_a_junction_into_the_s1_run(tmp_path):
    s1 = write_mini_s1(tmp_path / 's1_run' / '20261020_c_ept_s1_9', mini_matches())
    store, elsewhere = s1 / run_s1.STORE_DIR, tmp_path / 'elsewhere'
    elsewhere.mkdir()
    into_store, into_s1, away = tmp_path / 'into_store', tmp_path / 'into_s1', tmp_path / 'away'
    made = []
    try:
        for link, target in ((into_store, store), (into_s1, s1), (away, elsewhere)):
            make_junction(link, target)
            made.append(link)
        before = tree(tmp_path)
        for given in (s1, store):
            assert_refused(given, [into_store, into_store / 'later', into_store / 'a' / 'b', into_s1, into_s1 / 'later'])
        assert tree(tmp_path) == before
        out = terrain.build_terrain(s1, away / 'run')                          # a junction that leads away from the S1 run is not refused
        assert out.resolve() == (elsewhere / 'run' / 'terrain').resolve()
    finally:
        for link in made:
            os.rmdir(link)
        shutil.rmtree(elsewhere, ignore_errors=True)


@windows_only
def test_build_terrain_sees_through_the_extended_length_prefix(tmp_path):
    r"""``Path.resolve`` keeps the ``\\?\`` prefix, so the resolved path is not "inside" the S1 run by its text: the folders are compared
    by identity as well."""
    s1 = write_mini_s1(tmp_path / 's1_run' / '20261020_c_ept_s1_9', mini_matches())
    store, before = s1 / run_s1.STORE_DIR, tree(tmp_path)
    inside = [extended(s1), extended(s1 / 'later'), extended(s1 / 'a' / 'b'), extended(store / 'matches'), s1 / 'later', store]
    for given in (s1, extended(s1), store, extended(store)):
        assert_refused(given, inside)
    assert tree(tmp_path) == before
    beside = s1.with_name(s1.name + '_other')
    out = terrain.build_terrain(extended(s1), extended(beside))                # beside the S1 run, in the same spelling: not refused
    try:
        assert os.path.samefile(out, beside / 'terrain')                      # the same folder, whatever its spelling
    finally:
        shutil.rmtree(beside, ignore_errors=True)


@windows_only
@pytest.mark.parametrize('host', ['localhost', '127.0.0.1'])
def test_build_terrain_sees_through_a_loopback_administrative_share(tmp_path, host):
    s1 = write_mini_s1(tmp_path / 's1_run' / '20261020_c_ept_s1_9', mini_matches())
    if not admin_share(s1, host).is_dir():
        pytest.skip(f'the administrative share of this drive is not reachable as {host}')
    store, before = s1 / run_s1.STORE_DIR, tree(tmp_path)
    inside = [admin_share(s1, host), admin_share(s1 / 'later', host), admin_share(store / 'matches', host), s1 / 'later']
    for given in (s1, admin_share(s1, host), store, admin_share(store, host)):
        assert_refused(given, inside)
    assert tree(tmp_path) == before
    beside = s1.with_name(s1.name + '_other')
    out = terrain.build_terrain(admin_share(s1, host), admin_share(beside, host))      # beside the S1 run, same spelling: not refused
    try:
        assert os.path.samefile(out, beside / 'terrain')                      # the same folder, whatever its spelling
    finally:
        shutil.rmtree(beside, ignore_errors=True)


# ---- Terrain(path): what it loads and refuses ------------------------------------------------------------------------------------------
def test_terrain_opens_the_table_as_a_read_only_memmap(open_terrain):
    g = open_terrain.geodesic
    assert isinstance(g, np.memmap) and g.mode == 'r' and not g.flags.writeable and g.dtype == np.uint16
    assert open_terrain.walk.dtype == bool and open_terrain.walk.shape == (N, N) and open_terrain.main.all()
    assert open_terrain.manifest is None
    with pytest.raises(FileNotFoundError):
        open_terrain.verify()                                                               # nothing to check against without TERRAIN.json


def test_terrain_refuses_a_table_that_does_not_belong_to_the_grid(open_table, tmp_path):
    walk = OPEN.copy(); walk[3, 3] = False                                         # the table says "walkable" for this cell
    terrain.write_terrain_files(tmp_path, walk, ZERO100, ZERO100, open_table)
    with pytest.raises(ValueError, match='geodesic'):
        terrain.Terrain(tmp_path)


def test_terrain_refuses_a_wrong_table_shape_or_dtype(tmp_path):
    with pytest.raises(TypeError):
        terrain.write_terrain_files(tmp_path, OPEN, ZERO100, ZERO100, np.zeros((5625, 5625), np.int32))
    with pytest.raises(ValueError):
        terrain.write_terrain_files(tmp_path, OPEN, ZERO100, ZERO100, np.zeros((5624, 5625), np.uint16))
    np.savez(tmp_path / 'terrain.npz', walk=OPEN, frames100=ZERO100, kills100=ZERO100)     # a folder written by hand: the loader checks too
    np.save(tmp_path / 'geodesic.npy', np.zeros((10, 10), np.uint16))
    with pytest.raises(ValueError, match='geodesic'):
        terrain.Terrain(tmp_path)


def test_terrain_pickles_as_its_path_not_as_the_table(open_terrain):
    blob = pickle.dumps(open_terrain)
    assert len(blob) < 10_000                                                       # the 63 MB table is not copied into the message
    again = pickle.loads(blob)
    try:
        args = (np.array([1_234.0]), np.array([4_321.0]), np.array([9_000.0]), np.array([2_000.0]))
        assert np.array_equal(again.geo(*args), open_terrain.geo(*args)) and again.geodesic.mode == 'r'
    finally:
        del again
        gc.collect()


def test_terrain_verify_detects_a_changed_file(tmp_path):
    run, out = build(tmp_path)
    try:
        t = terrain.Terrain(out)
        try:
            assert t.verify() == t.manifest['sha256']
            data = bytearray((out / 'terrain.npz').read_bytes())
            data[-30] ^= 1                                                          # one bit in the last array
            (out / 'terrain.npz').write_bytes(bytes(data))
            with pytest.raises(ValueError, match=r'terrain\.npz'):
                t.verify()
        finally:
            del t
            gc.collect()
    finally:
        shutil.rmtree(run, ignore_errors=True)


# ---- what the module reaches (R10-h, R11-e, S2-13) ---------------------------------------------------------------------------------------
def test_no_module_level_import_reaches_the_s1_runner_or_store():
    assert not module_level_imports(Path(terrain.__file__)) & set(S1_MODULES)


@pytest.mark.parametrize('module', ['c_ept.terrain', 'c_ept.preprocessing.terrain'])
def test_importing_terrain_loads_no_s1_module(module):
    """A fresh interpreter: a worker that loads the terrain never pulls in the S1 runner chain (the runner is imported inside
    ``build_terrain`` only)."""
    code = PROTECTED_IMPORT_PRELUDE + f'import sys, {module}; sys.stdout.write(",".join(m for m in {S1_MODULES!r} if m in sys.modules))'
    r = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, encoding='utf-8')
    assert (r.returncode, r.stdout, r.stderr) == (0, '', '')


# ---- the real S1 run (C_EPT_S1_RUN) ---------------------------------------------------------------------------------------------------
def test_occupancy_equals_probe(s1_run):                     # C_EPT_S1_RUN needed
    f, k = terrain.occupancy_100(load_s1(s1_run), fit_rows(s1_run)); z = np.load(PROTO / 'terrain_occupancy.npz')
    assert np.array_equal(f, z['frames']) and np.array_equal(k, z['kills'])


def test_real_occupancy_totals_and_the_walkable_grid(s1_run, real_scan):
    """Measured 2026-10-21 on the S1 run 20261020_c_ept_s1_1: the 19,100 FIT matches hold the prototype's 5,244,590 frame points
    (5,053,590 in regular frames and 191,000 in game-end frames) and 991,410 kill points. At MIN_COUNT 150 (S2 gate S2G-2) 3,770 of the
    5,625 cells are walkable and they form one component, so the islands of R11-b do not occur in the real grid (the synthetic tests above
    cover them); 34,162 of the 6,236,000 hits lie in cells that stay walls. (Task 11 measured the first value, MIN_COUNT 5: 4,532 walkable
    cells, one component, 434 hits in walls.)"""
    f, k, _ = real_scan
    assert len(fit_rows(s1_run)) == 19_100
    assert (int(f.sum()), int(k.sum())) == (5_244_590, 991_410)
    walk = terrain.walkable(f, k)
    comps = terrain.components(walk)
    assert int(walk.sum()) == 3_770
    assert (len(comps.sizes), int(comps.main.sum()), int(walk.sum() - comps.main.sum())) == (1, 3_770, 0)
    hits = (f + k).reshape(75, 2, 75, 2).sum(axis=(1, 3)).T
    assert int(hits[~walk].sum()) == 34_162 and int(hits.sum()) == 6_236_000


def test_real_geodesic_table_joins_every_walkable_pair_and_stays_below_the_open_grid_maximum(real_scan):
    """Measured 2026-10-21 at MIN_COUNT 150 (S2 gate S2G-2; the longest was 20,482 u at the first value 5): every pair of walkable cells has
    a path, and the longest is 20,868 u, below the open grid's 20,930 u, so the 16 u float16 step that ``tokens`` states holds."""
    f, k, _ = real_scan
    walk = terrain.walkable(f, k)
    table = np.asarray(terrain.geodesic_table(walk))
    cells = np.flatnonzero(walk.ravel())
    pairs = table[np.ix_(cells, cells)]                                  # walkable cell x walkable cell
    assert not (pairs == terrain.UNREACHABLE).any()
    assert int(pairs.max()) == 20_868 < round(74 * DIAG)


def test_real_position_census(real_scan):
    """R11-d on the same run: S2-9 found one player-frame at exactly (0, 0); it is in the FIT matches, and nothing else is suspect."""
    _, _, census = real_scan
    quiet = dict(nan=0, zero_zero=0, outside_map=0, not_binned=0)
    assert census == {'regular_frames': dict(n=5_053_590, **{**quiet, 'zero_zero': 1}), 'game_end_frame': dict(n=191_000, **quiet),
                      'kills': dict(n=991_410, **quiet)}
