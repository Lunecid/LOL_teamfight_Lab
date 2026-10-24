"""Toy S2 stores for the tests of the EPT data layer (stage S4a Task 1), built on the heads world of ``s3_toys`` (ruling R1).

The heads world (``s3_toys.build_heads_world``) is a small random S1, S2 and label store with the real links between the tables: eight
matches (four FIT, one STOP, two DEV, one CAL), their frame rows, onsets, units, event rows and views. Its S2 holds only the three
arrays that the baselines read. The EPT reads all of them, so ``build_ept_world`` writes a second S2 store next to it that has the same
``views/*`` arrays (the same view numbers, so every row of ``head_rows`` finds its view) and every ``x/*`` array of ``tokens.VIEW_SPEC``,
with made-up values that obey the rules of the real arrays:

    positions    whole numbers on [0, 14,875], NaN only where the module text of ``tokens`` allows it (a player without a position
                 also has NaN in ``wall_fraction``, zone -1 and NaN rows in the geodesics; ``prev_x`` / ``prev_y`` are NaN when
                 ``has_prev`` is 0; the 18 trait columns are NaN when ``champ_unknown`` is 1; items, levels and wards have no position)
    padding      the rows of ``ev_*`` and ``ed_*`` after the last token carry the padding values (-1, team 2, NaN position, 0)
    constants    ``CONSTANT_PLAYER_COLUMN`` is 0 in every view, ``CONSTANT_SUMMARY`` 7,777 in every view, ``FIT_CONSTANT_SUMMARY`` 7 in
                 the views of FIT matches and 5 in the others (so it is constant over the FIT views only)
    shapes       ``EPT_EVENTS`` = ``EPT_EDGES`` = 128 event tokens and edges per view, the shapes of the real store (``symmetry.swap_view``
                 checks them); ``build_ept_world(n_events=7, n_edges=5)`` makes a world with other numbers, so that a loader which
                 hard-codes 128 instead of reading the shapes from the manifest fails

``shift`` adds a number to the numeric values of every view of a match that is not FIT (players, summary, landmarks, geodesics, and the
age, gold and position of the event tokens), so that a test can show that a statistic does not read them. ``world_view`` /
``write_views`` read and write views by name, for the comparison with ``symmetry.swap_view``.

``encoder_batch`` (stage S4a Task 2, the encoder) needs no store: it draws a random batch of view tensors with the keys, shapes, dtypes
and padding rules of ``data.ViewTensors`` for any number of views, event tokens and edges, so that the encoder tests can use sizes of
their own (a second batch of ``tests/test_ept_encoder.py`` checks the layout against the tensors of the 7 / 5 world).
``toy_static`` (Task 2, fix round 1) is the made-up terrain that the encoder tests and the tests of the heads give ``EPTEncoder`` as its
static channels, which the encoder requires: the 12 channels of ``encoder.static_channels`` (walkable, ten zones, wall fraction) of a grid
with a block of walls, or of an open grid, made with the same rules as the real helper (a test compares the open grid with it).
"""
# Ref: docs/superpowers/plans/2026-10-22-c-ept-s4a.md Task 1 (the tests of the data layer use the toy worlds of s3_toys) and Task 2 (the
#      encoder tests use ``encoder_batch`` and ``toy_static``).
from __future__ import annotations

import functools
import gc
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from c_ept import config as C
from c_ept import features, state, terrain, tokens, traits
from c_ept.store_core import Store
from c_ept.tests.s3_toys import WORLD_ROLES, WORLD_SEED, HeadsWorld, _put, build_heads_world

EPT_EVENTS, EPT_EDGES = 128, 128
EPT_SEED = 20261022
CONSTANT_PLAYER_COLUMN = 'flag_stasis'              # 0 in every view
CONSTANT_SUMMARY = 'rift_element_known'             # 7,777 in every view
FIT_CONSTANT_SUMMARY = 'rift_element_air'           # 7 in the FIT views, 5 in the others
PCOL = {name: i for i, name in enumerate(tokens.PLAYER_NUM)}
SCOL = {name: i for i, name in enumerate(tokens.SUMMARY)}
DTYPES = {name: dtype for name, (_, dtype) in tokens.VIEW_SPEC.items()}
NO_POSITION_CELL = (2, 3)                           # (view, player) whose position is missing in every world (a fixed case)
NO_PREV_CELL = (4, 6)
UNKNOWN_CHAMPION_CELL = (5, 1)


@dataclass(frozen=True, eq=False)
class EptWorld:
    """The heads world (``base``) with the EPT S2 store ``s2`` (same views) and ``fit_views`` (the view numbers of the FIT matches)."""
    base: HeadsWorld
    s2: Store
    fit_views: np.ndarray

    @property
    def s1(self):
        return self.base.s1

    @property
    def labels(self):
        return self.base.labels


def _positions(rng, shape) -> np.ndarray:
    return rng.integers(0, int(C.W) + 1, shape).astype(np.float64)


def _players(rng, n: int) -> np.ndarray:
    """(n, 10, 81) float64 players with the structure of ``tokens``: whole-number positions, NaN only where it is allowed."""
    players = rng.normal(5.0, 3.0, (n, 10, len(tokens.PLAYER_NUM)))
    xy, prev = _positions(rng, (n, 10, 2)), _positions(rng, (n, 10, 2))
    players[:, :, [PCOL['x'], PCOL['y']]] = xy
    players[:, :, [PCOL['prev_x'], PCOL['prev_y']]] = prev
    for name in ('has_prev', 'alive', 'champ_unknown'):
        players[:, :, PCOL[name]] = rng.integers(0, 2, (n, 10))
    players[:, :, PCOL['wall_fraction']] = rng.uniform(0.0, 1.0, (n, 10))
    players[:, :, PCOL[CONSTANT_PLAYER_COLUMN]] = 0.0
    missing = rng.random((n, 10)) < 0.08
    missing[NO_POSITION_CELL] = True
    players[:, :, PCOL['has_prev']][NO_PREV_CELL] = 0.0
    players[:, :, PCOL['champ_unknown']][UNKNOWN_CHAMPION_CELL] = 1.0
    players[:, :, PCOL['champ_unknown']][NO_POSITION_CELL] = 0.0
    no_prev = players[:, :, PCOL['has_prev']] == 0.0
    unknown = players[:, :, PCOL['champ_unknown']] == 1.0
    for name in ('x', 'y', 'wall_fraction'):
        players[missing, PCOL[name]] = np.nan
    for name in ('prev_x', 'prev_y'):
        players[no_prev, PCOL[name]] = np.nan
    for name in traits.TRAIT_NAMES:
        if name != 'champ_unknown':
            players[unknown, PCOL[name]] = np.nan
    return players


def random_x(rng, kind: np.ndarray, fit: np.ndarray, n_events: int = EPT_EVENTS, n_edges: int = EPT_EDGES) -> dict:
    """The ``x/*`` arrays (in the dtypes of ``tokens.VIEW_SPEC``, a dict of numpy arrays with one row per view) of ``len(kind)`` made-up
    views; ``kind`` is ``views/src`` and ``fit`` says which views belong to a FIT match. See the module text for the rules."""
    n = len(kind)
    players = _players(rng, n)
    x, y = players[:, :, PCOL['x']], players[:, :, PCOL['y']]
    zone = terrain.zone_of(x, y)
    # geodesics: Euclidean stand-ins; the row and the column of a player without a position are NaN
    dx, dy = x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :]
    geo_pp = np.hypot(dx, dy)
    lm = tokens.LANDMARK_XY
    geo_pl = np.hypot(x[:, :, None] - lm[None, None, :, 0], y[:, :, None] - lm[None, None, :, 1])
    # summary: small whole numbers, the plan-mandated twins, the three allowed NaN columns
    summary = rng.integers(0, 6, (n, len(tokens.SUMMARY))).astype(np.float64)
    game_time = rng.uniform(120.0, 2400.0, n).round(1)
    summary[:, SCOL['game_time_s']] = summary[:, SCOL['t_s']] = game_time
    summary[:, SCOL['frame_age_s']] = np.where(kind == 0, 0.0, rng.uniform(0.5, 59.5, n).round(1))
    summary[:, SCOL[CONSTANT_SUMMARY]] = 7777.0
    summary[:, SCOL[FIT_CONSTANT_SUMMARY]] = np.where(fit, 7.0, 5.0)
    for name in state.NAN_FEATURES:
        summary[rng.random(n) < 0.2, SCOL[name]] = np.nan
    summary[1, SCOL[state.NAN_FEATURES[0]]] = np.nan
    # landmarks: (standing, seconds left); the seconds cell of the herald, grubs and atakhan rows may be NaN
    landmarks = np.stack([rng.integers(0, 2, (n, len(state.LANDMARKS))).astype(np.float64),
                          rng.uniform(0.0, 300.0, (n, len(state.LANDMARKS))).round(1)], axis=-1)
    nan_rows = [i for i, lm_ in enumerate(state.LANDMARKS) if lm_['kind'] in ('herald', 'grubs', 'atakhan')]
    gone = rng.random((n, len(nan_rows))) < 0.3
    gone[1, 0] = True
    landmarks[:, nan_rows, 1] = np.where(gone, np.nan, landmarks[:, nan_rows, 1])
    # recent events, newest first (age grows), padding after the last one
    n_ev = rng.integers(0, n_events + 1, n)
    n_ev[0], n_ev[1] = 0, n_events                                      # a view without a token and a full one
    valid = np.arange(n_events)[None, :] < n_ev[:, None]
    ev_type = np.where(valid, rng.integers(0, len(C.EVENT_TYPES), (n, n_events)), -1)
    has_pos = valid & (ev_type < 4)                                      # kills, objectives, buildings and plates have a position
    ev_xy = np.where(has_pos[..., None], _positions(rng, (n, n_events, 2)), np.nan)
    ev = {
        'ev_type': ev_type, 'ev_team': np.where(valid, rng.integers(0, 3, (n, n_events)), 2),
        'ev_dt': np.where(valid, np.sort(rng.uniform(0.0, 120.0, (n, n_events)).round(1), axis=1), 0.0),
        'ev_xy': ev_xy, 'ev_actors': np.where(valid, rng.integers(0, 1024, (n, n_events)), 0),
        'ev_gold': np.where(valid, rng.integers(0, 4, (n, n_events)) * 150.0, 0.0),
        'ev_n': n_ev, 'ev_dropped': np.zeros(n, dtype=np.int64)}
    n_ed = rng.integers(0, n_edges + 1, n)
    n_ed[0], n_ed[1] = 0, n_edges
    valid_ed = np.arange(n_edges)[None, :] < n_ed[:, None]
    first = rng.integers(0, 10, (n, n_edges))
    ed = {
        'ed_i': np.where(valid_ed, first, -1), 'ed_j': np.where(valid_ed, (first + rng.integers(1, 10, (n, n_edges))) % 10, -1),
        'ed_dt': np.where(valid_ed, np.sort(rng.uniform(0.0, 120.0, (n, n_edges)).round(1), axis=1), 0.0),
        'ed_kind': np.where(valid_ed, rng.integers(0, len(tokens.EDGE_KINDS), (n, n_edges)), -1),
        'ed_n': n_ed, 'ed_dropped': np.zeros(n, dtype=np.int64)}
    arrays = {'players_num': players, 'players_zone': zone, 'landmarks': landmarks, **ev, **ed, 'summary': summary,
              'geo_pp': geo_pp, 'geo_pl': geo_pl}
    return {name: np.asarray(arrays[name]).astype(DTYPES[name]) for name in tokens.VIEW_SPEC}


def shift_numbers(arrays: dict, rows: np.ndarray, shift: float) -> None:
    """Add ``shift`` in place to the numeric values (not the NaN, not the padding) of the views ``rows`` of ``arrays``."""
    if not shift:
        return
    for name in ('players_num', 'summary', 'landmarks', 'geo_pp', 'geo_pl'):
        arrays[name][rows] += np.asarray(shift, dtype=arrays[name].dtype)
    valid = arrays['ev_type'][rows] >= 0
    for name in ('ev_dt', 'ev_gold'):
        arrays[name][rows] = np.where(valid, arrays[name][rows] + np.float32(shift), arrays[name][rows])
    arrays['ev_xy'][rows] += np.float32(shift)       # NaN (no position, padding) stays NaN


def write_views(root, arrays: dict, src) -> Store:
    """A finalised store under ``root`` that holds the ``x/*`` arrays of ``arrays`` (one row per view) and ``views/src``."""
    s = Store(root)
    _put(s, 'views/src', src, 'i1')
    for name, a in arrays.items():
        _put(s, f'x/{name}', a, DTYPES[name])
    s.finalize()
    return s


def build_ept_world(root, seed: int = EPT_SEED, roles=WORLD_ROLES, shift: float = 0.0, base_seed: int = WORLD_SEED,
                    n_events: int = EPT_EVENTS, n_edges: int = EPT_EDGES) -> EptWorld:
    """The heads world under ``root/base`` and, under ``root/s2``, the EPT S2 store of its views (see the module text)."""
    root = Path(root)
    base = build_heads_world(root / 'base', base_seed, roles)
    src = np.asarray(base.s2.array('views/src'))
    fit = features.split_views(base.s1, base.s2, 'FIT')
    arrays = random_x(np.random.default_rng(seed), src, fit, n_events, n_edges)
    shift_numbers(arrays, np.flatnonzero(~fit), shift)
    s = Store(root / 's2')
    for name in ('src', 'src_index', 'match', 'ref', 't0', 'frame_k'):
        a = np.asarray(base.s2.array(f'views/{name}'))
        _put(s, f'views/{name}', a, a.dtype)
    for name in tokens.VIEW_SPEC:
        _put(s, f'x/{name}', arrays[name], DTYPES[name])
    s.finalize()
    return EptWorld(base=base, s2=s, fit_views=np.flatnonzero(fit))


def world_view(store, v: int) -> dict:
    """The view ``v`` of an EPT S2 store as a dict of new numpy arrays in the layout of ``tokens.build_view`` (what ``symmetry`` swaps)."""
    return {name: np.array(store.array(f'x/{name}')[v]) for name in tokens.VIEW_SPEC}


@pytest.fixture(scope='module')
def ept_world(tmp_path_factory):
    """The EPT toy world (``build_ept_world``), written once per test module and read only."""
    folder = tmp_path_factory.mktemp('ept_world')
    world = build_ept_world(folder)
    yield world
    del world
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


@functools.lru_cache(maxsize=None)
def toy_static(walls: bool = True) -> np.ndarray:
    """The static channels of a made-up terrain, (75, 75, 12) float32 [iy, ix], read only: walkable, the ten zones one-hot and the wall
    fraction of the 5 x 5 window, in the order of ``encoder.STATIC_CHANNELS`` and with the rules of ``encoder.static_channels`` (the
    zones from ``terrain.zone_of`` at the cell centres, cells outside the grid count as walls). With ``walls`` a block of 15 x 15 cells in
    the middle of the map (ix 30-44, iy 20-34) cannot be walked; without it every cell is walkable, which is the open grid."""
    grid = terrain.GRID
    centre = terrain.CELL * (np.arange(grid) + 0.5)
    x, y = np.meshgrid(centre, centre)
    walk = np.ones((grid, grid), dtype=bool)
    if walls:
        walk[20:35, 30:45] = False
    one_hot = terrain.zone_of(x, y)[..., None] == np.arange(len(terrain.ZONES))
    padded = np.pad(walk, 2, constant_values=False)
    open_cells = sum(padded[dy:dy + grid, dx:dx + grid].astype(np.int64) for dy in range(5) for dx in range(5))
    out = np.concatenate([walk[..., None], one_hot, ((25 - open_cells) / 25.0)[..., None]], axis=-1).astype(np.float32)
    out.setflags(write=False)
    return out


def encoder_batch(views: int = 3, events: int = 9, edges: int = 7, seed: int = 0, dtype=torch.float32) -> dict:
    """A random batch of ``views`` views with ``events`` event tokens and ``edges`` edges each: the keys (``data.VIEW_KEYS``), shapes, dtypes
    and padding rules of ``data.ViewTensors``, but with made-up numbers and no store. ``dtype`` is that of the float tensors.

    The rules that it keeps: positions are whole numbers on [0, W] and W/2 where missing, a cell that may be NaN is 0.0 with its mask
    channel 1.0, the padding rows of the events and the edges are zero in every standardised and mask channel (type -1, team 2 and edge
    numbers -1), the edges join two different players, the ages grow along a row, and the geodesic tables are 1.2 times the Euclidean
    distance, standardised by fixed numbers, with a NaN row and column (mask 1.0, value 0.0) for the player without a position. Fixed
    cases: view 0 has no event token and no edge, view 1 has all of them (when there are two views), and the player 3 of view 0 has no
    position. The numbers inside one view do not belong together (the standardised position columns of ``players`` are drawn on their own).
    """
    from c_ept.ept import data
    rng = np.random.default_rng(seed)
    n, p, n_lm = views, 10, len(state.LANDMARKS)

    def real(a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype)

    def whole(a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, dtype=np.int64))

    # players: the mask channels first, then the zeroed cells, the zones and the geodesics of the positions
    xy = rng.integers(0, int(C.W) + 1, (n, p, 2)).astype(np.float64)
    missing = np.zeros((n, p), dtype=bool)
    missing[0, 3] = True
    xy[missing] = C.W / 2.0
    nan_at = np.array([data.PLAYER_COLUMNS.index(name) for name in data.PLAYER_NAN_COLUMNS])
    players_nan = rng.random((n, p, len(nan_at))) < 0.1
    for name in ('x', 'y'):
        players_nan[..., data.PLAYER_NAN_COLUMNS.index(name)] = missing
    players = rng.normal(size=(n, p, len(data.PLAYER_COLUMNS)))
    players[:, :, nan_at] = np.where(players_nan, 0.0, players[:, :, nan_at])
    zone = np.where(missing, -1, terrain.zone_of(xy[..., 0], xy[..., 1]))
    lm_xy = tokens.LANDMARK_XY
    pp = 1.2 * np.hypot(xy[:, :, None, 0] - xy[:, None, :, 0], xy[:, :, None, 1] - xy[:, None, :, 1])
    pl = 1.2 * np.hypot(xy[:, :, None, 0] - lm_xy[None, None, :, 0], xy[:, :, None, 1] - lm_xy[None, None, :, 1])
    pp_nan = missing[:, :, None] | missing[:, None, :]
    pl_nan = np.broadcast_to(missing[:, :, None], (n, p, n_lm))
    # landmarks and summary
    landmarks = rng.normal(size=(n, n_lm, 2))
    lm_nan = np.zeros((n, n_lm), dtype=bool)
    for i, lm in enumerate(state.LANDMARKS):
        if lm['kind'] in ('herald', 'grubs', 'atakhan'):
            lm_nan[:, i] = rng.random(n) < 0.4
    landmarks[:, :, 1] = np.where(lm_nan, 0.0, landmarks[:, :, 1])
    summary = rng.normal(size=(n, len(data.SUMMARY_COLUMNS)))
    summary_nan = rng.random((n, len(data.SUMMARY_NAN_COLUMNS))) < 0.3
    summary_at = [data.SUMMARY_COLUMNS.index(name) for name in data.SUMMARY_NAN_COLUMNS]
    summary[:, summary_at] = np.where(summary_nan, 0.0, summary[:, summary_at])
    # events, newest first: the ages grow along a row; kills, objectives, buildings and plates (type < 4) have a position
    n_ev = rng.integers(0, events + 1, n)
    n_ev[:2] = (0, events)[:min(n, 2)]
    valid = np.arange(events)[None, :] < n_ev[:, None]
    ev_type = np.where(valid, rng.integers(0, len(C.EVENT_TYPES), (n, events)), -1)
    located = valid & (ev_type < 4)
    ev_xy = np.where(located[..., None], rng.integers(0, int(C.W) + 1, (n, events, 2)), C.W / 2.0)
    ev_num = np.concatenate([np.where(valid, np.sort(rng.normal(size=(n, events)), axis=1), 0.0)[..., None],
                             np.where(valid, rng.normal(size=(n, events)), 0.0)[..., None],
                             np.where(located[..., None], (ev_xy - C.W / 2.0) / 4000.0, 0.0)], axis=-1)
    actors = np.where(valid[..., None], rng.random((n, events, p)) < 0.25, False)
    # edges between two different players, newest first
    n_ed = rng.integers(0, edges + 1, n)
    n_ed[:2] = (0, edges)[:min(n, 2)]
    valid_ed = np.arange(edges)[None, :] < n_ed[:, None]
    first = rng.integers(0, p, (n, edges))
    second = (first + rng.integers(1, p, (n, edges))) % p
    batch = {
        'view': whole(np.arange(n)), 'kind': whole(rng.integers(0, 3, n)), 'side': real(rng.integers(0, 2, n)), 'age': real(rng.normal(size=n)),
        'players': real(players), 'players_nan': real(players_nan), 'players_zone': whole(zone), 'players_xy': real(xy),
        'landmarks': real(landmarks), 'landmarks_nan': real(lm_nan[..., None]), 'summary': real(summary), 'summary_nan': real(summary_nan),
        'ev_valid': torch.as_tensor(valid), 'ev_type': whole(ev_type), 'ev_team': whole(np.where(valid, rng.integers(0, 3, (n, events)), 2)),
        'ev_num': real(ev_num), 'ev_xy': real(ev_xy), 'ev_nan': real((valid & ~located)[..., None]), 'ev_actors': real(actors),
        'ed_valid': torch.as_tensor(valid_ed), 'ed_i': whole(np.where(valid_ed, first, -1)), 'ed_j': whole(np.where(valid_ed, second, -1)),
        'ed_kind': whole(np.where(valid_ed, rng.integers(0, len(tokens.EDGE_KINDS), (n, edges)), -1)),
        'ed_dt': real(np.where(valid_ed, np.sort(rng.uniform(0.0, 120.0, (n, edges)), axis=1), 0.0)),
        'geo_pp': real(np.where(pp_nan, 0.0, (pp - 7000.0) / 4500.0)), 'geo_pp_nan': real(pp_nan),
        'geo_pl': real(np.where(pl_nan, 0.0, (pl - 7000.0) / 4500.0)), 'geo_pl_nan': real(pl_nan),
    }
    if set(batch) != set(data.VIEW_KEYS):
        raise RuntimeError('encoder_batch does not make exactly the keys of data.VIEW_KEYS')
    return batch
