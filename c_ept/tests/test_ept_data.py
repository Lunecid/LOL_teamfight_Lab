"""Tests of the data layer of the EPT (stage S4a Task 1): ``head_rows.HeadRows.view_numbers``, ``ept.data`` (FIT statistics, view tensors,
the team swap, the proxy selection and the batches).

No run folder, raw file or catalog is read. The data are the "EPT world" of ``ept_toys``: the heads world of ``s3_toys`` (small S1, S2 and
label stores with the real links between the tables) with a second S2 store that holds every array of a view (128 event tokens and 128
edges as in the real store; a second world with 7 and 5 shows that the shapes are read from the manifest and not written into the code).

    view numbers   the S2 view that row j of ``parts(head, kind)`` reads, for all 13 heads, against a feature builder that returns the
                   view number and against the plain-loop answers of ``s3_toys.HeadsOracle``
    statistics     FIT views only (a shift of the other roles changes nothing), NaN left out of the count, a constant column has scale 1,
                   the rotation centre for the positions, one pair of statistics for two columns that the swap exchanges, JSON and hash,
                   no role but FIT
    view tensors   the standardisation formula, NaN -> 0 with a mask channel, finite outputs, refusals for a NaN or an infinity where the
                   data may not have one and for a position with one NaN coordinate, shapes read from the manifest
    swap           an involution on views and targets, equal to ``symmetry.swap_view`` on views in every bit (the sign of a zero included),
                   targets that move with the inputs, the zone permutation derived from the zone rules, the mark permutation derived from
                   the players
    proxy          a seeded, nested selection of matches of FIT and STOP
    batches        one engagement in one batch, every row once per epoch, the rows of ``head_rows`` (K and O of the masked M4 matrix found
                   by name), FIT rows only for training, no default epoch
    view swap      (S4b preparation M7) the coins of ``swap_mode='random_view'`` (plan Q3: one fair coin for each view from SeedSequence([seed, epoch,
                   batch index, 0x5157]), no global random state), the plain batches of the other modes, ``swap_views`` (the rows follow the frame of their view)
"""
import gc
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from c_ept import config as C
from c_ept.preprocessing import features
from c_ept.data import head_rows
from c_ept import runlog
from c_ept.data.store_core import Store
from c_ept.domain import symmetry
from c_ept.preprocessing import terrain, tokens
from c_ept.domain import units
from c_ept.ept import data, encoder
from c_ept.tests.ept_toys import (CONSTANT_PLAYER_COLUMN, CONSTANT_SUMMARY, DTYPES, EPT_EDGES, EPT_EVENTS, FIT_CONSTANT_SUMMARY, NO_POSITION_CELL,
                                  PCOL, SCOL, build_ept_world, consistent_ept_world, ept_world, toy_static, world_view, write_views)  # noqa: F401  (the fixtures: ruling R1)
from c_ept.tests.s3_toys import KEY_PREFIX, HeadsOracle, _put

ALL_HEADS = [(name, 'masked') for name in head_rows.HEAD_NAMES] + [(name, 'open') for name in head_rows.OPEN_HEAD_NAMES]


def number_rows(world, role='FIT'):
    """``HeadRows`` whose feature builder returns the view number as the one feature column: the rows tell which view they read."""
    return head_rows.HeadRows(world.s1, world.s2, world.labels, role, lambda s2, views: np.asarray(views, dtype=np.float32)[:, None])


@pytest.fixture(scope='module')
def oracle(ept_world):
    return HeadsOracle(ept_world.base)                 # the views/* arrays of the EPT S2 are those of the heads world


# =====================================================================================================================================
# HeadRows.view_numbers
# =====================================================================================================================================
def test_view_numbers_follow_parts_order(ept_world, oracle):
    """Row j of ``parts(head, kind)`` reads the view ``view_numbers(head, kind)[j]``: against the builder, which hands the view number
    back as a feature, and against the plain-loop answers of the oracle (frame-clock view of the frame row or of the onset's row for a
    masked head, the src 1 view of the unit for an open M3 head, the src 2 view for the open M4 head)."""
    rows = number_rows(ept_world)
    src = np.asarray(ept_world.s2.array('views/src'))
    for head, kind in ALL_HEADS:
        p, v = rows.parts(head, kind), rows.view_numbers(head, kind)
        assert v.dtype == np.int64 and v.shape == p.y.shape[:1], (head, kind)
        assert np.array_equal(p.gather()[:, 0].astype(np.int64), v), (head, kind)                  # the builder's own answer
        if kind == 'masked':
            expected = ([oracle.view(0, r) for r in p.row] if head == 'M2Rate'
                        else [oracle.view(0, oracle.on_row[i]) for i in p.row])                  # p.row: a frame row / an onset
            assert (src[v] == 0).all(), (head, kind)                                          # never an event view: rule A3
        else:
            number = 2 if head == 'M4Conv' else 1
            expected = [oracle.view(number, oracle.event_row(u, number)) for u in p.row]       # p.row: the unit
            assert (src[v] == number).all(), (head, kind)
        assert v.tolist() == expected, (head, kind)


def test_view_numbers_follow_the_team_entries_of_the_gold_head_and_leave_out_the_onsets_without_a_window(ept_world, oracle):
    rows = number_rows(ept_world)
    fit = oracle.onsets('FIT')
    entries = [i for t in (0, 1) for i in fit if oracle.K[oracle.on_unit[i], t] >= 1]             # the team entries with a kill, blue first
    assert rows.view_numbers('M3Gold', 'masked').tolist() == [oracle.view(0, oracle.on_row[i]) for i in entries]
    kept = [i for i in fit if oracle.cexp[oracle.on_unit[i]] > 0]                                  # the units with a conversion window
    assert rows.view_numbers('M4Conv', 'masked').tolist() == [oracle.view(0, oracle.on_row[i]) for i in kept]
    units = [u for u in oracle.units('FIT') if oracle.event_row(u, 2) is not None]                 # the open M4 head: the src 2 view
    assert rows.view_numbers('M4Conv', 'open').tolist() == [oracle.view(2, oracle.event_row(u, 2)) for u in units]


def test_view_numbers_are_the_views_of_the_role_and_do_not_change_the_rows(ept_world):
    for role in ('FIT', 'STOP', 'DEV'):
        rows = number_rows(ept_world, role)
        before = [rows.parts(h, k).y.copy() for h, k in ALL_HEADS]
        views = [rows.view_numbers(h, k) for h, k in ALL_HEADS]
        assert all(np.array_equal(rows.parts(h, k).y, b) for (h, k), b in zip(ALL_HEADS, before)), role   # an accessor, nothing else
        roles = np.asarray(ept_world.s1.array('matches/role'))
        match = np.asarray(ept_world.s2.array('views/match'))
        for v in views:
            assert (roles[match[v]] == C.ROLES.index(role)).all(), role                            # the views of a role are the role's own


def test_view_numbers_refuse_a_head_and_a_kind_that_do_not_exist(ept_world):
    rows = number_rows(ept_world)
    for head, kind in (('M2Rate', 'open'), ('M2Mark', 'open'), ('M5', 'masked'), ('M3Size', 'both'), ('m3size', 'masked')):
        with pytest.raises(ValueError, match='there is no') as raised:
            rows.view_numbers(head, kind)
        assert KEY_PREFIX not in str(raised.value)


# =====================================================================================================================================
# the FIT statistics
# =====================================================================================================================================
@pytest.fixture(scope='module')
def stats(ept_world):
    return data.fit_stats(ept_world.s1, ept_world.s2)


@pytest.fixture
def make_world(tmp_path_factory):
    """``make_world(**kw)`` builds another EPT world (``ept_toys.build_ept_world``); the folders go when the test is over."""
    folders, worlds = [], []

    def make(**kw):
        folders.append(tmp_path_factory.mktemp('other_world'))
        worlds.append(build_ept_world(folders[-1], **kw))
        return worlds[-1]
    yield make
    worlds.clear()
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture
def make_store(tmp_path_factory):
    """``make_store(arrays, src)``: a store with the ``x/*`` arrays of ``arrays`` and ``views/src`` = ``src``; removed afterwards."""
    folders, stores = [], []

    def make(arrays, src):
        folders.append(tmp_path_factory.mktemp('scratch_store'))
        stores.append(write_views(folders[-1], arrays, src))
        return stores[-1]
    yield make
    stores.clear()
    gc.collect()
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)


def fit_rows(world, name):
    """The rows of the S2 array ``x/<name>`` that belong to FIT matches, as a numpy array."""
    return np.asarray(world.s2.array(f'x/{name}'))[world.fit_views]


def all_arrays(world):
    """Writable copies of every ``x/*`` array of the world."""
    return {name: np.array(world.s2.array(f'x/{name}')) for name in tokens.VIEW_SPEC}


class NoRead:
    """A store that fails the test when an array is read: a bad ``roles`` argument is refused before the first read."""

    def array(self, name):
        raise AssertionError(f'an array was read before the roles were checked: {name}')


def test_fit_stats_use_fit_views_only(ept_world, stats, make_world):
    """The numbers of every view of a match that is not FIT are shifted by 1,234.5: the statistics keep their bits and their hash."""
    other = make_world(shift=1234.5)
    assert np.array_equal(other.fit_views, ept_world.fit_views)
    near, far = (np.asarray(w.s2.array('x/players_num')) for w in (ept_world, other))
    outside = np.setdiff1d(np.arange(len(near)), ept_world.fit_views)
    assert np.nanmin(far[outside] - near[outside]) > 1000.0                                  # the shift is there, so the test can fail
    assert np.array_equal(far[ept_world.fit_views], near[ept_world.fit_views], equal_nan=True)
    near_xy, far_xy = (np.asarray(w.s2.array('x/ev_xy')) for w in (ept_world, other))        # the event positions are shifted too
    assert np.nanmin(far_xy[outside] - near_xy[outside]) > 1000.0
    assert np.array_equal(far_xy[ept_world.fit_views], near_xy[ept_world.fit_views], equal_nan=True)
    again = data.fit_stats(other.s1, other.s2)
    assert again.sha256 == stats.sha256 and again.n_views == stats.n_views == len(ept_world.fit_views)
    for name, cs in stats.arrays.items():
        a = again.arrays[name]
        assert np.array_equal(a.centre, cs.centre) and np.array_equal(a.scale, cs.scale) and np.array_equal(a.count, cs.count), name


def test_fit_stats_are_the_mean_and_scale_of_the_fit_cells_with_the_nan_left_out(ept_world, stats):
    p = fit_rows(ept_world, 'players_num').reshape(-1, len(tokens.PLAYER_NUM)).astype(np.float64)
    cs = stats.arrays['players_num']
    special = {PCOL[name] for name in data.POSITION_COLUMNS} | {PCOL[CONSTANT_PLAYER_COLUMN]}
    ordinary = [i for i in range(p.shape[1]) if i not in special]
    count = (~np.isnan(p)).sum(axis=0)
    assert (count[ordinary] > 0).all() and (count[PCOL['lvl_armor']] < len(p))          # an unknown champion has no traits: NaN not counted
    assert np.array_equal(cs.count, count)
    assert np.allclose(cs.centre[ordinary], np.nanmean(p, axis=0)[ordinary], rtol=1e-12, atol=0)
    assert np.allclose(cs.scale[ordinary], np.nanstd(p, axis=0)[ordinary], rtol=1e-12, atol=0)       # the scale is the population std
    assert stats.arrays['summary'].centre.shape == (len(tokens.SUMMARY) - 1,)                         # t_s is not an input
    landmarks = stats.arrays['landmarks']
    assert landmarks.centre.shape == (2,) and landmarks.count[1] < landmarks.count[0]                # the seconds cell of a nest may be NaN
    valid = np.arange(EPT_EVENTS)[None, :] < fit_rows(ept_world, 'ev_n')[:, None]
    dt = fit_rows(ept_world, 'ev_dt').astype(np.float64)[valid]                                   # the padding of the events is not counted
    ev = stats.arrays['ev_dt']
    assert ev.count[0] == valid.sum()
    assert ev.centre[0] == pytest.approx(dt.mean(), rel=1e-12) and ev.scale[0] == pytest.approx(dt.std(), rel=1e-12)
    pp = fit_rows(ept_world, 'geo_pp').astype(np.float64)                                            # one pair of numbers for all the pairs
    assert stats.arrays['geo_pp'].centre[0] == pytest.approx(np.nanmean(pp), rel=1e-12)
    assert stats.arrays['geo_pp'].scale[0] == pytest.approx(np.nanstd(pp), rel=1e-12)


def test_a_constant_column_has_scale_one_and_the_positions_are_centred_on_the_rotation_centre(ept_world, stats):
    players, summary = stats.arrays['players_num'], stats.arrays['summary']
    const = PCOL[CONSTANT_PLAYER_COLUMN]
    assert players.scale[const] == 1.0 and players.centre[const] == 0.0
    names = data.SUMMARY_COLUMNS
    assert 't_s' not in names and len(names) == len(tokens.SUMMARY) - 1
    for name, mean in ((CONSTANT_SUMMARY, 7777.0), (FIT_CONSTANT_SUMMARY, 7.0)):            # the second is constant over the FIT views only
        assert summary.scale[names.index(name)] == 1.0 and summary.centre[names.index(name)] == mean
    p = fit_rows(ept_world, 'players_num').reshape(-1, len(tokens.PLAYER_NUM)).astype(np.float64)
    for name in data.POSITION_COLUMNS:
        c = PCOL[name]
        assert players.centre[c] == C.W / 2.0
        assert players.scale[c] == pytest.approx(np.sqrt(np.nanmean((p[:, c] - C.W / 2.0) ** 2)), rel=1e-12), name
    ev_xy = fit_rows(ept_world, 'ev_xy').astype(np.float64)
    ev = stats.arrays['ev_xy']
    n_pos = int((~np.isnan(ev_xy[..., 0])).sum())                                          # the padding rows of ev_xy are NaN: not counted
    assert (ev.centre == C.W / 2.0).all() and ev.count.tolist() == [n_pos, n_pos]
    for axis in (0, 1):
        assert ev.scale[axis] == pytest.approx(np.sqrt(np.nanmean((ev_xy[..., axis] - C.W / 2.0) ** 2)), rel=1e-12)


def test_two_summary_columns_that_the_swap_exchanges_share_one_pair_of_statistics(ept_world, stats):
    perm, s = data.SUMMARY_PERM, stats.arrays['summary']
    moved = np.flatnonzero(perm != np.arange(len(perm)))
    assert moved.size >= 60                                                                # the blue and red columns of the objective state
    for field in ('centre', 'scale', 'count'):
        assert np.array_equal(getattr(s, field)[moved], getattr(s, field)[perm[moved]]), field
    c = int(moved[0])
    raw = fit_rows(ept_world, 'summary').astype(np.float64)
    in_store = lambda column: tokens.SUMMARY.index(data.SUMMARY_COLUMNS[column])                  # noqa: E731
    pair = np.concatenate([raw[:, in_store(c)], raw[:, in_store(perm[c])]])
    assert s.centre[c] == pytest.approx(np.nanmean(pair), rel=1e-12) and s.scale[c] == pytest.approx(np.nanstd(pair), rel=1e-12)
    assert s.count[c] == (~np.isnan(pair)).sum()


def test_stats_are_stored_as_json_with_a_hash_that_a_change_breaks(stats, tmp_path):
    path = tmp_path / 'STATS.json'
    sha = stats.save(path)
    again = data.Stats.load(path)
    assert sha == stats.sha256 == again.sha256 and json.loads(path.read_text(encoding='utf-8'))['sha256'] == sha
    for name, cs in stats.arrays.items():
        a = again.arrays[name]
        assert np.array_equal(a.centre, cs.centre) and np.array_equal(a.scale, cs.scale) and np.array_equal(a.count, cs.count), name
    text = path.read_text(encoding='utf-8')
    tampered = tmp_path / 'TAMPERED.json'
    tampered.write_text(text.replace('"n_views": ' + str(stats.n_views), '"n_views": ' + str(stats.n_views + 1)), encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        data.Stats.load(tampered)


def test_stats_refuse_a_role_other_than_fit_when_made_and_when_loaded(stats, tmp_path):
    """The statistics are those of FIT and nothing else (R14): a ``Stats`` that claims another role cannot be made, and a file that
    says so (with a hash that fits its content, so that only the role is wrong) cannot be loaded."""
    for role in ('STOP', 'DEV', 'TEST', 'fit', ''):
        with pytest.raises(ValueError, match='FIT'):
            data.Stats(arrays=stats.arrays, n_views=stats.n_views, role=role)
    content = {k: v for k, v in stats.to_json().items() if k != 'sha256'}
    for role in ('DEV', 'STOP'):
        content['role'] = role
        path = tmp_path / f'{role}.json'
        path.write_text(json.dumps({**content, 'sha256': runlog.sha256_json(content)}), encoding='utf-8')
        with pytest.raises(ValueError, match='FIT'):
            data.Stats.load(path)
    content['role'] = 'FIT'
    path = tmp_path / 'FIT.json'
    path.write_text(json.dumps({**content, 'sha256': runlog.sha256_json(content)}), encoding='utf-8')
    assert data.Stats.load(path).sha256 == stats.sha256                                       # the same file with the role of FIT loads


def test_fit_stats_refuse_every_role_but_fit_before_reading_anything(stats):
    assert data.DEFAULT_ROLES == {'fit': 'FIT'}
    for roles in ({'fit': 'DEV'}, {'fit': 'STOP'}, {'fit': 'TEST'}, {'fit': 'CAL'}, {'dev': 'DEV'}, {'fit': 'FIT', 'stop': 'STOP'}, {}):
        with pytest.raises(ValueError, match='FIT'):
            data.fit_stats(NoRead(), NoRead(), roles)
    with pytest.raises(ValueError, match='FIT'):
        data.fit_stats(NoRead(), NoRead(), None)
    assert stats.role == 'FIT'


def test_fit_stats_do_not_depend_on_how_the_views_are_cut_into_chunks_beyond_rounding(ept_world, stats, monkeypatch):
    """The merge of (DAT-3) over chunks of 7 views gives the statistics of one chunk up to rounding; one chunking gives the same bits."""
    monkeypatch.setattr(data, '_CHUNK', 7)
    cut = data.fit_stats(ept_world.s1, ept_world.s2)
    again = data.fit_stats(ept_world.s1, ept_world.s2)
    assert cut.sha256 == again.sha256 and cut.n_views == stats.n_views
    for name, cs in stats.arrays.items():
        c = cut.arrays[name]
        assert np.array_equal(c.count, cs.count), name
        assert np.allclose(c.centre, cs.centre, rtol=1e-12, atol=1e-9) and np.allclose(c.scale, cs.scale, rtol=1e-12, atol=0), name
    summary = cut.arrays['summary']
    perm = data.SUMMARY_PERM
    assert np.array_equal(summary.centre, summary.centre[perm])                                # still one pair of numbers for a pair
    assert np.array_equal(summary.scale, summary.scale[perm])


def test_the_running_moments_count_a_column_without_a_value_as_empty_and_a_constant_column_as_scale_one():
    nan = np.nan
    m = data._Moments(4)
    m.add(np.array([[nan, 1.0, 2.0, 5.0], [nan, 1.0, 6.0, nan]]))
    m.add(np.array([[nan, 1.0, 4.0, 7.0]]))
    cs = m.finish()
    assert cs.count.tolist() == [0, 3, 3, 2]
    assert cs.centre.tolist() == [0.0, 1.0, 4.0, 6.0]                                         # an empty column: centre 0, scale 1
    assert cs.scale[0] == 1.0 and cs.scale[1] == 1.0                                       # a constant column: scale 1
    assert cs.scale[2] == pytest.approx(np.sqrt(np.mean([4.0, 0.0, 4.0]))) and cs.scale[3] == pytest.approx(1.0)
    centred = m.finish(centred=(3,))
    assert centred.centre[3] == C.W / 2.0 and centred.scale[3] == pytest.approx(np.sqrt(np.mean([(5 - C.W / 2) ** 2, (7 - C.W / 2) ** 2])))
    with pytest.raises(ValueError):
        data.ColumnStats(np.zeros(2), np.zeros(2), np.zeros(2, dtype=np.int64))              # a scale of 0 is not a scale


# =====================================================================================================================================
# the view tensors
# =====================================================================================================================================
def expected_shapes(n, e, d):
    return {'view': (n,), 'kind': (n,), 'side': (n,), 'age': (n,), 'players': (n, 10, 81), 'players_nan': (n, 10, 23),
            'players_zone': (n, 10), 'players_xy': (n, 10, 2), 'landmarks': (n, 34, 2), 'landmarks_nan': (n, 34, 1), 'summary': (n, 106),
            'summary_nan': (n, 3), 'ev_valid': (n, e), 'ev_type': (n, e), 'ev_team': (n, e), 'ev_num': (n, e, 4), 'ev_xy': (n, e, 2),
            'ev_nan': (n, e, 1), 'ev_actors': (n, e, 10), 'ed_valid': (n, d), 'ed_i': (n, d), 'ed_j': (n, d), 'ed_kind': (n, d),
            'ed_dt': (n, d), 'geo_pp': (n, 10, 10), 'geo_pp_nan': (n, 10, 10), 'geo_pl': (n, 10, 34), 'geo_pl_nan': (n, 10, 34)}


INT_KEYS = {'view', 'kind', 'players_zone', 'ev_type', 'ev_team', 'ed_i', 'ed_j', 'ed_kind'}
BOOL_KEYS = {'ev_valid', 'ed_valid'}


def test_view_tensors_are_a_dict_of_tensors_with_the_shapes_of_the_store_manifest(ept_world, stats, make_world):
    views = np.array([5, 0, 7, 5, 200])                                                   # any order, a view twice
    other = make_world(n_events=7, n_edges=5)                                             # not the 128 and 128 of the real store
    for world, s, e, d in ((ept_world, stats, EPT_EVENTS, EPT_EDGES), (other, data.fit_stats(other.s1, other.s2), 7, 5)):
        b = data.ViewTensors(world.s2, views, s)
        assert isinstance(b, dict) and set(b) == set(expected_shapes(5, e, d)) == set(data.VIEW_KEYS)
        for key, shape in expected_shapes(5, e, d).items():
            t = b[key]
            assert isinstance(t, torch.Tensor) and tuple(t.shape) == shape, key
            assert t.dtype == (torch.int64 if key in INT_KEYS else torch.bool if key in BOOL_KEYS else torch.float32), key
        assert b['view'].tolist() == views.tolist()                                         # the rows follow the order asked for


def test_view_tensors_are_the_standardised_cells(ept_world, stats):
    views = np.array([7, 0, 7, 100, 31, 212])
    b = data.ViewTensors(ept_world.s2, views, stats)

    def std(raw, cs, keep=None):
        raw = raw.astype(np.float64)
        return np.where(np.isnan(raw), 0.0, (raw - cs.centre) / cs.scale).astype(np.float32)

    def stored(name):
        return np.asarray(ept_world.s2.array(f'x/{name}'))[views]
    summary = stored('summary')[:, [i for i, n in enumerate(tokens.SUMMARY) if n != 't_s']]
    assert torch.equal(b['players'], torch.from_numpy(std(stored('players_num'), stats.arrays['players_num'])))
    assert torch.equal(b['summary'], torch.from_numpy(std(summary, stats.arrays['summary'])))
    assert torch.equal(b['landmarks'], torch.from_numpy(std(stored('landmarks'), stats.arrays['landmarks'])))
    assert torch.equal(b['geo_pp'], torch.from_numpy(std(stored('geo_pp'), stats.arrays['geo_pp'])))
    assert torch.equal(b['geo_pl'], torch.from_numpy(std(stored('geo_pl'), stats.arrays['geo_pl'])))
    age = data.SUMMARY_COLUMNS.index('frame_age_s')
    assert torch.equal(b['age'], b['summary'][:, age]) and torch.equal(b['side'], torch.zeros(len(views)))
    src = np.asarray(ept_world.s2.array('views/src'))[views]
    assert b['kind'].tolist() == src.tolist() and set(src.tolist()) <= {0, 1, 2}
    x = b['players'][:, :, PCOL['x']].numpy()
    raw_x = stored('players_num')[:, :, PCOL['x']].astype(np.float64)
    scale_x, present = stats.arrays['players_num'].scale[PCOL['x']], ~np.isnan(raw_x)
    assert np.allclose(x[present], ((raw_x - C.W / 2.0) / scale_x)[present], atol=1e-6)                 # centred on W/2, not on the mean
    xy = b['players_xy'].numpy()                                                          # the raw position, the centre where it is missing
    assert np.array_equal(xy[~np.isnan(raw_x)], stored('players_num')[:, :, [PCOL['x'], PCOL['y']]][~np.isnan(raw_x)])
    assert (xy[np.isnan(raw_x)] == C.W / 2.0).all()


def test_allowed_nan_cells_become_zero_with_a_mask(ept_world, stats):
    n = len(ept_world.s2.array('views/src'))
    b = data.ViewTensors(ept_world.s2, np.arange(n), stats)
    names = ('players_num', 'summary', 'landmarks', 'ev_xy', 'geo_pp', 'geo_pl')
    raw = {name: np.asarray(ept_world.s2.array(f'x/{name}')) for name in names}
    # players: the allowed columns are the ones that tokens.NAN_ALLOWED names, in column order; there is no NaN anywhere else
    nan = np.isnan(raw['players_num'])
    allowed = [PCOL[name] for name in tokens.PLAYER_NUM if name in tokens.NAN_ALLOWED['players_num']['cells']]
    assert nan.any() and data.PLAYER_NAN_COLUMNS == tuple(tokens.PLAYER_NUM[i] for i in allowed) and len(allowed) == 23
    assert not np.delete(nan, allowed, axis=2).any()
    assert (b['players'].numpy()[nan] == 0.0).all()                                            # exactly zero where the cell was NaN
    assert np.array_equal(b['players_nan'].numpy(), nan[:, :, allowed].astype(np.float32))      # and the mask is 1 exactly there
    assert b['players_nan'].sum() == nan.sum() and b['players_nan'].numpy()[..., 0].any()       # x is NaN for a player without a position
    # the summary: the three NaN columns of tokens.NAN_ALLOWED
    s = raw['summary'][:, [i for i, name in enumerate(tokens.SUMMARY) if name != 't_s']]
    cols = [data.SUMMARY_COLUMNS.index(name) for name in tokens.NAN_ALLOWED['summary']['cells']]
    assert np.isnan(s).any() and data.SUMMARY_NAN_COLUMNS == tuple(data.SUMMARY_COLUMNS[i] for i in sorted(cols))
    assert (b['summary'].numpy()[np.isnan(s)] == 0.0).all()
    assert np.array_equal(b['summary_nan'].numpy(), np.isnan(s[:, sorted(cols)]).astype(np.float32))
    # the landmarks: the seconds cell of some rows
    lm = np.isnan(raw['landmarks'])
    assert lm.any() and not lm[..., 0].any() and (b['landmarks'].numpy()[lm] == 0.0).all()
    assert np.array_equal(b['landmarks_nan'].numpy(), lm[..., 1:].astype(np.float32))
    # the geodesics and the event positions
    for key, name in (('geo_pp', 'geo_pp'), ('geo_pl', 'geo_pl')):
        g = np.isnan(raw[name])
        assert g.any() and (b[key].numpy()[g] == 0.0).all() and np.array_equal(b[f'{key}_nan'].numpy(), g.astype(np.float32)), key
    valid = b['ev_valid'].numpy()
    missing = np.isnan(raw['ev_xy'][..., 0]) & valid                                         # a real event without a position (not padding)
    assert missing.any() and np.array_equal(b['ev_nan'].numpy()[..., 0], missing.astype(np.float32))
    assert (b['ev_xy'].numpy()[np.isnan(raw['ev_xy'])] == C.W / 2.0).all() and (b['ev_num'].numpy()[missing][:, 2:] == 0.0).all()
    for key, t in b.items():                                                                  # the network never sees a NaN or an infinity
        assert not t.is_floating_point() or torch.isfinite(t).all(), key


def test_padding_rows_of_the_events_and_edges_are_exactly_zero_and_the_masks_follow_the_counts(ept_world, stats):
    n = len(ept_world.s2.array('views/src'))
    b = data.ViewTensors(ept_world.s2, np.arange(n), stats)
    ev_n, ed_n = (np.asarray(ept_world.s2.array(f'x/{name}')) for name in ('ev_n', 'ed_n'))
    valid, valid_ed = np.arange(EPT_EVENTS)[None, :] < ev_n[:, None], np.arange(EPT_EDGES)[None, :] < ed_n[:, None]
    assert not valid.all() and valid.any() and not valid_ed.all() and valid_ed.any()
    assert np.array_equal(b['ev_valid'].numpy(), valid) and np.array_equal(b['ed_valid'].numpy(), valid_ed)
    pad, pad_ed = ~torch.from_numpy(valid), ~torch.from_numpy(valid_ed)
    assert (b['ev_num'][pad] == 0).all() and (b['ev_actors'][pad] == 0).all() and (b['ev_nan'][pad] == 0).all()
    assert (b['ev_xy'][pad] == C.W / 2.0).all() and (b['ev_type'][pad] == -1).all() and (b['ev_team'][pad] == 2).all()
    assert (b['ed_dt'][pad_ed] == 0).all() and (b['ed_i'][pad_ed] == -1).all()
    assert (b['ed_j'][pad_ed] == -1).all() and (b['ed_kind'][pad_ed] == -1).all()
    actors = np.asarray(ept_world.s2.array('x/ev_actors')).astype(np.int64)
    bits = (actors[..., None] >> np.arange(10)) & 1
    assert np.array_equal(b['ev_actors'].numpy(), bits.astype(np.float32))                      # one column for each participant
    assert np.array_equal(b['ed_dt'].numpy(), np.asarray(ept_world.s2.array('x/ed_dt')))         # in seconds: the decay is exp(-dt / kappa)
    assert np.array_equal(b['ed_i'].numpy(), np.asarray(ept_world.s2.array('x/ed_i')).astype(np.int64))


@pytest.mark.parametrize('name, index, value, text', [
    ('players_num', (3, 0, PCOL['level']), np.nan, 'players_num: NaN in level'),
    ('players_num', (3, 0, PCOL['level']), np.inf, 'players_num: an infinite value'),
    ('summary', (3, SCOL['dragon_up']), np.nan, 'summary: NaN in dragon_up'),
    ('landmarks', (3, 0, 1), np.nan, 'landmarks: NaN'),
    ('ev_dt', (1, 0), np.nan, 'ev_dt: NaN'),
    ('ed_dt', (1, 0), np.nan, 'ed_dt: NaN'),
    ('geo_pp', (3, 0, 1), np.inf, 'geo_pp: an infinite value')])
def test_a_nan_or_an_infinity_where_the_data_may_not_have_one_is_refused(ept_world, stats, make_store, name, index, value, text):
    arrays = all_arrays(ept_world)
    arrays[name][index] = value
    store = make_store(arrays, np.asarray(ept_world.s2.array('views/src')))
    with pytest.raises(ValueError, match=text):
        data.ViewTensors(store, np.array([index[0]]), stats)
    data.ViewTensors(store, np.array([index[0] + 1]), stats)                                 # another view of the same store is fine


def test_a_position_with_exactly_one_nan_coordinate_is_refused(ept_world, stats, make_store):
    """A missing position is NaN in both coordinates (``tokens.NAN_ALLOWED``). One NaN beside a number is a broken view: the mask
    channels would call one coordinate missing and the other present, and the statistics would count half a position."""
    src = np.asarray(ept_world.s2.array('views/src'))
    base = all_arrays(ept_world)
    players, events = base['players_num'], base['ev_xy']
    both = lambda a, i, j: ~np.isnan(a[..., i]) & ~np.isnan(a[..., j])                      # noqa: E731
    first = lambda mask: tuple(int(i) for i in np.argwhere(mask)[0])                          # noqa: E731
    cases = [('players_num', first(both(players, PCOL['x'], PCOL['y'])) + (PCOL['y'],), 'exactly one of x and y'),
             ('players_num', first(both(players, PCOL['prev_x'], PCOL['prev_y'])) + (PCOL['prev_x'],), 'exactly one of prev_x and prev_y'),
             ('ev_xy', first(both(events, 0, 1)) + (0,), 'ev_xy: exactly one')]
    for name, index, text in cases:
        arrays = all_arrays(ept_world)
        arrays[name][index] = np.nan
        store = make_store(arrays, src)
        with pytest.raises(ValueError, match=text):
            data.ViewTensors(store, np.array([index[0]]), stats)
    ok = all_arrays(ept_world)                                                               # a pair that is NaN in both is a missing position
    assert np.isnan(ok['players_num'][NO_POSITION_CELL][[PCOL['x'], PCOL['y']]]).all()
    data.ViewTensors(make_store(ok, src), np.array([NO_POSITION_CELL[0]]), stats)


@pytest.mark.parametrize('name, index, value, text', [
    ('ev_actors', (1, 0), 2000, 'ev_actors: values must lie'),
    ('ev_team', (1, 0), 3, 'ev_team: values must lie'),
    ('ed_i', (1, 0), 10, 'ed_i: values must lie'),
    ('ed_kind', (1, 0), 9, 'ed_kind: values must lie'),
    ('players_zone', (3, 1), 12, 'players_zone: values must lie'),
    ('ev_n', (1,), 0, 'ev_n and ev_type disagree'),
    ('ev_type', (1, 0), -1, 'ev_n and ev_type disagree'),
    ('ed_n', (1,), 0, 'ed_n, ed_i, ed_j and ed_kind disagree')])
def test_values_outside_their_ranges_and_padding_that_disagrees_with_the_counts_are_refused(
        ept_world, stats, make_store, name, index, value, text):
    """A participant, a team or an actor mask outside its range would be a wrong index in the swap or the encoder, and a count that does
    not match the padding would let a padding row in as an event or drop a real one."""
    arrays = all_arrays(ept_world)
    arrays[name][index] = value
    store = make_store(arrays, np.asarray(ept_world.s2.array('views/src')))
    with pytest.raises(ValueError, match=text):
        data.ViewTensors(store, np.array([index[0]]), stats)


def test_view_tensors_refuse_bad_view_numbers_and_stats_of_another_layout(ept_world, stats):
    n = len(ept_world.s2.array('views/src'))
    for views in (np.array([n]), np.array([-1]), np.array([[0, 1]]), np.array([0.5])):
        with pytest.raises((ValueError, TypeError)):
            data.ViewTensors(ept_world.s2, views, stats)
    assert tuple(data.ViewTensors(ept_world.s2, np.array([], dtype=np.int64), stats)['players'].shape) == (0, 10, 81)
    broken = data.Stats(arrays={**stats.arrays, 'summary': data.ColumnStats(np.zeros(5), np.ones(5), np.zeros(5, dtype=np.int64))},
                        n_views=stats.n_views)
    with pytest.raises(ValueError, match='summary'):
        data.ViewTensors(ept_world.s2, np.array([0]), broken)


# =====================================================================================================================================
# the team swap
# =====================================================================================================================================
def same_bits(x, y):
    """Same dtype, same values and, for a float tensor, the same sign bit in every cell: ``torch.equal`` alone calls -0.0 and +0.0
    equal, and it compares a float32 and a float64 tensor by value."""
    return x.dtype == y.dtype and torch.equal(x, y) and (not x.is_floating_point() or torch.equal(torch.signbit(x), torch.signbit(y)))


def equal_batches(a, b, skip=()):
    """The keys on which two batches differ in a bit (a key that one has and the other lacks counts)."""
    keys = (set(a) | set(b)) - set(skip)
    return sorted(k for k in keys if k not in a or k not in b or not same_bits(a[k], b[k]))


def mirror_arrays(world, ids):
    """The ``x/*`` arrays of the views ``ids`` after ``symmetry.swap_view`` and, since that leaves them in place, the permutation of the
    landmark rows and of the landmark columns of ``geo_pl`` (the step that its text hands to the loader): the numpy reference."""
    rows = {name: [] for name in tokens.VIEW_SPEC}
    for v in ids:
        ref = symmetry.swap_view(world_view(world.s2, int(v)))
        ref['landmarks'] = ref['landmarks'][symmetry.LANDMARK_SWAP]
        ref['geo_pl'] = ref['geo_pl'][:, symmetry.LANDMARK_SWAP]
        for name in rows:
            rows[name].append(ref[name])
    return {name: np.stack(a) for name, a in rows.items()}


def test_swapped_view_tensors_equal_the_standardised_swap_view_of_the_numpy_reference(ept_world, stats, make_store):
    """``swap(ViewTensors(v))`` is ``ViewTensors(swap_view(v))`` (landmarks permuted) in every bit, the sign bit of a zero included: the
    swap of the tensors is the swap of the data, not a second definition of it. This holds because the positions are centred on W/2, the
    summary columns that the swap exchanges share their statistics (DAT-4, DAT-5), the positions are whole numbers (a float32 ``W - x``
    is exact for them), and the swap negates as ``0.0 - t`` so that a +0.0 stays +0.0."""
    n = len(ept_world.s2.array('views/src'))
    ids = np.arange(n)
    mirrored = make_store(mirror_arrays(ept_world, ids), np.asarray(ept_world.s2.array('views/src')))
    b = data.ViewTensors(ept_world.s2, ids, stats)
    reference = data.ViewTensors(mirrored, ids, stats)
    swapped = data.swap(b)
    assert equal_batches(swapped, reference, skip=('side',)) == []
    assert (swapped['side'] == 1.0).all() and (reference['side'] == 0.0).all() and (b['side'] == 0.0).all()
    for key in ('players', 'players_xy', 'players_zone', 'landmarks', 'summary', 'ev_team', 'ev_xy', 'ev_actors', 'ed_i', 'geo_pp',
                'geo_pl'):
        assert not torch.equal(swapped[key], b[key]), key                             # the reference is not the identity: the test can fail


def test_the_swap_never_turns_a_zero_into_a_negative_zero(ept_world, stats):
    """A missing position and a padding row are +0.0 in the centred position channels, and the swap changes the sign of those channels.
    ``-t`` of a +0.0 is -0.0, which ``torch.equal`` cannot tell from +0.0 and the bits can; the swap must give +0.0 (``0.0 - t``)."""
    b = data.ViewTensors(ept_world.s2, np.arange(len(ept_world.s2.array('views/src'))), stats)
    swapped = data.swap(b)
    negative_zero = lambda t: bool(((t == 0) & torch.signbit(t)).any())                      # noqa: E731
    turned = {'players': b['players'][..., [PCOL[name] for name in data.POSITION_COLUMNS]], 'ev_num': b['ev_num'][..., 2:]}
    for name, t in turned.items():
        assert (t == 0).any(), name                                                          # the zeros are there, so the test can fail
    for key, t in b.items():
        assert not t.is_floating_point() or not negative_zero(t), key                      # the input has none ...
        assert not swapped[key].is_floating_point() or not negative_zero(swapped[key]), key      # ... and the swap makes none


def test_the_swap_of_the_view_tensors_is_an_involution_and_leaves_its_input_alone(ept_world, stats):
    b = data.ViewTensors(ept_world.s2, np.arange(0, 213, 4), stats)
    before = {k: t.clone() for k, t in b.items()}
    s = data.swap(b)
    assert equal_batches(b, before) == [] and set(s) == set(b)                           # the input is not changed
    s['players'] += 1.0                                                                  # the output does not share memory with the input
    assert equal_batches(b, before) == []
    assert equal_batches(data.swap(data.swap(b)), b) == []
    assert equal_batches(data.swap(b), data.swap(data.swap(data.swap(b)))) == []
    assert all(type(v) is torch.Tensor for v in s.values()) and type(s) is dict


@pytest.mark.skipif(os.environ.get('C_EPT_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
                    reason='opt-in: set C_EPT_CUDA_TESTS=1 (and have a GPU that nothing else, such as the local model, is using)')
def test_swap_runs_on_the_device_of_the_batch(ept_world, stats):
    """A batch that was moved to the GPU is swapped there (the cost check of S4a times the GPU too); the result equals the CPU swap.
    Opt-in, because the test makes a CUDA context, which a suite run must not do while the GPU belongs to another job."""
    batch = next(data.Batches(engagement_rows(ept_world), 4, 1, stats=stats).epoch(0))
    swapped = data.swap({k: v.to('cuda') for k, v in batch.items()})
    assert all(t.device.type == 'cuda' for t in swapped.values())
    assert equal_batches({k: v.cpu() for k, v in swapped.items()}, data.swap(batch)) == []
    assert equal_batches({k: v.cpu() for k, v in data.swap(swapped).items()}, batch) == []


def test_swap_refuses_a_key_that_it_does_not_know_how_to_swap(ept_world, stats):
    b = data.ViewTensors(ept_world.s2, np.array([0, 1]), stats)
    for extra in ('mystery', 'M3Kills/masked/mystery', 'M9Kills/masked/y', 'M3Kills/both/y'):
        with pytest.raises(ValueError, match='how to swap') as raised:
            data.swap({**b, extra: torch.zeros(2)})
        assert extra in str(raised.value)
    with pytest.raises(TypeError):
        data.swap({'players': np.zeros((2, 10, 81))})
    assert data.swap({}) == {}
    with pytest.raises(ValueError, match='mark'):                                    # a value outside its range is no mark / participant
        data.swap({'M2Mark/masked/y': torch.tensor([0, 52])})
    with pytest.raises(ValueError, match='participant'):
        data.swap({'ed_i': torch.tensor([[0, 10]])})


def test_the_zone_permutation_is_derived_from_the_zone_rules_for_every_zone():
    """The ten zones of ``terrain.ZONE_RULES`` each have one partner under the rotation (u, v) -> (1 - u, 1 - v): the zone of the
    rotated point is a function of the zone of the point on a lattice of the map and of whole numbers on the threshold lines. So no
    zone is left without a partner (the case that the brief provides for), and the table is the partner of the zone rules."""
    k = np.arange(0.0, C.W + 1.0, 125.0)
    gx, gy = np.meshgrid(k, k)
    extra = np.arange(0, int(C.W) + 1)                                               # the whole numbers on x + y = 5,355 and x + y = 24,395
    line_x = np.concatenate([extra, extra])                                               # (the two base thresholds, 0.36 W and 1.64 W)
    line_y = np.concatenate([5355 - extra, 24395 - extra])
    x, y = np.concatenate([gx.ravel(), line_x]), np.concatenate([gy.ravel(), line_y])
    zone, rotated = terrain.zone_of(x, y), terrain.zone_of(C.W - x, C.W - y)
    pairs = {(int(a), int(b)) for a, b in zip(zone, rotated)}
    assert {a for a, _ in pairs} == set(range(len(terrain.ZONES)))                          # every zone occurs
    assert len(pairs) == len(terrain.ZONES) and pairs == {(z, int(data.ZONE_PERM[z])) for z in range(len(terrain.ZONES))}
    names = {(terrain.ZONES[a], terrain.ZONES[b]) for a, b in pairs}
    assert names == {('base_blue', 'base_red'), ('base_red', 'base_blue'), ('top', 'bot'), ('bot', 'top'), ('mid', 'mid'),
                     ('river', 'river'), ('jg_blue_top', 'jg_red_bot'), ('jg_red_bot', 'jg_blue_top'),
                     ('jg_blue_bot', 'jg_red_top'), ('jg_red_top', 'jg_blue_bot')}          # worked out from the rules by hand
    assert np.array_equal(data.ZONE_PERM[data.ZONE_PERM], np.arange(10))
    t = data.swap({'M2Zone/masked/y': torch.tensor([-1, 0, 2, 9]), 'M2Zone/masked/m': torch.tensor([0, 1, 2, 3])})
    assert t['M2Zone/masked/y'].tolist() == [-1, 1, 4, 6]                                    # -1 (no position) stays -1


def test_the_mark_permutation_follows_the_players_who_change_places():
    """Mark m = 25 d + 5 i + j (d the killer's team, i his slot, j the victim's slot), 50 / 51 an execution of a blue / red victim: the
    mark of the swapped kill is the mark of the kill between the players who stand in the other rows after the swap."""
    def other(pid):                                                                    # the player who has this player's row after the swap
        return (pid - 1 + 5) % 10 + 1
    for killer in range(1, 11):
        for victim in range(1, 11):
            assert data.MARK_PERM[units.first_kill_mark(killer, victim)] == units.first_kill_mark(other(killer), other(victim))
    for victim in range(1, 11):                                                          # an execution has no killer: only the victim moves
        assert data.MARK_PERM[units.first_kill_mark(0, victim)] == units.first_kill_mark(0, other(victim))
    assert data.MARK_PERM[50] == 51 and data.MARK_PERM[51] == 50 and data.MARK_PERM[0] == 25 and data.MARK_PERM[24] == 49
    assert np.array_equal(data.MARK_PERM, [symmetry.swap_mark(m) for m in range(C.N_MARKS)])
    assert data.K_PERM.tolist() == [1, 0, 3, 2] and data.TEAM_PERM.tolist() == [1, 0]
    assert np.array_equal(data.CELL_PERM, symmetry.swap_O(np.arange(18))) and np.array_equal(data.PLAYER_PERM, symmetry.PLAYER_ORDER)


def test_a_swapped_mark_names_the_players_that_the_swapped_rows_hold(ept_world, stats):
    """Each player gets its row number as a tag. After the swap the killer and the victim of every mark sit in the rows that the swapped
    mark names: the pair of the first kill moves together with the players (review focus 1)."""
    b = dict(data.ViewTensors(ept_world.s2, np.array([20, 21]), stats))
    tag = PCOL[CONSTANT_PLAYER_COLUMN]
    b['players'] = b['players'].clone()
    b['players'][:, :, tag] = torch.arange(10, dtype=torch.float32)[None, :]
    marks = torch.arange(C.N_MARKS)
    n = len(marks)
    for name, value in (('pos', marks % 2), ('match', torch.zeros(n, dtype=torch.long)), ('row', marks), ('unit', marks),
                        ('q', torch.zeros(n, dtype=torch.long)), ('y', marks)):
        b[f'M2Mark/masked/{name}'] = value
    s = data.swap(b)
    new = s['M2Mark/masked/y']

    def row_of(tag_value, view):                                                      # where the player with this tag stands after the swap
        return int((s['players'][view, :, tag] == tag_value).nonzero()[0, 0])
    for m in range(50):
        view, d, i, j = m % 2, m // 25, (m % 25) // 5, m % 5
        killer, victim = row_of(5 * d + i, view), row_of(5 * (1 - d) + j, view)
        m2 = int(new[m])
        assert killer == 5 * (m2 // 25) + (m2 % 25) // 5 and victim == 5 * (1 - m2 // 25) + m2 % 5, m
    for m, side in ((50, 0), (51, 1)):                                                    # an execution of the blue / red victim
        after = 0 if row_of(5 * side, m % 2) < 5 else 1                                   # the team of that victim after the swap
        assert int(new[m]) == 50 + after and after == 1 - side


# =====================================================================================================================================
# the rows of an engagement and the batches
# =====================================================================================================================================
def engagement_rows(world, role='FIT', matches=None):
    return data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, role), matches)


def role_of_views(world, views):
    """The role name of the match of each view number."""
    match = np.asarray(world.s2.array('views/match'))[np.asarray(views)]
    return [C.ROLES[int(r)] for r in np.asarray(world.s1.array('matches/role'))[match]]


def key(head, kind, field):
    return f'{head}/{kind}/{field}'


UNIT_BLOCKS = [(h, k) for h, k in ALL_HEADS if h != 'M2Rate']             # the blocks whose rows belong to an engagement


def expected_swap(head, kind, field):
    """How a field of a row block changes under the swap, written out from the design (the review focus 1), not from the module."""
    if field == 'm':
        return 'mark'
    if field == 'z':
        return 'zone'
    if field == 'team':
        return 'team'
    if field in ('K', 'O'):
        return 'k' if field == 'K' else 'cells'
    if field == 'y':
        return {('M2Mark', 'masked'): 'mark', ('M2Zone', 'masked'): 'zone', ('M3Kills', 'masked'): 'teams', ('M3Kills', 'open'): 'teams',
                ('M3Obj', 'masked'): 'cells', ('M3Obj', 'open'): 'cells', ('M4Conv', 'masked'): 'cells', ('M4Conv', 'open'): 'cells'
                }.get((head, kind))
    return None


def reference_swap(change, t):
    """The change of ``expected_swap`` applied to the tensor ``t`` by plain Python loops over the numpy reference of ``symmetry``."""
    values = t.tolist()
    if t.numel() == 0:
        return t.clone()
    if change == 'mark':
        return torch.tensor([symmetry.swap_mark(v) for v in values], dtype=t.dtype)
    if change == 'zone':
        return torch.tensor([int(terrain.ZONE_SWAP[v]) if v >= 0 else -1 for v in values], dtype=t.dtype)
    if change == 'team':
        return torch.tensor([1 - v for v in values], dtype=t.dtype)
    if change == 'teams':
        return torch.tensor([[row[1], row[0]] for row in values], dtype=t.dtype).reshape(t.shape)
    if change == 'k':
        return torch.tensor([[r[1], r[0], r[3], r[2]] for r in values], dtype=t.dtype).reshape(t.shape)
    return torch.tensor(np.asarray(symmetry.swap_O(np.array(values))), dtype=t.dtype).reshape(t.shape)       # 'cells'


def test_swap_is_an_involution_and_moves_targets_with_inputs(ept_world, stats):
    """The brief's test. On every batch of an epoch: swap(swap(b)) == b; the pair of the first kill (the mark, the zone), the kills and
    the gold entries of the teams, K and O (the inputs of M3Obj and M4Conv and the targets of M3Kills, M3Obj and M4Conv) move with the
    players and positions of the views, while everything team-free stays. Every change is computed here by loops over the
    numpy reference of ``symmetry`` and ``terrain``, not by the rules of the module."""
    rows = engagement_rows(ept_world)
    batches = data.Batches(rows, 4, 1, stats=stats)
    moved = {}
    for batch in batches.epoch(0):
        swapped = data.swap(batch)
        assert set(swapped) == set(batch) and equal_batches(data.swap(swapped), batch) == []             # an involution
        assert equal_batches(swapped, data.swap(batch)) == []                                             # and deterministic
        for head, kind in ALL_HEADS:
            for field in data.ROW_FIELDS[(head, kind)]:
                k = key(head, kind, field)
                change = expected_swap(head, kind, field)
                want = batch[k] if change is None else reference_swap(change, batch[k])
                assert torch.equal(swapped[k], want), k
                if change is not None and len(batch[k]):
                    moved[(head, kind, field)] = moved.get((head, kind, field), False) or not torch.equal(batch[k], want)
        assert (swapped['side'] == 1.0).all() and (batch['side'] == 0.0).all()
        plain = [PCOL[n] for n in tokens.PLAYER_NUM if n not in data.POSITION_COLUMNS]
        turned = [PCOL[n] for n in data.POSITION_COLUMNS]
        assert torch.equal(swapped['players'][:, :5][..., plain], batch['players'][:, 5:][..., plain])      # the teams change places ...
        assert torch.equal(swapped['players'][:, 5:][..., plain], batch['players'][:, :5][..., plain])
        want = 0.0 - batch['players'][:, 5:][..., turned]  # subtraction keeps a positive zero (DAT-6)
        assert torch.equal(swapped['players'][:, :5][..., turned].contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))
        assert torch.equal(swapped['players_xy'][:, :5], C.W - batch['players_xy'][:, 5:])
        for k_ in ('view', 'kind', 'age', 'ev_type', 'ev_valid', 'ed_valid', 'ed_kind', 'ed_dt', 'ev_nan', 'summary_nan'):
            assert torch.equal(swapped[k_], batch[k_]), k_                                             # team-free inputs stay
    assert all(moved.values()) and {'mark', 'zone'} <= {expected_swap(h, k, f) for (h, k, f) in moved}
    targets = {(h, k, f) for (h, k, f) in moved if f == 'y'}
    assert targets >= {('M2Mark', 'masked', 'y'), ('M3Kills', 'open', 'y'), ('M4Conv', 'open', 'y'), ('M3Obj', 'masked', 'y')}


def test_the_fields_that_the_table_calls_team_free_really_are(ept_world):
    """The size class ``s`` depends on the people of the two teams through their minimum and maximum, so it does not change when the teams
    change places; the kill layer is a property of a gold entry, which keeps its row when its team flips."""
    for blue in range(0, 11):
        for red in range(0, 11):
            assert units.size_class(blue, red) == units.size_class(red, blue)
    team_free = ('layer', 'q', 's', 'pos', 'match', 'row', 'unit', 'e')
    assert all(change is None for (h, k, f), change in data.FIELD_SWAP.items() if f in team_free)
    assert data.FIELD_SWAP[('M3Gold', 'masked', 'y')] is None and data.FIELD_SWAP[('M3Size', 'open', 'y')] is None      # gold bin, size


def test_the_table_of_team_dependent_fields_lists_every_field_of_every_block(ept_world):
    assert set(data.FIELD_SWAP) == {(h, k, f) for (h, k), fields in data.ROW_FIELDS.items() for f in fields}
    assert set(data.ROW_FIELDS) == set(head_rows.VALID_HEADS) and len(data.ROW_FIELDS) == 13
    for (head, kind, field), change in data.FIELD_SWAP.items():
        assert change == expected_swap(head, kind, field), (head, kind, field)                   # written out again in this file
    assert set(data.FIELD_SWAP.values()) == {None, 'mark', 'zone', 'team', 'teams', 'k', 'cells'}
    rows = engagement_rows(ept_world)
    counts = rows.counts()
    oracle_rows = HeadsOracle(ept_world.base)
    assert counts['units'] == len(rows.units) > 0 and counts['frames'] == len(rows.frames) > 0
    assert counts['rows']['M2Mark/masked'] == len(oracle_rows.onsets('FIT'))
    assert counts['rows']['M3Size/open'] == len(oracle_rows.units('FIT'))
    assert set(counts['rows']) == {f'{h}/{k}' for h, k in ALL_HEADS}


def test_the_rows_of_an_epoch_are_the_rows_of_head_rows_once_each_and_read_the_views_of_the_rule(ept_world, stats, oracle):
    hr = data.ept_head_rows(ept_world.s1, ept_world.s2, ept_world.labels, 'FIT')
    batches = data.Batches(data.EngagementRows(hr), 3, 5, stats=stats)
    epoch = list(batches.epoch(0))
    onset_unit = np.asarray(ept_world.s1.array('onsets/unit')).astype(np.int64)
    for head, kind in ALL_HEADS:
        p = hr.parts(head, kind, masked_obj_k=True)
        n = len(p.y)
        got = {f: torch.cat([b[key(head, kind, f)] for b in epoch]) for f in data.ROW_FIELDS[(head, kind)]}
        # the view of every row (the plain-loop answer of the oracle), through the position in the batch
        views = torch.cat([b['view'][b[key(head, kind, 'pos')]] for b in epoch]).tolist()
        row = got['row'].tolist()
        if head == 'M2Rate':
            want_views = [oracle.view(0, r) for r in row]
        elif kind == 'masked':
            want_views = [oracle.view(0, oracle.on_row[r]) for r in row]
        else:
            number = 2 if head == 'M4Conv' else 1
            want_views = [oracle.view(number, oracle.event_row(r, number)) for r in row]
        assert views == want_views, (head, kind)
        if head == 'M2Rate':                                                                  # dense (frame row, 18 bins) against the cells
            cells = sorted((int(p.row[j]), int(p.codes['q'][j]), int(p.y[j]), float(np.float32(p.e[j]))) for j in range(n))
            y, e = got['y'].tolist(), got['e'].tolist()
            seen = sorted((row[i], q, y[i][q], e[i][q]) for i in range(len(row)) for q in range(18) if e[i][q] > 0)
            assert seen == cells and len(set(row)) == len(row)                                # every frame row once
            assert all(y[i][q] == 0 for i in range(len(row)) for q in range(18) if e[i][q] == 0)
            continue
        columns = {'match': p.match, 'row': p.row, 'unit': p.row if kind == 'open' else onset_unit[p.row], 'y': p.y}
        columns.update({name: v for name, v in p.codes.items()})
        if p.team is not None:
            columns['team'] = p.team
        if head == 'M4Conv':                                                                  # the only head with an exposure other than 1
            columns['e'] = p.e.astype(np.float32)
        if head == 'M4Conv' and kind == 'masked':
            d, extra = p.feat.shape[1], features.M4_EXTRA_NAMES                                 # the matrix is [row, q, m, z, K, O]
            assert p.m4.shape[1] == d + len(extra) + 18
            k0 = d + extra.index('kills_blue')
            assert extra[extra.index('kills_blue'):] == ('kills_blue', 'kills_red', 'gold_blue_k', 'gold_red_k')
            columns['K'], columns['O'] = p.m4[:, k0:k0 + 4], p.m4[:, d + len(extra):]           # the true K and O of the input matrix
        else:
            if p.K is not None:
                columns['K'] = p.K.astype(np.float32)
            if p.O is not None:
                columns['O'] = p.O.astype(np.float32)
        assert set(columns) == set(data.ROW_FIELDS[(head, kind)]) - {'pos'}, (head, kind)

        def table(source):
            return sorted(tuple(np.asarray(source[f][i], dtype=np.float64).ravel().tolist() for f in sorted(columns)) for i in range(n))
        assert len(got['row']) == n, (head, kind)                                                # each row once per epoch
        assert table({f: got[f].numpy() for f in columns}) == table({f: np.asarray(columns[f]) for f in columns}), (head, kind)


def test_a_changed_layout_of_the_masked_m4_matrix_is_refused_and_not_read_at_the_old_offsets(ept_world, monkeypatch):
    """K and O of a masked M4 row are read from the columns of ``features.m4_masked_inputs`` by their names. A column that the function
    gains at the end (the old offsets would still find K and O) or an extra name that it lists must stop the build, not pass."""
    real = features.m4_masked_inputs
    monkeypatch.setattr(features, 'm4_masked_inputs', lambda *a: np.concatenate([real(*a), np.zeros((len(a[1]), 1), np.float32)], axis=1))
    with pytest.raises(ValueError, match='columns'):
        engagement_rows(ept_world)
    monkeypatch.setattr(features, 'm4_masked_inputs', real)
    engagement_rows(ept_world)                                                                  # the real layout is accepted
    monkeypatch.setattr(features, 'M4_EXTRA_NAMES', features.M4_EXTRA_NAMES + ('another',))
    with pytest.raises(ValueError, match='columns'):
        engagement_rows(ept_world)


def test_batches_keep_masked_and_open_views_of_one_engagement_together(ept_world, stats, oracle):
    rows = engagement_rows(ept_world)
    batches = data.Batches(rows, 3, 7, stats=stats)
    place = {}                                                                                # unit -> {batch index}
    in_batch = []
    for index, batch in enumerate(batches.epoch(0)):
        in_batch.append(set(batch['view'].tolist()))
        for head, kind in UNIT_BLOCKS:
            for unit in batch[key(head, kind, 'unit')].tolist():
                place.setdefault(unit, set()).add(index)
    assert place and all(len(where) == 1 for where in place.values())                         # one batch for every engagement
    masked_units = {u for u in place if any(oracle.on_unit[i] == u for i in oracle.onsets('FIT'))}
    open_units = set(oracle.units('FIT'))
    both = masked_units & open_units
    assert len(both) >= 5 and set(place) == masked_units | open_units
    for unit in both:                                                                      # the masked and the open views of the engagement
        batch_views = in_batch[next(iter(place[unit]))]
        onsets = [i for i in oracle.onsets('FIT') if oracle.on_unit[i] == unit]
        wanted = {oracle.view(0, oracle.on_row[i]) for i in onsets} | {oracle.view(1, oracle.event_row(unit, 1))}
        if oracle.event_row(unit, 2) is not None:
            wanted.add(oracle.view(2, oracle.event_row(unit, 2)))
        assert wanted <= batch_views, unit
    assert len(batches) == -(-len(place) // 3) == len(in_batch)
    sizes = [len({u for u, where in place.items() if i in where}) for i in range(len(in_batch))]
    assert max(sizes) <= 3 and min(sizes) >= 1 and max(sizes) - min(sizes) <= 1                # even batches, none empty


def test_no_dev_row_in_training_batches(ept_world, stats):
    fit_views = set(ept_world.fit_views.tolist())
    batches = data.Batches(engagement_rows(ept_world), 3, 11, stats=stats)
    for epoch in range(2):
        for batch in batches.epoch(epoch):
            assert set(batch['view'].tolist()) <= fit_views and set(role_of_views(ept_world, batch['view'].tolist())) == {'FIT'}
            roles = np.asarray(ept_world.s1.array('matches/role'))
            for head, kind in ALL_HEADS:
                assert set(roles[batch[key(head, kind, 'match')].numpy()].tolist()) <= {C.ROLES.index('FIT')}
    for role in ('DEV', 'STOP'):                                                           # training batches read FIT rows and nothing else
        other = engagement_rows(ept_world, role)
        with pytest.raises(ValueError, match='FIT'):
            data.Batches(other, 3, 11, stats=stats)
        with pytest.raises(ValueError, match='FIT'):
            data.Batches(other, 3, 11, stats=stats, training=True)
        views = {v for batch in data.Batches(other, 3, 11, stats=stats, training=False).epoch(0) for v in batch['view'].tolist()}
        assert views and set(role_of_views(ept_world, sorted(views))) == {role}            # the evaluation batches read the role's own rows
    for role in ('CAL', 'AUDIT', 'TEST', 'EXTERNAL'):                                         # S3 never reads these roles
        with pytest.raises(ValueError):
            data.ept_head_rows(ept_world.s1, ept_world.s2, ept_world.labels, role)


def test_the_order_of_an_epoch_depends_on_seed_and_epoch_only_and_an_epoch_can_be_resumed(ept_world, stats):
    rows = engagement_rows(ept_world)
    a, b = data.Batches(rows, 3, 5, stats=stats), data.Batches(rows, 3, 5, stats=stats)
    first, again = list(a.epoch(0)), list(b.epoch(0))
    assert len(first) == len(again) == len(a) and all(equal_batches(x, y) == [] for x, y in zip(first, again))      # the same bits
    second = list(a.epoch(1))
    assert any(not torch.equal(x['view'], y['view']) for x, y in zip(first, second))              # another epoch, another order
    other_seed = list(data.Batches(rows, 3, 6, stats=stats).epoch(0))
    assert any(not torch.equal(x['view'], y['view']) for x, y in zip(first, other_seed))
    resumed = list(a.epoch(1, start=2))
    assert len(resumed) == len(second) - 2 and all(equal_batches(x, y) == [] for x, y in zip(resumed, second[2:]))
    assert list(a.epoch(0, start=len(a))) == []
    with pytest.raises(ValueError):
        list(a.epoch(0, start=len(a) + 1))
    order = lambda batches, epoch: [b['view'].tolist() for b in batches.epoch(epoch)]             # noqa: E731
    evaluation = [data.Batches(engagement_rows(ept_world, 'DEV'), 3, seed, stats=stats, training=False) for seed in (1, 2)]
    assert order(evaluation[0], 0) == order(evaluation[0], 1) == order(evaluation[1], 0)         # a fixed order, whatever the seed


def test_batches_have_no_default_epoch_to_iterate(ept_world, stats):
    """``for batch in batches`` would give the order of epoch 0 every time. The epoch is asked for by its number (``epoch(e)``)."""
    batches = data.Batches(engagement_rows(ept_world), 3, 5, stats=stats)
    with pytest.raises(TypeError):
        iter(batches)
    with pytest.raises(TypeError):
        list(batches)
    assert len(list(batches.epoch(0))) == len(batches) > 1


# =====================================================================================================================================
# the FIT-constant columns and the population of the statistics (S4b preparation M5, decisions D7 and D12)
# =====================================================================================================================================
D7_SUMMARY_FLAGS = ('atakhan_in_patch',) + tuple(n for n in tokens.SUMMARY if n.startswith('rule_uncertain_'))     # the nine flags of the decision
D7_PLAYER_FLAGS = tuple(n for n in tokens.PLAYER_NUM if n.startswith('flag_'))                                    # five always-zero fields of the toy
PAIR_B, PAIR_R = 'old_champion_kill_b', 'old_champion_kill_r'                  # two summary columns that the team swap exchanges
ELITE_B, ELITE_R = 'old_elite_monster_kill_b', 'old_elite_monster_kill_r'
WARD_B, WARD_R = 'old_ward_placed_b', 'old_ward_placed_r'


@pytest.fixture
def edit_world(ept_world, tmp_path_factory):
    """``edit_world(change)``: the EPT world with the ``x/*`` arrays that ``change(arrays)`` edited in place (same views, same S1); removed afterwards."""
    folders, worlds = [], []

    def make(change):
        folders.append(tmp_path_factory.mktemp('edited_world'))
        arrays = all_arrays(ept_world)
        change(arrays)
        s = Store(folders[-1] / 's2')
        for name in ('src', 'src_index', 'match', 'ref', 't0', 'frame_k'):
            a = np.asarray(ept_world.base.s2.array(f'views/{name}'))
            _put(s, f'views/{name}', a, a.dtype)
        for name in tokens.VIEW_SPEC:
            _put(s, f'x/{name}', arrays[name], DTYPES[name])
        s.finalize()
        worlds.append(SimpleNamespace(s1=ept_world.s1, s2=s, fit_views=ept_world.fit_views, base=ept_world.base))
        return worlds[-1]
    yield make
    worlds.clear()
    gc.collect()
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)


def fit_matches_of(world):
    """The FIT matches of the world (global match numbers, sorted), read from the views of FIT."""
    return tuple(int(m) for m in np.unique(np.asarray(world.s2.array('views/match'))[world.fit_views]))


def views_of(world, matches):
    """The FIT views of the given matches."""
    vm = np.asarray(world.s2.array('views/match'))
    return np.intersect1d(world.fit_views, np.flatnonzero(np.isin(vm, matches)))


def d7_edit(arrays):
    """The flags of the D7 list are 0 in every view; the pair old_elite_monster_kill_b / _r is 0 in both; old_champion_kill_b is 0 while its partner varies."""
    for name in D7_SUMMARY_FLAGS + (ELITE_B, ELITE_R, PAIR_B):
        arrays['summary'][:, tokens.SUMMARY.index(name)] = 0.0
    for name in D7_PLAYER_FLAGS:
        arrays['players_num'][:, :, tokens.PLAYER_NUM.index(name)] = 0.0


def run_stats(world, matches=None, drop='store_and_subset', population='selected_fit'):
    """``fit_stats`` the way a profile of S4b asks for it: the D7 list of the whole FIT store, then the statistics of the selected FIT."""
    store = data.store_constant_columns(world.s2, world.fit_views)
    return data.fit_stats(world.s1, world.s2, matches=matches, column_drop=drop, stats_population=population, store_constant=store)


def test_store_constant_columns_found_on_the_whole_store(ept_world, edit_world):
    w = edit_world(d7_edit)
    found = data.store_constant_columns(w.s2, w.fit_views)
    assert isinstance(found, tuple) and all(isinstance(n, str) for n in found)
    assert set(D7_SUMMARY_FLAGS) | set(D7_PLAYER_FLAGS) | {'t_s', ELITE_B, ELITE_R, CONSTANT_SUMMARY, FIT_CONSTANT_SUMMARY} <= set(found)
    assert 'game_time_s' not in found and 'totalGold' not in found and 'frame_age_s' not in found
    assert PAIR_B not in found and PAIR_R not in found                              # one column of the pair varies: judged on the pooled values
    assert found == features.dropped_columns(w.s2, w.fit_views, pooled=True)        # one rule for the EPT and the baselines
    # the list is judged on the views that are given: FIT_CONSTANT_SUMMARY is 7 in the FIT views and 5 in the others
    assert FIT_CONSTANT_SUMMARY not in data.store_constant_columns(w.s2, np.arange(len(w.s2.array('views/src'))))


def test_swap_pairs_are_dropped_together(edit_world):
    w = edit_world(d7_edit)
    found = data.store_constant_columns(w.s2, w.fit_views)
    assert (ELITE_B in found) and (ELITE_R in found) and PAIR_B not in found and PAIR_R not in found
    assert PAIR_B in features.dropped_columns(w.s2, w.fit_views)                    # the baseline list of every earlier run drops the half of the pair
    stats = run_stats(w, fit_matches_of(w))
    assert ELITE_B in stats.dropped and ELITE_R in stats.dropped and PAIR_B not in stats.dropped and PAIR_R not in stats.dropped
    names = data.SUMMARY_COLUMNS
    kept = [n for n in stats.dropped if n in names]
    assert all(names[data.SUMMARY_PERM[names.index(n)]] in kept for n in kept)       # closed under the exchange of the swap
    with pytest.raises(ValueError, match='swap'):                                    # a list that drops one half of a pair is refused
        data.Stats(arrays=stats.arrays, n_views=stats.n_views, dropped=('t_s', PAIR_B), dropped_store=('t_s', PAIR_B), column_drop='store')


def test_dropped_columns_are_zero_and_get_no_gradient(ept_world, edit_world):
    w = edit_world(d7_edit)
    stats = run_stats(w, fit_matches_of(w))
    plain = data.fit_stats(w.s1, w.s2)
    n = len(w.s2.array('views/src'))                                                  # every view, FIT or not: the other roles must not leak a value in
    b, c = data.ViewTensors(w.s2, np.arange(n), stats), data.ViewTensors(w.s2, np.arange(n), plain)
    names = data.SUMMARY_COLUMNS
    out_s = [names.index(x) for x in stats.dropped if x in names]
    out_p = [data.PLAYER_COLUMNS.index(x) for x in stats.dropped if x in data.PLAYER_COLUMNS]
    assert FIT_CONSTANT_SUMMARY in stats.dropped and names.index(FIT_CONSTANT_SUMMARY) in out_s and len(out_p) >= len(D7_PLAYER_FLAGS)
    assert (c['summary'][:, names.index(FIT_CONSTANT_SUMMARY)] != 0).any()           # without the drop the other roles' 5 would be an input of -2
    assert (b['summary'][:, out_s] == 0).all() and (b['players'][:, :, out_p] == 0).all()
    keep_s = [i for i in range(len(names)) if i not in out_s]
    keep_p = [i for i in range(len(data.PLAYER_COLUMNS)) if i not in out_p]
    assert torch.equal(b['summary'][:, keep_s], c['summary'][:, keep_s]) and torch.equal(b['players'][:, :, keep_p], c['players'][:, :, keep_p])
    for key in ('age', 'landmarks', 'ev_num', 'geo_pp', 'geo_pl', 'players_nan', 'summary_nan'):
        assert torch.equal(b[key], c[key]), key                                       # nothing else changes (these columns have no value to drop)
    # no gradient: the weights of the first layer that read a dropped column get exactly 0 and the others do not
    torch.manual_seed(0)
    enc = encoder.EPTEncoder(encoder.EncoderConfig(d=16, layers=1, heads=2, dropout=0.0, field_channels=2), toy_static())
    views = np.arange(0, n, 5)
    out = enc(data.ViewTensors(w.s2, views, stats))
    out.h_pool.square().sum().backward()
    grad_p, grad_s = enc.players_in.weight.grad, enc.summary_in.weight.grad
    assert (grad_p[:, out_p] == 0).all() and (grad_s[:, out_s] == 0).all()
    assert (grad_p[:, keep_p] != 0).any() and (grad_s[:, keep_s] != 0).any()
    assert enc.players_in.weight.shape[1] == len(data.PLAYER_COLUMNS) + len(data.PLAYER_NAN_COLUMNS)       # the input width does not change (Q5)
    assert enc.summary_in.weight.shape[1] == len(data.SUMMARY_COLUMNS) + len(data.SUMMARY_NAN_COLUMNS)


def test_a_dropped_column_without_any_fit_value_takes_its_mask_channel_with_it(ept_world, edit_world):
    p_none, p_some = 'lvl_armor', 'lvl_attackSpeed'
    s_none, s_some = data.SUMMARY_NAN_COLUMNS[0], data.SUMMARY_NAN_COLUMNS[1]
    fit = ept_world.fit_views

    def change(arrays):
        arrays['players_num'][fit, :, tokens.PLAYER_NUM.index(p_none)] = np.nan                    # no value in FIT at all
        arrays['players_num'][fit, :, tokens.PLAYER_NUM.index(p_some)] = 2.0                        # one value ...
        arrays['players_num'][fit[0], 0, tokens.PLAYER_NUM.index(p_some)] = np.nan                  # ... and a NaN
        arrays['summary'][fit, tokens.SUMMARY.index(s_none)] = np.nan
        arrays['summary'][fit, tokens.SUMMARY.index(s_some)] = 3.0
        arrays['summary'][fit[0], tokens.SUMMARY.index(s_some)] = np.nan
    w = edit_world(change)
    stats = run_stats(w, fit_matches_of(w))
    assert {p_none, p_some, s_none, s_some} <= set(stats.dropped)
    n = len(w.s2.array('views/src'))
    b = data.ViewTensors(w.s2, np.arange(n), stats)
    raw_players = np.asarray(w.s2.array('x/players_num'))
    nan_cols = data.PLAYER_NAN_COLUMNS
    assert (b['players_nan'][:, :, nan_cols.index(p_none)] == 0).all()                           # the channel of a column with no FIT value goes with it
    assert np.array_equal(b['players_nan'][:, :, nan_cols.index(p_some)].numpy(), np.isnan(raw_players[:, :, tokens.PLAYER_NUM.index(p_some)]).astype(np.float32))
    assert b['players_nan'][:, :, nan_cols.index(p_some)].sum() > 0                               # a column with a value keeps its channel
    raw_summary = np.asarray(w.s2.array('x/summary'))
    s_cols = data.SUMMARY_NAN_COLUMNS
    assert (b['summary_nan'][:, s_cols.index(s_none)] == 0).all()
    assert np.array_equal(b['summary_nan'][:, s_cols.index(s_some)].numpy(), np.isnan(raw_summary[:, tokens.SUMMARY.index(s_some)]).astype(np.float32))
    assert b['summary_nan'][:, s_cols.index(s_some)].sum() > 0
    for key in ('players', 'summary'):
        assert torch.isfinite(b[key]).all()


def test_frame_age_is_never_dropped(ept_world, edit_world):
    ordinary = run_stats(ept_world, fit_matches_of(ept_world))
    assert 'frame_age_s' not in ordinary.dropped and 'frame_age_s' not in data.store_constant_columns(ept_world.s2, ept_world.fit_views)
    age = tokens.SUMMARY.index('frame_age_s')

    def flat(arrays):
        arrays['summary'][:, age] = 0.0
    w = edit_world(flat)
    with pytest.raises(ValueError, match='frame_age_s'):                              # the store says it is constant: the loader stops for a report
        data.store_constant_columns(w.s2, w.fit_views)
    with pytest.raises(ValueError, match='frame_age_s'):                              # asked for by hand
        data.fit_stats(ept_world.s1, ept_world.s2, column_drop='store', store_constant=('t_s', 'frame_age_s'))
    with pytest.raises(ValueError, match='frame_age_s'):
        data.Stats(arrays=ordinary.arrays, n_views=ordinary.n_views, dropped=('t_s', 'frame_age_s'), dropped_store=('t_s', 'frame_age_s'), column_drop='store')
    matches = fit_matches_of(ept_world)
    member = views_of(ept_world, matches[:1])

    def flat_member(arrays):
        arrays['summary'][member, age] = 5.0                                          # constant inside the selected FIT, not over the store
    w2 = edit_world(flat_member)
    store = data.store_constant_columns(w2.s2, w2.fit_views)
    assert 'frame_age_s' not in store
    with pytest.raises(ValueError, match='constant inside the selected FIT'):         # the run's own subset must not drop it either, and the message says why (review m6)
        data.fit_stats(w2.s1, w2.s2, matches=matches[:1], column_drop='store_and_subset', stats_population='selected_fit', store_constant=store)
    kept = data.fit_stats(w2.s1, w2.s2, matches=matches[:1], column_drop='store', stats_population='selected_fit', store_constant=store)
    assert 'frame_age_s' not in kept.dropped


def test_selected_fit_population_uses_only_member_matches(ept_world, edit_world):
    matches = fit_matches_of(ept_world)
    member, others = matches[:2], matches[2:]
    legacy = data.fit_stats(ept_world.s1, ept_world.s2, matches=member)
    chosen = data.fit_stats(ept_world.s1, ept_world.s2, matches=member, stats_population='selected_fit')
    assert chosen.population == 'selected_fit' and legacy.population == 'whole_fit'
    assert chosen.sha256 != legacy.sha256 and chosen.to_json()['version'] == 2 and chosen.to_json()['population'] == 'selected_fit'      # the record is part of the statistics
    assert chosen.to_json()['column_drop'] == 'none' and chosen.to_json()['dropped'] == [] and chosen.to_json()['dropped_store'] == []
    assert chosen.n_views == len(views_of(ept_world, member)) < len(ept_world.fit_views)
    for name, cs in legacy.arrays.items():
        a = chosen.arrays[name]
        assert np.array_equal(a.centre, cs.centre) and np.array_equal(a.scale, cs.scale) and np.array_equal(a.count, cs.count), name
    whole = data.fit_stats(ept_world.s1, ept_world.s2)
    assert not np.array_equal(whole.arrays['players_num'].centre, chosen.arrays['players_num'].centre)      # the population matters
    moved = views_of(ept_world, others)

    def change(arrays):                                                                # the FIT matches outside the selection are moved far away
        arrays['summary'][moved, tokens.SUMMARY.index('game_time_s')] += 1000.0
        arrays['players_num'][moved, :, tokens.PLAYER_NUM.index('totalGold')] += 1000.0
    w = edit_world(change)
    again = data.fit_stats(w.s1, w.s2, matches=member, stats_population='selected_fit')
    assert again.sha256 == chosen.sha256                                              # only the member matches are read
    assert data.fit_stats(w.s1, w.s2).sha256 != whole.sha256
    for call in (lambda: data.fit_stats(NoRead(), NoRead(), matches=member, stats_population='selected_fit', column_drop='store', store_constant=('level',)),     # not the D7 list: no t_s
                 lambda: data.fit_stats(NoRead(), NoRead(), matches=member, stats_population='selected_fit', column_drop='store_and_subset', store_constant=()),
                 lambda: data.fit_stats(NoRead(), NoRead(), stats_population='selected_fit'),            # the selection is required, before any read
                 lambda: data.fit_stats(NoRead(), NoRead(), stats_population='everything'),
                 lambda: data.fit_stats(NoRead(), NoRead(), column_drop='all'),
                 lambda: data.fit_stats(NoRead(), NoRead(), matches=member, column_drop='store', store_constant=('t_s',)),     # the legacy population is not for the new drop
                 lambda: data.fit_stats(NoRead(), NoRead(), store_constant=('t_s',))):                  # a list without a column_drop that uses it
        with pytest.raises(ValueError):
            call()


def test_subset_constant_column_dropped_for_that_run_only(edit_world):
    plain = edit_world(lambda arrays: None)
    matches = fit_matches_of(plain)
    member = views_of(plain, matches[:2])
    level, half = tokens.PLAYER_NUM.index('level'), tokens.SUMMARY.index(WARD_B)

    def change(arrays):
        arrays['players_num'][member, :, level] = 4.0                                  # constant inside the selection ...
        arrays['summary'][member, half] = 3.0                                          # ... one half of a swap pair, the other half varies there
        arrays['summary'][member, tokens.SUMMARY.index(ELITE_B)] = 1.0
        arrays['summary'][member, tokens.SUMMARY.index(ELITE_R)] = 1.0                  # ... and a pair that is constant in both halves
        for name, value in (('x', 100.0), ('y', C.W / 2.0)):                              # a position that is constant inside the selection: x at 100, y at the rotation centre
            col = tokens.PLAYER_NUM.index(name)
            arrays['players_num'][member, :, col] = np.where(np.isnan(arrays['players_num'][member, :, col]), np.nan, value)
    w = edit_world(change)
    store = data.store_constant_columns(w.s2, w.fit_views)
    assert 'level' not in store and ELITE_B not in store and WARD_B not in store          # the whole FIT store varies in them
    a = data.fit_stats(w.s1, w.s2, matches=matches[:2], column_drop='store_and_subset', stats_population='selected_fit', store_constant=store)
    assert 'level' in a.dropped and ELITE_B in a.dropped and ELITE_R in a.dropped
    assert WARD_B not in a.dropped and WARD_R not in a.dropped                             # the pair is judged on its pooled values: ward_placed_r varies
    assert 'y' in a.dropped and 'x' not in a.dropped                                       # a position is judged with its rotation x -> W - x: only W/2 stays itself
    assert 'level' not in a.dropped_store and a.dropped_store == store and set(store) <= set(a.dropped)
    assert a.column_drop == 'store_and_subset' and a.population == 'selected_fit'
    b = data.fit_stats(w.s1, w.s2, matches=matches[:2], column_drop='store', stats_population='selected_fit', store_constant=store)
    assert b.dropped == store                                                          # 'store' leaves the subset-constant columns in
    c = run_stats(w, matches)                                                          # another run (all the FIT matches): nothing is dropped beyond the store
    assert c.dropped == store == c.dropped_store
    assert data.store_constant_columns(w.s2, w.fit_views) == store                     # the store list is not changed by a run
    with pytest.raises(ValueError, match='vary'):                                      # a store list that names a column that varies in the selected FIT
        data.fit_stats(w.s1, w.s2, matches=matches[:2], column_drop='store', stats_population='selected_fit', store_constant=store + ('totalGold',))


def test_none_and_whole_fit_defaults_are_bit_identical(ept_world, stats, tmp_path):
    explicit = data.fit_stats(ept_world.s1, ept_world.s2, column_drop='none', stats_population='whole_fit', store_constant=())
    assert explicit.sha256 == stats.sha256 and explicit.to_json() == stats.to_json()
    body = stats.to_json()
    assert body['version'] == 1 and not {'dropped', 'dropped_store', 'population', 'column_drop'} & set(body)       # the file of every earlier run
    assert stats.dropped == stats.dropped_store == () and stats.population == 'whole_fit' and stats.column_drop == 'none'
    views = np.arange(0, len(ept_world.s2.array('views/src')), 3)
    first, second = data.ViewTensors(ept_world.s2, views, stats), data.ViewTensors(ept_world.s2, views, explicit)
    assert first.keys() == second.keys() and all(torch.equal(first[k], second[k]) for k in first)
    member = fit_matches_of(ept_world)[:2]
    assert data.fit_stats(ept_world.s1, ept_world.s2, matches=member).sha256 == data.fit_stats(
        ept_world.s1, ept_world.s2, matches=member, column_drop='none', stats_population='whole_fit').sha256     # the legacy selection by matches
    # the bytes of save(): the earlier writer, written out by hand
    path = tmp_path / 'STATS.json'
    stats.save(path)
    hand = {'version': 1, 'kind': 'ept_view_stats', 'role': 'FIT', 'n_views': int(stats.n_views), 'centre': data.CENTRE,
            'columns': {'players_num': list(data.PLAYER_COLUMNS), 'summary': list(data.SUMMARY_COLUMNS)},
            'arrays': {name: {'centre': cs.centre.tolist(), 'scale': cs.scale.tolist(), 'count': cs.count.tolist()} for name, cs in sorted(stats.arrays.items())}}
    hand['sha256'] = runlog.sha256_json({k: v for k, v in hand.items()})
    assert path.read_text(encoding='utf-8') == json.dumps(hand, indent=1, ensure_ascii=False) + '\n'


def test_a_version_one_stats_file_still_loads_and_gives_the_same_values(ept_world, stats, tmp_path):
    body = {'version': 1, 'kind': 'ept_view_stats', 'role': 'FIT', 'n_views': int(stats.n_views), 'centre': data.CENTRE,
            'columns': {'players_num': list(data.PLAYER_COLUMNS), 'summary': list(data.SUMMARY_COLUMNS)},
            'arrays': {name: {'centre': cs.centre.tolist(), 'scale': cs.scale.tolist(), 'count': cs.count.tolist()} for name, cs in sorted(stats.arrays.items())}}
    path = tmp_path / 'V1.json'
    path.write_text(json.dumps({**body, 'sha256': runlog.sha256_json(body)}, indent=1), encoding='utf-8')
    old = data.Stats.load(path)
    assert old.sha256 == stats.sha256 and old.dropped == () and old.dropped_store == () and old.population == 'whole_fit' and old.column_drop == 'none'
    views = np.arange(0, len(ept_world.s2.array('views/src')), 4)
    a, b = data.ViewTensors(ept_world.s2, views, stats), data.ViewTensors(ept_world.s2, views, old)
    assert all(torch.equal(a[k], b[k]) for k in a)
    with_extra = {**body, 'dropped': ['t_s']}                                           # a version-1 file has no list of dropped columns
    path.write_text(json.dumps({**with_extra, 'sha256': runlog.sha256_json(with_extra)}), encoding='utf-8')
    with pytest.raises(ValueError, match='version'):
        data.Stats.load(path)
    for version in (0, 3):
        body_v = {**body, 'version': version}
        path.write_text(json.dumps({**body_v, 'sha256': runlog.sha256_json(body_v)}), encoding='utf-8')
        with pytest.raises(ValueError, match='version'):
            data.Stats.load(path)


def test_a_version_two_file_records_the_lists_with_their_hashes_and_a_change_breaks_it(edit_world, tmp_path):
    w = edit_world(d7_edit)
    stats = run_stats(w, fit_matches_of(w))
    assert stats.dropped and stats.dropped_store
    path = tmp_path / 'STATS.json'
    sha = stats.save(path)
    js = json.loads(path.read_text(encoding='utf-8'))
    assert js['version'] == 2 and js['population'] == 'selected_fit' and js['column_drop'] == 'store_and_subset'
    assert js['dropped'] == list(stats.dropped) and js['dropped_store'] == list(stats.dropped_store)
    assert js['dropped_sha256'] == runlog.sha256_json(list(stats.dropped)) and js['dropped_store_sha256'] == runlog.sha256_json(list(stats.dropped_store))
    again = data.Stats.load(path)
    assert again.sha256 == sha == stats.sha256 and again.dropped == stats.dropped and again.dropped_store == stats.dropped_store
    assert again.population == 'selected_fit' and again.column_drop == 'store_and_subset'
    views = np.arange(0, len(w.s2.array('views/src')), 4)
    a, b = data.ViewTensors(w.s2, views, stats), data.ViewTensors(w.s2, views, again)
    assert all(torch.equal(a[k], b[k]) for k in a)
    text = path.read_text(encoding='utf-8')
    tampered = tmp_path / 'TAMPERED.json'
    tampered.write_text(text.replace('"atakhan_in_patch",', '', 1), encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        data.Stats.load(tampered)
    # a file whose lists are re-signed but are not those of its claims is refused as well
    content = {k: v for k, v in js.items() if k != 'sha256'}
    shorter = [n for n in js['dropped_store'] if n != 'atakhan_in_patch']                # a valid list, but not the one whose hash the file states
    assert shorter != js['dropped_store'] and 't_s' in shorter
    for change in ({'column_drop': 'none'}, {'population': 'everywhere'}, {'dropped_store': ['nothing']}, {'dropped': js['dropped'] + ['totalGold_x']},
                   {'dropped_store': shorter}, {'dropped_store_sha256': runlog.sha256_json(shorter)}, {'dropped_sha256': runlog.sha256_json(shorter)}):
        bad = {**content, **change}
        tampered.write_text(json.dumps({**bad, 'sha256': runlog.sha256_json(bad)}), encoding='utf-8')
        with pytest.raises(ValueError):
            data.Stats.load(tampered)


def test_a_version_two_file_with_both_settings_at_their_defaults_is_refused(stats, tmp_path):
    """One content, one hash (review m2): the body of such statistics is version 1, so a version-2 file of the defaults would load into a Stats whose hash differs from its own."""
    body = {k: v for k, v in stats.to_json().items() if k != 'sha256'}
    body.update(version=2, population='whole_fit', column_drop='none', dropped=[], dropped_store=[], dropped_sha256=runlog.sha256_json([]),
                dropped_store_sha256=runlog.sha256_json([]))
    path = tmp_path / 'V2DEFAULT.json'
    path.write_text(json.dumps({**body, 'sha256': runlog.sha256_json(body)}), encoding='utf-8')
    with pytest.raises(ValueError, match='defaults'):
        data.Stats.load(path)


def test_the_stats_check_what_they_record(ept_world, stats):
    kw = dict(arrays=stats.arrays, n_views=stats.n_views)
    for bad in ({'column_drop': 'some'}, {'population': 'some'}, {'dropped': ('t_s',)},                                  # a list with column_drop 'none'
                {'column_drop': 'store', 'dropped': ('no_such_column',), 'dropped_store': ('no_such_column',)},
                {'column_drop': 'store', 'dropped': ('t_s',), 'dropped_store': ('t_s', 'level')},                        # the store list is part of the list
                {'column_drop': 'store', 'dropped': ('t_s', 'level'), 'dropped_store': ('t_s',)},                        # 'store' drops no more than the store list
                {'column_drop': 'store_and_subset', 'dropped': ('t_s',), 'dropped_store': ()}):                           # the store list always holds t_s
        with pytest.raises(ValueError):
            data.Stats(**kw, **bad)
    ok = data.Stats(**kw, column_drop='store_and_subset', dropped=('t_s', 'level'), dropped_store=('t_s',), population='selected_fit')
    assert ok.dropped == ('t_s', 'level') and ok.dropped_store == ('t_s',)


def test_eptdrop_list_equals_b1_list_when_pooled(ept_world, edit_world):
    w = edit_world(d7_edit)
    matches = fit_matches_of(w)
    store = data.store_constant_columns(w.s2, w.fit_views)
    assert store == features.dropped_columns(w.s2, w.fit_views, pooled=True)
    whole = data.fit_stats(w.s1, w.s2, column_drop='store_and_subset', stats_population='whole_fit', store_constant=store)
    assert whole.dropped == features.dropped_columns(w.s2, w.fit_views, pooled=True) and whole.dropped_store == store
    for member in (matches, matches[:2], matches[1:3]):
        stats = data.fit_stats(w.s1, w.s2, matches=member, column_drop='store_and_subset', stats_population='selected_fit', store_constant=store)
        assert stats.dropped == features.dropped_columns(w.s2, views_of(w, member), pooled=True), member            # the B1 list of that run
        assert set(store) <= set(stats.dropped)
    assert 't_s' in whole.dropped                                                       # on both sides
    assert not {'frame_age_s'} & set(whole.dropped)


def test_batches_without_stats_are_index_batches_that_the_caller_can_complete_with_view_tensors(ept_world, stats):
    """``Batches(rows, batch_size, seed)`` as the brief writes it: the distinct views and the row blocks; ``ViewTensors`` of those views
    completes the batch, bit for bit what ``Batches(..., stats=stats)`` gives."""
    rows = engagement_rows(ept_world)
    complete = list(data.Batches(rows, 3, 7, stats=stats).epoch(0))
    index = list(data.Batches(rows, 3, 7).epoch(0))
    assert len(index) == len(complete) > 1
    row_keys = {key(h, k, f) for (h, k), fields in data.ROW_FIELDS.items() for f in fields}
    for part, full in zip(index, complete):
        assert set(part) == {'view'} | row_keys and part['view'].dtype == torch.int64
        assert all(torch.equal(part[k], full[k]) for k in part)
        assert equal_batches({**part, **data.ViewTensors(ept_world.s2, part['view'], stats)}, full) == []
        assert equal_batches(data.swap(part), {k: v for k, v in data.swap(full).items() if k in part}) == []       # swap works on a part


def test_the_data_layer_runs_under_deterministic_algorithms(ept_world, stats):
    """The reported values come from CPU runs with ``torch.use_deterministic_algorithms(True)`` (A6a): batches and swap must work there."""
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        first = list(data.Batches(engagement_rows(ept_world), 4, 1, stats=stats).epoch(0))
        again = list(data.Batches(engagement_rows(ept_world), 4, 1, stats=stats).epoch(0))
        assert all(equal_batches(a, b) == [] for a, b in zip(first, again))
        assert all(equal_batches(data.swap(data.swap(b)), b) == [] for b in first)
    finally:
        torch.use_deterministic_algorithms(before)


def test_the_proxy_matches_restrict_the_rows_to_a_quarter_of_the_fit_matches(ept_world, stats):
    proxy = data.proxy_matches(ept_world.s1)
    full, small = engagement_rows(ept_world), engagement_rows(ept_world, matches=proxy['FIT'])
    assert 0 < len(small.units) < len(full.units) and 0 < len(small.frames) < len(full.frames)
    for batch in data.Batches(small, 3, 1, stats=stats).epoch(0):
        for head, kind in ALL_HEADS:
            assert set(batch[key(head, kind, 'match')].tolist()) <= set(proxy['FIT'].tolist())
        assert set(role_of_views(ept_world, batch['view'].tolist())) == {'FIT'}
    stop = engagement_rows(ept_world, 'STOP', proxy['STOP'])
    assert len(stop.units) > 0


def test_batches_refuse_a_bad_size_stats_or_rows(ept_world, stats):
    rows = engagement_rows(ept_world)
    for size in (0, -1, 1.5, True, '3'):
        with pytest.raises((ValueError, TypeError)):
            data.Batches(rows, size, 1, stats=stats)
    with pytest.raises(TypeError):
        data.Batches(rows, 3, 1, stats=object())
    with pytest.raises(TypeError):
        data.Batches(object(), 3, 1, stats=stats)
    for seed in (-1, 1.5, True):
        with pytest.raises(ValueError):
            data.Batches(rows, 3, seed, stats=stats)
    assert len(data.Batches(data.ept_head_rows(ept_world.s1, ept_world.s2, ept_world.labels, 'FIT'), 3, 1, stats=stats)) == len(
        data.Batches(rows, 3, 1, stats=stats))                                                # HeadRows are wrapped


# =====================================================================================================================================
# the proxy selection
# =====================================================================================================================================
class RolesOnly:
    """An S1 stand-in that holds ``matches/role`` and fails the test when anything else is read."""

    def __init__(self, names):
        self.roles = np.array([C.ROLES.index(n) for n in names], dtype=np.int8)

    def array(self, name):
        assert name == 'matches/role', f'the proxy selection read {name}'
        return self.roles


ROLE_NAMES = ('FIT', 'STOP', 'DEV', 'FIT', 'FIT', 'STOP', 'CAL', 'FIT', 'FIT', 'FIT', 'STOP', 'FIT', 'DEV', 'FIT', 'FIT', 'STOP')


def test_proxy_selection_is_nested_and_seeded():
    store = RolesOnly(ROLE_NAMES)
    fit = [i for i, n in enumerate(ROLE_NAMES) if n == 'FIT']
    stop = [i for i, n in enumerate(ROLE_NAMES) if n == 'STOP']
    assert (len(fit), len(stop)) == (9, 4)
    shares = (0.1, 0.25, 0.5, 0.75, 1.0)
    picks = [data.proxy_matches(store, share, 11) for share in shares]
    assert [len(p['FIT']) for p in picks] == [1, 3, 5, 7, 9] and [len(p['STOP']) for p in picks] == [1, 1, 2, 3, 4]       # ceil(share n)
    for small, large in zip(picks, picks[1:]):                                             # the same seed: a small share is in a larger one
        for role in ('FIT', 'STOP'):
            assert set(small[role].tolist()) <= set(large[role].tolist()), role
    assert picks[-1]['FIT'].tolist() == fit and picks[-1]['STOP'].tolist() == stop
    for p in picks:                                                                           # only FIT and STOP, sorted, of the right role
        assert set(p) == {'FIT', 'STOP'} and all(a.dtype == np.int64 and a.tolist() == sorted(set(a.tolist())) for a in p.values())
        assert set(p['FIT'].tolist()) <= set(fit) and set(p['STOP'].tolist()) <= set(stop)
    assert all(np.array_equal(x[r], y[r]) for x, y in zip(picks, [data.proxy_matches(store, s, 11) for s in shares]) for r in x)
    picked = {tuple(data.proxy_matches(store, 0.25, seed)['FIT'].tolist()) for seed in range(30)}
    assert len(picked) > 1                                                                    # the seed changes the pick
    default = data.proxy_matches(store)
    assert data.PROXY_SEED == 20261022 and all(np.array_equal(default[r], data.proxy_matches(store, 0.25, 20261022)[r]) for r in default)
    one = data.proxy_matches(RolesOnly(('FIT', 'STOP')), 0.25, 3)                              # one match of each: the minimum is one
    assert one['FIT'].tolist() == [0] and one['STOP'].tolist() == [1]


def test_proxy_selection_refuses_a_bad_share_and_a_role_without_matches():
    store = RolesOnly(ROLE_NAMES)
    for share in (0, -0.1, 1.5, float('nan')):
        with pytest.raises(ValueError, match='share'):
            data.proxy_matches(store, share)
    with pytest.raises(ValueError, match='STOP'):
        data.proxy_matches(RolesOnly(('FIT', 'FIT', 'DEV')), 0.5)
    with pytest.raises(ValueError, match='FIT'):
        data.proxy_matches(RolesOnly(('STOP', 'DEV')), 0.5)


# =====================================================================================================================================
# structure
# =====================================================================================================================================
def test_the_data_module_reads_no_key_no_holdout_and_no_cohort_module():
    from pathlib import Path
    from c_ept.tests.conftest import all_imports
    source = Path(data.__file__)
    text = source.read_text(encoding='utf-8')
    assert 'matches/key' not in text and 'allow_holdout' not in text and 'split_module' not in text and 'bucket(' not in text
    assert not all_imports(source) & {'c_ept.cohort', 'c_ept.reader', 'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.run_s3', 'pickle', 'lightgbm'}


def test_the_toy_world_obeys_the_nan_rules_of_the_views(ept_world):
    from c_ept.tests.view_helpers import nan_conditions_broken, nan_outside_allowed
    for v in range(len(ept_world.s2.array('views/src'))):
        view = world_view(ept_world.s2, v)
        assert nan_outside_allowed(view) == {} and nan_conditions_broken(view) == [], v


@pytest.mark.parametrize('dtype,integer', [(torch.float32, torch.int32), (torch.float64, torch.int64)])
def test_deferred_centered_coordinates_preserve_exact_zero_bits(dtype, integer):
    """DAT-6 uses subtraction: both zero signs map to +0; unary minus is not the same bit operation."""
    coordinates = [data.PLAYER_COLUMNS.index(c) for c in data.POSITION_COLUMNS]
    value = torch.zeros((1, 10, len(data.PLAYER_COLUMNS)), dtype=dtype)
    sample = torch.tensor([0.0, -0.0, 1.25, -2.5], dtype=dtype)
    value[..., coordinates] = sample
    baseline = value.clone()
    got = data.swap({'players': value})['players']
    expected = value[:, data.PLAYER_PERM.tolist()].clone()
    expected[..., coordinates] = torch.tensor([0.0, 0.0, -1.25, 2.5], dtype=dtype)
    assert torch.equal(got.contiguous().view(integer), expected.contiguous().view(integer))
    assert torch.equal(value.view(integer), baseline.view(integer))
    unary = -sample
    assert not torch.equal(got[0, 0, coordinates].view(integer), unary.view(integer))
    twice = data.swap({'players': got})['players']
    normalized = baseline.clone()
    normalized[..., coordinates] = torch.tensor([0.0, 0.0, 1.25, -2.5], dtype=dtype)
    assert torch.equal(twice.contiguous().view(integer), normalized.contiguous().view(integer))


# =====================================================================================================================================
# the random view swap of the training batches (EPT-T012 S4b preparation M7, decisions D10 P1 and D17, plan Q3)
# =====================================================================================================================================
SWAP_STREAM = 0x5157                                         # the last word of the seed sequence of the swap coins (plan Q3)


def swap_batches(world, stats, mode, *, seed=5, size=3, rows=None):
    return data.Batches(engagement_rows(world) if rows is None else rows, size, seed, stats=stats, swap_mode=mode)


def reference_coins(seed, epoch, index, views):
    """Plan Q3 written out: the generator of SeedSequence([seed, epoch, batch index, 0x5157]) gives one fair coin for each distinct view of the batch, in the order of ``view``."""
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, index, SWAP_STREAM])).integers(0, 2, size=views).astype(bool)


def test_the_swap_modes_are_named_and_checked(ept_world, stats):
    assert data.SWAP_MODES == ('average', 'random_view', 'none')
    rows = engagement_rows(ept_world)
    assert data.Batches(rows, 3, 5, stats=stats).swap_mode == 'average'
    for mode in data.SWAP_MODES:
        assert data.Batches(rows, 3, 5, stats=stats, swap_mode=mode).swap_mode == mode
    for bad in ('Random_view', 'random', 'swap', '', None, 1, True):
        with pytest.raises(ValueError, match='swap_mode'):
            data.Batches(rows, 3, 5, stats=stats, swap_mode=bad)
    stop = engagement_rows(ept_world, 'STOP')
    with pytest.raises(ValueError, match='swap_mode'):                     # the order and the frames of an evaluation batch are fixed: no coin
        data.Batches(stop, 3, 5, stats=stats, training=False, swap_mode='random_view')
    assert data.Batches(stop, 3, 5, stats=stats, training=False).swap_mode == 'average'


def test_random_view_batches_carry_one_fair_coin_for_each_view_from_the_seeded_generator(ept_world, stats):
    """``swap_view`` is a bool tensor with one entry for each distinct view of the batch (the entries of ``view``); the entry is the coin of the generator of plan Q3, so a test can
    recompute it from the seed, the epoch and the index of the batch alone."""
    batches = swap_batches(ept_world, stats, 'random_view')
    seen = []
    for epoch in range(3):
        for index, batch in enumerate(batches.epoch(epoch)):
            flag = batch['swap_view']
            assert flag.dtype == torch.bool and flag.shape == batch['view'].shape
            assert flag.tolist() == reference_coins(5, epoch, index, len(batch['view'])).tolist(), (epoch, index)
            seen += flag.tolist()
    assert len(seen) >= 30 and True in seen and False in seen and 0.25 < sum(seen) / len(seen) < 0.75          # fair coins, not one side (control: the test can see a constant)


def test_the_other_modes_give_the_plain_batches_bit_for_bit_and_random_view_only_adds_the_coins(ept_world, stats):
    """test_average_default_is_bit_identical, data level. 'average' (the default) and 'none' emit no ``swap_view`` and every key of every batch is that of ``Batches`` without the setting;
    'random_view' emits the recorded batch and the coins, and nothing else differs: the rows stay in the frame they were recorded in until the coins are applied."""
    rows = engagement_rows(ept_world)
    plain = [list(data.Batches(rows, 3, 5, stats=stats).epoch(e)) for e in range(2)]
    for mode in ('average', 'none'):
        got = [list(swap_batches(ept_world, stats, mode).epoch(e)) for e in range(2)]
        assert [len(x) for x in got] == [len(x) for x in plain]
        assert all(equal_batches(a, b) == [] and 'swap_view' not in a and list(a) == list(b) for xs, ys in zip(got, plain) for a, b in zip(xs, ys)), mode
    coins = [list(swap_batches(ept_world, stats, 'random_view').epoch(e)) for e in range(2)]
    for xs, ys in zip(coins, plain):
        for a, b in zip(xs, ys):
            assert equal_batches(a, b, skip=('swap_view',)) == [] and list(a)[-1] == 'swap_view' and 'swap_view' in a and 'swap_view' not in b


def test_the_coins_depend_on_seed_epoch_and_batch_index_only_and_leave_the_global_random_state_alone(ept_world, stats):
    """Plan Q3: not the global torch or numpy stream (a dropout mask would move, and a prefetch thread would race), but a generator of its own for each batch. The coins are the same whatever
    the global state is, whatever is drawn between the batches (dropout, numpy), and for a resumed epoch; another seed, epoch or batch gives other coins."""
    import random
    batches = swap_batches(ept_world, stats, 'random_view')
    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    before = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), np.random.get_state()[2], random.getstate())
    first = [b['swap_view'] for b in batches.epoch(0)]
    assert torch.equal(before[0], torch.get_rng_state()) and np.array_equal(before[1], np.random.get_state()[1]) and before[2] == np.random.get_state()[2] and before[3] == random.getstate()
    torch.manual_seed(99)
    np.random.seed(99)
    between = []
    for batch in swap_batches(ept_world, stats, 'random_view').epoch(0):
        torch.nn.functional.dropout(torch.ones(64), 0.5, training=True)                      # what a training step draws between two batches
        np.random.random(3)
        between.append(batch['swap_view'])
    assert len(first) == len(between) and all(torch.equal(a, b) for a, b in zip(first, between))
    resumed = [b['swap_view'] for b in batches.epoch(0, start=1)]
    assert len(resumed) == len(first) - 1 and all(torch.equal(a, b) for a, b in zip(resumed, first[1:]))      # a resumed epoch repeats the coins of the batches it skips to
    other = {'epoch': [b['swap_view'] for b in batches.epoch(1)], 'seed': [b['swap_view'] for b in swap_batches(ept_world, stats, 'random_view', seed=6).epoch(0)]}
    for name, flags in other.items():
        assert any(not torch.equal(a, b) for a, b in zip(first, flags)), name


def reference_framed(batch, flag):
    """The batch as each view chose: the views of a coin that came up are the swapped views (``data.swap``, tested against the numpy reference above), the rows of such a view the swapped rows
    (the design table ``expected_swap`` applied by the plain loops of ``reference_swap``; the first-kill team ``d`` becomes 1 - d, -1 stays)."""
    plain = {k: v for k, v in batch.items() if k != 'swap_view'}
    whole = data.swap({k: plain[k] for k in data.VIEW_KEYS})
    out = {k: torch.where(flag.reshape(-1, *([1] * (plain[k].dim() - 1))), whole[k], plain[k]) for k in data.VIEW_KEYS}
    for head, kind in ALL_HEADS:
        pos = plain[key(head, kind, 'pos')]
        mask = flag[pos]
        for field in data.ROW_FIELDS[(head, kind)] + tuple(f for f in data.OPTIONAL_ROW_FIELDS.get((head, kind), ()) if key(head, kind, f) in plain):
            t = plain[key(head, kind, field)]
            if field in data.OPTIONAL_ROW_FIELDS.get((head, kind), ()):
                moved = torch.where(t >= 0, 1 - t, t)
            else:
                change = expected_swap(head, kind, field)
                moved = t if change is None else reference_swap(change, t)
            out[key(head, kind, field)] = torch.where(mask.reshape(-1, *([1] * (t.dim() - 1))), moved, t)
    return out


@pytest.mark.parametrize('law', ['poisson', 'shifted'])
def test_rows_follow_their_view_frame(consistent_ept_world, law):
    """Plan M7: the rows that read one view are all in the frame of that view, inputs and targets together. For every batch of three epochs, every field of every block (the open
    first-kill team ``d`` of the shifted law included) is the swapped field where the coin of the row's view came up and the recorded field where it did not, and the views are swapped in
    the same places; the reference is written from the design table, not from ``swap_views``. The control counts that the coins really moved a field of every kind."""
    world = consistent_ept_world
    stats_ = data.fit_stats(world.s1, world.s2)
    rows = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT', kill_law=law))
    moved = set()
    for epoch in range(3):
        for batch in swap_batches(world, stats_, 'random_view', rows=rows).epoch(epoch):
            flag = batch['swap_view']
            want = reference_framed(batch, flag)
            got = data.swap_views(batch)
            assert 'swap_view' not in got and set(got) == set(want) and equal_batches(got, want) == []
            for head, kind in ALL_HEADS:
                mask = flag[batch[key(head, kind, 'pos')]]
                for field in data.ROW_FIELDS[(head, kind)] + tuple(f for f in data.OPTIONAL_ROW_FIELDS.get((head, kind), ()) if key(head, kind, f) in batch):
                    if bool(mask.any()) and not torch.equal(got[key(head, kind, field)], batch[key(head, kind, field)]):
                        moved.add((head, kind, field))
    assert {('M2Mark', 'masked', 'y'), ('M2Zone', 'masked', 'm'), ('M2Zone', 'masked', 'y'), ('M3Kills', 'open', 'y'), ('M3Obj', 'masked', 'K'), ('M4Conv', 'open', 'O'),
            ('M3Gold', 'masked', 'team')} <= moved
    assert (('M3Kills', 'open', 'd') in moved) == (law == 'shifted')


def test_swap_views_takes_the_flags_of_the_batch_and_refuses_what_it_cannot_frame(ept_world, stats):
    batch = next(iter(swap_batches(ept_world, stats, 'random_view').epoch(0)))
    n = len(batch['view'])
    plain = {k: v for k, v in batch.items() if k != 'swap_view'}
    down, up = dict(plain, swap_view=torch.zeros(n, dtype=torch.bool)), dict(plain, swap_view=torch.ones(n, dtype=torch.bool))
    assert equal_batches(data.swap_views(down), plain) == []                        # no coin up: the recorded batch, bit for bit
    assert equal_batches(data.swap_views(up), data.swap(plain)) == []               # every coin up: the whole batch seen from the other side
    before = {k: v.clone() for k, v in down.items()}
    out = data.swap_views(down)
    assert equal_batches(down, before) == [] and all(out[k].data_ptr() != plain[k].data_ptr() for k in out if out[k].numel())      # new tensors, the input is not changed
    for bad in (plain, dict(plain, swap_view=torch.zeros(n, dtype=torch.long)), dict(plain, swap_view=torch.zeros(n + 1, dtype=torch.bool)), dict(plain, swap_view=torch.zeros(n, 1, dtype=torch.bool))):
        with pytest.raises(ValueError, match='swap_view'):
            data.swap_views(bad)
    with pytest.raises(ValueError, match='unknown_key'):                            # as ``swap``: a key without a rule would stay unswapped
        data.swap_views(dict(down, unknown_key=torch.zeros(n)))
    with pytest.raises(ValueError, match='pos'):
        data.swap_views({k: v for k, v in down.items() if k != key('M3Size', 'open', 'pos')})
