"""Tests of ``c_ept.head_rows``: the rows of the baseline heads, shared by B0 and B1 (stage S3 Task 7, ruling R18).

No run folder, raw file or catalog is read. The data are the "heads world" of ``s3_toys`` (small S1, S2 and label stores with the real
links between the tables). The B0 tests (``test_baselines_b0``) already pin the whole of what B0 makes of these rows (design, target,
exposure, row id, refusals); the tests here pin what the move into a shared module adds:

    builder     the feature builder is the only thing that gives the rows their features, and it is called with the views that the rule A3
                allows (the frame-clock view of the onset's row for a masked head, the event view src 1 / src 2 of the unit for an open one)
    parts       what ``HeadParts`` holds for each head: the feature rows (not gathered: ``take``), the categories, K and O, the matrix of
                ``features.m4_masked_inputs`` for the masked M4 head; B0 and B1 read the same rows
    names       ``baselines_b0`` has the shared classes and constants, not copies
    structure   the module imports no model, no runner, no raw reader; it names only the arrays it is documented to read
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from c_ept.models import baselines_b0
from c_ept.preprocessing import features
from c_ept.data import head_rows
from c_ept.valuation import value
from c_ept.tests.conftest import all_imports, module_level_imports
from c_ept.tests.s3_toys import KEY_PREFIX, HeadsOracle, heads_world  # noqa: F401  (the fixture is imported by ruling R1)

SOURCE = Path(head_rows.__file__)
ALL_HEADS = [(name, 'masked') for name in head_rows.HEAD_NAMES] + [(name, 'open') for name in head_rows.OPEN_HEAD_NAMES]


# =====================================================================================================================================
# a recording feature builder (the plain-loop answers to "which view" are ``s3_toys.HeadsOracle``)
# =====================================================================================================================================
class Recorder:
    """A feature builder that returns, for each view, (the view number, ``views/src`` of the view, 7.0) and remembers every call."""

    def __init__(self, s2):
        self.src = np.asarray(s2.array('views/src'))
        self.calls = []

    def __call__(self, s2, views):
        views = np.asarray(views)
        self.calls.append(views.copy())
        return np.column_stack([views, self.src[views], np.full(len(views), 7)]).astype(np.float32)


@pytest.fixture(scope='module')
def oracle(heads_world):
    return HeadsOracle(heads_world)


def rows_with(heads_world, build, role='FIT'):
    return head_rows.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, role, build)


def b0_rows(heads_world, role='FIT'):
    return rows_with(heads_world, lambda s2, views: features.b0(s2, views, heads_world.fit_means), role)


def fit_dropped(heads_world):
    return features.dropped_columns(heads_world.s2, heads_world.fit_views)


# =====================================================================================================================================
# the feature builder
# =====================================================================================================================================
def test_the_rows_of_a_masked_head_have_the_features_of_the_frame_clock_view_of_the_onset_row(heads_world, oracle):
    rec = Recorder(heads_world.s2)
    p = rows_with(heads_world, rec).parts('M2Mark', 'masked')
    fit = oracle.onsets('FIT')
    expected = [oracle.view(0, oracle.on_row[i]) for i in fit]
    got = p.gather()
    assert got[:, 0].tolist() == expected and (got[:, 1] == 0).all() and (got[:, 2] == 7).all()
    assert p.row.tolist() == fit                                                    # the id of the evaluation row of an onset is the onset


def test_every_masked_head_asks_the_builder_for_frame_clock_views_only(heads_world):
    for name in head_rows.HEAD_NAMES:
        rec = Recorder(heads_world.s2)
        rows_with(heads_world, rec).parts(name, 'masked')
        assert rec.calls, name
        assert all((rec.src[v] == 0).all() for v in rec.calls), name              # never an event view: A3


def test_the_open_m3_heads_ask_for_the_src_1_views_and_the_open_m4_head_for_the_src_2_views(heads_world, oracle):
    units = oracle.units('FIT')
    for name in ('M3Size', 'M3Kills', 'M3Gold', 'M3Obj'):
        rec = Recorder(heads_world.s2)
        rows_with(heads_world, rec).parts(name, 'open')
        assert all((rec.src[v] == 1).all() for v in rec.calls), name
    rec = Recorder(heads_world.s2)
    p = rows_with(heads_world, rec).parts('M4Conv', 'open')
    assert all((rec.src[v] == 2).all() for v in rec.calls)
    with_view = [u for u in units if oracle.event_row(u, 2) is not None]
    assert p.gather()[:, 0].tolist() == [oracle.view(2, oracle.event_row(u, 2)) for u in with_view] and p.row.tolist() == with_view
    rec = Recorder(heads_world.s2)
    p = rows_with(heads_world, rec).parts('M3Size', 'open')
    assert p.gather()[:, 0].tolist() == [oracle.view(1, oracle.event_row(u, 1)) for u in units]


def test_the_builder_decides_the_width_of_the_rows(heads_world):
    rows = b0_rows(heads_world)
    assert rows.parts('M2Mark', 'masked').feat.shape[1] == features.N_B0
    dropped = fit_dropped(heads_world)
    width = len(features.b1_columns(dropped))
    b1 = rows_with(heads_world, lambda s2, views: features.b1(s2, views, dropped))
    for name, kind in ALL_HEADS:
        p = b1.parts(name, kind)
        assert p.feat.shape[1] == width and p.feat.dtype == np.float32, (name, kind)
        if (name, kind) == ('M4Conv', 'masked'):
            assert p.m4.shape[1] == width + 25                                      # the matrix of features.m4_masked_inputs is wider


def test_a_b1_builder_gives_the_b1_rows_of_the_views_of_the_rule(heads_world, oracle):
    dropped = fit_dropped(heads_world)
    rows = rows_with(heads_world, lambda s2, views: features.b1(s2, views, dropped))
    fit = oracle.onsets('FIT')
    views = [oracle.view(0, oracle.on_row[i]) for i in fit]
    assert np.array_equal(rows.parts('M2Zone', 'masked').gather(), features.b1(heads_world.s2, views, dropped), equal_nan=True)
    units = oracle.units('FIT')
    views = [oracle.view(1, oracle.event_row(u, 1)) for u in units]
    assert np.array_equal(rows.parts('M3Size', 'open').gather(), features.b1(heads_world.s2, views, dropped), equal_nan=True)


def test_the_rows_do_not_depend_on_how_the_builder_is_made(heads_world):
    """The same builder as a plain function and as a partial gives the same rows: only the matrix that comes back matters."""
    import functools
    a = b0_rows(heads_world).parts('M3Size', 'masked').gather()
    b = rows_with(heads_world, functools.partial(features.b0, fit_means=heads_world.fit_means)).parts('M3Size', 'masked').gather()
    assert np.array_equal(a, b)


# =====================================================================================================================================
# what the parts hold
# =====================================================================================================================================
def test_the_feature_rows_are_not_gathered_for_the_heads_of_onsets_and_the_take_index_picks_them(heads_world, oracle):
    p = b0_rows(heads_world).parts('M2Rate', 'masked')
    n_rows = len(np.flatnonzero(features.split_rows(heads_world.s1, 'FIT')))
    assert p.feat.shape[0] == n_rows < len(p.y) and p.take.shape == p.y.shape and p.take.max() < n_rows       # (row, bin) cells, not copies
    o = b0_rows(heads_world).parts('M3Size', 'masked')
    assert o.feat.shape[0] == n_rows and len(o.take) == len(oracle.onsets('FIT'))
    assert np.array_equal(o.gather(), o.feat[o.take])
    u = b0_rows(heads_world).parts('M3Size', 'open')
    assert u.take is None and np.array_equal(u.gather(), u.feat)                    # the open heads have one view for each unit


def test_the_categories_each_head_conditions_on(heads_world):
    rows = b0_rows(heads_world)
    names = {('M2Rate', 'masked'): {'q'}, ('M2Mark', 'masked'): {'q'}, ('M2Zone', 'masked'): {'q', 'm'},
             ('M3Size', 'masked'): {'q', 'm', 'z'}, ('M3Kills', 'masked'): {'q', 'm', 'z', 's'},
             ('M3Gold', 'masked'): {'q', 'm', 'z', 's', 'layer'},
             ('M3Obj', 'masked'): {'q', 'm', 'z', 's'}, ('M4Conv', 'masked'): {'q', 'm', 'z'},
             ('M3Size', 'open'): set(), ('M3Kills', 'open'): {'s'}, ('M3Gold', 'open'): {'s', 'layer'}, ('M3Obj', 'open'): {'s'},
             ('M4Conv', 'open'): set()}
    assert set(names) == set(ALL_HEADS)
    for key, expected in names.items():
        p = rows.parts(*key)
        assert set(p.codes) == expected, key
        assert all(a.dtype == np.int64 and a.shape == p.y.shape[:1] for a in p.codes.values()), key


def test_k_is_given_to_the_masked_object_head_on_request_only_and_never_to_the_other_masked_heads(heads_world, oracle):
    rows = b0_rows(heads_world)
    for name in head_rows.HEAD_NAMES:
        p = rows.parts(name, 'masked')
        assert p.K is None and p.O is None, name                                      # B0 reads no K and no O of a masked head (P6)
    p = rows.parts('M3Obj', 'masked', masked_obj_k=True)
    fit = oracle.onsets('FIT')
    assert p.K.shape == (len(fit), 4) and p.O is None
    for j, i in enumerate(fit):
        assert np.array_equal(p.K[j], oracle.kblock(oracle.on_unit[i])), j
    for name in ('M2Rate', 'M2Mark', 'M3Kills', 'M3Gold', 'M4Conv'):                  # the flag is for the masked object head alone
        q = rows.parts(name, 'masked', masked_obj_k=True)
        assert q.K is None, name


def test_the_open_heads_carry_the_true_k_and_the_open_m4_head_the_true_o(heads_world, oracle):
    rows = b0_rows(heads_world)
    units = oracle.units('FIT')
    for name in ('M3Size', 'M3Kills', 'M3Gold'):
        p = rows.parts(name, 'open')
        assert p.K is None and p.O is None, name
    p = rows.parts('M3Obj', 'open')
    assert p.O is None and [p.K[j].tolist() for j in range(len(units))] == [oracle.kblock(u).tolist() for u in units]
    p = rows.parts('M4Conv', 'open')
    with_view = [u for u in units if oracle.event_row(u, 2) is not None]
    assert np.array_equal(p.K, np.array([oracle.kblock(u) for u in with_view])) and np.array_equal(p.O, oracle.O[with_view])
    assert np.array_equal(p.e, oracle.cexp[with_view] / 1000.0) and p.n_zero_exposure == len(units) - len(with_view)


def test_the_masked_m4_rows_are_the_matrix_of_the_input_builder(heads_world, oracle, monkeypatch):
    """The matrix built by ``features.m4_masked_inputs`` is the m4 of the parts, and the features and categories are read back from it
    (a mark put into the matrix by a spy is in ``m4``, in ``feat`` and in the categories)."""
    real = features.m4_masked_inputs
    seen = []

    def spy(row_view, q, m, z, K, O):
        out = real(row_view, q, m, z, K, O).copy()
        d = out.shape[1] - 25
        out[:, 0] = -123.0                                                          # a mark in the first feature column
        out[:, d + 3] = -456.0                                                      # ... and in the kills of the blue team
        seen.append((len(row_view), np.asarray(q).copy(), np.asarray(K).copy()))
        return out
    monkeypatch.setattr(features, 'm4_masked_inputs', spy)
    p = b0_rows(heads_world).parts('M4Conv', 'masked')
    assert len(seen) == 1 and seen[0][0] == len(oracle.onsets('FIT'))
    d = features.N_B0
    assert p.m4.shape == (len(p.y), d + 25) and p.m4.dtype == np.float32 and p.take is None
    assert (p.m4[:, 0] == -123.0).all() and (p.m4[:, d + 3] == -456.0).all()
    assert np.array_equal(p.feat, p.m4[:, :d]) and (p.feat[:, 0] == -123.0).all()
    keep = np.flatnonzero(oracle.cexp[oracle.on_unit[oracle.onsets('FIT')]] > 0)
    assert np.array_equal(p.codes['q'], seen[0][1][keep])                           # the categories are read back from the matrix
    assert np.array_equal(p.row, np.array(oracle.onsets('FIT'))[keep])


def test_the_masked_m4_matrix_is_built_from_the_true_k_and_o_of_the_units(heads_world, oracle):
    p = b0_rows(heads_world).parts('M4Conv', 'masked')
    d = features.N_B0
    fit = np.array(oracle.onsets('FIT'))
    keep = fit[oracle.cexp[oracle.on_unit[fit]] > 0]
    for j, i in enumerate(keep):
        u = oracle.on_unit[i]
        assert np.array_equal(p.m4[j, d + 3:d + 7], oracle.kblock(u).astype(np.float32)), j
        assert np.array_equal(p.m4[j, d + 7:], oracle.O[u].astype(np.float32)), j
        assert p.codes['q'][j] == oracle.on_bin[i] and p.row[j] == i


def test_the_parts_arrays_are_read_only_where_they_belong_to_the_cache_and_fresh_otherwise(heads_world):
    rows = b0_rows(heads_world)
    first = rows.parts('M2Mark', 'masked')
    assert not first.feat.flags.writeable and not first.take.flags.writeable and not first.y.flags.writeable
    again = rows.parts('M2Mark', 'masked')
    assert again.feat is first.feat and again.y is first.y                          # the cached arrays are shared, so they must be frozen
    m4 = rows.parts('M4Conv', 'masked')
    assert m4.m4.flags.writeable                                                    # fresh: a model may change it in place
    m4.m4[...] = 0.0
    assert rows.parts('M4Conv', 'masked').m4.any()


def test_gold_entries_are_the_team_entries_with_a_kill_blue_first(heads_world, oracle):
    p = b0_rows(heads_world).parts('M3Gold', 'masked')
    fit = oracle.onsets('FIT')
    entries = [(i, t) for t in (0, 1) for i in fit if oracle.K[oracle.on_unit[i], t] >= 1]
    assert p.team.tolist() == [t for _, t in entries] and p.row.tolist() == [i for i, _ in entries]
    assert p.codes['layer'].tolist() == [min(oracle.K[oracle.on_unit[i], t], 6) for i, t in entries] and p.e is None
    assert p.gather().shape[0] == len(entries)
    o = b0_rows(heads_world).parts('M3Gold', 'open')
    units = oracle.units('FIT')
    entries = [(u, t) for t in (0, 1) for u in units if oracle.K[u, t] >= 1]
    assert o.team.tolist() == [t for _, t in entries] and o.row.tolist() == [u for u, _ in entries] and o.take.tolist() == [
        units.index(u) for u, _ in entries]


def test_the_frame_rows_of_a_role_with_their_features(heads_world, oracle):
    f = b0_rows(heads_world).frame()
    mask = features.split_rows(heads_world.s1, 'FIT')
    assert np.array_equal(f.rows, np.flatnonzero(mask)) and f.feat.shape == (mask.sum(), features.N_B0)
    assert np.array_equal(f.match, oracle.fr_match[f.rows])
    rec = Recorder(heads_world.s2)
    g = rows_with(heads_world, rec).frame()
    assert g.feat[:, 0].tolist() == [oracle.view(0, r) for r in f.rows] and (g.feat[:, 1] == 0).all()
    stop = b0_rows(heads_world, 'STOP').frame()
    assert set(stop.rows).isdisjoint(set(f.rows)) and len(stop.rows) > 0


def test_a_role_without_frame_rows_is_refused(heads_world):
    from c_ept.tests.s3_toys import build_heads_world
    import gc
    import shutil
    import tempfile
    folder = Path(tempfile.mkdtemp())
    try:
        world = build_heads_world(folder, roles=('FIT', 'FIT', 'STOP'))
        means = world.fit_means
        rows = head_rows.HeadRows(world.s1, world.s2, world.labels, 'DEV', lambda s2, views: features.b0(s2, views, means))
        with pytest.raises(ValueError, match='no frame row'):
            rows.frame()
        del rows, world
    finally:
        gc.collect()
        shutil.rmtree(folder, ignore_errors=True)


# =====================================================================================================================================
# the match-subset path (matches=): the labels of the matches that are not selected are never read, and the rows that stand in for
# them hold a sentinel, not a valid value (EPT-T010 A1-F7: a 0 is a valid gold bin and a valid kill gold)
# =====================================================================================================================================
SUBSET = [1, 3]                  # two of the four FIT matches of the heads world


def subset_rows(heads_world, matches=SUBSET):
    return head_rows.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, 'FIT',
                              lambda s2, views: features.b0(s2, views, heads_world.fit_means), matches=matches)


def assert_same_on_selected(full, sub, mask, what):
    for name in ('y', 'e', 'match', 'row', 'team', 'K', 'O', 'm4'):
        a, b = getattr(full, name), getattr(sub, name)
        if a is None or b is None:
            assert a is None and b is None, (what, name)
        else:
            assert np.array_equal(np.asarray(a)[mask], np.asarray(b), equal_nan=True), (what, name)
    assert full.codes.keys() == sub.codes.keys(), what
    for name in full.codes:
        assert np.array_equal(full.codes[name][mask], sub.codes[name]), (what, 'codes', name)
    assert np.array_equal(full.gather()[mask], sub.gather(), equal_nan=True), (what, 'features')


@pytest.mark.parametrize('head, kind', ALL_HEADS)
def test_the_subset_path_gives_the_rows_of_the_full_path_on_the_selected_matches(heads_world, head, kind):
    full, sub = b0_rows(heads_world), subset_rows(heads_world)
    for masked_obj_k in ((False, True) if (head, kind) == ('M3Obj', 'masked') else (False,)):
        pf, ps = full.parts(head, kind, masked_obj_k=masked_obj_k), sub.parts(head, kind, masked_obj_k=masked_obj_k)
        mask = np.isin(pf.match, SUBSET)
        assert mask.any() and not mask.all() and len(ps.match) == mask.sum(), (head, kind)       # a proper, nonempty subset
        assert_same_on_selected(pf, ps, mask, (head, kind, masked_obj_k))


def test_the_unselected_label_rows_of_the_subset_path_hold_a_sentinel_and_the_selected_ones_are_the_full_values(heads_world):
    full, sub = b0_rows(heads_world), subset_rows(heads_world)
    chosen = np.isin(np.asarray(heads_world.s1.array('units/match')), SUBSET)
    assert chosen.any() and not chosen.all()
    gold_full, gold_sub = full._labels('labels/kill_gold'), sub._labels('labels/kill_gold')
    bin_full, bin_sub = full._labels('labels/gold_bin'), sub._labels('labels/gold_bin')
    assert gold_sub.dtype == gold_full.dtype and bin_sub.dtype == bin_full.dtype and gold_sub.shape == gold_full.shape
    assert np.isnan(gold_sub[~chosen]).all()                       # 0.0 is a valid kill gold (a team without a kill)
    assert (bin_sub[~chosen] == -1).all()                          # 0 is a valid gold bin
    assert np.array_equal(gold_sub[chosen], gold_full[chosen], equal_nan=True) and np.array_equal(bin_sub[chosen], bin_full[chosen])


def test_reading_the_kill_gold_of_an_unselected_unit_is_refused_instead_of_giving_zeros(heads_world):
    sub = subset_rows(heads_world)
    chosen = np.isin(np.asarray(heads_world.s1.array('units/match')), SUBSET)
    with pytest.raises(ValueError, match='not finite'):
        sub._kraw(np.flatnonzero(~chosen)[:3])
    assert np.isfinite(sub._kraw(np.flatnonzero(chosen)[:3])).all()


@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_the_sentinel_of_a_signed_integer_label_array_is_minus_one(dtype):
    assert head_rows._unselected_label_fill(np.dtype(dtype)) == -1


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_the_sentinel_of_a_floating_label_array_is_nan(dtype):
    assert np.isnan(head_rows._unselected_label_fill(np.dtype(dtype)))


@pytest.mark.parametrize('dtype, largest', [(np.uint8, 255), (np.uint16, 65535), (np.uint32, 2**32 - 1)])
def test_the_sentinel_of_an_unsigned_label_array_is_its_largest_value(dtype, largest):
    assert head_rows._unselected_label_fill(np.dtype(dtype)) == largest


def test_a_boolean_label_array_has_no_free_value_and_is_refused_in_the_subset_path():
    with pytest.raises(TypeError, match='no value'):
        head_rows._unselected_label_fill(np.dtype(bool))


# =====================================================================================================================================
# refusals
# =====================================================================================================================================
class NoRead:
    """A store that fails the test when an array is read: a bad role is refused before the first read."""

    def array(self, name):
        raise AssertionError(f'an array was read before the role was checked: {name}')


def test_a_holdout_role_and_a_role_s3_does_not_use_are_refused_before_any_array_is_read(heads_world):
    for role in ('TEST', 'EXTERNAL', 'CAL', 'AUDIT', 'UNUSED', 'NOPE'):
        with pytest.raises(ValueError):
            head_rows.HeadRows(NoRead(), heads_world.s2, heads_world.labels, role, lambda s2, views: None)


def test_a_head_and_a_kind_that_do_not_exist_are_refused(heads_world):
    rows = b0_rows(heads_world)
    for head, kind in (('M2Rate', 'open'), ('M2Mark', 'open'), ('M5', 'masked'), ('M3Size', 'both'), ('m3size', 'masked')):
        with pytest.raises(ValueError, match='there is no'):
            rows.parts(head, kind)


def test_no_message_of_the_shared_rows_holds_a_match_key(heads_world):
    rows = b0_rows(heads_world)
    s1, s2, labels = heads_world.s1, heads_world.s2, heads_world.labels
    for call in (lambda: rows.parts('M5', 'masked'), lambda: head_rows.HeadRows(s1, s2, labels, 'TEST', None)):
        with pytest.raises(ValueError) as raised:
            call()
        assert KEY_PREFIX not in str(raised.value) and KEY_PREFIX not in repr(raised.value)


# =====================================================================================================================================
# the names that B0 shares
# =====================================================================================================================================
def test_baselines_b0_has_the_shared_classes_and_constants_not_copies():
    for name in ('HeadData', 'HeadScores', 'HeadBatch', 'HEAD_NAMES', 'OPEN_HEAD_NAMES', 'PREDICT_ALL_WIDTHS', 'TEAMS', 'MIX', 'KINDS',
                 'N_Q', 'N_M', 'N_SIZE', 'N_LAYER', 'N_ZONE', 'N_GOLD', 'N_CELLS', 'N_K'):
        assert getattr(baselines_b0, name) is getattr(head_rows, name), name
    assert issubclass(baselines_b0.HeadRows, head_rows.HeadRows) and baselines_b0.HeadRows is not head_rows.HeadRows


def test_the_shared_constants_are_the_values_of_the_spec():
    assert head_rows.HEAD_NAMES == ('M2Rate', 'M2Mark', 'M2Zone', 'M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')
    assert head_rows.OPEN_HEAD_NAMES == ('M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')
    assert head_rows.KINDS == ('masked', 'open') and head_rows.TEAMS == ('blue', 'red') and head_rows.MIX == 1e-6
    assert (head_rows.N_Q, head_rows.N_M, head_rows.N_SIZE, head_rows.N_LAYER, head_rows.N_ZONE, head_rows.N_GOLD, head_rows.N_CELLS,
            head_rows.N_K) == (18, 52, 4, 6, 10, 31, 18, 4)
    assert head_rows.PREDICT_ALL_WIDTHS == {'rate': 18, 'mark_prob': 52, 'zone_prob': 10, 'size_prob': 4, 'kill_rate': 2, 'gold_prob': 62,
                                            'obj_rate': 18, 'conv_rate': 18}
    assert sum(head_rows.PREDICT_ALL_WIDTHS.values()) == 184
    assert head_rows.VALID_HEADS == set(ALL_HEADS) and len(head_rows.VALID_HEADS) == 13


def test_the_b0_data_is_the_one_hot_design_of_the_shared_parts(heads_world):
    """B0's design of the rows is a function of the parts alone: the numeric columns of the parts, then the one-hot of the categories."""
    rows = baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, 'FIT')
    d = rows.data('M3Size', 'masked')
    p = rows.parts('M3Size', 'masked')
    assert np.array_equal(d.X[:, :features.N_B0], p.gather().astype(np.float64)) and np.array_equal(d.y, p.y)
    assert np.array_equal(d.match, p.match) and np.array_equal(d.row, p.row) and d.e is p.e is None
    z_block = d.X[:, features.N_B0 + 18 + 52:features.N_B0 + 18 + 52 + 11]
    assert np.array_equal(z_block.argmax(axis=1) - 1, p.codes['z'])


# =====================================================================================================================================
# the K condition (S4b preparation M3, decision D21 B16): under 'binned' the gold of the K block is the representative of the observed gold
# bin, the value the composer feeds to the K-conditioned heads, and not the true continuous gold
# =====================================================================================================================================
class LabelOverlay:
    """The label store of the heads world with some arrays replaced."""

    def __init__(self, base, arrays):
        self.base, self.arrays = base, arrays

    def array(self, name):
        return self.arrays[name] if name in self.arrays else self.base.array(name)


def k_rows(world, k_condition=None, role='FIT', matches=None, labels=None):
    """The rows of the heads world with the b0 builder; ``k_condition`` None leaves the keyword out (the default of the code)."""
    kw = {} if k_condition is None else {'k_condition': k_condition}
    if matches is not None:
        kw['matches'] = matches
    return head_rows.HeadRows(world.s1, world.s2, world.labels if labels is None else labels, role, lambda s2, views: features.b0(s2, views, world.fit_means), **kw)


def composer_k(oracle, reps, u) -> np.ndarray:
    """kills blue, kills red, gold blue and red in thousands of the unit u as the K of the composer (``compose._draw_row``: gold = reps[bin] for a team with a kill, 0 for a team
    without one), written from the text of the composer and not from the rows under test."""
    gold = [reps[oracle.gold_bin[u, t]] if oracle.K[u, t] >= 1 else 0.0 for t in (0, 1)]
    return np.array([oracle.K[u, 0], oracle.K[u, 1], gold[0] / 1000.0, gold[1] / 1000.0])


K_HEADS = [('M3Obj', 'masked'), ('M4Conv', 'masked'), ('M3Obj', 'open'), ('M4Conv', 'open')]       # the heads that read the K block (the masked M4 head in its matrix)


def test_binned_k_condition_matches_composer_support(heads_world, oracle):
    """Every K amount of a binned row is 0 or the representative of a gold bin, the same 31 numbers that the composer feeds to the K-conditioned heads, and the K block of every
    row is the composer's K of that unit (a team with a kill: reps[its bin] in thousands; a team without one: 0)."""
    from c_ept.tests.test_compose import rich_composer
    binned, raw = k_rows(heads_world, 'binned'), k_rows(heads_world)
    reps = binned.k_reps()
    # the one source of the representatives: the in-scope FIT means of the label store (not the labels/gold_reps of the S3 as-built, which are the means of every FIT unit)
    in_scope = value.fit_gold_bin_means(heads_world.s1, heads_world.labels, population='in_scope').means
    every_unit = value.fit_gold_bin_means(heads_world.s1, heads_world.labels, population='all').means
    assert np.array_equal(reps, in_scope) and reps.shape == (31,) and reps.dtype == np.float64 and not reps.flags.writeable
    assert not np.array_equal(in_scope, every_unit)                          # the control: the world has FIT units out of scope whose gold moves the default means
    fit, units = oracle.onsets('FIT'), oracle.units('FIT')
    d = features.N_B0
    masked_obj = binned.parts('M3Obj', 'masked', masked_obj_k=True)
    assert [masked_obj.K[j].tolist() for j in range(len(fit))] == [composer_k(oracle, reps, oracle.on_unit[i]).tolist() for i in fit]
    open_obj = binned.parts('M3Obj', 'open')
    assert [open_obj.K[j].tolist() for j in range(len(units))] == [composer_k(oracle, reps, u).tolist() for u in units]
    with_view = [u for u in units if oracle.event_row(u, 2) is not None]
    open_conv = binned.parts('M4Conv', 'open')
    assert [open_conv.K[j].tolist() for j in range(len(with_view))] == [composer_k(oracle, reps, u).tolist() for u in with_view]
    masked_conv = binned.parts('M4Conv', 'masked')
    kept = [i for i in fit if oracle.cexp[oracle.on_unit[i]] > 0]
    assert len(kept) == len(masked_conv.y)
    for j, i in enumerate(kept):
        assert np.array_equal(masked_conv.m4[j, d + 3:d + 7], composer_k(oracle, reps, oracle.on_unit[i]).astype(np.float32)), j
    # the support: every gold of a K block is 0 or the representative of a bin, in thousands as the heads see it
    support = np.concatenate([[0.0], reps]) / 1000.0
    for block in (masked_obj.K, open_obj.K, open_conv.K):
        assert np.isin(block[:, 2:], support).all()
    assert np.isin(masked_conv.m4[:, d + 5:d + 7], support.astype(np.float32)).all()
    assert (open_obj.K[:, 2:] > 0).any() and len(np.unique(open_obj.K[:, 2:][open_obj.K[:, 2:] > 0])) > 3       # many bins occur, not one
    assert not np.isin(raw.parts('M3Obj', 'open').K[:, 2:], support).all()     # the control: the true continuous gold of the raw rows leaves the support
    # the composer holding the same array draws golds of the same support (and no other): rows and composer share one source of representatives
    c = rich_composer(S=60, seed=5, gold_reps=reps)
    drawn = np.concatenate([c.simulate_onsets(i).gold.ravel() for i in range(len(c.rows))])
    assert np.array_equal(c.gold_reps, reps) and np.isin(drawn, np.concatenate([[0.0], reps])).all()
    assert (drawn > 0).sum() > 20 and len(np.unique(drawn[drawn > 0])) > 3
    other = rich_composer(S=60, seed=5, gold_reps=every_unit)
    assert not np.isin(np.concatenate([other.simulate_onsets(i).gold.ravel() for i in range(len(other.rows))]), np.concatenate([[0.0], reps])).all()   # the S3 labels leave it


def test_the_default_k_condition_gives_the_true_gold_rows_of_the_oracle_bit_for_bit_and_asks_for_no_representatives(heads_world, oracle, monkeypatch):
    """What this proves: the rows of the default keyword, and of the explicit 'raw', are equal to each other field by field, and the K blocks are the K of the labels as the oracle
    reads them (the true continuous gold in thousands), and no representative is asked for. It cannot compare with the code before the task (the before/after proof script does)."""
    def refuse(*args, **kw):
        raise AssertionError('the representatives were asked for under the raw condition')
    monkeypatch.setattr(value, 'fit_gold_bin_means', refuse)
    default, explicit = k_rows(heads_world), k_rows(heads_world, 'raw')
    assert default.k_condition == explicit.k_condition == 'raw' and default.k_reps() is None and explicit.k_reps() is None
    for head, kind in ALL_HEADS:
        x, y = default.parts(head, kind, masked_obj_k=True), explicit.parts(head, kind, masked_obj_k=True)
        for name in head_rows.HeadParts._fields:
            u, v = getattr(x, name), getattr(y, name)
            if isinstance(u, dict):
                assert u.keys() == v.keys() and all(np.array_equal(u[k], v[k]) for k in u), (head, kind, name)
            elif isinstance(u, np.ndarray):
                assert np.array_equal(u, v, equal_nan=True) and u.dtype == v.dtype, (head, kind, name)
            else:
                assert u == v, (head, kind, name)
    fit, units = oracle.onsets('FIT'), oracle.units('FIT')
    masked_obj = default.parts('M3Obj', 'masked', masked_obj_k=True)
    assert all(np.array_equal(masked_obj.K[j], oracle.kblock(oracle.on_unit[i])) for j, i in enumerate(fit))
    open_obj = default.parts('M3Obj', 'open')
    assert all(np.array_equal(open_obj.K[j], oracle.kblock(u)) for j, u in enumerate(units))


def test_the_binned_condition_changes_the_k_inputs_of_four_heads_and_nothing_else(heads_world):
    """The targets, the exposures, the categories, the features and the gold bins of every head are the raw rows': the gold layer and the targets do not change."""
    raw, binned = k_rows(heads_world), k_rows(heads_world, 'binned')
    changed = []
    for head, kind in ALL_HEADS:
        a, b = raw.parts(head, kind, masked_obj_k=True), binned.parts(head, kind, masked_obj_k=True)
        for name in head_rows.HeadParts._fields:
            u, v = getattr(a, name), getattr(b, name)
            same = (all(np.array_equal(u[k], v[k]) for k in u) and u.keys() == v.keys() if isinstance(u, dict)
                    else np.array_equal(u, v, equal_nan=True) if isinstance(u, np.ndarray) else u == v)
            if not same:
                changed.append((head, kind, name))
    assert changed == [('M3Obj', 'masked', 'K'), ('M4Conv', 'masked', 'm4'), ('M3Obj', 'open', 'K'), ('M4Conv', 'open', 'K')]
    pa, pb = raw.parts('M4Conv', 'masked'), binned.parts('M4Conv', 'masked')
    d = features.N_B0
    assert np.array_equal(pa.m4[:, :d + 5], pb.m4[:, :d + 5]) and np.array_equal(pa.m4[:, d + 7:], pb.m4[:, d + 7:])          # features, q, m, z, both kill counts and O
    assert not np.array_equal(pa.m4[:, d + 5:d + 7], pb.m4[:, d + 5:d + 7])                                                  # the gold columns are the only difference


def test_the_representatives_are_those_of_the_whole_fit_whatever_the_role_and_the_selected_matches(heads_world, oracle):
    """D12: the definition of the target variable is a whole-FIT asset. A DEV or STOP row set, and the 25 % style selection of FIT matches, hold the same 31 numbers."""
    reference = k_rows(heads_world, 'binned').k_reps()
    for kw in ({'role': 'STOP'}, {'role': 'DEV'}, {'matches': SUBSET}):
        assert np.array_equal(k_rows(heads_world, 'binned', **kw).k_reps(), reference), kw
    # ... and the rows of the selection are the rows of the full path on the selected matches, with these representatives
    full, sub = k_rows(heads_world, 'binned'), k_rows(heads_world, 'binned', matches=SUBSET)
    for head, kind in K_HEADS:
        pf, ps = full.parts(head, kind), sub.parts(head, kind)
        mask = np.isin(pf.match, SUBSET)
        assert mask.any() and not mask.all()
        assert_same_on_selected(pf, ps, mask, (head, kind))
    # the gold of the K block of a STOP row is still read from the FIT representatives (not from the means of the STOP units)
    stop = k_rows(heads_world, 'binned', role='STOP').parts('M3Obj', 'open')
    assert np.isin(stop.K[:, 2:], np.concatenate([[0.0], reference]) / 1000.0).all()


def test_the_k_condition_is_one_of_two_names_and_binned_needs_the_label_store_before_any_array_is_read():
    assert head_rows.K_CONDITIONS == ('raw', 'binned')
    assert [head_rows.check_k_condition(name) for name in head_rows.K_CONDITIONS] == ['raw', 'binned']
    for bad in ('Binned', 'bin', 'continuous', '', None, 1, ('raw',), b'raw'):
        with pytest.raises(ValueError, match='k_condition'):
            head_rows.check_k_condition(bad)
        with pytest.raises(ValueError, match='k_condition'):
            head_rows.HeadRows(NoRead(), NoRead(), NoRead(), 'FIT', None, k_condition=bad)
    with pytest.raises(ValueError, match='label'):                              # the binned K reads labels/gold_bin and the representatives: no label store, no rows
        head_rows.HeadRows(NoRead(), NoRead(), None, 'FIT', None, k_condition='binned')
    parameters = inspect.signature(head_rows.HeadRows.__init__).parameters       # the keyword comes after kill_law and is keyword only: no positional call can change
    names = list(parameters)                                                      # relative order: a later task may add keywords after these (review m3)
    assert names.index('kill_law') == names.index('matches') + 1 and names.index('k_condition') == names.index('kill_law') + 1
    assert parameters['k_condition'].default == 'raw' and parameters['k_condition'].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_binned_rows_name_their_fallback_bins_and_the_sha256_of_their_representatives(heads_world):
    """Review m1 and m4: the bins of the representatives that had no in-scope FIT entry (they hold the middle of their range) are named, as the S3 labels named theirs, and the 31
    numbers have an identity (sha256 of the float64 little-endian bytes, ``value.gold_reps_sha256``). Under 'raw' there are none of them."""
    raw, binned = k_rows(heads_world), k_rows(heads_world, 'binned')
    result = value.fit_gold_bin_means(heads_world.s1, heads_world.labels, population='in_scope')
    fallback = binned.k_reps_fallback()
    assert fallback == result.fallback and isinstance(fallback, tuple) and list(fallback) == sorted(fallback) and 0 < len(fallback) < 31       # the toy world has bins without entry
    assert all(binned.k_reps()[b] == value.GOLD_FALLBACK[b] for b in fallback)                                 # a fallback bin holds the middle of its range
    sha = binned.k_reps_sha256()
    assert sha == value.gold_reps_sha256(binned.k_reps()) and len(sha) == 64 and sha == k_rows(heads_world, 'binned', role='DEV').k_reps_sha256()
    assert raw.k_reps_fallback() is None and raw.k_reps_sha256() is None


def test_the_binned_rows_refuse_a_store_without_any_in_scope_fit_entry(heads_world):
    """Review m1: all 31 representatives would be midpoints (a wrong or role-scoped store); the rows say so instead of training on made-up numbers."""
    n_units = len(np.asarray(heads_world.s1.array('units/match')))
    nothing = LabelOverlay(heads_world.s1, {'units/in_scope': np.zeros(n_units, dtype=bool)})
    rows = head_rows.HeadRows(nothing, heads_world.s2, heads_world.labels, 'FIT', lambda s2, views: features.b0(s2, views, heads_world.fit_means), k_condition='binned')
    with pytest.raises(ValueError, match='no team entry with a kill in any gold bin') as raised:
        rows.k_reps()
    assert KEY_PREFIX not in str(raised.value)


def test_a_team_with_a_kill_and_no_gold_bin_is_refused_under_the_binned_condition(heads_world, oracle):
    """A team with a kill must have a gold bin (the same rule as the gold head), else its K amount would be a made-up 0; a team without a kill is 0 by the rule."""
    unit = next(u for u in oracle.units('FIT') if oracle.K[u, 0] >= 1)
    bins = np.array(oracle.gold_bin)
    bins[unit, 0] = -1
    broken = k_rows(heads_world, 'binned', labels=LabelOverlay(heads_world.labels, {'labels/gold_bin': bins}))
    with pytest.raises(ValueError, match=r'1 team\(s\) with a kill have no gold bin') as raised:
        broken.parts('M3Obj', 'open')
    assert KEY_PREFIX not in str(raised.value)
    k_rows(heads_world, 'binned').parts('M3Obj', 'open')                          # the intact labels pass
    # the units of the matches that a selection leaves out hold -1 in the bin array: asking for the K of one of them is refused, not given a 0
    sub = k_rows(heads_world, 'binned', matches=SUBSET)
    chosen = np.isin(np.asarray(heads_world.s1.array('units/match')), SUBSET)
    other = [u for u in range(len(chosen)) if not chosen[u] and oracle.K[u, 0] >= 1][:3]
    assert other
    with pytest.raises(ValueError, match='no gold bin'):
        sub._kraw(np.array(other))


# =====================================================================================================================================
# structure
# =====================================================================================================================================
def test_the_module_imports_no_model_no_runner_no_cohort_no_raw_reader_no_value_at_module_level():
    imported = module_level_imports(SOURCE)
    forbidden = {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.cohort', 'c_ept.runlog', 'c_ept.terrain', 'c_ept.store_build', 'c_ept.reader',
                 'c_ept.tokens', 'c_ept.value', 'c_ept.compose', 'c_ept.units', 'c_ept.targets', 'c_ept.towertier', 'c_ept.baselines_b0',
                 'c_ept.baselines_b1', 'c_ept.heads', 'c_ept.store_core', 'lightgbm', 'sklearn'}
    assert not imported & forbidden, imported & forbidden
    assert not all_imports(SOURCE) & {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.reader', 'c_ept.baselines_b0', 'c_ept.baselines_b1', 'pickle'}


def test_the_module_names_only_the_arrays_it_is_documented_to_read():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.count('/') == 1 and ' ' not in node.value and node.value.split('/')[0].isidentifier()}
    documented = {'matches/role', 'frame_rows/match', 'frame_rows/t0', 'frame_rows/exposure_ms', 'onsets/row', 'onsets/bin', 'onsets/unit',
                  'onsets/mark', 'units/match', 'units/mark', 'units/size', 'units/K', 'units/O', 'units/conv', 'units/conv_exposure_ms',
                  'units/in_scope', 'event_rows/unit', 'event_rows/kind', 'event_rows/ref', 'views/src', 'views/src_index', 'views/match',
                  'views/ref', 'labels/gold_bin', 'labels/kill_gold'}
    assert names <= documented, names - documented
    assert {'views/src', 'views/src_index', 'onsets/row', 'units/K', 'labels/gold_bin'} <= names          # the scan looks at the row code
    text = SOURCE.read_text(encoding='utf-8')
    assert 'matches/key' not in text and 'allow_holdout' not in text


def test_the_module_does_not_read_the_kills_or_the_objectives_itself():
    text = SOURCE.read_text(encoding='utf-8')
    for forbidden in ("'kills/", "'objectives/", "'frames/", "'units/f'", "'units/delta'", "'units/last'", "'units/kill0'", 'pickle'):
        assert forbidden not in text, forbidden                          # the zone comes from features.unit_zone


def test_the_module_calls_the_input_builder_through_the_features_module():
    """``features.m4_masked_inputs`` is looked up at the call, so that the leakage spy of a test sees every use (leakage test (a))."""
    text = SOURCE.read_text(encoding='utf-8')
    assert 'features.m4_masked_inputs(' in text and 'import m4_masked_inputs' not in text
