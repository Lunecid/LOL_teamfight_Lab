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
from pathlib import Path

import numpy as np
import pytest

from c_ept import baselines_b0, features, head_rows
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
