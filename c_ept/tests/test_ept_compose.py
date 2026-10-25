"""Task6 synthetic protocol, units, frame-only binding and stale-handle regressions."""
import inspect
import numpy as np
import pytest
import torch
from c_ept.valuation import compose
from c_ept.preprocessing import features
from c_ept.models.heads import ComposerHeads
from c_ept.ept import data, encoder, heads
from c_ept.ept.compose_heads import EPTComposerHeads, MAX_HANDLE
from c_ept.tests.ept_toys import toy_static


def build_compose_world(root):
    """Complete EPT tokens on the value toy, including objective history for real Composer."""
    from types import SimpleNamespace
    from c_ept.preprocessing import tokens
    from c_ept.data.store_core import Store
    from c_ept.tests.s3_toys import build_value_s1, build_toy_s2, _put
    from c_ept.tests.ept_toys import random_x, DTYPES
    s1 = build_value_s1(root / 's1')
    base = build_toy_s2(root / 'view_metadata', s1)
    src = np.asarray(base.array('views/src'))
    arrays = random_x(np.random.default_rng(19), src, features.split_views(s1, base, 'FIT'), 7, 5)
    s2 = Store(root / 's2')
    for name in ('src', 'src_index', 'match', 'ref', 't0', 'frame_k'):
        a = np.asarray(base.array('views/' + name))
        _put(s2, 'views/' + name, a, a.dtype)
    for name in tokens.VIEW_SPEC:
        _put(s2, 'x/' + name, arrays[name], DTYPES[name])
    s2.finalize()
    return SimpleNamespace(s1=s1, s2=s2)


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    return build_compose_world(tmp_path_factory.mktemp('ept_composer'))


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def stats(world):
    return data.fit_stats(world.s1, world.s2)


def model(seed=4):
    torch.manual_seed(seed)
    return heads.EPT(encoder.EncoderConfig(d=16, layers=1, heads=2, dropout=0.2), toy_static())


@pytest.fixture
def bound(world, stats):
    h = EPTComposerHeads(model(), stats)
    views = np.flatnonzero(world.s2.array('views/src') == 0)[:3]
    h.bind(world.s2, views)
    return h, views


def cond():
    return np.array([2, 0]), np.array([1, 7]), np.array([5, 50]), np.array([3, -1]), np.array([0, 3])


def test_ept_heads_follow_the_composer_protocol(bound, world, stats):
    h, views = bound
    assert isinstance(h, ComposerHeads)
    assert (h.feature_kind, h.analytic_ok, h.obj_uses_K, h.conv_uses_KO) == ('ept', False, True, True)
    rows, q, m, z, s = cond()
    frame = heads.FrameAveraged(h.model).eval()
    with torch.no_grad():
        enc = frame.encode(data.ViewTensors(world.s2, views, stats))
        a = frame.m2(enc, torch.ones(2, 18), pos=torch.tensor(rows))
        b = frame.m3(enc, *map(torch.tensor, (q, m, z)), pos=torch.tensor(rows))
    np.testing.assert_allclose(h.rate(rows), a.rate.numpy(), rtol=1e-6)
    np.testing.assert_allclose(h.mark_prob(rows, q), a.mark_logp.exp().numpy()[np.arange(2), q], rtol=1e-6)
    np.testing.assert_allclose(h.zone_prob(rows, q, m), a.zone_logp.exp().numpy()[np.arange(2), q, m], rtol=1e-6)
    np.testing.assert_allclose(h.size_prob(rows, q, m, z), b.size_logp.exp().numpy(), rtol=1e-6)
    np.testing.assert_allclose(h.kill_rate(rows, q, m, z, s), b.kill_rate.numpy()[np.arange(2), s], rtol=1e-6)
    layers = np.array([[1, 6], [5, 2]])
    expected = np.stack([[b.gold_logp.exp().numpy()[i, s[i], t, layers[i, t] - 1] for t in (0, 1)] for i in range(2)])
    np.testing.assert_allclose(h.gold_prob(rows, q, m, z, s, layers), expected, rtol=1e-6)
    for law in (h.mark_prob(rows, q), h.zone_prob(rows, q, m), h.size_prob(rows, q, m, z), h.gold_prob(rows, q, m, z, s, layers)):
        assert (law > 0).all()
        np.testing.assert_allclose(law.sum(-1), 1., atol=2e-7)


def test_obj_and_conversion_gold_are_scaled_exactly_once(bound, monkeypatch):
    h, _ = bound
    captured = []
    obj, conv = heads.FrameAveraged.m3_obj, heads.FrameAveraged.m4
    def spy_obj(self, enc, q, m, z, s, K, pos=None):
        captured.append(('obj', K.detach().cpu().numpy().copy(), None))
        return obj(self, enc, q, m, z, s, K, pos=pos)
    def spy_conv(self, enc, q, m, z, K, O, exposure, pos=None):
        captured.append(('conv', K.detach().cpu().numpy().copy(), exposure.detach().cpu().numpy().copy()))
        return conv(self, enc, q, m, z, K, O, exposure, pos=pos)
    monkeypatch.setattr(heads.FrameAveraged, 'm3_obj', spy_obj)
    monkeypatch.setattr(heads.FrameAveraged, 'm4', spy_conv)
    raw = np.array([[2., 5., 1500., 3250.]])
    saved = raw.copy()
    assert h.obj_rate([0], [2], [3], [4], [1], raw).shape == (1, 18)
    X = features.m4_masked_inputs(h.row_view([0]), np.array([2]), np.array([3]), np.array([4]), raw, np.arange(18)[None])
    saved_X = X.copy()
    assert h.conv_rate(X).shape == (1, 18)
    assert [x[0] for x in captured] == ['obj', 'conv']
    for _, K, _ in captured:
        np.testing.assert_array_equal(K, [[2., 5., 1.5, 3.25]])
    np.testing.assert_array_equal(captured[1][2], np.ones(1))
    np.testing.assert_array_equal(raw, saved)
    np.testing.assert_array_equal(X, saved_X)


def test_modes_gradients_and_encoding_cache_are_preserved(world, stats, monkeypatch):
    m = model()
    m.train()
    m.heads.eval()  # mixed mode must survive
    flags = [x.training for x in m.modules()]
    calls = []
    original = m.encode
    def spy(batch):
        calls.append((torch.is_grad_enabled(), m.training))
        return original(batch)
    monkeypatch.setattr(m, 'encode', spy)
    h = EPTComposerHeads(m, stats)
    h.bind(world.s2, np.flatnonzero(world.s2.array('views/src') == 0)[:2])
    first = h.rate([0])
    np.testing.assert_array_equal(h.rate([0]), first)
    assert calls == [(False, False), (False, False)]
    assert [x.training for x in m.modules()] == flags
    assert all(p.grad is None for p in m.parameters())
    def failed(*args, **kwargs):
        raise RuntimeError('synthetic inference failure')
    monkeypatch.setattr(h.frame, 'm2', failed)
    with pytest.raises(RuntimeError, match='synthetic inference'):
        h.rate([0])
    assert [x.training for x in m.modules()] == flags


def test_bind_refuses_event_views_before_payload_or_encoding(bound, world):
    h, _ = bound
    old = h.row_view([0])
    class MetadataOnly:
        def array(self, name):
            assert name == 'views/src', 'feature payload was read before refusal'
            return world.s2.array(name)
    event = np.flatnonzero(world.s2.array('views/src') != 0)[:1]
    with pytest.raises(ValueError, match='frame-clock'):
        h.bind(MetadataOnly(), event)
    np.testing.assert_array_equal(h.row_view([0]), old)


@pytest.mark.parametrize('bad', [np.array([[0]]), np.empty((0, 2), dtype=int), np.array([0.5]), np.array([np.nan]), np.array([-1]), np.array([2**64-1], dtype=np.uint64), np.array([True])])
def test_bind_rejects_bad_selections_without_replacing_cache(bound, world, bad):
    h, _ = bound
    old = h.row_view([0])
    with pytest.raises((ValueError, TypeError)):
        h.bind(world.s2, bad)
    np.testing.assert_array_equal(h.row_view([0]), old)


def test_bool_views_duplicates_and_reordering(bound, world):
    h, views = bound
    mask = np.zeros(len(world.s2.array('views/src')), dtype=bool)
    mask[views] = True
    h.bind(world.s2, mask)
    base = h.rate([0, 1, 2])
    h.bind(world.s2, views[[2, 0, 2]])
    np.testing.assert_allclose(h.rate([0, 1, 2]), base[[2, 0, 2]], rtol=1e-6)
    np.testing.assert_array_equal(h.row_view([2, 0, 2])[:, 0], [2, 0, 2])


def test_rebinding_invalidates_composer_and_stale_m4_input(bound, world):
    from c_ept.tests.test_leakage import make_composer
    h, _ = bound
    comp = make_composer(world.s1, world.s2, h)
    old = h.row_view([0])
    X = features.m4_masked_inputs(old, np.array([0]), np.array([0]), np.array([0]), np.zeros((1, 4)), np.zeros((1, 18)))
    h.bind(world.s2, comp.rows.views[::-1])
    assert h.row_view([0])[0, 0] == old[0, 0]
    with pytest.raises(RuntimeError, match='bound again'):
        comp.simulate([0])
    with pytest.raises(ValueError, match='stale'):
        h.conv_rate(X)


@pytest.mark.parametrize('value', [np.nan, 0.5, -1., 999999.])
def test_m4_rejects_invalid_row_handles(bound, value):
    h, _ = bound
    X = features.m4_masked_inputs(h.row_view([0]), np.array([0]), np.array([0]), np.array([0]), np.zeros((1, 4)), np.zeros((1, 18)))
    X[0, 0] = value
    with pytest.raises(ValueError, match='handle|row'):
        h.conv_rate(X)


def test_generation_never_silently_loses_float32_precision(bound, world):
    h, views = bound
    h._generation = MAX_HANDLE - 1
    h.bind(world.s2, views)
    assert h.row_view([0])[0, 1] == MAX_HANDLE
    with pytest.raises(ValueError, match='generation'):
        h.bind(world.s2, views)
    assert h.row_view([0])[0, 1] == MAX_HANDLE


@pytest.mark.parametrize('rows', [[True], [0.2], [np.nan], [-1], [3], np.array([[0]])])
def test_row_indices_are_strict(bound, rows):
    with pytest.raises((ValueError, TypeError)):
        bound[0].rate(rows)


@pytest.mark.parametrize('method,args', [('mark_prob', ([18],)), ('zone_prob', ([0], [52])), ('size_prob', ([0], [0], [-2])), ('kill_rate', ([0], [0], [0], [4])), ('gold_prob', ([0], [0], [0], [0], [[0, 7]]))])
def test_conditions_are_checked_before_narrowing(bound, method, args):
    with pytest.raises((ValueError, TypeError)):
        getattr(bound[0], method)([0], *args)


def test_empty_bind_and_calls(world, stats):
    h = EPTComposerHeads(model(), stats)
    h.bind(world.s2, np.empty(0, dtype=np.int64))
    assert h.row_view([]).shape == (0, 2)
    assert h.rate([]).shape == (0, 18)
    assert h.mark_prob([], []).shape == (0, 52)
    assert h.zone_prob([], [], []).shape == (0, 10)
    assert h.size_prob([], [], [], []).shape == (0, 4)
    assert h.kill_rate([], [], [], [], []).shape == (0, 2)
    assert h.gold_prob([], [], [], [], [], np.empty((0, 2), dtype=int)).shape == (0, 2, 31)
    assert h.obj_rate([], [], [], [], [], np.empty((0, 4))).shape == (0, 18)
    assert h.conv_rate(np.empty((0, 27))).shape == (0, 18)


def test_dead_time_slot_accepts_only_none_in_s4a(bound, world, monkeypatch):
    from c_ept.tests.test_leakage import PRICES, TOY_GOLD_REPS, state_tier_model
    h, _ = bound
    rows = compose.row_table(world.s1, world.s2, ('FIT', 'STOP', 'DEV'))
    assert inspect.signature(compose.Composer).parameters['dead_time'].kind is inspect.Parameter.KEYWORD_ONLY
    comp = compose.Composer(h, PRICES, state_tier_model(), TOY_GOLD_REPS, rows, world.s2, S=2, dead_time=None)
    def forbidden(*args):
        raise AssertionError('bind called on refused dead_time')
    monkeypatch.setattr(h, 'bind', forbidden)
    for value in (False, 0, 1., {}, 'off'):
        with pytest.raises(ValueError, match='dead_time'):
            compose.Composer(h, PRICES, state_tier_model(), TOY_GOLD_REPS, rows, world.s2, S=2, dead_time=value)
    with pytest.raises(ValueError, match='closed|analytic'):
        comp.expect([0])


def test_raw_gold_conversion_matches_generic_masked_input_rounding(bound, monkeypatch):
    h, _ = bound
    captured = []
    original = heads.FrameAveraged.m3_obj
    def spy(self, enc, q, m, z, s, K, pos=None):
        captured.append(K.detach().cpu().float().numpy().copy())
        return original(self, enc, q, m, z, s, K, pos=pos)
    monkeypatch.setattr(heads.FrameAveraged, 'm3_obj', spy)
    raw = np.array([[2., 5., 16777217., 33554435.]])
    X = features.m4_masked_inputs(h.row_view([0]), np.array([0]), np.array([0]), np.array([0]), raw, np.zeros((1, 18)))
    h.obj_rate([0], [0], [0], [0], [0], raw)
    np.testing.assert_array_equal(captured[0], X[:, 5:9])


def test_encoding_failure_preserves_cache_generation_and_modes(bound, world, monkeypatch):
    h, views = bound
    handles, rates = h.row_view([0]), h.rate([0])
    modes = [m.training for m in h.frame.modules()]
    def fail(batch):
        raise RuntimeError('synthetic encode failure')
    monkeypatch.setattr(h.frame, 'encode', fail)
    with pytest.raises(RuntimeError, match='synthetic encode failure'):
        h.bind(world.s2, views[::-1])
    np.testing.assert_array_equal(h.row_view([0]), handles)
    np.testing.assert_array_equal(h.rate([0]), rates)
    assert [m.training for m in h.frame.modules()] == modes
