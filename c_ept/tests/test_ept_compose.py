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
    # rate() reaches the heads through frame.m2_rate (the rate-only path since d9d7ac8; it used to call frame.m2), so the failure is injected there.
    # frame.m2 is patched too: the test then still means "a failure inside the head inference restores the modes" if the routing changes again.
    monkeypatch.setattr(h.frame, 'm2_rate', failed)
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


def test_the_ept_composer_heads_declare_the_zone_mark_of_their_model_and_the_composer_refuses_the_other(world, stats):
    """S4b preparation M9: the model object carries its own setting; ``EPTComposerHeads`` declares it and ``Composer`` compares it with its own before it binds anything."""
    from c_ept.tests.s3_toys import TOY_COMPOSE_PRICES, TOY_GOLD_REPS, toy_row_table, toy_tier_model
    torch.manual_seed(4)
    on, off = model(), heads.EPT(encoder.EncoderConfig(d=16, layers=1, heads=2, dropout=0.2), toy_static(), zone_mark=False)
    assert EPTComposerHeads(on, stats).zone_mark is True and EPTComposerHeads(off, stats).zone_mark is False
    assert EPTComposerHeads(heads.FrameAveraged(off), stats).zone_mark is False
    for model_, asked in ((on, False), (off, True)):
        h = EPTComposerHeads(model_, stats)
        with pytest.raises(ValueError, match='zone_mark'):
            compose.Composer(h, TOY_COMPOSE_PRICES, toy_tier_model(), TOY_GOLD_REPS, toy_row_table(), world.s2, S=2, seed=1, zone_mark=asked)
        assert h._enc is None                                                       # refused before the bind: nothing was encoded


def controlled_heads(stats):
    """EPT composer heads with a small constant onset rate, so that the simulation is not empty and stays bounded (as the leakage tests do)."""
    m = model(seed=17)
    with torch.no_grad():
        m.heads.rate_net.out.weight.zero_()
        m.heads.rate_net.out.bias.fill_(np.log(0.002))
        m.heads.obj_net.out.bias.fill_(-2.)
        m.heads.conv_net.out.bias.fill_(-5.)
    return EPTComposerHeads(m, stats)


def test_the_exposure_of_the_composer_is_checked_before_the_ept_heads_encode_anything(world, stats):
    from c_ept.tests.s3_toys import TOY_COMPOSE_PRICES, TOY_GOLD_REPS, toy_row_table, toy_tier_model
    h = EPTComposerHeads(model(), stats)
    with pytest.raises(ValueError, match='exposure'):
        compose.Composer(h, TOY_COMPOSE_PRICES, toy_tier_model(), TOY_GOLD_REPS, toy_row_table(), world.s2, S=2, seed=1, exposure='Nominal')
    assert h._enc is None


def test_the_ept_composer_with_the_nominal_exposure_is_the_observed_composer_of_the_same_rows_with_a_far_game_end_bit_for_bit(world, stats):
    """S4b preparation M12 (decision D14) with the real EPT composer heads: the nominal composition of the cut rows of the toy world equals the observed composition of the same rows (same t0, same frame rows, so the
    same random stream) with the game end moved far away; the observed composition of the cut rows is another one."""
    import dataclasses
    from c_ept import config as C
    from c_ept.tests.s3_toys import TOY_COMPOSE_PRICES, TOY_GOLD_REPS, toy_tier_model
    rows = compose.row_table(world.s1, world.s2, ('FIT', 'STOP', 'DEV'))
    far_T = rows.t0_ms + 100_000_000
    hi = np.minimum(rows.t0_ms + C.H_OCC_MS, far_T - C.END_CUT_MS)
    far = dataclasses.replace(rows, T_ms=far_T, exposure_ms=np.clip(hi[:, None] - (rows.t0_ms[:, None] + C.BIN_MS * np.arange(C.N_BINS)[None, :]), 0, C.BIN_MS))
    idx = list(range(0, len(rows), 3))
    assert (rows.exposure_ms[idx] < C.BIN_MS).any()                                             # the composed rows themselves include rows that the game end cuts (M12 review m4)

    def composer(table, exposure):
        return compose.Composer(controlled_heads(stats), TOY_COMPOSE_PRICES, toy_tier_model(), TOY_GOLD_REPS, table, world.s2, S=40, seed=3, exposure=exposure)
    nominal, observed_far, observed_cut = composer(rows, 'nominal'), composer(far, 'observed'), composer(rows, 'observed')
    a, b, c = nominal.simulate(idx), observed_far.simulate(idx), observed_cut.simulate(idx)
    assert (a.mean != 0).any()
    for field in ('mean', 'sd', 'q19'):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field
    assert np.array_equal(nominal.expected_onsets(idx), observed_far.expected_onsets(idx))
    assert not np.array_equal(a.mean, c.mean) and not np.array_equal(nominal.expected_onsets(idx), observed_cut.expected_onsets(idx))


# ---- S4b preparation M14 (Q6 = A, T011 A2-F6): the views are encoded in fixed chunks and the cache keeps only what the heads read ----------------------------------------------
def frame_views(world):
    return np.flatnonzero(world.s2.array('views/src') == 0)


def head_outputs(h, seed=5):
    """Every public output of the composer heads for a fixed set of conditions over all bound views (rows repeat and come in any order)."""
    rng = np.random.default_rng(seed)
    n = h._n
    rows = rng.integers(0, n, 3 * n)
    k = len(rows)
    q, m, z, s = rng.integers(0, 18, k), rng.integers(0, 52, k), rng.integers(-1, 10, k), rng.integers(0, 4, k)
    layer = rng.integers(1, 7, (k, 2))
    kills = np.column_stack([rng.integers(0, 6, (k, 2)), rng.integers(0, 12, (k, 2)) * 350.])
    cells = rng.integers(0, 3, (k, 18)).astype(float)
    X = features.m4_masked_inputs(h.row_view(rows), q, m, z, kills, cells)
    return {'rate': h.rate(rows), 'mark_prob': h.mark_prob(rows, q), 'zone_prob': h.zone_prob(rows, q, m), 'size_prob': h.size_prob(rows, q, m, z),
            'kill_rate': h.kill_rate(rows, q, m, z, s), 'gold_prob': h.gold_prob(rows, q, m, z, s, layer), 'obj_rate': h.obj_rate(rows, q, m, z, s, kills),
            'conv_rate': h.conv_rate(X)}


def bound_heads(world, stats, *, chunk=None, network=None, **kwargs):
    h = EPTComposerHeads(model() if network is None else network, stats, **kwargs) if chunk is None else EPTComposerHeads(model() if network is None else network, stats, chunk_views=chunk, **kwargs)
    h.bind(world.s2, frame_views(world))
    return h


def cache_tensors(h):
    """The tensors of the encoding cache of the heads, whichever frames and fields it holds."""
    enc = h._enc
    parts = [enc.original] + ([] if enc.swapped is None else [enc.swapped])
    return [t for part in parts for t in part] + [enc.side]


def cache_bytes(h):
    """The bytes that the cache holds in memory: the storage of every tensor once (a view of a large buffer counts as the whole buffer)."""
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in cache_tensors(h)}
    return sum(storages.values())


def test_none_chunk_is_bit_identical(world, stats):
    """chunk_views=None (the default) encodes every view in one pass as before: bit for bit the outputs of the frame average on one batch of all bound views, and the keyword None is the default."""
    views = frame_views(world)
    default, explicit = bound_heads(world, stats), bound_heads(world, stats, chunk=None)
    assert default.chunk_views is None and explicit.chunk_views is None
    frame = heads.FrameAveraged(default.model).eval()
    rows, q, m, z, s = np.arange(len(views)), np.arange(len(views)) % 18, np.arange(len(views)) % 52, np.arange(len(views)) % 11 - 1, np.arange(len(views)) % 4
    with torch.no_grad():
        enc = frame.encode(data.ViewTensors(world.s2, views, stats))
        a = frame.m2(enc, torch.ones(len(views), 18), pos=torch.tensor(rows))
        b = frame.m3(enc, *map(torch.tensor, (q, m, z)), pos=torch.tensor(rows))
        rate = frame.m2_rate(enc, torch.ones(len(views), 18), pos=torch.tensor(rows))
    assert np.array_equal(default.rate(rows), rate.numpy())
    assert np.array_equal(default.mark_prob(rows, q), a.mark_logp.exp().numpy()[rows, q])
    assert np.array_equal(default.zone_prob(rows, q, m), a.zone_logp.exp().numpy()[rows, q, m])
    assert np.array_equal(default.size_prob(rows, q, m, z), b.size_logp.exp().numpy())
    assert np.array_equal(default.kill_rate(rows, q, m, z, s), b.kill_rate.numpy()[rows, s])
    one, two = head_outputs(default), head_outputs(explicit)
    for name in one:
        assert np.array_equal(one[name], two[name]), name
    assert default._enc.original.h_tok.shape[1] > 10 and default._enc.original.mask.numel() > 0             # the cache of the single pass is the whole encoding as before


def test_fixed_chunk_is_deterministic_across_runs(world, stats):
    """The same chunk size gives bit-identical outputs in two bindings of one object and in two objects (here 25 views in chunks of 4: six full chunks and one of a single view)."""
    network = model()
    first = bound_heads(world, stats, chunk=4, network=network)
    a = head_outputs(first)
    first.bind(world.s2, frame_views(world))
    again = head_outputs(first)
    second = head_outputs(bound_heads(world, stats, chunk=4, network=network))
    for name in a:
        assert np.array_equal(a[name], again[name]), name
        assert np.array_equal(a[name], second[name]), name


def test_cache_keeps_only_head_inputs(world, stats):
    """The chunked cache holds the player states (n, 10, d), the pooled state (n, d) and the side marker of every view, nothing of the other tokens and no view of a larger buffer; the heads read only these."""
    n, d = len(frame_views(world)), 16
    full, chunked = bound_heads(world, stats), bound_heads(world, stats, chunk=4)
    assert full._enc.original.h_tok.shape == (n, 52, d)                                            # the single pass keeps every token (52 = 10 players + 34 landmarks + 7 events + 1 summary)
    kept = chunked._enc
    assert kept.original.h_tok.shape == (n, 10, d) and kept.original.h_pool.shape == (n, d) and kept.side.shape == (n,)
    assert kept.original.mask.numel() == 0 and kept.swapped.mask.numel() == 0
    exact = sum(t.numel() * t.element_size() for t in cache_tensors(chunked))
    assert cache_bytes(chunked) == exact                                                           # no tensor is a slice that keeps a whole chunk of the encoder alive
    assert cache_bytes(chunked) * 4 < cache_bytes(full)                                            # N_tok / 11 of the state (52 tokens on the toy; 173 / 11 = 15.7 at the cap of 128 events), and no mask
    # the heads read only what the cache keeps: the same outputs when the other tokens of the full encoding are destroyed
    reference = head_outputs(full)
    poisoned = bound_heads(world, stats, network=full.model)
    for part in (poisoned._enc.original, poisoned._enc.swapped):
        part.h_tok[:, 10:-1] = float('nan')                                                       # the landmark and event tokens (the last token is the summary: the pooled state is a view of it)
        part.mask.fill_(False)
    killed = head_outputs(poisoned)
    for name in reference:
        assert np.array_equal(reference[name], killed[name]), name
    # a chunk of at least the number of views is one pass: the compact cache gives the outputs of the full cache bit for bit
    whole = head_outputs(bound_heads(world, stats, chunk=1000, network=full.model))
    for name in reference:
        assert np.array_equal(reference[name], whole[name]), name


def test_chunked_outputs_within_1e_6_relative_of_one_pass(world, stats):
    """Documented bound: another chunk size moves the CPU outputs at the level of the last digits of float32. Relative 2e-6 for B in {1, 2} and 1e-6 for B >= 4. Measured on this toy model (single thread CPU, 25 views):
    B = 1 5.7e-7 (the maximum), B = 2 3.5e-7, B = 4 and 24 2.9e-7, B = 7, 13 and 25 identical; the real-scale value of plan Q6 (T011) is 1.15e-7 at B = 256. A chunk size is part of the evaluation identity, so the bound
    is documentation, not a result."""
    network = model()
    reference = head_outputs(bound_heads(world, stats, network=network))
    for chunk in (1, 2, 4, 7, 13, 24, 25, 26):
        got = head_outputs(bound_heads(world, stats, chunk=chunk, network=network))
        for name in reference:
            np.testing.assert_allclose(got[name], reference[name], rtol=2e-6 if chunk <= 2 else 1e-6, atol=0, err_msg=f'{name} chunk {chunk}')


@pytest.mark.parametrize('average', [True, False])
def test_chunks_keep_the_frame_average_of_the_model_as_one_pass_does(world, stats, average):
    """M7 inside chunks: under the average each chunk holds both frames (the swapped encoding too), under FrameAveraged(average=False) one frame and no swapped encoding; the outputs are those of the one pass."""
    network = model()
    wrapper = lambda: heads.FrameAveraged(network, average=average)
    one = bound_heads(world, stats, network=wrapper())
    chunked = bound_heads(world, stats, chunk=4, network=wrapper())
    assert (chunked._enc.swapped is None) == (not average) and (one._enc.swapped is None) == (not average)
    assert np.array_equal(chunked._enc.side.numpy(), one._enc.side.numpy())
    reference, got = head_outputs(one), head_outputs(chunked)
    for name in reference:
        np.testing.assert_allclose(got[name], reference[name], rtol=1e-6, atol=0, err_msg=name)
    if average:                                                                                      # the swapped frame of the chunks is the swapped frame of the pass
        np.testing.assert_allclose(chunked._enc.swapped.h_pool.numpy(), one._enc.swapped.h_pool.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(chunked._enc.swapped.h_tok.numpy(), one._enc.swapped.h_tok[:, :10].numpy(), rtol=1e-5, atol=1e-6)
    whole = head_outputs(bound_heads(world, stats, chunk=1000, network=wrapper()))
    for name in reference:
        assert np.array_equal(whole[name], reference[name]), name
    other = head_outputs(bound_heads(world, stats, chunk=4, network=heads.FrameAveraged(network, average=not average)))
    assert any(not np.array_equal(other[name], got[name]) for name in got)                           # the average is really applied (and really absent) inside the chunks


@pytest.mark.parametrize('zone_mark', [True, False])
def test_chunks_keep_the_zone_mark_setting_of_the_model(world, stats, zone_mark):
    """M9 inside chunks: the setting of the model survives the chunked encoding, the outputs are those of the one pass, and the Composer still compares it with its own."""
    from c_ept.tests.s3_toys import TOY_COMPOSE_PRICES, TOY_GOLD_REPS, toy_row_table, toy_tier_model
    torch.manual_seed(4)
    network = heads.EPT(encoder.EncoderConfig(d=16, layers=1, heads=2, dropout=0.2), toy_static(), zone_mark=zone_mark)
    one, chunked = bound_heads(world, stats, network=network), bound_heads(world, stats, chunk=4, network=network)
    assert chunked.zone_mark is zone_mark and one.zone_mark is zone_mark
    reference, got = head_outputs(one), head_outputs(chunked)
    for name in reference:
        np.testing.assert_allclose(got[name], reference[name], rtol=1e-6, atol=0, err_msg=name)
    fresh = EPTComposerHeads(network, stats, chunk_views=4)
    with pytest.raises(ValueError, match='zone_mark'):
        compose.Composer(fresh, TOY_COMPOSE_PRICES, toy_tier_model(), TOY_GOLD_REPS, toy_row_table(), world.s2, S=2, seed=1, zone_mark=not zone_mark)
    assert fresh._enc is None


def test_the_real_composer_with_chunked_heads_is_deterministic_and_a_single_chunk_is_the_default_bit_for_bit(world, stats):
    """The Composer on the EPT heads: chunk_views at least the number of views simulates exactly as the default, a smaller chunk is reproducible."""
    from c_ept.tests.s3_toys import TOY_COMPOSE_PRICES, TOY_GOLD_REPS, toy_tier_model
    rows = compose.row_table(world.s1, world.s2, ('FIT', 'STOP', 'DEV'))
    idx = list(range(0, len(rows), 3))

    def simulate(**kwargs):
        h = controlled_heads(stats) if not kwargs else EPTComposerHeads(controlled_heads(stats).model, stats, **kwargs)
        c = compose.Composer(h, TOY_COMPOSE_PRICES, toy_tier_model(), TOY_GOLD_REPS, rows, world.s2, S=40, seed=3)
        r = c.simulate(idx)
        return r, c.expected_onsets(idx), c.simulate_onsets(idx[0])
    base, onsets, draws = simulate()
    assert (base.mean != 0).any()
    for kwargs in ({'chunk_views': None}, {'chunk_views': 100_000}):
        r, o, d = simulate(**kwargs)
        for field in ('mean', 'sd', 'q19'):
            assert np.array_equal(getattr(r, field), getattr(base, field)), (kwargs, field)
        assert np.array_equal(o, onsets)
        for field in ('sim', 'q', 'm', 'z', 's', 'k', 'gold_bin', 'gold', 'O', 'conv', 'value'):
            assert np.array_equal(getattr(d, field), getattr(draws, field)), (kwargs, field)
    a, b = simulate(chunk_views=3), simulate(chunk_views=3)
    for field in ('mean', 'sd', 'q19'):
        assert np.array_equal(getattr(a[0], field), getattr(b[0], field)), field
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2].value, b[2].value)


@pytest.mark.parametrize('bad', [0, -1, 2.5, 4.0, True, False, '256', b'4', [4], np.float32(4), np.array([4]), float('nan')])
def test_chunk_views_must_be_a_positive_integer_or_none(world, stats, bad):
    """A chunk size that is not a positive integer (or None) is refused when the heads are made, before anything is encoded, and the message does not repeat it."""
    with pytest.raises(ValueError, match='chunk_views') as raised:
        EPTComposerHeads(model(), stats, chunk_views=bad)
    assert repr(bad) not in str(raised.value) and str(bad) not in str(raised.value)


def test_chunk_views_accepts_integers_and_numpy_integers_and_is_read_back_as_an_int(world, stats):
    for value in (1, 256, np.int64(7), np.uint8(3)):
        h = EPTComposerHeads(model(), stats, chunk_views=value)
        assert h.chunk_views == int(value) and type(h.chunk_views) is int
    assert inspect.signature(EPTComposerHeads).parameters['chunk_views'].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(EPTComposerHeads).parameters['chunk_views'].default is None


def test_a_failed_chunked_bind_keeps_the_old_cache_generation_and_modes(world, stats, monkeypatch):
    h = bound_heads(world, stats, chunk=4)
    handles, rates = h.row_view([0]), h.rate([0])
    modes = [m.training for m in h.frame.modules()]
    original, calls = h.frame.encode, []

    def failing_second_chunk(batch):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('synthetic encode failure')
        return original(batch)
    monkeypatch.setattr(h.frame, 'encode', failing_second_chunk)
    with pytest.raises(RuntimeError, match='synthetic encode failure'):
        h.bind(world.s2, frame_views(world)[::-1])
    assert len(calls) == 2
    np.testing.assert_array_equal(h.row_view([0]), handles)
    np.testing.assert_array_equal(h.rate([0]), rates)
    assert [m.training for m in h.frame.modules()] == modes


def test_a_chunked_bind_encodes_at_most_chunk_views_views_at_a_time_in_the_requested_order(world, stats, monkeypatch):
    h = EPTComposerHeads(model(), stats, chunk_views=4)
    views = frame_views(world)[[5, 0, 5, 9, 2, 24, 1, 7, 7, 3]]
    seen, original = [], h.frame.encode

    def spy(batch):
        seen.append(batch['view'].numpy().copy())
        return original(batch)
    monkeypatch.setattr(h.frame, 'encode', spy)
    h.bind(world.s2, views)
    assert [len(x) for x in seen] == [4, 4, 2] and np.array_equal(np.concatenate(seen), views)
    np.testing.assert_array_equal(h.row_view(np.arange(len(views)))[:, 0], np.arange(len(views)))


def test_a_chunk_is_released_before_the_next_chunk_is_encoded(world, stats, monkeypatch):
    """The point of the chunks is memory: when chunk c + 1 is encoded nothing of the encoder output of chunk c is alive (the cache of a chunk owns its memory and is no view of the output), and nothing of the last chunk is alive after the bind."""
    import gc
    import weakref
    h = EPTComposerHeads(model(), stats, chunk_views=4)
    original, previous, alive = h.frame.encode, [], []

    def spy(batch):
        gc.collect()
        alive.append(sum(ref() is not None for ref in previous))
        out = original(batch)
        previous[:] = [weakref.ref(t) for part in (out.original, out.swapped) if part is not None for t in (part.h_tok, part.h_pool)]
        return out
    monkeypatch.setattr(h.frame, 'encode', spy)
    h.bind(world.s2, frame_views(world))
    gc.collect()
    assert len(alive) == 7 and alive == [0] * 7
    assert sum(ref() is not None for ref in previous) == 0