"""Synthetic M2 component checks; no encoder forward, stores, training or benchmarks."""
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from c_ept.ept import heads
from c_ept.ept.compose_heads import EPTComposerHeads
from c_ept.ept.encoder import Encoded, EncoderConfig


@pytest.fixture(scope='module', autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def encoded(seed=7, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(3, 45, dtype=torch.bool)
    mask[:, -3:] = False
    tok = (torch.randn(3, 45, 16, generator=g, dtype=dtype) * mask[..., None]).requires_grad_()
    pool = torch.randn(3, 16, generator=g, dtype=dtype).requires_grad_()
    return Encoded(tok, pool, mask)


def small_heads(dtype=torch.float32):
    torch.manual_seed(13)
    return heads.EPTHeads(EncoderConfig(d=16, layers=1, heads=2)).to(dtype=dtype)


def legacy_m2(hd, enc, exposure, pos=None):
    """Frozen pre-split M2 operations, including their shapes and evaluation order."""
    h, players, team = heads._read(enc, pos)
    e = heads._exposure(exposure, h.shape[0], heads.N_Q, h)
    emb_q = hd.bin_emb.weight[:heads.N_Q]
    net = hd.rate_net
    eta = net.finish(net.project(0, h)[:, None] + net.project(1, emb_q)[None]).squeeze(-1)
    emb_m = hd.mark.every(players, team)
    net = hd.mark_net
    score = net.finish(net.project(0, emb_m)[:, None] + net.project(1, emb_q)[None, :, None]).squeeze(-1)
    net = hd.zone_net
    zone = net.finish(net.project(0, emb_m)[:, None] + net.project(1, emb_q)[None, :, None] + net.project(2, h)[:, None, None])
    return heads.M2Out(torch.exp(eta) * e, heads.mixed_log_softmax(score), heads.mixed_log_softmax(zone))


def gradients(outputs, inputs):
    # Different weights per cell exercise the entire output, including shared embeddings.
    loss = sum((t * torch.linspace(-0.7, 0.9, t.numel(), dtype=t.dtype).reshape(t.shape)).sum() for t in outputs)
    return torch.autograd.grad(loss, inputs, allow_unused=True)


def assert_same_gradients(a, b):
    for x, y in zip(a, b):
        assert (x is None) == (y is None)
        if x is not None:
            assert torch.equal(x, y)
            assert torch.isfinite(x).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('training', [False, True])
def test_public_m2_retains_all_outputs_gradients_and_registration(dtype, training):
    hd, enc = small_heads(dtype).train(training), encoded(dtype=dtype)
    pos = torch.tensor([2, 0, 2])
    e = torch.linspace(0, 3, 54, dtype=dtype).reshape(3, 18).requires_grad_()
    params = list(hd.parameters())
    names = [(name, id(p)) for name, p in hd.named_parameters()]
    state = deepcopy(hd.state_dict())
    inputs = [enc.h_pool, enc.h_tok, e, *params]
    expected, actual = legacy_m2(hd, enc, e, pos), hd.m2(enc, e, pos)
    assert type(actual) is heads.M2Out
    for a, b in zip(expected, actual):
        assert torch.equal(a, b)
    assert_same_gradients(gradients(expected, inputs), gradients(actual, inputs))
    hd.m2_rate(enc, e, pos)
    assert [(name, id(p)) for name, p in hd.named_parameters()] == names
    assert list(hd.state_dict()) == list(state)
    assert all(torch.equal(hd.state_dict()[k], v) for k, v in state.items())
    hd.load_state_dict(state, strict=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('exposure_kind', ['scalar', 'rows', 'bins'])
def test_rate_matches_baseline_formula_and_gradients(dtype, training, exposure_kind):
    hd, enc = small_heads(dtype).train(training), encoded(dtype=dtype)
    pos = torch.tensor([2, 0, 2])
    value = {'scalar': 0.7, 'rows': [0., 1.5, 0.4],
             'bins': torch.arange(54).reshape(3, 18) % 5}[exposure_kind]
    e = torch.as_tensor(value, dtype=dtype).requires_grad_()
    inputs = [enc.h_pool, enc.h_tok, e, *hd.parameters()]
    expected, actual = legacy_m2(hd, enc, e, pos).rate, hd.m2_rate(enc, e, pos)
    assert torch.equal(actual, expected)
    assert_same_gradients(gradients([expected], inputs), gradients([actual], inputs))
    h = enc.h_pool.index_select(0, pos)
    net = hd.rate_net
    eta = net.out(F.gelu(net.maps[0](h)[:, None] + net.maps[1](hd.bin_emb.weight[:18])[None] + net.bias)).squeeze(-1)
    assert torch.equal(actual, eta.exp() * (e[:, None] if e.ndim == 1 else e))


def forbid_laws(hd, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('an unused mark/zone branch was invoked')
    monkeypatch.setattr(hd.mark, 'every', forbidden)
    monkeypatch.setattr(hd.mark_net, 'finish', forbidden)
    monkeypatch.setattr(hd.zone_net, 'finish', forbidden)


def test_eval_skips_laws_and_zero_exposure_has_zero_state_and_parameter_gradient(monkeypatch):
    hd, enc = small_heads().eval(), encoded()
    forbid_laws(hd, monkeypatch)
    state = torch.get_rng_state().clone()
    actual = hd.m2_rate(enc, 0.0)
    assert torch.equal(actual, torch.zeros(3, 18))
    assert torch.equal(torch.get_rng_state(), state)
    grads = torch.autograd.grad(actual.sum(), [enc.h_pool, enc.h_tok, *hd.parameters()], allow_unused=True)
    assert grads[1] is None
    assert all(g is None or torch.count_nonzero(g) == 0 for g in grads)


class HeadModel(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.heads = hd

    def encode(self, batch):
        raise AssertionError('component tests must not run an encoder')


def frame_encoded(dtype=torch.float32):
    return heads.FrameEncoded(encoded(7, dtype), encoded(19, dtype), torch.tensor([False, True, False]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_frame_rate_is_exactly_symmetric_and_has_full_path_gradients(dtype):
    frame = heads.FrameAveraged(HeadModel(small_heads(dtype))).eval()
    enc = frame_encoded(dtype)
    pos, e = torch.tensor([2, 0, 2, 1]), torch.tensor([0., 1., 2., 0.5], dtype=dtype)
    inputs = [enc.original.h_pool, enc.original.h_tok, enc.swapped.h_pool, enc.swapped.h_tok, *frame.parameters()]
    expected, actual = frame.m2(enc, e, pos).rate, frame.m2_rate(enc, e, pos)
    assert torch.equal(expected, actual)
    assert_same_gradients(gradients([expected], inputs), gradients([actual], inputs))
    reverse = heads.FrameEncoded(enc.swapped, enc.original, ~enc.side)
    assert torch.equal(actual, frame.m2_rate(reverse, e, pos))


@pytest.mark.parametrize('framed', [False, True])
def test_train_preserves_discarded_branch_calls_dropout_rng_and_gradients(framed):
    hd = small_heads()
    model = heads.FrameAveraged(HeadModel(hd)) if framed else hd
    model.train()
    enc = frame_encoded() if framed else encoded()
    calls = []

    def stochastic_hook(name):
        def hook(module, args, output):
            calls.append(name)
            return F.dropout(output, p=0.4, training=module.training)
        return hook

    handles = [getattr(hd, name).out.register_forward_hook(stochastic_hook(name))
               for name in ('rate_net', 'mark_net', 'zone_net')]
    try:
        state = torch.get_rng_state().clone()
        expected = model.m2(enc, 1.0).rate
        expected_rng, expected_calls = torch.get_rng_state().clone(), list(calls)
        expected_next = torch.rand(5)
        expected_grad = gradients([expected], list(model.parameters()))
        calls.clear()
        torch.set_rng_state(state)
        actual = model.m2_rate(enc, 1.0)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert torch.equal(torch.rand(5), expected_next)
        assert calls == expected_calls == ['rate_net', 'mark_net', 'zone_net'] * (2 if framed else 1)
        assert_same_gradients(expected_grad, gradients([actual], list(model.parameters())))
    finally:
        for handle in handles:
            handle.remove()


class LegacyHeads:
    """The supported plain-object heads protocol has m2 and no mode or m2_rate."""
    def __init__(self):
        self.calls = []

    def m2(self, enc, exposure, pos=None):
        self.calls.append((enc, exposure, pos))
        h, _, _ = heads._read(enc, pos)
        rate = h[:, :1].exp().expand(-1, 18) * heads._exposure(exposure, len(h), 18, h)
        return heads.M2Out(rate, h.new_full((len(h), 18, 52), -4.), h.new_full((len(h), 18, 52, 10), -3.))


def test_frame_supports_legacy_plain_object_heads_without_rate_method():
    hd, enc = LegacyHeads(), frame_encoded()
    frame = heads.FrameAveraged(HeadModel(hd)).eval()
    pos, e = torch.tensor([2, 0, 2]), torch.ones(3, 18)
    expected = frame.m2(enc, e, pos).rate
    hd.calls.clear()
    assert torch.equal(frame.m2_rate(enc, e, pos), expected)
    assert len(hd.calls) == 2
    assert hd.calls[0][0] is enc.original and hd.calls[1][0] is enc.swapped
    assert all(call[1] is e and call[2] is pos for call in hd.calls)


@pytest.mark.parametrize('framed', [False, True])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('pos', [None, torch.tensor([2, 0, 2]), torch.empty(0, dtype=torch.int64)])
def test_pos_masks_and_empty_rows_retain_baseline(framed, training, pos):
    hd = small_heads()
    model = heads.FrameAveraged(HeadModel(hd)) if framed else hd
    model.train(training)
    enc = frame_encoded() if framed else encoded()
    n = 3 if pos is None else len(pos)
    e = torch.ones(n, 18)
    assert torch.equal(model.m2(enc, e, pos).rate, model.m2_rate(enc, e, pos))
    assert model.m2_rate(enc, e, pos).shape == (n, 18)


@pytest.mark.parametrize('framed', [False, True])
@pytest.mark.parametrize('bad_pos', [torch.tensor([3]), torch.tensor([-1]), torch.tensor([0.]),
                                    torch.tensor([True]), torch.tensor([0], dtype=torch.uint8),
                                    torch.zeros(1, 1, dtype=torch.int64), [0]])
def test_bad_pos_retains_full_path_validation(framed, bad_pos):
    hd = small_heads()
    model = heads.FrameAveraged(HeadModel(hd)) if framed else hd
    model.eval()
    enc = frame_encoded() if framed else encoded()
    with pytest.raises(ValueError) as baseline:
        model.m2(enc, 1.0, bad_pos)
    with pytest.raises(ValueError) as component:
        model.m2_rate(enc, 1.0, bad_pos)
    assert str(component.value) == str(baseline.value)


@pytest.mark.parametrize('framed', [False, True])
@pytest.mark.parametrize('bad_e', [-1., float('nan'), float('inf'), torch.ones(2),
                                  torch.ones(3, 17), torch.ones(2, 18)])
def test_bad_exposure_retains_full_path_validation(framed, bad_e):
    hd = small_heads()
    model = heads.FrameAveraged(HeadModel(hd)) if framed else hd
    model.eval()
    enc = frame_encoded() if framed else encoded()
    with pytest.raises(ValueError) as baseline:
        model.m2(enc, bad_e)
    with pytest.raises(ValueError) as component:
        model.m2_rate(enc, bad_e)
    assert str(component.value) == str(baseline.value)


class HeadOnlyEPT(heads.EPT):
    """Only the EPT interface needed by the adapter, with no encoder allocation."""
    def __init__(self, hd):
        nn.Module.__init__(self)
        self.heads = hd

    def encode(self, batch):
        raise AssertionError('component tests must not run an encoder')


def test_composer_rate_skips_laws_preserves_modes_rng_and_empty_protocol(monkeypatch):
    hd, enc = small_heads(), frame_encoded()
    adapter = EPTComposerHeads(HeadOnlyEPT(hd), stats=None)
    adapter._enc, adapter._n = enc, 3       # synthetic bound cache; no stores or view tensors
    rows = np.array([2, 0, 2])
    with adapter._evaluation():
        expected = adapter._m2(rows).rate.cpu().numpy()
    hd.train()
    hd.mark_net.eval()                    # restore heterogeneous modes exactly
    modes = [(m, m.training) for m in adapter.frame.modules()]
    forbid_laws(hd, monkeypatch)
    state = torch.get_rng_state().clone()
    actual = adapter.rate(rows)
    assert np.array_equal(expected, actual)
    assert actual.dtype == np.float32 and actual.shape == (3, 18)
    assert all(m.training == mode for m, mode in modes)
    assert torch.equal(torch.get_rng_state(), state)
    assert adapter.rate(np.array([], dtype=np.int64)).shape == (0, 18)
