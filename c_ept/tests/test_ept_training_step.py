"""실제 모듈을 가짜 torch/loss와 로드한다. CPU/GPU tensor 연산은 없다 (마지막 시험만 예외로, 장난감 세계의 한 학습 단계를 실제 torch로 CPU에서 돌린다)."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('finite', [True, False])
def test_batch_order_finiteness_and_component_boundaries(monkeypatch, finite):
    import c_ept.ept
    seen = []
    total = SimpleNamespace(backward=lambda: seen.append('backward'))
    def total_loss(values, scales, masks):
        assert values == {'block': 'scores'} and masks == {'block': 'mask'} and scales == {'block': 2.}
        seen.append('loss')
        return total
    fake_loss = SimpleNamespace(total_loss=total_loss)
    fake_torch = SimpleNamespace(isfinite=lambda value: finite if value is total else None)
    path = Path(__file__).parents[1] / 'ept/training_step.py'
    spec = importlib.util.spec_from_file_location('_synthetic_ept_training_step', path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'torch', fake_torch)
        patch.setattr(c_ept.ept, 'loss', fake_loss, raising=False)
        spec.loader.exec_module(module)
    def encode(batch):
        assert batch == 'batch'
        seen.append('encode')
        return 'encoded'
    def score(fa, encoded, batch):
        assert encoded == 'encoded' and batch == 'batch'
        seen.append('score')
        return {'block': (None, 'mask', 'scores')}
    ticks = iter([1., 3., 4., 7., 8., 12.])
    components = dict(transfer_seconds=9., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
    args = (SimpleNamespace(encode=encode), 'batch', SimpleNamespace(step=lambda: seen.append('optimizer')), {'block': 2.}, 17, components)
    if finite:
        assert module.optimize_batch(*args, score=score, clock=lambda: next(ticks)) is total
        assert seen == ['encode', 'score', 'loss', 'backward', 'optimizer']
        assert components == dict(transfer_seconds=9., forward_seconds=2., backward_seconds=3., optimizer_seconds=4.)
    else:
        with pytest.raises(FloatingPointError, match='step 17'):
            module.optimize_batch(*args, score=score, clock=lambda: next(ticks))
        assert seen == ['encode', 'score', 'loss']
        assert components == dict(transfer_seconds=9., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)


@pytest.mark.parametrize('api', ['rate_only', 'legacy', 'noncallable', 'refused'])
def test_score_blocks_rate_api_and_legacy_fallback_without_tensor_work(api):
    import ast
    # 실제 score_blocks 정의와 합성 객체만 실행한다.
    tree = ast.parse((Path(__file__).parents[1] / 'ept/train.py').read_text(encoding='utf-8'))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'score_blocks')
    batch = {'y': [1], 'e': 'synthetic exposure', 'pos': 'synthetic positions'}
    seen = []
    def full(enc, exposure, pos=None):
        seen.append(('full', enc, exposure, pos))
        return SimpleNamespace(rate='synthetic rate')
    def rate(enc, exposure, pos=None):
        seen.append(('rate', enc, exposure, pos))
        if api == 'refused':
            raise ValueError('synthetic rate refusal')
        return 'synthetic rate'
    fa = SimpleNamespace(m2=full)
    if api != 'legacy':
        fa.m2_rate = 0 if api == 'noncallable' else rate
    fields = [('M2Rate', 'masked'), ('M2Mark', 'masked'), ('M2Zone', 'masked')]
    ns = {'ept_data': SimpleNamespace(ROW_FIELDS=fields, block=lambda raw, head, kind: batch if head == 'M2Rate' else {'y': []}),
          'loss': SimpleNamespace(block_key=lambda h, k: h + '/' + k, head_nll=lambda h, value, target: value),
          'step_guard': SimpleNamespace(trusted=lambda: False)}          # an untrusted step (S4b speed-ups item 3): the lists of the plan are not read
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<score_blocks pure test>', 'exec'), ns)
    if api == 'refused':
        with pytest.raises(ValueError, match='rate refusal'):
            ns['score_blocks'](fa, 'encoded', {})
    else:
        assert ns['score_blocks'](fa, 'encoded', {}) == {'M2Rate/masked': ('M2Rate', batch, 'synthetic rate')}
    assert seen == [('rate' if api in ('rate_only', 'refused') else 'full', 'encoded', batch['e'], batch['pos'])]


@pytest.mark.parametrize('was_training', [True, False])
@pytest.mark.parametrize('interrupted', [True, False])
def test_stop_enters_rate_only_wrapper_eval_and_restores_original_mode(was_training, interrupted):
    import ast
    import contextlib
    tree = ast.parse((Path(__file__).parents[1] / 'ept/train.py').read_text(encoding='utf-8'))
    # evaluate_stop delegates its body to _stop_meter (S4b-prep M9), so both definitions are compiled together
    fns = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('evaluate_stop', '_stop_meter')]
    assert sorted(node.name for node in fns) == ['_stop_meter', 'evaluate_stop']
    model = SimpleNamespace(training=was_training)
    model.eval = lambda: setattr(model, 'training', False)
    model.train = lambda mode: setattr(model, 'training', mode)
    seen = []
    class Frame:
        def __init__(self, inner):
            assert inner is model and not inner.training
            self.training = True
        def eval(self):
            self.training = False
            seen.append('wrapper_eval')
            return self
    def epoch(index):
        assert index == 0 and not model.training and seen == ['wrapper_eval']
        if interrupted:
            raise ValueError('synthetic STOP interruption')
        return iter([])
    ns = {'Mapping': dict, 'torch': SimpleNamespace(device=lambda x: x, no_grad=contextlib.nullcontext),
          'heads': SimpleNamespace(FrameAveraged=Frame), 'StopMeter': lambda: SimpleNamespace(value=lambda scales: 17.)}
    exec(compile(ast.Module(body=fns, type_ignores=[]), '<STOP mode regression>', 'exec'), ns)
    if interrupted:
        with pytest.raises(ValueError, match='STOP interruption'):
            ns['evaluate_stop'](model, SimpleNamespace(epoch=epoch), {})
    else:
        assert ns['evaluate_stop'](model, SimpleNamespace(epoch=epoch), {}) == 17.
    assert model.training == was_training and seen == ['wrapper_eval']


# ---- EPT-T011 final cleanup, group 4 review m5: one whole training step, the old rate path against the rate-only path -----------------------
# The tests of ``test_ept_m2_components`` compare the rate alone. Here the rate sits where it sits in a run: next to the shared ``fa.m2`` call of the mark
# and zone blocks, next to the M3 and M4 heads on the same encoding, in ``total_loss`` and behind ``backward``. The old path is the former ``m2_rate`` of
# ``FrameAveraged`` in training mode, which ran the full ``m2`` and kept its ``.rate`` (rebuilt here from ``m2(...).rate``). This is the only test of this
# file that uses real torch tensors: the toy EPT world of ``ept_toys`` (d = 16, two layers, dropout 0.1) on the CPU with one thread; no real store, no GPU.
def test_one_training_step_gives_the_same_loss_and_gradients_through_the_old_full_m2_rate_path_and_the_rate_only_path(tmp_path, monkeypatch):
    import gc

    import torch

    from c_ept.ept import data, encoder, heads, loss, train, training_step
    from c_ept.tests.ept_toys import build_ept_world, toy_static

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        world = build_ept_world(tmp_path, n_events=7, n_edges=5)
        stats = data.fit_stats(world.s1, world.s2)
        rows = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT'))
        batch = next(iter(data.Batches(rows, len(rows.units), 7, stats=stats, training=True).epoch(0)))
        scales = {key: 1.0 + 0.25 * i for i, key in enumerate(loss.BLOCKS)}            # unequal on purpose: every block must reach the total with its own scale
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2), toy_static())
        model.train()
        fa = heads.FrameAveraged(model)
        full_calls = []                                                               # the calls of the full FrameAveraged.m2, shared mark/zone call included
        real_m2 = heads.FrameAveraged.m2
        monkeypatch.setattr(heads.FrameAveraged, 'm2', lambda self, enc, exposure, pos=None: (full_calls.append(None), real_m2(self, enc, exposure, pos))[1])
        rate_only = heads.FrameAveraged.m2_rate

        def step():
            """One ``training_step.optimize_batch`` with a no-op optimizer, from the same seed: the dropout of the encoder draws the same numbers."""
            model.zero_grad(set_to_none=True)
            full_calls.clear()
            components = dict(transfer_seconds=0., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(11)
                total = training_step.optimize_batch(fa, batch, SimpleNamespace(step=lambda: None), scales, 0, components, score=train.score_blocks, clock=lambda: 0.)
            return total.detach().clone(), {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}, len(full_calls)

        monkeypatch.setattr(heads.FrameAveraged, 'm2_rate', lambda self, enc, exposure, pos=None: self.m2(enc, exposure, pos).rate)          # the old path
        old_loss, old_grads, old_calls = step()
        monkeypatch.setattr(heads.FrameAveraged, 'm2_rate', rate_only)                                                                         # the rate-only path
        new_loss, new_grads, new_calls = step()

        assert old_calls == new_calls + 1                                             # the paths differ: the old rate calls the full m2 once more, the new one not at all
        assert torch.isfinite(old_loss) and torch.equal(old_loss, new_loss)
        assert list(old_grads) == list(new_grads)
        for name in old_grads:
            assert (old_grads[name] is None) == (new_grads[name] is None), name
            if old_grads[name] is not None:
                assert torch.isfinite(new_grads[name]).all() and torch.equal(old_grads[name], new_grads[name]), name
        reached = {name for name, g in new_grads.items() if g is not None}
        assert {'rate_net.maps.0.weight', 'bin_emb.weight'} <= {name.removeprefix('heads.') for name in reached}             # the rate head and the shared parts of the step
        assert any(g is not None and bool(g.abs().sum() > 0) for g in new_grads.values())
        del world, rows, batch, model, fa
        gc.collect()                                                                  # Windows keeps a folder with open memory maps; pytest rotates its own temporary folders
    finally:
        torch.set_num_threads(threads)


# ---- EPT-T012 (S4b preparation) M6, decision D8: the terrain smoothness weight lambda_Omega in the training loss -------------------------------
# A tiny stand-in of ``FrameAveraged`` (a Module with a real ``TerrainField`` and one weight) and a loss that depends on both, so the gradient of the field is the sum of the gradient of
# the data loss and lambda * the gradient of R_Omega. The field is the 75 x 75 grid: channel 0 = the column number ix, channel 1 = 2 * the row number iy, every cell walkable.
# R_Omega by hand: the 75 * 74 = 5,550 horizontal pairs differ by 1 in channel 0 (squared 1), the 5,550 vertical pairs by 2 in channel 1 (squared 4), N_adj = 11,100, so
# R_Omega = 5,550 * (1 + 4) / 11,100 = 2.5.
def _smooth_world(monkeypatch, *, field_channels=2, with_field=True):
    import torch

    from c_ept.ept import encoder, loss, training_step

    class Fa(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([3.0]))
            if with_field:
                self.field = encoder.TerrainField(field_channels)
                if field_channels == 2:
                    with torch.no_grad():
                        ix, iy = torch.meshgrid(torch.arange(75.0), torch.arange(75.0), indexing='xy')
                        self.field.learned[..., 0], self.field.learned[..., 1] = ix, 2.0 * iy

        def encode(self, batch):
            return 'encoded'

    fa = Fa()

    def score(fa_, encoded, batch):
        nll = 2.0 * fa_.w.sum() + (0.5 * fa_.field.learned.sum() if with_field else 0.0)           # the data loss: depends on the weight and on every cell of the field
        return {'block': (None, 'mask', nll)}

    monkeypatch.setattr(loss, 'total_loss', lambda nlls, scales, targets: nlls['block'])
    return fa, score, training_step


def _step(fa, score, training_step, **kwargs):
    components = dict(transfer_seconds=0., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
    total = training_step.optimize_batch(fa, 'batch', type('Opt', (), {'step': lambda self: None})(), {'block': 1.0}, 0, components, score=score, clock=lambda: 0., **kwargs)
    return total, components


def test_smooth_weight_adds_weight_times_penalty(monkeypatch):
    import torch

    fa, score, training_step = _smooth_world(monkeypatch)
    assert fa.field.penalty().item() == 2.5                                       # the hand-computed R_Omega of the ramp field
    reference = torch.autograd.grad(fa.field.penalty(), fa.field.learned)[0]      # the gradient of R_Omega alone
    total, components = _step(fa, score, training_step, smooth_weight=0.5)        # lambda = 0.5, a power of two: the scaling is exact
    data_value = 2.0 * 3.0 + 0.5 * fa.field.learned.sum().item()
    assert total.item() == data_value                                             # what is returned is the data loss, without the penalty (the fit curve of every lambda is comparable)
    assert components['terrain_penalty'] == 2.5                                   # the unweighted R_Omega of this step, summed over the steps of the epoch by the caller
    assert torch.equal(fa.w.grad, torch.tensor([2.0]))                            # the weight is not in the penalty
    assert torch.equal(fa.field.learned.grad, 0.5 + 0.5 * reference)              # d(total + lambda R)/d learned = 0.5 + lambda * dR/d learned
    assert bool((reference.abs() > 0).any())                                      # the penalty does have a gradient on this field (the control of the test)
    _, other = _step(*_reset(fa, score, training_step), smooth_weight=2.0)
    assert other['terrain_penalty'] == 2.5


def _reset(fa, score, training_step):
    fa.zero_grad(set_to_none=True)
    return fa, score, training_step


def test_zero_weight_is_bit_identical(monkeypatch):
    import torch

    fa, score, training_step = _smooth_world(monkeypatch)
    monkeypatch.setattr(type(fa.field), 'penalty', lambda self: (_ for _ in ()).throw(AssertionError('the penalty must not be computed under a weight of 0')))
    default_total, default_components = _step(fa, score, training_step)
    default_grads = {n: p.grad.clone() for n, p in fa.named_parameters()}
    fa.zero_grad(set_to_none=True)
    zero_total, zero_components = _step(fa, score, training_step, smooth_weight=0.0)
    for name, p in fa.named_parameters():
        assert torch.equal(p.grad, default_grads[name]), name
    assert torch.equal(default_total, zero_total) and default_total.dtype == zero_total.dtype
    assert torch.equal(default_grads['field.learned'], torch.full_like(fa.field.learned, 0.5))      # only the data loss reaches the field
    assert 'terrain_penalty' not in default_components and default_components == zero_components   # the components of a default step are the four times of every earlier run


def test_the_smooth_weight_is_the_last_keyword_only_argument_with_the_default_zero():
    import inspect

    from c_ept.ept import training_step

    parameter = inspect.signature(training_step.optimize_batch).parameters['smooth_weight']
    assert parameter.default == 0.0 and parameter.kind is inspect.Parameter.KEYWORD_ONLY and list(inspect.signature(training_step.optimize_batch).parameters)[-1] == 'smooth_weight'


@pytest.mark.parametrize('kind', ['no_field', 'zero_channels', 'n2'])
def test_no_field_means_no_penalty(monkeypatch, kind):
    """A model without a learned field (no field module, ``field_channels=0``, or the team-set N2) gets a penalty of 0 and no error: the gradient is that of the data loss."""
    import torch

    from c_ept.ept import encoder
    from c_ept.tests.ept_toys import toy_static

    if kind == 'n2':
        from c_ept.models.neural import TeamSetModel
        fa, score, training_step = _smooth_world(monkeypatch, with_field=False)
        torch.manual_seed(5)
        fa.n2 = TeamSetModel(encoder.EncoderConfig(d=16, layers=1, heads=2, field_channels=0), toy_static(), hidden=8)      # N2 has a field (the static channels) but no learned channel
        assert sum(m.learned.numel() for m in fa.n2.modules() if isinstance(m, encoder.TerrainField)) == 0
    else:
        fa, score, training_step = _smooth_world(monkeypatch, field_channels=0, with_field=kind == 'zero_channels')
    plain_total, _ = _step(fa, score, training_step)
    plain = {n: p.grad.clone() for n, p in fa.named_parameters() if p.grad is not None}
    fa.zero_grad(set_to_none=True)
    total, components = _step(fa, score, training_step, smooth_weight=0.5)
    assert torch.equal(total, plain_total) and components.get('terrain_penalty', 0.0) == 0.0
    for name, p in fa.named_parameters():
        if p.numel() == 0:
            continue                                                              # a field without learned channels: an empty tensor, which may carry an empty gradient
        assert (p.grad is None) == (name not in plain) and (p.grad is None or torch.equal(p.grad, plain[name])), name


def test_the_field_of_the_team_set_model_is_smoothed_like_the_one_of_the_ept(monkeypatch):
    """N2 (``TeamSetModel``) owns a ``TerrainField`` with the same penalty (``team_set.py``: "exposes the existing TerrainField (including its penalty)"), so the weight reaches it."""
    import torch

    from c_ept.ept import encoder
    from c_ept.models.neural import TeamSetModel
    from c_ept.tests.ept_toys import toy_static

    fa, score, training_step = _smooth_world(monkeypatch, with_field=False)
    torch.manual_seed(5)
    fa.n2 = TeamSetModel(encoder.EncoderConfig(d=16, layers=1, heads=2, field_channels=2), toy_static(), hidden=8)
    with torch.no_grad():
        fa.n2.encoder.field.learned[3, 4, 0] = 1.0                                # one cell differs from its (walkable) neighbours
    expected = fa.n2.encoder.field.penalty().item()
    assert expected > 0
    _, components = _step(fa, score, training_step, smooth_weight=0.5)
    assert components['terrain_penalty'] == expected
    assert bool((fa.n2.encoder.field.learned.grad != 0).any())


def test_recording_the_penalty_raises_no_warning_and_keeps_the_value_and_the_graph_of_the_loss_term(monkeypatch):
    """The record of R_Omega is a plain number (the penalty is detached for it), so no "Converting a tensor with requires_grad=True to a scalar" warning; the loss term keeps its graph
    (the gradient of the field is still the data gradient plus lambda * the gradient of R_Omega)."""
    import warnings

    import torch

    fa, score, training_step = _smooth_world(monkeypatch)
    reference = torch.autograd.grad(fa.field.penalty(), fa.field.learned)[0]
    converted = []

    class Spy(torch.Tensor):
        """The penalty, with a note of whether it still carried a graph when it was turned into a Python number. (torch warns about that conversion only once per process, so the
        warning alone would pass in a test that runs after another one has used it up.)"""

        def __float__(self):
            converted.append(self.requires_grad)
            return super().__float__()

    real = training_step.terrain_penalty
    monkeypatch.setattr(training_step, 'terrain_penalty', lambda fa_: real(fa_).as_subclass(Spy))
    with warnings.catch_warnings():
        warnings.simplefilter('error')                                            # any warning, the scalar-conversion one included, fails the step
        _, components = _step(fa, score, training_step, smooth_weight=0.5)
    assert converted == [False]                                                   # converted once, and without a graph
    assert components['terrain_penalty'] == 2.5 and type(components['terrain_penalty']) is float
    assert torch.equal(fa.field.learned.grad, 0.5 + 0.5 * reference)              # the loss term kept its graph


def test_a_loss_that_is_not_finite_because_of_the_penalty_stops_the_step(monkeypatch):
    import torch

    fa, score, training_step = _smooth_world(monkeypatch)
    with torch.no_grad():
        fa.field.learned[0, 0, 0] = 1e30                                          # the squared difference overflows float32: the penalty is inf
    seen = []
    optimizer = type('Opt', (), {'step': lambda self: seen.append('step')})()
    with pytest.raises(FloatingPointError, match='step 4'):
        training_step.optimize_batch(fa, 'batch', optimizer, {'block': 1.0}, 4, dict(forward_seconds=0., backward_seconds=0., optimizer_seconds=0.), score=score, clock=lambda: 0., smooth_weight=0.5)
    assert seen == [] and fa.w.grad is None                                       # no backward, no optimizer step
