"""실제 모듈을 가짜 torch/loss와 로드한다. CPU/GPU tensor 연산은 없다."""
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
          'loss': SimpleNamespace(block_key=lambda h, k: h + '/' + k, head_nll=lambda h, value, target: value)}
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
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_stop')
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
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<STOP mode regression>', 'exec'), ns)
    if interrupted:
        with pytest.raises(ValueError, match='STOP interruption'):
            ns['evaluate_stop'](model, SimpleNamespace(epoch=epoch), {})
    else:
        assert ns['evaluate_stop'](model, SimpleNamespace(epoch=epoch), {}) == 17.
    assert model.training == was_training and seen == ['wrapper_eval']
