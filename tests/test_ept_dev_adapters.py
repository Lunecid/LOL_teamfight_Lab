"""Synthetic CPU-only probes; no actual model/data loading or training."""
import importlib.abc
import os
import sys
from types import SimpleNamespace

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


class _ProtectedImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('c_ept.cohort', 'research_pipeline.cohort_policy') or fullname.startswith('research_pipeline.holdout_r3'):
            raise AssertionError('protected import forbidden in synthetic adapter tests')


_guard = _ProtectedImports()
sys.meta_path.insert(0, _guard)
import numpy as np
import pytest
import torch
from c_ept.evaluation.adapters import B1OpenHeads, run_inference, sample_open_value, _aggregate
from c_ept.evaluation.endpoints import m2_endpoints, m4_endpoints, endpoint
from c_ept.ept import data, encoder, heads, loss
from c_ept.models import baselines_b1 as B1
from c_ept.valuation import flowprice
from c_ept.tests.ept_toys import build_ept_world, toy_static
from c_ept.tests.s3_toys import toy_tier_model

torch.set_num_threads(1)


def teardown_module():
    sys.meta_path.remove(_guard)


class Overlay:
    """Small synthetic additions to the existing linked toy stores."""
    def __init__(self, base, arrays):
        self.base, self.arrays = base, arrays

    def array(self, key):
        if key == 'matches/key':
            raise AssertionError('no match keys needed')
        return self.arrays[key] if key in self.arrays else self.base.array(key)


def constant_b1():
    """Real canonical B1 prediction classes with hand-specified degenerate state.

    No fit or file load: these are the existing constant/Jeffreys fallback paths.
    """
    columns = B1.features.b1_columns(())
    groups = {}
    for kind, names in (('masked', B1.HEAD_NAMES), ('open', B1.OPEN_HEAD_NAMES)):
        group = {}
        for name in names:
            obj = getattr(B1, name)(kind)
            layout = B1.make_layout(name, kind, columns)
            for member in obj.parts:
                member.layout_, member.n_features_ = layout, layout.width
                member.info_ = [{'synthetic': True}]
                if hasattr(member, 'base_'):
                    width = {'M2Rate': 1, 'M3Kills': 2, 'M3Obj': 18, 'M4Conv': 18}[name]
                    member.boosters_ = [None] * width
                    member.base_ = np.full(width, np.log(.02 if name in ('M2Rate', 'M4Conv') else .4))
                    member.squeeze_ = name == 'M2Rate'
                else:
                    member.boosters_, member.classes_ = [None], np.array([0])
            group[name] = obj
        groups[kind] = group
    return B1.B1Heads(groups['masked'], groups['open'], ())


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    world = build_ept_world(tmp_path_factory.mktemp('dev_adapter'), roles=('FIT', 'STOP', 'DEV'), n_events=7, n_edges=5)
    # The heads toy has no objectives table: make an explicitly empty history.
    s1 = Overlay(world.s1, {'matches/obj_off': np.zeros(4, dtype=np.int64),
                          **{f'objectives/{k}': np.empty(0, dtype=np.int64) for k in ('t', 'kind', 'sub', 'taker')}})
    n_units = len(s1.array('units/f'))
    labels = Overlay(world.labels, {'labels/f0_unit': np.resize(np.array([-1., 0., 1.]), n_units),
                                   'labels/gold_reps': np.arange(31, dtype=float) * 100})
    stats = data.fit_stats(s1, world.s2)  # only generated synthetic FIT tokens
    torch.manual_seed(17)
    model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=.1, kappas=1, field_channels=2), toy_static())
    bundle = SimpleNamespace(s1=s1, s2=world.s2, labels=labels, role='DEV',
                             prices={k: 100. for k in flowprice.GROUPS}, tier=toy_tier_model())
    return bundle, model, stats, constant_b1()


def test_m2_observed_window_keeps_censored_negative_and_positive():
    e = np.zeros((3, 18)); e[0, :2] = [10, 3]; e[1, :6] = 10
    y = np.zeros_like(e, dtype=int); y[1, 2] = 1
    out = m2_endpoints(np.array([7, 8, 9]), np.array([10, 11, 12]), y, e, np.full_like(e, .02), np.full_like(e, .03))
    for ep in out:
        assert ep['row'].tolist() == [10, 11]
        assert ep['y'].tolist() == [False, True]
        np.testing.assert_allclose(ep['p_ept'], -np.expm1(-np.array([13, 60]) * .02))
    assert out[0]['counts']['clipped_window'] == 1


def test_m4_canonical_cells_and_positive_exposure_only():
    y = np.zeros((2, 18), dtype=int); y[0, [0, 9, 12]] = 1
    out = m4_endpoints(np.array([2, 3]), np.array([4, 5]), y, np.array([7., 0.]), np.full((2, 18), .1), np.full((2, 18), .2))
    assert [bool(e['y'][0]) for e in out] == [True, False, False, True, True, False]
    assert all(e['row'].tolist() == [4] for e in out)
    assert out[0]['p_ept'][0] == pytest.approx(-np.expm1(-.7))


def test_joint_chain_uses_generated_k_and_mixture_probability():
    size = np.array([.25, .75, 0., 0.])
    kill = np.zeros((4, 2))
    gold = np.zeros((4, 2, 6, 31)); gold[..., 0] = 1
    prices = np.zeros(18); prices[0] = 1
    seen = []
    def objects(s, k):
        seen.append(k.copy())
        rates = np.zeros((len(s), 18)); rates[:, 0] = np.where(s == 0, 0., 2.)
        return rates
    kw = dict(row=17, match=9, draws=1000, seed=20261022, check=lambda: None)
    a = sample_open_value((size, kill, gold), objects, prices, np.arange(31.), batch_size=32, **kw)
    b = sample_open_value((size, kill, gold), objects, prices, np.arange(31.), batch_size=7, **kw)
    assert a == b
    assert a[0] == pytest.approx(.75 * (1 - np.exp(-2)), abs=4 * a[2])
    assert a[0] + a[1] == pytest.approx(1)
    assert all(np.count_nonzero(k) == 0 for k in seen)


def test_chain_gold_zero_kills_and_own_layer():
    size = np.array([1., 0., 0., 0.])
    kill = np.tile([2., 0.], (4, 1))
    gold = np.zeros((4, 2, 6, 31))
    for layer in range(6):
        gold[:, :, layer, layer + 1] = 1
    seen = []
    def objects(s, K):
        seen.append(K.copy())
        return np.zeros((len(s), 18))
    p, draw, se, _ = sample_open_value((size, kill, gold), objects, np.ones(18), np.arange(31.) * 100,
        row=1, match=2, draws=1000, seed=20261022, check=lambda: None, batch_size=32)
    k = np.concatenate(seen)
    np.testing.assert_array_equal(k[:, 2], np.minimum(k[:, 0], 6) * 100)
    assert np.count_nonzero(k[:, [1, 3]]) == 0
    assert p == pytest.approx(1 - np.exp(-2), abs=4 * se)
    assert p + draw == pytest.approx(1)


def test_b1_generated_gold_converts_to_thousands_once():
    b1 = constant_b1()
    obj = B1OpenHeads(b1)
    captured = []
    original = b1.open['M3Obj'].predict_blocks
    def spy(numeric, **codes):
        captured.append(numeric.copy())
        return original(numeric, **codes)
    b1.open['M3Obj'].predict_blocks = spy
    K = np.array([[2., 1., 1500., 900.]])
    obj.objects(np.zeros((1, len(b1.columns)), dtype=np.float32), np.array([2]), K)
    np.testing.assert_allclose(captured[0][0, -4:], [2, 1, 1.5, .9])
    np.testing.assert_array_equal(K, [[2, 1, 1500, 900]])


def test_guard_before_payload_access():
    class Stop(Exception): pass
    def check(): raise Stop()
    with pytest.raises(Stop):
        run_inference(object(), object(), object(), object(), check=check, device='cpu')


def test_missing_open_head_refused_without_masked_fallback():
    b1 = constant_b1()
    del b1.open['M3Obj']
    with pytest.raises(ValueError, match='missing B1 open'):
        B1OpenHeads(b1)


def test_invalid_probability_and_duplicate_row_refused():
    with pytest.raises(ValueError, match='probabilities'):
        endpoint('x', 'M2', 't0', 60, 'y', [1], [2], [1], [np.nan], [.5])
    with pytest.raises(ValueError, match='duplicate'):
        endpoint('x', 'M2', 't0', 60, 'y', [1, 1], [2, 2], [0, 1], [.2, .3], [.5, .6])
    with pytest.raises(ValueError, match='multiple matches'):
        _aggregate(np.array([1, 2]), np.array([3, 3]), np.array([1., 2.]))


def test_real_ept_and_b1_api_integration_and_mode_restore(synthetic, monkeypatch):
    bundle, model, stats, b1 = synthetic
    # Calls through real EPT/FrameAveraged and real canonical B1 predictions.
    def forbid(*args, **kwargs): raise AssertionError('no fit/load/optimizer/composer fallback')
    monkeypatch.setattr(B1.B1Heads, 'fit', forbid)
    monkeypatch.setattr(B1.B1Heads, 'composer_heads', forbid)
    monkeypatch.setattr(data, 'fit_stats', forbid)
    monkeypatch.setattr(torch, 'load', forbid)
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbid)
    model.train(); model.encoder.eval()
    modes = [m.training for m in model.modules()]
    params = [p.detach().clone() for p in model.parameters()]
    ticks = []
    out = run_inference(bundle, model, stats, b1, check=lambda: ticks.append(1), device='cpu',
                        batch_size=32, draws=12, scales={k: 2. for k in loss.BLOCKS})
    assert len(out['endpoints']) == 10 and len(out['nll']) == 13
    assert [m.training for m in model.modules()] == modes
    assert all(torch.equal(a, b) for a, b in zip(params, model.parameters()))
    assert all(p.grad is None for p in model.parameters())
    assert len(ticks) > 20
    for ep in out['endpoints']:
        assert np.isfinite(ep['p_ept']).all() and np.isfinite(ep['p_b1']).all()
        assert len(ep['row']) == len(np.unique(ep['row']))
        assert np.all(ep['match'] == 2)
    m3 = out['endpoints'][3]
    assert np.array_equal(m3['y'], bundle.labels.array('labels/f0_unit')[m3['row']] > 0)
    assert np.array_equal(m3['draw_truth'], bundle.labels.array('labels/f0_unit')[m3['row']] == 0)
    assert np.all(m3['p_ept'] + m3['p_draw_ept'] <= 1 + 1e-12)
    for result in out['nll']:
        assert result['common_head'] is (result['name'] in {
            'M2Rate/masked','M2Mark/masked','M2Zone/masked','M3Size/open',
            'M3Kills/open','M3Gold/open','M3Obj/open','M4Conv/open'})
        assert np.isfinite(result['nll_ept']).all() and np.isfinite(result['nll_b1']).all()
        np.testing.assert_allclose(result['nll_ept_normalized'], result['nll_ept'] / 2)
    # Match the independently exposed B1.nll -> HeadScores HR-5 aggregation.
    for kind in ('masked', 'open'):
        for name, scores in b1.nll(bundle.s1, bundle.s2, bundle.labels, 'DEV', kind).items():
            actual = next(r for r in out['nll'] if r['name'] == f'{name}/{kind}')
            values, matches = scores.per_row()
            np.testing.assert_allclose(actual['nll_b1'], values)
            np.testing.assert_array_equal(actual['match'], matches)


def test_failure_restores_modes(synthetic, monkeypatch):
    bundle, model, stats, b1 = synthetic
    modes = [m.training for m in model.modules()]
    def fail(*args, **kwargs): raise RuntimeError('synthetic head failure')
    monkeypatch.setattr(b1.masked['M2Rate'], 'predict', fail)
    with pytest.raises(RuntimeError, match='synthetic head failure'):
        run_inference(bundle, model, stats, b1, check=lambda: None, device='cpu', draws=2)
    assert [m.training for m in model.modules()] == modes


def test_m3_prediction_is_invariant_to_actual_future_K_O_and_truth(synthetic):
    bundle, model, stats, b1 = synthetic
    altered = SimpleNamespace(**vars(bundle))
    altered.s1 = Overlay(bundle.s1, {'units/O': bundle.s1.array('units/O') + 3})
    gold = np.array(bundle.labels.array('labels/kill_gold'))
    altered.labels = Overlay(bundle.labels, {'labels/kill_gold': np.where(gold > 0, gold + 5000, gold),
                                            'labels/f0_unit': -bundle.labels.array('labels/f0_unit')})
    a, b = [run_inference(v, model, stats, b1, check=lambda: None, device='cpu', draws=12)['endpoints'][3]
            for v in (bundle, altered)]
    for key in ('row', 'match', 'p_ept', 'p_b1', 'p_draw_ept', 'p_draw_b1', 'mc_se_ept', 'mc_se_b1'):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a['y'], b['y'])


def no_dev_conversion(bundle, *, remove_views=True):
    """A coherent zero-window population: every DEV engagement ends at game end.

    Remove its M4 event rows and views with a complete index remap. All other
    toy token rows retain their original content. Keeping views intentionally
    produces the invalid positive-view/zero-window fixture below.
    """
    s1, s2 = bundle.s1, bundle.s2
    unit_match = np.asarray(s1.array('units/match'))
    dev = unit_match == 2
    ex = np.array(s1.array('units/conv_exposure_ms')); ex[dev] = 0
    conv = np.array(s1.array('units/conv')); conv[dev] = 0
    delta = np.array(s1.array('units/delta')); delta[dev] = s1.array('matches/T_ms')[2]
    overrides = {'units/conv_exposure_ms': ex, 'units/conv': conv, 'units/delta': delta}
    out = SimpleNamespace(**vars(bundle))
    if remove_views:
        kind, units = s1.array('event_rows/kind'), s1.array('event_rows/unit')
        keep = ~((kind == 2) & dev[units])
        for field in ('kind', 'unit', 'ref', 'frame_k', 't0', 'frame_row'):
            overrides['event_rows/' + field] = np.asarray(s1.array('event_rows/' + field))[keep]
        remap = np.full(len(keep), -1, dtype=np.int64); remap[keep] = np.arange(keep.sum())
        src, index = s2.array('views/src'), np.array(s2.array('views/src_index'))
        viewkeep = np.ones(len(src), dtype=bool)
        viewkeep[src > 0] = keep[index[src > 0]]
        index[src > 0] = remap[index[src > 0]]
        class FilteredViews:
            def array(self, key):
                return (index if key == 'views/src_index' else np.asarray(s2.array(key)))[viewkeep]
        out.s2 = FilteredViews()
    out.s1 = Overlay(s1, overrides)
    return out


def test_empty_legal_conversion_is_unavailable_not_fake_available(synthetic):
    from c_ept.evaluation.metrics import summarize_endpoints
    bundle, model, stats, b1 = synthetic
    empty = no_dev_conversion(bundle)
    out = run_inference(empty, model, stats, b1, check=lambda: None, device='cpu', draws=8,
                        scales={k: 2. for k in loss.BLOCKS})
    m4 = [e for e in out['endpoints'] if e['module'] == 'M4']
    assert len(m4) == 6 and all(len(e['row']) == 0 for e in m4)
    assert all(e['counts']['unavailable_reason'] for e in m4)
    s = summarize_endpoints(out['endpoints'], out['nll'], bootstrap=0)
    for e in s['endpoints']:
        if e['module'] == 'M4':
            assert e['metrics']['brier']['ept']['estimate'] is None
            assert e['metrics']['brier']['reason'] == 'no_rows'
            assert e['roc']['ept']['status'] == 'unavailable'
    assert s['structural_normalized_nll']['status'] == 'unavailable'
    assert s['structural_normalized_nll']['reason'] == 'empty_common_head'


def test_incomplete_conversion_views_remain_a_hard_failure(synthetic):
    bundle, model, stats, b1 = synthetic
    with pytest.raises(ValueError, match='without a conversion window'):
        run_inference(no_dev_conversion(bundle, remove_views=False), model, stats, b1,
                      check=lambda: None, device='cpu', draws=2)


def test_adapter_metrics_report_full_synthetic_integration(synthetic, tmp_path):
    import json
    from pathlib import Path
    from c_ept.evaluation.metrics import summarize_endpoints
    from c_ept.evaluation.reporting import write_report
    bundle, model, stats, b1 = synthetic
    out = run_inference(bundle, model, stats, b1, check=lambda: None, device='cpu', draws=12,
                        scales={k: 2. for k in loss.BLOCKS})
    summary = summarize_endpoints(out['endpoints'], out['nll'], bootstrap=17, check=lambda: None)
    assert summary['structural_normalized_nll']['status'] == 'available'
    assert set(summary['structural_normalized_nll']['heads']) == {
        'M2Rate/masked','M2Mark/masked','M2Zone/masked','M3Size/open',
        'M3Kills/open','M3Gold/open','M3Obj/open','M4Conv/open'}
    m3 = next(e for e in summary['endpoints'] if e['module'] == 'M3')
    assert m3['horizon_s'] is None
    assert set(m3['mc_uncertainty']) == {'mc_se_ept', 'mc_se_b1', 'mc_se_draw_ept', 'mc_se_draw_b1'}
    paths = write_report(summary, tmp_path / 'report', provenance={**out['provenance'], 'synthetic': True}, render=False)
    assert all(Path(p).is_file() for p in paths.values())
    saved = json.loads(Path(paths['metrics_json']).read_text(encoding='utf-8'))
    assert saved['endpoints'] == summary['endpoints']
    assert saved['structural_normalized_nll'] == summary['structural_normalized_nll']
    assert saved['provenance']['synthetic'] is True


def test_structural_nll_excludes_five_masked_m3_m4_blocks():
    from c_ept.evaluation.adapters import _nll_record
    from c_ept.evaluation.metrics import summarize_endpoints
    # Independent oracle: excluded blocks must not change the G9b estimand.
    expected={'M2Rate/masked','M2Mark/masked','M2Zone/masked','M3Size/open',
              'M3Kills/open','M3Gold/open','M3Obj/open','M4Conv/open'}
    records=[]
    for key in loss.BLOCKS:
        head,kind=key.split('/')
        records.append(_nll_record(head,kind,np.array([1]),np.array([0]),
            np.array([0. if key in expected else 1.]),np.array([0.]),{},
            {k:1. for k in loss.BLOCKS}))
    result=summarize_endpoints([],records,bootstrap=0)
    structural=result['structural_normalized_nll']
    assert set(structural['heads'])==expected and len(records)==13
    assert structural['metrics']['ept']['estimate']==0.
    assert structural['metrics']['delta']['estimate']==0.
    excluded=[r for r in result['nll'] if r['name'] not in expected]
    assert len(excluded)==5 and all(r['metrics']['nll']['ept']['estimate']==1. for r in excluded)
