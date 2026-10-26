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
from c_ept.data import head_rows
from c_ept.domain import symmetry
from c_ept.evaluation import adapters
from c_ept.preprocessing import features, tokens
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


# --- EPT-T012 S4b preparation M12 (decision D14): the primary population of the t0 forecasts and the counts of the M2 endpoints ----------------------------------------------------------
def test_primary_rows_is_the_integer_millisecond_condition_hi_minus_t0_at_least_H():
    """hi = min(t0 + 180 s, T - 30 s), so hi - t0 >= H x 1000 is exactly t0 + H <= T - 30 s (H = 60, 120, 180 s). Boundary: equal is primary, one ms less is not; the comparison is in whole ms."""
    from c_ept.evaluation.endpoints import primary_rows
    t0 = np.array([120_300, 120_300, 120_300, 0, 5], dtype=np.int64)
    hi = t0 + np.array([60_000, 59_999, 60_001, 180_000, -3], dtype=np.int64)
    got = primary_rows(t0, hi, 60)
    assert got.dtype == bool and got.tolist() == [True, False, True, True, False]
    assert primary_rows(t0, hi, 120).tolist() == [False, False, False, True, False] and primary_rows(t0, hi, 180).tolist() == [False, False, False, True, False]
    assert primary_rows(t0[:0], hi[:0], 60).shape == (0,)
    big = np.array([2**53 + 1], dtype=np.int64)                       # a float64 comparison could not tell these two apart; whole ms can
    assert primary_rows(big, big + 60_000, 60).tolist() == [True] and primary_rows(big, big + 59_999, 60).tolist() == [False]


@pytest.mark.parametrize('t0, hi, H, error', [
    (np.array([1.0]), np.array([100_000]), 60, TypeError), (np.array([1]), np.array([100_000.0]), 60, TypeError),
    (np.array([True]), np.array([100_000]), 60, TypeError), (np.array([1]), np.array([100_000, 5]), 60, ValueError),
    (np.array([[1]]), np.array([[100_000]]), 60, ValueError), (np.array([1]), np.array([100_000]), 59, ValueError),
    (np.array([1]), np.array([100_000]), 60.0, ValueError), (np.array([1]), np.array([100_000]), True, ValueError),
    (np.array([1]), np.array([100_000]), 240, ValueError)])
def test_primary_rows_refuses_floats_booleans_other_shapes_and_horizons_that_are_none_of_60_120_180(t0, hi, H, error):
    from c_ept.evaluation.endpoints import primary_rows
    with pytest.raises(error):
        primary_rows(t0, hi, H)


def population_rows():
    """Five rows with the windows hi - t0 = 180, 130, 70, 42.7 and 0 s (the DEV toy match 2 has the rows with 180, 162.8, 102.8 and 42.8 s): the observed exposure of a row is the window cut into 10 s bins."""
    window_ms = np.array([180_000, 130_000, 70_000, 42_700, 0])
    e = np.clip(window_ms[:, None] - 10_000 * np.arange(18)[None, :], 0, 10_000) / 1000.0
    t0 = np.full(5, 120_300, dtype=np.int64)
    return t0, t0 + window_ms, e


def test_m2_population_selects_the_primary_rows_of_each_horizon_and_records_the_candidate_and_selected_counts():
    from c_ept.evaluation.endpoints import primary_rows
    t0, hi, e = population_rows()
    match, row = np.array([7, 7, 8, 8, 9]), np.array([10, 11, 12, 13, 14])
    y = np.zeros((5, 18), dtype=int); y[0, 2] = 1; y[1, 8] = 1; y[2, 1] = 1; y[3, 3] = 1
    rate = np.linspace(.01, .05, 18)[None, :] * np.ones((5, 1))
    default = m2_endpoints(match, row, y, e, rate, 1.5 * rate)
    pop = {h: primary_rows(t0, hi, h) for h in (60, 120, 180)}
    assert [p.tolist() for p in pop.values()] == [[True, True, True, False, False], [True, True, False, False, False], [True, False, False, False, False]]
    chosen = m2_endpoints(match, row, y, e, rate, 1.5 * rate, population=pop)
    for h, a, b in zip((60, 120, 180), default, chosen):
        keep = np.isin(a['row'], row[pop[h]])
        assert b['row'].tolist() == a['row'][keep].tolist() and b['row'].tolist() == row[pop[h]].tolist()
        for field in ('match', 'y', 'p_ept', 'p_b1'):
            assert np.array_equal(b[field], a[field][keep]), (h, field)
        assert b['counts']['population_candidate'] == 5 and b['counts']['population_selected'] == int(pop[h].sum())
        assert b['counts']['eligible'] == int(pop[h].sum()) and b['counts']['clipped_window'] == 0         # a primary row has the whole window of the horizon
        assert b['counts']['candidate'] == 5 and b['counts']['zero_exposure'] == 1                        # the row without any exposure is no candidate with exposure, as before
        assert 'population_selected' not in a['counts'] and 'population_candidate' not in a['counts']
    explicit = m2_endpoints(match, row, y, e, rate, 1.5 * rate, population=None)
    for a, b in zip(default, explicit):
        assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in a)


def test_secondary_rows_give_every_comparator_the_same_exposure():
    """The rows that are not primary form the secondary population: both comparators are scored on the observed (cut) exposure of the very same rows, so a window that the game end cut counts the same for the
    two. With rates r and 1.5 r the two probabilities are 1 - exp(-r sum(e)) and 1 - exp(-1.5 r sum(e)) of one array e."""
    from c_ept.evaluation.endpoints import primary_rows
    t0, hi, e = population_rows()
    match, row = np.array([7, 7, 8, 8, 9]), np.array([10, 11, 12, 13, 14])
    y = np.zeros((5, 18), dtype=int); y[2, 1] = 1; y[3, 3] = 1
    r = .02
    rate = np.full((5, 18), r)
    secondary = {h: ~primary_rows(t0, hi, h) for h in (60, 120, 180)}
    out = m2_endpoints(match, row, y, e, rate, 1.5 * rate, population=secondary)
    for h, ep in zip((60, 120, 180), out):
        q = h // 10
        rows_ = np.flatnonzero(secondary[h] & (e.sum(1) > 0))
        assert ep['row'].tolist() == row[rows_].tolist()
        seen = e[rows_, :q].sum(1)
        np.testing.assert_allclose(ep['p_ept'], -np.expm1(-r * seen), rtol=1e-12)
        np.testing.assert_allclose(ep['p_b1'], -np.expm1(-1.5 * r * seen), rtol=1e-12)
        assert ep['counts']['clipped_window'] == int((seen < h).sum()) and ep['counts']['clipped_window'] > 0       # the secondary rows are the ones whose window the game end cut
        assert ep['counts']['population_selected'] == int(secondary[h].sum())
    primary = m2_endpoints(match, row, y, e, rate, 1.5 * rate, population={h: ~m for h, m in secondary.items()})
    for a, b, c in zip(primary, out, m2_endpoints(match, row, y, e, rate, 1.5 * rate)):
        assert sorted(a['row'].tolist() + b['row'].tolist()) == sorted(c['row'].tolist())          # the two populations partition the rows that have exposure


@pytest.mark.parametrize('bad', [np.ones(5, dtype=bool), [True] * 5, {60: np.ones(5, dtype=bool)}, {60: np.ones(5, dtype=bool), 120: np.ones(5, dtype=bool), 180: np.ones(4, dtype=bool)},
                                  {60: np.ones(5), 120: np.ones(5), 180: np.ones(5)}, {60: np.ones(5, dtype=bool), 120: np.ones(5, dtype=bool), 180: np.ones(5, dtype=bool), 240: np.ones(5, dtype=bool)},
                                  {60: np.ones((5, 1), dtype=bool), 120: np.ones(5, dtype=bool), 180: np.ones(5, dtype=bool)}])
def test_m2_population_must_be_a_mask_of_every_row_for_each_of_the_three_horizons(bad):
    t0, hi, e = population_rows()
    rate = np.full((5, 18), .02)
    with pytest.raises(ValueError, match='population'):
        m2_endpoints(np.arange(5), np.arange(5), np.zeros((5, 18), dtype=int), e, rate, rate, population=bad)


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


# --- EPT-T011 A4-F8: the bundle role is mandatory, and M3 truth is validated before any head runs -------------------
class _NoStoreRead:
    """Bundle stand-in without `role`: reading any store (or anything else) fails loudly."""
    def __getattr__(self, name):
        if name == 'role':
            raise AttributeError(name)
        raise AssertionError(f'bundle.{name} was read before the role check')


def test_bundle_without_role_is_refused_before_any_store_read():
    with pytest.raises(ValueError, match='only DEV'):
        run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu')


@pytest.mark.parametrize('role', ['STOP', 'FIT', 'TEST', 'dev', '', None])
def test_bundle_with_any_other_role_is_refused_before_any_store_read(role):
    bundle = _NoStoreRead()
    bundle.__dict__['role'] = role         # present in the instance dict, so __getattr__ is not consulted for it
    with pytest.raises(ValueError, match='only DEV'):
        run_inference(bundle, object(), object(), object(), check=lambda: None, device='cpu')


# --- S4b preparation K6d: the rows of one role (DEV by default, STOP, TEST only with the gate's opening) ---------------------------------
def with_roles_swapped(bundle, a, b):
    """The synthetic bundle in which the matches of the roles ``a`` and ``b`` have changed roles (``matches/role``): every row of ``a`` is now a row of ``b`` and the other way round; nothing else moves."""
    from c_ept import config as C
    code = np.array(bundle.s1.array('matches/role'))
    ia, ib = C.ROLES.index(a), C.ROLES.index(b)
    swapped = np.where(code == ia, ib, np.where(code == ib, ia, code)).astype(code.dtype)
    out = SimpleNamespace(**vars(bundle))
    out.s1 = Overlay(bundle.s1, {'matches/role': swapped})
    return out


def without_role_marks(result):
    """The result of ``run_inference`` with the provenance entries that name the role removed, so that the rows of two roles can be compared bit for bit."""
    out = dict(result)
    out['provenance'] = {k: v for k, v in result['provenance'].items() if k not in ('role', 'role_note')}
    return out


def test_the_stop_role_evaluates_the_rows_of_the_stop_matches_and_nothing_else(synthetic):
    """K6d item 1: every row selection takes the role. The STOP match of the toy world under the role STOP is, row for row and bit for bit, the same match once it is labelled DEV and
    asked for as DEV (and the DEV matches under STOP are the DEV result of the swapped world): a row selection that stayed literally 'DEV' would read other matches and differ."""
    bundle, model, stats, b1 = synthetic
    stop_bundle = SimpleNamespace(**{**vars(bundle), 'role': 'STOP'})
    stop = run_inference(stop_bundle, model, stats, b1, role='STOP', **WINNER_TRUTH_RUN)
    dev = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    swapped = with_roles_swapped(bundle, 'STOP', 'DEV')
    as_dev = run_inference(swapped, model, stats, b1, **WINNER_TRUTH_RUN)           # the role DEV of the world whose STOP match is called DEV
    assert_same(without_role_marks(stop), without_role_marks(as_dev))
    assert stop['provenance']['role'] == 'STOP' and dev['provenance']['role'] == 'DEV'
    matches = {int(m) for record in stop['nll'] for m in record['match']}
    assert matches == {1} and {int(m) for record in dev['nll'] for m in record['match']} == {2}           # the toy world: match 1 is the STOP match, match 2 the DEV match
    assert sum(len(ep['row']) for ep in stop['endpoints']) > 0 and 'role_note' in stop['provenance'] and 'role_note' not in dev['provenance']
    note = stop['provenance']['role_note']
    # review I1: neutral about the row sets. The matched B1 stops on the same matched subset (membership.stop) as the EPT, and this call cannot see either row set (B1Heads records the role only)
    assert 'for early stopping' in note and 'not a held-out estimate' in note and 'its own training run' in note and 'cannot see' in note
    assert 'the STOP matches are the early-stopping data' not in note and 'its subset of them' not in note and 'subset' not in note.replace('matched subset', '')


def test_the_public_primary_masks_are_the_population_filter_hi_minus_t0_at_least_h_in_whole_ms(synthetic):
    """K6e: the composed V_start pass of the evaluation runner asks this module for the primary population instead of reading ``frame_rows/hi`` itself (the runner is a strict module: it may not import the
    private ``_primary_masks``); the answer is the integer rule of decision D14 on any rows, and it is the very function that the occurrence endpoints use."""
    bundle, *_ = synthetic
    t0, hi = (np.asarray(bundle.s1.array(f'frame_rows/{name}')) for name in ('t0', 'hi'))
    rows = np.array([3, 0, 7, 7, 12])
    masks = adapters.primary_masks(bundle.s1, rows)
    assert list(masks) == [60, 120, 180]
    for h, mask in masks.items():
        assert mask.dtype == np.bool_ and np.array_equal(mask, hi[rows] - t0[rows] >= h * 1000)
    assert adapters.primary_masks(bundle.s1, np.empty(0, dtype=np.int64))[60].shape == (0,)
    with pytest.raises(ValueError, match='frame_rows'):
        adapters.primary_masks(bundle.s1, np.array([len(t0)]))
    private = adapters._primary_masks(bundle.s1, rows)
    assert all(np.array_equal(private[h], masks[h]) for h in masks)


def test_the_stop_role_takes_the_swapped_b1_frame_rows_too(synthetic):
    """The second frame of B1 (``b1_swap_average``, decision B22) is a third row view of the same role: with it the STOP result equals the DEV result of the world whose STOP match is called DEV."""
    bundle, model, stats, b1 = synthetic
    stop = run_inference(SimpleNamespace(**{**vars(bundle), 'role': 'STOP'}), model, stats, b1, role='STOP', b1_swap_average=True, **WINNER_TRUTH_RUN)
    as_dev = run_inference(with_roles_swapped(bundle, 'STOP', 'DEV'), model, stats, b1, b1_swap_average=True, **WINNER_TRUTH_RUN)
    assert_same(without_role_marks(stop), without_role_marks(as_dev))
    assert stop['provenance']['b1_swap_average'] is True


def test_the_stop_role_takes_the_rows_of_b0_too(synthetic):
    """The B0 rows of the relative NLL are the rows of the role, never the DEV rows: with a B0 the STOP result equals the DEV result of the world whose STOP match is called DEV, ``nll_b0`` included."""
    import warnings
    from c_ept.models import baselines_b0
    bundle, model, stats, b1 = synthetic
    means = features.b0_fit_means(bundle.s2, np.flatnonzero(features.split_views(bundle.s1, bundle.s2, 'FIT')))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                                                          # the toy heads do not converge: the known warnings of the B0 tests
        b0 = baselines_b0.B0Heads.fit(bundle.s1, bundle.s2, bundle.labels, means)
    stop = run_inference(SimpleNamespace(**{**vars(bundle), 'role': 'STOP'}), model, stats, b1, role='STOP', b0=b0, **WINNER_TRUTH_RUN)
    as_dev = run_inference(with_roles_swapped(bundle, 'STOP', 'DEV'), model, stats, b1, b0=b0, **WINNER_TRUTH_RUN)
    assert_same(without_role_marks(stop), without_role_marks(as_dev))
    assert all('nll_b0' in record for record in stop['nll']) and {int(m) for record in stop['nll'] for m in record['match']} == {1}


def test_the_stop_role_is_scored_on_the_stop_rows_by_the_b1_scorer_too(synthetic):
    bundle, model, stats, b1 = synthetic
    stop = run_inference(SimpleNamespace(**{**vars(bundle), 'role': 'STOP'}), model, stats, b1, role='STOP', **WINNER_TRUTH_RUN)
    for kind in ('masked', 'open'):
        for name, scores in b1.nll(bundle.s1, bundle.s2, bundle.labels, 'STOP', kind).items():
            actual = next(r for r in stop['nll'] if r['name'] == f'{name}/{kind}')
            values, matches = scores.per_row()
            np.testing.assert_allclose(actual['nll_b1'], values)
            np.testing.assert_array_equal(actual['match'], matches)


def test_without_the_monte_carlo_every_other_record_is_the_same_bits_and_no_draw_and_no_f0_truth_is_used(synthetic, monkeypatch):
    """K6d, decision D30: ``monte_carlo=False`` is the closed-form path. The M3 sign endpoint is the only thing that goes; the other endpoints and the thirteen NLL records are, bit for bit, those of the
    full evaluation, for the DEV role and for STOP; no open value is drawn; ``labels/f0_unit`` is not read."""
    import inspect
    bundle, model, stats, b1 = synthetic
    parameter = inspect.signature(run_inference).parameters['monte_carlo']
    assert parameter.default is True and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    cases = (('DEV', bundle), ('STOP', SimpleNamespace(**{**vars(bundle), 'role': 'STOP'})))
    fulls = {role: run_inference(given, model, stats, b1, role=role, **WINNER_TRUTH_RUN) for role, given in cases}
    for role, given in cases:
        assert_same(fulls[role], run_inference(given, model, stats, b1, role=role, monte_carlo=True, **WINNER_TRUTH_RUN))
    draws = []
    monkeypatch.setattr(adapters, 'sample_open_value', lambda *a, **k: draws.append(1))
    class NoF0(Overlay):
        def array(self, key):
            if key == 'labels/f0_unit':
                pytest.fail('labels/f0_unit was read')
            return super().array(key)
    for role, given in cases:
        full = fulls[role]
        closed = run_inference(SimpleNamespace(**{**vars(given), 'labels': NoF0(bundle.labels, {})}), model, stats, b1, role=role, monte_carlo=False, **WINNER_TRUTH_RUN)
        assert draws == [] and closed['provenance']['monte_carlo'] is False and 'monte_carlo' not in full['provenance']
        assert [ep['name'] for ep in closed['endpoints']] == [ep['name'] for ep in full['endpoints'] if ep['name'] != 'M3/open/positive_value'] and len(closed['endpoints']) == 9
        for ep in closed['endpoints']:
            assert_same(ep, next(other for other in full['endpoints'] if other['name'] == ep['name']))
        assert_same(closed['nll'], full['nll']) and len(closed['nll']) == 13
        drawn = ('seed', 'rng', 'mc_uncertainty')                                                  # what only a drawn evaluation has (review M2): no draw, no seed, no stream, no sampling error
        assert closed['provenance']['draws'] == 0 and not set(drawn) & set(closed['provenance']) and full['provenance']['draws'] == 12
        assert_same({k: v for k, v in closed['provenance'].items() if not k.startswith('monte_carlo') and k != 'draws'}, {k: v for k, v in full['provenance'].items() if k not in (*drawn, 'draws')})


@pytest.fixture(scope='module')
def closed_world(tmp_path_factory):
    """The consistent toy world (its units have the kills of their first-kill team) with a tiny EPT and a real B1 fitted for the shifted kill law and the binned K condition: the settings of S4b."""
    import gc
    world = build_ept_world(tmp_path_factory.mktemp('closed_world'), n_events=7, n_edges=5, consistent_first_kill=True)
    n_matches = len(np.asarray(world.s1.array('matches/role')))
    s1 = Overlay(world.s1, {'matches/obj_off': np.zeros(n_matches + 1, dtype=np.int64), **{f'objectives/{k}': np.empty(0, dtype=np.int64) for k in ('t', 'kind', 'sub', 'taker')}})
    n_units = len(s1.array('units/f'))
    labels = Overlay(world.labels, {'labels/f0_unit': np.resize(np.array([-1., 0., 1.]), n_units), 'labels/gold_reps': np.arange(31, dtype=float) * 100, 'labels/obj_value': np.zeros((n_units, 2))})
    stats = data.fit_stats(s1, world.s2)
    torch.manual_seed(17)
    model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=.1, kappas=1, field_channels=2), toy_static())
    bundle = SimpleNamespace(s1=s1, s2=world.s2, labels=labels, role='DEV', prices={k: 100. for k in flowprice.GROUPS}, tier=toy_tier_model())
    dropped = features.dropped_columns(world.s2, np.flatnonzero(features.split_views(world.s1, world.s2, 'FIT')))
    small = {'num_leaves': 7, 'min_data_in_leaf': 5, 'num_iterations': 12, 'min_data_per_group': 5, 'cat_smooth': 1.0, 'num_threads': 1}
    b1 = B1.B1Heads.fit(world.s1, world.s2, world.labels, dropped, params=small, kill_law='shifted', k_condition='binned')
    yield bundle, model, stats, b1
    del world
    gc.collect()


def test_the_closed_path_is_the_full_evaluations_bits_under_the_shifted_law_the_binned_k_and_the_binned_winner_truth(closed_world):
    """Review M5: the settings of S4b (kill_law shifted, k_condition binned with the in-scope FIT representatives, winner_truth binned) with ``monte_carlo=False``: the binned-truth labels (``labels/gold_bin``
    of the winner truth, ``labels/obj_value``) and ``labels/f0_unit`` are not read, no draw is made, and every record that is left is the full evaluation's, bit for bit, for DEV and for STOP."""
    bundle, model, stats, b1 = closed_world
    settings = dict(check=lambda: None, device='cpu', draws=6, batch_size=4, kill_law='shifted', k_condition='binned', gold_reps='in_scope_fit', winner_truth='binned',
                    scales={k: 2. for k in loss.BLOCKS})
    for role in ('DEV', 'STOP'):
        given = SimpleNamespace(**{**vars(bundle), 'role': role})
        full = run_inference(given, model, stats, b1, role=role, **settings)
        assert full['provenance']['winner_truth'] == 'binned' and full['provenance']['kill_law'] == 'shifted' and full['provenance']['k_condition'] == 'binned'
        assert any(ep['name'] == 'M3/open/positive_value' and len(ep['row']) > 0 for ep in full['endpoints']) or role == 'STOP'
        read = []

        class Watching(Overlay):
            def array(self, key):
                read.append(key)
                return super().array(key)
        closed = run_inference(SimpleNamespace(**{**vars(given), 'labels': Watching(bundle.labels, {})}), model, stats, b1, role=role, monte_carlo=False, **settings)
        assert 'labels/f0_unit' not in read and 'labels/obj_value' not in read and 'labels/gold_reps' not in read                  # the binned reps come from the FIT bin means, the truth labels are not needed
        assert [ep['name'] for ep in closed['endpoints']] == [ep['name'] for ep in full['endpoints'] if ep['name'] != 'M3/open/positive_value']
        for ep in closed['endpoints']:
            assert_same(ep, next(other for other in full['endpoints'] if other['name'] == ep['name']))
        assert_same(closed['nll'], full['nll'])
        assert closed['provenance']['draws'] == 0 and closed['provenance']['kill_law'] == 'shifted' and closed['provenance']['gold_reps_sha256'] == full['provenance']['gold_reps_sha256']
    assert len(full['nll']) == 13 and all(np.isfinite(r['nll_ept']).all() for r in full['nll'])


def test_the_progress_callback_sees_every_head_block_begin_and_end_with_counts_and_changes_no_bit(synthetic):
    """Review M3: ``progress(event, block, counts)`` is called at the begin (empty counts) and at the end of each of the 13 head blocks, in the order of ``data.ROW_FIELDS``; the end counts are whole
    numbers: ``rows`` and ``views`` (the views that the block encodes, with repeats between batches) and, for the open M3 size block with a Monte Carlo, ``units`` and ``draws``. It reads no array: the result is the default's bits."""
    import inspect
    bundle, model, stats, b1 = synthetic
    parameter = inspect.signature(run_inference).parameters['progress']
    assert parameter.default is None and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    seen = []
    plain = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    watched = run_inference(bundle, model, stats, b1, progress=lambda event, block, counts: seen.append((event, block, dict(counts))), **WINNER_TRUTH_RUN)
    assert_same(plain, watched)
    names = [loss.block_key(head, kind) for head, kind in data.ROW_FIELDS]
    assert len(names) == 13 and [(event, block) for event, block, _ in seen] == [(event, name) for name in names for event in ('begin', 'end')]
    assert all(counts == {} for event, _, counts in seen if event == 'begin')
    ends = {block: counts for event, block, counts in seen if event == 'end'}
    assert all(set(counts) <= {'rows', 'views', 'units', 'draws'} and {'rows', 'views'} <= set(counts) for counts in ends.values())
    assert all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for counts in ends.values() for v in counts.values())
    assert ends['M2Rate/masked']['rows'] > 0 and ends['M2Rate/masked']['views'] > 0
    assert ends['M3Size/open']['draws'] == 12 and ends['M3Size/open']['units'] == ends['M3Size/open']['rows'] and [b for b, c in ends.items() if 'draws' in c] == ['M3Size/open']
    seen.clear()
    run_inference(bundle, model, stats, b1, monte_carlo=False, progress=lambda event, block, counts: seen.append((event, block, dict(counts))), **WINNER_TRUTH_RUN)
    assert [c for e, b, c in seen if e == 'end' and b == 'M3Size/open'][0].keys() == {'rows', 'views'}                                  # no draw, so no draws and no units of draws
    with pytest.raises(ValueError, match='progress'):
        run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu', progress=3)


def test_the_guard_of_a_block_boundary_runs_after_the_block_before_is_reported_finished_and_before_the_next_begins(synthetic):
    """K6d re-check R1: the guard at the top of each turn of the block loop runs between the ``'end'`` of the block before and the ``'begin'`` of the next one (and after the last ``'end'``), so a guard
    that fails there (the wall cap, a stop request) finds no block open and cannot close a finished block as failed. Only the order of two calls changes: the result is the default's bits.

    해석: 블록과 블록 사이의 경계 점검은 앞 블록의 끝 알림 뒤, 다음 블록의 시작 알림 앞에서 한다. 점검이 이 자리에서 실패해도 이미 끝난 블록이 실패로 닫히지 않는다."""
    bundle, model, stats, b1 = synthetic
    names = [loss.block_key(head, kind) for head, kind in data.ROW_FIELDS]
    seen = []
    given = {**WINNER_TRUTH_RUN, 'check': lambda: seen.append(('check', None))}
    watched = run_inference(bundle, model, stats, b1, progress=lambda event, block, counts: seen.append((event, block)), **given)
    assert_same(run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN), watched)
    assert seen[:1] == [('check', None)] and seen.index(('begin', names[0])) > 0 and seen[seen.index(('begin', names[0])) - 1] == ('check', None)        # the guards before the first block
    for before, name in zip(names, names[1:]):
        at = seen.index(('begin', name))
        assert seen[at - 2:at] == [('end', before), ('check', None)], (before, name)                                                                   # end, the guard, then begin
    assert seen[-2:] == [('end', names[-1]), ('check', None)]                                                                                         # and the guard after the last block


@pytest.mark.parametrize('bad', [None, 0, 1, 'yes'])
def test_monte_carlo_must_be_a_bool_and_is_checked_before_any_store_read(bad):
    with pytest.raises(ValueError, match='monte_carlo'):
        run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu', monte_carlo=bad)


def test_the_default_role_is_dev_and_the_explicit_dev_is_bit_identical(synthetic):
    import inspect
    bundle, model, stats, b1 = synthetic
    parameters = inspect.signature(run_inference).parameters
    assert parameters['role'].default == 'DEV' and parameters['holdout'].default is None
    assert parameters['role'].kind is inspect.Parameter.KEYWORD_ONLY and parameters['holdout'].kind is inspect.Parameter.KEYWORD_ONLY
    default = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    assert_same(default, run_inference(bundle, model, stats, b1, role='DEV', **WINNER_TRUTH_RUN))
    assert 'role_note' not in default['provenance'] and default['provenance']['role'] == 'DEV'


@pytest.mark.parametrize('role', ['FIT', 'CAL', 'AUDIT', 'EXTERNAL', 'UNUSED', 'dev', 'Stop', '', None, 1, ('DEV',)])
def test_every_role_but_dev_stop_and_the_gated_test_is_refused_before_any_store_read(role):
    bundle = _NoStoreRead()
    bundle.__dict__['role'] = role if isinstance(role, str) else 'DEV'
    with pytest.raises(ValueError, match='role must be'):
        run_inference(bundle, object(), object(), object(), check=lambda: None, device='cpu', role=role)


def test_the_role_test_needs_the_opening_that_only_the_gate_makes():
    """No opening, a plain dict that looks like the ledger entry, and an opening that someone makes by hand are refused before any store read; the opening of another role is not accepted either."""
    entry = {'seq': 3, 'event': 'test_opened', 'when': 1.0, 'previous_sha256': 'a' * 64, 'record': {}}
    bundle = _NoStoreRead()
    bundle.__dict__['role'] = 'TEST'
    for holdout in (None, entry, 'opened', True):
        with pytest.raises(ValueError, match='HoldoutOpening'):
            run_inference(bundle, object(), object(), object(), check=lambda: None, device='cpu', role='TEST', holdout=holdout)
    with pytest.raises(TypeError, match='HoldoutGate.open'):
        adapters.HoldoutOpening(entry)                                                          # made by hand: refused
    for role in ('DEV', 'STOP'):                                                                 # an opening that nothing asks for is not carried along
        bundle.__dict__['role'] = role
        with pytest.raises(ValueError, match='belongs to the role TEST'):
            run_inference(bundle, object(), object(), object(), check=lambda: None, device='cpu', role=role, holdout=entry)


def test_the_holdout_opening_is_made_inside_the_gates_open_function_only():
    """The constructor looks at its caller: a function called ``open`` of the module ``c_ept.run.evaluate`` may make one (and only for the ``test_opened`` entry), every other caller may not."""
    entry = {'seq': 3, 'event': 'test_opened', 'when': 1.0, 'previous_sha256': 'a' * 64, 'record': {}}
    source = 'def open(entry):\n    return HoldoutOpening(entry)\n'
    for module, name, allowed in (('c_ept.run.evaluate', 'open', True), ('c_ept.run.evaluate', 'close', False), ('c_ept.run.other', 'open', False)):
        scope = {'__name__': module, 'HoldoutOpening': adapters.HoldoutOpening}
        exec(compile(source.replace('open', name), f'<{module}>', 'exec'), scope)
        if allowed:
            opened = scope[name](entry)
            assert opened == entry and isinstance(opened, dict) and opened['event'] == 'test_opened' and opened['seq'] == 3
            with pytest.raises(ValueError, match='test_opened'):
                scope[name]({**entry, 'event': 'dev_access'})
        else:
            with pytest.raises(TypeError, match='HoldoutGate.open'):
                scope[name](entry)


def test_a_bundle_of_another_role_than_the_call_is_refused_and_the_message_names_the_role_of_the_call(synthetic):
    bundle, model, stats, b1 = synthetic                                                        # a DEV bundle
    with pytest.raises(ValueError, match='only STOP'):
        run_inference(bundle, model, stats, b1, role='STOP', **WINNER_TRUTH_RUN)
    stop_bundle = SimpleNamespace(**{**vars(bundle), 'role': 'STOP'})
    with pytest.raises(ValueError, match='only DEV'):
        run_inference(stop_bundle, model, stats, b1, **WINNER_TRUTH_RUN)                        # a STOP bundle asked as the default role
    with pytest.raises(ValueError, match='only STOP'):
        run_inference(SimpleNamespace(**{k: v for k, v in vars(bundle).items() if k != 'role'}), model, stats, b1, role='STOP', **WINNER_TRUTH_RUN)    # no role: never defaulted


def poisoned_f0(bundle, value, *, dev_open):
    """The synthetic bundle with one `labels/f0_unit` entry replaced by `value`.

    dev_open=True poisons an in-scope DEV unit (a unit of the open M3 population);
    dev_open=False poisons a unit of another role, which the DEV evaluation never reads."""
    match = np.asarray(bundle.s1.array('units/match'))
    scope = np.asarray(bundle.s1.array('units/in_scope')).astype(bool)
    f0 = np.array(bundle.labels.array('labels/f0_unit'), dtype=float)
    target = np.flatnonzero((match == 2) & scope)[0] if dev_open else np.flatnonzero(match != 2)[0]
    f0[target] = value
    out = SimpleNamespace(**vars(bundle))
    out.labels = Overlay(bundle.labels, {'labels/f0_unit': f0})
    return out


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_open_unit_truth_is_refused_before_the_first_head_runs(synthetic, monkeypatch, bad):
    bundle, model, stats, b1 = synthetic
    calls = []
    original = b1.masked['M2Rate'].predict          # the first head block that run_inference evaluates
    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(b1.masked['M2Rate'], 'predict', spy)
    with pytest.raises(ValueError, match='nonfinite M3 f0 truth'):
        run_inference(poisoned_f0(bundle, bad, dev_open=True), model, stats, b1,
                      check=lambda: None, device='cpu', draws=2)
    assert calls == []


def test_truth_check_does_not_refuse_units_outside_the_dev_open_population(synthetic, monkeypatch):
    # Same spy, clean control: the early check is exactly the old end check moved forward, not a wider one.
    bundle, model, stats, b1 = synthetic
    calls = []
    original = b1.masked['M2Rate'].predict
    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(b1.masked['M2Rate'], 'predict', spy)
    out = run_inference(poisoned_f0(bundle, np.nan, dev_open=False), model, stats, b1,
                        check=lambda: None, device='cpu', draws=2)
    assert calls and len(out['endpoints']) == 10


def test_provenance_comparison_text_defaults_to_t003_and_is_a_parameter(synthetic):
    import json
    bundle, model, stats, b1 = synthetic
    common = dict(check=lambda: None, device='cpu', draws=2)
    default = run_inference(bundle, model, stats, b1, **common)['provenance']['comparison']
    assert default == 'whole-FIT B1 versus FIT/STOP-25%-EPT pilot; no generalization or controlled architecture claim'
    text = 'matched 25% diagnostic: EPT versus B1'
    provenance = run_inference(bundle, model, stats, b1, comparison=text, **common)['provenance']
    assert provenance['comparison'] == text
    # With the matched text, nothing else the adapter records carries the T003 scope wording either.
    assert 'whole-FIT' not in json.dumps(provenance, default=str)


def test_the_two_t003_texts_have_distinct_names_and_keep_their_emitted_strings():
    """EPT-T011 group 1 review M4: adapters and metrics each had a T003_COMPARISON with a different text. The emitted T003 strings are unchanged."""
    import inspect
    from c_ept.evaluation import adapters, metrics
    assert adapters.T003_PROVENANCE_COMPARISON == 'whole-FIT B1 versus FIT/STOP-25%-EPT pilot; no generalization or controlled architecture claim'
    assert metrics.T003_SUMMARY_COMPARISON == 'EPT FIT/STOP 25% pilot versus existing whole-FIT B1; unequal training scope'
    assert not hasattr(adapters, 'T003_COMPARISON') and not hasattr(metrics, 'T003_COMPARISON')       # one name for two texts invited a wrong import
    assert inspect.signature(adapters.run_inference).parameters['comparison'].default is adapters.T003_PROVENANCE_COMPARISON
    assert inspect.signature(metrics.summarize_endpoints).parameters['comparison'].default is metrics.T003_SUMMARY_COMPARISON


@pytest.mark.parametrize('comparison', ['', 3, None])
def test_provenance_comparison_text_is_validated_before_any_store_read(comparison):
    with pytest.raises(ValueError, match='comparison'):
        run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu',
                      comparison=comparison)


# --- EPT-T011 A4-F6: B1 takes all draws of a row in one call; the guard is called once per row ------------------------
# ``legacy_open_draws`` is a frozen copy of the sampler as it was before A4-F6 (``batch_size`` draws per ``objects`` call, ``check()`` before each
# call), returning its intermediate arrays too. The new path must give the same bits, array_equal and no tolerance.
def legacy_open_draws(laws, objects, prices, gold_reps, *, row, match, draws, seed, check, batch_size):
    from c_ept.evaluation.adapters import _law
    from c_ept.valuation import compose
    size, kills, gold = laws
    size = _law('size', size, (4,), True)
    kills = _law('kills', kills, (4, 2))
    gold = _law('gold', gold, (4, 2, 6, 31), True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(match), int(row)]))
    s = compose._inverse_cdf(np.broadcast_to(size, (draws, 4)), rng.random(draws))
    k = rng.poisson(kills[s]).astype(np.int64)
    layer = np.clip(k, 1, 6) - 1
    u = rng.random((draws, 2))
    bins = np.column_stack([compose._inverse_cdf(gold[s, t, layer[:, t]], u[:, t]) for t in (0, 1)])
    g = np.where(k > 0, gold_reps[bins], 0.0)
    K = np.column_stack((k, g))
    rates = np.empty((draws, 18))
    for lo in range(0, draws, batch_size):
        check()
        hi = min(lo + batch_size, draws)
        rates[lo:hi] = objects(s[lo:hi], K[lo:hi])
    rates = _law('objects', rates, (draws, 18))
    O = rng.poisson(rates)
    f = g[:, 0] - g[:, 1] + (O * prices).sum(axis=1)
    plus, draw = float(np.mean(f > 0)), float(np.mean(f == 0))
    se = np.sqrt(plus * (1 - plus) / (draws - 1))
    draw_se = np.sqrt(draw * (1 - draw) / (draws - 1))
    return rates, O, f, (plus, draw, float(se), float(draw_se))


def open_laws():
    size = np.array([.3, .4, .2, .1])
    kills = np.array([[.9, .6], [1.4, 1.1], [2.1, .7], [3.0, 2.5]])
    gold = np.full((4, 2, 6, 31), 1 / 31)
    return size, kills, gold


def real_lightgbm_b1():
    """The fixture's B1 with a real LightGBM M3Obj open head (18 tiny boosters, one thread) instead of its constant fallback."""
    import lightgbm as lgb
    b1 = constant_b1()
    head = b1.open['M3Obj']
    layout = B1.make_layout('M3Obj', 'open', B1.features.b1_columns(()))
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, layout.width)).astype(np.float32)
    params = {'objective': 'poisson', 'num_leaves': 6, 'min_data_in_leaf': 10, 'learning_rate': .2, 'verbose': -1, 'num_threads': 1,
              'deterministic': True, 'force_row_wise': True}
    head.boosters_ = [lgb.train({**params, 'seed': c + 1}, lgb.Dataset(X, label=rng.poisson(.5 + np.abs(X[:, c % 5]))), num_boost_round=6)
                      for c in range(18)]
    head.base_ = np.full(18, np.log(.4))
    return b1


def open_objects(kind, monkeypatch):
    """A batch-independent ``objects(s, K)``: 'fake' is a table lookup with exact arithmetic, 'lightgbm' the real B1OpenHeads.objects with real boosters."""
    if kind == 'fake':
        table = np.random.default_rng(3).uniform(.01, .9, size=(4, 18))
        return lambda s, K: table[s] * (1. + .001 * K[:, :1]) + .0005 * K[:, 3:4]
    monkeypatch.setattr(B1, 'NUM_THREADS', 1)                    # the P9 value is 8; the rows are independent of it
    numeric = np.random.default_rng(8).normal(size=(1, len(B1.features.b1_columns(())))).astype(np.float32)
    heads_ = B1OpenHeads(real_lightgbm_b1())
    return lambda s, K: heads_.objects(np.repeat(numeric, len(s), axis=0), s, K)


@pytest.mark.parametrize('kind', ['fake', 'lightgbm'])
@pytest.mark.parametrize('chunk', [None, 1, 7, 32, 1000])
def test_batch_independent_objects_give_bit_identical_draws_for_every_chunk_and_for_all_draws(kind, chunk, monkeypatch):
    from c_ept.evaluation.adapters import _open_draws
    objects = open_objects(kind, monkeypatch)
    prices = np.linspace(-3, 3, 18)
    reps = np.arange(31.) * 100
    kw = dict(row=17, match=9, draws=1000, seed=20261022, check=lambda: None)
    rates, O, f, summary = legacy_open_draws(open_laws(), objects, prices, reps, batch_size=32, **kw)         # the old path
    n_rates, n_O, n_f = _open_draws(open_laws(), objects, prices, reps, batch_size=chunk, **kw)
    assert np.array_equal(rates, n_rates) and np.array_equal(O, n_O) and np.array_equal(f, n_f)
    assert rates.dtype == n_rates.dtype and O.dtype == n_O.dtype and f.dtype == n_f.dtype
    new = sample_open_value(open_laws(), objects, prices, reps, batch_size=chunk, **kw)
    assert new == summary                                                    # plus, draw and both standard errors, to the last bit
    assert (new[0], new[1]) == (float(np.mean(f > 0)), float(np.mean(f == 0)))


def test_a_batch_dependent_model_does_change_with_the_chunk_so_the_comparison_can_fail():
    """Control of the test above and the reason the neural path keeps batch_size: if the rows of a model depend on the batch of the call (as a GPU
    kernel can), the arrays differ between chunk 32 and all draws."""
    from c_ept.evaluation.adapters import _open_draws
    table = np.random.default_rng(3).uniform(.01, .9, size=(4, 18))
    kw = dict(row=17, match=9, draws=1000, seed=20261022, check=lambda: None)
    dependent = lambda s, K: table[s] * (1. + 1e-9 * len(s))                                  # noqa: E731
    a = _open_draws(open_laws(), dependent, np.ones(18), np.arange(31.), batch_size=32, **kw)[0]
    b = _open_draws(open_laws(), dependent, np.ones(18), np.arange(31.), batch_size=None, **kw)[0]
    assert not np.array_equal(a, b) and np.allclose(a, b, rtol=1e-6)


def test_check_is_called_once_per_row_and_before_the_first_objects_call():
    order = []
    objects = lambda s, K: (order.append('objects'), np.full((len(s), 18), .1))[1]           # noqa: E731
    for batch_size in (32, 7, None):
        order.clear()
        sample_open_value(open_laws(), objects, np.ones(18), np.arange(31.), row=1, match=2, draws=100, seed=3,
                          check=lambda: order.append('check'), batch_size=batch_size)
        assert order.count('check') == 1 and order[0] == 'check'                  # was one call for every chunk of batch_size draws
        assert order.count('objects') == (1 if batch_size is None else -(-100 // batch_size))


@pytest.mark.parametrize('bad', [0, -1, True, 2.5, '8'])
def test_the_monte_carlo_chunk_is_a_positive_integer_or_none(bad):
    with pytest.raises(ValueError, match='batch_size'):
        sample_open_value(open_laws(), lambda s, K: np.full((len(s), 18), .1), np.ones(18), np.arange(31.), row=1, match=2, draws=10, seed=3,
                          check=lambda: None, batch_size=bad)


@pytest.mark.parametrize('batch_size', [None, 5])
@pytest.mark.parametrize('bad', [0, 1, -3, True, 2.5, '100'])
def test_the_monte_carlo_draw_count_is_an_integer_of_at_least_two_with_a_clear_message(bad, batch_size):
    """EPT-T011 final cleanup, m9: ``draws=0`` with ``batch_size=None`` made the chunk 0 and ``range`` said "arg 3 must not be zero". The check is the one of
    ``run_inference`` (an integer of at least 2: the standard error divides by ``draws - 1``), so a direct call fails as clearly as the run does."""
    from c_ept.evaluation.adapters import _open_draws
    objects = lambda s, K: np.full((len(s), 18), .1)                                         # noqa: E731
    kw = dict(row=1, match=2, seed=3, check=lambda: None, batch_size=batch_size)
    for call in (sample_open_value, _open_draws):
        with pytest.raises(ValueError, match='draws must be an integer >= 2'):
            call(open_laws(), objects, np.ones(18), np.arange(31.), draws=bad, **kw)


def test_the_monte_carlo_draw_count_check_leaves_the_smallest_valid_count_alone():
    objects = lambda s, K: np.full((len(s), 18), .1)                                         # noqa: E731
    p, draw, se, draw_se = sample_open_value(open_laws(), objects, np.ones(18), np.arange(31.), row=1, match=2, draws=2, seed=3,
                                             check=lambda: None, batch_size=None)
    assert 0. <= p <= 1. and 0. <= draw <= 1. and np.isfinite([se, draw_se]).all()


def test_neural_monte_carlo_keeps_the_batch_size_chunks_and_b1_takes_every_draw_in_one_call(synthetic, monkeypatch):
    import c_ept.evaluation.adapters as adapters
    bundle, model, stats, b1 = synthetic
    seen, real = [], adapters.sample_open_value

    def spy(laws, objects, prices, gold_reps, **kw):
        sizes = []

        def counting(s, K):
            sizes.append(len(s))
            return objects(s, K)
        seen.append((kw['batch_size'], sizes))
        return real(laws, counting, prices, gold_reps, **kw)
    monkeypatch.setattr(adapters, 'sample_open_value', spy)
    run_inference(bundle, model, stats, b1, check=lambda: None, device='cpu', batch_size=5, draws=12)
    assert seen and len(seen) % 2 == 0                                       # per row: the EPT call, then the B1 call
    for (ept_batch, ept_sizes), (b1_batch, b1_sizes) in zip(seen[0::2], seen[1::2]):
        assert ept_batch == 5 and ept_sizes == [5, 5, 2]                    # the neural chunking is the run's batch_size, as before
        assert b1_batch is None and b1_sizes == [12]                        # B1: every draw of the row in one call


# --- S4b-prep M2, decision D5 tie rule (i): the truth of the winner endpoint on the binned gold ---------------------------------------------
# ``winner_truth='gold'`` (the default) reads labels/f0_unit as every earlier run did; ``'binned'`` computes the gold term of the truth on the representatives of
# the gold bins (labels/gold_bin, labels/gold_reps), the variable the models use, so that two teams in one bin tie on the gold term on both sides; the objective
# term stays the realized attributed prices of labels/obj_value (a tower at its realized tier price, where the models price it at the expected tier price). Only the
# truth of the M3 sign endpoint may change. The heads read labels/gold_bin too, so these tests leave it as the toy world made it and vary labels/obj_value, which no
# head reads.
WINNER_TRUTH_RUN = dict(check=lambda: None, device='cpu', draws=12, scales={k: 2. for k in loss.BLOCKS})
# The DEV open units of the toy world are 19 .. 26. Their bins (read off the toy world; the second test,
# test_binned_winner_truth_changes_only_the_truth_of_the_m3_sign_endpoint, pins them as TOY_DEV_BINS) and, with the representatives
# 100 b of the fixture, the gold part of the binned f of each unit by hand:
#   unit | bins     | rep blue - rep red
#    19  | 16, 23   | 1600 - 2300 = -700
#    20  | -1, -1   | 0 - 0 (no kill on either side)
#    21  |  7,  7   | 700 - 700 = 0 (the same bin: the pair that the raw gold decides, 775 against 770)
#    22  | 30, 13   | 3000 - 1300 = 1700
#    23  | 22, 30   | 2200 - 3000 = -800
#    24  | 13, 21   | 1300 - 2100 = -800
#    25  | -1, 14   | 0 - 1400 = -1400
#    26  | 19, -1   | 1900 - 0 = 1900
TOY_DEV_BINS = [[16, 23], [-1, -1], [7, 7], [30, 13], [22, 30], [13, 21], [-1, 14], [19, -1]]
# the objective values given to the eight units, in the order of the units: an equal pair on the unit without a kill (still a tie), 30 for blue on the pair of
# one bin (the objective breaks the tie), 15 for red on a pair that red leads, and 1400 for blue that cancels the 1400 of red's gold exactly (a tie by objectives)
TOY_DEV_OBJ = [[0., 0.], [12.5, 12.5], [30., 0.], [0., 0.], [0., 15.], [0., 0.], [1400., 0.], [0., 0.]]
TOY_DEV_BINNED_F = [-700.0, 0.0, 30.0, 1700.0, -815.0, -800.0, 0.0, 1900.0]


def assert_same(a, b, path=''):
    """Equality of two ``run_inference`` results with no tolerance: arrays by dtype, shape and bytes, everything else by type and value."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            assert_same(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f'{path}[{i}]')
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), path
    else:
        assert type(a) is type(b) and (a == b or (a != a and b != b)), path


def dev_open_units(bundle):
    match = np.asarray(bundle.s1.array('units/match'))
    scope = np.asarray(bundle.s1.array('units/in_scope')).astype(bool)
    return np.flatnonzero((match == 2) & scope)


def with_obj_value(bundle, obj):
    """The synthetic bundle with the labels/obj_value array ``obj`` (n_units, 2)."""
    out = SimpleNamespace(**vars(bundle))
    out.labels = Overlay(bundle.labels, {'labels/obj_value': obj})
    return out


def toy_obj_value(bundle, values=TOY_DEV_OBJ, fill=0.0):
    """labels/obj_value that gives the DEV open units the ``values`` and every other unit ``fill``."""
    obj = np.full((len(bundle.s1.array('units/f')), 2), fill)
    obj[dev_open_units(bundle)] = values
    return obj


def oracle_binned_f(bins, obj, reps, unit):
    """The binned f of one unit in plain Python: a team with a kill has the representative of its bin, a team without has 0, the objectives are added after."""
    gold = [0.0 if bins[unit][t] < 0 else float(reps[bins[unit][t]]) for t in (0, 1)]
    return (gold[0] + obj[unit][0]) - (gold[1] + obj[unit][1])


def test_winner_truth_default_is_gold_and_explicit_gold_is_bit_identical_in_every_field(synthetic):
    import inspect
    bundle, model, stats, b1 = synthetic
    assert inspect.signature(run_inference).parameters['winner_truth'].default == 'gold'
    default = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    explicit = run_inference(bundle, model, stats, b1, winner_truth='gold', **WINNER_TRUTH_RUN)
    assert_same(default, explicit)
    assert 'winner_truth' not in default['provenance'] and 'winner_truth_note' not in default['provenance']       # absent = the default (as kill_law)
    f0 = np.asarray(bundle.labels.array('labels/f0_unit'))
    m3 = default['endpoints'][3]
    assert m3['y'].tolist() == (f0[m3['row']] > 0).tolist() and m3['draw_truth'].tolist() == (f0[m3['row']] == 0).tolist()      # the raw f0, as before
    # the default reads no label array that earlier runs did not read: labels/obj_value is poison here and the result is the same
    poisoned = with_obj_value(bundle, np.full((len(bundle.s1.array('units/f')), 2), np.nan))
    assert_same(default, run_inference(poisoned, model, stats, b1, **WINNER_TRUTH_RUN))
    assert_same(default, run_inference(poisoned, model, stats, b1, winner_truth='gold', **WINNER_TRUTH_RUN))


def test_binned_winner_truth_changes_only_the_truth_of_the_m3_sign_endpoint(synthetic):
    bundle, model, stats, b1 = synthetic
    labelled = with_obj_value(bundle, toy_obj_value(bundle))
    gold = run_inference(labelled, model, stats, b1, **WINNER_TRUTH_RUN)
    binned = run_inference(labelled, model, stats, b1, winner_truth='binned', **WINNER_TRUTH_RUN)
    assert len(binned['endpoints']) == len(gold['endpoints']) == 10
    for i, (g, b) in enumerate(zip(gold['endpoints'], binned['endpoints'])):
        if i != 3:
            assert_same(g, b, f'endpoint {i}')                                   # the predictions and the truth of every other endpoint
    g, b = gold['endpoints'][3], binned['endpoints'][3]
    assert g.keys() == b.keys()
    for key in g:
        if key not in ('y', 'draw_truth', 'counts'):
            assert_same(g[key], b[key], key)                                     # the predictions p+, p_draw, their Monte Carlo errors, the rows, the matches
    assert b['row'].tolist() == dev_open_units(bundle).tolist()
    bins = np.asarray(bundle.labels.array('labels/gold_bin'))
    assert bins[b['row']].tolist() == TOY_DEV_BINS                               # the premise of the hand numbers above
    reps = np.asarray(bundle.labels.array('labels/gold_reps'))
    obj = np.asarray(labelled.labels.array('labels/obj_value'))
    want = np.array([oracle_binned_f(bins, obj, reps, int(u)) for u in b['row']])
    assert want.tolist() == TOY_DEV_BINNED_F
    assert b['y'].tolist() == [False, False, True, True, False, False, False, True]          # f > 0: blue wins; unit 21 is won by blue through its objective
    assert b['draw_truth'].tolist() == [False, True, False, False, False, False, True, False]     # f == 0: units 20 and 25 tie
    f0 = np.asarray(bundle.labels.array('labels/f0_unit'))[b['row']]
    assert g['y'].tolist() == (f0 > 0).tolist() and g['draw_truth'].tolist() == (f0 == 0).tolist()     # the default is the raw f0 of the labels
    assert b['y'].tolist() != g['y'].tolist() and b['draw_truth'].tolist() != g['draw_truth'].tolist()   # the rule changed both truths here
    assert {k: v for k, v in b['counts'].items() if k not in ('positive', 'negative', 'draw')} == \
        {k: v for k, v in g['counts'].items() if k not in ('positive', 'negative', 'draw')}
    assert (b['counts']['positive'], b['counts']['negative'], b['counts']['draw']) == (3, 5, 2)
    assert_same(gold['nll'], binned['nll'])                                      # no NLL moves: the truth of the winner is no training or scoring target
    provenance = dict(binned['provenance'])
    assert provenance.pop('winner_truth') == 'binned'
    note = provenance.pop('winner_truth_note')
    # the note says what is the models' variable (the gold term) and what is not (the objective term: realized prices, a tower at its realized tier price)
    assert 'D5' in note and 'gold term' in note and 'realized' in note and 'expected tier price' in note
    assert 'variable the models decide the winner on' not in note
    assert_same(provenance, gold['provenance'])


@pytest.mark.parametrize('rule', ['Gold', 'raw', '', None, 3, ('gold',), b'gold'])
def test_an_unknown_winner_truth_is_refused_before_any_store_read(rule):
    with pytest.raises(ValueError, match='winner_truth'):
        run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu', winner_truth=rule)


def test_binned_truth_labels_are_validated_before_the_first_head_and_only_at_the_dev_open_units(synthetic, monkeypatch):
    bundle, model, stats, b1 = synthetic
    open_units = dev_open_units(bundle)
    outside = np.flatnonzero(np.asarray(bundle.s1.array('units/match')) != 2)[0]
    calls = []
    original = b1.masked['M2Rate'].predict
    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(b1.masked['M2Rate'], 'predict', spy)
    for what, bad, unit in (('NaN objectives', np.nan, open_units[0]), ('infinite objectives', np.inf, open_units[-1]),
                            ('negative infinite objectives', -np.inf, open_units[3])):
        obj = toy_obj_value(bundle)
        obj[unit, 1] = bad
        with pytest.raises(ValueError, match='winner_truth'):
            run_inference(with_obj_value(bundle, obj), model, stats, b1, winner_truth='binned', **WINNER_TRUTH_RUN)
        assert calls == [], what                                                 # refused before the first head ran, as the f0 check is
        run_inference(with_obj_value(bundle, obj), model, stats, b1, **WINNER_TRUTH_RUN)         # the default does not read the array: no refusal
        calls.clear()
    # a unit outside the DEV open population holds poison: it is not read, so the early check is no wider than the units the truth uses
    poison = toy_obj_value(bundle)
    poison[outside] = np.nan
    out = run_inference(with_obj_value(bundle, poison), model, stats, b1, winner_truth='binned', **WINNER_TRUTH_RUN)
    assert calls and len(out['endpoints']) == 10
    assert_same(out, run_inference(with_obj_value(bundle, toy_obj_value(bundle)), model, stats, b1, winner_truth='binned', **WINNER_TRUTH_RUN))


def test_binned_truth_with_an_empty_dev_open_population_has_an_empty_truth_and_no_error(synthetic):
    """No in-scope DEV unit: the M3 sign endpoint is empty under either rule and the rule 'binned' neither reads a unit nor refuses (review m5)."""
    bundle, model, stats, b1 = synthetic
    empty = SimpleNamespace(**vars(bundle))
    empty.s1 = Overlay(bundle.s1, {'units/in_scope': np.zeros(len(bundle.s1.array('units/f')), dtype=bool)})
    gold = run_inference(empty, model, stats, b1, **WINNER_TRUTH_RUN)
    binned = run_inference(with_obj_value(empty, toy_obj_value(bundle)), model, stats, b1, winner_truth='binned', **WINNER_TRUTH_RUN)
    for out in (gold, binned):
        m3 = out['endpoints'][3]
        assert m3['row'].size == 0 and m3['y'].size == 0 and m3['draw_truth'].size == 0
        assert m3['counts']['eligible'] == 0 and m3['counts']['unavailable_reason'] == 'no_DEV_in_scope_units'
    assert_same(gold['endpoints'], binned['endpoints'])
    assert_same(gold['nll'], binned['nll'])
    provenance = dict(binned['provenance'])
    assert provenance.pop('winner_truth') == 'binned' and 'winner_truth_note' in provenance
    provenance.pop('winner_truth_note')
    assert_same(provenance, gold['provenance'])


def test_binned_truth_still_refuses_an_f0_that_is_missing_before_the_first_head_runs(synthetic, monkeypatch):
    """The labels/f0_unit check of the default is kept under the rule 'binned': a unit that is not valued has no f0, no bins and no objectives."""
    bundle, model, stats, b1 = synthetic
    calls = []
    original = b1.masked['M2Rate'].predict
    monkeypatch.setattr(b1.masked['M2Rate'], 'predict', lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    with pytest.raises(ValueError, match='nonfinite M3 f0 truth'):
        run_inference(poisoned_f0(with_obj_value(bundle, toy_obj_value(bundle)), np.nan, dev_open=True), model, stats, b1,
                      winner_truth='binned', **WINNER_TRUTH_RUN)
    assert calls == []


# --- EPT-T012 S4b preparation M7 (decisions D10 P1, D17): the evaluation keeps the two-frame average unless the checkpoint's identity says eval_average=False -------------------------
def test_eval_average_is_true_by_default_and_false_evaluates_the_recorded_frame_only(synthetic, monkeypatch):
    """The model object carries no identity, so ``run_inference`` takes ``eval_average`` as the other settings of the training identity (kill_law, k_condition): the caller (the kernel, K6)
    passes ``identity.get('eval_average', True)``. The default and the explicit True are the same result in every field and encode every chunk twice; False encodes every view once, as recorded,
    changes only the numbers of the EPT (B1 is untouched) and says so in the provenance (absent means True)."""
    import inspect
    bundle, model, stats, b1 = synthetic
    assert inspect.signature(run_inference).parameters['eval_average'].default is True
    calls, real = [], heads.EPT.encode
    monkeypatch.setattr(heads.EPT, 'encode', lambda self, batch: (calls.append(batch['side'].clone()), real(self, batch))[1])
    default = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    twice = len(calls)
    calls.clear()
    explicit = run_inference(bundle, model, stats, b1, eval_average=True, **WINNER_TRUTH_RUN)
    assert_same(default, explicit) and len(calls) == twice
    assert twice > 0 and twice % 2 == 0 and default['provenance']['frame_averaged'] is True and 'eval_average' not in default['provenance']
    calls.clear()
    single = run_inference(bundle, model, stats, b1, eval_average=False, **WINNER_TRUTH_RUN)
    assert len(calls) * 2 == twice and not any(side.any() for side in calls)                      # every view once, in the recorded frame
    assert single['provenance']['frame_averaged'] is False and single['provenance']['eval_average'] is False and 'eval_average_note' in single['provenance']
    assert {k: v for k, v in single['provenance'].items() if k not in ('frame_averaged', 'eval_average', 'eval_average_note')} == {k: v for k, v in default['provenance'].items() if k != 'frame_averaged'}
    for a, b in zip(default['nll'], single['nll']):
        assert np.array_equal(a['nll_b1'], b['nll_b1'])
    assert any(not np.array_equal(a['nll_ept'], b['nll_ept']) for a, b in zip(default['nll'], single['nll']))      # the EPT scores moved (control)


def test_a_frame_averaged_model_must_agree_with_eval_average_and_eval_average_must_be_a_bool(synthetic):
    bundle, model, stats, b1 = synthetic
    one = heads.FrameAveraged(model, average=False)
    with pytest.raises(ValueError, match='eval_average'):
        run_inference(bundle, one, stats, b1, **WINNER_TRUTH_RUN)                                 # a one-frame wrapper with the default True
    with pytest.raises(ValueError, match='eval_average'):
        run_inference(bundle, heads.FrameAveraged(model), stats, b1, eval_average=False, **WINNER_TRUTH_RUN)
    accepted = run_inference(bundle, one, stats, b1, eval_average=False, **WINNER_TRUTH_RUN)
    assert_same(accepted, run_inference(bundle, model, stats, b1, eval_average=False, **WINNER_TRUTH_RUN))
    for bad in (None, 0, 1, 'no'):
        with pytest.raises(ValueError, match='eval_average'):
            run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu', eval_average=bad)


# --- EPT-T012 S4b preparation M9 (decision D9): the evaluation of a model without the zone mark -------------------------------------------------------------------------------------------
class _PoisonStore:
    def __getattr__(self, name):
        raise AssertionError(f'a store was read ({name}) before the zone_mark check')


def zone_off_model():
    """The model of the ``synthetic`` fixture (same seed and size) built without the zone mark."""
    torch.manual_seed(17)
    return heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=.1, kappas=1, field_channels=2), toy_static(), zone_mark=False)


def test_zone_mark_is_true_by_default_must_be_a_bool_and_must_agree_with_the_model_before_any_store_read(synthetic):
    """The model object carries no identity, but it knows its own ``zone_mark`` (``heads.zone_mark``). ``run_inference`` takes the setting of the checkpoint's training identity (the kernel, K6,
    passes ``identity.get('zone_mark', True)``) and refuses a model that declares the other one, before any store is read; a value that is not a bool is a ValueError before the role check."""
    import inspect
    bundle, model, stats, b1 = synthetic
    assert inspect.signature(run_inference).parameters['zone_mark'].default is True
    off = zone_off_model()
    poisoned = SimpleNamespace(role='DEV', s1=_PoisonStore(), s2=_PoisonStore(), labels=_PoisonStore(), prices=bundle.prices, tier=bundle.tier)
    for candidate, asked in ((model, False), (off, True), (heads.FrameAveraged(off), True), (heads.FrameAveraged(model), False)):
        with pytest.raises(ValueError, match='zone_mark'):
            run_inference(poisoned, candidate, stats, b1, zone_mark=asked, **WINNER_TRUTH_RUN)
    for bad in (None, 0, 1, 'no'):
        with pytest.raises(ValueError, match='zone_mark'):
            run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu', zone_mark=bad)


def test_zone_mark_false_leaves_the_zone_block_unavailable_and_says_so_and_true_is_the_default(synthetic):
    """With the zone mark off the M2Zone/masked block is not scored (its record is the empty 'unavailable' one, so the thirteen blocks stay and a consumer finds the reason), the other twelve blocks are
    scored, and the provenance says ``zone_mark`` False (absent means True). The explicit True is the default result in every field."""
    bundle, model, stats, b1 = synthetic
    off = zone_off_model()
    default = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    assert_same(default, run_inference(bundle, model, stats, b1, zone_mark=True, **WINNER_TRUTH_RUN))
    assert 'zone_mark' not in default['provenance'] and 'zone_mark_note' not in default['provenance']
    result = run_inference(bundle, off, stats, b1, zone_mark=False, **WINNER_TRUTH_RUN)
    assert result['provenance']['zone_mark'] is False and 'zone_mark_note' in result['provenance']
    assert {k: v for k, v in result['provenance'].items() if k not in ('zone_mark', 'zone_mark_note')} == default['provenance']
    names = [r['name'] for r in result['nll']]
    assert names == [r['name'] for r in default['nll']] and len(names) == 13
    zone = next(r for r in result['nll'] if r['name'] == 'M2Zone/masked')
    assert zone['counts']['unavailable_reason'] == 'zone_mark_off' and zone['counts']['eligible'] == 0 and len(zone['row']) == 0 and len(zone['nll_ept']) == 0 and len(zone['nll_b1']) == 0
    scored = next(r for r in default['nll'] if r['name'] == 'M2Zone/masked')
    assert len(scored['row']) > 0                                                             # control: the default scores the block
    for a, b in zip(default['nll'], result['nll']):
        if a['name'] != 'M2Zone/masked':
            assert b['counts'].get('unavailable_reason') == a['counts'].get('unavailable_reason') and len(b['row']) == len(a['row']) and np.array_equal(a['nll_b1'], b['nll_b1']), a['name']
    assert len(result['endpoints']) == len(default['endpoints'])


# --- EPT-T012 S4b preparation M12 (decision D14): the exposure and the population of the t0 forecasts of run_inference -------------------------------------------------------------------
M2_NAMES = ('M2/t0/H60/occurrence', 'M2/t0/H120/occurrence', 'M2/t0/H180/occurrence')
# The DEV match (match 2) of the toy world has the frame rows 22 .. 32 with the windows hi - t0 of 180 s (22 .. 29), 162.8 s (30), 102.8 s (31) and 42.8 s (32): by hand, the rows primary at H = 60 / 120 / 180
# are 10 / 9 / 8 and the secondary ones 1 / 2 / 3.
TOY_PRIMARY_COUNTS = (10, 9, 8)


def m2_of(result):
    return [e for e in result['endpoints'] if e['name'] in M2_NAMES]


def others_of(result):
    return [e for e in result['endpoints'] if e['name'] not in M2_NAMES]


def dev_window_ends(bundle, rows):
    """``(t0, hi)`` in whole ms of the frame rows ``rows``, read from the store."""
    return np.asarray(bundle.s1.array('frame_rows/t0'))[rows], np.asarray(bundle.s1.array('frame_rows/hi'))[rows]


def test_exposure_is_observed_and_population_is_all_by_default_and_the_explicit_defaults_are_the_same_bits(synthetic):
    import inspect
    bundle, model, stats, b1 = synthetic
    parameters = inspect.signature(run_inference).parameters
    assert parameters['exposure'].default == 'observed' and parameters['population'].default == 'all'
    assert parameters['exposure'].kind is inspect.Parameter.KEYWORD_ONLY and parameters['population'].kind is inspect.Parameter.KEYWORD_ONLY
    default = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    assert_same(default, run_inference(bundle, model, stats, b1, exposure='observed', population='all', **WINNER_TRUTH_RUN))
    assert not {'exposure', 'exposure_note', 'population', 'population_note'} & set(default['provenance'])               # absent means observed and all
    assert all('population_selected' not in e['counts'] for e in m2_of(default)) and [e['name'] for e in m2_of(default)] == list(M2_NAMES)


@pytest.mark.parametrize('key, bad', [('exposure', None), ('exposure', 'Nominal'), ('exposure', ''), ('exposure', 1), ('exposure', True),
                                       ('population', None), ('population', 'Primary'), ('population', ''), ('population', 0), ('population', False)])
def test_exposure_and_population_are_checked_before_any_store_read(key, bad):
    with pytest.raises(ValueError, match=key):
        run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu', **{key: bad})


@pytest.mark.parametrize('population', ['all', 'secondary'])
def test_the_nominal_exposure_is_refused_for_any_population_but_the_primary_one_before_any_store_read(population):
    """Nominal exposures are the exposures of the rows whose horizon the game end does not cut: on the others the window would have to be read from T. Refused, naming T, before any store is read."""
    with pytest.raises(ValueError, match='primary'):
        run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu', exposure='nominal', population=population)


def test_the_primary_and_secondary_populations_partition_the_m2_endpoints_and_leave_everything_else_alone(synthetic):
    bundle, model, stats, b1 = synthetic
    default = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    primary = run_inference(bundle, model, stats, b1, population='primary', **WINNER_TRUTH_RUN)
    secondary = run_inference(bundle, model, stats, b1, population='secondary', **WINNER_TRUTH_RUN)
    for name, result in (('primary', primary), ('secondary', secondary)):
        assert_same(others_of(default), others_of(result))                       # the M3 and M4 endpoints are delta-conditional: they keep the observed exposure and no population
        assert_same(default['nll'], result['nll'])                               # the NLL of every block is untouched (a likelihood, not a forecast)
        assert result['provenance']['population'] == name and 'population_note' in result['provenance'] and 'exposure' not in result['provenance']
        assert {k: v for k, v in result['provenance'].items() if k not in ('population', 'population_note')} == default['provenance']
    for h, d, p, s, n_primary in zip((60, 120, 180), m2_of(default), m2_of(primary), m2_of(secondary), TOY_PRIMARY_COUNTS):
        t0, hi = dev_window_ends(bundle, d['row'])
        by_hand = (hi - t0) >= h * 1000                                          # whole ms, from the store
        assert int(by_hand.sum()) == n_primary and len(d['row']) == 11
        assert p['row'].tolist() == d['row'][by_hand].tolist() and s['row'].tolist() == d['row'][~by_hand].tolist()
        for field in ('match', 'y', 'p_ept', 'p_b1'):
            assert np.array_equal(p[field], d[field][by_hand]) and np.array_equal(s[field], d[field][~by_hand]), (h, field)      # the same numbers on the same rows: the observed exposure of both comparators
        assert p['counts']['population_selected'] == n_primary and s['counts']['population_selected'] == 11 - n_primary
        assert p['counts']['population_candidate'] == s['counts']['population_candidate'] == 11 and p['counts']['clipped_window'] == 0
        assert s['counts']['clipped_window'] == 11 - n_primary and p['counts']['eligible'] == n_primary
        assert d['counts']['candidate'] == p['counts']['candidate'] == s['counts']['candidate']
    assert [len(e['row']) for e in m2_of(primary)] == list(TOY_PRIMARY_COUNTS) and [len(e['row']) for e in m2_of(secondary)] == [1, 2, 3]


def test_nominal_exposure_on_the_primary_population_is_the_primary_result_and_says_so_in_the_provenance(synthetic):
    """On the primary rows the observed exposure of the first H/10 bins is the nominal 10 s, so the nominal evaluation gives the same endpoints, bit for bit, as the observed one on the same rows; only the
    provenance (and nothing else) says that the forecast was T-blind."""
    bundle, model, stats, b1 = synthetic
    observed = run_inference(bundle, model, stats, b1, population='primary', **WINNER_TRUTH_RUN)
    nominal = run_inference(bundle, model, stats, b1, exposure='nominal', population='primary', **WINNER_TRUTH_RUN)
    assert_same(observed['endpoints'], nominal['endpoints'])
    assert_same(observed['nll'], nominal['nll'])
    assert nominal['provenance']['exposure'] == 'nominal' and 'exposure_note' in nominal['provenance'] and nominal['provenance']['population'] == 'primary'
    assert {k: v for k, v in nominal['provenance'].items() if k not in ('exposure', 'exposure_note')} == observed['provenance']
    assert 'population_selected' in nominal['endpoints'][0]['counts']


def with_exposure_cut(bundle, rows, upto_bin=0):
    """The synthetic bundle in which the frame rows ``rows`` have no observed exposure from the bin ``upto_bin`` on (and the window end ``frame_rows/hi`` = t0 + the exposure that is left)."""
    t0, ex = np.asarray(bundle.s1.array('frame_rows/t0')), np.array(bundle.s1.array('frame_rows/exposure_ms'))
    hi = np.array(bundle.s1.array('frame_rows/hi'))
    ex[rows, upto_bin:] = 0
    hi[rows] = t0[rows] + ex[rows].sum(axis=1)
    out = SimpleNamespace(**vars(bundle))
    out.s1 = Overlay(bundle.s1, {'frame_rows/exposure_ms': ex, 'frame_rows/hi': hi})
    return out


@pytest.mark.parametrize('population', ['primary', 'secondary'])
def test_population_candidate_is_the_candidate_count_also_for_rows_without_exposure_and_for_an_empty_block(synthetic, population):
    """M12 review m2: ``population_candidate`` and ``candidate`` have one denominator, every frame row of the role, including the rows without any exposure that the block leaves out (neither population can select them);
    ``population_selected`` counts the rows that the mask selects among the rows of the block. An M2 block that is empty under a population still carries both keys (selected 0)."""
    bundle, model, stats, b1 = synthetic
    cut = with_exposure_cut(bundle, [32], 0)                                         # the frame row 32 (the last DEV row) has no exposure at all: it is left out of the block
    result = run_inference(cut, model, stats, b1, population=population, **WINNER_TRUTH_RUN)
    wanted = list(TOY_PRIMARY_COUNTS) if population == 'primary' else [0, 1, 2]       # row 32 was a secondary row at every horizon and is no row of the block now
    for ep, selected in zip(m2_of(result), wanted):
        c = ep['counts']
        assert c['candidate'] == c['population_candidate'] == 11 and c['zero_exposure'] == 1 and c['population_selected'] == selected, c
    nothing = with_exposure_cut(bundle, list(range(22, 33)), 0)                      # every DEV row without exposure (and so without onsets): the M2 block is empty
    onsets = {name: np.asarray(nothing.s1.array(f'onsets/{name}')) for name in ('row', 'bin', 'unit', 'mark')}
    keep = onsets['row'] < 22
    nothing.s1 = Overlay(nothing.s1, {f'onsets/{name}': a[keep] for name, a in onsets.items()})
    for ep in m2_of(run_inference(nothing, model, stats, b1, population=population, **WINNER_TRUTH_RUN)):
        c = ep['counts']
        assert len(ep['row']) == 0 and c['candidate'] == c['population_candidate'] == 11 and c['population_selected'] == 0 and 'unavailable_reason' in c, c


def test_the_nominal_exposure_refuses_a_store_whose_window_ends_and_exposures_disagree(synthetic):
    """The nominal 10 s is the observed exposure on the primary rows; a store whose ``frame_rows/hi`` says a row is primary while its ``frame_rows/exposure_ms`` is cut would make the nominal numbers differ from the observed
    ones, and is refused (the observed evaluation does not check it: it uses the exposure it is given)."""
    bundle, model, stats, b1 = synthetic
    t0, match = np.asarray(bundle.s1.array('frame_rows/t0')), np.asarray(bundle.s1.array('frame_rows/match'))
    hi = np.array(bundle.s1.array('frame_rows/hi'))
    hi[match == 2] = t0[match == 2] + 180_000                                       # every DEV row claims the whole 180 s window; the exposure of the rows 30 .. 32 says otherwise
    liar = SimpleNamespace(**vars(bundle))
    liar.s1 = Overlay(bundle.s1, {'frame_rows/hi': hi})
    with pytest.raises(ValueError, match='disagree'):
        run_inference(liar, model, stats, b1, exposure='nominal', population='primary', **WINNER_TRUTH_RUN)
    run_inference(liar, model, stats, b1, population='primary', **WINNER_TRUTH_RUN)


def longer_game(bundle, extra_ms=100_000):
    """The synthetic bundle in which the DEV match (match 2) ends ``extra_ms`` later: ``matches/T_ms``, ``frame_rows/hi`` and ``frame_rows/exposure_ms`` of its frame rows as ``frame_rows`` makes them. The onsets and
    the labels stay: the first H/10 bins of the rows that are primary in both versions are the same bins with the same counts."""
    s1 = bundle.s1
    match, t0 = np.asarray(s1.array('frame_rows/match')), np.asarray(s1.array('frame_rows/t0'))
    hi, exposure = np.array(s1.array('frame_rows/hi')), np.array(s1.array('frame_rows/exposure_ms'))
    T = np.array(s1.array('matches/T_ms'))
    T[2] += extra_ms
    rows = np.flatnonzero(match == 2)
    hi[rows] = np.minimum(t0[rows] + 180_000, T[2] - 30_000)
    exposure[rows] = np.clip(hi[rows, None] - (t0[rows, None] + 10_000 * np.arange(18)[None, :]), 0, 10_000)
    out = SimpleNamespace(**vars(bundle))
    out.s1 = Overlay(s1, {'matches/T_ms': T, 'frame_rows/hi': hi, 'frame_rows/exposure_ms': exposure})
    return out


def test_nominal_primary_endpoints_are_bit_identical_when_the_game_end_changes_and_the_observed_ones_are_not(synthetic):
    """D14: T enters the evaluation only through the population filter. The game of the DEV match is made 100 s longer: the frame rows that are primary at H = 60 s in both versions (22 .. 31) have bit-identical nominal
    endpoints; the observed evaluation of all rows moves (the row 31 had a window of 102.8 s at H = 120 s and has the whole window now)."""
    bundle, model, stats, b1 = synthetic
    longer = longer_game(bundle)
    old = run_inference(bundle, model, stats, b1, exposure='nominal', population='primary', **WINNER_TRUTH_RUN)
    new = run_inference(longer, model, stats, b1, exposure='nominal', population='primary', **WINNER_TRUTH_RUN)
    a, b = m2_of(old)[0], m2_of(new)[0]                                           # H = 60 s
    common = np.isin(b['row'], a['row'])
    assert a['row'].tolist() == list(range(22, 32)) and b['row'].tolist() == list(range(22, 33)) and common.sum() == 10
    for field in ('match', 'y', 'p_ept', 'p_b1'):
        assert np.array_equal(a[field], b[field][common]), field
    default_old = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    default_new = run_inference(longer, model, stats, b1, **WINNER_TRUTH_RUN)
    h120_old, h120_new = m2_of(default_old)[1], m2_of(default_new)[1]
    i = h120_old['row'].tolist().index(31)
    j = h120_new['row'].tolist().index(31)
    assert h120_old['p_b1'][i] != h120_new['p_b1'][j]                                # the constant B1 rate .02 per s: 1 - exp(-.02 x 102.8) against 1 - exp(-.02 x 120) (the EPT probability of the toy is saturated at 1)
    assert h120_old['p_b1'][i] == pytest.approx(-np.expm1(-.02 * 102.8), rel=1e-6) and h120_new['p_b1'][j] == pytest.approx(-np.expm1(-.02 * 120.0), rel=1e-6)


# --- EPT-T012 S4b preparation M11 (decision D16): the heads of the relative structural metric are the structural heads of the adapter ---------------------------------------------------
def test_the_relative_metric_averages_the_structural_heads_of_the_adapter():
    from c_ept.evaluation import adapters, metrics
    assert metrics.RELATIVE_HEADS == adapters.STRUCTURAL_HEADS and len(metrics.RELATIVE_HEADS) == 8


def test_records_with_nll_b0_leave_the_summary_of_the_old_columns_unchanged(synthetic):
    """The training-scale composite and every per-head summary are secondary and unchanged: ``summarize_endpoints`` reads only the model and B1 columns, so a record with ``nll_b0`` summarises exactly
    as the same record without it (the records are those of a real run, with a stand-in ``nll_b0`` column added)."""
    from c_ept.evaluation.metrics import summarize_endpoints
    bundle, model, stats, b1 = synthetic
    out = run_inference(bundle, model, stats, b1, check=lambda: None, device='cpu', draws=4, batch_size=32, scales={k: 2. for k in loss.BLOCKS})
    extra = [dict(r, nll_b0=r['nll_b1'] * 1.5) for r in out['nll']]
    a = summarize_endpoints(out['endpoints'], out['nll'], bootstrap=20)
    b = summarize_endpoints(out['endpoints'], extra, bootstrap=20)
    assert a == b


# --- EPT-T012 S4b preparation M13 (decision B22): B1 gets the two-team-frame average at test time (an S7 option) ----------------------------------------------------------------------------
# The EPT evaluation averages the model over the recorded frame and its team swap (``FrameAveraged``); B22 gives B1 the same, so that the comparison of the two is not one of a symmetric model against a model
# that is not. B1 predicts from the features of the swapped views and the swapped conditions, the outputs go back by the tables of ``data``, and the two answers are averaged. The tests use
# the toy EPT world, its mirror (every view through ``symmetry.swap_view``, the numpy reference of the swap) and B1 heads whose predictions are fixed smooth functions of their design, so that a
# frame that is not swapped right or not averaged shows.
CATEGORICAL_HEADS = ('M2Mark', 'M2Zone', 'M3Size', 'M3Gold')
FAKE_WIDTH = {'M2Rate': 1, 'M2Mark': head_rows.N_M, 'M2Zone': head_rows.N_ZONE, 'M3Size': head_rows.N_SIZE, 'M3Kills': 2, 'M3Gold': head_rows.N_GOLD, 'M3Obj': head_rows.N_CELLS, 'M4Conv': head_rows.N_CELLS}


def fake_predict(X, W, categorical, squeeze):
    """A prediction that is a fixed smooth function of the design row (NaN as 0), row by row: probabilities for a categorical head, positive rates for a count head."""
    z = np.tanh(np.nan_to_num(np.asarray(X, dtype=np.float64)) / 100.0) @ W
    if categorical:
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    out = 0.02 + np.exp(0.3 * z)
    return out[:, 0] if squeeze else out


def asymmetric_b1(seed=11, kill_law='poisson'):
    """The B1 of the fixture with every single model replaced by ``fake_predict`` on its own random weights: a model that is not invariant under the team swap. ``kill_law='shifted'`` gives the open M3Kills head
    the design of the shifted law (it has the category d, the first-kill team)."""
    import functools
    b1 = constant_b1()
    rng = np.random.default_rng(seed)
    for group in (b1.masked, b1.open):
        for name, head in group.items():
            for member in head.parts:
                if kill_law == 'shifted' and (name, group is b1.open) == ('M3Kills', True):
                    member.layout_ = B1.make_layout('M3Kills', 'open', features.b1_columns(()), 'shifted')
                W = rng.normal(0.0, 1.5, (member.layout_.width, FAKE_WIDTH[name]))
                member.predict = functools.partial(fake_predict, W=W, categorical=name in CATEGORICAL_HEADS, squeeze=name == 'M2Rate')
    return b1


def put_array(store, name, values, dtype):
    a = store.allocate(name, values.shape, dtype)
    a[...] = values
    a.flush()


def mirror_of(world, folder):
    """The S2 store of the EPT world with every view through ``symmetry.swap_view`` (the numpy reference of the team swap), the ``views/*`` arrays as they are."""
    from c_ept.data.store_core import Store
    from c_ept.tests.ept_toys import world_view
    s2 = world.s2
    n = len(s2.array('views/src'))
    columns = {name: [] for name in tokens.VIEW_SPEC}
    for v in range(n):
        ref = symmetry.swap_view(world_view(s2, v))
        for name in columns:
            columns[name].append(ref[name])
    out = Store(folder)
    for name in ('src', 'src_index', 'match', 'ref', 't0', 'frame_k'):
        a = np.asarray(s2.array(f'views/{name}'))
        put_array(out, f'views/{name}', a, a.dtype)
    for name, rows in columns.items():
        put_array(out, f'x/{name}', np.stack(rows), np.asarray(s2.array(f'x/{name}')).dtype)
    out.finalize()
    return out


@pytest.fixture(scope='module')
def mirrored(tmp_path_factory):
    """The EPT toy world (the shapes of the real store, which ``swap_view`` checks) and its mirror."""
    folder = tmp_path_factory.mktemp('b1_swap')
    world = build_ept_world(folder / 'world', roles=('FIT', 'STOP', 'DEV'))
    yield world, mirror_of(world, folder / 'mirror')


def test_the_swapped_b1_features_are_the_b1_features_of_the_swap_view_store(mirrored):
    """``features.b1_swapped(s2, views, dropped)`` is ``features.b1`` of the views after the team swap, in every bit (the swap of the summary columns, the players, the rotation of the positions and the
    geodesics): checked against the store whose views went through ``symmetry.swap_view``. It is its own inverse, and it is not the identity (the control)."""
    world, mirror = mirrored
    n = len(world.s2.array('views/src'))
    views = np.arange(n)
    for dropped in ((), features.dropped_columns(world.s2, world.fit_views, pooled=True)):
        swapped = features.b1_swapped(world.s2, views, dropped)
        want = features.b1(mirror, views, dropped)
        assert swapped.dtype == want.dtype == np.float32 and swapped.shape == want.shape == (n, len(features.b1_columns(dropped)))
        assert np.array_equal(swapped, want, equal_nan=True)
        assert np.array_equal(features.b1_swapped(mirror, views, dropped), features.b1(world.s2, views, dropped), equal_nan=True)
        assert not np.array_equal(swapped, features.b1(world.s2, views, dropped), equal_nan=True)
    mask = np.zeros(n, dtype=bool)
    mask[::3] = True
    assert np.array_equal(features.b1_swapped(world.s2, mask, ()), features.b1_swapped(world.s2, np.flatnonzero(mask), ()), equal_nan=True)         # a mask, as for b1
    assert np.array_equal(features.b1_swapped(world.s2, views[::-1], ()), features.b1_swapped(world.s2, views, ())[::-1], equal_nan=True)         # the order of the views is kept
    assert features.b1_swapped(world.s2, views[:0], ()).shape == (0, len(features.b1_columns(())))


def all_blocks_parts(rows):
    return {(head, kind): rows.parts(head, kind, masked_obj_k=True) for (head, kind) in data.ROW_FIELDS}


def same_parts(a, b):
    for field in a._fields:
        x, y = getattr(a, field), getattr(b, field)
        if field == 'codes':
            assert x.keys() == y.keys()
            for key in x:
                assert np.array_equal(x[key], y[key]), key
        else:
            assert (x is None) == (y is None), field
            if x is not None:
                assert np.array_equal(x, y, equal_nan=True), field


def test_swap_conditions_agree_with_the_swap_of_the_ept_row_blocks_and_are_an_involution(synthetic):
    """``swap_conditions`` swaps the team-dependent inputs of the rows of a B1 head by the tables of ``data``: the mark and the zone of the conditions, the two teams of K and of O (also inside the masked M4
    matrix), the team of a gold entry. Every field of every block is compared with ``data.swap`` of the same field (the rule engine that the EPT batches go through); what the swap does not change (q, s, the layer, the
    targets, the exposures, the features) is the same object. Applied twice it gives the rows back bit for bit."""
    import torch
    bundle, model, stats, b1 = synthetic
    rows = B1.HeadRows(bundle.s1, bundle.s2, bundle.labels, b1.dropped, 'DEV')
    parts = all_blocks_parts(rows)
    assert len(parts) == 13
    for (head, kind), p in parts.items():
        swapped = adapters.swap_conditions(p, head, kind)
        batch = {}
        for name, a in {**p.codes, 'team': p.team, 'K': p.K, 'O': p.O}.items():
            if a is not None and (head, kind, name) in data.FIELD_SWAP:
                batch[data.row_key(head, kind, name)] = torch.as_tensor(np.array(a))
        if p.m4 is not None:                                                    # the masked M4 head: K and O are the last columns of the input matrix
            d = p.feat.shape[1]
            assert p.m4.shape[1] == d + 7 + 18
            batch[data.row_key(head, kind, 'K')] = torch.as_tensor(np.array(p.m4[:, d + 3:d + 7]))
            batch[data.row_key(head, kind, 'O')] = torch.as_tensor(np.array(p.m4[:, d + 7:]))
        assert batch or head == 'M2Rate' or (head, kind) == ('M3Size', 'open'), (head, kind)       # these two have nothing that the swap changes
        expected = data.swap(batch)
        for name in p.codes:
            key = data.row_key(head, kind, name)
            if key in expected:
                assert np.array_equal(swapped.codes[name], expected[key].numpy()), (head, kind, name)
        if p.team is not None:
            assert np.array_equal(swapped.team, expected[data.row_key(head, kind, 'team')].numpy()) and not np.array_equal(swapped.team, p.team)
        if p.K is not None:
            assert np.array_equal(swapped.K, expected[data.row_key(head, kind, 'K')].numpy()) and swapped.K.shape == p.K.shape
        if p.O is not None:
            assert np.array_equal(swapped.O, expected[data.row_key(head, kind, 'O')].numpy())
        if p.m4 is not None:
            assert np.array_equal(swapped.m4[:, d + 3:d + 7], expected[data.row_key(head, kind, 'K')].numpy())
            assert np.array_equal(swapped.m4[:, d + 7:], expected[data.row_key(head, kind, 'O')].numpy())
            for j, name in enumerate(('q', 'm', 'z')):
                assert np.array_equal(swapped.m4[:, d + j], swapped.codes[name]) and np.array_equal(p.m4[:, d + j], p.codes[name])      # the matrix and the codes tell one story
            assert swapped.m4.dtype == p.m4.dtype and swapped.m4 is not p.m4
        for field in ('feat', 'take', 'y', 'e', 'match', 'row'):
            assert getattr(swapped, field) is getattr(p, field), (head, kind, field)
        same_parts(adapters.swap_conditions(swapped, head, kind), p)
    assert any(p.team is not None for p in parts.values()) and any(p.m4 is not None for p in parts.values())          # the blocks with a team, K, O and a matrix are there


def test_swap_conditions_by_hand():
    """The first-kill team d (the open M3Kills rows of the shifted kill law) becomes 1 - d and an execution (-1) stays; marks, zones (-1 stays), the two teams of K and O and the team of a gold entry by hand."""
    from c_ept.preprocessing import terrain
    feat = np.zeros((5, 2), dtype=np.float32)
    kills = head_rows.HeadParts(feat, None, {'s': np.array([0, 1, 2, 3, 0]), 'd': np.array([-1, 0, 1, 0, 1])}, np.zeros((5, 2), dtype=np.int64), np.ones(5), np.zeros(5, dtype=np.int64),
                                np.arange(5), first_team=np.array([-1, 0, 1, 0, 1]))
    out = adapters.swap_conditions(kills, 'M3Kills', 'open')
    assert out.codes['d'].tolist() == [-1, 1, 0, 1, 0] and out.first_team.tolist() == [-1, 1, 0, 1, 0] and out.codes['s'].tolist() == [0, 1, 2, 3, 0]
    O = np.zeros((2, 18), dtype=np.int64)
    O[0, 0], O[1, 17] = 3, 2
    obj = head_rows.HeadParts(feat[:2], None, {'q': np.array([4, 5]), 'm': np.array([0, 50]), 'z': np.array([-1, 2]), 's': np.array([1, 2])}, O, np.ones(2), np.zeros(2, dtype=np.int64), np.arange(2),
                              K=np.array([[2., 1., 3.0, 1.5], [0., 4., 0.0, 6.5]]), O=O, team=np.array([0, 1]))
    out = adapters.swap_conditions(obj, 'M4Conv', 'open')
    assert out.codes['m'].tolist() == [25, 51] and out.codes['z'].tolist() == [-1, int(terrain.ZONE_SWAP[2])] == [-1, 4] and out.codes['q'].tolist() == [4, 5]
    assert out.K.tolist() == [[1., 2., 1.5, 3.0], [4., 0., 6.5, 0.0]] and out.team.tolist() == [1, 0]
    assert out.O[0, 1] == 3 and out.O[1, 16] == 2 and out.O.sum() == 5 and obj.codes['m'].tolist() == [0, 50]         # blue taker cell 0 -> red cell 1; the input is not changed
    marks = adapters.swap_conditions(head_rows.HeadParts(feat[:4], None, {'q': np.zeros(4, dtype=np.int64), 'm': np.array([0, 25, 49, 51])}, np.zeros(4, dtype=np.int64), None,
                                                         np.zeros(4, dtype=np.int64), np.arange(4)), 'M2Zone', 'masked')
    assert marks.codes['m'].tolist() == [25, 0, 24, 50]


def test_back_permutation_is_the_swap_of_the_target_of_the_head():
    """The axis of a prediction that moves back to the recorded frame is the axis that the swap moves in the target of the head: the 52 marks, the 10 zones, the two teams, the 18 cells; the size law, the rate and
    the gold law of a team entry stay (the team entry changes its model instead). Each table is its own inverse."""
    import torch
    for (head, kind), change in ((hk, data.FIELD_SWAP[(*hk, 'y')]) for hk in data.ROW_FIELDS):
        perm = adapters.back_permutation(head, kind)
        assert (perm is None) == (change is None), (head, kind)
        if perm is None:
            continue
        assert np.array_equal(perm[perm], np.arange(len(perm))), (head, kind)
        key = data.row_key(head, kind, 'y')
        if change in ('mark', 'zone'):
            labels = np.arange(len(perm))
            assert data.swap({key: torch.as_tensor(labels)})[key].numpy().tolist() == perm[labels].tolist(), (head, kind)
        else:
            y = np.arange(3 * len(perm)).reshape(3, len(perm))
            assert np.array_equal(data.swap({key: torch.as_tensor(y)})[key].numpy(), y[:, perm]), (head, kind)
    assert [adapters.back_permutation(h, k) is None for h, k in (('M2Rate', 'masked'), ('M3Size', 'masked'), ('M3Size', 'open'), ('M3Gold', 'masked'), ('M3Gold', 'open'))] == [True] * 5
    assert [len(adapters.back_permutation(h, k)) for h, k in (('M2Mark', 'masked'), ('M2Zone', 'masked'), ('M3Kills', 'open'), ('M3Obj', 'masked'), ('M4Conv', 'open'))] == [52, 10, 2, 18, 18]


def twin_parts(world, mirror, head, kind):
    """The rows of a head in the recorded frame (``P0``) and the rows of the team-swapped world (``P1``: the features of the mirror, the conditions swapped), the very rows that the average sees as the original and the swapped frame."""
    rows0 = B1.HeadRows(world.s1, world.s2, world.labels, (), 'DEV')
    rows1 = B1.HeadRows(world.s1, mirror, world.labels, (), 'DEV')
    return rows0.parts(head, kind, masked_obj_k=True), adapters.swap_conditions(rows1.parts(head, kind, masked_obj_k=True), head, kind)


def test_b1_swap_average_is_invariant_to_team_labels(mirrored):
    """The B1 prediction averaged over the two frames does not depend on which team is called blue: for every one of the thirteen heads, the average of the rows P0 with the swapped rows P1 is, moved back by the table of
    the head, the average of P1 with P0 -- in every bit -- whereas the prediction of one frame is not (the heads are not symmetric; the control). The rows P1 are the rows of the team-swapped world: the features of the
    store that went through ``symmetry.swap_view`` (not of ``b1_swapped``) and the conditions swapped by ``swap_conditions``. Two wrong swaps are caught by the same comparison: no swap at all and a swap of the features only."""
    world, mirror = mirrored
    b1 = asymmetric_b1()
    columns = features.b1_columns(())
    seen = 0
    for (head, kind) in data.ROW_FIELDS:
        p0, p1 = twin_parts(world, mirror, head, kind)
        fitted = getattr(b1, kind)[head]
        idx = np.arange(len(p0.row))
        assert len(idx) > 1, (head, kind)
        perm = adapters.back_permutation(head, kind)
        a = adapters.b1_predict_averaged(head, kind, fitted, p0, p1, columns, idx)
        b = adapters.b1_predict_averaged(head, kind, fitted, p1, p0, columns, idx)
        assert a.shape == b.shape and a.dtype == np.float64 and np.isfinite(a).all(), (head, kind)
        assert np.array_equal(a, b if perm is None else b[..., perm]), (head, kind)
        single_0 = adapters.b1_predict_averaged(head, kind, fitted, p0, p0, columns, idx)                     # no swap at all
        assert not np.array_equal(a, single_0), (head, kind)
        features_only = p1._replace(codes=p0.codes, K=p0.K, O=p0.O, team=p0.team, first_team=p0.first_team,
                                    m4=None if p0.m4 is None else np.hstack([p1.m4[:, :p0.feat.shape[1]], p0.m4[:, p0.feat.shape[1]:]]))     # the swapped features, the recorded conditions
        no_conditions = adapters.b1_predict_averaged(head, kind, fitted, p0, features_only, columns, idx)
        has_conditions = p0.m4 is not None or p0.team is not None or p0.K is not None or p0.O is not None or bool({'m', 'z', 'd'} & set(p0.codes))
        assert np.array_equal(a, no_conditions) != has_conditions, (head, kind)                          # a head with a mark, zone, K, O or team needs them swapped; the others have only the features to swap
        seen += 1
    assert seen == 13


def test_the_open_b1_laws_and_objects_are_invariant_to_team_labels_when_averaged(mirrored):
    """The open M3 laws of B1 (size, the kills of the two teams by size, the gold law of each team by size and layer) and its object rates, averaged over the two frames, are invariant: size unchanged, the teams
    of the kills and of the gold moved back by ``TEAM_PERM``, the cells by ``CELL_PERM`` (and the generated K swapped for the swapped frame by ``K_PERM``). The first-kill team d of the shifted law is swapped with the frame."""
    world, mirror = mirrored
    b1 = asymmetric_b1()
    open_heads = adapters.B1OpenHeads(b1)
    averaged = adapters.B1SwapAveragedOpenHeads(open_heads)
    p0, p1 = twin_parts(world, mirror, 'M3Size', 'open')
    n0, n1 = p0.feat[:4], p1.feat[:4]
    a = averaged.laws(n0, n1, check=lambda: None)
    b = averaged.laws(n1, n0, check=lambda: None)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1][..., data.TEAM_PERM]) and np.array_equal(a[2], b[2][:, :, data.TEAM_PERM])
    plain = open_heads.laws(n0, check=lambda: None)
    assert not np.array_equal(a[1], plain[1]) and not np.array_equal(a[2], plain[2])                  # the average moved the laws (control)
    for x, y in zip(plain, a):                                                                         # still probabilities / rates of the right shape
        assert x.shape == y.shape and np.isfinite(y).all()
    assert np.allclose(a[0].sum(-1), 1.0) and np.allclose(a[2].sum(-1), 1.0)
    size = np.array([0, 1, 2, 3, 1, 0, 2, 3])
    K = np.array([[2., 1., 1500., 700.], [0., 3., 0., 2500.], [4., 4., 3000., 3000.], [1., 0., 400., 0.]] * 2)
    c = averaged.objects(n0[:1].repeat(8, axis=0), n1[:1].repeat(8, axis=0), size, K)
    d = averaged.objects(n1[:1].repeat(8, axis=0), n0[:1].repeat(8, axis=0), size, K[:, data.K_PERM])
    assert c.shape == (8, 18) and np.array_equal(c, d[:, data.CELL_PERM])
    assert not np.array_equal(c, open_heads.objects(n0[:1].repeat(8, axis=0), size, K))
    for first in (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, -1])):                                  # the shifted law: the open kill head reads d; it is swapped with the frame
        assert adapters.swap_first_team(adapters.swap_first_team(first)).tolist() == first.tolist()
        assert adapters.swap_first_team(first).tolist() == [(-1 if f < 0 else 1 - f) for f in first.tolist()]
    shifted = adapters.B1SwapAveragedOpenHeads(adapters.B1OpenHeads(asymmetric_b1(kill_law='shifted')))        # an open kill head that has the slot d of the shifted law
    d0 = np.array([-1, 0, 1, 0])
    e = shifted.laws(n0, n1, check=lambda: None, d=d0)
    f = shifted.laws(n1, n0, check=lambda: None, d=adapters.swap_first_team(d0))
    assert np.array_equal(e[0], f[0]) and np.array_equal(e[1], f[1][..., data.TEAM_PERM]) and np.array_equal(e[2], f[2][:, :, data.TEAM_PERM])
    assert not np.array_equal(e[1], shifted.laws(n1, n0, check=lambda: None, d=d0)[1][..., data.TEAM_PERM])      # the control: with d left as it is the invariance fails
    with pytest.raises(ValueError, match='first-kill team'):
        shifted.laws(n0, n1, check=lambda: None)


def run_pair(bundle, model, stats, b1, **extra):
    common = dict(check=lambda: None, device='cpu', draws=12, scales={k: 2. for k in loss.BLOCKS})
    return run_inference(bundle, model, stats, b1, **common), run_inference(bundle, model, stats, b1, b1_swap_average=True, **common, **extra)


def test_defaults_are_bit_identical(synthetic):
    """``b1_swap_average`` is a keyword-only option whose default False is every earlier evaluation: the explicit False is the default result in every field and the provenance has no new key."""
    import inspect
    bundle, model, stats, b1 = synthetic
    parameter = inspect.signature(run_inference).parameters['b1_swap_average']
    assert parameter.default is False and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    default = run_inference(bundle, model, stats, b1, **WINNER_TRUTH_RUN)
    assert_same(default, run_inference(bundle, model, stats, b1, b1_swap_average=False, **WINNER_TRUTH_RUN))
    assert not {'b1_swap_average', 'b1_swap_average_note'} & set(default['provenance'])
    other = asymmetric_b1()                                                       # also for heads that are not symmetric: False is the one-frame evaluation, whatever the heads
    assert_same(run_inference(bundle, model, stats, other, **WINNER_TRUTH_RUN), run_inference(bundle, model, stats, other, b1_swap_average=False, **WINNER_TRUTH_RUN))


@pytest.mark.parametrize('bad', [None, 0, 1, 'yes', 'True', 1.0])
def test_b1_swap_average_must_be_a_bool_and_is_checked_before_any_store_read(bad):
    with pytest.raises(ValueError, match='b1_swap_average'):
        run_inference(_NoStoreRead(), object(), object(), object(), check=lambda: None, device='cpu', b1_swap_average=bad)


def test_b1_swap_average_on_symmetric_heads_changes_exactly_the_mark_and_zone_blocks_and_nothing_of_the_ept(synthetic):
    """The constant heads of the fixture predict the same rate for both teams and the same size and gold law whatever the frame, so the average changes nothing for them (the mean of a number with itself is the
    number, in every bit): the eleven blocks without a mark or zone axis, the open Monte Carlo and every endpoint of the M3 and M4 heads are the default's. Their mark and zone laws put the mass on the class 0 of the
    recorded frame, which the swap moves to another mark (0 <-> 25) and another zone (0 <-> 1): the M2Mark and M2Zone blocks of B1 change. The EPT is the same in every bit, and the provenance says that B1 was averaged."""
    bundle, model, stats, b1 = synthetic
    default, swapped = run_pair(bundle, model, stats, b1)
    changed = [r['name'] for r, s in zip(default['nll'], swapped['nll']) if not np.array_equal(r['nll_b1'], s['nll_b1'])]
    assert changed == ['M2Mark/masked', 'M2Zone/masked']
    for r, s in zip(default['nll'], swapped['nll']):
        for key in set(r) - {'nll_b1', 'nll_b1_normalized'}:
            assert_same(r[key], s[key], f'{r["name"]}/{key}')                                       # the EPT columns, the rows, the counts: untouched
    for d, s in zip(default['endpoints'], swapped['endpoints']):
        assert_same(d, s, d['name'])                                                      # no endpoint moved: the M2 occurrence uses the (symmetric) rate, the open Monte Carlo the symmetric laws
    assert swapped['provenance']['b1_swap_average'] is True and 'B22' in swapped['provenance']['b1_swap_average_note']
    assert {k: v for k, v in swapped['provenance'].items() if k not in ('b1_swap_average', 'b1_swap_average_note')} == default['provenance']


def expected_b1_nll(bundle, b1, head, kind):
    """The B1 NLL of a block under the average, from public pieces only: the rows of the recorded frame and of the swapped frame, ``swap_conditions``, ``b1_predict_averaged``, the NLL of ``nll`` and the HR-5 sum."""
    from c_ept.models import nll as nll_math
    rows = B1.HeadRows(bundle.s1, bundle.s2, bundle.labels, b1.dropped, 'DEV')
    swapped_rows = head_rows.HeadRows(bundle.s1, bundle.s2, bundle.labels, 'DEV', lambda store, views: features.b1_swapped(store, views, b1.dropped))
    p0 = rows.parts(head, kind, masked_obj_k=True)
    p1 = adapters.swap_conditions(swapped_rows.parts(head, kind, masked_obj_k=True), head, kind)
    pred = np.concatenate([adapters.b1_predict_averaged(head, kind, getattr(b1, kind)[head], p0, p1, b1.columns, np.arange(lo, min(lo + 32, len(p0.row))))
                           for lo in range(0, len(p0.row), 32)])               # in the chunks of 32 rows that run_inference uses (a matrix product need not give one bit for any split)
    if p0.e is not None:
        e = p0.e if pred.ndim == 1 else p0.e[:, None]
        score = nll_math.poisson(p0.y, pred, e)
        score = score.sum(-1) if score.ndim > 1 else score
        keep = p0.e > 0
    else:
        score, keep = nll_math.categorical(p0.y, pred), p0.y >= 0
    return _aggregate(p0.match[keep], p0.row[keep], score[keep])[2]


def test_run_inference_scores_b1_on_the_average_of_the_two_frames(synthetic):
    """With heads that are not symmetric the B1 numbers of every block move and the EPT numbers do not; the B1 NLL of a block of each kind (a categorical head, a count head with K, a gold head, a conversion head with
    K and O in its matrix) is the NLL of the averaged prediction that the public pieces give, in every bit. ``b1_swap_average=False`` is still the one-frame result."""
    bundle, model, stats, _ = synthetic
    b1 = asymmetric_b1()
    default, swapped = run_pair(bundle, model, stats, b1)
    moved = [r['name'] for r, s in zip(default['nll'], swapped['nll']) if not np.array_equal(r['nll_b1'], s['nll_b1'])]
    scored = [r['name'] for r in default['nll'] if len(r['row'])]
    assert moved == scored and len(scored) >= 12                                                         # every B1 block that has a row moved
    for r, s in zip(default['nll'], swapped['nll']):
        assert np.array_equal(r['nll_ept'], s['nll_ept']) and np.array_equal(r['row'], s['row']) and r['counts'] == s['counts']
    for head, kind in (('M2Mark', 'masked'), ('M3Obj', 'masked'), ('M3Gold', 'open'), ('M4Conv', 'masked'), ('M2Rate', 'masked')):
        record = next(r for r in swapped['nll'] if r['name'] == f'{head}/{kind}')
        assert np.array_equal(record['nll_b1'], expected_b1_nll(bundle, b1, head, kind)), (head, kind)
    assert any(not np.array_equal(d['p_b1'], s['p_b1']) for d, s in zip(default['endpoints'], swapped['endpoints']))      # the endpoints of B1 moved too (the M2 rate and the open Monte Carlo)
    for d, s in zip(default['endpoints'], swapped['endpoints']):
        for key in ('p_ept', 'row', 'match', 'y'):
            assert np.array_equal(d[key], s[key]), (d['name'], key)


# --- M13 review fixes (B22): the swap average fails closed on a dropped list that the team swap does not close ---------------------------------------------------------------------------
def summary_pair():
    """The names of the first two summary columns that the team swap exchanges (``symmetry.SUMMARY_SWAP``)."""
    perm = symmetry.SUMMARY_SWAP
    i = next(i for i in range(len(perm)) if perm[i] != i)
    return tokens.SUMMARY[i], tokens.SUMMARY[int(perm[i])]


def test_a_dropped_list_is_closed_under_the_team_swap_when_a_dropped_summary_column_has_its_partner_dropped_too():
    """``features.swap_unclosed_columns(dropped)`` names the pairs (dropped column, kept partner) of the summary columns that the swap exchanges: one member dropped and the other kept makes the swapped frame feed the
    dropped column's (FIT-constant) values into the kept column. A pair dropped together, a column that is its own partner (t_s) and every player column (dropped for all ten players, kept by the rotation) are closed."""
    a, b = summary_pair()
    assert features.swap_unclosed_columns((a,)) == ((a, b),) and features.swap_unclosed_columns((b,)) == ((b, a),)
    assert features.swap_unclosed_columns((a, b)) == () and features.swap_unclosed_columns(()) == () and features.swap_unclosed_columns(('t_s',)) == ()
    assert features.swap_unclosed_columns(tuple(tokens.PLAYER_NUM)) == ()
    assert features.swap_unclosed_columns(tuple(tokens.SUMMARY) + tuple(tokens.PLAYER_NUM)) == ()
    with pytest.raises(ValueError, match='column'):
        features.swap_unclosed_columns(('not_a_column',))


@pytest.fixture(scope='module')
def fitted_world(tmp_path_factory):
    """The toy EPT world with the default roles (the STOP match gives the validation rows that a real fit needs), its bundle and model, and two real small B1 sets: one fitted on a dropped list that the swap does not
    close (one member of a swap pair) and one on the pooled list of the toy FIT views."""
    world = build_ept_world(tmp_path_factory.mktemp('fitted_world'), n_events=7, n_edges=5)
    n_matches = len(np.asarray(world.s1.array('matches/role')))
    s1 = Overlay(world.s1, {'matches/obj_off': np.zeros(n_matches + 1, dtype=np.int64), **{f'objectives/{k}': np.empty(0, dtype=np.int64) for k in ('t', 'kind', 'sub', 'taker')}})
    n_units = len(s1.array('units/f'))
    labels = Overlay(world.labels, {'labels/f0_unit': np.resize(np.array([-1., 0., 1.]), n_units), 'labels/gold_reps': np.arange(31, dtype=float) * 100})
    stats = data.fit_stats(s1, world.s2)
    torch.manual_seed(17)
    model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=.1, kappas=1, field_channels=2), toy_static())
    bundle = SimpleNamespace(s1=s1, s2=world.s2, labels=labels, role='DEV', prices={k: 100. for k in flowprice.GROUPS}, tier=toy_tier_model())
    small = {'num_leaves': 7, 'min_data_in_leaf': 5, 'num_iterations': 12, 'min_data_per_group': 5, 'cat_smooth': 1.0, 'num_threads': 1}
    pooled = features.dropped_columns(world.s2, np.flatnonzero(features.split_views(world.s1, world.s2, 'FIT')), pooled=True)
    a, _ = summary_pair()
    legacy = B1.B1Heads.fit(world.s1, world.s2, world.labels, (a,), params=small)
    closed = B1.B1Heads.fit(world.s1, world.s2, world.labels, pooled, params=small)
    return bundle, model, stats, legacy, closed, pooled


def test_b1_swap_average_refuses_a_b1_set_whose_dropped_list_the_team_swap_does_not_close(fitted_world):
    """The average is exactly invariant for any B1, but with one member of a swap pair dropped the swapped frame would read FIT-constant values in the kept column: B1 would be silently degraded against the EPT.
    ``run_inference`` refuses it with a ValueError that names the two columns (no data value), before any store is read; the same set is accepted without the average (every earlier evaluation), and a set fitted on
    the pooled list is accepted with it."""
    bundle, model, stats, legacy, pooled_set, pooled = fitted_world
    a, b = summary_pair()
    assert features.swap_unclosed_columns(legacy.dropped) == ((a, b),) and features.swap_unclosed_columns(pooled) == ()
    poisoned = SimpleNamespace(role='DEV', s1=_PoisonStore(), s2=_PoisonStore(), labels=_PoisonStore(), prices=bundle.prices, tier=bundle.tier)
    with pytest.raises(ValueError, match='team swap') as raised:
        run_inference(poisoned, model, stats, legacy, b1_swap_average=True, **WINNER_TRUTH_RUN)
    assert a in str(raised.value) and b in str(raised.value)
    one_frame = run_inference(bundle, model, stats, legacy, **WINNER_TRUTH_RUN)
    assert 'b1_swap_average' not in one_frame['provenance']
    assert run_inference(bundle, model, stats, pooled_set, b1_swap_average=True, **WINNER_TRUTH_RUN)['provenance']['b1_swap_average'] is True


def test_the_swapped_b1_rows_must_be_the_rows_of_the_recorded_frame(fitted_world, monkeypatch):
    """``b1_predict_averaged`` averages row i of the recorded frame with row i of the swapped frame: run_inference compares match, row, target, exposure and design rows of the two and fails closed otherwise."""
    bundle, model, stats, _, b1, _ = fitted_world
    real = adapters.swap_conditions

    def shifted(parts, head, kind):
        out = real(parts, head, kind)
        return out._replace(row=out.row[::-1].copy()) if head == 'M2Mark' else out
    monkeypatch.setattr(adapters, 'swap_conditions', shifted)
    with pytest.raises(ValueError, match='swapped B1 frame'):
        run_inference(bundle, model, stats, b1, b1_swap_average=True, **WINNER_TRUTH_RUN)
