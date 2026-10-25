"""Synthetic-only, CPU tests; never import adapters, loaders or model payloads."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from c_ept.evaluation.metrics import summarize_endpoints
from c_ept.evaluation.reporting import write_report


def endpoint(**changes):
    result = dict(name='M2-60', module='M2', layer='t0', horizon_s=60,
                  target='first_kill_ge1', match=['a', 'b'], row=['r0', 'r1'],
                  y=[False, True], p_ept=[.2, .8], p_b1=[.4, .6])
    result.update(changes)
    return result


def metric(summary, name='auc'):
    return summary['endpoints'][0]['metrics'][name]


def pairwise_auc(y, p, w):
    # Independent, deliberately quadratic hand oracle used only on tiny fixtures.
    positive = [i for i, v in enumerate(y) if v == 1]
    negative = [i for i, v in enumerate(y) if v == 0]
    denominator = sum(w[i] for i in positive) * sum(w[i] for i in negative)
    if denominator == 0:
        return None
    return sum(w[i] * w[j] * (1 if p[i] > p[j] else .5 if p[i] == p[j] else 0)
               for i in positive for j in negative) / denominator


def test_hand_brier_and_weighted_ties():
    base = summarize_endpoints([endpoint()], bootstrap=0)
    assert metric(base)['ept']['estimate'] == 1
    assert metric(base, 'brier')['ept']['estimate'] == pytest.approx(.04)
    assert metric(base, 'brier')['b1']['estimate'] == pytest.approx(.16)
    assert metric(base, 'brier')['delta']['estimate'] == pytest.approx(-.12)
    e = endpoint(match=['a', 'a', 'b'], row=['r0', 'r1', 'r2'],
                 y=[0, 1, 1], p_ept=[.5, .5, 1], p_b1=[.5, .5, .5])
    s = summarize_endpoints([e], bootstrap=0)
    assert metric(s)['ept']['estimate'] == pytest.approx(5 / 6)
    assert metric(s)['b1']['estimate'] == .5
    assert metric(s, 'brier')['ept']['estimate'] == pytest.approx(.125)
    assert s['endpoints'][0]['counts']['weight_sum'] == 2
    roc = s['endpoints'][0]['roc']['ept']
    assert roc['thresholds'] == [None, 1, .5]
    assert roc['fpr'] == [0, 0, 1]
    assert roc['tpr'] == pytest.approx([0, 2 / 3, 1])


def test_paired_bootstrap_matches_explicit_repeated_match_oracle():
    e = endpoint(match=['a', 'a', 'b', 'c', 'c'], row=list('uvwxy'),
                 y=[0, 1, 1, 0, 1], p_ept=[.1, .6, .6, .8, .9], p_b1=[.8, .3, .6, .2, .7])
    s = summarize_endpoints([e], bootstrap=137, seed=91)
    rng = np.random.default_rng(91)
    y = np.array(e['y'])
    indices = np.array([0, 0, 1, 2, 2])
    original_weights = np.array([.5, .5, 1, .5, .5])
    auc, brier = [], []
    has_repeated = False
    for _ in range(137):
        selected = rng.integers(3, size=3)
        repeated_rows = np.concatenate([np.flatnonzero(indices == m) for m in selected])
        has_repeated |= len(set(selected)) < 3
        # Repeated physical copies retain their original weights; never divide
        # by the new row count of a match, which would erase its multiplicity.
        w = original_weights[repeated_rows]
        a, b = [], []
        for model in ('ept', 'b1'):
            p = np.asarray(e[f'p_{model}'])[repeated_rows]
            a.append(pairwise_auc(y[repeated_rows], p, w))
            b.append(np.average((p - y[repeated_rows]) ** 2, weights=w))
        if all(v is not None for v in a):
            auc.append(a)
        brier.append(b)
    assert has_repeated
    for name, draws in [('auc', auc), ('brier', brier)]:
        values = metric(s, name)
        assert values['valid_replicates'] == len(draws)
        assert values['invalid_replicates'] == 137 - len(draws)
        column = 'conditional_ci95' if len(draws) < 137 else 'ci95'
        draws = np.asarray(draws)
        for model, expected in [('ept', draws[:, 0]), ('b1', draws[:, 1]), ('delta', draws[:, 0] - draws[:, 1])]:
            assert values[model][column] == pytest.approx(np.quantile(expected, [.025, .975]))


def test_identical_models_have_exact_zero_paired_delta():
    s = summarize_endpoints([endpoint(p_b1=[.2, .8])], bootstrap=61)
    for name in ('auc', 'brier'):
        d = metric(s, name)['delta']
        assert d['estimate'] == 0
        assert (d['ci95'] or d['conditional_ci95']) == [0, 0]


def test_single_class_replicates_invalidate_primary_auc_ci():
    s = summarize_endpoints([endpoint()], bootstrap=200)
    a = metric(s)
    assert 0 < a['invalid_replicates'] < 200
    assert a['valid_replicates'] + a['invalid_replicates'] == 200
    for model in ('ept', 'b1', 'delta'):
        assert a[model]['ci95'] is None
        assert a[model]['conditional_ci95'] is not None
    assert metric(s, 'brier')['invalid_replicates'] == 0
    assert metric(s, 'brier')['ept']['ci95'] is not None


@pytest.mark.parametrize('labels', [[0, 0], [1, 1]])
def test_single_class_is_unavailable_not_half_or_zero(labels):
    s = summarize_endpoints([endpoint(y=labels)], bootstrap=12)
    a = metric(s)
    assert a['reason'] == 'single_class'
    assert a['invalid_replicates'] == 12
    for model in ('ept', 'b1', 'delta'):
        assert a[model] == dict(estimate=None, ci95=None, conditional_ci95=None)
    assert s['endpoints'][0]['roc']['ept']['fpr'] is None
    assert metric(s, 'brier')['ept']['estimate'] is not None
    json.dumps(s, allow_nan=False)


def test_empty_endpoint_has_explicit_missing_denominator():
    e = endpoint(match=[], row=[], y=[], p_ept=[], p_b1=[])
    s = summarize_endpoints([e], bootstrap=3)
    assert metric(s)['reason'] == 'no_rows'
    assert metric(s, 'brier')['ept']['estimate'] is None
    assert s['endpoints'][0]['counts']['n_matches'] == 0
    assert all(b['observed_fraction'] is None for b in s['endpoints'][0]['reliability']['ept'])


@pytest.mark.parametrize('change', [
    {'match': ['a', 'a'], 'row': ['same', 'same']},
    {'match': [['a'], ['b']]}, {'row': [['a'], ['b']]},
    {'p_ept': [[.2], [.8]]}, {'p_b1': [.5]}, {'y': [0, 2]},
    {'y': [0, np.nan]}, {'y': ['0', '1']},
    {'p_ept': [np.nan, .8]}, {'p_b1': [.5, np.inf]},
    {'p_ept': [-.001, .8]}, {'p_b1': [.5, 1.01]},
    {'match': ['a', None]}, {'row': ['a', '']}, {'horizon_s': 0},
    {'horizon_s': float('nan')}, {'horizon_s': True}, {'module': ''},
    {'mc_se_ept': [-.1, .1]}, {'counts': {'selected': np.nan}},
    {'draw_truth': [0, 1]},
    {'draw_truth': [0, 1], 'p_draw_ept': [.1, .1], 'p_draw_b1': [.1, .1]},
    {'draw_truth': [1, 0], 'p_draw_ept': [.9, .5], 'p_draw_b1': [.1, .1]},
])
def test_invalid_inputs_fail_closed(change):
    with pytest.raises((ValueError, TypeError)):
        summarize_endpoints([endpoint(**change)], bootstrap=0)


def test_missing_labels_and_duplicate_names_rejected():
    missing = endpoint()
    del missing['y']
    with pytest.raises(ValueError, match='missing y'):
        summarize_endpoints([missing], bootstrap=0)
    with pytest.raises(ValueError, match='duplicate'):
        summarize_endpoints([endpoint(), endpoint()], bootstrap=0)


def test_composite_row_keys_and_boolean_labels_accepted():
    e = endpoint(row=[('frame', 1), ('frame', 2)])
    assert metric(summarize_endpoints([e], bootstrap=0))['ept']['estimate'] == 1


def test_m3_completion_value_preserves_explicit_nonfixed_horizon():
    e = endpoint(name='M3-positive-value', module='M3', layer='open',
                 horizon_s=None, target='positive_value')
    summary = summarize_endpoints([e], bootstrap=7)
    result = summary['endpoints'][0]
    assert result['horizon_s'] is None
    assert result['module'] == 'M3' and result['layer'] == 'open'
    assert result['target'] == 'positive_value'
    assert result['metrics']['auc']['ept']['estimate'] == 1
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    del e['horizon_s']
    with pytest.raises(ValueError, match='missing horizon_s'):
        summarize_endpoints([e], bootstrap=0)


@pytest.mark.parametrize('horizon', [True, False, 0, -1, np.nan, np.inf, -np.inf, '60'])
def test_m3_nonfixed_horizon_does_not_relax_invalid_numeric_contract(horizon):
    with pytest.raises(ValueError, match='horizon_s'):
        summarize_endpoints([endpoint(module='M3', layer='open', horizon_s=horizon)], bootstrap=0)


def test_calibration_probability_one_and_empty_bins():
    s = summarize_endpoints([endpoint(p_ept=[0., 1.])], bootstrap=0)
    bins = s['endpoints'][0]['reliability']['ept']
    assert len(bins) == 10
    assert bins[0]['n_rows'] == bins[-1]['n_rows'] == 1
    assert bins[0]['observed_fraction'] == 0
    assert bins[-1]['right_closed'] is True
    assert bins[-1]['mean_prediction'] == bins[-1]['observed_fraction'] == 1
    assert bins[1]['n_rows'] == 0
    assert bins[1]['mean_prediction'] is None
    assert bins[1]['observed_fraction'] is None


def test_every_decimal_bin_boundary_and_match_weighted_calibration():
    e = endpoint(match=['a'] * 10 + ['b'], row=list(range(11)), y=[0] * 10 + [1],
                 p_ept=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
                 p_b1=[.5] * 11)
    s = summarize_endpoints([e], bootstrap=0)
    bins = s['endpoints'][0]['reliability']['ept']
    assert [b['n_rows'] for b in bins] == [1] * 9 + [2]
    assert bins[-1]['weight_sum'] == pytest.approx(1.1)
    assert bins[-1]['observed_fraction'] == pytest.approx(1 / 1.1)
    assert bins[-1]['mean_prediction'] == pytest.approx(1.09 / 1.1)


def test_default_bootstrap_contract_and_tied_auc_oracle():
    default = summarize_endpoints([endpoint()])
    assert default['bootstrap']['replicates'] == 2000
    assert default['bootstrap']['seed'] == 20261022
    assert metric(default)['valid_replicates'] + metric(default)['invalid_replicates'] == 2000
    rng = np.random.default_rng(72)
    for _ in range(20):
        y = rng.integers(2, size=15)
        y[:2] = [0, 1]
        p = rng.integers(5, size=15) / 4
        match = rng.integers(4, size=15)
        w = 1 / np.bincount(match)[match]
        e = endpoint(match=match, row=np.arange(15), y=y, p_ept=p, p_b1=p)
        s = summarize_endpoints([e], bootstrap=0)
        assert metric(s)['ept']['estimate'] == pytest.approx(pairwise_auc(y, p, w))


def test_draw_and_mc_diagnostics_use_endpoint_weights():
    e = endpoint(match=['a', 'a', 'b'], row=list('abc'), y=[0, 0, 1],
                 p_ept=[.2, .3, .8], p_b1=[.3, .4, .6], draw_truth=[1, 0, 0],
                 p_draw_ept=[.7, .6, .1], p_draw_b1=[.5, .4, .2],
                 mc_se_ept=[.01, .03, .04], counts={'candidate': 5, 'selected': 3, 'excluded': {'risk': 2}})
    s = summarize_endpoints([e], bootstrap=10)
    out = s['endpoints'][0]
    assert out['draw']['n_rows'] == 1
    assert out['draw']['weighted_fraction'] == .25
    assert out['mc_uncertainty']['mc_se_ept']['weighted_mean'] == pytest.approx(.03)
    assert out['mc_uncertainty']['mc_se_ept']['max'] == .04
    assert out['counts']['selection']['excluded'] == {'risk': 2}
    assert 'draw_brier' in out['metrics']


def test_seed_repeatability_and_periodic_guard():
    calls = []
    a = summarize_endpoints([endpoint()], bootstrap=83, seed=31, check=lambda: calls.append(1))
    b = summarize_endpoints([endpoint()], bootstrap=83, seed=31)
    assert a == b
    assert len(calls) >= 6
    count = 0

    def stop():
        nonlocal count
        count += 1
        if count == 4:
            raise RuntimeError('budget')

    with pytest.raises(RuntimeError, match='budget'):
        summarize_endpoints([endpoint()], bootstrap=2000, check=stop)
    assert count == 4
    with pytest.raises(RuntimeError, match='guard'):
        summarize_endpoints([], check=lambda: False)


@pytest.mark.parametrize('options', [{'bootstrap': -1}, {'bootstrap': True}, {'bootstrap': 1.5}, {'seed': -1}, {'seed': True}])
def test_bootstrap_options_validated(options):
    with pytest.raises(ValueError):
        summarize_endpoints([], **options)


def nll_head(name='kill', **changes):
    result = dict(name=name, match=['a', 'a', 'b'], row=list('abc'),
                  nll_ept=[1., 3., 8.], nll_b1=[2., 4., 10.], scale=2.)
    result.update(changes)
    return result


def test_nll_fixed_scale_and_explicit_structural_heads_only():
    heads = [nll_head(common_head=True), nll_head('objective', nll_ept=[2., 6., 16.], nll_b1=[4., 8., 20.], scale=4., common_head=True),
             nll_head('not_structural', nll_ept=[1000., 1000., 1000.])]
    s = summarize_endpoints([], heads, bootstrap=89, seed=17)
    h = s['nll'][0]['metrics']
    assert h['nll']['ept']['estimate'] == 5
    assert h['normalized_nll']['ept']['estimate'] == 2.5
    assert h['nll']['delta']['estimate'] == -1.5
    structural = s['structural_normalized_nll']
    assert structural['heads'] == ['kill', 'objective']
    assert structural['metrics']['ept']['estimate'] == 2.5
    assert structural['metrics']['delta']['estimate'] == -.75
    assert structural['metrics']['ept']['ci95'] is not None
    assert s == summarize_endpoints([], heads, bootstrap=89, seed=17)
    assert summarize_endpoints([], [nll_head()], bootstrap=0)['structural_normalized_nll']['metrics'] is None


def test_structural_sparse_head_replicates_not_silently_averaged():
    a = nll_head('a', match=['a'], row=['x'], nll_ept=[2.], nll_b1=[3.], common_head=True)
    b = nll_head('b', match=['b'], row=['y'], nll_ept=[4.], nll_b1=[5.], common_head=True)
    s = summarize_endpoints([], [a, b], bootstrap=100)
    result = s['structural_normalized_nll']['metrics']
    assert result['invalid_replicates'] > 0
    assert result['ept']['ci95'] is None
    assert result['ept']['conditional_ci95'] == [1.5, 1.5]


@pytest.mark.parametrize('change', [{'scale': 0}, {'scale': np.inf}, {'scale': True}, {'nll_ept': [1, 2, np.nan]},
                                    {'row': ['a', 'a', 'c']}, {'common_head': True, 'scale': None}])
def test_invalid_nll_rejected(change):
    with pytest.raises(ValueError):
        summarize_endpoints([], [nll_head(**change)], bootstrap=0)


def test_report_unavailable_semantics_and_json_csv_svg_consistency(tmp_path):
    s = summarize_endpoints([endpoint(name='M3 <draw>', module='M3', layer='open', y=[0, 0])], bootstrap=13)
    paths = write_report(s, tmp_path, provenance={'synthetic': True}, render=False)
    assert set(paths) == {'metrics_json', 'metrics_csv', 'reliability_csv', 'counts_csv', 'report', 'svg'}
    assert all(Path(p).is_absolute() and Path(p).is_file() for p in paths.values())
    loaded = json.loads(Path(paths['metrics_json']).read_text(encoding='utf-8'))
    assert loaded['endpoints'] == s['endpoints']
    assert loaded['provenance'] == {'synthetic': True}
    rows = list(csv.DictReader(Path(paths['metrics_csv']).open(encoding='utf-8-sig')))
    auc = next(r for r in rows if r['metric'] == 'auc')
    assert auc['ept'] == auc['ept_ci95_low'] == auc['ept_conditional_ci95_low'] == 'unavailable'
    brier = next(r for r in rows if r['metric'] == 'brier')
    assert float(brier['ept']) == metric(s, 'brier')['ept']['estimate']
    text = Path(paths['report']).read_text(encoding='utf-8')
    assert all(term in text for term in ['25%', '전체 FIT', '조건부', '단일 학습 seed', 'unavailable', '학습 13블록'])
    svg = Path(paths['svg']).read_text(encoding='utf-8')
    ET.fromstring(svg)
    assert 'M3 &lt;draw&gt;' in svg
    assert 'EPT: unavailable' in svg
    assert 'nan' not in svg.lower()
    assert not list(tmp_path.glob('*.png'))


def test_report_does_not_hide_nonfinite_metadata_or_create_outputs(tmp_path):
    s = summarize_endpoints([], bootstrap=0)
    target = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='nonfinite'):
        write_report(s, target, provenance={'bad': np.nan})
    assert not target.exists()


def test_no_endpoint_or_nll_aggregate_auc(tmp_path):
    s = summarize_endpoints([], bootstrap=0)
    assert 'auc' not in s
    paths = write_report(s, tmp_path)
    assert 'NLL: unavailable' in Path(paths['svg']).read_text(encoding='utf-8')
