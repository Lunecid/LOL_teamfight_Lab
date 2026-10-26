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


def test_default_comparison_text_stays_the_t003_whole_fit_pilot_wording():
    # The T003 path (EPT 25% pilot versus the existing whole-FIT B1) is described accurately by this text.
    s = summarize_endpoints([endpoint()], bootstrap=0)
    assert s['comparison'] == 'EPT FIT/STOP 25% pilot versus existing whole-FIT B1; unequal training scope'
    assert s['bootstrap']['delta'] == 'EPT-B1'
    assert s['limitations'] == ['Single training seed; CI covers match sampling, not training-seed variation.',
                                'M3/M4 are conditional layers, not whole t0 or composed-value performance.',
                                'No aggregate AUC and no all-training-block NLL sum.']


def test_comparison_delta_and_limitation_text_are_parameters():
    s = summarize_endpoints([endpoint()], bootstrap=0, comparison='matched design: A versus C',
                            delta_label='A-C', limitations=['first', 'second'])
    assert s['comparison'] == 'matched design: A versus C'
    assert s['bootstrap']['delta'] == 'A-C' and s['limitations'] == ['first', 'second']
    assert 'whole-FIT' not in json.dumps(s)


@pytest.mark.parametrize('options', [{'comparison': ''}, {'comparison': 3}, {'delta_label': ''}, {'delta_label': None},
                                     {'limitations': 'one string'}, {'limitations': [1]}, {'limitations': []}])
def test_comparison_text_options_validated(options):
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


# =====================================================================================================================================
# S4b preparation M11 (decision D16): the structural metric is the mean over the eight heads of R_h = sum w NLL_model / sum w NLL_B0
# =====================================================================================================================================
from c_ept.evaluation import inference
from c_ept.evaluation.metrics import RELATIVE_HEADS, relative_structural

R_NAMES = ('M2Rate/masked', 'M2Mark/masked', 'M2Zone/masked', 'M3Size/open', 'M3Kills/open', 'M3Gold/open', 'M3Obj/open', 'M4Conv/open')


def rel_record(name, match, ept, b0, row=None, counts=None):
    """One NLL record as ``run_inference`` writes it (the HR-5 rows of a head, the model's and B0's NLL of each row)."""
    n = len(match)
    return dict(name=name, match=np.asarray(match, dtype=np.int64), row=np.arange(n) if row is None else np.asarray(row, dtype=np.int64),
                nll_ept=np.asarray(ept, dtype=float), nll_b1=np.asarray(ept, dtype=float), nll_b0=np.asarray(b0, dtype=float),
                counts=counts if counts is not None else {'eligible': n})


def rel_records(match, ept, b0, factors=(1, 1, 1, 1, 1, 1, 1, 1)):
    """The eight structural heads on the same rows; head i has the model NLL ``factors[i] * ept``. Plus a masked block that is not structural."""
    out = [rel_record(name, match, np.asarray(ept, dtype=float) * f, b0) for name, f in zip(R_NAMES, factors)]
    out.append(rel_record('M3Size/masked', match, ept, b0))
    return out


def test_the_relative_heads_are_the_eight_structural_heads_of_the_decision():
    assert RELATIVE_HEADS == R_NAMES


def test_relative_nll_hand_example_has_the_match_weighted_ratio_per_head_and_the_mean_of_eight():
    """Two matches, rows [0, 0, 1]: the match means of the model are 3 and 6 (rows 2, 4 | 6) and of B0 2 and 3 (1, 3 | 3), so R = (3 + 6) / (2 + 3) = 1.8; the pooled row ratio 12 / 7 is another number.
    Head i has the model NLL f_i times that: R_i = 1.8 f_i and the structural metric is the mean of the eight, 1.8 * mean(f) = 2.025."""
    f = (1, 2, .5, 1.5, 1, 1, 1, 1)
    records = rel_records([0, 0, 1], [2, 4, 6], [1, 3, 3], f)
    matrix = inference.resample_matrix(2, B=40)
    out = relative_structural(records, matrix, ids=[0, 1])
    assert out['status'] == 'available' and out['reason'] is None and out['n_matches'] == 2 and out['bootstrap_B'] == 40
    assert abs(12 / 7 - 1.8) > .05
    for name, fi in zip(R_NAMES, f):
        head = out['secondary']['per_head'][name]
        assert head['status'] == 'available' and head['metrics']['ept']['estimate'] == pytest.approx(1.8 * fi, rel=1e-14, abs=0), name
        assert head['metrics']['b0']['estimate'] == 1.0
    assert set(out['secondary']['per_head']) == set(R_NAMES)                                       # the non-structural masked block is not a head of the metric
    est = out['metrics']['ept']['estimate']
    assert est == pytest.approx(1.8 * np.mean(f), rel=1e-14, abs=0) and out['metrics']['b0']['estimate'] == 1.0
    assert out['metrics']['delta']['estimate'] == pytest.approx(est - 1.0, rel=1e-14, abs=0)


def test_b0_through_the_relative_metric_scores_exactly_one_per_head_with_a_degenerate_interval():
    """B0 as the model: the numerator and the denominator are the same numbers, so every R_h and their mean are 1.0 exactly, in every resample (the interval is [1, 1])."""
    rng = np.random.default_rng(3)
    match = np.repeat(np.arange(5), [3, 1, 4, 2, 6])
    nll = rng.uniform(.1, 4., len(match))
    records = rel_records(match, nll, nll)
    out = relative_structural(records, inference.resample_matrix(5, B=300), ids=list(range(5)))
    assert out['metrics']['ept']['estimate'] == 1.0 and out['metrics']['ept']['ci95'] == [1.0, 1.0] and out['metrics']['delta']['estimate'] == 0.0
    for name in R_NAMES:
        m = out['secondary']['per_head'][name]['metrics']
        assert m['ept']['estimate'] == 1.0 and m['ept']['ci95'] == [1.0, 1.0] and m['delta']['ci95'] == [0.0, 0.0], name
    assert out['metrics']['ci_status'] == 'available' and out['metrics']['invalid_replicates'] == 0


def test_relative_nll_uses_the_same_rows_and_aggregation():
    """R_h is the ratio of the two match-balanced means of the SAME rows of the head (weights 1/n_m of the head's rows, the aggregation of ``summarize_endpoints``): the ratio of its two columns. A head that
    lacks a match leaves that match out of both sums (a_m = d_m = 0 on the common id list). The bootstrap interval is the percentile interval of the ratio estimator on the common matrix, written out here
    by hand with python loops (numerator and denominator recomputed together on each resample)."""
    rng = np.random.default_rng(11)
    heads = {}
    for i, name in enumerate(R_NAMES):
        counts = [2 + i % 3, 3, 1 + i % 2, 4, 1, 2 + (i + 1) % 3]
        if i == 4:
            counts[3] = 0                                                                      # the fifth head has no row of match 3
        match = np.repeat(np.arange(6), counts)
        heads[name] = (match, rng.uniform(.5, 3., len(match)), rng.uniform(.5, 3., len(match)))
    records = [rel_record(name, *heads[name]) for name in R_NAMES]
    matrix = inference.resample_matrix(6, B=250)
    out = relative_structural(records, matrix, ids=list(range(6)))
    assert out['n_matches'] == 6 and out['metrics']['invalid_replicates'] == 0
    summary = summarize_endpoints([], [dict(name=n, match=m, row=np.arange(len(m)), nll_ept=e, nll_b1=b) for n, (m, e, b) in heads.items()], bootstrap=0)
    ratios = []
    for entry in summary['nll']:
        point = entry['metrics']['nll']
        ratios.append(point['ept']['estimate'] / point['b1']['estimate'])
        got = out['secondary']['per_head'][entry['name']]['metrics']['ept']['estimate']
        assert got == pytest.approx(ratios[-1], rel=1e-12, abs=0), entry['name']
    assert out['metrics']['ept']['estimate'] == pytest.approx(np.mean(ratios), rel=1e-12, abs=0)
    # the interval by hand: per-match means a_m, d_m of the head (0 for an absent match), T*_b = sum_m c_bm a_m / sum_m c_bm d_m, the mean over the heads for the structural metric
    hand = {}
    for name, (match, e, b) in heads.items():
        a, d = [0.] * 6, [0.] * 6
        for m in range(6):
            sel = [j for j in range(len(match)) if match[j] == m]
            if sel:
                a[m], d[m] = sum(e[j] for j in sel) / len(sel), sum(b[j] for j in sel) / len(sel)
        hand[name] = [sum(matrix[r, m] * a[m] for m in range(6)) / sum(matrix[r, m] * d[m] for m in range(6)) for r in range(len(matrix))]
        ci = out['secondary']['per_head'][name]['metrics']['ept']['ci95']
        assert ci == pytest.approx(np.quantile(hand[name], [.025, .975]).tolist(), rel=1e-12, abs=0), name
    mean_draws = [np.mean([hand[n][r] for n in R_NAMES]) for r in range(len(matrix))]
    assert out['metrics']['ept']['ci95'] == pytest.approx(np.quantile(mean_draws, [.025, .975]).tolist(), rel=1e-12, abs=0)


def test_a_resample_whose_denominator_is_zero_is_counted_and_the_interval_is_withheld_not_replaced():
    """Few matches: B0's NLL is nonzero in match 0 only (the others are 0), so a resample that draws match 0 no times has the denominator 0 and its ratio does not exist. The point estimate is reported; the interval
    is unavailable (ci_status says why, the invalid replicates are counted) and the conditional interval over the valid resamples is named as such; nothing raises and no value is invented."""
    match = np.array([0, 1, 2])
    records = rel_records(match, [2., 3., 4.], [1., 0., 0.])
    matrix = np.array([[1, 1, 1], [0, 2, 1], [2, 0, 1], [1, 0, 2]], dtype=np.int32)       # row 1 has no draw of match 0
    out = relative_structural(records, matrix, ids=[0, 1, 2])
    head = out['secondary']['per_head']['M2Rate/masked']
    assert head['status'] == 'available'
    m = head['metrics']
    assert m['valid_replicates'] == 3 and m['invalid_replicates'] == 1 and m['ci_status'] == 'unavailable_invalid_replicates'
    assert m['ept']['estimate'] == 9.0 and m['ept']['ci95'] is None                         # (2 + 3 + 4) / (1 + 0 + 0)
    assert m['ept']['conditional_ci95'] == pytest.approx(np.quantile([9., 4., 10.], [.025, .975]).tolist())       # the valid rows: 9 / 1, (2 * 2 + 4) / 2, (2 + 2 * 4) / 1
    assert out['status'] == 'available' and out['metrics']['ci_status'] == 'unavailable_invalid_replicates' and out['metrics']['ept']['ci95'] is None
    assert out['metrics']['invalid_replicates'] == 1                                      # the mean needs all eight heads in a resample


def test_a_head_whose_point_denominator_is_zero_is_unavailable_and_so_is_the_mean():
    records = rel_records([0, 0, 1], [1., 2., 3.], [0., 0., 0.])
    out = relative_structural(records, inference.resample_matrix(2, B=20), ids=[0, 1])
    head = out['secondary']['per_head']['M3Obj/open']
    assert head['status'] == 'unavailable' and head['reason'] == 'zero_denominator' and head['metrics'] is None
    assert out['status'] == 'unavailable' and out['metrics'] is None and 'zero_denominator' in out['reason'] and 'M3Obj/open' in out['reason']


def test_an_unavailable_head_makes_the_mean_unavailable_and_the_other_seven_stay_reported():
    """No hidden average over seven: the decision defines the metric as the mean over the eight heads, so with one head not scored (A7 leaves M2Zone out, reason zone_mark_off) the structural value is
    unavailable and says which head and why; the per-head R_h of the other seven are reported (secondary)."""
    records = rel_records([0, 0, 1], [2., 4., 6.], [1., 3., 3.])
    zone = next(r for r in records if r['name'] == 'M2Zone/masked')
    zone.update(match=np.empty(0, dtype=np.int64), row=np.empty(0, dtype=np.int64), nll_ept=np.empty(0), nll_b1=np.empty(0), nll_b0=np.empty(0),
                counts={'eligible': 0, 'unavailable_reason': 'zone_mark_off'})
    out = relative_structural(records, inference.resample_matrix(2, B=20), ids=[0, 1])
    assert out['status'] == 'unavailable' and out['metrics'] is None and 'M2Zone/masked' in out['reason'] and 'zone_mark_off' in out['reason']
    per = out['secondary']['per_head']
    assert per['M2Zone/masked']['status'] == 'unavailable' and per['M2Zone/masked']['reason'] == 'zone_mark_off' and per['M2Zone/masked']['metrics'] is None
    assert all(per[n]['status'] == 'available' and per[n]['metrics']['ept']['estimate'] == pytest.approx(1.8) for n in R_NAMES if n != 'M2Zone/masked')
    empty = next(r for r in rel_records([0], [1.], [1.]) if r['name'] == 'M3Size/open')            # an empty block without a stated reason is still not a number
    empty.update(match=np.empty(0, dtype=np.int64), row=np.empty(0, dtype=np.int64), nll_ept=np.empty(0), nll_b1=np.empty(0), nll_b0=np.empty(0), counts={'eligible': 0})
    other = relative_structural([empty if r['name'] == 'M3Size/open' else r for r in rel_records([0], [1.], [1.])], inference.resample_matrix(1, B=5), ids=[0])
    assert other['secondary']['per_head']['M3Size/open']['reason'] == 'no_rows' and other['status'] == 'unavailable'


def test_relative_structural_refuses_records_that_are_not_a_complete_paired_set():
    matrix = inference.resample_matrix(2, B=10)
    records = rel_records([0, 0, 1], [2., 4., 6.], [1., 3., 3.])
    with pytest.raises(ValueError, match='M3Obj/open'):
        relative_structural([r for r in records if r['name'] != 'M3Obj/open'], matrix, ids=[0, 1])
    with pytest.raises(ValueError, match='nll_b0'):
        relative_structural([{k: v for k, v in r.items() if k != 'nll_b0'} if r['name'] == 'M4Conv/open' else r for r in records], matrix, ids=[0, 1])
    with pytest.raises(ValueError, match='duplicate'):
        relative_structural(records + [records[0]], matrix, ids=[0, 1])
    with pytest.raises(ValueError, match='length'):
        relative_structural([dict(r, nll_b0=r['nll_b0'][:-1]) if r['name'] == 'M2Mark/masked' else r for r in records], matrix, ids=[0, 1])
    with pytest.raises(ValueError, match='nonfinite'):
        relative_structural([dict(r, nll_b0=np.array([1., np.nan, 3.])) if r['name'] == 'M2Mark/masked' else r for r in records], matrix, ids=[0, 1])
    with pytest.raises(ValueError):                                                      # the matrix has one column for each match of the common list
        relative_structural(records, inference.resample_matrix(3, B=10), ids=[0, 1])
    with pytest.raises(ValueError, match='not in ids'):
        relative_structural(records, inference.resample_matrix(1, B=10), ids=[0])
    with pytest.raises(ValueError, match='sorted'):
        relative_structural(records, matrix, ids=[1, 0])


def test_the_common_id_list_of_the_judging_split_is_a_required_keyword():
    """Fail closed (M11 review m1): a union of the matches that happen to have rows is not the split's list, so the caller must pass it; a call without ``ids`` is a TypeError."""
    import inspect
    records = rel_records([0, 0, 1], [2., 4., 6.], [1., 3., 3.])
    p = inspect.signature(relative_structural).parameters['ids']
    assert p.default is inspect.Parameter.empty and p.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match='ids'):
        relative_structural(records, inference.resample_matrix(2, B=10))
    with pytest.raises(ValueError, match='ids'):
        relative_structural(records, inference.resample_matrix(2, B=10), ids=None)
