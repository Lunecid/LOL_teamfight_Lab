"""Tests of the Gate 0 probe (EPT-T013, author decision D31): ``c_ept/run/gate0.py``.

Synthetic data only, CPU only, no real store and no T009 payload. Sections: the pre-registered decision rule (one test per branch), the statistics on hand-made endpoints (every number is checked
against a hand calculation), the outputs, the loader of the T009 receipts on a synthetic run folder (it refuses every changed hash), the evaluation on the toy world (role STOP, primary rows, nominal
exposure, no Monte Carlo) and the supervised entry.

해석: Gate 0 모듈의 시험이다. 판정 규칙의 가지마다 시험 하나, 통계는 손으로 계산한 값과 대조하고, 적재기는 합성 T009 폴더에서 해시가 바뀐 모든 경우를 거부하는지 본다.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from c_ept.run import gate0
from c_ept.run import io as run_io

# ---- the pre-registered decision rule (task record EPT-T013, fixed 2026-10-26 before any number is computed) ---------------------------------------------------


def passed(**over):
    """A result on which every precondition of a decision holds: integrity checks passed, the equivalence proof passed, the primary statistic is there."""
    base = {'integrity': {'passed': True, 'failures': []}, 'equivalence': {'status': 'passed'}, 'primary': {'delta': -0.01, 'ci95': [-0.02, -0.001]},
            'secondary': {'anything': 'at all'}}
    base.update(over)
    return base


@pytest.mark.parametrize('delta, ci, verdict', [
    (-0.01, (-0.02, -0.001), 'GO'),                                   # delta < 0 and the upper end of the interval < 0
    (-0.01, (-0.02, -1e-12), 'GO'),                                   # strictly below 0, however little
    (-0.01, (-0.02, 0.005), 'INCONCLUSIVE'),                          # delta < 0 and the interval contains 0
    (-0.01, (-0.02, 0.0), 'INCONCLUSIVE'),                            # the upper end touches 0: the interval contains 0, not GO
    (-0.01, (0.0, 0.02), 'INCONCLUSIVE'),                             # the lower end touches 0
    (0.0, (-0.02, -0.01), 'NO-GO'),                                   # delta = 0 is NO-GO, whatever the interval says
    (-0.0, (-0.02, 0.01), 'NO-GO'),
    (0.004, (-0.01, 0.02), 'NO-GO'),                                  # delta > 0
    (0.004, (0.001, 0.02), 'NO-GO')])
def test_the_verdict_follows_the_pre_registered_rule_branch_by_branch(delta, ci, verdict):
    answer = gate0.decide(passed(primary={'delta': delta, 'ci95': list(ci)}))
    assert answer['verdict'] == verdict
    assert answer['rule_text'] == gate0.RULES[verdict] and answer['void_reasons'] == []


def test_a_point_below_zero_with_an_interval_wholly_above_zero_is_not_a_go_and_says_so():
    """The rule has no branch for it (delta < 0, interval above 0: a percentile interval need not hold its point estimate); the third branch of the D13 C4 code takes everything that is neither a stop nor a
    proceed: INCONCLUSIVE, with a note."""
    answer = gate0.decide(passed(primary={'delta': -0.001, 'ci95': [0.002, 0.01]}))
    assert answer['verdict'] == 'INCONCLUSIVE' and answer['notes'] and 'above 0' in answer['notes'][0]


@pytest.mark.parametrize('result', [
    passed(integrity={'passed': False, 'failures': ['the arms are not on the same rows']}),                # an integrity check failed
    passed(integrity={'passed': True, 'failures': ['x']}),                                                 # passed but failures listed: inconsistent, void
    passed(integrity=None), passed(integrity={'failures': []}), passed(integrity={'passed': 1, 'failures': []}),
    passed(equivalence={'status': 'failed'}), passed(equivalence={'status': 'not_recorded'}), passed(equivalence=None), passed(equivalence={}),
    passed(equivalence={'status': 'passed '}),
    passed(primary=None), passed(primary={}), passed(primary={'delta': None, 'ci95': [-1.0, 0.0]}), passed(primary={'delta': float('nan'), 'ci95': [-1.0, 0.0]}),
    passed(primary={'delta': -0.01, 'ci95': None}), passed(primary={'delta': -0.01, 'ci95': [-0.02]}), passed(primary={'delta': -0.01, 'ci95': [0.0, -0.02]}),
    passed(primary={'delta': -0.01, 'ci95': [float('nan'), 0.01]}), passed(primary={'delta': -0.01, 'ci95': [-0.02, float('inf')]}),
    passed(primary={'delta': True, 'ci95': [-0.02, -0.01]}), passed(primary={'delta': '-0.01', 'ci95': [-0.02, -0.01]})])
def test_an_integrity_or_equivalence_failure_or_a_primary_statistic_that_is_missing_or_malformed_is_void_never_a_verdict(result):
    answer = gate0.decide(result)
    assert answer['verdict'] == 'VOID' and answer['void_reasons'] and answer['rule_text'] == gate0.RULES['VOID']


def test_the_equivalence_proof_may_have_explained_differences_and_the_run_is_still_decided():
    assert gate0.decide(passed(equivalence={'status': 'differences_explained'}))['verdict'] == 'GO'
    assert gate0.EQUIVALENCE_STATUSES == ('passed', 'differences_explained', 'failed')


def test_secondary_results_never_change_the_verdict_and_decide_changes_nothing_it_is_given():
    plain, loud = passed(), passed(secondary={'ept_minus_n2': {'delta': -9.0, 'ci95': [-10.0, -8.0]}, 'nll': 'huge', 'brier_skill': float('nan')})
    before = json.dumps(loud, default=str, sort_keys=True)
    assert gate0.decide(plain)['verdict'] == gate0.decide(loud)['verdict'] == 'GO'
    assert json.dumps(loud, default=str, sort_keys=True) == before
    no_go = passed(primary={'delta': 0.1, 'ci95': [0.05, 0.15]}, secondary={'ept_minus_n2': {'delta': -9.0, 'ci95': [-10.0, -8.0]}})        # a strong EPT - N2 difference does not overturn NO-GO
    assert gate0.decide(no_go)['verdict'] == 'NO-GO'


def test_the_rule_texts_are_those_of_the_task_record():
    """The four texts are the task record's, word for word where it fixes a number or a condition."""
    assert set(gate0.RULES) == set(gate0.VERDICTS) == {'GO', 'INCONCLUSIVE', 'NO-GO', 'VOID'}
    assert gate0.RULES['GO'].startswith('Delta < 0 and the upper end of the interval < 0') and 'K6c, K8, K9, X1' in gate0.RULES['GO']
    assert gate0.RULES['INCONCLUSIVE'].startswith('Delta < 0 and the interval contains 0') and 'inconclusive' in gate0.RULES['INCONCLUSIVE']
    assert gate0.RULES['NO-GO'].startswith('Delta >= 0') and 'engineering stays paused' in gate0.RULES['NO-GO'].lower()
    assert 'integrity check fails' in gate0.RULES['VOID'] and 'equivalence proof' in gate0.RULES['VOID']


# ---- the statistics, on hand-made rows --------------------------------------------------------------------------------------------------------------------------
# Four matches 10..13; match 13 has no primary row (it is on the common list and drops out of both sums). Rows: match 10 has 2 rows, 11 has 1, 12 has 3.
IDS = np.array([10, 11, 12, 13])
MATCH = np.array([10, 10, 11, 12, 12, 12])
ROW = np.arange(6)
Y = np.array([1, 0, 1, 0, 0, 1])
P_E = np.array([0.8, 0.3, 0.6, 0.1, 0.2, 0.7])
P_B = np.array([0.5, 0.5, 0.9, 0.3, 0.3, 0.3])
# by hand, per match, mean over its rows of the squared error:   EPT  10: (0.04 + 0.09) / 2 = 0.065;  11: 0.16;  12: (0.01 + 0.04 + 0.09) / 3 = 0.14 / 3
#                                                               B1   10: (0.25 + 0.25) / 2 = 0.25;   11: 0.01;  12: (0.09 + 0.09 + 0.49) / 3 = 0.67 / 3
A = np.array([0.065 - 0.25, 0.16 - 0.01, 0.14 / 3 - 0.67 / 3, 0.0])                    # a_m of (G0-1)
BRIER_E, BRIER_B = (0.065 + 0.16 + 0.14 / 3) / 3, (0.25 + 0.01 + 0.67 / 3) / 3         # weights 1/n_m, three matches present
DELTA = A.sum() / 3


def test_the_brier_contrast_is_the_hand_calculation_of_g0_1_with_one_over_n_m_weights():
    from c_ept.evaluation import proper
    matrix = np.array([[1, 1, 1, 1]])
    out = gate0.brier_contrast(MATCH, Y, P_E, P_B, IDS, matrix)
    assert out['brier_first'] == pytest.approx(BRIER_E, rel=1e-13) and out['brier_second'] == pytest.approx(BRIER_B, rel=1e-13)
    assert out['delta'] == pytest.approx(DELTA, rel=1e-13) and out['delta'] == pytest.approx(out['brier_first'] - out['brier_second'], abs=1e-15)
    weight = 1.0 / np.bincount(MATCH - 10)[MATCH - 10]                                   # the library's weighted Brier gives the same two numbers
    assert out['brier_first'] == pytest.approx(proper.brier(P_E, Y, weight), rel=1e-13) and out['brier_second'] == pytest.approx(proper.brier(P_B, Y, weight), rel=1e-13)
    assert (out['n_matches'], out['n_present_matches'], out['n_rows']) == (4, 3, 6)


def test_the_bootstrap_interval_is_the_percentile_interval_of_the_ratio_over_the_given_resamples_by_hand():
    """(G0-2): numerator and denominator are recomputed together on each resample; the resample rows below are chosen so that the six draws are hand-computable."""
    matrix = np.array([[1, 1, 1, 1], [3, 0, 0, 1], [0, 2, 1, 1], [0, 0, 4, 0], [2, 2, 0, 0], [1, 0, 2, 1]])
    out = gate0.brier_contrast(MATCH, Y, P_E, P_B, IDS, matrix)
    num = matrix @ A
    den = matrix @ np.array([1.0, 1.0, 1.0, 0.0])
    draws = num / den                                                                   # c.a / c.d: the match 13 has d = 0 and does not count in a resample's denominator
    assert den.tolist() == [3.0, 3.0, 3.0, 4.0, 4.0, 3.0]
    assert out['ci95'] == pytest.approx(np.quantile(draws, [0.025, 0.975]).tolist(), rel=1e-13) and out['ci_status'] == 'available'
    assert out['bootstrap_B'] == 6 and out['draws_undefined'] == 0
    from c_ept.evaluation import inference
    assert out['p'] == pytest.approx(inference.two_sided_p(draws, DELTA))


def test_a_resample_without_a_present_match_has_no_ratio_so_the_interval_is_missing_and_says_why():
    matrix = np.array([[1, 1, 1, 1], [0, 0, 0, 4]])                                    # the second resample draws only the match without rows
    out = gate0.brier_contrast(MATCH, Y, P_E, P_B, IDS, matrix)
    assert out['ci95'] is None and out['ci_status'] == 'undefined_resamples' and out['draws_undefined'] == 1 and out['p'] is None
    assert out['delta'] == pytest.approx(DELTA, rel=1e-13)                              # the point estimate is still there


@pytest.mark.parametrize('what', ['rows_outside_ids', 'bad_probability', 'bad_outcome', 'shape'])
def test_the_brier_contrast_refuses_what_is_not_a_probability_an_outcome_or_a_row_of_the_split(what):
    match, y, p = MATCH.copy(), Y.copy(), P_E.copy()
    if what == 'rows_outside_ids':
        match[0] = 99
    elif what == 'bad_probability':
        p[0] = 1.5
    elif what == 'bad_outcome':
        y[0] = 2
    else:
        p = p[:-1]
    with pytest.raises(ValueError):
        gate0.brier_contrast(match, y, p, P_B, IDS, np.array([[1, 1, 1, 1]]))


def test_the_brier_skill_is_one_minus_the_ratio_to_the_constant_forecast_of_the_weighted_base_rate():
    """(G0-4): the base rate is the match-balanced mean of the outcome, the base Brier that of the constant forecast on the same rows and weights."""
    out = gate0.brier_skill(MATCH, Y, P_E, IDS)
    per_match = {m: Y[MATCH == m] for m in (10, 11, 12)}
    pbar = np.mean([v.mean() for v in per_match.values()])                              # (0.5 + 1 + 1/3) / 3
    base = np.mean([np.mean((pbar - v) ** 2) for v in per_match.values()])
    assert out['base_rate'] == pytest.approx(pbar, rel=1e-13) and out['brier_base'] == pytest.approx(base, rel=1e-13)
    assert out['brier_model'] == pytest.approx(BRIER_E, rel=1e-13) and out['skill'] == pytest.approx(1 - BRIER_E / base, rel=1e-13)
    constant = gate0.brier_skill(MATCH, Y, np.full(6, pbar), IDS)
    assert constant['skill'] == pytest.approx(0.0, abs=1e-13)                           # the base forecast has no skill against itself
    degenerate = gate0.brier_skill(MATCH[:1], np.array([1]), np.array([0.5]), IDS)
    assert degenerate['skill'] is None and degenerate['brier_base'] == 0.0              # all outcomes equal: the base Brier is 0 and the skill is undefined


NLL_MATCH = np.array([10, 10, 12])
NLL_ROW = np.array([3, 4, 9])


def test_the_nll_contrast_of_a_head_is_the_match_balanced_mean_difference_with_its_interval():
    first, second = np.array([1.0, 3.0, 2.0]), np.array([0.5, 1.0, 3.5])
    matrix = np.array([[1, 1, 1, 1], [2, 0, 2, 0], [0, 0, 4, 0]])
    out = gate0.nll_contrast(NLL_MATCH, NLL_ROW, first, second, IDS, matrix)
    a = np.array([(0.5 + 2.0) / 2, 0.0, -1.5, 0.0])                                       # the per-match mean of the row differences (match 10 has two rows, match 12 one)
    d = np.array([1.0, 0.0, 1.0, 0.0])
    assert out['status'] == 'available' and out['delta'] == pytest.approx(-0.125) and out['n_rows'] == 3 and out['n_matches_present'] == 2
    assert out['mean_first'] == pytest.approx(2.0) and out['mean_second'] == pytest.approx(2.125)
    draws = (matrix @ a) / (matrix @ d)
    assert out['ci95'] == pytest.approx(np.quantile(draws, [0.025, 0.975]).tolist())


def test_an_empty_head_is_unavailable_with_its_reason_and_no_other_head_is_averaged_in():
    out = gate0.nll_contrast(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0), np.empty(0), IDS, np.array([[1, 1, 1, 1]]), reason='no_STOP_in_scope_units')
    assert out['status'] == 'unavailable' and out['reason'] == 'no_STOP_in_scope_units' and out['delta'] is None and out['n_rows'] == 0


def test_the_nll_contrast_refuses_a_non_finite_value():
    with pytest.raises(ValueError):
        gate0.nll_contrast(NLL_MATCH, NLL_ROW, np.array([1.0, np.nan, 2.0]), np.array([0.5, 1.0, 3.5]), IDS, np.array([[1, 1, 1, 1]]))


CALL_PROVENANCE = {'role': 'STOP', 'exposure': 'nominal', 'population': 'primary', 'monte_carlo': False, 'draws': 0, 'frame_averaged': True}      # what a closed-form STOP call on the primary rows says of itself


def make_arm(name, p_model, *, nll_shift=0.0, heads=('M2Rate/masked', 'M3Size/open')):
    """An arm on the hand-made rows: the three M2 occurrence endpoints (same rows, the given probabilities of the model, the B1 probabilities P_B) and two NLL records (the second head empty)."""
    endpoints = {f'M2/t0/H{h}/occurrence': {'match': MATCH, 'row': ROW, 'y': Y, 'p_model': p_model, 'p_b1': P_B} for h in (60, 120, 180)}
    nll = {heads[0]: {'match': NLL_MATCH, 'row': NLL_ROW, 'nll_model': np.array([1.0, 3.0, 2.0]) + nll_shift, 'nll_b1': np.array([0.5, 1.0, 3.5]), 'reason': None},
           heads[1]: {'match': np.empty(0, dtype=np.int64), 'row': np.empty(0, dtype=np.int64), 'nll_model': np.empty(0), 'nll_b1': np.empty(0), 'reason': 'no_STOP_in_scope_units'}}
    return {'name': name, 'endpoints': endpoints, 'nll': nll, 'ids': IDS, 'provenance': dict(CALL_PROVENANCE)}


def test_analyze_gives_the_primary_statistic_and_every_secondary_row_from_two_arms():
    matrix = np.array([[1, 1, 1, 1], [3, 0, 0, 1], [0, 2, 1, 1]])
    p_n2 = np.clip(P_E + 0.05, 0, 1)
    out = gate0.analyze({'ept': make_arm('ept', P_E), 'n2': make_arm('n2', p_n2, nll_shift=-0.25)}, IDS, matrix)
    primary = out['primary']
    assert primary['endpoint'] == 'M2/t0/H60/occurrence' and primary['contrast'] == 'ept_minus_b1' and primary['metric'] == 'brier'
    assert primary['delta'] == pytest.approx(DELTA, rel=1e-13) and primary['brier_first'] == pytest.approx(BRIER_E, rel=1e-13) and primary['bootstrap_B'] == 3
    assert primary['rows_per_match'] == [2, 1, 3, 0] and primary['n_matches'] == 4 and primary['n_present_matches'] == 3
    brier = out['secondary']['brier']
    assert set(brier) == {'ept_minus_b1', 'n2_minus_b1', 'ept_minus_n2'} and all(set(v) == {'H60', 'H120', 'H180'} for v in brier.values())
    assert brier['ept_minus_b1']['H60']['delta'] == primary['delta']
    assert brier['n2_minus_b1']['H120']['delta'] == pytest.approx(gate0.brier_contrast(MATCH, Y, p_n2, P_B, IDS, matrix)['delta'])
    assert brier['ept_minus_n2']['H180']['delta'] == pytest.approx(gate0.brier_contrast(MATCH, Y, P_E, p_n2, IDS, matrix)['delta'])
    nll = out['secondary']['nll_per_head']
    assert set(nll) == {'ept_minus_b1', 'n2_minus_b1', 'ept_minus_n2'} and set(nll['ept_minus_b1']) == {'M2Rate/masked', 'M3Size/open'}
    assert nll['ept_minus_b1']['M2Rate/masked']['delta'] == pytest.approx(-0.125) and nll['n2_minus_b1']['M2Rate/masked']['delta'] == pytest.approx(-0.125 - 0.25)
    assert nll['ept_minus_n2']['M2Rate/masked']['delta'] == pytest.approx(0.25)                                   # EPT - N2: the same rows, N2 is 0.25 lower
    assert nll['ept_minus_b1']['M3Size/open']['status'] == 'unavailable' and nll['ept_minus_b1']['M3Size/open']['reason'] == 'no_STOP_in_scope_units'
    assert set(out['secondary']['brier_skill']) == {'ept', 'n2', 'b1'} and out['secondary']['brier_skill']['b1']['H60']['brier_model'] == pytest.approx(BRIER_B, rel=1e-13)
    assert 'no cross-head average' in out['secondary']['nll_note']
    json.dumps(out, allow_nan=False)                                                                            # strict JSON: plain numbers only


def test_analyze_without_the_n2_arm_still_gives_the_primary_and_marks_what_needs_n2_unavailable():
    matrix = np.array([[1, 1, 1, 1]])
    out = gate0.analyze({'ept': make_arm('ept', P_E)}, IDS, matrix)
    assert out['primary']['delta'] == pytest.approx(DELTA, rel=1e-13)
    assert out['secondary']['brier']['n2_minus_b1']['H60']['status'] == 'unavailable' and 'n2' in out['secondary']['brier']['ept_minus_n2']['H60']['reason']
    assert out['secondary']['nll_per_head']['ept_minus_n2']['M2Rate/masked']['status'] == 'unavailable'


def test_analyze_without_the_ept_arm_has_no_primary_statistic():
    out = gate0.analyze({'n2': make_arm('n2', P_E)}, IDS, np.array([[1, 1, 1, 1]]))
    assert out['primary'] is None and out['secondary']['brier']['n2_minus_b1']['H60']['status'] == 'available'


def test_the_structural_relative_metric_is_unavailable_without_b0_predictions_on_the_same_rows():
    out = gate0.analyze({'ept': make_arm('ept', P_E)}, IDS, np.array([[1, 1, 1, 1]]))
    assert out['secondary']['structural_relative']['status'] == 'unavailable' and 'B0' in out['secondary']['structural_relative']['reason']


def test_the_two_arms_that_are_not_on_the_same_rows_have_no_ept_minus_n2_contrast():
    other = make_arm('n2', P_E)
    for ep in other['endpoints'].values():
        ep['row'] = ep['row'] + 100
    out = gate0.analyze({'ept': make_arm('ept', P_E), 'n2': other}, IDS, np.array([[1, 1, 1, 1]]))
    cell = out['secondary']['brier']['ept_minus_n2']['H60']
    assert cell['status'] == 'unavailable' and 'rows' in cell['reason'] and out['primary']['delta'] == pytest.approx(DELTA, rel=1e-13)
    checks = {c['name']: c for c in gate0.check_arms({'ept': make_arm('ept', P_E), 'n2': other})}
    assert checks['arms_on_the_same_rows']['passed'] is False and checks['b1_probabilities_equal']['passed'] is True


def inference_result(p_model=P_E):
    """A hand-made ``run_inference`` result: the three M2 occurrence endpoints (the model's probability is called ``p_ept`` there, also for N2), one M4 endpoint that the probe ignores, two NLL
    records and a provenance with the ``role_note`` that the probe must not copy."""
    from c_ept.evaluation import endpoints as endpoint_module
    made = [endpoint_module.endpoint(f'M2/t0/H{h}/occurrence', 'M2', 't0', h, 'first_kill_ge1', MATCH, ROW, Y, p_model, P_B, counts={'candidate': 8, 'eligible': 6}) for h in (60, 120, 180)]
    made.append(endpoint_module.endpoint('M4/open/dragon/blue', 'M4', 'open', None, 'dragon_blue_ge1', MATCH, ROW, Y, p_model, P_B, counts={}))
    nll = [{'name': 'M2Rate/masked', 'match': NLL_MATCH, 'row': NLL_ROW, 'nll_ept': np.array([1.0, 3.0, 2.0]), 'nll_b1': np.array([0.5, 1.0, 3.5]), 'counts': {'eligible': 3}},
           {'name': 'M3Size/open', 'match': np.empty(0, dtype=np.int64), 'row': np.empty(0, dtype=np.int64), 'nll_ept': np.empty(0), 'nll_b1': np.empty(0),
            'counts': {'eligible': 0, 'unavailable_reason': 'no_STOP_in_scope_units'}}]
    return {'endpoints': made, 'nll': nll, 'provenance': {**CALL_PROVENANCE,
                                                          'role_note': 'each arm used the STOP rows that its own training run recorded'}}


def test_an_arm_keeps_the_three_m2_endpoints_and_every_nll_record_and_drops_the_role_note(tmp_path):
    arm = gate0.arm_from_inference('ept', inference_result(), IDS)
    assert list(arm['endpoints']) == list(gate0.M2_ENDPOINTS) and list(arm['nll']) == ['M2Rate/masked', 'M3Size/open']
    assert arm['endpoints']['M2/t0/H60/occurrence']['p_model'].tolist() == P_E.tolist() and {k: arm['endpoints']['M2/t0/H60/occurrence']['counts'][k] for k in ('candidate', 'eligible', 'positive', 'matches')} == {'candidate': 8, 'eligible': 6, 'positive': 3, 'matches': 3}
    assert arm['nll']['M3Size/open']['reason'] == 'no_STOP_in_scope_units' and arm['nll']['M2Rate/masked']['reason'] is None
    assert 'role_note' not in arm['provenance'] and arm['provenance']['monte_carlo'] is False
    json.dumps(arm['provenance'], allow_nan=False)
    with pytest.raises(ValueError, match='no endpoint'):
        gate0.arm_from_inference('ept', {**inference_result(), 'endpoints': inference_result()['endpoints'][:2]}, IDS)
    # the arrays go to one file with a hash, and come back bit for bit
    manifest = gate0.save_arm(tmp_path / 'ept', arm)
    again = gate0.load_arm(tmp_path / 'ept', 'ept', manifest, arm['provenance'])
    for key, entry in arm['endpoints'].items():
        assert all(np.array_equal(entry[field], again['endpoints'][key][field]) and entry[field].dtype == again['endpoints'][key][field].dtype for field in ('match', 'row', 'y', 'p_model', 'p_b1'))
    assert all(np.array_equal(entry['nll_model'], again['nll'][key]['nll_model']) and entry['reason'] == again['nll'][key]['reason'] for key, entry in arm['nll'].items())
    assert manifest['arrays_sha256'] == run_io.sha_file(tmp_path / 'ept' / gate0.ARRAYS_FILE)
    json.dumps(manifest, allow_nan=False)
    with pytest.raises(FileExistsError):                                                                 # a saved arm is never written over
        gate0.save_arm(tmp_path / 'ept', arm)


def test_a_changed_or_missing_arrays_file_is_refused_before_one_array_is_read(tmp_path):
    arm = gate0.arm_from_inference('ept', inference_result(), IDS)
    manifest = gate0.save_arm(tmp_path / 'ept', arm)
    path = tmp_path / 'ept' / gate0.ARRAYS_FILE
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(gate0.Gate0Refused, match='changed'):
        gate0.load_arm(tmp_path / 'ept', 'ept', manifest)
    path.unlink()
    with pytest.raises(gate0.Gate0Refused, match='missing'):
        gate0.load_arm(tmp_path / 'ept', 'ept', manifest)


def test_an_arm_holds_the_sorted_id_list_and_refuses_one_that_is_not_sorted_and_unique(tmp_path):
    arm = gate0.arm_from_inference('ept', inference_result(), IDS)
    assert arm['ids'].tolist() == IDS.tolist() and arm['ids'].dtype == np.int64
    again = gate0.load_arm(tmp_path / 'ept', 'ept', gate0.save_arm(tmp_path / 'ept', arm))
    assert again['ids'].tolist() == IDS.tolist()
    for bad in ([13, 12, 11], [10, 10, 11], [], [[10, 11]]):
        with pytest.raises(ValueError, match='sorted, unique'):
            gate0.arm_from_inference('ept', inference_result(), bad)


@pytest.mark.parametrize('change', [{'role': 'DEV'}, {'role': 'TEST'}, {'population': 'all'}, {'exposure': 'observed'}, {'monte_carlo': True}, {'draws': 12}, {'frame_averaged': False},
                                    {'kill_law': 'shifted'}, {'k_condition': 'binned'}, {'seed': 20261022}, {'rng': 'x'}, {'zone_mark': False}, {'b0': {'kill_law': 'poisson'}}, {'monte_carlo': 0}])
def test_the_call_settings_of_an_arm_are_those_of_the_closed_form_stop_call_on_the_primary_rows(change):
    good = make_arm('ept', P_E)
    assert all(c['passed'] for c in gate0.check_arm(good)) and {c['name'] for c in gate0.check_arm(good)} == {'ept_call_settings', 'ept_rows_present'}
    bad = {**good, 'provenance': {**good['provenance'], **change}}
    failed = [c for c in gate0.check_arm(bad) if not c['passed']]
    assert [c['name'] for c in failed] == ['ept_call_settings'] and next(iter(change)) in failed[0]['detail']


def test_an_arm_without_rows_in_an_m2_endpoint_fails_its_check():
    arm = make_arm('n2', P_E)
    arm['endpoints']['M2/t0/H180/occurrence'] = {**arm['endpoints']['M2/t0/H180/occurrence'], 'row': np.empty(0, dtype=np.int64)}
    failed = [c for c in gate0.check_arm(arm) if not c['passed']]
    assert [c['name'] for c in failed] == ['n2_rows_present'] and 'H180' in failed[0]['detail']


def test_two_arms_aggregated_on_different_id_lists_fail_the_check():
    arms = {'ept': make_arm('ept', P_E), 'n2': {**make_arm('n2', P_E), 'ids': np.array([10, 11, 12, 14])}}
    assert {c['name'] for c in gate0.check_arms(arms) if not c['passed']} == {'common_id_list_equal'}


def test_check_arms_passes_for_two_arms_on_the_same_rows_with_the_same_b1_and_fails_for_another_b1():
    arms = {'ept': make_arm('ept', P_E), 'n2': make_arm('n2', P_E)}
    assert all(c['passed'] for c in gate0.check_arms(arms))
    arms['n2']['endpoints']['M2/t0/H60/occurrence'] = {**arms['n2']['endpoints']['M2/t0/H60/occurrence'], 'p_b1': P_B + 0.01}
    failed = {c['name'] for c in gate0.check_arms(arms) if not c['passed']}
    assert failed == {'b1_probabilities_equal'}


def test_b1_is_compared_only_on_the_rows_that_the_two_arms_share_so_a_row_mismatch_is_never_a_b1_mismatch():
    """Review I1: a row mismatch drops the N2 rows only; the B1 check (which voids) is about the same rows, since B1 values of different rows are not comparable."""
    arms = {'ept': make_arm('ept', P_E), 'n2': make_arm('n2', P_E)}
    endpoint = arms['n2']['endpoints']['M2/t0/H60/occurrence']
    arms['n2']['endpoints']['M2/t0/H60/occurrence'] = {**endpoint, 'row': endpoint['row'] + 1, 'p_b1': P_B + 0.01}               # other rows AND another B1 column
    failed = {c['name'] for c in gate0.check_arms(arms) if not c['passed']}
    assert failed == {'arms_on_the_same_rows'}


# ---- the loader of the T009 run, on a synthetic run folder ------------------------------------------------------------------------------------------------------------
# A made-up T009 matched-25% run: the receipts, the heads, the statistics and the B1 are written by this file (not by the T009 code) in the format of ``comparison.runtime`` (stage receipts of schema
# ept-comparison-stage-v1, RUN.json with the campaign configuration), with real S-size EPT and N2 weights of random numbers. Nothing here is a real checkpoint or a real store.
import dataclasses
import gc
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

from c_ept.comparison import models as comparison_models
from c_ept.comparison import protocol as P
from c_ept.comparison import runtime as comparison_runtime

ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = '20261025_c_ept_matched25_9'
OLD_COMMIT_T009 = '681a3cd' + '0' * 33
SPEC = {'ept': comparison_models.specification('ept'), 'n2': comparison_models.specification('n2')}


class Overlay:
    """The toy S1 with the arrays that the heads world lacks (an empty objectives table)."""

    def __init__(self, base, arrays):
        self.base, self.arrays = base, arrays

    def array(self, key):
        return self.arrays[key] if key in self.arrays else self.base.array(key)

    def __getattr__(self, name):
        return getattr(self.base, name)


@pytest.fixture(scope='module')
def pieces(tmp_path_factory):
    """The made-up inputs of a T009 run: a toy world with three STOP matches, the selected membership, the statistics, a real small B1 (saved once), the weights of EPT-S and N2 and the static channels."""
    import torch
    from c_ept.comparison.data import select_membership
    from c_ept.ept import data, encoder
    from c_ept.models import baselines_b1 as B1
    from c_ept.preprocessing import features
    from c_ept.tests.ept_toys import build_ept_world, toy_static
    folder = tmp_path_factory.mktemp('gate0_pieces')
    world = build_ept_world(folder / 'world', roles=('FIT', 'FIT', 'FIT', 'FIT', 'STOP', 'STOP', 'STOP', 'DEV'), n_events=7, n_edges=5)
    n_matches = len(np.asarray(world.s1.array('matches/role')))
    s1 = Overlay(world.s1, {'matches/obj_off': np.zeros(n_matches + 1, dtype=np.int64), **{f'objectives/{k}': np.empty(0, dtype=np.int64) for k in ('t', 'kind', 'sub', 'taker')}})
    membership = select_membership(s1, share=.25, selection_seed=20261022)
    stats = data.fit_stats(s1, world.s2, matches=membership.fit)
    dropped = features.dropped_columns(world.s2, np.flatnonzero(features.split_views(world.s1, world.s2, 'FIT')))
    small = {'num_leaves': 7, 'min_data_in_leaf': 5, 'num_iterations': 12, 'min_data_per_group': 5, 'cat_smooth': 1.0, 'num_threads': 1}
    b1 = B1.B1Heads.fit(world.s1, world.s2, world.labels, dropped, params=small)
    b1.save(folder / 'b1')
    static = toy_static()
    torch.manual_seed(5)
    states = {name: comparison_models.build_model(encoder.SIZES['S'], static, SPEC[name]).state_dict() for name in ('ept', 'n2')}
    yield SimpleNamespace(world=world, world_root=folder / 'world', s1=s1, membership=membership, stats=stats, b1_folder=folder / 'b1', static=static, states=states, bundle=SimpleNamespace(role='DEV', s1=s1, static=static))
    del world, s1
    gc.collect()


class SyntheticT009:
    """Writes a synthetic T009 run folder under ``root`` and loads it; ``head_changes`` / ``info_changes`` / ``identity_changes`` change the final head, the fit receipt details and the training identity before they
    are sealed (so that a later check, not the hash, is what refuses), ``execution_authorized`` is the flag of the configuration."""

    def __init__(self, root, pieces, *, head_changes=None, info_changes=None, identity_changes=None, execution_authorized=False, prepare_rows_stats=None, states=None):
        import torch
        from c_ept.ept import encoder
        self.pieces, self.root = pieces, Path(root)
        states = states or pieces.states
        self.run = self.root / RUN_NAME
        from c_ept import config as C
        from c_ept import legacy
        self.config = P.read_json(ROOT / 'configs/ept_comparison.example.json')            # the v1 template of this branch, made into the v2 configuration that the T009 operator wrote
        self.config['schema'] = 'ept-comparison-operator-v2'
        self.config['output'] = str(self.run)
        self.config['source_commit'] = OLD_COMMIT_T009
        self.config['execution_authorized'] = execution_authorized
        self.config['inputs']['s3'] = {'root': str(self.root / 's3'), 'files': {'RUN.json': None, 'STAGES.json': None}}
        self.config['inputs']['scales'] = {'root': str(self.root / 'scales'), 'files': {'B0_SCALES.json': None}}
        self.config['inputs']['rules'] = {'root': str(self.root / 'rules'), 'files': {'RULE_PAYOUTS.json': None}}
        self.config['inputs']['legacy'] = {'root': str(C.LEGACY_ROOT), 'files': legacy.pins()['files']}
        (self.root / 'scales').mkdir(parents=True)
        doc = scales_document()
        (self.root / 'scales' / 'B0_SCALES.json').write_bytes(json.dumps(doc, indent=1).encode('utf-8'))
        self.config['inputs']['scales']['files']['B0_SCALES.json'] = P.sha_file(self.root / 'scales' / 'B0_SCALES.json')
        self.config['expected_scale_provenance'] = {key: doc[key] for key in ('s3_run', 's1', 's2', 'labels_store_sha256', 'model_files_sha256')}
        self.scales_sha = self.config['inputs']['scales']['files']['B0_SCALES.json']
        stats, membership = pieces.stats, pieces.membership
        rows = {'role': 'synthetic', 'stats_sha256': prepare_rows_stats or stats.sha256}
        folder = self.run / 'prepare'
        folder.mkdir(parents=True)
        stats.save(folder / 'STATS.json')
        comparison_runtime.atomic_json(folder / 'membership.json', membership.to_record())
        comparison_runtime.atomic_json(folder / 'inputs.json', {'paths': {'s1': str(pieces.world_root / 'base' / 's1')}, 'toy': {'root': str(pieces.world_root)}, 'note': 'a synthetic inputs record'})
        self.prepare_details = {'stats_sha256': stats.sha256, 'membership_sha256': membership.sha256, 'neural_rows': {'fit': rows, 'stop': rows}}
        self.seal('prepare', None, self.prepare_details, [folder / 'STATS.json', folder / 'membership.json', folder / 'inputs.json'])
        shutil.copytree(pieces.b1_folder, self.run / 'fit' / 'b1' / 'model')
        b1_info = {'architecture': 'lightgbm_b1_original_p9', 'seed': 1, 'membership_sha256': membership.sha256, 'stats_sha256': stats.sha256, 'scale_sha256': self.scales_sha}
        comparison_runtime.atomic_json(self.run / 'fit' / 'b1' / 'result.json', b1_info)
        self.seal('fit', 'b1', b1_info, [*sorted(p for p in (self.run / 'fit' / 'b1' / 'model').rglob('*') if p.is_file()), self.run / 'fit' / 'b1' / 'result.json'])
        for name in ('ept', 'n2'):
            fit = self.run / 'fit' / name
            epoch = fit / 'model' / 'epoch_0014'
            epoch.mkdir(parents=True)
            torch.save(states[name], epoch / 'best.pt')
            pointer = {'folder': 'epoch_0014', 'sha256': P.sha_file(epoch / 'best.pt'), 'bytes': (epoch / 'best.pt').stat().st_size}
            identity = {'model_spec': SPEC[name], 'encoder': dataclasses.asdict(encoder.SIZES['S']), 'fit': rows, 'stop': rows, 'epochs': 30, 'seed': 20261022, 'lr': .001, 'weight_decay': .01,
                        'warmup_steps': 10, 'patience': 2, 'dtype': 'torch.float32', 'threads': 8, 'deterministic_algorithms': True, 'device': 'cuda', 'torch': torch.__version__,
                        'static_sha256': hashlib.sha256(pieces.static.tobytes()).hexdigest(), **(identity_changes or {})}
            head = {'done': True, 'best_epoch': 13, 'best': pointer, 'identity': identity, **(head_changes or {})}
            comparison_runtime.atomic_json(epoch / 'head.json', head)
            (epoch / 'epoch_metrics.json').write_text('{}\n', encoding='utf-8')
            (epoch / 'epoch_metrics.csv').write_text('epoch\n', encoding='utf-8')
            parameters = sum(int(v.numel()) for v in states[name].values())
            info = {'architecture': SPEC[name], 'seed': 20261022, 'epochs_run': 20, 'best_epoch': 13, 'stopped_early': True, 'head': 'model/epoch_0014/head.json', 'head_sha256': P.sha_file(epoch / 'head.json'),
                    'best': pointer, 'model_parameters': parameters, 'membership_sha256': membership.sha256, 'stats_sha256': stats.sha256, 'scale_sha256': self.scales_sha, **(info_changes or {})}
            comparison_runtime.atomic_json(fit / 'result.json', info)
            self.seal('fit', name, info, [epoch / 'head.json', epoch / 'epoch_metrics.json', epoch / 'epoch_metrics.csv', fit / 'result.json'])
        self.write_run_record()

    def write_run_record(self):
        comparison_runtime.atomic_json(self.run / 'RUN.json', {'schema': 'ept-comparison-run-v1', 'config': self.config, 'config_sha256': P.digest(self.config), 'state': 'running', 'stage': 'matched25'})

    def seal(self, stage, model, details, files, **changes):
        folder = self.run / stage / (model or '')
        record = {'schema': 'ept-comparison-stage-v1', 'state': 'completed', 'stage': stage, 'model': model, 'config_sha256': P.digest(self.config), 'details': details,
                  'artifacts': {Path(p).resolve().relative_to(self.run.resolve()).as_posix(): P.sha_file(p) for p in files}, **changes}
        comparison_runtime.atomic_json(folder / 'stage.json', record)

    def reseal(self, stage, model=None, changes=None):
        """Write the receipt of a stage again over the files as they are now (so that an edit of a file is not what refuses), with ``changes`` to the record."""
        folder = self.run / stage / (model or '')
        old = json.loads((folder / 'stage.json').read_text(encoding='utf-8'))
        old['artifacts'] = {rel: P.sha_file(self.run / rel) for rel in old['artifacts']}
        old.update(changes or {})
        comparison_runtime.atomic_json(folder / 'stage.json', old)

    def load(self, *, static=None, models=('ept', 'n2')):
        bundle = SimpleNamespace(**{**vars(self.pieces.bundle), **({} if static is None else {'static': static})})
        return gate0.load_t009(self.run, device='cpu', bundle_loader=lambda meta, check: bundle, models=models)


def scales_document():
    """A complete, valid ``B0_SCALES.json`` record written by hand (the format of ``ept.scales``)."""
    from c_ept.ept import loss, scales
    keys = list(loss.BLOCKS)
    values = {k: 2.0 for k in keys}
    return {'schema': 'b0_scales_v1', 's3_run': '20261022_c_ept_s3_3', 'role': 'FIT', 'rule': scales.RULE, 's1': {'run': '20261020_c_ept_s1_1', 'files_sha256': '1' * 64},
            's2': {'run': '20261021_c_ept_s2_3', 'files_sha256': '2' * 64}, 'labels_store_sha256': '3' * 64,
            'model_files_sha256': {'fit/b0/b0_heads.json': '4' * 64, 'fit/b0/b0_heads.npz': '5' * 64}, 'scales': values, 'n_rows': {k: 10 for k in keys}, 'n_unscored': {k: 0 for k in keys}, 'by_size': {},
            'n_rows_unit': {k: {'M2Rate': 'frame_row', 'M3Gold': 'team_entry'}.get(k.split('/')[0], 'design_row') for k in keys},
            'n_unscored_unit': {k: {'M2Rate': 'time_bin_cell', 'M3Gold': 'team_entry'}.get(k.split('/')[0], 'design_row') for k in keys},
            'size_normalised': sorted(f'{h}/{k}' for h in loss.SIZE_NORMALISED for k in ('masked', 'open')),
            'scales_sha256': hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}


def test_the_loader_returns_every_part_after_checking_the_hashes_and_needs_no_execution_flag(tmp_path, pieces):
    import torch
    for flag in (False, True):                                                         # the T009 flag is not a condition of the probe, whatever it says
        run = SyntheticT009(tmp_path / f'flag_{flag}', pieces, execution_authorized=flag)
        parts = run.load()
        assert parts.config_sha256 == P.digest(run.config) and parts.config['execution_authorized'] is flag
        assert parts.stats.sha256 == pieces.stats.sha256 and parts.membership.sha256 == pieces.membership.sha256 and set(parts.scales) and set(parts.models) == {'ept', 'n2'}
        assert parts.b1.fit_roles == ('FIT',) and parts.b1.stop_roles == ('STOP',)
        for name, model in parts.models.items():                                       # the best state only, loaded strictly: every tensor is the saved one, in evaluation mode
            assert not model.training and all(torch.equal(pieces.states[name][key], value.cpu()) for key, value in model.state_dict().items())
        assert parts.summary['models']['ept']['best_epoch'] == 13 and parts.summary['models']['n2']['best_folder'] == 'epoch_0014'
        assert parts.verified['fit/ept/model/epoch_0014/best.pt'] == parts.summary['models']['ept']['best_sha256']
        assert {'prepare/STATS.json', 'prepare/membership.json', 'fit/b1/result.json', 'fit/ept/model/epoch_0014/head.json', 'fit/n2/stage.json', 'prepare/stage.json'} <= set(parts.verified)
        assert set(parts.summary['receipts']) == {'prepare', 'fit/b1', 'fit/ept', 'fit/n2'} and all(len(v) == 64 for v in parts.verified.values())
        assert parts.summary['execution_authorized_in_config'] is flag
        json.dumps(parts.summary, allow_nan=False)
    only = run.load(models=('n2',))
    assert set(only.models) == {'n2'}


def test_the_stop_bundle_is_the_same_stores_declared_as_stop_and_nothing_else_changes(pieces):
    bundle = pieces.bundle
    stop = gate0.stop_bundle(bundle)
    assert bundle.role == 'DEV' and stop.role == 'STOP' and stop.s1 is bundle.s1 and stop.static is bundle.static


MUTATIONS = {
    'a head file of the EPT changed': lambda run: (run.run / 'fit/ept/model/epoch_0014/head.json').write_bytes((run.run / 'fit/ept/model/epoch_0014/head.json').read_bytes() + b' '),
    'the epoch metrics file changed': lambda run: (run.run / 'fit/n2/model/epoch_0014/epoch_metrics.csv').write_text('epoch\n1\n', encoding='utf-8'),
    'the statistics changed': lambda run: (run.run / 'prepare/STATS.json').write_bytes((run.run / 'prepare/STATS.json').read_bytes() + b' '),
    'the membership changed': lambda run: (run.run / 'prepare/membership.json').write_bytes((run.run / 'prepare/membership.json').read_bytes() + b' '),
    'a B1 file changed': lambda run: next(p for p in sorted((run.run / 'fit/b1/model').rglob('*.json'))).write_bytes(next(p for p in sorted((run.run / 'fit/b1/model').rglob('*.json'))).read_bytes() + b' '),
    'the B1 result changed': lambda run: (run.run / 'fit/b1/result.json').write_bytes((run.run / 'fit/b1/result.json').read_bytes() + b' '),
    'a receipt is missing': lambda run: (run.run / 'fit/n2/stage.json').unlink(),
    'the prepare receipt is missing': lambda run: (run.run / 'prepare/stage.json').unlink(),
    'a stage is not completed': lambda run: run.reseal('fit', 'ept', {'state': 'running'}),
    'a receipt of another configuration': lambda run: run.reseal('fit', 'n2', {'config_sha256': 'f' * 64}),
    'a receipt names another stage': lambda run: run.reseal('fit', 'n2', {'model': 'ept'}),
}


EXPECTED_REFUSAL = {
    'a head file of the EPT changed': 'stage artifact changed: fit/ept/model/epoch_0014/head.json',
    'the epoch metrics file changed': 'stage artifact changed: fit/n2/model/epoch_0014/epoch_metrics.csv',
    'the statistics changed': 'stage artifact changed: prepare/STATS.json',
    'the membership changed': 'stage artifact changed: prepare/membership.json',
    'a B1 file changed': 'stage artifact changed: fit/b1/model/',
    'the B1 result changed': 'stage artifact changed: fit/b1/result.json',
    'a receipt is missing': 'stage.json: regular JSON file required',
    'the prepare receipt is missing': 'stage.json: regular JSON file required',
    'a stage is not completed': 'stage binding incomplete',
    'a receipt of another configuration': 'stage binding incomplete',
    'a receipt names another stage': 'stage identity mismatch'}


@pytest.mark.parametrize('what', sorted(MUTATIONS))
def test_a_changed_or_missing_receipt_artifact_is_refused_by_name(tmp_path, pieces, what):
    run = SyntheticT009(tmp_path, pieces)
    MUTATIONS[what](run)
    with pytest.raises(gate0.Gate0Refused, match=EXPECTED_REFUSAL[what]):
        run.load()


def test_the_best_weights_are_hashed_before_they_are_loaded_and_a_changed_byte_is_refused(tmp_path, pieces):
    run = SyntheticT009(tmp_path, pieces)
    best = run.run / 'fit/ept/model/epoch_0014/best.pt'
    data = bytearray(best.read_bytes())
    data[len(data) // 2] ^= 1                                                          # the same size, one bit flipped: only the hash can tell
    best.write_bytes(bytes(data))
    with pytest.raises(gate0.Gate0Refused, match='best-only payload changed'):
        run.load()
    best.write_bytes(bytes(data) + b'x')                                               # another size
    with pytest.raises(gate0.Gate0Refused, match='invalid best-only file'):
        run.load()
    best.unlink()
    with pytest.raises(gate0.Gate0Refused, match='invalid best-only file'):
        run.load()


@pytest.mark.parametrize('damage', ['a missing key', 'an unexpected key', 'a shape that differs'])
def test_weights_that_are_not_the_architecture_are_refused_by_the_strict_load_even_with_every_hash_right(tmp_path, pieces, damage):
    import torch
    state = dict(pieces.states['ept'])
    key = next(iter(state))
    if damage == 'a missing key':
        del state[key]
    elif damage == 'an unexpected key':
        state['an.extra.weight'] = torch.zeros(2)
    else:
        state[key] = torch.zeros(state[key].numel() + 1)
    run = SyntheticT009(tmp_path, pieces, states={**pieces.states, 'ept': state}, info_changes={'model_parameters': sum(int(v.numel()) for v in pieces.states['ept'].values())})
    with pytest.raises(gate0.Gate0Refused, match='do not load strictly'):
        run.load()


def test_a_best_pointer_that_differs_from_the_final_head_is_refused(tmp_path, pieces):
    run = SyntheticT009(tmp_path, pieces, head_changes={'best_epoch': 12})              # the head says 12, the fit receipt says 13, both files are sealed
    with pytest.raises(gate0.Gate0Refused, match='best pointer differs'):
        run.load()


@pytest.mark.parametrize('setting', [{'zone_mark': False}, {'kill_law': 'shifted'}, {'k_condition': 'binned'}, {'eval_average': False}, {'a_new_setting': 1}])
def test_a_checkpoint_whose_training_has_a_setting_the_evaluation_does_not_pass_on_is_refused(tmp_path, pieces, setting):
    run = SyntheticT009(tmp_path, pieces, identity_changes=setting)
    with pytest.raises(gate0.Gate0Refused, match=next(iter(setting))):
        run.load()


@pytest.mark.parametrize('identity, message', [({'dtype': 'torch.bfloat16'}, 'training settings'), ({'deterministic_algorithms': False}, 'training settings'), ({'threads': 4}, 'training settings'),
                                              ({'seed': 1}, 'training settings'), ({'static_sha256': 'e' * 64}, 'static input differs')])
def test_the_training_identity_must_be_the_matched_profile_and_the_static_input_the_one_of_the_stores(tmp_path, pieces, identity, message):
    run = SyntheticT009(tmp_path, pieces, identity_changes=identity)
    with pytest.raises(gate0.Gate0Refused, match=message):
        run.load()


def test_the_static_channels_of_the_stores_must_be_the_ones_the_model_was_trained_with(tmp_path, pieces):
    run = SyntheticT009(tmp_path, pieces)
    with pytest.raises(gate0.Gate0Refused, match='static input differs'):
        run.load(static=pieces.static + 1.0)


def test_statistics_that_are_not_the_ones_of_the_training_rows_are_refused(tmp_path, pieces):
    run = SyntheticT009(tmp_path, pieces, prepare_rows_stats='d' * 64)                  # the heads and the preparation agree with each other, not with the statistics file
    with pytest.raises(gate0.Gate0Refused, match='statistics differ'):
        run.load()


def test_a_parameter_count_that_differs_from_the_fit_receipt_is_refused(tmp_path, pieces):
    run = SyntheticT009(tmp_path, pieces, info_changes={'model_parameters': 7})
    with pytest.raises(gate0.Gate0Refused, match='parameter count'):
        run.load()


def test_the_scales_file_and_the_campaign_record_must_be_the_ones_the_configuration_binds(tmp_path, pieces):
    run = SyntheticT009(tmp_path / 'a', pieces)
    (run.root / 'scales' / 'B0_SCALES.json').write_bytes((run.root / 'scales' / 'B0_SCALES.json').read_bytes() + b' ')
    with pytest.raises(gate0.Gate0Refused, match='scales file'):
        run.load()
    run = SyntheticT009(tmp_path / 'b', pieces)
    record = json.loads((run.run / 'RUN.json').read_text(encoding='utf-8'))
    record['config']['settings']['batch_size'] = 64                                    # a changed fixed setting of the matched profile
    (run.run / 'RUN.json').write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(gate0.Gate0Refused):
        run.load()
    run = SyntheticT009(tmp_path / 'c', pieces)
    record = json.loads((run.run / 'RUN.json').read_text(encoding='utf-8'))
    record['config']['source_commit'] = 'a' * 40                                        # a valid configuration whose digest is not the recorded one
    (run.run / 'RUN.json').write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(gate0.Gate0Refused, match='digest'):
        run.load()
    run = SyntheticT009(tmp_path / 'd', pieces)
    shutil.copytree(run.run, tmp_path / 'd' / 'moved')                                  # the same files in another folder are not the run of this configuration
    with pytest.raises(gate0.Gate0Refused, match='output'):
        gate0.load_t009(tmp_path / 'd' / 'moved', device='cpu', bundle_loader=lambda meta, check: pieces.bundle)
    shutil.rmtree(tmp_path / 'd' / 'moved')
    (run.run / 'RUN.json').unlink()
    with pytest.raises(gate0.Gate0Refused):
        run.load()


def v1_of(config):
    """The configuration in the format of this branch's protocol module (schema v1, six inputs): not the format of the T009 run."""
    return {**config, 'schema': 'ept-comparison-operator-v1', 'inputs': {k: v for k, v in config['inputs'].items() if k in ('s1', 's2', 'labels', 'terrain', 's3', 'prices')}}


def _set(path, value):
    def change(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return config
    return change


CONFIG_CHANGES = {
    'the v1 schema of this branch': (v1_of, 'comparison profile mismatch'),
    'the v2 schema with the six inputs of v1': (lambda c: {**c, 'inputs': v1_of(c)['inputs']}, 'input identities required'),
    'an extra top-level field': (lambda c: {**c, 'extra': 1}, 'fields mismatch'),
    'another profile': (_set(['profile'], 'other'), 'profile mismatch'),
    'a changed fixed setting': (_set(['settings', 'batch_size'], 64), 'fixed matched25 settings changed'),
    'a setting of another type': (_set(['settings', 'share'], 1), 'fixed matched25 settings changed'),
    'changed limits': (_set(['limits', 'total_seconds'], 1), 'campaign limits changed'),
    'an execution flag that is not a boolean': (_set(['execution_authorized'], 1), 'must be boolean'),
    'another python': (_set(['python'], 'C:/other/python.exe'), 'fixed Python'),
    'a relative output': (_set(['output'], 'runs/x'), 'absolute'),
    'a bad source commit': (_set(['source_commit'], 'abc'), 'invalid source commit'),
    'a bad manifest hash': (_set(['source_manifest_sha256'], 'xyz'), 'invalid source manifest hash'),
    'a relative input root': (_set(['inputs', 's1', 'root'], 'relative'), 'explicit input root'),
    'an input with no files': (_set(['inputs', 'terrain', 'files'], {}), 'file identity map'),
    'a bad input hash': (_set(['inputs', 's1', 'files', 'STORE.json'], 'nothex'), 'invalid input hash'),
    'other files of the scales': (_set(['inputs', 'scales', 'files'], {'other.json': None}), 'input scales are not the fixed ones'),
    'S3 without its lineage files': (_set(['inputs', 's3', 'files'], {'RUN.json': None}), 'S3 lineage'),
    'no scale provenance': (_set(['expected_scale_provenance'], None), 'scale provenance'),
}


@pytest.mark.parametrize('what', sorted(CONFIG_CHANGES))
def test_the_t009_configuration_is_validated_by_the_rules_of_its_own_schema_v2(tmp_path, pieces, what):
    change, message = CONFIG_CHANGES[what]
    run = SyntheticT009(tmp_path, pieces)
    run.config = change(json.loads(json.dumps(run.config)))
    run.write_run_record()
    with pytest.raises(gate0.Gate0Refused, match=message):
        run.load()


def test_a_configuration_of_the_t009_schema_is_valid_whatever_its_execution_flag_says(tmp_path, pieces):
    for flag in (False, True):
        config = SyntheticT009(tmp_path / str(flag), pieces, execution_authorized=flag).config
        assert gate0.validate_t009_config(config) is config
    assert gate0.T009_CONFIG_SCHEMA == 'ept-comparison-operator-v2' and gate0.T009_INPUTS == {'s1', 's2', 'labels', 'terrain', 's3', 'prices', 'scales', 'rules', 'legacy'}
    with pytest.raises(gate0.Gate0Refused):
        gate0.validate_t009_config('not a configuration')


def test_legacy_pins_that_are_not_the_ones_of_the_t009_configuration_are_refused(tmp_path, pieces):
    run = SyntheticT009(tmp_path, pieces)
    run.config['inputs']['legacy']['files'] = {**run.config['inputs']['legacy']['files'], 'gameplay/__init__.py': 'f' * 64}
    run.write_run_record()
    with pytest.raises(gate0.Gate0Refused, match='legacy pins'):
        run.load()


def test_a_membership_that_is_not_the_one_of_the_role_table_is_refused(tmp_path, pieces):
    run = SyntheticT009(tmp_path, pieces)
    other = SimpleNamespace(**{**vars(pieces.bundle), 's1': Overlay(pieces.s1, {'matches/role': np.asarray(pieces.s1.array('matches/role'))[::-1].copy()})})
    with pytest.raises(gate0.Gate0Refused):
        gate0.load_t009(run.run, device='cpu', bundle_loader=lambda meta, check: other)


def test_models_must_be_a_selection_of_ept_and_n2(tmp_path, pieces):
    run = SyntheticT009(tmp_path, pieces)
    for bad in ((), ('b1',), ('ept', 'ept'), ('ept', 'x')):
        with pytest.raises(gate0.Gate0Refused, match='models must be'):
            run.load(models=bad)


def test_the_check_function_is_called_while_the_files_are_hashed(tmp_path, pieces):
    run = SyntheticT009(tmp_path, pieces)
    calls = []
    bundle = pieces.bundle
    gate0.load_t009(run.run, device='cpu', check=lambda: calls.append(1), bundle_loader=lambda meta, check: bundle)
    assert len(calls) > 10
    class Stop(Exception):
        pass

    def stop():
        raise Stop
    with pytest.raises(Stop):
        gate0.load_t009(run.run, device='cpu', check=stop, bundle_loader=lambda meta, check: bundle)


# ---- the common resample matrix and the common id list -------------------------------------------------------------------------------------------------------------
def test_the_bootstrap_matrix_is_the_first_2000_rows_of_the_common_10000_row_matrix_with_the_recorded_hashes():
    from c_ept.evaluation import inference
    matrix, record = gate0.common_matrix(5)
    full = inference.resample_matrix(5)                                                  # B = 10,000, seed 20261022: the matrix of every comparison of the project
    assert full.shape == (10_000, 5) and matrix.shape == (2000, 5) and np.array_equal(matrix, full[:2000])
    assert record == {'numpy': np.__version__, 'seed': 20261022, 'matrix_rows': 10_000, 'matrix_sha256': hashlib.sha256(np.ascontiguousarray(full).tobytes()).hexdigest(), 'B': 2000,
                      'used_sha256': hashlib.sha256(np.ascontiguousarray(full[:2000]).tobytes()).hexdigest(), 'n_matches': 5}
    assert gate0.BOOTSTRAP_B == 2000
    with pytest.raises(ValueError):
        gate0.common_matrix(5, 10_001)
    with pytest.raises(ValueError):
        gate0.common_matrix(5, 0)


def test_the_common_id_list_is_the_sorted_unique_list_of_the_stop_matches_of_the_role_table(pieces):
    from c_ept import config
    ids = gate0.stop_match_ids(pieces.s1)
    roles = np.asarray(pieces.s1.array('matches/role'))
    assert ids.tolist() == np.flatnonzero(roles == config.ROLES.index('STOP')).tolist() == sorted(set(ids.tolist())) and len(ids) == 3
    with pytest.raises(ValueError, match='no STOP match'):
        gate0.stop_match_ids(Overlay(pieces.s1, {'matches/role': np.zeros(len(roles), dtype=roles.dtype)}))


# ---- the evaluation on the toy world: STOP rows, primary population, nominal exposure, no Monte Carlo ----------------------------------------------------------------------
@pytest.fixture(scope='module')
def evaluation(pieces):
    """A STOP bundle on the toy world, a tiny EPT and a tiny N2 of random weights and the constant B1 stand-in: the real inference runs on them on the CPU."""
    import torch
    from c_ept.ept import encoder
    from c_ept.tests.ept_toys import toy_static
    from c_ept.tests.run_evaluate_bodies import constant_b1
    n_units = len(pieces.s1.array('units/f'))
    labels = Overlay(pieces.world.labels, {'labels/f0_unit': np.resize(np.array([-1., 0., 1.]), n_units), 'labels/gold_reps': np.arange(31, dtype=float) * 100})
    bundle = SimpleNamespace(role='STOP', s1=pieces.s1, s2=pieces.world.s2, labels=labels, static=toy_static())
    torch.manual_seed(11)
    cfg = encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=.1, kappas=1, field_channels=2)
    models = {'ept': comparison_models.build_model(cfg, toy_static(), SPEC['ept']), 'n2': comparison_models.build_model(cfg, toy_static(), comparison_models.specification('n2', hidden=8))}
    return SimpleNamespace(bundle=bundle, models=models, stats=pieces.stats, b1=constant_b1(), scales={key: 2.0 for key in __import__('c_ept.ept.loss', fromlist=['BLOCKS']).BLOCKS})


def test_the_evaluation_is_the_closed_form_call_on_the_stop_primary_rows_with_the_nominal_exposure_and_no_draw(evaluation, pieces, monkeypatch):
    from c_ept.evaluation import adapters
    from c_ept import config
    draws = []
    monkeypatch.setattr(adapters, 'sample_open_value', lambda *a, **k: draws.append(1))
    calls = []
    results = gate0.evaluate_gate0(evaluation.bundle, evaluation.models, evaluation.stats, evaluation.b1, device='cpu', check=lambda: calls.append(1), scales=evaluation.scales)
    assert list(results) == ['ept', 'n2'] and draws == [] and len(calls) > 20
    roles = np.asarray(pieces.s1.array('matches/role'))
    stop = set(np.flatnonzero(roles == config.ROLES.index('STOP')).tolist())
    t0, hi = np.asarray(pieces.s1.array('frame_rows/t0')), np.asarray(pieces.s1.array('frame_rows/hi'))
    for name, result in results.items():
        provenance = result['provenance']
        assert (provenance['role'], provenance['exposure'], provenance['population'], provenance['monte_carlo'], provenance['draws']) == ('STOP', 'nominal', 'primary', False, 0)
        assert 'M3/open/positive_value' not in {ep['name'] for ep in result['endpoints']} and len(result['nll']) == 13
        assert {int(m) for ep in result['endpoints'] for m in ep['match']} <= stop and {int(m) for record in result['nll'] for m in record['match']} <= stop
        for horizon in (60, 120, 180):                                                   # every row of an M2 endpoint is a primary row of its horizon: hi - t0 >= H x 1000 ms
            endpoint = next(ep for ep in result['endpoints'] if ep['name'] == f'M2/t0/H{horizon}/occurrence')
            assert len(endpoint['row']) and (hi[endpoint['row']] - t0[endpoint['row']] >= horizon * 1000).all()
            assert endpoint['counts']['population_selected'] >= endpoint['counts']['eligible'] and endpoint['counts']['population_candidate'] >= endpoint['counts']['population_selected']
        assert 'role_note' in provenance                                                 # the call says it (the arm drops this text; the probe states the selection itself)


def test_the_result_is_the_bits_of_the_plain_inference_call_with_those_settings(evaluation):
    from c_ept.evaluation import adapters
    results = gate0.evaluate_gate0(evaluation.bundle, {'ept': evaluation.models['ept']}, evaluation.stats, evaluation.b1, device='cpu', check=lambda: None, scales=evaluation.scales)
    plain = adapters.run_inference(evaluation.bundle, evaluation.models['ept'], evaluation.stats, evaluation.b1, check=lambda: None, device='cpu', batch_size=32, scales=evaluation.scales,
                                   comparison=P.matched_text('EPT', 'B1')['comparison'], role='STOP', exposure='nominal', population='primary', monte_carlo=False)
    assert list(results) == ['ept']
    for one, other in zip(results['ept']['endpoints'], plain['endpoints']):
        assert one['name'] == other['name'] and all(np.array_equal(one[key], other[key]) for key in ('match', 'row', 'y', 'p_ept', 'p_b1'))
    for one, other in zip(results['ept']['nll'], plain['nll']):
        assert one['name'] == other['name'] and all(np.array_equal(one[key], other[key]) for key in ('match', 'row', 'nll_ept', 'nll_b1'))
    assert results['ept']['provenance'] == plain['provenance']


@pytest.mark.parametrize('role', ['DEV', 'TEST', 'FIT', 'CAL', None])
def test_a_bundle_that_does_not_declare_stop_is_refused_before_anything_is_read(evaluation, role):
    class Unread:
        def array(self, name):
            raise AssertionError('a store was read')
    bundle = SimpleNamespace(role=role, s1=Unread(), s2=Unread(), labels=Unread(), static=evaluation.bundle.static)
    with pytest.raises(gate0.Gate0Refused, match='role STOP only'):
        gate0.evaluate_gate0(bundle, evaluation.models, evaluation.stats, evaluation.b1, device='cpu', check=lambda: None)


def test_only_ept_and_n2_are_evaluated(evaluation):
    with pytest.raises(gate0.Gate0Refused, match='models must be'):
        gate0.evaluate_gate0(evaluation.bundle, {'b1': object()}, evaluation.stats, evaluation.b1, device='cpu', check=lambda: None)


def test_with_a_log_each_arm_leaves_one_inference_stage_and_one_begin_end_pair_for_each_of_the_thirteen_head_blocks(evaluation, tmp_path):
    from c_ept.ept import data, loss
    from c_ept.run import evaluate as run_evaluate
    log = run_evaluate.StageLog(tmp_path / 'PROGRESS.jsonl')
    gate0.evaluate_gate0(evaluation.bundle, evaluation.models, evaluation.stats, evaluation.b1, device='cpu', check=lambda: None, scales=evaluation.scales, log=log)
    lines = [json.loads(text) for text in (tmp_path / 'PROGRESS.jsonl').read_text(encoding='utf-8').splitlines()]
    names = [loss.block_key(head, kind) for head, kind in data.ROW_FIELDS]
    for model in ('ept', 'n2'):
        mine = [line for line in lines if line['model'] == model]
        stages = [(line['event'], line['stage']) for line in mine if line['stage'] == 'inference']
        assert stages == [('begin', 'inference'), ('end', 'inference')]
        blocks = [(line['event'], line['detail']['block']) for line in mine if line['stage'] == 'block']
        assert blocks == [(event, name) for name in names for event in ('begin', 'end')]
        end = next(line for line in mine if line['stage'] == 'inference' and line['event'] == 'end')
        assert end['status'] == 'ok' and end['detail'] == {'role': 'STOP', 'pass': 'primary_nominal', 'exposure': 'nominal', 'population': 'primary'} and end['counts']['endpoint_rows'] > 0
        assert all(line['status'] == 'ok' for line in mine if line['event'] == 'end')
    assert [line['model'] for line in lines if line['stage'] == 'inference' and line['event'] == 'begin'] == ['ept', 'n2']       # the arms run one after the other, each with its own stage


# ---- the equivalence record, the timings, the assembly of the result and the outputs ------------------------------------------------------------------------------------
OLD_COMMIT, NEW_COMMIT = '681a3cd' + '0' * 33, 'b077b9f' + '1' * 33
RESULT_IDS = np.arange(100, 130)


def write_equivalence(path, **over):
    record = {'schema': 'ept-gate0-equivalence-v1', 'status': 'passed', 'old_commit': OLD_COMMIT, 'new_commit': NEW_COMMIT, 'keys_compared': 412, 'differences': [], **over}
    path.write_text(json.dumps(record), encoding='utf-8')
    return path


def test_the_equivalence_record_is_read_strictly_and_a_difference_needs_its_cause(tmp_path):
    ok = gate0.read_equivalence(write_equivalence(tmp_path / 'a.json'))
    assert ok['status'] == 'passed' and ok['keys_compared'] == 412 and ok['file_sha256'] == run_io.sha_file(tmp_path / 'a.json')
    explained = gate0.read_equivalence(write_equivalence(tmp_path / 'b.json', status='differences_explained', explanation='the sort order of two keys', differences=[{'key': 'epoch_metrics', 'cause': 'a timestamp'}]))
    assert explained['status'] == 'differences_explained'
    assert gate0.read_equivalence(write_equivalence(tmp_path / 'c.json', status='failed', explanation='the closed-form endpoint differs in 3 keys', differences=[{'key': 'x'}]))['status'] == 'failed'
    for name, over in {'schema': {'schema': 'x'}, 'status': {'status': 'maybe'}, 'old commit': {'old_commit': 'abc'}, 'new commit': {'new_commit': 'G' * 40}, 'no keys': {'keys_compared': 0},
                       'bool keys': {'keys_compared': True}, 'passed with a difference': {'differences': [{'key': 'k', 'cause': 'c'}]}, 'failed without explanation': {'status': 'failed'},
                       'explained without difference': {'status': 'differences_explained', 'explanation': 'e'},
                       'explained without cause': {'status': 'differences_explained', 'explanation': 'e', 'differences': [{'key': 'k'}]},
                       'differences not a list': {'differences': 'none'}}.items():
        with pytest.raises(gate0.Gate0Refused):
            gate0.read_equivalence(write_equivalence(tmp_path / f'{abs(hash(name))}.json', **over))
    with pytest.raises(gate0.Gate0Refused):
        gate0.read_equivalence(tmp_path / 'missing.json')


def write_progress(path, *, models=('ept', 'n2'), skip=()):
    """A progress file with the lines that a run leaves: the supervisor's stage lines, each child's stages and one block pair per block (made with the real ``StageLog`` and a clock that gives 0.5 s per call)."""
    from c_ept.run import evaluate as run_evaluate

    class Step:
        def __init__(self, start):
            self.now = start

        def __call__(self):
            self.now += 0.5
            return self.now
    log = run_evaluate.StageLog(path, wall=Step(1000.0), monotonic=Step(0.0))
    with log.stage('gate0'):
        for model in models:
            with log.stage('child', model):
                with log.stage('arm', model):
                    with log.stage('load', model):
                        pass
                    with log.stage('inference', model, detail={'role': 'STOP'}):
                        for block in ('M2Rate/masked', 'M3Size/open'):
                            if (model, block) in skip:
                                continue
                            with log.stage('block', model, detail={'role': 'STOP', 'pass': 'primary_nominal', 'block': block}) as counts:
                                counts.update(rows=10, views=7)


def test_the_timings_come_from_the_end_lines_and_what_is_not_there_is_none(tmp_path):
    write_progress(tmp_path / 'PROGRESS.jsonl', skip=(('n2', 'M3Size/open'),))
    timing = gate0.timings_from_progress(tmp_path / 'PROGRESS.jsonl')
    assert timing['status'] == 'available' and timing['attempt'] == 1 and timing['total_s'] > 0
    assert timing['ept']['blocks']['M3Size/open'] == {'elapsed_s': 0.5, 'rows': 10, 'views': 7, 'status': 'ok'} and set(timing['ept']['blocks']) == {'M2Rate/masked', 'M3Size/open'}
    assert set(timing['n2']['blocks']) == {'M2Rate/masked'}                                                           # the block that was not run has no entry: not filled in
    assert all(timing[m][key] is not None for m in ('ept', 'n2') for key in ('arm_s', 'load_s', 'inference_s', 'child_s')) and timing['ept']['status'] == 'ok'
    only = tmp_path / 'one'
    only.mkdir()
    write_progress(only / 'PROGRESS.jsonl', models=('ept',))
    timing = gate0.timings_from_progress(only / 'PROGRESS.jsonl')
    assert timing['n2'] == {'arm_s': None, 'load_s': None, 'inference_s': None, 'child_s': None, 'blocks': {}, 'status': None}
    assert gate0.timings_from_progress(tmp_path / 'nothing.jsonl') == {'status': 'unavailable'}
    json.dumps(timing, allow_nan=False)


UNCHANGED = object()
NUMERIC_OK = {**gate0.REQUIRED_NUMERIC, 'sdp_backends': {'flash': True, 'math': True}, 'amp_used_by_this_code': False, 'compile_used_by_this_code': False, 'cublas_workspace_config': ':4096:8',
              'torch_version': '2.5.1', 'cuda_version': '12.1', 'cudnn_version': 90100, 'threads': 8, 'device_name': 'NVIDIA test card'}
CODE_IDENTITY = {'schema': 'ept-run-identity-v1', 'commit': NEW_COMMIT, 'files': {'c_ept/run/gate0.py': 'd' * 64}, 'protected': []}


def verified_files(model):
    return {'prepare/stage.json': '1' * 64, 'prepare/STATS.json': '2' * 64, 'fit/b1/stage.json': '3' * 64, 'fit/b1/result.json': '4' * 64, f'fit/{model}/stage.json': ('5' if model == 'ept' else '6') * 64,
            f'fit/{model}/model/epoch_0014/best.pt': ('7' if model == 'ept' else '8') * 64}


def t009_summary(model):
    return {'config_sha256': 'a' * 64, 'execution_authorized_in_config': False, 'source_commit': OLD_COMMIT, 'profile': 'matched25_ept_b1_n2_v1',
            'config_settings': {'dtype': 'float32', 'tf32': False, 'amp': False, 'native_threads': 8, 'batch_size': 32, 'prefetch': 2, 'n2_hidden': 512, 'neural_seed': 20261022, 'max_epochs': 30},
            'receipts': {'prepare': '1' * 64, 'fit/b1': '3' * 64, f'fit/{model}': '5' * 64}, 'membership': {'sha256': 'b' * 64, 'fit_matches': 4775, 'stop_matches': 401, 'selection_seed': 20261022, 'share': 0.25},
            'stats_sha256': 'c' * 64, 'scales_sha256': 'e' * 64, 'b1': {'dropped_columns': 3, 'dropped_sha256': 'f' * 64, 'fit_roles': ['FIT'], 'stop_roles': ['STOP']}, 'stats_dropped': 0,
            'models': {model: {'architecture': 'ept_s_v1' if model == 'ept' else 'n2_teamset_v1', 'best_epoch': 14 if model == 'ept' else 20, 'best_folder': 'epoch_0014', 'best_sha256': '7' * 64, 'best_bytes': 100,
                               'head_sha256': '9' * 64, 'parameters': 1234, 'training_seed': 20261022, 'identity_torch': '2.5.1', 'identity_dtype': 'torch.float32', 'identity_threads': 8,
                               'identity_device': 'cuda', 'identity_deterministic_algorithms': True}}}


def result_arm(name, quality):
    """An arm of 30 matches with one row each (all outcomes 1 for the matches of the first kind and mixed after): 'better' puts the model clearly closer to the outcome than B1 (0.5), 'worse' clearly
    farther, 'mixed' better on 17 matches and worse on 13 (a mean below 0 with an interval that reaches 0)."""
    n = len(RESULT_IDS)
    y = (np.arange(n) % 2).astype(np.int8)
    good = np.where(y == 1, 0.8, 0.2)
    bad = np.where(y == 1, 0.3, 0.7)
    p = {'better': good, 'worse': bad, 'mixed': np.where(np.arange(n) < 17, good, bad)}[quality]
    endpoints = {f'M2/t0/H{h}/occurrence': {'match': RESULT_IDS, 'row': np.arange(n), 'y': y.astype(bool), 'p_model': p, 'p_b1': np.full(n, 0.5), 'counts': {'candidate': n, 'eligible': n}} for h in (60, 120, 180)}
    nll = {'M2Rate/masked': {'match': RESULT_IDS, 'row': np.arange(n), 'nll_model': np.full(n, 1.0), 'nll_b1': np.full(n, 1.1), 'reason': None, 'counts': {'eligible': n}}}
    return {'name': name, 'endpoints': endpoints, 'nll': nll, 'ids': RESULT_IDS, 'provenance': dict(CALL_PROVENANCE)}


def write_result_run(out, *, quality=('better', 'better'), arm_changes=None, doc_changes=None, stage_changes=None, equivalence_changes=None, start_changes=None):
    """A finished run folder: the two arms (ARM.json and ARRAYS.npz), the progress file, the stage receipts, the equivalence record and the start record; returns the three for ``assemble``."""
    out.mkdir(parents=True, exist_ok=True)
    write_progress(out / 'PROGRESS.jsonl')
    stages = {}
    for model, kind in zip(('ept', 'n2'), quality):
        arm = {**result_arm(model, kind), **((arm_changes or {}).get(model) or {})}
        doc = {'schema': gate0.ARM_SCHEMA, 'model': model, 'manifest': gate0.save_arm(out / model, arm), 'provenance': arm['provenance'],
               'settings': {'batch_size': 32, 'device': 'cuda', 'bundle_loader': gate0.DEFAULT_BUNDLE_LOADER, 'identity_gate': 'start_gate'}, 'numeric': dict(NUMERIC_OK), 't009': t009_summary(model),
               'verified': verified_files(model), 'code_identity': dict(CODE_IDENTITY), 'store_evidence': {}, **((doc_changes or {}).get(model) or {})}
        run_io.write_json_atomic(out / model / 'ARM.json', doc)
        stages[model] = {'state': 'completed', 'exit_code': 0, 'elapsed_s': 10.0, 'cap_s': 10800.0, 'arm_sha256': run_io.sha_file(out / model / 'ARM.json'), **((stage_changes or {}).get(model) or {})}
    equivalence = gate0.read_equivalence(write_equivalence(out / 'equivalence.json', **(equivalence_changes or {})))
    start = {'t009_run': 'D:/LOL_Data/runs/20261025_c_ept_matched25_1', 't009_config_sha256': 'a' * 64, 'approval': 'author approval of 2026-10-26 for EPT-T013', 'e6': {'running': False}, 'code_identity': dict(CODE_IDENTITY),
             'wall_s': 21600, 'stage_caps_s': {'ept': 10800, 'n2': 10800}, 'device': 'cuda', 'bundle_loader': gate0.DEFAULT_BUNDLE_LOADER, 'identity_gate': 'start_gate', 'started_wall': 1000.0,
             **(start_changes or {})}
    return stages, equivalence, start


@pytest.mark.parametrize('quality, verdict', [(('better', 'better'), 'GO'), (('worse', 'better'), 'NO-GO'), (('mixed', 'worse'), 'INCONCLUSIVE')])
def test_a_finished_run_gets_the_verdict_of_the_rule_on_its_primary_statistic(tmp_path, quality, verdict):
    stages, equivalence, start = write_result_run(tmp_path, quality=quality)
    result = gate0.assemble(tmp_path, stages=stages, equivalence=equivalence, start=start)
    assert result['integrity']['passed'] and result['integrity']['failures'] == [] and result['verdict']['verdict'] == verdict and result['verdict']['void_reasons'] == []
    primary = result['primary']
    assert primary['endpoint'] == 'M2/t0/H60/occurrence' and primary['bootstrap_B'] == 2000 and primary['n_matches'] == 30 and primary['n_present_matches'] == 30 and primary['rows_per_match'] == [1] * 30
    assert (primary['delta'] < 0) == (verdict != 'NO-GO') and result['verdict']['delta'] == primary['delta'] and result['verdict']['ci95'] == primary['ci95']
    if verdict == 'GO':
        assert primary['delta'] == pytest.approx(-0.21) and primary['ci95'][1] < 0
    if verdict == 'INCONCLUSIVE':
        assert primary['delta'] < 0 <= primary['ci95'][1]
    assert result['bootstrap']['B'] == 2000 and result['bootstrap']['matrix_rows'] == 10_000 and result['bootstrap']['n_matches'] == 30
    assert result['data']['n_matches'] == 30 and result['data']['role'] == 'STOP' and result['data']['dev_and_test_opened'] is False and result['data']['early_stopping_subset_matches'] == 401
    assert 'Both arms stopped early on the 401-match STOP subset' in result['data']['selection_note'] and 'fit_head(fit_rows, stop_rows)' in result['data']['selection_note']
    assert set(result['secondary']) == {'brier', 'brier_skill', 'nll_per_head', 'nll_note', 'structural_relative'} and result['secondary']['structural_relative']['status'] == 'unavailable'
    assert {c['name'] for c in result['integrity']['checks']} >= {'ept_stage_completed', 'n2_arm_readable', 'ept_call_settings', 'n2_numeric_policy', 'ept_production_inputs', 'common_id_list_equal',
                                                                  'arms_on_the_same_rows', 'b1_probabilities_equal', 'shared_inputs_equal', 'equivalence_binds_these_commits', 'ept_code_identity',
                                                                  'ept_t009_configuration'}
    assert result['code_identity']['supervisor']['commit'] == NEW_COMMIT and result['code_identity']['ept']['commit'] == NEW_COMMIT
    assert result['provenance']['config_sha256'] == 'a' * 64 and result['provenance']['verified_files']['fit/ept/model/epoch_0014/best.pt'] == '7' * 64
    assert set(result['files_sha256']) == {'PROGRESS.jsonl', 'ept/ARM.json', 'ept/ARRAYS.npz', 'n2/ARM.json', 'n2/ARRAYS.npz'} and all(len(v) == 64 for v in result['files_sha256'].values())
    assert result['timing']['status'] == 'available' and result['numeric_policy']['torch_version_equals_training_identity'] == {'ept': True, 'n2': True}
    json.dumps(result, allow_nan=False)


N2_ROWS = {k: v for k, v in result_arm('n2', 'better')['endpoints'].items()}
VOID_FAILURES = {                                                               # a check that the primary statistic depends on fails: no decision
    'the EPT stage was refused': dict(stage_changes={'ept': {'state': 'refused', 'reason': 'a hash differs'}}, expect='ept_stage_completed'),
    'the EPT stage hit its cap': dict(stage_changes={'ept': {'state': 'cap', 'reason': 'the wall cap'}}, expect='ept_stage_completed'),
    'ARM.json of the EPT changed after the stage': dict(stage_changes={'ept': {'arm_sha256': '0' * 64}}, expect='ept_arm_readable'),
    'the equivalence proof failed': dict(equivalence_changes={'status': 'failed', 'explanation': 'three keys differ', 'differences': [{'key': 'a'}]}, expect=None),
    'the equivalence proof names another new commit': dict(equivalence_changes={'new_commit': 'c' * 40}, expect='equivalence_binds_these_commits'),
    'the equivalence proof names another old commit': dict(equivalence_changes={'old_commit': 'c' * 40}, expect='equivalence_binds_these_commits'),
    'an EPT call setting differs': dict(arm_changes={'ept': {'provenance': {**CALL_PROVENANCE, 'population': 'all'}}}, expect='ept_call_settings'),
    'the numeric policy of the EPT differs': dict(doc_changes={'ept': {'numeric': {**NUMERIC_OK, 'cudnn_allow_tf32': True}}}, expect='ept_numeric_policy'),
    'autocast was on in the EPT stage': dict(doc_changes={'ept': {'numeric': {**NUMERIC_OK, 'autocast_enabled': True}}}, expect='ept_numeric_policy'),
    'a test loader gave the stores to the EPT': dict(doc_changes={'ept': {'settings': {'batch_size': 32, 'bundle_loader': 'c_ept.tests.x:y', 'identity_gate': 'start_gate'}}}, expect='ept_production_inputs'),
    'the EPT stage ran another commit': dict(doc_changes={'ept': {'code_identity': {**CODE_IDENTITY, 'commit': 'e' * 40}}}, expect='ept_code_identity'),
    'the EPT stage read another T009 configuration': dict(doc_changes={'ept': {'t009': {**t009_summary('ept'), 'config_sha256': 'b' * 64}}}, expect='ept_t009_configuration'),
    'the stages verified different B1 files': dict(doc_changes={'n2': {'verified': {**verified_files('n2'), 'fit/b1/result.json': '0' * 64}}}, expect='shared_inputs_equal'),
    'B1 gives other probabilities in the two arms': dict(arm_changes={'n2': {'endpoints': {k: {**v, 'p_b1': v['p_b1'] + 0.01} for k, v in N2_ROWS.items()}}}, expect='b1_probabilities_equal'),
}
STANDING_FAILURES = {                                                           # only the secondary arm is wrong or missing: the verdict of the EPT stage stands, every N2 row is unavailable
    'the N2 stage failed': dict(stage_changes={'n2': {'state': 'failed', 'reason': 'a CUDA error'}}, expect='n2_stage_completed'),
    'the N2 stage hit its wall cap': dict(stage_changes={'n2': {'state': 'cap', 'reason': 'the wall cap of this stage is used up'}}, expect='n2_stage_completed'),
    'the N2 stage was refused': dict(stage_changes={'n2': {'state': 'refused', 'reason': 'E6'}}, expect='n2_stage_completed'),
    'the N2 stage was not run': dict(stage_changes={'n2': {'state': 'not_run', 'reason': 'the wall budget is used up'}}, expect='n2_stage_completed'),
    'ARM.json of the N2 changed after the stage': dict(stage_changes={'n2': {'arm_sha256': '0' * 64}}, expect='n2_arm_readable'),
    'an N2 call setting differs': dict(arm_changes={'n2': {'provenance': {**CALL_PROVENANCE, 'population': 'all'}}}, expect='n2_call_settings'),
    'the numeric policy of N2 differs': dict(doc_changes={'n2': {'numeric': {**NUMERIC_OK, 'cudnn_allow_tf32': True}}}, expect='n2_numeric_policy'),
    'the identity gate was skipped for N2': dict(doc_changes={'n2': {'settings': {'batch_size': 32, 'bundle_loader': gate0.DEFAULT_BUNDLE_LOADER, 'identity_gate': 'skipped'}}}, expect='n2_production_inputs'),
    'the N2 stage ran another commit': dict(doc_changes={'n2': {'code_identity': {**CODE_IDENTITY, 'commit': 'e' * 40}}}, expect='n2_code_identity'),
    'the N2 stage read another T009 configuration': dict(doc_changes={'n2': {'t009': {**t009_summary('n2'), 'config_sha256': 'b' * 64}}}, expect='n2_t009_configuration'),
    'the arms are on other id lists': dict(arm_changes={'n2': {'ids': np.arange(100, 131)}}, expect='common_id_list_equal'),
    'the arms are on other rows': dict(arm_changes={'n2': {'endpoints': {k: {**v, 'row': v['row'] + 1} for k, v in N2_ROWS.items()}}}, expect='arms_on_the_same_rows'),
}


@pytest.mark.parametrize('what', sorted(VOID_FAILURES))
def test_a_failed_check_that_the_primary_statistic_depends_on_makes_the_verdict_void_and_names_the_check(tmp_path, what):
    spec = dict(VOID_FAILURES[what])
    expect = spec.pop('expect')
    stages, equivalence, start = write_result_run(tmp_path, **spec)
    result = gate0.assemble(tmp_path, stages=stages, equivalence=equivalence, start=start)
    assert result['verdict']['verdict'] == 'VOID' and result['verdict']['void_reasons']
    if expect is not None:
        failed = [c for c in result['integrity']['checks'] if not c['passed']]
        assert expect in [c['name'] for c in failed] and result['integrity']['passed'] is False and {c['scope'] for c in failed if c['name'] == expect} == {'primary'}
    else:
        assert any('equivalence' in reason for reason in result['verdict']['void_reasons'])


@pytest.mark.parametrize('what', sorted(STANDING_FAILURES))
def test_a_failure_of_the_secondary_arm_alone_leaves_the_verdict_and_makes_every_n2_row_unavailable_with_the_reason(tmp_path, what):
    """Review I1, option (a): the task record says that secondary results never overturn the primary rule and that VOID is a failed integrity check of the decision, so an N2 cap hit, crash or refusal does not void
    the verdict. The check that failed is a warning (scope 'secondary'), N2 is unavailable (``n2_unavailable``) and the primary statistic is that of the EPT stage."""
    spec = dict(STANDING_FAILURES[what])
    expect = spec.pop('expect')
    stages, equivalence, start = write_result_run(tmp_path, **spec)
    result = gate0.assemble(tmp_path, stages=stages, equivalence=equivalence, start=start)
    assert result['verdict']['verdict'] == 'GO' and result['verdict']['void_reasons'] == [] and result['integrity']['passed'] is True and result['integrity']['failures'] == []
    failed = {c['name']: c for c in result['integrity']['checks'] if not c['passed']}
    assert expect in failed and failed[expect]['scope'] == 'secondary' and 'n2_unavailable' in failed and failed['n2_unavailable']['scope'] == 'secondary'
    assert set(result['integrity']['warnings']) >= {failed[expect]['detail'], failed['n2_unavailable']['detail']} and all(c['scope'] == 'primary' for c in result['integrity']['checks'] if c['passed'] and c['name'].startswith('ept_'))
    assert result['primary']['delta'] == pytest.approx(-0.21) and result['secondary']['brier']['n2_minus_b1']['H60']['status'] == 'unavailable'
    assert result['secondary']['brier']['ept_minus_n2']['H180']['status'] == 'unavailable' and result['secondary']['nll_per_head']['n2_minus_b1']['M2Rate/masked']['status'] == 'unavailable'
    assert result['secondary']['brier']['ept_minus_b1']['H120']['status'] == 'available' and 'n2' not in result['models'] and 'n2' not in result['numeric_policy']['arms']
    text = gate0.render_report_ko(result)
    assert '판정에 영향 없는 경고' in text and failed['n2_unavailable']['detail'] in text and '**GO' in text
    section = text.split('### 4.1 판정에 영향 없는 경고')[1].split('## 5.')[0]                                      # every warning is listed on its own line of that section, apart from the table of checks
    assert all(f'- {warning}' in section for warning in result['integrity']['warnings']) and len(result['integrity']['warnings']) >= 2


def test_the_n2_stage_runs_second_so_the_epts_primary_statistic_is_not_waiting_for_it():
    assert gate0.NEURAL_MODELS == ('ept', 'n2') and list(gate0.STAGE_CAPS_S) == ['ept', 'n2']


def test_a_failure_of_the_ept_stage_alone_is_void_and_the_n2_statistics_are_still_there(tmp_path):
    stages, equivalence, start = write_result_run(tmp_path, stage_changes={'ept': {'state': 'failed', 'reason': 'a CUDA error'}})
    result = gate0.assemble(tmp_path, stages=stages, equivalence=equivalence, start=start)
    assert result['verdict']['verdict'] == 'VOID' and result['primary'] is None and result['secondary']['brier']['n2_minus_b1']['H60']['status'] == 'available'
    assert result['secondary']['brier']['ept_minus_b1']['H60']['status'] == 'unavailable' and 'ept' not in result['models'] and 'n2' in result['models']


def test_an_n2_stage_that_did_not_finish_keeps_the_ept_files_and_leaves_no_n2_entry_in_the_hashes(tmp_path):
    stages, equivalence, start = write_result_run(tmp_path, stage_changes={'n2': {'state': 'failed', 'reason': 'the wall cap'}})
    result = gate0.assemble(tmp_path, stages=stages, equivalence=equivalence, start=start)
    assert result['stages']['n2']['state'] == 'failed' and set(result['files_sha256']) >= {'ept/ARM.json'} and 'n2/ARM.json' not in result['files_sha256']


def test_with_neither_arm_there_is_no_statistic_and_the_verdict_is_void(tmp_path):
    stages, equivalence, start = write_result_run(tmp_path, stage_changes={'ept': {'state': 'failed'}, 'n2': {'state': 'refused'}})
    result = gate0.assemble(tmp_path, stages=stages, equivalence=equivalence, start=start)
    assert result['primary'] is None and result['secondary'] is None and result['bootstrap'] is None and result['data'] is None and result['verdict']['verdict'] == 'VOID'
    assert 'integrity check failed' in result['verdict']['void_reasons'][0]
    text = gate0.render_report_ko(result)
    assert 'VOID' in text and '주 통계가 없다' in text


def test_the_outputs_are_strict_json_and_a_korean_report_that_names_the_verdict_and_the_caveats(tmp_path):
    stages, equivalence, start = write_result_run(tmp_path, quality=('mixed', 'worse'))
    result = gate0.assemble(tmp_path, stages=stages, equivalence=equivalence, start=start)
    hashes = gate0.write_outputs(tmp_path, result)
    assert set(hashes) == {'GATE0.json', 'GATE0_KO.md'} and all(len(v) == 64 for v in hashes.values())
    document = json.loads((tmp_path / 'GATE0.json').read_text(encoding='utf-8'), parse_constant=lambda name: pytest.fail('a non-finite constant: ' + name))
    assert document['schema'] == 'ept-gate0-v1' and document['task'] == 'EPT-T013' and document['verdict']['verdict'] == 'INCONCLUSIVE' and document['classification'] == gate0.CLASSIFICATION
    assert document['files_sha256']['GATE0_KO.md'] == hashes['GATE0_KO.md'] == run_io.sha_file(tmp_path / 'GATE0_KO.md') and hashes['GATE0.json'] == run_io.sha_file(tmp_path / 'GATE0.json')
    assert {'verdict', 'primary', 'secondary', 'integrity', 'equivalence', 'bootstrap', 'data', 'models', 'caveats', 'numeric_policy', 'provenance', 'code_identity', 'stages', 'start', 'timing',
            'files_sha256'} <= set(document)
    assert document['numeric_policy']['t009_not_recorded'] and document['numeric_policy']['required']['float32_matmul_precision'] == 'highest' and 'role_note' not in json.dumps(document)
    text = (tmp_path / 'GATE0_KO.md').read_text(encoding='utf-8')
    assert '**INCONCLUSIVE' in text and 'Δ = Brier(EPT) − Brier(B1)' in text and '## 6. 주의사항' in text and '## 7. 시간' in text and '## 9. 출처와 해시' in text
    assert all(caveat in text for caveat in gate0.CAVEATS) and 'Poisson kill law' in text and 'One seed' in text and 'GATE0.json' in text
    assert document['primary']['ci95'][0] < 0 and f'{document["primary"]["delta"]:.6f}' in text and text.split('### 3.3')[1].split('### 3.4')[0].count('| M2Rate/masked |') == 3     # one head row in each of the three contrasts
    assert '| M2Rate/masked | 0.5 |' in text and '#### EPT 머리 블록별' in text and '#### N2 머리 블록별' in text                                  # the block timings of both arms
    with pytest.raises(gate0.Gate0Refused, match='never written over'):
        gate0.write_outputs(tmp_path, result)


# ---- the stage body and the supervised entry --------------------------------------------------------------------------------------------------------------------------------
TOY_LOADER = 'c_ept.tests.run_gate0_bodies:toy_bundle'
E6_CLEAR = {'running': False, 'matched': [], 'bridges_ignored': 0, 'scanned': 3, 'cmdline_unreadable': 0, 'rule': 'a synthetic process list'}


@pytest.fixture(autouse=True)
def restore_torch_state():
    """The probe sets process-wide numeric settings (``configure_float32``); give them back after each test."""
    import torch
    state = (torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.get_num_threads())
    yield
    torch.set_float32_matmul_precision(state[0])
    torch.use_deterministic_algorithms(state[1])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = state[2:6]
    torch.set_num_threads(state[6])


def head_commit():
    import subprocess
    return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()


def toy_bundle_of(pieces):
    from c_ept.tests import run_gate0_bodies
    return run_gate0_bodies.toy_bundle({'toy': {'root': str(pieces.world_root)}}, lambda: None)


def test_one_stage_writes_the_arm_of_its_model_on_the_toy_world_and_leaves_the_progress_lines(tmp_path, pieces):
    from c_ept.ept import data, loss
    run = SyntheticT009(tmp_path / 't009', pieces)
    out = tmp_path / 'out'
    out.mkdir()
    gate0.stage_main(out, 'ept', run.run, 'cpu', lambda meta, check: toy_bundle_of(pieces), 'skipped')
    doc = json.loads((out / 'ept' / 'ARM.json').read_text(encoding='utf-8'))
    assert doc['schema'] == gate0.ARM_SCHEMA and doc['model'] == 'ept' and doc['settings'] == {'batch_size': 32, 'device': 'cpu', 'bundle_loader': 'a function', 'identity_gate': 'skipped', 'role': 'STOP',
                                                                                                 'exposure': 'nominal', 'population': 'primary', 'monte_carlo': False}
    assert doc['numeric']['float32_matmul_precision'] == 'highest' and gate0.numeric_failures(doc['numeric']) == [] and doc['code_identity']['commit'] == head_commit()
    assert {'c_ept/run/gate0.py', 'c_ept/evaluation/adapters.py', 'c_ept/ept/heads.py', 'c_ept/models/baselines_b1.py', 'c_ept/comparison/runtime.py', 'c_ept/evaluation/endpoints.py'} <= set(doc['code_identity']['files'])
    assert len(doc['code_identity_at_start_sha256']) == 64 and set(gate0.STAGE_ENTRY_MODULES) <= {name for name in __import__('sys').modules}      # the closure holds the late imports too
    assert doc['t009']['models']['ept']['best_epoch'] == 13 and 'fit/ept/model/epoch_0014/best.pt' in doc['verified']
    assert 'role_note' not in doc['provenance'] and doc['provenance']['role'] == 'STOP' and doc['provenance']['monte_carlo'] is False
    arm = gate0.load_arm(out / 'ept', 'ept', doc['manifest'], doc['provenance'])
    assert arm['ids'].tolist() == gate0.stop_match_ids(toy_bundle_of(pieces).s1).tolist() and all(c['passed'] for c in gate0.check_arm(arm))
    assert len(arm['nll']) == 13 and set(arm['endpoints']) == set(gate0.M2_ENDPOINTS)
    lines = [json.loads(text) for text in (out / gate0.PROGRESS_FILE).read_text(encoding='utf-8').splitlines()]
    names = [loss.block_key(head, kind) for head, kind in data.ROW_FIELDS]
    assert [line['detail']['block'] for line in lines if line['stage'] == 'block' and line['event'] == 'end'] == names
    assert [(line['event'], line['stage']) for line in lines if line['stage'] in ('arm', 'load', 'inference')] == [('begin', 'arm'), ('begin', 'load'), ('end', 'load'), ('begin', 'inference'),
                                                                                                                  ('end', 'inference'), ('end', 'arm')]
    with pytest.raises(gate0.Gate0Refused, match='never run twice'):                                          # a stage is not run twice into one folder
        gate0.stage_main(out, 'ept', run.run, 'cpu', lambda meta, check: toy_bundle_of(pieces), 'skipped')


@pytest.mark.parametrize('arguments, message', [(('b1', 'cpu', 'skipped'), 'stage model'), (('ept', 'tpu', 'skipped'), 'device'), (('ept', 'cpu', 'maybe'), 'identity_gate')])
def test_a_stage_refuses_an_unknown_model_device_or_identity_gate_before_reading_anything(tmp_path, arguments, message):
    with pytest.raises(gate0.Gate0Refused, match=message):
        gate0.stage_main(tmp_path, arguments[0], tmp_path / 'nothing', arguments[1], 'c_ept.tests.run_gate0_bodies:toy_bundle', arguments[2])
    assert not (tmp_path / 'ept').exists()


def test_a_stage_that_is_refused_by_the_loader_leaves_the_failed_stage_lines_and_no_arm(tmp_path, pieces):
    run = SyntheticT009(tmp_path / 't009', pieces)
    (run.run / 'fit/n2/stage.json').unlink()
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(gate0.Gate0Refused, match='stage.json'):
        gate0.stage_main(out, 'n2', run.run, 'cpu', lambda meta, check: toy_bundle_of(pieces), 'skipped')
    assert not (out / 'n2' / 'ARM.json').exists()
    ends = [json.loads(text) for text in (out / gate0.PROGRESS_FILE).read_text(encoding='utf-8').splitlines() if '"end"' in text]
    assert [(line['stage'], line['status'], line['error']) for line in ends] == [('load', 'failed', 'Gate0Refused'), ('arm', 'failed', 'Gate0Refused')]


def test_the_match_keys_of_the_s1_store_are_read_for_the_scrubbing(tmp_path, pieces):
    run = SyntheticT009(tmp_path / 't009', pieces)
    keys = gate0.s1_match_keys(run.run)
    assert len(keys) == 8 and all(isinstance(key, str) and key for key in keys)
    (run.run / 'prepare' / 'inputs.json').unlink()
    with pytest.raises(gate0.Gate0Refused):
        gate0.s1_match_keys(run.run)


def supervised(tmp_path, pieces, *, run=None, **over):
    """``supervise`` on the synthetic T009 run (``run``, or a new one) and the toy world, on the CPU, with the seams of the tests (E6 answered, memory sampled by a constant, the toy loader, the identity
    gate skipped)."""
    import io as stdio
    run = run if run is not None else SyntheticT009(tmp_path / 't009', pieces)
    equivalence = write_equivalence(tmp_path / 'equivalence.json', old_commit=OLD_COMMIT_T009, new_commit=head_commit())
    runs = tmp_path / 'runs'
    runs.mkdir()
    stream = stdio.StringIO()
    keys = gate0.s1_match_keys(run.run)
    options = dict(equivalence=equivalence, approval='author approval of 2026-10-26 (synthetic test)', device='cpu', runs_root=runs, e6=lambda: dict(E6_CLEAR), measure=lambda: 10 ** 9, cuda_visible_devices='',
                   threads=1, bundle_loader=TOY_LOADER, identity_gate='skipped', scrub_keys=keys, stream=stream, poll_s=0.2)
    options.update(over)
    code = gate0.supervise(run.run, runs / '20261026_c_ept_gate0_1', **options)
    return code, runs / '20261026_c_ept_gate0_1', stream.getvalue(), keys


def test_a_supervised_run_on_the_toy_world_writes_every_output_and_has_no_decision_without_the_production_inputs(tmp_path, pieces):
    code, out, printed, keys = supervised(tmp_path, pieces)
    assert code == 3 and 'Gate 0 verdict: VOID' in printed                                                    # the toy loader and the skipped identity gate: outputs, no decision
    assert {p.name for p in out.iterdir()} == {'START.json', 'EQUIVALENCE.json', 'PROGRESS.jsonl', 'ept', 'n2', 'logs', 'GATE0.json', 'GATE0_KO.md'}
    document = json.loads((out / 'GATE0.json').read_text(encoding='utf-8'))
    failed = {c['name'] for c in document['integrity']['checks'] if not c['passed']}
    assert failed == {'ept_production_inputs', 'n2_production_inputs', 'n2_unavailable'}                      # everything else passed: the stage receipts, the arms, the numeric policy, the commits; N2 is then dropped
    assert document['integrity']['passed'] is False and len(document['integrity']['failures']) == 1 and len(document['integrity']['warnings']) == 2     # the EPT's check voids, the two N2 ones are warnings
    assert document['verdict']['verdict'] == 'VOID' and document['primary'] is not None and document['primary']['n_matches'] == 3 and document['bootstrap']['B'] == 2000
    assert [document['stages'][m]['state'] for m in ('ept', 'n2')] == ['completed', 'completed'] and document['start']['approval'].startswith('author approval')
    start = json.loads((out / 'START.json').read_text(encoding='utf-8'))
    assert start['schema'] == 'ept-gate0-start-v1' and start['wall_s'] == 21600 and start['stage_caps_s'] == {'ept': 10800, 'n2': 10800} and start['e6'] == E6_CLEAR
    assert start['child_settings']['cublas_workspace_config'] == ':4096:8' and start['code_identity']['commit'] == head_commit() and (out / 'EQUIVALENCE.json').read_bytes() == (tmp_path / 'equivalence.json').read_bytes()
    for model in ('ept', 'n2'):
        stage = json.loads((out / model / 'STAGE.json').read_text(encoding='utf-8'))
        assert stage['state'] == 'completed' and stage['exit_code'] == 0 and stage['arm_sha256'] == run_io.sha_file(out / model / 'ARM.json') and 0 < stage['elapsed_s'] <= stage['cap_s']
        assert not any(key in (out / 'logs' / f'{model}.log').read_text(encoding='utf-8') for key in keys)      # the end of the child's output holds no match key
    timing = document['timing']
    assert timing['status'] == 'available' and all(timing[m]['child_s'] and timing[m]['inference_s'] and len(timing[m]['blocks']) == 13 for m in ('ept', 'n2')) and timing['total_s'] > 0
    text = (out / 'GATE0_KO.md').read_text(encoding='utf-8')
    assert 'VOID' in text and '## 1. 판정' in text and all(key not in text for key in keys)
    ids_ept = np.load(out / 'ept' / 'ARRAYS.npz')['ids']
    assert ids_ept.tolist() == np.load(out / 'n2' / 'ARRAYS.npz')['ids'].tolist()
    with pytest.raises(gate0.Gate0Refused, match='exists'):                                                    # a run is never written into an existing folder
        gate0.supervise(tmp_path / 't009' / RUN_NAME, out, equivalence=tmp_path / 'equivalence.json', approval='x', device='cpu', runs_root=tmp_path / 'runs', e6=lambda: dict(E6_CLEAR),
                        measure=lambda: 10 ** 9, cuda_visible_devices='', threads=1, identity_gate='skipped')


def test_a_stage_that_hits_its_cap_is_recorded_and_the_other_stage_still_runs_so_its_timing_is_not_hidden(tmp_path, pieces):
    code, out, printed, _ = supervised(tmp_path, pieces, stage_caps_s={'ept': 0.5, 'n2': 600})
    document = json.loads((out / 'GATE0.json').read_text(encoding='utf-8'))
    ept, n2 = document['stages']['ept'], document['stages']['n2']
    assert ept['state'] == 'cap' and 'wall cap' in ept['reason'] and ept['exit_code'] is None and n2['state'] == 'completed'
    assert code == 3 and document['verdict']['verdict'] == 'VOID' and any('ept stage did not finish' in r for r in document['verdict']['void_reasons'])
    lines = [json.loads(text) for text in (out / 'PROGRESS.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [(line['model'], line['status'], line['error']) for line in lines if line['stage'] == 'child' and line['event'] == 'end'] == [('ept', 'failed', 'BudgetStop'), ('n2', 'ok', None)]
    assert len(document['timing']['n2']['blocks']) == 13 and document['timing']['ept']['blocks'] == {} and document['primary'] is None       # the EPT arm is missing: no primary statistic
    assert document['secondary'] is None and any(not c['passed'] and c['name'] == 'n2_production_inputs' for c in document['integrity']['checks'])      # the toy loader is no production input: N2 is dropped too, so no arm is left


def test_the_n2_stage_gets_what_the_epts_leaves_of_the_wall_budget_minus_the_reserve_not_a_fixed_three_hours(tmp_path, pieces):
    """Review m1: the share of a stage is ``cap = min(cap_s, wall - elapsed - 15 min)``, so the EPT stage runs first with at most its 3 h and N2 gets what is left (less than 3 h when the EPT stage took long); the
    receipts record the caps that were given."""
    code, out, printed, _ = supervised(tmp_path, pieces, wall_s=1000.0)                                          # the caps stay at the default 3 h: the wall budget is the smaller one
    ept, n2 = (json.loads((out / model / 'STAGE.json').read_text(encoding='utf-8')) for model in ('ept', 'n2'))
    assert ept['state'] == 'completed' and n2['state'] == 'completed' and 0 < ept['cap_s'] <= 1000 - gate0.FINALISATION_S
    assert 0 < n2['cap_s'] <= ept['cap_s'] - ept['elapsed_s'] and gate0.STAGE_CAPS_S == {'ept': 10800, 'n2': 10800}


def test_a_stage_that_is_refused_stops_the_run_and_the_other_stage_is_not_run(tmp_path, pieces):
    run = SyntheticT009(tmp_path / 't009', pieces)
    (run.run / 'fit' / 'ept' / 'stage.json').unlink()                                                             # the EPT stage cannot read its receipt: its child exits with 2
    code, out, printed, _ = supervised(tmp_path, pieces, run=run)
    document = json.loads((out / 'GATE0.json').read_text(encoding='utf-8'))
    assert code == 3 and document['stages']['ept']['state'] == 'refused' and document['stages']['ept']['exit_code'] == 2 and document['stages']['n2']['state'] == 'not_run'
    assert 'earlier stage was refused' in document['stages']['n2']['reason'] and not (out / 'n2' / 'ARM.json').exists()
    assert 'stage.json' in (out / 'logs' / 'ept.log').read_text(encoding='utf-8') and document['verdict']['verdict'] == 'VOID' and 'Gate 0 verdict: VOID' in printed


@pytest.mark.parametrize('which, error', [('e6', 'GateRefused'), ('identity', 'IdentityChanged')])
def test_a_gate_that_refuses_between_the_stages_ends_the_run_with_its_outputs_and_names_the_gate(tmp_path, pieces, which, error):
    from c_ept.run import guards as guards_module
    from c_ept.run import identity as identity_module
    calls = []

    def e6():
        calls.append(1)
        if which == 'e6' and len(calls) == 2:                                                                       # the first call is the start gate, the second is the one before the EPT stage
            raise guards_module.GateRefused('e6', 'a model server is running (a synthetic one)')
        return dict(E6_CLEAR)

    def verify(identity, repo=None):
        if which == 'identity':
            raise identity_module.IdentityChanged('imported code changed since the identity was bound', ['c_ept/run/gate0.py'])
    code, out, printed, _ = supervised(tmp_path, pieces, e6=e6, verify=verify)
    document = json.loads((out / 'GATE0.json').read_text(encoding='utf-8'))
    ept = document['stages']['ept']
    assert code == 3 and ept['state'] == 'refused' and ept['reason'].startswith(error) and document['stages']['n2']['state'] == 'not_run' and document['verdict']['verdict'] == 'VOID'
    assert (out / 'GATE0_KO.md').is_file() and not (out / 'ept' / 'ARM.json').exists()


@pytest.mark.parametrize('what, over, message', [
    ('an empty approval', {'approval': '  '}, 'approval'),
    ('another device', {'device': 'tpu'}, 'device'),
    ('another identity gate', {'identity_gate': 'maybe'}, 'identity gate'),
    ('a cap that is not a number', {'stage_caps_s': {'ept': 0, 'n2': 5}}, 'caps'),
    ('a cap for one model only', {'stage_caps_s': {'ept': 5}}, 'caps')])
def test_the_supervisor_refuses_what_it_cannot_start_with_before_it_writes_anything(tmp_path, pieces, what, over, message):
    with pytest.raises(gate0.Gate0Refused, match=message):
        supervised(tmp_path, pieces, **over)
    assert not (tmp_path / 'runs' / '20261026_c_ept_gate0_1').exists()


def test_the_output_must_be_a_new_gate0_folder_of_the_runs_root_apart_from_the_t009_run(tmp_path, pieces):
    run = SyntheticT009(tmp_path / 't009', pieces)
    equivalence = write_equivalence(tmp_path / 'equivalence.json', old_commit=OLD_COMMIT_T009, new_commit=head_commit())
    runs = tmp_path / 'runs'
    runs.mkdir()
    common = dict(equivalence=equivalence, approval='x', device='cpu', runs_root=runs, e6=lambda: dict(E6_CLEAR), measure=lambda: 10 ** 9, cuda_visible_devices='', threads=1, identity_gate='skipped')
    for name in ('gate0', '20261026_c_ept_matched25_2', '2026_c_ept_gate0_1', '20261026_c_ept_gate0_x'):
        with pytest.raises(gate0.Gate0Refused, match='new folder'):
            gate0.supervise(run.run, runs / name, **common)
    with pytest.raises(gate0.Gate0Refused, match='new folder'):
        gate0.supervise(run.run, tmp_path / '20261026_c_ept_gate0_1', **common)                                  # not a folder of the runs root
    (runs / '20261026_c_ept_gate0_1').mkdir()
    with pytest.raises(gate0.Gate0Refused, match='exists'):
        gate0.supervise(run.run, runs / '20261026_c_ept_gate0_1', **common)
    with pytest.raises(gate0.Gate0Refused):                                                                    # no RUN.json in the T009 folder
        gate0.supervise(tmp_path / 'runs', runs / '20261026_c_ept_gate0_2', **common)
    equivalence_bad = tmp_path / 'bad.json'
    equivalence_bad.write_text('{}', encoding='utf-8')
    with pytest.raises(gate0.Gate0Refused, match='schema'):
        gate0.supervise(run.run, runs / '20261026_c_ept_gate0_3', **{**common, 'equivalence': equivalence_bad})
    assert sorted(p.name for p in runs.iterdir()) == ['20261026_c_ept_gate0_1']                                 # nothing was made by the refusals


def test_a_model_server_that_runs_refuses_the_start_and_nothing_is_written(tmp_path, pieces):
    def running():
        raise guards_module.GateRefused('e6', 'a Strata / llama / ollama model server is running')
    from c_ept.run import guards as guards_module
    with pytest.raises(guards_module.GateRefused, match='model server'):
        supervised(tmp_path, pieces, e6=running)
    assert not (tmp_path / 'runs' / '20261026_c_ept_gate0_1').exists()


def test_the_command_line_needs_its_four_arguments_hands_them_to_the_supervisor_and_scrubs_with_the_keys(monkeypatch, tmp_path, capsys):
    seen = {}
    monkeypatch.setattr(gate0, 's1_match_keys', lambda run: seen.setdefault('keys', ['KR_123']) and ['KR_123'])
    monkeypatch.setattr(gate0, 'supervise', lambda *a, **k: seen.update(args=a, kwargs=k) or 0)
    command = ['--t009', 'T', '--out', 'O', '--equivalence', 'E', '--approval', 'the approval text']
    assert gate0.main(command) == 0
    assert seen['args'] == ('T', 'O') and seen['kwargs'] == {'equivalence': 'E', 'approval': 'the approval text', 'scrub_keys': ['KR_123']}
    for missing in range(4):
        with pytest.raises(SystemExit) as stop:
            gate0.main([x for i, x in enumerate(command) if i // 2 != missing])
        assert stop.value.code == 2
    monkeypatch.setattr(gate0, 'supervise', lambda *a, **k: (_ for _ in ()).throw(gate0.Gate0Refused('bad KR_123 here')))
    assert gate0.main(command) == 2 and 'bad <key> here' in capsys.readouterr().err                              # a refusal is exit code 2 and the key is scrubbed


# ---- review I2: the binding of the record to the commits and the checks of the T009 record are made before anything is created ---------------------------------------------------------
def refused_before_anything(tmp_path, pieces, message, *, edit_run=None, equivalence_changes=None, **over):
    """``supervise`` on a synthetic T009 run that ``edit_run`` damages, with an equivalence record that ``equivalence_changes`` alters: a ``Gate0Refused`` (exit code 2 under ``main``) and not one file made."""
    run = SyntheticT009(tmp_path / 't009', pieces)
    if edit_run is not None:
        edit_run(run)
    equivalence = write_equivalence(tmp_path / 'equivalence.json', **{'old_commit': OLD_COMMIT_T009, 'new_commit': head_commit(), **(equivalence_changes or {})})
    runs = tmp_path / 'runs'
    runs.mkdir()
    options = dict(equivalence=equivalence, approval='x', device='cpu', runs_root=runs, e6=lambda: dict(E6_CLEAR), measure=lambda: 10 ** 9, cuda_visible_devices='', threads=1, identity_gate='skipped')
    options.update(over)
    with pytest.raises(gate0.Gate0Refused, match=message):
        gate0.supervise(run.run, runs / '20261026_c_ept_gate0_1', **options)
    assert list(runs.iterdir()) == []                                                                              # no output folder, no lock, nothing


def _rewrite_run_record(run, **changes):
    record = json.loads((run.run / 'RUN.json').read_text(encoding='utf-8'))
    record.update(changes)
    (run.run / 'RUN.json').write_text(json.dumps(record), encoding='utf-8')


def _set_config(run, **changes):
    run.config.update(changes)
    run.write_run_record()


def _damage_legacy_pins(run):
    run.config['inputs']['legacy']['files'] = {**run.config['inputs']['legacy']['files'], 'gameplay/__init__.py': 'f' * 64}
    run.write_run_record()


def _use_the_v1_configuration(run):
    run.config = v1_of(run.config)
    run.write_run_record()


EARLY_REFUSALS = {
    'a record made for another commit than this code': (None, {'new_commit': 'c' * 40}, 'equivalence record was made for'),
    'a record that compares another T009 source commit': (None, {'old_commit': 'c' * 40}, 'T009 source commit'),
    'a T009 configuration without a source commit': (lambda run: _set_config(run, source_commit=None), None, 'T009 source commit'),
    'a RUN.json of another schema': (lambda run: _rewrite_run_record(run, schema='ept-comparison-run-v2'), None, 'not a comparison run record'),
    'a RUN.json whose digest is not that of its configuration': (lambda run: _rewrite_run_record(run, config_sha256='0' * 64), None, 'digest'),
    'a T009 configuration of the v1 format': (_use_the_v1_configuration, None, 'profile mismatch'),
    'a T009 configuration with another fixed setting': (lambda run: _set_config(run, settings={**run.config['settings'], 'batch_size': 64}), None, 'fixed matched25 settings changed'),
    'legacy pins that are not those of this code': (_damage_legacy_pins, None, 'legacy pins'),
    'a run folder that is not the output of its configuration': (lambda run: _set_config(run, output=str(run.root / 'elsewhere')), None, 'not the output'),
}


@pytest.mark.parametrize('what', sorted(EARLY_REFUSALS))
def test_the_supervisor_refuses_before_it_makes_the_output_folder_what_the_gpu_stages_could_only_find_out_late(tmp_path, pieces, what):
    edit, changes, message = EARLY_REFUSALS[what]
    refused_before_anything(tmp_path, pieces, message, edit_run=edit, equivalence_changes=changes)


def test_a_record_for_this_commit_and_this_t009_run_starts_without_any_of_those_refusals(tmp_path, pieces):
    code, out, printed, _ = supervised(tmp_path, pieces, stage_caps_s={'ept': 0.5, 'n2': 0.5})                    # both stages are cut at once: only the start is of interest here
    assert out.is_dir() and (out / 'START.json').is_file() and code == 3
    start = json.loads((out / 'START.json').read_text(encoding='utf-8'))
    assert start['code_identity']['commit'] == head_commit() and json.loads((out / 'EQUIVALENCE.json').read_text(encoding='utf-8'))['new_commit'] == head_commit()


# ---- review minors: the free-GPU check, the copy of the record, the late imports, the amp and compile record, the limits of the proof -----------------------------------------------------
SMI_GPU = 'NVIDIA GeForce RTX 5080, 5266, 16303, 22'                                                               # name, memory.used, memory.total, utilization.gpu (the read of the lead's desktop)
SMI_PLAIN = ('| Processes:                                                                              |\n'
             '|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n'
             '|    0   N/A  N/A            4120    C+G   ...ndows\\System32\\dwm.exe                      N/A |\n'
             '|    0   N/A  N/A            9876    C+G   ...aming\\Wallpaper\\wallpaper64.exe             N/A |\n'
             '|    0   N/A  N/A           11200    C+G   C:\\Program Files\\Google\\Chrome\\chrome.exe        N/A |\n')


def fake_smi(*, gpu=SMI_GPU, apps='', plain='', code=0, listing_code=0):
    """A stand-in of ``subprocess.run`` that answers the three nvidia-smi calls of ``gate0.gpu_state``: the memory query (``code``), the compute-apps query and the plain table (``listing_code``)."""
    def run(command, **kwargs):
        if any(part.startswith('--query-compute-apps') for part in command):
            return SimpleNamespace(returncode=listing_code, stdout=apps, stderr='')
        if any(part.startswith('--query-gpu') for part in command):
            return SimpleNamespace(returncode=code, stdout=gpu, stderr='')
        return SimpleNamespace(returncode=listing_code, stdout=plain, stderr='')
    return run


def test_the_gpu_state_is_read_from_nvidia_smi_with_the_free_memory_the_utilisation_and_the_visible_process_names():
    state = gate0.gpu_state(run=fake_smi(apps='4321, C:\\Python\\python.exe, [N/A]\n', plain=SMI_PLAIN), find=lambda: 'nvidia-smi')
    assert state == {'name': 'NVIDIA GeForce RTX 5080', 'memory_total_mib': 16303, 'memory_used_mib': 5266, 'memory_free_mib': 16303 - 5266, 'utilization_percent': 22,
                     'process_names': ['chrome.exe', 'dwm.exe', 'python.exe', 'wallpaper64.exe']}
    assert gate0.GPU_MIN_FREE_MIB == 8192 and gate0.check_gpu_free(state) is state                                  # 11,037 MiB free: the desktop with its apps and a utilisation of 22% is not refused
    json.dumps(state, allow_nan=False)


def test_a_utilisation_or_a_process_list_that_nvidia_smi_does_not_give_is_none_and_empty_and_never_a_refusal():
    state = gate0.gpu_state(run=fake_smi(gpu='NVIDIA GeForce RTX 5080, 100, 16303, [N/A]', listing_code=1), find=lambda: 'nvidia-smi')
    assert state['utilization_percent'] is None and state['process_names'] == [] and gate0.check_gpu_free(state) is state


def test_the_free_memory_is_the_one_condition_it_passes_at_exactly_8192_mib_free_and_is_refused_below():
    def state(free):
        return gate0.gpu_state(run=fake_smi(gpu=f'a card, {16303 - free}, 16303, 90', plain=SMI_PLAIN), find=lambda: 'nvidia-smi')
    exact = state(8192)
    assert exact['memory_free_mib'] == 8192 and gate0.check_gpu_free(exact) is exact                                 # at the limit; a busy GPU (90%) with processes is no reason
    with pytest.raises(gate0.Gate0Refused, match='not enough free GPU memory') as refusal:
        gate0.check_gpu_free(state(8191))
    assert '8191' in str(refusal.value) and '8192' in str(refusal.value)


@pytest.mark.parametrize('what, smi', [
    ('a failing nvidia-smi', dict(code=9)),
    ('an answer that is not a table', dict(gpu='no devices were found')),
    ('a memory that is not a number', dict(gpu='a card, [N/A], 16303, 5'))])
def test_a_gpu_that_nvidia_smi_cannot_describe_is_refused(what, smi):
    with pytest.raises(gate0.Gate0Refused, match='nvidia-smi'):
        gate0.gpu_state(run=fake_smi(**smi), find=lambda: 'nvidia-smi')


def test_no_nvidia_smi_means_the_gpu_cannot_be_confirmed_free():
    with pytest.raises(gate0.Gate0Refused, match='nvidia-smi'):
        gate0.gpu_state(run=fake_smi(), find=lambda: None)


def gpu_record(free, *, used=5266, util=22, names=('chrome.exe',)):
    return {'name': 'a card', 'memory_total_mib': free + used, 'memory_used_mib': used, 'memory_free_mib': free, 'utilization_percent': util, 'process_names': list(names)}


def test_the_supervisor_checks_the_gpu_before_it_makes_anything_on_the_gpu_and_never_on_the_cpu(tmp_path, pieces):
    refused_before_anything(tmp_path, pieces, 'not enough free GPU memory', device='cuda', gpu_probe=lambda: gpu_record(8191), cuda_visible_devices='0')
    asked = []
    refused_before_anything(tmp_path / 'again', pieces, 'approval', device='cpu', gpu_probe=lambda: asked.append(1), approval=' ')             # on the CPU the probe is not called
    assert asked == []


def test_the_state_of_the_gpu_is_recorded_in_the_start_record_when_it_was_free(tmp_path, pieces):
    free = gpu_record(11037)
    run = SyntheticT009(tmp_path / 't009', pieces)
    equivalence = write_equivalence(tmp_path / 'equivalence.json', old_commit=OLD_COMMIT_T009, new_commit=head_commit())
    runs = tmp_path / 'runs'
    runs.mkdir()
    code = gate0.supervise(run.run, runs / '20261026_c_ept_gate0_1', equivalence=equivalence, approval='x', device='cuda', runs_root=runs, e6=lambda: dict(E6_CLEAR), measure=lambda: 10 ** 9,
                           cuda_visible_devices='', threads=1, identity_gate='skipped', gpu_probe=lambda: free, stage_caps_s={'ept': 0.5, 'n2': 0.5}, bundle_loader=TOY_LOADER, poll_s=0.2,
                           stream=__import__('io').StringIO())                                                       # device cuda with no visible GPU: both children are cut or refused, only the start record is of interest
    assert code == 3
    start = json.loads((runs / '20261026_c_ept_gate0_1' / 'START.json').read_text(encoding='utf-8'))
    assert start['gpu'] == free and json.loads((runs / '20261026_c_ept_gate0_1' / 'GATE0.json').read_text(encoding='utf-8'))['start']['gpu'] == free


def test_the_equivalence_copy_is_made_from_the_bytes_that_were_read_not_from_the_file_as_it_is_later(tmp_path, pieces, monkeypatch):
    run = SyntheticT009(tmp_path / 't009', pieces)
    path = write_equivalence(tmp_path / 'equivalence.json', old_commit=OLD_COMMIT_T009, new_commit=head_commit())
    original = path.read_bytes()
    real = gate0.read_equivalence_bytes

    def read_then_change(given):
        record, raw = real(given)
        Path(given).write_text(json.dumps({'schema': 'something else'}), encoding='utf-8')                         # the source changes after it was read
        return record, raw
    monkeypatch.setattr(gate0, 'read_equivalence_bytes', read_then_change)
    runs = tmp_path / 'runs'
    runs.mkdir()
    gate0.supervise(run.run, runs / '20261026_c_ept_gate0_1', equivalence=path, approval='x', device='cpu', runs_root=runs, e6=lambda: dict(E6_CLEAR), measure=lambda: 10 ** 9, cuda_visible_devices='',
                    threads=1, identity_gate='skipped', stage_caps_s={'ept': 0.5, 'n2': 0.5}, bundle_loader=TOY_LOADER, poll_s=0.2, stream=__import__('io').StringIO())
    out = runs / '20261026_c_ept_gate0_1'
    start = json.loads((out / 'START.json').read_text(encoding='utf-8'))
    assert (out / 'EQUIVALENCE.json').read_bytes() == original and run_io.sha_file(out / 'EQUIVALENCE.json') == start['equivalence_sha256']


def test_read_equivalence_bytes_gives_the_record_and_the_bytes_that_the_hash_names(tmp_path):
    path = write_equivalence(tmp_path / 'a.json')
    record, raw = gate0.read_equivalence_bytes(path)
    assert raw == path.read_bytes() and record['file_sha256'] == hashlib.sha256(raw).hexdigest() == gate0.read_equivalence(path)['file_sha256']
    with pytest.raises(gate0.Gate0Refused):
        gate0.read_equivalence_bytes(tmp_path / 'missing.json')


def test_the_start_gate_covers_the_late_imported_files_of_the_final_identity_too(tmp_path, pieces, monkeypatch):
    run = SyntheticT009(tmp_path / 't009', pieces)
    out = tmp_path / 'out'
    out.mkdir()
    seen = []
    monkeypatch.setattr(gate0.run_identity, 'start_gate', lambda repo, identity: seen.append(set(identity.files)))
    gate0.stage_main(out, 'ept', run.run, 'cpu', lambda meta, check: toy_bundle_of(pieces), 'start_gate')
    assert len(seen) == 2 and seen[0] <= seen[1]                                                                    # the identity of the start, then the identity of the end (a superset)
    refused = tmp_path / 'out2'
    refused.mkdir()
    calls = []

    def second_call_refuses(repo, identity):
        calls.append(1)
        if len(calls) == 2:
            raise gate0.run_identity.IdentityChanged('imported files are modified, untracked or ignored in the working tree', ['c_ept/late_import.py'])
    monkeypatch.setattr(gate0.run_identity, 'start_gate', second_call_refuses)
    with pytest.raises(gate0.run_identity.IdentityChanged):
        gate0.stage_main(refused, 'n2', run.run, 'cpu', lambda meta, check: toy_bundle_of(pieces), 'start_gate')
    assert not (refused / 'n2' / 'ARM.json').exists() and not (refused / 'n2' / 'ARRAYS.npz').exists()               # nothing is left that could be read as an arm


def test_the_record_of_the_numeric_state_measures_autocast_and_names_amp_and_compile_as_properties_of_the_code():
    import torch
    record = gate0.apply_numeric_policy()                                                                          # the T009 settings are on; the autouse fixture gives the process state back afterwards
    assert record['autocast_enabled'] is False and record['amp_used_by_this_code'] is False and record['compile_used_by_this_code'] is False and 'amp' not in record and 'compile' not in record
    assert gate0.REQUIRED_NUMERIC['autocast_enabled'] is False and gate0.numeric_failures(record) == []
    with torch.autocast('cpu'):
        inside = gate0.numeric_record()
    assert inside['autocast_enabled'] is True and gate0.numeric_failures(inside) == ['autocast_enabled=True']


def test_the_report_states_the_limits_of_the_equivalence_proof_in_section_5(tmp_path):
    stages, equivalence, start = write_result_run(tmp_path)
    result = gate0.assemble(tmp_path, stages=stages, equivalence=equivalence, start=start)
    assert result['equivalence']['limits'] == list(gate0.EQUIVALENCE_LIMITS) and len(gate0.EQUIVALENCE_LIMITS) >= 5
    section = gate0.render_report_ko(result).split('## 5.')[1].split('## 6.')[0]
    assert all(limit in section for limit in gate0.EQUIVALENCE_LIMITS) and any('CPU' in limit for limit in gate0.EQUIVALENCE_LIMITS) and any('M4' in limit for limit in gate0.EQUIVALENCE_LIMITS)


def _section_7(tmp_path, name, gpu):
    stages, equivalence, start = write_result_run(tmp_path / name, start_changes={'gpu': gpu})
    result = gate0.assemble(tmp_path / name, stages=stages, equivalence=equivalence, start=start)
    assert result['start']['gpu'] == gpu
    return gate0.render_report_ko(result).split('## 7.')[1].split('## 8.')[0]


def test_the_report_gives_the_gpu_at_the_start_and_says_when_the_timings_are_not_exclusive(tmp_path):
    """Lead ruling: other GPU users do not change the numbers under the deterministic settings, only the time and the memory headroom; so the timing section must say that the seconds are not those of an exclusive
    GPU when the utilisation at the start was above 0, give the memory and the visible process names, and say that it is unknown when the utilisation was not read."""
    busy = _section_7(tmp_path, 'busy', gpu_record(11037, util=22, names=('chrome.exe', 'dwm.exe')))
    assert 'GPU를 독점한 값이 아니다' in busy and '22%' in busy and '11,037' in busy and '5,266' in busy and '16,303' in busy and 'chrome.exe' in busy and 'dwm.exe' in busy
    idle = _section_7(tmp_path, 'idle', gpu_record(11037, util=0, names=()))
    assert '0%' in idle and 'GPU를 독점한 값이 아니다' not in idle and '알 수 없다' not in idle
    unknown = _section_7(tmp_path, 'unknown', gpu_record(11037, util=None))
    assert '알 수 없다' in unknown and 'GPU를 독점한 값이 아니다' not in unknown
    stages, equivalence, start = write_result_run(tmp_path / 'cpu')
    section = gate0.render_report_ko(gate0.assemble(tmp_path / 'cpu', stages=stages, equivalence=equivalence, start=start)).split('## 7.')[1].split('## 8.')[0]
    assert 'GPU 상태' not in section                                                                                 # a run without a GPU record (the CPU) has no such line