"""Audit follow-up tests (synthetic only; no run data): bootstrap acceptance gate, fit key/overwrite guards, report wording."""
import copy
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import iq20260915_common as Q  # noqa: E402
import iq20260915_fit as FIT  # noqa: E402
import iq20260915_postrun_checks as PR  # noqa: E402

MODELS = PR.BOOT_MODELS


def _cell(n_matches=40, seed=0, single_class=False):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.array([f'KR_{i}' for i in range(n_matches)]), 2)
    y = np.ones(len(g), dtype=int) if single_class else (rng.random(len(g)) < 0.5).astype(int)
    preds = {m: rng.random(len(g)) for m in MODELS}
    return y, g, preds


@pytest.fixture(scope='module')
def valid_record():
    y, g, preds = _cell()
    return y, g, Q.paired_bootstrap(y, preds, g, Q.CONTRASTS)


def test_valid_eligible_bootstrap_record_is_accepted(valid_record):
    y, g, bt = valid_record
    problems, eligible = PR.bootstrap_cell_acceptance(bt, y, g)
    assert eligible and problems == []


@pytest.mark.parametrize('mutate', [
    lambda b: b.update(replicates=999),
    lambda b: b.update(seed=1),
    lambda b: b['pairs'].pop(),
    lambda b: b['pairs'].reverse(),
    lambda b: b['pairs'][0]['a_minus_b']['brier'].update(ci95=None),
    lambda b: b['pairs'][1]['a_minus_b']['logloss'].update(ci95=[0.1, -0.1]),
    lambda b: b['model_ci']['pt_winner']['auc'].update(finite_replicates=998),
    lambda b: b.update(degenerate_single_class_replicates=1000),
    lambda b: b.update(matches=39),
    lambda b: b['model_ci'].pop('old_A_specialist'),
    lambda b: b.clear() or b.update(computed=False, reason='skipped'),
])
def test_incomplete_or_incoherent_eligible_records_are_rejected(valid_record, mutate):
    y, g, bt = valid_record
    bad = copy.deepcopy(bt)
    mutate(bad)
    problems, eligible = PR.bootstrap_cell_acceptance(bad, y, g)
    assert eligible and problems


def test_sparse_and_single_class_cells_require_explicit_skip(valid_record):
    _, _, bt = valid_record
    for y, g, _ in (_cell(n_matches=29), _cell(single_class=True)):
        assert PR.bootstrap_cell_acceptance(dict(computed=False, reason='matches<30 or one class'), y, g) == ([], False)
        assert PR.bootstrap_cell_acceptance(copy.deepcopy(bt), y, g)[0]
        assert PR.bootstrap_cell_acceptance(dict(computed=False), y, g)[0]


def test_pre_fit_key_guard_blocks_duplicate_match_s_keys():
    assert FIT.assert_unique_keys(np.array(['a', 'a', 'b']), np.array([1, 2, 1])) == 3
    with pytest.raises(SystemExit):
        FIT.assert_unique_keys(np.array(['a', 'a', 'b']), np.array([1, 1, 1]))


def test_partial_family_artifacts_are_detected_and_left_untouched(tmp_path):
    assert FIT.partial_family_artifacts(tmp_path, 'lgbm', 'T') == []
    b = tmp_path / 'models' / 'T' / 'lgbm' / 'lgbm_L15_M50.joblib'
    b.parent.mkdir(parents=True)
    b.write_bytes(b'attempt bundle')
    (tmp_path / 'internal_stop').mkdir()
    (tmp_path / 'internal_stop' / 'lgbm_T.json').write_text('{}', encoding='utf-8')
    before = hashlib.sha256(b.read_bytes()).hexdigest()
    found = FIT.partial_family_artifacts(tmp_path, 'lgbm', 'T')
    assert found == ['models/T/lgbm/lgbm_L15_M50.joblib', 'internal_stop/lgbm_T.json']
    assert FIT.partial_family_artifacts(tmp_path, 'pt', 'T') == [] and FIT.partial_family_artifacts(tmp_path, 'lgbm', 'N') == []
    assert b.exists() and hashlib.sha256(b.read_bytes()).hexdigest() == before
    src = (ROOT / 'scripts' / 'iq20260915_fit.py').read_text(encoding='utf-8')
    assert src.index('partial = partial_family_artifacts(base, fam, coh)') < src.index("D = Q.load_trainval(") < src.index('assert_unique_keys(g, D') < src.index('for i, cfg in enumerate(cfgs)')


def test_report_no_longer_hardcodes_magnitude_or_infers_y_baseline_from_p_pre():
    src = (ROOT / 'scripts' / 'iq20260915_report.py').read_text(encoding='utf-8')
    assert '셋째~넷째' not in src and '0.25에 가깝' not in src and '기저 Brier' not in src
    assert 'P(W=1 | S_pre)' in src and 'P(ΔV>0 | S_pre)' in src and "positive_rate_match_weighted" in src
    assert "p['a_minus_b']['brier']['estimate']" in src


def test_timing_gate_uses_latest_passing_contract_run_before_first_full_fit():
    src = (ROOT / 'scripts' / 'iq20260915_postrun_checks.py').read_text(encoding='utf-8')
    assert "ct = max((r for r in runs if r['passed'] and ts(r['at']) <= first_fit)" in src
    assert "OUT / 'contract_tests' / 'result.json'" not in src
