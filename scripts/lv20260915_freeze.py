"""Label validity stage F: freeze both comparator families and chosen calibrators.

Requires: protocol.json, passing synthetic pytest contracts, passing TRAIN raw fixtures, passing smoke and full
V-model checks, and complete manifests for B_reg and B_econ. Verifies every adapter hash, confirms that the
outcome access log so far contains only TRAIN folds / V_CAL / V_SELECT and that no sealed generated label was
opened. Writes frozen_manifest.json (refuses to overwrite). After this file exists TEST/external W and labels open.
"""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import lv20260915_common as K  # noqa: E402

import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402


def main():
    st = K.Status('freeze')
    K.log_command()
    fz_path = K.frozen_manifest_path()
    if fz_path.exists():
        raise SystemExit('frozen_manifest.json exists; refusing to overwrite')
    problems = []
    pytest_logs = sorted((K.OUT / 'contracts').glob('pytest_run*.txt'))
    last = pytest_logs[-1].read_text(encoding='utf-8', errors='replace') if pytest_logs else ''
    if not re.search(r'\b\d+ passed\b', last) or re.search(r'\b(failed|error)\b', last):
        problems.append(f'synthetic pytest contracts not passing in {pytest_logs[-1].name if pytest_logs else None}')
    fixtures = C.read_json(K.OUT / 'contracts' / 'train_fixture_checks.json')
    if not fixtures['all_pass']:
        problems.append('TRAIN raw fixture checks failed')
    for kind in ('smoke', 'full'):
        p = K.OUT / 'contracts' / f'v_model_checks_{kind}.json'
        if not p.exists() or not C.read_json(p)['all_pass']:
            problems.append(f'v_model_checks_{kind} missing or failing')
    models = {}
    for M in K.MODELS:
        man_p = K.OUT / f'v_models_manifest_{M}.json'
        sel_p = K.OUT / 'selection' / f'selection_{M}.json'
        man = C.read_json(man_p)
        sel = C.read_json(sel_p)
        files = {man['final']['path']: man['final']['sha256']}
        for k, sha in man['candidates_sha256'].items():
            files[str(Path('models') / M / f'v_final_{k}.joblib')] = sha
        for k, sha in man['oof_sha256'].items():
            files[man['oof_paths'][k]] = sha
        actual = {p: C.sha256_file(K.OUT / p) for p in files}
        bad = [p for p in files if actual[p] != files[p]]
        if bad:
            problems.append(f'{M}: adapter hash mismatch {bad}')
        if sel['chosen'] != man['chosen'] or sel['chosen_sha256'] != man['final']['sha256']:
            problems.append(f'{M}: selection/manifest disagree')
        models[M] = dict(chosen=man['chosen'], final_path=man['final']['path'], final_sha256=man['final']['sha256'],
                         candidates_sha256=man['candidates_sha256'], oof_sha256=man['oof_sha256'], oof_paths=man['oof_paths'],
                         selection_sha256=C.sha256_file(sel_p), manifest_sha256=C.sha256_file(man_p),
                         predictions_sha256=man['predictions_sha256'],
                         calibration_fit=sel['sigmoid_fit'] if man['chosen'] == 'sigmoid_pos' else None,
                         converged_all=bool(man['final_fit']['converged'] and all(v['fit']['converged'] for v in man['folds'].values())))
    allowed = K.ParentReadOnly.PRE_FREEZE_OUTCOME_ROLES
    acc = [json.loads(x) for x in (K.OUT / 'outcome_access_log.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    viol = [a for a in acc if a['set'] != 'MAIN' or not a['sub_roles'] or not set(a['sub_roles']) <= allowed]
    if viol:
        problems.append(f'pre-freeze outcome access outside TRAIN/V_CAL/V_SELECT: {viol[:3]}')
    lab_acc = [json.loads(x) for x in (K.OUT / 'label_access_log.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] \
        if (K.OUT / 'label_access_log.jsonl').exists() else []
    sealed = [a for a in lab_acc if a['set'] in K.SEALED_LABEL_SETS]
    if sealed:
        problems.append(f'sealed generated labels opened before freeze: {sealed[:3]}')
    generated = [p for p in (K.OUT / 'alt_labels').rglob('*')] if (K.OUT / 'alt_labels').exists() else []
    if generated or (K.OUT / 'eval').exists():
        problems.append('alternative labels or evaluation outputs exist before freeze')
    if problems:
        st.update('failed', 'freeze', error='; '.join(problems)[:1500], next_step='resolve before any sealed access')
        raise SystemExit('freeze refused: ' + '; '.join(problems))
    fz = dict(role=K.ROLE_TAG, version=K.VERSION, frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
              protocol_sha256=C.sha256_file(K.OUT / 'protocol.json'), feature_lists_sha256=C.sha256_file(K.OUT / 'feature_lists.json'),
              models=models,
              contracts=dict(pytest_log=pytest_logs[-1].name, pytest_log_sha256=C.sha256_file(pytest_logs[-1]),
                             train_fixture_checks_sha256=C.sha256_file(K.OUT / 'contracts' / 'train_fixture_checks.json'),
                             v_model_checks_smoke_sha256=C.sha256_file(K.OUT / 'contracts' / 'v_model_checks_smoke.json'),
                             v_model_checks_full_sha256=C.sha256_file(K.OUT / 'contracts' / 'v_model_checks_full.json')),
              primary_A=dict(v_final_sha256=C.read_json(K.FC / 'v_models_manifest.json')['final']['sha256'],
                             v_oof_sha256=C.read_json(K.FC / 'v_models_manifest.json')['oof_sha256'],
                             parent_frozen_manifest_sha256=C.sha256_file(K.FC / 'frozen_manifest.json')),
              pre_freeze_outcome_access=[dict(time=a['time'], set=a['set'], sub_roles=a['sub_roles'], purpose=a['purpose'], rows=a['rows'])
                                         for a in acc],
              pre_freeze_label_access=[dict(time=a['time'], set=a['set'], purpose=a['purpose']) for a in lab_acc],
              statement=('Both comparator families (base fits, calibration choice and fold adapters) are frozen before this study '
                         'reads any TEST/external/Q_CAL/Q_SELECT outcome or any TEST/external generated label, and before any '
                         'comparator label or score exists. Parent TEST results were already known (prior exposure).'))
    sha = C.write_json(fz_path, fz)
    st.update('complete', 'freeze', frozen_manifest_sha256=sha, next_step='V evaluation and comparator labels')
    return 0


if __name__ == '__main__':
    sys.exit(main())
