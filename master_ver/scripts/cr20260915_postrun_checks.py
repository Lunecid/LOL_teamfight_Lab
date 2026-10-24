"""Post-run verification -> validation.json (independent of the pipeline code paths where possible)."""
from __future__ import annotations

import os

os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402

import numpy as np  # noqa: E402


def ts(s):
    return time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S'))


def main():
    checks, details = {}, {}

    def add(name, ok, **info):
        checks[name] = bool(ok)
        if info:
            details[name] = info

    fz = C.read_json(K.OUT / 'frozen_manifest.json')
    frozen_at = ts(fz['frozen_at'])
    # 1 integrity of read-only inputs
    diff = C.read_json(K.OUT / 'integrity' / 'snapshot_diff.json')
    for key, d in diff.items():
        if 'equal' in d:
            add(f'readonly_{key}_unchanged', d['equal'])
        else:
            add(f'readonly_{key}_unchanged', not d['changed'] and not d['removed'] and not d['added'],
                changed=d['changed'][:10], removed=d['removed'][:10], added=d['added'][:10])
    # 2 ordering
    cm = C.read_json(K.OUT / 'cohorts' / 'cohort_manifest.json')
    prov = C.read_json(K.OUT / 'role_supervision_provenance.json')
    proto = C.read_json(K.OUT / 'protocol.json')
    ct = C.read_json(K.OUT / 'contract_tests' / 'result.json')
    rman = C.read_json(K.OUT / 'role_models_manifest.json') if (K.OUT / 'role_models_manifest.json').exists() else None
    sels = {p.name: C.read_json(p) for p in sorted((K.OUT / 'selection').glob('*.json'))}
    sel_times = [ts(s['written_at']) for s in sels.values() if 'written_at' in s]
    add('cohort_manifest_before_protocol_and_fits', ts(cm['written_at']) <= ts(proto['written_at']) <= min(sel_times))
    add('contract_tests_passed_before_full_fits', ct['passed'] and ts(ct['at']) <= min(sel_times), run=ct['run'], at=ct['at'])
    if rman:
        add('role_provenance_before_role_models', ts(prov['written_at']) <= ts(rman['written_at']))
    add('all_selections_before_freeze', max(sel_times) <= frozen_at, last_selection=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(max(sel_times))),
        frozen_at=fz['frozen_at'])
    log = [json.loads(l) for l in (K.OUT / 'label_access_log.jsonl').read_text(encoding='utf-8').splitlines()] if (K.OUT / 'label_access_log.jsonl').exists() else []
    test_access = [r for r in log if any(s == 'MAIN_TEST' or s.startswith('EXT_') for s in r['sets'])]
    add('test_external_labels_opened_only_after_freeze', all(r['frozen_manifest_exists'] and ts(r['time']) >= frozen_at for r in test_access),
        accesses=len(test_access), first=test_access[0]['time'] if test_access else None)
    # 3 frozen hashes
    now = {p: C.sha256_file(K.OUT / p) == sha for p, sha in fz['frozen_files_sha256'].items()}
    add('frozen_artifacts_unchanged_now', all(now.values()), files=len(now))
    ha = C.read_json(K.OUT / 'eval' / 'hashes_after.json')
    add('frozen_artifacts_unchanged_through_evaluation', all(ha.values()))
    # 4 cohorts
    add('cohort_join_one_to_one_566452', cm['join']['ok'])
    add('cohort_partition_all_sets', all(v['T_and_N_intersection'] == 0 and v['T_union_N_equals_known'] for v in cm['sets'].values()))
    add('detector_train_fixture_pass', cm['train_detector_fixture']['pass_'])
    add('external_redetection_bitwise', all(v['pass_'] for v in cm['external_checks'].values()))
    # 5 A fits
    ok_spec = all(all(s['serialization_identity'].values()) and all(s['fitting_origins'].values()) for n, s in sels.items() if n.startswith('q_specialist_') and 'cohort' in s)
    add('specialist_serialization_and_fit_origins', ok_spec)
    add('specialist_predictor_sets_equal_frozen_full_run', all(s['manifest']['predictor_sets_sha256'] == C.read_json(K.FC / 'q_pre_only_schema.json')['predictor_sets_sha256']
                                                             for n, s in sels.items() if n.startswith('q_specialist_') and 'cohort' in s))
    # 6 role models
    if rman:
        add('role_models_converged', all(v['fit_record']['converged'] for v in rman['models'].values()),
            n_iter={k: v['fit_record']['n_iter'] for k, v in rman['models'].items()})
        with np.load(K.OUT / 'role_outputs' / 'MAIN_TRAIN_roles.npz', allow_pickle=False) as z:
            gm, gen = z['match'].astype(str), z['generator'].astype(str)
        exp = np.array(['oof_fold%d' % C.train_fold(m) for m in gm.tolist()])
        add('role_oof_generator_is_own_heldout_fold_model', np.array_equal(gen, exp), matches=int(len(gm)))
        add('role_oof_self_match_exclusion_zero', rman['outputs']['MAIN_TRAIN']['oof_self_match_exclusion_violations'] == 0)
    # 7 arms
    for coh in fz.get('arms', {}):
        s = sels[f'arms_{coh}_h90.json']
        add(f'arms_{coh}_reload_identity_select', all(all(l['serialization_identity_select'].values()) for l in s['fit_logs'].values()))
        add(f'arms_{coh}_ridge_converged', all(l.get('ridge_converged', True) for l in s['fit_logs'].values()))
    # 8 evaluation reproduction and independent metric reconstruction
    RA = C.read_json(K.OUT / 'eval' / 'results_A.json')['results']
    add('pooled_E_baseline_reproduces_full_run_metrics', all(RA[n][h]['E_pooled_all_valid_rows']['reproduces_full_run_metrics'] for n in RA for h in RA[n]))
    maxd = dict(auc=0.0, brier=0.0, logloss=0.0)
    cells = 0
    boot_point = 0.0
    for n in RA:
        for h in RA[n]:
            for coh, cc in RA[n][h]['cohorts'].items():
                with np.load(K.OUT / 'eval' / 'predictions' / f'A_{n}_{h}_{coh}.npz', allow_pickle=False) as z:
                    y, g, pool = z['y'], z['match'], z['pooled']
                    spec = z[f"spec_{cc['specialist_chosen']}"]
                w = C.weights(g)
                for key, p in (('pooled', pool), ('specialist', spec)):
                    rec = dict(auc=C.auc_direct(y, p, w), brier=C.brier_direct(y, p, w), logloss=C.logloss_direct(y, p, w))
                    for k in maxd:
                        if rec[k] is not None and cc[key][k] is not None:
                            maxd[k] = max(maxd[k], abs(rec[k] - cc[key][k]))
                    cells += 1
                    if 'bootstrap' in cc:
                        bp = cc['bootstrap']['model_ci'][key]
                        boot_point = max(boot_point, abs(bp['auc']['estimate'] - cc[key]['auc']), abs(bp['brier']['estimate'] - cc[key]['brier']))
    if (K.OUT / 'eval' / 'results_C.json').exists():
        RC = C.read_json(K.OUT / 'eval' / 'results_C.json')['results']
        for n in RC:
            for coh, cc in RC[n].items():
                with np.load(K.OUT / 'eval' / 'predictions' / f'C_{n}_h90_{coh}.npz', allow_pickle=False) as z:
                    y, g = z['y'], z['match']
                    w = C.weights(g)
                    for c in list(cc['metrics'])[:40]:
                        p = z[c]
                        rec = dict(auc=C.auc_direct(y, p, w), brier=C.brier_direct(y, p, w), logloss=C.logloss_direct(y, p, w))
                        for k in maxd:
                            if rec[k] is not None and cc['metrics'][c][k] is not None:
                                maxd[k] = max(maxd[k], abs(rec[k] - cc['metrics'][c][k]))
                        cells += 1
                ident = [v['within_1e_12'] for v in cc['reload_identity_first_rows'].values()] + \
                        [v['within_1e_12'] for v in cc.get('ablation_reload_identity_first_rows', {}).values()]
                add(f'C_{n}_{coh}_reload_identity_within_1e-12', all(ident))
    add('independent_metric_reconstruction_lt_1e-10', max(maxd.values()) < 1e-10, max_abs=maxd, cells=cells)
    add('bootstrap_point_estimates_equal_evaluate_lt_1e-10', boot_point < 1e-10, max_abs=boot_point)
    # 9 SHAP
    sp = K.OUT / 'shap' / 'shap_summary.json'
    if sp.exists():
        sh = C.read_json(sp)
        add('shap_checks', sh['all_checks_pass'], per_cohort={k: v['checks'] for k, v in sh['cohorts'].items()})
    # 10 smoke isolation
    smoke = K.OUT / 'smoke_train_only'
    add('smoke_outputs_have_no_eval_or_test_predictions', not (smoke / 'eval').exists() and not any(smoke.rglob('*TEST*')) and not any(smoke.rglob('EXT_*')))
    res = dict(version=K.VERSION, role=K.ROLE_TAG, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), checks=checks,
               failed=[k for k, v in checks.items() if not v], details=details)
    sha = C.write_json(K.OUT / 'validation.json', res)
    print(json.dumps(dict(failed=res['failed'], n=len(checks), sha256=sha)))
    return 0 if not res['failed'] else 1


if __name__ == '__main__':
    sys.exit(main())
