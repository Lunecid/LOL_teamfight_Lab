"""Stage V: post-run verification -> validation.json (independent recomputation where possible; real pass/fail)."""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')

import sys  # noqa: E402

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import fc20260915_shap as FS  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import ta20260916_common as T  # noqa: E402
import bsh20260916_common as B  # noqa: E402
import ta20260916_postrun_checks as TPC  # noqa: E402  (pure helper: ts)

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

OUT = B.OUT
ts = TPC.ts


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    B.log_command()
    st = B.Status('postrun_checks')
    checks, info = {}, {}

    def add(name, ok, **extra):
        checks[name] = bool(ok)
        if extra:
            info[name] = C.clean(extra)
        st.log(f'{"PASS" if ok else "FAIL"} {name}')

    try:
        proto = C.read_json(OUT / 'protocol.json')
        fz = C.read_json(B.frozen_path(OUT))
        summ = C.read_json(OUT / 'shap' / 'shap_summary.json')
        ct = C.read_json(OUT / 'contract_tests' / 'result.json')
        frozen_t = ts(fz['frozen_at'])
        diff = C.read_json(OUT / 'integrity' / 'snapshot_diff.json')
        for k, v in diff.items():
            if isinstance(v, dict):
                add(f'parent_unchanged_{k}', v['equal'], changed=v.get('changed'), removed=v.get('removed'), added=v.get('added'), checked=v.get('checked'))
        add('parent_expected_hashes_protocol_equal_now', all(C.sha256_file(B.ROOT / p) == sha for p, sha in proto['parents']['expected_parent_file_sha256'].items()))
        add('spec_unchanged_since_protocol', C.sha256_file(B.SPEC) == proto['spec_sha256'])
        add('parent_frozen_manifests_unchanged', C.sha256_file(Q.frozen_path(B.IQ)) == fz['parent_references']['iq_frozen_manifest_sha256']
            and C.sha256_file(T.frozen_path(B.TA)) == fz['parent_references']['ta_frozen_manifest_sha256'])
        add('parent_eval_prediction_files_unchanged', all(C.sha256_file(B.parent_eval_predictions(m, c)[0]) == sha for k, sha in fz['parent_references']['eval_predictions_sha256'].items()
                                                          for m, c in [(k.rsplit('_', 1)[0], k.rsplit('_', 1)[1])]))
        add('bundles_unchanged', all(C.sha256_file(B.ROOT / r['bundle']) == r['bundle_sha256'] for m in B.MODELS for r in [fz['models'][m][c] for c in B.COHORTS]))
        snap_t = ts(C.read_json(OUT / 'integrity' / 'snapshot_before.json')['taken_at'])
        runs = [C.read_json(p) for p in sorted(OUT.glob('contract_tests/result_run*.json'), key=lambda p: int(p.stem.replace('result_run', '')))]
        smoke_fz = C.read_json(B.frozen_path(B.SMOKE))
        smoke_sum = C.read_json(B.SMOKE / 'shap' / 'shap_summary.json')
        latest_pass = max(ts(r['at']) for r in runs if r['passed'])
        add('order_snapshot_tests_protocol_smoke_latest_tests_freeze_explanation',
            snap_t <= ts(runs[0]['at']) <= ts(proto['written_at']) <= ts(smoke_fz['frozen_at']) <= ts(smoke_sum['evaluated_at']) and latest_pass <= frozen_t <= ts(summ['evaluated_at']))
        add('contract_tests_passed_no_skips', ct['passed'] and ct['failed_count'] == 0 and ct['skipped_count'] == 0 and ct['error_count'] == 0, passed=ct['passed_count'], run=ct['run'])
        add('final_source_hashes_equal_frozen_for_freeze_shap_common',
            all(C.sha256_file(B.ROOT / p) == sha for p, sha in fz['source_sha256'].items()
                if p.split('/')[-1] in ('bsh20260916_common.py', 'bsh20260916_freeze.py', 'bsh20260916_shap.py', 'fc20260915_shap.py', 'iq20260915_common.py', 'ta20260916_common.py',
                                        'fc20260915_common.py', 'cr20260915_common.py')))
        acc = [json.loads(l) for l in (OUT / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        sealed = [a for a in acc if a['sealed']]
        add('sealed_accesses_only_after_freeze', bool(sealed) and all(a['frozen_manifest_exists'] and ts(a['time']) >= frozen_t for a in sealed), n_sealed=len(sealed))
        add('no_sealed_access_before_freeze_recorded_in_manifest', fz['sealed_accesses_before_freeze'] == 0)
        sacc = [json.loads(l) for l in (B.SMOKE / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        add('smoke_read_train_only', all(a['set'] == 'MAIN_TRAIN' for a in sacc) and smoke_sum['smoke_train_only'])
        add('smoke_checks_pass', smoke_sum['all_checks_pass'])
        add('freeze_prechecks_all_pass', all(all(v.values()) for v in fz['pre_freeze_checks'].values()))
        add('group_map_equals_protocol_and_schema', fz['groups'] == proto['groups']['record'] == B.group_map_record(list(C.read_json(B.FC / 'q_pre_only_schema.json')['predictor_sets']['ridge']),
                                                                                                                  C.read_json(B.FC / 'q_pre_only_schema.json')['shap_groups']))
        # explanation reconstruction from saved arrays
        F_te, Lb_te, Co_te, _ = B.load_parent_set('MAIN_TEST', OUT, 'post-run reconstruction')
        F_tr, Lb_tr, Co_tr, _ = B.load_parent_set('MAIN_TRAIN', OUT, 'post-run reconstruction (background)')
        names = [str(x) for x in F_te['input_names']]
        ridge = list(C.read_json(B.FC / 'q_pre_only_schema.json')['predictor_sets']['ridge'])
        cols = [names.index(n) for n in ridge]
        gidx = [fz['group_indices'][g] for g in B.GROUPS]
        m_te, s_te, m_tr, s_tr = F_te['match'].astype(str), F_te['s_ms'], F_tr['match'].astype(str), F_tr['s_ms']
        p_pre_all = F_te['X_input'][:, names.index('p_pre_V')]
        for model in B.MODELS:
            for coh in B.COHORTS:
                rec = fz['models'][model][coh]
                b = joblib.load(B.ROOT / rec['bundle'])
                f = B.model_fn(b, rec['calibration'])
                tr_rows = np.flatnonzero((Lb_tr['valid_h90'] == 1) & (Co_tr['cohort'] == B.COHORT_CODE[coh]))
                bg_expect = B.select_rows(B.TAG_BG, tr_rows, m_tr, s_tr, fz['rows']['n_background'], coh)
                te_rows_all = np.flatnonzero((Lb_te['valid_h90'] == 1) & (Co_te['cohort'] == B.COHORT_CODE[coh]))
                cell_rows = {'all': te_rows_all, 'B40': te_rows_all[B.b40_mask(p_pre_all[te_rows_all])]}
                cs = summ['models'][model][coh]['cells']
                for cell in B.CELLS:
                    with np.load(OUT / 'shap' / f'shap_{model}_{coh}_{cell}.npz', allow_pickle=False) as z:
                        S = {k: z[k] for k in z.files}
                    ex_expect = B.select_rows(B.TAG_EXPLAIN, cell_rows[cell], m_te, s_te, fz['rows']['n_explain'], coh, cell)
                    ok = np.array_equal(S['explained_row_index'], ex_expect) and np.array_equal(S['background_row_index'], bg_expect)
                    ok &= np.array_equal(S['y'], Lb_te['Y_h90'][ex_expect]) and np.array_equal(S['p_pre'], p_pre_all[ex_expect])
                    ok &= bool(np.max(np.abs(S['phi'].sum(1) + S['base'] - S['q_final'])) < 1e-8)
                    ok &= bool(np.array_equal(f(F_te['X_input'][ex_expect][:, cols]), S['q_final']))
                    Zb = F_tr['X_input'][bg_expect][:, cols]
                    re = np.stack([FS.group_exact_shapley(f, F_te['X_input'][ex_expect[i]][cols], Zb, gidx)[0] for i in range(2)])
                    ok &= bool(np.array_equal(re, S['phi'][:2]))
                    mean_abs = np.mean(np.abs(S['phi']), 0)
                    ok &= bool(np.max(np.abs(mean_abs - np.array([cs[cell]['global_mean_abs'][g] for g in B.GROUPS]))) < 1e-12)
                    ok &= cs[cell]['checks']['pass'] and cs[cell]['explained_rows'] == len(ex_expect) == fz['rows']['n_explain'] and cs[cell]['eligible_rows'] == len(cell_rows[cell])
                    ok &= cs[cell]['checks']['phi_plus_base_equals_parent_evaluation_prediction_max_abs'] < 1e-8 and cs[cell]['checks']['parent_eval_file_sha256_equals_frozen']
                    add(f'{model}_{coh}_{cell}_selection_additivity_recomputation_and_parent_prediction_identity', ok)
                b = None
        stat = C.read_json(OUT / 'status.json')
        add('required_stage_groups_complete', all(stat['groups'].get(g, {}).get('state') == 'complete' for g in ('driver_full', 'freeze', 'shap')))
        add('no_bytecode_written_in_parent_scripts_tests', diff['pycache_scripts']['equal'] and diff['pycache_tests']['equal'])
        failed = [k for k, v in checks.items() if not v]
        C.write_json(OUT / 'validation.json', dict(role=B.ROLE_TAG, version=B.VERSION, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), n_checks=len(checks), failed=failed,
                                                    passed=len(checks) - len(failed), checks=checks, info=info, frozen_manifest_sha256=C.sha256_file(B.frozen_path(OUT)),
                                                    shap_summary_sha256=C.sha256_file(OUT / 'shap' / 'shap_summary.json')))
        st.update('complete' if not failed else 'failed', 'validated', n_checks=len(checks), failed=failed[:20], next_step='report')
        print(f'checks {len(checks)} failed {len(failed)}: {failed}')
        return 0 if not failed else 1
    except Exception as exc:
        st.log(traceback.format_exc())
        B.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'validate', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
