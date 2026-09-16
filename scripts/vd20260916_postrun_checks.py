"""Stage V: post-run verification -> validation.json (recomputation from the saved per-row arrays; real pass/fail)."""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys  # noqa: E402

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import vd20260916_common as V  # noqa: E402
import ta20260916_postrun_checks as TPC  # noqa: E402  (pure helper: ts)

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

OUT = V.OUT
ts = TPC.ts


def main():
    V.log_command()
    st = V.Status('postrun_checks')
    checks, info = {}, {}

    def add(name, ok, **extra):
        checks[name] = bool(ok)
        if extra:
            info[name] = C.clean(extra)
        st.log(f'{"PASS" if ok else "FAIL"} {name}')

    try:
        proto = C.read_json(OUT / 'protocol.json')
        fz = C.read_json(V.frozen_path(OUT))
        res = C.read_json(OUT / 'results.json')
        ct = C.read_json(OUT / 'contract_tests' / 'result.json')
        diff = C.read_json(OUT / 'integrity' / 'snapshot_diff.json')
        for k, v in diff.items():
            if isinstance(v, dict):
                add(f'parent_unchanged_{k}', v['equal'], changed=v.get('changed'), removed=v.get('removed'), added=v.get('added'), checked=v.get('checked'))
        add('parent_expected_hashes_protocol_equal_now', all(C.sha256_file(V.ROOT / p) == sha for p, sha in proto['parents']['expected_parent_file_sha256'].items()))
        add('spec_unchanged_since_protocol', C.sha256_file(V.SPEC) == proto['spec_sha256'])
        add('adapter_unchanged', C.sha256_file(V.V_ADAPTER) == fz['adapter_sha256'] == proto['parents']['adapter_sha256'])
        snap_t = ts(C.read_json(OUT / 'integrity' / 'snapshot_before.json')['taken_at'])
        runs = [C.read_json(p) for p in sorted(OUT.glob('contract_tests/result_run*.json'), key=lambda p: int(p.stem.replace('result_run', '')))]
        smoke_fz, smoke_res = C.read_json(V.frozen_path(V.SMOKE)), C.read_json(V.SMOKE / 'results.json')
        latest_pass = max(ts(r['at']) for r in runs if r['passed'])
        add('order_snapshot_tests_protocol_smoke_tests_freeze_analysis',
            snap_t <= ts(runs[0]['at']) <= ts(proto['written_at']) <= ts(smoke_fz['frozen_at']) <= ts(smoke_res['evaluated_at']) and latest_pass <= ts(fz['frozen_at']) <= ts(res['evaluated_at']))
        add('contract_tests_passed_no_skips', ct['passed'] and ct['failed_count'] == 0 and ct['skipped_count'] == 0 and ct['error_count'] == 0, passed=ct['passed_count'], run=ct['run'])
        add('smoke_checks_pass', smoke_res['checks']['pass'] and smoke_res['smoke'])
        add('final_source_hashes_equal_frozen_for_analyze_common', all(C.sha256_file(V.ROOT / p) == sha for p, sha in fz['source_sha256'].items()
                                                                       if p.split('/')[-1] in ('vd20260916_common.py', 'vd20260916_analyze.py', 'fc20260915_common.py')))
        acc = [json.loads(l) for l in (OUT / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        sealed = [a for a in acc if a['sealed']]
        add('sealed_accesses_only_after_freeze', bool(sealed) and all(a['frozen_manifest_exists'] and ts(a['time']) >= ts(fz['frozen_at']) for a in sealed), n_sealed=len(sealed))
        sacc = [json.loads(l) for l in (V.SMOKE / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        add('smoke_read_validation_only_unsealed', all(a['set'] in ('MAIN_VALIDATION', 'extract/MAIN/states') and not a['sealed'] for a in sacc))
        add('analysis_checks_pass', res['checks']['pass'], checks=res['checks'])
        add('all_valid_rows_collected_once', all(res['sets'][s]['rows'] == res['sets'][s]['valid_rows_expected'] == 163576 if s == 'MAIN_TEST' else res['sets'][s]['rows'] == res['sets'][s]['valid_rows_expected']
                                                for s in V.SETS), rows={s: res['sets'][s]['rows'] for s in V.SETS})
        # recomputation from saved arrays
        for s in V.SETS:
            with np.load(OUT / 'rows' / f'{s}_h90_decomposition.npz', allow_pickle=False) as z:
                R = {k: z[k] for k in z.files}
            Lb, Co = V.load_labels(s, OUT, 'post-run recomputation')
            li = R['label_row_index']
            ok = np.array_equal(R['match'], Lb['match'][li]) and np.array_equal(R['s_ms'], Lb['s'][li]) and np.array_equal(R['y'], Lb['Y_h90'][li]) and np.array_equal(R['cohort'], Co['cohort'][li])
            ok &= np.array_equal(R['p_pre'], Lb['p_pre'][li]) and np.array_equal(R['p_post'], Lb['p_post_h90'][li])
            ok &= bool(np.max(np.abs(R['group_sums'].sum(1) - R['delta_logit'])) < 1e-9) and bool(np.max(np.abs(R['group_sums'][:, list(V.GROUPS).index('champion_identity')])) == 0.0)
            ok &= bool(np.array_equal((R['delta_logit'] > 0).astype(int), R['y'].astype(int)))
            ok &= bool(np.max(np.abs((V.logit(R['p_post']) - V.logit(R['p_pre'])) - R['delta_logit'])) < 1e-8)
            ok &= len(set(li.tolist())) == len(li) == int((Lb['valid_h90'] == 1).sum())
            T_mask = R['cohort'] == V.COHORT_CODE['T']
            rep = res['sets'][s]['strata']['T:all']
            absG = np.abs(R['group_sums'][T_mask])
            ok &= bool(abs(rep['n'] - int(T_mask.sum())) == 0 and abs(rep['mean_abs_delta_logit'] - float(np.mean(np.abs(R['delta_logit'][T_mask])))) < 1e-12)
            ok &= bool(max(abs(rep['mean_abs_group'][g] - float(absG[:, k].mean())) for k, g in enumerate(V.GROUPS)) < 1e-12)
            ok &= bool(abs(rep['time_related_share_of_sum_abs_columns'] - float(R['time_related_abs_sum'][T_mask].sum() / R['abs_column_sum'][T_mask].sum())) < 1e-12)
            add(f'{s}_saved_rows_identities_and_T_summary_recomputed', ok)
        stat = C.read_json(OUT / 'status.json')
        add('required_stage_groups_complete', all(stat['groups'].get(g, {}).get('state') == 'complete' for g in ('analyze',)))
        add('no_bytecode_written_in_parent_scripts_tests', diff['pycache_scripts']['equal'] and diff['pycache_tests']['equal'])
        failed = [k for k, v in checks.items() if not v]
        C.write_json(OUT / 'validation.json', dict(role=V.ROLE_TAG, version=V.VERSION, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), n_checks=len(checks), failed=failed,
                                                    passed=len(checks) - len(failed), checks=checks, info=info, results_sha256=C.sha256_file(OUT / 'results.json')))
        st.update('complete' if not failed else 'failed', 'validated', n_checks=len(checks), failed=failed[:20], next_step='report')
        print(f'checks {len(checks)} failed {len(failed)}: {failed}')
        return 0 if not failed else 1
    except Exception as exc:
        st.log(traceback.format_exc())
        V.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'validate', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
