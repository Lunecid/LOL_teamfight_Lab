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
import iq20260915_common as Q  # noqa: E402
import ta20260916_common as T  # noqa: E402
import bsh20260916_common as B  # noqa: E402
import cs20260916_common as CS  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

OUT = CS.OUT
ts = CS.ts


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    CS.log_command()
    st = CS.Status('postrun_checks')
    checks, info = {}, {}

    def add(name, ok, **extra):
        checks[name] = bool(ok)
        if extra:
            info[name] = C.clean(extra)
        st.log(f'{"PASS" if ok else "FAIL"} {name}')

    try:
        proto = C.read_json(OUT / 'protocol.json')
        fz = C.read_json(CS.frozen_path(OUT))
        summ = C.read_json(OUT / 'shap' / 'shap_summary.json')
        ct = C.read_json(OUT / 'contract_tests' / 'result.json')
        frozen_t = ts(fz['frozen_at'])
        diff = C.read_json(OUT / 'integrity' / 'snapshot_diff.json')
        # documentation written by this session during the run (the related-work / measurement addendum); parents are never edited or removed
        OWN_ADDED_DOCS = {'docs/tog_delta_v_20260916/related_work_measurement_addendum_20260916.md', 'docs/tog_delta_v_20260916/references_addendum.bib'}
        for k, v in diff.items():
            if isinstance(v, dict):
                ok = v['equal'] or (k == 'inventory_docs' and not v.get('changed') and not v.get('removed') and set(v.get('added', [])) <= OWN_ADDED_DOCS)
                add(f'parent_unchanged_{k}', ok, changed=v.get('changed'), removed=v.get('removed'), added=v.get('added'), checked=v.get('checked'),
                    note=None if v['equal'] else 'only this session\'s own addendum files were added under docs/; no parent file changed or removed')
        add('parent_expected_hashes_protocol_equal_now', all(C.sha256_file(CS.ROOT / p) == sha for p, sha in proto['parents']['expected_parent_file_sha256'].items()))
        add('spec_unchanged_since_protocol', C.sha256_file(CS.SPEC) == proto['spec_sha256'])
        pp = proto['parents']
        add('parent_frozen_manifests_and_bundles_unchanged', C.sha256_file(Q.frozen_path(CS.IQ)) == pp['iq_frozen_manifest_sha256'] and C.sha256_file(T.frozen_path(CS.TA)) == pp['ta_frozen_manifest_sha256']
            and C.sha256_file(B.frozen_path(CS.BSH)) == pp['bsh_frozen_manifest_sha256'] and C.sha256_file(CS.BSH / 'shap' / 'shap_summary.json') == pp['bsh_shap_summary_sha256']
            and all(C.sha256_file(CS.parent_winner(m, 'T')[0]) == v['bundle_sha256_parent'] for m, v in fz['models'].items()) and C.sha256_file(CS.V_ADAPTER) == fz['adapter_sha256'])
        # ordering and sealing
        snap_t = ts(C.read_json(OUT / 'integrity' / 'snapshot_before.json')['taken_at'])
        runs = [C.read_json(p) for p in sorted(OUT.glob('contract_tests/result_run*.json'), key=lambda p: int(p.stem.replace('result_run', '')))]
        latest_pass = max(ts(r['at']) for r in runs if r['passed'])
        smoke_fz, smoke_sum = C.read_json(CS.frozen_path(CS.SMOKE)), C.read_json(CS.SMOKE / 'shap' / 'shap_summary.json')
        rash = {coh: C.read_json(OUT / 'rashomon' / f'{coh}.json') for coh in CS.COHORTS}
        rash_t = max(ts(r['written_at']) for r in rash.values())
        add('order_snapshot_tests_protocol_smoke_rashomon_tests_freeze_shap',
            snap_t <= ts(runs[0]['at']) <= ts(proto['written_at']) <= ts(smoke_fz['frozen_at']) <= ts(smoke_sum['written_at']) and rash_t <= latest_pass <= frozen_t <= ts(summ['written_at']))
        add('contract_tests_passed_no_skips', ct['passed'] and ct['failed_count'] == 0 and ct['skipped_count'] == 0 and ct['error_count'] == 0, passed=ct['passed_count'], run=ct['run'])
        add('final_source_hashes_equal_frozen', all(C.sha256_file(CS.ROOT / p) == sha for p, sha in fz['source_sha256'].items()
                                                   if p.split('/')[-1] in ('cs20260916_common.py', 'cs20260916_shap.py', 'cs20260916_rashomon.py', 'cs20260916_protocol_freeze.py', 'fc20260915_shap.py',
                                                                           'bsh20260916_common.py', 'iq20260915_common.py', 'ta20260916_common.py', 'fc20260915_common.py', 'cr20260915_common.py')))
        add('pinned_files_unchanged_since_freeze', all(C.sha256_file(OUT / p) == sha for p, sha in fz['pinned_files_sha256'].items()) and C.sha256_file(OUT / 'protocol.json') == fz['protocol_sha256'])
        acc = [json.loads(l) for l in (OUT / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        sealed = [a for a in acc if a['sealed']]
        add('sealed_accesses_only_after_freeze', bool(sealed) and all(a['frozen_manifest_exists'] and ts(a['time']) >= frozen_t for a in sealed), n_sealed=len(sealed))
        sacc = [json.loads(l) for l in (CS.SMOKE / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        add('smoke_read_train_only', all(a['set'] == 'MAIN_TRAIN' for a in sacc) and smoke_sum['all_checks_pass'])
        add('shap_summary_all_checks_pass', summ['all_checks_pass'] and summ['frozen_manifest_sha256'] == C.sha256_file(CS.frozen_path(OUT)))
        # Part A recomputation from saved arrays and fresh-process reproduction
        fz_fc = C.read_json(CS.FC / 'frozen_manifest.json')
        adapter = C.load_v_adapter(CS.V_ADAPTER, fz_fc['v_final_sha256'])
        schema = C.read_json(CS.FC / 'q_pre_only_schema.json')
        ridge, names = list(schema['predictor_sets']['ridge']), list(schema['input_names_all'])
        smap = CS.StateMap(adapter, names, ridge)
        gidx = [proto['part_a']['group_indices'][g] for g in CS.BASE_GROUPS]
        add('base_groups_recomputed_equal_protocol', smap.base_groups() == gidx and C.sha256_json(smap.base_names) == fz['base_names_sha256'])
        bundles = {m: (joblib.load(CS.parent_winner(m, 'T')[0]), CS.parent_winner(m, 'T')[1]) for m in CS.MODELS}
        f = CS.composite_fn(adapter, smap, bundles)
        F_te, Lb_te, Co_te = CS.load_parent_set('MAIN_TEST', OUT, 'post-run recomputation (sealed; after freeze)')
        F_tr, _, _ = CS.load_parent_set('MAIN_TRAIN', OUT, 'post-run background rows')
        te_key = {(m, int(s)): i for i, (m, s) in enumerate(zip(F_te['match'].astype(str).tolist(), F_te['s_ms'].tolist()))}
        tr_key = {(m, int(s)): i for i, (m, s) in enumerate(zip(F_tr['match'].astype(str).tolist(), F_tr['s_ms'].tolist()))}
        for m in CS.MODELS:
            for cell in CS.CELLS:
                with np.load(OUT / 'shap' / f'shap_{m}_T_{cell}.npz', allow_pickle=False) as z:
                    Z = {k: z[k] for k in z.files}
                rec = summ['models'][m]['cells'][cell]
                ok = np.max(np.abs(Z['phi'].sum(1) + Z['base'] - Z['F'])) < 1e-9 and np.max(np.abs(Z['v_full'] - Z['F'])) < 1e-9
                ma = np.mean(np.abs(Z['phi']), 0)
                ok &= max(abs(ma[i] - rec['mean_abs'][g]) for i, g in enumerate(CS.BASE_GROUPS)) < 1e-12 and max(abs(ma[i] / ma.sum() - rec['share_of_sum_abs'][g]) for i, g in enumerate(CS.BASE_GROUPS)) < 1e-12
                bs = np.load(CS.BSH / 'shap' / f'shap_{m}_T_{cell}.npz', allow_pickle=False)
                ok &= np.array_equal(bs['explained_match'].astype(str), Z['explained_match'].astype(str)) and np.array_equal(bs['explained_s_ms'], Z['explained_s_ms']) and np.max(np.abs(bs['q_final'] - Z['F'])) < 1e-9
                ok &= np.array_equal(bs['phi'], Z['phi_input_space'])
                ex = np.asarray([te_key[(mm, int(s))] for mm, s in zip(Z['explained_match'].astype(str).tolist(), Z['explained_s_ms'].tolist())])
                bg = np.asarray([tr_key[(mm, int(s))] for mm, s in zip(Z['background_match'].astype(str).tolist(), Z['background_s_ms'].tolist())])
                ok &= np.array_equal(ex, Z['explained_row_index']) and np.array_equal(bg, Z['background_row_index'])
                Zb_bg = smap.base_from_inputs(F_tr['X_input'][bg])
                Zb_ex = smap.base_from_inputs(F_te['X_input'][ex[:2]])
                for r in range(2):
                    res = CS.exact_group_shapley_multi(f, list(CS.MODELS), Zb_ex[r], Zb_bg, gidx)
                    ok &= np.array_equal(res[m][0], Z['phi'][r]) and res[m][1] == Z['base'][r]
                add(f'partA_{m}_{cell}_efficiency_summary_rows_and_fresh_recomputation_verified', ok, rows=int(len(ex)))
        F_te = Lb_te = Co_te = F_tr = None
        # Part B recomputation; candidates whose recomputed prediction is not bitwise identical to the parent's saved column must be the
        # incremental-q logistic ones (the parent fit/predicted from column-fancy-indexed F-ordered arrays) and the difference must be numerical noise
        F_va, Lb_va, Co_va = CS.load_parent_set('MAIN_VALIDATION', OUT, 'post-run Part B recomputation (unsealed)')
        ridge_ix = [names.index(n) for n in ridge]
        for coh in CS.COHORTS:
            r = rash[coh]
            flagged = [c for c in r['candidates'] if not c['saved_prediction_identical']]
            ok = r['n_candidates'] == 78 and set(r['saved_prediction_not_identical']) == {c['name'] for c in flagged} and all(c['source'] == 'iq' and c['family'] == 'logit' for c in flagged)
            rows = CS.cohort_rows(F_va, Lb_va, Co_va, coh, sub_roles=('Q_SELECT',))
            Xv = np.ascontiguousarray(F_va['X_input'][rows][:, ridge_ix])
            gv, sv = F_va['match'][rows].astype(str), F_va['s_ms'][rows]
            maxd = 0.0
            for src, fam in sorted({(c['source'], c['family']) for c in flagged}):
                with np.load(CS.candidate_predictions(src, fam, coh), allow_pickle=False) as z:
                    pos = {kk: i for i, kk in enumerate(zip(z['match'].astype(str).tolist(), z['s_ms'].tolist()))}
                    jx = np.asarray([pos[(m, int(v))] for m, v in zip(gv.tolist(), sv.tolist())])
                    saved = {c['name']: z[f"{c['config']}__{c['calibration']}"][jx] for c in flagged if (c['source'], c['family']) == (src, fam)}
                for cfg in sorted({c['config'] for c in flagged if (c['source'], c['family']) == (src, fam)}):
                    b = joblib.load(next(p for s_, f_, cf, p in CS.candidate_bundles(coh) if (s_, f_, cf) == (src, fam, cfg)))
                    raw = b['base'].raw(Xv)
                    for cal in CS.CALS:
                        nm = f'{src}:{fam}:{cfg}__{cal}'
                        if nm in saved:
                            maxd = max(maxd, float(np.max(np.abs(Q.calibrate(b, cal, raw) - saved[nm]))))
            ok &= maxd < 1e-9
            info[f'partB_{coh}_flagged_candidates'] = dict(n_flagged=len(flagged), max_abs_diff_recomputed_vs_saved=maxd)
            best = min(c['brier'] for c in r['candidates'])
            ok &= abs(best - r['best_brier']) < 1e-15
            for eps_key, s_ in r['epsilon_sets'].items():
                members = [c for c in r['candidates'] if c['brier'] <= best + s_['epsilon']]
                ok &= [c['name'] for c in members] == s_['members'] and s_['n_members'] == len(members)
                for g in CS.INPUT_GROUPS:
                    vals = [c['reliance'][g] for c in members]
                    ok &= abs(min(vals) - s_['reliance_range'][g]['min']) < 1e-15 and abs(max(vals) - s_['reliance_range'][g]['max']) < 1e-15
            ok &= all(abs(np.mean(c['reliance_seeds'][g]) - c['reliance'][g]) < 1e-12 for c in r['candidates'] for g in CS.INPUT_GROUPS)
            add(f'partB_{coh}_candidates_epsilon_sets_and_ranges_recomputed', ok, n_candidates=r['n_candidates'], sets={k: v['n_members'] for k, v in r['epsilon_sets'].items()})
        stat = C.read_json(OUT / 'status.json')
        add('required_stage_groups_complete', all(stat['groups'].get(g, {}).get('state') == 'complete' for g in ('rashomon', 'shap', 'shap_smoke')))
        add('no_bytecode_written_in_parent_scripts_tests', diff['pycache_scripts']['equal'] and diff['pycache_tests']['equal'])
        failed = [k for k, v in checks.items() if not v]
        C.write_json(OUT / 'validation.json', dict(role=CS.ROLE_TAG, version=CS.VERSION, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), n_checks=len(checks), failed=failed, passed=len(checks) - len(failed),
                                                    checks=checks, info=info, frozen_manifest_sha256=C.sha256_file(CS.frozen_path(OUT)), shap_summary_sha256=C.sha256_file(OUT / 'shap' / 'shap_summary.json')))
        st.update('complete' if not failed else 'failed', 'validated', n_checks=len(checks), failed=failed[:20], next_step='report')
        print(f'checks {len(checks)} failed {len(failed)}: {failed}')
        return 0 if not failed else 1
    except Exception as exc:
        st.log(traceback.format_exc())
        CS.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'validate', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
