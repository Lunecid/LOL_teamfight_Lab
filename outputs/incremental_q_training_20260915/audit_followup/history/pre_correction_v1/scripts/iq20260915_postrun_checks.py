"""Stage V: post-run verification -> validation.json (independent recomputation where possible; real pass/fail).

Re-derives from saved artifacts and parent inputs: parent integrity diff, stage ordering and sealed-access timing,
frozen hashes, counts and schema, PT dimension and TRAIN-only knots / scalers / imputers, logistic convergence,
LightGBM stop-record consistency and stop allocation, equal match weights, Q_CAL calibrator refits, Q_SELECT selection
reproduction with direct formulas, reload identity of the frozen winners on MAIN TEST rows, legacy joins, cell
membership and an independent metric / bootstrap point reconstruction from the saved evaluation predictions.
"""
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
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def ts(s):
    return time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S'))


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    Q.log_command()
    st = Q.Status('postrun_checks')
    checks, details = {}, {}

    def add(name, ok, **info):
        checks[name] = bool(ok)
        if info:
            details[name] = C.clean(info)
        st.log(f'{"PASS" if ok else "FAIL"} {name} ' + (json.dumps(C.clean(info), ensure_ascii=False)[:300] if info else ''))

    try:
        OUT = Q.OUT
        proto = C.read_json(OUT / 'protocol.json')
        fz = C.read_json(OUT / 'frozen_manifest.json')
        frozen_t = ts(fz['frozen_at'])
        schema = C.read_json(Q.FC / 'q_pre_only_schema.json')
        ridge = schema['predictor_sets']['ridge']
        # ---------------------------------------------------------------- 1 parents
        diff = C.read_json(OUT / 'integrity' / 'snapshot_diff.json')
        for k, v in diff.items():
            if isinstance(v, dict):
                add(f'parent_unchanged_{k}', v['equal'], changed=v.get('changed'), removed=v.get('removed'), added=v.get('added'), checked=v.get('checked'))
        add('parent_expected_hashes_protocol_equal_now', all(C.sha256_file(Q.ROOT / p) == sha for p, sha in proto['parents']['expected_parent_file_sha256'].items()),
            files=len(proto['parents']['expected_parent_file_sha256']))
        add('spec_unchanged_since_protocol', C.sha256_file(Q.SPEC) == proto['spec_sha256'])
        # ---------------------------------------------------------------- 2 ordering
        snap_b = C.read_json(OUT / 'integrity' / 'snapshot_before.json')
        ct = C.read_json(OUT / 'contract_tests' / 'result.json')
        sfz = C.read_json(Q.frozen_path(Q.SMOKE))
        sels = {f'{f}_{c}': C.read_json(OUT / 'selection' / f'{f}_{c}.json') for f in Q.FAMILIES for c in Q.COHORTS}
        first_fit = min(ts(s['written_at']) - s['seconds_total'] for s in sels.values())
        runs = [C.read_json(p) for p in sorted((OUT / 'contract_tests').glob('result_run*.json'))]
        first_pass = min(ts(r['at']) for r in runs if r['passed'])
        add('order_snapshot_tests_protocol_smoke_latest_tests_before_full_fits',
            ts(snap_b['taken_at']) <= first_pass <= ts(proto['written_at']) <= ts(sfz['frozen_at']) <= ts(ct['at']) <= first_fit,
            snapshot=snap_b['taken_at'], first_passing_tests=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first_pass)),
            protocol=proto['written_at'], smoke_frozen=sfz['frozen_at'], latest_passing_tests=ct['at'],
            first_full_fit_start=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first_fit)),
            test_runs=[dict(run=r['run'], at=r['at'], passed=r['passed']) for r in runs],
            note='contract tests run 1 preceded the protocol (no fits); run 2 re-ran them on the final code after the smoke fix')
        smoke_eval = C.read_json(Q.SMOKE / 'eval' / 'results.json')
        add('smoke_evaluation_completed_before_full_fits', smoke_eval['complete'] and ts(smoke_eval['evaluated_at']) <= first_fit)
        add('all_selections_before_freeze', max(ts(s['written_at']) for s in sels.values()) <= frozen_t, frozen_at=fz['frozen_at'])
        add('contract_tests_passed_no_skips', ct['passed'] and ct['failed_count'] == 0 and ct['skipped_count'] == 0 and ct['error_count'] == 0,
            passed=ct['passed_count'], run=ct['run'])
        acc = [json.loads(l) for l in (OUT / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        sealed = [a for a in acc if a['sealed']]
        add('sealed_accesses_only_after_freeze', bool(sealed) and all(a['frozen_manifest_exists'] and ts(a['time']) >= frozen_t for a in sealed),
            sealed_accesses=len(sealed), first=sealed[0]['time'] if sealed else None)
        add('no_sealed_access_before_freeze_recorded_in_manifest', fz['sealed_accesses_before_freeze'] == 0)
        sacc = [json.loads(l) for l in (Q.SMOKE / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        add('smoke_read_train_only', all(a['set'] == 'MAIN_TRAIN' for a in sacc) and not any(Q.SMOKE.rglob('*TEST*')) and not any(Q.SMOKE.rglob('*EXT_*')),
            accesses=len(sacc))
        res = C.read_json(OUT / 'eval' / 'results.json')
        add('evaluation_after_freeze', ts(res['evaluated_at']) >= frozen_t and res['complete'])
        # ---------------------------------------------------------------- 3 frozen hashes
        now = {p: C.sha256_file(OUT / p) == sha for p, sha in fz['frozen_files_sha256'].items()}
        add('frozen_artifacts_unchanged_now', all(now.values()), files=len(now))
        hb, ha = C.read_json(OUT / 'eval' / 'hashes_before.json'), C.read_json(OUT / 'eval' / 'hashes_after.json')
        add('frozen_artifacts_unchanged_through_evaluation', all(hb['results'].values()) and all(ha['results'].values()) and res['frozen_hashes_unchanged'])
        add('protocol_hash_in_freeze', fz['protocol_sha256'] == C.sha256_file(OUT / 'protocol.json'))
        # ---------------------------------------------------------------- 4 per cohort re-derivation on TRAIN / Q roles
        vman = C.read_json(Q.FC / 'v_models_manifest.json')
        stop_flags = {}
        for coh in Q.COHORTS:
            D = Q.load_trainval(OUT, smoke=False, cohort=coh)
            names = D['names']
            M = {k: D['role'] == k for k in ('TRAIN', 'Q_CAL', 'Q_SELECT')}
            add(f'{coh}_counts_equal_specification', all(int(M[k].sum()) == Q.EXPECTED_COUNTS[k][coh] for k in M),
                counts={k: int(m.sum()) for k, m in M.items()})
            prov = Q.oof_provenance(D['g'][M['TRAIN']], D['sr_true'][M['TRAIN']], D['adapter_id'][M['TRAIN']], D['adapter_sha256'][M['TRAIN']], vman)
            add(f'{coh}_train_rows_use_own_heldout_fold_oof_V', prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold'], **prov)
            gtr = D['g'][M['TRAIN']]
            stop = Q.stop_mask(gtr)
            stop_flags[coh] = dict(zip(gtr.tolist(), stop.tolist()))
            Xtr = D['X'][M['TRAIN']]
            ytr = D['y'][M['TRAIN']]
            for fam in Q.FAMILIES:
                key = f'{fam}_{coh}'
                sel = sels[key]
                add(f'{key}_selection_counts_equal_specification', all(sel['split_counts'][k]['rows'] == Q.EXPECTED_COUNTS[k][coh] for k in M))
                add(f'{key}_input_schema', sel['input_names_sha256'] == C.sha256_json(list(Q.PT_INPUTS) if fam == 'pt' else ridge)
                    and sel['n_inputs'] == (2 if fam == 'pt' else Q.EXPECTED_RIDGE_COUNT))
                add(f'{key}_weights_equal_per_match', all(abs(w['per_match_total_min'] - w['per_match_total_max']) < 1e-9 and abs(w['weight_mean'] - 1) < 1e-9
                                                          for w in sel['weights'].values()),
                    membership={k: w['membership_sha256'] for k, w in sel['weights'].items()})
                add(f'{key}_weight_membership_matches_roles', all(sel['weights'][k]['membership_sha256'] == Q.membership_sha(D['g'][M[k]]) for k in M))
                with np.load(OUT / 'predictions' / f'{key}_trainval.npz', allow_pickle=False) as z:
                    Z = {k: z[k] for k in z.files}
                ms, mc = M['Q_SELECT'], M['Q_CAL']
                ws, wc = Q.weights(D['g'][ms]), Q.weights(D['g'][mc])
                direct = {c: (C.brier_direct(D['y'][ms], Z[c][ms], ws), C.logloss_direct(D['y'][ms], Z[c][ms], ws)) for c in Q.candidate_names(fam) if sel['eligible'][c]}
                add(f'{key}_selection_reproduced_independently', Q.select_rule(direct)[0] == sel['chosen'] == fz['family_winners'][key]['chosen'],
                    chosen=sel['chosen'], eligible=int(sum(sel['eligible'].values())))
                cal_ok, knot_ok, scaler_ok, imp_ok, dim_ok, conv, stop_ok = {}, {}, {}, {}, {}, {}, {}
                for cfg in Q.config_names(fam):
                    b = joblib.load(OUT / 'models' / coh / fam / f'{cfg}.joblib')
                    raw_c = Z[f'raw__{cfg}'][mc]
                    ref = K.fit_calibrators(raw_c, D['y'][mc], wc)
                    cal_ok[cfg] = bool(np.array_equal(ref['sigmoid'].coef_, b['calibrators']['sigmoid'].coef_)
                                       and np.array_equal(ref['sigmoid'].intercept_, b['calibrators']['sigmoid'].intercept_)
                                       and np.array_equal(ref['isotonic'].X_thresholds_, b['calibrators']['isotonic'].X_thresholds_)
                                       and np.array_equal(ref['isotonic'].y_thresholds_, b['calibrators']['isotonic'].y_thresholds_))
                    base = b['base']
                    if fam == 'pt':
                        from sklearn.preprocessing import SplineTransformer, StandardScaler
                        p, t = Xtr[:, names.index('p_pre_V')], Xtr[:, names.index('time_minutes')]
                        kp = SplineTransformer(**Q.SPLINE).fit(p.reshape(-1, 1)).bsplines_[0].t
                        kt = SplineTransformer(**Q.SPLINE).fit(t.reshape(-1, 1)).bsplines_[0].t
                        knot_ok[cfg] = bool(np.array_equal(kp, base.pt.p_spline.bsplines_[0].t) and np.array_equal(kt, base.pt.t_spline.bsplines_[0].t))
                        sc = StandardScaler().fit(base.pt.transform(p, t))
                        scaler_ok[cfg] = bool(np.array_equal(sc.mean_, base.scaler.mean_) and np.array_equal(sc.scale_, base.scaler.scale_))
                        dim_ok[cfg] = len(base.pt.names) == Q.PT_DIM == base.design_record['columns'] and base.design_record['rank_standardized'] == Q.PT_DIM
                    elif fam == 'logit':
                        from sklearn.preprocessing import StandardScaler
                        Xr = Xtr[:, [names.index(n) for n in ridge]]
                        imp_ok[cfg] = bool(np.array_equal(base.imputer.statistics_, np.median(Xr, axis=0)))
                        sc = StandardScaler().fit(Xr)
                        scaler_ok[cfg] = bool(np.allclose(sc.mean_, base.scaler.mean_, rtol=0, atol=1e-10) and int(base.scaler.n_samples_seen_) == len(Xr))
                    if fam in ('pt', 'logit'):
                        conv[cfg] = dict(eligible=base.eligible, attempts=[(a['max_iter'], a['n_iter'], a['converged']) for a in base.attempts])
                    else:
                        Xr = Xtr[:, [names.index(n) for n in ridge]]
                        imp_ok[cfg] = bool(np.array_equal(base.imputer.statistics_, np.median(Xr, axis=0)))
                        sr = base.stop_record
                        ok = sr['fit90_stop10_match_overlap'] == 0 and sr['full']['rows'] == int(M['TRAIN'].sum()) \
                            and sr['fit90']['rows'] + sr['stop10']['rows'] == int(M['TRAIN'].sum()) \
                            and sr['stop10']['membership_sha256'] == Q.membership_sha(gtr[stop]) and sr['fit90']['membership_sha256'] == Q.membership_sha(gtr[~stop])
                        for s_, r in sr['seeds'].items():
                            ok = ok and r['best_iteration'] == r['first_argmin_iteration'] == int(np.argmin(r['stop_curve'])) + 1 \
                                and r['final_n_estimators'] == r['final_num_trees'] == r['best_iteration'] \
                                and abs(r['stop_brier_recomputed_from_predict'] - r['best_stop_weighted_brier']) < 1e-9 \
                                and r['evaluator_metrics_seen'] == ['weighted_brier'] \
                                and (r['early_stopped'] and r['rounds_evaluated'] == r['best_iteration'] + Q.LGBM_PATIENCE or not r['early_stopped'])
                        stop_ok[cfg] = bool(ok)
                        conv[cfg] = {s_: dict(best=r['best_iteration'], rounds=r['rounds_evaluated'], cap_reached=r['cap_reached']) for s_, r in sr['seeds'].items()}
                    b = None
                add(f'{key}_calibrators_equal_q_cal_refit', all(cal_ok.values()), per_config=cal_ok)
                if fam == 'pt':
                    add(f'{key}_pt_knots_equal_train_refit', all(knot_ok.values()), per_config=knot_ok)
                    add(f'{key}_pt_scaler_equals_train_refit', all(scaler_ok.values()), per_config=scaler_ok)
                    add(f'{key}_pt_dimension_48_full_rank', all(dim_ok.values()), per_config=dim_ok)
                if fam == 'logit':
                    add(f'{key}_imputer_median_equals_train', all(imp_ok.values()), per_config=imp_ok)
                    add(f'{key}_scaler_equals_train_refit', all(scaler_ok.values()), per_config=scaler_ok)
                if fam in ('pt', 'logit'):
                    add(f'{key}_selected_candidate_converged', conv[sel['chosen_config']]['eligible'], convergence=conv)
                if fam == 'lgbm':
                    add(f'{key}_imputer_median_equals_train', all(imp_ok.values()), per_config=imp_ok)
                    add(f'{key}_stop_records_consistent', all(stop_ok.values()), per_config=stop_ok, iterations=conv)
                add(f'{key}_reload_identity_at_fit', all(all(v.values()) for v in sel['reload_identity'].values()))
                Z = None
            D = Xtr = None
        shared = set(stop_flags['T']) & set(stop_flags['N'])
        add('stop_allocation_same_for_T_and_N', all(stop_flags['T'][m] == stop_flags['N'][m] for m in shared), shared_matches=len(shared))
        add('freeze_prechecks_all_pass', all(all(v for n, v in ck.items() if not n.startswith('INFO_')) for ck in fz['pre_freeze_checks'].values()))
        add('overall_winner_rule_consistent', all(o['equals_rule_over_three_family_winners'] for o in fz['overall_winner'].values()),
            overall={c: o['chosen'] for c, o in fz['overall_winner'].items()})
        # ---------------------------------------------------------------- 5 evaluation reconstruction
        maxd = dict(auc=0.0, brier=0.0, logloss=0.0)
        boot_d, cells_checked, cell_ok, legacy_ok, counts_ok = 0.0, 0, True, True, True
        for set_name in Q.EVAL_SETS:
            for coh in Q.COHORTS:
                r = res['results'][set_name][coh]
                counts_ok &= r['rows'] == Q.EXPECTED_COUNTS[set_name][coh]
                lc = r['checks']['legacy']
                legacy_ok &= bool(lc['every_row_joined_once'] and lc['y_equal_parent_labels'] and lc['specialist_chosen_equals_parent_frozen'] and lc['parent_keys_unique'])
                with np.load(OUT / r['predictions_file'], allow_pickle=False) as z:
                    E = {k: z[k] for k in z.files}
                cells = Q.cell_masks(E['p_pre'], E['time_minutes'])
                cell_ok &= all(np.array_equal(cells[k], E[f'cell__{k}']) for k in cells)
                y, g = E['y'].astype(int), E['match'].astype(str)
                for c, m in cells.items():
                    if not m.any():
                        continue
                    w = Q.weights(g[m])
                    for name in res['named_models']:
                        p = E[f'named__{name}'][m]
                        rec = dict(auc=C.auc_direct(y[m], p, w), brier=C.brier_direct(y[m], p, w), logloss=C.logloss_direct(y[m], p, w))
                        ref = r['metrics_named'][c][name]
                        for k in maxd:
                            if rec[k] is not None and ref[k] is not None:
                                maxd[k] = max(maxd[k], abs(rec[k] - ref[k]))
                            elif (rec[k] is None) != (ref[k] is None):
                                maxd[k] = float('inf')
                        cells_checked += 1
                for c, bt in r['bootstrap'].items():
                    if bt.get('computed', True) is False:
                        continue
                    for name, mc in bt['model_ci'].items():
                        for k in ('brier', 'logloss', 'auc'):
                            if mc[k]['estimate'] is not None and r['metrics_named'][c][name][k] is not None:
                                boot_d = max(boot_d, abs(mc[k]['estimate'] - r['metrics_named'][c][name][k]))
                fam_ok = all(np.array_equal(E[f'named__{f}_winner'], E[f'cand__{fz["family_winners"][f"{f}_{coh}"]["chosen"]}']) for f in Q.FAMILIES)
                add(f'eval_{set_name}_{coh}_named_winners_equal_frozen_candidates', fam_ok and np.array_equal(E['named__overall_winner'], E[f'cand__{fz["overall_winner"][coh]["chosen"]}']))
                E = None
        RA = C.read_json(Q.CR / 'eval' / 'results_A.json')['results']
        rep = 0.0
        for set_name in Q.EVAL_SETS:
            for coh in Q.COHORTS:
                pc = RA[set_name]['h90']['cohorts'][coh]
                mine = res['results'][set_name][coh]['metrics_named']['all']
                for mine_key, parent_key in (('old_A_specialist', 'specialist'), ('old_pooled', 'pooled')):
                    for k in ('brier', 'logloss', 'auc'):
                        rep = max(rep, abs(mine[mine_key][k] - pc[parent_key][k]))
        add('legacy_columns_reproduce_parent_reported_metrics_lt_1e-12', rep < 1e-12, max_abs=rep)
        add('evaluation_counts_equal_specification', counts_ok)
        add('legacy_joins_complete_and_consistent', legacy_ok)
        add('cell_membership_recomputed_equal', cell_ok)
        add('independent_metric_reconstruction_lt_1e-10', max(maxd.values()) < 1e-10, max_abs=maxd, cells=cells_checked)
        add('bootstrap_point_estimates_equal_cell_metrics_lt_1e-10', boot_d < 1e-10, max_abs=boot_d)
        # ---------------------------------------------------------------- 6 reload frozen winners on MAIN TEST (fresh process, exact)
        F, Lb, Co, _ = Q.load_parent_set('MAIN_TEST', OUT, 'post-run reload identity of frozen winners')
        names = [str(x) for x in F['input_names']]
        for coh in Q.COHORTS:
            m = (Lb['valid_h90'] == 1) & (Co['cohort'] == Q.COHORT_CODE[coh])
            with np.load(OUT / 'eval' / 'predictions' / f'MAIN_TEST_h90_{coh}.npz', allow_pickle=False) as z:
                saved = {f: z[f'named__{f}_winner'] for f in Q.FAMILIES}
                keys_equal = np.array_equal(z['match'].astype(str), F['match'][m].astype(str)) and np.array_equal(z['s_ms'], F['s_ms'][m])
            ok = keys_equal
            for f in Q.FAMILIES:
                w = fz['family_winners'][f'{f}_{coh}']
                b = joblib.load(OUT / w['bundle'])
                ok = ok and bool(np.array_equal(Q.bundle_predict(b, w['calibration'], F['X_input'][m], names), saved[f]))
                b = None
            add(f'MAIN_TEST_{coh}_frozen_winners_reload_identical_in_fresh_process', ok)
        F = Lb = Co = None
        # ---------------------------------------------------------------- 7 status
        stat = C.read_json(OUT / 'status.json')
        required = ['driver_full', 'freeze', 'evaluate'] + [f'fit_{f}_{c}' for f in Q.FAMILIES for c in Q.COHORTS]
        add('required_stage_groups_complete', all(stat['groups'].get(g, {}).get('state') == 'complete' for g in required),
            states={g: stat['groups'].get(g, {}).get('state') for g in required})
        fails = []
        if (OUT / 'failures.jsonl').exists():
            fails = [json.loads(l) for l in (OUT / 'failures.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        details['retained_failures'] = [dict(time=f['time'], stage=f['stage'], error=f['error'][:300]) for f in fails]
        out = dict(version=Q.VERSION, role=Q.ROLE_TAG, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), n_checks=len(checks), checks=checks,
                   failed=[k for k, v in checks.items() if not v], details=details)
        sha = C.write_json(OUT / 'validation.json', out)
        st.update('complete' if not out['failed'] else 'failed', 'postrun_checks', n=len(checks), failed=out['failed'], validation_sha256=sha,
                  next_step='report')
        print(json.dumps(dict(n=len(checks), failed=out['failed'], sha256=sha)))
        return 0 if not out['failed'] else 1
    except Exception as exc:
        st.log(traceback.format_exc())
        Q.log_failure('postrun_checks', repr(exc), traceback=traceback.format_exc()[-3000:])
        C.write_json(Q.OUT / 'validation_partial_on_error.json', dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), error=repr(exc), checks=checks, details=details))
        st.update('failed', 'postrun_checks', error=repr(exc)[:1000], next_step='inspect')
        return 3


if __name__ == '__main__':
    sys.exit(main())
