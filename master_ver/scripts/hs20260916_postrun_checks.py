"""Stage V: post-run verification -> validation.json (independent recomputation where possible; real pass/fail), per horizon."""
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
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import ta20260916_common as T  # noqa: E402
import hs20260916_common as HS  # noqa: E402
import ta20260916_postrun_checks as TPC  # noqa: E402  (pure helpers: ts)

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

OUT = HS.OUT
BOOT_CELLS = ('all', 'B40', 'B45')
ts = TPC.ts


def bootstrap_cell_acceptance(bt, y, g):
    y = np.asarray(y).astype(int)
    g = np.asarray(g).astype(str)
    nm = int(len(np.unique(g)))
    classes = int(len(np.unique(y))) if len(y) else 0
    eligible = nm >= HS.SPARSE_MATCHES and classes == 2
    if not eligible:
        ok = bt.get('computed', True) is False and bool(bt.get('reason')) and not bt.get('pairs')
        return ([] if ok else [f'ineligible cell (matches={nm}, classes={classes}) lacks an explicit computed=False skip']), False
    if bt.get('computed', True) is False:
        return [f'eligible cell skipped: {bt.get("reason")}'], True
    P = []
    reps, deg = bt.get('replicates'), bt.get('degenerate_single_class_replicates')
    if reps != HS.BOOT_REPS or bt.get('seed') != HS.BOOT_SEED:
        P.append(f'replicates/seed {reps}/{bt.get("seed")}')
    if bt.get('matches') != nm or bt.get('rows') != len(y):
        P.append('matches/rows differ from the cell')
    if not isinstance(deg, int) or isinstance(deg, bool) or not isinstance(reps, int) or not 0 <= deg < reps:
        return P + ['degenerate count incoherent'], True
    if set(bt.get('model_ci', {})) != set(HS.BOOT_MODELS):
        P.append('model_ci models differ')
    if [(p.get('a'), p.get('b'), p.get('label')) for p in bt.get('pairs', [])] != [tuple(c) for c in HS.CONTRASTS]:
        P.append('contrasts differ from the planned six')

    def interval(rec, expected, where):
        c = rec.get('ci95') if isinstance(rec, dict) else None
        if not isinstance(rec, dict) or rec.get('finite_replicates') != expected or rec.get('estimate') is None or not np.isfinite(rec['estimate']):
            P.append(f'{where}: replicates/estimate')
        if not c or len(c) != 2 or not all(v is not None and np.isfinite(v) for v in c) or c[0] > c[1]:
            P.append(f'{where}: interval')
    for m, mc in bt.get('model_ci', {}).items():
        for k in ('brier', 'logloss', 'auc'):
            interval(mc.get(k), reps - deg if k == 'auc' else reps, f'model_ci {m} {k}')
    for p in bt.get('pairs', []):
        for k in ('brier', 'logloss', 'auc'):
            interval(p.get('a_minus_b', {}).get(k), reps - deg if k == 'auc' else reps, f'{p.get("a")}-{p.get("b")} {k}')
    return P, True


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    HS.log_command()
    st = HS.Status('postrun_checks')
    checks, info = {}, {}

    def add(name, ok, **extra):
        checks[name] = bool(ok)
        if extra:
            info[name] = C.clean(extra)
        st.log(f'{"PASS" if ok else "FAIL"} {name}')

    try:
        proto = C.read_json(OUT / 'protocol.json')
        fz = C.read_json(HS.frozen_path(OUT))
        res = C.read_json(OUT / 'eval' / 'results.json')
        ct = C.read_json(OUT / 'contract_tests' / 'result.json')
        schema = C.read_json(HS.FC / 'q_pre_only_schema.json')
        ridge = list(schema['predictor_sets']['ridge'])
        reg = proto['fixed_configurations']['registry']
        frozen_t = ts(fz['frozen_at'])
        horizons = fz['horizons']
        # 1 parents
        diff = C.read_json(OUT / 'integrity' / 'snapshot_diff.json')
        for k, v in diff.items():
            if isinstance(v, dict):
                add(f'parent_unchanged_{k}', v['equal'], changed=v.get('changed'), removed=v.get('removed'), added=v.get('added'), checked=v.get('checked'))
        add('parent_expected_hashes_protocol_equal_now', all(C.sha256_file(HS.ROOT / p) == sha for p, sha in proto['parents']['expected_parent_file_sha256'].items()),
            files=len(proto['parents']['expected_parent_file_sha256']))
        add('spec_unchanged_since_protocol', C.sha256_file(HS.SPEC) == proto['spec_sha256'])
        pr = fz['parent_references']
        add('parent_frozen_manifests_unchanged', C.sha256_file(Q.frozen_path(HS.IQ)) == pr['iq_frozen_manifest_sha256'] and C.sha256_file(T.frozen_path(HS.TA)) == pr['ta_frozen_manifest_sha256']
            and C.sha256_file(HS.CR / 'frozen_manifest.json') == pr['cohort_role_frozen_manifest_sha256'])
        add('legacy_prediction_files_unchanged', all(C.sha256_file(HS.legacy_path(k.rsplit('_h', 1)[0], int(k.rsplit('_h', 1)[1].split('_')[0]), k.rsplit('_', 1)[1])) == sha
                                                     for k, sha in pr['legacy_predictions_sha256'].items()), files=len(pr['legacy_predictions_sha256']))
        reg_now, _ = HS.h90_registry()
        add('registry_equals_parent_h90_winners', reg_now == reg, registry=reg)
        # 2 ordering
        snap_t = ts(C.read_json(OUT / 'integrity' / 'snapshot_before.json')['taken_at'])
        runs = [C.read_json(p) for p in sorted(OUT.glob('contract_tests/result_run*.json'), key=lambda p: int(p.stem.replace('result_run', '')))]
        smoke_fz = C.read_json(HS.frozen_path(HS.SMOKE))
        smoke_eval = C.read_json(HS.SMOKE / 'eval' / 'results.json')
        keys = [f'h{h}_{f}_{c}' for h in horizons for c in HS.COHORTS for f in HS.FAMILIES]
        sels = {k: C.read_json(OUT / 'selection' / f'{k}.json') for k in keys}
        first_fit = min(ts(C.read_json(OUT / 'status' / f'fit_{k}.json')['history'][0]['at']) for k in keys)
        latest_pass = max(ts(r['at']) for r in runs if r['passed'])
        add('order_snapshot_tests_protocol_smoke_latest_tests_before_full_fits',
            snap_t <= ts(runs[0]['at']) <= ts(proto['written_at']) <= ts(smoke_fz['frozen_at']) <= ts(smoke_eval['evaluated_at']) and latest_pass <= first_fit,
            first_full_fit=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first_fit)), latest_passing_tests=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_pass)))
        add('all_selections_before_freeze', max(ts(s['written_at']) for s in sels.values()) <= frozen_t, frozen_at=fz['frozen_at'])
        add('contract_tests_passed_no_skips', ct['passed'] and ct['failed_count'] == 0 and ct['skipped_count'] == 0 and ct['error_count'] == 0, passed=ct['passed_count'], run=ct['run'])
        add('final_source_hashes_equal_frozen_for_fit_freeze_evaluate_common',
            all(C.sha256_file(HS.ROOT / p) == sha for p, sha in fz['source_sha256'].items()
                if p.split('/')[-1] in ('hs20260916_common.py', 'hs20260916_fit.py', 'hs20260916_freeze.py', 'hs20260916_evaluate.py', 'iq20260915_common.py',
                                        'ta20260916_common.py', 'fc20260915_common.py', 'cr20260915_common.py')))
        acc = [json.loads(l) for l in (OUT / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        sealed = [a for a in acc if a['sealed']]
        add('sealed_accesses_only_after_freeze', bool(sealed) and all(a['frozen_manifest_exists'] and ts(a['time']) >= frozen_t for a in sealed), n_sealed=len(sealed))
        add('no_sealed_access_before_freeze_recorded_in_manifest', fz['sealed_accesses_before_freeze'] == 0)
        sacc = [json.loads(l) for l in (HS.SMOKE / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        add('smoke_read_train_only', all(a['set'] == 'MAIN_TRAIN' for a in sacc) and not any(HS.SMOKE.rglob('*TEST*')) and not any(HS.SMOKE.rglob('*EXT_*')))
        add('evaluation_after_freeze', ts(res['evaluated_at']) >= frozen_t and res['complete'])
        now = {p: C.sha256_file(OUT / p) == sha for p, sha in fz['frozen_files_sha256'].items()}
        add('frozen_artifacts_unchanged_now', all(now.values()), files=len(now))
        hb, ha = C.read_json(OUT / 'eval' / 'hashes_before.json'), C.read_json(OUT / 'eval' / 'hashes_after.json')
        add('frozen_artifacts_unchanged_through_evaluation', all(hb['results'].values()) and all(ha['results'].values()) and res['frozen_hashes_unchanged'])
        add('freeze_prechecks_all_pass', all(all(ck.values()) for ck in fz['pre_freeze_checks'].values()))
        # 3 fit records per horizon x cohort
        vman = C.read_json(HS.FC / 'v_models_manifest.json')
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        for h in horizons:
            for coh in HS.COHORTS:
                D = HS.load_trainval(OUT, h, smoke=False, cohort=coh)
                names = D['names']
                M = {k: D['role'] == k for k in ('TRAIN', 'Q_CAL', 'Q_SELECT')}
                g, y, X = D['g'], D['y'], D['X']
                add(f'h{h}_{coh}_counts_equal_specification', all(int(M[k].sum()) == HS.EXPECTED_COUNTS[k][coh] for k in M))
                trainfold = np.char.startswith(D['sr_true'], 'fold')
                prov = HS.oof_provenance(g[trainfold], D['sr_true'][trainfold], D['adapter_id'][trainfold], D['adapter_sha256'][trainfold], vman)
                add(f'h{h}_{coh}_train_rows_use_own_heldout_fold_oof_V', prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold'])
                ridge_cols = HS.input_columns(names, 'logit', ridge)
                Xtr_r = X[M['TRAIN']][:, ridge_cols]
                med = np.median(Xtr_r, axis=0)
                ref_scaler = StandardScaler().fit(SimpleImputer(strategy='median').fit(Xtr_r).transform(Xtr_r))
                ws, wc = HS.weights(g[M['Q_SELECT']]), HS.weights(g[M['Q_CAL']])
                for fam in HS.FAMILIES:
                    key = f'h{h}_{fam}_{coh}'
                    sel = sels[key]
                    cfg = sel['config']
                    with np.load(OUT / 'predictions' / f'{key}_trainval.npz', allow_pickle=False) as z:
                        Z = {k: z[k] for k in z.files}
                    ok = sel['split_counts']['TRAIN']['rows'] == HS.EXPECTED_COUNTS['TRAIN'][coh] and cfg == reg[coh][fam]
                    ok &= all(abs(w['per_match_total_min'] - w['per_match_total_max']) < 1e-9 and abs(w['weight_mean'] - 1) < 1e-9 for w in sel['weights'].values())
                    ok &= all(sel['weights'][k]['membership_sha256'] == HS.membership_sha(g[M[k]]) for k in M)
                    ok &= np.array_equal(Z['match'].astype(str), g) and np.array_equal(Z['s_ms'], D['s_ms']) and np.array_equal(Z['y'], y)
                    direct = {c: (C.brier_direct(y[M['Q_SELECT']], Z[c][M['Q_SELECT']], ws), C.logloss_direct(y[M['Q_SELECT']], Z[c][M['Q_SELECT']], ws)) for c in HS.candidate_names(cfg) if sel['eligible'][c]}
                    ok &= HS.select_rule(direct)[0] == sel['chosen'] == fz['family_winners'][key]['chosen']
                    b = joblib.load(OUT / fz['family_winners'][key]['bundle'])
                    cals = K.fit_calibrators(Z[f'raw__{cfg}'][M['Q_CAL']], y[M['Q_CAL']], wc)
                    ok &= all(np.array_equal(K.apply_calibration(cal, cals[cal], Z[f'raw__{cfg}'][M[r]]), Z[f'{cfg}__{cal}'][M[r]]) for cal in ('sigmoid', 'isotonic') for r in ('Q_CAL', 'Q_SELECT'))
                    ok &= np.array_equal(Z[f'{cfg}__raw'], Z[f'raw__{cfg}'])
                    base = b['base']
                    if fam == 'logit':
                        ok &= np.array_equal(base.imputer.statistics_, med) and np.array_equal(base.scaler.mean_, ref_scaler.mean_) and np.array_equal(base.scaler.scale_, ref_scaler.scale_)
                    elif fam == 'pt':
                        ok &= base.design_record['columns'] == Q.PT_DIM and base.pt.fit_rows == int(M['TRAIN'].sum())
                    elif fam == 'lgbm':
                        ok &= np.array_equal(base.imputer.statistics_, med)
                    else:
                        ok &= np.array_equal(base.imputer.statistics_, med) and np.array_equal(base.scaler.mean_, ref_scaler.mean_) and np.array_equal(base.scaler.scale_, ref_scaler.scale_)
                    if fam in HS.SEEDED:
                        ok &= np.array_equal(Z[f'raw__{cfg}'], np.mean([Z[f'seedraw__{cfg}__{s}'] for s in HS.SEEDS[fam]], axis=0))
                        sr = C.read_json(OUT / 'internal_stop' / f'{key}.json')['stop_record']
                        ok &= sr['fit90_stop10_match_overlap'] == 0 and sr['full']['rows'] == int(M['TRAIN'].sum())
                        for s, r in sr['seeds'].items():
                            curve = r['stop_curve']
                            if fam == 'lgbm':
                                ok &= r['best_iteration'] == r['first_argmin_iteration'] == int(np.argmin(curve)) + 1 == r['final_n_estimators']
                            else:
                                ok &= r['best_epoch'] == r['first_argmin_epoch'] == int(np.argmin(curve)) + 1 == r['refit_epochs']
                    ok &= all(sel['reload_identity'][cfg].values())
                    b = Z = None
                    add(f'{key}_fit_selection_calibration_preprocessing_stop_and_reload_verified', ok, chosen=sel['chosen'])
                D = None
        # 4 evaluation reconstruction
        RA = C.read_json(HS.CR / 'eval' / 'results_A.json')['results']
        maxd, boot_d, cells_checked, cell_ok, legacy_ok, counts_ok, rep = dict(auc=0.0, brier=0.0, logloss=0.0), 0.0, 0, True, True, True, 0.0
        boot_problems, boot_counts = [], dict(eligible=0, ineligible_explicit_skip=0)
        for h in horizons:
            for set_name in HS.EVAL_SETS:
                for coh in HS.COHORTS:
                    r = res['results'][f'h{h}'][set_name][coh]
                    counts_ok &= r['rows'] == HS.EXPECTED_COUNTS[set_name][coh]
                    lc = r['checks']['legacy']
                    legacy_ok &= bool(lc['every_row_joined_once'] and lc['y_equal_parent_labels'] and lc['specialist_chosen_equals_parent_frozen'] and lc['parent_keys_unique'] and lc['file_sha256_equals_frozen'])
                    with np.load(OUT / r['predictions_file'], allow_pickle=False) as z:
                        E = {k: z[k] for k in z.files}
                    cells = HS.cell_masks(E['p_pre'], E['time_minutes'])
                    cell_ok &= all(np.array_equal(cells[k], E[f'cell__{k}']) for k in cells)
                    y, g = E['y'].astype(int), E['match'].astype(str)
                    for c, m in cells.items():
                        if not m.any():
                            continue
                        w = HS.weights(g[m])
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
                    for c in BOOT_CELLS:
                        probs, elig = bootstrap_cell_acceptance(r['bootstrap'][c], y[E[f'cell__{c}']], g[E[f'cell__{c}']])
                        boot_counts['eligible' if elig else 'ineligible_explicit_skip'] += 1
                        boot_problems += [f'h{h} {set_name} {coh} {c}: {p}' for p in probs]
                    for c, bt in r['bootstrap'].items():
                        if bt.get('computed', True) is False:
                            continue
                        for name, mc in bt['model_ci'].items():
                            for k in ('brier', 'logloss', 'auc'):
                                if mc[k]['estimate'] is not None and r['metrics_named'][c][name][k] is not None:
                                    boot_d = max(boot_d, abs(mc[k]['estimate'] - r['metrics_named'][c][name][k]))
                    fam_ok = all(np.array_equal(E[f'named__{f}_winner'], E[f'cand__{f}:{fz["family_winners"][f"h{h}_{f}_{coh}"]["chosen"]}']) for f in HS.FAMILIES)
                    add(f'eval_h{h}_{set_name}_{coh}_named_winners_equal_frozen_candidates', fam_ok and np.array_equal(E['named__overall_winner'], E[f'cand__{fz["overall_winner"][f"h{h}_{coh}"]["chosen"]}']))
                    pc = RA[set_name][f'h{h}']['cohorts'][coh]
                    mine = r['metrics_named']['all']
                    for mine_key, parent_key in (('old_A_specialist', 'specialist'), ('old_pooled', 'pooled')):
                        for k in ('brier', 'logloss', 'auc'):
                            if mine[mine_key][k] is not None and pc[parent_key][k] is not None:
                                rep = max(rep, abs(mine[mine_key][k] - pc[parent_key][k]))
                    E = None
        add('legacy_columns_reproduce_parent_reported_metrics_lt_1e-12', rep < 1e-12, max_abs=rep)
        add('evaluation_counts_equal_specification', counts_ok)
        add('legacy_joins_complete_and_consistent', legacy_ok)
        add('cell_membership_recomputed_equal', cell_ok)
        add('independent_metric_reconstruction_lt_1e-10', max(maxd.values()) < 1e-10, max_abs=maxd, cells=cells_checked)
        add('bootstrap_point_estimates_equal_cell_metrics_lt_1e-10', boot_d < 1e-10, max_abs=boot_d)
        add('bootstrap_acceptance_all_B40_B45_cells', not boot_problems and sum(boot_counts.values()) == len(horizons) * len(HS.EVAL_SETS) * len(HS.COHORTS) * len(BOOT_CELLS),
            counts=boot_counts, problems=boot_problems[:50])
        # 5 fresh-process reload of the winners on MAIN TEST
        for h in horizons:
            F, Lb, Co, _ = HS.load_parent_set('MAIN_TEST', OUT, f'post-run reload identity h{h}', h)
            names = [str(x) for x in F['input_names']]
            for coh in HS.COHORTS:
                m = (Lb[f'valid_h{h}'] == 1) & (Co['cohort'] == HS.COHORT_CODE[coh])
                with np.load(OUT / 'eval' / 'predictions' / f'MAIN_TEST_h{h}_{coh}.npz', allow_pickle=False) as z:
                    saved = {f: z[f'named__{f}_winner'] for f in HS.FAMILIES}
                    ok = np.array_equal(z['match'].astype(str), F['match'][m].astype(str)) and np.array_equal(z['s_ms'], F['s_ms'][m])
                for f in HS.FAMILIES:
                    wn = fz['family_winners'][f'h{h}_{f}_{coh}']
                    b = joblib.load(OUT / wn['bundle'])
                    ok = ok and bool(np.array_equal(HS.bundle_predict(b, wn['calibration'], F['X_input'][m], names), saved[f]))
                    b = None
                add(f'MAIN_TEST_h{h}_{coh}_frozen_winners_reload_identical_in_fresh_process', ok)
            F = Lb = Co = None
        stat = C.read_json(OUT / 'status.json')
        required = ['driver_full', 'freeze', 'evaluate'] + [f'fit_{k}' for k in keys]
        add('required_stage_groups_complete', all(stat['groups'].get(g, {}).get('state') == 'complete' for g in required))
        add('no_bytecode_written_in_parent_scripts_tests', diff['pycache_scripts']['equal'] and diff['pycache_tests']['equal'])
        failed = [k for k, v in checks.items() if not v]
        C.write_json(OUT / 'validation.json', dict(role=HS.ROLE_TAG, version=HS.VERSION, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), n_checks=len(checks), failed=failed,
                                                    passed=len(checks) - len(failed), checks=checks, info=info, frozen_manifest_sha256=C.sha256_file(HS.frozen_path(OUT)),
                                                    results_sha256=C.sha256_file(OUT / 'eval' / 'results.json')))
        st.update('complete' if not failed else 'failed', 'validated', n_checks=len(checks), failed=failed[:20], next_step='report')
        print(f'checks {len(checks)} failed {len(failed)}: {failed}')
        return 0 if not failed else 1
    except Exception as exc:
        st.log(traceback.format_exc())
        HS.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'validate', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
