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
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import dd20260916_common as DD  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

OUT = DD.OUT
BOOT_CELLS = ('all', 'B40', 'B45')
ts = DD.ts


def bootstrap_cell_acceptance(bt, y, g):
    y = np.asarray(y).astype(int)
    g = np.asarray(g).astype(str)
    nm = int(len(np.unique(g)))
    classes = int(len(np.unique(y))) if len(y) else 0
    eligible = nm >= DD.SPARSE_MATCHES and classes == 2
    if not eligible:
        ok = bt.get('computed', True) is False and bool(bt.get('reason')) and not bt.get('pairs')
        return ([] if ok else [f'ineligible cell (matches={nm}, classes={classes}) lacks an explicit skip']), False
    if bt.get('computed', True) is False:
        return [f'eligible cell skipped: {bt.get("reason")}'], True
    P = []
    reps, deg = bt.get('replicates'), bt.get('degenerate_single_class_replicates')
    if reps != DD.BOOT_REPS or bt.get('seed') != DD.BOOT_SEED:
        P.append('replicates/seed')
    if bt.get('matches') != nm or bt.get('rows') != len(y):
        P.append('matches/rows differ from the cell')
    if set(bt.get('model_ci', {})) != set(DD.BOOT_MODELS):
        P.append('model_ci models differ')
    if [(p.get('a'), p.get('b'), p.get('label')) for p in bt.get('pairs', [])] != [tuple(c) for c in DD.CONTRASTS]:
        P.append('contrasts differ from the planned four')
    for m, mc in bt.get('model_ci', {}).items():
        for k in ('brier', 'logloss', 'auc'):
            c = mc[k].get('ci95')
            if mc[k].get('estimate') is None or not c or c[0] > c[1]:
                P.append(f'model_ci {m} {k}')
    for p in bt.get('pairs', []):
        for k in ('brier', 'logloss', 'auc'):
            c = p['a_minus_b'][k].get('ci95')
            if p['a_minus_b'][k].get('estimate') is None or not c or c[0] > c[1]:
                P.append(f'{p["a"]}-{p["b"]} {k}')
    return P, True


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    DD.log_command()
    st = DD.Status('postrun_checks')
    checks, info = {}, {}

    def add(name, ok, **extra):
        checks[name] = bool(ok)
        if extra:
            info[name] = C.clean(extra)
        st.log(f'{"PASS" if ok else "FAIL"} {name}')

    try:
        proto = C.read_json(OUT / 'protocol.json')
        fz = C.read_json(DD.frozen_path(OUT))
        res = C.read_json(OUT / 'eval' / 'results.json')
        ct = C.read_json(OUT / 'contract_tests' / 'result.json')
        census = C.read_json(OUT / 'census.json')
        schema = C.read_json(DD.FC / 'q_pre_only_schema.json')
        ridge = list(schema['predictor_sets']['ridge'])
        reg = proto['fixed_configurations']['registry']
        frozen_t = ts(fz['frozen_at'])
        # 1 parents and pinned inputs
        diff = C.read_json(OUT / 'integrity' / 'snapshot_diff.json')
        for k, v in diff.items():
            if isinstance(v, dict):
                add(f'parent_unchanged_{k}', v['equal'], changed=v.get('changed'), removed=v.get('removed'), added=v.get('added'), checked=v.get('checked'))
        add('parent_expected_hashes_protocol_equal_now', all(C.sha256_file(DD.ROOT / p) == sha for p, sha in proto['parents']['expected_parent_file_sha256'].items()), files=len(proto['parents']['expected_parent_file_sha256']))
        add('spec_unchanged_since_protocol', C.sha256_file(DD.SPEC) == proto['spec_sha256'])
        add('boundary_specs_unchanged', C.sha256_file(DD.DEV_SPEC) == proto['definitions']['dev']['spec']['sha256'] and C.sha256_file(DD.POOLED_SPEC) == proto['definitions']['frozen']['spec']['sha256'])
        pr = fz['parent_references']
        add('parent_frozen_manifests_unchanged', C.sha256_file(Q.frozen_path(DD.IQ)) == pr['iq_frozen_manifest_sha256'] and C.sha256_file(DD.CR / 'frozen_manifest.json') == pr['cohort_role_frozen_manifest_sha256'])
        add('parent_prediction_and_bundle_files_unchanged', all(C.sha256_file(DD.IQ / 'eval' / 'predictions' / f'{k.rsplit("_", 1)[0]}_h{DD.H}_{k.rsplit("_", 1)[1]}.npz') == sha for k, sha in pr['parent_predictions_sha256'].items())
            and all(C.sha256_file(DD.IQ / v['bundle']) == v['bundle_sha256'] for v in pr['iq_winners'].values()))
        reg_now, _, _ = DD.registry()
        add('registry_equals_iq_h90_winners', reg_now == reg)
        # 2 detector reproduction, fixture, census consistency
        fcheck = C.read_json(OUT / 'redetect' / 'frozen_check.json')
        add('frozen_definition_reproduces_parent_exposures', fcheck['mismatching_matches'] == 0 and fcheck['matches_compared'] >= DD.FROZEN_CHECK_MATCHES and fcheck['errors'] == 0
            and fcheck['cluster_counts_unknown_rows'] == 0, matches=fcheck['matches_compared'], rows=fcheck['exposure_rows_parent'])
        fx = C.read_json(OUT / 'rebuild' / 'fixture_check.json')
        add('rebuild_fixture_reproduces_parent_rows_bitwise', fx['checks']['all_equal'] and fx['matches'] >= DD.FIXTURE_MATCHES, checks=fx['checks'])
        add('dev_redetection_complete_no_errors', all(C.read_json(OUT / 'redetect' / f'{s}_dev.json')['errors'] == 0 and C.read_json(OUT / 'redetect' / f'{s}_dev.json')['loaded'] == C.read_json(OUT / 'redetect' / f'{s}_dev.json')['matches']
                                                   for s in ['MAIN'] + [f'EXT_{x}' for x in DD.EXT]))
        add('census_redetect_files_unchanged', all(C.sha256_file(OUT / 'redetect' / n) == sha for n, sha in census['redetect_files_sha256'].items()))
        cons, no_expo = True, {}
        dev_main = DD.read_redetect('MAIN', 'dev')
        for s in DD.ALL_SETS:
            rb = C.read_json(OUT / 'rebuild' / f'{s}.json')
            ch = C.read_json(OUT / 'changed' / f'{s}.json')
            cs = census['sets'][s]
            cons &= rb['changed_matches'] == cs['matches_changed'] == len(ch['matches']) and rb['rebuilt_rows'] == sum(len(v) for v in ch['rows'].values())
            cons &= rb['dev_rows'] == cs['rows_dev'] and rb['dev_rows'] == rb['parent_rows_copied'] + rb['rebuilt_rows']
            # matches absent from the dev csv are matches without any engagement: they must be exactly the matches without parent exposures
            par, _ = DD.parent_exposures(s)
            dev = dev_main if s.startswith('MAIN') else DD.read_redetect(s, 'dev')
            ids = [r[0] for r in DD.set_matches(s)]
            no_dev, no_par = {m for m in ids if m not in dev}, {m for m in ids if m not in par}
            # a match without exposures on one side only lost (or gained) all its engagements under the dev definition: it must be a changed match
            no_expo[s] = dict(no_exposures_dev=len(no_dev), no_exposures_parent=len(no_par), lost_all=len(no_dev - no_par), gained_all=len(no_par - no_dev),
                              one_sided_all_in_changed=(no_dev ^ no_par) <= set(ch['matches']))
            cons &= (no_dev ^ no_par) <= set(ch['matches']) and len(no_dev) == cs['matches_missing_in_dev']
        add('census_changed_lists_and_rebuild_summaries_consistent', cons, matches_without_exposures=no_expo)
        # 3 dev population files: parent-source rows equal the parent rows; rebuilt keys = dev exposures of changed matches
        pop_ok, pop_info = True, {}
        for s in DD.ALL_SETS:
            F, Lb, Co = DD.load_dev_set(s, OUT, f'post-run population check {s}')
            PF, PL, PC = DD.load_parent_set(s, OUT, f'post-run population check {s} (parent)')
            ch = C.read_json(OUT / 'changed' / f'{s}.json')
            src = F['row_source'].astype(str)
            pk = {(m, int(v)): i for i, (m, v) in enumerate(zip(PF['match'].astype(str).tolist(), PF['s_ms'].tolist()))}
            par_rows = np.flatnonzero(src == 'parent')
            ix = np.asarray([pk.get((m, int(v)), -1) for m, v in zip(F['match'][par_rows].astype(str).tolist(), F['s_ms'][par_rows].tolist())], dtype=np.int64)
            ok = bool(np.all(ix >= 0)) and not np.isin(F['match'][par_rows].astype(str), ch['matches']).any()
            ok = ok and np.array_equal(F['X_input'][par_rows], PF['X_input'][ix], equal_nan=True) and np.array_equal(Lb['p_pre'][par_rows], PL['p_pre'][ix], equal_nan=True)
            ok = ok and np.array_equal(Lb['Y_h90'][par_rows].astype(int), PL['Y_h90'][ix].astype(int)) and np.array_equal(Lb['valid_h90'][par_rows].astype(int), PL['valid_h90'][ix].astype(int))
            ok = ok and np.array_equal(Co['cohort'][par_rows].astype(int), PC['cohort'][ix].astype(int)) and np.array_equal(Co['n_min'][par_rows].astype(int), PC['n_min'][ix].astype(int))
            unchanged = ~np.isin(PF['match'].astype(str), ch['matches'])
            ok = ok and int(unchanged.sum()) == len(par_rows)
            reb = np.flatnonzero(src == 'rebuilt')
            keys_reb = set(zip(F['match'][reb].astype(str).tolist(), [int(v) for v in F['s_ms'][reb].tolist()]))
            keys_dev = set((m, int(r['s'])) for m, rows in ch['rows'].items() for r in rows)
            ok = ok and keys_reb == keys_dev and np.all(np.isin(F['match'][reb].astype(str), ch['matches']))
            pop_info[s] = dict(parent_rows=len(par_rows), rebuilt_rows=len(reb), ok=bool(ok))
            pop_ok &= bool(ok)
            F = Lb = Co = PF = PL = PC = None
        add('dev_population_parent_rows_bitwise_and_rebuilt_keys_exact', pop_ok, sets=pop_info)
        # 4 ordering and sealing
        snap_t = ts(C.read_json(OUT / 'integrity' / 'snapshot_before.json')['taken_at'])
        runs = [C.read_json(p) for p in sorted(OUT.glob('contract_tests/result_run*.json'), key=lambda p: int(p.stem.replace('result_run', '')))]
        latest_pass = max(ts(r['at']) for r in runs if r['passed'])
        fit_hist = C.read_json(OUT / 'status' / 'fit_freeze.json')['history']
        first_fit = ts(fit_hist[0]['at'])
        add('order_snapshot_tests_protocol_detect_census_fixture_rebuild_tests_fit_freeze',
            snap_t <= ts(runs[0]['at']) <= ts(proto['written_at']) <= ts(fcheck['written_at']) <= ts(census['written_at']) <= ts(fx['written_at'])
            <= ts(C.read_json(OUT / 'rebuild' / 'MAIN_VALIDATION.json')['written_at']) <= latest_pass <= first_fit <= frozen_t)
        add('sealed_rebuilds_after_freeze', all(ts(C.read_json(OUT / 'rebuild' / f'{s}.json')['written_at']) >= frozen_t for s in DD.EVAL_SETS))
        add('contract_tests_passed_no_skips', ct['passed'] and ct['failed_count'] == 0 and ct['skipped_count'] == 0 and ct['error_count'] == 0, passed=ct['passed_count'], run=ct['run'])
        add('final_source_hashes_equal_frozen', all(C.sha256_file(DD.ROOT / p) == sha for p, sha in fz['source_sha256'].items()
                                                   if p.split('/')[-1] in ('dd20260916_common.py', 'dd20260916_rebuild.py', 'dd20260916_fit_freeze.py', 'dd20260916_evaluate.py', 'dd20260916_redetect.py',
                                                                           'dd20260916_census.py', 'iq20260915_common.py', 'fc20260915_common.py', 'cr20260915_common.py', 'fc20260915_extract.py', 'measure_postkill_objective_delay.py')))
        acc = [json.loads(l) for l in (OUT / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        sealed = [a for a in acc if a['sealed']]
        add('sealed_accesses_only_after_freeze', bool(sealed) and all(a['frozen_manifest_exists'] and ts(a['time']) >= frozen_t for a in sealed), n_sealed=len(sealed))
        add('no_sealed_access_before_freeze_recorded_in_manifest', fz['sealed_accesses_before_freeze'] == 0)
        add('evaluation_after_freeze', ts(res['evaluated_at']) >= frozen_t and res['complete'])
        now = {p: C.sha256_file(OUT / p) == sha for p, sha in fz['frozen_files_sha256'].items()}
        add('frozen_artifacts_unchanged_now', all(now.values()), files=len(now))
        hb, ha = C.read_json(OUT / 'eval' / 'hashes_before.json'), C.read_json(OUT / 'eval' / 'hashes_after.json')
        add('frozen_artifacts_unchanged_through_evaluation', all(hb['results'].values()) and all(ha['results'].values()) and res['frozen_hashes_unchanged'])
        add('freeze_prechecks_all_pass', all(all(ck.values()) for ck in fz['pre_freeze_checks'].values()))
        # 5 fit records
        vman = C.read_json(DD.FC / 'v_models_manifest.json')
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        for coh in DD.COHORTS:
            D = DD.load_trainval(OUT, coh)
            names = D['names']
            M = {k: D['role'] == k for k in ('TRAIN', 'Q_CAL', 'Q_SELECT')}
            g, y = D['g'], D['y']
            trainfold = np.char.startswith(D['sr_true'], 'fold')
            prov = DD.oof_provenance(g[trainfold], D['sr_true'][trainfold], D['adapter_id'][trainfold], D['adapter_sha256'][trainfold], vman)
            add(f'{coh}_train_rows_use_own_heldout_fold_oof_V', prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold'])
            ridge_ix = [names.index(n) for n in ridge]
            Xtr = np.ascontiguousarray(D['X'][M['TRAIN']][:, ridge_ix])
            med = np.median(Xtr, axis=0)
            ref_scaler = StandardScaler().fit(SimpleImputer(strategy='median').fit(Xtr).transform(Xtr))
            ws, wc = DD.weights(g[M['Q_SELECT']]), DD.weights(g[M['Q_CAL']])
            for fam in DD.FAMILIES:
                key = f'{fam}_{coh}'
                sel = C.read_json(OUT / 'selection' / f'{key}.json')
                cfg = sel['config']
                with np.load(OUT / 'predictions' / f'{key}_trainval.npz', allow_pickle=False) as z:
                    Z = {k: z[k] for k in z.files}
                ok = cfg == reg[coh][fam] and sel['split_counts']['TRAIN']['rows'] == int(M['TRAIN'].sum())
                ok &= all(abs(w['per_match_total_min'] - w['per_match_total_max']) < 1e-9 and abs(w['weight_mean'] - 1) < 1e-9 for w in sel['weights'].values())
                ok &= all(sel['weights'][k]['membership_sha256'] == DD.membership_sha(g[M[k]]) for k in M)
                ok &= np.array_equal(Z['match'].astype(str), g) and np.array_equal(Z['s_ms'], D['s_ms']) and np.array_equal(Z['y'], y)
                direct = {c: (C.brier_direct(y[M['Q_SELECT']], Z[c][M['Q_SELECT']], ws), C.logloss_direct(y[M['Q_SELECT']], Z[c][M['Q_SELECT']], ws)) for c in DD.candidate_names(cfg) if sel['eligible'][c]}
                ok &= DD.select_rule(direct)[0] == sel['chosen'] == fz['family_winners'][key]['chosen']
                b = joblib.load(OUT / fz['family_winners'][key]['bundle'])
                cals = K.fit_calibrators(Z[f'raw__{cfg}'][M['Q_CAL']], y[M['Q_CAL']], wc)
                ok &= all(np.array_equal(K.apply_calibration(cal, cals[cal], Z[f'raw__{cfg}'][M[r]]), Z[f'{cfg}__{cal}'][M[r]]) for cal in ('sigmoid', 'isotonic') for r in ('Q_CAL', 'Q_SELECT'))
                ok &= np.array_equal(Z[f'{cfg}__raw'], Z[f'raw__{cfg}'])
                base = b['base']
                if fam == 'logit':
                    ok &= np.array_equal(base.imputer.statistics_, med) and np.array_equal(base.scaler.mean_, ref_scaler.mean_) and np.array_equal(base.scaler.scale_, ref_scaler.scale_)
                else:
                    ok &= np.array_equal(base.imputer.statistics_, med) and np.array_equal(Z[f'raw__{cfg}'], np.mean([Z[f'seedraw__{cfg}__{s}'] for s in DD.SEEDS[fam]], axis=0))
                    sr = C.read_json(OUT / 'internal_stop' / f'{key}.json')['stop_record']
                    ok &= sr['fit90_stop10_match_overlap'] == 0 and sr['full']['rows'] == int(M['TRAIN'].sum())
                    for s, r in sr['seeds'].items():
                        ok &= r['best_iteration'] == r['first_argmin_iteration'] == int(np.argmin(r['stop_curve'])) + 1 == r['final_n_estimators']
                ok &= all(sel['reload_identity'][cfg].values())
                b = Z = None
                add(f'{key}_fit_selection_calibration_preprocessing_stop_and_reload_verified', ok, chosen=sel['chosen'])
            D = None
        # 6 evaluation reconstruction
        maxd, boot_d, cells_checked, cell_ok, ident_ok, pop_ok = dict(auc=0.0, brier=0.0, logloss=0.0), 0.0, 0, True, True, True
        boot_problems, boot_counts = [], dict(eligible=0, ineligible_explicit_skip=0)
        for set_name in DD.EVAL_SETS:
            rb = C.read_json(OUT / 'rebuild' / f'{set_name}.json')
            for coh in DD.COHORTS:
                r = res['results'][set_name][coh]
                idn = r['checks']['identity']
                ident_ok &= bool(idn['y_equal'] and idn['file_sha256_equals_frozen'] and idn['common_rows_found_in_parent'] == idn['common_rows'] and all(idn[f'frozen_{f}_equals_parent_on_common_rows'] for f in ('pt', 'logit', 'lgbm')))
                pop_ok &= r['rows'] == rb['valid_h90']['by_cohort_dev'][coh] and r['population']['old_rows'] == DD.EXPECTED_COUNTS_OLD[set_name][coh]
                with np.load(OUT / r['predictions_file'], allow_pickle=False) as z:
                    E = {k: z[k] for k in z.files}
                cells = DD.cell_masks(E['p_pre'], E['time_minutes'])
                cell_ok &= all(np.array_equal(cells[k], E[f'cell__{k}']) for k in cells)
                y, g = E['y'].astype(int), E['match'].astype(str)
                for c, m in cells.items():
                    if not m.any():
                        continue
                    w = DD.weights(g[m])
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
                    boot_problems += [f'{set_name} {coh} {c}: {p}' for p in probs]
                for c, bt in r['bootstrap'].items():
                    if bt.get('computed', True) is False:
                        continue
                    for name, mc in bt['model_ci'].items():
                        for k in ('brier', 'logloss', 'auc'):
                            if mc[k]['estimate'] is not None and r['metrics_named'][c][name][k] is not None:
                                boot_d = max(boot_d, abs(mc[k]['estimate'] - r['metrics_named'][c][name][k]))
                E = None
        add('frozen_winners_equal_parent_predictions_on_common_rows', ident_ok)
        add('evaluation_rows_equal_dev_population_valid_cohort_rows_and_old_counts', pop_ok)
        add('cell_membership_recomputed_equal', cell_ok)
        add('independent_metric_reconstruction_lt_1e-10', max(maxd.values()) < 1e-10, max_abs=maxd, cells=cells_checked)
        add('bootstrap_point_estimates_equal_cell_metrics_lt_1e-10', boot_d < 1e-10, max_abs=boot_d)
        add('bootstrap_acceptance_all_B40_B45_cells', not boot_problems and sum(boot_counts.values()) == len(DD.EVAL_SETS) * len(DD.COHORTS) * len(BOOT_CELLS), counts=boot_counts, problems=boot_problems[:50])
        # 7 fresh-process reload on dev MAIN TEST
        F, Lb, Co = DD.load_dev_set('MAIN_TEST', OUT, 'post-run reload identity')
        names = [str(x) for x in F['input_names']]
        winners = fz['parent_references']['iq_winners']
        for coh in DD.COHORTS:
            m = (Lb['valid_h90'] == 1) & (Co['cohort'] == DD.COHORT_CODE[coh])
            with np.load(OUT / 'eval' / 'predictions' / f'MAIN_TEST_h{DD.H}_{coh}.npz', allow_pickle=False) as z:
                saved = {n: z[f'named__{n}'] for n in DD.NAMED}
                ok = np.array_equal(z['match'].astype(str), F['match'][m].astype(str)) and np.array_equal(z['s_ms'], F['s_ms'][m])
            for fam in DD.FAMILIES:
                wn = fz['family_winners'][f'{fam}_{coh}']
                b = joblib.load(OUT / wn['bundle'])
                ok = ok and bool(np.array_equal(DD.bundle_predict(b, wn['calibration'], F['X_input'][m], names), saved[f'refit_{fam}']))
            for fam in ('pt', 'logit', 'lgbm'):
                wn = winners[f'{fam}_{coh}']
                b = joblib.load(DD.IQ / wn['bundle'])
                ok = ok and bool(np.array_equal(Q.bundle_predict(b, wn['calibration'], F['X_input'][m], names), saved[f'frozen_{fam}']))
            add(f'MAIN_TEST_{coh}_winners_reload_identical_in_fresh_process', ok)
        F = Lb = Co = None
        stat = C.read_json(OUT / 'status.json')
        required = ['driver_detect', 'driver_rebuild_dev', 'driver_fit', 'driver_sealed', 'fit_freeze', 'evaluate', 'census'] + [f'rebuild_{s}' for s in DD.ALL_SETS]
        add('required_stage_groups_complete', all(stat['groups'].get(g, {}).get('state') == 'complete' for g in required))
        add('no_bytecode_written_in_parent_scripts_tests', diff['pycache_scripts']['equal'] and diff['pycache_tests']['equal'])
        failed = [k for k, v in checks.items() if not v]
        C.write_json(OUT / 'validation.json', dict(role=DD.ROLE_TAG, version=DD.VERSION, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), n_checks=len(checks), failed=failed, passed=len(checks) - len(failed),
                                                    checks=checks, info=info, frozen_manifest_sha256=C.sha256_file(DD.frozen_path(OUT)), results_sha256=C.sha256_file(OUT / 'eval' / 'results.json')))
        st.update('complete' if not failed else 'failed', 'validated', n_checks=len(checks), failed=failed[:20], next_step='report')
        print(f'checks {len(checks)} failed {len(failed)}: {failed}')
        return 0 if not failed else 1
    except Exception as exc:
        st.log(traceback.format_exc())
        DD.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'validate', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
