"""Stage V: post-run verification -> validation.json (independent recomputation where possible; real pass/fail).

Re-derives from saved artifacts and parent inputs: parent integrity diff, stage ordering and sealed-access timing,
frozen hashes (own and iq), counts and schema, TRAIN-only imputers / scalers, stop-record consistency and the shared
stop allocation, architecture parameter counts, equal match weights, Q_CAL calibrator refits, seed-mean identity,
Q_SELECT selection reproduction with direct formulas, the five-family ranking, iq comparator joins reproducing the iq
reported metrics, cell membership, an independent metric / bootstrap point reconstruction from the saved evaluation
predictions, bootstrap acceptance gates, and reload identity of the frozen winners on MAIN TEST rows in a fresh process.
"""
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

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

OUT = T.OUT
BOOT_CELLS = ('all', 'B40', 'B45')


def ts(s):
    return time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S'))


def bootstrap_cell_acceptance(bt, y, g):
    """Acceptance of one saved all/B40/B45 bootstrap record against the saved rows of that cell -> (problems, eligible)."""
    y = np.asarray(y).astype(int)
    g = np.asarray(g).astype(str)
    nm = int(len(np.unique(g)))
    classes = int(len(np.unique(y))) if len(y) else 0
    eligible = nm >= T.SPARSE_MATCHES and classes == 2
    if not eligible:
        ok = bt.get('computed', True) is False and bool(bt.get('reason')) and not bt.get('pairs')
        return ([] if ok else [f'ineligible cell (matches={nm}, classes={classes}) lacks an explicit computed=False skip']), False
    if bt.get('computed', True) is False:
        return [f'eligible cell (matches={nm}, classes={classes}) skipped: {bt.get("reason")}'], True
    P = []
    reps, deg = bt.get('replicates'), bt.get('degenerate_single_class_replicates')
    if reps != T.BOOT_REPS:
        P.append(f'replicates {reps} != {T.BOOT_REPS}')
    if bt.get('seed') != T.BOOT_SEED:
        P.append(f'seed {bt.get("seed")} != {T.BOOT_SEED}')
    if bt.get('matches') != nm or bt.get('rows') != len(y):
        P.append(f'matches/rows {bt.get("matches")}/{bt.get("rows")} != cell {nm}/{len(y)}')
    if not isinstance(deg, int) or isinstance(deg, bool) or not isinstance(reps, int) or not 0 <= deg < reps:
        return P + [f'degenerate count {deg} incoherent with replicates {reps}'], True
    if set(bt.get('model_ci', {})) != set(T.BOOT_MODELS):
        P.append(f'model_ci models {sorted(bt.get("model_ci", {}))} != {sorted(T.BOOT_MODELS)}')
    got = [(p.get('a'), p.get('b'), p.get('label')) for p in bt.get('pairs', [])]
    if got != [tuple(c) for c in T.CONTRASTS]:
        P.append(f'contrasts {got} != planned six')

    def interval(rec, expected, where):
        c = rec.get('ci95') if isinstance(rec, dict) else None
        if not isinstance(rec, dict) or rec.get('finite_replicates') != expected:
            P.append(f'{where}: finite replicates {rec.get("finite_replicates") if isinstance(rec, dict) else None} != {expected}')
        if not isinstance(rec, dict) or rec.get('estimate') is None or not np.isfinite(rec['estimate']):
            P.append(f'{where}: non-finite point estimate')
        if not c or len(c) != 2 or not all(v is not None and np.isfinite(v) for v in c) or c[0] > c[1]:
            P.append(f'{where}: interval {c} not finite and ordered')

    for m, mc in bt.get('model_ci', {}).items():
        for k in ('brier', 'logloss', 'auc'):
            interval(mc.get(k), reps - deg if k == 'auc' else reps, f'model_ci {m} {k}')
    for p in bt.get('pairs', []):
        for k in ('brier', 'logloss', 'auc'):
            interval(p.get('a_minus_b', {}).get(k), reps - deg if k == 'auc' else reps, f'{p.get("a")}-{p.get("b")} {k}')
    return P, True


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    T.log_command()
    st = T.Status('postrun_checks')
    checks, info = {}, {}

    def add(name, ok, **extra):
        checks[name] = bool(ok)
        if extra:
            info[name] = C.clean(extra)
        st.log(f'{"PASS" if ok else "FAIL"} {name}')

    try:
        proto = C.read_json(OUT / 'protocol.json')
        fz = C.read_json(T.frozen_path(OUT))
        res = C.read_json(OUT / 'eval' / 'results.json')
        ct = C.read_json(OUT / 'contract_tests' / 'result.json')
        schema = C.read_json(T.FC / 'q_pre_only_schema.json')
        ridge = list(schema['predictor_sets']['ridge'])
        frozen_t = ts(fz['frozen_at'])
        # ---------------------------------------------------------------- 1 parents
        diff = C.read_json(OUT / 'integrity' / 'snapshot_diff.json')
        for k, v in diff.items():
            if isinstance(v, dict):
                add(f'parent_unchanged_{k}', v['equal'], changed=v.get('changed'), removed=v.get('removed'), added=v.get('added'), checked=v.get('checked'))
        add('parent_expected_hashes_protocol_equal_now', all(C.sha256_file(T.ROOT / p) == sha for p, sha in proto['parents']['expected_parent_file_sha256'].items()),
            files=len(proto['parents']['expected_parent_file_sha256']))
        add('spec_unchanged_since_protocol', C.sha256_file(T.SPEC) == proto['spec_sha256'])
        add('iq_frozen_manifest_unchanged', C.sha256_file(Q.frozen_path(T.IQ)) == fz['iq_references']['frozen_manifest_sha256'] == proto['parents']['incremental_q_frozen_manifest_sha256'])
        add('iq_eval_prediction_files_unchanged', all(C.sha256_file(T.iq_eval_predictions_path(s, c)) == sha for k, sha in fz['iq_references']['eval_predictions_sha256'].items()
                                                      for s, c in [(k.rsplit('_h90_', 1)[0], k.rsplit('_h90_', 1)[1])]), files=len(fz['iq_references']['eval_predictions_sha256']))
        # ---------------------------------------------------------------- 2 ordering
        snap_t = ts(C.read_json(OUT / 'integrity' / 'snapshot_before.json')['taken_at'])
        runs = sorted(OUT.glob('contract_tests/result_run*.json'), key=lambda p: int(p.stem.replace('result_run', '')))
        run_recs = [C.read_json(p) for p in runs]
        proto_t = ts(proto['written_at'])
        smoke_fz = C.read_json(T.frozen_path(T.SMOKE))
        smoke_eval = C.read_json(T.SMOKE / 'eval' / 'results.json')
        sels = {f'{f}_{c}': C.read_json(OUT / 'selection' / f'{f}_{c}.json') for f in T.FAMILIES for c in T.COHORTS}
        fit_starts = [ts(C.read_json(OUT / 'status' / f'fit_{f}_{c}.json')['history'][0]['at']) for f in T.FAMILIES for c in T.COHORTS]
        first_fit = min(fit_starts)
        latest_pass = max(ts(r['at']) for r in run_recs if r['passed'])
        add('order_snapshot_tests_protocol_smoke_latest_tests_before_full_fits',
            snap_t <= ts(run_recs[0]['at']) <= proto_t <= ts(smoke_fz['frozen_at']) <= ts(smoke_eval['evaluated_at']) and latest_pass <= first_fit,
            snapshot=C.read_json(OUT / 'integrity' / 'snapshot_before.json')['taken_at'], first_tests=run_recs[0]['at'], protocol=proto['written_at'],
            smoke_frozen=smoke_fz['frozen_at'], smoke_evaluated=smoke_eval['evaluated_at'], latest_passing_tests=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_pass)),
            first_full_fit=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first_fit)))
        add('smoke_evaluation_completed_before_full_fits', smoke_eval['complete'] and ts(smoke_eval['evaluated_at']) <= first_fit)
        add('all_selections_before_freeze', max(ts(s['written_at']) for s in sels.values()) <= frozen_t, frozen_at=fz['frozen_at'])
        add('contract_tests_passed_no_skips', ct['passed'] and ct['failed_count'] == 0 and ct['skipped_count'] == 0 and ct['error_count'] == 0,
            passed=ct['passed_count'], run=ct['run'])
        add('final_source_hashes_equal_frozen_for_fit_freeze_evaluate_common',
            all(C.sha256_file(T.ROOT / p) == sha for p, sha in fz['source_sha256'].items()
                if p.split('/')[-1] in ('ta20260916_common.py', 'ta20260916_fit.py', 'ta20260916_freeze.py', 'ta20260916_evaluate.py', 'iq20260915_common.py',
                                        'fc20260915_common.py', 'cr20260915_common.py')),
            note='report / post-run scripts may be edited after the freeze; fitting, freezing and evaluation sources must not')
        # ---------------------------------------------------------------- 3 access log and sealed sets
        acc = [json.loads(l) for l in (OUT / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        sealed = [a for a in acc if a['sealed']]
        add('sealed_accesses_only_after_freeze', bool(sealed) and all(a['frozen_manifest_exists'] and ts(a['time']) >= frozen_t for a in sealed), n_sealed=len(sealed))
        add('no_sealed_access_before_freeze_recorded_in_manifest', fz['sealed_accesses_before_freeze'] == 0)
        sacc = [json.loads(l) for l in (T.SMOKE / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        add('smoke_read_train_only', all(a['set'] == 'MAIN_TRAIN' for a in sacc) and not any(T.SMOKE.rglob('*TEST*')) and not any(T.SMOKE.rglob('*EXT_*')),
            smoke_sets=sorted({a['set'] for a in sacc}))
        add('evaluation_after_freeze', ts(res['evaluated_at']) >= frozen_t and res['complete'])
        now = {p: C.sha256_file((OUT if p == 'protocol.json' else OUT) / p) == sha for p, sha in fz['frozen_files_sha256'].items()}
        add('frozen_artifacts_unchanged_now', all(now.values()), files=len(now))
        hb, ha = C.read_json(OUT / 'eval' / 'hashes_before.json'), C.read_json(OUT / 'eval' / 'hashes_after.json')
        add('frozen_artifacts_unchanged_through_evaluation', all(hb['results'].values()) and all(ha['results'].values()) and res['frozen_hashes_unchanged'])
        add('protocol_hash_in_freeze', fz['protocol_sha256'] == C.sha256_file(OUT / 'protocol.json'))
        add('freeze_prechecks_all_pass', all(all(v for n, v in ck.items() if not n.startswith('INFO_')) for ck in fz['pre_freeze_checks'].values()))
        add('overall_winner_rule_consistent', all(o['equals_rule_over_two_family_winners'] for o in fz['overall_winner_mlp_families'].values()),
            overall={c: o['chosen'] for c, o in fz['overall_winner_mlp_families'].items()})
        add('five_family_ranking_on_identical_rows', all(all(v['iq_rows_identical'].values()) for v in fz['five_family_q_select_ranking'].values()),
            ranking={c: v['ranking'] for c, v in fz['five_family_q_select_ranking'].items()})
        # ---------------------------------------------------------------- 4 fit records per cohort
        vman = C.read_json(T.FC / 'v_models_manifest.json')
        stop_flags = {}
        for coh in T.COHORTS:
            D = T.load_trainval(OUT, smoke=False, cohort=coh)
            names = D['names']
            M = {k: D['role'] == k for k in ('TRAIN', 'Q_CAL', 'Q_SELECT')}
            g, y, X = D['g'], D['y'], D['X']
            cols = T.input_columns(names, ridge)
            add(f'{coh}_counts_equal_specification', all(int(M[k].sum()) == T.EXPECTED_COUNTS[k][coh] for k in M), counts={k: int(M[k].sum()) for k in M})
            trainfold = np.char.startswith(D['sr_true'], 'fold')
            prov = T.oof_provenance(g[trainfold], D['sr_true'][trainfold], D['adapter_id'][trainfold], D['adapter_sha256'][trainfold], vman)
            add(f'{coh}_train_rows_use_own_heldout_fold_oof_V', prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold'], **prov)
            sm = T.stop_mask(g[M['TRAIN']])
            stop_flags[coh] = dict(zip(g[M['TRAIN']].tolist(), sm.tolist()))
            Xtr = X[M['TRAIN']][:, cols]
            med = np.median(Xtr, axis=0)
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            # reference preprocessing refit on the cohort TRAIN rows (StandardScaler sets scale_ = 1 for zero-variance columns;
            # a raw np.std comparison wrongly failed those columns in validation run 1, kept in audit_followup/)
            ref_scaler = StandardScaler().fit(SimpleImputer(strategy='median').fit(Xtr).transform(Xtr))
            zero_var_cols = int(np.sum(Xtr.std(axis=0) == 0))
            for fam in T.FAMILIES:
                key = f'{fam}_{coh}'
                sel = sels[key]
                stoprec = C.read_json(OUT / 'internal_stop' / f'{key}.json')['configs']
                with np.load(OUT / 'predictions' / f'{key}_trainval.npz', allow_pickle=False) as z:
                    Z = {k: z[k] for k in z.files}
                add(f'{key}_selection_counts_equal_specification', all(sel['split_counts'][k]['rows'] == T.EXPECTED_COUNTS[k][coh] for k in M))
                add(f'{key}_input_schema', sel['input_names_sha256'] == C.sha256_json(ridge) and sel['n_inputs'] == 352)
                add(f'{key}_weights_equal_per_match', all(abs(w['per_match_total_min'] - w['per_match_total_max']) < 1e-9 and abs(w['weight_mean'] - 1) < 1e-9
                                                          for w in sel['weights'].values()))
                add(f'{key}_weight_membership_matches_roles', all(sel['weights'][k]['membership_sha256'] == T.membership_sha(g[M[k]]) for k in M))
                add(f'{key}_prediction_rows_match_data', np.array_equal(Z['match'].astype(str), g) and np.array_equal(Z['s_ms'], D['s_ms']) and np.array_equal(Z['y'], y))
                ws = T.weights(g[M['Q_SELECT']])
                direct = {c: (C.brier_direct(y[M['Q_SELECT']], Z[c][M['Q_SELECT']], ws), C.logloss_direct(y[M['Q_SELECT']], Z[c][M['Q_SELECT']], ws))
                          for c in T.candidate_names(fam) if sel['eligible'][c]}
                add(f'{key}_selection_reproduced_independently', T.select_rule(direct)[0] == sel['chosen'] == fz['family_winners'][key]['chosen'],
                    chosen=sel['chosen'], max_abs_diff=max(abs(direct[c][0] - sel['select_metrics'][c]['brier']) for c in direct))
                cal_ok, imp_ok, sc_ok, stop_ok, seed_ok, arch_ok = {}, {}, {}, {}, {}, {}
                wc = T.weights(g[M['Q_CAL']])
                for cfg in T.config_names(fam):
                    b = joblib.load(OUT / 'models' / coh / fam / f'{cfg}.joblib')
                    raw_cal = Z[f'raw__{cfg}'][M['Q_CAL']]
                    cals = K.fit_calibrators(raw_cal, y[M['Q_CAL']], wc)
                    cal_ok[cfg] = all(np.array_equal(K.apply_calibration(cal, cals[cal], Z[f'raw__{cfg}'][M[r]]), Z[f'{cfg}__{cal}'][M[r]])
                                      for cal in ('sigmoid', 'isotonic') for r in ('Q_CAL', 'Q_SELECT')) and np.array_equal(Z[f'{cfg}__raw'], Z[f'raw__{cfg}'])
                    imp_ok[cfg] = np.array_equal(b['base'].imputer.statistics_, med)
                    sc_ok[cfg] = np.array_equal(b['base'].scaler.mean_, ref_scaler.mean_) and np.array_equal(b['base'].scaler.scale_, ref_scaler.scale_) \
                        and np.array_equal(b['base'].scaler.var_, ref_scaler.var_) and int(np.max(b['base'].scaler.n_samples_seen_)) == int(M['TRAIN'].sum())
                    seeds = [np.array_equal(Z[f'raw__{cfg}'], np.mean([Z[f'seedraw__{cfg}__{s}'] for s in T.SEEDS], axis=0))]
                    seed_ok[cfg] = all(seeds) and len(b['base'].states) == len(T.SEEDS)
                    sr = stoprec[cfg]
                    ok = sr['fit90_stop10_match_overlap'] == 0 and sr['fit90']['rows'] + sr['stop10']['rows'] == sr['full']['rows'] == int(M['TRAIN'].sum())
                    for s, r in sr['seeds'].items():
                        curve = r['stop_curve']
                        ok &= r['best_epoch'] == r['first_argmin_epoch'] == int(np.argmin(curve)) + 1 == r['refit_epochs']
                        ok &= r['epochs_evaluated'] == len(curve) <= T.OPT['max_epochs']
                        ok &= r['early_stopped'] == (len(curve) < T.OPT['max_epochs'])
                        ok &= (not r['early_stopped']) or (len(curve) == r['best_epoch'] + T.OPT['patience'])
                        ok &= abs(r['best_stop_weighted_brier'] - curve[r['best_epoch'] - 1]) < 1e-12
                    stop_ok[cfg] = bool(ok)
                    p = T.config_params(fam, cfg)
                    arch_ok[cfg] = b['base'].arch['n_params'] == proto['families']['architectures'][cfg]['n_params'] == T.param_count(T.build_module(fam, 352, p['width'], p['dropout']))
                    b = None
                add(f'{key}_calibrators_equal_q_cal_refit', all(cal_ok.values()), per_config=cal_ok)
                add(f'{key}_imputer_median_equals_train', all(imp_ok.values()), per_config=imp_ok)
                add(f'{key}_scaler_equals_train_refit', all(sc_ok.values()), per_config=sc_ok, zero_variance_train_columns=zero_var_cols,
                    note='exact equality with a StandardScaler refit on cohort TRAIN (scale_ = 1 on zero-variance columns)')
                add(f'{key}_stop_records_consistent', all(stop_ok.values()), per_config=stop_ok,
                    best_epochs={cfg: {s: r['best_epoch'] for s, r in stoprec[cfg]['seeds'].items()} for cfg in stoprec},
                    cap_reached={cfg: {s: r['cap_reached'] for s, r in stoprec[cfg]['seeds'].items()} for cfg in stoprec})
                add(f'{key}_raw_equals_seed_mean', all(seed_ok.values()), per_config=seed_ok)
                add(f'{key}_architecture_parameter_counts_equal_protocol', all(arch_ok.values()), per_config=arch_ok)
                add(f'{key}_reload_identity_at_fit', all(all(v.values()) for v in sel['reload_identity'].values()))
                Z = None
            D = None
        shared = set(stop_flags['T']) & set(stop_flags['N'])
        add('stop_allocation_same_for_T_and_N', all(stop_flags['T'][m] == stop_flags['N'][m] for m in shared), shared_matches=len(shared),
            stop_share={c: float(np.mean(list(v.values()))) for c, v in stop_flags.items()})
        # ---------------------------------------------------------------- 5 evaluation reconstruction
        maxd = dict(auc=0.0, brier=0.0, logloss=0.0)
        boot_d, cells_checked, cell_ok, join_ok, counts_ok = 0.0, 0, True, True, True
        boot_problems, boot_counts = [], dict(eligible=0, ineligible_explicit_skip=0)
        iq_res = C.read_json(T.IQ / 'eval' / 'results.json')
        iq_rep = 0.0
        for set_name in T.EVAL_SETS:
            for coh in T.COHORTS:
                r = res['results'][set_name][coh]
                counts_ok &= r['rows'] == T.EXPECTED_COUNTS[set_name][coh]
                jc = r['checks']['iq_comparators']
                join_ok &= bool(jc['every_row_joined_once'] and jc['y_equal_iq_labels'] and jc['parent_keys_unique'] and jc['file_sha256_equals_frozen']
                                and jc['iq_winner_columns_equal_frozen_candidates'])
                with np.load(OUT / r['predictions_file'], allow_pickle=False) as z:
                    E = {k: z[k] for k in z.files}
                cells = T.cell_masks(E['p_pre'], E['time_minutes'])
                cell_ok &= all(np.array_equal(cells[k], E[f'cell__{k}']) for k in cells)
                y, g = E['y'].astype(int), E['match'].astype(str)
                for c, m in cells.items():
                    if not m.any():
                        continue
                    w = T.weights(g[m])
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
                if set(r['bootstrap']) != set(BOOT_CELLS):
                    boot_problems.append(f'{set_name} {coh}: bootstrap cells {sorted(r["bootstrap"])} != {list(BOOT_CELLS)}')
                for c in BOOT_CELLS:
                    if c not in r['bootstrap']:
                        continue
                    probs, elig = bootstrap_cell_acceptance(r['bootstrap'][c], y[E[f'cell__{c}']], g[E[f'cell__{c}']])
                    boot_counts['eligible' if elig else 'ineligible_explicit_skip'] += 1
                    boot_problems += [f'{set_name} {coh} {c}: {p}' for p in probs]
                if 'CI not computed' not in str(r.get('time_cells_bootstrap', '')):
                    boot_problems.append(f'{set_name} {coh}: missing explicit time-cell CI-not-computed statement')
                for c, bt in r['bootstrap'].items():
                    if bt.get('computed', True) is False:
                        continue
                    for name, mc in bt['model_ci'].items():
                        for k in ('brier', 'logloss', 'auc'):
                            if mc[k]['estimate'] is not None and r['metrics_named'][c][name][k] is not None:
                                boot_d = max(boot_d, abs(mc[k]['estimate'] - r['metrics_named'][c][name][k]))
                fam_ok = all(np.array_equal(E[f'named__{f}_winner'], E[f'cand__{fz["family_winners"][f"{f}_{coh}"]["chosen"]}']) for f in T.FAMILIES)
                add(f'eval_{set_name}_{coh}_named_winners_equal_frozen_candidates',
                    fam_ok and np.array_equal(E['named__overall_winner_mlp_families'], E[f'cand__{fz["overall_winner_mlp_families"][coh]["chosen"]}']))
                # iq comparator columns reproduce the iq reported metrics on the same rows
                iq_named = iq_res['results'][set_name][coh]['metrics_named']
                for name in T.IQ_WINNERS + T.IQ_LEGACY:
                    for c in ('all', 'B40', 'B45'):
                        for k in ('brier', 'logloss', 'auc'):
                            a, b = r['metrics_named'][c][name][k], iq_named[c][name][k]
                            if a is not None and b is not None:
                                iq_rep = max(iq_rep, abs(a - b))
                            elif (a is None) != (b is None):
                                iq_rep = float('inf')
                E = None
        add('iq_comparator_columns_reproduce_iq_reported_metrics_lt_1e-12', iq_rep < 1e-12, max_abs=iq_rep)
        add('evaluation_counts_equal_specification', counts_ok)
        add('iq_joins_complete_and_consistent', join_ok)
        add('cell_membership_recomputed_equal', cell_ok)
        add('independent_metric_reconstruction_lt_1e-10', max(maxd.values()) < 1e-10, max_abs=maxd, cells=cells_checked)
        add('bootstrap_point_estimates_equal_cell_metrics_lt_1e-10', boot_d < 1e-10, max_abs=boot_d)
        add('bootstrap_acceptance_all_B40_B45_cells', not boot_problems and sum(boot_counts.values()) == len(T.EVAL_SETS) * len(T.COHORTS) * len(BOOT_CELLS),
            counts=boot_counts, problems=boot_problems[:50],
            rule='eligible (>=30 matches, 2 classes): 1000 reps, seed 20260915, 6 planned contrasts, finite ordered CI, finite replicates '
                 '(AUC = reps - degenerate), 0 <= degenerate < reps; ineligible: explicit computed=False with reason; eligibility from saved cell masks')
        # ---------------------------------------------------------------- 6 reload frozen winners on MAIN TEST (fresh process, exact)
        F, Lb, Co, _ = T.load_parent_set('MAIN_TEST', OUT, 'post-run reload identity of frozen winners')
        names = [str(x) for x in F['input_names']]
        for coh in T.COHORTS:
            m = (Lb['valid_h90'] == 1) & (Co['cohort'] == T.COHORT_CODE[coh])
            with np.load(OUT / 'eval' / 'predictions' / f'MAIN_TEST_h90_{coh}.npz', allow_pickle=False) as z:
                saved = {f: z[f'named__{f}_winner'] for f in T.FAMILIES}
                keys_equal = np.array_equal(z['match'].astype(str), F['match'][m].astype(str)) and np.array_equal(z['s_ms'], F['s_ms'][m])
            ok = keys_equal
            for f in T.FAMILIES:
                wn = fz['family_winners'][f'{f}_{coh}']
                b = joblib.load(OUT / wn['bundle'])
                ok = ok and bool(np.array_equal(T.bundle_predict(b, wn['calibration'], F['X_input'][m], names), saved[f]))
                b = None
            add(f'MAIN_TEST_{coh}_frozen_winners_reload_identical_in_fresh_process', ok)
        F = Lb = Co = None
        # ---------------------------------------------------------------- 7 status / bytecode
        stat = C.read_json(OUT / 'status.json')
        required = ['driver_full', 'freeze', 'evaluate'] + [f'fit_{f}_{c}' for f in T.FAMILIES for c in T.COHORTS]
        add('required_stage_groups_complete', all(stat['groups'].get(g, {}).get('state') == 'complete' for g in required),
            states={g: stat['groups'].get(g, {}).get('state') for g in required})
        add('no_bytecode_written_in_parent_scripts_tests', diff['pycache_scripts']['equal'] and diff['pycache_tests']['equal'])
        failed = [k for k, v in checks.items() if not v]
        out = dict(role=T.ROLE_TAG, version=T.VERSION, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), n_checks=len(checks), failed=failed,
                   passed=len(checks) - len(failed), checks=checks, info=info, frozen_manifest_sha256=C.sha256_file(T.frozen_path(OUT)),
                   results_sha256=C.sha256_file(OUT / 'eval' / 'results.json'), scope='implementation, identity, split, hash and numeric verification; '
                   'not semantic validity of Y, not causal effects, not an untouched confirmation')
        C.write_json(OUT / 'validation.json', out)
        st.update('complete' if not failed else 'failed', 'validated', n_checks=len(checks), failed=failed[:20], next_step='report')
        print(f'checks {len(checks)} failed {len(failed)}: {failed}')
        return 0 if not failed else 1
    except Exception as exc:
        st.log(traceback.format_exc())
        T.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'validate', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
