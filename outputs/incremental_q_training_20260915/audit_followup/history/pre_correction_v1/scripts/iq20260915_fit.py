"""Stage F: one family x one cohort (h90): six configurations on ALL cohort TRAIN rows, calibrators on cohort Q_CAL,
selection of the 18 joint candidates on cohort Q_SELECT (Brier, then log loss, then name).

--smoke runs the identical code on TRAIN-only pseudo roles (never VALIDATION/TEST) under smoke_train_only/.
Resume: an existing selection file for (family, cohort) is never refit or overwritten; after the freeze nothing is fit.
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

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def gates(base, smoke):
    if not (Q.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json must precede every fit')
    if Q.frozen_path(base).exists():
        raise SystemExit(f'{Q.frozen_path(base)} exists: selections are frozen, no fitting')
    if not smoke:
        ct = Q.OUT / 'contract_tests' / 'result.json'
        if not ct.exists() or not C.read_json(ct).get('passed'):
            raise SystemExit('contract tests must pass before full fits')
        if not (Q.frozen_path(Q.SMOKE).exists() and (Q.SMOKE / 'eval' / 'results.json').exists()):
            raise SystemExit('TRAIN-only smoke (fit, freeze, evaluation) must complete before full fits')
        if (Q.OUT / 'eval').exists():
            raise SystemExit('eval/ exists before full fits')


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', choices=Q.COHORTS, required=True)
    ap.add_argument('--family', choices=Q.FAMILIES, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    Q.log_command()
    base = Q.SMOKE if args.smoke else Q.OUT
    fam, coh = args.family, args.cohort
    st = Q.Status(f'fit_{fam}_{coh}' + ('_smoke' if args.smoke else ''))
    sel_path = base / 'selection' / f'{fam}_{coh}.json'
    try:
        gates(base, args.smoke)
        if sel_path.exists():
            st.update('complete', 'resume_skip', note=f'{sel_path.name} exists; not refit', next_step='next family/cohort')
            return 0
        schema = C.read_json(Q.FC / 'q_pre_only_schema.json')
        ridge = list(schema['predictor_sets']['ridge'])
        st.update('running', 'load', next_step='fit configurations')
        D = Q.load_trainval(base, smoke=args.smoke, cohort=coh)
        names = D['names']
        if names != schema['input_names_all'] or C.q_feature_sets(names)['ridge'] != ridge or len(ridge) != Q.EXPECTED_RIDGE_COUNT:
            raise SystemExit('input schema differs from parent q_pre_only_schema.json')
        X, y, g = D['X'], D['y'], D['g']
        M = {k: D['role'] == k for k in ('TRAIN', 'Q_CAL', 'Q_SELECT')}
        counts = {k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m]))),
                          positive_rate_match_weighted=float(np.average(y[m], weights=Q.weights(g[m]))) if m.any() else None) for k, m in M.items()}
        if not args.smoke:
            bad = {k: (counts[k]['rows'], Q.EXPECTED_COUNTS[k][coh]) for k in M if counts[k]['rows'] != Q.EXPECTED_COUNTS[k][coh]}
            if bad:
                raise SystemExit(f'count mismatch blocks fitting: {bad}')
        if int(np.sum(M['TRAIN'] & M['Q_CAL'])) or len(set(g[M['TRAIN']]) & set(g[M['Q_CAL']] if M['Q_CAL'].any() else [])) \
                or len(set(g[M['TRAIN']]) & set(g[M['Q_SELECT']])) or len(set(g[M['Q_CAL']]) & set(g[M['Q_SELECT']])):
            raise SystemExit('TRAIN / Q_CAL / Q_SELECT match overlap')
        ridge_ix = [names.index(n) for n in ridge]
        nonfinite = {k: int((~np.isfinite(X[m][:, ridge_ix])).sum()) for k, m in M.items()}
        if any(nonfinite.values()):
            raise SystemExit(f'non-finite inputs among valid rows (documented stop): {nonfinite}')
        trainfold = np.char.startswith(D['sr_true'], 'fold')
        vman = C.read_json(Q.FC / 'v_models_manifest.json')
        prov = Q.oof_provenance(g[trainfold], D['sr_true'][trainfold], D['adapter_id'][trainfold], D['adapter_sha256'][trainfold], vman)
        if not (prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold']):
            raise SystemExit(f'own-match OOF provenance failed: {prov}')
        W = {k: Q.weights(g[m]) for k, m in M.items()}
        wrec = {k: Q.weight_record(g[m], W[k]) for k, m in M.items()}
        man = dict(role=Q.ROLE_TAG, version=Q.VERSION, cohort=coh, family=fam, horizon_s=Q.H, smoke_train_only=bool(args.smoke),
                   output_is='q probability of generated label Y_h90 within the cohort (NOT a V win probability)',
                   label_definition='Y_h90 = 1[V(endpoint_h90) - V(s-1) > 0] (parent labels unchanged; TRAIN OOF V)',
                   predictor_sets_sha256=schema['predictor_sets_sha256'], protocol_sha256=C.sha256_file(Q.OUT / 'protocol.json'),
                   roles='TRAIN fit; Q_CAL calibrators; Q_SELECT selection' + (' (smoke pseudo roles from TRAIN folds)' if args.smoke else ''))
        cfgs = Q.config_names(fam)
        mdir = base / 'models' / coh / fam
        mdir.mkdir(parents=True, exist_ok=True)
        input_names = list(Q.PT_INPUTS) if fam == 'pt' else ridge
        cols = Q.input_columns(names, fam, ridge)
        P, raws, seedraws, fitrec, calrec, hashes, eligible, ident, stoprec = {}, {}, {}, {}, {}, {}, {}, {}, {}
        total_t0 = time.time()
        for i, cfg in enumerate(cfgs):
            st.update('running', f'fit_{cfg}', processed=i, total=len(cfgs), next_step='calibrate / save / reload')
            t0 = time.time()
            Xtr = X[M['TRAIN']][:, cols]
            if fam == 'lgbm':
                basem = Q.LgbmQBase(cfg, input_names).fit(Xtr, y[M['TRAIN']], g[M['TRAIN']], log=st.log)
                ok_fit = True
                stoprec[cfg] = basem.stop_record
                fitrec[cfg] = dict(params=basem.params, seeds=list(basem.seeds),
                                   seeds_summary={s: {k: v for k, v in r.items() if k != 'stop_curve'} for s, r in basem.stop_record['seeds'].items()},
                                   imputer_full_n_features=int(basem.imputer.n_features_in_))
            else:
                basem = Q.LinearQBase(fam, cfg, input_names).fit(Xtr, y[M['TRAIN']], W['TRAIN'])
                ok_fit = basem.eligible
                fitrec[cfg] = dict(attempts=basem.attempts, eligible_converged=basem.eligible, design=basem.design_record)
            Xtr = None
            fit_s = time.time() - t0
            raws[cfg] = {k: basem.raw(X[m][:, cols]) for k, m in M.items()}
            if fam == 'lgbm':
                seedraws[cfg] = {k: basem.seed_raw(X[m][:, cols]) for k, m in M.items()}
                if not all(np.allclose(seedraws[cfg][k].mean(axis=0), raws[cfg][k], rtol=0, atol=1e-15) for k in M):
                    raise SystemExit('three-seed mean differs from raw candidate prediction')
            cals = K.fit_calibrators(raws[cfg]['Q_CAL'], y[M['Q_CAL']], W['Q_CAL'])
            calrec[cfg] = Q.calibrator_record(cals, raws[cfg]['Q_CAL'])
            bundle = Q.make_bundle(fam, cfg, basem, cals, names, ridge, dict(man, fitted_on='base: all cohort TRAIN rows; calibrators: cohort Q_CAL rows'))
            path = mdir / f'{cfg}.joblib'
            joblib.dump(bundle, path)
            hashes[cfg] = C.sha256_file(path)
            for cal in Q.CALS:
                P[f'{cfg}__{cal}'] = {k: Q.calibrate(bundle, cal, raws[cfg][k]) for k in M}
            b2 = joblib.load(path)
            ident[cfg] = {cal: all(np.array_equal(Q.bundle_predict(b2, cal, X[m], names), P[f'{cfg}__{cal}'][k]) for k, m in M.items() if k != 'TRAIN')
                          for cal in Q.CALS}
            for cal in Q.CALS:
                cand = f'{cfg}__{cal}'
                eligible[cand] = bool(ok_fit and all(np.isfinite(P[cand][k]).all() for k in ('Q_CAL', 'Q_SELECT')))
            fitrec[cfg]['seconds_fit'] = round(fit_s, 1)
            fitrec[cfg]['seconds_total'] = round(time.time() - t0, 1)
            st.update('running', f'fitted_{cfg}', processed=i + 1, total=len(cfgs), seconds=fitrec[cfg]['seconds_total'],
                      eligible=ok_fit, reload_identity=ident[cfg],
                      select_brier={cal: round(C.brier_direct(y[M['Q_SELECT']], P[f'{cfg}__{cal}']['Q_SELECT'], W['Q_SELECT']), 6) for cal in Q.CALS},
                      next_step='next configuration')
            basem = bundle = b2 = None
        cands = Q.candidate_names(fam)
        met = {c: {k: C.evaluate(y[m], P[c][k], g[m], bins=(k == 'Q_SELECT')) for k, m in M.items()} for c in cands}
        seed_met = {}
        if fam == 'lgbm':
            seed_met = {cfg: {str(s): {k: C.evaluate(y[M[k]], seedraws[cfg][k][j], g[M[k]], bins=False) for k in ('Q_CAL', 'Q_SELECT')}
                              for j, s in enumerate(Q.LGBM_SEEDS)} for cfg in cfgs}
        elig = {c: (met[c]['Q_SELECT']['brier'], met[c]['Q_SELECT']['logloss']) for c in cands if eligible[c]}
        if not elig:
            raise SystemExit('no eligible candidate in family (documented failure)')
        chosen, ranking = Q.select_rule(elig)
        if not all(all(v.values()) for v in ident.values()):
            raise SystemExit(f'reload identity failed: {ident}')
        sel = dict(
            role=Q.ROLE_TAG, manifest=man, family=fam, cohort=coh, horizon_s=Q.H,
            rule='lowest cohort Q_SELECT match-weighted Brier, then log loss, then candidate name (lexical); ineligible candidates excluded',
            candidates_lexical_tie_order=cands, eligible=eligible, chosen=chosen, chosen_config=Q.split_candidate(chosen)[0],
            chosen_calibration=Q.split_candidate(chosen)[1], ranking=ranking,
            select_metrics={c: {k: met[c]['Q_SELECT'][k] for k in ('brier', 'logloss', 'auc', 'intercept', 'slope', 'ece_10bin')} for c in cands},
            calibrate_metrics={c: {k: met[c]['Q_CAL'][k] for k in ('brier', 'logloss', 'auc')} for c in cands},
            train_in_sample_metrics_DESCRIPTIVE={c: {k: met[c]['TRAIN'][k] for k in ('brier', 'logloss', 'auc')} for c in cands},
            lgbm_seed_metrics_DESCRIPTIVE=seed_met, split_counts=counts, weights=wrec,
            weight_sources=dict(TRAIN='weights(TRAIN cohort match ids)', Q_CAL='weights(Q_CAL cohort match ids) -> calibrators',
                                Q_SELECT='weights(Q_SELECT cohort match ids) -> selection metrics'),
            fit_records=fitrec, calibrators=calrec, bundle_sha256=hashes, reload_identity=ident, oof_provenance=prov,
            input_names_sha256=C.sha256_json(input_names), n_inputs=len(input_names), seconds_total=round(time.time() - total_t0, 1),
            environment=Q.environment(), source_sha256=Q.own_source_hashes(),
            new_frozen_manifest_exists=Q.frozen_path(base).exists(), written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        if fam == 'lgbm':
            C.write_json(base / 'internal_stop' / f'lgbm_{coh}.json', dict(role=Q.ROLE_TAG, cohort=coh, configs=stoprec))
        arrays = dict(match=g, s_ms=D['s_ms'], role=D['role'], y=y, p_pre=D['p_pre'], time_minutes=X[:, names.index('time_minutes')])
        for cfg in cfgs:
            full = np.full(len(y), np.nan)
            for k, m in M.items():
                full[m] = raws[cfg][k]
            arrays[f'raw__{cfg}'] = full
            if fam == 'lgbm':
                for j, s in enumerate(Q.LGBM_SEEDS):
                    fs = np.full(len(y), np.nan)
                    for k, m in M.items():
                        fs[m] = seedraws[cfg][k][j]
                    arrays[f'seedraw__{cfg}__{s}'] = fs
        for c in cands:
            full = np.full(len(y), np.nan)
            for k, m in M.items():
                full[m] = P[c][k]
            arrays[c] = full
        sel['predictions_sha256'] = C.save_npz(base / 'predictions' / f'{fam}_{coh}_trainval.npz', **arrays)
        sha = C.write_json(sel_path, sel)
        st.update('complete', 'selected', chosen=chosen, selection_sha256=sha, seconds=sel['seconds_total'],
                  top3=[(r, round(met[r]['Q_SELECT']['brier'], 6)) for r in ranking[:3]], next_step='next family/cohort, then freeze')
        return 0
    except SystemExit as exc:
        Q.log_failure(st.group, exc)
        st.update('failed', 'fit', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        Q.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'fit', error=repr(exc)[:1000], next_step='inspect, fix, rerun (failure retained)')
        return 3


if __name__ == '__main__':
    sys.exit(main())
