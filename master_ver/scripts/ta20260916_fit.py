"""Stage F: one family x one cohort (h90): six configurations x three seeds on ALL cohort TRAIN rows (after the stop
phase), calibrators on cohort Q_CAL, selection of the 18 joint candidates on cohort Q_SELECT (Brier, log loss, name).

--smoke runs the identical code on TRAIN-only pseudo roles (never VALIDATION/TEST) under smoke_train_only/.
Resume: an existing selection file for (family, cohort) is never refit or overwritten; partial artifacts block a rerun;
after the freeze nothing is fit.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ta20260916_common as T  # noqa: E402
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def assert_unique_keys(g, s_ms):
    keys = list(zip(np.asarray(g).astype(str).tolist(), np.asarray(s_ms).astype(np.int64).tolist()))
    dup = len(keys) - len(set(keys))
    if dup:
        raise SystemExit(f'pre-fit key guard: {dup} duplicated (match, s_ms) keys among {len(keys)} rows; fitting blocked')
    return len(keys)


def partial_family_artifacts(base, fam, coh):
    base = Path(base)
    found = sorted(p for p in (base / 'models' / coh / fam).glob('*.joblib')) if (base / 'models' / coh / fam).exists() else []
    others = [base / 'predictions' / f'{fam}_{coh}_trainval.npz', base / 'internal_stop' / f'{fam}_{coh}.json']
    found += [p for p in others if p.exists()]
    return [p.relative_to(base).as_posix() for p in found]


def gates(base, smoke):
    if not (T.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json must precede every fit')
    if T.frozen_path(base).exists():
        raise SystemExit(f'{T.frozen_path(base)} exists: selections are frozen, no fitting')
    if not smoke:
        ct = T.OUT / 'contract_tests' / 'result.json'
        if not ct.exists() or not C.read_json(ct).get('passed'):
            raise SystemExit('contract tests must pass before full fits')
        if not (T.frozen_path(T.SMOKE).exists() and (T.SMOKE / 'eval' / 'results.json').exists()):
            raise SystemExit('TRAIN-only smoke (fit, freeze, evaluation) must complete before full fits')
        if (T.OUT / 'eval').exists():
            raise SystemExit('eval/ exists before full fits')


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', choices=T.COHORTS, required=True)
    ap.add_argument('--family', choices=T.FAMILIES, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    T.log_command()
    base = T.SMOKE if args.smoke else T.OUT
    fam, coh = args.family, args.cohort
    st = T.Status(f'fit_{fam}_{coh}' + ('_smoke' if args.smoke else ''))
    sel_path = base / 'selection' / f'{fam}_{coh}.json'
    try:
        gates(base, args.smoke)
        if sel_path.exists():
            st.update('complete', 'resume_skip', note=f'{sel_path.name} exists; not refit', next_step='next family/cohort')
            return 0
        partial = partial_family_artifacts(base, fam, coh)
        if partial:
            raise SystemExit(f'partial-family artifacts exist without {sel_path.name} and would be overwritten: {partial}. '
                             'Retained untouched; move them to a named attempt directory before rerunning this family/cohort.')
        schema = C.read_json(T.FC / 'q_pre_only_schema.json')
        ridge = list(schema['predictor_sets']['ridge'])
        st.update('running', 'load', next_step='fit configurations')
        D = T.load_trainval(base, smoke=args.smoke, cohort=coh)
        names = D['names']
        if names != schema['input_names_all'] or C.q_feature_sets(names)['ridge'] != ridge or len(ridge) != T.EXPECTED_RIDGE_COUNT:
            raise SystemExit('input schema differs from parent q_pre_only_schema.json')
        X, y, g = D['X'], D['y'], D['g']
        n_unique_keys = assert_unique_keys(g, D['s_ms'])
        M = {k: D['role'] == k for k in ('TRAIN', 'Q_CAL', 'Q_SELECT')}
        counts = {k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m]))),
                          positive_rate_match_weighted=float(np.average(y[m], weights=T.weights(g[m]))) if m.any() else None) for k, m in M.items()}
        if not args.smoke:
            bad = {k: (counts[k]['rows'], T.EXPECTED_COUNTS[k][coh]) for k in M if counts[k]['rows'] != T.EXPECTED_COUNTS[k][coh]}
            if bad:
                raise SystemExit(f'count mismatch blocks fitting: {bad}')
        if int(np.sum(M['TRAIN'] & M['Q_CAL'])) or len(set(g[M['TRAIN']]) & set(g[M['Q_CAL']] if M['Q_CAL'].any() else [])) \
                or len(set(g[M['TRAIN']]) & set(g[M['Q_SELECT']])) or len(set(g[M['Q_CAL']]) & set(g[M['Q_SELECT']])):
            raise SystemExit('TRAIN / Q_CAL / Q_SELECT match overlap')
        cols = T.input_columns(names, ridge)
        nonfinite = {k: int((~np.isfinite(X[m][:, cols])).sum()) for k, m in M.items()}
        if any(nonfinite.values()):
            raise SystemExit(f'non-finite inputs among valid rows (documented stop): {nonfinite}')
        trainfold = np.char.startswith(D['sr_true'], 'fold')
        vman = C.read_json(T.FC / 'v_models_manifest.json')
        prov = T.oof_provenance(g[trainfold], D['sr_true'][trainfold], D['adapter_id'][trainfold], D['adapter_sha256'][trainfold], vman)
        if not (prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold']):
            raise SystemExit(f'own-match OOF provenance failed: {prov}')
        W = {k: T.weights(g[m]) for k, m in M.items()}
        wrec = {k: T.weight_record(g[m], W[k]) for k, m in M.items()}
        man = dict(role=T.ROLE_TAG, version=T.VERSION, cohort=coh, family=fam, horizon_s=T.H, smoke_train_only=bool(args.smoke),
                   output_is='q probability of generated label Y_h90 within the cohort (NOT a V win probability)',
                   label_definition='Y_h90 = 1[V(endpoint_h90) - V(s-1) > 0] (parent labels unchanged; TRAIN OOF V)',
                   predictor_sets_sha256=schema['predictor_sets_sha256'], protocol_sha256=C.sha256_file(T.OUT / 'protocol.json'),
                   device=args.device, roles='TRAIN fit; Q_CAL calibrators; Q_SELECT selection' + (' (smoke pseudo roles from TRAIN folds)' if args.smoke else ''))
        cfgs = T.config_names(fam)
        mdir = base / 'models' / coh / fam
        mdir.mkdir(parents=True, exist_ok=True)
        P, raws, seedraws, fitrec, calrec, hashes, eligible, ident, stoprec = {}, {}, {}, {}, {}, {}, {}, {}, {}
        total_t0 = time.time()
        for i, cfg in enumerate(cfgs):
            st.update('running', f'fit_{cfg}', processed=i, total=len(cfgs), next_step='calibrate / save / reload')
            t0 = time.time()
            Xtr = X[M['TRAIN']][:, cols]
            basem = T.MlpQBase(fam, cfg, ridge, device=args.device).fit(Xtr, y[M['TRAIN']], g[M['TRAIN']], log=st.log)
            Xtr = None
            fit_s = time.time() - t0
            stoprec[cfg] = basem.stop_record
            fitrec[cfg] = dict(params=basem.params, seeds=list(basem.seeds), arch=basem.arch,
                               seeds_summary={s: {k: v for k, v in r.items() if k != 'stop_curve'} for s, r in basem.stop_record['seeds'].items()},
                               imputer_full_n_features=int(basem.imputer.n_features_in_), scaler_n_samples_seen=int(np.max(basem.scaler.n_samples_seen_)))
            t2 = time.time()
            seedraws[cfg] = {k: basem.seed_raw(X[m][:, cols]) for k, m in M.items()}
            raws[cfg] = {k: seedraws[cfg][k].mean(axis=0) for k in M}  # == basem.raw by definition (seed mean)
            pred_s = time.time() - t2
            cals = K.fit_calibrators(raws[cfg]['Q_CAL'], y[M['Q_CAL']], W['Q_CAL'])
            calrec[cfg] = T.calibrator_record(cals, raws[cfg]['Q_CAL'])
            bundle = T.make_bundle(fam, cfg, basem, cals, names, ridge, dict(man, fitted_on='base: all cohort TRAIN rows; calibrators: cohort Q_CAL rows'))
            path = mdir / f'{cfg}.joblib'
            joblib.dump(bundle, path)
            hashes[cfg] = C.sha256_file(path)
            for cal in T.CALS:
                P[f'{cfg}__{cal}'] = {k: T.calibrate(bundle, cal, raws[cfg][k]) for k in M}
            b2 = joblib.load(path)
            ident[cfg] = {cal: all(np.array_equal(T.bundle_predict(b2, cal, X[m], names), P[f'{cfg}__{cal}'][k]) for k, m in M.items() if k != 'TRAIN')
                          for cal in T.CALS}
            for cal in T.CALS:
                cand = f'{cfg}__{cal}'
                eligible[cand] = bool(all(np.isfinite(P[cand][k]).all() for k in ('Q_CAL', 'Q_SELECT')))
            fitrec[cfg]['seconds_fit'] = round(fit_s, 1)
            fitrec[cfg]['seconds_predict_cpu64'] = round(pred_s, 1)
            fitrec[cfg]['seconds_total'] = round(time.time() - t0, 1)
            st.update('running', f'fitted_{cfg}', processed=i + 1, total=len(cfgs), seconds=fitrec[cfg]['seconds_total'],
                      best_epochs={s: r['best_epoch'] for s, r in basem.stop_record['seeds'].items()}, reload_identity=ident[cfg],
                      select_brier={cal: round(C.brier_direct(y[M['Q_SELECT']], P[f'{cfg}__{cal}']['Q_SELECT'], W['Q_SELECT']), 6) for cal in T.CALS},
                      next_step='next configuration')
            basem = bundle = b2 = None
        cands = T.candidate_names(fam)
        met = {c: {k: C.evaluate(y[m], P[c][k], g[m], bins=(k == 'Q_SELECT')) for k, m in M.items()} for c in cands}
        seed_met = {cfg: {str(s): {k: C.evaluate(y[M[k]], seedraws[cfg][k][j], g[M[k]], bins=False) for k in ('Q_CAL', 'Q_SELECT')}
                          for j, s in enumerate(T.SEEDS)} for cfg in cfgs}
        elig = {c: (met[c]['Q_SELECT']['brier'], met[c]['Q_SELECT']['logloss']) for c in cands if eligible[c]}
        if not elig:
            raise SystemExit('no eligible candidate in family (documented failure)')
        chosen, ranking = T.select_rule(elig)
        if not all(all(v.values()) for v in ident.values()):
            raise SystemExit(f'reload identity failed: {ident}')
        sel = dict(
            role=T.ROLE_TAG, manifest=man, family=fam, cohort=coh, horizon_s=T.H,
            rule='lowest cohort Q_SELECT match-weighted Brier, then log loss, then candidate name (lexical); ineligible candidates excluded',
            candidates_lexical_tie_order=cands, eligible=eligible, chosen=chosen, chosen_config=T.split_candidate(chosen)[0],
            chosen_calibration=T.split_candidate(chosen)[1], ranking=ranking,
            select_metrics={c: {k: met[c]['Q_SELECT'][k] for k in ('brier', 'logloss', 'auc', 'intercept', 'slope', 'ece_10bin')} for c in cands},
            calibrate_metrics={c: {k: met[c]['Q_CAL'][k] for k in ('brier', 'logloss', 'auc')} for c in cands},
            train_in_sample_metrics_DESCRIPTIVE={c: {k: met[c]['TRAIN'][k] for k in ('brier', 'logloss', 'auc')} for c in cands},
            seed_metrics_DESCRIPTIVE=seed_met, split_counts=counts, pre_fit_unique_row_keys=n_unique_keys, weights=wrec,
            weight_sources=dict(TRAIN='weights(TRAIN cohort match ids)', Q_CAL='weights(Q_CAL cohort match ids) -> calibrators',
                                Q_SELECT='weights(Q_SELECT cohort match ids) -> selection metrics'),
            fit_records=fitrec, calibrators=calrec, bundle_sha256=hashes, reload_identity=ident, oof_provenance=prov,
            input_names_sha256=C.sha256_json(ridge), n_inputs=len(ridge), seconds_total=round(time.time() - total_t0, 1),
            environment=T.environment(), source_sha256=T.own_source_hashes(),
            new_frozen_manifest_exists=T.frozen_path(base).exists(), written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        C.write_json(base / 'internal_stop' / f'{fam}_{coh}.json', dict(role=T.ROLE_TAG, cohort=coh, family=fam, configs=stoprec))
        arrays = dict(match=g, s_ms=D['s_ms'], role=D['role'], y=y, p_pre=D['p_pre'], time_minutes=X[:, names.index('time_minutes')])
        for cfg in cfgs:
            full = np.full(len(y), np.nan)
            for k, m in M.items():
                full[m] = raws[cfg][k]
            arrays[f'raw__{cfg}'] = full
            for j, s in enumerate(T.SEEDS):
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
        T.log_failure(st.group, exc)
        st.update('failed', 'fit', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        T.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'fit', error=repr(exc)[:1000], next_step='inspect, fix, rerun (failure retained)')
        return 3


if __name__ == '__main__':
    sys.exit(main())
