"""Full-corpus stage Q: pre-engagement q candidates for h90 (primary), then h60/h120 (independent sensitivity).

Restores the P4 candidate pool and selection rule on the FULL manifest: TRAIN (15.14, held-out-fold labels and
p_pre_V) fits; Q_CAL fits calibrators; Q_SELECT chooses (Brier, then log loss, then name). Equal match weights.
After all three selections are saved with bundle hashes, frozen_manifest.json freezes every V/q artifact before
any TEST/external label or prediction. q diagnostics on V_CAL/V_SELECT are computed after selection only.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import fc20260915_data as D  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import SplineTransformer, StandardScaler  # noqa: E402

ORDER = (90, 60, 120)


def load_split(L, name):
    with np.load(L.base / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok')}
    with np.load(L.base / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        keys = ['match', 's', 'sub_role', 'adapter_id', 'adapter_sha256', 'p_pre'] + \
               [f'{k}_h{h}' for h in C.HORIZONS_S for k in ('valid', 'Y', 'delta', 'endpoint')]
        Lb = {k: z[k] for k in keys}
    if not (np.array_equal(F['match'], Lb['match']) and np.array_equal(F['s_ms'], Lb['s'])):
        raise ValueError(f'{name}: feature/label row order differs')
    ok = F['pre_ok'] == 1
    if not np.array_equal(F['X_input'][ok, -1], Lb['p_pre'][ok]):
        raise ValueError(f'{name}: p_pre_V column differs from label p_pre')
    return F, Lb


def fit_candidates(X, y, g, M, cix, sets, man):
    tr, ca = M['train'], M['calibrate']
    w, cw = C.weights(g[tr]), C.weights(g[ca])
    B, fitlog = {}, {}

    def common(cand, key):
        return dict(candidate=cand, manifest=man, input_set=key, input_columns=cix[key], input_names=sets[key])
    B['constant'] = dict(candidate='constant', manifest=man, input_set=None, input_columns=[], input_names=[],
                         prior=float(np.average(y[tr], weights=w)))
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        for cand, pipe in (('p_pre_logistic', make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))),
                           ('p_pre_spline', make_pipeline(SplineTransformer(n_knots=5, degree=3), StandardScaler(),
                                                          LogisticRegression(C=1, max_iter=2000)))):
            pipe.fit(X[tr][:, cix['p_pre']], y[tr], logisticregression__sample_weight=w)
            B[cand] = dict(common(cand, 'p_pre'), pipeline=pipe, fitted_on='TRAIN')
        ridge = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=3000))
        ridge.fit(X[tr][:, cix['ridge']], y[tr], logisticregression__sample_weight=w)
        B['ridge_raw'] = dict(common('ridge_raw', 'ridge'), pipeline=ridge, fitted_on='TRAIN')
        fitlog['ridge_seconds'] = round(time.time() - t0, 1)
        from lightgbm import LGBMClassifier
        seeds = []
        t1 = time.time()
        for seed in C.ECON_SEEDS:
            m = LGBMClassifier(random_state=seed, **C.LGBM_PARAMS)
            m.fit(X[tr][:, cix['economic']], y[tr], sample_weight=w)
            seeds.append(m)
        fitlog['economic_seconds'] = round(time.time() - t1, 1)
        B['economic_raw'] = dict(common('economic_raw', 'economic'), seed_models=seeds, seeds=list(C.ECON_SEEDS), fitted_on='TRAIN')
        raw_ca = {'ridge': C.predict_q_bundle(B['ridge_raw'], X[ca]), 'economic': C.predict_q_bundle(B['economic_raw'], X[ca])}
        for base in ('ridge', 'economic'):
            p = raw_ca[base]
            sg = LogisticRegression(C=1e6, max_iter=1000).fit(C.logit8(p), y[ca], sample_weight=cw)
            iso = IsotonicRegression(out_of_bounds='clip').fit(p, y[ca], sample_weight=cw)
            extra = dict(base_pipeline=ridge) if base == 'ridge' else dict(seed_models=seeds, seeds=list(C.ECON_SEEDS))
            B[base + '_sigmoid'] = dict(common(base + '_sigmoid', base), calibrator=sg, **extra, fitted_on='base: TRAIN; calibrator: Q_CAL')
            B[base + '_isotonic'] = dict(common(base + '_isotonic', base), calibrator=iso, **extra, fitted_on='base: TRAIN; calibrator: Q_CAL')
    fitlog['warnings'] = sorted(set(f'{x.category.__name__}: {str(x.message)[:200]}' for x in wl))
    tot = np.bincount(np.unique(g[tr], return_inverse=True)[1], weights=w)
    fitlog['train_weight'] = dict(rows=int(tr.sum()), sum=float(w.sum()), per_match_total_min=float(tot.min()), per_match_total_max=float(tot.max()))
    fitlog['calibrate_weight'] = dict(rows=int(ca.sum()), sum=float(cw.sum()))
    fitlog['n_iter'] = dict(ridge=int(ridge[-1].n_iter_[0]), p_pre_logistic=int(B['p_pre_logistic']['pipeline'][-1].n_iter_[0]),
                            p_pre_spline=int(B['p_pre_spline']['pipeline'][-1].n_iter_[0]))
    fitlog['converged'] = dict(ridge=fitlog['n_iter']['ridge'] < 3000, p_pre_logistic=fitlog['n_iter']['p_pre_logistic'] < 2000,
                               p_pre_spline=fitlog['n_iter']['p_pre_spline'] < 2000)
    fitlog['isotonic_thresholds'] = {b: int(len(B[b + '_isotonic']['calibrator'].X_thresholds_)) for b in ('ridge', 'economic')}
    fitlog['sigmoid_params'] = {b: dict(slope=float(B[b + '_sigmoid']['calibrator'].coef_[0, 0]),
                                        intercept=float(B[b + '_sigmoid']['calibrator'].intercept_[0])) for b in ('ridge', 'economic')}
    fitlog['constant_prior'] = B['constant']['prior']
    return B, fitlog


def origins_check(B, X, y, g, M, cix):
    o = {}
    Xtr, Xca = X[M['train']], X[M['calibrate']]
    for cand, key in (('p_pre_logistic', 'p_pre'), ('ridge_raw', 'ridge')):
        sc = B[cand]['pipeline'][0]
        ref = StandardScaler().fit(Xtr[:, cix[key]])
        o[cand + '_scaler_equals_train_refit'] = bool(np.array_equal(sc.mean_, ref.mean_) and np.array_equal(sc.scale_, ref.scale_))
        o[cand + '_scaler_n_samples_seen_equals_train_rows'] = int(sc.n_samples_seen_) == int(M['train'].sum())
    sp = B['p_pre_spline']['pipeline'][0]
    ref = SplineTransformer(n_knots=5, degree=3).fit(Xtr[:, cix['p_pre']])
    o['spline_knots_equal_train_refit'] = bool(np.array_equal(sp.bsplines_[0].t, ref.bsplines_[0].t))
    cw = C.weights(g[M['calibrate']])
    for base in ('ridge', 'economic'):
        raw = C.predict_q_bundle(B[base + '_raw'], Xca)
        sg = LogisticRegression(C=1e6, max_iter=1000).fit(C.logit8(raw), y[M['calibrate']], sample_weight=cw)
        o[base + '_sigmoid_equals_q_cal_refit'] = bool(np.array_equal(sg.coef_, B[base + '_sigmoid']['calibrator'].coef_)
                                                       and np.array_equal(sg.intercept_, B[base + '_sigmoid']['calibrator'].intercept_))
        iso = IsotonicRegression(out_of_bounds='clip').fit(raw, y[M['calibrate']], sample_weight=cw)
        o[base + '_isotonic_equals_q_cal_refit'] = bool(np.array_equal(iso.X_thresholds_, B[base + '_isotonic']['calibrator'].X_thresholds_)
                                                        and np.array_equal(iso.y_thresholds_, B[base + '_isotonic']['calibrator'].y_thresholds_))
    return o


def main():
    import warnings as _w
    _w.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    L = D.Layout(args.smoke)
    st = C.Status(L.base, 'fit_q')
    try:
        if L.frozen_manifest.exists():
            raise SystemExit('frozen_manifest.json exists; q selections already frozen')
        lman = C.read_json(L.base / 'labels' / 'labels_trainval_manifest.json')
        vman = C.read_json(L.base / 'v_models_manifest.json')
        FT, LT = load_split(L, 'MAIN_TRAIN')
        FV, LV = load_split(L, 'MAIN_VALIDATION')
        names = [str(x) for x in FT['input_names']]
        if names != [str(x) for x in FV['input_names']] or names[-1] != C.P_PRE:
            raise SystemExit('q input schema mismatch')
        sets = C.q_feature_sets(names)
        bad = [n for n in sets['ridge'] if any(t in n for t in C.Q_FORBIDDEN_TOKENS) and n != C.P_PRE]
        if bad or any('champion_id' in n for n in sets['ridge'] + sets['economic']) or 'snapshot_age_s' in names:
            raise SystemExit(f'forbidden q predictors: {bad}')
        cix = {k: [names.index(n) for n in v] for k, v in sets.items()}
        sets_sha = C.sha256_json(sets)
        schema = dict(role=C.ROLE_TAG, input_names_all=names, predictor_sets=sets, predictor_sets_sha256=sets_sha,
                      ridge_count=len(sets['ridge']), economic_count=len(sets['economic']),
                      categorical_excluded=[n for n in names if n.endswith('champion_id')],
                      feature_units={n: C.feature_unit(n) for n in names if not n.endswith('champion_id')},
                      shap_groups={n: C.shap_group(n) for n in names},
                      excluded=['snapshot_age_s (audit only)', 'champion IDs', 'positions', 'post-state, L, duration, endpoint/reason, '
                                'future membership, final outcome, event counts after q_pre, fold ids, match ids'],
                      p_pre_V='held-out-fold V on TRAIN rows; final V on all other rows')
        C.write_json(L.base / 'q_pre_only_schema.json', schema)
        st.update('running', 'schema', ridge=len(sets['ridge']), economic=len(sets['economic']), next_step='fit h90')
        X = np.vstack([FT['X_input'], FV['X_input']])
        g = np.concatenate([FT['match'], FV['match']]).astype(str)
        sr = np.concatenate([FT['sub_role'], FV['sub_role']]).astype(str)
        s_ms = np.concatenate([FT['s_ms'], FV['s_ms']])
        pre_ok = np.concatenate([FT['pre_ok'], FV['pre_ok']]) == 1
        is_train = np.char.startswith(sr, 'fold')
        selections, fitlogs, bundle_hashes, diag, identity, origins, counts = {}, {}, {}, {}, {}, {}, {}
        for h in ORDER:
            valid = np.concatenate([LT[f'valid_h{h}'], LV[f'valid_h{h}']]) == 1
            y = np.concatenate([LT[f'Y_h{h}'], LV[f'Y_h{h}']]).astype(int)
            M = dict(train=valid & is_train, calibrate=valid & (sr == 'Q_CAL'), select=valid & (sr == 'Q_SELECT'),
                     v_cal_diag=valid & (sr == 'V_CAL'), v_select_diag=valid & (sr == 'V_SELECT'))
            if not np.all(pre_ok[valid]) or not np.isfinite(X[valid]).all():
                raise SystemExit(f'h{h}: valid rows with missing/non-finite pre inputs')
            counts[f'h{h}'] = {k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m]))),
                                       positive_rate_match_weighted=float(np.average(y[m], weights=C.weights(g[m]))) if m.any() else None)
                               for k, m in M.items()}
            man = dict(role=C.ROLE_TAG, output_is='q probability of generated label Y (NOT a V win probability)', horizon_s=h,
                       label_definition=f'Y_h{h} = 1[V(endpoint_h{h}) - V(s-1) > 0], same adapter; exact zero -> 0',
                       label_adapters='TRAIN: held-out-fold V; VALIDATION: final V', v_final_sha256=vman['final']['sha256'],
                       v_oof_sha256=vman['oof_sha256'], labels_trainval_manifest_sha256=C.sha256_file(L.base / 'labels' / 'labels_trainval_manifest.json'),
                       predictor_sets_sha256=sets_sha, protocol_sha256=C.sha256_file(C.OUT / 'protocol.json'))
            st.update('running', f'fit_h{h}', counts=counts[f'h{h}'], next_step=f'select h{h}')
            B, fl = fit_candidates(X, y, g, M, cix, sets, man)
            fitlogs[f'h{h}'] = fl
            mdir = L.base / 'models' / 'q' / f'h{h}'
            mdir.mkdir(parents=True, exist_ok=True)
            bundle_hashes[f'h{h}'] = {}
            P = {}
            for cand in C.Q_CANDIDATES:
                path = mdir / f'{cand}.joblib'
                joblib.dump(B[cand], path)
                bundle_hashes[f'h{h}'][cand] = C.sha256_file(path)
                P[cand] = {k: C.predict_q_bundle(B[cand], X[M[k]]) for k in ('train', 'calibrate', 'select')}
            met = {c: {k: C.evaluate(y[M[k]], P[c][k], g[M[k]]) for k in ('train', 'calibrate', 'select')} for c in C.Q_CANDIDATES}
            rows = [(met[c]['select']['brier'], met[c]['select']['logloss'], c) for c in C.Q_CANDIDATES]
            chosen = min(rows)[2]
            sel = dict(role=C.ROLE_TAG, manifest=man, horizon_s=h, rule='lowest Q_SELECT match-weighted Brier, then log loss, then name',
                       chosen=chosen, selected_ridge_variant=min(r for r in rows if r[2].startswith('ridge_'))[2],
                       selected_economic_variant=min(r for r in rows if r[2].startswith('economic_'))[2],
                       ranking=[r[2] for r in sorted(rows)], select_metrics={c: {m: met[c]['select'][m] for m in ('brier', 'logloss', 'auc')} for c in C.Q_CANDIDATES},
                       split_counts=counts[f'h{h}'], bundle_sha256=bundle_hashes[f'h{h}'], fit_log=fl,
                       written_at=time.strftime('%Y-%m-%d %H:%M:%S'), test_or_external_labels_exist=(L.base / 'labels' / 'MAIN_TEST_labels.npz').exists())
            sel_sha = C.write_json(L.base / 'selection' / f'q_h{h}.json', sel)
            selections[f'h{h}'] = dict(chosen=chosen, sha256=sel_sha, ridge=sel['selected_ridge_variant'], economic=sel['selected_economic_variant'])
            st.update('running', f'selected_h{h}', chosen=chosen, select_brier={c: round(met[c]['select']['brier'], 6) for c in C.Q_CANDIDATES},
                      selection_sha256=sel_sha, next_step='serialization identity and diagnostics')
            # serialization identity, fitting origins, diagnostics on V roles (after selection)
            identity[f'h{h}'] = {}
            for cand in C.Q_CANDIDATES:
                b = joblib.load(mdir / f'{cand}.joblib')
                identity[f'h{h}'][cand] = all(np.array_equal(C.predict_q_bundle(b, X[M[k]]), P[cand][k]) for k in ('train', 'calibrate', 'select'))
            origins[f'h{h}'] = origins_check(B, X, y, g, M, cix)
            diag[f'h{h}'] = {c: {k: C.evaluate(y[M[k]], C.predict_q_bundle(B[c], X[M[k]]), g[M[k]], bins=False)
                                 for k in ('v_cal_diag', 'v_select_diag')} for c in (chosen, 'constant')}
            C.write_json(L.base / 'q_fit_metrics' / f'h{h}.json', dict(role=C.ROLE_TAG, horizon_s=h, metrics=met, v_role_diagnostics_after_selection=diag[f'h{h}'],
                                                                        serialization_identity=identity[f'h{h}'], fitting_origins=origins[f'h{h}']))
            C.save_npz(L.base / 'predictions' / f'q_h{h}_trainval.npz', match=g, s_ms=s_ms, sub_role=sr, valid=valid, y=np.where(valid, y, -1),
                       **{c: np.where(valid, C.predict_q_bundle(B[c], np.nan_to_num(X)), np.nan) for c in C.Q_CANDIDATES})
            B = None
        failed = [f'h{h}:{c}' for h in ORDER for c, ok in identity[f'h{h}'].items() if not ok] + \
                 [f'h{h}:{k}' for h in ORDER for k, ok in origins[f'h{h}'].items() if not ok]
        if failed:
            raise SystemExit(f'pre-freeze checks failed: {failed}')
        frozen = dict(role=C.ROLE_TAG, protocol_sha256=C.sha256_file(C.OUT / 'protocol.json'),
                      selection_v_sha256=C.sha256_file(L.base / 'selection_v.json'), v_chosen=vman['chosen'],
                      v_final_sha256=vman['final']['sha256'], v_final_path=vman['final']['path'],
                      v_candidates_sha256=vman['candidates_sha256'], v_oof_sha256=vman['oof_sha256'],
                      q_selections=selections, q_bundle_sha256=bundle_hashes, q_predictor_sets_sha256=sets_sha,
                      labels_trainval_manifest_sha256=C.sha256_file(L.base / 'labels' / 'labels_trainval_manifest.json'),
                      frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                      statement='All V and q model/calibration/horizon selections are frozen here. TEST (15.16) and external labels '
                                'and predictions are generated only after this file exists; no refitting or reselection afterwards.',
                      test_labels_existed_before_freeze=(L.base / 'labels' / 'MAIN_TEST_labels.npz').exists(),
                      pre_freeze_checks=dict(serialization_identity=identity, fitting_origins=origins))
        fsha = C.write_json(L.frozen_manifest, frozen)
        st.update('complete', 'fit_q', selections=selections, frozen_manifest_sha256=fsha,
                  next_step='TEST/external labels then evaluation')
        return 0
    except SystemExit as exc:
        st.update('failed', 'fit_q', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'fit_q', error=repr(exc), next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
