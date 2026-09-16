"""Stage A2: independent q pools on teamfight (T) and non-teamfight (N) engagements; pooled frozen q stays the reference.

For cohort c in (T, N) and h in (90, 60, 120): the unchanged full-run candidate pool and parameters
(fc20260915_fit_q.fit_candidates: constant, p_pre logistic/spline, ridge C=.01 raw/sigmoid/isotonic, economic LightGBM
250 trees/15 leaves/lr .04/min child 100/lambda 1/colsample .9 seeds 7/42/123 raw/sigmoid/isotonic) is fit on all
eligible TRAIN rows of the cohort (held-out-fold labels and p_pre_V), calibrators on the cohort's Q_CAL rows, and the
choice made on the cohort's Q_SELECT rows by the unchanged rule (match-weighted Brier, then log loss, then name).
Scalers / splines / trees see only cohort TRAIN rows. Cohort membership, counts, duration and outcomes are not q
inputs (the input schema is the frozen full-run schema). No TEST or external row is read.
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
import fc20260915_fit_q as FQ  # noqa: E402
import cr20260915_common as K  # noqa: E402
import cr20260915_data as KD  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

ORDER = (90, 60, 120)


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--horizons', default='90,60,120')
    args = ap.parse_args()
    base = K.OUT / 'smoke_train_only' if args.smoke else K.OUT
    st = K.Status('fit_cohort_q' + ('_smoke' if args.smoke else ''))
    try:
        if (K.OUT / 'frozen_manifest.json').exists():
            raise SystemExit('frozen_manifest.json exists; selections are frozen')
        if not (K.OUT / 'cohorts' / 'cohort_manifest.json').exists():
            raise SystemExit('cohort manifest missing (must precede fitting)')
        if not args.smoke:
            ct = K.OUT / 'contract_tests' / 'result.json'
            if not ct.exists() or not C.read_json(ct).get('passed'):
                raise SystemExit('contract tests must pass before full fits')
        D = KD.load_trainval(smoke=args.smoke)
        names = D['names']
        prior_schema = C.read_json(K.FC / 'q_pre_only_schema.json')
        if names != prior_schema['input_names_all']:
            raise SystemExit('q input schema differs from the frozen full-run schema')
        sets = C.q_feature_sets(names)
        if C.sha256_json(sets) != prior_schema['predictor_sets_sha256']:
            raise SystemExit('predictor sets differ from the frozen full run')
        forbidden = ('cohort', 'cluster', 'present', 'n_min', 'scale', 'fine', 'role_slot', 'teamPosition', 'position')
        bad = [n for n in sets['ridge'] + sets['economic'] if any(t in n for t in forbidden)]
        if bad:
            raise SystemExit(f'cohort/role/position columns among predictors: {bad}')
        cix = {k: [names.index(n) for n in v] for k, v in sets.items()}
        lman = C.read_json(K.FC / 'labels' / 'labels_trainval_manifest.json')
        vman = C.read_json(K.FC / 'v_models_manifest.json')
        fz_prior = C.read_json(K.FC / 'frozen_manifest.json')
        cm_sha = C.sha256_file(K.OUT / 'cohorts' / 'cohort_manifest.json')
        results = {}
        horizons = [int(h) for h in args.horizons.split(',')]
        for h in [x for x in ORDER if x in horizons]:
            y = D[f'Y_h{h}']
            pooled_chosen = fz_prior['q_selections'][f'h{h}']['chosen']
            pooled_bundle = joblib.load(K.FC / 'models' / 'q' / f'h{h}' / f'{pooled_chosen}.joblib')
            if C.sha256_file(K.FC / 'models' / 'q' / f'h{h}' / f'{pooled_chosen}.joblib') != fz_prior['q_bundle_sha256'][f'h{h}'][pooled_chosen]:
                raise SystemExit('pooled frozen q hash differs')
            for coh in K.COHORTS:
                M = KD.masks(D, h, coh)
                if not np.all(D['pre_ok'][M['train'] | M['calibrate'] | M['select']]):
                    raise SystemExit('valid rows with missing pre inputs')
                counts = {k: dict(rows=int(m.sum()), matches=int(len(np.unique(D['g'][m]))),
                                  positive_rate_match_weighted=float(np.average(y[m], weights=C.weights(D['g'][m]))) if m.any() else None)
                          for k, m in M.items()}
                man = dict(role=K.ROLE_TAG, cohort=coh, cohort_rule=('min(cluster_blue, cluster_red) >= 4' if coh == 'T' else 'known scale and min < 4'),
                           output_is='q probability of generated label Y within the cohort (NOT a V win probability)', horizon_s=h,
                           label_definition=f'Y_h{h} = 1[V(endpoint_h{h}) - V(s-1) > 0] (frozen full-run labels, unchanged)',
                           v_final_sha256=vman['final']['sha256'], v_oof_sha256=vman['oof_sha256'],
                           labels_trainval_manifest_sha256=C.sha256_file(K.FC / 'labels' / 'labels_trainval_manifest.json'),
                           predictor_sets_sha256=prior_schema['predictor_sets_sha256'], cohort_manifest_sha256=cm_sha,
                           smoke_train_only=bool(args.smoke))
                st.update('running', f'fit_{coh}_h{h}', counts=counts, next_step='select')
                t0 = time.time()
                B, fl = FQ.fit_candidates(D['X'], y, D['g'], M, cix, sets, man)
                fl['fit_seconds_total'] = round(time.time() - t0, 1)
                mdir = base / 'models' / 'q_specialist' / coh / f'h{h}'
                mdir.mkdir(parents=True, exist_ok=True)
                hashes, P = {}, {}
                for cand in C.Q_CANDIDATES:
                    joblib.dump(B[cand], mdir / f'{cand}.joblib')
                    hashes[cand] = C.sha256_file(mdir / f'{cand}.joblib')
                    P[cand] = {k: C.predict_q_bundle(B[cand], D['X'][M[k]]) for k in M}
                met = {c: {k: C.evaluate(y[M[k]], P[c][k], D['g'][M[k]], bins=(k == 'select')) for k in M} for c in C.Q_CANDIDATES}
                rows = [(met[c]['select']['brier'], met[c]['select']['logloss'], c) for c in C.Q_CANDIDATES]
                chosen = min(rows)[2]
                ident = {}
                for cand in C.Q_CANDIDATES:
                    b2 = joblib.load(mdir / f'{cand}.joblib')
                    ident[cand] = all(np.array_equal(C.predict_q_bundle(b2, D['X'][M[k]]), P[cand][k]) for k in M)
                origins = FQ.origins_check(B, D['X'], y, D['g'], M, cix)
                pooled_diag = {k: C.evaluate(y[M[k]], C.predict_q_bundle(pooled_bundle, D['X'][M[k]]), D['g'][M[k]], bins=False)
                               for k in ('calibrate', 'select')}
                sel = dict(role=K.ROLE_TAG, manifest=man, cohort=coh, horizon_s=h,
                           rule='lowest cohort Q_SELECT match-weighted Brier, then log loss, then name (unchanged full-run rule)',
                           chosen=chosen, selected_ridge_variant=min(r for r in rows if r[2].startswith('ridge_'))[2],
                           selected_economic_variant=min(r for r in rows if r[2].startswith('economic_'))[2],
                           ranking=[r[2] for r in sorted(rows)],
                           select_metrics={c: {m: met[c]['select'][m] for m in ('brier', 'logloss', 'auc')} for c in C.Q_CANDIDATES},
                           split_counts=counts, bundle_sha256=hashes, fit_log=fl, serialization_identity=ident, fitting_origins=origins,
                           pooled_reference=dict(chosen=pooled_chosen, sha256=fz_prior['q_bundle_sha256'][f'h{h}'][pooled_chosen],
                                                 DIAGNOSTIC_cohort_validation_metrics_not_used_for_selection=pooled_diag),
                           written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                           new_frozen_manifest_exists=(K.OUT / 'frozen_manifest.json').exists())
                if not all(ident.values()) or not all(origins.values()):
                    raise SystemExit(f'{coh} h{h}: serialization/origin checks failed {ident} {origins}')
                sha = C.write_json(base / 'selection' / f'q_specialist_{coh}_h{h}.json', sel)
                C.write_json(base / 'q_fit_metrics' / f'specialist_{coh}_h{h}.json', dict(role=K.ROLE_TAG, metrics=met))
                vmask = M['train'] | M['calibrate'] | M['select']
                C.save_npz(base / 'predictions' / f'specialist_{coh}_h{h}_trainval.npz', match=D['g'][vmask], s_ms=D['s_ms'][vmask],
                           sub_role=D['sr'][vmask], y=y[vmask], **{c: C.predict_q_bundle(B[c], D['X'][vmask]) for c in C.Q_CANDIDATES},
                           pooled_chosen=C.predict_q_bundle(pooled_bundle, D['X'][vmask]))
                results[f'{coh}_h{h}'] = dict(chosen=chosen, selection_sha256=sha, bundle_sha256=hashes,
                                              select_brier={c: round(met[c]['select']['brier'], 6) for c in C.Q_CANDIDATES},
                                              pooled_select_brier_DIAG=round(pooled_diag['select']['brier'], 6))
                st.update('running', f'selected_{coh}_h{h}', chosen=chosen, select_brier=results[f'{coh}_h{h}']['select_brier'],
                          pooled_select_brier_DIAG=results[f'{coh}_h{h}']['pooled_select_brier_DIAG'], seconds=fl['fit_seconds_total'],
                          next_step='next cohort/horizon')
                B = None
        C.write_json(base / 'selection' / 'q_specialist_summary.json', dict(role=K.ROLE_TAG, results=results,
                                                                            written_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        st.update('complete', 'fit_cohort_q', results={k: v['chosen'] for k, v in results.items()},
                  next_step='role models / arms, then freeze')
        return 0
    except SystemExit as exc:
        st.update('failed', 'fit_cohort_q', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'fit_cohort_q', error=repr(exc), next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
