"""POST-FREEZE DIAGNOSTIC (TRAIN / VALIDATION only; never used for selection or evaluation).

The participant-arm ridge and the A-specialist ridge use the same rows, inputs, weights and C=.01 but were fit through
different array paths; their Q_SELECT predictions differ. This quantifies the difference and refits both paths with a
tight lbfgs tolerance (tol=1e-10, max_iter=20000) to test whether it is optimizer stopping (sklearn default tol=1e-4).
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402
import cr20260915_data as KD  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def stats(d):
    a = np.abs(d)
    return dict(mean_abs=float(a.mean()), p50=float(np.quantile(a, .5)), p99=float(np.quantile(a, .99)), max=float(a.max()),
                rows_gt_1e_3=int((a > 1e-3).sum()), rows=int(len(a)))


def main():
    warnings.filterwarnings('ignore')
    if not (K.OUT / 'frozen_manifest.json').exists():
        raise SystemExit('diagnostic is post-freeze only')
    D = KD.load_trainval(smoke=False)
    names = D['names']
    sets = C.q_feature_sets(names)
    cols = [names.index(n) for n in sets['ridge']]
    out = dict(role='POST_FREEZE_DIAGNOSTIC_NOT_USED_FOR_SELECTION_OR_EVALUATION', written_at=None, cohorts={})
    for coh in K.COHORTS:
        M = KD.masks(D, 90, coh)
        a = joblib.load(K.OUT / 'models' / 'q_specialist' / coh / 'h90' / 'ridge_raw.joblib')
        b = joblib.load(K.OUT / 'models' / 'arms' / coh / 'h90' / 'participant_ridge_raw.joblib')
        Xs = D['X'][M['select']]
        pa = C.predict_q_bundle(a, Xs)
        pb = K.predict_arm_bundle(b, dict(X=D['X'], names=names), np.flatnonzero(M['select']))
        la, lb = a['pipeline'][-1], b['base_model'].pipeline[-1]
        rec = dict(select_prediction_difference=stats(pa - pb), n_iter=dict(A_specialist=int(la.n_iter_[0]), participant_arm=int(lb.n_iter_[0])),
                   scaler_mean_equal=bool(np.array_equal(a['pipeline'][0].mean_, b['base_model'].pipeline[0].mean_)),
                   scaler_scale_equal=bool(np.array_equal(a['pipeline'][0].scale_, b['base_model'].pipeline[0].scale_)),
                   standardized_coef_max_abs_diff=float(np.max(np.abs(la.coef_ - lb.coef_))),
                   intercept_abs_diff=float(abs(la.intercept_[0] - lb.intercept_[0])))
        w = C.weights(D['g'][M['train']])
        y = D['Y_h90'][M['train']]
        X_a = D['X'][M['train']][:, cols]                                   # A path
        idx_all = np.flatnonzero(M['train'] | M['calibrate'] | M['select'])
        X_b = D['X'][idx_all][:, cols][np.flatnonzero(M['train'][idx_all])]  # arm path
        tight = {}
        for tag, Xt in (('A_path', X_a), ('arm_path', X_b)):
            t0 = time.time()
            p = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=20000, tol=1e-10)).fit(Xt, y, logisticregression__sample_weight=w)
            tight[tag] = dict(pipe=p, n_iter=int(p[-1].n_iter_[0]), seconds=round(time.time() - t0, 1))
        pta = tight['A_path']['pipe'].predict_proba(Xs[:, cols])[:, 1]
        ptb = tight['arm_path']['pipe'].predict_proba(Xs[:, cols])[:, 1]
        rec['tight_tolerance_refits'] = dict(
            n_iter={k: v['n_iter'] for k, v in tight.items()}, seconds={k: v['seconds'] for k, v in tight.items()},
            tight_A_vs_tight_arm=stats(pta - ptb), production_A_vs_tight=stats(pa - pta), production_arm_vs_tight=stats(pb - ptb),
            select_brier=dict(production_A=float(C.brier_direct(D['Y_h90'][M['select']], pa, C.weights(D['g'][M['select']]))),
                              production_arm=float(C.brier_direct(D['Y_h90'][M['select']], pb, C.weights(D['g'][M['select']]))),
                              tight=float(C.brier_direct(D['Y_h90'][M['select']], pta, C.weights(D['g'][M['select']])))))
        out['cohorts'][coh] = rec
        print(coh, rec['select_prediction_difference'], rec['n_iter'], rec['tight_tolerance_refits']['n_iter'],
              rec['tight_tolerance_refits']['tight_A_vs_tight_arm']['max'], rec['tight_tolerance_refits']['production_A_vs_tight']['max'],
              rec['tight_tolerance_refits']['production_arm_vs_tight']['max'], flush=True)
    out['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    C.write_json(K.OUT / 'diagnostics' / 'ridge_optimizer_path_diagnostic.json', out)


if __name__ == '__main__':
    main()
