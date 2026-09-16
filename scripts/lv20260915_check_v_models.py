"""Label validity contract: independent checks of fitted comparator V families (smoke or full).

Reconstructs every adapter's fit membership from the fold hash (or smoke pseudo roles), verifies file hashes,
versions, preprocessing membership (scaler rows = fit rows), calibration partition, and reload identity of saved
predictions. No outcome W is read here.
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
import lv20260915_common as K  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime')

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

TRAIN_ROLES = [f'fold{k}' for k in range(C.N_FOLDS)]


def check_models(smoke, smoke_mod=12):
    from lv20260915_fit_v import smoke_roles
    base = K.OUT / 'smoke_train_only' if smoke else K.OUT
    P = K.ParentReadOnly()
    names = P.manifest('MAIN')['names']
    R = P.load_v_rows('MAIN', TRAIN_ROLES if smoke else TRAIN_ROLES + ['V_CAL', 'V_SELECT'])
    role = smoke_roles(R['match'].tolist(), smoke_mod) if smoke else R['sub_role']
    tr = np.char.startswith(role.astype(str), 'fold')
    fold_of_row = np.full(len(role), -1)
    fold_of_row[tr] = [int(s[4:]) for s in role[tr].tolist()]
    fold_of = dict(zip(R['match'][tr].tolist(), fold_of_row[tr].tolist()))
    rows_of_match = {}
    for m in R['match'][tr].tolist():
        rows_of_match[m] = rows_of_match.get(m, 0) + 1
    se = role == 'V_SELECT'
    ca = role == 'V_CAL'
    out = {}
    for M in K.MODELS:
        man = C.read_json(base / f'v_models_manifest_{M}.json')
        sel = C.read_json(base / 'selection' / f'selection_{M}.json')
        r = dict(chosen=man['chosen'], selection_chosen_equal=sel['chosen'] == man['chosen'], adapters={})
        final_fit = sorted(fold_of)
        for aid, path, sha in [('final', man['final']['path'], man['final']['sha256'])] + \
                [(f'oof_{k}', man['oof_paths'][k], man['oof_sha256'][k]) for k in sorted(man['oof_paths'])]:
            ad = K.load_alt_adapter(base / path, sha, M)
            fr = ad.fit_record
            if aid == 'final':
                fit_set = final_fit
                held = None
            else:
                k = int(aid[-1])
                fit_set = sorted(m for m, f in fold_of.items() if f != k)
                held = k
            num = ad.base.named_steps['preprocess'].named_transformers_['numeric']
            ohe = ad.base.named_steps['preprocess'].named_transformers_['champions']
            fit_rows = sum(rows_of_match[m] for m in fit_set)
            cal = ad.calibrator
            a = dict(sha256_ok=True, model_version=ad.model_version, alt_model=ad.alt_model,
                     fit_set_sha_equals_reconstruction=C.sha256_json(fit_set) == fr['fit_match_ids_sha256'],
                     held_out_fold_ok=(held is None) or fr.get('held_out_fold') == held,
                     held_out_matches_absent_from_fit=(held is None) or not any(fold_of[m] == held for m in fit_set),
                     scaler_rows_equal_fit_rows=int(np.max(num.named_steps['scale'].n_samples_seen_)) == fit_rows == fr['fit_rows'],
                     onehot_handle_unknown_ignore=ohe.handle_unknown == 'ignore',
                     feature_names_equal_rule=ad.feature_names == K.model_features(M, names),
                     calibration=cal.kind, calibration_family_equals_chosen=cal.kind == man['chosen'],
                     calibration_rows_equal_v_cal=(cal.kind == 'raw') or (cal.fit_info['n'] == int(ca.sum())))
            a['pass'] = all(v for k2, v in a.items() if isinstance(v, bool))
            r['adapters'][aid] = a
            if aid == 'final':
                final_ad = ad
        with np.load(base / 'predictions' / f'v_train_oof_and_select_{M}.npz', allow_pickle=False) as z:
            sel_saved = z[f'select_final_{man["chosen"]}']
            sel_match = z['select_match']
            train_match = z['train_match']
            oof = z['train_oof_p']
            tfold = z['train_fold']
        repred = final_ad.predict_matrix(R['X'][se], names, C.STATE_VERSION)
        r['reload_select_predictions_equal'] = bool(np.array_equal(repred, sel_saved) and np.array_equal(sel_match, R['match'][se]))
        idx = np.flatnonzero(tr)[::53]
        k_ok = True
        for k in range(C.N_FOLDS):
            ad = K.load_alt_adapter(base / man['oof_paths'][f'fold{k}'], man['oof_sha256'][f'fold{k}'], M)
            sub = idx[fold_of_row[idx] == k]
            pos = np.searchsorted(np.flatnonzero(tr), sub)
            k_ok &= bool(np.array_equal(ad.predict_matrix(R['X'][sub], names, C.STATE_VERSION), oof[pos]))
            k_ok &= bool(np.all(tfold[pos] == k) and np.array_equal(train_match[pos], R['match'][sub]))
        r['reload_oof_predictions_equal_sample'] = k_ok
        r['converged_all'] = bool(man['final_fit']['converged'] and all(v['fit']['converged'] for v in man['folds'].values()))
        r['pass'] = r['selection_chosen_equal'] and all(a['pass'] for a in r['adapters'].values()) and \
            r['reload_select_predictions_equal'] and r['reload_oof_predictions_equal_sample']
        out[M] = r
    return dict(smoke=smoke, models=out, all_pass=all(v['pass'] for v in out.values()),
                checked_at=time.strftime('%Y-%m-%d %H:%M:%S'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    K.log_command()
    res = check_models(args.smoke)
    C.write_json(K.OUT / 'contracts' / f'v_model_checks_{"smoke" if args.smoke else "full"}.json', res)
    print({M: v['pass'] for M, v in res['models'].items()}, 'all_pass', res['all_pass'])
    return 0 if res['all_pass'] else 2


if __name__ == '__main__':
    sys.exit(main())
