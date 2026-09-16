"""Objective-channel ablation contract: independent checks of the fitted B_noobj adapters (smoke or full).

Reconstructs fit membership per adapter from the fold hash (or smoke pseudo roles), verifies hashes/versions, scaler
rows, calibration partition, that B inputs are exactly A's inputs restricted to the retained columns (same state
indices and values), reload identity of saved predictions, and the unknown_objective_team_count TRAIN prevalence.
No outcome W is read.
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
import lv20260915_analysis as A  # noqa: E402
import oc20260915_common as K  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime')

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

TRAIN_ROLES = [f'fold{k}' for k in range(C.N_FOLDS)]


def check_models(smoke, smoke_mod=12):
    from oc20260915_fit_v import smoke_roles
    base = K.OUT / 'smoke_train_only' if smoke else K.OUT
    P = K.parent_reader()
    names = P.manifest('MAIN')['names']
    R = P.load_v_rows('MAIN', TRAIN_ROLES if smoke else TRAIN_ROLES + ['V_CAL', 'V_SELECT'])
    role = smoke_roles(R['match'].tolist(), smoke_mod) if smoke else R['sub_role']
    tr = np.char.startswith(role.astype(str), 'fold')
    fold_row = np.full(len(role), -1)
    fold_row[tr] = [int(s[4:]) for s in role[tr].tolist()]
    fold_of = dict(zip(R['match'][tr].tolist(), fold_row[tr].tolist()))
    rows_of = {}
    for m in R['match'][tr].tolist():
        rows_of[m] = rows_of.get(m, 0) + 1
    se, ca = role == 'V_SELECT', role == 'V_CAL'
    man = C.read_json(base / 'v_models_manifest_B_noobj.json')
    sel = C.read_json(base / 'selection' / 'selection_secondary_calibration.json')
    ev = C.read_json(K.OUT / 'feature_evidence.json')
    vman = C.read_json(K.FC / 'v_models_manifest.json')
    A_ad = C.load_v_adapter(K.FC / vman['final']['path'], vman['final']['sha256'])
    out = dict(smoke=smoke, secondary_chosen=man['secondary']['chosen'], selection_equal=sel['secondary_chosen'] == man['secondary']['chosen'],
               primary_family=man['primary']['family'], adapters={})
    for fam, fd in man['families'].items():
        items = [('final', fd['final_path'], fd['final_sha256'])] + [(f'oof_{k}', fd['oof_paths'][k], fd['oof_sha256'][k]) for k in sorted(fd['oof_paths'])]
        for aid, path, sha in items:
            ad = K.load_b_adapter(base / path, sha)
            fr = ad.fit_record
            held = None if aid == 'final' else int(aid[-1])
            fit_set = sorted(fold_of) if held is None else sorted(m for m, f in fold_of.items() if f != held)
            num = ad.base.named_steps['preprocess'].named_transformers_['numeric']
            fit_rows = sum(rows_of[m] for m in fit_set)
            a = dict(fit_set_sha_equals_reconstruction=C.sha256_json(fit_set) == fr['fit_match_ids_sha256'],
                     held_out_ok=held is None or (fr.get('held_out_fold') == held and not any(fold_of[m] == held for m in fit_set)),
                     scaler_rows_equal_fit_rows=int(np.max(num.named_steps['scale'].n_samples_seen_)) == fit_rows == fr['fit_rows'],
                     feature_names_equal_evidence=ad.feature_names == ev['retained_ordered'],
                     retained_state_indices_equal_A=ad.keep == [A_ad.keep[A_ad.feature_names.index(n)] for n in ad.feature_names],
                     estimator_C_equals_A=ad.base.named_steps['model'].C == A_ad.base.named_steps['model'].C == C.V_C,
                     solver_equal_A=ad.base.named_steps['model'].solver == A_ad.base.named_steps['model'].solver,
                     calibration_kind=ad.calibrator.kind == fam,
                     calibration_rows_equal_v_cal=fam == 'raw' or ad.calibrator.fit_info['n'] == int(ca.sum()))
            a['pass'] = all(a.values())
            out['adapters'][f'{fam}/{aid}'] = a
    # retained values identical to A inputs on every TRAIN row
    b_final = K.load_b_adapter(base / man['primary']['final_path'], man['primary']['final_sha256'])
    XA = R['X'][:, A_ad.keep]
    XB = R['X'][:, b_final.keep]
    idx = [A_ad.feature_names.index(n) for n in b_final.feature_names]
    out['retained_values_identical_to_A_inputs'] = bool(np.array_equal(XB, XA[:, idx]))
    out['dropped_columns_absent'] = not any(K.dropped_token(n) for n in b_final.feature_names)
    with np.load(base / 'predictions' / 'v_train_oof_and_select_B_noobj.npz', allow_pickle=False) as z:
        Z = {k: z[k] for k in z.files}
    out['reload_select_raw_equal'] = bool(np.array_equal(b_final.predict_matrix(R['X'][se], names, C.STATE_VERSION), Z['select_final_raw'])
                                          and np.array_equal(Z['select_match'], R['match'][se]))
    tri = np.flatnonzero(tr)
    samp = tri[::53]
    ok = True
    for k in range(C.N_FOLDS):
        ad = K.load_b_adapter(base / man['primary']['oof_paths'][f'fold{k}'], man['primary']['oof_sha256'][f'fold{k}'])
        sub = samp[fold_row[samp] == k]
        pos = np.searchsorted(tri, sub)
        ok &= bool(np.array_equal(ad.predict_matrix(R['X'][sub], names, C.STATE_VERSION), Z['train_oof_p_raw'][pos]) and np.all(Z['train_fold'][pos] == k))
    out['reload_oof_raw_equal_sample'] = ok
    u = R['X'][tr][:, names.index(K.UNKNOWN_COL)] > 0
    out['unknown_objective_team_count_TRAIN_bucket'] = dict(rows=int(tr.sum()), nonzero_rows=int(u.sum()),
                                                            nonzero_row_share=float(u.mean()),
                                                            nonzero_match_weighted=A.wrate(u, R['match'][tr])[1],
                                                            max=float(R['X'][tr][:, names.index(K.UNKNOWN_COL)].max()))
    out['converged_all'] = man['converged_all']
    out['all_pass'] = bool(out['selection_equal'] and out['primary_family'] == 'raw' and all(a['pass'] for a in out['adapters'].values())
                           and out['retained_values_identical_to_A_inputs'] and out['dropped_columns_absent'] and out['reload_select_raw_equal']
                           and out['reload_oof_raw_equal_sample'])
    out['checked_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    K.log_command()
    res = check_models(args.smoke)
    C.write_json(K.OUT / 'contracts' / f'v_model_checks_{"smoke" if args.smoke else "full"}.json', res)
    print('all_pass', res['all_pass'], 'adapters', sum(a['pass'] for a in res['adapters'].values()), len(res['adapters']),
          res['retained_values_identical_to_A_inputs'], res['unknown_objective_team_count_TRAIN_bucket'])
    return 0 if res['all_pass'] else 2


if __name__ == '__main__':
    sys.exit(main())
