"""Objective-channel ablation stage V: B_noobj final + 5 OOF adapters (primary raw; secondary sigmoid family).

Same estimator, C=.01, equal-match weights, TRAIN-only preprocessing and folds as A; columns = A columns minus the 176
named objective columns (feature_evidence.json). Primary = raw. Secondary: positive-slope sigmoid on V_CAL for the
final base and every fold base; raw vs sigmoid chosen on V_SELECT (log loss, Brier, name) and recorded before the OOF
adapters. Outcome access limited to TRAIN folds, V_CAL, V_SELECT (logged in this root). --smoke: TRAIN-only pseudo roles.
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
import oc20260915_common as K  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime')
sys.path.insert(0, str(C.WT))

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

TRAIN_ROLES = [f'fold{k}' for k in range(C.N_FOLDS)]
FAMILIES = ('raw', 'sigmoid_pos')


def smoke_roles(match_ids, mod):
    out = []
    for m in match_ids:
        if C.h8('oc-smoke-subset:', m) % mod:
            out.append('')
            continue
        f = C.train_fold(m)
        out.append('V_CAL' if f == 3 else 'V_SELECT' if f == 4 else f'fold{C.h8("oc-smoke-fold:", m) % C.N_FOLDS}')
    return np.asarray(out)


def retained_features(names):
    ev = C.read_json(K.OUT / 'feature_evidence.json')
    feats = ev['retained_ordered']
    if C.sha256_json(feats) != ev['retained_sha256'] or len(feats) != K.EXPECTED['retained']:
        raise SystemExit('feature evidence hash/count mismatch')
    if K.feature_split([n for n in names if n != 'snapshot_age_s'])[0] != feats:
        raise SystemExit('retained list differs from the token rule on current state names')
    return feats


def fit_base(X, y, g, names, feats, st, label):
    from sklearn.exceptions import ConvergenceWarning
    from train.state_value_experiment import logistic
    keep = [names.index(c) for c in feats]
    Xf = X[:, keep]
    w = C.weights(g)
    attempts = []
    for max_iter in K.MAX_ITER_LADDER:
        model = logistic(feats, C=C.V_C)
        model.set_params(model__max_iter=max_iter)
        t0 = time.time()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            model.fit(Xf, y, model__sample_weight=w)
        lr = model.named_steps['model']
        n_iter = int(np.max(lr.n_iter_))
        attempts.append(dict(max_iter=max_iter, n_iter=n_iter, seconds=round(time.time() - t0, 1), converged=n_iter < max_iter,
                             convergence_warnings=sorted({str(x.message)[:200] for x in wl if issubclass(x.category, ConvergenceWarning)}),
                             other_warnings=sorted({f'{x.category.__name__}: {str(x.message)[:160]}' for x in wl
                                                    if not issubclass(x.category, ConvergenceWarning)})[:10]))
        st.log(f'{label}: attempt max_iter={max_iter} n_iter={n_iter} converged={n_iter < max_iter} secs={attempts[-1]["seconds"]}')
        if n_iter < max_iter:
            break
    ct = model.named_steps['preprocess']
    enc = {nm: [feats[i] for i in ix] for nm, _, ix in ct.transformers_ if nm in ('numeric', 'champions')}
    num = ct.named_transformers_['numeric']
    ohe = ct.named_transformers_['champions']
    lr = model.named_steps['model']
    if enc['champions'] != [f'participant_slot{i}_champion_id' for i in range(10)] or any(k.endswith('champion_id') for k in enc['numeric']):
        raise ValueError('champion IDs not categorical')
    if ohe.handle_unknown != 'ignore' or lr.solver != 'liblinear' or lr.C != C.V_C or lr.random_state != 7:
        raise ValueError('estimator differs from A family')
    fit_champs = [set(np.unique(Xf[:, feats.index(f'participant_slot{i}_champion_id')]).tolist()) for i in range(10)]
    rec = dict(label=label, rows=int(len(y)), matches=int(len(np.unique(g))), C=lr.C, solver=lr.solver, random_state=lr.random_state,
               max_iter_final=lr.max_iter, attempts=attempts, n_iter=attempts[-1]['n_iter'], converged=attempts[-1]['converged'],
               n_features_in=len(feats), n_numeric_scaled=len(enc['numeric']), champion_categories_per_slot=[len(c) for c in ohe.categories_],
               coef_count=int(lr.coef_.shape[1]),
               preprocessing_membership=dict(scaler_n_samples_seen=int(np.max(num.named_steps['scale'].n_samples_seen_)), fit_rows=int(len(y)),
                                             onehot_categories_equal_fit_values=all(set(np.asarray(c).tolist()) == fit_champs[i] for i, c in enumerate(ohe.categories_))),
               weight_sum=float(w.sum()), positive_rate_match_weighted=float(np.average(y, weights=w)),
               fit_match_ids_sha256=C.sha256_json(sorted(set(np.asarray(g).tolist()))))
    if rec['preprocessing_membership']['scaler_n_samples_seen'] != len(y) or not rec['preprocessing_membership']['onehot_categories_equal_fit_values']:
        raise ValueError('preprocessing membership differs from fit rows')
    return model, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-mod', type=int, default=12)
    args = ap.parse_args()
    base = K.OUT / 'smoke_train_only' if args.smoke else K.OUT
    st = K.Status('fit_B_noobj' + ('_smoke' if args.smoke else ''), out=base)
    K.log_command()
    try:
        protocol_sha = C.sha256_file(K.OUT / 'protocol.json')
        if K.frozen_manifest_path().exists() and not args.smoke:
            raise SystemExit('frozen_manifest.json exists; fits are closed')
        sel_path = base / 'selection' / 'selection_secondary_calibration.json'
        if sel_path.exists():
            raise SystemExit(f'{sel_path} exists; refusing to refit')
        P = K.parent_reader()
        names = P.manifest('MAIN')['names']
        feats = retained_features(names)
        st.update('running', 'load_rows', n_features=len(feats), next_step='fit final')
        if args.smoke:
            R = P.load_v_rows('MAIN', TRAIN_ROLES)
            role = smoke_roles(R['match'].tolist(), args.smoke_mod)
            W = P.load_outcomes('MAIN', TRAIN_ROLES, purpose='B_noobj TRAIN-only smoke fit (pseudo roles)')
        else:
            R = P.load_v_rows('MAIN', TRAIN_ROLES + ['V_CAL', 'V_SELECT'])
            role = R['sub_role']
            W = P.load_outcomes('MAIN', TRAIN_ROLES + ['V_CAL', 'V_SELECT'], purpose='B_noobj fit (TRAIN), sigmoid fit (V_CAL), secondary selection (V_SELECT)')
        y_all = np.asarray([W[m][0] for m in R['match'].tolist()], dtype=np.int64)
        if not np.isin(y_all, [0, 1]).all():
            raise SystemExit('invalid outcomes')
        tr = np.char.startswith(role.astype(str), 'fold')
        ca = role == 'V_CAL'
        se = role == 'V_SELECT'
        fold = np.full(len(role), -1)
        fold[tr] = [int(s[4:]) for s in role[tr].tolist()]
        key_check = {}
        if not args.smoke:
            with np.load(K.FC / 'predictions' / 'v_train_oof_and_select.npz', allow_pickle=False) as z:
                pk = {k: z[k] for k in ('train_match', 'train_query_ms', 'train_fold', 'select_match', 'select_query_ms')}
            with np.load(K.FC / 'eval' / 'predictions' / 'v_MAIN_V_CAL.npz', allow_pickle=False) as z:
                b = z['is_bucket_sample'] == 1
                cal_keys = (z['match'][b], z['query_ms'][b])
            key_check = dict(train_rows=int(tr.sum()), train_matches=int(len(np.unique(R['match'][tr]))),
                             train_keys_equal_parent=bool(np.array_equal(R['match'][tr], pk['train_match']) and np.array_equal(R['query_ms'][tr], pk['train_query_ms'])),
                             train_folds_equal_parent=bool(np.array_equal(fold[tr], pk['train_fold'])),
                             select_keys_equal_parent=bool(np.array_equal(R['match'][se], pk['select_match']) and np.array_equal(R['query_ms'][se], pk['select_query_ms'])),
                             cal_keys_equal_parent=bool(np.array_equal(R['match'][ca], cal_keys[0]) and np.array_equal(R['query_ms'][ca], cal_keys[1])))
            if (key_check['train_rows'], key_check['train_matches']) != (424160, 74168) or not all(v for v in key_check.values() if isinstance(v, bool)):
                raise SystemExit(f'primary key identity failed: {key_check}')
        Xtr, ytr, gtr, ftr = R['X'][tr], y_all[tr], R['match'][tr], fold[tr]
        Xca, yca, gca = R['X'][ca], y_all[ca], R['match'][ca]
        Xse, yse, gse = R['X'][se], y_all[se], R['match'][se]
        census = {nm: dict(rows=int(len(g)), matches=int(len(np.unique(g)))) for nm, g in (('TRAIN', gtr), ('V_CAL', gca), ('V_SELECT', gse))}
        R = None
        st.update('running', 'fit_final', census=census, key_check=key_check, next_step='calibration')
        keep = [names.index(c) for c in feats]
        basem, rec_final = fit_base(Xtr, ytr, gtr, names, feats, st, 'B_noobj_final_all_train')
        sig = C.PositiveSlopeSigmoid().fit(basem.predict_proba(Xca[:, keep])[:, 1], yca, C.weights(gca))
        common = dict(model=K.MODEL, fit_partition='TRAIN (15.14) all folds' + (' [SMOKE pseudo]' if args.smoke else ''), fit_rows=rec_final['rows'],
                      fit_matches=rec_final['matches'], fit_match_ids_sha256=rec_final['fit_match_ids_sha256'], protocol_sha256=protocol_sha, smoke=args.smoke)
        finals = {'raw': K.NoObjWinProbV(names, feats, basem, C.IdentityCalibrator(), dict(common, adapter='final', calibration='raw (identity)', role='PRIMARY')),
                  'sigmoid_pos': K.NoObjWinProbV(names, feats, basem, sig, dict(common, adapter='final', calibration='sigmoid_pos', calibration_partition='V_CAL',
                                                                                   calibration_fit=sig.fit_info, role='SECONDARY candidate'))}
        mdir = base / 'models' / K.MODEL
        mdir.mkdir(parents=True, exist_ok=True)
        sel_scores, final_sha, sel_preds, cal_preds = {}, {}, {}, {}
        for fam, ad in finals.items():
            p = ad.predict_matrix(Xse, names, C.STATE_VERSION)
            sel_preds[fam] = p
            cal_preds[fam] = ad.predict_matrix(Xca, names, C.STATE_VERSION)
            sel_scores[fam] = dict(C.score(yse, p, C.weights(gse)), evaluation=C.evaluate(yse, p, gse))
            path = mdir / f'v_final_{fam}.joblib'
            joblib.dump(ad, path)
            final_sha[fam] = C.sha256_file(path)
            if not np.array_equal(K.load_b_adapter(path, final_sha[fam]).predict_matrix(Xse, names, C.STATE_VERSION), p):
                raise ValueError('reload differs')
        chosen = min(FAMILIES, key=lambda k: (sel_scores[k]['logloss'], sel_scores[k]['brier'], k))
        selection = dict(role=K.ROLE_TAG, stage='SECONDARY calibration selection (primary stays raw)', protocol_sha256=protocol_sha, smoke=args.smoke,
                         rule='lowest V_SELECT match-weighted log loss, then Brier, then name', secondary_chosen=chosen, primary='raw',
                         select_scores={k: {m: v[m] for m in ('logloss', 'brier', 'auc')} for k, v in sel_scores.items()},
                         select_evaluation={k: v['evaluation'] for k, v in sel_scores.items()},
                         calibrate_scores_diagnostic={k: C.score(yca, cal_preds[k], C.weights(gca)) for k in FAMILIES},
                         final_sha256=final_sha, sigmoid_fit=sig.fit_info, final_fit=rec_final, census=census, key_check=key_check,
                         written_at=time.strftime('%Y-%m-%d %H:%M:%S'), written_before=['OOF adapters', 'freeze', 'TEST/external access'])
        sel_sha = C.write_json(sel_path, selection)
        st.update('running', 'secondary_selected', secondary_chosen=chosen, select_logloss={k: v['logloss'] for k, v in sel_scores.items()}, next_step='OOF')
        oof = {fam: np.full(len(ytr), np.nan) for fam in FAMILIES}
        fold_recs, oof_sha, sel_fold = {}, {fam: {} for fam in FAMILIES}, {}
        for k in range(C.N_FOLDS):
            fm = ftr != k
            b_k, rec_k = fit_base(Xtr[fm], ytr[fm], gtr[fm], names, feats, st, f'B_noobj_oof_fold{k}')
            held = set(gtr[ftr == k].tolist())
            fit_set = set(gtr[fm].tolist())
            membership = dict(held_out_matches=len(held), fit_matches=len(fit_set), overlap=len(held & fit_set),
                              held_out_match_ids_sha256=C.sha256_json(sorted(held)), fit_match_ids_sha256=rec_k['fit_match_ids_sha256'],
                              fit_folds=sorted({int(x) for x in np.unique(ftr[fm])}))
            if membership['overlap'] or k in membership['fit_folds']:
                raise ValueError(f'fold {k} membership violated')
            sig_k = C.PositiveSlopeSigmoid().fit(b_k.predict_proba(Xca[:, keep])[:, 1], yca, C.weights(gca))
            cals = {'raw': (C.IdentityCalibrator(), dict(calibration='raw (identity)', role='PRIMARY')),
                    'sigmoid_pos': (sig_k, dict(calibration='sigmoid_pos', calibration_partition='V_CAL', calibration_fit=sig_k.fit_info,
                                                role='SECONDARY' if chosen == 'sigmoid_pos' else 'SECONDARY candidate not chosen (sign-invariance check only)'))}
            hm = ftr == k
            fold_recs[k] = dict(fit=rec_k, membership=membership, calibration_sigmoid=sig_k.fit_info)
            for fam, (cal, desc) in cals.items():
                ad = K.NoObjWinProbV(names, feats, b_k, cal, dict(model=K.MODEL, adapter=f'oof_fold{k}', fit_partition=f'TRAIN folds {membership["fit_folds"]}',
                                                                  fit_rows=rec_k['rows'], fit_matches=rec_k['matches'], fit_match_ids_sha256=rec_k['fit_match_ids_sha256'],
                                                                  held_out_fold=k, protocol_sha256=protocol_sha, selection_sha256=sel_sha, smoke=args.smoke, **desc))
                path = mdir / f'v_oof_fold{k}_{fam}.joblib'
                joblib.dump(ad, path)
                oof_sha[fam][f'fold{k}'] = C.sha256_file(path)
                re = K.load_b_adapter(path, oof_sha[fam][f'fold{k}'])
                oof[fam][hm] = re.predict_matrix(Xtr[hm], names, C.STATE_VERSION)
                if fam == 'raw':
                    sel_fold[k] = re.predict_matrix(Xse, names, C.STATE_VERSION)
            fold_recs[k]['heldout_train_eval_raw'] = C.evaluate(ytr[hm], oof['raw'][hm], gtr[hm], bins=False)
            st.update('running', 'oof_folds', processed=k + 1, total=C.N_FOLDS, next_step='manifest')
        if not all(np.isfinite(v).all() for v in oof.values()):
            raise ValueError('incomplete OOF predictions')
        PF = np.stack([sel_fold[k] for k in range(C.N_FOLDS)])
        pred_sha = C.save_npz(base / 'predictions' / 'v_train_oof_and_select_B_noobj.npz', train_match=gtr, train_fold=ftr,
                              train_oof_p_raw=oof['raw'], train_oof_p_sigmoid_pos=oof['sigmoid_pos'], select_match=gse,
                              select_final_raw=sel_preds['raw'], select_final_sigmoid_pos=sel_preds['sigmoid_pos'], select_fold_p_raw=PF,
                              cal_match=gca, cal_final_raw=cal_preds['raw'], cal_final_sigmoid_pos=cal_preds['sigmoid_pos'],
                              role=np.asarray('B_noobj predictions (outcomes not included)'))
        rel = lambda p: str(p.relative_to(base))  # noqa: E731
        manifest = dict(role=K.ROLE_TAG, model=K.MODEL, model_version=K.MODEL_VERSION, protocol_sha256=protocol_sha, selection_sha256=sel_sha,
                        primary=dict(family='raw', final_path=rel(mdir / 'v_final_raw.joblib'), final_sha256=final_sha['raw'],
                                     oof_paths={f'fold{k}': rel(mdir / f'v_oof_fold{k}_raw.joblib') for k in range(C.N_FOLDS)}, oof_sha256=oof_sha['raw']),
                        secondary=dict(chosen=chosen, final_path=rel(mdir / f'v_final_{chosen}.joblib'), final_sha256=final_sha[chosen],
                                       oof_paths={f'fold{k}': rel(mdir / f'v_oof_fold{k}_{chosen}.joblib') for k in range(C.N_FOLDS)}, oof_sha256=oof_sha[chosen]),
                        families={fam: dict(final_sha256=final_sha[fam], oof_sha256=oof_sha[fam],
                                            oof_paths={f'fold{k}': rel(mdir / f'v_oof_fold{k}_{fam}.joblib') for k in range(C.N_FOLDS)},
                                            final_path=rel(mdir / f'v_final_{fam}.joblib')) for fam in FAMILIES},
                        state_version=C.STATE_VERSION, state_names_sha256=C.sha256_json(names), feature_names=feats, feature_names_sha256=C.sha256_json(feats),
                        folds=fold_recs, final_fit=rec_final, census=census, key_check=key_check,
                        heldout_train_oof_eval_raw=C.evaluate(ytr, oof['raw'], gtr), final_v_select_eval_raw=sel_scores['raw']['evaluation'],
                        converged_all=bool(rec_final['converged'] and all(v['fit']['converged'] for v in fold_recs.values())),
                        predictions_sha256=pred_sha, completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        C.write_json(base / 'v_models_manifest_B_noobj.json', manifest)
        st.update('complete', 'fit_B_noobj', secondary_chosen=chosen, converged_all=manifest['converged_all'],
                  oof_auc_raw=manifest['heldout_train_oof_eval_raw']['auc'], next_step='independent model checks, freeze')
        return 0
    except SystemExit as exc:
        st.update('failed', 'fit_B_noobj', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'fit_B_noobj', error=repr(exc), next_step='fix and rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
