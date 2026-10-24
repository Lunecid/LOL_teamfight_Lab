"""Label validity stage V: the two fixed comparator V families (B_reg, B_econ), no search.

For --model M: final base on ALL parent TRAIN bucket queries (exact primary keys), raw vs positive-slope sigmoid
(V_CAL fit), selection on V_SELECT (log loss, Brier, name), then five OOF fold adapters on the existing folds with
the chosen family (fold-specific V_CAL sigmoid if chosen). Outcome access is limited to TRAIN folds, V_CAL and
V_SELECT and is logged in this study's root. --smoke: TRAIN-only code-path run on a hashed TRAIN subset with pseudo
roles (folds 0-2 -> pseudo TRAIN with re-hashed folds, fold 3 -> pseudo V_CAL, fold 4 -> pseudo V_SELECT).
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
sys.path.insert(0, str(C.WT))

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

TRAIN_ROLES = [f'fold{k}' for k in range(C.N_FOLDS)]


def smoke_roles(match_ids, mod):
    """TRAIN-only pseudo roles on the hashed subset h8('lv-smoke-subset:',m) % mod == 0; others -> ''."""
    out = []
    for m in match_ids:
        if C.h8('lv-smoke-subset:', m) % mod:
            out.append('')
            continue
        f = C.train_fold(m)
        out.append('V_CAL' if f == 3 else 'V_SELECT' if f == 4 else f'fold{C.h8("lv-smoke-fold:", m) % C.N_FOLDS}')
    return np.asarray(out)


def fit_base(model_name, X, y, g, names, st, label):
    from sklearn.exceptions import ConvergenceWarning
    from train.state_value_experiment import logistic
    feats = K.model_features(model_name, names)
    keep = [names.index(c) for c in feats]
    Xf = X[:, keep]
    w = C.weights(g)
    attempts = []
    for max_iter in K.ALT_MAX_ITER_LADDER:
        model = logistic(feats, C=K.MODEL_SPECS[model_name]['C'])
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
        if n_iter < max_iter:
            break
        st.log(f'{label}: n_iter {n_iter} reached max_iter {max_iter}; repeating the same fit with a higher cap')
    ct = model.named_steps['preprocess']
    enc = {nm: [feats[i] for i in ix] for nm, _, ix in ct.transformers_ if nm in ('numeric', 'champions')}
    num = ct.named_transformers_['numeric']
    ohe = ct.named_transformers_['champions']
    lr = model.named_steps['model']
    if enc['champions'] != [f'participant_slot{i}_champion_id' for i in range(10)] or any(k.endswith('champion_id') for k in enc['numeric']):
        raise ValueError('champion IDs not treated as categorical')
    if type(ohe).__name__ != 'OneHotEncoder' or ohe.handle_unknown != 'ignore':
        raise ValueError('champion encoder is not one-hot with handle_unknown=ignore')
    fit_champs = [set(np.unique(Xf[:, feats.index(f'participant_slot{i}_champion_id')]).tolist()) for i in range(10)]
    rec = dict(label=label, model=model_name, rows=int(len(y)), matches=int(len(np.unique(g))), C=K.MODEL_SPECS[model_name]['C'],
               solver=lr.solver, random_state=lr.random_state, max_iter_final=lr.max_iter, attempts=attempts,
               n_iter=attempts[-1]['n_iter'], converged=attempts[-1]['converged'], n_features_in=len(feats),
               n_numeric_scaled=len(enc['numeric']), champion_one_hot=enc['champions'],
               champion_categories_per_slot=[len(c) for c in ohe.categories_], coef_count=int(lr.coef_.shape[1]),
               preprocessing_membership=dict(
                   scaler_n_samples_seen=int(np.max(num.named_steps['scale'].n_samples_seen_)),
                   imputer_statistics_sha256=C.sha256_json(np.round(num.named_steps['impute'].statistics_, 12).tolist()),
                   scaler_mean_sha256=C.sha256_json(np.round(num.named_steps['scale'].mean_, 12).tolist()),
                   onehot_categories_equal_fit_values=all(set(np.asarray(c).tolist()) == fit_champs[i] for i, c in enumerate(ohe.categories_)),
                   fit_rows=int(len(y))),
               weight_sum=float(w.sum()), positive_rate_match_weighted=float(np.average(y, weights=w)),
               fit_match_ids_sha256=C.sha256_json(sorted(set(np.asarray(g).tolist()))))
    if rec['preprocessing_membership']['scaler_n_samples_seen'] != len(y) or not rec['preprocessing_membership']['onehot_categories_equal_fit_values']:
        raise ValueError('preprocessing membership differs from fit rows')
    st.log(f'{label}: rows={rec["rows"]} matches={rec["matches"]} n_iter={rec["n_iter"]} converged={rec["converged"]} '
           f'secs={sum(a["seconds"] for a in attempts)}')
    return model, feats, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=K.MODELS, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-mod', type=int, default=12)
    args = ap.parse_args()
    M = args.model
    base = K.OUT / 'smoke_train_only' if args.smoke else K.OUT
    st = K.Status(f'fit_{M}' + ('_smoke' if args.smoke else ''), out=base)
    K.log_command()
    try:
        protocol_sha = C.sha256_file(K.OUT / 'protocol.json')
        if K.frozen_manifest_path().exists() and not args.smoke:
            raise SystemExit('frozen_manifest.json exists; comparator fits are closed')
        sel_path = base / 'selection' / f'selection_{M}.json'
        if sel_path.exists():
            raise SystemExit(f'{sel_path} exists; refusing to refit')
        P = K.ParentReadOnly()
        names = P.manifest('MAIN')['names']
        st.update('running', 'load_rows', next_step='fit final')
        if args.smoke:
            R = P.load_v_rows('MAIN', TRAIN_ROLES)
            pseudo = smoke_roles(R['match'].tolist(), args.smoke_mod)
            role = pseudo
            W = P.load_outcomes('MAIN', TRAIN_ROLES, purpose=f'TRAIN-only smoke fit {M} (pseudo roles)')
        else:
            R = P.load_v_rows('MAIN', TRAIN_ROLES + ['V_CAL', 'V_SELECT'])
            role = R['sub_role']
            W = P.load_outcomes('MAIN', TRAIN_ROLES + ['V_CAL', 'V_SELECT'],
                                purpose=f'{M}: fit (TRAIN folds), calibration (V_CAL), selection (V_SELECT)')
        if R['names'] != names:
            raise SystemExit('state names drift')
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
                cal_keys = (z['match'][z['is_bucket_sample'] == 1], z['query_ms'][z['is_bucket_sample'] == 1])
            key_check = dict(
                train_rows=int(tr.sum()), train_matches=int(len(np.unique(R['match'][tr]))),
                train_keys_equal_parent=bool(np.array_equal(R['match'][tr], pk['train_match']) and np.array_equal(R['query_ms'][tr], pk['train_query_ms'])),
                train_folds_equal_parent=bool(np.array_equal(fold[tr], pk['train_fold'])),
                select_keys_equal_parent=bool(np.array_equal(R['match'][se], pk['select_match']) and np.array_equal(R['query_ms'][se], pk['select_query_ms'])),
                cal_keys_equal_parent=bool(np.array_equal(R['match'][ca], cal_keys[0]) and np.array_equal(R['query_ms'][ca], cal_keys[1])),
                fold_equals_hash=bool(all(C.train_fold(m) == f for m, f in zip(R['match'][tr][::97].tolist(), fold[tr][::97].tolist()))))
            if (key_check['train_rows'], key_check['train_matches']) != (424160, 74168) or not all(
                    v for k, v in key_check.items() if isinstance(v, bool)):
                raise SystemExit(f'primary TRAIN/V_CAL/V_SELECT key identity failed: {key_check}')
        Xtr, ytr, gtr, ftr = R['X'][tr], y_all[tr], R['match'][tr], fold[tr]
        Xca, yca, gca = R['X'][ca], y_all[ca], R['match'][ca]
        Xse, yse, gse = R['X'][se], y_all[se], R['match'][se]
        census = {nm: dict(rows=int(len(g)), matches=int(len(np.unique(g)))) for nm, g in (('TRAIN', gtr), ('V_CAL', gca), ('V_SELECT', gse))}
        census['TRAIN_by_fold'] = {k: dict(rows=int((ftr == k).sum()), matches=int(len(np.unique(gtr[ftr == k])))) for k in range(C.N_FOLDS)}
        R = None
        st.update('running', 'fit_final', census=census, key_check=key_check, next_step='calibration candidates')

        basem, feats, rec_final = fit_base(M, Xtr, ytr, gtr, names, st, f'{M}_final_all_train')
        keep = [names.index(c) for c in feats]
        p_cal = basem.predict_proba(Xca[:, keep])[:, 1]
        sig = C.PositiveSlopeSigmoid().fit(p_cal, yca, C.weights(gca))
        common = dict(model=M, spec=K.MODEL_SPECS[M], fit_partition='TRAIN (15.14) all folds' + (' [SMOKE pseudo]' if args.smoke else ''),
                      fit_rows=rec_final['rows'], fit_matches=rec_final['matches'], fit_match_ids_sha256=rec_final['fit_match_ids_sha256'],
                      protocol_sha256=protocol_sha, smoke=args.smoke)
        cands = {'raw': K.AltWinProbV(M, names, feats, basem, C.IdentityCalibrator(),
                                      dict(common, adapter='final', calibration='raw (identity)', calibration_partition=None)),
                 'sigmoid_pos': K.AltWinProbV(M, names, feats, basem, sig,
                                              dict(common, adapter='final', calibration='sigmoid_pos', calibration_partition='V_CAL',
                                                   calibration_fit=sig.fit_info))}
        mdir = base / 'models' / M
        mdir.mkdir(parents=True, exist_ok=True)
        sel_scores, cand_hash, sel_preds, cal_preds = {}, {}, {}, {}
        for k, ad in cands.items():
            p = ad.predict_matrix(Xse, names, C.STATE_VERSION)
            sel_preds[k] = p
            cal_preds[k] = ad.predict_matrix(Xca, names, C.STATE_VERSION)
            sc = C.score(yse, p, C.weights(gse))
            sel_scores[k] = dict(sc, evaluation=C.evaluate(yse, p, gse))
            path = mdir / f'v_final_{k}.joblib'
            joblib.dump(ad, path)
            cand_hash[k] = C.sha256_file(path)
            re = K.load_alt_adapter(path, cand_hash[k], M)
            if not np.array_equal(re.predict_matrix(Xse, names, C.STATE_VERSION), p):
                raise ValueError('reloaded adapter predictions differ')
        chosen = min(cands, key=lambda k: (sel_scores[k]['logloss'], sel_scores[k]['brier'], k))
        selection = dict(role=K.ROLE_TAG, stage=f'{M} calibration selection', model=M, protocol_sha256=protocol_sha, smoke=args.smoke,
                         rule='lowest V_SELECT match-weighted log loss, then Brier, then name (parent order)', chosen=chosen,
                         select_scores={k: {m: v[m] for m in ('logloss', 'brier', 'auc')} for k, v in sel_scores.items()},
                         select_evaluation={k: v['evaluation'] for k, v in sel_scores.items()},
                         calibrate_scores_diagnostic={k: C.score(yca, cal_preds[k], C.weights(gca)) for k in cands},
                         candidate_sha256=cand_hash, chosen_path=str((mdir / f'v_final_{chosen}.joblib').relative_to(base)),
                         chosen_sha256=cand_hash[chosen], sigmoid_fit=sig.fit_info, final_fit=rec_final, census=census,
                         key_check=key_check, q_labels_used=False,
                         isotonic='not a candidate (parent protocol: flat regions can erase delta)',
                         written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                         written_before=['OOF fold adapters', 'freeze', 'any TEST/external W or generated label access'])
        sel_sha = C.write_json(sel_path, selection)
        st.update('running', 'selected', chosen=chosen, select_logloss={k: v['logloss'] for k, v in sel_scores.items()},
                  selection_sha256=sel_sha, next_step='OOF fold adapters')

        oof = np.full(len(ytr), np.nan)
        fold_recs, fold_hash, sel_fold = {}, {}, {}
        for k in range(C.N_FOLDS):
            fm = ftr != k
            b_k, feats_k, rec_k = fit_base(M, Xtr[fm], ytr[fm], gtr[fm], names, st, f'{M}_oof_fold{k}')
            if feats_k != feats:
                raise ValueError('fold feature drift')
            held = set(gtr[ftr == k].tolist())
            fit_set = set(gtr[fm].tolist())
            membership = dict(held_out_matches=len(held), fit_matches=len(fit_set), overlap=len(held & fit_set),
                              held_out_match_ids_sha256=C.sha256_json(sorted(held)), fit_match_ids_sha256=rec_k['fit_match_ids_sha256'],
                              fit_folds=sorted({int(x) for x in np.unique(ftr[fm])}))
            if membership['overlap'] or k in membership['fit_folds']:
                raise ValueError(f'fold {k} membership violated')
            if chosen == 'sigmoid_pos':
                cal_k = C.PositiveSlopeSigmoid().fit(b_k.predict_proba(Xca[:, keep])[:, 1], yca, C.weights(gca))
                cal_desc = dict(calibration='sigmoid_pos', calibration_partition='V_CAL', calibration_fit=cal_k.fit_info)
            else:
                cal_k = C.IdentityCalibrator()
                cal_desc = dict(calibration='raw (identity)', calibration_partition=None)
            ad = K.AltWinProbV(M, names, feats, b_k, cal_k, dict(
                model=M, adapter=f'oof_fold{k}', fit_partition=f'TRAIN folds {membership["fit_folds"]}', fit_rows=rec_k['rows'],
                fit_matches=rec_k['matches'], fit_match_ids_sha256=rec_k['fit_match_ids_sha256'], held_out_fold=k,
                protocol_sha256=protocol_sha, selection_sha256=sel_sha, smoke=args.smoke, **cal_desc))
            path = mdir / f'v_oof_fold{k}.joblib'
            joblib.dump(ad, path)
            fold_hash[k] = C.sha256_file(path)
            re = K.load_alt_adapter(path, fold_hash[k], M)
            hm = ftr == k
            oof[hm] = re.predict_matrix(Xtr[hm], names, C.STATE_VERSION)
            sel_fold[k] = re.predict_matrix(Xse, names, C.STATE_VERSION)
            fold_recs[k] = dict(fit=rec_k, membership=membership, calibration=cal_desc, sha256=fold_hash[k],
                                heldout_train_eval=C.evaluate(ytr[hm], oof[hm], gtr[hm], bins=False),
                                v_select_eval=C.evaluate(yse, sel_fold[k], gse, bins=False))
            st.update('running', 'oof_folds', processed=k + 1, total=C.N_FOLDS, fold_sha256=fold_hash[k], next_step='manifest')
        if not np.isfinite(oof).all():
            raise ValueError('incomplete OOF predictions')
        PF = np.stack([sel_fold[k] for k in range(C.N_FOLDS)])
        pred_sha = C.save_npz(base / 'predictions' / f'v_train_oof_and_select_{M}.npz', train_match=gtr, train_fold=ftr, train_oof_p=oof,
                              select_match=gse, select_final_raw=sel_preds['raw'], select_final_sigmoid_pos=sel_preds['sigmoid_pos'],
                              select_fold_p=PF, cal_match=gca, cal_final_raw=cal_preds['raw'], cal_final_sigmoid_pos=cal_preds['sigmoid_pos'],
                              model=np.asarray(M), chosen=np.asarray(chosen),
                              role=np.asarray('comparator V predictions on TRAIN (held-out fold), V_SELECT and V_CAL; outcomes not included'))
        manifest = dict(role=K.ROLE_TAG, model=M, spec=K.MODEL_SPECS[M], protocol_sha256=protocol_sha, selection_sha256=sel_sha,
                        chosen=chosen, smoke=args.smoke,
                        final=dict(path=str((mdir / f'v_final_{chosen}.joblib').relative_to(base)), sha256=cand_hash[chosen]),
                        candidates_sha256=cand_hash, oof_sha256={f'fold{k}': fold_hash[k] for k in range(C.N_FOLDS)},
                        oof_paths={f'fold{k}': str((mdir / f'v_oof_fold{k}.joblib').relative_to(base)) for k in range(C.N_FOLDS)},
                        model_version=f'lv20260915_{M}_wp_v2_participant_order', state_version=C.STATE_VERSION,
                        state_names_sha256=C.sha256_json(names), feature_names=feats, feature_names_sha256=C.sha256_json(feats),
                        folds=fold_recs, final_fit=rec_final, census=census, key_check=key_check,
                        heldout_train_oof_eval=C.evaluate(ytr, oof, gtr), final_v_select_eval=sel_scores[chosen]['evaluation'],
                        fold_dispersion_on_v_select=dict(mean_across_row_sd=float(PF.std(axis=0).mean()),
                                                         max_across_row_sd=float(PF.std(axis=0).max())),
                        predictions_sha256=pred_sha, completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                        note='comparator for label sensitivity; cross-fitting of generated targets, not proof of semantic truth')
        C.write_json(base / f'v_models_manifest_{M}.json', manifest)
        st.update('complete', f'fit_{M}', chosen=chosen, oof_auc=manifest['heldout_train_oof_eval']['auc'],
                  oof_logloss=manifest['heldout_train_oof_eval']['logloss'],
                  converged_all=bool(rec_final['converged'] and all(v['fit']['converged'] for v in fold_recs.values())),
                  next_step='freeze after both comparators')
        return 0
    except SystemExit as exc:
        st.update('failed', f'fit_{M}', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', f'fit_{M}', error=repr(exc), next_step='fix and rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
