"""Full-corpus stage V: final V, calibration choice (V_CAL/V_SELECT), five fresh OOF fold adapters.

Primary V family: expanded StateV2 logistic (train.state_value_experiment.logistic, C=.01 predeclared, liblinear),
equal total weight per match, fitted on ALL eligible 15.14 TRAIN bucket queries. Candidates raw (identity) and
positive-slope sigmoid fitted on V_CAL only; choice by V_SELECT match-weighted log loss, then Brier, then name.
selection_v.json (with adapter hashes) is written before any OOF adapter, label, q model or TEST access.
Fold-k adapters are fitted only on TRAIN matches of the other four folds; the selected calibration family is
applied uniformly (per-fold sigmoid fitted on V_CAL). Nothing here reads TEST/Q_CAL/Q_SELECT/external outcomes.
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

sys.path.insert(0, str(C.WT))
os.environ['LOL_OUTPUT_ROOT'] = str(C.OUT / 'runtime')

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

CHAMPION_COLUMNS = tuple(f'participant_slot{i}_champion_id' for i in range(10))


def fit_base(X, y, g, names, st, label):
    from train.state_value_experiment import logistic
    from train.temporal_winprob import feature_matrix
    from sklearn.exceptions import ConvergenceWarning
    Xf, cols = feature_matrix(X, names, 'expanded')
    if 'snapshot_age_s' in cols or len(cols) != len(names) - 1:
        raise ValueError('expanded feature schema unexpected')
    model = logistic(cols, C=C.V_C)
    w = C.weights(g)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        model.fit(Xf, y, model__sample_weight=w)
    secs = time.time() - t0
    ct = model.named_steps['preprocess']
    enc = {nm: [cols[i] for i in ix] for nm, _, ix in ct.transformers_ if nm in ('numeric', 'champions')}
    lr = model.named_steps['model']
    n_iter = int(np.max(lr.n_iter_))
    rec = dict(label=label, rows=int(len(y)), matches=int(len(np.unique(g))), seconds=round(secs, 1),
               C=C.V_C, solver=lr.solver, max_iter=lr.max_iter, n_iter=n_iter, converged=n_iter < lr.max_iter,
               convergence_warnings=sorted({str(x.message)[:200] for x in wl if issubclass(x.category, ConvergenceWarning)}),
               other_warnings=sorted({f'{x.category.__name__}: {str(x.message)[:160]}' for x in wl
                                      if not issubclass(x.category, ConvergenceWarning)})[:10],
               n_features_in=len(cols), n_numeric_scaled=len(enc['numeric']), champion_one_hot=enc['champions'],
               champion_categories_per_slot=[len(c) for c in ct.named_transformers_['champions'].categories_],
               weight_sum=float(w.sum()), weight_mean=float(w.mean()),
               positive_rate_match_weighted=float(np.average(y, weights=w)),
               fit_match_ids_sha256=C.sha256_json(sorted(set(np.asarray(g).tolist()))))
    if enc['champions'] != list(CHAMPION_COLUMNS) or any(k.endswith('champion_id') for k in enc['numeric']):
        raise ValueError('champion IDs not treated as categorical')
    if type(ct.named_transformers_['champions']).__name__ != 'OneHotEncoder':
        raise ValueError('champion encoder is not one-hot')
    st.log(f'{label}: fitted rows={rec["rows"]} matches={rec["matches"]} secs={rec["seconds"]} n_iter={n_iter} '
           f'conv_warn={len(rec["convergence_warnings"])}')
    return model, cols, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='TRAIN-only smoke (pseudo roles from TRAIN folds)')
    args = ap.parse_args()
    L = D.Layout(args.smoke)
    out = L.base
    st = C.Status(out, 'fit_v')
    try:
        st.update('running', 'load', next_step='fit final V')
        man = L.manifest('MAIN')
        protocol_sha = C.sha256_file(C.OUT / 'protocol.json') if (C.OUT / 'protocol.json').exists() else None
        if not args.smoke and man['protocol_sha256'] != protocol_sha:
            raise SystemExit('extraction manifest protocol hash differs from protocol.json')
        if (out / 'selection_v.json').exists():
            raise SystemExit('selection_v.json exists; refusing to refit (use a new output root)')
        names = man['names']
        train_roles = [f'fold{k}' for k in range(C.N_FOLDS)]
        TR = D.load_v_rows(L, 'MAIN', train_roles)
        CA = D.load_v_rows(L, 'MAIN', ['V_CAL'])
        SE = D.load_v_rows(L, 'MAIN', ['V_SELECT'])
        if TR['names'] != names:
            raise SystemExit('names drift')
        W = D.load_outcomes(L, 'MAIN', train_roles + ['V_CAL', 'V_SELECT'], purpose='V fit (TRAIN), V calibration (V_CAL), V selection (V_SELECT)')

        def target(R):
            y = np.asarray([W[m][0] for m in R['match'].tolist()], dtype=np.int64)
            if not np.isin(y, [0, 1]).all():
                raise ValueError('invalid outcome in V rows')
            return y
        yTR, yCA, ySE = target(TR), target(CA), target(SE)
        fold = np.asarray([int(s[4:]) for s in TR['sub_role'].tolist()])
        census = {nm: dict(rows=int(len(R['match'])), matches=int(len(np.unique(R['match']))),
                           rows_per_match=round(len(R['match']) / max(1, len(np.unique(R['match']))), 3))
                  for nm, R in (('TRAIN', TR), ('V_CAL', CA), ('V_SELECT', SE))}
        census['TRAIN_by_fold'] = {k: dict(rows=int((fold == k).sum()), matches=int(len(np.unique(TR['match'][fold == k]))))
                                   for k in range(C.N_FOLDS)}
        st.update('running', 'fit_final_base', census=census, next_step='calibration candidates')

        # ---------------- final base + calibration candidates
        base, cols, rec_final = fit_base(TR['X'], yTR, TR['match'], names, st, 'final_all_train')
        from train.temporal_winprob import feature_matrix
        keep = [names.index(c) for c in cols]
        p_cal = base.predict_proba(CA['X'][:, keep])[:, 1]
        sig = C.PositiveSlopeSigmoid().fit(p_cal, yCA, C.weights(CA['match']))
        fit_common = dict(fit_partition='TRAIN (15.14) all folds', fit_rows=rec_final['rows'], fit_matches=rec_final['matches'],
                          fit_match_ids_sha256=rec_final['fit_match_ids_sha256'], protocol_sha256=protocol_sha, smoke=args.smoke)
        cands = {
            'raw': C.FullCorpusWinProbV2(names, cols, base, C.IdentityCalibrator(),
                                         dict(fit_common, adapter='final', calibration='raw (identity)', calibration_partition=None)),
            'sigmoid_pos': C.FullCorpusWinProbV2(names, cols, base, sig,
                                                 dict(fit_common, adapter='final', calibration='sigmoid_pos',
                                                      calibration_partition='V_CAL', calibration_fit=sig.fit_info)),
        }
        mdir = out / 'models' / 'v'
        mdir.mkdir(parents=True, exist_ok=True)
        sel_scores, cand_hash, sel_preds = {}, {}, {}
        for k, ad in cands.items():
            p = ad.predict_matrix(SE['X'], names, C.STATE_VERSION)
            sel_preds[k] = p
            w = C.weights(SE['match'])
            sc = C.score(ySE, p, w)
            sel_scores[k] = dict(sc, evaluation=C.evaluate(ySE, p, SE['match']))
            path = mdir / f'v_final_{k}.joblib'
            joblib.dump(ad, path)
            cand_hash[k] = C.sha256_file(path)
            re = C.load_v_adapter(path, cand_hash[k])
            if not np.array_equal(re.predict_matrix(SE['X'], names, C.STATE_VERSION), p):
                raise ValueError('reloaded V adapter predictions differ')
        chosen = min(cands, key=lambda k: (sel_scores[k]['logloss'], sel_scores[k]['brier'], k))
        forced = os.environ.get('FC_SMOKE_FORCE_V_CALIBRATION')
        if forced:
            if not args.smoke:
                raise SystemExit('forced calibration is a smoke-only code-path test')
            chosen = forced
        cal_scores = {k: C.score(yCA, cands[k].predict_matrix(CA['X'], names, C.STATE_VERSION), C.weights(CA['match'])) for k in cands}
        selection = dict(role=C.ROLE_TAG, stage='V calibration selection', protocol_sha256=protocol_sha, smoke=args.smoke,
                         rule='lowest V_SELECT match-weighted log loss, then Brier, then name', chosen=chosen,
                         select_scores={k: {m: v[m] for m in ('logloss', 'brier', 'auc')} for k, v in sel_scores.items()},
                         select_evaluation={k: v['evaluation'] for k, v in sel_scores.items()},
                         calibrate_scores_diagnostic=cal_scores, candidate_sha256=cand_hash,
                         chosen_path=str((mdir / f'v_final_{chosen}.joblib').relative_to(C.OUT)), chosen_sha256=cand_hash[chosen],
                         sigmoid_fit=sig.fit_info, final_fit=rec_final, census=census,
                         isotonic='not a candidate (protocol design choice: flat regions can erase delta)',
                         written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                         written_before=['OOF fold adapters', 'labels', 'q fitting', 'any TEST/external access'])
        sel_sha = C.write_json(out / 'selection_v.json', selection)
        st.update('running', 'v_selected', chosen=chosen, select_logloss={k: v['logloss'] for k, v in sel_scores.items()},
                  selection_sha256=sel_sha, next_step='OOF fold adapters')

        # ---------------- OOF fold adapters
        oof = np.full(len(yTR), np.nan)
        fold_recs, fold_hash, sel_fold_preds = {}, {}, {}
        for k in range(C.N_FOLDS):
            tr = fold != k
            b_k, cols_k, rec_k = fit_base(TR['X'][tr], yTR[tr], TR['match'][tr], names, st, f'oof_fold{k}')
            if cols_k != cols:
                raise ValueError('fold feature schema drift')
            held = set(TR['match'][fold == k].tolist())
            fit_set = set(TR['match'][tr].tolist())
            membership = dict(held_out_matches=len(held), fit_matches=len(fit_set), overlap=len(held & fit_set),
                              held_out_match_ids_sha256=C.sha256_json(sorted(held)), fit_match_ids_sha256=rec_k['fit_match_ids_sha256'],
                              fit_folds=sorted({int(x) for x in np.unique(fold[tr])}))
            if membership['overlap'] or k in membership['fit_folds']:
                raise ValueError(f'fold {k} fitting membership violated')
            if chosen == 'sigmoid_pos':
                pc = b_k.predict_proba(CA['X'][:, keep])[:, 1]
                cal_k = C.PositiveSlopeSigmoid().fit(pc, yCA, C.weights(CA['match']))
                cal_desc = dict(calibration='sigmoid_pos', calibration_partition='V_CAL', calibration_fit=cal_k.fit_info)
            else:
                cal_k = C.IdentityCalibrator()
                cal_desc = dict(calibration='raw (identity)', calibration_partition=None)
            ad = C.FullCorpusWinProbV2(names, cols, b_k, cal_k, dict(
                adapter=f'oof_fold{k}', fit_partition=f'TRAIN folds {membership["fit_folds"]}', fit_rows=rec_k['rows'],
                fit_matches=rec_k['matches'], fit_match_ids_sha256=rec_k['fit_match_ids_sha256'], held_out_fold=k,
                protocol_sha256=protocol_sha, selection_v_sha256=sel_sha, smoke=args.smoke, **cal_desc))
            path = mdir / f'v_oof_fold{k}.joblib'
            joblib.dump(ad, path)
            fold_hash[k] = C.sha256_file(path)
            re = C.load_v_adapter(path, fold_hash[k])
            hmask = fold == k
            oof[hmask] = re.predict_matrix(TR['X'][hmask], names, C.STATE_VERSION)
            sel_fold_preds[k] = re.predict_matrix(SE['X'], names, C.STATE_VERSION)
            fold_recs[k] = dict(fit=rec_k, membership=membership, calibration=cal_desc, sha256=fold_hash[k],
                                heldout_train_eval=C.evaluate(yTR[hmask], oof[hmask], TR['match'][hmask]),
                                v_select_eval=C.evaluate(ySE, sel_fold_preds[k], SE['match']))
            st.update('running', 'oof_folds', processed=k + 1, total=C.N_FOLDS, fold_sha256=fold_hash[k], next_step='OOF diagnostics')
        if not np.isfinite(oof).all():
            raise ValueError('incomplete OOF predictions')
        P = np.stack([sel_fold_preds[k] for k in range(C.N_FOLDS)])
        final_sel = sel_preds[chosen]
        dispersion = dict(rows=int(P.shape[1]), mean_across_row_sd=float(P.std(axis=0).mean()),
                          p90_across_row_sd=float(np.quantile(P.std(axis=0), .9)),
                          max_across_row_sd=float(P.std(axis=0).max()),
                          mean_abs_fold_minus_final={k: float(np.mean(np.abs(P[k] - final_sel))) for k in range(C.N_FOLDS)},
                          max_abs_fold_minus_final={k: float(np.max(np.abs(P[k] - final_sel))) for k in range(C.N_FOLDS)})
        oof_eval = C.evaluate(yTR, oof, TR['match'])
        C.save_npz(out / 'predictions' / 'v_train_oof_and_select.npz', train_match=TR['match'], train_query_ms=TR['query_ms'],
                   train_fold=fold, train_oof_p=oof, select_match=SE['match'], select_query_ms=SE['query_ms'],
                   select_final_raw=sel_preds['raw'], select_final_sigmoid_pos=sel_preds['sigmoid_pos'], select_fold_p=P,
                   role=np.asarray('V predictions vs W on TRAIN (held-out fold) and V_SELECT; outcome arrays not included'))
        v_manifest = dict(role=C.ROLE_TAG, protocol_sha256=protocol_sha, selection_v_sha256=sel_sha, chosen=chosen, smoke=args.smoke,
                          final=dict(path=str((mdir / f'v_final_{chosen}.joblib').relative_to(C.OUT)), sha256=cand_hash[chosen]),
                          candidates_sha256=cand_hash, oof_sha256={f'fold{k}': fold_hash[k] for k in range(C.N_FOLDS)},
                          oof_paths={f'fold{k}': str((mdir / f'v_oof_fold{k}.joblib').relative_to(C.OUT)) for k in range(C.N_FOLDS)},
                          model_version=C.MODEL_VERSION, state_version=C.STATE_VERSION, state_names_sha256=C.sha256_json(names),
                          feature_names=cols, folds=fold_recs, heldout_train_oof_eval=oof_eval, fold_dispersion_on_v_select=dispersion,
                          final_v_select_eval=sel_scores[chosen]['evaluation'], final_fit=rec_final,
                          note='cross-fitting of generated targets, not proof of semantic truth',
                          completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        C.write_json(out / 'v_models_manifest.json', v_manifest)
        st.update('complete', 'fit_v', chosen=chosen, oof_auc=oof_eval['auc'], oof_logloss=oof_eval['logloss'],
                  next_step='labels for TRAIN/VALIDATION')
        return 0
    except SystemExit as exc:
        st.update('failed', 'fit_v', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'fit_v', error=repr(exc), next_step='inspect, fix, rerun (no selection written yet?)')
        return 3


if __name__ == '__main__':
    sys.exit(main())
