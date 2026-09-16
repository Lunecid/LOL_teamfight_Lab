"""Stage C: controlled h90 representation comparison on T and N, role-block ablations on T (all selections pre-TEST).

Arms (same TRAIN rows, Q_CAL calibrators, Q_SELECT rule within each cohort):
  participant  existing participant-slot representation (frozen ridge set, 352 numeric): ridge C=.01 and a NEW
               full-feature LightGBM (economic-model parameters, seeds 7/42/123, all 352 numeric pre features);
  draft        participant representation + per-slot champion one-hot and unordered spell multi-hot (encoders fit on
               cohort TRAIN rows only; indicators unscaled for the ridge); same ridge/tree parameters;
  role         global/objective/time/p_pre features + role block z[team, role, f] = sum_i w_i,role x_i,f (16
               non-champion participant features), blue-minus-red role differences, difference x time_minutes for
               totalGold_norm / xp_norm / level_norm, and team role-uncertainty / missingness summaries; no participant
               slot block, no direct champion/spell indicators. TRAIN rows use OOF role posteriors of their fold;
               Q_CAL / Q_SELECT rows use the final role model (draft inputs only).
Each base gets raw / sigmoid (Platt on logit, Q_CAL) / isotonic (Q_CAL) variants; within-arm choice and the overall
h90 choice (arm winners plus the unchanged A specialist choice) use match-weighted Q_SELECT Brier, log loss, name.
Ablations: five leave-one-role-block-out refits of the role ridge on T (same rows / calibration rule); the primary
ablation variant uses the calibration preselected for the full role ridge on T. Nothing here reads TEST or external rows.
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
import cr20260915_common as K  # noqa: E402
import cr20260915_data as KD  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

H = K.PRIMARY_H


def attach_draft_and_roles(D, base, smoke):
    """Per-row draft IDs and role posteriors (TRAIN: OOF fold model; VALIDATION: final model)."""
    with np.load(K.OUT / 'draft' / 'MAIN_draft.npz', allow_pickle=False) as z:
        dm = z['match'].astype(str)
        champ, spa, spb = z['champion_id'], z['spell_a'], z['spell_b']
    pos = {m: i for i, m in enumerate(dm.tolist())}
    ix = np.asarray([pos[m] for m in D['g'].tolist()])
    inputs = dict(X=D['X'], names=D['names'], champ=champ[ix], spa=spa[ix], spb=spb[ix])
    P, cc = K.participant_columns(D['names'])
    ok = D['pre_ok']
    checks = dict(champion_state_equals_draft_rows=int(np.sum(np.all(D['X'][ok][:, cc].astype(np.int64) == inputs['champ'][ok], axis=1))),
                  pre_ok_rows=int(ok.sum()))
    W = np.full((len(D['g']), 10, 5), np.nan)
    U = np.full((len(D['g']), 2, len(K.UNCERTAINTY_FEATURES)), np.nan)
    gen_row = np.empty(len(D['g']), dtype='U12')
    files = ['MAIN_TRAIN'] + ([] if smoke else ['MAIN_VALIDATION'])
    for name in files:
        with np.load(base / 'role_outputs' / f'{name}_roles.npz', allow_pickle=False) as z:
            rm = z['match'].astype(str)
            Wm, Um, gm = z['W'], z['U'], z['generator'].astype(str)
        rpos = {m: i for i, m in enumerate(rm.tolist())}
        rows = np.flatnonzero(np.char.startswith(D['sr_true'], 'fold') if name == 'MAIN_TRAIN' else ~np.char.startswith(D['sr_true'], 'fold'))
        j = np.asarray([rpos[m] for m in D['g'][rows].tolist()])
        W[rows], U[rows], gen_row[rows] = Wm[j], Um[j], gm[j]
    tr = np.char.startswith(D['sr_true'], 'fold')
    expected = np.where(tr, np.char.add('oof_', D['sr_true']) if not smoke else gen_row, 'final')
    checks['role_generator_matches_row_split'] = bool(np.array_equal(gen_row, expected))
    checks['train_rows_use_oof_fold_of_own_match'] = bool(np.all(gen_row[tr] == np.char.add('oof_', D['sr_true'][tr]))) if not smoke else 'smoke pseudo folds'
    checks['role_arrays_finite'] = bool(np.isfinite(W).all() and np.isfinite(U).all())
    if checks['champion_state_equals_draft_rows'] != checks['pre_ok_rows'] or not checks['role_generator_matches_row_split'] or not checks['role_arrays_finite']:
        raise SystemExit(f'draft/role attachment checks failed: {checks}')
    inputs['W'], inputs['U'] = W, U
    return inputs, checks


def fit_arm(arm, inputs, y, g, M, st, tag, drop_groups=()):
    rows_all = M['train'] | M['calibrate'] | M['select']
    idx_all = np.flatnonzero(rows_all)
    loc = {k: np.flatnonzero(M[k][idx_all]) for k in ('train', 'calibrate', 'select')}
    fz = K.ArmFeaturizer(arm, inputs['names'], drop_groups=drop_groups).fit(inputs, M['train'])
    t0 = time.time()
    Mat = fz.transform(inputs, idx_all)
    log = dict(features=len(fz.feature_names), transform_seconds=round(time.time() - t0, 1), rows=int(len(idx_all)))
    w = C.weights(g[M['train']])
    cw = C.weights(g[M['calibrate']])
    ytr, yca = y[M['train']], y[M['calibrate']]
    bases = {}
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        t1 = time.time()
        bases['ridge'] = K.RidgeBase().fit(Mat[loc['train']], ytr, w, fz.n_scaled)
        log['ridge_seconds'] = round(time.time() - t1, 1)
        log['ridge_n_iter'] = bases['ridge'].n_iter
        log['ridge_converged'] = bases['ridge'].converged
        st.update('running', f'{tag}_{arm}_ridge_done', seconds=log['ridge_seconds'], n_iter=log['ridge_n_iter'], next_step='lgbm')
        if not drop_groups:
            t2 = time.time()
            bases['lgbm'] = K.LgbmBase().fit(Mat[loc['train']], ytr, w)
            log['lgbm_seconds'] = round(time.time() - t2, 1)
            st.update('running', f'{tag}_{arm}_lgbm_done', seconds=log['lgbm_seconds'], next_step='calibrate')
    log['warnings'] = sorted(set(f'{x.category.__name__}: {str(x.message)[:160]}' for x in wl))
    B, P = {}, {}
    split_mats = {k: (Mat[loc[k]] if k == 'train' else fz.transform(inputs, idx_all[loc[k]])) for k in loc}
    for bname, bm in bases.items():
        # calibration / selection predictions use the same featurize-then-predict path as a reloaded bundle on those rows
        raw = {k: bm.raw(split_mats[k]) for k in loc}
        cals = K.fit_calibrators(raw['calibrate'], yca, cw)
        log[f'{bname}_isotonic_thresholds'] = int(len(cals['isotonic'].X_thresholds_))
        log[f'{bname}_sigmoid'] = dict(slope=float(cals['sigmoid'].coef_[0, 0]), intercept=float(cals['sigmoid'].intercept_[0]))
        for cal in K.CALS:
            name = f'{arm}_{bname}_{cal}' if not drop_groups else f'drop_{"+".join(drop_groups)}_{bname}_{cal}'
            B[name] = dict(candidate=name, arm=arm, base=bname, calibration=cal, featurizer=fz, base_model=bm,
                           calibrator=cals.get(cal), feature_names=fz.feature_names, drop_groups=list(drop_groups),
                           fitted_on='featurizer + base: cohort TRAIN; calibrator: cohort Q_CAL' if cal != 'raw' else 'featurizer + base: cohort TRAIN')
            P[name] = {k: K.apply_calibration(cal, cals.get(cal), raw[k]) for k in loc}
    return B, P, log, Mat, loc


def select(met, names):
    rows = [(met[c]['select']['brier'], met[c]['select']['logloss'], c) for c in names]
    return min(rows)[2], [r[2] for r in sorted(rows)]


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--cohorts', default='T,N')
    args = ap.parse_args()
    base = K.OUT / 'smoke_train_only' if args.smoke else K.OUT
    st = K.Status('fit_arms' + ('_smoke' if args.smoke else ''))
    try:
        if (K.OUT / 'frozen_manifest.json').exists():
            raise SystemExit('frozen manifest exists')
        if not args.smoke:
            ct = K.OUT / 'contract_tests' / 'result.json'
            if not ct.exists() or not C.read_json(ct).get('passed'):
                raise SystemExit('contract tests must pass before full fits')
        rman = C.read_json(base / 'role_models_manifest.json')
        D = KD.load_trainval(smoke=args.smoke)
        inputs, attach_checks = attach_draft_and_roles(D, base, args.smoke)
        st.update('running', 'attached', checks=attach_checks, next_step='fit arms')
        y = D[f'Y_h{H}']
        g = D['g']
        summary = dict(role=K.ROLE_TAG, horizon_s=H, attach_checks=attach_checks, role_models_manifest_sha256=C.sha256_file(base / 'role_models_manifest.json'),
                       cohorts={})
        for coh in args.cohorts.split(','):
            M = KD.masks(D, H, coh)
            spec_sel = C.read_json(base / 'selection' / f'q_specialist_{coh}_h{H}.json')
            mdir = base / 'models' / 'arms' / coh / f'h{H}'
            mdir.mkdir(parents=True, exist_ok=True)
            allB_hash, met, logs, preds_sel = {}, {}, {}, {}
            arm_choice = {}
            for arm in K.ARMS:
                st.update('running', f'{coh}_{arm}', next_step='fit')
                B, P, log, Mat, loc = fit_arm(arm, inputs, y, g, M, st, coh)
                logs[arm] = log
                for cand, b in B.items():
                    b['manifest'] = dict(role=K.ROLE_TAG, cohort=coh, horizon_s=H, arm=arm, smoke_train_only=bool(args.smoke),
                                         cohort_manifest_sha256=C.sha256_file(K.OUT / 'cohorts' / 'cohort_manifest.json'),
                                         role_models_manifest_sha256=summary['role_models_manifest_sha256'])
                    joblib.dump(b, mdir / f'{cand}.joblib')
                    allB_hash[cand] = C.sha256_file(mdir / f'{cand}.joblib')
                    met[cand] = {k: C.evaluate(y[M[k]], P[cand][k], g[M[k]], bins=(k == 'select')) for k in loc}
                    preds_sel[cand] = P[cand]['select']
                # reload identity on Q_SELECT rows
                ident, ident_diff = {}, {}
                for cand in B:
                    b2 = joblib.load(mdir / f'{cand}.joblib')
                    p2 = K.predict_arm_bundle(b2, inputs, np.flatnonzero(M['select']))
                    ident[cand] = bool(np.array_equal(p2, P[cand]['select']))
                    ident_diff[cand] = float(np.max(np.abs(p2 - P[cand]['select'])))
                logs[arm]['serialization_identity_select'] = ident
                logs[arm]['serialization_max_abs_diff_select'] = ident_diff
                if not all(ident.values()):
                    raise SystemExit(f'{coh} {arm}: reload identity failed {ident}')
                arm_choice[arm], rank = select(met, K.arm_candidates(arm))
                logs[arm]['ranking'] = rank
                if arm == 'participant':
                    sb = joblib.load(base / 'models' / 'q_specialist' / coh / f'h{H}' / 'ridge_raw.joblib')
                    sp = C.predict_q_bundle(sb, D['X'][M['select']])
                    logs[arm]['ridge_raw_equals_A_specialist_ridge_raw_select'] = dict(
                        bitwise=bool(np.array_equal(sp, P['participant_ridge_raw']['select'])),
                        max_abs_diff=float(np.max(np.abs(sp - P['participant_ridge_raw']['select']))))
                st.update('running', f'{coh}_{arm}_selected', chosen=arm_choice[arm],
                          select_brier={c: round(met[c]['select']['brier'], 6) for c in K.arm_candidates(arm)}, next_step='next arm')
                B = P = Mat = None
            spec_choice = spec_sel['chosen']
            spec_metrics = spec_sel['select_metrics'][spec_choice]
            overall_rows = [(met[arm_choice[a]]['select']['brier'], met[arm_choice[a]]['select']['logloss'], arm_choice[a]) for a in K.ARMS]
            overall_rows.append((spec_metrics['brier'], spec_metrics['logloss'], f'A_specialist:{spec_choice}'))
            overall = min(overall_rows)[2]
            sel = dict(role=K.ROLE_TAG, cohort=coh, horizon_s=H,
                       rule='within arm: lowest cohort Q_SELECT match-weighted Brier, then log loss, then name; overall: same rule over '
                            'the three arm winners and the unchanged A specialist choice',
                       arm_winners=arm_choice, A_specialist_choice=spec_choice, overall_choice=overall,
                       overall_ranking=[r[2] for r in sorted(overall_rows)],
                       select_metrics={c: {m: met[c]['select'][m] for m in ('brier', 'logloss', 'auc')} for c in met},
                       A_specialist_select_metrics=spec_metrics, split_counts={k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m])))) for k, m in M.items()},
                       bundle_sha256=allB_hash, fit_logs=logs, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                       new_frozen_manifest_exists=(K.OUT / 'frozen_manifest.json').exists(),
                       note='participant ridge arm uses the same inputs, rows and C as the A specialist ridge (identity recorded); '
                            'the full-feature LightGBM and the draft/role arms are new models')
            sha = C.write_json(base / 'selection' / f'arms_{coh}_h{H}.json', sel)
            C.write_json(base / 'q_fit_metrics' / f'arms_{coh}_h{H}.json', dict(role=K.ROLE_TAG, metrics=met))
            summary['cohorts'][coh] = dict(selection_sha256=sha, arm_winners=arm_choice, overall=overall)
            st.update('running', f'{coh}_overall_selected', arm_winners=arm_choice, overall=overall, next_step='ablations' if coh == 'T' else 'next')
            # --------------------------------------------- ablations on T
            if coh == 'T':
                ridge_role = [c for c in K.arm_candidates('role') if '_ridge_' in c]
                pre = select(met, ridge_role)[0]
                pre_cal = pre.split('_')[-1]
                adir = base / 'models' / 'ablations' / coh / f'h{H}'
                adir.mkdir(parents=True, exist_ok=True)
                ab = dict(role=K.ROLE_TAG, cohort=coh, horizon_s=H, preselected_full_role_ridge=pre, primary_calibration=pre_cal,
                          rule='each leave-one-role-block-out ridge is refit on cohort TRAIN with its own calibrators on Q_CAL; the primary '
                               'ablation variant uses the calibration preselected for the full role ridge; each ablation\'s own Q_SELECT '
                               'choice among raw/sigmoid/isotonic is also recorded', blocks={})
                for r in K.ROLES:
                    grp = f'role_{r}'
                    st.update('running', f'ablation_drop_{grp}', next_step='fit')
                    B, P, log, Mat, loc = fit_arm('role', inputs, y, g, M, st, f'{coh}_ablation', drop_groups=(grp,))
                    hashes, am = {}, {}
                    for cand, b in B.items():
                        b['manifest'] = dict(role=K.ROLE_TAG, cohort=coh, horizon_s=H, ablation_drop=grp, smoke_train_only=bool(args.smoke))
                        joblib.dump(b, adir / f'{cand}.joblib')
                        hashes[cand] = C.sha256_file(adir / f'{cand}.joblib')
                        am[cand] = {k: C.evaluate(y[M[k]], P[cand][k], g[M[k]], bins=False) for k in ('calibrate', 'select')}
                    own = select({c: dict(select=am[c]['select']) for c in am}, list(am))[0]
                    primary = f'drop_{grp}_ridge_{pre_cal}'
                    ab['blocks'][grp] = dict(primary_candidate=primary, own_q_select_choice=own, bundle_sha256=hashes, fit_log=log,
                                             dropped_features=len([n for n in K.role_feature_names() if K.role_group(n) == grp]),
                                             select_metrics={c: {m: am[c]['select'][m] for m in ('brier', 'logloss', 'auc')} for c in am},
                                             full_role_ridge_select_metrics={m: met[pre]['select'][m] for m in ('brier', 'logloss', 'auc')})
                    B = P = Mat = None
                ab['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                summary['ablations_sha256'] = C.write_json(base / 'selection' / f'ablations_{coh}_h{H}.json', ab)
        summary['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        C.write_json(base / 'selection' / 'arms_summary.json', summary)
        st.update('complete', 'fit_arms', cohorts=summary['cohorts'], next_step='freeze')
        return 0
    except SystemExit as exc:
        st.update('failed', 'fit_arms', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'fit_arms', error=repr(exc), next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
