"""Stage S: exact 7-group interventional Shapley for each cohort's frozen ROLE-ARM winner (final calibrated probability).

Groups: role_TOP, role_JUNGLE, role_MIDDLE, role_BOTTOM, role_UTILITY (blue and red role features, blue-minus-red
differences and difference x time interactions of that role), global_context (all non-participant state features incl.
time and p_pre_V) and role_uncertainty (team assignment entropy, max-marginal summaries, missing/unseen counts):
2^7 = 128 coalitions, v(S) = mean over background rows of f(x_S, background_notS), f = frozen selected calibrated q.
Explained rows: 256 TEST 15.16 h90-valid cohort rows by sha256('cr20260915_shap_explain:<cohort>:<match>:<s>'); background:
128 cohort TRAIN h90-valid rows by sha256('cr20260915_shap_background:<cohort>:<match>:<s>') with OOF role posteriors.
Explains the role model even when the overall h90 winner is another arm. Role ambiguity and correlated / derived features
make interventional attributions descriptive, not causal lane importance (Lundberg & Lee 2017 framework).
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import fc20260915_shap as FS  # noqa: E402
import cr20260915_common as K  # noqa: E402
import cr20260915_data as KD  # noqa: E402

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

N_EXPLAIN, N_BG, N_CASES, BOOT = 256, 128, 8, 1000
BANDS = ((0, 10), (10, 20), (20, 30), (30, 1000))


def rank(tag, coh, m, s):
    return hashlib.sha256(f'{tag}:{coh}:{m}:{int(s)}'.encode()).hexdigest()


def inputs_for(F, draft_path, roles_path, split_filter=None):
    with np.load(draft_path, allow_pickle=False) as z:
        dm = z['match'].astype(str)
        champ, spa, spb = z['champion_id'], z['spell_a'], z['spell_b']
    with np.load(roles_path, allow_pickle=False) as z:
        rm = z['match'].astype(str)
        R = {k: z[k] for k in ('W', 'U', 'P', 'max_marginal', 'entropy', 'flag_champion_unseen', 'generator')}
    g = F['match'].astype(str)
    dpos = {m: i for i, m in enumerate(dm.tolist())}
    rpos = {m: i for i, m in enumerate(rm.tolist())}
    di = np.asarray([dpos[m] for m in g.tolist()])
    ri = np.asarray([rpos[m] for m in g.tolist()])
    names = [str(x) for x in F['input_names']]
    inp = dict(X=np.nan_to_num(F['X_input']), names=names, champ=champ[di], spa=spa[di], spb=spb[di], W=R['W'][ri], U=R['U'][ri])
    extra = dict(P=R['P'][ri], max_marginal=R['max_marginal'][ri], entropy=R['entropy'][ri], unseen=R['flag_champion_unseen'][ri],
                 generator=R['generator'][ri])
    return inp, extra


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    st = K.Status('shap')
    out = K.OUT / 'shap'
    out.mkdir(parents=True, exist_ok=True)
    try:
        fz = C.read_json(K.OUT / 'frozen_manifest.json')
        if not fz.get('arms'):
            raise SystemExit('no frozen arms (C blocked): SHAP not applicable')
        F_te, Lb_te, Co_te = KD.load_eval_set('MAIN_TEST', 'SHAP explanation rows of role-arm winners')
        with np.load(K.FC / 'labels' / 'MAIN_TRAIN_features_pre_only.npz', allow_pickle=False) as z:
            F_tr = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok')}
        with np.load(K.FC / 'labels' / 'MAIN_TRAIN_labels.npz', allow_pickle=False) as z:
            v_tr = z['valid_h90'] == 1
        with np.load(K.OUT / 'cohorts' / 'MAIN_TRAIN_cohort.npz', allow_pickle=False) as z:
            co_tr = z['cohort']
        inp_te, ex_te = inputs_for(F_te, K.OUT / 'draft' / 'MAIN_draft.npz', K.OUT / 'role_outputs' / 'MAIN_TEST_roles.npz')
        inp_tr, ex_tr = inputs_for(F_tr, K.OUT / 'draft' / 'MAIN_draft.npz', K.OUT / 'role_outputs' / 'MAIN_TRAIN_roles.npz')
        if not np.all(ex_tr['generator'] == np.char.add('oof_', F_tr['sub_role'].astype(str))):
            raise SystemExit('background role posteriors are not the OOF fold outputs')
        summary = dict(role=K.ROLE_TAG, frozen_manifest_sha256=C.sha256_file(K.OUT / 'frozen_manifest.json'), groups=list(K.ROLE_GROUPS), coalitions=128,
                       reference='Lundberg & Lee (2017), arXiv:1705.07874; exact group Shapley by full coalition enumeration',
                       scope='explains the frozen role-arm winner of each cohort (not necessarily the overall h90 winner)',
                       caveats=['interventional replacement breaks derived relations (differences, x time, role weights vs uncertainty summaries)',
                                'role posteriors are estimated meta roles from draft; flex picks / swaps remain uncertain',
                                'attributions are associations with generated labels, not causal lane importance or player skill',
                                'bootstrap intervals cover explanation-sample variability over 256 rows only'],
                       cohorts={})
        for coh, sel in fz['arms'].items():
            cand = sel['arm_winners']['role']
            path = K.OUT / 'models' / 'arms' / coh / 'h90' / f'{cand}.joblib'
            if C.sha256_file(path) != fz['frozen_files_sha256'][f'models/arms/{coh}/h90/{cand}.joblib']:
                raise SystemExit('role winner hash mismatch')
            b = joblib.load(path)
            fz_ = b['featurizer']
            fn = fz_.feature_names
            gidx = [[j for j, n in enumerate(fn) if K.role_group(n) == gname] for gname in K.ROLE_GROUPS]
            if sorted(sum(gidx, [])) != list(range(len(fn))):
                raise SystemExit('groups do not partition features')
            te_rows = np.flatnonzero((Lb_te['valid_h90'] == 1) & (Co_te['cohort'] == K.COHORT_CODE[coh]))
            tr_rows = np.flatnonzero(v_tr & (co_tr == K.COHORT_CODE[coh]))
            ex = np.array(sorted(te_rows, key=lambda i: rank('cr20260915_shap_explain', coh, F_te['match'][i], F_te['s_ms'][i]))[:N_EXPLAIN])
            bg = np.array(sorted(tr_rows, key=lambda i: rank('cr20260915_shap_background', coh, F_tr['match'][i], F_tr['s_ms'][i]))[:N_BG])
            Ze = fz_.transform(inp_te, ex)
            Zb = fz_.transform(inp_tr, bg)

            def f(Z, b=b):
                return K.apply_calibration(b['calibration'], b.get('calibrator'), b['base_model'].raw(Z))
            t0 = time.time()
            phi = np.zeros((len(ex), len(K.ROLE_GROUPS)))
            v0, vall = np.zeros(len(ex)), np.zeros(len(ex))
            for r in range(len(ex)):
                phi[r], v0[r], vall[r] = FS.group_exact_shapley(f, Ze[r], Zb, gidx)
                if (r + 1) % 32 == 0:
                    st.update('running', f'{coh}_group_shapley', processed=r + 1, total=len(ex), next_step='summaries')
            fx = f(Ze)
            checks = dict(additivity_max_abs=float(np.max(np.abs(phi.sum(1) + v0 - fx))), full_coalition_max_abs=float(np.max(np.abs(vall - fx))),
                          empty_coalition_unique=bool(np.ptp(v0) < 1e-12), runtime_s=round(time.time() - t0, 1),
                          model_evaluations=int(len(ex) * 128 * len(bg)))
            # selected calibrated prediction equals the evaluation-path prediction of the frozen bundle on these rows
            p_eval = K.predict_arm_bundle(b, inp_te, ex)
            checks['phi_plus_base_equals_selected_calibrated_prediction_max_abs'] = float(np.max(np.abs(phi.sum(1) + v0 - p_eval)))
            b2 = joblib.load(path)
            again = np.stack([FS.group_exact_shapley(lambda Z: K.apply_calibration(b2['calibration'], b2.get('calibrator'), b2['base_model'].raw(Z)),
                                                     Ze[i], Zb, gidx)[0] for i in range(3)])
            checks['reload_first3_bitwise_equal'] = bool(np.array_equal(again, phi[:3]))
            checks['selection_deterministic'] = bool(np.array_equal(
                np.array(sorted(te_rows, key=lambda i: rank('cr20260915_shap_explain', coh, F_te['match'][i], F_te['s_ms'][i]))[:N_EXPLAIN]), ex))
            if b['base'] == 'ridge':
                beta, b0 = b['base_model'].linear_parts()
                mu = Zb.mean(axis=0)
                lin_phi = beta[None, :] * (Ze - mu[None, :])
                grouped = np.stack([lin_phi[:, idx].sum(1) for idx in gidx], 1)
                gb = np.stack([FS.group_exact_shapley(lambda Z: b['base_model'].decision(Z), Ze[i], Zb, gidx)[0] for i in range(32)])
                checks['implementation_check_on_raw_logodds_max_abs'] = float(np.max(np.abs(gb - grouped[:32])))
            checks['pass'] = bool(checks['additivity_max_abs'] < 1e-8 and checks['full_coalition_max_abs'] < 1e-8 and checks['empty_coalition_unique']
                                  and checks['reload_first3_bitwise_equal'] and checks['phi_plus_base_equals_selected_calibrated_prediction_max_abs'] < 1e-8
                                  and checks.get('implementation_check_on_raw_logodds_max_abs', 0) < 1e-8)
            w = C.weights(F_te['match'][ex])
            rng = np.random.default_rng(K.BOOT_SEED)
            bs = np.stack([np.mean(np.abs(phi[rng.integers(0, len(ex), len(ex))]), axis=0) for _ in range(BOOT)])
            minutes = F_te['s_ms'][ex] / 60000.
            bands = {}
            for lo, hi in BANDS:
                mm = (minutes >= lo) & (minutes < hi)
                bands[f'{lo}-{hi}' if hi < 1000 else f'{lo}+'] = dict(n=int(mm.sum()), mean_abs=dict(zip(K.ROLE_GROUPS, np.mean(np.abs(phi[mm]), 0).tolist())) if mm.any() else None,
                                                                     mean_signed=dict(zip(K.ROLE_GROUPS, np.mean(phi[mm], 0).tolist())) if mm.any() else None)
            cases = sorted(range(len(ex)), key=lambda r: rank('cr20260915_shap_case', coh, F_te['match'][ex[r]], F_te['s_ms'][ex[r]]))[:N_CASES]
            local = []
            for r in cases:
                i = ex[r]
                mid = str(F_te['match'][i])
                try:
                    meta = json.loads((C.CACHE_MAIN / f'{mid}.meta.json').read_text(encoding='utf-8'))
                    names_by_pid = meta['static_meta'].get('champion_name_by_pid', {})
                    tm = {int(k): int(v) for k, v in meta['team_map'].items()}
                    order = sorted(tm, key=lambda pid: (tm[pid], pid))
                    champ_names = [names_by_pid.get(str(p), '') for p in order]
                except Exception:
                    champ_names = [''] * 10
                teams = []
                for t, sl in (('blue', range(0, 5)), ('red', range(5, 10))):
                    teams.append(dict(team=t, assignment_entropy=float(ex_te['entropy'][i][0 if t == 'blue' else 1]),
                                      participants=[dict(slot=j, champion_id=int(inp_te['champ'][i, j]), champion=champ_names[j],
                                                         role_marginals=dict(zip(K.ROLES, np.round(inp_te['W'][i, j], 4).tolist())),
                                                         estimated_role=K.ROLES[int(np.argmax(inp_te['W'][i, j]))],
                                                         max_marginal=float(ex_te['max_marginal'][i, j]), champion_unseen_in_role_vocabulary=bool(ex_te['unseen'][i, j]))
                                                    for j in sl]))
                local.append(dict(match=mid, s_ms=int(F_te['s_ms'][i]), start_minute=round(float(F_te['s_ms'][i]) / 60000, 2),
                                  y_h90=int(Lb_te['Y_h90'][i]), q_final=float(fx[r]), base_value=float(v0[r]),
                                  group_shapley=dict(zip(K.ROLE_GROUPS, phi[r].tolist())), teams=teams,
                                  note='role marginals are draft-based meta-role estimates, not observed lanes'))
            C.save_npz(out / f'shap_{coh}.npz', explained_row_index=ex, background_row_index=bg, explained_match=F_te['match'][ex], explained_s_ms=F_te['s_ms'][ex],
                       background_match=F_tr['match'][bg], background_s_ms=F_tr['s_ms'][bg], group_names=np.array(K.ROLE_GROUPS), phi=phi, base=v0, q_final=fx)
            summary['cohorts'][coh] = dict(
                model=cand, arm_winners=sel['arm_winners'], overall_h90_winner=sel['overall'], explains_role_model_even_if_not_overall_winner=True,
                explained_rows=int(len(ex)), explained_matches=int(len(np.unique(F_te['match'][ex]))), background_rows=int(len(bg)),
                background_matches=int(len(np.unique(F_tr['match'][bg]))), eligible_test_rows=int(len(te_rows)), eligible_train_rows=int(len(tr_rows)),
                coverage_note=f'{len(ex)} of {len(te_rows)} eligible TEST cohort rows ({len(ex) / max(1, len(te_rows)):.2%}); descriptive sample',
                features=len(fn), group_sizes=dict(zip(K.ROLE_GROUPS, map(len, gidx))), output='final selected calibrated probability',
                base_value_mean=float(v0.mean()), checks=checks,
                global_mean_abs=dict(zip(K.ROLE_GROUPS, np.mean(np.abs(phi), 0).tolist())),
                global_mean_abs_match_weighted=dict(zip(K.ROLE_GROUPS, np.average(np.abs(phi), axis=0, weights=w).tolist())),
                global_mean_abs_bootstrap_ci95={gname: np.quantile(bs[:, k], [.025, .975]).tolist() for k, gname in enumerate(K.ROLE_GROUPS)},
                global_mean_signed=dict(zip(K.ROLE_GROUPS, np.mean(phi, 0).tolist())), time_bands=bands, local_cases=local)
            st.update('running', f'{coh}_done', model=cand, checks=checks, mean_abs=summary['cohorts'][coh]['global_mean_abs'], next_step='next cohort')
        summary['all_checks_pass'] = all(v['checks']['pass'] for v in summary['cohorts'].values())
        C.write_json(out / 'shap_summary.json', summary)
        st.update('complete' if summary['all_checks_pass'] else 'complete_with_failed_checks', 'shap', next_step='post-run checks, report')
        return 0
    except SystemExit as exc:
        st.update('failed', 'shap', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'shap', error=repr(exc), next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
