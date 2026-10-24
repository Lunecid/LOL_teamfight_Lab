"""Full-corpus stage S: SHAP for the frozen chosen h90 q (restored P4 methods; Lundberg & Lee 2017).

Explained rows: 512 TEST (15.16) h90-valid rows by sha256('fc20260915_shap_explain:<match>:<s>') rank; background:
256 TRAIN h90-valid rows by sha256('fc20260915_shap_background:<match>:<s>') rank; 8 local cases. Feature-level
values are exact interventional SHAP on the declared log-odds scale for linear winners (base score only for isotonic
winners) or LightGBM pred_contrib on raw tree log-odds (base score); group-level exact interventional Shapley over
all 2^G coalitions explains the FINAL output. Additivity identities and reload determinism are checked.
Model association only: no causal esports, player-skill or importance-as-SHAP claims.
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
import fc20260915_data as D  # noqa: E402

import argparse  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats  # noqa: E402

N_EXPLAIN, N_BG, N_CASES = 512, 256, 8


def rank_key(tag, match, s):
    return hashlib.sha256(f'{tag}:{match}:{int(s)}'.encode()).hexdigest()


def group_exact_shapley(f, xrow, Bg, groups):
    G = len(groups)
    nb, d = Bg.shape
    K = 1 << G
    colmask = np.zeros((K, d), dtype=bool)
    for k, idx in enumerate(groups):
        on = ((np.arange(K) >> k) & 1).astype(bool)
        colmask[np.ix_(on, idx)] = True
    Z = np.where(colmask[:, None, :], xrow[None, None, :], Bg[None, :, :]).reshape(K * nb, d)
    v = f(Z).reshape(K, nb).mean(axis=1)
    fact = [math.factorial(i) for i in range(G + 1)]
    phi = np.zeros(G)
    for S in range(K):
        size = bin(S).count('1')
        for k in range(G):
            if not (S >> k) & 1:
                phi[k] += fact[size] * fact[G - size - 1] / fact[G] * (v[S | (1 << k)] - v[S])
    return phi, float(v[0]), float(v[K - 1])


def linear_parts(b):
    kind = b['candidate']
    pipe = b['pipeline'] if kind in ('ridge_raw', 'p_pre_logistic') else b['base_pipeline']
    sc, lr = pipe[0], pipe[-1]
    beta = lr.coef_[0] / sc.scale_
    b0 = lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_)
    if kind == 'ridge_sigmoid':
        return beta, b0, float(b['calibrator'].coef_[0, 0]), float(b['calibrator'].intercept_[0]), pipe
    return beta, b0, 1.0, 0.0, pipe


def output_fn(b, scale):
    kind, cols = b['candidate'], b['input_columns']
    if scale == 'probability':
        return lambda Z: C.predict_q_bundle(b, Z)
    if kind in ('ridge_raw', 'p_pre_logistic', 'p_pre_spline'):
        return lambda Z: b['pipeline'].decision_function(Z[:, cols])
    if kind == 'ridge_sigmoid':
        return lambda Z: b['calibrator'].decision_function(C.logit8(b['base_pipeline'].predict_proba(Z[:, cols])[:, 1]))
    if kind == 'economic_sigmoid':
        return lambda Z: b['calibrator'].decision_function(C.logit8(np.mean([m.predict_proba(Z[:, cols])[:, 1] for m in b['seed_models']], axis=0)))
    raise ValueError((kind, scale))


def main():
    import warnings as _w
    _w.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    L = D.Layout(args.smoke)
    st = C.Status(L.base, 'shap')
    out_dir = L.base / 'shap'
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        fz = C.read_json(L.frozen_manifest)
        h = C.PRIMARY_H
        chosen = fz['q_selections'][f'h{h}']['chosen']
        path = L.base / 'models' / 'q' / f'h{h}' / f'{chosen}.joblib'
        if C.sha256_file(path) != fz['q_bundle_sha256'][f'h{h}'][chosen]:
            raise SystemExit('chosen q bundle hash differs from frozen manifest')
        b = joblib.load(path)
        test_name = 'MAIN_TEST' if not args.smoke else 'MAIN_VALIDATION'
        with np.load(L.base / 'labels' / f'{test_name}_features_pre_only.npz', allow_pickle=False) as z:
            FT = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role')}
        with np.load(L.base / 'labels' / f'{test_name}_labels.npz', allow_pickle=False) as z:
            vt, yt = z[f'valid_h{h}'] == 1, z[f'Y_h{h}']
        with np.load(L.base / 'labels' / 'MAIN_TRAIN_features_pre_only.npz', allow_pickle=False) as z:
            FB = {k: z[k] for k in ('X_input', 'match', 's_ms')}
        with np.load(L.base / 'labels' / 'MAIN_TRAIN_labels.npz', allow_pickle=False) as z:
            vb = z[f'valid_h{h}'] == 1
        names = [str(x) for x in FT['input_names']]
        te_idx = np.flatnonzero(vt)
        tr_idx = np.flatnonzero(vb)
        ex = np.array(sorted(te_idx, key=lambda i: rank_key('fc20260915_shap_explain', FT['match'][i], FT['s_ms'][i]))[:N_EXPLAIN])
        bg = np.array(sorted(tr_idx, key=lambda i: rank_key('fc20260915_shap_background', FB['match'][i], FB['s_ms'][i]))[:N_BG])
        cases = np.array(sorted(ex, key=lambda i: rank_key('fc20260915_shap_case', FT['match'][i], FT['s_ms'][i]))[:N_CASES])
        Xe_full, Xb_full = FT['X_input'][ex], FB['X_input'][bg]
        kind, cols, in_names = b['candidate'], b['input_columns'], b['input_names']
        summary = dict(role=C.ROLE_TAG, target=f'h{h}', model=kind, bundle_sha256=fz['q_bundle_sha256'][f'h{h}'][chosen],
                       explained_rows=int(len(ex)), explained_matches=int(len(set(FT['match'][ex].tolist()))),
                       background_rows=int(len(bg)), background_matches=int(len(set(FB['match'][bg].tolist()))),
                       explain_split='TEST 15.16 (h90-valid rows)' if not args.smoke else 'SMOKE pseudo-validation',
                       background_split='TRAIN 15.14 (h90-valid rows; inputs include held-out-fold p_pre_V)',
                       selection_rule="sha256('fc20260915_shap_explain:<match>:<s>') ascending; background "
                                      "sha256('fc20260915_shap_background:<match>:<s>'); cases sha256('fc20260915_shap_case:<match>:<s>')",
                       reference='Lundberg & Lee (2017), A Unified Approach to Interpreting Model Predictions, NeurIPS (arXiv:1705.07874)',
                       not_causal=True, not_player_skill=True, not_feature_importance_relabelled=True)
        groups_all = sorted(set(C.shap_group(n) for n in in_names))
        gidx = [[j for j, n in enumerate(in_names) if C.shap_group(n) == gr] for gr in groups_all]
        summary['grouping'] = {gr: [in_names[j] for j in idx] for gr, idx in zip(groups_all, gidx)}
        summary['feature_units'] = {n: C.feature_unit(n) for n in in_names}
        phi, base = None, None
        st.update('running', 'feature_level', model=kind, next_step='group exact Shapley')
        if kind == 'constant':
            summary.update(method='constant output: zero attribution', scale='log-odds', base_value=float(np.log(b['prior'] / (1 - b['prior']))),
                           exact_final_output=True)
            phi = np.zeros((len(ex), 0))
        elif kind in ('ridge_raw', 'ridge_sigmoid', 'ridge_isotonic', 'p_pre_logistic'):
            beta, b0, a, c, pipe = linear_parts(b)
            Xe, Xb = Xe_full[:, cols], Xb_full[:, cols]
            mu = Xb.mean(axis=0)
            phi = a * beta[None, :] * (Xe - mu[None, :])
            base = float(c + a * (b0 + np.sum(beta * mu)))
            f_lin = c + a * (b0 + Xe @ beta)
            if kind == 'ridge_sigmoid':
                f_act = b['calibrator'].decision_function(C.logit8(pipe.predict_proba(Xe)[:, 1]))
            else:
                f_act = pipe.decision_function(Xe)
            p_raw = pipe.predict_proba(np.vstack([Xe, Xb]))[:, 1]
            summary.update(method='exact linear interventional SHAP: phi_j = a*beta_j*(x_j - background mean_j), beta_j = coef_j/scale_j',
                           scale=('calibrated log-odds (sigmoid calibration affine in raw log-odds)' if kind == 'ridge_sigmoid' else
                                  'raw ridge log-odds = BASE SCORE, NOT the final isotonic-calibrated probability' if kind == 'ridge_isotonic'
                                  else 'log-odds of final q'),
                           exact_final_output=kind != 'ridge_isotonic', base_value=base,
                           additivity_max_abs_residual_vs_linear_reconstruction=float(np.max(np.abs(phi.sum(1) + base - f_lin))),
                           additivity_max_abs_residual_vs_actual_pipeline_output=float(np.max(np.abs(phi.sum(1) + base - f_act))),
                           logit_clip_active_rows=int(np.sum((p_raw < 1e-8) | (p_raw > 1 - 1e-8))))
        elif kind == 'p_pre_spline':
            fn = output_fn(b, 'log-odds')
            fe, fb = fn(Xe_full), fn(Xb_full)
            base = float(np.mean(fb))
            phi = (fe - base)[:, None]
            summary.update(method='single-input exact interventional SHAP phi = f(x) - mean_background f', scale='log-odds of final q',
                           exact_final_output=True, base_value=base,
                           additivity_max_abs_residual_vs_actual_pipeline_output=float(np.max(np.abs(phi.sum(1) + base - fe))))
        elif kind.startswith('economic_'):
            Xe = Xe_full[:, cols]
            contribs = [m.booster_.predict(Xe, pred_contrib=True, num_threads=4) for m in b['seed_models']]
            margins = [m.booster_.predict(Xe, raw_score=True, num_threads=4) for m in b['seed_models']]
            Cm = np.mean(contribs, axis=0)
            phi, base_rows = Cm[:, :-1], Cm[:, -1]
            base = float(base_rows[0])
            summary.update(method='LightGBM TreeSHAP pred_contrib (tree_path_dependent; reference = training cover), mean over seeds 7/42/123',
                           scale='mean per-seed raw tree log-odds = BASE SCORE, not the final ' + ('mean probability' if kind == 'economic_raw' else 'calibrated probability'),
                           exact_final_output=False, base_value=base,
                           additivity_max_abs_residual_vs_mean_raw_margin=float(np.max(np.abs(Cm.sum(1) - np.mean(margins, axis=0)))),
                           base_value_constant_across_rows=bool(np.all(base_rows == base_rows[0])))
        G_phi = None
        if kind != 'constant':
            scale = 'probability' if kind.endswith('_isotonic') or kind == 'economic_raw' else 'log-odds'
            fn = output_fn(b, scale)
            gcols = [[cols[j] for j in idx] for idx in gidx]
            t0 = time.time()
            G_phi = np.zeros((len(ex), len(groups_all)))
            v0, vall = [], []
            for r, xrow in enumerate(Xe_full):
                ph, a0, a1 = group_exact_shapley(fn, xrow, Xb_full, gcols)
                G_phi[r] = ph
                v0.append(a0)
                vall.append(a1)
                if (r + 1) % 64 == 0:
                    st.update('running', 'group_exact_shapley', processed=r + 1, total=len(ex), next_step='summaries')
            f_ex = fn(Xe_full)
            gs = dict(method='exact interventional Shapley over feature groups by full enumeration of 2^G coalitions; v(S) = mean over TRAIN '
                             'background rows of f(x_S, background_notS)', scale=scale + ' of the final selected q',
                      groups=groups_all, coalitions=1 << len(groups_all), background_rows=int(len(bg)),
                      model_evaluations=int(len(ex) * (1 << len(groups_all)) * len(bg)), runtime_s=round(time.time() - t0, 1),
                      additivity_max_abs_residual=float(np.max(np.abs(G_phi.sum(1) + np.array(v0) - f_ex))),
                      full_coalition_equals_output_max_abs=float(np.max(np.abs(np.array(vall) - f_ex))),
                      empty_coalition_value_unique=bool(np.ptp(v0) < 1e-12), empty_coalition_value=float(v0[0]))
            if phi is not None and phi.shape[1] == len(in_names) and kind in ('ridge_raw', 'ridge_sigmoid', 'p_pre_logistic', 'p_pre_spline'):
                grouped = np.stack([phi[:, idx].sum(1) for idx in gidx], axis=1)
                gs['max_abs_diff_vs_grouped_feature_level_exact_shap'] = float(np.max(np.abs(grouped - G_phi)))
            if kind == 'ridge_isotonic':
                pipe = b['base_pipeline']
                fb = lambda Z: pipe.decision_function(Z[:, cols])  # noqa: E731
                nv = min(64, len(ex))
                Vp = np.stack([group_exact_shapley(fb, Xe_full[i], Xb_full, gcols)[0] for i in range(nv)])
                grouped = np.stack([phi[:nv, idx].sum(1) for idx in gidx], axis=1)
                gs['implementation_check_on_raw_base_score'] = dict(rows=nv, max_abs_diff_vs_grouped_linear_shap=float(np.max(np.abs(Vp - grouped))))
                fin = C.predict_q_bundle(b, Xe_full)
                raw = pipe.predict_proba(Xe_full[:, cols])[:, 1]
                gs['isotonic_notes'] = dict(sklearn_isotonic_predict='linear interpolation between thresholds (piecewise linear, may contain flat regions)',
                                            explained_rows_final_exact_0_or_1=int(np.sum((fin == 0) | (fin == 1))),
                                            explained_rows_distinct_final_values=int(len(np.unique(fin))),
                                            spearman_final_vs_raw=float(scipy.stats.spearmanr(fin, raw).statistic))
            summary['group_exact_final_output'] = gs
        wex = C.weights(FT['match'][ex])
        if phi is not None and phi.shape[1] == len(in_names) and len(in_names):
            glob = pd.DataFrame(dict(feature=in_names, group=[C.shap_group(n) for n in in_names], unit=[C.feature_unit(n) for n in in_names],
                                     mean_abs_shap=np.mean(np.abs(phi), axis=0), mean_abs_shap_match_weighted=np.average(np.abs(phi), axis=0, weights=wex),
                                     mean_signed_shap=np.mean(phi, axis=0), background_mean_raw=Xb_full[:, cols].mean(axis=0)))
            glob = glob.sort_values(['mean_abs_shap', 'feature'], ascending=[False, True]).reset_index(drop=True)
            glob.to_csv(out_dir / 'global_feature_mean_abs_shap.csv', index=False, float_format='%.17g')
            grp = pd.DataFrame(dict(group=groups_all, n_features=[len(i) for i in gidx],
                                    mean_abs_sum_of_feature_shap=[float(np.mean(np.abs(phi[:, i].sum(1)))) for i in gidx],
                                    mean_abs_group_exact_shapley_final_output=np.mean(np.abs(G_phi), axis=0) if G_phi is not None else np.nan,
                                    mean_signed_group_exact_shapley_final_output=np.mean(G_phi, axis=0) if G_phi is not None else np.nan))
            grp.to_csv(out_dir / 'global_group_shap.csv', index=False, float_format='%.17g')
            top = [in_names.index(f) for f in glob.feature[:30]]
            Xt = FB['X_input'][vb][:, [cols[j] for j in top]]
            sd = Xt.std(axis=0)
            ok = sd > 0
            Rm = np.corrcoef(Xt[:, ok], rowvar=False) if ok.sum() > 1 else np.zeros((0, 0))
            tn = [in_names[j] for j, k in zip(top, ok) if k]
            pairs = sorted([dict(a=tn[i], b=tn[k], pearson_r_train=float(Rm[i, k])) for i in range(len(tn)) for k in range(i + 1, len(tn))
                            if abs(Rm[i, k]) >= .8], key=lambda d: (-abs(d['pearson_r_train']), d['a'], d['b']))
            summary['correlated_top30_pairs_abs_r_ge_0_8'] = dict(count=len(pairs), strongest40=pairs[:40])
            summary['top_features'] = glob.head(25)[['feature', 'group', 'unit', 'mean_abs_shap', 'mean_signed_shap']].to_dict('records')
            summary['group_table'] = grp.to_dict('records')
        f_final = C.predict_q_bundle(b, Xe_full)
        local = []
        for i in cases:
            r = int(np.flatnonzero(ex == i)[0])
            rec = dict(match=str(FT['match'][i]), s_ms=int(FT['s_ms'][i]), y=int(yt[i]), q_final=float(f_final[r]), base_value=base)
            if phi is not None and phi.shape[1]:
                order = np.argsort(-np.abs(phi[r]), kind='stable')[:12]
                rec['top_features'] = [dict(feature=in_names[j], group=C.shap_group(in_names[j]), value=float(Xe_full[r, cols[j]]),
                                            background_mean=float(Xb_full[:, cols[j]].mean()), shap=float(phi[r, j])) for j in order]
                rec['sum_shap_plus_base'] = float(phi[r].sum() + (base or 0.0))
            if G_phi is not None:
                rec['group_exact_final_output'] = dict(zip(groups_all, map(float, G_phi[r])))
            local.append(rec)
        summary['local_cases'] = local
        # determinism after reload
        b2 = joblib.load(path)
        det = dict(explained_rows_reselected_equal=bool(np.array_equal(
            np.array(sorted(te_idx, key=lambda i: rank_key('fc20260915_shap_explain', FT['match'][i], FT['s_ms'][i]))[:N_EXPLAIN]), ex)))
        pos = [int(np.flatnonzero(ex == i)[0]) for i in cases]
        if phi is not None and phi.shape[1] and kind in ('ridge_raw', 'ridge_sigmoid', 'ridge_isotonic', 'p_pre_logistic'):
            beta, b0, a, c, pipe = linear_parts(b2)
            phi2 = a * beta[None, :] * (Xe_full[pos][:, cols] - Xb_full[:, cols].mean(axis=0)[None, :])
            det['case_shap_bitwise_equal_after_reload'] = bool(np.array_equal(phi2, phi[pos]))
        elif phi is not None and phi.shape[1] and kind.startswith('economic_'):
            phi2 = np.mean([m.booster_.predict(Xe_full[pos][:, cols], pred_contrib=True, num_threads=4) for m in b2['seed_models']], axis=0)[:, :-1]
            det['case_treeshap_equal_after_reload_1e-12'] = bool(np.max(np.abs(phi2 - phi[pos])) <= 1e-12)
        if G_phi is not None:
            fn2 = output_fn(b2, 'probability' if kind.endswith('_isotonic') or kind == 'economic_raw' else 'log-odds')
            gcols = [[cols[j] for j in idx] for idx in gidx]
            again = np.stack([group_exact_shapley(fn2, Xe_full[p_], Xb_full, gcols)[0] for p_ in pos[:3]])
            det['group_exact_first3_cases_bitwise_equal_after_reload'] = bool(np.array_equal(again, G_phi[pos[:3]]))
        summary['determinism'] = det
        checks = {}
        add = [summary[k] for k in ('additivity_max_abs_residual_vs_linear_reconstruction', 'additivity_max_abs_residual_vs_mean_raw_margin',
                                    'additivity_max_abs_residual_vs_actual_pipeline_output') if k in summary]
        if 'group_exact_final_output' in summary:
            add.append(summary['group_exact_final_output']['additivity_max_abs_residual'])
        checks['additivity_lt_1e-8'] = bool(add) and max(add) < 1e-8
        gx = summary.get('group_exact_final_output', {})
        if 'max_abs_diff_vs_grouped_feature_level_exact_shap' in gx:
            checks['group_exact_matches_feature_level'] = gx['max_abs_diff_vs_grouped_feature_level_exact_shap'] < 1e-8
        if 'implementation_check_on_raw_base_score' in gx:
            checks['group_exact_verified_on_base_score'] = gx['implementation_check_on_raw_base_score']['max_abs_diff_vs_grouped_linear_shap'] < 1e-8
        checks['deterministic'] = all(det.values())
        summary['checks'] = checks
        C.save_npz(out_dir / 'shap_values.npz', explained_row_index=ex, background_row_index=bg, case_row_index=cases,
                   explained_match=FT['match'][ex], explained_s_ms=FT['s_ms'][ex], feature_names=np.array(in_names),
                   shap=phi if phi is not None else np.zeros((len(ex), 0)), base_value=np.array(base if base is not None else np.nan),
                   group_names=np.array(groups_all), group_exact=G_phi if G_phi is not None else np.zeros((len(ex), 0)), q_final=f_final)
        C.write_json(out_dir / 'shap_summary.json', summary)
        st.update('complete' if all(checks.values()) else 'complete_with_failed_checks', 'shap', checks=checks, next_step='report / temporal comparator')
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
