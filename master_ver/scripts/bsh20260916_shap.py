"""Stage S: exact 7-group Shapley of the frozen models on hash-selected MAIN TEST rows (cells all and B40) per cohort.
--smoke: TRAIN-only pseudo cells (rows of matches with train_fold 4 as pseudo-test, rest as background) with small sizes."""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bsh20260916_common as B  # noqa: E402
import fc20260915_common as C  # noqa: E402
import fc20260915_shap as FS  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def parent_prediction_for_rows(model, coh, base, match, s_ms):
    """Frozen parent evaluation prediction of the explained rows (sealed; exact key join)."""
    path, key = B.parent_eval_predictions(model, coh)
    B.log_access(base, f'parent evaluation predictions {model} {coh}', path.name, 'parent frozen MAIN_TEST predictions', True)
    with np.load(path, allow_pickle=False) as z:
        pm, ps, pv = z['match'].astype(str), z['s_ms'], z[key]
    pos = {k: i for i, k in enumerate(zip(pm.tolist(), ps.tolist()))}
    ix = np.asarray([pos[(m, int(s))] for m, s in zip(match.tolist(), s_ms.tolist())], dtype=np.int64)
    return pv[ix].astype(np.float64), C.sha256_file(path)


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    B.log_command()
    base = B.SMOKE if args.smoke else B.OUT
    st = B.Status('shap' + ('_smoke' if args.smoke else ''))
    out = base / 'shap'
    try:
        if not B.frozen_path(base).exists():
            raise SystemExit('frozen_manifest.json required')
        fz = C.read_json(B.frozen_path(base))
        if bool(fz['smoke_train_only']) != bool(args.smoke):
            raise SystemExit('frozen manifest mode differs')
        if not args.smoke and C.sha256_file(B.OUT / 'protocol.json') != fz['protocol_sha256']:
            raise SystemExit('protocol changed since the freeze')
        n_ex, n_bg = fz['rows']['n_explain'], fz['rows']['n_background']
        gidx = [fz['group_indices'][g] for g in B.GROUPS]
        schema = C.read_json(B.FC / 'q_pre_only_schema.json')
        ridge, names = list(schema['predictor_sets']['ridge']), list(schema['input_names_all'])
        if B.group_map_record(ridge, schema['shap_groups']) != fz['groups']:
            raise SystemExit('group map differs from the freeze')
        cols = [names.index(n) for n in ridge]
        F_tr, Lb_tr, Co_tr, _ = B.load_parent_set('MAIN_TRAIN', base, 'background rows')
        if [str(x) for x in F_tr['input_names']] != names:
            raise SystemExit('schema drift (TRAIN)')
        if args.smoke:
            fold = np.asarray([C.train_fold(m) for m in F_tr['match'].astype(str).tolist()])
            F_te, Lb_te, Co_te = F_tr, Lb_tr, Co_tr
            te_ok, tr_ok = fold == 4, fold != 4
        else:
            F_te, Lb_te, Co_te, _ = B.load_parent_set('MAIN_TEST', base, 'explained rows (sealed)')
            if [str(x) for x in F_te['input_names']] != names:
                raise SystemExit('schema drift (TEST)')
            te_ok, tr_ok = np.ones(len(Lb_te['valid_h90']), dtype=bool), np.ones(len(Lb_tr['valid_h90']), dtype=bool)
        m_te, s_te = F_te['match'].astype(str), F_te['s_ms']
        m_tr, s_tr = F_tr['match'].astype(str), F_tr['s_ms']
        summary = dict(role=B.ROLE_TAG, version=B.VERSION, smoke_train_only=bool(args.smoke), frozen_manifest_sha256=C.sha256_file(B.frozen_path(base)),
                       groups=list(B.GROUPS), group_sizes=[len(ix) for ix in gidx], coalitions=1 << len(B.GROUPS),
                       reference='Lundberg & Lee (2017), arXiv:1705.07874; exact group Shapley by full coalition enumeration (fc20260915_shap.group_exact_shapley)',
                       value_function='mean over background rows of f(x_S, background_notS); f = frozen final calibrated q', models={}, all_checks_pass=True)
        total = len(B.MODELS) * len(B.COHORTS) * len(B.CELLS)
        done = 0
        for model in B.MODELS:
            summary['models'][model] = {}
            for coh in B.COHORTS:
                rec = fz['models'][model][coh]
                path = B.ROOT / rec['bundle']
                if C.sha256_file(path) != rec['bundle_sha256']:
                    raise SystemExit(f'{model} {coh}: bundle hash mismatch')
                bnd = joblib.load(path)
                f = B.model_fn(bnd, rec['calibration'])
                tr_rows = np.flatnonzero((Lb_tr['valid_h90'] == 1) & (Co_tr['cohort'] == B.COHORT_CODE[coh]) & tr_ok)
                bg = B.select_rows(B.TAG_BG, tr_rows, m_tr, s_tr, n_bg, coh)
                Zb = F_tr['X_input'][bg][:, cols]
                te_rows_all = np.flatnonzero((Lb_te['valid_h90'] == 1) & (Co_te['cohort'] == B.COHORT_CODE[coh]) & te_ok)
                p_pre_all = F_te['X_input'][:, names.index('p_pre_V')]
                cell_rows = {'all': te_rows_all, 'B40': te_rows_all[B.b40_mask(p_pre_all[te_rows_all])]}
                per_cell, ex_sets = {}, {}
                for cell in B.CELLS:
                    st.update('running', f'{model}_{coh}_{cell}', processed=done, total=total, next_step='group shapley')
                    rows = cell_rows[cell]
                    ex = B.select_rows(B.TAG_EXPLAIN, rows, m_te, s_te, n_ex, coh, cell)
                    ex_sets[cell] = set(ex.tolist())
                    Ze = F_te['X_input'][ex][:, cols]
                    t0 = time.time()
                    phi = np.zeros((len(ex), len(B.GROUPS)))
                    v0, vall = np.zeros(len(ex)), np.zeros(len(ex))
                    for r in range(len(ex)):
                        phi[r], v0[r], vall[r] = FS.group_exact_shapley(f, Ze[r], Zb, gidx)
                        if (r + 1) % 32 == 0:
                            st.update('running', f'{model}_{coh}_{cell}', processed=done, total=total, rows_done=r + 1, rows=len(ex), next_step='group shapley')
                    fx = f(Ze)
                    checks = dict(additivity_max_abs=float(np.max(np.abs(phi.sum(1) + v0 - fx))), full_coalition_max_abs=float(np.max(np.abs(vall - fx))),
                                  empty_coalition_unique=bool(np.ptp(v0) < 1e-12), runtime_s=round(time.time() - t0, 1), model_evaluations=int(len(ex) * (1 << len(B.GROUPS)) * len(bg)),
                                  selection_deterministic=bool(np.array_equal(B.select_rows(B.TAG_EXPLAIN, rows, m_te, s_te, n_ex, coh, cell), ex)))
                    b2 = joblib.load(path)
                    f2 = B.model_fn(b2, rec['calibration'])
                    again = np.stack([FS.group_exact_shapley(f2, Ze[i], Zb, gidx)[0] for i in range(min(3, len(ex)))])
                    checks['reload_first3_bitwise_equal'] = bool(np.array_equal(again, phi[:len(again)]))
                    b2 = f2 = None
                    if args.smoke:
                        checks['phi_plus_base_equals_parent_evaluation_prediction_max_abs'] = None
                        checks['parent_eval_file_sha256'] = None
                    else:
                        p_par, psha = parent_prediction_for_rows(model, coh, base, m_te[ex], s_te[ex])
                        checks['phi_plus_base_equals_parent_evaluation_prediction_max_abs'] = float(np.max(np.abs(phi.sum(1) + v0 - p_par)))
                        checks['parent_eval_file_sha256'] = psha
                        checks['parent_eval_file_sha256_equals_frozen'] = psha == fz['parent_references']['eval_predictions_sha256'][f'{model}_{coh}']
                    checks['pass'] = bool(checks['additivity_max_abs'] < 1e-8 and checks['full_coalition_max_abs'] < 1e-8 and checks['empty_coalition_unique']
                                          and checks['reload_first3_bitwise_equal'] and checks['selection_deterministic']
                                          and (args.smoke or (checks['phi_plus_base_equals_parent_evaluation_prediction_max_abs'] < 1e-8 and checks['parent_eval_file_sha256_equals_frozen'])))
                    summary['all_checks_pass'] &= checks['pass']
                    w = B.weights(m_te[ex])
                    rng = np.random.default_rng(B.BOOT_SEED)
                    bs = np.stack([np.mean(np.abs(phi[rng.integers(0, len(ex), len(ex))]), axis=0) for _ in range(B.BOOT_REPS)])
                    minutes = s_te[ex] / 60000.
                    bands = {}
                    for lo, hi in B.BANDS:
                        mm = (minutes >= lo) & (minutes < hi)
                        bands[f'{lo}-{hi}' if hi < 1000 else f'{lo}+'] = dict(n=int(mm.sum()), mean_abs=dict(zip(B.GROUPS, np.mean(np.abs(phi[mm]), 0).tolist())) if mm.any() else None,
                                                                             mean_signed=dict(zip(B.GROUPS, np.mean(phi[mm], 0).tolist())) if mm.any() else None)
                    case_pos = sorted(range(len(ex)), key=lambda r: B.rank_key(B.TAG_CASE, coh, cell, m_te[ex[r]], int(s_te[ex[r]])))[:B.N_CASES]
                    local = [dict(match=str(m_te[ex[r]]), s_ms=int(s_te[ex[r]]), start_minute=round(float(s_te[ex[r]]) / 60000, 2), y_h90=int(Lb_te['Y_h90'][ex[r]]),
                                  p_pre=float(p_pre_all[ex[r]]), q_final=float(fx[r]), base_value=float(v0[r]), group_shapley=dict(zip(B.GROUPS, phi[r].tolist())))
                             for r in case_pos]
                    mean_abs = np.mean(np.abs(phi), 0)
                    per_cell[cell] = dict(explained_rows=int(len(ex)), explained_matches=int(len(np.unique(m_te[ex]))), eligible_rows=int(len(rows)),
                                          eligible_matches=int(len(np.unique(m_te[rows]))), coverage=f'{len(ex)} of {len(rows)} eligible rows',
                                          positive_rate_explained=float(np.mean(Lb_te['Y_h90'][ex])), mean_q=float(np.mean(fx)), mean_p_pre=float(np.mean(p_pre_all[ex])),
                                          base_value=float(v0.mean()), checks=checks,
                                          global_mean_abs=dict(zip(B.GROUPS, mean_abs.tolist())), global_mean_abs_rank=[B.GROUPS[i] for i in np.argsort(-mean_abs)],
                                          global_mean_abs_match_weighted=dict(zip(B.GROUPS, np.average(np.abs(phi), axis=0, weights=w).tolist())),
                                          global_mean_abs_bootstrap_ci95={g: np.quantile(bs[:, k], [.025, .975]).tolist() for k, g in enumerate(B.GROUPS)},
                                          global_mean_signed=dict(zip(B.GROUPS, np.mean(phi, 0).tolist())),
                                          share_of_total_abs=dict(zip(B.GROUPS, (mean_abs / mean_abs.sum()).tolist())), time_bands=bands, local_cases=local)
                    C.save_npz(out / f'shap_{model}_{coh}_{cell}.npz', explained_row_index=ex, background_row_index=bg, explained_match=m_te[ex], explained_s_ms=s_te[ex],
                               background_match=m_tr[bg], background_s_ms=s_tr[bg], group_names=np.array(B.GROUPS), phi=phi, base=v0, q_final=fx,
                               y=Lb_te['Y_h90'][ex], p_pre=p_pre_all[ex])
                    done += 1
                    st.update('running', f'{model}_{coh}_{cell}_done', processed=done, total=total, checks_pass=checks['pass'], mean_abs=per_cell[cell]['global_mean_abs'], next_step='next cell')
                overlap = ex_sets['all'] & ex_sets['B40']
                summary['models'][model][coh] = dict(bundle=rec['bundle'], candidate=rec['candidate'], calibration=rec['calibration'], background_rows=int(len(bg)),
                                                     background_matches=int(len(np.unique(m_tr[bg]))), cells=per_cell,
                                                     overlap_all_B40=dict(rows=len(overlap), share_of_B40_set=len(overlap) / max(1, len(ex_sets['B40']))),
                                                     all_minus_B40_mean_abs=dict(zip(B.GROUPS, (np.array([per_cell['all']['global_mean_abs'][g] for g in B.GROUPS])
                                                                                                 - np.array([per_cell['B40']['global_mean_abs'][g] for g in B.GROUPS])).tolist())),
                                                     note='all versus B40 differences are descriptive (different rows; no paired interval)')
                bnd = None
        summary['evaluated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        sha = C.write_json(out / 'shap_summary.json', summary)
        st.update('complete' if summary['all_checks_pass'] else 'failed', 'shap', summary_sha256=sha, all_checks_pass=summary['all_checks_pass'], next_step='snapshot after, post-run checks, report')
        return 0 if summary['all_checks_pass'] else 1
    except SystemExit as exc:
        B.log_failure(st.group, exc)
        st.update('failed', 'shap', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        B.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'shap', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
