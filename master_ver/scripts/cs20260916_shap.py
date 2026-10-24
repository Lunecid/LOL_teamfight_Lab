"""Stage S (Part A): exact 7-group Shapley of the COMPOSITE system F(S) = q(h(S), V(S)) on the balanced-SHAP rows (cohort T, cells all and
B40) for the frozen final q (LightGBM) and the plain MLP; derived columns and p_pre recomputed under masking.
--smoke: TRAIN-only pseudo cells (matches with train_fold 4 as pseudo-test rows, others as background) with small sizes."""
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
import cs20260916_common as CS  # noqa: E402
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import bsh20260916_common as B  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def bsh_rows(model, coh, cell):
    """Explained / background row keys and the input-space phi of the balanced-SHAP stage for the same model, cohort and cell."""
    with np.load(CS.BSH / 'shap' / f'shap_{model}_{coh}_{cell}.npz', allow_pickle=False) as z:
        return dict(ex_match=z['explained_match'].astype(str), ex_s=z['explained_s_ms'], bg_match=z['background_match'].astype(str), bg_s=z['background_s_ms'],
                    phi=z['phi'], q_final=z['q_final'], base=z['base'], groups=[str(g) for g in z['group_names']])


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    CS.log_command()
    base = CS.SMOKE if args.smoke else CS.OUT
    st = CS.Status('shap' + ('_smoke' if args.smoke else ''))
    out = base / 'shap'
    try:
        if not CS.frozen_path(base).exists():
            raise SystemExit('frozen_manifest.json required')
        fz = C.read_json(CS.frozen_path(base))
        if bool(fz['smoke']) != bool(args.smoke):
            raise SystemExit('frozen manifest mode differs')
        proto = C.read_json(CS.OUT / 'protocol.json')
        if C.sha256_file(CS.OUT / 'protocol.json') != fz['protocol_sha256']:
            raise SystemExit('protocol changed since the freeze')
        fz_fc = C.read_json(CS.FC / 'frozen_manifest.json')
        adapter = C.load_v_adapter(CS.V_ADAPTER, fz_fc['v_final_sha256'])
        schema = C.read_json(CS.FC / 'q_pre_only_schema.json')
        ridge, names = list(schema['predictor_sets']['ridge']), list(schema['input_names_all'])
        smap = CS.StateMap(adapter, names, ridge)
        if C.sha256_json(smap.base_names) != fz['base_names_sha256']:
            raise SystemExit('base names differ from the freeze')
        gidx = [proto['part_a']['group_indices'][g] for g in CS.BASE_GROUPS]
        bundles = {}
        for m in CS.MODELS:
            p, cal, sha, cand = CS.parent_winner(m, 'T')
            if C.sha256_file(p) != sha:
                raise SystemExit(f'{m}: bundle hash differs from the parent freeze')
            bundles[m] = (joblib.load(p), cal)
        f = CS.composite_fn(adapter, smap, bundles)
        n_ex, n_bg = (CS.SMOKE_EXPLAIN, CS.SMOKE_BG) if args.smoke else (CS.N_EXPLAIN, CS.N_BG)
        F_tr, Lb_tr, Co_tr = CS.load_parent_set('MAIN_TRAIN', base, 'background rows (and smoke pseudo-test rows)')
        m_tr, s_tr = F_tr['match'].astype(str), F_tr['s_ms']
        summary = dict(role=CS.ROLE_TAG, version=CS.VERSION, smoke=bool(args.smoke), frozen_manifest_sha256=C.sha256_file(CS.frozen_path(base)), models={}, all_checks_pass=True,
                       value_function=proto['part_a']['value_function'], groups=list(CS.BASE_GROUPS))
        coh = 'T'
        tr_rows = CS.cohort_rows(F_tr, Lb_tr, Co_tr, coh)
        if args.smoke:
            fold = np.asarray([C.train_fold(m) for m in m_tr[tr_rows].tolist()])
            te_rows_all, bg_pool = tr_rows[fold == 4], tr_rows[fold != 4]
            bg = bg_pool[:n_bg]
            X_te_all, m_te, s_te, ppre_te = F_tr['X_input'], m_tr, s_tr, F_tr['X_input'][:, -1]
        else:
            F_te, Lb_te, Co_te = CS.load_parent_set('MAIN_TEST', base, 'Part A explained rows (sealed; after freeze)')
            te_rows_all = CS.cohort_rows(F_te, Lb_te, Co_te, coh)
            m_te, s_te, X_te_all, ppre_te = F_te['match'].astype(str), F_te['s_ms'], F_te['X_input'], F_te['X_input'][:, -1]
            bg = CS.select_rows(B.TAG_BG, tr_rows, m_tr, s_tr, n_bg, coh)
        Zb_bg = smap.base_from_inputs(F_tr['X_input'][bg])
        per_model = {m: {} for m in CS.MODELS}
        for cell in CS.CELLS:
            rows = te_rows_all if cell == 'all' else te_rows_all[B.b40_mask(ppre_te[te_rows_all])]
            ex = rows[:n_ex] if args.smoke else CS.select_rows(B.TAG_EXPLAIN, rows, m_te, s_te, n_ex, coh, cell)
            st.update('running', f'shap_{coh}_{cell}', processed=0, total=len(ex), next_step='exact composite Shapley')
            Xe = X_te_all[ex]
            Zb_ex = smap.base_from_inputs(Xe)
            # reconstruction checks on the explained rows
            S_re = smap.state_from_base(Zb_ex)
            sidx = [j for j, n in enumerate(smap.state_names) if n != 'snapshot_age_s']
            cols = [smap.input_pos[smap.state_names[j]] for j in sidx]
            recon = dict(state_max_abs=float(np.max(np.abs(S_re[:, sidx] - Xe[:, cols]))))
            if args.smoke:
                recon['p_pre_max_abs'] = None  # smoke rows are TRAIN rows whose stored p_pre is the OOF fold adapter's; the final adapter is not expected to reproduce it
            else:
                p_re = adapter.predict_matrix(S_re, smap.state_names, adapter.state_version)
                recon['p_pre_max_abs'] = float(np.max(np.abs(p_re - Xe[:, -1])))
            fx_all = f(Zb_ex)
            phi = {m: np.zeros((len(ex), len(CS.BASE_GROUPS))) for m in CS.MODELS}
            v0 = {m: np.zeros(len(ex)) for m in CS.MODELS}
            vall = {m: np.zeros(len(ex)) for m in CS.MODELS}
            t0 = time.time()
            for r in range(len(ex)):
                res = CS.exact_group_shapley_multi(f, list(CS.MODELS), Zb_ex[r], Zb_bg, gidx)
                for m in CS.MODELS:
                    phi[m][r], v0[m][r], vall[m][r] = res[m]
                if (r + 1) % 32 == 0 or r + 1 == len(ex):
                    st.update('running', f'shap_{coh}_{cell}', processed=r + 1, total=len(ex), seconds=round(time.time() - t0, 1), next_step='checks')
            again = {m: np.stack([CS.exact_group_shapley_multi(f, list(CS.MODELS), Zb_ex[r], Zb_bg, gidx)[m][0] for r in range(min(3, len(ex)))]) for m in CS.MODELS}
            w = CS.weights(m_te[ex])
            for m in CS.MODELS:
                fx = fx_all[m]
                checks = dict(reconstruction=recon, additivity_max_abs=float(np.max(np.abs(phi[m].sum(1) + v0[m] - fx))), full_coalition_max_abs=float(np.max(np.abs(vall[m] - fx))),
                              rerun_first3_bitwise_equal=bool(np.array_equal(again[m], phi[m][:len(again[m])])))
                if args.smoke:
                    checks['rows_equal_balanced_shap'] = None
                    checks['F_equals_parent_stored_prediction_max_abs'] = None
                    phi_in = None
                else:
                    bs = bsh_rows(m, coh, cell)
                    checks['rows_equal_balanced_shap'] = bool(np.array_equal(bs['ex_match'], m_te[ex]) and np.array_equal(bs['ex_s'], s_te[ex]) and np.array_equal(bs['bg_match'], m_tr[bg]) and np.array_equal(bs['bg_s'], s_tr[bg]))
                    checks['F_equals_parent_stored_prediction_max_abs'] = float(np.max(np.abs(fx - bs['q_final'])))
                    phi_in = bs['phi']
                checks['pass'] = bool(checks['additivity_max_abs'] < 1e-9 and checks['full_coalition_max_abs'] < 1e-9 and checks['rerun_first3_bitwise_equal'] and recon['state_max_abs'] < 1e-12
                                      and (recon['p_pre_max_abs'] is None or recon['p_pre_max_abs'] < 1e-12)
                                      and (args.smoke or (checks['rows_equal_balanced_shap'] and checks['F_equals_parent_stored_prediction_max_abs'] < 1e-9)))
                summary['all_checks_pass'] &= checks['pass']
                mean_abs = np.mean(np.abs(phi[m]), 0)
                rng = np.random.default_rng(CS.BOOT_SEED)
                boot = np.stack([np.mean(np.abs(phi[m][rng.integers(0, len(ex), len(ex))]), axis=0) for _ in range(CS.BOOT_REPS)])
                rec = dict(rows=int(len(ex)), matches=int(len(np.unique(m_te[ex]))), checks=checks, mean_abs=dict(zip(CS.BASE_GROUPS, mean_abs.tolist())),
                           share_of_sum_abs=dict(zip(CS.BASE_GROUPS, (mean_abs / mean_abs.sum()).tolist())), mean_signed=dict(zip(CS.BASE_GROUPS, np.mean(phi[m], 0).tolist())),
                           mean_abs_match_weighted=dict(zip(CS.BASE_GROUPS, np.average(np.abs(phi[m]), axis=0, weights=w).tolist())),
                           mean_abs_row_bootstrap_ci95_DESCRIPTIVE={g: [float(v) for v in np.quantile(boot[:, i], [0.025, 0.975])] for i, g in enumerate(CS.BASE_GROUPS)},
                           mean_base_value=float(np.mean(v0[m])), mean_F=float(np.mean(fx)), mean_p_pre=float(np.mean(ppre_te[ex])))
                if phi_in is not None:
                    ma_in = np.mean(np.abs(phi_in), 0)
                    rec['input_space_balanced_shap'] = dict(groups=bs['groups'], mean_abs=dict(zip(bs['groups'], ma_in.tolist())), share_of_sum_abs=dict(zip(bs['groups'], (ma_in / ma_in.sum()).tolist())),
                                                            sum_mean_abs=float(ma_in.sum()))
                    rec['sum_mean_abs_composite'] = float(mean_abs.sum())
                per_model[m][cell] = rec
                arrays = dict(explained_row_index=ex, background_row_index=bg, explained_match=m_te[ex], explained_s_ms=s_te[ex], background_match=m_tr[bg], background_s_ms=s_tr[bg],
                              group_names=np.array(CS.BASE_GROUPS), phi=phi[m], base=v0[m], v_full=vall[m], F=fx, p_pre=ppre_te[ex], time_minutes=Xe[:, names.index('time_minutes')])
                if phi_in is not None:
                    arrays.update(phi_input_space=phi_in, input_space_groups=np.array(bs['groups']))
                C.save_npz(out / f'shap_{m}_{coh}_{cell}.npz', **arrays)
                st.log(f'{m} {coh} {cell}: checks pass={checks["pass"]} shares ' + ' '.join(f'{g[:6]}={v:.3f}' for g, v in rec['share_of_sum_abs'].items()))
        for m in CS.MODELS:
            p, cal, sha, cand = CS.parent_winner(m, 'T')
            summary['models'][m] = dict(bundle=str(p.relative_to(CS.ROOT).as_posix()), candidate=cand, calibration=cal, bundle_sha256=sha, background_rows=int(len(bg)), cells=per_model[m])
        summary['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        sha = C.write_json(out / 'shap_summary.json', summary)
        st.update('complete' if summary['all_checks_pass'] else 'failed', 'shap', summary_sha256=sha, all_checks_pass=summary['all_checks_pass'], next_step='post-run checks')
        return 0 if summary['all_checks_pass'] else 1
    except SystemExit as exc:
        CS.log_failure(st.group, exc)
        st.update('failed', 'shap', error=str(exc)[:1000], next_step='inspect')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        CS.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'shap', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
