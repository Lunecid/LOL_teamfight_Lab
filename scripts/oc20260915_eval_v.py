"""Objective-channel ablation stage T1: frozen W evaluation of raw A vs raw B_noobj on identical saved bucket keys.

Requires frozen_manifest.json. A predictions are recomputed and must equal the parent saved predictions (metrics use
the saved values). Equal-match weighted (primary) and row-weighted metrics, calibration and time bands. MAIN_TEST
1000-replicate paired match bootstrap of B-minus-A Brier/log loss under both weightings (fixed models). Secondary
calibration family reported separately. unknown_objective_team_count nonzero prevalence per set.
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

import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

BANDS = ((0, 2), (2, 10), (10, 20), (20, 30), (30, 1000))


def row_eval(y, p):
    if not len(y):
        return dict(n=0, empty=True)
    w = np.ones(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = C.calibration(np.asarray(y).astype(int), np.asarray(p, float), w)
    out.pop('bins', None)
    out['ece_10bin'] = C.ece(np.asarray(y).astype(int), np.asarray(p, float), w)
    out['citl_intercept_offset'] = C.citl(np.asarray(y).astype(int), np.asarray(p, float), w)
    return out


def cells(y, p, g, t_ms):
    mins = t_ms / 60000.
    out = dict(match_weighted=C.evaluate(y, p, g), row_weighted=row_eval(y, p), time_bands={})
    for lo, hi in BANDS:
        m = (mins >= lo) & (mins < hi)
        key = '<2' if lo == 0 else (f'{lo}-{hi}' if hi < 1000 else f'{lo}+')
        out['time_bands'][key] = dict(match_weighted=C.evaluate(y[m], p[m], g[m], bins=False), row_weighted=row_eval(y[m], p[m])) if m.any() \
            else dict(n=0, matches=0, empty=True)
    return out


def loss_bootstrap(y, pA, pB, g, reps=K.BOOT_REPS, seed=K.BOOT_SEED):
    y = np.asarray(y, float)
    u, inv = np.unique(g, return_inverse=True)
    nm = len(u)
    eps = np.finfo(np.float64).eps
    wm = C.weights(g)

    def losses(p):
        q = np.clip(p, eps, 1 - eps)
        return dict(brier=(p - y) ** 2, logloss=-(y * np.log(q) + (1 - y) * np.log(1 - q)))
    LA, LB = losses(pA), losses(pB)
    S = {}
    for wname, w in (('match_weighted', wm), ('row_weighted', np.ones(len(y)))):
        den = np.bincount(inv, weights=w, minlength=nm)
        for met in ('brier', 'logloss'):
            S[(wname, met)] = (np.bincount(inv, weights=w * (LB[met] - LA[met]), minlength=nm), den)
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(reps) for k in S}
    for r in range(reps):
        mult = np.bincount(rng.integers(0, nm, size=nm), minlength=nm).astype(float)
        for k, (num, den) in S.items():
            draws[k][r] = (mult @ num) / (mult @ den)
    out = dict(replicates=reps, seed=seed, matches=int(nm), rows=int(len(y)), fixed_models=True, direction='B_noobj minus A (positive = B worse)',
               scope='evaluation-match resampling only; no training/selection uncertainty', results={})
    for (wname, met), (num, den) in S.items():
        out['results'].setdefault(wname, {})[met] = dict(estimate=float(num.sum() / den.sum()), ci95=np.quantile(draws[(wname, met)], [.025, .975]).tolist(),
                                                         fraction_replicates_B_worse=float(np.mean(draws[(wname, met)] > 0)),
                                                         first5=draws[(wname, met)][:5].tolist())
    return out


def main():
    st = K.Status('eval_v')
    K.log_command()
    try:
        if not K.frozen_manifest_path().exists():
            raise SystemExit('frozen_manifest.json required')
        fz = C.read_json(K.frozen_manifest_path())
        P = K.parent_reader()
        names = P.manifest('MAIN')['names']
        A_ad = C.load_v_adapter(K.FC / fz['primary_A']['v_final_path'], fz['primary_A']['v_final_sha256'])
        B_raw = K.load_b_adapter(K.OUT / fz['primary']['final_path'], fz['primary']['final_sha256'])
        sec = fz['secondary']['chosen']
        B_sec = K.load_b_adapter(K.OUT / fz['secondary']['final_path'], fz['secondary']['final_sha256'])
        ui = names.index(K.UNKNOWN_COL)
        targets = [('MAIN', ['TEST'], 'MAIN_TEST', 'primary_context')] + [(s, ['EXTERNAL'], f'EXT_{s}', 'external_context') for s in K.EXT] + \
                  [('MAIN', ['V_SELECT'], 'MAIN_V_SELECT', 'diagnostic (secondary selection partition)'), ('MAIN', ['V_CAL'], 'MAIN_V_CAL', 'diagnostic (sigmoid fit)'),
                   ('MAIN', ['Q_CAL'], 'MAIN_Q_CAL', 'diagnostic'), ('MAIN', ['Q_SELECT'], 'MAIN_Q_SELECT', 'diagnostic')]
        res = dict(role=K.ROLE_TAG, frozen_manifest_sha256=C.sha256_file(K.frozen_manifest_path()), primary='raw A vs raw B_noobj',
                   secondary_family=sec, secondary_note='secondary B calibration reported separately; never substitutes the primary raw comparison',
                   sets={})
        for i, (set_id, roles, name, kind) in enumerate(targets):
            st.update('running', f'w_eval_{name}', processed=i, total=len(targets))
            acc = {k: [] for k in ('match', 'query_ms', 'A', 'B', 'B_sec', 'unknown')}
            for sp, _op, _cm in P.chunk_paths(set_id):
                with np.load(sp, allow_pickle=False) as z:
                    vm = z['v_match']
                    if not len(vm):
                        continue
                    role_of = dict(zip(z['m_match'].tolist(), z['m_sub_role'].tolist()))
                    keep = np.isin(np.asarray([role_of[m] for m in vm.tolist()]), roles) & (z['v_is_bucket_sample'] == 1)
                    if not keep.any():
                        continue
                    X = z['v_X'][keep]
                    acc['match'].append(vm[keep])
                    acc['query_ms'].append(z['v_query_ms'][keep])
                acc['A'].append(A_ad.predict_matrix(X, names, C.STATE_VERSION))
                acc['B'].append(B_raw.predict_matrix(X, names, C.STATE_VERSION))
                acc['B_sec'].append(B_sec.predict_matrix(X, names, C.STATE_VERSION))
                acc['unknown'].append(X[:, ui])
            R = {k: np.concatenate(v) for k, v in acc.items()}
            W = P.load_outcomes(set_id, roles, purpose=f'frozen W evaluation raw A vs raw B_noobj on {name} bucket keys')
            y = np.asarray([W[m][0] for m in R['match'].tolist()])
            with np.load(K.FC / 'eval' / 'predictions' / f'v_{name}.npz', allow_pickle=False) as z:
                b = z['is_bucket_sample'] == 1
                pm, pq, ps = z['match'][b], z['query_ms'][b], z['p_raw'][b]
            ident = dict(keys_equal=bool(np.array_equal(pm, R['match']) and np.array_equal(pq, R['query_ms'])))
            ident['A_bitwise_equal_parent'] = bool(ident['keys_equal'] and np.array_equal(ps, R['A']))
            ident['A_max_abs_diff'] = float(np.max(np.abs(ps - R['A']))) if ident['keys_equal'] else None
            if not ident['keys_equal'] or ident['A_max_abs_diff'] > 1e-12:
                raise SystemExit(f'{name}: A not reproduced {ident}')
            R['A'] = ps
            g = R['match']
            u = R['unknown'] > 0
            cell = dict(kind=kind, rows=int(len(y)), matches=int(len(np.unique(g))), primary_identity=ident,
                        A_raw=cells(y, R['A'], g, R['query_ms']), B_noobj_raw=cells(y, R['B'], g, R['query_ms']),
                        unknown_objective_team_count=dict(nonzero_rows=int(u.sum()), nonzero_row_share=float(u.mean()),
                                                          nonzero_match_weighted=A.wrate(u, g)[1], max=float(R['unknown'].max())))
            cell['B_minus_A_point'] = {wn: {m: cell['B_noobj_raw'][wn][m] - cell['A_raw'][wn][m] for m in ('auc', 'brier', 'logloss')}
                                       for wn in ('match_weighted', 'row_weighted')}
            cell['prediction_difference'] = dict(pearson=float(np.corrcoef(R['A'], R['B'])[0, 1]), mean_abs=float(np.mean(np.abs(R['B'] - R['A']))),
                                                 p90_abs=float(np.quantile(np.abs(R['B'] - R['A']), .9)), max_abs=float(np.max(np.abs(R['B'] - R['A']))))
            cell['SECONDARY_B_calibrated'] = dict(family=sec, note='identical to primary raw' if sec == 'raw' else 'separate secondary result',
                                                  metrics=C.evaluate(y, R['B_sec'], g, bins=False) if sec != 'raw' else None)
            if name == 'MAIN_TEST':
                st.update('running', 'bootstrap_MAIN_TEST')
                cell['bootstrap_B_minus_A'] = loss_bootstrap(y, R['A'], R['B'], g)
            C.save_npz(K.OUT / 'eval' / 'w_predictions' / f'w_{name}.npz', match=R['match'], query_ms=R['query_ms'], winner_blue=y, p_A_raw=R['A'],
                       p_B_noobj_raw=R['B'], p_B_noobj_secondary=R['B_sec'], unknown_objective_team_count=R['unknown'],
                       secondary_family=np.asarray(sec), A_sha256=np.asarray(fz['primary_A']['v_final_sha256']),
                       B_raw_sha256=np.asarray(fz['primary']['final_sha256']), B_secondary_sha256=np.asarray(fz['secondary']['final_sha256']))
            res['sets'][name] = cell
            st.log(f"{name}: rows={cell['rows']} A auc={cell['A_raw']['match_weighted']['auc']:.4f} ll={cell['A_raw']['match_weighted']['logloss']:.4f} "
                   f"B auc={cell['B_noobj_raw']['match_weighted']['auc']:.4f} ll={cell['B_noobj_raw']['match_weighted']['logloss']:.4f}")
        st.update('running', 'heldout_train')
        with np.load(K.FC / 'predictions' / 'v_train_oof_and_select.npz', allow_pickle=False) as z:
            tm, tq, tf, pA = z['train_match'], z['train_query_ms'], z['train_fold'], z['train_oof_p']
        with np.load(K.OUT / 'predictions' / 'v_train_oof_and_select_B_noobj.npz', allow_pickle=False) as z:
            if not (np.array_equal(z['train_match'], tm) and np.array_equal(z['train_fold'], tf)):
                raise SystemExit('TRAIN OOF keys differ')
            pB = z['train_oof_p_raw']
        W = P.load_outcomes('MAIN', [f'fold{k}' for k in range(C.N_FOLDS)], purpose='held-out TRAIN W diagnostics A/B OOF')
        y = np.asarray([W[m][0] for m in tm.tolist()])
        res['sets']['MAIN_TRAIN_heldout_fold'] = dict(kind='diagnostic', rows=int(len(y)), A_raw=cells(y, pA, tm, tq), B_noobj_raw=cells(y, pB, tm, tq))
        res['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        C.write_json(K.OUT / 'results' / 'w_eval.json', res)
        st.update('complete', 'eval_v', next_step='labels')
        return 0
    except SystemExit as exc:
        st.update('failed', 'eval_v', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'eval_v', error=repr(exc), next_step='fix and rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
