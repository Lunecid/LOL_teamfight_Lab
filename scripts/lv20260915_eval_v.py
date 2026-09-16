"""Label validity stage T1: frozen W-evaluation of A, B_reg and B_econ on saved bucket query keys.

Requires this study's frozen_manifest.json. For each set, bucket rows are streamed from the parent extraction and
scored by A (parent final raw adapter) and both frozen comparators (chosen family). A predictions are compared with
the parent's saved evaluation predictions (identity of primary). Outcomes W are loaded through the gated wrapper
(access logged here). Metrics: parent C.evaluate cells, broad time bands with an explicit <2 band, and for MAIN_TEST
a 1000-replicate paired match bootstrap of Brier/log-loss differences (fixed models).
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

import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

TIME_BANDS = ((0, 2), (2, 10), (10, 20), (20, 30), (30, 1000))


def load_models(names):
    fz = C.read_json(K.frozen_manifest_path())
    vman = C.read_json(K.FC / 'v_models_manifest.json')
    if vman['final']['sha256'] != fz['primary_A']['v_final_sha256']:
        raise SystemExit('primary A hash differs from frozen record')
    ads = {'A': C.load_v_adapter(K.FC / vman['final']['path'], vman['final']['sha256'])}
    for M in K.MODELS:
        ads[M] = K.load_alt_adapter(K.OUT / fz['models'][M]['final_path'], fz['models'][M]['final_sha256'], M)
    return fz, vman, ads


def stream_bucket(P, set_id, roles, ads, names):
    acc = {k: [] for k in ('match', 'query_ms', 'snapshot_ms')}
    pr = {k: [] for k in ads}
    for sp, _op, _cm in P.chunk_paths(set_id):
        with np.load(sp, allow_pickle=False) as z:
            mm, sr = z['m_match'], z['m_sub_role']
            vm = z['v_match']
            if not len(vm):
                continue
            role_of = dict(zip(mm.tolist(), sr.tolist()))
            keep = np.isin(np.asarray([role_of[m] for m in vm.tolist()]), roles) & (z['v_is_bucket_sample'] == 1)
            if not keep.any():
                continue
            X = z['v_X'][keep]
            acc['match'].append(vm[keep])
            acc['query_ms'].append(z['v_query_ms'][keep])
            acc['snapshot_ms'].append(z['v_snapshot_ms'][keep])
        for k, ad in ads.items():
            pr[k].append(ad.predict_matrix(X, names, C.STATE_VERSION))
    R = {k: np.concatenate(v) for k, v in acc.items()}
    R['p'] = {k: np.concatenate(v) for k, v in pr.items()}
    return R


def cells(y, p, g, t_ms):
    out = dict(overall=C.evaluate(y, p, g))
    mins = t_ms / 60000.
    out['time_bands'] = {}
    for lo, hi in TIME_BANDS:
        m = (mins >= lo) & (mins < hi)
        key = '<2' if lo == 0 else (f'{lo}-{hi}' if hi < 1000 else f'{lo}+')
        out['time_bands'][key] = C.evaluate(y[m], p[m], g[m], bins=False) if m.any() else dict(n=0, matches=0, empty=True)
    return out


def loss_bootstrap(y, preds, g, pairs, reps=K.BOOT_REPS, seed=K.BOOT_SEED):
    y = np.asarray(y, dtype=float)
    u, inv = np.unique(g, return_inverse=True)
    nm = len(u)
    w = C.weights(g)
    Wm = np.bincount(inv, weights=w, minlength=nm)
    eps = np.finfo(np.float64).eps
    S = {}
    for k, p in preds.items():
        q = np.clip(p, eps, 1 - eps)
        S[k] = dict(brier=np.bincount(inv, weights=w * (p - y) ** 2, minlength=nm),
                    logloss=np.bincount(inv, weights=w * -(y * np.log(q) + (1 - y) * np.log(1 - q)), minlength=nm))
    rng = np.random.default_rng(seed)
    draws = {k: {m: np.empty(reps) for m in ('brier', 'logloss')} for k in preds}
    for r in range(reps):
        mult = np.bincount(rng.integers(0, nm, size=nm), minlength=nm).astype(float)
        den = mult @ Wm
        for k in preds:
            for m in ('brier', 'logloss'):
                draws[k][m][r] = (mult @ S[k][m]) / den
    point = {k: {m: float(S[k][m].sum() / Wm.sum()) for m in ('brier', 'logloss')} for k in preds}
    out = dict(replicates=reps, seed=seed, matches=int(nm), rows=int(len(y)), fixed_models=True,
               scope='test-sample (match) uncertainty only; no training/selection variability; AUC not bootstrapped', pairs=[])
    for a, b in pairs:
        out['pairs'].append(dict(a=a, b=b, **{m: dict(estimate_a_minus_b=point[a][m] - point[b][m],
                                                       ci95=np.quantile(draws[a][m] - draws[b][m], [.025, .975]).tolist())
                                               for m in ('brier', 'logloss')}))
    return out


def main():
    st = K.Status('eval_v')
    K.log_command()
    try:
        if not K.frozen_manifest_path().exists():
            raise SystemExit('frozen_manifest.json required')
        P = K.ParentReadOnly()
        names = P.manifest('MAIN')['names']
        fz, vman, ads = load_models(names)
        targets = [('MAIN', ['TEST'], 'MAIN_TEST', 'primary_context')] + \
                  [(s, ['EXTERNAL'], f'EXT_{s}', 'external_context') for s in K.EXT] + \
                  [('MAIN', ['V_SELECT'], 'MAIN_V_SELECT', 'diagnostic: comparator calibration selection partition'),
                   ('MAIN', ['V_CAL'], 'MAIN_V_CAL', 'diagnostic: calibration fit partition'),
                   ('MAIN', ['Q_CAL'], 'MAIN_Q_CAL', 'diagnostic'), ('MAIN', ['Q_SELECT'], 'MAIN_Q_SELECT', 'diagnostic')]
        results = dict(role=K.ROLE_TAG, frozen_manifest_sha256=C.sha256_file(K.frozen_manifest_path()),
                       models={'A': dict(sha256=vman['final']['sha256'], calibration='raw (parent selection)'),
                               **{M: dict(sha256=fz['models'][M]['final_sha256'], calibration=fz['models'][M]['chosen']) for M in K.MODELS}},
                       interpretation='W prediction quality context; a comparator with poorer W metrics disagreeing with A is not '
                                      'evidence that A labels are wrong', sets={})
        for i, (set_id, roles, name, kind) in enumerate(targets):
            st.update('running', f'v_eval_{name}', processed=i, total=len(targets), next_step='next set')
            R = stream_bucket(P, set_id, roles, ads, names)
            W = P.load_outcomes(set_id, roles, purpose=f'frozen W evaluation of A/B_reg/B_econ on {name} bucket keys')
            y = np.asarray([W[m][0] for m in R['match'].tolist()])
            if not np.isin(y, [0, 1]).all():
                raise ValueError(f'{name}: invalid outcomes')
            with np.load(K.FC / 'eval' / 'predictions' / f'v_{name}.npz', allow_pickle=False) as z:
                b = z['is_bucket_sample'] == 1
                pk_match, pk_q, p_saved = z['match'][b], z['query_ms'][b], z['p_raw'][b]
            ident = dict(keys_equal=bool(np.array_equal(pk_match, R['match']) and np.array_equal(pk_q, R['query_ms'])))
            ident['A_predictions_bitwise_equal_parent'] = bool(ident['keys_equal'] and np.array_equal(p_saved, R['p']['A']))
            ident['A_max_abs_diff_vs_parent'] = float(np.max(np.abs(p_saved - R['p']['A']))) if ident['keys_equal'] else None
            if not ident['keys_equal'] or ident['A_max_abs_diff_vs_parent'] > 1e-12:
                raise SystemExit(f'{name}: primary A bucket predictions not reproduced: {ident}')
            R['p']['A'] = p_saved  # primary metrics from the parent's saved predictions (recomputation only verifies)
            g = R['match']
            cell = dict(kind=kind, rows=int(len(y)), matches=int(len(np.unique(g))), primary_identity=ident,
                        by_model={k: cells(y, R['p'][k], g, R['query_ms']) for k in ads})
            cell['A_minus_B_point'] = {M: {m: cell['by_model']['A']['overall'][m] - cell['by_model'][M]['overall'][m]
                                           for m in ('auc', 'brier', 'logloss')} for M in K.MODELS}
            cell['prediction_agreement'] = {M: dict(pearson=float(np.corrcoef(R['p']['A'], R['p'][M])[0, 1]),
                                                    mean_abs_diff=float(np.mean(np.abs(R['p']['A'] - R['p'][M]))),
                                                    p90_abs_diff=float(np.quantile(np.abs(R['p']['A'] - R['p'][M]), .9)))
                                            for M in K.MODELS}
            if name == 'MAIN_TEST':
                st.update('running', 'bootstrap_MAIN_TEST', next_step='external sets')
                cell['bootstrap_loss_differences'] = loss_bootstrap(y, R['p'], g, [('A', 'B_reg'), ('A', 'B_econ')])
            C.save_npz(K.OUT / 'eval' / 'v_predictions' / f'v_{name}.npz', match=R['match'], query_ms=R['query_ms'],
                       snapshot_ms=R['snapshot_ms'], winner_blue=y, p_A=R['p']['A'], p_B_reg=R['p']['B_reg'], p_B_econ=R['p']['B_econ'],
                       A_sha256=np.asarray(vman['final']['sha256']), B_reg_sha256=np.asarray(fz['models']['B_reg']['final_sha256']),
                       B_econ_sha256=np.asarray(fz['models']['B_econ']['final_sha256']),
                       role=np.asarray('bucket-key W predictions for context; outcomes are targets, never features'))
            results['sets'][name] = cell
            st.log(f'{name}: rows={cell["rows"]} ' + ' '.join(f'{k}:auc={cell["by_model"][k]["overall"]["auc"]:.4f},ll={cell["by_model"][k]["overall"]["logloss"]:.4f}' for k in ads))
        # held-out TRAIN fold predictions (A parent OOF, B OOF)
        st.update('running', 'heldout_train', next_step='write results')
        with np.load(K.FC / 'predictions' / 'v_train_oof_and_select.npz', allow_pickle=False) as z:
            tm, tq, tf, pA = z['train_match'], z['train_query_ms'], z['train_fold'], z['train_oof_p']
        W = P.load_outcomes('MAIN', [f'fold{k}' for k in range(C.N_FOLDS)], purpose='held-out TRAIN V diagnostics (A/B OOF)')
        y = np.asarray([W[m][0] for m in tm.tolist()])
        tr = dict(kind='diagnostic: each TRAIN query predicted by the fold adapter that excluded its match', rows=int(len(y)), by_model={})
        tr['by_model']['A'] = cells(y, pA, tm, tq)
        for M in K.MODELS:
            with np.load(K.OUT / 'predictions' / f'v_train_oof_and_select_{M}.npz', allow_pickle=False) as z:
                if not (np.array_equal(z['train_match'], tm) and np.array_equal(z['train_fold'], tf)):
                    raise SystemExit(f'{M}: TRAIN OOF keys differ from parent')
                tr['by_model'][M] = cells(y, z['train_oof_p'], tm, tq)
        results['sets']['MAIN_TRAIN_heldout_fold'] = tr
        results['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        C.write_json(K.OUT / 'results' / 'v_eval.json', results)
        st.update('complete', 'eval_v', next_step='comparator labels')
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
