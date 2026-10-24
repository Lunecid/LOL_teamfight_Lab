"""Data access for the cohort / role experiment (reads the frozen full-corpus run; never writes there).

TRAIN (15.14, held-out-fold labels and p_pre_V) and VALIDATION (15.15) rows come from the frozen label/feature files.
TEST/external label arrays are opened only through load_eval_set(), which requires this task's frozen manifest and
logs every access. Smoke mode is TRAIN-ONLY: VALIDATION arrays are never loaded and pseudo roles are formed from
TRAIN folds (folds 0-2 pseudo TRAIN, fold 3 pseudo Q_CAL, fold 4 pseudo Q_SELECT) so no smoke can inherit FULL/TEST rows.
"""
from __future__ import annotations

import numpy as np

import fc20260915_common as C
import cr20260915_common as K

LABEL_KEYS = ['match', 's', 'L', 'sub_role', 'p_pre'] + [f'{k}_h{h}' for h in K.HS for k in ('valid', 'Y')]


def _load_labeled(name):
    with np.load(K.FC / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok')}
    with np.load(K.FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        Lb = {k: z[k] for k in LABEL_KEYS}
    with np.load(K.OUT / 'cohorts' / f'{name}_cohort.npz', allow_pickle=False) as z:
        Co = {k: z[k] for k in ('match', 's', 'cohort', 'fine', 'scale_known', 'n_min', 'teamfight_cut3_DIAG', 'teamfight_cut5_DIAG',
                                'presence_n_min_DIAG')}
    if not (np.array_equal(F['match'], Lb['match']) and np.array_equal(F['s_ms'], Lb['s'])):
        raise ValueError(f'{name}: feature/label row order differs')
    if not (np.array_equal(Co['match'].astype(str), Lb['match'].astype(str)) and np.array_equal(Co['s'], Lb['s'])):
        raise ValueError(f'{name}: cohort/label row order differs')
    ok = F['pre_ok'] == 1
    if not np.array_equal(F['X_input'][ok, -1], Lb['p_pre'][ok]):
        raise ValueError(f'{name}: p_pre_V column differs from label p_pre')
    return F, Lb, Co


def smoke_sub_role(match_ids):
    f = np.asarray([C.train_fold(m) for m in match_ids])
    return np.where(f <= 2, np.char.add('fold', f.astype(str)), np.where(f == 3, 'Q_CAL', 'Q_SELECT'))


def load_trainval(smoke=False):
    """Concatenated TRAIN + VALIDATION rows (smoke: TRAIN only with pseudo roles)."""
    parts = ['MAIN_TRAIN'] if smoke else ['MAIN_TRAIN', 'MAIN_VALIDATION']
    loaded = [_load_labeled(p) for p in parts]
    names = [str(x) for x in loaded[0][0]['input_names']]
    for F, _, _ in loaded[1:]:
        if [str(x) for x in F['input_names']] != names:
            raise ValueError('input schema differs between TRAIN and VALIDATION')
    D = dict(names=names, parts=parts, smoke=bool(smoke))
    D['X'] = np.vstack([F['X_input'] for F, _, _ in loaded])
    D['g'] = np.concatenate([F['match'] for F, _, _ in loaded]).astype(str)
    D['s_ms'] = np.concatenate([F['s_ms'] for F, _, _ in loaded])
    D['pre_ok'] = np.concatenate([F['pre_ok'] for F, _, _ in loaded]) == 1
    D['sr'] = np.concatenate([F['sub_role'] for F, _, _ in loaded]).astype(str)
    D['sr_true'] = D['sr'].copy()
    if smoke:
        if not np.all(np.char.startswith(D['sr'], 'fold')):
            raise ValueError('smoke must be TRAIN-only')
        D['sr'] = smoke_sub_role(D['g'])
    for k in ('cohort', 'fine', 'scale_known', 'n_min', 'teamfight_cut3_DIAG', 'teamfight_cut5_DIAG', 'presence_n_min_DIAG'):
        D[k] = np.concatenate([Co[k] for _, _, Co in loaded])
    for h in K.HS:
        D[f'valid_h{h}'] = np.concatenate([Lb[f'valid_h{h}'] for _, Lb, _ in loaded]) == 1
        D[f'Y_h{h}'] = np.concatenate([Lb[f'Y_h{h}'] for _, Lb, _ in loaded]).astype(int)
    if smoke:
        keep = smoke_keep(D['g'])
        for k, v in list(D.items()):
            if isinstance(v, np.ndarray) and len(v) == len(keep):
                D[k] = v[keep]
    D['is_train'] = np.char.startswith(D['sr'], 'fold')
    return D


def smoke_keep(match_ids, modulus=None):
    """Code-path smoke subset: matches with sha256('cr-smoke-subset:'+match)[:8] mod CR_SMOKE_MOD == 0 (default 8)."""
    import os
    mod = int(modulus or os.environ.get('CR_SMOKE_MOD', '8'))
    return np.asarray([C.h8('cr-smoke-subset:', m) % mod == 0 for m in match_ids])


def masks(D, h, cohort=None):
    """train / calibrate / select masks for horizon h and cohort ('T', 'N', 'E' or None=E)."""
    v = D[f'valid_h{h}']
    if cohort in ('T', 'N'):
        v = v & (D['cohort'] == K.COHORT_CODE[cohort])
    return dict(train=v & D['is_train'], calibrate=v & (D['sr'] == 'Q_CAL'), select=v & (D['sr'] == 'Q_SELECT'))


def load_eval_set(name, purpose):
    """TEST / external rows (features, labels, cohorts). Requires this task's frozen manifest; access logged."""
    K.outcome_gate(purpose, [name], frozen_required=True)
    F, Lb, Co = _load_labeled(name)
    return F, Lb, Co
