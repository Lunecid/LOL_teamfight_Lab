"""Full-corpus data access: extraction chunks, role mapping, outcome access control (logged).

Outcome files (W) are loaded only through load_outcomes(), which appends to outcome_access_log.jsonl and
refuses TEST/external outcomes before frozen_manifest.json exists. Smoke runs remap TRAIN matches only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

import fc20260915_common as C


class Layout:
    def __init__(self, smoke=False):
        self.smoke = bool(smoke)
        tag = os.environ.get('FC_SMOKE_TAG', '')
        self.base = (C.OUT / ('smoke_train_only' + (f'_{tag}' if tag else ''))) if smoke else C.OUT
        self.extract = self.base / 'extract'

    def set_dir(self, set_id):
        return self.extract / set_id

    def manifest(self, set_id):
        return C.read_json(self.set_dir(set_id) / 'extraction_manifest.json')

    @property
    def frozen_manifest(self):
        return self.base / 'frozen_manifest.json'


def smoke_role(mid):
    """TRAIN-only smoke: folds 0-2 -> pseudo TRAIN with re-hashed pseudo folds; fold 3 -> V_CAL/Q_CAL; fold 4 -> V_SELECT/Q_SELECT."""
    f = C.train_fold(mid)
    half = C.h8('smoke-val-20260915:', mid) % 2
    if f == 3:
        return 'VALIDATION', ('V_CAL' if half == 0 else 'Q_CAL')
    if f == 4:
        return 'VALIDATION', ('V_SELECT' if half == 0 else 'Q_SELECT')
    return 'TRAIN', f'fold{C.h8("smoke-fold-20260915:", mid) % 5}'


def chunk_paths(layout, set_id):
    man = layout.manifest(set_id)
    for cm in man['chunks']:
        yield C.OUT / cm['path'], C.OUT / cm['outcome_path'], cm


def match_table(layout, set_id, verify_hashes=False):
    """Per-match records (no outcomes)."""
    cols = None
    parts = []
    for sp, _op, cm in chunk_paths(layout, set_id):
        if verify_hashes and C.sha256_file(sp) != cm['states_sha256']:
            raise ValueError(f'chunk hash changed: {sp}')
        with np.load(sp, allow_pickle=False) as z:
            if cols is None:
                cols = [k for k in z.files if k.startswith('m_')]
            parts.append({k: z[k] for k in cols})
    T = {k[2:]: np.concatenate([p[k] for p in parts]) for k in cols}
    if layout.smoke:
        rr = [smoke_role(m) for m in T['match']]
        T['role'] = np.asarray([r[0] for r in rr])
        T['sub_role'] = np.asarray([r[1] for r in rr])
    return T


def load_v_rows(layout, set_id, sub_roles, bucket_only=True, with_X=True):
    """V query rows for matches whose sub_role is in sub_roles (None = all)."""
    out = {k: [] for k in ('match', 'query_ms', 'bucket', 'snapshot_ms', 'unassigned', 'is_bucket_sample', 'sub_role', 'X')}
    names = None
    for sp, _op, cm in chunk_paths(layout, set_id):
        with np.load(sp, allow_pickle=False) as z:
            names = z['names'].tolist() if names is None else names
            mm = z['m_match']
            sr = z['m_sub_role'] if not layout.smoke else np.asarray([smoke_role(m)[1] for m in mm])
            role_of = dict(zip(mm.tolist(), sr.tolist()))
            vm = z['v_match']
            vsr = np.asarray([role_of[m] for m in vm.tolist()]) if len(vm) else np.zeros(0, dtype='U10')
            keep = np.ones(len(vm), dtype=bool) if sub_roles is None else np.isin(vsr, list(sub_roles))
            if bucket_only:
                keep &= z['v_is_bucket_sample'] == 1
            if not keep.any():
                continue
            out['match'].append(vm[keep])
            out['sub_role'].append(vsr[keep])
            for k in ('query_ms', 'bucket', 'snapshot_ms', 'unassigned', 'is_bucket_sample'):
                out[k].append(z['v_' + k][keep])
            if with_X:
                out['X'].append(z['v_X'][keep])
    R = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items() if k != 'X'}
    if with_X:
        R['X'] = np.concatenate(out['X']) if out['X'] else np.zeros((0, len(names or [])))
    R['names'] = names
    return R


def _log_access(layout, set_id, sub_roles, purpose, n):
    rec = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), set=set_id, sub_roles=sorted(sub_roles) if sub_roles else None,
               purpose=purpose, rows=int(n), frozen_manifest_exists=layout.frozen_manifest.exists())
    layout.base.mkdir(parents=True, exist_ok=True)
    with open(layout.base / 'outcome_access_log.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')


def load_outcomes(layout, set_id, sub_roles, purpose):
    """match -> (winner_blue, terminal_ms, reason). TEST/external require the frozen manifest first."""
    sealed = set_id != 'MAIN' or (sub_roles is None) or any(r == 'TEST' for r in sub_roles)
    if sealed and not layout.frozen_manifest.exists():
        raise PermissionError(f'outcomes of {set_id}/{sub_roles} are sealed until frozen_manifest.json exists')
    res = {}
    for sp, op, cm in chunk_paths(layout, set_id):
        with np.load(sp, allow_pickle=False) as z:
            mm = z['m_match']
            sr = z['m_sub_role'] if not layout.smoke else np.asarray([smoke_role(m)[1] for m in mm])
        wanted = set(mm[np.isin(sr, list(sub_roles))].tolist()) if sub_roles is not None else set(mm.tolist())
        if not wanted:
            continue
        with np.load(op, allow_pickle=False) as o:
            if str(o['plan_sha256']) != cm['plan_sha256']:
                raise ValueError('outcome chunk plan mismatch')
            for m, w, t, r in zip(o['o_match'].tolist(), o['o_winner_blue'].tolist(), o['o_terminal_ms'].tolist(), o['o_reason'].tolist()):
                if m in wanted:
                    res[m] = (int(w), int(t), r)
    _log_access(layout, set_id, sub_roles, purpose, len(res))
    return res


E_SCALAR = ('s', 'L', 'q_pre', 'next_start_stored', 'game_end_stored', 'same_match_overlap_stored', 'end_observed_stored',
            'first_kill_raw_count', 'kills_at_L', 'next_kill', 'next_start_recomputed', 'next_start_eff', 'game_end',
            'support_start', 'last_frame', 'pre_ok', 'pre_snapshot', 'pre_unassigned', 'nonkill_at_L')
E_H = ('endpoint', 'valid', 'post_snapshot', 'post_unassigned', 'after_raw_kills', 'after_nonkill_same_ts_groups')


def load_engagements(layout, set_id, sub_roles, states=True, counts=False):
    """All engagement rows (valid or not) for the requested sub roles, in chunk order."""
    acc = {}
    names = count_keys = None

    def add(k, v):
        acc.setdefault(k, []).append(v)

    for sp, _op, cm in chunk_paths(layout, set_id):
        with np.load(sp, allow_pickle=False) as z:
            names = z['names'].tolist() if names is None else names
            count_keys = z['count_keys'].tolist() if count_keys is None else count_keys
            mm = z['m_match']
            sr = z['m_sub_role'] if not layout.smoke else np.asarray([smoke_role(m)[1] for m in mm])
            role_of = dict(zip(mm.tolist(), sr.tolist()))
            em = z['e_match']
            esr = np.asarray([role_of[m] for m in em.tolist()]) if len(em) else np.zeros(0, dtype='U10')
            keep = np.ones(len(em), dtype=bool) if sub_roles is None else np.isin(esr, list(sub_roles))
            if not keep.any():
                continue
            add('match', em[keep])
            add('sub_role', esr[keep])
            for k in E_SCALAR:
                add(k, z['e_' + k][keep])
            add('pre_reason', z['e_pre_reason'][keep])
            for h in C.HORIZONS_S:
                for k in E_H:
                    add(f'{k}_h{h}', z[f'e_{k}_h{h}'][keep])
                add(f'reasons_h{h}', z[f'e_reasons_h{h}'][keep])
                add(f'invalid_h{h}', z[f'e_invalid_h{h}'][keep])
                if states:
                    add(f'X_post_h{h}', z[f'e_X_post_h{h}'][keep])
                if counts:
                    add(f'after_h{h}', z[f'e_after_h{h}'][keep])
            if states:
                add('X_pre', z['e_X_pre'][keep])
            if counts:
                add('during', z['e_during'][keep])
    R = {k: np.concatenate(v) for k, v in acc.items()}
    R['names'] = names
    R['count_keys'] = count_keys
    return R
