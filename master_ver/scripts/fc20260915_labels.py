"""Full-corpus stage L: generated engagement labels and pre-only q inputs.

TRAIN engagements: p_pre and p_post from the held-out-fold adapter of the row's match (the match's W never entered
that adapter's fit or calibration). VALIDATION (all four roles), TEST and external engagements: final V.
delta_h = V(post_h) - V(pre) with the same adapter, Y_h = 1[delta_h > 0], exact zero -> 0. TEST/external labels
are generated only with --group test_external after frozen_manifest.json exists.
Diagnostic only: final-V (in-sample) labels on TRAIN to report OOF-vs-final label shift (never q targets).
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
from collections import Counter  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

HS = C.HORIZONS_S


def label_group(L, st, set_id, sub_roles, adapter_for, vman, names, out_name, diag_final=None):
    E = D.load_engagements(L, set_id, sub_roles, states=True, counts=True)
    n = len(E['match'])
    st.log(f'{out_name}: {n} engagement rows loaded')
    if n == 0:
        return dict(rows=0)
    adapter_id = np.empty(n, dtype='U16')
    adapter_sha = np.empty(n, dtype='U64')
    p_pre = np.full(n, np.nan)
    p_post = {h: np.full(n, np.nan) for h in HS}
    diag = {h: np.full(n, np.nan) for h in HS} if diag_final is not None else None
    diag_pre = np.full(n, np.nan) if diag_final is not None else None
    groups = sorted(set(adapter_for(sr) for sr in np.unique(E['sub_role']).tolist()))
    for aid in groups:
        path, sha = (vman['final']['path'], vman['final']['sha256']) if aid == 'final' else \
            (vman['oof_paths'][aid.replace('oof_', '')], vman['oof_sha256'][aid.replace('oof_', '')])
        ad = C.load_v_adapter(C.OUT / path if not L.smoke else C.OUT / path, sha)
        rows = np.asarray([adapter_for(sr) == aid for sr in E['sub_role'].tolist()])
        adapter_id[rows] = aid
        adapter_sha[rows] = sha
        pre_ok = rows & (E['pre_ok'] == 1)
        if pre_ok.any():
            p_pre[pre_ok] = ad.predict_matrix(E['X_pre'][pre_ok], names, C.STATE_VERSION)
            if diag_final is not None:
                diag_pre[pre_ok] = diag_final.predict_matrix(E['X_pre'][pre_ok], names, C.STATE_VERSION)
        for h in HS:
            v = rows & (E[f'valid_h{h}'] == 1)
            if v.any():
                p_post[h][v] = ad.predict_matrix(E[f'X_post_h{h}'][v], names, C.STATE_VERSION)
                if diag_final is not None:
                    diag[h][v] = diag_final.predict_matrix(E[f'X_post_h{h}'][v], names, C.STATE_VERSION)
        st.log(f'{out_name}: adapter {aid} scored {int(rows.sum())} rows')
    checks = {}
    arrays = dict(match=E['match'], sub_role=E['sub_role'], adapter_id=adapter_id, adapter_sha256=adapter_sha,
                  p_pre=p_pre, model_version=np.asarray(C.MODEL_VERSION), state_version=np.asarray(C.STATE_VERSION))
    for k in D.E_SCALAR + ('pre_reason',):
        arrays[k] = E[k]
    arrays['pre_snapshot_age_s_AUDIT'] = np.where(E['pre_ok'] == 1, (E['q_pre'] - E['pre_snapshot']) / 1000., np.nan)
    for h in HS:
        v = E[f'valid_h{h}'] == 1
        delta = np.where(v, p_post[h] - p_pre, np.nan)
        Y = np.where(v, (delta > 0).astype(np.int8), -1).astype(np.int8)
        checks[f'h{h}'] = dict(valid_rows=int(v.sum()), finite_valid=bool(np.isfinite(delta[v]).all()),
                               label_formula_exact=bool(np.array_equal(Y[v], (p_post[h][v] - p_pre[v] > 0).astype(np.int8))),
                               exact_zero_delta=int(np.sum(delta[v] == 0)), positive=int(np.sum(Y[v] == 1)),
                               valid_requires_pre_ok=bool(np.all(E['pre_ok'][v] == 1)))
        arrays[f'endpoint_h{h}'] = E[f'endpoint_h{h}']
        arrays[f'valid_h{h}'] = E[f'valid_h{h}']
        arrays[f'reasons_h{h}'] = E[f'reasons_h{h}']
        arrays[f'invalid_h{h}'] = E[f'invalid_h{h}']
        for k in ('post_snapshot', 'post_unassigned', 'after_raw_kills', 'after_nonkill_same_ts_groups'):
            arrays[f'{k}_h{h}'] = E[f'{k}_h{h}']
        arrays[f'post_snapshot_age_s_h{h}'] = np.where(v, (E[f'endpoint_h{h}'] - E[f'post_snapshot_h{h}']) / 1000., np.nan)
        arrays[f'post_snapshot_after_L_h{h}'] = np.where(v, (E[f'post_snapshot_h{h}'] > E['L']).astype(np.int8), -1)
        arrays[f'p_post_h{h}'] = p_post[h]
        arrays[f'delta_h{h}'] = delta
        arrays[f'Y_h{h}'] = Y
        arrays[f'after_counts_h{h}'] = E[f'after_h{h}']
        if diag is not None:
            dd = np.where(v, diag[h] - diag_pre, np.nan)
            yd = (dd > 0).astype(np.int8)
            w = C.weights(E['match'][v]) if v.any() else None
            checks[f'h{h}']['DIAGNOSTIC_final_V_in_sample_vs_oof_label_agreement'] = dict(
                rows=float(np.mean(yd[v] == Y[v])) if v.any() else None,
                match_weighted=float(np.average(yd[v] == Y[v], weights=w)) if v.any() else None,
                mean_abs_delta_diff=float(np.mean(np.abs(dd[v] - delta[v]))) if v.any() else None,
                oof_positive_rate=float(np.mean(Y[v] == 1)) if v.any() else None,
                final_positive_rate=float(np.mean(yd[v] == 1)) if v.any() else None)
    arrays['during_counts'] = E['during']
    arrays['count_keys'] = np.asarray(E['count_keys'])
    # duplicated endpoints imply identical post predictions (same adapter, same query)
    dup = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        m = (E[f'valid_h{a}'] == 1) & (E[f'valid_h{b}'] == 1) & (E[f'endpoint_h{a}'] == E[f'endpoint_h{b}'])
        dup[f'{a}_{b}'] = dict(equal_endpoints=int(m.sum()), post_prediction_equal=bool(np.array_equal(p_post[a][m], p_post[b][m])))
        if not dup[f'{a}_{b}']['post_prediction_equal']:
            raise ValueError(f'duplicated endpoints with different predictions {a}/{b}')
    mono = (E['valid_h60'] == 1) & (E['valid_h90'] == 1) & (E['valid_h120'] == 1)
    checks['endpoints_monotone_60_90_120'] = bool(np.all((E['endpoint_h60'][mono] <= E['endpoint_h90'][mono])
                                                          & (E['endpoint_h90'][mono] <= E['endpoint_h120'][mono])))
    checks['no_raw_kill_in_L_endpoint'] = {f'h{h}': int(np.sum(E[f'after_raw_kills_h{h}'][E[f'valid_h{h}'] == 1] != 0)) for h in HS}
    checks['duplicated_endpoints'] = dup
    checks['first_kill_is_raw_kill_rows_failing'] = int(np.sum(E['first_kill_raw_count'] == 0))
    checks['last_kill_is_raw_kill_rows_failing'] = int(np.sum(E['kills_at_L'] == 0))
    checks['ambiguous_simultaneous_first_kill_rows'] = int(np.sum(E['first_kill_raw_count'] > 1))
    checks['next_start_stored_vs_recomputed_mismatch_rows'] = int(np.sum((E['next_start_recomputed'] >= 0) & (E['next_start_stored'] != E['next_start_recomputed'])))
    checks['game_end_stored_equals_recomputed_mismatch'] = int(np.sum((E['game_end'] >= 0) & (E['game_end'] != E['game_end_stored'])))
    checks['pre_snapshot_not_after_query'] = bool(np.all(E['pre_snapshot'][E['pre_ok'] == 1] <= E['q_pre'][E['pre_ok'] == 1]))
    checks['post_snapshot_not_after_endpoint'] = {f'h{h}': bool(np.all(E[f'post_snapshot_h{h}'][E[f'valid_h{h}'] == 1] <= E[f'endpoint_h{h}'][E[f'valid_h{h}'] == 1])) for h in HS}
    reasons = {f'h{h}': dict(Counter(E[f'invalid_h{h}'][E[f'valid_h{h}'] == 0].tolist())) for h in HS}
    lab_sha = C.save_npz(L.base / 'labels' / f'{out_name}_labels.npz', **arrays)
    # pre-only q inputs
    keep = [i for i, nm in enumerate(names) if nm != 'snapshot_age_s']
    input_names = [names[i] for i in keep] + [C.P_PRE]
    ok = E['pre_ok'] == 1
    Xin = np.full((n, len(input_names)), np.nan)
    Xin[ok, :-1] = E['X_pre'][ok][:, keep]
    Xin[ok, -1] = p_pre[ok]
    bad = [nm for nm in input_names if any(t in nm for t in C.Q_FORBIDDEN_TOKENS if t != 'champion_id') and nm != C.P_PRE]
    feat_sha = C.save_npz(L.base / 'labels' / f'{out_name}_features_pre_only.npz', X_input=Xin, input_names=np.asarray(input_names),
                          match=E['match'], s_ms=E['s'], sub_role=E['sub_role'], pre_ok=E['pre_ok'],
                          role=np.asarray('PRE_ONLY_Q_INPUTS: StateV2 at q_pre minus snapshot_age_s, plus p_pre_V'))
    summary = dict(rows=n, matches=int(len(np.unique(E['match']))), labels_sha256=lab_sha, features_sha256=feat_sha,
                   adapters=groups, checks=checks, invalid_reasons=reasons, forbidden_input_names=bad,
                   pre_ok_rows=int(ok.sum()),
                   exclusions_by_match_file=f'labels/{out_name}_labels.npz (invalid_h*/pre_reason per row with match id)')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=('trainval', 'test_external'), required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--sets', default='KR_16.13,KR_16.14_pilot,KR_16.15,NA1_16.13')
    args = ap.parse_args()
    L = D.Layout(args.smoke)
    st = C.Status(L.base, f'labels_{args.group}')
    try:
        vman = C.read_json(L.base / 'v_models_manifest.json')
        sel = C.read_json(L.base / 'selection_v.json')
        if sel['chosen'] != vman['chosen']:
            raise SystemExit('V manifest/selection disagree')
        names = L.manifest('MAIN')['names']
        summaries = {}
        if args.group == 'trainval':
            if (L.base / 'labels' / 'labels_trainval_manifest.json').exists():
                raise SystemExit('trainval labels already written')
            # reconstruct fold fit sets from the extraction and check against adapter fit records
            TRm = D.load_v_rows(L, 'MAIN', [f'fold{k}' for k in range(C.N_FOLDS)], with_X=False)
            fold_of = {m: int(s[4:]) for m, s in zip(TRm['match'].tolist(), TRm['sub_role'].tolist())}
            member = {}
            for k in range(C.N_FOLDS):
                fit_set = sorted({m for m, f in fold_of.items() if f != k})
                ad = C.load_v_adapter(C.OUT / vman['oof_paths'][f'fold{k}'], vman['oof_sha256'][f'fold{k}'])
                member[f'fold{k}'] = dict(reconstructed_fit_set_sha_equals_adapter=C.sha256_json(fit_set) == ad.fit_record['fit_match_ids_sha256'],
                                          held_out_fold=ad.fit_record['held_out_fold'])
                if not member[f'fold{k}']['reconstructed_fit_set_sha_equals_adapter'] or ad.fit_record['held_out_fold'] != k:
                    raise SystemExit(f'fold {k} adapter fit membership cannot be reconstructed')
            final = C.load_v_adapter(C.OUT / vman['final']['path'], vman['final']['sha256'])
            st.update('running', 'labels_train', next_step='validation labels')
            s_tr = label_group(L, st, 'MAIN', [f'fold{k}' for k in range(C.N_FOLDS)], lambda sr: 'oof_' + sr, vman, names,
                               'MAIN_TRAIN', diag_final=final)
            # self-outcome exclusion on actual rows: each TRAIN engagement match is absent from its adapter's fit set
            with np.load(L.base / 'labels' / 'MAIN_TRAIN_labels.npz', allow_pickle=False) as z:
                em, aid = z['match'].tolist(), z['adapter_id'].tolist()
            viol = sum(1 for m, a in zip(em, aid) if m in fold_of and fold_of[m] != int(a[-1]))
            not_v = sum(1 for m in set(em) if m not in fold_of)
            s_tr['oof_self_outcome_exclusion'] = dict(rows_checked=len(em), rows_whose_adapter_fit_includes_match=viol,
                                                      engagement_matches_without_V_rows_not_in_any_fit=not_v,
                                                      adapter_membership=member)
            if viol:
                raise SystemExit('OOF self-outcome exclusion violated')
            summaries['MAIN_TRAIN'] = s_tr
            st.update('running', 'labels_validation', next_step='manifest')
            summaries['MAIN_VALIDATION'] = label_group(L, st, 'MAIN', list(C.VAL_ROLES), lambda sr: 'final', vman, names, 'MAIN_VALIDATION')
            C.write_json(L.base / 'labels' / 'labels_trainval_manifest.json', dict(
                role=C.ROLE_TAG, v_models_manifest_sha256=C.sha256_file(L.base / 'v_models_manifest.json'),
                selection_v_sha256=C.sha256_file(L.base / 'selection_v.json'), summaries=summaries,
                written_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        else:
            if not L.frozen_manifest.exists():
                raise SystemExit('TEST/external labels require frozen_manifest.json (all selections frozen)')
            fz = C.read_json(L.frozen_manifest)
            if fz['v_final_sha256'] != vman['final']['sha256']:
                raise SystemExit('frozen manifest V hash differs')
            sets = [('MAIN', ['TEST'], 'MAIN_TEST')] + [(s, ['EXTERNAL'], f'EXT_{s}') for s in args.sets.split(',')]
            for set_id, roles, name in sets:
                if not (L.set_dir(set_id) / 'extraction_manifest.json').exists():
                    summaries[name] = dict(blocker=f'extraction for {set_id} not available')
                    continue
                st.update('running', f'labels_{name}', next_step='next set')
                summaries[name] = label_group(L, st, set_id, roles, lambda sr: 'final', vman, names, name)
            C.write_json(L.base / 'labels' / 'labels_test_external_manifest.json', dict(
                role=C.ROLE_TAG, frozen_manifest_sha256=C.sha256_file(L.frozen_manifest), summaries=summaries,
                written_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        st.update('complete', f'labels_{args.group}', next_step='q fitting' if args.group == 'trainval' else 'evaluation')
        return 0
    except SystemExit as exc:
        st.update('failed', f'labels_{args.group}', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', f'labels_{args.group}', error=repr(exc), next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
