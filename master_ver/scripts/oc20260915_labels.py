"""Objective-channel ablation stage T2: raw B_noobj labels for all 7 sets and the existing h60/h90/h120 endpoints.

Requires frozen_manifest.json. Keys/endpoints/valid masks/snapshots must equal the parent label files; A is recomputed
with the parent adapters and must equal the saved labels bit for bit (saved values stored). TRAIN rows use the B OOF
adapter of their fold, other sets the B final adapter; the same adapter at both ends. Also: sigmoid-family labels for
the calibration sign-invariance verification (secondary family recorded), and unknown_objective_team_count prevalence.
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

import numpy as np  # noqa: E402

SET_ROLES = {'MAIN_TRAIN': ('MAIN', [f'fold{k}' for k in range(C.N_FOLDS)]), 'MAIN_VALIDATION': ('MAIN', list(C.VAL_ROLES)),
             'MAIN_TEST': ('MAIN', ['TEST']), **{f'EXT_{s}': (s, ['EXTERNAL']) for s in K.EXT}}
LAB_KEYS = (['match', 's', 'L', 'q_pre', 'sub_role', 'pre_ok', 'pre_snapshot', 'p_pre', 'adapter_id', 'adapter_sha256']
            + [f'{k}_h{h}' for h in K.HS for k in ('endpoint', 'valid', 'post_snapshot', 'p_post', 'delta', 'Y')])


def main():
    st = K.Status('labels')
    K.log_command()
    try:
        if not K.frozen_manifest_path().exists():
            raise SystemExit('frozen_manifest.json required')
        fz = C.read_json(K.frozen_manifest_path())
        fz_sha = C.sha256_file(K.frozen_manifest_path())
        P = K.parent_reader()
        names = P.manifest('MAIN')['names']
        ui = names.index(K.UNKNOWN_COL)
        paths = {'A': {'final': (K.FC / fz['primary_A']['v_final_path'], fz['primary_A']['v_final_sha256'])}}
        for k in range(C.N_FOLDS):
            paths['A'][f'oof_fold{k}'] = (K.FC / fz['primary_A']['v_oof_paths'][f'fold{k}'], fz['primary_A']['v_oof_sha256'][f'fold{k}'])
        for fam, fd in fz['families'].items():
            paths[f'B_{fam}'] = {'final': (K.OUT / fd['final_path'], fd['final_sha256']),
                                 **{f'oof_fold{k}': (K.OUT / fd['oof_paths'][f'fold{k}'], fd['oof_sha256'][f'fold{k}']) for k in range(C.N_FOLDS)}}
        cache = {}

        def adapter(model, aid):
            if (model, aid) not in cache:
                p, sha = paths[model][aid]
                cache[(model, aid)] = C.load_v_adapter(p, sha) if model == 'A' else K.load_b_adapter(p, sha)
            return cache[(model, aid)]
        summary = {}
        for si, name in enumerate(K.LABEL_SETS):
            set_id, roles = SET_ROLES[name]
            st.update('running', f'load_{name}', processed=si, total=len(K.LABEL_SETS))
            E = P.load_engagements(set_id, roles, states=True, counts=False)
            lab = P.label_file(name, f'objective ablation identity and B_noobj labels {name}', keys=LAB_KEYS)
            co = K.LV.cohort_file(name) if hasattr(K, 'LV') else None
            n = len(E['match'])
            ident = dict(rows=n, keys_equal=bool(np.array_equal(E['match'], lab['match']) and np.array_equal(E['s'], lab['s'])),
                         L_q_pre_sub_role_equal=bool(np.array_equal(E['L'], lab['L']) and np.array_equal(E['q_pre'], lab['q_pre'])
                                                     and np.array_equal(E['sub_role'], lab['sub_role'])),
                         pre_ok_snapshot_equal=bool(np.array_equal(E['pre_ok'], lab['pre_ok']) and np.array_equal(E['pre_snapshot'], lab['pre_snapshot'])),
                         unique_keys=len(set(zip(lab['match'].tolist(), lab['s'].tolist()))) == n,
                         cohort_keys_equal=bool(np.array_equal(co['match'].astype(str), lab['match'].astype(str)) and np.array_equal(co['s'], lab['s'])
                                                and np.array_equal(co['L'], lab['L'])))
            for h in K.HS:
                ident[f'h{h}_endpoint_valid_snapshot_equal'] = bool(np.array_equal(E[f'endpoint_h{h}'], lab[f'endpoint_h{h}']) and
                                                                   np.array_equal(E[f'valid_h{h}'], lab[f'valid_h{h}']) and
                                                                   np.array_equal(E[f'post_snapshot_h{h}'], lab[f'post_snapshot_h{h}']))
                ident[f'h{h}_cohort_valid_equal'] = bool(np.array_equal(co[f'valid_h{h}'], lab[f'valid_h{h}']))
            if not all(v for v in ident.values() if isinstance(v, bool)):
                raise SystemExit(f'{name}: identity failed {ident}')
            if name == 'MAIN_TRAIN':
                fold_hash = np.asarray([C.train_fold(m) for m in E['match'].tolist()])
                if not np.array_equal(np.char.add('fold', fold_hash.astype(str)), E['sub_role'].astype(str)):
                    raise SystemExit('TRAIN sub_role differs from fold hash')
                aids = np.char.add('oof_', E['sub_role'].astype(str))
            else:
                aids = np.full(n, 'final')
            if not np.array_equal(aids.astype(str), lab['adapter_id'].astype(str)):
                raise SystemExit(f'{name}: adapter assignment differs from parent')
            Xpost = {h: E[f'X_post_h{h}'] for h in K.HS}
            valid = {h: E[f'valid_h{h}'] for h in K.HS}
            arrays = dict(match=E['match'], s=E['s'], L=E['L'], q_pre=E['q_pre'], sub_role=E['sub_role'], pre_ok=E['pre_ok'], pre_snapshot=E['pre_snapshot'],
                          cohort=co['cohort'], fine=co['fine'], frozen_manifest_sha256=np.asarray(fz_sha), secondary_family=np.asarray(fz['secondary']['chosen']),
                          unknown_objective_team_count_pre=np.where(E['pre_ok'] == 1, E['X_pre'][:, ui], np.nan))
            for h in K.HS:
                arrays[f'endpoint_h{h}'] = E[f'endpoint_h{h}']
                arrays[f'valid_h{h}'] = E[f'valid_h{h}']
                arrays[f'post_snapshot_h{h}'] = E[f'post_snapshot_h{h}']
                arrays[f'same_pre_post_frame_h{h}'] = np.where(E[f'valid_h{h}'] == 1, (E[f'post_snapshot_h{h}'] == E['pre_snapshot']).astype(np.int8), -1)
            sm = {}
            for model in ('A', 'B_raw', 'B_sigmoid_pos'):
                st.update('running', f'{name}_{model}', processed=si, total=len(K.LABEL_SETS))
                ads = {aid: adapter(model, aid) for aid in sorted(set(aids.tolist()))}
                out = A.generate_labels(aids, ads, names, C.STATE_VERSION, E['X_pre'], E['pre_ok'], Xpost, valid)
                if not np.array_equal(out['adapter_used'], aids):
                    raise SystemExit('adapter used differs')
                if model == 'A':
                    chk = dict(p_pre_bitwise_equal=bool(np.array_equal(out['p_pre'], lab['p_pre'], equal_nan=True)))
                    for h in K.HS:
                        chk[f'h{h}_p_post_bitwise_equal'] = bool(np.array_equal(out['p_post'][h], lab[f'p_post_h{h}'], equal_nan=True))
                        chk[f'h{h}_delta_bitwise_equal'] = bool(np.array_equal(out['delta'][h], lab[f'delta_h{h}'], equal_nan=True))
                        chk[f'h{h}_Y_equal'] = bool(np.array_equal(out['Y'][h], lab[f'Y_h{h}']))
                    ident['A_recomputation'] = chk
                    if not all(chk.values()):
                        raise SystemExit(f'{name}: A not reproduced bit for bit {chk}')
                    arrays['p_pre_A'] = lab['p_pre']
                    for h in K.HS:
                        arrays[f'p_post_h{h}_A'] = lab[f'p_post_h{h}']
                        arrays[f'delta_h{h}_A'] = lab[f'delta_h{h}']
                        arrays[f'Y_h{h}_A'] = lab[f'Y_h{h}']
                    arrays['adapter_id_A'] = lab['adapter_id']
                    arrays['adapter_sha256_A'] = lab['adapter_sha256']
                else:
                    tag = 'B' if model == 'B_raw' else 'Bsig'
                    arrays[f'p_pre_{tag}'] = out['p_pre']
                    for h in K.HS:
                        arrays[f'p_post_h{h}_{tag}'] = out['p_post'][h]
                        arrays[f'delta_h{h}_{tag}'] = out['delta'][h]
                        arrays[f'Y_h{h}_{tag}'] = out['Y'][h]
                    arrays[f'adapter_id_{tag}'] = out['adapter_used']
                    arrays[f'adapter_sha256_{tag}'] = np.asarray([paths[model][a][1] for a in out['adapter_used'].tolist()])
            s = dict(identity=ident, rows=n, matches=int(len(np.unique(E['match']))))
            for h in K.HS:
                v = valid[h] == 1
                dB, dS = arrays[f'delta_h{h}_B'][v], arrays[f'delta_h{h}_Bsig'][v]
                s[f'h{h}'] = dict(valid_rows=int(v.sum()),
                                  positive={m: int(np.sum(arrays[f'Y_h{h}_{m}'][v] == 1)) for m in ('A', 'B')},
                                  exact_zero_delta={m: int(np.sum(arrays[f'delta_h{h}_{m}'][v] == 0)) for m in ('A', 'B', 'Bsig')},
                                  label_formula_errors={m: int(np.sum(arrays[f'Y_h{h}_{m}'][v] != (arrays[f'delta_h{h}_{m}'][v] > 0))) for m in ('A', 'B', 'Bsig')},
                                  invalid_label_minus1={m: bool(np.all(arrays[f'Y_h{h}_{m}'][~v] == -1)) for m in ('A', 'B', 'Bsig')},
                                  same_pre_post_frame_rows=int(np.sum(arrays[f'same_pre_post_frame_h{h}'][v] == 1)),
                                  sign_invariance_sigmoid_vs_raw=dict(label_mismatch_rows=int(np.sum(arrays[f'Y_h{h}_B'][v] != arrays[f'Y_h{h}_Bsig'][v])),
                                                                      direction_reversal_rows=int(np.sum((dB * dS) < 0)),
                                                                      collapsed_to_zero_rows=int(np.sum((dB != 0) & (dS == 0))),
                                                                      max_abs_raw_delta_at_mismatch=float(np.max(np.abs(dB[arrays[f'Y_h{h}_B'][v] != arrays[f'Y_h{h}_Bsig'][v]])))
                                                                      if np.any(arrays[f'Y_h{h}_B'][v] != arrays[f'Y_h{h}_Bsig'][v]) else None))
            pre_ok = E['pre_ok'] == 1
            u = arrays['unknown_objective_team_count_pre'][pre_ok] > 0
            v90 = valid[90] == 1
            u90 = arrays['unknown_objective_team_count_pre'][v90] > 0
            s['unknown_objective_team_count_pre'] = dict(pre_ok_rows=int(pre_ok.sum()), nonzero_rows=int(u.sum()), nonzero_row_share=float(u.mean()) if len(u) else None,
                                                         nonzero_match_weighted=A.wrate(u, E['match'][pre_ok])[1] if len(u) else None,
                                                         valid_h90_nonzero_row_share=float(u90.mean()) if len(u90) else None,
                                                         valid_h90_nonzero_match_weighted=A.wrate(u90, E['match'][v90])[1] if len(u90) else None)
            s['alt_labels_sha256'] = C.save_npz(K.OUT / 'labels' / f'{name}_B_noobj_labels.npz', **arrays)
            summary[name] = s
            st.log(f"{name}: rows={n} h90 pos A={s['h90']['positive']['A']} B={s['h90']['positive']['B']} zeros={s['h90']['exact_zero_delta']} "
                   f"sig_mismatch={s['h90']['sign_invariance_sigmoid_vs_raw']}")
            E = arrays = None
        C.write_json(K.OUT / 'labels' / 'labels_manifest.json', dict(role=K.ROLE_TAG, frozen_manifest_sha256=fz_sha, summaries=summary,
                                                                     secondary_family=fz['secondary']['chosen'],
                                                                     secondary_note='secondary family is raw: secondary labels are identical to the primary B labels'
                                                                     if fz['secondary']['chosen'] == 'raw' else 'secondary labels stored as Bsig',
                                                                     written_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        st.update('complete', 'labels', processed=len(K.LABEL_SETS), total=len(K.LABEL_SETS), next_step='analysis')
        return 0
    except SystemExit as exc:
        st.update('failed', 'labels', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'labels', error=repr(exc), next_step='fix and rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
