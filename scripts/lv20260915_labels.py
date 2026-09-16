"""Label validity stage T2: comparator labels at the three EXISTING endpoints, primary identity, observed quantities.

Requires this study's frozen_manifest.json. For every label set the parent engagement states are re-read (pure
reader), the parent label arrays are opened through the gated wrapper, and:
  * keys/endpoints/validity/snapshots must equal the parent label file exactly; cohort file (match, s, L, valid) too;
  * A (parent final + OOF adapters) is recomputed and compared with saved p_pre/p_post/delta/Y (saved values kept);
  * B_reg and B_econ labels: TRAIN rows -> comparator OOF adapter of the row fold, others -> comparator final;
    the same adapter at q_pre and at the endpoint; delta = V(post) - V(pre); Y = 1[delta > 0], exact zero -> 0;
  * observed StateV2 quantities for each horizon (credited kills, deaths, team gold/xp differences, objective state
    differences) are written separately.
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
import lv20260915_common as K  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime')

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

SET_ROLES = {'MAIN_TRAIN': ('MAIN', [f'fold{k}' for k in range(C.N_FOLDS)]), 'MAIN_VALIDATION': ('MAIN', list(C.VAL_ROLES)),
             'MAIN_TEST': ('MAIN', ['TEST']), **{f'EXT_{s}': (s, ['EXTERNAL']) for s in K.EXT}}
LAB_KEYS = (['match', 's', 'L', 'q_pre', 'sub_role', 'pre_ok', 'pre_snapshot', 'p_pre', 'adapter_id', 'adapter_sha256']
            + [f'{k}_h{h}' for h in K.HS for k in ('endpoint', 'valid', 'post_snapshot', 'p_post', 'delta', 'Y')])


def adapters_for(fz, vman):
    out = {'A': {'final': (K.FC / vman['final']['path'], vman['final']['sha256'])}}
    for k in range(C.N_FOLDS):
        out['A'][f'oof_fold{k}'] = (K.FC / vman['oof_paths'][f'fold{k}'], vman['oof_sha256'][f'fold{k}'])
    for M in K.MODELS:
        m = fz['models'][M]
        out[M] = {'final': (K.OUT / m['final_path'], m['final_sha256'])}
        for k in range(C.N_FOLDS):
            out[M][f'oof_fold{k}'] = (K.OUT / m['oof_paths'][f'fold{k}'], m['oof_sha256'][f'fold{k}'])
    return out


def load_adapter(model, path, sha):
    return C.load_v_adapter(path, sha) if model == 'A' else K.load_alt_adapter(path, sha, model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default=','.join(K.LABEL_SETS))
    args = ap.parse_args()
    st = K.Status('labels')
    K.log_command()
    try:
        if not K.frozen_manifest_path().exists():
            raise SystemExit('frozen_manifest.json required')
        fz = C.read_json(K.frozen_manifest_path())
        fz_sha = C.sha256_file(K.frozen_manifest_path())
        vman = C.read_json(K.FC / 'v_models_manifest.json')
        P = K.ParentReadOnly()
        names = P.manifest('MAIN')['names']
        paths = adapters_for(fz, vman)
        cache = {}

        def adapter(model, aid):
            if (model, aid) not in cache:
                cache[(model, aid)] = load_adapter(model, *paths[model][aid])
            return cache[(model, aid)]
        summary = {}
        sets = args.sets.split(',')
        for si, name in enumerate(sets):
            set_id, roles = SET_ROLES[name]
            st.update('running', f'load_{name}', processed=si, total=len(sets), next_step='identity checks')
            E = P.load_engagements(set_id, roles, states=True, counts=False)
            lab = P.label_file(name, f'primary label identity and comparator label generation for {name}', keys=LAB_KEYS)
            co = K.cohort_file(name)
            n = len(E['match'])
            ident = dict(rows=n, keys_equal=bool(np.array_equal(E['match'], lab['match']) and np.array_equal(E['s'], lab['s'])),
                         L_q_pre_equal=bool(np.array_equal(E['L'], lab['L']) and np.array_equal(E['q_pre'], lab['q_pre'])),
                         sub_role_equal=bool(np.array_equal(E['sub_role'], lab['sub_role'])),
                         pre_ok_snapshot_equal=bool(np.array_equal(E['pre_ok'], lab['pre_ok']) and np.array_equal(E['pre_snapshot'], lab['pre_snapshot'])),
                         unique_keys=len(set(zip(lab['match'].tolist(), lab['s'].tolist()))) == n,
                         cohort_keys_equal=bool(np.array_equal(co['match'].astype(str), lab['match'].astype(str))
                                                and np.array_equal(co['s'], lab['s']) and np.array_equal(co['L'], lab['L'])))
            for h in K.HS:
                ident[f'h{h}_endpoint_valid_snapshot_equal'] = bool(np.array_equal(E[f'endpoint_h{h}'], lab[f'endpoint_h{h}'])
                                                                   and np.array_equal(E[f'valid_h{h}'], lab[f'valid_h{h}'])
                                                                   and np.array_equal(E[f'post_snapshot_h{h}'], lab[f'post_snapshot_h{h}']))
                ident[f'h{h}_cohort_valid_equal'] = bool(np.array_equal(co[f'valid_h{h}'], lab[f'valid_h{h}']))
            if not all(v for v in ident.values() if isinstance(v, bool)):
                raise SystemExit(f'{name}: exact key/endpoint/cohort identity failed: {ident}')
            if name == 'MAIN_TRAIN':
                aids = np.char.add('oof_', E['sub_role'].astype(str))
                fold_hash = np.asarray([C.train_fold(m) for m in E['match'].tolist()])
                ident['train_sub_role_equals_fold_hash'] = bool(np.array_equal(np.char.add('fold', fold_hash.astype(str)), E['sub_role'].astype(str)))
                if not ident['train_sub_role_equals_fold_hash']:
                    raise SystemExit('TRAIN sub_role differs from the fold hash')
            else:
                aids = np.full(n, 'final')
            if not np.array_equal(aids.astype(str), lab['adapter_id'].astype(str)):
                raise SystemExit(f'{name}: parent adapter ids differ from the reconstructed assignment')
            Xpost = {h: E[f'X_post_h{h}'] for h in K.HS}
            valid = {h: E[f'valid_h{h}'] for h in K.HS}
            arrays = dict(match=E['match'], s=E['s'], L=E['L'], q_pre=E['q_pre'], sub_role=E['sub_role'], pre_ok=E['pre_ok'],
                          cohort=co['cohort'], fine=co['fine'], frozen_manifest_sha256=np.asarray(fz_sha), state_version=np.asarray(C.STATE_VERSION))
            for h in K.HS:
                arrays[f'endpoint_h{h}'] = E[f'endpoint_h{h}']
                arrays[f'valid_h{h}'] = E[f'valid_h{h}']
            recomputed = {}
            for model in K.ALL_V:
                st.update('running', f'{name}_{model}', processed=si, total=len(sets))
                ads = {aid: adapter(model, aid) for aid in sorted(set(aids.tolist()))}
                out = A.generate_labels(aids, ads, names, C.STATE_VERSION, E['X_pre'], E['pre_ok'], Xpost, valid)
                if model == 'A':
                    recomputed = out
                    chk = dict(p_pre_bitwise_equal=bool(np.array_equal(out['p_pre'], lab['p_pre'], equal_nan=True)),
                               p_pre_max_abs_diff=float(np.nanmax(np.abs(out['p_pre'] - lab['p_pre']))) if np.isfinite(lab['p_pre']).any() else None)
                    for h in K.HS:
                        v = valid[h] == 1
                        chk[f'h{h}_p_post_bitwise_equal'] = bool(np.array_equal(out['p_post'][h], lab[f'p_post_h{h}'], equal_nan=True))
                        chk[f'h{h}_delta_max_abs_diff'] = float(np.max(np.abs(out['delta'][h][v] - lab[f'delta_h{h}'][v]))) if v.any() else None
                        chk[f'h{h}_Y_equal'] = bool(np.array_equal(out['Y'][h], lab[f'Y_h{h}']))
                    ident['A_recomputation'] = chk
                    if not all(chk[f'h{h}_Y_equal'] for h in K.HS) or (chk['p_pre_max_abs_diff'] or 0) > 1e-12:
                        raise SystemExit(f'{name}: primary A labels not reproduced: {chk}')
                    arrays['p_pre_A'] = lab['p_pre']
                    for h in K.HS:
                        arrays[f'p_post_h{h}_A'] = lab[f'p_post_h{h}']
                        arrays[f'delta_h{h}_A'] = lab[f'delta_h{h}']
                        arrays[f'Y_h{h}_A'] = lab[f'Y_h{h}']
                    arrays['adapter_id_A'] = lab['adapter_id']
                    arrays['adapter_sha256_A'] = lab['adapter_sha256']
                else:
                    arrays[f'p_pre_{model}'] = out['p_pre']
                    for h in K.HS:
                        arrays[f'p_post_h{h}_{model}'] = out['p_post'][h]
                        arrays[f'delta_h{h}_{model}'] = out['delta'][h]
                        arrays[f'Y_h{h}_{model}'] = out['Y'][h]
                    arrays[f'adapter_id_{model}'] = out['adapter_used']
                    arrays[f'adapter_sha256_{model}'] = np.asarray([paths[model][a][1] for a in out['adapter_used'].tolist()])
                    arrays[f'model_version_{model}'] = np.asarray(f'lv20260915_{model}_wp_v2_participant_order')
                    arrays[f'calibration_{model}'] = np.asarray(fz['models'][model]['chosen'])
                if not np.array_equal(out['adapter_used'], aids):
                    raise SystemExit(f'{name}/{model}: adapter used differs from assignment')
            recomputed = None
            ssum = dict(identity=ident, rows=n, matches=int(len(np.unique(E['match']))))
            for h in K.HS:
                v = valid[h] == 1
                ssum[f'h{h}'] = dict(valid_rows=int(v.sum()),
                                     positive={m: int(np.sum(arrays[f'Y_h{h}_{m}'][v] == 1)) for m in K.ALL_V},
                                     exact_zero_delta={m: int(np.sum(arrays[f'delta_h{h}_{m}'][v] == 0)) for m in K.ALL_V},
                                     nonfinite_valid_delta={m: int(np.sum(~np.isfinite(arrays[f'delta_h{h}_{m}'][v]))) for m in K.ALL_V},
                                     label_formula_errors={m: int(np.sum(arrays[f'Y_h{h}_{m}'][v] != (arrays[f'delta_h{h}_{m}'][v] > 0))) for m in K.ALL_V},
                                     invalid_rows_label_minus1={m: bool(np.all(arrays[f'Y_h{h}_{m}'][~v] == -1)) for m in K.ALL_V})
            lab_sha = C.save_npz(K.OUT / 'alt_labels' / f'{name}_alt_labels.npz', **arrays)
            obs = dict(match=E['match'], s=E['s'], pre_snapshot=E['pre_snapshot'])
            for h in K.HS:
                o = A.observed_from_states(names, E['X_pre'], E[f'X_post_h{h}'])
                for k, val in o.items():
                    obs[f'{k}_h{h}'] = val
                obs[f'post_snapshot_h{h}'] = E[f'post_snapshot_h{h}']
            obs['units'] = np.asarray('kills/deaths: event counts; *_norm: normalized cache units (totalGold/25000, xp/20000); '
                                      'state_*: StateV2 count differences')
            obs_sha = C.save_npz(K.OUT / 'observed' / f'{name}_observed.npz', **obs)
            ssum.update(alt_labels_sha256=lab_sha, observed_sha256=obs_sha)
            summary[name] = ssum
            E = None
            arrays = None
            obs = None
            st.log(f'{name}: rows={n} ' + ' '.join(f'h90 pos {m}={ssum["h90"]["positive"][m]}' for m in K.ALL_V))
        man_path = K.OUT / 'alt_labels' / 'alt_labels_manifest.json'
        prev = C.read_json(man_path)['summaries'] if man_path.exists() else {}
        prev.update(summary)
        C.write_json(man_path, dict(role=K.ROLE_TAG, frozen_manifest_sha256=fz_sha, summaries=prev, written_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        st.update('complete', 'labels', processed=len(sets), total=len(sets), next_step='analysis')
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
