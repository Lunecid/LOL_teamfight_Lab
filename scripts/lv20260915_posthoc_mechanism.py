"""Label validity POST-HOC mechanism check (added after seeing results; NOT prespecified).

Observation that motivated it: on rows whose post frame equals the pre frame (stale_same_frame), B_econ labels were
almost always positive in T. B_econ inputs are time_minutes, time_minutes_sq and frame snapshot resources; with the
same frame, only the two time columns change, so sign(delta_B_econ) should equal the sign of the time-only logit
change of the adapter used (sigmoid/raw calibration is monotone). This script verifies that identity from the frozen
adapter coefficients, and tabulates the stale share by set/cohort. No model is fitted or changed.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import lv20260915_analysis as A  # noqa: E402
import lv20260915_common as K  # noqa: E402

import time  # noqa: E402

import numpy as np  # noqa: E402


def time_logit_change(ad, q_pre, q_post):
    feats = ad.feature_names
    ct = ad.base.named_steps['preprocess']
    num_cols = [feats[i] for nm, _, ix in ct.transformers_ if nm == 'numeric' for i in ix]
    sc = ct.named_transformers_['numeric'].named_steps['scale']
    coef = ad.base.named_steps['model'].coef_[0]
    i_t, i_t2 = num_cols.index('time_minutes'), num_cols.index('time_minutes_sq')   # numeric block comes first in the transformer
    t0, t1 = q_pre / 60000., q_post / 60000.
    return coef[i_t] * (t1 - t0) / sc.scale_[i_t] + coef[i_t2] * (t1 ** 2 - t0 ** 2) / sc.scale_[i_t2]


def main():
    st = K.Status('posthoc_mechanism')
    K.log_command()
    fz = C.read_json(K.frozen_manifest_path())
    P = K.ParentReadOnly()
    ads = {'final': K.load_alt_adapter(K.OUT / fz['models']['B_econ']['final_path'], fz['models']['B_econ']['final_sha256'], 'B_econ')}
    for k in range(C.N_FOLDS):
        ads[f'oof_fold{k}'] = K.load_alt_adapter(K.OUT / fz['models']['B_econ']['oof_paths'][f'fold{k}'], fz['models']['B_econ']['oof_sha256'][f'fold{k}'], 'B_econ')
    res = dict(status='POST_HOC_NOT_PRESPECIFIED', added_at=time.strftime('%Y-%m-%d %H:%M:%S'),
               motivation='B_econ P(Y=1) on stale_same_frame T rows was ~0.98 in MAIN_TEST analysis', sets={})
    for name in K.LABEL_SETS:
        with np.load(K.OUT / 'alt_labels' / f'{name}_alt_labels.npz', allow_pickle=False) as z:
            Z = {k: z[k] for k in ('match', 's', 'q_pre', 'endpoint_h90', 'valid_h90', 'cohort', 'adapter_id_B_econ', 'delta_h90_B_econ',
                                   'Y_h90_B_econ', 'Y_h90_A', 'Y_h90_B_reg', 'delta_h90_A')}
        with np.load(K.OUT / 'observed' / f'{name}_observed.npz', allow_pickle=False) as z:
            pre_snap, post_snap = z['pre_snapshot'], z['post_snapshot_h90']
        v = Z['valid_h90'] == 1
        stale = v & (pre_snap == post_snap)
        tl = np.full(len(v), np.nan)
        for aid, ad in ads.items():
            m = stale & (Z['adapter_id_B_econ'] == aid)
            if m.any():
                tl[m] = time_logit_change(ad, Z['q_pre'][m].astype(float), Z['endpoint_h90'][m].astype(float))
        cell = {}
        for c, code in (('E', None), ('T', 1), ('N', 0)):
            cm = v if code is None else v & (Z['cohort'] == code)
            sm = stale & cm
            g = Z['match'][sm]
            cell[c] = dict(valid_rows=int(cm.sum()), stale_rows=int(sm.sum()), stale_share=float(sm.sum() / max(1, cm.sum())),
                           sign_identity_rows=int(np.sum((Z['delta_h90_B_econ'][sm] > 0) == (tl[sm] > 0))),
                           sign_identity_all=bool(np.all((Z['delta_h90_B_econ'][sm] > 0) == (tl[sm] > 0))),
                           P_Y1_B_econ_stale=A.wrate(Z['Y_h90_B_econ'][sm] == 1, g)[1] if sm.any() else None,
                           P_Y1_A_stale=A.wrate(Z['Y_h90_A'][sm] == 1, g)[1] if sm.any() else None,
                           disagree_A_B_econ_stale=A.wrate(Z['Y_h90_A'][sm] != Z['Y_h90_B_econ'][sm], g)[1] if sm.any() else None,
                           disagree_A_B_econ_nonstale=A.wrate(Z['Y_h90_A'][cm & ~stale] != Z['Y_h90_B_econ'][cm & ~stale], Z['match'][cm & ~stale])[1],
                           disagree_A_B_reg_nonstale=A.wrate(Z['Y_h90_A'][cm & ~stale] != Z['Y_h90_B_reg'][cm & ~stale], Z['match'][cm & ~stale])[1],
                           time_only_logit_change_quantiles=np.quantile(tl[sm], [0, .5, 1]).tolist() if sm.any() else None)
        res['sets'][name] = cell
    res['all_sign_identity'] = all(c['sign_identity_all'] for s in res['sets'].values() for c in s.values() if c['stale_rows'])
    res['interpretation'] = ('On stale-frame rows the B_econ label is a function of game time only (inputs other than time are identical); '
                             'this is a property of the narrower snapshot comparator, not evidence about engagement outcomes.')
    C.write_json(K.OUT / 'results' / 'posthoc_mechanism_B_econ_stale_frames.json', res)
    amend_p = K.OUT / 'protocol_amendments.json'
    amend = C.read_json(amend_p) if amend_p.exists() else dict(note='post-hoc additions after protocol.json; protocol.json itself unchanged', items=[])
    if not any(i['id'] == 'PH1' for i in amend['items']):
        amend['items'].append(dict(id='PH1', added_at=res['added_at'], status='POST_HOC_NOT_PRESPECIFIED', after_seeing='MAIN_TEST observed-direction table',
                                   what='B_econ stale-frame time-only sign identity and stale-share table (scripts/lv20260915_posthoc_mechanism.py)',
                                   affects='interpretation only; no label, model, selection or primary result changed'))
    C.write_json(amend_p, amend)
    st.update('complete', 'posthoc_mechanism', all_sign_identity=res['all_sign_identity'], next_step='post-run checks')
    return 0


if __name__ == '__main__':
    sys.exit(main())
