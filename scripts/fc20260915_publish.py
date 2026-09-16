"""Publish outcome-free query keys, sampling census, label/endpoint distributions by fold/role, external structural-zero
features and non-comparable pilot context (outputs/full_corpus_training_20260915/published). Read-only over artifacts.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import fc20260915_data as D  # noqa: E402

OUT = C.OUT
PUB = OUT / 'published'
EXT = ('KR_16.13', 'KR_16.14_pilot', 'KR_16.15', 'NA1_16.13')


def q(x, ps=(0, .01, .05, .25, .5, .75, .95, .99, 1)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.quantile(x, ps).tolist() if len(x) else None


def main():
    PUB.mkdir(parents=True, exist_ok=True)
    L = D.Layout(False)
    keys = {k: [] for k in ('set', 'match', 'role', 'query_ms', 'bucket', 'snapshot_ms', 'is_bucket_sample')}
    census = {}
    for set_id in ('MAIN',) + EXT:
        for sp, _op, cm in D.chunk_paths(L, set_id):
            with np.load(sp, allow_pickle=False) as z:
                role = dict(zip(z['m_match'].tolist(), z['m_sub_role'].tolist()))
                vm = z['v_match']
                if not len(vm):
                    continue
                keys['set'].append(np.full(len(vm), set_id, dtype='U16'))
                keys['match'].append(vm)
                keys['role'].append(np.asarray([role[m] for m in vm.tolist()], dtype='U10'))
                for k in ('query_ms', 'bucket', 'snapshot_ms', 'is_bucket_sample'):
                    keys[k].append(z['v_' + k])
    K = {k: np.concatenate(v) for k, v in keys.items()}
    K['frame_age_ms'] = K['query_ms'] - K['snapshot_ms']
    C.save_npz(PUB / 'v_query_keys.npz', **K, note=np.asarray('outcome-free V query keys; is_bucket_sample=1 rows are the fit/calibration/selection/evaluation '
                                                              'sample; TEST/external also store full minute grids'))
    for s in ('MAIN',) + EXT:
        ms = K['set'] == s
        for r in sorted(set(K['role'][ms].tolist())):
            m = ms & (K['role'] == r)
            b = m & (K['is_bucket_sample'] == 1)
            nmatch = len(np.unique(K['match'][b]))
            per_match = np.bincount(np.unique(K['match'][b], return_inverse=True)[1]) if b.any() else np.zeros(0)
            census[f'{s}/{r}'] = dict(bucket_queries=int(b.sum()), matches=int(nmatch), stored_rows=int(m.sum()),
                                      queries_per_match_quantiles=q(per_match, (0, .25, .5, .75, 1)),
                                      bucket_index_counts={str(k): int(v) for k, v in sorted(Counter(K['bucket'][b].tolist()).items())},
                                      frame_age_s_quantiles=q(K['frame_age_ms'][b] / 1000.),
                                      query_minute_quantiles=q(K['query_ms'][b] / 60000.))
    C.write_json(PUB / 'v_query_census.json', dict(role=C.ROLE_TAG, rule='one sha256-ranked minute query per nonempty 5-minute bucket; weights equal per match',
                                                   census=census))
    # label / endpoint distributions by TRAIN fold and VALIDATION role
    dist = {}
    for name in ('MAIN_TRAIN', 'MAIN_VALIDATION', 'MAIN_TEST') + tuple(f'EXT_{s}' for s in EXT):
        with np.load(OUT / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
            lab = {k: z[k] for k in z.files if not k.endswith('counts') and k != 'count_keys'}
        for r in sorted(set(lab['sub_role'].tolist())):
            rm = lab['sub_role'] == r
            cell = dict(adapter=sorted(set(lab['adapter_id'][rm].tolist())), rows=int(rm.sum()), matches=int(len(np.unique(lab['match'][rm]))),
                        p_pre_quantiles=q(lab['p_pre'][rm]), pre_snapshot_age_s_quantiles=q(lab['pre_snapshot_age_s_AUDIT'][rm]))
            for h in C.HORIZONS_S:
                v = rm & (lab[f'valid_h{h}'] == 1)
                w = C.weights(lab['match'][v]) if v.any() else None
                cell[f'h{h}'] = dict(valid_rows=int(v.sum()), positive_rate_rows=float(np.mean(lab[f'Y_h{h}'][v] == 1)) if v.any() else None,
                                     positive_rate_match_weighted=float(np.average(lab[f'Y_h{h}'][v] == 1, weights=w)) if v.any() else None,
                                     exact_zero=int(np.sum(lab[f'delta_h{h}'][v] == 0)), delta_quantiles=q(lab[f'delta_h{h}'][v]),
                                     abs_delta_le_0p01_rate=float(np.mean(np.abs(lab[f'delta_h{h}'][v]) <= .01)) if v.any() else None,
                                     endpoint_minus_L_s_quantiles=q((lab[f'endpoint_h{h}'][v] - lab['L'][v]) / 1000.),
                                     post_snapshot_age_s_quantiles=q(lab[f'post_snapshot_age_s_h{h}'][v]),
                                     post_snapshot_after_L_rate=float(np.mean(lab[f'post_snapshot_after_L_h{h}'][v] == 1)) if v.any() else None,
                                     end_reasons=dict(Counter(lab[f'reasons_h{h}'][v].tolist())))
            dist[f'{name}/{r}'] = cell
    C.write_json(PUB / 'label_endpoint_distributions.json', dict(role=C.ROLE_TAG, note='TRAIN fold cells use their held-out-fold adapter; others final V',
                                                                 distributions=dist))
    # structural zero features externally vs in-distribution validation roles
    rv = C.read_json(OUT / 'eval' / 'results_v.json')['results']
    names = L.manifest('MAIN')['names']
    ref = np.mean([np.asarray(list(rv[f'MAIN_{r}']['feature_availability']['zero_fraction_bucket_rows'].values())) for r in ('V_SELECT', 'V_CAL', 'Q_CAL', 'Q_SELECT')], axis=0)
    sz = {}
    for s in EXT:
        zf = np.asarray(list(rv[f'EXT_{s}']['feature_availability']['zero_fraction_bucket_rows'].values()))
        sz[s] = dict(always_zero_externally_but_not_in_validation=[n for n, a, b in zip(names, zf, ref) if a == 1.0 and b < 1.0],
                     zero_fraction_increase_gt_0p2=[dict(feature=n, validation=float(b), external=float(a)) for n, a, b in zip(names, zf, ref) if a - b > .2])
    C.write_json(PUB / 'external_structural_zero_features.json', dict(role=C.ROLE_TAG, reference='mean zero fraction over MAIN validation roles (15.15)',
                                                                       interpretation='always-zero externally = class/feature not observed in that set; not verified as a genuine in-game zero',
                                                                       sets=sz))
    # pilot context
    p4 = C.read_json(C.ROOT / 'outputs/q_v3_baselines/results.json')
    p1 = C.read_json(C.ROOT / 'outputs/independent_v2_participant_order/results.json')
    ctx = dict(role='NON-COMPARABLE CONTEXT: different labeler (pilot V), cohort (9,198 matches / 26,693 engagements), splits and label scale; '
                    'not an improvement or regression claim',
               p4_A90_test={c: {k: p4['metrics']['A90'][c]['test'][k] for k in ('n', 'matches', 'auc', 'brier', 'logloss')} for c in p4['metrics']['A90']},
               p4_selected=p4['selections']['A90']['chosen'],
               p1_test_one_minute=p1.get('test_one_minute_V2'),
               full_corpus_h90_test_chosen=C.read_json(OUT / 'eval' / 'results_q.json')['results']['MAIN_TEST']['h90']['metrics'][
                   C.read_json(OUT / 'selection' / 'q_h90.json')['chosen']])
    C.write_json(PUB / 'pilot_context_noncomparable.json', ctx)
    print('published', time.strftime('%H:%M:%S'), {s: len(v['always_zero_externally_but_not_in_validation']) for s, v in sz.items()})


if __name__ == '__main__':
    main()
