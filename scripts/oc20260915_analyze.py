"""Objective-channel ablation stage T3: A vs B_noobj agreement, bootstrap, strata, objective subgroups, q target dependence.

Requires frozen_manifest.json and labels/. Pure arithmetic on saved arrays (parent label counters, this study's labels,
label-validity observed StateV2 differences with exact key verification). No fitting, selection or threshold change.
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

import csv  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

COHORTS = {'E': None, 'T': 1, 'N': 0}
HPAIRS = ((60, 90), (90, 120), (60, 120))
OWNED = ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan', 'soul_owned')
LAB_KEYS = (['match', 's', 'p_pre', 'pre_snapshot', 'during_counts', 'after_counts_h90', 'count_keys']
            + [f'{k}_h{h}' for h in K.HS for k in ('endpoint', 'valid', 'reasons', 'post_snapshot', 'post_snapshot_after_L', 'delta', 'Y')])


def rates(ind, g):
    r, m = A.wrate(ind, g)
    return dict(row=r, match_weighted=m)


def wmeans(x, g):
    if not len(x):
        return dict(row=None, match_weighted=None)
    return dict(row=float(np.mean(x)), match_weighted=float(np.average(x, weights=C.weights(g))))


def cell(mask, g, L, h, oriented_sign=None):
    gg = g[mask]
    out = A.cell_meta(gg)
    if out['empty']:
        return out
    dA, dB = L[f'delta_h{h}_A'][mask], L[f'delta_h{h}_B'][mask]
    out['disagreement'] = rates(L[f'Y_h{h}_A'][mask] != L[f'Y_h{h}_B'][mask], gg)
    out['P_Y1_A'] = rates(L[f'Y_h{h}_A'][mask] == 1, gg)
    out['P_Y1_B'] = rates(L[f'Y_h{h}_B'][mask] == 1, gg)
    out['mean_delta_A'] = wmeans(dA, gg)
    out['mean_delta_B'] = wmeans(dB, gg)
    out['mean_abs_delta_B_minus_A'] = wmeans(np.abs(dB - dA), gg)
    out['small_abs_delta_le_0.01_A'] = rates(np.abs(dA) <= .01, gg)
    out['small_abs_delta_le_0.01_B'] = rates(np.abs(dB) <= .01, gg)
    if oriented_sign is not None:
        sgn = oriented_sign[mask]
        out['oriented_delta_A'] = wmeans(sgn * dA, gg)
        out['oriented_delta_B'] = wmeans(sgn * dB, gg)
        out['oriented_delta_A_minus_B'] = wmeans(sgn * (dA - dB), gg)
    return out


def load(P, name):
    lab = P.label_file(name, f'objective ablation analysis {name}', keys=LAB_KEYS)
    with np.load(K.OUT / 'labels' / f'{name}_B_noobj_labels.npz', allow_pickle=False) as z:
        L = {k: z[k] for k in z.files}
    with np.load(K.LVO / 'observed' / f'{name}_observed.npz', allow_pickle=False) as z:
        obs = {k: z[k] for k in z.files if k in ('match', 's', 'pre_snapshot', 'post_snapshot_h90') or (k.startswith('state_') and k.endswith('_h90'))}
    keys = dict(labels_equal_parent=bool(np.array_equal(L['match'], lab['match']) and np.array_equal(L['s'], lab['s'])),
                observed_equal_parent=bool(np.array_equal(obs['match'], lab['match']) and np.array_equal(obs['s'], lab['s'])),
                observed_snapshots_equal=bool(np.array_equal(obs['pre_snapshot'], lab['pre_snapshot']) and np.array_equal(obs['post_snapshot_h90'], lab['post_snapshot_h90'])))
    for h in K.HS:
        keys[f'A_h{h}_equal_parent'] = bool(np.array_equal(L[f'Y_h{h}_A'], lab[f'Y_h{h}']) and np.array_equal(L[f'delta_h{h}_A'], lab[f'delta_h{h}'], equal_nan=True))
        keys[f'valid_endpoint_h{h}_equal'] = bool(np.array_equal(L[f'valid_h{h}'], lab[f'valid_h{h}']) and np.array_equal(L[f'endpoint_h{h}'], lab[f'endpoint_h{h}']))
    if not all(keys.values()):
        raise SystemExit(f'{name}: key identity failed {keys}')
    return lab, L, obs, keys


def analyze_set(P, name):
    lab, L, obs, keys = load(P, name)
    g = lab['match'].astype(str)
    ck = [str(k) for k in lab['count_keys']]
    cohort = L['cohort']
    cm = {c: (np.ones(len(g), bool) if code is None else cohort == code) for c, code in COHORTS.items()}
    res = dict(rows=int(len(g)), matches=int(len(np.unique(g))), key_identity=keys,
               valid_h90={c: int(np.sum((lab['valid_h90'] == 1) & cm[c])) for c in COHORTS}, unknown_cohort_valid_h90=int(np.sum((lab['valid_h90'] == 1) & (cohort == -1))))
    # agreement
    unchanged = (lab['endpoint_h60'] == lab['endpoint_h90']) & (lab['endpoint_h90'] == lab['endpoint_h120'])
    res['agreement'] = {}
    for h in K.HS:
        v = lab[f'valid_h{h}'] == 1
        res['agreement'][f'h{h}'] = {}
        for c in COHORTS:
            m = v & cm[c]
            a = A.agreement_cell(L[f'Y_h{h}_A'][m], L[f'Y_h{h}_B'][m], L[f'delta_h{h}_A'][m], L[f'delta_h{h}_B'][m], g[m])
            for sub, sm in (('endpoint_unchanged_60_90_120', m & unchanged), ('endpoint_changed_across_horizons', m & ~unchanged)):
                cc = A.cell_meta(g[sm])
                if not cc['empty']:
                    cc['disagreement_row'], cc['disagreement_match_weighted'] = A.wrate(L[f'Y_h{h}_A'][sm] != L[f'Y_h{h}_B'][sm], g[sm])
                a[sub] = cc
            res['agreement'][f'h{h}'][c] = a
    res['horizon'] = {}
    v_all = (lab['valid_h60'] == 1) & (lab['valid_h90'] == 1) & (lab['valid_h120'] == 1)
    for model in ('A', 'B'):
        res['horizon'][model] = {}
        for c in COHORTS:
            res['horizon'][model][c] = {}
            for ha, hb in HPAIRS:
                m = v_all & cm[c]
                dis = L[f'Y_h{ha}_{model}'] != L[f'Y_h{hb}_{model}']
                same = lab[f'endpoint_h{ha}'] == lab[f'endpoint_h{hb}']
                res['horizon'][model][c][f'{ha}_vs_{hb}'] = dict(
                    all=dict(A.cell_meta(g[m]), **dict(zip(('row', 'match_weighted'), A.wrate(dis[m], g[m])))),
                    changed_endpoint=dict(A.cell_meta(g[m & ~same]), **dict(zip(('row', 'match_weighted'), A.wrate(dis[m & ~same], g[m & ~same])))),
                    unchanged_endpoint=dict(rows=int(np.sum(m & same)), disagreements=int(np.sum(dis[m & same]))))
    v90 = lab['valid_h90'] == 1
    res['bootstrap_h90'] = {c: A.paired_match_bootstrap(g[v90 & cm[c]], {'A_vs_B_noobj': L['Y_h90_A'][v90 & cm[c]] != L['Y_h90_B'][v90 & cm[c]]},
                                                        K.BOOT_REPS, K.BOOT_SEED) for c in COHORTS}
    fam = {'abs_delta_A': A.bin_abs_delta(np.nan_to_num(L['delta_h90_A'])), 'start_minutes': A.bin_start_minutes(lab['s']),
           'p_pre_A': A.bin_p_pre(np.nan_to_num(L['p_pre_A'])), 'ending_reason': lab['reasons_h90'].astype(str),
           'post_snapshot_after_L': lab['post_snapshot_after_L_h90'].astype(str), 'same_pre_post_frame': L['same_pre_post_frame_h90'].astype(str)}
    res['strata_h90'] = {}
    for c in COHORTS:
        m0 = v90 & cm[c]
        res['strata_h90'][c] = {fn: {b: cell(m0 & (arr == b), g, L, 90) for b in sorted(set(arr[m0].tolist()))} for fn, arr in fam.items()}
    # objectives
    dur = lab['during_counts']
    aft = lab['after_counts_h90']
    windows = {'full_(q_pre,e]': dur + aft, 'after_(L,e]': aft}
    res['objectives_h90'] = dict(note='overlapping descriptive subgroups; not causal event value, not timing counterfactual, not immediate reward; '
                                      'reference rows may carry pre-existing objective history', cohorts={})
    team_state = {}
    for d in A.DRAGON_ELEMENTS:
        team_state[d] = (obs[f'state_dragon_{d}_blue_h90'], obs[f'state_dragon_{d}_red_h90'])
    element_team_mismatch = {}
    for d in A.DRAGON_ELEMENTS:
        tot = windows['full_(q_pre,e]'][:, ck.index(f'dragon_{d}')]
        b, r = team_state[d]
        element_team_mismatch[d] = int(np.sum(v90 & ((b + r) > tot)))
    res['element_state_team_exceeds_counter_rows'] = element_team_mismatch
    for c in COHORTS:
        base = v90 & cm[c]
        oc = {}
        for wname, arr in windows.items():
            ow = {}
            known_any = np.zeros(len(g), bool)
            for obj in OWNED:
                b, r, tot, unk = A.objective_team_columns(arr, ck, obj)
                known_any |= (b + r) > 0
                sgn = np.where((b > 0) & (r == 0), 1.0, np.where((r > 0) & (b == 0), -1.0, 0.0))
                ow[obj] = dict(blue_only=cell(base & (b > 0) & (r == 0), g, L, 90, sgn), red_only=cell(base & (r > 0) & (b == 0), g, L, 90, sgn),
                               single_team_oriented=cell(base & (sgn != 0), g, L, 90, sgn),
                               both_teams_DIAGNOSTIC=cell(base & (b > 0) & (r > 0), g, L, 90), multiple_DIAGNOSTIC=cell(base & (tot >= 2), g, L, 90),
                               unknown_team_DIAGNOSTIC=cell(base & (unk > 0), g, L, 90))
            ow['soul_teamid0_unassigned_DIAGNOSTIC'] = dict(present=cell(base & (arr[:, ck.index('soul_teamid0_unassigned')] > 0), g, L, 90))
            for d in A.DRAGON_ELEMENTS:
                cnt = arr[:, ck.index(f'dragon_{d}')]
                sb, sr = team_state[d]
                if wname.startswith('full'):
                    usable = np.ones(len(g), bool)
                else:
                    usable = dur[:, ck.index(f'dragon_{d}')] == 0     # all interval acquisitions of this element after L
                bt = usable & (cnt > 0) & (sb > 0) & (sr == 0)
                rt = usable & (cnt > 0) & (sr > 0) & (sb == 0)
                sgn = np.where(bt, 1.0, np.where(rt, -1.0, 0.0))
                ow[f'dragon_{d}'] = dict(present_any_team=cell(base & (cnt > 0), g, L, 90), blue_only=cell(base & bt, g, L, 90, sgn),
                                         red_only=cell(base & rt, g, L, 90, sgn), single_team_oriented=cell(base & (sgn != 0), g, L, 90, sgn),
                                         both_teams_DIAGNOSTIC=cell(base & usable & (cnt > 0) & (sb > 0) & (sr > 0), g, L, 90),
                                         multiple_DIAGNOSTIC=cell(base & (cnt >= 2), g, L, 90),
                                         unknown_team_DIAGNOSTIC=cell(base & usable & (cnt > (sb + sr)), g, L, 90),
                                         team_ambiguous_mixed_intervals_DIAGNOSTIC=cell(base & ~usable & (cnt > 0), g, L, 90))
            ow['no_known_team_acquisition_reference'] = dict(rows=cell(base & ~known_any, g, L, 90))
            oc[wname] = ow
        res['objectives_h90']['cohorts'][c] = oc
    return res


def q_dependence():
    with np.load(K.OUT / 'labels' / 'MAIN_TEST_B_noobj_labels.npz', allow_pickle=False) as z:
        L = {k: z[k] for k in ('match', 's', 'valid_h90', 'cohort', 'Y_h90_A', 'Y_h90_B')}
    lk = {(m, int(s)): i for i, (m, s) in enumerate(zip(L['match'].tolist(), L['s'].tolist()))}
    audit = C.read_json(K.AUDIT / 'calculations.json')['metrics']
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    out = dict(scope='frozen main h90 specialist q (trained on A labels) scored against A and B_noobj labels; target dependence, no refit/ranking', cohorts={})
    for c, code in (('T', 1), ('N', 0)):
        with np.load(K.CR / 'eval' / 'predictions' / f'A_MAIN_TEST_h90_{c}.npz', allow_pickle=False) as z:
            Z = {k: z[k] for k in z.files}
        col = 'spec_' + str(Z['specialist_chosen'])
        idx = np.asarray([lk[(m, int(s))] for m, s in zip(Z['match'].tolist(), Z['s_ms'].tolist())])
        info = dict(rows=int(len(idx)), unique=len(set(idx.tolist())) == len(idx), valid=bool(np.all(L['valid_h90'][idx] == 1)),
                    cohort_ok=bool(np.all(L['cohort'][idx] == code)), covers_all=int(len(idx)) == int(np.sum((L['valid_h90'] == 1) & (L['cohort'] == code))),
                    y_equals_A=bool(np.array_equal(Z['y'], L['Y_h90_A'][idx].astype(np.int64))))
        if not all(v for v in info.values() if isinstance(v, bool)):
            raise SystemExit(f'q join failed {c}: {info}')
        _, inv, cnt = np.unique(Z['match'], return_inverse=True, return_counts=True)
        w = 1 / cnt[inv]
        p = Z[col]
        sc = {}
        for lab in ('A', 'B'):
            y = L[f'Y_h90_{lab}'][idx].astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sc[f'Y_{lab}'] = dict(auc=float(roc_auc_score(y, p, sample_weight=w)), brier=float(brier_score_loss(y, p, sample_weight=w)),
                                      logloss=float(log_loss(y, p, labels=[0, 1], sample_weight=w)), positive_rate_match_weighted=float(np.average(y, weights=w)),
                                      auc_row=float(roc_auc_score(y, p)), brier_row=float(brier_score_loss(y, p)))
        ref = audit[c][col]
        rep = max(abs(sc['Y_A'][k] - ref[k]) for k in ('auc', 'brier', 'logloss'))
        if rep > 1e-12:
            raise SystemExit(f'q A scores not reproduced {c}: {rep}')
        out['cohorts'][c] = dict(specialist=col, keys=info, scores=sc, B_minus_A={k: sc['Y_B'][k] - sc['Y_A'][k] for k in ('auc', 'brier', 'logloss')},
                                 A_reproduction_max_abs_diff=rep, label_disagreement_on_q_rows=float(np.mean(L['Y_h90_A'][idx] != L['Y_h90_B'][idx])))
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


def flat(prefix, c):
    r = dict(prefix, rows=c.get('rows'), matches=c.get('matches'), sparse_lt30_matches=c.get('sparse_lt30_matches'))
    for k, v in c.items():
        if isinstance(v, dict) and 'row' in v:
            r[f'{k}_row'] = v['row']
            r[f'{k}_match_weighted'] = v['match_weighted']
    return r


def main():
    st = K.Status('analyze')
    K.log_command()
    try:
        if not K.frozen_manifest_path().exists():
            raise SystemExit('frozen_manifest.json required')
        P = K.parent_reader()
        results = {}
        for i, name in enumerate(K.LABEL_SETS):
            st.update('running', f'analyze_{name}', processed=i, total=len(K.LABEL_SETS))
            results[name] = analyze_set(P, name)
            C.write_json(K.OUT / 'results' / 'per_set' / f'{name}.json', results[name])
            e = results[name]['bootstrap_h90']
            st.log(f"{name}: h90 dis E={e['E']['estimates']['A_vs_B_noobj']['match_weighted']} T={e['T']['estimates']['A_vs_B_noobj']['match_weighted']}")
        qd = q_dependence()
        C.write_json(K.OUT / 'results' / 'q_target_dependence.json', qd)
        T = K.OUT / 'tables'
        rows = []
        for n, r in results.items():
            for h in K.HS:
                for c in COHORTS:
                    a = r['agreement'][f'h{h}'][c]
                    row = dict(set=n, horizon=h, cohort=c, rows=a['rows'], matches=a['matches'], sparse=a['sparse_lt30_matches'],
                               disagreement_row=a.get('disagreement_row'), disagreement_match_weighted=a.get('disagreement_match_weighted'),
                               positive_A_match_weighted=a.get('positive_rate_A_match_weighted'), positive_B_match_weighted=a.get('positive_rate_B_match_weighted'),
                               exact_zero_A=a.get('exact_zero_delta_A'), exact_zero_B=a.get('exact_zero_delta_B'), delta_diff_mean_B_minus_A=a.get('delta_diff_mean_B_minus_A'),
                               delta_absdiff_mean=a.get('delta_absdiff_mean'), delta_absdiff_p90=a.get('delta_absdiff_p90'), delta_pearson=a.get('delta_pearson'),
                               delta_spearman=a.get('delta_spearman'))
                    if h == 90:
                        e = r['bootstrap_h90'][c]['estimates']['A_vs_B_noobj']
                        row.update(ci95_match_lo=e['match_weighted']['ci95'][0], ci95_match_hi=e['match_weighted']['ci95'][1],
                                   ci95_row_lo=e['row']['ci95'][0], ci95_row_hi=e['row']['ci95'][1])
                    rows.append(row)
        write_csv(T / 'agreement.csv', rows)
        write_csv(T / 'strata_h90.csv', [flat(dict(set=n, cohort=c, family=f, bin=b), cc) for n, r in results.items() for c, fams in r['strata_h90'].items()
                                         for f, bins in fams.items() for b, cc in bins.items()])
        write_csv(T / 'objectives_h90_overlapping.csv', [flat(dict(set=n, cohort=c, window=w, category=o, subgroup=s), cc) for n, r in results.items()
                                                         for c, oc in r['objectives_h90']['cohorts'].items() for w, ow in oc.items()
                                                         for o, subs in ow.items() for s, cc in subs.items()])
        write_csv(T / 'horizon_sign_disagreement.csv', [dict(set=n, model=m, cohort=c, horizons=p, rows=v['all']['rows'], disagreement_row=v['all'].get('row'),
                                                             changed_rows=v['changed_endpoint']['rows'], changed_disagreement_row=v['changed_endpoint'].get('row'),
                                                             unchanged_rows=v['unchanged_endpoint']['rows'], unchanged_disagreements=v['unchanged_endpoint']['disagreements'])
                                                        for n, r in results.items() for m, d in r['horizon'].items() for c, dd in d.items() for p, v in dd.items()])
        write_csv(T / 'q_target_dependence.csv', [dict(cohort=c, specialist=v['specialist'], label=lab, **s) for c, v in qd['cohorts'].items() for lab, s in v['scores'].items()])
        C.write_json(K.OUT / 'results' / 'analysis_index.json', dict(sets=list(results), written_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        st.update('complete', 'analyze', next_step='post-run checks')
        return 0
    except SystemExit as exc:
        st.update('failed', 'analyze', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'analyze', error=repr(exc), next_step='fix and rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
