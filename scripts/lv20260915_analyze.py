"""Label validity stage T3: agreement, bootstrap, strata, observed-outcome and objective diagnostics, q label dependence.

Requires this study's frozen_manifest.json and alt_labels/observed outputs. Pure arithmetic on saved arrays; no model
is fitted or selected. Diagnostics are not label exclusion rules, error bars for V, or semantic accuracy.
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

import csv  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

PAIRS = (('A', 'B_reg'), ('A', 'B_econ'))
COHORTS = {'E': None, 'T': 1, 'N': 0}
HPAIRS = ((60, 90), (90, 120), (60, 120))
LAB_KEYS = (['match', 's', 'L', 'q_pre', 'p_pre', 'pre_snapshot', 'pre_snapshot_age_s_AUDIT', 'during_counts', 'count_keys']
            + [f'{k}_h{h}' for h in K.HS for k in ('endpoint', 'valid', 'reasons', 'post_snapshot', 'post_snapshot_after_L',
                                                   'post_snapshot_age_s', 'after_counts', 'after_raw_kills', 'delta', 'Y')])


def load_set(P, name):
    lab = P.label_file(name, f'label validity analysis {name}', keys=LAB_KEYS)
    with np.load(K.OUT / 'alt_labels' / f'{name}_alt_labels.npz', allow_pickle=False) as z:
        alt = {k: z[k] for k in z.files}
    with np.load(K.OUT / 'observed' / f'{name}_observed.npz', allow_pickle=False) as z:
        obs = {k: z[k] for k in z.files}
    if not (np.array_equal(alt['match'], lab['match']) and np.array_equal(alt['s'], lab['s'])
            and np.array_equal(obs['match'], lab['match']) and np.array_equal(obs['s'], lab['s'])):
        raise SystemExit(f'{name}: alt/observed keys differ from parent labels')
    for h in K.HS:
        if not (np.array_equal(alt[f'Y_h{h}_A'], lab[f'Y_h{h}']) and np.array_equal(alt[f'delta_h{h}_A'], lab[f'delta_h{h}'], equal_nan=True)):
            raise SystemExit(f'{name}: stored A labels differ from parent')
    return lab, alt, obs


def rates(ind, g):
    r, m = A.wrate(ind, g)
    return dict(row=r, match_weighted=m)


def label_cell(mask, g, alt, h, dA_small=True):
    """Standard diagnostic cell for rows in mask (already restricted to valid rows)."""
    gg = g[mask]
    out = A.cell_meta(gg)
    if out['empty']:
        return out
    for m in K.ALL_V:
        out[f'P_Y1_{m}'] = rates(alt[f'Y_h{h}_{m}'][mask] == 1, gg)
    if dA_small:
        out['small_abs_delta_A_le_0.01'] = rates(np.abs(alt[f'delta_h{h}_A'][mask]) <= .01, gg)
    for a, b in PAIRS:
        out[f'disagree_{a}_{b}'] = rates(alt[f'Y_h{h}_{a}'][mask] != alt[f'Y_h{h}_{b}'][mask], gg)
    return out


def analyze_set(P, name, st):
    lab, alt, obs = load_set(P, name)
    g = lab['match'].astype(str)
    keys = [str(k) for k in lab['count_keys']]
    cohort = alt['cohort']
    res = dict(rows=int(len(g)), matches=int(len(np.unique(g))), cohort_counts_valid_h90={
        'T': int(np.sum((lab['valid_h90'] == 1) & (cohort == 1))), 'N': int(np.sum((lab['valid_h90'] == 1) & (cohort == 0))),
        'unknown': int(np.sum((lab['valid_h90'] == 1) & (cohort == -1)))})
    cmask = {c: (np.ones(len(g), bool) if code is None else cohort == code) for c, code in COHORTS.items()}
    # ------------------------------------------------ agreement (all horizons)
    unchanged_all = (lab['endpoint_h60'] == lab['endpoint_h90']) & (lab['endpoint_h90'] == lab['endpoint_h120'])
    res['agreement'] = {}
    for h in K.HS:
        v = lab[f'valid_h{h}'] == 1
        res['agreement'][f'h{h}'] = {}
        for c in COHORTS:
            m = v & cmask[c]
            res['agreement'][f'h{h}'][c] = {}
            for a, b in PAIRS:
                cell = A.agreement_cell(alt[f'Y_h{h}_{a}'][m], alt[f'Y_h{h}_{b}'][m], alt[f'delta_h{h}_{a}'][m], alt[f'delta_h{h}_{b}'][m], g[m])
                for sub, sm in (('endpoint_unchanged_60_90_120', m & unchanged_all), ('endpoint_changed_across_horizons', m & ~unchanged_all)):
                    cc = A.cell_meta(g[sm])
                    if not cc['empty']:
                        cc['disagreement_row'], cc['disagreement_match_weighted'] = A.wrate(alt[f'Y_h{h}_{a}'][sm] != alt[f'Y_h{h}_{b}'][sm], g[sm])
                    cell[sub] = cc
                res['agreement'][f'h{h}'][c][f'{a}_vs_{b}'] = cell
    # ------------------------------------------------ horizon sign disagreement per model
    res['horizon'] = {}
    v_all = (lab['valid_h60'] == 1) & (lab['valid_h90'] == 1) & (lab['valid_h120'] == 1)
    for model in K.ALL_V:
        res['horizon'][model] = {}
        for c in COHORTS:
            res['horizon'][model][c] = {}
            for ha, hb in HPAIRS:
                m = v_all & cmask[c]
                dis = alt[f'Y_h{ha}_{model}'] != alt[f'Y_h{hb}_{model}']
                same = lab[f'endpoint_h{ha}'] == lab[f'endpoint_h{hb}']
                cell = dict(all=dict(A.cell_meta(g[m]), **dict(zip(('row', 'match_weighted'), A.wrate(dis[m], g[m])))),
                            changed_endpoint=dict(A.cell_meta(g[m & ~same]), **dict(zip(('row', 'match_weighted'), A.wrate(dis[m & ~same], g[m & ~same])))),
                            unchanged_endpoint=dict(A.cell_meta(g[m & same]), disagreements=int(np.sum(dis[m & same]))))
                res['horizon'][model][c][f'{ha}_vs_{hb}'] = cell
    # ------------------------------------------------ bootstrap h90
    res['bootstrap_h90'] = {}
    v90 = lab['valid_h90'] == 1
    for c in COHORTS:
        m = v90 & cmask[c]
        ind = {f'{a}_vs_{b}': (alt[f'Y_h90_{a}'][m] != alt[f'Y_h90_{b}'][m]) for a, b in PAIRS}
        res['bootstrap_h90'][c] = A.paired_match_bootstrap(g[m], ind, K.BOOT_REPS, K.BOOT_SEED)
    # ------------------------------------------------ strata h90
    fam = {
        'abs_delta_A': A.bin_abs_delta(np.nan_to_num(alt['delta_h90_A'])),
        'p_pre_A': A.bin_p_pre(np.nan_to_num(alt['p_pre_A'])),
        'start_minutes': A.bin_start_minutes(lab['s']),
        'ending_reason': lab['reasons_h90'].astype(str),
        'post_snapshot_after_L': lab['post_snapshot_after_L_h90'].astype(str),
        'post_frame_age_s': A.bin_age_s(np.nan_to_num(lab['post_snapshot_age_s_h90'], nan=-1)),
        'pre_frame_age_s': A.bin_age_s(np.nan_to_num(lab['pre_snapshot_age_s_AUDIT'], nan=-1)),
        'fine_scale': np.asarray([A.FINE_NAMES[int(x)] for x in alt['fine']]),
    }
    res['strata_h90'] = {}
    for c in COHORTS:
        m0 = v90 & cmask[c]
        res['strata_h90'][c] = {}
        for fname, arr in fam.items():
            res['strata_h90'][c][fname] = {}
            for b in sorted(set(arr[m0].tolist())):
                mm = m0 & (arr == b)
                cell = label_cell(mm, g, alt, 90)
                if not cell['empty']:
                    for a, bb in PAIRS:
                        cell[f'mean_abs_delta_diff_{bb}'] = float(np.mean(np.abs(alt[f'delta_h90_{bb}'][mm] - alt['delta_h90_A'][mm])))
                res['strata_h90'][c][fname][b] = cell
    # ------------------------------------------------ observed outcomes (all horizons; h90 primary)
    res['observed'] = {}
    ck = lab['during_counts'][:, keys.index('champion_kill')]
    for h in K.HS:
        v = lab[f'valid_h{h}'] == 1
        ksign = A.sign3(obs[f'kill_diff_h{h}'])
        gcat = A.resource_category(obs[f'gold_diff_change_norm_h{h}'], obs['pre_snapshot'], obs[f'post_snapshot_h{h}'])
        xcat = A.resource_category(obs[f'xp_diff_change_norm_h{h}'], obs['pre_snapshot'], obs[f'post_snapshot_h{h}'])
        after_ck = lab[f'after_counts_h{h}'][:, keys.index('champion_kill')]
        credited = obs[f'kills_blue_h{h}'] + obs[f'kills_red_h{h}']
        after_L = lab[f'post_snapshot_after_L_h{h}'] == 1
        res['observed'][f'h{h}'] = {}
        for c in COHORTS:
            m = v & cmask[c]
            o = dict(meta=A.cell_meta(g[m]),
                     interval_definitions=dict(during='(q_pre, L]', full='(q_pre, e]', kills_after_L_rows=int(np.sum(after_ck[m] != 0)),
                                               after_raw_kills_nonzero_rows=int(np.sum(lab[f'after_raw_kills_h{h}'][m] != 0)),
                                               kill_diff_during_equals_full=bool(np.all(after_ck[m] == 0))),
                     uncredited_kill_rows=int(np.sum((ck[m] - credited[m]) > 0)),
                     uncredited_kills_total=int(np.sum(ck[m] - credited[m])),
                     credited_exceeds_raw_rows=int(np.sum((ck[m] - credited[m]) < 0)),
                     kill_diff_quantiles=np.quantile(obs[f'kill_diff_h{h}'][m], [0, .05, .25, .5, .75, .95, 1]).tolist() if m.any() else None,
                     kills={}, gold={}, xp={}, gold_by_post_frame_after_L={}, kill_by_gold={})
            for code, nm in A.KILL_CODES.items():
                o['kills'][nm] = label_cell(m & (ksign == code), g, alt, h)
            for code, nm in A.RESOURCE_CODES.items():
                o['gold'][nm] = label_cell(m & (gcat == code), g, alt, h)
                o['xp'][nm] = label_cell(m & (xcat == code), g, alt, h)
                for flag, fn in ((True, 'post_frame_after_L'), (False, 'no_post_frame_after_L')):
                    o['gold_by_post_frame_after_L'].setdefault(fn, {})[nm] = label_cell(m & (gcat == code) & (after_L == flag), g, alt, h, dA_small=False)
            for kc, kn in A.KILL_CODES.items():
                o['kill_by_gold'][kn] = {gn: int(np.sum(m & (ksign == kc) & (gcat == gc))) for gc, gn in A.RESOURCE_CODES.items()}
            nonstale = m & np.isin(gcat, (1, 0, -1))
            o['gold_change_norm_quantiles_nonstale'] = np.quantile(obs[f'gold_diff_change_norm_h{h}'][nonstale], [.05, .25, .5, .75, .95]).tolist() if nonstale.any() else None
            o['gold_change_cache_gold_quantiles_nonstale'] = [x * 25000.0 for x in o['gold_change_norm_quantiles_nonstale']] if nonstale.any() else None
            o['stale_same_frame_rows'] = int(np.sum(m & (gcat == 2)))
            res['observed'][f'h{h}'][c] = o
    # ------------------------------------------------ objectives (h90)
    h = 90
    v = lab['valid_h90'] == 1
    dur = lab['during_counts']
    aft = lab['after_counts_h90']
    windows = {'during_(q_pre,L]': dur, 'after_(L,e]': aft, 'full_(q_pre,e]': dur + aft}
    res['objectives_h90'] = dict(note='subgroups overlap (shared/contested acquisitions and combinations); never sum rows across subgroups',
                                 cohorts={})
    for c in COHORTS:
        m = v & cmask[c]
        oc = {}
        for wname, arr in windows.items():
            ow = {}
            known_any = np.zeros(len(g), bool)
            for obj in A.OBJECTIVES_OWNED:
                blue, red, total, unknown = A.objective_team_columns(arr, keys, obj)
                known_any |= (blue + red) > 0
                ow[obj] = dict(any_known_team=label_cell(m & ((blue + red) > 0), g, alt, h),
                               blue_only=label_cell(m & (blue > 0) & (red == 0), g, alt, h),
                               red_only=label_cell(m & (red > 0) & (blue == 0), g, alt, h),
                               both_teams=label_cell(m & (red > 0) & (blue > 0), g, alt, h),
                               unknown_team_present_DIAGNOSTIC=label_cell(m & (unknown > 0), g, alt, h))
            for d in A.DRAGON_ELEMENTS:
                ow[f'dragon_{d}'] = dict(present_team_not_in_counter=label_cell(m & (arr[:, keys.index(f'dragon_{d}')] > 0), g, alt, h))
                if wname.startswith('full'):
                    b = obs[f'state_dragon_{d}_blue_h90'] > 0
                    r = obs[f'state_dragon_{d}_red_h90'] > 0
                    ow[f'dragon_{d}']['blue_only_from_state'] = label_cell(m & b & ~r, g, alt, h)
                    ow[f'dragon_{d}']['red_only_from_state'] = label_cell(m & r & ~b, g, alt, h)
            ow['soul_teamid0_unassigned_DIAGNOSTIC'] = dict(present=label_cell(m & (arr[:, keys.index('soul_teamid0_unassigned')] > 0), g, alt, h))
            ow['no_known_team_objective_reference'] = dict(rows=label_cell(m & ~known_any, g, alt, h))
            if wname.startswith('full'):
                blue_d = arr[:, keys.index('dragon_blue')]
                red_d = arr[:, keys.index('dragon_red')]
                sb = sum(obs[f'state_dragon_{d}_blue_h90'] for d in A.DRAGON_ELEMENTS)
                sr = sum(obs[f'state_dragon_{d}_red_h90'] for d in A.DRAGON_ELEMENTS)
                ow['dragon_team_counter_vs_state_mismatch_rows'] = int(np.sum(m & ((blue_d != sb) | (red_d != sr))))
            oc[wname] = ow
        res['objectives_h90']['cohorts'][c] = oc
    return res, lab, alt, obs


def q_dependence(P):
    with np.load(K.OUT / 'alt_labels' / 'MAIN_TEST_alt_labels.npz', allow_pickle=False) as z:
        alt = {k: z[k] for k in ('match', 's', 'valid_h90', 'cohort') + tuple(f'Y_h90_{m}' for m in K.ALL_V)}
    lk = {(m, int(s)): i for i, (m, s) in enumerate(zip(alt['match'].tolist(), alt['s'].tolist()))}
    audit = C.read_json(K.AUDIT / 'calculations.json')['metrics']
    rq = C.read_json(K.FC / 'eval' / 'results_q.json')['results']['MAIN_TEST']['h90']['metrics']
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    out = dict(scope='diagnostic label dependence of saved q predictions trained on A labels; no ranking/selection/ceiling claim',
               sets={})
    specs = [('T', K.CR / 'eval' / 'predictions' / 'A_MAIN_TEST_h90_T.npz', audit['T'], 1),
             ('N', K.CR / 'eval' / 'predictions' / 'A_MAIN_TEST_h90_N.npz', audit['N'], 0),
             ('E', K.FC / 'eval' / 'predictions' / 'q_MAIN_TEST_h90.npz', rq, None)]
    for c, path, ref, code in specs:
        with np.load(path, allow_pickle=False) as z:
            Z = {k: z[k] for k in z.files}
        idx = np.asarray([lk[(m, int(s))] for m, s in zip(Z['match'].tolist(), Z['s_ms'].tolist())])
        keyinfo = dict(rows=int(len(idx)), unique=len(set(idx.tolist())) == len(idx), all_valid=bool(np.all(alt['valid_h90'][idx] == 1)),
                       cohort_ok=bool(code is None or np.all(alt['cohort'][idx] == code)),
                       y_equals_primary_Y=bool(np.array_equal(Z['y'], alt['Y_h90_A'][idx].astype(np.int64))))
        expected_rows = int(np.sum((alt['valid_h90'] == 1) & ((alt['cohort'] == code) if code is not None else True)))
        keyinfo['covers_all_valid_rows'] = keyinfo['rows'] == expected_rows
        if not all(v for v in keyinfo.values() if isinstance(v, bool)):
            raise SystemExit(f'q dependence {c}: key/label join failed {keyinfo}')
        g = Z['match']
        _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
        w = 1 / cnt[inv]
        cols = [k for k in Z if k not in ('match', 's_ms', 'y', 'fine', 'specialist_chosen', 'chosen', 'bundle_sha256')]
        chosen = str(Z.get('specialist_chosen', Z.get('chosen')))
        cell = dict(keys=keyinfo, chosen=chosen, columns={}, reproduction={})
        for col in cols:
            p = Z[col]
            cc = {}
            for m in K.ALL_V:
                y = alt[f'Y_h90_{m}'][idx].astype(int)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cc[f'Y_{m}'] = dict(auc=float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else None,
                                        brier=float(brier_score_loss(y, p, sample_weight=w)),
                                        logloss=float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
                                        positive_rate_match_weighted=float(np.average(y, weights=w)))
            for m in K.MODELS:
                cc[f'Y_{m}_minus_Y_A'] = {k: (cc[f'Y_{m}'][k] - cc['Y_A'][k]) if cc[f'Y_{m}'][k] is not None else None for k in ('auc', 'brier', 'logloss')}
            cell['columns'][col] = cc
            if col in ref:
                r = ref[col]
                diffs = {k: abs(cc['Y_A'][k] - r[k]) for k in ('auc', 'brier', 'logloss') if r.get(k) is not None}
                cell['reproduction'][col] = dict(reference=('tog_readiness_audit calculations.json' if c != 'E' else 'full_corpus results_q.json'),
                                                 max_abs_diff=max(diffs.values()), reproduced=max(diffs.values()) <= 1e-12)
        if not cell['reproduction'] or not all(v['reproduced'] for v in cell['reproduction'].values()):
            raise SystemExit(f'q dependence {c}: primary scores not reproduced: {cell["reproduction"]}')
        out['sets'][c] = cell
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_bytes(b'')
        return
    cols = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


def flatten_cell(prefix, cell):
    r = dict(prefix)
    for k in ('rows', 'matches', 'sparse_lt30_matches'):
        r[k] = cell.get(k)
    for k, v in cell.items():
        if isinstance(v, dict) and 'row' in v:
            r[f'{k}_row'] = v['row']
            r[f'{k}_match_weighted'] = v['match_weighted']
    for k in ('mean_abs_delta_diff_B_reg', 'mean_abs_delta_diff_B_econ'):
        if k in cell:
            r[k] = cell[k]
    return r


def main():
    st = K.Status('analyze')
    K.log_command()
    try:
        if not K.frozen_manifest_path().exists():
            raise SystemExit('frozen_manifest.json required')
        P = K.ParentReadOnly()
        results = {}
        for i, name in enumerate(K.LABEL_SETS):
            st.update('running', f'analyze_{name}', processed=i, total=len(K.LABEL_SETS), next_step='next set')
            res, *_ = analyze_set(P, name, st)
            results[name] = res
            C.write_json(K.OUT / 'results' / 'per_set' / f'{name}.json', res)
            st.log(f'{name}: h90 E dis A-B_reg={res["agreement"]["h90"]["E"]["A_vs_B_reg"].get("disagreement_match_weighted")} '
                   f'A-B_econ={res["agreement"]["h90"]["E"]["A_vs_B_econ"].get("disagreement_match_weighted")}')
        st.update('running', 'q_dependence', next_step='tables')
        qd = q_dependence(P)
        C.write_json(K.OUT / 'results' / 'q_label_dependence.json', qd)
        # ------------------------------------------------ tables
        T = K.OUT / 'tables'
        rows = []
        for name, res in results.items():
            for h in K.HS:
                for c in COHORTS:
                    for a, b in PAIRS:
                        cell = res['agreement'][f'h{h}'][c][f'{a}_vs_{b}']
                        r = dict(set=name, horizon=h, cohort=c, pair=f'{a}_vs_{b}', rows=cell['rows'], matches=cell['matches'],
                                 sparse_lt30_matches=cell['sparse_lt30_matches'], disagreement_row=cell.get('disagreement_row'),
                                 disagreement_match_weighted=cell.get('disagreement_match_weighted'),
                                 positive_A_match_weighted=cell.get('positive_rate_A_match_weighted'),
                                 positive_B_match_weighted=cell.get('positive_rate_B_match_weighted'),
                                 exact_zero_A=cell.get('exact_zero_delta_A'), exact_zero_B=cell.get('exact_zero_delta_B'),
                                 delta_absdiff_mean=cell.get('delta_absdiff_mean'), delta_diff_mean_B_minus_A=cell.get('delta_diff_mean_B_minus_A'),
                                 delta_pearson=cell.get('delta_pearson'), delta_spearman=cell.get('delta_spearman'),
                                 unchanged_endpoint_rows=cell.get('endpoint_unchanged_60_90_120', {}).get('rows'),
                                 unchanged_endpoint_disagreement_row=cell.get('endpoint_unchanged_60_90_120', {}).get('disagreement_row'),
                                 changed_endpoint_rows=cell.get('endpoint_changed_across_horizons', {}).get('rows'),
                                 changed_endpoint_disagreement_row=cell.get('endpoint_changed_across_horizons', {}).get('disagreement_row'))
                        if h == 90:
                            bs = res['bootstrap_h90'][c]
                            if not bs.get('empty') and f'{a}_vs_{b}' in bs.get('estimates', {}):
                                e = bs['estimates'][f'{a}_vs_{b}']
                                r.update(ci95_match_weighted_lo=e['match_weighted']['ci95'][0], ci95_match_weighted_hi=e['match_weighted']['ci95'][1],
                                         ci95_row_lo=e['row']['ci95'][0], ci95_row_hi=e['row']['ci95'][1])
                        rows.append(r)
        write_csv(T / 'agreement.csv', rows)
        rows = []
        for name, res in results.items():
            for model, d in res['horizon'].items():
                for c, dd in d.items():
                    for pair, cell in dd.items():
                        rows.append(dict(set=name, model=model, cohort=c, horizons=pair, rows=cell['all']['rows'],
                                         disagreement_row=cell['all'].get('row'), disagreement_match_weighted=cell['all'].get('match_weighted'),
                                         changed_endpoint_rows=cell['changed_endpoint']['rows'],
                                         changed_endpoint_disagreement_row=cell['changed_endpoint'].get('row'),
                                         unchanged_endpoint_rows=cell['unchanged_endpoint']['rows'],
                                         unchanged_endpoint_disagreements=cell['unchanged_endpoint']['disagreements']))
        write_csv(T / 'horizon_sign_disagreement.csv', rows)
        rows = []
        for name, res in results.items():
            for c, fams in res['strata_h90'].items():
                for fname, bins in fams.items():
                    for b, cell in bins.items():
                        rows.append(flatten_cell(dict(set=name, cohort=c, family=fname, bin=b), cell))
        write_csv(T / 'strata_h90.csv', rows)
        rows = []
        for name, res in results.items():
            for h in K.HS:
                for c, o in res['observed'][f'h{h}'].items():
                    for var in ('kills', 'gold', 'xp'):
                        for cat, cell in o[var].items():
                            rows.append(flatten_cell(dict(set=name, horizon=h, cohort=c, variable=var, category=cat), cell))
        write_csv(T / 'observed_directions.csv', rows)
        rows = []
        for name, res in results.items():
            for c, oc in res['objectives_h90']['cohorts'].items():
                for wname, ow in oc.items():
                    for obj, subs in ow.items():
                        if not isinstance(subs, dict):
                            continue
                        for sub, cell in subs.items():
                            rows.append(flatten_cell(dict(set=name, cohort=c, window=wname, objective=obj, subgroup=sub), cell))
        write_csv(T / 'objectives_h90_overlapping_subgroups.csv', rows)
        rows = []
        for c, cell in qd['sets'].items():
            for col, cc in cell['columns'].items():
                for m in K.ALL_V:
                    rows.append(dict(cohort=c, prediction=col, chosen=col in (cell['chosen'], 'spec_' + cell['chosen']), label=f'Y_{m}', **cc[f'Y_{m}']))
        write_csv(T / 'q_label_dependence_h90_main_test.csv', rows)
        C.write_json(K.OUT / 'results' / 'analysis_index.json', dict(role=K.ROLE_TAG, sets=list(results), written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                                   frozen_manifest_sha256=C.sha256_file(K.frozen_manifest_path())))
        st.update('complete', 'analyze', next_step='review packet')
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
