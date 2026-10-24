"""Full-corpus stage T: frozen evaluation of V vs W and q vs generated Y (TEST 15.16 and every external set).

Requires frozen_manifest.json and TEST/external labels generated after it. Verifies every frozen model hash before
and after. No refitting, recalibration or reselection. Match bootstrap (1000, fixed models) for primary comparisons.
Also: counts/exclusions for every role, V diagnostics on validation roles and held-out TRAIN, label distributions,
feature availability (structural zeros, unseen champion IDs) for external sets, and non-comparable pilot context.
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
import csv  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

BOOT_REPS = 1000
BOOT_SEED = 20260915
EXT = ('KR_16.13', 'KR_16.14_pilot', 'KR_16.15', 'NA1_16.13')
HS = C.HORIZONS_S


# ------------------------------------------------------------------ hashing of frozen artifacts
def frozen_files(L, fz):
    files = {fz['v_final_path']: fz['v_final_sha256']}
    vman = C.read_json(L.base / 'v_models_manifest.json')
    for k, sha in fz['v_candidates_sha256'].items():
        files[str((L.base / 'models' / 'v' / f'v_final_{k}.joblib').relative_to(C.OUT))] = sha
    for k, sha in fz['v_oof_sha256'].items():
        files[vman['oof_paths'][k]] = sha
    for h, d in fz['q_bundle_sha256'].items():
        for c, sha in d.items():
            files[str((L.base / 'models' / 'q' / h / f'{c}.joblib').relative_to(C.OUT))] = sha
    for h, d in fz['q_selections'].items():
        files[str((L.base / 'selection' / f'q_{h}.json').relative_to(C.OUT))] = d['sha256']
    files[str((L.base / 'selection_v.json').relative_to(C.OUT))] = fz['selection_v_sha256']
    return files


def verify_hashes(files):
    return {p: dict(expected=sha, actual=C.sha256_file(C.OUT / p), equal=C.sha256_file(C.OUT / p) == sha) for p, sha in files.items()}


# ------------------------------------------------------------------ bootstrap (paired, match unit, fixed models)
def bootstrap(y, preds, g, pairs, reps=BOOT_REPS, seed=BOOT_SEED):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(float)
    u, inv = np.unique(g, return_inverse=True)
    nm = len(u)
    w = C.weights(g)
    eps = np.finfo(np.float64).eps
    models = sorted({m for pr in pairs for m in pr})
    Wm = np.bincount(inv, weights=w, minlength=nm)
    S = {}
    for m in models:
        p = np.asarray(preds[m], dtype=float)
        q = np.clip(p, eps, 1 - eps)
        S[m] = dict(brier=np.bincount(inv, weights=w * (p - y) ** 2, minlength=nm),
                    logloss=np.bincount(inv, weights=w * -(y * np.log(q) + (1 - y) * np.log(1 - q)), minlength=nm))
    point = {m: C.score(y.astype(int), preds[m], w) for m in models}
    rng = np.random.default_rng(seed)
    draws = {m: {k: np.full(reps, np.nan) for k in ('auc', 'brier', 'logloss')} for m in models}
    single = 0
    for r in range(reps):
        mult = np.bincount(rng.integers(0, nm, size=nm), minlength=nm).astype(float)
        rw = w * mult[inv]
        keep = rw > 0
        two = len(np.unique(y[keep])) == 2
        single += int(not two)
        for m in models:
            draws[m]['brier'][r] = float(mult @ S[m]['brier'] / (mult @ Wm))
            draws[m]['logloss'][r] = float(mult @ S[m]['logloss'] / (mult @ Wm))
            if two:
                draws[m]['auc'][r] = float(roc_auc_score(y[keep], np.asarray(preds[m])[keep], sample_weight=rw[keep]))
    out = dict(replicates=reps, seed=seed, unit='match resampled with replacement; weights = equal-per-match x multiplicity',
               fixed_models=True, matches=int(nm), rows=int(len(y)), single_class_replicates=single,
               scope='test-sample uncertainty only; training, calibration and selection variability not included',
               model_ci={m: {k: dict(estimate=point[m][k], ci95=(np.nanquantile(draws[m][k], [.025, .975]).tolist()
                                                                   if np.isfinite(draws[m][k]).any() else None))
                             for k in ('auc', 'brier', 'logloss')} for m in models}, pairs=[])
    for a, b in pairs:
        d = {}
        for k in ('auc', 'brier', 'logloss'):
            diff = draws[a][k] - draws[b][k]
            ok = np.isfinite(diff)
            pa, pb = point[a][k], point[b][k]
            d[k] = dict(estimate=None if pa is None or pb is None else pa - pb,
                        ci95=np.quantile(diff[ok], [.025, .975]).tolist() if ok.any() else None,
                        fraction_replicates_a_better=float(np.mean(diff[ok] > 0) if k == 'auc' else np.mean(diff[ok] < 0)) if ok.any() else None)
        out['pairs'].append(dict(a=a, b=b, a_minus_b=d))
    return out


# ------------------------------------------------------------------ V
def v_stream(L, set_id, sub_roles, adapters, names, include_grid):
    """Predict every stored V row chunk by chunk; keep X only for bucket rows' strata masks."""
    acc = {k: [] for k in ('match', 'query_ms', 'snapshot_ms', 'is_bucket', 'unassigned')}
    P = {k: [] for k in adapters}
    strata = {}
    zero_frac = np.zeros(len(names))
    n_bucket = 0
    champ_vals = []
    for sp, _op, cm in D.chunk_paths(L, set_id):
        with np.load(sp, allow_pickle=False) as z:
            mm = z['m_match']
            sr = z['m_sub_role']
            role_of = dict(zip(mm.tolist(), sr.tolist()))
            vm = z['v_match']
            if not len(vm):
                continue
            keep = np.isin(np.asarray([role_of[m] for m in vm.tolist()]), list(sub_roles))
            if not include_grid:
                keep &= z['v_is_bucket_sample'] == 1
            if not keep.any():
                continue
            X = z['v_X'][keep]
            isb = z['v_is_bucket_sample'][keep] == 1
            for k in acc:
                acc[k].append({'match': vm, 'query_ms': z['v_query_ms'], 'snapshot_ms': z['v_snapshot_ms'],
                               'is_bucket': z['v_is_bucket_sample'], 'unassigned': z['v_unassigned']}[k][keep])
            for k, ad in adapters.items():
                P[k].append(ad.predict_matrix(X, names, C.STATE_VERSION))
            s = C.objective_strata(X[isb], names, z['v_unassigned'][keep][isb])
            for k, v in s.items():
                strata.setdefault(k, []).append(v)
            zero_frac += (X[isb] == 0).sum(axis=0)
            n_bucket += int(isb.sum())
            champ_vals.append(X[isb][:, [names.index(f'participant_slot{i}_champion_id') for i in range(10)]].astype(np.int64))
    R = {k: np.concatenate(v) for k, v in acc.items()}
    R['p'] = {k: np.concatenate(v) for k, v in P.items()}
    R['strata_bucket'] = {k: np.concatenate(v) for k, v in strata.items()}
    R['zero_fraction_bucket'] = (zero_frac / max(1, n_bucket)).tolist()
    R['champions_bucket'] = np.concatenate(champ_vals) if champ_vals else np.zeros((0, 10), dtype=np.int64)
    return R


def v_cells(y, p, g, t_ms, strata=None):
    out = dict(overall=C.evaluate(y, p, g))
    minutes = t_ms / 60000.
    out['time_bands'] = {}
    for lo, hi in C.TIME_BANDS_V:
        m = (minutes >= lo) & (minutes < hi)
        out['time_bands'][f'{lo}-{hi}' if hi < 1000 else f'{lo}+'] = C.evaluate(y[m], p[m], g[m], bins=False) if m.any() else dict(n=0)
    if strata is not None:
        out['objective_strata'] = {k: (C.evaluate(y[m], p[m], g[m], bins=False) if m.any() else dict(n=0, matches=0)) for k, m in strata.items()}
    return out


def evaluate_v(L, st, fz, names, prior):
    vdir = L.base / 'models' / 'v'
    adapters = {k: C.load_v_adapter(vdir / f'v_final_{k}.joblib', sha) for k, sha in fz['v_candidates_sha256'].items()}
    chosen = fz['v_chosen']
    other = [k for k in adapters if k != chosen][0]
    results = {}
    train_vocab = [set(int(x) for x in cats) for cats in adapters[chosen].base.named_steps['preprocess'].named_transformers_['champions'].categories_]
    TRz = None
    targets = [('MAIN', ['TEST'], 'MAIN_TEST', True, 'primary')] + [(s, ['EXTERNAL'], f'EXT_{s}', True, 'primary_per_set') for s in EXT] + \
              [('MAIN', ['V_SELECT'], 'MAIN_V_SELECT', False, 'diagnostic (used for V calibration choice)'),
               ('MAIN', ['V_CAL'], 'MAIN_V_CAL', False, 'diagnostic (used for V calibration fit)'),
               ('MAIN', ['Q_CAL'], 'MAIN_Q_CAL', False, 'diagnostic (V untouched by these outcomes)'),
               ('MAIN', ['Q_SELECT'], 'MAIN_Q_SELECT', False, 'diagnostic (V untouched by these outcomes)')]
    for set_id, roles, name, grid, kind in targets:
        if not (L.set_dir(set_id) / 'extraction_manifest.json').exists():
            results[name] = dict(blocker=f'extraction for {set_id} unavailable')
            continue
        st.update('running', f'v_eval_{name}', next_step='next V cell')
        R = v_stream(L, set_id, roles, adapters, names, include_grid=grid)
        W = D.load_outcomes(L, set_id, roles, purpose=f'frozen V evaluation {name}')
        y = np.asarray([W[m][0] for m in R['match'].tolist()])
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f'{name}: invalid outcomes on V rows')
        b = R['is_bucket'] == 1
        g = R['match']
        cell = dict(kind=kind, rows_bucket=int(b.sum()), matches=int(len(np.unique(g[b]))), rows_grid=int(len(y)) if grid else None)
        preds_b = {chosen: R['p'][chosen][b], other: R['p'][other][b], 'train_prior_constant': np.full(int(b.sum()), prior)}
        cell['bucket'] = {k: v_cells(y[b], p, g[b], R['query_ms'][b], R['strata_bucket'] if k == chosen else None) for k, p in preds_b.items()}
        if grid:
            cell['full_trajectory_secondary'] = {k: v_cells(y, R['p'][k], g, R['query_ms']) for k in (chosen,)}
            mins = (R['query_ms'] // 60000).astype(int)
            curve = []
            for mi in range(2, int(mins.max()) + 1):
                m = mins == mi
                if m.sum() == 0:
                    continue
                e = C.evaluate(y[m], R['p'][chosen][m], g[m], bins=False)
                curve.append(dict(minute=mi, n=e['n'], matches=e['matches'], auc=e.get('auc'), brier=e.get('brier'),
                                  logloss=e.get('logloss'), sparse=e.get('sparse_lt30_matches')))
            cell['full_trajectory_minute_curve'] = curve
        if kind.startswith('primary'):
            cell['bootstrap_bucket'] = bootstrap(y[b], preds_b, g[b], [(chosen, 'train_prior_constant'), (chosen, other)])
        champs = R['champions_bucket']
        cell['feature_availability'] = dict(
            zero_fraction_bucket_rows=dict(zip(names, R['zero_fraction_bucket'])),
            unseen_champion_id_fraction_by_slot=[float(np.mean([c not in train_vocab[i] for c in champs[:, i]])) if len(champs) else None for i in range(10)],
            rows_with_any_unseen_champion=float(np.mean([any(c not in train_vocab[i] for i, c in enumerate(r)) for r in champs])) if len(champs) else None)
        C.save_npz(L.base / 'eval' / 'predictions' / f'v_{name}.npz', match=R['match'], query_ms=R['query_ms'], snapshot_ms=R['snapshot_ms'],
                   is_bucket_sample=R['is_bucket'], winner_blue=y, **{f'p_{k}': v for k, v in R['p'].items()},
                   chosen=np.asarray(chosen), v_final_sha256=np.asarray(fz['v_final_sha256']))
        results[name] = cell
        st.log(f'V {name}: bucket AUC={cell["bucket"][chosen]["overall"]["auc"]} logloss={cell["bucket"][chosen]["overall"]["logloss"]}')
    # held-out TRAIN (OOF) diagnostics
    with np.load(L.base / 'predictions' / 'v_train_oof_and_select.npz', allow_pickle=False) as z:
        TRz = {k: z[k] for k in ('train_match', 'train_query_ms', 'train_fold', 'train_oof_p')}
    W = D.load_outcomes(L, 'MAIN', [f'fold{k}' for k in range(C.N_FOLDS)], purpose='held-out TRAIN V diagnostics')
    y = np.asarray([W[m][0] for m in TRz['train_match'].tolist()])
    results['MAIN_TRAIN_heldout_fold'] = dict(kind='diagnostic: each TRAIN query predicted by the fold adapter that excluded its match',
                                              overall=v_cells(y, TRz['train_oof_p'], TRz['train_match'], TRz['train_query_ms']),
                                              by_fold={f'fold{k}': C.evaluate(y[TRz['train_fold'] == k], TRz['train_oof_p'][TRz['train_fold'] == k],
                                                                              TRz['train_match'][TRz['train_fold'] == k], bins=False) for k in range(C.N_FOLDS)})
    return results


# ------------------------------------------------------------------ q
def q_strata(names, X, s_ms, lab, count_keys):
    out = []
    minutes = s_ms / 60000.
    for lo, hi in C.TIME_BANDS_Q:
        out.append((f'start_minutes_{lo}_{hi}', (minutes >= lo) & (minutes < hi)))

    def col(n):
        return X[:, names.index(n)]
    for obj in ('baron', 'elder', 'herald', 'horde', 'atakhan', 'dragon', 'soul', 'dragon_AIR', 'dragon_EARTH', 'dragon_FIRE',
                'dragon_WATER', 'dragon_HEXTECH', 'dragon_CHEMTECH', 'dragon_OTHER'):
        if obj == 'dragon':
            v = col('blue_dragons') + col('red_dragons')
        elif obj == 'soul':
            v = col('blue_soul_event_recorded') + col('red_soul_event_recorded')
        else:
            v = col('blue_' + obj) + col('red_' + obj)
        out.append((f'pre_history_{obj}_acquired', v > 0))
        out.append((f'pre_history_{obj}_not_acquired', v <= 0))
    return out


def posthoc_strata(lab, h, count_keys, mask):
    out = []
    objs = ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan', 'soul_owned', 'soul_teamid0_unassigned', 'tower', 'inhibitor', 'plate')
    for window, arr in (('during', lab['during_counts']), ('after_last_kill', lab[f'after_counts_h{h}'])):
        for obj in objs:
            v = arr[mask][:, count_keys.index(obj)]
            out.append((f'posthoc_{window}_{obj}_present', v > 0))
    return out


def evaluate_q(L, st, fz, prior_note):
    res = {}
    sets = [('MAIN_TEST', 'primary')] + [(f'EXT_{s}', 'primary_per_set') for s in EXT]
    bundles = {}
    for h in HS:
        bundles[h] = {c: joblib.load(L.base / 'models' / 'q' / f'h{h}' / f'{c}.joblib') for c in C.Q_CANDIDATES}
    # serialization: reload predictions equal saved TRAIN/VAL predictions
    ser = {}
    for h in HS:
        with np.load(L.base / 'predictions' / f'q_h{h}_trainval.npz', allow_pickle=False) as z:
            saved = {c: z[c] for c in C.Q_CANDIDATES}
            valid = z['valid']
        with np.load(L.base / 'labels' / 'MAIN_TRAIN_features_pre_only.npz', allow_pickle=False) as a, \
                np.load(L.base / 'labels' / 'MAIN_VALIDATION_features_pre_only.npz', allow_pickle=False) as b:
            X = np.vstack([a['X_input'], b['X_input']])
        ser[f'h{h}'] = {c: bool(np.array_equal(C.predict_q_bundle(bundles[h][c], np.nan_to_num(X[valid])), saved[c][valid])) for c in C.Q_CANDIDATES}
        X = None
    res['serialization_identity_trainval'] = ser
    for name, kind in sets:
        fpath = L.base / 'labels' / f'{name}_features_pre_only.npz'
        if not fpath.exists():
            res[name] = dict(blocker='labels/features not available')
            continue
        st.update('running', f'q_eval_{name}', next_step='next q set')
        with np.load(fpath, allow_pickle=False) as z:
            F = {k: z[k] for k in z.files}
        with np.load(L.base / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
            lab = {k: z[k] for k in z.files}
        names = [str(x) for x in F['input_names']]
        count_keys = [str(x) for x in lab['count_keys']]
        g_all = F['match'].astype(str)
        cellset = dict(kind=kind)
        for h in HS:
            sel = C.read_json(L.base / 'selection' / f'q_h{h}.json')
            chosen = sel['chosen']
            v = lab[f'valid_h{h}'] == 1
            X = F['X_input'][v]
            y = lab[f'Y_h{h}'][v].astype(int)
            g = g_all[v]
            P = {c: C.predict_q_bundle(bundles[h][c], X) for c in C.Q_CANDIDATES}
            cell = dict(rows=int(v.sum()), matches=int(len(np.unique(g))), chosen=chosen,
                        positive_rate_match_weighted=float(np.average(y, weights=C.weights(g))) if len(y) else None,
                        exact_zero_delta=int(np.sum(lab[f'delta_h{h}'][v] == 0)),
                        metrics={c: C.evaluate(y, P[c], g) for c in C.Q_CANDIDATES},
                        probability_ranges={c: dict(min=float(P[c].min()), max=float(P[c].max()),
                                                    exact_0_or_1=int(np.sum((P[c] == 0) | (P[c] == 1))),
                                                    exact_0_or_1_with_opposite_label=int(np.sum(((P[c] == 1) & (y == 0)) | ((P[c] == 0) & (y == 1)))))
                                            for c in C.Q_CANDIDATES})
            strata = q_strata(names, X, lab['s'][v], lab, count_keys) + posthoc_strata(lab, h, count_keys, v)
            cell['strata'] = [dict(stratum=nm, rows=int(m.sum()), matches=int(len(np.unique(g[m]))),
                                   chosen=(C.evaluate(y[m], P[chosen][m], g[m], bins=False) if m.any() else None),
                                   constant=(C.evaluate(y[m], P['constant'][m], g[m], bins=False) if m.any() else None))
                              for nm, m in strata]
            if h == C.PRIMARY_H:
                pairs = [(chosen, b) for b in dict.fromkeys(['constant', 'p_pre_logistic', 'p_pre_spline',
                                                             sel['selected_economic_variant'], sel['selected_ridge_variant']]) if b != chosen]
            else:
                pairs = [(chosen, 'constant')] if chosen != 'constant' else []
            cell['bootstrap'] = bootstrap(y, P, g, pairs) if pairs and len(np.unique(g)) >= 20 else dict(skipped='no pairs or < 20 matches')
            cell['label_distribution'] = dict(
                delta_quantiles=np.quantile(lab[f'delta_h{h}'][v], [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist() if v.any() else None,
                end_reasons=dict(Counter(lab[f'reasons_h{h}'][v].tolist())),
                post_snapshot_after_L_rate=float(np.mean(lab[f'post_snapshot_after_L_h{h}'][v] == 1)) if v.any() else None,
                duration_s_quantiles=np.quantile((lab[f'endpoint_h{h}'][v] - lab['L'][v]) / 1000., [0, .25, .5, .75, 1]).tolist() if v.any() else None)
            C.save_npz(L.base / 'eval' / 'predictions' / f'q_{name}_h{h}.npz', match=g, s_ms=lab['s'][v], y=y,
                       **{c: P[c] for c in C.Q_CANDIDATES}, chosen=np.asarray(chosen),
                       bundle_sha256=np.asarray(fz['q_bundle_sha256'][f'h{h}'][chosen]))
            cellset[f'h{h}'] = cell
            st.log(f'q {name} h{h}: chosen={chosen} auc={cell["metrics"][chosen]["auc"]} brier={cell["metrics"][chosen]["brier"]} '
                   f'logloss={cell["metrics"][chosen]["logloss"]}')
        # cross-horizon diagnostic: frozen h90 q against Y60/Y120 on rows valid at both
        ch90 = C.read_json(L.base / 'selection' / 'q_h90.json')['chosen']
        cross = {}
        for h in (60, 120):
            m = (lab['valid_h90'] == 1) & (lab[f'valid_h{h}'] == 1)
            if m.any():
                p90 = C.predict_q_bundle(bundles[90][ch90], F['X_input'][m])
                cross[f'h90_q_vs_Y_h{h}'] = C.evaluate(lab[f'Y_h{h}'][m].astype(int), p90, g_all[m], bins=False)
                cross[f'label_disagreement_h90_vs_h{h}'] = float(np.mean(lab['Y_h90'][m] != lab[f'Y_h{h}'][m]))
        cellset['cross_horizon_diagnostic'] = cross
        res[name] = cellset
    # q diagnostics on V_CAL/V_SELECT (after selection) are in q_fit_metrics/h*.json
    return res


# ------------------------------------------------------------------ counts
def counts_all(L):
    out = {}
    targets = [('MAIN', None)] + [(s, None) for s in EXT]
    for set_id, _ in targets:
        if not (L.set_dir(set_id) / 'extraction_manifest.json').exists():
            out[set_id] = dict(blocker='not extracted')
            continue
        T = D.match_table(L, set_id)
        E = D.load_engagements(L, set_id, None, states=False, counts=False)
        roles = sorted(set(T['sub_role'].tolist()))
        per = {}
        for r in roles:
            mm = T['sub_role'] == r
            em = E['sub_role'] == r
            cell = dict(raw_matches=int(mm.sum()), loaded=int(T['loaded'][mm].sum()), v_eligible_matches=int(T['v_eligible'][mm].sum()),
                        v_bucket_queries=int(T['n_bucket'][mm & (T['v_eligible'] == 1)].sum()),
                        v_grid_queries=int(T['n_grid'][mm & (T['v_eligible'] == 1)].sum()),
                        v_exclusions=dict(Counter(T['v_reason'][mm & (T['v_eligible'] == 0)].tolist())),
                        load_failures=dict(Counter(T['load_reason'][mm & (T['loaded'] == 0)].tolist())),
                        matches_with_exposures=int((T['n_exposures'][mm] > 0).sum()), exposure_rows=int(em.sum()),
                        unmapped_elite_monster_events=int(T['unmapped_elite'][mm].sum()),
                        participant_order_differs_from_role_slots=int((T['order_differs'][mm] == 1).sum()))
            for h in HS:
                vh = em & (E[f'valid_h{h}'] == 1)
                cell[f'h{h}'] = dict(valid_rows=int(vh.sum()), valid_matches=int(len(np.unique(E['match'][vh]))),
                                     excluded_rows=int((em & (E[f'valid_h{h}'] == 0)).sum()),
                                     exclusion_reasons=dict(Counter(E[f'invalid_h{h}'][em & (E[f'valid_h{h}'] == 0)].tolist())))
            per[r] = cell
        excl_rows = []
        for i in np.flatnonzero((E['valid_h90'] == 0) | (E['valid_h60'] == 0) | (E['valid_h120'] == 0)):
            excl_rows.append([set_id, E['sub_role'][i], E['match'][i], int(E['s'][i]), int(E['L'][i]),
                              E['invalid_h60'][i], E['invalid_h90'][i], E['invalid_h120'][i]])
        vex = [[set_id, T['sub_role'][i], T['match'][i], T['v_reason'][i], T['load_reason'][i]] for i in np.flatnonzero(T['v_eligible'] == 0)]
        (L.base / 'eval').mkdir(parents=True, exist_ok=True)
        with open(L.base / 'eval' / f'exclusions_engagement_{set_id}.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['set', 'role', 'match', 's_ms', 'L_ms', 'invalid_h60', 'invalid_h90', 'invalid_h120'])
            w.writerows(excl_rows)
        with open(L.base / 'eval' / f'exclusions_v_{set_id}.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['set', 'role', 'match', 'v_reason', 'load_reason'])
            w.writerows(vex)
        out[set_id] = per
    for s in EXT + ('EUW1_complete',):
        pm = C.OUT / 'external' / s / 'prepared_manifest.json'
        if pm.exists():
            d = C.read_json(pm)
            out.setdefault('external_preparation', {})[s] = dict(expected=d.get('expected'), written=d.get('written'), excluded=d.get('excluded'),
                                                                 exclusion_reasons=d.get('exclusion_reasons'), blocker=d.get('blocker'),
                                                                 exposure_rows=d.get('exposure_rows'), prior_use=d.get('prior_use'),
                                                                 availability={k: v for k, v in (d.get('availability') or {}).items() if k != 'raw_counts'})
    return out


def main():
    import warnings as _w
    _w.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--part', default='counts,v,q')
    args = ap.parse_args()
    L = D.Layout(args.smoke)
    st = C.Status(L.base, 'evaluate')
    try:
        fz = C.read_json(L.frozen_manifest)
        lt = L.base / 'labels' / 'labels_test_external_manifest.json'
        if not lt.exists() and not args.smoke:
            raise SystemExit('TEST/external labels missing')
        files = frozen_files(L, fz)
        before = verify_hashes(files)
        if not all(v['equal'] for v in before.values()):
            raise SystemExit('frozen artifact hash changed before evaluation')
        C.write_json(L.base / 'eval' / 'hashes_before.json', before)
        names = L.manifest('MAIN')['names']
        prior = C.read_json(L.base / 'selection_v.json')['final_fit']['positive_rate_match_weighted']
        parts = args.part.split(',')
        if 'counts' in parts:
            st.update('running', 'counts', next_step='V evaluation')
            C.write_json(L.base / 'eval' / 'counts_and_exclusions.json', dict(role=C.ROLE_TAG, counts=counts_all(L)))
        if 'v' in parts:
            rv = evaluate_v(L, st, fz, names, prior)
            C.write_json(L.base / 'eval' / 'results_v.json', dict(role=C.ROLE_TAG, v_chosen=fz['v_chosen'], v_final_sha256=fz['v_final_sha256'],
                                                                  train_prior_constant=prior, results=rv,
                                                                  semantics='P(final Blue win | observed state); associational'))
        if 'q' in parts:
            rq = evaluate_q(L, st, fz, None)
            C.write_json(L.base / 'eval' / 'results_q.json', dict(role=C.ROLE_TAG, selections=fz['q_selections'], results=rq,
                                                                  semantics='q = P(generated Y = 1 | pre-state); labels are model-defined, not truth'))
        after = verify_hashes(files)
        C.write_json(L.base / 'eval' / 'hashes_after.json', after)
        unchanged = all(v['equal'] for v in after.values())
        st.update('complete' if unchanged else 'failed', 'evaluate', hashes_unchanged=unchanged, next_step='SHAP (h90 chosen q)')
        return 0 if unchanged else 1
    except SystemExit as exc:
        st.update('failed', 'evaluate', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'evaluate', error=repr(exc), next_step='inspect, fix, rerun (models frozen; evaluation is read-only)')
        return 3


if __name__ == '__main__':
    sys.exit(main())
