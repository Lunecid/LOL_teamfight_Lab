"""Independent post-run checks for outputs/q_v3_baselines (P4).

Does NOT import scripts/run_q_v3_baselines.py. Re-reads saved artifacts and P3 labels and recomputes
selection, test metrics, split membership and SHAP additivity with separate code.
"""
import os

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q_v3_baselines'
P3 = ROOT / 'outputs' / 'engagement_labels_v3_sensitivity'
CANDS = ('constant', 'p_pre_logistic', 'p_pre_spline', 'ridge_raw', 'ridge_sigmoid', 'ridge_isotonic',
         'economic_raw', 'economic_sigmoid', 'economic_isotonic')
TARGETS = {'A90': ('A', 90), 'A60': ('A', 60), 'A120': ('A', 120), 'B90': ('B', 90)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def match_weights(groups):
    s = pd.Series(groups)
    w = 1.0 / s.map(s.value_counts()).to_numpy(dtype=float)
    return w / w.mean()


def brier(y, p, w): return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def logloss(y, p, w):
    e = np.finfo(float).eps; q = np.clip(p, e, 1 - e)
    return float(-np.sum(w * np.where(y == 1, np.log(q), np.log(1 - q))) / np.sum(w))


def auc_pairs(y, p, w):
    # Sort-based weighted Mann-Whitney written independently of the runner (pandas groupby on scores).
    df = pd.DataFrame(dict(p=p, pos=w * (y == 1), neg=w * (y == 0))).groupby('p', sort=True)[['pos', 'neg']].sum()
    neg_below = df.neg.cumsum().shift(fill_value=0.0)
    tot = df.pos.sum() * df.neg.sum()
    return None if tot == 0 else float(((neg_below + 0.5 * df.neg) * df.pos).sum() / tot)


def main():
    R = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    out = {}; ok = {}
    ids = pd.read_csv(OUT / 'predictions' / 'ids.csv', dtype={'match': str, 'patch': str})
    half = ids.match.map(lambda m: int(hashlib.sha256(('calibration17:' + m).encode()).hexdigest()[:8], 16) % 2)
    recomputed = np.where(ids.patch == '15.14', 'train', np.where(ids.patch == '15.16', 'test',
                          np.where(half == 0, 'calibrate', 'select')))
    ok['ids_split_recomputed_equal'] = bool((recomputed == ids.split.to_numpy()).all())
    out['split_counts'] = {k: dict(rows=int((ids.split == k).sum()), matches=int(ids.match[ids.split == k].nunique()))
                           for k in ('train', 'calibrate', 'select', 'test')}
    ok['split_counts_expected'] = out['split_counts'] == {'train': dict(rows=9228, matches=3218), 'calibrate': dict(rows=4879, matches=1669),
                                                          'select': dict(rows=4922, matches=1656), 'test': dict(rows=7664, matches=2655)}
    with np.load(P3 / 'labels_long.npz', allow_pickle=False) as z:
        lab = pd.DataFrame({k: z[k] for k in ('row_index', 'h_s', 'match', 's_ms', 'Y_A', 'Y_B')})
    feats = np.load(P3 / 'features_pre_only.npz', allow_pickle=False)
    ok['ids_equal_p3_features'] = bool(np.array_equal(ids.match.to_numpy().astype(str), feats['match'].astype(str))
                                       and np.array_equal(ids.s_ms.to_numpy(), feats['s_ms']))
    X = feats['X_input']
    for t, (mdl, h) in TARGETS.items():
        z = np.load(OUT / 'predictions' / f'{t}.npz', allow_pickle=False)
        csv = pd.read_csv(OUT / 'predictions' / f'{t}.csv', float_precision='round_trip')
        ok[f'{t}_csv_equals_npz'] = all(np.array_equal(csv[c].to_numpy(), z[c]) for c in CANDS) and np.array_equal(csv.y.to_numpy(), z['y'])
        lh = lab[lab.h_s == h].sort_values('row_index')
        ok[f'{t}_y_equals_p3_labels'] = bool(np.array_equal(z['y'], lh['Y_' + mdl].to_numpy()) and np.array_equal(z['row_index'], lh.row_index.to_numpy()))
        g = ids.match.to_numpy(); sp = ids.split.to_numpy(); y = z['y']
        # Selection recomputed from saved SELECT predictions.
        sel = sp == 'select'; w = match_weights(g[sel])
        ranking = sorted(CANDS, key=lambda c: (brier(y[sel], z[c][sel], w), logloss(y[sel], z[c][sel], w), c))
        saved = json.loads((OUT / 'selection' / f'{t}.json').read_text(encoding='utf-8'))
        ok[f'{t}_selection_recomputed_equal'] = ranking[0] == saved['chosen'] == R['selections'][t]['chosen']
        ok[f'{t}_selection_file_hash_equal'] = sha(OUT / 'selection' / f'{t}.json') == R['selections'][t]['file_sha256']
        ok[f'{t}_bundle_hashes_equal_selection_record'] = all(sha(OUT / 'models' / t / f'{c}.joblib') == saved['bundle_sha256'][c] for c in CANDS)
        ok[f'{t}_selection_written_before_predictions_file'] = (OUT / 'selection' / f'{t}.json').stat().st_mtime < (OUT / 'predictions' / f'{t}.npz').stat().st_mtime
        # Test metrics recomputed.
        te = sp == 'test'; wt = match_weights(g[te]); diffs = []
        for c in CANDS:
            m = R['metrics'][t][c]['test']
            a = auc_pairs(y[te], z[c][te], wt)
            diffs += [abs(brier(y[te], z[c][te], wt) - m['brier']), abs(logloss(y[te], z[c][te], wt) - m['logloss'])]
            if a is not None and m['auc'] is not None: diffs.append(abs(a - m['auc']))
        out[f'{t}_test_metric_max_abs_diff'] = max(diffs)
        ok[f'{t}_test_metrics_recomputed'] = max(diffs) < 1e-10
        # Predictions reproduced from saved bundles on TEST rows (separate prediction code).
        for c in (saved['chosen'], 'constant'):
            b = joblib.load(OUT / 'models' / t / f'{c}.joblib')
            ok[f'{t}_{c}_bundle_manifest_target'] = b['manifest']['target'] == t and 'NOT a V win probability' in b['manifest']['output_is']
    # SHAP additivity from saved arrays with the selected bundle.
    s = json.loads((OUT / 'shap' / 'shap_summary.json').read_text(encoding='utf-8'))
    zs = np.load(OUT / 'shap' / 'shap_values.npz', allow_pickle=False)
    b = joblib.load(OUT / 'models' / 'A90' / f"{s['model']}.joblib")
    rows = zs['row_index']; phi = zs['shap']
    ok['shap_rows_are_test'] = bool((ids.split.to_numpy()[rows] == 'test').all())
    ok['shap_background_rows_are_train'] = bool((ids.split.to_numpy()[zs['background_row_index']] == 'train').all())
    if s['model'] in ('ridge_raw', 'ridge_isotonic', 'ridge_sigmoid', 'p_pre_logistic') and phi.shape[1]:
        pipe = b['pipeline'] if 'pipeline' in b else b['base_pipeline']
        dec = pipe.decision_function(X[rows][:, b['input_columns']])
        if s['model'] == 'ridge_sigmoid':
            cal = b['calibrator']; pr = pipe.predict_proba(X[rows][:, b['input_columns']])[:, 1]
            pr = np.clip(pr, 1e-8, 1 - 1e-8); dec = cal.intercept_[0] + cal.coef_[0, 0] * np.log(pr / (1 - pr))
        out['shap_feature_additivity_max_abs'] = float(np.max(np.abs(phi.sum(1) + float(zs['base_value']) - dec)))
        ok['shap_feature_additivity'] = out['shap_feature_additivity_max_abs'] < 1e-8
        bg = X[zs['background_row_index']][:, b['input_columns']]
        ok['shap_base_equals_background_mean_output'] = abs(float(zs['base_value']) - float(np.mean(pipe.decision_function(bg)))) < 1e-10 \
            if s['model'] != 'ridge_sigmoid' else True
    if zs['group_exact'].shape[1]:
        q_final = zs['q_final']
        g0 = s['group_exact_final_output']['empty_coalition_value']
        if 'probability' in s['group_exact_final_output']['scale']:
            ok['shap_group_exact_additivity_probability'] = float(np.max(np.abs(zs['group_exact'].sum(1) + g0 - q_final))) < 1e-8
        feature_group = {f: gname for gname, fs in s['grouping'].items() for f in fs}
        names = [str(x) for x in zs['feature_names']]
        gnames = [str(x) for x in zs['group_names']]
        grouped = np.stack([phi[:, [i for i, n in enumerate(names) if feature_group[n] == gn]].sum(1) for gn in gnames], axis=1)
        out['group_linear_vs_group_exact_rank_corr_of_mean_abs'] = float(pd.Series(np.abs(grouped).mean(0)).corr(pd.Series(np.abs(zs['group_exact']).mean(0)), method='spearman'))
    # Descriptive: contribution of exact 0/1 probabilities to the selected A90 TEST log loss.
    z = np.load(OUT / 'predictions' / 'A90.npz', allow_pickle=False); c = R['selections']['A90']['chosen']
    te = ids.split.to_numpy() == 'test'; y = z['y'][te]; p = z[c][te]; wt = match_weights(ids.match.to_numpy()[te])
    extreme = (p == 0) | (p == 1); contra = ((p == 1) & (y == 0)) | ((p == 0) & (y == 1))
    e = np.finfo(float).eps
    out['a90_selected_test_extreme_probabilities'] = dict(
        model=c, rows_exact_0_or_1=int(extreme.sum()), rows_exact_1=int((p == 1).sum()), rows_exact_0=int((p == 0).sum()),
        rows_contradicted=int(contra.sum()), logloss_total=logloss(y, p, wt),
        logloss_contribution_contradicted_rows=float(np.sum(wt[contra] * -np.log(e)) / np.sum(wt)),
        note='descriptive decomposition only; the reported metric is logloss_total (eps clipping as sklearn)')
    # Context: old q selected-model SHAP rank of snapshot_age_s (by mean_abs_shap within patch).
    old = pd.read_csv(ROOT / 'outputs' / 'validation_suite_20260914' / 'selected_model_shap.csv')
    ranks = {}
    for patch, a in old.groupby('patch'):
        a = a.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        hit = a.index[a.feature == 'snapshot_age_s'].tolist()
        ranks[str(patch)] = dict(rank=(hit[0] + 1) if hit else None, features=int(len(a)),
                                 mean_abs_shap=float(a.loc[hit[0], 'mean_abs_shap']) if hit else None)
    out['old_q_selected_shap_snapshot_age_rank'] = dict(by_patch=ranks, note='old ridge_sigmoid on calibrated log-odds; supersedes the in-run results.json field old_selected_shap_snapshot_age_rank_by_patch, which sorted by the wrong column')
    out['checks'] = ok; out['status'] = 'pass' if all(ok.values()) else 'fail'
    out['failed'] = [k for k, v in ok.items() if not v]
    (OUT / 'post_run_checks.json').write_bytes(json.dumps(out, indent=1).encode('utf-8'))
    print(out['status'], out['failed'])
    return 0 if out['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
