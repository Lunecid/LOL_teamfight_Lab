"""Diagnostic: can the MAGNITUDE of delta-V_end be predicted from pre-cutoff information?

The sign classifier on sign(delta-V_end) came out less predictable than market_event on the same
rows (0.6344 vs 0.6649), and weighting by |delta-V| made it worse.  Before deciding what role
delta-V should play, the remaining question is whether its magnitude - the part no gold label can
express - carries any pre-fight signal at all.  Same learner family, same splits, same X.

Reports Spearman / Pearson / R^2 for |delta-V| and signed delta-V, plus AUC for 'is this a
top-quartile swing', and the same by game-time band.  A diagnostic, not a deliverable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import match_weights
from scripts.train_engagement_predictor_v3 import load_split

TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))


def regressor(trees=250):
    return LGBMRegressor(n_estimators=trees, num_leaves=15, learning_rate=.04, min_child_samples=40,
                         reg_lambda=1., colsample_bytree=.9, random_state=7, n_jobs=4, verbosity=-1)


def describe(y_true, y_pred, groups):
    w = match_weights(groups)
    top = y_true >= np.quantile(y_true, .75)
    return dict(n=int(len(y_true)), matches=int(len(set(groups.tolist()))),
                spearman=float(spearmanr(y_true, y_pred).correlation),
                pearson=float(pearsonr(y_true, y_pred)[0]),
                r2=float(r2_score(y_true, y_pred, sample_weight=w)),
                auc_top_quartile=float(roc_auc_score(top, y_pred, sample_weight=w)) if 0 < top.sum() < len(top) else None,
                target_median=float(np.median(y_true)), target_q90=float(np.quantile(y_true, .9)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v3-dir', type=Path, default=ROOT / 'outputs/temporal_winprob_v3_buckets')
    ap.add_argument('--dataset', type=Path, default=ROOT / 'outputs/state_value_main_50k')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/engagement_predictor_v3/magnitude_diagnostic.json')
    ap.add_argument('--trees', type=int, default=250)
    a = ap.parse_args()
    started = time.time()
    tr = load_split(a.dataset, a.v3_dir / 'engagement_changes_predict_train.npz')
    te = load_split(a.dataset, a.v3_dir / 'engagement_changes.npz')
    out = {'question': 'Is |delta-V_end| (or signed delta-V_end) predictable from pre-cutoff X?',
           'learner': {'trees': a.trees, 'num_leaves': 15, 'learning_rate': .04, 'seed': 7}, 'targets': {}}
    for name, f in (('abs_deltaV_pp', lambda t: np.abs(t['delta']) * 100.),
                    ('signed_deltaV_pp', lambda t: t['delta'] * 100.)):
        m = regressor(a.trees).fit(tr['X'], f(tr), sample_weight=match_weights(tr['match']))
        pred = m.predict(te['X'])
        cell = {'test': describe(f(te), pred, te['match']), 'by_game_time': {}}
        for lo, hi in TIME_BANDS:
            mask = (te['minute'] >= lo) & (te['minute'] < hi)
            if mask.sum() > 50:
                cell['by_game_time'][f'{lo}-{hi}'] = describe(f(te)[mask], pred[mask], te['match'][mask])
        # a trivial reference: the training-set median, no information
        cell['no_information_reference'] = describe(f(te), np.full(len(te['y']), np.median(f(tr))) +
                                                    np.random.default_rng(7).normal(0, 1e-9, len(te['y'])), te['match'])
        out['targets'][name] = cell
        print(f"[magnitude] {name}: spearman {cell['test']['spearman']:.4f} r2 {cell['test']['r2']:.4f} "
              f"top-quartile AUC {cell['test']['auc_top_quartile']} ({time.time()-started:.0f}s)", flush=True)
    out['elapsed_seconds'] = round(time.time() - started, 2)
    a.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k: v['test'] for k, v in out['targets'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
