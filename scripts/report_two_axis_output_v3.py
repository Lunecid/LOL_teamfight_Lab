"""Two-axis engagement output under the final contract (decided 2026-09-10, revised same day).

  axis 1  outcome  : market_event (who won the engagement), predicted from X at cutoff
  axis 2  value    : delta-V_end = V3(S_end) - V3(S_pre), the swing that outcome produced (pp)

Nothing here is retrained.  It joins the market_event predictor's test predictions to the v3
delta-V rows and asks the question the two-axis framing raises: does the outcome predictor do
better or worse on the engagements that mattered more?  Reports AUC by |delta-V| quartile, by
headroom, by scale and by game phase, and writes a per-engagement table for the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import match_weights

TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))


def cell(y, p, groups):
    w = match_weights(groups)
    two = len(set(y.tolist())) == 2
    return dict(n=int(len(y)), matches=int(len(set(groups.tolist()))),
                positive_rate=float(np.average(y, weights=w)),
                auc=float(roc_auc_score(y, p, sample_weight=w)) if two else None,
                brier=float(brier_score_loss(y, p, sample_weight=w)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pred-dir', type=Path, default=ROOT / 'outputs/engagement_predictor_v3')
    ap.add_argument('--v3-dir', type=Path, default=ROOT / 'outputs/temporal_winprob_v3_buckets')
    a = ap.parse_args()

    with np.load(a.pred_dir / 'predictions.npz', allow_pickle=False) as z:
        ids, match = z['test_id'].astype(str), z['test_match'].astype(str)
        y_market, p_market = z['test_market'], z['test_p_market']
        delta, scale, minute = z['test_delta'], z['test_scale'], z['test_minute']
    with np.load(a.v3_dir / 'engagement_changes.npz', allow_pickle=False) as z:
        order = {e: i for i, e in enumerate(z['id'].astype(str))}
        value_pre = z['expanded'][[order[e] for e in ids], 0]
    headroom = np.minimum(value_pre, 1. - value_pre)
    abs_pp = np.abs(delta) * 100.
    signed_pp = delta * 100.

    has = y_market >= 0
    summary = {
        'contract': {'outcome_label': 'market_event', 'value_axis': 'delta-V_end (pp)',
                     'boundary': 'end = last cluster kill', 'value_model': 'v3 bucket-sampled',
                     'weights': 'match weights only'},
        'rows': {'total': int(len(ids)), 'with_outcome_label': int(has.sum()),
                 'market_ties_dropped': int((~has).sum())},
        'outcome_predictor_overall': cell(y_market[has], p_market[has], match[has]),
        'by_abs_deltaV_quartile': {}, 'by_headroom_quartile': {}, 'by_scale': {}, 'by_game_time': {},
        'value_axis_descriptives': {},
    }
    edges = np.quantile(abs_pp[has], [0, .25, .5, .75, 1.])
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        m = has & (abs_pp >= lo) & (abs_pp <= hi if q == 3 else abs_pp < hi)
        summary['by_abs_deltaV_quartile'][f'Q{q+1} [{lo:.1f},{hi:.1f}]pp'] = cell(y_market[m], p_market[m], match[m])
    hedges = np.quantile(headroom[has], [0, .25, .5, .75, 1.])
    for q in range(4):
        lo, hi = hedges[q], hedges[q + 1]
        m = has & (headroom >= lo) & (headroom <= hi if q == 3 else headroom < hi)
        summary['by_headroom_quartile'][f'Q{q+1} [{lo:.3f},{hi:.3f}]'] = cell(y_market[m], p_market[m], match[m])
    for name, m in (('pick', (scale >= 0) & (scale <= 1)), ('skirmish', (scale >= 2) & (scale <= 3)),
                    ('teamfight', scale >= 4), ('unknown', scale < 0)):
        mm = has & m
        if mm.any():
            summary['by_scale'][name] = cell(y_market[mm], p_market[mm], match[mm])
    for lo, hi in TIME_BANDS:
        mm = has & (minute >= lo) & (minute < hi)
        if mm.any():
            summary['by_game_time'][f'{lo}-{hi}'] = cell(y_market[mm], p_market[mm], match[mm])
    # the value axis on its own, for the rows the outcome label covers and for the ties it drops
    for name, m in (('labelled', has), ('market_ties', ~has)):
        summary['value_axis_descriptives'][name] = {
            'n': int(m.sum()), 'median_abs_pp': float(np.median(abs_pp[m])),
            'mean_signed_pp': float(signed_pp[m].mean()),
            'abs_pp_quantiles': {str(q): float(np.quantile(abs_pp[m], q)) for q in (.25, .5, .75, .9)},
            'median_headroom': float(np.median(headroom[m]))}
    # agreement between the predicted outcome and the sign of the value axis
    pred_sign = (p_market >= .5).astype(int)
    summary['predicted_outcome_vs_value_sign_agreement'] = float(((signed_pp > 0).astype(int) == pred_sign)[has].mean())
    summary['realized_outcome_vs_value_sign_agreement'] = float(((signed_pp > 0).astype(int) == y_market)[has].mean())

    (a.pred_dir / 'two_axis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with open(a.pred_dir / 'two_axis_test.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['engagement_id', 'match', 'scale', 'minute', 'market_event', 'p_market_event',
                    'deltaV_end_pp', 'abs_deltaV_pp', 'value_pre', 'headroom'])
        for i in range(len(ids)):
            w.writerow([ids[i], match[i], int(scale[i]), f'{minute[i]:.2f}', int(y_market[i]),
                        f'{p_market[i]:.5f}', f'{signed_pp[i]:.3f}', f'{abs_pp[i]:.3f}',
                        f'{value_pre[i]:.5f}', f'{headroom[i]:.5f}'])
    print(json.dumps({k: summary[k] for k in ('outcome_predictor_overall', 'by_abs_deltaV_quartile',
                                              'by_scale', 'by_game_time',
                                              'predicted_outcome_vs_value_sign_agreement')}, indent=2))


if __name__ == '__main__':
    main()
