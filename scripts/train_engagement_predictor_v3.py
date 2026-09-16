"""Engagement predictor on the agreed contract: X_cutoff -> sign(delta-V_end), weighted by |delta-V_end|.

Contract (decided 2026-09-10):
  label     y_e = 1[ V3(S_end) - V3(S_pre) > 0 ]        end = last cluster kill, pre = cutoff - 1ms
  weight    w_e = |delta-V_end| x match weight          big swings count more; matches count equally
  input     X_e = the 7,106 pre-cutoff features already stored per engagement
  compare   y_market_event on the SAME rows, same learner, same protocol
  train     predict_train matches (never seen by the value model)   test  predict_test matches

The learner, folds, trees and match weighting replicate train/state_value_experiment.py so the
number here is comparable to the 50k experiment's 0.6228 - which was measured on labels from a
value model that still carried snapshot_age_s and a one-minute-per-match fit.

Also re-asks the match-outcome question on the clean labels: does adding the predicted
engagement to pre-fight information improve final-win prediction?  (A / B / C stacking with
a match-level paired bootstrap, exactly as before.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import assert_disjoint, learner, match_weights, metrics, paired_bootstrap

TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_split(dataset, changes_path):
    """Join v3 delta-V rows to the stored pre-cutoff X on engagement id."""
    with np.load(changes_path, allow_pickle=False) as z:
        ids = z['id'].astype(str)
        match = z['match'].astype(str)
        expanded = z['expanded']
        query = z['query_ms']
        scale = z['scale']
        objectives = z['objectives']                    # (n, 4 types, 2 teams), pre->end
    delta = expanded[:, 1] - expanded[:, 0]
    ok = np.isfinite(delta)
    order = {eid: i for i, eid in enumerate(ids)}
    X = None                                            # width taken from the first stored row
    market = np.full(len(ids), -1, dtype=np.int8)
    eq3 = np.full(len(ids), -1, dtype=np.int8)
    winner = np.full(len(ids), -1, dtype=np.int8)
    seen = np.zeros(len(ids), dtype=bool)
    for mid in sorted(set(match.tolist())):
        with np.load(dataset / 'matches' / (mid + '.npz'), allow_pickle=False) as f:
            eids = f['engagement_id'].astype(str)
            Xm, ym, eq, w = f['X'], f['y_market_event'], f['y_attention_value_win'], int(f['winner'])
        if X is None:
            X = np.full((len(ids), Xm.shape[1]), np.nan, dtype=np.float32)
        for k, eid in enumerate(eids):
            i = order.get(eid)
            if i is None:
                continue
            X[i], market[i], eq3[i], winner[i], seen[i] = Xm[k], ym[k], eq[k], w, True
    if not seen.all():
        raise ValueError(f'{(~seen).sum()} labelled engagements have no stored X row')
    keep = ok & (delta != 0)
    return dict(id=ids[keep], match=match[keep], X=X[keep], delta=delta[keep],
                y=(delta[keep] > 0).astype(np.int8), value_pre=expanded[keep, 0],
                market=market[keep], eq3=eq3[keep], winner=winner[keep], scale=scale[keep],
                minute=query[keep, 0] / 60000., objectives=objectives[keep],
                dropped=dict(nonfinite=int((~ok).sum()), exact_zero=int((ok & (delta == 0)).sum())))


def weighted_metrics(y, p, groups, extra_w=None):
    """Match-weighted metrics; optionally also with the |delta-V| emphasis used in training."""
    base = metrics(y, p, groups)
    if extra_w is not None:
        w = match_weights(groups) * extra_w
        w = w / w.mean()
        base['auc_delta_weighted'] = float(roc_auc_score(y, p, sample_weight=w)) if len(set(y.tolist())) == 2 else None
        base['brier_delta_weighted'] = float(brier_score_loss(y, p, sample_weight=w))
    return base


def oof_predict(X, y, groups, weights, *, folds, trees):
    pred = np.full(len(y), np.nan)
    for tr, va in GroupKFold(folds).split(X, y, groups):
        assert_disjoint(groups[tr], groups[va])
        m = learner(trees=trees)
        m.fit(X[tr], y[tr], sample_weight=weights[tr])
        pred[va] = m.predict_proba(X[va])[:, 1]
    if not np.isfinite(pred).all():
        raise ValueError('incomplete OOF predictions')
    return pred


def run(a):
    started = time.time()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'status.json', {'stage': 'loading', 'pid': os.getpid()})

    tr = load_split(a.dataset, a.v3_dir / 'engagement_changes_predict_train.npz')
    te = load_split(a.dataset, a.v3_dir / 'engagement_changes.npz')
    assert_disjoint(tr['match'], te['match'])
    protocol = json.loads((a.v3_dir / 'protocol.json').read_text(encoding='utf-8'))
    for role, ids in protocol['splits'].items():
        if role != 'engagement':
            assert_disjoint(ids, tr['match'], te['match'])

    def weights_for(t, use_delta):
        w = match_weights(t['match'])
        if use_delta:
            w = w * np.abs(t['delta'])
        return w / w.mean()

    report = {
        'contract': {'label': 'sign(V3(S_end) - V3(S_pre))', 'boundary': 'end = last cluster kill',
                     'weight': '|delta-V_end| x match weight, mean-normalised',
                     'input': 'X at cutoff, 7,106 features', 'value_model': protocol['version'],
                     'value_model_ids': json.loads((a.v3_dir / 'results.json').read_text(encoding='utf-8'))['model_ids']},
        'rows': {'predict_train': {'n': int(len(tr['y'])), 'matches': int(len(set(tr['match'].tolist()))),
                                   'blue_rate': float(tr['y'].mean()), 'dropped': tr['dropped']},
                 'predict_test': {'n': int(len(te['y'])), 'matches': int(len(set(te['match'].tolist()))),
                                  'blue_rate': float(te['y'].mean()), 'dropped': te['dropped']}},
        'learner': {'folds': a.folds, 'trees': a.trees, 'num_leaves': 15, 'learning_rate': .04, 'seed': 7},
    }

    # ---- primary: sign(delta-V_end), |delta| weighted ---------------------------------
    write_json(out / 'status.json', {'stage': 'oof_primary', 'pid': os.getpid()})
    w_tr = weights_for(tr, True)
    oof = oof_predict(tr['X'], tr['y'], tr['match'], w_tr, folds=a.folds, trees=a.trees)
    primary = learner(trees=a.trees).fit(tr['X'], tr['y'], sample_weight=w_tr)
    p = primary.predict_proba(te['X'])[:, 1]
    joblib.dump(primary, out / 'engagement_predictor_deltaV_weighted.joblib')
    report['deltaV_weighted'] = {
        'oof_train': weighted_metrics(tr['y'], oof, tr['match'], np.abs(tr['delta'])),
        'test': weighted_metrics(te['y'], p, te['match'], np.abs(te['delta']))}
    print(f"[predictor] primary test AUC {report['deltaV_weighted']['test']['auc']:.4f} "
          f"({time.time()-started:.0f}s)", flush=True)

    # ---- secondary: same label, match weights only (isolates the |delta| weighting) ----
    write_json(out / 'status.json', {'stage': 'unweighted_variant', 'pid': os.getpid()})
    plain = learner(trees=a.trees).fit(tr['X'], tr['y'], sample_weight=weights_for(tr, False))
    p_plain = plain.predict_proba(te['X'])[:, 1]
    report['deltaV_unweighted'] = {'test': weighted_metrics(te['y'], p_plain, te['match'], np.abs(te['delta']))}

    # ---- comparison currency: market_event on the SAME rows, same learner -------------
    write_json(out / 'status.json', {'stage': 'market_event_comparison', 'pid': os.getpid()})
    mtr, mte = tr['market'] >= 0, te['market'] >= 0
    gold = learner(trees=a.trees).fit(tr['X'][mtr], tr['market'][mtr],
                                      sample_weight=match_weights(tr['match'][mtr]))
    p_gold = gold.predict_proba(te['X'])[:, 1]
    joblib.dump(gold, out / 'engagement_predictor_market_event.joblib')
    report['market_event'] = {
        'train_rows_labelled': int(mtr.sum()), 'test_rows_labelled': int(mte.sum()),
        'test': metrics(te['market'][mte], p_gold[mte], te['match'][mte])}
    # cross-currency: how well does each predictor rank the OTHER label?
    report['cross_currency'] = {
        'deltaV_predictor_vs_market_label': metrics(te['market'][mte], p[mte], te['match'][mte]),
        'market_predictor_vs_deltaV_label': metrics(te['y'], p_gold, te['match'])}

    # ---- where the two currencies disagree, and what sits there --------------------
    agree = (te['market'][mte] == te['y'][mte])
    dragons = te['objectives'][:, 1, :].sum(axis=1) > 0          # any elemental dragon pre->end
    barons = te['objectives'][:, 0, :].sum(axis=1) > 0
    disagree_idx = np.flatnonzero(mte)[~agree]
    report['label_disagreement'] = {
        'agreement': float(agree.mean()), 'disagreeing_rows': int((~agree).sum()),
        'dragon_share_among_disagreeing': float(dragons[disagree_idx].mean()),
        'dragon_share_among_agreeing': float(dragons[np.flatnonzero(mte)[agree]].mean()),
        'baron_share_among_disagreeing': float(barons[disagree_idx].mean()),
        'baron_share_among_agreeing': float(barons[np.flatnonzero(mte)[agree]].mean()),
        'note': 'market_event prices an elemental dragon at 0 gold; delta-V does not'}

    # ---- baselines and decompositions -------------------------------------------------
    shortcut = learner(trees=a.trees).fit(tr['value_pre'].reshape(-1, 1), tr['y'],
                                          sample_weight=weights_for(tr, False))
    report['value_pre_only_baseline'] = metrics(
        te['y'], shortcut.predict_proba(te['value_pre'].reshape(-1, 1))[:, 1], te['match'])
    report['by_scale'] = {}
    for name, mask in (('pick', (te['scale'] >= 0) & (te['scale'] <= 1)),
                       ('skirmish', (te['scale'] >= 2) & (te['scale'] <= 3)),
                       ('teamfight', te['scale'] >= 4), ('unknown', te['scale'] < 0)):
        if mask.any():
            report['by_scale'][name] = metrics(te['y'][mask], p[mask], te['match'][mask])
    report['by_game_time'] = {}
    for lo, hi in TIME_BANDS:
        mask = (te['minute'] >= lo) & (te['minute'] < hi)
        if mask.any():
            report['by_game_time'][f'{lo}-{hi}'] = metrics(te['y'][mask], p[mask], te['match'][mask])
    report['label_agreement_test'] = {
        'market_event': float(agree.mean()),
        'attention_value_win': float((te['eq3'][te['eq3'] >= 0] == te['y'][te['eq3'] >= 0]).mean())}

    # ---- does predicted engagement help final-win prediction? (clean labels) ----------
    write_json(out / 'status.json', {'stage': 'match_outcome_stacking', 'pid': os.getpid()})
    match_preds = {}
    for name, add_tr, add_te in (('A_pre_information', None, None),
                                 ('B_plus_predicted_engagement', oof, p),
                                 ('C_plus_realized_label_RETROSPECTIVE', tr['y'], te['y'])):
        Xa = tr['X'] if add_tr is None else np.column_stack([tr['X'], add_tr])
        Xb = te['X'] if add_te is None else np.column_stack([te['X'], add_te])
        m = learner(trees=a.trees).fit(Xa, tr['winner'], sample_weight=match_weights(tr['match']))
        match_preds[name] = m.predict_proba(Xb)[:, 1]
        report[name] = metrics(te['winner'], match_preds[name], te['match'])
        print(f'[predictor] {name} AUC {report[name]["auc"]:.6f} ({time.time()-started:.0f}s)', flush=True)
    report['B_minus_A'] = paired_bootstrap(te['winner'], match_preds['A_pre_information'],
                                          match_preds['B_plus_predicted_engagement'], te['match'])

    np.savez_compressed(out / 'predictions.npz', train_id=tr['id'], train_oof=oof, train_y=tr['y'],
                        train_delta=tr['delta'], test_id=te['id'], test_match=te['match'],
                        test_y=te['y'], test_delta=te['delta'], test_market=te['market'],
                        test_p_deltaV_weighted=p, test_p_deltaV_unweighted=p_plain,
                        test_p_market=p_gold, test_scale=te['scale'], test_minute=te['minute'],
                        **match_preds)
    report['elapsed_seconds'] = round(time.time() - started, 2)
    report['status'] = 'complete'
    write_json(out / 'results.json', report)
    from importlib.metadata import version
    write_json(out / 'manifest.json', {
        'command': ' '.join([sys.executable, *sys.argv]), 'python': sys.version,
        'platform': platform.platform(),
        'packages': {q: version(q) for q in ('numpy', 'scikit-learn', 'lightgbm', 'joblib')},
        'inputs': {n: sha256(a.v3_dir / n) for n in ('engagement_changes.npz',
                                                     'engagement_changes_predict_train.npz',
                                                     'expanded_model.joblib', 'maymin_model.joblib')},
        'source': {n: sha256(ROOT / n) for n in ('scripts/train_engagement_predictor_v3.py',
                                                 'train/state_value_experiment.py')},
        'elapsed_seconds': report['elapsed_seconds']})
    write_json(out / 'status.json', {'stage': 'complete', 'pid': os.getpid(),
                                     'elapsed_seconds': report['elapsed_seconds']})
    print(json.dumps({k: report[k] for k in ('rows', 'label_disagreement', 'B_minus_A')}, indent=2,
                     ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v3-dir', type=Path, default=ROOT / 'outputs/temporal_winprob_v3_buckets')
    ap.add_argument('--dataset', type=Path, default=ROOT / 'outputs/state_value_main_50k')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/engagement_predictor_v3')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--trees', type=int, default=250)
    args = ap.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / 'status.json', {'stage': 'failed', 'error': repr(exc), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
