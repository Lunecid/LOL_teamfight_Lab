"""Score the same engagement query points with the v3 (bucket-sampled) models.

Boundaries, splits, engagement population and query times are IDENTICAL to v2 - the same
observed_queries() on the same fight index over the same 9,198 held-out matches.  Only the two
models change, so any difference in delta-V is attributable to the refit and nothing else.

Writes engagement_changes.npz in the v2 layout so the validation A and B scripts run unchanged
against this directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.temporal_winprob import observed_queries


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def run(a):
    started = time.time()
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'status_engagement.json', {'stage': 'running', 'pid': os.getpid()})

    protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
    if a.split == 'engagement':
        engagement_ids = protocol['splits']['engagement']          # = predict_test
    else:
        # predict_train is disjoint from every value-model partition, so labels scored here can
        # train the engagement predictor without the value model ever having seen those matches.
        old_splits = json.loads((a.previous_eval / 'match_splits.json').read_text(encoding='utf-8'))
        engagement_ids = old_splits[a.split]
        for role, ids in protocol['splits'].items():
            if set(ids) & set(engagement_ids):
                raise ValueError(f'{a.split} overlaps value partition {role}')
    models = {f: joblib.load(out / f'{f}_model.joblib') for f in ('maymin', 'expanded')}

    os.environ['LOL_OUTPUT_ROOT'] = str(out / 'runtime_engagement')
    os.environ['LOL_CFG_PRESET'] = 'v3.3'
    os.environ['LOL_CFG_OVERRIDES'] = json.dumps({
        'CACHE_DIRNAME': str(a.cache_dir.resolve()), 'FIGHT_INDEX_CACHE_ENABLED': False,
        'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False})
    from core.config import cfg, NODE_FEATURE_NAMES
    cfg.LABEL_TIE_POLICY = 'random'
    from data.cache_io import load_match_cache
    from data.index_split import build_fight_index
    from gameplay.state_value import StateBuilder, final_outcome

    records, excluded = [], Counter()
    for number, mid in enumerate(engagement_ids, 1):
        pack = load_match_cache(mid)
        if pack is None:
            raise ValueError('source cache disappeared: ' + mid)
        _, terminal = final_outcome(pack['events'])
        builder = StateBuilder(pack, NODE_FEATURE_NAMES)
        refs = build_fight_index(cache_match_ids=[mid])
        # The engagement population is pinned to v2's: same ids, same eligibility.
        with np.load(a.dataset / 'matches' / (mid + '.npz'), allow_pickle=False) as old:
            eligible = set(old['engagement_id'])
        rows = []
        for ref in refs:
            eid = f'{mid}:{ref.t_start_ts}'
            if eid not in eligible:
                continue
            try:
                q = observed_queries(ref.t_start_ts, ref.last_kill_ts, terminal, int(builder.ts[-1]))
                states = [builder.at(t) if t >= 0 else None for t in q]
            except ValueError as exc:
                excluded[str(exc)] += 1
                continue
            present = [i for i, s in enumerate(states) if s is not None]
            mat = np.array([list(states[i].values.values()) for i in present])
            probs = {}
            for family, model in models.items():
                p = np.full(4, np.nan)
                p[present] = model.predict_proba(mat)[:, 1]
                probs[family] = p
            objectives = []
            for obj in ('baron', 'dragons', 'elder', 'soul_event_recorded'):
                objectives.append([states[1].values[t + obj] - states[0].values[t + obj]
                                   for t in ('blue_', 'red_')])
            rows.append((eid, q, [s.snapshot_ms if s else -1 for s in states],
                         min(ref.det_cluster_blue, ref.det_cluster_red), objectives, probs))
        if rows:
            rec = {'id': np.array([r[0] for r in rows], dtype=str),
                   'match': np.repeat(mid, len(rows)),
                   'query_ms': np.array([r[1] for r in rows], dtype=np.int64).reshape(-1, 4),
                   'snapshot_ms': np.array([r[2] for r in rows], dtype=np.int64).reshape(-1, 4),
                   'scale': np.array([r[3] for r in rows]),
                   'objectives': np.array([r[4] for r in rows]).reshape(-1, 4, 2)}
            for family in models:
                rec[family] = np.array([r[5][family] for r in rows]).reshape(-1, 4)
            records.append(rec)
        if number % 500 == 0:
            write_json(out / 'status_engagement.json',
                       {'stage': f'engagement {number}/{len(engagement_ids)}', 'pid': os.getpid(),
                        'elapsed_seconds': round(time.time() - started, 1)})
            print(f'[v3 engagement] {number}/{len(engagement_ids)} '
                  f'elapsed={time.time()-started:.0f}s', flush=True)

    table = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
    assert len(set(table['id'])) == len(table['id'])
    for family in models:
        table[family + '_delta'] = table[family][:, 1:] - table[family][:, :1]
    np.savez_compressed(out / a.output_name, **table)

    report = {'engagements': {'n': len(table['id']), 'matches': len(set(table['match'])),
                              'boundary_exclusions': dict(excluded)}, 'changes': {}}
    for family in models:
        report['changes'][family] = {}
        for i, horizon in enumerate(('last_kill_proxy', 'plus_30s', 'plus_60s')):
            d = table[family + '_delta'][:, i]
            valid = np.isfinite(d)
            report['changes'][family][horizon] = {
                'n': int(valid.sum()),
                'mean_percentage_points': float(np.mean(d[valid]) * 100),
                'blue_increase_fraction': float(np.mean(d[valid] > 0)),
                'tie_fraction': float(np.mean(d[valid] == 0)),
                'absolute_change_pp_quantiles': np.quantile(np.abs(d[valid]) * 100, [.25, .5, .75, .9]).tolist()}
    report['model_direction_agreement'] = float(
        np.mean(np.sign(table['maymin_delta'][:, 0]) == np.sign(table['expanded_delta'][:, 0])))
    report['observed_objective_windows'] = {
        obj: int(np.any(table['objectives'][:, i, :] > 0, axis=1).sum())
        for i, obj in enumerate(('baron', 'dragons', 'elder', 'soul_event_recorded'))}
    report['split'] = a.split
    report['output'] = a.output_name
    report['elapsed_seconds'] = round(time.time() - started, 2)
    report['status'] = 'complete'
    suffix = '' if a.split == 'engagement' else '_' + a.split
    write_json(out / f'engagement_report{suffix}.json', report)
    write_json(out / 'status_engagement.json',
               {'stage': 'complete', 'pid': os.getpid(), 'split': a.split,
                'elapsed_seconds': report['elapsed_seconds']})
    print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/temporal_winprob_v3_buckets')
    ap.add_argument('--dataset', type=Path, default=ROOT / 'outputs/state_value_main_50k')
    ap.add_argument('--cache-dir', type=Path,
                    default=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13'))
    ap.add_argument('--previous-eval', type=Path, default=ROOT / 'outputs/state_value_main_50k_eval')
    ap.add_argument('--split', default='engagement', choices=('engagement', 'predict_train'),
                    help="'engagement' = predict_test (v2 population); 'predict_train' = predictor training matches")
    ap.add_argument('--output-name', default='engagement_changes.npz')
    args = ap.parse_args()
    try:
        run(args)
    except Exception as exc:
        write_json(args.out_dir / 'status_engagement.json',
                   {'stage': 'failed', 'error': repr(exc), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
