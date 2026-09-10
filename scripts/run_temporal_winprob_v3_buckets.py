"""v3: refit the snapshot win-probability models with 5-minute bucket sampling.

v2 followed Maymin (2021) and sampled ONE random minute per match.  That inherits the match-length
distribution, so only 211 of 7,004 fit rows (3.01%) sit past minute 30.  Validations A and C traced
two defects to that thin tail: the 30+ band is symmetrically overconfident (ECE 0.0929) and the
expanded model's elder response is inverted (correct sign on 2 of 52 events, elder median 33.9 min).

Jalovaara (2024) §5.1 samples per 5-MINUTE BUCKET precisely to stop training data concentrating
where matches are dense.  This run applies that, and changes NOTHING else:

  * same splits, read from the same match_splits.json
  * same two families, same feature sets (7 / 361), same C candidates, same 3-fold grouped CV
  * same calibration variants and the same selection rule (lowest select-partition log loss)
  * TEST SAMPLING IS UNCHANGED - one random minute per test match, same seed - so the headline
    test metric is directly comparable to v2, and the full-grid time bands are identical in
    definition.  Only fit / calibrate / select change.

v2 artefacts are never touched; everything lands in a new output directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import assert_disjoint, logistic, metrics, select_value_regularization
from train.temporal_winprob import (stable_int, random_minute_index, feature_matrix,
                                    maymin_model, calibrated_variants)

BUCKET_MS = 300_000          # 5 minutes, per Jalovaara (2024) 5.1
TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bucket_minute_indices(match_id, grid_times, bucket_ms=BUCKET_MS):
    """One seeded random grid point per 5-minute bucket; deterministic in the match id."""
    buckets = defaultdict(list)
    for i, t in enumerate(grid_times):
        buckets[int(t) // bucket_ms].append(i)
    picked = []
    for b in sorted(buckets):
        idxs = buckets[b]
        rng = np.random.default_rng(stable_int(f'bucket:7:{match_id}:{b}'))
        picked.append(idxs[int(rng.integers(len(idxs)))])
    return picked


def ece_from(y, p, w, bins=10):
    """Match-weighted expected calibration error over fixed equal-width bins."""
    edges = np.linspace(0., 1., bins + 1)
    total, out = w.sum(), 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi + (1e-10 if hi >= 1. else 0.))
        if mask.any():
            ww = w[mask]
            out += float(ww.sum() / total) * abs(
                float(np.average(y[mask], weights=ww)) - float(np.average(p[mask], weights=ww)))
    return float(out)


def match_weights(groups):
    _, ix, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / counts[ix]
    return w / w.mean()


def run(a):
    started = time.time()
    out, dataset = a.out_dir.resolve(), a.dataset.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise ValueError('completed output already exists; use a new directory')
    manifest = read_json(dataset / 'manifest.json')
    assert manifest['status'] == 'complete'
    names = read_json(dataset / 'schema.json')['state_names']
    old_splits = read_json(a.previous_eval / 'match_splits.json')
    ranked = sorted(old_splits['value_train'], key=lambda m: stable_int('partition:7:' + m))
    n = len(ranked)
    splits = {'fit': ranked[:int(.7*n)], 'calibrate': ranked[int(.7*n):int(.85*n)],
              'select': ranked[int(.85*n):], 'test': old_splits['value_validation'],
              'engagement': old_splits['predict_test']}
    assert_disjoint(*splits.values())
    sources = ['train/temporal_winprob.py', 'scripts/run_temporal_winprob_v3_buckets.py',
               'gameplay/state_value.py', 'train/state_value_experiment.py']
    protocol = {
        'version': 'independent_wp_v3_bucket_sampling',
        'derived_from': 'outputs/temporal_winprob_v2 (protocol independent_wp_v2_no_age)',
        'single_change': 'fit/calibrate/select sample one grid point per 5-minute bucket per match '
                         '(Jalovaara 2024 5.1) instead of one random minute per match (Maymin 2021 3.2)',
        'test_sampling': 'UNCHANGED: one seeded random minute per test match, identical to v2',
        'source_dataset': str(dataset), 'excluded_value_features': ['snapshot_age_s'],
        'bucket_ms': BUCKET_MS, 'splits': splits,
        'source_hashes': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'prior_data_use': 'same 50k exploratory corpus; this is not a newly untouched external test'}
    write_json(out / 'protocol.json', protocol)
    for p in sources:
        saved = out / 'source_snapshot' / p
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes((ROOT / p).read_bytes())

    write_json(out / 'status.json', {'stage': 'sampling', 'pid': os.getpid()})
    data, test_grid, sampling = {}, [], []
    for role in ('fit', 'calibrate', 'select', 'test'):
        X, y, groups, ts = [], [], [], []
        for mid in splits[role]:
            with np.load(dataset/'matches'/(mid+'.npz'), allow_pickle=False) as f:
                grid, grid_times, winner = f['value_states'], f['value_times'], int(f['winner'])
            picked = ([random_minute_index(mid, len(grid))] if role == 'test'
                      else bucket_minute_indices(mid, grid_times))
            for i in picked:
                X.append(grid[i]); y.append(winner); groups.append(mid); ts.append(int(grid_times[i]))
                sampling.append({'match': mid, 'role': role, 'time_ms': int(grid_times[i])})
            if role == 'test':
                test_grid.append((mid, winner, grid_times, grid))
        data[role] = (np.asarray(X), np.asarray(y), np.asarray(groups), np.asarray(ts))
    write_json(out / 'sampled_minutes.json', sampling)

    fit, cal, sel, test = [data[r] for r in ('fit', 'calibrate', 'select', 'test')]
    census = {}
    for role in ('fit', 'calibrate', 'select', 'test'):
        t = data[role][3]
        census[role] = {'rows': int(len(t)), 'matches': int(len(set(data[role][2].tolist()))),
                        'rows_per_match': round(len(t) / len(set(data[role][2].tolist())), 3),
                        'bands': {f'{lo}-{hi}': int(((t >= lo*60000) & (t < hi*60000)).sum())
                                  for lo, hi in TIME_BANDS}}
    write_json(out / 'sampling_census.json', census)
    print('[v3] sampling census:', json.dumps(census['fit'], ensure_ascii=False), flush=True)

    write_json(out / 'status.json', {'stage': 'training', 'pid': os.getpid()})
    candidates, candidate_scores, chosen = {}, {}, {}
    for family in ('maymin', 'expanded'):
        X, cols = feature_matrix(fit[0], names, family)
        if family == 'maymin':
            base = maymin_model().fit(X, fit[1])
        elif a.fixed_c:
            base = logistic(cols, C=a.fixed_c).fit(X, fit[1])
            write_json(out / 'expanded_training_cv.json',
                       {'skipped': 'C fixed by --fixed-c to isolate the sampling change',
                        'C': a.fixed_c})
        else:
            C, cv = select_value_regularization(X, fit[1], fit[2], cols, candidates=(.001, .01, .1, 1.))
            write_json(out / 'expanded_training_cv.json', cv)
            base = logistic(cols, C=C).fit(X, fit[1])
        print(f'[v3] {family} base fitted ({time.time()-started:.0f}s)', flush=True)
        for method, model in calibrated_variants(family, names, base, cal[0], cal[1]).items():
            key = family + '_' + method
            candidates[key] = model
            candidate_scores[key] = metrics(sel[1], model.predict_proba(sel[0])[:, 1], sel[2])
        chosen[family] = min((k for k in candidates if k.startswith(family+'_')),
                             key=lambda k: candidate_scores[k]['log_loss'])
    write_json(out / 'selection.json', {'chosen': chosen, 'scores': candidate_scores})
    models = {family: candidates[key] for family, key in chosen.items()}

    model_ids = {}
    for family, model in models.items():
        path = out / (family + '_model.joblib')
        joblib.dump(model, path)
        model_ids[family] = hashlib.sha256(path.read_bytes()).hexdigest()

    report = {'status': 'evaluating', 'protocol': protocol['version'], 'chosen': chosen,
              'model_ids': model_ids, 'sampling_census': census,
              'partition_matches': {r: len(ids) for r, ids in splits.items()},
              'test_one_minute': {}, 'test_time_bands': {}, 'primary': 'expanded',
              'model_semantics': 'P(final Blue win | current observed state); observed differences, not causal effects'}
    grid_X = np.concatenate([z[3] for z in test_grid])
    grid_t = np.concatenate([z[2] for z in test_grid])
    grid_y = np.concatenate([np.repeat(z[1], len(z[2])) for z in test_grid])
    grid_g = np.concatenate([np.repeat(z[0], len(z[2])) for z in test_grid])
    grid_probs = {}
    for family, model in models.items():
        p = model.predict_proba(test[0])[:, 1]
        cell = metrics(test[1], p, test[2])
        cell['ece'] = ece_from(test[1], p, match_weights(test[2]))
        report['test_one_minute'][family] = cell
        grid_probs[family] = model.predict_proba(grid_X)[:, 1]
        report['test_time_bands'][family] = {}
        for lo, hi in TIME_BANDS:
            mask = (grid_t >= lo*60000) & (grid_t < hi*60000)
            if mask.any():
                band = metrics(grid_y[mask], grid_probs[family][mask], grid_g[mask])
                band['ece'] = ece_from(grid_y[mask], grid_probs[family][mask],
                                       match_weights(grid_g[mask]))
                report['test_time_bands'][family][f'{lo}-{hi}'] = band
    report['test_one_minute']['training_prior'] = metrics(
        test[1], np.repeat(fit[1].mean(), len(test[1])), test[2])
    np.savez_compressed(out / 'independent_time_curves.npz',
                        match=grid_g, time_ms=grid_t, winner=grid_y, **grid_probs)
    report['status'] = 'complete'
    report['elapsed_seconds'] = round(time.time() - started, 2)
    write_json(out / 'results.json', report)
    write_json(out / 'status.json', {'stage': 'complete', 'pid': os.getpid(),
                                     'elapsed_seconds': report['elapsed_seconds']})
    print(json.dumps({'chosen': chosen,
                      'test_one_minute': {f: {k: report['test_one_minute'][f][k]
                                              for k in ('auc', 'brier', 'log_loss', 'ece')}
                                          for f in ('maymin', 'expanded')},
                      'expanded_bands': {b: {k: v[k] for k in ('n', 'matches', 'auc', 'brier', 'ece')}
                                         for b, v in report['test_time_bands']['expanded'].items()},
                      'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, default=ROOT / 'outputs/state_value_main_50k')
    ap.add_argument('--previous-eval', type=Path, default=ROOT / 'outputs/state_value_main_50k_eval')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/temporal_winprob_v3_buckets')
    ap.add_argument('--fixed-c', type=float, default=None,
                    help='skip C selection and use this value (isolates the sampling change)')
    args = ap.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / 'status.json',
                   {'stage': 'failed', 'error': repr(exc), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
