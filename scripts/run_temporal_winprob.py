"""Fit independent snapshot WP models, evaluate time curves, then measure fight changes."""
import argparse
from collections import Counter
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
    maymin_model, calibrated_variants, observed_queries)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(a):
    started = time.time()
    out, dataset = a.out_dir.resolve(), a.dataset.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise ValueError('completed output already exists')
    manifest = read_json(dataset / 'manifest.json')
    assert manifest['status'] == 'complete'
    names = read_json(dataset / 'schema.json')['state_names']
    old_splits = read_json(a.previous_eval / 'match_splits.json')
    ranked = sorted(old_splits['value_train'], key=lambda m: stable_int('partition:7:' + m))
    n = len(ranked)
    splits = {'fit': ranked[:int(.7*n)], 'calibrate': ranked[int(.7*n):int(.85*n)],
              'select': ranked[int(.85*n):], 'test': old_splits['value_validation'],
              'engagement': old_splits['predict_test']}
    if a.engagement_matches:
        splits['engagement'] = sorted(splits['engagement'], key=lambda m: stable_int('engagement:7:' + m))[:a.engagement_matches]
    assert_disjoint(*splits.values())
    sources = ['train/temporal_winprob.py', 'scripts/run_temporal_winprob.py',
               'gameplay/state_value.py', 'train/state_value_experiment.py']
    protocol = {'version': 'independent_wp_v2_no_age', 'source_dataset': str(dataset),
                'excluded_value_features': ['snapshot_age_s'],
                'sampling': 'one seeded random nonterminal minute per match, existing grid from minute 2',
                'selection': 'lowest selection-match log_loss; expanded family mandatory for primary objectives',
                'boundaries': 'cutoff-1; last cluster kill inclusive; +30s; +60s; no future snapshots',
                'engagement_matches': a.engagement_matches, 'splits': splits,
                'source_hashes': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                'prior_data_use': 'same 50k exploratory corpus; this is not a newly untouched external test'}
    if (out / 'protocol.json').exists() and read_json(out / 'protocol.json') != protocol:
        raise ValueError('protocol changed; new output directory required')
    write_json(out / 'protocol.json', protocol)
    for p in sources:
        saved = out/'source_snapshot'/p
        saved.parent.mkdir(parents=True,exist_ok=True)
        saved.write_bytes((ROOT/p).read_bytes())
    write_json(out / 'status.json', {'stage': 'training', 'pid': os.getpid()})
    data, test_grid, sampling = {}, [], []
    for role in ('fit', 'calibrate', 'select', 'test'):
        X, y, groups, ts = [], [], [], []
        for mid in splits[role]:
            with np.load(dataset/'matches'/(mid+'.npz'), allow_pickle=False) as f:
                # Neither fight pre/post matrices nor engagement labels enter training.
                grid, grid_times, winner = f['value_states'], f['value_times'], int(f['winner'])
            i = random_minute_index(mid, len(grid))
            X.append(grid[i]); y.append(winner); groups.append(mid); ts.append(int(grid_times[i]))
            sampling.append({'match': mid, 'role': role, 'time_ms': int(grid_times[i])})
            if role == 'test':
                test_grid.append((mid, winner, grid_times, grid))
        data[role] = (np.asarray(X), np.asarray(y), np.asarray(groups), np.asarray(ts))
    write_json(out / 'sampled_minutes.json', sampling)
    fit, cal, sel, test = [data[r] for r in ('fit', 'calibrate', 'select', 'test')]
    candidates, candidate_scores, chosen = {}, {}, {}
    for family in ('maymin', 'expanded'):
        X, cols = feature_matrix(fit[0], names, family)
        if family == 'maymin':
            base = maymin_model().fit(X, fit[1])
        else:
            C, cv = select_value_regularization(X, fit[1], fit[2], cols, candidates=(.001, .01, .1, 1.))
            write_json(out / 'expanded_training_cv.json', cv)
            base = logistic(cols, C=C).fit(X, fit[1])
        variants = calibrated_variants(family, names, base, cal[0], cal[1])
        for method, model in variants.items():
            key = family + '_' + method
            candidates[key] = model
            candidate_scores[key] = metrics(sel[1], model.predict_proba(sel[0])[:, 1], sel[2])
        chosen[family] = min((k for k in candidates if k.startswith(family+'_')), key=lambda k: candidate_scores[k]['log_loss'])
    write_json(out / 'selection.json', {'chosen': chosen, 'scores': candidate_scores})
    models = {family: candidates[key] for family, key in chosen.items()}
    # Freeze selection and artifacts BEFORE inspecting independent test or fights.
    model_ids = {}
    for family, model in models.items():
        path = out/(family+'_model.joblib')
        tmp = path.with_suffix('.tmp')
        joblib.dump(model, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('frozen model changed on resume; use a new output directory')
        tmp.replace(path)
        model_ids[family] = digest
    report = {'status': 'evaluating', 'protocol': protocol['version'], 'chosen': chosen,
              'model_ids': model_ids, 'partition_matches': {r:len(ids) for r,ids in splits.items()},
              'test_one_minute': {}, 'test_time_bands': {}, 'primary': 'expanded',
              'model_semantics': 'P(final Blue win | current observed state), then observed time differences, not causal effects'}
    grid_X = np.concatenate([z[3] for z in test_grid])
    grid_t = np.concatenate([z[2] for z in test_grid])
    grid_y = np.concatenate([np.repeat(z[1], len(z[2])) for z in test_grid])
    grid_g = np.concatenate([np.repeat(z[0], len(z[2])) for z in test_grid])
    grid_probs = {}
    for family, model in models.items():
        p = model.predict_proba(test[0])[:, 1]
        report['test_one_minute'][family] = metrics(test[1], p, test[2])
        grid_probs[family] = model.predict_proba(grid_X)[:, 1]
        report['test_time_bands'][family] = {}
        for lo, hi in ((2,10), (10,20), (20,30), (30,1000)):
            mask = (grid_t >= lo*60000) & (grid_t < hi*60000)
            if mask.any():
                report['test_time_bands'][family][f'{lo}-{hi}'] = metrics(grid_y[mask], grid_probs[family][mask], grid_g[mask])
    report['test_one_minute']['training_prior'] = metrics(test[1], np.repeat(fit[1].mean(),len(test[1])),test[2])
    np.savez_compressed(out/'independent_time_curves.npz', match=grid_g, time_ms=grid_t, winner=grid_y, **grid_probs)
    write_json(out / 'value_validation.json', report)
    print('[WP] independent model fitting and temporal validation complete', flush=True)
    del grid_X, grid_y, grid_g, grid_probs, test_grid

    # This second stage can read engagement boundaries; the value models are frozen.
    write_json(out / 'status.json', {'stage': 'engagement_changes', 'pid': os.getpid()})
    os.environ['LOL_OUTPUT_ROOT'] = str(out/'runtime')
    os.environ['LOL_CFG_PRESET'] = 'v3.3'
    os.environ['LOL_CFG_OVERRIDES'] = json.dumps({'CACHE_DIRNAME': str(a.cache_dir.resolve()),
        'FIGHT_INDEX_CACHE_ENABLED': False, 'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False})
    from core.config import cfg, NODE_FEATURE_NAMES
    cfg.LABEL_TIE_POLICY = 'random'
    from data.cache_io import load_match_cache
    from data.index_split import build_fight_index
    from gameplay.state_value import StateBuilder, final_outcome
    cache_out = out/'engagement_matches'
    cache_out.mkdir(exist_ok=True)
    records, excluded = [], Counter()
    for number, mid in enumerate(splits['engagement'], 1):
        path = cache_out/(mid+'.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                records.append({k:z[k] for k in z.files})
            continue
        pack = load_match_cache(mid)
        if pack is None:
            raise ValueError('source cache disappeared: '+mid)
        _, terminal = final_outcome(pack['events'])
        builder = StateBuilder(pack, NODE_FEATURE_NAMES)
        refs = build_fight_index(cache_match_ids=[mid])
        with np.load(dataset/'matches'/(mid+'.npz'), allow_pickle=False) as old:
            eligible = set(old['engagement_id'])
        rows, match_excluded = [], Counter()
        for ref in refs:
            eid = f'{mid}:{ref.t_start_ts}'
            if eid not in eligible:
                continue
            try:
                q = observed_queries(ref.t_start_ts, ref.last_kill_ts, terminal, int(builder.ts[-1]))
                states = [builder.at(t) if t >= 0 else None for t in q]
            except ValueError as exc:
                match_excluded[str(exc)] += 1
                continue
            probs = {}
            present = [i for i,s in enumerate(states) if s is not None]
            mat = np.array([list(states[i].values.values()) for i in present])
            for family, model in models.items():
                p = np.full(4, np.nan)
                p[present] = model.predict_proba(mat)[:,1]
                probs[family] = p
            objectives = []
            for obj in ('baron','dragons','elder','soul_event_recorded'):
                objectives.append([states[1].values[t+obj]-states[0].values[t+obj] for t in ('blue_','red_')])
            rows.append((eid, q, [s.snapshot_ms if s else -1 for s in states],
                         min(ref.det_cluster_blue,ref.det_cluster_red), objectives, probs))
        rec = {'_excluded':np.array([json.dumps(dict(match_excluded))]),
               'id':np.array([r[0] for r in rows],dtype=str), 'match':np.repeat(mid,len(rows)),
               'query_ms':np.array([r[1] for r in rows],dtype=np.int64).reshape(-1,4),
               'snapshot_ms':np.array([r[2] for r in rows],dtype=np.int64).reshape(-1,4),
               'scale':np.array([r[3] for r in rows]), 'objectives':np.array([r[4] for r in rows]).reshape(-1,4,2)}
        for family in models:
            rec[family] = np.array([r[5][family] for r in rows]).reshape(-1,4)
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, **rec)
        path.with_suffix('.tmp').replace(path)
        records.append(rec)
        if number % 100 == 0:
            print(f'[engagement] {number}/{len(splits["engagement"])} elapsed={time.time()-started:.0f}s', flush=True)
    table = {k:np.concatenate([r[k] for r in records]) for k in records[0]}
    for record in table.pop('_excluded'):
        excluded.update(json.loads(str(record)))
    assert len(set(table['id'])) == len(table['id'])
    assert set(table['match']).issubset(splits['engagement'])
    assert_disjoint(table['match'], *[splits[k] for k in ('fit','calibrate','select','test')])
    for family in models:
        table[family+'_delta'] = table[family][:,1:] - table[family][:,:1]
    np.savez_compressed(out/'engagement_changes.npz', **table)
    report['engagements'] = {'n':len(table['id']), 'matches':len(set(table['match'])),
                             'boundary_exclusions':dict(excluded), 'snapshot_age_ms_quantiles':np.quantile(table['query_ms'][:,:2]-table['snapshot_ms'][:,:2],[.5,.9,1]).tolist()}
    report['changes'] = {}
    for family in models:
        report['changes'][family] = {}
        for i, horizon in enumerate(('last_kill_proxy','plus_30s','plus_60s')):
            d = table[family+'_delta'][:,i]
            valid = np.isfinite(d)
            report['changes'][family][horizon] = {'n':int(valid.sum()), 'mean_percentage_points':float(np.mean(d[valid])*100),
                'blue_increase_fraction':float(np.mean(d[valid]>0)), 'tie_fraction':float(np.mean(d[valid]==0)),
                'absolute_change_pp_quantiles':np.quantile(np.abs(d[valid])*100,[.25,.5,.75,.9]).tolist()}
    report['model_direction_agreement'] = float(np.mean(np.sign(table['maymin_delta'][:,0])==np.sign(table['expanded_delta'][:,0])))
    report['observed_objective_windows'] = {obj:int(np.any(table['objectives'][:,i,:]>0,axis=1).sum())
                                           for i,obj in enumerate(('baron','dragons','elder','soul_event_recorded'))}
    report['status'] = 'complete'
    report['elapsed_seconds'] = round(time.time()-started,2)
    write_json(out/'results.json', report)
    write_json(out/'status.json', {'stage':'complete','pid':os.getpid()})
    print(json.dumps({k:report[k] for k in ('test_one_minute','engagements','changes','observed_objective_windows')},indent=2),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--previous-eval',type=Path,required=True)
    ap.add_argument('--cache-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--engagement-matches',type=int,default=0,help='0 = all existing held-out engagement matches')
    args = ap.parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.out_dir.exists():
            write_json(args.out_dir/'status.json',{'stage':'failed','error':repr(exc),'pid':os.getpid()})
        raise
