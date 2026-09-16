"""P1: independent V2 snapshot win-probability training and calibration.

Rebuilds objective_history_v2_participant_order states from the raw match cache at the EXACT
stored queries of temporal_winprob_v3_buckets (sampled_minutes.json rows and the full test grid
keys of independent_time_curves.npz), then repeats that run's model family, regularisation rule,
calibration variants and selection rule unchanged. Old artefacts are read only; every output
lands in outputs/independent_v2_participant_order (or its smoke_run subdirectory).
"""
from __future__ import annotations

import os

_WORKER = os.environ.get('P1_WORKER') == '1'
_THREADS = '1' if _WORKER else '4'
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = _THREADS
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('TEAMFIGHT_WORKSPACE') and Path(os.environ['TEAMFIGHT_WORKSPACE']).resolve() != ROOT:
    raise SystemExit('script is not inside TEAMFIGHT_WORKSPACE')
WT = ROOT / 'worktrees' / 'engagement-state-value'
sys.path.insert(0, str(WT))
OLD = WT / 'outputs' / 'temporal_winprob_v3_buckets'
OLD_DATASET = WT / 'outputs' / 'state_value_main_50k'
OLD_EVAL = WT / 'outputs' / 'state_value_main_50k_eval'
V2FIX = ROOT / 'outputs' / 'state_value_v2_fix'
CACHE = Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
OUT_BASE = ROOT / 'outputs' / 'independent_v2_participant_order'
PARTITIONS = ('fit', 'calibrate', 'select', 'test')
FAMILIES = ('maymin', 'expanded')
C_CANDIDATES = (.001, .01, .1, 1.)
TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))
N_BOOT_PRIMARY = 2000
N_BOOT_STRATA = 500
BOOT_SEED = 7

LEGACY_SOURCES = ('gameplay/state_value.py', 'gameplay/state_value_v2.py', 'train/temporal_winprob.py',
                  'train/state_value_experiment.py', 'scripts/run_temporal_winprob_v3_buckets.py',
                  'scripts/build_state_value_dataset.py', 'core/config.py', 'data/cache_io.py')
NEW_SOURCES = {'scripts/train_independent_v2.py': ROOT / 'scripts' / 'train_independent_v2.py',
               'worktrees/engagement-state-value/train/independent_winprob_v2.py':
                   WT / 'train' / 'independent_winprob_v2.py'}

os.environ['LOL_OUTPUT_ROOT'] = str(OUT_BASE / 'runtime')
os.environ['LOL_CFG_PRESET'] = 'v3.3'
os.environ['LOL_CFG_OVERRIDES'] = json.dumps({
    'CACHE_DIRNAME': str(CACHE), 'FIGHT_INDEX_CACHE_ENABLED': False,
    'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False, 'CACHE_IN_RAM': False})


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


class Run:
    def __init__(self, out, smoke):
        self.out, self.smoke, self.started = out, smoke, time.time()
        out.mkdir(parents=True, exist_ok=True)
        self.status = {'stage': 'starting', 'pid': os.getpid(), 'smoke': smoke, 'history': []}

    def log(self, msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")} +{time.time()-self.started:7.0f}s] {msg}'
        print(line, flush=True)
        with (self.out / 'run.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    def stage(self, name, **info):
        self.status.update(stage=name, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                           elapsed_seconds=round(time.time() - self.started, 1), **info)
        if not self.status['history'] or self.status['history'][-1]['stage'] != name:
            self.status['history'].append({'stage': name, 'at': self.status['updated']})
        write_json(self.out / 'status.json', self.status)
        self.log(f'stage={name} {json.dumps(_jsonable(info), ensure_ascii=False)[:300]}')


def legacy_hashes():
    files = {p: sha256_file(WT / p) for p in LEGACY_SOURCES}
    files['tests/test_state_value_v2_contract.py'] = sha256_file(ROOT / 'tests' / 'test_state_value_v2_contract.py')
    old = {str(p.relative_to(OLD)).replace('\\', '/'): sha256_file(p) for p in sorted(OLD.rglob('*')) if p.is_file()}
    fix = {str(p.relative_to(V2FIX)).replace('\\', '/'): sha256_file(p) for p in sorted(V2FIX.glob('*')) if p.is_file()}
    return {'legacy_sources': files, 'old_v_artifacts': old, 'state_value_v2_fix': fix}


def expected_v2_names():
    v1 = read_json(OLD_DATASET / 'schema.json')['state_names']
    return v1, [re.sub(r'^slot(\d+)_', r'participant_slot\1_', n) for n in v1]


# ----------------------------------------------------------------------------- worker side
_CTX = {}


def _context():
    if not _CTX:
        from core.config import cfg, NODE_FEATURE_NAMES, CACHE_DIR
        from data import ram_cache
        cfg.CACHE_IN_RAM = False
        if hasattr(cfg, 'CACHE_MATCH_PACKS_IN_RAM'):
            cfg.CACHE_MATCH_PACKS_IN_RAM = False
        if ram_cache._ram_cache_enabled():
            raise RuntimeError('RAM pack cache unexpectedly enabled')
        if CACHE_DIR.resolve() != CACHE.resolve():
            raise RuntimeError(f'cache dir mismatch: {CACHE_DIR}')
        from data.cache_io import load_match_cache
        from gameplay.state_value import final_outcome, StateBuilder as Legacy, DRAGONS
        from gameplay.state_value_v2 import StateBuilder, soul_element, state_matrix, STATE_VERSION
        _CTX.update(load=load_match_cache, final_outcome=final_outcome, Legacy=Legacy, V2=StateBuilder,
                    soul_element=soul_element, state_matrix=state_matrix, DRAGONS=DRAGONS,
                    node_names=list(NODE_FEATURE_NAMES), feature_version=str(getattr(cfg, 'FEATURE_VERSION', '')),
                    state_version=STATE_VERSION)
    return _CTX


def _why_unreadable(mid, ctx):
    files = [CACHE / f'{mid}{s}' for s in ('.npz', '.events.json', '.meta.json')]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        return 'missing ' + ','.join(missing)
    meta = read_json(files[2])
    if str(meta.get('feature_version', '')) != ctx['feature_version']:
        return f"feature_version {meta.get('feature_version')!r}"
    return 'loader returned None'


def _load(mid, ctx):
    pack = ctx['load'](mid)
    if pack is None:
        raise ValueError('unreadable_or_incompatible_cache: ' + _why_unreadable(mid, ctx))
    return pack


def extract_chunk(task):
    """Build V2 states for every planned query of every match; any failure blocks the chunk."""
    started = time.time()
    ctx = _context()
    names = task['names']
    idx = {n: i for i, n in enumerate(names)}
    rows = defaultdict(list)
    per = defaultdict(list)
    grids, souls, errors = {}, {}, {}
    for mid, times in task['items']:
        try:
            pack = _load(mid, ctx)
            winner, terminal = ctx['final_outcome'](pack['events'])
            builder = ctx['V2'](pack, ctx['node_names'])
            ts = np.asarray(pack['minute_ts'], dtype=np.int64)
            states = [builder.at(t) for t in times]
            X = ctx['state_matrix'](states, names)
            if not np.isfinite(X).all():
                raise ValueError('non-finite state')
            for t, s in zip(times, states):
                snap = int(ts[np.searchsorted(ts, t, side='right') - 1])
                if s.query_ms != t or s.snapshot_ms != snap or s.snapshot_ms > t:
                    raise ValueError(f'query/snapshot time violation at {t}')
            audit = []
            for e in pack['events']:
                if e.get('type') != 'DRAGON_SOUL_GIVEN':
                    continue
                rec = {'ts': int(e['timestamp']), 'team': int(e.get('teamId', 0) or 0),
                       'fields': {f: e.get(f) for f in ('name', 'dragonSoul', 'soulType') if f in e}}
                try:
                    rec['element'] = ctx['soul_element'](e)
                except ValueError as exc:
                    rec['element'] = 'ERROR:' + str(exc)
                audit.append(rec)
            for i, (t, s) in enumerate(zip(times, states)):
                owned, unassigned = {100: set(), 200: set()}, 0
                for rec in audit:
                    if rec['ts'] <= t:
                        if rec['team'] in owned:
                            owned[rec['team']].add(rec['element'])
                        else:
                            unassigned += 1
                for team, prefix in ((100, 'blue_'), (200, 'red_')):
                    for d in ctx['DRAGONS']:
                        if X[i, idx[f'{prefix}soul_{d}']] != float(d in owned[team]):
                            raise ValueError(f'soul flag mismatch {prefix}{d} at {t}')
                    if X[i, idx[prefix + 'soul_event_recorded']] != float(bool(owned[team])):
                        raise ValueError(f'soul recorded mismatch {prefix} at {t}')
                if s.unassigned_soul_events != unassigned:
                    raise ValueError(f'unassigned soul count mismatch at {t}')
            tm = {int(k): int(v) for k, v in pack['meta']['team_map'].items()}
            slots = pack['meta'].get('role_slots') or {}
            try:
                role_order = sorted(tm, key=lambda p: int(slots[p]))
                reordered = int(role_order != list(builder.participant_order))
            except (KeyError, TypeError, ValueError):
                reordered = -1
        except Exception as exc:
            errors[mid] = f'{type(exc).__name__}: {exc}'
            continue
        rows['match'].extend([mid] * len(times))
        rows['time_ms'].extend(times)
        rows['snapshot_ms'].extend(s.snapshot_ms for s in states)
        rows['unassigned'].extend(s.unassigned_soul_events for s in states)
        rows['X'].append(X)
        per['match_ids'].append(mid)
        per['winner'].append(winner)
        per['terminal_ms'].append(terminal)
        per['last_frame_ms'].append(int(ts[-1]))
        per['patch'].append(str(pack['meta'].get('patch', '')))
        per['reordered'].append(reordered)
        grids[mid] = list(range(120000, min(terminal, int(ts[-1]) + 1), 60000))
        if audit:
            souls[mid] = audit
    summary = {'chunk_id': task['chunk_id'], 'errors': errors, 'matches': len(per['match_ids']),
               'rows': len(rows['time_ms']), 'seconds': round(time.time() - started, 1), 'written': False}
    if errors:
        return summary
    save_npz(Path(task['path']),
             match=np.asarray(rows['match'], dtype='U16'), time_ms=np.asarray(rows['time_ms'], dtype=np.int64),
             snapshot_ms=np.asarray(rows['snapshot_ms'], dtype=np.int64),
             unassigned_soul_events=np.asarray(rows['unassigned'], dtype=np.int32),
             X=np.concatenate(rows['X']).astype(np.float64),
             match_ids=np.asarray(per['match_ids'], dtype='U16'), winner=np.asarray(per['winner'], dtype=np.int8),
             terminal_ms=np.asarray(per['terminal_ms'], dtype=np.int64),
             last_frame_ms=np.asarray(per['last_frame_ms'], dtype=np.int64),
             patch=np.asarray(per['patch'], dtype='U16'), reordered=np.asarray(per['reordered'], dtype=np.int8),
             raw_grid_json=np.asarray(json.dumps(grids)), soul_audit_json=np.asarray(json.dumps(souls)),
             names_sha256=np.asarray(sha256_json(names)), state_version=np.asarray(ctx['state_version']),
             plan_sha256=np.asarray(task['plan_sha256']))
    summary['written'] = True
    return summary


def audit_chunk(task):
    """Real-cache checks: role-slot invariance, future-data invariance, v1 loader fidelity."""
    ctx = _context()
    names, v1_names = task['names'], task['v1_names']
    results = []
    for mid, times in task['items']:
        rec = {'match': mid, 'queries': len(times), 'failures': []}
        try:
            pack = _load(mid, ctx)
            winner, _ = ctx['final_outcome'](pack['events'])
            V2, nn, sm = ctx['V2'], ctx['node_names'], ctx['state_matrix']
            base_states = [V2(pack, nn).at(t) for t in times]
            base = sm(base_states, names)
            rec['unassigned_soul_events_max'] = max(s.unassigned_soul_events for s in base_states)
            tm = {int(k): int(v) for k, v in pack['meta']['team_map'].items()}
            blue = sorted(p for p in tm if tm[p] == 100)
            red = sorted(p for p in tm if tm[p] == 200)
            variants = {'deleted_key': None, 'none': None, 'empty': {},
                        'reversed_within_team': {p: i for i, p in enumerate(blue[::-1] + red[::-1])},
                        'swapped_teams': {p: i for i, p in enumerate(red + blue)},
                        'garbage': {'invalid': 'future label'}}
            for vname, slots in variants.items():
                meta = dict(pack['meta'])
                if vname == 'deleted_key':
                    meta.pop('role_slots', None)
                else:
                    meta['role_slots'] = copy.deepcopy(slots)
                X = sm([V2(dict(pack, meta=meta), nn).at(t) for t in times], names)
                if not np.array_equal(X, base):
                    rec['failures'].append('role_slots_' + vname)
            rec['role_variants_checked'] = len(variants)
            ts = np.asarray(pack['minute_ts'], dtype=np.int64)
            future_failures, checked = 0, 0
            for i, t in enumerate(times):
                node = np.array(pack['node_minute'], copy=True)
                node[ts > t] = node[ts > t] * -3.5 + 777.0
                past = [e for e in pack['events'] if int(e['timestamp']) <= t]
                mutated = []
                for e in pack['events']:
                    if int(e['timestamp']) <= t:
                        continue
                    m = dict(e)
                    for k in ('teamId', 'killerTeamId', 'winningTeam'):
                        if int(m.get(k, 0) or 0) in (100, 200):
                            m[k] = 300 - int(m[k])
                    for k in ('killerId', 'victimId'):
                        if int(m.get(k, 0) or 0) > 0:
                            m[k] = int(m[k]) % 10 + 1
                    mutated.append(m)
                fake = [dict(type='DRAGON_SOUL_GIVEN', timestamp=t + 1, teamId=100, name='Cloud', dragonSoul='FIRE'),
                        dict(type='ELITE_MONSTER_KILL', timestamp=t + 1, killerTeamId=200, killerId=6,
                             monsterType='BARON_NASHOR'),
                        dict(type='CHAMPION_KILL', timestamp=t + 1, killerId=1, victimId=6),
                        dict(type='GAME_END', timestamp=t + 2, winningTeam=200)]
                for events in (past, past + mutated + fake):
                    x = sm([V2(dict(pack, node_minute=node, events=events), nn).at(t)], names)[0]
                    checked += 1
                    future_failures += int(not np.array_equal(x, base[i]))
            rec['future_variants_checked'] = checked
            if future_failures:
                rec['failures'].append(f'future_invariance:{future_failures}')
            legacy = ctx['Legacy'](pack, nn)
            v1_states = [legacy.at(t) for t in times]
            if any(list(s.values) != v1_names for s in v1_states):
                rec['failures'].append('v1_schema_order')
            v1 = np.asarray([[s.values[n] for n in v1_names] for s in v1_states], dtype=np.float32)
            with np.load(OLD_DATASET / 'matches' / f'{mid}.npz', allow_pickle=False) as z:
                pos = {int(t): i for i, t in enumerate(z['value_times'])}
                on_grid = [i for i, t in enumerate(times) if t in pos]
                stored = z['value_states'][[pos[times[i]] for i in on_grid]]
                old_winner = int(z['winner'])
            rec['v1_compared_queries'] = len(on_grid)
            rec['v1_recompute_equal'] = bool(on_grid) and bool(np.array_equal(v1[on_grid], stored))
            rec['v1_recompute_max_abs_diff'] = float(np.max(np.abs(v1[on_grid] - stored))) if on_grid else None
            if not rec['v1_recompute_equal']:
                rec['failures'].append('v1_loader_fidelity')
            rec['old_dataset_winner_equal'] = bool(old_winner == winner)
            if not rec['old_dataset_winner_equal']:
                rec['failures'].append('winner_mismatch_old_dataset')
        except Exception as exc:
            rec['failures'].append(f'exception {type(exc).__name__}: {exc}')
        results.append(rec)
    return results


# ----------------------------------------------------------------------------- protocol checks
def overlap_table(sets):
    keys = list(sets)
    return {f'{a}&{b}': len(sets[a] & sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}


def load_protocol_inputs():
    proto = read_json(OLD / 'protocol.json')
    sampled = read_json(OLD / 'sampled_minutes.json')
    with np.load(OLD / 'independent_time_curves.npz', allow_pickle=False) as z:
        curves = {k: z[k] for k in z.files}
    return proto, sampled, curves, read_json(OLD / 'results.json')


def protocol_checks(proto, sampled, curves, old_results):
    c, fail = {}, []
    splits = proto['splits']
    c['split_keys'] = sorted(splits)
    if set(splits) != {*PARTITIONS, 'engagement'}:
        fail.append('unexpected split keys')
    sets = {k: set(v) for k, v in splits.items()}
    c['split_sizes'] = {k: len(v) for k, v in splits.items()}
    c['split_duplicates'] = {k: len(v) - len(sets[k]) for k, v in splits.items()}
    c['protocol_overlaps'] = overlap_table(sets)
    if any(c['split_duplicates'].values()) or any(c['protocol_overlaps'].values()):
        fail.append('protocol split duplicates or overlaps')
    v_all = set().union(*(sets[p] for p in PARTITIONS))
    with (V2FIX / 'rows.csv').open(encoding='utf-8', newline='') as f:
        fix_matches = {r['match'] for r in csv.DictReader(f)}
    ms = read_json(OLD_EVAL / 'match_splits.json')
    external = {'protocol_engagement': sets['engagement'], 'state_value_v2_fix_engagement_sample': fix_matches,
                'eval_predict_train': set(ms['predict_train']), 'eval_predict_test': set(ms['predict_test'])}
    c['engagement_sources'] = {k: len(v) for k, v in external.items()}
    c['engagement_overlap_with_V'] = {k: {p: len(v & sets[p]) for p in PARTITIONS} for k, v in external.items()}
    if any(n for d in c['engagement_overlap_with_V'].values() for n in d.values()):
        fail.append('engagement match leakage into a V partition')
    c['eval_split_consistency'] = {
        'value_train_equals_fit_calibrate_select': set(ms['value_train']) == sets['fit'] | sets['calibrate'] | sets['select'],
        'value_validation_equals_test': set(ms['value_validation']) == sets['test'],
        'predict_test_equals_engagement': set(ms['predict_test']) == sets['engagement']}
    if not all(c['eval_split_consistency'].values()):
        fail.append('protocol splits disagree with state_value_main_50k_eval match_splits')
    c['sampled_rows'] = len(sampled)
    c['sampled_row_keys'] = sorted({tuple(sorted(r)) for r in sampled})
    if c['sampled_row_keys'] != [('match', 'role', 'time_ms')]:
        fail.append('sampled_minutes row schema changed')
    by_role = defaultdict(list)
    for r in sampled:
        by_role[r['role']].append(r)
    c['sampled_by_role'] = {}
    for p in PARTITIONS:
        rows = by_role.get(p, [])
        census = old_results['sampling_census'][p]
        cell = {'rows': len(rows), 'census_rows': census['rows'], 'matches': len({r['match'] for r in rows}),
                'match_set_equals_protocol_split': {r['match'] for r in rows} == sets[p]}
        cell['bands'] = {f'{lo}-{hi}': sum(lo * 60000 <= r['time_ms'] < hi * 60000 for r in rows) for lo, hi in TIME_BANDS}
        cell['bands_equal_census'] = cell['bands'] == census['bands']
        if not (cell['match_set_equals_protocol_split'] and cell['rows'] == census['rows'] and cell['bands_equal_census']):
            fail.append(f'sampled_minutes/{p} disagrees with protocol or census')
        c['sampled_by_role'][p] = cell
    if set(by_role) != set(PARTITIONS):
        fail.append('sampled_minutes roles changed')
    keys = [(r['match'], int(r['time_ms'])) for r in sampled]
    c['sampled_duplicate_keys'] = len(keys) - len(set(keys))
    test_rows = Counter(r['match'] for r in by_role['test'])
    c['test_rows_per_match_all_one'] = set(test_rows.values()) == {1}
    if c['sampled_duplicate_keys'] or not c['test_rows_per_match_all_one']:
        fail.append('sampled_minutes duplicate keys or test rows not one per match')
    gm, gt, gy = curves['match'], curves['time_ms'].astype(np.int64), curves['winner'].astype(np.int64)
    grid_keys = list(zip(gm.tolist(), gt.tolist()))
    c['grid_rows'] = len(grid_keys)
    c['grid_duplicate_keys'] = len(grid_keys) - len(set(grid_keys))
    c['grid_match_set_equals_test'] = set(gm.tolist()) == sets['test']
    winners = defaultdict(set)
    for m, y in zip(gm.tolist(), gy.tolist()):
        winners[m].add(y)
    c['grid_winner_constant_per_match'] = all(len(v) == 1 for v in winners.values())
    grid_set = Counter(grid_keys)
    c['test_sampled_keys_in_grid_once'] = all(grid_set[(r['match'], int(r['time_ms']))] == 1 for r in by_role['test'])
    c['grid_band_rows'] = {f'{lo}-{hi}': int(((gt >= lo * 60000) & (gt < hi * 60000)).sum()) for lo, hi in TIME_BANDS}
    c['grid_band_rows_equal_old_results'] = all(
        c['grid_band_rows'][b] == old_results['test_time_bands']['expanded'][b]['n'] for b in c['grid_band_rows'])
    if c['grid_duplicate_keys'] or not (c['grid_match_set_equals_test'] and c['grid_winner_constant_per_match']
                                        and c['test_sampled_keys_in_grid_once'] and c['grid_band_rows_equal_old_results']):
        fail.append('stored test grid inconsistent with protocol')
    return c, fail


def subset_for_smoke(proto, sampled, curves, n):
    keep = set()
    for p in PARTITIONS:
        keep.update(proto['splits'][p][:n])
    rows = [dict(r, row_index=i) for i, r in enumerate(sampled) if r['match'] in keep]
    mask = np.isin(curves['match'], sorted(keep & set(proto['splits']['test'])))
    return keep, rows, np.flatnonzero(mask)


def build_plan(proto, rows, curves, grid_index):
    times = defaultdict(set)
    for r in rows:
        times[r['match']].add(int(r['time_ms']))
    for m, t in zip(curves['match'][grid_index].tolist(), curves['time_ms'][grid_index].tolist()):
        times[m].add(int(t))
    order = [m for p in PARTITIONS for m in proto['splits'][p] if m in times]
    if len(order) != len(times):
        raise ValueError('planned matches outside V partitions')
    return [(m, sorted(times[m])) for m in order]


# ----------------------------------------------------------------------------- extraction
def run_extraction(run, plan, names, workers, chunk_size):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    chunk_dir = run.out / 'states' / 'chunks'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    tasks, reused = [], 0
    starts = list(range(0, len(plan), chunk_size))
    for c, s in enumerate(starts):
        items = plan[s:s + chunk_size]
        plan_sha = sha256_json([names, items])
        path = chunk_dir / f'chunk_{c:04d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z['plan_sha256']) != plan_sha or str(z['names_sha256']) != sha256_json(names):
                    raise ValueError(f'stale checkpoint {path.name}: plan changed; use a new output directory')
            reused += 1
            continue
        tasks.append(dict(chunk_id=c, path=str(path), items=items, names=names, plan_sha256=plan_sha))
    total, done, rows_done, errors = len(starts), reused, 0, {}
    run.stage('extraction', chunks_total=total, chunks_done=done, chunks_reused=reused, workers=workers)
    if tasks:
        os.environ['P1_WORKER'] = '1'
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
                futures = {ex.submit(extract_chunk, t): t['chunk_id'] for t in tasks}
                for fut in as_completed(futures):
                    try:
                        s = fut.result()
                    except Exception as exc:
                        errors[f'chunk_{futures[fut]:04d}'] = f'worker failure {type(exc).__name__}: {exc}'
                        continue
                    errors.update(s['errors'])
                    done += int(s['written'])
                    rows_done += s['rows']
                    run.stage('extraction', chunks_total=total, chunks_done=done, chunks_reused=reused,
                              rows_this_session=rows_done, match_errors=len(errors), last_chunk_seconds=s['seconds'])
        finally:
            os.environ['P1_WORKER'] = '0'
    if errors:
        write_json(run.out / 'errors.json', {'stage': 'extraction', 'match_or_chunk_errors': errors,
                                             'note': 'no state was skipped; failing chunks were not checkpointed'})
        raise RuntimeError(f'extraction failed for {len(errors)} matches/chunks; see errors.json')
    return total


# ----------------------------------------------------------------------------- assembly
def assemble(run, n_chunks, names, rows, curves, grid_index, plan):
    from gameplay.state_value_v2 import STATE_VERSION
    parts, per, raw_grids, souls = defaultdict(list), defaultdict(list), {}, {}
    for c in range(n_chunks):
        with np.load(run.out / 'states' / 'chunks' / f'chunk_{c:04d}.npz', allow_pickle=False) as z:
            if str(z['state_version']) != STATE_VERSION or str(z['names_sha256']) != sha256_json(names):
                raise ValueError(f'chunk {c} state version/schema mismatch')
            for k in ('match', 'time_ms', 'snapshot_ms', 'unassigned_soul_events', 'X'):
                parts[k].append(z[k])
            for k in ('match_ids', 'winner', 'terminal_ms', 'last_frame_ms', 'patch', 'reordered'):
                per[k].append(z[k])
            raw_grids.update(json.loads(str(z['raw_grid_json'])))
            souls.update(json.loads(str(z['soul_audit_json'])))
    A = {k: np.concatenate(v) for k, v in parts.items()}
    P = {k: np.concatenate(v) for k, v in per.items()}
    key = {}
    for i, k in enumerate(zip(A['match'].tolist(), A['time_ms'].tolist())):
        if k in key:
            raise ValueError(f'duplicate state key {k}')
        key[k] = i
    checks = {'planned_queries': sum(len(t) for _, t in plan), 'built_states': len(key),
              'planned_matches': len(plan), 'built_matches': int(len(P['match_ids']))}
    if checks['planned_queries'] != checks['built_states'] or checks['planned_matches'] != checks['built_matches']:
        raise ValueError(f'state count mismatch {checks}')
    winner = dict(zip(P['match_ids'].tolist(), P['winner'].astype(int).tolist()))
    s_idx = np.asarray([key[(r['match'], int(r['time_ms']))] for r in rows], dtype=np.int64)
    gm, gt = curves['match'][grid_index], curves['time_ms'][grid_index].astype(np.int64)
    g_idx = np.asarray([key[k] for k in zip(gm.tolist(), gt.tolist())], dtype=np.int64)
    sampled = dict(row_index=np.asarray([r['row_index'] for r in rows], dtype=np.int64),
                   match=A['match'][s_idx], role=np.asarray([r['role'] for r in rows], dtype='U10'),
                   time_ms=A['time_ms'][s_idx], snapshot_ms=A['snapshot_ms'][s_idx],
                   unassigned_soul_events=A['unassigned_soul_events'][s_idx],
                   winner=np.asarray([winner[r['match']] for r in rows], dtype=np.int64), X=A['X'][s_idx])
    grid = dict(grid_index=np.asarray(grid_index, dtype=np.int64), match=A['match'][g_idx], time_ms=A['time_ms'][g_idx],
                snapshot_ms=A['snapshot_ms'][g_idx], unassigned_soul_events=A['unassigned_soul_events'][g_idx],
                winner=np.asarray([winner[m] for m in gm.tolist()], dtype=np.int64), X=A['X'][g_idx])
    A = None
    checks['sampled_rows'] = int(len(s_idx))
    checks['sampled_keys_elementwise_equal'] = bool(
        sampled['match'].tolist() == [r['match'] for r in rows] and
        sampled['time_ms'].tolist() == [int(r['time_ms']) for r in rows])
    checks['grid_rows'] = int(len(g_idx))
    checks['grid_keys_elementwise_equal'] = bool(np.array_equal(grid['match'], gm) and np.array_equal(grid['time_ms'], gt))
    checks['grid_winner_mismatch_vs_stored'] = int((grid['winner'] != curves['winner'][grid_index]).sum())
    checks['states_finite'] = bool(np.isfinite(sampled['X']).all() and np.isfinite(grid['X']).all())
    checks['snapshot_not_after_query'] = bool((sampled['snapshot_ms'] <= sampled['time_ms']).all()
                                              and (grid['snapshot_ms'] <= grid['time_ms']).all())
    grid_times = defaultdict(list)
    for m, t in zip(gm.tolist(), gt.tolist()):
        grid_times[m].append(t)
    checks['test_grid_equals_raw_rebuilt_grid_mismatches'] = sum(sorted(v) != raw_grids[m] for m, v in grid_times.items())
    raw_sets = {m: set(v) for m, v in raw_grids.items()}
    checks['sampled_times_outside_raw_grid'] = sum(int(r['time_ms']) not in raw_sets[r['match']] for r in rows)
    bad = [k for k in ('sampled_keys_elementwise_equal', 'grid_keys_elementwise_equal', 'states_finite',
                       'snapshot_not_after_query') if not checks[k]]
    if bad or checks['grid_winner_mismatch_vs_stored']:
        raise ValueError(f'assembly checks failed: {bad} winner_mismatch={checks["grid_winner_mismatch_vs_stored"]}')
    meta = dict(names=np.asarray(names), state_version=np.asarray(STATE_VERSION))
    save_npz(run.out / 'states' / 'sampled_states_v2.npz', **sampled, **meta)
    save_npz(run.out / 'states' / 'test_grid_states_v2.npz', **grid, **meta)
    table = {m: {'winner_blue': int(w), 'terminal_ms': int(te), 'last_frame_ms': int(lf), 'patch': str(pa),
                 'participant_order_differs_from_role_slots': int(ro)}
             for m, w, te, lf, pa, ro in zip(P['match_ids'].tolist(), P['winner'], P['terminal_ms'],
                                             P['last_frame_ms'], P['patch'].tolist(), P['reordered'])}
    write_json(run.out / 'states' / 'match_table.json', table)
    return sampled, grid, table, souls, checks


def soul_summary(souls, proto):
    role_of = {m: p for p in PARTITIONS for m in proto['splits'][p]}
    six = {'AIR', 'EARTH', 'FIRE', 'WATER', 'HEXTECH', 'CHEMTECH'}
    owned, unassigned, raw = Counter(), Counter(), Counter()
    matches = defaultdict(set)
    for m, events in souls.items():
        for e in events:
            raw[json.dumps(e['fields'], sort_keys=True, ensure_ascii=False)] += 1
            if e['team'] in (100, 200):
                owned[e['element']] += 1
                matches['owned'].add(m)
            else:
                unassigned[f"teamId={e['team']}:{e['element']}"] += 1
                matches['unassigned'].add(m)
    recognized = {k for k in owned if k in six} | {k.split(':')[1] for k in unassigned if k.split(':')[1] in six}
    return {'owned_events_by_element': dict(owned), 'unassigned_events_by_team_element': dict(unassigned),
            'raw_field_combinations': dict(raw.most_common()),
            'owned_OTHER_or_ERROR_events': sum(v for k, v in owned.items() if k not in six),
            'unassigned_OTHER_or_ERROR_events': sum(v for k, v in unassigned.items() if k.split(':')[1] not in six),
            'kinds_present_and_recognized': sorted(recognized), 'all_six_kinds_recognized': recognized == six,
            'matches_with_owned_soul': len(matches['owned']), 'matches_with_unassigned_soul': len(matches['unassigned']),
            'matches_with_both': len(matches['owned'] & matches['unassigned']),
            'matches_by_partition': {k: dict(Counter(role_of[m] for m in v)) for k, v in matches.items()}}


# ----------------------------------------------------------------------------- training
def partition(sampled, p):
    mask = sampled['role'] == p
    return sampled['X'][mask], sampled['winner'][mask], sampled['match'][mask], sampled['time_ms'][mask]


def train_models(run, names):
    import joblib
    from train.state_value_experiment import logistic, metrics, select_value_regularization
    from train.temporal_winprob import feature_matrix, maymin_model, calibrated_variants
    from train.independent_winprob_v2 import (IndependentWinProbabilityV2, CHAMPION_COLUMNS, MODEL_VERSION,
                                               categorical_columns, load_adapter)
    from gameplay.state_value_v2 import STATE_VERSION
    with np.load(run.out / 'states' / 'sampled_states_v2.npz', allow_pickle=False) as z:
        if str(z['state_version']) != STATE_VERSION or z['names'].tolist() != names:
            raise ValueError('sampled state file schema mismatch')
        sampled = {k: z[k] for k in ('match', 'role', 'time_ms', 'winner', 'X')}
    fit, cal, sel = (partition(sampled, p) for p in ('fit', 'calibrate', 'select'))
    old_cv = read_json(OLD / 'expanded_training_cv.json')
    candidates, scores, chosen, info = {}, {}, {}, {}
    for family in FAMILIES:
        X, cols = feature_matrix(fit[0], names, family)
        cell = {'n_features': len(cols), 'feature_names': cols, 'fit_rows': int(len(fit[1])),
                'fit_matches': int(len(set(fit[2].tolist()))), 'final_fit_sample_weight': 'none (as original runner)'}
        if family == 'maymin':
            base = maymin_model().fit(X, fit[1])
            cell['regularization'] = 'penalty=None (as original maymin_model)'
        else:
            if categorical_columns(cols) != list(CHAMPION_COLUMNS) or 'snapshot_age_s' in cols:
                raise ValueError('expanded feature schema unsafe for frozen logistic() helper')
            run.log('expanded: grouped 3-fold CV over C candidates on fit only')
            C, cv = select_value_regularization(X, fit[1], fit[2], cols, candidates=C_CANDIDATES)
            same_folds = cv['folds'] == old_cv['folds']
            write_json(run.out / 'expanded_training_cv.json',
                       dict(cv, state_version=STATE_VERSION, folds_identical_to_original=same_folds,
                            original_chosen_C=old_cv['chosen_C'],
                            rule='unchanged: GroupKFold(3) on fit, candidates .001/.01/.1/1, match-weighted CV fits, '
                                 'minimum match-weighted log loss'))
            base = logistic(cols, C=C).fit(X, fit[1])
            ct = base.named_steps['preprocess']
            enc = {nm: [cols[i] for i in ix] for nm, _, ix in ct.transformers_ if nm in ('numeric', 'champions')}
            if (enc['champions'] != list(CHAMPION_COLUMNS) or any(k.endswith('champion_id') for k in enc['numeric'])
                    or type(ct.named_transformers_['champions']).__name__ != 'OneHotEncoder'
                    or len(enc['numeric']) + len(enc['champions']) != len(cols)):
                raise ValueError('fitted preprocessing does not treat participant champion IDs as categorical')
            cell.update(C=C, cv_folds_identical_to_original=same_folds, categorical_one_hot=enc['champions'],
                        n_numeric_scaled=len(enc['numeric']),
                        champion_categories_per_slot=[len(c) for c in ct.named_transformers_['champions'].categories_])
        run.log(f'{family}: base fitted; fitting raw/sigmoid/isotonic on calibrate')
        for method, frozen in calibrated_variants(family, names, base, cal[0], cal[1]).items():
            adapter = IndependentWinProbabilityV2(family, names, base, frozen.calibration, frozen.calibrator,
                                                  fit_record={'fit_partition': 'fit', 'calibration_partition': 'calibrate',
                                                              'selection_partition': 'select'})
            p = adapter.predict_matrix(sel[0], names, STATE_VERSION)
            if not np.array_equal(p, frozen.predict_proba(sel[0])[:, 1]):
                raise ValueError('adapter output differs from frozen SnapshotWinProbability')
            candidates[adapter.key], scores[adapter.key] = adapter, metrics(sel[1], p, sel[2])
        chosen[family] = min((k for k in candidates if k.startswith(family + '_')), key=lambda k: scores[k]['log_loss'])
        info[family] = cell
        run.log(f'{family}: select log loss ' + ', '.join(f"{k}={v['log_loss']:.5f}" for k, v in scores.items()
                                                          if k.startswith(family)) + f' -> {chosen[family]}')
    write_json(run.out / 'selection.json', {'chosen': chosen, 'scores': scores, 'rule': 'lowest select-partition '
               'match-weighted log loss per family; test never used', 'state_version': STATE_VERSION,
               'model_version': MODEL_VERSION})
    model_dir = run.out / 'models'
    (model_dir / 'candidates').mkdir(parents=True, exist_ok=True)
    model_ids, candidate_ids, sel_preds, models = {}, {}, {}, {}
    for key, adapter in candidates.items():
        path = model_dir / 'candidates' / f'{key}_v2.joblib'
        joblib.dump(adapter, path)
        candidate_ids[key] = sha256_file(path)
        sel_preds[key] = adapter.predict_matrix(sel[0], names, STATE_VERSION)
    for family, key in chosen.items():
        path = model_dir / f'{family}_model_v2.joblib'
        joblib.dump(candidates[key], path)
        model_ids[family] = sha256_file(path)
        models[family] = load_adapter(path)
        if not np.array_equal(models[family].predict_matrix(sel[0], names, STATE_VERSION), sel_preds[key]):
            raise ValueError('reloaded adapter predictions differ')
    save_npz(run.out / 'predictions' / 'select_partition_candidates_v2.npz', match=sel[2], time_ms=sel[3],
             winner=sel[1], model_version=np.asarray(MODEL_VERSION), state_version=np.asarray(STATE_VERSION), **sel_preds)
    write_json(model_dir / 'inference_adapter.json', {
        'class': 'train.independent_winprob_v2.IndependentWinProbabilityV2',
        'loader': 'train.independent_winprob_v2.load_adapter(path)',
        'module_sha256': sha256_file(WT / 'train' / 'independent_winprob_v2.py'),
        'model_version': MODEL_VERSION, 'state_version': STATE_VERSION, 'state_names': names,
        'chosen': chosen, 'model_sha256': model_ids, 'candidate_sha256': candidate_ids,
        'feature_names': {f: models[f].feature_names for f in FAMILIES},
        'guards': ['predict_states: StateV2 objects only; version and exact feature order via state_matrix',
                   'predict_matrix: requires declared state_version and exact names list',
                   'bare numpy arrays and legacy slotN_ schemas rejected before vectorisation'],
        'semantics': 'P(final Blue win | current observed state); associational, not causal'})
    return models, chosen, scores, model_ids, candidate_ids, info


# ----------------------------------------------------------------------------- evaluation helpers
SIX_SOULS = ('AIR', 'EARTH', 'FIRE', 'WATER', 'HEXTECH', 'CHEMTECH')


def match_weights(groups):
    _, ix, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / counts[ix]
    return w / w.mean()


def ece_from(y, p, w, bins=10):
    """Identical to run_temporal_winprob_v3_buckets.ece_from (match-weighted, fixed bins)."""
    edges = np.linspace(0., 1., bins + 1)
    total, out = w.sum(), 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi + (1e-10 if hi >= 1. else 0.))
        if mask.any():
            ww = w[mask]
            out += float(ww.sum() / total) * abs(
                float(np.average(y[mask], weights=ww)) - float(np.average(p[mask], weights=ww)))
    return float(out)


def calibration_line(y, p, w):
    """Weighted logistic recalibration: intercept with slope fixed at 1, and slope (Van Calster et al. 2019)."""
    from scipy.special import expit
    q = np.clip(p, 1e-10, 1 - 1e-10)
    lp = np.log(q / (1 - q))

    def newton(Z, offset):
        beta = np.zeros(Z.shape[1])
        for _ in range(200):
            mu = expit(offset + Z @ beta)
            grad = Z.T @ (w * (y - mu))
            hess = Z.T @ (Z * (w * mu * (1 - mu))[:, None])
            step = np.linalg.solve(hess + 1e-12 * np.eye(len(beta)), grad)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-10:
                return beta, True
        return beta, False

    a, ok_a = newton(np.ones((len(y), 1)), lp)
    b, ok_b = newton(np.column_stack([np.ones(len(y)), lp]), np.zeros(len(y)))
    return {'calibration_intercept': float(a[0]), 'calibration_slope': float(b[1]),
            'slope_model_intercept': float(b[0]), 'calibration_line_converged': bool(ok_a and ok_b)}


def evaluate(y, p, groups):
    from train.state_value_experiment import metrics
    y = np.asarray(y).astype(np.int64)
    cell = metrics(y, p, groups)
    w = match_weights(groups)
    cell['ece'] = ece_from(y, p, w)
    if len(set(y.tolist())) == 2:
        cell.update(calibration_line(y.astype(float), p, w))
    return cell


def row_losses(y, p):
    eps = np.finfo(np.float64).eps
    q = np.clip(p, eps, 1 - eps)
    return (p - y) ** 2, -(y * np.log(q) + (1 - y) * np.log(1 - q))


def paired_bootstrap(y, p_new, p_old, groups, n_boot, seed):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(float)
    unique, inverse = np.unique(groups, return_inverse=True)
    base = match_weights(groups)
    if len(set(y.tolist())) < 2 or len(unique) < 20:
        return {'status': 'insufficient', 'matches': int(len(unique)), 'rows': int(len(y))}
    bn, ln = row_losses(y, p_new)
    bo, lo = row_losses(y, p_old)

    def stats(w):
        k = w > 0
        return (float(roc_auc_score(y[k], p_new[k], sample_weight=w[k]) - roc_auc_score(y[k], p_old[k], sample_weight=w[k])),
                float(np.average(bn - bo, weights=w)), float(np.average(ln - lo, weights=w)))

    point = stats(base)
    rng = np.random.default_rng(seed)
    draws, skipped = [], 0
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        w = base * mult[inverse]
        if len(set(y[w > 0].tolist())) < 2:
            skipped += 1
            continue
        draws.append(stats(w))
    D = np.asarray(draws)
    out = {'status': 'ok', 'difference': 'V2 minus V1 (stored)', 'matches': int(len(unique)), 'rows': int(len(y)),
           'n_boot_requested': n_boot, 'n_boot_valid': int(len(D)), 'skipped_single_class': skipped, 'seed': seed,
           'better_direction': {'auc': 'positive', 'brier': 'negative', 'log_loss': 'negative'},
           'uncertainty_scope': 'test matches resampled with replacement (equal total weight per match); fitted '
                                'models, calibrators and selections held fixed, so training/selection variability '
                                'is not included'}
    for j, name in enumerate(('auc', 'brier', 'log_loss')):
        col = D[:, j]
        better = col > 0 if name == 'auc' else col < 0
        out[name] = {'point': point[j], 'ci95': np.quantile(col, [.025, .975]).tolist(),
                     'bootstrap_se': float(col.std(ddof=1)), 'fraction_replicates_V2_better': float(better.mean())}
    return out


def objective_strata(X, names, unassigned):
    ix = {n: i for i, n in enumerate(names)}

    def col(n):
        return X[:, ix[n]]

    def either(n):
        return (col('blue_' + n) > 0) | (col('red_' + n) > 0)

    owned = either('soul_event_recorded')
    unassigned = np.asarray(unassigned) > 0
    strata = {
        'no_baron_elder_or_soul_history': ~(either('baron_ever') | either('elder_ever') | owned | unassigned),
        'dragons_any': either('dragons'),
        'dragon_count_diff_ge2': np.abs(col('blue_dragons') - col('red_dragons')) >= 2,
        'herald_or_grubs_any': either('herald_ever') | either('horde_ever'),
        'atakhan_any': either('atakhan_ever'),
        'inhibitor_kill_any': either('inhibitor_kills'),
        'baron_ever_any': either('baron_ever'),
        'baron_acquired_last_180s_any': either('baron_acquired_last_180s'),
        'elder_ever_any': either('elder_ever'),
        'elder_acquired_last_180s_any': either('elder_acquired_last_180s'),
        'owned_soul_any': owned,
        'unassigned_soul_teamId0_any_DIAGNOSTIC': unassigned,
        'unassigned_soul_teamId0_without_owned_soul_DIAGNOSTIC': unassigned & ~owned,
        'unknown_objective_team_count_positive': col('unknown_objective_team_count') > 0,
    }
    for d in SIX_SOULS:
        strata[f'owned_soul_{d}'] = either(f'soul_{d}')
    return strata


def agreement(p_new, p_old):
    d = np.abs(p_new - p_old)
    return {'pearson_r': float(np.corrcoef(p_new, p_old)[0, 1]), 'mean_abs_diff': float(d.mean()),
            'abs_diff_quantiles_50_90_99': np.quantile(d, [.5, .9, .99]).tolist(),
            'fraction_abs_diff_gt_0.05': float((d > .05).mean()), 'fraction_abs_diff_gt_0.10': float((d > .10).mean())}


# ----------------------------------------------------------------------------- evaluation
def band_label(lo, hi):
    return f'{lo}-{hi}' if hi < 1000 else f'{lo}+'


def compare_cell(y, groups, v2, v1, boot_family=None, n_boot=N_BOOT_STRATA):
    cell = {'rows': int(len(y)), 'matches': int(len(set(groups.tolist()))),
            'positive_rate_match_weighted': float(np.average(y, weights=match_weights(groups))) if len(y) else None}
    if not len(y):
        cell['status'] = 'empty'
        return cell
    for f in FAMILIES:
        cell[f] = {'V2': evaluate(y, v2[f], groups), 'V1_stored': evaluate(y, v1[f], groups)}
    if boot_family:
        cell[boot_family]['paired_bootstrap_V2_minus_V1'] = paired_bootstrap(y, v2[boot_family], v1[boot_family],
                                                                             groups, n_boot, BOOT_SEED)
    return cell


def evaluate_all(run, models, names, curves, grid_index, chosen, model_ids):
    from gameplay.state_value_v2 import STATE_VERSION
    from train.independent_winprob_v2 import MODEL_VERSION
    from train.state_value_experiment import metrics
    with np.load(run.out / 'states' / 'sampled_states_v2.npz', allow_pickle=False) as z:
        sampled = {k: z[k] for k in ('match', 'role', 'time_ms', 'winner', 'X')}
    with np.load(run.out / 'states' / 'test_grid_states_v2.npz', allow_pickle=False) as z:
        grid = {k: z[k] for k in ('match', 'time_ms', 'winner', 'unassigned_soul_events', 'X', 'grid_index')}
    key_to_curve = {k: i for i, k in enumerate(zip(curves['match'].tolist(), curves['time_ms'].tolist()))}
    fit_y = sampled['winner'][sampled['role'] == 'fit']
    t = sampled['role'] == 'test'
    tX, ty, tg, tt = sampled['X'][t], sampled['winner'][t], sampled['match'][t], sampled['time_ms'][t]
    ci = np.asarray([key_to_curve[k] for k in zip(tg.tolist(), tt.tolist())], dtype=np.int64)
    checks = {'test_one_minute_rows': int(len(ty)), 'test_one_minute_keys_matched_to_stored_v1': int(len(ci)),
              'test_one_minute_winner_mismatch_vs_stored': int((curves['winner'][ci] != ty).sum())}
    v1 = {f: curves[f][ci].astype(float) for f in FAMILIES}
    v2 = {f: models[f].predict_matrix(tX, names, STATE_VERSION) for f in FAMILIES}
    one = {}
    for f in FAMILIES:
        one[f] = {'V2': evaluate(ty, v2[f], tg), 'V1_stored': evaluate(ty, v1[f], tg),
                  'paired_bootstrap_V2_minus_V1': paired_bootstrap(ty, v2[f], v1[f], tg, N_BOOT_PRIMARY, BOOT_SEED),
                  'prediction_agreement': agreement(v2[f], v1[f])}
        bs = one[f]['paired_bootstrap_V2_minus_V1']
        if bs['status'] == 'ok':
            checks[f'{f}_bootstrap_point_equals_metric_difference'] = all(
                abs(bs[k]['point'] - (one[f]['V2'][k] - one[f]['V1_stored'][k])) < 1e-9 for k in ('auc', 'brier', 'log_loss'))
    one['training_prior'] = metrics(ty, np.repeat(fit_y.mean(), len(ty)), tg)
    run.log('one-minute test: ' + '; '.join(
        f"{f} V2 auc={one[f]['V2']['auc']:.4f} brier={one[f]['V2']['brier']:.4f} ll={one[f]['V2']['log_loss']:.4f} | "
        f"V1 auc={one[f]['V1_stored']['auc']:.4f} brier={one[f]['V1_stored']['brier']:.4f} ll={one[f]['V1_stored']['log_loss']:.4f}"
        for f in FAMILIES))
    pred_dir = run.out / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    labels = {'v2_state_version': STATE_VERSION, 'v2_model_version': MODEL_VERSION,
              'v1_protocol': 'independent_wp_v3_bucket_sampling', 'v1_state_version': 'objective_history_v1'}
    with (pred_dir / 'test_one_minute_v1_v2.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['match', 'time_ms', 'winner_blue', 'v2_state_version', 'v2_model_version', 'v2_maymin_' + chosen['maymin'],
                    'v2_expanded_' + chosen['expanded'], 'v1_protocol', 'v1_state_version', 'v1_maymin_stored', 'v1_expanded_stored'])
        for i in range(len(ty)):
            w.writerow([tg[i], int(tt[i]), int(ty[i]), STATE_VERSION, MODEL_VERSION, repr(float(v2['maymin'][i])),
                        repr(float(v2['expanded'][i])), labels['v1_protocol'], labels['v1_state_version'],
                        repr(float(v1['maymin'][i])), repr(float(v1['expanded'][i]))])
    meta_arrays = {k: np.asarray(v) for k, v in labels.items()}
    meta_arrays.update({f'v2_{f}_chosen': np.asarray(chosen[f]) for f in FAMILIES})
    meta_arrays.update({f'v2_{f}_model_sha256': np.asarray(model_ids[f]) for f in FAMILIES})
    save_npz(pred_dir / 'test_one_minute_v1_v2.npz', match=tg, time_ms=tt, winner_blue=ty,
             **{f'v2_{f}': v2[f] for f in FAMILIES}, **{f'v1_{f}_stored': v1[f] for f in FAMILIES}, **meta_arrays)

    gX, gy, gm, gt, gu = grid['X'], grid['winner'], grid['match'], grid['time_ms'], grid['unassigned_soul_events']
    checks['grid_rows'] = int(len(gy))
    checks['grid_keys_equal_stored'] = bool(np.array_equal(gm, curves['match'][grid_index]) and
                                            np.array_equal(gt, curves['time_ms'][grid_index]) and
                                            np.array_equal(grid['grid_index'], grid_index))
    v1g = {f: curves[f][grid_index].astype(float) for f in FAMILIES}
    v2g = {f: models[f].predict_matrix(gX, names, STATE_VERSION) for f in FAMILIES}
    checks['predictions_finite_in_unit_interval'] = bool(all(
        np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() for a in [*v2.values(), *v2g.values()]))
    save_npz(pred_dir / 'test_grid_v1_v2.npz', match=gm, time_ms=gt, winner_blue=gy, grid_index=grid_index,
             **{f'v2_{f}': v2g[f] for f in FAMILIES}, **{f'v1_{f}_stored': v1g[f] for f in FAMILIES}, **meta_arrays)
    run.stage('evaluation', part='time_bands')
    bands = {'all_grid': compare_cell(gy, gm, v2g, v1g)}
    for lo, hi in TIME_BANDS:
        mask = (gt >= lo * 60000) & (gt < hi * 60000)
        bands[band_label(lo, hi)] = compare_cell(gy[mask], gm[mask], {f: v2g[f][mask] for f in FAMILIES},
                                                 {f: v1g[f][mask] for f in FAMILIES}, boot_family='expanded')
    run.stage('evaluation', part='objective_strata')
    strata = {}
    for name, mask in objective_strata(gX, names, gu).items():
        cell = compare_cell(gy[mask], gm[mask], {f: v2g[f][mask] for f in FAMILIES},
                            {f: v1g[f][mask] for f in FAMILIES}, boot_family='expanded' if mask.sum() else None)
        cell['rows_by_time_band'] = {band_label(lo, hi): int((mask & (gt >= lo * 60000) & (gt < hi * 60000)).sum())
                                     for lo, hi in TIME_BANDS}
        strata[name] = cell
    return {'test_one_minute': one, 'test_grid_time_bands': bands, 'test_grid_objective_strata': strata}, checks


# ----------------------------------------------------------------------------- audit + validation helpers
def _rank(m):
    return hashlib.sha256(f'p1audit:{m}'.encode()).hexdigest()


def run_audit(run, proto, rows, table, souls, keep, names, v1_names, per_partition, workers):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    times = defaultdict(set)
    for r in rows:
        times[r['match']].add(int(r['time_ms']))
    chosen = []
    for p in PARTITIONS:
        chosen.extend(sorted((m for m in proto['splits'][p] if m in keep), key=_rank)[:per_partition])
    owned = sorted((m for m, ev in souls.items() if any(e['team'] in (100, 200) for e in ev)), key=_rank)
    unassigned = sorted((m for m, ev in souls.items() if any(e['team'] not in (100, 200) for e in ev)), key=_rank)
    extra = [m for m in owned if m not in chosen][:20] + [m for m in unassigned if m not in chosen][:20]
    matches = list(dict.fromkeys(chosen + extra))
    for m in matches:
        for e in souls.get(m, []):
            for t in (e['ts'], (e['ts'] // 60000 + 1) * 60000):
                if 0 <= t < table[m]['terminal_ms'] and t <= table[m]['last_frame_ms']:
                    times[m].add(int(t))
    items = [(m, sorted(times[m])) for m in matches]
    tasks = [dict(items=items[i:i + 10], names=names, v1_names=v1_names) for i in range(0, len(items), 10)]
    records = []
    os.environ['P1_WORKER'] = '1'
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
            for out in ex.map(audit_chunk, tasks):
                records.extend(out)
                run.stage('audit', matches_done=len(records), matches_total=len(items))
    finally:
        os.environ['P1_WORKER'] = '0'
    role_of = {m: p for p in PARTITIONS for m in proto['splits'][p]}
    summary = {
        'matches': len(records), 'queries': sum(r['queries'] for r in records),
        'matches_by_partition': dict(Counter(role_of[r['match']] for r in records)),
        'selection': f'{per_partition} per partition by sha256 rank, plus up to 20 owned-soul and 20 teamId-0-soul '
                     'matches; soul timestamps and the next minute added as extra queries',
        'matches_with_owned_soul': sum(r['match'] in set(owned) for r in records),
        'matches_with_unassigned_soul': sum(r['match'] in set(unassigned) for r in records),
        'role_slot_variant_state_checks': sum(r.get('role_variants_checked', 0) * r['queries'] for r in records),
        'role_slot_failures': sum(any(f.startswith('role_slots') for f in r['failures']) for r in records),
        'future_variant_state_checks': sum(r.get('future_variants_checked', 0) for r in records),
        'future_invariance_failures': sum(any(f.startswith('future') for f in r['failures']) for r in records),
        'v1_loader_fidelity_matches_equal': sum(bool(r.get('v1_recompute_equal')) for r in records),
        'v1_compared_queries': sum(r.get('v1_compared_queries', 0) for r in records),
        'old_dataset_winner_mismatches': sum(r.get('old_dataset_winner_equal') is False for r in records),
        'matches_with_any_failure': sum(bool(r['failures']) for r in records),
        'failures': [r for r in records if r['failures']]}
    write_json(run.out / 'states' / 'audit_real_cache.json', {'summary': summary, 'records': records})
    return summary


def old_model_reproduction(curves, grid_index):
    import joblib
    old_ids = read_json(OLD / 'results.json')['model_ids']
    out = {'model_sha256_equal_results_model_ids': {f: sha256_file(OLD / f'{f}_model.joblib') == old_ids[f] for f in FAMILIES}}
    models = {f: joblib.load(OLD / f'{f}_model.joblib') for f in FAMILIES}
    gm, gt = curves['match'][grid_index], curves['time_ms'][grid_index]
    positions = defaultdict(list)
    for i, m in enumerate(gm.tolist()):
        positions[m].append(i)
    stored = np.empty((len(grid_index), len(models['expanded'].state_names)), dtype=np.float32)
    for m, pos in positions.items():
        with np.load(OLD_DATASET / 'matches' / f'{m}.npz', allow_pickle=False) as z:
            where = {int(t): i for i, t in enumerate(z['value_times'])}
            stored[pos] = z['value_states'][[where[int(t)] for t in gt[pos]]]
    ct = models['expanded'].base.named_steps['preprocess']
    _, v1_cols = __import__('train.temporal_winprob', fromlist=['feature_matrix']).feature_matrix(
        stored[:1], models['expanded'].state_names, 'expanded')
    out['v1_expanded_champion_handling'] = {
        'one_hot_columns': [v1_cols[i] for i in ct.transformers_[1][2]],
        'encoder': type(ct.named_transformers_['champions']).__name__,
        'note': 'V1 champion IDs are one-hot per role slot; the V1 object vectorises bare arrays by position'}
    try:
        models['expanded'].predict_proba(stored[:2].astype(np.float64))
        out['v1_object_accepts_any_same_width_array_without_schema_check'] = True
    except Exception:
        out['v1_object_accepts_any_same_width_array_without_schema_check'] = False
    for f, model in models.items():
        p = model.predict_proba(stored)[:, 1]
        out[f] = {'rows': int(len(p)), 'max_abs_diff_old_model_on_stored_v1_states_vs_stored_curve':
                  float(np.max(np.abs(p - curves[f][grid_index]))),
                  'object': f'{type(model).__module__}.{type(model).__name__}', 'calibration': model.calibration,
                  'state_names_example': model.state_names[4], 'label': 'V1 (objective_history_v1, role-slot order)'}
    return out, stored


def v1_v2_state_census(stored, v2X, names, gm, table, unassigned):
    v2 = v2X.astype(np.float32)
    diff = v2 != stored
    part = [i for i, n in enumerate(names) if n.startswith('participant_slot')]
    soul = [i for i, n in enumerate(names) if '_soul_' in n]
    u = names.index('unknown_objective_team_count')
    other = [i for i in range(len(names)) if i not in set(part) | set(soul) | {u}]
    reordered = np.asarray([table[m]['participant_order_differs_from_role_slots'] for m in gm.tolist()])
    idx = {n: i for i, n in enumerate(names)}
    specific = np.zeros(len(gm), dtype=bool)
    for d in SIX_SOULS:
        specific |= (v2X[:, idx[f'blue_soul_{d}']] > 0) | (v2X[:, idx[f'red_soul_{d}']] > 0)
    v1_other = (stored[:, idx['blue_soul_OTHER']] > 0) | (stored[:, idx['red_soul_OTHER']] > 0)
    return {
        'rows': int(len(gm)), 'matches': int(len(set(gm.tolist()))),
        'matches_participant_order_differs_from_role_slots': int(len({m for m, r in zip(gm.tolist(), reordered) if r == 1})),
        'matches_role_slots_invalid': int(len({m for m, r in zip(gm.tolist(), reordered) if r == -1})),
        'rows_participant_block_differs': int(diff[:, part].any(axis=1).sum()),
        'rows_participant_block_differs_in_matches_not_reordered': int((diff[:, part].any(axis=1) & (reordered == 0)).sum()),
        'rows_soul_columns_differ': int(diff[:, soul].any(axis=1).sum()),
        'rows_v1_soul_OTHER_set': int(v1_other.sum()),
        'rows_v2_specific_owned_soul_set': int(specific.sum()),
        'rows_v2_soul_OTHER_set': int(((v2X[:, idx['blue_soul_OTHER']] > 0) | (v2X[:, idx['red_soul_OTHER']] > 0)).sum()),
        'rows_unknown_objective_team_count_differs': int(diff[:, u].sum()),
        'unknown_count_v1_minus_v2_equals_unassigned_soul_events': bool(np.array_equal(
            stored[:, u].astype(np.int64) - v2X[:, u].astype(np.int64), np.asarray(unassigned, dtype=np.int64))),
        'rows_other_columns_differ': int(diff[:, other].any(axis=1).sum()),
        'other_columns_with_differences': [names[i] for i in other if diff[:, i].any()]}


def adapter_guard_tests(model, names, v1_names, X, times):
    from gameplay.state_value import State
    from gameplay.state_value_v2 import StateV2, STATE_VERSION
    from train.independent_winprob_v2 import IndependentWinProbabilityV2
    states = [StateV2(dict(zip(names, map(float, row))), int(t), int(t), STATE_VERSION, 0) for row, t in zip(X, times)]

    def raises(fn, exc):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    return {
        'state_objects_equal_matrix_path': bool(np.array_equal(model.predict_states(states),
                                                               model.predict_matrix(X, names, STATE_VERSION))),
        'rejects_legacy_State_objects': raises(lambda: model.predict_states(
            [State(dict(zip(v1_names, map(float, row))), 0, 0) for row in X]), ValueError),
        'rejects_reordered_StateV2_features': raises(lambda: model.predict_states(
            [StateV2(dict(reversed(list(s.values.items()))), s.query_ms, s.snapshot_ms, STATE_VERSION, 0) for s in states]), ValueError),
        'rejects_wrong_declared_version': raises(lambda: model.predict_matrix(X, names, 'objective_history_v1'), ValueError),
        'rejects_v1_names': raises(lambda: model.predict_matrix(X, v1_names, STATE_VERSION), ValueError),
        'rejects_bare_array_predict_proba': raises(lambda: model.predict_proba(X), TypeError),
        'rejects_v1_schema_at_construction': raises(lambda: IndependentWinProbabilityV2('expanded', v1_names, model.base), ValueError),
        'rejects_nonfinite_states': raises(lambda: model.predict_matrix(np.full_like(X, np.nan), names, STATE_VERSION), ValueError)}


# ----------------------------------------------------------------------------- pipeline
def execute(run, a):
    import joblib
    import scipy
    import sklearn
    from gameplay.state_value_v2 import STATE_VERSION
    from train.independent_winprob_v2 import MODEL_VERSION, CHAMPION_COLUMNS, check_v2_names
    if (run.out / 'results.json').exists():
        raise ValueError('completed results already exist; use a new output directory')
    if (run.out / 'errors.json').exists():
        stale = run.out / f'errors_previous_attempt_{time.strftime("%Y%m%d_%H%M%S")}.json'
        (run.out / 'errors.json').replace(stale)
        run.log(f'archived previous errors.json as {stale.name}')
    run.stage('preflight')
    hashes_before = legacy_hashes()
    write_json(run.out / 'hashes_before.json', hashes_before)
    new_hashes = {k: sha256_file(p) for k, p in NEW_SOURCES.items()}
    snap = run.out / 'source_snapshot'
    for k, p in NEW_SOURCES.items():
        (snap / k).parent.mkdir(parents=True, exist_ok=True)
        (snap / k).write_bytes(p.read_bytes())
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                           f'--basetemp={run.out / "pytest_basetemp"}', str(ROOT / 'tests' / 'test_state_value_v2_contract.py')],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
    contract = {'command': 'python -m pytest -q -p no:cacheprovider tests/test_state_value_v2_contract.py',
                'returncode': proc.returncode, 'summary': lines[-1] if lines else ''}
    run.log(f'contract tests: {contract}')
    if proc.returncode != 0:
        raise RuntimeError('state_value_v2 contract tests failed')

    run.stage('protocol_checks')
    proto, sampled_rows, curves, old_results = load_protocol_inputs()
    pchecks, pfail = protocol_checks(proto, sampled_rows, curves, old_results)
    if pfail:
        write_json(run.out / 'errors.json', {'stage': 'protocol_checks', 'failures': pfail, 'checks': pchecks})
        raise RuntimeError(f'protocol checks failed: {pfail}')
    v1_names, names = expected_v2_names()
    check_v2_names(names)
    forbidden = [n for n in names if any(tok in n.lower() for tok in
                                         ('winner', 'winning', 'game_end', 'duration', 'terminal', 'outcome'))]
    if forbidden or 'snapshot_age_s' not in names:
        raise ValueError(f'unexpected state names: {forbidden}')
    if a.smoke:
        keep, rows, grid_index = subset_for_smoke(proto, sampled_rows, curves, a.smoke)
    else:
        keep = set().union(*(set(proto['splits'][p]) for p in PARTITIONS))
        rows = [dict(r, row_index=i) for i, r in enumerate(sampled_rows)]
        grid_index = np.arange(len(curves['match']), dtype=np.int64)
    plan = build_plan(proto, rows, curves, grid_index)
    chunk_size = a.chunk_size or (5 if a.smoke else 100)
    protocol = {
        'version': 'independent_v2_participant_order_p1', 'state_version': STATE_VERSION, 'model_version': MODEL_VERSION,
        'smoke': bool(a.smoke), 'smoke_matches_per_partition': a.smoke or None,
        'derived_from': 'worktrees/engagement-state-value/outputs/temporal_winprob_v3_buckets (independent_wp_v3_bucket_sampling)',
        'single_change': 'state representation objective_history_v1 -> objective_history_v2_participant_order; '
                         'splits, queries, model families, C rule, calibration variants and selection rule unchanged',
        'state_schema': {
            'n_state_names': len(names), 'names_sha256': sha256_json(names), 'names': names,
            'derivation': 'v1 schema.json state_names with slotN_ renamed participant_slotN_; same order',
            'builder': 'gameplay.state_value_v2.StateBuilder, validated with state_matrix',
            'participant_order': 'team 100 then 200, then numeric participantId; role_slots never used',
            'node_feature_names': 'core.config.NODE_FEATURE_NAMES',
            'raw_reader': 'data.cache_io.load_match_cache (read-only .npz/.meta.json/.events.json, v3.3 preset)',
            'target': 'final Blue winner from GAME_END via gameplay.state_value.final_outcome; never a feature',
            'excluded_value_features': ['snapshot_age_s'], 'snapshot_age_s': 'kept in state files as audit metadata only',
            'categorical_features': list(CHAMPION_COLUMNS),
            'soul_teamId_0': 'not ownership; excluded from unknown_objective_team_count; exposed only as '
                             'StateV2.unassigned_soul_events metadata (diagnostic strata)'},
        'model_schema': {
            'families': {'maymin': 'legacy 7-feature Maymin family (time, per-team kills/towers/monsters), '
                                   'StandardScaler + unpenalised logistic regression',
                         'expanded': 'all state features except snapshot_age_s; numeric median-impute + scale, '
                                     'participant champion IDs one-hot; L2 logistic (liblinear)'},
            'primary': 'expanded', 'C_candidates': list(C_CANDIDATES),
            'C_rule': 'GroupKFold(3) on fit only (match-weighted CV fits), minimum match-weighted log loss; '
                      'original run used the same rule (chosen C 0.001)',
            'final_base_fit': 'fit partition only, unweighted (as original runner)',
            'calibrators': 'raw / sigmoid (logit Platt, C=1e6) / isotonic, fitted on calibrate only',
            'selection': 'lowest select-partition match-weighted log loss per family; test never selects'},
        'splits': {p: proto['splits'][p] for p in (*PARTITIONS, 'engagement')},
        'sources': {'protocol.json': sha256_file(OLD / 'protocol.json'),
                    'sampled_minutes.json': sha256_file(OLD / 'sampled_minutes.json'),
                    'independent_time_curves.npz': sha256_file(OLD / 'independent_time_curves.npz'),
                    'results.json': sha256_file(OLD / 'results.json'),
                    'expanded_training_cv.json': sha256_file(OLD / 'expanded_training_cv.json'),
                    'state_value_v2_fix/rows.csv': sha256_file(V2FIX / 'rows.csv'),
                    'state_value_main_50k_eval/match_splits.json': sha256_file(OLD_EVAL / 'match_splits.json'),
                    'state_value_main_50k/schema.json': sha256_file(OLD_DATASET / 'schema.json')},
        'engagement_exclusions': {'sources': pchecks['engagement_sources'],
                                  'overlap_with_V_partitions': pchecks['engagement_overlap_with_V'],
                                  'note': 'state_value_v2_fix/states.npz is the engagement sample and is never read'},
        'protocol_checks': pchecks,
        'plan': {'matches': len(plan), 'queries': sum(len(t) for _, t in plan), 'sampled_rows': len(rows),
                 'grid_rows': int(len(grid_index)), 'chunk_size': chunk_size},
        'settings': {'workers': a.workers, 'blas_threads_main': 4, 'blas_threads_worker': 1, 'gpu': 'disabled',
                     'n_boot_primary': N_BOOT_PRIMARY, 'n_boot_strata': N_BOOT_STRATA, 'boot_seed': BOOT_SEED,
                     'time_bands_minutes': [list(b) for b in TIME_BANDS], 'audit_per_partition': a.audit_per_partition},
        'source_hashes': {'legacy': hashes_before['legacy_sources'], 'new': new_hashes},
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                     'scipy': scipy.__version__, 'joblib': joblib.__version__},
        'data_use': 'same 50k exploratory corpus (patches 15.14-15.16); not an untouched external test; '
                    '15.16 and 26.13 were examined previously'}
    write_json(run.out / 'protocol.json', protocol)

    n_chunks = run_extraction(run, plan, names, a.workers, chunk_size)
    run.stage('assembly')
    sampled, grid, table, souls, achecks = assemble(run, n_chunks, names, rows, curves, grid_index, plan)
    souls_sum = soul_summary(souls, proto)
    write_json(run.out / 'states' / 'soul_event_audit.json', {'summary': souls_sum, 'events_by_match': souls})
    trained_sets = {p: set(sampled['match'][sampled['role'] == p].tolist()) for p in PARTITIONS}
    built_by_partition = {p: {'rows': int((sampled['role'] == p).sum()), 'matches': len(trained_sets[p])} for p in PARTITIONS}
    leakage_actual = {'overlaps_between_state_partitions': overlap_table(trained_sets),
                      'engagement_overlap': {k: {p: len(trained_sets[p] & set(v)) for p in PARTITIONS}
                                             for k, v in (('protocol_engagement', proto['splits']['engagement']),)},
                      'partition_sets_equal_protocol_subset': all(
                          trained_sets[p] == set(proto['splits'][p]) & keep for p in PARTITIONS)}
    grid_u = grid['unassigned_soul_events']
    grid_X_small, grid_t_small = grid['X'][:5].copy(), grid['time_ms'][:5].copy()
    sampled = None
    run.stage('audit')
    audit = run_audit(run, proto, rows, table, souls, keep, names, v1_names, a.audit_per_partition, a.workers)

    run.stage('v1_reproduction')
    reproduction, stored_v1 = old_model_reproduction(curves, grid_index)
    census = v1_v2_state_census(stored_v1, grid['X'], names, grid['match'], table, grid_u)
    stored_v1 = grid = None

    run.stage('training')
    models, chosen, scores, model_ids, candidate_ids, train_info = train_models(run, names)
    run.stage('evaluation', part='test_one_minute')
    evaluation, echecks = evaluate_all(run, models, names, curves, grid_index, chosen, model_ids)

    run.stage('validation')
    guards = adapter_guard_tests(models['expanded'], names, v1_names, grid_X_small, grid_t_small)
    reported = {}
    if not a.smoke:
        one, bands = evaluation['test_one_minute'], evaluation['test_grid_time_bands']
        keys = ('auc', 'brier', 'log_loss', 'ece')
        reported = {f: {'one_minute': all(abs(one[f]['V1_stored'][k] - old_results['test_one_minute'][f][k]) < 1e-9 for k in keys),
                        'time_bands': all(abs(bands[band_label(lo, hi)][f]['V1_stored'][k] -
                                              old_results['test_time_bands'][f][f'{lo}-{hi}'][k]) < 1e-9
                                          for lo, hi in TIME_BANDS for k in keys)} for f in FAMILIES}
    hashes_after = legacy_hashes()
    write_json(run.out / 'hashes_after.json', hashes_after)
    changed = sorted(f'{group}/{k}' for group in hashes_before for k in set(hashes_before[group]) | set(hashes_after[group])
                     if hashes_before[group].get(k) != hashes_after[group].get(k))
    old_proto_hashes = proto['source_hashes']
    v2fix_record = read_json(V2FIX / 'results.json').get('module_sha256')
    checks = {
        'contract_tests_pass': contract['returncode'] == 0,
        'protocol_split_and_engagement_leakage_zero': not any(pchecks['protocol_overlaps'].values()) and not any(
            n for d in pchecks['engagement_overlap_with_V'].values() for n in d.values()),
        'state_partitions_leakage_zero': not any(leakage_actual['overlaps_between_state_partitions'].values()) and not any(
            n for d in leakage_actual['engagement_overlap'].values() for n in d.values()),
        'partition_sets_equal_protocol': leakage_actual['partition_sets_equal_protocol_subset'],
        'sampled_query_membership_identical': achecks['sampled_keys_elementwise_equal'] and all(
            c['match_set_equals_protocol_split'] and c['rows'] == c['census_rows'] for c in pchecks['sampled_by_role'].values()),
        'test_grid_keys_identical': achecks['grid_keys_elementwise_equal'] and echecks['grid_keys_equal_stored'],
        'winner_matches_stored': achecks['grid_winner_mismatch_vs_stored'] == 0 and echecks['test_one_minute_winner_mismatch_vs_stored'] == 0
                                 and audit['old_dataset_winner_mismatches'] == 0,
        'states_finite': achecks['states_finite'],
        'predictions_finite_in_unit_interval': echecks['predictions_finite_in_unit_interval'],
        'future_invariance_no_failures': audit['future_invariance_failures'] == 0 and audit['future_variant_state_checks'] > 0,
        'snapshot_not_after_query': achecks['snapshot_not_after_query'],
        'role_slots_invariance_no_failures': audit['role_slot_failures'] == 0 and audit['role_slot_variant_state_checks'] > 0,
        'soul_elements_no_OTHER_or_ERROR': souls_sum['owned_OTHER_or_ERROR_events'] == 0 and souls_sum['unassigned_OTHER_or_ERROR_events'] == 0,
        'all_six_soul_kinds_recognized': souls_sum['all_six_kinds_recognized'],
        'v1_loader_fidelity_all_equal': audit['v1_loader_fidelity_matches_equal'] == audit['matches'],
        'real_cache_audit_no_failures': audit['matches_with_any_failure'] == 0,
        'adapter_guards_all_pass': all(guards.values()),
        'champion_ids_categorical': train_info['expanded']['categorical_one_hot'] == list(CHAMPION_COLUMNS),
        'v1_old_model_reproduces_stored_probabilities': all(
            reproduction[f]['max_abs_diff_old_model_on_stored_v1_states_vs_stored_curve'] < 1e-9 for f in FAMILIES)
            and all(reproduction['model_sha256_equal_results_model_ids'].values()),
        'v1_stored_probabilities_reproduce_reported_metrics': all(all(v.values()) for v in reported.values()) if reported else None,
        'legacy_sources_and_old_artifacts_unchanged_during_run': not changed,
        'legacy_sources_equal_old_protocol_source_hashes': all(
            hashes_before['legacy_sources'][k] == v for k, v in old_proto_hashes.items()),
        'bootstrap_point_equals_metric_difference': all(v for k, v in echecks.items() if k.endswith('metric_difference')),
        'cv_folds_identical_to_original': train_info['expanded']['cv_folds_identical_to_original'] if not a.smoke else None}
    failed = [k for k, v in checks.items() if v is False]
    informational = {'state_value_v2_module_sha256': hashes_before['legacy_sources']['gameplay/state_value_v2.py'],
                     'state_value_v2_fix_results_module_sha256': v2fix_record,
                     'equal': hashes_before['legacy_sources']['gameplay/state_value_v2.py'] == v2fix_record}
    validation = {'status': 'pass' if not failed else 'fail', 'failed_checks': failed, 'checks': checks,
                  'informational_v2_module_hash': informational,
                  'contract_tests': contract, 'protocol_checks': pchecks, 'assembly_checks': achecks,
                  'actual_state_partition_leakage': leakage_actual, 'evaluation_checks': echecks,
                  'real_cache_audit': audit, 'soul_events': {k: v for k, v in souls_sum.items()},
                  'adapter_guards': guards, 'v1_reproduction': reproduction, 'v1_reported_metric_reproduction': reported,
                  'v1_v2_state_census_test_grid': census, 'changed_files_during_run': changed,
                  'claims_not_made': ['no causal effect of objectives or states', 'no all-patch generalization',
                                      'no untouched external test (same exploratory corpus; 15.16 and 26.13 examined earlier)',
                                      'bootstrap excludes training/selection variability',
                                      'no engagement labels or q model in this task']}
    write_json(run.out / 'validation.json', validation)
    census_rows = {p: {'rows': int(c['rows']), 'matches': int(c['matches']), 'bands': c['bands']}
                   for p, c in pchecks['sampled_by_role'].items()}
    results = {
        'status': 'complete' if not failed else 'complete_with_failed_validation', 'smoke': bool(a.smoke),
        'protocol': protocol['version'], 'state_version': STATE_VERSION, 'model_version': MODEL_VERSION,
        'primary': 'expanded', 'chosen': chosen, 'model_sha256': model_ids, 'candidate_sha256': candidate_ids,
        'selection_log_loss': {k: v['log_loss'] for k, v in scores.items()}, 'training': train_info,
        'sampling_census_protocol': census_rows, 'built_state_counts': achecks, 'built_by_partition': built_by_partition,
        'test_one_minute_V2': {f: evaluation['test_one_minute'][f]['V2'] for f in FAMILIES},
        'test_one_minute_training_prior': evaluation['test_one_minute']['training_prior'],
        'test_grid_time_bands_V2': {b: {f: c[f]['V2'] for f in FAMILIES if f in c}
                                    for b, c in evaluation['test_grid_time_bands'].items()},
        'validation_status': validation['status'],
        'model_semantics': 'P(final Blue win | current observed state); observed differences, not causal effects',
        'elapsed_seconds': round(time.time() - run.started, 1)}
    write_json(run.out / 'results.json', results)
    comparison = {
        'definitions': {'V2': f'{MODEL_VERSION} on {STATE_VERSION} states rebuilt from raw cache',
                        'V1_stored': 'independent_wp_v3_bucket_sampling probabilities stored in independent_time_curves.npz '
                                     '(objective_history_v1, role-slot order); not relabelled',
                        'keys': 'exact (match, time_ms) keys; one-minute test rows from sampled_minutes.json role=test; '
                                'grid rows from independent_time_curves.npz',
                        'weights': 'equal total weight per match within each cell',
                        'difference': 'V2 minus V1; AUC higher is better, Brier/log loss lower is better'},
        'chosen': {'V2': chosen, 'V1': old_results['chosen']},
        **evaluation, 'state_census_test_grid': census}
    write_json(run.out / 'comparison_v1_v2.json', comparison)
    write_report(run.out)
    if (run.out / 'errors.json').exists() and not failed:
        run.log('note: an errors.json from an earlier attempt exists; this run completed')
    run.stage('complete', validation_status=validation['status'], failed_checks=failed)
    return validation['status']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', type=int, default=0, help='matches per partition; writes to smoke_run/')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunk-size', type=int, default=0)
    ap.add_argument('--audit-per-partition', type=int, default=40)
    ap.add_argument('--report-only', action='store_true')
    a = ap.parse_args()
    if not 1 <= a.workers <= 4:
        raise SystemExit('workers must be between 1 and 4')
    out = OUT_BASE / f'smoke_run_n{a.smoke}' if a.smoke else OUT_BASE
    if a.report_only:
        write_report(out)
        return 0
    run = Run(out, bool(a.smoke))
    try:
        status = execute(run, a)
    except Exception as exc:
        errors = read_json(out / 'errors.json') if (out / 'errors.json').exists() else {}
        errors.update(failed_stage=run.status.get('stage'), error=repr(exc), traceback=traceback.format_exc())
        write_json(out / 'errors.json', errors)
        run.status['error'] = repr(exc)
        run.stage('failed')
        raise
    return 0 if status == 'pass' else 2


# ----------------------------------------------------------------------------- report
def _f(x, d=4):
    return '-' if x is None else f'{x:.{d}f}'


def _ci(cell, metric, d=4):
    if not cell or cell.get('status') != 'ok':
        return '-'
    m = cell[metric]
    return f"{m['point']:+.{d}f} [{m['ci95'][0]:+.{d}f}, {m['ci95'][1]:+.{d}f}]"


def write_report(out):
    R, C, V, P = (read_json(out / n) for n in ('results.json', 'comparison_v1_v2.json', 'validation.json', 'protocol.json'))
    S, CV = read_json(out / 'selection.json'), read_json(out / 'expanded_training_cv.json')
    old_sel = read_json(OLD / 'selection.json')
    one, bands, strata, cen = C['test_one_minute'], C['test_grid_time_bands'], C['test_grid_objective_strata'], C['state_census_test_grid']
    au, soul, ch = V['real_cache_audit'], V['soul_events'], V['checks']
    L = ['# P1: independent V2 snapshot win probability', '']
    if R['smoke']:
        L += ['**SMOKE RUN on a tiny subset. Metrics below are not interpretable.**', '']
    L += [f"- Run status: `{R['status']}`; validation: `{V['status']}`"
          + (f" (failed: {', '.join(V['failed_checks'])})" if V['failed_checks'] else ''),
          f"- State version `{R['state_version']}`; model version `{R['model_version']}`; primary family `expanded`.",
          f"- Chosen V2 models: maymin `{R['chosen']['maymin']}`, expanded `{R['chosen']['expanded']}` "
          f"(V1 stored: `{C['chosen']['V1']['maymin']}`, `{C['chosen']['V1']['expanded']}`).",
          '- Semantics: P(final Blue win | current observed state). Observed differences only; no causal claims.', '']
    L += ['## What changed (V1 -> V2)', '',
          '- Participant blocks are ordered by team (100, 200) then numeric participantId. Postgame `role_slots` are '
          'never read; features are renamed `slotN_*` -> `participant_slotN_*` in the same column order.',
          '- Dragon soul element is parsed from `name`/`dragonSoul`/`soulType` with aliases; conflicting fields fail.',
          '- `DRAGON_SOUL_GIVEN` events with teamId 0 are not ownership: they no longer count in '
          '`unknown_objective_team_count` and are exposed only as `StateV2.unassigned_soul_events` (diagnostic).',
          '- Everything else is the V1 protocol: same splits, same stored queries, same families, same C rule, '
          'same calibrators and selection rule.', '',
          f"State census on the stored test grid ({cen['rows']} rows, {cen['matches']} matches), V2 vs stored V1 states:", '',
          '| quantity | value |', '|---|---|',
          f"| matches whose participant order differs from role_slots | {cen['matches_participant_order_differs_from_role_slots']} |",
          f"| rows with any participant-block difference | {cen['rows_participant_block_differs']} |",
          f"| ... of which in matches not reordered (expected 0) | {cen['rows_participant_block_differs_in_matches_not_reordered']} |",
          f"| rows with any soul-column difference | {cen['rows_soul_columns_differ']} |",
          f"| rows with V1 soul_OTHER set / V2 soul_OTHER set | {cen['rows_v1_soul_OTHER_set']} / {cen['rows_v2_soul_OTHER_set']} |",
          f"| rows with a V2 specific owned soul element | {cen['rows_v2_specific_owned_soul_set']} |",
          f"| rows where unknown_objective_team_count differs | {cen['rows_unknown_objective_team_count_differs']} |",
          f"| V1 minus V2 unknown count equals teamId-0 soul events (all rows) | {cen['unknown_count_v1_minus_v2_equals_unassigned_soul_events']} |",
          f"| rows where any other column differs (expected 0) | {cen['rows_other_columns_differ']} |", '']
    pc = V['protocol_checks']
    L += ['## Protocol reuse', '', '| partition | protocol matches | stored sampled rows | rows rebuilt | match set equal |',
          '|---|---|---|---|---|']
    for p in PARTITIONS:
        c = pc['sampled_by_role'][p]
        L.append(f"| {p} | {pc['split_sizes'][p]} | {c['census_rows']} | {R['built_by_partition'][p]['rows']} | "
                 f"{c['match_set_equals_protocol_split']} |")
    b = V['assembly_checks']
    L += ['', f"- Built {b['built_states']} unique states for {b['built_matches']} matches; sampled rows {b['sampled_rows']} "
          f"(elementwise key order equal: {b['sampled_keys_elementwise_equal']}); stored test grid rows {b['grid_rows']} "
          f"(keys equal: {b['grid_keys_elementwise_equal']}). No state was skipped.",
          f"- Engagement exclusions (overlap with any V partition must be 0): "
          + '; '.join(f"{k} n={pc['engagement_sources'][k]} overlap={sum(v.values())}" for k, v in pc['engagement_overlap_with_V'].items()),
          f"- Expanded C: grouped 3-fold CV on fit over {CV['scores'] and [s['C'] for s in CV['scores']]}; chosen C={CV['chosen_C']} "
          f"(original {CV['original_chosen_C']}); folds identical to original: {CV.get('folds_identical_to_original')}.",
          '- Final bases fitted unweighted on fit (as the original); raw/sigmoid/isotonic fitted on calibrate; '
          'selection by select log loss. Test never selected anything.', '',
          '| candidate | V2 select log loss | V1 select log loss |', '|---|---|---|']
    for k in S['scores']:
        L.append(f"| {k} | {_f(S['scores'][k]['log_loss'], 5)} | {_f(old_sel['scores'].get(k, {}).get('log_loss'), 5)} |")
    L += ['', '## Before/after: one query per test match', '',
          f"Rows {one['expanded']['V2']['n']}, matches {one['expanded']['V2']['matches']}; keys matched exactly to stored V1 probabilities.", '',
          '| family | model | AUC | Brier | log loss | ECE | cal. intercept | cal. slope |', '|---|---|---|---|---|---|---|---|']
    for f in FAMILIES:
        for tag in ('V1_stored', 'V2'):
            c = one[f][tag]
            L.append(f"| {f} | {tag} | {_f(c['auc'])} | {_f(c['brier'])} | {_f(c['log_loss'])} | {_f(c['ece'])} | "
                     f"{_f(c.get('calibration_intercept'), 3)} | {_f(c.get('calibration_slope'), 3)} |")
    pr = one['training_prior']
    L += [f"| - | training prior | {_f(pr['auc'])} | {_f(pr['brier'])} | {_f(pr['log_loss'])} | - | - | - |", '',
          'Paired test-match bootstrap, V2 minus V1 (models fixed; training/selection variability not included):', '',
          '| family | n valid | AUC diff [95% CI] | Brier diff [95% CI] | log loss diff [95% CI] | share V2 better (AUC/Brier/LL) |',
          '|---|---|---|---|---|---|']
    for f in FAMILIES:
        bs = one[f]['paired_bootstrap_V2_minus_V1']
        share = '/'.join(_f(bs[m]['fraction_replicates_V2_better'], 3) for m in ('auc', 'brier', 'log_loss')) if bs.get('status') == 'ok' else '-'
        L.append(f"| {f} | {bs.get('n_boot_valid', '-')} | {_ci(bs, 'auc')} | {_ci(bs, 'brier')} | {_ci(bs, 'log_loss')} | {share} |")
    ag = one['expanded']['prediction_agreement']
    L += ['', f"Expanded V2 vs V1 probability agreement: r={_f(ag['pearson_r'])}, mean |diff|={_f(ag['mean_abs_diff'])}, "
          f"share |diff|>0.05={_f(ag['fraction_abs_diff_gt_0.05'], 3)}.", '',
          '## Full stored test grid by time band (expanded)', '',
          '| band (min) | rows | matches | AUC V1/V2 | Brier V1/V2 | log loss V1/V2 | ECE V1/V2 | slope V1/V2 | log loss diff [95% CI] |',
          '|---|---|---|---|---|---|---|---|---|']

    def row(name, c):
        if c.get('status') == 'empty' or 'expanded' not in c:
            return f"| {name} | {c.get('rows', 0)} | {c.get('matches', 0)} | - | - | - | - | - | - |"
        e1, e2 = c['expanded']['V1_stored'], c['expanded']['V2']
        pair = lambda k, d=4: f"{_f(e1.get(k), d)}/{_f(e2.get(k), d)}"
        return (f"| {name} | {c['rows']} | {c['matches']} | {pair('auc')} | {pair('brier')} | {pair('log_loss')} | "
                f"{pair('ece')} | {pair('calibration_slope', 3)} | {_ci(c['expanded'].get('paired_bootstrap_V2_minus_V1'), 'log_loss')} |")
    for name, c in bands.items():
        L.append(row(name, c))
    L += ['', '## Objective histories on the full stored test grid (expanded)', '',
          'Strata are defined from the V2 state at the query (history up to that time). They are confounded with game time '
          'and each other; they describe where the two representations differ, not effects. teamId-0 soul rows are a '
          'diagnostic, not owned souls.', '',
          '| stratum | rows | matches | AUC V1/V2 | Brier V1/V2 | log loss V1/V2 | ECE V1/V2 | slope V1/V2 | log loss diff [95% CI] |',
          '|---|---|---|---|---|---|---|---|---|']
    for name, c in strata.items():
        L.append(row(name, c))
    L += ['', '## Validation', '', '| check | result |', '|---|---|']
    L += [f'| {k} | {v} |' for k, v in ch.items()]
    L += ['', f"- Real-cache audit: {au['matches']} matches ({au['matches_by_partition']}), {au['queries']} queries; "
          f"{au['role_slot_variant_state_checks']} role-slot variant states (deleted/None/empty/reversed/team-swapped/garbage), "
          f"{au['future_variant_state_checks']} future-perturbed states (future frames rescaled, future events dropped or "
          f"mutated plus injected fake soul/baron/kill/GAME_END); failures {au['matches_with_any_failure']}. "
          f"V1 recomputation from raw cache equal to stored V1 states in {au['v1_loader_fidelity_matches_equal']}/{au['matches']} matches.",
          f"- Every built state was checked against raw soul events (owned flags per element and teamId-0 counts). "
          f"Owned soul events by element: {soul['owned_events_by_element']}; teamId-0 events: {soul['unassigned_events_by_team_element']}; "
          f"all six kinds recognized: {soul['all_six_kinds_recognized']}.",
          f"- Adapter guards: {V['adapter_guards']}.",
          f"- Old V1 models on stored V1 states reproduce stored probabilities: "
          f"{ {f: V['v1_reproduction'][f]['max_abs_diff_old_model_on_stored_v1_states_vs_stored_curve'] for f in FAMILIES} } (max abs diff).",
          f"- Legacy sources, old V artifacts and state_value_v2_fix files hashed before and after the run; changed: {V['changed_files_during_run'] or 'none'}.",
          f"- V1 object champion handling: {V['v1_reproduction']['v1_expanded_champion_handling']['encoder']} over "
          f"{len(V['v1_reproduction']['v1_expanded_champion_handling']['one_hot_columns'])} role-slot champion columns; V1 object "
          f"vectorises bare same-width arrays without a schema check: "
          f"{V['v1_reproduction']['v1_object_accepts_any_same_width_array_without_schema_check']} (hence the guarded V2 adapter).",
          '', '## Tests run', '',
          f"- `{V['contract_tests']['command']}`: {V['contract_tests']['summary']} (rerun inside the pipeline before extraction).",
          *([f"- Smoke runs (tiny subsets, same code path): "
             + '; '.join(f"`{d.name}` validation={read_json(d / 'validation.json')['status'] if (d / 'validation.json').exists() else 'incomplete'}"
                         for d in sorted(OUT_BASE.glob('smoke_run*')) if d.is_dir())] if not R['smoke'] else []),
          '- Pipeline assertions listed above (leakage, query membership, winners, finiteness, future invariance, role-slot invariance, '
          'soul parsing, schema/version guards, adapter equivalence to the frozen SnapshotWinProbability on select, reload equality).', '',
          '## Limitations', '',
          '- Same exploratory 50k corpus (patches 15.14-15.16) as V1; this is not an untouched external test and makes no '
          'all-patch generalization claim. Patches 15.16 and 26.13 were examined previously.',
          '- Bootstrap intervals resample test matches with fitted models, calibrators and selections fixed; they omit '
          'training, CV and selection variability.',
          '- Grid rows within a match are dependent; cells use equal total weight per match but band/stratum results are '
          'descriptive, and strata are confounded with time and with each other.',
          '- The original final base fits were unweighted while CV fits were match-weighted; this was reproduced, not changed.',
          '- States were rebuilt in float64 whereas V1 trained on stored float32 states; the historical loader '
          '(`data.cache_io.load_match_cache`) was reused read-only.',
          '- teamId-0 soul events have unknown game meaning; they are reported only as a diagnostic.',
          '- This validates snapshot win probability only. It does not validate the engagement 90-second endpoint, '
          'engagement labels, or any q model.', '',
          '## Next-stage readiness', '',
          '- A versioned adapter (`train.independent_winprob_v2.IndependentWinProbabilityV2`, loaded with `load_adapter`) now '
          'serves V2 states with schema/version guards. Engagement p_pre/Y regeneration must use this adapter on '
          'freshly built V2 states; old V1 probabilities must not be copied.',
          '- Readiness is conditional on validation status above; no engagement labels or q models were produced here.', '',
          '## Artifacts', '',
          '- `protocol.json`, `status.json`, `run.log`, `results.json`, `validation.json`, `comparison_v1_v2.json`',
          '- `states/chunks/*.npz` (restart checkpoints), `states/sampled_states_v2.npz`, `states/test_grid_states_v2.npz`, '
          '`states/match_table.json`, `states/soul_event_audit.json`, `states/audit_real_cache.json`',
          '- `models/maymin_model_v2.joblib`, `models/expanded_model_v2.joblib`, `models/candidates/*.joblib`, `models/inference_adapter.json`',
          '- `predictions/test_one_minute_v1_v2.csv|.npz`, `predictions/test_grid_v1_v2.npz`, `predictions/select_partition_candidates_v2.npz`',
          '- `expanded_training_cv.json`, `selection.json`, `hashes_before.json`, `hashes_after.json`, `source_snapshot/`', '',
          '## References', '',
          '- Maymin, P. Z. (2021). Smart kills and worthless deaths: eSports analytics for League of Legends. '
          'Journal of Quantitative Analysis in Sports. DOI 10.1515/jqas-2019-0096',
          '- Decroos, T., Bransen, L., Van Haaren, J., Davis, J. (2019). Actions speak louder than goals: valuing player '
          'actions in soccer. KDD 2019. DOI 10.1145/3292500.3330758',
          '- Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., Steyerberg, E. W. (2019). Calibration: the '
          'Achilles heel of predictive analytics. BMC Medicine. DOI 10.1186/s12916-019-1466-7',
          '', 'These references support game-state value modelling and probability validation practice; they do not '
          'validate this exact state schema or the 90-second engagement endpoint.', '']
    (out / 'REPORT.md').write_text('\n'.join(L), encoding='utf-8')

if __name__ == '__main__':
    raise SystemExit(main())
