"""P2: temporal win-probability refactor and evaluation (exploratory v3).

Candidate A = the frozen P1 expanded V2 adapter f(X_t, t) (no refit, no recalibration).
Candidate B = one SimpleRNN over causal one-minute histories X_{t-8min}, ..., X_t (seeds 17/29/43, calibrated
per seed on CALIBRATE, calibration chosen per seed on SELECT, primary = arithmetic mean of the three).
A vs B is decided on SELECT log loss (exact tie keeps A) and frozen in selection.json before any TEST
prediction is computed. Exact P1 splits and query keys are reused. Engagement endpoint transport, trajectories
and the frame/event update audit are DIAGNOSTIC only. All outputs land in outputs/temporal_winprob_v3.
"""
from __future__ import annotations

import os

_WORKER = os.environ.get('P2_WORKER') == '1'
_THREADS = '1' if _WORKER else '4'
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = _THREADS
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys

sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('TEAMFIGHT_WORKSPACE') and Path(os.environ['TEAMFIGHT_WORKSPACE']).resolve() != ROOT:
    raise SystemExit('script is not inside TEAMFIGHT_WORKSPACE')
WT = ROOT / 'worktrees' / 'engagement-state-value'
sys.path.insert(0, str(WT))
sys.path.insert(0, str(ROOT / 'scripts'))
OLD = WT / 'outputs' / 'temporal_winprob_v3_buckets'
OLD_DATASET = WT / 'outputs' / 'state_value_main_50k'
OLD_EVAL = WT / 'outputs' / 'state_value_main_50k_eval'
V2FIX = ROOT / 'outputs' / 'state_value_v2_fix'
P1 = ROOT / 'outputs' / 'independent_v2_participant_order'
CACHE = Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
OUT_BASE = ROOT / 'outputs' / 'temporal_winprob_v3'
PARTITIONS = ('fit', 'calibrate', 'select', 'test')
SEEDS = (17, 29, 43)
EPOCHS, BATCH, LR, UNITS, DROPOUT, RHO, EPS = 50, 64, 1e-3, 8, .25, .9, 1e-7
N_BOOT_PRIMARY, N_BOOT_EXPLORATORY, BOOT_SEED = 2000, 500, 7
TRAJECTORY_MATCHES, TRAJECTORY_CADENCE_MS, FRAME_AUDIT_MATCHES, REPLAY_AUDIT_PER_PARTITION = 12, 10_000, 40, 6
MIN_EVAL_QUERY_MS = 120_000
EXPECTED_ENGAGEMENT_ROWS = 26_693
AGE_BINS_S = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 1e9))
DELTA_BINS = ((0, .005), (.005, .01), (.01, .02), (.02, .05), (.05, 1.01))

LEGACY_WT_SOURCES = ('gameplay/state_value.py', 'gameplay/state_value_v2.py', 'train/temporal_winprob.py',
                     'train/state_value_experiment.py', 'scripts/run_temporal_winprob_v3_buckets.py',
                     'scripts/build_state_value_dataset.py', 'core/config.py', 'data/cache_io.py',
                     'train/independent_winprob_v2.py')
LEGACY_ROOT_SOURCES = ('scripts/train_independent_v2.py', 'tests/test_state_value_v2_contract.py',
                       'scripts/regenerate_state_v2.py', 'scripts/run_validation_suite.py')
NEW_SOURCES = {'scripts/run_temporal_winprob_v3.py': ROOT / 'scripts' / 'run_temporal_winprob_v3.py',
               'scripts/temporal_winprob_v3_eval.py': ROOT / 'scripts' / 'temporal_winprob_v3_eval.py',
               'scripts/temporal_winprob_v3_report.py': ROOT / 'scripts' / 'temporal_winprob_v3_report.py',
               'tests/test_temporal_winprob_v3_history.py': ROOT / 'tests' / 'test_temporal_winprob_v3_history.py',
               'worktrees/engagement-state-value/train/temporal_history_winprob_v3.py':
                   WT / 'train' / 'temporal_history_winprob_v3.py'}

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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.bool_):
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
        self.status = {'stage': 'starting', 'pid': os.getpid(), 'smoke': smoke, 'history': [], 'blockers': []}

    def log(self, msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")} +{time.time() - self.started:7.0f}s] {msg}'
        print(line, flush=True)
        with (self.out / 'run.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    def stage(self, name, quiet=False, **info):
        self.status.update(stage=name, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                           elapsed_seconds=round(time.time() - self.started, 1), detail=_jsonable(info))
        if not self.status['history'] or self.status['history'][-1]['stage'] != name:
            self.status['history'].append({'stage': name, 'at': self.status['updated']})
        write_json(self.out / 'status.json', self.status)
        if not quiet:
            self.log(f'stage={name} {json.dumps(_jsonable(info), ensure_ascii=False)[:400]}')

    def blocker(self, stage, exc):
        rec = {'stage': stage, 'error': repr(exc), 'traceback': traceback.format_exc(),
               'at': time.strftime('%Y-%m-%d %H:%M:%S')}
        self.status['blockers'].append({k: rec[k] for k in ('stage', 'error', 'at')})
        errors = read_json(self.out / 'errors.json') if (self.out / 'errors.json').exists() else {}
        errors.setdefault('blockers', []).append(rec)
        write_json(self.out / 'errors.json', errors)
        self.log(f'BLOCKER in {stage}: {exc!r} (independent stages continue)')
        write_json(self.out / 'status.json', self.status)


def _hash_tree(base, recursive=True):
    if not base.exists():
        return {}
    files = sorted(base.rglob('*') if recursive else base.glob('*'))
    return {str(p.relative_to(base)).replace('\\', '/'): sha256_file(p) for p in files if p.is_file()}


def legacy_hashes():
    src = {p: sha256_file(WT / p) for p in LEGACY_WT_SOURCES}
    src.update({p: sha256_file(ROOT / p) for p in LEGACY_ROOT_SOURCES})
    return {'legacy_sources': src, 'p1_outputs': _hash_tree(P1), 'old_v_artifacts': _hash_tree(OLD),
            'state_value_v2_fix': _hash_tree(V2FIX, recursive=False)}


def _rank(tag, m):
    return hashlib.sha256(f'{tag}:{m}'.encode()).hexdigest()


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
        from gameplay.state_value import final_outcome
        from gameplay.state_value_v2 import state_matrix, STATE_VERSION
        import train.temporal_history_winprob_v3 as T
        _CTX.update(load=load_match_cache, final_outcome=final_outcome, state_matrix=state_matrix, T=T,
                    node_names=list(NODE_FEATURE_NAMES), state_version=STATE_VERSION,
                    feature_version=str(getattr(cfg, 'FEATURE_VERSION', '')))
    return _CTX


def _load(mid, ctx):
    pack = ctx['load'](mid)
    if pack is None:
        raise ValueError(f'unreadable_or_incompatible_cache: {mid}')
    return pack


def _frozen(task):
    """Frozen candidates in a worker, verified by hash on every load."""
    ctx = _context()
    T = ctx['T']
    key = (task['a_path'], task['a_sha256'], task.get('b_dir'), task.get('b_sha256'))
    if ctx.get('frozen_key') != key:
        A = T.CandidateA.load(task['a_path'], task['a_sha256'])
        B = T.CandidateB.load(task['b_dir'], task['b_sha256']) if task.get('b_dir') else None
        ctx.update(frozen_key=key, A=A, B=B,
                   sA=T.TemporalWinProbabilityService(A), sB=T.TemporalWinProbabilityService(B) if B else None)
    return ctx['A'], ctx['B'], ctx['sA'], ctx['sB']


def event_category(e):
    typ = e.get('type')
    if typ == 'CHAMPION_KILL':
        return 'champion_kill', None
    if typ == 'ELITE_MONSTER_KILL':
        team = int(e.get('killerTeamId', 0) or 0)
        monster, sub = e.get('monsterType', ''), str(e.get('monsterSubType', '')).upper()
        if monster == 'DRAGON':
            return ('elder' if sub == 'ELDER_DRAGON' else 'dragon'), team
        return {'BARON_NASHOR': 'baron', 'RIFTHERALD': 'herald', 'HORDE': 'horde', 'ATAKHAN': 'atakhan'}.get(monster, 'monster_other'), team
    if typ == 'DRAGON_SOUL_GIVEN':
        team = int(e.get('teamId', 0) or 0)
        return ('soul_owned' if team in (100, 200) else 'soul_unassigned'), team
    if typ == 'BUILDING_KILL':
        lost = int(e.get('teamId', 0) or 0)
        return ('inhibitor' if e.get('buildingType') == 'INHIBITOR_BUILDING' else 'tower'), (300 - lost if lost in (100, 200) else 0)
    if typ == 'TURRET_PLATE_DESTROYED':
        lost = int(e.get('teamId', 0) or 0)
        return 'plate', (300 - lost if lost in (100, 200) else 0)
    return None, None


def extract_chunk(task):
    """States at every planned history position; any failure blocks the whole chunk (never skipped)."""
    started = time.time()
    ctx = _context()
    T = ctx['T']
    names = task['names']
    rows, per, errors = defaultdict(list), defaultdict(list), {}
    for mid, times in task['items']:
        try:
            pack = _load(mid, ctx)
            winner, terminal = ctx['final_outcome'](pack['events'])
            src = T.BatchSource(mid, pack, ctx['node_names'])
            built = [t for t in times if t >= src.support_start_ms]
            states = [src.state_at(t) for t in built]
            X = ctx['state_matrix'](states, names)
            if not np.isfinite(X).all():
                raise ValueError('non-finite state')
            ts = np.asarray(pack['minute_ts'], dtype=np.int64)
            for t, s in zip(built, states):
                snap = int(ts[np.searchsorted(ts, t, side='right') - 1])
                if s.query_ms != t or s.snapshot_ms != snap or s.snapshot_ms > t:
                    raise ValueError(f'query/snapshot time violation at {t}')
        except Exception as exc:
            errors[mid] = f'{type(exc).__name__}: {exc}'
            continue
        rows['match'].extend([mid] * len(built))
        rows['time_ms'].extend(built)
        rows['snapshot_ms'].extend(s.snapshot_ms for s in states)
        rows['unassigned'].extend(s.unassigned_soul_events for s in states)
        rows['X'].append(X)
        per['match_ids'].append(mid)
        per['winner'].append(winner)
        per['terminal_ms'].append(terminal)
        per['last_frame_ms'].append(int(ts[-1]))
        per['support_start_ms'].append(src.support_start_ms)
        per['unsupported_times'].append(len(times) - len(built))
        per['patch'].append(str(pack['meta'].get('patch', '')))
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
             support_start_ms=np.asarray(per['support_start_ms'], dtype=np.int64),
             unsupported_times=np.asarray(per['unsupported_times'], dtype=np.int64),
             patch=np.asarray(per['patch'], dtype='U16'), names_sha256=np.asarray(sha256_json(names)),
             state_version=np.asarray(ctx['state_version']), plan_sha256=np.asarray(task['plan_sha256']))
    summary['written'] = True
    return summary


def _event_records(events, frames):
    frame_set = set(int(f) for f in frames)
    all_counts = Counter(int(e.get('timestamp', -1)) for e in events)
    T_types = set(_context()['T'].PREDICTOR_EVENT_TYPES)
    pred_counts = Counter(int(e['timestamp']) for e in events if e.get('type') in T_types)
    out = []
    for e in events:
        cat, team = event_category(e)
        if cat is None:
            continue
        ts = int(e['timestamp'])
        detail = ' '.join(f'{k}={e[k]}' for k in ('monsterType', 'monsterSubType', 'buildingType', 'towerType', 'name',
                                                   'killerTeamId', 'teamId', 'killerId', 'victimId') if k in e)
        out.append({'ts': ts, 'type': e.get('type'), 'category': cat, 'team': team, 'detail': detail,
                    'n_events_same_ts': all_counts[ts], 'n_predictor_events_same_ts': pred_counts[ts],
                    'frame_at_ts': ts in frame_set})
    return out


def _replay_stream(pack):
    ts = np.asarray(pack['minute_ts'], dtype=np.int64)
    return sorted([(int(t), 0, i) for i, t in enumerate(ts)] +
                  [(int(e['timestamp']), 1, i) for i, e in enumerate(pack['events'])], key=lambda r: (r[0], r[1]))


def _replay_compare(ctx, mid, pack, queries, batch_histories, names, sA, sB):
    """Chronological ObservationLog replay; histories and single-history predictions must equal batch ones exactly."""
    T = ctx['T']
    log = T.ObservationLog(mid, pack['meta']['team_map'], ctx['node_names'])
    stream = _replay_stream(pack)
    pos = 0
    rec = {'queries': len(queries), 'matrix_mismatch': 0, 'mask_mismatch': 0, 'snapshot_mismatch': 0,
           'p_A_mismatch': 0, 'p_B_mismatch': 0}
    order = np.argsort(queries, kind='mergesort')
    for i in order:
        q = queries[i]
        while pos < len(stream) and stream[pos][0] <= q:
            t, kind, j = stream[pos]
            if kind == 0:
                log.add_frame(mid, t, pack['node_minute'][j], ctx['node_names'])
            else:
                log.add_event(mid, pack['events'][j])
            pos += 1
        hb, hi = batch_histories[i], T.assemble_history(log.source(q), q, names)
        rec['matrix_mismatch'] += int(not np.array_equal(hb.matrix(), hi.matrix()))
        rec['mask_mismatch'] += int(hb.mask != hi.mask)
        rec['snapshot_mismatch'] += int(hb.snapshot_ms != hi.snapshot_ms)
        rec['p_A_mismatch'] += int(sA.predict(hb).p != sA.predict(hi).p)
        if sB is not None:
            pb, pi = sB.predict(hb), sB.predict(hi)
            rec['p_B_mismatch'] += int(pb.p != pi.p or pb.components != pi.components)
    return rec


def trajectory_task(task):
    ctx = _context()
    T = ctx['T']
    A, B, sA, sB = _frozen(task)
    names, mid = task['names'], task['match']
    pack = _load(mid, ctx)
    winner, terminal = ctx['final_outcome'](pack['events'])
    ts = np.asarray(pack['minute_ts'], dtype=np.int64)
    src = T.BatchSource(mid, pack, ctx['node_names'])
    end_q = min(terminal - 1, int(ts[-1]))
    reasons = defaultdict(set)
    for q in range(MIN_EVAL_QUERY_MS, end_q + 1, task['cadence_ms']):
        reasons[q].add(f"cadence_{task['cadence_ms'] // 1000}s")
    for f in ts.tolist():
        for q, r in ((f - 1, 'frame_update_minus_1ms'), (f, 'frame_update')):
            if MIN_EVAL_QUERY_MS <= q <= end_q:
                reasons[q].add(r)
    events = _event_records(pack['events'], ts.tolist())
    for e in events:
        for q, r in ((e['ts'] - 1, 'event_minus_1ms'), (e['ts'], 'event')):
            if MIN_EVAL_QUERY_MS <= q <= end_q:
                reasons[q].add(r)
    queries = sorted(reasons)
    histories = [T.assemble_history(src, q, names) for q in queries]
    pa = sA.predict_many(histories)
    pb = sB.predict_many(histories) if sB else None
    replay = _replay_compare(ctx, mid, pack, queries, histories, names, sA, sB)
    replay['max_abs_diff_batch_many_vs_single_B'] = (
        float(max(abs(pb[i].p - sB.predict(histories[i]).p) for i in range(0, len(histories), 25))) if sB else None)
    return {'match': mid, 'winner_blue': int(winner), 'terminal_ms': int(terminal), 'last_frame_ms': int(ts[-1]),
            'query_ms': queries, 'reasons': ['|'.join(sorted(reasons[q])) for q in queries],
            'p_A': [p.p for p in pa], 'p_B': [p.p for p in pb] if pb else None,
            'p_B_members': {s: [p.components[s] for p in pb] for s in pb[0].components} if pb else None,
            'snapshot_ms': [p.snapshot_ms for p in pa], 'snapshot_age_s': [p.snapshot_age_ms / 1000. for p in pa],
            'valid_positions': [int(sum(p.history_mask)) for p in pa],
            'history_snapshot_ms': [list(p.history_snapshot_ms) for p in pa],
            'frames_ms': ts.tolist(), 'events': [e for e in events if e['ts'] <= end_q + 1], 'replay': replay,
            'model_A': dict(zip(('candidate', 'version', 'sha256', 'calibration'), pa[0].model_key()[:4])),
            'model_B': dict(zip(('candidate', 'version', 'sha256', 'calibration'), pb[0].model_key()[:4])) if pb else None}


INTERVAL_CATEGORIES = ('champion_kill', 'dragon', 'elder', 'baron', 'herald', 'horde', 'atakhan', 'soul_owned',
                       'soul_unassigned', 'tower', 'inhibitor', 'plate', 'monster_other')
TEAM_CATEGORIES = ('dragon', 'elder', 'baron', 'soul_owned', 'tower', 'inhibitor')


def endpoint_chunk(task):
    """Frozen A/B at the stored engagement endpoints (pre = s-1 ms, post = endpoint ms). DIAGNOSTIC only."""
    started = time.time()
    ctx = _context()
    T = ctx['T']
    A, B, sA, sB = _frozen(task)
    names = task['names']
    out, errors = defaultdict(list), {}
    for mid, rows in task['items']:
        try:
            pack = _load(mid, ctx)
            winner, terminal = ctx['final_outcome'](pack['events'])
            ts = np.asarray(pack['minute_ts'], dtype=np.int64)
            src = T.BatchSource(mid, pack, ctx['node_names'])
            hist = []
            for r in rows:
                pre_q, post_q = int(r['s']) - 1, int(r['endpoint'])
                if not pre_q < post_q:
                    raise ValueError(f'endpoint not after pre query for row {r["row_index"]}')
                hist += [T.assemble_history(src, pre_q, names), T.assemble_history(src, post_q, names)]
            pa = sA.predict_many(hist)
            pb = sB.predict_many(hist) if sB else None
            cats = []
            for r in rows:
                pre_q, post_q = int(r['s']) - 1, int(r['endpoint'])
                c = Counter()
                for e in pack['events']:
                    t = int(e.get('timestamp', -1))
                    if pre_q < t <= post_q:
                        cat, team = event_category(e)
                        if cat:
                            c[cat] += 1
                            if cat in TEAM_CATEGORIES and team in (100, 200):
                                c[f"{cat}_{'blue' if team == 100 else 'red'}"] += 1
                cats.append(c)
        except Exception as exc:
            errors[mid] = f'{type(exc).__name__}: {exc}'
            continue
        for i, r in enumerate(rows):
            a0, a1 = pa[2 * i], pa[2 * i + 1]
            dA = T.endpoint_delta(a0, a1)
            out['row_index'].append(int(r['row_index']))
            out['match'].append(mid)
            out['winner'].append(int(winner))
            out['terminal_ms'].append(int(terminal))
            out['last_frame_ms'].append(int(ts[-1]))
            out['pre_query_ms'].append(a0.query_ms)
            out['post_query_ms'].append(a1.query_ms)
            out['pre_hist_snapshot_ms'].append(a0.history_snapshot_ms)
            out['post_hist_snapshot_ms'].append(a1.history_snapshot_ms)
            out['pre_X_last'].append(hist[2 * i].matrix()[-1])
            out['post_X_last'].append(hist[2 * i + 1].matrix()[-1])
            out['p_pre_A'].append(a0.p)
            out['p_post_A'].append(a1.p)
            out['delta_A'].append(dA['delta'])
            if pb:
                b0, b1 = pb[2 * i], pb[2 * i + 1]
                out['p_pre_B'].append(b0.p)
                out['p_post_B'].append(b1.p)
                out['delta_B'].append(T.endpoint_delta(b0, b1)['delta'])
                for s in b0.components:
                    out[f'p_pre_B_seed{s}'].append(b0.components[s])
                    out[f'p_post_B_seed{s}'].append(b1.components[s])
            out['events'].append([cats[i].get(k, 0) for k in EVENT_COLUMNS])
    summary = {'chunk_id': task['chunk_id'], 'errors': errors, 'rows': len(out['row_index']),
               'seconds': round(time.time() - started, 1), 'written': False}
    if errors:
        return summary
    arrays = {k: np.asarray(v) for k, v in out.items() if k not in ('match',)}
    arrays['match'] = np.asarray(out['match'], dtype='U16')
    arrays['plan_sha256'] = np.asarray(task['plan_sha256'])
    save_npz(Path(task['path']), **arrays)
    summary['written'] = True
    return summary


EVENT_COLUMNS = tuple(INTERVAL_CATEGORIES) + tuple(f'{c}_{t}' for c in TEAM_CATEGORIES for t in ('blue', 'red'))


def column_groups(names):
    from gameplay.state_value import SNAPSHOT_FIELDS
    groups = []
    for n in names:
        if n in ('time_minutes', 'time_minutes_sq'):
            g = 'clock_time'
        elif n == 'snapshot_age_s':
            g = 'audit_snapshot_age_not_predictor'
        elif n.endswith('_x_time'):
            g = 'time_interaction_derived'
        elif n.startswith('participant_slot'):
            field = n.split('_', 2)[2]
            if field in SNAPSHOT_FIELDS or field == 'champion_id':
                g = 'frame_node_fields'
            elif field == 'death_age_minutes':
                g = 'clock_age'
            elif field == 'death_last_30s':
                g = 'clock_window_flag'
            elif field == 'death_since_snapshot':
                g = 'death_since_snapshot'
            else:
                g = 'participant_event_history'
        elif n.startswith(('blue_', 'red_')):
            if n.endswith('_age_minutes'):
                g = 'clock_age'
            elif '_acquired_last_' in n:
                g = 'clock_window_flag'
            else:
                g = 'team_event_history'
        else:
            g = 'team_event_history'
        groups.append(g)
    return groups


def discrete_changes(x0, x1, groups):
    """Groups whose values changed beyond the 1 ms clock drift between two adjacent queries."""
    changed = set()
    for i, g in enumerate(groups):
        if g in ('clock_time', 'time_interaction_derived', 'audit_snapshot_age_not_predictor'):
            continue
        d = abs(float(x1[i]) - float(x0[i]))
        if g == 'clock_age':
            if d > 1.5 / 60000.:
                changed.add(g)
        elif d != 0:
            changed.add(g)
    return sorted(changed)


def frame_audit_task(task):
    """Queries at x-1 ms and x for raw frame timestamps and predictor-event timestamps (co-occurrence only)."""
    ctx = _context()
    T = ctx['T']
    A, B, sA, sB = _frozen(task)
    names, groups, mid = task['names'], task['groups'], task['match']
    pack = _load(mid, ctx)
    winner, terminal = ctx['final_outcome'](pack['events'])
    ts = np.asarray(pack['minute_ts'], dtype=np.int64)
    frame_set = set(ts.tolist())
    src = T.BatchSource(mid, pack, ctx['node_names'])
    end_q = min(terminal - 1, int(ts[-1]))
    all_at = Counter(int(e.get('timestamp', -1)) for e in pack['events'])
    pred_at = defaultdict(Counter)
    for e in pack['events']:
        if e.get('type') in T.PREDICTOR_EVENT_TYPES:
            pred_at[int(e['timestamp'])][event_category(e)[0]] += 1
    xs = sorted({x for x in frame_set | set(pred_at) if MIN_EVAL_QUERY_MS <= x - 1 and x <= end_q})
    h0 = [T.assemble_history(src, x - 1, names) for x in xs]
    h1 = [T.assemble_history(src, x, names) for x in xs]
    pa = sA.predict_many(h0 + h1)
    pb = sB.predict_many(h0 + h1) if sB else None
    recs = []
    n = len(xs)
    for i, x in enumerate(xs):
        M0, M1 = h0[i].matrix(), h1[i].matrix()
        last = discrete_changes(M0[-1], M1[-1], groups)
        earlier = [k for k in range(8) if h0[i].mask[k] and discrete_changes(M0[k], M1[k], groups)]
        frame = x in frame_set
        n_pred = int(sum(pred_at[x].values()))
        if frame and n_pred:
            cls = 'frame_and_event_same_ts'
        elif frame:
            cls = 'frame_only'
        elif n_pred == 1:
            cls = 'single_predictor_event'
        else:
            cls = 'multiple_predictor_events_same_ts'
        positions_crossing = [k for k in range(8) if (x - (8 - k) * 60000) in frame_set or (x - (8 - k) * 60000) in pred_at]
        recs.append({'match': mid, 'x_ms': int(x), 'update_class': cls, 'frame_update_at_x': frame,
                     'predictor_events_at_x': dict(pred_at[x]), 'all_events_at_x': int(all_at[x]),
                     'last_position_groups_changed': last, 'n_last_position_groups_changed': len(last),
                     'earlier_positions_with_discrete_change': earlier,
                     'earlier_positions_on_frame_or_event_ts': positions_crossing,
                     'dp_A': pa[n + i].p - pa[i].p, 'dp_B': (pb[n + i].p - pb[i].p) if pb else None,
                     'p_A_before': pa[i].p, 'p_B_before': pb[i].p if pb else None,
                     'snapshot_before_ms': pa[i].snapshot_ms, 'snapshot_at_x_ms': pa[n + i].snapshot_ms})
    return {'match': mid, 'records': recs, 'frames': len(ts), 'events': len(pack['events'])}


def _perturb_after(pack, t):
    q = dict(pack)
    ts = np.asarray(pack['minute_ts'], dtype=np.int64)
    node = np.array(pack['node_minute'], copy=True)
    node[ts > t] = node[ts > t] * -3.5 + 777.
    q['node_minute'] = node
    events = []
    for e in pack['events']:
        m = dict(e)
        if int(e.get('timestamp', -1)) > t:
            for k in ('teamId', 'killerTeamId', 'winningTeam'):
                if int(m.get(k, 0) or 0) in (100, 200):
                    m[k] = 300 - int(m[k])
            for k in ('killerId', 'victimId'):
                if int(m.get(k, 0) or 0) > 0:
                    m[k] = int(m[k]) % 10 + 1
        events.append(m)
    events += [dict(type='DRAGON_SOUL_GIVEN', timestamp=t + 1, teamId=100, name='Cloud', dragonSoul='FIRE'),
               dict(type='ELITE_MONSTER_KILL', timestamp=t + 1, killerTeamId=200, killerId=6, monsterType='BARON_NASHOR'),
               dict(type='CHAMPION_KILL', timestamp=t + 1, killerId=1, victimId=6),
               dict(type='GAME_END', timestamp=t + 2, winningTeam=200)]
    q['events'] = events
    return q


def replay_audit_task(task):
    """Real-cache: future perturbation invariance, incremental replay equality, pipeline vs API equality."""
    ctx = _context()
    T = ctx['T']
    A, B, sA, sB = _frozen(task)
    names, mid = task['names'], task['match']
    pack = _load(mid, ctx)
    src = T.BatchSource(mid, pack, ctx['node_names'])
    queries = [int(q) for q in task['queries']]
    hist = [T.assemble_history(src, q, names) for q in queries]
    rec = {'match': mid, 'partition': task['partition'], 'queries': len(queries), 'future_matrix_mismatch': 0,
           'future_mask_mismatch': 0, 'future_p_A_mismatch': 0, 'future_p_B_mismatch': 0, 'masked_queries': 0}
    for q, h in zip(queries, hist):
        hp = T.assemble_history(T.BatchSource(mid, _perturb_after(pack, q), ctx['node_names']), q, names)
        rec['masked_queries'] += int(not all(h.mask))
        rec['future_matrix_mismatch'] += int(not np.array_equal(h.matrix(), hp.matrix()))
        rec['future_mask_mismatch'] += int(h.mask != hp.mask or h.snapshot_ms != hp.snapshot_ms)
        rec['future_p_A_mismatch'] += int(sA.predict(h).p != sA.predict(hp).p)
        if sB:
            rec['future_p_B_mismatch'] += int(sB.predict(h).p != sB.predict(hp).p)
    rec['incremental'] = _replay_compare(ctx, mid, pack, queries, hist, names, sA, sB)
    pipe = {'rows': 0, 'matrix_mismatch': 0, 'mask_mismatch': 0, 'p_A_not_exact': 0, 'p_B_not_exact': 0,
            'p_B_max_abs_diff': 0.}
    for item in task.get('pipeline', []):
        h = hist[queries.index(int(item['query_ms']))]
        pipe['rows'] += 1
        pipe['matrix_mismatch'] += int(not np.array_equal(h.matrix(), np.asarray(item['matrix'])))
        pipe['mask_mismatch'] += int(list(h.mask) != list(item['mask']))
        pipe['p_A_not_exact'] += int(sA.predict(h).p != item['p_A'])
        if sB and item.get('p_B') is not None:
            p = sB.predict(h).p
            pipe['p_B_not_exact'] += int(p != item['p_B'])
            pipe['p_B_max_abs_diff'] = max(pipe['p_B_max_abs_diff'], abs(p - item['p_B']))
    rec['pipeline_vs_api'] = pipe
    return rec


# ----------------------------------------------------------------------------- main side: pools
def run_pool(run, fn, tasks, workers, stage, label):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results, failures = [], {}
    if not tasks:
        return results, failures
    os.environ['P2_WORKER'] = '1'
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
            futures = {ex.submit(fn, t): i for i, t in enumerate(tasks)}
            done = 0
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    failures[str(futures[fut])] = f'{type(exc).__name__}: {exc}'
                done += 1
                run.stage(stage, quiet=done % max(1, len(tasks) // 10) != 0 and done != len(tasks),
                          part=label, tasks_done=done, tasks_total=len(tasks), task_failures=len(failures))
    finally:
        os.environ['P2_WORKER'] = '0'
    return results, failures


# ----------------------------------------------------------------------------- protocol
def overlap_table(sets):
    keys = list(sets)
    return {f'{a}&{b}': len(sets[a] & sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}


def load_inputs():
    with np.load(OLD / 'independent_time_curves.npz', allow_pickle=False) as z:
        curves = {k: z[k] for k in ('match', 'time_ms', 'winner')}
    return {'old_proto': read_json(OLD / 'protocol.json'), 'sampled': read_json(OLD / 'sampled_minutes.json'),
            'curves': curves, 'p1_proto': read_json(P1 / 'protocol.json'), 'p1_val': read_json(P1 / 'validation.json'),
            'p1_res': read_json(P1 / 'results.json'), 'p1_sel': read_json(P1 / 'selection.json'),
            'p1_info': read_json(P1 / 'models' / 'inference_adapter.json')}


def protocol_checks(I, names):
    c, fail = {}, []
    splits = I['old_proto']['splits']
    sets = {k: set(v) for k, v in splits.items()}
    c['p1_validation_status'] = I['p1_val']['status']
    c['p1_status'] = I['p1_res']['status']
    if c['p1_validation_status'] != 'pass':
        fail.append('P1 validation did not pass')
    c['p1_chosen_expanded'] = I['p1_res']['chosen']['expanded']
    c['p1_splits_equal_original'] = all(I['p1_proto']['splits'][k] == splits[k] for k in (*PARTITIONS, 'engagement'))
    c['split_sizes'] = {k: len(v) for k, v in splits.items()}
    c['split_duplicates'] = {k: len(v) - len(sets[k]) for k, v in splits.items()}
    c['split_overlaps'] = overlap_table(sets)
    if not c['p1_splits_equal_original'] or any(c['split_duplicates'].values()) or any(c['split_overlaps'].values()):
        fail.append('splits differ from P1 or overlap')
    with (V2FIX / 'rows.csv').open(encoding='utf-8', newline='') as f:
        fix_rows = list(csv.DictReader(f))
    ms = read_json(OLD_EVAL / 'match_splits.json')
    external = {'protocol_engagement': sets['engagement'], 'state_value_v2_fix_rows': {r['match'] for r in fix_rows},
                'eval_predict_train': set(ms['predict_train']), 'eval_predict_test': set(ms['predict_test'])}
    c['engagement_sources'] = {k: len(v) for k, v in external.items()}
    c['engagement_overlap_with_V'] = {k: {p: len(v & sets[p]) for p in PARTITIONS} for k, v in external.items()}
    if any(n for d in c['engagement_overlap_with_V'].values() for n in d.values()):
        fail.append('engagement match leakage into a V partition')
    c['engagement_rows'] = len(fix_rows)
    c['engagement_rows_expected'] = EXPECTED_ENGAGEMENT_ROWS
    c['engagement_row_keys_unique'] = len({(r['match'], r['s']) for r in fix_rows}) == len(fix_rows)
    with np.load(V2FIX / 'states.npz', allow_pickle=False) as z:
        c['engagement_states_shape'] = {'pre': list(z['pre'].shape), 'post': list(z['post'].shape)}
        c['engagement_states_names_equal_model'] = z['names'].tolist() == names and str(z['state_version']) == I['p1_proto']['state_version']
    if len(fix_rows) != EXPECTED_ENGAGEMENT_ROWS or not c['engagement_row_keys_unique'] or not c['engagement_states_names_equal_model']:
        fail.append('state_value_v2_fix engagement rows/states differ from expectation')
    sampled = I['sampled']
    by_role = defaultdict(list)
    for r in sampled:
        by_role[r['role']].append(r)
    c['sampled_rows'] = len(sampled)
    c['sampled_by_role'] = {p: {'rows': len(by_role[p]), 'matches': len({r['match'] for r in by_role[p]}),
                                'match_set_equals_split': {r['match'] for r in by_role[p]} == sets[p]} for p in PARTITIONS}
    keys = [(r['match'], int(r['time_ms'])) for r in sampled]
    c['sampled_duplicate_keys'] = len(keys) - len(set(keys))
    c['test_rows_one_per_match'] = set(Counter(r['match'] for r in by_role['test']).values()) == {1}
    if set(by_role) != set(PARTITIONS) or c['sampled_duplicate_keys'] or not c['test_rows_one_per_match'] or not all(
            v['match_set_equals_split'] for v in c['sampled_by_role'].values()):
        fail.append('sampled query keys inconsistent with splits')
    with np.load(P1 / 'states' / 'sampled_states_v2.npz', allow_pickle=False) as z:
        c['p1_sampled_keys_equal_source_order'] = (z['match'].tolist() == [r['match'] for r in sampled] and
                                                   z['time_ms'].tolist() == [int(r['time_ms']) for r in sampled] and
                                                   z['role'].tolist() == [r['role'] for r in sampled])
        c['p1_state_names_equal_model'] = z['names'].tolist() == names
    cv = I['curves']
    gk = list(zip(cv['match'].tolist(), cv['time_ms'].astype(np.int64).tolist()))
    c['grid_rows'] = len(gk)
    c['grid_duplicate_keys'] = len(gk) - len(set(gk))
    c['grid_match_set_equals_test'] = set(cv['match'].tolist()) == sets['test']
    with np.load(P1 / 'states' / 'test_grid_states_v2.npz', allow_pickle=False) as z:
        c['p1_grid_keys_equal_source'] = bool(np.array_equal(z['match'], cv['match']) and np.array_equal(z['time_ms'], cv['time_ms']))
    if not (c['p1_sampled_keys_equal_source_order'] and c['p1_state_names_equal_model'] and c['p1_grid_keys_equal_source']
            and c['grid_match_set_equals_test'] and not c['grid_duplicate_keys']):
        fail.append('P1 state keys differ from the stored protocol keys')
    c['candidate_A_sha256'] = sha256_file(P1 / 'models' / 'expanded_model_v2.joblib')
    c['candidate_A_sha256_equals_p1_record'] = (c['candidate_A_sha256'] == I['p1_info']['model_sha256']['expanded']
                                                == I['p1_res']['model_sha256']['expanded'])
    c['state_names_equal_p1_protocol'] = I['p1_proto']['state_schema']['names'] == names
    if not (c['candidate_A_sha256_equals_p1_record'] and c['state_names_equal_p1_protocol']):
        fail.append('frozen candidate A hash or schema differs from P1 record')
    return c, fail, fix_rows


def select_subsets(I, fix_rows, smoke):
    splits = I['old_proto']['splits']
    if smoke:
        keep = set()
        for p in PARTITIONS:
            keep.update(splits[p][:smoke])
    else:
        keep = set().union(*(set(splits[p]) for p in PARTITIONS))
    rows = [dict(r, row_index=i) for i, r in enumerate(I['sampled']) if r['match'] in keep]
    grid_index = np.flatnonzero(np.isin(I['curves']['match'], sorted(keep & set(splits['test']))))
    if smoke:
        eng_matches = list(dict.fromkeys(r['match'] for r in fix_rows))[:max(10, smoke)]
        eng = [dict(r, row_index=i) for i, r in enumerate(fix_rows) if r['match'] in set(eng_matches)]
    else:
        eng = [dict(r, row_index=i) for i, r in enumerate(fix_rows)]
    test_keep = [m for m in splits['test'] if m in keep]
    traj = sorted(test_keep, key=lambda m: _rank('p2_trajectory', m))[:(2 if smoke else TRAJECTORY_MATCHES)]
    frame = [m for m in sorted(test_keep, key=lambda m: _rank('p2_frame_audit', m)) if m not in traj][:(2 if smoke else FRAME_AUDIT_MATCHES)]
    replay = {p: sorted([m for m in splits[p] if m in keep], key=lambda m: _rank('p2_replay_audit', m))[:(2 if smoke else REPLAY_AUDIT_PER_PARTITION)]
              for p in PARTITIONS}
    return keep, rows, grid_index, eng, traj, frame, replay


def build_plan(splits, rows, curves, grid_index, keep):
    queries = defaultdict(set)
    for r in rows:
        queries[r['match']].add(int(r['time_ms']))
    for m, t in zip(curves['match'][grid_index].tolist(), curves['time_ms'][grid_index].astype(np.int64).tolist()):
        queries[m].add(int(t))
    times = {m: sorted({h for q in qs for h in (q - (8 - k) * 60000 for k in range(9)) if h >= 0}) for m, qs in queries.items()}
    order = [m for p in PARTITIONS for m in splits[p] if m in times]
    if len(order) != len(times) or any(m not in keep for m in order):
        raise ValueError('planned matches outside the V partitions')
    return [(m, times[m]) for m in order]


# ----------------------------------------------------------------------------- extraction and assembly
def run_extraction(run, plan, names, workers, chunk_size):
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
    run.stage('extraction', chunks_total=len(starts), chunks_reused=reused, chunks_to_build=len(tasks), workers=workers)
    results, failures = run_pool(run, extract_chunk, tasks, workers, 'extraction', 'history states')
    errors = dict(failures)
    for s in results:
        errors.update(s['errors'])
    run.log(f"extraction: built {sum(s['written'] for s in results)} chunks, rows {sum(s['rows'] for s in results)}, "
            f"reused {reused}, errors {len(errors)}")
    if errors:
        write_json(run.out / 'errors.json', {'stage': 'extraction', 'match_or_chunk_errors': errors,
                                             'note': 'no state was skipped; failing chunks were not checkpointed'})
        raise RuntimeError(f'extraction failed for {len(errors)} matches/chunks; see errors.json')
    return len(starts)


def sequence_arrays(q_match, q_time, support, key_index):
    q_time = np.asarray(q_time, dtype=np.int64)
    times = q_time[:, None] - (8 - np.arange(9)) * 60000
    sup = np.asarray([support[m] for m in q_match.tolist()], dtype=np.int64)
    mask = times >= sup[:, None]
    index = np.full(times.shape, -1, dtype=np.int64)
    for i, m in enumerate(q_match.tolist()):
        for k in range(9):
            if mask[i, k]:
                index[i, k] = key_index[(m, int(times[i, k]))]
    return index, mask, times


def assemble(run, n_chunks, names, splits, rows, curves, grid_index, plan):
    import train.temporal_history_winprob_v3 as T
    from gameplay.state_value_v2 import STATE_VERSION
    role_of = {m: p for p in PARTITIONS for m in splits[p]}
    parts = {p: defaultdict(list) for p in PARTITIONS}
    per = defaultdict(list)
    for c in range(n_chunks):
        with np.load(run.out / 'states' / 'chunks' / f'chunk_{c:04d}.npz', allow_pickle=False) as z:
            if str(z['state_version']) != STATE_VERSION or str(z['names_sha256']) != sha256_json(names):
                raise ValueError(f'chunk {c} state version/schema mismatch')
            roles = np.asarray([role_of[m] for m in z['match'].tolist()])
            for p in PARTITIONS:
                sel = roles == p
                if sel.any():
                    for k in ('match', 'time_ms', 'snapshot_ms', 'unassigned_soul_events', 'X'):
                        parts[p][k].append(z[k][sel])
            for k in ('match_ids', 'winner', 'terminal_ms', 'last_frame_ms', 'support_start_ms', 'unsupported_times', 'patch'):
                per[k].append(z[k])
    P = {k: np.concatenate(v) for k, v in per.items()}
    table = {m: {'winner_blue': int(w), 'terminal_ms': int(te), 'last_frame_ms': int(lf), 'support_start_ms': int(ss),
                 'unsupported_planned_times': int(un), 'patch': str(pa), 'partition': role_of[m]}
             for m, w, te, lf, ss, un, pa in zip(P['match_ids'].tolist(), P['winner'], P['terminal_ms'], P['last_frame_ms'],
                                                 P['support_start_ms'], P['unsupported_times'], P['patch'].tolist())}
    if len(table) != len(plan):
        raise ValueError('built match count differs from plan')
    support = {m: v['support_start_ms'] for m, v in table.items()}
    tables, checks = {}, {'planned_matches': len(plan), 'planned_times': sum(len(t) for _, t in plan)}
    for p in PARTITIONS:
        if not parts[p]:
            continue
        tables[p] = {k: np.concatenate(v) for k, v in parts[p].items()}
    checks['built_states_by_partition'] = {p: int(len(t['time_ms'])) for p, t in tables.items()}
    checks['unsupported_planned_times'] = int(P['unsupported_times'].sum())
    checks['support_start_ms_values'] = dict(Counter(int(v) for v in P['support_start_ms']))
    keymaps = {}
    for p, t in tables.items():
        km = {}
        for i, k in enumerate(zip(t['match'].tolist(), t['time_ms'].tolist())):
            if k in km:
                raise ValueError(f'duplicate state key {k}')
            km[k] = i
        keymaps[p] = km
    seqs = {}
    rows_by_role = defaultdict(list)
    for r in rows:
        rows_by_role[r['role']].append(r)
    for p in PARTITIONS:
        rr = rows_by_role[p]
        qm = np.asarray([r['match'] for r in rr], dtype='U16')
        qt = np.asarray([int(r['time_ms']) for r in rr], dtype=np.int64)
        index, mask, times = sequence_arrays(qm, qt, support, keymaps[p])
        seqs[p] = dict(row_index=np.asarray([r['row_index'] for r in rr], dtype=np.int64), match=qm, time_ms=qt,
                       winner=np.asarray([table[m]['winner_blue'] for m in qm.tolist()], dtype=np.int64),
                       index=index, mask=mask, times=times)
    gm = curves['match'][grid_index].astype('U16')
    gt = curves['time_ms'][grid_index].astype(np.int64)
    index, mask, times = sequence_arrays(gm, gt, support, keymaps['test'])
    seqs['test_grid'] = dict(grid_index=np.asarray(grid_index, dtype=np.int64), match=gm, time_ms=gt,
                             winner=np.asarray([table[m]['winner_blue'] for m in gm.tolist()], dtype=np.int64),
                             index=index, mask=mask, times=times)
    checks['membership'] = {}
    checks['query_validity_violations'] = {}
    for name, s in seqs.items():
        p = 'test' if name == 'test_grid' else name
        t = tables[p]
        checks['membership'][name] = T.check_sequence_membership(t['match'], t['time_ms'], s['index'], s['mask'],
                                                                 s['times'], s['match'], s['time_ms'])
        bad = sum(not (table[m]['support_start_ms'] <= q <= table[m]['last_frame_ms'] and q < table[m]['terminal_ms']
                       and table[m]['partition'] == p) for m, q in zip(s['match'].tolist(), s['time_ms'].tolist()))
        checks['query_validity_violations'][name] = int(bad)
        checks.setdefault('rows', {})[name] = int(len(s['time_ms']))
        checks.setdefault('masked_positions', {})[name] = int((~s['mask']).sum())
        checks.setdefault('rows_with_any_masked_position', {})[name] = int((~s['mask']).any(axis=1).sum())
    # Query-position states must be bitwise equal to P1's rebuilt states.
    with np.load(P1 / 'states' / 'sampled_states_v2.npz', allow_pickle=False) as z:
        p1 = {k: z[k] for k in ('match', 'role', 'time_ms', 'snapshot_ms', 'unassigned_soul_events', 'winner', 'X')}
    eq = {}
    for p in PARTITIONS:
        s, t = seqs[p], tables[p]
        li = s['index'][:, -1]
        pi = s['row_index']
        eq[p] = bool(np.array_equal(p1['match'][pi], s['match']) and np.array_equal(p1['time_ms'][pi], s['time_ms'])
                     and np.array_equal(p1['X'][pi], t['X'][li]) and np.array_equal(p1['snapshot_ms'][pi], t['snapshot_ms'][li])
                     and np.array_equal(p1['unassigned_soul_events'][pi], t['unassigned_soul_events'][li])
                     and np.array_equal(p1['winner'][pi], s['winner']))
    p1 = None
    with np.load(P1 / 'states' / 'test_grid_states_v2.npz', allow_pickle=False) as z:
        s, t = seqs['test_grid'], tables['test']
        gi, li = s['grid_index'], s['index'][:, -1]
        eq['test_grid'] = bool(np.array_equal(z['match'][gi], s['match']) and np.array_equal(z['time_ms'][gi], s['time_ms'])
                               and np.array_equal(z['X'][gi], t['X'][li]) and np.array_equal(z['snapshot_ms'][gi], t['snapshot_ms'][li])
                               and np.array_equal(z['unassigned_soul_events'][gi], t['unassigned_soul_events'][li])
                               and np.array_equal(z['winner'][gi], s['winner']))
    checks['query_states_bitwise_equal_p1'] = eq
    checks['states_finite'] = bool(all(np.isfinite(t['X']).all() for t in tables.values()))
    checks['snapshot_not_after_position'] = bool(all((t['snapshot_ms'] <= t['time_ms']).all() for t in tables.values()))
    fails = [n for n, m in checks['membership'].items() if m['violations']]
    fails += [n for n, v in checks['query_validity_violations'].items() if v]
    fails += [n for n, v in eq.items() if not v]
    if fails or not checks['states_finite'] or not checks['snapshot_not_after_position'] or checks['unsupported_planned_times']:
        write_json(run.out / 'states' / 'assembly_checks.json', checks)
        raise ValueError(f'assembly checks failed: {fails}')
    for p, t in tables.items():
        save_npz(run.out / 'states' / f'state_table_{p}.npz', **t, names=np.asarray(names), state_version=np.asarray(STATE_VERSION))
    for name, s in seqs.items():
        save_npz(run.out / 'states' / f'sequences_{name}.npz', **s, history_version=np.asarray(T.HISTORY_VERSION))
    write_json(run.out / 'states' / 'match_table.json', table)
    checks['sequence_file_sha256'] = {n: sha256_file(run.out / 'states' / f'sequences_{n}.npz') for n in seqs}
    write_json(run.out / 'states' / 'assembly_checks.json', checks)
    return tables, seqs, table, checks


# ----------------------------------------------------------------------------- candidates and selection
def gather_encoded(Z, C, index, mask):
    pad = len(Z)
    Zp = np.vstack([Z, np.zeros((1, Z.shape[1]))])
    Cp = np.vstack([C, np.full((1, C.shape[1]), -1, dtype=C.dtype)])
    idx = np.where(mask, index, pad)
    return Zp, Cp, idx


def b_table_predict(fn, Z, C, index, mask, batch=4096):
    """Apply fn(Xnum, C, mask) over gathered encoded histories in fixed batches."""
    Zp, Cp, idx = gather_encoded(Z, C, index, mask)
    outs = []
    for s in range(0, len(idx), batch):
        outs.append(fn(Zp[idx[s:s + batch]], Cp[idx[s:s + batch]], mask[s:s + batch]))
    return outs


def candidate_a_select(run, names, tables, seqs, I, smoke):
    import train.temporal_history_winprob_v3 as T
    from gameplay.state_value_v2 import STATE_VERSION
    from temporal_winprob_v3_eval import cell_metrics
    from train.state_value_experiment import metrics
    A = T.CandidateA.load(P1 / 'models' / 'expanded_model_v2.joblib', I['p1_info']['model_sha256']['expanded'])
    s = seqs['select']
    X_last = tables['select']['X'][s['index'][:, -1]]
    p = A.predict_last_matrix(X_last, names, STATE_VERSION)
    with np.load(P1 / 'predictions' / 'select_partition_candidates_v2.npz', allow_pickle=False) as z:
        key = {k: i for i, k in enumerate(zip(z['match'].tolist(), z['time_ms'].tolist()))}
        ix = np.asarray([key[k] for k in zip(s['match'].tolist(), s['time_ms'].tolist())])
        stored = z[I['p1_res']['chosen']['expanded']][ix]
    m = metrics(s['winner'], p, s['match'])
    info = {'candidate': T.CANDIDATE_A, 'model_version': A.candidate_version, 'calibration': A.calibration,
            'model_sha256': A.model_sha256, 'select_metrics': m, 'select_cell': cell_metrics(s['winner'], p, s['match']),
            'select_predictions_bitwise_equal_p1_stored': bool(np.array_equal(p, stored)),
            'select_log_loss_equals_p1_selection': None if smoke else bool(
                abs(m['log_loss'] - I['p1_sel']['scores'][I['p1_res']['chosen']['expanded']]['log_loss']) < 1e-12)}
    if not info['select_predictions_bitwise_equal_p1_stored'] or info['select_log_loss_equals_p1_selection'] is False:
        raise ValueError('frozen candidate A does not reproduce P1 select predictions')
    run.log(f"candidate A select log loss {m['log_loss']:.6f} (P1-equal: {info['select_predictions_bitwise_equal_p1_stored']})")
    return A, p, info


def train_candidate_b(run, names, tables, seqs, smoke):
    import train.temporal_history_winprob_v3 as T
    from train.state_value_experiment import metrics
    from temporal_winprob_v3_eval import cell_metrics
    model_dir = run.out / 'models'
    (model_dir / 'training').mkdir(parents=True, exist_ok=True)
    fit_X = tables['fit']['X']
    pre = T.HistoryPreprocessor.fit(names, fit_X)
    enc = {}
    for p in ('fit', 'calibrate', 'select'):
        enc[p] = pre.encode_states(tables[p]['X'])
    unseen = {}
    for p in ('fit', 'calibrate', 'select'):
        Cv = enc[p][1]
        s = seqs[p]
        pos = s['index'][s['mask']]
        unseen[p] = {'states_with_unseen_champion': int((Cv == -1).any(axis=1).sum()), 'states': int(len(Cv)),
                     'query_rows_with_unseen_champion_at_query': int((Cv[s['index'][:, -1]] == -1).any(axis=1).sum()),
                     'valid_history_positions_with_unseen': int((Cv[pos] == -1).any(axis=1).sum())}
    fit_origin = {'partition': 'fit', 'unique_states': int(len(fit_X)), 'matches': int(len(set(tables['fit']['match'].tolist()))),
                  'rule': 'mean/std over every unique FIT state used by any FIT history position (states at the '
                          'query and at earlier one-minute positions); champion vocabularies from the same states',
                  'fit_states_sha256': hashlib.sha256(np.ascontiguousarray(fit_X).tobytes()).hexdigest(),
                  'n_numeric': pre.n_numeric, 'n_champion_columns': pre.n_champion_columns, 'input_width': pre.input_width,
                  'zero_variance_numeric': [n for n, sc in zip(pre.numeric_names, pre.numeric_scale) if sc == 1. and
                                            np.all(fit_X[:, names.index(n)] == fit_X[0, names.index(n)])],
                  'champion_categories_per_slot': [len(v) for v in pre.champion_vocab],
                  'unseen_champion_behaviour': 'ID absent from the FIT category list of that participant slot contributes '
                                               'a zero one-hot vector (like OneHotEncoder(handle_unknown="ignore") in A)',
                  'unseen_counts': unseen}
    s = seqs['fit']
    config = {'architecture': T.ARCHITECTURE, 'epochs': EPOCHS, 'batch_size': BATCH, 'optimizer': 'RMSprop',
              'lr': LR, 'rho': RHO, 'eps': EPS, 'loss': 'binary cross entropy (with logits, mean over batch)',
              'sample_weight': 'none (one sample per FIT sampled query row, as P1 final fit)',
              'shuffle': 'numpy default_rng(seed).permutation per epoch', 'torch_threads': 4,
              'fit_rows': int(len(s['winner'])), 'fit_states_sha256': fit_origin['fit_states_sha256'],
              'sequence_index_sha256': hashlib.sha256(np.ascontiguousarray(s['index']).tobytes()).hexdigest(),
              'preprocessor_sha256': sha256_json({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in pre.arrays().items()})}
    import torch
    config['torch_version'] = torch.__version__
    params, logs, parity = {}, {}, {}
    sel = seqs['select']
    for seed in SEEDS:
        cfg_sha = sha256_json(dict(config, seed=seed))
        ck = model_dir / 'training' / f'seed_{seed}.npz'
        if ck.exists():
            with np.load(ck, allow_pickle=False) as z:
                if str(z['config_sha256']) == cfg_sha:
                    params[seed] = {k: z[k].copy() for k in T.PARAM_NAMES}
                    logs[seed] = read_json(model_dir / 'training' / f'seed_{seed}_log.json')
                    parity[seed] = logs[seed]['torch_numpy_parity']
                    run.log(f'B seed {seed}: reused training checkpoint {ck.name}')
                    continue
            raise ValueError(f'stale training checkpoint {ck.name}; use a new output directory')
        run.log(f'B seed {seed}: training {EPOCHS} epochs on {len(s["winner"])} FIT rows (CPU, 4 threads)')

        def on_epoch(rec, seed=seed):
            run.stage('training_B', quiet=rec['epoch'] % 10 != 0, seed=seed, epoch=rec['epoch'], epochs=EPOCHS,
                      train_bce_with_dropout=round(rec['train_bce_with_dropout'], 5), epoch_seconds=rec['seconds'])
            with (model_dir / 'training' / f'seed_{seed}_epochs.log').open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
        started = time.time()
        pr, hist, eval_torch = T.train_simplernn(enc['fit'][0], enc['fit'][1], s['index'], s['mask'], s['winner'],
                                                 pre.input_width, seed=seed, units=UNITS, dropout=DROPOUT, lr=LR,
                                                 batch_size=BATCH, epochs=EPOCHS, rho=RHO, eps=EPS, threads=4,
                                                 on_epoch=on_epoch)
        n = min(4096, len(sel['winner']))
        Zp, Cp, idx = gather_encoded(*enc['select'], sel['index'][:n], sel['mask'][:n])
        p_np, _ = T.simplernn_forward(pr, Zp[idx], Cp[idx], sel['mask'][:n])
        parity[seed] = {'rows': n, 'max_abs_diff_numpy_float64_vs_torch_float32': float(np.max(np.abs(p_np - eval_torch(Zp[idx], Cp[idx], sel['mask'][:n]))))}
        if parity[seed]['max_abs_diff_numpy_float64_vs_torch_float32'] > 1e-5:
            raise ValueError(f'numpy inference differs from torch forward for seed {seed}: {parity[seed]}')
        params[seed], logs[seed] = pr, {'seed': seed, 'config': config, 'config_sha256': cfg_sha, 'epochs': hist,
                                        'seconds': round(time.time() - started, 1), 'torch_numpy_parity': parity[seed]}
        save_npz(ck, **pr, config_sha256=np.asarray(cfg_sha))
        write_json(model_dir / 'training' / f'seed_{seed}_log.json', logs[seed])
    run.stage('calibration_B')
    raw = {p: {} for p in ('calibrate', 'select')}
    for p in raw:
        for seed in SEEDS:
            raw[p][seed] = np.concatenate([o[0] for o in b_table_predict(
                lambda X, C, M, pr=params[seed]: T.simplernn_forward(pr, X, C, M), *enc[p], seqs[p]['index'], seqs[p]['mask'])])
    cal_s, sel_s = seqs['calibrate'], seqs['select']
    members, member_info, sel_member_p = [], {}, {}
    for seed in SEEDS:
        specs, _, check = T.fit_calibrations(raw['calibrate'][seed], cal_s['winner'])
        if max(check.values()) > 1e-12:
            raise ValueError(f'numpy calibration differs from sklearn for seed {seed}: {check}')
        variants = {}
        for method in T.CALIBRATIONS:
            p = T.apply_calibration(specs[method], raw['select'][seed])
            variants[method] = {'select_metrics': metrics(sel_s['winner'], p, sel_s['match']),
                                'select_cell': cell_metrics(sel_s['winner'], p, sel_s['match'])}
        chosen = min(T.CALIBRATIONS, key=lambda k: variants[k]['select_metrics']['log_loss'])
        members.append({'seed': seed, 'params': params[seed], 'calibration': specs[chosen]})
        sel_member_p[seed] = T.apply_calibration(specs[chosen], raw['select'][seed])
        member_info[seed] = {'variants': variants, 'chosen_calibration': chosen, 'numpy_vs_sklearn_calibration_max_abs': check,
                             'calibration_fit_partition': 'calibrate', 'calibration_fit_rows': int(len(cal_s['winner'])),
                             'sigmoid': {k: specs['sigmoid'][k] for k in ('sigmoid_coef', 'sigmoid_intercept')},
                             'isotonic_thresholds': int(len(specs['isotonic']['isotonic_x']))}
        run.log(f'B seed {seed}: select log loss ' + ', '.join(
            f"{k}={variants[k]['select_metrics']['log_loss']:.5f}" for k in T.CALIBRATIONS) + f' -> {chosen}')
    B = T.CandidateB(pre, members)
    p_before = np.mean(np.stack([sel_member_p[s] for s in SEEDS]), axis=0)
    manifest = B.save(model_dir / 'candidate_B')
    B = T.CandidateB.load(model_dir / 'candidate_B', expected_bundle_sha256=manifest['bundle_sha256'])
    outs = b_table_predict(lambda X, C, M: B.predict_encoded(X, C, M), *enc['select'], sel_s['index'], sel_s['mask'])
    p_after = np.concatenate([o[0] for o in outs])
    members_after = {s: np.concatenate([o[1][s] for o in outs]) for s in SEEDS}
    identity = {'ensemble_bitwise_equal_after_save_load': bool(np.array_equal(p_before, p_after)),
                'members_bitwise_equal_after_save_load': bool(all(np.array_equal(sel_member_p[s], members_after[s]) for s in SEEDS))}
    if not all(identity.values()):
        raise ValueError(f'candidate B save/load changed predictions: {identity}')
    m = metrics(sel_s['winner'], p_after, sel_s['match'])
    run.log(f"B primary (mean of 3 calibrated seeds) select log loss {m['log_loss']:.6f}; bundle {manifest['bundle_sha256'][:12]}")
    info = {'candidate': T.CANDIDATE_B, 'model_version': T.CANDIDATE_B_VERSION, 'bundle_sha256': manifest['bundle_sha256'],
            'calibration': B.calibration, 'members': member_info, 'select_metrics': m,
            'select_cell': cell_metrics(sel_s['winner'], p_after, sel_s['match']), 'save_load_identity': identity,
            'torch_numpy_parity': parity, 'training_seconds': {s: logs[s].get('seconds') for s in SEEDS},
            'final_epoch_train_bce': {s: logs[s]['epochs'][-1]['train_bce_with_dropout'] for s in SEEDS},
            'preprocessing_fit_origin': fit_origin, 'config': config, 'manifest': manifest}
    enc = None
    return B, p_after, members_after, info


def write_selection(run, A_info, B_info, b_blocker):
    import train.temporal_history_winprob_v3 as T
    pred_dir = run.out / 'predictions'
    existing = sorted(p.name for p in pred_dir.glob('test*')) if pred_dir.exists() else []
    if existing:
        raise ValueError(f'test prediction files exist before selection: {existing}')
    a_ll = A_info['select_metrics']['log_loss']
    b_ll = B_info['select_metrics']['log_loss'] if B_info else None
    chosen = 'B' if (b_ll is not None and b_ll < a_ll) else 'A'
    sel = {'rule': 'lower SELECT-partition match-weighted log loss between A (frozen P1 expanded adapter) and B primary '
                   '(arithmetic mean of the three per-seed calibrated SimpleRNNs); an exact tie keeps A; B unavailable keeps A',
           'metric_function': 'train.state_value_experiment.metrics (match-weighted, as P1)',
           'written_before_any_test_prediction': True, 'test_prediction_files_at_selection': existing,
           'A': {k: A_info[k] for k in ('candidate', 'model_version', 'calibration', 'model_sha256', 'select_metrics',
                                        'select_predictions_bitwise_equal_p1_stored')},
           'B_available': B_info is not None, 'B_blocker': b_blocker,
           'B': ({k: B_info[k] for k in ('candidate', 'model_version', 'bundle_sha256', 'calibration', 'select_metrics')}
                 | {'members': {str(s): {'chosen_calibration': v['chosen_calibration'],
                                         'select_log_loss': {m: v['variants'][m]['select_metrics']['log_loss'] for m in T.CALIBRATIONS}}
                                for s, v in B_info['members'].items()}}) if B_info else None,
           'select_log_loss': {'A': a_ll, 'B_primary': b_ll, 'B_minus_A': (b_ll - a_ll) if b_ll is not None else None},
           'chosen_candidate': chosen, 'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'caveats': ['exploratory: the historical test partition was examined in earlier work (P1/V1)',
                       "B's per-seed calibration and the A-vs-B decision both use SELECT, as did P1's calibration choice for A",
                       'selection decides the frozen P2 candidate; TEST results for both A and B are reported regardless']}
    write_json(run.out / 'selection.json', sel)
    frozen = {'chosen_candidate': chosen, 'selection_json_sha256': sha256_file(run.out / 'selection.json'),
              'A': {'path': str(P1 / 'models' / 'expanded_model_v2.joblib'), 'sha256': A_info['model_sha256'],
                    'model_version': A_info['model_version'], 'calibration': A_info['calibration']},
              'B': {'bundle_dir': str(run.out / 'models' / 'candidate_B'), 'bundle_sha256': B_info['bundle_sha256'],
                    'model_version': B_info['model_version'], 'calibration': B_info['calibration']} if B_info else None,
              'history_version': T.HISTORY_VERSION, 'module_version': T.MODULE_VERSION,
              'module_sha256': sha256_file(WT / 'train' / 'temporal_history_winprob_v3.py')}
    write_json(run.out / 'models' / 'frozen_models.json', frozen)
    run.log(f"selection frozen: chosen={chosen}; A select ll={a_ll:.6f}; B primary select ll={b_ll}")
    return sel, frozen


def frozen_task_fields(frozen):
    f = {'a_path': frozen['A']['path'], 'a_sha256': frozen['A']['sha256']}
    if frozen.get('B'):
        f.update(b_dir=frozen['B']['bundle_dir'], b_sha256=frozen['B']['bundle_sha256'])
    return f


# ----------------------------------------------------------------------------- test evaluation (after selection)
def test_evaluation(run, names, tables, seqs, frozen, I, n_boot_primary, n_boot_expl, smoke):
    import train.temporal_history_winprob_v3 as T
    from gameplay.state_value_v2 import STATE_VERSION
    import temporal_winprob_v3_eval as E
    sel = read_json(run.out / 'selection.json')
    if sha256_file(run.out / 'selection.json') != frozen['selection_json_sha256'] or sel['chosen_candidate'] != frozen['chosen_candidate']:
        raise ValueError('selection.json changed after freezing')
    A = T.CandidateA.load(frozen['A']['path'], frozen['A']['sha256'])
    B = T.CandidateB.load(frozen['B']['bundle_dir'], frozen['B']['bundle_sha256']) if frozen.get('B') else None
    t = tables['test']
    Z, C = (B.preprocessor.encode_states(t['X']) if B else (None, None))
    preds, checks = {}, {}
    for name in ('test', 'test_grid'):
        s = seqs[name]
        X_last = t['X'][s['index'][:, -1]]
        cell = {'A': A.predict_last_matrix(X_last, names, STATE_VERSION)}
        if B:
            outs = b_table_predict(lambda X, Cc, M: B.predict_encoded(X, Cc, M), Z, C, s['index'], s['mask'])
            cell['B'] = np.concatenate([o[0] for o in outs])
            for seed in SEEDS:
                cell[f'B_seed{seed}'] = np.concatenate([o[1][seed] for o in outs])
        cell['snapshot_ms'] = t['snapshot_ms'][s['index'][:, -1]]
        cell['unassigned'] = t['unassigned_soul_events'][s['index'][:, -1]]
        cell['X_last'] = X_last
        preds[name] = cell
    with np.load(P1 / 'predictions' / 'test_one_minute_v1_v2.npz', allow_pickle=False) as z:
        key = {k: i for i, k in enumerate(zip(z['match'].tolist(), z['time_ms'].tolist()))}
        ix = np.asarray([key[k] for k in zip(seqs['test']['match'].tolist(), seqs['test']['time_ms'].tolist())])
        checks['A_one_minute_bitwise_equal_p1_stored'] = bool(np.array_equal(z['v2_expanded'][ix], preds['test']['A']))
    with np.load(P1 / 'predictions' / 'test_grid_v1_v2.npz', allow_pickle=False) as z:
        gi = seqs['test_grid']['grid_index']
        checks['A_grid_bitwise_equal_p1_stored'] = bool(np.array_equal(z['v2_expanded'][gi], preds['test_grid']['A']) and
                                                        np.array_equal(z['match'][gi], seqs['test_grid']['match']))
    checks['predictions_finite_in_unit_interval'] = bool(all(
        np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() for c in preds.values() for k, v in c.items()
        if k == 'A' or k.startswith('B')))
    pred_dir = run.out / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_cols = [k for k in preds['test'] if k == 'A' or k.startswith('B')]
    s = seqs['test']
    with (pred_dir / 'test_one_minute_A_B.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['match', 'time_ms', 'winner_blue', 'snapshot_ms', 'snapshot_age_s', 'valid_history_positions',
                    *[f'p_{k}' for k in model_cols], 'model_A_sha256', 'model_B_bundle_sha256', 'chosen_candidate'])
        for i in range(len(s['winner'])):
            w.writerow([s['match'][i], int(s['time_ms'][i]), int(s['winner'][i]), int(preds['test']['snapshot_ms'][i]),
                        (int(s['time_ms'][i]) - int(preds['test']['snapshot_ms'][i])) / 1000., int(s['mask'][i].sum()),
                        *[repr(float(preds['test'][k][i])) for k in model_cols], frozen['A']['sha256'],
                        frozen['B']['bundle_sha256'] if frozen.get('B') else '', frozen['chosen_candidate']])
    g = seqs['test_grid']
    meta = dict(model_A_sha256=np.asarray(frozen['A']['sha256']), chosen_candidate=np.asarray(frozen['chosen_candidate']),
                model_B_bundle_sha256=np.asarray(frozen['B']['bundle_sha256'] if frozen.get('B') else ''),
                history_version=np.asarray(T.HISTORY_VERSION), state_version=np.asarray(STATE_VERSION))
    save_npz(pred_dir / 'test_one_minute_A_B.npz', match=s['match'], time_ms=s['time_ms'], winner_blue=s['winner'],
             snapshot_ms=preds['test']['snapshot_ms'], history_mask=s['mask'],
             **{f'p_{k}': preds['test'][k] for k in model_cols}, **meta)
    save_npz(pred_dir / 'test_grid_A_B.npz', match=g['match'], time_ms=g['time_ms'], winner_blue=g['winner'],
             grid_index=g['grid_index'], snapshot_ms=preds['test_grid']['snapshot_ms'], history_mask=g['mask'],
             **{f'p_{k}': preds['test_grid'][k] for k in model_cols}, **meta)
    run.stage('test_evaluation', part='overall')
    res = {'checks': checks, 'chosen_candidate': frozen['chosen_candidate'],
           'weights': 'equal total weight per match within each cell',
           'na_policy': {'min_cell_matches': E.MIN_CELL_MATCHES, 'min_class_matches': E.MIN_CLASS_MATCHES,
                         'rule': 'fewer matches than min_cell_matches -> all metrics NA; fewer than min_class_matches matches '
                                 'in either outcome class -> AUC and calibration intercept/slope NA'}}
    one = {k: preds['test'][k] for k in model_cols}
    res['test_one_minute'] = E.compare_cell(s['winner'], s['match'], one, n_boot_primary if B else 0, BOOT_SEED)
    grid = {k: preds['test_grid'][k] for k in model_cols}
    gy, gm, gt = g['winner'], g['match'], g['time_ms']
    res['test_grid_overall'] = E.compare_cell(gy, gm, grid, n_boot_primary if B else 0, BOOT_SEED)
    if B:
        res['agreement_B_vs_A'] = {'test_one_minute': E.agreement(one['B'], one['A']), 'test_grid': E.agreement(grid['B'], grid['A'])}
        res['primary_bootstrap_label'] = f'B primary minus A, {n_boot_primary} paired match bootstrap replicates'
    run.log('test one-minute: ' + '; '.join(f"{k} ll={_f(res['test_one_minute'][k]['log_loss'])} auc={_f(res['test_one_minute'][k]['auc'])}"
                                            for k in model_cols))
    run.stage('test_evaluation', part='time_bands')
    res['test_grid_time_bands'] = {}
    for lo, hi in E.TIME_BANDS:
        mask = (gt >= lo * 60000) & (gt < hi * 60000)
        res['test_grid_time_bands'][E.band_label(lo, hi)] = E.compare_cell(
            gy[mask], gm[mask], {k: v[mask] for k, v in grid.items()}, n_boot_expl if B else 0, BOOT_SEED)
    minute_rows, res['test_grid_one_minute_bands'] = [], {}
    max_minute = int(gt.max() // 60000)
    for minute in range(2, max_minute + 1):
        mask = (gt >= minute * 60000) & (gt < (minute + 1) * 60000)
        if not mask.any():
            continue
        cell = E.compare_cell(gy[mask], gm[mask], {k: grid[k][mask] for k in ('A', 'B') if k in grid},
                              n_boot_expl if B else 0, BOOT_SEED)
        res['test_grid_one_minute_bands'][str(minute)] = cell
        row = {'minute': minute, 'rows': cell['rows'], 'matches': cell['matches'], 'status': cell['A']['status']}
        for k in ('A', 'B'):
            if k in cell:
                for metric in ('auc', 'brier', 'log_loss', 'ece', 'calibration_intercept', 'calibration_slope'):
                    row[f'{metric}_{k}'] = cell[k][metric]
        bs = cell.get('paired_bootstrap_B_minus_A')
        if bs and bs.get('status') == 'ok':
            for metric in ('auc', 'brier', 'log_loss'):
                row[f'{metric}_diff_B_minus_A'] = bs[metric]['point']
                row[f'{metric}_diff_ci_lo'], row[f'{metric}_diff_ci_hi'] = bs[metric]['ci95']
        minute_rows.append(row)
        if minute % 10 == 0:
            run.stage('test_evaluation', part='one_minute_bands', minute=minute, max_minute=max_minute)
    metrics_dir = run.out / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in minute_rows for k in r}, key=lambda k: (k not in ('minute', 'rows', 'matches', 'status'), k))
    with (metrics_dir / 'test_grid_one_minute_bands.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in minute_rows:
            w.writerow({k: ('NA' if r.get(k) is None else r.get(k)) for k in cols})
    E.plot_minute_bands(metrics_dir / 'test_grid_one_minute_bands.png', minute_rows)
    run.stage('test_evaluation', part='objective_strata')
    res['test_grid_objective_strata'] = {}
    for name, mask in E.objective_strata(preds['test_grid']['X_last'], names, preds['test_grid']['unassigned']).items():
        cell = E.compare_cell(gy[mask], gm[mask], {k: v[mask] for k, v in grid.items() if k in ('A', 'B')},
                              n_boot_expl if (B and mask.any()) else 0, BOOT_SEED)
        cell['rows_by_time_band'] = {E.band_label(lo, hi): int((mask & (gt >= lo * 60000) & (gt < hi * 60000)).sum())
                                     for lo, hi in E.TIME_BANDS}
        res['test_grid_objective_strata'][name] = cell
    ages = (gt - preds['test_grid']['snapshot_ms']) / 1000.
    res['test_grid_snapshot_age_s_quantiles'] = np.quantile(ages, [0, .01, .5, .99, 1]).tolist()
    res['history_masking'] = {n: {'rows': int(len(seqs[n]['mask'])), 'rows_with_masked_positions': int((~seqs[n]['mask']).any(axis=1).sum()),
                                  'valid_positions_distribution': dict(Counter(seqs[n]['mask'].sum(axis=1).tolist()))}
                              for n in ('test', 'test_grid')}
    write_json(metrics_dir / 'test_metrics.json', res)
    if not all(checks.values()):
        raise ValueError(f'test evaluation checks failed: {checks}')
    return res, preds


def _f(x, d=4):
    return 'NA' if x is None else f'{x:.{d}f}'


# ----------------------------------------------------------------------------- diagnostics (after selection)
def trajectories_stage(run, names, traj_matches, frozen, workers):
    import temporal_winprob_v3_eval as E
    tasks = [dict(frozen_task_fields(frozen), names=names, match=m, cadence_ms=TRAJECTORY_CADENCE_MS) for m in traj_matches]
    results, failures = run_pool(run, trajectory_task, tasks, workers, 'trajectories', 'per match')
    if failures:
        raise RuntimeError(f'trajectory task failures: {failures}')
    order = {m: i for i, m in enumerate(traj_matches)}
    results.sort(key=lambda r: order[r['match']])
    d = run.out / 'trajectories'
    d.mkdir(parents=True, exist_ok=True)
    has_b = frozen.get('B') is not None
    with (d / 'trajectory_points.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        seeds = sorted(results[0]['p_B_members']) if has_b else []
        w.writerow(['match', 'query_ms', 'time_minutes', 'queried_for', 'p_A', 'p_B', *[f'p_B_seed{s}' for s in seeds],
                    'snapshot_ms', 'snapshot_age_s', 'valid_history_positions', 'history_snapshot_ms'])
        for r in results:
            for i, q in enumerate(r['query_ms']):
                w.writerow([r['match'], q, q / 60000., r['reasons'][i], repr(r['p_A'][i]), repr(r['p_B'][i]) if has_b else '',
                            *[repr(r['p_B_members'][s][i]) for s in seeds], r['snapshot_ms'][i], r['snapshot_age_s'][i],
                            r['valid_positions'][i], ' '.join(map(str, r['history_snapshot_ms'][i]))])
    with (d / 'trajectory_events.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['match', 'ts_ms', 'type', 'category', 'team', 'detail', 'n_events_same_ts', 'n_predictor_events_same_ts', 'frame_at_ts'])
        for r in results:
            for e in r['events']:
                w.writerow([r['match'], e['ts'], e['type'], e['category'], e['team'], e['detail'], e['n_events_same_ts'],
                            e['n_predictor_events_same_ts'], e['frame_at_ts']])
    with (d / 'trajectory_frames.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['match', 'frame_index', 'frame_ts_ms', 'interval_since_previous_ms'])
        for r in results:
            for i, t in enumerate(r['frames_ms']):
                w.writerow([r['match'], i, t, t - r['frames_ms'][i - 1] if i else ''])
    note = ('p(final Blue win) evaluated at query times: 10 s cadence plus 1 ms before/at each raw frame and predictor event. '
            'Lines hold each evaluated value (no smoothing). Frames ~60 s apart; events have ms timestamps.')
    E.plot_trajectories_png(d / 'trajectories_12_matches.png', results, has_b)
    E.plot_trajectories_html(d / 'trajectories_12_matches.html', results, has_b, note)
    summary = {'matches': traj_matches, 'selection': 'first by sha256("p2_trajectory:"+match) among test matches; fixed in '
                                                     'protocol.json before training',
               'query_cadence_ms': TRAJECTORY_CADENCE_MS, 'min_query_ms': MIN_EVAL_QUERY_MS,
               'queries': {r['match']: len(r['query_ms']) for r in results},
               'winner_blue': {r['match']: r['winner_blue'] for r in results},
               'frame_interval_ms_quantiles': np.quantile(np.concatenate([np.diff(r['frames_ms']) for r in results]),
                                                          [0, .05, .5, .95, 1]).tolist(),
               'incremental_replay_equality': {r['match']: r['replay'] for r in results},
               'models': {'A': results[0]['model_A'], 'B': results[0]['model_B']},
               'resolution_statement': 'Frame-derived inputs (gold, level, XP, HP/MP, alive, CS, champion) update only at raw '
                                       'frame timestamps (~60 s cadence); event-derived inputs update at event timestamps '
                                       '(integer ms); time features and ages change with the query clock. Querying every '
                                       '10 s (or every 1 s) does not create 10 s (or 1 s) observations.'}
    write_json(d / 'trajectory_summary.json', summary)
    return summary


def endpoint_stage(run, names, eng_rows, frozen, workers, smoke):
    import temporal_winprob_v3_eval as E
    d = run.out / 'endpoint_diagnostics'
    (d / 'chunks').mkdir(parents=True, exist_ok=True)
    by_match = defaultdict(list)
    for r in eng_rows:
        by_match[r['match']].append(r)
    items = [(m, [{'row_index': r['row_index'], 's': int(r['s']), 'endpoint': int(r['endpoint'])} for r in rs])
             for m, rs in by_match.items()]
    size = 5 if smoke else 150
    tasks, reused = [], 0
    for c, s in enumerate(range(0, len(items), size)):
        chunk = items[s:s + size]
        plan_sha = sha256_json([names, chunk, frozen.get('A'), frozen.get('B')])
        path = d / 'chunks' / f'endpoint_{c:04d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z['plan_sha256']) != plan_sha:
                    raise ValueError(f'stale endpoint checkpoint {path.name}')
            reused += 1
            continue
        tasks.append(dict(frozen_task_fields(frozen), names=names, items=chunk, chunk_id=c, path=str(path), plan_sha256=plan_sha))
    n_chunks = len(range(0, len(items), size))
    run.stage('endpoint_diagnostics', chunks_total=n_chunks, chunks_reused=reused)
    results, failures = run_pool(run, endpoint_chunk, tasks, workers, 'endpoint_diagnostics', 'engagement rows')
    errors = dict(failures)
    for s in results:
        errors.update(s['errors'])
    if errors:
        write_json(d / 'errors_endpoint.json', errors)
        raise RuntimeError(f'endpoint diagnostics failed for {len(errors)} matches/chunks')
    parts = defaultdict(list)
    for c in range(n_chunks):
        with np.load(d / 'chunks' / f'endpoint_{c:04d}.npz', allow_pickle=False) as z:
            for k in z.files:
                if k != 'plan_sha256':
                    parts[k].append(z[k])
    R = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(R['row_index'], kind='mergesort')
    R = {k: v[order] for k, v in R.items()}
    fix = {r['row_index']: r for r in eng_rows}
    checks = {'rows': int(len(R['row_index'])), 'rows_expected': len(eng_rows),
              'row_indices_equal': R['row_index'].tolist() == sorted(fix)}
    with np.load(V2FIX / 'states.npz', allow_pickle=False) as z:
        ri = R['row_index']
        checks['pre_states_bitwise_equal_v2_fix'] = bool(np.array_equal(z['pre'][ri], R['pre_X_last']))
        checks['post_states_bitwise_equal_v2_fix'] = bool(np.array_equal(z['post'][ri], R['post_X_last']))
    checks['pre_snapshot_equal_rows_csv'] = bool(all(int(fix[i]['pre_snapshot']) == int(s) for i, s in zip(R['row_index'].tolist(), R['pre_hist_snapshot_ms'][:, -1].tolist())))
    checks['post_snapshot_equal_rows_csv'] = bool(all(int(fix[i]['post_snapshot']) == int(s) for i, s in zip(R['row_index'].tolist(), R['post_hist_snapshot_ms'][:, -1].tolist())))
    checks['post_query_before_terminal'] = int((R['post_query_ms'] < R['terminal_ms']).sum())
    checks['post_query_at_or_after_terminal'] = int((R['post_query_ms'] >= R['terminal_ms']).sum())
    checks['match_ids_equal_rows_csv'] = bool(all(fix[i]['match'] == m for i, m in zip(R['row_index'].tolist(), R['match'].tolist())))
    has_b = 'p_pre_B' in R
    y, g = R['winner'], R['match']
    pre_age = (R['pre_query_ms'] - R['pre_hist_snapshot_ms'][:, -1]) / 1000.
    post_age = (R['post_query_ms'] - R['post_hist_snapshot_ms'][:, -1]) / 1000.
    s_ms = R['pre_query_ms'] + 1
    summary = {'role': 'DIAGNOSTIC endpoint transport of frozen A/B; not used for V training/calibration/selection; no labels or q',
               'definitions': {'pre_query_ms': 's - 1 (stored engagement start s; scripts/regenerate_state_v2.py)',
                               'post_query_ms': 'stored endpoint (ms)', 'delta': 'p(post) - p(pre), same frozen model',
                               'target_for_quality': 'final Blue win W (evaluates p quality, not the unobservable true delta)',
                               'weights': 'equal total weight per match (dependent rows and endpoints)'},
               'checks': checks, 'models': {'A': frozen['A'], 'B': frozen.get('B')}}
    q = {}
    for cand in (['A', 'B'] if has_b else ['A']):
        q[cand] = {'pre': E.cell_metrics(y, R[f'p_pre_{cand}'], g), 'post': E.cell_metrics(y, R[f'p_post_{cand}'], g),
                   'pooled_endpoints': E.cell_metrics(np.r_[y, y], np.r_[R[f'p_pre_{cand}'], R[f'p_post_{cand}']], np.r_[g, g])}
    summary['endpoint_probability_quality_vs_final_W'] = q
    if has_b:
        summary['pooled_endpoint_bootstrap_B_minus_A_exploratory'] = E.paired_bootstrap(
            np.r_[y, y], np.r_[R['p_pre_B'], R['p_post_B']], np.r_[R['p_pre_A'], R['p_post_A']], np.r_[g, g],
            50 if smoke else N_BOOT_EXPLORATORY, BOOT_SEED)
    w = E.match_weights(g)

    def wmean(x, m=None):
        m = np.ones(len(x), dtype=bool) if m is None else m
        return float(np.average(x[m], weights=w[m])) if m.any() else None

    def delta_cell(m):
        cell = {'rows': int(m.sum()), 'matches': int(len(set(g[m].tolist())))}
        if not m.any():
            return cell
        cell['mean_delta_A'] = wmean(R['delta_A'], m)
        cell['mean_abs_delta_A'] = wmean(np.abs(R['delta_A']), m)
        if has_b:
            sa, sb = np.sign(R['delta_A']), np.sign(R['delta_B'])
            cell.update(mean_delta_B=wmean(R['delta_B'], m), mean_abs_delta_B=wmean(np.abs(R['delta_B']), m),
                        sign_disagreement_rate_match_weighted=wmean((sa != sb).astype(float), m),
                        sign_disagreement_rows=int((sa != sb)[m].sum()),
                        exact_zero_delta_rows={'A': int((R['delta_A'][m] == 0).sum()), 'B': int((R['delta_B'][m] == 0).sum())},
                        pearson_delta_A_B=float(np.corrcoef(R['delta_A'][m], R['delta_B'][m])[0, 1]) if m.sum() > 2 else None)
        return cell

    summary['delta_overall'] = delta_cell(np.ones(len(y), dtype=bool))
    summary['delta_quantiles'] = {k: np.quantile(R[k], [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist()
                                  for k in (['delta_A', 'delta_B'] if has_b else ['delta_A'])}
    near = {}
    for lo, hi in DELTA_BINS:
        near[f'abs_delta_A_{lo}-{hi}'] = delta_cell((np.abs(R['delta_A']) >= lo) & (np.abs(R['delta_A']) < hi))
        if has_b:
            near[f'abs_delta_B_{lo}-{hi}'] = delta_cell((np.abs(R['delta_B']) >= lo) & (np.abs(R['delta_B']) < hi))
            mx = np.maximum(np.abs(R['delta_A']), np.abs(R['delta_B']))
            near[f'max_abs_delta_AB_{lo}-{hi}'] = delta_cell((mx >= lo) & (mx < hi))
    summary['disagreement_by_delta_magnitude'] = near
    bands = {'0-2': (s_ms < 120000)}
    for lo, hi in E.TIME_BANDS:
        bands[E.band_label(lo, hi)] = (s_ms >= lo * 60000) & (s_ms < hi * 60000)
    summary['delta_by_engagement_start_band'] = {k: delta_cell(m) for k, m in bands.items()}
    summary['delta_by_pre_snapshot_age_s'] = {f'{lo}-{hi if hi < 1e9 else "inf"}': delta_cell((pre_age >= lo) & (pre_age < hi)) for lo, hi in AGE_BINS_S}
    summary['delta_by_post_snapshot_age_s'] = {f'{lo}-{hi if hi < 1e9 else "inf"}': delta_cell((post_age >= lo) & (post_age < hi)) for lo, hi in AGE_BINS_S}
    summary['quality_by_pre_snapshot_age_s'] = {f'{lo}-{hi if hi < 1e9 else "inf"}': {c: E.cell_metrics(y[m], R[f'p_pre_{c}'][m], g[m]) for c in q}
                                                for lo, hi in AGE_BINS_S for m in [(pre_age >= lo) & (pre_age < hi)]}
    summary['quality_by_post_snapshot_age_s'] = {f'{lo}-{hi if hi < 1e9 else "inf"}': {c: E.cell_metrics(y[m], R[f'p_post_{c}'][m], g[m]) for c in q}
                                                 for lo, hi in AGE_BINS_S for m in [(post_age >= lo) & (post_age < hi)]}
    summary['source_freshness'] = {
        'pre_snapshot_age_s_quantiles': np.quantile(pre_age, [0, .05, .25, .5, .75, .95, 1]).tolist(),
        'post_snapshot_age_s_quantiles': np.quantile(post_age, [0, .05, .25, .5, .75, .95, 1]).tolist(),
        'same_snapshot_frame_pre_and_post_rows': int((R['pre_hist_snapshot_ms'][:, -1] == R['post_hist_snapshot_ms'][:, -1]).sum()),
        'pre_valid_history_positions': dict(Counter((R['pre_hist_snapshot_ms'] >= 0).sum(axis=1).tolist())),
        'post_valid_history_positions': dict(Counter((R['post_hist_snapshot_ms'] >= 0).sum(axis=1).tolist())),
        'training_query_age_note': 'A and B were trained on minute-aligned queries whose snapshot age is almost always ~60 s; '
                                   'engagement endpoints have ages spread over 0-60 s, so age strata are reported, not modelled.'}
    ev = R['events']
    summary['delta_by_objective_event_within_interval'] = {
        c: {'present': delta_cell(ev[:, j] > 0), 'absent': delta_cell(ev[:, j] == 0)} for j, c in enumerate(EVENT_COLUMNS)}
    summary['interval_event_rows'] = {c: int((ev[:, j] > 0).sum()) for j, c in enumerate(EVENT_COLUMNS)}
    summary['attribution_note'] = ('Objective events inside (pre, post] co-occur with the delta; other inputs (frames, kills, '
                                   'clock) change in the same interval, so no event is credited with the delta.')
    cols = ['row_index', 'match', 'patch', 's_ms', 'pre_query_ms', 'endpoint_query_ms', 'winner_blue', 'pre_snapshot_ms',
            'pre_snapshot_age_s', 'post_snapshot_ms', 'post_snapshot_age_s', 'pre_valid_history_positions',
            'post_valid_history_positions', 'p_pre_A', 'p_post_A', 'delta_A']
    seed_cols = sorted(k for k in R if k.startswith(('p_pre_B_seed', 'p_post_B_seed')))
    if has_b:
        cols += ['p_pre_B', 'p_post_B', 'delta_B', 'sign_A', 'sign_B', 'sign_disagree', *seed_cols]
    cols += [f'events_{c}' for c in EVENT_COLUMNS]
    with (d / 'endpoint_rows_DIAGNOSTIC.csv').open('w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for i in range(len(y)):
            r = fix[int(R['row_index'][i])]
            row = [int(R['row_index'][i]), R['match'][i], r['patch'], int(s_ms[i]), int(R['pre_query_ms'][i]), int(R['post_query_ms'][i]),
                   int(y[i]), int(R['pre_hist_snapshot_ms'][i, -1]), pre_age[i], int(R['post_hist_snapshot_ms'][i, -1]), post_age[i],
                   int((R['pre_hist_snapshot_ms'][i] >= 0).sum()), int((R['post_hist_snapshot_ms'][i] >= 0).sum()),
                   repr(float(R['p_pre_A'][i])), repr(float(R['p_post_A'][i])), repr(float(R['delta_A'][i]))]
            if has_b:
                sa, sb = int(np.sign(R['delta_A'][i])), int(np.sign(R['delta_B'][i]))
                row += [repr(float(R['p_pre_B'][i])), repr(float(R['p_post_B'][i])), repr(float(R['delta_B'][i])), sa, sb, int(sa != sb),
                        *[repr(float(R[k][i])) for k in seed_cols]]
            row += [int(v) for v in ev[i]]
            wr.writerow(row)
    save_npz(d / 'endpoint_rows_DIAGNOSTIC.npz', **{k: v for k, v in R.items() if not k.endswith('X_last')},
             event_columns=np.asarray(EVENT_COLUMNS), role=np.asarray('DIAGNOSTIC_ONLY'))
    if has_b:
        E.plot_endpoint_deltas(d / 'endpoint_delta_A_vs_B_DIAGNOSTIC.png', R['delta_A'], R['delta_B'])
    write_json(d / 'endpoint_summary_DIAGNOSTIC.json', summary)
    bad = [k for k in ('row_indices_equal', 'pre_states_bitwise_equal_v2_fix', 'post_states_bitwise_equal_v2_fix',
                       'pre_snapshot_equal_rows_csv', 'post_snapshot_equal_rows_csv', 'match_ids_equal_rows_csv') if not checks[k]]
    if bad or checks['rows'] != checks['rows_expected']:
        raise ValueError(f'endpoint provenance checks failed: {bad}')
    return summary


def frame_audit_stage(run, names, matches, frozen, workers):
    groups = column_groups(names)
    tasks = [dict(frozen_task_fields(frozen), names=names, groups=groups, match=m) for m in matches]
    results, failures = run_pool(run, frame_audit_task, tasks, workers, 'frame_update_audit', 'per match')
    if failures:
        raise RuntimeError(f'frame audit failures: {failures}')
    recs = [r for res in results for r in res['records']]
    d = run.out / 'frame_update_audit'
    d.mkdir(parents=True, exist_ok=True)
    with (d / 'update_pairs.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['match', 'x_ms', 'update_class', 'frame_update_at_x', 'predictor_events_at_x', 'all_events_at_x',
                    'last_position_groups_changed', 'earlier_positions_with_discrete_change',
                    'earlier_positions_on_frame_or_event_ts', 'dp_A', 'dp_B', 'snapshot_before_ms', 'snapshot_at_x_ms'])
        for r in sorted(recs, key=lambda r: (r['match'], r['x_ms'])):
            w.writerow([r['match'], r['x_ms'], r['update_class'], r['frame_update_at_x'], json.dumps(r['predictor_events_at_x']),
                        r['all_events_at_x'], '|'.join(r['last_position_groups_changed']),
                        ' '.join(map(str, r['earlier_positions_with_discrete_change'])),
                        ' '.join(map(str, r['earlier_positions_on_frame_or_event_ts'])), repr(r['dp_A']),
                        '' if r['dp_B'] is None else repr(r['dp_B']), r['snapshot_before_ms'], r['snapshot_at_x_ms']])
    has_b = bool(recs) and recs[0]['dp_B'] is not None

    def qs(v):
        return np.quantile(np.abs(v), [.5, .9, .99, 1]).tolist() if len(v) else None

    by_class = {}
    for cls in sorted({r['update_class'] for r in recs}):
        rs = [r for r in recs if r['update_class'] == cls]
        dA = np.asarray([r['dp_A'] for r in rs])
        cell = {'pairs': len(rs), 'abs_dp_A_q50_q90_q99_max': qs(dA), 'dp_A_exact_zero': int((dA == 0).sum()),
                'pairs_with_no_discrete_input_change_at_query': sum(not r['last_position_groups_changed'] for r in rs),
                'pairs_with_more_than_one_input_group_changed': sum(len(r['last_position_groups_changed']) > 1 for r in rs),
                'pairs_with_multiple_raw_events_same_ts': sum(r['all_events_at_x'] > 1 for r in rs),
                'groups_changed_counts': dict(Counter(g for r in rs for g in r['last_position_groups_changed']))}
        if has_b:
            dB = np.asarray([r['dp_B'] for r in rs])
            cell.update(abs_dp_B_q50_q90_q99_max=qs(dB), dp_B_exact_zero=int((dB == 0).sum()),
                        pairs_with_B_earlier_position_change=sum(bool(r['earlier_positions_with_discrete_change']) for r in rs),
                        sign_disagreement_nonzero=int(((np.sign(dA) != np.sign(dB)) & (dA != 0) & (dB != 0)).sum()))
        by_class[cls] = cell
    summary = {'matches': matches, 'selection': 'first by sha256("p2_frame_audit:"+match) among test matches, excluding '
                                                'trajectory matches; fixed in protocol.json before training',
               'pairs': len(recs), 'query_precision': 'integer ms: x-1 and x', 'min_query_ms': MIN_EVAL_QUERY_MS,
               'boundaries': 'every raw frame timestamp and every predictor-event timestamp (CHAMPION_KILL, ELITE_MONSTER_KILL, '
                             'DRAGON_SOUL_GIVEN, BUILDING_KILL, TURRET_PLATE_DESTROYED) with x-1 >= 120000 and x < terminal',
               'by_update_class': by_class,
               'duplicate_timestamps': {'boundaries_with_multiple_raw_events': sum(r['all_events_at_x'] > 1 for r in recs),
                                        'boundaries_with_multiple_predictor_events': sum(sum(r['predictor_events_at_x'].values()) > 1 for r in recs),
                                        'boundaries_with_frame_and_predictor_event': sum(r['update_class'] == 'frame_and_event_same_ts' for r in recs)},
               'column_groups': dict(Counter(groups)),
               'interpretation': 'Observed co-occurrence of a probability change with a frame update or event timestamp; not '
                                 'a causal allocation. When several inputs change at one timestamp (several events, a frame '
                                 'and an event, or B history positions crossing other boundaries), the change is not assigned '
                                 'to any single event. Auxiliary timestamps/labels are never predictors.'}
    write_json(d / 'frame_update_audit_summary.json', summary)
    return summary


def replay_audit_stage(run, names, replay_matches, rows, seqs, tables, frozen, workers, smoke):
    import train.temporal_history_winprob_v3 as T
    from gameplay.state_value_v2 import STATE_VERSION
    A = T.CandidateA.load(frozen['A']['path'], frozen['A']['sha256'])
    B = T.CandidateB.load(frozen['B']['bundle_dir'], frozen['B']['bundle_sha256']) if frozen.get('B') else None
    tasks = []
    for p, ms in replay_matches.items():
        s = seqs[p]
        tab = tables[p]
        for m in ms:
            ix = np.flatnonzero(s['match'] == m)[:3]
            if p == 'test':
                gix = np.flatnonzero(seqs['test_grid']['match'] == m)
                extra = [('test_grid', i) for i in gix[[0, len(gix) // 2]]] if len(gix) else []
            else:
                extra = []
            items = [(p, int(i)) for i in ix] + extra
            pipeline = []
            for name, i in items:
                ss = seqs[name]
                H = np.zeros((9, len(names)))
                H[ss['mask'][i]] = tab['X'][ss['index'][i][ss['mask'][i]]]
                item = {'query_ms': int(ss['time_ms'][i]), 'matrix': H.tolist(), 'mask': ss['mask'][i].tolist(),
                        'p_A': float(A.predict_last_matrix(H[-1:], names, STATE_VERSION)[0])}
                if B:
                    Z, C = B.preprocessor.encode_states(tab['X'][ss['index'][i][ss['mask'][i]]])
                    Zp = np.zeros((1, 9, Z.shape[1]))
                    Cp = np.full((1, 9, C.shape[1]), -1)
                    Zp[0, ss['mask'][i]], Cp[0, ss['mask'][i]] = Z, C
                    item['p_B'] = float(B.predict_encoded(Zp, Cp, ss['mask'][i][None])[0][0])
                pipeline.append(item)
            queries = sorted({it['query_ms'] for it in pipeline} | {q for q in (120000, 300000) if q <= min(it['query_ms'] for it in pipeline)})
            tasks.append(dict(frozen_task_fields(frozen), names=names, match=m, partition=p, queries=queries, pipeline=pipeline))
    results, failures = run_pool(run, replay_audit_task, tasks, workers, 'replay_audits', 'per match')
    if failures:
        raise RuntimeError(f'replay audit failures: {failures}')
    tot = lambda key, sub=None: sum((r[sub][key] if sub else r[key]) for r in results)
    summary = {'matches': {p: ms for p, ms in replay_matches.items()}, 'records': results,
               'future_perturbation': {'queries': tot('queries'), 'masked_queries': tot('masked_queries'),
                                       'matrix_mismatch': tot('future_matrix_mismatch'), 'mask_or_snapshot_mismatch': tot('future_mask_mismatch'),
                                       'p_A_mismatch': tot('future_p_A_mismatch'), 'p_B_mismatch': tot('future_p_B_mismatch'),
                                       'perturbation': 'frames after t rescaled (x*-3.5+777); events after t team/participant '
                                                       'mutated; fake soul/baron/kill at t+1 and GAME_END at t+2 injected'},
               'incremental_replay': {k: tot(k, 'incremental') for k in ('queries', 'matrix_mismatch', 'mask_mismatch',
                                                                         'snapshot_mismatch', 'p_A_mismatch', 'p_B_mismatch')},
               'pipeline_vs_api': {'rows': tot('rows', 'pipeline_vs_api'), 'matrix_mismatch': tot('matrix_mismatch', 'pipeline_vs_api'),
                                   'mask_mismatch': tot('mask_mismatch', 'pipeline_vs_api'), 'p_A_not_exact': tot('p_A_not_exact', 'pipeline_vs_api'),
                                   'p_B_not_exact': tot('p_B_not_exact', 'pipeline_vs_api'),
                                   'p_B_max_abs_diff': max(r['pipeline_vs_api']['p_B_max_abs_diff'] for r in results)},
               'scope': 'retrospective causal replay over stored raw cache; not a real-time API validation'}
    write_json(run.out / 'validation_replay_audits.json', summary)
    return summary


def api_guard_checks(names, frozen, match, workers_unused=None):
    """Runtime guard checks on one real test match (in addition to pytest)."""
    ctx = _context()
    T = ctx['T']
    A, B, sA, sB = _frozen(frozen_task_fields(frozen))
    pack = _load(match, ctx)
    ts = np.asarray(pack['minute_ts'], dtype=np.int64)
    src = T.BatchSource(match, pack, ctx['node_names'])

    def raises(fn, exc):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    q0, q1 = int(ts[3]) + 5, int(ts[5]) + 5
    out = {'delta_rejects_A_vs_B_model_mismatch': raises(lambda: T.endpoint_delta(sA.predict_source(src, q0), sB.predict_source(src, q1)), T.ModelVersionError) if sB else None,
           'delta_rejects_reversed_endpoints': raises(lambda: T.endpoint_delta(sA.predict_source(src, q1), sA.predict_source(src, q0)), T.ChronologyError),
           'service_rejects_wrong_schema': raises(lambda: T.assemble_history(src, q0, names[::-1]), ValueError),
           'bundle_rejects_wrong_expected_hash': raises(lambda: T.CandidateB.load(frozen['B']['bundle_dir'], '0' * 64), T.ModelVersionError) if frozen.get('B') else None,
           'candidate_A_rejects_wrong_hash': raises(lambda: T.CandidateA.load(frozen['A']['path'], '0' * 64), T.ModelVersionError)}
    log = T.ObservationLog(match, pack['meta']['team_map'], ctx['node_names'])
    log.add_frame(match, int(ts[0]), pack['node_minute'][0], ctx['node_names'])
    log.add_frame(match, int(ts[1]), pack['node_minute'][1], ctx['node_names'])
    out['log_rejects_out_of_order_frame'] = raises(lambda: log.add_frame(match, int(ts[0]), pack['node_minute'][0], ctx['node_names']), T.ChronologyError)
    out['log_rejects_cross_match_event'] = raises(lambda: log.add_event('OTHER_MATCH', {'type': 'CHAMPION_KILL', 'timestamp': int(ts[1]) + 1}), T.CrossMatchError)
    out['log_rejects_future_query'] = raises(lambda: log.source(int(ts[1]) - 1), T.FutureObservationError)
    out['log_rejects_wrong_node_schema'] = raises(lambda: log.add_frame(match, int(ts[2]), pack['node_minute'][2], ctx['node_names'][::-1]), T.SchemaError)
    return out


# ----------------------------------------------------------------------------- pipeline
def execute(run, a):
    import scipy
    import sklearn
    import torch
    import train.temporal_history_winprob_v3 as T
    from gameplay.state_value_v2 import STATE_VERSION
    smoke = a.smoke
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
    for k, p in NEW_SOURCES.items():
        (run.out / 'source_snapshot' / k).parent.mkdir(parents=True, exist_ok=True)
        (run.out / 'source_snapshot' / k).write_bytes(p.read_bytes())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'--basetemp={run.out / "pytest_basetemp"}',
                           str(ROOT / 'tests' / 'test_temporal_winprob_v3_history.py'), str(ROOT / 'tests' / 'test_state_value_v2_contract.py')],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=1800, env=env)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
    (run.out / 'pytest_output.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
    tests = {'command': 'python -m pytest -q -p no:cacheprovider tests/test_temporal_winprob_v3_history.py '
                        'tests/test_state_value_v2_contract.py', 'returncode': proc.returncode, 'summary': lines[-1] if lines else ''}
    run.log(f'tests: {tests}')
    if proc.returncode != 0:
        raise RuntimeError('contract tests failed; see pytest_output.log')

    run.stage('protocol_checks')
    I = load_inputs()
    names = list(T.CandidateA.load(P1 / 'models' / 'expanded_model_v2.joblib', I['p1_info']['model_sha256']['expanded']).state_names)
    pchecks, pfail, fix_rows = protocol_checks(I, names)
    if pfail:
        write_json(run.out / 'errors.json', {'stage': 'protocol_checks', 'failures': pfail, 'checks': pchecks})
        raise RuntimeError(f'protocol checks failed: {pfail}')
    keep, rows, grid_index, eng_rows, traj, frame_matches, replay_matches = select_subsets(I, fix_rows, smoke)
    splits = I['old_proto']['splits']
    plan = build_plan(splits, rows, I['curves'], grid_index, keep)
    chunk_size = 5 if smoke else 100
    n_boot_primary = 100 if smoke else N_BOOT_PRIMARY
    n_boot_expl = 30 if smoke else N_BOOT_EXPLORATORY
    protocol = {
        'version': 'temporal_winprob_v3_p2_exploratory', 'module_version': T.MODULE_VERSION, 'history_version': T.HISTORY_VERSION,
        'state_version': STATE_VERSION, 'smoke': bool(smoke), 'smoke_matches_per_partition': smoke or None,
        'target': 'W = final Blue win (GAME_END via gameplay.state_value.final_outcome); read only as training/evaluation target',
        'estimand': 'p_t ~ P(W=1 | observations available at t); A = f(X_t, t); B = f(X_{t-8min}, ..., X_t) at one-minute '
                    'positions; associational approximations, not causal models',
        'candidates': {
            'A': {'name': T.CANDIDATE_A, 'definition': 'exact frozen P1 expanded V2 adapter; no refit or recalibration',
                  'path': str(P1 / 'models' / 'expanded_model_v2.joblib'), 'sha256': pchecks['candidate_A_sha256'],
                  'model_version': I['p1_res']['model_version'], 'chosen_in_p1': I['p1_res']['chosen']['expanded']},
            'B': {'name': T.CANDIDATE_B, 'model_version': T.CANDIDATE_B_VERSION, 'architecture': T.ARCHITECTURE,
                  'training': {'epochs': EPOCHS, 'batch_size': BATCH, 'optimizer': 'RMSprop', 'lr': LR, 'rho': RHO, 'eps': EPS,
                               'loss': 'binary cross entropy', 'seeds': list(SEEDS), 'early_stopping': 'none (fixed budget)',
                               'sample_weight': 'none; one sample per FIT sampled query row (histories are inputs, not extra labelled rows)',
                               'device': 'CPU only (torch threads 4); CUDA hidden', 'runtime': f'torch {torch.__version__} (already installed)'},
                  'provenance_of_choices': '9 positions, 8 recurrent units and dropout .25 follow Silva et al. (SBGames 2018); tanh, '
                                           'sigmoid/BCE, RMSprop lr .001, batch 64, 50 epochs and seeds 17/29/43 are P2 design '
                                           'choices; no architecture or hyperparameter search',
                  'calibration': 'raw / sigmoid (Platt on logit, C=1e6) / isotonic fitted per seed on CALIBRATE only; one chosen '
                                 'per seed by SELECT match-weighted log loss',
                  'primary': 'arithmetic mean of the three selected calibrated seed probabilities (predeclared)'}},
        'selection_rule': 'A vs B primary by SELECT match-weighted log loss; exact tie keeps A; written to selection.json before '
                          'any TEST prediction; chosen candidate frozen by hash (models/frozen_models.json)',
        'history_assembly': {
            'positions_ms': 't-480000, t-420000, ..., t (nine positions, 60000 ms step)',
            'position_state': 'gameplay.state_value_v2.StateBuilder.at(position): latest frame with timestamp <= position and '
                              'predictor events with timestamp <= position',
            'support': 'a position is valid iff position >= max(0, first observed frame timestamp); earlier positions are '
                       'zero-filled and masked (recurrence skips them); the query position must be valid',
            'shared_code': 'train.temporal_history_winprob_v3.assemble_history (API) and the pipeline table index use '
                           'history_times(); equality audited on real matches (validation_replay_audits.json)'},
        'feature_availability_rules': {
            'frames': 'node fields (SNAPSHOT_FIELDS, champion_id) from the latest raw frame at or before each position; raw '
                      'frames every ~60 s (first frame at 0 ms)',
            'events': 'CHAMPION_KILL, ELITE_MONSTER_KILL, DRAGON_SOUL_GIVEN, BUILDING_KILL, TURRET_PLATE_DESTROYED with '
                      'timestamp <= position (integer ms)',
            'roster': 'participant->team map from match metadata at start; participant blocks ordered team then participantId; '
                      'postgame role_slots never used',
            'excluded_predictors': ['snapshot_age_s (audit only)', 'final winner/GAME_END', 'final duration or percent of final '
                                    'match time', 'teamId-0 soul events (diagnostic only)', 'engagement labels/timestamps',
                                    'future frames/events, future padding, normalisation or interpolation'],
            'champion_encoding': 'categorical: A one-hot per slot (P1); B FIT-only per-slot one-hot, unseen ID -> zero vector'},
        'splits': {p: splits[p] for p in (*PARTITIONS, 'engagement')},
        'query_keys': {'sampled': 'temporal_winprob_v3_buckets/sampled_minutes.json (exact rows and order)',
                       'test_grid': 'temporal_winprob_v3_buckets/independent_time_curves.npz match/time_ms',
                       'sampled_rows_used': len(rows), 'grid_rows_used': int(len(grid_index)),
                       'sampled_keys_sha256': sha256_json([[r['match'], int(r['time_ms']), r['role']] for r in rows]),
                       'grid_keys_sha256': sha256_json([I['curves']['match'][grid_index].tolist(), I['curves']['time_ms'][grid_index].tolist()])},
        'sources': {'old_protocol.json': sha256_file(OLD / 'protocol.json'), 'sampled_minutes.json': sha256_file(OLD / 'sampled_minutes.json'),
                    'independent_time_curves.npz': sha256_file(OLD / 'independent_time_curves.npz'),
                    'p1_protocol.json': sha256_file(P1 / 'protocol.json'), 'p1_results.json': sha256_file(P1 / 'results.json'),
                    'p1_selection.json': sha256_file(P1 / 'selection.json'),
                    'p1_sampled_states_v2.npz': sha256_file(P1 / 'states' / 'sampled_states_v2.npz'),
                    'p1_test_grid_states_v2.npz': sha256_file(P1 / 'states' / 'test_grid_states_v2.npz'),
                    'state_value_v2_fix/rows.csv': sha256_file(V2FIX / 'rows.csv'),
                    'state_value_v2_fix/states.npz': sha256_file(V2FIX / 'states.npz'),
                    'regenerate_state_v2.py (endpoint definition)': sha256_file(ROOT / 'scripts' / 'regenerate_state_v2.py')},
        'protocol_checks': pchecks,
        'evaluation': {'test_one_minute': 'exact P1 one query per test match', 'test_grid': 'full stored test grid',
                       'metrics': 'AUC, Brier, log loss, ECE, reliability bins, calibration intercept/slope, N (rows, matches)',
                       'weights': 'equal total weight per match within each cell',
                       'bands': 'every one-minute band and 2-10/10-20/20-30/30+ minutes', 'strata': 'P1 objective strata incl. elder and souls',
                       'na_policy': 'cells with <30 matches -> all metrics NA; <5 matches in either outcome class -> AUC and calibration line NA',
                       'bootstrap': {'primary': f'{n_boot_primary} paired match bootstrap replicates, B primary minus A, models fixed',
                                     'exploratory': f'{n_boot_expl} replicates per band/stratum; no multiple-testing certainty claims',
                                     'seed': BOOT_SEED}},
        'trajectories': {'matches': traj, 'rule': 'sha256("p2_trajectory:"+match) ascending among test matches',
                         'query_cadence_ms': TRAJECTORY_CADENCE_MS, 'extra_queries': 'x-1 and x for raw frames and predictor events'},
        'engagement_endpoint_transport': {'rows': len(eng_rows), 'source': 'outputs/state_value_v2_fix/rows.csv',
                                          'pre_query': 's - 1 ms', 'post_query': 'endpoint ms', 'role': 'DIAGNOSTIC only',
                                          'never_used_for': 'V training, calibration, selection, labels or q'},
        'frame_update_audit': {'matches': frame_matches, 'rule': 'sha256("p2_frame_audit:"+match) among test matches, excluding trajectory matches'},
        'replay_audits': {'matches': replay_matches, 'rule': 'sha256("p2_replay_audit:"+match) per partition'},
        'plan': {'matches': len(plan), 'history_states': sum(len(t) for _, t in plan), 'chunk_size': chunk_size},
        'resources': {'workers': a.workers, 'blas_threads_main': 4, 'blas_threads_worker': 1, 'torch_threads': 4, 'gpu': 'disabled',
                      'note': 'user GPU queue untouched'},
        'source_hashes': {'legacy': hashes_before['legacy_sources'], 'new': new_hashes},
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                     'scipy': scipy.__version__, 'torch': torch.__version__},
        'references': {
            'Kim, Lee, Chung (CoG 2020)': 'per-minute LoL win probability and calibration motivation; its uncertainty-aware loss is NOT reproduced',
            'Hodge et al. (IEEE ToG 13(4):368-379, DOI 10.1109/TG.2019.2948469)': 'live professional Dota 2 prediction motivation only',
            'Maymin (2021, DOI 10.1515/jqas-2019-0096)': 'snapshot probability/value reference (candidate A family)',
            'Silva, Pappa, Chaimowicz (SBGames 2018)': 'SimpleRNN history comparator (9 steps, 8 units, dropout .25); adapted, not reproduced',
            'Jalovaara (2024, master thesis)': '5-minute bucket sampling of the reused query keys; supporting, not peer-reviewed journal evidence'},
        'data_use': 'same exploratory 50k corpus (patches 15.14-15.16) as P1; historical test was examined before; 15.16 and 26.13 '
                    'are not untouched tests; no all-patch claim; independent future-patch confirmation pending',
        'written_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    write_json(run.out / 'protocol.json', protocol)
    run.log(f"protocol.json written: {len(plan)} matches, {protocol['plan']['history_states']} history states, "
            f"{len(rows)} sampled rows, {len(grid_index)} grid rows, {len(eng_rows)} engagement rows")

    n_chunks = run_extraction(run, plan, names, a.workers, chunk_size)
    run.stage('assembly')
    tables, seqs, match_table, achecks = assemble(run, n_chunks, names, splits, rows, I['curves'], grid_index, plan)
    run.log(f"assembly: states {achecks['built_states_by_partition']}, rows {achecks['rows']}, masked rows "
            f"{achecks['rows_with_any_masked_position']}, P1-equal {achecks['query_states_bitwise_equal_p1']}")

    run.stage('candidate_A_select')
    A, pA_sel, A_info = candidate_a_select(run, names, tables, seqs, I, smoke)
    B_info, b_blocker = None, None
    run.stage('training_B')
    try:
        B, pB_sel, pB_members_sel, B_info = train_candidate_b(run, names, tables, seqs, smoke)
        write_json(run.out / 'models' / 'candidate_B_training_summary.json', {k: v for k, v in B_info.items() if k != 'manifest'})
    except Exception as exc:
        b_blocker = repr(exc)
        run.blocker('training_B', exc)
    run.stage('selection')
    selection, frozen = write_selection(run, A_info, B_info, b_blocker)

    results = {'selection': selection}
    stage_status = {'training_B': 'blocked' if b_blocker else 'complete'}
    for name, fn in (
            ('test_evaluation', lambda: test_evaluation(run, names, tables, seqs, frozen, I, n_boot_primary, n_boot_expl, smoke)[0]),
            ('trajectories', lambda: trajectories_stage(run, names, traj, frozen, a.workers)),
            ('endpoint_diagnostics', lambda: endpoint_stage(run, names, eng_rows, frozen, a.workers, smoke)),
            ('frame_update_audit', lambda: frame_audit_stage(run, names, frame_matches, frozen, a.workers)),
            ('replay_audits', lambda: replay_audit_stage(run, names, replay_matches, rows, seqs, tables, frozen, a.workers, smoke)),
            ('api_guard_checks', lambda: api_guard_checks(names, frozen, traj[0]))):
        run.stage(name)
        try:
            results[name] = fn()
            stage_status[name] = 'complete'
        except Exception as exc:
            stage_status[name] = 'blocked'
            run.blocker(name, exc)

    run.stage('validation')
    hashes_after = legacy_hashes()
    write_json(run.out / 'hashes_after.json', hashes_after)
    changed = sorted(f'{g}/{k}' for g in hashes_before for k in set(hashes_before[g]) | set(hashes_after[g])
                     if hashes_before[g].get(k) != hashes_after[g].get(k))
    p1_after = read_json(P1 / 'hashes_after.json')
    frozen_vs_p1 = {g: all(hashes_before[g].get(k) == v for k, v in p1_after[g].items())
                    for g in ('state_value_v2_fix', 'old_v_artifacts')}
    frozen_vs_p1['legacy_sources'] = all(hashes_before['legacy_sources'].get(k) == v for k, v in p1_after['legacy_sources'].items())
    te = results.get('test_evaluation') or {}
    ep = results.get('endpoint_diagnostics') or {}
    rp = results.get('replay_audits') or {}
    tr = results.get('trajectories') or {}
    guards = results.get('api_guard_checks') or {}
    mem = achecks['membership']
    checks = {
        'contract_tests_pass': tests['returncode'] == 0,
        'protocol_checks_pass': not pfail,
        'splits_and_query_keys_equal_p1': pchecks['p1_splits_equal_original'] and pchecks['p1_sampled_keys_equal_source_order'] and pchecks['p1_grid_keys_equal_source'],
        'engagement_matches_disjoint_from_all_V_partitions': not any(n for d in pchecks['engagement_overlap_with_V'].values() for n in d.values()),
        'history_positions_same_match_and_partition': all(m['cross_match_positions'] == 0 and m['time_mismatch_positions'] == 0 for m in mem.values()),
        'history_masks_consistent_and_query_position_valid': all(m['masked_with_index'] == 0 and m['valid_without_index'] == 0 and m['last_position_masked'] == 0 and m['position_times_wrong'] == 0 for m in mem.values()),
        'no_future_history_positions': all(m['future_positions'] == 0 for m in mem.values()) and achecks['snapshot_not_after_position'],
        'query_states_bitwise_equal_p1': all(achecks['query_states_bitwise_equal_p1'].values()),
        'candidate_A_reproduces_p1_select': A_info['select_predictions_bitwise_equal_p1_stored'],
        'candidate_A_reproduces_p1_test': (te.get('checks', {}).get('A_one_minute_bitwise_equal_p1_stored') and te.get('checks', {}).get('A_grid_bitwise_equal_p1_stored')) if te else None,
        'candidate_B_trained': B_info is not None,
        'candidate_B_torch_numpy_parity': all(v['max_abs_diff_numpy_float64_vs_torch_float32'] <= 1e-5 for v in B_info['torch_numpy_parity'].values()) if B_info else None,
        'candidate_B_save_load_identity': all(B_info['save_load_identity'].values()) if B_info else None,
        'preprocessing_fit_on_fit_only': (B_info['preprocessing_fit_origin']['partition'] == 'fit') if B_info else None,
        'selection_written_before_test_predictions': selection['written_before_any_test_prediction'] and not selection['test_prediction_files_at_selection'],
        'predictions_finite_in_unit_interval': te.get('checks', {}).get('predictions_finite_in_unit_interval') if te else None,
        'future_perturbation_invariance_real_cache': (rp['future_perturbation']['matrix_mismatch'] == 0 and rp['future_perturbation']['mask_or_snapshot_mismatch'] == 0 and rp['future_perturbation']['p_A_mismatch'] == 0 and rp['future_perturbation']['p_B_mismatch'] == 0 and rp['future_perturbation']['masked_queries'] > 0) if rp else None,
        'incremental_equals_batch_current_query': (all(v == 0 for k, v in rp['incremental_replay'].items() if k != 'queries') and all(
            all(v == 0 for k, v in r.items() if k.endswith('mismatch')) for r in tr['incremental_replay_equality'].values())) if (rp and tr) else None,
        'pipeline_histories_equal_api_histories': (rp['pipeline_vs_api']['matrix_mismatch'] == 0 and rp['pipeline_vs_api']['mask_mismatch'] == 0 and rp['pipeline_vs_api']['p_A_not_exact'] == 0 and rp['pipeline_vs_api']['p_B_max_abs_diff'] <= 1e-12) if rp else None,
        'api_guards_all_raise': all(v for v in guards.values() if v is not None) if guards else None,
        'endpoint_rows_and_states_equal_v2_fix': all(ep['checks'][k] for k in ('row_indices_equal', 'pre_states_bitwise_equal_v2_fix', 'post_states_bitwise_equal_v2_fix', 'pre_snapshot_equal_rows_csv', 'post_snapshot_equal_rows_csv', 'match_ids_equal_rows_csv')) and (ep['checks']['rows'] == (EXPECTED_ENGAGEMENT_ROWS if not smoke else ep['checks']['rows_expected'])) if ep else None,
        'legacy_and_p1_files_unchanged_during_run': not changed,
        'frozen_legacy_hashes_equal_p1_record': all(frozen_vs_p1.values())}
    failed = [k for k, v in checks.items() if v is False]
    incomplete = [k for k, v in checks.items() if v is None]
    status = 'pass' if not failed and not incomplete else ('fail' if failed else 'incomplete')
    validation = {'status': status, 'failed_checks': failed, 'incomplete_checks': incomplete, 'checks': checks,
                  'stage_status': stage_status, 'blockers': run.status['blockers'], 'tests': tests,
                  'assembly_checks': {k: v for k, v in achecks.items() if k != 'sequence_file_sha256'},
                  'api_guard_checks': guards, 'changed_files_during_run': changed, 'frozen_vs_p1_record': frozen_vs_p1,
                  'candidate_A_select': {k: v for k, v in A_info.items() if k not in ('select_metrics', 'select_cell')},
                  'claims_not_made': ['no causal effect of objectives, engagements or states',
                                      'no real-time public API validation (retrospective causal replay only)',
                                      'no reproduction of Kim et al. uncertainty loss or of Silva et al. results',
                                      'no all-patch generalisation; 15.16/26.13 are not untouched tests',
                                      'bootstrap intervals exclude training/selection variability',
                                      'endpoint outputs are diagnostics, not replacement labels or q training data']}
    write_json(run.out / 'validation.json', validation)
    summary = {'status': 'complete' if not run.status['blockers'] else 'complete_with_blockers', 'smoke': bool(smoke),
               'validation_status': status, 'stage_status': stage_status, 'chosen_candidate': selection['chosen_candidate'],
               'select_log_loss': selection['select_log_loss'], 'frozen_models': frozen,
               'test_one_minute': {k: {m: te['test_one_minute'][k][m] for m in ('rows', 'matches', 'auc', 'brier', 'log_loss', 'ece', 'calibration_intercept', 'calibration_slope')}
                                   for k in te.get('test_one_minute', {}) if k == 'A' or k == 'B' or k.startswith('B_seed')} if te else None,
               'elapsed_seconds': round(time.time() - run.started, 1)}
    write_json(run.out / 'results.json', summary)
    from temporal_winprob_v3_report import write_report
    try:
        write_report(run.out)
    except Exception as exc:
        run.blocker('report', exc)
    run.stage('complete' if not run.status['blockers'] else 'complete_with_blockers', validation_status=status,
              failed_checks=failed, incomplete_checks=incomplete)
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', type=int, default=0, help='matches per partition; writes to smoke_run_nN/')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--report-only', action='store_true')
    a = ap.parse_args()
    if not 1 <= a.workers <= 4:
        raise SystemExit('workers must be between 1 and 4')
    out = OUT_BASE / f'smoke_run_n{a.smoke}' if a.smoke else OUT_BASE
    if a.report_only:
        from temporal_winprob_v3_report import write_report
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


if __name__ == '__main__':
    sys.exit(main())
