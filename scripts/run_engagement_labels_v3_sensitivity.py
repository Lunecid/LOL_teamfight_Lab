"""P3: frozen-A engagement label and window sensitivity (EXPLORATORY candidate labels; not ground truth).

Cohort: the exact (match, s) rows and order of outputs/state_value_v2_fix/rows.csv (no new detection).
Endpoints: operational B rule (scripts/engagement_labels_v3_rules.py) for h in {60, 90, 120} s; B90 is the
working primary, 60/120 are sensitivity windows. Values: the SAME frozen P1 expanded V2 model A at
q_pre = s - 1 ms and at each endpoint; delta_A = p_post_A - p_pre_A, Y_A = 1(delta_A > 0). The frozen P2
history ensemble B is scored through the P2 history adapter as a diagnostic model-sensitivity comparator only.
Nothing is trained, recalibrated or selected here; no q model is fitted. All outputs land in
outputs/engagement_labels_v3_sensitivity.
"""
from __future__ import annotations

import os

_WORKER = os.environ.get('P3_WORKER') == '1'
_THREADS = '1' if _WORKER else '4'
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = _THREADS
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys

sys.dont_write_bytecode = True

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
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
import engagement_labels_v3_rules as R  # noqa: E402  (pure, no side effects)

CACHE = Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
OUT_BASE = ROOT / 'outputs' / 'engagement_labels_v3_sensitivity'
V2FIX = ROOT / 'outputs' / 'state_value_v2_fix'
EXPOSURES = ROOT / 'outputs' / 'postkill_objective_delay_full' / 'exposures.csv'
WV_LABELS = ROOT / 'outputs' / 'window_validation_20260914' / 'labels.csv'
BB_VALUES = ROOT / 'outputs' / 'b_boundary_60_90_120' / 'dynamic_values.csv'
VS_ROWS = ROOT / 'outputs' / 'validation_suite_20260914' / 'rows.csv'
PA_PRED = ROOT / 'outputs' / 'position_ablation_20260914' / 'predictions.csv'
P1 = ROOT / 'outputs' / 'independent_v2_participant_order'
P2 = ROOT / 'outputs' / 'temporal_winprob_v3'
VD = WT / 'outputs' / 'temporal_winprob_v3_buckets'
OLD_EVAL = WT / 'outputs' / 'state_value_main_50k_eval'
ORIGINAL_REPO = Path('C:/Users/todtj/PycharmProjects/LOL_teamfight')

EXPECTED_ROWS, EXPECTED_MATCHES = 26_693, 9_198
EXPECTED_A_SHA256 = '7ca9dd1403ad35c6f0d043bc595dbded28a0d90c2024346f499f22732a3af53c'
HS = R.HORIZONS_S
QKEYS = ('pre',) + tuple(f'h{h}' for h in HS)
WORKERS = 4
N_BOOT_MAIN, N_BOOT_CELL, BOOT_SEED = 2000, 1000, 20260915
MIN_CELL_MATCHES = 30
P2_PROB_TOLERANCE = 1e-12
DETERMINISM_MATCHES, PERTURBATION_MATCHES = 24, 8
TIME_BANDS = ((0, 2), (2, 10), (10, 20), (20, 30), (30, 1000))
ROLE = 'EXPLORATORY_CANDIDATE_LABELS_NOT_GROUND_TRUTH_NOT_PRODUCTION'

WT_SOURCES = ('gameplay/state_value.py', 'gameplay/state_value_v2.py', 'train/temporal_winprob.py',
              'train/independent_winprob_v2.py', 'train/temporal_history_winprob_v3.py', 'core/config.py',
              'data/cache_io.py', 'train/state_value_experiment.py', 'scripts/run_temporal_winprob_v3_buckets.py',
              'scripts/build_state_value_dataset.py')
ROOT_SOURCES = ('scripts/event_boundary_cif.py', 'scripts/regenerate_state_v2.py', 'scripts/run_validation_suite.py',
                'scripts/validate_windows.py', 'scripts/measure_postkill_full.py',
                'scripts/measure_postkill_objective_delay.py', 'scripts/ablate_observed_position.py',
                'scripts/run_temporal_winprob_v3.py', 'scripts/temporal_winprob_v3_eval.py',
                'scripts/temporal_winprob_v3_report.py', 'scripts/train_independent_v2.py',
                'scripts/report_b_three_horizons.py', 'tests/test_temporal_winprob_v3_history.py',
                'tests/test_state_value_v2_contract.py',
                'docs/TEMPORAL_WINPROB_LITERATURE_ALIGNMENT_20260915.md',
                'docs/LABEL_ENDPOINT_RULE_AND_EXAMPLES_20260914.md', 'docs/DEFINITION_EVIDENCE_REGISTER_20260914.md',
                'docs/POST_ENGAGEMENT_BOUNDARY_METHODS_20260914.md', 'docs/EVENT_BOUNDARY_CIF_PROTOCOL_20260914.md',
                'docs/B_BOUNDARY_60_90_120_PROTOCOL_20260914.md', 'docs/B_BOUNDARY_60_90_120_RESULTS_20260914.md',
                'docs/WINDOW_VALIDATION_PROTOCOL_20260914.md', 'docs/CLAUDE_EXECUTE_P3_LABEL_SENSITIVITY.md')
NEW_SOURCES = ('scripts/run_engagement_labels_v3_sensitivity.py', 'scripts/engagement_labels_v3_rules.py',
               'scripts/engagement_labels_v3_report.py', 'tests/test_engagement_labels_v3_rules.py')
FROZEN_TREES = {'state_value_v2_fix': V2FIX, 'p1_outputs': P1, 'p2_outputs': P2, 'old_v_artifacts': VD}
FROZEN_FILES = {'exposures.csv': EXPOSURES, 'window_validation/labels.csv': WV_LABELS,
                'window_validation/results.json': WV_LABELS.parent / 'results.json',
                'b_boundary/dynamic_values.csv': BB_VALUES, 'b_boundary/results.json': BB_VALUES.parent / 'results.json',
                'b_boundary/paired_comparison.json': BB_VALUES.parent / 'paired_comparison.json',
                'validation_suite/rows.csv': VS_ROWS, 'position_ablation/predictions.csv': PA_PRED,
                'postkill_full/run.json': EXPOSURES.parent / 'run.json',
                'state_value_main_50k_eval/match_splits.json': OLD_EVAL / 'match_splits.json'}

os.environ['LOL_OUTPUT_ROOT'] = str(OUT_BASE / 'runtime')
os.environ['LOL_CFG_PRESET'] = 'v3.3'
os.environ['LOL_CFG_OVERRIDES'] = json.dumps({
    'CACHE_DIRNAME': str(CACHE), 'FIGHT_INDEX_CACHE_ENABLED': False,
    'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False, 'CACHE_IN_RAM': False})


# ----------------------------------------------------------------------------- utilities
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
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def rank(tag, key):
    return hashlib.sha256(f'{tag}:{key}'.encode()).hexdigest()


def inf_to_int(x, missing=-1):
    return missing if x == math.inf else int(x)


class Run:
    def __init__(self, out, smoke):
        self.out, self.smoke, self.started = out, smoke, time.time()
        (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (out / 'logs').mkdir(parents=True, exist_ok=True)
        self.status = {'stage': 'starting', 'pid': os.getpid(), 'smoke': smoke, 'role': ROLE, 'history': [], 'blockers': []}

    def log(self, msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")} +{time.time() - self.started:7.0f}s] {msg}'
        print(line, flush=True)
        with (self.out / 'logs' / 'run.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    def stage(self, name, quiet=False, **info):
        self.status.update(stage=name, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                           elapsed_seconds=round(time.time() - self.started, 1), detail=_jsonable(info))
        if not self.status['history'] or self.status['history'][-1]['stage'] != name:
            self.status['history'].append({'stage': name, 'at': self.status['updated']})
        write_json(self.out / 'checkpoints' / 'status.json', self.status)
        if not quiet:
            self.log(f'stage={name} {json.dumps(_jsonable(info), ensure_ascii=False)[:600]}')

    def blocker(self, stage, exc):
        rec = {'stage': stage, 'error': repr(exc), 'traceback': traceback.format_exc(), 'at': time.strftime('%Y-%m-%d %H:%M:%S')}
        self.status['blockers'].append({k: rec[k] for k in ('stage', 'error', 'at')})
        errors = read_json(self.out / 'errors.json') if (self.out / 'errors.json').exists() else {}
        errors.setdefault('blockers', []).append(rec)
        write_json(self.out / 'errors.json', errors)
        self.log(f'BLOCKER in {stage}: {exc!r}')
        write_json(self.out / 'checkpoints' / 'status.json', self.status)


def run_pool(run, fn, tasks, workers, stage, label):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results, failures = [], {}
    if not tasks:
        return results, failures
    os.environ['P3_WORKER'] = '1'
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
        os.environ['P3_WORKER'] = '0'
    return results, failures


def chunked(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


# ----------------------------------------------------------------------------- hashing
def _hash_tree(base):
    if not base.exists():
        return {}
    skip = {'runtime', 'pytest_basetemp', 'dev_pytest_basetemp', '__pycache__'}
    out = {}
    for p in sorted(base.rglob('*')):
        rel = p.relative_to(base)
        if p.is_file() and not (set(rel.parts) & skip):
            out[str(rel).replace('\\', '/')] = sha256_file(p)
    return out


def hash_cache_task(task):
    return {mid: {ext: sha256_file(CACHE / f'{mid}{ext}') for ext in ('.events.json', '.meta.json', '.npz')}
            for mid in task['matches']}


def frozen_hashes(run, cohort_matches):
    out = {'worktree_sources': {p: sha256_file(WT / p) for p in WT_SOURCES},
           'root_sources_and_docs': {p: sha256_file(ROOT / p) for p in ROOT_SOURCES},
           'frozen_files': {k: sha256_file(p) for k, p in FROZEN_FILES.items()},
           'original_repository_detector_sources': {str(p.relative_to(ORIGINAL_REPO)).replace('\\', '/'): sha256_file(p)
                                                    for folder in ('core', 'data', 'gameplay')
                                                    for p in sorted((ORIGINAL_REPO / folder).glob('*.py'))}}
    for k, base in FROZEN_TREES.items():
        out[k] = _hash_tree(base)
    tasks = [{'matches': c} for c in chunked(sorted(cohort_matches), 400)]
    res, fail = run_pool(run, hash_cache_task, tasks, WORKERS, run.status['stage'], 'hash raw cache')
    if fail:
        raise RuntimeError(f'raw cache hashing failed: {fail}')
    cache = {}
    for r in res:
        cache.update(r)
    out['raw_cache_cohort_files'] = dict(sorted(cache.items()))
    out['raw_cache_cohort_files_sha256'] = sha256_json(out['raw_cache_cohort_files'])
    return out


def compare_hashes(before, after):
    changed = []
    for group, d in before.items():
        if not isinstance(d, dict):
            if d != after.get(group):
                changed.append(group)
            continue
        a = after.get(group, {})
        for k in sorted(set(d) | set(a)):
            if d.get(k) != a.get(k):
                changed.append(f'{group}/{k}')
    return changed


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
        from gameplay.state_value_v2 import STATE_VERSION
        import train.temporal_history_winprob_v3 as T
        _CTX.update(load=load_match_cache, final_outcome=final_outcome, T=T, node_names=list(NODE_FEATURE_NAMES),
                    state_version=STATE_VERSION, feature_version=str(getattr(cfg, 'FEATURE_VERSION', '')))
    return _CTX


def _frozen(task):
    ctx = _context()
    T = ctx['T']
    key = (task['a_path'], task['a_sha256'], task['b_dir'], task['b_sha256'])
    if ctx.get('frozen_key') != key:
        A = T.CandidateA.load(task['a_path'], task['a_sha256'])
        B = T.CandidateB.load(task['b_dir'], task['b_sha256'])
        ctx.update(frozen_key=key, A=A, B=B, sA=T.TemporalWinProbabilityService(A), sB=T.TemporalWinProbabilityService(B))
    return ctx['A'], ctx['B'], ctx['sA'], ctx['sB']


def _load_raw_events(mid):
    return json.loads((CACHE / f'{mid}.events.json').read_text(encoding='utf-8'))


def boundary_chunk(task):
    """Raw-event boundary reconstruction (read-only JSON + frame timestamps). Never drops a row."""
    started = time.time()
    out, errors = [], {}
    for mid, rows, match_exposures in task['items']:
        try:
            events = _load_raw_events(mid)
            with np.load(CACHE / f'{mid}.npz', allow_pickle=False) as z:
                mts = np.asarray(z['minute_ts'], dtype=np.int64)
            support_start, last_frame = max(0, int(mts[0])), int(mts[-1])
            kills = sorted(int(e['timestamp']) for e in events if e.get('type') == 'CHAMPION_KILL')
            kc = Counter(kills)
            ends = sorted(int(e['timestamp']) for e in events if e.get('type') == 'GAME_END')
            end_semantics = max(ends) if ends else max(int(e['timestamp']) for e in events)
            cat_events, nonkill_ts = [], Counter()
            for e in events:
                cats, team = R.categorize(e)
                if cats:
                    cat_events.append((int(e['timestamp']), cats, team))
                if e.get('type') in R.NON_KILL_PREDICTOR_TYPES:
                    nonkill_ts[int(e['timestamp'])] += 1
            cat_events.sort(key=lambda x: x[0])
            exp_sorted = sorted(match_exposures, key=lambda x: (x['s'], x['L']))
            next_map = {x['s']: (exp_sorted[i + 1]['s'] if i + 1 < len(exp_sorted) else end_semantics, i + 1 < len(exp_sorted))
                        for i, x in enumerate(exp_sorted)}
            for r in rows:
                s, L = int(r['s']), int(r['L'])
                q_pre, K = s - 1, s + 15_000
                ns_stored, end_stored = int(r['next_start']), int(r['end'])
                ns_recomputed, has_later_exposure = next_map[s]
                game_end = end_semantics
                nk = R.next_kill_after(kills, L)
                ns_eff = R.effective_next_start(ns_stored, game_end)
                rec = {'row_index': int(r['row_index']), 'match': mid, 's_ms': s, 'L_ms': L, 'q_pre_ms': q_pre,
                       'first_kill_ms': K, 'first_kill_raw_count': kc.get(K, 0), 'kills_at_L_count': kc.get(L, 0),
                       'kills_in_first_to_last_inclusive': bisect_right(kills, L) - bisect_right(kills, K - 1),
                       'kills_in_s_to_first_kill_exclusive': bisect_right(kills, K - 1) - bisect_right(kills, s - 1),
                       'next_kill_ms': inf_to_int(nk), 'next_kill_absent': int(nk == math.inf),
                       'next_kill_raw_count': kc.get(nk, 0) if nk != math.inf else 0,
                       'next_start_stored_ms': ns_stored, 'next_start_recomputed_ms': int(ns_recomputed),
                       'next_start_matches_recomputed': int(ns_stored == ns_recomputed),
                       'later_exposure_exists': int(has_later_exposure),
                       'next_engagement_absent_sentinel': int(ns_eff == math.inf),
                       'sentinel_conflict_later_exposure_at_or_after_end': int(has_later_exposure and ns_eff == math.inf),
                       'next_start_effective_ms': inf_to_int(ns_eff),
                       'game_end_ms': int(game_end), 'game_end_stored_exposure_ms': end_stored,
                       'game_end_matches_exposure': int(game_end == end_stored),
                       'game_end_events_count': len(ends), 'game_end_raw_min_ms': ends[0] if ends else -1,
                       'game_end_raw_max_ms': ends[-1] if ends else -1,
                       'game_end_ambiguous': int(len(set(ends)) != 1), 'end_observed_stored': int(r['end_observed']),
                       'same_match_overlap_stored': int(r['same_match_overlap']),
                       'support_start_ms': support_start, 'last_frame_ms': last_frame,
                       'nonkill_events_at_L': nonkill_ts.get(L, 0),
                       'nonkill_events_at_next_kill_ts': nonkill_ts.get(nk, 0) if nk != math.inf else 0,
                       'nonkill_events_at_game_end_ts': nonkill_ts.get(int(game_end), 0),
                       'nonkill_events_at_next_start_ts': nonkill_ts.get(ns_eff, 0) if ns_eff != math.inf else 0}
                during = R.count_events(cat_events, q_pre, L)
                for k in R.COUNT_KEYS:
                    rec[f'during_{k}'] = during.get(k, 0)
                for h in HS:
                    e, reasons, cand = R.endpoint_rule(L, h, nk, ns_eff, game_end)
                    val = R.endpoint_validity(e, L, q_pre, support_start, last_frame)
                    rec[f'endpoint_h{h}_ms'] = e
                    rec[f'end_reasons_h{h}'] = '|'.join(reasons)
                    rec[f'n_tied_reasons_h{h}'] = len(reasons)
                    for reason in R.REASONS:
                        rec[f'reason_{reason}_h{h}'] = int(reason in reasons)
                        rec[f'candidate_{reason}_h{h}_ms'] = inf_to_int(cand[reason])
                    for k, v in val.items():
                        rec[f'{k}_h{h}'] = int(v)
                    rec[f'valid_h{h}'] = int(all(val.values()))
                    after = R.count_events(cat_events, L, e)
                    for k in R.COUNT_KEYS:
                        rec[f'after_h{h}_{k}'] = after.get(k, 0)
                    rec[f'after_h{h}_nonkill_same_timestamp_groups'] = int(sum(1 for t, n in nonkill_ts.items() if L < t <= e and n >= 2))
                    rec[f'after_h{h}_raw_kills'] = bisect_right(kills, e) - bisect_right(kills, L)
                out.append(rec)
        except Exception as exc:
            errors[mid] = f'{type(exc).__name__}: {exc}'
    return {'chunk_id': task['chunk_id'], 'rows': out, 'errors': errors, 'seconds': round(time.time() - started, 1)}


def score_chunk(task):
    """Frozen A and B at q_pre and the three endpoints through the P2 history adapter."""
    started = time.time()
    ctx = _context()
    T = ctx['T']
    A, B, sA, sB = _frozen(task)
    names = task['names']
    out, errors, model_keys = defaultdict(list), {}, set()
    for mid, rows in task['items']:
        try:
            pack = _load(mid, ctx)
            if task.get('perturb_after_ms') is not None:
                pack = _perturb_after(pack, int(task['perturb_after_ms'][mid]))
            winner, terminal = ctx['final_outcome'](pack['events'])
            ts = np.asarray(pack['minute_ts'], dtype=np.int64)
            src = T.BatchSource(mid, pack, ctx['node_names'])
            hists, b1, b2 = [], [], []
            for r in rows:
                q = {'pre': int(r['q_pre']), **{f'h{h}': int(r[f'e{h}']) for h in HS}}
                if not (q['pre'] < q['h60'] <= q['h90'] <= q['h120']):
                    raise ValueError(f'non-monotone or non-positive windows for row {r["row_index"]}')
                hq = {k: T.assemble_history(src, v, names) for k, v in q.items()}
                hists.append(hq)
                b1 += [hq['pre'], hq['h90']]                 # exact P2 endpoint batch composition
                b2 += [hq['pre'], hq['h60'], hq['h120']]     # independent batch (pre recomputed)
            pa1, pb1 = sA.predict_many(b1), sB.predict_many(b1)
            pa2, pb2 = sA.predict_many(b2), sB.predict_many(b2)
        except Exception as exc:
            errors[mid] = f'{type(exc).__name__}: {exc}'
            continue
        for i, r in enumerate(rows):
            # Canonical per-query prediction: P2's exact [pre, post90] batch; a window whose endpoint equals the
            # 90 s endpoint reuses that prediction (same query -> same state); otherwise the second batch.
            # Independently batched values are kept as a batch-composition rounding diagnostic.
            same60, same120 = int(r['e60']) == int(r['e90']), int(r['e120']) == int(r['e90'])
            pa = {'pre': pa1[2 * i], 'h90': pa1[2 * i + 1], 'h60': pa1[2 * i + 1] if same60 else pa2[3 * i + 1],
                  'h120': pa1[2 * i + 1] if same120 else pa2[3 * i + 2]}
            pb = {'pre': pb1[2 * i], 'h90': pb1[2 * i + 1], 'h60': pb1[2 * i + 1] if same60 else pb2[3 * i + 1],
                  'h120': pb1[2 * i + 1] if same120 else pb2[3 * i + 2]}
            for mdl, p2b in (('A', pa2), ('B', pb2)):
                out[f'p_h60_{mdl}_batch2'].append(p2b[3 * i + 1].p)
                out[f'p_h120_{mdl}_batch2'].append(p2b[3 * i + 2].p)
            out['row_index'].append(int(r['row_index']))
            out['match'].append(mid)
            out['winner_blue_AUDIT_ONLY'].append(int(winner))
            out['terminal_ms_AUDIT_ONLY'].append(int(terminal))
            out['last_frame_ms'].append(int(ts[-1]))
            out['support_start_ms'].append(int(max(0, ts[0])))
            for k in QKEYS:
                a, b, hist = pa[k], pb[k], hists[i][k]
                if not (a.snapshot_ms == b.snapshot_ms == hist.snapshot_ms[-1]) or not (a.query_ms == b.query_ms == hist.query_ms):
                    raise ValueError('A/B prediction metadata disagree with the independently assembled history')
                if tuple(b.history_snapshot_ms) != tuple(hist.snapshot_ms) or tuple(b.history_mask) != tuple(hist.mask):
                    raise ValueError('canonical B prediction history differs from the independently assembled history')
                model_keys.add(('A',) + a.model_key())
                model_keys.add(('B',) + b.model_key())
                out[f'{k}_query_ms'].append(hist.query_ms)
                out[f'{k}_snapshot_ms'].append(hist.snapshot_ms[-1])
                out[f'{k}_hist_snapshot_ms'].append(hist.snapshot_ms)
                out[f'{k}_hist_mask'].append(hist.mask)
                out[f'{k}_X_last'].append(hist.matrix()[-1])
                out[f'{k}_unassigned_soul_events'].append(int(hist.states[-1].unassigned_soul_events))
                out[f'p_{k}_A'].append(a.p)
                out[f'p_{k}_B'].append(b.p)
                for s in sorted(b.components):
                    out[f'p_{k}_B_seed{s}'].append(b.components[s])
            out['p_pre_A_batch2'].append(pa2[3 * i].p)
            out['p_pre_B_batch2'].append(pb2[3 * i].p)
            for h in HS:
                out[f'delta_A_h{h}'].append(T.endpoint_delta(pa['pre'], pa[f'h{h}'])['delta'])
                out[f'delta_B_h{h}'].append(T.endpoint_delta(pb['pre'], pb[f'h{h}'])['delta'])
            try:
                T.endpoint_delta(pa['pre'], pb['h90'])
                out['guard_mixed_A_B_delta_rejected'].append(0)
            except T.ModelVersionError:
                out['guard_mixed_A_B_delta_rejected'].append(1)
    summary = {'chunk_id': task['chunk_id'], 'errors': errors, 'rows': len(out['row_index']),
               'seconds': round(time.time() - started, 1), 'written': False, 'model_keys': sorted(model_keys)}
    if errors:
        return summary
    arrays = {k: np.asarray(v) for k, v in out.items() if k != 'match'}
    arrays['match'] = np.asarray(out['match'], dtype='U16')
    arrays['plan_sha256'] = np.asarray(task['plan_sha256'])
    save_npz(Path(task['path']), **arrays)
    summary['written'] = True
    return summary


def _load(mid, ctx):
    pack = ctx['load'](mid)
    if pack is None:
        raise ValueError(f'unreadable_or_incompatible_cache: {mid}')
    return pack


def _perturb_after(pack, t):
    """Future-perturbation guard: rewrite every observation strictly after t (never the past)."""
    rng = np.random.default_rng(int(t) % (2 ** 32))
    pack = dict(pack)
    ts = np.asarray(pack['minute_ts'], dtype=np.int64)
    node = np.array(pack['node_minute'], copy=True)
    future = ts > t
    if future.any():
        keep_champion = node[future][:, :, :].copy()
        node[future] = rng.uniform(0., 1., size=node[future].shape).astype(node.dtype)
        ctx = _context()
        ci = ctx['node_names'].index('champion_id')
        node[np.ix_(np.flatnonzero(future), np.arange(node.shape[1]), [ci])] = keep_champion[:, :, [ci]]
    pack['node_minute'] = node
    events = [dict(e) for e in pack['events'] if int(e.get('timestamp', -1)) <= t]
    tail = [e for e in pack['events'] if int(e.get('timestamp', -1)) > t and e.get('type') == 'GAME_END']
    events.append({'type': 'CHAMPION_KILL', 'timestamp': t + 1, 'killerId': 1, 'victimId': 6})
    events.append({'type': 'ELITE_MONSTER_KILL', 'timestamp': t + 2, 'monsterType': 'BARON_NASHOR', 'killerTeamId': 200, 'killerId': 7})
    events.append({'type': 'BUILDING_KILL', 'timestamp': t + 3, 'teamId': 100, 'buildingType': 'INHIBITOR_BUILDING'})
    pack['events'] = events + tail
    return pack


def raw_audit_chunk(task):
    """Independent re-read of raw events/frames with numpy; checks stored boundaries and snapshots."""
    fails, checked, errors = defaultdict(list), Counter(), {}
    for mid, rows in task['items']:
        try:
            with open(CACHE / f'{mid}.events.json', 'rb') as f:
                events = json.load(f)
            with np.load(CACHE / f'{mid}.npz', allow_pickle=False) as z:
                mts = np.asarray(z['minute_ts'], dtype=np.int64)
            kill_ts = np.sort(np.fromiter((int(e['timestamp']) for e in events if e.get('type') == 'CHAMPION_KILL'), dtype=np.int64))
            ge = np.fromiter((int(e['timestamp']) for e in events if e.get('type') == 'GAME_END'), dtype=np.int64)
            for r in rows:
                key = (mid, int(r['s_ms']))
                s, L = int(r['s_ms']), int(r['L_ms'])
                ns = int(r['next_start_effective_ms'])
                ns = math.inf if ns < 0 else ns

                def check(name, ok):
                    checked[name] += 1
                    if not ok:
                        fails[name].append(key)
                i_after = int(np.searchsorted(kill_ts, L, side='right'))
                nk = int(kill_ts[i_after]) if i_after < len(kill_ts) else math.inf
                check('first_kill_is_raw_kill', bool(np.any(kill_ts == s + 15_000)))
                check('last_kill_is_raw_kill', bool(np.any(kill_ts == L)))
                check('last_kill_not_before_first_kill', L >= s + 15_000)
                check('game_end_observed_unique', len(ge) >= 1 and int(ge.min()) == int(ge.max()))
                g_min = int(ge.min()) if len(ge) else math.inf
                g_max = int(ge.max()) if len(ge) else max(int(e['timestamp']) for e in events)
                es = []
                for h in HS:
                    e = int(r[f'endpoint_h{h}_ms'])
                    es.append(e)
                    n_kill = int(np.searchsorted(kill_ts, e, side='right') - np.searchsorted(kill_ts, L, side='right'))
                    check(f'no_raw_kill_in_L_endpoint_h{h}', n_kill == 0)
                    check(f'endpoint_before_next_kill_h{h}', e < nk)
                    check(f'endpoint_before_next_eligible_start_h{h}', e < ns)
                    check(f'endpoint_before_game_end_h{h}', e < g_min)
                    check(f'endpoint_within_horizon_h{h}', e <= L + 1000 * h)
                    check(f'endpoint_ge_L_h{h}', e >= L)
                    check(f'endpoint_gt_q_pre_h{h}', e > s - 1)
                    check(f'endpoint_tight_recomputed_h{h}', e == min(L + 1000 * h, nk - 1, ns - 1, g_max - 1))
                    check(f'endpoint_within_frames_h{h}', int(mts[0]) <= e <= int(mts[-1]))
                    fi = int(np.searchsorted(mts, e, side='right') - 1)
                    check(f'post_snapshot_equals_latest_frame_le_endpoint_h{h}', int(mts[fi]) == int(r[f'h{h}_snapshot_ms']))
                    check(f'post_snapshot_le_query_h{h}', int(r[f'h{h}_snapshot_ms']) <= e)
                fi = int(np.searchsorted(mts, s - 1, side='right') - 1)
                check('pre_snapshot_equals_latest_frame_le_q_pre', fi >= 0 and int(mts[fi]) == int(r['pre_snapshot_ms']))
                check('pre_snapshot_le_q_pre', int(r['pre_snapshot_ms']) <= s - 1)
                check('endpoints_monotone_60_90_120', es[0] <= es[1] <= es[2])
        except Exception as exc:
            errors[mid] = f'{type(exc).__name__}: {exc}'
    return {'fails': {k: v for k, v in fails.items()}, 'checked': dict(checked), 'errors': errors}


def events_window_task(task):
    out = {}
    for ex in task['examples']:
        events = _load_raw_events(ex['match'])
        lo, hi = int(ex['lo']), int(ex['hi'])
        rows = []
        for e in sorted(events, key=lambda e: int(e.get('timestamp', -1))):
            t = int(e.get('timestamp', -1))
            if lo <= t <= hi and (e.get('type') in ('CHAMPION_KILL', 'GAME_END') or e.get('type') in R.NON_KILL_PREDICTOR_TYPES):
                cats, team = R.categorize(e)
                rows.append({'timestamp_ms': t, 'type': e.get('type'), 'categories': cats, 'team': team,
                             'monsterSubType': e.get('monsterSubType'), 'buildingType': e.get('buildingType'),
                             'killerId': e.get('killerId'), 'victimId': e.get('victimId')})
        out[ex['example_id']] = rows
    return out


# ----------------------------------------------------------------------------- main-side stages
def load_frozen_identity():
    fm = read_json(P2 / 'models' / 'frozen_models.json')
    sel = read_json(P2 / 'selection.json')
    p2val = read_json(P2 / 'validation.json')
    p2res = read_json(P2 / 'results.json')
    return {'frozen_models': fm, 'selection_chosen': sel.get('chosen_candidate', sel.get('chosen')),
            'selection_sha256': sha256_file(P2 / 'selection.json'), 'p2_validation_status': p2val['status'],
            'p2_results_status': p2res['status'], 'a_path': str(P1 / 'models' / 'expanded_model_v2.joblib'),
            'a_sha256': fm['A']['sha256'], 'b_dir': str(P2 / 'models' / 'candidate_B'), 'b_sha256': fm['B']['bundle_sha256'],
            'history_version': fm['history_version'], 'module_version': fm['module_version'],
            'module_sha256_recorded': fm['module_sha256']}


def write_protocol(run, ident, smoke):
    protocol = {
        'version': 'engagement_labels_v3_sensitivity_p3', 'role': ROLE, 'smoke': smoke, 'written_before_execution': True,
        'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'design': 'Codex design (docs/CLAUDE_EXECUTE_P3_LABEL_SENSITIVITY.md); Claude Opus 5 implementation and execution',
        'cohort': {'source': 'outputs/state_value_v2_fix/rows.csv', 'rule': 'exact (match, s) set AND order; no new detection, no dropping',
                   'expected_rows': EXPECTED_ROWS, 'expected_matches': EXPECTED_MATCHES,
                   'L_source': 'unique (match, s) join to outputs/postkill_objective_delay_full/exposures.csv',
                   'historic_selection_to_disclose': 'window_validation h=120 valid rows of the V1 engagement split (via event_boundary_cif.py -> '
                                                     'b_boundary_60_90_120 -> position_ablation -> validation_suite -> regenerate_state_v2); '
                                                     'NOT all engagements'},
        'definitions': {
            'q_pre_ms': 's - 1', 'first_kill_ms': 's + 15000 (confirmed against raw CHAMPION_KILL; simultaneous kills flagged, never used to exclude)',
            'next_kill_ms': 'first raw CHAMPION_KILL timestamp strictly greater than L; absent -> +inf',
            'next_eligible_start_ms': 'stored exposures next_start; sentinel next_start >= game_end means absent (+inf), as event_boundary_cif.py',
            'game_end_ms': 'raw GAME_END timestamp (max if several, matching exposure semantics; ambiguity flagged); compatibility with exposures end checked',
            'endpoint_h_ms': 'min(L + 1000*h, next_kill - 1, next_eligible_start - 1, game_end - 1), integer ms, h in {60, 90, 120}',
            'inclusion': 'state at a query includes frames and predictor events with timestamp <= query (closed), as the B_next_kill rule',
            'tied_reasons': 'every reason whose candidate equals the endpoint is stored (horizon|next_kill|next_engagement_start|game_end)',
            'validity': 'endpoint >= L, endpoint > q_pre, q_pre >= first frame, endpoint <= last frame; any invalid row blocks success',
            'objective_acquisition': 'never a terminal event',
            'event_attribution': 'during engagement = (q_pre, L]; after last kill = (L, endpoint_h]; both teams; no spatial filter',
            'p_pre_A': 'frozen A at q_pre', 'p_post_A_h': 'the SAME frozen A at endpoint_h',
            'delta_A_h': 'p_post_A_h - p_pre_A (train.temporal_history_winprob_v3.endpoint_delta; model-key guarded)',
            'Y_A_h': '1(delta_A_h > 0); exact zero -> 0 (non-improvement), reported separately',
            'near_zero_bands': list(R.NEAR_ZERO_BANDS), 'near_zero_role': 'sensitivity descriptions, NOT label thresholds',
            'B': 'frozen P2 SimpleRNN history ensemble via assemble_history/TemporalWinProbabilityService; diagnostic comparator only; '
                 'A and B are never mixed in one delta',
            'value_semantics': 'estimated probability-point change of final Blue win between two observation times under a frozen model; '
                               'not monetary value, not a causal effect'},
        'primary_and_sensitivity': {'working_primary': 'B90 with model A', 'sensitivity': ['B60', 'B120', 'model B at all windows'],
                                    'selection': 'no horizon or model is selected from these results'},
        'reference_tolerances_declared_before_scoring': {
            'keys_and_order': 'exact', 'endpoints': 'exact integer equality for ALL rows (B90 vs state_value_v2_fix endpoint)',
            'states_and_snapshots': 'bitwise equality (tolerance 0) of pre and post-90 StateV2 vectors vs state_value_v2_fix/states.npz and snapshots vs rows.csv',
            'p2_b90_probabilities': {'fields': ['p_pre_A', 'p_post_A', 'delta_A', 'p_pre_B', 'p_post_B', 'delta_B', 'B seed components'],
                                     'preferred': 'bitwise equality', 'declared_abs_tolerance': P2_PROB_TOLERANCE,
                                     'rule': 'pass iff max |diff| <= tolerance; exact-equality counts reported; a nonzero diff with bitwise-equal '
                                             'states is diagnosed separately as operation-order rounding'},
            'batch_composition': 'pre and post-90 are scored in the exact P2 per-match [pre, post90] batch; post-60/120 in a second batch '
                                 '[pre, post60, post120]; pre batch invariance is audited'},
        'analyses': {
            'weights': 'row-weighted and equal-total-weight-per-match reported distinctly',
            'bootstrap': {'unit': 'match', 'replicates_main': N_BOOT_MAIN, 'replicates_cells': N_BOOT_CELL, 'seed': BOOT_SEED,
                          'interval': 'percentile 95%', 'models': 'fixed; no training/selection uncertainty included'},
            'main_rates': ['sign flips A 60vs90/90vs120/60vs120', 'A vs B label disagreement at 60/90/120'],
            'breakdowns': {'patch': 'stored patch', 'start_time_bands_minutes': TIME_BANDS,
                           'objectives': list(R.OBJECTIVE_CATEGORIES) + list(R.STRUCTURE_CATEGORIES),
                           'sparse_cell_rule': f'< {MIN_CELL_MATCHES} matches -> counts reported, rates flagged sparse/unreliable'},
            'v1_comparison': 'only rows whose (match, s, h) and B-rule endpoint equal outputs/b_boundary_60_90_120 exactly; '
                             'V1 p_pre from window_validation h=-1; disagreement is model-version disagreement, never an error rate',
            'examples': 'actual rows per boundary type chosen by sha256("p3_example:<type>:<match>:<s>") ascending; synthetic examples only where '
                        'no actual case exists and marked SYNTHETIC'},
        'verification': {
            'independent_raw_audit': 'separate numpy re-read of raw events/frames (no additional kill in (L, e], e before next eligible start and '
                                     'game end, tight recomputation, last/first kill are raw kills, monotone endpoints, snapshot <= query)',
            'determinism': f'{DETERMINISM_MATCHES} matches by sha256("p3_determinism:<match>") re-scored in fresh processes; bitwise equality',
            'future_perturbation': f'{PERTURBATION_MATCHES} matches: observations after the match max endpoint rewritten; all stored '
                                   'predictions must be bitwise unchanged',
            'partitions': 'zero match overlap with V1 buckets, P1 and P2 fit/calibrate/select/test and V1 eval value_train/value_validation',
            'hashes': 'frozen inputs, frozen models, P1/P2/V1 artifacts, worktree modules, original detector sources and the cohort raw cache files '
                      'hashed before and after'},
        'frozen_models': ident, 'expected_A_sha256': EXPECTED_A_SHA256,
        'resources': {'workers': WORKERS, 'blas_threads_main': 4, 'blas_threads_worker': 1, 'gpu': 'disabled (CUDA hidden); user GPU queue untouched',
                      'python': sys.executable},
        'outputs': 'outputs/engagement_labels_v3_sensitivity only',
        'claims_not_made': ['labels are exploratory candidates, not validated ground truth or production replacement',
                            'no causal effect of engagements or objectives', 'objectives after the last kill are not attributed to the engagement',
                            'larger windows do not imply causal acquisition', 'the cohort is not all engagements (historic availability/selection)',
                            'no q model trained; no horizon/model selected on these outcomes'],
        'references': {
            'Maymin (2021), DOI 10.1515/jqas-2019-0096': 'state-based win probability and valuing actions by win-probability change (rationale for delta)',
            'operational B rule': 'OUR event-based design (docs/LABEL_ENDPOINT_RULE_AND_EXAMPLES_20260914.md); not a rule proven by Maymin or any paper',
            'Austin, Lee, Fine (2016), DOI 10.1161/CIRCULATIONAHA.115.017719': 'competing-event diagnostics motivation only; does not establish attribution',
            'Decroos et al. (2019), DOI 10.1145/3292500.3330758': 'precedent for an explicitly chosen evaluation horizon; not a justification of 60/90/120 s'},
    }
    protocol['protocol_sha256'] = sha256_json(protocol)
    write_json(run.out / 'protocol.json', protocol)
    return protocol


def cohort_lineage(run, smoke):
    import pandas as pd
    v2 = pd.read_csv(V2FIX / 'rows.csv', dtype={'patch': str})
    c = {}
    c['v2_rows'] = len(v2)
    c['v2_matches'] = int(v2.match.nunique())
    c['v2_keys_unique'] = bool(not v2.duplicated(['match', 's']).any())
    c['v2_rows_expected'] = c['v2_rows'] == EXPECTED_ROWS and c['v2_matches'] == EXPECTED_MATCHES
    c['v2_rows_sha256'] = sha256_file(V2FIX / 'rows.csv')
    v2['row_index'] = np.arange(len(v2))
    ex = pd.read_csv(EXPOSURES, dtype={'patch': str})
    c['exposures_rows'] = len(ex)
    c['exposures_matches'] = int(ex.match.nunique())
    c['exposures_key_duplicates'] = int(ex.duplicated(['match', 's']).sum())
    joined = v2.merge(ex, on=['match', 's'], how='left', validate='one_to_one', suffixes=('', '_exposure'), indicator=True)
    c['v2_rows_without_exposure'] = int((joined['_merge'] != 'both').sum())
    joined = joined.drop(columns=['_merge']).sort_values('row_index', kind='mergesort').reset_index(drop=True)
    c['join_order_preserved'] = bool((joined.row_index.to_numpy() == np.arange(len(v2))).all())
    c['patch_equal_exposure'] = bool((joined.patch == joined.patch_exposure).all())
    c['cohort_same_match_overlap_rows'] = int(joined.same_match_overlap.sum())
    c['cohort_end_observed_zero_rows'] = int((joined.end_observed != 1).sum())
    # Historic chain: key identity and order at every hop.
    def keys(df):
        return list(zip(df.match.tolist(), df.s.astype(np.int64).tolist()))
    k2 = keys(v2)
    vs = pd.read_csv(VS_ROWS, dtype={'patch': str})
    pa = pd.read_csv(PA_PRED, dtype={'patch': str})
    bb = pd.read_csv(BB_VALUES)
    wv = pd.read_csv(WV_LABELS, dtype={'patch': str})
    bb90 = bb[(bb.h == 90) & (bb.valid == 1)]
    wv120 = wv[(wv.h == 120) & (wv.valid == 1)]
    chain = {}
    for name, df in [('validation_suite_rows', vs), ('position_ablation_predictions', pa), ('b_boundary_h90_valid', bb90),
                     ('window_validation_h120_valid', wv120)]:
        kk = keys(df)
        chain[name] = {'rows': len(kk), 'same_set': set(kk) == set(k2), 'same_order': kk == k2}
    chain['v2_endpoint_equals_b_boundary_h90_endpoint'] = bool((bb90.endpoint.to_numpy() == v2.endpoint.to_numpy()).all()) if chain['b_boundary_h90_valid']['same_order'] else None
    chain['v2_endpoint_equals_validation_suite_endpoint'] = bool((vs.endpoint.to_numpy() == v2.endpoint.to_numpy()).all()) if chain['validation_suite_rows']['same_order'] else None
    c['historic_chain'] = chain
    wv_L = wv[wv.h == -1][['match', 's', 'L']]
    m = joined[['match', 's', 'L']].merge(wv_L, on=['match', 's'], suffixes=('', '_wv'), validate='one_to_one')
    c['L_equals_window_validation_L'] = bool((m.L == m.L_wv).all()) and len(m) == len(joined)
    # Census of the parent population (no new detection; reconstructable exclusions only).
    proto = read_json(VD / 'protocol.json')
    eng = set(proto['splits']['engagement'])
    splits_eval = read_json(OLD_EVAL / 'match_splits.json')
    ex_eng = ex[ex.match.isin(eng)]
    wv_keys = wv[wv.h == -1]
    overlap_rows = wv_keys[wv_keys.overlap == 1]
    invalid120 = wv[(wv.h == 120) & (wv.valid == 0)]
    no_exposure = sorted(eng - set(ex.match))
    census = {
        'v1_engagement_split_matches': len(eng),
        'v1_engagement_split_equals_eval_predict_test': eng == set(splits_eval['predict_test']),
        'engagement_matches_with_exposure_rows': int(ex_eng.match.nunique()),
        'engagement_matches_without_any_exposure_row': len(no_exposure),
        'engagement_matches_without_exposure_cache_files_present': int(sum((CACHE / f'{m_}.meta.json').exists() for m_ in no_exposure)),
        'engagement_matches_without_exposure_note': 'no detector engagement row in the parent exposures (postkill_full run.json: 210000 loaded, 0 errors); '
                                                   'cause not re-derived because re-detection is out of scope',
        'exposure_rows_in_engagement_matches': len(ex_eng),
        'window_validation_engagements': int(len(wv_keys)),
        'window_validation_equals_exposure_rows': set(keys(wv_keys)) == set(keys(ex_eng)),
        'excluded_same_match_overlap_rows (next_start <= L)': int(len(overlap_rows)),
        'excluded_overlap_keys': [[a, int(b)] for a, b in keys(overlap_rows)],
        'window_validation_h120_invalid_rows': int(len(invalid120)),
        'h120_invalid_rows_all_overlap': set(keys(invalid120)) == set(keys(overlap_rows)),
        'other_availability_exclusions (endpoint beyond last frame / negative)': int(len(invalid120) - len(overlap_rows)),
        'cohort_rows': len(v2), 'cohort_matches': c['v2_matches'],
        'cohort_equals_exposure_rows_minus_overlap': set(k2) == (set(keys(ex_eng)) - set(keys(overlap_rows))),
        'patch_rows_cohort': v2.patch.value_counts().sort_index().to_dict(),
        'patch_rows_all_exposures': ex.patch.value_counts().sort_index().to_dict(),
        'patch_matches_engagement_split_with_exposures': ex_eng.groupby('patch').match.nunique().to_dict(),
        'corpus_note': 'V1 engagement split = predict_test of the 50k exploratory corpus (patches 15.14-15.16); the parent exposures cover the full '
                       '210k cached corpus. The cohort is therefore a split-restricted, overlap-excluded subset, NOT all engagements.'}
    c['census'] = census
    # Row provenance from P2 endpoint diagnostics (same order).
    with np.load(P2 / 'endpoint_diagnostics' / 'endpoint_rows_DIAGNOSTIC.npz', allow_pickle=False) as z:
        c['p2_endpoint_row_index_equals_v2_order'] = bool((z['row_index'] == np.arange(len(v2))).all())
        c['p2_endpoint_match_equals_v2'] = bool((z['match'] == v2.match.to_numpy().astype('U16')).all())
        c['p2_endpoint_post_query_equals_v2_endpoint'] = bool((z['post_query_ms'] == v2.endpoint.to_numpy()).all())
    fail = []
    if not (c['v2_rows_expected'] and c['v2_keys_unique'] and c['v2_rows_without_exposure'] == 0 and c['join_order_preserved']
            and c['patch_equal_exposure'] and c['cohort_same_match_overlap_rows'] == 0 and c['L_equals_window_validation_L']):
        fail.append('cohort/exposure join')
    if not all(v['same_order'] for k, v in chain.items() if isinstance(v, dict)) or not chain['v2_endpoint_equals_b_boundary_h90_endpoint']:
        fail.append('historic chain key/order/endpoint identity')
    if not (c['p2_endpoint_row_index_equals_v2_order'] and c['p2_endpoint_match_equals_v2'] and c['p2_endpoint_post_query_equals_v2_endpoint']):
        fail.append('P2 endpoint diagnostics row identity')
    if smoke:
        mids = sorted(joined.match.unique(), key=lambda mm: rank('p3_smoke', mm))[:smoke]
        joined = joined[joined.match.isin(set(mids))].reset_index(drop=True)
    return joined, ex, c, fail


def run_boundaries(run, cohort, ex, workers):
    import pandas as pd
    by_match = defaultdict(list)
    for r in cohort[['row_index', 'match', 's', 'L', 'next_start', 'end', 'same_match_overlap', 'end_observed']].to_dict('records'):
        by_match[r['match']].append(r)
    exm = ex[ex.match.isin(set(by_match))][['match', 's', 'L']]
    exp_by_match = defaultdict(list)
    for r in exm.to_dict('records'):
        exp_by_match[r['match']].append({'s': int(r['s']), 'L': int(r['L'])})
    items = [(m, by_match[m], exp_by_match[m]) for m in sorted(by_match)]
    tasks = [{'chunk_id': i, 'items': ch} for i, ch in enumerate(chunked(items, 200))]
    res, fail = run_pool(run, boundary_chunk, tasks, workers, 'boundaries', 'raw boundary reconstruction')
    errors = dict(fail)
    for r in res:
        errors.update(r['errors'])
    if errors:
        write_json(run.out / 'checkpoints' / 'errors_boundaries.json', errors)
        raise RuntimeError(f'boundary reconstruction failed for {len(errors)} matches/chunks')
    rows = [row for r in res for row in r['rows']]
    bnd = pd.DataFrame(rows).sort_values('row_index', kind='mergesort').reset_index(drop=True)
    if not (bnd.row_index.to_numpy() == cohort.row_index.to_numpy()).all():
        raise RuntimeError('boundary rows differ from the cohort rows')
    bnd.insert(2, 'patch', cohort.patch.to_numpy())
    bnd.to_csv(run.out / 'checkpoints' / 'boundaries.csv', index=False)
    return bnd


def b90_reference_audit(run, cohort, bnd):
    import pandas as pd
    ref = cohort.endpoint.to_numpy()
    mine = bnd['endpoint_h90_ms'].to_numpy()
    mism = np.flatnonzero(ref != mine)
    out = {'rows_compared': int(len(ref)), 'all_rows_compared': True, 'mismatches': int(len(mism)),
           'mismatch_rows': [{'row_index': int(bnd.row_index[i]), 'match': bnd.match[i], 's': int(bnd.s_ms[i]),
                              'reference_endpoint': int(ref[i]), 'reconstructed_endpoint': int(mine[i]),
                              'reconstructed_reasons': bnd['end_reasons_h90'][i]} for i in mism[:500]],
           'reference_data_overwritten': False}
    bb = pd.read_csv(BB_VALUES)
    for h in HS:
        b = bb[bb.h == h][['match', 's', 'endpoint']].merge(bnd[['match', 's_ms', f'endpoint_h{h}_ms']].rename(columns={'s_ms': 's'}),
                                                          on=['match', 's'], validate='one_to_one')
        out[f'b_boundary_v1_run_h{h}_rows'] = int(len(b))
        out[f'b_boundary_v1_run_h{h}_endpoint_mismatches'] = int((b.endpoint != b[f'endpoint_h{h}_ms']).sum())
    a_style = np.minimum(np.minimum(bnd.L_ms + 90_000, bnd.next_start_stored_ms - 1), bnd.game_end_stored_exposure_ms - 1)
    out['guard_A_style_next_engagement_only_rule_differs_rows'] = int((a_style.to_numpy() != mine).sum())
    out['guard_note'] = ('validate_windows.py A-style endpoint min(L+h, next_start-1, end-1) ignores the next raw kill; the count above shows '
                         'rows where it would differ from B90, demonstrating it was not used.')
    write_json(run.out / 'reference_audit_b90_endpoints.json', out)
    return out


def run_scoring(run, bnd, names, ident, workers, subdir='score_chunks', match_filter=None, perturb=None, chunk_size=150, stage='scoring'):
    d = run.out / 'checkpoints' / subdir
    d.mkdir(parents=True, exist_ok=True)
    by_match = defaultdict(list)
    cols = ['row_index', 'match', 'q_pre_ms'] + [f'endpoint_h{h}_ms' for h in HS]
    for r in bnd[cols].to_dict('records'):
        if match_filter is not None and r['match'] not in match_filter:
            continue
        by_match[r['match']].append({'row_index': int(r['row_index']), 'q_pre': int(r['q_pre_ms']),
                                     **{f'e{h}': int(r[f'endpoint_h{h}_ms']) for h in HS}})
    items = [(m, by_match[m]) for m in sorted(by_match)]
    frozen = {k: ident[k] for k in ('a_path', 'a_sha256', 'b_dir', 'b_sha256')}
    tasks, reused, paths = [], 0, []
    for c, chunk in enumerate(chunked(items, chunk_size)):
        plan_sha = sha256_json([names, chunk, frozen, perturb and {m: perturb[m] for m, _ in chunk}])
        path = d / f'chunk_{c:04d}.npz'
        paths.append(path)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z['plan_sha256']) == plan_sha:
                    reused += 1
                    continue
            raise ValueError(f'stale scoring checkpoint {path}')
        tasks.append(dict(frozen, names=names, items=chunk, chunk_id=c, path=str(path), plan_sha256=plan_sha,
                          perturb_after_ms=perturb and {m: perturb[m] for m, _ in chunk}))
    run.stage(stage, chunks_total=len(paths), chunks_reused=reused)
    res, fail = run_pool(run, score_chunk, tasks, workers, stage, 'frozen A/B scoring')
    errors = dict(fail)
    keys = set()
    for s in res:
        errors.update(s['errors'])
        keys.update(tuple(k) for k in s['model_keys'])
    if errors:
        write_json(run.out / 'checkpoints' / f'errors_{subdir}.json', errors)
        raise RuntimeError(f'scoring failed for {len(errors)} matches/chunks')
    parts = defaultdict(list)
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                if k != 'plan_sha256':
                    parts[k].append(z[k])
    S = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(S['row_index'], kind='mergesort')
    return {k: v[order] for k, v in S.items()}, sorted(keys)


def p2_reference_audit(run, S, cohort, names):
    import pandas as pd
    ri = S['row_index']
    c = {'rows': int(len(ri)), 'row_index_equals_cohort': bool((ri == cohort.row_index.to_numpy()).all()),
         'match_equals_cohort': bool((S['match'] == cohort.match.to_numpy().astype('U16')).all())}
    with np.load(V2FIX / 'states.npz', allow_pickle=False) as z:
        c['state_names_equal_v2_fix'] = z['names'].tolist() == list(names)
        c['pre_states_bitwise_equal_v2_fix'] = bool(np.array_equal(z['pre'][ri], S['pre_X_last']))
        c['post90_states_bitwise_equal_v2_fix'] = bool(np.array_equal(z['post'][ri], S['h90_X_last']))
        c['pre_state_rows_not_bitwise_equal'] = int((z['pre'][ri] != S['pre_X_last']).any(axis=1).sum())
        c['post90_state_rows_not_bitwise_equal'] = int((z['post'][ri] != S['h90_X_last']).any(axis=1).sum())
    c['pre_snapshot_equal_rows_csv'] = bool((S['pre_snapshot_ms'] == cohort.pre_snapshot.to_numpy()).all())
    c['post90_snapshot_equal_rows_csv'] = bool((S['h90_snapshot_ms'] == cohort.post_snapshot.to_numpy()).all())
    c['pre_unassigned_soul_equal_rows_csv'] = bool((S['pre_unassigned_soul_events'] == cohort.pre_unassigned_soul_events.to_numpy()).all())
    c['post90_unassigned_soul_equal_rows_csv'] = bool((S['h90_unassigned_soul_events'] == cohort.post_unassigned_soul_events.to_numpy()).all())
    c['post90_query_equals_v2_endpoint'] = bool((S['h90_query_ms'] == cohort.endpoint.to_numpy()).all())
    p = {}
    with np.load(P2 / 'endpoint_diagnostics' / 'endpoint_rows_DIAGNOSTIC.npz', allow_pickle=False) as z:
        Z = {k: z[k][ri] for k in z.files if z[k].ndim >= 1 and len(z[k]) == EXPECTED_ROWS}
    pairs = [('p_pre_A', 'p_pre_A'), ('p_post_A', 'p_h90_A'), ('delta_A', 'delta_A_h90'), ('p_pre_B', 'p_pre_B'),
             ('p_post_B', 'p_h90_B'), ('delta_B', 'delta_B_h90')]
    pairs += [(f'p_pre_B_seed{s}', f'p_pre_B_seed{s}') for s in (17, 29, 43)] + [(f'p_post_B_seed{s}', f'p_h90_B_seed{s}') for s in (17, 29, 43)]
    for zk, sk in pairs:
        diff = np.abs(Z[zk] - S[sk])
        p[zk] = {'rows_exactly_equal': int((Z[zk] == S[sk]).sum()), 'max_abs_diff': float(diff.max()), 'within_declared_tolerance': bool(diff.max() <= P2_PROB_TOLERANCE)}
    ints = [('pre_query_ms', 'pre_query_ms'), ('post_query_ms', 'h90_query_ms'), ('pre_hist_snapshot_ms', 'pre_hist_snapshot_ms'),
            ('post_hist_snapshot_ms', 'h90_hist_snapshot_ms'), ('winner', 'winner_blue_AUDIT_ONLY'), ('terminal_ms', 'terminal_ms_AUDIT_ONLY'),
            ('last_frame_ms', 'last_frame_ms')]
    for zk, sk in ints:
        p[zk] = {'exactly_equal': bool(np.array_equal(Z[zk], S[sk]))}
    c['p2_b90_identity'] = p
    c['p2_probabilities_all_bitwise_equal'] = all(v['rows_exactly_equal'] == len(ri) for k, v in p.items() if 'rows_exactly_equal' in v)
    c['p2_probabilities_within_declared_tolerance'] = all(v['within_declared_tolerance'] for v in p.values() if 'within_declared_tolerance' in v)
    c['p2_integer_fields_equal'] = all(v['exactly_equal'] for v in p.values() if 'exactly_equal' in v)
    if not c['p2_probabilities_all_bitwise_equal'] and c['pre_states_bitwise_equal_v2_fix'] and c['post90_states_bitwise_equal_v2_fix']:
        c['rounding_diagnosis'] = 'states bitwise equal; nonzero probability differences attributed to operation-order rounding (see max_abs_diff)'
    ok = (c['row_index_equals_cohort'] and c['match_equals_cohort'] and c['state_names_equal_v2_fix'] and c['pre_states_bitwise_equal_v2_fix']
          and c['post90_states_bitwise_equal_v2_fix'] and c['pre_snapshot_equal_rows_csv'] and c['post90_snapshot_equal_rows_csv']
          and c['pre_unassigned_soul_equal_rows_csv'] and c['post90_unassigned_soul_equal_rows_csv'] and c['post90_query_equals_v2_endpoint']
          and c['p2_probabilities_within_declared_tolerance'] and c['p2_integer_fields_equal'])
    c['pass'] = bool(ok)
    write_json(run.out / 'reference_audit_p2_b90_values.json', c)
    return c


def build_labels(run, bnd, S, ident, names, model_keys, cohort):
    import pandas as pd
    N = len(bnd)
    source_hash = {}
    fh = read_json(run.out / 'hashes_before.json')['raw_cache_cohort_files']
    for m_, d in fh.items():
        source_hash[m_] = sha256_json(d)
    pre_age = (S['pre_query_ms'] - S['pre_snapshot_ms']) / 1000.
    long_rows = []
    base_cols = [c for c in bnd.columns if not (c.startswith('after_h') or any(c.endswith(f'_h{h}') or f'_h{h}_' in c for h in HS))]
    for h in HS:
        base = bnd[base_cols].copy()
        base.insert(3, 'h_s', h)
        df = {}
        df['endpoint_ms'] = bnd[f'endpoint_h{h}_ms']
        df['end_reasons'] = bnd[f'end_reasons_h{h}']
        df['n_tied_reasons'] = bnd[f'n_tied_reasons_h{h}']
        for reason in R.REASONS:
            df[f'reason_{reason}'] = bnd[f'reason_{reason}_h{h}']
            df[f'candidate_{reason}_ms'] = bnd[f'candidate_{reason}_h{h}_ms']
        for k in ('endpoint_ge_L', 'endpoint_gt_q_pre', 'endpoint_le_last_frame', 'q_pre_ge_support', 'q_pre_le_last_frame', 'valid'):
            df[k] = bnd[f'{k}_h{h}']
        df['duration_after_last_kill_s'] = (bnd[f'endpoint_h{h}_ms'] - bnd.L_ms) / 1000.
        df['window_from_q_pre_s'] = (bnd[f'endpoint_h{h}_ms'] - bnd.q_pre_ms) / 1000.
        df['pre_snapshot_ms'] = S['pre_snapshot_ms']
        df['pre_snapshot_age_s'] = pre_age
        df['post_snapshot_ms'] = S[f'h{h}_snapshot_ms']
        df['post_snapshot_age_s'] = (S[f'h{h}_query_ms'] - S[f'h{h}_snapshot_ms']) / 1000.
        df['post_snapshot_after_L'] = (S[f'h{h}_snapshot_ms'] > bnd.L_ms.to_numpy()).astype(int)
        df['same_snapshot_frame_pre_post'] = (S[f'h{h}_snapshot_ms'] == S['pre_snapshot_ms']).astype(int)
        df['pre_unassigned_soul_events'] = S['pre_unassigned_soul_events']
        df['post_unassigned_soul_events'] = S[f'h{h}_unassigned_soul_events']
        df['pre_valid_history_positions_B'] = S['pre_hist_mask'].sum(axis=1)
        df['post_valid_history_positions_B'] = S[f'h{h}_hist_mask'].sum(axis=1)
        for mdl in ('A', 'B'):
            pre, post, delta = S[f'p_pre_{mdl}'], S[f'p_h{h}_{mdl}'], S[f'delta_{mdl}_h{h}']
            df[f'p_pre_{mdl}'] = pre
            df[f'p_post_{mdl}'] = post
            df[f'delta_{mdl}'] = delta
            df[f'Y_{mdl}'] = (delta > 0).astype(int)
            df[f'delta_{mdl}_exact_zero'] = (delta == 0).astype(int)
            df[f'delta_{mdl}_negative'] = (delta < 0).astype(int)
            for b in R.NEAR_ZERO_BANDS:
                df[f'abs_delta_{mdl}_le_{b}'] = (np.abs(delta) <= b).astype(int)
        for s in (17, 29, 43):
            df[f'p_pre_B_seed{s}'] = S[f'p_pre_B_seed{s}']
            df[f'p_post_B_seed{s}'] = S[f'p_h{h}_B_seed{s}']
        df['label_disagreement_A_vs_B'] = (df['Y_A'] != df['Y_B']).astype(int)
        for k in R.COUNT_KEYS:
            df[f'after_last_kill_{k}'] = bnd[f'after_h{h}_{k}']
        df['after_last_kill_nonkill_same_timestamp_groups'] = bnd[f'after_h{h}_nonkill_same_timestamp_groups']
        df['after_last_kill_raw_kills'] = bnd[f'after_h{h}_raw_kills']
        df['reference_v2_fix_endpoint_ms'] = cohort.endpoint.to_numpy() if h == 90 else -1
        df['reference_b90_endpoint_equal'] = (bnd.endpoint_h90_ms.to_numpy() == cohort.endpoint.to_numpy()).astype(int) if h == 90 else -1
        df['model_A_version'] = ident['frozen_models']['A']['model_version']
        df['model_A_sha256'] = ident['a_sha256']
        df['model_B_version'] = ident['frozen_models']['B']['model_version']
        df['model_B_bundle_sha256'] = ident['b_sha256']
        df['history_version'] = ident['history_version']
        df['state_version'] = model_keys_state_version(model_keys)
        df['source_cache_files_sha256'] = base.match.map(source_hash)
        df['role'] = ROLE
        long_rows.append(pd.concat([base, pd.DataFrame(df, index=base.index)], axis=1))
    L = pd.concat(long_rows, ignore_index=True).sort_values(['row_index', 'h_s'], kind='mergesort').reset_index(drop=True)
    return L


def model_keys_state_version(model_keys):
    versions = {k[-1] for k in model_keys}
    if len(versions) != 1:
        raise ValueError(f'multiple state versions in predictions: {versions}')
    return next(iter(versions))


FORBIDDEN_FEATURE_TOKENS = ('endpoint', 'end_reason', 'L_ms', 'last_kill', 'next_kill', 'next_start', 'game_end', 'winner', 'terminal',
                            'post', 'delta', 'Y_', 'after_last_kill', 'during_', 'duration', 'participant_membership', 'reason')


def write_features_and_schema(run, S, names, cohort, labels, ident):
    names = list(names)
    ci = names.index('snapshot_age_s')
    keep = [i for i, n in enumerate(names) if n != 'snapshot_age_s']
    input_names = [names[i] for i in keep] + ['p_pre_A']
    X = np.column_stack([S['pre_X_last'][:, keep], S['p_pre_A']])
    categorical = [n for n in input_names if n.endswith('champion_id')]
    bad = [n for n in input_names if any(tok in n for tok in FORBIDDEN_FEATURE_TOKENS) and n != 'p_pre_A']
    guards = {'input_columns': len(input_names), 'state_predictors': len(keep), 'forbidden_token_columns': bad,
              'snapshot_age_s_in_inputs': 'snapshot_age_s' in input_names, 'position_columns_added': [n for n in input_names if n.endswith(('_x', '_y')) or 'position' in n],
              'categorical_champion_columns': len(categorical), 'finite': bool(np.isfinite(X).all()),
              'inputs_bitwise_equal_v2_fix_pre_minus_age': None, 'p_pre_A_column_equals_scored': bool(np.array_equal(X[:, -1], S['p_pre_A']))}
    with np.load(V2FIX / 'states.npz', allow_pickle=False) as z:
        guards['inputs_bitwise_equal_v2_fix_pre_minus_age'] = bool(np.array_equal(z['pre'][S['row_index']][:, keep], X[:, :-1]))
    guards['pass'] = bool(not bad and not guards['snapshot_age_s_in_inputs'] and not guards['position_columns_added'] and guards['finite']
                          and guards['inputs_bitwise_equal_v2_fix_pre_minus_age'] and guards['p_pre_A_column_equals_scored'] and len(categorical) == 10)
    save_npz(run.out / 'features_pre_only.npz', X_input=X, input_names=np.asarray(input_names), categorical_input_names=np.asarray(categorical),
             row_index=S['row_index'], match=S['match'], s_ms=cohort.s.to_numpy().astype(np.int64), patch_split_metadata=cohort.patch.to_numpy().astype('U8'),
             pre_query_ms=S['pre_query_ms'], role=np.asarray('PRE_ONLY_INPUTS_EXPLORATORY'))
    save_npz(run.out / 'audit_only' / 'pre_snapshot_age_s_AUDIT_ONLY.npz', row_index=S['row_index'], snapshot_age_s=S['pre_X_last'][:, ci],
             pre_snapshot_ms=S['pre_snapshot_ms'])
    post = {f'post_state_h{h}': S[f'h{h}_X_last'] for h in HS}
    save_npz(run.out / 'audit_only' / 'states_pre_post_AUDIT_ONLY_NOT_Q_FEATURES.npz', row_index=S['row_index'], names=np.asarray(names),
             pre_state=S['pre_X_last'], **post, **{f'h{h}_hist_snapshot_ms': S[f'h{h}_hist_snapshot_ms'] for h in HS},
             pre_hist_snapshot_ms=S['pre_hist_snapshot_ms'], winner_blue_AUDIT_ONLY=S['winner_blue_AUDIT_ONLY'],
             terminal_ms_AUDIT_ONLY=S['terminal_ms_AUDIT_ONLY'], role=np.asarray('AUDIT_ONLY_OUTCOME_SIDE'))
    label_cols = list(labels.columns)

    def role_of(col):
        if col in ('row_index', 'match', 's_ms', 'h_s'):
            return 'key'
        if col == 'patch':
            return 'split_metadata_not_feature'
        if col in ('Y_A',):
            return 'target_primary_candidate (h=90 working primary; h=60/120 sensitivity)'
        if col in ('delta_A',):
            return 'target_value_candidate'
        if col in ('Y_B', 'delta_B', 'label_disagreement_A_vs_B') or col.startswith('p_post_B') or col.startswith('p_pre_B'):
            return 'diagnostic_model_sensitivity_not_target'
        if col in ('p_pre_A',):
            return 'input_duplicate_of_features_file (pre-only)'
        if col in ('q_pre_ms', 'pre_snapshot_ms', 'pre_unassigned_soul_events', 'pre_valid_history_positions_B'):
            return 'pre_side_audit_only'
        if col == 'pre_snapshot_age_s':
            return 'audit_only_not_predictor'
        return 'outcome_metadata_or_audit_only_forbidden_in_q_features'
    schema = {'role': ROLE,
              'files': {
                  'features_pre_only.npz': {'X_input': {'columns': [{'name': n, 'role': ('input_categorical' if n in categorical else 'input')} for n in input_names]},
                                            'row_index/match/s_ms': 'key', 'patch_split_metadata': 'split_metadata_not_feature',
                                            'pre_query_ms': 'key_metadata_not_feature'},
                  'labels_long.csv': {c: role_of(c) for c in label_cols},
                  'labels_long.npz': 'same columns as labels_long.csv (exact float64)',
                  'audit_only/states_pre_post_AUDIT_ONLY_NOT_Q_FEATURES.npz': 'audit only; post states and final winner are outcome-side and forbidden in q',
                  'audit_only/pre_snapshot_age_s_AUDIT_ONLY.npz': 'audit only; snapshot age is not a predictor'},
              'rules': ['q inputs are PRE ONLY: StateV2 at s-1 (minus snapshot_age_s) and p_pre_A',
                        'never in q features: L, endpoints, end reasons, candidate times, durations, future membership/participants, game end, '
                        'final winner, post states, deltas, labels, after/during event counts, model B values',
                        'champion IDs are categorical', 'no positions added', 'patch is a split variable, not a feature'],
              'state_version': model_keys_state_version_from_ident(ident), 'feature_guards': guards}
    write_json(run.out / 'feature_schema.json', schema)
    return guards


def model_keys_state_version_from_ident(ident):
    return 'objective_history_v2_participant_order'


def match_weighted(values, groups):
    import pandas as pd
    s = pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(groups)).mean()
    return float(s.mean())


def rate_pair(values, groups, mask=None):
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    if mask is not None:
        v, g = v[mask], g[mask]
    if len(v) == 0:
        return {'rows': 0, 'matches': 0, 'row_weighted': None, 'match_weighted': None}
    return {'rows': int(len(v)), 'matches': int(len(np.unique(g))), 'row_weighted': float(v.mean()), 'match_weighted': match_weighted(v, g)}


def quantiles(x):
    q = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    return dict(zip([str(v) for v in q], np.quantile(np.asarray(x, dtype=float), q).tolist()))


class MatchBootstrap:
    """Per-match aggregated numerators/denominators; rates recomputed for resampled matches (models fixed)."""

    def __init__(self, groups, n_boot, seed):
        self.u, self.inv = np.unique(np.asarray(groups), return_inverse=True)
        rng = np.random.default_rng(seed)
        self.counts = np.stack([np.bincount(rng.integers(0, len(self.u), len(self.u)), minlength=len(self.u)) for _ in range(n_boot)]).astype(np.float64)

    def ci(self, values, mask=None):
        v = np.asarray(values, dtype=float)
        m = np.ones(len(v), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        num = np.bincount(self.inv[m], weights=v[m], minlength=len(self.u))
        den = np.bincount(self.inv[m], minlength=len(self.u)).astype(float)
        has = den > 0
        per = np.where(has, num / np.where(has, den, 1), 0.)
        rw = (self.counts @ num) / np.maximum(self.counts @ den, 1e-300)
        mw = (self.counts @ per) / np.maximum(self.counts @ has.astype(float), 1e-300)
        return {'row_weighted_ci95': np.quantile(rw, [.025, .975]).tolist(), 'match_weighted_ci95': np.quantile(mw, [.025, .975]).tolist(),
                'replicates': int(len(self.counts))}


def cell_bootstrap(values, groups, n_boot, seed):
    v = np.asarray(values, dtype=float)
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    num = np.bincount(inv, weights=v, minlength=len(u))
    den = np.bincount(inv, minlength=len(u)).astype(float)
    per = num / den
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    rw = num[idx].sum(1) / den[idx].sum(1)
    mw = per[idx].mean(1)
    return {'row_weighted_ci95': np.quantile(rw, [.025, .975]).tolist(), 'match_weighted_ci95': np.quantile(mw, [.025, .975]).tolist()}


def analyses(run, labels, smoke):
    import pandas as pd
    res = {'role': ROLE, 'no_selection': 'no horizon or model is chosen from these results; B90/A remains the working primary'}
    W = {h: labels[labels.h_s == h].sort_values('row_index', kind='mergesort').reset_index(drop=True) for h in HS}
    g = W[90].match.to_numpy()
    n_boot = 200 if smoke else N_BOOT_MAIN
    n_cell = 100 if smoke else N_BOOT_CELL
    boot = MatchBootstrap(g, n_boot, BOOT_SEED)
    # 1. per-window distributions
    per = {}
    for h in HS:
        d = W[h]
        cell = {'rows': len(d), 'matches': int(d.match.nunique())}
        for mdl in ('A', 'B'):
            delta = d[f'delta_{mdl}'].to_numpy()
            cell[mdl] = {'positive': {'count': int((delta > 0).sum()), **rate_pair(delta > 0, g)},
                         'exact_zero': {'count': int((delta == 0).sum()), **rate_pair(delta == 0, g)},
                         'negative': {'count': int((delta < 0).sum()), **rate_pair(delta < 0, g)},
                         'near_zero_descriptions': {str(b): {'count': int((np.abs(delta) <= b).sum()), **rate_pair(np.abs(delta) <= b, g)} for b in R.NEAR_ZERO_BANDS},
                         'delta_quantiles': quantiles(delta), 'delta_mean': {'row_weighted': float(delta.mean()), 'match_weighted': match_weighted(delta, g)},
                         'abs_delta_mean': {'row_weighted': float(np.abs(delta).mean()), 'match_weighted': match_weighted(np.abs(delta), g)},
                         'positive_rate_bootstrap': boot.ci(delta > 0)}
        dur = d.duration_after_last_kill_s.to_numpy()
        cell['effective_followup_after_last_kill_s'] = {'quantiles': quantiles(dur), 'mean': {'row_weighted': float(dur.mean()), 'match_weighted': match_weighted(dur, g)},
                                                        'reached_horizon': rate_pair(d.reason_horizon.to_numpy() == 1, g),
                                                        'full_horizon_duration_equals_h': rate_pair(dur == h, g)}
        cell['terminal_reasons_inclusive'] = {r_: {'count': int(d[f'reason_{r_}'].sum()), **rate_pair(d[f'reason_{r_}'] == 1, g)} for r_ in R.REASONS}
        cell['terminal_reason_tie_sets'] = {k: int(v) for k, v in d.end_reasons.value_counts().sort_index().items()}
        cell['rows_with_tied_reasons'] = int((d.n_tied_reasons > 1).sum())
        cell['availability'] = {'valid_rows': int(d.valid.sum()), 'invalid_rows': int((d.valid == 0).sum()), **rate_pair(d.valid == 1, g),
                                'post_valid_history_positions_B': {str(k): int(v) for k, v in d.post_valid_history_positions_B.value_counts().sort_index().items()}}
        age = d.post_snapshot_age_s.to_numpy()
        cell['staleness'] = {'post_snapshot_age_s_quantiles': quantiles(age),
                             'post_snapshot_not_after_L': rate_pair(d.post_snapshot_after_L == 0, g),
                             'same_snapshot_frame_pre_and_post': rate_pair(d.same_snapshot_frame_pre_post == 1, g),
                             'post_snapshot_age_gt_30s': rate_pair(age > 30, g), 'post_snapshot_age_gt_45s': rate_pair(age > 45, g)}
        per[str(h)] = cell
    pre_age = W[90].pre_snapshot_age_s.to_numpy()
    res['pre_snapshot_age_s'] = {'quantiles': quantiles(pre_age), 'age_gt_30s': rate_pair(pre_age > 30, g), 'age_gt_45s': rate_pair(pre_age > 45, g),
                                 'pre_valid_history_positions_B': {str(k): int(v) for k, v in W[90].pre_valid_history_positions_B.value_counts().sort_index().items()}}
    res['per_window'] = per
    # 2. paired flips and model disagreement
    flips = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        same = W[a].endpoint_ms.to_numpy() == W[b].endpoint_ms.to_numpy()
        cell = {'same_endpoint': rate_pair(same, g)}
        for mdl in ('A', 'B'):
            f = W[a][f'Y_{mdl}'].to_numpy() != W[b][f'Y_{mdl}'].to_numpy()
            cell[mdl] = {'flip': {'count': int(f.sum()), **rate_pair(f, g)}, 'flip_bootstrap': boot.ci(f),
                         'flip_among_different_endpoints': rate_pair(f, g, ~same), 'flips_among_same_endpoints': int(f[same].sum())}
        flips[f'{a}_vs_{b}'] = cell
    res['paired_window_flips'] = flips
    dis = {}
    for h in HS:
        d = W[h]
        f = (d.Y_A != d.Y_B).to_numpy()
        sign3 = np.sign(d.delta_A.to_numpy()) != np.sign(d.delta_B.to_numpy())
        dis[str(h)] = {'label_disagreement': {'count': int(f.sum()), **rate_pair(f, g)}, 'bootstrap': boot.ci(f),
                       'three_valued_sign_disagreement': rate_pair(sign3, g),
                       'pearson_delta_A_B': float(np.corrcoef(d.delta_A, d.delta_B)[0, 1])}
    res['model_A_vs_B_disagreement'] = dis
    res['bootstrap_note'] = f'{n_boot} match-level resamples (seed {BOOT_SEED}); frozen models; no training, calibration or selection uncertainty included.'
    # V1 comparison on exact keys/windows only
    bb = pd.read_csv(BB_VALUES)
    wv = pd.read_csv(WV_LABELS)
    pre_v1 = wv[wv.h == -1][['match', 's', 'expanded']].rename(columns={'expanded': 'v1_p_pre', 's': 's_ms'})
    v1 = {}
    for h in HS:
        b = bb[bb.h == h][['match', 's', 'endpoint', 'p_post', 'valid']].rename(columns={'s': 's_ms', 'endpoint': 'v1_endpoint', 'p_post': 'v1_p_post', 'valid': 'v1_valid'})
        m = W[h][['row_index', 'match', 's_ms', 'endpoint_ms', 'delta_A', 'Y_A']].merge(b, on=['match', 's_ms'], how='left', validate='one_to_one').merge(pre_v1, on=['match', 's_ms'], how='left', validate='one_to_one')
        exact = (m.v1_endpoint == m.endpoint_ms) & (m.v1_valid == 1) & m.v1_p_pre.notna() & m.v1_p_post.notna()
        mm = m[exact]
        yv1 = (mm.v1_p_post - mm.v1_p_pre > 0).to_numpy()
        f = yv1 != (mm.Y_A.to_numpy() == 1)
        v1[str(h)] = {'rows_exact_key_and_endpoint': int(exact.sum()), 'rows_not_comparable': int((~exact).sum()),
                      'v1_positive_rate': rate_pair(yv1, mm.match.to_numpy()), 'v2A_positive_rate_same_rows': rate_pair(mm.Y_A == 1, mm.match.to_numpy()),
                      'label_disagreement_v1_vs_v2A': rate_pair(f, mm.match.to_numpy()),
                      'pearson_delta_v1_v2A': float(np.corrcoef(mm.v1_p_post - mm.v1_p_pre, mm.delta_A)[0, 1]) if len(mm) > 2 else None,
                      'note': 'V1 = frozen v1 expanded on objective_history_v1 states (B rule endpoints); model-version disagreement, not an error rate'}
        if len(mm) == len(W[h]):
            v1[str(h)]['bootstrap'] = boot.ci(f)
    res['v1_comparison'] = v1
    # 3. breakdowns
    s_min = W[90].s_ms.to_numpy() / 60000.
    cells = {}

    def cell_stats(mask, h_for_labels=90, with_flips=True):
        d = W[h_for_labels]
        gm = g[mask]
        n_m = int(len(np.unique(gm))) if mask.any() else 0
        out = {'rows': int(mask.sum()), 'matches': n_m, 'sparse': n_m < MIN_CELL_MATCHES}
        if not mask.any():
            return out
        ya = d.Y_A.to_numpy()[mask]
        out['Y_A_positive'] = rate_pair(ya, gm)
        out['Y_B_positive'] = rate_pair(d.Y_B.to_numpy()[mask], gm)
        out['A_vs_B_disagreement'] = rate_pair((d.Y_A != d.Y_B).to_numpy()[mask], gm)
        out['mean_delta_A'] = {'row_weighted': float(d.delta_A.to_numpy()[mask].mean()), 'match_weighted': match_weighted(d.delta_A.to_numpy()[mask], gm)}
        if n_m >= 2:
            out['Y_A_positive_bootstrap'] = cell_bootstrap(ya, gm, n_cell, BOOT_SEED)
            out['A_vs_B_disagreement_bootstrap'] = cell_bootstrap((d.Y_A != d.Y_B).to_numpy()[mask], gm, n_cell, BOOT_SEED)
        if with_flips:
            for a, b in ((60, 90), (90, 120), (60, 120)):
                f = (W[a].Y_A != W[b].Y_A).to_numpy()[mask]
                out[f'flip_A_{a}_vs_{b}'] = rate_pair(f, gm)
        if n_m < MIN_CELL_MATCHES:
            out['sparse_note'] = f'fewer than {MIN_CELL_MATCHES} matches: rates and intervals are unreliable'
        return out
    for p_ in sorted(W[90].patch.astype(str).unique()):
        cells[f'patch={p_}'] = cell_stats(W[90].patch.astype(str).to_numpy() == p_)
    for lo, hi in TIME_BANDS:
        lab = f'{lo}-{hi}' if hi < 1000 else f'{lo}+'
        cells[f'start_minutes={lab}'] = cell_stats((s_min >= lo) & (s_min < hi))
    res['breakdown_patch_and_time'] = cells
    obj = {}
    cats = list(R.OBJECTIVE_CATEGORIES) + list(R.STRUCTURE_CATEGORIES) + ['monster_other']
    for h in HS:
        d = W[h]
        oh = {}
        for c in cats:
            present = d[f'after_last_kill_{c}'].to_numpy() > 0
            oh[c] = {'present_after_last_kill': cell_stats(present, h, with_flips=False), 'absent_after_last_kill': {'rows': int((~present).sum())},
                     'events_after_last_kill': int(d[f'after_last_kill_{c}'].sum())}
            if c in R.OWNED_CATEGORIES:
                oh[c]['present_blue'] = int((d[f'after_last_kill_{c}_blue'] > 0).sum())
                oh[c]['present_red'] = int((d[f'after_last_kill_{c}_red'] > 0).sum())
        obj[str(h)] = oh
    res['breakdown_objectives_after_last_kill'] = obj
    during = {}
    d = W[90]
    for c in cats:
        present = d[f'during_{c}'].to_numpy() > 0
        during[c] = {'present_during_engagement': cell_stats(present, 90, with_flips=False), 'events_during': int(d[f'during_{c}'].sum())}
    res['breakdown_objectives_during_engagement_q_pre_to_L'] = during
    extra = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        gain = {}
        for c in cats:
            more = W[b][f'after_last_kill_{c}'].to_numpy() > W[a][f'after_last_kill_{c}'].to_numpy()
            gain[c] = {'rows_with_additional_events_in_longer_window': int(more.sum())}
        extra[f'{a}_to_{b}'] = gain
    res['objective_events_added_by_longer_window'] = extra
    res['objective_note'] = ('Objective occurrences are raw co-occurring events in (L, endpoint]; they are not attributed to the engagement and a larger '
                             'window does not imply causal acquisition by the engagement. soul_teamid0_unassigned is diagnostic (not owned).')
    d = W[90]
    simul = {'first_kill_simultaneous_raw_kills (anchor ambiguity)': int((d.first_kill_raw_count > 1).sum()),
             'first_kill_missing_in_raw': int((d.first_kill_raw_count == 0).sum()),
             'last_kill_simultaneous_raw_kills': int((d.kills_at_L_count > 1).sum()),
             'last_kill_missing_in_raw': int((d.kills_at_L_count == 0).sum()),
             'next_kill_simultaneous_raw_kills': int((d.next_kill_raw_count > 1).sum()),
             'rows_with_nonkill_event_at_L_timestamp (included in during window)': int((d.nonkill_events_at_L > 0).sum()),
             'rows_with_nonkill_event_at_next_kill_timestamp (excluded when next_kill is terminal)': int((d.nonkill_events_at_next_kill_ts > 0).sum()),
             'rows_with_nonkill_event_at_game_end_timestamp': int((d.nonkill_events_at_game_end_ts > 0).sum()),
             'rows_with_nonkill_event_at_next_start_timestamp': int((d.nonkill_events_at_next_start_ts > 0).sum()),
             'game_end_ambiguous_rows': int(d.game_end_ambiguous.sum())}
    for h in HS:
        simul[f'rows_with_same_timestamp_nonkill_groups_after_last_kill_h{h}'] = int((W[h].after_last_kill_nonkill_same_timestamp_groups > 0).sum())
        simul[f'rows_with_tied_terminal_reasons_h{h}'] = int((W[h].n_tied_reasons > 1).sum())
        simul[f'rows_next_kill_terminal_with_nonkill_event_at_that_timestamp_h{h}'] = int(((W[h].reason_next_kill == 1) & (W[h].nonkill_events_at_next_kill_ts > 0)).sum())
    res['simultaneous_event_occurrences'] = simul
    write_json(run.out / 'analysis_results.json', res)
    return res


def label_audits(run, labels, S):
    W = {h: labels[labels.h_s == h].sort_values('row_index', kind='mergesort').reset_index(drop=True) for h in HS}
    c = {}
    for mdl in ('A', 'B'):
        pre = [W[h][f'p_pre_{mdl}'].to_numpy() for h in HS]
        c[f'p_pre_{mdl}_bitwise_identical_all_h'] = bool(all(np.array_equal(pre[0], p) for p in pre[1:]))
        c[f'p_pre_{mdl}_batch_invariant_rows_not_equal'] = int((S[f'p_pre_{mdl}'] != S[f'p_pre_{mdl}_batch2']).sum())
        c[f'p_pre_{mdl}_batch_invariant_max_abs_diff'] = float(np.abs(S[f'p_pre_{mdl}'] - S[f'p_pre_{mdl}_batch2']).max())
        for h in HS:
            d = W[h]
            c[f'delta_{mdl}_h{h}_equals_post_minus_pre_bitwise'] = bool(np.array_equal(d[f'delta_{mdl}'].to_numpy(), d[f'p_post_{mdl}'].to_numpy() - d[f'p_pre_{mdl}'].to_numpy()))
            c[f'Y_{mdl}_h{h}_equals_delta_gt_0'] = bool(np.array_equal(d[f'Y_{mdl}'].to_numpy(), (d[f'delta_{mdl}'].to_numpy() > 0).astype(int)))
            c[f'p_{mdl}_h{h}_finite_unit_interval'] = bool(np.isfinite(d[f'p_post_{mdl}']).all() and ((d[f'p_post_{mdl}'] >= 0) & (d[f'p_post_{mdl}'] <= 1)).all())
        for a, b in ((60, 90), (90, 120), (60, 120)):
            same = W[a].endpoint_ms.to_numpy() == W[b].endpoint_ms.to_numpy()
            c[f'duplicate_endpoint_same_post_{mdl}_{a}_{b}_violations'] = int((W[a][f'p_post_{mdl}'].to_numpy()[same] != W[b][f'p_post_{mdl}'].to_numpy()[same]).sum())
            c[f'duplicate_endpoint_rows_{a}_{b}'] = int(same.sum())
        # Batch-composition rounding diagnostic (informational): independently batched post-60/120 predictions.
        diag = {}
        for h in (60, 120):
            p_can, p_b2 = S[f'p_h{h}_{mdl}'], S[f'p_h{h}_{mdl}_batch2']
            y_b2 = ((p_b2 - S[f'p_pre_{mdl}']) > 0).astype(int)
            diag[f'h{h}'] = {'rows_not_bitwise_equal': int((p_can != p_b2).sum()), 'max_abs_diff': float(np.abs(p_can - p_b2).max()),
                             'label_changes_under_independent_batch': int((y_b2 != W[h][f'Y_{mdl}'].to_numpy()).sum())}
        c[f'batch_composition_diagnostic_{mdl}'] = diag
    for a, b in ((60, 90), (90, 120), (60, 120)):
        same = S[f'h{a}_query_ms'] == S[f'h{b}_query_ms']
        c[f'duplicate_endpoint_state_bitwise_{a}_{b}_violations'] = int((S[f'h{a}_X_last'][same] != S[f'h{b}_X_last'][same]).any(axis=1).sum())
        c[f'duplicate_endpoint_history_snapshots_{a}_{b}_violations'] = int((S[f'h{a}_hist_snapshot_ms'][same] != S[f'h{b}_hist_snapshot_ms'][same]).any(axis=1).sum())
    # Cross-row coincidence (informational): an endpoint equal to a later engagement's q_pre in the same match.
    pre_key = {(m, int(q)): i for i, (m, q) in enumerate(zip(S['match'].tolist(), S['pre_query_ms'].tolist()))}
    cross = {}
    for h in HS:
        hits = [(i, pre_key[(m, int(q))]) for i, (m, q) in enumerate(zip(S['match'].tolist(), S[f'h{h}_query_ms'].tolist())) if (m, int(q)) in pre_key]
        if hits:
            ii, jj = np.asarray(hits).T
            cross[f'h{h}'] = {'rows': len(hits), 'state_bitwise_violations': int((S[f'h{h}_X_last'][ii] != S['pre_X_last'][jj]).any(axis=1).sum()),
                              'A_max_abs_diff': float(np.abs(S[f'p_h{h}_A'][ii] - S['p_pre_A'][jj]).max()),
                              'B_max_abs_diff': float(np.abs(S[f'p_h{h}_B'][ii] - S['p_pre_B'][jj]).max())}
        else:
            cross[f'h{h}'] = {'rows': 0}
    c['cross_row_endpoint_equals_later_q_pre'] = cross
    for h in HS:
        hs = S[f'h{h}_hist_snapshot_ms']
        mask = S[f'h{h}_hist_mask'].astype(bool)
        positions = S[f'h{h}_query_ms'][:, None] - (8 - np.arange(9))[None, :] * 60_000
        c[f'history_snapshot_after_position_h{h}'] = int(((hs > positions) & mask).sum())
        c[f'query_equals_endpoint_h{h}'] = bool(np.array_equal(S[f'h{h}_query_ms'], W[h].endpoint_ms.to_numpy()))
        c[f'post_snapshot_le_query_h{h}'] = bool((S[f'h{h}_snapshot_ms'] <= S[f'h{h}_query_ms']).all())
    hs = S['pre_hist_snapshot_ms']
    positions = S['pre_query_ms'][:, None] - (8 - np.arange(9))[None, :] * 60_000
    c['history_snapshot_after_position_pre'] = int(((hs > positions) & S['pre_hist_mask'].astype(bool)).sum())
    c['pre_snapshot_lt_s'] = bool((S['pre_snapshot_ms'] < W[90].s_ms.to_numpy()).all())
    c['pre_query_equals_s_minus_1'] = bool(np.array_equal(S['pre_query_ms'], W[90].s_ms.to_numpy() - 1))
    c['mixed_A_B_delta_guard_rejected_all_rows'] = bool((S['guard_mixed_A_B_delta_rejected'] == 1).all())
    c['monotone_endpoints'] = bool(((W[60].endpoint_ms.to_numpy() <= W[90].endpoint_ms.to_numpy()) & (W[90].endpoint_ms.to_numpy() <= W[120].endpoint_ms.to_numpy())).all())
    bad = [k for k, v in c.items() if (isinstance(v, bool) and not v) or (k.endswith(('violations', '_after_position_h60', '_after_position_h90', '_after_position_h120', '_after_position_pre')) and v)]
    bad += [f'cross_row_state_{k}' for k, v in cross.items() if v.get('state_bitwise_violations')]
    c['failed'] = bad
    c['pass'] = not bad
    write_json(run.out / 'label_audits.json', c)
    return c


def raw_audit(run, labels, workers):
    d = labels[labels.h_s == 90].sort_values('row_index', kind='mergesort')
    cols = ['row_index', 'match', 's_ms', 'L_ms', 'next_start_effective_ms', 'pre_snapshot_ms']
    base = d[cols].reset_index(drop=True)
    for h in HS:
        w = labels[labels.h_s == h].sort_values('row_index', kind='mergesort').reset_index(drop=True)
        base[f'endpoint_h{h}_ms'] = w.endpoint_ms.to_numpy()
        base[f'h{h}_snapshot_ms'] = w.post_snapshot_ms.to_numpy()
    by_match = defaultdict(list)
    for r in base.to_dict('records'):
        by_match[r['match']].append(r)
    tasks = [{'items': ch} for ch in chunked([(m, by_match[m]) for m in sorted(by_match)], 250)]
    res, fail = run_pool(run, raw_audit_chunk, tasks, workers, 'independent_raw_audit', 'raw-event audit')
    fails, checked, errors = defaultdict(list), Counter(), dict(fail)
    for r in res:
        errors.update(r['errors'])
        checked.update(r['checked'])
        for k, v in r['fails'].items():
            fails[k].extend(v)
    out = {'checked': dict(checked), 'failure_counts': {k: len(fails.get(k, [])) for k in checked},
           'failure_examples': {k: v[:50] for k, v in fails.items()}, 'errors': errors}
    hard = [k for k in checked if fails.get(k) and k != 'game_end_observed_unique']
    out['hard_failures'] = hard
    out['game_end_not_unique_rows'] = len(fails.get('game_end_observed_unique', []))
    out['pass'] = bool(not hard and not errors and checked)
    write_json(run.out / 'independent_raw_event_audit.json', out)
    return out


def partition_checks(run, cohort):
    matches = set(cohort.match)
    vd = read_json(VD / 'protocol.json')['splits']
    p1 = read_json(P1 / 'protocol.json')['splits']
    p2 = read_json(P2 / 'protocol.json')['splits']
    ev = read_json(OLD_EVAL / 'match_splits.json')
    out = {}
    for name, sp in (('v1_buckets', vd), ('p1', p1), ('p2', p2)):
        out[name] = {k: len(matches & set(sp[k])) for k in ('fit', 'calibrate', 'select', 'test')}
        out[name + '_cohort_subset_of_engagement'] = matches <= set(sp['engagement'])
    out['v1_eval'] = {k: len(matches & set(ev[k])) for k in ('value_train', 'value_validation', 'predict_train')}
    out['v1_eval_cohort_subset_of_predict_test'] = matches <= set(ev['predict_test'])
    out['pass'] = bool(all(v == 0 for name in ('v1_buckets', 'p1', 'p2', 'v1_eval') for v in out[name].values())
                       and out['v1_buckets_cohort_subset_of_engagement'] and out['p1_cohort_subset_of_engagement']
                       and out['p2_cohort_subset_of_engagement'] and out['v1_eval_cohort_subset_of_predict_test'])
    return out


def determinism_and_perturbation(run, bnd, S, names, ident, workers, smoke):
    mids = sorted(bnd.match.unique(), key=lambda m: rank('p3_determinism', m))[:(4 if smoke else DETERMINISM_MATCHES)]
    sub = bnd[bnd.match.isin(set(mids))]
    D, _ = run_scoring(run, sub, names, ident, min(workers, 2), subdir='determinism_rerun', match_filter=set(mids), chunk_size=6, stage='determinism')
    idx = np.searchsorted(S['row_index'], D['row_index'])
    det = {'matches': mids, 'rows': int(len(D['row_index']))}
    unequal = [k for k in D if not np.array_equal(D[k], S[k][idx])]
    det['arrays_compared'] = len(D)
    det['arrays_not_bitwise_equal'] = unequal
    # Boundary rerun in a fresh process.
    import pandas as pd
    cohort_like = sub.rename(columns={'s_ms': 's', 'L_ms': 'L', 'next_start_stored_ms': 'next_start', 'game_end_stored_exposure_ms': 'end',
                                      'same_match_overlap_stored': 'same_match_overlap', 'end_observed_stored': 'end_observed'})
    ex = pd.read_csv(EXPOSURES, usecols=['match', 's', 'L'])
    items = []
    for m in mids:
        rows = cohort_like[cohort_like.match == m][['row_index', 's', 'L', 'next_start', 'end', 'same_match_overlap', 'end_observed']].to_dict('records')
        exps = [{'s': int(a), 'L': int(b)} for a, b in ex[ex.match == m][['s', 'L']].to_numpy()]
        items.append((m, rows, exps))
    res, fail = run_pool(run, boundary_chunk, [{'chunk_id': 0, 'items': items}], 1, 'determinism', 'boundary rerun')
    rer = pd.DataFrame([r for x in res for r in x['rows']]).sort_values('row_index').reset_index(drop=True)
    orig = sub.sort_values('row_index').reset_index(drop=True).drop(columns=['patch'])
    det['boundary_rerun_equal'] = bool(not fail and list(rer.columns) == list(orig.columns) and rer.equals(orig))
    det['pass'] = bool(not unequal and det['boundary_rerun_equal'])
    pm = mids[:(2 if smoke else PERTURBATION_MATCHES)]
    after = {m: int(bnd[bnd.match == m][[f'endpoint_h{h}_ms' for h in HS]].to_numpy().max()) for m in pm}
    P, _ = run_scoring(run, bnd[bnd.match.isin(set(pm))], names, ident, min(workers, 2), subdir='future_perturbation', match_filter=set(pm),
                       perturb=after, chunk_size=4, stage='future_perturbation')
    idx = np.searchsorted(S['row_index'], P['row_index'])
    skip = {'winner_blue_AUDIT_ONLY', 'terminal_ms_AUDIT_ONLY', 'last_frame_ms'}
    changed = [k for k in P if k not in skip and not np.array_equal(P[k], S[k][idx])]
    pert = {'matches': pm, 'rows': int(len(P['row_index'])), 'perturbed_after_ms': after, 'arrays_changed': changed,
            'excluded_from_comparison': sorted(skip), 'pass': not changed,
            'what_is_perturbed': 'frame node values (except champion IDs) strictly after the match max endpoint are replaced; predictor events after it '
                                 'are removed and a fake kill, baron and inhibitor are inserted just after it'}
    return det, pert


def select_examples(run, labels, workers):
    d = labels[labels.h_s == 90].sort_values('row_index', kind='mergesort').reset_index(drop=True)
    W = {h: labels[labels.h_s == h].sort_values('row_index', kind='mergesort').reset_index(drop=True) for h in HS}
    any_tie = (W[60].n_tied_reasons > 1) | (W[90].n_tied_reasons > 1) | (W[120].n_tied_reasons > 1)
    types = {
        'horizon_wins_h90': d.end_reasons == 'horizon',
        'next_kill_truncates_h90': d.end_reasons == 'next_kill',
        'next_engagement_truncates_h90': d.end_reasons == 'next_engagement_start',
        'game_end_truncates_h90': d.end_reasons == 'game_end',
        'tied_terminal_reasons_any_h': any_tie,
        'simultaneous_raw_kills_at_first_kill': d.first_kill_raw_count > 1,
        'nonkill_event_at_next_kill_timestamp_next_kill_terminal_h90': (d.reason_next_kill == 1) & (d.nonkill_events_at_next_kill_ts > 0),
        'objective_after_last_kill_included_h90': sum(d[f'after_last_kill_{c}'] for c in ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan')) > 0,
        'three_distinct_endpoints_60_90_120': (W[60].endpoint_ms < W[90].endpoint_ms) & (W[90].endpoint_ms < W[120].endpoint_ms),
    }
    chosen, missing = [], []
    for t, mask in types.items():
        pos_all = np.flatnonzero(mask.to_numpy())
        if not len(pos_all):
            missing.append(t)
            continue
        pos = min(pos_all.tolist(), key=lambda p_: rank(f'p3_example:{t}', f'{d.match.iat[p_]}:{int(d.s_ms.iat[p_])}'))
        best = d.iloc[pos]
        i = int(best['row_index'])
        ex = {'example_id': t, 'actual_case': True, 'match': str(best['match']), 's_ms': int(best['s_ms']), 'row_index': i, 'patch': str(best['patch']),
              'selection_rule': f'min sha256("p3_example:{t}:<match>:<s>") among {int(mask.sum())} qualifying rows',
              'qualifying_rows': int(mask.sum()), **{k: int(best[k]) for k in ('q_pre_ms', 'first_kill_ms', 'L_ms', 'first_kill_raw_count', 'kills_at_L_count',
                                                                            'next_kill_ms', 'next_start_stored_ms', 'next_start_effective_ms', 'game_end_ms',
                                                                            'pre_snapshot_ms')},
              'p_pre_A': float(best['p_pre_A']), 'p_pre_B': float(best['p_pre_B']), 'windows': {}}
        for h in HS:
            w = W[h].iloc[pos]
            if int(w['row_index']) != i:
                raise ValueError('example window rows misaligned')
            ex['windows'][str(h)] = {'endpoint_ms': int(w['endpoint_ms']), 'end_reasons': str(w['end_reasons']),
                                     'duration_after_last_kill_s': float(w['duration_after_last_kill_s']),
                                     'candidates_ms': {r_: int(w[f'candidate_{r_}_ms']) for r_ in R.REASONS}, 'post_snapshot_ms': int(w['post_snapshot_ms']),
                                     'p_post_A': float(w['p_post_A']), 'delta_A': float(w['delta_A']), 'Y_A': int(w['Y_A']),
                                     'p_post_B': float(w['p_post_B']), 'delta_B': float(w['delta_B']), 'Y_B': int(w['Y_B']),
                                     'after_last_kill_counts': {c: int(w[f'after_last_kill_{c}']) for c in R.COUNT_KEYS if int(w[f'after_last_kill_{c}'])}}
        hi = max(ex['windows']['120']['endpoint_ms'], ex['next_kill_ms'] if ex['next_kill_ms'] > 0 else 0)
        ex['lo'], ex['hi'] = ex['q_pre_ms'] - 1000, min(hi + 5000, ex['game_end_ms'] + 1000)
        chosen.append(ex)
    res, fail = run_pool(run, events_window_task, [{'examples': chosen}], 1, 'examples', 'raw event windows')
    ev = res[0] if res else {}
    for ex in chosen:
        ex['raw_events_in_window'] = ev.get(ex['example_id'], [])
    out = {'role': 'ACTUAL cohort rows (no invented cases); raw events shown are read-only cache events', 'examples': chosen,
           'types_without_actual_case': missing, 'errors': fail}
    write_json(run.out / 'timeline_examples.json', out)
    return out


def execute(run, a):
    import pandas as pd
    smoke = a.smoke
    if (run.out / 'results.json').exists():
        raise ValueError('completed results already exist; use a new output directory')
    run.stage('preflight')
    ident = load_frozen_identity()
    pre_fail = []
    if ident['a_sha256'] != EXPECTED_A_SHA256 or ident['frozen_models']['chosen_candidate'] != 'A':
        pre_fail.append('frozen A identity differs from the P3 specification')
    if ident['p2_validation_status'] != 'pass' or ident['p2_results_status'] != 'complete':
        pre_fail.append('P2 not complete/pass')
    if sha256_file(Path(ident['a_path'])) != EXPECTED_A_SHA256:
        pre_fail.append('A model file hash mismatch')
    if pre_fail:
        raise RuntimeError(f'preflight failed: {pre_fail}')
    v2_matches = sorted(set(pd.read_csv(V2FIX / 'rows.csv', usecols=['match']).match))
    if smoke:
        v2_matches = sorted(v2_matches, key=lambda mm: rank('p3_smoke', mm))[:smoke]
    run.log(f'hashing frozen inputs and {len(v2_matches)} matches of raw cache files (before)')
    hb = frozen_hashes(run, v2_matches)
    write_json(run.out / 'hashes_before.json', hb)
    snap = run.out / 'source_snapshot'
    for p in NEW_SOURCES:
        if (ROOT / p).exists():
            (snap / p).parent.mkdir(parents=True, exist_ok=True)
            (snap / p).write_bytes((ROOT / p).read_bytes())
    new_hashes = {p: sha256_file(ROOT / p) for p in NEW_SOURCES if (ROOT / p).exists()}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', P3_WORKER='0')
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'--basetemp={run.out / "pytest_basetemp"}',
                           'tests/test_engagement_labels_v3_rules.py', 'tests/test_temporal_winprob_v3_history.py', 'tests/test_state_value_v2_contract.py'],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=1800, env=env)
    (run.out / 'logs' / 'pytest_output.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
    tests = {'command': 'python -m pytest -q -p no:cacheprovider tests/test_engagement_labels_v3_rules.py tests/test_temporal_winprob_v3_history.py '
                        'tests/test_state_value_v2_contract.py', 'returncode': proc.returncode, 'summary': lines[-1] if lines else ''}
    run.log(f'tests: {tests}')
    if proc.returncode != 0:
        raise RuntimeError('contract tests failed; see logs/pytest_output.log')

    run.stage('protocol')
    protocol = write_protocol(run, ident, smoke)
    run.log(f'protocol.json written before execution sha256={protocol["protocol_sha256"]}')

    run.stage('cohort_lineage')
    cohort, ex, lineage, lfail = cohort_lineage(run, smoke)
    run.log(f'cohort rows={len(cohort)} matches={cohort.match.nunique()} lineage_failures={lfail}')

    run.stage('boundaries')
    bnd = run_boundaries(run, cohort, ex, WORKERS)
    b_checks = {'rows': len(bnd), 'first_kill_missing': int((bnd.first_kill_raw_count == 0).sum()),
                'first_kill_simultaneous': int((bnd.first_kill_raw_count > 1).sum()),
                'last_kill_missing': int((bnd.kills_at_L_count == 0).sum()), 'next_start_recompute_mismatch': int((bnd.next_start_matches_recomputed == 0).sum()),
                'game_end_exposure_mismatch': int((bnd.game_end_matches_exposure == 0).sum()), 'game_end_ambiguous': int(bnd.game_end_ambiguous.sum()),
                'end_observed_zero': int((bnd.end_observed_stored != 1).sum()), 'sentinel_conflicts': int(bnd.sentinel_conflict_later_exposure_at_or_after_end.sum()),
                'invalid_rows': {str(h): int((bnd[f'valid_h{h}'] == 0).sum()) for h in HS},
                'non_monotone_rows': int(((bnd.endpoint_h60_ms > bnd.endpoint_h90_ms) | (bnd.endpoint_h90_ms > bnd.endpoint_h120_ms)).sum()),
                'kills_in_s_to_first_kill_rows': int((bnd.kills_in_s_to_first_kill_exclusive > 0).sum())}
    b_checks['invalid_keys'] = {str(h): [[m_, int(s_)] for m_, s_ in bnd.loc[bnd[f'valid_h{h}'] == 0, ['match', 's_ms']].to_numpy()[:200]] for h in HS}
    write_json(run.out / 'boundary_checks.json', b_checks)
    run.log(f'boundary checks {json.dumps({k: v for k, v in b_checks.items() if k != "invalid_keys"})}')

    run.stage('b90_reference_audit')
    ref = b90_reference_audit(run, cohort, bnd)
    run.log(f'B90 endpoint audit: compared={ref["rows_compared"]} mismatches={ref["mismatches"]}')

    run.stage('scoring')
    from train.temporal_history_winprob_v3 import CandidateA
    names = list(CandidateA.load(ident['a_path'], ident['a_sha256']).state_names)
    S, model_keys = run_scoring(run, bnd, names, ident, WORKERS)
    run.log(f'scored rows={len(S["row_index"])} model_keys={len(model_keys)}')

    run.stage('p2_reference_audit')
    p2ref = p2_reference_audit(run, S, cohort, names)
    run.log(f'P2 B90 identity pass={p2ref["pass"]} bitwise={p2ref["p2_probabilities_all_bitwise_equal"]}')

    run.stage('labels_features_schema')
    labels = build_labels(run, bnd, S, ident, names, model_keys, cohort)
    labels.to_csv(run.out / 'labels_long.csv', index=False)
    save_npz(run.out / 'labels_long.npz', **{c: (labels[c].to_numpy().astype(str) if labels[c].dtype == object else labels[c].to_numpy()) for c in labels.columns})
    guards = write_features_and_schema(run, S, names, cohort, labels, ident)
    run.log(f'labels rows={len(labels)} feature guards pass={guards["pass"]}')

    run.stage('label_audits')
    laud = label_audits(run, labels, S)
    run.log(f'label audits pass={laud["pass"]} failed={laud["failed"]}')

    run.stage('independent_raw_audit')
    raud = raw_audit(run, labels, WORKERS)
    run.log(f'raw audit pass={raud["pass"]} hard_failures={raud["hard_failures"]}')

    run.stage('analyses')
    ana = analyses(run, labels, smoke)

    run.stage('examples')
    exs = select_examples(run, labels, WORKERS)

    run.stage('determinism')
    det, pert = determinism_and_perturbation(run, bnd, S, names, ident, WORKERS, smoke)
    run.log(f'determinism pass={det["pass"]} perturbation pass={pert["pass"]}')

    run.stage('partitions')
    parts = partition_checks(run, cohort)

    run.stage('hashes_after')
    ha = frozen_hashes(run, v2_matches)
    write_json(run.out / 'hashes_after.json', ha)
    changed = compare_hashes(hb, ha)
    new_after = {p: sha256_file(ROOT / p) for p in NEW_SOURCES if (ROOT / p).exists()}

    provenance = {'role': ROLE, 'lineage_checks': lineage, 'lineage_failures': lfail, 'boundary_checks': {k: v for k, v in b_checks.items() if k != 'invalid_keys'},
                  'inputs': {k: str(p) for k, p in FROZEN_FILES.items()}, 'frozen_trees': {k: str(p) for k, p in FROZEN_TREES.items()},
                  'raw_cache': str(CACHE), 'raw_cache_cohort_files_sha256': hb['raw_cache_cohort_files_sha256'],
                  'frozen_models': ident, 'model_keys_observed': model_keys, 'new_sources_sha256': new_hashes,
                  'new_sources_unchanged_during_run': new_hashes == new_after,
                  'historic_selection_chain': [
                      'measure_postkill_full.py: frozen v3.3 detector over 210,000 cached matches -> exposures.csv (next_start = next detected start or game-end sentinel)',
                      'validate_windows.py: V1 engagement split (9,844 predict_test matches) -> 26,712 exposure rows in 9,198 matches; h=120 A-style endpoint; '
                      'overlap rows (next_start <= L) and endpoints outside frames marked invalid',
                      'event_boundary_cif.py --out b_boundary_60_90_120: keys = window_validation h=120 valid rows (26,693); B-rule endpoints h=60/90/120',
                      'ablate_observed_position.py: h=90 valid B rows merged with V1 p_pre -> predictions.csv',
                      'run_validation_suite.py features(): predictions.csv x b_boundary h=90 -> rows.csv',
                      'regenerate_state_v2.py: rows.csv -> state_value_v2_fix/rows.csv (same keys, endpoints) + V2 states',
                      'P3 (this run): exact state_value_v2_fix keys and order']}
    write_json(run.out / 'source_provenance.json', provenance)

    checks = {
        'contract_tests_pass': proc.returncode == 0,
        'cohort_exact_rows_order_and_lineage': not lfail,
        'no_invalid_rows_any_window': all(v == 0 for v in b_checks['invalid_rows'].values()),
        'first_and_last_kill_present_in_raw': b_checks['first_kill_missing'] == 0 and b_checks['last_kill_missing'] == 0,
        'next_start_recomputed_equal_stored': b_checks['next_start_recompute_mismatch'] == 0,
        'game_end_raw_compatible_with_exposures': b_checks['game_end_exposure_mismatch'] == 0 and b_checks['end_observed_zero'] == 0,
        'no_sentinel_conflicts': b_checks['sentinel_conflicts'] == 0,
        'b90_endpoints_equal_all_state_value_v2_fix_endpoints': ref['mismatches'] == 0,
        'b60_b90_b120_equal_v1_b_boundary_run': all(ref[f'b_boundary_v1_run_h{h}_endpoint_mismatches'] == 0 for h in HS),
        'p2_b90_states_probabilities_identity': p2ref['pass'],
        'feature_schema_guards': guards['pass'],
        'label_audits': laud['pass'],
        'independent_raw_event_audit': raud['pass'],
        'deterministic_rerun_bitwise': det['pass'],
        'future_perturbation_invariance': pert['pass'],
        'zero_overlap_with_V_partitions': parts['pass'],
        'frozen_inputs_unchanged': not changed,
        'new_sources_unchanged_during_run': new_hashes == new_after,
    }
    failed = [k for k, v in checks.items() if not v]
    validation = {'status': 'pass' if not failed else 'fail', 'failed_checks': failed, 'checks': checks, 'role': ROLE,
                  'tests': tests, 'boundary_checks': b_checks, 'b90_reference_audit': ref, 'p2_reference_audit': p2ref,
                  'feature_guards': guards, 'label_audits': laud, 'independent_raw_audit': {k: v for k, v in raud.items() if k != 'failure_examples'},
                  'determinism': det, 'future_perturbation': pert, 'partitions': parts, 'changed_frozen_files': changed,
                  'game_end_not_unique_rows (informational)': raud.get('game_end_not_unique_rows'),
                  'p4_gate': ('READY for P4 as exploratory candidate labels (no unresolved boundary/source mismatch)' if not failed
                              else 'BLOCKED: unresolved check failures listed in failed_checks')}
    write_json(run.out / 'validation.json', validation)
    results = {'status': 'complete', 'smoke': bool(smoke), 'validation_status': validation['status'], 'role': ROLE,
               'cohort': {'rows': len(cohort), 'matches': int(cohort.match.nunique()), 'label_rows': len(labels)},
               'frozen_models': {'A': ident['frozen_models']['A'], 'B': ident['frozen_models']['B'], 'history_version': ident['history_version']},
               'headline': {
                   'positive_rate_A': {h: ana['per_window'][str(h)]['A']['positive'] for h in HS},
                   'exact_zero_A': {h: ana['per_window'][str(h)]['A']['exact_zero']['count'] for h in HS},
                   'flips_A': {k: v['A']['flip'] for k, v in ana['paired_window_flips'].items()},
                   'flips_A_bootstrap': {k: v['A']['flip_bootstrap'] for k, v in ana['paired_window_flips'].items()},
                   'A_vs_B_disagreement': {h: ana['model_A_vs_B_disagreement'][str(h)]['label_disagreement'] for h in HS},
                   'A_vs_B_disagreement_bootstrap': {h: ana['model_A_vs_B_disagreement'][str(h)]['bootstrap'] for h in HS},
                   'terminal_reasons_inclusive': {h: {r_: v['count'] for r_, v in ana['per_window'][str(h)]['terminal_reasons_inclusive'].items()} for h in HS},
                   'mean_followup_s': {h: ana['per_window'][str(h)]['effective_followup_after_last_kill_s']['mean'] for h in HS}},
               'examples_without_actual_case': exs['types_without_actual_case'],
               'elapsed_seconds': round(time.time() - run.started, 1),
               'claims_not_made': protocol['claims_not_made']}
    write_json(run.out / 'results.json', results)
    run.stage('complete', validation_status=validation['status'], failed_checks=failed)
    return validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0, help='number of cohort matches for a smoke run (separate output folder)')
    ap.add_argument('--out', type=str, default='')
    a = ap.parse_args()
    out = Path(a.out) if a.out else (OUT_BASE / f'smoke_run_n{a.smoke}' if a.smoke else OUT_BASE)
    if not str(out.resolve()).startswith(str(OUT_BASE.resolve())):
        raise SystemExit('outputs must stay inside outputs/engagement_labels_v3_sensitivity')
    run = Run(out, a.smoke)
    try:
        v = execute(run, a)
        run.log(f'finished validation={v["status"]} failed={v["failed_checks"]}')
        return 0 if v['status'] == 'pass' else 2
    except Exception as exc:
        run.blocker(run.status.get('stage', 'unknown'), exc)
        run.stage('failed', error=repr(exc))
        return 1


if __name__ == '__main__':
    sys.exit(main())
