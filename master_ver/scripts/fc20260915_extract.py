"""Full-corpus stage E: decode cache files into StateV2 query states and engagement boundaries/states.

For every match of a set (main 210,000-match manifest, or an adapted external cache):
  * reads npz arrays, events.json and meta.json with the existing loader (actual decoding, not headers);
  * V dataset: causal minute grid t = 120000 + k*60000 (t < terminal GAME_END, first_frame <= t <= last frame),
    one deterministic hash-chosen query per nonempty 5-minute bucket; TEST/external sets also keep the
    full grid (secondary trajectories); the final Blue winner W is written to a SEPARATE outcome file;
  * engagements: every parent exposure row (main) or frozen-detector exposure row (external), B-rule
    endpoints for h = 60/90/120 s (scripts/engagement_labels_v3_rules.py), validity flags and reasons,
    StateV2 at q_pre = s-1 ms and at each valid endpoint, snapshot ages, raw event counts during/after;
  * per-match exclusion reasons and raw event-type census (objective availability audit).
Nothing is fitted, scored or selected here. Checkpoints resume only on identical plan hashes.
"""
from __future__ import annotations

import os

_WORKER = os.environ.get('FC_WORKER') == '1'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys

sys.dont_write_bytecode = True
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
import fc20260915_common as C  # noqa: E402

sys.path.insert(0, str(C.WT))
import json  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(C.OUT / 'runtime')
os.environ['LOL_CFG_PRESET'] = 'v3.3'
os.environ['LOL_CFG_OVERRIDES'] = json.dumps({
    'CACHE_DIRNAME': str(C.CACHE_MAIN), 'FIGHT_INDEX_CACHE_ENABLED': False,
    'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False, 'CACHE_IN_RAM': False})

import argparse  # noqa: E402
from bisect import bisect_right  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

import engagement_labels_v3_rules as R  # noqa: E402

HS = C.HORIZONS_S
MAPPED_MONSTERS = ('DRAGON', 'BARON_NASHOR', 'RIFTHERALD', 'HORDE', 'ATAKHAN')
E_KEYS = (['match', 's', 'L', 'q_pre', 'next_start_stored', 'game_end_stored', 'same_match_overlap_stored',
           'end_observed_stored', 'first_kill_raw_count', 'kills_at_L', 'next_kill', 'next_start_recomputed',
           'next_start_eff', 'game_end', 'support_start', 'last_frame', 'pre_ok', 'pre_snapshot', 'pre_unassigned',
           'nonkill_at_L', 'pre_reason', 'X_pre', 'during']
          + [f'{k}_h{h}' for h in HS for k in ('endpoint', 'valid', 'post_snapshot', 'post_unassigned', 'after_raw_kills',
                                               'after_nonkill_same_ts_groups', 'reasons', 'invalid', 'X_post', 'after')])
_CTX = {}


def _context():
    if not _CTX:
        from core.config import cfg, NODE_FEATURE_NAMES
        import data.cache_io as cio
        from data import ram_cache
        cfg.CACHE_IN_RAM = False
        if hasattr(cfg, 'CACHE_MATCH_PACKS_IN_RAM'):
            cfg.CACHE_MATCH_PACKS_IN_RAM = False
        if ram_cache._ram_cache_enabled():
            raise RuntimeError('RAM pack cache unexpectedly enabled')
        from gameplay.state_value import final_outcome
        from gameplay.state_value_v2 import StateBuilder, STATE_VERSION, state_matrix
        if STATE_VERSION != C.STATE_VERSION:
            raise RuntimeError('state version drift')
        _CTX.update(cio=cio, cfg=cfg, node_names=list(NODE_FEATURE_NAMES), final_outcome=final_outcome,
                    V2=StateBuilder, state_matrix=state_matrix, feature_version=str(cfg.FEATURE_VERSION))
    return _CTX


def expected_state_names():
    """State names depend only on code: build one state on a synthetic fixture (no data read)."""
    ctx = _context()
    nn = ctx['node_names']
    node = np.zeros((3, 10, len(nn)))
    pack = dict(minute_ts=np.array([0, 60000, 120000]), node_minute=node, events=[],
                meta=dict(team_map={p: 100 if p <= 5 else 200 for p in range(1, 11)}))
    return list(ctx['V2'](pack, nn).at(60000).values)


def _why_unreadable(cache_dir, mid, ctx):
    files = [Path(cache_dir) / f'{mid}{s}' for s in ('.npz', '.events.json', '.meta.json')]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        return 'missing_files:' + ','.join(missing)
    try:
        meta = json.loads(files[2].read_text(encoding='utf-8'))
    except Exception as exc:
        return f'meta_json_error:{type(exc).__name__}'
    if str(meta.get('feature_version', '')) != ctx['feature_version']:
        return f"feature_version_mismatch:{meta.get('feature_version')!r}"
    return 'loader_rejected_schema_or_decode_error'


def _event_census(events, census):
    for e in events:
        typ = str(e.get('type'))
        census['type:' + typ] += 1
        if typ == 'ELITE_MONSTER_KILL':
            mt, sub = str(e.get('monsterType')), str(e.get('monsterSubType'))
            team = int(e.get('killerTeamId', 0) or 0)
            census[f'elite:{mt}:{sub}:team_{"ok" if team in (100, 200) else team}'] += 1
            if mt not in MAPPED_MONSTERS:
                census['elite_unmapped_monsterType:' + mt] += 1
        elif typ == 'DRAGON_SOUL_GIVEN':
            fields = '|'.join(f'{f}={e.get(f)}' for f in ('name', 'dragonSoul', 'soulType') if f in e)
            team = int(e.get('teamId', 0) or 0)
            census[f'soul:{fields}:team_{"ok" if team in (100, 200) else team}'] += 1
        elif typ == 'BUILDING_KILL':
            census[f'building:{e.get("buildingType")}:{e.get("towerType")}'] += 1


def _state(cache, builder, q):
    if q not in cache:
        cache[q] = builder.at(q)
    return cache[q]


def extract_chunk(task):
    started = time.time()
    ctx = _context()
    cio = ctx['cio']
    cio.CACHE_DIR = Path(task['cache_dir'])
    names = task['names']
    nn = ctx['node_names']
    K = len(R.COUNT_KEYS)
    M = defaultdict(list)   # match-level
    V = defaultdict(list)   # V query rows
    E = defaultdict(list)   # engagement rows
    O = defaultdict(list)   # outcomes (separate file)
    census = Counter()
    worker_errors = {}
    for item in task['items']:
        mid = item['match_id']
        exps = item['exposures']
        m = dict(match=mid, role=item['role'], sub_role=str(item['sub_role']), loaded=0, load_reason='',
                 builder_ok=0, builder_reason='', v_eligible=0, v_reason='', n_grid=0, n_bucket=0,
                 first_frame=-1, last_frame=-1, n_exposures=len(exps), order_differs=-1, patch_meta='',
                 patch_ok=0, unmapped_elite=0, game_end_events=0)
        outcome = dict(match=mid, winner=-1, terminal=-1, reason='not_loaded')
        pack = None
        try:
            pack = cio.load_match_cache(mid)
        except Exception as exc:  # loader normally returns None; keep the reason
            m['load_reason'] = f'loader_exception:{type(exc).__name__}:{exc}'[:200]
        builder = None
        ts = None
        if pack is None:
            if not m['load_reason']:
                m['load_reason'] = _why_unreadable(task['cache_dir'], mid, ctx)
        else:
            m['loaded'] = 1
            events = pack['events']
            ts = np.asarray(pack['minute_ts'], dtype=np.int64)
            m['first_frame'], m['last_frame'] = int(ts[0]), int(ts[-1])
            m['patch_meta'] = str(pack['meta'].get('patch', ''))
            m['patch_ok'] = int(m['patch_meta'] == item['patch_expected'])
            before = sum(v for k, v in census.items() if k.startswith('elite_unmapped'))
            _event_census(events, census)
            m['unmapped_elite'] = sum(v for k, v in census.items() if k.startswith('elite_unmapped')) - before
            m['game_end_events'] = sum(1 for e in events if e.get('type') == 'GAME_END')
            try:
                builder = ctx['V2'](pack, nn)
                m['builder_ok'] = 1
                tm = {int(k): int(v) for k, v in pack['meta']['team_map'].items()}
                slots = pack['meta'].get('role_slots') or {}
                try:
                    role_order = sorted(tm, key=lambda p: int(slots[p]))
                    m['order_differs'] = int(role_order != list(builder.participant_order))
                except (KeyError, TypeError, ValueError):
                    m['order_differs'] = -1
            except Exception as exc:
                m['builder_reason'] = f'{type(exc).__name__}:{exc}'[:200]
            try:
                winner, terminal = ctx['final_outcome'](events)
                outcome.update(winner=int(winner), terminal=int(terminal), reason='ok')
            except Exception as exc:
                outcome['reason'] = f'final_outcome:{type(exc).__name__}:{exc}'[:200]
        states = {}
        # ------------------------------------------------ V queries
        if builder is not None and outcome['reason'] == 'ok':
            grid = C.value_grid(outcome['terminal'], ts[0], ts[-1])
            m['n_grid'] = len(grid)
            if not grid:
                m['v_reason'] = 'no_eligible_minute_query'
            else:
                sampled = C.bucket_sample(mid, grid)
                m['n_bucket'] = len(sampled)
                queries = grid if item['want_grid'] else sampled
                sset = set(sampled)
                try:
                    sts = [_state(states, builder, t) for t in queries]
                    X = ctx['state_matrix'](sts, names)
                    rows = []
                    for t, st in zip(queries, sts):
                        snap = int(ts[np.searchsorted(ts, t, side='right') - 1])
                        if st.query_ms != t or st.snapshot_ms != snap or st.snapshot_ms > t:
                            raise ValueError(f'query/snapshot violation at {t}')
                        rows.append((t, t // C.BUCKET_MS, int(t in sset), st.snapshot_ms, st.unassigned_soul_events))
                    for t, b, isb, snap, un in rows:
                        V['match'].append(mid)
                        V['query_ms'].append(t)
                        V['bucket'].append(b)
                        V['is_bucket_sample'].append(isb)
                        V['snapshot_ms'].append(snap)
                        V['unassigned'].append(un)
                    V['X'].append(X)
                    m['v_eligible'] = 1
                except Exception as exc:
                    m['v_reason'] = f'state_error:{type(exc).__name__}:{exc}'[:200]
        elif pack is None:
            m['v_reason'] = 'cache_not_loaded'
        elif builder is None:
            m['v_reason'] = 'invalid_roster_or_state_schema'
        else:
            m['v_reason'] = 'game_end_missing_or_conflicting'
        O['match'].append(mid)
        O['winner'].append(outcome['winner'])
        O['terminal'].append(outcome['terminal'])
        O['reason'].append(outcome['reason'])
        # ------------------------------------------------ engagements
        if exps:
            if pack is not None:
                events = pack['events']
                kills = sorted(int(e['timestamp']) for e in events if e.get('type') == 'CHAMPION_KILL')
                kc = Counter(kills)
                ends = sorted(int(e['timestamp']) for e in events if e.get('type') == 'GAME_END')
                end_sem = max(ends) if ends else max(int(e['timestamp']) for e in events)
                cat_events, nonkill_ts = [], Counter()
                for e in events:
                    cats, team = R.categorize(e)
                    if cats:
                        cat_events.append((int(e['timestamp']), cats, team))
                    if e.get('type') in R.NON_KILL_PREDICTOR_TYPES:
                        nonkill_ts[int(e['timestamp'])] += 1
                cat_events.sort(key=lambda x: x[0])
                exp_sorted = sorted(exps, key=lambda x: (x['s'], x['L']))
                next_map = {}
                for i, x in enumerate(exp_sorted):
                    next_map[(x['s'], x['L'])] = exp_sorted[i + 1]['s'] if i + 1 < len(exp_sorted) else end_sem
            for x in exps:
                s, L = int(x['s']), int(x['L'])
                E['match'].append(mid)
                E['s'].append(s)
                E['L'].append(L)
                E['q_pre'].append(s - 1)
                E['next_start_stored'].append(int(x['next_start']))
                E['game_end_stored'].append(int(x['end']))
                E['same_match_overlap_stored'].append(int(x['same_match_overlap']))
                E['end_observed_stored'].append(int(x['end_observed']))
                row_reason = ''
                if pack is None:
                    row_reason = 'cache_not_loaded'
                elif builder is None:
                    row_reason = 'invalid_roster_or_state_schema'
                if row_reason:
                    for k, v in (('first_kill_raw_count', -1), ('kills_at_L', -1), ('next_kill', -1),
                                 ('next_start_recomputed', -1), ('next_start_eff', -1), ('game_end', -1),
                                 ('support_start', -1), ('last_frame', -1), ('pre_ok', 0), ('pre_snapshot', -1),
                                 ('pre_unassigned', -1), ('nonkill_at_L', -1)):
                        E[k].append(v)
                    E['pre_reason'].append(row_reason)
                    E['X_pre'].append(np.full(len(names), np.nan))
                    E['during'].append(np.full(K, -1, dtype=np.int32))
                    for h in HS:
                        for k, v in (('endpoint', -1), ('valid', 0), ('post_snapshot', -1), ('post_unassigned', -1),
                                     ('after_raw_kills', -1), ('after_nonkill_same_ts_groups', -1)):
                            E[f'{k}_h{h}'].append(v)
                        E[f'reasons_h{h}'].append('')
                        E[f'invalid_h{h}'].append(row_reason)
                        E[f'X_post_h{h}'].append(np.full(len(names), np.nan))
                        E[f'after_h{h}'].append(np.full(K, -1, dtype=np.int32))
                    continue
                K_ms = s + 15_000
                nk = R.next_kill_after(kills, L)
                ns_eff = R.effective_next_start(int(x['next_start']), end_sem)
                support_start, last_frame = max(0, int(ts[0])), int(ts[-1])
                E['first_kill_raw_count'].append(kc.get(K_ms, 0))
                E['kills_at_L'].append(kc.get(L, 0))
                E['next_kill'].append(-1 if nk == math.inf else int(nk))
                E['next_start_recomputed'].append(int(next_map[(s, L)]))
                E['next_start_eff'].append(-1 if ns_eff == math.inf else int(ns_eff))
                E['game_end'].append(int(end_sem))
                E['support_start'].append(support_start)
                E['last_frame'].append(last_frame)
                E['nonkill_at_L'].append(nonkill_ts.get(L, 0))
                during = R.count_events(cat_events, s - 1, L)
                E['during'].append(np.asarray([during.get(k, 0) for k in R.COUNT_KEYS], dtype=np.int32))
                pre_reason = ''
                if not (support_start <= s - 1 <= last_frame):
                    pre_reason = 'q_pre_outside_observed_frames'
                    E['X_pre'].append(np.full(len(names), np.nan))
                    E['pre_snapshot'].append(-1)
                    E['pre_unassigned'].append(-1)
                else:
                    try:
                        st = _state(states, builder, s - 1)
                        E['X_pre'].append(ctx['state_matrix']([st], names)[0])
                        E['pre_snapshot'].append(st.snapshot_ms)
                        E['pre_unassigned'].append(st.unassigned_soul_events)
                    except Exception as exc:
                        pre_reason = f'pre_state_error:{type(exc).__name__}:{exc}'[:160]
                        E['X_pre'].append(np.full(len(names), np.nan))
                        E['pre_snapshot'].append(-1)
                        E['pre_unassigned'].append(-1)
                E['pre_ok'].append(int(not pre_reason))
                E['pre_reason'].append(pre_reason)
                for h in HS:
                    e_ms, reasons, cand = R.endpoint_rule(L, h, nk, ns_eff, end_sem)
                    val = R.endpoint_validity(e_ms, L, s - 1, support_start, last_frame)
                    inval = [k for k, ok in val.items() if not ok]
                    if int(x['next_start']) <= L:
                        inval.insert(0, 'same_match_overlap_next_start_le_L')
                    if pre_reason:
                        inval.append(pre_reason)
                    E[f'endpoint_h{h}'].append(int(e_ms))
                    E[f'reasons_h{h}'].append('|'.join(reasons))
                    after = R.count_events(cat_events, L, e_ms)
                    E[f'after_h{h}'].append(np.asarray([after.get(k, 0) for k in R.COUNT_KEYS], dtype=np.int32))
                    E[f'after_raw_kills_h{h}'].append(bisect_right(kills, e_ms) - bisect_right(kills, L))
                    E[f'after_nonkill_same_ts_groups_h{h}'].append(
                        int(sum(1 for t, n in nonkill_ts.items() if L < t <= e_ms and n >= 2)))
                    post = None
                    if not inval:
                        try:
                            post = _state(states, builder, int(e_ms))
                            if post.snapshot_ms > e_ms:
                                raise ValueError('future snapshot')
                        except Exception as exc:
                            inval.append(f'post_state_error:{type(exc).__name__}:{exc}'[:160])
                            post = None
                    E[f'valid_h{h}'].append(int(not inval))
                    E[f'invalid_h{h}'].append('|'.join(inval))
                    if post is not None:
                        E[f'X_post_h{h}'].append(ctx['state_matrix']([post], names)[0])
                        E[f'post_snapshot_h{h}'].append(post.snapshot_ms)
                        E[f'post_unassigned_h{h}'].append(post.unassigned_soul_events)
                    else:
                        E[f'X_post_h{h}'].append(np.full(len(names), np.nan))
                        E[f'post_snapshot_h{h}'].append(-1)
                        E[f'post_unassigned_h{h}'].append(-1)
        for k, v in m.items():
            M[k].append(v)
    arrays = {}
    for k, v in M.items():
        arrays['m_' + k] = np.asarray(v, dtype=('U200' if isinstance(v[0], str) else np.int64))
    nV = len(V['match'])
    arrays['v_match'] = np.asarray(V['match'], dtype='U24')
    for k in ('query_ms', 'bucket', 'is_bucket_sample', 'snapshot_ms', 'unassigned'):
        arrays['v_' + k] = np.asarray(V[k], dtype=np.int64)
    arrays['v_X'] = np.concatenate(V['X']).astype(np.float64) if V['X'] else np.zeros((0, len(names)))
    if arrays['v_X'].shape[0] != nV:
        raise RuntimeError('V row bookkeeping mismatch')
    nE = len(E['match'])
    if not nE:
        for k in E_KEYS:
            E[k] = []
    for k, v in E.items():
        if k in ('match',) or k.startswith('reasons_') or k.startswith('invalid_') or k == 'pre_reason':
            arrays['e_' + k] = np.asarray(v, dtype='U200') if nE else np.zeros(0, dtype='U200')
        elif k.startswith('X_'):
            arrays['e_' + k] = np.vstack(v).astype(np.float64) if nE else np.zeros((0, len(names)))
        elif k == 'during' or k.startswith('after_h'):
            arrays['e_' + k] = np.vstack(v).astype(np.int32) if nE else np.zeros((0, K), dtype=np.int32)
        else:
            arrays['e_' + k] = np.asarray(v, dtype=np.int64)
    arrays.update(names=np.asarray(names), count_keys=np.asarray(R.COUNT_KEYS), plan_sha256=np.asarray(task['plan_sha256']),
                  set_id=np.asarray(task['set_id']), state_version=np.asarray(C.STATE_VERSION),
                  census_json=np.asarray(json.dumps(dict(census), sort_keys=True)))
    h_states = C.save_npz(Path(task['path']), **arrays)
    h_out = C.save_npz(Path(task['outcome_path']), o_match=np.asarray(O['match'], dtype='U24'),
                       o_winner_blue=np.asarray(O['winner'], dtype=np.int64),
                       o_terminal_ms=np.asarray(O['terminal'], dtype=np.int64),
                       o_reason=np.asarray(O['reason'], dtype='U200'), plan_sha256=np.asarray(task['plan_sha256']),
                       set_id=np.asarray(task['set_id']),
                       role=np.asarray('OUTCOME_SIDE_TARGET_ONLY_NOT_A_FEATURE'))
    return dict(chunk_id=task['chunk_id'], matches=len(M['match']), v_rows=nV, e_rows=nE,
                seconds=round(time.time() - started, 1), states_sha256=h_states, outcomes_sha256=h_out,
                loaded=int(sum(M['loaded'])), v_eligible=int(sum(M['v_eligible'])), errors=worker_errors)


# ------------------------------------------------------------------ plans
def read_main_manifest():
    rows = list(csv.DictReader(open(C.MAIN_MANIFEST, encoding='utf-8', newline='')))
    counts = Counter((r['role'], r['patch']) for r in rows)
    for role, (patch, n) in C.MAIN_EXPECTED.items():
        if counts.get((role, patch), 0) != n:
            raise SystemExit(f'manifest count mismatch {role} {patch}: {counts.get((role, patch))} != {n}')
    if len(rows) != C.MAIN_TOTAL or len({r['match_id'] for r in rows}) != C.MAIN_TOTAL:
        raise SystemExit('manifest total/unique mismatch')
    return rows


def read_exposures(path, id_to_patch):
    by = defaultdict(list)
    bad_patch, not_in = 0, 0
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r['match'] not in id_to_patch:
                not_in += 1
                continue
            if r['patch'] != id_to_patch[r['match']]:
                bad_patch += 1
            by[r['match']].append(dict(s=int(r['s']), L=int(r['L']), next_start=int(r['next_start']), end=int(r['end']),
                                       same_match_overlap=int(r['same_match_overlap']),
                                       end_observed=int(r['end_observed']), patch=r['patch']))
    if bad_patch or not_in:
        raise SystemExit(f'exposure rows with patch mismatch {bad_patch} / not in manifest {not_in}')
    return by


def sub_role(role, mid):
    if role == 'TRAIN':
        return f'fold{C.train_fold(mid)}'
    if role == 'VALIDATION':
        return C.validation_role(mid)
    return role


def build_items(args):
    if args.set == 'MAIN':
        rows = read_main_manifest()
        id_to_patch = {r['match_id']: r['patch'] for r in rows}
        ex = read_exposures(C.EXPOSURES, id_to_patch)
        if args.smoke:
            rows = [r for r in rows if r['role'] == 'TRAIN'][:args.smoke]
        items = [dict(match_id=r['match_id'], role=r['role'], sub_role=sub_role(r['role'], r['match_id']),
                      patch_expected=r['patch'], want_grid=r['role'] == 'TEST', exposures=ex.get(r['match_id'], []))
                 for r in rows]
        return items, str(C.CACHE_MAIN), dict(manifest_sha256=C.sha256_file(C.MAIN_MANIFEST),
                                              exposures_sha256=C.sha256_file(C.EXPOSURES))
    set_dir = C.OUT / 'external' / args.set
    man = C.read_json(set_dir / 'prepared_manifest.json')
    if man['status'] != 'complete':
        raise SystemExit('external preparation incomplete')
    ex = defaultdict(list)
    with open(set_dir / 'exposures.csv', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            ex[r['match']].append(dict(s=int(r['s']), L=int(r['L']), next_start=int(r['next_start']), end=int(r['end']),
                                       same_match_overlap=int(r['same_match_overlap']),
                                       end_observed=int(r['end_observed']), patch=r['patch']))
    items = [dict(match_id=r['match_id'], role=args.set, sub_role='EXTERNAL', patch_expected=r['api_patch'],
                  want_grid=True, exposures=ex.get(r['match_id'], []))
             for r in man['matches'] if r['cache_status'] == 'written']
    return items, str(set_dir / 'cache'), dict(prepared_manifest_sha256=C.sha256_file(set_dir / 'prepared_manifest.json'),
                                               exposures_sha256=C.sha256_file(set_dir / 'exposures.csv'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='MAIN')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--chunk-size', type=int, default=200)
    ap.add_argument('--smoke', type=int, default=0, help='TRAIN-only smoke on the first N TRAIN matches')
    args = ap.parse_args()
    ws = os.environ.get('TEAMFIGHT_WORKSPACE')
    if ws and Path(ws).resolve() != C.ROOT.resolve():
        raise SystemExit('script root differs from TEAMFIGHT_WORKSPACE')
    base = (C.OUT / 'smoke_train_only' / 'extract') if args.smoke else (C.OUT / 'extract')
    group = f'extract_{args.set}' + ('_smoke' if args.smoke else '')
    st = C.Status(C.OUT if not args.smoke else C.OUT / 'smoke_train_only', group)
    try:
        proto_path = C.OUT / 'protocol.json'
        if proto_path.exists():
            protocol_sha = C.sha256_file(proto_path)
        elif args.smoke:
            protocol_sha = 'SMOKE_TRAIN_ONLY_BEFORE_PROTOCOL'
        else:
            raise SystemExit('protocol.json must be written before full extraction')
        st.update('running', 'plan', next_step='decode caches')
        items, cache_dir, input_hashes = build_items(args)
        names = expected_state_names()
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        out_dir = base / args.set
        (out_dir / 'states').mkdir(parents=True, exist_ok=True)
        (out_dir / 'outcomes_SEALED').mkdir(parents=True, exist_ok=True)
        tasks, reused, chunk_meta = [], 0, []
        for c, s0 in enumerate(range(0, len(items), args.chunk_size)):
            part = items[s0:s0 + args.chunk_size]
            plan_sha = C.sha256_json(dict(protocol_sha256=protocol_sha, inputs=input_hashes, cache_dir=cache_dir,
                                          names=names, items=part, set=args.set))
            path = out_dir / 'states' / f'chunk_{c:05d}.npz'
            opath = out_dir / 'outcomes_SEALED' / f'chunk_{c:05d}.npz'
            chunk_meta.append(dict(chunk=c, matches=len(part), plan_sha256=plan_sha, path=str(path.relative_to(C.OUT)),
                                   outcome_path=str(opath.relative_to(C.OUT))))
            if path.exists() and opath.exists():
                with np.load(path, allow_pickle=False) as z:
                    ok = str(z['plan_sha256']) == plan_sha
                with np.load(opath, allow_pickle=False) as z:
                    ok = ok and str(z['plan_sha256']) == plan_sha
                if not ok:
                    raise SystemExit(f'stale checkpoint {path.name}: plan hash differs; refusing to mix')
                reused += 1
                continue
            tasks.append(dict(chunk_id=c, set_id=args.set, cache_dir=cache_dir, items=part, names=names,
                              plan_sha256=plan_sha, path=str(path), outcome_path=str(opath)))
        total_m = len(items)
        done_m = sum(cm['matches'] for cm in chunk_meta) - sum(len(t['items']) for t in tasks)
        st.update('running', 'decode', processed=done_m, total=total_m, chunks_total=len(chunk_meta),
                  chunks_reused=reused, workers=args.workers, next_step='assemble manifest')
        results = []
        t0 = time.time()
        done_this = 0
        if tasks:
            os.environ['FC_WORKER'] = '1'
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as ex:
                futs = {ex.submit(extract_chunk, t): t for t in tasks}
                for fut in as_completed(futs):
                    t = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as exc:
                        st.update('failed', 'decode', processed=done_m, total=total_m,
                                  error=f'chunk {t["chunk_id"]}: {type(exc).__name__}: {exc}',
                                  next_step='fix worker and resume (completed chunks reused)')
                        st.log(traceback.format_exc())
                        for f2 in futs:
                            f2.cancel()
                        raise
                    results.append(r)
                    done_m += r['matches']
                    done_this += r['matches']
                    rate = done_this / max(1e-9, time.time() - t0)
                    st.update('running', 'decode', processed=done_m, total=total_m, chunks_done=len(results) + reused,
                              chunks_total=len(chunk_meta), rate_per_s=round(rate, 2),
                              eta_s=round((total_m - done_m) / max(rate, 1e-9)), last_chunk_seconds=r['seconds'],
                              next_step='assemble manifest')
        # manifest of chunk files (hashes), counts
        summary = dict(set=args.set, smoke=bool(args.smoke), protocol_sha256=protocol_sha, inputs=input_hashes,
                       cache_dir=cache_dir, names=names, names_sha256=C.sha256_json(names), matches=total_m,
                       chunks=[])
        agg = Counter()
        census = Counter()
        for cm in chunk_meta:
            with np.load(C.OUT / cm['path'], allow_pickle=False) as z:
                if z['names'].tolist() != names or str(z['plan_sha256']) != cm['plan_sha256']:
                    raise SystemExit(f'chunk {cm["chunk"]} schema/plan mismatch')
                agg['matches'] += len(z['m_match'])
                agg['loaded'] += int(z['m_loaded'].sum())
                agg['v_eligible'] += int(z['m_v_eligible'].sum())
                agg['v_rows'] += len(z['v_match'])
                agg['v_bucket_rows'] += int(z['v_is_bucket_sample'].sum())
                agg['e_rows'] += len(z['e_match'])
                for h in HS:
                    agg[f'e_valid_h{h}'] += int(z[f'e_valid_h{h}'].sum())
                census.update(json.loads(str(z['census_json'])))
            cm['states_sha256'] = C.sha256_file(C.OUT / cm['path'])
            cm['outcomes_sha256'] = C.sha256_file(C.OUT / cm['outcome_path'])
            summary['chunks'].append(cm)
        summary['totals'] = dict(agg)
        summary['event_census'] = dict(sorted(census.items()))
        summary['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        C.write_json(out_dir / 'extraction_manifest.json', summary)
        st.update('complete', 'decode', processed=total_m, total=total_m, totals=dict(agg),
                  next_step='fit V (main) / label (external after freeze)')
        return 0
    except SystemExit as exc:
        st.update('failed', 'plan', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'exception', error=repr(exc), next_step='inspect log, fix, resume')
        return 3


if __name__ == '__main__':
    sys.exit(main())
