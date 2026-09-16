"""Full-corpus stage X: adapt external raw sets to the cache format and run the frozen v3.3 detector.

Uses the ORIGINAL repository modules read-only (the exact sources of the parent full-corpus detector run, hash
verified against outputs/postkill_objective_delay_full/run.json before and after). Per match:
  raw detail/timeline bytes -> SHA-256 check against the frozen preflight manifest -> game condition checks
  (queueId 420, mapId 11, gameVersion prefix == api_patch, 10 participants on teams 100/200)
  -> cache files written inside outputs/full_corpus_training_20260915/external/<set>/cache with the same
  functions and fields as data.cache_io.prebuild_cache -> loaded with data.cache_io.load_match_cache
  -> gameplay.fights.detect_fights + data.index_split._fight_to_ref_row -> exposure rows with
  measure_postkill_objective_delay.analyse_match (same code as the parent exposures.csv).
Raw key availability (known-unavailable vs genuine zero) is audited per set. No winner/target field is read here
except what the unchanged builder copies into events.json (GAME_END is an event); nothing is scored.

--verify-main N re-detects the first N TRAIN matches of the main cache and compares with parent exposures.
"""
from __future__ import annotations

import os

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

sys.path.insert(0, str(C.REPO))
import json  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(C.OUT / 'runtime_repo')
os.environ['LOL_CFG_PRESET'] = 'v3.3'
os.environ['LOL_CFG_OVERRIDES'] = json.dumps({
    'CACHE_DIRNAME': str(C.OUT / 'external' / '_unused_cache_dir'), 'FIGHT_INDEX_CACHE_ENABLED': False,
    'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False, 'CACHE_IN_RAM': False})

import argparse  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

KEYS_PF = ('currentGold', 'totalGold', 'level', 'xp', 'minionsKilled', 'jungleMinionsKilled', 'position', 'championStats')
KEYS_CS = ('health', 'healthMax', 'power', 'powerMax')
DETECTOR_CFG = ('TF2_KILL_CLUSTER_GAP_MS', 'CLUSTER_MAX_DIAMETER', 'TF2_VALIDITY_RADIUS', 'TF2_ENGAGE_PRE_KILL_MS',
                'TF2_MIN_PER_TEAM', 'FEATURE_VERSION', 'FRAME_MS', 'PATCH_LEVEL')
_CTX = {}


def _context():
    if not _CTX:
        from core.config import cfg
        import data.cache_io as cio
        from gameplay.fights import detect_fights, build_anchors_from_events, normalize_patch
        from data.index_split import _fight_to_ref_row
        from core.roles import get_role_slots_from_detail
        import measure_postkill_objective_delay as MPD
        cfg.CACHE_IN_RAM = False
        cfg.DUMP_FIGHTS = False
        _CTX.update(cfg=cfg, cio=cio, detect=detect_fights, anchors=build_anchors_from_events, norm=normalize_patch,
                    ref=_fight_to_ref_row, roles=get_role_slots_from_detail, MPD=MPD)
    return _CTX


def detector_config():
    cfg = _context()['cfg']
    return {k: getattr(cfg, k, None) for k in DETECTOR_CFG}


def repo_source_check():
    run = C.read_json(C.ROOT / 'outputs/postkill_objective_delay_full/run.json')
    out = {}
    for p, recorded in run['source_hashes'].items():
        cur = hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None
        out[str(Path(p).relative_to(C.REPO)).replace('\\', '/')] = cur == recorded
    out['scripts/measure_postkill_objective_delay.py'] = C.sha256_file(_SCRIPTS / 'measure_postkill_objective_delay.py')
    return out


def _atomic_bytes(path, data):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def detect_exposures(mid, pack, cache_dir):
    ctx = _context()
    patch = pack['meta']['patch']
    fights = ctx['detect'](pack, pack['meta']['team_map'])
    refs = []
    for f in fights:
        r = ctx['ref'](f, pack['minute_ts'], match_id=mid, patch=patch)
        if r is not None:
            refs.append(dict(patch=patch, s=r['t_start_ts'], L=r['last_kill_ts'], x=r['anchor_x'], y=r['anchor_y']))
    ctx['MPD'].CACHE = Path(cache_dir)
    _pairs, ex, stats = ctx['MPD'].analyse_match((mid, refs, float('inf'), 4264))
    return refs, ex, dict(stats)


def prepare_chunk(task):
    ctx = _context()
    cio = ctx['cio']
    cache_dir = Path(task['cache_dir'])
    cio.CACHE_DIR = cache_dir
    records, exposures = [], []
    avail = Counter()
    for row in task['rows']:
        mid = row['match_id']
        rec = dict(match_id=mid, set_id=task['set_id'], api_patch=row['api_patch'], public_patch_db=row['public_patch'],
                   game_version=row['game_version'], prior_use=row['prior_use'], cache_status='excluded', reason='',
                   frames=0, refs=0, exposures=0, detector_stats={})
        try:
            raw = Path(row['raw_folder'])
            db = (raw / 'detail' / f'{mid}.json').read_bytes()
            tb = (raw / 'timeline' / f'{mid}.json').read_bytes()
            if hashlib.sha256(db).hexdigest() != row['detail_sha256'] or hashlib.sha256(tb).hexdigest() != row['timeline_sha256']:
                rec['reason'] = 'raw_sha256_mismatch_vs_frozen_manifest'
                records.append(rec)
                continue
            detail, tl = json.loads(db), json.loads(tb)
            if not isinstance(detail, dict) or not isinstance(tl, dict):
                rec['reason'] = 'raw_json_not_object'
                records.append(rec)
                continue
            info = detail.get('info', {}) or {}
            gv = str(info.get('gameVersion', ''))
            cond = dict(queue=int(info.get('queueId', -1) or -1) == 420, map=int(info.get('mapId', -1) or -1) == 11,
                        version='.'.join(gv.split('.')[:2]) == row['api_patch'])
            if not all(cond.values()):
                rec['reason'] = 'game_condition_failed:' + ','.join(k for k, v in cond.items() if not v)
                records.append(rec)
                continue
            parts = info.get('participants', None)
            tm = {}
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict):
                        pid, tid = int(p.get('participantId', 0) or 0), int(p.get('teamId', 0) or 0)
                        if pid > 0 and tid in (100, 200):
                            tm[pid] = tid
            if not isinstance(parts, list) or len(parts) < 10 or len(tm) < 10:
                rec['reason'] = 'participants_or_team_map_incomplete'
                records.append(rec)
                continue
            frames = (tl.get('info', {}) or {}).get('frames', []) or []
            rec['frames'] = len(frames)
            for fr in frames:
                for pid, pf in (fr.get('participantFrames', {}) or {}).items():
                    avail['participant_frames'] += 1
                    for k in KEYS_PF:
                        avail['pf_' + k] += int(k in pf)
                    cs = pf.get('championStats') or {}
                    for k in KEYS_CS:
                        avail['cs_' + k] += int(k in cs)
                for e in fr.get('events', []) or []:
                    avail['event:' + str(e.get('type'))] += 1
            avail['detail_participants_championId'] += sum(1 for p in parts if isinstance(p, dict) and int(p.get('championId', 0) or 0) > 0)
            avail['detail_participants'] += len(parts)
            cache = cio._parse_timeline_to_minute_cache_compat(tl, tm, detail=detail)
            if not cache or any(k not in cache for k in ('minute_ts', 'node_minute', 'global_minute')):
                rec['reason'] = 'timeline_parse_empty_or_missing_arrays'
                records.append(rec)
                continue
            anchors = ctx['anchors'](cache['events'])
            npz_path, ev_path, meta_path = cio.cache_paths(mid)
            arrays = dict(minute_ts=cache['minute_ts'], node_minute=cache['node_minute'], global_minute=cache['global_minute'],
                          gold_team_minute=cache['gold_team_minute'])
            if cache.get('xy_raw_minute', None) is not None:
                arrays['xy_raw_minute'] = cache['xy_raw_minute']
            tmp = npz_path.with_name(npz_path.name + f'.tmp{os.getpid()}')
            with open(tmp, 'wb') as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, npz_path)
            _atomic_bytes(ev_path, json.dumps(cache['events'], ensure_ascii=False).encode('utf-8'))
            meta = {'match_id': mid, 'patch_full': gv, 'patch': ctx['norm'](gv), 'team_map': tm,
                    'role_slots': ctx['roles'](detail), 'anchors': anchors, 'anchor_is_norm': False,
                    'static_meta': cio._extract_static_meta_from_detail(detail),
                    'interp_policy': cio._interp_policy_snapshot(),
                    'feature_version': str(getattr(ctx['cfg'], 'FEATURE_VERSION', ''))}
            _atomic_bytes(meta_path, json.dumps(meta, ensure_ascii=False).encode('utf-8'))
            pack = cio.load_match_cache(mid)
            if pack is None:
                rec['reason'] = 'written_cache_rejected_by_load_match_cache'
                records.append(rec)
                continue
            if pack['meta']['patch'] != row['api_patch']:
                rec['reason'] = f"normalized_patch_mismatch:{pack['meta']['patch']}"
                records.append(rec)
                continue
            refs, ex, stats = detect_exposures(mid, pack, cache_dir)
            rec.update(cache_status='written', refs=len(refs), exposures=len(ex), detector_stats=stats,
                       minute_ts_first=int(pack['minute_ts'][0]), minute_ts_last=int(pack['minute_ts'][-1]),
                       npz_sha256=C.sha256_file(npz_path), events_sha256=C.sha256_file(ev_path), meta_sha256=C.sha256_file(meta_path))
            exposures.extend(ex)
        except Exception as exc:
            rec['reason'] = f'exception:{type(exc).__name__}:{exc}'[:300]
            rec['traceback_tail'] = traceback.format_exc()[-800:]
        records.append(rec)
    return dict(chunk_id=task['chunk_id'], records=records, exposures=exposures, availability=dict(avail))


def verify_main(n, st):
    """TRAIN-only reproduction check of the parent detector on the main cache."""
    ctx = _context()
    ctx['cio'].CACHE_DIR = C.CACHE_MAIN
    rows = [r for r in csv.DictReader(open(C.MAIN_MANIFEST, encoding='utf-8')) if r['role'] == 'TRAIN'][:n]
    ids = {r['match_id'] for r in rows}
    parent = defaultdict(list)
    with open(C.EXPOSURES, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r['match'] in ids:
                parent[r['match']].append(r)
    mism, compared = [], 0
    for r in rows:
        mid = r['match_id']
        pack = ctx['cio'].load_match_cache(mid)
        _refs, ex, _ = detect_exposures(mid, pack, C.CACHE_MAIN)
        mine = [tuple(str(x[k]) for k in ('match', 'patch', 's', 'L', 'next_start', 'end', 'same_match_overlap', 'end_observed')) for x in ex]
        theirs = [tuple(str(x[k]) for k in ('match', 'patch', 's', 'L', 'next_start', 'end', 'same_match_overlap', 'end_observed')) for x in parent[mid]]
        compared += 1
        if mine != theirs:
            mism.append(dict(match=mid, redetected=len(mine), parent=len(theirs)))
    res = dict(train_matches_compared=compared, exposure_rows_parent=sum(len(v) for v in parent.values()),
               mismatching_matches=len(mism), mismatches=mism[:20], detector_config=detector_config(),
               repo_sources_equal_parent=repo_source_check())
    C.write_json(C.OUT / 'external' / 'detector_reproduction_train_check.json', res)
    st.log(json.dumps({k: v for k, v in res.items() if k != 'repo_sources_equal_parent'})[:600])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='KR_16.15,KR_16.14_pilot,NA1_16.13,KR_16.13,EUW1_complete')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--chunk-size', type=int, default=250)
    ap.add_argument('--verify-main', type=int, default=0)
    args = ap.parse_args()
    st = C.Status(C.OUT, 'external_prepare')
    try:
        proto_sha = C.sha256_file(C.OUT / 'protocol.json')
        src_before = repo_source_check()
        if not all(v for k, v in src_before.items() if k.endswith('.py') and not k.startswith('scripts/')):
            raise SystemExit('original repository detector sources differ from the parent run hashes')
        st.update('running', 'source_check', detector_config=detector_config(), next_step='verify detector on TRAIN sample')
        if args.verify_main:
            res = verify_main(args.verify_main, st)
            st.update('running', 'detector_train_reproduction', mismatching_matches=res['mismatching_matches'],
                      compared=res['train_matches_compared'], next_step='prepare external sets')
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        for set_id in args.sets.split(','):
            set_dir = C.OUT / 'external' / set_id
            (set_dir / 'cache').mkdir(parents=True, exist_ok=True)
            (set_dir / 'chunks').mkdir(parents=True, exist_ok=True)
            man_path = C.EXTERNAL_MANIFEST_DIR / f'external_{set_id}.csv'
            rows = list(csv.DictReader(open(man_path, encoding='utf-8', newline='')))
            expected = C.EXTERNAL_SETS[set_id][0]
            if len(rows) != expected or len({r['match_id'] for r in rows}) != expected:
                raise SystemExit(f'{set_id}: manifest rows {len(rows)} != expected {expected}')
            base = dict(set_id=set_id, manifest=str(man_path.relative_to(C.ROOT)), manifest_sha256=C.sha256_file(man_path),
                        expected=expected, api_patch=C.EXTERNAL_SETS[set_id][1], prior_use=C.EXTERNAL_SETS[set_id][2],
                        protocol_sha256=proto_sha, detector_config=detector_config())
            if not rows:
                C.write_json(set_dir / 'prepared_manifest.json', dict(base, status='complete', matches=[],
                             blocker='no complete raw detail/timeline pairs in the frozen manifest; set unavailable, not a completed test'))
                with open(set_dir / 'exposures.csv', 'w', encoding='utf-8', newline='') as f:
                    csv.writer(f).writerow(['match', 'patch', 's', 'L', 'next_start', 'end', 'same_match_overlap', 'end_observed'])
                st.update('running', f'prepared_{set_id}', processed=0, total=0, next_step='next set')
                continue
            tasks, chunk_files = [], []
            for c, s0 in enumerate(range(0, len(rows), args.chunk_size)):
                part = rows[s0:s0 + args.chunk_size]
                plan = C.sha256_json(dict(protocol=proto_sha, set=set_id, rows=[r['match_id'] for r in part],
                                          manifest=base['manifest_sha256']))
                cf = set_dir / 'chunks' / f'chunk_{c:04d}.json'
                chunk_files.append((cf, plan))
                if cf.exists() and C.read_json(cf).get('plan_sha256') == plan:
                    continue
                tasks.append(dict(chunk_id=c, set_id=set_id, rows=part, cache_dir=str(set_dir / 'cache'), plan=plan, path=str(cf)))
            done = sum(1 for cf, plan in chunk_files if cf.exists() and C.read_json(cf).get('plan_sha256') == plan) * args.chunk_size
            st.update('running', f'prepare_{set_id}', processed=min(done, len(rows)), total=len(rows), next_step='exposures')
            t0 = time.time()
            if tasks:
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as ex:
                    futs = {ex.submit(prepare_chunk, t): t for t in tasks}
                    for fut in as_completed(futs):
                        t = futs[fut]
                        r = fut.result()
                        C.write_json(Path(t['path']), dict(plan_sha256=t['plan'], **r))
                        done += len(t['rows'])
                        rate = (done) / max(1e-9, time.time() - t0)
                        st.update('running', f'prepare_{set_id}', processed=min(done, len(rows)), total=len(rows),
                                  rate_per_s=round(rate, 2), next_step='exposures')
            records, exposures, avail = [], [], Counter()
            for cf, plan in chunk_files:
                d = C.read_json(cf)
                if d['plan_sha256'] != plan:
                    raise SystemExit(f'stale chunk {cf}')
                records.extend(d['records'])
                exposures.extend(d['exposures'])
                avail.update(d['availability'])
            with open(set_dir / 'exposures.csv', 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['match', 'patch', 's', 'L', 'next_start', 'end', 'same_match_overlap', 'end_observed'])
                w.writeheader()
                w.writerows(exposures)
            pf = max(1, avail['participant_frames'])
            availability = dict(raw_counts=dict(avail),
                                participant_frame_key_presence={k: avail['pf_' + k] / pf for k in KEYS_PF},
                                champion_stats_key_presence={k: avail['cs_' + k] / pf for k in KEYS_CS},
                                detail_championId_presence=avail['detail_participants_championId'] / max(1, avail['detail_participants']))
            availability['structurally_unavailable_keys'] = [k for k, v in {**availability['participant_frame_key_presence'],
                                                                           **availability['champion_stats_key_presence']}.items() if v == 0]
            reasons = Counter(r['reason'] for r in records if r['cache_status'] != 'written')
            C.write_json(set_dir / 'prepared_manifest.json', dict(
                base, status='complete', matches=records, written=sum(r['cache_status'] == 'written' for r in records),
                excluded=len(records) - sum(r['cache_status'] == 'written' for r in records), exclusion_reasons=dict(reasons),
                exposure_rows=len(exposures), exposures_sha256=C.sha256_file(set_dir / 'exposures.csv'),
                availability=availability, completed_at=time.strftime('%Y-%m-%d %H:%M:%S')))
            st.update('running', f'prepared_{set_id}', processed=len(rows), total=len(rows), written=len(records) - sum(reasons.values()),
                      exclusion_reasons=dict(reasons), exposure_rows=len(exposures), next_step='next set')
        src_after = repo_source_check()
        C.write_json(C.OUT / 'external' / 'repo_source_check.json', dict(before=src_before, after=src_after, unchanged=src_before == src_after))
        st.update('complete', 'external_prepare', next_step='extract external sets (states) with fc20260915_extract.py --set <set>',
                  repo_sources_unchanged=src_before == src_after)
        return 0
    except SystemExit as exc:
        st.update('failed', 'external_prepare', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'external_prepare', error=repr(exc), next_step='inspect, fix, resume')
        return 3


if __name__ == '__main__':
    sys.exit(main())
