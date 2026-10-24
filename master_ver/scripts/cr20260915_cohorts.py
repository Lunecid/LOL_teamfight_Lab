"""Stage A1: restore the v3.3 participation scale columns and write the cohort manifest before any fitting.

Main corpus: cluster_blue/cluster_red (and presence present_blue/red, diagnostic) are recovered from all 32
corpus_shards_v33 shards by the exact (groups=match, engage_ts=s) key and joined one-to-one to every engagement row
of the frozen full-corpus labels (566,452 exposure rows). The stored meaning is kept: gameplay/fights.py
det_cluster_* = kill-cluster participants plus in-radius (3,000 u) interaction actors incl. shop events
(TF2_EXCLUDE_SHOP_INTERACTIONS False); a merged engagement keeps its earlier candidate's counts; the ref row stores
int(count or -1). A -1 is read as zero only where the frozen detector, re-run on that match, shows the raw count is
the integer 0 at the same (s, L); otherwise the row is flagged unknown (kept in the pooled E baseline only).
Detector reproduction is also checked on hash-selected TRAIN-only fixture matches (counts and exposure tuples).
External sets: the same frozen detector re-run on the already-adapted caches regenerates the participation fields;
exposure tuples must equal the stored external exposures bitwise and keys must equal the frozen label rows.

Cohorts: E = all engagement rows; T = known & min(blue, red) >= 4; N = known & min < 4 (pick <= 1, skirmish 2..3
diagnostic). No labels (Y), deltas or outcomes are read here.
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
import cr20260915_common as K  # noqa: E402

import json  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime_repo')
os.environ['LOL_CFG_PRESET'] = 'v3.3'
os.environ['LOL_CFG_OVERRIDES'] = json.dumps({
    'CACHE_DIRNAME': str(C.CACHE_MAIN), 'FIGHT_INDEX_CACHE_ENABLED': False,
    'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False, 'CACHE_IN_RAM': False})

import argparse  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

EXPO_FIELDS = ('match', 'patch', 's', 'L', 'next_start', 'end', 'same_match_overlap', 'end_observed')
DET_KEYS = ('det_cluster_blue', 'det_cluster_red', 'det_present_blue', 'det_present_red')
_CTX = {}


def _context():
    if not _CTX:
        sys.path.insert(0, str(K.REPO))
        from core.config import cfg
        import data.cache_io as cio
        from gameplay.fights import detect_fights
        from data.index_split import _fight_to_ref_row
        import measure_postkill_objective_delay as MPD
        cfg.CACHE_IN_RAM = False
        cfg.DUMP_FIGHTS = False
        _CTX.update(cfg=cfg, cio=cio, detect=detect_fights, ref=_fight_to_ref_row, MPD=MPD)
    return _CTX


def detector_settings():
    cfg = _context()['cfg']
    keys = ('TF2_KILL_CLUSTER_GAP_MS', 'CLUSTER_MAX_DIAMETER', 'TF2_VALIDITY_RADIUS', 'TF2_ENGAGE_PRE_KILL_MS',
            'TF2_MIN_PER_TEAM', 'TF2_INTERACTION_RADIUS', 'TF2_EXCLUDE_SHOP_INTERACTIONS', 'FIGHT_HORIZON_SEC', 'FEATURE_VERSION')
    return {k: getattr(cfg, k, None) for k in keys}


def detect_chunk(task):
    """Re-run the frozen detector on one list of matches of one cache directory."""
    ctx = _context()
    cio = ctx['cio']
    cio.CACHE_DIR = Path(task['cache_dir'])
    ctx['MPD'].CACHE = Path(task['cache_dir'])
    out = []
    for mid in task['matches']:
        rec = dict(match=mid, loaded=0, error='', refs=[], exposures=[])
        try:
            pack = cio.load_match_cache(mid)
            if pack is None:
                rec['error'] = 'cache_not_loaded'
                out.append(rec)
                continue
            rec['loaded'] = 1
            patch = pack['meta']['patch']
            fights = ctx['detect'](pack, pack['meta']['team_map'])
            refs = []
            for f in fights:
                r = ctx['ref'](f, pack['minute_ts'], match_id=mid, patch=patch)
                if r is None:
                    continue
                raw = {k: (repr(f[k]) if k in f else None) for k in DET_KEYS}
                raw_int = {k: (int(f[k]) if (k in f and isinstance(f[k], (int, np.integer))) else None) for k in DET_KEYS}
                refs.append(dict(patch=patch, s=int(r['t_start_ts']), L=int(r['last_kill_ts']), x=r['anchor_x'], y=r['anchor_y'],
                                 ref_cluster_blue=int(r['det_cluster_blue']), ref_cluster_red=int(r['det_cluster_red']),
                                 ref_present_blue=int(r['det_present_blue']), ref_present_red=int(r['det_present_red']),
                                 raw=raw, raw_int=raw_int))
            _pairs, ex, _stats = ctx['MPD'].analyse_match((mid, [dict(patch=x['patch'], s=x['s'], L=x['L'], x=x['x'], y=x['y']) for x in refs],
                                                            float('inf'), 4264))
            rec['refs'] = [{k: v for k, v in x.items() if k not in ('x', 'y')} for x in refs]
            rec['exposures'] = [tuple(str(e[k]) for k in EXPO_FIELDS) for e in ex]
        except Exception as exc:
            rec['error'] = f'{type(exc).__name__}:{exc}'[:300]
        out.append(rec)
    return out


def run_detector(matches, cache_dir, workers, st, label):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    matches = list(matches)
    chunks = [matches[i:i + 100] for i in range(0, len(matches), 100)]
    res = {}
    t0 = time.time()
    if not chunks:
        return res
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
        futs = [ex.submit(detect_chunk, dict(cache_dir=str(cache_dir), matches=c)) for c in chunks]
        for fut in as_completed(futs):
            for rec in fut.result():
                res[rec['match']] = rec
            rate = len(res) / max(1e-9, time.time() - t0)
            if len(res) % 2000 < 100 or len(res) == len(matches):
                st.update('running', f'detector_{label}', processed=len(res), total=len(matches), rate_per_s=round(rate, 1),
                          next_step='compare')
    return res


def read_exposures(path):
    by = defaultdict(list)
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            by[r['match']].append(tuple(r[k] for k in EXPO_FIELDS))
    return by


def label_keys(name):
    """Row keys of a frozen label file (no Y / delta / outcome member is read)."""
    with np.load(K.FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        R = {k: z[k] for k in ('match', 's', 'L', 'sub_role', 'pre_ok', 'valid_h60', 'valid_h90', 'valid_h120')}
    with np.load(K.FC / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        fm, fs = z['match'], z['s_ms']
    if not (np.array_equal(fm, R['match']) and np.array_equal(fs, R['s'])):
        raise SystemExit(f'{name}: label/feature row order differs')
    R['match'] = R['match'].astype(str)
    R['sub_role'] = R['sub_role'].astype(str)
    return R


def fixture_rank(mid):
    return hashlib.sha256(f'cr20260915-detector-fixture:{mid}'.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--fixture-n', type=int, default=400)
    ap.add_argument('--skip-external', action='store_true')
    args = ap.parse_args()
    st = K.Status('cohorts')
    out_dir = K.OUT / 'cohorts'
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if (out_dir / 'cohort_manifest.json').exists():
            raise SystemExit('cohort_manifest.json already exists (cohorts are written once before fitting)')
        if (K.OUT / 'selection').exists():
            raise SystemExit('selection directory exists: cohort manifest must predate fitting')
        st.update('running', 'input_hashes', next_step='load shards')
        shard_paths = sorted(K.SHARDS.glob('shard_*.npz'))
        smani = C.read_json(K.SHARDS / 'manifest.json')
        if len(shard_paths) != 32 or smani.get('num_shards') != 32 or not smani.get('complete'):
            raise SystemExit('v3.3 shard set incomplete')
        if smani['scale'] != dict(pick_max=K.PICK_MAX, skirmish_min=K.SKIRMISH_MIN, teamfight_min=K.TEAMFIGHT_MIN):
            raise SystemExit(f'manifest scale block differs from spec: {smani["scale"]}')
        inputs = dict(shard_manifest_sha256=C.sha256_file(K.SHARDS / 'manifest.json'), shard_manifest_scale=smani['scale'],
                      shard_manifest_detector=smani['detector'], shard_git_commit=smani.get('git_commit'),
                      shard_sha256={p.name: C.sha256_file(p) for p in shard_paths},
                      parent_exposures_sha256=C.sha256_file(C.EXPOSURES))
        lman_tv = C.read_json(K.FC / 'labels' / 'labels_trainval_manifest.json')
        lman_te = C.read_json(K.FC / 'labels' / 'labels_test_external_manifest.json')
        names = ['MAIN_TRAIN', 'MAIN_VALIDATION', 'MAIN_TEST'] + [f'EXT_{s}' for s in K.EXT]
        label_hash_checks = {}
        for nm in names:
            summ = (lman_tv if nm in ('MAIN_TRAIN', 'MAIN_VALIDATION') else lman_te)['summaries'][nm]
            label_hash_checks[nm] = dict(labels=C.sha256_file(K.FC / 'labels' / f'{nm}_labels.npz') == summ['labels_sha256'],
                                         features=C.sha256_file(K.FC / 'labels' / f'{nm}_features_pre_only.npz') == summ['features_sha256'])
        if not all(v['labels'] and v['features'] for v in label_hash_checks.values()):
            raise SystemExit(f'frozen label/feature files changed: {label_hash_checks}')
        inputs['frozen_label_hashes_equal_prior_manifests'] = label_hash_checks
        st.update('running', 'load_shards', next_step='join')
        keys = ('groups', 'patch', 'engage_ts', 'cluster_blue', 'cluster_red', 'present_blue', 'present_red')
        parts = {k: [] for k in keys}
        shard_of = []
        for i, p in enumerate(shard_paths):
            with np.load(p, allow_pickle=False) as z:
                for k in keys:
                    parts[k].append(np.asarray(z[k]))
                shard_of.append(np.full(len(z['groups']), i, dtype=np.int16))
        A = {k: np.concatenate(v) for k, v in parts.items()}
        A['shard'] = np.concatenate(shard_of)
        A['groups'] = A['groups'].astype(str)
        index = {}
        for i, (m, s) in enumerate(zip(A['groups'].tolist(), A['engage_ts'].tolist())):
            if (m, s) in index:
                raise SystemExit(f'duplicate shard key {(m, s)}')
            index[(m, s)] = i
        # ------------------------------------------------ main join
        rows = {}
        used = np.zeros(len(A['groups']), dtype=np.int64)
        for nm in names[:3]:
            R = label_keys(nm)
            ix = np.asarray([index.get((m, int(s)), -1) for m, s in zip(R['match'].tolist(), R['s'].tolist())], dtype=np.int64)
            if (ix < 0).any():
                raise SystemExit(f'{nm}: {int((ix < 0).sum())} label rows have no shard row')
            np.add.at(used, ix, 1)
            R['shard_row'] = ix
            rows[nm] = R
        join = dict(shard_rows=int(len(used)), shard_rows_used_once=int((used == 1).sum()), shard_rows_unused=int((used == 0).sum()),
                    shard_rows_used_multiple=int((used > 1).sum()),
                    label_rows=int(sum(len(r['match']) for r in rows.values())),
                    valid_rows={f'h{h}': int(sum(int((r[f'valid_h{h}'] == 1).sum()) for r in rows.values())) for h in K.HS})
        par = read_exposures(C.EXPOSURES)
        join['parent_exposure_rows'] = int(sum(len(v) for v in par.values()))
        l_by = defaultdict(list)
        for nm in names[:3]:
            R = rows[nm]
            for m, s, L in zip(R['match'].tolist(), R['s'].tolist(), R['L'].tolist()):
                l_by[m].append((int(s), int(L)))
        join['label_sL_equals_parent_exposure_sL_all_matches'] = all(
            sorted(l_by[m]) == sorted((int(t[2]), int(t[3])) for t in par[m]) for m in set(par) | set(l_by))
        join['patch_equal'] = bool(all(
            np.array_equal(A['patch'][rows[nm]['shard_row']].astype(str),
                           np.full(len(rows[nm]['match']), {'MAIN_TRAIN': '15.14', 'MAIN_VALIDATION': '15.15', 'MAIN_TEST': '15.16'}[nm]))
            for nm in names[:3]))
        ok_join = (join['shard_rows_used_once'] == join['shard_rows'] == join['label_rows'] == 566452 and join['valid_rows']['h90'] == 566104
                   and join['parent_exposure_rows'] == 566452 and join['label_sL_equals_parent_exposure_sL_all_matches'] and join['patch_equal'])
        join['ok'] = bool(ok_join)
        st.update('running', 'joined', join=join, next_step='negative-count provenance and TRAIN fixtures')
        if not ok_join:
            raise SystemExit(f'join check failed: {join}')
        # ------------------------------------------------ negative counts: frozen detector re-run on those matches
        neg = (A['cluster_blue'] < 0) | (A['cluster_red'] < 0)
        neg_matches = sorted(set(A['groups'][neg].tolist()))
        det_neg = run_detector(neg_matches, C.CACHE_MAIN, args.workers, st, 'negative_count_matches')
        neg_proof = []
        proven = np.zeros(len(A['groups']), dtype=bool)
        for i in np.flatnonzero(neg):
            m, s = A['groups'][i], int(A['engage_ts'][i])
            rec = det_neg.get(m, {})
            hits = [x for x in rec.get('refs', []) if x['s'] == s]
            L_label = [L for (ss, L) in l_by[m] if ss == s]
            item = dict(match=m, s=s, L_label=L_label[0] if L_label else None, shard_cluster_blue=int(A['cluster_blue'][i]),
                        shard_cluster_red=int(A['cluster_red'][i]), detector_error=rec.get('error'), redetected=len(hits))
            if len(hits) == 1:
                x = hits[0]
                item.update(L_redetected=x['L'], raw_det=x['raw'], ref_cluster_blue=x['ref_cluster_blue'], ref_cluster_red=x['ref_cluster_red'])
                sides_ok = []
                for side in ('blue', 'red'):
                    stored = int(A[f'cluster_{side}'][i])
                    raw = x['raw_int'][f'det_cluster_{side}']
                    if stored < 0:
                        sides_ok.append(raw == 0)
                    else:
                        sides_ok.append(raw == stored)
                item['proven_minus1_is_int_zero'] = bool(all(sides_ok) and x['L'] == item['L_label']
                                                         and x['ref_cluster_blue'] == int(A['cluster_blue'][i])
                                                         and x['ref_cluster_red'] == int(A['cluster_red'][i]))
            else:
                item['proven_minus1_is_int_zero'] = False
            proven[i] = item['proven_minus1_is_int_zero']
            neg_proof.append(item)
        sub_of = {}
        for nm in names[:3]:
            for m, sr in zip(rows[nm]['match'].tolist(), rows[nm]['sub_role'].tolist()):
                sub_of[m] = sr
        for item in neg_proof:
            item['split_role'] = sub_of.get(item['match'])
        C.write_json(out_dir / 'negative_count_provenance.json', dict(
            rule='a stored -1 is read as zero only if the frozen detector re-run on the match yields exactly one engagement at the same s '
                 'with the same L, raw det_cluster value integer 0 on the -1 side, equal raw value on the other side and the same '
                 'int(count or -1) ref value; otherwise unknown',
            code_provenance='gameplay/fights.py sets det_cluster_blue/red = int(len(participants & team)) in the only candidate '
                            'constructor; _merge_adjacent_candidates copies the earlier dict; data/index_split._fight_to_ref_row '
                            'stores int(fight.get(key, -1) or -1), so 0 -> -1',
            note='re-detection reads the cache timeline of these matches only (no outcome, label or V output); counts for all rows '
                 'are already read from the shards', rows=neg_proof, rows_total=int(neg.sum()), matches=len(neg_matches),
            proven_rows=int(proven[neg].sum()), unknown_rows=int((neg & ~proven).sum())))
        st.update('running', 'negative_counts', rows=int(neg.sum()), proven=int(proven[neg].sum()), next_step='TRAIN fixture')
        # ------------------------------------------------ TRAIN-only detector fixture
        train_ids = sorted(set(rows['MAIN_TRAIN']['match'].tolist()))
        man_train = [r['match_id'] for r in csv.DictReader(open(C.MAIN_MANIFEST, encoding='utf-8')) if r['role'] == 'TRAIN']
        fixture = sorted(man_train, key=fixture_rank)[:args.fixture_n]
        det_fix = run_detector(fixture, C.CACHE_MAIN, args.workers, st, 'train_fixture')
        fx = Counter()
        fx_bad = []
        for m in fixture:
            rec = det_fix[m]
            if rec['error']:
                fx['detector_errors'] += 1
                fx_bad.append(dict(match=m, error=rec['error']))
                continue
            same_expo = rec['exposures'] == par.get(m, [])
            fx['matches'] += 1
            fx['exposure_tuple_equal'] += int(same_expo)
            for x in rec['refs']:
                fx['engagements'] += 1
                i = index.get((m, x['s']), -1)
                if i < 0:
                    fx['engagement_not_in_shards'] += 1
                    fx_bad.append(dict(match=m, s=x['s'], reason='not in shards'))
                    continue
                eq = all(int(A[k][i]) == x['ref_' + k] for k in ('cluster_blue', 'cluster_red', 'present_blue', 'present_red'))
                fx['counts_equal_shard'] += int(eq)
                fx['raw_keys_present_int'] += int(all(v is not None for v in x['raw_int'].values()))
                fx['raw_zero_side'] += int(x['raw_int']['det_cluster_blue'] == 0 or x['raw_int']['det_cluster_red'] == 0)
                if not eq:
                    fx_bad.append(dict(match=m, s=x['s'], shard={k: int(A[k][i]) for k in ('cluster_blue', 'cluster_red', 'present_blue', 'present_red')},
                                       redetected={k: x['ref_' + k] for k in ('cluster_blue', 'cluster_red', 'present_blue', 'present_red')}))
            if not same_expo:
                fx_bad.append(dict(match=m, reason='exposure tuples differ', redetected=len(rec['exposures']), parent=len(par.get(m, []))))
        fixture_res = dict(selection="first N TRAIN (15.14) manifest matches by sha256('cr20260915-detector-fixture:'+match_id)",
                           n=len(fixture), fixture_matches_sha256=C.sha256_json(fixture), counts=dict(fx), mismatches=fx_bad[:50],
                           detector_settings=detector_settings(),
                           pass_=bool(fx['matches'] == len(fixture) and fx['exposure_tuple_equal'] == len(fixture)
                                      and fx['counts_equal_shard'] == fx['engagements'] and fx['raw_keys_present_int'] == fx['engagements']))
        C.write_json(out_dir / 'detector_train_fixture.json', fixture_res)
        st.update('running', 'train_fixture', fixture=dict(fx), pass_=fixture_res['pass_'], next_step='external detector')
        if not fixture_res['pass_']:
            raise SystemExit('TRAIN detector fixture failed; stored counts cannot be tied to the frozen detector')
        # ------------------------------------------------ assemble main cohorts
        manifest = dict(version=K.VERSION, role=K.ROLE_TAG, written_at=None, inputs=inputs, join=join,
                        rules=dict(E='every engagement row of the frozen full-corpus labels (valid or not); comparisons use valid_h rows',
                                   participation='v3.3 stored cluster_blue/cluster_red (det_cluster_*): kill participants + in-radius '
                                                 'interaction actors (3,000 u, shop events included); NOT the presence count',
                                   T=f'known & min(cluster_blue, cluster_red) >= {K.TEAMFIGHT_MIN} (manifest scale.teamfight_min)',
                                   N='known & min < 4 (E minus T minus unknown)', fine='pick min <= 1, skirmish 2..3, teamfight >= 4',
                                   negative='-1 -> 0 only when proven by re-detection (negative_count_provenance.json); else unknown',
                                   unknown='excluded from T/N comparisons, retained in pooled E baseline',
                                   diagnostic_cut3_cut5='same counts, cuts 3 and 5 (diagnostic only; main cut stays 4)',
                                   presence='present_blue/red min, diagnostic only',
                                   stale_definition_note='ENGAGEMENT_SCALE_DEFINITION.md (v2, cut3) is stale and not used'),
                        sets={})
        for nm in names[:3]:
            R = rows[nm]
            ix = R['shard_row']
            cb, cr = A['cluster_blue'][ix].astype(np.int64), A['cluster_red'][ix].astype(np.int64)
            pb, pr = A['present_blue'][ix].astype(np.int64), A['present_red'][ix].astype(np.int64)
            nz = proven[ix]
            save_set(out_dir, nm, R, cb, cr, pb, pr, nz, np.full(len(ix), 'v33_shard', dtype='U40'), ix, A['shard'][ix], manifest)
        # ------------------------------------------------ external sets
        ext_checks = {}
        if not args.skip_external:
            for s in K.EXT:
                nm = f'EXT_{s}'
                R = label_keys(nm)
                set_dir = K.FC / 'external' / s
                prep = C.read_json(set_dir / 'prepared_manifest.json')
                mids = [r['match_id'] for r in prep['matches'] if r['cache_status'] == 'written']
                det = run_detector(mids, set_dir / 'cache', args.workers, st, nm)
                ext_par = read_exposures(set_dir / 'exposures.csv')
                errors = {m: d['error'] for m, d in det.items() if d['error']}
                expo_equal = sum(int(det[m]['exposures'] == ext_par.get(m, [])) for m in mids)
                lookup = {}
                dup = 0
                for m in mids:
                    for x in det[m]['refs']:
                        if (m, x['s']) in lookup:
                            dup += 1
                        lookup[(m, x['s'])] = x
                n = len(R['match'])
                cb, cr, pb, pr = (np.full(n, -99, dtype=np.int64) for _ in range(4))
                nz = np.zeros(n, dtype=bool)
                miss, L_mismatch = 0, 0
                raw_nonint = 0
                for j, (m, s_, L_) in enumerate(zip(R['match'].tolist(), R['s'].tolist(), R['L'].tolist())):
                    x = lookup.get((m, int(s_)))
                    if x is None:
                        miss += 1
                        continue
                    if x['L'] != int(L_):
                        L_mismatch += 1
                        continue
                    cb[j], cr[j], pb[j], pr[j] = x['ref_cluster_blue'], x['ref_cluster_red'], x['ref_present_blue'], x['ref_present_red']
                    rb, rr = x['raw_int']['det_cluster_blue'], x['raw_int']['det_cluster_red']
                    raw_nonint += int(rb is None or rr is None)
                    nz[j] = (rb is not None and rr is not None and (cb[j] >= 0 or rb == 0) and (cr[j] >= 0 or rr == 0))
                label_expo = sorted((m, int(s_), int(L_)) for m, s_, L_ in zip(R['match'].tolist(), R['s'].tolist(), R['L'].tolist()))
                det_expo = sorted((m, int(t[2]), int(t[3])) for m in mids for t in det[m]['exposures'])
                ext_checks[nm] = dict(matches=len(mids), detector_errors=len(errors), error_examples=dict(list(errors.items())[:5]),
                                      exposure_tuples_equal_matches=expo_equal, duplicate_s_keys=dup, label_rows=n,
                                      label_rows_without_redetected=miss, label_rows_L_mismatch=L_mismatch,
                                      label_keys_equal_redetected_exposure_keys=label_expo == det_expo,
                                      rows_with_raw_count_not_int=raw_nonint, rows_negative_count=int(((cb < 0) | (cr < 0)).sum()),
                                      rows_negative_count_proven_zero=int((((cb < 0) | (cr < 0)) & nz).sum()),
                                      exposures_csv_sha256=C.sha256_file(set_dir / 'exposures.csv'),
                                      prepared_manifest_sha256=C.sha256_file(set_dir / 'prepared_manifest.json'))
                ok = (not errors and expo_equal == len(mids) and dup == 0 and miss == 0 and L_mismatch == 0
                      and ext_checks[nm]['label_keys_equal_redetected_exposure_keys'] and raw_nonint == 0)
                ext_checks[nm]['pass_'] = bool(ok)
                st.update('running', f'external_{nm}', check=ext_checks[nm], next_step='next set')
                if not ok:
                    raise SystemExit(f'{nm}: external re-detection check failed {ext_checks[nm]}')
                save_set(out_dir, nm, R, cb, cr, pb, pr, nz | ((cb >= 0) & (cr >= 0)),
                         np.full(n, 'frozen_detector_rerun_on_adapted_cache', dtype='U40'), np.full(n, -1), np.full(n, -1), manifest)
        manifest['external_checks'] = ext_checks
        manifest['train_detector_fixture'] = dict(pass_=fixture_res['pass_'], counts=fixture_res['counts'])
        manifest['negative_counts'] = dict(main_rows=int(neg.sum()), main_proven_zero=int(proven[neg].sum()))
        manifest['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        manifest['statement'] = ('Written before any cohort/role/arm model fitting. Contains keys, counts, sources and rules only; '
                                 'no generated label, delta, V output or outcome was read by this stage.')
        msha = C.write_json(out_dir / 'cohort_manifest.json', manifest)
        st.update('complete', 'cohorts', cohort_manifest_sha256=msha, next_step='contract tests, then specialist q fits')
        return 0
    except SystemExit as exc:
        st.update('failed', 'cohorts', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'cohorts', error=repr(exc), next_step='inspect, fix, rerun')
        return 3


def save_set(out_dir, nm, R, cb, cr, pb, pr, neg_zero, source, shard_row, shard_file, manifest):
    n_min, known, cohort, fine = K.scale_classes(cb, cr, neg_zero)
    pres_min = np.minimum(pb, pr)
    cut = {}
    for c in (3, 5):
        cut[c] = np.where(~known, -1, (n_min >= c).astype(np.int64))
    arrays = dict(match=R['match'], s=R['s'], L=R['L'], sub_role=R['sub_role'], valid_h60=R['valid_h60'], valid_h90=R['valid_h90'],
                  valid_h120=R['valid_h120'], cluster_blue=cb, cluster_red=cr, present_blue=pb, present_red=pr,
                  negative_read_as_zero=neg_zero & ((cb < 0) | (cr < 0)), n_min=n_min, scale_known=known.astype(np.int8),
                  cohort=cohort.astype(np.int8), fine=fine.astype(np.int8), teamfight_cut3_DIAG=cut[3].astype(np.int8),
                  teamfight_cut5_DIAG=cut[5].astype(np.int8), presence_n_min_DIAG=pres_min, source=source,
                  shard_row=np.asarray(shard_row), shard_file_index=np.asarray(shard_file),
                  role=np.asarray('COHORT_MEMBERSHIP_POST_CUTOFF_NOT_A_PREDICTOR'))
    sha = C.save_npz(out_dir / f'{nm}_cohort.npz', **arrays)
    T = cohort == 1
    N = cohort == 0
    U = cohort == -1
    by = {}
    for sr in sorted(set(R['sub_role'].tolist())):
        m = R['sub_role'] == sr
        cell = {}
        for h in K.HS:
            v = m & (R[f'valid_h{h}'] == 1)
            cell[f'h{h}'] = dict(valid=int(v.sum()), T=int((v & T).sum()), N=int((v & N).sum()), unknown=int((v & U).sum()),
                                 T_matches=int(len(np.unique(R['match'][v & T]))), N_matches=int(len(np.unique(R['match'][v & N]))),
                                 pick=int((v & (fine == 0)).sum()), skirmish=int((v & (fine == 1)).sum()))
        cell['all_rows'] = dict(rows=int(m.sum()), T=int((m & T).sum()), N=int((m & N).sum()), unknown=int((m & U).sum()))
        by[sr] = cell
    joint = np.zeros((7, 7), dtype=np.int64)
    ok = (cb >= -1) & (cr >= -1)
    np.add.at(joint, (np.clip(cb[ok], -1, 5) + 1, np.clip(cr[ok], -1, 5) + 1), 1)
    manifest['sets'][nm] = dict(
        file=f'cohorts/{nm}_cohort.npz', sha256=sha, rows=int(len(cb)), matches=int(len(np.unique(R['match']))),
        T=int(T.sum()), N=int(N.sum()), unknown=int(U.sum()), T_and_N_intersection=int((T & N).sum()),
        T_union_N_equals_known=bool(np.array_equal(T | N, known)), pick=int((fine == 0).sum()), skirmish=int((fine == 1).sum()),
        teamfight=int((fine == 2).sum()), cut3_teamfight_DIAG=int((cut[3] == 1).sum()), cut5_teamfight_DIAG=int((cut[5] == 1).sum()),
        presence_n_min_ge4_DIAG=int((pres_min >= 4).sum()), negative_count_rows=int(((cb < 0) | (cr < 0)).sum()),
        negative_read_as_zero_rows=int((neg_zero & ((cb < 0) | (cr < 0))).sum()), by_sub_role=by,
        joint_counts_rows_blue_minus1_to_5_cols_red_minus1_to_5=joint.tolist(),
        source=sorted(set(source.tolist())))


if __name__ == '__main__':
    sys.exit(main())
