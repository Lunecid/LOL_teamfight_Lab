"""Label validity contract: TRAIN raw-cache fixtures (event interval counting, kill credit, frame/normalizer trace).

Deterministic 40 TRAIN matches (lowest sha256('lv20260915-fixture:'+match) among matches with a valid h90 row).
Raw cache files of those matches only are read (sha256 before/after). Independent recount from raw events.json is
compared with the parent stored counters and StateV2 state differences; cache normalizers are traced against
gold_team_minute (raw team gold) and integer grids. TRAIN only; no outcome W is read.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import lv20260915_analysis as A  # noqa: E402
import lv20260915_common as K  # noqa: E402

import json  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime')
os.environ['LOL_CFG_PRESET'] = 'v3.3'
os.environ['LOL_CFG_OVERRIDES'] = json.dumps({'CACHE_DIRNAME': str(C.CACHE_MAIN), 'FIGHT_INDEX_CACHE_ENABLED': False,
                                              'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False, 'CACHE_IN_RAM': False})
sys.path.insert(0, str(C.WT))

import hashlib  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

import engagement_labels_v3_rules as R  # noqa: E402

N_FIX = 40
DEN = {'totalGold_norm': 25000.0, 'curGold_norm': 4000.0, 'xp_norm': 20000.0, 'level_norm': 18.0, 'laneCS_norm': 400.0, 'jgCS_norm': 250.0}
CONTROLS = {'totalGold_norm': (24000.0, 26000.0), 'xp_norm': (19000.0, 21000.0), 'level_norm': (17.0, 19.0)}


def near_int_fraction(v, den, tol=0.02):
    v = np.asarray(v, dtype=np.float64)
    v = v[v > 0]
    if not len(v):
        return None
    x = v * den
    return float(np.mean(np.abs(x - np.round(x)) <= tol))


def main():
    st = K.Status('contracts_train_fixtures')
    K.log_command()
    try:
        P = K.ParentReadOnly()
        lab = P.label_file('MAIN_TRAIN', 'TRAIN raw fixture contract (counters, kills, frames)',
                           keys=['match', 's', 'L', 'q_pre', 'sub_role', 'pre_snapshot', 'during_counts', 'count_keys']
                           + [f'{k}_h{h}' for h in K.HS for k in ('endpoint', 'valid', 'post_snapshot', 'after_counts', 'after_raw_kills')])
        if not np.all(np.char.startswith(lab['sub_role'].astype(str), 'fold')):
            raise SystemExit('MAIN_TRAIN label file contains non-TRAIN rows')
        keys = [str(k) for k in lab['count_keys']]
        if keys != list(R.COUNT_KEYS):
            raise SystemExit('count key order differs from rules module')
        vm = sorted(set(lab['match'][lab['valid_h90'] == 1].tolist()), key=lambda m: hashlib.sha256(f'lv20260915-fixture:{m}'.encode()).hexdigest())
        fixtures = vm[:N_FIX]
        fset = set(fixtures)
        st.update('running', 'load_states', fixtures=len(fixtures), next_step='raw recount')
        S = {}
        names = None
        for sp, _op, cm in P.chunk_paths('MAIN'):
            with np.load(sp, allow_pickle=False) as z:
                em = z['e_match']
                sel = np.isin(em, fixtures)
                if not sel.any():
                    continue
                names = z['names'].tolist()
                if not np.all(np.char.startswith(z['m_sub_role'][np.isin(z['m_match'], fixtures)].astype(str), 'fold')):
                    raise SystemExit('fixture match not TRAIN')
                part = dict(match=em[sel], s=z['e_s'][sel], X_pre=z['e_X_pre'][sel])
                for h in K.HS:
                    part[f'X_post_h{h}'] = z[f'e_X_post_h{h}'][sel]
                for k, v in part.items():
                    S.setdefault(k, []).append(v)
        S = {k: np.concatenate(v) for k, v in S.items()}
        # join chunk states to label rows by exact key
        lk = {(m, int(s)): i for i, (m, s) in enumerate(zip(lab['match'].tolist(), lab['s'].tolist())) if m in fset}
        if len(lk) != len(S['match']):
            raise SystemExit(f'fixture key count differs: labels {len(lk)} states {len(S["match"])}')
        order = np.asarray([lk[(m, int(s))] for m, s in zip(S['match'].tolist(), S['s'].tolist())])
        ix = {n: i for i, n in enumerate(names)}

        from core.config import NODE_FEATURE_NAMES
        nn = list(NODE_FEATURE_NAMES)
        raw_hash_before, raw_hash_after = {}, {}
        res = dict(fixtures=len(fixtures), fixture_match_ids_sha256=C.sha256_json(fixtures), engagement_rows=int(len(order)),
                   counter_mismatch_rows={'during': 0}, counter_mismatch_examples=[], kill_credit_mismatch_rows=0,
                   kill_during_vs_full_mismatch_rows=0, raw_kills_after_L_nonzero_valid_rows=0, first_kill_missing_rows=0,
                   last_kill_missing_rows=0, pre_frame_mismatch_rows=0, post_frame_mismatch_rows=0,
                   state_node_value_mismatch=0, element_team_state_vs_raw_mismatch_rows=0,
                   counter_team_vs_state_team_attribution_note='counters use killerTeamId only; StateV2 uses killerTeamId or the killer roster team',
                   gold_team_minute=dict(frames=0, max_abs_diff_gold=0.0, max_rel_diff=0.0), integer_grid={}, rows_checked_valid={})
        for h in K.HS:
            res['counter_mismatch_rows'][f'after_h{h}'] = 0
            res['rows_checked_valid'][f'h{h}'] = 0
        grid_vals = {k: [] for k in DEN}
        for n_done, mid in enumerate(fixtures):
            paths = [C.CACHE_MAIN / f'{mid}{sfx}' for sfx in ('.npz', '.events.json', '.meta.json')]
            for p in paths:
                raw_hash_before[p.name] = C.sha256_file(p)
            events = json.loads(paths[1].read_text(encoding='utf-8'))
            meta = json.loads(paths[2].read_text(encoding='utf-8'))
            with np.load(paths[0], allow_pickle=False) as z:
                ts = z['minute_ts'].astype(np.int64)
                node = z['node_minute'].astype(np.float64)
                gtm = z['gold_team_minute'].astype(np.float64)
            tm = {int(k): int(v) for k, v in meta['team_map'].items()}
            order_pid = sorted(tm, key=lambda p: (tm[p], p))
            rows = A.raw_event_rows(events)
            kill_ts = sorted(t for t, typ, _e in rows if typ == 'CHAMPION_KILL')
            if node.shape[2] != len(nn):
                raise SystemExit('node feature count differs from NODE_FEATURE_NAMES')
            gi = nn.index('totalGold_norm')
            for team, col in ((100, 0), (200, 1)):
                pids = [p for p in tm if tm[p] == team]
                s_norm = node[:, [p - 1 for p in pids], gi].sum(axis=1) * DEN['totalGold_norm']
                d = np.abs(s_norm - gtm[:, col])
                res['gold_team_minute']['frames'] += len(d)
                res['gold_team_minute']['max_abs_diff_gold'] = max(res['gold_team_minute']['max_abs_diff_gold'], float(d.max()))
                rel = d / np.maximum(1.0, gtm[:, col])
                res['gold_team_minute']['max_rel_diff'] = max(res['gold_team_minute']['max_rel_diff'], float(rel.max()))
            for k in DEN:
                grid_vals[k].append(node[:, :, nn.index(k)].ravel())
            for j in np.flatnonzero(S['match'] == mid):
                i = order[j]
                q_pre, L, s = int(lab['q_pre'][i]), int(lab['L'][i]), int(lab['s'][i])
                if kill_ts.count(s + 15000) == 0:
                    res['first_kill_missing_rows'] += 1
                if kill_ts.count(L) == 0:
                    res['last_kill_missing_rows'] += 1
                dur = A.raw_interval_counts(rows, tm, q_pre, L)
                if any(dur.get(k, 0) != int(lab['during_counts'][i][kk]) for kk, k in enumerate(keys)):
                    res['counter_mismatch_rows']['during'] += 1
                    if len(res['counter_mismatch_examples']) < 5:
                        res['counter_mismatch_examples'].append(dict(window='during', row=int(i), raw={k: dur.get(k, 0) for k in keys},
                                                                     stored=lab['during_counts'][i].tolist()))
                pre_frame = int(ts[np.searchsorted(ts, q_pre, side='right') - 1])
                if pre_frame != int(lab['pre_snapshot'][i]):
                    res['pre_frame_mismatch_rows'] += 1
                fi = int(np.searchsorted(ts, q_pre, side='right') - 1)
                for slot, pid in enumerate(order_pid):
                    for f in ('totalGold_norm', 'xp_norm'):
                        if S['X_pre'][j, ix[f'participant_slot{slot}_{f}']] != node[fi, pid - 1, nn.index(f)]:
                            res['state_node_value_mismatch'] += 1
                for h in K.HS:
                    if int(lab[f'valid_h{h}'][i]) != 1:
                        continue
                    res['rows_checked_valid'][f'h{h}'] += 1
                    e = int(lab[f'endpoint_h{h}'][i])
                    aft = A.raw_interval_counts(rows, tm, L, e)
                    if any(aft.get(k, 0) != int(lab[f'after_counts_h{h}'][i][kk]) for kk, k in enumerate(keys)):
                        res['counter_mismatch_rows'][f'after_h{h}'] += 1
                        if len(res['counter_mismatch_examples']) < 5:
                            res['counter_mismatch_examples'].append(dict(window=f'after_h{h}', row=int(i)))
                    if aft.get('champion_kill', 0) != 0 or int(lab[f'after_raw_kills_h{h}'][i]) != 0:
                        res['raw_kills_after_L_nonzero_valid_rows'] += 1
                    full = A.raw_interval_counts(rows, tm, q_pre, e)
                    ob = A.observed_from_states(names, S['X_pre'][j:j + 1], S[f'X_post_h{h}'][j:j + 1])
                    if (full.get('credited_kill_blue', 0) != ob['kills_blue'][0] or full.get('credited_kill_red', 0) != ob['kills_red'][0]):
                        res['kill_credit_mismatch_rows'] += 1
                    if (dur.get('credited_kill_blue', 0), dur.get('credited_kill_red', 0)) != (full.get('credited_kill_blue', 0), full.get('credited_kill_red', 0)):
                        res['kill_during_vs_full_mismatch_rows'] += 1
                    post_frame = int(ts[np.searchsorted(ts, e, side='right') - 1])
                    if post_frame != int(lab[f'post_snapshot_h{h}'][i]):
                        res['post_frame_mismatch_rows'] += 1
                    # per-element team from raw with the StateV2 team rule (killerTeamId or killer roster team)
                    mism = False
                    for d in A.DRAGON_ELEMENTS:
                        cnt = {100: 0, 200: 0}
                        for t, typ, ev in rows:
                            if q_pre < t <= e and typ == 'ELITE_MONSTER_KILL' and str(ev.get('monsterType')) == 'DRAGON':
                                sub = str(ev.get('monsterSubType', '')).upper()
                                if sub == 'ELDER_DRAGON':
                                    continue
                                el = sub[:-7] if sub.endswith('_DRAGON') else sub
                                el = el if el in A.DRAGON_ELEMENTS else 'OTHER'
                                team = A._int(ev.get('killerTeamId')) or tm.get(A._int(ev.get('killerId')), 0)
                                if el == d and team in cnt:
                                    cnt[team] += 1
                        if cnt[100] != ob[f'state_dragon_{d}_blue'][0] or cnt[200] != ob[f'state_dragon_{d}_red'][0]:
                            mism = True
                    res['element_team_state_vs_raw_mismatch_rows'] += int(mism)
            for p in paths:
                raw_hash_after[p.name] = C.sha256_file(p)
            if n_done % 10 == 9:
                st.update('running', 'raw_recount', processed=n_done + 1, total=len(fixtures))
        for k, den in DEN.items():
            v = np.concatenate(grid_vals[k])
            res['integer_grid'][k] = dict(declared_denominator=den, near_integer_fraction=near_int_fraction(v, den),
                                          controls={str(c): near_int_fraction(v, c) for c in CONTROLS.get(k, ())},
                                          max_value=float(v.max()), clip_upper=5.0 if k != 'level_norm' else 2.0,
                                          values_at_clip=int(np.sum(v >= (5.0 if k != 'level_norm' else 2.0))))
        res['raw_files_unchanged'] = raw_hash_before == raw_hash_after
        res['raw_files_hashed'] = len(raw_hash_before)
        res['normalizer_source'] = dict(file=str(C.REPO / 'gameplay' / 'pipeline_cache.py'),
                                        sha256=C.sha256_file(C.REPO / 'gameplay' / 'pipeline_cache.py'),
                                        constants=dict(DEN_TOT_G=25000.0, DEN_XP=20000.0, DEN_CUR_G=4000.0, DEN_LEVEL=18.0,
                                                       DEN_LANECS=400.0, DEN_JGCS=250.0),
                                        formula='float32(clip(raw / DEN, 0, 5)); level clip [0, 2]; gold_team_minute = raw totalGold team sum')
        gt = res['gold_team_minute']
        res['checks'] = dict(
            during_counters_exact=res['counter_mismatch_rows']['during'] == 0,
            after_counters_exact=all(res['counter_mismatch_rows'][f'after_h{h}'] == 0 for h in K.HS),
            credited_kills_equal_state_difference=res['kill_credit_mismatch_rows'] == 0,
            kills_during_equal_full_interval=res['kill_during_vs_full_mismatch_rows'] == 0,
            no_raw_kill_after_L_in_valid_window=res['raw_kills_after_L_nonzero_valid_rows'] == 0,
            first_and_last_kill_present=res['first_kill_missing_rows'] == 0 and res['last_kill_missing_rows'] == 0,
            frames_exact=res['pre_frame_mismatch_rows'] == 0 and res['post_frame_mismatch_rows'] == 0,
            state_equals_node_cache=res['state_node_value_mismatch'] == 0,
            dragon_element_team_state_equals_raw=res['element_team_state_vs_raw_mismatch_rows'] == 0,
            gold_normalizer_25000_reproduces_gold_team_minute=gt['max_abs_diff_gold'] <= 2.0,
            xp_grid_20000=(res['integer_grid']['xp_norm']['near_integer_fraction'] or 0) > .99
            and all((x or 0) < .5 for x in res['integer_grid']['xp_norm']['controls'].values()),
            raw_files_unchanged=res['raw_files_unchanged'])
        res['all_pass'] = all(res['checks'].values())
        res['runtime_dirs_created'] = sorted(str(p.relative_to(K.OUT)) for p in (K.OUT / 'runtime').rglob('*')) if (K.OUT / 'runtime').exists() else []
        res['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        C.write_json(K.OUT / 'contracts' / 'train_fixture_checks.json', res)
        st.update('complete' if res['all_pass'] else 'failed', 'train_fixtures', checks=res['checks'],
                  error=None if res['all_pass'] else 'fixture check failed (see contracts/train_fixture_checks.json)',
                  next_step='TRAIN-only smoke fits')
        return 0 if res['all_pass'] else 2
    except SystemExit as exc:
        st.update('failed', 'train_fixtures', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'train_fixtures', error=repr(exc), next_step='fix and rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
