"""Stage B0: pregame draft fields for every match, and weak role supervision provenance (15.14 TRAIN only).

Draft (all main 210,000 matches and all written external caches): per StateV2 slot (team then participant ID) the
participant ID, team, champion ID (meta.static_meta.champion_by_pid) and the two selected summoner spells
(meta.static_meta.summoner_spells_by_pid, unordered). Missing values are 0 and flagged.

Role supervision: the original raw match-detail files of the main corpus are not on disk under the project-configured
paths (core/config.py DETAIL_DIR default <repo>/data/raw/matches/kr/detail does not exist; docs/DATA_AVAILABILITY.md
records the cache as the sole copy). Therefore the cached meta.role_slots of 15.14 TRAIN matches are used ONLY as an
explicitly WEAK role annotation: core/roles.get_role_slots_from_detail maps teamPosition (or individualPosition) to
slots and silently fills unmatched participants by participant ID. JSON key order of role_slots preserves the
assignment order (role-matched pids in role order, then ID-filled pids), so a team whose slot sequence in key order is
not strictly increasing certainly used the ID fill: that team is excluded from role supervision (not from q cohorts).
Whole-team or trailing-slot fills are undetectable and remain a disclosed weakness. role_slots of VALIDATION/TEST
matches and of external caches are never read here. The raw 15.14 detail files that do exist outside the project
(2 matches, user Downloads) are audited for teamPosition agreement; they are not supervision.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

ROLE_ORDER_REPO = ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')


def parse_meta(path, read_roles):
    """-> dict with slot arrays; role_slots parsed with key order only if read_roles."""
    meta = json.loads(Path(path).read_text(encoding='utf-8'))
    rec = dict(roster_ok=0, pids=[0] * 10, team=[0] * 10, champ=[0] * 10, spell_a=[0] * 10, spell_b=[0] * 10, patch=str(meta.get('patch', '')),
               role_label=[-1] * 10, team_supervision_ok=[0, 0], team_reason=['not_read', 'not_read'])
    try:
        tm = {int(k): int(v) for k, v in (meta.get('team_map') or {}).items()}
    except (TypeError, ValueError):
        tm = {}
    if set(tm) != set(range(1, 11)) or any(list(tm.values()).count(t) != 5 for t in (100, 200)):
        rec['team_reason'] = ['invalid_roster', 'invalid_roster']
        return rec
    rec['roster_ok'] = 1
    order = sorted(tm, key=lambda pid: (tm[pid], pid))
    sm = meta.get('static_meta') or {}
    cb = sm.get('champion_by_pid') or {}
    sb = sm.get('summoner_spells_by_pid') or {}
    for i, pid in enumerate(order):
        rec['pids'][i] = pid
        rec['team'][i] = tm[pid]
        try:
            rec['champ'][i] = max(0, int(cb.get(str(pid), 0) or 0))
        except (TypeError, ValueError):
            rec['champ'][i] = 0
        sp = sb.get(str(pid)) or {}
        for key, dst in (('summoner_spell_1_id', 'spell_a'), ('summoner_spell_2_id', 'spell_b')):
            try:
                rec[dst][i] = max(0, int(sp.get(key, 0) or 0))
            except (TypeError, ValueError):
                rec[dst][i] = 0
    if not read_roles:
        return rec
    rs = meta.get('role_slots')
    if not isinstance(rs, dict) or len(rs) != 10:
        rec['team_reason'] = ['role_slots_missing_or_incomplete'] * 2
        return rec
    try:
        seq = [(int(k), int(v)) for k, v in rs.items()]
    except (TypeError, ValueError):
        rec['team_reason'] = ['role_slots_not_integer'] * 2
        return rec
    slot_of = dict(seq)
    teams_seen = [tm.get(p) for p, _ in seq]
    grouped = teams_seen == sorted(teams_seen, key=lambda t: (t != 100,))  # team 100 entries precede team 200 entries
    for ti, (tid, off) in enumerate(((100, 0), (200, 5))):
        pids_t = [p for p, _ in seq if tm.get(p) == tid]
        slots_t = [s for p, s in seq if tm.get(p) == tid]
        if len(pids_t) != 5 or sorted(slots_t) != list(range(off, off + 5)):
            rec['team_reason'][ti] = 'invalid_slot_block_or_duplicate'
            continue
        if not grouped:
            rec['team_reason'][ti] = 'key_order_not_grouped_by_team_(repo_final_pid_fallback)'
            continue
        if any(b <= a for a, b in zip(slots_t, slots_t[1:])):
            rec['team_reason'][ti] = 'detectable_participant_id_fill'
            continue
        rec['team_supervision_ok'][ti] = 1
        rec['team_reason'][ti] = 'weak_annotation_in_order_(ID_fill_not_detectable)'
        for i, pid in enumerate(order):
            if tm[pid] == tid:
                rec['role_label'][i] = slot_of[pid] - off
    return rec


def work(task):
    out = []
    for mid, path, read_roles in task:
        try:
            r = parse_meta(path, read_roles)
            r['error'] = ''
        except Exception as exc:
            r = dict(roster_ok=0, pids=[0] * 10, team=[0] * 10, champ=[0] * 10, spell_a=[0] * 10, spell_b=[0] * 10, patch='',
                     role_label=[-1] * 10, team_supervision_ok=[0, 0], team_reason=['meta_error'] * 2, error=f'{type(exc).__name__}:{exc}'[:200])
        r['match'] = mid
        out.append(r)
    return out


def run_pool(items, workers, st, label):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    chunks = [items[i:i + 2000] for i in range(0, len(items), 2000)]
    res = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
        for part in ex.map(work, chunks):
            res.extend(part)
            st.update('running', f'draft_{label}', processed=len(res), total=len(items), next_step='assemble')
    return res


def to_arrays(res, roles_col):
    A = dict(match=np.asarray([r['match'] for r in res]), roster_ok=np.asarray([r['roster_ok'] for r in res], dtype=np.int8),
             pid=np.asarray([r['pids'] for r in res], dtype=np.int16), team=np.asarray([r['team'] for r in res], dtype=np.int16),
             champion_id=np.asarray([r['champ'] for r in res], dtype=np.int64), spell_a=np.asarray([r['spell_a'] for r in res], dtype=np.int64),
             spell_b=np.asarray([r['spell_b'] for r in res], dtype=np.int64), meta_patch=np.asarray([r['patch'] for r in res]),
             meta_error=np.asarray([r['error'] for r in res]))
    if roles_col:
        A['weak_role_label'] = np.asarray([r['role_label'] for r in res], dtype=np.int8)
        A['team_supervision_ok'] = np.asarray([r['team_supervision_ok'] for r in res], dtype=np.int8)
        A['team_reason'] = np.asarray([r['team_reason'] for r in res])
    return A


def audit_downloads():
    """Raw 15.14 detail files that exist outside the project paths: teamPosition agreement with cached role_slots."""
    base = Path(os.path.expanduser('~')) / 'Downloads'
    found = []
    main_ids = {r['match_id']: r for r in csv.DictReader(open(C.MAIN_MANIFEST, encoding='utf-8'))}
    seen = set()
    for p in sorted(base.glob('KR_77*.json')):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(d, dict) or 'info' not in d or not isinstance(d['info'].get('participants'), list):
            continue
        if not all(isinstance(q, dict) and 'teamId' in q and 'championId' in q for q in d['info']['participants']):
            continue  # timeline files also carry info.participants (participantId/puuid only)
        mid = (d.get('metadata') or {}).get('matchId')
        if mid not in main_ids:
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if (mid, digest) in seen:
            found.append(dict(file=p.name, match=mid, sha256=digest, duplicate_of_previous=True))
            continue
        seen.add((mid, digest))
        meta = json.loads((C.CACHE_MAIN / f'{mid}.meta.json').read_text(encoding='utf-8'))
        parts = d['info']['participants']
        rows, agree, ind_fallback = [], 0, 0
        sm = meta['static_meta']
        for q in parts:
            pid = int(q['participantId'])
            tp, ip = q.get('teamPosition') or '', q.get('individualPosition') or ''
            ind_fallback += int(not tp)
            slot = int(meta['role_slots'][str(pid)])
            role_slot = ROLE_ORDER_REPO[slot % 5]
            agree += int(tp == role_slot)
            rows.append(dict(pid=pid, team=int(q['teamId']), teamPosition=tp, individualPosition=ip, cached_slot_role=role_slot,
                             champion_equal_cache=int(q['championId']) == int(sm['champion_by_pid'][str(pid)]),
                             spells_equal_cache=sorted([int(q['summoner1Id']), int(q['summoner2Id'])]) ==
                             sorted([int(sm['summoner_spells_by_pid'][str(pid)]['summoner_spell_1_id']),
                                     int(sm['summoner_spells_by_pid'][str(pid)]['summoner_spell_2_id'])])))
        found.append(dict(file=p.name, match=mid, sha256=digest, manifest_role=main_ids[mid]['role'], gameVersion=d['info'].get('gameVersion'),
                          participants=rows, teamPosition_equals_cached_role_slots=agree, participants_n=len(parts),
                          teamPosition_empty=ind_fallback))
    return dict(location='user Downloads folder (outside project-configured raw paths); identity fields not persisted', files=found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    st = K.Status('draft')
    out_dir = K.OUT / 'draft'
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if (K.OUT / 'role_models').exists():
            raise SystemExit('role models already exist; provenance must be written before fitting')
        rows = list(csv.DictReader(open(C.MAIN_MANIFEST, encoding='utf-8')))
        items = [(r['match_id'], str(C.CACHE_MAIN / f"{r['match_id']}.meta.json"), r['role'] == 'TRAIN') for r in rows]
        role_of = {r['match_id']: r['role'] for r in rows}
        res = run_pool(items, args.workers, st, 'MAIN')
        by = {r['match']: r for r in res}
        res = [by[r['match_id']] for r in rows]
        A = to_arrays(res, roles_col=True)
        A['split_role'] = np.asarray([role_of[m] for m in A['match'].tolist()])
        A['sub_role'] = np.asarray([('fold%d' % C.train_fold(m)) if role_of[m] == 'TRAIN' else (C.validation_role(m) if role_of[m] == 'VALIDATION' else role_of[m])
                                    for m in A['match'].tolist()])
        tr = A['split_role'] == 'TRAIN'
        if (A['weak_role_label'][~tr] != -1).any() or not np.all(A['team_reason'][~tr] == 'not_read'):
            raise SystemExit('role annotations touched outside TRAIN')
        A['role'] = np.asarray('DRAFT_FIELDS_PREGAME; weak_role_label = cached role_slots of 15.14 TRAIN only (TARGET ONLY, never a predictor)')
        sha_main = C.save_npz(out_dir / 'MAIN_draft.npz', **A)
        ext = {}
        for s in K.EXT:
            prep = C.read_json(K.FC / 'external' / s / 'prepared_manifest.json')
            mids = [r['match_id'] for r in prep['matches'] if r['cache_status'] == 'written']
            items = [(m, str(K.FC / 'external' / s / 'cache' / f'{m}.meta.json'), False) for m in mids]
            rr = run_pool(items, args.workers, st, s)
            byx = {r['match']: r for r in rr}
            Ax = to_arrays([byx[m] for m in mids], roles_col=False)
            Ax['role'] = np.asarray('DRAFT_FIELDS_PREGAME_EXTERNAL (role_slots not read)')
            ext[s] = dict(file=f'draft/EXT_{s}_draft.npz', sha256=C.save_npz(out_dir / f'EXT_{s}_draft.npz', **Ax), matches=len(mids),
                          roster_invalid=int((Ax['roster_ok'] == 0).sum()), meta_errors=int((Ax['meta_error'] != '').sum()),
                          champion_missing_participants=int((Ax['champion_id'] <= 0).sum()),
                          spell_missing_participants=int(((Ax['spell_a'] <= 0) | (Ax['spell_b'] <= 0)).sum()))
        # provenance summary (TRAIN)
        reasons = Counter(A['team_reason'][tr].ravel().tolist())
        ok_teams = int(A['team_supervision_ok'][tr].sum())
        per_role = Counter(A['weak_role_label'][tr][A['weak_role_label'][tr] >= 0].tolist())
        identity = 0
        for i in np.flatnonzero(tr):
            lab = A['weak_role_label'][i]
            identity += int(all(lab[j] == (j % 5) for j in range(10)))
        prov = dict(
            version=K.VERSION, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            source='cached meta.role_slots of 15.14 TRAIN matches in D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13',
            strength='WEAK role annotation (proxy), not human truth and not raw teamPosition',
            raw_detail_availability=dict(project_configured_detail_dir='C:/Users/todtj/PycharmProjects/LOL_teamfight/data/raw/matches/kr/detail (core/config.py default; LOL_DETAIL_DIR unset)',
                                         exists=Path('C:/Users/todtj/PycharmProjects/LOL_teamfight/data/raw/matches/kr/detail').exists(),
                                         legacy_default='C:/Users/todtj/PycharmProjects/Lol_project/data/raw/matches/kr/detail holds a later 15.18-15.22 collection with 0 main-corpus IDs (search report)',
                                         docs='docs/DATA_AVAILABILITY.md / DATA_MANIFEST.md state the cache is the sole copy of the main corpus'),
            generator='core/roles.py get_role_slots_from_detail: role = teamPosition or individualPosition; lowest pid per role gets the slot; '
                      'unfilled slots get remaining pids in ID order (silent fill); invalid participants -> {pid: pid-1}',
            individualPosition_note='when teamPosition is empty Riot reports individualPosition "Invalid" (truthy), which matches no role, '
                                    'so the individualPosition fallback does not rescue a missing teamPosition in practice',
            exclusion_rules=dict(invalid_roster='team_map not 5+5 over pids 1..10', invalid_slot_block_or_duplicate='team slots not a permutation of its block',
                                 key_order_not_grouped_by_team='repo final pid-1 fallback used', detectable_participant_id_fill='team slot sequence in JSON key order not strictly increasing'),
            undetectable='whole-team ID fill and fills whose slots all follow the role-matched slots cannot be detected; they remain in supervision',
            train_matches=int(tr.sum()), train_teams=int(2 * tr.sum()), teams_used=ok_teams, team_reasons=dict(reasons),
            participants_used=int((A['weak_role_label'][tr] >= 0).sum()), role_label_counts={K.ROLES[k]: int(v) for k, v in sorted(per_role.items())},
            train_matches_with_identity_slot_order=identity,
            available_raw_audit=audit_downloads(),
            not_read=['role_slots of 15.15 VALIDATION / 15.16 TEST matches', 'external role_slots / raw teamPosition (post-freeze diagnostics only)'],
            statement='Provenance written before any role model is fitted. Agreement of role predictions with this proxy is not validated role accuracy.')
        prov_sha = C.write_json(K.OUT / 'role_supervision_provenance.json', prov)
        man = dict(version=K.VERSION, written_at=time.strftime('%Y-%m-%d %H:%M:%S'), main=dict(file='draft/MAIN_draft.npz', sha256=sha_main,
                   matches=int(len(A['match'])), roster_invalid=int((A['roster_ok'] == 0).sum()), meta_errors=int((A['meta_error'] != '').sum()),
                   champion_missing_participants=int((A['champion_id'] <= 0).sum()),
                   spell_missing_participants=int(((A['spell_a'] <= 0) | (A['spell_b'] <= 0)).sum()),
                   by_split={r: int((A['split_role'] == r).sum()) for r in ('TRAIN', 'VALIDATION', 'TEST')}),
                   external=ext, role_supervision_provenance_sha256=prov_sha,
                   slot_order='StateV2 participant order: sorted by (teamId, participantId); slots 1-5 Blue(100), 6-10 Red(200)')
        C.write_json(out_dir / 'draft_manifest.json', man)
        st.update('complete', 'draft', teams_used=ok_teams, team_reasons=dict(reasons), next_step='contract tests; role models')
        return 0
    except SystemExit as exc:
        st.update('failed', 'draft', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'draft', error=repr(exc), next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
