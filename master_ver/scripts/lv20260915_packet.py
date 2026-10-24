"""Label validity stage C: deterministic TRAIN-only blinded human review packet (NOT automated ground truth).

120 unique-match cases maximum from MAIN_TRAIN h90 valid rows (protocol review_packet section): S1 40 T small |delta_A|,
S2 40 T A-vs-B disagreement, S3 20 T observed kill/gold direction conflict with an objective, S4 20 N reference.
Raw cache files are read only for the selected matches (sha256 before/after). The reviewer Markdown is built from an
allowlisted view without A/B probabilities, delta, Y, stratum, q, W or identifiers; mapping and analysis values go to
PRIVATE files in the output root. Human review is recorded as UNPERFORMED.
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

import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

PK = K.OUT / 'review_packet'
STRATA = (('S1_T_small_abs_delta_A', 40), ('S2_T_A_disagrees_with_either_B', 40),
          ('S3_T_observed_direction_conflict_with_objective', 20), ('S4_N_reference', 20))
TEAM = {100: '블루', 200: '레드'}
DRAGON_KO = {'FIRE': '화염', 'WATER': '바다', 'EARTH': '대지', 'AIR': '바람', 'HEXTECH': '마법공학', 'CHEMTECH': '화학공학', 'OTHER': '기타'}
MONSTER_KO = {'BARON_NASHOR': '내셔 남작(바론)', 'RIFTHERALD': '협곡의 전령', 'HORDE': '공허 유충', 'ATAKHAN': '아타칸'}
TOWER_KO = {'OUTER_TURRET': '외곽 포탑', 'INNER_TURRET': '내부 포탑', 'BASE_TURRET': '억제기 포탑', 'NEXUS_TURRET': '넥서스 포탑'}
LANE_KO = {'TOP_LANE': '탑', 'MID_LANE': '미드', 'BOT_LANE': '바텀'}
REASON_KO = {'horizon': '마지막 킬 후 90초 상한', 'next_kill': '다음 챔피언 킬 직전', 'next_engagement_start': '다음 교전 시작 직전',
             'game_end': '경기 종료 직전'}
FINE_KO = {0: '픽(한쪽 1명 이하)', 1: '소규모 교전(양측 최소 2-3명)', 2: '한타(양측 최소 4명 이상)', -1: '규모 미상'}


def mmss(ms):
    ms = int(ms)
    sign = '-' if ms < 0 else ''
    ms = abs(ms)
    return f'{sign}{ms // 60000:02d}:{(ms % 60000) / 1000:04.1f}'


def _i(v):
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def describe(e, tm, champ):
    typ = str(e.get('type'))

    def who(pid):
        pid = _i(pid)
        return f'{TEAM[tm[pid]]} {champ.get(pid, "?")}' if pid in tm else '챔피언 아님(포탑/미니언/몬스터 등)'
    if typ == 'CHAMPION_KILL':
        pos = e.get('position') or {}
        pos_s = f" @({_i(pos.get('x'))}, {_i(pos.get('y'))})" if isinstance(pos, dict) and pos else ''
        assists = [who(a) for a in (e.get('assistingParticipantIds') or [])]
        return f"킬: {who(e.get('killerId'))} → {who(e.get('victimId'))}" + (f" (어시스트: {', '.join(assists)})" if assists else '') + pos_s
    if typ == 'ELITE_MONSTER_KILL':
        team = _i(e.get('killerTeamId'))
        tname = TEAM.get(team, f'팀 미기록(killerTeamId={team})')
        mt = str(e.get('monsterType'))
        if mt == 'DRAGON':
            sub = str(e.get('monsterSubType', '')).upper()
            name = '장로 드래곤' if sub == 'ELDER_DRAGON' else f"{DRAGON_KO.get(sub[:-7] if sub.endswith('_DRAGON') else sub, '기타')} 드래곤"
        else:
            name = MONSTER_KO.get(mt, mt)
        return f'오브젝트: {tname} 팀 {name} 처치'
    if typ == 'DRAGON_SOUL_GIVEN':
        team = _i(e.get('teamId'))
        return f"용의 영혼: {TEAM.get(team, '소유 팀 미기록(teamId 0)')} ({e.get('name') or e.get('dragonSoul') or '?'})"
    if typ == 'BUILDING_KILL':
        lost = _i(e.get('teamId'))
        what = '억제기' if e.get('buildingType') == 'INHIBITOR_BUILDING' else TOWER_KO.get(str(e.get('towerType')), '포탑')
        return f"구조물: {TEAM.get(300 - lost, '?')} 팀이 {TEAM.get(lost, '?')} {LANE_KO.get(str(e.get('laneType')), '')} {what} 파괴"
    if typ == 'TURRET_PLATE_DESTROYED':
        lost = _i(e.get('teamId'))
        return f"포탑 방패: {TEAM.get(lost, '?')} {LANE_KO.get(str(e.get('laneType')), '')} 포탑 방패 파괴"
    return None


def state_summary(names, x):
    ix = {n: i for i, n in enumerate(names)}

    def v(n):
        return int(round(x[ix[n]]))
    out = {}
    for t in ('blue', 'red'):
        towers = sum(v(f'{t}_tower_{k}') for k in ('OUTER_TURRET', 'INNER_TURRET', 'BASE_TURRET', 'NEXUS_TURRET', 'OTHER'))
        souls = [d for d in A.DRAGON_ELEMENTS if x[ix[f'{t}_soul_{d}']] > 0]
        out[t] = dict(kills=v(f'{t}_kills'), towers=towers, inhibitors=v(f'{t}_inhibitor_kills'), plates=v(f'{t}_plates'),
                      dragons=v(f'{t}_dragons'), elder=v(f'{t}_elder'), baron=v(f'{t}_baron'), herald=v(f'{t}_herald'),
                      grubs=v(f'{t}_horde'), atakhan=v(f'{t}_atakhan'), soul=','.join(DRAGON_KO[d] for d in souls) or '-')
    return out


def frame_resources(ts, node, gtm, nn, tm, champ, frame_ms):
    fi = int(np.searchsorted(ts, frame_ms))
    if fi >= len(ts) or int(ts[fi]) != int(frame_ms):
        raise ValueError('frame time not found')
    gi, xi, li, ai = (nn.index(k) for k in ('totalGold_norm', 'xp_norm', 'level_norm', 'alive'))
    out = dict(frame_time=mmss(frame_ms))
    for team, col in ((100, 0), (200, 1)):
        pids = sorted(p for p in tm if tm[p] == team)
        out[TEAM[team]] = dict(team_total_gold=int(round(gtm[fi, col])),
                               team_xp_cache_x20000=int(round(sum(node[fi, p - 1, xi] for p in pids) * 20000)),
                               alive_at_frame=int(sum(node[fi, p - 1, ai] > 0 for p in pids)),
                               levels=', '.join(f'{champ.get(p, "?")} {int(round(node[fi, p - 1, li] * 18))}' for p in pids))
    return out


def main():
    st = K.Status('packet')
    K.log_command()
    try:
        if not K.frozen_manifest_path().exists():
            raise SystemExit('frozen_manifest.json required (comparator labels needed for strata)')
        if (K.OUT / 'PRIVATE_review_case_key.json').exists():
            raise SystemExit('packet already built')
        P = K.ParentReadOnly()
        keys = ['match', 's', 'L', 'q_pre', 'sub_role', 'pre_snapshot', 'valid_h90', 'endpoint_h90', 'reasons_h90', 'post_snapshot_h90',
                'during_counts', 'after_counts_h90', 'count_keys', 'next_kill', 'next_start_eff', 'game_end']
        lab = P.label_file('MAIN_TRAIN', 'TRAIN-only review packet selection and case content', keys=keys)
        if not np.all(np.char.startswith(lab['sub_role'].astype(str), 'fold')):
            raise SystemExit('packet source must be TRAIN only')
        with np.load(K.OUT / 'alt_labels' / 'MAIN_TRAIN_alt_labels.npz', allow_pickle=False) as z:
            alt = {k: z[k] for k in z.files}
        with np.load(K.OUT / 'observed' / 'MAIN_TRAIN_observed.npz', allow_pickle=False) as z:
            obs = {k: z[k] for k in ('match', 's', 'kill_diff_h90', 'gold_diff_change_norm_h90', 'pre_snapshot', 'post_snapshot_h90')}
        co = K.cohort_file('MAIN_TRAIN')
        if not (np.array_equal(alt['match'], lab['match']) and np.array_equal(obs['s'], lab['s']) and np.array_equal(co['s'], lab['s'])):
            raise SystemExit('packet inputs not aligned')
        ck = [str(k) for k in lab['count_keys']]
        v = lab['valid_h90'] == 1
        T = v & (alt['cohort'] == 1)
        N = v & (alt['cohort'] == 0)
        full = lab['during_counts'] + lab['after_counts_h90']
        obj_known = np.zeros(len(v), bool)
        for obj in A.OBJECTIVES_OWNED:
            b, r, _t, _u = A.objective_team_columns(full, ck, obj)
            obj_known |= (b + r) > 0
        ksign = A.sign3(obs['kill_diff_h90'])
        gcat = A.resource_category(obs['gold_diff_change_norm_h90'], obs['pre_snapshot'], obs['post_snapshot_h90'])
        kconf, gconf = A.direction_conflict(alt['Y_h90_A'], ksign, gcat)
        masks = [T & (np.abs(alt['delta_h90_A']) <= .01),
                 T & ((alt['Y_h90_A'] != alt['Y_h90_B_reg']) | (alt['Y_h90_A'] != alt['Y_h90_B_econ'])),
                 T & obj_known & (kconf | gconf),
                 N]
        strata = [(nm, k, mk) for (nm, k), mk in zip(STRATA, masks)]
        cases, report = A.select_packet(lab['match'], lab['s'], strata, K.PACKET_SEED)
        cases.sort(key=lambda c: A.case_order_key(K.PACKET_SEED, c['match'], c['s']))
        for i, c in enumerate(cases, 1):
            c['case_id'] = f'LV-{i:03d}'
        st.update('running', 'selected', cases=len(cases), report=report, next_step='load states and raw cache')
        sel = {(c['match'], c['s']): c for c in cases}
        names = P.manifest('MAIN')['names']
        Xs = {}
        for sp, _op, _cm in P.chunk_paths('MAIN'):
            with np.load(sp, allow_pickle=False) as z:
                em = z['e_match']
                hit = np.flatnonzero(np.isin(em, [c['match'] for c in cases]))
                if not len(hit):
                    continue
                es = z['e_s']
                for j in hit:
                    key = (str(em[j]), int(es[j]))
                    if key in sel:
                        Xs[key] = (z['e_X_pre'][j], z['e_X_post_h90'][j])
        if len(Xs) != len(cases):
            raise SystemExit('state rows for selected cases incomplete')
        W = P.load_outcomes('MAIN', [f'fold{k}' for k in range(C.N_FOLDS)], purpose='analysis key W for TRAIN review cases (not in reviewer view)')
        with np.load(K.FC / 'predictions' / 'q_h90_trainval.npz', allow_pickle=False) as z:
            qk = {(m, int(s)): i for i, (m, s) in enumerate(zip(z['match'].tolist(), z['s_ms'].tolist()))}
            qsel = C.read_json(K.FC / 'selection' / 'q_h90.json')
            q_chosen = qsel['chosen']
            qv = z[q_chosen]
        os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime')
        sys.path.insert(0, str(C.WT))
        os.environ['LOL_CFG_PRESET'] = 'v3.3'
        os.environ['LOL_CFG_OVERRIDES'] = json.dumps({'CACHE_DIRNAME': str(C.CACHE_MAIN), 'FIGHT_INDEX_CACHE_ENABLED': False,
                                                      'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False, 'CACHE_IN_RAM': False})
        from core.config import NODE_FEATURE_NAMES
        nn = list(NODE_FEATURE_NAMES)
        raw_before, raw_after = {}, {}
        views, private_key, analysis_rows = [], {}, []
        for n_done, c in enumerate(cases):
            i = c['row_index']
            mid = c['match']
            paths = [C.CACHE_MAIN / f'{mid}{sfx}' for sfx in ('.npz', '.events.json', '.meta.json')]
            for p in paths:
                raw_before[p.name] = C.sha256_file(p)
            events = json.loads(paths[1].read_text(encoding='utf-8'))
            meta = json.loads(paths[2].read_text(encoding='utf-8'))
            with np.load(paths[0], allow_pickle=False) as z:
                ts = z['minute_ts'].astype(np.int64)
                node = z['node_minute'].astype(np.float64)
                gtm = z['gold_team_minute'].astype(np.float64)
            tm = {int(k): int(val) for k, val in meta['team_map'].items()}
            champ = {int(k): str(val) for k, val in (meta.get('static_meta', {}).get('champion_name_by_pid') or {}).items()}
            s, L, q_pre, e = int(lab['s'][i]), int(lab['L'][i]), int(lab['q_pre'][i]), int(lab['endpoint_h90'][i])
            rows = A.raw_event_rows(events)
            timeline = []
            for t, typ, ev in rows:
                if s - 60000 < t <= e:
                    d = describe(ev, tm, champ)
                    if d is None:
                        continue
                    phase = '시작 전(참고)' if t < s else ('교전 구간 [s, L]' if t <= L else 'L 이후 후속 (L, e]')
                    timeline.append(dict(time=mmss(t), phase=phase, event=d))
            reasons = str(lab['reasons_h90'][i]).split('|')
            term = []
            if 'next_kill' in reasons:
                nk = int(lab['next_kill'][i])
                for t, typ, ev in rows:
                    if t == nk and typ == 'CHAMPION_KILL':
                        term.append(f'{mmss(t)} ' + describe(ev, tm, champ) + ' — 라벨 구간 밖(e 직후 사건)')
            if 'next_engagement_start' in reasons:
                term.append(f"{mmss(int(lab['next_start_eff'][i]))} 다음 교전 시작 시각(그 교전의 첫 킬 15초 전, 사후 탐지)")
            if 'game_end' in reasons:
                term.append(f"{mmss(int(lab['game_end'][i]))} 경기 종료 (결과 비공개)")
            if 'horizon' in reasons:
                term.append(f'{mmss(L + 90000)} 마지막 킬 후 90초 상한')
            xpre, xpost = Xs[(mid, s)]
            post_frame = int(lab['post_snapshot_h90'][i])
            view = dict(
                case_id=c['case_id'],
                times=dict(start_s=mmss(s), first_kill_K=mmss(s + 15000), last_kill_L=mmss(L), pre_query=mmss(q_pre), endpoint_e=mmss(e),
                           after_L_seconds=round((e - L) / 1000, 1), ending=' + '.join(REASON_KO.get(r, r) for r in reasons)),
                scale=dict(fine=FINE_KO[int(co['fine'][i])], participants_blue=int(co['cluster_blue'][i]), participants_red=int(co['cluster_red'][i]),
                           n_min=int(co['n_min'][i])),
                champions={TEAM[t]: ', '.join(champ.get(p, '?') for p in sorted(pp for pp in tm if tm[pp] == t)) for t in (100, 200)},
                pre_state=state_summary(names, xpre), end_state=state_summary(names, xpost),
                resources=dict(pre=frame_resources(ts, node, gtm, nn, tm, champ, int(lab['pre_snapshot'][i])),
                               post=frame_resources(ts, node, gtm, nn, tm, champ, post_frame),
                               post_frame_after_L=post_frame > L, same_frame=post_frame == int(lab['pre_snapshot'][i])),
                timeline=timeline, terminating_event=term,
                unavailable=['프레임은 약 1분 간격: 골드/경험치/레벨/생존 여부는 표시된 프레임 시각의 값이며 사이 구간은 관측되지 않음',
                             '챔피언 위치는 킬 사건 좌표만 제공; 교전 중 이동·스킬 적중·시야·피해량 흐름 등은 원시 사건 목록으로 확인 불가',
                             '참여 인원은 탐지기의 사후 추정치이며 실제 전투 참가자 정답이 아님'] +
                            (['L 이후 새 프레임이 없어 마지막 킬 이후 자원 변화는 관측되지 않음'] if post_frame <= L else []) +
                            (['시작 전과 종료 시점이 같은 프레임: 자원 변화 관측 불가'] if post_frame == int(lab['pre_snapshot'][i]) else []))
            if set(view) != set(A.REVIEWER_ALLOWED_FIELDS):
                raise SystemExit('reviewer view fields differ from allowlist')
            views.append(view)
            private_key[c['case_id']] = dict(match=mid, s=s, row_index=i, set='MAIN_TRAIN', sub_role=str(lab['sub_role'][i]))
            qi = qk.get((mid, s))
            analysis_rows.append(dict(case_id=c['case_id'], stratum=c['stratum'], inclusion_probability_conditional=c['inclusion_probability_conditional'],
                                      eligible_rows_in_match=c['eligible_rows_in_match'], eligible_matches_remaining=c['eligible_matches_remaining'],
                                      **{f'p_pre_{m}': float(alt[f'p_pre_{m}'][i]) for m in K.ALL_V},
                                      **{f'p_post_h90_{m}': float(alt[f'p_post_h90_{m}'][i]) for m in K.ALL_V},
                                      **{f'delta_h90_{m}': float(alt[f'delta_h90_{m}'][i]) for m in K.ALL_V},
                                      **{f'Y_h90_{m}': int(alt[f'Y_h90_{m}'][i]) for m in K.ALL_V},
                                      q_pooled_h90_in_sample_train=float(qv[qi]) if qi is not None else None, q_candidate=q_chosen,
                                      W_final_blue_win=int(W[mid][0]), kill_diff_full=float(obs['kill_diff_h90'][i]),
                                      gold_diff_change_norm=float(obs['gold_diff_change_norm_h90'][i]), gold_category=A.RESOURCE_CODES[int(gcat[i])],
                                      objective_known_team_full=bool(obj_known[i]), kill_conflict=bool(kconf[i]), gold_conflict=bool(gconf[i]),
                                      cohort=int(alt['cohort'][i])))
            for p in paths:
                raw_after[p.name] = C.sha256_file(p)
            if n_done % 20 == 19:
                st.update('running', 'case_content', processed=n_done + 1, total=len(cases))
        # ------------------------------------------------ reviewer markdown
        md = ['# 교전 사례 검토 패킷 (블라인드) — label_validity_full_20260915', '',
              '## 안내', '',
              '- 이 문서는 사람 전문가 검토용이다. 사례는 15.14 학습(TRAIN) 경기에서 결정적 규칙으로 추출했으며, 대표 표본이 아니라 층화 사례 발굴이다.',
              '- 모델 확률, 모델 라벨, 사례 선정 사유, 경기 결과는 의도적으로 숨겼다. 경기 ID·플레이어 식별자는 포함하지 않는다. 경기를 외부에서 찾아보지 말 것.',
              '- 표시 정보: 교전 시각(s = 첫 킬 15초 전, K = 첫 킬, L = 마지막 킬, 사전 시점 = s 직전 1ms, e = 라벨 종료 시점), 탐지 규모, 양 팀 챔피언, '
              '사전/종료 시점 누적 상태, 1분 프레임의 자원 요약, 킬·오브젝트·구조물 원시 사건, 종료를 결정한 사건.',
              '- **중요:** 팀의 전략적 우세에 대한 판단은 관측된 경기 승리의 인과적 정답이 아니다. 원시 사건 목록만으로는 위치 선정·스킬 적중·시야 등 실력/운영의 정확성을 판단할 수 없다.',
              '- 확신이 없으면 "판단 불가(insufficient)" 또는 "불명확(unclear)"을 선택한다.', '',
              '## 응답 항목 (review_form_blank.csv에 기입)', '',
              '- (a) 관측된 단기 교환 결과: `Blue` / `Red` / `neutral` / `insufficient`',
              '- (b) 표시된 구간 전체의 전략적 우세: `Blue` / `Red` / `neutral` / `insufficient`',
              '- (c) 종료 시점 e: `truncates_continuing_fight`(계속되던 싸움을 자름) / `new_encounter`(새 교전이 시작됨) / `unclear`',
              '- (d) 판단에 부족했던 관측 정보 (자유 기술)',
              '- (e) 확신도: 1(매우 낮음) ~ 5(매우 높음)', '',
              '영문 요약: blinded TRAIN-only case packet for human interpretation; model outputs, labels, selection strata and game results '
              'are hidden. Team strategic judgment is not observed game-winning causal truth; raw events cannot establish positioning or skill accuracy.', '']
        for vw in views:
            t = vw['times']
            md += [f"## {vw['case_id']}", '',
                   f"- 시각: s {t['start_s']} / K {t['first_kill_K']} / L {t['last_kill_L']} / 사전 시점 {t['pre_query']} / e {t['endpoint_e']} "
                   f"(L 이후 {t['after_L_seconds']}초, 종료 규칙: {t['ending']})",
                   f"- 규모: {vw['scale']['fine']}; 탐지 참여 인원 블루 {vw['scale']['participants_blue']} / 레드 {vw['scale']['participants_red']}",
                   f"- 블루 챔피언: {vw['champions']['블루']}", f"- 레드 챔피언: {vw['champions']['레드']}", '',
                   '| 누적 상태 | 블루 사전 | 레드 사전 | 블루 e | 레드 e |', '|---|---:|---:|---:|---:|']
            for k, ko in (('kills', '킬'), ('towers', '포탑'), ('inhibitors', '억제기'), ('plates', '포탑 방패'), ('dragons', '원소 드래곤'),
                          ('elder', '장로'), ('baron', '바론'), ('herald', '전령'), ('grubs', '공허 유충'), ('atakhan', '아타칸'), ('soul', '영혼')):
                md.append(f"| {ko} | {vw['pre_state']['blue'][k]} | {vw['pre_state']['red'][k]} | {vw['end_state']['blue'][k]} | {vw['end_state']['red'][k]} |")
            md += ['', '| 자원 프레임 | 프레임 시각 | 팀 | 총 골드 | 경험치(캐시값×20000) | 생존 | 레벨 |', '|---|---|---|---:|---:|---:|---|']
            for lab_ko, key in (('사전', 'pre'), ('종료', 'post')):
                r = vw['resources'][key]
                for tk in ('블루', '레드'):
                    md.append(f"| {lab_ko} | {r['frame_time']} | {tk} | {r[tk]['team_total_gold']} | {r[tk]['team_xp_cache_x20000']} | {r[tk]['alive_at_frame']} | {r[tk]['levels']} |")
            md += ['', '| 시각 | 구간 | 원시 사건 |', '|---|---|---|']
            md += [f"| {ev['time']} | {ev['phase']} | {ev['event']} |" for ev in vw['timeline']] or ['| - | - | 사건 없음 |']
            md += ['', '- 종료를 결정한 사건: ' + ('; '.join(vw['terminating_event']) or '-'),
                   '- 관측 한계: ' + ' / '.join(vw['unavailable']), '']
        md_text = '\n'.join(md) + '\n'
        # ------------------------------------------------ blind checks
        found_ids = [m for m in private_key.values() if m['match'] in md_text or m['match'].split('_')[-1] in md_text]
        forbidden = [tok for tok in A.REVIEWER_FORBIDDEN_TOKENS if tok in md_text]
        decimals = re.findall(r'(?<![\d:.])0\.\d{2,}', md_text)
        blind = dict(match_ids_found=len(found_ids), forbidden_tokens_found=forbidden, zero_point_decimals_found=decimals[:10],
                     view_fields=list(A.REVIEWER_ALLOWED_FIELDS))
        blind['pass'] = not found_ids and not forbidden and not decimals
        if not blind['pass']:
            raise SystemExit(f'reviewer blind check failed: {blind}')
        PK.mkdir(parents=True, exist_ok=True)
        (PK / 'REVIEWER_PACKET_KO.md').write_bytes(md_text.encode('utf-8'))
        form_cols = ['case_id', 'reviewer_id', 'a_short_term_exchange', 'b_strategic_advantage', 'c_endpoint', 'd_missing_observation',
                     'e_confidence_1to5', 'notes']
        with open(PK / 'review_form_blank.csv', 'w', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(form_cols)
            for vw in views:
                wr.writerow([vw['case_id']] + [''] * (len(form_cols) - 1))
        C.write_json(K.OUT / 'PRIVATE_review_case_key.json', dict(note='PRIVATE: case -> match mapping; do not share with reviewers', cases=private_key))
        with open(K.OUT / 'PRIVATE_review_analysis_key.csv', 'w', encoding='utf-8', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(analysis_rows[0].keys()))
            wr.writeheader()
            wr.writerows(analysis_rows)
        status = dict(human_review='UNPERFORMED', reviewers=[], ratings_filled=0,
                      note='Expert names and ratings are intentionally empty and must never be filled with AI judgments.')
        C.write_json(PK / 'review_status.json', status)
        selection = dict(role=K.ROLE_TAG, source='MAIN_TRAIN h90 valid rows (OOF labels)', seed=K.PACKET_SEED, strata_report=report,
                         cases=len(cases), cases_by_stratum={nm: sum(c['stratum'] == nm for c in cases) for nm, _ in STRATA},
                         unique_matches=len({c['match'] for c in cases}) == len(cases),
                         shortages={r['stratum']: r['shortage'] for r in report},
                         inclusion_probability_note='conditional on higher-priority selections, treating the hash as a uniform random permutation; '
                                                    'deterministic selection; stratified case finding, not a representative accuracy sample',
                         scale_disclosure_note='reviewers see detected scale; N-reference cases are therefore distinguishable from T cases',
                         blind_check=blind, raw_files_read=len(raw_before), raw_files_unchanged=raw_before == raw_after,
                         reviewer_packet_sha256=C.sha256_file(PK / 'REVIEWER_PACKET_KO.md'),
                         review_form_sha256=C.sha256_file(PK / 'review_form_blank.csv'),
                         private_key_sha256=C.sha256_file(K.OUT / 'PRIVATE_review_case_key.json'),
                         private_analysis_key_sha256=C.sha256_file(K.OUT / 'PRIVATE_review_analysis_key.csv'),
                         human_review='UNPERFORMED', written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        C.write_json(K.OUT / 'results' / 'review_packet_selection.json', selection)
        if not selection['raw_files_unchanged']:
            raise SystemExit('raw cache files changed during packet build')
        st.update('complete', 'packet', cases=len(cases), shortages=selection['shortages'], next_step='post-run checks')
        return 0
    except SystemExit as exc:
        st.update('failed', 'packet', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'packet', error=repr(exc), next_step='fix and rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
