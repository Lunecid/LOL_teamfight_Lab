"""Objective-channel ablation stage R: Korean REPORT.md, DEFINITION_AND_EVIDENCE.md, tables/PUBLICATION_TABLES.md.

Numbers come from result files; narrative direction statements are guarded by assertions. Run after post-run checks.
"""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import oc20260915_common as K  # noqa: E402

import time  # noqa: E402

O = K.OUT
SETS = list(K.LABEL_SETS)
NONPILOT = [s for s in SETS if s != 'EXT_KR_16.14_pilot']
MAIN_W = ['MAIN_TEST', 'MAIN_V_CAL', 'MAIN_V_SELECT', 'MAIN_Q_CAL', 'MAIN_Q_SELECT', 'MAIN_TRAIN_heldout_fold']

UNKNOWN_V2 = ('V2 모델 입력 기준: 쿼리 시각 이하 사건 중 팀이 100/200으로 식별되지 않은 ELITE_MONSTER_KILL(killerTeamId 및 처치자 로스터 팀으로도 미식별)과 '
              'BUILDING_KILL/TURRET_PLATE_DESTROYED(teamId 미식별)의 누적 수. gameplay/state_value_v2.py가 teamId 미식별 DRAGON_SOUL_GIVEN을 legacy builder에 넘기기 전에 '
              '제거하고 unassigned_soul_events로 따로 세므로(모델 입력 아님), 미할당 영혼은 이 열에 들어가지 않는다. 오브젝트와 구조물이 섞여 있어 다른 입력을 바꾸지 않고 분리할 수 없으므로 그대로 유지했다.')
ERRATUM = ('정정(보고서 작성 후 Codex 소스 검토에서 발견, 문서만 해당): protocol.json·feature_evidence.json·oc20260915_common.UNKNOWN_SEMANTICS의 설명은 legacy builder 기준으로 '
           '미할당 영혼 분기를 포함했으나 V2 입력에는 해당하지 않는다. 동결 파일은 출처 보존을 위해 수정하지 않았고, 176/185 특징 결정·모델·결과는 바뀌지 않는다. errata.json과 CODEX_SOURCE_SEMANTICS_NOTE.md 참조.')


def f(x, d=4):
    return '-' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def pct(x, d=2):
    return '-' if x is None else f'{100 * x:.{d}f}%'


def ci(e, d=2):
    return f"{pct(e['estimate'], d)} [{pct(e['ci95'][0], d)}, {pct(e['ci95'][1], d)}]"


def table(h, rows):
    esc = lambda c: str(c).replace('|', r'\|')  # noqa: E731
    return ['| ' + ' | '.join(esc(c) for c in h) + ' |', '|' + '|'.join('---' for _ in h) + '|'] + ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in rows]


def main():
    st = K.Status('report')
    K.log_command()
    val = C.read_json(O / 'validation.json')
    proto = C.read_json(O / 'protocol.json')
    fz = C.read_json(O / 'frozen_manifest.json')
    ev = C.read_json(O / 'feature_evidence.json')
    man = C.read_json(O / 'v_models_manifest_B_noobj.json')
    sel = C.read_json(O / 'selection' / 'selection_secondary_calibration.json')
    w = C.read_json(O / 'results' / 'w_eval.json')
    lm = C.read_json(O / 'labels' / 'labels_manifest.json')
    per = {n: C.read_json(O / 'results' / 'per_set' / f'{n}.json') for n in SETS}
    qd = C.read_json(O / 'results' / 'q_target_dependence.json')
    vmc = C.read_json(O / 'contracts' / 'v_model_checks_full.json')

    # ------------------------------------------------ guarded claims
    assert val['all_pass']
    for n in MAIN_W + ['EXT_KR_16.13', 'EXT_KR_16.15', 'EXT_NA1_16.13']:
        s = w['sets'][n]
        assert s['B_noobj_raw']['match_weighted']['logloss'] > s['A_raw']['match_weighted']['logloss'], n
        assert s['B_noobj_raw']['match_weighted']['auc'] < s['A_raw']['match_weighted']['auc'], n
    pil = w['sets']['EXT_KR_16.14_pilot']
    assert pil['B_noobj_raw']['match_weighted']['logloss'] < pil['A_raw']['match_weighted']['logloss']
    bw = w['sets']['MAIN_TEST']['bootstrap_B_minus_A']['results']
    assert all(bw[wn][m]['ci95'][0] > 0 for wn in bw for m in bw[wn])
    dis = {n: {c: per[n]['bootstrap_h90'][c]['estimates']['A_vs_B_noobj'] for c in ('E', 'T', 'N')} for n in SETS}
    rng_main = [dis[n][c]['match_weighted']['estimate'] for n in ('MAIN_TRAIN', 'MAIN_VALIDATION', 'MAIN_TEST') for c in ('E', 'T', 'N')]
    rng_np = [dis[n][c]['match_weighted']['estimate'] for n in NONPILOT for c in ('E', 'T', 'N')]
    t_gt_n = all(dis[n]['T']['match_weighted']['estimate'] > dis[n]['N']['match_weighted']['estimate'] for n in NONPILOT)
    assert t_gt_n
    zeros = sum(v for s in lm['summaries'].values() for h in K.HS for v in s[f'h{h}']['exact_zero_delta'].values())
    sig_mis = sum(s[f'h{h}']['sign_invariance_sigmoid_vs_raw']['label_mismatch_rows'] for s in lm['summaries'].values() for h in K.HS)
    assert sig_mis == 0 and zeros == 0
    sE = per['MAIN_TEST']['strata_h90']['E']
    assert sE['abs_delta_A']['[0,.005]']['disagreement']['row'] > sE['abs_delta_A']['>.02']['disagreement']['row']
    assert sE['same_pre_post_frame']['1']['disagreement']['row'] < sE['same_pre_post_frame']['0']['disagreement']['row']
    obT = per['MAIN_TEST']['objectives_h90']['cohorts']['T']
    for wn in ('full_(q_pre,e]', 'after_(L,e]'):
        for o in ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan', 'soul_owned'):
            c = obT[wn][o]['single_team_oriented']
            assert c['oriented_delta_A_minus_B']['match_weighted'] > 0 and c['oriented_delta_B']['match_weighted'] > 0, (wn, o)
        ref = obT[wn]['no_known_team_acquisition_reference']['rows']['disagreement']['row']
        assert all(obT[wn][o]['single_team_oriented']['disagreement']['row'] > ref for o in ('baron', 'dragon', 'atakhan', 'soul_owned', 'horde')), wn
    obE = per['MAIN_TEST']['objectives_h90']['cohorts']['E']
    for wn in ('full_(q_pre,e]', 'after_(L,e]'):
        for o in ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan', 'soul_owned'):
            assert obE[wn][o]['single_team_oriented']['oriented_delta_A_minus_B']['match_weighted'] > 0, (wn, o)
    assert obE['full_(q_pre,e]']['horde']['blue_only']['oriented_delta_B']['match_weighted'] < 0
    tb = w['sets']['MAIN_TEST']
    gap = {k: tb['B_noobj_raw']['time_bands'][k]['match_weighted']['logloss'] - tb['A_raw']['time_bands'][k]['match_weighted']['logloss'] for k in ('2-10', '10-20', '20-30', '30+')}
    assert gap['2-10'] < gap['10-20'] < gap['20-30'] < gap['30+']
    qT, qN = qd['cohorts']['T'], qd['cohorts']['N']
    assert qT['scores']['Y_B']['auc'] > qT['scores']['Y_A']['auc'] and abs(qN['B_minus_A']['auc']) < .002
    t_all = w['sets']['MAIN_TEST']

    now = time.strftime('%Y-%m-%d %H:%M:%S')
    R = ['# 명시적 오브젝트 채널 제거 실험 보고서 (objective_channel_ablation_20260915)', '',
         f'작성: {now} KST. 설계·동결 명세: Codex (`docs/CLAUDE_OBJECTIVE_CHANNEL_ABLATION_20260915.md`), 구현·실행·검증: Claude Opus 5.', '',
         '> **판정:** 명세의 실행 단계 1–8을 전체 코퍼스(주 21만 경기 지정 패치 역할 + 외부 4세트 21,190경기)에서 끝까지 실행했고 사후 검증 '
         f"{val['passed']}/{val['total']}개가 통과했다. 이름에 바론·장로·드래곤·영혼·전령·유충·아타칸이 들어간 176개 열을 제거한 B_noobj는 "
         f"주 TEST에서 최종 승패(W) 예측이 A보다 나빴다(경기가중 log loss 차이 B−A {f(bw['match_weighted']['logloss']['estimate'], 4)}, 95% [{f(bw['match_weighted']['logloss']['ci95'][0], 4)}, {f(bw['match_weighted']['logloss']['ci95'][1], 4)}]). "
         f"h90 라벨 부호는 주 TEST에서 경기가중 {pct(dis['MAIN_TEST']['E']['match_weighted']['estimate'])}(E), {pct(dis['MAIN_TEST']['T']['match_weighted']['estimate'])}(T) 달라졌다. "
         '이것은 명시 오브젝트 정보의 **예측적 기여와 위에 보고한 라벨 부호 민감도**에 관한 탐색적 근거다. 개별 라벨의 정답성이나 오브젝트의 인과 효과를 입증하지 않는다.', '',
         '## 0. 연구 성격과 범위', '',
         '- 이미 TEST 결과를 본 뒤의 **탐색적 후속 실험**이다. 새 확증, 의미적 정답, 인과 효과를 주장하지 않는다. 사용자 결정에 따라 사람 검토는 이 과제에 포함하지 않았으며, 의미 타당성 검증은 미수행으로 남는다.',
         f"- protocol.json {proto['written_at']}(B 적합 전), frozen_manifest.json {fz['frozen_at']}(TEST/외부 결과·라벨 접근 전). "
         f"동결 전 결과 접근은 TRAIN fold·V_CAL·V_SELECT뿐이었다({len(fz['pre_freeze_outcome_access'])}건).",
         '- A(주 V: expanded StateV2 logistic C=.01, raw, 최종+5 OOF), 주 라벨, h90 주 상한과 h60/h120, E/T/N, 종료점, 패치 분할은 바꾸지 않았다. q를 다시 학습하지 않았다.',
         '- 부모(전체 학습, 코호트·역할, 라벨 타당성) 산출물은 읽기만 했다. 읽은 파일 해시와 세 루트의 전체 파일 목록이 전후 동일하다. 새 접근 로그는 이 루트에만 있다.', '']
    R += ['## 1. 실행 전(Before)과 실행 후(After)', '']
    R += table(['항목', 'Before', 'After'], [
        ['오브젝트 정보의 역할', 'B_econ 비교는 오브젝트·킬·구조물 등을 함께 제거해 분리 불가', '다른 입력은 그대로 두고 명명된 오브젝트 176열만 제거한 통제 비교 완료'],
        ['W 예측 효용', '미확인', f"주 TEST B−A log loss {f(t_all['B_minus_A_point']['match_weighted']['logloss'], 4)}, AUC {f(t_all['B_minus_A_point']['match_weighted']['auc'], 4)} (경기가중)"],
        ['라벨 민감도', '미확인', f"주 h90 E/T/N 경기가중 불일치 {pct(min(rng_main), 1)}–{pct(max(rng_main), 1)}"],
        ['오브젝트 획득 사례', '방향 연관만 기술', '획득 팀 기준 Δ를 A와 B로 비교(기술 통계)'],
        ['q 표적 의존성', 'B_reg/B_econ만', 'B_noobj 라벨로 동결 specialist 채점'],
        ['의미 타당성', '사람 검토 미수행', '이번 과제 범위 밖. 여전히 미수행']])
    # 2 features
    R += ['', '## 2. 특징 제거 규칙과 증거', '',
          f"- 규칙: A의 expanded 열 {ev['counts']['primary']}개 가운데 소문자 이름에 `{', '.join(K.OBJECTIVE_TOKENS)}` 중 하나라도 포함된 열을 제거한다. "
          f"제거 {ev['counts']['dropped']}개, 유지 {ev['counts']['retained']}개. 명세의 176/185와 같다. 순서 목록과 해시는 `feature_evidence.json`에 있다.",
          f"- 유지 열 sha256 `{ev['retained_sha256'][:16]}…`, 제거 열 sha256 `{ev['dropped_sha256'][:16]}…`. 두 목록은 적합 전에 protocol과 함께 저장했다.", '']
    R += table(['출처(StateV2 코드 블록)', '열 수'], [[k, v] for k, v in ev['columns_by_source'].items()])
    ub = vmc['unknown_objective_team_count_TRAIN_bucket']
    R += ['', '- 유지: 킬/사망/생존/HP/MP, 골드·경험치·레벨·CS·챔피언, 시간·시간², 포탑/포탑 방패/억제기와 각 시간 상호작용, `unknown_objective_team_count`. snapshot_age_s는 A와 같이 제외.',
          '- 제거에는 슬롯별 `baron/elder_death_since_acquisition`(오브젝트 획득 후 사망, 버프 대리변수)과 모든 오브젝트 시간 상호작용이 포함된다.',
          f"- `unknown_objective_team_count` 의미: {UNKNOWN_V2}",
          f"- **{ERRATUM}**",
          f"- 이 열이 0이 아닌 비율: TRAIN 버킷 {pct(ub['nonzero_row_share'])}(행) / {pct(ub['nonzero_match_weighted'])}(경기가중), 최대 {f(ub['max'], 0)}. 평가 세트는 7절 표에 있다.",
          '- **범위:** 이것은 이름으로 지정한 오브젝트 채널의 제거다. 골드·경험치·CS, 구조물, 시간, 팀 미상 합산 열은 오브젝트 정보를 간접적으로 담을 수 있다. 오브젝트 없는 데이터나 조건부 인과 효과가 아니다.', '']
    # 3 fit
    R += ['## 3. B_noobj 적합', '',
          f"- A와 같은 TRAIN 버킷 질의 {man['census']['TRAIN']['rows']:,}개 / {man['census']['TRAIN']['matches']:,}경기, 경기별 동일 가중치, C=.01, liblinear, random_state 7, 기존 5-fold. 전처리는 적합 행에서만 학습했다.",
          f"- 최종 적합 n_iter {man['final_fit']['n_iter']}, fold n_iter {', '.join(str(v['fit']['n_iter']) for v in man['folds'].values())}. 모두 1500회 안에 첫 시도로 수렴했고 반복 상한 상승은 없었다.",
          f"- 독립 점검 {sum(a['pass'] for a in vmc['adapters'].values())}/{len(vmc['adapters'])} 어댑터 통과: 적합 경기 집합 해시 재구성, 제외 fold, 스케일러 행 수, 보정 파티션. "
          f"B 입력이 A 입력의 유지 열과 **값까지 동일**함을 TRAIN 전 행에서 확인했다({vmc['retained_values_identical_to_A_inputs']}).",
          f"- **보조 보정:** V_CAL sigmoid 대 raw를 V_SELECT에서 비교했다. log loss raw {f(sel['select_scores']['raw']['logloss'], 6)}, sigmoid {f(sel['select_scores']['sigmoid_pos']['logloss'], 6)}이어서 **raw가 선택**됐다. "
          '따라서 보조 결과는 주 결과와 같다. sigmoid 계열은 부호 불변성 확인에만 썼다.', '']
    # 4 W
    R += ['## 4. W(최종 Blue 승리) 예측: 동일 버킷 키', '', '경기가중(경기마다 총가중치 동일)이 주 지표이고 행가중은 함께 보고한다. A 예측은 9개 세트 모두 부모 저장값과 비트 단위로 같았다.', '']
    rows = []
    for n, s in w['sets'].items():
        a, b = s['A_raw']['match_weighted'], s['B_noobj_raw']['match_weighted']
        ar, br = s['A_raw']['row_weighted'], s['B_noobj_raw']['row_weighted']
        rows.append([n, s['rows'], f(a['auc']), f(b['auc']), f(a['brier']), f(b['brier']), f(a['logloss']), f(b['logloss']), f(br['logloss'] - ar['logloss'], 4),
                     f(a.get('slope'), 3), f(b.get('slope'), 3)])
    R += table(['세트', '질의 행', 'AUC A', 'AUC B', 'Brier A', 'Brier B', 'LL A', 'LL B', 'LL B−A 행가중', '기울기 A', '기울기 B'], rows)
    R += ['', f"주 TEST 경기 단위 대응 bootstrap 1,000회(모델 고정, 평가 경기 재표집만): "
              f"경기가중 Brier B−A {f(bw['match_weighted']['brier']['estimate'], 5)} [{f(bw['match_weighted']['brier']['ci95'][0], 5)}, {f(bw['match_weighted']['brier']['ci95'][1], 5)}], "
              f"log loss {f(bw['match_weighted']['logloss']['estimate'], 5)} [{f(bw['match_weighted']['logloss']['ci95'][0], 5)}, {f(bw['match_weighted']['logloss']['ci95'][1], 5)}]; "
              f"행가중 Brier {f(bw['row_weighted']['brier']['estimate'], 5)} [{f(bw['row_weighted']['brier']['ci95'][0], 5)}, {f(bw['row_weighted']['brier']['ci95'][1], 5)}], "
              f"log loss {f(bw['row_weighted']['logloss']['estimate'], 5)} [{f(bw['row_weighted']['logloss']['ci95'][0], 5)}, {f(bw['row_weighted']['logloss']['ci95'][1], 5)}]. "
              '모든 복제에서 B가 나빴다.', '']
    rows = [[band, f(t_all['A_raw']['time_bands'][band].get('match_weighted', {}).get('auc')), f(t_all['B_noobj_raw']['time_bands'][band].get('match_weighted', {}).get('auc')),
             f(t_all['A_raw']['time_bands'][band].get('match_weighted', {}).get('logloss')), f(t_all['B_noobj_raw']['time_bands'][band].get('match_weighted', {}).get('logloss')),
             t_all['A_raw']['time_bands'][band].get('match_weighted', {}).get('n', 0)] for band in t_all['A_raw']['time_bands']]
    R += ['주 TEST 시간대(경기가중):', ''] + table(['분', 'AUC A', 'AUC B', 'LL A', 'LL B', '질의 행'], rows)
    R += ['', '**해석:** 명시 오브젝트 채널을 빼면 모든 주 파티션과 KR 16.13·KR 16.15·NA1 16.13에서 W 예측이 나빠졌다. log loss 차이는 2–10분보다 10분 이후에서 크고 후반으로 갈수록 커진다. '
              'KR 16.14 pilot(1,125질의)에서는 B가 약간 나았지만 표본이 작아 결론을 내리지 않는다. W 예측 개선은 예측 효용의 근거일 뿐, 개별 교전 라벨이 옳다는 증거가 아니다.', '']
    # 5 agreement
    R += ['## 5. 라벨 부호 민감도: A 대 B_noobj', '',
          f"Y=1[Δ>0], 같은 종료점과 같은 어댑터를 양 끝에 사용. 모든 세트·상한·모델에서 정확한 Δ=0은 {zeros}건이다. 구간은 경기 단위 1,000회 대응 bootstrap 백분위 95%이며 학습 불확실성은 포함하지 않는다.", '']
    rows = []
    for n in SETS:
        for c in ('E', 'T', 'N'):
            a = per[n]['agreement']['h90'][c]
            rows.append([n, c, a['rows'], a['matches'], ci(dis[n][c]['match_weighted']), ci(dis[n][c]['row']),
                         f"{pct(a['positive_rate_A_match_weighted'], 1)} / {pct(a['positive_rate_B_match_weighted'], 1)}",
                         f(a['delta_diff_mean_B_minus_A'], 5), f(a['delta_absdiff_mean'], 4), f(a['delta_spearman'], 3), '예' if a['sparse_lt30_matches'] else ''])
    R += table(['세트', '코호트', '행', '경기', '불일치 경기가중 [95%]', '불일치 행 [95%]', 'P(Y=1) A/B', '평균 ΔB−ΔA', '평균 |ΔB−ΔA|', 'Δ Spearman', '<30경기'], rows)
    mtE = per['MAIN_TEST']['agreement']['h90']['E']
    R += ['', f"- 비파일럿 세트 경기가중 불일치 범위 {pct(min(rng_np), 1)}–{pct(max(rng_np), 1)}. 비파일럿 모든 세트에서 T가 N보다 약간 높다.",
          f"- 세 상한의 종료점이 같은 행 / 달라지는 행(주 TEST E): {pct(mtE['endpoint_unchanged_60_90_120']['disagreement_row'])} / {pct(mtE['endpoint_changed_across_horizons']['disagreement_row'])}.",
          f"- 동일 어댑터의 양의 기울기 sigmoid를 양 끝에 적용한 라벨과 raw 라벨의 불일치: 전체 {sig_mis}행(방향 역전 0). 수학적으로 기대되는 구현 확인이며 독립 증거가 아니다.", '']
    hz = per['MAIN_TEST']['horizon']
    R += ['상한 간 부호 불일치(주 TEST E):', ''] + table(['V', '상한 쌍', '전체', '종료점 달라진 행', '그 행 불일치', '같은 종료점 불일치 수'],
                                                  [[m, p, pct(v['all']['row']), v['changed_endpoint']['rows'], pct(v['changed_endpoint']['row']), v['unchanged_endpoint']['disagreements']]
                                                   for m in ('A', 'B') for p, v in hz[m]['E'].items()])
    # 6 strata
    R += ['', '## 6. h90 층화 (주 TEST, 진단용)', '', '제외 기준이나 새 임계값이 아니다. same_pre_post_frame=1은 사전·종료 시점의 최신 프레임이 같은 행이다(L 이후 새 프레임 없음보다 좁은 조건).', '']
    for c in ('E', 'T'):
        rows = []
        for fam, bins in per['MAIN_TEST']['strata_h90'][c].items():
            for b, cc in bins.items():
                if cc.get('empty'):
                    continue
                rows.append([fam, b, cc['rows'], cc['matches'], pct(cc['disagreement']['row']), pct(cc['disagreement']['match_weighted']), f(cc['mean_delta_A']['row'], 4),
                             f(cc['mean_delta_B']['row'], 4), f(cc['mean_abs_delta_B_minus_A']['row'], 4), pct(cc['small_abs_delta_le_0.01_A']['row'], 1),
                             pct(cc['small_abs_delta_le_0.01_B']['row'], 1), '예' if cc['sparse_lt30_matches'] else ''])
        R += [f'### 코호트 {c}', ''] + table(['층', '구간', '행', '경기', '불일치 행', '불일치 경기가중', '평균 ΔA', '평균 ΔB', '평균 |ΔB−ΔA|', '|ΔA|≤.01', '|ΔB|≤.01', '<30경기'], rows) + ['']
    R += [f"- |Δ_A|≤.005 행의 불일치 {pct(sE['abs_delta_A']['[0,.005]']['disagreement']['row'])}, |Δ_A|>.02 행 {pct(sE['abs_delta_A']['>.02']['disagreement']['row'])}(E). 작은 Δ에서 부호가 더 쉽게 바뀐다.",
          f"- 같은 프레임 행의 불일치 {pct(sE['same_pre_post_frame']['1']['disagreement']['row'])}, 다른 프레임 {pct(sE['same_pre_post_frame']['0']['disagreement']['row'])}(E). "
          '같은 프레임에서는 두 모델의 Δ가 사건 이력 채널과 시간으로만 달라지므로 차이가 작다.', '']
    # 7 unknown col table
    R += ['## 7. unknown_objective_team_count 비영 비율', '']
    rows = [[n, pct(s['unknown_objective_team_count']['nonzero_row_share']), pct(s['unknown_objective_team_count']['nonzero_match_weighted']), f(s['unknown_objective_team_count']['max'], 0)]
            for n, s in w['sets'].items() if 'unknown_objective_team_count' in s]
    R += ['W 버킷 질의:', ''] + table(['세트', '행', '경기가중', '최대'], rows)
    rows = [[n, pct(s['unknown_objective_team_count_pre']['nonzero_row_share']), pct(s['unknown_objective_team_count_pre']['nonzero_match_weighted']),
             pct(s['unknown_objective_team_count_pre']['valid_h90_nonzero_row_share'])] for n, s in lm['summaries'].items()]
    R += ['', '교전 사전 상태(q_pre):', ''] + table(['세트', '행', '경기가중', '유효 h90 행'], rows)
    R += ['', '외부 세트에서 비율이 더 높다. 이 열은 A와 B 모두에 남아 있으므로 두 모델 차이의 원인은 아니지만, B에 남은 간접 오브젝트 정보의 한 경로가 될 수 있다.', '']
    # 8 objectives
    R += ['## 8. 오브젝트 획득 사례의 A/B 비교 (주 TEST h90, 기술 통계)', '',
          '- 창: full=(q_pre, e], after=(L, e]. 카운터는 부모 라벨의 during/after 계수, 원소별 드래곤 팀은 label-validity observed의 StateV2 차이(키 정확 대조)다. '
          '(L, e]의 원소별 팀은 같은 원소가 (q_pre, L]에 없을 때만 확정했다.',
          '- 획득 팀 기준 Δ: Blue 단독 획득 행은 +Δ, Red 단독 획득 행은 −Δ. 획득 팀 쪽으로 추정 승률이 얼마나 움직였는지를 나타낸다. '
          '**사건의 인과 가치, 시점 반사실, 오브젝트의 즉각 보상이 아니다.** 부분집합은 겹치므로 더하지 않는다. 기준 행(해당 창에 팀 확인 획득 없음)에도 이전 오브젝트 이력이 있을 수 있다.', '']
    for c in ('T', 'E'):
        oc = per['MAIN_TEST']['objectives_h90']['cohorts'][c]
        rows = []
        for wn in ('after_(L,e]', 'full_(q_pre,e]'):
            for o in ('baron', 'elder', 'atakhan', 'dragon', 'soul_owned', 'herald', 'horde'):
                for sub in ('single_team_oriented', 'blue_only', 'red_only', 'both_teams_DIAGNOSTIC', 'multiple_DIAGNOSTIC', 'unknown_team_DIAGNOSTIC'):
                    cc = oc[wn][o].get(sub, {})
                    if not cc or cc.get('empty'):
                        continue
                    rows.append([wn, o, sub, cc['rows'], pct(cc['disagreement']['row']), f(cc['mean_delta_A']['match_weighted'], 4), f(cc['mean_delta_B']['match_weighted'], 4),
                                 f(cc.get('oriented_delta_A', {}).get('match_weighted'), 4), f(cc.get('oriented_delta_B', {}).get('match_weighted'), 4),
                                 f(cc.get('oriented_delta_A_minus_B', {}).get('match_weighted'), 4), pct(cc['small_abs_delta_le_0.01_A']['row'], 1),
                                 pct(cc['small_abs_delta_le_0.01_B']['row'], 1), '예' if cc['sparse_lt30_matches'] else ''])
            for sub, cc in ((k, v) for k, v in oc[wn].items() if k.startswith(('soul_teamid0', 'no_known'))):
                cc = list(cc.values())[0]
                rows.append([wn, sub, '-', cc['rows'], pct(cc['disagreement']['row']), f(cc['mean_delta_A']['match_weighted'], 4), f(cc['mean_delta_B']['match_weighted'], 4),
                             '-', '-', '-', pct(cc['small_abs_delta_le_0.01_A']['row'], 1), pct(cc['small_abs_delta_le_0.01_B']['row'], 1), ''])
        R += [f'### 코호트 {c} (경기가중 평균)', ''] + table(['창', '오브젝트', '부분집합', '행', '불일치', '평균 ΔA', '평균 ΔB', '획득팀 ΔA', '획득팀 ΔB', '획득팀 A−B', '|ΔA|≤.01', '|ΔB|≤.01', '<30경기'], rows) + ['']
    rows = []
    for wn in ('after_(L,e]', 'full_(q_pre,e]'):
        for d in ('AIR', 'EARTH', 'FIRE', 'WATER', 'HEXTECH', 'CHEMTECH', 'OTHER'):
            ow = obT[wn][f'dragon_{d}']
            cc = ow.get('single_team_oriented', {})
            amb = ow.get('team_ambiguous_mixed_intervals_DIAGNOSTIC', {})
            if not cc or cc.get('empty'):
                rows.append([wn, d, 0, '-', '-', '-', '-', amb.get('rows', 0)])
                continue
            rows.append([wn, d, cc['rows'], pct(cc['disagreement']['row']), f(cc['oriented_delta_A']['match_weighted'], 4), f(cc['oriented_delta_B']['match_weighted'], 4),
                         f(cc['oriented_delta_A_minus_B']['match_weighted'], 4), amb.get('rows', 0)])
    R += ['### 원소 드래곤별 (주 TEST T, 단일 팀 획득)', ''] + table(['창', '원소', '행', '불일치', '획득팀 ΔA', '획득팀 ΔB', 'A−B', '팀 모호(진단)'], rows)
    R += ['', '**요약:** 주 TEST T와 E의 모든 오브젝트 범주에서 단일 팀 획득 행의 획득 팀 방향 Δ는 A가 B보다 크다(두 창 모두). T에서는 B의 획득 팀 방향 Δ도 모든 범주에서 양수다. E의 유충(horde) full 창에서는 B가 거의 0이고 Blue 단독 행은 음수(−0.0011)다. B가 킬·구조물·골드·경험치 등 남은 채널로 획득과 함께 오는 변화를 일부 반영하기 때문으로 읽을 수 있지만, 경로를 분해하지는 않았다. '
          'T에서 획득 행의 A/B 불일치는 팀 확인 획득이 없는 기준 행보다 높다(바론·드래곤·아타칸·영혼·유충). 모든 외부 세트와 코호트의 전체 표는 `tables/objectives_h90_overlapping.csv`에 있다. OTHER 드래곤은 관측되지 않았다. '
          '작은 행 수(예: 장로)와 pilot 세트에서는 넓은 결론을 내리지 않는다.', '']
    # 9 q
    R += ['## 9. 동결 specialist q의 표적 의존성 (주 TEST h90)', '', 'A 라벨로 학습된 q 예측을 그대로 두고 채점 라벨만 바꿨다. 재학습·순위·B 라벨의 최고 성능 주장이 아니다. A 점수는 감사값과 1e−12 이내로 일치했다.', '']
    rows = [[c, v['specialist'], v['keys']['rows'], pct(v['label_disagreement_on_q_rows']), f(v['scores']['Y_A']['auc']), f(v['scores']['Y_B']['auc']), f(v['scores']['Y_A']['brier']),
             f(v['scores']['Y_B']['brier']), f(v['scores']['Y_A']['logloss']), f(v['scores']['Y_B']['logloss'])] for c, v in qd['cohorts'].items()]
    R += table(['코호트', 'specialist', '행', '라벨 불일치', 'AUC vs Y_A', 'AUC vs Y_B', 'Brier vs Y_A', 'Brier vs Y_B', 'LL vs Y_A', 'LL vs Y_B'], rows)
    R += ['', f"T에서는 같은 q가 B_noobj 라벨에서 오히려 점수가 높다(AUC +{f(qT['B_minus_A']['auc'], 4)}). N에서는 거의 같다. "
              'B 라벨이 교전 중·후 오브젝트 획득에 덜 반응해 사전 정보로 더 예측되기 쉬울 가능성이 있지만, 이 원인은 검증하지 않았다. 표적 선택이 q 성능 수치에 영향을 준다는 점만 확인했다.', '']
    # 10 validation
    R += ['## 10. 검증', ''] + table(['검사', '결과'], [[k, '통과' if v['pass_'] else '실패'] for k, v in val['checks'].items()])
    R += ['', '사전 계약: 합성 pytest 8개(열 규칙·순서·해시, 유지/제외 열, 대소문자 무관 토큰, 공유 단조 보정 부호 보존, 접근 게이트, 어댑터 가드) 통과, TRAIN-only 스모크 적합과 독립 점검 통과, '
              '전체 적합 후 독립 점검 통과 뒤 동결했다. **자동 검사는 계산·동일성·분할·재현성 검증이며 의미 타당성이나 인과 귀속을 대신하지 않는다.**',
          '- 실패한 실행: 파이프라인 단계(protocol→보고서)는 모두 첫 실행에서 끝났다. 보고서 작성 뒤 정정 기록(errata.json)을 쓰는 인라인 보조 명령이 한 번 구문 오류로 '
          '아무 파일도 쓰지 못하고 실패했고, 보조 스크립트로 다시 실행했다. 보고서 문구 수정에 따른 보고서 재생성 기록도 `logs/report_run*.txt`에 남겼다. 명령은 `commands.txt`에 있다.', '']
    R += ['## 11. 해석 한계', '',
          '- 조건부 인과 효과가 아니다. B에는 골드/경험치/CS, 구조물, 킬, 시간, 팀 미상 합산 열이 남아 있어 오브젝트 정보를 간접적으로 담는다.',
          '- 불일치율은 라벨이 이 특징 블록에 얼마나 민감한지를 보여줄 뿐, 어느 라벨이 실제 한타 결과에 더 맞는지 정하지 않는다.',
          '- bootstrap은 고정 모델의 평가 경기 재표집만 반영한다. 적합·보정·선택 변동과 다중 비교 보정은 포함하지 않았다.',
          '- 외부 16.xx 세트에는 아타칸 사건이 없고(부모 감사), unknown_objective_team_count 비율이 주 코퍼스보다 높다. 과거 접근 이력이 있어 미접촉 확인 표본이 아니다.',
          '- 정의 산출(G/D)에는 15.16이 쓰였다(부모 감사). 사람의 의미 검토는 이번에도 수행하지 않았다.', '',
          '## 12. 근거 문헌의 사용 범위', '']
    R += table(['근거', '지지', '지지하지 않음'], [
        ['Maymin 2021 (doi:10.1515/jqas-2019-0096)', '상태 승률 변화 기반 가치평가', '176열 규칙, 90초, C=.01, 층 경계'],
        ['Kim, Lee & Chung, IEEE CoG 2020 (https://ieee-cog.org/2020/papers/paper_221.pdf)', '승률 확률 품질·보정의 중요성', '논문 방법 재현'],
        ['Jacobs & Wallach (arXiv:1912.05511)', '구성 개념과 측정의 구분', '예측 개선 또는 수치 일치로 구성 타당도 확보'],
        ['우리 설계/상속 정의', '토큰 규칙, raw 주 비교, 층·창·부분집합, h90·종료 규칙', '(문헌이 정한 값 아님)']])
    R += ['', '## 13. 다음 단계', '',
          '1. 원고: A의 명시 오브젝트 채널이 held-out W 예측을 개선한다는 탐색적 결과와, 라벨의 약 5% 부호가 이 블록에 의존한다는 민감도를 제한 사항과 함께 보고한다. 인과·정답 표현은 쓰지 않는다.',
          '2. 사용자 결정에 따라 사람 검토는 진행하지 않는다. 이후 기존 한타 예측·SHAP 분석을 정리할 때 이번 결과를 가치 모델의 입력 근거로 연결하되, 의미 타당성이 검증됐다고 표현하지 않는다.',
          '3. 간접 경로(골드·구조물·팀 미상 열)의 분해가 필요하면, 새 사전 명세로 추가 절제를 설계한다. 현재 A·h90·라벨은 그대로 유지한다.', '',
          '## 14. 파일', '']
    files = ['protocol.json', 'feature_evidence.json', 'source_hashes.json', 'frozen_manifest.json', 'validation.json', 'status.json', 'v_models_manifest_B_noobj.json',
             'selection/selection_secondary_calibration.json', 'results/w_eval.json', 'labels/labels_manifest.json', 'results/q_target_dependence.json',
             'contracts/v_model_checks_full.json', 'tables/agreement.csv', 'tables/strata_h90.csv', 'tables/objectives_h90_overlapping.csv',
             'tables/horizon_sign_disagreement.csv', 'tables/q_target_dependence.csv', 'integrity/parent_hashes_after.json', 'integrity/parent_inventory_after.json']
    R += table(['파일', 'sha256(앞 16자)'], [[fn, C.sha256_file(O / fn)[:16]] for fn in files if (O / fn).exists()])
    R += ['', '세트별 전체 결과 `results/per_set/*.json`, 예측·라벨 배열 `eval/w_predictions/*.npz`, `labels/*_B_noobj_labels.npz`, 모델 `models/B_noobj/`. 해시는 보고서 작성 시점 값이다.', '']
    (O / 'REPORT.md').write_bytes(('\n'.join(R) + '\n').encode('utf-8'))

    D = ['# 정의와 근거 (objective_channel_ablation_20260915)', '', '근거 종류: L=문헌, D=게임 규칙/구현, E=우리 데이터 검증, A=설계 선택 또는 상속 정의.', '']
    D += table(['용어', '정의', '근거', '비고'], [
        ['W', 'GAME_END winningTeam의 최종 Blue 승리', 'D', 'V 학습 목표'],
        ['V_A', 'expanded StateV2 logistic C=.01, raw, 최종+5 OOF', 'A(부모)', '변경 없음'],
        ['V_B_noobj', 'V_A 입력에서 이름에 baron/elder/dragon/soul/herald/horde/atakhan이 포함된 176열 제거(185열 유지), 나머지 동일, raw', 'A', '명시 채널 절제, 인과 절제 아님'],
        ['보조 보정', 'V_CAL sigmoid 대 raw를 V_SELECT로 선택', 'A', '선택: raw'],
        ['Δ, Y', 'V(e 상태) − V(q_pre 상태), 같은 어댑터; Y=1[Δ>0]', 'A(상속)', f'정확한 0: {zeros}'],
        ['불일치', 'Y_A ≠ Y_B, 행·경기가중; 경기 bootstrap 1,000회', 'A', ''],
        ['same_pre_post_frame', '사전·종료 최신 프레임 시각 동일', 'A', 'L 이후 새 프레임 없음보다 좁음'],
        ['full / after 창', '(q_pre, e] / (L, e], 하한 제외·상한 포함', 'D/A', 'TRAIN 원시 사건으로 이전 연구에서 검증(E)'],
        ['획득 팀 기준 Δ', 'Blue 단독 +Δ, Red 단독 −Δ', 'A', '기술 통계, 인과 가치 아님'],
        ['unknown_objective_team_count', UNKNOWN_V2, 'D', '유지. 정정 사항은 errata.json'],
        ['q 표적 의존성', '동결 specialist를 Y_B로 채점', 'A', '재학습 없음']])
    D += ['', '## 문헌', '',
          '- Maymin (2021), Smart kills and worthless deaths, doi:10.1515/jqas-2019-0096 — 상태 승률 변화 가치평가(L).',
          '- Kim, Lee & Chung, IEEE CoG 2020, https://ieee-cog.org/2020/papers/paper_221.pdf — 승률 확률 품질(L). 재현 아님.',
          '- Jacobs & Wallach, Measurement and Fairness, arXiv:1912.05511 — 구성 개념과 측정(L).',
          '- 부모 근거 장부: outputs/label_validity_full_20260915/DEFINITION_AND_EVIDENCE.md, docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md.', '',
          '## 주장하지 않는 것', ''] + [f'- {c}' for c in proto['claims_not_made']] + ['']
    (O / 'DEFINITION_AND_EVIDENCE.md').write_bytes(('\n'.join(D) + '\n').encode('utf-8'))

    Pt = ['# Publication tables — objective_channel_ablation_20260915', '', 'Exploratory after prior test exposure; named objective-channel ablation, not causal. Values from results/*.json.', '',
          '## Table 1. W prediction, raw A vs raw B_noobj (equal-match weighted)', '']
    Pt += table(['Set', 'Rows', 'AUC A', 'AUC B', 'Brier A', 'Brier B', 'Log loss A', 'Log loss B'],
                [[n, s['rows'], f(s['A_raw']['match_weighted']['auc']), f(s['B_noobj_raw']['match_weighted']['auc']), f(s['A_raw']['match_weighted']['brier']),
                  f(s['B_noobj_raw']['match_weighted']['brier']), f(s['A_raw']['match_weighted']['logloss']), f(s['B_noobj_raw']['match_weighted']['logloss'])] for n, s in w['sets'].items()])
    Pt += ['', '## Table 2. MAIN_TEST bootstrap B minus A (1000 match resamples, fixed models)', '']
    Pt += table(['Weighting', 'Metric', 'Estimate', '95% interval'], [[wn, m, f(v['estimate'], 5), f"[{f(v['ci95'][0], 5)}, {f(v['ci95'][1], 5)}]"] for wn, d in bw.items() for m, v in d.items()])
    Pt += ['', '## Table 3. h90 sign disagreement A vs B_noobj (match-weighted, 95% match bootstrap)', '']
    Pt += table(['Set', 'Cohort', 'Rows', 'Matches', 'Disagreement'], [[n, c, per[n]['agreement']['h90'][c]['rows'], per[n]['agreement']['h90'][c]['matches'], ci(dis[n][c]['match_weighted'])]
                                                                      for n in SETS for c in ('E', 'T', 'N')])
    Pt += ['', '## Table 4. MAIN_TEST h90 T single-team acquisitions: acquiring-team-oriented delta (match-weighted; descriptive)', '']
    Pt += table(['Window', 'Objective', 'Rows', 'Disagreement', 'Oriented dA', 'Oriented dB', 'A-B'],
                [[wn, o, obT[wn][o]['single_team_oriented']['rows'], pct(obT[wn][o]['single_team_oriented']['disagreement']['row']),
                  f(obT[wn][o]['single_team_oriented']['oriented_delta_A']['match_weighted'], 4), f(obT[wn][o]['single_team_oriented']['oriented_delta_B']['match_weighted'], 4),
                  f(obT[wn][o]['single_team_oriented']['oriented_delta_A_minus_B']['match_weighted'], 4)]
                 for wn in ('after_(L,e]', 'full_(q_pre,e]') for o in ('baron', 'elder', 'atakhan', 'dragon', 'soul_owned', 'herald', 'horde')])
    Pt += ['', '## Table 5. Frozen specialist q scored against A and B_noobj labels (MAIN_TEST h90)', '']
    Pt += table(['Cohort', 'q', 'AUC Y_A', 'AUC Y_B', 'Brier Y_A', 'Brier Y_B'], [[c, v['specialist'], f(v['scores']['Y_A']['auc']), f(v['scores']['Y_B']['auc']),
                                                                               f(v['scores']['Y_A']['brier']), f(v['scores']['Y_B']['brier'])] for c, v in qd['cohorts'].items()])
    (O / 'tables' / 'PUBLICATION_TABLES.md').write_bytes(('\n'.join(Pt) + '\n').encode('utf-8'))
    st.update('complete', 'report', report_sha256=C.sha256_file(O / 'REPORT.md'), next_step='done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
