"""Label validity stage R: REPORT.md (Korean), DEFINITION_AND_EVIDENCE.md, tables/PUBLICATION_TABLES.md.

All numbers are read from result files written by earlier stages; narrative direction statements are guarded by
assertions so the text cannot silently contradict the computed results. Run after post-run checks.
"""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import lv20260915_common as K  # noqa: E402

import math  # noqa: E402
import time  # noqa: E402

O = K.OUT
SETS = list(K.LABEL_SETS)
EXT_SETS = [s for s in SETS if s.startswith('EXT_')]


def f(x, d=4):
    return '-' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def pct(x, d=2):
    return '-' if x is None else f'{100 * x:.{d}f}%'


def ci(e, d=2):
    return f"{pct(e['estimate'], d)} [{pct(e['ci95'][0], d)}, {pct(e['ci95'][1], d)}]"


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return out


def main():
    st = K.Status('report')
    K.log_command()
    val = C.read_json(O / 'validation.json')
    proto = C.read_json(O / 'protocol.json')
    fz = C.read_json(O / 'frozen_manifest.json')
    ve = C.read_json(O / 'results' / 'v_eval.json')
    per = {n: C.read_json(O / 'results' / 'per_set' / f'{n}.json') for n in SETS}
    qd = C.read_json(O / 'results' / 'q_label_dependence.json')
    ph = C.read_json(O / 'results' / 'posthoc_mechanism_B_econ_stale_frames.json')
    pk = C.read_json(O / 'results' / 'review_packet_selection.json')
    fx = C.read_json(O / 'contracts' / 'train_fixture_checks.json')
    lm = C.read_json(O / 'alt_labels' / 'alt_labels_manifest.json')
    mans = {M: C.read_json(O / f'v_models_manifest_{M}.json') for M in K.MODELS}
    sels = {M: C.read_json(O / 'selection' / f'selection_{M}.json') for M in K.MODELS}
    amend = C.read_json(O / 'protocol_amendments.json')
    status = C.read_json(O / 'status.json')

    # ------------------------------------------------ guarded claims
    assert val['all_pass'], 'report requires passing validation'
    mt = ve['sets']['MAIN_TEST']['by_model']
    assert mt['A']['overall']['logloss'] < mt['B_reg']['overall']['logloss'] < mt['B_econ']['overall']['logloss']
    boot_v = {p['b']: p for p in ve['sets']['MAIN_TEST']['bootstrap_loss_differences']['pairs']}
    assert boot_v['B_reg']['logloss']['ci95'][1] < 0 and boot_v['B_econ']['logloss']['ci95'][1] < 0
    dis = {n: {c: per[n]['bootstrap_h90'][c]['estimates'] for c in ('E', 'T', 'N')} for n in SETS}
    nonpilot = [n for n in SETS if n != 'EXT_KR_16.14_pilot']
    assert all(dis[n][c]['A_vs_B_reg']['match_weighted']['estimate'] < .03 for n in SETS for c in ('E', 'T', 'N'))
    assert all(.15 < dis[n][c]['A_vs_B_econ']['match_weighted']['estimate'] < .35 for n in nonpilot for c in ('E', 'T', 'N'))
    assert all(dis[n]['T']['A_vs_B_econ']['match_weighted']['estimate'] < dis[n]['N']['A_vs_B_econ']['match_weighted']['estimate'] for n in nonpilot)
    assert ph['all_sign_identity']
    tt = per['MAIN_TEST']['strata_h90']['T']['abs_delta_A']
    assert tt['[0,.005]']['disagree_A_B_reg']['row'] > tt['>.02']['disagree_A_B_reg']['row']
    exact_zero_any = sum(v for s in lm['summaries'].values() for h in K.HS for v in s[f'h{h}']['exact_zero_delta'].values())
    obsT = per['MAIN_TEST']['observed']['h90']['T']
    assert obsT['interval_definitions']['kill_diff_during_equals_full']
    qT = qd['sets']['T']['columns']['spec_ridge_raw']
    k15 = ve['sets']['EXT_KR_16.15']['by_model']
    assert k15['B_reg']['overall']['logloss'] < k15['A']['overall']['logloss']
    assert all(s['primary_identity']['A_predictions_bitwise_equal_parent'] for s in ve['sets'].values() if 'primary_identity' in s)
    assert all(r['p_pre_bitwise_equal'] and all(r[f'h{h}_p_post_bitwise_equal'] and r[f'h{h}_Y_equal'] for h in K.HS)
               for r in (s['identity']['A_recomputation'] for s in lm['summaries'].values()))
    assert all(len(mans[M]['final_fit']['attempts']) == 1 and all(len(v['fit']['attempts']) == 1 for v in mans[M]['folds'].values()) for M in K.MODELS)
    e_unch = per['MAIN_TEST']['agreement']['h90']['E']['A_vs_B_econ']
    assert e_unch['endpoint_unchanged_60_90_120']['disagreement_row'] > e_unch['endpoint_changed_across_horizons']['disagreement_row']
    for n in nonpilot:
        assert .17 < ph['sets'][n]['T']['disagree_A_B_econ_nonstale'] < .21 and .23 < ph['sets'][n]['N']['disagree_A_B_econ_nonstale'] < .26, n
    for c in ('E', 'T'):
        sp = per['MAIN_TEST']['strata_h90'][c]['post_snapshot_after_L']
        assert sp['0']['disagree_A_B_econ']['row'] > sp['1']['disagree_A_B_econ']['row']
    assert qd['sets']['N']['columns']['spec_p_pre_logistic']['Y_B_econ']['auc'] > qd['sets']['N']['columns']['spec_p_pre_logistic']['Y_A']['auc']
    assert qd['sets']['E']['columns']['p_pre_logistic']['Y_B_econ']['auc'] > qd['sets']['E']['columns']['p_pre_logistic']['Y_A']['auc']
    assert obsT['kills']['blue_more_kills']['P_Y1_A']['match_weighted'] > .8 and abs(obsT['kills']['tie']['P_Y1_A']['match_weighted'] - .5) < .05
    small_share = {}
    for c in ('E', 'T'):
        bins = per['MAIN_TEST']['strata_h90'][c]['abs_delta_A']
        tot = sum(b['disagree_A_B_reg']['row'] * b['rows'] for b in bins.values() if not b.get('empty'))
        small_share[c] = sum(bins[b]['disagree_A_B_reg']['row'] * bins[b]['rows'] for b in ('[0,.005]', '(.005,.01]') if b in bins) / tot

    now = time.strftime('%Y-%m-%d %H:%M:%S')
    R = []
    R += ['# 전체 코퍼스 라벨 타당성 진단 보고서 (label_validity_full_20260915)', '',
          f'작성: {now} KST, Claude Opus 5 구현·실행 / 설계: Codex (`docs/CLAUDE_LABEL_VALIDITY_FULL_20260915.md`).', '',
          '> **판정:** 계획된 A–D 단계를 전체 코퍼스(21만 경기의 지정 패치 역할 + 외부 4세트)에서 끝까지 실행했고, 사후 검증 '
          f"{val['passed']}/{val['total']}개가 통과했다. 이 결과는 **사람 검토와 표적 후속 연구로 넘어갈 준비가 되었다**는 뜻이다. "
          '**라벨이 실제 한타 승리를 타당하게 측정한다는 검증은 아니다.** 전문가 검토는 수행되지 않았다(UNPERFORMED).', '',
          '## 0. 사전 노출과 연구 성격', '',
          '- 이 연구는 **탐색적 후속 진단**이다. 새 확증 검정이 아니다. protocol.json 작성 전에 이미 본 결과: 부모 전체 학습의 V·q TEST/외부 지표, '
          '코호트·역할 결과, ToG 감사의 재계산(주 TEST h90 라벨 분포, |Δ| 비율, 상한 간 부호 반전, T/N 지표).',
          f"- protocol.json: {proto['written_at']} (sha256 `{C.sha256_file(O / 'protocol.json')[:16]}…`). 대안 V 적합(스모크 포함)보다 먼저 작성했다.",
          f"- frozen_manifest.json: {fz['frozen_at']}. 동결 전 결과 W 접근은 TRAIN fold·V_CAL·V_SELECT뿐이었고(기록 {len(fz['pre_freeze_outcome_access'])}건), "
          'TEST/외부 생성 라벨은 열지 않았다.',
          '- 주 V(A: expanded logistic C=.01, raw), 기존 OOF 어댑터, 주 라벨, E/T/N 규칙, cutoff s−1ms, h90 주 상한과 h60/h120은 바꾸지 않았다. '
          'q는 다시 학습하지 않았다. 새 h·특징·모델을 TEST 성능으로 고르지 않았다.',
          '- 사후 추가 분석은 1건이다(PH1, `protocol_amendments.json`). 결과를 본 뒤 해석을 위해 추가했으며 라벨·모델·선택은 바꾸지 않았다.', '']
    R += ['## 1. 실행 전(Before)과 실행 후(After)', '']
    R += table(['항목', 'Before (이 실행 전)', 'After (이 실행 후)'], [
        ['주 라벨', '고정: ΔV_A 부호, h90 사건 종료', '변경 없음. 7개 세트·3상한에서 A를 다시 계산해 저장값과 비트 단위로 일치'],
        ['대안 V에 대한 안정성', '미확인', f"B_reg(C=.1) 불일치 약 {pct(dis['MAIN_TEST']['E']['A_vs_B_reg']['match_weighted']['estimate'], 1)}; "
                                     f"B_econ(경제 스냅샷) 불일치 약 {pct(dis['MAIN_TEST']['E']['A_vs_B_econ']['match_weighted']['estimate'], 1)} (주 TEST h90 E)"],
        ['관측 사건·자원과의 방향', '체계적 표 없음', '킬·골드·경험치·오브젝트 방향별 P(Y=1), 작은 Δ 비율, A/B 불일치를 전체 세트에서 표로 산출'],
        ['의미 타당성(전문가 판정)', '없음', '블라인드 120사례 패킷만 구축. 판정은 수행하지 않음(UNPERFORMED)'],
        ['q의 라벨 의존성', '미확인', 'A 라벨로 학습된 저장 q를 대안 Y로 채점한 진단표(순위·선택 없음)']])
    R += ['', '## 2. 실행 기록', '',
          '- 순서: protocol → 합성 계약 17개(pytest) → TRAIN 원시 캐시 고정 사례 계약 → TRAIN-only 스모크 적합(두 모델) → 독립 V 모델 점검(스모크) '
          '→ 전체 적합(B_reg, B_econ 병렬 2프로세스) → 독립 V 모델 점검(전체) → 동결 → W 평가 / 대안 라벨(병렬) → 리뷰 패킷 → 분석 → 사후 메커니즘(PH1) → 사후 검증 → 보고서.',
          '- 자원: 단일 스레드 프로세스 최대 2개 동시 실행(BLAS/OMP 스레드 1), GPU 미사용, 설치·삭제·커밋·외부 메시지 없음. 명령 전체는 `commands.txt`, 로그는 `logs/`.',
          '- **실패한 실행(보존):** 사후 검증 1차는 외부 raw 폴더 경로 필드를 잘못된 manifest에서 찾아 `KeyError: raw_folder`로 중단됐다. preflight CSV에서 읽도록 고쳤다. '
          '2차는 16/17로, V 모델 점검과 동결이 같은 초(16:02:17)에 실행되어 엄격한 `<` 시각 비교가 실패했다. 동결 manifest에 기록된 점검 파일 해시 비교로 바꿨고, 이것이 더 강한 조건이다. '
          '3차에서 17/17 통과. 로그는 `logs/postrun_checks_stdout_run{1,2,3}.txt`에 있다.',
          f"- 부모 무결성: 읽은 부모 파일 {val['checks']['parent_artifacts_unchanged']['hashed_files']}개의 sha256과 두 부모 루트의 파일 {val['checks']['parent_artifacts_unchanged']['inventory_files']}개(크기·mtime)가 전후 동일하다. "
          '부모의 outcome/label 접근 로그에도 새 기록이 없다. 새 접근 기록은 이 루트의 `outcome_access_log.jsonl`, `label_access_log.jsonl`에만 있다.', '']
    # ------------------------------------------------ A
    R += ['## 3. A. 고정된 대안 가치 모델 두 개', '',
          '- **B_reg:** A와 같은 expanded 특징 361개, 같은 전처리와 logistic, **C=.1**. 정규화 민감도를 보려는 것이며 새 구조를 주장하지 않는다.',
          '- **B_econ:** time_minutes, time_minutes_sq, 슬롯별 totalGold/curGold/level/xp/laneCS/jgCS 정규화값(60개), 챔피언 ID 10개 one-hot. C=.01. '
          '킬·사망·생존·체력/마나·오브젝트·구조물·이력·경과시간 채널은 없다. **일부러 좁힌 스냅샷 경제 비교 모델이다.** 오브젝트 이득은 이후 관측된 자원을 통해 들어올 수 있으므로 '
          '"오브젝트 없는 인과 절제"라고 부르지 않는다.',
          '- 적합: A의 TRAIN 버킷 질의 키와 정확히 같은 424,160행 / 74,168경기, 경기별 동일 가중치. 기존 5-fold, V_CAL에서 raw 대 양의 기울기 sigmoid, '
          'V_SELECT 선택(log loss → Brier → 이름), 선택 계열을 fold 어댑터 전체에 적용. 격자 탐색 없음.', '']
    rows = []
    for M in K.MODELS:
        mf = mans[M]
        rows.append([M, mf['spec']['C'] if 'spec' in mf else K.MODEL_SPECS[M]['C'], len(mf['feature_names']), sels[M]['chosen'],
                     f(sels[M]['select_scores']['raw']['logloss'], 6), f(sels[M]['select_scores']['sigmoid_pos']['logloss'], 6),
                     f"{mf['final_fit']['n_iter']} ({'수렴' if mf['final_fit']['converged'] else '미수렴'})",
                     ', '.join(str(v['fit']['n_iter']) for v in mf['folds'].values()),
                     f"{sum(a['seconds'] for a in mf['final_fit']['attempts'])}s"])
    R += table(['모델', 'C', '입력 열', '선택 보정', 'V_SELECT LL raw', 'V_SELECT LL sigmoid', '최종 n_iter', 'fold n_iter', '최종 적합 시간'], rows)
    R += ['', f"B_reg는 sigmoid가 raw보다 V_SELECT log loss에서 {f(sels['B_reg']['select_scores']['raw']['logloss'] - sels['B_reg']['select_scores']['sigmoid_pos']['logloss'], 6)} 낮아 선택됐다. 차이는 매우 작지만 규칙대로 따랐다. "
          '반복 상한을 올릴 필요는 없었다(모든 적합이 1500회 안에 수렴).', '',
          '### W(최종 Blue 승리) 예측 품질: 버킷 질의 키', '']
    rows = []
    for n, s in ve['sets'].items():
        for m in K.ALL_V:
            o = s['by_model'][m]['overall']
            rows.append([n, m, s['rows'], f(o['auc']), f(o['brier']), f(o['logloss']), f(o.get('slope'), 3), f(o.get('ece_10bin'), 4)])
    R += table(['세트', 'V', '질의 행', 'AUC', 'Brier', 'log loss', '보정 기울기', 'ECE'], rows)
    tb = ve['sets']['MAIN_TEST']['by_model']
    R += ['', f"주 TEST 경기 bootstrap(1,000회, 모델 고정): A−B_reg log loss {f(boot_v['B_reg']['logloss']['estimate_a_minus_b'], 5)} "
              f"[{f(boot_v['B_reg']['logloss']['ci95'][0], 5)}, {f(boot_v['B_reg']['logloss']['ci95'][1], 5)}], Brier {f(boot_v['B_reg']['brier']['estimate_a_minus_b'], 5)}; "
              f"A−B_econ log loss {f(boot_v['B_econ']['logloss']['estimate_a_minus_b'], 5)} [{f(boot_v['B_econ']['logloss']['ci95'][0], 5)}, {f(boot_v['B_econ']['logloss']['ci95'][1], 5)}], "
              f"Brier {f(boot_v['B_econ']['brier']['estimate_a_minus_b'], 5)}. 2–10분 AUC: A {f(tb['A']['time_bands']['2-10']['auc'])}, B_reg {f(tb['B_reg']['time_bands']['2-10']['auc'])}, "
              f"B_econ {f(tb['B_econ']['time_bands']['2-10']['auc'])}. <2분 구간은 V 격자가 2분에 시작하므로 비어 있다(n=0).",
          '', '**해석:** 주 TEST에서 W 예측은 A가 가장 좋고, B_reg는 A와 매우 가깝고, B_econ은 뚜렷이 나쁘다. 외부 KR 16.15(작은 세트)에서는 B_reg가 A보다 약간 좋다. '
              '**훨씬 나쁜 V와의 불일치는 A가 틀렸다는 증거가 아니다.** 이 표는 아래 불일치를 읽는 맥락이다. 주 A 예측은 9개 세트 모두에서 부모 저장값과 비트 단위로 같았다.', '']
    # ------------------------------------------------ B agreement
    R += ['## 4. B. 라벨 일치: A 대 B', '',
          'Y=1[Δ>0], 정확히 0이면 0. 같은 종료점과 같은 어댑터를 전후에 사용했다. 불일치 = Y_A≠Y_B. 구간은 경기 단위 1,000회 대응 bootstrap의 백분위 95%이며, '
          f"모델은 고정했고 학습 불확실성은 포함하지 않는다. 모든 세트·상한·모델에서 정확한 Δ=0은 {exact_zero_any}건이었다.", '']
    rows = []
    for n in SETS:
        for c in ('E', 'T', 'N'):
            a = per[n]['agreement']['h90'][c]
            rows.append([n, c, a['A_vs_B_reg']['rows'], a['A_vs_B_reg']['matches'], ci(dis[n][c]['A_vs_B_reg']['match_weighted']),
                         pct(a['A_vs_B_reg']['disagreement_row']), ci(dis[n][c]['A_vs_B_econ']['match_weighted']), pct(a['A_vs_B_econ']['disagreement_row']),
                         f"{pct(a['A_vs_B_reg']['positive_rate_A_match_weighted'], 1)} / {pct(a['A_vs_B_reg']['positive_rate_B_match_weighted'], 1)} / {pct(a['A_vs_B_econ']['positive_rate_B_match_weighted'], 1)}",
                         f"{f(a['A_vs_B_reg']['delta_spearman'], 3)} / {f(a['A_vs_B_econ']['delta_spearman'], 3)}",
                         '예' if a['A_vs_B_reg']['sparse_lt30_matches'] else ''])
    R += table(['세트', '코호트', '행', '경기', 'A≠B_reg 경기가중 [95%]', 'A≠B_reg 행', 'A≠B_econ 경기가중 [95%]', 'A≠B_econ 행', 'P(Y=1) A/B_reg/B_econ', 'Δ Spearman reg/econ', '<30경기'], rows)
    mtE = per['MAIN_TEST']['agreement']['h90']['E']
    R += ['', f"- 종료점이 60/90/120에서 모두 같은 행과 달라지는 행을 나눠 보면(주 TEST E): A≠B_reg는 {pct(mtE['A_vs_B_reg']['endpoint_unchanged_60_90_120']['disagreement_row'])} / "
              f"{pct(mtE['A_vs_B_reg']['endpoint_changed_across_horizons']['disagreement_row'])}, A≠B_econ은 {pct(mtE['A_vs_B_econ']['endpoint_unchanged_60_90_120']['disagreement_row'])} / "
              f"{pct(mtE['A_vs_B_econ']['endpoint_changed_across_horizons']['disagreement_row'])}. B_econ 불일치는 60초 상한 전에 다른 종료 사건으로 끝나 세 상한의 종료점이 같은 행에서 더 크다.", '']
    hz = per['MAIN_TEST']['horizon']
    rows = []
    for m in K.ALL_V:
        for p in ('60_vs_90', '90_vs_120', '60_vs_120'):
            cc = hz[m]['E'][p]
            rows.append([m, p, pct(cc['all']['row']), cc['changed_endpoint']['rows'], pct(cc['changed_endpoint']['row']), cc['unchanged_endpoint']['rows'], cc['unchanged_endpoint']['disagreements']])
    R += ['상한 간 부호 불일치(주 TEST E, 모델별):', '']
    R += table(['V', '상한 쌍', '전체 행', '종료점 달라진 행', '그 행의 불일치', '종료점 같은 행', '같은 행 불일치 수(0이어야 함)'], rows)
    R += ['', f"**요약:** 정규화만 바꾼 B_reg에서는 라벨 부호가 거의 유지된다(모든 세트·코호트에서 경기가중 {math.ceil(1000 * max(dis[n][c]['A_vs_B_reg']['match_weighted']['estimate'] for n in SETS for c in ('E', 'T', 'N'))) / 10:.1f}% 이하). "
              f"스냅샷 경제 모델 B_econ에서는 비파일럿 세트에서 {pct(min(dis[n][c]['A_vs_B_econ']['match_weighted']['estimate'] for n in nonpilot for c in ('E', 'T', 'N')), 0)}–{pct(max(dis[n][c]['A_vs_B_econ']['match_weighted']['estimate'] for n in nonpilot for c in ('E', 'T', 'N')), 0)}가 뒤집힌다(KR 16.14 pilot E/N은 더 크고 구간이 넓다). "
              'T가 N보다 덜 뒤집힌다(모든 비파일럿 세트). 즉 **라벨은 같은 계열 안의 정규화에는 안정적이지만, 가치 모델이 무엇을 상태로 보느냐에는 크게 의존한다.** '
              '이것은 V가 라벨 정의의 일부라는 뜻이며, 어느 쪽이 "참" 라벨인지는 이 수치로 정할 수 없다.', '']
    # strata
    R += ['## 5. h90 층화 진단 (주 TEST)', '', '진단용 층이다. 라벨 제외 기준도, V의 오차범위도 아니다.', '']
    for c in ('E', 'T'):
        rows = []
        for fam in ('abs_delta_A', 'post_snapshot_after_L', 'post_frame_age_s', 'ending_reason', 'start_minutes', 'p_pre_A', 'fine_scale'):
            for b, cell in per['MAIN_TEST']['strata_h90'][c][fam].items():
                if cell.get('empty'):
                    continue
                rows.append([fam, b, cell['rows'], cell['matches'], pct(cell['P_Y1_A']['match_weighted'], 1), pct(cell['disagree_A_B_reg']['row']),
                             pct(cell['disagree_A_B_econ']['row']), f(cell.get('mean_abs_delta_diff_B_reg'), 4), f(cell.get('mean_abs_delta_diff_B_econ'), 4),
                             '예' if cell['sparse_lt30_matches'] else ''])
        R += [f'### 코호트 {c}', '']
        R += table(['층', '구간', '행', '경기', 'P(Y_A=1)', 'A≠B_reg', 'A≠B_econ', '평균|ΔB_reg−ΔA|', '평균|ΔB_econ−ΔA|', '<30경기'], rows)
        R += ['']
    R += [f"- |Δ_A|≤.005 구간(T)의 A≠B_reg는 {pct(tt['[0,.005]']['disagree_A_B_reg']['row'])}이고, |Δ_A|>.02에서는 {pct(tt['>.02']['disagree_A_B_reg']['row'])}이다. "
          f"A≠B_reg 행 가운데 |Δ_A|≤.01인 행의 비율은 E {pct(small_share['E'], 1)}, T {pct(small_share['T'], 1)}이다. B_reg와의 불일치는 대부분 작은 Δ 행에서 생긴다.",
          '- L 이후 새 프레임이 없는 행(post_snapshot_after_L=0)에서 B_econ 불일치가 크다. 원인의 일부(사전·종료 프레임이 같은 행)는 6절에서 확인했다.', '']
    # posthoc
    R += ['## 6. 사후 메커니즘 확인 (PH1, 사전 명시 아님)', '',
          '결과를 본 뒤 추가했다. 사전 프레임과 종료 프레임이 같은 행(stale_same_frame)에서는 B_econ 입력 가운데 시간 두 열만 변한다. '
          '그러면 Δ_B_econ의 부호는 어댑터 계수의 시간 항만으로 정해져야 한다. 동결된 계수로 이 부호를 해석적으로 계산해 저장 라벨과 비교했다.', '']
    rows = []
    for n in SETS:
        for c in ('E', 'T', 'N'):
            v = ph['sets'][n][c]
            rows.append([n, c, v['stale_rows'], pct(v['stale_share'], 1), '예' if v['sign_identity_all'] else '아니오', pct(v['P_Y1_B_econ_stale'], 1),
                         pct(v['P_Y1_A_stale'], 1), pct(v['disagree_A_B_econ_stale'], 1), pct(v['disagree_A_B_econ_nonstale'], 1), pct(v['disagree_A_B_reg_nonstale'], 2)])
    R += table(['세트', '코호트', '같은 프레임 행', '비율', '시간항 부호 일치(전 행)', 'P(Y_Becon=1) 같은 프레임', 'P(Y_A=1) 같은 프레임', 'A≠B_econ 같은 프레임', 'A≠B_econ 다른 프레임', 'A≠B_reg 다른 프레임'], rows)
    R += ['', '**해석:** 같은 프레임 행에서는 B_econ 라벨이 게임 시간만의 함수다(모든 세트에서 부호 일치 100%). 이것은 좁은 스냅샷 비교 모델의 성질이며 교전 결과에 대한 증거가 아니다. '
              '다른 프레임 행에서도 A≠B_econ은 비파일럿 세트에서 T 약 18–20%, N 약 24–25%로 남는다. 따라서 B_econ과의 불일치를 프레임 부재만으로 설명할 수는 없다.', '']
    # observed
    R += ['## 7. 관측 사건·자원 방향과의 비교 (정답 아님)', '',
          '- **구간 정의:** during=(q_pre, L], full=(q_pre, e], q_pre=s−1ms. 킬은 StateV2의 팀 킬 크레딧(killerId가 그 팀 로스터)이다. 종료 규칙상 (L, e]에는 원시 챔피언 킬이 없다. '
          f"그래서 두 구간의 킬 차이는 모든 유효 행에서 같고, 행별로 검증했다(주 TEST T: (L,e] 킬 행 {obsT['interval_definitions']['kills_after_L_rows']}). "
          f"크레딧 없는 킬(처형 등)이 있는 행은 {obsT['uncredited_kill_rows']}개다(주 TEST T).",
          '- **자원 단위:** 팀 합 차이(Blue−Red)의 변화이며 단위는 정규화 캐시값이다. 출처는 `gameplay/pipeline_cache.py`: totalGold/25000, xp/20000, float32, clip [0,5]. '
          f"TRAIN 40경기 {fx['gold_team_minute']['frames']}프레임에서 합×25000이 원시 팀 골드와 최대 {f(fx['gold_team_minute']['max_abs_diff_gold'], 4)} 골드 차이였다. "
          f"외부 KR 16.15 원시 타임라인 5경기에서도 정규화값×분모와 원시 값의 최대 차이가 {f(max(val['checks']['normalization_source_traced']['external_raw']['max_abs_diff'].values()), 4)} 이하였다. "
          '표의 "캐시 골드"는 이 검증을 근거로 한 환산값이다. 프레임은 약 1분 간격이므로 사전 프레임과 종료 프레임 사이의 변화만 관측된다. 같은 프레임이면 "stale_same_frame"으로 따로 두고, 0(동률)으로 취급하지 않는다.', '']
    for c in ('T', 'E'):
        o = per['MAIN_TEST']['observed']['h90'][c]
        rows = []
        for var, cats in (('kills', o['kills']), ('gold', o['gold']), ('xp', o['xp'])):
            for cat, cell in cats.items():
                if cell.get('empty'):
                    continue
                rows.append([var, cat, cell['rows'], pct(cell['P_Y1_A']['match_weighted'], 1), pct(cell['P_Y1_B_reg']['match_weighted'], 1), pct(cell['P_Y1_B_econ']['match_weighted'], 1),
                             pct(cell['small_abs_delta_A_le_0.01']['row'], 1), pct(cell['disagree_A_B_reg']['row']), pct(cell['disagree_A_B_econ']['row'])])
        R += [f'### 주 TEST h90 코호트 {c}: 관측 방향별 라벨', '']
        R += table(['변수', '관측 방향', '행', 'P(Y_A=1)', 'P(Y_Breg=1)', 'P(Y_Becon=1)', '|Δ_A|≤.01', 'A≠B_reg', 'A≠B_econ'], rows)
        R += ['', '킬 방향 × 골드 범주 행 수: ' + '; '.join(f"{k}: " + ', '.join(f'{g2}={n2}' for g2, n2 in d.items()) for k, d in o['kill_by_gold'].items()),
              f"비-stale 골드 변화 분위수(5/25/50/75/95%, 캐시 골드): {', '.join(f(x, 0) for x in o['gold_change_cache_gold_quantiles_nonstale'])}", '']
    R += ['**읽는 법:** Blue 킬이 더 많은 행에서 P(Y_A=1)이 높고 동률 행은 약 50%다. 이것은 V_A가 킬 이력을 상태로 쓰기 때문에 기대되는 연관이지 독립적 정답 확인이 아니다. '
          '반대 방향 행(Blue 킬이 많은데 Y_A=0 등)은 오류 판정이 아니라 사람 검토 대상 후보다. 이런 휴리스틱을 정답으로 한 "정확도"는 계산하지 않았다.', '']
    ob = per['MAIN_TEST']['objectives_h90']['cohorts']['T']
    rows = []
    for w in ('after_(L,e]', 'full_(q_pre,e]'):
        for obj in ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan', 'soul_owned'):
            for sub in ('blue_only', 'red_only', 'both_teams', 'unknown_team_present_DIAGNOSTIC'):
                cell = ob[w][obj].get(sub, {})
                if not cell or cell.get('empty'):
                    continue
                rows.append([w, obj, sub, cell['rows'], cell['matches'], pct(cell['P_Y1_A']['match_weighted'], 1), pct(cell['P_Y1_B_econ']['match_weighted'], 1),
                             pct(cell['small_abs_delta_A_le_0.01']['row'], 1), pct(cell['disagree_A_B_reg']['row']), pct(cell['disagree_A_B_econ']['row']),
                             '예' if cell['sparse_lt30_matches'] else ''])
    R += ['### 주 TEST h90 T: 오브젝트 획득(겹치는 부분집합, 합산 금지)', '']
    R += table(['구간', '오브젝트', '팀', '행', '경기', 'P(Y_A=1)', 'P(Y_Becon=1)', '|Δ_A|≤.01', 'A≠B_reg', 'A≠B_econ', '<30경기'], rows)
    R += ['', '원소별 드래곤의 팀은 카운터에 없다. full 구간의 원소별 팀은 StateV2 차이로 산출했다(`tables/objectives_h90_overlapping_subgroups.csv`). '
          f"카운터 팀 합과 StateV2 팀 합이 다른 행: {ob['full_(q_pre,e]']['dragon_team_counter_vs_state_mismatch_rows']}. teamId 0 영혼과 팀 미상 오브젝트는 진단값으로만 둔다. "
          '외부 16.xx 세트에는 아타칸 사건이 없다(부모 감사). 고정 골드 교환비나 오브젝트 인과 효과는 추정하지 않았다.', '']
    # q
    R += ['## 8. 저장된 q의 라벨 의존성 (주 TEST h90, 진단)', '',
          'A 라벨로 학습된 q 예측을 그대로 두고 채점 라벨만 바꿨다. 새 모델 순위, 선택, 성능 상한 주장이 아니다. Y_A 점수는 감사값(T/N: calculations.json, E: results_q.json)과 1e−12 이내로 재현됐다.', '']
    rows = []
    for c, cell in qd['sets'].items():
        for col, cc in cell['columns'].items():
            if col in ('spec_constant', 'constant'):
                continue
            chosen = col in (cell['chosen'], 'spec_' + cell['chosen'])
            rows.append([c, col + (' (선택)' if chosen else ''), f(cc['Y_A']['auc']), f(cc['Y_B_reg']['auc']), f(cc['Y_B_econ']['auc']),
                         f(cc['Y_A']['brier']), f(cc['Y_B_reg']['brier']), f(cc['Y_B_econ']['brier'])])
    R += table(['코호트', 'q 예측', 'AUC vs Y_A', 'AUC vs Y_Breg', 'AUC vs Y_Becon', 'Brier vs Y_A', 'Brier vs Y_Breg', 'Brier vs Y_Becon'], rows)
    R += ['', f"T 선택 specialist(ridge_raw)의 AUC는 Y_A {f(qT['Y_A']['auc'])} → Y_B_reg {f(qT['Y_B_reg']['auc'])} → Y_B_econ {f(qT['Y_B_econ']['auc'])}이다. "
              'p_pre만 쓰는 기준선은 Y_B_econ에서 오히려 높아지는 경우가 있다(N/E). B_econ 라벨이 사전 우세·시간과 더 강하게 묶이는 성질로 읽을 수 있지만 원인은 검증하지 않았다. '
              'q 결론이 V 선택에 민감하다는 뜻이다.', '']
    # C packet
    R += ['## 9. C. 블라인드 사람 검토 패킷 (자동 정답 아님)', '',
          f"- 출처: MAIN_TRAIN h90 유효 행(OOF 라벨)만. 사례 {pk['cases']}개, 경기 중복 없음: {pk['unique_matches']}. 우선순위 S1>S2>S3>S4, 해시 시드 {K.PACKET_SEED}.",
          '- 포함 확률은 해시를 균등 무작위 순열로 볼 때 앞선 층 선택에 대한 조건부 값이다: min(1, k/M_남은경기) × 1/r_경기. 실제 선택은 결정적이다. 대표 표본이 아니라 층화 사례 발굴이다.', '']
    rows = [[r['stratum'], r['k'], r['eligible_rows'], r['eligible_matches'], r['eligible_matches_remaining'], r['selected'], r['shortage'], f(r['match_inclusion_probability'], 6)]
            for r in pk['strata_report']]
    R += table(['층', 'k', '적격 행', '적격 경기', '남은 적격 경기', '선택', '부족', '경기 포함확률'], rows)
    R += ['', '- 검토자용 파일: `review_packet/REVIEWER_PACKET_KO.md`, `review_packet/review_form_blank.csv`. A/B 확률, Δ, 생성 Y, 선택 층, q, W, 경기 ID는 없다. '
              f"블라인드 검사 통과: {pk['blind_check']['pass']} (경기 ID 0, 금지 토큰 0, 0.xx 확률 형식 0).",
          '- 비공개 파일(검토자에게 주지 않음): `PRIVATE_review_case_key.json`(사례→경기), `PRIVATE_review_analysis_key.csv`(A/B 확률·Δ·Y·층·q·W).',
          '- 검토자는 탐지 규모를 볼 수 있다. 그래서 N 기준 사례(S4)는 T 사례와 구별될 수 있다. 사양이 규모 표시를 요구하므로 공개 사항으로 남긴다.',
          '- 원시 사건 목록만으로는 위치 선정·스킬 적중·시야 등을 확인할 수 없다. 팀의 전략적 판단은 관측된 경기 승리의 인과적 정답이 아니다.',
          '- **사람 검토: UNPERFORMED.** 검토자 이름과 평점은 비어 있으며 AI 판단으로 채우지 않는다.', '']
    # D validation
    R += ['## 10. D. 계약·사후 검증', '']
    R += table(['검사', '결과'], [[k, '통과' if v['pass_'] else '실패'] for k, v in val['checks'].items()])
    R += ['', f"사전 계약: 합성 pytest 17개 통과, TRAIN 원시 고정 사례 40경기·{fx['engagement_rows']}행에서 카운터·킬 크레딧·프레임·원소 팀·정규화 12개 검사 통과, "
              '스모크와 전체 V 모델 독립 점검 통과(적합 경기 집합 해시 재구성, 전처리 행 수, 보정 파티션, 재적재 예측 일치). '
              '**수치 검사가 통과했다고 구성 타당도가 입증되지는 않는다.**', '']
    # limits and refs
    R += ['## 11. 해석 한계', '',
          '- 불일치율은 라벨이 V 정의에 얼마나 민감한지를 보여준다. 어느 V가 실제 한타 결과를 더 잘 측정하는지는 정하지 않는다. B_econ은 W 예측도 더 나쁘다.',
          '- 관측 킬·골드·오브젝트 방향은 V_A의 입력과 겹친다(킬·오브젝트는 A의 상태 채널). 따라서 연관은 순환적이며 독립 타당성 근거가 아니다.',
          '- 전체 경기 상태 Δ에는 교전 밖 사건과 시간 경과도 들어간다. 1분 프레임에서는 L 이후 자원이 관측되지 않는 행이 많다.',
          '- 모든 bootstrap은 모델을 고정한 경기 재표집이다. 적합·보정·선택 변동은 포함하지 않았고 다중 비교 보정도 하지 않았다.',
          '- E와 T는 사후 사건으로 선택한 모집단이다. 정의 산출(G/D)에는 15.16도 쓰였다(부모 감사 §5).', '',
          '## 12. 근거 문헌의 사용 범위', '']
    R += table(['근거', '지지하는 것', '지지하지 않는 것'], [
        ['Maymin 2021 (doi:10.1515/jqas-2019-0096)', '상태 승률과 승률 변화 기반 가치평가의 근거', '우리 종료 규칙·90초·Δ 부호가 실제 한타 승리라는 검증'],
        ['Kim, Lee & Chung, CoG 2020', 'LoL 승률 확률 품질·보정의 동기', '논문 손실 함수 재현'],
        ['Jacobs & Wallach (arXiv:1912.05511)', '구성 개념과 측정의 구분, 구성 타당도 논의 틀', '수치 일치로 구성 타당도 자동 확보'],
        ['Austin, Lee & Fine 2016 + 사건 경계 CIF 결과', '경쟁 사건 종료점의 방법론적 처리', '90초 최적성·실제 전투 귀속'],
        ['우리 설계 선택', 'B_reg/B_econ 설정, 층·구간, 120사례 패킷', '(선행 논문이 정해 준 값이 아님)']])
    R += ['', '## 13. 다음 단계와 준비 상태', '',
          '1. **사람 검토 실행:** 두 명 이상의 독립 검토자가 `review_form_blank.csv`를 채운다. 검토 전 분석 키를 열지 않는다. 판정 불일치와 확신도를 보고한다. TRAIN 사례이므로 개발 정보로만 쓴다.',
          '2. **V 의존성 표적 후속:** B_econ 불일치 가운데 프레임 부재로 설명되지 않는 부분(다른 프레임 행의 약 20%)을 사례로 검토한다. 이벤트 채널을 가진 다른 계열 V와 비교할지는 새 사전 명세로 정한다.',
          f'3. **작은 Δ 규약:** A≠B_reg의 약 {pct(small_share["E"], 0)}가 |Δ_A|≤.01 행에서 생긴다(주 TEST E). 작은 Δ를 불확실 범주로 둘지는 TEST를 보지 않은 새 명세로 결정한다. 이번 결과로 임계값을 고르지 않는다.',
          '4. 원고에서 라벨은 "V_A가 정의한 추정 승률 개선 방향"으로만 서술하고, 대안 V에 따른 불일치율과 PH1 메커니즘을 제한 사항으로 공개한다.', '']
    R += ['## 14. 파일', '']
    files = ['protocol.json', 'protocol_amendments.json', 'feature_lists.json', 'source_hashes.json', 'frozen_manifest.json', 'validation.json', 'status.json',
             'results/v_eval.json', 'results/q_label_dependence.json', 'results/posthoc_mechanism_B_econ_stale_frames.json', 'results/review_packet_selection.json',
             'alt_labels/alt_labels_manifest.json', 'v_models_manifest_B_reg.json', 'v_models_manifest_B_econ.json', 'contracts/train_fixture_checks.json',
             'contracts/v_model_checks_full.json', 'tables/agreement.csv', 'tables/strata_h90.csv', 'tables/observed_directions.csv',
             'tables/objectives_h90_overlapping_subgroups.csv', 'tables/q_label_dependence_h90_main_test.csv', 'tables/horizon_sign_disagreement.csv',
             'review_packet/REVIEWER_PACKET_KO.md', 'review_packet/review_form_blank.csv', 'PRIVATE_review_case_key.json', 'PRIVATE_review_analysis_key.csv']
    R += table(['파일', 'sha256(앞 16자)'], [[fn, C.sha256_file(O / fn)[:16]] for fn in files if (O / fn).exists()])
    R += ['', '세트별 전체 결과는 `results/per_set/*.json`, 예측·라벨 배열은 `eval/v_predictions/*.npz`, `alt_labels/*.npz`, `observed/*.npz`에 있다. '
          'status.json과 이 표의 해시는 보고서 작성 시점 값이다.', '']
    (O / 'REPORT.md').write_bytes(('\n'.join(R) + '\n').encode('utf-8'))

    # ------------------------------------------------ DEFINITION_AND_EVIDENCE
    D = ['# 정의와 근거 (label_validity_full_20260915)', '',
         '근거 종류: L=문헌 방법, D=게임 규칙/구현, E=우리 데이터 추정·검증, A=연구 설계 선택.', '']
    D += table(['기호/용어', '정의', '근거 종류', '비고'], [
        ['W', '최종 Blue 승리(GAME_END winningTeam)', 'D', 'V의 학습 목표. 특징 아님'],
        ['V_A', 'expanded StateV2 logistic C=.01, raw; TRAIN은 자기 경기를 뺀 fold 어댑터', 'A (부모)', '변경 없음'],
        ['V_B_reg', 'V_A와 같은 특징·전처리, C=.1, 선택 보정 sigmoid_pos', 'A', '정규화 민감도'],
        ['V_B_econ', 'time, time², 슬롯 자원 정규화값 60개, 챔피언 one-hot; C=.01; 선택 보정 raw', 'A', '좁은 스냅샷 비교 모델, 인과 절제 아님'],
        ['q_pre', 's − 1 ms', 'A (부모)', ''],
        ['e_h', 'min(L+h, 다음 킬−1, 다음 교전 시작−1, 경기 종료−1)', 'A (부모)', '기존 종료점 재사용'],
        ['Δ_M,h', 'V_M(e_h 상태) − V_M(q_pre 상태), 같은 어댑터', 'A', ''],
        ['Y_M,h', '1[Δ_M,h > 0], 정확히 0이면 0', 'A', f'이번 실행에서 정확한 0: {exact_zero_any}건'],
        ['불일치', 'Y_A ≠ Y_B, 행가중·경기가중', 'A', '경기 bootstrap 1,000회'],
        ['during / after / full', '(q_pre, L] / (L, e] / (q_pre, e]; 하한 제외, 상한 포함', 'D/A', 'TRAIN 원시 사건으로 검증(E)'],
        ['팀 킬 차이', 'StateV2 팀 킬 크레딧 차이 Blue−Red; 동률은 별도 범주', 'D', 'during=full을 행별로 확인'],
        ['자원 변화', '팀 합 차이 변화, 정규화 단위(골드/25000, xp/20000)', 'D/E', 'gold_team_minute와 외부 원시 타임라인으로 환산 검증'],
        ['stale_same_frame', '종료 프레임 = 사전 프레임', 'A', '변화 미관측. 동률이 아님'],
        ['오브젝트 부분집합', 'blue_only/red_only/both/unknown; 서로 겹칠 수 있음', 'D/A', '합산 금지'],
        ['리뷰 층 S1–S4', 'T 작은 |Δ_A|; T A≠B; T 관측 방향 충돌+오브젝트; N 기준', 'A', '대표 표본 아님'],
        ['PH1', '같은 프레임 행에서 B_econ 부호 = 시간항 부호', 'E (사후)', '사전 명시 아님']])
    D += ['', '## 문헌', '',
          '- Maymin (2021), Smart kills and worthless deaths, Journal of Quantitative Analysis in Sports, doi:10.1515/jqas-2019-0096 — 상태 승률·승률 변화 가치평가(L).',
          '- Kim, Lee & Chung, IEEE CoG 2020 (https://ieee-cog.org/2020/papers/paper_221.pdf) — LoL 승률 확률 품질·보정 동기(L). 방법 재현 아님.',
          '- Jacobs & Wallach, Measurement and Fairness, arXiv:1912.05511 — 구성 개념과 측정의 구분, 구성 타당도(L).',
          '- Austin, Lee & Fine (2016), doi:10.1161/CIRCULATIONAHA.115.017719 — 경쟁 사건 누적발생확률(L). 90초 근거 아님.',
          '- 프로젝트 내부: docs/EVENT_BOUNDARY_CIF_RESULTS_20260914.md, docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md(감사 출처 목록 재사용).', '',
          '## 주장하지 않는 것', ''] + [f'- {c}' for c in proto['claims_not_made']] + ['']
    (O / 'DEFINITION_AND_EVIDENCE.md').write_bytes(('\n'.join(D) + '\n').encode('utf-8'))

    # ------------------------------------------------ publication tables
    Pt = ['# Publication tables — label_validity_full_20260915', '',
          'Exploratory diagnostics after prior test exposure. Labels are model-defined (V_A); agreement is not validity. Values from results/*.json.', '',
          '## Table 1. W prediction on bucket query keys (match-weighted)', '']
    Pt += table(['Set', 'V', 'Rows', 'AUC', 'Brier', 'Log loss', 'Cal. slope'],
                [[n, m, s['rows'], f(s['by_model'][m]['overall']['auc']), f(s['by_model'][m]['overall']['brier']), f(s['by_model'][m]['overall']['logloss']),
                  f(s['by_model'][m]['overall'].get('slope'), 3)] for n, s in ve['sets'].items() for m in K.ALL_V])
    Pt += ['', '## Table 2. h90 label disagreement with A (match-weighted, 1000 match-bootstrap 95% percentile interval, fixed models)', '']
    Pt += table(['Set', 'Cohort', 'Rows', 'Matches', 'A vs B_reg', 'A vs B_econ'],
                [[n, c, per[n]['agreement']['h90'][c]['A_vs_B_reg']['rows'], per[n]['agreement']['h90'][c]['A_vs_B_reg']['matches'],
                  ci(dis[n][c]['A_vs_B_reg']['match_weighted']), ci(dis[n][c]['A_vs_B_econ']['match_weighted'])] for n in SETS for c in ('E', 'T', 'N')])
    Pt += ['', '## Table 3. B_econ stale-frame mechanism (post hoc, not prespecified)', '']
    Pt += table(['Set', 'Cohort', 'Stale share', 'Time-only sign identity', 'A vs B_econ (stale)', 'A vs B_econ (non-stale)'],
                [[n, c, pct(ph['sets'][n][c]['stale_share'], 1), ph['sets'][n][c]['sign_identity_all'], pct(ph['sets'][n][c]['disagree_A_B_econ_stale'], 1),
                  pct(ph['sets'][n][c]['disagree_A_B_econ_nonstale'], 1)] for n in SETS for c in ('E', 'T', 'N')])
    Pt += ['', '## Table 4. MAIN_TEST h90 T: P(Y=1) by observed direction (not ground truth)', '']
    o = per['MAIN_TEST']['observed']['h90']['T']
    Pt += table(['Variable', 'Direction', 'Rows', 'P(Y_A=1)', 'P(Y_Breg=1)', 'P(Y_Becon=1)'],
                [[var, cat, cell['rows'], pct(cell['P_Y1_A']['match_weighted'], 1), pct(cell['P_Y1_B_reg']['match_weighted'], 1), pct(cell['P_Y1_B_econ']['match_weighted'], 1)]
                 for var in ('kills', 'gold') for cat, cell in o[var].items() if not cell.get('empty')])
    Pt += ['', '## Table 5. Saved q (trained on A) scored against alternative labels, MAIN_TEST h90 (diagnostic)', '']
    Pt += table(['Cohort', 'q', 'AUC Y_A', 'AUC Y_B_reg', 'AUC Y_B_econ', 'Brier Y_A', 'Brier Y_B_reg', 'Brier Y_B_econ'],
                [[c, col, f(cc['Y_A']['auc']), f(cc['Y_B_reg']['auc']), f(cc['Y_B_econ']['auc']), f(cc['Y_A']['brier']), f(cc['Y_B_reg']['brier']), f(cc['Y_B_econ']['brier'])]
                 for c, cell in qd['sets'].items() for col, cc in cell['columns'].items() if col in ('pooled', 'spec_' + cell['chosen'], cell['chosen'], 'spec_p_pre_spline', 'p_pre_spline')])
    Pt += ['', '## Table 6. Review packet denominators (TRAIN only; human review UNPERFORMED)', '']
    Pt += table(['Stratum', 'k', 'Eligible rows', 'Eligible matches', 'Remaining', 'Selected', 'Shortage'],
                [[r['stratum'], r['k'], r['eligible_rows'], r['eligible_matches'], r['eligible_matches_remaining'], r['selected'], r['shortage']] for r in pk['strata_report']])
    (O / 'tables' / 'PUBLICATION_TABLES.md').write_bytes(('\n'.join(Pt) + '\n').encode('utf-8'))
    st.update('complete', 'report', report_sha256=C.sha256_file(O / 'REPORT.md'), next_step='done: human review UNPERFORMED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
