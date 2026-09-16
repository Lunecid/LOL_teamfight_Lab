"""Stage R: Korean REPORT.md and DEFINITION_AND_EVIDENCE.md from saved artifacts only (guarded; no new computation on data)."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from pathlib import Path  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import ta20260916_common as T  # noqa: E402

OUT = T.OUT
LABEL = {'mlp_winner': 'plain MLP(신규)', 'resmlp_winner': 'residual MLP(신규)', 'overall_winner_mlp_families': 'MLP 두 family 1위(신규)',
         'pt_winner': 'iq PT 기준선', 'logit_winner': 'iq 전체 logistic', 'lgbm_winner': 'iq 전체 LightGBM',
         'old_A_specialist': '기존 A specialist', 'old_pooled': '기존 pooled q', 'old_p_pre_spline': '기존 p_pre spline',
         'old_p_pre_logistic': '기존 p_pre logistic', 'old_constant': '기존 상수'}
MODELS = ('mlp_winner', 'resmlp_winner', 'lgbm_winner', 'logit_winner', 'pt_winner', 'old_A_specialist', 'old_pooled', 'old_p_pre_spline', 'old_constant')
CELL_KO = {'all': '전체', 'B40': 'B40', 'B45': 'B45', 'time_0_10': '[0,10)분', 'time_10_20': '[10,20)분', 'time_20_30': '[20,30)분', 'time_30_inf': '[30,∞)분'}
FAM_KO = {'mlp': 'plain MLP', 'resmlp': 'residual MLP'}


def f(x, d=5):
    return '—' if x is None else f'{x:.{d}f}'


def fs(x, d=5):
    return '—' if x is None else f'{x:+.{d}f}'


def ci(c, d=5):
    return '—' if not c else f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'


def verdict(delta):
    """Brier / log loss contrast a - b: negative = a better."""
    c = delta.get('ci95')
    if not c or delta.get('estimate') is None:
        return '미계산'
    if c[1] < 0:
        return 'a 우세(구간 0 미포함)'
    if c[0] > 0:
        return 'b 우세(구간 0 미포함)'
    return '구간이 0 포함'


def auc_verdict(delta):
    c = delta.get('ci95')
    if not c or delta.get('estimate') is None:
        return '미계산'
    if c[0] > 0:
        return 'a 우세(구간 0 미포함)'
    if c[1] < 0:
        return 'b 우세(구간 0 미포함)'
    return '구간이 0 포함'


def main():
    T.log_command()
    proto = C.read_json(OUT / 'protocol.json')
    fz = C.read_json(T.frozen_path(OUT))
    res = C.read_json(OUT / 'eval' / 'results.json')
    val = C.read_json(OUT / 'validation.json') if (OUT / 'validation.json').exists() else None
    ct = C.read_json(OUT / 'contract_tests' / 'result.json')
    sels = {f'{fm}_{c}': C.read_json(OUT / 'selection' / f'{fm}_{c}.json') for fm in T.FAMILIES for c in T.COHORTS}
    stops = {f'{fm}_{c}': C.read_json(OUT / 'internal_stop' / f'{fm}_{c}.json')['configs'] for fm in T.FAMILIES for c in T.COHORTS}
    fails = [json.loads(l) for l in (OUT / 'failures.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (OUT / 'failures.jsonl').exists() else []
    R = res['results']
    lines = []
    w = lines.append
    w('# 동일 352입력 plain MLP / residual MLP 대 iq 학습기 (h90, T/N) — Track A 완결 실행 보고서')
    w('')
    w(f'실행 {T.VERSION}. 설계·구현·실행: Claude (Codex 라인 계승, 2026-09-16). 부모 실행 `incremental_q_training_20260915`(iq)의 동결 승자와 동일 행에서 비교한다. '
      '기존 TEST/외부 결과 노출 후의 **탐색적 후속 실험**이며 미접촉 확증이 아니다. 라벨 Y=1[ΔV_h90>0]는 모델 정의 결과이지 실제 한타 승리 정답이 아니다. '
      f'GPU(cuda) 학습은 사용자 명시 승인(2026-09-16)에 따른다. 예측값 정의는 저장된 float32 가중치의 CPU float64 추론(스레드 {T.PRED_THREADS}, 청크 {T.PRED_CHUNK}).')
    w('')
    # ---------------------------------------------------------------- 1 summary
    w('## 1. 핵심 요약')
    w('')
    for coh, title in (('T', 'T(한타, 주 분석) — MAIN TEST 15.16'), ('N', 'N(비한타, 보조) — MAIN TEST 15.16')):
        r = R['MAIN_TEST'][coh]
        w(f'### {title}')
        w('')
        for cell in ('all', 'B40', 'B45'):
            mn = r['metrics_named'][cell]
            best = min((mn[m]['brier'], m) for m in MODELS if mn[m]['brier'] is not None)
            w(f"- {CELL_KO[cell]} ({r['cells'][cell]['rows']:,}행/{r['cells'][cell]['matches']:,}경기) Brier 점추정 최저(서술): {LABEL[best[1]]} {f(best[0])}; "
              + '; '.join(f'{LABEL[m]} {f(mn[m]["brier"])}' for m in MODELS[:5]) + '.')
        for cell in ('all', 'B40'):
            bt = r['bootstrap'][cell]
            if bt.get('computed', True) is False:
                w(f'- {CELL_KO[cell]}: 부트스트랩 미계산({bt["reason"]}).')
                continue
            for p in bt['pairs']:
                d = p['a_minus_b']
                w(f"- {CELL_KO[cell]} · {p['label']}: ΔBrier {fs(d['brier']['estimate'])} {ci(d['brier']['ci95'])} ({verdict(d['brier'])}), "
                  f"Δlog loss {fs(d['logloss']['estimate'])} {ci(d['logloss']['ci95'])}, ΔAUC {fs(d['auc']['estimate'], 4)} {ci(d['auc']['ci95'], 4)} ({auc_verdict(d['auc'])}).")
        w('')
    five = fz['five_family_q_select_ranking']
    w('Q_SELECT 5-family 순위(정보용; 동일 cohort Q_SELECT 행, iq 승자는 재적합 안 함): '
      + '; '.join(f"{c}: {' > '.join(v['ranking'])}" for c, v in five.items()) + '.')
    w('')
    # ---------------------------------------------------------------- 2 before/after
    w('## 2. 진행 전/후')
    w('')
    w('| 항목 | 이전(iq까지) | 이번 완료 |')
    w('|---|---|---|')
    w('| 동일 352입력 신경망 | 없음(MLP/residual MLP는 iq 명세에서 유보) | plain MLP·residual MLP 각 6설정×3시드, 내부 stop10 경기가중 Brier 조기종료, 전체 TRAIN 재적합, 보정 3개, Q_SELECT 선정 |')
    w('| 비교 대상 | PT·logistic·LightGBM(iq) 대 기존 참조 | 두 MLP 승자 대 iq 세 승자·기존 참조를 같은 행에서 짝지은 부트스트랩 |')
    w('| 균형 상태 | B40/B45 셀 보고 | 동일 셀에서 MLP 결과 추가 |')
    w('| h60/h120, SHAP, Track B | 미실행 | 이번에도 미실행(명시) |')
    w('')
    # ---------------------------------------------------------------- 3 data contract
    w('## 3. 데이터 계약과 사전 검사')
    w('')
    dc = proto['data_contract']
    w(f"- 코퍼스 {dc['corpus']}. 라벨: {dc['label']}. 코호트: {dc['cohorts']}.")
    w(f"- 입력 {proto['inputs']['learners']} (sha256 {proto['inputs']['ridge_names_sha256'][:12]}…). 전처리: {proto['inputs']['preprocessing']}.")
    w(f"- 계약 테스트 run{ct['run']}: {ct['passed_count']} passed / {ct['failed_count']} failed / {ct['skipped_count']} skipped ({ct['at']}). "
      f"프로토콜 동결 {proto['written_at']}, smoke 동결 {C.read_json(T.frozen_path(T.SMOKE))['frozen_at']}, 전체 동결 {fz['frozen_at']}.")
    w(f"- 봉인 세트 접근은 이 실행의 frozen_manifest 이후에만 허용(iq 동결과 무관). 동결 전 봉인 접근 {fz['sealed_accesses_before_freeze']}건.")
    w('')
    # ---------------------------------------------------------------- 4 fit records
    w('## 4. 적합 기록')
    w('')
    w(f"- 최적화: {T.OPT['optimizer']} lr {T.OPT['lr']}, weight_decay {T.OPT['weight_decay']}({T.OPT['weight_decay_scope']}), batch {T.OPT['batch_size']}, "
      f"최대 {T.OPT['max_epochs']} epoch, patience {T.OPT['patience']}, float32, AMP 없음. 손실 {proto['families']['loss']}.")
    w(f"- 정지 집단: {proto['families']['stopping']['allocation']}. 재적합: {proto['families']['stopping']['refit']}.")
    w('')
    w('| family/cohort | 설정 | 파라미터 수 | 시드별 최적 epoch (7/42/123) | cap 도달 | stop Brier 최적(시드 평균) | 적합 초 | CPU 예측 초 |')
    w('|---|---|---:|---|---|---:|---:|---:|')
    for key, sel in sels.items():
        for cfg in T.config_names(sel['family']):
            fr = sel['fit_records'][cfg]
            sr = stops[key][cfg]['seeds']
            be = '/'.join(str(sr[str(s)]['best_epoch']) for s in T.SEEDS)
            cap = sum(int(sr[str(s)]['cap_reached']) for s in T.SEEDS)
            sb = sum(sr[str(s)]['best_stop_weighted_brier'] for s in T.SEEDS) / len(T.SEEDS)
            w(f"| {key} | {cfg} | {fr['arch']['n_params']:,} | {be} | {cap}/3 | {f(sb, 6)} | {fr['seconds_fit']} | {fr['seconds_predict_cpu64']} |")
    w('')
    # ---------------------------------------------------------------- 5 selection
    w('## 5. Q_SELECT 선정 (Brier → log loss → 이름)')
    w('')
    for coh in T.COHORTS:
        w(f'### {coh}')
        w('')
        w('| family | 선정 후보 | Q_SELECT Brier | log loss | AUC | 적격 후보 | 상위 3 |')
        w('|---|---|---:|---:|---:|---:|---|')
        for fm in T.FAMILIES:
            sel = sels[f'{fm}_{coh}']
            m = sel['select_metrics'][sel['chosen']]
            w(f"| {FAM_KO[fm]} | {sel['chosen']} | {f(m['brier'], 6)} | {f(m['logloss'], 6)} | {f(m['auc'], 4)} | {sum(sel['eligible'].values())}/18 | {', '.join(sel['ranking'][:3])} |")
        w('')
        v = five[coh]
        w('5-family 순위(정보용): ' + ' > '.join(f"{k} (Brier {f(v['metrics'][k]['brier'], 6)})" for k in v['ranking']) + '.')
        w('')
    # ---------------------------------------------------------------- 6 sealed evaluation
    w('## 6. 봉인 평가: MAIN TEST 15.16')
    w('')
    for coh, title in (('T', 'T (주)'), ('N', 'N (보조)')):
        r = R['MAIN_TEST'][coh]
        w(f'### 6.{1 if coh == "T" else 2} {title}')
        w('')
        w(f"행 {r['rows']:,} / 경기 {r['matches']:,}; 승자 {r['winners']}; MLP 두 family 1위 {r['overall']}.")
        w('')
        w('| 모델 | 전체 Brier | 전체 log loss | 전체 AUC | B40 Brier | B40 log loss | B40 AUC | B45 Brier | B45 AUC | 전체 ECE | 전체 보정 기울기 | 정확히 0/1(반대 라벨) |')
        w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
        for m in MODELS:
            a, b40, b45 = r['metrics_named']['all'][m], r['metrics_named']['B40'][m], r['metrics_named']['B45'][m]
            w(f"| {LABEL[m]} | {f(a['brier'])} | {f(a['logloss'])} | {f(a['auc'], 4)} | {f(b40['brier'])} | {f(b40['logloss'])} | {f(b40['auc'], 4)} | "
              f"{f(b45['brier'])} | {f(b45['auc'], 4)} | {f(a['ece_10bin'], 4)} | {f(a['slope'], 3)} | {a['exact_prob_0'] + a['exact_prob_1']} ({a['exact_0_or_1_opposite_label']}) |")
        w('')
        w('**계획된 paired 대비 (a − b; Brier/log loss 음수 = a 우세; 1000 경기 부트스트랩, seed 20260915)**')
        w('')
        w('| 셀 | 대비 | ΔBrier [95%] | 판정 | Δlog loss [95%] | ΔAUC [95%] | 퇴화 복제 |')
        w('|---|---|---|---|---|---|---:|')
        for cell in ('all', 'B40', 'B45'):
            bt = r['bootstrap'][cell]
            if bt.get('computed', True) is False:
                w(f"| {CELL_KO[cell]} | — | 미계산 ({bt['reason']}) | — | — | — | — |")
                continue
            for p in bt['pairs']:
                d = p['a_minus_b']
                w(f"| {CELL_KO[cell]} | {p['label']} | {fs(d['brier']['estimate'])} {ci(d['brier']['ci95'])} | {verdict(d['brier'])} | "
                  f"{fs(d['logloss']['estimate'])} {ci(d['logloss']['ci95'])} | {fs(d['auc']['estimate'], 4)} {ci(d['auc']['ci95'], 4)} | {bt['degenerate_single_class_replicates']} |")
        w('')
        w('**시간 구간별 Brier (점추정만; 구간 CI는 명세대로 미계산)**')
        w('')
        tcells = [k for k in r['metrics_named'] if k.startswith('time_')]
        w('| 모델 | ' + ' | '.join(CELL_KO[k] for k in tcells) + ' | ' + ' | '.join(f'B40×{CELL_KO[k]}' for k in tcells) + ' |')
        w('|---|' + '---:|' * (2 * len(tcells)))
        for m in MODELS:
            w(f'| {LABEL[m]} | ' + ' | '.join(f(r['metrics_named'][k][m]['brier']) for k in tcells) + ' | '
              + ' | '.join(f(r['metrics_named'][f"B40_x_{k}"][m]['brier']) for k in tcells) + ' |')
        w('')
        w('시간 셀 행 수: ' + ', '.join(f"{CELL_KO[k]} {r['cells'][k]['rows']:,}" for k in tcells) + '.')
        w('')
        for fm in T.FAMILIES:
            sm = r['winner_config_seed_metrics_DESCRIPTIVE'][fm]['all']
            w(f"{FAM_KO[fm]} 선정 설정의 seed별 전체 Brier(서술용, raw): " + ', '.join(f"seed {s} {f(sm[str(s)]['brier'])}" for s in T.SEEDS)
              + f"; 3-seed 평균 후 보정한 선정 후보 {f(r['metrics_named']['all'][f'{fm}_winner']['brier'])}.")
        w('')
    # ---------------------------------------------------------------- 7 external
    w('## 7. 외부 세트(각각 따로; 이전 노출 있음, 새 수집·EUW 주장 없음)')
    w('')
    w('| 세트 | cohort | 셀 | 행/경기 | plain MLP Brier | residual MLP Brier | iq LightGBM Brier | iq logistic Brier | iq PT Brier | resMLP−LGBM ΔBrier [95%] | 판정 |')
    w('|---|---|---|---|---:|---:|---:|---:|---:|---|---|')
    for set_name in T.EVAL_SETS[1:]:
        for coh in T.COHORTS:
            r = R[set_name][coh]
            for cell in ('all', 'B40', 'B45'):
                mn = r['metrics_named'][cell]
                bt = r['bootstrap'][cell]
                if bt.get('computed', True) is False:
                    dv, vd = f"미계산({bt['reason']})", '—'
                else:
                    d = bt['pairs'][0]['a_minus_b']['brier']
                    dv, vd = f"{fs(d['estimate'])} {ci(d['ci95'])}", verdict(d)
                sparse = '*' if mn['mlp_winner']['sparse_lt30_matches'] else ''
                w(f"| {set_name} | {coh} | {CELL_KO[cell]}{sparse} | {r['cells'][cell]['rows']:,}/{r['cells'][cell]['matches']:,} | {f(mn['mlp_winner']['brier'])} | "
                  f"{f(mn['resmlp_winner']['brier'])} | {f(mn['lgbm_winner']['brier'])} | {f(mn['logit_winner']['brier'])} | {f(mn['pt_winner']['brier'])} | {dv} | {vd} |")
    w('')
    w('\\* 30경기 미만 희소 셀: 구간·일반화 주장 없음. 나머지 계획 대비와 모든 모델·셀 지표는 `eval/results.json`에 있다.')
    w('')
    # ---------------------------------------------------------------- 8 interpretation
    w('## 8. 해석(음성·혼합 결과 포함)')
    w('')
    for coh in T.COHORTS:
        r = R['MAIN_TEST'][coh]
        for cell in ('all', 'B40'):
            bt = r['bootstrap'][cell]
            if bt.get('computed', True) is False:
                continue
            for p in bt['pairs']:
                d = p['a_minus_b']['brier']
                w(f"- {coh} {CELL_KO[cell]}, {p['label']}: ΔBrier {fs(d['estimate'])} {ci(d['ci95'])} → {verdict(d)}.")
    for coh in T.COHORTS:
        r = R['MAIN_TEST'][coh]
        const = r['metrics_named']['all']['old_constant']['brier']
        worse = []
        for cell, mn in r['metrics_named'].items():
            cb = mn['old_constant']['brier']
            for m in MODELS:
                if m != 'old_constant' and mn[m]['brier'] is not None and cb is not None and mn[m]['brier'] >= cb:
                    worse.append(f"{CELL_KO.get(cell, cell)} {LABEL[m]} {f(mn[m]['brier'])} ≥ 상수 {f(cb)}")
        w(f"- **음성 결과({coh}, MAIN TEST, 점추정·CI 없음):** Brier가 TRAIN 상수 이상인 셀: " + ('; '.join(worse) if worse else '없음') + '.')
    ext_v = []
    for set_name in T.EVAL_SETS[1:]:
        for coh in T.COHORTS:
            bt = R[set_name][coh]['bootstrap']['all']
            ext_v.append(f"{set_name} {coh}: " + ('미계산' if bt.get('computed', True) is False else verdict(bt['pairs'][0]['a_minus_b']['brier'])))
    w('- 외부 세트 전체 행의 주 대비(residual MLP − LightGBM) 판정: ' + '; '.join(ext_v) + '.')
    w('- 두 MLP는 iq LightGBM·logistic과 **같은 352열**을 읽는다(PT 기준선만 p_pre·시간). 6설정×3시드는 제한된 탐색 관례이며 계산량 동등이나 전역 최적을 뜻하지 않는다. '
      'iq 승자는 이 실행에서 재적합·재선정하지 않았고 같은 프로토콜의 이전 단계 결과다.')
    w('- 부트스트랩 구간은 고정 모델의 평가 표본 불확실성만 반영한다(학습·보정·선정 불확실성 제외, 다중 비교 보정 없음). 여러 셀·세트의 결과를 함께 읽고 한 셀의 구간으로 결론을 고르지 않는다.')
    w('- p_pre와 q는 서로 다른 확률이다. B40/B45는 V가 경기를 균형으로 본 셀일 뿐이며 Y의 균형이나 q의 기준 Brier를 함의하지 않는다.')
    w('')
    # ---------------------------------------------------------------- 9 verification
    w('## 9. 검증과 실패 기록')
    w('')
    if val:
        w(f"- validation.json: {val['n_checks']}개 검사, 실패 {len(val['failed'])}개 ({val['checked_at']}). "
          + (f"실패 항목: {val['failed']}." if val['failed'] else '부모 무결성 diff all_equal, 동결 hash 불변, 선정·보정기·전처리·stop 기록·seed 평균·아키텍처 파라미터 수 독립 재현, iq 비교열이 iq 보고 지표를 <1e-12로 재현, 저장 예측으로부터 지표 재계산 <1e-10, 부트스트랩 수용 게이트, 새 프로세스 재로딩 동일 예측(CPU float64).'))
    else:
        w('- validation.json 미생성(사후 검사 미실행).')
    w(f"- 계약 테스트: run{ct['run']} {ct['passed_count']} passed ({ct['at']}); 스냅샷 → 테스트 → 프로토콜 → smoke → 테스트 → 전체 적합 → 동결 → 봉인 평가 순서.")
    if fails:
        w('- 보존된 실패/재시도:')
        for x in fails:
            w(f"  - {x['time']} `{x['stage']}`: {x['error'][:200]}")
    else:
        w('- 보존된 실패 기록 없음.')
    w('')
    # ---------------------------------------------------------------- 10 limits
    w('## 10. 한계와 비주장')
    w('')
    w('- 기존 TEST/외부 결과 노출 후 수행한 탐색적 비교다. 새 미래 패치 확인이 필요하다.')
    w('- h90만 적합했다. h60/h120, SHAP, revised V, Track B(CoG 표현 어댑터), 개발 전용 탐지기 재추정은 **실행하지 않았다**.')
    w('- T/N은 사후 참가 규모로 나눈 retrospective 조건부 평가이며 실시간 라우팅 검증이 아니다.')
    w('- GPU float32 학습의 stop 곡선은 진단이며, 예측값은 CPU float64 정의로 동결·재현했다. 학습 자체의 재실행 비트 동일성은 게이트가 아니다(재로딩 동일성만 게이트).')
    w('- 최종 W에 대한 인과효과나 Y의 의미 타당성은 검증하지 않았다.')
    w('')
    # ---------------------------------------------------------------- 11 artefacts
    w('## 11. 산출물')
    w('')
    for line in ('`protocol.json`: 사전 동결 계약·후보 등록부·정책', '`contract_tests/`: 계약 테스트 결과(run별)', '`smoke_train_only/`: TRAIN-only smoke 전체 경로',
                 '`models/<cohort>/<family>/<config>.joblib`: 설정별 번들(전처리+3시드 가중치+Q_CAL 보정기)', '`selection/<family>_<cohort>.json`: 18후보 지표·선정·적합 기록',
                 '`internal_stop/<family>_<cohort>.json`: stop 곡선·최적 epoch·membership hash', '`predictions/<family>_<cohort>_trainval.npz`: TRAIN/Q_CAL/Q_SELECT raw·seed·보정 예측',
                 '`frozen_manifest.json`: 동결 manifest(모든 hash, 승자, iq 참조 hash, 5-family 순위)', '`access_log.jsonl`: 부모 데이터 접근 기록(봉인 세트는 동결 후)',
                 '`eval/results.json`: 셀별 지표·부트스트랩', '`eval/predictions/<set>_h90_<cohort>.npz`: 행별 평가 예측(36후보·seed·iq 비교열·셀 표시)',
                 '`integrity/`: 부모 snapshot 전/후/diff', '`validation.json`: 사후 검증', '`status.json, logs/, commands.txt, failures.jsonl`: 진행·명령·실패 보존',
                 '`DEFINITION_AND_EVIDENCE.md`: 정의·근거·상태 표'):
        w(f'- {line}')
    w('')
    w('## 12. 참고')
    w('')
    w('- Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS 34. https://arxiv.org/abs/2106.11959 — ResNet형 tabular 기준선의 동기(우리 352입력·후보 6개·구조의 외부 정답성 아님).')
    w('- Ke, G. et al. (2017). LightGBM. NeurIPS 30. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html — 비교 대상 gradient boosting 방법 출처.')
    w('- Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR. https://arxiv.org/abs/1711.05101 — AdamW.')
    w('- PyTorch 2.10 / scikit-learn 1.6.1 공식 문서(로컬 설치) — 구현 API 출처. 폭·dropout·학습률·batch·epoch cap·patience·seed·보고 기준은 우리 운영 선택이다.')
    C.write_json(OUT / 'report_manifest.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), results_sha256=C.sha256_file(OUT / 'eval' / 'results.json'),
                                                    frozen_manifest_sha256=C.sha256_file(T.frozen_path(OUT)), validation_sha256=C.sha256_file(OUT / 'validation.json') if val else None))
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # ---------------------------------------------------------------- definitions and evidence
    d_lines = []
    d = d_lines.append
    d('# 정의·근거·상태 (Track A MLP, 2026-09-16)')
    d('')
    d('## 진행 전/후')
    d('')
    d('| 요소 | 이전 | 이번 | 근거 유형 |')
    d('|---|---|---|---|')
    d('| 교전 모집단·라벨·분할 | iq와 동일(동결) | 변경 없음 | 부모 계약(fc/cr) |')
    d('| 입력 | iq 352열 | 동일 352열 | 부모 스키마 hash |')
    d('| 학습기 | PT·logistic·LightGBM | + plain MLP, residual MLP | DELTA_Q 프로토콜 §3, COG_MODEL_COVERAGE §2 (우리 운영 선택) |')
    d('| 정지·재적합 | iq LightGBM stop10 규칙 | 같은 stop10 배정, epoch 단위 patience 10, 전체 TRAIN 재적합 | 프로토콜 §3-2·3 |')
    d('| 예측 정의 | sklearn/LightGBM CPU | 저장 float32 가중치의 CPU float64 추론 | 재현성 설계 선택 |')
    d('| 비교·부트스트랩 | 5 대비 | 6 대비(주: residual MLP − LightGBM) | 프로토콜 §5 |')
    d('')
    d('## 주요 수치 근거')
    d('')
    for coh in T.COHORTS:
        r = R['MAIN_TEST'][coh]
        for cell in ('all', 'B40'):
            bt = r['bootstrap'][cell]
            if bt.get('computed', True) is False:
                continue
            p = bt['pairs'][0]
            dd = p['a_minus_b']['brier']
            d(f"- MAIN TEST {coh} {cell} {p['label']}: ΔBrier {fs(dd['estimate'])} {ci(dd['ci95'])} — `eval/results.json` → results.MAIN_TEST.{coh}.bootstrap.{cell}.pairs[0]")
    d('')
    d('## 무결성')
    d('')
    d(f"- frozen_manifest sha256 {C.sha256_file(T.frozen_path(OUT))}; results sha256 {C.sha256_file(OUT / 'eval' / 'results.json')}; "
      + (f"validation {val['n_checks']}검사 실패 {len(val['failed'])}" if val else 'validation 미실행') + '.')
    d('- 부모 read-only, 새 root만 기록, 봉인 세트는 이 실행의 동결 이후 접근, 학습·동결·평가 소스 hash 동결 manifest에 기록.')
    (OUT / 'DEFINITION_AND_EVIDENCE.md').write_text('\n'.join(d_lines) + '\n', encoding='utf-8')
    print('wrote REPORT.md and DEFINITION_AND_EVIDENCE.md')


if __name__ == '__main__':
    main()
