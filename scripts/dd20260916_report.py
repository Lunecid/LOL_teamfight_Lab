"""Stage R: Korean REPORT.md and DEFINITION_AND_EVIDENCE.md from saved artifacts only."""
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
import dd20260916_common as DD  # noqa: E402

OUT = DD.OUT
LABEL = {'refit_lgbm': '재적합 LightGBM', 'refit_logit': '재적합 logistic', 'frozen_lgbm': '동결 부모 LightGBM', 'frozen_logit': '동결 부모 logistic', 'frozen_pt': '동결 부모 PT'}


def f(x, d=5):
    return '—' if x is None else f'{x:.{d}f}'


def fs(x, d=5):
    return '—' if x is None else f'{x:+.{d}f}'


def ci(c, d=5):
    return '—' if not c else f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'


def verdict(d):
    c = d.get('ci95')
    if not c or d.get('estimate') is None:
        return '미계산'
    return 'a 우세(구간 0 미포함)' if c[1] < 0 else ('b 우세(구간 0 미포함)' if c[0] > 0 else '구간이 0 포함')


def pair(bt, a, b):
    if bt.get('computed', True) is False:
        return None
    for p in bt['pairs']:
        if p['a'] == a and p['b'] == b:
            return p['a_minus_b']
    return None


def main():
    DD.log_command()
    proto = C.read_json(OUT / 'protocol.json')
    fz = C.read_json(DD.frozen_path(OUT))
    res = C.read_json(OUT / 'eval' / 'results.json')
    census = C.read_json(OUT / 'census.json')
    val = C.read_json(OUT / 'validation.json') if (OUT / 'validation.json').exists() else None
    ct = C.read_json(OUT / 'contract_tests' / 'result.json')
    fcheck = C.read_json(OUT / 'redetect' / 'frozen_check.json')
    fx = C.read_json(OUT / 'rebuild' / 'fixture_check.json')
    R = res['results']
    fails = [json.loads(l) for l in (OUT / 'failures.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (OUT / 'failures.jsonl').exists() else []
    L = []
    w = L.append
    d = proto['definitions']
    w('# 개발 패치 전용 정의(G 14.0 s, D 4,285 u)로 재탐지: 모집단 변화와 h90 q 결과 — 실행 보고서')
    w('')
    w(f"실행 {DD.VERSION}. 설계·구현·실행·검증: Claude(Codex 라인 계승, 2026-09-16). 동결 정의(pooled 15.14–15.16: G {d['frozen']['spec']['gap_s']:.4f} s → 13.7 s, D {d['frozen']['spec']['diameter_u']:.2f} → 4,264 u) 대신 "
      f"TRAIN 패치(15.14) 단독 추정치(G {d['dev']['spec']['gap_s']:.4f} s → 14.0 s, D {d['dev']['spec']['diameter_u']:.2f} → 4,285 u)로 부모와 같은 탐지 코드를 전 경기에 다시 돌리고, 바뀐 경기만 부모 추출 함수로 재구축·동결 V로 재라벨한 뒤 "
      f"기준 팔(352열, iq 구성 고정)을 재적합했다. 동결 상수의 재탐지는 TRAIN {fcheck['matches_compared']:,}경기에서 부모 노출 행과 정확히 일치({fcheck['mismatching_matches']} 불일치), "
      f"부모 노출로 재구축한 {fx['matches']}경기 fixture는 부모 행과 비트 단위로 같았다(all_equal={fx['checks']['all_equal']}). 기존 TEST 노출 후의 탐색적 실행이며 라벨은 모델 정의 결과다.")
    w('')
    w('## 1. 모집단 조사 (노출 행 = 교전; 8필드 동일 = 동일 행)')
    w('')
    w('| 세트 | 경기 | 변경 경기 | 비율 | 부모 행 | dev 행 | 공통 | 제거 | 추가 | dev 유효 h90 T / N (부모 T / N) |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for s in DD.ALL_SETS:
        c = census['sets'][s]
        rb = C.read_json(OUT / 'rebuild' / f'{s}.json')
        v = rb['valid_h90']
        w(f"| {s} | {c['matches']:,} | {c['matches_changed']:,} | {100 * c['matches_changed_frac']:.2f} % | {c['rows_parent']:,} | {c['rows_dev']:,} | {c['rows_common']:,} | {c['rows_removed']:,} | {c['rows_added']:,} | "
          f"{v['by_cohort_dev']['T']:,} / {v['by_cohort_dev']['N']:,} ({v['by_cohort_parent']['T']:,} / {v['by_cohort_parent']['N']:,}) |")
    w('')
    w('## 2. MAIN TEST 15.16: 옛 모집단(부모 보고) vs dev 모집단(같은 동결 모델), 그리고 재적합 (전체 셀)')
    w('')
    w('| cohort | 모델 | 옛 모집단 Brier | 옛 AUC | dev 모집단 Brier | dev AUC | dev 공통 행 Brier | dev 재구축 행 Brier (행) |')
    w('|---|---|---:|---:|---:|---:|---:|---|')
    for coh in DD.COHORTS:
        r = R['MAIN_TEST'][coh]
        mn, old, bs = r['metrics_named']['all'], r['old_metrics_named']['all'], r['metrics_by_row_source_DESCRIPTIVE']
        for fam in ('lgbm', 'logit', 'pt'):
            o = old[f'{fam}_winner']
            reb = bs.get('rebuilt', {}).get(f'frozen_{fam}')
            w(f"| {coh} | {LABEL[f'frozen_{fam}']} | {f(o['brier'])} | {f(o['auc'], 4)} | {f(mn[f'frozen_{fam}']['brier'])} | {f(mn[f'frozen_{fam}']['auc'], 4)} | {f(bs['parent'][f'frozen_{fam}']['brier'])} | "
              f"{f(reb['brier']) if reb else '—'} ({reb['rows'] if reb else 0}) |")
        for fam in DD.FAMILIES:
            reb = bs.get('rebuilt', {}).get(f'refit_{fam}')
            w(f"| {coh} | {LABEL[f'refit_{fam}']} | — | — | {f(mn[f'refit_{fam}']['brier'])} | {f(mn[f'refit_{fam}']['auc'], 4)} | {f(bs['parent'][f'refit_{fam}']['brier'])} | {f(reb['brier']) if reb else '—'} ({reb['rows'] if reb else 0}) |")
        p = r['population']
        w(f"| {coh} | 행 수 | {p['old_rows']:,} | | {p['dev_rows']:,} (공통 {p['parent_source_rows']:,}, 재구축 {p['rebuilt_rows']:,}, 옛 행 탈락 {p['old_rows_dropped']:,}) | | | |")
    w('')
    for coh in DD.COHORTS:
        bt = R['MAIN_TEST'][coh]['bootstrap']['all']
        parts = [f"{p['label'].split(': ')[1]}: ΔBrier {fs(p['a_minus_b']['brier']['estimate'])} {ci(p['a_minus_b']['brier']['ci95'])} ({verdict(p['a_minus_b']['brier'])})" for p in bt.get('pairs', [])]
        w(f"- {coh} 전체 셀 짝지은 부트스트랩(dev 모집단, 1,000회): " + '; '.join(parts) + '.')
    w('')
    w('## 3. 균형 셀 B40 / B45 (dev 모집단)')
    w('')
    w('| cohort | 셀 | 행 | 동결 PT | 동결 LightGBM | 재적합 LightGBM | 재적합−동결 LightGBM ΔBrier [95%] | 동결 LightGBM−PT ΔBrier [95%] |')
    w('|---|---|---:|---:|---:|---:|---|---|')
    for coh in DD.COHORTS:
        r = R['MAIN_TEST'][coh]
        for cell in ('B40', 'B45'):
            mn = r['metrics_named'][cell]
            d1 = pair(r['bootstrap'][cell], 'refit_lgbm', 'frozen_lgbm')
            d2 = pair(r['bootstrap'][cell], 'frozen_lgbm', 'frozen_pt')
            w(f"| {coh} | {cell} | {r['cells'][cell]['rows']:,} | {f(mn['frozen_pt']['brier'])} | {f(mn['frozen_lgbm']['brier'])} | {f(mn['refit_lgbm']['brier'])} | "
              f"{(fs(d1['brier']['estimate']) + ' ' + ci(d1['brier']['ci95'])) if d1 else '미계산'} | {(fs(d2['brier']['estimate']) + ' ' + ci(d2['brier']['ci95'])) if d2 else '미계산'} |")
    w('')
    w('## 4. 외부 세트 (dev 모집단 전체 행)')
    w('')
    w('| 세트 | cohort | dev 행 (재구축) | 옛 행 | 옛 LightGBM Brier | dev 동결 LightGBM | dev 재적합 LightGBM | 재적합−동결 ΔBrier [95%] |')
    w('|---|---|---|---:|---:|---:|---:|---|')
    for s in DD.EVAL_SETS[1:]:
        for coh in DD.COHORTS:
            r = R[s][coh]
            mn, old, p = r['metrics_named']['all'], r['old_metrics_named']['all'], r['population']
            d1 = pair(r['bootstrap']['all'], 'refit_lgbm', 'frozen_lgbm')
            w(f"| {s} | {coh} | {p['dev_rows']:,} ({p['rebuilt_rows']:,}) | {p['old_rows']:,} | {f(old['lgbm_winner']['brier'])} | {f(mn['frozen_lgbm']['brier'])} | {f(mn['refit_lgbm']['brier'])} | "
              f"{(fs(d1['brier']['estimate']) + ' ' + ci(d1['brier']['ci95'])) if d1 else '미계산'} |")
    w('')
    w('## 5. 선정과 정지 기록 (dev TRAIN 적합)')
    w('')
    w('| cohort | family | 구성 | 선정 보정 | dev Q_SELECT Brier | TRAIN 행(재구축) | 정지 반복(시드 7/42/123) | 적합 초 |')
    w('|---|---|---|---|---:|---|---|---:|')
    for coh in DD.COHORTS:
        for fam in DD.FAMILIES:
            s = C.read_json(OUT / 'selection' / f'{fam}_{coh}.json')
            stop = '—'
            if fam in DD.SEEDED:
                sr = C.read_json(OUT / 'internal_stop' / f'{fam}_{coh}.json')['stop_record']['seeds']
                stop = '/'.join(str(sr[str(sd)]['best_iteration']) for sd in DD.SEEDS[fam])
            rc = s['manifest']['row_source_counts']['TRAIN']
            w(f"| {coh} | {LABEL[f'refit_{fam}']} | {s['config']} | {s['chosen_calibration']} | {f(s['select_metrics'][s['chosen']]['brier'], 6)} | {s['split_counts']['TRAIN']['rows']:,} ({rc.get('rebuilt', 0):,}) | {stop} | {s['fit_record']['seconds_fit']} |")
    w('')
    w('## 6. 해석')
    w('')
    cT = census['sets']['MAIN_TEST']
    rT = R['MAIN_TEST']['T']
    w(f"- 정의를 TRAIN 패치 단독 추정치로 바꾸면 MAIN TEST에서 경기의 {100 * cT['matches_changed_frac']:.2f} %가 바뀌고(노출 행 기준 제거 {cT['rows_removed']:,}/추가 {cT['rows_added']:,}), 유효 T 행은 {rT['population']['old_rows']:,} → {rT['population']['dev_rows']:,}(재구축 {rT['population']['rebuilt_rows']:,})이다.")
    w(f"- 같은 동결 LightGBM의 T Brier는 옛 모집단 {f(rT['old_metrics_named']['all']['lgbm_winner']['brier'])} → dev 모집단 {f(rT['metrics_named']['all']['frozen_lgbm']['brier'])}(공통 행만 {f(rT['metrics_by_row_source_DESCRIPTIVE']['parent']['frozen_lgbm']['brier'])}); "
      f"차이는 행 구성의 차이이지 모델 차이가 아니며 두 모집단 사이에 구간은 계산하지 않는다.")
    d1 = pair(rT['bootstrap']['all'], 'refit_lgbm', 'frozen_lgbm')
    if d1:
        w(f"- 주 대비(dev 모집단, T, LightGBM): 재적합 − 동결 ΔBrier {fs(d1['brier']['estimate'])} {ci(d1['brier']['ci95'])} → {verdict(d1['brier'])}. 재적합이 동결 모델과 구별되지 않으면 정의 변경은 학습된 q를 바꾸지 않는다.")
    w('- 판정 규칙: 짝지은 구간이 0을 포함하지 않으면 우세. 구간은 고정 모델의 평가 표본 불확실성만 반영하며, 정의 변경의 "순수 효과"라는 표현은 쓰지 않는다(모집단이 다르다).')
    w('')
    w('## 7. 검증과 실패 기록')
    w('')
    if val:
        w(f"- validation.json: {val['n_checks']}개 검사, 실패 {len(val['failed'])}개 ({val['checked_at']})." + (f" 실패: {val['failed']}" if val['failed'] else ''))
    else:
        w('- validation.json 미생성.')
    w(f"- 계약 테스트 run{ct['run']} {ct['passed_count']} passed ({ct['at']}); 스냅샷 → 테스트 → 프로토콜 → 재탐지(동결 검사·dev 전 세트) → 조사 → fixture → TRAIN/VALIDATION 재구축 → 테스트 → 재적합·동결 → TEST/외부 재구축 → 봉인 평가. 동결 {fz['frozen_at']}.")
    if fails:
        w('- 보존된 실패/재시도: ' + '; '.join(f"{x['time']} `{x['stage']}`: {x['error'][:120]}" for x in fails))
    w('')
    w('## 8. 한계와 비주장')
    w('')
    w('- 15.14 + 15.15 합동 추정치는 만들지 않았다(패치별 명세만 존재). V·라벨 정의·코호트 규칙은 그대로다. 사람 검토·인과 해석은 포함하지 않는다.')
    w('- 옛 모집단과 dev 모집단의 지표 차이는 서술적 비교이며, 정의 상수 이외의 탐지 스위치(병합·길이 제한·presence)는 바꾸지 않았다.')
    w('')
    w('## 9. 산출물')
    w('')
    for line in ('`protocol.json` 두 정의·반올림 규칙·주 수량', '`redetect/` 재탐지 csv·`frozen_check.json`', '`census.json`, `changed/<set>.json`', '`rebuild/` fixture·세트별 요약, `extract/`, `labels/`, `cohorts/` (dev 모집단)',
                 '`selection/`, `models/`, `predictions/`, `internal_stop/`, `frozen_manifest.json`', '`eval/results.json`, `eval/predictions/<set>_h90_<cohort>.npz` (row_source 포함)', '`validation.json`, `integrity/`, `status.json`, `logs/`, `commands.txt`, `failures.jsonl`'):
        w(f'- {line}')
    (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    D = ['# 정의·근거·상태 (개발 패치 전용 정의 재탐지, 2026-09-16)', '', '| 요소 | 동결 | 이번(dev) | 근거 |', '|---|---|---|---|',
         f"| G | 13.7 s (pooled {d['frozen']['spec']['gap_s']:.4f}) | 14.0 s (15.14 {d['dev']['spec']['gap_s']:.4f}) | spec_pooled / spec_15.14 |",
         f"| D | 4,264 u (pooled {d['frozen']['spec']['diameter_u']:.2f}) | 4,285 u (15.14 {d['dev']['spec']['diameter_u']:.2f}) | 같은 반올림 규칙 |",
         '| 탐지 코드 | 부모 | 동일(동결 상수 재현 정확) | redetect/frozen_check.json |', '| 추출·라벨 | 부모 | 동일 함수·동결 V (fixture 비트 일치) | rebuild/fixture_check.json |', '',
         f"- frozen_manifest sha256 {C.sha256_file(DD.frozen_path(OUT))}; results sha256 {C.sha256_file(OUT / 'eval' / 'results.json')}; " + (f"validation {val['n_checks']}검사 실패 {len(val['failed'])}" if val else 'validation 미실행') + '.']
    (OUT / 'DEFINITION_AND_EVIDENCE.md').write_text('\n'.join(D) + '\n', encoding='utf-8')
    C.write_json(OUT / 'report_manifest.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), results_sha256=C.sha256_file(OUT / 'eval' / 'results.json')))
    print('wrote REPORT.md and DEFINITION_AND_EVIDENCE.md')


if __name__ == '__main__':
    main()
