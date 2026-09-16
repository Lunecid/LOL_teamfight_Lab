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
import cs20260916_common as CS  # noqa: E402

OUT = CS.OUT
GK = {'time': '시간', 'economy_and_experience': '경제·경험치', 'combat_and_survival': '전투·생존', 'objectives': '오브젝트', 'structures': '구조물', 'health_mana_other': '체력·마나·기타',
      'champion_identity': '챔피언 정체성', 'prior_win_probability': 'p_pre(사전 승률)', 'time_and_observation_age': '시간'}
MK = {'final_q': '최종 q(LightGBM)', 'plain_mlp': 'plain MLP'}


def pct(x):
    return f'{100 * x:.1f} %'


def main():
    CS.log_command()
    proto = C.read_json(OUT / 'protocol.json')
    fz = C.read_json(CS.frozen_path(OUT))
    summ = C.read_json(OUT / 'shap' / 'shap_summary.json')
    rash = {coh: C.read_json(OUT / 'rashomon' / f'{coh}.json') for coh in CS.COHORTS}
    val = C.read_json(OUT / 'validation.json') if (OUT / 'validation.json').exists() else None
    ct = C.read_json(OUT / 'contract_tests' / 'result.json')
    fails = [json.loads(l) for l in (OUT / 'failures.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (OUT / 'failures.jsonl').exists() else []
    L = []
    w = L.append
    w('# 합성 시스템 F(S) = q(h(S), V(S))의 설명과 모형군 의존도 범위 — 실행 보고서')
    w('')
    w(f'실행 {CS.VERSION}. 설계·구현·실행·검증: Claude(Codex 라인 계승, 2026-09-16). 새 학습 없음. Part A: 마스킹한 원 상태에서 파생 열(×time 94, time² 1)과 p_pre = V(S)를 다시 계산하는 '
      f'합성 함수의 7그룹 정확 Shapley(128 연합, 배경 {proto["part_a"]["n_background"]}행), 균형 SHAP과 같은 MAIN TEST T 행({proto["part_a"]["n_explain"]}행/셀). '
      f'Part B: 동일 352열 후보 78개(코호트당)의 Q_SELECT 기반 ε-Rashomon 집합과 그룹 순열 의존도 범위. 기존 TEST 노출 후의 탐색적 실행이며, 귀속은 적합된 모형의 서술이지 인과 기여가 아니다.')
    w('')
    w('## 1. Part A — 합성 설명 vs 입력 공간 설명 (T, 평균 |φ| 비중)')
    w('')
    for m in CS.MODELS:
        w(f'### {MK[m]}')
        w('')
        w('| 그룹 | 전체 셀 합성 | 전체 셀 입력 공간(균형 SHAP) | B40 합성 | B40 입력 공간 |')
        w('|---|---:|---:|---:|---:|')
        ca, cb = summ['models'][m]['cells']['all'], summ['models'][m]['cells']['B40']
        ia, ib = ca['input_space_balanced_shap']['share_of_sum_abs'], cb['input_space_balanced_shap']['share_of_sum_abs']
        rows = [('prior_win_probability', None, ia.get('prior_win_probability'), None, ib.get('prior_win_probability'))]
        for g in CS.BASE_GROUPS:
            gi = 'time_and_observation_age' if g == 'time' else g
            rows.append((g, ca['share_of_sum_abs'][g], ia.get(gi), cb['share_of_sum_abs'][g], ib.get(gi)))
        for g, a, b, c_, d in rows:
            w(f"| {GK[g]} | {pct(a) if a is not None else '(재계산됨)'} | {pct(b) if b is not None else '—'} | {pct(c_) if c_ is not None else '(재계산됨)'} | {pct(d) if d is not None else '—'} |")
        w(f"| Σ 평균 \\|φ\\| | {ca['sum_mean_abs_composite']:.4f} | {ca['input_space_balanced_shap']['sum_mean_abs']:.4f} | {cb['sum_mean_abs_composite']:.4f} | {cb['input_space_balanced_shap']['sum_mean_abs']:.4f} |")
        w('')
        w(f"- 검사: 재구성 최대 오차 상태 {ca['checks']['reconstruction']['state_max_abs']:.1e} / p_pre {ca['checks']['reconstruction']['p_pre_max_abs']:.1e}; 효율성 {ca['checks']['additivity_max_abs']:.1e}; "
          f"F = 부모 저장 예측 최대 차 {ca['checks']['F_equals_parent_stored_prediction_max_abs']:.1e}; 행 동일 {ca['checks']['rows_equal_balanced_shap']}; 3행 재실행 비트 일치 {ca['checks']['rerun_first3_bitwise_equal']}.")
        w('')
    w('해석 규칙: 합성 설명에는 p_pre 그룹이 없다(재계산되므로). 입력 공간에서 p_pre가 가졌던 기여가 어느 상태 그룹으로 흘러가는지가 이 표의 질문이다. 두 설명은 다른 대상(q의 입력 vs 원 상태 전체)에 대한 것이며 "수정본"이 아니다. 챔피언 정체성 그룹은 V 경로로만 F에 들어간다.')
    w('')
    w('## 2. Part B — ε-Rashomon 집합의 그룹 의존도 범위 (Q_SELECT, 순열 시 Brier 증가)')
    w('')
    for coh in CS.COHORTS:
        r = rash[coh]
        w(f"### {coh} — 후보 {r['n_candidates']}개, 최저 Brier {r['best_brier']:.6f} ({r['best_candidate']})")
        w('')
        w('| ε | 집합 크기 | family 구성 | p_pre 1위 비율 | 서로 다른 순서 수 | ' + ' | '.join(GK[g] + ' [min, max]' for g in CS.INPUT_GROUPS) + ' |')
        w('|---|---:|---|---:|---:|' + '---|' * len(CS.INPUT_GROUPS))
        for k, s_ in r['epsilon_sets'].items():
            fams = ', '.join(f'{f} {n}' for f, n in s_['families'].items() if n)
            w(f"| {s_['epsilon']} | {s_['n_members']} | {fams} | {pct(s_['p_pre_rank_1_fraction']) if s_['p_pre_rank_1_fraction'] is not None else '—'} | {s_['distinct_orderings']} | "
              + ' | '.join(f"[{s_['reliance_range'][g]['min']:+.5f}, {s_['reliance_range'][g]['max']:+.5f}]" for g in CS.INPUT_GROUPS) + ' |')
        w('')
        top = r['epsilon_sets'][str(CS.PRIMARY_EPS)]['most_relied_group_counts']
        w(f"- ε = {CS.PRIMARY_EPS} 집합에서 가장 의존하는 그룹의 빈도: " + ', '.join(f'{GK[g]} {n}' for g, n in sorted(top.items(), key=lambda kv: -kv[1])) + '.')
        w('')
    w('## 3. 해석')
    w('')
    w('- Part A의 비중 이동은 "원 상태가 V 경로와 직접 경로를 합쳐 최종 예측에 어떻게 반영되는가"에 대한 답이다. 마스킹은 여전히 주변 배경(개입적)이라 상관 특징의 비현실 조합은 남아 있고, p_pre·파생 열의 일관성만 확보됐다.')
    w('- Part B의 범위가 좁으면 검증 성능이 동급인 모든 후보가 그 그룹에 비슷하게 의존한다는 뜻이고, 넓으면 귀속이 모형 선택에 달렸다는 뜻이다. 이는 Fisher 등의 MCR 개념을 적합된 후보 집합에 경험적으로 적용한 것이며, 모형군 전체의 최적화가 아니다.')
    w('- 순열 의존도는 주변 순열이므로 의존 그룹을 분포 밖으로 밀어낸다(개입적 SHAP과 같은 한계). 인과 해석 없음.')
    w('')
    w('## 4. 검증과 실패 기록')
    w('')
    if val:
        w(f"- validation.json: {val['n_checks']}개 검사, 실패 {len(val['failed'])}개 ({val['checked_at']})." + (f" 실패: {val['failed']}" if val['failed'] else ''))
    else:
        w('- validation.json 미생성.')
    w(f"- 계약 테스트 run{ct['run']} {ct['passed_count']} passed ({ct['at']}); 스냅샷 → 테스트 → 프로토콜·smoke 동결 → smoke → Part B(미봉인) → 테스트 → 동결 → Part A(봉인 TEST 행) → 스냅샷 → 사후검증. 동결 {fz['frozen_at']}.")
    if fails:
        w('- 보존된 실패/재시도: ' + '; '.join(f"{x['time']} `{x['stage']}`: {x['error'][:120]}" for x in fails))
    w('')
    w('## 5. 산출물')
    w('')
    for line in ('`protocol.json` 그룹·행·모형·ε·검사', '`rashomon/<cohort>.json` 후보별 의존도·ε 집합·범위', '`frozen_manifest.json`', '`shap/shap_<model>_T_<cell>.npz`(합성 φ + 입력 공간 φ), `shap/shap_summary.json`',
                 '`validation.json`, `integrity/`, `status.json`, `logs/`, `commands.txt`, `failures.jsonl`'):
        w(f'- {line}')
    (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    D = ['# 정의·근거·상태 (합성 SHAP·Rashomon, 2026-09-16)', '', '| 요소 | 균형 SHAP(③) | 이번 | 근거 |', '|---|---|---|---|',
         '| 설명 대상 | q의 입력 공간(352열, p_pre 포함 7그룹) | 합성 F(S): 원 상태 266 기저열 7그룹, p_pre·파생 열 재계산 | 명세 Part A |',
         '| 행·배경·모형 | MAIN TEST T 256행/셀, TRAIN 배경 128, iq LightGBM·Track A MLP | 동일(키 일치 검사) | shap_summary checks |',
         '| 모형군 의존도 | — | Q_SELECT ε-Rashomon 78후보/코호트, 순열 의존도 범위 | rashomon/<cohort>.json |', '',
         f"- frozen_manifest sha256 {C.sha256_file(CS.frozen_path(OUT))}; shap_summary sha256 {C.sha256_file(OUT / 'shap' / 'shap_summary.json')}; " + (f"validation {val['n_checks']}검사 실패 {len(val['failed'])}" if val else 'validation 미실행') + '.']
    (OUT / 'DEFINITION_AND_EVIDENCE.md').write_text('\n'.join(D) + '\n', encoding='utf-8')
    C.write_json(OUT / 'report_manifest.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), shap_summary_sha256=C.sha256_file(OUT / 'shap' / 'shap_summary.json')))
    print('wrote REPORT.md and DEFINITION_AND_EVIDENCE.md')


if __name__ == '__main__':
    main()
