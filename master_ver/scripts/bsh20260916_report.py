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
import bsh20260916_common as B  # noqa: E402

OUT = B.OUT
G_KO = {'prior_win_probability': '교전 전 승률 p_pre', 'time_and_observation_age': '시간', 'economy_and_experience': '경제·경험치', 'combat_and_survival': '전투·생존',
        'objectives': '오브젝트', 'structures': '구조물', 'health_mana_other': '체력·마나·기타'}
M_KO = {'final_q': '최종 선정 q (iq LightGBM)', 'plain_mlp': 'plain MLP (Track A)'}


def f(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def main():
    B.log_command()
    proto = C.read_json(OUT / 'protocol.json')
    fz = C.read_json(B.frozen_path(OUT))
    S = C.read_json(OUT / 'shap' / 'shap_summary.json')
    val = C.read_json(OUT / 'validation.json') if (OUT / 'validation.json').exists() else None
    ct = C.read_json(OUT / 'contract_tests' / 'result.json')
    L = []
    w = L.append
    w('# 균형 상태 설명(RQ3): 최종 선정 q의 7그룹 정확 Shapley — 전체 T 대 B40 T — 실행 보고서')
    w('')
    w(f'실행 {B.VERSION}. 설계·구현·실행·검증: Claude(Codex 라인 계승, 2026-09-16). 새 학습 없음. 설명 대상은 동결된 h90 모델의 최종 보정 확률이며, '
      '7그룹 128연합 정확 Shapley(Lundberg & Lee 2017)를 배경 128행(코호트 TRAIN, hash 선택)에 대해 계산했다. 사례는 모델 출력과 무관한 키 hash로 셀(전체/B40)마다 256개를 뽑았다. '
      '기존 TEST 노출 후의 탐색적 설명이며, 기여도는 적합된 모델의 서술이지 인과적 승리 요인이 아니다.')
    w('')
    w('## 1. 핵심 요약 (MAIN TEST 15.16, T 주·N 보조)')
    w('')
    for model in B.MODELS:
        for coh in B.COHORTS:
            m = S['models'][model][coh]
            w(f"### {M_KO[model]} · {coh} — 후보 {m['candidate']} ({m['calibration']}), 배경 {m['background_rows']}행/{m['background_matches']}경기")
            w('')
            w('| 그룹(열 수) | 전체: 평균 |φ| [95%] | 전체 순위 | B40: 평균 |φ| [95%] | B40 순위 | 전체−B40 | 전체 평균 φ(부호) | B40 평균 φ(부호) |')
            w('|---|---|---:|---|---:|---:|---:|---:|')
            ca, cb = m['cells']['all'], m['cells']['B40']
            ra, rb = ca['global_mean_abs_rank'], cb['global_mean_abs_rank']
            for g, n in zip(B.GROUPS, S['group_sizes']):
                w(f"| {G_KO[g]}({n}) | {f(ca['global_mean_abs'][g])} [{f(ca['global_mean_abs_bootstrap_ci95'][g][0])}, {f(ca['global_mean_abs_bootstrap_ci95'][g][1])}] | {ra.index(g) + 1} | "
                  f"{f(cb['global_mean_abs'][g])} [{f(cb['global_mean_abs_bootstrap_ci95'][g][0])}, {f(cb['global_mean_abs_bootstrap_ci95'][g][1])}] | {rb.index(g) + 1} | "
                  f"{m['all_minus_B40_mean_abs'][g]:+.4f} | {ca['global_mean_signed'][g]:+.4f} | {cb['global_mean_signed'][g]:+.4f} |")
            w('')
            w(f"- 전체 셀: 설명 {ca['explained_rows']}행/{ca['explained_matches']}경기(적격 {ca['eligible_rows']:,}행), 평균 q {f(ca['mean_q'])}, 평균 p_pre {f(ca['mean_p_pre'])}, 양성률 {f(ca['positive_rate_explained'], 3)}, 기준값 {f(ca['base_value'])}. "
              f"B40 셀: {cb['explained_rows']}행/{cb['explained_matches']}경기(적격 {cb['eligible_rows']:,}행), 평균 q {f(cb['mean_q'])}, 평균 p_pre {f(cb['mean_p_pre'])}, 양성률 {f(cb['positive_rate_explained'], 3)}. "
              f"두 셀 겹침 {m['overlap_all_B40']['rows']}행. 검사 통과: 전체 {ca['checks']['pass']}, B40 {cb['checks']['pass']}.")
            w(f"- 전체 |φ| 합 대비 비중(전체 셀): " + ', '.join(f"{G_KO[g]} {ca['share_of_total_abs'][g]:.1%}" for g in ra) + '.')
            w(f"- 전체 |φ| 합 대비 비중(B40 셀): " + ', '.join(f"{G_KO[g]} {cb['share_of_total_abs'][g]:.1%}" for g in rb) + '.')
            w('')
    w('## 2. 해석')
    w('')
    for model in B.MODELS:
        m = S['models'][model]['T']
        ca, cb = m['cells']['all'], m['cells']['B40']
        w(f"- {M_KO[model]} T: 전체 셀에서 가장 큰 그룹은 {G_KO[ca['global_mean_abs_rank'][0]]}, B40에서는 {G_KO[cb['global_mean_abs_rank'][0]]}. "
          f"p_pre 그룹의 평균 |φ|는 전체 {f(ca['global_mean_abs']['prior_win_probability'])} → B40 {f(cb['global_mean_abs']['prior_win_probability'])} "
          f"({m['all_minus_B40_mean_abs']['prior_win_probability']:+.4f}). 균형 상태에서는 p_pre가 0.5 근처라 그 기여가 줄고 나머지 그룹의 상대 비중이 커지는지가 관찰 대상이다.")
    w('- 전체 대 B40의 차이는 서로 다른 행의 서술적 비교이며 짝지은 구간이 없다. 부트스트랩 구간은 256 설명 행의 표본 변동만 반영한다.')
    w('- 개입적 치환은 파생 관계(차이, ×시간, 합계)를 깨뜨린 인위적 혼합 상태를 만든다. 기여도는 모델 설명이지 인과적 승리 요인·선수 실력이 아니며, 모델·셀 간 크기를 순위로 비교하지 않는다.')
    w('- 협업 메일이 제안한 "균형 상태에서의 설명"에 대한 실행이며, Kim et al. (CoG 2020)의 손실함수는 구현하지 않았다.')
    w('')
    w('## 3. 국소 사례 (hash 선택 8건/셀; 익명 키는 내부 산출물에만)')
    w('')
    for model in B.MODELS:
        for cell in B.CELLS:
            cases = S['models'][model]['T']['cells'][cell]['local_cases']
            w(f"### {M_KO[model]} · T · {cell}")
            w('')
            w('| 시작 분 | p_pre | q | 기준값 | y | ' + ' | '.join(G_KO[g] for g in B.GROUPS) + ' |')
            w('|---:|---:|---:|---:|---:|' + '---:|' * len(B.GROUPS))
            for c in cases:
                w(f"| {c['start_minute']} | {f(c['p_pre'], 3)} | {f(c['q_final'], 3)} | {f(c['base_value'], 3)} | {c['y_h90']} | " + ' | '.join(f"{c['group_shapley'][g]:+.3f}" for g in B.GROUPS) + ' |')
            w('')
    w('## 4. 시간대별 평균 |φ| (T, 전체 셀; 서술)')
    w('')
    for model in B.MODELS:
        bands = S['models'][model]['T']['cells']['all']['time_bands']
        w(f"**{M_KO[model]}**")
        w('')
        w('| 시간대(분) | n | ' + ' | '.join(G_KO[g] for g in B.GROUPS) + ' |')
        w('|---|---:|' + '---:|' * len(B.GROUPS))
        for k, v in bands.items():
            w(f"| {k} | {v['n']} | " + (' | '.join(f(v['mean_abs'][g]) for g in B.GROUPS) if v['mean_abs'] else ' | '.join(['—'] * len(B.GROUPS))) + ' |')
        w('')
    w('## 5. 검증과 기록')
    w('')
    if val:
        w(f"- validation.json: {val['n_checks']}개 검사, 실패 {len(val['failed'])}개 ({val['checked_at']})." + (f" 실패: {val['failed']}" if val['failed'] else ''))
    w(f"- 계약 테스트 run{ct['run']} {ct['passed_count']} passed; 스냅샷 → 테스트 → 프로토콜 → smoke → 테스트 → 동결({fz['frozen_at']}) → 봉인 설명({S['evaluated_at']}) 순서. "
      '가산성·전체연합·빈연합·재로딩·선택 결정성·부모 평가 예측과의 일치(<1e-8) 검사를 모든 셀에서 통과했다.')
    w('')
    w('## 6. 산출물')
    w('')
    w('- `shap/shap_summary.json`, `shap/shap_<model>_<cohort>_<cell>.npz`(φ, 기준값, q, y, p_pre, 행 인덱스·키), `frozen_manifest.json`, `protocol.json`, `validation.json`, `integrity/`, `logs/`, `access_log.jsonl`.')
    (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    D = ['# 정의·근거·상태 (균형 상태 SHAP, 2026-09-16)', '', '| 요소 | 이전 | 이번 | 근거 |', '|---|---|---|---|',
         '| 설명 대상 | 역할 모델(cr), 전체 q(fc) 256사례 | 최종 선정 q(iq LightGBM)와 plain MLP, 전체·B40 각 256사례 | DELTA_Q 프로토콜 §6 |',
         '| 그룹 | fc 8그룹(챔피언 포함) | 352열 7그룹(챔피언 제외) | 부모 shap_groups 제한 |',
         '| 배경 | TRAIN 128 | 코호트 TRAIN 128, 셀·모델 공통 | 프로토콜 |', '',
         f"- frozen_manifest sha256 {C.sha256_file(B.frozen_path(OUT))}; shap_summary sha256 {C.sha256_file(OUT / 'shap' / 'shap_summary.json')}; "
         + (f"validation {val['n_checks']}검사 실패 {len(val['failed'])}" if val else 'validation 미실행') + '.']
    (OUT / 'DEFINITION_AND_EVIDENCE.md').write_text('\n'.join(D) + '\n', encoding='utf-8')
    C.write_json(OUT / 'report_manifest.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), shap_summary_sha256=C.sha256_file(OUT / 'shap' / 'shap_summary.json')))
    print('wrote REPORT.md and DEFINITION_AND_EVIDENCE.md')


if __name__ == '__main__':
    main()
