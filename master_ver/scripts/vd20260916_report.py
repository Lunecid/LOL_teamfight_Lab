"""Stage R: Korean REPORT.md from saved artifacts only."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import vd20260916_common as V  # noqa: E402

OUT = V.OUT
G_KO = {'time_and_observation_age': '시간(time_minutes, sq)', 'economy_and_experience': '경제·경험치', 'combat_and_survival': '전투·생존', 'objectives': '오브젝트',
        'structures': '구조물', 'health_mana_other': '체력·마나·기타', 'champion_identity': '챔피언 one-hot'}


def f(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def pct(x):
    return '—' if x is None else f'{100 * x:.1f}%'


def main():
    V.log_command()
    res = C.read_json(OUT / 'results.json')
    val = C.read_json(OUT / 'validation.json') if (OUT / 'validation.json').exists() else None
    proto = C.read_json(OUT / 'protocol.json')
    L = []
    w = L.append
    w('# 라벨의 작동 원리: 동결 V의 Δlogit 열별 정확 분해 (h90) — 실행 보고서')
    w('')
    w(f"실행 {V.VERSION}. 새 학습 없음. 동결 V(ColumnTransformer 361→{res['transformed_columns']}열 + LogisticRegression C={proto['model']['C']}, raw 보정)에 대해 "
      'Δlogit = βᵀ(t(S_e) − t(S_pre)) = Σ_j c_j 를 모든 h90 유효 행에서 계산하고 7블록(6그룹 + 챔피언 one-hot)과 시간 관련 열로 합산했다. '
      'Y = 1[Δlogit > 0]이 정확히 성립한다(단조 sigmoid, raw 보정). 기존 TEST 노출 후의 탐색적 분석이며, 항의 크기는 적합된 선형 가치 모형의 서술이지 인과 기여가 아니다.')
    w('')
    ch = res['checks']
    w(f"검사: 행 {ch['rows']:,}, p_pre·p_post 부모 라벨과 정확 일치 {ch['p_pre_exact']:,}/{ch['p_pre_rows']:,}·{ch['p_post_exact']:,}/{ch['p_pre_rows']:,}, "
      f"Σ블록−Δlogit 최대 {ch['max_abs_sum_minus_decision']:.1e}, 챔피언 블록 최대 {ch['max_abs_champion']:.1e}, logit 차−Δlogit 최대 {ch['max_abs_logit_diff_minus_decision']:.1e}, "
      f"sign(Δlogit)=Y {ch['sign_equals_Y']:,}/{ch['sign_rows']:,}, 통과 {ch['pass']}.")
    w('')
    for s in V.SETS:
        S = res['sets'][s]
        w(f"## {s} — {S['rows']:,}행/{S['matches']:,}경기 (동일 프레임 비율 T {pct(S['same_frame_share']['T'])}, N {pct(S['same_frame_share']['N'])})")
        w('')
        for coh in ('T', 'N'):
            a = S['strata'][f'{coh}:all']
            w(f"### {coh} 전체 (n={a['n']:,}; 평균 |Δlogit| {f(a['mean_abs_delta_logit'])}, 중앙 {f(a['median_abs_delta_logit'])}, 평균 |ΔV| {f(a['mean_abs_delta_V'])}, 양성률 {f(a['positive_rate'], 3)})")
            w('')
            w('| 블록 | 평균 |g| | Σ|g| 비중 | 평균 g(부호) | 최대 블록 빈도 | 부호 결정 블록 빈도 |')
            w('|---|---:|---:|---:|---:|---:|')
            for g in V.GROUPS:
                w(f"| {G_KO[g]} | {f(a['mean_abs_group'][g])} | {pct(a['share_of_sum_abs'][g])} | {a['mean_signed_group'][g]:+.4f} | {pct(a['dominant_group_frequency'][g])} | {pct(a['sign_deciding_group_frequency'][g])} |")
            w('')
            w(f"- 시간 관련 열(time_minutes, time_minutes_sq, 모든 ×time)의 Σ|c_j| 비중: {pct(a['time_related_share_of_sum_abs_columns'])} (그룹 비중과 겹치므로 합산하지 않음). "
              f"최대 블록의 부호가 Δlogit과 같은 비율 {pct(a['dominant_group_sign_agrees_with_delta'])}; 한 블록만으로 |Δlogit|을 넘는 행 {pct(a['single_group_exceeds_total_share'])}.")
            w('')
            w('| 층 | n | 평균 |Δlogit| | 양성률 | ' + ' | '.join(G_KO[g] + ' 비중' for g in V.GROUPS if g != 'champion_identity') + ' | 시간열 비중 | 부호결정 1위 |')
            w('|---|---:|---:|---:|' + '---:|' * (len(V.GROUPS) - 1) + '---:|---|')
            for key, lab in [(f'{coh}:same_frame', '동일 프레임'), (f'{coh}:new_frame', '새 프레임')] + [(f'{coh}:absdelta_{V.stratum_name(k)}', f'|ΔV| {V.stratum_name(k)}') for k in range(len(V.DELTA_STRATA))] \
                    + [(f'{coh}:time_{lo}_{hi if hi < 1000 else "inf"}', f'{lo}–{hi if hi < 1000 else "∞"}분') for lo, hi in V.BANDS]:
                x = S['strata'].get(key, {})
                if not x or x.get('n', 0) == 0:
                    w(f'| {lab} | 0 | — | — |' + ' — |' * (len(V.GROUPS) - 1) + ' — | — |')
                    continue
                top = max(x['sign_deciding_group_frequency'], key=x['sign_deciding_group_frequency'].get)
                w(f"| {lab} | {x['n']:,} | {f(x['mean_abs_delta_logit'])} | {f(x['positive_rate'], 3)} | " + ' | '.join(pct(x['share_of_sum_abs'][g]) for g in V.GROUPS if g != 'champion_identity')
                  + f" | {pct(x['time_related_share_of_sum_abs_columns'])} | {G_KO[top]} |")
            w('')
    w('## 상위 열 (전체 수집 행 기준 평균 |c_j|; 변환 열 이름)')
    w('')
    w('| 순위 | 열 | 그룹 | 시간 관련 | 평균 |c_j| | 평균 c_j | β_j |')
    w('|---:|---|---|---|---:|---:|---:|')
    for i, t in enumerate(res['top_columns_by_mean_abs_contribution'], 1):
        w(f"| {i} | `{t['column']}` | {G_KO[t['group']]} | {'예' if t['time_related'] else ''} | {f(t['mean_abs'])} | {t['mean_signed']:+.4f} | {t['beta']:+.4f} |")
    w('')
    w('## 해석')
    w('')
    w('- 라벨은 선형 가치 모형의 로짓 차분이며, 챔피언 one-hot 같은 정적 항은 정확히 상쇄된다. 남는 것은 두 시점 사이에 변한 상태 열의 β 가중 변화량이다.')
    w('- 동일 프레임 행(마지막 킬 이후 새 프레임이 없는 경우)에서는 프레임 기반 열(경제·체력 등)이 변하지 않고, 사건 기반 열(킬·사망·오브젝트·건물)과 시간 열만 변한다. 그 층의 블록 비중이 이를 보여준다.')
    w('- "부호 결정 블록"은 |g|가 큰 순서로 훑어 Δlogit과 같은 부호를 가진 첫 블록이며, 한 블록만으로 |Δlogit|을 넘는 행의 비율은 나머지 블록이 반대 방향으로 상쇄되는 빈도를 뜻한다.')
    w('- 시간 관련 열 비중은 그룹 분류와 겹치므로 별도로 보고하며 합산하지 않는다. V(S(L))가 저장되지 않아 pre→L→e 분리는 하지 않았다.')
    w('')
    w('## 검증')
    w('')
    if val:
        w(f"- validation.json: {val['n_checks']}개 검사, 실패 {len(val['failed'])}개 ({val['checked_at']})." + (f" 실패: {val['failed']}" if val['failed'] else ''))
    w('- 부모 p_pre/p_post 정확 재현, Σ블록 = Δlogit, 챔피언 블록 = 0, logit 차 = Δlogit, sign = Y 전 행, 유효 행 전수 1회 수집, 저장 배열로부터 T 요약 재계산, 봉인 접근 시점, 부모 무결성.')
    (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    C.write_json(OUT / 'report_manifest.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), results_sha256=C.sha256_file(OUT / 'results.json')))
    print('wrote REPORT.md')


if __name__ == '__main__':
    main()
