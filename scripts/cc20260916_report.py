"""Stage R: Korean REPORT.md and DEFINITION_AND_EVIDENCE.md from saved artifacts only (no new computation on data)."""
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
import cc20260916_common as CC  # noqa: E402

OUT = CC.OUT
ARM_KO = {'base': '기준(352열)', 'tags': '+슬롯 태그', 'class_state': '+클래스 상태 집계', 'class_pairs': '+클래스 쌍(정글-미드·바텀·매치업)',
          'identity': '+챔피언 정체성 one-hot', 'class_pairs_identity': '+클래스 쌍+정체성', 'draft_class': '클래스 조합만(상태 없음)', 'draft_identity': '정체성만(상태 없음)'}
FAM_KO = {'lgbm': 'LightGBM', 'logit': 'logistic'}
CELL_KO = {'all': '전체', 'B40': 'B40', 'B45': 'B45'}


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
    CC.log_command()
    proto = C.read_json(OUT / 'protocol.json')
    fz = C.read_json(CC.frozen_path(OUT))
    res = C.read_json(OUT / 'eval' / 'results.json')
    val = C.read_json(OUT / 'validation.json') if (OUT / 'validation.json').exists() else None
    ct = C.read_json(OUT / 'contract_tests' / 'result.json')
    R = res['results']
    fails = [json.loads(l) for l in (OUT / 'failures.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (OUT / 'failures.jsonl').exists() else []
    L = []
    w = L.append
    w('# 챔피언 클래스 특징이 q를 개선하는가: 슬롯 태그·클래스 조건부 상태·포지션 쌍 클래스 조합 vs 챔피언 정체성 (h90, T/N) — 실행 보고서')
    w('')
    w(f'실행 {CC.VERSION}. 설계·구현·실행·검증: Claude(Codex 라인 계승, 2026-09-16; 사용자 제안). 구성은 iq h90 승자(T: {proto["fixed_configurations"]["registry"]["T"]}, '
      f'N: {proto["fixed_configurations"]["registry"]["N"]})로 고정하고 팔(arm)마다 보정만 Q_SELECT에서 선택했다. 클래스 출처는 Data Dragon 태그 6종(패치별 표, sha 고정), '
      f'정체성 어휘는 TRAIN 챔피언 {proto["identity_vocab"]["n"]}종. 기존 TEST/외부 노출 후의 탐색적 후속 실험이며, 라벨 Y_h90=1[ΔV_h90>0]는 모델 정의 결과이지 실제 한타 승리 정답이 아니다. '
      f'사전 등록된 주 비교: T · LightGBM · class_pairs − base · 전체 셀 · Brier.')
    w('')
    w('## 1. 핵심 요약 (MAIN TEST 15.16 전체 셀; 짝지은 경기 부트스트랩 1000회, seed 20260915; Δ = 팔 − base, 음수 = 팔 우세)')
    w('')
    for coh, title in (('T', 'T(한타, 주)'), ('N', 'N(비한타, 보조)')):
        r = R['MAIN_TEST'][coh]
        w(f'### {title} — {r["rows"]:,}행/{r["matches"]:,}경기')
        w('')
        w('| 팔 | 입력 폭 | LightGBM Brier | ΔBrier vs base [95%] | 판정 | LightGBM AUC | logistic Brier | ΔBrier vs base [95%] | 판정 | LightGBM B40 Brier |')
        w('|---|---:|---:|---|---|---:|---:|---|---|---:|')
        mn, m40, bt = r['metrics_named']['all'], r['metrics_named']['B40'], r['bootstrap']['all']
        for arm in CC.ARMS:
            cells = [ARM_KO[arm], str(proto['features']['arms'][arm]['width'])]
            for fam in ('lgbm', 'logit'):
                nm = f'{fam}_{arm}'
                cells.append(f(mn[nm]['brier']))
                if arm == 'base':
                    cells += ['—', '기준']
                else:
                    d = pair(bt, nm, f'{fam}_base')
                    cells += ([f"{fs(d['brier']['estimate'])} {ci(d['brier']['ci95'])}", verdict(d['brier'])] if d else ['미계산', '—'])
                if fam == 'lgbm':
                    cells.append(f(mn[nm]['auc'], 4))
            cells.append(f(m40['lgbm_' + arm]['brier']))
            w('| ' + ' | '.join(cells) + ' |')
        w('')
        w(f"- 부모 iq 승자(같은 구성, 같은 데이터) Brier: LightGBM {f(mn['parent_lgbm_winner']['brier'])}, logistic {f(mn['parent_logit_winner']['brier'])}; "
          f"이 실행의 base 팔과의 최대 절대 차이 {r['checks']['parent_vs_base_max_abs_diff_DESCRIPTIVE']}.")
        for fam in ('lgbm', 'logit'):
            parts = []
            for a, b in (('class_pairs', 'identity'), ('class_pairs_identity', 'identity'), ('class_pairs_identity', 'class_pairs')):
                d = pair(bt, f'{fam}_{a}', f'{fam}_{b}')
                if d:
                    parts.append(f"{a} − {b}: {fs(d['brier']['estimate'])} {ci(d['brier']['ci95'])} ({verdict(d['brier'])})")
            w(f"- {FAM_KO[fam]} 클래스 vs 정체성: " + '; '.join(parts) + '.')
        w(f"- 조합 단독(상태 없음) AUC: 클래스 조합 LightGBM {f(mn['lgbm_draft_class']['auc'], 4)} / logistic {f(mn['logit_draft_class']['auc'], 4)}; "
          f"정체성 LightGBM {f(mn['lgbm_draft_identity']['auc'], 4)} / logistic {f(mn['logit_draft_identity']['auc'], 4)} (0.5 = 정보 없음).")
        w('')
    pd = pair(R['MAIN_TEST']['T']['bootstrap']['all'], 'lgbm_class_pairs', 'lgbm_base')
    if pd:
        w(f"**주 비교(사전 등록)**: T · LightGBM · class_pairs − base · 전체 · ΔBrier {fs(pd['brier']['estimate'])} {ci(pd['brier']['ci95'])} → {verdict(pd['brier'])}; "
          f"ΔAUC {fs(pd['auc']['estimate'], 4)} {ci(pd['auc']['ci95'], 4)}; Δlog loss {fs(pd['logloss']['estimate'])} {ci(pd['logloss']['ci95'])}.")
        w('')
    w('## 2. 균형 상태 셀(B40)과 시간대: LightGBM class_pairs − base (MAIN TEST)')
    w('')
    w('| cohort | 셀 | 행 | base Brier | class_pairs Brier | identity Brier | ΔBrier class_pairs−base [95%] | 판정 |')
    w('|---|---|---:|---:|---:|---:|---|---|')
    for coh in CC.COHORTS:
        r = R['MAIN_TEST'][coh]
        for cell in ('all', 'B40', 'B45', 'time_0_10', 'time_10_20', 'time_20_30', 'time_30_inf'):
            mn = r['metrics_named'][cell]
            d = pair(r['bootstrap'][cell], 'lgbm_class_pairs', 'lgbm_base') if cell in r['bootstrap'] else None
            dv, vd = (f"{fs(d['brier']['estimate'])} {ci(d['brier']['ci95'])}", verdict(d['brier'])) if d else ('미계산(시간 셀은 구간 없음)' if cell.startswith('time') else '미계산', '—')
            w(f"| {coh} | {cell} | {r['cells'][cell]['rows']:,} | {f(mn['lgbm_base']['brier'])} | {f(mn['lgbm_class_pairs']['brier'])} | {f(mn['lgbm_identity']['brier'])} | {dv} | {vd} |")
    w('')
    w('## 3. 외부 세트(전체 행; 1년 뒤 패치, 미학습 챔피언 포함; 이전 노출 있음)')
    w('')
    w('| 세트 | cohort | 행 | LightGBM base | class_pairs | identity | class_pairs_identity | draft_class AUC | Δ class_pairs−base [95%] | 판정 | Δ identity−base [95%] |')
    w('|---|---|---:|---:|---:|---:|---:|---:|---|---|---|')
    for set_name in CC.EVAL_SETS[1:]:
        for coh in CC.COHORTS:
            r = R[set_name][coh]
            mn = r['metrics_named']['all']
            bt = r['bootstrap']['all']
            d = pair(bt, 'lgbm_class_pairs', 'lgbm_base')
            di = pair(bt, 'lgbm_identity', 'lgbm_base')
            dv, vd = (f"{fs(d['brier']['estimate'])} {ci(d['brier']['ci95'])}", verdict(d['brier'])) if d else ('미계산', '—')
            w(f"| {set_name} | {coh} | {r['rows']:,} | {f(mn['lgbm_base']['brier'])} | {f(mn['lgbm_class_pairs']['brier'])} | {f(mn['lgbm_identity']['brier'])} | "
              f"{f(mn['lgbm_class_pairs_identity']['brier'])} | {f(mn['lgbm_draft_class']['auc'], 4)} | {dv} | {vd} | "
              f"{(fs(di['brier']['estimate']) + ' ' + ci(di['brier']['ci95'])) if di else '미계산'} |")
    w('')
    w('## 4. 선정된 보정과 정지 기록')
    w('')
    w('| cohort | family | 팔 | 입력 | 선정 보정 | Q_SELECT Brier | 정지 반복(시드 7/42/123) | 특징 초 | 적합 초 |')
    w('|---|---|---|---:|---|---:|---|---:|---:|')
    for coh in CC.COHORTS:
        for fam in CC.FAMILIES:
            for arm in CC.ARMS:
                key = f'{fam}_{arm}_{coh}'
                s = C.read_json(OUT / 'selection' / f'{key}.json')
                stop = '—'
                if fam in CC.SEEDED:
                    sr = C.read_json(OUT / 'internal_stop' / f'{key}.json')['stop_record']['seeds']
                    stop = '/'.join(str(sr[str(sd)]['best_iteration']) for sd in CC.SEEDS[fam])
                w(f"| {coh} | {FAM_KO[fam]} | {arm} | {s['n_inputs']} | {s['chosen_calibration']} | {f(s['select_metrics'][s['chosen']]['brier'], 6)} | {stop} | {s['fit_record']['seconds_features']} | {s['fit_record']['seconds_fit']} |")
    w('')
    w('## 5. 해석')
    w('')
    for coh in CC.COHORTS:
        bt = R['MAIN_TEST'][coh]['bootstrap']['all']
        if bt.get('computed', True) is False:
            continue
        for fam in CC.FAMILIES:
            w(f"- {coh} 전체 · {FAM_KO[fam]}: " + '; '.join(f"{p['a'].split('_', 1)[1]} {fs(p['a_minus_b']['brier']['estimate'])} → {verdict(p['a_minus_b']['brier'])}"
                                                          for p in bt['pairs'] if p['a'].startswith(fam) and p['b'] == f'{fam}_base') + '.')
        mn = R['MAIN_TEST'][coh]['metrics_named']['all']
        best = {fam: min((a for a in CC.ARMS), key=lambda a: mn[f'{fam}_{a}']['brier']) for fam in CC.FAMILIES}
        w(f"- {coh} 전체 셀 최저 Brier 팔(서술): " + ', '.join(f'{FAM_KO[fam]} {best[fam]}' for fam in CC.FAMILIES) + '.')
    w('- 판정 규칙: Δ(팔 − base)의 95 % 구간이 0을 포함하지 않으면서 음수이면 팔 우세. 구간은 고정 모델의 평가 표본 불확실성만 반영하며(학습·보정·선정 불확실성 제외), 20개 대비에 다중 비교 보정은 없다.')
    w('- 챔피언 정보는 라벨에 정확히 상쇄되므로(V 분해) 조합은 "누가 한타를 이기는가"를 통해서만 q에 들어온다. 구성(하이퍼파라미터)이 352열 승자로 고정돼 있어 더 넓은 입력에서 최적이라는 보장은 없다.')
    w('')
    w('## 6. 검증과 실패 기록')
    w('')
    if val:
        w(f"- validation.json: {val['n_checks']}개 검사, 실패 {len(val['failed'])}개 ({val['checked_at']})." + (f" 실패: {val['failed']}" if val['failed'] else ''))
    else:
        w('- validation.json 미생성.')
    w(f"- 계약 테스트 run{ct['run']} {ct['passed_count']} passed ({ct['at']}); 스냅샷 → 테스트 → 프로토콜 → smoke(TRAIN 1/8, 32 적합) → 테스트 → 전체 적합(32) → 동결 → 봉인 평가 순서. 동결 {fz['frozen_at']}.")
    if fails:
        w('- 보존된 실패/재시도: ' + '; '.join(f"{x['time']} `{x['stage']}`: {x['error'][:120]}" for x in fails))
    w('')
    w('## 7. 한계와 비주장')
    w('')
    w('- 태그 6종은 거친 분류이며 세부 클래스(13종)는 다루지 않았다. 챔피언 단위 시너지는 데이터에서 반복되지 않아(경기당 고유 5인 조합) 정체성 팔은 챔피언별 가산 효과 이상을 학습할 수 없다.')
    w('- MLP/GPU 팔, 그래프 표현(Track B), V·라벨 변경, 인과 해석, 사람 검토는 포함하지 않는다. 결과는 고정 구성의 표 형식 학습기에 한한다.')
    w('')
    w('## 8. 산출물')
    w('')
    for line in ('`ddragon/` 승인된 Data Dragon 다운로드·`tag_table.json`', '`protocol.json` 사전 동결 계약(팔·블록·어휘·주 비교)', '`selection/<family>_<arm>_<cohort>.json` 3후보 지표·선정·특징 행렬 hash',
                 '`internal_stop/` LightGBM 정지 기록', '`models/<cohort>/<family>/<arm>/<config>.joblib` 번들', '`frozen_manifest.json` 동결(32 선정, 부모 참조 hash)',
                 '`eval/results.json`, `eval/predictions/<set>_h90_<cohort>.npz`', '`validation.json` 사후 검증', '`integrity/`, `status.json`, `logs/`, `commands.txt`, `failures.jsonl`'):
        w(f'- {line}')
    (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    D = ['# 정의·근거·상태 (챔피언 클래스 특징, 2026-09-16)', '', '| 요소 | 이전 | 이번 | 근거 |', '|---|---|---|---|',
         '| 챔피언 정보 in q | 없음(p_pre 경유 가산항뿐) | 슬롯 태그·클래스 집계·클래스 쌍·정체성 one-hot 팔 | 명세 §Feature blocks |',
         '| 클래스 출처 | — | Data Dragon tags 6종, 패치별 표(sha 고정) | ddragon/fetch_manifest.json |',
         '| 구성 | iq h90 승자 | 고정(재선정 없음), 팔마다 보정만 선택 | 프로토콜 규칙 |',
         '| 주 비교 | — | T · LightGBM · class_pairs − base · 전체 · Brier | protocol.json evaluation.primary |', '',
         f"- frozen_manifest sha256 {C.sha256_file(CC.frozen_path(OUT))}; results sha256 {C.sha256_file(OUT / 'eval' / 'results.json')}; "
         + (f"validation {val['n_checks']}검사 실패 {len(val['failed'])}" if val else 'validation 미실행') + '.']
    (OUT / 'DEFINITION_AND_EVIDENCE.md').write_text('\n'.join(D) + '\n', encoding='utf-8')
    C.write_json(OUT / 'report_manifest.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), results_sha256=C.sha256_file(OUT / 'eval' / 'results.json')))
    print('wrote REPORT.md and DEFINITION_AND_EVIDENCE.md')


if __name__ == '__main__':
    main()
