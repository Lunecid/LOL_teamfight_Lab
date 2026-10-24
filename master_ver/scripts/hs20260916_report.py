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
import hs20260916_common as HS  # noqa: E402

OUT = HS.OUT
LABEL = {'pt_winner': 'PT 기준선', 'logit_winner': '전체 logistic', 'lgbm_winner': '전체 LightGBM', 'mlp_winner': 'plain MLP', 'resmlp_winner': 'residual MLP',
         'overall_winner': '15후보 1위', 'old_A_specialist': '기존 A specialist', 'old_pooled': '기존 pooled q', 'old_p_pre_spline': '기존 p_pre spline',
         'old_p_pre_logistic': '기존 p_pre logistic', 'old_constant': '기존 상수'}
MODELS = ('lgbm_winner', 'mlp_winner', 'resmlp_winner', 'logit_winner', 'pt_winner', 'old_A_specialist', 'old_pooled', 'old_p_pre_spline', 'old_constant')
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


def main():
    HS.log_command()
    proto = C.read_json(OUT / 'protocol.json')
    fz = C.read_json(HS.frozen_path(OUT))
    res = C.read_json(OUT / 'eval' / 'results.json')
    val = C.read_json(OUT / 'validation.json') if (OUT / 'validation.json').exists() else None
    ct = C.read_json(OUT / 'contract_tests' / 'result.json')
    R = res['results']
    horizons = res['horizons']
    iq_res = C.read_json(HS.IQ / 'eval' / 'results.json')['results']
    ta_res = C.read_json(HS.TA / 'eval' / 'results.json')['results']
    fails = [json.loads(l) for l in (OUT / 'failures.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (OUT / 'failures.jsonl').exists() else []
    L = []
    w = L.append
    w('# 종료 상한 민감도: h90 선정 학습기 5종을 h60·h120 라벨로 재적합 (T/N) — 실행 보고서')
    w('')
    w(f'실행 {HS.VERSION}. 설계·구현·실행·검증: Claude(Codex 라인 계승, 2026-09-16). 구성(하이퍼파라미터)은 h90 승자(iq·Track A)로 고정하고 보정만 Q_SELECT에서 선택했다. '
      'h90 열은 재적합하지 않았고 iq/Track A 결과를 서술적으로 병기한다(라벨이 달라 상한 간 구간은 계산하지 않음). 기존 TEST/외부 노출 후의 탐색적 후속 실험이며, '
      '라벨 Y_h=1[ΔV_h>0]는 모델 정의 결과이지 실제 한타 승리 정답이 아니다.')
    w('')
    w('## 1. 핵심 요약 (MAIN TEST 15.16, 짝지은 경기 부트스트랩 1000회, seed 20260915)')
    w('')
    for h in horizons:
        for coh, title in (('T', 'T(한타, 주)'), ('N', 'N(비한타, 보조)')):
            r = R[f'h{h}']['MAIN_TEST'][coh]
            w(f'### h{h} · {title} — {r["rows"]:,}행/{r["matches"]:,}경기')
            w('')
            for cell in ('all', 'B40'):
                mn = r['metrics_named'][cell]
                w(f"- {CELL_KO[cell]} Brier: " + '; '.join(f'{LABEL[m]} {f(mn[m]["brier"])}' for m in MODELS[:5]) + f"; 기존 A specialist {f(mn['old_A_specialist']['brier'])}.")
                bt = r['bootstrap'][cell]
                if bt.get('computed', True) is False:
                    w(f'- {CELL_KO[cell]}: 부트스트랩 미계산({bt["reason"]}).')
                    continue
                for p in bt['pairs']:
                    d = p['a_minus_b']
                    w(f"- {CELL_KO[cell]} · {p['label']}: ΔBrier {fs(d['brier']['estimate'])} {ci(d['brier']['ci95'])} ({verdict(d['brier'])}), ΔAUC {fs(d['auc']['estimate'], 4)} {ci(d['auc']['ci95'], 4)}.")
            w('')
    w('## 2. 상한별 비교(서술; MAIN TEST 전체 셀 Brier / AUC; h90은 iq·Track A의 동결 결과, 같은 행)')
    w('')
    w('| cohort | 모델 | h60 Brier | h90 Brier | h120 Brier | h60 AUC | h90 AUC | h120 AUC |')
    w('|---|---|---:|---:|---:|---:|---:|---:|')
    for coh in HS.COHORTS:
        for fam in HS.FAMILIES:
            src90 = (ta_res if fam in HS.TA_FAMILIES else iq_res)['MAIN_TEST'][coh]['metrics_named']['all'][f'{fam}_winner']
            row = {h: R[f'h{h}']['MAIN_TEST'][coh]['metrics_named']['all'][f'{fam}_winner'] for h in horizons}
            w(f"| {coh} | {LABEL[f'{fam}_winner']} | {f(row[60]['brier'])} | {f(src90['brier'])} | {f(row[120]['brier'])} | {f(row[60]['auc'], 4)} | {f(src90['auc'], 4)} | {f(row[120]['auc'], 4)} |")
        for name in ('old_A_specialist', 'old_pooled', 'old_constant'):
            src90 = iq_res['MAIN_TEST'][coh]['metrics_named']['all'][name]
            row = {h: R[f'h{h}']['MAIN_TEST'][coh]['metrics_named']['all'][name] for h in horizons}
            w(f"| {coh} | {LABEL[name]} | {f(row[60]['brier'])} | {f(src90['brier'])} | {f(row[120]['brier'])} | {f(row[60]['auc'], 4)} | {f(src90['auc'], 4)} | {f(row[120]['auc'], 4)} |")
    w('')
    w('주의: 상한이 다르면 라벨 Y가 달라지므로 열 사이의 차이는 학습기 차이가 아니라 라벨(종료점) 차이를 포함한다. 같은 열 안의 순서만 학습기 비교다. '
      'h90의 기존 참조(A specialist·pooled·상수)는 iq 평가에서, h60·h120은 cohort_role 참조를 이 실행에서 같은 행에 붙인 것이다.')
    w('')
    w('## 3. 선정된 보정과 정지 기록')
    w('')
    w('| horizon | cohort | family | 고정 구성 | 선정 보정 | Q_SELECT Brier | 정지 기록(시드 7/42/123) | 적합 초 |')
    w('|---|---|---|---|---|---:|---|---:|')
    for h in horizons:
        for coh in HS.COHORTS:
            for fam in HS.FAMILIES:
                key = f'h{h}_{fam}_{coh}'
                s = C.read_json(OUT / 'selection' / f'{key}.json')
                stop = '—'
                if fam in HS.SEEDED:
                    sr = C.read_json(OUT / 'internal_stop' / f'{key}.json')['stop_record']['seeds']
                    k = 'best_iteration' if fam == 'lgbm' else 'best_epoch'
                    stop = ('반복 ' if fam == 'lgbm' else 'epoch ') + '/'.join(str(sr[str(sd)][k]) for sd in HS.SEEDS[fam])
                w(f"| h{h} | {coh} | {LABEL[f'{fam}_winner']} | {s['config']} | {s['chosen_calibration']} | {f(s['select_metrics'][s['chosen']]['brier'], 6)} | {stop} | {s['fit_record']['seconds_fit']} |")
    w('')
    w('## 4. 외부 세트(전체 행; 주 대비 LightGBM − PT의 판정; 이전 노출 있음)')
    w('')
    w('| horizon | 세트 | cohort | 행 | LightGBM Brier | plain MLP | residual MLP | logistic | PT | LGBM−PT ΔBrier [95%] | 판정 |')
    w('|---|---|---|---:|---:|---:|---:|---:|---:|---|---|')
    for h in horizons:
        for set_name in HS.EVAL_SETS[1:]:
            for coh in HS.COHORTS:
                r = R[f'h{h}'][set_name][coh]
                mn = r['metrics_named']['all']
                bt = r['bootstrap']['all']
                dv, vd = ('미계산', '—') if bt.get('computed', True) is False else (f"{fs(bt['pairs'][0]['a_minus_b']['brier']['estimate'])} {ci(bt['pairs'][0]['a_minus_b']['brier']['ci95'])}", verdict(bt['pairs'][0]['a_minus_b']['brier']))
                w(f"| h{h} | {set_name} | {coh} | {r['rows']:,} | {f(mn['lgbm_winner']['brier'])} | {f(mn['mlp_winner']['brier'])} | {f(mn['resmlp_winner']['brier'])} | {f(mn['logit_winner']['brier'])} | {f(mn['pt_winner']['brier'])} | {dv} | {vd} |")
    w('')
    w('## 5. 해석')
    w('')
    for h in horizons:
        for coh in HS.COHORTS:
            bt = R[f'h{h}']['MAIN_TEST'][coh]['bootstrap']['all']
            if bt.get('computed', True) is False:
                continue
            w(f"- h{h} {coh} 전체: " + '; '.join(f"{p['label'].split(': ')[1]} {fs(p['a_minus_b']['brier']['estimate'])} → {verdict(p['a_minus_b']['brier'])}" for p in bt['pairs']) + '.')
    w('- 학습기 순서가 상한에 따라 바뀌는지가 이 실험의 질문이다. 위 판정을 h90(iq: LightGBM ≈ logistic; Track A: plain MLP ≈ LightGBM > residual MLP, N에서 LightGBM > MLP > logistic)와 견주어 읽는다.')
    w('- 부트스트랩 구간은 고정 모델의 평가 표본 불확실성만 반영한다(학습·보정·선정 불확실성 제외, 다중 비교 보정 없음). 구성은 h90에서 선정된 것이므로 h60/h120에서 최적이라는 보장이 없다.')
    w('')
    w('## 6. 검증과 실패 기록')
    w('')
    if val:
        w(f"- validation.json: {val['n_checks']}개 검사, 실패 {len(val['failed'])}개 ({val['checked_at']})." + (f" 실패: {val['failed']}" if val['failed'] else ''))
    else:
        w('- validation.json 미생성.')
    w(f"- 계약 테스트 run{ct['run']} {ct['passed_count']} passed ({ct['at']}); 스냅샷 → 테스트 → 프로토콜 → smoke(h60) → 테스트 → 전체 적합(h60, h120) → 동결 → 봉인 평가 순서. 동결 {fz['frozen_at']}.")
    if fails:
        w('- 보존된 실패/재시도: ' + '; '.join(f"{x['time']} `{x['stage']}`: {x['error'][:120]}" for x in fails))
    w('')
    w('## 7. 한계와 비주장')
    w('')
    w('- h60/h120은 보조 결과이며 주 분석은 h90이다. 상한을 TEST 결과로 고르지 않았다.')
    w('- 구성 고정·보정만 선택이므로 각 상한에서의 최적 구성 비교가 아니다. 사람 검토·인과효과·미접촉 확증은 포함하지 않는다.')
    w('')
    w('## 8. 산출물')
    w('')
    for line in ('`protocol.json` 사전 동결 계약·고정 구성 등록부', '`selection/h<h>_<family>_<cohort>.json` 3후보 지표·선정', '`internal_stop/` LightGBM·MLP 정지 기록',
                 '`models/h<h>/<cohort>/<family>/<config>.joblib` 번들', '`frozen_manifest.json` 동결(두 상한, 부모 참조 hash)', '`eval/results.json`, `eval/predictions/<set>_h<h>_<cohort>.npz`',
                 '`validation.json` 사후 검증', '`integrity/`, `status.json`, `logs/`, `commands.txt`, `failures.jsonl`'):
        w(f'- {line}')
    (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    D = ['# 정의·근거·상태 (종료 상한 민감도, 2026-09-16)', '', '| 요소 | 이전 | 이번 | 근거 |', '|---|---|---|---|',
         '| 상한 | h90 주 분석(iq, Track A) | h60·h120 재적합 | DELTA_Q 프로토콜 §3-6 |',
         '| 구성 | h90에서 선정 | 고정(재선정 없음), 보정만 선택 | 프로토콜 규칙 |',
         '| 라벨 | Y_h90 | Y_h60, Y_h120(부모 라벨, 동일 종료 규칙) | 부모 계약 |',
         '| 행 | h90 유효 | 상한별 유효 = 동일 수(코호트 manifest) | cohort_manifest.json |', '',
         f"- frozen_manifest sha256 {C.sha256_file(HS.frozen_path(OUT))}; results sha256 {C.sha256_file(OUT / 'eval' / 'results.json')}; "
         + (f"validation {val['n_checks']}검사 실패 {len(val['failed'])}" if val else 'validation 미실행') + '.']
    (OUT / 'DEFINITION_AND_EVIDENCE.md').write_text('\n'.join(D) + '\n', encoding='utf-8')
    C.write_json(OUT / 'report_manifest.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), results_sha256=C.sha256_file(OUT / 'eval' / 'results.json')))
    print('wrote REPORT.md and DEFINITION_AND_EVIDENCE.md')


if __name__ == '__main__':
    main()
