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
import iq20260915_common as Q  # noqa: E402

OUT = Q.OUT
LABEL = {'pt_winner': 'PT 기준선(신규)', 'logit_winner': '전체 logistic(신규)', 'lgbm_winner': '전체 LightGBM(신규)', 'overall_winner': '전체 1위(신규)',
         'old_A_specialist': '기존 A specialist', 'old_pooled': '기존 pooled q', 'old_p_pre_spline': '기존 p_pre spline',
         'old_p_pre_logistic': '기존 p_pre logistic', 'old_constant': '기존 상수'}
MODELS = ('pt_winner', 'logit_winner', 'lgbm_winner', 'old_A_specialist', 'old_pooled', 'old_p_pre_spline', 'old_p_pre_logistic', 'old_constant')
CELL_KO = {'all': '전체', 'B40': 'B40', 'B45': 'B45', 'time_0_10': '[0,10)분', 'time_10_20': '[10,20)분', 'time_20_30': '[20,30)분',
           'time_30_inf': '[30,∞)분'}


def f(x, d=5):
    return '—' if x is None else f'{x:.{d}f}'


def fs(x, d=5):
    return '—' if x is None else f'{x:+.{d}f}'


def ci(c, d=5):
    return '—' if not c else f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'


def verdict(delta):
    c = delta.get('ci95')
    if delta.get('estimate') is None or not c:
        return '구간 없음'
    if c[1] < 0:
        return 'a 우세(구간 0 미포함)'
    if c[0] > 0:
        return 'b 우세(구간 0 미포함)'
    return '구간이 0 포함'


def auc_verdict(delta):
    c = delta.get('ci95')
    if delta.get('estimate') is None or not c:
        return '구간 없음'
    if c[0] > 0:
        return 'a 우세(구간 0 미포함)'
    if c[1] < 0:
        return 'b 우세(구간 0 미포함)'
    return '구간이 0 포함'


def esc(s):
    return str(s).replace('|', '\\|')


def main():
    Q.log_command()
    proto = C.read_json(OUT / 'protocol.json')
    fz = C.read_json(OUT / 'frozen_manifest.json')
    res = C.read_json(OUT / 'eval' / 'results.json')
    val = C.read_json(OUT / 'validation.json')
    diff = C.read_json(OUT / 'integrity' / 'snapshot_diff.json')
    sels = {f'{fa}_{c}': C.read_json(OUT / 'selection' / f'{fa}_{c}.json') for fa in Q.FAMILIES for c in Q.COHORTS}
    stops = {c: C.read_json(OUT / 'internal_stop' / f'lgbm_{c}.json') for c in Q.COHORTS}
    ct = C.read_json(OUT / 'contract_tests' / 'result.json')
    fails = [json.loads(l) for l in (OUT / 'failures.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (OUT / 'failures.jsonl').exists() else []
    assert res['complete'] and res['frozen_hashes_unchanged'], 'evaluation incomplete'
    assert set(res['results']) == set(Q.EVAL_SETS)
    R = res['results']
    L = []
    w = L.append
    ok_val = not val['failed']
    w('# 교전 전 승률+시간 강화 기준선 대 동일 352입력 전체 상태 q (h90, T/N) — 실행 보고서')
    w('')
    w(f'작성: {time.strftime("%Y-%m-%d %H:%M")} (Claude Opus 5 구현·실행, Codex 설계·독립 감사 예정). 명세: `docs/CLAUDE_INCREMENTAL_Q_TRAIN_20260915.md` '
      f'(sha256 `{proto["spec_sha256"][:16]}…`). 실행 루트: `outputs/incremental_q_training_20260915`.')
    w('')
    w('**성격:** 기존 TEST 15.16·외부 세트 결과를 이미 본 뒤의 **탐색적 후속 실험**이다. 미개봉 확인 연구가 아니며, 라벨 Y는 모델이 정의한 '
      'ΔV 개선 방향(정답 아님)이다. 최종 승패 W의 인과효과나 Y의 의미 타당성은 검증하지 않았다.')
    w('')
    w(f'**검증 상태:** validation.json {val["n_checks"]}개 검사 중 실패 {len(val["failed"])}개'
      + (' — 모든 필수 검사 통과.' if ok_val else f' — 실패: {", ".join(val["failed"])}. 아래 결과는 해당 실패를 감안해 읽어야 한다.'))
    w('')
    # ------------------------------------------------------------------ summary
    w('## 1. 핵심 요약')
    w('')
    for coh, title in (('T', 'T(한타, 주 분석)'), ('N', 'N(비한타, 보조)')):
        r = R['MAIN_TEST'][coh]
        w(f'### {title} — MAIN TEST 15.16')
        w('')
        w(f'- 선정(Q_SELECT, TEST 미사용): PT `{r["winners"]["pt"]}`, 전체 logistic `{r["winners"]["logit"]}`, 전체 LightGBM `{r["winners"]["lgbm"]}`; '
          f'전체 1위 `{r["overall"]}`.')
        for c in ('all', 'B40', 'B45'):
            mm = r['metrics_named'][c]
            best = min((mm[m]['brier'], m) for m in MODELS if mm[m]['brier'] is not None)
            w(f'- {CELL_KO[c]} ({r["cells"][c]["rows"]:,}행/{r["cells"][c]["matches"]:,}경기) Brier 점추정 최저(서술): {LABEL[best[1]]} {best[0]:.5f}; '
              f'신규 PT {f(mm["pt_winner"]["brier"])}, logistic {f(mm["logit_winner"]["brier"])}, LightGBM {f(mm["lgbm_winner"]["brier"])}, '
              f'기존 A {f(mm["old_A_specialist"]["brier"])}, pooled {f(mm["old_pooled"]["brier"])}.')
        for c in ('all', 'B40'):
            bt = r['bootstrap'].get(c, {})
            if bt.get('computed', True) is False:
                w(f'- {CELL_KO[c]}: 부트스트랩 미계산 ({bt["reason"]}).')
                continue
            for p in bt['pairs']:
                d = p['a_minus_b']
                w(f'- {CELL_KO[c]} · {esc(p["label"])}: ΔBrier {fs(d["brier"]["estimate"])} {ci(d["brier"]["ci95"])} ({verdict(d["brier"])}), '
                  f'Δlog loss {fs(d["logloss"]["estimate"])} {ci(d["logloss"]["ci95"])}, ΔAUC {fs(d["auc"]["estimate"], 4)} {ci(d["auc"]["ci95"], 4)}.')
        w('')
    prim = R['MAIN_TEST']['T']['bootstrap']['all']['pairs'][0]
    assert prim['a'] == 'lgbm_winner' and prim['b'] == 'pt_winner'
    prim40 = R['MAIN_TEST']['T']['bootstrap']['B40']['pairs'][0] if R['MAIN_TEST']['T']['bootstrap']['B40'].get('pairs') else None
    w('**사전 지정 주 대비(전체 LightGBM − PT, T h90):** 전체 행 ' + verdict(prim['a_minus_b']['brier'])
      + (f', B40 {verdict(prim40["a_minus_b"]["brier"])}' if prim40 else '') + '. 방향과 무관하게 모든 계획 대비를 아래에 그대로 보고한다. '
      'TEST에서 더 좋은 쌍을 주 대비로 승격하지 않았다.')
    w('')
    # ------------------------------------------------------------------ before / after
    w('## 2. 진행 전/후')
    w('')
    w('| 항목 | 이번 실행 전 | 이번 실행 후 |')
    w('|---|---|---|')
    w(f'| p_pre 기준선 | 기존 cohort specialist 후보 안의 p_pre 단일 spline(균일 knot, C=1)·logistic | p_pre_V×time_minutes 텐서 spline(quantile knot, 48열) + C 6개 × 보정 3개, Q_SELECT 선정 |')
    w(f'| 동일 352입력 logistic | ridge C=0.01 고정 1개(기존 specialist 후보) | C 6개(1e-4…10), tol 1e-8, 보정 3개 = 18 후보 |')
    w(f'| 동일 352입력 LightGBM | 없음(기존 LightGBM은 경제 입력 부분집합, 250 trees 고정) | num_leaves×min_child_samples 6개, 내부 stop10 경기가중 Brier 조기종료, 3 seed 평균, 보정 3개 |')
    w(f'| 기존 선택(h90) | T `{fz["legacy_references"]["A_specialist_h90_chosen"]["T"]}`, N `{fz["legacy_references"]["A_specialist_h90_chosen"]["N"]}`, pooled `{fz["legacy_references"]["pooled_h90_chosen"]}` | 참조로만 유지(재적합·재선정 없음) |')
    w(f'| 선정·동결 | — | {fz["frozen_at"]} 동결 후 TEST/외부 개봉(access_log.jsonl) |')
    w('| 미실행(연기) | — | h60/h120 재적합, MLP·residual MLP, CoG 전체 lineup, SHAP, V 재적합, logit-label 분해, 원시 시간 보정, 탐지기 재학습, C 역할 모델 비교 |')
    w('')
    # ------------------------------------------------------------------ data contract
    w('## 3. 데이터 계약과 사전 검사')
    w('')
    w('| 역할 | T 행 | N 행 | 비고 |')
    w('|---|---:|---:|---|')
    for k in ('TRAIN', 'Q_CAL', 'Q_SELECT', 'MAIN_TEST', 'EXT_KR_16.13', 'EXT_KR_16.14_pilot', 'EXT_KR_16.15', 'EXT_NA1_16.13'):
        note = {'TRAIN': '15.14 전체 적격 행으로 최종 재적합', 'Q_CAL': '15.15 보정기 적합', 'Q_SELECT': '15.15 후보 선정', 'MAIN_TEST': '15.16 봉인 평가'}.get(k, '외부(이전 노출 있음)')
        w(f'| {k} | {Q.EXPECTED_COUNTS[k]["T"]:,} | {Q.EXPECTED_COUNTS[k]["N"]:,} | {note} |')
    w('')
    pv = {c: sels[f'pt_{c}']['oof_provenance'] for c in Q.COHORTS}
    assert all(p['rows'] == p['adapter_id_equals_own_heldout_fold'] == p['adapter_sha_equals_manifest_oof_hash'] for p in pv.values())
    w(f'- 명세 기대 행 수와 모든 적합·평가 행 수가 일치했다(불일치 시 적합 차단 규칙). TRAIN 행의 p_pre/Y는 자기 경기 제외 OOF V: '
      f'T {pv["T"]["rows"]:,}행·N {pv["N"]["rows"]:,}행 전부 adapter_id=자기 held-out fold, adapter sha256=V manifest OOF hash '
      f'(부모 manifest: 자기 경기 포함 위반 0, fit set 재구성 일치). V는 재적합하지 않았다.')
    w('- 기존 참조 열(A specialist·pooled)의 전체 행 Brier/log loss/AUC는 부모 `eval/results_A.json` 보고값과 정확히 일치했다(최대 차이 0).')
    w(f'- 입력: 부모 `q_pre_only_schema.json` predictor_sets["ridge"] 352열(순서 고정, p_pre_V 포함, 챔피언 ID·snapshot_age 제외, sha256 '
      f'`{proto["inputs"]["ridge_names_sha256"][:16]}…`). 모든 적합·평가 행 유한값(대치기는 항등). PT 입력은 p_pre_V와 time_minutes(q_pre/60000)뿐.')
    w(f'- 계약 테스트 {ct["passed_count"]}개 통과(run {ct["run"]}; 합성·TRAIN-only). 부모 테스트 모음은 부모 루트에 임시 디렉터리를 만들기 때문에 재실행하지 않았다.')
    w('- TRAIN-only smoke(1/8 경기 부분집합, fold0-2/3/4 가상 역할)로 적합→동결→평가 경로를 먼저 실행했다. smoke 모델은 최종 모델이 아니다.')
    w('')
    # ------------------------------------------------------------------ families
    w('## 4. 적합 기록')
    w('')
    w('### 4.1 PT 설계와 logistic 수렴')
    w('')
    for coh in Q.COHORTS:
        pt = sels[f'pt_{coh}']['fit_records']
        d0 = pt[Q.config_names('pt')[0]]['design']
        w(f'- {coh} PT: 설계 {d0["columns"]}열, 표준화 후 rank {d0["rank_standardized"]}, p 내부 knot 고유값 {d0["pt"]["p_interior_knots_unique"]}, '
          f't 내부 knot 고유값 {d0["pt"]["t_interior_knots_unique"]} (중복 knot 없음 여부 = {d0["pt"]["p_interior_knots_unique"] == 5 and d0["pt"]["t_interior_knots_unique"] == 5}).')
        w(f'  - p knot(quantile): {", ".join(f"{x:.4f}" for x in d0["pt"]["p_knot_vector"][3:-3])}; t knot(분): {", ".join(f"{x:.2f}" for x in d0["pt"]["t_knot_vector"][3:-3])}')
    w('')
    w('| cohort | family | C | 반복 수 | 수렴 | 재시도 | 적합 초 | 진단 max|grad| |')
    w('|---|---|---:|---:|---|---|---:|---:|')
    for coh in Q.COHORTS:
        for fa in ('pt', 'logit'):
            for cfg, fr in sels[f'{fa}_{coh}']['fit_records'].items():
                a = fr['attempts'][-1]
                w(f'| {coh} | {fa} | {a["C"]:g} | {a["n_iter"]} | {a["converged"]} | {len(fr["attempts"]) - 1} | {fr["seconds_fit"]} | {a["diagnostic_max_abs_gradient"]:.2e} |')
    retries = [(coh, fa, cfg, fr['attempts'][0]['n_iter'], fr['attempts'][-1]['n_iter'], fr['attempts'][-1]['converged'])
               for coh in Q.COHORTS for fa in ('pt', 'logit') for cfg, fr in sels[f'{fa}_{coh}']['fit_records'].items() if len(fr['attempts']) > 1]
    w('')
    if retries:
        w('- 사전 선언 재시도(첫 시도 max_iter 5000 미수렴 → 20000, tol·데이터 동일, 첫 시도 계수 보존): '
          + '; '.join(f'{c} `{cfg}` 첫 시도 {a}회 → 재시도 {b}회 수렴={ok}' for c, fa, cfg, a, b, ok in retries)
          + ('. 재시도 후 미수렴 후보는 없었으며 모든 후보가 선정 적격이었다.' if all(r[5] for r in retries) else '. 재시도 후에도 미수렴한 후보는 선정에서 제외했다.'))
    else:
        w('- 재시도가 필요한 logistic 적합은 없었다.')
    w('')
    w('### 4.2 LightGBM 내부 조기종료(stop10, 경기가중 Brier, patience 50)')
    w('')
    w('| cohort | 설정 | seed 7 / 42 / 123 최적 반복 | cap(1000) 도달 | stop10 Brier(seed 평균) | 적합 초 |')
    w('|---|---|---|---|---:|---:|')
    for coh in Q.COHORTS:
        for cfg, sr in stops[coh]['configs'].items():
            ss = sr['seeds']
            w(f'| {coh} | {cfg} | {" / ".join(str(ss[str(s)]["best_iteration"]) for s in Q.LGBM_SEEDS)} | '
              f'{any(ss[str(s)]["cap_reached"] for s in Q.LGBM_SEEDS)} | {sum(ss[str(s)]["best_stop_weighted_brier"] for s in Q.LGBM_SEEDS) / 3:.5f} | '
              f'{sels[f"lgbm_{coh}"]["fit_records"][cfg]["seconds_fit"]} |')
    s0 = stops['T']['configs'][Q.config_names('lgbm')[0]]
    caps = [(coh, cfg, s, r['best_iteration']) for coh in Q.COHORTS for cfg, sr in stops[coh]['configs'].items() for s, r in sr['seeds'].items() if r['cap_reached']]
    w('')
    if caps:
        w('- **1000 tree 상한 도달(patience 50 소진 전 종료):** ' + '; '.join(f'{c} `{cfg}` seed {s} (최적 {b})' for c, cfg, s, b in caps)
          + '. 명세의 최대 1000 trees 상한을 그대로 적용했으며 상한을 늘리지 않았다. 해당 seed의 최적 반복은 상한에 의해 제한되었을 수 있다.')
    else:
        w('- 1000 tree 상한에 도달한 seed는 없었다.')
    w(f'- stop 배정은 경기 hash로 T/N 동일. 예) T: fit90 {s0["fit90"]["rows"]:,}행/{s0["fit90"]["matches"]:,}경기, stop10 {s0["stop10"]["rows"]:,}행/'
      f'{s0["stop10"]["matches"]:,}경기, 겹침 {s0["fit90_stop10_match_overlap"]}. 각 seed는 선택 반복 수로 전체 cohort TRAIN에서 처음부터 재적합했다.')
    w('')
    # ------------------------------------------------------------------ selection
    w('## 5. Q_SELECT 선정 (Brier → log loss → 이름)')
    w('')
    for coh in Q.COHORTS:
        w(f'### {coh}')
        w('')
        w('| family | 선정 후보 | Q_SELECT Brier | log loss | AUC | 적격 후보 | 상위 3 |')
        w('|---|---|---:|---:|---:|---:|---|')
        for fa in Q.FAMILIES:
            s = sels[f'{fa}_{coh}']
            m = s['select_metrics'][s['chosen']]
            w(f'| {fa} | `{s["chosen"]}` | {f(m["brier"], 6)} | {f(m["logloss"], 6)} | {f(m["auc"], 4)} | {sum(s["eligible"].values())}/18 | '
              + ', '.join(f'`{c}` {s["select_metrics"][c]["brier"]:.6f}' for c in s['ranking'][:3]) + ' |')
        o = fz['overall_winner'][coh]
        w('')
        w(f'- 전체 1위(54 후보 동일 규칙): `{o["chosen"]}`. 선택된 보정기: ' + '; '.join(
            f'{fa} sigmoid 기울기 {sels[f"{fa}_{coh}"]["calibrators"][sels[f"{fa}_{coh}"]["chosen_config"]]["sigmoid"]["slope"]:.3f}' for fa in Q.FAMILIES)
          + ' (q sigmoid에는 양의 기울기 제약 없음; 기록용).')
        w('')
    # ------------------------------------------------------------------ TEST tables
    w('## 6. 봉인 평가: MAIN TEST 15.16')
    w('')
    w('평가 가중치는 각 셀 안에서 경기별 총 가중치가 같도록 다시 계산했다. B40/B45와 시간 구간은 동결 p_pre와 교전 전 시간으로만 정해져 모든 모델에 공통이다. '
      '보정 절편/기울기는 서술용 적합이며 평가 예측을 바꾸지 않았다. 30경기 미만 셀은 희소(sparse)로 표시하고 구간·일반화 결론을 내리지 않는다.')
    w('')
    for coh in Q.COHORTS:
        r = R['MAIN_TEST'][coh]
        w(f'### 6.{1 if coh == "T" else 2} {coh} ({"주" if coh == "T" else "보조"})')
        w('')
        cells = r['cells']
        w(f'셀 크기: ' + ', '.join(f'{CELL_KO.get(k, k)} {v["rows"]:,}행/{v["matches"]:,}경기' for k, v in cells.items() if not k.startswith('B40_x')))
        w('')
        w('| 모델 | 전체 Brier | 전체 log loss | 전체 AUC | B40 Brier | B40 log loss | B40 AUC | B45 Brier | B45 AUC | 전체 ECE | 전체 보정 기울기 |')
        w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        mn = r['metrics_named']
        for m in MODELS:
            a, b, c = mn['all'][m], mn['B40'][m], mn['B45'][m]
            w(f'| {LABEL[m]} | {f(a["brier"])} | {f(a["logloss"])} | {f(a["auc"], 4)} | {f(b["brier"])} | {f(b["logloss"])} | {f(b["auc"], 4)} | '
              f'{f(c["brier"])} | {f(c["auc"], 4)} | {f(a["ece_10bin"], 4)} | {f(a["slope"], 3)} |')
        w('')
        w(f'양성률(경기가중): 전체 {f(mn["all"]["pt_winner"]["positive_rate_match_weighted"], 4)}, B40 {f(mn["B40"]["pt_winner"]["positive_rate_match_weighted"], 4)}, '
          f'B45 {f(mn["B45"]["pt_winner"]["positive_rate_match_weighted"], 4)}.')
        w('')
        w('**계획된 paired 대비 (a − b; Brier/log loss 음수 = a 우세; 1000 경기 부트스트랩, seed 20260915)**')
        w('')
        w('| 셀 | 대비 | ΔBrier [95%] | 판정 | Δlog loss [95%] | ΔAUC [95%] | 퇴화 복제 |')
        w('|---|---|---|---|---|---|---:|')
        for cname in ('all', 'B40', 'B45'):
            bt = r['bootstrap'][cname]
            if bt.get('computed', True) is False:
                w(f'| {CELL_KO[cname]} | — | 미계산: {esc(bt["reason"])} | | | | |')
                continue
            for p in bt['pairs']:
                d = p['a_minus_b']
                w(f'| {CELL_KO[cname]} | {esc(p["label"])} | {fs(d["brier"]["estimate"])} {ci(d["brier"]["ci95"])} | {verdict(d["brier"])} | '
                  f'{fs(d["logloss"]["estimate"])} {ci(d["logloss"]["ci95"])} | {fs(d["auc"]["estimate"], 4)} {ci(d["auc"]["ci95"], 4)} ({auc_verdict(d["auc"])}) | '
                  f'{bt["degenerate_single_class_replicates"]} |')
        w('')
        w('**시간 구간별 Brier (점추정만; 구간 CI는 명세대로 미계산)**')
        w('')
        tcells = [k for k in cells if k.startswith('time_')]
        w('| 모델 | ' + ' | '.join(f'{CELL_KO[k]} ({cells[k]["rows"]:,}/{cells[k]["matches"]:,})' for k in tcells) + ' | '
          + ' | '.join(f'B40×{CELL_KO[k]} ({cells["B40_x_" + k]["rows"]:,}/{cells["B40_x_" + k]["matches"]:,})' for k in tcells) + ' |')
        w('|---|' + '---:|' * (2 * len(tcells)))
        for m in MODELS:
            w(f'| {LABEL[m]} | ' + ' | '.join(f(mn[k][m]['brier']) + ('*' if mn[k][m]['sparse_lt30_matches'] else '') for k in tcells) + ' | '
              + ' | '.join(f(mn['B40_x_' + k][m]['brier']) + ('*' if mn['B40_x_' + k][m]['sparse_lt30_matches'] else '') for k in tcells) + ' |')
        w('')
        w('(*: 30경기 미만 희소 셀)')
        w('')
        seeds = r['lgbm_winner_config_seed_metrics_DESCRIPTIVE']['all']
        w(f'LightGBM 선정 설정의 seed별 전체 Brier(서술용, raw): ' + ', '.join(f'seed {s} {f(v["brier"])}' for s, v in seeds.items())
          + f'; 3-seed 평균 후 보정한 선정 후보 {f(mn["all"]["lgbm_winner"]["brier"])}.')
        w('')
    # ------------------------------------------------------------------ external
    w('## 7. 외부 세트(각각 따로; 이전 노출 있음, 새 수집·EUW 주장 없음)')
    w('')
    w('| 세트 | cohort | 셀 | 행/경기 | PT Brier | logistic Brier | LightGBM Brier | 기존 A Brier | 기존 spline Brier | LGBM−PT ΔBrier [95%] | 판정 |')
    w('|---|---|---|---|---:|---:|---:|---:|---:|---|---|')
    for s in Q.EVAL_SETS[1:]:
        for coh in Q.COHORTS:
            r = R[s][coh]
            for cname in ('all', 'B40', 'B45'):
                mn = r['metrics_named'][cname]
                bt = r['bootstrap'][cname]
                if bt.get('computed', True) is False:
                    dtxt, vtxt = f'미계산({esc(bt["reason"])})', '—'
                else:
                    d = bt['pairs'][0]['a_minus_b']['brier']
                    dtxt, vtxt = f'{fs(d["estimate"])} {ci(d["ci95"])}', verdict(d)
                sp = '*' if mn['pt_winner']['sparse_lt30_matches'] else ''
                w(f'| {s} | {coh} | {CELL_KO[cname]}{sp} | {r["cells"][cname]["rows"]:,}/{r["cells"][cname]["matches"]:,} | {f(mn["pt_winner"]["brier"])} | '
                  f'{f(mn["logit_winner"]["brier"])} | {f(mn["lgbm_winner"]["brier"])} | {f(mn["old_A_specialist"]["brier"])} | {f(mn["old_p_pre_spline"]["brier"])} | {dtxt} | {vtxt} |')
    w('')
    w('외부 세트의 나머지 계획 대비(logistic−PT, LightGBM−logistic, PT−기존 spline, LightGBM−기존 A)와 모든 모델·셀 지표는 `eval/results.json`에 있다.')
    w('')
    # ------------------------------------------------------------------ interpretation
    w('## 8. 해석(음성·혼합 결과 포함)')
    w('')
    tT = R['MAIN_TEST']['T']
    pairs = {p['label']: p for p in tT['bootstrap']['all']['pairs']}
    for lab, p in pairs.items():
        d = p['a_minus_b']['brier']
        w(f'- T 전체, {esc(lab)}: ΔBrier {fs(d["estimate"])} {ci(d["ci95"])} → {verdict(d)}.')
    if tT['bootstrap']['B40'].get('pairs'):
        for p in tT['bootstrap']['B40']['pairs']:
            d = p['a_minus_b']['brier']
            w(f'- T B40, {esc(p["label"])}: ΔBrier {fs(d["estimate"])} {ci(d["ci95"])} → {verdict(d)}.')
    nb = R['MAIN_TEST']['N']['bootstrap']['all']['pairs'][0]['a_minus_b']['brier']
    w(f'- N 전체(보조), 전체 LightGBM − PT: ΔBrier {fs(nb["estimate"])} {ci(nb["ci95"])} → {verdict(nb)}.')
    ext_votes = []
    for s in Q.EVAL_SETS[1:]:
        for coh in Q.COHORTS:
            bt = R[s][coh]['bootstrap']['all']
            if bt.get('computed', True) is not False:
                ext_votes.append((s, coh, verdict(bt['pairs'][0]['a_minus_b']['brier'])))
    w('- 외부 세트 전체 행의 주 대비 판정: ' + '; '.join(f'{s} {c}: {v}' for s, c, v in ext_votes) + '.')
    for coh in Q.COHORTS:
        mn = R['MAIN_TEST'][coh]['metrics_named']
        worse = [(c, m, mn[c][m]['brier'], mn[c]['old_constant']['brier']) for c in mn for m in MODELS[:-1]
                 if mn[c][m]['brier'] is not None and mn[c]['old_constant']['brier'] is not None and mn[c][m]['brier'] >= mn[c]['old_constant']['brier']]
        if worse:
            w(f'- **음성 결과({coh}, MAIN TEST, 점추정·CI 없음):** Brier가 TRAIN 상수 이상인 셀: '
              + '; '.join(f'{CELL_KO.get(c, c.replace("B40_x_", "B40×").replace("time_", "t"))} {LABEL[m]} {b:.5f} ≥ 상수 {k:.5f}' for c, m, b, k in worse) + '.')
    lgT = R['MAIN_TEST']['T']['metrics_named']
    w(f'- T 균형 셀에서는 모든 모델의 AUC가 낮다(B40: PT {f(lgT["B40"]["pt_winner"]["auc"], 3)}, logistic {f(lgT["B40"]["logit_winner"]["auc"], 3)}, '
      f'LightGBM {f(lgT["B40"]["lgbm_winner"]["auc"], 3)}, 기존 A {f(lgT["B40"]["old_A_specialist"]["auc"], 3)}). 전체 T에서 PT 대비 개선이 보이더라도 '
      'p_pre가 .4–.6인 한타에서 새 LightGBM이 PT를 넘는다는 근거는 이번 구간으로는 확인되지 않았고, 새 LightGBM은 기존 A specialist보다 B40/B45 Brier가 높았다.')
    nlog = sels['logit_N']
    if nlog['chosen'] == 'logit_C0.01__isotonic' and fz['legacy_references']['A_specialist_h90_chosen']['N'] == 'ridge_isotonic':
        w(f'- N 전체 logistic 선정 후보 `{nlog["chosen"]}`는 기존 N specialist(`ridge_isotonic`: StandardScaler → LogisticRegression C=0.01, max_iter 3000, 기본 tol, '
          f'Q_CAL isotonic)와 C·보정 방식이 같고 tol(1e-8)·중앙값 대치기(항등)만 다르다. MAIN TEST N 전체 Brier도 '
          f'{f(R["MAIN_TEST"]["N"]["metrics_named"]["all"]["logit_winner"]["brier"])} 대 {f(R["MAIN_TEST"]["N"]["metrics_named"]["all"]["old_A_specialist"]["brier"])}로 거의 같다(서술).')
    for coh in Q.COHORTS:
        bts = {c: R['MAIN_TEST'][coh]['bootstrap'][c] for c in ('all', 'B40', 'B45')}
        est = [p['a_minus_b']['brier']['estimate'] for bt in bts.values() if bt.get('computed', True) is not False for p in bt['pairs']
               if p['a_minus_b']['brier']['estimate'] is not None]
        prim = {c: bt['pairs'][0]['a_minus_b']['brier']['estimate'] for c, bt in bts.items() if bt.get('computed', True) is not False}
        mn = R['MAIN_TEST'][coh]['metrics_named']
        if est:
            w(f'- 크기(저장된 대비에서 계산, {coh} MAIN TEST): 계획 대비 5개 × all/B40/B45 셀의 ΔBrier 점추정 범위 {min(est):+.5f} ~ {max(est):+.5f}; '
              f'주 대비(LightGBM − PT) ' + ', '.join(f'{CELL_KO[c]} {v:+.5f} (PT Brier 대비 {100 * v / mn[c]["pt_winner"]["brier"]:+.2f}%)' for c, v in prim.items()) + '.')
    w('- p_pre와 q는 서로 다른 확률이다. p_pre = V가 추정한 교전 전 상태의 최종 Blue 승리 확률 P(W=1 | S_pre)이고, q = P(ΔV>0 | S_pre), 즉 라벨 Y의 확률이다. '
      'B40/B45는 p_pre 기준으로 V가 경기를 균형으로 본 셀일 뿐이며, 균형 V가 Y의 균형(양성률 .5)이나 q의 기준 Brier를 함의하지 않는다.')
    w('- Y 기준선은 측정값으로만 적는다(MAIN TEST, 경기가중 Y 양성률 / 같은 셀 TRAIN 상수 q의 Brier): '
      + '; '.join(f'{coh} {CELL_KO[c]} {f(R["MAIN_TEST"][coh]["metrics_named"][c]["old_constant"]["positive_rate_match_weighted"], 4)} / '
                  f'{f(R["MAIN_TEST"][coh]["metrics_named"][c]["old_constant"]["brier"])}' for coh in Q.COHORTS for c in ('all', 'B40', 'B45'))
      + '. B40/B45와 전체는 구성·유병률·시간대가 달라 둘의 차이를 "어려움의 인과효과"로 해석하지 않는다.')
    w('- 기존 참조(A specialist, pooled, p_pre spline/logistic, 상수)는 더 오래된 고정 적합 예산의 결과이며 이번 6설정 탐색과 동등한 튜닝 대상이 아니다. 6설정은 제한된 탐색 관례로, '
      '계산량 동등성이나 전역 최적을 뜻하지 않는다. PT와 전체 입력 family의 비교는 입력 정보량이 의도적으로 다르고, logistic 대 LightGBM만 352입력을 고정한다.')
    w('- 부트스트랩 구간은 고정 모델의 평가 표본 불확실성만 반영한다(학습·보정·선정 불확실성 제외, 다중 비교 보정 없음). 여러 셀·세트의 결과를 함께 읽고 한 셀의 구간으로 결론을 고르지 않는다.')
    w('')
    # ------------------------------------------------------------------ validation / failures
    w('## 9. 검증과 실패 기록')
    w('')
    fdiff = C.read_json(OUT / 'integrity' / 'snapshot_final_diff.json') if (OUT / 'integrity' / 'snapshot_final_diff.json').exists() else {}
    w(f'- validation.json: {val["n_checks"]}개 검사, 실패 {len(val["failed"])}개 ({val["checked_at"]}). 부모 무결성 diff all_equal = {diff.get("all_equal")}; '
      f'post-run 검사·보고서 1차 생성 이후 최종 재확인(snapshot_final_diff.json) all_equal = {fdiff.get("all_equal")}.')
    groups = {}
    for k in val['checks']:
        groups.setdefault(k.split('_')[0], []).append(k)
    w('- 주요 검사: 부모 파일 hash/inventory 불변, 명세·protocol 순서(snapshot → 테스트 → protocol → smoke → 최종 테스트 → 전체 적합 → 동결 → 봉인 접근), '
      'Q_SELECT 선정의 독립 재현(직접 공식), Q_CAL 보정기 재적합 일치, PT knot/scaler·대치기 TRAIN 재적합 일치, LightGBM stop 기록 일관성, '
      '동결 승자 재로딩 동일 예측(새 프로세스, MAIN TEST), 저장 예측으로부터 지표 독립 재계산(<1e-10), 부트스트랩 점추정 일치, 셀 구성 재계산, 기존 참조 join 완전성.')
    if fails:
        w('- 보존된 실패/재시도:')
        for x in fails:
            w(f'  - {x["time"]} `{x["stage"]}`: {esc(x["error"][:220])}')
        w('  - smoke 평가 첫 시도는 동결 파일 목록의 protocol.json 경로를 smoke 하위 폴더에서 찾다 실패했다(코드 경로 오류). 경로만 수정 후 smoke 평가를 재실행했고, '
          '선정·모델은 바뀌지 않았다. 계약 테스트는 수정된 최종 코드로 다시 통과(run 2)한 뒤 전체 적합을 시작했다.')
    else:
        w('- 보존된 실패 기록 없음.')
    hist = OUT / 'audit_followup' / 'history' / 'pre_correction_v1' / 'manifest.json'
    if hist.exists():
        w('- **감사 후속 수정(2026-09-15, 재적합·재선정·라벨/동결 산출물 변경 없음):** (1) 보고서 해석에서 결과 전에 고정해 둔 개선 폭 문구와 '
          'p_pre≈.5에서 Y 기준 Brier≈.25를 추론한 문구를 삭제하고, 저장된 대비·측정 양성률·상수 Brier에서 계산한 문장과 p_pre/q 구분으로 교체; '
          '(2) post-run 검사에 all/B40/B45 부트스트랩 수용 게이트(적격 셀: 1000회·seed 20260915·계획 대비 5개·유한 순서 구간·유한 복제 수·퇴화 수 일관성; '
          '희소/단일 클래스 셀: 명시적 미계산) 추가; (3) fit 스크립트에 적합 전 (match, s_ms) 고유성 검사와 부분 family 번들 덮어쓰기 차단(코드 전용 사후 guard, '
          '실제 학습 소스 아님) 추가. 수정 전 보고서·검증·스크립트는 `audit_followup/history/pre_correction_v1/`, 영수증은 '
          '`audit_followup/AUDIT_CORRECTION_RECEIPT.json`. 동결 후 코드 변경(report/snapshot/postrun/fit 스크립트)은 모델·예측에 영향이 없으며 '
          '학습·동결 당시 소스는 frozen_manifest.json source_sha256과 `outputs/claude_dispatch_incremental_q/training_source_snapshot`에 보존되어 있다.')
    w('')
    # ------------------------------------------------------------------ limits / artifacts / refs
    w('## 10. 한계와 비주장')
    w('')
    w('- 기존 TEST/외부 결과 노출 후 수행한 탐색적 비교다. 새 미래 패치 확인이 필요하다.')
    w('- h90만 재적합했다. h60/h120 새 적합, MLP/residual MLP, 전체 CoG lineup, SHAP, revised V, logit-label 분해, 원시 시간 보정, 탐지기 재학습은 **실행하지 않았다**.')
    w('- T/N은 사후 참가 규모로 나눈 retrospective 조건부 평가이며 실시간 라우팅 검증이 아니다.')
    w('- 전처리(knot·중앙값·표준화)는 부모 파이프라인 관례대로 비가중 TRAIN 행에 적합했고, 모델 손실만 경기 가중치를 사용했다(protocol interpretation_choices).')
    w('')
    w('## 11. 산출물')
    w('')
    for p, d in (('protocol.json', '사전 동결 계약·후보 등록부·정책'), ('contract_tests/', '계약 테스트 결과(run별)'), ('smoke_train_only/', 'TRAIN-only smoke 전체 경로'),
                 ('models/<cohort>/<family>/<config>.joblib', '설정별 번들(전처리+모델+Q_CAL 보정기)'), ('selection/<family>_<cohort>.json', '18후보 지표·선정·적합 기록'),
                 ('internal_stop/lgbm_<cohort>.json', 'LightGBM stop 곡선·반복 수·membership hash'), ('predictions/<family>_<cohort>_trainval.npz', 'TRAIN/Q_CAL/Q_SELECT raw·보정 예측, seed 예측'),
                 ('frozen_manifest.json', '동결 manifest(모든 hash, 승자)'), ('access_log.jsonl', '부모 데이터 접근 기록(봉인 세트는 동결 후)'),
                 ('eval/results.json', '셀별 지표·부트스트랩'), ('eval/predictions/<set>_h90_<cohort>.npz', '행별 평가 예측(54후보·seed·기존 참조·셀 표시)'),
                 ('integrity/', '부모 snapshot 전/후/diff'), ('validation.json', '사후 검증'), ('status.json, logs/, commands.txt, failures.jsonl', '진행·명령·실패 보존'),
                 ('DEFINITION_AND_EVIDENCE.md', '정의·근거·상태 표')):
        w(f'- `{p}`: {d}')
    w('')
    w('## 12. 참고')
    w('')
    w('- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30. '
      'https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html — gradient boosting 방법 출처(이 표적에서의 우수성 근거 아님).')
    w('- scikit-learn 공식 문서(로컬 설치 1.6.1 사용): LogisticRegression https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html , '
      'SplineTransformer https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html — 구현 API 출처. '
      '이번 실행은 네트워크 조회를 하지 않았고 API 동작은 설치 버전 코드·테스트로 확인했다.')
    w('- PT 텐서 구성, knot 수, C 격자, 트리 상한, seed, 보고 기준(30경기, B40/B45)은 우리 운영 선택이며 외부에서 확립된 최적값이 아니다.')
    w('')
    report = '\n'.join(L) + '\n'
    (OUT / 'REPORT.md').write_bytes(report.encode('utf-8'))
    # ------------------------------------------------------------------ definitions and evidence
    D = []
    d = D.append
    d('# 정의와 근거 (incremental q training 2026-09-15)')
    d('')
    d('상태 어휘: **실행·검증** = 이번 실행에서 수행하고 validation.json 검사로 확인; **상속** = 부모 산출물 그대로 사용(재계산 없음, hash 확인); **연기** = 이번 범위에서 실행하지 않음.')
    d('')
    d('## 진행 전/후')
    d('')
    d('- 전: 기존 T/N specialist 후보군(상수, p_pre 단일 logistic/spline, ridge C=.01, 경제 입력 LightGBM 250 trees)과 pooled q가 h90 참조였다. '
      'p_pre와 시간의 상호작용 기준선, C 탐색 logistic, 352입력 LightGBM 조기종료 family는 없었다.')
    d(f'- 후: T/N 각각 PT·전체 logistic·전체 LightGBM 3 family × 18 후보를 적합·선정·동결({fz["frozen_at"]})하고 MAIN TEST와 외부 4세트에서 평가했다. '
      f'validation.json 실패 {len(val["failed"])}개.')
    d('')
    d('| 용어 | 정의 | 근거 파일 | 상태 |')
    d('|---|---|---|---|')
    rows = [
        ('Y (h90)', '1[V(endpoint_h90) − V(s−1ms) > 0], 같은 V adapter 양 끝; 정확히 0이면 0', 'full_corpus_training_20260915/labels/*_labels.npz', '상속'),
        ('p_pre', 'V(S_pre) = P(최종 Blue 승리 W=1 | S_pre) 추정, TRAIN은 자기 경기 제외 OOF fold adapter, 그 외 final V', 'labels npz adapter_id/sha256; selection *.oof_provenance', '상속·검증'),
        ('q', 'P(ΔV>0 | S_pre) = Y의 확률 추정. p_pre 균형(B40/B45)은 Y 균형을 함의하지 않음; Y 기준선은 측정 양성률·상수 Brier로만 기술', 'eval/results.json', '실행·검증'),
        ('T / N', 'T: 알려진 min(cluster_blue, cluster_red) ≥ 4; N: 알려진 min < 4', 'cohort_role_training_20260915/cohorts', '상속'),
        ('352 입력', 'q_pre_only_schema.json predictor_sets["ridge"] 순서 그대로', 'protocol.json inputs', '실행·검증'),
        ('PT 설계', 'SplineTransformer(n_knots=5, degree=3, quantile, include_bias=False, extrapolation=constant) p_pre_V·time_minutes 각 6열 + 36 곱(p major) = 48열 → 표준화 → 가중 logistic', 'selection/pt_*.json design', '실행·검증'),
        ('경기 가중치', '각 적합/평가 부분집합 안에서 경기 총합 동일, 행 평균 1', 'selection weights, eval 셀', '실행·검증'),
        ('stop10', "int(sha256('iq20260915_stop:'+match)[:8],16) mod 10 == 0 인 TRAIN 경기", 'internal_stop/lgbm_*.json', '실행·검증'),
        ('조기종료', 'stop10 경기가중 Brier, patience 50, 동점은 이른 반복; 선택 반복 수로 전체 TRAIN 재적합', 'internal_stop', '실행·검증'),
        ('보정', 'raw / sigmoid(logit(clip 1e-8) LogisticRegression C=1e6) / isotonic(clip), cohort Q_CAL', 'selection calibrators', '실행·검증'),
        ('선정 규칙', 'cohort Q_SELECT 경기가중 Brier → log loss → 후보 이름', 'selection/*.json, frozen_manifest.json', '실행·검증'),
        ('B40 / B45', '.40 ≤ p_pre ≤ .60 / .45 ≤ p_pre ≤ .55 (경계 포함)', 'eval/predictions cell__*', '실행·검증'),
        ('시간 구간', 'time_minutes [0,10), [10,20), [20,30), [30,∞)', 'eval/predictions cell__time_*', '실행·검증'),
        ('부트스트랩', '경기 단위 1000회, seed 20260915, 셀 내 모든 모델 동일 draw, 퍼센타일 95%', 'eval/results.json bootstrap', '실행·검증'),
        ('기존 참조', 'A_<set>_h90_<cohort>.npz의 spec_<동결 선택>, pooled, spec_p_pre_spline/logistic/constant', 'eval checks.legacy', '상속·검증'),
        ('h60/h120 새 적합, MLP, CoG, SHAP, V 재적합', '—', '—', '연기'),
    ]
    for r_ in rows:
        d('| ' + ' | '.join(esc(x) for x in r_) + ' |')
    d('')
    d('## 주요 수치 근거')
    d('')
    for coh in Q.COHORTS:
        r = R['MAIN_TEST'][coh]
        mn = r['metrics_named']['all']
        d(f'- MAIN TEST {coh} 전체 Brier: ' + ', '.join(f'{LABEL[m]} {f(mn[m]["brier"])}' for m in MODELS) + f' (eval/results.json, 예측 `{r["predictions_file"]}`).')
    d('')
    d('## 무결성')
    d('')
    d(f'- 부모 snapshot diff all_equal = {diff.get("all_equal")}; 봉인 세트 접근은 동결 이후만(validation.json `sealed_accesses_only_after_freeze` = '
      f'{val["checks"].get("sealed_accesses_only_after_freeze")}).')
    d(f'- 명세 sha256 `{proto["spec_sha256"]}`, protocol sha256 `{fz["protocol_sha256"]}`, frozen_manifest sha256 `{res["frozen_manifest_sha256"]}`.')
    d('')
    (OUT / 'DEFINITION_AND_EVIDENCE.md').write_bytes(('\n'.join(D) + '\n').encode('utf-8'))
    C.write_json(OUT / 'report_manifest.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), report_sha256=C.sha256_file(OUT / 'REPORT.md'),
                                                    definitions_sha256=C.sha256_file(OUT / 'DEFINITION_AND_EVIDENCE.md'),
                                                    validation_sha256=C.sha256_file(OUT / 'validation.json'), results_sha256=C.sha256_file(OUT / 'eval' / 'results.json')))
    Q.Status('report').update('complete', 'report', report_sha256=C.sha256_file(OUT / 'REPORT.md'), validation_failed=len(val['failed']),
                              next_step='Codex independent audit')
    print('REPORT.md', C.sha256_file(OUT / 'REPORT.md'))


if __name__ == '__main__':
    main()
