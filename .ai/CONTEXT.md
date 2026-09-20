# .ai/CONTEXT.md — 현재 작업 맥락 (journal_manuscript_v1 집필 단계)

작성: 2026-09-20, Claude 계획 세션. 기준 커밋 **21391b2** (`feature/fight-boundary-pipeline`).

## 현재 목표
저널 원고 `docs/journal_manuscript_v1/`를 **검증 연구**로 마감한다. 실험은 닫혔다 (`docs/JOURNAL_FINISH_LOCK_20260920.md`, `docs/RESPONSE_EVIDENCE_MATRIX_20260920.md`: RR0~RRX CLOSED). 지금은 Methods/Results/Discussion 스파인의 **교정·전사** 단계이며, Introduction/Abstract/Conclusion/Related Work는 그 뒤에 쓴다.

## 브랜치와 인계
- 작업서·보고서 양식은 Claude가 `claude/happy-heisenberg-h93ypi`에 push한다. 저자가 로컬 `feature/fight-boundary-pipeline`에 merge한 뒤 Cursor에서 실행하고, 보고서를 push한다.
- `codex/research-snapshot-20260917`은 별도 계열(공통 조상 `576cd17`, 병합되지 않음). 그 문서는 참조만 한다.
- `docs/tog_manuscript/`는 CoG 확장(승자 예측) 계보. 읽기 전용.

## 문서 지도 (읽기 우선순위)
| 목적 | 파일 |
|---|---|
| 원고 스파인 | `docs/journal_manuscript_v1/README.md`, `00_MIGRATION_OLD_VS_FREEZE.md`, `01_OUTLINE_AND_CLAIM_EVIDENCE.md`, `02_METHODS_CORE.md`, `03_RESULTS.md`, `04_DISCUSSION.md` |
| 마감 규칙·Forbidden | `docs/JOURNAL_FINISH_LOCK_20260920.md`, `docs/RESPONSE_EVIDENCE_MATRIX_20260920.md`, `docs/Q_RESULT_SCOPE_LOCK_20260920.md` |
| RQ 문구 (LOCKED, 변경 금지) | `docs/JOURNAL_RESEARCH_PLAN_20260919.md` §6, `docs/J_RQ1_SCOPE_LOCK_20260920.md`, `docs/COG_SUCCESSION_LOCK_20260919.md` §5 |
| 정의 (SVI, B40, h90, PT) | `docs/COMMON_RESEARCH_SPINE_20260919.md` §3, `docs/V_Q_TARGET_LOCK_20260920.md`, `docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md`, `docs/V_DYNAMIC_FRAME_CONTRACT_20260919.md` |
| V 동결 | `docs/V_EVALUATOR_FREEZE_CLOSE_20260920.md`, `docs/A_MLP_expanded_evaluator_meta_20260919.json`, `docs/BAND_LEDGER_WAVE4_CORRECTED_20260919.md`, `docs/V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md` |
| 결과 수치 | `docs/REVIEW_RESPONSE_RR12_RESULTS_20260920.{md,json}`, `REVIEW_RESPONSE_RR3_{QUIET,CLOSEOUT}_20260920.*`, `REVIEW_RESPONSE_RR4_MARGIN_20260920.md`, `REVIEW_RESPONSE_RR5_RR6B_20260920.md`, `REVIEW_RESPONSE_RR6A_CORP_20260920.md`, `REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md`, `TEST_BAND_LEDGER_MLP_FIT85_20260920.md`, `Q_NEWV_FIT85_PRIMARY_20260920.md`, `Q_NEWV_FIT85_TRANSFER_16X_20260920.md` |
| 실행 정직성 | `docs/REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md`, `docs/REVIEW_RESPONSE_RR0_MANIFEST_20260920.json`, `docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md` |
| 코호트 | `docs/PAPER_COHORT_CONTRACT_20260919.md`, `docs/EXPERIMENT_INVENTORY_COHORT_20260919.md` |
| 교전 정의 (v3.3 현재 정의 + 과거 파일럿 민감도 혼재) | `docs/tog_manuscript/sec_definition.tex` (읽기 전용) |
| 정의의 최종 근거 (코드) | `scripts/rr20260920_review_response_rr12.py`, `..._rr4_margin.py`, `..._rr5_rr6b.py`, `..._rr6a_corp.py`, `..._rrx_external.py`, `scripts/forecast_diagnostics.py`, `scripts/rr20260920_q_newv_primary_fit.py`, `scripts/rr20260920_q_build_newv_labels.py`, `scripts/rr20260920_q_train_oof_mlp_folds.py`, `scripts/rr20260919_v_redesign_fit_wave4_corrected.py` |

## 고정된 계약 (작업서가 허용하지 않는 한 변경 금지)
- **SVI** = strategic value improvement. \(Y_{\mathrm{SVI}}=\mathbf 1[\Delta\widehat V>0]\); \(\Delta\widehat V=0\)은 \(Y=0\); pre/post 점수 결측은 라벨 전 제외.
- **B40** = \(0.40\le p_{\mathrm{pre}}\le 0.60\) (폭 20%p), 새 fit85 \(p_{\mathrm{pre}}\)로 재계산. 15.16 TEST B40 n=5,423 / 4,945경기.
- **V̂** = fit85 `A_MLP_expanded` 번들 (sha16 `ac459cc4397630a9`), Expanded361(수치 351 + 챔피언 ID 10 → 8차원 임베딩 → 입력 431), hidden (256,128), PosSlopeSigmoid g on V_CAL, 선정 규칙 L_time on V_SELECT. 재학습·재선정 금지.
- **q** = `logit_state` (수치 351 + \(p_{\mathrm{pre}}\) = 352 입력, 챔피언 ID 제외), Q_SELECT 경기가중 Brier로 선정, 보정기 identity. 주 기준선 **PT_flex** (n_knots_p=4, n_knots_t=4, C=0.01, degree 3, uniform knots, identity), 연속성 PT_linear.
- **라벨 경로**: 15.14 TRAIN = 경기 단위 5-fold OOF \(\widehat V^{(-k)}\) (pre·post·B40 모두 같은 fold 평가기); Q_CAL/Q_SELECT/TEST/EXT = fit85 동결 번들.
- **평가**: 셀별 경기 가중 \(w_i=1/n_m\) 재계산(Brier·logloss·AUC·CORP 모두); 부트스트랩 = 경기 단위 paired percentile 2,000회 seed 7 (RR4 B40 슬라이스 seed 8; `Q_NEWV_FIT85_PRIMARY` 표는 800회). `bootstrap_fraction_positive`는 p-value가 아님.
- **CI가 있는 분석**: RR12(ΔBrier), RR3(quiet), RR4(λ 슬라이스). **없는 분석**: V→W 표, RR5a/5b, RR6a CORP, RR6b, RRX 외부.
- **RR4**는 임계값 미만 행 **제외**(subsample), down-weight 아님. **RR5b**는 비가중 건수.

## 검증된 수치 원장 (요약)
| 항목 | 값 | 출처 (파일 → 위치) |
|---|---|---|
| V→W TEST 시간상태 | n 347,234 (60,190경기), Brier 0.1552, AUC 0.8542; MCB 0.0002, DSC 0.0948 | `TEST_BAND_LEDGER_MLP_FIT85` L5; `RR6A_CORP` row `V_to_W_timeline|overall` |
| V→W 초기 밴드 [2,10) | n 120,186, Brier 0.2283, AUC 0.6642 | `RR6A_CORP` row `t_2_10` (TEST_BAND_LEDGER는 `t_0_10`으로 표기, 동일 셀) |
| V_pre / V_post → W (n 32,981) | Brier 0.1442 / 0.1181, AUC 0.8750 / 0.9149 | `RR6A_CORP` rows `V_to_W_eng_pre/post` |
| q TEST all-T | n 32,981 / 24,020경기, Brier 0.2355, logloss 0.6637, AUC 0.6403 | `RR12 json` `TEST_all.q_base` |
| ΔBrier q−PT_flex all-T | −0.00373 [−0.00461, −0.00279], p_gt0 0 | `RR12 json` `bootstrap.all_q_minus_PT_flex` |
| ΔBrier q−PT_linear all-T | −0.00422 [−0.00515, −0.00323] | `bootstrap.all_q_minus_PT_linear` (800회 표는 [−0.00516, −0.00325]) |
| B40 ΔBrier q−PT_flex | −0.00214 [−0.00422, −0.00003]; n 5,423 / 4,945 | `bootstrap.B40_q_minus_PT_flex`; `census.test_B40` (RR4 재추출 상한 −0.0001, LIT bridge −0.00007: 점추정 동일, 원고는 RR12 값) |
| H = D_B40 − D_outside | 0.00181 [−0.00060, 0.00422] | `bootstrap.H_B40_minus_outside` |
| CORP q / PT_flex all-T | MCB 0.00086 / 0.00053, DSC 0.0154 / 0.0113 | `RR6A_CORP` rows `q_to_SVI` |
| RR3 quiet (TEST) | E[|ΔV|_f − |ΔV|_q] 0.0633 [0.0612, 0.0654]; n 9,140 / 8,438경기; coverage 0.277 | `RR3_CLOSEOUT`, `RR3_QUIET json` |
| RR5a kill-axis agree (match-wtd, decided) | 0.904 all-T / 0.928 B40 | `RR5_RR6B` RR5a 표 |
| RR5b next objective 180 s | SVI+ Blue 6,004 / decided 9,934 = 0.604; SVI− 4,072 / 10,277 = 0.396; game_ended_in_window 8,112; censored 0 | `RR5_RR6B` L35–L38 |
| RR6b horizons | flip 0.015 / 0.007 / 0.019 (h60–90 / h90–120 / h60–120); same-endpoint 0.594 / 0.810 / 0.594 | `RR5_RR6B` horizon 표 |
| RR4 λ ∈ {0, .25, .5, 1} | all-T ΔBrier ≈ −0.004, CI가 0 제외; n 32,981 → 26,297 (λ=1) | `RR4_MARGIN` λ 표 |
| EXT KR / NA1 16.13 | n 5,202 / 5,312; V_pre Brier 0.1511 / 0.1514; ΔBrier(q−PT_flex) +0.0026 / +0.0040; ΔMCB 0.0045 / 0.0063; ΔDSC 0.0019 / 0.0023; q MCB 0.0065 / 0.0100; q DSC 0.0082 / 0.0064 (CI 없음) | `RRX_EXTERNAL` 요약표 |
| EXT vs PT_linear (다른 기준선) | +0.00274 / +0.00342 | `Q_NEWV_FIT85_TRANSFER_16X` |
| V 시간상태 census | train 424,160 (fit 360,479 + stop 63,681), V_CAL 106,580, V_SELECT 108,558, TEST 347,234 | `A_MLP_expanded_evaluator_meta_20260919.json` `meta.census` |
| q 교전 역할 | 15.14 TRAIN 39,605 / 15.15 Q_CAL 10,390 · Q_SELECT 10,195 · 미배정 20,730 (VAL 41,315) / 15.16 TEST 32,981 / pooled T 113,901 (83,108경기) | `PAPER_COHORT_CONTRACT` §2; `EXPERIMENT_INVENTORY_COHORT` §E |
| PT_flex 선택 설정 | n_knots_p 4, n_knots_t 4, C 0.01, degree 3; 보정기 identity | `RR12 json` `PT_flex`, `calibrator_choice` |
| V 보정 | PosSlopeSigmoid coef 0.9076, intercept 0.0640 | `A_MLP_expanded_evaluator_meta` `calibration` |

## 알려진 문서 간 불일치와 원고에서의 선택
- B40 CI 상한 세 가지 인쇄 → **RR12 값** 채택, 출처 명시.
- 외부 ΔBrier 두 계열 → 원고 표는 **q−PT_flex (RRX)**. PT_linear 계열은 별도 언급만.
- PT_linear 연속성 CI 두 가지(800회 vs 2,000회) → **RR12(2,000회)** 채택.
- 첫 시간대 라벨 `t_0_10` vs `t_2_10` → 원고는 **[2,10)** (t<2분 제외).
- `PAPER_COHORT_CONTRACT` §5–§6 수치는 **구 V 계보**. 새 원고 헤드라인으로 옮기지 않는다 (T004가 배너 추가).
- 부트스트랩 설정은 문서에 인쇄돼 있지 않고 코드에만 있다 → T002가 Methods에 명시.

## 이 브랜치에 없는 것 (찾지 말 것)
- 교전 코호트 플래그(`cohort == 1`, `valid_h90`)를 생성한 2026-09-15 파이프라인 코드와 `outputs/` 산출물. T의 자격 규칙 문장이 `docs/`에서 확인되지 않으면 `\pending`으로 남긴다.
- CoG 118 리뷰 원문 (요약만 `docs/CLAUDE_TOG_PAPER_PLAN.md` §7–8).
- 협업자 교정 patch (리뷰어 샌드박스에만 존재). 작업서는 그 지적을 저장소 근거로 재구성한 것.
