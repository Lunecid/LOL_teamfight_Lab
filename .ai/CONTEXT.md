# .ai/CONTEXT.md — 현재 작업 맥락 (journal_manuscript_v1 집필 단계)

작성: 2026-09-20, Claude 계획 세션. 기준 커밋 **21391b2** (`feature/fight-boundary-pipeline`).

## 현재 목표
저널 freeze는 유지한다. 보완 실험은 [`docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md`](../docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md) 경로만.  
**T019 (2026-09-21):** E2 §5.4 프레임 층화 DONE; §5.2 S_hold는 T022에서 해소.  
**T020 (2026-09-21):** E4 §7.1/7.2 horizon strata + peer LR label-transfer DONE; §7.3 e_fixed / §7.4 OAT 미실행.  
**T021 (2026-09-21):** E5 §8.1 동일사례 대응 + nextobj 연관 DONE; market_event/R0–R1 INCOMPLETE.  
**T022 (2026-09-21):** CACHE_MAIN(`D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13`) 복구 + E2 S_hold DONE → `docs/SUPPLEMENTARY_E2_S_HOLD_20260921.md`.  
다음 보완 후보: e_fixed / 정의 OAT / market_event(캐시 사용); 또는 원고 주장 정리.  
석사 Markdown 골격: `docs/master_thesis_v1/`; LaTeX: `thesis/`.

## 브랜치와 인계
- 주 작업 브랜치: `feature/fight-boundary-pipeline`.
- 작업서·보고서는 `.ai/tasks/` · `.ai/reports/`.
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
| q 교전 역할 | 15.14 TRAIN 39,605 / 15.15 Q_CAL 10,390 · Q_SELECT 10,195 · V_CAL 10,191 · V_SELECT 10,539 (VAL 41,315) / 15.16 TEST 32,981 / pooled T 113,901 (pooled 경기 수는 문서에 없음) | `PAPER_COHORT_CONTRACT` §2; `EXPERIMENT_INVENTORY_COHORT` §E |
| PT_flex 선택 설정 | n_knots_p 4, n_knots_t 4, C 0.01, degree 3; 보정기 identity | `RR12 json` `PT_flex`, `calibrator_choice` |
| V 보정 | PosSlopeSigmoid coef 0.9076, intercept 0.0640 | `A_MLP_expanded_evaluator_meta` `calibration` |

## 알려진 문서 간 불일치와 원고에서의 선택
- B40 CI 상한 세 가지 인쇄 → **RR12 값** 채택, 출처 명시.
- 외부 ΔBrier 두 계열 → 원고 표는 **q−PT_flex (RRX)**. PT_linear 계열은 별도 언급만.
- PT_linear 연속성 CI 두 가지(800회 vs 2,000회) → **RR12(2,000회)** 채택.
- 첫 시간대 라벨 `t_0_10` vs `t_2_10` → 원고는 **[2,10)** (t<2분 제외).
- `PAPER_COHORT_CONTRACT` §5–§6 수치는 **구 V 계보**. 새 원고 헤드라인으로 옮기지 않는다 (T004가 배너 추가).
- 부트스트랩 설정은 문서에 인쇄돼 있지 않고 코드에만 있다 → T002가 Methods에 명시.

## 계보 확인 결과 (2026-09-20 밤, `codex/research-snapshot-20260917` = `e0ec3d0`에서 읽음)
- **T 코호트 규칙:** `cohort == 1` ⟺ `min(cluster_blue, cluster_red) ≥ 4` (`e0ec3d0:scripts/cr20260915_common.py` L37–60 `scale_classes`); 다른 필터 없음; v3.3 teamfight 계급(`fine == 2`)과 모든 세트에서 동일. `{SET}_cohort.npz`에는 `cohort`(1 T / 0 N / −1 unknown), `fine`(0 pick / 1 skirmish / 2 teamfight), `n_min`, `scale_known` 열이 있음(`cr20260915_cohorts.py` L441–444).
- **sub_role:** 경기 단위 sha256 해시(`fc20260915_common.py` L65–77): TRAIN fold = %5, VALIDATION = %4 → V_CAL / V_SELECT / Q_CAL / Q_SELECT. 15.15의 T 20,730행 = V_CAL 10,191 + V_SELECT 10,539(**미배정이 아니라 V 보정·선정 경기**; q에 사용 금지).
- **역할별 T 경기 수(`cohort_manifest.json` `by_sub_role.*.h90.T_matches`):** TRAIN fold 5,732/5,858/5,725/5,845/5,829; Q_CAL 7,550; Q_SELECT 7,469; V_CAL 7,448; V_SELECT 7,632; TEST 24,020; EXT KR16.13 3,859 / NA1 3,955 / KR16.15 377 / pilot 80. pooled 합 83,108은 기록된 필드가 아님.
- **h90 endpoint:** s = 첫 킬 − 15 s, q_pre = s − 1 ms, L = 마지막 킬; endpoint_h = min(L + 1000h, 다음 킬 − 1, 다음 교전 시작 − 1, 경기 종료 − 1) [ms]; 오브젝트·ace는 종료 사건 아님(`e0ec3d0:scripts/engagement_labels_v3_rules.py` `endpoint_rule`/`endpoint_validity`; `docs/LABEL_ENDPOINT_RULE_AND_EXAMPLES_20260914.md`). 유효 566,104/566,452, 무효 348행 전부 N(overlap: 다음 교전 시작 ≤ L). main TEST h90 실현: L 이후 평균 45.493 s / 중앙 39.440 s, 상한 도달 14.286%, 다음 킬 종료 ≈78.9%, L 이후 새 프레임 없음 35.145%.
- **9/15 코호트별 학습(구 로지스틱 V):** T specialist vs pooled ΔBrier −0.00166 [−0.00235, −0.00104]; N −0.00042; skirmish(2–3) −0.00023 [−0.00043, −0.00004]; pick −0.00083. PT 기준선 없음. "N 특화는 robust하지 않음"; T/N 기전 설명 철회(X-31).

## 규모 분리 실험선 (T007–T009; 계약 `docs/SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md`)
- 저자 결정: 석사/v2 계약으로 먼저 실행 → 결과 후 저널 포함 결정; 비교 = T vs **S = `cohort==0 & fine==1`(2 ≤ n_min ≤ 3)**; pick 제외; Cursor 실행.
- 코호트 하드코딩 지점: `rr20260920_q_build_newv_labels.py` `cohort_keys` L59–71, `rr20260920_q_train_oof_mlp_folds.py` `cohort_t_keys` L76–85. q 적합·RR12 평가는 라벨 파일에서 코호트를 물려받음(경로만 매개변수화).
- fold 평가기: `outputs/q_newv_fit85_20260920/oof_evaluators/V_oof_fold{k}_mlp_expanded.joblib` + `bundle_oof_fold{k}.joblib` — 문서로 존재 미보증, 실행 전 확인.
- S h90 census: TRAIN fold0–4 24,030/25,005/24,520/24,690/24,804; Q_CAL 31,302; Q_SELECT 31,059; TEST 101,205; EXT KR16.13 15,641 / NA1 16,100 / KR16.15 1,307 / pilot 285.
- 진행 상태 (2026-09-20 밤): **T007 통과**(`7cee54e`, 검토 `.ai/reports/T007_review.md`) · **T008 통과**(`325ab76`, 검토 `.ai/reports/T008_review.md`; S 라벨 4역할 census 완전 일치, S∩T 키 0, fold 분리 V-held 기준 확인, T 동결 무변경) · **T009 조건부 통과**(`aa2249f`, 검토 `.ai/reports/T009_review.md`) · **T010 통과**(`69130a4`, 검토 `.ai/reports/T010_review.md`; identity 주 대비 −0.00335 [−0.00379, −0.00288], 민감도 −0.00346, 부호 동일; E1 인코딩 손상 5개 문서는 Claude가 복구) · **저자 결정 2026-09-21: 계약 §8 승격 C(저널 본문 2-코호트)** → **T011 완료**(`9caa943`, 잠금 개정·추가만, E1 복구) · **T012 완료**(원고 2-코호트 통합; `\pending` 2건 저자 문안: pick 제외 사유, EXT T/S 병기 문장). **T013 완료**(집필 준비 패키지: `docs/journal_manuscript_v1/00_WRITING_DOSSIER.md`가 인덱스; 실험 원장 `EXPERIMENT_LEDGER.md`; 초록/서론/관련연구/결론 초안 05–08; 리뷰어 방어·모의 심사·자기 검토; 그림 3장 SVG; ToG venue 체크리스트; 이연 계획; 변경 보고+퀴즈). 다음: 저자가 DOSSIER §2의 결정 5건(RQ 구조, `\pending` 2건, 가용성·익명 패키지, 표/그림 배분, AI 고지) → 초안 확정 → T014(LaTeX 포팅·outputs 그림·페이지 검사, Cursor).
- 확정 S 수치(identity, `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json`): S TEST q_S 0.24581 / PT_flex_S 0.24916 / AUC 0.577·0.536; ΔBrier(q_S − PT_flex_S) −0.00335 [−0.00379, −0.00288] (primary); q_S − q_T→S −0.00194 [−0.00236, −0.00153]; q_S − q_TS −0.00024 [−0.00039, −0.00009]; q_TS − q_T (T) +0.00071 [+0.00018, +0.00126]; S∩B40 −0.00210 [−0.00266, −0.00149], H_S +0.00178 [+0.00096, +0.00259]; q_T→S − PT_flex_S −0.00141 [−0.00186, −0.00094]; CORP q_S MCB 0.0008 / DSC 0.0048, PT_flex_S 0.0007 / 0.0014; EXT S KR/NA1 16.13 −0.0018 / −0.0020 (CI 없음). 원고 규칙: 코호트별 블록·코호트별 주 대비, 합동 행·코호트 간 통계·규모 기울기 서술 금지(계약 §6).
- T009 잠정 수치(RR12 2단계 선택 변형, sigmoid; 계약 문언 identity 값은 T010): S TEST q_S − PT_flex_S = −0.00346 [−0.00382, −0.00307] (primary) · q_S − q_T→S −0.00138 [−0.00168, −0.00108] (raw 쌍 근사 ≈ −0.0019) · q_S − q_TS −0.00012 [−0.00024, +0.00001] · T TEST q_TS − q_T +0.00071 [+0.00018, +0.00126] · S∩B40 −0.00210 [−0.00256, −0.00160], H +0.00196 [+0.00128, +0.00262] · EXT S (raw, CI 없음) KR/NA1 16.13 −0.0018/−0.0020. 출처 `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json`, rr12 `_S_qS/_S_qT/_S_qTS/_qTS_on_T.json`.
- T009 이탈: S 3회 rr12 실행에서 2단계 보정기 선택이 전 모델 sigmoid → 계약 §4 "identity calibrator"·arm (2) "as-is"와 불일치(R3). 표 1 lgbm 행은 Q_SELECT 값(R4). rr12 L825 B40 문장 하드코딩(R5). RR0 동결 digest 대조 미실시(G2).
- T008 확정 사실: reuse 모드의 fold held 집합은 V-bucket(`load_v_rows … sub_role==fold{k}`) 기준이며 refit 경로와 동일 구성; 교전 fold 경기 집합은 그 진부분집합(fold당 V-only 981–1,054경기). fold별 S 경기 11,962/12,352/12,138/12,138/12,230; S TEST 경기 49,730 / Q_CAL 15,254 / Q_SELECT 15,282 / TRAIN 60,820. S TEST B40 31,675행(≈31.3 %; T 16.4 %) — 코호트 간 나란히 비교 금지. manifest: `docs/SCALE_SPLIT_RR0_MANIFEST_20260920.json`.
- 잠금 입장: 실험 자체를 금지하는 잠금 없음(I1 허용, M-RQ1 담당). 저널 승격 시 해제할 문장: COMMON_RESEARCH_SPINE §3 "주 교전 T; N은 보조·부록", PAPER_COHORT_CONTRACT "all models … these rows only", JOURNAL_FINISH_LOCK "Block manuscript on new performance — No".

## 이 브랜치에 없는 것 (찾지 말 것)
- 2026-09-15 파이프라인의 실행 코드 전체와 `outputs/` 산출물. 단, 규칙을 정하는 파일 6개는 `docs/lineage_20260915/`에 바이트 동일 사본으로 있고(README에 발췌·출처 줄 번호), Methods의 `\pending` 3건은 T005로 해소됨.
- CoG 118 리뷰 원문 (요약만 `docs/CLAUDE_TOG_PAPER_PLAN.md` §7–8).
- 협업자 교정 patch (리뷰어 샌드박스에만 존재). 작업서는 그 지적을 저장소 근거로 재구성한 것.
