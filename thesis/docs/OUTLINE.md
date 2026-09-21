# 논문 구조 (OUTLINE)

`tools/gen_outline.py`가 `latex/main.tex`에서 도달 가능한 장·절 파일로부터 생성한다 (손으로 고치지 않음). 조립 순서: 표지 → 면지 → 속표지·인준지 → 차례 → 국문 초록 → 본문 1–8장 → 참고 문헌 → 부록 A–E → 영문 초록.

| 장 | 절 | 라벨 | 파일 | 포함 객체 |
|---|---|---|---|---|
| **서론** (`ch:intro`) | | | | |
| | 리그 오브 레전드와 한타 | `sec:intro-game` | `latex/chapters/01_introduction/01_game_and_teamfight.tex` | `tables/tab_terms` |
| | 문제 | `sec:intro-problem` | `latex/chapters/01_introduction/02_problem.tex` |  |
| | 기존 접근과 남은 어려움 | `sec:intro-prior` | `latex/chapters/01_introduction/03_prior_and_challenge.tex` |  |
| | 본 논문의 접근과 주요 결과 | `sec:intro-approach` | `latex/chapters/01_introduction/04_approach_results.tex` | `figures/fig_pipeline` |
| | 연구 질문 | `sec:intro-rq` | `latex/chapters/01_introduction/05_research_questions.tex` |  |
| | 기여와 논문의 구성 | `sec:intro-contrib` | `latex/chapters/01_introduction/06_contributions_outline.tex` |  |
| **관련 연구** (`ch:related`) | | | | |
| | MOBA 경기의 승률 모형 | `sec:rel-winprob` | `latex/chapters/02_related_work/01_winprob_models.tex` |  |
| | 교전의 검출과 교전 단위 예측 | `sec:rel-encounter` | `latex/chapters/02_related_work/02_encounter_detection.tex` |  |
| | 사건의 가치를 승률 변화로 매기기 | `sec:rel-eventvalue` | `latex/chapters/02_related_work/03_event_value.tex` |  |
| | 적정 점수, 보정, 짝 비교 | `sec:rel-scores` | `latex/chapters/02_related_work/04_scores_calibration.tex` |  |
| | 남은 문제와 본 논문의 위치 | `sec:rel-position` | `latex/chapters/02_related_work/05_gap_and_position.tex` | `tables/tab_prior_vs_thesis` |
| **데이터와 사전 정보 구성** (`ch:data`) | | | | |
| | 비동기 공개 텔레메트리의 구조 | `sec:data-telemetry` | `latex/chapters/03_data/01_telemetry.tex` |  |
| | 원천 코퍼스 | `sec:data-corpus` | `latex/chapters/03_data/02_corpus.tex` | `tables/tab_corpus_census` |
| | 데이터 역할 | `sec:data-roles` | `latex/chapters/03_data/03_roles.tex` | `tables/tab_roles_evaluator`, `tables/tab_roles_engagement` |
| | 교전 전 상태와 입력의 구성 | `sec:data-inputs` | `latex/chapters/03_data/04_prestate_inputs.tex` |  |
| | 시간 계약과 누출 방지 | `sec:data-contract` | `latex/chapters/03_data/05_time_contract.tex` |  |
| **교전 사례의 정의와 구성** (`ch:engagement`) | | | | |
| | (도입) | `` | `latex/chapters/04_engagement/00_lead.tex` |  |
| | 킬 에피소드와 시간 경계 G | `sec:eng-temporal` | `latex/chapters/04_engagement/01_temporal_boundary.tex` |  |
| | 공간 경계 D | `sec:eng-spatial` | `latex/chapters/04_engagement/02_spatial_boundary.tex` |  |
| | 존재 게이트 R, B, M | `sec:eng-gate` | `latex/chapters/04_engagement/03_presence_gate.tex` |  |
| | 조립 규칙과 모든 상수의 출처 | `sec:eng-constants` | `latex/chapters/04_engagement/04_assembly_constants.tex` | `tables/tab_constants` |
| | 규모 계급과 코호트 T, S | `sec:eng-scale` | `latex/chapters/04_engagement/05_scale_cohorts.tex` |  |
| | 결과 시점 규칙 | `sec:eng-endpoint` | `latex/chapters/04_engagement/06_endpoint_rule.tex` |  |
| | 킬에 조건화된 정의의 분석 대상 | `sec:eng-killless` | `latex/chapters/04_engagement/07_killless.tex` |  |
| | 정의 민감도 | `sec:eng-sensitivity` | `latex/chapters/04_engagement/08_sensitivity_lineage.tex` | `tables/gen/tab_oat` |
| | 타당성 요약 | `sec:eng-validity` | `latex/chapters/04_engagement/09_validity_summary.tex` |  |
| **경기 승패 기반 결과 가치** (`ch:value`) | | | | |
| | 설계 원칙: 측정과 예측의 분리 | `sec:value-principle` | `latex/chapters/05_value/01_design_principle.tex` |  |
| | 동결 승률 평가기 | `sec:value-evaluator` | `latex/chapters/05_value/02_evaluator.tex` |  |
| | 평가기의 경기 승패 예측 품질 (V W) | `sec:value-vw` | `latex/chapters/05_value/03_vw_quality.tex` | `tables/tab_vw_quality` |
| | 구간 변화와 전략적 가치 개선의 정의 | `sec:value-svi` | `latex/chapters/05_value/04_svi_definition.tex` |  |
| | 방향, 평균, 규모는 다른 객체이다 | `sec:value-triad` | `latex/chapters/05_value/05_triad.tex` |  |
| | 교전 구간과 비교전 구간의 대조 | `sec:value-quiet` | `latex/chapters/05_value/06_quiet_contrast.tex` |  |
| | 물질적 결과와 다음 오브젝트와의 대응 | `sec:value-correspondence` | `latex/chapters/05_value/07_correspondence.tex` | `tables/gen/tab_material_axes`, `tables/tab_correspondence`, `tables/gen/tab_nextobj` |
| | 같은 사례에서의 다른 결과 정의와 후속 결과 모형 | `sec:value-samecase` | `latex/chapters/05_value/07b_same_case_labels.tex` |  |
| | 지평과 동료 평가기에 대한 안정성 | `sec:value-horizon` | `latex/chapters/05_value/08_horizon_stability.tex` | `tables/gen/tab_horizon_strata_T`, `tables/gen/tab_horizon_strata_S`, `tables/gen/tab_peer_transfer` |
| | 시간 경과와 관측 갱신의 분해 | `sec:value-shold` | `latex/chapters/05_value/09_clock_update_decomposition.tex` | `tables/gen/tab_shold` |
| | M-RQ2에 대한 답 | `sec:value-answer` | `latex/chapters/05_value/10_answer.tex` |  |
| **교전 전 정보 기반 예측** (`ch:prediction`) | | | | |
| | 방향 예측기 q | `sec:pred-q` | `latex/chapters/06_prediction/01_predictor.tex` |  |
| | 승률과 시간의 기준선 | `sec:pred-baselines` | `latex/chapters/06_prediction/02_baselines.tex` |  |
| | 선정, 보정, 동결 | `sec:pred-selection` | `latex/chapters/06_prediction/03_selection_freeze.tex` |  |
| | 평가 규칙 | `sec:pred-evaluation` | `latex/chapters/06_prediction/04_evaluation_rules.tex` |  |
| | 주 결과: 한타 코호트 T | `sec:pred-primary-T` | `latex/chapters/06_prediction/05_primary_T.tex` | `tables/tab_primary_T` |
| | 주 결과: 소규모 교전 코호트 S | `sec:pred-primary-S` | `latex/chapters/06_prediction/06_primary_S.tex` | `tables/tab_primary_S` |
| | 코호트 사이의 전이와 합동 학습 (이차, 예비 선언) | `sec:pred-transfer` | `latex/chapters/06_prediction/07_transfer_pooling.tex` | `tables/tab_transfer_pooling` |
| | 정보군 비교 | `sec:pred-infogroups` | `latex/chapters/06_prediction/08_learner_information.tex` | `tables/gen/tab_infogroups_arms`, `tables/gen/tab_infogroups_contrasts` |
| | M-RQ3에 대한 답 | `sec:pred-answer` | `latex/chapters/06_prediction/09_answer.tex` |  |
| **검증과 적용 범위** (`ch:validation`) | | | | |
| | 초기 우세에 대한 의존: 균형 구간 | `sec:val-b40` | `latex/chapters/07_validation/01_balanced_band.tex` | `tables/tab_b40` |
| | 작은 변화에 대한 민감도 (사후) | `sec:val-lambda` | `latex/chapters/07_validation/02_small_change_filters.tex` |  |
| | 관측 갱신 층화 | `sec:val-frame` | `latex/chapters/07_validation/02b_frame_strata.tex` | `tables/gen/tab_frame_strata` |
| | 확률 품질: 점수 분해 | `sec:val-corp` | `latex/chapters/07_validation/03_score_decomposition.tex` | `tables/tab_corp` |
| | 다른 패치와 지역: 점수 전용 외부 평가 | `sec:val-external` | `latex/chapters/07_validation/04_external.tex` | `tables/tab_ext_T`, `tables/tab_ext_S`, `tables/gen/tab_ext_bootstrap`, `tables/gen/tab_ext_common_h` |
| | M-RQ4에 대한 답 | `sec:val-answer-rq4` | `latex/chapters/07_validation/05_answer.tex` |  |
| **통합 논의와 결론** (`ch:discussion`) | | | | |
| | 결과의 연결 | `sec:disc-chain` | `latex/chapters/08_discussion/01_chain.tex` |  |
| | 해석의 범위 | `sec:disc-scope` | `latex/chapters/08_discussion/02_scope_of_interpretation.tex` | `tables/tab_scope` |
| | 평가기 의존성과 점수 분해의 읽기 | `sec:disc-evaluator` | `latex/chapters/08_discussion/03_evaluator_dependence.tex` |  |
| | 한계 | `sec:disc-limitations` | `latex/chapters/08_discussion/04_limitations.tex` |  |
| | 향후 과제 | `sec:disc-future` | `latex/chapters/08_discussion/05_future_work.tex` |  |
| | 결론 | `sec:disc-conclusion` | `latex/chapters/08_discussion/06_conclusion.tex` |  |
| **chapters/appendix/chapter** (``) | | | | |
| | 재현 자료 | `app:repro` | `latex/chapters/appendix/A_reproducibility.tex` |  |
| | 교전 정의 상수의 전체 표 | `app:constants` | `latex/chapters/appendix/B_constants.tex` |  |
| | 데이터 역할 census 보충 | `app:census` | `latex/chapters/appendix/C_census.tex` |  |
| | 수치의 증거 추적 | `app:trace` | `latex/chapters/appendix/D_evidence_trace.tex` |  |
| | 실행 상태와 이연된 계획 | `app:deferred` | `latex/chapters/appendix/E_deferred.tex` | `tables/gen/tab_status` |

## M-RQ ↔ 장

| M-RQ | 장 | 주 결과 | 보완 실험 |
|---|---|---|---|
| M-RQ1 교전 사례 구성 | 3, 4 | 정의·상수표·T/S 규칙 | e_fixed (4.6), OAT 사례 구성 (4.8) |
| M-RQ2 결과 가치 | 5 | C1–C8 | 물질 축·후속 결과 (5.7–5.8), 지평·동료 평가기 (5.9), S_hold (5.10) |
| M-RQ3 사전 예측 | 6 | C9–C10, C18–C20 | 정보군 비교 (6.8) |
| M-RQ4 적용 범위 | 7 | C11–C17, C21–C22 | 관측 갱신 층화 (7.3), 외부 부트스트랩 (7.5) |
