# 논문 구조 (OUTLINE)

8장 골격은 `docs/master_thesis_v1/README.md` 와 `docs/MASTER_THESIS_RESEARCH_PLAN_20260919.md` §7 을 따른다. Overleaf 프로젝트 루트는 `latex/`.
상태 열: `TODO` = 작성 미완, `저자 확인` = `\pendingauthor{}` 결정 대기 (`CLAIMS.md` 저자 pending 표).

## 조립 순서 (`latex/main.tex`)

표지 → 면지 → 속표지·인준지 → 차례(장·표·그림) → 국문 초록 → 본문 1–8장 → 참고 문헌 → 부록 A–E → 영문 초록

## M-RQ ↔ 장

| M-RQ | 장 | 주 claim |
|---|---|---|
| M-RQ1 교전 사례 구성 | 3, 4 | 정의·상수표·T/S 규칙 |
| M-RQ2 결과 가치 | 5 | C1–C8 |
| M-RQ3 사전 예측 | 6 | C9–C10, C18–C20 |
| M-RQ4 적용 범위 | 7 | C11–C17, C21–C22 |

## 장·절 지도

| 구분 | 제목 | 파일 | 포함 객체 | 상태 |
|---|---|---|---|---|
| **장** | **서론** | `latex/chapters/01_introduction/chapter.tex` | | |
| 절 | 연구 배경과 목적 | `latex/chapters/01_introduction/01_background.tex` |  |  |
| 절 | 선행 연구에서 학위논문으로: 네 단계의 개선 | `latex/chapters/01_introduction/02_improvement_logic.tex` | `figures/fig_pipeline` |  |
| 절 | 연구 질문 | `latex/chapters/01_introduction/03_research_questions.tex` |  |  |
| 절 | 기여와 비주장 | `latex/chapters/01_introduction/04_contributions_nonclaims.tex` |  |  |
| 절 | 증거의 지위와 집필 규칙 | `latex/chapters/01_introduction/05_epistemic_status.tex` |  |  |
| 절 | 논문의 구성 | `latex/chapters/01_introduction/06_outline.tex` |  |  |
| **장** | **관련연구와 선행연구의 위치** | `latex/chapters/02_related_work/chapter.tex` | | |
| 절 | (도입 문단) | `latex/chapters/02_related_work/00_lead.tex` |  |  |
| 절 | MOBA 경기의 승률 모형 | `latex/chapters/02_related_work/01_winprob_models.tex` |  |  |
| 절 | 교전 검출과 교전 단위 예측 | `latex/chapters/02_related_work/02_encounter_detection.tex` |  |  |
| 절 | 사건의 가치를 승률 변화로 매기기 | `latex/chapters/02_related_work/03_event_value.tex` |  |  |
| 절 | 적정 점수, 보정, 짝 비교 | `latex/chapters/02_related_work/04_scores_calibration.tex` |  |  |
| 절 | 측정 모형으로서의 교전 정의 | `latex/chapters/02_related_work/05_measurement_validity.tex` |  |  |
| 절 | 선행 연구의 위치 | `latex/chapters/02_related_work/06_position_of_prior_work.tex` |  |  |
| **장** | **데이터와 사전 정보 구성** | `latex/chapters/03_data/chapter.tex` | | |
| 절 | 비동기 공개 텔레메트리의 구조 | `latex/chapters/03_data/01_telemetry.tex` |  |  |
| 절 | 원천 코퍼스 | `latex/chapters/03_data/02_corpus.tex` | `tables/tab_corpus_census` |  |
| 절 | 데이터 역할 | `latex/chapters/03_data/03_roles.tex` | `tables/tab_roles_evaluator`, `tables/tab_roles_engagement` |  |
| 절 | 교전 전 상태와 입력의 구성 | `latex/chapters/03_data/04_prestate_inputs.tex` |  |  |
| 절 | 시간 계약과 누출 방지 | `latex/chapters/03_data/05_time_contract.tex` |  |  |
| **장** | **교전 사례의 정의와 구성** | `latex/chapters/04_engagement/chapter.tex` | | |
| 절 | (도입 문단) | `latex/chapters/04_engagement/00_lead.tex` |  |  |
| 절 | 킬 에피소드와 시간 경계 $G$ | `latex/chapters/04_engagement/01_temporal_boundary.tex` |  |  |
| 절 | 공간 경계 $D$ | `latex/chapters/04_engagement/02_spatial_boundary.tex` |  |  |
| 절 | 존재 게이트 $R$, $B$, $M$ | `latex/chapters/04_engagement/03_presence_gate.tex` |  |  |
| 절 | 조립 규칙과 모든 상수의 출처 | `latex/chapters/04_engagement/04_assembly_constants.tex` | `tables/tab_constants` |  |
| 절 | 규모 계급과 코호트 T, S | `latex/chapters/04_engagement/05_scale_cohorts.tex` |  | 저자 확인 |
| 절 | 결과 시점 규칙 | `latex/chapters/04_engagement/06_endpoint_rule.tex` |  |  |
| 절 | 킬에 조건화된 정의가 볼 수 없는 것 | `latex/chapters/04_engagement/07_killless.tex` |  |  |
| 절 | 정의 민감도: 계보의 증거와 미실행 항목 | `latex/chapters/04_engagement/08_sensitivity_lineage.tex` |  |  |
| 절 | 타당성 요약 | `latex/chapters/04_engagement/09_validity_summary.tex` |  |  |
| **장** | **경기 승패 기반 결과 가치** | `latex/chapters/05_value/chapter.tex` | | |
| 절 | 설계 원칙: 측정과 예측의 분리 | `latex/chapters/05_value/01_design_principle.tex` |  |  |
| 절 | 동결 승률 평가기 $\Vhat$ | `latex/chapters/05_value/02_evaluator.tex` |  |  |
| 절 | 평가기의 경기 승패 예측 품질 ($V \to W$) | `latex/chapters/05_value/03_vw_quality.tex` | `tables/tab_vw_quality` |  |
| 절 | 구간 변화와 전략적 가치 개선의 정의 | `latex/chapters/05_value/04_svi_definition.tex` |  |  |
| 절 | 방향, 평균, 규모는 다른 객체이다 | `latex/chapters/05_value/05_triad.tex` |  |  |
| 절 | 교전 구간과 비교전 구간의 대조 | `latex/chapters/05_value/06_quiet_contrast.tex` |  |  |
| 절 | 물질적 결과와 다음 오브젝트와의 대응 | `latex/chapters/05_value/07_correspondence.tex` | `tables/tab_correspondence` |  |
| 절 | 지평과 동료 평가기에 대한 안정성 | `latex/chapters/05_value/08_horizon_stability.tex` | `tables/tab_horizon` |  |
| 절 | M-RQ2에 대한 답과 해석 제한 | `latex/chapters/05_value/09_interpretation_limits.tex` |  |  |
| **장** | **교전 전 정보 기반 예측** | `latex/chapters/06_prediction/chapter.tex` | | |
| 절 | 방향 예측기 $q$ | `latex/chapters/06_prediction/01_predictor.tex` |  |  |
| 절 | 승률과 시간의 기준선 | `latex/chapters/06_prediction/02_baselines.tex` |  |  |
| 절 | 선정, 보정, 동결 | `latex/chapters/06_prediction/03_selection_freeze.tex` |  |  |
| 절 | 평가 규칙 | `latex/chapters/06_prediction/04_evaluation_rules.tex` |  |  |
| 절 | 주 결과: 한타 코호트 T | `latex/chapters/06_prediction/05_primary_T.tex` | `tables/tab_primary_T` |  |
| 절 | 주 결과: 소규모 교전 코호트 S | `latex/chapters/06_prediction/06_primary_S.tex` | `tables/tab_primary_S` |  |
| 절 | 코호트 사이의 전이와 합동 학습 (이차, 예비 선언) | `latex/chapters/06_prediction/07_transfer_pooling.tex` | `tables/tab_transfer_pooling` |  |
| 절 | 학습기와 정보 구성에 관하여 말할 수 있는 것 | `latex/chapters/06_prediction/08_learner_information.tex` |  |  |
| 절 | M-RQ3에 대한 답 | `latex/chapters/06_prediction/09_answer.tex` |  |  |
| **장** | **검증과 적용 범위** | `latex/chapters/07_validation/chapter.tex` | | |
| 절 | 초기 우세에 대한 의존: 균형 구간 $\Bforty$ | `latex/chapters/07_validation/01_balanced_band.tex` | `tables/tab_b40` |  |
| 절 | 작은 변화에 대한 민감도 (사후) | `latex/chapters/07_validation/02_small_change_filters.tex` |  |  |
| 절 | 확률 품질: 점수 분해 | `latex/chapters/07_validation/03_score_decomposition.tex` | `tables/tab_corp` |  |
| 절 | 다른 패치와 지역: 점수 전용 외부 평가 | `latex/chapters/07_validation/04_external.tex` | `tables/tab_ext_T`, `tables/tab_ext_S` | 저자 확인 |
| 절 | M-RQ4에 대한 답 | `latex/chapters/07_validation/05_answer.tex` |  |  |
| **장** | **통합 논의와 결론** | `latex/chapters/08_discussion/chapter.tex` | | |
| 절 | 네 단계는 어떻게 연결되는가 | `latex/chapters/08_discussion/01_chain.tex` |  |  |
| 절 | 선행 연구 대비 기여의 성격 | `latex/chapters/08_discussion/02_contribution_framing.tex` |  |  |
| 절 | 평가기 의존성과 점수 분해의 읽기 | `latex/chapters/08_discussion/03_evaluator_dependence.tex` |  |  |
| 절 | 주장하지 않는 것 | `latex/chapters/08_discussion/04_forbidden.tex` | `tables/tab_forbidden` |  |
| 절 | 한계 | `latex/chapters/08_discussion/05_limitations.tex` |  |  |
| 절 | 향후 과제 | `latex/chapters/08_discussion/06_future_work.tex` |  |  |
| 절 | 결론 | `latex/chapters/08_discussion/07_conclusion.tex` |  |  |
| **장** | **부록** | `latex/chapters/appendix/chapter.tex` | | |
| 절 | 재현 자료 | `latex/chapters/appendix/A_reproducibility.tex` |  | 저자 확인 |
| 절 | 교전 정의 상수의 전체 표 | `latex/chapters/appendix/B_constants.tex` |  |  |
| 절 | 데이터 역할 census 보충 | `latex/chapters/appendix/C_census.tex` |  |  |
| 절 | 수치의 증거 추적 | `latex/chapters/appendix/D_evidence_trace.tex` |  |  |
| 절 | 이연된 계획 | `latex/chapters/appendix/E_deferred.tex` |  |  |

## 객체 목록

| 라벨 | 파일 | 사용처 |
|---|---|---|
| `tab:b40` | `latex/tables/tab_b40.tex` | `chapters/07_validation/01_balanced_band.tex` |
| `tab:constants` | `latex/tables/tab_constants.tex` | `chapters/04_engagement/04_assembly_constants.tex` |
| `tab:corp` | `latex/tables/tab_corp.tex` | `chapters/07_validation/03_score_decomposition.tex` |
| `tab:corpus-census` | `latex/tables/tab_corpus_census.tex` | `chapters/03_data/02_corpus.tex` |
| `tab:correspondence` | `latex/tables/tab_correspondence.tex` | `chapters/05_value/07_correspondence.tex` |
| `tab:ext-S` | `latex/tables/tab_ext_S.tex` | `chapters/07_validation/04_external.tex` |
| `tab:ext-T` | `latex/tables/tab_ext_T.tex` | `chapters/07_validation/04_external.tex` |
| `tab:forbidden` | `latex/tables/tab_forbidden.tex` | `chapters/08_discussion/04_forbidden.tex` |
| `tab:horizon` | `latex/tables/tab_horizon.tex` | `chapters/05_value/08_horizon_stability.tex` |
| `tab:primary-S` | `latex/tables/tab_primary_S.tex` | `chapters/06_prediction/06_primary_S.tex` |
| `tab:primary-T` | `latex/tables/tab_primary_T.tex` | `chapters/06_prediction/05_primary_T.tex` |
| `tab:roles-engagement` | `latex/tables/tab_roles_engagement.tex` | `chapters/03_data/03_roles.tex` |
| `tab:roles-evaluator` | `latex/tables/tab_roles_evaluator.tex` | `chapters/03_data/03_roles.tex` |
| `tab:transfer-pooling` | `latex/tables/tab_transfer_pooling.tex` | `chapters/06_prediction/07_transfer_pooling.tex` |
| `tab:vw-quality` | `latex/tables/tab_vw_quality.tex` | `chapters/05_value/03_vw_quality.tex` |
| `fig:pipeline` | `latex/figures/fig_pipeline.tex` | `chapters/01_introduction/02_improvement_logic.tex` |
