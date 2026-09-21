# 논문 구조 (OUTLINE)

원고의 장·절 지도이다. 파일 경로는 저장소 기준이며, Overleaf 프로젝트 루트는 `latex/` 이다.
상태 열의 `TODO` 는 본문 또는 포함 객체에 `\todo{}` 가 남아 있음을 뜻한다.
절을 추가·삭제·재배열할 때는 해당 장의 `chapter.tex` 의 `\input` 목록만 고치고, 이 문서를 갱신한다.

## 조립 순서 (`latex/main.tex`)

1. 1구역: 표지 → 면지 → 속표지 및 인준지 (`pnuthesis.cls` 의 `\makecover`, `\makeblankpage`, `\makeapproval`)
2. 2구역: 차례 → 표 차례 → 그림 차례 (`\makefrontmatter`) → 국문 초록 (`frontmatter/abstract_ko.tex`)
3. 3구역: 본문 제 1–11 장 → 참고문헌 (`bib/references.bib`, IEEEtranN) → 부록 A–D → 영문 초록 (`frontmatter/abstract_en.tex`)

## 장·절 지도

| 구분 | 제목 | 파일 | 포함 객체 | 상태 |
|---|---|---|---|---|
| **장** | **제 1 장 서론** | `latex/chapters/01_introduction/chapter.tex` | | |
| 절 | 1.1 연구 배경 | `latex/chapters/01_introduction/01_background.tex` |  |  |
| 절 | 1.2 문제 정의와 도전 과제 | `latex/chapters/01_introduction/02_challenges.tex` |  |  |
| 절 | 1.3 연구 질문 | `latex/chapters/01_introduction/03_research_questions.tex` |  |  |
| 절 | 1.4 기여 | `latex/chapters/01_introduction/04_contributions.tex` |  |  |
| 절 | 1.5 논문의 구성 | `latex/chapters/01_introduction/05_outline.tex` |  |  |
| **장** | **제 2 장 배경 이론 및 관련 연구** | `latex/chapters/02_background/chapter.tex` | | |
| 절 | 제 2 장 도입 문단 | `latex/chapters/02_background/00_lead.tex` |  |  |
| 절 | 2.1 리그 오브 레전드와 Match-V5 텔레메트리 | `latex/chapters/02_background/01_telemetry.tex` |  |  |
| 절 | 2.2 MOBA 승패 예측 연구 | `latex/chapters/02_background/02_related_work.tex` |  |  |
| 절 | 2.3 이진 분류의 확률적 정식화 | `latex/chapters/02_background/03_classification.tex` |  |  |
| 절 | 2.4 평가 지표의 정의와 기원 | `latex/chapters/02_background/04_metrics.tex` |  |  |
| 절 | 2.5 예측기 비교를 위한 통계 검정 | `latex/chapters/02_background/05_statistical_tests.tex` |  |  |
| 절 | 2.6 학습 모델의 계보 | `latex/chapters/02_background/06_model_lineage.tex` |  |  |
| 절 | 2.7 학습 목적함수의 변형과 정규화 | `latex/chapters/02_background/07_objectives.tex` |  |  |
| 절 | 2.8 데이터 누출, 공변량 이동, 선택 편향 | `latex/chapters/02_background/08_leakage_shift_bias.tex` |  |  |
| 절 | 2.9 설명 가능성 | `latex/chapters/02_background/09_explainability.tex` |  |  |
| 절 | 2.10 검출기 검증을 위한 일치도 지표 | `latex/chapters/02_background/10_agreement.tex` |  |  |
| **장** | **제 3 장 문제 정식화** | `latex/chapters/03_problem/chapter.tex` | | |
| 절 | 3.1 텔레메트리의 형식적 모델 | `latex/chapters/03_problem/01_telemetry_model.tex` |  |  |
| 절 | 3.2 교전의 정의 | `latex/chapters/03_problem/02_engagement_definition.tex` |  |  |
| 절 | 3.3 예측 문제 | `latex/chapters/03_problem/03_prediction_problem.tex` |  |  |
| 절 | 3.4 무누출 계약 | `latex/chapters/03_problem/04_no_leakage_contract.tex` |  |  |
| 절 | 3.5 표기 요약 | `latex/chapters/03_problem/05_notation.tex` | `tables/tab_notation` |  |
| **장** | **제 4 장 킬 조건부 교전 국소화** | `latex/chapters/04_localization/chapter.tex` | | |
| 절 | 제 4 장 도입 문단 | `latex/chapters/04_localization/00_lead.tex` |  |  |
| 절 | 4.1 5 초 위치 격자 | `latex/chapters/04_localization/01_position_grid.tex` |  |  |
| 절 | 4.2 시간적 킬 군집화 | `latex/chapters/04_localization/02_temporal_clustering.tex` |  |  |
| 절 | 4.3 공간 지름 분할 | `latex/chapters/04_localization/03_spatial_split.tex` |  |  |
| 절 | 4.4 온셋 검증 | `latex/chapters/04_localization/04_onset_validation.tex` |  |  |
| 절 | 4.5 인접 후보 병합 | `latex/chapters/04_localization/05_merge.tex` |  |  |
| 절 | 4.6 교전 유형 분류와 부가 정보 | `latex/chapters/04_localization/06_fight_type.tex` |  |  |
| 절 | 4.7 알고리즘과 복잡도 | `latex/chapters/04_localization/07_algorithm_complexity.tex` | `algorithms/alg_localization` |  |
| 절 | 4.8 검출기 타당성 검증 설계 | `latex/chapters/04_localization/08_validation_design.tex` |  |  |
| **장** | **제 5 장 교환가치 라벨** | `latex/chapters/05_label/chapter.tex` | | |
| 절 | 5.1 동기 | `latex/chapters/05_label/01_motivation.tex` |  |  |
| 절 | 5.2 정의 | `latex/chapters/05_label/02_definition.tex` | `tables/tab_label_weights` |  |
| 절 | 5.3 특수 킬 마커와 동점 처리 | `latex/chapters/05_label/03_special_kills_ties.tex` |  |  |
| 절 | 5.4 보조 타깃 | `latex/chapters/05_label/04_auxiliary_targets.tex` |  |  |
| **장** | **제 6 장 다중 모달 특징 표현** | `latex/chapters/06_features/chapter.tex` | | |
| 절 | 제 6 장 도입 문단 | `latex/chapters/06_features/00_lead.tex` |  |  |
| 절 | 6.1 노드 특징 ($F_{\mathrm{node | `latex/chapters/06_features/01_node_features.tex` | `tables/tab_node_groups` |  |
| 절 | 6.2 전역 특징 ($F_{\mathrm{glob | `latex/chapters/06_features/02_global_features.tex` |  |  |
| 절 | 6.3 사건 특징 ($F_{\mathrm{ev | `latex/chapters/06_features/03_event_features.tex` |  |  |
| 절 | 6.4 관측 창 구성과 정규화 | `latex/chapters/06_features/04_window_normalization.tex` |  |  |
| 절 | 6.5 테이블 요약 표현 | `latex/chapters/06_features/05_tabular_summary.tex` |  |  |
| 절 | 6.6 플레이어 상호작용 그래프 | `latex/chapters/06_features/06_interaction_graph.tex` |  |  |
| **장** | **제 7 장 예측 모델과 도메인 지식 처치** | `latex/chapters/07_models/chapter.tex` | | |
| 절 | 제 7 장 도입 문단 | `latex/chapters/07_models/00_lead.tex` |  |  |
| 절 | 7.1 테이블 기준선: LightGBM | `latex/chapters/07_models/01_lightgbm.tex` |  |  |
| 절 | 7.2 정합 입력 다층 퍼셉트론 | `latex/chapters/07_models/02_matched_mlp.tex` |  |  |
| 절 | 7.3 순차 모델 | `latex/chapters/07_models/03_sequential.tex` |  |  |
| 절 | 7.4 그래프 신경망 | `latex/chapters/07_models/04_gnn.tex` |  |  |
| 절 | 7.5 시공간 모델 | `latex/chapters/07_models/05_spatiotemporal.tex` |  |  |
| 절 | 7.6 계층적 융합 | `latex/chapters/07_models/06_layered_fusion.tex` |  |  |
| 절 | 7.7 스태킹 앙상블과 사후 보정 | `latex/chapters/07_models/07_stacking_calibration.tex` |  |  |
| 절 | 7.8 도메인 지식 처치 T1--T7 | `latex/chapters/07_models/08_treatments.tex` |  |  |
| **장** | **제 8 장 실험 설계** | `latex/chapters/08_experimental_setup/chapter.tex` | | |
| 절 | 8.1 데이터 | `latex/chapters/08_experimental_setup/01_data.tex` | `tables/tab_data` |  |
| 절 | 8.2 분할 | `latex/chapters/08_experimental_setup/02_split.tex` |  |  |
| 절 | 8.3 학습 설정 | `latex/chapters/08_experimental_setup/03_training.tex` | `tables/tab_training` |  |
| 절 | 8.4 평가 지표와 통계 검정 | `latex/chapters/08_experimental_setup/04_metrics_tests.tex` |  |  |
| 절 | 8.5 절제 프로토콜 | `latex/chapters/08_experimental_setup/05_ablation_protocol.tex` | `tables/tab_protocol` |  |
| 절 | 8.6 재현성 | `latex/chapters/08_experimental_setup/06_reproducibility.tex` |  |  |
| **장** | **제 9 장 결과** | `latex/chapters/09_results/chapter.tex` | | |
| 절 | 제 9 장 도입 문단 | `latex/chapters/09_results/00_lead.tex` |  | TODO |
| 절 | 9.1 기준 성능 | `latex/chapters/09_results/01_baseline.tex` | `tables/tab_baseline` |  |
| 절 | 9.2 표현 효과와 학습기 효과의 분해 | `latex/chapters/09_results/02_decomposition.tex` |  |  |
| 절 | 9.3 조건부 예측 가능성 | `latex/chapters/09_results/03_conditional.tex` | `tables/tab_conditional` |  |
| 절 | 9.4 도메인 지식 처치의 효과 | `latex/chapters/09_results/04_ablation.tex` | `tables/tab_ablation` `figures/fig_forest` | TODO |
| 절 | 9.5 특징 기여도 해석 | `latex/chapters/09_results/05_shap.tex` | `tables/tab_shap` | TODO |
| 절 | 9.6 검출기 검증 결과 | `latex/chapters/09_results/06_detector.tex` | `tables/tab_detector` | TODO |
| **장** | **제 10 장 고찰** | `latex/chapters/10_discussion/chapter.tex` | | |
| 절 | 10.1 정보 입도 한계 | `latex/chapters/10_discussion/01_granularity.tex` |  |  |
| 절 | 10.2 설계 결정에 대한 논의 | `latex/chapters/10_discussion/02_design_decisions.tex` |  |  |
| 절 | 10.3 코드--논문 감사 | `latex/chapters/10_discussion/03_audit.tex` | `tables/tab_audit` | TODO |
| 절 | 10.4 타당성 위협 | `latex/chapters/10_discussion/04_threats.tex` |  |  |
| 절 | 10.5 한계 | `latex/chapters/10_discussion/05_limitations.tex` |  |  |
| **장** | **제 11 장 결론 및 향후 연구** | `latex/chapters/11_conclusion/chapter.tex` | | |
| 절 | 11.1 결론 | `latex/chapters/11_conclusion/01_conclusion.tex` |  | TODO |
| 절 | 11.2 향후 연구 | `latex/chapters/11_conclusion/02_future_work.tex` |  |  |
| **장** | **부록 모듈: 각 부록을 순서대로 조립** | `latex/chapters/appendix/chapter.tex` | | |
| 절 | 부록 A 특징 목록 | `latex/chapters/appendix/A_features.tex` | `tables/tab_app_node` |  |
| 절 | 부록 B 모델 하이퍼파라미터 | `latex/chapters/appendix/B_hyperparameters.tex` | `tables/tab_app_hparams` |  |
| 절 | 부록 C 교전 국소화 상수 | `latex/chapters/appendix/C_detection_constants.tex` | `tables/tab_app_detection` |  |
| 절 | 부록 D 처치 설정과 하이퍼파라미터 격자 | `latex/chapters/appendix/D_treatments.tex` | `tables/tab_app_treatments` |  |

## 객체 목록 (표·그림·알고리즘)

| 라벨 | 파일 | 사용처 | 상태 |
|---|---|---|---|
| `tab:ablation` | `latex/tables/tab_ablation.tex` | `chapters/09_results/04_ablation.tex` | TODO |
| `tab:app-detection` | `latex/tables/tab_app_detection.tex` | `chapters/appendix/C_detection_constants.tex` |  |
| `tab:app-hparams` | `latex/tables/tab_app_hparams.tex` | `chapters/appendix/B_hyperparameters.tex` |  |
| `tab:app-node` | `latex/tables/tab_app_node.tex` | `chapters/appendix/A_features.tex` |  |
| `tab:app-treatments` | `latex/tables/tab_app_treatments.tex` | `chapters/appendix/D_treatments.tex` |  |
| `tab:audit` | `latex/tables/tab_audit.tex` | `chapters/10_discussion/03_audit.tex` |  |
| `tab:baseline` | `latex/tables/tab_baseline.tex` | `-` |  |
| `tab:conditional` | `latex/tables/tab_conditional.tex` | `-` |  |
| `tab:data` | `latex/tables/tab_data.tex` | `chapters/08_experimental_setup/01_data.tex` |  |
| `tab:detector` | `latex/tables/tab_detector.tex` | `chapters/09_results/06_detector.tex` | TODO |
| `tab:label-weights` | `latex/tables/tab_label_weights.tex` | `chapters/05_label/02_definition.tex` |  |
| `tab:node-groups` | `latex/tables/tab_node_groups.tex` | `chapters/06_features/01_node_features.tex` |  |
| `tab:notation` | `latex/tables/tab_notation.tex` | `chapters/03_problem/05_notation.tex` |  |
| `tab:protocol` | `latex/tables/tab_protocol.tex` | `chapters/08_experimental_setup/05_ablation_protocol.tex` |  |
| `tab:shap` | `latex/tables/tab_shap.tex` | `chapters/09_results/05_shap.tex` | TODO |
| `tab:training` | `latex/tables/tab_training.tex` | `chapters/08_experimental_setup/03_training.tex` |  |
| `fig:forest` | `latex/figures/fig_forest.tex` | `chapters/09_results/04_ablation.tex` | TODO |
| `alg:localization` | `latex/algorithms/alg_localization.tex` | `chapters/04_localization/07_algorithm_complexity.tex` |  |

## 연구 질문 ↔ 절 대응

| RQ | 답하는 절 | 근거 객체 |
|---|---|---|
| RQ1 국소화 | 4.1–4.8, 9.6 | `alg:localization`, `tab:detector` |
| RQ2 표현 | 9.1, 9.2 | `tab:baseline` |
| RQ3 학습기 | 7.2, 9.2 | `tab:baseline` |
| RQ4 조건부 예측 가능성 | 9.3, 10.1 | `tab:conditional` |
| RQ5 도메인 지식 처치 | 7.8, 8.5, 9.4 | `tab:ablation`, `fig:forest`, `tab:app-treatments` |
