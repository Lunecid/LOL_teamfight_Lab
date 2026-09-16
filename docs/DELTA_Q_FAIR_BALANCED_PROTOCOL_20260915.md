# 새 ΔV 표적의 공정 비교와 균형 상태 한타 분석

2026-09-15. 상태: **설계 명세 작성, 구현·학습 미실행**. Codex 설계/감사, Claude 구현/실행. 기존 결과를 확인한 뒤 정한 탐색적 후속 연구이며 사전등록·독립 확인 연구가 아니다. 이번 문서 작성은 장시간 학습 실행을 의미하지 않는다.

후속 사용자 요구 반영: [CoG 전체 모델 포함 추가 명세](COG_MODEL_COVERAGE_DELTA_Q_20260915.md)가 모델 범위를 확장한다. 308의 8개 모델과 118의 Layered+Logit 진단을 포함하고 plain MLP를 residual MLP와 분리한다. 아래 동일 352입력 비교와 CoG 입력 표현 비교는 서로 다른 트랙이다. 그래프/이벤트 adapter의 입력 충돌은 구현 전에 해소해야 한다.

## 1. 연구 질문과 이번 변경

- RQ1: 같은 교전 전 정보를 제공할 때, 선형·트리·신경망의 ΔV 개선 방향 예측과 확률 품질은 어떻게 다른가?
- RQ2: 교전 전 경기 승률이 50% 부근인 한타에서도, 전체 경기 우세만 사용하는 기준선을 넘어서는 예측력이 남는가?
- RQ3: 이 균형 상태에서 모델이 사용하는 특징군 및 기존 역할 모델의 설명은 전체 한타와 어떻게 다른가?

현재 동결된 V, ΔV 라벨, 경계, 모집단과 패치 역할을 유지한다. 새 학습은 q 비교에 한정한다. 승률 가치 모델을 새로 만들거나 Kim 논문의 손실함수를 도입하는 실험은 포함하지 않는다. 전체 모집단을 50:50 근처 사례로 대체하거나 해당 사례에 추가 학습 가중치를 주지 않는다.

이전 CoG의 수동 교환 점수, 기존 market_event 표적의 딥러닝 결과, 현재 ΔV 표적의 결과는 별도 계보다. 과거 딥러닝 모델의 성능표·선정 파라미터를 이번 표적의 검증 결과로 재사용하지 않는다. 코드 구조를 재사용하면 출처·변경점을 남긴다.

## 2. 동결 계약

기준: outputs/full_corpus_training_20260915 및 outputs/cohort_role_training_20260915의 스키마·라벨·분할·cohort manifest. 원본, 기존 models/predictions/reports, label_validity_full 및 objective_channel_ablation 결과는 읽기 전용이다. 새 root는 outputs/delta_q_fair_balanced_20260915, 새 코드 접두사는 dqfb20260915_. 기존 root로 로그를 쓰는 helper를 그대로 호출하지 않는다.

- MAIN 원시 210,000경기: 15.14 TRAIN, 15.15 VALIDATION, 15.16 TEST. 기존 외부 KR16.13, KR16.14 pilot, KR16.15, NA1 16.13은 각자 평가하며 합쳐서 새로운 검증 세트라고 부르지 않는다.
- T = 알려진 min(cluster_blue, cluster_red) >= 4인 한타; N = 알려진 min < 4인 부분. 규모 미상은 T/N 모두에서 제외하고 E에만 유지한다(기존 실행에서는 미상 0). 두 모집단을 각각 학습한다. T가 주요 주장 대상, N은 보조 비교다. 사후 규모로 분리한 retrospective conditional evaluation이며 실시간 라우팅 검증이 아니다.
- pre = s - 1ms. e(h) = min(last_kill+h, next_global_kill-1ms, next_engagement_start-1ms, game_end-1ms). 없는 사건은 무한대, 기존 중첩 제외 마스크 유지.
- p_pre=V(S_pre), ΔV=V(S_e)-V(S_pre), Y=1[ΔV>0]. 같은 행 양 끝에 같은 V; TRAIN은 해당 경기를 제외한 기존 OOF V. V와 라벨을 새 q의 결과에 맞춰 바꾸지 않는다.
- h90이 주 분석. h60/h120은 기존 라벨을 사용하는 민감도 분석으로 유지하며 주 상한을 재선택하지 않는다.
- 공통 입력은 q_pre_only_schema.json의 predictor_sets['ridge']에 해당하는 **352개 수치 특징**이다. 실제 JSON 경로와 순서는 구현 전 확인해 hash를 동결한다. p_pre_V 포함; 챔피언 ID, snapshot_age, 사후 위치/보간, 종료·참가·역할 정답·ΔV·최종 승자 등 추가 입력 금지. 기존 feature selector에서 352가 아니면 원인 확인 전 학습 중단.
- 역할 모델은 추가 draft 정보를 이용하므로 공통 352입력의 알고리즘 순위표에 섞지 않는다. 기존 역할 모델의 동결 예측을 별도 정보 표현 비교표에 싣는다.

## 3. 공정한 비교의 의미

동일 사례·라벨·사전 원정보, 동일 TRAIN 기반 전처리, 동일 경기 가중치, 동일 validation 역할·후보 수·seed 정책을 맞춘다. 서로 다른 모델의 최적화 알고리즘·파라미터 수·실행 시간까지 같다는 뜻은 아니다. 실제 학습 시간과 파라미터/트리 수, 수렴 상태를 보고한다.

### 비교 대상

1. 무정보 TRAIN 상수, p_pre만 쓰는 기존 logistic/spline 기준선.
2. 기존 동결 T/N specialist와 pooled q: 이전 상태에 대한 기준점. 새 탐색 예산이 없으므로 새로운 공정 후보군과 구분한다.
3. 새 L2 logistic regression: 공통 352입력.
4. 새 LightGBM: 공통 352입력. 과거 경제 입력만 쓰던 LightGBM과 구분한다.
5. 새 residual MLP: 공통 352입력의 표형 데이터용 신경망. Gorishniy 등의 ResNet형 기준선에서 동기를 얻은 명시적 구현이며, 원 논문 모델의 정확한 재현이나 최신 모든 deep 모델의 대표라고 부르지 않는다.

입력 결측 처리 규약은 기존 소스에서 확인해 동결하고 전 모델에 같은 원정보를 제공한다. 선형/신경망의 표준화는 TRAIN 적합, 트리는 같은 처리된 수치의 비표준화 표현을 허용한다. 실패나 수렴 문제 때문에 TEST를 본 뒤 유리한 모델만 남기지 않는다.

### 명시적 탐색 예산

각 새 family당 후보 6개. 확률 보정 후보는 동일하게 raw / sigmoid / isotonic 3개. 평가 log loss와 sigmoid의 logit 입력 clipping은 기존 helper의 명시적 정책을 재사용하고, 원 확률 및 0/1 출력 수를 보존한다. q의 기존 sigmoid는 logit에 대한 LogisticRegression 보정이며 양의 기울기 제약을 가정하지 않는다. 기울기를 기록하고 V에서 사용한 단조 보정 주장과 혼동하지 않는다.

- Logistic C: [0.0001, 0.001, 0.01, 0.1, 1, 10]. 기존 수렴 규약 적용, max_iter 증가 시 로그와 이유 보존.
- LightGBM: num_leaves [15,31,63] × min_child_samples [50,100]의 6조합. lr .04, 최대 1000 trees, lambda_l2 1, feature_fraction .9, subsample 1. 부스트/샘플 가중 구현을 명시한다.
- Residual MLP: width [128,256,512] × dropout [.1,.3]의 6조합. 입력 선형 투영, residual block 2개, 각 block LayerNorm -> Linear(width,2width) -> ReLU -> Dropout -> Linear(2width,width) -> Dropout -> residual add, 최종 LayerNorm -> ReLU -> Linear(width,1). weighted BCE-with-logits, AdamW lr .001, weight_decay .0001, batch_size 512, 최대 100epochs. preprocessing/전체 validation 계산 float 정밀도·loss 가중 정규화 기록.
- stochastic 모델은 seed [7,42,123] 모두 실행하고 3개 raw 확률의 평균을 해당 후보의 주 예측으로 사용한다. seed별 결과도 보존하며 좋은 seed 선택 금지. logistic은 결정적 solver를 사용해 중복 fitting하지 않는다. 이는 3-seed ensemble 대 단일 deterministic 모델 비교임을 명시하고 seed별 성능 분산도 제공한다.

경기 가중치는 기존 fc20260915_common.weights처럼 현재 fit 집단에서 각 경기의 총 가중치가 같고 행 평균이 1이 되도록 만든다. 신경망은 uniform shuffled minibatch에서 mean(w_i * BCE_i)를 사용한다. 매 minibatch의 가중치 합으로 다시 나누거나 weighted sampler와 중복 적용하지 않는다. 전체 집단 loss는 sum(w_i * BCE_i)/sum(w_i)로 집계하고 수치 일치 fixture를 둔다.

예산 수치·구조는 이번 연구의 고정된 운영 선택이며 선행 논문이 정한 최적값이 아니다. 이 제한된 탐색으로 deep 모델 일반의 우열을 주장하지 않는다. 후보 수를 동일하게 정하는 것만으로 완벽한 계산 공정성이 확보되는 것도 아니다.

### 적합·조기종료·보정·선정 순서

1. 기존 TRAIN 내부에서 경기 단위 hash `dqfb20260915_stop:<match_id>`를 기준으로 약 10%를 내부 조기종료 집단으로 지정한다. T/N 동일 경기에는 동일 배정. Q_CAL, Q_SELECT, TEST 사용 금지. 전처리는 이 단계의 실제 fit 90%에서만 적합한다. fit90/stop10/전체 refit 각각의 집단 안에서 경기 가중치를 다시 계산한다.
2. 각 tree/neural 후보·seed는 내부 stop의 경기 가중 Brier로 최적 iteration/epoch를 정한다. LightGBM에는 명시적 weighted Brier evaluator를 사용하며 기본 log loss를 조기종료 지표로 묵인하지 않는다. tree patience 50 rounds, neural patience 10 epochs; 동점이면 더 이른 checkpoint. 최대 cap 도달/실패를 기록한다. 내부 stop은 새 분할이 아니라 TRAIN 안의 최적화 보조 집단이며 최종 학습에서 제외하지 않는다.
3. 정해진 epoch/iteration로 후보·seed를 **해당 cohort의 전체 적격 TRAIN**에서 처음부터 다시 학습한다. 전처리도 전체 cohort TRAIN에 재적합. 일부 TRAIN smoke 결과를 최종 모델로 사용하지 않는다.
4. raw seed 평균에 대해 cohort Q_CAL에서 동일 경기 가중 calibrator를 적합한다. Q_SELECT의 경기 가중 Brier, 다음 log loss, 다음 고정 candidate 이름 순으로 family별 6×3 후보를 선택한다. Q_CAL/Q_SELECT 역할을 서로 바꾸거나 test 성능으로 보정법을 선택하지 않는다.
5. 전체 cohort Q_SELECT에서 최종 family도 같은 규칙으로 선택한다. **균형 상태에서 별도 모델·calibrator·seed를 선택하지 않는다.** 선택과 모든 후보 hash를 저장한 뒤 새 TEST/external 평가를 시작한다.
6. h60/h120에서는 h90에서 선정한 family별 hyperparameters를 고정한 채 해당 horizon의 TRAIN 라벨로 재학습하고, 동일 Q_CAL/Q_SELECT 규칙으로 보정만 선택한다. 그 적합도 TEST를 열기 전에 마친다. 균형 상태 분석은 h90 주 분석, 다른 상한은 보조 결과다.

CPU 총 4 threads, GPU 없이 순차 실행이 기본이다. CPU 자원으로 실행하지 못하면 실패/자원 상태를 남기고 deep 비교를 미완료로 표시한다. 축소 표본·축소 후보로 몰래 대체하거나 timeout을 모델 열등성으로 해석하지 않는다. 설치/추가 의존성이 필요한 경우 해당 결정이 구현 단계의 별도 사항임을 기록한다.

## 4. 균형 상태와 어려움의 분리

전체 cohort를 그대로 평가하면서 동일 동결 V의 p_pre로 다음 집단을 만든다.

- 주 균형 집단 B40: .40 <= p_pre <= .60.
- 보조 균형 집단 B45: .45 <= p_pre <= .55. B40 안에 포함되므로 독립 표본처럼 취급하지 않는다.
- 전체 구간: [0,.2), [.2,.4), [.4,.6], (.6,.8], (.8,1]. 모든 행이 정확히 하나에 포함되도록 경계 검사.
- 시간대: pre 기준 [0,10), [10,20), [20,30), [30,∞)분. 빈 셀도 표시한다.

위 threshold는 메일을 구체화한 연구 선택이며 논문의 권고값이 아니다. threshold를 결과에 맞춰 움직이거나 작은 Δ를 삭제하지 않는다. p_pre가 .5라는 것은 모델이 추정한 경기 균형이지, 실제 승률의 정답이나 한타 난이도 판정이 아니다. q가 .5 부근인 사례의 빈도는 별도 confidence 진단이며, 모델마다 다른 q로 주 평가 집단을 고르면 안 된다.

TRAIN의 균형 집단 표시에는 기존 OOF p_pre 사용. 학습/모델 선택은 전체 cohort로 수행한다. 후속 balanced-only 학습, 어려운 사례 가중치, 선택적 예측/기권, Δ 회귀는 이번 범위에 포함하지 않는다.

## 5. 지표와 통계

- 주 지표 Brier, 보조 log loss/AUC, prevalence, calibration intercept/slope, 고정 10구간 reliability/ECE. AUC를 정확도라고 쓰지 않는다. calibration slope가 좁은 확률 범위나 적은 표본에서 불안정하면 NA/진단을 기록한다.
- 모든 표에 행 수·고유 경기 수·양성 수/비율, 가중 방식을 표시한다. 한 셀 안 각 경기의 총 가중치를 동일하게 한다. 이는 셀마다 행 가중치가 달라질 수 있음을 뜻한다.
- 주 비교: T/h90에서 (a) Q_SELECT로 선정한 전체 새 q minus p_pre spline, (b) 새 LightGBM minus 새 logistic, (c) 새 residual MLP minus 새 LightGBM. 전체 T와 B40 각각 보고한다. 어느 결과가 유리한지에 따라 비교쌍을 바꾸지 않는다. 기존 specialist와의 차이는 보조 paired 비교다.
- 평가 경기 단위 paired bootstrap 1000회, seed 20260915. 같은 재표집 경기와 그 중복 횟수를 모든 비교 모델에 적용한다. ΔBrier/Δlog loss/ΔAUC와 95% percentile 구간을 저장한다. 고정 모델의 평가 표본 불확실성이며 전체 fitting/선정 불확실성은 아니다. 복수 결과는 탐색적 구간으로 함께 보고하고 한 셀의 유의성으로 결론을 선택하지 않는다.
- AUC는 양 클래스가 없으면 NA; 30경기 미만 셀은 희소 표시하고 구간/일반화 주장을 하지 않는다. 유효 bootstrap 수와 퇴화 replicate도 기록한다. 30은 운영적 보고 기준이지 통계적 충분성 보증이 아니다.
- 기존 V의 p_pre에 대해서는 실제 최종 승패 W를 사용해 B40·시간·패치별 calibration을 별도 진단한다. q는 Y로 평가한다. Y로 V를, W로 q를 보정하지 않는다. 이 진단 때문에 V나 집단 배정을 새로 변경하지 않는다.
- 전체와 B40의 차이는 집단 구성·유병률·시간대가 다르므로 난이도의 인과효과로 해석하지 않는다. 각 시간 셀 안 기준선 대비 추가 성능도 함께 제시한다.

## 6. SHAP과 역할 설명

이전 256사례 SHAP 안에서 B40만 남겨 전체 균형 집단 설명이라고 부르지 않는다. 새 hash 기준으로 h90 T 전체와 B40에서 각각 최대 256개 사례를 정하고 겹치는 사례를 표시한다. 새 모델 성능을 보고 사례를 고르지 않는다. 배경은 동일한 cohort TRAIN 최대 128사례로 고정한다.

공통-input 최종 선정 q는 기존 q_pre_only_schema.json의 shap_groups를 352열에 제한해 재사용한다: time_and_observation_age 2, health_mana_other 21, economy_and_experience 60, combat_and_survival 68, objectives 176, structures 24, prior_win_probability 1개. 합계 352, 중복/누락 없음과 mapping hash를 TEST 평가 전에 동결한다. time 그룹 이름에 age가 있어도 snapshot_age를 다시 넣지 않는다. unknown_objective_team_count는 부모 분류대로 health_mana_other에 남으며 순수 체력 그룹이라고 설명하지 않는다. 128 coalitions의 final calibrated probability group Shapley를 사용한다. 기존 역할 모델은 기존 5역할+global+uncertainty 7그룹으로 별도 설명한다.

각 모델에서 전체/B40의 signed local examples와 mean absolute contribution, 표본 수를 보고한다. 합산/재로딩 일치 검사. 모델·그룹 구성이 다른 표의 SHAP 크기를 직접 우열 순위로 비교하지 않는다. 상관된 특징과 인위적 혼합 상태를 사용하는 설명의 한계를 밝히고 인과적 승리 요인이라고 쓰지 않는다.

## 7. 완료 기준과 산출물

구현 전에 protocol.json에 본 문서 hash, parent artifact hashes, 정확한 split/cohort/feature 목록, 후보·seed·preprocessing·calibration·grouping 정책을 기록한다. 기존 label/state V를 새로 추출하거나 fitting하지 않는다.

필수 계약 검사: 정확한 row-key joins, 동일 라벨/endpoint/mask, T/N partition, 352열과 forbidden fields, fit-only 전처리, 기존 OOF 자기 경기 제외, split disjointness, balanced 경계와 시간 bins, 경기 가중 minibatch loss, ensemble/calibration 순서, TRAIN-only smoke, TEST 접근 전 selection freeze, 저장·재로딩 동일 예측, 독립 metric 재계산, SHAP 합산, 부모 파일 보존. 단순 PID나 preflight 통과는 완료가 아니다.

저장: RUN_STATUS.json, protocol.json, label/feature/split hashes, candidate registry, command/environment/failure logs, selection manifest, models, 행별 frozen predictions, cohort/balanced/time/patch metrics, paired bootstrap, calibration tables, SHAP arrays/group manifest, validation.json, 한국어 REPORT.md, DEFINITION_AND_EVIDENCE.md. 원자료/계정 식별자/인증키를 Claude 메시지에 넣지 않는다. 허용된 명세·코드·집계 중심 전달, 계산은 로컬에서 한다.

REPORT는 진행 전/후를 나누고 모델 개선 없음·희소 집단·실패도 포함한다. 기본 라벨·모델을 테스트 순위로 교체하지 않는다. 이 설계는 사람 검토를 요구하지 않으며, 생성 라벨의 의미 타당성이나 실시간 발생 탐지·한타 인과효과까지 검증한 것으로 확장하지 않는다.

## 8. 문헌·메일·자체 선택의 구분

- Gorishniy et al., NeurIPS 2021, *Revisiting Deep Learning Models for Tabular Data*, https://arxiv.org/abs/2106.11959 : ResNet형 강한 tabular baseline과 동일 평가/튜닝 프로토콜 비교의 동기. 우리 352입력, 후보 6개, residual 구조와 CPU 예산의 외부 정답성을 제공하지 않는다.
- Ke et al., NeurIPS 2017, *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html : gradient-boosted tree 방법의 출처. 새 표적에서의 우수성은 실험 전 미확인.
- Kim, Lee & Chung, CoG 2020, *A Confidence-Calibrated MOBA Game Winner Predictor*, https://ieee-cog.org/2020/papers/paper_221.pdf : 경기 승률의 입력 불확실성·보정 동기. balanced 한타에 집중하자는 제안은 협업 메일에서 왔으며 Kim 논문의 동일 실험이라고 쓰지 않는다. uncertainty-aware loss는 미구현.
- Lundberg & Lee, NeurIPS 2017, *A Unified Approach to Interpreting Model Predictions*, https://arxiv.org/abs/1705.07874 : 가산적 모델 설명의 근거. 표본·그룹·배경 선택은 우리 설계.
- ΔV 가치 해석·종료·경계의 문헌/운영 근거: docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md, docs/LABEL_VALIDITY_FINDINGS_20260915.md, docs/OBJECTIVE_CHANNEL_ABLATION_FINDINGS_20260915.md를 연결한다.

전체 연구의 경계 추정 단계가 TEST 패치를 사용한 기존 제한은 이번 q 비교로 해결되지 않는다. 독립 미래 패치 확인과 원고 통합은 별도 남은 작업이다.

## 9. 설계 단계 코드 대조 기록

Luna 탐색 및 Codex 통합 확인 완료. 재사용 경로: cr20260915_data.masks, fc20260915_fit_q.fit_candidates, fc20260915_common.weights/q_feature_sets/predict_q_bundle, cr20260915_common.fit_calibrators. 352열·그룹 수는 저장 JSON에서 직접 확인했다. 기존 deep/MLP 코드에 원래 표적·전처리·손실·조기종료 차이가 있어 이번 adapter를 새로 검증해야 한다. 각 과거 deep 실행의 정확한 target override는 전부 확인되지 않았으므로 일괄 단정하지 않는다.

검토 의견 중 minibatch마다 sum(weighted loss)/sum(batch weights)를 권하는 부분은 채택하지 않았다. uniform sampling과 전체 fit 평균 1로 정규화한 고정 가중치에서 mean(weighted loss)가 의도한 전체 경기 가중 risk의 불편 추정량이다. batch별 가변 분모는 다른 추정량이므로 구분한다. 전체 stop/evaluation에는 정확한 전체 가중 분모를 사용한다. 이 기록은 설계 대조이며 실제 신경망 구현/학습 검증 완료를 뜻하지 않는다.
