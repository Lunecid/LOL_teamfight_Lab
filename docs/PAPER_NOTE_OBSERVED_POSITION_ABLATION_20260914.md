# 논문용 기록: 마지막 관측 위치 추가 실험

2026-09-14. 사용자 요청으로 실험의 조건·결과·해석 및 원고 문장 초안을 보존한다. 이 문서는 새로운 실험 결과가 아니라 완료된 위치 ablation의 논문용 요약이다.

## 연구 질문과 비교 조건

미래 관측을 사용하지 않는 마지막 관측 좌표가 교전 결과 예측에 추가 정보를 제공하는가?

- 표적: 사건 기반 종료 B와 최대90초 상한으로 계산한 고정 가치 모델의 승률 개선 라벨 Y=1[Δp>0].
- B: 다음 추가 킬, 다음 적격 교전 시작, 경기 종료 중 먼저 발생한 사건 직전에 종료한다. 오브젝트 획득 자체는 종료 사건이 아니다.
- 기준: 기존 StateBuilder의 시점 안전 상태 특징+p_pre,363열.
- 비교: 기준에10명 전체의 마지막 관측 x/y좌표20열 추가,383열.
- snapshot_age_s는 두 조건에 공통 포함한다. 미래 프레임 보간, 미래 첫 킬 중심, 사후 참여자에 의한 위치 선택은 사용하지 않는다.
- 기존 역할 슬롯 순서 사용. 역할 메타데이터의 사전 가용성 가정은 두 조건 공통이며 별도 감사 대상이다.
- 학습기: LightGBM,250trees,15leaves,learning_rate0.04,min_child_samples100,lambda1,colsample_bytree0.9. seed7/42/123. 튜닝·early stopping 없음.
- 학습 가중치는 경기별 가중치 합이 동일하도록 구성한다. 평가 지표도 경기 동일 가중이다.

## 표본과 분할

| 패치 | 역할 | 경기 수 | 교전 수 |
|---|---|---:|---:|
| 15.14 | 학습 | 3,218 | 9,228 |
| 15.15 | 개발 확인 | 3,325 | 9,801 |
| 15.16 | 탐색 평가 | 2,655 | 7,664 |
| 합계 | | 9,198 | 26,693 |

교전 표본은 가치 모델의 fit/calibrate/select/test 경기 집합과 교집합0인 기존 engagement 파티션이다.15.16은 프로젝트에서 이미 사용한 패치이므로 완전 미사용 외부 검증이라고 표현하지 않는다.

## 핵심 결과:15.16,3seed평균 확률

| 지표 | 위치 제외 | 위치 추가 | 추가−제외 | 차이의95% CI |
|---|---:|---:|---:|---:|
| AUC | 0.558069 | 0.561191 | +0.003122 | −0.002872~+0.009325 |
| Brier score | 0.254335 | 0.253549 | −0.000786 | −0.002169~+0.000541 |
| Log loss | 0.703702 | 0.702036 | −0.001666 | −0.004560~+0.001246 |

AUC는 증가가 개선, Brier/logloss는 감소가 개선이다. CI는 평가 경기 단위 paired bootstrap500회(seed7). 학습 모델을 고정한 표집 불확실성이며, 모든 학습 과정을 재표집한 불확실성은 아니다. seed별 결과는 원본 results.json에 보존했다.

## 해석과 현재 처리

세 지표의 점추정치는 소폭 개선되었으나 차이의95% CI가 모두0을 포함했다. 이번 단순 모델·표본·분할에서는 마지막 관측 좌표의 명확한 추가 효과를 확인하지 못했다. 이는 위치의 무용성을 증명하거나 두 조건의 통계적 동등성을 입증한 결과가 아니다. 이 결과에 근거해 현 단계의 기준 모델은 위치 제외로 유지하고 위치 추가는 민감도 분석으로 보존하는 것이 제안되었다. 모든 최종 모델에서 영구 제외한다는 확정은 아니다.

이 비교는 앞서 시행한 '다음 프레임 위치 복원 오차' 실험과 다르다. 여기서는 실제 관측 좌표의 교전 예측 기여를 평가했으며, 보간/외삽 좌표는 넣지 않았다. 기존7116열 원고 모델의 전체 재현이나 다른 모든 학습기에서의 효과를 입증하지 않는다. AUC가 약0.56이라는 절대 수준과 확률 지표도 함께 공개하여 작은 상대 개선을 과장하지 않는다.

관측 나이별 AUC는 별도 보고서의 탐색 진단이다. 특히 신선한 좌표에서 더 큰 점추정 개선이 보이더라도, 별도 불확실성·상호작용 검증 없이 '최근 위치에서 효과가 입증됐다'고 주장하지 않는다.

## 논문용 한국어 초안

“마지막 관측 위치의 추가 예측 기여를 평가하기 위해, 동일한 교전 표본, 표적, 시간순 패치 분할 및 학습 설정에서 위치를 제외한 기준 모델과10명 선수의 마지막 관측 좌표20개를 추가한 모델을 비교하였다. 관측 경과 시간은 두 조건에 동일하게 포함하였다. 패치15.16의2,655경기7,664교전에서3개 시드의 평균 예측 확률을 평가한 결과, 경기 동일 가중 AUC는0.5581에서0.5612로 증가하였다. 그러나 차이0.0031의 경기 단위 bootstrap95% 신뢰구간은[−0.0029,0.0093]으로0을 포함했으며, Brier score 및 log loss 차이의 신뢰구간도0을 포함했다. 따라서 본 실험에서는 마지막 관측 좌표의 명확한 추가 효과를 확인하지 못했다. 이 결과는 평가한 모델과 관측 해상도에 한정되며, 위치 정보 일반의 무용성을 의미하지 않는다.”

## English manuscript draft

“To assess the incremental predictive contribution of last-observed positions, we compared a position-free baseline with a model augmented by the 20 last-observed coordinates of all 10 players, using identical engagement samples, targets, chronological patch splits, and training settings. Snapshot age was included in both conditions. On 7,664 engagements from 2,655 matches in patch 15.16, the match-weighted AUC of predictions averaged across three seeds increased from 0.5581 to 0.5612. However, the paired match-bootstrap 95% confidence interval for the AUC difference of 0.0031 was [−0.0029, 0.0093]; the confidence intervals for the Brier-score and log-loss differences also included zero. We therefore found no clear incremental benefit from last-observed coordinates under this experimental configuration. This finding is specific to the evaluated model and observation resolution and does not establish that positional information is generally uninformative.”

투고 시 대상 저널의 표기 형식에 맞추고, 본 비교가 탐색적 패치 평가라는 점을 방법 절에 명시한다.

## 검증 및 원자료

- 중복 교전0, 여러 패치에 속한 동일 경기0, 미래 snapshot0.
- 위치 추가 모델의 앞363열이 기준 입력과 동일하며 관측 나이도 양 조건에 존재함을 검사했다.
- [실행 전 명세](POSITION_ABLATION_PROTOCOL_20260914.md)
- [전체 결과 및 관측 나이별 진단](POSITION_ABLATION_RESULTS_20260914.md)
- [수치 원본](../outputs/position_ablation_20260914/results.json)
- [입력 열](../outputs/position_ablation_20260914/feature_names.json)
- [예측 확률](../outputs/position_ablation_20260914/predictions.csv)
- [검증 기록](../outputs/position_ablation_20260914/verification.json)
- [학습/평가 코드](../scripts/ablate_observed_position.py)
- [위치 복원 오차 실험: 별도 질문](CAUSAL_POSITION_RESULTS_20260914.md)
