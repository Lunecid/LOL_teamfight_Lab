# LoL V→ΔV→q 연구용 계량경제학·확률예측 문헌 지도

검색 확인일: 2026-09-20. 아래 적용 제안은 선행연구의 게임 데이터상 성능을 보고한 것이 아니다. 원 논문의 조건과 현재 프로젝트의 가정을 구분한다.

## 우선 읽기

Christoffersen–Diebold (2006) → Foster–Stine (2021) → Dimitriadis et al. (2021) → Clements–Harvey (2010) → Giacomini–White (2006).

## 현재 작업과 연결

- RR1·RR2: 확률예측 포섭/추가 정보와 조건별 손실 비교.
- RR3·RR4: 사건 기반 배경 변화 비교, 평균/방향/변동성 분리.
- RR6: 시점별 보정과 전체 경기 예측 경로의 정합성.
- 석사 심화/별도 v2: 예측 경로 후처리, 상태별 예측 결합, V 재적합 불확실성.

## 논문별 정보

### 1. Financial Asset Returns, Direction-of-Change Forecasting, and Volatility Dynamics
Peter F. Christoffersen and Francis X. Diebold (2006). Management Science.

- 논문: https://pubsonline.informs.org/doi/10.1287/mnsc.1060.0520
- 공개본·원문 저장소 또는 이전 작업본: https://www.nber.org/papers/w10009
- 우리 적용과 제한: 방향 예측과 조건부 평균·변동성의 관계. SVI 유지 논리와 상태별 변동 규모 진단. 금융 수익률 가정이 게임에 자동 적용되지는 않음.

### 2. Threshold Martingales and the Evolution of Forecasts
Dean P. Foster and Robert A. Stine (2021). arXiv preprint.

- 논문: https://arxiv.org/abs/2105.06834
- 공개본·원문 저장소 또는 이전 작업본: https://arxiv.org/pdf/2105.06834
- 우리 적용과 제한: 시점별 확률 품질과 예측 경로의 변동성을 구분. 전체 경기 경로의 진단·별도 버전 후처리 후보. 프리프린트. 필터는 [0,1] 밖 예측을 낼 수 있어 그대로 확률 평가기로 쓰지 않음.

### 3. Forecast Rationality Tests Based on Multi-Horizon Bounds
Andrew J. Patton and Allan Timmermann (2012). Journal of Business & Economic Statistics.

- 논문: https://doi.org/10.1080/07350015.2012.634337
- 공개본·원문 저장소 또는 이전 작업본: https://cepr.org/publications/dp8194
- 우리 적용과 제한: 다중 시점 예측·수정의 합리성 진단. 고정 최종 결과·정보집합 조건과 교전 사후 선택의 차이를 구분.

### 4. Forecast Encompassing Tests and Probability Forecasts
Michael P. Clements and David I. Harvey (2010). Journal of Applied Econometrics.

- 논문: https://doi.org/10.1002/jae.1097
- 공개본·원문 저장소 또는 이전 작업본: https://wrap.warwick.ac.uk/id/eprint/4972/
- 우리 적용과 제한: 확률예측의 추가 정보·포섭. 단독 ΔBrier 개선과 포섭은 다름. q/PT 결합은 개발자료에서만 적합하는 별도 버전.

### 5. Tests of Conditional Predictive Ability
Raffaella Giacomini and Halbert White (2006). Econometrica.

- 논문: https://doi.org/10.1111/j.1468-0262.2006.00718.x
- 우리 적용과 제한: 전체 평균과 상태별 예측력 구분. B40·시간대 손실 차이 분석. 원 논문의 시계열 검정을 경기 단위 자료에 그대로 적용했다고 쓰지 않음.

### 6. Strictly Proper Scoring Rules, Prediction, and Estimation
Tilmann Gneiting and Adrian E. Raftery (2007). Journal of the American Statistical Association.

- 논문: https://doi.org/10.1198/016214506000001437
- 공개본·원문 저장소 또는 이전 작업본: https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation-revised
- 우리 적용과 제한: V와 q 확률 평가에 Brier/log loss를 사용하는 근거. 적정 점수규칙이 라벨 의미의 타당성을 보증하지 않음.

### 7. Stable Reliability Diagrams for Probabilistic Classifiers
Timo Dimitriadis and Tilmann Gneiting and Alexander I. Jordan (2021). Proceedings of the National Academy of Sciences.

- 논문: https://doi.org/10.1073/pnas.2016191118
- 공개본·원문 저장소 또는 이전 작업본: https://publikationen.bibliothek.kit.edu/1000130510
- 우리 적용과 제한: CORP reliability와 점수 분해. V→W, q→SVI를 각각 진단. TEST에서 진단용으로 얻은 재보정 곡선을 동결 예측 개선에 사용하지 않음.

### 8. Using Daily Stock Returns: The Case of Event Studies
Stephen J. Brown and Jerold B. Warner (1985). Journal of Financial Economics.

- 논문: https://doi.org/10.1016/0304-405X(85)90042-X
- 우리 적용과 제한: 사건 전후 기대 외 변동과 사건 유발 분산, 데이터 종속성의 설계. Quiet 참고 비교에 아이디어를 차용하되 인과효과로 해석하지 않음.

### 9. Smart Kills and Worthless Deaths: eSports Analytics for League of Legends
Philip Z. Maymin (2021). Journal of Quantitative Analysis in Sports.

- 논문: https://doi.org/10.1515/jqas-2019-0096
- 공개본·원문 저장소 또는 이전 작업본: https://www.degruyterbrill.com/document/doi/10.1515/jqas-2019-0096/html
- 우리 적용과 제한: LoL 사건을 추정 승률 변화로 평가하는 직접 선행연구. 고해상도 자료, 사후 기여 평가와 공개 분 단위 자료의 사전 방향 예측을 구분.

### 10. Actions Speak Louder than Goals: Valuing Player Actions in Soccer
Tom Decroos and Lotte Bransen and Jan Van Haaren and Jesse Davis (2019). Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

- 논문: https://doi.org/10.1145/3292500.3330758
- 공개본·원문 저장소 또는 이전 작업본: https://arxiv.org/abs/1802.07127
- 우리 적용과 제한: 상황에 따른 공격·수비 결과 확률 변화로 행동 가치를 평가. 최종 경기 승률과 다음 득실점 확률은 다른 대상.

### 11. Exploring the Difficulty of Estimating Win Probability: A Simulation Study
Ryan S. Brill and Ronald Yurko and Abraham J. Wyner (2026). Journal of Quantitative Analysis in Sports.

- 논문: https://doi.org/10.1515/jqas-2024-0130
- 공개본·원문 저장소 또는 이전 작업본: https://arxiv.org/abs/2406.16171
- 우리 적용과 제한: 경기별 공유 승패와 관측 종속성의 추정 불확실성. 원고는 2024 preprint, 2025 online, 2026 issue. 특정 시뮬레이션의 ESS·구간 폭을 우리 데이터에 복사하지 않음.

### 12. A Practitioner's Guide to Cluster-Robust Inference
A. Colin Cameron and Douglas L. Miller (2015). Journal of Human Resources.

- 논문: https://doi.org/10.3368/jhr.50.2.317
- 공개본·원문 저장소 또는 이전 작업본: https://escholarship.org/uc/item/1jq5d0pq
- 우리 적용과 제한: 같은 경기 안의 교전·시간 행 종속성 고려. 경기 가중 통계와 cluster bootstrap의 근거. 클러스터 사이 독립성 가정도 명시.

### 13. A Simple Nonparametric Test of Predictive Performance
M. Hashem Pesaran and Allan Timmermann (1992). Journal of Business & Economic Statistics.

- 논문: https://doi.org/10.1080/07350015.1992.10509922
- 우리 적용과 제한: 방향 예측 성능과 주변 확률을 구분하는 고전적 기준. 기본 검정의 자료 가정 점검 필요. Brier 주 평가를 대체하지 않음.

### 14. Comparing Predictive Accuracy
Francis X. Diebold and Roberto S. Mariano (1995). Journal of Business & Economic Statistics.

- 논문: https://doi.org/10.1080/07350015.1995.10524599
- 우리 적용과 제한: 예측 손실 차이 비교의 고전적 근거. 현재 경기 bootstrap을 그대로 DM test라고 부르지 않음.

### 15. The Model Confidence Set
Peter R. Hansen and Asger Lunde and James M. Nason (2011). Econometrica.

- 논문: https://doi.org/10.3982/ECTA5771
- 우리 적용과 제한: 여러 후보에서 하나의 확정 우승자보다 구분하기 어려운 모델 집합을 보고하는 방법. 현재 동결 모델 재선정의 이유로 쓰지 않음.

### 16. On the Failure of the Bootstrap for Matching Estimators
Alberto Abadie and Guido W. Imbens (2008). Econometrica.

- 논문: https://doi.org/10.3982/ECTA6474
- 공개본·원문 저장소 또는 이전 작업본: https://www.nber.org/papers/t0325
- 우리 적용과 제한: 일반적인 nearest-neighbor matching 추정량의 bootstrap 실패. 현재 경기 내부 매칭의 경기별 대비를 자동 무효화하지 않으며 대상 추정량과 재표집 단위를 따로 따져야 함.

### 17. Extreme-Path Benchmarks for Sequential Probability Forecasts
Jonathan Pipping-Gam{\'o}n and Abraham J. Wyner (2026). arXiv preprint.

- 논문: https://arxiv.org/abs/2601.18774v3
- 공개본·원문 저장소 또는 이전 작업본: https://arxiv.org/html/2601.18774v3
- 우리 적용과 제한: 승률 경로의 최고점·최종 패배 등 극값의 기준분포. v3 프리프린트, Annals of Applied Statistics에 투고했다고 표기. 채택·정식출판으로 인용하지 않음.

## 사용 원칙

현재 동결된 V/q 결과를 바꾸지 않고 인용·검증을 먼저 보강한다. 새로운 필터·보정·결합 모델을 개발하면 별도 버전으로 TRAIN/개발자료에서 선택한다. TEST 진단곡선을 원래 동결 평가기의 out-of-sample 성능 향상으로 제시하지 않는다. 매칭·교전 선택은 무작위 처치가 아니며, ΔV 또는 배경조정 ΔV는 자동으로 교전의 인과효과가 되지 않는다.
