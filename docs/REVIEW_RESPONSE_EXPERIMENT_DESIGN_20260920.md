# 리뷰 대응 실험 설계 — SVI 방향 예측의 추가 정보와 측정 근거

- 작성일: 2026-09-20
- 기준 저장소: Lunecid/LOL_teamfight_Lab
- 기준 브랜치/커밋: feature/fight-boundary-pipeline / b6d0c769ac30bba31b5d86fa5f7711462e7fa12e
- 상태: **실행 제안. 새 분석의 완료·성능·통계적 유의성을 보고하는 문서가 아니다.**
- 근거: 사용자가 제공한 Francisco A. B. Sampaio의 이메일, CoG 원문, 첨부 9월 19일 설계 문서, 위 커밋의 q 계약·측정표·결과표·구현.
- 새로운 수치 규칙은 모두 이 문서의 **제안값**이다. 리뷰어의 요구값이나 이미 검증된 최적값이 아니다.
- 기존 TEST/외부 결과를 이미 확인했다. 이 후속 계획을 새로 기록해도 과거 노출이 사라지지 않는다. 전체 묶음은 탐색적 리뷰 대응 분석이다.

## 1. 목적 → 개선 → 예측 → 검증

**목적:** 한타 전 공개정보로, 정의한 교전 전후 구간에서 모델 추정 경기 승률이 어느 방향으로 움직일지 예측한다.

**개선:** CoG의 교전 이득 예측을 계승하되, 고정 교환 계수 대신 경기 승패로 학습한 V를 통해 상태 변화를 평가한다. 기존 CoG가 킬 차이만 사용한 연구였다고 쓰지 않는다. [S1, S2]

**예측:** q(X_pre)=P(ΔV>0|X_pre). 정확한 변화량이나 종료 승률을 예측하는 주 과제로 바꾸지 않는다.

**검증:**
- 축 A — 초기 승률과 시간을 넘어서는 추가 예측 정보인가? (리뷰 1·2)
- 축 B — ΔV는 배경 변화·이진화·가치모형 구성에 얼마나 의존하는가? (리뷰 3·4·6)
- 축 C — 관측된 물질적 결과 및 V 구간 밖 결과와 어떻게 대응하는가? (리뷰 5)

PT/B40는 핵심 검증 수단이며 연구 목적 자체가 아니다. 좋은 결과가 나와야 완료되는 설계가 아니라, 각 우려에 답할 수 있어야 완료되는 설계다.

## 2. 유지할 것과 변경할 것

### 2.1 유지

| 대상 | 고정 규칙 |
|---|---|
| 최종 V | A_MLP_expanded fit85의 전처리·가중치·보정기를 포함한 동일 번들 |
| V 입력 | Expanded361, 수치 351 + 챔피언 범주 10; 임베딩 후 MLP 입력 431 |
| 주 라벨 | Y_SVI = 1[ΔV>0], ΔV = V(post)-V(pre) |
| 주 사례 | 기존 T, 동일 onset/cutoff, h90 종료 규칙 |
| 정확한 0 | 유효한 ΔV=0은 Y=0이지만 별도 동률 플래그 유지; 하락과 구분 |
| 결측 | Y=-1 등 명시적 무효. 0/레드 방향으로 강제 변환 금지 |
| 주 q | 기존 logit_state의 원모델 가중치와 입력 정의를 고정 |
| TRAIN | 현재 경기-fold OOF V로 만든 p_pre·ΔV·Y를 재사용. 재생성 시 같은 제외-fold 평가기로 일괄 생성 |
| Q_CAL/Q_SELECT/TEST | 기존 역할별 자료와 동결 fit85 라벨을 유지 |
| 자료 역할 | 15.14 TRAIN, 15.15 개발 역할, 15.16 평가, 16.x score-only 외부 평가 |
| 미래 정보 | 실제 endpoint·실제 구간 길이·사후 참여자·post 상태는 q/기준선의 입력으로 금지 |

h90는 모든 구간의 실제 길이가 90초라는 뜻이 아니다. 실험에서는 저장된 pre 질의와 실제 endpoint의 차이를 사용한다.

### 2.2 이번에 추가

1. 초기 승률·시간의 유연한 기준선과 동일 정책의 확률 보정 비교.
2. 전체와 B40의 병렬 보고, 좁은 승률 구간별 직접 비교와 불확실성.
3. 새 V로 다시 계산한 동일 길이 무킬 참고 구간.
4. 작은 변화의 사후적 민감도 분석. 주 라벨·주 표본은 유지.
5. 기존 물질적 대응·종료점·V 사양 분석 보강 및 시간별 확률 보정.
6. 제한된 V-외부 결과 대응 분석.

**하지 않을 것:** V 재탐색, q 아키텍처 순위 대회 재개, B40에서 이기도록 임계값 튜닝, 새 결과가 좋을 때만 주 분석으로 승격, 기존 old_V 결과의 숫자 재사용.

## 3. 현재 근거와 아직 없는 근거

아래는 기준 커밋에서 보고·확인된 내용이다. 로컬 NPZ/joblib/results.json을 재실행해 확인한 것은 아니다.

| 항목 | 현 상태 | 이번 조치 |
|---|---|---|
| q 전체 비교 | logit_state AUC .6403; PT .6164; ΔBrier -.00422, CI [-.00516,-.00325] 보고 [S4] | 보존. 현재 PT는 선형 기준선이라고 명시 |
| B40 | q Brier .2474, PT .2497; q AUC .5669, PT .5209 [S4] | 이미 계산한 CI가 있으면 연결; 없으면 고정 예측에서 재집계 |
| PT 구현 | p_pre와 time 두 입력의 Logistic [S5] | PT_linear로 이름 구분. PT_flex 추가 |
| q 보정 | 현재 주 코드에는 별도 q calibrator 없음. LGBM은 Q_CAL을 early stop에 사용 [S5] | 기존 실행은 raw 확률 실행으로 보존; 선택된 logit 원모델만 고정하여 별도 보정 분석 |
| 새 ΔV 분포·물질적 대응 | 작성됨 [S3] | 분모·가중치·버전 검증 후 재사용 |
| h60/h90/h120, V 사양 비교 | 작성됨 [S3] | 공통 사례에서 불일치의 크기·상태별 분포 추가 |
| 새 V Quiet | Deferred [S3] | 신규 수행. old_V quiet 비율 재사용 금지 |
| V 시간대별 성능 | AUC/Brier/logloss 보고 | reliability·기울기/절편은 별도 진단으로 연결 |
| 외부 | 주요 16.13 집단에서 q−PT 악화 보고 | 숨기지 않음. frozen V→W와 q→SVI 진단을 분리 |

## 4. 실행 공통 규칙 — RR0

### 4.1 출처·재현

- source_commit, input_impl, V_bundle_sha, q_base_sha, calibration_sha, label_sha, feature_schema_sha, case_key_hash를 기록한다.
- V용 g_V와 q/기준선용 g_q를 구분한다. 이번 보정 비교에서 g_V는 바꾸지 않는다.
- 원래 q, PT_linear, 원래 성능표는 읽기 전용 보존한다. 새 분석은 별도 디렉터리와 ID를 사용한다.
- 저장된 예측을 재집계할 때 대상 match/s 키의 1:1 일치를 검사한다. 누락·중복을 조용히 버리지 않는다.
- 새 calibrator와 baseline을 학습하는 것은 별도 리뷰 대응 버전이다. 원래 q가 처음부터 이 조건에서 선택됐다고 쓰지 않는다.

### 4.2 경기 가중과 신뢰구간

집단 G 안에서 경기 m의 사례 수를 n_mG라 하고 w_iG=1/n_mG로 둔다.

Brier_G(q) = Σ_i∈G w_iG (y_i-q_i)^2 / Σ_i∈G w_iG.

주 차이 D_G = Brier_G(q)-Brier_G(PT_flex). 음수가 개선이다.

- 각 평가 셀에서 경기 가중을 다시 계산한다. 전체 가중을 단순히 마스킹한 값과 섞지 않는다.
- 점추정은 원자료에서 직접 계산한다. 부트스트랩 복제 평균으로 바꾸지 않는다.
- 새 집계는 경기 단위 paired bootstrap 2,000회, seed=7을 제안한다. 같은 재표집 경기로 모든 모델·집단 대비를 계산한다.
- AUC·logloss·reliability에도 같은 평가 측도를 적용한다. AUC는 클래스가 하나면 NA로 보고한다.
- 95% 구간은 고정된 평가기·라벨·선택 아래의 경기 표본 변동이다. 모델 적합·라벨모형의 전체 불확실성을 포함한다고 쓰지 않는다.
- 여러 좁은 구간의 pointwise CI를 독립적인 확증 검정 목록처럼 읽지 않는다. 이들 결과는 진단이다.

### 4.3 평가의 우선순위

- 기존 전체 q−PT_linear 결과는 원래 결과로 유지.
- **새 리뷰 대응 핵심 대비:** B40의 q_RR−PT_flex_RR ΔBrier.
- 전체 T의 동일 대비를 반드시 나란히 보고. B45·좁은 승률 구간·시간별은 보조 진단.
- 기존 τ=.001은 있던 해석 기준으로 표시할 수 있으나 실용가치가 검증된 보편 문턱이 아니다.
- 이 새 우선순위는 이미 TEST를 본 뒤의 리뷰 대응 계획이다. 소급적 사전등록이나 untouched validation으로 표현하지 않는다.

## 5. RR1 — 초기 승률·시간 이상의 추가 예측 정보 (리뷰 1)

### 질문

현재 q의 이득이 단순한 초기 승률·시간 요약의 부족한 함수형을 보완한 것인가, 그 요약 밖의 정보도 도움이 되는가?

### 비교 대상

| ID | 입력 | 형태 |
|---|---|---|
| constant | 없음 | OOF TRAIN의 경기 가중 양성률 |
| b_linear | p_pre | 기존 1변수 Logistic |
| b_spline | p_pre | cubic spline + L2 Logistic |
| PT_linear | p_pre, time | 기존 2변수 Logistic |
| PT_flex | p_pre, time | 각각의 spline 주효과 + tensor-product 상호작용 + L2 Logistic |
| q_base | 기존 352개 사전 수치 입력 | 동결 logit_state 원확률 |
| q_RR | q_base의 점수만 | 개발자료에서 선택한 identity 또는 하나의 증가 sigmoid 보정 |

PT_flex = sigmoid(a + f_p(p) + f_t(t) + f_pt(p,t)).

**제안 예산:** degree=3, p knots {4,6}, time knots 4, include_bias=False, C {.01,.1,1}. TRAIN OOF p/time의 경기 가중 분위수로 knots를 적합. 시간 범위 밖 extrapolation은 constant로 명시하고 발생률을 보고. 특성생성과 가중치 평균-1 정규화 규칙을 저장한다. 이 설정은 앞으로의 제한된 실험안이지 검증된 최적값이 아니다. [M1]

**Shipped RR1 (see addendum):** 실제 실행은 `SplineTransformer` **uniform** knots, fit weight \(1/n_m\) (**mean-1 미적용**), 선정은 **2단계**(raw로 knots/\(C\) 선택 → 그 승자에 identity vs sigmoid). 공동 `설정×보정` 전수 비교는 하지 않았다. 원고는 shipped 절차로만 기술한다 → [REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md](REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md).

### 적합·보정·선정

1. 새 b_spline/PT_flex는 기존 OOF TRAIN 라벨과 p_pre로 적합.
2. TRAIN 외부 자료로 knots/scaler/정규화 파라미터를 추정하지 않음.
3. 원모델은 고정. Q_CAL에서 각 예측의 증가 sigmoid 후보를 적합한다. identity(raw)도 보존한다.
4. **제안:** Q_SELECT 전체 T에서 baseline 설정 **및** identity/sigmoid를 함께 선택. **Shipped:** raw Brier로 설정 선택 후, 그 설정에 대해 identity/sigmoid만 비교 (addendum). B40 점수로 별도 선택하지 않음.
5. logit_state 자체의 특징·가중치는 다시 적합하지 않는다. q_RR은 이 원모델에 덧붙인 보정만 다를 수 있는 별도 평가기이다.
6. 모든 규칙/보정/설정을 저장한 뒤 15.16 재집계. q_base 원결과와 q_RR 새결과를 모두 보존한다.

### 보고

전체 T와 B40에서 모델별 Brier/logloss/AUC/n/n_matches/양성률, q−PT_flex 및 q−b_spline의 paired CI. raw 예측 비교와 보정 포함 비교를 구분해, 이득이 보정 차이에 의존하는지도 보여준다.

### 해석

유연한 기준선 이후에도 차이가 남으면 '시험한 승률·시간 기준선 이상의 추가 예측 정보'라고 쓴다. 초기 상태를 완벽히 통제했다거나 전술 메커니즘을 발견했다고 쓰지 않는다. 차이가 줄거나 사라져도 결과이며, 더 약한 기준선을 주 결과로 골라 되돌리지 않는다.

Offset/init_score는 필수 아님. 수행한다면 match-OOF로 만든 SVI baseline의 logit을 offset으로 사용해야 하고, 최종 경기 승률 p_pre 자체를 SVI 확률로 간주하지 않는다. Offset도 초기 상태 이용을 기계적으로 금지하는 장치는 아니다.

## 6. RR2 — 균형 상태와 좁은 승률 구간의 비교 (리뷰 1·2)

### 집단

- 전체 T와 B40(.40≤p≤.60): 본문 주 표의 병렬 패널.
- B45(.45≤p≤.55): 보조.
- 기본 좁은 구간: [0,.05), [.05,.10), …, [.95,1]. 정확히 .5는 [.5,.55)에 배치한다. B40/B45 경계 규칙은 기존 계약 유지.
- 각 구간의 경기 수·사례 수·양성/음성 수·양성률 공개. 제안 경고: 200경기 미만 또는 한 클래스 30사례 미만이면 low-support 표시; 유리한 결과를 위해 bin을 합치지 않는다.
- 시간대는 [2,10), [10,20), [20,30), [30,∞). 기존 표기 0–10이라도 pre-2 제외를 주석으로 적는다.

### 계산

모델을 구간마다 재학습하지 않고 RR1의 고정 예측으로 각 구간에서 직접 q/PT 성능 비교. p 범위가 좁다고 baseline AUC를 자동으로 .5로 가정하지 않는다.

D_B40와 D_outside를 각각 계산하고 H=D_B40−D_outside의 paired 경기 bootstrap을 함께 산출한다. 전체 유의/B40 비유의 차이를 곧 효과 이질성이라고 쓰지 않는다.

### 산출

- 표 T1: 전체·B40 핵심 대비.
- 그림 F1: p bin별 양성률, 평균 signed ΔV, q−PT_flex 손실차이(각 통계를 분리).
- 보조표 S1: B45, 좁은 bin·시간별 원 지표/분모/CI.

B40 CI가 0을 포함하면 '추가 이득을 명확히 확인하지 못했다'고 쓴다. 실질적으로 중요한 개선도 배제하는지는 구간 폭을 따로 해석한다. 데이터에 정보가 본질적으로 없다는 결론으로 바꾸지 않는다.

## 7. RR3 — 새 V 기반 무킬 참고 구간과 배경 변화 (리뷰 3)

### 최소 필수 작업: 같은 길이의 관측 비교

- 각 교전 i의 실제 길이 L_i = endpoint_i − pre_query_i를 사용. 이름 h90를 90초 상수로 바꾸지 않는다.
- quiet 후보는 같은 역할·같은 경기의 적격 질의에서 시작하며 pre부터 L_i 동안 CHAMPION_KILL이 없어야 한다. 시작 상태에 이미 반영된 시점의 사건을 중복 제거하지 않도록 구간 convention을 상태빌더와 일치시킨다.
- 원래 교전 pre/end 구간과 겹치는 후보, terminal/관측 지원 범위를 넘는 후보 제외.
- 오브젝트 사건·파밍·성장은 허용한다. 그런 변화까지 제거하면 리뷰가 묻는 배경 변화를 축소하게 된다.
- 같은 동결 V/상태빌더/g로 시작·끝을 평가한다. 과거 이력만으로 새 현재 상태를 대체하지 않는다.

### 제안 매칭 규칙

| 항목 | 제안값/규칙 |
|---|---|
| 주 매칭 | 동일 경기, 1:1, 사용한 quiet 구간 재사용 금지 |
| 시작 시간 차이 | ≤120초 |
| 초기 V 차이 | ≤.025 |
| 길이 | 실제 L_i와 동일 |
| 후보 결정 | 표준화 거리 (Δp/.025)^2+(Δt/120)^2 최소, 동률은 고정 해시 |
| 제외 대상 | interval 결과/ΔV/킬 결과를 보고 유리한 후보 선택 금지 |
| 반복·겹침 | 선택된 quiet 구간끼리 중복·겹침을 관리하고 경기 단위 상관을 추론에 반영 |

위 수치는 제안이다. 개발자료의 매칭 가능성과 공변량 균형만 확인하여 동결하고, TEST ΔV 차이를 키우도록 변경하지 않는다.

같은 경기에서 충분한 매칭이 불가능하면 이를 결과로 보고한다. 필요시 **별도 보조 분석**으로 같은 역할/패치의 다른 경기 매칭을 미리 정의한다. 이 경우 같은 경기 설계와 섞지 말고 집단 차이 및 적절한 의존성 추론을 다시 명시한다. 주 분석 실패를 숨기기 위한 자동 fallback은 금지.

### 보고할 비교

- 매칭 성공률, matched/unmatched 교전의 p/time/길이 분포.
- 매칭 후 p/time/길이와 snapshot_age·frame-refresh 횟수의 균형.
- 같은 matched 교전만 사용한 E[ΔV], P(ΔV>0), mean/median/분위수 |ΔV|와 분포.
- pairwise signed 차이와 absolute 차이를 경기 단위로 집계하여 CI.
- frame 갱신 동일 조건의 층화 진단. 사후 갱신으로 엄격히 제한한 분석은 별도 하위분석으로 표시.

'교전 |ΔV|가 크다'는 결과만으로 방향 예측 가능성이나 인과효과를 증명하지 않는다. 반대로 차이가 작아도 원래 결과를 숨기지 않는다.

### 권장 보조: 기대 drift와 state volatility

개발용 Q_CAL의 quiet 자료만으로, 고정 fit85 V 아래에서 z=(p_pre,time,L)의 조건부 평균 μ_Q(z)와 변동 규모 s_Q(z)를 추정한다. 튜닝이 필요하면 그 자료 안의 경기-fold로 수행; TEST/외부를 학습에 넣지 않음. 기술통계를 위해 TEST quiet의 분포를 계산하는 것은 가능하지만, 이를 TEST의 'out-of-sample drift model'이라고 부르지 않는다.

제안 구현: fixed p(.1 폭)×time(4개)×duration(≤30,30–60,60–90,>90초) 셀의 경기 가중 평균과 centered absolute change의 75분위수. 지원 경기 <100이면 duration→time 순으로 합치는 사전 fallback을 기록. epsilon=1e-6. 이는 조밀한 conditional oracle이 아니라 참고 추정량이다.

ΔV_adj = ΔV − μ_Q(z).

원본과 조정 변화의 분포·부호 변화만 우선 비교한다. **q_base를 그대로 Y_adj에 적용한 점수를 새 목표의 주 예측 성능처럼 부르지 않는다.** Y_adj 예측이 필요하면 별도 표적·적합·보정 버전이 필요하다.

L은 사후 길이이므로 drift/측정 진단에만 사용. q나 RR1 baseline 입력에 주입 금지. quiet가 무작위가 아니므로 ΔV_adj도 교전의 인과적 기여가 아니다.

## 8. RR4 — 이진 목표 유지의 근거와 작은 변화 민감도 (리뷰 4)

### 유지할 선택

주 목표는 방향이다. continuous ΔV/상태별 중앙값 목표는 주 과제로 바꾸지 않는다. 원문 리뷰는 재검토 제안이며, 모든 대안을 수행해야 한다는 의미로 해석하지 않는다.

### 필수 진단

1. RR2의 P(Y=1|p)와 E[ΔV|p]를 나란히 제시해 빈도와 크기를 구분.
2. fixed q_RR/PT_flex_RR 예측을 같은 사례에 보관.
3. RR3의 개발자료에서 추정한 s_Q(z)에 비해 |ΔV|가 작은 사례를 제외하는 **사후 민감도**를 수행.
4. 제안 λ∈{0,.25,.5,1}; λ=0은 모든 적격 사례를 의미. λ>0은 |ΔV|≥λ max(s_Q(z),1e-6).
5. 각 λ의 coverage, n/matches, 시간·p·양성률 변화, AUC/Brier/ΔBrier/CI 보고. TEST 점수가 가장 좋은 λ를 주 결과로 선택하지 않음.

이 필터는 post 값으로 사례를 선택하므로 실시간 예측 대상을 선택하는 규칙이나 배포용 abstention 규칙이 아니다. 쉬운 사례만 남을 수 있다. 제외한 사례 수와 집단 변화가 핵심 결과다.

quiet 규모 추정이 지원 부족이면 미완료로 표시하고, 미리 고정한 절대 cutoff {0,.005,.01,.02}를 **다른 정의의 보조 분석**으로만 사용한다. 이를 상태별 volatility 보정과 동일하게 부르지 않는다.

### 선택 부록 — 요구 시에만 확대

- 원 ΔV 회귀: 0 예측, 개발자료에서 적합한 μ_eng(p,time), 고정 정보집합의 ridge 회귀를 MAE/MSE로 비교. 동일 TRAIN OOF continuous target과 역할별 ΔV 사용.
- 상태 상대 중앙값 또는 drift-adjusted binary는 다른 표적이다. 실행한다면 별도 ID/라벨/적합 필요.
- 회귀의 양호한 성능을 기대하거나 martingale이 자동으로 성립한다고 가정하지 않는다.
- 이번 최소 리뷰 대응에는 회귀 모델군 확대나 주 라벨 교체를 요구하지 않는다.

## 9. RR5 — V와 독립적으로 정의된 결과와의 대응 (리뷰 5)

### RR5a: 같은 구간의 물질적 대응 — 재사용/보강

새 V에서 계산된 kill_diff, epic/structure/objective net, alive_diff_post를 사용. 분모·타이·가중치를 명시하고 정확한 2×2/다범주 건수를 제공한다. Weighted agreement와 unweighted disagreement를 여집합처럼 섞지 않는다.

동일 구간의 킬·오브젝트·생존 정보가 V의 특징과 겹칠 수 있으므로 independent ground-truth accuracy가 아니라 correspondence라고 쓴다. alive가 last-frame/추정 값이면 그 관측 정의와 freshness도 명시한다.

### RR5b: 종료점 이후 결과 — 제한된 신규 확인

- 이벤트 목록: ELITE_MONSTER_KILL 중 프로젝트 schema로 식별 가능한 주요 오브젝트(예: dragon/baron/herald 등)의 명시적 enum 목록을 DEV에서 동결. gold 가중치를 새로 붙이지 않음.
- 관측 창: (endpoint, endpoint+180초], **180초는 제안값**.
- 라벨 정의/추출 함수는 V의 확률이나 q 점수를 읽지 않는다.
- 첫 적격 오브젝트 획득 팀을 Blue/Red로 기록. 같은 시각 양팀 획득은 tie/ambiguous로 별도 표기.
- 관측 창 전체에 오브젝트가 없으면 none; 그 전에 경기가 끝나면 game_ended_before_objective; 원자료가 끊기면 observation_censored로 구분. no event를 Red로 변환하지 않는다.
- SVI 방향 또는 ΔV 구간별 후속 결과 분포를 전체·B40에서 보고. 시간·초기 우세별 표도 함께 두면 기저 상태 연관을 확인할 수 있다.

이 결과는 V의 종료상태 입력에 직접 들어가지 않는 외부 정의라는 장점이 있지만, 경기 상태와 공통 원인을 공유하며 초점 교전의 결과라는 인과 보증은 없다. 새 목적을 '다음 오브젝트 예측'으로 바꾸지 않고 수렴적 대응 근거로만 사용한다.

## 10. RR6 — V/endpoint 안정성과 시간별 calibration (리뷰 6)

### RR6a: V의 실제 W에 대한 보정

- 같은 fit85 평가기 번들. 새로운 per-band V나 보정기를 만들지 않음.
- timeline 질의: [2,10), [10,20), [20,30), [30,∞).
- 실제 교전 pre/end 질의: 별도 표. post 성능은 실제 post 시각으로 시간대를 자르며, 같은 교전의 pre→post 비교는 별도의 paired/pre-band 표로 명시.
- 각 셀에서 n/matches, Brier/logloss/AUC, 10개 고정 확률 bin의 weighted reliability, ECE, calibration intercept/slope와 경기 bootstrap CI.
- reliability에는 예측 평균과 실제 W 빈도, 각 bin의 표본 수를 함께 표시. ECE 단독 합격 문턱은 두지 않음.
- 진단식 logit P(W=1)=a+b logit(p)의 (a,b)는 결과 설명용 적합치이며 TEST 예측을 재보정하는 데 쓰지 않음.
- q의 calibration은 실제 W가 아니라 SVI에 대한 별도 표로 명시. 둘을 같은 calibration으로 합치지 않음. [M2]

### RR6b: 라벨 안정성

- h60/h90/h120: 공통 유효 사례와 각 정의의 포함/제외 사례 수를 따로 보고. 실제 endpoint가 동일한 비율과 길이 차이도 보고해 높은 일치율의 이유를 구분.
- primary MLP vs corrected Logistic: 같은 사례·시점의 ΔV와 부호. 기존 A0는 protocol이 다른 legacy 비교로 구분.
- flip 비율뿐 아니라 |ΔV|, p/time, 관측 갱신별 불일치, ΔV의 상관과 절대차이를 제시.
- V가 바뀌는 비교에서는 B40를 primary V로 고정한 paired 분석을 주로 하고, 각 V 자체 B40의 사례 이동은 별도 구성 분석으로 보고.
- optional no-objective V: 동일 적합 범위/MLP 설정의 추가 사양으로만 시행. 구 V에서 나온 5.4%를 새 V의 수치로 재사용하지 않음.
- 기존 h90 q를 alternate target에 적용하면 fixed-predictor label perturbation test일 뿐, 그 새 목표를 적합한 모델 성능이 아니다. core에서는 라벨 안정성에 집중.

예측력이 높은 V가 올바른 교전 라벨을 보증한다거나 모든 적절한 V가 같은 부호를 내야 한다고 가정하지 않는다.

## 11. RRX — 외부 전이 해석 (기존 결과 보존, 리뷰 밖의 일반화 보조)

주요 KR/NA1 16.13의 이득 소실을 반드시 보고한다. 리뷰 대응이 잘 되었다는 이유로 외부 결과를 생략하지 않는다.

- same frozen V→실제 W, q→SVI를 각각 시간대별로 평가.
- 현재 외부 코드의 T 키·feature order·유효값 검사에 fail-closed를 적용; cohort 파일이 없으면 모든 교전을 T로 간주하지 않음.
- baseline constant는 MAIN TRAIN의 저장된 경기 가중 양성률을 재사용. EXT 결과로 새 prior를 적합하지 않음.
- 새 q_RR/PT_flex_RR는 개발자료에서만 선택/보정하고 외부에 score-only 적용.
- 외부 CI를 포함하되 작은 표본/pilot을 큰 집단과 합쳐 성공으로 요약하지 않음.
- 외부 성능 악화를 보정 문제라고 가정하지 않음. V 품질 저하·q의 상태관계 변화·입력 지원범위 등의 진단을 분리.

## 12. 실행 순서와 최소 산출물

| 순서 | 실행 | 재학습 여부 | 최소 산출물 |
|---|---|---|---|
| 0 | RR0 버전·행·분할·가중 계약 | 없음 | response_manifest.json, case_inventory.csv |
| 1A | RR1 유연한 baseline 및 q 보정 확인 | 저차원 baseline/보정만 | baseline_selection.json, prediction_table.csv |
| 1B | RR6 보정·안정성 기존 결과 보강 | V/q 원모델 재학습 없음 | calibration_by_time.csv, stability_table.csv |
| 2 | RR2 전체/B40/좁은 구간 비교 | 없음 | lift_by_state.csv, paired_ci.json |
| 3 | RR3 새 V quiet 비교 | 상태 점수 계산; 선택적으로 μ/s 적합 | quiet_pairs.csv, match_balance.csv, quiet_contrast.csv |
| 4 | RR4 작은 변화 민감도 | 없음 | margin_sensitivity.csv |
| 5 | RR5a 물질적 대응 검토 + RR5b 후속 오브젝트 | 없음 | material_crosstab.csv, next_objective.csv |
| 6 | RRX 보존된 외부 적용·진단 | 없음 | transfer_dual_stage.csv |
| 7 | 리뷰 항목별 답변 작성 | 없음 | RESPONSE_EVIDENCE_MATRIX.md |

RR1/2는 예측 검증, RR3/4/5/6은 측정 검증으로 병렬 진행할 수 있다. RR4의 state-volatility 버전만 RR3의 μ/s를 의존한다. 저널 J-RQ1↔석사 M-RQ2에 측정, 저널 J-RQ2/3↔석사 M-RQ3/4에 예측 검증을 공유한다. RQ 번호를 리뷰 번호에 맞춰 재편하지 않는다.

## 13. 원고·답변에서 사용할 핵심 결과물

- T1: 전체/B40 — q, PT_linear, PT_flex, b_spline의 지표와 차이/CI.
- T2: 매칭된 교전과 quiet — signed/absolute 변화, matched coverage 및 균형.
- T3: 물질적 결과·후속 오브젝트 대응 — 타이와 미관측 포함.
- T4: 시간별 V 보정 및 endpoint/V 사양 안정성.
- F1: 좁은 p 구간별 추가 예측 이득; 원 양성률/평균 변화는 다른 패널/그림으로 구분.
- F2: quiet/교전 ΔV 분포와 작은 변화 제외 시 coverage−ΔBrier의 동반 보고.
- 캔들 예시는 부록/설명용. 실제 pre/post와 사전 q 표식을 분리하고 사전 추출 규칙으로 사례를 고른다. 전체 성능을 캔들 몇 개로 증명하지 않는다.

## 14. 결과별 허용 결론과 종료 조건

| 결과 | 허용 결론 | 금지 결론 |
|---|---|---|
| flexible PT를 넘어선 이득 | 시험한 p/time 요약 이상의 정보가 유용 | 모든 초기 우세를 제거; 전술 메커니즘 학습 증명 |
| B40 CI가 0 포함 | 이 표본/정밀도에서 이득 미확정 | 정보가 전혀 없음; 연구 실패 |
| quiet 대비 큰 변화 | 유사한 무킬 구간보다 큰 V 변화 관측 | 교전의 순수 인과효과 |
| 작은 변화 제외 후 개선 | 사후 선택집단의 예측이 다름 | 배포 시 그 사례를 미리 선택 가능 |
| material와 높은 일치 | 정의 간 대응 | 외부 한타 정답 정확도 |
| horizon은 안정, V는 일부 불일치 | 어떤 선택에 민감한지 한정 | 라벨 noise 완전 제거 |
| 전이에서 이득 소실 | 적용 범위의 제한 | 작은 pilot만 근거로 전이 성공 |

**종료 조건:** 모든 검증이 긍정적일 것, B40에서 반드시 유의할 것, V를 더 개선할 것이 아니다. 각 리뷰 항목에 대해 (a) 수행한 분석과 근거, (b) 미채택 대안과 이유, (c) 남는 한계를 구분해 답할 수 있으면 이 리뷰 대응 묶음을 마감한다.

## 15. 출처와 제안의 경계

[S1] 제공된 CoG 원문: Kill-Conditioned Engagement Outcome Prediction in League of Legends Under Minute-Resolution Public Telemetry. 목적·교환 가치·공개 관측 범위의 계승 근거.

[S2] 제공된 STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md, PAPER_COHORT_CONTRACT_20260919.md, SVI_MODEL_RESELECTION_TRANSFER_20260919.md. 주 목표와 코퍼스 역할의 계승 근거. 이전 표본/old_V 수치·후보표를 현재 실행의 완료 근거로 사용하지 않음.

[S3] https://github.com/Lunecid/LOL_teamfight_Lab/blob/b6d0c769ac30bba31b5d86fa5f7711462e7fa12e/docs/NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md

[S4] https://github.com/Lunecid/LOL_teamfight_Lab/blob/b6d0c769ac30bba31b5d86fa5f7711462e7fa12e/docs/Q_NEWV_FIT85_PRIMARY_20260920.md

[S5] https://github.com/Lunecid/LOL_teamfight_Lab/blob/b6d0c769ac30bba31b5d86fa5f7711462e7fa12e/scripts/rr20260920_q_newv_primary_fit.py

[S6] https://github.com/Lunecid/LOL_teamfight_Lab/blob/b6d0c769ac30bba31b5d86fa5f7711462e7fa12e/docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md

[M1] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html — spline 기저 생성·knots·경계 처리의 구현 근거. 이 문서의 후보 수·C·매칭 기준·bin·후속 180초를 정당화하는 선행 실험은 아님.

[M2] https://scikit-learn.org/1.8/modules/calibration.html — 확률손실과 보정의 구분, reliability의 의미. 이 문서의 구체적 실험 수치 규칙은 제안.

기준 자료에 구현 또는 결과가 없으면 미완료/추가안으로 표시했다. 본 설계는 새 실험을 실행하거나 저장소를 수정하지 않았다.
