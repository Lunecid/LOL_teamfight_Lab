# LoL 교전 가치·사전 예측 연구: 보완 실험 설계안

작성일: 2026-09-21  
검토 기준: `Lunecid/LOL_teamfight_Lab`, `feature/fight-boundary-pipeline`, `f80d780acec389620b0212488ea4200c63a4e710`  
저장소 경로: `docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md` (2026-09-21 Downloads 초안 반영)  
연계: [`master_thesis_v1/00_INVENTORY_AND_STATUS.md`](master_thesis_v1/00_INVENTORY_AND_STATUS.md) §4, [`thesis/docs/TASKS.md`](../thesis/docs/TASKS.md) §C  
상태: **실행 전 제안. 실제 자료 재분석·모델 적합·원격 저장소 변경을 수행한 결과가 아니다.**  
목표: 기존 journal freeze를 유지하고, 검증 가능한 주장과 필요한 보완 실험을 연결한다.

## 1. 보존할 주 연구

- 주 평가기: fit85 `A_MLP_expanded`, sha16 `ac459cc4397630a9`.
- 주 라벨: `Y_SVI = 1[p_post - p_pre > 0]`. Exact zero는 0; 결측 점수는 제외.
- 주 예측기: T 및 S의 기존 동결 `logit_state`; 주 기준선 `PT_flex` / `PT_flex_S`.
- 주 코호트: T `n_min >= 4`, S `n_min in {2,3}`; picks는 기존 주 분석에 추가하지 않는다.
- TRAIN 15.14: 경기 단위 OOF 평가기로 pre/post, 라벨, p_pre, B40를 함께 생성.
- 15.15: V_CAL/V_SELECT/Q_CAL/Q_SELECT 기존 경기 역할 유지.
- 15.16 TEST 및 기존 16.x 외부 자료: 이전 노출을 인정. 새 분석을 사전에 적어도 기존 자료가 새로운 확인적 TEST로 바뀌지 않는다.
- 새 실험은 별도 supplementary/thesis 경로에 저장한다. 유리한 결과만 기존 헤드라인과 교체하지 않는다.

## 2. 부족함의 종류와 필요한 대응

| ID | 의문 | 주 대응 | 위치 |
|---|---|---|---|
| G0 | 실제 실행 파일과 동결 계약이 일치하는가? | 무결성·시간 가용성·재현 검사 | 저널/학위 공통 우선 |
| E1 | 승률 요약 외 추가 이득은 어떤 직접 관측원에서 오는가? | 프레임×사건 2×2 정보군 비교 | 학위 핵심; 정보군 주장을 유지하는 저널에도 필요 |
| E2 | ΔV가 새 상태 관측보다 시간/정보 갱신에 의존하는가? | 동결 V의 기계적 시간경과 진단, 프레임 갱신 층화 | 측정 논리 보강 |
| E3 | 외부 T 실패/S 유지가 표본 변동을 넘는가? | 기존 고정 예측의 paired 경기 bootstrap | 외부 주장을 위한 우선 보강 |
| E4 | 평가기·endpoint·탐지 정의를 바꿔도 핵심 해석이 유지되는가? | 단계적 측정·사례 민감도 | 학위 M-RQ1/2; 저널 주장 범위에 따라 |
| E5 | SVI가 기존 사건 점수보다 더 유용한 지표인가? | 동일 사례 비교 및 독립적으로 기록된 후속 결과와의 조건부 대응 | 개선된 지표라는 강한 주장을 할 때 |
| E6 | 이전 TEST 노출 없이 재검증되는가? | 진짜 미사용 경기의 잠금 평가 | 확인적/강한 일반화 주장을 할 때 |

실험의 완료 조건은 유의한 개선이 아니라, 미리 정한 비교를 실행하고 결과에 맞는 주장 범위를 선택하는 것이다.

## 3. G0: 실행 무결성 검사

### 3.1 필수 검사

1. 코드 commit, import된 실제 모듈의 경로/내용 해시, 환경 버전을 기록한다. 별도 로컬 worktree 모듈을 조용히 우선 import하지 않는다.
2. 평가기 파일을 최종 저장한 뒤 SHA256을 계산한다. `n_eng_labeled` 등의 후속 수정 뒤에는 해시를 다시 계산한다.
3. S 실행에서 evaluator reuse를 강제하고, T 평가기 경로에 쓰기 작업이 발생하면 실패한다.
4. 결합 키 `(match_id, cutoff_ms)`의 중복/누락을 모두 집계한다. 예상하지 않은 누락은 실패시킨다. 정의를 바꾸는 E4는 다른 이벤트 기반 대응 키를 쓴다.
5. V 학습·early-stop에 held fold 경기가 섞이지 않았는지 **경기 집합 자체/해시**로 검사한다. 집합 크기 일치만으로 대체하지 않는다.
6. 피처 순서·정규화·UNK·보정기를 포함한 저장 번들의 reload parity를 검사한다.
7. 기존 논문 헤드라인을 기존 가중치와 동일 키로 다시 집계한다. 수치 오차 허용은 성능을 본 뒤 정하지 않는다.
8. 필수 데이터가 부족하면 hard failure 또는 명시적 incomplete 상태로 남긴다. 무결성 FAIL 데이터를 정상 경로의 최종 파일로 게시하지 않는다.

### 3.2 미래 정보 불변성 검사

이미 선택한 query 및 교전 메타데이터를 고정한 상태에서 query 이후 사건/프레임을 삭제·변형하여 **X_pre만** 다시 구성한다. X_pre가 바뀌면 입력 누수 경로를 조사한다. 미래 킬을 지웠을 때 탐지 집합이나 라벨까지 동일해야 한다고 요구하지 않는다. 그것들은 의도적으로 사후 정보를 읽는 별도 경로다.

경계 테스트: cutoff에 정확히 위치한 사건, ±1ms 사건, 같은 timestamp의 여러 사건, 프레임 경계, 종료 직전 query, 알 수 없는 챔피언 ID, 알 수 없는 목표물 팀.

### 3.3 판정

- 문서/해시 기록만 잘못됐고 예측이 동일: provenance 수정.
- 입력/라벨/행 집합/평가값 변경: 영향받는 결과만 새 버전으로 재산출하고 변경 장부에 원인과 영향 기록.
- 안전 검사를 건너뛴 채 새 실험 진행 금지.

## 4. E1: 프레임×사건 정보군 실험

### 4.1 질문과 자료

질문: 기존 p_pre를 공유한 뒤 프레임 특성 및 사건 특성을 직접 추가하는 효용이 있는가?

전체 V나 원천 자료를 제거하는 실험은 아니다. p_pre는 F/E 정보를 이미 압축하고 있으므로, 해석은 **기존 승률 요약에 조건부인 직접 특징 추가 효용**이다.

기존 문서 불일치: `MASTER_THESIS_RESEARCH_PLAN` §5는 LGBM을, `THESIS_V2_FOLLOWUPS` F6는 logistic을 적었다. 본 제안은 원 석사계획의 **LGBM 네 arm**을 주 모듈로 사용하고 logistic은 선택적 교차점검으로 분리한다. F1과 F2는 서로 포함되지 않으므로 일렬 단계 모형으로 부르지 않는다.

### 4.2 실제 열 기준 설계안

런타임 매니페스트와 원천 코드를 대조한 뒤 아래 목록을 고정한다. 명목 차원만 맞추는 임의 절삭은 금지한다.

- C: p_pre, time_minutes, time_minutes_sq: 3열.
- Q0: unknown_objective_team_count: 1열. 자료품질에 관한 관측 신호로, 모든 arm에서 공통 통제한다.
- F: 10명×9 프레임 특성: 90열.
- E: 플레이어 사건 70 + 팀 사건 94 + 해당 팀 사건의 시간 파생항 94 = 258열.
- 챔피언 ID는 현행 q의 원천정보에 맞춰 모두 제외한다.

| arm | 입력 | 예상 차원 |
|---|---|---:|
| M-F0 | C+Q0 | 4 |
| M-F1 | C+Q0+F | 94 |
| M-F2 | C+Q0+E | 262 |
| M-F3 | C+Q0+F+E | 352 |

Q0를 공통 통제로 놓는 것은 기존 포괄 계획을 구체화한 제안이다. 따라서 M-F0를 순수 p,t-only 또는 PT_flex와 동일하다고 쓰지 않는다. 엄격한 p,t 기준선 충분성을 별도로 확인할 때에만 C 3열 LGBM 한 arm을 추가하고 별도 대비로 표시한다.

사건 원천을 제거하면 event×time, 사건에서 만든 최근성, death_since_snapshot 등 파생열도 함께 제거한다. 두 원천을 모두 요구하는 혼합열이 추가로 발견되면 F/E만의 arm에는 넣지 않고 full-only로 분류하며 예상 차원을 수정한다. 현재 매니페스트에 없는 위치·스킬 준비 정보를 임의로 있다고 가정하지 않는다.

### 4.3 학습 설정 제안

- T/S를 각각 학습한다. 기존 Y, B40, 역할, 평가 키 고정.
- TRAIN에만 학습·전처리를 맞춘다.
- 모든 arm에서 경기 가중치를 사용한다. 적합 시에는 mean-one 정규화된 경기 가중치, 평가 시에는 셀 안에서 다시 계산한 1/n_m을 사용한다.
- 적합 가중치 정규화는 기존 q와 다를 수 있으므로 새 M-F 실행과 기존 q를 동일 실행이라고 부르지 않는다.
- LGBM 후보: num_leaves {15,31} × min_child_samples {100,300}; 총 4개/arm/코호트.
- 공통 제안: learning_rate .03, 최대 2000 trees, reg_lambda 5, colsample_bytree .8, subsample .8, subsample_freq 1, seed 7.
- Q_CAL: 경기 가중 logloss로 early stopping patience 50. 검증 sample weight도 전달한다.
- Q_SELECT: 경기 가중 Brier 최저 후보 선택; 동률 규칙 logloss → 단순한 후보 ID 순.
- 보정기: 모든 M-F arm에 identity를 고정. 확률보정 경쟁을 새로 추가하지 않는다.
- 이후 TEST와 외부를 한 번에 점수화한다. 주 q/PT 선택을 바꾸지 않는다.
- 최소 4 arms × 4 candidates × 2 cohorts = 32개의 소형 후보 적합. 이는 실행 수 제안이지 시간/성능 보장이 아니다.

### 4.4 주 대비와 해석

코호트 c에서 경기 가중 Brier L을 사용한다.

- 사건 추가 효용: D_E|F,c = L(M-F3,c) - L(M-F1,c).
- 프레임 추가 효용: D_F|E,c = L(M-F3,c) - L(M-F2,c).
- 두 코호트 합계 네 개를 한 추론 family로 등록한다.
- M-F1−M-F0, M-F2−M-F0, M-F3−M-F0는 보조 대비.
- 상호작용 L11−L10−L01+L00는 필요할 때만 보조 보고. 학습 손실의 비가산성이지 게임 기전·인과적 상호작용이 아니다.

한 원천의 추가 이득이 없으면 '선택한 학습기·예산·공유 승률 하에서는 명확한 추가 효용이 관측되지 않았다'고 쓴다. 정보 부재, 협동 무의미, 데이터의 예측 한계로 해석하지 않는다.

### 4.5 동일입력 학습기 질문의 처리

학위 M-RQ3에서 학습기 선택을 유지하려면 별도의 full 352열 Logistic과 LGBM을 비교한다. 현행 362열 LGBM 후보를 352열 logistic과 엄격한 matched-input 비교라고 쓰지 않는다.

최소 옵션: M-F3 LGBM을 사용하고 같은 TRAIN 전처리·역할·가중치·identity에서 352열 logistic을 C {0.01,0.1,1,10}로 Q_SELECT 선택한다. 이 제한된 사양 비교를 모든 아키텍처의 우열로 일반화하지 않는다. 공통 q를 재선정하는 실험이 아니다.

## 5. E2: 시간경과·관측 갱신 진단

### 5.1 질문

ΔV는 사건 결과뿐 아니라 query 시간, 사건×시간, 최근성·경과시간, 새 분 프레임의 유입에 따라 변할 수 있다. 본 실험은 이 성분을 진단하며, V를 재선정하지 않는다.

### 5.2 기계적 시간경과 probe

각 교전 i의 pre query t0, 실제 endpoint t1을 사용한다.

1. A_i = V(S_pre(t0)).
2. pre까지의 프레임/사건 원천만 유지하고 query를 t1로 진행시킨 진단용 상태 S_hold(t1)를 만든다.
3. time, time², time×event, death_age, 최근 30초 표시 등 시간의 모든 파생 경로를 원천 규칙에 맞춰 일관되게 재계산한다. pre 이후 관측은 읽지 않는다.
4. H_i = V(S_hold(t1)), P_i = V(S_post(t1)).
5. d_clock = H_i−A_i, d_update = P_i−H_i.
6. d_clock+d_update = P_i−A_i를 수치적으로 검사한다.

이 분해는 **선택한 갱신 순서의 정확한 산술 분해**다. S_hold는 실제 무교전 잠재결과가 아니다. 수동 골드·회복·행동을 정확히 시뮬레이션하는 것도 아니다. 오래된 프레임을 유지해 학습 분포 밖의 상태가 될 수 있으므로 지원 범위와 hold 경과시간을 함께 보고한다. 재구성 규칙을 확인할 수 없으면 시간열 두 개만 고쳐 실행하지 말고 해당 probe를 미완료로 남긴다.

### 5.3 보고할 값

- E|ΔV|, E|d_clock|, E|d_update|, 각 부호의 분포.
- Y_SVI와 1[d_update>0]의 불일치율; exact-zero 처리 별도.
- 변화량/기계적 잔차의 부호 일치 및 p_pre·게임시간·길이별 결과.
- d_clock/ΔV 비율은 ΔV≈0에서 발산하므로 주 지표로 사용하지 않는다.
- d_clock과 d_update의 반대 부호·상쇄 가능성을 명시한다.

### 5.4 실제 관측 갱신 층화

각 행에 pre_frame_ts, post_frame_ts, pre/post_frame_age, n_new_frames, 실제 followup 길이, 종료 사유를 붙인다. same-frame / new-frame와 pre_frame_age <30s / >=30s의 2×2 셀을 사용한다. 층 안에서 경기 가중치를 다시 계산한다.

각 셀에서 기존 q−PT ΔBrier, V_pre/V_post→W 품질, |ΔV|, 양성률, 표본 수를 보고한다. 시간·p_pre 분포도 함께 제시한다. 사후 endpoint나 frame-refresh 상태를 q의 사전 입력에 넣지 않는다. 셀 차이는 진단이며 갱신이 성능을 '유발했다'는 인과 설명이 아니다.

### 5.5 quiet와의 결합

기존 RR3를 다시 처음부터 만들지 않는다. 기존 matched subset에서 시작 frame age(예: 10초 caliper)와 새 프레임 수를 추가로 맞춘 강화 매칭을 보조로 수행할 수 있다. caliper는 결과를 보기 전에 고정하고 실패율/coverage를 모두 보고한다. coverage를 높이려고 결과 확인 후 기준을 완화하지 않는다. 기존 28% 범위를 전체 교전으로 일반화하지 않는다.

## 6. E3: 외부 고정 예측의 불확실성

대상: KR16.13 T/S, NA1 16.13 T/S. 작은 pilot 집단은 보조 보고만 한다.

각 경기 m, 코호트 c에서

 d_mc = mean_{i in(m,c)} [(q_i−y_i)^2 − (PT_i−y_i)^2]
 D_c = mean_m d_mc.

- 동일 경기의 전체 행을 하나의 cluster로 resample한다.
- 같은 bootstrap draw를 q/PT와 분석하려는 코호트들에 공유한다.
- 재학습하지 않는다. 보정기를 맞추지 않는다.
- 주 대비는 네 domain×cohort D_c.
- 10,000회 bootstrap, seed 7을 새 보완 규칙으로 제안한다. 기존 2,000회/95% 원장 수치를 덮어쓰지 않는다.
- 일반 95% percentile CI와 네 대비 family를 고려한 98.75% Bonferroni percentile CI를 나란히 제시한다. 이는 재표본추출 근사 CI이며 model-fit uncertainty를 포함하지 않는다.
- bootstrap에서 Δ>0 비율을 고전적 p-value라고 쓰지 않는다.
- CORP를 bootstrap으로 추론할 경우 각 draw에서 isotonic을 다시 적합한다. 고정 isotonic curve만 재사용한 결과를 전체 CORP 불확실성으로 부르지 않는다. 예산상 불필요하면 CORP는 기존 점추정 진단으로 남긴다.

선택적 T/S 차이: 두 코호트 모두 있는 경기에서 H=mean_m(d_mT−d_mS)를 계산한다. 이는 common-match population의 대비이고, 전체 코호트의 D_T−D_S와는 다른 대상이다. 그룹 표본과 대상변화를 따로 보고한다. 외부 T/S 기전 주장의 근거로 쓰지 않는다.

판정: CI 전체가 양수면 tested q의 손실이 더 크다는 증거, 음수면 반대. 0을 포함하면 불확실. 유의하지 않다는 사실을 동등하다는 뜻으로 쓰지 않는다.

## 7. E4: 평가기·구간·정의에 대한 민감도

### 7.1 먼저 기존 결과 재사용

RR6b에는 T의 h60/h90/h120 부호 일치와 동일 endpoint 비율, 일부 peer V 부호 일치가 이미 있다. 단순 전체 일치율 재실험은 새 기여가 적다. 우선 B40, 작은 |ΔV|, same/new-frame, 시간대, S를 보완한다.

### 7.2 라벨 이식 진단: 재학습 없음

- 기존 선택 이력이 있는 peer 평가기(예: 저장된 LR)를 한 개 추가 사용한다. TEST에서 잘 맞는 peer를 새로 고르지 않는다.
- 같은 pre/post 원천, 같은 교전 키, common-valid 표본에서 peer 라벨 Y_peer를 만든다.
- 기존 Y_main와의 flip, 특히 B40/작은 ΔV에서의 flip을 보고한다.
- q_main과 PT_main의 **기존 예측 자체를 고정**하고 Y_peer에 대한 ΔBrier도 산출한다. 이때 q 입력 p_pre를 peer 값으로 몰래 교체하지 않는다.
- 이 비교는 frozen predictions의 label-transfer 진단이다. peer evaluator로 처음부터 학습한 pipeline의 성능이 아니다.
- 전체 방법의 평가기 불변성을 주장할 때만 peer의 TRAIN OOF 라벨을 만들고 q/PT 둘 다 같은 역할에서 다시 학습하는 두 번째 실험을 추가한다. 원래 주 결과와 분리한다.

### 7.3 horizon이 아니라 stopping rule도 검사

h60/90/120은 같은 다음 킬에서 잘리기 때문에 endpoint가 자주 동일하다. 따라서 동일 endpoint 행과 다른 endpoint 행을 분리해서 읽는다.

하나의 대안 종점 제안: e_fixed = min(last_kill+30초, match_end−1ms), 다음 킬에 의한 조기 종료는 적용하지 않는다. 기존 상태 구성의 유효성 검사는 그대로 적용한다. 유효 공통 사례를 보고하고, 이 구간에는 추가 교전이 들어올 수 있어 attribution 의미가 넓어진다고 밝힌다. 실제 길이나 종료 사유를 사전 모델 입력으로 사용하지 않는다.

### 7.4 교전 정의 OAT

고정 20,000경기를 hash로 뽑되 patch 비율을 유지한다. 이는 계산 예산안이며 검정력이 보장된 표본 크기가 아니다.

reference G=13.7s, D=4264, R=1600, B=15s에 대해:

- G: 12.2 / 15.7s.
- D: 4000 / 4500u.
- R: 1400 / 1800u.
- B: 10 / 20s.

한 번에 하나만 변경하므로 기준 포함 9설정이다. G 대안은 기존 추정구간을 참고하고, D/R/B는 명시적 운영적 교란값이다. 모든 값을 통계적 CI 경계라고 부르지 않는다. D의 매우 좁은 표본추출 CI와 정의의 전체 불확실성을 혼동하지 않는다.

보고: 경기/교전 수, T↔S 이동, 포함/제외, 공통/추가/제거 kill clusters, pre/post frame, ΔV flip, 고정 q/PT 대비.

다른 정의에서 cutoff가 바뀌므로 `(match,s)`로만 같은 사건을 매칭하지 않는다. 원천 CHAMPION_KILL의 안정적 ID 집합(원천 event index와 timestamp/killer/victim 등으로 충돌 확인)을 사용한다. 중복/분할/병합 여부를 기록한다. 우선 동일 earliest-kill을 가진 1:1 공통 anchor 분석을 주 안정성 표로 사용하고, one-to-many는 별도 census에 남긴다. cohort가 바뀐 행은 새 집단 결과와 공통집단 결과를 구분한다.

### 7.5 비교 공정성의 필수 규칙

Frozen-score mode: V/q/PT 모두 고정. 변경되는 것은 사례·query·라벨뿐이다. 운용 정의 변화에 대한 기존 모델의 적용 민감도다.

Refit mode(선택): V만 고정하고 변경 정의에 맞춰 q와 PT를 **둘 다** TRAIN에서 재학습, Q_CAL/Q_SELECT만 사용하여 선정한다. 새 TRAIN 사례는 해당 경기의 기존 OOF V로 라벨을 만든다.

기존 후속 계획의 'q는 고정한 채 PT knots만 새로 선택'을 유일한 민감도 분석으로 쓰지 않는다. 적응 기회가 비대칭이다.

## 8. E5: 같은 사례에서 기존 지표와 비교 (강한 개선 주장에 조건부)

### 8.1 최소 대응 분석

동일한 교전 사례에서 SVI, kill-net, 기존 CoG operational label을 각각 계산한다. exact label 함수와 계수·attribution·window를 명시한다. 과거 다른 코퍼스의 숫자를 가져와 직접 비교하지 않는다.

- 예전 구간을 그대로 쓰는 operational continuity와, 구간을 맞춘 비교는 별도 표로 구분한다. 같은 endpoint로 다시 계산한 값을 원래 CoG 결과라고 쓰지 않는다.
- agreement뿐 아니라 no-decision/tie, SVI와 킬 차이의 불일치, 구조물/목표물 교환이 동반된 사례를 분모와 함께 보고한다.
- 불일치 사례는 결과를 보고 예쁜 사례를 고르지 말고 사전 정의한 셀(예: kill+ & SVI−, kill− & SVI+, kill=0 등) 안에서 hash순 표본을 고른다.
- 수치 추적: query → frame/event prefix → V_pre/V_post → SVI → q → 실제 기록된 후속 사건.

### 8.2 선택적 후속 결과 대응 모델

'기존 사건 점수보다 유용한 결과 지표'를 주장하려면 다음 보조 검증을 고려한다.

평가 시점은 pre가 아니라 engagement endpoint이다. 지표의 사후 활용을 검증하는 별도 모델 r이며 q의 사전 정확도로 보고하지 않는다.

목표 U: endpoint 뒤 180초 안의 첫 elite objective 결과를 Blue, Red, full-followup none, match-ended-before-objective, tied/ambiguous로 분리한다. 실제 관측 중단은 별도 결측 처리한다. 정확한 이벤트 타입/같은 timestamp 처리 규칙을 사전 고정한다.

- R0: 초기 p/time, endpoint time, interval length, 같은 구간 material nets.
- R1: R0 + ΔV_main. 또는 방향만 추가한 별도 사양을 사전 선택한다.
- 동일 경기에 동일 역할, 동일 저복잡도 다항 logistic과 예산 사용. TRAIN의 ΔV는 OOF, evaluation은 fit85.
- 주 비교: 경기 가중 multiclass Brier(sum over classes) 또는 사전 선택한 logloss.
- none/end를 Red로 처리하지 않는다. 결정된 Blue/Red 사례만 분석하면 selected-denominator 결과라고 표시한다.

중요: p_post = p_pre + ΔV이므로, p_pre와 ΔV를 추가한 모델은 post win-probability 정보 추가와 재매개화 관계다. 양성 결과가 SVI의 유일한 옳음이나 진짜 교전 가치, 인과효과를 증명하지 않는다. 이 실험은 기존 관측 지표를 넘어 후속 결과에 대응하는 추가 정보를 점검하는 것이다.

## 9. E6: 진짜 미사용 평가 (강한 확인적 주장에 조건부)

과거에 보지 않은 경기 ID 목록과 수집/검색 규칙을 점수 확인 전에 hash로 고정한다. 이미 15.16 또는 16.x 전체의 성능을 본 뒤 다시 나누는 것은 새로운 확인적 평가가 아니다.

- 1순위 최소안: 주 V/q/PT를 그대로 적용하는 score-only 새 평가.
- 새로운 보정기 실험이 필요하면 distinct new matches를 chronological adapt/eval로 나누고 q와 PT에 동일한 적응 기회를 준다. 이것은 별도 v2 실험이다.
- V까지 보정하면 ΔV magnitude와 B40가 바뀔 수 있으므로 q-only calibration과 혼합하지 않는다.
- region/patch별로 평가하고, 목표에 필요한 모든 집단을 보고한다. 성능에 따라 수집을 중단하지 않는다.
- 표본수는 경기별 loss difference의 validation/pilot 표준편차 s_d로 계획한다. CI 반폭 h 목표라면 n≈(z*s_d/h)^2를 출발점으로 사용한다. h=.001 등은 정밀도 목표의 제안이지 실용적 가치의 공인 기준이 아니다. 가족 수에 맞는 z와 bootstrap 실효성을 점검한다.

새 평가가 불가능하면 현재 논문을 탐색적 검증으로 제출할 수는 있으나, 미노출 확인적 재현을 주장하지 않는다.

## 10. 공통 통계 계약

- 과제별 동일 평가 키/가중치. 결측이 arm마다 다르면 common-valid와 제외 사유 보고.
- sample-row bootstrap 대신 match cluster resampling.
- 모든 손실 대비에 방향 통일: a−b 음수는 a 우세.
- 같은 표본의 모델 대비와, 서로 다른 표본에서 계산한 절대점수 차이를 구분한다.
- matched group 간 차이는 직접 대비 CI로 판단한다. '한쪽 유의, 다른 쪽 비유의'나 CI overlap 여부만으로 이질성 여부를 결정하지 않는다.
- 주/보조 대비, 다중비교 family, tail quantile, seed, bootstrap 횟수를 실행 전에 JSON에 기록한다.
- 기존 헤드라인의 95% CI를 유지하되, 새 보완 분석의 조정 CI를 별도 열에 둔다.
- CI는 고정된 모델과 라벨 생성 규칙에 조건부다. nested bootstrap/평가기 재적합 변동을 포괄한다고 주장하지 않는다.
- seed sensitivity는 recipe 안정성 진단이다. 여러 seed를 독립 데이터셋처럼 t-test하지 않는다.
- 작은 대비의 실용성은 Brier=.001 등의 임의 한계 하나로 자동 결정하지 않는다.

## 11. 불필요한 실험과 표현 수정

### 실험보다 표현을 고칠 것

1. '진짜 한타 승리' → '해당 동결 평가기로 정의한 SVI'.
2. '교전 시작 전' → 실제 combat onset이 확인되지 않았다면 'first-kill에 앞선 operational cutoff 이전'.
3. '실시간 한타 추천' → retrospective kill-conditioned sample의 방향 예측.
4. 'q>PT이므로 교전 특이 신호를 발견' → scalar p/time 기준선 이상의 예측 정보.
5. 'UNC=.25는 irreducible uncertainty' → outcome prevalence를 사용하는 상수예측의 Brier component.
6. T/S 절대 AUC 차이 → 구성/상태 분포와 사후 집단선택을 고려하지 않은 본질적 규모 비교 금지.

### 성능 추적을 위해 반복할 필요가 없는 것

- 새 Transformer/GNN/거대한 모델 zoo.
- 기존 TEST에서 유리한 B40/threshold/endpoint를 다시 고르기.
- 성공할 때까지 외부 재보정 반복.
- ΔV를 Δlogit으로 바꾼 뒤 동일 이진 방향 라벨을 새 과제라고 주장하기.

엄격히 증가하는 g이면 sign(g(p_post)−g(p_pre))=sign(p_post−p_pre). 따라서 clipping/tie 처리 차이를 제외하면 Δlogit의 방향은 기존 SVI와 같다. 보정의 양의 단조성도 방향을 유지하지만, magnitude/p_pre/B40는 바뀔 수 있다.

## 12. 최소 실행 조합과 완료 기준

### 좁은 범위 저널 검증 원고

G0 → E3 → E2의 최소 진단 및 E4의 peer/endpoint 보완 → 원고 주장 정리.

저널에서 정보군·학습기의 기여를 본 연구질문으로 유지하면 E1도 수행한다. 이를 수행하지 않으면 해당 RQ 문구/기여를 좁힌다. E5는 '기존보다 좋은 가치 지표'를 주장할 때, E6는 확인적/일반화 주장을 강화할 때 선택한다.

### 현재 네 RQ를 유지하는 석사논문

G0 + E1 + E2 + E3 + E4의 작은 정의 census/사례 비교 + E5의 최소 사례 추적.

학위논문 분량을 늘리려고 모든 대안을 전수 실행하지 않는다. 마지막 표에는 질문, 방법, 결과, 허용된 주장, 남는 한계가 한 행으로 이어져야 한다.

### 산출물 제안

- `protocol.json`: 코드/데이터/모델 해시, 마스크, 역할, 대비, seed, 사전 결정/변경 기록.
- `feature_groups.json`: 열 이름, 원천 그룹, 파생 부모, 가용 timestamp, arm membership.
- `integrity.json`: G0 검사와 실패 사유.
- `predictions.parquet`: match_id, engagement_event_key, cohort, split_role, p_pre, y, q, baseline, model_id, evaluator_id, endpoint_rule; 선수 식별자 제외.
- `results.json`, `tables.csv`: 행/경기 수, 점수, Δ, CI, 표본/추론 지위.
- `claim_decision.md`: 긍정/음성/미결 결과에 맞춘 문장과 해석 제한.

경로는 제안이며 아직 저장소에 생성·push한 파일명이 아니다.

## 13. 설계 근거

프로젝트 소스(동일 commit의 파일):
- `docs/MASTER_THESIS_RESEARCH_PLAN_20260919.md` §§4–6.
- `docs/journal_manuscript_v1/THESIS_V2_FOLLOWUPS.md` F1–F6.
- `docs/V_FEATURE_MANIFEST_RUNTIME_20260919.md`.
- `docs/V_MODEL_INPUT_DESIGN_20260919/STATEV2_REFERENCE_FEATURE_SET.json`.
- `docs/V_MODEL_INPUT_DESIGN_20260919/V_MODEL_INPUT_DESIGN_RECOMMENDATION_20260919.md`.
- `docs/REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md`.
- `docs/REVIEW_RESPONSE_RR5_RR6B_20260920.md`.
- `docs/journal_manuscript_v1/02_METHODS_CORE.md`, `03_RESULTS.md`.
- `scripts/rr20260920_q_train_oof_mlp_folds.py`, `rr20260920_review_response_rr12.py`.

방법/투고 근거:
- Dimitriadis, Gneiting & Jordan (2021), Stable reliability diagrams for probabilistic classifiers. PNAS. DOI: 10.1073/pnas.2016191118.
- Kapoor & Narayanan (2023), Leakage and the reproducibility crisis in machine-learning-based science. Patterns. DOI: 10.1016/j.patter.2023.100804.
- IEEE Transactions on Games, Submission guidelines, 2026-09-21 확인: 확장 학회논문은 고정 비율보다 추가된 학술 기여와 선행연구 관계를 명시해야 한다.

이 근거들은 개별 제안의 자동 승인을 의미하지 않는다. 하이퍼파라미터·교란값·정밀도 목표는 본 문서의 제안이며 실험 전 확정 대상이다.
