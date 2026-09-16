# Track A 완결: 동일 352입력 plain MLP·residual MLP 대 iq 학습기 — 결과와 ToG 해석

2026-09-16. 설계·구현·실행·검증: Claude(Codex 라인 계승; 사용자 결정 2026-09-16). GPU(cuda) 학습은 사용자 명시 승인. 기존 모델·라벨·종료점·원고·iq 산출물은 변경하지 않았다. 기존 TEST/외부 결과 노출 후의 **탐색적 후속 실험**이며 미접촉 확증이 아니다.

**판단:** 동일한 352개 교전 전 입력에서 신경망은 LightGBM을 넘지 못했다. plain MLP는 T(한타) 전체에서 LightGBM·logistic과 **동률**(ΔBrier 구간 0 포함), residual MLP는 LightGBM·plain MLP보다 **유의하게 나쁘다**. N(비한타)에서는 LightGBM이 두 MLP보다 유의하게 좋고, 두 MLP는 logistic·PT 기준선보다 유의하게 좋다. 이 결과는 DELTA_Q 프로토콜 RQ1("같은 정보에서 선형·트리·신경망이 어떻게 다른가")에 대한 답이며, 신경망 일반의 우열이나 최적 구조를 확정하지 않는다.

## 1. 실행 전후와 고정 조건

- 부모: `incremental_q_training_20260915`(iq)의 동결 승자(PT·전체 logistic·전체 LightGBM)와 그 평가 예측을 같은 행에서 비교. iq 승자는 재적합·재선정하지 않았다.
- 새 family 2종, 각 6설정(폭 128/256/512 × dropout .1/.3) × 시드 7/42/123 × 보정 3개 = 18 후보. AdamW(1e-3, weight decay 1e-4, 모든 파라미터), batch 512, ≤100 epoch, patience 10(stop10 경기가중 Brier, iq와 같은 배정), 전체 TRAIN 재적합(선정 epoch 수), Q_CAL 보정, Q_SELECT(Brier→log loss→이름) 선정, 동결 후 봉인 평가.
- 예측값 정의: 저장된 float32 가중치의 **CPU float64 1스레드 추론**(MKL_CBWR=AVX2,STRICT). 적합 시 재로딩, 동결 전 재계산, 사후 새 프로세스 재로딩 모두 예측이 비트 단위로 동일했다.
- 봉인 게이트는 iq의 동결이 아니라 **이 실행의** `frozen_manifest.json`이며, TEST/외부/iq 평가 예측은 동결 이후에만 열렸다(접근 로그).
- 선정된 설정: T plain `mlp_W128_D0.3__raw`(최적 epoch 3/4/3), T residual `resmlp_W128_D0.1__raw`(**1/1/1**), N plain `mlp_W128_D0.3__sigmoid`(7/7/10), N residual `resmlp_W128_D0.1__sigmoid`(7/2/4). 100 epoch cap 도달 0건. 폭 128이 네 경우 모두 선정됐고, 더 넓은 폭은 Q_SELECT에서 밀렸다.

## 2. MAIN TEST 15.16 (봉인, 경기별 동일 가중치)

T 32,981행/24,020경기, N 130,595행/54,182경기. 짝지은 경기 부트스트랩 1,000회, seed 20260915. ΔBrier는 a−b, 음수가 a 우세.

| cohort | 셀 | 대비 | ΔBrier [95%] | 판정 |
|---|---|---|---|---|
| T | 전체 | **주: residual MLP − LightGBM** | +0.00061 [+0.00012, +0.00110] | LightGBM 우세 |
| T | 전체 | plain MLP − LightGBM | +0.00007 [−0.00046, +0.00057] | 구간 0 포함 |
| T | 전체 | plain MLP − logistic | +0.00006 [−0.00032, +0.00043] | 구간 0 포함 |
| T | 전체 | residual − plain MLP | +0.00054 [+0.00017, +0.00089] | plain 우세 |
| T | 전체 | plain MLP − PT 기준선 | −0.00106 [−0.00173, −0.00038] | MLP 우세 |
| T | 전체 | residual MLP − PT 기준선 | −0.00052 [−0.00110, +0.00005] | 구간 0 포함 |
| T | B40 | 주: residual MLP − LightGBM | +0.00010 [−0.00135, +0.00132] | 구간 0 포함 |
| T | B40 | plain MLP − LightGBM | −0.00100 [−0.00241, +0.00026] | 구간 0 포함 (ΔAUC +0.016 [+0.002, +0.033]) |
| N | 전체 | 주: residual MLP − LightGBM | +0.00110 [+0.00079, +0.00145] | LightGBM 우세 |
| N | 전체 | plain MLP − logistic | −0.00291 [−0.00328, −0.00249] | MLP 우세 |
| N | 전체 | residual − plain MLP | +0.00001 [−0.00014, +0.00016] | 구간 0 포함 |
| N | B40 | 주: residual MLP − LightGBM | +0.00185 [+0.00123, +0.00241] | LightGBM 우세 |

전체 Brier(T): 기존 A specialist 0.22817 < LightGBM 0.22839 ≈ logistic 0.22840 ≈ plain MLP 0.22846 < residual MLP 0.22900 < PT 0.22952 < 상수 0.25000. 절대 AUC(T 전체): plain MLP 0.6694, residual 0.6668.

Q_SELECT(검증) 5-family 순위도 같은 그림이다: T LightGBM > plain MLP > logistic > residual MLP > PT; N LightGBM > plain MLP > residual MLP > logistic > PT.

## 3. 외부 세트(각각 따로; 이전 노출 있음)

주 대비(residual MLP − LightGBM, 전체 행) 판정: KR 16.13 T·N 구간 0 포함; KR 16.14 pilot T(101행)·N(374행) 구간 0 포함; KR 16.15 T·N 구간 0 포함; **NA1 16.13 T는 LightGBM 우세**(+0.00285 [구간 0 초과]), N 구간 0 포함. plain MLP − logistic은 KR 16.13 N·NA1 N에서 MLP 우세, 나머지 구간 0 포함. 두 MLP − PT는 큰 N 세트 3곳에서 MLP 우세, T에서는 전부 구간 0 포함. 즉 한타 T의 외부 전이에서 신경망의 추가 이득은 확인되지 않았고 NA에서는 LightGBM보다 나빴다.

## 4. 해석

- **정보가 같으면 학습기 차이가 작다.** T 전체에서 LightGBM·logistic·plain MLP의 Brier 차이는 0.0001 이내이며 구간이 0을 포함한다. 남는 구조적 차이는 residual MLP의 열세(+0.0005~+0.0006)뿐이다.
- **residual MLP는 즉시 과적합한다.** T 선정 설정이 세 시드 모두 1 epoch에서 최적이었고, 시드별 raw Brier 산포(0.2295–0.2300)가 plain MLP(0.2287–0.2295)보다 나쁘다. 고정된 lr 1e-3·batch 512·patience 10은 프로토콜의 운영 선택이며, 학습률·정칙화 탐색을 넓히면 결과가 달라질 수 있다 — 그것은 다른 예산의 실험이다.
- **N에서는 트리가 확실히 앞선다**(+0.0011, B40 +0.0019). 이 차이는 모델 클래스 효과이며 표현 효과가 아니다(입력 동일).
- 이 실험은 old market_event 표적의 딥 비교(FT/TabNet/SAINT)를 대체하지 않는다. 표적·코호트·프로토콜이 다르다.

## 5. 검증과 기록

- 계약 테스트 19개 두 번 통과(run1, 최종 코드 run2, 모두 전체 적합 전). 스냅샷 → 테스트 → 프로토콜 → TRAIN-only smoke(적합·동결·평가) → 테스트 → 전체 적합 → 동결 → 봉인 평가 → 스냅샷 → 사후검증 순서.
- 사후검증 **115개 검사 통과**(run2). run1은 4개 검사가 실패했는데 실패 원인은 검사 코드였다: `StandardScaler.scale_`을 `np.std`와 비교했고, 영변수 12열(OTHER 영혼/드래곤/타워 열과 시간 상호작용)에서 scikit-learn은 scale_=1로 둔다. 저장된 전처리기는 TRAIN 재적합과 정확히 같았다. 실패 기록과 수정 영수증은 `audit_followup/`에 보존했고 모델·예측·동결·평가는 바뀌지 않았다.
- 독립 재현: 선정 규칙(직접 공식), 보정기 Q_CAL 재적합, 대치기·스케일러 TRAIN 재적합, stop 기록 일관성, seed 평균 = raw, 아키텍처 파라미터 수 = 프로토콜, iq 비교열이 iq 보고 지표를 <1e-12로 재현, 저장 예측으로부터 지표 재계산 <1e-10, 부트스트랩 수용 게이트(30셀), 새 프로세스 재로딩 동일 예측.

산출물: [명세](CLAUDE_TRACK_A_MLP_20260916.md), [실행 보고서](../outputs/track_a_mlp_20260916/REPORT.md), [프로토콜](../outputs/track_a_mlp_20260916/protocol.json), [동결](../outputs/track_a_mlp_20260916/frozen_manifest.json), [검증](../outputs/track_a_mlp_20260916/validation.json), [정정 영수증](../outputs/track_a_mlp_20260916/audit_followup/POSTRUN_CHECK_CORRECTION_RECEIPT.json).

## 6. 논문에서 사용할 문장

> 동일한 352개 교전 전 입력, 동일한 분할·가중치·보정·선정 규칙에서 plain MLP는 한타 결과 방향 예측의 Brier가 LightGBM·정규화 로지스틱과 0.0001 이내로 같았고(짝지은 구간이 0을 포함), residual MLP는 두 모델보다 유의하게 나빴다(+0.0006). 비한타 교전에서는 LightGBM이 두 신경망보다 유의하게 좋았다(+0.0011). 이 비교는 제한된 탐색 예산(family당 6설정, 3시드)의 결과이며, 신경망 일반의 열등이나 최적 구조를 주장하지 않는다.

## 7. 다음

이 결과로 DELTA_Q 프로토콜의 Track A(동일 정보 알고리즘 비교)는 닫힌다. 남은 순서는 ①h60/h120 재적합(h90 선정 하이퍼파라미터 고정, 보정만 재선택) ②균형 상태 SHAP(RQ3) ③V 작동 원리 분해(Δlogit) ④Track B(CoG 표현 어댑터 설계 후 학습) ⑤TRAIN-only G/D ⑥ΔV 표적 원고 재작성이다. 사람 검토 120사례는 여전히 응답 0이다.
