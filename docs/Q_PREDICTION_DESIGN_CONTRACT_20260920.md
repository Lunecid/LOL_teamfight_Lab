# q 예측 설계·실행 계약 — 새 V(fit85 MLP 사양) 위 SVI 방향

**Date:** 2026-09-20  
**Status:** **EXECUTION LOCK.** 예측 성공 주장 아님. V 아키텍처 재탐색 금지.  
**RQ:** 저널 **J-RQ2** / 석사 **M-RQ3** (측정 = J-RQ1 / M-RQ2).  
**Depends on:** [V_Q_TARGET_LOCK_20260920.md](V_Q_TARGET_LOCK_20260920.md), [V_EVALUATOR_FREEZE_CLOSE_20260920.md](V_EVALUATOR_FREEZE_CLOSE_20260920.md), [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md)

**구분:** “\(V\)로 **측정**한 변화” ≠ “\(q\)가 **사전 예측**한 변화 방향”.

### Cohort scope (학습 ≠ 전이)

| 범위 | 역할 | 금지 |
|---|---|---|
| **210,000 KR 15.14–15.16** | 원천 코퍼스. 15.14 fit / 15.15 Q_CAL·Q_SELECT / 15.16 TEST | 15.16으로 재선정 |
| **16.x KR / NA1** | **동결 후 score-only 전이** ([PAPER_COHORT §4](PAPER_COHORT_CONTRACT_20260919.md)) | EXT를 TRAIN에 섞어 V/q **재적합** |

“16 버전 NA까지” = **전이 평가에 포함**. TRAIN에 NA를 넣어 다시 학습하는 것이 아님.

---

## 0. 실행 순서 (고정 문구)

> **V 평가기 버전 확정 → TRAIN OOF 라벨 생성 → q 후보·기준선 학습 → Q_CAL 보정 → Q_SELECT 선정 → q·평가 규칙 전체 동결 → TEST(15.16) 평가 → 16.x KR/NA1 score-only 전이 → 예측 결과 시각화**

“Q_SELECT 선정 → TEST 봉인”보다 **“Q_SELECT 선정 → q와 평가 규칙 동결 → TEST 평가”**가 정확하다.

동결 대상은 q 가중치만이 아니다.

| 대상 | 함께 고정 |
|---|---|
| **결과 정의** | V 버전, 교전 범위, pre/end, h90, ΔV=0·결측 처리 |
| **결과 정의 — 교전 범위 개정 (2026-09-21)** | 범위 = T(≥4) ∪ S(2–3), pick 제외. S 라벨은 같은 fit85 번들·같은 fold 평가기로 생성(재적합 없음). 근거: SCALE_SPLIT_EXPERIMENT_CONTRACT §2–§3, RR0 manifest. 위 행의 다른 항목은 불변. |
| **예측 입력** | 특징 목록·순서, 전처리, cutoff |
| **예측모델** | q와 기준선, 보정기, 선정 이력 |
| **평가·그림** | 지표·가중치, 방향 판정 임계값, 사례 추출 규칙 |

TEST 라벨(예: 32,981)을 **미리 계산**해 두는 것과, 그 집단 성능으로 **후보·임계값을 다시 고르는 것**은 다르다. 후자는 금지.  
이력: 탐색적 후속 연구 · 과거 TEST 노출 — 공통 계약과 동일하게 유지.

---

## 1. 예측 문제

\[
q(X_{\mathrm{pre}})
=
\widehat{P}\big(Y_{\mathrm{SVI}}=1\mid X_{\mathrm{pre}}\big),
\qquad
Y_{\mathrm{SVI}}=\mathbf{1}[\Delta\widehat{V}>0].
\]

주 목표 = **방향**. 연속 ΔV 회귀는 이번 묶음 밖.  
\(q\) 출력 ≠ 최종 \(W\) 확률 ≠ “한타 후 승률이 \(q\)가 된다”.

---

## 2. OOF — 이번 단계의 실행 관문

### 2.1 Fold = **경기** 단위 (교전 단위 아님)

같은 경기의 **모든 교전**과 **모든 V 학습용 시간 상태**는 같은 fold.

약 **7.8천×5**는 **생성할 TRAIN 교전 라벨 분량**이지, fold별 \(V\)가 학습하는 사례 수가 아니다.

\(V^{(-k)}\)는 제외 fold \(k\) 경기들을 빼고, 남은 경기들의 **전체 적격 시간 상태**

\[
(\text{match},\,t,\,X_{\le t},\,W)
\]

로 학습한다. **한타 시점만** 모아 \(V\)를 다시 학습하지 않는다.

### 2.2 한 교전의 \(p_{\mathrm{pre}}\), ΔV, Y는 **같은** 제외-fold \(V\)

경기 \(m\in\) fold \(k\):

\[
p_{\mathrm{pre},i}^{\mathrm{OOF}}=\widehat{V}^{(-k)}(X_{\mathrm{pre},i}),
\quad
\Delta\widehat{V}_i^{\mathrm{OOF}}
=
\widehat{V}^{(-k)}(X_{\mathrm{post},i})
-
\widehat{V}^{(-k)}(X_{\mathrm{pre},i}),
\quad
Y_i^{\mathrm{OOF}}=\mathbf{1}[\Delta\widehat{V}_i^{\mathrm{OOF}}>0].
\]

**금지:** 라벨만 OOF로 바꾸고 q/PT 입력의 \(p_{\mathrm{pre}}\)에 fit85·구 V 값을 남김.  
TRAIN 행의 \(p_{\mathrm{pre}}\), ΔV, Y, B40는 **동일** \(\widehat{V}^{(-k)}\)에서 일관 생성.

### 2.3 ‘같은 사양’ ≠ ‘같은 가중치’

OOF 평가기는 fit85 번들 가중치 복제가 **아니다**.  
wave-4 **Expanded embedding MLP와 동일 아키텍처·사전 설정**으로 제외-fold 자료에 **새로 적합**.

| 규칙 | 내용 |
|---|---|
| 제외 | fold \(k\) 경기는 가중치 적합·**전처리(ProfileBundle) 적합**·**early-stop holdout**에 모두 없음 |
| Stop | 남은 경기 안에서 match-holdout 15% (seed=7), fit85 규약과 동일 |
| 보정 \(g^{(-k)}\) | **fold마다 V_CAL에서 PosSlopeSigmoid 재적합** (공통 \(g\) 재사용 안 함). 증가 보정이 부호를 유지하는 것과, 확률 **수준**이 fold마다 같은 것은 별개 — 차이를 fold 기록에 남김 |
| 금지 | OOF↔fit85 분포 차이를 이유로 TRAIN을 최종 fit85로 다시 찍어 대체 |

### 2.4 역할 분리 (이름 혼동 금지)

| 역할 | 용도 |
|---|---|
| **V_CAL / V_SELECT** | \(V\) 보정·선정 (WP). OOF \(g^{(-k)}\)는 **V_CAL** |
| **Q_CAL / Q_SELECT** | \(q\) 보정·선정 (SVI). 라벨은 **동결 fit85**로 생성 가능 (V 학습 미포함 전제 유지) |
| **TEST** | \(q\) 주 평가. 라벨 = 동결 fit85. **선정에 사용 금지** |

“CAL/SEL 라벨은 동결 fit85 OK” = **Q_CAL·Q_SELECT**를 뜻한다. V_CAL/V_SELECT와 동일 집단으로 취급하지 않는다.

### 2.5 OOF 완료 검사 (다음 단계 진입 조건)

| 검사 | 통과 조건 |
|---|---|
| 경기 분리 | 평가받는 경기 행 ∉ 해당 \(V\) 적합·stop 자료 |
| 두 끝점 일관성 | pre/post = 동일 \(\widehat{V}^{(-k)}\) |
| 행 완전성 | 적격 TRAIN T∩valid_h90 교전마다 라벨 정확히 한 번 |
| 값 일관성 | \(p_{\mathrm{pre}}\), ΔV, Y, 마스크 상호 일치; B40는 이 \(p_{\mathrm{pre}}\)로 재계산 |
| 추적 | match·s, fold_id, 평가기·보정·전처리 버전/해시 |

**`TRAIN_oof_h90.npz` + 이 검사 PASS** = OOF 단계 완료.

---

## 3. q 후보·절차

| 모델 | 역할 |
|---|---|
| constant | TRAIN SVI 발생률 |
| \(b(p)\) | \(p_{\mathrm{pre}}\)만 |
| **PT_linear** (코드 키 `PT`) | \(p_{\mathrm{pre}}\)+time, 선형 로지스틱 — **이번 주 기준선** |
| PT_flex (후속) | 스플라인·상호작용 등 유연 \(p,t\) — 새 라벨로만 재적합, 보조 대비 |
| logit / LGBM / (선택 MLP) | 더 넓은 사전 특징 — MLP는 V 승자와 무관한 **후보** |

모두 **새 OOF(또는 역할별) 라벨·새 \(p_{\mathrm{pre}}\)**로 적합. 구 V PT 재사용 금지.

절차 (계약 목표):

> **TRAIN에서 학습·조기종료 → Q_CAL에서 q 보정 → Q_SELECT에서 후보 선택 → 규칙 동결 → TEST 평가**

**실행 현실 (2026-09-20 primary):** Q_CAL은 LGBM **early stopping**에만 사용. logit / PT / \(b(p)\)에 **별도 확률 보정기 없음**. 이 표를 “Q_CAL 보정 완료 비교”라고 쓰지 말 것. 보정 버전은 별도 실행·문서.

선정: Q_SELECT **경기 가중 Brier** (전체 T). \(L_{\mathrm{time}}\)을 q에 복사하지 않음.  
주 대비 (TEST, 동일 행·동일 가중):

\[
\Delta\mathrm{Brier}=\mathrm{Brier}(q)-\mathrm{Brier}(\mathrm{PT}_{\mathrm{linear}})
\]

(음수 ⇒ q 손실 더 작음). 점추정·match bootstrap = 같은 가중 정의. \(\tau=0.001\).  
해석 범위: [Q_RESULT_SCOPE_LOCK_20260920.md](Q_RESULT_SCOPE_LOCK_20260920.md).

---

## 4. 시각화 — 측정 vs 예측 분리

| 그림 | 보여주는 것 |
|---|---|
| **측정 차트** | 동결 \(V\)의 교전 전후 승률 변화 |
| **예측 차트** | 그 **방향**을 \(q\)가 사전에 어떻게 예측했는지 + 적중 |

예측 차트에서 **캔들 높이 = 실측** \(p_{\mathrm{pre}},p_{\mathrm{post}}\).  
\(q\)는 **별도 화살표/표식/확률** (예: \(q=0.30\) = “증가 확률 30%” ≠ “승률이 30%가 된다”).

규칙:

- 중간 질의 없으면 고가·저가 **심지 금지** (pre/post 몸통만).
- 맞춘 사례만 추출 금지 — 고정 해시/사전 규칙; 성공·실패 전시 시 **사례용** 명시.
- \(\Delta V=0\) 별도 표시 — \(Y=0\)을 전부 “레드 상승”으로 그리지 않음.
- **전체 근거의 본체 = TEST 표 + \(q-\mathrm{PT}\)**. 캔들 몇 개 적중 ≠ 성능 주장.

---

## 5. 즉시 실행 묶음

> **경기 단위 5-fold MLP \(V^{(-k)}\) 적합 → 같은 제외-fold로 TRAIN \(p_{\mathrm{pre}}\)·ΔV·SVI → 완전성·분리 검사 → 기준선·q 적합**  
> → Q_SELECT 선정·동결 → TEST 평가 → 예측 차트

산출: `outputs/q_newv_fit85_20260920/`  
스크립트: `scripts/rr20260920_q_train_oof_mlp_folds.py`

---

## 6. 한 문장

> \(V\)는 변화의 **측정 기준**을 제공하고, \(q\)는 그 변화를 한타 **전에** 예측할 수 있는지를 시험한다 — TRAIN 라벨은 **경기-fold OOF MLP**로만 완성한다.
