# 전체 승률 모형 \(\widehat{V}\) — 설계서·참고문헌 묶음

**작성일:** 2026-09-19  
**상태:** 현행 권위 묶음 (provisional freeze = `shared_lgbm`)  
**범위:** 경기 최종 승패 \(W\)에 대한 공개상태 승률 \(\widehat{V}(X_{\le t})=\widehat{P}(W=1\mid X_{\le t})\).  
SVI / \(q\)는 **이 모형이 동결된 뒤**의 다운스트림이며, 구 V 산출(`outputs/svi_*_20260919`)과 섞어 인용하지 않는다.

로컬 동결 매니페스트(대용량 `outputs/`는 gitignore):  
`outputs/v_redesign_20260919/freeze_manifest.json`

---

## 1. 읽는 순서 (설계)

| # | 문서 | 역할 |
|---|---|---|
| 1 | [COG_SUCCESSION_LOCK_20260919.md](COG_SUCCESSION_LOCK_20260919.md) | CoG 목적 계승 · I1–I4 · strawman 금지 |
| 2 | [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md) | **핵심 계약:** Choice A 공유 시간조건부 기본, 선정·ΔV·작업 패키지 V-1…V-4 |
| 3 | [V1_TASK_CONTRACT_20260919.md](V1_TASK_CONTRACT_20260919.md) | 입력·분할·가중·α_b·캘리브 필드 잠금 |
| 4 | [V_DYNAMIC_FRAME_CONTRACT_20260919.md](V_DYNAMIC_FRAME_CONTRACT_20260919.md) | 프레임 시계 · 시간대 **보고** 규칙 (개발 순서는 redesign이 상위) |
| 5 | [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md) | 210k / 15.14·15.15·15.16 역할 · 15.16 T 봉인 |
| 6 | [COMMON_RESEARCH_SPINE_20260919.md](COMMON_RESEARCH_SPINE_20260919.md) | 저널·석사 공통 척추에서의 \(\widehat{V}\) 위치 |
| 7 | [TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md](TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md) | CoG/ToG/우리 시간대 성능 대조 (라벨 혼동 금지) |
| 8 | [RELATED_FORECAST_VALUE_LIT_20260919.md](RELATED_FORECAST_VALUE_LIT_20260919.md) | 참고문헌 **사용 범위** (아래 §3 요약) |
| 9 | [EXPERIMENT_DESIGN_MAP_20260919.md](EXPERIMENT_DESIGN_MAP_20260919.md) | 목적→개선→예측→검증 설계도 |
| 10 | [SHARED_EXPERIMENT_MATRIX_20260919.md](SHARED_EXPERIMENT_MATRIX_20260919.md) / [EXPERIMENT_INVENTORY_COHORT_20260919.md](EXPERIMENT_INVENTORY_COHORT_20260919.md) | 실험 ID · 산출 목록 |

**구 V evidence (인용 시 old_V만):** [SVI_EVIDENCE_CITE_SHEET_20260919.md](SVI_EVIDENCE_CITE_SHEET_20260919.md)

---

## 2. 잠긴 설계 요약 (한 페이지)

### 2.1 Estimand

\[
\widehat{V}_\theta(X_{\le t})=\widehat{P}(W=1\mid X_{\le t}),\quad
W=\mathbf{1}\{\text{Blue wins the match}\}.
\]

동결 후 \(\Delta\widehat{V}=\widehat{V}(X_{\le t_{\mathrm{end}}})-\widehat{V}(X_{\le t_{\mathrm{pre}}})\), \(\mathrm{SVI}=1[\Delta\widehat{V}>0]\).

### 2.2 Architecture

| Choice | 정의 | 지위 |
|---|---|---|
| **A (default / freeze)** | 단일 공유 \(\theta\), 시간에 조건화된 입력 | **provisional freeze:** LightGBM `shared_lgbm` |
| B (ablation) | 밴드별 LightGBM | V-4 경계 \|Δp\| excess로 **기각** |

### 2.3 Split · weights · selection

| Slice | Role |
|---|---|
| 15.14 `fold0..4` | Fit (+ OOF adapters for TRAIN engagement labels) |
| 15.15 `V_CAL` | Calibration (PosSlopeSigmoid on logit) |
| 15.15 `V_SELECT` | Select by \(L_{\mathrm{time}}=\sum_b \alpha_b\mathrm{Brier}_b\), \(\alpha_b=1/4\) |
| 15.16 `TEST` | Sealed band ledger / application only — **not** selection |

- Query grid: Match-V5 minute frames; **pre-2 min excluded** (from 120 s).  
- Within-band: match-equal weights.  
- Features: StateV2 expanded − `snapshot_age_s` (361 cols; champion_id categorical).

### 2.4 Continuity (V-4) before SVI

- Interior vs band-boundary consecutive \|Δp\|.  
- Multi-spec \(\mathrm{sign}(\Delta\widehat{V})\) agree.  
- Per-band rejected: boundary excess 0.0299 > shared 0.0161.

### 2.5 Work packages

| ID | Content | Status (2026-09-19) |
|---|---|---|
| V-1 | Task contract | done |
| V-2 | Candidate compare (shared / per-band) | done |
| V-3 | Freeze + band ledger | **shared_lgbm** provisional |
| V-4 | ΔV continuity | done (rejects per-band) |
| Then | New SVI / B40 / \(q\) on this lineage only | lean primary started (`svi_newv_*`, local outputs) |

### 2.6 Reproducibility scripts (repo)

| Script | Role |
|---|---|
| `scripts/rr20260919_v_redesign_fit.py` | Fit + V_SELECT + TEST ledger |
| `scripts/rr20260919_v_redesign_continuity.py` | V-4 jumps / sign agree |
| `scripts/rr20260919_v_redesign_oof.py` | 5-fold OOF adapters + TRAIN labels |
| `scripts/rr20260919_v_redesign_relabel.py` | VAL/TEST labels with frozen V |
| `scripts/rr20260919_svi_v_time_strata.py` | \(\widehat{V}\to W\) time strata (old/new reporting) |
| `scripts/rr20260919_svi_newv_primary.py` | Lean \(q\) slate on **new** V labels |

---

## 3. 참고 논문 · 사용 범위

상세 표·금지 주장: [RELATED_FORECAST_VALUE_LIT_20260919.md](RELATED_FORECAST_VALUE_LIT_20260919.md).  
시간대 수치 대조: [TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md](TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md).

### 3.1 \(\widehat{V}\) 재설계 — 먼저 읽을 것

| 문헌 | 우리에게 쓰는 것 | 쓰지 않는 것 |
|---|---|---|
| **Hodge et al. (2021)** IEEE ToG — live esports WP | 시간대 평가 · history · per-minute = Choice B 선례 | ΔV용으로 분 단위 모형 자동 채택 |
| **Kim, Lee & Chung (2020)** IEEE CoG — calibrated LoL WP | 확률 품질 · 캘리브 비교 | day-one DU loss 의무 |
| **Maymin (2021)** JQAS — smart kills / WP change | 매치 WP → 사건 가치 \(\Delta\widehat{V}\) · **공유** 모형 | sparse logistic = 천장 |
| **Ke et al. (2022)** IEEE CoG | (옵션) 과거 교전 → 매치 WP | “다음 교전 승자”와 동일시 |
| **Hitar-García et al. (2023)** IEEE ToG | (옵션) 프리게임/시너지 | in-game time-band WP 논문으로 취급 |

**읽기 순서:** Hodge → Kim → Maymin.

### 3.2 점수·예측 가치 프레임 (\(q\) / SVI 이후)

| 문헌 | 사용 |
|---|---|
| **Gneiting & Raftery (2007)** | Proper scores (Brier / log) for selection |
| **Chong & Hendry (1986)**; **Fair & Shiller (1989/90)** | Incremental information 배경 |
| **Clements & Harvey (2010)** | Probability encompassing (Brier/log에 가깝) |
| **Giacomini & White (2006)** | State-dependent predictive ability **동기** (우리 검정이 곧 GW는 아님) |
| **Diebold & Mariano (1995)** | Loss-differential 계보 (match bootstrap ≠ DM 통계 자동) |
| **Brill–Yurko–Wyner (2024)** | WP 불확실성 / 의존 프레이밍 |
| WPA / LI (Mills; Tango; MLB glossary) | \(\Delta\widehat{V}\) 계보; B40 ≠ leverage 정의 |

### 3.3 한 문장 스탠스

문헌은 **시간 인식 매치 WP를 어떻게 만들고 점수화할지**, 그리고 baseline 대비 정보가 더해지는지를 **어떻게 물을지** 알려 준다.  
균형 prior가 본질적으로 예측 불가능하다거나, 큰 \(|\Delta\widehat{V}|\)가 깨끗한 라벨임을 **증명하지는 않는다**.

---

## 4. 관련이나 이 묶음의 중심이 아닌 문서

| 문서 | 이유 |
|---|---|
| [STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md](STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md) | SVI/라벨 redesign — V 동결 **이후** |
| [SVI_MODEL_RESELECTION_TRANSFER_20260919.md](SVI_MODEL_RESELECTION_TRANSFER_20260919.md) | \(q\) 재선정 — **구 V** 라인 |
| [JOURNAL_RESEARCH_PLAN_20260919.md](JOURNAL_RESEARCH_PLAN_20260919.md) / [MASTER_THESIS_RESEARCH_PLAN_20260919.md](MASTER_THESIS_RESEARCH_PLAN_20260919.md) | 원고 분기 RQ — V 계약의 상위/병렬 |

---

## 5. One-sentence lock

> Build a **shared time-conditional** \(\widehat{V}(X_{\le t})=\widehat{P}(W=1\mid X_{\le t})\), select on **time-aware proper scores**, record band performance as a ledger, prove \(\Delta\widehat{V}\) is not an artifact of model switching — **then** rebuild SVI and \(q\).
