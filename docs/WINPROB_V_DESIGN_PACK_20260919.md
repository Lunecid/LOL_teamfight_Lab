# 전체 승률 모형 \(\widehat{V}\) — 설계서·참고문헌 묶음

**작성일:** 2026-09-19  
**상태:** 잠정 기준 = Choice A `shared_lgbm` (**a priori**). V-2 wave-1 · V-4 **일부** 진단 완료. 최종 측정모형 승인 전: 동일 \(g\circ f\) 검증·DEV 모델전환 분리·시간대/적용시점 표 확정.  
**범위:** \(\widehat{V}(X_{\le t})=\widehat{P}(W=1\mid X_{\le t})\).  
구 V 산출(`outputs/svi_*_20260919`)과 새 V를 섞어 인용하지 않는다.

로컬: `outputs/v_redesign_20260919/freeze_manifest.json`  
GitHub 수치 발췌: [BAND_LEDGER_SHARED_LGBM_20260919.md](BAND_LEDGER_SHARED_LGBM_20260919.md)

---

## 0. 최종 평가 함수 (LOCK)

\[
\boxed{
\widehat{V}_{\mathrm{final}}(X)
=
g_{\mathrm{frozen}}
\!\bigl(
f_{\mathrm{frozen}}
\!\bigl(T_{\mathrm{frozen}}(X)\bigr)
\bigr)
}
\]

\(T\)=전처리, \(f\)=학습기, \(g\)=V_CAL PosSlopeSigmoid.  
**선정·TEST ledger·연속성 primary·SVI 재라벨**은 모두 이 함수를 쓴다. 원확률 \(f(X)\)는 보정 영향 진단용.

---

## 1. 읽는 순서 (설계)

| # | 문서 | 역할 |
|---|---|---|
| 1 | [COG_SUCCESSION_LOCK_20260919.md](COG_SUCCESSION_LOCK_20260919.md) | CoG 목적 · I1–I4 |
| 2 | [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md) | 핵심 계약 |
| 3 | [V1_TASK_CONTRACT_20260919.md](V1_TASK_CONTRACT_20260919.md) | 필드 잠금 |
| 4 | [BAND_LEDGER_SHARED_LGBM_20260919.md](BAND_LEDGER_SHARED_LGBM_20260919.md) | **새 V 시간대 성능표** |
| 5 | [V_DYNAMIC_FRAME_CONTRACT_20260919.md](V_DYNAMIC_FRAME_CONTRACT_20260919.md) | 프레임·밴드 **보고** |
| 6 | [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md) | 210k / 15.16 T |
| 7 | [RELATED_FORECAST_VALUE_LIT_20260919.md](RELATED_FORECAST_VALUE_LIT_20260919.md) | 문헌 사용 범위 |
| 8 | [TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md](TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md) | CoG / **로컬 ToG 파이프라인** / 구V 대조 (**≠ Hodge 논문 수치**) |

---

## 2. 설계 요약

### 2.1 Estimand · architecture

\(W=\mathbf{1}\{\text{Blue wins}\}\). 동결 후 \(\Delta\widehat{V}=V_{\mathrm{end}}-V_{\mathrm{pre}}\), \(\mathrm{SVI}=1[\Delta\widehat{V}>0]\).

| Choice | 지위 |
|---|---|
| **A shared** | **a priori default / provisional freeze** |
| B per-band | ablation; V_SELECT에서 \(L_{\mathrm{time}}\) 미세 우위만으로는 승격하지 않음. TEST 경계 excess는 **경고**이지 “모델전환 잡음 입증·기각”이 아님 |

### 2.2 Split · sample · selection

| Slice | Role |
|---|---|
| 15.14 fold0..4 | Fit (+ OOF TRAIN labels) |
| 15.15 V_CAL | \(g\) |
| 15.15 V_SELECT | \(L_{\mathrm{time}}\) 선정 (**only**) |
| 15.16 TEST | sealed ledger + exploratory continuity **진단** — **선정 아님** |

- 학습/평가 표본: **`bucket_only=True`** (분 grid의 bucket sample).  
- \(\alpha_b=1/4\); 필수 밴드 결손 시 후보 **ineligible** (α 항을 조용히 제거하지 않음).  
- Tie-break (wave-1): V_SELECT **overall** match-weighted logloss (시간균형 logloss 아님 — V1에 명시).

### 2.3 Continuity (V-4) — 범위

| 항목 | 상태 |
|---|---|
| 연속 bucket 질의 interior vs boundary \|Δp\| (calibrated) | 실행됨 — **경고 신호** |
| multi-spec sign(ΔV) agree | 실행됨 |
| 동일 상태 \(D_{\mathrm{switch}}(x)=\|V_{b+1}(x)-V_b(x)\|\) | **미실행** |
| Quiet / frame-refresh / 사건매칭 | **미실행** |

표현: *“공유 모델을 잠정 채택했으며, 밴드별에서 더 큰 경계 구간 변화가 관측되어 추가 모델전환 진단이 필요하다.”*

### 2.4 Work packages

| ID | Status |
|---|---|
| V-1 | complete |
| V-2 wave 1 | shared/per-band LGBM + legacy compare (**MLP·history K 미실행**) |
| V-3 | provisional freeze `shared_lgbm` + band ledger |
| V-4 | **partial** (위 표) |
| Then | new SVI/\(q\) exploratory (`svi_newv_*`); V 수정 시 전량 재발급 |

### 2.5 Scripts

`rr20260919_v_redesign_{fit,continuity,oof,relabel}.py`, `rr20260919_svi_newv_primary.py`, …

---

## 3. 참고 논문 (요약)

상세: [RELATED_FORECAST_VALUE_LIT_20260919.md](RELATED_FORECAST_VALUE_LIT_20260919.md).

| 문헌 | 사용 |
|---|---|
| Hodge et al. 2021 IEEE ToG | 시간대 평가·Choice B **선례** (우리 로컬 `temporal_winprob_v3` 표 ≠ 이 논문 수치) |
| Kim et al. 2020 IEEE CoG | 캘리브·확률 품질 |
| Maymin 2021 JQAS | \(\Delta\widehat{V}\) · **공유** 모형 |
| Gneiting–Raftery 2007 | proper scores |

---

## 4. One-sentence lock (softened)

> 시간 인식 확률평가로 공유 시간조건부 \(\widehat{V}\)를 선택·기록하고, 모델 전환·관측 갱신·사양 변경에 따른 \(\Delta\widehat{V}\) 민감성을 점검한 뒤 SVI와 \(q\)를 재구성한다.
