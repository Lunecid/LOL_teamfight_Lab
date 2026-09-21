# 석사논문 현황 인벤토리 (Phase 1)

작성: 2026-09-21  
목적: 저널 스파인·규모분리·잠금 문서에 이미 있는 완료 증거와, 계획만 있고 미실행인 항목을 구분한다.  
규칙: 새 수치·새 실험 없음. 모든 숫자는 아래 출처 문서에서 옮긴다.

---

## 1. 프로젝트 위치

| 축 | 상태 | 근거 |
|---|---|---|
| 저널 (`journal_manuscript_v1`) | 검증 연구 마감·집필 스파인 + T013 집필 패키지 | `JOURNAL_FINISH_LOCK_20260920.md`, `README.md` |
| 석사 계획 | 제목안·M-RQ1–4·목차안 확정(승인≠제출) | `MASTER_THESIS_RESEARCH_PLAN_20260919.md` |
| 규모분리 T/S | 실행·보고·저널 본문 승격(옵션 C) 완료 | `SCALE_SPLIT_*`, T007–T012 |
| CoG 계보 | 읽기 전용 | `docs/tog_manuscript/` |

저널 역할 = 예측 가능성 + 검증(동결 측정).  
석사 역할 = 같은 증거를 **설계 근거·대안·한계** 중심으로 장 단위 확장 (`JOURNAL_FINISH_LOCK` §Journal vs thesis split).

---

## 2. 완료 · 저널과 공유 (C1–C17)

측정·예측·검증 스택. 상세 claim↔수치↔아티팩트는 [`../journal_manuscript_v1/01_OUTLINE_AND_CLAIM_EVIDENCE.md`](../journal_manuscript_v1/01_OUTLINE_AND_CLAIM_EVIDENCE.md).

| 묶음 | Claim | 핵심 수치 (원장) | 출처 |
|---|---|---|---|
| V→W | C1–C3 | TEST timeline Brier 0.1552 / AUC 0.8542; [2,10) 0.2283 / 0.6642; eng pre/post 0.1442·0.1181 | RR6a, `TEST_BAND_LEDGER` |
| ΔV/SVI | C4–C8 | quiet Δ\|ΔV\| 0.0633 [0.0612, 0.0654]; kill-axis 0.904; next-obj 0.604 vs 0.396; horizon flip 0.007–0.019 | RR3, RR4 triad, RR5, RR6b |
| q vs PT (T) | C9–C14 | ΔBrier −0.00373 [−0.00461, −0.00279]; B40 −0.00214; H 0.00181 [−0.00060, 0.00422]; CORP ΔMCB/ΔDSC | RR12, RR6a, RR4 |
| 외부 T | C15–C17 | KR/NA1 ΔBrier +0.0026 / +0.0040; V_pre ≈0.151 | RRX |

동결 계약 요약 (`.ai/CONTEXT.md`):

- \(\widehat V\) = fit85 `A_MLP_expanded` (sha16 `ac459cc4397630a9`)
- SVI = \(1[\Delta\widehat V>0]\); \(\Delta\widehat V=0\) → \(Y=0\)
- B40 = \(0.40\le p_{\mathrm{pre}}\le 0.60\)
- q = logit_state 352입력; 주 기준선 PT_flex; 보정기 identity
- 평가: 경기 가중 \(w_i=1/n_m\); paired percentile bootstrap 2,000회 seed 7 (예외 명시)

인식론 태그: 주 대비는 `EXPLORATORY_REVIEW_RESPONSE_PRIOR_TEST_EXPOSURE` (Methods §0).

---

## 3. 완료 · 석사/저널 병행 — 규모분리 S (C18–C22)

계약: [`../SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md`](../SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md)  
결과: [`../SCALE_SPLIT_TvsS_RESULTS_20260920.md`](../SCALE_SPLIT_TvsS_RESULTS_20260920.md) (identity = 주결과)

| Claim | 핵심 수치 | 해석 한도 |
|---|---|---|
| C18 | ΔBrier(q_S−PT_flex_S) −0.00335 [−0.00379, −0.00288]; AUC 0.5767 vs 0.5360; n=101,205 / 49,730경기 | Within-S only; C9와 나란히 비교 금지 |
| C19 | q_T→S−PT −0.00141; q_S−q_T→S −0.00194 | 기전 금지 (X-31) |
| C20 | q_S−q_TS −0.00024; q_TS−q_T +0.00071 | 코호트별 q 유지 |
| C21 | S∩B40 −0.00210; H_S +0.00178 | T B40와 병치 금지 |
| C22 | EXT S KR/NA1 −0.0018 / −0.0020 (CI 없음) | T EXT와 대비 검정 아님; `\pending{ext-TS-sentence}` |

작업 이력: T007(CLI) → T008(S 라벨) → T009(arms) → T010(identity 주) → T011(잠금 승격) → T012(원고 통합).

---

## 4. 계획만 · 미실행 (석사 심화 / v2)

통합 설계안(실행 전 제안, 2026-09-21): [`../SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md`](../SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md) — G0·E1–E6. 저널 freeze 유지; 결과는 supplementary/thesis 경로만.

| ID | 내용 | 문서 | 상태 |
|---|---|---|---|
| **G0** | 실행 무결성·미래정보 불변성·reload parity | 보완설계 §3; [`../G0_INTEGRITY_STATUS_20260921.md`](../G0_INTEGRITY_STATUS_20260921.md) | **PARTIAL_PHASE_B** (digest·T/S 헤드라인 재집계·OOF load PASS; G0.5/F INCOMPLETE; G0.6 PARTIAL) |
| E1 / M-F0–F3 | 프레임×사건 2×2 (LGBM 4 arm + Q0) | 보완설계 §4; 계획서 §5 | 미실행 |
| E2 | 시간경과·프레임 갱신 진단 | 보완설계 §5 | 미실행 |
| E3 | 외부 paired match-bootstrap (T/S×KR/NA1) | 보완설계 §6; ≈F4 일부 | 미실행 |
| E4 | peer/endpoint/정의 OAT 민감도 | 보완설계 §7; ≈F5 | 미실행 |
| E5 | 동일 사례 기존 지표 비교·사례 추적 | 보완설계 §8; 계획서 §6.1 | 미본문화·조건부 |
| E6 | 진짜 미사용 평가 | 보완설계 §9 | 조건부 deferred |
| F1 | 외부 확률 어댑터 | `THESIS_V2_FOLLOWUPS.md` | deferred |
| F2 | n_min 셀(2/3/4/5+) 재점수 | 동일 | deferred |
| F3 | PT_flex_S 단일 객체 재사용 | 동일 | deferred (위생) |

이 항목들을 “본 연구에서 보였다”고 쓰지 않는다.

석사 최소 실행 조합(보완설계 §12): **G0 + E1 + E2 + E3 + E4(소) + E5(최소)**.  
좁은 저널 보강: **G0 → E3 → E2(+E4 일부)**; 정보군 RQ를 유지할 때만 E1.

---

## 5. 금지 주장 (원고·석사 공통)

출처: `JOURNAL_FINISH_LOCK` Forbidden + claim map F1–F8.

1. 공개 정보가 AUC 0.64를 넘을 수 없다.
2. B40 약화가 관측되지 않은 전투 숙련의 증거다.
3. 재보정만으로 외부 전이가 복구된다 / 외부 실패는 보정만의 문제다.
4. 최적 아키텍처·망라적 순위.
5. 물질/다음 오브젝트 대응 = q 정확도 또는 인과 ATT.
6. 한타/소규모 교전이 “더 예측 가능하다” / 규모 기울기.
7. T/S 차이의 기전 설명 (X-31 철회).
8. 합동 T∪S·pick·소규모 EXT를 해당 코호트 주 대비로 대체.

---

## 6. 문서 지도 (집필 시 읽기 순서)

1. 이 인벤토리 → [`01_MRQ_CLAIM_MAP.md`](01_MRQ_CLAIM_MAP.md)
2. 석사 계획서 [`../MASTER_THESIS_RESEARCH_PLAN_20260919.md`](../MASTER_THESIS_RESEARCH_PLAN_20260919.md)
3. 보완 실험 설계 [`../SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md`](../SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md)
4. LaTeX 구현 [`../../thesis/README.md`](../../thesis/README.md)
5. 저널 Methods/Results/Discussion `02`–`04`
6. 수치: `.ai/CONTEXT.md` 원장, `REVIEW_RESPONSE_*`, `SCALE_SPLIT_TvsS_RESULTS_*`
7. 정의·코호트: `COMMON_RESEARCH_SPINE`, `PAPER_COHORT_CONTRACT`, `V_*` / `Q_*` LOCK·CONTRACT
8. CoG 정의 계보(읽기 전용): `tog_manuscript/sec_definition.tex`

열지 않음: `outputs/`, `data/raw/`, 체크포인트, 대형 NPZ/joblib (`AGENTS.md`).

---

## 7. 저자 pending (석사 본문에도 이월)

| 항목 | 위치 |
|---|---|
| RQ 구조 최종(저널 J-RQ 표 vs 계획서 3문항) | `.ai/RQ_DECISION_BRIEF.md` — **문구 변경 금지**까지 유지 |
| pick 제외 사유 문장 | T012 `\pending` |
| EXT T/S 병기 문장 | `\pending{ext-TS-sentence}` |
| DOSSIER 결정 5건 | `journal_manuscript_v1/00_WRITING_DOSSIER.md` §2 |

---

## 8. 한 줄 요약

**완료:** 동결 \(\widehat V\)·SVI·T/S 코호트별 q−PT 검증·물질/quiet/외부 진단.  
**미실행:** 정보군 모듈·어댑터·정의 재민감도·사례 추적 본문.  
**다음(이 패키지):** 부산대 제본 순서의 국문 장 골격에 위 증거를 재배치.
