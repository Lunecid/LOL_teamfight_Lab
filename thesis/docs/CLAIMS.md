# 주장–근거 지도 (CLAIMS)

저널 `docs/journal_manuscript_v1/01_OUTLINE_AND_CLAIM_EVIDENCE.md` 의 C1–C22 를 학위논문 장·객체에 대응시킨다. 상태 어휘는 저널 원장을 따른다 (supported / exploratory / correspondence / decomposition / secondary / ordering).

| ID | 주장 | 장·절 | 근거 객체 | 상태 | 해석 한도 |
|---|---|---|---|---|---|
| C1 | 동결 fit85 MLP V̂가 15.16 timeline에서 경기 승패를 순위화한다 (Brier .1552, AUC .8542) | 5.3 | `tab:vw-quality` | supported | 측정 품질 ≠ SVI 타당성; CORP는 분해 |
| C2 | 초반 밴드 [2,10)이 후반보다 약하다 | 5.3 | `tab:vw-quality` | supported | 시간 조건부; TEST 재조정 근거 아님 |
| C3 | 교전 시점 V_pre/V_post가 초반 timeline보다 낫다 | 5.3 | `tab:vw-quality` | supported | 같은 동결 평가기 |
| C4 | ΔV의 방향·부호평균·규모는 별개 객체 | 5.5 | 본문 | supported | 예측 가능성 증명 아님 |
| C5 | 짝지은 quiet 대비 교전 |ΔV| ≫ (0.0633 [0.0612, 0.0654]) | 5.6 | 본문 | exploratory scope | ATT 아님; 전체 교전 아님 |
| C6 | 결정된 킬축 부호와 SVI 일치 0.904 | 5.7 | `tab:correspondence` | correspondence | ≠ q 정확도 |
| C7 | 180 s 내 결정 엘리트 오브젝트 블루 비율 0.604 vs 0.396 | 5.7 | `tab:correspondence` | correspondence | 비가중; 창 안 경기 종료가 대부분 |
| C8 | 지평 flip 낮음(0.007–0.019); 같은 endpoint 공유 0.59–0.81 | 5.8 | `tab:horizon` | supported | 독립 다중 지평 합리성 아님 |
| C9 | 15.16 all-T에서 q가 PT_flex보다 Brier 우세 (−0.00373 [−0.00461, −0.00279]) | 6.5 | `tab:primary-T` | **exploratory** | 시험된 (p,t) 요약 대비; 학습기 비교 아님 (352 vs 362) |
| C10 | PT_linear 연속성 −0.00422 | 6.5 | `tab:primary-T` | exploratory | 역사적 주 대비 |
| C11 | B40 소폭 이득 −0.00214 [−0.00422, −0.00003] | 7.1 | `tab:b40` | exploratory | 상단 0 근처; ≥0.001 명확 이득 주장 금지 |
| C12 | H = D_B40 − D_outside 가 명확히 0이 아님을 주장하지 않음 | 7.1 | `tab:b40` | exploratory | 균형 상태가 더 어렵다고 결론짓지 않음 |
| C13 | Brier 이득이 DSC↑, MCB↑ 와 동행 (ΔMCB +0.00033, ΔDSC +0.00406) | 7.3 | `tab:corp` | decomposition | 게임 기전 아님 |
| C14 | λ·s_Q 제외 후에도 all-T 이득 유지 (≈ −0.004) | 7.2 | 본문 | post-hoc | TEST에서 λ 선택 금지 |
| C15 | KR/NA1 16.13 T: q Brier가 PT_flex보다 나쁨 (+0.0026 / +0.0040) | 7.4 | `tab:ext-T` | ordering (구간 없음) | 점수 전용 동결; 원인 식별 아님 |
| C16 | 외부에서 q DSC > PT 이나 MCB 손실 지배 | 7.4 | `tab:ext-T` | diagnostic | 재보정 복구 증명 아님 |
| C17 | 외부 V_pre Brier ≈ 0.15 | 7.4 | `tab:ext-T` | supported | 외부 ΔV 라벨 검증 아님 |
| C18 | all-S에서 q_S가 PT_flex_S 우세 (−0.00335 [−0.00379, −0.00288]; identity) | 6.6 | `tab:primary-S` | **exploratory** | Within-S only; C9와 비교 금지; 민감도 −0.00346 |
| C19 | 동결 T 모델이 S에 신호 일부 전이; q_S가 더함 (−0.00141; −0.00194) | 6.7 | `tab:transfer-pooling` | secondary | 기전 금지 (X-31) |
| C20 | 합동 T∪S는 어느 쪽에도 이득 없음 (−0.00024; +0.00071) | 6.7 | `tab:transfer-pooling` | secondary | 보정기 의존 (시그모이드에서 구간이 0에 닿음) |
| C21 | S∩B40 이득 nonzero (−0.00210), H_S +0.00178 | 7.1 | `tab:b40` | exploratory | Within-S; C11/C12 옆에 두지 않음 |
| C22 | KR/NA1 16.13 S: q_S Brier < PT_flex_S (−0.0018 / −0.0020) | 7.4 | `tab:ext-S` | ordering (구간 없음) | T 대비 아님; 기전 없음; 저자 문장 pending |

## M-RQ ↔ 장 ↔ Claim

| M-RQ | 장 | 주 claim | 공백·미실행 |
|---|---|---|---|
| M-RQ1 사례 구성 | 3, 4 | 정의·상수표·T/S 규칙 (L 계보 인용) | F5 재민감도 미실행; 사람 주석 없음 |
| M-RQ2 결과 가치 | 5 | C1–C8 | — |
| M-RQ3 사전 예측 | 6 | C9–C10, C18–C20 | M-F0–M-F3 미실행; 학습기 비교 불성립 |
| M-RQ4 적용 범위 | 7 | C11–C17, C21–C22 | F1 어댑터, F4 T–S 외부 대비 이연 |

## 금지 주장 (F1–F8) → `latex/tables/tab_forbidden.tex`

## 저자 pending (본문에 `\pendingauthor{}` 로 표시)

| 항목 | 위치 |
|---|---|
| pick 제외 사유 문장 | 4.5 (`chapters/04_engagement/05_scale_cohorts.tex`) |
| EXT T/S 병기 문장 유지 여부 (초록 Variant A/B) | 7.4, 영문 초록 |
| 릴리스 태그·DOI·파생 자료 라이선스 | 부록 A |
| 학위명·학위수여년월·심사일·심사위원 | `config/meta.tex` |
| RQ 구조 최종 (저널 J-RQ; 석사 M-RQ 문구는 고정) | `.ai/RQ_DECISION_BRIEF.md` |
