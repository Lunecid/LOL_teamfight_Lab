# LoL Teamfight Lab — 교전 정의·측정 체계 v3 릴리스 보고서

| 항목 | 내용 |
|---|---|
| 문서 | Release Report / Patch Notes, 정의 v3 · 코퍼스 v3.3 |
| 기간 | 2026-09-07 ~ 2026-09-11 |
| 브랜치 | `feature/fight-boundary-pipeline` (986b564 → 60945ed, 28 커밋) |
| 상태 | 정의·라벨·입력 확정, 외부 감사 대응 완료, 코퍼스 v3.3 결과 확정 |
| 대상 독자 | ToG 확장 논문 저자, 리뷰 대응, 후속 실험 담당 |
| 근거 문서 | `docs/DEFINITION_EVIDENCE.md`(1~20절), `docs/ENGAGEMENT_DEFINITION_V3.md`, `docs/INPUT_FEATURE_AUDIT_V3.md`, `docs/REPRESENTATION_AUDIT_V3.md` |

---

## 1. 요약 (Executive Summary)

1. **"한타"를 손으로 정한 상수 대신 산출 절차로 정의했다.** 시간 경계 G와 공간 경계 D는 코퍼스
   208,141경기에서 골짜기와 교차점으로 읽고, 존재 게이트 R·B는 게임 규칙(사망 XP 공유 반경 1,600 u,
   킬 크레딧 창 15 s)에서 가져왔다. 세 패치가 하나의 정의 안에 들어오고, 새 패치는 같은
   파이프라인이 판정한다.
2. **정의 변경은 교전의 수를 바꾸지만 예측 가능성은 바꾸지 않는다.** 게이트 두 점(0.669 vs 0.660),
   G·D 변경(교전 +8%, AUC 불변), 정의 v2 → v3(0.746 → 0.736)이 모두 0.01 안이고, 같은 싸움에서는
   차이가 없다. *(All three comparisons predate the leak fix; see the §5.1 note. The G × D sensitivity
   against the v3.3 headline does not exist yet: `\pending{gd_sensitivity_v33}{G x D sensitivity against the v3.3 headline}`.)*
   **반면 입력 누수는 결과를 바꿨다.** 경기 길이로 정규화된 time_norm 하나가 teamfight
   AUC를 0.05 올리고 있었고, 누수를 걷어낸 v3.3에서 "한타가 가장 예측 가능하다"는 규모 기울기는
   채택 컷 `n_min ≥ 4`에서 사라진다(pick − teamfight −0.002 [−0.007, +0.003]). 깨끗한 헤드라인은
   **0.670**이다. *(2026-09-11, was: "0.670(규모 무관)". The pick − teamfight gap depends on where the
   teamfight cut is placed (`docs/CLAUDE_TOG_PAPER_PLAN.md` §4), so it is stated at the adopted cut only.
   Values at `n_min ≥ 3 / 4 / 5` (2026-09-14): +0.0108 [+0.0070, +0.0147] / −0.0019 [−0.0065, +0.0028] /
   −0.0105 [−0.0170, −0.0041], so the sign changes between the cuts. Sources of these values and of −0.002 [−0.007, +0.003]:
   see the §5.3 note.)*
3. **라벨을 "게임이 지급한 골드"로 바꿨다.** 보간된 분 골드(market_lex)의 추가 0.03 AUC는 컷오프
   이전에 앞서 있던 팀을 맞힌 점수임을 5.5% 불일치 행에서 증명했고(0.846 vs 0.244), 사건별 팀 골드
   가격표를 프레임 회귀로 복원해(R² 0.96) 이벤트 가격 라벨 market_event를 주 라벨로 제안한다.
4. **입력 1,015개 기본 특징을 전수조사해 누수 2건, 무의미 블록 1건, 파이프라인 결함 2건을 고쳤다.**
   경기 길이로 정규화된 time_norm, 경기 전체 파괴 위치를 담은 공간 앵커, 원점 거리로 계산되던
   공간 블록, 한 프레임 오래된 값을 읽던 보간 인덱스, 실제 경로에 적용되지 않던 time_norm 수정.
   정렬 검증 98.7~100%.
5. **논문 프레이밍은 "모델을 만들었다"가 아니라 "이 예측 문제를 제대로 측정했다"다.** 정의·타당도,
   입력 감사, 게임 가격 라벨, 규모 분해, 그리고 예정된 "교전 예측이 경기 승패 예측을 개선하는가" 층.

---

## 2. 배경과 목표

- CoG 2026 제출본은 정원 컷으로 탈락했고(메타는 accept 권고), 리뷰는 라벨 정당화·정의·시간 해상도·
  킬 없는 한타를 지적했다. ToG 확장은 이 지적에 근거로 답해야 한다.
- 기존 탐지기(teamfight_v2)는 G 18 s, D 4,000 u, R 1,800 u, B 10 s를 도메인 상수로 고정하고 있었다.
  목표는 (1) 이 상수들을 데이터 또는 게임 규칙에서 산출하는 절차로 대체하고, (2) 라벨과 입력이
  예측 상황(첫 킬 15 s 전, 직전 30 s 관측)과 정확히 같은 정보 집합 위에 있음을 검증하는 것이다.
- 제약: 공개 텔레메트리(Match-V5)는 60 s 프레임과 ms 사건뿐이며 피해 이벤트가 없다. 존재 게이트의
  R·B는 이 데이터로 추정할 수 없다.

---

## 3. 변경 내역 (Patch Notes)

### 3.1 정의 파이프라인 (신규)

| 항목 | 내용 | 커밋 |
|---|---|---|
| 시간 경계 G | 연속 킬 간격 log 밀도의 쌍봉(5.7 / 62.7 s) 사이 KDE 골짜기 = **13.7 s** [12.2, 15.7], ARI ≥ 0.9 평원 10~18 s. 혼합모형 교차 14.5 s, 대역폭 0.05~0.16에서 13.6~14.1 | 986b564, 9f0c941, 31a5aca |
| 공간 경계 D | 같은 챔피언 공유율 50% 교차점 = **4,264 u** [4,256, 4,274]. 영역별 3,974(top)~4,412(base). 라인 방향 비등방성 1.5배는 보고만 | 31a5aca |
| 패치 이동 판정 | 15.14 / 15.15 / 15.16: G 14.0 / 13.5 / 13.7, D 4,285 / 4,265 / 4,241 → **pooled** | 31a5aca |
| 기각한 대안 | 경로 거리(AUC 0.939 vs 0.940), 중심 앵커(판정 6% 변화, 편향 없음), 영역 멤버십 규칙, 지진학 η 근접도(직사각형과 92% 일치) | 986b564 |
| 인프라 | `BoundarySpec` JSON → 탐지기 오버라이드, 패치 인덱스 캐시, 벡터화된 부트스트랩, 비교 스크립트 | 9f0c941, 31a5aca |

### 3.2 존재 게이트 (R, B, M)

| 항목 | 내용 | 커밋 |
|---|---|---|
| 규칙 앵커 | R = 1,600 u(챔피언 사망 XP 공유 반경), B = 15 s(킬·어시스트 크레딧 창; OpenDota 사전 창과 동일), M = 2(Ke 2022). 괄호 [1,350 시야, 1,800 최장 진입기 Zac E] / [5 s 전투 이탈, 15 s]. R and B sources: League of Legends Wiki, "Experience (champion)" (1,600 u) and "Kill" (15 s on Summoner's Rift), checked 2026-09-11. Neither page records a change to either value, including for the corpus patches 15.14-15.16 (`DEFINITION_EVIDENCE.md` §15) | 439c7ca, 4cc72dc |
| 감도 | R×B 12칸: 적격 비율 9.6~34.2%(R), 13.8~26.5%(B). 교전 수를 정하는 상수는 G·D가 아니라 R·B | 439c7ca |
| 재학습 | (1,800, 10) 994,365 교전 0.669 vs (1,600, 15) 531,984 교전 0.660; 공통 32,820 교전에서 0.659 vs 0.658 | c850f72 |
| Data Dragon 강등 | range 필드가 돌진기에서 자리표시자(Zac E 300) → 커버리지 근거를 보조로 | 439c7ca |

### 3.3 라벨과 예측 상황

| 항목 | 내용 | 커밋 |
|---|---|---|
| 타임라인 | 관측 [τ−30, τ], τ = 첫 킬 − 15 s, 라벨 [τ, max(마지막 킬, τ+35)] (= 첫 킬 + 20 s, CoG와 동일 사후 범위) | b9b2207, bd8a8c8 |
| 근거 | 에피소드 지속 p95 17 s, 근처 전환 사건 첫 킬 후 중앙 27 s(의도적으로 라벨 밖), 관측 창 15/30/60 s와 라벨 창 35/45/60 s 감도 오차 안 (pilot on pre-leak-fix features; see the §5.5 caveat) | bd8a8c8, 224f63d |
| market_event | 게임이 지급한 킬 골드(bounty + shutdown) + 회귀 가격표(플레이트 120, 외곽 540, 내부 730, 억제기 타워 510, 넥서스 185, 억제기 85, 바론 1,555, 장로 1,390, 전령 170, 아타칸 240, 유충 40, 와드 킬 25, 어시스트 45, 킬 +20) → ±300 g 데드존 → 킬 → 생존자 → 구조물 → 무승부 제외 | 5285e8d, 5b5f9c8 |
| market_lex 진단 | 두 라벨 불일치 5.5% 행에서 lex 모델 0.846 / event 모델 0.244; lex 라벨은 컷오프 골드 선두와 66.7% 일치 → 추가 AUC는 "앞서 있던 팀" 점수 | 9bea96a |
| 보간 결정 | 입력 = 마지막 프레임 유지 + 프레임 나이(GRU-D), 선택 = 다리 보간(Brownian bridge), 라벨 = 무보간 통화 | 224f63d |

### 3.4 입력 피처 감사

| 항목 | 내용 | 커밋 |
|---|---|---|
| 누수 1 | time_norm = t / (T−1), 경기 총 길이로 정규화 → 절대 시간 / 45분 (`TIME_NORM_ABSOLUTE`) | 5b5f9c8 |
| 누수 2 | 공간 앵커 = 경기 전체 타워 파괴·오브젝트 처치 위치 → 정적 지도 + τ 이전 파괴만 (`ANCHORS_CAUSAL`, `gameplay/anchors.py`, `config/game_rules/map_anchors.json`) | 5b5f9c8 |
| 무의미 블록 | 공간 블록이 중심 상대 좌표로 계산돼 원점 거리였음 → 절대 좌표 시퀀스(`xy_abs_seq`) 전달. 구역 특징 복원(river 66%, jungle 28%) | 5b5f9c8 |
| 추가 | frame_age_s (`TAB_FRAME_AGE_FEATURE`), 7,106차원 | 5b5f9c8 |
| 결함 1 | 보간 인덱스 side="left": 프레임 시각에 캡된 질의가 앞 프레임을 읽음(스칼라 60~120 s 오래됨, 좌표와 불일치) → side="right" | 86095c8 |
| 결함 2 | build_ms_sequence가 전역 벡터를 스냅숏으로 교체해 절대 time_norm이 덮임 → 교체 후 재적용 | 86095c8 |
| 표현 감사 | 원천 필드 추적(미사용 = 경기 후 집계·goldPerSecond·victimTeamfightDamage·FEAT_UPDATE), 죽은 API 스탯 6개, 식별자 2,240열 숫자 입력 → `--categorical` 변형 | aac7131 |

### 3.5 파이프라인·인프라

| 항목 | 내용 | 커밋 |
|---|---|---|
| `LOL_CFG_OVERRIDES` | JSON 환경변수로 CFG 필드 덮어쓰기, 워커 프로세스까지 전달, 미지 필드는 예외 | 35dde2c |
| 킬 인덱스 캐시 키 | TF2_* 상수 포함(게이트만 다른 실행이 캐시를 공유하던 문제) | 35dde2c |
| 샤드 다중 라벨 | `build_corpus_shard.py --extra-labels`, `run_scale_decomposition.py --y-key --teamfight-min --categorical` | 224f63d, fc9a17b, aac7131 |
| 실험 스크립트 | `run_presence_gate_points`, `summarize_presence_gate_points`, `run_prediction_situation_pilot`, `estimate_event_prices`, `build_corpus_v3`, `audit_representation` | 각 커밋 |
| 테스트 | 462 수집, 신규 8파일(env overrides, config, time_norm, anchors, abs xy, market_event, frame alignment) | — |

### 3.6 외부 감사 대응 (v3.2 → v3.3)

| # | 지적 | 조치 | 측정 |
|---|---|---|---|
| 1 | 마지막 킬이 라벨 창 끝과 같으면 제외 | end = 마지막 킬 + 1 ms | 창 변경 7.4%, 전체 라벨 반전 1.8% |
| 2 | 특징 이름 유실(plan 경로) | 이름 통일, 불일치 시 예외 | — |
| 3 | 샤드 무결성 미검증 | manifest + 분해 전 대조 | — |
| 4 | 라벨이 지도 전체 사건 포함 | 앵커 D 안으로 귀속, `@window` 비교판 | 반전 3.0% |
| 5 | 보조 라벨 표본 종속 | 공통 모집단 + 라벨별 무승부 마스크 | — |
| 6 | 아이템 해시 범주형 오지정 | 제외 | — |
| 7 | 회귀 가격 해석 | 표기 수정 + 경기 단위 부트스트랩 | CI 폭 수 골드, 가격표 불변 |
| 8 | 설정 분산 | 오버라이드 병합·충돌 중단, 실효 설정 기록 | — |

근거: `docs/DEFINITION_EVIDENCE.md` 21절. 커밋 2af4eab.

### 3.7 문서

`DEFINITION_EVIDENCE.md` 10~20절(파이프라인, 경로 거리, 앵커, 문헌, 전체 코퍼스, 존재 게이트,
두 점 재학습, 정의 확정, 라벨·예측 상황, v3 결과, 입력 감사), `DEFINITION_LITERATURE.md`,
`references/definition_refs.bib`(VERIFIED 표시, 보간 문헌 6편 추가), `references/definition_section_draft.tex`,
`ENGAGEMENT_DEFINITION_V3.md`, `INPUT_FEATURE_AUDIT_V3.md`, `REPRESENTATION_AUDIT_V3.md`, `DATA_MANIFEST.md`.

---

## 4. 확정 정의 (v3)

| 조건 | 값 | 출처 |
|---|---|---|
| 킬 ≥ 1 | — | 관측 가능성 (킬 없는 encounter는 범위 밖). *v3.3 kill-less grid (2026-09-11):* `features/tog_revision/killless_grid/summary.json`, 20,000 matches, seed 7. Teamfight gate, row `r1600_t4_dG_g15` (≥ 4 alive per side within 1,600 u for ≥ 13.7 s, 15 s grace): 591 of 19,155 **proximity encounters** have no kill, 3.09 % (591/19,155), 0.02955 per match (591/20,000). R = 1,200 u and 2,000 u give 2.00 % and 3.47 % (`r1200_t4_dG_g15`, `r2000_t4_dG_g15`); a 10 s grace gives 4.34 % (`r1600_t4_dG_g10`). These are shares of proximity encounters, not of engagements. *was:* 4.9 %, a v2-constant measurement (R 1,800 u, 10 s grace, 3 per side for ≥ 20 s) over 6,195 proximity encounters in 2,000 matches (`features/killless/r1800_d20_t3.json`) |
| 시간 G | 13.7 s (평원 10~18) | 코퍼스 골짜기 |
| 공간 D | 4,264 u | 코퍼스 교차점 |
| 존재 R / B / M | 1,600 u / 15 s / 2 | 게임 규칙 (League of Legends Wiki, "Experience (champion)" and "Kill", checked 2026-09-11; no change recorded, including for patches 15.14-15.16) / Ke 2022 |
| 정리 | 60 s 상한, 전멸 시 종료, 15 s·2,000 u 후속 흡수 | 발표 코드 |
| 규모 | pick ≤ 1 (19.1%), skirmish 2~3 (60.3%), teamfight ≥ 4 (20.6%) | 참여 분포의 봉우리·골짜기. Shares of the 532,547 labelled v3.3 engagements: 101,798 / 320,878 / 109,829, each n/532,547 (`features/scale_decomposition_v33_market_event.json`, `by_participation_scale`; the three counts sum to 532,505). *was:* 18.9 / 60.7 / 20.4 % (v3, before the input-audit fixes) |

예측 상황: 첫 킬 15초 전, 양 팀이 첫 킬 위치 1,600 u 안에 2명 이상 있는 순간, 직전 30초의 상태
(프레임 1장 + ms 사건 + 고정 정보 + 지형)만 보고 물질적 승자(게임이 지급한 골드 기준)를 맞힌다.

---

## 5. 지표

### 5.1 정의 변경의 영향 (예측력은 불변)

| 비교 | 교전 수 | AUC | 같은 싸움에서 |
|---|---|---|---|
| 게이트 (1,800, 10) → (1,600, 15), G 18·D 4,000, CoG 프로토콜 | 994,365 → 531,984 | 0.669 → 0.660 | 0.659 vs 0.658 |
| G·D 18/4,000 → 13.7/4,264, 같은 게이트 | 531,984 → 574,312 | 0.660 → 0.659 | — |
| v2 → v3, ToG 프로토콜, market_lex | 948,369 → 541,767 | 0.746 → 0.736 | tf ≥ 4: 0.808 vs 0.805 |

*Note (2026-09-11):* all three rows predate the input-audit fixes (leak fix commit 5b5f9c8, 2026-09-09 09:33;
the rows were committed in c850f72 at 02:04 and 1a93134 at 08:28), so their AUCs are on features that still
contained the `time_norm` and spatial-anchor leaks and hold only as within-row comparisons. In the third row,
948,369 is the v2 corpus size and 0.746 / 0.736 / 0.808 / 0.805 are leak-affected values retracted as current
claims (§5.3). The current headline is v3.3: 0.670 on 532,547 engagements.

### 5.2 코퍼스 v3 (ToG 프로토콜: 전체 교전, 경기 단위 5겹 OOF, LightGBM, teamfight ≥ 4)

| 라벨 | n | 전체 | pick | skirmish | teamfight | pick − tf |
|---|---|---|---|---|---|---|
| market_lex | 541,767 | 0.736 | 0.709 | 0.719 | 0.805 | −0.096 |
| market_event (무가격) | 535,444 | 0.702 | 0.673 | 0.686 | 0.770 | −0.098 |
| Eq.3 | 540,774 | 0.682 | 0.657 | 0.666 | 0.744 | −0.088 |

교차 채점: event 모델 → lex 라벨 0.731 (lex 모델 0.736). 세 라벨 모두 규모 기울기 −0.09~−0.10.

*Note (2026-09-11):* §5.2 is the v3 corpus on leak-affected features (built before commit 5b5f9c8). Its scale
gradient of −0.09 to −0.10 came from the `time_norm` leak and is retracted; on the audit-fixed v3.3 features it
does not appear at the adopted cut `n_min ≥ 4` (§5.3; `DEFINITION_EVIDENCE.md` §22).

### 5.3 코퍼스 v3.3 (수정 특징 + 귀속·가격 라벨 + 공통 모집단) — v3 대비

| 실행 | n | 전체 | pick | skirmish | teamfight | pick − tf [95% CI] |
|---|---|---|---|---|---|---|
| **v3.3 market_event (귀속·가격, 주 라벨)** | 532,547 | **0.670** | 0.679 | 0.663 | 0.681 | −0.002 [−0.007, +0.003] |
| v3.3 market_event, 식별자 범주형 | 532,547 | 0.661 | 0.666 | 0.654 | 0.675 | −0.008 |
| v3.3 market_event@window (창 전체) | 541,627 | 0.669 | 0.677 | 0.661 | 0.686 | −0.009 |
| v3.3 market_lex (귀속) | 538,593 | 0.694 | 0.701 | 0.685 | 0.712 | −0.011 |
| v3.3 market_lex@window | 541,895 | 0.693 | 0.700 | 0.685 | 0.712 | −0.013 |
| v3.3 Eq.3 | 561,721 | 0.624 | 0.666 | 0.621 | 0.599 | **+0.067** |
| v3 market_lex (누수 특징) | 541,767 | 0.736 | 0.709 | 0.719 | 0.805 | −0.096 |
| v3 market_event (무가격·창 전체, 누수 특징) | 535,444 | 0.702 | 0.673 | 0.686 | 0.770 | −0.098 |
| v3 Eq.3 (누수 특징) | 540,774 | 0.682 | 0.657 | 0.666 | 0.744 | −0.088 |
| v2 발표 market_lex (tf ≥ 3, 누수 특징; retracted) | 948,369 | 0.746 | 0.716 | 0.715 | 0.784 | −0.068 |

*Note (2026-09-11):* the pick − tf column is the gap at the adopted cut `n_min ≥ 4` only. The gap depends on
where the teamfight cut is placed (`docs/CLAUDE_TOG_PAPER_PLAN.md` §4), so no row here supports "no scale
gradient" without the cut.

Values at `n_min ≥ 3 / 4 / 5` (filled 2026-09-14): pick − teamfight is +0.0108 [+0.0070, +0.0147],
−0.0019 [−0.0065, +0.0028] and −0.0105 [−0.0170, −0.0041]. Source:
`D:/LOL_Project/fusion_2615/features/tog_revision/A6-definition-sensitivity/scale_cut_sensitivity_v33.json`,
keys `participation.cuts.{3,4,5}.gaps.pick_minus_teamfight` (`point`, `ci_2.5`, `ci_97.5`). The file was written
2026-09-11 16:22. It re-scores the stored OOF predictions of the first row with only the cut changed, using 400
match-level replicates. Its `reproduces_main_loop_table` block records that these values reproduce, to four
decimals, the cut table quoted in `docs/CLAUDE_TOG_PAPER_PLAN.md` §2 (400 replicates, seed 7).
Its `n_min ≥ 4` value differs from the first row of the table in the fourth decimal, for two reasons:

- That file reads the 42 rows whose participation count is −1 as zero, which places them in pick.
  `run_scale_decomposition.py`, which produced the first row, leaves them unclassified.
- The first row uses 1,000 replicates, not 400.

Read the second way (`participation_other_negative_count_reading.cuts`), the same file gives +0.0107 [+0.0069, +0.0146],
−0.0020 [−0.0065, +0.0027] and −0.0106 [−0.0171, −0.0043]. The pre-rewrite copy of the file,
`superseded/scale_cut_sensitivity_v33.20260911_1127.json`, holds this second set of values.

Source of the first row: `features/scale_decomposition_v33_market_event.json`. Class AUCs are 0.67887 / 0.66333 /
0.68089. pick − teamfight = 0.67887 − 0.68089 = −0.0020. The 95 % CI [−0.0065, +0.0026] is
`bootstrap.pick_minus_teamfight` (2.5 % and 97.5 %, 1,000 match-level replicates), rounded in the table to
[−0.007, +0.003].

### 5.4 학습기 비교 (2026-09-11, 전체 코퍼스, 패치 홀드아웃)

CoG 리뷰 R2("현대 태뷸러 구조를 시도하지 않았다")에 대한 답. **분할은 패치 홀드아웃** —
학습 15.14 (187,547) / 검증 15.15 (191,184) / **시험 15.16 (153,816)**. 전 학습기가 같은 행·같은
라벨(`y_market_event`, 무승부 33,905행 제외)·같은 분할을 받는다. 입력 동일성은
`scripts/audit_model_comparison_inputs.py`가 19개 항목으로 검증했다(verdict PASS,
`features/model_comparison_input_audit.json`; *was:* 14개 항목).

| 모델 | 인용 | 열 | 파라미터 / 트리 | 검증 | **시험 AUC** | 초 |
|---|---|---:|---:|---:|---:|---:|
| LightGBM (용량 확대) | @ke2017lightgbm | 6,164 | 544 trees | 0.6698 | **0.6692** | 680 |
| LightGBM (발표 설정) | @ke2017lightgbm | 6,164 | 381 trees | 0.6671 | 0.6665 | 144 |
| LightGBM (딥 파이프라인) | @ke2017lightgbm | 7,106 | 306 trees | 0.6666 | 0.6661 | 255 |
| FT-Transformer | @gorishniy2021revisiting | 7,106 | 285,857 | 0.6577 | 0.6561 | 4,146 |
| MLP | 표준 베이스라인 | 7,106 | 4,167,681 | 0.6570 | 0.6552 | 42 |
| SAINT | @somepalli2021saint (preprint) | 7,106 | 459,713 | 0.6565 | 0.6546 | 4,621 |
| 정칙화 로지스틱 | 고전 | 6,164 | — | 0.6444 | 0.6408 | 37 |
| TabNet | @arik2021tabnet | 7,106 | 4,075,989 | 0.6194 | 0.6188 † | 179 |
| **리드 8열 (절제)** | — | **8** | 102 trees | 0.6274 | **0.6261** | 4 |

발표 설정 대비 짝지은 경기 단위 부트스트랩 95% CI (300회): 용량 확대 [+0.0021, +0.0034],
로지스틱 [−0.0274, −0.0239], 리드 8열 [−0.0424, −0.0383].

*Notes (2026-09-11).*

- **Parameter counts** are trainable parameters (`sum(p.numel())`) of the models built with the configuration in
  `scripts/run_deep_tabular_baselines.py` at commit 60945ed, which produced the runs above
  (`features/deep_tabular_v33_patch_{full,ft,saint}.json`): FT-Transformer and SAINT with d_token 32, 3 layers and
  8 heads over 1,016 tokens (1,015 bases + `frame_age_s`) x 7 statistics; MLP 7,106 → 512 → 512 → 512 → 1; TabNet
  with n_d = n_a = 64, 4 decision steps and 7,106 inputs. LightGBM rows give the tree count at the early-stopping
  best iteration (`features/model_comparison_v33_patch.json`, `features/deep_tabular_v33_patch_full.json`).
  The neural learners are **not capacity-matched**: MLP (4.17 M) and TabNet (4.08 M) have 9 to 15 times the
  parameters of SAINT (0.46 M) and FT-Transformer (0.29 M), so reading 1 below is not a like-for-like comparison.
  None of the neural learners received a hyperparameter search (see the limitation paragraph below).
- **† TabNet's 0.6188 was produced with the sparsity regulariser's sign reversed.** `train_torch` computes
  `loss = loss - sparsity_lambda * model.entropy` (`scripts/run_deep_tabular_baselines.py:430` at commit 60945ed,
  λ = 1e-4), where `entropy` is the positive entropy of the feature-selection masks (line 365). Subtracting it
  rewards dense masks. TabNet adds the sparsity regularisation to the overall loss, with coefficient λ_sparse, so
  that masks become sparse (Arik & Pfister, @arik2021tabnet, AAAI 2021). The passage was read on 2026-09-11 in the
  arXiv version, arXiv:1908.07442v5 of 9 Dec 2020, at the end of the "Feature selection" paragraph on PDF p. 5.
  The row is kept for provenance only; do not cite 0.6188 as TabNet's result. TabNet is being re-run with the
  sign corrected:
  `\pending{tabnet_rerun}{TabNet test AUC on the v3.3 patch holdout with the sparsity term added to the loss}`.
- Corrected against `features/model_comparison_v33_patch.json`: the lead-only row takes 4.4 s (*was:* 20), and its
  paired CI is [−0.0424, −0.0383] (*was:* [−0.0425, −0.0384]).

**읽히는 것.**

1. **부스팅 트리가 이기고, 어텐션은 MLP를 넘지 못한다** (not capacity-matched; see the notes above). FT-Transformer 0.6561 / MLP 0.6552 /
   SAINT 0.6546이 0.0015 안에 몰린다. 구 코퍼스(192,727행, 무작위 경기 분할)에서는 FT가 MLP를
   0.024 앞섰는데 전체 코퍼스·패치 홀드아웃에서 그 격차가 사라진다.
2. **0.67은 용량 한계가 아니다.** 400그루→3,000그루·31잎→127잎이 +0.0027이다.
3. **특징 집합이 버는 몫은 +0.043.** 리드 8열 0.6261 → 전체 0.6692.
4. **채택 컷 `n_min ≥ 4`에서 클래스 간 평평함은 모델이 만든다.** *(2026-09-11, was: "규모 무관성은 모델이
   만든다". The classes are flat at this cut only; see the §5.3 note.)* 리드 8열은 pick 0.6017 / skirmish 0.6156 / **teamfight 0.6718**로
   기울기가 있는데(+0.070), 전체 모델은 0.680 / 0.662 / 0.681로 평평하다. "앞선 팀이 이긴다"가
   한타에서만 잘 통하고, 모델이 pick에서 리드 밖 신호를 찾아 격차를 메운다. 철회된 v2 기울기와
   **부호가 반대**라는 점도 누수 설명과 일관된다.
   *(Note, 2026-09-14: the last sentence is not supported as written. The lead-only gap has the **same** sign as
   the retracted v2 gap: pick − teamfight = 0.6017 − 0.6718 = −0.0701 (`features/model_comparison_v33_patch.json`,
   `models.lead_only.by_participation_scale`), against −0.0677 for v2. What does differ is where the extra AUC
   sits. The full model's gain over lead-only is +0.0782 in pick (0.6799 − 0.6017) and +0.0090 in teamfight
   (0.6808 − 0.6718; `models.lgbm_deep`). The `time_norm` leak instead added +0.0026 in pick (0.6207 − 0.6181) and
   +0.0491 in teamfight (0.6912 − 0.6421; `features/leak_ablation_v33.json`, `configs.time_leak` vs `configs.clean`,
   5,000 matches). Do not cite the sentence until the author confirms which comparison was meant.)*
5. **패치 홀드아웃이 더 어렵다.** 같은 LightGBM이 무작위 경기 분할 0.6767 → 패치 홀드아웃 0.6661.

산출물: `features/model_comparison_v33_patch.json`, `deep_tabular_v33_patch_{full,ft,saint}.json`,
`model_comparison_input_audit.json`. 스크립트: `scripts/run_model_comparison_v33.py`,
`scripts/run_deep_tabular_baselines.py --split patch --y-key y_market_event`.

**한계.** 딥 모델의 토큰 뷰는 기저 1,015개를 각각 한 토큰으로 삼고 7개 통계를 입력으로 준다
(7,105 토큰에 대한 어텐션은 계산 불가). v3.3은 `frame_age_s` 한 열이 더 있어 이를 별도 토큰
`[값, 0, …]`으로 붙였다 — 총 1,016 토큰이며 어떤 열도 버리지 않는다. 하이퍼파라미터는
d_token 32 / 3층 / 8헤드 / lr 1e-4로 고정했고 구조별 탐색은 하지 않았다.

**누수 절제 (5,000경기, 같은 교전·같은 라벨, 플래그만 토글):** clean 0.620 (teamfight 0.642) →
time_norm 누수만 0.639 (teamfight 0.691) → 앵커 누수만 0.624 (0.654) → 둘 다 0.640 (0.687).
teamfight의 +0.05는 time_norm(경기 길이) 누수다.

읽는 법: (1) 깨끗한 입력에서 채택 컷 `n_min ≥ 4`의 pick − teamfight는 −0.002 [−0.007, +0.003]이고
Eq.3에서는 뒤집힌다(+0.067). *(2026-09-11, was: "규모 기울기는 없고". The gap depends on the cut; see the §5.3
note.)* (2) 라벨 통화 차이
(lex − event 0.024)는 분 골드 추세의 몫으로 남는다. (3) 귀속은 AUC를 바꾸지 않고(0.669 vs 0.670)
라벨 2.2%를 바꾼다. (4) 식별자 범주형은 −0.009로 해롭다. (5) v2·v3의 teamfight 0.78~0.81과
규모 차이 0.1은 철회한다.

### 5.5 파일럿·감사 지표

| 지표 | 값 |
|---|---|
| 관측 창 15 / 30 / 60 s (553경기) ¹ | 0.614 / 0.626 / 0.631 |
| 라벨 창 35 / 45 / 60 s ¹ | 일치 100 / 97.5 / 94.3%, AUC 0.626 / 0.627 / 0.629 |
| 프레임 나이 특징 ¹ | +0.004 |
| 보간 오차(희생자 킬 위치 대비) | 유지 중앙 2,126 u, 선형 1,149 u |
| 정렬 검증 (수정 후) | 프레임 특징 98.7%, 이벤트·time_norm·라벨 100% |
| 라벨 창 끝 수정 / 사건 귀속 (553경기) | 라벨 반전 1.8% / 3.0% |
| 가격표 회귀 | R² 0.96, 표준오차 0.1~7 g, 패치 간 차이 ≤ 5 g |

¹ *Caveat (2026-09-11).* These three rows come from the prediction-situation pilot
(`features/prediction_situation_pilot.json`, written 2026-09-09 05:33), which ran before the input-audit fixes
(commits 5b5f9c8 at 09:33 and 86095c8): pre-leak-fix features (7,105 columns, no `frame_age_s`), label
`market_lex`, n = 1,360 labelled rows from 1,438 engagements in 553 matches (1,346 and 1,324 rows at the 45 s and
60 s label windows), match-grouped 5-fold OOF LightGBM. They are not v3.3 results. The context-window sweep on v3.3
features (`docs/CLAUDE_TOG_PAPER_PLAN.md` §5 item 16) has not produced results yet:
`\pending{ctx_sweep_v33}{context-window sweep 15/30/60/120 s on v3.3 features}`.

---

## 6. 위험과 한계

- **R·B는 추정이 아니라 규칙 앵커다.** 실측은 리플레이 영상(Track A)에서만 가능하다.
- **후반(20분+) 킬 간격은 단봉이다.** G는 그 구간에서 평원으로만 정당화된다.
- **위치 정보는 구역 수준이다.** 프레임 유지 오차 중앙 2,126 u. 존재 게이트는 추정이며 감도로 보고한다.
- **가격표는 회귀값이다.** 용은 CS 보정 후 0, 첫 타워 보너스는 제외. 규칙 원문과 다르면 규칙이 우선한다.
- **v2 수치와의 연속성은 부록으로만 유지된다.** G·D·R·B·라벨이 모두 바뀌었으므로 .669/.675 재현은 (1,800, 10)+Eq.3 조합에서만 성립한다.
- **캐시 v3는 죽은 API 필드 6개를 담고 있고, 소울 이벤트 키 불일치(무해)와 goldPerSecond 미사용이 남아 있다.** 캐시 v4 항목.

---

## 7. 결정 로그

| 날짜 | 결정 | 근거 |
|---|---|---|
| 09-07 | 상수를 손으로 정하지 않고 산출 절차를 만든다 | 리뷰 지적, 일반성 주장 |
| 09-08 | D는 직선 거리 단일 값 (경로 거리·타원 기각) | 11절 |
| 09-08 | 앵커는 첫 킬 위치 유지 (중심 6% 변화, 편향 없음) | 12절 |
| 09-08 | 전체 코퍼스 spec을 현행으로, 파일럿은 보관 | 14절 |
| 09-09 | 본문 운영점 (1,600 u, 15 s), (1,800, 10)은 부록 | 15~16절 |
| 09-09 | 규모 컷 teamfight ≥ 4 | 4, 17절 |
| 09-09 | 관측 창 30 s, 라벨 창 τ+35 s 유지; 보간 = 유지 + 나이 | 18절 |
| 09-09 | 주 라벨 market_event(가격판) 제안, lex·Eq.3은 견고성 열 | 19, 19.1, 20절 |
| 09-09 | 논문은 표현·측정 감사 버전, 채널 분리 입력 + 단일 학습기 | 대화 기록 |

---

## 8. 로드맵

1. ~~v3.2~~ v3.3 결과 확정 완료. preset `v3.3`과 규칙 앵커 spec으로 재현 경로 고정(완료). 논문 초안 본문 수치를 0.670으로 갱신하고, 규모 차이는 채택 컷 `n_min ≥ 4`의 pick − teamfight
   −0.002 [−0.007, +0.003]와 컷별 값(`n_min ≥ 3 / 4 / 5`: +0.0108 [+0.0070, +0.0147] / −0.0019 [−0.0065, +0.0028] /
   −0.0105 [−0.0170, −0.0041]; source in the §5.3 note)으로 적는다. *(2026-09-11, was: "0.670·규모 무관으로 갱신".
   Cut values filled 2026-09-14.)*
2. 채널 절제 실험(고정 / 스냅숏 / 사건 / 지형; 단독·제외·누적) — 표현 감사의 본문 표.
3. ~~심층 대조(FT-Transformer·SAINT·MLP·TabNet)~~ **완료** — §5.4 참조 (*was:* "9절 참조"). v3.3 행렬, 패치 홀드아웃. TabNet의 정규화 부호 수정 재실행은
   `\pending{tabnet_rerun}{TabNet test AUC on the v3.3 patch holdout with the sparsity term added to the loss}`.
   SHAP은 미실행이며 v3.2 샤드는 삭제됐으므로 v3.3 행렬에서 돌려야 한다.
4. "교전 예측이 경기 승패 예측을 개선하는가" 층: M0 상태만 / M1 + 교전 예측 / M2 + 실현 결과.
5. R·B 이동 판정을 규칙 상수표(패치별 XP 반경·크레딧 창·가격표)로 교체.
6. 캐시 v4: 죽은 스탯 제거, goldPerSecond·victimTeamfightDamage·FEAT_UPDATE 검토.
7. Track A(리플레이 영상)로 B·R과 프레임 유지 오차 실측.

---

## 9. 부록

### 9.1 커밋 이력 (`feature/fight-boundary-pipeline`)

```
aac7131 Representation audit report; optional categorical handling for identifier columns
86095c8 Representation audit script; two defects it found are fixed
5b5f9c8 Input feature audit: two leaks fixed, spatial block repaired, frame age added; event prices from the frames
9bea96a Where market_lex's extra AUC comes from: the 5.5% rows where the label follows the pre-cutoff gold lead
bfecdc6 Compact v3 spec: definition, pipeline, inputs, labels, protocols, numbers, open items
1a93134 Corpus v3 results: three labels on 541,767 engagements, same scale gradient under each
fc9a17b run_scale_decomposition: drop unlabelled rows only from row-aligned arrays
224f63d Pilot results, interpolation decision, and multi-label shards for corpus v3
5285e8d Event-priced market label and the prediction-situation pilot
bd8a8c8 Prediction situation: reasons for the lead, context window, label horizon, interpolation and verdict
b9b2207 Corpus v3 build driver, v3 class cut, and the label / prediction-situation definition
4cc72dc Definition settled: rule-anchored presence gate (1,600 u, 15 s) as the operating point
c850f72 Presence gate at two points: 0.669 vs 0.660 test AUC, equal on shared engagements
35dde2c Config overrides via LOL_CFG_OVERRIDES; presence-gate constants in the fight-index cache key
439c7ca Presence gate R and B: game-rule brackets, R x B sensitivity, fight openers
31a5aca Fight boundary on the full corpus: 208,141 matches, verdict pooled
9f0c941 Scale the boundary pipeline to the full corpus
4f20fee Definition methodology: literature analysis, verified references, and section draft
986b564 Fight-boundary definition pipeline: data-driven G/D per patch with drift verdict
```

### 9.2 재현 명령

```
# 정의 파이프라인 (전체 코퍼스)
LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_fight_boundary_pipeline.py --n-matches-per-patch 0 --n-boot 200 --seed 7 --datadragon --out-dir <out>
# 존재 게이트 두 점
python scripts/run_presence_gate_points.py --points ref_R1800_B10:1800:10000 rule_R1600_B15:1600:15000 --seed 7
# 가격표
python scripts/estimate_event_prices.py --n-matches-per-patch 8000 --seed 7 --output <json> --table config/game_rules/event_prices.json
# 코퍼스 v3.2 + 분해
python scripts/build_corpus_v3.py --out-dir <shards> --output <json> --label-type market_event --extra-labels market_lex,attention_value_win
python scripts/run_scale_decomposition.py --shards <shards> --output <json> --y-key y --teamfight-min 4 [--categorical]
# 감사
python scripts/audit_representation.py --shards <shards> --lgbm-run "<run_*>" --output <json>
```

### 9.3 산출물

`D:/LOL_Project/fusion_2615/features/fight_boundary_full/`(정의 spec), `runs_presence_gate/`, `runs_corpus_v3/`,
`corpus_shards_v3_mlex/`, `corpus_shards_v32_mevent/`, `features/scale_decomposition_v3_*.json`,
`features/scale_decomposition_v32_*.json`, `features/event_prices.json`, `features/representation_audit_v31*.json`,
`config/fight_boundary/`, `config/game_rules/`.

### 9.4 용어

| 용어 | 뜻 |
|---|---|
| 교전(engagement) | 시간 G·공간 D로 묶인 킬 에피소드 중 존재 게이트를 통과한 것 |
| 한타(teamfight) | 양 팀 모두 4명 이상 참여한 교전 |
| τ | 컷오프, 첫 킬 − 15 s. 예측 시점 |
| 존재 게이트 | τ에 첫 킬 위치 R 안에 양 팀 각 M명 이상 생존 |
| market_event | 창 안에서 게임이 지급한 골드로 승자를 정하는 라벨 |
| ToG 프로토콜 | 전체 교전, 경기 단위 5겹 OOF, 규모 분해 |
| CoG 프로토콜 | 패치 홀드아웃 15.14 / 15.15 / 15.16, 분할당 10만 |
