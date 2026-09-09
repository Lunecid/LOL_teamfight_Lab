# 교전·한타 정의와 예측 구조 v3 (2026-09-09 현재)

`docs/DEFINITION_EVIDENCE.md`(1~19절)의 압축판. 근거와 측정치는 그쪽을 본다. 이 문서는 "지금
무엇이 정의돼 있고, 무엇이 들어가고, 무엇이 나오고, 어디까지 확인됐는가"만 적는다.

## 0. 한 장 요약

```
시간축 (첫 킬 = 0)
  -45 s ------------ -15 s ---------- 0 -------- +20 s (또는 마지막 킬)
  |  관측 창 30 s    |  τ 컷오프      | 첫 킬    |  라벨 창 끝
  [τ-30, τ] 상태·사건 -> 예측 -> [τ, max(마지막 킬, τ+35)] 결과 채점
```

- **교전(engagement)**: 양 팀 챔피언이 같은 순간 같은 장소에 마주 서 있다가 그 자리에서
  누군가 죽는 사건. **한타(teamfight)**: 양 팀 모두 4명 이상이 참여한 교전.
- **예측 상황**: 첫 킬 15초 전, 양 팀이 이미 첫 킬 위치 1,600 u 안에 2명 이상 있는 순간,
  직전 30초의 상태만 보고 이 교전의 물질적 승자를 맞힌다.
- **현재 숫자(v3, 전체 교전 541,767)**: 전체 AUC 0.702~0.736(라벨에 따라), teamfight 0.77~0.81,
  pick과 teamfight 차이 -0.09~-0.10(모든 라벨에서 동일).

## 1. 정의 (코퍼스 v3)

| 조건 | 규칙 | 값 | 출처 | 근거 (DEFINITION_EVIDENCE 절) |
|---|---|---|---|---|
| 1 킬 | 챔피언 킬 ≥ 1 | — | 관측 가능성 | 공개 텔레메트리에 피해 이벤트 없음. 킬 없는 encounter(4.9%)는 범위 밖 명시 (3.1, 5) |
| 2 시간 G | 연속 킬 간격 ≤ G면 같은 사건 | 13.7 s (평원 10~18) | 코퍼스 208,141경기 | 킬 간격 log 밀도 쌍봉(5.7 / 62.7 s)의 골짜기. 패치별 14.0/13.5/13.7. 후반은 단봉이라 평원으로만 정당화 (14) |
| 3 공간 D | 사건 안 두 킬 거리 > D면 분리 | 4,264 u (CI 4,256~4,274) | 코퍼스 | 같은 챔피언 공유율 50% 교차점. 영역별 3,974~4,412. 라인 방향 1.5배 비등방성은 보고만 (14, 11) |
| 4 존재 R, B, M | τ = 첫 킬 − B에 첫 킬 위치 R 안 양 팀 각 M명 생존 | R 1,600 u, B 15 s, M 2 | 게임 규칙(사망 XP 공유 반경, 킬 크레딧 창), Ke 2022 | 데이터로 추정 불가. 괄호 [1,350 시야, 1,800 최장 진입기] / [5 s 전투 이탈, 15 s]. 게이트는 교전 수를 -50~+80% 바꾸지만 예측력은 불변 (15, 16) |
| 5 정리 | 60 s 상한, 전멸 시 종료, 끝난 뒤 15 s·2,000 u 후속 킬 흡수, 2분 이전·마지막 30 s 제외 | 발표 코드 | — | 추격·귀환 연쇄 차단 (3.6, 3.7) |

**규모 클래스** (사후, 참여 인원의 작은 쪽 min): pick ≤ 1 (v3 18.9%), skirmish 2~3 (60.7%),
teamfight ≥ 4 (20.4%). 분포의 봉우리(2v2, 5v5)와 골짜기(4v4)에서 읽음. 모델 입력·층화에 쓰지 않음.

**범위 밖**: 킬 없는 대치, 싸움의 "시작 시각"(B는 리드), 비참여 챔피언의 정확한 위치.

## 2. 파이프라인 구조

```
Match-V5 timeline/detail --> 캐시(분 프레임 node_minute·global_minute, events)          data/cache_io.py
  |-> 정의 파이프라인: 킬 쌍 -> G(골짜기) · D(교차점) · R/B(규칙표) -> BoundarySpec       analysis/*, scripts/run_fight_boundary_pipeline.py
  |-> 탐지(teamfight_v2): 시간 연쇄 G -> 공간 분할 D -> 존재 게이트(R, B, M) -> 정리 규칙   gameplay/fights.py, fight_clustering.py
  |      => FightRef (match, τ, 라벨 창 끝, 첫/마지막 킬, 참여·존재 인원)
  |-> 라벨: 창 [τ, max(마지막 킬, τ+35)]에서 market_event / market_lex / Eq.3               gameplay/labels.py
  |-> 특징: [τ-30, τ] 5 s x 6 구간, 노드 76x10 · 전역 26 · 이벤트 44 · 임베딩                gameplay/pipeline.py, features.py
  |      => 탭 표현 7,105 = 1,015 기본 x 7 통계(last/mean/std/min/max/delta/slope)          train/baseline.py
  |-> 모델: LightGBM(기준) / FT-Transformer·SAINT·MLP·TabNet / 시퀀스·그래프·퓨전            train/*
  |-> 평가: ToG(전체, 경기 5겹 OOF, 규모 분해) · CoG(패치 홀드아웃, 100k/분할)               scripts/run_scale_decomposition.py, runner.py
상수 주입: LOL_CFG_OVERRIDES(JSON) -> 워커 프로세스까지 전달; 킬 인덱스 캐시 키에 TF2_* 포함   core/config.py, data/index_split.py
```

## 3. 입력 (정보 집합)

**컷오프** τ = 첫 킬 − 15 s. **관측 창** [τ − 30 s, τ], 5 s 구간 6개(L = 6).

| 채널 | 차원 | 내용 | 시간 해상도 | 보간 |
|---|---|---|---|---|
| 플레이어 노드 | 76 x 10 | 스냅숏 17(좌표·레벨·XP·골드·CS·CC·HP·MP·생존·바론/장로·영혼·궁 레벨), 상태 11, 룬 11, 챔피언 스탯 25, 피해 12 | 60 s 프레임 | 마지막 프레임 유지(ffill). 좌표는 프레임 사이 선형이되 τ 이하 프레임으로 캡 → 결과적으로 유지. 이동 > 7,000 u 또는 사망·부활이면 보간 안 함 |
| 팀 전역 | 26 | time_norm, 밴 10, 골드·XP·레벨·CS·정글CS·생존 차이, 누적 킬·타워·억제기·용·바론·전령·아타칸·플레이트·유충 차이 | 60 s 프레임 | 유지 |
| 이벤트 집계 | 44 x 6 | 구간별 킬·현상금·처형·연속·다중킬·에이스·용·바론·전령·아타칸·유충·타워·억제기·플레이트·오브젝트 현상금·와드 4종·아이템 3종 (팀별) | ms 정확 | 없음 |
| 범주 임베딩 | — | 챔피언, 룬, 소환사 주문 | 고정 | — |
| (예정) 신선도 | 1 | τ − 마지막 프레임 시각 | — | 파일럿 +0.004; GRU-D의 관측 후 경과 시간 |

**인과 계약**: 어떤 특징도 τ 이후를 읽지 않는다(`interpolate_node_global(max_snapshot_ms = τ − 1)`,
`tests/test_pipeline_interp_no_future.py`). 앵커(첫 킬 위치)와 존재 게이트는 표본 **선택**에만 쓰이는
회고 정보이며, 존재 게이트의 5 s 격자(선형 + 킬 궤적 다리 보간)는 특징 빌더가 쓰지 않는다.

**보간 오차(측정)**: 희생자 킬 위치 대비 프레임 유지 오차 중앙 2,126 u(프레임 후 0~10 s 616 → 50~60 s
3,746), 선형(다음 프레임 사용) 1,149 u. 위치 특징은 "구역" 수준으로만 신뢰.

## 4. 출력 (라벨)

창 [τ, T_end], T_end = max(마지막 킬, τ + 35 s) (= 첫 킬 + 20 s 이상, CoG의 사후 범위와 동일).
양성 y = 1은 블루 승, 균형 약 0.51.

| 라벨 | 시장 판정 통화 | 데드존 | 보정 순서 | 무승부 | v3 AUC |
|---|---|---|---|---|---|
| **market_event** (제안 주 라벨) | 게임이 지급한 킬 골드(bounty + shutdownBounty), ms 정확, 보간 없음 | ±300 g | 클러스터 킬 → 마지막 킬 시점 생존자 → 구조물·오브젝트 부호 | 제외 (1.2%) | 0.702 |
| market_lex (ToG 기존) | 팀 골드(60 s 프레임 선형 보간)의 창 변화 차이 | ±300 g | 같음 | 제외 | 0.736 |
| Eq.3 attention_value_win (CoG) | 연구자 가중 사건 가치의 소프트맥스 합 | — | — | 동전 | 0.682 |

가중치는 두지 않는다. 골드는 게임의 환율이고 데드존 300 g은 기본 킬 하나이며, 보정 순서는
물질적으로 대등한 싸움에서만 작동한다. market_lex의 추가 0.03은 그 분의 경제 추세(파밍)에서 온다.
라벨 창 45/60 s, 관측 창 15/60 s로 바꿔도 결과가 오차 안(18.2절).

## 5. 평가 프로토콜

| 이름 | 표본 | 분할 | 학습기 | 보고 |
|---|---|---|---|---|
| ToG 주 | 전체 교전(무승부 제외) | 경기 단위 5겹 GroupKFold, OOF 고정 | LightGBM 400 / 0.05 / 31 | AUC 전체·참여 규모·존재 규모·패치, 부트스트랩 1,000 |
| CoG 연속성 | 분할당 100,000 | 패치 홀드아웃 15.14 학습 / 15.15 검증 / 15.16 시험 | 같음 | 시험 AUC·AP·Brier·단계별 |
| 심층 대조 | ToG 행렬 | 같은 겹 | FT-Transformer, SAINT, MLP, TabNet | AUC |

## 6. 현재 숫자

**정의 파이프라인(208,141경기)**: G 13.7 s [12.2, 15.7], D 4,264 u [4,256, 4,274]; 패치별 G 14.0/13.5/13.7,
D 4,285/4,265/4,241 → verdict pooled.

**존재 게이트 두 점(발표 파이프라인, G 18·D 4,000)**: (1,800 u, 10 s) 994,365 교전, 시험 AUC 0.669;
(1,600 u, 15 s) 531,984, 0.660; 공통 교전에서 0.659 vs 0.658.

**코퍼스 v3**

| 프로토콜 / 라벨 | n | 전체 AUC | pick | skirmish | teamfight | pick − tf |
|---|---|---|---|---|---|---|
| CoG, Eq.3 (시험) | 574,312 | 0.659 [0.656, 0.662] | early 0.611 | mid 0.696 | late 0.795 | — |
| ToG, market_lex | 541,767 | 0.736 | 0.709 | 0.719 | 0.805 | −0.096 |
| ToG, market_event | 535,444 | 0.702 | 0.673 | 0.686 | 0.770 | −0.098 |
| ToG, Eq.3 | 540,774 | 0.682 | 0.657 | 0.666 | 0.744 | −0.088 |
| (v2 발표, market_lex, tf ≥ 3) | 948,369 | 0.746 | 0.716 | 0.715 | 0.784 (≥ 4: 0.808) | −0.068 |

교차 채점: event 모델을 lex 라벨에 채점하면 0.731(lex 모델 0.736). 두 모델이 배운 것은 같다.
컷오프 시점 존재 기준 teamfight(2.1%)는 0.837.

**파일럿(553경기)**: 관측 창 15/30/60 s 0.614/0.626/0.631; 라벨 창 35/45/60 s 일치 ≥ 94%, AUC 동일;
신선도 +0.004.

## 7. 결정된 것 · 제안 대기 · 남은 작업

**결정됨**: 정의 v3(1절), 운영점 (1,600 u, 15 s), 규모 컷 ≥ 4, 관측 창 30 s, 라벨 창 τ + 35 s,
보간(입력 = 유지 + 신선도, 선택 = 다리, 라벨 = 무보간 통화 후보), 본문 하나·나머지 부록.

**제안 대기(사용자 확인)**: 주 라벨 = market_event (헤드라인 0.702, teamfight 0.770), market_lex·Eq.3은 견고성 열.

**남은 작업**

1. 주 라벨 확정 → `config/fight_boundary/spec_*.json`·`core/config.py` 기본값(G, D, R, B, H, LABEL_TYPE)을 v3로 갱신, 논문 초안 수치 갱신.
2. 신선도 특징을 특징 파이프라인에 정식 추가(현재는 파일럿에서 열 추가로만 검증).
3. R·B 이동 판정을 Data Dragon 커버리지 대신 규칙 상수표(패치별 XP 반경·크레딧 창)로 교체.
4. 관측 창 15/30/60 s를 전체 코퍼스 탭 표현에서 재확인.
5. 심층 대조·SHAP을 v3 행렬에서 재실행.
6. "교전 예측이 경기 승패 예측을 개선하는가" 층(M0 상태만 / M1 +교전 예측 / M2 +실현 결과).
7. Track A(리플레이 영상)로 B·R 실측: 첫 접촉 → 첫 킬 시간, 접촉 시 참여자 거리.

## 8. 파일 지도

| 무엇 | 어디 |
|---|---|
| 근거 전문 | `docs/DEFINITION_EVIDENCE.md` (1~19절), 문헌 `docs/DEFINITION_LITERATURE.md`, 서지 `docs/references/definition_refs.bib` |
| 논문 정의 절 초안 | `docs/references/definition_section_draft.tex` (= `paper/tog/`) |
| 경계 spec (현행 파이프라인 산출) | `config/fight_boundary/` (전체 코퍼스), `sample_3000/` (파일럿) |
| 정의 파이프라인 | `analysis/{kill_pairs,temporal_boundary,spatial_boundary,ability_range,boundary_spec,fight_boundary_pipeline}.py`, `scripts/run_fight_boundary_pipeline.py`, `scripts/compare_boundary_specs.py` |
| 게이트·라벨 실험 | `scripts/run_presence_gate_points.py`, `summarize_presence_gate_points.py`, `run_prediction_situation_pilot.py` |
| v3 구축 | `scripts/build_corpus_v3.py` → `build_corpus_shard.py --extra-labels` → `run_scale_decomposition.py --y-key --teamfight-min 4` |
| 산출물 | `D:/LOL_Project/fusion_2615/corpus_shards_v3_mlex/`, `features/scale_decomposition_v3_*.json`, `runs_corpus_v3/`, `runs_presence_gate/`, `features/fight_boundary*/` (목록: `docs/DATA_MANIFEST.md`) |
