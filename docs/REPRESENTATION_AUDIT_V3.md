# 표현 감사 v3: JSON 필드에서 모델 입력까지 (2026-09-09)

질문: Match-V5 JSON에 있는 것이 캐시 → 시퀀스 → 탭 열 → 학습기까지 **빠짐없이, 제때, 올바른
값으로** 들어가는가. 도구: `scripts/audit_representation.py` (네 단계), 결과
`features/representation_audit_v31.json`(수정 전) / `representation_audit_v31_fixed.json`(수정 후).
입력 피처의 누수 감사는 `docs/INPUT_FEATURE_AUDIT_V3.md`가 다루고, 이 문서는 "제대로 들어가는가"를
다룬다.

## 0. 결론

- 원천 필드는 의도한 것이 모두 캐시로 들어간다. 캐시가 쓰지 않는 원천 필드는 경기 후
  집계(detail의 challenges 132개, 최종 아이템, 핑, K/D/A, win)와 몇 개의 부가 이벤트뿐이며,
  경기 후 집계는 정의상 컷오프에서 알 수 없으므로 **쓰면 안 되는 것**이다.
- Riot API가 항상 0으로 주는 챔피언 스탯 6개(abilityHaste, armorPen, bonusArmorPenPercent,
  bonusMagicPenPercent, cooldownReduction, physicalVamp)는 죽은 열이다(60열 × 7 통계 = 420열).
- 정렬 검증에서 결함 2개를 찾아 고쳤다. (1) 프레임 유지 특징이 의도보다 **한 프레임 더
  오래된** 값을 읽었다. (2) 절대 time_norm 수정이 실제 경로에 적용되지 않고 있었다. 수정 후
  프레임 특징 98.7% 정확(나머지 1.3%는 컷오프 전 2.5 s 안에 프레임이 들어온 경우로 구간 중심
  의미상 정상), 이벤트·time_norm·라벨 100%.
- 식별자 열(챔피언·주문·룬·아이템 해시·밴, 2,240열)이 숫자로 들어간다. 트리는 ID 값의
  크기로 분기하므로 표현이 나쁘다. v3.2 분해에서 LightGBM 범주형 처리(`--categorical`)를
  변형으로 잰다.

## 1. 원천 → 캐시: 필드 추적

**timeline `participantFrames`** (프레임 60 s)

| 원천 필드 | 캐시 특징 | 비고 |
|---|---|---|
| position.x/y | x_norm, y_norm (+ xy_raw_minute) | 특징에서는 중심 상대, 공간 블록은 절대 |
| level, xp | level_norm, xp_norm | /18, /28,000 |
| currentGold, totalGold | curGold_norm, totalGold_norm | /6,000, /35,000 |
| goldPerSecond | **미사용** | gps_norm은 totalGold / 경과 시간으로 파생 |
| minionsKilled, jungleMinionsKilled | laneCS_norm, jgCS_norm | /400, /250 |
| timeEnemySpentControlled | ccTime_norm | /600 |
| championStats 25 | cs_* 25 | 6개는 API가 항상 0 |
| damageStats 12 | ds_* 12 | 누적값 |

**timeline events**

| 이벤트 | 캐시·특징 | 비고 |
|---|---|---|
| CHAMPION_KILL (killer, victim, assisting, bounty, shutdownBounty, killStreakLength, position, victimDamageDealt/Received, victimTeamfightDamageDealt/Received) | 탐지·라벨·이벤트 집계(kills, bounty, shutdown, killstreak) | victimDamage*는 오프너 분석에만, **victimTeamfightDamage\*는 미사용**(신규 필드, 후보) |
| CHAMPION_SPECIAL_KILL (killType, multiKillLength) | multikill, ace, 라벨 보너스 | |
| BUILDING_KILL, TURRET_PLATE_DESTROYED | tower/inhib/plate 집계, 누적 차이, 앵커(τ 이전), 가격 라벨 | |
| ELITE_MONSTER_KILL (monsterType, subType, killerTeamId) | dragon/baron/herald/atakhan/horde 집계·누적, 영혼(4번째 용 서브타입에서 파생), 버프 잔여, 가격 라벨 | |
| DRAGON_SOUL_GIVEN (name) | 미사용 (영혼은 용 처치에서 파생) | 빌더는 dragonSoul/soulType 키를 찾으나 API 키는 name; 파생 경로가 있어 무해 |
| WARD_PLACED / WARD_KILL (wardType) | ward_placed/kill, control_ward_* 집계, 가격 라벨(와드 킬) | |
| ITEM_PURCHASED / SOLD / UNDO | item_pur/sold/undo 집계, 아이템 해시 16 | ITEM_DESTROYED 미사용 |
| SKILL_LEVEL_UP | ult_level_norm | |
| LEVEL_UP | 미사용 (레벨은 프레임) | |
| GAME_END (winningTeam) | 경기 승패(M0/M1/M2 층 전용) | 교전 특징에는 안 들어감 |
| FEAT_UPDATE, OBJECTIVE_BOUNTY_*, PAUSE_END | 미사용 | FEAT_UPDATE(2025 위업)는 후보 |

**detail** (경기 단위)

| 원천 | 캐시·특징 | 비고 |
|---|---|---|
| participants.championId, championName | champion_id, champion_name_id | |
| summoner1Id/2Id | summoner_spell_1/2_id | |
| perks.styles, statPerks | primary/sub style, rune 6, stat perk 3 | |
| teamPosition/role | role_slots (역할 순서) | |
| teams.bans | blue/red_ban_0..4 | |
| challenges(132), item0-6, 핑, kills/deaths/assists, win, 피해 총계 | **미사용, 의도적** | 경기 후 집계 = 컷오프에서 알 수 없음 |

## 2. 캐시 건강 (400경기)

- 항상 0 (죽은 필드): cs_abilityHaste, cs_armorPen, cs_bonusArmorPenPercent, cs_bonusMagicPenPercent,
  cs_cooldownReduction, cs_physicalVamp.
- 0이 99.9% 이상이지만 희귀 사건이라 정상: has_elder, elder_remain_norm, soul_chemtech.
- 경기 안에서 변하지 않는 것: 밴 10(정상). 그 외 노드 특징은 모두 변한다.

## 3. 탭 열 (v3.1 부분 샤드, 67,829행 × 7,106열)

| 군 | 열 | 상수 열 | 0이 99% 이상 |
|---|---|---|---|
| 플레이어 챔피언 스탯 | 1,750 | 816 | 972 |
| 플레이어 아이템 해시 | 1,120 | 0 | 181 |
| 플레이어 스냅숏 | 910 | 230 | 236 |
| 플레이어 피해 | 840 | 240 | 240 |
| 플레이어 룬 | 770 | 330 | 330 |
| 플레이어 상태 | 770 | 310 | 690 |
| 이벤트 집계 | 308 | 36 | 122 |
| 플레이어 식별자 | 280 | 120 | 120 |
| 공간 | 175 | 42 | 100 |
| 전역 팀 차이 | 105 | 0 | 9 |
| 밴 | 70 | 30 | 30 |
| time_norm | 7 | 0 | 0 |
| frame_age_s | 1 | 0 | 0 |

상수 열 2,154. 완전 상수 기본 특징 66 = 죽은 스탯 6 × 10 슬롯 + pos_* 6. 고정 특징(룬·식별자·밴)의
mean/std/min/max/delta/slope는 정의상 상수(last만 정보). 식별자 열 2,240개가 숫자다.

## 4. 학습기가 받는 것

| 프로토콜 | 열 | 제거 규칙 |
|---|---|---|
| ToG 분해 | 4,891~4,952 (v3.1: 상수 제거 후) | 상수 열만 |
| CoG LightGBM (v3 실행) | 3,050 | 상수 960 (밴 60, 식별자 240, 룬 660) + 준상수 1,362 (아이템 해시 960, 상태 360, 공간 42) + 상관 0.98 |

CoG 경로 3,050열의 구성: 챔피언 스탯 1,172, 스냅숏 489, 피해 476, 이벤트 250, 상태 164, 아이템 160,
공간 111, 룬 110, 팀 차이 65, 식별자 40, 밴 10, 시간 3.

## 5. 정렬 검증 (실제 교전 101개, 40경기)

| 검사 | 수정 전 | 수정 후 |
|---|---|---|
| 마지막 구간의 프레임 유지 특징 = τ 이하 마지막 프레임 값 (6 특징 × 10 슬롯) | 2,255 / 6,000 (38%) | 5,982 / 6,060 (98.7%) |
| 이벤트 구간 합 = 창 안 원시 사건 수 | 100% | 100% |
| time_norm = 구간 중심 / 45분 | 0% | 100% |
| frame_age_s = τ − 마지막 프레임 | 98% | 98% |
| 이 경로의 라벨 = 샤드 라벨 | 100% | 100% |

**결함 1.** `interpolate_node_global`이 질의 시각을 τ 이하 마지막 프레임 시각으로 캡한 뒤
`searchsorted(side="left") − 1`로 프레임 인덱스를 잡아, 정확히 프레임 시각에 캡된 질의(컷오프 뒤
구간의 대부분)가 **그 앞 프레임**을 가리켰다. 스칼라(ffill)는 60~120 s 전 프레임을, 좌표(α = 1)는
맞는 프레임을 읽어 서로 어긋났다. `side="right"`로 고쳤다.

**결함 2.** `build_ms_sequence`가 노드 보간 뒤 전역 벡터를 `global_from_prev_snapshot`으로 교체하므로
보간 함수 안에서 쓴 절대 time_norm이 캐시의 경기 길이 상대값으로 덮였다. 교체된 벡터 위에서 다시
쓰도록 고쳤다.

남은 1.3%: 마지막 구간의 상태는 구간 중심(τ − 2.5 s)에서 읽으므로 τ 전 2.5 s 안에 도착한 프레임은
아직 보지 않는다. 구간 의미상 정상이며 문서화한다. `tests/test_frame_alignment.py`.

## 6. 결정과 남은 것

- 죽은 스탯 6개는 상수 제거로 자동 탈락. 캐시 v4에서 제외 예정.
- 식별자 열: v3.2 분해에서 `--categorical` 변형(champion·spell·rune·item hash·ban의 last 열을
  LightGBM 범주형으로)의 AUC 차이를 잰다. 개선이 있으면 기본으로 채택.
- 후보 신규 입력: victimTeamfightDamageDealt/Received(킬 이벤트의 새 필드; 라벨·오프너 분석용,
  τ 이전 킬에 한해 특징 가능), FEAT_UPDATE(위업), goldPerSecond(현재는 파생값).
- v3.1(결함 포함)로 만들던 샤드는 폐기했고 v3.2(`corpus_shards_v32_mevent/`)로 재구축 중.
  v3(누수 포함) → v3.2 AUC 차이가 누수와 결함의 크기다.
