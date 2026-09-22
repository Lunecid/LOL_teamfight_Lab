# 입력 특징 전수조사 (FEATURE CENSUS)

`tools/gen_feature_census.py`가 `docs/`의 매니페스트에서 생성한다 (손으로 고치지 않음). `outputs/`, `data/raw/`, 체크포인트, 대형 NPZ/joblib은 열지 않았다 (`AGENTS.md`). 모든 수는 아래 출처의 필드를 전사한 값이며 자료에서 다시 계산한 값이 아니다.

| 출처 | 제공하는 것 |
|---|---|
| `docs/V_FEATURE_MANIFEST_RUNTIME_20260919.md` | 실행 폭, 그룹별 열 수, `state_value_v2.py` hash16 |
| `docs/supplementary_e1_20260921/feature_groups.json` | 열 이름 전체, 그룹 배정, E1 arm 구성 |
| `docs/V_MODEL_INPUT_DESIGN_20260919/STATEV2_REFERENCE_FEATURE_SET.json` | 열별 자료형·원천 문자열, 차원 사전 |
| `docs/A_MLP_expanded_evaluator_meta_20260919.json` | 동결 평가기의 전처리 차원 |
| `scripts/rr20260920_q_newv_primary_fit.py` | q·LightGBM 후보의 입력 조립 코드 |

## 1. 두 특징 경로의 구분

이 저장소에는 서로 다른 두 특징 경로가 있고, 열 수를 혼동하는 원인이 여기에 있다.

| | StateV2 | 탭·시퀀스 표현 |
|---|---|---|
| 구현 | `gameplay/state_value_v2.py` (별도 worktree `worktrees/engagement-state-value/`) | 저장소의 `gameplay/{pipeline_cache,pipeline_interp,features,feature_spatial,anchors}.py` |
| 폭 | 원 362 → Expanded 361 → 수치 351 | 기본 1,015 × 통계 7 = 7,105 (+1) |
| 쓰임 | 승률 평가기 V-hat, SVI 라벨, 방향 예측기 q | 교전 승자 라벨(`market_event`) 아래의 정의 민감도·누출 절제 |
| 경기 시각 | `time_minutes`, `time_minutes_sq` (절대 분) | `time_norm` = t / 45분, 1에서 절단 (`TIME_NORM_ABSOLUTE`) |
| 좌표·지도 앵커 | 없음 | 있음 (`feature_spatial.py`, `anchors.py`, `config/game_rules/map_anchors.json`) |
| 프레임 나이 | `snapshot_age_s`를 기록하고 입력에서 제외 | `frame_age_s`를 기본 특징으로 포함 (`TAB_FRAME_AGE_FEATURE`) |

`TIME_NORM_ABSOLUTE`(45분 정규화)와 `ANCHORS_CAUSAL`(컷오프 이전 파괴만 반영하는 지도 앵커)은 **탭·시퀀스 경로의 설정**이다 (`core/config.py` L437–L442, `core/presets.py` L28). StateV2에는 좌표·앵커·거리·구역 열이 없으므로 이 두 설정은 351열의 성질이 아니다.

## 2. 폭의 대조

| 폭 | 구성 | 쓰임 | 출처 필드 |
|---|---|---|---|
| 362 | Expanded 361 + `snapshot_age_s` | 감사 기록 | `Raw width` |
| 361 | 수치 351 + 챔피언 ID 10 | V-hat 입력, LightGBM 폭 | `Expanded`, `preproc_dims.lgbm_width` = 361 |
| 351 | 챔피언 ID 제외 | q의 수치 블록 | `dimensions.numeric_expanded`, `preproc_dims.n_numeric` |
| 352 | 수치 351 + p_pre | `logit_state` (q) | `rr20260920_q_newv_primary_fit.py` L236–L241 |
| 362 | Expanded 361 + p_pre | `lgbm_state` (진단 후보) | 같은 스크립트 L243–L245 |
| 257 | 수치 351 − 팀 사건 × 시각 94 | Core267 절제 | `dimensions.numeric_core` |
| 7,105 | 탭 기본 1,015 × 통계 7 | 정의 민감도·누출 절제 | `INPUT_FEATURE_AUDIT_V3.md` |

두 학습기의 입력은 다르다. `logit_state`는 352열, `lgbm_state`는 362열이며, 차이는 챔피언 ID 10열이다. 그래서 학습기 간 정합 비교는 이 짝으로 답해지지 않는다.

실행 매니페스트와 협력자 재구성 참조는 이름과 순서가 정확히 일치한다 (`Order exact match: True`, only-in-runtime 0, only-in-reference 0). `state_value_v2.py` hash16 = `418ff787b5cddd2a`이며 동결 평가기 메타의 `meta.state_value_v2_sha256_16`과 같다.

## 3. 블록별 전수

| 블록 | 슬롯 × 항목 | 열 | 자료형 | 원천 문자열 |
|---|---|---|---|---|
| 경기 시각 (`clock`) | 1 × 2 | 2 | 수치 | query_ms; absolute clock, never final match duration |
| 자료 품질 (`quality`) | 1 × 1 | 1 | 수치 | observed event metadata; diagnostic inclusion to be audited |
| 참가자 상태 (`player_snapshot`) | 10 × 9 | 90 | 수치 | last observed participant frame; norm formula must be recovered from source |
| 참가자 사건 (`player_event`) | 10 × 7 | 70 | 수치 | events no later than query; buff-history proxies, not exact active buffs |
| 팀 사건 (`team_event`) | 2 × 47 | 94 | 수치 | observed cumulative/acquisition history; no exact active-buff claim |
| 팀 사건 × 시각 (`team_time_interaction`) | 2 × 47 | 94 | 수치 | 팀 사건 × `query_ms`/1800000 ms |
| 챔피언 ID (`champion`) | 10 × 1 | 10 | 범주 | pre-known roster; arbitrary identifier, not an ordinal value |
| **수치 소계** | | **351** | | |
| **합계** | | **361** | | |

### 3.1 경기 시각 (`clock`, 2열)

`time_minutes` `time_minutes_sq` 

원천은 `query_ms; absolute clock, never final match duration`이다. 절대 시각이며 경기 총 길이로 정규화하지 않는다. 탭 경로의 `time_norm`(45분 분모)과 다른 열이다.

### 3.2 자료 품질 (`quality`, 1열)

`unknown_objective_team_count` 

원천은 `observed event metadata; diagnostic inclusion to be audited`이다. E1의 Q0가 이 열이다.

### 3.3 참가자 상태 (`player_snapshot`, 10 슬롯 × 9 = 90열)

슬롯마다 같은 항목을 갖는다. 항목:

- `totalGold_norm`
- `curGold_norm`
- `level_norm`
- `xp_norm`
- `hp_pct`
- `mp_pct`
- `alive`
- `laneCS_norm`
- `jgCS_norm`

원천은 `last observed participant frame; norm formula must be recovered from source`이다. `_norm` 접미의 정규화 상수는 매니페스트에 기록되어 있지 않다.

### 3.4 참가자 사건 (`player_event`, 10 슬롯 × 7 = 70열)

- `kills`
- `deaths`
- `death_since_snapshot`
- `death_last_30s`
- `death_age_minutes`
- `baron_death_since_acquisition`
- `elder_death_since_acquisition`

원천은 `events no later than query; buff-history proxies, not exact active buffs`이다. 바론·장로 관련 열은 활성 버프의 정확한 상태가 아니라 획득 이후 사망 여부의 대리 지표이다.

### 3.5 팀 사건 (`team_event`, 2 진영 × 47 = 94열)

진영 접두 `blue_`, `red_`가 같은 항목 집합을 갖는다. 항목:

- `atakhan`
- `atakhan_age_minutes`
- `atakhan_ever`
- `baron`
- `baron_acquired_last_120s`
- `baron_acquired_last_180s`
- `baron_acquired_last_300s`
- `baron_acquired_last_60s`
- `baron_age_minutes`
- `baron_ever`
- `dragon_AIR`
- `dragon_CHEMTECH`
- `dragon_EARTH`
- `dragon_FIRE`
- `dragon_HEXTECH`
- `dragon_OTHER`
- `dragon_WATER`
- `dragons`
- `elder`
- `elder_acquired_last_120s`
- `elder_acquired_last_180s`
- `elder_acquired_last_300s`
- `elder_acquired_last_60s`
- `elder_age_minutes`
- `elder_ever`
- `herald`
- `herald_age_minutes`
- `herald_ever`
- `horde`
- `horde_age_minutes`
- `horde_ever`
- `inhibitor_kills`
- `kills`
- `plates`
- `soul_AIR`
- `soul_CHEMTECH`
- `soul_EARTH`
- `soul_FIRE`
- `soul_HEXTECH`
- `soul_OTHER`
- `soul_WATER`
- `soul_event_recorded`
- `tower_BASE_TURRET`
- `tower_INNER_TURRET`
- `tower_NEXUS_TURRET`
- `tower_OTHER`
- `tower_OUTER_TURRET`

원천은 `observed cumulative/acquisition history; no exact active-buff claim`이다.

### 3.6 팀 사건 × 시각 (`team_time_interaction`, 94열)

팀 사건 94열 각각에 `query_ms`/1800000 ms (= 30분)를 곱한 파생 열이며, 부모 집합은 팀 사건 블록과 정확히 일치한다 (47개 항목, 두 진영). 예: `blue_kills_x_time` = `blue_kills` × `query_ms`/1800000. 분모는 탭 경로의 45분이 아니라 30분이다.

### 3.7 챔피언 ID (`champion`, 10열)

원천은 `pre-known roster; arbitrary identifier, not an ordinal value`이다. 범주형이며 q에서 제외한다. 동결 평가기는 어휘 172의 임베딩으로 읽는다 (`preproc_dims.emb_vocab` = 172).

## 4. 배제된 열

| 열 | 어디서 배제 | 근거 |
|---|---|---|
| `snapshot_age_s` | 원 362 → Expanded 361 | `V_FEATURE_MANIFEST_RUNTIME` `Expanded (drop snapshot_age_s)`; `feature_groups.json` `excluded[0]`; `STATEV2_REFERENCE_FEATURE_SET` `required_audit_metadata` |
| 챔피언 ID 10열 | Expanded 361 → 수치 351 (q) | `feature_groups.json` `excluded[1]`, `notes[2]`; `rr20260920_q_newv_primary_fit.py` L236 |

`snapshot_age_s`는 기록되고 감사 메타데이터로 요구되지만 어느 모형의 입력도 아니다. 7.3절의 관측 갱신 층화는 이 기록된 값으로 결과를 나누어 본다.

입력에 속하지 않는 값(`not_in_value_input`): `match_id`, `final_winner`, `final_duration`, `SVI`, `p_pre_from_this_same_V`, `engagement_endpoint`, `future_participation`, `future_interpolated_positions`.

감사 메타데이터로 요구되는 값(`required_audit_metadata`): `query_ms`, `snapshot_ms`, `snapshot_age_s`, `source_event_max_timestamp`, `source_commit`, `state_version`, `feature_order_hash`.

## 5. 없는 것

361개 열 이름 전체를 좌표·위치·거리·구역·앵커·프레임 나이·`time_norm`·룬·아이템·밴·피해·주문 패턴으로 검색한 결과 일치하는 이름은 0개이다. 따라서 다음은 이 표현에 없다.

- 챔피언 좌표, 싸움 중심 좌표, 팀 분산·대치 지표
- 지도 앵커까지의 거리, 구역(zone) 지시자, 타워 사거리 안 여부
- 마지막 프레임의 나이를 입력으로 쓰는 열
- 룬, 아이템 해시, 챔피언 스탯, 누적 피해, 밴

이 항목들은 탭·시퀀스 표현의 기본 특징 1,015개에는 있다 (`INPUT_FEATURE_AUDIT_V3.md` §1). 두 경로를 한 표현으로 서술하면 안 된다.

## 6. E1 정보군 arm과의 대응

| arm | X의 블록 | X 열 | p_pre | 차원 |
|---|---|---|---|---|
| M-F0 | 경기 시각 + 자료 품질 | 3 | 포함 | 4 |
| M-F1 | 경기 시각 + 자료 품질 + 참가자 상태 | 93 | 포함 | 94 |
| M-F2 | 경기 시각 + 자료 품질 + 참가자 사건 + 팀 사건 + 팀 사건 × 시각 | 261 | 포함 | 262 |
| M-F3 | 위의 전부 (= 수치 351열) | 351 | 포함 | 352 |

F = 참가자 상태 90열, E = 참가자 사건 70 + 팀 사건 94 + 팀 사건 × 시각 94 = 258열. M-F3의 352열은 q의 입력 열 수와 같고, 적합 가중치의 정규화와 학습기가 다르므로 절대 Brier는 같은 척도가 아니다.

`feature_groups.json`의 메모: p_pre is shared across arms (conditioning on win-prob summary). / M-F0 is not identical to PT_flex. / Champion IDs excluded to match current q numeric policy..
