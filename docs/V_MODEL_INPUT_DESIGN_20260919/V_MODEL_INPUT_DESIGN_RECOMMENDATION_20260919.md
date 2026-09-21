# V 모형·입력 설계 권고안

작성: 2026-09-19
검토 브랜치: `feature/fight-boundary-pipeline`
검토 커밋: `154caf9711b436d6d4b860533971c60c9f071993`
상태: **새 설계 제안. 원격 변경·실제 경기자료 재학습·성능 재현을 수행하지 않음.**

## 0. 목적과 범위

현재 단계의 목표는 실제 최종 경기 승패 W에 대해 시간조건부 승률 V를 적합하는 것이다. V의 출력 변화로 SVI를 만들거나 q를 다시 학습하는 단계는 그 뒤다. 동일한 평가기 V=g∘f∘T와 동일한 as-of 상태 구성을 질의 시점, 교전 직전, 교전 종료시점에 적용한다.

새 커밋의 A0(Logistic)를 기준선·잠정 후보로 유지하되, 구현 문제가 확인된 짧은 이력 RNN/TCN의 결과로 정보 부족이나 학습기 열세를 확정하지 않는다. A0가 다른 후보보다 우수하다는 원고 주장은 통제된 재평가의 범위로 제한한다.

## 1. 실제 소스와 제안의 구분

### 이번 푸시에서 확인한 자료

- `docs/V_REDESIGN_MODEL_DESIGN_SETTINGS_20260919.md`
- `scripts/rr20260919_v_redesign_fit_wave2.py`
- `scripts/rr20260919_v_redesign_fit_wave3_tier23.py`
- `scripts/rr20260919_v_redesign_continuity_wave2.py`
- wave-2, wave-3, continuity ledgers

### 기존 StateV2 계보 확인에 사용한 자료

커밋 `e0ec3d01496c0fa8f808d305fce18c3951ca9a29`의 다음 파일:

- `gameplay/state_value.py`
- `gameplay/state_value_v2.py`
- `train/state_value_experiment.py`
- `train/temporal_winprob.py`
- `scripts/fc20260915_data.py`

현재 브랜치의 스크립트는 로컬 data-root/scripts와 별도 worktree의 모듈을 불러온다. 현재 브랜치에서 `gameplay/state_value_v2.py`는 조회되지 않았다. 아래 361열 세부 구성은 위의 기존 StateV2 생성 코드로 재구성한 **참조 스키마**이며, 실제 실행의 이름·순서·소스 해시와 일치 확인 전에는 fitted manifest로 취급하지 않는다.

## 2. 우선 수정할 입력 처리

| 문제 | 코드상 근거 | 조치 |
|---|---|---|
| 오른쪽 정렬 이력에 prefix-length packing 사용 | `build_history_stacks`는 유효 관측을 뒤에 배치; RNN은 `pack_padded_sequence(x, lengths)` | RNN 입력을 유효 관측 앞쪽·패딩 뒤쪽으로 재배열하거나 `pack_sequence` 사용 |
| TCN의 마지막 시점 선택 불일치 | 오른쪽 정렬인데 `mask.sum()-1`을 인덱스로 사용 | 실제 valid index를 계산; 출력 시간축 정렬 및 tail crop 확인 |
| 희소 bucket rows를 과거 minute frames처럼 사용 | wave-3은 `bucket_only=True` 결과 자체를 스택 | 감독학습 질의 샘플과 이력 저장소를 분리 |
| H5의 교전 평가가 현재 as-of 상태를 사용하지 않음 | `history_at_times`는 마지막 bucket history만 반환 | 직전·종료시점의 실제 `X_pre`/`X_post`를 현재 토큰으로 포함하는 공통 builder |
| 챔피언 ID 처리 불일치 | Logistic categorical pipeline, current LGBM category; RF/MLP/H/S에는 float ID | one-hot 또는 명시적 categorical/embedding adapter |
| V_CAL 역할 중복 | early stop와 calibration에 V_CAL 사용 | TRAIN 내부 경기 분리로 early stop; V_CAL은 refit 이후 calibration만 |

### 합성 입력 확인

`[PAD,PAD,PAD,10,20]`과 length=2를 packing하면 실제 입력은 `[PAD,PAD]`다. `[10,20,PAD,PAD,PAD]`로 바꾸면 `[10,20]`이 들어간다. 동봉한 `synthetic_checks.json`은 이 API 동작을 실제 PyTorch로 확인한 기록이다. 경기자료에서 재학습한 결과가 아니다.

TCN의 현재 Conv1d 설정에서는 길이가 5→7→11→19로 늘어난다. 길이 증가 자체가 미래 누수를 뜻하지는 않지만, valid index 선택과 출력 시간축을 함께 검사해야 한다.

## 3. 기본 특징 어휘: 361개의 의미를 먼저 분리

| 블록 | 참조 열 수 | 내용 |
|---|---:|---|
| 플레이어 프레임 | 90 | 10명×9: totalGold, curGold, level, xp, hp, mp, alive, laneCS, jgCS |
| 플레이어 사건·이력 | 70 | 10명×7: kills, deaths, death_since_snapshot, death_last_30s, death_age, Baron/Elder 이후 사망 이력 |
| 팀 사건·구조물·목표물 | 94 | 팀당 47개 |
| 시간·품질 | 3 | time_minutes, time_minutes_sq, unknown_objective_team_count |
| 팀×시간 파생항 | 94 | 위 팀 특징×query_ms/1800000 |
| 챔피언 ID | 10 | 시작 로스터의 식별자; 순서형 실수가 아님 |
| 합계 | 361 | 수치 351+범주 10 |

스냅샷 age를 포함한 참조 상태는 362열이고 기존 expanded adapter가 이를 제외해 361열이 된다. age는 metadata로 보존한다. 수치 normalization의 정확한 분모는 별도 원천 코드/manifest를 확인해야 한다. `alive`는 이 참조 코드에서 프레임 값이므로, 이를 현재 시점의 정확한 생존 상태로 과장하지 않는다. buff 열도 획득·사망 이력이지 정확한 active buff가 아니다.

### 두 특징 profile

- **Expanded361 (주 비교):** 수치 351+범주 10. 기존 특징을 유지하되 자료형과 시간 처리를 교정한다.
- **Core267 (ablation):** 수치 257+범주 10. 94개 팀×시간 파생항만 제거한다. 정보원은 추가하지 않는다.

Core267이 항상 더 좋은 입력이라고 가정하지 않는다. 같은 학습기에서 Core267/Expanded361을 비교하여 명시적 시간 상호작용의 효용을 평가한다. 처음부터 중요한 상태 정보를 추가로 잘라 차원을 맞추지 않는다.

## 4. 학습기별 변환 후 차원

| 모델 | Expanded361 | Core267 |
|---|---|---|
| Logistic | 351+Σ슬롯별 TRAIN one-hot 수 | 257+Σ슬롯별 TRAIN one-hot 수 |
| RF | Logistic과 같은 one-hot 어휘 권장 | 동일 |
| Native LightGBM | 수치 351+명시적 범주 10=361 | 수치 257+범주 10=267 |
| Embedding MLP | e=8일 때 351+10×8=431 | 257+10×8=337 |

범주 어휘는 TRAIN에서 고정하고 UNK 처리를 명시한다. 챔피언 embedding은 공통 표를 사용하되 팀·슬롯별 배치를 유지한다. 참가자 ID순 슬롯을 TOP/JG/MID/BOT/SUP의 검증된 역할처럼 해석하지 않는다.

알고리즘의 엄격한 동일변환 비교가 필요하면 Logistic과 MLP에 같은 TRAIN one-hot 행렬을 먼저 제공한다. 그 다음 native LGBM/embedding MLP를 각 자료형에 맞춘 표현 비교로 별도 표시한다. 원천 정보 동일성과 변환 행렬 동일성은 다르다.

## 5. 시간 이력의 공통 계약

1. 학습의 감독 대상 질의는 기존 bucket 샘플을 유지할 수 있다.
2. 그러나 이력을 가져오는 원천은 샘플된 감독 행이 아니라 해당 경기의 전체 관측 저장소다.
3. 현재 토큰은 정확한 query_ms에서 구성한 as-of state다. 해당 시점까지 사건을 반영하고 자원은 마지막 관측을 사용한다.
4. 과거 K-1개는 현재 토큰이 사용하는 프레임보다 앞선 서로 다른 실제 프레임에서 가져온다. 같은 프레임을 carry-forward한 값을 새로운 관측으로 중복 계상하지 않는다.
5. K는 현재를 포함한 토큰 수다. K=5가 무조건 5분 구간을 의미하지 않으며 실제 timestamp와 lag를 기록한다.
6. 각 토큰에 `(token_timestamp-query_ms)/60000`을 제공하고 mask는 별도로 보존한다.
7. source missingness와 history padding은 다른 마스크다. scaling 뒤 padding 값을 다시 0으로 만들고 padding을 통계 적합에 넣지 않는다.
8. 학습 질의와 교전 pre/end 모두 같은 history builder를 호출한다. query batch 구성이 바뀌어도 동일한 query의 입력은 같아야 한다.
9. 미래 사건·미래 프레임 변경은 현재 입력을 바꾸지 않아야 한다.
10. 2분 이전의 target query 제외와 이전 이력 사용 가능성은 별개의 선택으로 기록한다.

### 권장 이력 차원

Expanded 수치 351 + 상대시간 1 = 토큰당 352. 챔피언 10개는 시간축 밖에 한 번 보관한다.

- sequence: `[B,K,352]`, mask `[B,K]`, champions `[B,10]`
- flat native LGBM: `K×352 + K(mask) + 10(category)`
- K3: 1069열; K5: 1775열

Core profile을 사용하는 보조 비교라면 토큰당 258, flat은 K3=787, K5=1305다. 모든 profile을 처음부터 완전 조합할 필요는 없다.

## 6. 모델 설계 제안

### Phase A: 교정된 현재 상태 모델

- Logistic: standard numeric+TRAIN one-hot; C 후보는 고정된 소수 범위에서 탐색한다.
- LightGBM: native categorical; leaves 15/31/63×min_child 100/300의 6설정 출발안을 사용한다. lr .03, 최대 3000 trees, L2 5, feature fraction .8, row fraction .8, row subsampling을 사용할 때 `subsample_freq=1`을 명시한다. 선택은 TRAIN 내부 stop과 V_SELECT에서 한다.
- MLP: Expanded embedding 입력 431→256→128→1을 출발점으로 사용한다. dropout .1/.3, AdamW, batch 1024, 최대100 epochs, patience10. 폭·학습률의 제한된 사전 후보를 JSON에 기록한다.
- 기존 A0와 legacy는 출처를 구분해 계속 비교한다. 기존모델 제외 사유와 보정/가중 스케일 차이를 숨기지 않는다.

위 설정은 최적값 주장이 아니라 초기 탐색 설계다. 기존 CoG의 목표·표현에 맞춘 하이퍼파라미터를 V의 최적값이라고 부르지 않는다.

### Phase B: 이력의 추가 가치

현재 LightGBM과 같은 원천정보·범주형 처리를 유지한 H3/H5를 비교한다. H5와 A0만 비교해서 이력의 가치가 없다고 결론내리지 않는다.

GRU 출발안:

- 토큰 `[B,K,352]`→Linear352→128→GRU(hidden128,1 layer)
- 현재 수치351→Linear128의 우회 경로
- 정적 챔피언 embedding 10×8=80
- 결합폭=history128+current128+champion80=336
- head336→128→1

Transformer 보조안:

- d_model128, 4heads, 2layers, FFN256
- 상대시간 표현과 validity mask
- 같은 현재 우회 경로와 정적 챔피언 경로

현재 우회 경로는 이력을 압축하다 현재상태를 잃지 않게 하려는 제안이다. 성능 보장은 아니다. 기존 BiGRU/BiLSTM/TCN은 correctness를 수정한 뒤 진단 비교로 남긴다. 모두 재학습한 다음에야 최신 순위나 시간별 열세를 해석한다.

### Phase C: 선택적 팀 구조 모델 — 그래프 없이 가능한 확장

Core267과 같은 원천 정보를 재배치한다.

- 플레이어 수치 `[B,10,16]`
- 챔피언 embedding8을 붙여 플레이어당24
- shared MLP24→64→32
- 팀 내5명 평균 pooling: Blue32, Red32
- 두 표현과 차분: 96
- 팀별 사건47×2=94, 시간·품질3
- readout96+94+3=193→128→1

이는 참가자 ID 순서에 덜 의존하는 표현을 검토하기 위한 제안이다. pooling은 정보를 압축하므로 항상 우수하지 않다. 검증된 라인 역할이나 위치 엣지가 없어도 만들 수 있지만, GNN 실험을 했다고 표현하지 않는다. 물리적 Blue/Red 진영 차이가 있으므로 확률 반전 대칭을 무조건 강제하지 않는다.

## 7. 선정과 동결

- TRAIN 내부 경기 단위 stop split을 분리한다.
- iteration/epoch를 정한 뒤 전체 TRAIN으로 refit한다.
- V_CAL은 보정에만 사용한다. raw와 공유 positive-slope sigmoid를 명시적 후보로 비교한다.
- V_SELECT는 시간 균형 Brier, 같은 가중치의 logloss를 사용한다.
- 가중치의 평균1 정규화 여부를 고정한다. 특히 Logistic의 C와 weight scale을 함께 기록한다.
- TEST/external은 동결 뒤 보고한다. 이전 TEST 노출 이력 때문에 이 후속 연구는 탐색적이다.
- 최종 선택은 실제 W의 확률 품질을 우선한다. 구 V와 SVI 부호가 비슷하다는 이유로 모델을 고르지 않는다.
- 개발자료에서 경계·이벤트·snapshot-age·off-grid 적합성을 점검한다. 무조건 더 매끄러운 모델을 고르거나 실제 사건 변화를 평활화하지 않는다.
- 최종 V의 OOF 학습계보·전처리·보정·state/history builder를 함께 버전 고정한다.

## 8. 실행 전 필수 게이트

1. 실제 361개 column names와 참조 집합의 차이를 검사하고 실행 feature order hash를 보존한다.
2. categorical 10개가 어느 모델에서도 부지불식간에 연속 ID로 처리되지 않는다.
3. K=1 이력 입력은 현재 snapshot 입력과 정보상 대응한다.
4. K=5이고 유효2개인 합성 사례에서 두 실제 관측만 RNN에 들어간다.
5. TCN 출력 시간과 마지막 valid index가 맞는다.
6. 미래자료를 변조해도 query 입력·출력이 변하지 않는다.
7. 한 교전 pre/end 사이에 새 bucket이 없더라도 새 사건이 현재 as-of 토큰에 반영된다.
8. 같은 query를 단독·배치로 구성했을 때 입력과 예측이 동일하다.
9. 선정/보정/부트스트랩의 행·가중치·목표가 맞는다.
10. 런타임 import된 모듈 해시, 입력·모델·calibrator·vocab·history builder 해시를 모두 기록한다.

## 9. 결론

공유 시간조건부 V라는 방향은 유지한다. 361열을 무조건 늘리기보다 의미별 자료형을 먼저 보존하고, 현재 상태의 비교를 통제한 뒤 실제 관측 이력을 추가한다. 다음 완료 기준은 더 복잡한 모델의 승리가 아니라, 원천정보·입력 시점·인코딩·마스킹을 설명할 수 있는 재현 가능한 비교다.
