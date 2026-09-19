# 공통 실험과 원고별 배치·병렬 실행 장부

작성일: 2026-09-19  
상태: 실행·집필 작업 분담안. 이 문서 자체는 새 학습을 수행하지 않았다.  
공통 권한: [COMMON_RESEARCH_SPINE_20260919.md](COMMON_RESEARCH_SPINE_20260919.md)  
산출물 목록: [EXPERIMENT_INVENTORY_COHORT_20260919.md](EXPERIMENT_INVENTORY_COHORT_20260919.md)

## 1. 기본 원칙

공통 연구는 하나이고 원고가 둘이다. 동일한 자료·라벨·모델 결과를 반복 실행하거나 독립적 재현 결과처럼 중복 계상하지 않는다. 실행 단위는 evidence_id이며, 각 원고가 그 결과를 사용하는 위치를 별도로 기록한다.

## 2. 개선 목적 → 실험 → 원고 배치

| ID | 단계 | 질문·작업 | 저널 배치 | 학위논문 배치 | 로컬 산출 / 상태 |
|---|---|---|---|---|---|
| C00 | 출발점 | CoG 목적·기존 설계·한계와 후속 변경 구분 | 서론·관련연구 | 1–2장 상세 | succession lock · design map |
| C01 | I1 | 원자료·시간가용성·T 정의·사례 키·포함/제외 | 방법 핵심 | 3–4장 전체 경로 | 기존 계약/정의 문서; evidence_id 연결 필요 |
| C02 | I1/I4 | 경계·규모·정의 민감도와 공통/변경 사례 | 핵심 민감도·보조표 | 4장 설계 비교 | 재사용 우선; 범위별 실행 확인 필요 |
| C03 | I2 | **V 재설계** (공유 시간조건부 기본; 시간대 평가·선정·동결) | 결과 정의의 근거 | 5장 상세 | **진행 중** [V_REDESIGN_CONTRACT](V_REDESIGN_CONTRACT_20260919.md); 구 ledger `svi_v_time_strata` = old V |
| C04 | I2/I4 | ΔV̂ 연속성·무킬·V 사양 안정성 → 그다음 SVI | 측정·안정성 핵심 | 5장 | V-4 after V-3; 구 horizon flip은 old V |
| C05 | I3 | 공통 입력 학습기 비교와 주 q 동결 | 예측 주 표 | 6장 | **보류** until new SVI; 기존 primary/reselection = old V |
| C06 | I4 | 같은 행 q−PT, 전체/B40/B45·시간대 | 검증 주 표 | 7장 상세 | `svi_lift_localization` · `svi_state_dependent` (H) |
| C07 | I2/I4 | 물질적 결과↔SVI 대응·불일치 | 측정 검증 | 5·7장 | overnight concordance · validation_suite · cohort_aligned |
| C08 | I4 | 무킬 참고 구간 (Quiet) | 보조 검증 | 7장 | overnight quiet matched · validation type-A |
| C09 | I4 | 고정 V̂·q·PT 외부 적용 | 전이 절 | 7장 | `svi_transfer_2026` · overnight stage_transfer |
| C10 | 설명 | V̂ 차분 분해와 q 특징군 귀속 구분 | 필요 시 대표 | 5·6장 | 재사용 우선; 인과·정보 부재 주장 금지 |
| M01 | I3 심화 | C / C+F / C+E / 전체 정보집합 | 선택적 요약 | 6장 대표 심화 | **미실행** |
| M02 | I1–I4 심화 | 원자료→라벨→예측 사례 추적 | 보조자료 | 장 간·부록 | **미실행** |

‘기존 보고’는 inventory/REPORT에 적힌 수준이며, evidence_id로 묶기 전에는 원고에 “재현 완료”로 쓰지 않는다.

## 3. 공동 작업의 의존 순서

### S0. 공통 출발점과 계약 정리

CoG 목적, I1–I4, 용어, 사례·라벨·입력·분할, 모델 동결 계보를 spine에 연결. 두 원고가 같은 정의를 쓴다.

### S1. 기존 근거 정리와 병렬 집필 시작

재사용 가능 보고 vs 미확인 항목 구분. 저널: 서론·방법·핵심 결과 골격. 석사: 1–5장 설계 근거·자료 흐름. 새 모델 실행 없이도 가능.

### S2. 공통 핵심의 빈 부분만 보완

표본·가중치·키 일치, V̂/라벨 민감도, 주 q/PT, Quiet 조건 등. 완료 실험은 하나의 evidence_id로 두 원고에 연결.

### S3. 석사 전용 심화와 저널 정리 병행

M01/M02는 별도 진단 버전. 공통 정의·라벨을 바꾸지 않으면 저널 주 q 재선정 불필요.

### S4. 교차 검증과 최종 통합

같은 실험에 다른 숫자·분모·해석이 없는지 검사. 제출 일정은 이 계획에서 임의 고정하지 않음.

## 4. 필수 체크 게이트

| 게이트 | 통과 조건 |
|---|---|
| G0 목적 | PT/B40·특정 모델 성과가 연구 목적을 대신하지 않음 |
| G1 구성 | 사례 키·cutoff·endpoint·제외·target·input version 추적 가능 |
| G2 비교 | 평가 행·가중치·라벨 동일; 선택에 TEST/외부 미사용 |
| G3 해석 | 정의 타당성 vs 예측 성능, V̂ vs q, 정보 vs 학습기 구분 |
| G4 공유 | 두 원고의 동일 결과가 같은 evidence_id |
| G5 보고 | 미결·악화·노출 이력 보존; 미실행을 완료로 쓰지 않음 |

## 5. 문서·산출물 경로

```text
docs/
  COG_SUCCESSION_LOCK_20260919.md
  EXPERIMENT_DESIGN_MAP_20260919.md
  EXPERIMENT_INVENTORY_COHORT_20260919.md
  COMMON_RESEARCH_SPINE_20260919.md
  JOURNAL_RESEARCH_PLAN_20260919.md
  MASTER_THESIS_RESEARCH_PLAN_20260919.md
  SHARED_EXPERIMENT_MATRIX_20260919.md
  journal/          # 추후
  master_thesis/    # 추후
outputs/
  svi_*_20260919/   # 공통 frozen
  master_information_blocks_<version>/   # M01 실행 시에만
```

계획만으로 데이터·모델을 복제하거나 기존 frozen 결과를 덮어쓰지 않는다.

## 6. 현 단계에서 만들지 않는 약속

B40에서 반드시 PT를 이긴다는 약속, 큰 딥러닝 성능 향상, 특정 학술지 게재·학위 심사 통과, 임의 마감, 확증 지위, 실시간 전술 추천은 이 계획의 산출물이 아니다.

## 자료 근거

출처 범위는 `COMMON_RESEARCH_SPINE_20260919.md` §8. 이 표는 목적→실험·원고 배치이며 기존 실행의 새 인증이 아니다.
