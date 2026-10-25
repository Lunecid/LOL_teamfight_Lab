# 다음 EPT 학습 계획 · 단일 pilot 승인

as_of: 2026-10-25 Asia/Seoul
state: SINGLE_PILOT_AUTHORIZED / IMPLEMENTATION_AND_REVIEW_BEFORE_START
related_task: [EPT-T001](../harness/tasks/EPT-T001-epoch-loss-logging.md)

## 2026-10-25 승인 갱신 — EPT-T002

저자는 CUDA·S·25%·patience2·최대30에폭의 **새 단일 학습**을 명시적으로 승인했다. 구현과 독립 검토를 마친 별도 프로필을 새 청결 checkout에서 한 번 시작한다. 세부 계약은 [EPT-T002](../harness/tasks/EPT-T002-single-pilot.md), 운영 안내는 [단일 학습 프로필](EPT_SINGLE_PILOT_KO.md)에 있다. 아래 T001 제안·미승인 문구는 승인 전 이력이며 이 갱신이 단일 pilot의 실행 범위를 확정한다.

이번 승인은 FIT·STOP 학습곡선과 STOP 선택/종료 확인까지다. DEV/Brier/B1 품질 비교, TEST/CAL/AUDIT, 전체 S4b 격자, 다른 seed/모델, 자동 재시도/Resume 및 공개 push는 포함하지 않는다. 기존 strict STOP 최소화, batch32/seed20261022/lr0.001/wd0.01/warmup10/native8을 유지한다. 완료 또는 실패 시 보고하고 멈춘다.

## 현재 승인과 완료한 근거

이번 승인은 에폭별 FIT/STOP 손실 JSON/CSV 구현·합성 검증·다음 계획 작성까지다. 실제 학습·Resume·최종 TEST·S4b/S6/S7 실행 승인이 아니다.
Task8 GPU 비용 프로필은 D:/LOL_Data/runs/20261025_c_ept_s4a_1, f867a51e582bd5f4011e6cc52b6f3476deb1b75a, 7/7·exit0으로 완료됐다. 다른 코드의 역사적 S CPU는 별도 보존, M CPU 미측정이다. 비용 측정 완료가 모델 품질 검증이나 G-S4a의 저자 결정을 대신하지 않는다.
[최종 비용 보고](D:/LOL_Data/runs/20261024_c_ept_task8_gpu_control_1/reports/report-v1-20261024T170239.587119Z/FINAL_REPORT_KO.md)와 [기존 학습 해석](D:/LOL_Data/runs/20261024_c_ept_task8_gpu_control_1/reports/learning-result-review-20261025/REPORT_KO.md)을 근거로 한다. 과거 원본은 새 기록 형식으로 소급 수정하지 않는다.

## 제안: S 모델의 25% 학습 1회로 먼저 확인

아래는 구체적인 검토안이며 설정 파일이나 실행 명령이 아니다. 저자의 선택 및 다음 실행 계약 검토 전 실제 실행하지 않는다.

| 항목 | 제안 | 근거/상태 |
|---|---|---|
| 목적 | S4b 전체 격자에 앞선 학습곡선·STOP 선택 및 최소 성능 점검 | 장기 수렴·최종 결과 확인으로 확대하지 않음 |
| 장치 / 크기 | RTX 5080 CUDA / S | S가 M보다 비용·GPU 여유에서 유리; 품질 우위 근거는 아님 |
| FIT / STOP | 기존 고정 역할의 25% proxy | 모집단·membership·라벨·시간경계 계약 유지 |
| seed / batch | 20261022 / 32 | 현재 비용 측정과 동일한 검토안; 다른 seed 반복은 후속 결정 |
| dtype / 스레드 | float32·기존 결정성 정책 / native 8 | 기존 runtime identity 유지 |
| optimizer / lr | 기존 AdamW, peak lr 0.001, weight_decay 0.01, warmup_steps 10 | 새 탐색 아님 |
| 종료 감시 | 기존 정규화·집계 계약의 STOP loss, 최소화 | FIT는 기울기, STOP은 선택/종료 |
| patience | 2회 연속 미개선 | 기존 구현·Task5 계약 유지안; 저자 변경 시 별도 결정 |
| 개선 규칙 | strict new_STOP < best_STOP | min_delta 독립 옵션 없음; 의미상 0, 동률은 미개선 |
| 최대 에폭 | 30 (안전 상한) | 고정 30회 수행이 아님. STOP 인내 소진 시 즉시 종료 |
| 예산 | 누적 trainer 시간 ≤24h, host private 표본 <40GiB | 기존 계약 유지; 측정은 연속 상한 보증이 아님 |
| 재시도/동시 실행 | 자동 재시도 0, 학습 1개 | 오류는 보고 후 정지. Strata OFF |
| 데이터 접근 | FIT·STOP 학습/선택, DEV 승인된 진단만 | TEST/CAL/AUDIT는 열지 않음 |

현재 규칙의 예: STOP가 5.0→4.9→4.9→4.95라면 2번째 에폭이 최선, 3/4번째 두 번 미개선으로 4번째에서 정지한다. 이는 규칙 설명용 합성 숫자이며 실제 연구 손실값이 아니다. 저장된 best_epoch는 0부터 시작하지만 사람이 보는 epoch 열은 1부터 시작한다.

최대 에폭은 스케줄·identity에도 들어간다. 현재 cosine 학습률의 total_steps=max_epochs×FIT배치수이므로 max_epochs를 2에서 30으로 바꾸면 학습 경로도 달라진다. 예전 2에폭 실행을 연장/Resume하거나 처음 두 에폭이 동일할 것이라 주장하지 않는다. 새 코드·설정·입력·환경으로 새 run identity를 만든다.

## 예상 비용과 안전 상한

S의 25% 비용 실측은 저장 전 평균 epoch 1,451.049초, 전체 2에폭 호출 2,904.908초였다. 동일 proxy·설정·계측 가정이면 10에폭 약4.03h, 30에폭 상한 약12.09h다. 실제 종료는 얼리 스톱에 따라 달라지며 이 추정은 신뢰구간/예약 시간 보증이 아니다. 로그 쓰기·스케줄 변화·호스트 경합도 비용을 바꿀 수 있다.

전체 FIT로 바로 확대하면 10에폭 S 약16.03h/M 약20.92h, 30에폭 S 약48.09h/M 약62.77h의 조건부 외삽이다. 현재 24h 제한에서 full FIT 30에폭은 성립하지 않는다. 전체 확대나 예산 상향을 자동 승인하지 않는다. 선택한 proxy에서 학습곡선과 최소 품질을 확인한 뒤 저자가 다음 범위를 정한다.

원 설계 7절은 GPU 탐색과 CPU 보고용 비트 재현을 구분한다. 이 CUDA pilot 제안은 탐색·진단이며 최종 보고 장치를 GPU로 자동 전환하지 않는다. 보고 장치 변경은 별도의 저자 결정이다.

## 기록과 해석

매 완료 에폭에서 숫자 FIT batch-mean loss와 STOP pooled evaluation loss, lr, epoch/index, seed/identity, best/early-stop 상태를 JSON/CSV로 남긴다. 완료된 head와 해시로 결속된 스냅샷만 완료 결과로 읽는다. 부분 쓰기는 진행 기록이지 성공 결과가 아니며, 손상은 이전 에폭으로 조용히 후퇴시키지 않는다. 완료 폴더의 `epoch_metrics.json`/`epoch_metrics.csv`와 `checkpoints.latest_metadata` 읽기 계약은 [손실 기록 안내](EPT_EPOCH_METRICS_KO.md)에 명시했다.

FIT 값은 학습 모드의 배치 평균 손실들의 평균이며 평가 손실이 아니다. STOP은 평가 모드·기존 경기/규모/블록 가중 계약의 집계다. 두 값을 단순 비교해 train-validation gap, 과적합, 일반화 향상을 단정하지 않는다. 각 곡선을 자신의 정의에 따라 별도로 제시한다. 얼리 스톱이 발동해도 일반화 성능이 입증된 것은 아니다.

손실곡선·선택 에폭·종료 이유·자원 기록을 매 결과 보고에 포함한다. 품질 확인은 기존 S4b의 60초 발생 Brier와 B1 비교 정의를 따른다. 원 설계 7–8절과 S4a 원 계획을 별도 원본 checkout에서 확인했다. 비교 자료는 15.21 DEV, 평가 가중은 경기당 1/n_m이며, 최종 주장의 세 seed 짝 비교·같은 경기 재표본 2,000회·95% 구간 규칙은 단일 pilot로 충족되지 않는다. 원 설계의 B2 대비 60초 Brier 보조 비교와 S4b의 B1(M2 발생 예측) 멈춤 점검을 혼동하지 않는다. 구체 Brier 평가 행/노출·scored 필터·B1 artifact와 예측 결속·학습곡선 평탄 판정 기준은 다음 S4b 작업서에서 고정해야 하며 이번 로깅 작업에서 새로 발명하지 않는다.

## 실행 전 순서와 관문

1. EPT-T001 로깅 구현·합성 검증·독립 검토 결과를 확정한다. 기존 v1/v2 비용 operator의 epochs=2 제약은 그대로다.
2. 저자가 장치·모델·patience/개선규칙·최대 에폭·25% pilot 및 최소 평가 범위를 결정한다. 확정안을 해당 실행 작업서와 결정 기록에 남긴다.
3. S4b 실행이 승인되면 별도 단계의 operator/profile·설정 identity·입력/평가 계약을 구현·검토한다. 비용 operator의 guard를 완화하거나 하위 trainer/runner 직접 호출로 우회하지 않는다. 이번 변경이 S4b 실행기를 이미 제공한다고 주장하지 않는다.
4. 검토된 소스를 커밋으로 고정한 새 청결 실행 checkout에서 입력·환경·경로·저장공간·Strata OFF·로그 경로를 Check한다. IDE의 사용자 주석/미커밋 파일은 실행 checkout으로 복사하지 않는다.
5. 저자가 승인한 Start만 수행한다. 새 에폭 결과의 JSON/CSV·그림을 확인하고, 실제 early_stopping/max_epochs/resource/failure를 구분해 보고한다. 누수·NaN/Inf·identity 불일치·손상·예산 초과 시 실패 증거를 보존하고 멈춘다.
6. 승인된 pilot 이후 저자와 논의한다. 27호출 S4b 전체 격자·3seed 변동·S5 잠금·S6 탐색·S7 반복·S8 TEST1회는 각각 기존 관문을 따른다.

## 저자에게 남은 선택

- 탐색·진단용 CUDA·S 모델·25% 단일 seed pilot에 동의하는가? 최종 보고 장치 정책은 따로 결정한다.
- 기존 patience2/엄격 개선을 유지하고, 최대30에폭을 안전 상한으로 둘 것인가?
- 해당 pilot의 B1/60초 Brier 평가 모집단·가중/기준을 원 계약대로 고정한 뒤 어디까지 확인할 것인가?

이 문서를 작성하거나 로깅 코드를 구현한 것으로 위 선택이나 실행 승인이 생기지 않는다. 현재 상태는 G-S4a 저자 결정 대기다.

## 이번에 확인한 원 계약 경로

- `C:/Users/todtj/PycharmProjects/LOL_teamfight_new/docs/superpowers/specs/2026-10-20-c-ept-design.md` 7–8절: 학습/STOP/DEV 역할, 평가 가중, GPU 탐색·CPU 보고, 단일 실행 예산, 다중 seed/구간/주장 범위.
- `C:/Users/todtj/PycharmProjects/LOL_teamfight_new/docs/superpowers/plans/2026-10-22-c-ept-s4a.md` Q10/Task5 및 S4a/S4b 구분: STOP patience2, 25% 대리 과제, G-S4a 뒤 별도 S4b 계획.
- `LOL_teamfight_new-c-ept-s4/.superpowers/sdd/2026-10-22-c-ept-s4a/task-5-contract.md`: head-last·호출 회계·자원 상한·종료한 호출의 무변경 계약.

원본 문서가 보존된 위치를 확인한 것이며 현재 실행의 membership/원시 자료를 열어 대조한 검증은 아니다.
