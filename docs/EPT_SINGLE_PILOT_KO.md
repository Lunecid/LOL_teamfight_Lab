# EPT 단일 학습 프로필

as_of: 2026-10-25 Asia/Seoul
contract: EPT-T002-single-pilot
state: INDEPENDENT_REVIEW_PASS / PRODUCTION_CHECK_REQUIRED_BEFORE_START

## 승인한 범위

CUDA·S·기존 FIT/STOP 25% proxy·patience2·최대30에폭·batch32·seed20261022. AdamW lr0.001, weight_decay0.01, warmup10, float32와 기존 결정성/native8 계약을 유지한다. STOP loss가 엄격히 낮아질 때만 개선이며 동률도 미개선이다. 두 번 연속 미개선이면 멈춘다. 최대30은 안전 상한이다.

훈련·선택만 수행한다. 기존 비용 프로필의 2에폭 제약은 바꾸지 않는다. 전용 training_operator/run_training과 run_ept_training.ps1의 Check/Start/Status를 사용하며 Resume는 제공하지 않는다. 학습률 스케줄의 최대 에폭이 달라져 과거 2에폭 호출과 다른 학습 경로다. 과거 결과를 연장하지 않는다.

## 실행과 예산

구현 → 합성 테스트 → 소스 동결 → 독립 검토 → 고정 local commit → 별도 청결 실행 checkout → 외부 설정/제어 폴더 → read-only Check → READY → 한 번 Start 순서다. 원본 측정 checkout 및 과거 CPU/GPU run은 보존한다. IDE 사용자 변경은 실행 checkout에 섞지 않는다. 이 실험을 위한 코드 고정만 로컬 커밋하고 공개 업로드하지 않는다.

Controller는 launch.once로 한 번의 호출만 예약하고 operator는 별도 operator_start.once로 실제 Start 중복을 차단한다. 양쪽 marker가 남으면 실패한 시도를 자동 재개하지 않는다. control/run.json의 명시적 run 경로와 code/config/run identity를 사용하며 최신 폴더를 추측하지 않는다. PROCESS의 실제 Python PID와 생성시각을 확인한다.

훈련 누적24시간, 표본 host private40GiB 미만을 기존 경계에서 검사한다. Controller는 준비 단계 포함 전체24시간 timeout과 프로세스 트리 종료를 담당한다. 감지는 최대2초 간격이고 종료 정리는 최대10초를 허용하므로 엄밀한 무지연 OS 종료 보증과 구분한다. 메모리 표본은 연속 상한 보증이 아니다. 자동 재시도0, 학습1개, Strata OFF. 모델·환경·GPU·인증·입력 설정을 자동 교체하지 않는다.

## 결과와 해석

완료 head에 해시로 결속된 epoch_metrics.json/csv만 완료 결과다. 매 에폭의 FIT batch-mean loss, STOP pooled evaluation loss, 학습률, best epoch, 종료 상태를 보존한다. FIT와 STOP은 정의와 가중이 다르므로 별도 패널로 그린다. 두 손실의 단순 차이를 일반화 격차로 해석하지 않는다. 음수인 유한 NLL은 오류로 단정하지 않는다.

감시는 .pt/가중치 payload를 읽지 않는 메타데이터 전용이다. 새 완료 에폭마다 한국어 보고서·JSON/CSV·HTML·가벼운 SVG를 만들고, 변화가 없으면 알리지 않는다. 종료 확인 뒤에만 전체 수치 PNG를 렌더링하고 실제 이미지와 원본 수치를 검토한다. 실패 시 부분 결과와 원인을 보존한다. 별도 개발 작업이 자원 경합을 만들 수 있어 독점 호스트 성능 실험이라고 주장하지 않는다.

단일 proxy·seed 결과는 전체 자료 수렴, 일반화, B1 우위, 최종 논문 성능을 입증하지 않는다. 이번 학습 종료 뒤 결과를 보고하고 멈춘다. 추가 학습·DEV/Brier 평가·S4b 전체·S6/S7·TEST 및 보고 장치 변경은 별도 저자 결정이다.

## 독립 검토 후 프로세스 소유 보완

조기 래퍼 종료로 자식을 놓치는 위험은 Windows Job Object로 보완한다. 래퍼는 AwaitJobBinding 옵션에서 표준입력 G를 기다리고, controller가 launch-owned 프로세스 handle을 Job에 결속한 뒤에만 진행한다. Job은 breakaway를 허용하지 않고 마지막 handle 종료 시 자식을 정리한다. 실제 PID/생성시각과 OS Job 소속을 함께 확인하며 실패 시 PID 숫자만 보고 임의 프로세스를 종료하지 않는다. 준비 검증에는 모델 대신 대기 프로세스를 사용한다.

동시 메타데이터 읽기는 제한된 일관 스냅샷 재확인으로 처리한다. 안정적으로 손상된 관측은 observation_error이며 실제 PROCESS/controller 실패와 구분한다. 감시자 자체가 사라지면 정상 bootstrapping으로 숨기지 않는다. 관련 Microsoft 기본 동작: [Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects), [AssignProcessToJobObject](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject).
