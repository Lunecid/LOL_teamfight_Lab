# EPT 에폭별 손실 기록 읽기

변경 범위: [EPT-T001](../harness/tasks/EPT-T001-epoch-loss-logging.md). 다음 학습 설정은 [별도 초안](EPT_NEXT_TRAINING_PLAN_KO.md)이다. 이 안내는 실제 학습·Resume 명령이 아니다.

## 저장 위치와 완료 기준

새 버전 trainer가 완료한 각 `epoch_XXXX/`에는 기존 가중치와 `head.json` 외에 `epoch_metrics.json`, `epoch_metrics.csv`가 생긴다. 두 파일은 그 에폭까지의 실제 누적 손실 이력 스냅샷이다. 여러 폴더의 CSV를 단순 연결하면 이전 에폭이 중복되므로 **최신 완료 폴더의 스냅샷 하나**를 읽는다.

새 완료 head는 `ept_train_checkpoint_v6`, 손실 문서는 `epoch-metrics-v1`이다. head의 epoch_metrics.files에는 JSON/CSV 각각 바이트 크기와 SHA256이 기록된다. `head.json`은 가중치 파일 쓰기와 손실 JSON/CSV 검증 뒤 마지막에 기록하는 완료 표지다. 가중치 바이트 해시 검증은 기존 payload 재개 경로가 별도로 수행하며, metadata-only 읽기가 그 검증을 대신하지 않는다. 파일 하나가 보이는 것만으로 그 에폭의 완료를 판단하지 않는다. 누락·손상·일관성 오류가 있는 최신 완료 폴더에서 예전 에폭으로 조용히 후퇴하지 않는다.

이는 개별 파일 원자 교체와 head-last 게시 계약이다. 디렉터리 전체가 OS 트랜잭션이거나 전원 장애 뒤 내구성을 보장한다는 뜻은 아니다. head가 없는 중간 폴더는 미완료이며 기존 재개 절차가 처리한다. 사람이 임의로 지우거나 완료 표지를 추가하지 않는다.

## 열과 JSON 상태

| CSV 열 | 의미 |
|---|---|
| epoch | 사람이 보는 완료 에폭 번호, 1부터 |
| epoch_index | 내부 루프 번호, 0부터 |
| fit_batch_mean_loss | 학습 모드에서 각 FIT 배치의 실제 손실을 평균한 진단값 |
| stop_loss | 기존 evaluate_stop의 정규화·scored-row/경기/규모 집계 계약으로 계산한 평가 손실 |
| lr_last_step | 해당 에폭 마지막 optimizer step에 실제 적용한 학습률 |

JSON은 같은 rows 외에 schema·identity·seed·정의·monitor·patience·best_epoch_index(0기반)·best_epoch(1기반)·bad_epochs·done·stopped_early·termination_reason을 포함한다. 종료 이유는 in_progress, early_stopping, max_epochs다. 자원 초과나 예외는 완료 로그를 성공 종료처럼 바꾸지 않으며 기존 실패/호출 기록을 함께 확인한다.

손실은 실제 유한 실수만 허용하며 bool/NaN/Inf를 거부한다. 연속 밀도의 NLL 등에서는 음수도 유효할 수 있으므로 모든 손실에 임의의 0 하한을 두지 않는다. epoch/lr/집계 길이·identity·선택 상태·CSV와 JSON의 일치를 검사한다. 값이 비어 있거나 과거 로그가 없다는 이유로 0을 만들지 않는다.

FIT와 STOP은 모드·집계 단위·가중이 다르다. 두 곡선의 높이 차이를 곧바로 과적합이나 일반화 격차로 해석하지 않는다. STOP 개선·선택과 예측 품질 지표(Brier 등)의 비교도 별개다. 학습률 열은 다음 step의 예정 lr이 아니다.

## 가중치 없이 읽기

개발 코드의 `c_ept.ept.checkpoints.latest_metadata(out_dir)`는 `(folder, head, history)`를 반환한다. head와 손실 JSON/CSV만 읽고 Torch·데이터셋·.pt를 읽지 않는다. `history['rows']`로 에폭별 값을 확인한다. 없으면 반환값은 None, legacy v5 head이면 history=None이다. 숫자가 0이라는 뜻이 아니다.

```python
from pathlib import Path
from c_ept.ept.checkpoints import latest_metadata

# 실행 승인된 새 run의 모델 학습 폴더를 지정한다. 이 코드는 상태 읽기뿐이다.
found = latest_metadata(Path(r"D:\YOUR_NEW_APPROVED_RUN\train\S_cuda"))
if found is None:
    print("완료된 체크포인트 없음")
else:
    folder, head, history = found
    if history is None:
        print("legacy: 숫자 손실 로그 미제공")
    else:
        print(history["termination_reason"], history["best_epoch"])
        for row in history["rows"]:
            print(row["epoch"], row["fit_batch_mean_loss"], row["stop_loss"])
```

운영 중에는 최신 완료 head·identity·해시를 함께 읽는다. CSV만 별도 복사하면 provenance를 잃을 수 있으므로 head/JSON과 검증된 source 경로를 보고서에 남긴다. 로깅 IO 비용은 새 실행의 실측에 포함된다. 과거 비용 실측을 이 버전의 처리속도로 다시 표시하지 않는다.

## 과거 버전과 재개

새 코드는 역사적 v5 checkpoint를 계속 읽을 수 있지만 구버전 v5 전용 코드가 새 v6를 읽거나 재개할 수 있다는 뜻은 아니다. 새 v6 run은 검토·고정된 해당 버전의 reader/trainer를 사용한다. 손실 export가 없는 완료 run을 소급 수정하지 않는다. 원래 실제 CPU/GPU run의 payload를 열어 로그를 추출하지 않았다.

향후 별도로 승인된 미완료 legacy 재개에서 trainer가 원래 저장된 진짜 loop 이력을 정상 복원한 경우에만 다음 새 완료 에폭에 누적 스냅샷을 남긴다. 누락되거나 비일관적인 loop 이력을 추정·0 채움하지 않는다. 새 형식에서 손실 파일이 없거나 손상된 경우를 legacy로 취급하지 않는다. 이미 완료된 호출은 로그를 다시 쓰지 않는다.

## 확인 범위

합성 fixture에서 수치·구조·쓰기 중단·재개·legacy 경로와 payload-free reader를 확인한다. 실제 자료 학습/가중치 reload, TEST/CAL/AUDIT, 다음 S4b 실행은 이번 작업에서 하지 않는다. 최종 검토·실제 명령·시험 결과는 [하네스 상태](../harness/status.md)와 연결된 결과 보고를 따른다.
