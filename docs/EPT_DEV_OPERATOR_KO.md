# EPT-T003: 학습 종료 후 단일 DEV 평가

현재는 별도 개발 폴더의 구현·합성 검증 단계다. 이 문서의 명령은 자동으로 실행되지 않는다. 현재 학습의 설정, 코드, 입력, checkpoint와 monitor는 변경하지 않는다. Strata는 OFF이며 health 호출이나 시작도 하지 않는다.

## 평가 대상과 산출물

`EPT_DEV_DEFINITIONS_KO.md`의 정의를 고정한 뒤, 기존 학습의 STOP 최선 에폭만 사용한다. 전체 계약상 적격 DEV를 평가하며 EPT가 학습한 25%를 DEV에도 적용하지 않는다. 기존 whole-FIT B1과 비교하므로 동일 학습 예산 비교가 아니다. DEV로 최선 에폭·모델·threshold를 선택하거나 calibration/refit하지 않는다.

각 endpoint의 경기 가중 pooled AUC/Brier, head별 원래 NLL, 명시적으로 공통인 head의 고정 B0 scale 정규화 NLL을 기록한다. paired match bootstrap 2,000회, seed 20261022를 사용한다. 단일 클래스 AUC와 무효 bootstrap은 null로 남기며 유효 표본만의 조건부 CI를 전체 CI로 표시하지 않는다. 전체 모듈을 합친 단일 AUC는 만들지 않는다. M3의 MC 오차와 무승부를 별도 기록한다.

새 출력 폴더에는 CHECK/RUN_MANIFEST/controller/phase/RESULT/final-audit JSON, 로컬 전용 `predictions.private.npz`, aggregate JSON/CSV, 한국어 보고서, SVG/PNG가 생긴다. prediction 파일은 공개하거나 채팅에 업로드하지 않는다. 완료 후 PNG를 실제로 열어 원본 JSON/CSV와 대조하고 `final-audit.json`의 수동 검토를 별도로 기록한다. 코드 검토 PASS나 정상 프로세스 종료만으로 과학적 성공·일반화·B1 우위·G1–G5 통과를 주장하지 않는다.

## 실행 준비

1. EPT-T002 학습이 정상 종료하고 최종 보고와 final-audit가 완성될 때까지 기다린다. 관측 오류와 학습 실패를 구별한다. 실패 run은 이 evaluator가 거부하며 재시도/Resume하지 않는다.
2. 독립 검토가 끝난 평가 소스를 로컬 고정 커밋으로 보관하고 전체 checkout을 청결하게 만든다. 필요한 경우 그 커밋의 전용 실행 checkout을 만든다. 학습 실행 checkout에는 변경을 적용하지 않는다. 공개 push는 하지 않는다.
3. `configs/ept_dev_evaluation.example.json`을 별도 제어 폴더에 복사한다. 예제는 의도적으로 실행 불가한 placeholder를 포함한다. `source`는 실제 평가 실행 checkout, `reviewed_commit`은 검토된 로컬 고정 커밋, `definitions_sha256`은 그 checkout 정의 문서의 SHA-256으로 채운다. `s1/s2/s3`은 부모 RUN.json의 `identity.inputs.paths`를 그대로 사용한다. 최종 학습 보고서·audit의 SHA-256을 채운다. 해시를 추측하거나 검사 예외를 만들지 않는다.
4. 출력은 `D:/LOL_Data/runs` 바로 아래 `ept_dev_`로 시작하는 **존재하지 않는** 폴더다. 실패 후 같은 폴더를 지우고 재실행하지 않는다. 별도 재시도에는 저자 결정이 필요하다.

다음은 설정 파일을 `D:/LOL_Data/runs/ept_dev_control_1/operator-config.json`에 확정한 경우의 명령이다. 실제 학습 종료 전 실행하지 않는다.

```powershell
& 'C:\Users\todtj\anaconda3\python.exe' -B 'C:\Users\todtj\.codex\worktrees\ept-dev-evaluation\LOL_teamfight\tools\ept_dev_evaluate.py' --config 'D:\LOL_Data\runs\ept_dev_control_1\operator-config.json' --action Check
```

`Check`는 기본 동작이며 표준 라이브러리·메타데이터만 사용한다. 정상 종료, controller/PROCESS/PID 생성세대/Job 정리, 고정 학습 identity와 소스, 입력 metadata, STOP 최선 pointer 및 최종 보고 해시를 확인한다. 모델 tensor·원시·보호 분할 payload를 읽지 않는다. READY는 실제 추론이 성공했다는 뜻이 아니다.

학습 종료 후 작성할 `reports/final-audit.json`의 평가 수락 계약은 `schema=ept-pilot-final-audit-v1`, `state=completed`, 정확한 `run/source/code_commit/operator_config_sha256/run_identity_sha256`, `final_report_sha256`, `manual_visual_review=verified_against_json_csv`다. `reviewed_artifacts`는 reports 폴더 기준 상대 경로를 SHA-256에 대응시키는 객체이며 실제 대조한 PNG·JSON·CSV가 모두 포함되어야 한다. 해시가 맞더라도 failed/PENDING/다른 run의 audit는 거부한다. 현재 학습 폴더에 이 audit를 미리 만들거나 검토 완료로 표시하지 않는다. 기존 최종 감사가 다른 형식이면 진짜 종료·그림 검토 근거를 확인한 뒤 이 계약으로 정리한다; 값만 맞추어 승인을 흉내 내지 않는다.

Start의 사전 검사도 평가 소유 Job 안에서 시간 제한을 적용한다. 학습이 끝나지 않았으면 실패 기록을 남기고 실제 추론 전에 거부한다. 사용한 전처리 배열 파일은 요청 시와 수락 전에 SHA-256·shape·dtype를 검증한다. 미사용 배열을 전체 열거하거나 Store.verify로 훑지 않는다. 공유 배열 파일 해시와 read-only memmap은 여러 role의 바이트가 담긴 파일을 접근할 수 있으나, 추론·채점은 canonical DEV 행에 한정한다.

종료 조건 및 설정 확정을 확인한 뒤 아래 한 번의 실행만 수행한다.

```powershell
& 'C:\Users\todtj\anaconda3\python.exe' -B 'C:\Users\todtj\.codex\worktrees\ept-dev-evaluation\LOL_teamfight\tools\ept_dev_evaluate.py' --config 'D:\LOL_Data\runs\ept_dev_control_1\operator-config.json' --action Start
```

실제 추론에서만 hash-bound `best.pt`의 tensor state와 기존 B1 모델을 프로그램이 읽는다. `last.pt`, optimizer, RNG payload를 로드하거나 학습을 재개하지 않는다. DEV 외 평가 역할·TEST/CAL/AUDIT에는 진입하지 않는다.

## 제한과 실패 처리

CUDA 0, float32, batch32, native threads8, seed20261022, MC1000, bootstrap2000을 고정한다. total8h, preflight0.5h, paired inference4h, metrics2h, report1.5h를 상한으로 두고 독립 supervisor가 자체 child Job만 관리한다. sampled private40GiB 미만을 검사한다. 표본 private, lifetime upper, GPU allocated/reserved를 구별한다. 표본 감시는 연속 메모리 상한 보장이 아니다.

OOM·시간 초과·identity 불일치·cleanup 미확인은 실패다. 배치 축소, CPU 전환, DEV 일부만 평가, 자동 재시도는 하지 않는다. stderr와 부분 산출물은 보존한다. 프로세스 종료와 Job 정리가 확인되지 않으면 결과를 최종 완료로 수락하지 않는다. 현재 학습 Job/PID/token은 조작하지 않는다.
