# EPT 모듈화 계획 (2026-10-24)

- 대상: `codex/modularize-three-projects`, 이 worktree만 수정. 커밋/push는 lead 담당.
- `c_ept/runtime/resources.py`: 프로세스 트리 메모리 합산, 중첩 scope sampler, E6 검사 시간/횟수. RUN3-2/3 의미 보존. 기계 측정과 E6 probe는 구체적 함수 주입.
- `run_s3.py`: 기존 private 이름은 runtime API의 얇은 호환 어댑터/별칭. 기존 `_tree_memory`/psutil monkeypatch를 호출 시 반영. 중복 구현 제거.
- `ept/train.py`: runtime sampler/트리 API 사용, runner 의존 제거. `_memory_sample`, `_clock`, `_write_atomic` 주입 지점 보존. 40 GiB 엄격 상한/24 h 포함 경계/박동 및 고아 호출 회계는 유지.
- `ept/checkpoints.py`: 원자 쓰기/직렬화/해시 검증/최신 완료 체크포인트 검색. v5 schema, head-last 기록, 손상 거부와 head 없는 torn 폴더 재사용, 과거 best 참조 보존. train의 얇은 wrapper는 현재 `_write_atomic`을 주입.
- `run_s4a.py`: runtime 공개 sampler/guard 및 checkpoints 공개 lookup 사용. 기존 S3 ledger/코드 정체성 검증은 유지.
- `ept/training_step.py`: 배치 계산만 분리 가능성을 검토. 학습률/RNG/손실, 측정 경계와 부분 실패 기록이 바뀌지 않는 경우에만 추출. 에폭/조기 종료/체크포인트 회계는 train 소유.
- 공개 경계 설명은 간결한 한국어. 수식 ID/과학 정의/입력과 holdout 접근 제한 유지. heads/compose는 별도 담당.

검증: 변경 파일 AST parse, diff whitespace, runner 역의존 탐지, 합성 임시 바이트/가짜 프로세스/시계/연산만 쓰는 가벼운 회귀 테스트. 실제 데이터/기존 checkpoint/동결 출력은 열지 않는다. Task8 동안 학습/GPU/벤치마크/무거운 suite 금지. 전체 targeted suite는 lead 실행용 명령으로 보고한다.

제약: Strata health 연결 차단. live Task8과 자원 경합을 피하려고 서버 시작 없이 좁은 직접 검토. 독립 리뷰/commit/push는 lead 인계. worktree의 root AGENTS.md 및 기존 함수/테스트 계약을 확인했으며, 초기 조사에는 소스가 참조하는 `.superpowers/sdd`와 `docs/superpowers/plans`가 없었다. 재개 후 lead가 추가한 `docs/superpowers/plans/2026-10-24-three-project-modularization.md` 작업 계약을 읽고 준수했다. 최신 직접 승인 범위가 변경 권한의 기준이다.


## 구현 및 검증 인계

상태: 구현 완료. Zeno 독립 정적 리뷰 ACCEPT, 최종 STOP wrapper `eval()` 조정까지 수용됨(lead/사용자 확인). 이 작업자는 commit/push를 하지 않았다. 다른 작업자의 heads/compose·패키징 변경은 수정하지 않았다.

변경 파일 (이 작업자 소유):

- 신규 `c_ept/runtime/__init__.py`, `c_ept/runtime/resources.py`
- 신규 `c_ept/ept/checkpoints.py`, `c_ept/ept/training_step.py`
- 수정 `c_ept/ept/train.py`, `c_ept/run_s3.py`, `c_ept/run_s4a.py`
- 수정 `c_ept/tests/test_ept_train.py`
- 신규 `c_ept/tests/test_runtime_resources.py`, `c_ept/tests/test_ept_checkpoints.py`, `c_ept/tests/test_ept_training_step.py`
- 이 계획/인계 문서

`score_blocks`는 공개 `m2_rate`가 callable이면 사용하며 없거나 callable이 아닌 기존 mock에는 `m2(...).rate`를 사용한다. 공개 API의 예외를 가리고 fallback하지 않는다. 다른 작업자의 FrameAveraged API는 학습 시 전체 M2 경로를 보존한다. STOP에서는 새 wrapper도 `.eval()`로 설정한다. 실제 검증에서 내부 모델만 eval인 새 wrapper가 기본 training=True여서 rate 경로를 사용하지 않는 문제가 발견되어 이 한 줄을 수정했고, 모델 원래 모드 복원은 그대로다. 비용 설명도 학습/평가 경로에 맞게 수정했다.

실행한 검증:

- 새 순수 합성 suite: **52 passed, 0.97 s**, `-W error`, plugin auto-load/cache off. 프로세스/시계/torch/loss는 가짜 객체, checkpoint는 임시 합성 바이트만 사용한다. 첫 시도의 Windows `interval=1e9` 경고는 실제 스레드 테스트 간격을 60 s로 바꿔 해결했다(기다리는 테스트가 아니며 즉시 종료 신호로 join한다).
- 기존 focused suite 1차: **22 passed, 1 failed**, pytest 24.15 s, import 포함 **27.521 s**. 실패는 위 STOP wrapper 모드 문제를 드러낸 평가 호출 횟수 assertion이다.
- 수정 후 영향 범위 focused suite: **6 passed**, pytest 14.48 s, import 포함 **18.006 s**. M2 평가/배치 모드, managed interruption→resume 비트 일치, best/last, Q10 시간 항목 및 schema 검증을 통과했다. 앞서 통과한 sampler, fake memory/clock, strict 40 GiB/inclusive 24 h, fsync/부분 write 및 heartbeat/orphan 검사는 변경되지 않았다.
- 두 기존 focused 실행의 총 경과 시간은 **45.527 s**(실패 진단 포함)이다. 순차 실행, OMP/MKL/OpenBLAS/NumExpr/VECLIB/BLIS/LOKY=1, CUDA_VISIBLE_DEVICES=-1, torch intra/inter-op=1. 이 프로세스 안에서만 deterministic_cpu의 default threads를 1로 바꾸고 finally 복원했다. 프로덕션의 8-thread 기본값은 수정하지 않았다. 실측 처리량/속도 비교가 아니다.
- 변경 Python 11개 AST parse 성공, tracked own diff whitespace 오류 없음.
- 기준 `282dcc36f2e175b33782fc0b4bcb31581e663612`와 정적 비교: train 선언 43개, resource accounting 메서드 10개, S3 선언 80개, sampler/window 구현 4개, guard window 메서드 3개가 문서 문자열 제외 동일하다. IO/검증 함수 4개와 주입 writer 선택을 제외한 save 순서도 동일하다. 추출 배치 함수를 다시 inline하면 전체 `_run` 에폭 조정 AST가 동일하다. evaluate_stop은 wrapper `.eval()` 추가만 다르다.

이연 사항과 리뷰 초점:

- Zeno 독립 정적 리뷰와 최종 eval 조정 수용은 완료되었다(lead/사용자 확인). Gemini 주석은 세 컴포넌트에 적용되었으며 실행 AST 동등성은 lead가 확인했다. commit/push는 lead 담당이다. 구현자 자체 검증을 독립 리뷰로 부르지 않는다.
- 기존 전체 targeted suite, hard-process interrupt/crash-loop subprocess matrix, 기본 8-thread 환경의 통합 검증, GPU/CUDA 및 실제 연구 입력은 미실행이다. Task8 실행과 경합하는 전체 회귀/학습/벤치마크는 실행하지 않았다.
- 순수 테스트 일부는 runner와 tensor 의존성을 로드하지 않기 위해 실제 AST 정의만 주입해 실행한다. 실제 통합을 전부 대체하지 않는다. 기존 focused 검사는 실제 새 runtime/checkpoint/학습 부품을 사용했다.
- resource 표본은 자식 누락/표본 사이 spike를 보장하지 않는다. lifetime upper는 per-stage peak가 아니며 기존 계약 그대로다.
- 계산 분리에는 Python 함수/clock callback 호출 비용이 추가된다. 측정 경계 의미는 유지되지만 시간 값 자체의 일치나 속도 개선은 주장하지 않는다.
- 새 rate 최적화의 학습 RNG 동등성/기울기 전체 검증은 다른 작업자의 `test_ept_m2_components.py` 및 아래 후속 명령으로 리뷰한다. 본 작업에서는 managed resume 결과/곡선 비트 일치와 eval dispatch를 확인했다.

Task8 종료 후 lead가 실행할 전체 targeted suite(지금 미실행):

```powershell
python -m pytest -q c_ept/tests/test_runtime_resources.py c_ept/tests/test_ept_checkpoints.py c_ept/tests/test_ept_training_step.py c_ept/tests/test_ept_train.py c_ept/tests/test_run_s3.py c_ept/tests/test_run_s4a.py c_ept/tests/test_ept_heads.py c_ept/tests/test_ept_m2_components.py c_ept/tests/test_ept_compose.py c_ept/tests/test_ept_loss.py
```

기존 focused 실행의 선택 노드: S3 nested sampler/thread/pair/unknown upper/fake process/E6 compatibility; train strict bounds, cached process-tree helper, guard가 별도 표본을 만들지 않는지, pending-write heartbeat 원점 4 cases 및 동일 시계, fsync, last/state-file torn write, managed resume, best/last, M2 shared evaluation, Q10 항목/schema. subprocess/real process tree/실제 데이터 fixture는 선택하지 않았다.


## 완료 상태 및 명령 근거 (최종)

Zeno 독립 정적 리뷰 **ACCEPT**, 최종 eval() 조정 포함. Gemini 문서 문자열은 lead가 세 컴포넌트에 적용했고 실행 AST 동등성을 확인했다. 해당 작업 중/후 이 구현자는 resources/checkpoints/training_step 문서 문자열을 수정하지 않았다. 추가 테스트는 실행하지 않았다.

추가 소유 파일: `c_ept/tests/test_ept_training_step_numeric.py`, `docs/superpowers/reports/2026-10-24-ept-test-evidence.json`.

마지막 숫자 검사: **6 passed, pytest 13.26 s / import 포함 16.171 s**, CPU/native intra/inter-op=1. 실제 AdamW 3-step closed-form 수치 검산, 합성 checkpoint IO로 optimizer 저장/검증/복원 후 가중치·step·exp_avg·exp_avg_sq 비트 일치, 기존 동일 seed model/optimizer/RNG, managed resume, warmup/cosine, eval RNG 검사가 통과했다. 기존 focused 실행의 총 경과 시간은 **61.698 s**(27.521 + 18.006 + 16.171; 첫 실패 진단 포함)이다. 별도 순수 suite는 **52 passed / 0.97 s, 경고 없음**이다.

정확한 pytest argv/node 목록, 환경변수, 프로세스 내 CPU=1 override/복원, 실행별 결과/시간은 `2026-10-24-ept-test-evidence.json`에 저장했다. 당시 실행한 stdin harness/도구 출력에서 옮긴 이력이며 새 실행 기록으로 표시하지 않았다. 임시 경로는 `$TEMP`로만 나타냈고 비밀/로컬 scratch 경로는 포함하지 않았다.

마지막 보고서 갱신 직전의 정적 parse/전체 trailing-space 검사 시도는 lead가 적용한 `c_ept/runtime/resources.py` 문서 공백에서 중단되었다. 해당 파일을 덮어쓰거나 수정하지 않았다. 새 numeric 파일은 실제 pytest에서 import/실행되었고, 앞선 변경 Python 11개 AST parse와 tracked own diff 검사는 성공했다. 문서 문자열 공백을 실행 오류로 보고하지 않으며 lead의 세 파일 executable-AST 동등성 확인을 기록한다.

전체 targeted suite는 **live Task8 때문에 이연 유지**한다. 위 후속 전체 targeted 명령에는 `c_ept/tests/test_ept_training_step_numeric.py`도 포함한다. 실제 연구 입력/GPU/벤치마크/전체 회귀는 실행하지 않았다. 코드 구현과 허용된 단기 합성 검증은 완료되었다. 이 작업자는 commit/push하지 않았다.
