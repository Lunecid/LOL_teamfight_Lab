# S4b run kernel guide / S4b 실행 핵심 안내

The run kernel (`c_ept/run`, decision D11) starts one run from one hash-locked profile file. This guide says what the run types are, which profiles exist, which commands start them,
what the exit codes and the ledger mean, how the TEST gate works, which numeric policy every run holds, and what is still pending. The old stacks (S3, S4a, EPT-T002/T003, EPT-T008/T009)
are frozen: they stay byte for byte unchanged (see section 8: their hashes are pinned by freeze records, so they carry no mark in the file).

실행 핵심(`c_ept/run`, 결정 D11)은 해시로 잠근 프로파일 파일 하나로 실행 한 번을 시작한다. 이 문서는 실행 종류, 프로파일, 명령, 종료 코드, 원장, TEST 관문, 숫자 정책, 아직 정해지지 않은 것을 적는다.
옛 스택(S3, S4a, EPT-T002/T003, EPT-T008/T009)은 결과를 그대로 두기 위해 동결했다. 해시가 동결 기록에 고정되어 있으므로 파일에는 표시를 넣지 않고 8절에 목록으로 둔다.

## 1. Run types / 실행 종류

| Command | Profile `kind` | What it does / 하는 일 |
|---|---|---|
| `train` | `train` | One training (statistics stage, then training stage; best STOP checkpoint, `termination_reason='budget'` at the wall cap) / 학습 한 번 |
| `evaluate` | `evaluate` | The evaluation of trained models on one role (`STOP`, `DEV`, `TEST`) / 한 역할에서 학습된 모형 평가 |
| `measure` | `measure` | The P6 bit-identity check: short calls with the same setting, each in a fresh process, compared bit for bit / 같은 설정 짧은 호출의 비트 동일성 |

The profile of a run must be of the kind of its command, else the command is refused (exit code 2).
프로파일의 `kind`가 명령과 다르면 거부한다.

## 2. Profiles / 프로파일

- Format and fields: `profiles/README.md`. A profile is strict JSON with LF line ends; the SHA-256 of its bytes is the lock and is recorded in the ledger header. `.gitattributes` keeps
  `profiles/**/*.json` free of line-end conversion. 필드 표와 규칙은 `profiles/README.md`.
- The S4b files are in `profiles/s4b/`. Only profiles whose every value is decided are written; they are written by `python -m tools.make_s4b_profiles` (deterministic; a final run takes the values below as arguments, no code change).
  - `train_NN_<tag>.json`: the trainings of the table of plan Task K9 (NN = 01 ... 13). Written now: `train_01_lambda0.json` only. Common settings: `kill_law` shifted, `k_condition` binned,
    `decay_groups` split, `column_drop` store_and_subset, `stats_population` selected_fit, `on_wall_cap` best_stop; 25 % trainings: 12 epochs, `swap_mode` random_view (10 and 11 carry their own),
    fixed membership `selection_seed` 20261022; seeds 1-3 are 20261022, 20261023, 20261024; training 13 (100 %): 8 epochs (the start value of D10 P2).
  - `evaluate_NN_<tag>.json`: the role STOP evaluation of a written training (decision D13), one model `tNN`. Written now: `evaluate_01_lambda0.json`.
  - `measure_p6.json`: the P6 measurement; it names `train_01_lambda0.json` by relative path and SHA-256. Written only when the steps are given (pending).
  - `PENDING.json`: what the S4b run plan still has to decide, with the dependency of each file (section 7).
- 학습 13개의 표는 계획 Task K9와 같다. 값이 모두 정해진 것(학습 1)만 쓰고, 최선 λ·학습률 등에 달린 것은 대리값 없이 `PENDING.json`에 의존하는 값과 함께 남긴다.

## 3. Commands / 명령

```
python -m c_ept.run.cli {train,evaluate,measure} --profile <file> --run-dir <folder> --inputs <manifest> --wall-hours <h> [--approval "<text>"] [options]
```

- `--inputs`: the inputs manifest (`ept-run-inputs-v1` for train and measure, `ept-run-eval-inputs-v1` for evaluate). `--spec`: the decision record (default: that of `c_ept.config`).
- `--device cuda|cpu` (default `cuda`), `--threads` (8), `--batch-size` (32), `--memory-gib` (40). A GPU run needs `--approval` with the text of the author's approval; it is recorded as given.
  A CPU run (tests) does not. GPU 실행은 저자 승인 문구가 있어야 시작한다.
- `train` with `on_wall_cap: best_stop` needs `--allowance epoch_s,checkpoint_write_s,finalisation_s` (the time the wall-cap stage holds beyond the 24 h of the trainer).
- `evaluate` DEV and TEST take only the default stage body; `--access-dir` (the access ledger), and for TEST `--lock` (the S5 lock file) and `--rehearsal-run` (the DEV rehearsal run folder).
- Every command goes through the supervisor (one Windows-Job-owned child per stage) and the K4 gates: native-thread policy, E6, filtered child environment, decision-record check, the start gate of
  the code identity, and the budget. A run folder is new, or the folder of the run to resume.
- No command here may be started without the author's approval of the S4b run plan (X1). 이 명령들은 S4b 실행 계획(X1)의 저자 승인 없이는 시작하지 않는다.

## 4. Exit codes / 종료 코드

| Code | Meaning / 뜻 |
|---|---|
| 0 | The run finished. A measurement that finds a difference has finished too: read `identical` in `MEASURE.json`. / 실행이 끝났다(측정이 차이를 찾아도 끝난 것이다) |
| 2 | A refusal: wrong command line, wrong profile kind, GPU run without an approval text, a custom stage body for DEV or TEST, or any refusal of a gate or runner. Nothing was reserved or computed that is lost. / 거부 |
| 3 | A computation that did not finish: a stage failed or the budget ran out during the run. / 계산이 끝나지 못했다 |

The old stacks keep their own exit codes. 옛 스택의 종료 코드는 바뀌지 않는다.

## 5. The ledger / 원장

Under the run folder (`c_ept/run/ledger.py`): `HEADER.json` (written once: the code identity, the fingerprint of the decision record, the SHA-256 of the profile, the runner constants
including the approval text and the numeric policy), `LEDGER.json` (the stages in order, each `begun` or `finished`, with the hashes of their inputs and outputs, and the supervisor's notes: stop
reasons and E6 records) and `stages/<s>.json` (one file per stage, bound to the header and chained to the previous stage). Finalisation is idempotent: on resume the recorded hashes are checked
again, and any mismatch stays fail-closed (a run is never silently continued on other inputs). `train/RECORD.json` and `evaluate/RECORD.json`, `evaluate/REPORT.json` and `MEASURE.json` hold
the results of the stages. 원장은 단계마다 입력·출력 해시를 남기고, 다시 시작할 때 해시가 하나라도 다르면 거부한다.

## 6. The TEST gate (D28) and the numeric policy (D10 P6) / TEST 관문과 숫자 정책

- DEV and STOP: STOP (decision D13) is where S4b decides. DEV is not opened in S4b. Every DEV evaluation writes one entry to the access ledger before it makes a number.
- TEST: the profile carries `holdout_lock_sha256`, the SHA-256 of the S5 lock file (required for the role TEST; the DEV rehearsal of a TEST run carries the same value). The gate opens TEST
  once, and only if the lock file has that hash, a finished DEV rehearsal of the same profile (the role aside) exists with the same code, loader, backend, inputs manifest and training records,
  and the ledger shows that TEST was never opened. DEV와 TEST 평가는 기본 단계 본문만 받는다. TEST 관문은 S5 잠금 파일의 해시, 같은 설정의 DEV 리허설, TEST를 한 번도 열지 않은 원장을 요구한다.
- Numeric policy (`c_ept/run/numeric.py`): float32, `float32_matmul_precision` highest, TF32 off (cuBLAS and cuDNN), cuDNN deterministic with benchmark off, deterministic algorithms with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `fill_uninitialized_memory` off, autocast refused. It is applied before any model is built, read back, refused if the state is not the policy, and
  recorded (with its hash) in the stage records and the ledger header. The reporting device is the deterministic GPU; CPU is not the reporting device. On the CPU none of these settings
  changes a value (K8 shows the kernel reproduces the T002 wrapper bit for bit on a CPU toy store). 보고 장치는 결정적 GPU이고, 정책은 모든 실행의 기록에 남는다.
- Evaluated numbers are decision data of S4b (closed-form metrics on the whole STOP split, decision D30), not held-out estimates.

## 7. Still pending / 아직 정해지지 않은 것

Listed with reasons in `profiles/s4b/PENDING.json`; the S4b run plan (X1, with the author's approval) decides them, and the generator then writes the final files.

- lambda_Omega values of trainings 2 and 3 (D8: 0 and two provisional values "set with reasons in the S4b plan"); only 0 (training 1) is written.
- The best lambda_Omega (D13): trainings 4-13 depend on it, so none is written. The best learning rate (D10 P3, D13): trainings 6-13 depend on it.
- The 50 % training (training 12): its membership rule and its epoch budget (and the two best values above).
- `chunk_views` of the evaluate profiles: left at the default; the M14 hand-off value is 256, and the runner refuses it until K6e wires the composed STOP-sample pass (`UNWIRED_FIELDS`).
  `strata`, `price_sensitivity`, `b1_swap_average` are refused until M17.
- The `checkpoints` of the evaluate profiles (the training run folders exist only after the trainings), and which of trainings 1-5 is the chosen configuration for the C4 seed group.
- The number of steps of the P6 measurement: at least warmup_steps (10) plus enough steps to cover the batch-shape variety; set at X1. `measure_p6.json` is not written until then.
  Every GPU measurement and every training needs the author's start approval.

## 8. Frozen old stacks and the freeze records that pin them / 동결된 옛 스택

These modules are not edited (not even a docstring line): a changed byte changes the hash in the freeze records below. Hashes in `harness/reports/EPT-T002-validation.json` (T002),
`harness/results/EPT-T005-source-freeze.json`, `EPT-T006-local-source-freeze.json` and `EPT-T008-source-freeze.json` list the files; `EPT-T003-source-freeze*.json` pins `tools/ept_dev_evaluate.py`.

| Stack | Module | Pinned by |
|---|---|---|
| S3 | `c_ept/workflows/run_s3.py` | T005, T006, T008 freeze |
| S4a | `c_ept/workflows/run_s4a.py`, `s4a_operator.py`, `s4a_runtime.py` | T002 validation, T005, T006, T008 freeze |
| EPT-T002 | `c_ept/workflows/run_training.py`, `training_operator.py` | T002 validation, T005, T006, T008 freeze |
| EPT-T002/T003 launchers | `tools/ept_training_controller.py`, `ept_training_job.py`, `ept_training_monitor.py`, `tools/ept_dev_evaluate.py` | T002 validation, T003 freeze |
| EPT-T008/T009 | `c_ept/comparison/operator.py`, `tools/ept_comparison_controller.py` | T008 freeze (operator.py) |

Do not extend them; new work goes into `c_ept/run`. 새 작업은 `c_ept/run`에 넣는다.
