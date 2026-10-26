# EPT·B1·N2 동일25% 비교 계약 · T008

이 문서는 구현 계약이다. **현재 실제 학습·평가 실행 승인은 없다.** 예제 프로필은 `execution_authorized=false`이며 source/input/hash binding이 비어 있어 실행 전 검증을 통과할 수 없다. 구현 수락과 실제 자료·환경 검증/실험 승인은 다르다.

## 왜 비교하는가

현재는 단일 역사적 EPT pilot 뒤 S4 설계 점검을 준비하는 단계다. EPT–B1은 강한 실용 기준선 대비 구조 성능, EPT–N2는 단순 집합 인코더 대비 복잡한 관계 인코더의 추가 가치를 묻는다. N2는 원 설계 필수 비교군이다. 과정 분해·두 시계·지형/관계의 각각의 주장은 원래 A1/A2/A3 대조가 담당한다. 이번 세 모델 비교가 그것을 대신하거나 인과 효과를 증명하지 않는다. MLP/LSTM은 T008 범위 밖이다.

## 공정성 고정값

- `profile=matched25_ept_b1_n2_v1`. 세 군 모두 동일하게 한 번 선택한 FIT25%·STOP25% 경기 manifest를 사용한다. 경기 단위 선택 seed20261022와 모델 seed를 분리한다. 보기를 무작위로 나눠 같은 경기가 FIT/STOP에 걸치게 하지 않는다. 실제 코호트 수는 후속 입력 결속 때 기록하며 이 구현에서 실제 membership을 생성하지 않았다.
- EPT/N2 seed20261022, B1은 원 P9 seed1·CPU8을 유지한다. 단일 seed 진단이므로 이것을3-seed paired 추정으로 보고하지 않는다.
- neural Stats, B1 drop columns, head bias/prior/초기화는 선택 FIT만 사용한다. Stats는 prepare에서 한 번 fit/저장하고 EPT/N2가 같은 hash를 읽는다. 기존 whole-FIT B1나 기존 EPT 최선 가중치를 재사용하지 않고 세 군을 새로 적합한다.
- 원 S3 B0 scale을 neural 학습과 공통8-head 구조 평가에 함께 사용한다. 가격·tier·정답 bins·지형/측지표도 원 고정 FIT 자산을 공유하고 전체 FIT를 사용한 auxiliary라는 provenance를 표시한다. 모든 정보가25%라는 실험이 아니다. scale 재추정/목표 재정의는 하지 않는다.
- EPT/N2는 동일한27개 predictive ViewTensors 필드, 시간 cutoff/가림/표지 조건/실제 노출/팀 교환과 동일한 EPTHeads 법칙을 사용한다. N2는 attention/message passing 대신 공유 entity MLP·팀 sum/mean/count·사건/표지점·관계 집합 집계를 사용한다. pooling 차이는 인코더 구조 차이다. B1은 원 공학적 특징을 쓰는 시스템 기준선이며 완전히 같은 representation을 쓰는 인코더 비교라고 부르지 않는다.
- 현재 기본 S구성의 총 학습 parameter 수: EPT1,128,306, N2(hidden512)1,330,083(+17.88%). 공통 head 포함20% band 안이다. 인코더만 비교하면+22.96%이므로 encoder-only parameter-matched라고 주장하지 않는다. 실제 실행 artifact에 재계수해 기록한다.
- NN float32/AMP off/TF32 off/최고 float32 matmul precision/결정성, batch32, lr0.001, wd0.01, warmup10, patience2, 최대30에폭. STOP strict 개선의 가장 이른 최선 checkpoint를 선택한다. B1은 원 최대1000 round·patience50을 유지하며 epoch과 tree round를 동일한 단위로 취급하지 않는다. HPO를 수행하지 않는 fixed-setting pilot이다.

## 성공·미결·철회 기준

1. 공학 gate: membership/role/target/exposure/row identity, selected-FIT fitting, 유효한 확률/NLL, architecture-bound best checkpoint, source/config/input hash, phase/전체 예산·Job cleanup·독립 검토가 모두 통과해야 후속 실행을 검토한다. 하나라도 실패하면 실행 불가이며 자동 재시도/Resume/작은 batch 대체를 하지 않는다.
2. pilot 손실 차이는 `score(EPT)-score(대조)`다. 상태는 경기 단위 짝 bootstrap 95% 구간 [lo, hi]로 정하며 점 차이의 부호만으로 정하지 않는다. 구간이 `available`이고 유한하며 lo ≤ hi일 때만 쓰고, hi < 0이면 잠정 개선(`provisional_improvement`), lo > 0이면 확장 전 원인 진단(`diagnose_before_expansion`), 그 밖에는 미결(`inconclusive`)이다. 구간이 없거나 유효하지 않으면 미결이고, 경계는 엄격해서 hi = 0이나 lo = 0도 0을 포함하므로 미결이다. 단일 seed의 점 차이/경기 bootstrap만으로 confirmed_success나 최종 연구 성공을 출력하지 않는다. 구간이0을 포함하면 미결이며 동등성 증명이 아니다. `pilot_decision`은 항상 `confirmed_success=false`다.
3. 향후 확증 조건은 기존 세 seed 짝 차이 부호 일치·EPT에 유리한95% 경기 bootstrap 구간과 해당 원 Holm 규칙이다. EPT–B1 주 비교를 이기지 못하면 구조 우위를 주장하지 않고 원 규칙대로 B1을 주 결과로 둔다. N2 대비 이득이 불확실하면 복잡한 인코더의 추가 이득을 주장하지 않는다. N2는 prospective 승인 전 보조 비교다. 다른 유리한 대조로 실패한 주 비교를 대체하지 않는다.
4. primary 구조는 M2 masked3+M3/M4 open5의 원 공통8 head·고정 B0 scale이다.13개 training block 합과 혼동하지 않는다. 각 조건부 module/endpoint의 NLL·AUC·Brier는 기존 T003 정의와 경기당1/n_m 가중을 유지한다. AUC는 weighted pooled 값이며 종합 AUC를 만들지 않는다. 빈/단일-class의 unavailable을0으로 바꾸지 않는다.
5. shared B1 reference 결과와 모든 행/정답/scale를 EPT/N2 사이에서 맞춘 뒤 세 쌍을 보고한다. 다른 행을 inner join으로 버리지 않는다. N2 결과를 EPT라는 이름으로 표시하지 않는다. CAL/AUDIT/TEST/외부 평가는 열지 않는다. 합성 V_start 확증과 전체 연구 목표 판정은 이 모듈 진단 바깥이다.

## 빠른 실행 준비와 한계

인벤토리는 RTX5080(도구 표기16303MiB),8 physical/16 logical CPU, RAM66206961664 bytes였다. 실제 훈련 벤치마크는 아니다. NN은 GPU에서 순차 fit/evaluate하고 B1은 native CPU8을 사용한다. 준비된 membership/Stats/고정 auxiliary를 공유하고 읽기 전용 store mapping과 순서 보존 CPU prefetch(queue<=2, producer1개)를 사용한다. 배치 순서·RNG·optimizer step·정밀도·head 계산을 생략하지 않는다. GPU 다중 fit 경합·auto batch·compile·AMP·TF32를 추가하지 않는다. 속도 개선량은 아직 미측정이다.

전체 제안 cap104h: prepare4h + 모든 fit72h + 평가24h + finalize4h. 단일 fit24h 이하, 단일 평가8h 이하. 평가/검증 예약을 학습에 전용하지 않는다. B1 reference-artifact 검증 stage는 별도 metric을 꾸미지 않으며 실제 B1 예측은 두 neural paired stage에서 측정·대조한다. 시간은 persistent parent ledger에 실패·정리까지 누계하고 phase/모델 순서를 고정한다. 단일 소유 controller만 ledger를 갱신한다.

sampled child-tree private40GiB 미만을 감시하되 연속 OS memory 보증으로 표현하지 않는다. GPU OOM은 실패로 기록한다. Windows Job에 과학 worker를 대기 상태로 결속한 뒤 G를 주고, 종료/실패 뒤 tree/job gone·job closed를 확인한다. 무관한 PID나 서비스는 조작하지 않는다. source/config/input의 경계 hash 및 작업 중 stat 검사·사용 store 종료 검증을 수행하되 OS 파일 잠금 보장이라고 주장하지 않는다.

## 소스와 실행 안내

새 구현은 `c_ept/models/neural/team_set.py`, `c_ept/comparison/`과 전용 controller다. `models/`는 기존 기준선/공통 우도 interface, `ept/`는 기존 인코더·head·loss·training·checkpoint subsystem이다. 경로 이동 없이 명시적 registry로 EPT/N2 생성과 best-only 로더의 architecture identity를 묶는다. 기존 T002/S4a profile·완료 실행은 유지한다.

예제: `configs/ept_comparison.example.json`. 실행 안내는 후속 바인딩을 위한 명령 형식이며 이번 구현에서 실제 입력 Check/Start를 실행하지 않았다.

```powershell
# 먼저 승인된 clean execution source/commit, 검토 manifest, 정확한 입력/auxiliary hash를 결속한다.
& 'C:/Users/todtj/anaconda3/python.exe' -B -m c_ept.comparison.operator --config '<절대 config 경로>' --action Check
# 실제 Start는 별도 저자 실행 승인과 execution_authorized=true 결속 뒤에만 수행한다.
# Status는 명시된 동일 campaign ledger만 읽는다. latest 폴더 추측/Resume는 없다.
```

Check 기본 경로는 stdlib config/source/작은 metadata 검증이며 torch/model·dataset payload를 로드하지 않는다. Worker는 소유 Job의 release grant 전 runtime을 import하지 않는다. 예제의 null binding/false authorization은 이 경로를 실제로 시작할 수 없게 한다. 공개 push/commit도 이번 작업에 포함하지 않는다.

Start의 preflight 시간은 prepare 예산과 전체 누계에 포함한다. 각 owned phase는 시작 전 full boundary 검증, worker, Job 정리, 부모의 receipt/source/input 해시 수락까지 같은 활성 예산으로 묶는다. 큰 파일 해시는 chunk 사이에도 예산을 검사한다. 부모 수락 실패나 phase 사이/마지막 자식 뒤의 예외도 campaign ledger와 FINAL에 실패로 남기며 다음 phase로 진행하지 않는다. 파일 I/O·OS 호출을 즉시 중단하는 hard real-time 보장은 아니다. DEV는 원 T003의 동일한 전체 적법 DEV 행을 세 군에 쓰고,25%는 predictor FIT/STOP 경기 선택을 뜻한다.

성공 FINAL marker의 기록 시간도 기록 뒤 terminal seal에서 finalize/전체 예산에 청구하고 초과하면 실패로 전환한다. FINAL 안의 ledger 복사본은 그 marker 기록 전 snapshot임을 명시한다. 최종 시간 누계의 권위는 CAMPAIGN.json이다. 마지막 ledger seal 자체의 파일 I/O까지 재귀적으로 완료 시각에 넣는 hard real-time 보장을 하지 않는다.
