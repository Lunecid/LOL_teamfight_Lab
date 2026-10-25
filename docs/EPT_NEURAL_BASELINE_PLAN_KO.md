# EPT·B1·신경망 비교군 계획

상태: **계획 초안·독립 검토 대상 / 구현·실제 실행 미승인**. 작성2026-10-25, EPT-T007. 이번 저자 요청은 계획과 독립 검토이며 학습·평가를 시작하지 않는다. 아래 새 구성·수치·프로필은 모두 제안이다. 원 설계·사전 등록·기존 실행을 소급 변경하지 않는다.

## 1. 먼저 바로잡을 현재 상태

원 설계6.4절에는 **N2 팀 집합 신경망이 필수**, N1 MLP가 선택으로 적혀 있다. LSTM은 이름을 지정한 필수 비교군이 아니며, 전체 이력 순환 요약은 선택 실험이다. 따라서 “신경망 비교군을 전혀 계획하지 않았다”가 아니라, 원 설계에 있는 N2를 구현·실행 목록으로 연결해야 하는 상태다. 추가 MLP와 LSTM은 아래 질문을 위해 별도 확장으로 제안한다. 과거 다른 학위논문 과제의 AUC를 현재 EPT 비교 결과로 옮기지 않는다. [S1:241–252]

EPT S는 이미 PyTorch Transformer이고 폭128·4층·attention head4의 크기 설정이다. 기존 단일 pilot 완료 기록은18에폭·STOP-best16이며 실제 DEV 품질 비교는 아직 수행하지 않았다. S/M은 모형 계열이 아니라 동일 EPT의 크기다. [S3; C1:78–107]

`c_ept/models/`는 현재 B0/B1 및 공통 head·NLL 인터페이스 묶음이다. 모든 모델을 열거하는 registry가 아니다. `c_ept/ept/`는 제안 모형 전용 subsystem으로 인코더·예측 헤드 외에 tensor data, loss, 학습, 체크포인트, 합성 adapter를 묶는다. 그래서 `models/`에 없다는 것이 모델 미구현을 뜻하지 않는다. 현재 명명은 발견성이 낮지만 이번에는 경로를 옮기지 않는다. [C2; C3:537–551]

## 2. 비교 질문과 우선순위

| ID/역할 | 답할 질문 | 입력/처리 제안 | 우선순위·해석 |
|---|---|---|---|
| B0 | 단순 상태 기반 기준보다 낫나 | 기존 GLM, 기존 계약 유지 | 공통 scale/기준선; 신경망 아님 |
| B1 | 실용적인 부스팅 기준보다 낫나 | 기존 LightGBM head bundle | 원 주 비교 유지 |
| N2-TeamSet | 관계형 Transformer 없이 공유 선수 표현·팀 집합으로 충분한가 | 공유 선수 MLP→진영별 masked sum/mean와 count→공통 문맥; 표지점·사건·관계 입력은 아래 공통 입력 계약 아래 처리 | **원 설계 필수**, 추가 비교군 중 먼저 |
| N1-MLP | 같은 과거 정보에 일반적인 비순환 비선형 모형이면 충분한가 | 고정 길이로 펼친 공통 view와 mask·시간 정보→MLP; 출력 interface는 아래 공통 법칙 | 원 선택 실험을 구체화하는 제안 |
| N-LSTM | 같은 제한된 사건 이력을 순차 처리하면 충분한가 | 최근 사건을 시간순 단방향 LSTM, 선수/표지점/요약 문맥과 결합 | 새 선택 비교; **전체 경기 이력 확장과 별개** |
| EPT-S | 기존 관계/시간 결합 설계의 효과가 있는가 | 기존 typed attention+관계 편향+공통 예측 법칙 | 구조 주 비교 및 기준 |

N2의 구체 문맥 결합과 N1의 입력 펼침은 원문에 상세 고정되어 있지 않으므로 이번 제안이다. N2는 선수별 공유 함수, 팀별 집합 집계와 글로벌 문맥만 사용하고 이웃 상태를 반복 갱신하는 message passing을 넣지 않는다. 그것은 별도 선택 N3의 질문이다. N2의 count를 따로 주어 mean만으로 인원수를 잃지 않게 한다. LSTM은 단순히 플레이어10명을 순서대로 읽는 모델이 아니다. 사건 토큰의 실제 시각 순서를 읽고 각 view 시작에 hidden/cell을 초기화한다.

**비교 층을 두 개로 나눈다.**

1. 주 확장 제안은 동일한 허용 원시 view를 받은 EPT/N2/N1/N-LSTM의 encoder 비교다. 내부 pooling으로 버리는 정보도 모형 설계의 일부이며, 결과를 attention 하나의 인과적 효과라고 부르지 않는다. 관계·지형·분해의 주장은 원래 A1/A3E/A3T 등 지정 제거 실험으로만 검증한다.
2. B1은 이미 공학적 요약 특징을 쓰는 실용 시스템 기준이다. N1-Tabular(별도 선택안)는 **B1과 동일한 특징 표**를 쓰는 MLP이며 이 둘 사이의 tree-vs-MLP 질문에만 쓴다. N1 공통-view와 N1-Tabular는 다른 모형이므로 이름/실행/비용을 합치지 않는다. 최소 pilot에는 N1 공통-view만 포함하고 N1-Tabular 추가는 별도 결정한다.

## 3. 동일 정보와 누설 방지

공통 view는 기존 `ViewTensors`/canonical row 정의를 따른다. 새 고빈도 위치·외부 경기·ID·미래 프레임을 추가하지 않는다. 선수10·표지점34·최대128 사건과 실제 padding mask, 최근120초 사건,2–10분 요약, 마지막 두 프레임, 결측·경과 시간·프레임 나이를 동일하게 제공한다. 기존 cap·절단·dtype·시간 경계·팀 교환 규칙을 유지한다. [S1:139–166; C4:459 이후]

- 시간 cutoff는 view마다 t0(M2/가린 보기), f(M3 열린 보기), delta(M4 열린 보기) 그대로다. 가린 M3/M4의 합성은 t0 이후 실제 K/O·delta 상태를 보지 않는다. 열린 M4에 허용되는 실제 K/O는 그 조건부 평가에만 사용한다.
- 지형·위치·공동 참여의 원자료를 어떤 모형만 추가로 보지 않는다. 공통 view의 필드별 허용표를 구현 전에 작성한다. EPT는 관계 값을 attention bias에, MLP는 고정 slot의 수치 입력에, N2는 endpoint 속성이 결합된 관계 집합의 공유 함수·집계에, LSTM은 동일 사건 입력과 문맥에 쓴다. 집합 집계/학습 투영은 다를 수 있지만 외부에서 모형별로 정보를 더하거나 빼지 않는다. 특정 필드를 처리하지 않는 단순 축소안은 matched-input 표에서 제외하고 reduced-input 보조로 표시한다.
- 공통 정적 지형 artifact는 같다. 학습 지형장을 EPT만 사용하면 그 결과는 architecture+지형장 결합 효과다. parameter-matched encoder 비교에서는 위치 채널/학습 field의 포함 정책도 전 모형 동일하게 잠근다. 지형장을 제외하는 군은 원래 A3T/A6로 명명하며 기준군을 몰래 바꾸지 않는다.
- N1은 canonical player/event slot을 펼치며 slot 의존성이 생길 수 있다. N2에는 선수 순열에 대한 invariance/equivariance를 시험한다. 모든 군에 기존 team-swap frame averaging을 동일하게 적용한다. N1이 N2의 선수 순열 성질까지 자동 보장한다고 표현하지 않는다.
- N-LSTM은 보이는 사건만 stable chronological order로 정렬하고 동시각은 기존 source 순서를 따른다. 미래 사건을 잘라낸 후 정렬한다. 다른 경기/행/배치의 hidden state를 넘기지 않으며 빈 사건열은 명시적인 zero state+mask로 처리한다. BiLSTM으로 미래를 읽지 않는다. 전체10분 원시 사건 또는 경기 전체 이력은 새로운 정보량 실험으로 분리한다.
- 시간 정보/가림 표시를 없애는 군은 원 A2-age에 속하며 기본 LSTM 비교와 혼합하지 않는다.

손검산 사례: t0=600초의 가린 view에서601초 실제 킬을 추가/변경해도 encoder에 전달되는 t0 관측 입력은 같아야 한다. M2 및 완전히 합성한 t0 예측은 평가 MC stream을 고정하면 같아야 한다. 가린 조건부 head를 직접 시험할 때는 미래 관측만 바꾸고 **주어진 prospective(q,m,z)·생성 K/O·MC stream을 고정**한다. 합법 조건 자체를 바꾸면 조건부 예측이 달라질 수 있으므로 그것을 누설로 판정하지 않는다. M4 열린 view도 delta와 실제 K/O 조건이 달라지면 예측이 달라질 수 있으나, 그 결과를 t0 예측으로 보고하지 않는다. padding slot에 큰 값을 넣어도 mask 밖의 예측은 변하지 않아야 한다.

## 4. 같은 목표와 출력 법칙

추가 신경망은 인코더만 바꾸고 기본 비교에서는 EPT의 M2/M3/M4 법칙·head dimension·조건부 구조·손실을 공유한다. 공유는 소스/수학 계약을 뜻하며 학습된 가중치의 공유나 EPT checkpoint로 비교군을 초기화하는 것이 아니다. 모든 군은 독립 초기화한다.

M2 rate/mark/zone, M3 size/K/gold/O, M4 conversion의 Poisson/범주 법칙, 종류/진영 순서, exposure, gold 단위, mask, 규모 전문가와 mixture 규칙을 유지한다. `Encoded(h_tok,h_pool,mask)`의 player slot과 valid token layout을 만족시키는 adapter가 필요하다. N2/N1/LSTM도 player-conditioned head가 쓰는 player representation을 반환해야 한다. 단일 global vector만 내고 표지 head를 상수/다른 분류기로 교체하지 않는다. [C1:127–136; C3:237,305,537–551]

학습은 기존13 block 정규화 NLL 및 M3 규모 정규화와 **기존 S3의 hash-bound B0 scale을 유지**한다. 학습용/평가용 provenance 필드는 구분하되 기본안에서는 같은 고정 scale artifact를 참조한다. 구조 평가는 **M2 masked3개+M3/M4 open5개=공통8 head**를 승인된 고정 B0 scale로 정규화한다. 남은 masked5개는 개별 진단으로만 남긴다. 다중 분류·회귀 head의 손실을 B1의 다른 목표와 단순 합산해 “종합 AUC”로 만들지 않는다. [S2:79–85]

새 모형을 “MLP/LSTM-EPT-head”로 명명해 **분해된 출력 법칙을 공유하는 encoder 비교**임을 밝힌다. 분해를 꺼서 V_start를 직접 내는 A1(19분위/CRPS)은 별개 원 필수 제거 실험이며 이 비교군이 대신하지 않는다. [S1:119–123,241–248; S4:9–20]

## 5. 자료량과 auxiliary fit-scope

현재 역사적 비교는 whole-FIT B1 대 FIT/STOP25% EPT이다. 또 기존 `run_training.py`는 **whole-FIT Stats를 먼저 학습하고 나서25% proxy를 선택**하며 S3의 B0 scale을 사용한다. 따라서 기존 pilot을 “모든 학습 정보가25%인 통제 실험”이라고 부를 수 없다. [C5:178–201; S2:110]

**새 비교의 권장안은 fresh matched predictor25% + 공통 고정 FIT assets 조건부 비교다(실행 제안).** 모든 학습 정보가25%라는 zero-auxiliary 실험을 뜻하지 않는다.

1. FIT와 STOP에서 동일한25% 경기 membership을 한 번 정하고 모든 모형에 같은 manifest를 준다. row 단위 무작위 분할을 하지 않는다. 이후 모든 가린/열린 행·같은 교전의 view를 그대로 묶는다.
2. membership selection seed와 초기화/훈련 seed를 분리한다. 제안 membership seed20261022, pilot 초기화 seed20261022; 후속3 seed는20261022/23/24를 제안하되 승인 전 config가 아니다. 현 operator는 한 seed로 proxy와 학습을 함께 정하므로 그대로 여러 번 돌려 matched cohort라고 부르지 않는다. [C4:837–865]
3. predictor용 normalization/imputation/category vocabulary는 해당 FIT25%만으로 fit한다. head 초기화·class fallback·학습 prior·column dropping도 predictor fitting으로 계수하고 같은 FIT25%로 제한한다. B1도 같은 FIT25%에서 **새로 적합**한다. 기존 whole-FIT B1를 matched 군으로 재사용하지 않는다. STOP은 선택만, DEV는 비교만 한다.
4. **기본안은 기존 S3 B0 scale을 재추정하지 않고 학습과 구조 평가에 공통으로 사용**한다. source/모집단·해시를 학습용/평가용 필드에 각각 결속한다. whole-FIT supervision으로 얻은 공통 손실 가중을 썼다는 한계를 공개한다. 부분 FIT/각 fold의 B0를 다시 fit해 scale을 바꾸는 것은 상대 task weight/G8 목적함수도 바꾸므로 기본 공정성 보정에 넣지 않는다. 이는 별도 저자 승인·protocol ID·추가 B0 적합/비용이 필요한 optional objective/zero-auxiliary 확장이다. strict predictor Stats API와 공통 scale loader가 구현·검토되기 전에는 기본안도 실행하지 않는다.
5. 기존 f0 가격·tier·gold bins 등 과학적 정답 정의는 모형마다 다시 추정하거나 바꾸지 않는다. 원 고정 FIT artifact를 공유하고 **auxiliary ledger**에 각 artifact의 학습 역할·모집단·해시·라벨 사용 여부를 기록한다. ledger는 (a) 정답 정의 artifact, (b) FIT occupancy terrain/측지표 등 고정 **입력** artifact, (c) supervision-derived B0 손실 가중을 구분한다. “25% predictor fit+공유 whole-FIT 고정 assets 조건부”라고 표시하며 모든 정보가25%라는 주장을 하지 않는다. DEV/CAL/AUDIT/TEST로 auxiliary를 fit하지 않는다.
6. 원 pilot은 별도 historical 열에 남긴다. 새 주 비교는 EPT도 같은 새 프로필에서 fresh fit하여 소스·STOP·전처리 차이를 제거한다. history와 새 run을 이어붙이지 않으며 재시작/Resume하지 않는다.

FIT 패치 빼기 선택(CV)에서는 각 fold의 predictor stats/범주표/초기화도 해당 fold FIT만으로 다시 fit한다. 기본안의 B0 scale·목표 정의·고정 입력 assets는 미리 고정하고 각 provenance를 별도로 표시한다. fold holdout의 정보로 predictor 통계를 재사용하지 않는다. whole-FIT auxiliary를 공유하는 선택/비교는 그 조건 아래의 비교이며 완전히 독립적인 auxiliary 일반화 검증은 아니다. fold별 B0 scale을 바꾸는 확장은 위 별도 목적함수 결정에 속한다.

손검산: FIT100경기 중25경기 manifest를 썼다면 B1·N2·N1·LSTM·EPT의 predictor-fit 경기 집합은 정확히 같은25개여야 한다. 같은 경기 수만 맞고 실제 경기 집합이 다르면 실패다. predictor Stats가100경기를 사용했다면 기본안의 fit-scope 위반으로 거부하고 legacy 조건부 결과로 몰래 바꾸지 않는다. 원 승인된 B0 학습/평가 scale·고정 f0 정의·고정 입력 artifact가100경기에서 왔다는 사실은 ledger에 공개하며 predictor fit과 구별한다. 이 공통100경기 auxiliary 사용은 기본안에서 허용하되 모든 정보가25%라는 주장을 금지한다.

## 6. 선택·에폭·규모·계산의 공정성

최소 pilot은 모형마다 사전에 고정한 설정1개·seed1개이며 HPO를 하지 않는다. DL은 S-width128 기반, batch32, float32, AdamW/lr0.001/wd0.01/warmup10을 출발 제안으로 삼고 patience2·최대30에폭·strict STOP 개선·가장 이른 최솟값 checkpoint를 공통으로 쓴다. 16에폭이 좋았다고 다른 군의 best를16으로 강제하지 않는다. [S3; C6:4–5,72]

N1 깊이2, N2 공유함수 깊이2, LSTM 단방향1층·hidden128을 초기 제안으로 둔다. 공통 head까지 포함한 학습 parameter 수, 고정/학습 field 수, batch memory와 optimizer step을 구현 후 계수한다. EPT-S의±20% parameter band를 맞추는 폭을 **실제 fit 없이** shape/count audit에서 정한다. 맞출 수 없으면 차이를 명시하고 “parameter-matched”라고 부르지 않는다. 단순히 hidden128이면 같은 규모라고 주장하지 않는다. 용량 band는 새 제안이며 구현 전 잠근다.

B1은 CPU/native8, 최대1000 boosting round와 patience50의 기존 알고리즘 단위를 유지한다. tree round50을 신경망 patience2와 같다고 취급하지 않는다. 신경망 최대30에폭과 patience2는 서로 다른 설정이다. pilot 새 프로필의 B1 seed를 같은 식별자로 둘지는 기존 고정 seed1(P9) 변경에 해당하므로 별도 승인 항목이다. seed1 그대로인 역사적 B1에는3-seed paired 학습 변동을 적용하지 않는다. [C7:40–46,95–100]

향후 S6의 본격 조정은 원 설계대로 FIT 패치 빼기3겹에서 M2/M3 NLL을 보고 fold 평균+변동 벌점으로 선택한다. 모형군마다 제안 후보 수·탐색 범위·동일 시간 상한을 사전 고정한다. trial에는 실패/timeout과 모든 fold를 포함하고 과거 pilot 노출도 ledger에 공개한다. EPT만 튜닝하고 MLP/LSTM은 기본값 하나인 표를 최종 tuned 비교로 내지 않는다. 동일 trial 수와 동일 시간 예산을 동시에 다 채우지 못하면 실제 수행 수/시간을 보고하고 한계를 명시한다. DEV로 새 width/history/학습률을 반복 선택하지 않는다. 원 약30회 TPE를 새5개 군으로 자동 확장하지 않는다. 그 full campaign은 별도 작업서·총예산 승인이 필요하다. [S1:269–273]

탐색/pilot DL 장치는 CUDA로 통일하고 B1 CPU의 차이를 공개한다. 공정성은 같은 GPU를 강제하는 것이 아니라 동일 비교 목표·모집단·정보·탐색 기회와 비용 공개다. 최종 보고 장치는 원 설계의 CPU 비트 재현 규칙과 후속 CUDA pilot 예외를 구별한다. CUDA pilot을 최종 CPU 재현으로 주장하지 않으며, 최종 장치 정책은 다음 실행 결정에서 명시한다.

## 7. 평가와 주장

- DEV15.21의 동일 canonical eligible rows를 사용한다. 각 endpoint마다 경기 m의 행 가중은1/n_m이며 AUC는 weighted pooled AUC다. 다른 모형에서 비어 있는 행을 inner join으로 버리지 않는다. 같은 조건부 목표와 같은 exposure를 비교한다.
- 주 구조 비교 **EPT 대 B1**와 기존 주 비교9개/Holm은 유지한다. 새 EPT–N2/N1/LSTM 결과는 우선 보조·탐색이다. 이를 새 확증 주장으로 승격하려면 DEV 결과를 보기 전에 비교 family와 다중성 규칙을 저자가 별도 확정해야 한다. 원 주 비교를 유리한 신경망 비교로 교체하지 않는다.
- NLL은 위 공통8 head; M2는 실제 관측 exposure의60/120/180초 발생 AUC/Brier, M3는 열린 보기의 sign(f0)>0(무승부 음성, 별도 집계), M4는 열린 보기 actual K/O 조건의 dragon/baron/tower×team 진단이다. 주 지평60초,120/180은 보조. M4 Brier는 T003 탐색 진단이지 원 주 지표를 대체하지 않는다. [S2:3–8절]
- 단일 seed pilot은 점 추정과 경기 bootstrap 진단만 제공한다. 후속 본 비교는 세 seed의 **짝 차이 평균**과 동일 경기 재표본을 세 seed·모형·head에 함께 적용한 bootstrap2000/95%를 사용한다. 현재 metrics는 단일 seed 진단이므로 후속 multi-seed 집계기는 새로 구현·검토해야 한다. 구간은 선택된 세 훈련 seed에 조건부인 경기 표집 불확실성이며 모든 training-RNG 변동의 구간이 아니다. init seed와 평가 MC seed를 분리해 MC draws1000을 모든 군에 동일 조건으로 적용하고 MC 오차도 보고한다. 같은 MC seed라도 서로 다른 Poisson law의 난수 소비가 동일하다는 보장은 없으므로 단순 동일 seed를 동일 draw 벡터 보장이라고 쓰지 않는다. seed1 고정 B1를 세 번 복제해3 seed라고 세지 않는다.
- 원 성공 조건은 세 seed 차이의 부호 일치와 EPT 쪽으로0을 넘는 구간(및 해당 Holm 규칙)이다. 새 신경망 대조를 이기더라도 원 구조 판정 B1 패배를 지우지 않는다. 단일/빈 클래스 AUC는 null, 미측정은0이 아니다.
- 종합 가치는 같은 V_start/H=60초, 동일19분위 CRPS 정의로만 평가한다. MLP/LSTM을 binary winner만 예측하도록 바꿨다면 full structural NLL 표에 넣지 않고 endpoint-only로 제한한다. A1/V_window/승리 확률을 섞지 않는다. full composition 검증은 T003 모듈 진단 바깥의 별도 구현·승인 gate다.
- CAL/AUDIT/TEST/외부 자료는 이번 계획·구현 확인에 열지 않는다. 최종 TEST는 원 S8 한 번 규칙을 유지한다.

## 8. 구현 순서와 검증 관문 — 이번에는 수행하지 않음

1. N2→N1→LSTM 순으로 encoder interface와 공통 view/input ledger를 구현한다. 새 실험 프로필에서 membership/train/MC seed, fit-scope, shared-head architecture identity를 분리한다. 기존 T002/S4a strict profile과 완료 run은 수정하지 않는다.
2. 현재 EPT 생성자는 EPTEncoder를 고정하며 평가 adapter는 EPT/FrameAveraged 타입만 받는다. 그러므로 새 군이 현재 도구에서 바로 실행 가능하다고 주장하지 않는다. 허용된 model registry/명시적 protocol adapter를 별도 구현·검토하고 metadata에 encoder ID와 전체 source identity를 포함한다. guard를 지우거나 새 군을 EPT인 것처럼 alias하지 않는다. [C3:537–551; C8:221–224]
3. 공통 전처리/가중/출력 law 계약 tests, shape와 mask·팀 교환·N2 선수 순열, 미래 perturbation, LSTM hidden reset/empty/tie-order, M3 generated K와 M4 observed K/O 분리, FIT-only stats/fold nesting, STOP earliest best, row mismatch·empty null, checkpoint reload/identity·예산 종료를 synthetic으로 검증한다. 상수 출력의 공통 head가 모든 adapter에서 같은 확률/NLL을 내는 손검산을 포함한다.
4. 독립 소스 검토 후 새로운 clean execution checkout·실제 config와 membership/auxiliary hash를 고정한다. 저자의 해당 단계 실행 승인 없이는 production Check/Start도 하지 않는다. 계획 수락은 구현 수락·실제 입력 검증·실험 승인과 다르다.

파일 제안은 `c_ept/models/neural/`에 N1/N2/LSTM encoder 구현을 두고 `c_ept/ept/`의 기존 loss/head를 공용 계약으로 사용한다. 별도 `c_ept/model_catalog.md` 또는 다음 모듈 지도 갱신으로 모든 군의 위치를 안내한다. 디렉터리가 공정성/품질을 보장하지 않는다. 기존 EPT를 `models/`로 이동하는 리팩터링은 이 실험에 묶지 않는다. 어떤 helper를 공통 패키지로 추출할지는 후속 구현 검토에서 계산/RNG/serialization 불변을 입증한 뒤 결정한다.

## 9. 단계·실행 수·예산 제안

현재 T007의 실제 실행 예산은 **모델 fit0·평가0·서비스 호출0**이다. 문서·원문 대조·독립 검토만 한다.

| 미래 단계 | 제안 실행 수 | 관문 |
|---|---|---|
| 구현/합성 검증 | 실제 fit/inference0 | 별도 구현 승인+독립 소스 수락 |
| matched predictor25% fixed pilot | EPT/N2/N1/LSTM/B1 각1 fresh fit=5; seed1개, 기존 B0 scale 재사용/추가 B0 fit0 | 동일 membership/auxiliary 프로필·B1 seed·fit-scope 승인 후에만 |
| 학습 곡선50/100% | 자동 실행0 |25% 결과와 사전 멈춤 기준 검토 후 별도 결정; raw FIT 확장 금지 |
| S6 tuned campaign | 이번 문서로 배정0 | FIT patch-CV·후보 공간·trial/fold 수·전처리 비용 포함한 새 총예산 작업서 필요 |
| 본 비교/반복 | 선택 완료 후5군×3 seed=15 fresh fits, B0 등 auxiliary 별도 계수 | 같은 모집단, 고정 선택 설정, 새로운 명시적 전체 예산; 필수 A1 등 대체 안 함 |

위 최소 pilot의 **제안 상한**은 전체120시간: metadata/preparation4h + 모든 fit 합계72h + 모든 모형 평가 합계40h + 검증/렌더4h. 단일 fit은 기존24h 이하이며 기본5군의 fit을 모두72h 누계로 제한한다. optional B0 재적합 확장은 이 pilot에 포함하지 않는다. 단일 평가8h 이하, child host-private 표본40GiB 미만·CUDA OOM fail-closed. 합계 상한이 먼저 닿으면 남은 군을 실행하지 않고 incomplete로 보고한다. 이는5개 fit이 항상 완료된다는 보장이나120h 예상 소요가 아니다. 평가·검증 시간을 학습으로 전용하지 않는다. 실제 장치 동시 활동과 shared preprocessing 비용도 남긴다.

미래 supervisor가 persistent campaign ledger로 준비/fit/fold/평가/실패/정리를 누계하고 소유 Job만 관리해야 한다. 현재 그런 다모형 campaign enforcer가 구현됐다고 주장하지 않는다. phase와 전체 deadline, sampled memory/GPU peak, child exit/Job cleanup을 실제로 검증하기 전 실행 금지다. timeout/OOM 뒤 작은 모형·batch로 자동 교체하거나 조용히 재시도하지 않는다. 시험용 resume 기능과 실제 Resume 승인은 별개다.

실행 전 저자 결정 표: (D1) N2 필수 우선+N1/LSTM 추가 범위, (D2) fresh predictor25%와 해당 Stats·공유 whole-FIT 고정 assets 조건부 설계(기존 B0 scale 유지; scale 재추정/zero-auxiliary는 별도 확장), (D3) parameter band와 초기 설정, (D4) B1 seed1 유지인지 새 paired-seed 프로필인지, (D5) CUDA pilot/최종 CPU 보고 정책, (D6) 단계별 총예산 및 새 비교군의 탐색/확증 지위. 이번 문서는 그 선택을 구체화한 안이지 실행 결정을 대신하지 않는다.

## 10. 근거와 해석 범위

로컬 코드의 line reference는 T006 로컬 소스 기준이다. 계획 동안 기존208개 source hash를 보존한다. 새로운 실제 결과를 만들거나 과거 metadata/payload를 다시 읽지 않았다.

- **S1** 원 설계: `C:/Users/todtj/PycharmProjects/LOL_teamfight_new/docs/superpowers/specs/2026-10-20-c-ept-design.md`,119–135(목표/시계),139–166(정보),195–207(구조),241–273(실험/학습),276–305(평가).
- **S4** 승인 사전 등록: 같은 디렉터리 `2026-10-20-c-ept-evaluation-prereg-proposal.md`,4–7,9–20,60–79. 원 승인 문서는 수정하지 않았다.
- **S2** 현재 endpoint/해석: `docs/EPT_DEV_DEFINITIONS_KO.md`, 특히79–85와110. **S3** pilot 이력: `harness/tasks/EPT-T003-dev-evaluation.md:44` 및 `docs/EPT_SINGLE_PILOT_KO.md`.
- **C1** `c_ept/ept/encoder.py`; **C2** `c_ept/MODULES.md:8–16`; **C3** `c_ept/ept/heads.py`; **C4** `c_ept/ept/data.py`; **C5** `c_ept/workflows/run_training.py`; **C6** `c_ept/ept/train.py`; **C7** `c_ept/models/baselines_b1.py`; **C8** `c_ept/evaluation/adapters.py`.
- Deep Sets의 집합 집계 근거: [Zaheer et al.,2017 원 논문](https://papers.nips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html). N2의 구체 EPT 연결은 본 계획의 제안이며 이 논문이 보장하지 않는다.
- 순환 비교군의 방법 근거: [Hochreiter & Schmidhuber,1997](https://doi.org/10.1162/neco.1997.9.8.1735). LSTM이 EPT보다 좋거나 오래된 시계 문제를 자동 해결한다고 가정하지 않는다.
- 동일 조정 절차를 사용한 tabular 비교의 참고: [Gorishniy et al.,2021](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html). 어떤 모델 계열이 보편적으로 우수하다는 근거로 쓰지 않는다.
