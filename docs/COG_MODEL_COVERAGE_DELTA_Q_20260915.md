# CoG 모델 전체 포함: ΔV 비교 명세 확장

2026-09-15. 사용자 요구: CoG에서 사용한 모델들을 새 ΔV 표적에서도 모두 비교. **이번 산출물은 모델 출처 확인과 실행 계약 확장이다. 새 모델 학습은 아직 미실행이다.** 부모 명세 DELTA_Q_FAIR_BALANCED_PROTOCOL_20260915.md와 함께 읽으며 모델 범위에 대해서는 이 추가 명세가 우선한다.

## 1. 실제 출판 버전과 추가 진단의 구분

주 출처는 tmp/pdfs/paper_308_review/paper_308_text.txt §III-D / Table II / Table III이다. CoG 308은 다음 8개 paradigm–representation pairings를 보고한다. docs/CoG2026_Paper.md의 25+ 지원 아키텍처 목록은 확장 기술보고서의 범위이며 실제 비교 완료 목록으로 대체하지 않는다.

| ID | 모델 | CoG 입력/역할 | 새 ΔV 실행 범위 |
|---|---|---|---|
| C01 | LightGBM | engineered X_tab | 공통 snapshot 입력 및 CoG 표현 대조 모두 |
| C02 | matched-input MLP | LightGBM과 같은 X_tab | **plain MLP** 추가; residual MLP로 대체하지 않음 |
| C03 | Bi-GRU | macro sequence | 사전 시계열 표현 트랙 |
| C04 | Transformer | macro sequence | 같은 사전 시계열 표현 트랙 |
| C05 | GraphSAGE | 마지막 player graph | 사전 그래프 표현 트랙 |
| C06 | ST-GNN | 6 bins의 player graphs | 사전 시공간 구조 표현 트랙 |
| C07 | Cross-Attn | event tokens + player states | 사전 이벤트/플레이어 표현 트랙 |
| C08 | Layered Fusion, logit=0 | global + graph + event | 사전 멀티모달 표현 트랙 |
| C09 | Layered+Logit | 위 fusion + LightGBM logit | **118-era 추가 진단**으로 별도 포함; 308의 아홉 번째 주 모델이라고 쓰지 않음 |

C09 근거: tmp/pdfs/paper118/page-6.png, 원본 run_paper_full.py의 PHASE_B_MODELS. PHASE_A_MODELS/PHASE_B_MODELS를 합치면 위 9개가 일치하며 MLP도 포함된다. 원본 경로 C:/Users/todtj/PycharmProjects/LOL_teamfight는 읽기 전용이다. 308의 8개는 저장된 원문에서 root가 직접 재확인했고, 118의 추가 진단은 Luna의 원문 이미지 조사 및 root의 runner 대조로 확인했다.

## 2. 동일 입력 비교와 표현 비교를 분리

**Track A: 동일 정보의 알고리즘 비교.** 기존 명세의 동일 352열에 logistic, LightGBM, plain MLP, residual MLP를 적용한다. plain MLP는 2 hidden layers [width,width], ReLU/dropout, 1-logit head; width [128,256,512] × dropout [.1,.3]의 6후보. optimizer/lr/weight decay/batch/cap/stop/seed/calibration은 부모 residual MLP와 동일하다. CoG MLP의 단순 구조 계승이며 과거 2980열을 새 352열 결과로 혼동하지 않는다.

**Track B: CoG 계열 표현 비교.** C01–C09 모두를 새 라벨·동일 평가 키·같은 사전 cutoff에서 재적합한다. 시계열을 한 snapshot 반복으로 만들거나 feature 차원을 시간으로 위장하지 않는다. 그래프·event 모델에 tabular 행을 임의 reshape해서 원 CoG 모델 재현이라고 부르지 않는다. 원래 정보 표현을 포함하는 비교는 순수 learner 효과와 다르므로 별도 표로 보고한다.

새 ΔV/현재 입력 계약을 유지하기 위해 입력 adapter가 필요하다. 기본 개발 기준은 사전 30초, 6개 5초 bins, node/global/event 정보의 관측 가능성이다. 시점 query는 pre-25s,...,pre의 6개이며 원시 사건 범위는 (pre-30s,pre]. 마지막 프레임 유지와 cutoff 이하 사건만 허용하고, 충분한 관측이 없는 앞 구간은 명시적 마스크를 제공한다. 모든 모델의 missing/mask 정보 접근을 문서화한다. 테스트 결과를 보고 lookback을 고르지 않는다.

**현재 제외한 좌표·사후 역할·새 champion 입력을 조용히 재도입하지 않는다.** 예를 들어 기존 spatial adjacency/MPNN 계열 입력이 좌표를 요구하면 기존 계약과의 충돌을 입력 manifest에 표시한다. 허용된 정보에서만 만든 team-relation graph 등의 대체가 필요하면 원래 공간 그래프와 다른 adapter임을 명시하고, 그 수학식·처리·baseline 정보 접근을 새 모델 적합 전에 설계 검토한다. 미정 입력 계약을 Claude가 임의로 결정해 full fit하면 안 된다. 이벤트 value descriptor가 과거 수동 라벨 계수를 포함하는지도 점검해 raw 관측 필드와 분리한다.

CoG 원래 입력을 그대로 사용하는 것이 목적일 경우, 별도 정보 추가 실험으로 명시하고 현재 주 비교를 덮어쓰지 않는다. 현재 확정된 것은 전체 모델 **포함 범위**이며, B의 모델별 adapter와 파라미터 카드는 구현 전 추가로 닫아야 한다.

## 3. 원래 선정 설정의 위치

308 Table II의 설정은 원래 연구에서 선택된 참고값이다: Bi-GRU 2x128; Transformer d64/head4/layers2; GraphSAGE/ST-GNN hidden96/layers2; Cross-Attn width128/head4; Layered width192. 새 ΔV에서는 이러한 값으로 출발하는 별도 고정 참고 실행과, 동일한 후보 수로 튜닝한 실행을 혼동하지 않는다. 기존 모델 AUC 수치를 새 표적에 복사하지 않는다.

B의 각 모델도 후보 6개/seed 7,42,123/경기 가중 학습/동일 Q_CAL·Q_SELECT라는 부모 예산 원칙을 따른다. 정확한 후보 목록, representation hash, optimizer·schedule·stop metric·성능/자원 cap은 model_cards.json에 고정한 뒤 학습한다. 이미 TEST를 확인했으므로 이 명세 고정이 연구 전체를 사전등록 연구로 바꾸지는 않는다.

LightGBM passthrough는 TRAIN의 동일 경기에서 fit한 예측을 학습 입력으로 쓰지 않는다. 별도 경기 그룹 OOF logit을 만들고, validation/test에는 TRAIN 최종 모델을 사용한다. **V 라벨의 OOF와 stacking logit의 OOF는 서로 다른 두 계약**이다. 기존 기술보고서의 train+validation refit/validation 재사용 동작을 그대로 실행하지 않는다. 전체 파이프라인에서 Q_CAL/Q_SELECT와 test 분리를 유지한다.

원본 run_paper_full.py는 옛 label/extractor, GPU/rtx5080/AMP/TF32 옵션, 기존 출력 경로가 포함되므로 그대로 실행하지 않는다. 새 runner/출력 root 아래에서 CPU<=4 및 동결 데이터 규약을 적용한다. 학습 규모·실패 때문에 모델을 조용히 누락하지 않는다. 자원이 부족한 항목은 미완료로 명시한다.

## 4. 실행 순서와 완료 정의

1. **완료:** 308 원문 8개 + 118/runner의 Layered+Logit 출처 대조 및 본 문서 기록.
2. **다음 구현:** 새 runner에 C01–C09와 추가 logistic/residual 모델 registry 생성; 각 ID에 원본 구현 경로/현재 adapter/입력 계약/후보 목록/상태를 기록한다.
3. Track A는 기존 352열을 재사용한다. Track B는 위 제약하에서 adapter와 동일 정보 baseline을 설계 검토한 뒤 TRAIN-only 계약 테스트를 수행한다. 계약이 미정인 모델을 검증 완료로 표시하지 않는다.
4. 새 Y_h 라벨로 T/N 각각 전체 TRAIN 학습, 독립 Q_CAL/Q_SELECT 선택, manifest 동결 후 TEST/외부 평가. h90 주·h60/120 보조 및 balanced B40/B45/시간·패치 분석은 부모 명세를 유지한다.
5. 원시 row-key·prediction 저장, 독립 지표 재계산, paired 경기 bootstrap, calibration, 전체/균형 SHAP, 이전/이후 상태를 보고한다. 9개 필수 항목 각각 실행·검증 완료 또는 명시적 미완료 사유가 있어야 한다. 전 항목의 결과가 있어야 'CoG 전체 모델 재비교 완료'로 말한다.

사람 검토나 새 가치평가 학습을 추가하지 않는다. 본 확장은 모델 범위의 누락을 보완하며 ΔV를 연속 회귀 목표로 변경하지 않는다. 새 q의 주 정답은 여전히 Y=1[ΔV>0]이다.
