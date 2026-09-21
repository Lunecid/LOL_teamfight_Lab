# 문체·표기 규칙 (STYLE)

원고 전체에서 같은 개념은 같은 말로 부른다. 새 용어를 도입하면 이 표에 먼저 추가한다.

## 용어집 (국문 ↔ 영문)

첫 등장 시 "국문(영문)" 으로 병기하고, 이후에는 국문만 쓴다. 코드 식별자는 `\texttt{}`.

| 국문 | 영문 | 비고 |
|---|---|---|
| 교전 | engagement | 방송 용어 "한타"는 서론에서 한 번만 언급 |
| 온셋 | onset (engage time) | 기호 $t_e$ (`\tengage`) |
| 교전 국소화 | engagement localization | "검출(detection)"은 알고리즘 코드명(`teamfight_v2`)을 가리킬 때만 |
| 킬 조건부 | kill-conditioned | |
| 교환가치 라벨 | exchange-value label | |
| 관측 창 / 라벨 창 | observation window / label window | $W_{\mathrm{obs}}$, $W_{\mathrm{lab}}$ |
| 구간 | bin | 5 초 구간 6 개 |
| 계단 유지 | step-hold (piecewise-constant, forward-fill) | |
| 무누출 계약 | no-leakage contract | 가정 3.1–3.3 |
| 공개 텔레메트리 | public telemetry | Riot Match-V5 |
| 프레임 / 사건 | frame / event | 60 초 프레임, 밀리초 사건 |
| 분 단위 | minute-resolution | |
| 정보 입도 | information granularity | |
| 공변량 이동 | covariate shift | |
| 선택 편향 | selection bias | Berkson 의 역설 |
| 노드·전역·사건 특징 | node / global / event features | 76 / 26 / 44 |
| 거시 시퀀스 | macro sequence | 순차 모델 입력, 95 차원 |
| 테이블 요약 | tabular summary | $\phi_{\mathrm{tab}}$ |
| 정합 입력 | matched input | LightGBM 과 같은 2,980 차원 입력의 MLP |
| 플레이어 상호작용 그래프 | player interaction graph | 가우스 커널 인접행렬 |
| 시공간 그래프 신경망 | spatio-temporal graph neural network | ST-GNN |
| 계층적 융합 | layered fusion | 게이트 융합 |
| 스태킹 | stacking | Wolpert 1992 |
| 처치 | treatment | T1–T7 (절제 인자) |
| 절제 실험 | ablation study | |
| 기준선 | baseline | |
| 하위 집단 | subgroup | 국면 / 골드 격차 / 유형 / 패치 |
| 보정 | calibration | ECE, 온도 조정 |
| 시간순 패치 홀드아웃 | chronological patch hold-out | 15.14 / 15.15 / 15.16 |
| 골드셋 | gold set | 검출기 검증용 표본 |
| 주석자 | annotator | 트랙 A / B |
| 감사 | audit | 코드–논문 대조 |

## 정의·출처 규칙

- 개념을 처음 쓰는 곳에서 `definition` 환경으로 **형식적 정의**를 적고, 그 개념이 **처음 제안된 문헌**을 `\cite` 로 단다.
- 기원이 다른 분야(신호 검출 이론, 기상 예보, 금융 기술 분석 등)에 있으면 그 사실을 한 문장으로 밝힌다.
- 정의 없이 쓰는 약어는 없다. 약어는 첫 등장 시 풀어 쓴다.

## 수치 규칙

- 실험·데이터 수치는 `config/numbers.tex` 매크로로만 쓴다 (`\aucLGBM{}`). 원고에 숫자를 직접 치지 않는다.
- 설계 상수(18 s, 4,000 단위 등)는 `core/config.py` 가 진실 공급원이며 부록 C 표에 전부 나열한다.
- 천 단위 구분은 `2{,}980` 처럼 `{,}` 를 쓴다. 단위 앞에는 띄어쓴다 (`30 초`, `1,800 단위`, `11 pt`).
- AUC 는 소수점 셋째 자리, 앞자리 0 생략 (`.675`). 차이는 `0.049` 처럼 0 을 붙인다.

## 문장 규칙

- 한 문장에 한 주장. 한 문단에 한 논지.
- 결과를 말할 때는 표·그림 번호를 먼저 가리키고 해석을 뒤에 둔다.
- "논문/본 논문" 은 이 학위논문을, "논문 \cite{baek2026killconditioned}" 은 CoG 2026 발표 논문을 가리킨다.
- 미확정 내용은 `\todo{}` 로 표시하고 `docs/OUTLINE.md` 상태 열에 반영한다.

## 표·그림·알고리즘 규칙

- 표 캡션은 위, 그림 캡션은 아래 (부산대 가이드).
- 파일 하나에 객체 하나. 파일 이름은 라벨에서 만든다 (`tab:label-weights` → `tables/tab_label_weights.tex`).
- 그림 원본은 `figures/src/` 에 PDF(벡터) 우선, PNG 는 300 dpi 이상.
- 알고리즘은 `algorithmic` 의사코드로, 입력·출력·복잡도를 함께 적는다.
