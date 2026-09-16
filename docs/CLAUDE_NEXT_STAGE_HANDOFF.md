# Claude 실행 인계: 동적 승률 기반 engagement 출력 검증

인계일: 2026-09-09. 기준 커밋: `ece1168` (`codex/engagement-state-value`).
역할: Codex는 연구 설계·결과 감사, Claude는 아래 명세의 구현·실험·보고 담당.
사용자는 현재 설계의 근거·인용·결과 전체를 Claude에게 넘기고 다음 단계를 실행하도록 요청했다.

## 1. 이번에 실제로 수행할 일

계획만 답하지 말고 검증 코드를 작성하여 실행하고 결과 파일까지 작성한다.
**현재 고정된 가치 모델의 교전 시각 신뢰성, 측정 경계 민감도, 오브젝트 반영을 검증한다.**
검증에서 결함을 찾으면 재현 근거와 최소 수정안을 보고한다. 고정 모형·기존 결과·정의는
이번 작업에서 변경하지 않는다. 새로운 engagement 예측기를 학습하거나 최종 승패 스태킹을
다시 돌리지 않는다. 다음 출력 계약의 최종 채택은 결과 회수 후 Codex와 사용자가 결정한다.

작업 경로: `C:/Users/todtj/문서/LOL_Teamfight/worktrees/engagement-state-value`
읽기 전용 원천 캐시: `D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13`
이 worktree 안에서 새 검증 스크립트·테스트·문서·산출물만 작성한다.
원본 `C:/Users/todtj/PycharmProjects/LOL_teamfight`와 원천 캐시는 수정하지 않는다.
새 패키지 설치, 원격 게시, git reset/checkout/merge/commit/push, 파일 삭제는 필요 없다.
다른 에이전트는 생성하지 말고 이번 Claude 세션에서 수행한다.

## 2. 우선 읽을 문서와 구현

현재 목표·가치 출력에 관한 우선순위는 이 인계서 → 아래 temporal 문서 → 과거 문서다.
과거 market_event/market_lex와 최종 승패 스태킹을 새 목표로 오인하지 않는다.

1. `docs/TEMPORAL_WINPROB_RESTART.md`: 논문에서 계승한 것/우리 확장, 고정 분할, 시간 계약.
2. `docs/TEMPORAL_WINPROB_RESULTS.md`: 최신 결과와 v1 오류·v2 수정·남은 한계.
3. `docs/ENGAGEMENT_DEFINITION_V3.md`, `docs/DEFINITION_EVIDENCE.md`,
   `docs/DEFINITION_LITERATURE.md`, `docs/references/definition_refs.bib`:
   기존 정의 산출 파이프라인·수치·문헌 대응. 문헌 메모와 BibTeX의 VERIFIED/미확인 표기를 구분한다.
4. `docs/RELEASE_REPORT_V3.md`, `docs/ENGAGEMENT_WINNER_DEFINITION.md`,
   `docs/ENGAGEMENT_SCALE_DEFINITION.md`: 기존 CoG/정의/결과 라벨의 역사적 맥락.
   이 문서의 물질적 승자 라벨을 이번 가치 변화 라벨로 그대로 가져오지 않는다.
5. `gameplay/state_value.py`, `train/temporal_winprob.py`,
   `scripts/run_temporal_winprob.py`, `scripts/audit_temporal_winprob.py`,
   `tests/test_temporal_winprob.py`, `tests/test_state_value.py`.
6. 필요한 기존 API: `data/cache_io.py`, `data/index_split.py`,
   `train/state_value_experiment.py`의 metrics 및 경기 단위 가중치.

## 3. 연구 목표와 설계 근거

주 기여는 데이터에서 engagement 정의를 산출하고 교전 이전 정보로 결과를 예측하는 것이다.
가치 모델은 작은 보조 모듈이다. 어떤 게임의 데이터든 무조건 이해하는 범용 모델을 이번에
구현했다고 주장하지 않는다. LoL은 프레임워크의 적용 사례다.

정의 산출은 단순한 임의 상수 나열이 아니다. 기존 코드/문서는 킬 간격의 log 밀도 골짜기와
공간적 챔피언 공유율 등으로 경계 후보를 구하고 안정성·패치별 차이를 평가한다.
단, 존재 반경/리드 타임/최소 인원 등 일부 선택은 도메인 규칙 또는 운영적 선택이다.
Halfaker의 세션 식별 방법을 킬 간격에 적용하는 것은 우리 전이이며 그 논문의 LoL 한타 정의가 아니다.
정의 파이프라인의 정확한 현재 상수는 코드/BoundarySpec와 문서의 일치 여부를 확인해서 기록한다.

승률 모형은 `V(S_t)=P(final Blue win | state observed by t)`로 학습한다.
최종 승패 W는 이진 라벨이지만 시점마다 조건부 입력이 달라지므로 확률은 변한다.
출력 후보는 `ΔV = V(S_after) - V(S_before)`이다. Blue 기준 양수/음수/0을 보존한다.
전후 상태에는 한타 외 성장·다른 지역 사건도 들어가므로 이는 관측 차이이며 인과효과가 아니다.
수동 오브젝트 점수를 합산하지 않는다. 입력 선택과 모델·경계 선택의 임의성까지 사라진 것은 아니다.

### 가치 모델 문헌과 적용 범위

- Philip Z. Maymin (2021, online 2020), *Smart kills and worthless deaths: eSports analytics
  for League of Legends*, JQAS 17(1):11–27. DOI `10.1515/jqas-2019-0096`.
  원문 https://d-nb.info/1367424143/34 , §3.2.
  시간·양 팀 킬/타워/대형 몬스터의 7개 입력 로지스틱, 경기당 무작위 한 분의 표본을 계승했다.
  원래 가중치/데이터의 정확 재현은 아니다. 원형은 총 골드를 제외하고 몬스터를 합산한다.
- Dong-Hee Kim, Changwoo Lee, Ki-Seok Chung (2020), *A Confidence-Calibrated MOBA Game
  Winner Predictor*, IEEE CoG. DOI `10.1109/COG47356.2020.9231878`.
  https://ieee-cog.org/2020/papers/paper_221.pdf
  분류 성능과 확률 보정의 구분 근거다. 해당 MLP·data-uncertainty loss를 구현한 것은 아니다.
- Ruben Ferreira, Jana Faganeli Pucer (2025), *Quantifying Player Death Impact in League
  of Legends*, SCORES 학생 연구 심포지엄 pp.51–54. DOI `10.51939/scores25.12`.
  https://zalozba.fri.uni-lj.si/SCORES2025.pdf#page=61
  별도 보정 집합·isotonic·사망 전/부활 후 확률 차이의 참고. XGBoost/SHAP를 재현한 것은 아니다.
  저널 논문과 근거 수준을 동일하게 표현하지 않는다.
- Perttu Jalovaara (2024), *Win probability estimation for strategic decision-making in
  esports*, Aalto 석사학위논문 §4.3.
  https://sal.aalto.fi/publications/pdf-files/theses/mas/tjal24a_public.pdf
  상태 가치 차이 WPA의 참고. 행동 구간과 우리의 교전 경계는 다르다.

새로운 문헌 주장을 추가한다면 원문 확인을 수행한다. 위 문헌의 미구현 방법이나
우리만의 361개 입력 설계를 선행연구가 직접 입증한 것처럼 표현하지 않는다.

## 4. 고정된 구현과 결과

가치 원형은 7변수 비정규화 로지스틱, 확장형은 362개 상태 필드 중 snapshot_age_s를
제외한 361개 변수의 정규화 로지스틱이다. 종류별 드래곤·영혼·바론·장로 획득/사망 이력을
구분하지만 버프 보유/만료를 완전히 복원한 것은 아니다.
학습 C 후보 .001/.01/.1/1은 fit 안의 3-fold에서 선택한다.
보정은 calibration에서 학습하고 selection의 log loss로 raw/sigmoid/isotonic 중 선택했다.
원형 raw, 확장형 sigmoid가 선택됐다. 이번 검증에서는 재선택/재보정하지 않는다.

기존 49,677경기 탐색 자료의 역할: fit 7,004 / calibration 1,501 / selection 1,502 /
test 4,986 / engagement 9,844. 역할 간 경기 중복 0.
승률 모형은 engagement pre/post와 무관한 분 그리드에서 경기당 무작위 한 점을 학습했다.
그리드는 2분 이후이며 이 자료는 과거에도 탐색했으므로 새로운 외부 시험 자료가 아니다.

시험 4,986경기에서 원형 AUC .827210 / Brier .170115 / log loss .509808,
확장형 AUC .846790 / Brier .159351 / log loss .478012.
교전 분석은 9,198경기 26,712행. +30초 26,095행, +60초 25,350행.
확장형 절대 Δ 중앙값은 즉시대용/+30/+60 각각 8.56/8.58/9.13%p.
즉시 Blue 평균 Δ -1.88%p, 원형/확장형 방향 일치 85.916%.
즉시 구간 오브젝트 창: Baron225 / Dragon1060 / Elder18 / Soul63 (중복 가능).
pre/end 스냅숏 나이 중앙값32.33초/p90 53.82초/max60.04초.
이전 감사와 관련 테스트12개는 통과했으나 이는 도메인 타당성 확정이 아니다.

시간 계약: pre=cutoff-1ms; cutoff=첫 킬-15초; end=클러스터 마지막 킬 inclusive;
추가 query=end+30초/end+60초. 종료 이상 또는 마지막 스냅숏 이후는 결측.
pre는 실제 전투 시작이 아니고 end는 실제 전투 종료가 아니다.

v1은 정각 학습 grid의 snapshot_age_s가 약60초에 몰린 반면 사건 query는0~60초라
표준화 외삽으로 주기적 왜곡이 생겼다. v2에서는 이 입력만 제거했다.
구모형은 다른 입력 고정/age0,30,60일 때 한 사례 .0711/.1857/.4046,
v2는 모두 .4043이었다. v1 재사용 금지. v2에도 모든 시각 해상도 문제가 해결된 것은 아니다.

### 읽기 전용 산출물

`outputs/temporal_winprob_v2/` 안의 protocol.json, sampled_minutes.json, selection.json,
results.json, audit.json, clock_diagnostic.json, *_model.joblib,
engagement_changes.npz, independent_time_curves.npz, source_snapshot/ 전체.
원천 가공 자료 `outputs/state_value_main_50k/` (schema.json, manifest.json, matches/*.npz),
기존 분할 `outputs/state_value_main_50k_eval/match_splits.json`.
집계 사본은 `docs/experiments/temporal_winprob_v2_*.json`.
`docs/CLAUDE_NEXT_STAGE_MANIFEST.json`의 경로·SHA256을 시작/종료 때 확인한다.
기존 run_temporal_winprob.py를 완료된 v2 디렉터리에 재실행하지 않는다.

## 5. 구현할 세 검증

### A. 교전 시점의 승률 신뢰성

- 기존 26,712행 전수의 pre/end/+30/+60 각 확률을 최종 W와 비교한다.
  원형·확장형 모두 AUC, Brier, log loss, 10개 고정 동일너비 확률 bin의 reliability/ECE를 보고한다.
- 동일 경기 반복 교전을 독립 표본으로 취급하지 않는다. 각 평가 부분집합 안에서 경기별 총
  가중치가 같도록 주 지표를 계산하고 원시 행 수와 고유 경기 수를 함께 보고한다.
- 시간대 2–10/10–20/20–30/30+ 및 snapshot age 0–15/15–30/30–45/45초 이상을 분해한다.
  나이는 해당 query 기준이다. 미래 스냅숏으로 보간하지 않는다.
- 시점별 모집단 차이를 구분한다. 전체 가용 표본 표와 네 시점 모두 유효한 공통 표본 표를 분리한다.
- 경기 단위 bootstrap(고정 seed7, 500회, 95% percentile CI)으로 적어도 전체 시점별
  Brier/log loss 및 두 모형 차이를 평가한다. 희소/단일 클래스면 NA와 이유를 기록한다.
- 이것은 사건 조건부 탐색 검증이다. 원래 무작위 시점 성능과 직접 비교해 인과적 우열을 주장하지 않는다.

### B. 출력 경계 민감도

- end/+30/+60 Δ의 평균·중앙값·절대값 분위수, 부호 분포, 경계 간 방향 전환율을 보고한다.
- 공통 행에서 pairwise 방향 일치, 변화량 차이, 원형/확장형 차이를 계산한다.
- |Δ|<=0/.01/.02/.05 (확률 단위; 0/1/2/5%p)별 중립으로 남기는 표본 비율을 민감도로만 보고한다.
  이를 자동으로 최종 threshold로 채택하거나 승자에 무작위 배정하지 않는다.
- cutoff/end의 스냅숏이 같은지, age 구간 및 규모별(min 참여자: <=1,2–3,>=4)
  결과를 분해한다. Blue 평균 음의 Δ가 특정 조건과 관련되는지도 기술적으로 점검한다.
- 종료에 가까워 후속 관측이 빠지는 경기와 남는 경기의 차이를 보고한다.

### C. 오브젝트가 입력/출력에 실제로 반영되는가

- 원천 이벤트로 pre→end, end→+30, +30→+60의 Baron/Dragon/Elder/Soul 획득을
  팀·종류·시각별 집계한다. 유효 구간만 비교하고 기존 즉시 집계를 재현한다.
- 사건 직전(t-1ms)/직후(t)의 상태를 같은 StateBuilder로 조회해 관련 필드가
  변하는지 확인한다. 동시에 일어난 사건은 별도 표기하고 효과를 분리했다고 주장하지 않는다.
- 희소한 즉시 Elder18창은 모두 확인하고, 나머지 범주는 Δ 크기와 무관한 고정 hash로
  최대20개씩 소스 사례를 검사한다. +30/+60에서 처음 획득된 사례도 포함한다.
- 모델의 입력 반응과 전후 관측 Δ를 구분한다. 버프 만료/사망 해제의 복원이 없는 경우 명시한다.
- 오브젝트 포함 여부의 예측 차이가 곧 해당 오브젝트의 인과적 가치라는 해석은 하지 않는다.

## 6. 완료 계약과 반환 형식

새 스크립트 권장 경로: `scripts/validate_temporal_winprob_claude.py`.
새 산출물: `outputs/temporal_winprob_claude_validation/`.
처음 status.json을 running으로 생성하고 단계·PID·시각을 갱신한다. 종료 때 complete/failed와 이유를 기록한다.
UTF-8 명시, PYTHONDONTWRITEBYTECODE=1, 기존 캐시 쓰기/인덱스 캐시 생성을 끈다.
Windows/경로와 원천 API는 기존 실행기 패턴을 따른다. 필요한 명령 실패를 숨기지 않는다.

필수 반환물:

1. `docs/CLAUDE_TEMPORAL_VALIDATION_REPORT.md`: 한국어, 질문별 결과·근거·한계·설계자에게 필요한 결정.
2. `outputs/temporal_winprob_claude_validation/results.json`: 지표·CI·분모·제외 이유·분해 결과.
3. `.../status.json`, `.../manifest.json`: 명령·Python/패키지 버전·소스/입력/모형 hash·실제 실행 시간.
4. 필요한 행 단위 npz/CSV, reliability·민감도 그림, 원천 사건 증거와 계산 로그.
5. 검증 코드와 의미 있는 계산/시각/결측 검사를 포함한 테스트, 실제 실행 결과.

새 구현에서 계산 버그가 나면 고쳐서 완료한다. 기존 설계의 결함 발견은 frozen 입력을
바꾸지 말고 재현과 수정 제안으로 반환한다. 근거 없이 모든 검증에 PASS를 부여하지 않는다.
끝 메시지에 완료 여부, 보고서 절대경로, 가장 중요한 발견 세 가지, 남은 결정 사항을 적는다.
