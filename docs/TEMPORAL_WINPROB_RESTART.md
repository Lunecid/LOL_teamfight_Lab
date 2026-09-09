# 연구 목표 재정리와 독립 승률 모델 실행

상태: v2 실행 및 산출물 감사 완료. 9,198경기에서 26,712개 교전을 측정했다.
[결과와 잔여 한계](TEMPORAL_WINPROB_RESULTS.md). 교전 예측기의 새 표적 학습은 아직 수행하지 않았다.

## 사용자 정정 반영

핵심 질문은 '교전 확률을 추가하면 최종 경기 승패를 더 잘 맞히는가'가 아니라
**'한타 전후 어느 팀의 승리 확률이 얼마나 변했고, 그 변화를 한타 전에 예측할 수 있는가'**다.
기존 50k 실험의 A/B 스태킹 결과는 과거 부가 분석으로 보존한다. 그 결과의 무개선을
사용자가 원하는 한타 승률 변화 연구의 실패로 해석하지 않는다.

이번 실행의 첫 단계는 한타와 독립적인 상태별 승률 모델이다. 다음 단계에서 고정된
모델을 기존 정의의 engagement 경계에 적용한다. 교전 예측기의 재학습이나 새로운
최종 승패 스태킹은 이번 단계에 넣지 않는다.

## 논문에서 가져온 것과 우리가 정한 것

1. **Maymin (2021), Smart kills and worthless deaths: eSports analytics for League of Legends.**
   Journal of Quantitative Analysis in Sports 17(1), 11–27. §3.2의 경과 시간 및 양 팀
   킬·타워·대형 몬스터 7개 입력 로지스틱 구조, 경기당 무작위 한 분의 표본을 계승한다.
   [원문](https://d-nb.info/1367424143/34), DOI 10.1515/jqas-2019-0096.
   여기서는 공개된 구조를 재학습한다. 과거 학습 가중치를 가져온 정확한 재현이 아니다.
   현재 몬스터 종류를 합산하는 매핑과 사용 가능한 2분 이후 그리드는 데이터에 따른 변경이다.
2. **Kim, Lee, Chung (2020), A Confidence-Calibrated MOBA Game Winner Predictor.**
   IEEE CoG. 승자 분류 정확도와 확률의 신뢰성을 구분하는 근거로 사용한다.
   [공식 원문](https://ieee-cog.org/2020/papers/paper_221.pdf).
   이 실행은 그 논문의 MLP나 data-uncertainty loss를 구현했다고 주장하지 않는다.
3. **Ferreira & Faganeli Pucer (2025), Quantifying Player Death Impact in League of Legends.**
   SCORES'25 학생 연구 심포지엄, pp.51–54. 별도 검증 자료를 이용한 isotonic 보정 및
   관측 구간 전후 확률 차이의 참고 근거다.
   [원문](https://zalozba.fri.uni-lj.si/SCORES2025.pdf#page=61).
   XGBoost와 SHAP 개인 귀속은 이번 실행에서 재현하지 않는다.
4. **Jalovaara (2024), Win probability estimation for strategic decision-making in esports.**
   Aalto 석사학위논문. §4.3의 WPA `w(z)-w(x)`와 가치모형/행동평가의 분리를 참고한다.
   [원문](https://sal.aalto.fi/publications/pdf-files/theses/mas/tjal24a_public.pdf).
   이 논문의 행동 지속 구간과 우리의 한타 경계는 다르다.

우리 확장형은 기존 362개 상태 중 snapshot_age_s를 제외한 361개 입력을 사용한 정규화 로지스틱 회귀다. 종류별 드래곤,
영혼 이벤트, 바론·장로 획득과 이후 사망 이력을 구분하며, 수동 전략 보상 점수를
라벨에 더하지 않는다. 이 입력 설계는 **우리 확장**이다. 정확한 버프 소유·만료는
이력으로 완전히 복원되지 않는다. 단순 원형은 오브젝트를 합산하므로 비교 기준으로
쓰고, 오브젝트 구분 요구를 충족하는 확장형의 보정 방법을 주 분석용으로 선택한다.

## 사전에 고정한 실행 절차

첫 실행(v1) 후 고빈도 곡선 검수에서 스냅숏 경과 시간에 따른 주기적 왜곡을 발견했다.
고정 정각 그리드는 API 프레임의 소폭 지연 때문에 직전 프레임을 선택한다. 학습 표본
첫 200개의 snapshot_age_s는 59.28~60.00초에 몰려 있었지만 실제 사건 조회에서는
0~60초를 사용한다. 좁은 학습 범위로 표준화한 이 항목이 분 사이에서 과도한 외삽을
일으켰다. v1 수치/그림/소스 snapshot은 `outputs/temporal_winprob_v1`에 보존하고,
이 필드를 승률 설명변수에서 제거한 v2를 별도 실행한다. 신선도 기록은 계속 남긴다.
이는 최종 AUC를 높이기 위한 선택이 아니라, 게임 상태를 고정하고 수집 시각 정보만
바꿨을 때 생기는 왜곡을 막는 수정이다. 테스트에 해당 불변성 검사를 추가했다.

- 소스 자료: `outputs/state_value_main_50k`. 이미 구축한 49,677경기의 캐시를 읽는다.
  학습에는 `value_states`와 `value_times`만 사용하고 교전 `pre/post`나 라벨은 사용하지 않는다.
  기존 코드가 생성한 비종료 분 단위 그리드는 2분부터 시작한다. 0–2분 성능은 주장하지 않는다.
- 기존 가치 학습 10,007경기를 고정 hash 순서로 fit 70%, calibration 15%, selection 15%로 분리한다.
  각 경기에서 seed 기반 무작위 한 시점을 선택한다. 표본 선택은 승패나 한타 발생에 의존하지 않는다.
- Maymin 구조: 7개 입력, 절편 포함 로지스틱 회귀, 비정규화, 수치 안정을 위한 표준화.
  대형 몬스터 합계는 elemental dragons + Baron + Elder + Herald + Horde + Atakhan이다.
  한 마리/종류별 게임 규칙 차이를 원형의 보편적 가치로 해석하지 않는다.
- 확장형: fit 경기 내부 3-fold CV, C={.001,.01,.1,1}, log loss로 선택한다.
- raw/sigmoid/isotonic 후보. 보정기는 calibration 경기만 사용하고, 각 모델 계열 내 후보는
  selection 경기의 log loss로 선택한다. 테스트나 한타 변화량으로 선택하지 않는다.
- 모델/선택 규칙을 저장·고정한 다음 기존 가치 검증 4,986경기에서 평가한다.
  경기당 무작위 한 시점의 주 지표와 전체 분 단위 곡선의 시간대별 지표를 구분한다.
  이 50k 자료는 앞선 탐색에서 사용됐으므로 완전히 새로운 외부 시험이라고 부르지 않는다.
- 한타 분석은 이 네 분할과 겹치지 않는 기존 predict_test 9,844경기에서 수행한다.
  기존 50k 교전 ID 모집단을 유지한다. 원래 전체 캐시의 전수 재분석은 아니다.

## 시각 계약과 출력

`V(S_t)`는 같은 고정 모형을 시점별 관측 상태에 적용한 승리 확률이다.
모형이 초마다 재학습되는 것은 아니다. 각 상태는 해당 시점 이하의 프레임/이벤트만 쓴다.

원천에 직접적인 피해 시작/종료 로그가 없으므로 다음 시점을 명시적으로 구분한다.

| 열 | 실제 의미 |
|---|---|
| pre | 기존 cutoff τ의 1ms 이전. τ는 첫 킬 15초 전의 예측 기준이며 실제 전투 시작을 관측한 값은 아님 |
| last_kill_proxy | 정의된 클러스터의 마지막 킬 포함 시점. 실제 모든 전투가 종료됐다는 보장은 없음 |
| plus_30s | 마지막 킬 이후 30초 |
| plus_60s | 마지막 킬 이후 60초 |

후속 시점이 경기 종료 이상이거나 마지막 관측 프레임보다 뒤면 결측으로 남긴다.
직후 대용 경계와 후속 구간을 합쳐 하나의 '즉시 효과'라고 표시하지 않는다.
각 행에 query/snapshot 시각과 원형·확장형 확률 및 delta를 보존한다.
일반 성장·다른 지역 사건도 상태 변화에 포함되므로 delta는 **관측 전후 변화**이며
한타가 없었을 세계와 비교한 순수 인과효과가 아니다.

분 단위 성장/HP 상태와 더 세밀한 사건 이력이 혼합되므로 초 단위 실제 승률을
완전히 복원했다고 주장하지 않는다. 입력의 시점별 신선도도 결과에 함께 보고한다.
등위 회귀의 평탄 구간에서 생기는 delta=0은 임의로 Blue/Red에 배정하지 않는다.

## 실행과 결과 경로

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
python scripts/run_temporal_winprob.py --dataset outputs/state_value_main_50k --previous-eval outputs/state_value_main_50k_eval --cache-dir D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13 --out-dir outputs/temporal_winprob_v2
python -m pytest tests/test_temporal_winprob.py tests/test_state_value.py -q
python scripts/audit_temporal_winprob.py --out-dir outputs/temporal_winprob_v2 --cache-dir D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13
```

`outputs/temporal_winprob_v2/status.json`이 실제 단계와 PID를 기록한다. 로그는
`outputs/temporal_winprob_v2.log`. 완료 시 `results.json` 생성.
`independent_time_curves.npz`는 한타와 독립적인 경기 승률 곡선,
`engagement_changes.npz`는 한타 전후 및 후속 변화량이다.
`protocol.json`, `sampled_minutes.json`, `selection.json`, 모형 파일/hash로 근거를 보존한다.
기존 모형과 결과를 덮어쓰지 않는다.
