# 오브젝트를 포함하는 engagement 출력: 구현과 파일럿 결과

상태: 구현 및 5,000경기 파일럿 완료. 별도 50,000경기 확대 실험도 완료했다.
확대 결과는 `STATE_VALUE_50K_RESULTS_20260909.md`를 참조한다. 아래 내용은 파일럿 당시의 기록이다.
기준 코드: Claude의 v3.3 수정 완료 커밋 `d607e3a`.
작업 브랜치: `codex/engagement-state-value`.

## 구현한 출력

교전 전후 상태를 최종 경기 승패로 학습한 고정 로지스틱 가치모형에 입력한다.
`delta_value = V(S_post) - V(S_pre)`가 양수면 Blue(1), 음수면 Red(0), 정확한
동률·유효하지 않은 상태는 -1이다. 교전 예측 모델은 기존 7,106개 사전 입력으로
이 라벨의 확률을 예측한다. 기존 engagement 정의와 Eq.3/market_event 라벨은 보존했다.

바론·장로 획득과 이후 사망 이력, 종류별 드래곤, 명시적 영혼 이벤트, 구조물과
참가자 성장·생존·챔피언을 별도 362개 상태로 표현한다. 전략적 보상 점수를
수동으로 합산하지 않는다. 획득 이력은 정확한 현재 버프 보유나 만료 상태와
다르며, 캐시의 불완전한 버프 플래그를 그대로 사실로 사용하지 않았다.

이는 **모형이 평가한 가치 변화**다. 교전의 인과적 기여나 객관적인 모든 의미의
교전 승리로 해석하지 않는다. 자세한 계약과 한계는 `STATE_VALUE_EXPERIMENT.md`에 있다.

## 완료된 파일럿

시드 기반 5,000경기를 선택했고, 비종료 상태를 구성할 수 없는 짧은 경기 46개를
제외한 4,954경기를 구축했다. 모든 분할은 경기 단위다. 가치 검증은 481경기의
14,400개 상태, engagement 시험은 968경기의 2,757개 교전이다.

초기 C=1 로지스틱 모형은 확률을 지나치게 확신했다. 이를 숨기거나 결과를
덮어쓰지 않고, 가치 학습 경기 내부의 3-fold 교차검증으로 C를 선택하는
제한된 정규화 선택 절차를 추가했다. 후보는 0.0001/0.001/0.01/0.1/1이며,
경기별 가중 log loss가 최소인 C=0.001이 선택됐다. 교전 시험 성능으로 C를 고르지 않았다.

| 가치 검증 모형 | AUC ↑ | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|
| 초기 고정 C=1 | 0.6972 | 0.2926 | 1.2710 |
| 학습 경기 내부 CV로 C 선택 | 0.8381 | 0.1634 | 0.4883 |
| 같은 C, 주요 오브젝트 변수 제거 | 0.8282 | 0.1684 | 0.5019 |
| 가치 학습 경기의 승리 비율 상수 | 0.5000 | 0.2491 | 0.6913 |

오브젝트를 포함한 모형의 관측 지표가 더 낫지만, 이 표만으로 각 오브젝트의
인과적 가치나 모든 라벨의 타당성이 입증되는 것은 아니다. 제거 모형은 동일 C를
쓴 진단이며 각각 최적 튜닝한 모형끼리의 비교가 아니다.

| 교전/경기 예측 | AUC ↑ | Brier ↓ |
|---|---:|---:|
| engagement 가치 상승 예측 | 0.5684 | 0.2511 |
| A: 교전 이전 입력 → 최종 경기 승패 | 0.7754 | 0.1935 |
| B: 같은 입력 + OOF engagement 확률 | 0.7756 | 0.1933 |
| C: 같은 입력 + 실현된 라벨, 사후 참고 | 0.7978 | 0.1834 |

B의 A 대비 AUC 개선 95% 구간은 [-0.0046, +0.0060], Brier 개선 구간은
[-0.0021, +0.0027]이다(경기 단위 paired bootstrap 300회, 학습 모형은 고정).
**추가 예측 성능 개선은 확인되지 않았다.** C는 미래 정보를 포함한 참고치여서
사전 예측 성능이나 실현 가능한 성과로 제시하지 않는다. Engagement의 AUC가
0.5보다 높아도 Brier가 약 0.25이므로 유용한 확률 예측이 확립됐다고 볼 수 없다.

시험 교전의 9.79%는 절대 가치 변화가 0.01 미만이다. 작은 변화의 부호는
모형 오차에 민감할 수 있다. 이 진단을 근거로 사후에 임의의 제거 임계값을 적용하지 않았다.

## 희귀 오브젝트와 확대 실험

장로가 관측된 가치 학습 경기는 Blue 9경기, Red 12경기이고 검증은 각 4경기뿐이다.
두 팀을 단순 합산하면 동일 경기가 중복될 수 있다. 반복된 상태 35/39개를
35/39개의 독립적인 장로 경기로 해석하면 안 된다. 바론과 드래곤을 포함했다는
사실만으로 장로의 영향까지 충분히 학습했다고 주장하지 않는다.

2026-09-09 13:50 KST에 최초 파일럿의 선택 경기 5,000개 전체를 제외한
새로운 50,000경기 표본으로 확대 실행을 시작했다. 정규화 선택 규칙, 입력,
라벨 정의, 교전 및 A/B 모델 설정은 고정한다. 새 표본은 이 파일럿과 겹치지 않지만,
원래 engagement 정의가 같은 원천 코퍼스에서 탐색됐다는 한계는 남는다.
50,000경기를 전체 약 210,000경기의 전수 검증이라고 부르지 않는다.

## 산출물과 재현

경로는 worktree `C:/Users/todtj/문서/LOL_Teamfight/worktrees/engagement-state-value` 기준이다.

- 초기 결과: `outputs/state_value_pilot_eval/results.json`.
- 정규화 선택 후 결과: `outputs/state_value_pilot_cv_eval/results.json`.
- 학습 내부 CV 기록: `outputs/state_value_pilot_cv_eval/value_regularization_selection.json`.
- 확대 실행 상태: `outputs/state_value_main_status.json`.
- 확대 데이터: `outputs/state_value_main_50k`; 평가: `outputs/state_value_main_50k_eval`.
- 확대 로그: `outputs/state_value_main_50k.log`, `outputs/state_value_main_50k.stderr.log`.

```powershell
python scripts/build_state_value_dataset.py --cache-dir D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13 --out-dir outputs/state_value_main_50k --n-matches 50000 --input full --exclude-selection outputs/state_value_pilot_full/selection.json
python scripts/run_state_value_experiment.py --dataset outputs/state_value_main_50k --out-dir outputs/state_value_main_50k_eval
```

확대 실행부터 settings에 소스 SHA-256과 제외 목록 hash를 저장한다. 이전 파일럿은
이 검사가 도입되기 전의 산출물로 보존하며, 재현 시 새로운 출력 디렉터리를 사용한다.
새 실행은 모델/라이브러리 버전, 라벨/타임스탬프/mask, 경기 분할, OOF membership,
모형 파일과 hash를 함께 저장한다. 원천 캐시는 읽기만 한다.

관련 테스트 21개 통과: 새 상태/라벨/OOF 검증과 기존 label window, attribution,
frame alignment, causal anchors, presets. 제외 목록 도입 후 별도 30경기 구축도
성공했고 파일럿과의 교집합이 비어 있음을 확인했다.
