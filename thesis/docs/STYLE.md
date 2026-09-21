# 문체·표기 규칙 (STYLE)

용어는 저널 원고 가드레일(`00_WRITING_DOSSIER.md` §6)과 같다. 새 용어는 이 표에 먼저 추가한다.

## 용어집 (국문 ↔ 영문 ↔ 기호)

| 국문 | 영문 | 기호 / 비고 |
|---|---|---|
| 교전 | engagement | 단위. "한타"는 코호트 T의 이름으로만 |
| 한타 코호트 / 소규모 교전 코호트 | teamfight T / skirmish S | T: n_min ≥ 4, S: 2 ≤ n_min ≤ 3; pick 제외 |
| 예측 시점 | prediction time / cutoff | τ = 첫 킬 − 15 s (`\tcut`) |
| 결과 시점 | outcome time / endpoint | e_h (`\eh`), h = 90 주 |
| 교전 전 상태 | pre-fight state | x_pre, S_pre |
| 승률 평가기 | win-probability evaluator | V̂ (`\Vhat`), fit85, 동결 |
| 구간 변화 | interval change | ΔV̂ (`\dV`) |
| 전략적 가치 개선 | strategic value improvement | SVI, Y_SVI = 1[ΔV̂ > 0] |
| 변화 방향 | direction of change | 목표. "크기"와 구분 |
| 방향 예측기 | direction predictor | q (`logit_state`), q_S |
| 승률·시간 스플라인 기준선 | spline baseline in win probability and time | PT_flex (`\PTflex`) |
| 균형 구간 | balanced slice | B40: 0.40 ≤ p_pre ≤ 0.60 |
| 적정 점수 | proper score | Brier, 로그 손실 |
| 점수 분해 | score decomposition | CORP: MCB, DSC, UNC (진단) |
| 짝 경기 군집 부트스트랩 | paired match-cluster bootstrap | 2,000회, 시드 7 |
| 점수 전용 | score-only | 외부 평가; 재적합 없음 |
| 비교전 구간 | quiet interval | RR3 |
| 계보 | lineage | L (정의·market_event), O (구 V) — 인용 전용 |
| 탐색적 | exploratory | 시험 패치 이전 노출 태그 |
| 이연 | deferred | F1–F6, M-F0–M-F3 |

## 금지 문구 (grep 대상)

"더 예측 가능", "규모 기울기", "재보정으로 복구", "최적 아키텍처", "AUC 상한", "전투 숙련의 증거", "T/S 기전", "합동 결과로 대체", "최초의 승률 변화 연구", "Diebold–Mariano 검정으로", "수익률/시장 효율". 부정문·인용 안에서만 허용.

## 정의·출처 규칙

- 개념의 첫 등장에서 `definition` 환경으로 형식적 정의를 적고 기원 문헌을 `\cite` 한다.
- 인용 키는 `latex/bib/definition_refs.bib`, `econometrics_for_lol.bib`, `thesis_extra.bib` 에 있는 것만. 새 문헌은 DOI 확인 후 `thesis_extra.bib` 에 VERIFIED 표기와 함께 추가.
- 계보 문서에 없는 문헌(Chong & Hendry, Fair & Shiller, Lock & Nettleton 등)은 인용하지 않는다.

## 수치 규칙

- 모든 수치는 `config/numbers.tex` 매크로 (`\aucLGBM{}` 형태는 폐기; 현재 매크로 목록은 `NUMBERS.md`).
- 구간은 `\ci{lo}{hi}`; 구간이 없는 표는 캡션에 "구간 없음"을 쓴다.
- 외부 수치는 항상 "점수 전용, 구간 없음"; 코호트 간 문장은 코호트 안 또는 예비 선언 이차 대비만.
- 양수 추출 비율을 "p = 0"으로 쓰지 않는다 ("0을 넘는 추출 없음").
- 천 단위 `{,}`, 단위 앞 띄어쓰기, AUC 소수 넷째 자리, ΔBrier 소수 다섯째 자리.

## 문장 규칙

- 한 문장에 한 주장. 결과는 표·그림을 먼저 가리키고 해석을 뒤에.
- "본 논문" = 학위논문, "선행 연구 \cite{baek2026killconditioned}" = CoG 2026, "동반 저널 원고" = `journal_manuscript_v1`.
- 미확정은 `\todo{}` (작성 미완) 또는 `\pendingauthor{}` (저자 결정 대기)로 구분한다.
