# P4 마무리 인계 — 03:43 KST

Opus session c3715076-18bd-4801-831b-e3ffee7e3b6b의 마지막 응답은 사용량 한도, 재개 안내 05:40 KST이다. 05:40 이전 재시도하지 않는다. 계산은 03:05 complete/validation pass이며 REPORT.md와 DEFINITION_AND_EVIDENCE.md가 존재한다. RUN_NOTES.md는 없고 REPORT의 해시 표는 뒤늦게 생성된 정의 문서를 missing으로 적고 있어 인계 문서가 미완성이다.

05:40 이후 같은 Opus 5 세션을 재개하여 다음만 수행한다. 새 학습/모델선택/테스트 기반 튜닝은 금지한다.

1. 실제 파일을 확인해 RUN_NOTES.md를 작성하고 보고서의 missing 문서 표를 정정한다. 원본 수치·모델·예측·코드·해시 스냅샷은 보존하고 문서 수정의 시각/이유를 남긴다.
2. SHAP는 개별 특징 수준에서 raw ridge log-odds, 그룹 수준에서 최종 isotonic 보정 q 확률을 설명한다. 이를 혼동하지 않도록 정의/표/캡션을 대조한다. 기존 키 additivity_max_abs_residual_vs_actual_pipeline_output은 raw score 척도이지 final probability 검증이 아님을 부연한다.
3. isotonic을 단순 step function이라 한 표현을 정정한다. 설치 sklearn1.6.1 _build_f는 scipy.interpolate.interp1d(kind='linear')로 threshold 사이를 보간한다. 동률/평탄 구간은 있지만 전 구간의 계단 함수는 아니다. 예측 재계산 없이 별도 문서 정오표를 추가한다. 근거: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html 및 outputs/q_v3_baselines/codex_shap_audit.json의 실제 설치 소스.
4. A90 TEST의 ridge_isotonic은 6행에서 p=1, 그중4행의 y=0이다(Codex 원시 CSV 확인). 전체 Log loss .703057이 상수 .693183보다 나쁘다는 사실을 유지한다. 클리핑을 새로 골라 결과를 개선하거나 다른 후보를 사후 승자로 바꾸지 않는다.
5. 최종 인계에서 필수 검사와 남은 과학적 조건을 분리하고 종료 상태를 남긴다. Codex의 독립 감사 파일 codex_metric_audit.json/codex_shap_audit.json을 링크한다.

Codex 확인: 512행의 그룹 Shapley(7그룹,128연합,TRAIN background256)에 대해 저장된 기여 합+baseline이 q_final과 일치하는지 독립 계산했다. 개별 특징의 log-odds 기여 합을 최종 확률 기여로 해석하면 안 된다. SHAP의 기본 인용 정보는 https://arxiv.org/abs/1705.07874 에서 직접 확인했다. 모든 연합을 열거해도 상관된 입력의 혼합 상태가 현실적인지/인과적인지는 별도 문제이다.
