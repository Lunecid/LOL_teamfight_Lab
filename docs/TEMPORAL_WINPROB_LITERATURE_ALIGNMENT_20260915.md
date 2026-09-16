# 시간에 따른 승률 추정: 문헌과 V2의 관계

2026-09-15. 사용자 요청에 따라 원고 인용·원문·현재 구현을 대조했다. 이 문서는 설계 검토이며 신규 실험 완료 기록이 아니다.

## 추정 대상

W는 최종 Blue 승리, H_t는 t까지 관측된 이력이다. 목표는 p_t = P(W=1 | H_t). 현재 V2는 이를 관측 상태 요약 X_t와 경과 시간 t를 이용한 f_theta(X_t,t)로 근사한다. 학습 라벨 W가 고정이어도 입력 정보가 변하므로 p_t는 변한다. 이는 과거 이력을 재귀적으로 처리하는 RNN과는 구별된다. 관측된 이력으로부터의 추정이며 완전한 게임 상태나 실제 실시간 API 가용성을 보장하지 않는다.

현재 코드: gameplay/state_value.py에는 time_minutes와 time_minutes_sq가 있으며 train/temporal_winprob.py의 expanded는 snapshot_age_s만 제외한다. Maymin family에는 시간과 양 팀의 킬/타워/대형 몬스터 합계가 들어간다. V2는 이 특징 구성 및 로지스틱 모델 계열을 유지한다. 5분 구간 표본추출은 학습 방식이지 추론 갱신 주기가 아니다.

## 직접 확인한 문헌과 사용 범위

1. Kim, Lee, Chung (CoG 2020), A Confidence-Calibrated MOBA Game Winner Predictor. https://ieee-cog.org/2020/papers/paper_221.pdf . 분 단위 LoL 자료, 경기 승리 확률 및 입력 불확실성을 반영한 신경망 보정. 정확도만으로 확률의 신뢰성을 판단할 수 없다는 직접 근거. V2의 로지스틱+raw/sigmoid/isotonic은 이 논문의 불확실성 손실 구현을 재현한 것이 아니다.
2. Hodge et al., Win Prediction in Multiplayer Esports: Live Professional Match Prediction, IEEE Transactions on Games 13(4), 368–379, DOI 10.1109/TG.2019.2948469 (온라인 2019, 권호 2021). 저자 소속기관 기록: https://portal.findresearcher.sdu.dk/en/publications/win-prediction-in-multiplayer-esports-live-professional-match-pre/ . Dota 2 프로 경기 실시간 승리 예측의 직접 사례. 이번에는 초록 및 서지 확인; 상세 모델/평가 프로토콜 재현은 원문 추가 대조 필요.
3. Maymin (2021), Smart kills and worthless deaths: eSports analytics for League of Legends, DOI 10.1515/jqas-2019-0096. https://d-nb.info/1367424143/34 . LoL 상태 기반 승률과 행동 평가의 직접 근거. §3.2의 경기당 무작위 한 시점 표집은 연속 표본의 상관성을 줄이기 위한 선택이다. 현재 expanded 및 전체 표집의 완전한 재현을 주장하지 않는다.
4. Silva, Pappa, Chaimowicz (SBGames 2018), Continuous Outcome Prediction of League of Legends Competitive Matches Using Recurrent Neural Networks. https://www.sbgames.org/sbgames2018/files/papers/ComputacaoShort/188226.pdf . SimpleRNN/LSTM/GRU 및 시간 구간별 평가, 경기 단위 분할. 이력 기반 대조군의 근거이며 현재 V2가 이 방법을 구현했다는 뜻은 아니다. CoG/ToG 논문이 아니다.
5. Jalovaara (2024), Win probability estimation for strategic decision-making in esports. https://sal.aalto.fi/publications/pdf-files/theses/mas/tjal24a_public.pdf . §5.1의 5분 구간당 균등 표집은 현재 표집 선택의 보조 근거. 석사학위논문이며 동료심사 학술논문으로 표시하지 않는다.

## 추가 검증 제안 — 아직 실행하지 않음

- 기존 V2는 동적 승률을 산출할 수 있는 상태 기반 기준선으로 유지한다. 단지 확률이 시간에 따라 달라져야 한다는 이유만으로 RNN으로 교체할 필요는 없다.
- 먼저 저장된 독립 V 테스트 경기의 전체 확률 궤적과 분 단위 Brier/log loss/보정, 원시 관측 갱신 시점 및 이벤트 시점을 함께 점검한다. 기존 시간 구간 평가를 확장한다.
- 교전 직전/종료 시점은 학습의 정분 시점과 관측 나이가 다르므로 별도로 검증한다. 프레임 갱신으로 발생하는 추정 변화와 이벤트 반영 변화를 구분한다. 초 단위로 질의 가능하다는 사실은 초 단위 상태를 관측했다는 뜻이 아니다.
- 시간 및 오브젝트 구간 오차를 확인하고, 필요 시 소규모 이력 기반 대조군을 동일 입력 가용성·경기/패치 분할·평가 시점으로 비교한다. 기존에 확인한 테스트는 탐색 자료로 명시하고, 수정 모델의 확증은 별도 미사용 패치에 둔다.
- 동일하게 동결한 모델/보정기로 p_pre와 p_post를 계산하여 DeltaV = p_post - p_pre를 구한다. 경기 도중 모델 또는 보정기를 교체하면 인위적 점프가 섞이므로 금지한다.
- 최종 W로 시점별 확률 품질을 평가할 수 있지만, 단일 경기에서 실제 p_t 또는 실제 DeltaV 정답은 직접 관측되지 않는다. 보정 통과만으로 교전 기여의 타당성이 보장되지 않는다. DeltaV는 정의된 관측 구간의 승률 변화이며 교전의 인과 효과가 아니다.
- 한타 라벨 재생성 전 이 검증을 우선한다. 90초 종료 규칙·오브젝트별 고정 가중치·모든 패치 일반화를 위 문헌이 정당화한다고 주장하지 않는다.
