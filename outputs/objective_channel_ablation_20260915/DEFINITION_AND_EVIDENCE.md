# 정의와 근거 (objective_channel_ablation_20260915)

근거 종류: L=문헌, D=게임 규칙/구현, E=우리 데이터 검증, A=설계 선택 또는 상속 정의.

| 용어 | 정의 | 근거 | 비고 |
|---|---|---|---|
| W | GAME_END winningTeam의 최종 Blue 승리 | D | V 학습 목표 |
| V_A | expanded StateV2 logistic C=.01, raw, 최종+5 OOF | A(부모) | 변경 없음 |
| V_B_noobj | V_A 입력에서 이름에 baron/elder/dragon/soul/herald/horde/atakhan이 포함된 176열 제거(185열 유지), 나머지 동일, raw | A | 명시 채널 절제, 인과 절제 아님 |
| 보조 보정 | V_CAL sigmoid 대 raw를 V_SELECT로 선택 | A | 선택: raw |
| Δ, Y | V(e 상태) − V(q_pre 상태), 같은 어댑터; Y=1[Δ>0] | A(상속) | 정확한 0: 0 |
| 불일치 | Y_A ≠ Y_B, 행·경기가중; 경기 bootstrap 1,000회 | A |  |
| same_pre_post_frame | 사전·종료 최신 프레임 시각 동일 | A | L 이후 새 프레임 없음보다 좁음 |
| full / after 창 | (q_pre, e] / (L, e], 하한 제외·상한 포함 | D/A | TRAIN 원시 사건으로 이전 연구에서 검증(E) |
| 획득 팀 기준 Δ | Blue 단독 +Δ, Red 단독 −Δ | A | 기술 통계, 인과 가치 아님 |
| unknown_objective_team_count | V2 모델 입력 기준: 쿼리 시각 이하 사건 중 팀이 100/200으로 식별되지 않은 ELITE_MONSTER_KILL(killerTeamId 및 처치자 로스터 팀으로도 미식별)과 BUILDING_KILL/TURRET_PLATE_DESTROYED(teamId 미식별)의 누적 수. gameplay/state_value_v2.py가 teamId 미식별 DRAGON_SOUL_GIVEN을 legacy builder에 넘기기 전에 제거하고 unassigned_soul_events로 따로 세므로(모델 입력 아님), 미할당 영혼은 이 열에 들어가지 않는다. 오브젝트와 구조물이 섞여 있어 다른 입력을 바꾸지 않고 분리할 수 없으므로 그대로 유지했다. | D | 유지. 정정 사항은 errata.json |
| q 표적 의존성 | 동결 specialist를 Y_B로 채점 | A | 재학습 없음 |

## 문헌

- Maymin (2021), Smart kills and worthless deaths, doi:10.1515/jqas-2019-0096 — 상태 승률 변화 가치평가(L).
- Kim, Lee & Chung, IEEE CoG 2020, https://ieee-cog.org/2020/papers/paper_221.pdf — 승률 확률 품질(L). 재현 아님.
- Jacobs & Wallach, Measurement and Fairness, arXiv:1912.05511 — 구성 개념과 측정(L).
- 부모 근거 장부: outputs/label_validity_full_20260915/DEFINITION_AND_EVIDENCE.md, docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md.

## 주장하지 않는 것

- causal objective effect
- objective-free data or complete removal of objective proxies
- semantic correctness of labels
- fresh confirmation
- best q performance for B labels
- human validation

