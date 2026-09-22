# 보완 실험 수치 레지스트리 (NUMBERS_SUPP)

`latex/config/numbers_supp.tex`와 `latex/tables/gen/*.tex`는 `tools/gen_supp.py`가 아래 출처에서 **생성**한다.
손으로 고치지 않는다. `tools/check_numbers_supp.py`가 재생성해 차이가 있으면 실패한다.
모든 값은 보완 실험(15.16 노출 이후의 진단, `docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md`)이며 헤드라인을 대체하지 않는다.
반올림: Brier 4자리, ΔBrier·구간 5자리(부호 포함), 비율 3자리, 초 1자리.

## 생성 표

| 파일 | 내용 | 출처 |
|---|---|---|
| `latex/tables/gen/tab_infogroups_arms.tex` | E1 정보군 arm별 TEST Brier (T, S) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.*.arm_brier`, `selected` |
| `latex/tables/gen/tab_infogroups_contrasts.tex` | E1 정보군 대비 ΔBrier와 95% 구간 (T, S) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.*.contrasts` |
| `latex/tables/gen/tab_frame_strata.tex` | E2 관측 갱신 2×2 층화 (T, S) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `section_5_4.*.cells` |
| `latex/tables/gen/tab_shold.tex` | E2 S_hold 산술 분해 (T) | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full` |
| `latex/tables/gen/tab_ext_bootstrap.tex` | E3 외부 표본의 짝 경기 군집 부트스트랩 (T/S × KR/NA1) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary` |
| `latex/tables/gen/tab_ext_common_h.tex` | E3 공통 경기 대비 H (KR, NA1) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `common_match_H` |
| `latex/tables/gen/tab_horizon_strata_T.tex` | E4 지평 층화와 h60 라벨 이식 (T) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T` |
| `latex/tables/gen/tab_horizon_strata_S.tex` | E4 지평 층화와 h60 라벨 이식 (S) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S` |
| `latex/tables/gen/tab_peer_transfer.tex` | E4 동료 평가기 라벨 이식 (T) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer` |
| `latex/tables/gen/tab_oat.tex` | E4 정의 상수 일변량 변경의 사례 구성 census (확장: I/SHOPEX/MR/MD) | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings`, `vs_ref` |
| `latex/tables/gen/tab_material_axes.tex` | E5 SVI와 물질 결과 축의 동일 사례 대응 (T, S) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.*.material_axes` |
| `latex/tables/gen/tab_nextobj.tex` | E5 다음 엘리트 오브젝트와의 연관 (T) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj` |

## 매크로

| 매크로 | 값 | 통계량 | 분자 | 분모 | 조건 | 가중 | 출처 파일 → 필드 |
|---|---|---|---|---|---|---|---|
| `\igTBrierFzero` | 0.2391 | M-F0 TEST Brier (T) | Σ w·(p−y)² | T 15.16 TEST 행 | 항등 보정기; 후보는 Q_SELECT 경기 가중 Brier 최저 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.arm_brier.M-F0` |
| `\igDimFzero` | 4 | M-F0 입력 차원 | — | — | — | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.selected.M-F0.n_features` |
| `\igTBrierFone` | 0.2384 | M-F1 TEST Brier (T) | Σ w·(p−y)² | T 15.16 TEST 행 | 항등 보정기; 후보는 Q_SELECT 경기 가중 Brier 최저 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.arm_brier.M-F1` |
| `\igDimFone` | 94 | M-F1 입력 차원 | — | — | — | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.selected.M-F1.n_features` |
| `\igTBrierFtwo` | 0.2353 | M-F2 TEST Brier (T) | Σ w·(p−y)² | T 15.16 TEST 행 | 항등 보정기; 후보는 Q_SELECT 경기 가중 Brier 최저 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.arm_brier.M-F2` |
| `\igDimFtwo` | 262 | M-F2 입력 차원 | — | — | — | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.selected.M-F2.n_features` |
| `\igTBrierFthree` | 0.2355 | M-F3 TEST Brier (T) | Σ w·(p−y)² | T 15.16 TEST 행 | 항등 보정기; 후보는 Q_SELECT 경기 가중 Brier 최저 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.arm_brier.M-F3` |
| `\igDimFthree` | 352 | M-F3 입력 차원 | — | — | — | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.selected.M-F3.n_features` |
| `\igTDEgF` | -0.00291 | ΔBrier D_E_given_F (T) | mean_m[mean_i(Brier_a − Brier_b)] | 24{,}020 경기 / 32{,}981 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.contrasts.D_E_given_F.delta_match_mean` |
| `\igTDEgFLo` | -0.00341 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...D_E_given_F.ci95[0]` |
| `\igTDEgFHi` | -0.00239 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...D_E_given_F.ci95[1]` |
| `\igTDFgE` | +0.00016 | ΔBrier D_F_given_E (T) | mean_m[mean_i(Brier_a − Brier_b)] | 24{,}020 경기 / 32{,}981 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.contrasts.D_F_given_E.delta_match_mean` |
| `\igTDFgELo` | -0.00023 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...D_F_given_E.ci95[0]` |
| `\igTDFgEHi` | +0.00056 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...D_F_given_E.ci95[1]` |
| `\igTFoneVsFzero` | -0.00072 | ΔBrier M-F1_minus_M-F0 (T) | mean_m[mean_i(Brier_a − Brier_b)] | 24{,}020 경기 / 32{,}981 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.contrasts.M-F1_minus_M-F0.delta_match_mean` |
| `\igTFoneVsFzeroLo` | -0.00118 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F1_minus_M-F0.ci95[0]` |
| `\igTFoneVsFzeroHi` | -0.00024 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F1_minus_M-F0.ci95[1]` |
| `\igTFtwoVsFzero` | -0.00378 | ΔBrier M-F2_minus_M-F0 (T) | mean_m[mean_i(Brier_a − Brier_b)] | 24{,}020 경기 / 32{,}981 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.contrasts.M-F2_minus_M-F0.delta_match_mean` |
| `\igTFtwoVsFzeroLo` | -0.00434 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F2_minus_M-F0.ci95[0]` |
| `\igTFtwoVsFzeroHi` | -0.00320 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F2_minus_M-F0.ci95[1]` |
| `\igTFthreeVsFzero` | -0.00363 | ΔBrier M-F3_minus_M-F0 (T) | mean_m[mean_i(Brier_a − Brier_b)] | 24{,}020 경기 / 32{,}981 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.T.contrasts.M-F3_minus_M-F0.delta_match_mean` |
| `\igTFthreeVsFzeroLo` | -0.00427 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F3_minus_M-F0.ci95[0]` |
| `\igTFthreeVsFzeroHi` | -0.00296 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F3_minus_M-F0.ci95[1]` |
| `\igSBrierFzero` | 0.2489 | M-F0 TEST Brier (S) | Σ w·(p−y)² | S 15.16 TEST 행 | 항등 보정기; 후보는 Q_SELECT 경기 가중 Brier 최저 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.S.arm_brier.M-F0` |
| `\igSBrierFone` | 0.2470 | M-F1 TEST Brier (S) | Σ w·(p−y)² | S 15.16 TEST 행 | 항등 보정기; 후보는 Q_SELECT 경기 가중 Brier 최저 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.S.arm_brier.M-F1` |
| `\igSBrierFtwo` | 0.2451 | M-F2 TEST Brier (S) | Σ w·(p−y)² | S 15.16 TEST 행 | 항등 보정기; 후보는 Q_SELECT 경기 가중 Brier 최저 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.S.arm_brier.M-F2` |
| `\igSBrierFthree` | 0.2440 | M-F3 TEST Brier (S) | Σ w·(p−y)² | S 15.16 TEST 행 | 항등 보정기; 후보는 Q_SELECT 경기 가중 Brier 최저 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.S.arm_brier.M-F3` |
| `\igSDEgF` | -0.00302 | ΔBrier D_E_given_F (S) | mean_m[mean_i(Brier_a − Brier_b)] | 49{,}730 경기 / 101{,}205 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.S.contrasts.D_E_given_F.delta_match_mean` |
| `\igSDEgFLo` | -0.00333 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...D_E_given_F.ci95[0]` |
| `\igSDEgFHi` | -0.00269 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...D_E_given_F.ci95[1]` |
| `\igSDFgE` | -0.00118 | ΔBrier D_F_given_E (S) | mean_m[mean_i(Brier_a − Brier_b)] | 49{,}730 경기 / 101{,}205 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.S.contrasts.D_F_given_E.delta_match_mean` |
| `\igSDFgELo` | -0.00152 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...D_F_given_E.ci95[0]` |
| `\igSDFgEHi` | -0.00084 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...D_F_given_E.ci95[1]` |
| `\igSFoneVsFzero` | -0.00195 | ΔBrier M-F1_minus_M-F0 (S) | mean_m[mean_i(Brier_a − Brier_b)] | 49{,}730 경기 / 101{,}205 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.S.contrasts.M-F1_minus_M-F0.delta_match_mean` |
| `\igSFoneVsFzeroLo` | -0.00227 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F1_minus_M-F0.ci95[0]` |
| `\igSFoneVsFzeroHi` | -0.00163 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F1_minus_M-F0.ci95[1]` |
| `\igSFtwoVsFzero` | -0.00378 | ΔBrier M-F2_minus_M-F0 (S) | mean_m[mean_i(Brier_a − Brier_b)] | 49{,}730 경기 / 101{,}205 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.S.contrasts.M-F2_minus_M-F0.delta_match_mean` |
| `\igSFtwoVsFzeroLo` | -0.00417 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F2_minus_M-F0.ci95[0]` |
| `\igSFtwoVsFzeroHi` | -0.00341 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F2_minus_M-F0.ci95[1]` |
| `\igSFthreeVsFzero` | -0.00497 | ΔBrier M-F3_minus_M-F0 (S) | mean_m[mean_i(Brier_a − Brier_b)] | 49{,}730 경기 / 101{,}205 행 | 음수 = 앞 arm이 낮음 | 경기 평균 | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `cohorts.S.contrasts.M-F3_minus_M-F0.delta_match_mean` |
| `\igSFthreeVsFzeroLo` | -0.00540 | 95% 구간 하한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F3_minus_M-F0.ci95[0]` |
| `\igSFthreeVsFzeroHi` | -0.00452 | 95% 구간 상한 | — | — | 경기 군집 백분위 부트스트랩 | — | `SUPPLEMENTARY_E1_INFOGROUPS_20260921.json` `...M-F3_minus_M-F0.ci95[1]` |
| `\fsTSameFrame` | 6.7 | 같은 프레임 비율 (%) | 예측·결과 시점이 같은 분 프레임을 읽은 행 | T TEST 행 | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `section_5_4.T.same_frame_rate` |
| `\fsTPreAgeLt` | 37.0 | 예측 시점 프레임 나이 <30 s 비율 (%) | pre_frame_age < 30 s | T TEST 행 | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...pre_age_lt30_rate` |
| `\fsTPreAgeMed` | 37.0 | 예측 시점 프레임 나이 중앙값 (s) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...pre_age_median_s` |
| `\fsTFollowMed` | 74.0 | 실제 추적 길이 중앙값 (s) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...followup_median_s` |
| `\fsTsyN` | 2{,}204 | 셀 행 수 (T, same_young) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.n` |
| `\fsTsyM` | 2{,}164 | 셀 경기 수 (T, same_young) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.n_matches` |
| `\fsTsyPsvi` | 0.530 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.P_SVI` |
| `\fsTsyEdv` | 0.094 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.E_abs_dV` |
| `\fsTsyDb` | -0.00090 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.delta_brier_q_minus_pt` |
| `\fsTsoN` | 20 | 셀 행 수 (T, same_old) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.n` |
| `\fsTsoM` | 20 | 셀 경기 수 (T, same_old) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.n_matches` |
| `\fsTsoPsvi` | 0.450 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.P_SVI` |
| `\fsTsoEdv` | 0.057 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.E_abs_dV` |
| `\fsTsoDb` | +0.00427 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.delta_brier_q_minus_pt` |
| `\fsTnyN` | 10{,}003 | 셀 행 수 (T, new_young) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.n` |
| `\fsTnyM` | 8{,}979 | 셀 경기 수 (T, new_young) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.n_matches` |
| `\fsTnyPsvi` | 0.486 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.P_SVI` |
| `\fsTnyEdv` | 0.119 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.E_abs_dV` |
| `\fsTnyDb` | -0.00347 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.delta_brier_q_minus_pt` |
| `\fsTnoN` | 20{,}754 | 셀 행 수 (T, new_old) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.n` |
| `\fsTnoM` | 16{,}923 | 셀 경기 수 (T, new_old) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.n_matches` |
| `\fsTnoPsvi` | 0.497 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.P_SVI` |
| `\fsTnoEdv` | 0.116 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.E_abs_dV` |
| `\fsTnoDb` | -0.00398 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.delta_brier_q_minus_pt` |
| `\fsTallN` | 32{,}981 | 셀 행 수 (T, all) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.n` |
| `\fsTallM` | 24{,}020 | 셀 경기 수 (T, all) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.n_matches` |
| `\fsTallPsvi` | 0.496 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.P_SVI` |
| `\fsTallEdv` | 0.115 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.E_abs_dV` |
| `\fsTallDb` | -0.00373 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.delta_brier_q_minus_pt` |
| `\fsSSameFrame` | 13.9 | 같은 프레임 비율 (%) | 예측·결과 시점이 같은 분 프레임을 읽은 행 | S TEST 행 | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `section_5_4.S.same_frame_rate` |
| `\fsSPreAgeLt` | 43.7 | 예측 시점 프레임 나이 <30 s 비율 (%) | pre_frame_age < 30 s | S TEST 행 | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...pre_age_lt30_rate` |
| `\fsSPreAgeMed` | 33.3 | 예측 시점 프레임 나이 중앙값 (s) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...pre_age_median_s` |
| `\fsSFollowMed` | 60.1 | 실제 추적 길이 중앙값 (s) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...followup_median_s` |
| `\fsSsyN` | 12{,}912 | 셀 행 수 (S, same_young) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.n` |
| `\fsSsyM` | 11{,}712 | 셀 경기 수 (S, same_young) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.n_matches` |
| `\fsSsyPsvi` | 0.569 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.P_SVI` |
| `\fsSsyEdv` | 0.078 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.E_abs_dV` |
| `\fsSsyDb` | -0.00141 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_young.delta_brier_q_minus_pt` |
| `\fsSsoN` | 1{,}114 | 셀 행 수 (S, same_old) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.n` |
| `\fsSsoM` | 1{,}101 | 셀 경기 수 (S, same_old) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.n_matches` |
| `\fsSsoPsvi` | 0.590 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.P_SVI` |
| `\fsSsoEdv` | 0.077 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.E_abs_dV` |
| `\fsSsoDb` | +0.00007 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.same_old.delta_brier_q_minus_pt` |
| `\fsSnyN` | 31{,}314 | 셀 행 수 (S, new_young) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.n` |
| `\fsSnyM` | 24{,}695 | 셀 경기 수 (S, new_young) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.n_matches` |
| `\fsSnyPsvi` | 0.493 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.P_SVI` |
| `\fsSnyEdv` | 0.091 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.E_abs_dV` |
| `\fsSnyDb` | -0.00396 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_young.delta_brier_q_minus_pt` |
| `\fsSnoN` | 55{,}865 | 셀 행 수 (S, new_old) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.n` |
| `\fsSnoM` | 36{,}561 | 셀 경기 수 (S, new_old) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.n_matches` |
| `\fsSnoPsvi` | 0.503 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.P_SVI` |
| `\fsSnoEdv` | 0.091 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.E_abs_dV` |
| `\fsSnoDb` | -0.00310 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.new_old.delta_brier_q_minus_pt` |
| `\fsSallN` | 101{,}205 | 셀 행 수 (S, all) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.n` |
| `\fsSallM` | 49{,}730 | 셀 경기 수 (S, all) | — | — | — | — | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.n_matches` |
| `\fsSallPsvi` | 0.509 | SVI 양성률 | Y=1 행 | 셀 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.P_SVI` |
| `\fsSallEdv` | 0.089 | E|ΔV| | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.E_abs_dV` |
| `\fsSallDb` | -0.00335 | ΔBrier q − PT_flex | Brier(q) − Brier(PT_flex) | 셀 행 | 동결 점수; 셀 안에서 가중 재계산 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `...cells.all.delta_brier_q_minus_pt` |
| `\shN` | 32{,}981 | S_hold 재구성 성공 행 (T TEST) | — | — | — | — | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.n_ok |
| `\shIdentity` | 1.000 | ΔV = d_clock + d_update 성립 비율 | |ΔV − (d_clock + d_update)| ≤ 1e−6 | 성공 행 | — | — | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.arith_identity_rate |
| `\shDvMean` | -0.00175 | ΔV̂ 평균 | — | — | — | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.delta_V.mean |
| `\shDvAbs` | 0.1154 | ΔV̂ 평균 절댓값 | — | — | — | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.delta_V.mean_abs |
| `\shDvPos` | 0.496 | ΔV̂ > 0 비율 | ΔV̂ > 0 행 | 성공 행 | 정확한 0 없음 (p_zero = 0) | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.delta_V.p_pos |
| `\shClockMean` | +0.00084 | d_clock 평균 | — | — | — | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.d_clock.mean |
| `\shClockAbs` | 0.0132 | d_clock 평균 절댓값 | — | — | — | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.d_clock.mean_abs |
| `\shClockPos` | 0.554 | d_clock > 0 비율 | d_clock > 0 행 | 성공 행 | 정확한 0 없음 (p_zero = 0) | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.d_clock.p_pos |
| `\shUpdMean` | -0.00258 | d_update 평균 | — | — | — | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.d_update.mean |
| `\shUpdAbs` | 0.1150 | d_update 평균 절댓값 | — | — | — | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.d_update.mean_abs |
| `\shUpdPos` | 0.494 | d_update > 0 비율 | d_update > 0 행 | 성공 행 | 정확한 0 없음 (p_zero = 0) | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.d_update.p_pos |
| `\shFlip` | 0.036 | Y_SVI ≠ 1[d_update > 0] 비율 | 불일치 행 | 성공 행 | — | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.Y_vs_d_update_pos_flip |
| `\shOpp` | 0.507 | sign(d_clock) ≠ sign(d_update) 비율 | 부호가 다른 행 | 성공 행 | 정확한 0 없음 | 비가중 | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.opposite_sign_clock_update_rate |
| `\shHoldAge` | 111.8 | hold 상태의 스냅샷 나이 평균 (s) | — | — | — | — | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.mean_hold_snapshot_age_s |
| `\shFollow` | 77.9 | 결과 시점까지의 평균 추적 길이 (s) | — | — | — | — | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`.mean_followup_s |
| `\shSubN` | 2{,}709 | 예비 부표본 행 수 | — | — | — | — | `SUPPLEMENTARY_E2_S_HOLD_20260921.json` `subsample.n_ok` |
| `\ebBoot` | 10{,}000 | 부트스트랩 반복 수 | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `n_boot` |
| `\ebSeed` | 7 | 시드 | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `seed` |
| `\ebTKRn` | 5{,}202 | 행 수 (T, KR_16.13) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.n_rows |
| `\ebTKRm` | 3{,}859 | 경기 수 (T, KR_16.13) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.n_matches |
| `\ebTKRBq` | 0.2483 | Brier q (동결) | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.brier_q |
| `\ebTKRBpt` | 0.2457 | Brier PT_flex (동결) | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.brier_pt |
| `\ebTKRD` | +0.00261 | ΔBrier q − PT_flex | mean_m d_m, d_m = mean_i[(q−y)² − (PT−y)²] | 3{,}859 경기 | 동결 모델, 재학습 없음 | 경기 평균 | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.bootstrap.estimate |
| `\ebTKRLo` | -0.00041 | 95% 구간 하한 | — | — | 경기 군집 백분위, 10,000회 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.bootstrap.ci95[0] |
| `\ebTKRHi` | +0.00572 | 95% 구간 상한 | — | — | 경기 군집 백분위, 10,000회 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.bootstrap.ci95[1] |
| `\ebTKRLoB` | -0.00115 | 98.75% 구간 하한 | — | — | Bonferroni 0.05/4 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.bootstrap.ci9875[0] |
| `\ebTKRHiB` | +0.00668 | 98.75% 구간 상한 | — | — | Bonferroni 0.05/4 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.bootstrap.ci9875[1] |
| `\ebTKRFrac` | 0.955 | 부트스트랩 추출 중 Δ>0 비율 (p값 아님) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|KR_16.13`.bootstrap.frac_boot_gt0 |
| `\ebTNAn` | 5{,}312 | 행 수 (T, NA1_16.13) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.n_rows |
| `\ebTNAm` | 3{,}955 | 경기 수 (T, NA1_16.13) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.n_matches |
| `\ebTNABq` | 0.2534 | Brier q (동결) | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.brier_q |
| `\ebTNABpt` | 0.2494 | Brier PT_flex (동결) | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.brier_pt |
| `\ebTNAD` | +0.00398 | ΔBrier q − PT_flex | mean_m d_m, d_m = mean_i[(q−y)² − (PT−y)²] | 3{,}955 경기 | 동결 모델, 재학습 없음 | 경기 평균 | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.bootstrap.estimate |
| `\ebTNALo` | +0.00086 | 95% 구간 하한 | — | — | 경기 군집 백분위, 10,000회 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.bootstrap.ci95[0] |
| `\ebTNAHi` | +0.00705 | 95% 구간 상한 | — | — | 경기 군집 백분위, 10,000회 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.bootstrap.ci95[1] |
| `\ebTNALoB` | +0.00009 | 98.75% 구간 하한 | — | — | Bonferroni 0.05/4 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.bootstrap.ci9875[0] |
| `\ebTNAHiB` | +0.00784 | 98.75% 구간 상한 | — | — | Bonferroni 0.05/4 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.bootstrap.ci9875[1] |
| `\ebTNAFrac` | 0.995 | 부트스트랩 추출 중 Δ>0 비율 (p값 아님) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.T|NA1_16.13`.bootstrap.frac_boot_gt0 |
| `\ebSKRn` | 15{,}641 | 행 수 (S, KR_16.13) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.n_rows |
| `\ebSKRm` | 7{,}874 | 경기 수 (S, KR_16.13) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.n_matches |
| `\ebSKRBq` | 0.2494 | Brier q (동결) | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.brier_q |
| `\ebSKRBpt` | 0.2512 | Brier PT_flex (동결) | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.brier_pt |
| `\ebSKRD` | -0.00180 | ΔBrier q − PT_flex | mean_m d_m, d_m = mean_i[(q−y)² − (PT−y)²] | 7{,}874 경기 | 동결 모델, 재학습 없음 | 경기 평균 | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.bootstrap.estimate |
| `\ebSKRLo` | -0.00328 | 95% 구간 하한 | — | — | 경기 군집 백분위, 10,000회 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.bootstrap.ci95[0] |
| `\ebSKRHi` | -0.00029 | 95% 구간 상한 | — | — | 경기 군집 백분위, 10,000회 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.bootstrap.ci95[1] |
| `\ebSKRLoB` | -0.00373 | 98.75% 구간 하한 | — | — | Bonferroni 0.05/4 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.bootstrap.ci9875[0] |
| `\ebSKRHiB` | +0.00011 | 98.75% 구간 상한 | — | — | Bonferroni 0.05/4 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.bootstrap.ci9875[1] |
| `\ebSKRFrac` | 0.008 | 부트스트랩 추출 중 Δ>0 비율 (p값 아님) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|KR_16.13`.bootstrap.frac_boot_gt0 |
| `\ebSNAn` | 16{,}100 | 행 수 (S, NA1_16.13) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.n_rows |
| `\ebSNAm` | 7{,}952 | 경기 수 (S, NA1_16.13) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.n_matches |
| `\ebSNABq` | 0.2503 | Brier q (동결) | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.brier_q |
| `\ebSNABpt` | 0.2523 | Brier PT_flex (동결) | — | — | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.brier_pt |
| `\ebSNAD` | -0.00204 | ΔBrier q − PT_flex | mean_m d_m, d_m = mean_i[(q−y)² − (PT−y)²] | 7{,}952 경기 | 동결 모델, 재학습 없음 | 경기 평균 | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.bootstrap.estimate |
| `\ebSNALo` | -0.00363 | 95% 구간 하한 | — | — | 경기 군집 백분위, 10,000회 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.bootstrap.ci95[0] |
| `\ebSNAHi` | -0.00044 | 95% 구간 상한 | — | — | 경기 군집 백분위, 10,000회 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.bootstrap.ci95[1] |
| `\ebSNALoB` | -0.00403 | 98.75% 구간 하한 | — | — | Bonferroni 0.05/4 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.bootstrap.ci9875[0] |
| `\ebSNAHiB` | +0.00001 | 98.75% 구간 상한 | — | — | Bonferroni 0.05/4 | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.bootstrap.ci9875[1] |
| `\ebSNAFrac` | 0.006 | 부트스트랩 추출 중 Δ>0 비율 (p값 아님) | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary.S|NA1_16.13`.bootstrap.frac_boot_gt0 |
| `\ebHKRn` | 3{,}181 | 공통 경기 수 (KR_16.13) | T와 S 행을 모두 가진 경기 | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `common_match_H.KR_16.13`.n_common |
| `\ebHKR` | +0.00364 | H = mean_m(d_mT − d_mS) | 경기별 T 손실차 − S 손실차 | 3{,}181 공통 경기 | 전체 코호트 D_T − D_S와 다른 대상 | 경기 평균 | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `common_match_H.KR_16.13`.H |
| `\ebHKRLo` | -0.00063 | H 95% 구간 하한 | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `common_match_H.KR_16.13`.ci95[0] |
| `\ebHKRHi` | +0.00790 | H 95% 구간 상한 | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `common_match_H.KR_16.13`.ci95[1] |
| `\ebHNAn` | 3{,}290 | 공통 경기 수 (NA1_16.13) | T와 S 행을 모두 가진 경기 | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `common_match_H.NA1_16.13`.n_common |
| `\ebHNA` | +0.00525 | H = mean_m(d_mT − d_mS) | 경기별 T 손실차 − S 손실차 | 3{,}290 공통 경기 | 전체 코호트 D_T − D_S와 다른 대상 | 경기 평균 | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `common_match_H.NA1_16.13`.H |
| `\ebHNALo` | +0.00094 | H 95% 구간 하한 | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `common_match_H.NA1_16.13`.ci95[0] |
| `\ebHNAHi` | +0.00960 | H 95% 구간 상한 | — | — | — | — | `SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `common_match_H.NA1_16.13`.ci95[1] |
| `\hzTAllN` | 32{,}981 | 범위 행 수 (T, all) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.all`.n |
| `\hzTAllFlipSixty` | 0.015 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.all`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzTAllFlipHundredTwenty` | 0.007 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.all`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzTAllSameEp` | 0.594 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.all`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzTAllFlipDiffEp` | 0.037 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.all`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzTAllDbMain` | -0.00373 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.frozen_pred_label_transfer.all.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzTAllDbAlt` | -0.00368 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...all.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzTBfortyN` | 5{,}423 | 범위 행 수 (T, B40) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.B40`.n |
| `\hzTBfortyFlipSixty` | 0.014 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.B40`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzTBfortyFlipHundredTwenty` | 0.006 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.B40`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzTBfortySameEp` | 0.553 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.B40`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzTBfortyFlipDiffEp` | 0.031 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.B40`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzTBfortyDbMain` | -0.00214 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.frozen_pred_label_transfer.B40.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzTBfortyDbAlt` | -0.00219 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...B40.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzTSmallN` | 13{,}151 | 범위 행 수 (T, small_abs_dV) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.small_abs_dV`.n |
| `\hzTSmallFlipSixty` | 0.031 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.small_abs_dV`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzTSmallFlipHundredTwenty` | 0.013 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.small_abs_dV`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzTSmallSameEp` | 0.652 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.small_abs_dV`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzTSmallFlipDiffEp` | 0.090 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.small_abs_dV`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzTSmallDbMain` | -0.00136 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.frozen_pred_label_transfer.small_abs_dV.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzTSmallDbAlt` | -0.00107 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...small_abs_dV.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzTSameFN` | 2{,}224 | 범위 행 수 (T, same_frame) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.same_frame`.n |
| `\hzTSameFFlipSixty` | 0.000 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.same_frame`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzTSameFFlipHundredTwenty` | 0.000 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.same_frame`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzTSameFSameEp` | 1.000 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.same_frame`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzTSameFFlipDiffEp` | --- | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.same_frame`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzTSameFDbMain` | -0.00084 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.frozen_pred_label_transfer.same_frame.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzTSameFDbAlt` | -0.00084 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...same_frame.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzTNewFN` | 30{,}757 | 범위 행 수 (T, new_frame) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.new_frame`.n |
| `\hzTNewFFlipSixty` | 0.016 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.new_frame`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzTNewFFlipHundredTwenty` | 0.007 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.new_frame`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzTNewFSameEp` | 0.565 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.new_frame`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzTNewFFlipDiffEp` | 0.037 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.new_frame`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzTNewFDbMain` | -0.00390 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.frozen_pred_label_transfer.new_frame.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzTNewFDbAlt` | -0.00384 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...new_frame.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzTTtwoTenN` | 2{,}265 | 범위 행 수 (T, t_2_10) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_2_10`.n |
| `\hzTTtwoTenFlipSixty` | 0.015 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_2_10`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzTTtwoTenFlipHundredTwenty` | 0.008 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_2_10`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzTTtwoTenSameEp` | 0.611 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_2_10`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzTTtwoTenFlipDiffEp` | 0.037 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_2_10`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzTTtwoTenDbMain` | -0.00285 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.frozen_pred_label_transfer.t_2_10.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzTTtwoTenDbAlt` | -0.00219 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...t_2_10.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzTTtenTwentyN` | 11{,}858 | 범위 행 수 (T, t_10_20) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_10_20`.n |
| `\hzTTtenTwentyFlipSixty` | 0.019 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_10_20`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzTTtenTwentyFlipHundredTwenty` | 0.008 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_10_20`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzTTtenTwentySameEp` | 0.557 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_10_20`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzTTtenTwentyFlipDiffEp` | 0.044 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_10_20`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzTTtenTwentyDbMain` | -0.00525 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.frozen_pred_label_transfer.t_10_20.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzTTtenTwentyDbAlt` | -0.00545 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...t_10_20.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzTTtwentyThirtyN` | 15{,}446 | 범위 행 수 (T, t_20_30) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_20_30`.n |
| `\hzTTtwentyThirtyFlipSixty` | 0.013 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_20_30`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzTTtwentyThirtyFlipHundredTwenty` | 0.006 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_20_30`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzTTtwentyThirtySameEp` | 0.588 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_20_30`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzTTtwentyThirtyFlipDiffEp` | 0.031 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_20_30`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzTTtwentyThirtyDbMain` | -0.00222 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.frozen_pred_label_transfer.t_20_30.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzTTtwentyThirtyDbAlt` | -0.00215 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...t_20_30.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzTTthirtyInfN` | 3{,}412 | 범위 행 수 (T, t_30_inf) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_30_inf`.n |
| `\hzTTthirtyInfFlipSixty` | 0.011 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_30_inf`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzTTthirtyInfFlipHundredTwenty` | 0.006 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_30_inf`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzTTthirtyInfSameEp` | 0.744 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_30_inf`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzTTthirtyInfFlipDiffEp` | 0.045 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.t_30_inf`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzTTthirtyInfDbMain` | -0.00388 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.frozen_pred_label_transfer.t_30_inf.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzTTthirtyInfDbAlt` | -0.00360 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...t_30_inf.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzTSameEpNinety` | 0.810 | 같은 결과 시점 비율 h90↔h120 (전체) | — | — | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.T.strata.all.endpoint_identity.h90_vs_h120.share_same_endpoint` |
| `\hzSAllN` | 101{,}205 | 범위 행 수 (S, all) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.all`.n |
| `\hzSAllFlipSixty` | 0.018 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.all`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzSAllFlipHundredTwenty` | 0.006 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.all`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzSAllSameEp` | 0.684 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.all`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzSAllFlipDiffEp` | 0.057 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.all`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzSAllDbMain` | -0.00335 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.frozen_pred_label_transfer.all.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzSAllDbAlt` | -0.00326 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...all.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzSBfortyN` | 31{,}675 | 범위 행 수 (S, B40) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.B40`.n |
| `\hzSBfortyFlipSixty` | 0.018 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.B40`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzSBfortyFlipHundredTwenty` | 0.006 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.B40`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzSBfortySameEp` | 0.672 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.B40`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzSBfortyFlipDiffEp` | 0.055 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.B40`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzSBfortyDbMain` | -0.00210 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.frozen_pred_label_transfer.B40.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzSBfortyDbAlt` | -0.00185 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...B40.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzSSmallN` | 40{,}468 | 범위 행 수 (S, small_abs_dV) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.small_abs_dV`.n |
| `\hzSSmallFlipSixty` | 0.037 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.small_abs_dV`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzSSmallFlipHundredTwenty` | 0.013 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.small_abs_dV`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzSSmallSameEp` | 0.698 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.small_abs_dV`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzSSmallFlipDiffEp` | 0.122 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.small_abs_dV`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzSSmallDbMain` | -0.00028 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.frozen_pred_label_transfer.small_abs_dV.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzSSmallDbAlt` | -0.00021 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...small_abs_dV.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzSSameFN` | 14{,}026 | 범위 행 수 (S, same_frame) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.same_frame`.n |
| `\hzSSameFFlipSixty` | 0.000 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.same_frame`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzSSameFFlipHundredTwenty` | 0.000 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.same_frame`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzSSameFSameEp` | 1.000 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.same_frame`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzSSameFFlipDiffEp` | --- | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.same_frame`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzSSameFDbMain` | -0.00120 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.frozen_pred_label_transfer.same_frame.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzSSameFDbAlt` | -0.00120 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...same_frame.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzSNewFN` | 87{,}179 | 범위 행 수 (S, new_frame) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.new_frame`.n |
| `\hzSNewFFlipSixty` | 0.021 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.new_frame`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzSNewFFlipHundredTwenty` | 0.007 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.new_frame`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzSNewFSameEp` | 0.633 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.new_frame`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzSNewFFlipDiffEp` | 0.057 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.new_frame`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzSNewFDbMain` | -0.00351 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.frozen_pred_label_transfer.new_frame.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzSNewFDbAlt` | -0.00339 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...new_frame.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzSTtwoTenN` | 50{,}444 | 범위 행 수 (S, t_2_10) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_2_10`.n |
| `\hzSTtwoTenFlipSixty` | 0.020 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_2_10`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzSTtwoTenFlipHundredTwenty` | 0.007 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_2_10`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzSTtwoTenSameEp` | 0.678 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_2_10`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzSTtwoTenFlipDiffEp` | 0.061 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_2_10`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzSTtwoTenDbMain` | -0.00306 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.frozen_pred_label_transfer.t_2_10.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzSTtwoTenDbAlt` | -0.00288 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...t_2_10.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzSTtenTwentyN` | 33{,}415 | 범위 행 수 (S, t_10_20) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_10_20`.n |
| `\hzSTtenTwentyFlipSixty` | 0.017 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_10_20`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzSTtenTwentyFlipHundredTwenty` | 0.006 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_10_20`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzSTtenTwentySameEp` | 0.696 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_10_20`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzSTtenTwentyFlipDiffEp` | 0.056 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_10_20`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzSTtenTwentyDbMain` | -0.00423 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.frozen_pred_label_transfer.t_10_20.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzSTtenTwentyDbAlt` | -0.00422 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...t_10_20.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzSTtwentyThirtyN` | 14{,}872 | 범위 행 수 (S, t_20_30) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_20_30`.n |
| `\hzSTtwentyThirtyFlipSixty` | 0.016 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_20_30`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzSTtwentyThirtyFlipHundredTwenty` | 0.005 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_20_30`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzSTtwentyThirtySameEp` | 0.667 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_20_30`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzSTtwentyThirtyFlipDiffEp` | 0.047 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_20_30`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzSTtwentyThirtyDbMain` | -0.00106 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.frozen_pred_label_transfer.t_20_30.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzSTtwentyThirtyDbAlt` | -0.00102 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...t_20_30.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzSTthirtyInfN` | 2{,}474 | 범위 행 수 (S, t_30_inf) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_30_inf`.n |
| `\hzSTthirtyInfFlipSixty` | 0.013 | SVI 뒤집힘 h60↔h90 | Y_h60 ≠ Y_h90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_30_inf`.pairwise_svi.h60_vs_h90.svi_flip_rate |
| `\hzSTthirtyInfFlipHundredTwenty` | 0.006 | SVI 뒤집힘 h90↔h120 | Y_h90 ≠ Y_h120 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_30_inf`.pairwise_svi.h90_vs_h120.svi_flip_rate |
| `\hzSTthirtyInfSameEp` | 0.722 | 같은 결과 시점 비율 h60↔h90 | e_60 = e_90 행 | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_30_inf`.endpoint_identity.h60_vs_h90.share_same_endpoint |
| `\hzSTthirtyInfFlipDiffEp` | 0.045 | 뒤집힘 | 다른 결과 시점 (h60↔h90) | 뒤집힘 행 | e_60 ≠ e_90 행 | 같은 프레임 범위는 다른 종점 행이 없어 정의되지 않음(---) | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.t_30_inf`.endpoint_identity.h60_vs_h90.among_diff_endpoint_svi_flip |
| `\hzSTthirtyInfDbMain` | +0.00082 | ΔBrier q−PT on Y_h90 | Brier(q)−Brier(PT) | 범위 행 | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.frozen_pred_label_transfer.t_30_inf.Y_h60.delta_brier_q_minus_pt_on_main` |
| `\hzSTthirtyInfDbAlt` | +0.00084 | ΔBrier q−PT on Y_h60 (라벨 이식) | Brier(q)−Brier(PT), 라벨만 Y_h60 | 범위 행 | 동결 예측 고정 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `...t_30_inf.Y_h60.delta_brier_q_minus_pt_on_alt` |
| `\hzSSameEpNinety` | 0.866 | 같은 결과 시점 비율 h90↔h120 (전체) | — | — | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon.S.strata.all.endpoint_identity.h90_vs_h120.share_same_endpoint` |
| `\peerName` | A\_LR\_expanded\_fit85 | 동료 평가기 식별자 | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.peer |
| `\peerN` | 32{,}981 | 행 수 (T TEST) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.n |
| `\peerSignAgree` | 0.914 | ΔV 부호 일치율 (주 대 동료 평가기) | sign 일치 행 | 32{,}979 비영 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.sign_agree_nonzero |
| `\peerFlipAll` | 0.086 | 라벨 뒤집힘 비율 (전체) | Y_peer ≠ Y_main 행 | 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.label_flip_vs_main_all |
| `\peerAllN` | 32{,}981 | 범위 행 수 (all) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.all.n |
| `\peerAllFlip` | 0.086 | 라벨 뒤집힘 비율 | Y_peer ≠ Y_main | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.all.label_flip_vs_main |
| `\peerAllDbMain` | -0.00373 | ΔBrier q−PT on Y_main | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.all.delta_brier_q_minus_pt_on_main |
| `\peerAllDbAlt` | -0.00125 | ΔBrier q−PT on Y_peer | 예측 고정, 라벨만 동료 평가기 | 범위 행 | q 입력의 p_pre는 교체하지 않음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.all.delta_brier_q_minus_pt_on_alt |
| `\peerBfortyN` | 5{,}423 | 범위 행 수 (B40) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.B40.n |
| `\peerBfortyFlip` | 0.072 | 라벨 뒤집힘 비율 | Y_peer ≠ Y_main | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.B40.label_flip_vs_main |
| `\peerBfortyDbMain` | -0.00214 | ΔBrier q−PT on Y_main | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.B40.delta_brier_q_minus_pt_on_main |
| `\peerBfortyDbAlt` | -0.00067 | ΔBrier q−PT on Y_peer | 예측 고정, 라벨만 동료 평가기 | 범위 행 | q 입력의 p_pre는 교체하지 않음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.B40.delta_brier_q_minus_pt_on_alt |
| `\peerSmallN` | 13{,}151 | 범위 행 수 (small_abs_dV) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.small_abs_dV.n |
| `\peerSmallFlip` | 0.187 | 라벨 뒤집힘 비율 | Y_peer ≠ Y_main | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.small_abs_dV.label_flip_vs_main |
| `\peerSmallDbMain` | -0.00136 | ΔBrier q−PT on Y_main | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.small_abs_dV.delta_brier_q_minus_pt_on_main |
| `\peerSmallDbAlt` | +0.00355 | ΔBrier q−PT on Y_peer | 예측 고정, 라벨만 동료 평가기 | 범위 행 | q 입력의 p_pre는 교체하지 않음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.small_abs_dV.delta_brier_q_minus_pt_on_alt |
| `\peerSameFN` | 2{,}224 | 범위 행 수 (same_frame) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.same_frame.n |
| `\peerSameFFlip` | 0.060 | 라벨 뒤집힘 비율 | Y_peer ≠ Y_main | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.same_frame.label_flip_vs_main |
| `\peerSameFDbMain` | -0.00084 | ΔBrier q−PT on Y_main | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.same_frame.delta_brier_q_minus_pt_on_main |
| `\peerSameFDbAlt` | -0.00107 | ΔBrier q−PT on Y_peer | 예측 고정, 라벨만 동료 평가기 | 범위 행 | q 입력의 p_pre는 교체하지 않음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.same_frame.delta_brier_q_minus_pt_on_alt |
| `\peerNewFN` | 30{,}757 | 범위 행 수 (new_frame) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.new_frame.n |
| `\peerNewFFlip` | 0.088 | 라벨 뒤집힘 비율 | Y_peer ≠ Y_main | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.new_frame.label_flip_vs_main |
| `\peerNewFDbMain` | -0.00390 | ΔBrier q−PT on Y_main | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.new_frame.delta_brier_q_minus_pt_on_main |
| `\peerNewFDbAlt` | -0.00118 | ΔBrier q−PT on Y_peer | 예측 고정, 라벨만 동료 평가기 | 범위 행 | q 입력의 p_pre는 교체하지 않음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.new_frame.delta_brier_q_minus_pt_on_alt |
| `\peerTtwoTenN` | 2{,}265 | 범위 행 수 (t_2_10) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_2_10.n |
| `\peerTtwoTenFlip` | 0.106 | 라벨 뒤집힘 비율 | Y_peer ≠ Y_main | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_2_10.label_flip_vs_main |
| `\peerTtwoTenDbMain` | -0.00285 | ΔBrier q−PT on Y_main | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_2_10.delta_brier_q_minus_pt_on_main |
| `\peerTtwoTenDbAlt` | -0.00287 | ΔBrier q−PT on Y_peer | 예측 고정, 라벨만 동료 평가기 | 범위 행 | q 입력의 p_pre는 교체하지 않음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_2_10.delta_brier_q_minus_pt_on_alt |
| `\peerTtenTwentyN` | 11{,}858 | 범위 행 수 (t_10_20) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_10_20.n |
| `\peerTtenTwentyFlip` | 0.093 | 라벨 뒤집힘 비율 | Y_peer ≠ Y_main | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_10_20.label_flip_vs_main |
| `\peerTtenTwentyDbMain` | -0.00525 | ΔBrier q−PT on Y_main | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_10_20.delta_brier_q_minus_pt_on_main |
| `\peerTtenTwentyDbAlt` | -0.00412 | ΔBrier q−PT on Y_peer | 예측 고정, 라벨만 동료 평가기 | 범위 행 | q 입력의 p_pre는 교체하지 않음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_10_20.delta_brier_q_minus_pt_on_alt |
| `\peerTtwentyThirtyN` | 15{,}446 | 범위 행 수 (t_20_30) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_20_30.n |
| `\peerTtwentyThirtyFlip` | 0.081 | 라벨 뒤집힘 비율 | Y_peer ≠ Y_main | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_20_30.label_flip_vs_main |
| `\peerTtwentyThirtyDbMain` | -0.00222 | ΔBrier q−PT on Y_main | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_20_30.delta_brier_q_minus_pt_on_main |
| `\peerTtwentyThirtyDbAlt` | +0.00129 | ΔBrier q−PT on Y_peer | 예측 고정, 라벨만 동료 평가기 | 범위 행 | q 입력의 p_pre는 교체하지 않음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_20_30.delta_brier_q_minus_pt_on_alt |
| `\peerTthirtyInfN` | 3{,}412 | 범위 행 수 (t_30_inf) | — | — | — | — | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_30_inf.n |
| `\peerTthirtyInfFlip` | 0.078 | 라벨 뒤집힘 비율 | Y_peer ≠ Y_main | 범위 행 | — | 비가중 | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_30_inf.label_flip_vs_main |
| `\peerTthirtyInfDbMain` | -0.00388 | ΔBrier q−PT on Y_main | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_30_inf.delta_brier_q_minus_pt_on_main |
| `\peerTthirtyInfDbAlt` | +0.00186 | ΔBrier q−PT on Y_peer | 예측 고정, 라벨만 동료 평가기 | 범위 행 | q 입력의 p_pre는 교체하지 않음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `peer_label_transfer`.by_scope.t_30_inf.delta_brier_q_minus_pt_on_alt |
| `\efN` | 32{,}981 | e_fixed 재구성 성공 행 (T TEST) | — | — | — | — | `SUPPLEMENTARY_E4_E_FIXED_20260921.json` `n_ok` |
| `\efSameEp` | 6.9 | h90과 같은 결과 시점 비율 (%) | e_fixed = e_90 행 | 행 | — | — | `SUPPLEMENTARY_E4_E_FIXED_20260921.json` `share_same_endpoint_as_h90` |
| `\efFollowFixed` | 57.5 | e_fixed 평균 추적 길이 (s) | — | — | — | — | `SUPPLEMENTARY_E4_E_FIXED_20260921.json` `mean_followup_s_fixed` |
| `\efFollowH` | 77.9 | h90 평균 추적 길이 (s) | — | — | — | — | `SUPPLEMENTARY_E4_E_FIXED_20260921.json` `mean_followup_s_h90` |
| `\efFlip` | 0.096 | SVI 뒤집힘 e_fixed↔h90 | Y_fixed ≠ Y_h90 행 | 행 | — | 비가중 | `SUPPLEMENTARY_E4_E_FIXED_20260921.json` `svi_flip_vs_h90` |
| `\efDbH` | -0.00373 | ΔBrier q−PT on Y_h90 | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_E_FIXED_20260921.json` `frozen_q_pt.delta_brier_q_minus_pt_on_Y_h90` |
| `\efDbFixed` | -0.00397 | ΔBrier q−PT on Y_fixed (라벨 이식) | 예측 고정, 라벨만 e_fixed | 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E4_E_FIXED_20260921.json` `frozen_q_pt.delta_brier_q_minus_pt_on_Y_fixed` |
| `\oatNrun` | 5{,}000 | OAT 실행 경기 수 (hash 선두) | — | — | 설계 예산 20,000; 검정력 보장 아님 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `n_matches_run` |
| `\oatNdesign` | 20{,}000 | OAT 설계 경기 수 | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `n_matches_design` |
| `\oatRefEng` | 18{,}192 | 검출 교전 수 (ref) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.ref.n_engagements` |
| `\oatRefT` | 6{,}801 | T 교전 수 (ref) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.ref.cohort_counts.T` |
| `\oatRefS` | 8{,}865 | S 교전 수 (ref) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.ref.cohort_counts.S` |
| `\oatRefP` | 2{,}526 | pick 교전 수 (ref) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.ref.cohort_counts.P` |
| `\oatGtwelveEng` | 18{,}748 | 검출 교전 수 (G_12200) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.G_12200.n_engagements` |
| `\oatGtwelveT` | 6{,}613 | T 교전 수 (G_12200) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.G_12200.cohort_counts.T` |
| `\oatGtwelveS` | 9{,}363 | S 교전 수 (G_12200) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.G_12200.cohort_counts.S` |
| `\oatGtwelveP` | 2{,}772 | pick 교전 수 (G_12200) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.G_12200.cohort_counts.P` |
| `\oatGtwelveCommon` | 18{,}134 | 기준과 공통 앵커 (G_12200) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.G_12200.n_common_anchors` |
| `\oatGtwelveRefOnly` | 58 | 기준에만 있는 교전 (G_12200) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.G_12200.n_ref_only` |
| `\oatGtwelveAltOnly` | 614 | 대안에만 있는 교전 (G_12200) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.G_12200.n_alt_only` |
| `\oatGfifteenEng` | 17{,}417 | 검출 교전 수 (G_15700) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.G_15700.n_engagements` |
| `\oatGfifteenT` | 6{,}670 | T 교전 수 (G_15700) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.G_15700.cohort_counts.T` |
| `\oatGfifteenS` | 8{,}455 | S 교전 수 (G_15700) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.G_15700.cohort_counts.S` |
| `\oatGfifteenP` | 2{,}292 | pick 교전 수 (G_15700) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.G_15700.cohort_counts.P` |
| `\oatGfifteenCommon` | 17{,}367 | 기준과 공통 앵커 (G_15700) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.G_15700.n_common_anchors` |
| `\oatGfifteenRefOnly` | 825 | 기준에만 있는 교전 (G_15700) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.G_15700.n_ref_only` |
| `\oatGfifteenAltOnly` | 50 | 대안에만 있는 교전 (G_15700) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.G_15700.n_alt_only` |
| `\oatDfourEng` | 18{,}247 | 검출 교전 수 (D_4000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.D_4000.n_engagements` |
| `\oatDfourT` | 6{,}746 | T 교전 수 (D_4000) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.D_4000.cohort_counts.T` |
| `\oatDfourS` | 8{,}955 | S 교전 수 (D_4000) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.D_4000.cohort_counts.S` |
| `\oatDfourP` | 2{,}546 | pick 교전 수 (D_4000) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.D_4000.cohort_counts.P` |
| `\oatDfourCommon` | 18{,}158 | 기준과 공통 앵커 (D_4000) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.D_4000.n_common_anchors` |
| `\oatDfourRefOnly` | 34 | 기준에만 있는 교전 (D_4000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.D_4000.n_ref_only` |
| `\oatDfourAltOnly` | 89 | 대안에만 있는 교전 (D_4000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.D_4000.n_alt_only` |
| `\oatDfourfiveEng` | 18{,}148 | 검출 교전 수 (D_4500) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.D_4500.n_engagements` |
| `\oatDfourfiveT` | 6{,}806 | T 교전 수 (D_4500) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.D_4500.cohort_counts.T` |
| `\oatDfourfiveS` | 8{,}824 | S 교전 수 (D_4500) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.D_4500.cohort_counts.S` |
| `\oatDfourfiveP` | 2{,}518 | pick 교전 수 (D_4500) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.D_4500.cohort_counts.P` |
| `\oatDfourfiveCommon` | 18{,}133 | 기준과 공통 앵커 (D_4500) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.D_4500.n_common_anchors` |
| `\oatDfourfiveRefOnly` | 59 | 기준에만 있는 교전 (D_4500) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.D_4500.n_ref_only` |
| `\oatDfourfiveAltOnly` | 15 | 대안에만 있는 교전 (D_4500) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.D_4500.n_alt_only` |
| `\oatRfourteenEng` | 12{,}750 | 검출 교전 수 (R_1400) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.R_1400.n_engagements` |
| `\oatRfourteenT` | 4{,}523 | T 교전 수 (R_1400) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.R_1400.cohort_counts.T` |
| `\oatRfourteenS` | 6{,}375 | S 교전 수 (R_1400) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.R_1400.cohort_counts.S` |
| `\oatRfourteenP` | 1{,}852 | pick 교전 수 (R_1400) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.R_1400.cohort_counts.P` |
| `\oatRfourteenCommon` | 12{,}518 | 기준과 공통 앵커 (R_1400) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.R_1400.n_common_anchors` |
| `\oatRfourteenRefOnly` | 5{,}674 | 기준에만 있는 교전 (R_1400) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.R_1400.n_ref_only` |
| `\oatRfourteenAltOnly` | 232 | 대안에만 있는 교전 (R_1400) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.R_1400.n_alt_only` |
| `\oatReighteenEng` | 22{,}912 | 검출 교전 수 (R_1800) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.R_1800.n_engagements` |
| `\oatReighteenT` | 7{,}932 | T 교전 수 (R_1800) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.R_1800.cohort_counts.T` |
| `\oatReighteenS` | 11{,}600 | S 교전 수 (R_1800) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.R_1800.cohort_counts.S` |
| `\oatReighteenP` | 3{,}380 | pick 교전 수 (R_1800) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.R_1800.cohort_counts.P` |
| `\oatReighteenCommon` | 17{,}864 | 기준과 공통 앵커 (R_1800) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.R_1800.n_common_anchors` |
| `\oatReighteenRefOnly` | 328 | 기준에만 있는 교전 (R_1800) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.R_1800.n_ref_only` |
| `\oatReighteenAltOnly` | 5{,}048 | 대안에만 있는 교전 (R_1800) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.R_1800.n_alt_only` |
| `\oatBtenEng` | 24{,}166 | 검출 교전 수 (B_10000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.B_10000.n_engagements` |
| `\oatBtenT` | 7{,}753 | T 교전 수 (B_10000) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.B_10000.cohort_counts.T` |
| `\oatBtenS` | 11{,}933 | S 교전 수 (B_10000) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.B_10000.cohort_counts.S` |
| `\oatBtenP` | 4{,}480 | pick 교전 수 (B_10000) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.B_10000.cohort_counts.P` |
| `\oatBtenCommon` | 16{,}212 | 기준과 공통 앵커 (B_10000) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.B_10000.n_common_anchors` |
| `\oatBtenRefOnly` | 1{,}980 | 기준에만 있는 교전 (B_10000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.B_10000.n_ref_only` |
| `\oatBtenAltOnly` | 7{,}954 | 대안에만 있는 교전 (B_10000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.B_10000.n_alt_only` |
| `\oatBtwentyEng` | 13{,}088 | 검출 교전 수 (B_20000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.B_20000.n_engagements` |
| `\oatBtwentyT` | 4{,}468 | T 교전 수 (B_20000) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.B_20000.cohort_counts.T` |
| `\oatBtwentyS` | 7{,}018 | S 교전 수 (B_20000) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.B_20000.cohort_counts.S` |
| `\oatBtwentyP` | 1{,}602 | pick 교전 수 (B_20000) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.B_20000.cohort_counts.P` |
| `\oatBtwentyCommon` | 11{,}608 | 기준과 공통 앵커 (B_20000) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.B_20000.n_common_anchors` |
| `\oatBtwentyRefOnly` | 6{,}584 | 기준에만 있는 교전 (B_20000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.B_20000.n_ref_only` |
| `\oatBtwentyAltOnly` | 1{,}480 | 대안에만 있는 교전 (B_20000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.B_20000.n_alt_only` |
| `\oatItwoEng` | 18{,}195 | 검출 교전 수 (I_2000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_2000.n_engagements` |
| `\oatItwoT` | 6{,}251 | T 교전 수 (I_2000) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_2000.cohort_counts.T` |
| `\oatItwoS` | 9{,}107 | S 교전 수 (I_2000) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_2000.cohort_counts.S` |
| `\oatItwoP` | 2{,}837 | pick 교전 수 (I_2000) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_2000.cohort_counts.P` |
| `\oatItwoCommon` | 18{,}163 | 기준과 공통 앵커 (I_2000) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_2000.n_common_anchors` |
| `\oatItwoRefOnly` | 29 | 기준에만 있는 교전 (I_2000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_2000.n_ref_only` |
| `\oatItwoAltOnly` | 32 | 대안에만 있는 교전 (I_2000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_2000.n_alt_only` |
| `\oatItwofiveEng` | 18{,}196 | 검출 교전 수 (I_2500) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_2500.n_engagements` |
| `\oatItwofiveT` | 6{,}544 | T 교전 수 (I_2500) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_2500.cohort_counts.T` |
| `\oatItwofiveS` | 8{,}994 | S 교전 수 (I_2500) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_2500.cohort_counts.S` |
| `\oatItwofiveP` | 2{,}658 | pick 교전 수 (I_2500) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_2500.cohort_counts.P` |
| `\oatItwofiveCommon` | 18{,}184 | 기준과 공통 앵커 (I_2500) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_2500.n_common_anchors` |
| `\oatItwofiveRefOnly` | 8 | 기준에만 있는 교전 (I_2500) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_2500.n_ref_only` |
| `\oatItwofiveAltOnly` | 12 | 대안에만 있는 교전 (I_2500) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_2500.n_alt_only` |
| `\oatIthreefiveEng` | 18{,}192 | 검출 교전 수 (I_3500) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_3500.n_engagements` |
| `\oatIthreefiveT` | 6{,}901 | T 교전 수 (I_3500) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_3500.cohort_counts.T` |
| `\oatIthreefiveS` | 8{,}872 | S 교전 수 (I_3500) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_3500.cohort_counts.S` |
| `\oatIthreefiveP` | 2{,}419 | pick 교전 수 (I_3500) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_3500.cohort_counts.P` |
| `\oatIthreefiveCommon` | 18{,}181 | 기준과 공통 앵커 (I_3500) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_3500.n_common_anchors` |
| `\oatIthreefiveRefOnly` | 11 | 기준에만 있는 교전 (I_3500) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_3500.n_ref_only` |
| `\oatIthreefiveAltOnly` | 11 | 대안에만 있는 교전 (I_3500) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_3500.n_alt_only` |
| `\oatIfourdEng` | 18{,}192 | 검출 교전 수 (I_4264) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_4264.n_engagements` |
| `\oatIfourdT` | 7{,}050 | T 교전 수 (I_4264) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_4264.cohort_counts.T` |
| `\oatIfourdS` | 8{,}847 | S 교전 수 (I_4264) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_4264.cohort_counts.S` |
| `\oatIfourdP` | 2{,}295 | pick 교전 수 (I_4264) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.I_4264.cohort_counts.P` |
| `\oatIfourdCommon` | 18{,}174 | 기준과 공통 앵커 (I_4264) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_4264.n_common_anchors` |
| `\oatIfourdRefOnly` | 18 | 기준에만 있는 교전 (I_4264) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_4264.n_ref_only` |
| `\oatIfourdAltOnly` | 18 | 대안에만 있는 교전 (I_4264) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.I_4264.n_alt_only` |
| `\oatShopexEng` | 18{,}195 | 검출 교전 수 (SHOPEX_on) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.SHOPEX_on.n_engagements` |
| `\oatShopexT` | 6{,}452 | T 교전 수 (SHOPEX_on) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.SHOPEX_on.cohort_counts.T` |
| `\oatShopexS` | 8{,}899 | S 교전 수 (SHOPEX_on) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.SHOPEX_on.cohort_counts.S` |
| `\oatShopexP` | 2{,}844 | pick 교전 수 (SHOPEX_on) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.SHOPEX_on.cohort_counts.P` |
| `\oatShopexCommon` | 18{,}169 | 기준과 공통 앵커 (SHOPEX_on) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.SHOPEX_on.n_common_anchors` |
| `\oatShopexRefOnly` | 23 | 기준에만 있는 교전 (SHOPEX_on) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.SHOPEX_on.n_ref_only` |
| `\oatShopexAltOnly` | 26 | 대안에만 있는 교전 (SHOPEX_on) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.SHOPEX_on.n_alt_only` |
| `\oatMRoneEng` | 18{,}198 | 검출 교전 수 (MR_1000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MR_1000.n_engagements` |
| `\oatMRoneT` | 6{,}838 | T 교전 수 (MR_1000) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MR_1000.cohort_counts.T` |
| `\oatMRoneS` | 8{,}884 | S 교전 수 (MR_1000) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MR_1000.cohort_counts.S` |
| `\oatMRoneP` | 2{,}476 | pick 교전 수 (MR_1000) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MR_1000.cohort_counts.P` |
| `\oatMRoneCommon` | 18{,}071 | 기준과 공통 앵커 (MR_1000) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MR_1000.n_common_anchors` |
| `\oatMRoneRefOnly` | 121 | 기준에만 있는 교전 (MR_1000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MR_1000.n_ref_only` |
| `\oatMRoneAltOnly` | 127 | 대안에만 있는 교전 (MR_1000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MR_1000.n_alt_only` |
| `\oatMRthreeEng` | 18{,}189 | 검출 교전 수 (MR_3000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MR_3000.n_engagements` |
| `\oatMRthreeT` | 6{,}777 | T 교전 수 (MR_3000) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MR_3000.cohort_counts.T` |
| `\oatMRthreeS` | 8{,}871 | S 교전 수 (MR_3000) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MR_3000.cohort_counts.S` |
| `\oatMRthreeP` | 2{,}541 | pick 교전 수 (MR_3000) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MR_3000.cohort_counts.P` |
| `\oatMRthreeCommon` | 18{,}149 | 기준과 공통 앵커 (MR_3000) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MR_3000.n_common_anchors` |
| `\oatMRthreeRefOnly` | 43 | 기준에만 있는 교전 (MR_3000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MR_3000.n_ref_only` |
| `\oatMRthreeAltOnly` | 40 | 대안에만 있는 교전 (MR_3000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MR_3000.n_alt_only` |
| `\oatMDfortyfiveEng` | 17{,}975 | 검출 교전 수 (MD_45000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MD_45000.n_engagements` |
| `\oatMDfortyfiveT` | 6{,}649 | T 교전 수 (MD_45000) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MD_45000.cohort_counts.T` |
| `\oatMDfortyfiveS` | 8{,}898 | S 교전 수 (MD_45000) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MD_45000.cohort_counts.S` |
| `\oatMDfortyfiveP` | 2{,}428 | pick 교전 수 (MD_45000) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MD_45000.cohort_counts.P` |
| `\oatMDfortyfiveCommon` | 17{,}700 | 기준과 공통 앵커 (MD_45000) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MD_45000.n_common_anchors` |
| `\oatMDfortyfiveRefOnly` | 492 | 기준에만 있는 교전 (MD_45000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MD_45000.n_ref_only` |
| `\oatMDfortyfiveAltOnly` | 275 | 대안에만 있는 교전 (MD_45000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MD_45000.n_alt_only` |
| `\oatMDninetyEng` | 18{,}027 | 검출 교전 수 (MD_90000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MD_90000.n_engagements` |
| `\oatMDninetyT` | 6{,}694 | T 교전 수 (MD_90000) | — | — | n_min ≥ 4 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MD_90000.cohort_counts.T` |
| `\oatMDninetyS` | 8{,}803 | S 교전 수 (MD_90000) | — | — | 2 ≤ n_min ≤ 3 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MD_90000.cohort_counts.S` |
| `\oatMDninetyP` | 2{,}530 | pick 교전 수 (MD_90000) | — | — | n_min ≤ 1 | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `settings.MD_90000.cohort_counts.P` |
| `\oatMDninetyCommon` | 17{,}903 | 기준과 공통 앵커 (MD_90000) | 가장 이른 킬 1:1 일치 | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MD_90000.n_common_anchors` |
| `\oatMDninetyRefOnly` | 289 | 기준에만 있는 교전 (MD_90000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MD_90000.n_ref_only` |
| `\oatMDninetyAltOnly` | 124 | 대안에만 있는 교전 (MD_90000) | — | — | — | — | `SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json` `vs_ref.MD_90000.n_alt_only` |
| `\maNT` | 32{,}981 | 행 수 (T) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.n` |
| `\maTKillAllNdec` | 28{,}711 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.kill_diff.all_T`.n_decided |
| `\maTKillAllTie` | 0.129 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.kill_diff.all_T`.tie_share |
| `\maTKillAllAgree` | 0.904 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.kill_diff.all_T`.agreement_rate_match_weighted_decided |
| `\maTKillAllDis` | 0.097 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.kill_diff.all_T`.disagreement_rate_unweighted_decided |
| `\maTKillBfortyNdec` | 4{,}653 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.kill_diff.B40`.n_decided |
| `\maTKillBfortyTie` | 0.142 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.kill_diff.B40`.tie_share |
| `\maTKillBfortyAgree` | 0.928 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.kill_diff.B40`.agreement_rate_match_weighted_decided |
| `\maTKillBfortyDis` | 0.074 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.kill_diff.B40`.disagreement_rate_unweighted_decided |
| `\maTEpicAllNdec` | 12{,}507 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.epic_net.all_T`.n_decided |
| `\maTEpicAllTie` | 0.621 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.epic_net.all_T`.tie_share |
| `\maTEpicAllAgree` | 0.818 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.epic_net.all_T`.agreement_rate_match_weighted_decided |
| `\maTEpicAllDis` | 0.182 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.epic_net.all_T`.disagreement_rate_unweighted_decided |
| `\maTEpicBfortyNdec` | 2{,}221 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.epic_net.B40`.n_decided |
| `\maTEpicBfortyTie` | 0.590 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.epic_net.B40`.tie_share |
| `\maTEpicBfortyAgree` | 0.842 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.epic_net.B40`.agreement_rate_match_weighted_decided |
| `\maTEpicBfortyDis` | 0.158 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.epic_net.B40`.disagreement_rate_unweighted_decided |
| `\maTStructAllNdec` | 16{,}106 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.structure_net.all_T`.n_decided |
| `\maTStructAllTie` | 0.512 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.structure_net.all_T`.tie_share |
| `\maTStructAllAgree` | 0.774 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.structure_net.all_T`.agreement_rate_match_weighted_decided |
| `\maTStructAllDis` | 0.228 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.structure_net.all_T`.disagreement_rate_unweighted_decided |
| `\maTStructBfortyNdec` | 2{,}014 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.structure_net.B40`.n_decided |
| `\maTStructBfortyTie` | 0.629 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.structure_net.B40`.tie_share |
| `\maTStructBfortyAgree` | 0.812 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.structure_net.B40`.agreement_rate_match_weighted_decided |
| `\maTStructBfortyDis` | 0.190 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.structure_net.B40`.disagreement_rate_unweighted_decided |
| `\maTObjAllNdec` | 21{,}779 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.objective_net.all_T`.n_decided |
| `\maTObjAllTie` | 0.340 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.objective_net.all_T`.tie_share |
| `\maTObjAllAgree` | 0.810 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.objective_net.all_T`.agreement_rate_match_weighted_decided |
| `\maTObjAllDis` | 0.191 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.objective_net.all_T`.disagreement_rate_unweighted_decided |
| `\maTObjBfortyNdec` | 3{,}099 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.objective_net.B40`.n_decided |
| `\maTObjBfortyTie` | 0.429 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.objective_net.B40`.tie_share |
| `\maTObjBfortyAgree` | 0.860 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.objective_net.B40`.agreement_rate_match_weighted_decided |
| `\maTObjBfortyDis` | 0.141 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.objective_net.B40`.disagreement_rate_unweighted_decided |
| `\maTAliveAllNdec` | 14{,}584 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.alive_diff_post.all_T`.n_decided |
| `\maTAliveAllTie` | 0.558 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.alive_diff_post.all_T`.tie_share |
| `\maTAliveAllAgree` | 0.794 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.alive_diff_post.all_T`.agreement_rate_match_weighted_decided |
| `\maTAliveAllDis` | 0.208 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.alive_diff_post.all_T`.disagreement_rate_unweighted_decided |
| `\maTAliveBfortyNdec` | 2{,}142 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.alive_diff_post.B40`.n_decided |
| `\maTAliveBfortyTie` | 0.605 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.alive_diff_post.B40`.tie_share |
| `\maTAliveBfortyAgree` | 0.788 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.alive_diff_post.B40`.agreement_rate_match_weighted_decided |
| `\maTAliveBfortyDis` | 0.212 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.material_axes.alive_diff_post.B40`.disagreement_rate_unweighted_decided |
| `\kdTN` | 2{,}780 | 킬 부호 ≠ SVI 행 수 | — | — | 킬 결정 행 안 | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.kill_disagree_obj`.n_kill_svi_disagree |
| `\kdTSvi` | 691 | 그중 오브젝트 순차가 SVI와 일치 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.kill_disagree_obj`.among_disagree_obj_agrees_SVI |
| `\kdTKill` | 1{,}079 | 그중 오브젝트 순차가 킬과 일치 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.kill_disagree_obj`.among_disagree_obj_agrees_kill |
| `\kdTTie` | 1{,}010 | 그중 오브젝트 순차 동률 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.kill_disagree_obj`.among_disagree_obj_tie |
| `\maNS` | 101{,}205 | 행 수 (S) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.n` |
| `\maSKillAllNdec` | 83{,}667 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.kill_diff.all_T`.n_decided |
| `\maSKillAllTie` | 0.173 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.kill_diff.all_T`.tie_share |
| `\maSKillAllAgree` | 0.894 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.kill_diff.all_T`.agreement_rate_match_weighted_decided |
| `\maSKillAllDis` | 0.108 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.kill_diff.all_T`.disagreement_rate_unweighted_decided |
| `\maSKillBfortyNdec` | 25{,}574 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.kill_diff.B40`.n_decided |
| `\maSKillBfortyTie` | 0.193 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.kill_diff.B40`.tie_share |
| `\maSKillBfortyAgree` | 0.923 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.kill_diff.B40`.agreement_rate_match_weighted_decided |
| `\maSKillBfortyDis` | 0.080 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.kill_diff.B40`.disagreement_rate_unweighted_decided |
| `\maSEpicAllNdec` | 23{,}599 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.epic_net.all_T`.n_decided |
| `\maSEpicAllTie` | 0.767 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.epic_net.all_T`.tie_share |
| `\maSEpicAllAgree` | 0.715 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.epic_net.all_T`.agreement_rate_match_weighted_decided |
| `\maSEpicAllDis` | 0.284 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.epic_net.all_T`.disagreement_rate_unweighted_decided |
| `\maSEpicBfortyNdec` | 4{,}761 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.epic_net.B40`.n_decided |
| `\maSEpicBfortyTie` | 0.850 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.epic_net.B40`.tie_share |
| `\maSEpicBfortyAgree` | 0.714 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.epic_net.B40`.agreement_rate_match_weighted_decided |
| `\maSEpicBfortyDis` | 0.285 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.epic_net.B40`.disagreement_rate_unweighted_decided |
| `\maSStructAllNdec` | 20{,}502 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.structure_net.all_T`.n_decided |
| `\maSStructAllTie` | 0.797 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.structure_net.all_T`.tie_share |
| `\maSStructAllAgree` | 0.684 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.structure_net.all_T`.agreement_rate_match_weighted_decided |
| `\maSStructAllDis` | 0.317 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.structure_net.all_T`.disagreement_rate_unweighted_decided |
| `\maSStructBfortyNdec` | 2{,}508 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.structure_net.B40`.n_decided |
| `\maSStructBfortyTie` | 0.921 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.structure_net.B40`.tie_share |
| `\maSStructBfortyAgree` | 0.719 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.structure_net.B40`.agreement_rate_match_weighted_decided |
| `\maSStructBfortyDis` | 0.281 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.structure_net.B40`.disagreement_rate_unweighted_decided |
| `\maSObjAllNdec` | 35{,}907 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.objective_net.all_T`.n_decided |
| `\maSObjAllTie` | 0.645 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.objective_net.all_T`.tie_share |
| `\maSObjAllAgree` | 0.709 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.objective_net.all_T`.agreement_rate_match_weighted_decided |
| `\maSObjAllDis` | 0.292 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.objective_net.all_T`.disagreement_rate_unweighted_decided |
| `\maSObjBfortyNdec` | 6{,}060 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.objective_net.B40`.n_decided |
| `\maSObjBfortyTie` | 0.809 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.objective_net.B40`.tie_share |
| `\maSObjBfortyAgree` | 0.726 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.objective_net.B40`.agreement_rate_match_weighted_decided |
| `\maSObjBfortyDis` | 0.273 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.objective_net.B40`.disagreement_rate_unweighted_decided |
| `\maSAliveAllNdec` | 34{,}992 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.alive_diff_post.all_T`.n_decided |
| `\maSAliveAllTie` | 0.654 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.alive_diff_post.all_T`.tie_share |
| `\maSAliveAllAgree` | 0.759 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.alive_diff_post.all_T`.agreement_rate_match_weighted_decided |
| `\maSAliveAllDis` | 0.243 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.alive_diff_post.all_T`.disagreement_rate_unweighted_decided |
| `\maSAliveBfortyNdec` | 8{,}345 | 결정 행 수 (축 ≠ 0) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.alive_diff_post.B40`.n_decided |
| `\maSAliveBfortyTie` | 0.737 | 동률 비율 | 축 = 0 행 | 범위 행 | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.alive_diff_post.B40`.tie_share |
| `\maSAliveBfortyAgree` | 0.782 | SVI 부호 = 축 부호 일치율 | 일치 행 | 결정 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.alive_diff_post.B40`.agreement_rate_match_weighted_decided |
| `\maSAliveBfortyDis` | 0.219 | 불일치율 (비가중) | 불일치 행 | 결정 행 | — | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.material_axes.alive_diff_post.B40`.disagreement_rate_unweighted_decided |
| `\kdSN` | 9{,}027 | 킬 부호 ≠ SVI 행 수 | — | — | 킬 결정 행 안 | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.kill_disagree_obj`.n_kill_svi_disagree |
| `\kdSSvi` | 2{,}200 | 그중 오브젝트 순차가 SVI와 일치 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.kill_disagree_obj`.among_disagree_obj_agrees_SVI |
| `\kdSKill` | 1{,}912 | 그중 오브젝트 순차가 킬과 일치 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.kill_disagree_obj`.among_disagree_obj_agrees_kill |
| `\kdSTie` | 4{,}915 | 그중 오브젝트 순차 동률 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.S.kill_disagree_obj`.among_disagree_obj_tie |
| `\noAllN` | 32{,}981 | 범위 행 수 (all_T) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.all_T.n |
| `\noAllNBR` | 20{,}211 | 180 s 안 첫 엘리트 오브젝트가 Blue 또는 Red인 행 | — | — | none/경기 종료/동률 제외 | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.all_T.n_decided_BlueRed |
| `\noAllSviAgree` | 0.604 | P(Y_SVI = 1[O = Blue] | O ∈ {Blue, Red}) | SVI 방향과 획득 팀이 일치한 행 | Blue|Red 행 | 선택된 분모 | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.all_T.SVI_agree_rate |
| `\noAllNKill` | 17{,}053 | Blue|Red 행 중 킬 차이 ≠ 0인 행 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.all_T.n_kill_decided_in_selected |
| `\noAllKillAgree` | 0.670 | P(sign(kill) = 1[O = Blue] | Blue|Red, kill ≠ 0) | 킬 부호와 획득 팀 일치 행 | Blue|Red ∩ 킬 결정 행 | SVI 일치율과 분모가 다름 | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.all_T.kill_agree_rate_among_kill_decided |
| `\noAllBlueShare` | 0.499 | Blue|Red 행 중 SVI = 1 비율 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.all_T.SVI_Blue_rate_among_selected |
| `\noBfortyN` | 5{,}423 | 범위 행 수 (B40) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.B40.n |
| `\noBfortyNBR` | 3{,}763 | 180 s 안 첫 엘리트 오브젝트가 Blue 또는 Red인 행 | — | — | none/경기 종료/동률 제외 | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.B40.n_decided_BlueRed |
| `\noBfortySviAgree` | 0.610 | P(Y_SVI = 1[O = Blue] | O ∈ {Blue, Red}) | SVI 방향과 획득 팀이 일치한 행 | Blue|Red 행 | 선택된 분모 | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.B40.SVI_agree_rate |
| `\noBfortyNKill` | 3{,}158 | Blue|Red 행 중 킬 차이 ≠ 0인 행 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.B40.n_kill_decided_in_selected |
| `\noBfortyKillAgree` | 0.638 | P(sign(kill) = 1[O = Blue] | Blue|Red, kill ≠ 0) | 킬 부호와 획득 팀 일치 행 | Blue|Red ∩ 킬 결정 행 | SVI 일치율과 분모가 다름 | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.B40.kill_agree_rate_among_kill_decided |
| `\noBfortyBlueShare` | 0.503 | Blue|Red 행 중 SVI = 1 비율 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.B40.SVI_Blue_rate_among_selected |
| `\noPosN` | 16{,}358 | 범위 행 수 (SVI_pos) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_pos.n |
| `\noPosNBR` | 9{,}934 | 180 s 안 첫 엘리트 오브젝트가 Blue 또는 Red인 행 | — | — | none/경기 종료/동률 제외 | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_pos.n_decided_BlueRed |
| `\noPosSviAgree` | 0.604 | P(Y_SVI = 1[O = Blue] | O ∈ {Blue, Red}) | SVI 방향과 획득 팀이 일치한 행 | Blue|Red 행 | 선택된 분모 | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_pos.SVI_agree_rate |
| `\noPosNKill` | 8{,}378 | Blue|Red 행 중 킬 차이 ≠ 0인 행 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_pos.n_kill_decided_in_selected |
| `\noPosKillAgree` | 0.664 | P(sign(kill) = 1[O = Blue] | Blue|Red, kill ≠ 0) | 킬 부호와 획득 팀 일치 행 | Blue|Red ∩ 킬 결정 행 | SVI 일치율과 분모가 다름 | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_pos.kill_agree_rate_among_kill_decided |
| `\noPosBlueShare` | 0.604 | Blue|Red 행 중 SVI = 1 비율 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_pos.SVI_Blue_rate_among_selected |
| `\noNegN` | 16{,}623 | 범위 행 수 (SVI_neg) | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_neg.n |
| `\noNegNBR` | 10{,}277 | 180 s 안 첫 엘리트 오브젝트가 Blue 또는 Red인 행 | — | — | none/경기 종료/동률 제외 | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_neg.n_decided_BlueRed |
| `\noNegSviAgree` | 0.604 | P(Y_SVI = 1[O = Blue] | O ∈ {Blue, Red}) | SVI 방향과 획득 팀이 일치한 행 | Blue|Red 행 | 선택된 분모 | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_neg.SVI_agree_rate |
| `\noNegNKill` | 8{,}675 | Blue|Red 행 중 킬 차이 ≠ 0인 행 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_neg.n_kill_decided_in_selected |
| `\noNegKillAgree` | 0.676 | P(sign(kill) = 1[O = Blue] | Blue|Red, kill ≠ 0) | 킬 부호와 획득 팀 일치 행 | Blue|Red ∩ 킬 결정 행 | SVI 일치율과 분모가 다름 | 비가중 | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_neg.kill_agree_rate_among_kill_decided |
| `\noNegBlueShare` | 0.396 | Blue|Red 행 중 SVI = 1 비율 | — | — | — | — | `SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.T.nextobj`.SVI_neg.SVI_Blue_rate_among_selected |
| `\meN` | 32{,}981 | market_event 계산 성공 행 (T TEST) | — | — | — | — | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `market_event.n_ok` |
| `\meFlip` | 0.134 | 교환가치 라벨 ≠ SVI 비율 | 부호가 다른 행 | 행 | 같은 (s, e_90+1 ms) 창; v3.3 market_event | 비가중 | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `market_event.label_flip_vs_SVI` |
| `\meAgree` | 0.866 | 교환가치 라벨 = SVI 비율 | — | — | — | 비가중 | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `market_event.agree_rate` |
| `\meDbSvi` | -0.00373 | ΔBrier q−PT on SVI | — | — | 동결 예측 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `market_event.delta_brier_q_minus_pt_on_SVI` |
| `\meDbMarket` | +0.00149 | ΔBrier q−PT on 교환가치 라벨 (라벨 이식) | 예측 고정, 라벨만 market_event | 행 | — | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `market_event.delta_brier_q_minus_pt_on_market` |
| `\rTrainN` | 39{,}605 | R 모형 TRAIN 행 (T, OOF ΔV) | — | — | — | — | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.train_n` |
| `\rQcalN` | 10{,}390 | R 모형 Q_CAL 행 | — | — | — | — | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.qcal_n` |
| `\rQselN` | 10{,}195 | R 모형 Q_SELECT 행 | — | — | — | — | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.qsel_n` |
| `\rTestN` | 32{,}981 | R 모형 TEST 행 | — | — | — | — | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.test_n` |
| `\rZeroBrier` | 0.6326 | R0 TEST 다항 Brier (클래스 합) | Σ_k (p_k − 1[U=k])² | TEST 행 | U = 180 s 안 첫 엘리트 오브젝트 결과 (6 클래스) | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.R0.brier_TEST` |
| `\rOneBrier` | 0.6254 | R1 TEST 다항 Brier | — | — | R1 = R0 + ΔV̂ | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.R1.brier_TEST` |
| `\rDelta` | -0.00724 | R1 − R0 다항 Brier 차 | — | — | 구간 없음 | 경기 가중 (1/n_m) | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.delta_brier_R1_minus_R0_TEST` |
| `\rZeroBR` | 0.4733 | R0 Brier, Blue|Red 선택 분모 | — | — | 선택된 분모 | — | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.R0.brier_TEST_BlueRed_selected` |
| `\rOneBR` | 0.4671 | R1 Brier, Blue|Red 선택 분모 | — | — | 선택된 분모 | — | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.R1.brier_TEST_BlueRed_selected` |
| `\rC` | 10 | R0/R1 L2 규제 C (Q_SELECT 선정) | — | — | — | — | `SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `r0_r1.R0.C` |
| `\gzOverall` | PARTIAL\_PHASE\_B | G0 전체 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `overall_status` |
| `\gzStatusOne` | PARTIAL | 검사 G0.1_provenance_record 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.G0.1_provenance_record.status` |
| `\gzStatusTwo` | PASS | 검사 G0.2_evaluator_hash_after_finalize 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.G0.2_evaluator_hash_after_finalize.status` |
| `\gzStatusThree` | PASS | 검사 G0.3_s_reuse_and_write_guard 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.G0.3_s_reuse_and_write_guard.status` |
| `\gzStatusFour` | PASS | 검사 G0.4_join_key_integrity 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.G0.4_join_key_integrity.status` |
| `\gzStatusFive` | INCOMPLETE | 검사 G0.5_fold_holdout_match_set 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.G0.5_fold_holdout_match_set.status` |
| `\gzStatusSix` | PARTIAL | 검사 G0.6_bundle_reload_parity 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.G0.6_bundle_reload_parity.status` |
| `\gzStatusSeven` | PASS | 검사 G0.7_headline_reaggregate 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.G0.7_headline_reaggregate.status` |
| `\gzStatusEight` | PASS | 검사 G0.8_missing_data_policy 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.G0.8_missing_data_policy.status` |
| `\gzStatusFuture` | INCOMPLETE | 검사 G0.F_future_info_invariance 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.G0.F_future_info_invariance.status` |
| `\gzStatusUnit` | PASS | 검사 unit_tests_provenance 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `checks.unit_tests_provenance.status` |
| `\gzDigestN` | 12 | digest MATCH 아티팩트 수 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `phase_b.digests.n_match` |
| `\gzDigestDiff` | 0 | digest 불일치 수 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `phase_b.digests.n_diff` |
| `\gzReaggT` | -0.00373234 | 동결 예측표 재집계 ΔBrier (T) | Brier(q)−Brier(PT_flex) | 32{,}981 행 / 24{,}020 경기 | — | 경기 가중 (1/n_m) | `G0_INTEGRITY_STATUS_20260921.json` `phase_b.headlines.cohorts.T.delta_brier_q_minus_PT_flex` |
| `\gzPrintT` | -0.00373 | 문서 인쇄값 (T) | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `...docs_delta_printed` |
| `\gzReaggS` | -0.00335262 | 동결 예측표 재집계 ΔBrier (S_identity) | Brier(q)−Brier(PT_flex) | 101{,}205 행 / 49{,}730 경기 | — | 경기 가중 (1/n_m) | `G0_INTEGRITY_STATUS_20260921.json` `phase_b.headlines.cohorts.S_identity.delta_brier_q_minus_PT_flex` |
| `\gzPrintS` | -0.00335 | 문서 인쇄값 (S_identity) | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `...docs_delta_printed` |
| `\gzFoldN` | 5 | 재적재한 OOF 평가기 수 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `phase_b.oof_reload.folds` |
| `\gzOofStatus` | PASS | OOF 재적재 상태 | — | — | — | — | `G0_INTEGRITY_STATUS_20260921.json` `phase_b.oof_reload.status` |
