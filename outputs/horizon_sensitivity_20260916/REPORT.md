# 종료 상한 민감도: h90 선정 학습기 5종을 h60·h120 라벨로 재적합 (T/N) — 실행 보고서

실행 horizon_sensitivity_20260916. 설계·구현·실행·검증: Claude(Codex 라인 계승, 2026-09-16). 구성(하이퍼파라미터)은 h90 승자(iq·Track A)로 고정하고 보정만 Q_SELECT에서 선택했다. h90 열은 재적합하지 않았고 iq/Track A 결과를 서술적으로 병기한다(라벨이 달라 상한 간 구간은 계산하지 않음). 기존 TEST/외부 노출 후의 탐색적 후속 실험이며, 라벨 Y_h=1[ΔV_h>0]는 모델 정의 결과이지 실제 한타 승리 정답이 아니다.

## 1. 핵심 요약 (MAIN TEST 15.16, 짝지은 경기 부트스트랩 1000회, seed 20260915)

### h60 · T(한타, 주) — 32,981행/24,020경기

- 전체 Brier: 전체 LightGBM 0.22781; plain MLP 0.22775; residual MLP 0.22847; 전체 logistic 0.22789; PT 기준선 0.22903; 기존 A specialist 0.22767.
- 전체 · PRIMARY: full LightGBM - PT baseline: ΔBrier -0.00122 [-0.00167, -0.00076] (a 우세(구간 0 미포함)), ΔAUC +0.0046 [+0.0028, +0.0064].
- 전체 · planned: full logistic - PT baseline: ΔBrier -0.00113 [-0.00175, -0.00051] (a 우세(구간 0 미포함)), ΔAUC +0.0048 [+0.0023, +0.0073].
- 전체 · planned: full LightGBM - full logistic: ΔBrier -0.00009 [-0.00060, +0.00040] (구간이 0 포함), ΔAUC -0.0002 [-0.0021, +0.0019].
- 전체 · planned: plain MLP - full LightGBM: ΔBrier -0.00006 [-0.00063, +0.00054] (구간이 0 포함), ΔAUC +0.0011 [-0.0011, +0.0034].
- 전체 · planned: residual MLP - full LightGBM: ΔBrier +0.00066 [+0.00019, +0.00113] (b 우세(구간 0 미포함)), ΔAUC -0.0023 [-0.0044, -0.0002].
- 전체 · planned: plain MLP - full logistic: ΔBrier -0.00015 [-0.00057, +0.00026] (구간이 0 포함), ΔAUC +0.0009 [-0.0007, +0.0026].
- B40 Brier: 전체 LightGBM 0.24951; plain MLP 0.24844; residual MLP 0.24973; 전체 logistic 0.24828; PT 기준선 0.25024; 기존 A specialist 0.24768.
- B40 · PRIMARY: full LightGBM - PT baseline: ΔBrier -0.00073 [-0.00172, +0.00021] (구간이 0 포함), ΔAUC +0.0275 [+0.0124, +0.0445].
- B40 · planned: full logistic - PT baseline: ΔBrier -0.00196 [-0.00349, -0.00047] (a 우세(구간 0 미포함)), ΔAUC +0.0485 [+0.0292, +0.0694].
- B40 · planned: full LightGBM - full logistic: ΔBrier +0.00123 [-0.00008, +0.00253] (구간이 0 포함), ΔAUC -0.0209 [-0.0369, -0.0052].
- B40 · planned: plain MLP - full LightGBM: ΔBrier -0.00106 [-0.00262, +0.00047] (구간이 0 포함), ΔAUC +0.0192 [+0.0033, +0.0345].
- B40 · planned: residual MLP - full LightGBM: ΔBrier +0.00022 [-0.00130, +0.00154] (구간이 0 포함), ΔAUC -0.0013 [-0.0169, +0.0158].
- B40 · planned: plain MLP - full logistic: ΔBrier +0.00016 [-0.00079, +0.00111] (구간이 0 포함), ΔAUC -0.0017 [-0.0111, +0.0074].

### h60 · N(비한타, 보조) — 130,595행/54,182경기

- 전체 Brier: 전체 LightGBM 0.23898; plain MLP 0.23982; residual MLP 0.23990; 전체 logistic 0.24308; PT 기준선 0.24499; 기존 A specialist 0.24347.
- 전체 · PRIMARY: full LightGBM - PT baseline: ΔBrier -0.00600 [-0.00642, -0.00556] (a 우세(구간 0 미포함)), ΔAUC +0.0412 [+0.0383, +0.0440].
- 전체 · planned: full logistic - PT baseline: ΔBrier -0.00191 [-0.00227, -0.00153] (a 우세(구간 0 미포함)), ΔAUC +0.0200 [+0.0167, +0.0231].
- 전체 · planned: full LightGBM - full logistic: ΔBrier -0.00409 [-0.00450, -0.00365] (a 우세(구간 0 미포함)), ΔAUC +0.0213 [+0.0186, +0.0239].
- 전체 · planned: plain MLP - full LightGBM: ΔBrier +0.00084 [+0.00051, +0.00117] (b 우세(구간 0 미포함)), ΔAUC -0.0040 [-0.0060, -0.0020].
- 전체 · planned: residual MLP - full LightGBM: ΔBrier +0.00091 [+0.00059, +0.00126] (b 우세(구간 0 미포함)), ΔAUC -0.0044 [-0.0064, -0.0024].
- 전체 · planned: plain MLP - full logistic: ΔBrier -0.00326 [-0.00362, -0.00288] (a 우세(구간 0 미포함)), ΔAUC +0.0173 [+0.0151, +0.0195].
- B40 Brier: 전체 LightGBM 0.24376; plain MLP 0.24557; residual MLP 0.24546; 전체 logistic 0.24699; PT 기준선 0.24973; 기존 A specialist 0.24708.
- B40 · PRIMARY: full LightGBM - PT baseline: ΔBrier -0.00598 [-0.00669, -0.00525] (a 우세(구간 0 미포함)), ΔAUC +0.0709 [+0.0636, +0.0785].
- B40 · planned: full logistic - PT baseline: ΔBrier -0.00275 [-0.00329, -0.00217] (a 우세(구간 0 미포함)), ΔAUC +0.0463 [+0.0382, +0.0544].
- B40 · planned: full LightGBM - full logistic: ΔBrier -0.00323 [-0.00386, -0.00257] (a 우세(구간 0 미포함)), ΔAUC +0.0246 [+0.0190, +0.0302].
- B40 · planned: plain MLP - full LightGBM: ΔBrier +0.00181 [+0.00121, +0.00238] (b 우세(구간 0 미포함)), ΔAUC -0.0132 [-0.0181, -0.0082].
- B40 · planned: residual MLP - full LightGBM: ΔBrier +0.00170 [+0.00110, +0.00232] (b 우세(구간 0 미포함)), ΔAUC -0.0127 [-0.0177, -0.0075].
- B40 · planned: plain MLP - full logistic: ΔBrier -0.00142 [-0.00189, -0.00098] (a 우세(구간 0 미포함)), ΔAUC +0.0114 [+0.0072, +0.0159].

### h120 · T(한타, 주) — 32,981행/24,020경기

- 전체 Brier: 전체 LightGBM 0.22825; plain MLP 0.22845; residual MLP 0.22891; 전체 logistic 0.22831; PT 기준선 0.22938; 기존 A specialist 0.22814.
- 전체 · PRIMARY: full LightGBM - PT baseline: ΔBrier -0.00113 [-0.00163, -0.00063] (a 우세(구간 0 미포함)), ΔAUC +0.0043 [+0.0023, +0.0063].
- 전체 · planned: full logistic - PT baseline: ΔBrier -0.00106 [-0.00166, -0.00047] (a 우세(구간 0 미포함)), ΔAUC +0.0045 [+0.0020, +0.0070].
- 전체 · planned: full LightGBM - full logistic: ΔBrier -0.00007 [-0.00059, +0.00042] (구간이 0 포함), ΔAUC -0.0002 [-0.0023, +0.0018].
- 전체 · planned: plain MLP - full LightGBM: ΔBrier +0.00020 [-0.00033, +0.00078] (구간이 0 포함), ΔAUC +0.0004 [-0.0018, +0.0025].
- 전체 · planned: residual MLP - full LightGBM: ΔBrier +0.00067 [+0.00017, +0.00120] (b 우세(구간 0 미포함)), ΔAUC -0.0024 [-0.0046, -0.0004].
- 전체 · planned: plain MLP - full logistic: ΔBrier +0.00013 [-0.00028, +0.00054] (구간이 0 포함), ΔAUC +0.0002 [-0.0014, +0.0017].
- B40 Brier: 전체 LightGBM 0.24945; plain MLP 0.24840; residual MLP 0.24983; 전체 logistic 0.24864; PT 기준선 0.25037; 기존 A specialist 0.24803.
- B40 · PRIMARY: full LightGBM - PT baseline: ΔBrier -0.00091 [-0.00200, +0.00028] (구간이 0 포함), ΔAUC +0.0284 [+0.0116, +0.0449].
- B40 · planned: full logistic - PT baseline: ΔBrier -0.00172 [-0.00323, -0.00028] (a 우세(구간 0 미포함)), ΔAUC +0.0436 [+0.0243, +0.0634].
- B40 · planned: full LightGBM - full logistic: ΔBrier +0.00081 [-0.00047, +0.00211] (구간이 0 포함), ΔAUC -0.0152 [-0.0310, -0.0006].
- B40 · planned: plain MLP - full LightGBM: ΔBrier -0.00105 [-0.00245, +0.00038] (구간이 0 포함), ΔAUC +0.0130 [-0.0019, +0.0281].
- B40 · planned: residual MLP - full LightGBM: ΔBrier +0.00038 [-0.00106, +0.00168] (구간이 0 포함), ΔAUC -0.0052 [-0.0205, +0.0112].
- B40 · planned: plain MLP - full logistic: ΔBrier -0.00025 [-0.00110, +0.00064] (구간이 0 포함), ΔAUC -0.0022 [-0.0118, +0.0063].

### h120 · N(비한타, 보조) — 130,595행/54,182경기

- 전체 Brier: 전체 LightGBM 0.23880; plain MLP 0.23942; residual MLP 0.23985; 전체 logistic 0.24255; PT 기준선 0.24493; 기존 A specialist 0.24255.
- 전체 · PRIMARY: full LightGBM - PT baseline: ΔBrier -0.00613 [-0.00656, -0.00570] (a 우세(구간 0 미포함)), ΔAUC +0.0416 [+0.0388, +0.0443].
- 전체 · planned: full logistic - PT baseline: ΔBrier -0.00238 [-0.00282, -0.00196] (a 우세(구간 0 미포함)), ΔAUC +0.0207 [+0.0174, +0.0239].
- 전체 · planned: full LightGBM - full logistic: ΔBrier -0.00375 [-0.00417, -0.00330] (a 우세(구간 0 미포함)), ΔAUC +0.0209 [+0.0182, +0.0234].
- 전체 · planned: plain MLP - full LightGBM: ΔBrier +0.00062 [+0.00029, +0.00094] (b 우세(구간 0 미포함)), ΔAUC -0.0029 [-0.0048, -0.0011].
- 전체 · planned: residual MLP - full LightGBM: ΔBrier +0.00104 [+0.00075, +0.00135] (b 우세(구간 0 미포함)), ΔAUC -0.0053 [-0.0071, -0.0034].
- 전체 · planned: plain MLP - full logistic: ΔBrier -0.00313 [-0.00350, -0.00274] (a 우세(구간 0 미포함)), ΔAUC +0.0180 [+0.0157, +0.0202].
- B40 Brier: 전체 LightGBM 0.24358; plain MLP 0.24514; residual MLP 0.24535; 전체 logistic 0.24658; PT 기준선 0.24970; 기존 A specialist 0.24659.
- B40 · PRIMARY: full LightGBM - PT baseline: ΔBrier -0.00612 [-0.00681, -0.00541] (a 우세(구간 0 미포함)), ΔAUC +0.0702 [+0.0628, +0.0778].
- B40 · planned: full logistic - PT baseline: ΔBrier -0.00312 [-0.00380, -0.00238] (a 우세(구간 0 미포함)), ΔAUC +0.0470 [+0.0389, +0.0547].
- B40 · planned: full LightGBM - full logistic: ΔBrier -0.00300 [-0.00370, -0.00233] (a 우세(구간 0 미포함)), ΔAUC +0.0232 [+0.0178, +0.0292].
- B40 · planned: plain MLP - full LightGBM: ΔBrier +0.00156 [+0.00100, +0.00209] (b 우세(구간 0 미포함)), ΔAUC -0.0108 [-0.0159, -0.0058].
- B40 · planned: residual MLP - full LightGBM: ΔBrier +0.00177 [+0.00119, +0.00233] (b 우세(구간 0 미포함)), ΔAUC -0.0134 [-0.0183, -0.0084].
- B40 · planned: plain MLP - full logistic: ΔBrier -0.00144 [-0.00200, -0.00093] (a 우세(구간 0 미포함)), ΔAUC +0.0124 [+0.0083, +0.0172].

## 2. 상한별 비교(서술; MAIN TEST 전체 셀 Brier / AUC; h90은 iq·Track A의 동결 결과, 같은 행)

| cohort | 모델 | h60 Brier | h90 Brier | h120 Brier | h60 AUC | h90 AUC | h120 AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| T | PT 기준선 | 0.22903 | 0.22952 | 0.22938 | 0.6667 | 0.6648 | 0.6653 |
| T | 전체 logistic | 0.22789 | 0.22840 | 0.22831 | 0.6715 | 0.6694 | 0.6699 |
| T | 전체 LightGBM | 0.22781 | 0.22839 | 0.22825 | 0.6713 | 0.6691 | 0.6696 |
| T | plain MLP | 0.22775 | 0.22846 | 0.22845 | 0.6725 | 0.6694 | 0.6700 |
| T | residual MLP | 0.22847 | 0.22900 | 0.22891 | 0.6690 | 0.6668 | 0.6672 |
| T | 기존 A specialist | 0.22767 | 0.22817 | 0.22814 | 0.6722 | 0.6702 | 0.6704 |
| T | 기존 pooled q | 0.22921 | 0.22983 | 0.22968 | 0.6665 | 0.6641 | 0.6645 |
| T | 기존 상수 | 0.24999 | 0.25000 | 0.25000 | 0.5000 | 0.5000 | 0.5000 |
| N | PT 기준선 | 0.24499 | 0.24504 | 0.24493 | 0.5779 | 0.5779 | 0.5788 |
| N | 전체 logistic | 0.24308 | 0.24269 | 0.24255 | 0.5979 | 0.5983 | 0.5994 |
| N | 전체 LightGBM | 0.23898 | 0.23869 | 0.23880 | 0.6192 | 0.6211 | 0.6204 |
| N | plain MLP | 0.23982 | 0.23978 | 0.23942 | 0.6152 | 0.6159 | 0.6174 |
| N | residual MLP | 0.23990 | 0.23979 | 0.23985 | 0.6148 | 0.6155 | 0.6151 |
| N | 기존 A specialist | 0.24347 | 0.24268 | 0.24255 | 0.5915 | 0.5984 | 0.5991 |
| N | 기존 pooled q | 0.24333 | 0.24310 | 0.24290 | 0.5962 | 0.5968 | 0.5978 |
| N | 기존 상수 | 0.25000 | 0.25002 | 0.25002 | 0.5000 | 0.5000 | 0.5000 |

주의: 상한이 다르면 라벨 Y가 달라지므로 열 사이의 차이는 학습기 차이가 아니라 라벨(종료점) 차이를 포함한다. 같은 열 안의 순서만 학습기 비교다. h90의 기존 참조(A specialist·pooled·상수)는 iq 평가에서, h60·h120은 cohort_role 참조를 이 실행에서 같은 행에 붙인 것이다.

## 3. 선정된 보정과 정지 기록

| horizon | cohort | family | 고정 구성 | 선정 보정 | Q_SELECT Brier | 정지 기록(시드 7/42/123) | 적합 초 |
|---|---|---|---|---|---:|---|---:|
| h60 | T | PT 기준선 | pt_C1 | raw | 0.228552 | — | 2.4 |
| h60 | T | 전체 logistic | logit_C0.001 | raw | 0.228584 | — | 2.7 |
| h60 | T | 전체 LightGBM | lgbm_L15_M100 | sigmoid | 0.227891 | 반복 98/242/198 | 10.1 |
| h60 | T | plain MLP | mlp_W128_D0.3 | raw | 0.228520 | epoch 3/4/9 | 22.8 |
| h60 | T | residual MLP | resmlp_W128_D0.1 | raw | 0.228610 | epoch 1/1/1 | 42.5 |
| h60 | N | PT 기준선 | pt_C1 | raw | 0.244158 | — | 8.7 |
| h60 | N | 전체 logistic | logit_C0.01 | raw | 0.242675 | — | 19.1 |
| h60 | N | 전체 LightGBM | lgbm_L15_M100 | sigmoid | 0.238319 | 반복 612/820/579 | 56.6 |
| h60 | N | plain MLP | mlp_W128_D0.3 | sigmoid | 0.239446 | epoch 5/7/6 | 109.0 |
| h60 | N | residual MLP | resmlp_W128_D0.1 | raw | 0.239472 | epoch 7/2/4 | 200.3 |
| h120 | T | PT 기준선 | pt_C1 | raw | 0.229384 | — | 1.9 |
| h120 | T | 전체 logistic | logit_C0.001 | raw | 0.228929 | — | 2.2 |
| h120 | T | 전체 LightGBM | lgbm_L15_M100 | raw | 0.228721 | 반복 291/239/157 | 11.0 |
| h120 | T | plain MLP | mlp_W128_D0.3 | raw | 0.228732 | epoch 3/4/8 | 39.5 |
| h120 | T | residual MLP | resmlp_W128_D0.1 | raw | 0.229154 | epoch 1/1/1 | 52.4 |
| h120 | N | PT 기준선 | pt_C1 | raw | 0.244131 | — | 7.3 |
| h120 | N | 전체 logistic | logit_C0.01 | isotonic | 0.242522 | — | 18.1 |
| h120 | N | 전체 LightGBM | lgbm_L15_M100 | sigmoid | 0.238320 | 반복 489/398/814 | 50.2 |
| h120 | N | plain MLP | mlp_W128_D0.3 | sigmoid | 0.239274 | epoch 5/7/10 | 182.4 |
| h120 | N | residual MLP | resmlp_W128_D0.1 | sigmoid | 0.239683 | epoch 4/2/4 | 227.4 |

## 4. 외부 세트(전체 행; 주 대비 LightGBM − PT의 판정; 이전 노출 있음)

| horizon | 세트 | cohort | 행 | LightGBM Brier | plain MLP | residual MLP | logistic | PT | LGBM−PT ΔBrier [95%] | 판정 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| h60 | EXT_KR_16.13 | T | 5,202 | 0.23499 | 0.23523 | 0.23506 | 0.23508 | 0.23629 | -0.00130 [-0.00245, -0.00015] | a 우세(구간 0 미포함) |
| h60 | EXT_KR_16.13 | N | 20,228 | 0.24132 | 0.24172 | 0.24178 | 0.24462 | 0.24592 | -0.00460 [-0.00563, -0.00354] | a 우세(구간 0 미포함) |
| h60 | EXT_KR_16.14_pilot | T | 101 | 0.22769 | 0.22776 | 0.22517 | 0.22888 | 0.22788 | -0.00019 [-0.00665, +0.00671] | 구간이 0 포함 |
| h60 | EXT_KR_16.14_pilot | N | 374 | 0.24162 | 0.24339 | 0.24400 | 0.24814 | 0.24448 | -0.00286 [-0.01164, +0.00605] | 구간이 0 포함 |
| h60 | EXT_KR_16.15 | T | 507 | 0.23498 | 0.23627 | 0.23640 | 0.23704 | 0.23753 | -0.00255 [-0.00632, +0.00097] | 구간이 0 포함 |
| h60 | EXT_KR_16.15 | N | 1,712 | 0.24083 | 0.24044 | 0.24124 | 0.24230 | 0.24692 | -0.00609 [-0.00978, -0.00219] | a 우세(구간 0 미포함) |
| h60 | EXT_NA1_16.13 | T | 5,312 | 0.23826 | 0.24133 | 0.24081 | 0.24274 | 0.24024 | -0.00198 [-0.00317, -0.00082] | a 우세(구간 0 미포함) |
| h60 | EXT_NA1_16.13 | N | 21,226 | 0.24225 | 0.24172 | 0.24166 | 0.24487 | 0.24660 | -0.00436 [-0.00541, -0.00333] | a 우세(구간 0 미포함) |
| h120 | EXT_KR_16.13 | T | 5,202 | 0.23587 | 0.23629 | 0.23571 | 0.23581 | 0.23713 | -0.00126 [-0.00246, -0.00005] | a 우세(구간 0 미포함) |
| h120 | EXT_KR_16.13 | N | 20,228 | 0.24166 | 0.24169 | 0.24212 | 0.24478 | 0.24603 | -0.00438 [-0.00543, -0.00337] | a 우세(구간 0 미포함) |
| h120 | EXT_KR_16.14_pilot | T | 101 | 0.22955 | 0.22508 | 0.22425 | 0.22851 | 0.22677 | +0.00278 [-0.00417, +0.01058] | 구간이 0 포함 |
| h120 | EXT_KR_16.14_pilot | N | 374 | 0.23985 | 0.24162 | 0.24281 | 0.24707 | 0.24359 | -0.00373 [-0.01201, +0.00409] | 구간이 0 포함 |
| h120 | EXT_KR_16.15 | T | 507 | 0.23548 | 0.23774 | 0.23774 | 0.23799 | 0.23798 | -0.00250 [-0.00677, +0.00147] | 구간이 0 포함 |
| h120 | EXT_KR_16.15 | N | 1,712 | 0.24237 | 0.24031 | 0.24123 | 0.24172 | 0.24618 | -0.00381 [-0.00734, +0.00015] | 구간이 0 포함 |
| h120 | EXT_NA1_16.13 | T | 5,312 | 0.23831 | 0.24175 | 0.24140 | 0.24274 | 0.24053 | -0.00222 [-0.00351, -0.00086] | a 우세(구간 0 미포함) |
| h120 | EXT_NA1_16.13 | N | 21,226 | 0.24180 | 0.24133 | 0.24129 | 0.24455 | 0.24666 | -0.00486 [-0.00590, -0.00386] | a 우세(구간 0 미포함) |

## 5. 해석

- h60 T 전체: full LightGBM - PT baseline -0.00122 → a 우세(구간 0 미포함); full logistic - PT baseline -0.00113 → a 우세(구간 0 미포함); full LightGBM - full logistic -0.00009 → 구간이 0 포함; plain MLP - full LightGBM -0.00006 → 구간이 0 포함; residual MLP - full LightGBM +0.00066 → b 우세(구간 0 미포함); plain MLP - full logistic -0.00015 → 구간이 0 포함.
- h60 N 전체: full LightGBM - PT baseline -0.00600 → a 우세(구간 0 미포함); full logistic - PT baseline -0.00191 → a 우세(구간 0 미포함); full LightGBM - full logistic -0.00409 → a 우세(구간 0 미포함); plain MLP - full LightGBM +0.00084 → b 우세(구간 0 미포함); residual MLP - full LightGBM +0.00091 → b 우세(구간 0 미포함); plain MLP - full logistic -0.00326 → a 우세(구간 0 미포함).
- h120 T 전체: full LightGBM - PT baseline -0.00113 → a 우세(구간 0 미포함); full logistic - PT baseline -0.00106 → a 우세(구간 0 미포함); full LightGBM - full logistic -0.00007 → 구간이 0 포함; plain MLP - full LightGBM +0.00020 → 구간이 0 포함; residual MLP - full LightGBM +0.00067 → b 우세(구간 0 미포함); plain MLP - full logistic +0.00013 → 구간이 0 포함.
- h120 N 전체: full LightGBM - PT baseline -0.00613 → a 우세(구간 0 미포함); full logistic - PT baseline -0.00238 → a 우세(구간 0 미포함); full LightGBM - full logistic -0.00375 → a 우세(구간 0 미포함); plain MLP - full LightGBM +0.00062 → b 우세(구간 0 미포함); residual MLP - full LightGBM +0.00104 → b 우세(구간 0 미포함); plain MLP - full logistic -0.00313 → a 우세(구간 0 미포함).
- 학습기 순서가 상한에 따라 바뀌는지가 이 실험의 질문이다. 위 판정을 h90(iq: LightGBM ≈ logistic; Track A: plain MLP ≈ LightGBM > residual MLP, N에서 LightGBM > MLP > logistic)와 견주어 읽는다.
- 부트스트랩 구간은 고정 모델의 평가 표본 불확실성만 반영한다(학습·보정·선정 불확실성 제외, 다중 비교 보정 없음). 구성은 h90에서 선정된 것이므로 h60/h120에서 최적이라는 보장이 없다.

## 6. 검증과 실패 기록

- validation.json: 91개 검사, 실패 0개 (2026-09-16 02:58:56).
- 계약 테스트 run2 10 passed (2026-09-16 02:32:43); 스냅샷 → 테스트 → 프로토콜 → smoke(h60) → 테스트 → 전체 적합(h60, h120) → 동결 → 봉인 평가 순서. 동결 2026-09-16 02:54:55.

## 7. 한계와 비주장

- h60/h120은 보조 결과이며 주 분석은 h90이다. 상한을 TEST 결과로 고르지 않았다.
- 구성 고정·보정만 선택이므로 각 상한에서의 최적 구성 비교가 아니다. 사람 검토·인과효과·미접촉 확증은 포함하지 않는다.

## 8. 산출물

- `protocol.json` 사전 동결 계약·고정 구성 등록부
- `selection/h<h>_<family>_<cohort>.json` 3후보 지표·선정
- `internal_stop/` LightGBM·MLP 정지 기록
- `models/h<h>/<cohort>/<family>/<config>.joblib` 번들
- `frozen_manifest.json` 동결(두 상한, 부모 참조 hash)
- `eval/results.json`, `eval/predictions/<set>_h<h>_<cohort>.npz`
- `validation.json` 사후 검증
- `integrity/`, `status.json`, `logs/`, `commands.txt`, `failures.jsonl`
