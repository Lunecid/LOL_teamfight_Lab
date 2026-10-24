r"""S3 report: S3_REPORT.json and five figures of a finished S3 run folder (stage S3 Task 13; what the controller builds the gate page G-S3c
from).

    python -m c_ept.report_s3 --run <S3 run folder> [--exclude-smoke <smoke run folder>] [--s1 <S1 run>] [--s3a <s3a run>]
                              [--runs-root <folder>] [--report-dir <folder>]

``report(run, exclude_smoke=None)`` reads files only and returns the path of ``<run>\report\S3_REPORT.json``; the figures are beside it.
It fits nothing, simulates nothing, loads no model and reads no raw match, so it does not need the E6 check of the stages (no Strata
process): it works on what the stages wrote. Every number of the report comes from those files or is computed from them; none is typed in.

What it reads. (1) The run folder: RUN.json, STAGES.json and the files of the stages ``labels``, ``fit``, ``compose`` and ``gate``, after
three checks: every stage is complete, RUN.json holds the ``<stage>_result`` of each and the complete ``s3_result``, and the output files
of every stage still hash to what STAGES.json recorded when the stage finished (``run_s3.stage_outputs``). (2) ``PRICES.json`` and
``RULE_PAYOUTS.json`` of the s3a run (their SHA-256 must be those of RUN.json). (3) The S1 store, found at ``--s1``, next to the run
folder or in the runs root, opened by ``run_s1.load_s1(verify=False)`` (its files hash must be the one of RUN.json): ``onsets/row``,
``onsets/bin`` and ``frame_rows/t0`` for the observed onsets, ``matches/key`` for the key scan. (4) For the sensitivity line alone, the
``smoke`` selection in RUN.json of a smoke run: ``exclude_smoke``, and, for a run that is not a smoke run and has none, the run that
``decisions.S3_run.smoke_run`` of the decision record names (the file ``args.spec`` of RUN.json), found next to the run folder or in the
runs root. The selection that the smoke run recorded must be the one ``run_s3.select_smoke`` gives for its n and seed on the S1 store
(``check_smoke_selection``). A run that is not finished, a file that changed after the run, an input that is not the one the run was
made from, a smoke selection that does not check, or a field that is missing is a ``ReportInputError`` (the command line prints
``report refused: ...`` and returns 2); nothing is written then. A line that cannot be computed because no smoke run can be named or
found is not a refusal: it is flagged (``SENSITIVITY_NOT_COMPUTED``). A run from before the G-S3b fix round lacks some optional fields
(``peak_private_upper_gib`` beside ``peak_private_gib``, the chunk counters of COMPOSE.json): they are None in the report, listed in
``inputs.missing_optional_fields`` and flagged (``OLD_RECORD_LAYOUT``). A call of the runner that was killed (no ``finished``) is listed
in ``cost.unfinished_calls`` and is no case of that; the B1 rounds of a fit stage that was resumed after B1 was saved come from the
lines of PROGRESS.jsonl, and what is still None is flagged (``B1_ROUNDS_NOT_RECORDED``).

S3_REPORT.json (the keys are sorted in the file, so what has an order has a list: ``group_order``, ``stage_order``, ``dev_nll.heads``).
The items of the plan, in order: ``d7`` (the beta curves, the flow price, the immediate value with its interval, whether the flow price
was adopted and why, the share of events left out because their window is not complete), ``dev_nll`` (heads x B0/B1 x masked/open),
``brier`` (M2 of B0 and B1 against B2), ``occurrence_deciles``, ``value_deciles`` (the deciles of the gate and, at H = 60 s, the PIT, the
hit rates and the CRPS from 19 quantiles of B0 and B1), ``mc_agreement`` (with the flat rows and, in ``by_expected_onsets``, the same
agreement in five bins of the expected onsets of the row), ``overdispersion``, ``accounting`` (the identity as GATE.json judges it, and
beside it ``value_check``: the gold half of criterion (iii) at the tolerance of ruling R48, which never changes ``gate_pass``), ``winner``
(shares of winners and draws, labelled with the units they count), ``dropped_columns``, ``b1_composition``. The carries of the
controller: ``sensitivity_without_smoke_dev``,
``window_slopes`` (B0 and B1, the windows 0-60, 60-120, 120-180 s and 0-180 s, and a check that the 0-60 s line of B0 is the occurrence
result of GATE.json), ``b0_convergence`` (B4), ``cost`` and ``key_scan``. Besides: ``run``, ``inputs``, ``gate`` (the verdict as GATE.json
holds it, with the bands: the report never judges the gate again, ruling R27), ``tower_pricing``, ``flags`` (what the reader should know
first), ``figures`` (file, title and the numbers drawn), ``report_code``, ``created``.

Two labels are binding (controller carry of 2026-10-22). (a) ``peak_private_upper_gib`` of the run records retains the maximum across
observations of sums of readable observed processes' lifetime peak_pagefile values. It may include pre-stage history and miss terminated
or unreadable processes; it is neither a complete historical-tree bound nor a per-stage peak (S3 rulings R79, R81 and R82). The report
names it ``lifetime_peak_bound_gib``, labels it ``LIFETIME_PEAK_LABEL`` and judges memory by ``peak_private_gib``, the sampled peak
(S3 ruling R43). (b) Criterion (iv) is reported with ``n_sd_zero`` and ``n_sd_zero_disagree`` beside the agreement share: a flat row (MC sd
0) agrees only on exact equality (ruling R28), and the section says how many of the disagreements come from flat rows.

Statistics. Every interval is a percentile 95% interval of B = 2,000 match-bootstrap replicates from the seed 20261021 with the weights 1
/ n_u, made by the functions of ``gate_s3`` (``occurrence_calibration``, ``value_calibration``), so the lines of the report and the gate
are the same statistic computed the same way. The windows: the expected onsets of a window are the differences of the cumulative
``expected_onsets`` columns of the compose files (a difference of two sums that is negative by rounding is set to 0), the observed ones
the S1 onsets in the bins 0-5, 6-11 and 12-17. B1 is composed on a seeded sample of the DEV matches (500 in the full run): its lines are
made on those rows alone and the report says so. The diagnostic from 19 quantiles has 20 bins; a row that ties with quantiles spreads
its weight evenly over the inclusive bins it covers. This coarse quantile-bin rule is not exact randomized PIT from a known full CDF;
nominal uniformity is not guaranteed even for a calibrated discrete law (S3 rulings R81 and R82). A Monte Carlo mean contains simulation
error (B1 has S = 200 simulations and no closed form); ``expected_slope_if_calibrated`` is a plug-in attenuation reference requiring
zero-mean MC error orthogonal to the latent mean and realized residual, with suitable independent draws for the estimated mean-error
variance. Finite weighting, centering and cross-row covariance prevent an exact calibrated finite-sample slope guarantee. The lines made
on a Monte Carlo mean (B1, and B0 on its simulated mean) carry no band flag: they carry that reference, ``slope_ratio_to_expected`` (the slope divided by
it) and the note to read them against it and not against 1; only B0 on its closed form is held to the band [0.8, 1.2] of the gate.

Keys. The finished text is checked before anything is written: the match keys of the S1 store by the scan of the runner
(``run_s3.Context``), and the game ids as whole runs of digits (``assert_clean``); a hit stops the report with the number of hits and
writes nothing. The figures are drawn in memory first, the PNG files are written next and S3_REPORT.json last, atomically. The traceback of
an unexpected error that the command line prints has the keys and game ids of the S1 store taken out (``scrubbed_traceback``); if they
cannot be read, only the type of the error is printed.

Figures (PNG, Malgun Gothic, mathtext for the subscripts): ``fig1_d7_beta`` (beta_r of the twelve groups with their intervals),
``fig2_occurrence_deciles`` (observed against expected onsets), ``fig3_value_deciles_pit`` (V_start deciles and the PIT histogram of H =
60 s), ``fig4_window_slopes`` (the slope per window with the band [0.9, 1.1]) and ``fig5_brier`` (the Brier scores with their intervals
and the paired differences).


방법. 이 모듈은 S3 Task13의 완료된 실행과 관문 기록을 보고 합성 가치·발생·가격·비용을 정리한다.
교전은 설계서의 연결된 킬 묶음이며 f₀는 도움 풀을 포함한 킬 골드와 귀속 오브젝트 가격 합의 파랑−빨강 값이다.
전환 가치는 교전 뒤 전환 오브젝트 가격의 같은 부호 차이이다(value의 VAL-4; compose의 COMP-4).
V_start는 첫 킬이 (t₀, t₀+H] 안이고 관측 창 끝 이하인 범위 안 교전들의 f₀와 전환 가치의 합이다.
선택된 교전의 전체 결과와 전환 가치는 지평 뒤에 생긴 부분도 포함한다(value의 VAL-8).
기본 보고 구간은 경기 가중 경기 부트스트랩의 백분위 구간이고 관문 판정은 기존 기록에서 보존한다.
공통 가정. 저장 경기·프레임 행은 0 기반이며 배열 정렬·고유성은 검사 범위 밖에서는 상류 계약이다.
수치 변환은 해당 함수의 실제 dtype을 따르고 시각의 ms/s는 해당 표에 구별한다.
S3 R81 한계: 유한 표본의 진단, 약한 선형 보정, 완전한 조건부 보정은 서로 다른 주장이다.
PIT는 분위수 빈의 거친 진단이고 MC 기울기 기준은 가정이 필요한 plug-in 근사이다(S3 판정 R81·R82).
수명 피크 합의 기록은 관측한 읽기 가능 프로세스 범위에 한정된다(S3 판정 R79·R82).
피어슨 분산비는 평균 모형이 맞을 때만 과산포 해석이 가능하다(S3 계획 P8; gate_s3.overdispersion).
식 목록. 보고의 통계와 표시값은 다음 공개 함수가 소유한다.
    (REP3-1) 분위수 빈 질량 진단 → pit_histogram
    (REP3-2) 엄격·포함 분위 적중률 → quantile_hits
    (REP3-3) 분위 격자 핀볼 대리 점수 → crps19
    (REP3-4) 창별 기대 개수 → window_expectations
    (REP3-5) 불일치 카운터 분해 → mc_section
    (REP3-6) 발생 수준별 모의 진단 → mc_by_rate
    (REP3-7) 불완전 창 사건 비율 → d7_section
    (REP3-8) 반복 한계 카운터 → convergence_section
    (REP3-9) 골드 회계 허용차 → accounting_value_check
    (REP3-10) 관문 수치 재현 대조 → compare_with_gate
    (REP3-11) 소규모 점검 경기 제외 후 경기 가중 → sensitivity_without_smoke
    (REP3-12) 모의 오차 감쇠 plug-in 기준 → expected_slope_if_calibrated
    (REP3-13) 가치의 약한 선형 보정 요약 → value_summary
    (REP3-14) 감쇠 기준 대비 비 → mc_value_summary
    (REP3-15) B1 행의 B0 위치 → b1_positions
    (REP3-16) 표본 행 비교와 모의 표준오차 → b1_composition
    (REP3-17) 기록된 메모리의 표시 → cost_section
    (REP3-18) 창별 관측 개수 → window_observed
    (REP3-19) 창별 약한 선형 발생 진단 → window_lines
    (REP3-20) 대체 입력 칸 집계 → build_report
    (REP3-21) PIT 진단 빈의 표시 위치 → fig_value
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md Task 13 (the report), decisions P6 (B1 composition: simulation only, reported) and
#      P7 (the gate and its bands), and the rulings R27 (gate_pass is the gate's), R28 (flat rows), R36 (the B0 M2Rate windows), R43
#      (the cap is judged on the sampled peak) and R48 (criterion (iii) at both levels: the gold half at 1e-6 gold).
# Ref: Czado, C., Gneiting, T. and Held, L. (2009), "Predictive model assessment for count data", Biometrics 65(4): 1254-1261 (the
#      exact randomized PIT for a known full CDF with point masses; background only, not an attribution of this coarse quantile-bin rule).
# Ref: Gneiting, T. and Raftery, A. E. (2007), "Strictly proper scoring rules, prediction, and estimation", JASA 102(477): 359-378 (the
#      CRPS is twice the integral of the quantile (pinball) loss over the levels; ``crps19`` takes the 19 levels k / 20 of design
#      section S1-4).
# Ref: Brier, G. W. (1950), Monthly Weather Review 78(1): 1-3, and Efron (1979), Annals of Statistics 7(1): 1-26 (as in ``run_s3`` and
#      ``gate_s3``).
# Ref: Fuller, W. A. (1987), "Measurement Error Models", Wiley (the slope of a regression on a variable measured with error is the true
#      slope times the reliability ratio var(true) / var(measured): ``expected_slope_if_calibrated``).
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from c_ept import config as C
from c_ept import features, flowprice, gate_s3, head_rows, nll, run_s3, runlog, store_core

BOOT_B = nll.BOOT_B                                # the match bootstrap of every S3 interval: B = 2,000 replicates ...
SEED = nll.BOOT_SEED                               # ... from this seed (the gate's constants; a test pins them)
N_QUANTILES = 19                                   # compose.Q_LEVELS: the quantiles 0.05 .. 0.95 of the simulated V_start
LEVELS = np.arange(1, N_QUANTILES + 1) / 20.0      # tau_k = k / 20
N_PIT_BINS = N_QUANTILES + 1                       # the 19 quantiles cut the unit interval of the PIT into 20 bins of 0.05
N_WINDOWS = 3                                      # 0-60, 60-120 and 120-180 s
WINDOWS = ((0, 60), (60, 120), (120, 180), (0, 180))      # the occurrence lines: the three windows and the whole 180 s
CUMULATIVE_TOLERANCE = 1e-9                        # a window difference of two sums may be negative by this (relative) rounding
FLAT_RELATIVE = 1e-9                               # expected counts whose range is below this (relative to their size) do not vary


class ReportInputError(ValueError):
    """The inputs of the report do not belong together or lack something the report needs. The message names the field, never a key.

    완료된 실행의 입력과 보고에 필요한 필드가 맞지 않을 때 쓰는 예외이다.
    """


# ---- the small statistics -------------------------------------------------------------------------------------------------------------
def _quantile_rows(fn: str, q19, realized, w) -> tuple:
    """``(q19, realized, w, total weight)`` as float64 arrays, or ReportInputError for shapes, values and weights that cannot be judged."""
    q = np.asarray(q19, dtype=np.float64)
    y = np.asarray(realized, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != N_QUANTILES:
        raise ReportInputError(f'{fn}: q19 must have the shape (n, {N_QUANTILES}), not {q.shape}')
    n = len(q)
    if n == 0:
        raise ReportInputError(f'{fn}: there is no row')
    if y.shape != (n,) or w.shape != (n,):
        raise ReportInputError(f'{fn}: realized {y.shape} and w {w.shape} must be vectors of the {n} rows of q19')
    if not (np.isfinite(q).all() and np.isfinite(y).all()):
        raise ReportInputError(f'{fn}: q19 or realized holds a NaN or an infinite value')
    if (np.diff(q, axis=1) < 0).any():
        raise ReportInputError(f'{fn}: a row of q19 decreases: the quantiles must not decrease with the level')
    if not np.isfinite(w).all() or (w < 0).any():
        raise ReportInputError(f'{fn}: a weight is negative, NaN or infinite')
    total = math.fsum(w)
    if not total > 0:
        raise ReportInputError(f'{fn}: the weights add up to 0')
    return q, y, w, total


def pit_histogram(q19, realized, w) -> dict:
    """The probability-integral-transform histogram of ``realized`` against predictive laws given by 19 quantiles (the levels k / 20).

    ``q19`` is (n, 19), non-decreasing along each row; ``realized`` and ``w`` (weights, sum > 0) are (n,). For a row, ``n_lt`` is the
    number of quantiles below the realized value and ``n_le`` the number at or below it. Bin j (j = 0 .. 19) is the PIT interval [j / 20,
    (j + 1) / 20]: a row with ``n_lt == n_le`` lies in bin ``n_lt``; a row that ties with quantiles (a point mass of the law, for example
    V_start = 0 when no engagement starts) has its unit weight spread evenly over the bins ``n_lt .. n_le``, as a coarse quantile-bin diagnostic rather than the exact
    randomized PIT of a known full discrete CDF. The nominal uniform reference need not hold even for a calibrated discrete law. Returns ``{'bins', 'mass' (the weighted share of each bin, adding up to 1),
    'expected_mass' (0.05), 'n_rows', 'n_tied_rows', 'tied_weight_share', 'max_abs_deviation' (of the mass from 0.05)}``. ReportInputError
    for inputs that cannot be judged.

    수식
    ----
    (REP3-1)  less_row = Σ_level [quantile_{row,level}<realized_row]; leq_row = Σ_level [quantile_{row,level}≤realized_row]
             span_row = leq_row−less_row+1
             mass_bin = max(0, Σ_row weight_row·[less_row≤bin≤leq_row]/span_row / weight_total)
             tied_rows = Σ_row [span_row>1]; tied_share = Σ_row weight_row·[span_row>1]/weight_total
             max_deviation = max_bin abs(mass_bin−1/20)

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, level, bin | 행·분위 열·진단 빈 | — | 0 기반: level=0..18, bin=0..19 | q,y,bins
      quantile, realized | 예측 분위수와 실현 가치 | 골드 | 유한, 분위 행 비감소 검사 | q19,realized
      weight_row, weight_total | 행 가중치와 합 | — | 유한·비음수, 합 양수 검사 | w,total
      less_row, leq_row, span_row | 엄격·포함 비교 수와 빈 수 | 개 | 0..19; span=1..20 | n_lt,n_le,span
      mass_bin, tied_rows, tied_share, max_deviation | 빈 질량·동점 행 수·가중 비율·최대 차이 | —; 행; — | 영 가중 행도 tied_rows에 포함 | mass,tied.sum(),반환 필드

    해석
    ----
    S3 R81에 따라 이는 동점 행의 가중치를 양 끝을 포함한 빈들에 균등 배분하는 거친 진단이다. 알려진 전체 이산 CDF의 정확한 randomized PIT와 같지 않으며 명목 기준 0.05는 보정된 이산 법에서도 보장되지 않는다. 따라서 기준에서 벗어난 것만으로 보정 실패를 증명하지 못한다. 양자화 전 분포를 분위수만으로 복원하지 않으며 중복 분위수는 허용된다.

    근거
    ----
    S3 계획 Task13; 설계서 §8(분위 진단); S3 판정 R81(거친 진단 한계).
    """
    q, y, w, total = _quantile_rows('pit_histogram', q19, realized, w)
    # (REP3-1): 정확한 CDF 점프 경계 대신 분위수 비교 수로 포함 빈들을 정한다.
    n_lt = (q < y[:, None]).sum(axis=1)
    n_le = (q <= y[:, None]).sum(axis=1)
    span = n_le - n_lt + 1
    bins = np.arange(N_PIT_BINS)[None, :]
    inside = (bins >= n_lt[:, None]) & (bins <= n_le[:, None])
    # (REP3-1)의 행 가중치는 양 끝 포함 빈 수로 나눠 배분한다.
    mass = np.maximum((inside * (w / span)[:, None]).sum(axis=0) / total, 0.0)
    tied = span > 1
    return {'bins': N_PIT_BINS, 'mass': mass, 'expected_mass': 1.0 / N_PIT_BINS, 'n_rows': int(len(y)), 'n_tied_rows': int(tied.sum()),
            'tied_weight_share': math.fsum(w[tied]) / total, 'max_abs_deviation': float(np.max(np.abs(mass - 1.0 / N_PIT_BINS)))}


def quantile_hits(q19, realized, w) -> dict:
    """For each level k / 20 the weighted share of rows whose realized value lies below the quantile (``below``) and at or below it
    (``at_or_below``). Population strict and inclusive probabilities bracket a calibrated level (they differ where the law has a point mass at
    the quantile); the finite-sample weighted shares need not bracket it. ``{'levels', 'below', 'at_or_below'}``.

    수식
    ----
    (REP3-2)  below_level = Σ_row weight_row·[realized_row<quantile_{row,level}]/weight_total
             at_or_below_level = Σ_row weight_row·[realized_row≤quantile_{row,level}]/weight_total
             nominal_level = (level+1)/20

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, level | 관측 행과 분위수 열 | — | 0 기반, level=0..18 | q,y,LEVELS
      quantile, realized | 예측 분위수와 실현값 | 골드 | 유한, 분위수 비감소 검사 | q19,realized
      weight_row, weight_total | 가중치와 합 | — | 유한·비음수, 합 양수 검사 | w,total
      below_level, at_or_below_level, nominal_level | 두 가중 적중률과 명목 수준 | — | [0,1]; 명목 0.05..0.95 | below,at_or_below,LEVELS

    해석
    ----
    S3 R81은 분위 수준의 보정을 모집단의 엄격·포함 확률 사이에 명목 수준이 놓이는 성질로 구별한다. 유한 표본의 가중 적중률에는 그 괄호 관계가 반드시 성립하지 않는다. 두 비율의 차이는 그 분위수 값과의 동점 질량을 나타내며 완전한 조건부 보정을 보증하지 않는다.

    근거
    ----
    설계서 §8(분위 적중률); S3 판정 R81(모집단과 표본).
    """
    q, y, w, total = _quantile_rows('quantile_hits', q19, realized, w)
    # (REP3-2): 동점을 분리하려고 엄격 비교와 포함 비교를 함께 보고한다.
    below = (w[:, None] * (y[:, None] < q)).sum(axis=0) / total
    at_or_below = (w[:, None] * (y[:, None] <= q)).sum(axis=0) / total
    return {'levels': LEVELS, 'below': below, 'at_or_below': at_or_below}


def crps19(q19, realized, w) -> float:
    """The finite-grid pinball proxy for CRPS, as the design measures V_start (S1-4): twice the mean over the 19 levels of the pinball loss at
    tau = k / 20, averaged over the rows with the weights ``w``. For a row whose 19 quantiles all equal quantile_value, that row's loss equals |realized - quantile_value|. If this holds for every
    row, the returned scalar is the weighted mean of those row absolute errors, and quantile_value may differ by row. This does not
    identify the full law.

    수식
    ----
    (REP3-3)  error_{row,level} = realized_row−quantile_{row,level}; nominal_level=(level+1)/20
             pinball_{row,level} = nominal_level·error (error≥0), (nominal_level−1)·error (error<0)
             score19 = Σ_row weight_row·(2/19)·Σ_level pinball_{row,level}/weight_total

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, level, nominal_level | 관측 행·분위 열·수준 | — | 0 기반 열 0..18 | q,LEVELS
      quantile, realized, error, pinball, score19 | 분위수·실현·잔차·손실·점수 | 골드 | 유한 입력, 손실 비음수 | q19,y,error,pinball,반환값
      weight_row, weight_total | 가중치와 합 | — | 유한·비음수, 합 양수 검사 | w,total

    해석
    ----
    설계서 S1-4의 같은 분위 격자로 평가하며 연속 분위 적분 CRPS의 유한 격자 대리값이다. S3 R81에 따라 한 행의 분위수가 모두 같으면 그 행의 손실은 해당 값과 실현값의 절대오차이다. 모든 행이 이 조건을 만족하면 반환값은 행마다 다를 수 있는 그 절대오차들의 가중평균이다. 이 조건만으로 전체 예측 법이 점질량임을 알 수는 없다. 이 점수의 비교는 같은 대상·가중치에서 이루어져야 한다.

    근거
    ----
    설계서 §8·S1-4(공통 분위 격자); S3 판정 R81(대리 점수 한계).
    """
    q, y, w, total = _quantile_rows('crps19', q19, realized, w)
    # (REP3-3): 같은 분위 격자의 핀볼 손실을 써 모형 간 평가 척도를 맞춘다.
    error = y[:, None] - q
    pinball = np.where(error >= 0, LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    return math.fsum(w * (2.0 * pinball.mean(axis=1))) / total


def window_expectations(cumulative) -> np.ndarray:
    """(n, 3): the expected onsets of the windows 0-60, 60-120 and 120-180 s from the cumulative columns of ``compose`` (up to 60, 120 and
    180 s): the first column and the differences. Two sums of the same masses can differ in the last bit, so a difference that is negative
    by no more than 1e-9 x max(1, abs(current cumulative value)) is set to 0; a larger decrease, a NaN or another shape is a ReportInputError.

    수식
    ----
    (REP3-4)  raw_diff_{row,window} = cumulative_{row,window} − cumulative_{row,window−1}, cumulative_{row,−1}=0
             tolerance_{row,window}=10^−9·max(1,abs(cumulative_{row,window}))
             expected_count = max(raw_diff,0), 단 raw_diff < −tolerance이면 거부

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, window | 행과 지평 열 | — | 0 기반, window=0,1,2 | c,diff
      cumulative, raw_diff, expected_count | 누적·차분·창별 기대 발생 수 | 개 | 유한 (n,3) 검사, n=0도 이 함수는 허용 | cumulative,c,diff,반환값
      tolerance | 차분의 상대 허용차 | 개 | 현재 누적 절댓값에 비례 | CUMULATIVE_TOLERANCE

    해석
    ----
    S3 Task13의 창별 진단은 누적 기대 개수를 차분하며 확률로 해석하지 않는다. 작은 음수의 영 절단은 합 차분의 반올림에 대한 구현 규약이다. 절단한 세 창의 합과 원래 마지막 누적값은 반올림 규모에서 다를 수 있다.

    근거
    ----
    S3 계획 Task13; S3 판정 R36(창별 발생); window_expectations(차분 규약).
    """
    c = np.asarray(cumulative, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != N_WINDOWS:
        raise ReportInputError(f'the cumulative expected onsets must have the shape (n, {N_WINDOWS}), not {c.shape}')
    if not np.isfinite(c).all():
        raise ReportInputError('the cumulative expected onsets hold a NaN or an infinite value')
    # (REP3-4): 차분의 작은 음수만 절단하고 큰 감소는 누적 입력 오류로 거부한다.
    diff = np.diff(c, axis=1, prepend=0.0)
    if (diff < -CUMULATIVE_TOLERANCE * np.maximum(1.0, np.abs(c))).any():
        raise ReportInputError('the cumulative expected onsets decrease with the horizon: they are not cumulative')
    return np.where(diff < 0.0, 0.0, diff)


# ---- sections made from the recorded dicts ----------------------------------------------------------------------------------------------
def mc_section(mc: dict, verdict_mc: dict) -> dict:
    """Criterion (iv), the simulation against the closed form, with the flat rows beside the agreement share.

    ``mc`` is ``results.mc`` of GATE.json (``gate_s3.mc_agreement``) and ``verdict_mc`` its ``verdict.mc`` entry (pass, share, min_share).
    Ruling R28: a row whose Monte Carlo sd is 0 (a flat row: all the simulated values are equal) agrees only on exact equality, so these
    rows can disagree for a reason that has nothing to do with the size of the simulation error. The section carries ``n_sd_zero`` and
    ``n_sd_zero_disagree`` beside ``share``, the disagreements that come from rows with a spread (``n_disagree_with_spread``) and a
    ``statement`` that says where the disagreements come from. ReportInputError for violations of the implemented count inequalities, which do not exhaust joint count consistency.

    수식
    ----
    (REP3-5)  disagree_rows = rows−agree_rows; spread_disagree_rows = disagree_rows−flat_disagree_rows

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      rows, agree_rows, flat_rows, flat_disagree_rows | 전체·일치·평탄·평탄 불일치 행 | 행 | int 변환 후 코드의 부등식만 검사 | n_rows,n_agree,flat,flat_disagree
      disagree_rows, spread_disagree_rows | 전체 불일치·분산 있는 불일치 수 | 행 | 입력 기록의 일관성은 추가 계약 | n_disagree,n_disagree_with_spread

    해석
    ----
    S3 R28의 평탄 행 사례를 전체 불일치에서 분리하되 share·pass·min_share는 저장된 관문에서 그대로 복사한다. 지역 검사는 일부 개수 부등식에 한정되어 모든 카운터 조합의 실현 가능성을 보증하지 않는다.

    근거
    ----
    S3 판정 R28(평탄 행), R27(관문 보존); S3 계획 P7(iv).
    """
    n_rows, n_agree = int(mc['n_rows']), int(mc['n_agree'])
    flat, flat_disagree = int(mc['n_sd_zero']), int(mc['n_sd_zero_disagree'])
    n_disagree = n_rows - n_agree
    if not (n_rows > 0 and 0 <= n_agree <= n_rows and 0 <= flat <= n_rows and 0 <= flat_disagree <= min(flat, n_disagree)):
        raise ReportInputError(f'criterion (iv): the counts do not belong together (rows {n_rows}, agree {n_agree}, flat {flat}, flat and '
                               f'disagree {flat_disagree})')
    if n_disagree == 0:
        statement = f'every row agrees ({flat} rows are flat: MC sd 0, which agree only on exact equality, ruling R28)'
    else:
        statement = (f'{flat_disagree} of the {n_disagree} rows that disagree are flat rows (MC sd 0: a flat row agrees only on exact '
                     f'equality, ruling R28; {flat} rows are flat); the other {n_disagree - flat_disagree} have a spread')
    return {'share': float(verdict_mc['share']), 'min_share': float(verdict_mc['min_share']), 'pass': bool(verdict_mc['pass']),
            'S': int(mc['S']), 'worst_z': None if mc.get('worst_z') is None else float(mc['worst_z']),
            'n_rows': n_rows, 'n_agree': n_agree, 'n_disagree': n_disagree, 'n_sd_zero': flat, 'n_sd_zero_disagree': flat_disagree,
            'n_disagree_with_spread': n_disagree - flat_disagree, 'statement': statement}


MC_RATE_BINS = 5                                            # quintiles of the expected onsets of a row (first 60 s)
MC_RATE_VARIABLE = 'expected_onsets of the first 60 s (B0 closed form, compose/b0_dev.npz)'
MC_RATE_NOTE = ('the rows cut at the quintiles of their expected onsets in the first 60 s; each bin is judged by the rule of criterion '
                '(iv) with the same S. Reported to show where the disagreements sit (the plan note for G-S3c: a 3-standard-error rule '
                'whose error is estimated from the draws under-agrees for strongly skewed draws, and the draws of a low-rate row are '
                'skewed); not judged')


def mc_by_rate(mc_mean, mc_sd, S: int, analytic, rate, n_bins: int = MC_RATE_BINS) -> dict:
    """Criterion (iv) by the rate of the row: where do the disagreements sit? The rows are cut into ``n_bins`` bins of ``rate`` (the
    expected onsets of the first 60 s): the quantiles k / n_bins of ``rate`` are the edges and a row goes to ``searchsorted(edges, rate,
    side='right')``, so rows of one rate are never split and a bin that stays empty is left out (the bins are numbered 0, 1, ... in the
    order of ``rate``). ``gate_s3.mc_agreement`` judges every bin by itself with the rule and the S of the gate; its counts add up to
    those of the whole. Returns ``{'n_bins', 'bins': [{'bin', 'rate_min', 'rate_max', 'n_rows', 'n_agree', 'share', 'n_sd_zero',
    'n_sd_zero_disagree', 'worst_z'}, ...]}``. ReportInputError for vectors of different lengths (or not 1-d, or empty), a rate that is
    not finite and ``n_bins`` below 1.

    수식
    ----
    (REP3-6)  edge_cut = Quantile(rate, cut/requested_bins), cut=1..requested_bins−1
             bin_row = searchsorted(edge,rate_row,side='right')
             agree_row = [abs(mc_mean_row−analytic_row)<3·mc_sd_row/√S] (mc_sd_row>0)
             agree_row = [mc_mean_row=analytic_row] (mc_sd_row=0); bin_share=mean(agree_row in bin)

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      rate, edge, cut, requested_bins, bin_row | 기대 발생 수·경계·절단 위치·요청 빈 수·배정 | 개; — | 유한 rate, 빈 수 양의 정수 검사, 0 기반 배정 | rate,edges,n_bins,which
      mc_mean, mc_sd, analytic | 모의 평균·표준편차·해석 기대값 | 골드 | 일치 검사는 gate_s3에 위임 | mc_mean,mc_sd,analytic
      S, agree_row, bin_share | 같은 모의 수·일치 표시·빈별 비율 | 회; — | S 양의 정수는 gate_s3 검사 | S,r
      Quantile | numpy 기본 선형 분위 연산 | 입력 단위 | 가중치 없이 모든 행 사용 | np.quantile

    해석
    ----
    S3 Task13의 발생 수준별 보고는 가중 분위가 아닌 행 분위로 빈을 나눈다. 동률을 같은 빈에 두고 빈 빈을 버린 뒤 반환 빈 번호를 영부터 다시 붙인다. 각 부분집합은 동일한 모의 수와 S3 P7(iv)·R28 규칙으로 진단하며 관문을 다시 판정하지 않는다.

    근거
    ----
    S3 계획 P7(iv), Task13·G-S3c(발생 수준 진단); S3 판정 R28(평탄 행).
    """
    mc_mean, mc_sd, analytic, rate = (np.asarray(a, dtype=np.float64) for a in (mc_mean, mc_sd, analytic, rate))
    if not (mc_mean.shape == mc_sd.shape == analytic.shape == rate.shape and mc_mean.ndim == 1 and mc_mean.size):
        raise ReportInputError('the agreement by rate needs non-empty vectors of one length: the means, the sds, the closed forms and the '
                               f'rates (shapes {mc_mean.shape}, {mc_sd.shape}, {analytic.shape}, {rate.shape})')
    if not np.isfinite(rate).all():
        raise ReportInputError('the expected onsets of the agreement by rate hold a NaN or an infinite value')
    if int(n_bins) != n_bins or n_bins < 1:
        raise ReportInputError(f'the agreement by rate needs a whole number of bins of at least 1, not {n_bins}')
    # (REP3-6): 행 분위 경계의 동률은 오른쪽 삽입으로 묶고 빈 빈은 보고에서 뺀다.
    edges = np.quantile(rate, np.arange(1, int(n_bins)) / int(n_bins))
    which = np.searchsorted(edges, rate, side='right')
    bins = []
    for k in np.unique(which):
        pick = which == k
        r = gate_s3.mc_agreement(mc_mean[pick], mc_sd[pick], S, analytic[pick])
        bins.append({'bin': len(bins), 'rate_min': float(rate[pick].min()), 'rate_max': float(rate[pick].max()), 'n_rows': r['n_rows'],
                     'n_agree': r['n_agree'], 'share': r['share'], 'n_sd_zero': r['n_sd_zero'],
                     'n_sd_zero_disagree': r['n_sd_zero_disagree'], 'worst_z': r['worst_z']})
    return {'n_bins': len(bins), 'bins': bins}


GROUPS = flowprice.GROUPS                                   # the twelve objective groups of D7, in the order of PRICES.json
RELATIVE_MINUTES = list(range(-4, 3))                       # r = -4 .. +2: the seven beta of a group
# every group has these; the flow fields are None where the flow fit is not estimable
D7_FIELDS = ('beta', 'beta_ci', 'price_flow', 'pretrend', 'pretrend_ci', 'adopted', 'fixed_immediate', 'used', 'value', 'immediate',
             'immediate_ci', 'n_events', 'n_excluded')
D7_OPTIONAL = ('price_flow_ci', 'n_event_matches', 'n_obj_unwindowed', 'obj_unwindowed_share')
D7_TOP = ('n_matches', 'n_window_rows', 'n_dropped_last', 'n_unknown_objectives', 'kill_gold_coef', 'kill_gold_coef_ci', 'B', 'seed')


def _used_because(e: dict) -> str:
    if e['fixed_immediate']:
        return 'fixed immediate group: always the immediate value'
    if e['used'] == 'flow':
        return 'flow price adopted: the interval of the pretrend contains 0'
    if e['price_flow'] is None or e['pretrend_ci'] is None:
        return 'the flow fit is not estimable: the immediate value'
    return 'the interval of the pretrend excludes 0: the immediate value'


def d7_section(prices: dict) -> dict:
    """Item 1, D7: for each of the twelve groups the seven beta of the event study (relative minutes -4 .. +2, beta_{-1} = 0) with their
    95% match-bootstrap intervals, the flow price, the pretrend ζ, whether the flow price was adopted, the immediate value with its
    interval, the value the run uses and why, and the share of the group's events that were left out because their window is not complete
    (``n_excluded / (n_events + n_excluded)``). ``prices`` is the PRICES.json of the s3a run. ReportInputError names the group and the
    field that is missing or malformed. The flow fields may be None for a group whose flow fit is not estimable.

    수식
    ----
    (REP3-7)  excluded_share = int(excluded_events)/(int(complete_events)+int(excluded_events)) (분모≠0), 아니면 None

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      complete_events, excluded_events, excluded_share | 완전 창·제외 사건 수와 제외 비율 | 사건; — | 개수 부호는 지역 미검사, 유효 기록 계약 | n_events,n_excluded,excluded_event_share
      ζ, r | 복사하는 앞 추세와 사건 상대 분 | 골드; 분 | 상대 분 −4..2, 추정 불가 None | pretrend,relative_minutes

    해석
    ----
    설계서 D7의 창 완전성 손실을 집계하고 기존 가격과 채택 여부는 재추정하지 않는다. 필드 존재·선택 길이·사용 가격 종류만 확인하며 개수 부호나 구간 정합성을 모두 검증하지 않는다. 추정 불가 필드와 없는 선택 필드는 None으로 유지한다.

    근거
    ----
    설계서 §8.1(D7); S3 계획 Task13; flowprice.price_table(가격 원천).
    """
    if not isinstance(prices, dict):
        raise ReportInputError('PRICES.json must be a JSON object')
    groups = {}
    for name in GROUPS:
        e = prices.get(name)
        if not isinstance(e, dict):
            raise ReportInputError(f'PRICES.json has no entry for the group {name}')
        missing = [f for f in D7_FIELDS if f not in e]
        if missing:
            raise ReportInputError(f'PRICES.json: the group {name} has no field {missing[0]}')
        for field in ('beta', 'beta_ci'):
            if e[field] is not None and len(e[field]) != len(RELATIVE_MINUTES):
                raise ReportInputError(f'PRICES.json: {field} of the group {name} must have {len(RELATIVE_MINUTES)} entries, not '
                                       f'{len(e[field])}')
        if e['used'] not in ('flow', 'immediate'):
            raise ReportInputError(f'PRICES.json: used of the group {name} must be flow or immediate, not {e["used"]!r}')
        total = int(e['n_events']) + int(e['n_excluded'])
        groups[name] = {**{f: e[f] for f in D7_FIELDS}, **{f: e.get(f) for f in D7_OPTIONAL},
                        'excluded_event_share': int(e['n_excluded']) / total if total else None, 'used_because': _used_because(e)}
    return {'relative_minutes': list(RELATIVE_MINUTES), 'group_order': list(GROUPS), 'groups': groups,
            'used_flow': [g for g in GROUPS if groups[g]['used'] == 'flow'],
            'used_immediate': [g for g in GROUPS if groups[g]['used'] == 'immediate'], **{k: prices.get(k) for k in D7_TOP}}


def convergence_section(b0_summary: dict, fits: dict) -> dict:
    """B4: the optimiser of every cell of every B0 head, from ``fit_result.b0.summary`` (``b0_summary``: ``{'masked': {head: summary},
    'open': {...}}``, each summary with one ``converged`` flag and one ``n_iter`` per cell) and the fit record of ``b0_heads.json``
    (``fits``: the ``max_iter`` of the Poisson and of the logistic fits). A head whose summary has ``n_events`` is a Poisson head. A cell
    ``hit_limit`` when its ``n_iter`` reached the iteration limit of its family; it did not converge (ruling R20) when scikit-learn raised
    a ConvergenceWarning or the limit was reached, so the cells at the limit are a part of the cells that did not converge. Returns the
    limits, every head with its cells, the totals and the names of the cells at the limit (``masked.M3Kills#0``: head and cell).

    수식
    ----
    (REP3-8)  hit_limit_cell = [int(iterations_cell)≥family_limit]
             not_converged = Σ_cell [¬bool(recorded_converged_cell)]; hit_count=Σ_cell hit_limit_cell

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      cell, iterations_cell, family_limit | 출력 칸·반복 수·족의 한계 | —; 회 | 0 기반; 한계는 Python int/not bool 검사, 양수 가정 | i,n,limits
      recorded_converged_cell, hit_limit_cell, not_converged, hit_count | 저장 수렴 표시·한계 표시·집계 | —; 칸 | bool/int 변환, upstream 수렴 계약 | c,cells,n_not_converged,n_hit_limit

    해석
    ----
    S3 R20의 적합 기록을 재현하여 한계에 닿은 칸과 수렴하지 않은 칸을 별도로 센다. 한계 칸이 비수렴 칸의 부분집합이라는 성질은 상류 적합의 계약이며 이 보고 함수가 강제하지 않는다. 반복 수·수렴 표시는 변환해서 복사하며 해당 목록들의 부합 여부를 확인한다.

    근거
    ----
    S3 판정 R20(수렴 기록); S3 계획 Task13(B4 보고).
    """
    limits = {}
    for family in ('poisson', 'logistic'):
        limit = (fits.get(family) or {}).get('max_iter') if isinstance(fits, dict) else None
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ReportInputError(f'the fit record of B0 has no whole max_iter for the {family} fits: the iteration limit is not known')
        limits[family] = limit
    heads, at_limit = {}, []
    for kind in ('masked', 'open'):
        for name, summary in b0_summary.get(kind, {}).items():
            label = f'{kind}.{name}'
            converged, n_iter = summary.get('converged'), summary.get('n_iter')
            if not (isinstance(converged, list) and isinstance(n_iter, list) and converged and len(converged) == len(n_iter)):
                raise ReportInputError(f'the B0 head {label} has no matching lists of converged flags and iteration counts')
            family = 'poisson' if 'n_events' in summary else 'logistic'
            cells = [{'cell': i, 'converged': bool(c), 'n_iter': int(n), 'hit_limit': int(n) >= limits[family]}
                     for i, (c, n) in enumerate(zip(converged, n_iter))]
            at_limit += [f'{label}#{c["cell"]}' for c in cells if c['hit_limit']]
            heads[label] = {'family': family, 'limit': limits[family], 'n_cells': len(cells),
                            'n_not_converged': sum(not c['converged'] for c in cells), 'n_hit_limit': sum(c['hit_limit'] for c in cells),
                            'cells': cells}
    return {'limits': limits, 'heads': heads, 'n_cells': sum(h['n_cells'] for h in heads.values()),
            'n_not_converged': sum(h['n_not_converged'] for h in heads.values()),
            'n_hit_limit': sum(h['n_hit_limit'] for h in heads.values()), 'cells_at_limit': at_limit}


ACCOUNTING_GOLD_TOLERANCE = 1e-6                    # ruling R48: the gold half of P7(iii) is judged to float rounding (runs give ~1.5e-8)
ACCOUNTING_RULE = ('R48 reading of P7(iii): counts exact (max_abs_error == 0) and gold within float rounding (<= 1e-6 gold); GATE.json '
                   'keeps its verdict')


def accounting_value_check(max_abs_value_error) -> dict:
    """The gold half of criterion (iii), P7: "킬 골드와 오브젝트 가치 모두". Ruling R48 reads the criterion at two levels: the counts are exact
    (``max_abs_error == 0``, the verdict that GATE.json holds) and the gold is within float rounding (``max_abs_value_error`` at most
    ``ACCOUNTING_GOLD_TOLERANCE`` = 1e-6 gold). This is the second level, shown beside the verdict: ``{'max_abs_value_error' (as the
    accounting record has it), 'tolerance_gold', 'pass', 'rule'}``. It never changes or recomputes ``gate_pass``. A NaN does not pass.
    ReportInputError for bool, non-numeric or negative values; NaN and positive infinity pass the input check and yield a false comparison.

    수식
    ----
    (REP3-9)  gold_ok = [max_gold_error≤10^−6]

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      max_gold_error, gold_ok | 기록된 최대 절대 골드 오차와 표시 | 골드; — | bool·비수치·음수 거부, NaN/+∞는 통과하지 않음 | max_abs_value_error,pass

    해석
    ----
    S3 R48의 골드 수준 검사를 원래 관문 옆에 표시하고 관문 verdict는 바꾸지 않는다. NaN과 양의 무한대는 지역 입력 거부 대상이 아니지만 비교 결과는 False이다.

    근거
    ----
    S3 판정 R48(골드 회계), R27(관문 보존); S3 계획 P7(iii).
    """
    x = max_abs_value_error
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)) or x < 0:
        raise ReportInputError(f'max_abs_value_error of ACCOUNTING.json must be a non-negative number, not {x!r}')
    return {'max_abs_value_error': x, 'tolerance_gold': ACCOUNTING_GOLD_TOLERANCE, 'pass': bool(x <= ACCOUNTING_GOLD_TOLERANCE),
            'rule': ACCOUNTING_RULE}


LINE_FIELDS = ('oe', 'oe_ci', 'slope', 'slope_ci', 'intercept', 'intercept_ci', 'n_rows', 'n_matches')       # of a line of the calibration
GATE_COMPARED = LINE_FIELDS                                             # a window line is compared with the gate's occurrence result in all
GATE_RTOL = 1e-12                                                       # the gate and the report compute the same sums: equal to rounding


def compare_with_gate(line: dict, gate_occurrence: dict) -> dict:
    """Whether the first window line (0-60 s of B0) is the occurrence result that GATE.json holds. Both are the same function on the
    same rows, weights, B and seed, so agreement is a provenance and numerical-reproducibility check; a difference requires
    investigation and does not identify an exclusive file-mismatch cause. All fields, including counts, use allclose with equal_nan. The numbers compared are ``GATE_COMPARED``: the
    observed / expected ratio, the slope and the intercept with their intervals, and the counts of rows and matches. Returns
    ``{'equal': bool, 'differing': [names of the numbers that differ]}``.

    수식
    ----
    (REP3-10)  field_equal = allclose(line_field, gate_field, rtol=10^−12, atol=10^−12, equal_nan=True)
             equal = all_{field∈LINE_FIELDS} field_equal
             유한 원소의 조건: abs(line_value−gate_value)≤10^−12+10^−12·abs(gate_value)

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      line_field, gate_field, line_value, gate_value | 보고 창과 저장 관문의 필드·원소 | 해당 지표 단위 | float64 변환, 행·경기 수도 같은 허용차 | line[k],gate_occurrence[k]
      LINE_FIELDS, field_equal, equal | 비교 필드와 결과 | — | oe,slope,intercept·각 구간·n_rows,n_matches; None 가능 | GATE_COMPARED,differing,equal

    해석
    ----
    S3 R81의 수치 재현 대조는 개수에도 같은 allclose 규칙을 사용한다. 창이 미계산이면 False를 먼저 반환하고 양쪽에 있는 B 또는 seed가 다르면 비교 불가인 None이다. 한쪽에서 메타데이터가 빠지면 그 메타데이터 대조를 건너뛴다. 불일치는 출처나 수치 재현성을 조사할 신호이며 특정 파일 불일치 원인만을 증명하지 않는다.

    근거
    ----
    S3 계획 Task13(창 대조); S3 판정 R81(비교 해석).
    """
    if not line.get('computed'):
        return {'equal': False, 'differing': ['computed']}
    if any(k in line and k in gate_occurrence and line[k] != gate_occurrence[k] for k in ('B', 'seed')):
        return {'equal': None, 'differing': [],
                'reason': 'the line was made with another B or seed than the gate: the two are not comparable'}
    differing = [k for k in GATE_COMPARED
                 if not np.allclose(np.asarray(line[k], dtype=np.float64), np.asarray(gate_occurrence[k], dtype=np.float64),
                                    rtol=GATE_RTOL, atol=GATE_RTOL, equal_nan=True)]
    return {'equal': not differing, 'differing': differing}


def sensitivity_without_smoke(match, expected_onsets, observed, expected_value, realized, listed, *, B: int = BOOT_B,
                              seed: int = SEED) -> dict:
    """Criteria (i) and (ii) of the gate on the DEV rows whose match is not in ``listed`` (the DEV matches of a smoke run: the B2-a promise
    of the G-S3b gate, that the gate is looked at again without the matches the smoke run has seen).

    The arrays hold one entry per DEV frame row: the match number, the expected and the observed onsets of the first 60 s, the expected
    V_start (closed form) and the realized one. The rows of the listed matches are dropped, the weights are 1 / n_u over the rows that are
    kept (``features.match_weights``), and ``gate_s3.occurrence_calibration`` and ``gate_s3.value_calibration`` run on them with B
    replicates from ``seed`` (the gate's own B and seed by default). ``in_band`` flags say whether each interval lies in the gate's band;
    nothing here is a pass or fail of the gate, which stays the verdict of GATE.json. If no row is left the result is ``{'computed':
    False, 'reason': ...}`` with the counts.

    수식
    ----
    (REP3-11)  keep_row = [match_row∉unique(int64(listed))]; weight_row=1/n_u (남긴 경기 u의 행)
             kept_rows=Σ_row keep_row; kept_matches=|unique(match[keep])|

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      match_row, listed, keep_row | 저장 경기 행·제외 명단·유지 표시 | — | 경기 행은 정수 dtype 검사, 부호는 지역 미검사 | match,listed,keep
      u, n_u, weight_row | 경기·유지한 경기의 행 수·가중치 | —; 행; — | u는 경기 의미, 저장 행 0 기반 | m,w,features.match_weights
      kept_rows, kept_matches, B, seed | 유지 수와 부트스트랩 설정 | 행; 경기; 회; — | 기본 관문 설정, 대체 인수 허용 | counts,B,seed

    해석
    ----
    S3 G-S3b의 약속에 따라 명단 경기를 통째로 빼고 남긴 행으로 경기 가중치를 다시 만든다. 발생·가치의 약한 선형 진단과 그 띠를 보고하며 원래 관문 verdict는 유지한다. 남은 행이 없으면 하류 검사를 하지 않고 computed=False이며 하류 ValueError도 미계산 사유로 기록한다. 이 제외는 관측 표본 민감도이고 분할 재설계가 아니다.

    근거
    ----
    S3 계획 P7; G-S3b B2-a(소규모 점검 경기 제외); S3 판정 R27(관문 보존).
    """
    match = np.asarray(match)
    if match.dtype.kind not in 'iu':
        raise ReportInputError(f'the match numbers must be integers, not {match.dtype}')
    columns = [np.asarray(a, dtype=np.float64) for a in (expected_onsets, observed, expected_value, realized)]
    if any(c.shape != match.shape or c.ndim != 1 for c in columns):
        raise ReportInputError('the DEV arrays of the sensitivity must be vectors of one length')
    e_on, o_on, e_val, r_val = columns
    listed = np.unique(np.asarray(list(listed), dtype=np.int64))
    drop = np.isin(match, listed)
    keep = ~drop
    counts = {'n_listed_dev_matches': int(len(listed)), 'n_listed_in_this_run': int(np.isin(listed, match).sum()),
              'n_rows_all': int(len(match)), 'n_matches_all': int(len(np.unique(match))), 'n_rows_dropped': int(drop.sum()),
              'n_matches_dropped': int(len(np.unique(match[drop]))), 'n_rows_kept': int(keep.sum()),
              'n_matches_kept': int(len(np.unique(match[keep]))), 'B': B, 'seed': seed}
    if not keep.any():
        return {'computed': False, 'reason': 'no DEV row is left after the listed matches are left out', **counts}
    m, w = match[keep], features.match_weights(match[keep])
    try:
        occurrence = gate_s3.occurrence_calibration(e_on[keep], o_on[keep], m, w, B=B, seed=seed)
        value = gate_s3.value_calibration(e_val[keep], r_val[keep], m, w, B=B, seed=seed)
    except ValueError as exc:
        return {'computed': False, 'reason': str(exc), **counts}
    judged = gate_s3.verdict({'occurrence': occurrence, 'value': value})
    return {'computed': True, **counts,
            'occurrence': {**occurrence, 'oe_in_band': judged['occurrence']['oe_in_band'],
                           'slope_in_band': judged['occurrence']['slope_in_band']},
            'value': {**value, 'slope_in_band': judged['value']['slope_in_band'], 'bias_in_band': judged['value']['bias_in_band']},
            'weights': '1 / n_m over the rows that are kept'}


def expected_slope_if_calibrated(mc_mean, mc_sd, S: int, w):
    """A plug-in attenuation reference for the slope on a Monte Carlo mean. ``mc_mean`` (n,) is the mean of ``S`` simulated values per
    row and ``mc_sd`` (n,) their standard deviation. Interpreting ``mc_sd ** 2 / S`` as mean-error variance requires appropriate
    independent draws. The errors-in-variables reference assumes zero-mean MC error orthogonal to the latent mean and realized residual;
    finite weighting, sample centering and cross-row covariance prevent an exact finite-sample slope guarantee. The computed reference
    is clipped below at zero and returns None when the weighted variance is not positive. B1 has no closed form and only S = 200
    simulations; this reference is a weak linear diagnostic under those assumptions, not full conditional calibration.

    수식
    ----
    (REP3-12)  weight_total=Σ_row weight_row; mean_mc=Σ_row weight_row·mc_mean_row/weight_total
             variance_mc=Σ_row weight_row·(mc_mean_row−mean_mc)^2/weight_total
             mean_se2=Σ_row weight_row·mc_sd_row^2/S/weight_total
             slope_reference=max(0,1−mean_se2/variance_mc) (variance_mc>0), 아니면 None

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, mc_mean, mc_sd | 행·모의 평균·모의 표준편차 | —; 골드 | 호환 모양·유한성은 가정, 여기서 미검사 | x,sd
      weight_row, weight_total, mean_mc | 가중치·합·가중 평균 | —; 골드 | 비음수·양의 합 가정, 여기서 미검사 | w,total,mean
      variance_mc, mean_se2 | 가중 모의 평균 분산과 평균 오차분산 추정 | 골드² | 유의미한 유한 입력 가정 | variance,sd**2/S
      S, slope_reference | 행별 모의 수와 감쇠 기준 | 회; — | S 양의 정수 검사, None 가능 | S,반환값

    해석
    ----
    S3 R81의 기준은 MC 오차 평균이 영이고 잠재 평균 및 실현 잔차와 직교할 때의 errors-in-variables plug-in 근사이다. 모의 평균의 오차분산을 추정하려면 적절한 독립 모의 계약이 필요하며 상관된 추출은 오차분산을 바꾼다. 유한 가중 표본의 중심화와 행 사이 공분산 때문에 정확한 표본 기울기를 보장하지 않는다. 코드가 직접 검사하는 것은 모의 수이며 나머지 배열·가중치 조건은 가정이다. 분산이 양수가 아니면 None이고 음의 기준은 영으로 자른다.

    근거
    ----
    S3 판정 R81(감쇠 기준 한계); S3 계획 P6(모의 보고).
    """
    if isinstance(S, bool) or not isinstance(S, (int, np.integer)) or S < 1:
        raise ReportInputError(f'the number of simulations S must be a positive whole number, not {S!r}')
    x, sd, w = (np.asarray(a, dtype=np.float64) for a in (mc_mean, mc_sd, w))
    total = math.fsum(w)
    # (REP3-12): 가중 중심 분산과 평균 MC 오차분산으로 plug-in 기준을 구성한다.
    mean = math.fsum(w * x) / total
    variance = math.fsum(w * (x - mean) ** 2) / total
    if not variance > 0:
        return None
    return max(0.0, 1.0 - (math.fsum(w * sd ** 2 / S) / total) / variance)


def value_summary(expected, realized, match, w, *, B: int = BOOT_B, seed: int = SEED, judged: bool = True) -> dict:
    """``gate_s3.value_calibration`` of the realized values on the expected ones, without the ten bins: the slope and the mean error with
    their match-bootstrap 95% intervals and, with ``judged`` (the default, for an expectation in closed form), whether they lie in the
    gate's bands (report only: ``slope_in_band`` and ``bias_in_band``), the weighted means of both, the counts. ``judged=False`` leaves the
    two flags out, for an expectation that is a Monte Carlo mean (``mc_value_summary``): the bands were registered for the closed form.

    수식
    ----
    (REP3-13)  mean_expected=Σ_row weight_row·expected_row/weight_total; mean_realized=Σ_row weight_row·realized_row/weight_total
             slope=Cov_w(expected,realized)/Var_w(expected); bias=mean_realized−mean_expected
             interval(stat)=[Pct2.5(stat*),Pct97.5(stat*)], 경기 재표본 B개

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, expected, realized, weight_row, weight_total | 관측 행·기대·실현·가중치·합 | —; 골드; — | gate_s3/nll 검증에 위임 | expected,realized,w
      mean_expected, mean_realized, slope, bias | 가중 평균·기울기·평균 오차 | 골드; —; 골드 | 양의 기대 분산 필요 | out,r
      stat*, interval, Pct, B, seed | 경기 재표본 통계·구간·백분위·횟수·시드 | 지표 단위; 회; — | 유효 경기 복제, 기본 B=2000 | gate_s3.value_calibration
      judged | 사전 등록 띠 표시 선택 | — | False면 두 band 플래그 생략 | judged
      Cov_w, Var_w | 가중 중심 공분산과 분산 | 골드² | 가중치 합으로 나눈 중심 곱·제곱합 | gate_s3._line
      stat | 원 표본의 기울기 또는 평균 오차 | 해당 지표 단위 | 유효 기대 분산 계약 | r

    경계: gate_s3._line은 중심 분산이 중심 제곱평균의 10^−12배 이하면 기울기를 NaN으로 두며 전 표본의 NaN 기울기는 ValueError이다.
    복제 하나라도 해당 통계가 NaN이면 그 통계 구간 양끝이 NaN이며 Pct는 numpy의 선형 백분위이다.

    해석
    ----
    S3 P7의 가치 진단은 기울기와 평균 오차의 경기 부트스트랩 구간을 보고한다. 이는 약한 선형 보정 진단이며 모든 입력에서 참 조건부 기대값과 같은 완전한 보정을 증명하지 않는다. judged가 참일 때만 기울기 띠 [0.8,1.2]와 오차 띠 [−50,50] 골드 안에 구간 전체가 드는지 표시한다. 실제 관문 verdict는 별도로 보존한다.

    근거
    ----
    S3 계획 P7(ii); gate_s3.value_calibration·verdict(구간·띠); S3 판정 R27, R81(해석 한계).
    """
    r = gate_s3.value_calibration(expected, realized, match, w, B=B, seed=seed)
    out = {'slope': r['slope'], 'slope_ci': r['slope_ci'], 'bias': r['bias'], 'bias_ci': r['bias_ci'],
           'mean_expected': nll.weighted_mean(expected, w), 'mean_realized': nll.weighted_mean(realized, w), 'n_rows': r['n_rows'],
           'n_matches': r['n_matches'], 'B': r['B'], 'seed': r['seed']}
    if judged:
        band = gate_s3.verdict({'value': r})['value']
        out.update(slope_in_band=band['slope_in_band'], bias_in_band=band['bias_in_band'])
    return out


MC_READING_NOTE = 'read against the attenuated reference, not against 1'


def mc_value_summary(mc_mean, mc_sd, S: int, realized, match, w, *, B: int = BOOT_B, seed: int = SEED) -> dict:
    """The calibration of the realized values on a Monte Carlo mean of ``S`` draws per row (B1 and the simulated mean of B0): the numbers
    of ``value_summary`` and no band flag. The error of the mean attenuates the slope (errors-in-variables), so the band [0.8, 1.2] of the
    gate, registered for the closed form, is not the reading of this line; ``expected_slope_if_calibrated`` is instead a plug-in
    attenuation reference under its stated error assumptions, without an exact finite-sample slope guarantee. ``slope_ratio_to_expected`` is the slope divided by that reference where the reference is
    above 0 (else None): a ratio near 1 is at most compatible with weak linear calibration under those assumptions, not proof of full
    conditional calibration. ``note`` says how to read the line.

    수식
    ----
    (REP3-14)  slope_ratio=slope/slope_reference (slope_reference 있음 ∧ slope_reference>0), 아니면 None

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      slope, slope_reference, slope_ratio | 모의 평균의 회귀 기울기·감쇠 기준·비 | — | 기준은 (REP3-12), 비 None 가능 | out['slope'],reference,ratio
      S, B, seed | 모의 수·부트스트랩 수·시드 | 회; 회; — | 하류 함수의 계약 | S,B,seed

    해석
    ----
    S3 P6에 따라 모의 평균에는 닫힌 꼴용 띠 플래그를 붙이지 않는다. 비가 일에 가깝다는 것은 S3 R81의 감쇠 가정 아래 약한 선형 진단과 양립할 수 있다는 뜻이며 완전한 조건부 보정의 증거가 아니다. 출력 note는 기존 읽기 지침을 그대로 운반하며 정확한 표본 기울기 보증으로 읽지 않는다.

    근거
    ----
    S3 계획 P6; S3 판정 R81(plug-in 비 해석).
    """
    out = value_summary(mc_mean, realized, match, w, B=B, seed=seed, judged=False)
    reference = expected_slope_if_calibrated(mc_mean, mc_sd, S, w)
    # (REP3-14): 영·미정 기준으로 나누지 않으며 닫힌 꼴용 띠를 재사용하지 않는다.
    ratio = out['slope'] / reference if reference is not None and reference > 0 else None
    return {**out, 'expected_slope_if_calibrated': reference, 'slope_ratio_to_expected': ratio, 'note': MC_READING_NOTE}


def b1_positions(b0: dict, b1: dict) -> np.ndarray:
    """The position in the B0 table of each row of B1 (the B1 sample is a part of the DEV rows, found by the frame row). ReportInputError
    when a row of B1 is not a row of B0 or lies in another match: the two compose files do not belong together.

    수식
    ----
    (REP3-15)  order=stable_argsort(frame0); insertion=searchsorted(frame0[order],frame1,side='left')
             position=order[min(insertion,len(order)−1)]
             요구: array_equal(frame0[position],frame1) ∧ array_equal(match0[position],match1)

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      frame0, frame1, match0, match1 | B0·B1의 프레임 행과 경기 행 | — | 저장 행 0 기반 | b0,b1
      order, insertion, position | 정렬 순열·삽입 위치·원래 행 위치 | — | 0 기반, B0 비어 있지 않음 가정 | order,pos

    해석
    ----
    S3 P6의 B1 표본을 같은 프레임·경기의 B0 행과 맞춰 비교한다. 왼쪽 삽입으로 중복 프레임은 첫 정렬 행을 택하므로 고유 프레임 계약이 필요하다. B0가 비면 IndexError가 날 수 있으며 이 함수가 모든 빈 입력을 ReportInputError로 정규화하지 않는다.

    근거
    ----
    S3 계획 P6; S3 계획 Task13(B1 같은 행 비교).
    """
    order = np.argsort(b0['frame_row'], kind='stable')
    pos = order[np.minimum(np.searchsorted(b0['frame_row'][order], b1['frame_row']), len(order) - 1)]
    if not (np.array_equal(b0['frame_row'][pos], b1['frame_row']) and np.array_equal(b0['match'][pos], b1['match'])):
        raise ReportInputError('the rows of B1 are not rows of the B0 DEV table (frame row and match differ): the two compose files do not '
                               'belong together')
    return pos


def b1_coverage_note(sample: str, rows: int, dev_rows: int, S: int | None = None) -> str:
    """What the reader must know about the rows of B1: all the DEV rows, or only a sample of them (500 matches in the full run), and the
    number of simulations ``S`` of each row when it is known.

    B1의 표본 범위와 알려진 모의 수를 독자에게 전달한다. 전체 DEV라는 분기는 행 수의 같음만 비교하므로 정확한 정렬·고유 행 계약이 필요하다.
    """
    sims = '' if S is None else f', S = {S} simulations per row'
    if rows == dev_rows:
        return f'B1 is composed on all {dev_rows:,} DEV rows ({sample}){sims}'
    return (f'B1 covers only its sample ({sample}): {rows:,} of the {dev_rows:,} DEV rows{sims}; its lines are made on those rows alone '
            'and are not comparable with the lines of B0 on all the DEV rows')


def _head_block(S: int, mc_mean, mc_sd, q19, expected_onsets, y, observed, match, w, B: int, seed: int) -> dict:
    """The composition statistics of one head set on the sampled rows at H = 60 s (column 0 of the arrays): the value of the simulated mean,
    its Monte Carlo error, the CRPS and the PIT of the 19 quantiles, and the occurrence calibration of the first 60 s.

    (REP3-16)의 계산·입력 항을 맡는다.
    """
    occ = gate_s3.occurrence_calibration(expected_onsets, observed, match, w, B=B, seed=seed)
    judged = gate_s3.verdict({'occurrence': occ})['occurrence']
    value = mc_value_summary(mc_mean, mc_sd, S, y, match, w, B=B, seed=seed)          # a Monte Carlo mean: no band flags on this line
    return {'S': int(S), 'mean_simulated_gold': nll.weighted_mean(mc_mean, w), 'mean_realized_gold': nll.weighted_mean(y, w),
            'value_simulated': value,
            'mc_se': {'rms_gold': math.sqrt(math.fsum(w * np.asarray(mc_sd, dtype=np.float64) ** 2 / S) / math.fsum(w)),
                      'expected_slope_if_calibrated': value['expected_slope_if_calibrated']},
            'crps19': crps19(q19, y, w), 'pit': pit_histogram(q19, y, w), 'quantile_hits': quantile_hits(q19, y, w),
            'occurrence_60': {**{k: occ[k] for k in LINE_FIELDS}, 'oe_in_band': judged['oe_in_band'],
                              'slope_in_band': judged['slope_in_band']}}


def b1_composition(b0: dict, b1: dict, realized, observed, *, sample: str, B: int = BOOT_B, seed: int = SEED) -> dict:
    """Item 11, the B1 composition: B1 is composed by simulation only (decision P6: S = 200 on a seeded sample of DEV matches, reported and
    not judged). ``b0`` and ``b1`` are the arrays of ``compose/b0_dev.npz`` and ``b1_dev.npz`` (``frame_row``, ``match``,
    ``expected_onsets``, ``mc_mean``, ``mc_sd``, ``q19``, ``S``; B0 also ``expected``, the closed form), ``realized`` the realized V_start
    at each horizon of the B0 rows and ``observed`` (n, 3) their observed onsets per window; both are aligned with the rows of ``b0``. The
    rows of B1 must be rows of B0 (the same frame row and match). The statistics at H = 60 s are the ones of the gate, for B1 and for B0
    on the same rows, with the weights 1 / n_u over the sampled rows: the calibration of V_start on the simulated mean (and, for B0, on
    the closed form), the Monte Carlo error of the mean and its plug-in attenuation reference under the stated error assumptions, the finite-grid CRPS proxy,
    the PIT, and the occurrence calibration of the first 60 s. ``sample`` says how the rows were chosen (COMPOSE.json).

    수식
    ----
    (REP3-16)  weight_row=1/n_u (B1 표본 경기 u의 행); coverage_share=sampled_rows/dev_rows
             rms_mc_se=√(Σ_row weight_row·mc_sd_row²/S / Σ_row weight_row)
             horizon_s=EVAL_H_MS[0]//1000

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      u, n_u, weight_row | 경기·표본 경기의 행 수·가중치 | —; 행; — | 저장 경기 행 0 기반 | match,w
      sampled_rows, dev_rows, coverage_share | B1·전체 DEV 행 수와 비 | 행; — | dev_rows>0 가정 | len(match),len(realized),share_of_dev_rows
      mc_sd, S, rms_mc_se | 모의 표준편차·수·표준오차 RMS | 골드; 회; 골드 | 독립 모의 해석은 (REP3-12) 계약 | _head_block,mc_se
      horizon_s | 평가 첫 열의 지평 | s | 열 0, H=60 s | C.EVAL_H_MS[0]

    해석
    ----
    S3 P6에 따라 B1과 B0를 같은 표본 행에서 비교하고 경기 가중치를 그 표본으로 다시 정한다. 실제 가치와 발생 수는 첫 지평 열에 맞추며 모의와 닫힌 꼴의 진단은 소유 함수의 식을 따른다. RMS는 행별 모의 표준오차 제곱의 경기 가중 평균에 제곱근을 취한 값이다. 기존 출력 note의 exactly right 표현은 S3 R81의 가정 있는 plug-in 기준으로만 읽어야 한다. 표본 비교는 DEV 전체 성능이나 완전한 조건부 보정을 보증하지 않는다.

    근거
    ----
    S3 계획 P6, Task13; S3 판정 R69(공개 소유), R81(감쇠 해석).
    """
    pos = b1_positions(b0, b1)
    realized, observed = np.asarray(realized, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    y, obs, match = realized[pos], observed[pos, 0], np.asarray(b1['match'])
    w = features.match_weights(match)
    S0, S1 = _scalar(b0['S']), _scalar(b1['S'])

    def block(S: int, h: dict) -> dict:
        return _head_block(S, h['mc_mean'][:, 0], h['mc_sd'][:, 0], h['q19'][:, 0, :], h['expected_onsets'][:, 0], y, obs, match, w, B,
                           seed)
    keys = ('mc_mean', 'mc_sd', 'q19', 'expected_onsets')
    blocks = {'b1': block(S1, b1), 'b0_same_rows': block(S0, {k: b0[k][pos] for k in keys})}
    blocks['b0_same_rows']['value_closed_form'] = value_summary(b0['expected'][pos, 0], y, match, w, B=B, seed=seed)
    note = (f'B1 is composed by simulation only (S = {S1} on the sampled rows, S = {S0} for B0), report only (decision P6). The Monte '
            'Carlo error of a mean of S draws attenuates the slope of the realized value on it; expected_slope_if_calibrated is the slope '
            'that a model which is exactly right would show, so the slope of B1 is to be read against that number and not against 1 '
            '(slope_ratio_to_expected is the slope divided by it); the lines made on a Monte Carlo mean carry no band flag, only the '
            'closed form of B0 does.')
    return {'rows': int(len(match)), 'matches': int(len(np.unique(match))), 'dev_rows': int(len(realized)),
            'share_of_dev_rows': len(match) / len(realized), 'S_b1': S1, 'S_b0': S0, 'sample': sample,
            'horizon_s': C.EVAL_H_MS[0] // 1000, **blocks, 'note': note}


def _scalar(saved) -> int:
    """The first flattened element cast to int; callers expect one whole element (``S`` and ``seed`` of the compose files). Shape and
    integrality are not validated here, and an empty array raises IndexError.

    (REP3-16)의 계산·입력 항을 맡는다.
    """
    return int(np.asarray(saved).reshape(-1)[0])


def findings(doc: dict) -> list:
    """The findings that go to the top of the report (as ``report_s2`` has its ``flags``): things the reader of the report should know
    before the numbers. They do not stop the report and they are not the gate's verdict. Each is ``{'code', 'message'}``; ``doc`` is the
    report as far as it is built (``run``, ``cost``, ``b0_convergence``, ``window_slopes``, ``mc_agreement``, ``inputs``,
    ``sensitivity_without_smoke_dev``, ``accounting``). The codes: SMOKE_RUN, MEMORY_OVER_CAP, MEMORY_NOT_MEASURED,
    B0_CELLS_AT_ITERATION_LIMIT, GATE_RECOMPUTATION_DIFFERS, MC_FLAT_ROW_DISAGREEMENT, WINDOW_NOT_COMPUTED, SENSITIVITY_NOT_COMPUTED
    (a run that should have the line without the matches of its smoke run and has not), B1_ROUNDS_NOT_RECORDED (a head with a round
    field that is still None), ACCOUNTING_GOLD_OVER_TOLERANCE (ruling R48) and OLD_RECORD_LAYOUT.

    저장·계산된 보고 항목의 주의 표시를 모으고 보고는 계속한다. 기존 메시지의 원인 단정이나 오차 초과 표현은 진단 결과의 모든 가능한 원인을 증명하지 않는다.
    """
    out = []

    def add(code: str, message: str) -> None:
        out.append({'code': code, 'message': message})
    if doc['run'].get('smoke'):
        add('SMOKE_RUN', 'this is a smoke run (a seeded subset of the matches): its numbers are not those of the full run')
    within = doc['cost']['memory'].get('within_cap')
    if within is False:
        add('MEMORY_OVER_CAP', 'the sampled peak private memory of the run (peak_private_gib) is above the cap')
    elif within is None:
        add('MEMORY_NOT_MEASURED', 'the peak private memory was not measured (the run had no psutil), so the cap cannot be judged')
    conv = doc['b0_convergence']
    if conv['n_hit_limit'] > 0:
        add('B0_CELLS_AT_ITERATION_LIMIT', f'{conv["n_hit_limit"]} of {conv["n_cells"]} B0 cells reached the iteration limit; B4 keeps the '
                                           'limits and reports them')
    check = doc['window_slopes']['check_against_gate']
    if check['equal'] is False:
        add('GATE_RECOMPUTATION_DIFFERS', 'the 0-60 s window line of B0 differs from the occurrence result of GATE.json in '
                                          f'{", ".join(check["differing"])}: the report was not made from the files the gate was made from')
    mc = doc['mc_agreement']
    if mc['n_sd_zero_disagree'] > 0:
        add('MC_FLAT_ROW_DISAGREEMENT', f'{mc["n_sd_zero_disagree"]} of the {mc["n_sd_zero"]} flat rows (MC sd 0) disagree with the closed '
                                        'form; a flat row agrees only on exact equality (ruling R28)')
    for head in ('b0', 'b1'):
        bad = [w['name'] for w in doc['window_slopes'][head]['windows'] if not w.get('computed')]
        if bad:
            add('WINDOW_NOT_COMPUTED', f'the occurrence line of {head.upper()} could not be computed for {", ".join(bad)}')
    sens = doc['sensitivity_without_smoke_dev']
    if sens.get('requested') and not sens.get('computed'):
        add('SENSITIVITY_NOT_COMPUTED', 'the gate without the smoke DEV matches (the B2-a promise of G-S3b) is not in this report: '
                                        f'{sens.get("reason")}')
    rounds = doc['cost']['b1_rounds_missing']
    if rounds:
        lacking = ', '.join(f'{field} of {len(labels)} head{"" if len(labels) == 1 else "s"}' for field, labels in rounds.items())
        add('B1_ROUNDS_NOT_RECORDED', f'the fit record lacks B1 rounds ({lacking}): the fit stage was resumed after B1 was saved, so a '
                                      'head is the line of PROGRESS.jsonl, or the record is older; they are None in the cost record')
    gold = doc['accounting']['value_check']
    if not gold['pass']:
        add('ACCOUNTING_GOLD_OVER_TOLERANCE', f'the largest gold error of the accounting identity, {gold["max_abs_value_error"]:g}, is '
                                              f'above the tolerance of {gold["tolerance_gold"]:g} gold (ruling R48); the verdict of '
                                              'GATE.json for criterion (iii) is not changed')
    missing = doc['inputs'].get('missing_optional_fields')
    if missing:
        add('OLD_RECORD_LAYOUT', f'the run records lack fields that the runner writes now ({", ".join(missing)}): they are None in the '
                                 'cost record')
    return out


MIN_GAME_ID_DIGITS = 9                             # game ids have 9 or 10 digits; a shorter number cannot be told from an ordinary count


class KeyInReport(RuntimeError):
    """A match key or a game id of the S1 store would be in a file that the report writes. The message tells how many, never which.

    최종 텍스트의 키·게임 번호 검사 실패를 개수만으로 전달하는 예외이다.
    """


def game_ids(keys) -> set:
    """The game ids (the digits after the last underscore of a key, as ``store_build._key_parts`` takes them out) of at least
    ``MIN_GAME_ID_DIGITS`` digits.

    입력 키에서 게임 번호 부분을 추려 문자열 검색의 대상 집합을 만든다. 최소 자릿수 조건에 맞는 부분만 포함한다.
    """
    from c_ept import store_build               # here: the report needs its helper only when it scans
    return {part for key in keys for part in store_build._key_parts(key)[1:] if len(part) >= MIN_GAME_ID_DIGITS}


def assert_clean(ctx, text: str) -> None:
    """RuntimeError when ``text`` names a match: the number of hits is told, never a key or an id. Two checks. The full keys are looked
    for by the scan of the runner (``run_s3.Context.assert_no_keys``, called with ``ctx``; ``ctx`` is a ``run_s3.Context``, or anything
    with a ``keys()`` method that gives the S1 match keys as text). The bare game ids are looked for among the whole runs of digits of
    the text: the same digits inside a longer number are not a hit (a float of 16 digits can hold ten digits of an id by chance; a run
    of digits that is exactly an id cannot be a count of this report).

    키 검색과 독립된 전체 숫자 토큰 검색을 출력 전에 적용한다. 이 검사는 주어진 S1 식별자 목록에 한정된다.
    """
    try:
        run_s3.Context.assert_no_keys(ctx, text)
    except RuntimeError as exc:
        raise KeyInReport(str(exc)) from None
    hits = len(game_ids(ctx.keys()) & set(re.findall(r'\d+', text)))
    if hits:
        raise KeyInReport(f'{hits} game ids occur as whole numbers in the text that was to be written')


LIFETIME_PEAK_LABEL = 'lifetime peak bound (monotone within a call), not a per-stage peak'
SAMPLED_PEAK_NOTE = ('the largest sampled sum (every 0.1 s) of the private memory of the process tree; the 40 GiB cap is judged on it '
                     '(ruling R43)')
COMPOSE_CHUNK_FIELDS = ('chunks_total', 'chunks_loaded', 'chunks_computed', 'rows_computed')     # written by the runner since the B3 round
RUNNER_CALL_FIELDS = ('started', 'finished', 'memory_samples', 'check_calls', 'check_s_mean', 'check_s_longest', 'stages_run',
                      'stages_skipped', 'stopped_by', 'workers', 'resumed', 'smoke', 'code_commit', 'code_dirty')
STAGE_RECORD_FIELDS = ('within_memory_cap', 'e6_check_calls', 'e6_check_s_total', 'e6_check_s_mean', 'started', 'finished')
COMPOSE_SIZE_FIELDS = ('workers', 'blocks', 'chunk_rows', 'S_b0', 'S_b1', 'rows', 'matches', 'b1_rows', 'b1_sample')
COMPOSE_CHECK_FIELDS = ('check_calls', 'check_s_total', 'check_s_mean', 'check_s_longest')
CALL_CHECK_FIELDS = ('call_check_calls', 'call_check_s_total', 'call_check_s_mean', 'call_check_s_longest', 'call_check_budget_s')
CALL_FIELDS = ('head_set', 'method', 'rows', 'block', 'workers', 'n_blocks', 'wall_s', 'seconds_per_block_wall', 'block_compute_s_estimate')
B1_ROUND_FIELDS = ('best_iteration', 'rounds_run', 'stopped_early', 'num_trees')                  # per cell, from the summary of a B1 head
PROGRESS_ROUND_FIELDS = ('best_iteration', 'rounds_run')            # the two that a line of fit/PROGRESS.jsonl holds at its top level


def _peaks_of(rec, missing: set) -> dict:
    """The two peaks of a run record under the names the report uses: ``peak_private_gib`` (the sampled peak, the one memory is judged by)
    and ``lifetime_peak_bound_gib`` (the record's ``peak_private_upper_gib``: the retained maximum across observations of sums of readable
    observed processes' lifetime peaks). Earlier history can be included and terminated or unreadable processes missed, so this is neither
    a complete historical-tree bound nor a per-stage peak (S3 rulings R79, R81 and R82); this helper only copies and renames the record. A record from before the field was written gives None, and the
    field name goes into ``missing``: that is a record that has ``peak_private_gib`` but not the bound. A record with neither (the start
    of a call that was killed before it wrote its end) says nothing about the layout of the runner."""
    rec = rec if isinstance(rec, dict) else {}
    if 'peak_private_gib' in rec and 'peak_private_upper_gib' not in rec:
        missing.add('peak_private_upper_gib')
    return {'peak_private_gib': rec.get('peak_private_gib'), 'lifetime_peak_bound_gib': rec.get('peak_private_upper_gib')}


def _timed(rec, missing: set, **extra) -> dict:
    rec = rec if isinstance(rec, dict) else {}
    return {'wall_s': rec.get('wall_s'), **_peaks_of(rec, missing), **{k: rec.get(k) for k in extra}}


def _b1_head(rec, missing: set) -> dict:
    """The time, the peaks and the rounds of one B1 head. The rounds are those of the ``summary`` of the head's record. A record without
    the summary (a fit stage that was resumed once B1 was saved gives each head as its line of ``fit/PROGRESS.jsonl``) gives
    ``best_iteration`` and ``rounds_run`` from its top level, where the line holds them; ``stopped_early`` and ``num_trees`` are then None.
    ``rounds_source`` says where the rounds came from: ``summary``, ``progress_line`` (at least one of them), or None when the record has
    none."""
    rec = rec if isinstance(rec, dict) else {}
    summary = rec.get('summary') if isinstance(rec.get('summary'), dict) else {}
    rounds, from_line = {}, False
    for field in B1_ROUND_FIELDS:
        value = summary.get(field)
        if value is None and field in PROGRESS_ROUND_FIELDS and rec.get(field) is not None:
            value, from_line = rec[field], True
        rounds[field] = value
    source = 'progress_line' if from_line else ('summary' if any(v is not None for v in rounds.values()) else None)
    return {**_timed(rec, missing, loaded=0), **rounds, 'rounds_source': source}


def _cost_calls(calls, missing: set) -> list:
    """The compose calls: time, blocks and the cost of the E6 checks of each. A call records no peak of its own (None). Runs of the first
    version of the runner wrote the counters of a call as ``check_calls`` ... without the ``call_`` prefix; they are read under the new
    names."""
    out = []
    for call in calls or []:
        counters = {f: call[f] if f in call else call.get(f[len('call_'):]) for f in CALL_CHECK_FIELDS}
        out.append({**{k: call.get(k) for k in CALL_FIELDS}, **counters, 'peak_private_gib': call.get('peak_private_gib'),
                    'lifetime_peak_bound_gib': call.get('peak_private_upper_gib')})
    return out


def cost_section(manifest: dict, ledger: dict, fit: dict, compose: dict) -> tuple:
    """Item 6 of the carries: wall time and peak memory of every stage, model, head and compose call, the rounds of the B1 heads and the
    counters of the E6 checks. ``manifest`` is RUN.json, ``ledger`` STAGES.json, ``fit`` FIT_FACTS.json and ``compose`` COMPOSE.json.
    Returns ``(section, missing_optional)``: ``missing_optional`` lists, sorted, the fields that the runner writes since the G-S3b fix
    round and the record lacks (an older run): they are None in the section and the report goes on.

    Memory is judged by ``peak_private_gib``, the sampled peak (S3 ruling R43). The runner's ``peak_private_upper_gib`` records the
    maximum across observations of sums of readable observed processes' lifetime ``peak_pagefile`` values. It may include earlier
    history and miss terminated or unreadable children, so it is neither a complete historical-tree bound nor a per-stage peak. It is
    reported under the existing name ``lifetime_peak_bound_gib`` with the label ``LIFETIME_PEAK_LABEL`` (S3 rulings R79 and R81).

    수식
    ----
    (REP3-17)  within_cap=[recorded_peak≤recorded_cap] (recorded_peak 있음), 아니면 None
             lifetime_display=recorded peak_private_upper_gib

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      recorded_peak, recorded_cap, within_cap | 저장된 표본 피크·한계·표시 | GiB; — | None 가능, 여기서 재측정 없음 | peak,cap,within_cap
      lifetime_display | 관측한 읽기 가능 프로세스의 수명 피크 합 최댓값 기록 | GiB | 과거 이력 포함 가능, 누락·종료 프로세스 제외 가능 | _peaks_of,lifetime_peak_bound_gib

    해석
    ----
    S3 R43에 따라 저장된 표본 메모리로 표시하고 비용은 기존 기록을 정리할 뿐 측정하지 않는다. 수명 피크는 S3 R79의 관측 가능 범위에 한정되며 완전한 역사적 트리의 상한이나 단계별 피크가 아니다. 저장된 비엄격 비교는 S3 R38의 엄격한 사람 승인 조건과 구별한다. 없는 선택 필드는 None과 누락 목록으로 남긴다.

    근거
    ----
    S3 판정 R43(표본 피크), R38(사람 관문), R79·R81(수명 범위).
    """
    missing: set = set()
    s3 = manifest.get('s3_result') or {}
    cap = manifest.get('memory_cap_gib', run_s3.MEMORY_CAP_GIB)
    peak = s3.get('peak_private_gib')
    stages = {}
    for name in run_s3.STAGES:
        rec = (ledger.get('stages') or {}).get(name) or {}
        stages[name] = _timed(rec, missing, **dict.fromkeys(STAGE_RECORD_FIELDS))
    b0 = fit.get('b0') or {}
    b0_heads = {'setup': _timed(b0.get('setup'), missing)}
    for label, rec in (b0.get('heads') or {}).items():
        b0_heads[label] = [_timed(seg, missing) for seg in rec] if isinstance(rec, list) else _timed(rec, missing)
    b1_heads = {label: _b1_head(rec, missing) for label, rec in ((fit.get('b1') or {}).get('heads') or {}).items()}
    b1_rounds_missing = {f: sorted(label for label, h in b1_heads.items() if h[f] is None) for f in B1_ROUND_FIELDS}
    b1_rounds_missing = {f: labels for f, labels in b1_rounds_missing.items() if labels}          # {} when every head has every field
    models = {name: _timed(fit.get(key), missing, loaded=0) for name, key in
              (('tier', 'tier'), ('b0', 'b0'), ('b1', 'b1'), ('b2', 'b2'), ('dev_scores', 'dev'))}
    for name in COMPOSE_CHECK_FIELDS + COMPOSE_CHUNK_FIELDS + ('chunks_by_head_set',):
        if name not in compose:
            missing.add(name)
    compose_part = {**_timed(compose, missing), **{k: compose.get(k) for k in COMPOSE_SIZE_FIELDS},
                    'e6_checks': {k: compose.get(k) for k in COMPOSE_CHECK_FIELDS},
                    'chunks': {k: compose.get(k) for k in COMPOSE_CHUNK_FIELDS}, 'chunks_by_head_set': compose.get('chunks_by_head_set'),
                    'calls': _cost_calls(compose.get('calls'), missing)}
    calls = [c if isinstance(c, dict) else {} for c in ledger.get('calls') or []]
    runner_calls = [{**_timed(c, missing), **{k: c.get(k) for k in RUNNER_CALL_FIELDS}} for c in calls]
    unfinished = [{'call': i, 'started': c.get('started'), 'stages_requested': c.get('stages_requested')}
                  for i, c in enumerate(calls) if not c.get('finished')]                 # a call that was killed wrote its start only
    section = {
        'memory': {'cap_gib': cap, 'judged_on': 'peak_private_gib', 'peak_private_gib': peak,
                   'within_cap': None if peak is None else bool(peak <= cap),
                   'lifetime_peak_bound_gib': _peaks_of(s3, missing)['lifetime_peak_bound_gib'],
                   'lifetime_peak_bound_label': LIFETIME_PEAK_LABEL,
                   'rule': 'the cap is judged on the sampled peak_private_gib (ruling R43); the lifetime peak bound is reported beside it'},
        'field_labels': {'peak_private_gib': SAMPLED_PEAK_NOTE,
                         'lifetime_peak_bound_gib': f'{LIFETIME_PEAK_LABEL}; the sum over the process tree of each process\'s lifetime '
                                                    'peak_pagefile, recorded by the runner as peak_private_upper_gib'},
        'machine': {k: manifest.get(k) for k in ('cpu', 'gpu', 'ram_gib', 'python', 'numpy')},
        'run': {'wall_s': s3.get('wall_s'), 'calls': s3.get('calls'), 'stage_wall_s': s3.get('stage_wall_s')},
        'stages': stages, 'models': models, 'b0_heads': b0_heads, 'b1_heads': b1_heads, 'b1_rounds_missing': b1_rounds_missing,
        'compose': compose_part, 'runner_calls': runner_calls, 'unfinished_calls': unfinished}
    return section, sorted(missing)


NLL_COLUMNS = ('b0_masked', 'b0_open', 'b1_masked', 'b1_open')


def nll_section(dev_nll: dict) -> dict:
    """Item 2: the DEV negative log-likelihood of every head for B0 and B1, with the input sets masked (the heads the composition uses)
    and open (the heads that also see the sampled engagement). ``dev_nll`` is DEV_NLL.json. Returns the columns ``NLL_COLUMNS``, the heads
    in the order of the model (``head_rows.HEAD_NAMES``) and ``table[head][column]``: the score of the cell (nll, nll_ci, n_rows,
    n_design_rows, n_unscored) or None where the head has no such set (the open sets have no M2 head); B2 is copied. ReportInputError
    for a family or a head the table has no place for.

    이미 계산된 DEV 점수를 머리와 입력 집합별 표로 배치한다. 해당 머리가 없는 칸은 None으로 남기고 여기서 점수를 다시 매기지 않는다.
    """
    table = {h: {c: None for c in NLL_COLUMNS} for h in head_rows.HEAD_NAMES}
    for family in ('b0', 'b1'):
        if not isinstance(dev_nll.get(family), dict):
            raise ReportInputError(f'DEV_NLL.json has no {family}')
        for kind in ('masked', 'open'):
            for head, cell in dev_nll[family].get(kind, {}).items():
                if head not in table:
                    raise ReportInputError(f'DEV_NLL.json {family}.{kind} has the head {head}, which the report has no row for')
                table[head][f'{family}_{kind}'] = cell
    if 'b2' not in dev_nll:
        raise ReportInputError('DEV_NLL.json has no b2')
    return {'columns': list(NLL_COLUMNS), 'heads': list(head_rows.HEAD_NAMES), 'table': table, 'b2': dev_nll['b2']}


# ---- the occurrence line of each window -------------------------------------------------------------------------------------------------
def window_observed(s1, frame_row) -> np.ndarray:
    """(n, 3) int64: for each frame row of ``frame_row``, the number of onsets in the S1 bins 0-5, 6-11 and 12-17 (the windows 0-60,
    60-120 and 120-180 s): the differences of ``run_s3.observed_onsets`` at 6, 12 and 18 bins, the function the gate counts its 60 s
    onsets with.

    수식
    ----
    (REP3-18)  observed_{row,window}=cumulative_count(row,end_bins_window)−cumulative_count(row,end_bins_{window−1})
             end_bins=EVAL_H_MS//BIN_MS=(6,12,18); 첫 차분의 이전 값=0

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, window, end_bins | 선택 프레임 행·창 열·구간 끝 | — | 저장 행 0 기반, window=0..2 | frame_row,ends
      cumulative_count, observed | 누적·창별 발생 수 | 개 | 유효 S1 계약, 반환 int64 | run_s3.observed_onsets,cumulative,반환값

    해석
    ----
    S3 Task13의 관측 창은 같은 S1 누적 발생 함수를 차분해 만든다. 첫 창은 구간 영부터 다섯까지이며 다음 창도 연속한 여섯 구간씩이다. 이는 정수 개수이고 기대값을 확률로 바꾸는 계산이 아니다.

    근거
    ----
    S3 계획 Task13; run_s3.observed_onsets(누적 개수); 설계서 §3.1(발생 구간).
    """
    ends = [h // C.BIN_MS for h in C.EVAL_H_MS]
    cumulative = np.column_stack([run_s3.observed_onsets(s1, frame_row, n_bins) for n_bins in ends])
    return np.diff(cumulative, axis=1, prepend=0).astype(np.int64)


def _window_inputs(expected_cum, observed, match) -> tuple:
    """``(cumulative expected, expected per window, observed counts, match, weights 1 / n_u)`` once the shapes and counts were checked.

    (REP3-19)의 계산·입력 항을 맡는다.
    """
    cumulative = np.asarray(expected_cum, dtype=np.float64)
    expected = window_expectations(cumulative)
    n = len(cumulative)
    counts, match = np.asarray(observed, dtype=np.float64), np.asarray(match)
    if counts.shape != (n, N_WINDOWS) or match.shape != (n,):
        raise ReportInputError(f'the observed counts {counts.shape} and the match numbers {match.shape} must belong to the {n} rows of the '
                               'expected onsets')
    if match.dtype.kind not in 'iu':
        raise ReportInputError(f'the match numbers must be integers, not {match.dtype}')
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ReportInputError('the observed onsets must be non-negative whole numbers')
    return cumulative, expected, counts, match, features.match_weights(match)


def window_lines(expected_cum, observed, match, *, B: int = BOOT_B, seed: int = SEED) -> dict:
    """The occurrence calibration (``gate_s3.occurrence_calibration``, the function of criterion (i)) of each window of the first 180 s.

    ``expected_cum`` (n, 3) are the cumulative expected onsets of ``compose`` (up to 60, 120 and 180 s), ``observed`` (n, 3) the onset
    counts of the bins 0-5, 6-11 and 12-17 (``window_observed``), ``match`` the match number of each row; the weights are 1 / n_u over the
    rows given. The lines are the windows 0-60, 60-120 and 120-180 s (expected: the first column and the differences of the cumulative
    columns) and 0-180 s (the last cumulative column against the row sums). Each carries observed / expected and the slope of the
    observed on the expected counts, both with their match-bootstrap 95% intervals (B replicates from ``seed``), the totals, the counts of
    rows and matches, and whether the intervals lie in the gate's bands. A window the calibration cannot judge (its expected counts do not
    vary, or add up to 0) is ``{'computed': False, 'reason': ...}`` and the other windows are still computed.

    수식
    ----
    (REP3-19)  flat_window=[ptp(expected)≤10^−9·max(1,max(abs(expected)))]
             window_total_expected=Σ_row expected_row; window_total_observed=Σ_row observed_row
             weight_row=1/n_u; oe=Σ_row weight_row·observed_row / Σ_row weight_row·expected_row
             slope=Cov_w(expected,observed)/Var_w(expected)
             aggregate_expected=cumulative[:,−1]; aggregate_observed=Σ_window observed[:,window]

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, window, expected, observed, cumulative | 행·창·기대·관측·누적 기대 | —; 개 | 0 기반 창 0..2, 유한 비음수 정수 관측 검사 | e,o,counts,cumulative
      flat_window, ptp, window_total_expected, window_total_observed | 무변동 표시·최대최소차·행 합계 | —; 개 | 합계는 비가중 | np.ptp,sum_expected,sum_observed
      u, n_u, weight_row, oe, slope | 경기·경기 행 수·가중치·비·기울기 | — | 정수 경기 dtype 검사, match_weights 사용 | match,w,r
      aggregate_expected, aggregate_observed | 마지막 누적 기대와 창 관측 합 | 개 | 작은 음수 절단 때문에 창 합과 미세 차이 가능 | pairs[-1]
      B, seed | 경기 복제 수와 시드 | 회; — | 기본 B=2000, 95% 백분위 | B,seed
      Cov_w, Var_w | 가중 중심 공분산과 분산 | 개² | 가중치 합으로 나눈 중심 곱·제곱합 | gate_s3._line

    경계: 기울기의 상대 퇴화 및 복제 NaN 구간 규약은 (REP3-13)과 같은 gate_s3._line·_interval을 따른다.

    해석
    ----
    S3 P7의 경기 가중 진단과 달리 반환 total은 행 개수를 비가중으로 합한 값이다. 상대 범위가 작으면 회귀를 건너뛰고 하류 ValueError는 해당 창만 미계산으로 남긴다. 나머지 창은 계속 계산하며 무변동 판정은 수치 규약이다. 발생 비와 기울기의 구간 전체가 각각 [0.95,1.05]와 [0.9,1.1] 안인지 표시하지만 이 약한 선형 진단이 완전한 조건부 보정을 보증하지 않는다. 집계 창은 절단 전 마지막 누적 열을 사용한다.

    근거
    ----
    S3 계획 P7(i), Task13; S3 판정 R36(창 진단), R81(보정 한계); gate_s3.occurrence_calibration(구간).
    """
    cumulative, expected, counts, match, w = _window_inputs(expected_cum, observed, match)
    # (REP3-19): 전체 창은 절단한 차분 합 대신 원래 마지막 누적값을 쓴다.
    pairs = [(expected[:, j], counts[:, j]) for j in range(N_WINDOWS)] + [(cumulative[:, -1], counts.sum(axis=1))]
    lines = []
    for (lo, hi), (e, o) in zip(WINDOWS, pairs):
        head = {'name': f'{lo}-{hi} s', 'from_s': lo, 'to_s': hi}
        # (REP3-19): 수치적으로 거의 평탄한 설명변수의 회귀 기울기를 보고하지 않는다.
        if np.ptp(e) <= FLAT_RELATIVE * max(1.0, float(np.abs(e).max())):     # the gate sees exact ties only: noise would give a slope
            lines.append({**head, 'computed': False, 'reason': 'the expected counts do not vary, so the line has no slope'})
            continue
        try:
            r = gate_s3.occurrence_calibration(e, o, match, w, B=B, seed=seed)
        except ValueError as exc:
            lines.append({**head, 'computed': False, 'reason': str(exc)})
            continue
        judged = gate_s3.verdict({'occurrence': r})['occurrence']
        lines.append({**head, 'computed': True, **{k: r[k] for k in ('oe', 'oe_ci', 'slope', 'slope_ci', 'intercept', 'intercept_ci')},
                      'oe_in_band': judged['oe_in_band'], 'slope_in_band': judged['slope_in_band'],
                      'sum_expected': float(e.sum()), 'sum_observed': float(o.sum()),
                      'n_rows': r['n_rows'], 'n_matches': r['n_matches'], 'B': r['B'], 'seed': r['seed']})
    return {'windows': lines, 'B': B, 'seed': seed, 'bands': {'oe': list(gate_s3.OE_BAND), 'slope': list(gate_s3.OCC_SLOPE_BAND)}}


# ---- reading a finished run folder ------------------------------------------------------------------------------------------------------
def _read_json(path, what: str) -> dict:
    """The JSON object in ``path``; ReportInputError (naming ``what``) when the file is missing or is not a JSON object."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise ReportInputError(f'{what} is missing ({path.name}): this is not a finished S3 run folder') from None
    except (OSError, ValueError) as exc:
        raise ReportInputError(f'{what} cannot be read as JSON ({type(exc).__name__}): {path.name}') from None
    if not isinstance(doc, dict):
        raise ReportInputError(f'{what} ({path.name}) is not a JSON object')
    return doc


def _need(doc, *path: str, source: str):
    """``doc[path[0]][path[1]]...``; ReportInputError naming the field that is missing and the file it was looked for in."""
    d = doc
    for i, step in enumerate(path):
        if not isinstance(d, dict) or step not in d:
            raise ReportInputError(f'{source} has no {".".join(path[:i + 1])}')
        d = d[step]
    return d


def read_run(run: Path) -> tuple:
    """``(RUN.json, STAGES.json)`` of a finished S3 run folder, after three checks. Every stage is complete in STAGES.json and has its
    ``<stage>_result`` in RUN.json, and RUN.json has the complete ``s3_result`` (the runner writes it when the last stage is done). And the
    output files of every stage are the ones the stage recorded when it finished (``run_s3.stage_outputs``): a report describes the files
    the run wrote and not a folder that was changed after it. ReportInputError says which check failed; nothing is written.

    완료 표시와 단계 산출물 기록을 확인해 보고의 출처를 묶는다. 현재 파일과 기록의 일치를 확인하는 것이 연구 해석의 타당성 검증을 대신하지는 않는다.
    """
    manifest = _read_json(run / 'RUN.json', 'RUN.json')
    if manifest.get('stage') != 's3':
        raise ReportInputError(f'{run.name} is not an S3 run folder (RUN.json: stage {manifest.get("stage")!r})')
    ledger = _read_json(run / 'STAGES.json', 'STAGES.json')
    if ledger.get('schema') != run_s3.SCHEMA:
        raise ReportInputError(f'STAGES.json has the schema {ledger.get("schema")!r}, not {run_s3.SCHEMA!r}')
    stages = ledger.get('stages') if isinstance(ledger.get('stages'), dict) else {}
    unfinished = [s for s in run_s3.STAGES if not (stages.get(s) or {}).get('complete')]
    if unfinished:
        raise ReportInputError(f'the S3 run is not finished: the stage(s) {", ".join(unfinished)} are not complete in STAGES.json '
                               '(a report describes a finished run)')
    for stage in run_s3.STAGES + ('s3',):
        if f'{stage}_result' not in manifest:
            raise ReportInputError(f'RUN.json has no {stage}_result: the run did not record the end of its {stage} stage')
    if not (manifest['s3_result'] or {}).get('complete'):
        raise ReportInputError('RUN.json: s3_result does not say the run is complete')
    for stage in run_s3.STAGES:
        try:
            now = run_s3.stage_outputs(run, stage)
        except run_s3.RunRefused as exc:
            raise ReportInputError(str(exc)) from None
        recorded = stages[stage].get('outputs') or {}
        if now != recorded:
            changed = sorted(k for k in set(now) | set(recorded) if now.get(k) != recorded.get(k))
            raise ReportInputError(f'the outputs of the stage {stage} changed since it finished ({", ".join(changed)}): a report describes '
                                   'the files that the run wrote')
    return manifest, ledger


def locate_run(name: str, run: Path, given, runs_root, what: str, option: str) -> Path:
    """The folder of the run ``name`` that RUN.json names: ``given`` (the option of the command line), else next to the run folder, else
    in the runs root. ReportInputError when none of them is there.

    명시한 경로를 우선하고 없으면 이웃 실행 폴더와 실행 루트를 순서대로 찾는다. 폴더 발견은 이름으로 찾은 입력의 내용 일치 확인과 별개이다.
    """
    if given is not None:
        if not Path(given).is_dir():
            raise ReportInputError(f'the {what} run folder given with {option} does not exist: {Path(given).name}')
        return Path(given)
    for base in (run.parent, Path(C.RUNS_ROOT if runs_root is None else runs_root)):
        if (base / name).is_dir():
            return base / name
    raise ReportInputError(f'the {what} run {name} that RUN.json names is not next to this run folder or in the runs root: give its folder '
                           f'with {option}')


def check_s3a(manifest: dict, s3a_dir: Path) -> dict:
    """The paths and SHA-256 of ``PRICES.json`` and ``RULE_PAYOUTS.json`` of the s3a run, after both hashes were compared with the ones
    RUN.json recorded (``s3a.prices_sha256``, ``s3a.rules_sha256``): the report shows the prices the labels were made from.

    가격과 규칙 파일의 해시를 실행 기록과 대조하여 사용한 라벨 가격의 출처를 확인한다.
    """
    out = {}
    for name, rel in (('prices', 'prices/PRICES.json'), ('rules', 'rules/RULE_PAYOUTS.json')):
        path = s3a_dir / rel
        if not path.is_file():
            raise ReportInputError(f'the s3a run {s3a_dir.name} has no {rel}')
        digest, want = store_core.sha(path), _need(manifest, 's3a', f'{name}_sha256', source='RUN.json')
        if digest != want:
            raise ReportInputError(f'{path.name} of the s3a run {s3a_dir.name} has the SHA-256 {digest[:12]}..., but RUN.json records '
                                   f'{str(want)[:12]}...: it is not the file that the run was made from')
        out[name] = {'path': path, 'sha256': digest}
    return out


def open_s1(path: Path, manifest: dict):
    """The S1 store of the run folder ``path``, read only and not hashed (``verify=False``: the files hash of its manifest is compared with
    the one that RUN.json recorded instead). ReportInputError when it cannot be opened or is another store.

    S1을 읽기 전용으로 열고 manifest의 파일 목록 해시를 실행 기록과 비교한다. 실제 배열 파일 각각의 해시를 다시 읽어 검증하지는 않는다.
    """
    from c_ept.run_s1 import load_s1                       # here: no path to the runner at module level
    try:
        store = load_s1(path, verify=False)
    except (FileNotFoundError, ValueError) as exc:
        raise ReportInputError(f'the S1 store of {Path(path).name} cannot be opened: {exc}') from None
    have, want = runlog.sha256_json(store.manifest['files']), _need(manifest, 's1', 'files_sha256', source='RUN.json')
    if have != want:
        store.maps.clear()
        raise ReportInputError(f'the S1 store {Path(path).name} is not the one the run was made from (files hash {have[:12]}..., RUN.json '
                               f'records {str(want)[:12]}...)')
    return store


def read_smoke_selection(path, manifest: dict) -> dict:
    """The DEV matches of the smoke selection in ``RUN.json`` of a smoke run (``path``: its run folder or its RUN.json). Match numbers are
    rows of the S1 store, so the smoke run must have been made on the same S1 store as this run: ReportInputError when it was not, when
    it is not a smoke run or when it cannot be read.

    소규모 점검 실행에 기록된 DEV 경기 행과 S1 출처를 읽는다. 선택의 재현성은 뒤의 목록·해시 대조에서 확인한다.
    """
    path = Path(path)
    file = path / 'RUN.json' if path.is_dir() else path
    name = path.name if path.is_dir() else path.parent.name
    if not file.is_file():
        raise ReportInputError(f'the smoke run {name} has no RUN.json: give the folder of the smoke run (or its RUN.json)')
    doc = _read_json(file, f'the RUN.json of the smoke run {name}')
    smoke = doc.get('smoke')
    dev = (smoke.get('matches') or {}).get('DEV') if isinstance(smoke, dict) else None
    if not isinstance(dev, list):
        raise ReportInputError(f'{name} is not a smoke run: its RUN.json holds no smoke selection with DEV matches')
    if (doc.get('s1') or {}).get('files_sha256') != manifest['s1']['files_sha256']:
        raise ReportInputError(f'the smoke run {name} was made on another S1 store than this run: its match rows are not these matches')
    return {'run': name, 'dev_matches': [int(m) for m in dev], 'selection_sha256': smoke.get('selection_sha256'), 'n': smoke.get('n'),
            'seed': smoke.get('seed'), 'asked': smoke.get('asked'), 'matches': smoke['matches'],
            'n_fit_matches': len(smoke['matches'].get('FIT') or []), 'n_stop_matches': len(smoke['matches'].get('STOP') or [])}


SMOKE_SOURCE_ARGUMENT = 'the exclude_smoke argument (--exclude-smoke)'
SMOKE_SOURCE_DECISION = 'decisions.S3_run.smoke_run of the decision record that RUN.json names'


def decision_smoke_run(manifest: dict) -> tuple:
    """``(name, '')``: the smoke run that ``decisions.S3_run.smoke_run`` of the decision record names, the record being the file that
    RUN.json names (``args.spec``, the path the runner was given). ``(None, reason)`` when RUN.json names no record, the file cannot be
    read as JSON, or holds no ``decisions.S3_run.smoke_run`` that is a bare run folder name (no path, not empty); the reason names the file
    and never its folder. Nothing is refused here: the caller decides what a missing name means. The hash of the record is not checked (the
    record grows after a run; the runner only records it).

    실행이 가리키는 결정 기록에서 단일 실행 폴더 이름을 찾는다. 누락·읽기 실패는 사유와 None으로 넘기며 결정 기록의 현재 해시를 검증하지 않는다.
    """
    spec = (manifest.get('args') or {}).get('spec')
    if not isinstance(spec, str) or not spec:
        return None, 'RUN.json names no decision record (args.spec), so decisions.S3_run.smoke_run cannot be read'
    path = Path(spec)
    try:
        node = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None, f'the decision record {path.name} that RUN.json names cannot be read as JSON'
    for step in ('decisions', 'S3_run', 'smoke_run'):
        node = node.get(step) if isinstance(node, dict) else None
    if not isinstance(node, str) or not node or node in ('.', '..') or '/' in node or '\\' in node:
        return None, f'{path.name} has no decisions.S3_run.smoke_run that is the name of a run folder'
    return node, ''


def check_smoke_selection(smoke: dict, s1) -> None:
    """ReportInputError unless the selection that a smoke run recorded (``smoke``: the dict of ``read_smoke_selection``) is the one that
    ``run_s3.select_smoke(s1, n, seed)`` gives on this S1 store for the ``n`` and the seed it records: the same matches in every role and
    the same ``selection_sha256``. The DEV matches that the sensitivity line leaves out are then the matches the smoke run was made on,
    and not matches of another store or another selection.

    S1과 기록된 선택 인수로 얻은 경기 목록과 선택 해시를 대조한다. 이는 선택 확인이며 모델 학습이나 성능 재평가가 아니다.
    """
    name, n, seed = smoke.get('run'), smoke.get('n'), smoke.get('seed')
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (n, seed)):
        raise ReportInputError(f'the smoke run {name} does not record the n and the seed of its selection: it cannot be checked')
    try:
        want = run_s3.select_smoke(s1, n, seed)
    except run_s3.RunRefused as exc:
        raise ReportInputError(f'the selection of the smoke run {name} cannot be made again on this S1 store: {exc}') from None
    if want['matches'] != smoke.get('matches') or want['selection_sha256'] != smoke.get('selection_sha256'):
        raise ReportInputError(f'the selection that the smoke run {name} recorded (n {n}, seed {seed}) is not the selection that '
                               'run_s3.select_smoke gives for them on this S1 store: its DEV matches are not matches of this selection')


def resolve_smoke(run: Path, manifest: dict, s1, given, runs_root) -> dict:
    """Which smoke run the sensitivity line leaves out, and why there may be none. ``{'requested', 'selection', 'reason'}``.

    ``given`` (the option ``--exclude-smoke``) wins. Without it, a run that is not a smoke run takes the smoke run that
    ``decisions.S3_run.smoke_run`` names (``decision_smoke_run``), found as the other inputs are (next to the run folder, then in the
    runs root); a smoke run itself has no other smoke run to leave out and asks for nothing. ``requested`` is True when the line was
    asked for (the option, or a run that is not a smoke run). ``selection`` is the smoke run's selection (``read_smoke_selection``
    with ``source``) once it was checked against this S1 store (``check_smoke_selection``), or None with the ``reason``. A smoke run that
    was found but is not a smoke run of this store, or whose selection does not check, is a ReportInputError, as is a bad ``given``; a
    name that cannot be found or read is only a missing line (the report flags it).

    명시 인수와 실행 종류에 따라 민감도 진단의 제외 대상을 결정한다. 스스로 소규모 점검 실행인 경로와 선택을 못 찾은 경로는 이후 선택 검사를 하지 않고 사유를 반환할 수 있다.
    """
    if given is not None:
        smoke, source = read_smoke_selection(given, manifest), SMOKE_SOURCE_ARGUMENT
    elif manifest.get('smoke') is not None:                 # a smoke run is told by its smoke block, as in the run section
        return {'requested': False, 'selection': None,
                'reason': 'this run is itself a smoke run, so there is no other smoke run to leave out; give one with --exclude-smoke '
                          'to see the gate without its DEV matches'}
    else:
        name, why = decision_smoke_run(manifest)
        if name is None:
            return {'requested': True, 'selection': None, 'reason': why}
        try:
            folder = locate_run(name, run, None, runs_root, 'smoke', '--exclude-smoke')
        except ReportInputError:
            return {'requested': True, 'selection': None,
                    'reason': f'the smoke run {name} that the decision record names is not next to this run folder or in the runs root: '
                              'give its folder with --exclude-smoke'}
        smoke, source = read_smoke_selection(folder, manifest), SMOKE_SOURCE_DECISION
    check_smoke_selection(smoke, s1)
    return {'requested': True, 'selection': {**smoke, 'source': source}, 'reason': None}


def _load_arrays(path: Path, names: tuple, what: str) -> dict:
    """The arrays ``names`` of an NPZ file as memory arrays; ReportInputError when the file or an array is not there."""
    try:
        with np.load(path, allow_pickle=False) as z:
            found = set(z.files)
            arrays = {n: z[n] for n in names if n in found}
    except (OSError, ValueError) as exc:
        raise ReportInputError(f'{what} cannot be read ({type(exc).__name__}): {Path(path).name}') from None
    for n in names:
        if n not in arrays:
            raise ReportInputError(f'{what} ({Path(path).name}) has no array {n}')
    return arrays


def _check_rows(arrays: dict, what: str, closed_form: bool) -> int:
    """The number of rows of the compose arrays, after every array was found to have the shape the report reads."""
    n = len(arrays['frame_row'])
    shapes = {'match': (n,), 'expected_onsets': (n, 3), 'mc_mean': (n, 3), 'mc_sd': (n, 3), 'q19': (n, 3, N_QUANTILES)}
    if closed_form:
        shapes['expected'] = (n, 3)
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ReportInputError(f'{what}: {name} has the shape {arrays[name].shape}, not {shape}')
    return n


# ---- the sections of the report that need the run folder -------------------------------------------------------------------------------
# the fields the report copies from GATE.json (results.occurrence, brier), ACCOUNTING.json and LABELS_SUMMARY.json
OCCURRENCE_FIELDS = ('oe', 'oe_ci', 'slope', 'slope_ci', 'intercept', 'intercept_ci', 'n_rows', 'n_matches', 'B', 'seed')
BRIER_FIELDS = ('m2_b0', 'm2_b1', 'b2', 'difference', 'n_rows', 'n_matches', 'positive_share', 'B', 'seed')
ACCOUNTING_FIELDS = ('max_abs_error', 'max_abs_value_error', 'n_matches', 'n_valued_matches', 'total')
WINNER_FIELDS = ('winner_blue', 'winner_red', 'draw', 'sure', 'n_units', 'n_valued_units', 'n_rows', 'n_rows_with_value')
WINNER_SCOPE = 'all valued units of FIT, STOP and DEV, out-of-scope units included'      # what LABELS_SUMMARY.json counts the shares of


def winner_labels(sure_f0: float) -> dict:
    """What each share of the winner section is a share of: the valued units of ``WINNER_SCOPE`` (``n_valued_units`` of
    LABELS_SUMMARY.json). Not the units of the DEV rows or of the engagements the gate looks at: the label store counts every valued unit
    of the three roles, also the units out of scope (a first kill before ``config.SCOPE_FIRST_MS`` or in the last ``END_CUT_MS`` of the
    match), which the composition and V_start leave out.

    이미 집계된 승자 비율의 분모를 설명하는 라벨을 만든다. 분모는 해당 세 역할의 모든 가치 부여 단위이며 DEV 발생 행이나 범위 안 교전만으로 좁히지 않는다.
    """
    return {'winner_blue': f'share of the valued units won by blue (f0 > 0): {WINNER_SCOPE}',
            'winner_red': f'share of the valued units won by red (f0 < 0): {WINNER_SCOPE}',
            'draw': f'share of the valued units that are a draw (f0 = 0): {WINNER_SCOPE}',
            'sure': f'share of the valued units in the sure stratum (|f0| >= {sure_f0:g} gold): {WINNER_SCOPE}'}


B0_ARRAYS = ('frame_row', 'match', 'expected', 'expected_onsets', 'mc_mean', 'mc_sd', 'q19', 'S')       # compose/b0_dev.npz
B1_ARRAYS = ('frame_row', 'match', 'expected_onsets', 'mc_mean', 'mc_sd', 'q19', 'S')                   # b1_dev.npz: no closed form
SENSITIVITY_NOTE = ('The gate verdict is the one in GATE.json. This is the same two criteria on the DEV rows without the DEV matches of '
                    'the smoke run (the B2-a promise of the G-S3b gate), a sensitivity check that is not judged.')
EXCLUDED_NOTE = 'excluded_event_share = n_excluded / (n_events + n_excluded): the events whose minute window is not complete (D7)'


def _run_section(run: Path, manifest: dict, ledger: dict) -> dict:
    smoke = manifest.get('smoke')
    selection = None if smoke is None else {
        'n': smoke.get('n'), 'seed': smoke.get('seed'), 'asked': smoke.get('asked'), 'selection_sha256': smoke.get('selection_sha256'),
        **{f'n_{role.lower()}_matches': len((smoke.get('matches') or {}).get(role) or []) for role in ('FIT', 'STOP', 'DEV')}}
    return {'name': run.name, 'smoke': smoke is not None, 'smoke_selection': selection, 'code_commit': manifest.get('code_commit'),
            'code_dirty': manifest.get('code_dirty'), 'created': manifest.get('created'), 's1_run': manifest['s1']['run'],
            's2_run': (manifest.get('s2') or {}).get('run'), 's3a_run': manifest['s3a']['run'],
            'hashes': {'s1_files_sha256': manifest['s1']['files_sha256'], 's2_files_sha256': (manifest.get('s2') or {}).get('files_sha256'),
                       'prices_sha256': manifest['s3a']['prices_sha256'], 'rules_sha256': manifest['s3a']['rules_sha256']},
            'roles': manifest.get('roles'), 'compose_constants': manifest.get('compose_constants'),
            'stage_order': list(run_s3.STAGES),
            'stages': {s: {'complete': True, 'wall_s': ledger['stages'][s].get('wall_s'), 'finished': ledger['stages'][s].get('finished')}
                       for s in run_s3.STAGES}, 'gate_pass': (manifest.get('gate_result') or {}).get('gate_pass')}


def _gate_part(gate: dict) -> tuple:
    """``(gate summary, mc section)``: the verdict as GATE.json holds it (the report never judges again), the pre-registered bands from
    ``gate_s3``, and criterion (iv) with its flat rows beside the share."""
    results, verdict = _need(gate, 'results', source='GATE.json'), _need(gate, 'verdict', source='GATE.json')
    mc = mc_section(_need(results, 'mc', source='GATE.json results'), _need(verdict, 'mc', source='GATE.json verdict'))
    flat = {k: mc[k] for k in ('n_rows', 'n_agree', 'n_disagree', 'n_sd_zero', 'n_sd_zero_disagree', 'n_disagree_with_spread')}
    summary = {'gate_pass': gate['gate_pass'], 'rule': _need(gate, 'constants', 'rule', source='GATE.json'), 'constants': gate['constants'],
               'verdict': {**{k: verdict[k] for k in ('occurrence', 'value', 'accounting', 'overdispersion', 'all_pass', 'missing')},
                           'mc': {**verdict['mc'], **flat}},
               'bands': {'oe': list(gate_s3.OE_BAND), 'occurrence_slope': list(gate_s3.OCC_SLOPE_BAND),
                         'value_slope': list(gate_s3.VALUE_SLOPE_BAND), 'bias_gold': list(gate_s3.BIAS_BAND),
                         'mc_share': gate_s3.MC_SHARE, 'dispersion_limit': gate_s3.DISPERSION_LIMIT},
               'n_dev_matches': gate.get('n_dev_matches'), 'n_dev_matches_with_rows': gate.get('n_dev_matches_with_rows'),
               'm2_b0_workers_against_parent_max_relative_difference': gate.get('m2_b0_workers_against_parent_max_relative_difference')}
    return summary, mc


def _value_part(b0: dict, b1_block: dict, realized60: np.ndarray, w0: np.ndarray, gate_value: dict) -> dict:
    """Item 5: the deciles of the gate, and the PIT, the hit rates and the CRPS of the 19 quantiles of V_start at H = 60 s for B0 (every DEV
    row) and B1 (its sample).

    (REP3-13, REP3-1, REP3-3)의 계산·입력 항을 맡는다.
    """
    q0 = b0['q19'][:, 0, :]
    hits0 = quantile_hits(q0, realized60, w0)
    return {'horizon_s': C.EVAL_H_MS[0] // 1000, 'deciles': gate_value['deciles'],
            **{k: gate_value[k] for k in ('slope', 'slope_ci', 'bias', 'bias_ci', 'n_rows', 'n_matches', 'B', 'seed')},
            'bands': {'slope': list(gate_s3.VALUE_SLOPE_BAND), 'bias_gold': list(gate_s3.BIAS_BAND)},
            'pit': {'b0': pit_histogram(q0, realized60, w0), 'b1': b1_block['pit']},
            'crps19': {'b0': crps19(q0, realized60, w0), 'b1': b1_block['crps19']},
            'quantile_hits': {'b0': hits0, 'b1': b1_block['quantile_hits']},
            'note': 'pit: the 19 quantiles of the simulated V_start cut the unit interval in 20 bins; a row that ties with quantiles (a '
                    'point mass, for example V_start = 0) spreads its weight over the bins it covers'}


def _sensitivity_part(smoke: dict, b0: dict, observed60, realized60, results: dict, B: int, seed: int) -> dict:
    """The sensitivity line. ``smoke`` is ``resolve_smoke``: without a selection the line is not computed and says why (``requested`` tells
    whether it was asked for, so that ``findings`` can flag a missing line of a run that should have it)."""
    smoke_ex = smoke['selection']
    if smoke_ex is None:
        return {'requested': smoke['requested'], 'computed': False, 'reason': smoke['reason']}
    r = sensitivity_without_smoke(b0['match'], b0['expected_onsets'][:, 0], observed60, b0['expected'][:, 0], realized60,
                                  smoke_ex['dev_matches'], B=B, seed=seed)
    out = {'requested': True, 'source': smoke_ex['source'], 'smoke_run': smoke_ex['run'],
           'smoke_selection_sha256': smoke_ex['selection_sha256'], **r, 'note': SENSITIVITY_NOTE}
    if r['computed']:
        occ, val = results['occurrence'], results['value']
        out['gate_values'] = {'occurrence': {k: occ[k] for k in ('oe', 'oe_ci', 'slope', 'slope_ci')},
                              'value': {k: val[k] for k in ('slope', 'slope_ci', 'bias', 'bias_ci')}}
        out['difference'] = {'occurrence_oe': r['occurrence']['oe'] - occ['oe'],
                             'occurrence_slope': r['occurrence']['slope'] - occ['slope'],
                             'value_slope': r['value']['slope'] - val['slope'], 'value_bias': r['value']['bias'] - val['bias']}
    return out


def build_report(run: Path, manifest: dict, ledger: dict, s1, ctx, prices: dict, smoke: dict, B: int, seed: int) -> dict:
    """Every section of S3_REPORT.json (but the figures, the findings and the key scan, which need the finished sections). Reads the files
    of the run (GATE.json, DEV_NLL.json, FIT_FACTS.json, ... and the compose arrays), the S1 onsets through ``s1`` and the label store;
    computes the lines that GATE.json does not hold (per window, without the smoke matches, the composition of B1, the PIT). ``smoke`` is
    what ``resolve_smoke`` gave: the smoke run whose DEV matches the sensitivity line leaves out, or why there is none.

    수식
    ----
    (REP3-20)  filled_cells_total=Σ_column recorded_fill_count_column; listed_fill_column=[recorded_fill_count_column≠0]

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      column, recorded_fill_count_column, filled_cells_total | 입력 열·기록된 대체 칸 수·합 | —; 칸 | 0 기반 열, 유효 적합 기록 계약 | b0_names,fit_inputs['b0_fill_counts'],n_b0_cells_filled
      listed_fill_column | 보고 사전에 남기는 열 표시 | — | 영인 열은 생략 | b0_fill_counts

    해석
    ----
    S3 Task13은 적합 때 기록한 대체 칸 수를 합하며 여기서 대체나 적합을 실행하지 않는다. 집계 외의 연구 통계는 각 공개 소유 함수에서 계산하거나 저장 결과를 복사한다. 한 번의 통과나 회계 일치로 새로운 과학적 주장을 만들지 않는다.

    근거
    ----
    S3 계획 Task13(보고 항목); FIT_INPUTS 기록 계약; S3 판정 R69(공개 소유).
    """
    smoke_ex = smoke['selection']
    gate = _read_json(run / 'gate' / 'GATE.json', 'GATE.json')
    fit = _read_json(run / 'fit' / 'FIT_FACTS.json', 'FIT_FACTS.json')
    fit_inputs = _read_json(run / 'fit' / 'FIT_INPUTS.json', 'FIT_INPUTS.json')
    dev_nll = _read_json(run / 'fit' / 'DEV_NLL.json', 'DEV_NLL.json')
    b0_heads = _read_json(run / 'fit' / 'b0' / 'b0_heads.json', 'b0_heads.json')
    accounting = _read_json(run / 'labels' / 'ACCOUNTING.json', 'ACCOUNTING.json')
    summary = _read_json(run / 'labels' / 'LABELS_SUMMARY.json', 'LABELS_SUMMARY.json')
    compose = _read_json(run / 'compose' / 'COMPOSE.json', 'COMPOSE.json')
    b0 = _load_arrays(run / 'compose' / 'b0_dev.npz', B0_ARRAYS, 'the B0 composition')
    b1 = _load_arrays(run / 'compose' / 'b1_dev.npz', B1_ARRAYS, 'the B1 composition')
    n0, n1 = _check_rows(b0, 'b0_dev.npz', True), _check_rows(b1, 'b1_dev.npz', False)
    try:
        labels = run_s3.open_labels(ctx)
    except run_s3.RunRefused as exc:
        raise ReportInputError(str(exc)) from None
    try:
        v_start = np.asarray(labels.array('labels/v_start'))
        realized = np.asarray(v_start[b0['frame_row']], dtype=np.float64)
    finally:
        labels.maps.clear()
    if realized.shape != (n0, 3) or not np.isfinite(realized[:, 0]).all():
        raise ReportInputError('labels/v_start has no finite V_start at H = 60 s for a DEV row of the composition')
    observed = window_observed(s1, b0['frame_row'])
    pos = b1_positions(b0, b1)
    w0 = features.match_weights(b0['match'])
    gate_summary, mc = _gate_part(gate)
    by_rate = mc_by_rate(b0['mc_mean'][:, 0], b0['mc_sd'][:, 0], mc['S'], b0['expected'][:, 0], b0['expected_onsets'][:, 0])
    mc['by_expected_onsets'] = {**by_rate, 'variable': MC_RATE_VARIABLE, 'note': MC_RATE_NOTE}
    results = gate['results']

    lines_b0 = window_lines(b0['expected_onsets'], observed, b0['match'], B=B, seed=seed)
    lines_b1 = window_lines(b1['expected_onsets'], observed[pos], b1['match'], B=B, seed=seed)
    sample = compose.get('b1_sample') or 'sample not recorded'
    slopes = {'b0': {'rows': n0, 'matches': int(len(np.unique(b0['match']))), 'sample': 'all the DEV rows',
                     'note': f'B0 is composed on all {n0:,} DEV rows', **lines_b0},
              'b1': {'rows': n1, 'matches': int(len(np.unique(b1['match']))), 'sample': sample,
                     'note': b1_coverage_note(sample, n1, n0, _scalar(b1['S'])), **lines_b1},
              'check_against_gate': {**compare_with_gate(lines_b0['windows'][0], results['occurrence']),
                                     'compared': 'the 0-60 s line of B0 against results.occurrence of GATE.json'},
              'note': 'expected: the first column and the differences of the cumulative expected_onsets columns of the compose files; '
                      'observed: the S1 onsets in the bins 0-5, 6-11 and 12-17 of each frame row; weights 1 / n_m; B = 2,000 '
                      'match-bootstrap replicates'}
    composition = b1_composition(b0, b1, realized[:, 0], observed, sample=sample, B=B, seed=seed)
    cost, missing = cost_section(manifest, ledger, fit, compose)
    b0_names = list(fit_inputs['b0_names'])
    from c_ept import value                                          # here: the label module is needed for one constant
    s2 = manifest.get('s2') or {}
    verdict = gate['verdict']
    doc = {
        'report': 'S3_REPORT', 'run': _run_section(run, manifest, ledger),
        'inputs': {'s1': {'run': manifest['s1']['run'], 'files_sha256': manifest['s1']['files_sha256'], 'checked': True},
                   's2': {'run': s2.get('run'), 'files_sha256': s2.get('files_sha256')},
                   's3a': {'run': manifest['s3a']['run'], 'prices_sha256': manifest['s3a']['prices_sha256'],
                           'rules_sha256': manifest['s3a']['rules_sha256'], 'checked': True},
                   'stage_outputs_verified': True, 'missing_optional_fields': missing,
                   'exclude_smoke': None if smoke_ex is None else {
                       **{k: smoke_ex[k] for k in ('run', 'selection_sha256', 'n', 'seed', 'source')}, 'checked': True}},
        'gate': gate_summary,
        'd7': {**d7_section(prices), 'note': EXCLUDED_NOTE, 'source': 'PRICES.json of the s3a run (SHA-256 checked against RUN.json)'},
        'dev_nll': nll_section(dev_nll),
        'brier': {**{k: gate['brier'][k] for k in BRIER_FIELDS},
                  'note': 'the forecasts of at least one onset within the first 60 s, M2 of B0 and of B1 against B2; reported, not judged'},
        'occurrence_deciles': {'horizon_s': C.EVAL_H_MS[0] // 1000, 'deciles': results['occurrence']['deciles'],
                               **{k: results['occurrence'][k] for k in OCCURRENCE_FIELDS},
                               'bands': {'oe': list(gate_s3.OE_BAND), 'slope': list(gate_s3.OCC_SLOPE_BAND)}},
        'value_deciles': _value_part(b0, composition['b1'], realized[:, 0], w0, results['value']),
        'mc_agreement': mc,
        'overdispersion': {**results['overdispersion'], 'limit': verdict['overdispersion']['limit']},
        'accounting': {**{k: accounting[k] for k in ACCOUNTING_FIELDS}, 'must_equal': verdict['accounting']['must_equal'],
                       'gate_pass': verdict['accounting']['pass'],
                       'value_check': accounting_value_check(accounting['max_abs_value_error'])},
        'winner': {**{k: summary[k] for k in WINNER_FIELDS}, 'sure_threshold_gold': value.SURE_F0, 'scope': WINNER_SCOPE,
                   'field_labels': winner_labels(value.SURE_F0)},
        'dropped_columns': {'dropped': fit_inputs['dropped'], 'n_dropped': len(fit_inputs['dropped']),
                            'n_fit_views': fit_inputs['n_fit_views'], 'n_fit_matches': fit_inputs['n_fit_matches'],
                            'b0_fill_counts': {n: int(c) for n, c in zip(b0_names, fit_inputs['b0_fill_counts']) if c},
                            'n_b0_cells_filled': int(sum(fit_inputs['b0_fill_counts']))},
        'b1_composition': composition,
        'sensitivity_without_smoke_dev': _sensitivity_part(smoke, b0, observed[:, 0], realized[:, 0], results, B, seed),
        'window_slopes': slopes,
        'b0_convergence': convergence_section(_need(fit, 'b0', 'summary', source='FIT_FACTS.json'),
                                              _need(b0_heads, 'fits', source='b0_heads.json')),
        'cost': cost,
        'tower_pricing': gate.get('tower_pricing')}
    return doc


# ---- figures (Korean labels, Malgun Gothic, mathtext for the subscripts) ----------------------------------------------------------------
FONT = 'Malgun Gothic'
INK, MUTED, ACCENT, WARN, GRID = '#15201f', '#50605e', '#0d6b66', '#8a5a00', '#d9e0df'
GROUP_KR = {'dragon': '원소 드래곤', 'elder': '장로 드래곤', 'grubs': '공허 유충', 'herald': '전령', 'baron': '바론', 'atakhan': '아타칸',
            'tower_outer': '포탑 1차', 'tower_inner': '포탑 2차', 'tower_base': '포탑 3차', 'tower_nexus': '넥서스 포탑', 'inhibitor': '억제기',
            'plate': '포탑 방패'}


def _plt():
    """pyplot on the Agg backend with Malgun Gothic and the plain minus sign (as ``report_s1``). RuntimeError when the font is missing: the
    Korean text would be boxes."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    try:
        font_manager.findfont(FONT, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(f'the font {FONT} is not available; the figure text needs its Hangul glyphs') from exc
    plt.rcParams.update({'font.family': FONT, 'axes.unicode_minus': False, 'font.size': 9, 'axes.titlesize': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 130, 'figure.dpi': 100})
    return plt


def _render(plt, fig) -> bytes:
    """The figure as PNG bytes. A warning about a glyph that the font lacks raises RuntimeError: no figure is kept with boxes in it."""
    buffer = io.BytesIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.savefig(buffer, format='png')
    plt.close(fig)
    bad = sorted({str(w.message) for w in caught if 'missing from font' in str(w.message) or 'Glyph' in str(w.message)})
    if bad:
        raise RuntimeError(f'a figure has glyphs that {FONT} lacks: {bad[:3]}')
    return buffer.getvalue()


def _f(values) -> np.ndarray:
    """A float array from numbers and (nested) lists of numbers in which None (a number that could not be computed) is NaN."""
    def plain(v):
        return [plain(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else (math.nan if v is None else v)
    return np.asarray(plain(values), dtype=np.float64)


def _g(v, digits: int = 0) -> str:
    return '-' if v is None or not math.isfinite(v) else f'{v:,.{digits}f}'


def _ci_text(ci, digits: int = 3) -> str:
    return '[-, -]' if ci is None or None in ci else f'[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]'


def fig_d7(plt, doc: dict) -> tuple:
    """기록된 사건 연구 계수와 구간·채택 결과를 그림에 옮긴다. 추정하지 못한 곡선은 그 상태를 표시하며 여기서 가격을 다시 추정하지 않는다."""
    d7 = doc['d7']
    minutes = np.array(d7['relative_minutes'])
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.6), sharex=True)
    for ax, name in zip(axes.flat, GROUPS):
        e = d7['groups'][name]
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(-0.5, color=MUTED, lw=0.6, ls=':')
        if e['beta'] is not None and e['beta_ci'] is not None:
            beta, ci = _f(e['beta']), _f(e['beta_ci'])
            ax.vlines(minutes, ci[:, 0], ci[:, 1], color=ACCENT, lw=2.2, alpha=0.45)
            ax.plot(minutes, beta, 'o-', color=ACCENT, ms=4, lw=1.2)
        else:
            ax.text(0.5, 0.5, '흐름 추정 없음', transform=ax.transAxes, ha='center', va='center', color=MUTED)
        use = '흐름' if e['used'] == 'flow' else '즉시'
        ax.set_title(f"{GROUP_KR[name]}  ({use} {_g(e['value'])})", fontsize=10, color=INK, loc='left', pad=30)
        tag = '고정 즉시' if e['fixed_immediate'] else ('채택' if e['adopted'] else '앞 추세 기각')
        share = e['excluded_event_share']
        ax.text(0.0, 1.03, f"p {_g(e['price_flow'])} · T {_g(e['pretrend'])} {_ci_text(e['pretrend_ci'], 0)} · {tag}\n"
                           f"완전 {e['n_events']:,} / 제외 {e['n_excluded']:,}" + ('' if share is None else f' ({share:.0%})'),
                transform=ax.transAxes, va='bottom', fontsize=7.5, color=MUTED)
        ax.set_xticks(minutes)
        ax.grid(axis='y', color=GRID, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel('사건 기준 상대 분 r (r = 0이 사건이 든 분)')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\beta_r$ (골드, $\beta_{-1}=0$)')
    title = f'그림 1. 쌓인 사건 연구의 $\\beta_r$과 95% 경기 부트스트랩 구간 (FIT {d7["n_matches"]:,}경기, B = {d7["B"]:,})'
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    numbers = {'groups': list(GROUPS), 'relative_minutes': d7['relative_minutes'], 'beta': {g: d7['groups'][g]['beta'] for g in GROUPS},
               'used': {g: d7['groups'][g]['used'] for g in GROUPS}, 'value': {g: d7['groups'][g]['value'] for g in GROUPS}}
    return title, numbers, _render(plt, fig)


def fig_occurrence(plt, doc: dict) -> tuple:
    """기록된 기대 십분위의 관측·기대 평균과 구간을 그린다. 일대일 참조선이나 십분위 일치만으로 완전한 조건부 보정을 주장하지 않는다."""
    occ = doc['occurrence_deciles']
    x, y = _f([d['mean_expected'] for d in occ['deciles']]), _f([d['mean_observed'] for d in occ['deciles']])
    fig, ax = plt.subplots(figsize=(6.8, 6.4), constrained_layout=True)
    both = np.r_[x, y]
    top = float(np.nanmax(both)) * 1.1 if np.isfinite(both).any() else 1.0
    ax.plot([0, top], [0, top], color=MUTED, lw=1.0, ls='--', label='관측 = 기대')
    ax.plot(x, y, 'o-', color=ACCENT, ms=5, lw=1.3, label='기대 십분위별 평균')
    for k, (xi, yi) in enumerate(zip(x, y), start=1):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(str(k), (xi, yi), textcoords='offset points', xytext=(4, -10), fontsize=7, color=MUTED)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel('기대 발생 수 (첫 60 s, 십분위 안 평균)')
    ax.set_ylabel('관측 발생 수')
    ax.grid(color=GRID, lw=0.5)
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    bands = occ['bands']
    ax.text(0.03, 0.97, f"관측 / 기대 {occ['oe']:.3f} {_ci_text(occ['oe_ci'])}  (띠 {bands['oe']})\n"
                        f"기울기 {occ['slope']:.3f} {_ci_text(occ['slope_ci'])}  (띠 {bands['slope']})",
            transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
    title = f"그림 2. 발생 보정: 첫 60 s의 발생 수, 관측 대 기대 (DEV {occ['n_rows']:,}행, {occ['n_matches']:,}경기)"
    ax.set_title(title, loc='left')
    numbers = {'mean_expected': [d['mean_expected'] for d in occ['deciles']], 'mean_observed': [d['mean_observed'] for d in occ['deciles']],
               'oe': occ['oe'], 'slope': occ['slope']}
    return title, numbers, _render(plt, fig)


def fig_value(plt, doc: dict) -> tuple:
    """수식
    ----
    (REP3-21)  pit_centre_bin=(bin+0.5)/pit_bins

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      bin, pit_bins, pit_centre_bin | 진단 빈 번호·빈 수·중심 | — | 0 기반, pit_bins=20 | np.arange, val['pit'], centres
      mass_bin | 이미 계산한 빈 질량 | — | (REP3-1)의 거친 진단 | b0,b1

    해석
    ----
    S3 Task13의 가치 십분위와 PIT를 함께 표시하며 재보정을 수행하지 않는다. B0 전체 DEV와 B1 표본의 분모를 범례로 구별한다. 고른 질량 선은 S3 R81의 명목 참조이며 보정된 이산 법의 보장이나 불일치 판정선이 아니다.

    근거
    ----
    S3 계획 Task13; S3 판정 R81(PIT 표시 한계)."""
    val = doc['value_deciles']
    x, y = _f([d['mean_expected'] for d in val['deciles']]), _f([d['mean_realized'] for d in val['deciles']])
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.6), constrained_layout=True)
    both = np.r_[x, y]
    lo, hi = (float(np.nanmin(both)), float(np.nanmax(both))) if np.isfinite(both).any() else (0.0, 1.0)
    pad = 0.08 * (hi - lo or 1.0)
    left.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1.0, ls='--', label='실현 = 기대')
    left.plot(x, y, 'o-', color=ACCENT, ms=5, lw=1.3, label='기대 십분위별 평균')
    left.set_xlabel(r'기대 $V_{\mathrm{start}}$ (골드, 십분위 안 평균)')
    left.set_ylabel(r'실현 $V_{\mathrm{start}}$ (골드)')
    left.grid(color=GRID, lw=0.5)
    left.legend(loc='lower right', frameon=False, fontsize=8.5)
    left.text(0.03, 0.97, f"기울기 {val['slope']:.3f} {_ci_text(val['slope_ci'])}  (띠 {val['bands']['slope']})\n"
                          f"평균 오차 {val['bias']:.1f} {_ci_text(val['bias_ci'], 1)} 골드  (띠 {val['bands']['bias_gold']})",
              transform=left.transAxes, va='top', fontsize=8.5, color=INK)
    left.set_title(r'골드 보정: 십분위, H = 60 s', loc='left')
    # (REP3-21): 빈 질량은 재추정하지 않고 진단 빈 중앙에 표시한다.
    centres = (np.arange(val['pit']['b0']['bins']) + 0.5) / val['pit']['b0']['bins']
    b0, b1 = np.asarray(val['pit']['b0']['mass']), np.asarray(val['pit']['b1']['mass'])
    right.bar(centres, b0, width=0.04, color=ACCENT, alpha=0.75,
              label=f"B0: DEV {val['pit']['b0']['n_rows']:,}행 (CRPS {val['crps19']['b0']:.1f} 골드)")
    right.plot(centres, b1, 'o-', color=WARN, ms=4, lw=1.2,
               label=f"B1: 표본 {val['pit']['b1']['n_rows']:,}행 (CRPS {val['crps19']['b1']:.1f} 골드)")
    right.axhline(val['pit']['b0']['expected_mass'], color=MUTED, ls='--', lw=1.0, label='고른 분포 (0.05)')
    right.set_ylim(0, 1.3 * float(max(b0.max(), np.nanmax(b1), val['pit']['b0']['expected_mass'])))
    right.set_xlabel('PIT: 실현값이 모의 분포의 어디에 있는가 (19분위가 나누는 20칸)')
    right.set_ylabel('행 비율 (경기 가중)')
    right.grid(axis='y', color=GRID, lw=0.5)
    right.legend(loc='upper center', frameon=False, fontsize=8.5, ncol=1)
    right.set_title(r'PIT 히스토그램, H = 60 s', loc='left')
    title = r'그림 3. $V_{\mathrm{start}}$ 보정: 십분위와 H = 60 s의 PIT'
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, color=INK)
    numbers = {'mean_expected': [d['mean_expected'] for d in val['deciles']], 'mean_realized': [d['mean_realized'] for d in val['deciles']],
               'pit_b0': b0.tolist(), 'pit_b1': b1.tolist(), 'crps19': val['crps19']}
    return title, numbers, _render(plt, fig)


def fig_windows(plt, doc: dict) -> tuple:
    """창별로 이미 계산한 기울기와 구간을 나란히 표시한다. 비유한 점·구간의 표시를 걸러내며 B1 표본 범위를 캡션으로 구별한다."""
    slopes = doc['window_slopes']
    names = [w['name'] for w in slopes['b0']['windows']]
    band = slopes['b0']['bands']['slope']
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    ax.axhspan(band[0], band[1], color=ACCENT, alpha=0.10, label=f'띠 [{band[0]}, {band[1]}]')
    ax.axhline(1.0, color=MUTED, lw=0.8, ls='--')
    numbers = {'windows': names, 'band': list(band)}
    for head, offset, color, label in (('b0', -0.1, ACCENT, f"B0: DEV {slopes['b0']['rows']:,}행"),
                                       ('b1', 0.1, WARN, f"B1: {slopes['b1']['rows']:,}행 (표본)")):
        lines = slopes[head]['windows']
        point = _f([w.get('slope') for w in lines])
        ci = _f([w.get('slope_ci', [None, None]) for w in lines])
        xs = np.arange(len(lines)) + offset
        ok = np.isfinite(point) & np.isfinite(ci).all(axis=1)
        ax.vlines(xs[ok], ci[ok, 0], ci[ok, 1], color=color, lw=2.4, alpha=0.5)
        ax.plot(xs[np.isfinite(point)], point[np.isfinite(point)], 'o', color=color, ms=6, label=label)
        numbers[head] = {'slope': [w.get('slope') for w in lines], 'slope_ci': [w.get('slope_ci') for w in lines]}
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel('기울기 (관측 발생 수를 기대 발생 수에 회귀)')
    ax.set_xlabel('창 (사건 모형의 시점 $t_0$ 이후)')
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.legend(loc='best', frameon=False, fontsize=8.5)
    title = r'그림 4. 창별 발생 보정 기울기와 95% 경기 부트스트랩 구간 (DEV, 기준 시점 $t_0$)'
    ax.set_title(title, loc='left')
    b0_rows, b1_rows = slopes['b0']['rows'], slopes['b1']['rows']
    caption = (f"B1은 DEV 전체({b0_rows:,}행)에서 모의했다" if b1_rows == b0_rows else
               f"B1은 DEV 전체가 아니라 표본({slopes['b1']['matches']:,}경기, {b1_rows:,}행 / DEV {b0_rows:,}행)에서만 모의했다")
    ax.text(0.0, -0.17, caption, transform=ax.transAxes, fontsize=8, color=MUTED, va='top')
    return title, numbers, _render(plt, fig)


def fig_brier(plt, doc: dict) -> tuple:
    """기록된 Brier 점수와 같은 경기 복제의 쌍 차이를 그린다. 이 함수는 점수·구간을 다시 추정하지 않는다."""
    brier = doc['brier']
    keys, labels = ('m2_b0', 'm2_b1', 'b2'), ('M2 (B0)', 'M2 (B1)', 'B2')
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 5.2), constrained_layout=True, gridspec_kw={'width_ratios': [1, 1.2]})
    values = _f([brier[k]['brier'] for k in keys])
    ci = _f([brier[k]['ci'] for k in keys])
    xs = np.arange(3)
    left.vlines(xs, ci[:, 0], ci[:, 1], color=ACCENT, lw=2.6, alpha=0.5)
    left.plot(xs, values, 'o', color=ACCENT, ms=7)
    for xi, vi in zip(xs, values):
        left.annotate(f'{vi:.4f}', (xi, vi), textcoords='offset points', xytext=(9, -3), fontsize=8.5, color=INK)
    left.set_xticks(xs, labels)
    left.set_xlim(-0.6, 2.9)
    left.set_ylabel('Brier 점수 (낮을수록 좋다)')
    left.grid(axis='y', color=GRID, lw=0.5)
    left.set_title('첫 60 s에 발생이 하나 이상인가', loc='left')
    pairs = (('m2_b0_minus_b2', 'M2 (B0) - B2'), ('m2_b1_minus_b2', 'M2 (B1) - B2'), ('m2_b0_minus_m2_b1', 'M2 (B0) - M2 (B1)'))
    diff = _f([brier['difference'][k]['value'] for k, _ in pairs])
    dci = _f([brier['difference'][k]['ci'] for k, _ in pairs])
    ys = np.arange(len(pairs))[::-1]
    right.axvline(0, color=MUTED, lw=0.8, ls='--')
    right.hlines(ys, dci[:, 0], dci[:, 1], color=WARN, lw=2.6, alpha=0.5)
    right.plot(diff, ys, 'o', color=WARN, ms=7)
    right.set_yticks(ys, [label for _, label in pairs])
    right.set_xlabel('Brier 차이와 95% 경기 부트스트랩 구간 (0보다 작으면 앞의 모형이 낫다)')
    right.grid(axis='x', color=GRID, lw=0.5)
    right.set_title('쌍별 차이 (같은 경기를 같이 뽑은 복제)', loc='left')
    title = f"그림 5. M2(B0, B1) 대 B2의 Brier 점수 (DEV {brier['n_rows']:,}행, {brier['n_matches']:,}경기, 양성 비율 {brier['positive_share']:.3f})"
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, color=INK)
    numbers = {'brier': {k: brier[k]['brier'] for k in keys}, 'ci': {k: brier[k]['ci'] for k in keys},
               'difference': {k: brier['difference'][k] for k, _ in pairs}}
    return title, numbers, _render(plt, fig)


FIGURE_MAKERS = {'fig1_d7_beta': fig_d7, 'fig2_occurrence_deciles': fig_occurrence, 'fig3_value_deciles_pit': fig_value,
                 'fig4_window_slopes': fig_windows, 'fig5_brier': fig_brier}


# ---- the report -------------------------------------------------------------------------------------------------------------------------
def report(run, exclude_smoke=None, *, s1=None, s3a=None, runs_root=None, report_dir=None) -> Path:
    """Write ``S3_REPORT.json`` and the five figures of the finished S3 run folder ``run`` into ``report_dir`` (default
    ``<run>\\report``) and return the path of the JSON. The report reads files only: RUN.json, STAGES.json and the stage outputs of
    ``run``; ``PRICES.json`` and ``RULE_PAYOUTS.json`` of the s3a run (their SHA-256 are compared with RUN.json); the S1 arrays
    ``onsets/row``, ``onsets/bin`` and ``frame_rows/t0`` (and the keys, for the scan) of the S1 store, found next to the run folder, in
    the runs root or at ``s1``; and, for the sensitivity line alone, the smoke selection in the RUN.json of ``exclude_smoke`` (a smoke
    run folder) or, when that is not given and the run is not a smoke run, of the run that ``decisions.S3_run.smoke_run`` of the decision
    record named by RUN.json gives (``resolve_smoke``). It fits nothing and loads no model.

    It writes nothing when it refuses (ReportInputError: a stage that is not complete, an output changed after the run finished, an input
    that is not the one the run was made from, a smoke selection that is not the one ``run_s3.select_smoke`` gives on the S1 store, a field
    that is missing) or when a match key or game id would be in the text. The order:
    read and check, compute, draw the figures in memory, scan the report text for the match keys and game ids of the S1 store
    (``assert_clean``), and only then write the figures and, last, the JSON.

    완료 실행의 출처를 검사한 뒤 항목을 구성하고 메모리에서 그림을 그려 식별자 검사를 통과한 결과만 쓴다. 보고는 관문 판정을 복사하며 학습이나 모델 적재를 수행하지 않는다.
    """
    run = Path(run)
    manifest, ledger = read_run(run)
    files = check_s3a(manifest, locate_run(_need(manifest, 's3a', 'run', source='RUN.json'), run, s3a, runs_root, 's3a', '--s3a'))
    store = open_s1(locate_run(_need(manifest, 's1', 'run', source='RUN.json'), run, s1, runs_root, 'S1', '--s1'), manifest)
    try:
        smoke = resolve_smoke(run, manifest, store, exclude_smoke, runs_root)    # a smoke run that does not check: refused early
        plt = _plt()                                                           # the font is looked for before the long part is done
        ctx = run_s3.Context(run=run, s1=store, s2=None, prices={}, rules=None, hashes={})
        B, seed = BOOT_B, SEED
        prices = _read_json(files['prices']['path'], 'PRICES.json')
        try:
            doc = build_report(run, manifest, ledger, store, ctx, prices, smoke, B, seed)
        except KeyError as exc:
            raise ReportInputError(f'a record of the run lacks the field {exc}: the report reads the layout that the runner '
                                   'writes') from None
        pngs, doc['figures'] = {}, {}
        for name, make in FIGURE_MAKERS.items():
            title, numbers, pngs[name] = make(plt, doc)
            doc['figures'][name] = {'file': f'{name}.png', 'title': title, 'numbers': numbers}
        out = run / 'report' if report_dir is None else Path(report_dir)
        doc['key_scan'] = {'checked': True, 'n_keys': len(ctx.keys()), 'min_game_id_digits': MIN_GAME_ID_DIGITS,
                           'files': ['S3_REPORT.json'],
                           'note': 'the text of S3_REPORT.json holds no S1 match key and no game id as a whole number (assert_clean)'}
        doc['report_dir'] = out.name
        state = runlog.code_state()
        doc['report_code'] = {'commit': state['code_commit'], 'dirty': state['code_dirty']}
        doc['created'] = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
        doc['flags'] = findings(doc)
        text = run_s3.render(doc)
        assert_clean(ctx, text)
        out.mkdir(parents=True, exist_ok=True)
        for name, png in pngs.items():
            store_core.write_atomic(out / f'{name}.png', png)
        store_core.write_atomic(out / 'S3_REPORT.json', text)
        return out / 'S3_REPORT.json'
    finally:
        store.maps.clear()


def build_parser() -> argparse.ArgumentParser:
    """보고 입력과 출력 경로를 명령행 선택지로 묶는다. 명령행 정의 자체는 계산이나 관문 실행을 시작하지 않는다."""
    ap = argparse.ArgumentParser(prog='python -m c_ept.report_s3',
                                 description='S3 report: S3_REPORT.json and the figures of a finished S3 run.')
    ap.add_argument('--run', type=Path, required=True, help='the S3 run folder (all five stages complete)')
    ap.add_argument('--exclude-smoke', type=Path, default=None, dest='exclude_smoke',
                    help='a smoke run folder: the gate is also recomputed without its DEV matches (the sensitivity line). Default for a '
                         'run that is not a smoke run: the run that decisions.S3_run.smoke_run of the decision record named by RUN.json '
                         'gives, looked for next to the run folder and in the runs root')
    ap.add_argument('--s1', type=Path, default=None, help='the S1 run folder (default: next to the run folder, or in the runs root)')
    ap.add_argument('--s3a', type=Path, default=None, help='the s3a run folder (default: next to the run folder, or in the runs root)')
    ap.add_argument('--runs-root', type=Path, default=None,
                    help=f'where the S1 and s3a runs are looked for after the folder of the run (default {C.RUNS_ROOT})')
    ap.add_argument('--report-dir', type=Path, default=None, help='where the report and the figures go (default <run>\\report)')
    return ap


def _s1_keys(a) -> list:
    """The match keys, as text, of the S1 store that the command line ``a`` asks the report to read: ``--s1``, or the S1 run that RUN.json
    names, found as ``report`` finds it. They are read only to take them out of a traceback."""
    from c_ept.run_s1 import load_s1
    folder = a.s1
    if folder is None:
        manifest = _read_json(Path(a.run) / 'RUN.json', 'RUN.json')
        folder = locate_run(_need(manifest, 's1', 'run', source='RUN.json'), Path(a.run), None, a.runs_root, 'S1', '--s1')
    store = load_s1(folder, verify=False)
    try:
        return [k.decode('ascii') for k in np.asarray(store.array('matches/key')).tolist() if k]
    finally:
        store.maps.clear()


def scrubbed_traceback(a, exc: BaseException) -> str:
    """The traceback of ``exc`` with every match key of the S1 store replaced by <key> and the game id in it by <id>
    (``run_s3._scrub_text``, the scrub of ``store_build``). An unexpected error may carry a key in its message, a file name or a value that
    it quotes. When the keys cannot be read to do this the traceback is not printed: the type of the error is named instead, with the
    reason.

    주어진 S1 키와 게임 번호를 예외 텍스트에서 가려 전달한다. 목록을 읽지 못하면 가릴 자료가 없으므로 예외 종류와 사유만 출력한다.
    """
    try:
        return run_s3._scrub_text(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)), _s1_keys(a))
    except Exception:                                                          # noqa: BLE001 - nothing to scrub with: nothing is printed
        return (f'{type(exc).__name__}: the traceback is not printed, because the match keys of the S1 store could not be read to take '
                'them out of it (give the S1 run with --s1)\n')


def main(argv=None) -> int:
    """The command line. Exit code 0 when the report was written; 2 when it was refused (the message says why; nothing was written); 1 for
    an error that is not a refusal (its traceback is printed, with the match keys and game ids of the S1 store taken out of it).

    보고의 성공·거부·예상 밖 오류를 명령행 상태로 전달한다.
    """
    a = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    try:
        path = report(a.run, a.exclude_smoke, s1=a.s1, s3a=a.s3a, runs_root=a.runs_root, report_dir=a.report_dir)
    except (ReportInputError, KeyInReport) as exc:
        print(f'report refused: {exc}', file=sys.stderr, flush=True)
        return 2
    except Exception as exc:                                                   # noqa: BLE001 - the traceback, scrubbed, is the message
        print(scrubbed_traceback(a, exc), file=sys.stderr, flush=True)
        return 1
    doc = json.loads(path.read_text(encoding='utf-8'))
    print(f'S3 report written: {path} ({time.perf_counter() - t0:.1f} seconds)')
    for f in doc['flags']:
        print(f'FLAG {f["code"]}: {f["message"]}')
    print(f'gate_pass {doc["gate"]["gate_pass"]}; figures: {", ".join(doc["figures"])}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
