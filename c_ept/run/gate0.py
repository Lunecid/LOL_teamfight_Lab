"""Gate 0 probe of the EPT project (EPT-T013, author decision D31): does the EPT-S encoder predict engagement onset better than the matched feature baseline B1 on the same rows?

Question and material (task record ``harness/tasks/EPT-T013-gate0.md``)
-----------------------------------------------------------------------
No retraining. The material is the finished T009 matched-25% run (campaign ``be15ba1f...``, source commit 681a3cd): B1 (``fit/b1/model``), the EPT-S best epoch and the N2-TeamSet best
epoch (``fit/ept``, ``fit/n2``), the selected-FIT statistics and the whole-FIT S3 scales (``prepare/``). Every artifact is read only after its sha256 has been checked against the T009 receipts, by the
checks of ``comparison.runtime.load_neural_best`` without its ``execution_authorized`` flag (this probe's own start approval is recorded instead). No optimizer, last or RNG state is loaded.
The rows are the whole STOP split of the 15.20 patch, the primary population (t0 + 60 s <= T - 30 s, ``endpoints.primary_rows``, D14), the nominal exposure (T enters no forecast), the closed
form only (``monte_carlo=False``): DEV and TEST are never opened.

What is computed
----------------
``run_inference(..., role='STOP', exposure='nominal', population='primary', monte_carlo=False)`` once for EPT with B1 and once for N2 with B1, each as its own supervised stage with its own wall cap
and receipt, so that a cap hit on one does not hide the other's timing. From the paired endpoints ``M2/t0/H{60,120,180}/occurrence`` and the thirteen per-row NLL records the statistics below are made
on one sorted list of the STOP match ids and the first 2,000 rows of the common 10,000-row match-resample matrix (``evaluation.inference``, seed 20261022).

The supervised entry
--------------------
``supervise`` (command line ``main``) runs the probe the way the kernel runs a stage (task record, "Supervision"): E6 (no model server; Strata is off), at least 8,192 MiB of free GPU memory (``check_gpu_free``; other GPU users are no reason to refuse, the state is recorded), the child environment of
T009 (deterministic float32 GPU: ``CUBLAS_WORKSPACE_CONFIG``, 8 threads, device 0, the legacy rule folder of the T009 configuration), a 6 h wall budget, a new output folder
``D:/LOL_Data/runs/<YYYYMMDD>_c_ept_gate0_<n>``, the code identity of the import closure (``identity.capture``, start gate), each stage in its own child process owned by a Windows Job (``supervisor.run_owned``).
Before the output folder is made the parent binds the equivalence record to this commit and to the T009 source commit and checks the RUN.json record of T009 (``check_t009_record``): a refusal there is exit code 2 and
nothing is created. The EPT stage (with the matched B1) runs first and has at most 3 h; the N2 stage gets what is left of the 6 h after the cleanup reserve, at most 3 h (``_run_stage``: ``min(3 h, wall - elapsed
- 15 min)``), so it can have less than 3 h when the EPT stage was slow. A stage (``stage_main``) loads only its own model, B1 and the stores (``load_t009``), makes the numeric policy of T009 explicit and recorded
(``apply_numeric_policy``), evaluates (``evaluate_gate0``) and leaves its arrays and ``ARM.json``; the supervisor then makes ``GATE0.json``, ``GATE0_KO.md`` and keeps ``PROGRESS.jsonl`` (one begin and end line for each
head block) and one receipt for each stage. The start approval of the author and the record of the old-versus-new equivalence proof are inputs and are recorded.

What voids the verdict is what the primary statistic depends on: the EPT stage (its receipt, hashes, call settings, rows, numeric policy, production inputs, code identity and T009 configuration), the matched B1 of
both stages when both finished, and the equivalence proof and its binding to the two commits. A run whose inputs were not the production ones has the verdict VOID. The N2 arm is secondary: a cap hit, a crash or a
refusal of the N2 stage, or any failed check of the N2 arm itself, leaves the verdict standing, makes every N2 cell 'unavailable' with the reason and is listed as a warning (``integrity['warnings']``).

Symbols
-------
    m, r        a match and a row; n_m the number of primary rows of m; ids the sorted list of the STOP matches (every match, also one without a primary row)
    y_r         the occurrence outcome of the row (a first kill within the horizon); p^E_r, p^B_r, p^N_r the forecasts of EPT, B1 and N2 on the same row
    a_m, d_m    the per-match numerator (a mean over the rows of m, weight 1 / n_m) and the presence d_m = [n_m >= 1]
    c_{b,m}     how often the resample b of the common matrix draws the match m; B = 2,000 rows are used
    Delta       Brier(first model) - Brier(second model); below 0 favours the first model

Equations
---------
    (G0-1)  Delta = Brier(EPT) - Brier(B1) = sum_m a_m / sum_m d_m,   a_m = (1 / n_m) sum_{r in m} [ (p^E_r - y_r)^2 - (p^B_r - y_r)^2 ]
    (G0-2)  Delta*_b = sum_m c_{b,m} a_m / sum_m c_{b,m} d_m   (b = 1..2000),   ci95 = (q_0.025, q_0.975) of Delta*                      (percentile bootstrap over whole matches)
    (G0-3)  NO-GO  <=>  Delta >= 0;   GO  <=>  Delta < 0 and the upper end of ci95 < 0;   INCONCLUSIVE  <=>  Delta < 0 and not GO;   VOID  <=>  an integrity check that the primary statistic depends on
            fails, or the old-versus-new equivalence proof fails with differences that cannot be explained (no decision); a failure of the N2 arm alone is not a VOID (review I1, option a)
    (G0-4)  BSS_M = 1 - Brier(M) / Brier(base),   base = the constant forecast pbar = sum_m a'_m / sum_m d_m with a'_m the per-match mean of y_r (the STOP base rate of the primary rows)

해석
----
이 모듈은 학습을 하지 않는다. 이미 끝난 T009 25% 맞춤 비교의 체크포인트를 영수증의 해시로 확인한 뒤, 같은 STOP 행에서 EPT-S가 B1보다 60초 발생 Brier를 낮추는지만 본다(식 G0-1).
판정 규칙(G0-3)은 계산 전에 못 박은 것이고(과제 기록), 보조 결과는 판정을 바꾸지 않는다. 옛 스택·시드 하나·15.20 패치 안의 예측이므로 논문 주장을 뒷받침하지 않는 탐색적 가부 판단이다.
EPT와 N2는 서로 다른 감독 단계로 돌려, 한쪽이 벽시계 상한에 걸려도 다른 쪽의 시간이 남는다. EPT 단계가 먼저 돌고 N2 단계는 남은 시간(정리 여유 15분을 뺀 뒤, 최대 3시간)을 받는다.
N2는 보조 팔이므로 N2 단계가 상한에 걸리거나 실패해도 판정은 그대로이고 N2 칸만 '없음'과 이유로 남는다(검토 I1). 판정을 무효로 만드는 것은 주 통계가 기대는 점검뿐이다.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from c_ept.comparison import models as comparison_models
from c_ept.comparison import protocol as comparison_protocol
from c_ept.comparison import runtime as comparison_runtime
from c_ept.evaluation import guard as evaluation_guard
from c_ept.evaluation import inference
from c_ept.run import guards, supervisor
from c_ept.run import identity as run_identity
from c_ept.run import io as run_io

PRIMARY_ENDPOINT = 'M2/t0/H60/occurrence'                                     # the 60 s onset probability: the one primary statistic
HORIZONS = (60, 120, 180)                                                     # the horizons of the M2 occurrence endpoints (s)
BOOTSTRAP_B = 2000                                                            # the first 2,000 rows of the common 10,000-row match-resample matrix (D13 C4)
CONTRASTS = (('ept_minus_b1', 'ept', 'b1'), ('n2_minus_b1', 'n2', 'b1'), ('ept_minus_n2', 'ept', 'n2'))      # (name, first model, second model); Delta = first - second
NLL_NOTE = ('per head NLL differences (first minus second) on the same rows, one row per head, in nats per row of the head (the HR-5 sum over the bins or teams of an evaluation row), balanced '
            'over matches with weight 1/n_{h,m}; no cross-head average and no scale is applied')
VERDICTS = ('GO', 'INCONCLUSIVE', 'NO-GO', 'VOID')
EQUIVALENCE_STATUSES = ('passed', 'differences_explained', 'failed')          # what the recorded old-versus-new equivalence proof may say
EQUIVALENCE_DECIDING = ('passed', 'differences_explained')                    # the statuses under which a run is decided (a proof that failed with differences that cannot be explained is VOID)

# The four texts of the decision rule, fixed 2026-10-26 before any number was computed (task record EPT-T013). The D13 C4 rule on one seed.
RULES = {
    'GO': 'Delta < 0 and the upper end of the interval < 0. S4b preparation resumes (K6c, K8, K9, X1), and speed optimisation becomes worthwhile.',
    'INCONCLUSIVE': 'Delta < 0 and the interval contains 0. Report to the author, who decides whether to proceed with an "inconclusive" flag.',
    'NO-GO': 'Delta >= 0. Engineering stays paused. Report to the author, who decides the direction: a paper centred on the process decomposition with B1 heads (spec 8: "if it does not win, '
             'B1 is the main result"), or a redesign of the encoder.',
    'VOID': 'No decision: any integrity check fails, or the old-versus-new equivalence proof fails and its differences cannot be explained.'}


def _number(value) -> bool:
    """True for a real number that is not a bool and is finite."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def decide(result: Mapping) -> dict:
    """The verdict of the pre-registered rule (G0-3) on a result, and nothing else: ``verdict`` (one of ``VERDICTS``), ``rule_text`` (the text of ``RULES`` that applies), ``void_reasons`` (why there is no
    decision, empty otherwise), ``notes`` and the two numbers the rule read. ``result`` is read, never changed.

    The result must hold ``integrity`` (``{'passed': True, 'failures': []}``), ``equivalence`` (``{'status': ...}``, one of ``EQUIVALENCE_STATUSES``) and ``primary`` (``{'delta': float, 'ci95': [lo, hi]}``, the
    Brier difference EPT - B1 on ``M2/t0/H60/occurrence`` and its 95% match-bootstrap interval). Nothing else is read: the secondary results never overturn the primary rule (task record).
    VOID: an integrity check of the primary statistic that failed or was not run (``assemble`` puts only those in ``integrity['failures']``: a failed secondary check is a warning there), an equivalence status that is not a deciding one, a primary statistic that is missing, not finite, or an interval that is not an ordered pair.
    Otherwise: Delta >= 0 is NO-GO (Delta = 0 too, whatever the interval says); Delta < 0 is GO when the upper end of the interval is below 0 and INCONCLUSIVE when it is not (an interval that touches 0 contains it).
    A Delta below 0 with an interval wholly above 0 has no branch in the rule; as in the C4 code of D13 it is not a GO and is INCONCLUSIVE, with a note.

    수식: (G0-3).
    해석: 계산 전에 못 박은 규칙을 그대로 옮겼다. 주 통계가 기대는 무결성 점검이나 동등성 증명이 실패했거나 주 통계가 없거나 깨졌으면 판정을 내리지 않는다(VOID). 차이가 0 이상이면 구간과 상관없이 NO-GO이고, 0보다 작고
    구간의 위쪽 끝이 0보다 작으면 GO이다. 구간이 0에 닿으면 0을 품은 것으로 본다(INCONCLUSIVE). 보조 결과(EPT - N2 등)는 읽지 않으므로 판정을 뒤집을 수 없다.
    """
    reasons, notes = [], []
    integrity = result.get('integrity') if isinstance(result, Mapping) else None
    if not isinstance(integrity, Mapping) or integrity.get('passed') is not True:
        failures = integrity.get('failures') if isinstance(integrity, Mapping) else None
        reasons.append('an integrity check failed or was not run' + (f': {"; ".join(str(f) for f in failures)}' if isinstance(failures, (list, tuple)) and failures else ''))
    elif integrity.get('failures'):
        reasons.append('the integrity record says passed but lists failures: ' + '; '.join(str(f) for f in integrity['failures']))
    equivalence = result.get('equivalence') if isinstance(result, Mapping) else None
    status = equivalence.get('status') if isinstance(equivalence, Mapping) else None
    if status not in EQUIVALENCE_DECIDING:
        reasons.append(f'the old-versus-new equivalence proof is not recorded as passed or explained (status {status!r})')
    primary = result.get('primary') if isinstance(result, Mapping) else None
    delta = primary.get('delta') if isinstance(primary, Mapping) else None
    interval = primary.get('ci95') if isinstance(primary, Mapping) else None
    if not _number(delta):
        reasons.append('the primary statistic (the Brier difference EPT - B1) is missing or not a finite number')
    if not (isinstance(interval, (list, tuple)) and len(interval) == 2 and all(_number(v) for v in interval) and interval[0] <= interval[1]):
        reasons.append('the interval of the primary statistic is missing, not finite or not an ordered pair')
    if reasons:
        return {'verdict': 'VOID', 'rule_text': RULES['VOID'], 'void_reasons': reasons, 'notes': notes, 'delta': None, 'ci95': None}
    low, high = float(interval[0]), float(interval[1])
    delta = float(delta)
    if delta >= 0:
        verdict = 'NO-GO'
    elif high < 0:
        verdict = 'GO'
    else:
        verdict = 'INCONCLUSIVE'
        if low > 0:
            notes.append('the point estimate is below 0 but the interval lies wholly above 0 (the rule has no branch for it; read as not a GO)')
    return {'verdict': verdict, 'rule_text': RULES[verdict], 'void_reasons': [], 'notes': notes, 'delta': delta, 'ci95': [low, high]}


# ---- the statistics -------------------------------------------------------------------------------------------------------------------------------
def _check_matrix(matrix, ids) -> np.ndarray:
    """``matrix`` as an array, ValueError unless it is a non-empty (B, n_matches) matrix with one column for each match of ``ids`` (the sorted, unique list of the STOP matches).

    수식: matrix[b, j] = c_{b,m_j}, 재표본 b가 j번째 경기 m_j를 뽑은 횟수이므로 열의 수는 경기 목록의 길이와 같아야 하고 행(B)은 1 이상이다.
    해석: 재표본 행렬의 열은 공통 STOP 경기 목록의 경기 하나씩에 맞는다. 모양이 어긋난 행렬은 엉뚱한 경기에 가중치를 주게 되므로 계산 전에 거부한다."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[1] != len(ids):
        raise ValueError(f'matrix must have the shape (B >= 1, {len(ids)}), not {matrix.shape}')
    return matrix


def _interval(numerator, presence, point, matrix) -> dict:
    """The percentile interval of the ratio estimator on the rows of ``matrix`` (G0-2), or why there is none: a resample that draws no present match has no ratio and the interval is then missing, never
    filled (the same rule as ``metrics.relative_structural`` and the C4 statistic of D13).

    수식: (G0-2)  Delta*_b = sum_m c_{b,m} a_m / sum_m c_{b,m} d_m,   분모 sum_m c_{b,m} d_m <= 0 이면 Delta*_b는 정의되지 않는다. ci95 = (q_0.025, q_0.975) of {Delta*_b}.
    해석: 재표본 하나가 주 모집단 행이 있는 경기를 하나도 뽑지 못하면 그 재표본의 비율은 0 나누기 0이다. 이를 0으로 채우면 구간이 아래로 쏠리므로, 정의되지 않은 재표본이 하나라도 있으면
    구간을 만들지 않고 그 수(draws_undefined)와 이유를 적는다. 판정 규칙은 구간이 없으면 판정을 내리지 않는다(VOID)."""
    denominator = matrix @ presence
    undefined = int((denominator <= 0).sum())
    out = {'bootstrap_B': int(matrix.shape[0]), 'draws_undefined': undefined}
    if undefined:
        return {**out, 'ci95': None, 'ci_status': 'undefined_resamples', 'p': None}
    draws = inference.bootstrap_stat(numerator, presence, matrix)
    return {**out, 'ci95': [float(v) for v in np.quantile(draws, [0.025, 0.975])], 'ci_status': 'available', 'p': float(inference.two_sided_p(draws, point))}


def _ratio(per_match_numerator, presence) -> float:
    """sum_m a_m / sum_m d_m, the point estimate (the resample matrix of ones).

    수식: Delta = sum_m a_m / sum_m d_m (G0-1), 재표본 행렬을 모든 경기를 한 번씩 뽑은 한 행(c_m = 1)으로 둔 비율이다.
    해석: 점추정은 부트스트랩과 같은 식을 쓰되 경기를 그대로 한 번씩 센 것이므로, 구간의 가운데와 같은 정의로 계산된다."""
    return float(inference.bootstrap_stat(per_match_numerator, presence, np.ones((1, len(presence))))[0])


def brier_contrast(match, y, p_first, p_second, ids, matrix) -> dict:
    """Delta = Brier(first) - Brier(second) of two forecasts of one binary outcome on the same rows (G0-1), balanced over matches (weight 1/n_m per row) on the common list ``ids``, with the percentile
    interval of the ratio estimator on the rows of ``matrix`` (G0-2). ValueError for a probability outside [0, 1], an outcome that is not 0 or 1, vectors of different lengths and a row of a match that
    is not in ``ids``.

    수식
    ----
    (G0-1)  Delta = sum_m a_m / sum_m d_m,   a_m = (1/n_m) sum_{r in m} [ (p1_r - y_r)^2 - (p2_r - y_r)^2 ],   d_m = [n_m >= 1]
    (G0-2)  Delta*_b = sum_m c_{b,m} a_m / sum_m c_{b,m} d_m,   ci95 = (q_0.025, q_0.975) of Delta*

      기호      | 뜻                                                 | 단위  | 범위       | 코드 변수
      p1, p2    | 두 예측(앞 모형, 뒤 모형)의 발생 확률               | -     | [0, 1]     | p_first, p_second
      a_m, d_m  | 경기별 분자(경기 안 평균)와 있음 표지               | 확률^2| d in {0,1} | per-match arrays
      c_{b,m}   | 재표본 b가 경기 m을 뽑은 횟수                       | 개    | 0..n       | matrix

    해석
    ----
    경기 안 행 평균(가중 1/n_m)을 경기별 분자로, 행이 있는 경기를 세는 표지를 분모로 하는 비 추정량이다. 행이 하나도 없는 경기는 공통 목록에는 있으나 분자·분모가 모두 0이어서 빠진다. 두 Brier가 같은 행·같은
    가중치에서 나오므로 차이는 두 Brier의 차와 같다. 재표본마다 분자와 분모를 함께 다시 합하며, 어떤 재표본이 있는 경기를 하나도 못 뽑으면 그 비는 정의되지 않으므로 구간을 비워 두고 이유를 적는다.
    """
    match = np.asarray(match)
    if match.ndim != 1 or not len(match):
        raise ValueError('match must be a non-empty vector of match numbers')
    n = len(match)
    arrays = {}
    for name, values in (('y', y), ('p_first', p_first), ('p_second', p_second)):
        arrays[name] = np.asarray(values, dtype=np.float64)
        if arrays[name].shape != (n,):
            raise ValueError(f'{name} must be a vector with one entry for each of the {n} rows, not {arrays[name].shape}')
    if not np.isin(arrays['y'], (0.0, 1.0)).all():
        raise ValueError('y must be binary (0 or 1)')
    for name in ('p_first', 'p_second'):
        if not np.isfinite(arrays[name]).all() or ((arrays[name] < 0) | (arrays[name] > 1)).any():
            raise ValueError(f'{name} holds a value that is not a probability')
    matrix = _check_matrix(matrix, ids)
    present = inference.match_presence(match, ids)                                         # ValueError for a row of a match outside ids
    squared = {name: (arrays[name] - arrays['y']) ** 2 for name in ('p_first', 'p_second')}
    per_match = lambda values: inference.match_aggregate(match, values, ids)[1]            # (1/n_m) sum over the rows of m
    numerator = per_match(squared['p_first'] - squared['p_second'])
    point = _ratio(numerator, present)
    return {'delta': point, 'brier_first': _ratio(per_match(squared['p_first']), present), 'brier_second': _ratio(per_match(squared['p_second']), present),
            'n_matches': int(len(ids)), 'n_present_matches': int(present.sum()), 'n_rows': int(n), **_interval(numerator, present, point, matrix)}


def brier_skill(match, y, p, ids) -> dict:
    """The Brier skill of a forecast against the base rate of the same rows (G0-4): ``base_rate`` (the match-balanced mean of the outcome), ``brier_base`` (the Brier of the constant forecast of
    the base rate), ``brier_model`` and ``skill`` = 1 - brier_model / brier_base (None when the base Brier is 0, all outcomes equal). A point value only: the base rate is the STOP rate of the primary
    rows itself, so the skill is not held out.

    수식: (G0-4) BSS = 1 - Brier(M) / Brier(base),  pbar = sum_m a'_m / sum_m d_m,  a'_m = (1/n_m) sum_{r in m} y_r.
    해석: 기준 예측은 STOP 주 모집단의 발생률(경기 균형 평균)을 모든 행에 똑같이 내는 것이다. 모형의 Brier를 이 기준의 Brier로 나눈 비를 1에서 뺀 값이 기량 점수이며 0이면 기준과 같다. 기준률도 같은 행에서 구했으므로
    표본 밖 값이 아니다.
    """
    match = np.asarray(match)
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if match.ndim != 1 or not len(match) or y.shape != match.shape or p.shape != match.shape:
        raise ValueError('match, y and p must be non-empty vectors of one length')
    if not np.isin(y, (0.0, 1.0)).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('y must be binary and p a probability')
    present = inference.match_presence(match, ids)
    per_match = lambda values: inference.match_aggregate(match, values, ids)[1]
    rate = _ratio(per_match(y), present)
    base, model = _ratio(per_match((rate - y) ** 2), present), _ratio(per_match((p - y) ** 2), present)
    return {'base_rate': rate, 'brier_base': base, 'brier_model': model, 'skill': float(1.0 - model / base) if base > 0 else None}


def nll_contrast(match, row, first, second, ids, matrix, *, reason=None) -> dict:
    """The difference of two per-row NLLs of one head on the same rows, first minus second, balanced over matches (weight 1/n_{h,m}) with the interval of the ratio estimator on ``matrix``: ``delta``,
    ``ci95``, ``mean_first`` and ``mean_second`` (the balanced mean NLL of each), ``n_rows``, ``n_matches_present``, ``status`` ('available', or 'unavailable' with ``reason`` for a head with no row).
    ValueError for a value that is not finite or a row that appears twice. There is no average over heads: the caller makes one row per head.

    수식: Delta_h = sum_m a_{h,m} / sum_m d_{h,m},  a_{h,m} = (1/n_{h,m}) sum_{r in m} (l1_r - l2_r)  (the same ratio estimator as G0-1 with the NLL difference of the row in place of the squared-error difference).
    해석: 머리마다 같은 행에서 두 모형의 NLL 차이를 경기 균형 평균으로 낸다. NLL은 머리마다 척도와 행 단위가 달라 머리 사이 평균을 내지 않는다.
    """
    match = np.asarray(match)
    matrix = _check_matrix(matrix, ids)
    if match.ndim != 1:
        raise ValueError('match must be a vector')
    if not len(match):
        return {'status': 'unavailable', 'reason': reason or 'no_rows', 'delta': None, 'ci95': None, 'ci_status': None, 'p': None, 'mean_first': None, 'mean_second': None,
                'n_rows': 0, 'n_matches_present': 0, 'bootstrap_B': int(matrix.shape[0]), 'draws_undefined': 0}
    row, first, second = np.asarray(row), np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if row.shape != match.shape or first.shape != match.shape or second.shape != match.shape:
        raise ValueError('match, row and the two NLL vectors must have one length')
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError('nonfinite NLL')
    if len(np.unique(row)) != len(row):
        raise ValueError('one evaluation row appears twice in a head')
    present = inference.match_presence(match, ids)
    per_match = lambda values: inference.match_aggregate(match, values, ids)[1]
    numerator = per_match(first - second)
    point = _ratio(numerator, present)
    return {'status': 'available', 'reason': None, 'delta': point, 'mean_first': _ratio(per_match(first), present), 'mean_second': _ratio(per_match(second), present),
            'n_rows': int(len(match)), 'n_matches_present': int(present.sum()), **_interval(numerator, present, point, matrix)}


def _same(a, b) -> bool:
    return np.asarray(a).shape == np.asarray(b).shape and bool(np.array_equal(a, b))


def _unavailable(reason: str) -> dict:
    return {'status': 'unavailable', 'reason': reason}


def _endpoint_of(arms, model, name):
    """The endpoint ``name`` of the arm of ``model`` ('ept' or 'n2'), or None when the arm or the endpoint is not there."""
    return arms[model]['endpoints'].get(name) if model in arms else None


def _brier_cell(arms, first, second, horizon, ids, matrix) -> dict:
    """One cell of the Brier contrast ``first - second`` at ``horizon``: against B1 each arm gives its own model and its own B1 probabilities (the same B1, checked by ``check_arms``); EPT - N2 needs both
    arms on the same rows.

    수식: Delta_{첫째-둘째}(H) = Brier(첫째) - Brier(둘째) on M2/t0/H{H}/occurrence (G0-1); B1이 둘째이면 그 팔이 낸 B1 확률을, N2 대 EPT이면 두 팔의 모형 확률을 같은 행에서 비교한다.
    해석: 한 칸은 한 대조의 한 지평이다. 필요한 팔이 끝나지 못했거나 두 팔의 행(경기, 행, 결과)이 다르면 값을 지어내지 않고 '없음'과 이유를 적는다. N2 칸이 없어도 EPT − B1 칸은 그대로 있다."""
    name = f'M2/t0/H{horizon}/occurrence'
    found = {model: _endpoint_of(arms, model, name) for model in ('ept', 'n2') if model in (first, second)}
    missing = [model for model, endpoint in found.items() if endpoint is None]
    if missing:
        return _unavailable('the ' + ' and '.join(missing) + ' forecasts of ' + name + ' are not there (the stage did not finish or the arm failed a check: see the integrity checks)')
    base = found[first]
    if second == 'b1':
        return {'status': 'available', **brier_contrast(base['match'], base['y'], base['p_model'], base['p_b1'], ids, matrix)}
    other = found[second]
    if not all(_same(base[key], other[key]) for key in ('match', 'row', 'y')):
        return _unavailable('the two models are not on the same rows (match, row and outcome differ)')
    return {'status': 'available', **brier_contrast(base['match'], base['y'], base['p_model'], other['p_model'], ids, matrix)}


def _skill_cell(arms, model, horizon, ids) -> dict:
    """One cell of the Brier skill of ``model`` ('ept', 'n2' or 'b1') at ``horizon`` (G0-4); B1 is read from whichever arm is there (both hold the same B1 column), 'unavailable' when no arm has the endpoint.

    수식: BSS_M = 1 - Brier(M) / Brier(base),   base = 같은 행의 경기 균형 발생률 pbar를 모든 행에 내는 상수 예측 (G0-4).
    해석: 모형이 상수 예측보다 얼마나 나은지(표본 밖 값이 아님). B1은 두 팔이 같은 값을 가지므로 있는 팔에서 읽고, 팔이 없으면 '없음'과 이유를 적는다."""
    name = f'M2/t0/H{horizon}/occurrence'
    arm_name = next((key for key in ((model,) if model != 'b1' else ('ept', 'n2')) if _endpoint_of(arms, key, name) is not None), None)
    if arm_name is None:
        return _unavailable(f'the {model} forecasts of {name} are not there (the stage did not finish or the arm failed a check: see the integrity checks)')
    endpoint = arms[arm_name]['endpoints'][name]
    return {'status': 'available', **brier_skill(endpoint['match'], endpoint['y'], endpoint['p_b1' if model == 'b1' else 'p_model'], ids)}


def _heads_of(arms) -> list:
    """The names of the NLL records of the arms in the order the first arm gives them (the order of the thirteen blocks of a run)."""
    heads = []
    for key in ('ept', 'n2'):
        for head in arms.get(key, {}).get('nll', {}):
            if head not in heads:
                heads.append(head)
    return heads


def _nll_cell(arms, first, second, head, ids, matrix) -> dict:
    """One cell of the NLL contrast ``first - second`` of a head: against B1 the arm's own two columns, EPT - N2 the two arms' model columns on the same rows.

    수식: Delta_NLL(head) = sum_m a_m / sum_m d_m, a_m = (1/n_m) sum_{r in m} [ nll1_r - nll2_r ] (G0-1과 같은 경기 균형, 단위는 머리 한 행의 nat), 머리마다 따로 한 줄이다.
    해석: 머리마다 같은 행의 NLL 차이를 본다. 머리 사이는 평균하지 않는다(단위와 행 수가 다르다). 필요한 팔이 없거나 두 팔의 행이 다르면 '없음'과 이유를 적는다."""
    records = {model: arms[model]['nll'].get(head) if model in arms else None for model in ('ept', 'n2') if model in (first, second)}
    missing = [model for model, record in records.items() if record is None]
    if missing:
        return _unavailable('the ' + ' and '.join(missing) + f' NLL of {head} is not there (the stage did not finish or the arm failed a check: see the integrity checks)')
    one = records[first]
    if second == 'b1':
        return nll_contrast(one['match'], one['row'], one['nll_model'], one['nll_b1'], ids, matrix, reason=one.get('reason'))
    two = records[second]
    if not (_same(one['match'], two['match']) and _same(one['row'], two['row'])):
        return _unavailable('the two models are not scored on the same rows of this head')
    return nll_contrast(one['match'], one['row'], one['nll_model'], two['nll_model'], ids, matrix, reason=one.get('reason'))

STRUCTURAL_UNAVAILABLE = ('the relative structural metric (D16) needs B0 predictions on the same rows; the T009 run holds no B0 and none is fitted by this probe, so the metric is not made')


def analyze(arms: Mapping, ids, matrix) -> dict:
    """The statistics of the probe from the finished arms: ``primary`` (the Brier difference EPT - B1 of ``M2/t0/H60/occurrence`` with its interval and the rows per match, None when the EPT arm is not there)
    and ``secondary`` (never decisive): ``brier`` (the three contrasts of the task record, EPT - B1, N2 - B1 and EPT - N2, at the horizons 60, 120 and 180 s), ``brier_skill`` (each of EPT, N2 and B1 against the
    base rate), ``nll_per_head`` (the same three contrasts for every head, one row per head, ``nll_note``) and ``structural_relative`` (unavailable: no B0).

    ``arms``: ``{'ept': arm, 'n2': arm}`` (either may be missing: what needs it is then 'unavailable' with the reason) where an arm is ``{'name', 'endpoints': {name: {match, row, y, p_model, p_b1}}, 'nll':
    {head: {match, row, nll_model, nll_b1, reason}}}`` (``arm_from_inference``). ``ids``: the one sorted list of the STOP matches; ``matrix``: the first rows of the common resample matrix.

    해석: 판정에 쓰이는 것은 primary 하나이고 나머지는 저자가 판단할 때 읽는 참고 자료이다. 한 팔이 끝나지 못하면 그 팔이 필요한 칸은 '없음'과 이유로 남는다.
    """
    ids, matrix = np.asarray(ids), None if matrix is None else np.asarray(matrix)
    matrix = _check_matrix(matrix, ids)
    brier = {name: {f'H{h}': _brier_cell(arms, first, second, h, ids, matrix) for h in HORIZONS} for name, first, second in CONTRASTS}
    primary = None
    cell = brier['ept_minus_b1']['H60']
    if cell['status'] == 'available':
        endpoint = arms['ept']['endpoints'][PRIMARY_ENDPOINT]
        counts = np.bincount(np.searchsorted(ids, np.asarray(endpoint['match'])), minlength=len(ids))
        primary = {**cell, 'endpoint': PRIMARY_ENDPOINT, 'contrast': 'ept_minus_b1', 'metric': 'brier', 'direction': 'Brier(EPT) - Brier(B1); below 0 favours the EPT',
                   'rows_per_match': [int(v) for v in counts]}
    skill = {model: {f'H{h}': _skill_cell(arms, model, h, ids) for h in HORIZONS} for model in ('ept', 'n2', 'b1')}
    heads = _heads_of(arms)
    nll = {name: {head: _nll_cell(arms, first, second, head, ids, matrix) for head in heads} for name, first, second in CONTRASTS}
    return {'primary': primary, 'secondary': {'brier': brier, 'brier_skill': skill, 'nll_per_head': nll, 'nll_note': NLL_NOTE, 'structural_relative': _unavailable(STRUCTURAL_UNAVAILABLE)}}


EXPECTED_PROVENANCE = {'role': 'STOP', 'exposure': 'nominal', 'population': 'primary', 'monte_carlo': False, 'draws': 0, 'frame_averaged': True}
ABSENT_PROVENANCE = ('kill_law', 'k_condition', 'eval_average', 'zone_mark', 'winner_truth', 'gold_reps', 'b1_swap_average', 'b0', 'seed', 'rng', 'mc_uncertainty')


def check_arm(arm: Mapping) -> list:
    """The integrity checks of one finished arm, from what ``run_inference`` says about its own call: the role STOP, the nominal exposure, the primary population, no Monte Carlo (draws 0, no seed, stream or
    sampling error), the two-frame average of the EPT, and none of the settings that the plain call does not have (kill law, K condition, zone mark, winner truth, representatives, B1 swap average, B0). Each
    check is ``{'name', 'passed', 'detail'}``; the three M2 endpoints have rows.

    해석: 호출이 스스로 밝힌 설정이 이 탐침이 요구한 것과 같은지, 그리고 세 지평의 끝점에 행이 있는지 본다."""
    provenance = arm['provenance']
    bad = [f'{key}={provenance.get(key)!r}' for key, wanted in EXPECTED_PROVENANCE.items() if provenance.get(key) != wanted or type(provenance.get(key)) is not type(wanted)]
    bad += [f'{key} is present' for key in ABSENT_PROVENANCE if key in provenance]
    empty = [name for name in M2_ENDPOINTS if not len(arm['endpoints'][name]['row'])]
    return [{'name': f'{arm["name"]}_call_settings', 'passed': not bad, 'detail': 'the call was the closed-form STOP call on the primary rows with the nominal exposure' if not bad else 'the call differs: ' + '; '.join(bad)},
            {'name': f'{arm["name"]}_rows_present', 'passed': not empty, 'detail': 'the three M2 occurrence endpoints have rows' if not empty else 'no row in ' + ', '.join(empty)}]


def check_arms(arms: Mapping) -> list:
    """The integrity checks that compare two finished arms: ``common_id_list_equal`` (both arms were aggregated on the same sorted list of STOP matches), ``arms_on_the_same_rows`` (every endpoint of
    EPT and N2 has the same match, row and outcome arrays) and ``b1_probabilities_equal`` (the one matched B1 gives the same probability on every endpoint row and the same NLL on every head row; it is
    compared on the endpoints and heads whose rows are the same, since B1 values of different rows are not comparable). With fewer than two arms nothing is compared and the list is empty. Each check is
    ``{'name', 'passed', 'detail'}``; ``assemble`` gives the first two the scope 'secondary' (a mismatch drops the N2 rows only) and the B1 check the scope 'primary' (a B1 that differs between the stages
    casts doubt on the B1 of the EPT stage too).

    해석: 두 모형이 같은 행에서 채점되었고 두 팔이 같은 B1 값을 받았는지 본다. 행이 다르면 N2 칸만 못 쓰고, 같은 행에서 B1 값이 다르면 EPT 쪽 B1도 믿을 수 없으므로 판정은 VOID이다.
    """
    if not all(key in arms for key in ('ept', 'n2')):
        return []
    one, two = arms['ept'], arms['n2']
    ids_ok = _same(one['ids'], two['ids'])
    rows_ok, rows_bad = True, []
    b1_ok, b1_bad = True, []
    for name in sorted(set(one['endpoints']) | set(two['endpoints'])):
        a, b = one['endpoints'].get(name), two['endpoints'].get(name)
        if a is None or b is None or not all(_same(a[key], b[key]) for key in ('match', 'row', 'y')):
            rows_ok = False
            rows_bad.append(name)
        elif not _same(a['p_b1'], b['p_b1']):
            b1_ok = False
            b1_bad.append(name)
    for head in sorted(set(one['nll']) | set(two['nll'])):
        a, b = one['nll'].get(head), two['nll'].get(head)
        if a is None or b is None or not (_same(a['match'], b['match']) and _same(a['row'], b['row'])):
            rows_ok = False
            rows_bad.append(head)
        elif not _same(a['nll_b1'], b['nll_b1']):
            b1_ok = False
            b1_bad.append(head)
    return [{'name': 'common_id_list_equal', 'passed': ids_ok, 'detail': 'both arms use the same sorted list of STOP matches' if ids_ok else 'the two arms were aggregated on different lists of STOP matches'},
            {'name': 'arms_on_the_same_rows', 'passed': rows_ok, 'detail': 'EPT and N2 are scored on the same rows of every endpoint and head' if rows_ok else 'rows differ in ' + ', '.join(rows_bad)},
            {'name': 'b1_probabilities_equal', 'passed': b1_ok, 'detail': 'the matched B1 gives the same values in both arms' if b1_ok else 'B1 values differ in ' + ', '.join(b1_bad)}]


# ---- one arm: what a stage keeps of ``run_inference`` -----------------------------------------------------------------------------------------------
class Gate0Refused(run_io.IORefused):
    """The probe refuses to go on: a receipt, a hash, a setting or a result is not what the pre-registration needs. An ``IORefused``, so a stage that raises it exits with code 2 (a refusal: nothing was
    computed that is lost) and the supervisor does not call it a failure of the computation. The message names the check and never a key or a path of the data."""


def _require(ok, message: str) -> None:
    if not ok:
        raise Gate0Refused(message)


def _plain(value):
    """A strict-JSON copy of ``value``: numpy scalars and arrays become numbers and lists, mappings and sequences are copied (the keys become texts)."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    return value


M2_ENDPOINTS = tuple(f'M2/t0/H{h}/occurrence' for h in HORIZONS)
ARM_SCHEMA = 'ept-gate0-arm-v1'
ARRAYS_FILE = 'ARRAYS.npz'


def arm_from_inference(name: str, result: Mapping, ids) -> dict:
    """The arm of the model ``name`` ('ept' or 'n2') from a ``run_inference`` result: the three M2 occurrence endpoints (``match``, ``row``, ``y``, ``p_model`` the model's probability, ``p_b1``, ``counts``)
    and every NLL record (``match``, ``row``, ``nll_model``, ``nll_b1``, ``reason``, ``counts``), the sorted list of the STOP matches ``ids`` that the statistics are aggregated on, and the provenance of the
    call without its ``role_note`` (this probe states the selection itself: both arms stopped early on the 401-match STOP subset of T009, the EPT through its STOP loss and B1 through ``fit_head(fit_rows,
    stop_rows)``). ValueError when an M2 endpoint is missing or ``ids`` is not a sorted, unique vector.

    해석: 판정에 필요한 배열만 남긴다. B1이 쓴 조기 종료 행에 대한 문구는 호출이 볼 수 없는 것을 말하므로 옮기지 않고, 이 모듈이 직접 적는다.
    """
    by_name = {endpoint['name']: endpoint for endpoint in result['endpoints']}
    missing = [endpoint for endpoint in M2_ENDPOINTS if endpoint not in by_name]
    if missing:
        raise ValueError(f'the inference result has no endpoint {missing}')
    endpoints = {key: {'match': np.asarray(by_name[key]['match']), 'row': np.asarray(by_name[key]['row']), 'y': np.asarray(by_name[key]['y']),
                       'p_model': np.asarray(by_name[key]['p_ept']), 'p_b1': np.asarray(by_name[key]['p_b1']), 'counts': _plain(by_name[key]['counts'])} for key in M2_ENDPOINTS}
    nll = {record['name']: {'match': np.asarray(record['match']), 'row': np.asarray(record['row']), 'nll_model': np.asarray(record['nll_ept']), 'nll_b1': np.asarray(record['nll_b1']),
                            'reason': (record.get('counts') or {}).get('unavailable_reason'), 'counts': _plain(record.get('counts') or {})} for record in result['nll']}
    provenance = {key: value for key, value in result.get('provenance', {}).items() if key != 'role_note'}
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or (len(ids) > 1 and not (ids[1:] > ids[:-1]).all()):
        raise ValueError('ids must be a non-empty, sorted, unique vector of match numbers')
    return {'name': name, 'endpoints': endpoints, 'nll': nll, 'ids': ids, 'provenance': _plain(provenance)}


def save_arm(folder, arm: Mapping) -> dict:
    """Write the arrays of ``arm`` to ``folder/ARRAYS.npz`` (a new file: FileExistsError otherwise) and return the manifest that a stage records: ``{'arrays_file', 'arrays_sha256', 'endpoints': {name:
    {'arrays': {field: key}, 'counts'}}, 'nll': {head: {'arrays': {field: key}, 'reason', 'counts'}}}``. The arrays are private local data (rows of the STOP split) and never enter a report.

    해석: 행 단위 배열은 한 파일에 모아 해시를 남긴다. 보고서에는 이 배열이 들어가지 않고 통계만 들어간다.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays, manifest = {'ids': np.asarray(arm['ids'], dtype=np.int64)}, {'arrays_file': ARRAYS_FILE, 'endpoints': {}, 'nll': {}, 'ids': {'key': 'ids', 'n': int(len(arm['ids'])), 'sha256': _sha_of_array(np.asarray(arm['ids'], dtype='<i8'))}}
    for group, fields in (('endpoints', ('match', 'row', 'y', 'p_model', 'p_b1')), ('nll', ('match', 'row', 'nll_model', 'nll_b1'))):
        for index, (key, entry) in enumerate(arm[group].items()):
            names = {field: f'{group[0]}{index:02d}_{field}' for field in fields}
            for field, array_key in names.items():
                arrays[array_key] = np.asarray(entry[field])
            manifest[group][key] = {'arrays': names, 'counts': _plain(entry['counts']), **({'reason': entry.get('reason')} if group == 'nll' else {})}
    path = folder / ARRAYS_FILE
    with open(path, 'xb') as stream:
        np.savez(stream, **arrays)
    manifest['arrays_sha256'] = run_io.sha_file(path)
    return manifest


def load_arm(folder, name: str, manifest: Mapping, provenance: Mapping | None = None) -> dict:
    """The arm saved by ``save_arm``; ``Gate0Refused`` when the arrays file is not the one the manifest names or its hash differs (the file is hashed before one array is read)."""
    path = Path(folder) / str(manifest.get('arrays_file'))
    _require(path.name == ARRAYS_FILE and path.is_file() and not path.is_symlink(), f'the arrays file of the {name} arm is missing')
    _require(run_io.sha_file(path) == manifest.get('arrays_sha256'), f'the arrays file of the {name} arm changed (its sha256 differs from the record)')
    arm = {'name': name, 'endpoints': {}, 'nll': {}, 'provenance': dict(provenance or {})}
    with np.load(path, allow_pickle=False) as stored:
        arm['ids'] = stored[manifest['ids']['key']]
        _require(_sha_of_array(arm['ids'].astype('<i8')) == manifest['ids']['sha256'] and len(arm['ids']) == manifest['ids']['n'], f'the id list of the {name} arm differs from its record')
        for key, entry in manifest['endpoints'].items():
            arm['endpoints'][key] = {**{field: stored[array_key] for field, array_key in entry['arrays'].items()}, 'counts': dict(entry['counts'])}
        for key, entry in manifest['nll'].items():
            arm['nll'][key] = {**{field: stored[array_key] for field, array_key in entry['arrays'].items()}, 'reason': entry.get('reason'), 'counts': dict(entry['counts'])}
    return arm


# ---- the numeric policy -----------------------------------------------------------------------------------------------------------------------------
def _probe(function):
    """``function()``, or None when it cannot answer (a process without a visible GPU: torch raises on the cuDNN version or the device name there). A value that is not there is recorded as None."""
    try:
        return function()
    except Exception:                                                           # noqa: BLE001 - whatever torch does without a device means "not available here"
        return None


def numeric_record() -> dict:
    """What the process settings are now, as plain JSON (perf-audit.md section 8 item 3: the EPT path never records them, so Gate 0 does): the default dtype, the float32 matmul precision, the TF32 flags of
    cuBLAS and cuDNN, the cuDNN deterministic and benchmark flags, whether deterministic algorithms are enforced, whether autocast is on (measured: ``torch.is_autocast_enabled`` for CUDA and CPU),
    the scaled-dot-product-attention backend flags, the cuBLAS workspace setting, the torch, CUDA and cuDNN versions, the thread count and the device name (None without a GPU). ``amp_used_by_this_code``
    and ``compile_used_by_this_code`` are not measurements of the process: they are properties of the code path (it has no autocast and no ``torch.compile`` call) and are named so.

    해석: 숫자 설정이 지금 어떤지를 있는 그대로 적는다. 이 값이 T009의 평가 때와 같아야 같은 가중치가 같은 비트를 낸다. 혼합 정밀도는 실제 켜져 있는지를 재고(autocast_enabled),
    컴파일은 이 코드 경로가 쓰지 않는다는 사실만 이름을 붙여 적는다(측정값이 아님).
    """
    import torch
    sdp = {name: bool(getattr(torch.backends.cuda, f'{name}_sdp_enabled')()) for name in ('flash', 'mem_efficient', 'math', 'cudnn') if hasattr(torch.backends.cuda, f'{name}_sdp_enabled')}
    return {'default_dtype': str(torch.get_default_dtype()), 'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32), 'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic), 'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()), 'sdp_backends': sdp,
            'autocast_enabled': bool(torch.is_autocast_enabled()) or bool(torch.is_autocast_enabled('cpu')), 'amp_used_by_this_code': False, 'compile_used_by_this_code': False,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'cudnn_version': _probe(torch.backends.cudnn.version), 'threads': int(torch.get_num_threads()), 'device_name': _probe(lambda: torch.cuda.get_device_name(0))}


REQUIRED_NUMERIC = {'default_dtype': 'torch.float32', 'float32_matmul_precision': 'highest', 'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False, 'cudnn_deterministic': True,
                    'cudnn_benchmark': False, 'deterministic_algorithms': True, 'autocast_enabled': False}


def numeric_failures(record: Mapping) -> list:
    """The settings of ``record`` that are not the T009 settings (``REQUIRED_NUMERIC``: float32, highest matmul precision, TF32 off for cuBLAS and cuDNN, cuDNN deterministic with no benchmark,
    deterministic algorithms, no autocast), as 'name=value' texts; empty when the process is set as T009 set it."""
    return [f'{name}={record.get(name)!r}' for name, wanted in REQUIRED_NUMERIC.items() if record.get(name) != wanted]


def apply_numeric_policy() -> dict:
    """Set the numeric policy of T009 (``comparison.models.configure_float32``: float32 required, highest matmul precision, TF32 off, cuDNN deterministic and not benchmarking, deterministic algorithms)
    and return ``numeric_record()``; ``Gate0Refused`` when a setting is not as required afterwards. It sets nothing else: the AMP, the compile and the attention backend are left as the library has them, and
    the record says which they are.

    해석: T009의 학습과 평가가 부른 같은 함수로 같은 설정을 켠다. 켠 뒤의 상태를 기록에 남기고, 하나라도 다르면 거부한다.
    """
    comparison_models.configure_float32()
    record = numeric_record()
    failures = numeric_failures(record)
    _require(not failures, 'the numeric policy of T009 could not be set: ' + '; '.join(failures))
    return record


# ---- the loader of the T009 run ------------------------------------------------------------------------------------------------------------------
RUN_FILE = 'RUN.json'
RUN_SCHEMA = 'ept-comparison-run-v1'
STAGE_SCHEMA = 'ept-comparison-stage-v1'
NEURAL_MODELS = ('ept', 'n2')


@contextlib.contextmanager
def _refusing():
    """Turn the refusals and the errors of a malformed or changed record (``ComparisonRefused``, ``ValueError``, ``KeyError``, ``TypeError``, a missing file) of the library calls into ``Gate0Refused``, so
    that the stage exits with code 2. A ``RuntimeError`` (a budget, a CUDA error) and an interrupt pass untouched."""
    try:
        yield
    except Gate0Refused:
        raise
    except (comparison_protocol.ComparisonRefused, ValueError, KeyError, TypeError, FileNotFoundError, NotADirectoryError) as error:
        raise Gate0Refused(f'{type(error).__name__}: {error}') from None


def _inside(root, relative) -> Path:
    with _refusing():
        return comparison_protocol.inside(root, relative)


def _hash_ok(value) -> bool:
    return comparison_protocol.hash_value(value)


def _read_json(path) -> dict:
    """The strict JSON object of ``path`` (``run.io.read_json``); a missing, linked, malformed or oversized file is a ``Gate0Refused`` that names the file, not its folder."""
    try:
        return run_io.read_json(path)
    except run_io.IORefused as error:
        raise Gate0Refused(f'{Path(path).name}: {error}') from None


@dataclasses.dataclass(frozen=True)
class T009Parts:
    """What ``load_t009`` returns, every part read only after its hash was checked: ``config`` (the T009 campaign configuration of ``RUN.json``) and its sha256, ``bundle`` (the verified read-only stores,
    the static channels, the prices and the tier model; its role is not STOP until ``stop_bundle``), ``membership`` (the matched 25% FIT and STOP selection), ``stats`` (the selected-FIT statistics),
    ``scales`` (the whole-FIT S3 scales), ``b1`` (the matched B1 heads), ``models`` (the requested neural models, best state only, strict load, in evaluation mode on ``device``), ``verified`` (every
    relative path whose hash was checked against a receipt and that hash) and ``summary`` (plain JSON about what was loaded, for the record)."""

    run_dir: Path
    config: dict
    config_sha256: str
    bundle: object
    membership: object
    stats: object
    scales: dict
    b1: object
    models: dict
    verified: dict
    summary: dict


def read_receipt(config: Mapping, stage: str, model=None, *, check=None) -> tuple:
    """``(record, verified)`` of the receipt ``<output>/<stage>[/<model>]/stage.json`` of the T009 run: it must be completed, bound to this configuration (``config_sha256`` = the digest of ``config``),
    name this stage and model, and every artifact that it records must have the recorded sha256 (``verified`` maps each relative path to it). ``Gate0Refused`` for a missing receipt, another schema, an
    unfinished stage, another configuration or a changed artifact.

    해석: T009가 단계마다 남긴 영수증을 읽는다. 완료 상태이고 같은 설정에 묶여 있으며 기록한 모든 파일의 해시가 지금도 같아야 한다.
    """
    with _refusing():
        folder = comparison_runtime.stage_path(config, stage, model)
    record = _read_json(folder / 'stage.json')
    _require(record.get('schema') == STAGE_SCHEMA and record.get('state') == 'completed' and record.get('config_sha256') == comparison_protocol.digest(config), 'stage binding incomplete')
    _require(record.get('stage') == stage and record.get('model') == model, 'stage identity mismatch')
    artifacts = record.get('artifacts')
    _require(isinstance(artifacts, dict) and artifacts and isinstance(record.get('details'), dict), 'the stage receipt records no artifacts')
    verified = {(f'{stage}/{model}' if model else stage) + '/stage.json': run_io.sha_file(folder / 'stage.json')}      # the receipt itself is recorded, not checked against anything
    for relative, expected in artifacts.items():
        path = _inside(config['output'], relative)
        _require(_hash_ok(expected) and path.is_file() and run_io.sha_file(path, check=check) == expected, f'stage artifact changed: {Path(relative).as_posix()}')
        verified[Path(relative).as_posix()] = expected
    return record, verified


def _load_neural(config, model: str, static, device, preparation, stats, check) -> tuple:
    """The best state of the neural model ``model`` of T009, built and loaded as ``comparison.runtime.load_neural_best`` does it, without its ``execution_authorized`` condition: the receipt of the
    fit stage, the architecture of the registry, the hash of the final head, the training identity (``runtime.validate_training_head``), the best pointer, the settings that the evaluation does not pass
    on (``evaluation.guard``), the static input, the hash and size of ``best.pt`` (read once: the bytes that were hashed are the bytes that are loaded) and a strict ``load_state_dict`` of the best
    weights only. Never the optimizer, the last state or an RNG state. Returns ``(model, verified, summary)``."""
    import torch
    from c_ept.ept import encoder
    record, verified = read_receipt(config, 'fit', model, check=check)
    info = record['details']
    with _refusing():
        spec = comparison_models.specification(model, hidden=config['settings']['n2_hidden'], prefetch=config['settings']['prefetch'])
        _require(info['architecture'] == spec, 'model registry identity mismatch')
        folder = comparison_runtime.stage_path(config, 'fit', model)
        head_path = _inside(folder, info['head'])
        _require(run_io.sha_file(head_path, check=check) == info['head_sha256'], 'final neural head changed')
        head = _read_json(head_path)
        comparison_runtime.validate_training_head(config, head, preparation, model)
        identity = head['identity']
        _require(head['done'] is True and head['best_epoch'] == info['best_epoch'] and head['best'] == info['best'], 'best pointer differs from final head')
        _require(identity['model_spec'] == spec, 'head architecture differs')
        unpassed = evaluation_guard.unpassed_evaluation_settings(identity)
        _require(not unpassed, f'the checkpoint was trained with settings that this evaluation does not pass on: {unpassed}')
        _require(identity['fit'].get('stats_sha256') == stats.sha256 == identity['stop'].get('stats_sha256'), 'the statistics differ from the ones the model was trained with')
        _require(identity['static_sha256'] == hashlib.sha256(static.tobytes()).hexdigest(), 'static input differs')
        best = _inside(folder / 'model', head['best']['folder']) / 'best.pt'
        _require(best.is_file() and not best.is_symlink() and best.stat().st_size == head['best']['bytes'], 'invalid best-only file')
        check()
        raw = best.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == head['best']['sha256'], 'best-only payload changed')
        built = comparison_models.build_model(encoder.EncoderConfig(**identity['encoder']), static, spec)
        try:
            built.load_state_dict(torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu'), strict=True)
        except RuntimeError as error:                                                       # missing or unexpected keys, a shape that differs: the best weights are not this architecture
            raise Gate0Refused('the best weights do not load strictly into the architecture: ' + str(error)[:300]) from None
        parameters = int(sum(p.numel() for p in built.parameters()))
        _require(info.get('model_parameters') == parameters, 'the parameter count differs from the fit receipt')
    verified[f'fit/{model}/model/{head["best"]["folder"]}/best.pt'] = head['best']['sha256']
    summary = {'architecture': spec['architecture'], 'best_epoch': int(info['best_epoch']), 'best_folder': head['best']['folder'], 'best_sha256': head['best']['sha256'], 'best_bytes': int(head['best']['bytes']),
               'head_sha256': info['head_sha256'], 'parameters': parameters, 'training_seed': identity['seed'], 'epochs_run': info.get('epochs_run'), 'stopped_early': info.get('stopped_early'),
               'identity_torch': identity.get('torch'), 'identity_dtype': identity['dtype'], 'identity_threads': identity['threads'], 'identity_device': identity['device'],
               'identity_deterministic_algorithms': identity['deterministic_algorithms']}
    return built.to(device).eval(), verified, summary


T009_CONFIG_SCHEMA = 'ept-comparison-operator-v2'                              # the schema of the T009 configuration (commit 681a3cd); the protocol module of this branch is still the earlier v1
T009_CONFIG_FIELDS = frozenset({'schema', 'profile', 'source', 'source_commit', 'source_manifest', 'source_manifest_sha256', 'python', 'output', 'inputs', 'settings', 'limits',
                                'expected_scale_provenance', 'execution_authorized'})
T009_INPUTS = frozenset({'s1', 's2', 'labels', 'terrain', 's3', 'prices', 'scales', 'rules', 'legacy'})
T009_INPUT_FILES = {'s1': {'STORE.json'}, 's2': {'STORE.json'}, 'labels': {'STORE.json'}, 'scales': {'B0_SCALES.json'}, 'rules': {'RULE_PAYOUTS.json'},
                    'terrain': {'TERRAIN.json', 'terrain.npz', 'geodesic.npy'}, 'prices': {'PRICES.json'}}


def validate_t009_config(config) -> dict:
    """The checks of ``protocol.validate_config`` of the commit that made the T009 run (681a3cd), done here because the protocol module of this branch is the earlier version: the schema
    ``ept-comparison-operator-v2``, exactly the configuration fields, the matched profile, the fixed settings and limits of ``protocol`` with their types (``FIXED``, ``LIMITS``), a boolean
    ``execution_authorized`` (its value is not a condition of this probe), the fixed Python, absolute paths, a source commit and manifest hash that are null or well formed, and the nine inputs
    ``T009_INPUTS`` (each a root and a non-empty map of files to null or sha256, with the file names the T009 profile fixes: the store metadata, the B0 scales, the rule payouts, the terrain and prices files,
    the RUN.json and STAGES.json of the S3 run). Returns ``config``; ``Gate0Refused`` for anything else.

    해석: T009 폴더의 설정은 이 가지의 protocol(v1)보다 나중 형식(v2: scales, rules, legacy 입력이 더 있음)이다. 그 형식의 검사를 그대로 옮겼고, 실행 허가 칸은 값이 무엇이든 받는다."""
    protocol = comparison_protocol
    ok = isinstance(config, dict) and set(config) == T009_CONFIG_FIELDS
    _require(ok, 'comparison config fields mismatch')
    _require(config['schema'] == T009_CONFIG_SCHEMA and config['profile'] == protocol.PROFILE, 'comparison profile mismatch (the T009 configuration is schema v2)')
    _require(config['settings'] == protocol.FIXED and all(type(config['settings'][k]) is type(v) for k, v in protocol.FIXED.items()), 'fixed matched25 settings changed')
    _require(config['limits'] == protocol.LIMITS and all(type(config['limits'][k]) is int for k in protocol.LIMITS), 'campaign limits changed')
    _require(type(config['execution_authorized']) is bool, 'execution_authorized must be boolean')
    _require(Path(config['python']) == protocol.PYTHON, 'fixed Python required')
    for key in ('source', 'source_manifest', 'output'):
        _require(isinstance(config[key], str) and Path(config[key]).is_absolute(), f'{key} must be an explicit absolute path')
    _require(config['source_commit'] is None or (isinstance(config['source_commit'], str) and re.fullmatch(r'[0-9a-f]{40}', config['source_commit']) is not None), 'invalid source commit')
    _require(config['source_manifest_sha256'] is None or _hash_ok(config['source_manifest_sha256']), 'invalid source manifest hash')
    inputs = config['inputs']
    _require(isinstance(inputs, dict) and set(inputs) == T009_INPUTS, 'input identities required')
    for name, entry in inputs.items():
        _require(isinstance(entry, dict) and set(entry) == {'root', 'files'}, f'{name} input fields mismatch')
        _require(isinstance(entry['root'], str) and Path(entry['root']).is_absolute(), 'explicit input root required')
        _require(isinstance(entry['files'], dict) and entry['files'], 'input file identity map required')
        for relative, sha in entry['files'].items():
            _inside(entry['root'], relative)
            _require(sha is None or _hash_ok(sha), 'invalid input hash')
    for name, names in T009_INPUT_FILES.items():
        _require(set(inputs[name]['files']) == names, f'the files of the input {name} are not the fixed ones')
    _require({'RUN.json', 'STAGES.json'} <= set(inputs['s3']['files']), 'S3 lineage metadata identities required')
    _require(isinstance(config['expected_scale_provenance'], dict), 'independent scale provenance required')
    return config


def check_t009_record(run_dir, run) -> dict:
    """The checks of the RUN.json record of a T009 run that need no data and no hash of a payload: it is a comparison run record, its configuration is valid under the T009 schema v2
    (``validate_t009_config``), the legacy pins of this code are those of that configuration, its digest is the recorded one, and the run folder is the output of the configuration. Returns the
    configuration; ``Gate0Refused`` otherwise. ``supervise`` makes these checks before it creates the output folder, ``load_t009`` makes them again in the stage.

    해석: RUN.json만 읽고 아는 것(스키마, 설정 형식, 규칙 핀, 해시, 폴더 위치)은 출력 폴더를 만들기 전에 부모가 먼저 확인한다. 자식 단계도 같은 확인을 다시 한다."""
    _require(isinstance(run, Mapping) and run.get('schema') == RUN_SCHEMA and isinstance(run.get('config'), dict), 'RUN.json is not a comparison run record')
    config = run['config']
    validate_t009_config(config)
    from c_ept import legacy
    _require(legacy.pins()['files'] == config['inputs']['legacy']['files'], 'the legacy pins of this code are not the ones of the T009 configuration')
    _require(run.get('config_sha256') == comparison_protocol.digest(config), 'the configuration of RUN.json differs from its recorded digest')
    _require(Path(config['output']).resolve() == Path(run_dir).resolve(), 'the run folder is not the output of its configuration')
    return config


def load_t009(run_dir, *, device, check=None, models=NEURAL_MODELS, bundle_loader=None) -> T009Parts:
    """Read the finished T009 matched-25% run in ``run_dir`` for the probe, and refuse on any mismatch.

    ``RUN.json`` holds the campaign configuration (schema v2, ``validate_t009_config``); its digest must be the recorded one, its fixed settings those of the profile and its output the folder given here. The
    ``execution_authorized`` flag of the configuration is NOT required (the probe's own start approval is recorded instead). Then, each read only after the sha256 of its receipt matched:
    ``prepare`` (the shared statistics, the membership, the inputs record), the fit stage of the matched B1 (loaded with ``B1Heads.load``, which checks its own file hashes), the whole-FIT scales (the file
    named by the configuration, with the provenance the configuration expects), the verified stores through ``bundle_loader(meta, check)`` (default ``comparison.runtime.load_bundle``; the membership is
    proved against the S1 role table), and the best state of each model of ``models`` (``_load_neural``). The numeric policy is not set here: call ``apply_numeric_policy`` first.

    해석: 끝난 T009 폴더에서 이 탐침에 필요한 것만, 영수증의 해시를 확인한 뒤에 읽는다. 실행 허가 칸은 요구하지 않고(이 탐침의 시작 승인이 따로 기록된다), 해시나 설정이 하나라도 다르면 거부한다.
    """
    from c_ept.comparison.data import Membership
    from c_ept.ept.data import Stats
    from c_ept.ept.scales import load_scales
    from c_ept.models.baselines_b1 import B1Heads
    check = check if check is not None else (lambda: None)
    loader = bundle_loader if bundle_loader is not None else comparison_runtime.load_bundle
    _require(isinstance(models, (tuple, list)) and models and set(models) <= set(NEURAL_MODELS) and len(set(models)) == len(models), f'models must be a non-empty selection of {NEURAL_MODELS}')
    from c_ept import legacy
    run_dir = Path(run_dir)
    config = check_t009_record(run_dir, _read_json(run_dir / RUN_FILE))
    digest = comparison_protocol.digest(config)
    prepare, verified = read_receipt(config, 'prepare', check=check)
    folder = comparison_runtime.stage_path(config, 'prepare')
    with _refusing():
        meta = _read_json(folder / 'inputs.json')
        membership = Membership.from_record(_read_json(folder / 'membership.json'))
        stats = Stats.load(folder / 'STATS.json')
        _require(stats.sha256 == prepare['details']['stats_sha256'] and membership.sha256 == prepare['details']['membership_sha256'], 'the shared statistics or the membership differ from the receipt')
        scales_path = _inside(config['inputs']['scales']['root'], 'B0_SCALES.json')
        scales_sha = run_io.sha_file(scales_path, check=check)
        _require(scales_sha == config['inputs']['scales']['files']['B0_SCALES.json'], 'the scales file is not the one the configuration binds')
        scales = load_scales(scales_path, expected_provenance=config['expected_scale_provenance'])
    bundle = loader(meta, check)
    with _refusing():
        membership.verify_against(bundle.s1)
    # B1: the receipt of its fit, its result file, the identities, then the heads (B1Heads.load checks the hash of every head file and every model file)
    b1_record, found = read_receipt(config, 'fit', 'b1', check=check)
    verified.update(found)
    b1_details = b1_record['details']
    _require(b1_details.get('membership_sha256') == membership.sha256 and b1_details.get('stats_sha256') == stats.sha256 and b1_details.get('scale_sha256') == scales_sha,
             'the B1 fit receipt names another membership, statistics or scales')
    with _refusing():
        b1 = B1Heads.load(comparison_runtime.stage_path(config, 'fit', 'b1') / 'model')
    _require(tuple(b1.fit_roles) == ('FIT',) and tuple(b1.stop_roles) == ('STOP',) and getattr(b1, 'kill_law', 'poisson') == 'poisson' and getattr(b1, 'k_condition', 'raw') == 'raw',
             'the B1 heads are not the plain-law, raw-K heads fitted on FIT with selection on STOP')
    loaded, summaries = {}, {}
    static = bundle.static
    for name in models:
        loaded[name], found, summaries[name] = _load_neural(config, name, static, device, prepare, stats, check)
        verified.update(found)
    summary = {'config_sha256': digest, 'execution_authorized_in_config': config['execution_authorized'], 'source_commit': config['source_commit'], 'profile': config['profile'],
               'config_settings': {key: config['settings'][key] for key in ('dtype', 'tf32', 'amp', 'native_threads', 'batch_size', 'prefetch', 'n2_hidden', 'neural_seed', 'max_epochs')},
               'receipts': {path[:-len('/stage.json')]: sha for path, sha in sorted(verified.items()) if path.endswith('/stage.json')},
               'membership': {'sha256': membership.sha256, 'fit_matches': len(membership.fit), 'stop_matches': len(membership.stop), 'selection_seed': membership.selection_seed, 'share': membership.share},
               'stats_sha256': stats.sha256, 'scales_sha256': scales_sha, 'legacy': {'root': config['inputs']['legacy']['root'], 'pins_sha256': legacy.pins_sha256()},
               'b1': {'dropped_columns': len(b1.dropped), 'dropped_sha256': run_io.canonical_sha256(list(b1.dropped)), 'fit_roles': list(b1.fit_roles), 'stop_roles': list(b1.stop_roles)},
               'stats_dropped': len(getattr(stats, 'dropped', ())), 'models': summaries}
    return T009Parts(run_dir=run_dir, config=config, config_sha256=digest, bundle=bundle, membership=membership, stats=stats, scales=scales, b1=b1, models=loaded, verified=verified, summary=_plain(summary))


def stop_bundle(bundle):
    """The bundle of ``load_t009`` with the role 'STOP': ``run_inference`` refuses a bundle that does not declare the role of its call, and the T009 bundle declares DEV (its one production builder). Every
    store, the static channels, the prices and the tier model are the same objects: only the declared role changes. DEV and TEST are not opened by this."""
    return SimpleNamespace(**{**vars(bundle), 'role': 'STOP'})


def _sha_of_array(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def common_matrix(n_matches: int, b: int = BOOTSTRAP_B) -> tuple:
    """``(matrix, record)``: the first ``b`` rows of the common 10,000-row match-resample matrix of ``n_matches`` matches (``evaluation.inference.resample_matrix``, seed 20261022; D13 takes B = 2,000 of
    it), and the record of it: the numpy version (the stream of ``Generator.integers`` is not promised across versions), the seed, the rows of the whole matrix and its sha256, B, the sha256 of the rows used
    and ``n_matches``. ValueError for a ``b`` outside 1..10,000.

    해석: 모든 비교가 같은 재표본 행렬을 쓴다. 이 탐침은 그 앞 2,000행만 쓰고, 행렬 전체와 쓴 부분의 해시를 남겨 다른 곳에서 만든 행렬과 비교할 수 있게 한다."""
    if isinstance(b, bool) or not isinstance(b, int) or not 1 <= b <= inference.B_DEFAULT:
        raise ValueError(f'b must be a whole number from 1 to {inference.B_DEFAULT}')
    full = inference.resample_matrix(n_matches, B=inference.B_DEFAULT, seed=inference.SEED)
    used = full[:b]
    return used, {'numpy': np.__version__, 'seed': inference.SEED, 'matrix_rows': int(full.shape[0]), 'matrix_sha256': _sha_of_array(full), 'B': int(b), 'used_sha256': _sha_of_array(used),
                  'n_matches': int(n_matches)}


def stop_match_ids(s1) -> np.ndarray:
    """The one sorted, unique list of the STOP matches (their global match numbers, from the role table ``matches/role``) on which every statistic is aggregated and every resample is drawn (D15).
    A STOP match with no primary row is on the list: it has weight 0 in every ratio and keeps the resample matrix the size of the split. ValueError for a split with no STOP match.

    해석: 통계는 모두 같은 STOP 경기 목록(정렬, 중복 없음) 위에서 모은다. 주 모집단 행이 하나도 없는 경기도 목록에는 있고, 비의 분자·분모에서는 0으로 빠진다."""
    from c_ept import config
    roles = np.asarray(s1.array('matches/role'))
    ids = np.flatnonzero(roles == config.ROLES.index('STOP')).astype(np.int64)
    if not len(ids):
        raise ValueError('the role table has no STOP match')
    return ids


# ---- the evaluation of one arm ---------------------------------------------------------------------------------------------------------------------
BATCH_SIZE = 32                                                               # the T009 batch (fixed by the matched profile)
INFERENCE_SETTINGS = {'role': 'STOP', 'exposure': 'nominal', 'population': 'primary', 'monte_carlo': False}      # what the probe asks run_inference for, nothing else


def evaluate_gate0(stop_bundle, models: Mapping, stats, b1, *, device, check, scales=None, log=None, batch_size: int = BATCH_SIZE) -> dict:
    """``run_inference(..., role='STOP', exposure='nominal', population='primary', monte_carlo=False)`` for each model of ``models`` (``{'ept': model, 'n2': model}``) with the one matched B1, and
    nothing else: the results by name, each as ``run_inference`` returns it. The closed-form quantities only (no Monte Carlo draw, no f0 truth read), the primary rows only (``population``), the nominal
    exposure (T enters no forecast), the whole STOP split, batch 32. ``Gate0Refused`` unless the bundle declares the role STOP (DEV and TEST are never read), ``ValueError`` for a model that is none of
    EPT and N2.

    ``log``: a ``run.evaluate.StageLog`` or None. With a log each model gets an ``inference`` stage (detail: the role, the pass, the exposure and the population) and, inside it, one ``block`` stage for
    each of the thirteen head blocks (``run.evaluate.BlockStages``), so that a run that hits its wall cap shows which block used the time. ``check``: the guard that ``run_inference`` calls between
    batches. ``scales``: the whole-FIT S3 scales (a fixed mapping, never fitted).

    해석: 추론 함수에 STOP 역할, 공칭 노출, 주 모집단, 몬테카를로 없음만 요구한다. EPT와 N2를 같은 B1과 따로 돌리고, 기록이 있으면 머리 블록 13개마다 시작과 끝 줄이 남는다.
    """
    from c_ept.evaluation import adapters
    from c_ept.run import evaluate as run_evaluate
    _require(getattr(stop_bundle, 'role', None) == 'STOP', 'the probe evaluates the role STOP only: the bundle does not declare it')
    _require(set(models) <= set(NEURAL_MODELS), f'models must be among {NEURAL_MODELS}')
    results = {}
    for name in NEURAL_MODELS:
        if name not in models:
            continue
        text = comparison_protocol.matched_text({'ept': 'EPT', 'n2': 'N2'}[name], 'B1')['comparison']
        detail = {'role': 'STOP', 'pass': 'primary_nominal', 'exposure': 'nominal', 'population': 'primary'}
        stage = log.stage('inference', name, detail=detail) if log is not None else contextlib.nullcontext({})
        blocks = run_evaluate.BlockStages(log, name, {'role': 'STOP', 'pass': 'primary_nominal'}) if log is not None else contextlib.nullcontext(None)
        with stage as counts, blocks as progress:
            result = adapters.run_inference(stop_bundle, models[name], stats, b1, check=check, device=device, batch_size=batch_size, scales=scales, comparison=text,
                                            progress=progress, **INFERENCE_SETTINGS)
            counts.update(endpoint_rows=sum(len(endpoint['row']) for endpoint in result['endpoints']), block_records=len(result['nll']))
        results[name] = result
    return results


# ---- the record of the old-versus-new equivalence proof -----------------------------------------------------------------------------------------------
EQUIVALENCE_SCHEMA = 'ept-gate0-equivalence-v1'
_COMMIT = re.compile(r'[0-9a-f]{40}')


EQUIVALENCE_LIMITS = (
    '증명은 CPU에서만 돌렸다. GPU 커널(cuBLAS, cuDNN, 어텐션 백엔드)은 비교하지 않았고, GPU에서만 나타나는 차이는 배제되지 않는다.',
    '증명에 쓴 EPT는 M2 확률이 거의 상수가 되게 맞춘 것이다(H60은 주 모집단 행에서 약 0.3319~0.3330). 학습된 모형보다 M2 끝점 비교의 변별력이 약하다.',
    'M4의 p_ept는 모두 1.0으로 포화되어 M4의 EPT 끝점 비교는 실질이 없다(M4 NLL은 비교했다).',
    '끝점의 M2 counts(후보·적격 행 수)는 비교하지 않았다.',
    '실제 load_bundle 저장소, 실제 척도, 실제 S3 보조 자료는 쓰지 않았다. 장난감 세계에 실제 S 크기 EPT, 은닉 512의 N2, 실제 작은 B1을 두었고, 엄격 적재와 Stats·Membership·B1의 저장 뒤 적재만 확인했다.')


def read_equivalence_bytes(path) -> tuple:
    """``(record, raw)``: the equivalence record of ``read_equivalence`` and the bytes it was read from (one read), so that a copy of the record can be made from exactly what was parsed and hashed
    (``record['file_sha256']`` is the SHA-256 of ``raw``). ``Gate0Refused`` for a link, a missing, malformed or oversized file and for any record that ``read_equivalence`` refuses.

    해석: 기록을 읽을 때 쓴 바이트를 함께 돌려준다. 복사본을 그 바이트로 만들면, 읽은 뒤에 원본이 바뀌어도 START.json의 해시와 EQUIVALENCE.json이 어긋나지 않는다."""
    captured = []
    try:
        record, digest = run_io.read_json_with_sha(path, check=captured.append)
    except run_io.IORefused as error:
        raise Gate0Refused(f'{Path(path).name}: {error}') from None
    return _checked_equivalence(record, digest), bytes(captured[0])


def read_equivalence(path) -> dict:
    """The record of the equivalence proof of the task record (the same weights must give the same closed-form endpoints and per-head NLL under the T009 commit and under this one, for EPT-S, N2 and B1; a
    script outside the repository makes it and writes this record), read strictly and checked: ``schema``, ``status`` (one of ``EQUIVALENCE_STATUSES``), ``old_commit`` and ``new_commit`` (40 hex digits),
    ``keys_compared`` (a whole number of at least 1), ``differences`` (a list of ``{'key', 'cause'}``: empty exactly for the status 'passed', and each with a cause for 'differences_explained') and an
    ``explanation`` for any other status than 'passed'. Returns the record with ``file_sha256``. ``Gate0Refused`` for anything else.

    해석: 옛 코드(T009를 학습한 커밋)와 새 코드가 같은 가중치에서 같은 값을 내는지 별도 스크립트가 증명하고, 그 결과를 이 기록으로 넘긴다. 차이가 있으면 원인이 적혀 있어야 한다."""
    return read_equivalence_bytes(path)[0]


def _checked_equivalence(record: dict, digest: str) -> dict:
    """The strict checks of ``read_equivalence`` on the parsed ``record`` whose file has the SHA-256 ``digest``."""
    _require(record.get('schema') == EQUIVALENCE_SCHEMA, 'the equivalence record has another schema')
    _require(record.get('status') in EQUIVALENCE_STATUSES, f'the equivalence status must be one of {EQUIVALENCE_STATUSES}')
    for key in ('old_commit', 'new_commit'):
        _require(isinstance(record.get(key), str) and _COMMIT.fullmatch(record[key]) is not None, f'the equivalence record needs {key} as 40 hex digits')
    count = record.get('keys_compared')
    _require(isinstance(count, int) and not isinstance(count, bool) and count >= 1, 'the equivalence record needs keys_compared of at least 1')
    differences = record.get('differences')
    _require(isinstance(differences, list) and all(isinstance(d, dict) and isinstance(d.get('key'), str) for d in differences), 'the equivalence record needs a list of differences')
    if record['status'] == 'passed':
        _require(not differences, 'a passed equivalence record lists no difference')
    else:
        _require(isinstance(record.get('explanation'), str) and record['explanation'].strip(), 'an equivalence record that is not passed needs an explanation')
    if record['status'] == 'differences_explained':
        _require(differences and all(isinstance(d.get('cause'), str) and d['cause'].strip() for d in differences), 'every difference of an explained record needs a cause')
    return {**record, 'file_sha256': digest}


# ---- the timings, from the progress record ---------------------------------------------------------------------------------------------------------
PROGRESS_FILE = 'PROGRESS.jsonl'


def timings_from_progress(path) -> dict:
    """The seconds each stage and each head block used, from the end lines of ``PROGRESS.jsonl`` (``run.evaluate.StageLog``): per model ``arm_s``, ``load_s``, ``inference_s`` and ``blocks`` (``{block:
    {'elapsed_s', 'rows', 'views', 'status'}}``), ``child_s`` (as the supervisor saw the stage child) and the whole run ``total_s``. What the file does not hold is None: missing telemetry is unavailable,
    never filled in. A file that does not exist gives ``{'status': 'unavailable'}``. The last attempt of the file is read.

    해석: 단계마다 걸린 초를 진행 기록의 끝 줄에서 읽는다. 기록에 없는 값은 None(측정되지 않음)이고 추정으로 채우지 않는다. 블록별 행·보기 수가 함께 있어 속도 개선이 필요할 때 근거가 된다."""
    path = Path(path)
    if not path.is_file():
        return {'status': 'unavailable'}
    lines = []
    for text in path.read_text(encoding='utf-8').splitlines():
        try:
            item = json.loads(text)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get('event') in ('begin', 'end'):
            lines.append(item)
    attempt = max((line.get('attempt', 1) for line in lines), default=1)
    ends = [line for line in lines if line['event'] == 'end' and line.get('attempt', 1) == attempt]
    out = {'status': 'available', 'attempt': attempt, 'total_s': next((line['elapsed_s'] for line in ends if line['stage'] == 'gate0'), None)}
    for model in NEURAL_MODELS:
        mine = [line for line in ends if line.get('model') == model]
        pick = lambda stage: next((line for line in mine if line['stage'] == stage), None)
        blocks = {line['detail']['block']: {'elapsed_s': line['elapsed_s'], 'rows': line['counts'].get('rows'), 'views': line['counts'].get('views'), 'status': line['status']}
                  for line in mine if line['stage'] == 'block'}
        out[model] = {'arm_s': (pick('arm') or {}).get('elapsed_s'), 'load_s': (pick('load') or {}).get('elapsed_s'), 'inference_s': (pick('inference') or {}).get('elapsed_s'),
                      'child_s': (pick('child') or {}).get('elapsed_s'), 'blocks': blocks, 'status': (pick('child') or pick('arm') or {}).get('status')}
    return out


# ---- the result: the arms, the integrity checks, the statistics and the verdict ----------------------------------------------------------------------------
RESULT_SCHEMA = 'ept-gate0-v1'
RESULT_FILE, REPORT_FILE, ARM_FILE, STAGE_FILE, START_FILE = 'GATE0.json', 'GATE0_KO.md', 'ARM.json', 'STAGE.json', 'START.json'
DEFAULT_BUNDLE_LOADER = 'c_ept.comparison.runtime:load_bundle'                  # the one production builder of the verified stores
CLASSIFICATION = 'exploratory go/no-go (EPT-T013, author decision D31); old stack, one training seed, the 15.20 patch only: it supports no paper claim'
NOT_RECORDED_BY_T009 = ('the float32 matmul precision as a measured state (configure_float32() sets it in the T009 code; the receipts do not say it)',
                        'the cuDNN deterministic, benchmark and TF32 flags and the cuBLAS TF32 flag as measured states (the configuration says tf32 False)',
                        'the scaled-dot-product-attention backend flags', 'the CUDA and cuDNN versions and the device name')
CAVEATS = ('Old stack: Poisson kill law, raw K condition, two-frame training, whole-FIT S3 auxiliary assets.',
           'On the primary rows the M2 onset endpoint does not use T. The M3/M4 per-head NLL values carry the old kill law for both arms alike.',
           'One seed: a seed difference cannot be separated from a model difference. The interval covers match sampling only, not training-seed variation.',
           'Within-patch (15.20) prediction only; patch invariance is not tested.',
           'The Brier skill uses the base rate of the same STOP rows: it is not a held-out value.')


CHECK_SCOPES = ('primary', 'secondary')
CROSS_ARM_SCOPE = {'common_id_list_equal': 'secondary', 'arms_on_the_same_rows': 'secondary', 'b1_probabilities_equal': 'primary'}      # a row mismatch drops the N2 rows; a B1 that differs doubts the EPT's B1 too


def _check(name: str, passed: bool, detail: str, scope: str = 'primary') -> dict:
    """One integrity check as ``{'name', 'passed', 'detail', 'scope'}``. ``scope`` 'primary' is a check that the primary statistic depends on (its failure is a VOID); 'secondary' is one that only the N2
    arm depends on (its failure is a warning and the N2 rows are unavailable).

    해석: 점검마다 범위가 붙는다. 주 통계(EPT − B1)가 기대는 점검이 실패하면 판정이 VOID이고, 보조 팔(N2)만 기대는 점검이 실패하면 경고로 남기고 N2 칸만 없앤다(검토 I1)."""
    return {'name': name, 'passed': bool(passed), 'detail': detail, 'scope': scope}


def _scoped(checks, scope) -> list:
    """``checks`` (``check_arm`` or ``check_arms`` output) with the scope ``scope`` (a mapping: by check name) added to each."""
    return [{**check, 'scope': scope[check['name']] if isinstance(scope, Mapping) else scope} for check in checks]


def _shared(verified: Mapping) -> dict:
    """The part of ``verified`` that the two arms share: the preparation stage and the matched B1."""
    return {path: sha for path, sha in verified.items() if path.startswith(('prepare/', 'fit/b1/'))}


def assemble(out_dir, *, stages: Mapping, equivalence: Mapping, start: Mapping) -> dict:
    """The result of the probe from what the stages left in ``out_dir``: read each finished arm (its ``ARM.json`` must be the one the supervisor recorded, and its arrays the one ``ARM.json`` names), make the
    integrity checks, the common resample matrix, the statistics (``analyze``) and the verdict (``decide``), and gather the provenance, the timings and the hashes. Nothing is written.

    ``stages``: ``{'ept': record, 'n2': record}``, the supervisor's receipt of each stage (``state`` 'completed' or not, ``exit_code``, ``elapsed_s``, ``cap_s``, ``arm_sha256`` for a finished stage).
    ``equivalence``: ``read_equivalence``. ``start``: the START record (the T009 run and its configuration digest, the approval text, E6, the code identity of the supervisor, the wall budget, the device,
    the bundle loader and whether the identity gate was passed).

    Which check can void: every check has a ``scope``. The 'primary' ones (the EPT stage, its arm, call settings, numeric policy, production inputs, code identity and T009 configuration, the B1 that both
    stages must give alike, the shared preparation and B1 files, the binding of the equivalence record to the commits) make ``integrity['failures']``; any of them failing is a VOID, and the primary statistic of an
    arm that did finish is still reported. The 'secondary' ones (the same checks of the N2 arm, and the id and row agreement of the two arms) make ``integrity['warnings']`` and decide nothing: the N2 arm is then dropped
    (``n2_unavailable``, a secondary check that names the reasons), so no N2 row is made from it and every cell that needs N2 is 'unavailable'. The cross-arm checks are made when both arms were read.

    해석: 단계가 남긴 것을 읽어 무결성 점검, 통계, 판정을 한 기록으로 모은다. 점검마다 범위가 있다. 주 통계가 기대는 점검(EPT 팔 전부, 두 단계가 같아야 하는 B1, 동등성 증명과 커밋의 묶음)이 실패하면 판정은
    VOID이고, 그래도 끝난 팔의 통계는 숨기지 않고 함께 적는다. 보조 팔 N2의 단계 실패나 점검 실패, 두 팔의 행 불일치는 경고로만 남고 N2 칸이 '없음'과 이유로 남을 뿐 판정은 그대로이다(검토 I1).
    """
    out_dir = Path(out_dir)
    checks, arms, docs = [], {}, {}
    scope_of = {'ept': 'primary', 'n2': 'secondary'}
    for model in NEURAL_MODELS:
        scope = scope_of[model]
        stage = stages.get(model) or {'state': 'not_run'}
        finished = stage.get('state') == 'completed'
        checks.append(_check(f'{model}_stage_completed', finished, f'the {model} stage finished' if finished else f'the {model} stage did not finish ({stage.get("state")}: {stage.get("reason")})', scope))
        if not finished:
            continue
        try:
            doc = _read_json(out_dir / model / ARM_FILE)
            _require(doc.get('schema') == ARM_SCHEMA and doc.get('model') == model, f'{ARM_FILE} of the {model} arm is not an arm record of that model')
            _require(run_io.sha_file(out_dir / model / ARM_FILE) == stage.get('arm_sha256'), f'{ARM_FILE} of the {model} arm differs from the hash that the supervisor recorded')
            arms[model] = load_arm(out_dir / model, model, doc['manifest'], doc['provenance'])
            docs[model] = doc
            checks.append(_check(f'{model}_arm_readable', True, f'the {model} arm was read after its hashes matched', scope))
        except (Gate0Refused, KeyError, TypeError, ValueError) as error:
            checks.append(_check(f'{model}_arm_readable', False, f'{model}: {type(error).__name__}: {error}', scope))
    for model, arm in arms.items():
        doc, scope = docs[model], scope_of[model]
        checks.extend(_scoped(check_arm(arm), scope))
        numeric = doc.get('numeric') or {}
        wrong = numeric_failures(numeric)
        checks.append(_check(f'{model}_numeric_policy', not wrong, 'the numeric policy was the T009 policy' if not wrong else 'the numeric policy differs: ' + '; '.join(wrong), scope))
        settings = doc.get('settings') or {}
        checks.append(_check(f'{model}_production_inputs', settings.get('bundle_loader') == DEFAULT_BUNDLE_LOADER and settings.get('identity_gate') == 'start_gate',
                             'the stores came from the production loader and the code identity passed the start gate' if settings.get('bundle_loader') == DEFAULT_BUNDLE_LOADER and settings.get('identity_gate') == 'start_gate'
                             else f'the stage did not use the production inputs (loader {settings.get("bundle_loader")!r}, identity gate {settings.get("identity_gate")!r})', scope))
        commit = (doc.get('code_identity') or {}).get('commit')
        checks.append(_check(f'{model}_code_identity', commit is not None and commit == (start.get('code_identity') or {}).get('commit'),
                             f'the stage and the supervisor ran the commit {str(commit)[:12]}' if commit == (start.get('code_identity') or {}).get('commit') else 'the stage ran another commit than the supervisor', scope))
        same = start.get('t009_config_sha256') is not None and (doc.get('t009') or {}).get('config_sha256') == start.get('t009_config_sha256')
        checks.append(_check(f'{model}_t009_configuration', same, 'the T009 configuration of the stage is the one the supervisor read' if same else 'the stage read another T009 configuration than the supervisor', scope))
    checks.extend(_scoped(check_arms(arms), CROSS_ARM_SCOPE))
    if len(docs) == 2:
        one, two = (_shared(docs[m].get('verified') or {}) for m in NEURAL_MODELS)
        checks.append(_check('shared_inputs_equal', bool(one) and one == two, 'both stages verified the same preparation and B1 files' if bool(one) and one == two else 'the two stages verified different preparation or B1 files'))
    read_docs = dict(docs)                                                           # every arm that was read, also an N2 arm that is dropped below: its file hashes and commit stay on record
    unavailable = [c['detail'] for c in checks if c['scope'] == 'secondary' and not c['passed']]
    if unavailable or 'n2' not in arms:
        arms.pop('n2', None)
        docs.pop('n2', None)
        checks.append(_check('n2_unavailable', False, 'the N2 arm is unavailable, every N2 row is left out: ' + '; '.join(unavailable or ['the N2 stage left no arm']), 'secondary'))
    summary = next((doc['t009'] for doc in docs.values()), None)
    commit = (start.get('code_identity') or {}).get('commit')
    bound = summary is not None and equivalence.get('old_commit') == summary.get('source_commit') and equivalence.get('new_commit') == commit
    checks.append(_check('equivalence_binds_these_commits', bound, 'the equivalence proof compares the T009 source commit with the commit that ran' if bound else
                         'the equivalence proof does not name the T009 source commit and the commit that ran'))
    ids = next((arm['ids'] for arm in arms.values()), None)
    analysis, matrix_record = {'primary': None, 'secondary': None}, None
    if ids is not None:
        matrix, matrix_record = common_matrix(len(ids))
        analysis = analyze(arms, ids, matrix)
    failures = [c['detail'] for c in checks if not c['passed'] and c['scope'] == 'primary']
    warnings = [c['detail'] for c in checks if not c['passed'] and c['scope'] == 'secondary']
    integrity = {'passed': not failures, 'failures': failures, 'warnings': warnings, 'checks': checks}
    files = {PROGRESS_FILE: run_io.sha_file(out_dir / PROGRESS_FILE) if (out_dir / PROGRESS_FILE).is_file() else None}
    for model in read_docs:
        files[f'{model}/{ARM_FILE}'] = run_io.sha_file(out_dir / model / ARM_FILE)
        files[f'{model}/{ARRAYS_FILE}'] = read_docs[model]['manifest']['arrays_sha256']
    result = {'schema': RESULT_SCHEMA, 'task': 'EPT-T013', 'decision': 'D31', 'classification': CLASSIFICATION, 'integrity': integrity,
              'equivalence': {**{key: equivalence.get(key) for key in ('status', 'old_commit', 'new_commit', 'keys_compared', 'differences', 'explanation', 'file_sha256')}, 'limits': list(EQUIVALENCE_LIMITS)},
              'primary': analysis['primary'], 'secondary': analysis['secondary'], 'bootstrap': matrix_record,
              'data': None if ids is None else _data_block(arms, ids, summary), 'models': {m: docs[m]['t009']['models'][m] for m in docs}, 'caveats': list(CAVEATS),
              'numeric_policy': _numeric_block(docs), 'provenance': _provenance_block(docs, start), 'code_identity': {'supervisor': start.get('code_identity'), **{m: read_docs[m].get('code_identity') for m in read_docs}},
              'stages': {m: dict(stages.get(m) or {'state': 'not_run'}) for m in NEURAL_MODELS}, 'start': {k: start.get(k) for k in ('approval', 'e6', 'gpu', 'wall_s', 'stage_caps_s', 'device', 'bundle_loader',
                                                                                                                             'identity_gate', 't009_run', 'started_wall')},
              'timing': timings_from_progress(out_dir / PROGRESS_FILE), 'files_sha256': files, 'store_evidence': {m: docs[m].get('store_evidence') for m in docs}}
    result['verdict'] = decide(result)
    return _plain(result)


# ---- the Korean report and the outputs -------------------------------------------------------------------------------------------------------------------
VERDICT_KO = {'GO': 'GO (진행)', 'INCONCLUSIVE': 'INCONCLUSIVE (결론 없음: 저자가 판단)', 'NO-GO': 'NO-GO (중단: 저자가 방향을 판단)', 'VOID': 'VOID (판정 없음)'}
CONTRAST_KO = {'ept_minus_b1': 'EPT − B1', 'n2_minus_b1': 'N2 − B1', 'ept_minus_n2': 'EPT − N2'}


def _num(value, digits: int = 6) -> str:
    return '없음' if value is None else f'{value:.{digits}f}'


def _span(interval) -> str:
    return '없음' if interval is None else f'[{interval[0]:.6f}, {interval[1]:.6f}]'


def _seconds(value) -> str:
    return '측정 없음' if value is None else f'{value:,.1f}'


def _cell_or_reason(cell: Mapping, text: str) -> str:
    return text if cell.get('status') == 'available' else f'없음 ({cell.get("reason")})'


def render_report_ko(result: Mapping) -> str:
    """The Korean report ``GATE0_KO.md`` of a result: the verdict and the rule it came from, the primary statistic, the secondary tables (never decisive), the integrity checks with their scope and the warnings
    that do not touch the verdict, the equivalence proof with its limits (``EQUIVALENCE_LIMITS``), the caveats, the timings (with the GPU state at the start and the note that they are not exclusive when the
    utilisation was above 0), the classification and the hashes. Every number is read from ``result``; the report
    decides nothing.

    해석: 판정과 그 근거 규칙, 주 통계, 보조 표, 정합성 점검(범위와 판정에 영향 없는 경고), 동등성 증명과 그 한계, 주의사항, 시간, 분류, 해시를 한국어로 적는다. 숫자는 모두 결과 기록에서 읽어 오며
    보고서가 판정을 바꾸지 않는다."""
    verdict, primary, secondary = result['verdict'], result['primary'], result['secondary']
    out = ['# EPT-T013 Gate 0 보고 (결정 D31)', '', f'분류: {result["classification"]}', '']
    out += ['## 1. 판정', '', f'- 판정: **{VERDICT_KO[verdict["verdict"]]}**', f'- 적용된 규칙(계산 전에 고정, 과제 기록): {verdict["rule_text"]}']
    if verdict['void_reasons']:
        out += ['- 판정을 내리지 못한 이유:'] + [f'  - {reason}' for reason in verdict['void_reasons']]
    out += [f'- 비고: {note}' for note in verdict['notes']]
    warnings = result['integrity'].get('warnings') or []
    out += [f'- 판정에 영향 없는 경고 {len(warnings)}건이 있다(4절). 판정은 주 통계가 기대는 점검만으로 정해졌다.'] if warnings else []
    out += ['- 판정에는 아래 주 통계 하나만 쓰였다. 보조 결과는 판정을 바꾸지 않는다.', '']
    out += ['## 2. 주 결과 (판정에 쓰인 유일한 통계)', '']
    if primary is None:
        out += ['EPT 팔이 끝나지 못해 주 통계가 없다.', '']
    else:
        data = result['data']
        out += ['| 항목 | 값 |', '|---|---|', f'| 끝점 | `{primary["endpoint"]}` (60초 안 첫 킬 발생 확률의 Brier) |',
                f'| Δ = Brier(EPT) − Brier(B1) (0보다 작으면 EPT 유리) | **{_num(primary["delta"])}** |',
                f'| 95% 구간 (경기 부트스트랩, 공통 10,000행 행렬의 앞 {primary["bootstrap_B"]:,}행, 시드 {result["bootstrap"]["seed"]}) | **{_span(primary["ci95"])}** |',
                f'| Brier(EPT) / Brier(B1) | {_num(primary["brier_first"])} / {_num(primary["brier_second"])} |',
                f'| 양측 p (조정 없음, 참고) | {_num(primary["p"], 4)} |',
                f'| 공통 STOP 경기 목록 / 주 모집단 행이 있는 경기 / 행 수 | {primary["n_matches"]:,} / {primary["n_present_matches"]:,} / {primary["n_rows"]:,} |',
                f'| 정의되지 않은 재표본 수 | {primary["draws_undefined"]} |', '']
        out += [f'- 자료: 15.20 패치 STOP 분할 전체, 주 모집단(t0 + H ≤ T − 30 s), 공칭 노출, 몬테카를로 없음, 배치 {data["batch_size"]}. DEV와 TEST는 열지 않았다.',
                f'- 선택 기록: {data["selection_note"]}', '']
    out += ['## 3. 보조 결과 (참고용, 판정을 바꾸지 않음)', '']
    if secondary is None:
        out += ['두 팔 모두 끝나지 못해 보조 결과가 없다.', '']
    else:
        out += ['### 3.1 Brier 차이 (앞 − 뒤, 0보다 작으면 앞 모형 유리)', '', '| 대조 | 지평 | Δ | 95% 구간 | Brier(앞) | Brier(뒤) | 경기(있음) | 행 |', '|---|---|---|---|---|---|---|---|']
        for name, by_horizon in secondary['brier'].items():
            for horizon, cell in by_horizon.items():
                if cell.get('status') != 'available':
                    out.append(f'| {CONTRAST_KO[name]} | {horizon} | {_cell_or_reason(cell, "")} | | | | | |')
                else:
                    mark = ' (주 통계)' if name == 'ept_minus_b1' and horizon == 'H60' else ''
                    out.append(f'| {CONTRAST_KO[name]}{mark} | {horizon} | {_num(cell["delta"])} | {_span(cell["ci95"])} | {_num(cell["brier_first"])} | {_num(cell["brier_second"])} | '
                               f'{cell["n_present_matches"]:,} | {cell["n_rows"]:,} |')
        out += ['', '### 3.2 Brier 기량 점수 (기준 = 같은 STOP 주 모집단 행의 발생률을 모든 행에 내는 상수 예측; 표본 밖 값이 아님)', '', '| 모형 | 지평 | 기량 점수 | Brier(모형) | Brier(기준) | 기준 발생률 |', '|---|---|---|---|---|---|']
        for model, by_horizon in secondary['brier_skill'].items():
            for horizon, cell in by_horizon.items():
                out.append(f'| {model.upper()} | {horizon} | ' + (f'{_num(cell["skill"])} | {_num(cell["brier_model"])} | {_num(cell["brier_base"])} | {_num(cell["base_rate"], 4)} |'
                                                                  if cell.get('status') == 'available' else f'{_cell_or_reason(cell, "")} | | | |'))
        out += ['', '### 3.3 머리별 NLL 차이 (앞 − 뒤, 같은 행, 머리마다 한 줄, 머리 사이 평균 없음)', '', secondary['nll_note'], '']
        for name, by_head in secondary['nll_per_head'].items():
            out += [f'#### {CONTRAST_KO[name]}', '', '| 머리 | Δ NLL (nat) | 95% 구간 | 평균 NLL(앞) | 평균 NLL(뒤) | 행 | 경기 |', '|---|---|---|---|---|---|---|']
            for head, cell in by_head.items():
                out.append(f'| {head} | ' + (f'{_num(cell["delta"])} | {_span(cell["ci95"])} | {_num(cell["mean_first"])} | {_num(cell["mean_second"])} | {cell["n_rows"]:,} | {cell["n_matches_present"]:,} |'
                                             if cell.get('status') == 'available' else f'{_cell_or_reason(cell, "")} | | | | | |'))
            out.append('')
        out += ['### 3.4 상대 구조 지표 (D16)', '', f'- {secondary["structural_relative"]["reason"]}', '']
    out += ['## 4. 정합성 점검', '', '범위 "주"는 주 통계가 기대는 점검이라 실패하면 판정이 VOID이고, "보조"는 N2 팔만 기대는 점검이라 실패하면 경고이며 N2 칸만 없어진다.', '', '| 점검 | 범위 | 결과 | 설명 |', '|---|---|---|---|']
    out += [f'| {check["name"]} | {"주" if check.get("scope") == "primary" else "보조"} | {"통과" if check["passed"] else "실패"} | {check["detail"]} |' for check in result['integrity']['checks']]
    if warnings:
        out += ['', '### 4.1 판정에 영향 없는 경고', ''] + [f'- {text}' for text in warnings]
    equivalence = result['equivalence']
    out += ['', '## 5. 동등성 증명 (T009 학습 커밋 ↔ 이 코드)', '', f'- 상태: **{equivalence["status"]}**; 비교한 키 {equivalence["keys_compared"]}개; 옛 커밋 `{str(equivalence["old_commit"])[:12]}`, 새 커밋 `{str(equivalence["new_commit"])[:12]}`.']
    out += [f'- 차이 {len(equivalence["differences"] or [])}개' + (': ' + '; '.join(f'{d["key"]} ({d.get("cause")})' for d in equivalence['differences']) if equivalence['differences'] else '') + '.']
    out += [f'- 설명: {equivalence["explanation"]}'] if equivalence.get('explanation') else []
    out += ['- 증명의 한계(결론은 바꾸지 않는다):'] + [f'  - {limit}' for limit in equivalence.get('limits') or []]
    out += ['', '## 6. 주의사항', ''] + [f'- {text}' for text in result['caveats']]
    out += ['', '## 7. 시간 (진행 기록에서 읽음; 측정되지 않은 값은 "측정 없음")', '']
    gpu = (result.get('start') or {}).get('gpu')
    if gpu:                                                                          # a run on the GPU: the state at the start (a run on the CPU has none)
        utilisation = gpu.get('utilization_percent')
        out += [f'- GPU 상태(시작 때): {gpu.get("name")}; 전체 {gpu["memory_total_mib"]:,} MiB, 사용 {gpu["memory_used_mib"]:,} MiB, 남음 {gpu["memory_free_mib"]:,} MiB; 사용률 '
                + ('읽지 못함' if utilisation is None else f'{utilisation}%') + '; 보이는 프로세스: ' + (', '.join(gpu.get('process_names') or []) or '없음') + '.']
        if utilisation is None:
            out += ['- 시작 때 GPU 사용률을 읽지 못했으므로 아래 시간이 GPU를 독점한 값인지 알 수 없다. 결정론적 설정에서 다른 GPU 사용자는 값을 바꾸지 않고 시간과 메모리 여유만 바꾼다.']
        elif utilisation > 0:
            out += [f'- 시작 때 GPU 사용률이 {utilisation}%였으므로 아래 시간은 GPU를 독점한 값이 아니다(다른 프로그램이 함께 썼다). 결정론적 설정에서 다른 GPU 사용자는 값을 바꾸지 않고 시간과 메모리 여유만 바꾼다.']
        out += ['']
    timing = result['timing']
    if timing.get('status') != 'available':
        out += ['진행 기록이 없다.', '']
    else:
        out += [f'- 실행 전체: {_seconds(timing.get("total_s"))} s', '', '| 모형 | 단계(감독자) s | 호출(arm) s | 적재 s | 추론 s |', '|---|---|---|---|---|']
        out += [f'| {m.upper()} | {_seconds(timing[m]["child_s"])} | {_seconds(timing[m]["arm_s"])} | {_seconds(timing[m]["load_s"])} | {_seconds(timing[m]["inference_s"])} |' for m in NEURAL_MODELS]
        for model in NEURAL_MODELS:
            blocks = timing[model]['blocks']
            if blocks:
                out += ['', f'#### {model.upper()} 머리 블록별', '', '| 블록 | s | 행 | 보기 | 상태 |', '|---|---|---|---|---|']
                out += [f'| {block} | {_seconds(entry["elapsed_s"])} | {entry["rows"]} | {entry["views"]} | {entry["status"]} |' for block, entry in blocks.items()]
        out.append('')
    numeric = result['numeric_policy']
    out += ['## 8. 숫자 설정', '', f'- 요구한 설정(T009와 같음): `{json.dumps(numeric["required"], ensure_ascii=False)}`']
    out += [f'- {m.upper()} 프로세스의 실제 설정: `{json.dumps({k: v for k, v in (record or {}).items()}, ensure_ascii=False)}`' for m, record in numeric['arms'].items()]
    out += ['- T009 영수증이 기록하지 않은 설정(코드의 `configure_float32()`로 켠 것을 믿음): ' + '; '.join(numeric['t009_not_recorded']) + '.', '']
    out += ['## 9. 출처와 해시', '']
    provenance = result['provenance']
    out += [f'- T009 실행 폴더: `{provenance["t009_run"]}`; 설정 sha256 `{provenance["config_sha256"]}`; 원천 커밋 `{provenance["source_commit"]}`.',
            f'- 이 실행의 코드 커밋: `{(result["code_identity"]["supervisor"] or {}).get("commit")}` (불러온 모듈 파일의 해시는 GATE0.json의 code_identity).']
    out += [f'- 재표본 행렬: 전체 {result["bootstrap"]["matrix_rows"]:,}행 sha256 `{result["bootstrap"]["matrix_sha256"]}`, 쓴 {result["bootstrap"]["B"]:,}행 sha256 `{result["bootstrap"]["used_sha256"]}`.'] if result['bootstrap'] else []
    out += [f'- 파일 해시: `{json.dumps(result["files_sha256"], ensure_ascii=False)}`', f'- 시작 승인 기록: {result["start"]["approval"]}', '']
    return '\n'.join(out)


def write_outputs(out_dir, result: Mapping) -> dict:
    """Write ``GATE0_KO.md`` (the Korean report) and ``GATE0.json`` (strict JSON: the whole result, with the hash of the report added) to ``out_dir`` and return ``{'GATE0.json': sha256, 'GATE0_KO.md':
    sha256}``. ``PROGRESS.jsonl`` is the durable timing record that the stages wrote while they ran (``run.evaluate.StageLog``, one line pair for each head block); it is not rewritten here, and its hash is in
    the result. ``Gate0Refused`` when either output exists (an output is never written over) and ``IORefused`` for a result that is not strict JSON.

    해석: 한국어 보고서를 먼저 쓰고 그 해시를 결과에 넣어 GATE0.json을 쓴다. 진행 기록은 실행 중에 이미 쓰였으므로 다시 쓰지 않는다."""
    out_dir = Path(out_dir)
    for name in (RESULT_FILE, REPORT_FILE):
        _require(not (out_dir / name).exists(), f'{name} exists: an output of the probe is never written over')
    text = render_report_ko(result)
    (out_dir / REPORT_FILE).write_text(text, encoding='utf-8', newline='\n')
    report_sha = run_io.sha_file(out_dir / REPORT_FILE)
    document = {**result, 'files_sha256': {**result['files_sha256'], REPORT_FILE: report_sha}}
    return {RESULT_FILE: run_io.write_json_atomic(out_dir / RESULT_FILE, document), REPORT_FILE: report_sha}


def _data_block(arms, ids, summary) -> dict:
    """What the statistics were made on: the split, the population, the exposure, the number of matches and rows, the early-stopping subset of T009 and the selection note."""
    first = next(iter(arms.values()))
    stop_subset = ((summary or {}).get('membership') or {}).get('stop_matches')
    return {'role': 'STOP', 'split': 'the whole 15.20 STOP split', 'n_matches': int(len(ids)), 'ids_sha256': _sha_of_array(np.asarray(ids, dtype='<i8')),
            'population': 'primary rows of each horizon (t0 + H <= T - 30 s, endpoints.primary_rows, D14)', 'exposure': 'nominal (10 s per occurrence bin; T enters no forecast)',
            'monte_carlo': False, 'dev_and_test_opened': False, 'batch_size': BATCH_SIZE,
            'endpoint_counts': {name: first['endpoints'][name]['counts'] for name in M2_ENDPOINTS},
            'early_stopping_subset_matches': stop_subset,
            'selection_note': (f'Both arms stopped early on the {stop_subset}-match STOP subset of T009 (membership.stop): the EPT through its STOP loss and B1 through fit_head(fit_rows, stop_rows). '
                               'The whole STOP split used here includes those matches for both arms alike; the numbers are the data of the go/no-go decision and not a held-out estimate.')}


def _numeric_block(docs) -> dict:
    """The numeric policy: what the probe requires, what each arm's process had, what the T009 receipts record and what they do not, and whether the torch version is the one of the training identity."""
    trained = {m: ((doc.get('t009') or {}).get('models') or {}).get(m) or {} for m, doc in docs.items()}
    recorded = {'configuration': next(((doc.get('t009') or {}).get('config_settings') for doc in docs.values()), None),
                'models': {m: {k: v for k, v in entry.items() if k.startswith('identity_')} for m, entry in trained.items()}}
    same_torch = {m: (docs[m].get('numeric') or {}).get('torch_version') == trained[m].get('identity_torch') for m in docs}
    return {'required': dict(REQUIRED_NUMERIC), 'arms': {m: doc.get('numeric') for m, doc in docs.items()}, 't009_recorded': recorded, 't009_not_recorded': list(NOT_RECORDED_BY_T009),
            'torch_version_equals_training_identity': same_torch}


def _provenance_block(docs, start) -> dict:
    """The T009 run that was read: its folder and configuration digest, the receipts, every file whose hash was checked against a receipt, the membership, the statistics and the scales."""
    summary = next((doc['t009'] for doc in docs.values()), {})
    verified = {}
    for doc in docs.values():
        verified.update(doc.get('verified') or {})
    return {'t009_run': start.get('t009_run'), 'config_sha256': summary.get('config_sha256'), 'source_commit': summary.get('source_commit'), 'profile': summary.get('profile'),
            'execution_authorized_in_config': summary.get('execution_authorized_in_config'), 'receipts': summary.get('receipts'), 'membership': summary.get('membership'),
            'stats_sha256': summary.get('stats_sha256'), 'scales_sha256': summary.get('scales_sha256'), 'b1': summary.get('b1'), 'verified_files': dict(sorted(verified.items()))}


# ---- the supervised entry ------------------------------------------------------------------------------------------------------------------------------
WALL_CAP_S = 6 * 3600                                                         # the whole run, the supervisor's budget (task record: a wall cap of 6 h)
STAGE_CAPS_S = {'ept': 3 * 3600, 'n2': 3 * 3600}                              # each arm's own most: the EPT stage first, then N2 with min(3 h, wall left - FINALISATION_S), so N2 can have less than 3 h; a cap hit on one stage does not hide the other's timing
FINALISATION_S = 900.0                                                        # kept for the statistics and the outputs after the last stage (the cleanup reserve)
GPU_MIN_FREE_MIB = 8192                                                       # a usable GPU: at least this much memory free (total - used); the desktop's own apps and utilisation are no reason to refuse (lead ruling after review m9)
MEMORY_BYTES = 40 * 1024 ** 3                                                 # the memory budget of the T009 campaign
RUNS_ROOT = Path('D:/LOL_Data/runs')
OUT_NAME = re.compile(r'\d{8}_c_ept_gate0_\d+')
IDENTITY_GATES = ('start_gate', 'skipped')                                    # 'skipped' is for the synthetic tests; a run that skipped the gate has no decision (check ``*_production_inputs``)
START_SCHEMA = 'ept-gate0-start-v1'
STAGE_SCHEMA_OUT = 'ept-gate0-stage-v1'
EXIT_OK, EXIT_REFUSED, EXIT_FAILED = 0, 2, 3                                  # the exit codes of the kernel (``supervisor``): a refusal is 2, anything that did not finish or has no decision is 3


def _resolve_callable(spec):
    """The function a stage is told by name (``'module:function'``); a function is returned as it is (in-process tests only)."""
    if callable(spec):
        return spec
    import importlib
    module, _, name = str(spec).partition(':')
    _require(module and name, 'a loader is named module:function')
    target = importlib.import_module(module)
    for part in name.split('.'):
        target = getattr(target, part)
    _require(callable(target), 'a loader must be callable')
    return target


# The entry modules of the code identity of a stage: ``identity.capture`` imports them and hashes every repository module that is loaded then. The stage imports most of what it uses late (the loader of
# the T009 run, the inference, the stores and the models), so they are named here and the identity at the start already holds them; the identity at the end must hold the same files with the same hashes.
STAGE_ENTRY_MODULES = ('c_ept.run.gate0', 'c_ept.run.evaluate', 'c_ept.evaluation.adapters', 'c_ept.evaluation.runtime', 'c_ept.evaluation.stores', 'c_ept.comparison.runtime', 'c_ept.comparison.data',
                       'c_ept.comparison.models', 'c_ept.ept.data', 'c_ept.ept.heads', 'c_ept.ept.encoder', 'c_ept.ept.loss', 'c_ept.ept.scales', 'c_ept.models.baselines_b1', 'c_ept.models.neural.team_set',
                       'c_ept.preprocessing.terrain', 'c_ept.valuation.value', 'c_ept.valuation.towertier', 'c_ept.valuation.compose', 'c_ept.legacy')


class _Guard:
    """The ``check`` that a stage hands to everything that can run long (the hashing of the T009 files, ``run_inference``): at most once in ``interval_s`` seconds it re-hashes the imported module files
    against the identity that was captured at the start (``identity.verify``: a file that changed raises ``IdentityChanged``, a file that cannot be read now raises ``Paused``). The wall cap is the
    supervisor's, which kills the process tree.

    해석: 단계 안에서는 불러온 코드 파일이 시작 때와 같은지만 가끔 확인한다. 벽시계 상한은 감독자가 프로세스 트리를 끊는 것으로 지킨다."""

    def __init__(self, identity, repo, *, interval_s: float = 60.0, clock=time.monotonic):
        self.identity, self.repo, self.interval_s, self.clock = identity, repo, interval_s, clock
        self.last = clock()

    def __call__(self) -> None:
        now = self.clock()
        if now - self.last >= self.interval_s:
            run_identity.verify(self.identity, repo=self.repo)
            self.last = now


def stage_main(run_dir, model: str, t009_run, device: str = 'cuda', bundle_loader=DEFAULT_BUNDLE_LOADER, identity_gate: str = 'start_gate', threads=None) -> None:
    """The body of one supervised stage (the child process of the supervisor calls ``stage_main(run_dir, *args)``): the arm of ``model`` ('ept' or 'n2') with the matched B1. Nothing is read before the
    code identity of the stage is captured (and, unless ``identity_gate`` is 'skipped', passed the start gate: the HEAD commit and the imported files unmodified in Git) and the numeric policy of T009 is set.
    Then ``load_t009`` (every hash checked), the STOP bundle, ``evaluate_gate0`` with the progress record, the verification of the stores that were read, and ``<run_dir>/<model>/ARRAYS.npz`` and
    ``ARM.json`` (the call's provenance, the settings, the numeric record, the T009 summary and verified hashes, the code identity and the store evidence). A new folder: a stage is never run twice into one.
    The modules that are imported late are in the identity at the end and not in the one at the start; that final identity passes the start gate too (Git HEAD, files unmodified) before anything is saved, so a
    modified late import leaves no arm (review m2).

    ``bundle_loader``: ``'module:function'`` of the verified-store builder (default ``comparison.runtime.load_bundle``). It is a seam for the synthetic tests and is recorded; a run with another loader (or with the
    identity gate skipped) fails its integrity check ``*_production_inputs`` and has no decision.

    해석: 한 모형(EPT 또는 N2)과 B1을 STOP 주 모집단에서 닫힌 꼴로 평가하는 하위 프로세스 본문이다. 코드 정체성과 숫자 설정을 먼저 고정하고, 해시를 확인한 자료만 읽으며, 결과 배열과 근거를 한 폴더에 남긴다.
    늦게 불러온 모듈은 시작 때의 정체성에 없으므로 끝의 정체성에도 시작 관문(Git에 있고 수정 없음)을 한 번 더 적용한 뒤에만 팔을 저장한다."""
    from c_ept.run import evaluate as run_evaluate
    _require(model in NEURAL_MODELS, f'the stage model must be one of {NEURAL_MODELS}')
    _require(identity_gate in IDENTITY_GATES, f'identity_gate must be one of {IDENTITY_GATES}')
    _require(device in ('cuda', 'cpu'), "the device must be 'cuda' or 'cpu'")
    _require(threads is None or re.fullmatch(r'[0-9]{1,4}', str(threads)) is not None and int(threads) >= 1, 'threads must be a whole number of at least 1')
    run_dir = Path(run_dir)
    folder = run_dir / model
    _require(not (folder / ARM_FILE).exists() and not (folder / ARRAYS_FILE).exists(), f'the {model} stage has already left its outputs: a stage is never run twice into one folder')
    repo = Path(__file__).resolve().parents[2]
    log = run_evaluate.StageLog(run_dir / PROGRESS_FILE)
    with log.stage('arm', model, detail={'role': 'STOP', 'device': device}) as counts:
        identity = run_identity.capture(STAGE_ENTRY_MODULES, repo=repo)
        if identity_gate == 'start_gate':
            run_identity.start_gate(repo, identity)
        guard = _Guard(identity, repo)
        numeric = apply_numeric_policy()
        import torch
        torch.set_num_threads(int(threads) if threads is not None else comparison_protocol.FIXED['native_threads'])         # the T009 worker did the same with its configuration's 8
        if device == 'cuda':
            _require(torch.cuda.is_available(), 'CUDA is required for this stage; there is no CPU fallback')
        with log.stage('load', model):
            parts = load_t009(t009_run, device=device, check=guard, models=(model,), bundle_loader=_resolve_callable(bundle_loader))
        bundle = stop_bundle(parts.bundle)
        ids = stop_match_ids(bundle.s1)
        results = evaluate_gate0(bundle, parts.models, parts.stats, parts.b1, device=device, check=guard, scales=parts.scales, log=log)
        evidence = {}
        for key in ('s1', 's2', 'labels'):                                                  # the stores that were read are still the files that were hashed when they were opened
            store = getattr(parts.bundle, key, None)
            if hasattr(store, 'verify_used'):
                store.verify_used()
                evidence[key] = store.used_evidence()
        arm = arm_from_inference(model, results[model], ids)
        run_identity.verify(identity, repo=repo)                                           # the code did not change while the stage ran
        final = run_identity.capture(STAGE_ENTRY_MODULES, repo=repo)                       # the import closure now: every module the stage really imported, also the ones that were imported late
        _require(final.commit == identity.commit and all(final.files.get(path) == sha for path, sha in identity.files.items()), 'the imported code changed while the stage ran')
        if identity_gate == 'start_gate':
            run_identity.start_gate(repo, final)                                           # the files that came in late (not in the identity of the start) are in Git, unmodified, too; nothing is saved before this
        manifest = save_arm(folder, arm)
        counts.update(rows=int(sum(len(e['row']) for e in arm['endpoints'].values())), matches=int(len(ids)))
        run_io.write_json_atomic(folder / ARM_FILE, _plain({
            'schema': ARM_SCHEMA, 'model': model, 'manifest': manifest, 'provenance': arm['provenance'], 'numeric': numeric, 't009': parts.summary, 'verified': parts.verified,
            'settings': {'batch_size': BATCH_SIZE, 'device': device, 'bundle_loader': bundle_loader if isinstance(bundle_loader, str) else 'a function', 'identity_gate': identity_gate, **INFERENCE_SETTINGS},
            'code_identity': final.to_record(), 'code_identity_at_start_sha256': run_io.canonical_sha256(identity.to_record()), 'store_evidence': evidence}))


class _StageCheck:
    """The ``check`` of ``run_owned`` for one stage: the stage's own wall share and the supervisor's budget (wall and memory of the process tree, sampled at most every ``interval_s`` seconds), both turned
    into ``BudgetStop`` (a stop during the run is a failure of the computation, not a refusal). The clock is read at every call, so a cap that has passed is found at the next poll.

    해석: 단계마다 정해 둔 벽시계 몫과 감독자의 전체 예산을 본다. 넘으면 자식 트리를 끊는다."""

    def __init__(self, budget, cap_s: float, *, clock=time.monotonic, interval_s: float = 5.0):
        self.budget, self.cap_s, self.clock, self.interval_s, self.began, self.last = budget, cap_s, clock, interval_s, clock(), None
        self.stop = supervisor.BudgetStop

    def __call__(self) -> None:
        now = self.clock()
        if now - self.began > self.cap_s:
            raise self.stop(f'the wall cap of this stage ({self.cap_s:g} s) is used up')
        if self.last is None or now - self.last >= self.interval_s or self.budget.elapsed_s() > self.budget.wall_s:
            try:
                self.budget.check()
            except guards.GateRefused as refusal:
                if refusal.gate == 'budget':
                    raise self.stop(str(refusal)) from None
                raise
            self.last = now


def s1_match_keys(t009_run) -> list:
    """The match keys of the S1 store of the T009 run (``matches/key``), for the scrubbing of every text that a run prints or records (the rule of the kernel: a key reaches no terminal, log or record). Read
    through the verified store of the loader; ``Gate0Refused`` when they cannot be read (a run that cannot scrub does not start).

    해석: 경기 키는 화면, 로그, 기록 어디에도 나가지 않아야 한다. 오류 글에서 키를 가리려면 키 목록이 필요하므로 S1 저장소에서 읽고, 못 읽으면 시작하지 않는다."""
    from c_ept.evaluation.stores import ReadOnlyVerifiedStore
    with _refusing():
        meta = _read_json(Path(t009_run) / 'prepare' / 'inputs.json')
        store = ReadOnlyVerifiedStore(meta['paths']['s1'], check=lambda: None)
        return [key.decode('ascii') if isinstance(key, bytes) else str(key) for key in np.asarray(store.array('matches/key')).tolist() if key]


def _stage_record(state: str, *, reason=None, exit_code=None, elapsed_s=None, cap_s=None, arm_sha256=None) -> dict:
    return {'schema': STAGE_SCHEMA_OUT, 'state': state, 'reason': reason, 'exit_code': exit_code, 'elapsed_s': elapsed_s, 'cap_s': cap_s, 'arm_sha256': arm_sha256}


def gpu_state(run=None, find=None) -> dict:
    """The state of the GPU from ``nvidia-smi``: ``{'name', 'memory_total_mib', 'memory_used_mib', 'memory_free_mib', 'utilization_percent', 'process_names'}`` for the first GPU listed (the card the children
    get: device 0). The memory query (``--query-gpu=name,memory.used,memory.total,utilization.gpu``) must answer: ``Gate0Refused`` when ``nvidia-smi`` is not there, fails or gives something that is not that
    table or a memory that is not a number. ``memory_free_mib = total - used``. ``utilization_percent`` is None when nvidia-smi says N/A. ``process_names`` are the base names of the processes that nvidia-smi lists
    (the compute-apps query and the ``.exe`` names of the Processes table of the plain call), sorted and unique; the lists are information only: a call that fails or a name that is not there leaves ``[]`` and
    never refuses (on Windows the per-process memory is N/A and the list is not reliable). ``run`` and ``find`` are seams for the tests (the default ``subprocess.run`` and a lookup of ``nvidia-smi`` on the
    PATH; ``find()`` gives the path or None).

    수식: memory_free = memory_total - memory_used (MiB).
    해석: GPU가 쓸 만한지 nvidia-smi로 읽는다. 결정은 남은 메모리 하나로만 하고, 사용률과 보이는 프로세스 이름은 기록용이다(결정론적 설정에서는 다른 GPU 사용자가 값을 바꾸지 않고 시간과 메모리 여유만 바꾼다).
    읽을 수 없으면 확인할 수 없으므로 시작하지 않는다."""
    import shutil
    import subprocess
    run = subprocess.run if run is None else run
    path = (find if find is not None else (lambda: shutil.which('nvidia-smi')))()
    _require(path, 'nvidia-smi was not found: the GPU cannot be checked')

    def ask(arguments, *, required: bool) -> str:
        try:
            done = run([path, *arguments], capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            if required:
                raise Gate0Refused(f'nvidia-smi could not be run ({type(error).__name__}): the GPU cannot be checked') from None
            return ''
        if done.returncode != 0:
            _require(not required, f'nvidia-smi failed with the code {done.returncode}: the GPU cannot be checked')
            return ''
        return done.stdout or ''
    lines = [line for line in ask(['--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], required=True).splitlines() if line.strip()]
    parts = [part.strip() for part in lines[0].split(',')] if lines else []
    _require(len(parts) >= 4 and all(re.fullmatch(r'[0-9]{1,9}', part) for part in parts[-3:-1]) and re.fullmatch(r'[0-9]{1,3}|\[N/A\]|N/A', parts[-1]) is not None,
             'nvidia-smi answered something that is not a table of GPU name, memory used, memory total and utilisation: the GPU cannot be checked')
    used, total = int(parts[-3]), int(parts[-2])
    names = set()
    for line in ask(['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], required=False).splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) >= 3 and fields[1]:
            names.add(re.split(r'[\\/]', ','.join(fields[1:-1]).strip())[-1])
    plain = ask([], required=False)
    names.update(re.findall(r'([^\s\\/|]+\.exe)', plain.partition('Processes:')[2], flags=re.IGNORECASE))
    return {'name': ','.join(parts[:-3]), 'memory_total_mib': total, 'memory_used_mib': used, 'memory_free_mib': total - used, 'utilization_percent': int(parts[-1]) if parts[-1].isdigit() else None,
            'process_names': sorted(name for name in names if name and not name.startswith('['))}


def check_gpu_free(state: Mapping, min_free_mib: int = GPU_MIN_FREE_MIB) -> Mapping:
    """``state`` (``gpu_state``) when at least ``min_free_mib`` (8,192) MiB of the GPU memory are free; ``Gate0Refused`` otherwise. Exactly ``min_free_mib`` free passes. Nothing else is a condition: not the
    utilisation, not the processes that are listed (the desktop's apps, a game client).

    해석: 남은 메모리(전체 − 사용)가 8,192 MiB 이상이면 시작한다. 정확히 8,192 MiB여도 통과한다. 데스크톱 앱이 켜져 있거나 사용률이 0이 아니어도 거부하지 않는다: 결정론적 설정에서는 값이 같고 시간만
    독점이 아니게 되므로(보고서에 사용률을 적는다) 메모리 여유만 본다."""
    _require(state['memory_free_mib'] >= min_free_mib, f'not enough free GPU memory: {state["memory_free_mib"]} MiB free ({state["memory_used_mib"]} of {state["memory_total_mib"]} MiB used), at least {min_free_mib} MiB are needed')
    return state


def supervise(t009_run, out_dir, *, equivalence, approval: str, device: str = 'cuda', wall_s: float = WALL_CAP_S, stage_caps_s=None, runs_root=None, repo=None, e6=None, measure=None,
              clock=time.monotonic, poll_s: float = 1.0, cuda_visible_devices: str = '0', threads: int = 8, bundle_loader: str = DEFAULT_BUNDLE_LOADER, identity_gate: str = 'start_gate',
              scrub_keys=(), stream=None, environ=None, capture=None, start_gate=None, verify=None, gpu_probe=None) -> int:
    """Run the probe under supervision and return the exit code (0: a verdict of GO, INCONCLUSIVE or NO-GO was written; 3: the outputs were written but there is no decision (VOID: a stage of the primary
    statistic did not finish or a check it depends on failed); 2: a refusal before anything was made, nothing is created and nothing computed is lost). The order (task record, "Supervision"):

    1. the gates: ``guards.native_policy``, the equivalence record (``read_equivalence_bytes``), the paths (``out_dir`` a new ``<YYYYMMDD>_c_ept_gate0_<n>`` folder of ``runs_root``, the T009 run a folder with its
       ``RUN.json``), the checks of the T009 record (``check_t009_record``: schema, T009 schema v2 configuration, legacy pins, digest, output), E6 (no model server runs; Strata is off), the child environment
       (deterministic float32 GPU as in T009: ``CUBLAS_WORKSPACE_CONFIG``, 8 threads, device 0), a ``guards.Budget`` of ``wall_s`` (6 h) and 40 GiB, the code identity of this process (``identity.capture`` of
       this module) and the binding of the equivalence record to what runs (its ``new_commit`` is this HEAD, its ``old_commit`` is the ``source_commit`` of the T009 configuration), then the start gate (HEAD and
       the imported files unmodified in Git) and, on the GPU, the check of the free GPU memory (``gpu_probe``, default ``gpu_state``; at least ``GPU_MIN_FREE_MIB``; not called for the CPU; its answer, with the utilisation and the visible process names, goes to START.json). Any refusal here is ``Gate0Refused``, exit code 2, and nothing is created;
    2. then ``out_dir`` is made, ``START.json`` (the approval text, E6, the GPU state, the identity, the caps) and the copy of the equivalence record (the bytes that were read and hashed) are written, and the progress
       record starts;
    3. one stage per model, the EPT with the matched B1 first and then N2 with it, each its own child process owned by a Windows Job (``supervisor.run_owned``), each with its own wall share and receipt
       (``<model>/STAGE.json``: state, exit code, seconds, hash of ``ARM.json``): the EPT at most ``stage_caps_s['ept']`` (3 h), N2 at most ``min(stage_caps_s['n2'], wall left - 15 min)`` (so less than 3 h when the EPT
       stage was slow); E6 and the identity are checked before each stage; a stage that is refused (exit 2) stops the run, a stage that fails or hits its cap does not stop the other from running (so its timing is
       not hidden); the end of the child's output is scrubbed of the match keys ``scrub_keys`` and kept in ``logs/<model>.log``;
    4. the statistics, the verdict and ``GATE0.json`` / ``GATE0_KO.md`` (``assemble``, ``write_outputs``).

    ``approval``: the text of the author's start approval of this run, recorded as it is (this probe does not require the ``execution_authorized`` flag of the T009 configuration). The remaining keywords
    are seams for the tests (``e6``, ``measure``, ``clock``, ``capture``, ``start_gate``, ``verify``, ``repo``, ``environ``, ``runs_root``, ``bundle_loader``, ``identity_gate``, ``gpu_probe``) and the device settings
    (``device``, ``cuda_visible_devices``, ``threads``): a run with a bundle loader other than the production one or with the identity gate skipped writes its outputs but has no decision.

    해석: 승인 문구, 동등성 증명 기록과 두 커밋의 묶음, T009 기록의 형식, E6, 빈 GPU, 새 출력 폴더, 코드 정체성을 모두 확인한 뒤에만 출력 폴더를 만든다. 하나라도 어긋나면 종료 코드 2이고 아무것도 만들지 않아
    GPU 시간도 폴더 이름도 잃지 않는다. EPT와 N2를 각자의 하위 프로세스(Windows Job 소유)와 각자의 벽시계 몫으로 따로 돌려 한쪽이 상한에 걸려도 다른 쪽의 시간이 남고, 끝나면 통계, 판정, 한국어
    보고서를 쓴다. 주 통계가 기대는 점검이 실패하면(VOID) 종료 코드 3이다. 보조 팔 N2만 실패하면 판정은 그대로이다.
    """
    guards.native_policy()
    from c_ept.run import evaluate as run_evaluate
    _require(isinstance(approval, str) and approval.strip(), 'the text of the start approval is required (it is recorded as given)')
    _require(device in ('cuda', 'cpu') and identity_gate in IDENTITY_GATES, 'a device of cuda or cpu and an identity gate of start_gate or skipped are required')
    caps = dict(STAGE_CAPS_S if stage_caps_s is None else stage_caps_s)
    _require(set(caps) == set(NEURAL_MODELS) and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in caps.values()), 'the stage caps must give a positive number of seconds for each model')
    t009_run, out_dir = Path(os.path.abspath(t009_run)), Path(os.path.abspath(out_dir))
    root = Path(os.path.abspath(RUNS_ROOT if runs_root is None else runs_root))
    _require(OUT_NAME.fullmatch(out_dir.name) is not None and out_dir.parent == root, f'the output must be a new folder <YYYYMMDD>_c_ept_gate0_<n> of {root.name}')
    _require(not os.path.lexists(out_dir), 'the output folder exists: a run is never written into an existing folder')
    _require(out_dir != t009_run and t009_run not in out_dir.parents and out_dir not in t009_run.parents, 'the output and the T009 run must not overlap')
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    equivalence_record, equivalence_raw = read_equivalence_bytes(equivalence)
    run_record = _read_json(t009_run / RUN_FILE)
    config = check_t009_record(t009_run, run_record)                                   # what the GPU stages would find out late (schema v2, legacy pins, digest, output) is found out here
    source_commit = config.get('source_commit')
    _require(isinstance(source_commit, str) and equivalence_record['old_commit'] == source_commit,
             f'the equivalence record does not compare the T009 source commit of this run (record: {equivalence_record["old_commit"][:12]}, T009 source commit: {str(source_commit)[:12]})')
    e6_record = (e6 if e6 is not None else guards.e6)()
    env = guards.child_environment(environ if environ is not None else os.environ, cuda_visible_devices=cuda_visible_devices, threads=threads)
    legacy_root = ((config.get('inputs') or {}).get('legacy') or {}).get('root')
    if isinstance(legacy_root, str) and legacy_root:                                   # the T009 operator gave its children the legacy rule folder of its configuration the same way
        env['C_EPT_LEGACY_ROOT'] = legacy_root
    import shutil
    _require(shutil.which('git', path=env['PATH']) is not None, 'git is not on the PATH that the children get (their identity check calls it)')
    budget = guards.Budget(wall_s, MEMORY_BYTES, clock=clock, measure=measure)
    identity = (capture if capture is not None else run_identity.capture)(('c_ept.run.gate0',), repo=repo)
    _require(equivalence_record['new_commit'] == identity.commit,
             f'the equivalence record was made for the commit {equivalence_record["new_commit"][:12]} and this code is the commit {str(identity.commit)[:12]}: make the record again for this commit')
    if identity_gate == 'start_gate':
        (start_gate if start_gate is not None else run_identity.start_gate)(repo, identity)
    gpu = check_gpu_free((gpu_probe if gpu_probe is not None else gpu_state)()) if device == 'cuda' else None
    budget.check()
    out_dir.mkdir(parents=True)                                                       # the reservation: nothing above wrote anything
    run_io.write_json_atomic(out_dir / START_FILE, {
        'schema': START_SCHEMA, 't009_run': str(t009_run), 't009_config_sha256': run_record['config_sha256'], 'approval': approval, 'e6': e6_record, 'gpu': gpu, 'code_identity': identity.to_record(),
        'wall_s': wall_s, 'stage_caps_s': caps, 'memory_bytes': MEMORY_BYTES, 'device': device, 'bundle_loader': bundle_loader, 'identity_gate': identity_gate, 'started_wall': time.time(),
        'equivalence_sha256': equivalence_record['file_sha256'], 'child_settings': {'cuda_visible_devices': cuda_visible_devices, 'threads': threads, 'cublas_workspace_config': env.get('CUBLAS_WORKSPACE_CONFIG'), 'legacy_root': env.get('C_EPT_LEGACY_ROOT')}})
    with (out_dir / 'EQUIVALENCE.json').open('xb') as copy:                           # the bytes that were parsed and hashed, not the file as it is now
        copy.write(equivalence_raw)
    _require(run_io.sha_file(out_dir / 'EQUIVALENCE.json') == equivalence_record['file_sha256'], 'the copy of the equivalence record differs from the record that was read')
    start = _read_json(out_dir / START_FILE)
    keys = list(scrub_keys)
    log = run_evaluate.StageLog(out_dir / PROGRESS_FILE, wall=time.time, monotonic=clock)
    stages, refused = {}, False
    with log.stage('gate0', detail={'role': 'STOP'}) as counts:
        for model in NEURAL_MODELS:
            if refused:
                stages[model] = _stage_record('not_run', reason='an earlier stage was refused')
                continue
            stages[model] = _run_stage(model, out_dir, t009_run, repo=repo, env=env, budget=budget, cap_s=caps[model], device=device, bundle_loader=bundle_loader, identity_gate=identity_gate, threads=threads, identity=identity,
                                       e6=e6 if e6 is not None else guards.e6, verify=verify if verify is not None else run_identity.verify, log=log, keys=keys, clock=clock, poll_s=poll_s)
            refused = stages[model]['state'] == 'refused'
        counts.update(stages_completed=sum(1 for s in stages.values() if s['state'] == 'completed'))
    with log.stage('analysis'):                                                      # after the line of the whole run, so that GATE0.json holds the seconds of the stages
        result = assemble(out_dir, stages=stages, equivalence=equivalence_record, start=start)
        write_outputs(out_dir, result)
    verdict = result['verdict']
    print(f'Gate 0 verdict: {verdict["verdict"]}' + (f' ({"; ".join(verdict["void_reasons"])})' if verdict['void_reasons'] else '') + f'; outputs in {out_dir.name}', file=stream if stream is not None else sys.stdout, flush=True)
    return EXIT_OK if verdict['verdict'] != 'VOID' else EXIT_FAILED


def _run_stage(model, out_dir, t009_run, *, repo, env, budget, cap_s, device, bundle_loader, identity_gate, threads, identity, e6, verify, log, keys, clock, poll_s) -> dict:
    """One stage of ``supervise``: E6 and the identity first, the child in its Job under its wall share, the receipt ``<model>/STAGE.json`` and the scrubbed end of its output in ``logs/<model>.log``. Returns the
    receipt (also written). ``BudgetStop`` (the stage's share or the supervisor's budget used up) is a state of the receipt ('cap'), not an error of the supervisor; an interrupt is.

    The wall share of the stage is ``cap = min(cap_s, wall_s - elapsed - FINALISATION_S)``: the EPT stage runs first and has at most ``cap_s`` (3 h); N2 has what is left of the 6 h after a 15 min reserve for
    the statistics and the outputs, at most ``cap_s``, so it has less than 3 h when the EPT stage took long. A share that is used up before the stage starts is the state 'not_run'.

    수식: cap = min(cap_s, W - t_경과 - 900 s)  (W = 6 h 벽시계 예산, cap_s = 3 h). 상태: 'refused'(E6나 코드 정체성이 막음, 또는 자식의 종료 코드 2), 'not_run'(몫이 없음), 'cap'(몫을 다 씀),
    'failed'(0이 아닌 종료 코드 또는 ARM.json 없음), 'completed'(종료 코드 0과 ARM.json).
    해석: 단계 하나를 자기 프로세스 트리에서 자기 몫의 시간만큼만 돌린다. 앞 단계(EPT)가 오래 걸리면 뒤 단계(N2)의 몫이 3시간보다 줄 수 있고, 몫이 없으면 돌리지 않는다. 막히거나 상한에 걸린 단계도
    영수증과 출력 끝부분(경기 키를 가린 것)을 남겨 시간이 숨지 않게 한다."""
    folder = out_dir / model
    folder.mkdir()
    try:
        e6()
        verify(identity, repo=repo)
    except (guards.GateRefused, run_identity.IdentityChanged, run_identity.Paused) as refusal:         # a gate that refuses between the stages ends the run with its outputs, not with nothing
        record = _stage_record('refused', reason=f'{type(refusal).__name__}: {refusal}')
        run_io.write_json_atomic(folder / STAGE_FILE, record)
        return record
    left = budget.wall_s - budget.elapsed_s() - FINALISATION_S
    if left <= 0:
        record = _stage_record('not_run', reason='the wall budget is used up', cap_s=0.0)
        run_io.write_json_atomic(folder / STAGE_FILE, record)
        return record
    cap = min(cap_s, left)
    argv = [sys.executable, '-I', '-X', 'utf8', '-B', '-u', '-c', supervisor._BOOTSTRAP, str(repo), 'c_ept.run.gate0:stage_main', str(out_dir), model, model, str(t009_run), device, bundle_loader, identity_gate, str(threads)]
    began, partial, result, stop = clock(), None, None, None
    try:
        with log.stage('child', model, detail={'role': 'STOP', 'device': device}):
            result = supervisor.run_owned(argv, env=env, cwd=repo, check=_StageCheck(budget, cap, clock=clock), poll_s=poll_s)
    except supervisor.BudgetStop as budget_stop:
        stop, partial = str(budget_stop), getattr(budget_stop, 'child_output', None)
    elapsed = clock() - began
    if result is not None:
        partial = (result.output, result.truncated)
    if partial is not None:
        logs = out_dir / 'logs'
        logs.mkdir(exist_ok=True)
        (logs / f'{model}.log').write_text(supervisor.scrub_tail(*partial, keys), encoding='utf-8', newline='\n')
    if stop is not None:
        record = _stage_record('cap', reason=stop, elapsed_s=elapsed, cap_s=cap)
    elif result.returncode == 0 and (folder / ARM_FILE).is_file():
        record = _stage_record('completed', exit_code=0, elapsed_s=elapsed, cap_s=cap, arm_sha256=run_io.sha_file(folder / ARM_FILE))
    elif result.returncode == 0:
        record = _stage_record('failed', reason='the stage exited with 0 but left no ARM.json', exit_code=0, elapsed_s=elapsed, cap_s=cap)
    else:
        record = _stage_record('refused' if result.returncode == supervisor.EXIT_REFUSED else 'failed', reason=f'the stage exited with code {result.returncode} (see logs/{model}.log)',
                               exit_code=result.returncode, elapsed_s=elapsed, cap_s=cap)
    run_io.write_json_atomic(folder / STAGE_FILE, record)
    return record


def main(argv=None) -> int:
    """The command line: ``--t009 <T009 run folder> --out <new gate0 folder> --equivalence <record> --approval <text>`` (always on the GPU). Start it in a new process that applies the native-thread policy
    before numpy or torch is imported::

        python -B -c "from c_ept.run import guards; guards.native_policy(); from c_ept.run import gate0; raise SystemExit(gate0.main())" --t009 ... --out ... --equivalence ... --approval ...

    Refusals and failures are printed with the match keys of the S1 store removed (``supervisor.main_guard``) and the exit code is the kernel's (0, 2, 3).

    해석: 실행 명령이다. 네이티브 정책을 먼저 적용한 새 프로세스에서 시작하고, 오류 글에서는 경기 키를 가린다."""
    import argparse
    parser = argparse.ArgumentParser(description='EPT-T013 Gate 0 probe: the EPT-S encoder against the matched B1 on the STOP split (supervised).')
    for name in ('--t009', '--out', '--equivalence', '--approval'):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)

    def run() -> int:
        keys = s1_match_keys(args.t009)
        return supervisor.main_guard(lambda: supervise(args.t009, args.out, equivalence=args.equivalence, approval=args.approval, scrub_keys=keys), scrub_keys=keys)
    return supervisor.main_guard(run)
