r"""[모듈 개요 및 하네스 위치]
이 모듈은 기준 모형(B0, B1)과 본 EPT 신경망 모형이 공유하는 표준화된 음의 로그 우도(NLL) 및
통계적 평가 지표(가중 평균, 경기 단위 군집 부트스트랩)를 계산하는 핵심 채점 엔진이다 (stage S3 Task 5).

A. 하네스 및 파이프라인 내 위치:
   - 모든 모형의 머리(heads)는 공통된 확률 분포 족(Probability Family)과 구간 분할 하에서
     칸(cell) 단위 NLL로 채점된다 (지침 F3, 결정 P4). 이를 통해 상이한 아키텍처의 성능 수치를 일대일 직접 비교할 수 있다.
   - 이 모듈은 특정 모델 구조나 저장소 구조에 종속되지 않고, 오직 순수 넘파이 배열 연산으로 엄밀하게 작동한다.

B. 모듈 블록 구성 및 평가 지표 체계:
   - 블록 1: 전역 상수 및 설정 (부트스트랩 표본 수, 고정 시드, 31개 골드 구간 등)
   - 블록 2: 노출(Exposure)을 고려한 포아송 음의 로그 우도 계산 (poisson; 식 NLL-1)
   - 블록 3: 다중 범주형 및 31구간 킬 골드 음의 로그 우도 계산 (categorical, gold; 식 NLL-2, NLL-3)
   - 블록 4: 경기 단위 가중 평균 및 군집 부트스트랩 신뢰구간 복제 (weighted_mean, match_bootstrap; 식 NLL-4, NLL-5)

C. 결측치 및 미관측 칸 처리 규약:
   1) 노출 0 (exposure = 0, poisson):
      - 경기가 조기 종료되었거나 전환 창이 비어 있어 관측 시간이 0인 칸.
      - 우도에 아무런 기여를 하지 않으므로 NLL = 0.0을 부여하며, 발생률(rate)은 읽지도 않는다.
      - 단, 노출이 0인데 사건 개수 y > 0이 기록되어 있다면 명백한 데이터 오류이므로 ValueError를 발생시킨다.
   2) 결측 범주 (idx = -1, categorical, gold):
      - 첫 킬의 좌표가 미관측된 구역(zone = -1)이거나, 해당 팀의 킬이 0이어서 킬 골드가 발생하지 않은 경우(gold bin = -1).
      - 확률표의 해당 행을 읽지 않고 NLL = 0.0을 부여하며, 호출자가 미평가 항목 수를 별도로 집계 보고한다.
   3) log(y!) 항 포함:
      - 모형 의존 항은 아니지만, 타 확률 족(예: 음이항 분포)과의 직접 NLL 비교를 위해 엄밀한 확률질량함수 음의 로그를 반환.
   4) 무한대 손실(+inf) 보존:
      - 모형이 실제 발생한 사건에 확률 0을 부여한 경우 참 손실값인 +inf를 그대로 반환하며,
        가중 평균 및 부트스트랩 계산 시 이를 조용히 평균내거나 NaN으로 은폐하지 않고 엄격히 차단한다.

D. 핵심 수식 (NLL-1 ~ NLL-5):
   (NLL-1)  포아송 칸 점수     ℓ_i = μ_i − y_i · log μ_i + log(y_i!),  μ_i = λ_i · e_i   (e_i > 0);     ℓ_i = 0   (e_i = 0)
   (NLL-2)  범주 칸 점수       ℓ_i = −log p_{i, c_i}   (c_i ≥ 0);     ℓ_i = 0   (c_i = −1)
   (NLL-3)  킬 골드 구간 점수  (NLL-2)에서 C = 31,  c_i = b_i = min(⌊g_i / 100⌋, 30)
   (NLL-4)  가중 평균          θ̂ = Σ_i w_i · v_i / Σ_i w_i
   (NLL-5)  경기 재표본 복제   θ*_j = Σ_u d_ju · ṽ_u / Σ_u d_ju · w̃_u,   d_j ~ Multinomial(M; 1/M, …, 1/M)   (j = 1, …, B)

E. 하네스 제약 및 공통 가정:
   - 경기 가중 1/n_u: 한 경기의 흐름을 공유하는 행들에 1/n_u의 가중치를 부여하여 경기가 단일 관측 단위가 되도록 정규화 (결정 P7, P9).
   - math.fsum을 사용한 완전 정밀 합산으로 행의 순서에 결과가 흔들리지 않음.
   - 군집 부트스트랩: 경기를 통째로 복원 추출(Multinomial)하여 동일 시드 하에서 서로 다른 모형 간 쌍체 비교 가능 보장.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P4 (the same probability family and the same bins for every model), P10
#      (exposure, cut at the game end) and Task 5; math-control note 03 sections C (the NLL with exposure, log y! included, exposure 0 adds
#      nothing and a count there is refused), D (the gold bins) and F (match weights 1/n_u).
# Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press;
#      doi:10.1017/CBO9781139013567 (the Poisson log-likelihood with an exposure, that is a log-exposure offset, in the mean).
# Ref: Efron (1979), "Bootstrap methods: another look at the jackknife", Annals of Statistics 7(1): 1-26; doi:10.1214/aos/1176344552,
#      and Cameron, Gelbach, Miller (2008), "Bootstrap-based improvements for inference with clustered errors", Review of Economics
#      and Statistics 90(3): 414-427; doi:10.1162/rest.90.3.414 (resample whole clusters, here matches); the draw is the one of
#      ``flowprice.bootstrap``.
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

BOOT_B = 2000                    # D7 and the plan: matches are resampled 2,000 times ... (``flowprice.BOOT_B``; a test pins the equality)
BOOT_SEED = 20261021             # ... from this seed (``flowprice.BOOT_SEED``)
N_GOLD_BINS = 31                 # ``value.N_GOLD_BINS``: 0-3,000 in steps of 100, then the open bin
_ROW_SUM_TOLERANCE = 1e-4        # a row of probabilities adds up to 1 within this (float32 outputs of a booster add up to 1 within 1e-6)


def _bad(mask) -> int:
    return int(np.count_nonzero(mask))


# =============================================================================
# 블록 1: 전역 상수 및 헬퍼
# =============================================================================
# (BOOT_B, BOOT_SEED, N_GOLD_BINS, _ROW_SUM_TOLERANCE, _bad)

# =============================================================================
# 블록 2: 노출(Exposure)을 고려한 포아송 음의 로그 우도 (NLL-1)
# =============================================================================
def poisson(y, rate, exposure) -> np.ndarray:
    """The NLL of counts ``y`` under a Poisson law with mean mu = ``rate`` x ``exposure`` (rate per second, exposure in seconds, or any pair
    of units that multiply to the expected count), one value per cell, float64 and the broadcast shape of the three arguments.

    NLL = mu - y log mu + log y! for a cell with exposure > 0 (0 log 0 is 0; a count with mu = 0 has the infinite loss), and 0 for a cell
    with exposure 0, whose rate is not read. ValueError for a count that is not a non-negative whole number, an exposure that is negative
    or not finite, a rate of an observed cell (exposure > 0) that is negative or not finite, and a count in a cell with exposure 0.

    수식
    ----
    (NLL-1)  ℓ_i = μ_i − y_i · log μ_i + log(y_i!),   μ_i = λ_i · e_i          (e_i > 0)
             ℓ_i = 0                                                          (e_i = 0; y_i = 0이어야 한다)
             규약: y_i = 0이면 y_i · log μ_i = 0 (0 · log 0 = 0). y_i > 0이고 μ_i = 0이면 ℓ_i = +∞. log(y_i!) = log Γ(y_i + 1).
             유도: p(y; μ) = exp(−μ) · μ^y / y! 이면 −log p(y_i; μ_i) = μ_i − y_i · log μ_i + log(y_i!)  (e_i > 0).
    (NLL-1) = 설계서 8절 M2 주 지표(행당 포아송 NLL)의 칸 항이다. 설계서 3.1절의 로그 우도 Σ[n log λ − λΔ](n은 여기의 y, Δ는 e)의 칸
    항에 −1을 곱하고 모형과 무관한 항 −y · log e + log(y!)를 더한 것과 같다 [도출].

      기호 | 뜻                        | 단위                     | 범위                     | 코드 변수
      ℓ_i  | i번째 칸의 음의 로그 우도 | nat                      | ≥ 0 (확률 0이면 +∞)      | 반환값의 i번째 원소
      y_i  | i번째 칸에서 센 개수      | 개                       | 0, 1, 2, …               | y
      λ_i  | 예측 발생률               | 1/s (M3에서는 개: e = 1) | 유한, ≥ 0 (e_i > 0인 칸) | rate (e_i = 0인 칸에서는 읽지 않음)
      e_i  | 노출: 그 칸을 관측한 시간 | s (M3에서는 무차원 1)    | 유한, ≥ 0                | exposure
      μ_i  | 기대 개수 λ_i · e_i       | 개                       | ≥ 0                      | mu

    해석
    ----
    식 (NLL-1)은 한 칸에서 센 개수 y_i를 평균 μ_i = λ_i · e_i의 포아송 분포로 볼 때 그 개수가 나올 확률의 음의 로그이고, 기대 개수는
    노출에 비례한다. 기대 NLL을 μ_i로 미분하면 1 − E[y_i | 입력]/μ_i이므로, 기대 NLL은 μ_i가 참된 조건부 기댓값일 때 가장 작고 이는
    개수의 참 분포가 포아송이 아니어도 성립한다 [도출]. 참 분포가 포아송이면 로그 점수가 엄격한 적정 점수이므로 이 최소점에서 예측
    분포가 참 분포와 같다. 모형과 무관한 log(y_i!)까지 넣었으므로 이 값은 포아송 확률질량의 음의 로그 그 자체이고, 같은 개수에 다른
    이산 분포(예: 과산포일 때의 음이항, 계획 P8)를 쓴 모형의 NLL과 그대로 비교된다. 관측하지 않은 칸(경기가 먼저 끝났거나 전환 창이
    비어 e_i = 0인 칸)은 ℓ_i = 0이고 발생률을 읽지 않으며(계획 P10, 검토 초점 3), 관측한 칸에서 μ_i = 0인데 y_i > 0이면 참값
    ℓ_i = +∞를 그대로 돌려준다.

    근거
    ----
    계획 P4(모든 모형에 같은 확률 족), P10(노출은 경기 끝(발생은 T − 30 s, 전환은 T)에서 자른 관측 시간), 검토 초점 3(노출 0인 칸은
    NLL에 0), 발생(M2) 개수의 포아송 가정은 계획 P8(과산포 검정), 설계서 3.1절(계수 과정의 우도)과 8절(M2 주 지표), 검산 노트
    math-control-03 C절(손 계산; 시험이 같은 수를 확인한다).
    Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press;
         doi:10.1017/CBO9781139013567 (노출이 있는 포아송 로그 우도, 로그 노출 오프셋). Gneiting, T. and Raftery, A. E. (2007),
         "Strictly proper scoring rules, prediction, and estimation", JASA 102(477): 359-378; doi:10.1198/016214506000001437 (로그
         점수는 적정 점수).
    """
    y, rate, exposure = np.broadcast_arrays(np.asarray(y, dtype=np.float64), np.asarray(rate, dtype=np.float64),
                                            np.asarray(exposure, dtype=np.float64))
    # 식 (NLL-1)의 정의역을 점수 앞에서 확인한다. 노출은 개수가 0인 칸에도 검사하여 음수나 NaN 노출이 말없이 0점이 되지 않게 한다.
    bad = ~np.isfinite(y) | (y < 0) | (y != np.floor(y))
    if bad.any():
        raise ValueError(f'poisson: {_bad(bad)} count(s) are not non-negative whole numbers')
    bad = ~np.isfinite(exposure) | (exposure < 0)
    if bad.any():
        raise ValueError(f'poisson: {_bad(bad)} exposure(s) are negative or not finite')
    seen = exposure > 0
    bad = seen & ~(np.isfinite(rate) & (rate >= 0))
    if bad.any():
        raise ValueError(f'poisson: {_bad(bad)} rate(s) of observed cells are negative or not finite')
    bad = ~seen & (y > 0)
    if bad.any():
        raise ValueError(f'poisson: {_bad(bad)} cell(s) have a count but no exposure')
    # 식 (NLL-1)의 μ = λ·e. 관측하지 않은 칸은 발생률 자리에 0을 넣으므로 그 칸의 μ는 0이고 NaN 발생률이 결과에 섞이지 않는다.
    mu = np.where(seen, rate, 0.0) * exposure                              # the rate of an unobserved cell is never used
    # 식 (NLL-1)의 y·log μ 항. y = 0이면 0·log 0 = 0, y > 0이고 μ = 0이면 −∞여서 아래에서 빼면 NLL이 +∞가 된다. y = 0이고 μ = 0인
    # 칸에서는 0 × (−∞)가 NaN으로 계산되지만 np.where가 버리며, 경고는 이 의도된 경우들을 위해 끈다.
    with np.errstate(divide='ignore', invalid='ignore'):
        weighted = np.where(y > 0, y * np.log(mu), 0.0)                    # y log mu, with 0 log 0 = 0 and y log 0 = -inf
    # 식 (NLL-1)의 합 μ − y·log μ + log(y!). log(y!)는 log Γ(y + 1)로 계산한다: 171!부터 float64의 범위를 넘기 때문이다.
    return np.where(seen, mu - weighted + gammaln(y + 1.0), 0.0)


# =============================================================================
# 블록 3: 범주형 및 31구간 킬 골드 음의 로그 우도 (NLL-2, NLL-3)
# =============================================================================
def _category_scores(what: str, idx, prob, width: int | None) -> np.ndarray:
    """[내부 유틸리티]: categorical 및 gold의 공통 로그 우도 계산기.

    [역할]:
    - 식 (NLL-2) ℓ_i = -log prob[i, idx[i]]를 계산한다 (c_i >= 0).
    - 결측 범주(idx[i] == -1; 예: 미관측 구역, 0킬 팀의 골드)에 대해서는 우도 기여 없이 0.0을 부여한다.
    - 확률표의 유효성(모든 원소 유한/비음수, 각 행의 합이 1.0)을 정밀 검증한다.
    - gold 호출 시 width=31로 고정하여 31개 골드 구간 규격을 강제한다 (식 NLL-3).
    """
    idx = np.asarray(idx)
    if idx.size == 0 and idx.ndim <= 1:
        idx = idx.astype(np.int64)
    if idx.dtype.kind not in 'iu':
        raise TypeError(f'{what}: the category numbers must be integers, not {idx.dtype}')
    prob = np.asarray(prob, dtype=np.float64)
    if idx.ndim != 1 or prob.ndim != 2:
        raise ValueError(f'{what}: idx must be 1-d and prob a table of shape (n entries, n categories), not {idx.shape} and {prob.shape}')
    if prob.shape[0] != idx.shape[0]:
        raise ValueError(f'{what}: idx has {idx.shape[0]} entries but prob has {prob.shape[0]} rows')
    k = prob.shape[1]
    if width is not None and k != width:
        raise ValueError(f'{what}: prob must have {width} columns, not {k}')
    bad = (idx < -1) | (idx >= k)
    if bad.any():
        raise ValueError(f'{what}: {_bad(bad)} category number(s) lie outside -1 .. {k - 1}')
    # 확률표의 검사는 범주가 있는 항목(c_i ≥ 0)의 행에만 한다: (NLL-2)가 c_i = −1인 행을 읽지 않으므로 그 행의 NaN은 거부하지 않는다.
    used = idx >= 0
    rows = prob[used]
    # 확률분포의 조건: p ≥ 0, 유한, Σ_c p = 1.
    bad = (~np.isfinite(rows) | (rows < 0)).any(axis=1)
    if bad.any():
        raise ValueError(f'{what}: {_bad(bad)} probability row(s) hold a probability that is negative, NaN or infinite')
    bad = np.abs(rows.sum(axis=1) - 1.0) > _ROW_SUM_TOLERANCE
    if bad.any():
        raise ValueError(f'{what}: {_bad(bad)} probability row(s) do not add up to 1 (to within {_ROW_SUM_TOLERANCE:g})')
    # 식 (NLL-2): 범주가 있으면 −log p_{i,c_i}, 없으면 0. p = 0이면 +∞가 참값이므로 0으로 나누는 경고만 끈다.
    out = np.zeros(len(idx))
    if used.any():
        with np.errstate(divide='ignore'):
            out[used] = -np.log(prob[used, idx[used]])
    return out + 0.0                                                       # -0.0 (the log of a certain outcome) becomes 0.0


def categorical(idx, prob) -> np.ndarray:
    """(n,) float64: -log prob[i, idx[i]], the NLL of the category ``idx[i]`` (0 .. C-1) under the row ``prob[i]`` (C probabilities that add
    up to 1). ``idx[i]`` = -1 (no category: a missing zone) scores 0 and its row is not read; the caller counts those entries. A probability
    of 0 for the category that happened scores +inf. ValueError for a non-integer (TypeError), a category outside -1 .. C-1, a table whose
    row count differs from the entries, a probability that is negative, NaN or infinite, and a row that does not add up to 1.

    수식
    ----
    (NLL-2)  ℓ_i = −log p_{i,c_i}          (c_i ∈ {0, 1, …, C−1})
             ℓ_i = 0                        (c_i = −1)
             p_{i,c_i} = 0이면 ℓ_i = +∞. 읽는 행은 확률분포(p_{i,c} ≥ 0, Σ_c p_{i,c} = 1, 허용오차 1e-4)여야 한다.

      기호    | 뜻                              | 단위 | 범위                                                          | 코드 변수
      ℓ_i     | i번째 항목의 음의 로그 확률     | nat  | ≥ 0 (허용오차 때문에 −10⁻⁴까지 작아질 수 있다); 확률 0이면 +∞ | 반환값의 i번째 원소
      c_i     | 실제 범주 번호 (−1은 범주 없음) | —    | −1, 0, …, C−1                                                 | idx[i]
      p_{i,c} | 항목 i에 준 범주 c의 확률       | —    | [0, 1]                                                        | prob[i, c]
      C       | 범주의 수                       | 개   | ≥ 1                                                           | prob.shape[1]

    해석
    ----
    식 (NLL-2)는 실제로 일어난 범주 c_i에 모형이 준 확률의 음의 로그, 곧 로그 손실이며, 읽는 행이 확률분포가 아니면 이 값이 우도가
    아니므로 점수 대신 거부한다. 로그 점수는 엄격한 적정 점수이므로 기댓값은 모형이 참된 조건부 확률을 낼 때 가장 작다. 같은 항목을
    예측하는 서로 다른 모형의 값은 그대로 비교할 수 있고, 단위가 (NLL-1)과 같은 nat이므로 한 행 안의 칸 점수끼리 더해 행의 점수를 만들
    수 있다. 범주를 알 수 없는 항목(c_i = −1; 예컨대 첫 킬의 위치가 기록되지 않아 구역을 모를 때)은 사건은 있었으나 범주가 관측되지
    않은 경우이므로 우도에서 빼고 ℓ_i = 0으로 두며, 그 행의 확률은 읽지 않는다. 이렇게 뺀 평균은 기록이 빠지는 일이 구역과 무관할
    때에만 전체 항목의 평균을 나타내므로, 그런 항목의 수는 호출한 쪽이 따로 세어 보고한다.

    근거
    ----
    계획 P4(모든 모형에 같은 확률 족과 구간)와 검토 초점 4(구역 −1은 ρ 학습에서 빼고 센다), 판정 R13(범주 출력에 균등 분포를 섞어 확률
    0이 없다), 설계서 3.1절(위치 표지 ρ의 Σ n log ρ 항)과 8절(M2 쌍 표지 log loss, M3 규모), 검산 노트 math-control-03 D절(시험이 같은
    수를 확인한다).
    Ref: Gneiting, T. and Raftery, A. E. (2007), "Strictly proper scoring rules, prediction, and estimation", JASA 102(477): 359-378;
         doi:10.1198/016214506000001437 (로그 점수는 적정 점수).
    """
    return _category_scores('categorical', idx, prob, None)


def gold(idx, prob) -> np.ndarray:
    """Like ``categorical`` over the 31 gold bins (``value.gold_bin``): ``prob`` has 31 columns; ``idx[i]`` = -1 (the team had no kill: no
    gold category) scores 0 and its row is not read.

    수식
    ----
    (NLL-3)  ℓ_i = −log p_{i,b_i}          (b_i ∈ {0, 1, …, 30})
             ℓ_i = 0                        (b_i = −1)
             b_i = min(⌊g_i / 100⌋, 30). 구간 나누기는 ``value.gold_bin``이 하고 이 함수는 구간 번호 b_i를 받는다.
             즉 (NLL-2)에서 C = 31, c_i = b_i이다.

      기호    | 뜻                           | 단위 | 범위                                                          | 코드 변수
      ℓ_i     | i번째 항목의 음의 로그 확률  | nat  | ≥ 0 (허용오차 때문에 −10⁻⁴까지 작아질 수 있다); 확률 0이면 +∞ | 반환값의 i번째 원소
      g_i     | 그 팀의 킬 골드              | 골드 | ≥ 0                                                           | — (입력이 아님)
      b_i     | 구간 번호 (−1은 킬이 0인 팀) | —    | −1, 0, …, 30                                                  | idx[i]
      p_{i,b} | 항목 i에 준 구간 b의 확률    | —    | [0, 1]                                                        | prob[i, b]

    해석
    ----
    식 (NLL-3)은 범주 점수 (NLL-2)를 팀의 킬 골드 구간에 쓴 것이며, 킬 골드를 0–3,000은 100골드씩 30구간, 3,000 이상은 한 구간으로
    나눈 31구간을 모든 모형이 같이 쓴다(계획 P4). 점수는 구간의 확률에 대한 것이어서 구간 안의 위치는 평가하지 않으며, 모든 모형이
    같은 31구간을 쓸 때에만 NLL을 맞댈 수 있으므로 열이 31개가 아닌 표는 거부한다. 킬이 0인 팀의 킬 골드는 0으로 정해진다(계획 P4).
    그래서 골드 분포는 킬 수 층(1–5, 6 이상)에 조건부로만 두고, 킬이 0인 팀은 b_i = −1로 표시하여 ℓ_i = 0을 주며 그 확률 행은 읽지
    않는다. 킬이 없었다는 정보는 킬 수의 포아송 점수 (NLL-1)에 이미 들어 있다.

    근거
    ----
    계획 P4(킬 수 층 조건부 31구간 범주), 설계서 8절(M3 K NLL, 골드), 검산 노트 math-control-03 D절(구간 나누기; 시험이 같은 수를
    확인한다).
    """
    return _category_scores('gold', idx, prob, N_GOLD_BINS)


# =============================================================================
# 블록 4: 경기 가중 평균 및 부트스트랩 신뢰구간 (NLL-4, NLL-5)
# =============================================================================
def _scored_rows(what: str, values, w) -> tuple:
    """[내부 유틸리티]: weighted_mean 및 match_bootstrap의 공통 입력 검증기.

    [역할]:
    - 점수 벡터 values와 가중치 벡터 w가 동일한 길이의 1차원 float64 배열인지 확인한다.
    - 점수에 NaN 또는 무한대(+inf/-inf)가 없는지 검증한다 (비유한 점수의 조용한 은폐 차단).
    - 가중치가 음수가 아니고 모두 유한한지 검증한다.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if v.ndim != 1 or w.shape != v.shape:
        raise ValueError(f'{what}: values and weights must be 1-d arrays of one length, not {v.shape} and {w.shape}')
    if v.size == 0:
        raise ValueError(f'{what}: there are no rows')
    if not np.isfinite(v).all():
        raise ValueError(f'{what}: {_bad(~np.isfinite(v))} score(s) are NaN or infinite')
    if not (np.isfinite(w).all() and (w >= 0).all()):
        raise ValueError(f'{what}: {_bad(~(np.isfinite(w) & (w >= 0)))} weight(s) are negative, NaN or infinite')
    return v, w


def weighted_mean(values, w) -> float:
    """sum(w v) / sum(w) as a Python float, summed exactly (``math.fsum``), so that the result does not depend on the order of the rows.
    ValueError for no rows, a score that is NaN or infinite, a weight that is negative, NaN or infinite, and weights that add up to 0.

    수식
    ----
    (NLL-4)  θ̂ = Σ_i w_i · v_i / Σ_i w_i
             경기 가중 w_i = 1 / n_u (i∈u)이면 경기 u의 가중치 합 Σ_{i∈u} w_i = 1이므로 Σ_i w_i = M이고
             θ̂ = (1 / M) · Σ_u v̄_u,   v̄_u = (1 / n_u) · Σ_{i∈u} v_i   [도출: 경기 평균의 평균]

      기호 | 뜻                                            | 단위      | 범위              | 코드 변수
      θ̂    | 가중 평균                                     | 점수 단위 | 유한              | 반환값
      v_i  | i번째 행의 점수 (행 NLL, Brier 점수 등)       | 점수 단위 | 유한              | values
      w_i  | i번째 행의 가중치                             | —         | 유한, ≥ 0, 합 > 0 | w
      n_u  | 경기 u의 행 수 (경기 번호는 호출한 쪽이 정함) | 행        | ≥ 1               | —
      M    | 경기의 수                                     | 경기      | ≥ 1               | —
      v̄_u  | 경기 u의 행 점수 평균                         | 점수 단위 | 유한              | —

    해석
    ----
    식 (NLL-4)는 행의 점수를 가중치로 평균한 값이다. 가중치를 경기 u의 행마다 1/n_u로 주면 경기마다 가중치의 합이 1이 되어 θ̂는 경기별
    평균 v̄_u의 단순 평균과 같아지므로, 행이 많은 경기가 평균을 끌지 않고 경기가 하나의 관측 단위가 된다 [도출]. 이 식은 가중치에 같은
    양수를 곱해도 변하지 않고, 가중치가 0인 행은 평균에 아무 영향도 주지 않는다. 행이 없거나 가중치의 합이 0이면 평균이 정의되지 않고,
    비유한 점수는 평균 전체를 말없이 NaN이나 ∞로 만들기 때문에 거부한다.

    근거
    ----
    설계서 8절 비교 공통 규칙(경기 가중 1/n_u), 계획 P7(관문 공통 조건: 경기 가중 1/n_u)과 P9(학습은 행마다 같은 무게, 평가는 1/n_u).
    가중치 1/n_u는 ``features.match_weights``가 만든다. 검산 노트 math-control-03 F절(시험이 같은 수를 확인한다).
    """
    v, w = _scored_rows('weighted_mean', values, w)
    # 식 (NLL-4)의 분모 Σ w_i. 합이 0이면 평균이 없다.
    total = math.fsum(w)
    if total <= 0:
        raise ValueError('weighted_mean: the weights add up to 0')
    # 식 (NLL-4)의 분자 Σ w_i·v_i. 곱 w_i·v_i는 한 번 반올림되고 그 뒤의 합이 정확하다.
    return math.fsum(w * v) / total


def _whole(what: str, x, minimum: int) -> int:
    """[내부 유틸리티]: 정수 매개변수 타입 및 최솟값 검증기 (bool 제외, int 호환 확인)."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise TypeError(f'match_bootstrap: {what} must be an integer, not {type(x).__name__}')
    if x < minimum:
        raise ValueError(f'match_bootstrap: {what} must be at least {minimum}, not {x}')
    return int(x)


def match_bootstrap(values, match, w, B=BOOT_B, seed=BOOT_SEED) -> np.ndarray:
    """(B,) float64: ``B`` replicates of ``weighted_mean(values, w)`` that resample whole matches (see the module text). ``match`` gives the
    match number of every row (any integers; the matches are taken in increasing order). The first replicates do not depend on ``B``.
    ValueError / TypeError as ``weighted_mean`` for the rows, except that weights adding up to 0 are not refused: then every replicate
    is NaN, as is any replicate whose drawn matches have no weight. Also for match numbers that are not integers, and for a B below 1 or
    a negative or non-integer seed.

    수식
    ----
    (NLL-5)  ṽ_u = Σ_{i∈u} w_i · v_i,     w̃_u = Σ_{i∈u} w_i                경기 u의 가중 합 (원래 표본에서 한 번 정한다)
             d_j = (d_j1, …, d_jM) ~ Multinomial(M; 1/M, …, 1/M)         복제 j의 경기별 횟수 (Σ_u d_ju = M)
             θ*_j = Σ_u d_ju · ṽ_u / Σ_u d_ju · w̃_u,     j = 1, …, B
             d_j는 하나의 난수열(씨앗은 코드의 seed)에서 복제마다 한 번씩 차례로 뽑으므로 복제 j는 B에 의존하지 않는다.
             모든 d_ju = 1이면 θ*_j = Σ_u ṽ_u / Σ_u w̃_u = θ̂이며 (NLL-4)와 같다.
             (NLL-4)처럼 가중치가 1/n_u이면 w̃_u = 1, ṽ_u = v̄_u(경기 평균)이므로 θ*_j = (1 / M) · Σ_u d_ju · v̄_u   [도출: 경기 평균들의 평균]

      기호     | 뜻                              | 단위         | 범위                            | 코드 변수
      θ*_j     | j번째 복제 (재표본의 가중 평균) | 점수 단위    | 유한 또는 NaN (가중치 0만 뽑힘) | out[j - 1] (코드의 b = j - 1)
      v_i, w_i | i번째 행의 점수와 가중치        | 점수 단위, — | 유한; 유한, ≥ 0                 | values, w
      u        | 크기 순서로 다시 매긴 경기 번호 | —            | 1, …, M                         | inverse (0부터)
      M        | 서로 다른 경기의 수             | 경기         | ≥ 1                             | M
      ṽ_u      | 경기 u의 가중 점수 합           | 점수 단위    | 유한                            | num
      w̃_u      | 경기 u의 가중치 합              | —            | ≥ 0                             | den
      d_ju     | 복제 j에서 경기 u가 뽑힌 횟수   | 회           | 0, 1, …, M                      | counts
      B        | 복제의 수                       | —            | ≥ 1                             | B

    해석
    ----
    식 (NLL-5)는 M개 경기에서 경기 하나를 균등하게 M번 복원 추출하여 다시 계산한 (NLL-4)의 복제 값이며, 경기 u가 뽑힌 횟수의 벡터
    d_j는 다항 분포를 따른다. 뽑힌 경기는 원래 표본에서 정한 가중 점수 합 ṽ_u와 가중치 합 w̃_u를 그대로 가져오므로 한 경기 안 행들의
    상관이 보존된다(군집 부트스트랩). 횟수는 경기 수 M, 씨앗, 복제 번호에만 의존하고 점수에는 의존하지 않으므로 같은 경기들에서 채점한
    두 모형은 같은 재표본을 받아 복제를 짝지어 차이의 구간을 만들 수 있다. 뽑힌 경기가 모두 가중치 0이면 분모가 0이어서 그 복제의
    평균은 정의되지 않으며 NaN으로 돌려준다. 가중치가 모두 0이면 모든 복제가 NaN이며, (NLL-4)와 달리 오류로 거부하지 않는다.

    근거
    ----
    설계서 8절 비교 공통 규칙(경기 부트스트랩, 가중 1/n_u, 95% 구간), 계획 P7(관문 공통 조건), 계획 전역 제약 D7 항(B와 시드; 추정기는
    설계서 8.1절), 판정 R26(같은 씨앗이 모든 S3 통계에서 같은 경기를 뽑는다). 구간은 호출한 쪽이 복제의 백분위 구간으로
    만든다(gate_s3.py).
    Ref: Efron (1979), "Bootstrap methods: another look at the jackknife", Annals of Statistics 7(1): 1-26;
         doi:10.1214/aos/1176344552, and Cameron, Gelbach, Miller (2008), "Bootstrap-based improvements for inference with clustered
         errors", Review of Economics and Statistics 90(3): 414-427; doi:10.1162/rest.90.3.414 (군집을 통째로 다시 뽑는다).
    """
    B = _whole('B', B, 1)
    seed = _whole('seed', seed, 0)
    v, w = _scored_rows('match_bootstrap', values, w)
    match = np.asarray(match)
    if match.shape != v.shape:
        raise ValueError(f'match_bootstrap: match has shape {match.shape} but there are {v.shape[0]} rows')
    if match.dtype.kind not in 'iu':
        raise TypeError(f'match_bootstrap: the match numbers must be integers, not {match.dtype}')
    # 식 (NLL-5)의 경기별 합 ṽ_u, w̃_u.
    # 경기 번호는 작은 수부터 0, 1, …로 다시 매겨지므로 복제는 번호의 값이 아니라 순서에만 의존한다(시험).
    _, inverse = np.unique(match, return_inverse=True)
    inverse = inverse.reshape(-1)
    M = int(inverse.max()) + 1
    num = np.bincount(inverse, weights=w * v, minlength=M)                  # ṽ_u
    den = np.bincount(inverse, weights=w, minlength=M)                      # w̃_u
    # 식 (NLL-5)의 횟수 d_j.
    rng = np.random.default_rng(seed)
    pvals = np.full(M, 1.0 / M)
    out = np.empty(B)
    # 식 (NLL-5)의 비율. 경고는 0/0(가중치 0만 뽑힌 복제)을 위해 끈다.
    with np.errstate(divide='ignore', invalid='ignore'):
        for b in range(B):                                                 # one replicate at a time, sums without BLAS: replicate j is
            counts = rng.multinomial(M, pvals)                             # the same for every B and on every machine's BLAS
            out[b] = np.sum(counts * num) / np.sum(counts * den)
    return out
