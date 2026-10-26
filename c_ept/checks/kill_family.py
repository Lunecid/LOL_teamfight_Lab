r"""C5 joint kill-family measurement tool (S4b preparation M16; author decisions D5 and D10 P8).

What it measures. Decision D5 moved the kill count of the team that took the first kill to K_d = 1 + N_d and left two questions to a measurement (the C5 measurement of the gate G-S3c):
which family N_d follows (Poisson or negative binomial) and whether the two teams' counts depend on each other. Decision D10 P8 says that the measurement is made on B1 heads and at the
level of the labels, without extra EPT fits. This module is that tool. The measurement run itself (on the real labels) happens only in S4b, with the author's approval; this task builds the
tool and tests it on synthetic counts.

Input: FIT labels only. The caller's duty is to pass the units of the role FIT (and for ``compare_b1`` the units of FIT for the fit, STOP for the early stopping and one more role for the
score, chosen by the plan of the run); the functions of this module read no store, no role and no match, so they cannot check it. Nothing is read from disk and nothing is written.

``compare(K, mark, size, w, joint_given='role') -> dict``: the label level. Units are sorted into cells (size class s of ``units.SIZE_NAMES`` x mark kind: champion mark 0 .. 49 or execution
mark 50, 51). In every cell four laws are fitted by weighted maximum likelihood and reported (NLL per unit weighted by w, the predicted share of units with no kill at all, the predicted
share of ties K_blue = K_red, the parameters and their number): ``poisson`` (independent Poisson: the law of every earlier run), ``shifted`` (D5: the first-kill team's count is 1 + N_d,
N_d Poisson; the other team Poisson; execution marks plain), ``shifted_nb`` (as ``shifted`` but N_d negative binomial: overdispersion on N_d only) and ``joint`` (the empirical table of the
pair of counts, zero outside the support). The counts are read in the order (first team, other team) for a champion mark and (blue, red) for an execution; the rates are per role, not per color
(an assumption of color symmetry that the heads of D7 have too).

``compare_b1(fit, stop, evaluate, ...) -> dict``: the B1 level. The kill head of the B1 path (``baselines_b1.M3Kills``) is fitted twice on the same features, once with ``kill_law='poisson'``
and once with ``'shifted'``, and both are scored on the same evaluation units. On the shifted head's means it also measures, without a further fit, the conditional negative binomial
dispersion of N_d and the correlation of the two teams' Pearson residuals (``conditional``).

Which level decides (author decision D29, 2026-10-26). The heads model the counts given their features, so the D5 questions (the family of N_d and the dependence of the two teams' counts)
are decided on the conditional statistics of ``compare_b1``. The label-level statistics of ``compare`` are descriptive: within a cell the units differ in intensity, so a Poisson mixture looks
overdispersed and a shared intensity looks like dependence even when the counts are conditionally Poisson and independent given the features. The rule and its thresholds are not in this
module; they are fixed in the S4b run plan before the real labels are opened.

Reading the label-level numbers. The table law always has the best NLL in sample (it is the maximum-likelihood distribution over all pair distributions of the cell), so its NLL says nothing
about dependence. The dependence alone is the mutual information, the NLL of the independent table (the product of the cell's own empirical margins) minus the NLL of the joint table,
with ``G = 2 n_eff MI``, its degrees of freedom and chi-square p. ``aic`` per law and the boundary likelihood-ratio test of the negative binomial against the shifted Poisson
(``family_test``) compare the parametric families in sample. No held-out score is given for the table: under the rule "zero mass outside the support" a pair of a held-out role that FIT has
not seen scores +infinity, so a held-out comparison would need a smoothing that changes the law; the three parametric laws have at most 3 parameters, so their in-sample optimism is O(1/n).
The colour-symmetry assumption (rates per role, not per colour) is checked by calling ``compare`` once on the units of marks 0 .. 24 and once on those of marks 25 .. 49: the orientation
makes the two results directly comparable.

수식 (식 번호는 ``(KF-n)``; 기호 표는 아래)

    (KF-1)  방향 정하기: d = m // 25 (m = 0 … 49), 처형 표지(m = 50, 51)는 d = −1.   (k_1, k_2) = (K_d, K_{1−d}) (d = 1이면 열을 바꾼다), 처형이면 (K_blue, K_red)
    (KF-2)  poisson:      K_j ~ Poisson(r_j),  r_j = Σ_i w_i k_{ij} / Σ_i w_i   (j = 1, 2)
    (KF-3)  shifted:      K_1 = 1 + N,  N ~ Poisson(λ),  K_2 ~ Poisson(μ);   λ = Σ_i w_i (k_{i1} − 1) / Σ_i w_i,  μ = Σ_i w_i k_{i2} / Σ_i w_i     (처형 칸은 (KF-2)와 같다)
    (KF-4)  shifted_nb:   N ~ NB2(mean λ, α):   P(N = n) = Γ(n + 1/α) / (Γ(1/α) n!) · (1/(1 + αλ))^{1/α} · (αλ/(1 + αλ))^n ,   Var N = λ + α λ² ,  α → 0은 포아송
                          λ̂ = Σ_i w_i n_i / Σ_i w_i (α와 무관한 최대우도 해),  α̂ = argmax_{α ≥ 0} Σ_i w_i log P(N = n_i)   (α = 0은 후보에 들어간다)
    (KF-5)  joint:        p̂_g(a, b) = Σ_i w_i 1[(k_{i1}, k_{i2}) = (a, b), g_i = g] / Σ_i w_i 1[g_i = g]   (g: 칸 하나(``joint_given='role'``) 또는 표지 하나(``'mark'``)); 자료에 없는 쌍은 0
    (KF-6)  칸 NLL:       NLL = Σ_i w_i ℓ_i / Σ_i w_i,   ℓ_i = −log P(k_{i1}, k_{i2}) (정의 (KF-2) … (KF-5)의 법칙; 두 팀은 독립이거나(앞의 셋) 표가 결합이다(넷째))
    (KF-7)  예측 몫:      무킬 = P(K_1 = 0) P(K_2 = 0) (결합 법칙은 p̂(0, 0)),  무승부 = Σ_k P(K_1 = k) P(K_2 = k) (결합 법칙은 Σ_k p̂(k, k));  관측 몫은 자료의 가중 비율
    (KF-8)  같은 단위 집합:  r_i = 1[d_i ≥ 0 그리고 k_{i1} = 0] 인 단위는 이동 법칙이 확률 0을 주므로(첫 킬 팀의 킬 수가 0) 네 법칙 모두에서 뺀다
    (KF-9)  독립 표:        q̂_g(a, b) = p̂_g,1(a) p̂_g,2(b) (칸 안의 두 주변 빈도의 곱);  MI = NLL(독립 표) − NLL(결합 표) ≥ 0;  G = 2 n_eff MI ~ χ²(df) (독립이면, 점근),  df = Σ_g (I_g − 1)(J_g − 1)
    (KF-10) 정보량 기준:    AIC = 2 n_eff NLL + 2 n_params   (n_eff = (Σ w)² / Σ w²: Kish의 유효 표본 크기)
    (KF-11) 경계 우도비:    LR = 2 n_eff (NLL_shifted − NLL_nb) ≥ 0,   α = 0이 모수 공간의 경계이므로 LR ~ ½ χ²(0) + ½ χ²(1),   p = ½ P(χ²(1) ≥ LR)  (LR = 0이면 p = 1)
    (KF-12) 조건부 통계:    (B1 수준) α̂_c = argmax_{α ≥ 0} Σ_i w_i log NB2(n_i; λ̂_i, α)  (평균 λ̂_i는 이동 머리의 예측으로 고정, 모수 하나);
                          ρ = 가중 상관(e_{i1}, e_{i2}),  e_{i1} = (k_{i1} − 1 − λ̂_i) / √λ̂_i,  e_{i2} = (k_{i2} − μ̂_i) / √μ̂_i  (피어슨 잔차)

      기호     | 뜻                                                          | 단위 | 범위            | 코드 변수
      m        | 첫 킬의 표지 (champion 0 … 49, 처형 50, 51)                    | —    | 0 … 51         | mark
      d        | 첫 킬 팀 (0 파랑, 1 빨강, 처형 −1)                              | —    | −1, 0, 1       | killlaw.first_team(mark)
      k_{ij}   | 단위 i에서 j번째 칸(첫 킬 팀, 다른 팀 순)의 킬 수                  | 개   | 0, 1, …        | Ko[:, j]
      w_i      | 단위의 가중치 (한 경기의 단위가 여럿이면 호출자가 정한다)           | —    | ≥ 0            | w
      r_j, λ, μ | 포아송의 율 (λ는 초과분 N의 평균, μ는 다른 팀의 평균)              | 개   | ≥ 0            | rate_first_team, rate_other_team
      α        | NB2의 과대분산 (Var N = λ + α λ²)                              | —    | ≥ 0            | alpha
      ℓ_i      | 단위의 음의 로그 가능도                                         | nat  | ≥ 0            | score
      g        | 결합 표의 조건 (칸 전체 또는 표지 하나)                          | —    | —              | joint_given
      MI, G    | 상호 정보량(독립 표 대비 결합 표의 이득)과 G 통계량                | nat, — | ≥ 0          | mutual_information, G
      n_eff    | Kish 유효 표본 크기 (가중치가 같으면 단위 수)                      | 개   | (0, n_kept]    | n_eff
      λ̂_i, μ̂_i | 이동 머리가 단위 i에 내는 N_d의 평균과 다른 팀 킬 수의 평균         | 개   | > 0            | shifted.predict(X)
      ρ        | 두 팀 피어슨 잔차의 가중 상관                                    | —    | [−1, 1]        | residual_correlation

해석. 이동 법칙은 첫 킬 팀의 킬 수가 0인 사건을 정확히 없앤다. 옛 포아송 법칙은 그 사건에 P(K_1 = 0) = e^{−r_1}만큼의 몫을 준다(r_1은 첫 킬 팀 킬 수의 평균, 곧 λ + 1). 자료에서 그 몫은 0이므로
챔피언 칸에서는 ``first_team_zero_share``가 잘못 놓인 질량을 직접 보여 준다. (KF-7)의 무킬 몫 e^{−(r_1 + r_2)}는 그 사건에 다른 팀의 무킬이 겹칠 몫이어서 그 질량을 줄여 보이고, 챔피언 칸의 관측·이동·NB·결합
무킬 몫은 구조적으로 0이므로(첫 킬이 곧 킬이다) 무킬 몫은 처형 칸에서 의미가 있다.
음이항은 N의 분산이 평균보다 클 때만 포아송을 이긴다. α가 0 근처이면 포아송으로 충분하다는 뜻이고, NB는 포아송을 포함하므로 표본 안에서는 NLL이 더 나빠지지 않는다. 결합 표는 모든 쌍 분포 가운데
표본 우도가 가장 큰 분포이므로 어떤 자료에서도 표본 안 NLL이 가장 좋다(Gibbs 부등식). 그 이득은 주변 분포의 불일치(포아송이 빈도와 다름)와 의존이 섞인 것이므로 의존만 가려내려면 독립 표와 비교한
상호 정보량을 본다(KF-9). 모든 NLL은 FIT 표본 안의 값이고 표본 밖 점수가 아니다. 칸 안의 단위는 강도가 서로 달라 라벨 수준의 α와 의존은 특징으로 설명되는 이질성을 포함한다(D29: 결정은 B1 수준의 조건부 통계로 한다).

근거. 저자 결정 D5 (N_d의 족과 결합 의존은 C5 측정에서 정한다), D10 P8 (C5는 B1 머리와 라벨 수준에서 잰다); 계획 S4b 준비 Task M16; M1 검토 m9 (이동 법칙 행은 K_d = 0인 단위를 거부하고 옛 법칙은 받으므로 두 법칙은 두 법칙이
모두 받는 단위에서 비교한다).
# Ref: Self, S. G. and Liang, K.-Y. (1987), Journal of the American Statistical Association 82(398): 605-610; doi:10.1080/01621459.1987.10478472 (경계의 우도비 검정: 혼합 ½χ²(0) + ½χ²(1)).
#      Akaike, H. (1974), IEEE Transactions on Automatic Control 19(6): 716-723; doi:10.1109/TAC.1974.1100705 (AIC). Kish, L. (1965), "Survey Sampling", Wiley (유효 표본 크기 (Σw)²/Σw²).
#      Cover, T. M. and Thomas, J. A. (2006), "Elements of Information Theory", 2nd ed., Wiley; doi:10.1002/047174882X (상호 정보량 = 결합 대 주변 곱의 KL; Gibbs 부등식).
#      Agresti, A. (2013), "Categorical Data Analysis", 3rd ed., Wiley (G 통계량 2 n MI와 df (I − 1)(J − 1)).
# Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press; doi:10.1017/CBO9781139013567 (NB2: 분산 μ + αμ², 계수 자료의 과대분산;
#      장 번호는 이 구현이 기대는 정보가 아니라 서지 정보). Hilbe, J. M. (2011), "Negative Binomial Regression", 2nd ed., Cambridge University Press; doi:10.1017/CBO9780511973420 (NB2 모수화: 평균 μ와
#      분산 μ + αμ², 역분산 r = 1/α).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2

from c_ept.domain import units
from c_ept.models import baselines_b1, killlaw
from c_ept.models import nll as nll_scores

LAWS = ('poisson', 'shifted', 'shifted_nb', 'joint')
KINDS = ('champion', 'execution')
JOINT_GIVEN = ('role', 'mark')
DROPPED_REASON = ('the first-kill team has no kill (the mark and the counts disagree): the shifted law gives the unit the probability 0, so the unit is left out of every law '
                  'to score all laws on one set of units')
ALPHA_GRID = np.log(np.logspace(-4, 2, 41))          # the NB2 dispersion is searched on this grid of log alpha (1e-4 .. 1e2), then refined between the neighbours of the best point
ALPHA_EDGE = 1e-8                                    # below this the NB2 pmf is the Poisson pmf (1/alpha would overflow the log-gamma terms)
TAIL_TARGET = 1e-6                                   # the support of the pmfs is widened until the mass beyond it is below this (the shares are sums over the support)
SUPPORT_MIN = 64                                     # start width of the support: min(SUPPORT_CAP, max(SUPPORT_MIN, SUPPORT_PER_COUNT max K + SUPPORT_SLACK)) counts 0 .. width − 1
SUPPORT_PER_COUNT = 4
SUPPORT_SLACK = 32
SUPPORT_CAP = 4096                                   # the widening stops here; a cell that still has more mass beyond reports its ``tail_mass`` and goes on (no abort)


# =============================================================================
# 블록 1: 입력 점검
# =============================================================================
def _integers(what: str, a) -> np.ndarray:
    a = np.asarray(a)
    if a.size == 0 and a.ndim <= 2:
        a = a.astype(np.int64)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{what} must be integers, not {a.dtype}')
    return a.astype(np.int64)


def _check_inputs(K, mark, size, w) -> tuple:
    """The checked ``(K (n, 2) int64, mark (n,) int64, size (n,) int64, w (n,) float64)``; ValueError for a shape, a range or a weight that cannot be used, TypeError for a non-integer count."""
    K = _integers('the kill counts K', K)
    if K.ndim != 2 or K.shape[1] != killlaw.N_TEAMS:
        raise ValueError(f'the kill counts K must have shape (n, {killlaw.N_TEAMS}), not {K.shape}')
    n = len(K)
    if K.size and K.min() < 0:
        raise ValueError('a kill count is negative')
    mark = _integers('the marks', mark)
    size = _integers('the size classes', size)
    if mark.shape != (n,) or size.shape != (n,):
        raise ValueError(f'mark and size must have shape ({n},), not {mark.shape} and {size.shape}')
    if mark.size and (mark.min() < 0 or mark.max() >= killlaw.N_MARKS):
        raise ValueError(f'a mark is 0 .. {killlaw.N_MARKS - 1}')
    if size.size and (size.min() < 0 or size.max() >= len(units.SIZE_NAMES)):
        raise ValueError(f'a size class is 0 .. {len(units.SIZE_NAMES) - 1}')
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f'the weights w must have shape ({n},), not {w.shape}')
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('a weight is negative or not finite')
    return K, mark, size, w


def refused_units(K, mark) -> np.ndarray:
    """(n,) bool: the units that the shifted law refuses, a champion mark whose first-kill team scored no kill (KF-8). The plain law accepts them.

    해석: 첫 킬이 그 팀의 킬이므로 자료에서는 일어날 수 없는 단위이다(표지와 ``units/K``가 어긋난 단위). 이동 법칙은 확률 0을 주고 옛 법칙은 받으므로, 두 법칙의 NLL을 같은 단위에서 비교하려고 이 단위를 센다."""
    K = _integers('the kill counts K', K)
    mark = _integers('the marks', mark)
    if K.ndim != 2 or K.shape[1] != killlaw.N_TEAMS or mark.shape != (len(K),):
        raise ValueError('K must have shape (n, 2) and mark shape (n,)')
    d = killlaw.first_team(mark)
    champion = d >= 0
    first = K[np.arange(len(K)), np.where(champion, d, 0)]
    return champion & (first < 1)


def _orient(K: np.ndarray, mark: np.ndarray) -> tuple:
    """(Ko (n, 2), d_o (n,)): the counts in the order (first team, other team), d_o = 0 for a champion mark and −1 for an execution (blue, red). (KF-1)

    해석: 방향을 맞추면 ``killlaw``의 이동 법칙이 열 0을 첫 킬 팀으로 읽는다(d_o = 0)."""
    d = killlaw.first_team(mark)
    Ko = np.where((d == 1)[:, None], K[:, ::-1], K)
    return Ko, np.where(d >= 0, 0, -1).astype(np.int64)


# =============================================================================
# 블록 2: 음이항 (NB2) 질량과 적합
# =============================================================================
def nb_logpmf(k, mean, alpha: float) -> np.ndarray:
    """log P(N = k) of the NB2 law with the mean ``mean`` and the dispersion ``alpha`` (Var N = mean + alpha mean²); ``alpha = 0`` is the Poisson law (``nll.poisson`` with the exposure 1,
    the same bits). ValueError for a negative count or alpha.

    (KF-4)의 로그 질량이다. r = 1/α이고 mean = 0이면 k = 0에 로그 질량 0, 그 밖에는 −∞이다.

    해석: 음이항은 평균이 Gamma(r, 평균 μ)로 흔들리는 포아송이다. α가 클수록 분산이 평균보다 크게 퍼지고 α → 0이면 포아송으로 수렴한다."""
    k = np.asarray(k, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if (k < 0).any() or (k != np.floor(k)).any():
        raise ValueError('a count is not a non-negative whole number')
    if not alpha >= 0 or not np.isfinite(alpha):
        raise ValueError(f'the dispersion alpha must be finite and not negative, not {alpha!r}')
    if alpha < ALPHA_EDGE:
        return -nll_scores.poisson(k, mean, 1.0)
    r = 1.0 / alpha
    with np.errstate(divide='ignore', invalid='ignore'):
        out = gammaln(k + r) - gammaln(r) - gammaln(k + 1.0) + r * np.log(r / (r + mean)) + k * np.log(mean / (r + mean))
    return np.where(mean == 0, np.where(k == 0, 0.0, -np.inf), out)


def _fit_alpha(n: np.ndarray, mean, w: np.ndarray) -> float:
    """The dispersion alpha ≥ 0 that maximises the weighted NB2 likelihood of the counts ``n`` with the means ``mean`` held fixed (a number, or one value per count); 0 when the Poisson law
    is at least as good. One parameter: a grid of log alpha, then a refinement between the neighbours of the best point, and alpha = 0 as a candidate of its own.

    (KF-4)와 (KF-12)의 α̂이다. 평균을 고정하고 α 하나만 맞추므로 라벨 수준(평균 = 표본 평균)과 B1 수준(평균 = 머리의 예측)이 같은 함수를 쓴다."""
    total = float(w.sum())

    def score(log_alpha: float) -> float:
        return float(-np.dot(w, nb_logpmf(n, mean, float(np.exp(log_alpha)))) / total)

    poisson = float(-np.dot(w, nb_logpmf(n, mean, 0.0)) / total)
    values = np.array([score(a) for a in ALPHA_GRID])
    best = int(np.argmin(values))
    lo, hi = ALPHA_GRID[max(best - 1, 0)], ALPHA_GRID[min(best + 1, len(ALPHA_GRID) - 1)]
    fit = minimize_scalar(score, bounds=(float(lo), float(hi)), method='bounded', options={'xatol': 1e-9})
    candidates = [(float(fit.fun), float(np.exp(fit.x))), (float(values[best]), float(np.exp(ALPHA_GRID[best])))]
    found, alpha = min(candidates)
    return alpha if found < poisson else 0.0


def _fit_nb(n: np.ndarray, w: np.ndarray) -> tuple:
    """``(mean, alpha)``: the weighted mean of the excess counts ``n`` and the dispersion that maximises the weighted NB2 likelihood, 0 when the Poisson law is at least as good.

    (KF-4)의 λ̂는 α와 무관하게 표본 평균이다(NB2의 점수 방정식이 Σ w (n − λ) / (λ (1 + αλ)) = 0). α는 로그 격자에서 가장 좋은 점을 찾고 이웃 사이에서 다듬으며, α = 0은 따로 겨뤄 포아송이 같거나 낫다면 0을 낸다."""
    mean = float(np.dot(w, n) / float(w.sum()))
    if mean == 0.0:
        return mean, 0.0
    return mean, _fit_alpha(n, mean, w)


def _kish(w: np.ndarray) -> float:
    """Kish's effective sample size (Σ w)² / Σ w² of positive weights (the number of units when the weights are equal). (KF-10)

    해석: 가중치가 고르지 않으면 단위당 NLL에 곱해 로그 가능도로 바꿀 때 단위 수보다 작은 표본 크기를 써서 정보량을 과장하지 않는다.
    Ref: Kish, L. (1965), "Survey Sampling", Wiley."""
    return float(w.sum() ** 2 / np.dot(w, w))


def boundary_p(lr: float) -> float:
    """The p value of the likelihood-ratio statistic ``lr`` for a parameter on the boundary of its space: the mixture ½ χ²(0) + ½ χ²(1) gives ½ P(χ²(1) ≥ lr), and 1 for lr = 0. (KF-11)

    Ref: Self, S. G. and Liang, K.-Y. (1987), J. Am. Stat. Assoc. 82(398): 605-610; doi:10.1080/01621459.1987.10478472."""
    return 1.0 if lr <= 0.0 else float(0.5 * chi2.sf(lr, 1))


# =============================================================================
# 블록 3: 칸 하나의 네 법칙
# =============================================================================
def _weighted_mean(w: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(w, x) / w.sum())


def _unit_scores(Ko, d_o, rate, law: str) -> np.ndarray:
    """(n,) the NLL ℓ_i of every unit under the independent law ``law`` ('poisson' or 'shifted') with the role rates ``rate`` (2,): the two teams' scores of ``killlaw.logpmf`` added."""
    return -killlaw.logpmf(Ko, np.broadcast_to(rate, Ko.shape), d_o, law).sum(axis=1)


def _shares(first: np.ndarray, other: np.ndarray) -> tuple:
    """``(no_kill, tie)`` of two independent teams whose count pmfs are ``first`` and ``other`` (vectors over k = 0, 1, …). (KF-7)"""
    return float(first[0] * other[0]), float(np.dot(first, other))


def _weighted_nll(w: np.ndarray, scores: np.ndarray) -> float:
    return float(np.dot(w, scores) / w.sum())


def _group_levels(groups: np.ndarray, k: np.ndarray, w: np.ndarray) -> tuple:
    """``(scores, levels)``: the margin law of one team inside every group (the weighted frequency of each count), as the NLL ``−log p̂_g(k_i)`` of every unit and the number of counts seen
    in each group (I_g or J_g). (KF-9)"""
    rows = np.column_stack([groups, k])
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    mass = np.bincount(inverse, weights=w, minlength=len(unique))
    group_ids, group_index = np.unique(unique[:, 0], return_inverse=True)
    group_index = group_index.reshape(-1)
    group_mass = np.bincount(group_index, weights=mass, minlength=len(group_ids))
    return -np.log((mass / group_mass[group_index])[inverse]), np.bincount(group_index, minlength=len(group_ids))


def _independent_table(Ko: np.ndarray, w: np.ndarray, groups: np.ndarray, joint_nll: float, n_eff: float) -> dict:
    """The reference of the joint table (KF-9): in every group the product of the two weighted empirical margins, zero outside each margin's support. Its NLL, parameter count and AIC, and
    the dependence that the table has on top of it: the mutual information MI = NLL(independent table) − NLL(joint table) (≥ 0), G = 2 n_eff MI, the degrees of freedom
    Σ_g (I_g − 1)(J_g − 1) and the chi-square p value (None without a degree of freedom). ``levels`` is [Σ_g I_g, Σ_g J_g]: the number of count values seen in each margin, summed over the
    groups, so under ``joint_given='mark'`` it is a sum over the marks and not the level count of one table (with one group, the cell, it is I and J).

    해석: 결합 표의 NLL은 어떤 자료에서도 가장 낮으므로 그것만으로는 의존을 말할 수 없다. 같은 표본에서 두 주변 분포를 따로 맞춘 독립 표와 견주면 남는 차이가 의존 하나이다(주변 분포의 불일치는 두 표에 똑같이 들어 있다).
    Ref: Cover, T. M. and Thomas, J. A. (2006), "Elements of Information Theory", 2nd ed., Wiley (MI = KL(결합 ‖ 주변의 곱) ≥ 0); Agresti, A. (2013), "Categorical Data Analysis", 3rd ed. (G = 2 n MI)."""
    s1, levels1 = _group_levels(groups, Ko[:, 0], w)
    s2, levels2 = _group_levels(groups, Ko[:, 1], w)
    nll = _weighted_nll(w, s1 + s2)
    n_params = int((levels1 - 1).sum() + (levels2 - 1).sum())
    df = int(((levels1 - 1) * (levels2 - 1)).sum())
    mi = max(nll - joint_nll, 0.0)
    g_stat = 2.0 * n_eff * mi
    return {'independent_table': {'nll': nll, 'n_params': n_params, 'aic': 2.0 * n_eff * nll + 2.0 * n_params}, 'mutual_information': mi, 'G': g_stat, 'df': df,
            'p_value': float(chi2.sf(g_stat, df)) if df > 0 else None, 'levels': [int(levels1.sum()), int(levels2.sum())]}


def _joint_law(Ko: np.ndarray, w: np.ndarray, groups: np.ndarray) -> dict:
    """The table law (KF-5): per group the weighted frequency of every pair seen, nothing elsewhere; the NLL of the units under their own group's table, the shares and the tables."""
    rows = np.column_stack([groups, Ko])
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    mass = np.bincount(inverse, weights=w, minlength=len(unique))
    group_ids, group_index = np.unique(unique[:, 0], return_inverse=True)
    group_index = group_index.reshape(-1)
    group_mass = np.bincount(group_index, weights=mass, minlength=len(group_ids))
    prob = mass / group_mass[group_index]
    scores = -np.log(prob[inverse])
    total = float(w.sum())
    pair = unique[:, 1:]
    tables = {}
    for g_pos, g in enumerate(group_ids):
        key = 'all' if len(group_ids) == 1 and g < 0 else str(int(g))
        sel = group_index == g_pos
        tables[key] = [[int(a), int(b), float(p)] for (a, b), p in zip(pair[sel], prob[sel])]
    return {'nll': _weighted_nll(w, scores), 'no_kill_share': float(mass[(pair == 0).all(axis=1)].sum() / total),
            'tie_share': float(mass[pair[:, 0] == pair[:, 1]].sum() / total), 'first_team_zero_share': float(mass[pair[:, 0] == 0].sum() / total),
            'n_params': int(len(unique) - len(group_ids)), 'tables': tables}


def _tail(*pmfs: np.ndarray) -> float:
    """The largest mass that one of the pmf vectors ``pmfs`` leaves beyond its support (1 − the sum), at least 0."""
    return max(0.0, max(1.0 - float(p.sum()) for p in pmfs))


def _fit_cell(Ko: np.ndarray, d_o: np.ndarray, marks: np.ndarray, w: np.ndarray, champion: bool, joint_given: str) -> tuple:
    """The laws of one cell on the accepted units with positive weight, and what is read across them: ``(laws, extras)``. ``laws``: for each of the four laws the NLL, the shares, the
    parameters, their number and the AIC; ``extras``: ``dependence`` (the independent table, MI, G; KF-9), ``family_test`` (the boundary LR test of the NB; KF-11), ``support`` (the width of
    the pmf support and the mass beyond it), ``n_kept`` and ``n_eff``. ``Ko`` the counts (first team, other team), ``d_o`` the first teams, ``marks`` the marks.

    해석: 지지(support)는 모든 pmf의 지지 밖 질량이 TAIL_TARGET보다 작아질 때까지 두 배로 늘린다. 상한(SUPPORT_CAP)에서도 모자라면 멈추지 않고 남은 질량(``tail_mass``)을 보고한다(몫만 그 질량만큼 낮게 합쳐진다; NLL은 지지를 읽지 않는다)."""
    axes = ['first_team', 'other_team'] if champion else ['blue_team', 'red_team']
    names = [f'rate_{a}' for a in axes]
    law_of = 'shifted' if champion else 'poisson'
    n_eff = _kish(w)
    plain_rate = np.array([_weighted_mean(w, Ko[:, 0]), _weighted_mean(w, Ko[:, 1])])
    shifted_rate = plain_rate - np.array([1.0, 0.0]) if champion else plain_rate.copy()
    mean, alpha = _fit_nb(Ko[:, 0] - 1, w) if champion else (None, None)

    def pmfs(rate, law, width):
        p, _ = killlaw.pmf_head(rate[None, :], d_o[:1], law, width)                        # the pmfs of the two roles, one pair for the cell: the rates are per role
        return p[0, 0], p[0, 1]

    width = int(min(SUPPORT_CAP, max(SUPPORT_MIN, SUPPORT_PER_COUNT * Ko.max() + SUPPORT_SLACK)))
    while True:
        p_first, p_other = pmfs(plain_rate, 'poisson', width)
        s_first, s_other = pmfs(shifted_rate, law_of, width)
        if champion:
            nb_first = np.zeros(width)
            nb_first[1:] = np.exp(nb_logpmf(np.arange(width - 1), mean, alpha))
        tail = _tail(p_first, p_other, s_first, s_other, *((nb_first,) if champion else ()))
        if tail < TAIL_TARGET or width >= SUPPORT_CAP:
            break
        width = min(2 * width, SUPPORT_CAP)
    out = {}
    out['poisson'] = {'nll': _weighted_nll(w, _unit_scores(Ko, d_o, plain_rate, 'poisson')), 'params': dict(zip(names, map(float, plain_rate))), 'n_params': 2}
    out['poisson']['no_kill_share'], out['poisson']['tie_share'] = _shares(p_first, p_other)
    out['shifted'] = {'nll': _weighted_nll(w, _unit_scores(Ko, d_o, shifted_rate, law_of)), 'params': dict(zip(names, map(float, shifted_rate))), 'n_params': 2}
    out['shifted']['no_kill_share'], out['shifted']['tie_share'] = _shares(s_first, s_other)
    if champion:
        other_scores = -killlaw.logpmf(Ko, np.broadcast_to(shifted_rate, Ko.shape), d_o, 'shifted')[:, 1]
        scores = -nb_logpmf(Ko[:, 0] - 1, mean, alpha) + other_scores
        out['shifted_nb'] = {'nll': _weighted_nll(w, scores), 'params': {'alpha': alpha, 'mean_excess': mean, names[1]: float(shifted_rate[1])}, 'n_params': 3}
        out['shifted_nb']['no_kill_share'], out['shifted_nb']['tie_share'] = _shares(nb_first, s_other)
        zero = {'poisson': float(p_first[0]), 'shifted': float(s_first[0]), 'shifted_nb': float(nb_first[0])}                # P(K_1 = 0): what the plain law misplaces
    else:                                                                       # an execution cell has no first team: nothing is shifted, the NB has no N_d to spread
        out['shifted_nb'] = {'nll': out['shifted']['nll'], 'params': {'alpha': None, 'mean_excess': None, **out['shifted']['params']}, 'n_params': 2,
                             'no_kill_share': out['shifted']['no_kill_share'], 'tie_share': out['shifted']['tie_share']}
        zero = {'poisson': None, 'shifted': None, 'shifted_nb': None}
    groups = np.full(len(Ko), -1, dtype=np.int64) if joint_given == 'role' else marks
    out['joint'] = _joint_law(Ko, w, groups)
    out['joint']['axes'] = axes
    if not champion:
        out['joint']['first_team_zero_share'] = None
    for law in ('poisson', 'shifted', 'shifted_nb'):
        out[law]['first_team_zero_share'] = zero[law]
    for law in LAWS:                                                            # (KF-10)
        out[law]['aic'] = 2.0 * n_eff * out[law]['nll'] + 2.0 * out[law]['n_params']
    lr = max(0.0, 2.0 * n_eff * (out['shifted']['nll'] - out['shifted_nb']['nll']))
    family = {'lr': lr, 'p_value': boundary_p(lr), 'alpha': alpha, 'n_eff': n_eff} if champion else None
    extras = {'n_kept': int(len(Ko)), 'n_eff': n_eff, 'dependence': _independent_table(Ko, w, groups, out['joint']['nll'], n_eff), 'family_test': family,
              'support': {'width': int(width), 'tail_mass': float(tail)}}
    return {law: out[law] for law in LAWS}, extras


def _observed(Ko: np.ndarray, w: np.ndarray) -> dict:
    total = float(w.sum())
    return {'no_kill_share': float(w[Ko.sum(axis=1) == 0].sum() / total), 'tie_share': float(w[Ko[:, 0] == Ko[:, 1]].sum() / total)}

# =============================================================================
# 블록 4: 라벨 수준 비교
# =============================================================================
def compare(K, mark, size, w, *, joint_given: str = 'role') -> dict:
    """The four kill laws in every cell (size x mark kind), fitted and scored on the same units (KF-2 … KF-8); see the module text for the fields. Inputs are FIT labels only (the caller's duty).

    ``K`` (n, 2) the kill counts (blue, red), ``mark`` (n,) the first-kill marks 0 … 51, ``size`` (n,) the size classes 0 … 3, ``w`` (n,) the weights. ``joint_given``: 'role' (the table of a
    cell) or 'mark' (a table for each mark of the cell). Units that the shifted law refuses (``refused_units``) are left out of all four laws and counted; a cell without a unit of positive
    weight has ``laws = None``. Returns lists, numbers, strings and None only (JSON).

    해석: 칸마다 네 법칙의 단위당 NLL과 예측 무킬·무승부 몫을 낸다. 전체(``total``)는 칸의 NLL을 칸 가중치로 가중 평균한 값이다."""
    if joint_given not in JOINT_GIVEN:
        raise ValueError(f'joint_given must be one of {JOINT_GIVEN}, not {joint_given!r}')
    K, mark, size, w = _check_inputs(K, mark, size, w)
    refused = refused_units(K, mark)
    champion = mark < killlaw.N_PAIR_MARKS
    cells, sums = {}, {law: 0.0 for law in LAWS}
    weight = 0.0
    for s, name in enumerate(units.SIZE_NAMES):
        for kind in KINDS:
            in_cell = (size == s) & (champion == (kind == 'champion'))
            keep = in_cell & ~refused & (w > 0)
            entry = {'size': name, 'kind': kind, 'n_units': int(in_cell.sum()), 'n_dropped': int((in_cell & refused).sum()), 'weight': float(w[keep].sum()), 'n_kept': 0, 'n_eff': None, 'observed': None, 'laws': None,
                     'dependence': None, 'family_test': None, 'support': None}
            if keep.any():
                Ko, d_o = _orient(K[keep], mark[keep])
                entry['observed'] = _observed(Ko, w[keep])
                entry['laws'], extras = _fit_cell(Ko, d_o, mark[keep], w[keep], kind == 'champion', joint_given)
                entry.update(extras)
                weight += entry['weight']
                for law in LAWS:
                    sums[law] += entry['weight'] * entry['laws'][law]['nll']
            cells[f'{name}/{kind}'] = entry
    total = {'n_units': int(len(K)), 'n_dropped': int(refused.sum()), 'weight': weight,
             'laws': {law: {'nll': sums[law] / weight} for law in LAWS} if weight > 0 else None}
    return {'laws': list(LAWS), 'sizes': list(units.SIZE_NAMES), 'kinds': list(KINDS), 'joint_given': joint_given, 'dropped_reason': DROPPED_REASON, 'cells': cells, 'total': total}


# =============================================================================
# 블록 5: B1 수준 비교
# =============================================================================
class B1Rows(NamedTuple):
    """The rows of the kill head of one role: ``X`` (n, p) the design of the head, ``K`` (n, 2) the counts, ``mark`` (n,) the first-kill marks and ``w`` (n,) the weights of the score (None: 1).

    The same design is given to both laws. A masked head has the mark in its design (the shifted law reads d from it); an open head needs a column for the first-kill team d (``killlaw.first_team``)
    in ``X``, which the caller adds for both laws. The weights are used for the evaluation score only (the fits of the B1 path are unweighted): compare_b1 refuses a w on the fit or stop rows."""
    X: np.ndarray
    K: np.ndarray
    mark: np.ndarray
    w: np.ndarray | None = None


def _check_rows(role: str, rows: B1Rows, width: int | None) -> tuple:
    X = np.asarray(rows.X)
    if X.ndim != 2:
        raise ValueError(f'{role}: the design X must be 2-d, not shape {X.shape}')
    if width is not None and X.shape[1] != width:
        raise ValueError(f'{role}: the design has {X.shape[1]} columns, the fit rows {width}')
    w = np.ones(len(X)) if rows.w is None else rows.w
    K, mark, _, w = _check_inputs(rows.K, rows.mark, np.zeros(len(X), dtype=np.int64), w)
    if len(K) != len(X):
        raise ValueError(f'{role}: X has {len(X)} rows and K {len(K)}')
    return X, K, mark, w


def _weighted_correlation(w: np.ndarray, a: np.ndarray, b: np.ndarray):
    """The weighted Pearson correlation of ``a`` and ``b`` (None for fewer than 2 units or a constant variable)."""
    if len(a) < 2:
        return None
    ma, mb = _weighted_mean(w, a), _weighted_mean(w, b)
    va, vb = _weighted_mean(w, (a - ma) ** 2), _weighted_mean(w, (b - mb) ** 2)
    if va <= 0.0 or vb <= 0.0:
        return None
    return float(_weighted_mean(w, (a - ma) * (b - mb)) / np.sqrt(va * vb))


def conditional_statistics(head, X, K, mark, w) -> dict:
    """The conditional statistics of (KF-12) on the evaluation rows, with the means of the fitted shifted head ``head`` held fixed.

    A head of another kill law than 'shifted' is refused (ValueError): its mean of K_d is not the mean of N_d (a head that names no law, such as a test stub, is read as shifted).
    ``champion``: the one-parameter NB2 dispersion of N_d = k_1 − 1 around the head's λ̂_i, the weighted NLL of the Poisson and of the NB2 law with that α (their gain, the boundary LR test
    with the effective sample size, KF-11) and the weighted correlation of the two teams' Pearson residuals; ``execution``: ``n_units``, ``n_eff`` and that correlation for the blue and
    red counts (no first team). None for a kind without a unit.

    해석: 머리의 평균으로 설명되고 남은 과대분산 α̂_c와 잔차 상관 ρ가 D5가 묻는 조건부 족과 조건부 의존이다. 특징이 설명하는 강도 차이는 이미 λ̂_i, μ̂_i에 들어 있으므로 라벨 수준 통계와 달리 그것으로는 α̂_c와 ρ가 커지지 않는다.
    Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed. (피어슨 잔차와 평균을 고정한 과대분산 점검)."""
    law = getattr(head, 'kill_law', 'shifted')
    if law != 'shifted':
        raise ValueError(f'the conditional statistics read the excess N_d around the mean of the shifted head, not of a head of the kill law {law!r}')
    rate = head.predict(X)                                                          # (n, 2): the head's mean of the blue and of the red cell
    d = killlaw.first_team(mark)
    out = {'champion': None, 'execution': None}
    champion = d >= 0
    if champion.any():
        idx = np.flatnonzero(champion)
        dd = d[idx]
        lam, mu = rate[idx, dd], rate[idx, 1 - dd]
        n_excess = K[idx, dd] - 1
        other = K[idx, 1 - dd]
        wc = w[idx]
        pos = wc > 0
        if pos.any():
            alpha = _fit_alpha(n_excess[pos], lam[pos], wc[pos])
            nll_poisson = _weighted_nll(wc[pos], -nb_logpmf(n_excess[pos], lam[pos], 0.0))
            nll_nb = _weighted_nll(wc[pos], -nb_logpmf(n_excess[pos], lam[pos], alpha))
            n_eff = _kish(wc[pos])
            lr = max(0.0, 2.0 * n_eff * (nll_poisson - nll_nb))
            out['champion'] = {'n_units': int(len(idx)), 'n_eff': n_eff, 'alpha': alpha, 'nll_poisson': nll_poisson, 'nll_nb': nll_nb, 'gain': nll_poisson - nll_nb, 'lr': lr,
                               'p_value': boundary_p(lr),
                               'residual_correlation': _weighted_correlation(wc[pos], ((n_excess - lam) / np.sqrt(lam))[pos], ((other - mu) / np.sqrt(mu))[pos])}
    if (~champion).any():
        idx = np.flatnonzero(~champion)
        wx = w[idx]
        pos = wx > 0
        res = (K[idx] - rate[idx]) / np.sqrt(rate[idx])
        out['execution'] = {'n_units': int(len(idx)), 'n_eff': _kish(wx[pos]) if pos.any() else None,
                            'residual_correlation': _weighted_correlation(wx[pos], res[pos, 0], res[pos, 1])}
    return out


def compare_b1(fit: B1Rows, stop: B1Rows, evaluate: B1Rows, *, kind: str = 'masked', layout=None, categorical=(), params=None, check=None) -> dict:
    """The kill head of the B1 path fitted with ``kill_law='poisson'`` and with ``'shifted'`` on the same features, scored on the same evaluation units, and the conditional statistics
    on which the D5 family and dependence are decided (author decision D29: the B1-level conditional statistics decide, the label-level ones of ``compare`` are descriptive; the rule and
    its thresholds are fixed in the S4b run plan before the real labels are opened).

    Both heads are fitted on ``fit`` with the early stopping on ``stop`` (``baselines_b1.M3Kills``, the head and the objective of the B1 path); the units that the shifted law refuses
    (``refused_units``) are left out of the fit rows, the stop rows and the evaluation rows for both laws, and counted per role (M1 review m9). ``evaluate.w`` weights the score
    (None: 1); a ``w`` on ``fit`` or ``stop`` is refused (ValueError), because the fits of the B1 path are unweighted and it would be ignored. ``kind``, ``layout``, ``categorical``,
    ``params`` and ``check`` go to the head. Both laws are fitted on the same design: for an open head the caller adds the first-team column d to ``X`` for both laws, so the plain NLL
    here is not the NLL of the production plain open head (whose layout has no d); the comparison isolates the law, not the layout.

    Returns ``n_units`` per role (total, dropped), ``dropped_reason``, ``nll`` per law, the ``difference`` poisson − shifted (positive: the shifted law scores better), the weight of the
    evaluation rows and ``conditional`` (KF-12): with the shifted head's means held fixed on the evaluation rows, ``champion`` = the NB2 dispersion ``alpha`` of N_d (one parameter),
    the NLLs of the Poisson and the NB2 law, their ``gain``, the boundary ``lr`` and ``p_value`` and the ``residual_correlation`` of the two teams' Pearson residuals, and ``execution`` = ``n_units``,
    ``n_eff`` (Kish) and that correlation for the blue and red counts (None for a kind without a unit).

    (KF-6)을 B1 머리로 한 것이다: 점수는 ``M3Kills.nll``이고 이동 법칙은 첫 킬 팀을 따로 받는다. 해석: 같은 특징, 같은 단위, 같은 조기 종료 행으로 법칙만 바꾼 비교이다. 조건부 통계는 평가 행의
    같은 머리 예측으로 맞추므로 머리를 다시 맞추지 않는다(모수는 α 하나)."""
    for role, rows in (('fit', fit), ('stop', stop)):
        if rows.w is not None:
            raise ValueError(f'{role}.w is given but the fits of the B1 path are unweighted: it would be ignored (only evaluate.w weights the score)')
    X_f, K_f, m_f, _ = _check_rows('fit', fit, None)
    width = X_f.shape[1]
    X_s, K_s, m_s, _ = _check_rows('stop', stop, width)
    X_e, K_e, m_e, w_e = _check_rows('evaluate', evaluate, width)
    kept = {}
    n_units = {}
    for role, (X, K, mark, w) in {'fit': (X_f, K_f, m_f, None), 'stop': (X_s, K_s, m_s, None), 'evaluate': (X_e, K_e, m_e, w_e)}.items():
        ok = ~refused_units(K, mark)
        n_units[role] = {'total': int(len(K)), 'dropped': int((~ok).sum())}
        if not ok.any():
            raise ValueError(f'no unit of the {role} rows is accepted by both laws')
        kept[role] = (X[ok], K[ok], mark[ok], None if w is None else w[ok])
    fx, fk, fm, _ = kept['fit']
    sx, sk, sm, _ = kept['stop']
    ex, ek, em, ew = kept['evaluate']
    ones = {'fit': np.ones(len(fk)), 'stop': np.ones(len(sk)), 'evaluate': np.ones(len(ek))}
    if (ew <= 0).all():
        raise ValueError('no unit of the evaluate rows has a positive weight')
    plain = baselines_b1.M3Kills(kind, params, 'poisson').fit(fx, fk, ones['fit'], categorical=categorical, layout=layout, valid=(sx, sk, ones['stop']), check=check)
    shifted = baselines_b1.M3Kills(kind, params, 'shifted').fit(fx, fk, ones['fit'], categorical=categorical, layout=layout, valid=(sx, sk, ones['stop']), check=check,
                                                                first_team=killlaw.first_team(fm), valid_first_team=killlaw.first_team(sm))
    nll_plain = plain.nll(ex, ek, ones['evaluate'])
    nll_shifted = shifted.nll(ex, ek, ones['evaluate'], first_team=killlaw.first_team(em))
    nll = {'poisson': _weighted_mean(ew, nll_plain), 'shifted': _weighted_mean(ew, nll_shifted)}
    return {'laws': ['poisson', 'shifted'], 'n_units': n_units, 'dropped_reason': DROPPED_REASON, 'nll': nll, 'difference': nll['poisson'] - nll['shifted'], 'weight': float(ew.sum()),
            'conditional': conditional_statistics(shifted, ex, ek, em, ew)}
