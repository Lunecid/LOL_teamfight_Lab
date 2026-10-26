r"""킬 수 법칙 (S4b 준비 M1, 저자 결정 D5): 첫 킬 팀의 킬 수를 1 + N_d로 옮기는 이동 법칙과 옛 포아송 법칙을 한 곳에서 계산한다.

방법. 교전의 첫 킬은 챔피언이 챔피언을 처치한 킬(표지 m = 0 … 49)이면 그 킬을 낸 팀 d = m // 25의 킬 수가 적어도 하나이다. 옛 법칙은 두 팀의 킬 수를 모두 포아송으로 두어
그 팀의 킬 수가 0일 확률 e^{-μ_d}를 모형에 맡겼는데, 자료에서는 일어날 수 없는 사건이다(첫 킬이 그 팀의 킬이므로). 이동 법칙은 K_d = 1 + N_d로 쓰고 초과분 N_d에 포아송을 둔다.
다른 팀의 킬 수는 그대로 포아송이고, 처형 표지(m = 50, 51: 챔피언이 아닌 것에 죽음)는 두 팀 모두 옛 법칙이다. 머리가 내는 팀별 율 (n, 2)의 모양은 그대로이고, 이동 법칙에서는 d 칸이 λ_d
(초과분의 평균), 다른 칸이 μ(킬 수의 평균)이다. N_d의 족(포아송 또는 음이항)과 두 팀 사이의 결합 의존은 S4b의 C5 측정(Task M16)이 정하므로 여기서는 포아송만 둔다.

식 (식 번호는 ``(KL-n)``; 기호 표는 아래)

    (KL-1)  d = m // 25  (m = 0 … 49),   d = −1  (m = 50, 51: 첫 킬 팀 없음)
    (KL-2)  poisson:  K_t ~ Poisson(r_t)  (t = 0, 1)  ;   log P(K_t = k) = −r_t + k log r_t − log k!
    (KL-3)  shifted:  K_d = 1 + N_d,  N_d ~ Poisson(r_d),  K_{1−d} ~ Poisson(r_{1−d})  (d ≥ 0) ;   d = −1이면 (KL-2)와 같다
            log P(K_d = k) = −r_d + (k − 1) log r_d − log (k − 1)!  (k ≥ 1),   log P(K_d = 0) = −∞
    (KL-4)  P(K_d ≥ k₀) = P(N_d ≥ k₀ − 1) = P(k₀ − 1, r_d)  (k₀ ≥ 2),   P(K_d ≥ 1) = 1 ;   P(K ≥ k₀) = P(k₀, r)  (옛 법칙)    (P(a, x): 정규화 하위 불완전 감마 함수)
    (KL-5)  표본: k = Poisson(r) 한 번(모양 (n, 2)) + 1[t = d]  (이동 법칙).  난수 호출의 수와 순서가 옛 법칙과 같다

      기호   | 뜻                                                         | 단위 | 범위                      | 코드 변수
      m      | 첫 킬의 표지 (킬러 팀 2 × 킬러 슬롯 5 × 피해자 슬롯 5 = 50, 처형 2)  | —    | 0 … 51                    | m
      d      | 첫 킬 팀 (0 파랑, 1 빨강), 처형이면 −1                      | —    | −1, 0, 1                  | d, first_team(m)
      k      | 팀별 킬 수                                                  | 개   | 0, 1, …                   | k
      r_t    | 머리가 낸 팀 t의 율: 이동 법칙에서 d 칸은 초과분의 평균 λ_d, 나머지 칸은 킬 수의 평균 μ | 개 | ≥ 0 | rate
      N_d    | 첫 킬 팀의 초과 킬 수 K_d − 1                               | 개   | 0, 1, …                   | excess(k, d, law)
      P(a,x) | 정규화 하위 불완전 감마 함수 (포아송 꼬리 P(N ≥ a) = P(a, x)) | —    | [0, 1]                    | scipy.special.gammainc

해석. 이동 법칙은 가능도의 한 항을 정확히 바꾼다: 첫 킬 팀의 칸은 K_d − 1에 대한 포아송 점수이고, 그 팀의 킬 수가 0인 행은 확률 0(점수 +∞)이다. 이 행은 자료에서 나올 수 없으므로 행을 만드는 쪽이
먼저 거부한다(``head_rows``의 단언, 검토 초점 2). 첫 킬 팀은 입력에서 관측되므로(마스크 읽기는 표지 m, 열린 읽기는 첫 킬 토큰) 모형이 d를 몰라서 생기는 문제는 없다.
기본값 ``law='poisson'``은 옛 코드와 같은 연산을 같은 순서로 하므로 비트 단위로 같은 값을 낸다(시험이 확인한다).

근거. 저자 결정 D5 (B1, 킬 수 지지: 이동 법칙 1 + N_d, 동률 규칙 (i)); 검토 A3-F1, A2-F1, A4-F4; 계획 S4b 준비 Task M1과 Q4.
Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press; doi:10.1017/CBO9781139013567
     (포아송 우도; 0에서 잘리거나 한 칸 옮긴 계수 모형의 장은 이 구현의 근거가 아니라 같은 족의 서지 정보). Abramowitz, M. and Stegun, I. A. (1964), "Handbook of
     Mathematical Functions", 6.5.13 (포아송 꼬리는 정규화 하위 불완전 감마 함수).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc

from c_ept.models import nll as nll_scores

LAWS = ('poisson', 'shifted')
N_PAIR_MARKS = 50                    # marks 0 .. 49 are pairs (killer team, killer slot, victim slot); 50 and 51 are executions (``units.EXEC_MARK0``)
PAIRS_PER_TEAM = 25                  # mark = (d 5 + i) 5 + j: the 25 marks of the team d are 25 d .. 25 d + 24 (``units.first_kill_mark``)
N_MARKS = N_PAIR_MARKS + 2
N_TEAMS = 2
N_HEAD = 6                           # the closed form writes P(K = k) for k = 0 .. 5 and the tail P(K >= 6) (``compose.N_POISSON``)


def check_law(law) -> str:
    """``law`` itself if it is one of ``LAWS``; ValueError naming ``kill_law`` otherwise."""
    if not isinstance(law, str) or law not in LAWS:
        raise ValueError(f'kill_law must be one of {LAWS}, not {law!r}')
    return law


# ---- the first-kill team of a mark -------------------------------------------------------------------------------------------------------
def _integers(what: str, values) -> np.ndarray:
    a = np.asarray(values)
    if a.size == 0 and a.ndim <= 1:
        a = a.astype(np.int64)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{what} must be integers, not {a.dtype}')
    return a.astype(np.int64)


def first_team(m) -> np.ndarray:
    """(...) int64: the team that took the first kill of the mark ``m`` (0 .. 51): ``m // 25`` for a champion mark (0 .. 49), −1 for an execution (50, 51).

    (KL-1)의 d를 계산한다. ValueError for a mark outside 0 .. 51, TypeError for a mark that is not an integer."""
    a = _integers('a mark', m)
    if a.size and (a.min() < 0 or a.max() >= N_MARKS):
        raise ValueError(f'a mark is 0 .. {N_MARKS - 1}, not {int(a.min())} .. {int(a.max())}')
    return np.where(a < N_PAIR_MARKS, a // PAIRS_PER_TEAM, -1)


def torch_first_team(m, check_range: bool = True):
    """The same as ``first_team`` for a torch tensor of marks (long, on its own device); ValueError and TypeError as ``first_team``.

    ``check_range=False`` (S4b speed-ups item 3, D32) leaves out the one check that reads the values back from the device (the range of the marks); the type check stays. The loss passes it for
    a step that runs trusted, whose batch passed the very same check on the CPU (``c_ept.ept.step_guard``); the default is the call of every earlier run.

    해석: 손실이 표지 텐서에서 바로 d를 구한다(마스크 행의 d는 표지에서 나온다). 범위 점검은 CPU에서 이미 같은 함수로 끝난 걸음에서만 끄므로 값은 달라지지 않는다."""
    import torch
    if m.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f'a mark must be an integer tensor (not uint8 or bool), not {m.dtype}')
    if check_range and m.numel() and (int(m.min()) < 0 or int(m.max()) >= N_MARKS):
        raise ValueError(f'a mark is 0 .. {N_MARKS - 1}')
    m = m.to(torch.long)
    return torch.where(m < N_PAIR_MARKS, torch.div(m, PAIRS_PER_TEAM, rounding_mode='floor'), torch.full_like(m, -1))


def _first_teams(d, n: int | None = None) -> np.ndarray:
    a = _integers('the first team d', d)
    if a.ndim != 1 or (n is not None and len(a) != n):
        raise ValueError(f'the first team d must have shape ({n if n is not None else "n"},), not {a.shape}')
    if a.size and (a.min() < -1 or a.max() > 1):
        raise ValueError('the first team d is 0 (blue), 1 (red) or -1 (an execution: no first team)')
    return a


def shift(d, law: str) -> np.ndarray:
    """(n, 2) int64: 1 at the entry of the first team and 0 elsewhere when ``law`` is 'shifted' (the count of that team is 1 + N_d); zeros for the plain law and
    for an execution (d = −1). ValueError for a d outside −1 .. 1.

    (KL-3)의 K_d = 1 + N_d에서 옮긴 몫 1[t = d]이다."""
    law = check_law(law)
    a = _first_teams(d)
    if law == 'poisson':
        return np.zeros((len(a), N_TEAMS), dtype=np.int64)
    return (a[:, None] == np.arange(N_TEAMS)[None]).astype(np.int64)


def excess(k, d, law: str) -> np.ndarray:
    """(n, 2) int64: the counts of the Poisson part of each cell, ``k − shift(d, law)``: N_d for the first team, K for the other. ValueError, with the number of rows
    (never a match id), when a first team has no kill (the shifted law gives that row the probability 0).

    해석: 머리를 적합할 때의 정답(B0·B1의 포아송 적합은 초과분을 목표로 삼는다)이고, 단언 하나이다. 옛 법칙은 k를 그대로 돌려준다."""
    k = _integers('the counts k', k)
    if k.ndim != 2 or k.shape[1] != N_TEAMS:
        raise ValueError(f'the counts k must have shape (n, {N_TEAMS}), not {k.shape}')
    out = k - shift(d, law) if law == 'shifted' else k.copy()
    if len(out) != len(k) or (out < 0).any():
        bad = int((out < 0).any(axis=1).sum())
        raise ValueError(f'{bad} row(s) have a first team with no kill: the shifted law gives them the probability 0')
    return out


# ---- the mass ----------------------------------------------------------------------------------------------------------------------------
def logpmf(k, rate, d, law: str) -> np.ndarray:
    """(n, 2) float64: log P(K_t = k_t) of the two teams of every row under ``law`` for the rates ``rate`` (n, 2); −inf for a first team with no kill under the shifted law.

    ``law='poisson'`` is minus ``nll.poisson`` with the exposure 1 (the score of the heads of earlier runs), ``d`` is not read. For 'shifted', ``d`` (n,) is the first team of
    each row (−1 for an execution: the plain law).

    (KL-2)와 (KL-3)의 로그 질량이다. 이동 법칙은 첫 킬 팀의 칸에서 k − 1에 대한 포아송 점수를 쓰고 k = 0이면 −∞이다."""
    k = np.asarray(k, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    law = check_law(law)
    if law == 'poisson':
        return -nll_scores.poisson(k, rate, 1.0)
    y = k - shift(d, law)
    impossible = y < 0
    score = nll_scores.poisson(np.where(impossible, 0.0, y), rate, 1.0)            # an impossible cell is scored with a harmless 0 and replaced below
    return np.where(impossible, -np.inf, -score)


def torch_logpmf(k, rate, d, law: str, check_range: bool = True):
    """The same values as ``logpmf`` as a differentiable float64 tensor: ``k`` (n, 2) counts, ``rate`` (n, 2) the rates, ``d`` (n,) integer tensor (not read for 'poisson').
    ``check_range=False``: as for ``torch_first_team``, the range of ``d`` is not read back (a trusted step, S4b speed-ups item 3, D32).

    해석: (LOSS-1)의 `_poisson_cells`와 같은 연산이다(로그는 y > 0일 때만 취해 0·log 0과 기울기 NaN을 막는다). 불가능한 칸은 −∞이고 그 칸으로 기울기가 가지 않는다."""
    import torch
    law = check_law(law)
    k = torch.as_tensor(k).to(torch.float64)
    rate = torch.as_tensor(rate).to(device=k.device, dtype=torch.float64)
    if law == 'shifted':
        d = torch.as_tensor(d, device=k.device)
        if d.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f'the first team d must be an integer tensor, not {d.dtype}')
        if check_range and d.numel() and (int(d.min()) < -1 or int(d.max()) > 1):
            raise ValueError('the first team d is 0 (blue), 1 (red) or -1 (an execution: no first team)')
        y = k - (d.to(torch.long)[:, None] == torch.arange(N_TEAMS, device=k.device)[None]).to(torch.float64)
    else:
        y = k
    impossible = y < 0
    y = torch.where(impossible, torch.zeros_like(y), y)
    log_rate = torch.log(torch.where(y > 0, rate, torch.ones_like(rate)))             # 0 log 0 = 0; y > 0 with rate 0 gives −inf and the score +inf
    cell = rate - y * log_rate + torch.lgamma(y + 1.0)
    return torch.where(impossible, torch.full_like(cell, -float('inf')), -cell)


# ---- the sample --------------------------------------------------------------------------------------------------------------------------
def sample(rng, rate, d, law: str) -> np.ndarray:
    """(n, 2) int64: the kills of the two teams of every row. One call ``rng.poisson(rate)`` of the shape of ``rate``, then 1 is added to the entry of the first team under the
    shifted law, so that the random stream is that of the plain law whatever the law is. ``d`` is not read for 'poisson'.

    (KL-5): 난수 순서(표지, 구역, 규모 → 킬 수 → 골드 → 오브젝트 → 전환)를 바꾸지 않는다."""
    law = check_law(law)
    k = rng.poisson(rate).astype(np.int64)
    if law == 'shifted':
        k += shift(d, law)
    return k


# ---- the closed form ---------------------------------------------------------------------------------------------------------------------
def tail_ge(k0: int, rate, d, law: str) -> np.ndarray:
    """P(K_t ≥ k0) of every cell of ``rate`` (n, 2) as an array of its shape: ``gammainc(k0, rate)`` for the plain law, and for the first team under the shifted law
    P(N_d ≥ k0 − 1) (1 if k0 ≤ 1). ``k0`` is a whole number of at least 1.

    (KL-4)의 꼬리이다. 합성의 닫힌 꼴이 킬 여섯 개 이상의 확률 한 칸으로 쓴다."""
    law = check_law(law)
    if isinstance(k0, bool) or not isinstance(k0, (int, np.integer)) or k0 < 1:
        raise ValueError(f'k0 must be a whole number of at least 1, not {k0!r}')
    rate = np.asarray(rate, dtype=np.float64)
    plain = gammainc(float(k0), rate)
    if law == 'poisson':
        return plain
    first = shift(d, law).astype(bool)
    if first.shape != rate.shape:
        raise ValueError(f'rate must have shape {first.shape} (one entry for each team of each row), not {rate.shape}')
    moved = np.ones_like(rate) if k0 == 1 else gammainc(float(k0 - 1), rate)
    return np.where(first, moved, plain)


def pmf_head(rate, d, law: str, width: int = N_HEAD) -> tuple:
    """``(p, tail)`` of the closed form: ``p`` (..., ``width``) with p[..., k] = P(K = k) for k = 0 … width − 1 and ``tail`` (...) = P(K ≥ width), for the rates ``rate`` (..., 2)
    [(n, 2) and (n,) ``d``]. The plain law is the table of the composer of earlier runs (the recurrence p_k = p_{k−1} r / k and the tail ``gammainc(width, r)``), operation for
    operation; under the shifted law the entries of the first team are that table moved by one place: P(K_d = k) = P(N_d = k − 1), P(K_d = 0) = 0, and the tail is P(N_d ≥ width − 1).

    (COMP-7)의 P(k; μ) (k = 0 … 5)와 꼬리 P(≥ 6; μ)이다(``compose._poisson_head``가 이 함수를 부른다)."""
    law = check_law(law)
    rate = np.asarray(rate, dtype=np.float64)
    p = np.empty(rate.shape + (width,))
    p[..., 0] = np.exp(-rate)
    for k in range(1, width):
        p[..., k] = p[..., k - 1] * rate / k
    tail = gammainc(float(width), rate)
    if law == 'poisson':
        return p, tail
    first = shift(d, law).astype(bool)
    if first.shape != rate.shape:
        raise ValueError(f'rate must have shape {first.shape} (one entry for each team of each row), not {rate.shape}')
    moved = np.zeros_like(p)
    moved[..., 1:] = p[..., :-1]
    return np.where(first[..., None], moved, p), np.where(first, gammainc(float(width - 1), rate), tail)
