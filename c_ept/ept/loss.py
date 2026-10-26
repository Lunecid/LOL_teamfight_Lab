r"""EPT의 손실 (S4a 과제 4): 머리별 음의 로그 우도(NLL), 머리의 평균, 규모별 정규화, B0 척도로 나눈 합.

이 모듈은 설계서 7절(머리별 NLL ÷ 그 머리의 B0 NLL, 같은 가중, M3는 규모별 정규화)과 8절(점수), 계획 Q8·P4·P9·P10·Q12, 판정 R13·R22·R-S4a-10·R-S4a-11을 구현한다.
이 모듈은 킬이 하나 이상 난 교전을 다룬다 (G-S3c C3). "블록"은 머리와 보기 종류의 한 쌍('M3Kills/open')이고 열세 개이다(가린 머리 8, 열린 머리 5).
방법. 머리의 출력에서 행이 읽는 칸의 점수를 매긴다. 포아송 머리는 기대 개수 μ에 대해 μ − y log μ + log y!를 (LOSS-1), 범주 머리는 균등 분포를 섞은 로그 확률의 음수를 (LOSS-2) 점수로 한다.
두 점수는 ``nll.py``의 점수와 수치로 같다(같은 변수, 같은 점수: 지침 F3). 한 머리의 손실은 점수가 있는 행의 평균이고 (LOSS-3), 총손실은 머리별 평균을 B0의 FIT 평균으로 나눈 합이다 (LOSS-4).
B0 평균은 ``scales.py``가 잰다.

출력에서 행이 읽는 칸 (n개 행, 행마다 보기 하나; 이름은 ``head_rows.HEAD_NAMES``, 출력은 ``ept/heads.py``)

    M2Rate   rate (n, 18)                [i, 구간]의 평균 μ = λ·e (노출 e > 0인 칸만)      y (n, 18) 개수              e (n, 18) 초
    M2Mark   mark_logp (n, 18, 52)        [i, q_i, :]                                        y (n,) 표지 0..51           q
    M2Zone   zone_logp (n, 18, 52, 10)    [i, q_i, m_i, :]                                   y (n,) 구역 −1..9           q, m
    M3Size   size_logp (n, 4)             [i, :]                                             y (n,) 규모 0..3
    M3Kills  kill_rate (n, 4, 2)          [i, s_i, 팀] 두 칸의 평균 (노출 1)                 y (n, 2) 팀별 킬 수         s
    M3Gold   gold_logp (n, 4, 2, 6, 31)   [i, s_i, 팀_i, 층_i − 1, :]                        y (n,) 골드 구간 0..30      s, team, layer (1..6)
    M3Obj    obj_rate (n, 18)             [i, 칸]의 평균 (노출 1)                            y (n, 18) 개수              s
    M4Conv   conv_rate (n, 18)            [i, 칸]의 평균 μ = λ·e                             y (n, 18) 개수              e (n,) 전환 창(초)

식 (식 번호는 ``(LOSS-n)``; 기호 표는 식을 계산하는 함수의 문서 문자열에 있다)

    (LOSS-1)  ℓ_{i,c} = μ_{i,c} − y_{i,c} log μ_{i,c} + log(y_{i,c}!)  (e_{i,c} > 0),   ℓ_{i,c} = 0  (e_{i,c} = 0) ;   ℓ_i = Σ_c ℓ_{i,c}
    (LOSS-2)  ℓ_i = −log p_{i,c_i}  (c_i ≥ 0),   ℓ_i = 0  (c_i = −1)
    (LOSS-3)  ℓ̄_head = (1 / N) Σ_{i 점수가 있는 행} ℓ_i ;   M3Kills·M3Gold·M3Obj: ℓ̄_head = (1 / n_size) Σ_{s 나온 규모} ℓ̄_{head,s},  ℓ̄_{head,s} = (1 / N_s) Σ_{i: s_i = s} ℓ_i
    (LOSS-4)  Loss_EPT = Σ_head ℓ̄_head / ℓ̄^B0_head

가정과 선택

    · M2Rate와 M4Conv의 출력은 평균 μ = λ·e이다 (판정 R-S4a-10). 식에서 노출을 다시 곱하지 않고, 노출은 칸을 관측했는지(e > 0)만 정한다.
    · 점수가 있는 행은 관측한 칸이나 범주가 하나라도 있는 행이다. B0가 점수에서 빼는 행(구역 −1, 노출 없음)과 같은 행이고 (판정 R22), 그런 행이 없는 블록은 손실에 0을 더한다.
    · 규모별 평균은 조건부 전문가 셋(M3Kills, M3Gold, M3Obj)에만 쓴다 (``SIZE_NORMALISED``; 설계서 3.2절). 규모 관문 M3Size는 주변 법칙이라 규모별로 가중하면 맞출 법칙이 바뀌므로
      행 평균이다 (검산 노트 04 B2).
    · 학습 행 가중은 모두 같다 (계획 P9). 점수는 float64로 계산하므로 출력이 float32여도 numpy 점수와 상대 1e-6 안에서 같다.

입력과 출력

    head_nll(head, out, target, exposure=None, *, kill_law='poisson') → (n,) float64 행 점수      scored_rows(head, target, exposure=None) → (n,) bool, 평균에 드는 행
    (kill_law='shifted'는 M3Kills의 이동 법칙 (LOSS-1k)이다: 첫 킬 팀의 킬 수가 1 + N_d, 결정 D5. 기본값은 옛 점수와 비트까지 같다.)
    head_losses(nlls, targets) → {블록: 머리의 평균}                         total_loss(nlls, scales, targets) → 0차원 float64, (LOSS-4)

믿는 걸음 (S4b 속도 개선 SPD 항목 3, 결정 D32; ``step_guard``, ``batch_prep``)

    · 이 모듈의 점검은 두 가지다. 자료 점검(범주·색인의 범위, 개수의 유한성과 정수성, 노출의 유한성과 부호)은 배치만의 함수이고, 모형 점검(칸의 평균이 음수이거나 유한하지 않다, 로그 법칙이 NaN이거나 1로
      합하지 않는다)은 모형이 낸 값의 함수이다. 이전에는 둘 다 걸음 안에서 장치의 값을 읽어 와(``int(t.min())``, ``bool(...any())``) 점검했고, 읽을 때마다 장치의 대기열이 비기를 기다렸다.
    · ``step_guard.trust(...)`` 안(믿는 걸음: 배치가 CPU에서 ``batch_prep.Preparation``으로 같은 함수로 점검을 마쳤다)에서는 자료 점검의 읽기를 건너뛰고(형·모양 점검은 읽기가 없으므로 그대로), 모형 점검은
      칸 수를 0차원 텐서로 ``StepGuards``에 미뤄 두었다가 걸음이 손실의 유한성과 함께 한 번에 읽는다(``training_step.optimize_batch``, 역전파 앞). 막히는 조건·오류의 종류·문구(칸 수 포함)·순서는 그대로이다.
      믿는 범위 밖(STOP 평가, 직접 부르는 쪽, 시험)은 모든 점검이 이전과 같이 즉시 일어난다. 점검을 모두 건너뛰는 것이 아니다: 건너뛰는 것은 CPU에서 같은 점검을 이미 통과한 자료의 읽기뿐이다.
    · ``counting_plan``은 평균에 드는 행의 번호 목록을 CPU에서 한 번 만들고, 믿는 걸음의 ``head_losses``는 불리언 마스크 ``rows[mask]`` 대신 ``rows.index_select(0, 목록)``으로 평균한다 (같은 원소, 같은 순서,
      같은 모양의 연속 텐서이므로 평균과 기울기가 비트까지 같다; 마스크는 몇 행인지 알려면 장치를 기다려야 한다).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md sections 7 (the loss: NLL of each head divided by the NLL of B0, equal weights, M3 normalised by size),
#      8 (the scores) and 3.2 (each expert is trained on the units of its size, the loss is normalised by size).
# Ref: docs/superpowers/plans/2026-10-22-c-ept-s4a.md decisions Q8 (the loss), Q12 and review focus 4 (cells without exposure); docs/superpowers/plans/2026-10-21-c-ept-s3.md
#      decisions P4 (one probability family for every model), P9 (one weight for every training row) and P10 (exposure); controller rulings R13, R22 (rows left out of a
#      mean are counted), R-S4a-10 (the means of M2 and M4 hold the exposure) and R-S4a-11 (one conversion window for each row); math-control note 04.
# Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press; doi:10.1017/CBO9781139013567 (the Poisson
#      log-likelihood with an exposure), as in ``nll``.
# Ref: Gneiting, T. and Raftery, A. E. (2007), "Strictly proper scoring rules, prediction, and estimation", JASA 102(477): 359-378; doi:10.1198/016214506000001437
#      (the logarithmic score is strictly proper: the expected loss is smallest at the true law).
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from c_ept.data import head_rows
from c_ept.data.head_rows import HEAD_NAMES, N_CELLS, N_GOLD, N_LAYER, N_M, N_Q, N_SIZE, N_ZONE, OPEN_HEAD_NAMES
from c_ept.ept import step_guard
from c_ept.models import killlaw

BLOCKS = tuple(f'{h}/masked' for h in HEAD_NAMES) + tuple(f'{h}/open' for h in OPEN_HEAD_NAMES)       # the thirteen blocks, in the order of the sum of the loss
SIZE_NORMALISED = frozenset({'M3Kills', 'M3Gold', 'M3Obj'})     # the experts P(K | s, x) and P(O | K, s, x); not the gate P(s | x)
ROW_SUM_TOLERANCE = 1e-4                                         # a probability row adds up to 1 within this: the number of ``nll`` (a test pins the equality)
if (N_Q, N_M, N_ZONE, N_SIZE, N_LAYER, N_GOLD, N_CELLS) != (18, 52, 10, 4, 6, 31, 18):
    raise RuntimeError('the widths of the heads (bins, marks, zones, sizes, kill layers, gold bins, cells) are not those of this module')

_OUT_TAIL = {'M2Rate': (N_Q,), 'M2Mark': (N_Q, N_M), 'M2Zone': (N_Q, N_M, N_ZONE), 'M3Size': (N_SIZE,), 'M3Kills': (N_SIZE, 2),
             'M3Gold': (N_SIZE, 2, N_LAYER, N_GOLD), 'M3Obj': (N_CELLS,), 'M4Conv': (N_CELLS,)}       # the shape of the output of a head after the row axis
_COUNT_WIDTH = {'M2Rate': N_Q, 'M3Kills': 2, 'M3Obj': N_CELLS, 'M4Conv': N_CELLS}                     # the Poisson heads: the cells of a row
_CATEGORIES = {'M2Mark': N_M, 'M2Zone': N_ZONE, 'M3Size': N_SIZE, 'M3Gold': N_GOLD}                    # the category heads: the classes of a row
_INDEX = {'M2Mark': {'q': (0, N_Q - 1)}, 'M2Zone': {'q': (0, N_Q - 1), 'm': (0, N_M - 1)}, 'M3Kills': {'s': (0, N_SIZE - 1)},
          'M3Gold': {'s': (0, N_SIZE - 1), 'team': (0, 1), 'layer': (1, N_LAYER)}}                  # the fields that pick the cell of a row, with their range
_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)         # an index or a category: not uint8 (a mask) and not bool
MEAN_MESSAGE = '{} observed cell(s) have a mean that is negative or not finite'      # the text of the model check of the rates, with the count of the cells where it goes
PLAN_PREFIX = 'plan/'                                                        # the keys of the index lists that `counting_plan` makes (`batch_prep`); no field of a block starts so
PLAN_NAMES = ('counted',) + tuple(f'size{k}' for k in range(N_SIZE))      # the lists of a block: the rows that count, and for the three experts the rows of each size


# ---- blocks ----------------------------------------------------------------------------------------------------------------------------------
def block_key(head: str, kind: str) -> str:
    """The key ``'<head>/<kind>'`` of a block; ValueError for a head and kind that do not exist (M2 has no open head)."""
    if (head, kind) not in head_rows.VALID_HEADS:
        raise ValueError(f'unknown block {head!r}/{kind!r}: the blocks are {BLOCKS}')
    return f'{head}/{kind}'


def split_key(key: str) -> tuple:
    """``(head, kind)`` of the block key ``key``; ValueError for a key that is not one of ``BLOCKS``."""
    head, slash, kind = str(key).partition('/')
    if not slash or (head, kind) not in head_rows.VALID_HEADS:
        raise ValueError(f'unknown block {key!r}: a block is "<head>/<kind>", one of {BLOCKS}')
    return head, kind


# ---- the checks of the inputs ----------------------------------------------------------------------------------------------------------------
def _check_head(head: str) -> None:
    if head not in _OUT_TAIL:
        raise ValueError(f'unknown head {head!r}: the heads are {HEAD_NAMES}')


def _fields(target) -> Mapping:
    """The block fields of a target: a mapping (``data.block``), or a tensor that stands for ``{'y': tensor}``."""
    if isinstance(target, torch.Tensor):
        return {'y': target}
    if not isinstance(target, Mapping):
        raise ValueError(f'the target must be a mapping of block fields or the tensor y, not {type(target).__name__}')
    return target


def _field(head: str, target: Mapping, name: str) -> torch.Tensor:
    if name not in target:
        raise ValueError(f"{head} needs the field '{name}' in its target")
    if not isinstance(target[name], torch.Tensor):
        raise ValueError(f"the field '{name}' of the target of {head} must be a tensor")
    return target[name]


def _index(name: str, t: torch.Tensor, n: int, low: int, high: int, like: torch.Tensor, what: str | None = None) -> torch.Tensor:
    """``t`` as an int64 tensor on the device of ``like`` after the check that it is an integer tensor (not uint8 or bool) of shape (n,) with values in low .. high."""
    if t.dtype not in _INT_DTYPES:
        raise TypeError(f'{name} must be an integer tensor (not uint8 or bool), not {t.dtype}')
    if tuple(t.shape) != (n,):
        raise ValueError(f'{name} must have the shape ({n},), one number for each row, not {tuple(t.shape)}')
    if n and not step_guard.trusted() and (int(t.min()) < low or int(t.max()) > high):      # a trusted step: the CPU made this check (`batch_prep`)
        raise ValueError(f'{what or name} must lie in {low} .. {high}')
    return t.to(device=like.device, dtype=torch.long)


def _counts(head: str, y: torch.Tensor, n: int, like: torch.Tensor) -> torch.Tensor:
    """The counts ``y`` (n, cells) of a Poisson head as float64 after the check that they are non-negative whole numbers."""
    width = _COUNT_WIDTH[head]
    if y.dtype == torch.bool or tuple(y.shape) != (n, width):
        raise ValueError(f'y of {head} must be a number tensor of shape ({n}, {width}), the counts of the cells, not {y.dtype} {tuple(y.shape)}')
    y = y.to(device=like.device, dtype=torch.float64)
    if not step_guard.trusted():
        bad = ~torch.isfinite(y) | (y < 0) | (y != torch.floor(y))
        if bool(bad.any()):
            raise ValueError(f'{int(bad.sum())} count(s) of {head} are not non-negative whole numbers')
    return y


def _exposure(e, n: int, width: int, like: torch.Tensor) -> torch.Tensor:
    """The exposure of M2Rate as float64 (n, width): a number, one value for each row (n,) or one for each cell (n, width), seconds, finite and not negative."""
    e = torch.as_tensor(e, dtype=torch.float64, device=like.device)
    if e.ndim == 0:
        e = e.expand(n, width)
    elif tuple(e.shape) == (n,):
        e = e[:, None].expand(n, width)
    elif tuple(e.shape) != (n, width):
        raise ValueError(f'the exposure of M2Rate must be a number, a tensor of shape ({n},) or one of shape ({n}, {width}), not {tuple(e.shape)}')
    if not step_guard.trusted() and (not bool(torch.isfinite(e).all()) or bool((e < 0).any())):
        raise ValueError('the exposure must be finite and not negative')
    return e


def _window(e, n: int, like: torch.Tensor) -> torch.Tensor:
    """The conversion window of M4Conv as float64 (n,): a number or one value for each row, seconds, finite and not negative; a value for each cell is refused (ruling R-S4a-11)."""
    e = torch.as_tensor(e, dtype=torch.float64, device=like.device)
    if e.ndim == 0:
        e = e.expand(n)
    elif tuple(e.shape) != (n,):
        raise ValueError(f'the exposure of M4Conv is the conversion window: a number or a tensor of shape ({n},), one value for each row, '
                         f'not shape {tuple(e.shape)} (ruling R-S4a-11)')
    if not step_guard.trusted() and (not bool(torch.isfinite(e).all()) or bool((e < 0).any())):
        raise ValueError('the exposure must be finite and not negative')
    return e


def _rows_of(head: str, target: Mapping) -> int:
    y = _field(head, target, 'y')
    if y.ndim < 1:
        raise ValueError(f'y of {head} must have a row axis')
    return int(y.shape[0])


# ---- the score of the cells and of the rows --------------------------------------------------------------------------------------------------
def _poisson_cells(mu: torch.Tensor, y: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    """Computes (LOSS-1) for the cells: ``mu`` (n, cells) float64 means, ``y`` float64 counts, ``seen`` bool (cells with exposure); the cell scores, 0 where not seen."""
    bad = seen & ~(torch.isfinite(mu) & (mu >= 0))
    guards = step_guard.current()
    if guards is not None:                                                                  # a trusted step: the model check is deferred, the data check was made on the CPU
        guards.defer(bad.sum(), MEAN_MESSAGE)
    else:
        if bool(bad.any()):
            raise ValueError(MEAN_MESSAGE.format(int(bad.sum())))
        bad = ~seen & (y > 0)
        if bool(bad.any()):
            raise ValueError(f'{int(bad.sum())} cell(s) have a count but no exposure')
    # The logarithm is taken only where the cell is observed and y > 0 (0 log 0 = 0; a count with the mean 0 has the score +infinity), so the junk mean of a cell that is
    # not observed (NaN, infinity, a negative number) meets no logarithm, whose gradient could turn it into NaN. The only operation that still sees it is the subtraction,
    # whose derivative is the constant 1, and the final ``where`` gives the cell the score 0 and the gradient 0.
    log_mu = torch.log(torch.where(seen & (y > 0), mu, torch.ones_like(mu)))
    cell = mu - y * log_mu + torch.lgamma(y + 1.0)
    return torch.where(seen, cell, torch.zeros_like(cell))


def _first_team(target: Mapping, n: int, like: torch.Tensor) -> torch.Tensor:
    """(n,) long: the first-kill team d (0 blue, 1 red, −1 an execution) of the rows of an M3Kills block under the shifted kill law: the field ``d`` of an open block, or
    the team of the mark ``m`` of a masked block (``killlaw.torch_first_team``). ValueError when the target has neither.

    (LOSS-1k)의 d_i를 구한다. 마스크 행은 표지에서(표지 m = 0 … 49의 팀 m // 25, 처형 50, 51은 −1), 열린 행은 행을 만들 때 적은 필드에서 얻는다."""
    if 'd' in target:
        return _index('d', _field('M3Kills', target, 'd'), n, -1, 1, like, what='the first-kill team d')
    if 'm' in target:
        return killlaw.torch_first_team(_index('m', _field('M3Kills', target, 'm'), n, 0, N_M - 1, like, what='the mark m'), check_range=not step_guard.trusted())
    raise ValueError("M3Kills under the shifted kill law needs the field 'm' (masked rows: the mark) or 'd' (open rows: the first-kill team) in its target")


def _shifted_kill_rows(mu: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """(LOSS-1k) for the rows: ``mu`` (n, 2) float64 rates, ``y`` (n, 2) float64 counts, ``d`` (n,) the first team; the row scores, +∞ for a first team without a kill.

    해석: 셀 점수는 ``killlaw.torch_logpmf``의 음수이고 `_poisson_cells`와 같은 연산이어서(처형 행은 비트까지 같다) 불가능한 칸만 +∞이다. 평균이 음수이거나 유한하지 않은 칸은 거부한다."""
    bad = ~(torch.isfinite(mu) & (mu >= 0))
    guards = step_guard.current()
    if guards is not None:                                                                  # a trusted step: the model check is deferred (one read for the step)
        guards.defer(bad.sum(), MEAN_MESSAGE)
    elif bool(bad.any()):
        raise ValueError(MEAN_MESSAGE.format(int(bad.sum())))
    return (-killlaw.torch_logpmf(y, mu, d, 'shifted', check_range=guards is None)).sum(-1)


def _category_rows(head: str, rows: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes (LOSS-2) for the rows: ``rows`` (n, C) the log laws that the rows read, ``y`` int64 categories (-1: none); the row scores, 0 for -1."""
    used = y >= 0
    guards = step_guard.current()
    if guards is not None:                                                                  # a trusted step: the two model checks are deferred, in their order, over the rows that are read
        law = rows.detach()
        guards.defer(((torch.isnan(law) | (law == float('inf'))).any(-1) & used).sum(), f'a log law of {head} holds NaN or +infinity in a row that is read')
        guards.defer((((law.double().exp().sum(-1) - 1.0).abs() > ROW_SUM_TOLERANCE) & used).sum(), f'a log law of {head} does not add up to 1 (to within {ROW_SUM_TOLERANCE:g})')
    elif bool(used.any()):
        law = rows[used]
        if bool((torch.isnan(law) | (law == float('inf'))).any()):
            raise ValueError(f'a log law of {head} holds NaN or +infinity in a row that is read')
        if bool(((law.double().exp().sum(-1) - 1.0).abs() > ROW_SUM_TOLERANCE).any()):
            raise ValueError(f'a log law of {head} does not add up to 1 (to within {ROW_SUM_TOLERANCE:g})')
    picked = rows.gather(1, y.clamp(min=0)[:, None])[:, 0].double()
    return torch.where(used, -picked, torch.zeros_like(picked)) + 0.0                                         # -0.0 (a certain outcome) becomes 0.0, as in ``nll``


def head_nll(head: str, out: torch.Tensor, target, exposure=None, *, kill_law: str = 'poisson') -> torch.Tensor:
    """The NLL of the rows of ``head``: (n,) float64, one value for each row, equal to the numpy score of ``nll.py`` on the same rows.

    ``head``: a name of ``head_rows.HEAD_NAMES``. ``out``: the output of the head for the n rows, a floating tensor of the shape in the table of the module text (one row
    of ``out`` for each row of the block: heads called with ``pos`` of the block). ``target``: the block fields (``data.block``) or the tensor y: ``y`` and the fields that pick
    the cell (``q``, ``m``, ``s``, ``team``, ``layer``; ``layer`` is 1 .. 6 and the output axis is ``layer - 1``). ``exposure``: M2Rate a number, (n,) or (n, 18) seconds; M4Conv a
    number or (n,) seconds (one conversion window for each row); None takes the field ``e`` of the target; M3Kills, M3Obj and the category heads take none. ValueError for an
    output or field of another shape, a category or index out of range, a count that is not a non-negative whole number or sits in a cell without exposure, an observed cell
    whose mean is negative or not finite, a log law that holds NaN or does not add up to 1, and an exposure that is not finite or is negative; TypeError for a category or index
    that is not an integer tensor.

    ``kill_law`` (S4b preparation M1, decision D5) is the kill law of the M3Kills rows and changes no other head: ``'poisson'`` (the default) is the score of every earlier run,
    bit for bit; ``'shifted'`` scores the count of the first-kill team d as 1 + N_d with N_d ~ Poisson(out[s, d]) and the other team as a plain Poisson count (``killlaw``). The
    first team comes from the field ``m`` (a masked block: ``d = m // 25``, −1 for an execution) or ``d`` (an open block of the shifted rows); a row whose first team has no
    kill has the probability 0 and the score +∞ (the rows refuse such a unit before they reach the loss), and ValueError if neither field is in the target.

    수식
    ----
    (LOSS-1)  ℓ_{i,c} = μ_{i,c} − y_{i,c} · log μ_{i,c} + log(y_{i,c}!)    (e_{i,c} > 0),    ℓ_{i,c} = 0    (e_{i,c} = 0),    ℓ_i = Σ_c ℓ_{i,c}
    (LOSS-2)  ℓ_i = −log p_{i,c_i}    (c_i ≥ 0),    ℓ_i = 0    (c_i = −1)
    (LOSS-1)은 ``nll``의 (NLL-1), (LOSS-2)는 (NLL-2)·(NLL-3)와 같은 점수이다. 포아송 머리의 μ는 머리가 낸 값 그대로이다 (노출을 다시 곱하지 않는다).
    (LOSS-1k) M3Kills, 이동 법칙:   ℓ_{i,t} = r_{i,t} − (y_{i,t} − 1[t = d_i]) · log r_{i,t} + log((y_{i,t} − 1[t = d_i])!),   r_{i,t} = out[i, s_i, t]
              (y_{i,d_i} = 0이면 ℓ_{i,d_i} = +∞) ;   d_i = m_i // 25 (m_i ≤ 49) 또는 −1 (m_i = 50, 51) ;   ℓ_i = Σ_t ℓ_{i,t}

      기호      | 뜻                                                  | 단위             | 범위                   | 코드 변수
      ℓ_i       | 행 i의 점수 (그 행의 칸 점수의 합)                  | nat              | ≥ 0 (확률 0이면 +∞)    | 반환값[i]
      ℓ_{i,c}   | 행 i의 칸 c의 점수                                  | nat              | ≥ 0                    | 칸 점수
      y_{i,c}   | 칸 c에서 센 개수                                    | 개               | 0, 1, …                | target['y']
      μ_{i,c}   | 칸 c의 기대 개수 λ·e (M3는 노출 1)                  | 개               | ≥ 0                    | out
      e_{i,c}   | 칸 c의 노출 (M4Conv는 행마다 한 값)                 | s                | ≥ 0                    | exposure
      c_i, p_{i,c} | 행 i의 실제 범주 번호 (−1은 범주 없음), 그 칸의 확률 | —            | −1, 0, …, C − 1; [0, 1] | target['y'], exp(읽은 로그 법칙)
      r_{i,t}, d_i | 이동 법칙에서 팀 t의 율(d 칸은 초과분 N_d의 평균 λ_d, 다른 칸은 킬 수의 평균 μ), 행 i의 첫 킬 팀 | 개, — | ≥ 0; −1, 0, 1 | out[i, s_i], killlaw.torch_first_team

    해석
    ----
    점수는 머리가 낸 법칙에서 정답이 나올 확률의 음의 로그이고, 로그 점수는 엄격한 적정 점수이므로 기대 손실은 머리가 참 조건부 법칙을 낼 때 가장 작다.
    모형과 무관한 log y!까지 넣어 B0·B1의 점수와 같은 수이다 (계획 P4). 노출이 0인 칸은 관측하지 않은 칸이므로 0이고 μ를 읽지 않으며 기울기도 없다 (계획 P10, 검토 초점 4).
    범주 −1(구역이 없는 첫 킬, 킬이 없는 팀의 골드)도 0이고 그 행의 법칙을 읽지 않는다. 관측한 칸에서 μ = 0인데 y > 0이거나 정답 범주의 확률이 0이면 참값 +∞를 그대로 돌려준다.
    이동 법칙 (LOSS-1k)는 첫 킬 팀의 칸에서 센 개수에서 1을 뺀 초과분에 포아송 점수를 매기고 다른 팀과 처형으로 시작한 교전은 옛 점수 그대로이다. 머리의 출력은 같은 모양이고
    d 칸의 값만 뜻이 λ_d로 바뀐다. 프레임 평균은 맞바꾼 틀의 출력을 같은 팀의 칸으로 되돌려 평균하므로(d도 같이 맞바뀐다) 이 점수가 읽는 칸이 틀 사이에서 어긋나지 않는다.

    근거
    ----
    계획 P4 (같은 확률 족), P10 (노출), Q12·검토 초점 4 (노출 0 칸), 판정 R-S4a-10 (μ = λ·e), R13 (섞은 법칙), 검산 노트 04 A절, Ref: Gneiting and Raftery 2007 (로그 점수는 적정 점수).
    저자 결정 D5 (이동 법칙 1 + N_d).
    """
    killlaw.check_law(kill_law)
    _check_head(head)
    target = _fields(target)
    n = _rows_of(head, target)
    if not isinstance(out, torch.Tensor) or not out.is_floating_point():
        raise ValueError(f'out of {head} must be a floating-point tensor')
    if tuple(out.shape[1:]) != _OUT_TAIL[head]:
        raise ValueError(f'out of {head} must have the shape (rows, {", ".join(map(str, _OUT_TAIL[head]))}), not {tuple(out.shape)}')
    if out.shape[0] != n:
        raise ValueError(f'out of {head} has {out.shape[0]} rows but the target has {n}')
    y = _field(head, target, 'y')
    picks = {name: _index(name, _field(head, target, name), n, low, high, out) for name, (low, high) in _INDEX.get(head, {}).items()}
    row = torch.arange(n, device=out.device)
    if head in _COUNT_WIDTH:
        y = _counts(head, y, n, out)
        if head == 'M3Kills':
            mu = out[row, picks['s']].double()
        else:
            mu = out.double()
        if head in ('M2Rate', 'M4Conv'):
            e = exposure if exposure is not None else target.get('e')
            if e is None:
                raise ValueError(f'{head} needs the exposure (an argument or the field e of the target)')
            e = _exposure(e, n, N_Q, out) if head == 'M2Rate' else _window(e, n, out)[:, None].expand(n, _COUNT_WIDTH[head])
            seen = e > 0
        else:
            if exposure is not None:
                raise ValueError(f'{head} has the exposure 1: it takes no exposure')
            seen = torch.ones_like(mu, dtype=torch.bool)
        if head == 'M3Kills' and kill_law == 'shifted':
            return _shifted_kill_rows(mu, y, _first_team(target, n, out))
        return _poisson_cells(mu, y, seen).sum(-1)
    if exposure is not None:
        raise ValueError(f'{head} is a category head: it has no exposure')
    if y.dtype not in _INT_DTYPES:
        raise TypeError(f'the category numbers y of {head} must be an integer tensor, not {y.dtype}')
    if tuple(y.shape) != (n,):
        raise ValueError(f'y of {head} must have the shape ({n},), one category for each row, not {tuple(y.shape)}')
    if n and not step_guard.trusted() and (int(y.min()) < -1 or int(y.max()) >= _CATEGORIES[head]):
        raise ValueError(f'the category numbers of {head} must lie in -1 .. {_CATEGORIES[head] - 1}: {int(((y < -1) | (y >= _CATEGORIES[head])).sum())} category number(s) '
                         'lie outside')
    y = y.to(device=out.device, dtype=torch.long)
    if head == 'M2Mark':
        law = out[row, picks['q']]
    elif head == 'M2Zone':
        law = out[row, picks['q'], picks['m']]
    elif head == 'M3Gold':
        law = out[row, picks['s'], picks['team'], picks['layer'] - 1]
    else:
        law = out
    return _category_rows(head, law, y)


def _scored(head: str, target: Mapping, exposure, n: int) -> torch.Tensor:
    """The bool mask of the rows of the mean of a block of ``n`` rows (``scored_rows`` for a block whose number of rows is known)."""
    if head in ('M3Kills', 'M3Obj'):
        y = target.get('y')
        return torch.ones(n, dtype=torch.bool, device=y.device if isinstance(y, torch.Tensor) else None)
    if head in _CATEGORIES:
        y = _field(head, target, 'y')
        if y.dtype not in _INT_DTYPES or tuple(y.shape) != (n,):
            raise ValueError(f'y of {head} must be an integer tensor of shape ({n},), one category for each row')
        return y >= 0
    e = exposure if exposure is not None else target.get('e')
    if e is None:
        raise ValueError(f'{head} needs the exposure (an argument or the field e of the target)')
    like = e if isinstance(e, torch.Tensor) else torch.zeros(())
    if head == 'M2Rate':
        return (_exposure(e, n, N_Q, like) > 0).any(-1)
    return _window(e, n, like) > 0


def scored_rows(head: str, target, exposure=None) -> torch.Tensor:
    """(n,) bool: the rows that count in the mean of ``head`` (the other rows add 0): a frame row with an observed bin (M2Rate), a conversion window above 0 (M4Conv), a
    category from 0 on (the category heads); the kills and the objectives are always observed. ``target`` and ``exposure`` as for ``head_nll``.

    해석: 평균에 드는 행은 관측한 칸이나 범주가 있어 점수가 0이 아닐 수 있는 행이다. B0가 점수에서 빼는 행(구역 −1, 노출 없음)과 같은 행이라 EPT의 평균과 B0의 평균이 같은 행 위에서 정의된다 (판정 R22).
    """
    _check_head(head)
    target = _fields(target)
    return _scored(head, target, exposure, _rows_of(head, target))


# ---- the mean of a head and the sum of the heads -------------------------------------------------------------------------------------------------
def plan_key(head: str, kind: str, name: str) -> str:
    """The key of the list ``name`` (one of ``PLAN_NAMES``) of the block ``(head, kind)`` in a prepared batch: ``'plan/<head>/<kind>/<name>'``."""
    return f'{PLAN_PREFIX}{block_key(head, kind)}/{name}'


def counting_plan(head: str, kind: str, target) -> dict:
    """``{plan_key: int64 (m,) tensor}``, the index lists of one block (S4b speed-ups item 3, D32): ``'counted'`` = the rows that count in the mean of the block, ascending (``nonzero`` of ``scored_rows``), and for
    the three size-normalised experts ``'size0'`` .. ``'size3'`` = the counted rows of each size, ascending. ``head_losses`` of a trusted step averages ``rows.index_select(0, list)`` instead of the boolean mask
    ``rows[mask]`` (``batch_prep``, (SPD-3)).

    해석: 마스크로 행을 고르려면 장치가 몇 행인지 알려 줘야(``nonzero``) 해서 걸음이 멈춘다. 같은 행의 번호를 CPU에서 한 번 구해 두면 평균은 같은 원소를 같은 순서로 모은 연속 텐서의 평균이어서 값이 같다."""
    target = _fields(target)
    counted = scored_rows(head, target)
    plan = {plan_key(head, kind, 'counted'): counted.nonzero().reshape(-1)}
    if head in SIZE_NORMALISED:
        size = _field(head, target, 's')
        for k in range(N_SIZE):
            plan[plan_key(head, kind, f'size{k}')] = (counted & (size == k)).nonzero().reshape(-1)
    return plan


def _plan_of(key: str, target: Mapping):
    """``{name: list}`` of the lists of the block ``key`` that its target carries (``train.score_blocks`` copies them from the prepared batch), or None when it carries none (the step then uses the masks)."""
    head, kind = split_key(key)
    names = PLAN_NAMES if head in SIZE_NORMALISED else PLAN_NAMES[:1]
    found = {name: target.get(plan_key(head, kind, name)) for name in names}
    return found if all(v is not None for v in found.values()) else None


def head_losses(nlls: Mapping, targets: Mapping) -> dict:
    """``{block: the mean ℓ̄_head of the block}`` (0-d float64 tensors) of the row scores ``nlls`` = ``{block: (n,) scores of head_nll}``; ``targets`` = ``{block: its block fields}``
    (the field ``s`` for the three experts, ``y`` for a category head, ``e`` for M2Rate and M4Conv). A block without a row that counts is left out of the result. ValueError for
    a key that is not a block, a missing target or field, scores and rows of another shape, and a size outside 0 .. 3.

    수식
    ----
    (LOSS-3)  ℓ̄_head = (1 / N) Σ_{i ∈ 점수가 있는 행} ℓ_i ;    M3Kills·M3Gold·M3Obj:  ℓ̄_head = (1 / n_size) Σ_{s ∈ 나온 규모} ℓ̄_{head,s},    ℓ̄_{head,s} = (1 / N_s) Σ_{i: s_i = s} ℓ_i

      기호          | 뜻                                              | 단위 | 범위       | 코드 변수
      ℓ_i           | 행 i의 점수 (``head_nll``의 반환값)             | nat  | ≥ 0        | nlls[block][i]
      N, N_s        | 점수가 있는 행의 수, 그 가운데 규모 s의 행의 수 | 행   | ≥ 1       | scored_rows의 합, 규모별 합
      s_i, n_size   | 행 i의 규모 (T, S, ASYM, P = 0..3), 배치에 나온 규모의 수 | — | 0..3; 1..4 | targets[block]['s']
      ℓ̄_head, ℓ̄_{head,s} | 머리의 평균, 규모 s 행의 평균            | nat  | ≥ 0        | 반환값, 규모별 평균

    해석
    ----
    한 머리의 손실은 점수가 있는 행의 평균이고 행 가중은 모두 같다 (계획 P9). 조건부 전문가 셋은 규모마다 행 평균을 낸 뒤 나온 규모끼리 평균한다.
    규모 P가 교전의 70.9%이고 T가 6.4%이므로 행 평균이면 P가 전문가의 손실을 지배한다 (설계서 2절 FIT 점검, 3.2절). 규모 관문은 모듈 머리글의 이유로 행 평균이다.
    배치에 행이 없는 규모는 평균에서 빠지고, 점수가 있는 행이 하나도 없는 블록은 결과에 없다.

    근거
    ----
    설계서 7절·3.2절 (M3는 규모별 정규화), 계획 Q8·P9, 판정 R22 (점수 없는 행), 검산 노트 04 B절.
    """
    out = {}
    for key in nlls:
        head, _ = split_key(key)
        if key not in targets:
            raise ValueError(f'no target for the block {key}')
        rows = nlls[key]
        if not isinstance(rows, torch.Tensor) or rows.ndim != 1:
            raise ValueError(f'the scores of the block {key} must be a tensor of shape (rows,)')
        target = _fields(targets[key])
        plan = _plan_of(key, target) if step_guard.trusted() else None
        if plan is not None:                                                                  # a trusted step: the lists of the rows that count, made once on the CPU (SPD-3)
            if int(target['y'].shape[0]) != int(rows.shape[0]):
                raise ValueError(f'the block {key} has {rows.shape[0]} scores but its target has {int(target["y"].shape[0])} rows (shape)')
            rows = rows.to(torch.float64)
            if head in SIZE_NORMALISED:
                means = [rows.index_select(0, plan[f'size{k}']).mean() for k in range(N_SIZE) if plan[f'size{k}'].numel()]
                if means:
                    out[key] = torch.stack(means).mean()
            elif plan['counted'].numel():
                out[key] = rows.index_select(0, plan['counted']).mean()
            continue
        counted = _scored(head, target, None, int(rows.shape[0]))
        if counted.shape != rows.shape:
            raise ValueError(f'the block {key} has {rows.shape[0]} scores but its target has {counted.shape[0]} rows (shape)')
        counted = counted.to(rows.device)
        if not bool(counted.any()):
            continue
        rows = rows.to(torch.float64)
        if head in SIZE_NORMALISED:
            size = _index('s', _field(head, target, 's'), rows.shape[0], 0, N_SIZE - 1, rows, what='the size class s')
            means = [rows[counted & (size == k)].mean() for k in range(N_SIZE) if bool((counted & (size == k)).any())]
            out[key] = torch.stack(means).mean()
        else:
            out[key] = rows[counted].mean()
    return out


def total_loss(nlls: Mapping, scales: Mapping, targets: Mapping) -> torch.Tensor:
    """The loss of the batch: a 0-d float64 tensor, the sum over the blocks of the head mean divided by the B0 scale of the block. ``nlls`` and ``targets`` as for
    ``head_losses``; ``scales``: ``{block: B0 scale}`` (``scales.b0_scales``), a positive finite number for every block of ``nlls``. The blocks are added in the order of
    ``BLOCKS``. ValueError as for ``head_losses``, for a scale that is missing or not positive and finite, and when no block has a row that counts.

    수식
    ----
    (LOSS-4)  Loss_EPT = Σ_head ℓ̄_head / ℓ̄^B0_head

      기호           | 뜻                                              | 단위 | 범위   | 코드 변수
      ℓ̄_head         | 블록의 평균 점수 (LOSS-3)                       | nat  | ≥ 0    | head_losses의 값
      ℓ̄^B0_head      | 같은 블록의 B0 평균 NLL, FIT 행에서 한 번 잰 상수 | nat  | > 0    | scales[block]
      Loss_EPT       | 한 배치의 손실 (B0 수준이면 블록 수 13)         | —    | ≥ 0    | 반환값

    해석
    ----
    머리마다 점수의 단위가 달라(M2 한 행은 18칸의 합이고 범주 머리는 몇 nat이다) 그냥 더하면 큰 머리가 기울기를 지배한다. B0 평균으로 나눈 비는 단위가 없고 B0 수준이면 1이므로, 상대 개선이 같으면 머리가 달라도 손실을 같은 만큼 줄인다.
    가중 합은 결합 우도가 아니다. 머리의 출력층은 따로이지만 일부 매개변수는 공유한다. 손실 가중은 공유 인코더, 조건 임베딩, M3 공유 은닉층에서 각 목표의 상대적인 학습 기여를 바꾼다. 상수는 FIT에서만 한 번 재어 고정하므로 움직이는 목표도, 다른 역할의 정보도 아니다 (계획 Q8).

    근거
    ----
    설계서 7절 (머리별 NLL ÷ B0 NLL, 같은 가중), 계획 Q8, 검산 노트 04 C절.
    """
    for key in nlls:
        split_key(key)
        if key not in scales:
            raise ValueError(f'no B0 scale for the block {key}')
        try:
            scale = float(scales[key])
        except (TypeError, ValueError):
            scale = float('nan')
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError(f'the B0 scale of the block {key} must be finite and positive, not {scales[key]!r}')
    means = head_losses(nlls, targets)
    if not means:
        raise ValueError('no block has a row that counts: there is nothing to average')
    total = None
    for key in BLOCKS:                                                                                         # a fixed order: the same bits whatever the order of the mappings
        if key in means:
            term = means[key] / float(scales[key])
            total = term if total is None else total + term
    return total
