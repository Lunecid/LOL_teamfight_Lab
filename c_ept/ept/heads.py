"""EPT의 머리 (S4a 과제 3): 부호화한 보기에서 M2(발생)·M3(결과)·M4(전환)의 율과 법칙을 내고, 팀을 맞바꾼 두 틀의 출력을 평균한다.

이 모듈은 설계서 3.1–3.3절(과정 분해: λ_q × π × ρ, 규모 관문 × 전문가, 전환), 4절(한 망이 두 읽기를 맡는다), 5.2절·6.1절(팀 교환 입력과 원래 입력의 프레임 평균),
계획 Q7(머리의 꼴)·Q12(범위 문구)와 판정 R13(모든 범주 출력에 균등 분포를 섞는다)을 구현한다. 이 모듈은 킬이 하나 이상 난 교전을 다룬다 (G-S3c C3).
"가린 읽기"는 t₀ 이전 정보만 가진 프레임 시계 보기에 조건 (q, m, z, …)을 주는 것이고, "열린 읽기"는 교전 중 사건 시계 보기에 q, m, z 없이 묻는 것이다
(셋을 모두 None으로 준다). 두 읽기는 가중을 공유한다.
방법. 머리의 입력은 부호화기의 요약 토큰 상태 h와 선수 토큰 열 개이다. 구간 q, 구역 z, 규모 s는 작은 임베딩 Γ_q, Γ_z, Γ_s로 바꾸고, 표지 m은 그 표지가 가리키는 킬러와
피해자의 선수 토큰 상태에서 임베딩 Γ_m을 만들어 (EH-2) 슬롯 번호를 입력으로 쓰지 않는다. 머리마다 은닉층 하나가 입력 사상의 합 위에 GELU를 씌운다. 포아송 머리는 로그 연결로
양의 평균을 내고 (EH-1, EH-6–EH-8), 범주 머리는 소프트맥스에 균등 분포를 섞는다 (EH-4). 프레임 평균 (EH-9)이 정확한 팀 대칭을 낸다.

출력 (머리 이름은 ``head_rows.HEAD_NAMES``; 행 n개, 행마다 보기 하나; ``HEAD_OUTPUT``이 이름을 방법과 필드로 옮긴다)

    M2Rate   m2      rate (n, 18)              구간 q의 기대 첫 킬 수 μ_q = λ_q Δ_q (노출 Δ_q = 1이면 초당 율 λ_q)
    M2Mark   m2      mark_logp (n, 18, 52)     log π(m | q)
    M2Zone   m2      zone_logp (n, 18, 52, 10) log ρ(z | q, m)
    M3Size   m3      size_logp (n, 4)          log P(s | q, m, z)
    M3Kills  m3      kill_rate (n, 4, 2)       규모 s, 팀 t (파랑, 빨강)의 킬 수 평균 μ_{s,t}
    M3Gold   m3      gold_logp (n, 4, 2, 6, 31) log p(b | s, t, ξ): 규모, 팀, 그 팀의 킬 층 ξ − 1, 골드 구간
    M3Obj    m3_obj  obj_rate (n, 18)          오브젝트 칸 c의 기대 개수 μ_c
    M4Conv   m4      conv_rate (n, 18)         전환 칸 c의 기대 개수 μ_c = e · λ_c (전환 창 e: 행마다 한 값, 초; e = 1이면 초당 율 λ_c)

식 (식 번호는 ``(EH-n)``; 기호 표는 식을 계산하는 함수의 문서 문자열에 있다)

    (EH-1)  η_q = Θ GELU(A^h h + A^q Γ_q) ;  λ_q = exp(η_q) ;  μ_q = λ_q Δ_q
    (EH-2)  Γ_m = A^kil h_m^kil + A^vic h_m^vic + E_kind[쌍] (m ≤ 49) ;  Γ_m = A^tm h̄_{t(m)} + E_kind[처형] (m = 50, 51) ;  Γ_m = E_kind[없음] (열린 읽기)
    (EH-3)  η_{q,m} = Θ GELU(A^m Γ_m + A^q Γ_q) ;  π(m | q)는 η_{q,·}의 소프트맥스에 (EH-4)를 쓴 것
    (EH-4)  p_c = (1 − ν) p_model,c + ν / C,  ν = 1e-6  (판정 R13)
    (EH-5)  ρ(z | q, m)은 Θ GELU(A^m Γ_m + A^q Γ_q + A^h h)의 소프트맥스에 (EH-4)를 쓴 것
    (EH-6)  ζ = GELU(A^h h + A^q Γ_q + A^m Γ_m + A^z Γ_z) ;  P(s | q, m, z)는 Θ^size ζ의 소프트맥스에 (EH-4) ;  μ_{s,t} = exp((Θ^kill ζ)_{s,t}) ;
            p(b | s, t, ξ)는 Θ^gold GELU(A^g ζ + E_slot[s, t, ξ])의 소프트맥스에 (EH-4)
    (EH-7)  μ_c = exp(Θ GELU(A^h h + A^q Γ_q + A^m Γ_m + A^z Γ_z + A^s Γ_s + A^K log(1 + K)))_c
    (EH-8)  μ_c = e · λ_c ;  λ_c = exp(Θ GELU(A^h h + A^q Γ_q + A^m Γ_m + A^z Γ_z + A^K log(1 + K) + A^O log(1 + O)))_c
    (EH-9)  ȳ(b) = ½ [ŷ(b) + Π ŷ(swap b)] (율과 확률의 산술평균) ;  log p̄ = logaddexp(log p(b), log Π p(swap b)) − log 2
    (EH-10) ȳ(b) = ŷ(F_c(b)) (한 틀: 보기마다 동전 c로 고른 틀에서 한 번, 평균도 Π도 없다; S4b 준비 M7, 결정 D10 P1·D17)

가정과 선택

    · M2의 rate와 M4의 conv_rate는 평균 μ = λ × 노출을 낸다: 노출이 0인 칸은 정확히 0이고 그 칸으로 기울기가 가지 않는다 (계획 Q7 "노출 포함", 검토 초점 4). 합성이 묻는
      초당 율은 노출 1로 얻는다. M3의 kill_rate와 obj_rate는 노출이 1인 평균이다.
    · 크기는 부호화기의 d를 따른다: 은닉 폭은 d, (q, m)의 모든 조합에 쓰는 은닉 폭은 max(8, d / 4), 임베딩 폭은 32이다. 층 구조와 GELU는 표준 선택이다. 임베딩과 최종 층의 가중은
      N(0, 0.02²)에서, 은닉층의 입력 사상은 PyTorch 기본값에서 시작하고 편향은 0에서 시작한다 (초기 율은 1이다; 자료의 평균으로 시작값을 정하는 일은 이 모듈이 하지 않는다).
    · 팀 대칭은 머리와 부호화기가 아니라 프레임 평균이 낸다 (과제 2). 정확한 불변은 드롭아웃을 끈 계산(평가 모드)에서 성립한다.
    · 조건은 입력 인자이고 머리는 그 밖의 것을 읽지 않는다 (가린 M3는 O를 받지 않는다: 계획 P4, O는 K와 규모의 조건부 결과이다; 가린 M4는 s를 받지 않는다: 설계서 3.3절
      누설 시험 (a)). K와 O는 연속 배치로 바꾸어 읽으므로 출력은 그 텐서의 값에만 달려 있고 걸음(stride)에는 달려 있지 않다.

입력과 출력

    EPT(cfg, static).encode(batch) → Encoded        EPTHeads(cfg).m2 / m3 / m3_obj / m4(enc, 조건, pos=None) → 위 표
    FrameAveraged(model).encode(batch) → FrameEncoded    FrameAveraged(model).m2 / m3 / m3_obj / m4(enc, 조건, pos=None) → 위 표, 두 틀의 평균
    FrameAveraged(model, average=False): 한 틀 (S4b 준비 M7, 결정 D10 P1·D17; 학습의 swap_mode 'random_view'·'none'과 eval_average=False의 평가): 부호화기 한 번, 머리의 답을 평균 없이 그대로 (EH-10)


m2_rate는 학습·평가 모두 발생률 분기만 계산한다. 건너뛴 표지·구역 분기는 rate의 값에도 손실 그래프에도 들어가지 않고 EPTHeads에 난수 연산이 없으므로, 학습 모드에서도 값과 기울기는 m2의 rate와 비트 단위로 같다
(EPT-T011 A2-F10; CPU에서 시험했으며 GPU는 시험하지 않았다: GPU에서는 결정적 알고리즘 설정(train.py의 use_deterministic_algorithms) 아래에서 같은 논리가 성립할 것으로 기대한다).
기존 m2의 출력 계약과 수식은 그대로다.

입력 점검과 믿는 걸음 (S4b 속도 개선 SPD 항목 3, 결정 D32): 아래 ``_whole``·``_counts``·``_exposure``·``_window``·``_view_index``의 값 읽기(범위, 유한성, 부호: ``int(t.min())``, ``bool(...any())``)는 장치에서 읽을 때마다
대기열이 빌 때까지 기다린다. 걸음이 ``step_guard.trust(...)`` 안에서 돌면(배치가 CPU에서 ``batch_prep.Preparation``으로 같은 함수로 같은 점검을 이미 통과했다) 이 읽기만 건너뛴다. 형·모양·dtype 점검은
읽기가 없어 그대로 일어나고, 믿는 범위 밖에서는 모든 점검이 이전과 같다. 값은 달라지지 않는다(점검은 값을 만들지 않고 막기만 한다).

참조 경로 확인(2026-10-24): 위에서 인용한 역사적 설계서·S4a 계획·Task 5 계약서 중 해당 경로는 이 checkout에 미동봉되어 있다. 인용은 과거 설계 출처이며 이번 검토에서 그 원문까지 확인했다는 뜻이 아니다. docs/superpowers/notes/equation-index.md는 이 checkout에 존재하지만, 파일 존재만으로 모든 수식의 정확성을 인증하지 않는다."""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md sections 3.1-3.3 (the factorisation of the process and its heads), 4 (one network for the two
#      readings), 5.2 and 6.1 (team swap: the outputs change places with the inputs; frame average) and 10 (law A1).
# Ref: docs/superpowers/plans/2026-10-22-c-ept-s4a.md decisions Q7 (the form of the heads), Q9, Q12 and review focus 1 and 4; docs/superpowers/plans/
#      2026-10-21-c-ept-s3.md decisions P4 (the factorisation and the probability family), P6 (what the masked heads condition on) and P10 (exposure);
#      controller rulings R13 (every categorical output is mixed with the uniform law) and R-S4a-6 (the swap is an exact involution).
# Ref: Hendrycks, D. and Gimpel, K. (2016), "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415 (the activation of the hidden layers).
# Ref: Puny, O., Atzmon, M., Ben-Hamu, H., Misra, I., Grover, A., Smith, E. J., Lipman, Y. (2022), "Frame Averaging for Invariant and Equivariant Network
#      Design", arXiv:2110.03336 (averaging over the elements of a small frame, here the two team frames, makes a network exactly invariant or equivariant;
#      abstract read on the arXiv API, 2026-10-23).
# Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press (the Poisson mean with a
#      log link and an exposure), as in ``nll``.
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from c_ept.preprocessing import terrain
from c_ept.domain import units
from c_ept.ept import data, step_guard
from c_ept.ept.encoder import Encoded, EncoderConfig, EPTEncoder, token_layout
from c_ept.data.head_rows import HEAD_NAMES, MIX, N_CELLS, N_GOLD, N_K, N_LAYER, N_M, N_Q, N_SIZE, N_ZONE

N_PAIR_MARKS = units.EXEC_MARK0                      # marks 0-49 are pairs (killer team, killer slot, victim slot); 50 and 51 are executions
N_TEAMS, N_SLOTS = 2, N_SIZE * 2 * N_LAYER           # the teams of K and O; the (size, team, kill layer) places of the gold law
D_EMB = 32                                           # the width of the embeddings of the bin, the zone and the size
_INIT_STD = 0.02                                     # the standard deviation of the starting values of the embeddings and of the output layers
_PAIR, _EXEC, _ABSENT = range(3)                     # the kinds of mark embedding (EH-2)
_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)         # not uint8: an index tensor of that type is a mask (``data._mark`` would read it so)
if (N_Q, N_M, N_PAIR_MARKS, N_ZONE, N_SIZE, N_LAYER, N_GOLD, N_CELLS, N_K) != (18, 52, 50, len(terrain.ZONES), 4, 6, 31, 18, 4) or N_ZONE != 10:
    raise RuntimeError('the widths of the heads (bins, marks, zones, sizes, kill layers, gold bins, cells, K) are not those of this module')

HEAD_OUTPUT = MappingProxyType({'M2Rate': ('m2', 'rate'), 'M2Mark': ('m2', 'mark_logp'), 'M2Zone': ('m2', 'zone_logp'),
                                'M3Size': ('m3', 'size_logp'), 'M3Kills': ('m3', 'kill_rate'), 'M3Gold': ('m3', 'gold_logp'),
                                'M3Obj': ('m3_obj', None), 'M4Conv': ('m4', None)})
if set(HEAD_OUTPUT) != set(HEAD_NAMES):
    raise RuntimeError('HEAD_OUTPUT must name every head of head_rows and no other')


def _mark_rows() -> tuple:
    """The player rows of the killer and of the victim of each of the 50 pair marks: mark = (d 5 + i) 5 + j (``units.first_kill_mark``), the rows of
    team d are 5 d .. 5 d + 4 in the views."""
    k = units.N_SLOTS
    m = np.arange(N_PAIR_MARKS)
    team, killer, victim = m // (k * k), (m // k) % k, m % k
    return k * team + killer, k * (1 - team) + victim


MARK_KILLER_ROW, MARK_VICTIM_ROW = _mark_rows()
# the tables of ``data`` (each is its own inverse) that move an axis of an output from one frame to the other (index i there = index PERM[i] here), as tensors
_MARK, _ZONE, _TEAM, _CELL = (torch.as_tensor(np.asarray(p, dtype=np.int64)).clone() for p in (data.MARK_PERM, data.ZONE_PERM, data.TEAM_PERM, data.CELL_PERM))


class M2Out(NamedTuple):
    """The outputs of ``EPTHeads.m2``: ``rate`` (n, 18), ``mark_logp`` (n, 18, 52), ``zone_logp`` (n, 18, 52, 10)."""
    rate: torch.Tensor
    mark_logp: torch.Tensor
    zone_logp: torch.Tensor


class M3Out(NamedTuple):
    """The outputs of ``EPTHeads.m3``: ``size_logp`` (n, 4), ``kill_rate`` (n, 4, 2), ``gold_logp`` (n, 4, 2, 6, 31)."""
    size_logp: torch.Tensor
    kill_rate: torch.Tensor
    gold_logp: torch.Tensor


# ---- the categorical laws -----------------------------------------------------------------------------------------------------------------
def mixed_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """The natural log of the law (1 - 1e-6) softmax(``logits``) + 1e-6 / C along ``dim`` (C = the length of that axis), in the dtype of ``logits``.
    Finite for every finite ``logits``, also when the softmax is one-hot.

    수식
    ----
    (EH-4)  p_c = (1 − ν) · p_model,c + ν / C,   p_model = softmax(η) ;   반환값 = log p_c = logaddexp(log(1 − ν) + log p_model,c, log(ν / C))

      기호        | 뜻                                              | 단위 | 범위      | 코드 변수
      η           | 최종 층의 출력 (소프트맥스 앞의 점수)           | —    | 실수      | logits
      p_model,c   | 섞기 전 범주 c의 확률                           | —    | [0, 1]    | log_softmax(logits)
      ν           | 균등 분포를 섞는 비율 (판정 R13)                | —    | 1e-6      | MIX
      C           | 범주의 수                                       | 개   | ≥ 1       | logits.shape[dim]
      p_c         | 섞은 뒤 범주 c의 확률                           | —    | [ν/C, 1]  | exp(반환값)

    해석
    ----
    범주 머리의 출력은 전체 범주 집합 위의 확률이고 모든 범주가 ν/C 이상이어서, 한 번도 나오지 않은 범주가 나와도 NLL이 유한하다. 섞은 뒤에도 합은 1이다.
    로그 공간에서 더하므로 소프트맥스가 한 범주에 몰려 다른 확률이 부동소수 밖으로 내려가도 −∞나 NaN이 되지 않는다. B0·B1이 같은 섞기를 쓰므로 같은 확률 족으로 NLL을 견준다.

    근거
    ----
    판정 R13, 계획 P4(같은 확률 족), 식 색인의 (B0-6)·(B1-4)와 같은 꼴.
    """
    floor = logits.new_full((), math.log(MIX / logits.shape[dim]))
    return torch.logaddexp(F.log_softmax(logits, dim) + math.log1p(-MIX), floor)


# ---- checks of the inputs ---------------------------------------------------------------------------------------------------------------------
def _whole(name: str, t, n: int, low: int, high: int, like: torch.Tensor) -> torch.Tensor:
    """``t`` as an int64 tensor on the device of ``like``, after the check that it is an integer tensor of shape (n,) with values in low .. high."""
    if not isinstance(t, torch.Tensor) or t.dtype not in _INT_DTYPES or tuple(t.shape) != (n,):
        raise ValueError(f'{name} must be an integer tensor (not uint8) of shape ({n},): one number for each row')
    if n and not step_guard.trusted() and (int(t.min()) < low or int(t.max()) > high):        # a trusted step: the CPU made this check (`batch_prep`)
        raise ValueError(f'{name} must lie in {low} .. {high}')
    return t.to(device=like.device, dtype=torch.long)


def _counts(name: str, t, n: int, width: int, like: torch.Tensor) -> torch.Tensor:
    """``t`` (n, width) in the dtype and on the device of ``like``, in the contiguous layout (so that the output depends on the values only, not on the
    strides of the caller's tensor: a matrix product and a vectorised log round by layout), after the check that it is finite and not negative."""
    if not isinstance(t, torch.Tensor) or tuple(t.shape) != (n, width):
        raise ValueError(f'{name} must be a tensor of shape ({n}, {width})')
    t = t.to(device=like.device, dtype=like.dtype).contiguous()
    if not step_guard.trusted() and (not bool(torch.isfinite(t).all()) or bool((t < 0).any())):
        raise ValueError(f'{name} must be finite and not negative')
    return t


def _exposure(e, n: int, width: int, like: torch.Tensor) -> torch.Tensor:
    """The exposure of the bins of M2 as a tensor that broadcasts against (n, width): a number, one value for each row (n,) or one for each bin
    (n, width), seconds, finite and not negative, in the dtype and on the device of ``like``."""
    e = torch.as_tensor(e, dtype=like.dtype, device=like.device)
    if e.ndim == 0:
        e = e.expand(n, 1)
    elif e.ndim == 1 and e.shape[0] == n:
        e = e[:, None]
    elif tuple(e.shape) != (n, width):
        raise ValueError(f'the exposure must be a number, a tensor of shape ({n},) or one of shape ({n}, {width}), got {tuple(e.shape)}')
    if not step_guard.trusted() and (not bool(torch.isfinite(e).all()) or bool((e < 0).any())):
        raise ValueError('the exposure must be finite and not negative')
    return e


def _window(e, n: int, like: torch.Tensor) -> torch.Tensor:
    """The conversion window of each row of M4 (``e`` of (EH-8)): a number or a tensor (n,) with one value for each row, seconds, finite and not
    negative, as an (n,) tensor in the dtype and on the device of ``like``. A value for each cell is refused (ruling R-S4a-11)."""
    e = torch.as_tensor(e, dtype=like.dtype, device=like.device)
    if e.ndim == 0:
        e = e.expand(n)
    elif tuple(e.shape) != (n,):
        raise ValueError(f'the exposure of M4 is the conversion window: a number or a tensor of shape ({n},), one value for each row, '
                         f'not shape {tuple(e.shape)} (ruling R-S4a-11)')
    if not step_guard.trusted() and (not bool(torch.isfinite(e).all()) or bool((e < 0).any())):
        raise ValueError('the exposure must be finite and not negative')
    return e


def _view_index(pos, views: int, like: torch.Tensor):
    """``pos`` as an int64 tensor on the device of ``like`` after the check that it is a 1-d integer tensor (not uint8) of view numbers in 0 .. views - 1;
    None if there is no ``pos``."""
    if pos is None:
        return None
    if not isinstance(pos, torch.Tensor) or pos.ndim != 1 or pos.dtype not in _INT_DTYPES:
        raise ValueError('pos must be a 1-d integer tensor (not uint8) of view numbers')
    if pos.numel() and not step_guard.trusted() and (int(pos.min()) < 0 or int(pos.max()) >= views):
        raise ValueError(f'pos must lie in 0 .. {views - 1}')
    return pos.to(device=like.device, dtype=torch.long)


def _row_count(enc: Encoded, pos) -> int:
    """The number of rows: the views of ``enc``, or the length of ``pos`` (checked)."""
    index = _view_index(pos, enc.h_pool.shape[0], enc.h_pool)
    return enc.h_pool.shape[0] if index is None else index.shape[0]


def _reading(n: int, like: torch.Tensor, q, m, z) -> tuple:
    """``(q, m, z)`` of the masked reading, checked and cast to int64, or ``(None, None, None)`` of the open one; ValueError if only some are given."""
    given = [x is not None for x in (q, m, z)]
    if any(given) and not all(given):
        raise ValueError('q, m and z are given together (masked reading) or all three are None (open reading)')
    if not all(given):
        return None, None, None
    return _whole('q', q, n, 0, N_Q - 1, like), _whole('m', m, n, 0, N_M - 1, like), _whole('z', z, n, -1, N_ZONE - 1, like)


def _read(enc: Encoded, pos) -> tuple:
    """The pooled state h (n, d), the player states (n, 10, d) and the mean state of each team (n, 2, d) of the rows: the views of ``enc`` in
    order, or the views ``pos`` (1-d integer tensor, a view may appear twice)."""
    players, pool = enc.h_tok[:, token_layout(0).players], enc.h_pool
    index = _view_index(pos, pool.shape[0], pool)
    if index is not None:
        players, pool = players.index_select(0, index), pool.index_select(0, index)
    n, d = pool.shape
    return pool, players, players.reshape(n, N_TEAMS, units.N_SLOTS, d).mean(2)


# ---- the parts of the heads -------------------------------------------------------------------------------------------------------------------
class _Mlp(nn.Module):
    """The hidden layer of (EH-1) written as a sum of maps (``project``) so that inputs of different shapes broadcast, then the bias, GELU (``act``) and,
    if there is one, the output layer Θ (``finish``)."""

    def __init__(self, widths: Sequence[int], hidden: int, out: int | None):
        super().__init__()
        self.maps = nn.ModuleList([nn.Linear(w, hidden, bias=False) for w in widths])
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.out = None if out is None else _output_layer(hidden, out)

    def project(self, k: int, x: torch.Tensor) -> torch.Tensor:
        return self.maps[k](x)

    def act(self, summed: torch.Tensor) -> torch.Tensor:
        return F.gelu(summed + self.bias)

    def finish(self, summed: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(summed))


def _output_layer(width: int, out: int) -> nn.Linear:
    """An output layer Θ: small random weights (N(0, 0.02²)) and a zero bias."""
    layer = nn.Linear(width, out)
    nn.init.normal_(layer.weight, std=_INIT_STD)
    nn.init.zeros_(layer.bias)
    return layer


class _MarkEmbedding(nn.Module):
    """The mark embedding Γ_m of (EH-2): ``forward(players, team_mean, m)`` one for each row (n, d), ``every(...)`` for all 52 marks (n, 52, d),
    ``absent(n)`` the value of the open reading."""

    def __init__(self, d: int):
        super().__init__()
        self.killer, self.victim, self.team = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.kind = nn.Embedding(3, d)
        nn.init.normal_(self.kind.weight, std=_INIT_STD)
        self.register_buffer('killer_row', torch.as_tensor(MARK_KILLER_ROW).clone(), persistent=False)
        self.register_buffer('victim_row', torch.as_tensor(MARK_VICTIM_ROW).clone(), persistent=False)

    def forward(self, players: torch.Tensor, team_mean: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(m.shape[0], device=m.device)
        pair_m = m.clamp(max=N_PAIR_MARKS - 1)                           # an execution has no pair: its row is computed and not used
        pair = (self.killer(players[rows, self.killer_row[pair_m]]) + self.victim(players[rows, self.victim_row[pair_m]])
                + self.kind.weight[_PAIR])
        victim_team = (m - N_PAIR_MARKS).clamp(0, N_TEAMS - 1)           # a pair has no team of the victim in this sense: likewise
        execution = self.team(team_mean[rows, victim_team]) + self.kind.weight[_EXEC]
        return torch.where((m < N_PAIR_MARKS)[:, None], pair, execution)

    def every(self, players: torch.Tensor, team_mean: torch.Tensor) -> torch.Tensor:
        pair = (self.killer(players.index_select(1, self.killer_row)) + self.victim(players.index_select(1, self.victim_row))
                + self.kind.weight[_PAIR])
        return torch.cat([pair, self.team(team_mean) + self.kind.weight[_EXEC]], dim=1)

    def absent(self, n: int) -> torch.Tensor:
        return self.kind.weight[_ABSENT].expand(n, -1)


# ---- the heads --------------------------------------------------------------------------------------------------------------------------------
class EPTHeads(nn.Module):
    """The heads M2, M3 and M4 of the EPT on the output of the encoder (``cfg`` is the ``EncoderConfig`` of that encoder; only ``d`` is read).

    Each method takes the ``Encoded`` of a batch of views and the conditions of the rows: ``enc`` has one view for each row, or ``pos`` (1-d integer
    tensor of view numbers) lists the view of each row (a view may serve several rows: the onsets of one frame row). ``q`` (n,) 0 .. 17 is the bin,
    ``m`` (n,) 0 .. 51 the mark, ``z`` (n,) -1 .. 9 the zone (-1: no position), ``s`` (n,) 0 .. 3 the size (integer tensors: uint8, which indexes as a mask,
    and bool are refused), ``K`` (n, 4) the kills of the blue and the red team and their kill gold in thousands, ``O`` (n, 18) the objectives (all not
    negative and finite); ``q``, ``m`` and ``z`` are given together (masked reading) or all three as None (open reading). ValueError for any condition
    that is of another shape or type, out of range or given in part. The outputs are in ``M2Out``, ``M3Out`` and the tensors of ``m3_obj`` and ``m4``.

    Parts (for tests and for starting values): ``bin_emb`` (the 18 bins and "none"), ``zone_emb`` (the zones -1 .. 9 and "none"), ``size_emb``, ``mark``
    (``_MarkEmbedding``), ``rate_net``, ``mark_net``, ``zone_net``, ``m3_net`` with ``size_out``, ``kill_out`` and ``gold_in``, ``gold_slot``,
    ``gold_out``, ``obj_net`` and ``conv_net``; the output layers are ``.out`` of the nets and ``size_out``, ``kill_out``, ``gold_out``.

    ``zone_mark`` (keyword-only; S4b preparation M9, author decision D9): True (the default) is every earlier model. False is the A7 design check "without the zone mark": the zone embedding Γ_z
    that goes into the maps A^z of M3, the objectives and M4 is set to 0 (see ``_conditions``), so those heads do not read z at all; the shapes, the parameters (the zone table, the M2Zone head
    and the maps A^z stay in the module, unused) and the order of the operations do not change, which keeps the checkpoint keys, the optimizer groups and the output shapes. The zone law
    ``m2().zone_logp`` is still computed (the composer does not draw from it) but the training does not score it: that is the business of ``train.score_blocks(zone_mark=False)``. The module
    knows its setting (``self.zone_mark``) because the model object carries no training identity: the composer and the evaluation check it against the setting they are given.
    """

    def __init__(self, cfg: EncoderConfig, *, zone_mark: bool = True):
        super().__init__()
        if not isinstance(zone_mark, bool):
            raise ValueError(f'zone_mark must be True (the zone mark is a condition of M3, the objectives and M4) or False (A7: no zone mark), not {zone_mark!r}')
        d, hidden, table = cfg.d, cfg.d, max(8, cfg.d // 4)
        self.cfg = cfg
        self.zone_mark = zone_mark
        self.bin_emb, self.zone_emb, self.size_emb = nn.Embedding(N_Q + 1, D_EMB), nn.Embedding(N_ZONE + 2, D_EMB), nn.Embedding(N_SIZE, D_EMB)
        self.mark = _MarkEmbedding(d)
        self.rate_net = _Mlp((d, D_EMB), hidden, 1)                                  # inputs: h, Γ_q
        self.mark_net = _Mlp((d, D_EMB), table, 1)                                   # Γ_m, Γ_q
        self.zone_net = _Mlp((d, D_EMB, d), table, N_ZONE)                           # Γ_m, Γ_q, h
        self.m3_net = _Mlp((d, D_EMB, d, D_EMB), hidden, None)                      # h, Γ_q, Γ_m, Γ_z
        self.size_out, self.kill_out = _output_layer(hidden, N_SIZE), _output_layer(hidden, N_SIZE * N_TEAMS)
        self.gold_in, self.gold_slot, self.gold_out = nn.Linear(hidden, hidden, bias=False), nn.Embedding(N_SLOTS, hidden), _output_layer(hidden, N_GOLD)
        self.obj_net = _Mlp((d, D_EMB, d, D_EMB, D_EMB, N_K), hidden, N_CELLS)       # h, Γ_q, Γ_m, Γ_z, Γ_s, log(1 + K)
        self.conv_net = _Mlp((d, D_EMB, d, D_EMB, N_K, N_CELLS), hidden, N_CELLS)    # h, Γ_q, Γ_m, Γ_z, log(1 + K), log(1 + O)
        for emb in (self.bin_emb, self.zone_emb, self.size_emb, self.gold_slot):
            nn.init.normal_(emb.weight, std=_INIT_STD)

    def _conditions(self, h: torch.Tensor, players: torch.Tensor, team: torch.Tensor, q, m, z) -> tuple:
        """The embeddings Γ_q, Γ_m, Γ_z of (EH-2) for the masked reading (q, m, z given) or the open one (all None).

        수식
        ----
        (EH-2z)  Γ_z = E_zone[z + 1] (기본값, zone_mark=True) ;   Γ_z = 0 (zone_mark=False, A7)      →  A^z Γ_z = 0 (A^z는 편향 없는 선형 사상)

          기호     | 뜻                                              | 단위 | 범위     | 코드 변수
          Γ_z      | 구역 임베딩 (z = −1과 "없음" 포함)              | —    | 실수     | zone_emb 또는 0
          A^z      | M3·M3_obj·M4의 구역 입력 사상                   | —    | 행렬     | m3_net.maps[3], obj_net.maps[3], conv_net.maps[3]
          zone_mark | 구역 표지를 조건으로 쓰는가 (결정 D9)           | —    | 참/거짓  | self.zone_mark

        해석
        ----
        구역 표지를 뺀 설계(A7)는 M3·오브젝트·M4가 z를 읽지 않는 모형이다. 사상을 지우는 대신 입력 Γ_z를 0으로 두면 항 A^z Γ_z는 정확히 0(편향 없는 사상이므로 +0.0)이어서 모양, 매개변수 목록,
        연산 순서가 그대로이고 체크포인트 키와 최적화기 묶음이 바뀌지 않는다. z가 달라도 출력이 같으며, 구역 임베딩은 기울기를 받지 않아 학습되지 않는다. 사상 A^z의 가중은 기울기가 0인 채로
        남아(0 입력의 기울기는 0) 출력에 쓰이지 않는다. 열린 읽기도 같은 0을 쓴다(열린 읽기의 "없음" 구역 값은 이 설계에서 존재하지 않는다). 기본값은 이 줄을 지나지 않으므로 이전 출력과 비트까지 같다.

        근거
        ----
        저자 결정 D9 (A7: 구역 표지는 공통 M2Rate + M2Mark NLL이 비열등이고 V_start CRPS와 가린 M3 NLL이 이득일 때만 둔다), 계획 M9, 검토 A2-F3.
        """
        n = h.shape[0]
        q, m, z = _reading(n, h, q, m, z)
        if q is None:
            none = h.new_full((n,), 0, dtype=torch.long)
            emb = self.bin_emb(none + N_Q), self.mark.absent(n), self.zone_emb(none + N_ZONE + 1)
        else:
            emb = self.bin_emb(q), self.mark(players, team, m), self.zone_emb(z + 1)
        if not self.zone_mark:
            return emb[0], emb[1], torch.zeros_like(emb[2])                                              # Γ_z = 0: a constant, so the zone table gets no gradient
        return emb

    def m2(self, enc: Encoded, exposure, pos=None) -> M2Out:
        """M2 (occurrence): for every row and each of the 18 bins q, the expected number of first kills ``rate`` (n, 18), the log law of the mark
        ``mark_logp`` (n, 18, 52) and the log law of the zone ``zone_logp`` (n, 18, 52, 10). ``exposure``: the seconds of each bin that were observed, a
        tensor (n, 18) (a number or (n,) also works), finite and not negative. ValueError for another exposure and for ``pos`` out of range.

        수식
        ----
        (EH-1)  η_q = Θ GELU(A^h h + A^q Γ_q) ;   λ_q = exp(η_q) ;   rate_q = μ_q = λ_q · Δ_q
        (EH-2)  Γ_m = A^kil h_m^kil + A^vic h_m^vic + E_kind[쌍] (m ≤ 49) ;   Γ_m = A^tm h̄_{t(m)} + E_kind[처형] (m = 50, 51)
        (EH-3)  η_{q,m} = Θ GELU(A^m Γ_m + A^q Γ_q) ;   π(m | q) = (EH-4)를 쓴 softmax_m(η_{q,·})
        (EH-5)  ρ(z | q, m) = (EH-4)를 쓴 softmax_z(Θ GELU(A^m Γ_m + A^q Γ_q + A^h h))

          기호             | 뜻                                                                       | 단위  | 범위      | 코드 변수
          h                | 요약 토큰의 상태 (보기의 풀링)                                           | —     | 실수      | enc.h_pool
          h_m^kil, h_m^vic | 표지 m의 킬러와 피해자의 선수 토큰 상태 (슬롯 번호를 쓰지 않는다)        | —     | 실수      | enc.h_tok의 선수 행
          h̄_t              | 팀 t의 선수 다섯의 평균 상태 (t(m) = m − 50: 처형당한 쪽의 팀)           | —     | 실수      | team_mean
          Γ_q, Γ_m         | 구간 임베딩(계획의 e_q), 표지 임베딩 (e는 노출에만 쓴다)                             | —     | 실수      | bin_emb, mark
          E_kind[·]        | 표지 종류 (쌍, 처형, 없음)의 학습 벡터                                   | —     | 실수      | mark.kind
          A^•, Θ           | 머리마다 따로 학습하는 입력 사상(은닉층 앞, 합에 편향 벡터 하나를 더한다)과 최종 층(편향 포함) | — | 행렬 | maps[k], bias, out
          GELU             | 은닉층의 활성 함수                                                       | —     | —         | F.gelu
          λ_q, Δ_q, μ_q    | 구간 q의 첫 킬 발생률, 노출, 기대 개수                                   | 1/s, s, 개 | ≥ 0   | exp(η) , exposure, rate
          η_q, η_{q,m}     | 로그 연결 앞의 점수 (발생률, 표지)                                       | —     | 실수      | 최종 층의 출력

        해석
        ----
        μ_q는 노출에 비례하고, 노출이 0인 구간(T − 30 s에서 잘려 관측 시간이 0인 구간; 등록부의 Δ_q)은 정확히 0이며 그 칸으로 기울기가 가지 않는다 (계획 P10, 검토 초점 4).
        log λ_q가 [h, Γ_q]를 은닉층 하나에 넣어 얻어지므로 같은 상태가 미치는 영향이 구간마다 다를 수 있다 (B2-a: 앞 시간 창마다 발생 기울기가 1.16, 0.88, 0.95로 달랐다). 표지는 킬러와 피해자의 선수 토큰 상태에서만
        점수를 얻고 슬롯 번호는 입력이 아니므로, 한 팀 안에서 선수를 바꿔 쓰면 표지 번호만 따라 움직인다 (설계서 5.2절); 처형 표지 둘은 처형당한 팀의 평균 상태에서 나온다
        (설계서 11절 2항). ρ는 규칙 구역 열 개 위의 법칙이고, 위치 없는 첫 킬(z = −1)은 범주가 아니어서 이 머리가 점수를 내지 않는다.

        근거
        ----
        계획 Q7 (log λ_q = MLP([h, Γ_q]), π: 쌍 점수의 소프트맥스, ρ: 열 구역), 설계서 3.1절 (λ_q × π × ρ), 판정 R13 ((EH-4)), 계획 P10 (노출).
        """
        h, players, team = _read(enc, pos)
        n = h.shape[0]
        e = _exposure(exposure, n, N_Q, h)
        emb_q = self.bin_emb.weight[:N_Q]                                                                 # (18, D_EMB): Γ_q of every bin
        eta = self._m2_eta(h, emb_q)
        score, zone = self._m2_law_logits(h, players, team, emb_q)
        return M2Out(torch.exp(eta) * e, mixed_log_softmax(score), mixed_log_softmax(zone))

    def m2_rate(self, enc: Encoded, exposure, pos=None) -> torch.Tensor:
        """구간별 첫 킬 기대 발생 수 rate(μ_q = λ_q × Δ_q)를 반환한다 (EH-1).

        매개변수:
        - enc (Encoded): 부호화된 상태 텐서.
        - exposure (Tensor 또는 float): 구간별 관측 노출 시간 Δ_q (초 단위). 형상 (n, 18), (n,) 또는 스칼라를 지원하며 유한한 비음수여야 한다.
        - pos (Tensor, 선택): 각 출력 행이 읽을 부호화 뷰의 인덱스; 같은 뷰를 여러 행이 읽을 수 있다.

        반환값:
        - rate (Tensor): 각 표본 및 18개 구간의 기대 첫 킬 수, 형상 (n, 18).

        동작 및 예외:
        - exposure 형상·범위 오류나 pos 유효 범위 초과 시 ValueError가 발생한다.
        - 노출이 0인 구간의 출력과 가중치 기울기는 정확히 0이다.
        - 학습·평가 모드 모두 표지와 구역 로짓 연산을 건너뛰고 m2와 같은 연산 순서로 _m2_eta만 계산하여 발생률을 산출한다.
          rate는 _m2_eta의 출력만 읽으므로 건너뛴 분기는 rate의 값에도 손실 그래프(기울기)에도 들어가지 않는다. EPTHeads에는 드롭아웃 등 난수 연산이 없어
          (가중치 초기화 말고는 없다) 그 분기를 건너뛰어도 난수 소비(RNG)가 같다. 그래서 학습 모드에서도 값과 모든 가중치의 기울기가 m2(...).rate와
          비트 단위로 같다 (EPT-T011 A2-F10; 시험: test_ept_m2_components, test_ept_heads, test_ept_training_step). 이 비트 일치는 CPU에서 시험했으며 GPU는 시험하지 않았다
          (GPU에서는 결정적 알고리즘 설정 아래에서 같은 논리가 성립할 것으로 기대한다). 머리에 난수 연산을 넣는다면 이 선택을 다시 해야 한다."""
        h, _, _ = _read(enc, pos)
        e = _exposure(exposure, h.shape[0], N_Q, h)
        eta = self._m2_eta(h, self.bin_emb.weight[:N_Q])
        return torch.exp(eta) * e

    def _m2_eta(self, h, emb_q):
        """The unchanged shape and operation order of the rate logits (EH-1)."""
        net = self.rate_net
        return net.finish(net.project(0, h)[:, None] + net.project(1, emb_q)[None]).squeeze(-1)

    def _m2_law_logits(self, h, players, team, emb_q):
        """The mark and zone logits (EH-2, EH-3, EH-5), in the order used by m2."""
        emb_m = self.mark.every(players, team)                                                            # (n, 52, d): (EH-2) for every mark
        # the scores of the 52 marks in each of the 18 bins, and the zone law of every (bin, mark): both are tables over (q, m), so the maps of the
        # three inputs are applied once to each input and only the sum is formed for every pair
        net = self.mark_net
        score = net.finish(net.project(0, emb_m)[:, None] + net.project(1, emb_q)[None, :, None]).squeeze(-1)    # (n, 18, 52): (EH-3)
        net = self.zone_net
        zone = net.finish(net.project(0, emb_m)[:, None] + net.project(1, emb_q)[None, :, None] + net.project(2, h)[:, None, None])   # (n, 18, 52, 10): (EH-5)
        return score, zone

    def m3(self, enc: Encoded, q, m, z, pos=None) -> M3Out:
        """M3 (outcome of the engagement) for every row and every size class: the log law of the size ``size_logp`` (n, 4), the mean number of kills of each
        team ``kill_rate`` (n, 4, 2) (blue, red) and the log law of the kill-gold bin of each team ``gold_logp`` (n, 4, 2, 6, 31): size, team, the team's
        own kill layer (1 .. 5 kills, 6 for six or more; index ξ − 1) and gold bin. ``q``, ``m``, ``z``: the conditions of the masked reading, or None
        (open reading). ValueError as for the conditions of ``EPTHeads``.

        수식
        ----
        (EH-6)  ζ = GELU(A^h h + A^q Γ_q + A^m Γ_m + A^z Γ_z) ;   P(s | q, m, z) = (EH-4)를 쓴 softmax_s(Θ^size ζ) ;   μ_{s,t} = exp((Θ^kill ζ)_{s,t})
                p(b | s, t, ξ) = (EH-4)를 쓴 softmax_b(Θ^gold GELU(A^g ζ + E_slot[s, t, ξ]))        Γ_m은 (EH-2)

          기호             | 뜻                                                              | 단위  | 범위     | 코드 변수
          h, Γ_q, Γ_m, A^•, Θ, GELU | ``m2``의 표와 같다                                     | —     | —        | m2의 표
          ζ                | 규모·킬·골드 출력이 공유하는 조건 벡터 (은닉층의 값)            | —     | 실수     | trunk
          Γ_z              | 구역 임베딩 (z = −1과 "없음"을 포함)                            | —     | 실수     | zone_emb
          s, t, ξ, b       | 규모, 팀 (파랑, 빨강), 그 팀의 킬 층, 킬 골드 구간              | —     | 0..3, 0..1, 1..6, 0..30 | 출력의 축
          μ_{s,t}          | 규모 s에서 팀 t의 킬 수 평균 (노출 1)                           | 개    | > 0      | kill_rate
          E_slot[s, t, ξ]  | (규모, 팀, 킬 층)마다의 학습 벡터                               | —     | 실수     | gold_slot
          Θ^size, Θ^kill, Θ^gold, A^g | 최종 층들과 골드 머리의 입력 사상                    | —     | 행렬     | size_out, kill_out, gold_out, gold_in

        해석
        ----
        한 교전의 결과를 규모 관문과 규모별 전문가로 나눈다: 규모의 법칙, 규모마다 팀별 킬 수의 포아송 평균, 규모와 팀과 그 팀의 킬 층마다 킬 골드 구간의 범주 법칙이다 (계획 P4).
        전문가는 규모마다 따로 둔 은닉층이 아니라 하나의 은닉층 ζ를 공유하는 규모별 출력 행(kill_out, gold_slot)이며, 설계서 3.2절의 A5는 이 구조를 단일 머리와 견준다.
        골드 법칙은 그 팀 자신의 킬 층만 읽는다: 상대 팀의 층은 입력이 아니고, 법칙마다 따로 둔 (규모, 팀, 층) 벡터 하나만 은닉층에 더해진다. 합성의 닫힌 꼴이 두 팀을 같은
        층으로 물어 한 팀의 법칙을 뽑기 때문이다 (compose (COMP-7)). 열린 읽기는 q, m, z 자리에 "없음" 값을 넣고 같은 가중을 쓴다 (설계서 4절).

        근거
        ----
        계획 P4 (P(s) · P(K | s) · P(O | K, s); 킬 수는 포아송, 골드는 킬 층 조건부 31구간 범주), 계획 Q7, 판정 R13, 설계서 3.2절 (규모 관문 × 전문가).
        """
        h, players, team = _read(enc, pos)
        n = h.shape[0]
        emb_q, emb_m, emb_z = self._conditions(h, players, team, q, m, z)
        net = self.m3_net
        trunk = net.act(net.project(0, h) + net.project(1, emb_q) + net.project(2, emb_m) + net.project(3, emb_z))   # (n, d): ζ
        slot = F.gelu(self.gold_in(trunk)[:, None] + self.gold_slot.weight[None])                                     # (n, 48, d) the three-way places
        gold = self.gold_out(slot).reshape(n, N_SIZE, N_TEAMS, N_LAYER, N_GOLD)
        return M3Out(mixed_log_softmax(self.size_out(trunk)), torch.exp(self.kill_out(trunk)).reshape(n, N_SIZE, N_TEAMS), mixed_log_softmax(gold))

    def m3_obj(self, enc: Encoded, q, m, z, s, K, pos=None) -> torch.Tensor:
        """The expected number of objectives of each of the 18 cells (kind x taker) of the engagement, (n, 18), given the size class ``s`` (n,) and the kills
        ``K`` (n, 4) of the engagement; ``q``, ``m``, ``z`` as for ``m3``. ValueError as for the conditions of ``EPTHeads``.

        수식
        ----
        (EH-7)  μ_c = exp(η_c),   η = Θ GELU(A^h h + A^q Γ_q + A^m Γ_m + A^z Γ_z + A^s Γ_s + A^K log(1 + K))

          기호      | 뜻                                                          | 단위     | 범위   | 코드 변수
          h, Γ_q, Γ_m, Γ_z, A^•, Θ, GELU | ``m2``·``m3``의 표와 같다              | —        | —      | m2·m3의 표
          μ_c       | 칸 c = 2 · 종류 + 취한 팀의 오브젝트 기대 개수 (노출 1)     | 개       | > 0    | 반환값
          Γ_s       | 규모 임베딩                                                 | —        | 실수   | size_emb
          K         | 파랑·빨강 킬 수, 파랑·빨강 킬 골드 (천 골드)                | 개, 천 골드 | ≥ 0 | K
          A^s, A^K  | 규모와 log(1 + K)의 입력 사상                               | —        | 행렬   | maps[4], maps[5]

        해석
        ----
        오브젝트는 칸마다 포아송이고 K와 규모에 조건부이다 (계획 P4). 킬 수와 골드는 오른쪽 꼬리가 길어서 log(1 + K)로 넣고, 킬이 없을 때 0이다. 머리는 O를 받지 않는다:
        O는 이 머리의 정답이다.

        근거
        ----
        계획 P4 (칸별 포아송, O는 K와 규모 조건부), 계획 Q9 (EPT의 가린 오브젝트 머리는 K를 읽는다), 계획 Q7, 설계서 3.2절.
        """
        h, players, team = _read(enc, pos)
        n = h.shape[0]
        emb_q, emb_m, emb_z = self._conditions(h, players, team, q, m, z)
        emb_s = self.size_emb(_whole('s', s, n, 0, N_SIZE - 1, h))
        net = self.obj_net
        eta = net.finish(net.project(0, h) + net.project(1, emb_q) + net.project(2, emb_m) + net.project(3, emb_z) + net.project(4, emb_s)
                         + net.project(5, torch.log1p(_counts('K', K, n, N_K, h))))
        return torch.exp(eta)

    def m4(self, enc: Encoded, q, m, z, K, O, exposure, pos=None) -> torch.Tensor:                # noqa: E741  (O: the objectives, as in the registry)
        """The expected number of objectives converted in the window after the engagement, for each of the 18 cells, (n, 18), given the kills ``K`` (n, 4) and
        the objectives ``O`` (n, 18) of the engagement; ``q``, ``m``, ``z`` as for ``m3``. ``exposure``: the conversion window in seconds, one value for each row:
        a number or a tensor (n,), finite and not negative. ValueError for a value for each cell (n, 18) (ruling R-S4a-11) and as for the conditions of ``EPTHeads``.

        수식
        ----
        (EH-8)  μ_c = e · λ_c,   λ_c = exp(η_c),   η = Θ GELU(A^h h + A^q Γ_q + A^m Γ_m + A^z Γ_z + A^K log(1 + K) + A^O log(1 + O))

          기호      | 뜻                                                          | 단위  | 범위   | 코드 변수
          h, Γ_q, Γ_m, Γ_z, K, A^•, Θ, GELU | ``m2``·``m3``·``m3_obj``의 표와 같다 | —     | —      | 앞의 표
          μ_c       | 칸 c의 전환 수 기대 개수 (e = 1이면 λ_c)                    | 개    | ≥ 0    | 반환값
          λ_c       | 칸 c의 전환율                                               | 1/s   | ≥ 0    | exp(η)
          e         | 전환 창에서 관측한 시간 (노출; 행마다 한 값)                | s     | ≥ 0    | exposure
          O         | 교전의 오브젝트 18칸                                        | 개    | ≥ 0    | O
          A^O       | log(1 + O)의 입력 사상                                      | —     | 행렬   | maps[5]

        해석
        ----
        전환은 칸마다 포아송이고 노출에 비례하며, 창이 비었거나 경기가 끝난 행(e = 0)은 0이고 그 칸으로 기울기가 가지 않는다 (계획 P10, 검토 초점 4).
        창은 행 하나에 하나이므로 팀 교환이 칸을 맞바꿔도 그 값은 그대로이고, 칸마다 다른 창은 맞바꾼 틀의 짝 칸에 옮겨 가므로 받지 않는다 (판정 R-S4a-11).
        가린 읽기는 (q, m, z, K, O)로, 열린 읽기는 q, m, z 없이 교전 끝 보기의 상태와 (K, O)로 묻는다 (설계서 3.3절). 머리는 규모 s를 받지 않는다: 가린 M4는 (q, m, z, K, O)만 본다.

        근거
        ----
        계획 Q7 ("노출 포함", 가린 보기는 (q, m, z, K, O), 열린 보기는 δ 보기), 계획 P10 (전환 노출), 판정 R-S4a-11 (창은 행마다 한 값), 설계서 3.3절 (누설 시험 (a)).
        """
        h, players, team = _read(enc, pos)
        n = h.shape[0]
        emb_q, emb_m, emb_z = self._conditions(h, players, team, q, m, z)
        window = _window(exposure, n, h)
        net = self.conv_net
        eta = net.finish(net.project(0, h) + net.project(1, emb_q) + net.project(2, emb_m) + net.project(3, emb_z)
                         + net.project(4, torch.log1p(_counts('K', K, n, N_K, h))) + net.project(5, torch.log1p(_counts('O', O, n, N_CELLS, h))))
        return torch.exp(eta) * window[:, None]


class EPT(nn.Module):
    """The encoder and the heads of the EPT in one module: ``encode(batch)`` runs the encoder (``batch``: the tensors of ``data.ViewTensors``), ``heads`` is the
    ``EPTHeads``; ``cfg`` and ``static`` are those of ``EPTEncoder``.

    해석: 부호화기와 머리를 한 모듈에 담아 학습 고리와 프레임 평균이 하나의 모형을 다루게 한다. 가중은 둘이 따로 갖는다.
    """

    def __init__(self, cfg: EncoderConfig, static: np.ndarray | bool, *, zone_mark: bool = True):
        super().__init__()
        self.encoder = EPTEncoder(cfg, static)
        self.heads = EPTHeads(cfg, zone_mark=zone_mark)

    def encode(self, batch: Mapping) -> Encoded:
        return self.encoder(batch)


# ---- the frame average ------------------------------------------------------------------------------------------------------------------------
# which row field of ``data.ROW_FIELDS`` stands for each condition that the swap changes, and what ``data.FIELD_SWAP`` must say about it
_CONDITION_FIELDS = MappingProxyType({'m': ('M2Zone', 'masked', 'm', 'mark'), 'z': ('M3Size', 'masked', 'z', 'zone'),
                                      'K': ('M3Obj', 'masked', 'K', 'k'), 'O': ('M4Conv', 'masked', 'O', 'cells')})
for _name, (_head, _kind, _field, _change) in _CONDITION_FIELDS.items():
    if data.FIELD_SWAP[(_head, _kind, _field)] != _change:
        raise RuntimeError(f'data.FIELD_SWAP does not swap the condition {_name} as {_change!r}: the frame average would swap it wrongly')


def _swapped(**given) -> dict:
    """(EH-9)의 조건 교환: ``data.FIELD_SWAP``으로 m·z와 K·O를 함께 바꾼다(None은 유지)."""
    keys = {name: data.row_key(*_CONDITION_FIELDS[name][:3]) for name, t in given.items() if t is not None}
    out = data.swap({key: given[name] for name, key in keys.items()})
    return {name: (out[keys[name]] if name in keys else None) for name in given}


def _back(t: torch.Tensor, dim: int, perm: torch.Tensor) -> torch.Tensor:
    """The output ``t`` of one frame with its axis ``dim`` moved to the other frame: index i of the other frame = index perm[i] (the tables are their own
    inverse, ``data`` checks it, so the direction does not matter)."""
    return t.index_select(dim, perm.to(t.device))


class FrameEncoded(NamedTuple):
    """The encodings of one batch in the frame it came in (``original``) and in its team swap (``swapped``), and ``side`` (views,) bool, True for a view that
    the batch recorded in the swapped frame (side marker 1); made by ``FrameAveraged.encode``.

    One frame (``FrameAveraged(model, average=False)``, S4b preparation M7): ``original`` is the encoding of every view in the frame that was chosen for it, ``swapped`` is None, ``side`` is the
    side marker of the encoded views (True: the view was swapped by its coin) and ``rows`` is the batch in those frames (``data.swap_views``: the rows that read a swapped view are swapped with it)
    when the batch had coins, else None (every view as recorded, the rows of the batch are the rows to score). Under the average ``rows`` is None."""
    original: Encoded
    swapped: Encoded | None
    side: torch.Tensor
    rows: Mapping | None = None


class FrameAveraged(nn.Module):
    """The model run in the two team frames and averaged. ``fa.encode(batch)`` encodes the batch and, in a separate call on a batch of the same shape, its team
    swap ``data.swap`` (a ``FrameEncoded``); ``fa.m2``, ``m3``, ``m3_obj`` and ``m4`` take that and the conditions of the frame of the batch (the arguments of the
    heads of ``model``). ``m3``, ``m3_obj`` and ``m4`` check and cast the conditions (``m``, ``z``, ``s``, ``K``, ``O``) and the window of M4 before the swap
    indexes the m and z tables and permutes the columns of K and O, with the ValueError of the heads (also for heads that accept anything); they then ask the heads of ``model`` about both frames with
    the cast tensors (the swapped frame with swapped marks, zones, K and O; q, s, the exposure and ``pos`` do not change; None stays None and a zone of -1 stays
    -1), move the swapped answer back by the tables of ``data`` and average. ``model`` has ``encode(batch) -> Encoded`` and ``heads`` (an ``EPT``). TypeError
    for another object.

    ``average=False`` (keyword only, default True; S4b preparation M7, decisions D10 P1 and D17) is the wrapper of the runs that use one frame: the training of ``swap_mode='random_view'`` and
    ``'none'`` and the evaluation with ``eval_average=False``. ``encode`` then makes one call of the encoder on the views of the batch, each view in its chosen frame (``data.swap_views``
    when the batch has the coins ``swap_view``; as recorded when it has none), and ``m2``, ``m2_rate``, ``m3``, ``m3_obj`` and ``m4`` return the answer of the heads for that frame as it is: nothing
    is swapped, moved back or averaged, the conditions are those of the rows in the frame of their view (the rows come back swapped in ``FrameEncoded.rows``, which ``train.score_blocks`` reads), and
    the conditions are checked and cast as under the average. The averaging wrapper refuses a batch with ``swap_view`` (ValueError: its coins ask for one frame), ``average`` that is not a bool is a TypeError.

    수식
    ----
    (EH-9)  ȳ(b) = ½ [ŷ(b) + Π ŷ(swap b)]   (율 μ와 확률 p의 산술평균) ;   log p̄(b) = logaddexp(log p̂(b), log Π p̂(swap b)) − log 2
            (Π ŷ)[i] = ŷ[PERM[i]] ;  PERM = MARK_PERM (표지 축), ZONE_PERM (구역 축), TEAM_PERM (팀 축), CELL_PERM (오브젝트·전환 칸 축)
            log 법칙의 평균은 기록된 틀(진영 표시 0)에서 계산하고, 진영 표시 1인 행은 그 결과에 Π를 쓴다 (Π는 자기 역이어서 같은 값이다)
            Σ_c p̄_c = 1 ; p̄_c ≥ ν/C   ((EH-4)의 공통 균등 혼합 하한)
    (EH-10) ȳ(b) = ŷ(F_c(b)) (한 틀, average=False) ;  F_c(b) = 보기 v마다 swap b_v (c_v = 1) 또는 b_v (c_v = 0), 그 보기를 읽는 행의 조건과 정답도 같은 틀 (DAT-11) ;  평균도 Π도 없다

      기호      | 뜻                                                                | 단위 | 범위     | 코드 변수
      ŷ(b)      | 보기 묶음 b에서 머리가 낸 율 μ 또는 확률 p (p̂는 그 가운데 확률)  | —    | ≥ 0      | 머리의 반환값
      swap b    | 팀을 맞바꾼 묶음 (입력, 조건이 함께 바뀐다)                       | —    | —        | data.swap, _swapped
      Π         | 맞바꾼 틀의 출력을 원래 틀로 되돌리는 축 순열                     | —    | —        | _back, _MARK, _ZONE, _TEAM, _CELL
      ȳ, p̄      | 프레임 평균한 율, 확률                                            | —    | ≥ 0      | 반환값
      진영 표시 | 묶음이 보기를 기록한 틀 (0 기록대로, 1 팀 교환)                   | —    | 0 또는 1 | enc.side
      F_c(b)    | 동전 c대로 보기마다 틀을 고른 묶음 (한 틀)                         | —    | —        | data.swap_views, enc.rows
      c_v       | 보기 v의 동전 (1이면 맞바꾼 틀)                                    | —    | 0 또는 1 | batch['swap_view']

    해석
    ----
    팀 교환은 두 번 쓰면 제자리로 오므로, 맞바꾼 틀의 출력을 되돌려 원래 틀의 출력과 평균하면 부호화기와 머리가 팀 교환에 불변이 아니어도 평균은 정확히 불변이다.
    −log p와 μ − y log μ의 볼록성으로 평균한 예측의 NLL은 두 틀의 NLL 평균을 넘지 않는다 [도출].
    비트 단위의 불변을 위해 두 틀을 같은 모양의 별개 호출로 계산한다: 한 묶음에 쌓으면 같은 행이 놓이는 자리에 따라 반올림이 달라진다.
    log 법칙의 평균은 기록된 틀에서 계산한다: 벡터화한 exp·log1p는 같은 값도 버퍼의 자리에 따라 마지막 비트가 다르다.
    이 정확한 불변은 평가 모드에서만 성립한다: 드롭아웃을 켜면 두 틀의 마스크가 다르다.
    한 틀 (EH-10)은 부호화기를 보기마다 한 번만 돌리므로 학습 비용이 두 틀 평균의 절반이다. 평균하지 않으므로 출력은 팀 교환에 불변이 아니다: 정확한 불변은 평가의 프레임 평균(eval_average=True)이 내고,
    학습에서 보기마다 동전으로 틀을 고르면 같은 가중이 두 틀의 입력을 모두 보게 되어 가중 공유 교환 증강이 된다 (결정 D17의 C3 문구). 대칭 없음 점검(결정 D17)은 이 한 틀을 동전 없이(swap_mode='none')
    쓰고 평가도 한 틀(eval_average=False)로 하여, 증강과 평균을 모두 뺀 모형을 잰다.

    근거
    ----
    설계서 5.2·6.1절과 법칙 A1 (정확한 대칭은 프레임 평균), 계획 Q7과 검토 초점 1 (반쪽 적용), 판정 R13, R-S4a-6 (swap은 자기 역, 표는 ``data``의 것), R-S4a-11 (M4 창은 행마다 한 값).
    Ref: Puny 외 2022 (위 Ref).
    """

    def __init__(self, model, *, average: bool = True):
        super().__init__()
        if not callable(getattr(model, 'encode', None)) or not hasattr(model, 'heads'):
            raise TypeError('the model of a FrameAveraged must have encode(batch) and heads (an EPT)')
        if not isinstance(average, bool):
            raise TypeError(f'average must be True (both frames, the mean) or False (one frame), not {average!r}')
        self.model, self.average = model, average

    def encode(self, batch: Mapping) -> FrameEncoded:
        """Encode ``batch`` (the view tensors; the keys of the row blocks are ignored) and, in a separate call on a batch of the same shape, its team swap.

        ``average=False``: one call of the encoder on the views in their chosen frames (``data.swap_views`` for a batch with the coins ``swap_view``; the recorded views for a batch without);
        ``swapped`` is None, ``rows`` the batch in those frames (None without coins). ValueError for a batch with ``swap_view`` under the average.

        해석: 두 틀을 한 묶음에 쌓지 않고 같은 모양의 별개 호출로 부호화해, 같은 행이 두 호출에서 같은 자리에서 같은 연산을 받게 한다. 한 틀은 부호화기를 한 번만 부르고, 동전이 선 보기는
        ``data.swap_views``가 입력과 그 보기를 읽는 행을 함께 맞바꾼 것을 부호화한다. 학습 고리는 이 변환을 복사 앞에서 CPU 배치에 이미 적용해 두므로(S4b 속도 개선 SPD 항목 1, 결정 D32,
        ``batch_prep.prepare_batch``) 거기서 온 배치에는 동전이 없고 이 함수는 기록대로의 틀을 부호화한다. 동전이 든 배치를 직접 주는 쪽(시험, 다른 호출자)은 지금도 같은 결과를 얻는다
        (같은 함수이고 모든 연산이 정확해서 어디서 적용해도 같은 비트이다).
        """
        if not self.average:
            framed = data.swap_views(batch) if 'swap_view' in batch else None
            views = {k: (batch if framed is None else framed)[k] for k in data.VIEW_KEYS if k in batch}
            return FrameEncoded(self.model.encode(views), None, views['side'] > 0.5, framed)
        if 'swap_view' in batch:
            raise ValueError('the batch has swap_view (one frame for each view) but this FrameAveraged averages both frames: use FrameAveraged(model, average=False)')
        views = {k: batch[k] for k in data.VIEW_KEYS if k in batch}          # the keys of the row blocks are no input and need no swap
        original = self.model.encode(views)
        return FrameEncoded(original, self.model.encode(data.swap(views)), views['side'] > 0.5)

    def m2(self, enc: FrameEncoded, exposure, pos=None) -> M2Out:
        """(EH-9) for ``EPTHeads.m2``: the marks of ``mark_logp`` by ``MARK_PERM``, the marks and zones of ``zone_logp``; the rate has no axis to move.
        An encoding of one frame (``enc.swapped`` None, ``average=False``) gives the answer of the heads for that frame, as it is (EH-10)."""
        if enc.swapped is None:
            return self.model.heads.m2(enc.original, exposure, pos)
        a, b = self.model.heads.m2(enc.original, exposure, pos), self.model.heads.m2(enc.swapped, exposure, pos)
        side = _side(enc, pos, a.rate)
        return M2Out(_mean(a.rate, b.rate), _log_mean(a.mark_logp, b.mark_logp, side, lambda t: _back(t, -1, _MARK)),
                     _log_mean(a.zone_logp, b.zone_logp, side, lambda t: _back(_back(t, 2, _MARK), 3, _ZONE)))

    def m2_rate(self, enc: FrameEncoded, exposure, pos=None) -> torch.Tensor:
        """좌우 프레임 대칭(EH-9)을 적용하여 구간별 첫 킬 기대 개수(EH-1)의 앙상블 평균을 계산한다.

        매개변수:
        - enc (FrameEncoded): 원본(original) 및 교환(swapped) 부호화 상태.
        - exposure (Tensor 또는 float): 구간별 관측 노출 시간 (초 단위).
        - pos (Tensor, 선택): 평가할 행의 위치 인덱스.

        반환값:
        - rate (Tensor): 원본과 교환 프레임 출력의 평균 기대 첫 킬 수, 형상 (n, 18).

        동작 및 대체 규칙:
        - 하위 머리에 호출 가능한 m2_rate가 있으면 학습·평가 모드와 관계없이 표지·구역 분기를 건너뛰고 양쪽 프레임의 rate만 각각 계산한 뒤 _side 검증을 거쳐 평균을 취한다.
          학습 모드에서도 값과 기울기는 m2(...).rate와 비트 단위로 같다: 건너뛴 분기는 rate와 손실 그래프에 들어가지 않고 EPTHeads에 난수 연산이 없다 (EPT-T011 A2-F10;
          CPU에서 시험했으며 GPU는 시험하지 않았다).
        - 하위 머리에 m2_rate가 없거나 호출 불가능한 레거시 mock 객체이면 전체 m2를 호출해 그 rate를 취하여 기존 프로토콜을 그대로 보존한다.
        - 드롭아웃이 비활성화된 평가 모드에서 정확한 팀 교환 불변성이 성립한다."""
        rate = getattr(self.model.heads, 'm2_rate', None)
        if not callable(rate):
            return self.m2(enc, exposure, pos).rate
        if enc.swapped is None:                   # one frame (S4b preparation M7, EH-10): the rate-only call of the heads for that frame, nothing to average
            return rate(enc.original, exposure, pos)
        a, b = rate(enc.original, exposure, pos), rate(enc.swapped, exposure, pos)
        _side(enc, pos, a)                    # retain the side/row selection of the full M2 path
        return _mean(a, b)

    def m3(self, enc: FrameEncoded, q, m, z, pos=None) -> M3Out:
        """(EH-9) for ``EPTHeads.m3``: the teams of ``kill_rate`` and ``gold_logp`` by ``TEAM_PERM``; the size law has no axis to move.

        이동 킬 법칙 (S4b 준비 M1, 결정 D5)에서도 이 평균은 바뀌지 않는다. 팀 t 칸의 값은 그 틀의 첫 킬 팀 d에 따라 λ_d(t = d) 또는 μ(t ≠ d)의 뜻을 갖는데, 팀을 맞바꾼 틀의 첫 킬 팀은
        1 − d이고 그 틀의 출력을 ``TEAM_PERM``으로 되돌리면 맞바꾼 틀에서 팀 1 − t의 칸이 원래 틀의 팀 t 칸으로 온다. t = d이면 맞바꾼 틀의 (1 − d) 칸, 곧 그 틀의 첫 킬 팀 칸(λ)이 오므로
        평균은 항상 같은 뜻(λ는 λ와, μ는 μ와)끼리 이루어진다. 손실이 쓰는 d는 기록된 틀의 것이고 평균한 출력도 기록된 틀에 있다."""
        like = enc.original.h_pool
        q, m, z = _reading(_row_count(enc.original, pos), like, q, m, z)
        if enc.swapped is None:                   # one frame (S4b preparation M7, EH-10): the conditions are already those of the frame of each view
            return self.model.heads.m3(enc.original, q, m, z, pos)
        s = _swapped(m=m, z=z)
        a, b = self.model.heads.m3(enc.original, q, m, z, pos), self.model.heads.m3(enc.swapped, q, s['m'], s['z'], pos)
        side = _side(enc, pos, a.kill_rate)
        return M3Out(_log_mean(a.size_logp, b.size_logp, side, lambda t: t), _mean(a.kill_rate, _back(b.kill_rate, -1, _TEAM)),
                     _log_mean(a.gold_logp, b.gold_logp, side, lambda t: _back(t, 2, _TEAM)))

    def m3_obj(self, enc: FrameEncoded, q, m, z, s, K, pos=None) -> torch.Tensor:
        """(EH-9) for ``EPTHeads.m3_obj``: the cells by ``CELL_PERM``; the size ``s`` is the same in both frames."""
        like, n = enc.original.h_pool, _row_count(enc.original, pos)
        q, m, z = _reading(n, like, q, m, z)
        s, K = _whole('s', s, n, 0, N_SIZE - 1, like), _counts('K', K, n, N_K, like)
        if enc.swapped is None:                   # one frame (S4b preparation M7, EH-10)
            return self.model.heads.m3_obj(enc.original, q, m, z, s, K, pos)
        t = _swapped(m=m, z=z, K=K)
        a = self.model.heads.m3_obj(enc.original, q, m, z, s, K, pos)
        return _mean(a, _back(self.model.heads.m3_obj(enc.swapped, q, t['m'], t['z'], s, t['K'], pos), -1, _CELL))

    def m4(self, enc: FrameEncoded, q, m, z, K, O, exposure, pos=None) -> torch.Tensor:           # noqa: E741
        """(EH-9) for ``EPTHeads.m4``: the cells by ``CELL_PERM``; the conversion window is one value for each row and the same in both frames."""
        like, n = enc.original.h_pool, _row_count(enc.original, pos)
        q, m, z = _reading(n, like, q, m, z)
        K, O, window = _counts('K', K, n, N_K, like), _counts('O', O, n, N_CELLS, like), _window(exposure, n, like)
        if enc.swapped is None:                   # one frame (S4b preparation M7, EH-10)
            return self.model.heads.m4(enc.original, q, m, z, K, O, window, pos)
        t = _swapped(m=m, z=z, K=K, O=O)
        a = self.model.heads.m4(enc.original, q, m, z, K, O, window, pos)
        return _mean(a, _back(self.model.heads.m4(enc.swapped, q, t['m'], t['z'], t['K'], t['O'], window, pos), -1, _CELL))


def _side(enc: FrameEncoded, pos, like: torch.Tensor) -> torch.Tensor:
    """The ``side`` of the rows (the views of ``enc`` in order, or ``pos``) on the device of ``like``."""
    side = enc.side.to(like.device)
    return side if pos is None else side.index_select(0, pos.to(device=like.device, dtype=torch.long))


def _mean(a: torch.Tensor, b_back: torch.Tensor) -> torch.Tensor:
    """(EH-9) for a rate: the mean of the answer of the frame ``a`` and that of the other frame already moved back ``b_back``."""
    return 0.5 * (a + b_back)


def _log_mean(a: torch.Tensor, b: torch.Tensor, side: torch.Tensor, back) -> torch.Tensor:
    """(EH-9) for a log law: ``a`` is the answer for the frame of the batch, ``b`` that for its swap (not yet moved), ``back`` moves an answer of one frame to
    the other, ``side`` (n,) bool says which rows came in the swapped frame."""
    side = side.reshape(-1, *([1] * (a.dim() - 1)))
    # the log of the mean is formed in the recorded frame (side 0) of every row (the reason is in the text of ``FrameAveraged``); a row that came in
    # the swapped frame gets the result moved by ``back``
    recorded, swapped = torch.where(side, b, a), torch.where(side, a, b)
    mean = torch.logaddexp(recorded, back(swapped)) - math.log(2.0)
    return torch.where(side, back(mean), mean)
