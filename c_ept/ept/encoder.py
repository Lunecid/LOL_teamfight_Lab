r"""EPT의 관계형 부호화기: S2 보기의 실체와 관계를 토큰 상태로 바꾼다.

설계서 5.1–5.3·6.1절, 계획 Q5·Q6, 판정 R-S4a-1을 구현한다. 보기는 기준 시각 하나에서 본 S2 한 행이다.
선수 10명, 표지점 34개, 최근 사건, 요약을 차례로 놓고 표준화 입력·가림·범주·보기 맥락과 학습 지형장을 더한다.
각 층은 유형·거리·측지·공동 참여 편향을 어텐션 점수에 더한다. 식과 기호의 전체 정의는 아래 소유 함수에 있다.
슬롯 번호 임베딩은 없으므로 같은 팀 선수의 재배열은 출력 선수 행의 재배열에 대응한다. 상류 간선 선택의 불변성은 별도이다.

식 목록
    (ENC-1) EPTEncoder.physical_positions: 진영 표시를 풀어 선수·사건과 표지점을 물리 좌표로 돌린다.
    (ENC-2) bilinear, TerrainField.sample: 네 칸의 쌍선형 보간으로 지형장을 읽는다.
    (ENC-3) EPTEncoder.forward: 실체 입력과 범주·맥락·지형 표본을 토큰으로 합친다.
    (ENC-4) EPTEncoder.forward: 앞 정규화·관계 편향 어텐션·잔차·GELU 층 뒤 토큰을 정규화한다.
    (ENC-5) RelationalBias.forward: 유형 상수와 거리·측지 거리의 학습 함수를 합친다.
    (ENC-6) Coparticipation.forward: 양의 척도별 지수 감쇠 간선을 합친다.
    (ENC-7) TerrainField.penalty: 이웃한 보행 칸의 학습 지형장 차이 제곱을 평균한다.

기본 크기는 계획 Q6의 S·M이며 탐색과 매끄러움 가중 선택은 S6이다. 편향은 모든 층이 공유한다.
선수–사건 유형은 참여 여부로 나눈다. 거리·측지 항은 선수가 낀 쌍에만 있고 방향 편향은 없다.
정적 지형 채널은 학습 채널에 이어 붙여 사상한다. 생략하려면 static=False를 명시한다.
채움 사건은 키와 입력에서 빠지고 출력은 0이다. 사건·간선 수는 STORE.json에서 온 배치 모양으로 읽는다.
부호화기 자체는 팀 교환 불변이 아니며 정확한 대칭은 평가 모드의 프레임 평균이 낸다.
CPU 결정적 모드의 재현은 같은 시드·스레드 수를 전제로 한다. 학습 거리·측지 항의 교환 비교는 반올림 오차를 허용한다.
입력: data.ViewTensors의 배치. 출력: Encoded(h_tok, h_pool, mask). 보조: static_channels(Terrain), token_layout(사건 칸 수).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md sections 5.1-5.3 (layers of the window, entities and relations, the terrain field) and
#      6.1 (encoder: typed attention, distance, geodesic and co-participation biases, terrain field).
# Ref: docs/superpowers/plans/2026-10-22-c-ept-s4a.md decisions Q3, Q5, Q6, Q12; ruling R-S4a-1 (the counts come from the STORE.json manifest, so the
#      encoder reads the shape of the batch), R-S4a-3 (the standardised position columns that the encoder reads are centred on W/2, so the swap is
#      a sign flip of the input), R-S4a-9 (the weights of an undirected edge kind are tied; design 5.2: a slot number is no role).
# Ref: Ying, C. et al. 2021, "Do Transformers Really Perform Bad for Graph Representation?" (the title of the arXiv record; structural terms added
#      to the attention scores as learned biases, one per head, shared by the layers [detail from memory, not re-read]); arXiv:2106.05234
#      (abstract read on the arXiv API, 2026-10-23).
# Ref: Liu, Z. et al. 2022, "Swin Transformer V2: Scaling Up Capacity and Resolution", CVPR 2022, pp. 11999-12009 (a continuous position bias:
#      a small network of log-spaced coordinates; the form of the distance bias here); doi:10.1109/CVPR52688.2022.01170 (Crossref record read
#      2026-10-23; the abstract names the log-spaced continuous position bias, arXiv:2111.09883).
# Ref: Hawkes, A. G. 1971, "Spectra of some self-exciting and mutually exciting point processes", Biometrika 58(1):83-90 (the exponential kernel of
#      the co-participation decay; the learned multi-scale sum is this project's, design 6.2 claim C3); doi:10.1093/biomet/58.1.83 (Crossref
#      record read 2026-10-23).
# Ref: Che, Z. et al. 2018, "Recurrent Neural Networks for Multivariate Time Series with Missing Values", Scientific Reports 8:6085 (a zero fill,
#      a mask channel and an age as inputs); doi:10.1038/s41598-018-24271-9, as in ``data``.
# Ref: Xiong, R. et al. 2020, "On Layer Normalization in the Transformer Architecture" (the normalisation before each branch, "pre-LN");
#      arXiv:2002.04745 (abstract read on the arXiv API, 2026-10-23).
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from c_ept import config as C
from c_ept.preprocessing import terrain, tokens
from c_ept.ept import data

N_PLAYERS, N_LANDMARKS = data.N_PLAYERS, data.N_LANDMARKS
N_KINDS = 3                                          # the kinds of a view: frame clock, M3 event clock, M4 event clock (``views/src``)
N_EVENT_TYPES = len(C.EVENT_TYPES)
N_TEAM_CODES = 3                                     # the team of an event: blue 0, red 1, none 2 (``tokens``)
N_ZONE_CODES = len(terrain.ZONES) + 1                # the zones 0 .. 9 and -1 (no position)
N_EDGE_KINDS = len(tokens.EDGE_KINDS)
UNDIRECTED_EDGE_KINDS = (tokens.EDGE_SIDE, tokens.EDGE_OBJECTIVE)      # S2 orders other participants by team then index; tying discards endpoint order (R-S4a-12)
PAIR_ALLY, PAIR_ENEMY, PAIR_LANDMARK, PAIR_EVENT, PAIR_ACTOR, PAIR_OTHER = range(6)       # the pair types c_ij (ENC-5)
N_PAIR_TYPES = 6
STATIC_CHANNELS = ('walkable', *(f'zone_{name}' for name in terrain.ZONES), 'wall_fraction')
_INIT_STD = 0.02                                     # the standard deviation of the starting values of embeddings and bias tables
_BIAS_HIDDEN = 16                                    # the width of the small network of a distance bias
_KAPPA_FIRST, _KAPPA_LAST = 5.0, 80.0                # seconds: the decay scales start on a geometric row from the first to the last
_REQUIRED = tuple(k for k in data.VIEW_KEYS if k != 'view')      # the view number is no input
_X_CHANNEL = data.PLAYER_NAN_COLUMNS.index('x')      # the mask channel that says a player has no position


@dataclass(frozen=True)
class EncoderConfig:
    """The size of an ``EPTEncoder``: the token width ``d``, the number of layers ``layers`` and of attention heads ``heads`` (``d`` a multiple of
    ``heads``), the ``dropout`` of the attention weights and of the two branches of a layer, the number ``kappas`` of decay scales of the
    co-participation bias and the number ``field_channels`` of learned channels of the terrain field (0: only the static channels). ValueError for
    a whole number below its minimum (1; 0 for ``field_channels``), a ``d`` that ``heads`` does not divide and a ``dropout`` outside [0, 1).

    해석: 기본값은 계획 Q6의 크기 S(d = 128, 층 4, 머리 4)이고, 크기 M(d = 256, 층 6, 머리 8)은 비용 점검에 함께 쓴다. 크기 탐색은 S6의 몫이다 (계획 Q6).
    드롭아웃 0.1은 탐색 전의 흔한 값이다.
    """
    d: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    kappas: int = 3
    field_channels: int = 8

    def __post_init__(self):
        for name in ('d', 'layers', 'heads', 'kappas'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f'{name} must be a whole number of at least 1, got {v!r}')
        if isinstance(self.field_channels, bool) or not isinstance(self.field_channels, int) or self.field_channels < 0:
            raise ValueError(f'field_channels must be a whole number of at least 0, got {self.field_channels!r}')
        if self.d % self.heads:
            raise ValueError(f'd = {self.d} must be a multiple of heads = {self.heads}')
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f'dropout must be in [0, 1), got {self.dropout!r}')


SIZES = MappingProxyType({'S': EncoderConfig(d=128, layers=4, heads=4), 'M': EncoderConfig(d=256, layers=6, heads=8)})


class TokenLayout(NamedTuple):
    """Where the tokens of a view stand: ``players``, ``landmarks`` and ``events`` are slices of the token axis, ``summary`` is its last number."""
    players: slice
    landmarks: slice
    events: slice
    summary: int


def token_layout(n_events: int) -> TokenLayout:
    """The layout of the tokens of a view with ``n_events`` event slots: the players 0-9, the landmarks 10-43, the events from 44, the summary last.

    해석: 머리가 선수 토큰을 이 번호로 읽으므로 토큰의 순서는 이 모듈의 계약이다.
    """
    fixed = N_PLAYERS + N_LANDMARKS
    return TokenLayout(slice(0, N_PLAYERS), slice(N_PLAYERS, fixed), slice(fixed, fixed + n_events), fixed + n_events)


class Encoded(NamedTuple):
    """The output of the encoder: ``h_tok`` (V, N_tok, d) the state of every token (0 for a padded one), ``h_pool`` (V, d) the state of the
    summary token, ``mask`` (V, N_tok) bool, True for a real token. N_tok = 10 + 34 + the number of event slots + 1 (``token_layout``).

    해석: 채움 토큰은 어떤 토큰의 어텐션에도 들지 않고 그 출력은 0이다. 사건을 합치는 머리는 마스크로 실제 토큰을 고르고, 풀링은 요약 토큰의 상태이다.
    """
    h_tok: torch.Tensor
    h_pool: torch.Tensor
    mask: torch.Tensor


class Positions(NamedTuple):
    """The physical places of the located tokens: ``xy`` (V, N_loc, 2) in map units for the players, the landmarks and the event slots (N_loc =
    10 + 34 + events), ``known`` (V, N_loc) bool, False for a token without a position (a missing position, a padded slot); the place of such
    a token is the middle of the map, W/2, whatever the batch holds there."""
    xy: torch.Tensor
    known: torch.Tensor


# ---- the learned terrain field ----------------------------------------------------------------------------------------------------------
def static_channels(terr: terrain.Terrain) -> np.ndarray:
    """The fixed channels of the terrain field, (75, 75, 12) float32 [iy, ix], at the centres of the cells: walkable (1.0 or 0.0), the ten zones
    (one-hot) and the wall fraction (``STATIC_CHANNELS``); ``terr`` is the ``Terrain`` of the S2 run.

    해석: 데이터가 그린 지형(보행 가능 격자, 구역 규칙, 주변 벽 비율)은 학습하지 않는 입력이고, 학습 채널이 그 위에서 지형의 나머지를 배운다.
    """
    centre = terrain.CELL * (np.arange(terrain.GRID) + 0.5)
    x, y = np.meshgrid(centre, centre)
    one_hot = terr.zone(x, y)[..., None] == np.arange(len(terrain.ZONES))
    return np.concatenate([terr.walk[..., None], one_hot, terr.wall_fraction(x, y)[..., None]], axis=-1).astype(np.float32)


def bilinear(table: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """The bilinear interpolation of ``table`` (75, 75, channels) [iy, ix] at the points ``xy`` (..., 2) in map units: (..., channels).

    수식
    ----
    (ENC-2)  p̃ = p / 200 − ½ ;  (ix₀, iy₀) = ⌊p̃⌋ ;  (F_x, F_y) = p̃ − ⌊p̃⌋ ;  ix₁ = ix₀ + 1, iy₁ = iy₀ + 1 (네 번호는 0 .. 74로 자른다)
             Ω(p) = (1 − F_y) [(1 − F_x) Ω[iy₀, ix₀] + F_x Ω[iy₀, ix₁]] + F_y [(1 − F_x) Ω[iy₁, ix₀] + F_x Ω[iy₁, ix₁]]

      기호            | 뜻                                              | 단위      | 범위          | 코드 변수
      p               | 점의 자리 (x, y)                                | 지도 단위 | 실수          | xy
      p̃               | 점의 연속 칸 좌표 (칸 중심에서 정수)            | 칸        | 실수          | g
      ix₀, iy₀, ix₁, iy₁ | 점을 둘러싼 칸의 번호 (잘라서 0 .. 74)        | 칸        | 0 .. 74       | x0, y0, x1, y1
      F_x, F_y        | 칸 사이에서 점의 소수 위치                      | —         | [0, 1)        | wx, wy
      Ω[iy, ix]       | 칸 (ix, iy)의 값 (한 칸은 200 지도 단위)        | 표의 값   | 실수          | table

    해석
    ----
    표의 값은 칸 중심에 놓이고, 점의 값은 둘러싼 네 중심의 값을 x와 y에서 각각 직선으로 이은 것이다. x와 y 각각에 선형이므로 칸 중심의 값이 그런
    함수인 장은 안쪽에서 정확히 재현된다. 칸 중심보다 바깥에서는 번호를 잘라 가장자리 칸의 값을 되풀이한다. 한 점에서 오는 기울기는 둘러싼 네 칸에만 간다.
    torch의 ``grid_sample``은 쓰지 않고 네 칸을 번호로 읽는다: CUDA에서 그 역전파는 결정적 모드에서 막히기 때문이다.

    근거
    ----
    설계서 5.3절(토큰마다 쌍선형 표본), 계획 Q6(학습 지형장), 계획 Q3(GPU 결정적 모드는 G-S4a에서 정한다), torch 문서 ``use_deterministic_algorithms``.
    """
    grid = table.shape[0]
    g = xy / terrain.CELL - 0.5
    low = torch.floor(g)
    wx, wy = (g - low)[..., 0:1], (g - low)[..., 1:2]
    low = low.long()
    x0, y0 = low[..., 0].clamp(0, grid - 1), low[..., 1].clamp(0, grid - 1)
    x1, y1 = (low[..., 0] + 1).clamp(0, grid - 1), (low[..., 1] + 1).clamp(0, grid - 1)
    flat = table.reshape(grid * grid, -1)
    top = (1 - wx) * flat[y0 * grid + x0] + wx * flat[y0 * grid + x1]
    bottom = (1 - wx) * flat[y1 * grid + x0] + wx * flat[y1 * grid + x1]
    return (1 - wy) * top + wy * bottom


class TerrainField(nn.Module):
    """The terrain field Ω: a learned vector of ``channels`` numbers on every cell of the 75 x 75 grid (all zero at the start) beside the fixed
    ``static`` channels (75, 75, S) with S >= 1, channel 0 the walkable flag (0.0 or 1.0), or none. ``sample(xy)`` gives the vector at points of the
    map, (..., channels + S); ``penalty()`` the smoothness penalty. ValueError for static channels of another shape, a non-finite value or a
    walkable flag that is not 0 or 1. Without static channels every cell is walkable.

    수식
    ----
    (ENC-2)  Ω[iy, ix] = [Ω_learn[iy, ix] · walk[iy, ix] ; Ω_static[iy, ix]],  Ω(p)는 ``bilinear``

      기호        | 뜻                                          | 단위     | 범위        | 코드 변수
      Ω_learn     | 칸마다 학습하는 벡터 (시작은 0)             | 표의 값  | 실수        | learned (75, 75, 채널 수)
      walk        | 칸이 보행 가능하면 1, 아니면 0              | —        | 0 또는 1    | walk
      Ω_static    | 고정 정적 채널                              | 표의 값  | 실수        | static (75, 75, S)

    해석
    ----
    학습 채널은 0에서 시작하므로 처음에는 정적 채널만이 토큰에 지형을 알려 주고, 학습이 진행되면서 지형의 나머지 구조를 채운다. 보행 불가 칸의 학습 값에
    walk = 0을 곱해 표본에서 빠지므로 그 칸에는 기울기가 가지 않는다. 장은 지도의 물리 좌표에 붙어 있어서 팀 교환 뒤에도 같은 자리는 같은 값을 갖는다.

    근거
    ----
    설계서 5.3절(학습 지형장 G6c), 계획 Q6.
    """

    def __init__(self, channels: int, static: np.ndarray | None = None):
        super().__init__()
        grid = terrain.GRID
        if static is None:
            walk, static = np.ones((grid, grid, 1), dtype=np.float32), np.zeros((grid, grid, 0), dtype=np.float32)
        else:
            static = np.asarray(static, dtype=np.float32)
            if static.ndim != 3 or static.shape[:2] != (grid, grid) or static.shape[2] < 1:
                raise ValueError(f'the static channels must have the shape ({grid}, {grid}, S) with S >= 1, got {static.shape}')
            if not np.isfinite(static).all() or not np.isin(static[..., 0], (0.0, 1.0)).all():
                raise ValueError('the static channels must be finite and their channel 0 (walkable) must be 0 or 1')
            walk = static[..., :1]
        self.learned = nn.Parameter(torch.zeros(grid, grid, channels))
        self.register_buffer('walk', torch.as_tensor(walk.copy()), persistent=False)
        self.register_buffer('static', torch.as_tensor(static.copy()), persistent=False)
        self.n_channels = channels + static.shape[2]

    def table(self) -> torch.Tensor:
        """(ENC-2): the table Ω (75, 75, channels + S)."""
        return torch.cat([self.learned * self.walk, self.static], dim=-1)

    def sample(self, xy: torch.Tensor) -> torch.Tensor:
        """(ENC-2): Ω(p) at the points ``xy`` (..., 2) in map units, (..., channels + S)."""
        return bilinear(self.table(), xy)

    def penalty(self) -> torch.Tensor:
        """The smoothness penalty R_Ω of the learned channels, a 0-d tensor; its weight is set in S6 (spec 5.3).

        수식
        ----
        (ENC-7)  R_Ω = (1 / N_adj) Σ_{(n, n′)} ‖Ω_learn[n] − Ω_learn[n′]‖²   (n, n′: 위아래·좌우로 이웃한 두 칸, 둘 다 보행 가능)

          기호   | 뜻                                              | 단위      | 범위      | 코드 변수
          R_Ω    | 매끄러움 벌점                                   | 표의 값²  | ≥ 0       | 반환값
          n, n′  | 한 칸의 (ix, iy)                                | 칸        | 0 .. 74   | learned의 앞 두 축
          N_adj  | 둘 다 보행 가능한 이웃 쌍의 수 (0이면 1로 나눈다) | 쌍      | ≥ 0       | across.sum() + down.sum()

        해석
        ----
        이웃한 칸의 학습 벡터가 크게 다르면 벌점이 커지므로 지형장은 매끄러운 쪽으로 당겨진다. 벽 칸이 낀 쌍은 빠지므로 벽 칸의 값이 이웃 칸을 끌어당기지
        않는다(벽 칸의 값은 표본에도 들지 않는다). 쌍이 하나도 없으면 벌점은 0이다.

        근거
        ----
        설계서 5.3절(매끄러움 벌점, 가중은 S6).
        """
        v, w = self.learned, self.walk[..., 0]
        across, down = w[:, 1:] * w[:, :-1], w[1:] * w[:-1]
        total = ((v[:, 1:] - v[:, :-1]).square().sum(-1) * across).sum() + ((v[1:] - v[:-1]).square().sum(-1) * down).sum()
        return total / (across.sum() + down.sum()).clamp_min(1.0)


# ---- the relational attention bias --------------------------------------------------------------------------------------------------------
def _landmark_places() -> torch.Tensor:
    """(2, 34, 2) float64: the physical place of landmark row j when side is 0 and when it is 1 (the row then holds landmark LANDMARK_PERM[j])."""
    table = np.asarray(tokens.LANDMARK_XY, dtype=np.float64)
    return torch.as_tensor(np.stack([table, table[data.LANDMARK_PERM]]))


class _Scalar(nn.Module):
    """A small network from one number to one bias per head (the MLP of (ENC-5))."""

    def __init__(self, heads: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, _BIAS_HIDDEN), nn.GELU(), nn.Linear(_BIAS_HIDDEN, heads))
        nn.init.normal_(self.net[2].weight, std=_INIT_STD)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[..., None])


class Coparticipation(nn.Module):
    """The co-participation bias: the decay scales ``log_kappa`` (``kappas`` numbers, ``kappa`` = their exponentials, all positive) and the weights
    ``w`` and ``w_back`` (heads, edge kinds, scales). An edge adds ``w`` of its kind to the entry [i_e, j_e] of its two persons and ``w_back`` to
    [j_e, i_e]; the kinds of ``UNDIRECTED_EDGE_KINDS`` add ``w`` to both (their rows of ``w_back`` start at 0 and are never read).

    해석: 척도는 로그로 두어 학습이 곱셈으로 움직이고 늘 양수이다. 시작은 5 s(킬 자극 감쇠 5.3 s, 설계서 5.2절 근거)에서 80 s(발생 반등 지연 60–120 s,
    창 120 s) 사이의 등비 수열이고, 가중은 작은 난수로 시작한다.
    """

    def __init__(self, heads: int, kappas: int):
        super().__init__()
        start = [math.sqrt(_KAPPA_FIRST * _KAPPA_LAST)] if kappas == 1 else [
            _KAPPA_FIRST * (_KAPPA_LAST / _KAPPA_FIRST) ** (r / (kappas - 1)) for r in range(kappas)]
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(start)))
        self.w = nn.Parameter(_INIT_STD * torch.randn(heads, N_EDGE_KINDS, kappas))
        self.w_back = nn.Parameter(_INIT_STD * torch.randn(heads, N_EDGE_KINDS, kappas))
        undirected = torch.zeros(N_EDGE_KINDS, dtype=torch.bool)
        undirected[list(UNDIRECTED_EDGE_KINDS)] = True
        self.register_buffer('undirected', undirected, persistent=False)
        with torch.no_grad():
            self.w_back[:, undirected] = 0.0

    @property
    def kappa(self) -> torch.Tensor:
        """κ = exp(log κ) of (ENC-6): the decay scales, (kappas,), positive."""
        return self.log_kappa.exp()

    def decay(self, age: torch.Tensor) -> torch.Tensor:
        """exp(−Δt / κ) of (ENC-6) for the ages ``age`` (...) in seconds: (..., kappas), in (0, 1] for an age >= 0."""
        return torch.exp(-age[..., None] / self.kappa.clamp_min(torch.finfo(self.log_kappa.dtype).tiny))

    def forward(self, batch: Mapping) -> torch.Tensor:
        """The co-participation bias of the player pairs of every view, (V, heads, 10, 10), from ``ed_valid``, ``ed_i``, ``ed_j``,
        ``ed_kind`` and ``ed_dt``.

        수식
        ----
        (ENC-6)  b_co,ij = Σ_{e: (i_e, j_e) = (i, j)} Σ_κ w_{k_e,κ} exp(−Δt_e / κ)  +  Σ_{e: (i_e, j_e) = (j, i)} Σ_κ w′_{k_e,κ} exp(−Δt_e / κ)
                 w′_{k,κ} = w_{k,κ}  (k = 킬 편 쌍, 오브젝트 쌍: 판정 R-S4a-9) ;  κ = exp(log κ), 합은 학습하는 감쇠 척도 전부에 대해

          기호        | 뜻                                              | 단위  | 범위      | 코드 변수
          b_co,ij     | 선수 쌍 (i, j)의 공동 참여 편향 (머리마다)      | 편향  | 실수      | 반환값
          e           | 한 보기의 유효한 간선 (채움 칸은 뺀다)          | —     | —         | ed_valid
          i_e, j_e    | 간선 e의 첫째·둘째 사람 (선수 슬롯)             | —     | 0 .. 9    | ed_i, ed_j
          k_e         | 간선의 종류 (킬러–피해자, 도움–피해자, 킬 편 쌍, 피해 준 사람–피해자, 오브젝트 쌍) | — | 0 .. 4 | ed_kind
          Δt_e        | 간선의 나이 (기준 시각까지)                     | s     | ≥ 0       | ed_dt
          κ           | 감쇠 척도 (척도 수만큼; κ = exp(log κ))         | s     | > 0       | kappa
          w, w′       | 머리·종류·척도 κ별 정방향, 역방향 가중          | 편향  | 실수      | w, w_back (방향 있는 종류만)

        해석
        ----
        학습한 세 척도의 지수 감쇠 합으로 짧고 긴 공동 참여 기억을 나타낸다.
        킬러–피해자, 도움–피해자, 피해 준 사람–피해자 간선은 끝점 순서가 역할이므로 방향별 가중을 쓴다.
        S2는 킬러를 앞에 두고 나머지 참가자를 팀, 팀 안 번호 순으로 적는다.
        킬 편 쌍과 오브젝트 쌍의 가중을 묶으면 끝점 순서 정보가 사라지며 오브젝트의 마지막 타격자는 이 간선에서 식별되지 않는다.
        채움 간선은 0이고,
        같은 쌍의 여러 간선은 더해진다.

        근거
        ----
        설계서 5.2절(공동 참여 간선; 슬롯 번호는 역할이 아니다), 6.1절(Σ exp(−Δt / κ)), 11절 14항(간선 목록, κ는 학습), 계획 Q6(κ 척도 3개), 판정 R-S4a-9
        (순서가 슬롯 순위인 종류는 방향을 묶는다). 지수 핵: Hawkes 1971 (위 Ref).
        """
        valid = batch['ed_valid']
        decay = torch.where(valid[..., None], self.decay(torch.where(valid, batch['ed_dt'], 0.0)), 0.0)          # (V, edges, scales)
        kind = torch.where(valid, batch['ed_kind'], 0)
        w_back = torch.where(self.undirected[None, :, None], self.w, self.w_back)                               # (heads, kinds, scales)
        forward = torch.einsum('hvdr,vdr->vdh', self.w[:, kind], decay)                                          # Σ_κ w exp(−Δt / κ) per edge
        backward = torch.einsum('hvdr,vdr->vdh', w_back[:, kind], decay)
        first = F.one_hot(torch.where(valid, batch['ed_i'], 0), N_PLAYERS).to(decay.dtype) * valid[..., None]
        second = F.one_hot(torch.where(valid, batch['ed_j'], 0), N_PLAYERS).to(decay.dtype) * valid[..., None]
        return torch.einsum('vdp,vdh,vdq->vhpq', first, forward, second) + torch.einsum('vdp,vdh,vdq->vhpq', second, backward, first)


class RelationalBias(nn.Module):
    """The attention bias between all tokens of a view: a table ``type_bias`` (heads, 6) of constants for the pair types, the small networks
    ``dist`` (ally, enemy, landmark, event) and ``geo`` (ally, enemy, landmark) of the distance and the geodesic bias, and ``co``
    (``Coparticipation``). ``forward(positions, batch)`` gives the bias (V, heads, N_tok, N_tok), one value per head, for every pair of tokens.
    """

    def __init__(self, heads: int, kappas: int):
        super().__init__()
        self.heads = heads
        self.type_bias = nn.Parameter(_INIT_STD * torch.randn(heads, N_PAIR_TYPES))
        self.dist = nn.ModuleList([_Scalar(heads) for _ in range(4)])
        self.geo = nn.ModuleList([_Scalar(heads) for _ in range(3)])
        self.co = Coparticipation(heads, kappas)
        group = torch.arange(N_PLAYERS) // (N_PLAYERS // 2)
        self.register_buffer('ally', group[:, None] == group[None, :], persistent=False)

    def forward(self, positions: Positions, batch: Mapping) -> torch.Tensor:
        """The bias of (ENC-4) for every ordered pair of tokens (i, j) of every view.

        수식
        ----
        (ENC-5)  b_type,ij = U[c_ij]
                 b_dist,ij = MLP^dist_{c_ij}(log(1 + ‖p_i − p_j‖ / L_U))     (선수가 낀 쌍, 두 자리를 모두 알 때; 아니면 0)
                 b_geo,ij  = MLP^geo_{c_ij}(x̃^geo_ij)                       (선수–선수, 선수–표지점 쌍, 측지 칸이 가려지지 않았을 때; 아니면 0)
                 (선수–사건의 두 유형, 비참여와 참여는 같은 거리 망 dist[3]을 쓴다)
                 반환값 = b_type,ij + b_dist,ij + b_geo,ij + b_co,ij          (b_co는 선수–선수 쌍에만; 쌍 (i, j)와 (j, i)의 유형·거리·측지 항은 같다)

          기호         | 뜻                                                          | 단위       | 범위      | 코드 변수
          c_ij         | 쌍 유형: 아군, 적, 선수–표지점, 선수–사건(비참여·참여), 그 밖 | —          | 0 .. 5    | PAIR_*
          U[c]         | 유형 c의 학습 상수 (머리마다)                               | 점수 단위  | 실수      | type_bias
          p_i          | 토큰 i의 물리 좌표                                          | 지도 단위  | 0 .. W    | positions.xy
          L_U          | 교전 연결 거리 2,000                                        | 지도 단위  | —         | C.L_U
          MLP^dist_c   | 유형 c(아군, 적, 표지점, 사건)의 작은 망 (1 → 16 → 머리 수) | 점수 단위  | 실수      | dist[0 .. 3]
          x̃^geo_ij     | 표준화한 측지 거리 (data의 geo_pp, geo_pl)                  | 무차원     | 실수      | geo_pp, geo_pl
          MLP^geo_c    | 유형 c(아군, 적, 표지점)의 작은 망                          | 점수 단위  | 실수      | geo[0 .. 2]
          b_co,ij      | ``Coparticipation``의 값 (ENC-6)                            | 점수 단위  | 실수      | self.co(batch)

        해석
        ----
        어텐션 점수는 내용 점수에 쌍의 관계가 더해진다: 같은 편인지, 얼마나 가까운지, 걸어서 얼마나 먼지, 최근에 함께 엮였는지이다. 거리는 교전 연결 거리를
        단위로 로그를 씌워 가까운 거리를 세밀하게 보고, 쌍 유형마다 따로 둔 작은 망이 머리마다 그 거리의 영향을 학습한다. 위치를 모르는 선수·사건과 채움 사건은
        거리 항이, 가려진 측지 칸은 측지 항이 0이고, 사건 쌍에는 측지 항이 없다. 모든 값이 두 토큰의 물리 좌표, 측지 칸, 간선과 참여 표시에서만 나오므로
        팀 교환은 쌍의 번호만 바꾸고 값은 바꾸지 않는다.

        근거
        ----
        설계서 5.2·6.1절(아군·적 유형, 세 편향), 계획 Q6(편향 네 가지), 법칙 A1(팀 교환), A9(편향 하나의 연산). 작은 망: Swin V2, 점수에 더하기: Graphormer (위 Ref).
        """
        p, heads = N_PLAYERS, self.heads
        n_view, n_events = batch['ev_valid'].shape
        n_all = p + N_LANDMARKS + n_events + 1
        tb = self.type_bias
        # Euclidean distance of every player to every located token, in units of the link distance (an unknown place is the middle of the
        # map, so no junk reaches the network; the term is switched off for such a pair below)
        xy, known = positions.xy, positions.known
        near = torch.log1p(torch.linalg.vector_norm(xy[:, :p, None, :] - xy[:, None, :, :], dim=-1) / C.L_U)                  # (V, 10, N_loc)
        located = known[:, :p, None] & known[:, None, :]
        d_pp = torch.where(self.ally[..., None], self.dist[0](near[:, :, :p]), self.dist[1](near[:, :, :p]))
        dist = torch.where(located[..., None], torch.cat([d_pp, self.dist[2](near[:, :, p:p + N_LANDMARKS]),
                                                          self.dist[3](near[:, :, p + N_LANDMARKS:])], dim=2), 0.0)          # (V, 10, N_loc, heads)
        # geodesic distance (player-player, player-landmark): a masked cell is not read; events have none
        ok_pp, ok_pl = batch['geo_pp_nan'] < 0.5, batch['geo_pl_nan'] < 0.5
        x_pp, x_pl = torch.where(ok_pp, batch['geo_pp'], 0.0), torch.where(ok_pl, batch['geo_pl'], 0.0)
        g_pp = torch.where(self.ally[..., None], self.geo[0](x_pp), self.geo[1](x_pp))
        geo = torch.cat([torch.where(ok_pp[..., None], g_pp, 0.0), torch.where(ok_pl[..., None], self.geo[2](x_pl), 0.0),
                         dist.new_zeros(n_view, p, n_events, heads)], dim=2)                                                   # (V, 10, N_loc, heads)
        # the pair types: the teams are the two blocks of five player rows; a player takes part in an event if the event lists him
        actor = batch['ev_valid'][:, None, :] & (batch['ev_actors'].transpose(1, 2) > 0.5)                                    # (V, 10, events)
        type_pp = torch.where(self.ally, tb[:, PAIR_ALLY][:, None, None], tb[:, PAIR_ENEMY][:, None, None])                  # (heads, 10, 10)
        type_pl = tb[:, PAIR_LANDMARK][:, None, None].expand(heads, p, N_LANDMARKS)
        type_pe = torch.where(actor[:, None], tb[:, PAIR_ACTOR][None, :, None, None], tb[:, PAIR_EVENT][None, :, None, None])  # (V, heads, 10, E)
        other = tb[:, PAIR_OTHER][None, :, None, None]
        # the rows of the players (query = player, key = any token); the other rows are their transpose, and the pairs without a player have the
        # constant of the type "other" only
        rows = torch.cat([
            type_pp[None] + dist[:, :, :p].permute(0, 3, 1, 2) + geo[:, :, :p].permute(0, 3, 1, 2) + self.co(batch),
            type_pl[None] + dist[:, :, p:p + N_LANDMARKS].permute(0, 3, 1, 2) + geo[:, :, p:p + N_LANDMARKS].permute(0, 3, 1, 2),
            type_pe + dist[:, :, p + N_LANDMARKS:].permute(0, 3, 1, 2),
            other.expand(n_view, heads, p, 1)], dim=-1)                                                                        # (V, heads, 10, N_tok)
        rest = torch.cat([rows[..., p:].transpose(-1, -2), other.expand(n_view, heads, n_all - p, n_all - p)], dim=-1)
        return torch.cat([rows, rest], dim=2)


# ---- the encoder ----------------------------------------------------------------------------------------------------------------------------
class _Block(nn.Module):
    """One layer of (ENC-4): attention with the bias, then the feed-forward branch, each after a normalisation and with a residual connection."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.attention_dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        v, n, d = h.shape
        q, k, val = self.qkv(self.norm1(h)).view(v, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, val, attn_mask=bias, dropout_p=self.attention_dropout if self.training else 0.0)
        h = h + self.dropout(self.out(a.transpose(1, 2).reshape(v, n, d)))
        return h + self.dropout(self.ffn(self.norm2(h)))


class EPTEncoder(nn.Module):
    """The relational encoder of the EPT. ``cfg`` is an ``EncoderConfig``. ``static`` is required: the fixed channels of the terrain field
    (``static_channels(Terrain)``, 75 x 75 x S, channel 0 the walkable flag) or ``False`` for an encoder without them (no walkable mask, no zones,
    no wall fraction); leaving it out is a TypeError and None or True a ValueError, so that an encoder never loses them unasked.
    ``forward(batch)`` takes the tensors of ``data.ViewTensors`` (the keys of ``data.VIEW_KEYS`` but ``view``; other keys, such as the row blocks
    of a batch, are ignored) and returns an ``Encoded``; the number of event slots and of edges is read from the shapes. ValueError for a missing
    key.

    Parts: ``bias`` (``RelationalBias``), ``field`` (``TerrainField``, None if there is no learned and no static channel) and the layers in
    ``blocks``. ``attention_bias(batch)`` gives the bias of a batch and ``physical_positions(batch)`` the places that distances and the field read.
    """

    def __init__(self, cfg: EncoderConfig, static: np.ndarray | bool):
        super().__init__()
        if static is None or static is True:
            raise ValueError('the static channels of the terrain field are required: pass static_channels(Terrain) (75 x 75 x S), or '
                             'static=False for an encoder without them (no walkable mask, no zones, no wall fraction)')
        fixed = None if static is False else static
        d = cfg.d
        self.cfg = cfg
        self.players_in = nn.Linear(len(data.PLAYER_COLUMNS) + len(data.PLAYER_NAN_COLUMNS), d)
        self.landmarks_in = nn.Linear(3, d)
        self.events_in = nn.Linear(5, d)
        self.summary_in = nn.Linear(len(data.SUMMARY_COLUMNS) + len(data.SUMMARY_NAN_COLUMNS), d)
        self.context_in = nn.Linear(2, d)
        self.zone_emb, self.group_emb = nn.Embedding(N_ZONE_CODES, d), nn.Embedding(2, d)
        self.landmark_emb, self.kind_emb = nn.Embedding(N_LANDMARKS, d), nn.Embedding(N_KINDS, d)
        self.type_emb, self.team_emb = nn.Embedding(N_EVENT_TYPES, d), nn.Embedding(N_TEAM_CODES, d)
        for emb in (self.zone_emb, self.group_emb, self.landmark_emb, self.kind_emb, self.type_emb, self.team_emb):
            nn.init.normal_(emb.weight, std=_INIT_STD)
        self.register_buffer('group', torch.arange(N_PLAYERS) // (N_PLAYERS // 2), persistent=False)       # the team block of a player row
        self.register_buffer('landmark_xy', _landmark_places(), persistent=False)
        self.field = TerrainField(cfg.field_channels, fixed) if cfg.field_channels or fixed is not None else None
        if self.field is not None:
            self.field_proj = nn.Linear(self.field.n_channels, d, bias=False)
        self.bias = RelationalBias(cfg.heads, cfg.kappas)
        self.blocks = nn.ModuleList([_Block(d, cfg.heads, cfg.dropout) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(d)

    def physical_positions(self, batch: Mapping) -> Positions:
        """The places of the players, the landmarks and the event slots on the map that the teams share.

        수식
        ----
        (ENC-1)  p_i = p^in_i (side = 0),  p_i = W − p^in_i (side = 1)    선수·사건 토큰 i
                 p_j = X_j (side = 0),  p_j = X_{j′} (side = 1),  j′ = LANDMARK_PERM[j]    표지점 행 j

          기호  | 뜻                                                     | 단위      | 범위     | 코드 변수
          p^in_i   | 배치의 틀에서 잰 토큰 i의 자리 (x, y)                  | 지도 단위 | 0 .. W   | players_xy, ev_xy
          p_i   | 토큰 i의 물리 좌표                                     | 지도 단위 | 0 .. W   | 반환값 xy
          W     | 지도 폭 14,875 (180° 회전의 상수)                      | 지도 단위 | —        | data.W
          X_j   | 표지점 행 j의 물리 자리                                | 지도 단위 | 0 .. W   | tokens.LANDMARK_XY
          side  | 진영 표시 (기록대로 0, 팀 교환 뒤 1)                   | —         | 0 또는 1 | side

        해석
        ----
        팀 교환은 위치를 180° 돌리고 진영 표시를 뒤집으므로, 진영 표시가 1이면 W − p^in_i가 돌리지 않은 지도 위의 자리이다. 교환 뒤 표지점 행은 반대편 건물의
        자료를 담으므로 그 건물의 자리를 쓰고, 둥지는 교환에서 제자리여서 자리도 그대로이다. 거리와 지형장을 모두 이 물리 좌표로 재므로 같은 토큰은 팀
        교환 뒤에도 같은 자리를 가리킨다. 자리를 모르는 선수와 사건, 채움 칸은 위치를 모르는 토큰으로 표시하고 그 자리는 지도 가운데 W/2로 둔다.

        근거
        ----
        설계서 5.2절(180° 회전 + 진영 표시), 5.3절(절대 좌표), 법칙 A1, ``symmetry`` 모듈(건물은 맞바뀌고 둥지는 제자리).
        """
        flipped = batch['side'] > 0.5
        turn = lambda xy: torch.where(flipped[:, None, None], data.W - xy, xy)                              # noqa: E731
        landmarks = self.landmark_xy[flipped.long()].to(batch['players_xy'].dtype)
        known_players = batch['players_nan'][..., _X_CHANNEL] < 0.5
        known_events = batch['ev_valid'] & (batch['ev_nan'][..., 0] < 0.5)
        xy = torch.cat([turn(batch['players_xy']), landmarks, turn(batch['ev_xy'])], dim=1)
        known = torch.cat([known_players, torch.ones_like(known_players[:, :1]).expand(-1, N_LANDMARKS), known_events], dim=1)
        return Positions(torch.where(known[..., None], xy, torch.full_like(xy, data.W / 2.0)), known)       # no junk of an unknown place goes on

    def attention_bias(self, batch: Mapping) -> torch.Tensor:
        """The bias (V, heads, N_tok, N_tok) of (ENC-4) for a batch, without the mask of the padded keys: ``RelationalBias`` at the physical
        places."""
        return self.bias(self.physical_positions(batch), batch)

    def _embed(self, batch: Mapping, positions: Positions) -> torch.Tensor:
        """The token input h of (ENC-3), (V, N_tok, d)."""
        valid = batch['ev_valid']
        h_players = (self.players_in(torch.cat([batch['players'], batch['players_nan']], -1)) + self.zone_emb(batch['players_zone'] + 1)
                     + self.group_emb(self.group))
        h_landmarks = self.landmarks_in(torch.cat([batch['landmarks'], batch['landmarks_nan']], -1)) + self.landmark_emb.weight
        events = torch.where(valid[..., None], torch.cat([batch['ev_num'], batch['ev_nan']], -1), 0.0)           # a padded slot is never read
        h_events = (self.events_in(events) + self.type_emb(torch.where(valid, batch['ev_type'], 0))
                    + self.team_emb(torch.where(valid, batch['ev_team'], 2)))
        h_summary = self.summary_in(torch.cat([batch['summary'], batch['summary_nan']], -1))[:, None]
        context = self.kind_emb(batch['kind']) + self.context_in(torch.stack([batch['age'], batch['side']], -1))
        located = torch.cat([h_players, h_landmarks, h_events], dim=1)
        if self.field is not None:
            located = located + torch.where(positions.known[..., None], self.field_proj(self.field.sample(positions.xy)), 0.0)
        return torch.cat([located, h_summary], dim=1) + context[:, None]

    def forward(self, batch: Mapping) -> Encoded:
        """Encode the views of ``batch``: the token input (ENC-3), the bias (ENC-5, ENC-6) and the layers (ENC-4).

        수식
        ----
        (ENC-3)  h_i = P_cls(i) [x̃_i ; ν_i] + E_i + Z_v + 1[p_i를 안다] · P_fld Ω(p_i)
        (ENC-4)  α_ij = softmax_j((P_qry h̃_i)·(P_key h̃_j) / √d_k + b_type,ij + b_dist,ij + b_geo,ij + b_co,ij)   (j는 실제 토큰만)
                 h_i ← h_i + P_out Σ_j α_ij P_val h̃_j ;   h_i ← h_i + FFN(LN(h_i)) ;   h̃ = LN(h) (층마다)   머리마다 편향이 다르다
                 반환 h_tok,i = LN(h_i) (실제 토큰), 채움 토큰은 0   (마지막 층 뒤의 정규화 하나)

          기호             | 뜻                                                              | 단위   | 범위   | 코드 변수
          h_i              | 토큰 i의 상태 (반환하는 값은 마지막 정규화를 거친 h_tok,i)      | —      | 실수   | h (반환값 h_tok)
          h̃_i              | 층 정규화한 상태 LN(h_i) (어텐션 앞, 망 앞에서 각각)            | —      | 실수   | norm1(h), norm2(h)
          α_ij             | 토큰 i가 토큰 j에 두는 어텐션 가중 (머리마다; j에 대해 합 1)   | —      | [0, 1] | softmax의 값
          P_cls(i)         | 토큰 종류(선수, 표지점, 사건, 요약)별 입력 사상                 | —      | 행렬   | players_in, landmarks_in, events_in, summary_in
          x̃_i, ν_i         | 표준화한 입력 행과 가림 채널 (data의 x̃, ν)                      | —      | 실수   | players, landmarks, ev_num, summary 와 *_nan
          E_i              | 범주 임베딩의 합: 선수 구역·팀 묶음, 표지점 번호, 사건 종류·팀    | —      | 실수   | zone_emb, group_emb, landmark_emb, type_emb, team_emb
          Z_v              | 보기 맥락: 보기 종류 임베딩 + 선형 사상(프레임 나이 a, 진영 표시)   | —      | 실수   | kind_emb, context_in
          P_fld Ω(p_i)     | 지형장 표본의 선형 사상 (위치를 아는 토큰만)                    | —      | 실수   | field_proj, field.sample
          P_qry, P_key, P_val, P_out | 어텐션의 질의·키·값·출력 사상                          | —      | 행렬   | block.qkv, block.out
          d_k              | 머리 하나의 폭 d / 머리 수 (과제 설명서의 √d의 d는 이 폭이다)   | —      | ≥ 1    | d // heads
          b_type … b_co    | 관계 편향 (ENC-5, ENC-6)                                        | 점수   | 실수   | attention_bias
          FFN, LN          | 폭 4d의 GELU 망, 층 정규화                                      | —      | —      | block.ffn, block.norm1, block.norm2

        해석
        ----
        각 토큰은 자기 입력과 범주, 보기 맥락, 땅의 학습된 성질을 더한 벡터로 시작해, 층마다 다른 토큰을 내용 점수와 관계 편향으로 가중해 읽는다. 선수 슬롯
        번호는 입력으로도 편향으로도 들어가지 않는다: 간선의 방향은 순서가 역할인 종류에만 있고 순서가 참가자 번호의 순위인 종류는 두 방향이 같으므로, 한 팀
        안에서 선수를 바꿔 쓰면 선수 토큰의 출력이 같은 순서로 바뀔 뿐이다. 프레임 나이와 보기 종류, 진영 표시가 입력이어서 같은 망이 프레임 시계와 사건 시계
        보기를 함께 다룬다. 채움 사건 칸은 키에서 빠지고 출력이 0이며, 요약 토큰의 상태가 풀링이다.

        근거
        ----
        설계서 4절(프레임 나이, 두 시계 한 망), 5.1–5.3절, 6.1절, 계획 Q5(나이·보기 종류 입력), Q6(구조), 판정 R-S4a-1(개수는 STORE.json에서 읽고 이 모듈은
        배치의 모양에서 읽는다), R-S4a-9(순서가 슬롯 순위인 간선 종류는 방향을 묶는다). 앞 정규화: Xiong 2020 (위 Ref).
        """
        missing = [k for k in _REQUIRED if k not in batch]
        if missing:
            raise ValueError(f'the batch lacks the view tensors {missing}')
        valid = batch['ev_valid']
        mask = torch.cat([valid.new_ones(valid.shape[0], N_PLAYERS + N_LANDMARKS), valid, valid.new_ones(valid.shape[0], 1)], dim=1)
        positions = self.physical_positions(batch)
        h = self._embed(batch, positions)
        # a padded key gets −∞ and so no weight; the bias is shared by all layers
        pad = torch.zeros_like(mask, dtype=h.dtype).masked_fill(~mask, float('-inf'))[:, None, None, :]
        bias = self.bias(positions, batch) + pad
        for block in self.blocks:
            h = block(h, bias)
        h = torch.where(mask[..., None], self.norm(h), 0.0)
        return Encoded(h, h[:, -1], mask)
