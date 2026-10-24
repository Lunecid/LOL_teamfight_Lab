r"""Terrain that the data itself draws: the walkable grid of the FIT positions, its geodesic table, zones and the local wall fraction.

Inputs. Only the S1 store is read, and only the FIT matches of it (so nothing from DEV, CAL or AUDIT shapes an input map [S2-13]). Raw
match files are never opened. ``scan_positions`` counts every player position of the regular frames (``frames/num`` x, y), of the game-end
frame (``matches/end_xy``) and of the kills (``kills/x, y``) in a 150 x 150 histogram of 100 u cells on [0, 15,000]^2, exactly as the
prototype ``terrain_occupancy_probe.py`` does (``np.histogram2d``, result indexed [ix, iy]); positions that are NaN or lie outside the
range are not binned. ``walkable`` pools 2 x 2 of those cells to the 75 x 75 grid of 200 u cells, indexed [iy, ix], with the cell number
75 iy + ix, and calls a cell walkable when it holds at least ``MIN_COUNT`` = 150 hits (S2 gate S2G-2, 2026-10-21; it was 5: the hits per
cell have two modes, and the walls sit in the low one). Cell (ix, iy) covers x in [200 ix, 200 ix + 200).

Graph and geodesic table. Only walkable cells are nodes. A cell is joined to its four neighbours at 200 u, and to a diagonal neighbour at
200 sqrt(2) u only when both orthogonal neighbours are walkable as well (no cutting of corners). ``geodesic_table`` runs Dijkstra from every
walkable cell and keeps the distances rounded to whole map units in a (5625, 5625) uint16 array; a wall cell and an unreachable pair are
65,535 (``UNREACHABLE``). The cast is range-checked: a reachable distance of 65,535 or more raises (the open grid's longest is 20,930).
``casts.narrow`` refuses every float-to-integer change (the distances are rounded float64) and has no unsigned target (and ``casts.py``
is part of the S1 golden lock), so this one cast has its own check.

Islands. A data-derived grid can hold small components that are cut off from the main map. ``components`` labels them (the same edges as
the geodesic graph) and picks the largest (a tie: the one with the lowest cell number). ``Terrain.snap`` puts a position on the walkable
cell whose centre is nearest (Euclidean; a tie goes to the smaller cell number), but only among the cells of the largest component, so the
distance between two snapped positions is always finite; ``Terrain.geo`` raises if it ever finds 65,535 [R11-b]. A position that lies off
the map or in a wall is snapped like any other, and is never dropped [S2-9]. A missing position (NaN) gives -1 (``snap``, ``zone``) or
NaN (``geo``, ``wall_fraction``); an infinite one raises.

Zones. ``ZONE_RULES`` holds the thresholds in coordinates normalised by the swap constant, xn = pos_x / 14,875 and yn = pos_y / 14,875,
because the
team swap of the views maps pos_x to 14,875 - pos_x, that is xn to 1 - xn [R11-a]. The rules are applied in this order, with strict
inequalities:

    base_blue   xn + yn < 0.36                  base_red    xn + yn > 1.64
    top         xn < 0.13 or yn > 1 - 0.13      bot         yn < 0.13 or xn > 1 - 0.13
    mid         |xn - yn| < 0.07                river       |xn + yn - 1| < 0.07
    jungle      the rest: xn < yn (xn - yn < 0) is the top half and xn > yn the bottom half, xn + yn < 1 the blue and xn + yn > 1 the
                red side

A rotation by 180 degrees (xn, yn) -> (1 - xn, 1 - yn) turns base_blue into base_red, top into bot, mid and river into themselves and each
jungle quarter into the opposite one, so the rule set is symmetric. ``ZONE_SWAP`` is not written by hand: it is derived at import from
the rules, on a lattice of the map and its mirror image, and the import fails if the mapping is not a permutation. The rules are
evaluated in map units from the centre of the map, xc = pos_x - map_width / 2 and yc = pos_y - map_width / 2 with thresholds multiplied by
map_width: that is the same
rule, but a mirror image negates xc and yc exactly, so a position and its mirror image always get swapped zones, also on a threshold
line (whole numbers sit on x + y = 5,355). The thresholds are fixed by the brief; the S2 report checks them on a map picture (spec
section 11, item 12).

Wall fraction. For a position, the cell that contains it (floor(x / 200), floor(y / 200)); the share of the 5 x 5 window around that
cell that is not walkable, over the whole grid, cells outside the 75 x 75 grid counted as walls. There is no snapping here [R11-c].

``build_terrain(s1_run, run)`` writes ``<run>/terrain/{terrain.npz, geodesic.npy, TERRAIN.json}``. It checks that no model server runs
first [S2-1], refuses a ``run`` that is the S1 run folder or lies inside it (an S1 run is read only; the resolved paths are compared and
so is the identity of the folders, which sees a relative name, a junction, the ``\\?\`` prefix and an administrative share of this
machine), refuses to overwrite a terrain folder, reads the FIT rows of the S1 run, and records in TERRAIN.json: the S1 run name, the
SHA-256 of the sorted ``files`` map of its STORE.json (``runlog.sha256_json``), the number of FIT matches, the counts of the components,
the counts of suspect positions (exactly (0, 0), outside [0, 14,875]^2, NaN: per regular frames, game-end frame and kills), the SHA-256
of the two data files, and the git commit and dirty flag (``runlog.code_state``; the helpers that write RUN.json, so the two records
agree). No match key is read into any output. The archive is written without time stamps, so the same inputs give the same bytes.
``Terrain(path)`` opens such a folder; the table is a read-only memmap. Every cast to a narrower dtype goes through ``casts.narrow``
[S2-4], except the cast of the geodesic table to uint16, which has its own range check (``_to_uint16``): ``narrow`` refuses every
float-to-integer change (the rounded distances are float64) and has no unsigned target, and either reason alone rules it out. This
module imports no S1 module at module level (``c_ept.run_s1`` is imported inside ``build_terrain``; ``c_ept.runlog`` loads none of
them).

방법에서의 역할
---------------
이 모듈은 데이터가 그리는 지형을 만든다. FIT 경기의 선수 위치와 킬 위치를 센 점유 히스토그램에서 보행 가능 격자를 정하고, 격자 위의 8방향 최단 경로로 칸 사이
측지 거리표를 만들고, 위치를 구역과 국소 벽 비율로 바꾼다. 위치를 가장 가까운 보행 칸에 붙이는 규칙(snap)이 측지 거리의 입력을 정한다. 지형은 위치가 놓인 곳을
모형에 알리는 패치 불변 입력이다 [설계서 5.3절]. 격자 칸 번호는 75·iy + ix이고 칸은 200 u이다. 거리 단위는 모두 지도 단위 u이다.

식 목록
-------
    (TER-1)  점유 히스토그램 hist와 센 위치의 분류      `scan_positions` (`_histogram`, `_tally`)
    (TER-2)  보행 가능 칸                              `walkable`
    (TER-3)  보행 그래프의 간선과 가중                  `geodesic_table` (`_graph`, `components`)
    (TER-4)  측지 거리표(반올림과 도달 불가 표시)         `geodesic_table` (`_to_uint16`)
    (TER-5)  위치를 보행 칸에 붙이기                    `Terrain.snap`
    (TER-6)  두 위치의 측지 거리                        `Terrain.geo`
    (TER-7)  구역 규칙                                 `zone_of` (`_zone_thresholds`)
    (TER-8)  국소 벽 비율                              `Terrain.wall_fraction` (`_wall_table`)
    (TER-9)  구역의 짝 ZONE_SWAP                       `zone_of` (`ZONE_SWAP`, `_derive_zone_swap`)

가정
----
- 지형의 입력은 S1 저장소의 FIT 경기뿐이다. `build_terrain`이 역할 배열에서 FIT 행을 골라 그 사실을 만들고, `scan_positions`는 받은 행을 검사하지 않는다. 원시 경기
  파일은 열지 않는다 [S1–S2 판정 S2-13, 설계서 5.3절].
- 지형은 시즌 안에서 변하지 않는다고 가정하고 시즌마다 다시 만든다 [설계서 5.3절]. 이 가정은 코드가 검사하지 않는다.
- 위치 좌표 (pos_x, pos_y)는 지도 단위이며 코드의 x, y이고, 지도 폭 map_width는 코드의 W이다. NaN은 위치 없음이다. 무한대는 `zone_of`, `snap`, `geo`,
  `wall_fraction`이 ValueError로 거부하고 `scan_positions`는 거부하지 않고 census에 센다. 지도 밖이나 벽 안의 위치도 버리지 않고 같은 규칙으로 처리한다
  [S1–S2 판정 S2-9].
- 좁힘은 `casts.narrow`를 거치고(범위 검사, S1–S2 판정 S2-4), 측지 표의 uint16 변환만 자체 범위 검사를 쓴다(영어 본문의 이유).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 5.3 (patch-invariant inputs, terrain: walkable grid from FIT position
#      occupancy, geodesic table as a distance bias, zone tokens, local wall fraction) and section 11 item 12 (75 x 75 grid, 150 hits since
#      the S2 gate S2G-2, no corner cutting, uint16 map units with 65,535 = unreachable, snapping to the nearest walkable cell, 5 x 5 wall
#      fraction, zone rules in normalised coordinates).
# Ref: D:\LOL_Data\runs\20261019_link_rule_fit_1\terrain_occupancy_probe.py (the 150 x 150 occupancy histogram that the probe-equality test
#      reproduces; the 5-hit threshold that the probe draws its picture with was replaced by 150 at the S2 gate S2G-2) and
#      D:\LOL_Data\runs\20261020_terrain_spatial_probe_1\terrain_spatial_probe.py (zone_abs, the thresholds of the zone rules, and the
#      180-degree swap of the zones).
# Ref: E. W. Dijkstra, "A note on two problems in connexion with graphs", Numerische Mathematik 1 (1959) 269-271 (shortest paths; here
#      scipy.sparse.csgraph.dijkstra from every walkable cell).
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from c_ept import config as C
from c_ept import runlog
from c_ept.casts import narrow

# ---- grids ------------------------------------------------------------------------------------------------------------------------
GRID100 = 150                    # cells of the occupancy histogram: 100 u, [0, 15,000]^2
GRID = 75                        # cells of the walkable grid: 200 u, [0, 15,000]^2; cell number = GRID * iy + ix
CELL = 200.0                     # map units per cell of the walkable grid
MAP_LIMIT = 15_000.0             # the histogram covers [0, MAP_LIMIT]^2
MIN_COUNT = 150                  # hits that make a 200 u cell walkable: S2 gate S2G-2, 2026-10-21; was 5; walls sit in the low mode of
#                                  the bimodal occupancy
N_CELLS = GRID * GRID
UNREACHABLE = 65_535             # geodesic table: a wall cell, or a pair without a path
_CHUNK_MATCHES = 1_000           # matches read at a time from the S1 store
_CHUNK_SOURCES = 256             # Dijkstra sources per call (256 x 5625 float64 = 11.5 MB)
_CHUNK_SNAP = 256                # positions per distance matrix in the brute-force part of ``snap``
_COL_X, _COL_Y = C.FRAME_FIELDS.index('x'), C.FRAME_FIELDS.index('y')
_DIAGONAL = CELL * math.sqrt(2.0)
_TERRAIN_DIR, _TERRAIN_TMP = 'terrain', 'terrain.tmp'
_FILES = ('terrain.npz', 'geodesic.npy')

# ---- zones ------------------------------------------------------------------------------------------------------------------------
ZONES = ('base_blue', 'base_red', 'top', 'mid', 'bot', 'river', 'jg_blue_top', 'jg_blue_bot', 'jg_red_top', 'jg_red_bot')
ZONE_RULES = MappingProxyType({
    'norm': C.W,                 # xn = pos_x / norm, yn = pos_y / norm; the team swap maps pos_x to norm - pos_x
    'base_sum_below': 0.36,      # base_blue: xn + yn < 0.36
    'base_sum_above': 1.64,      # base_red: xn + yn > 1.64 (the mirror image of the blue threshold: 2 - 0.36)
    'lane': 0.13,                # top: xn < 0.13 or yn > 1 - 0.13; bot: yn < 0.13 or xn > 1 - 0.13
    'mid_band': 0.07,            # mid: |xn - yn| < 0.07
    'river_band': 0.07,          # river: |xn + yn - 1| < 0.07
    'priority': ('base', 'lane', 'mid', 'river', 'jungle'),
})
_ZONE = {name: i for i, name in enumerate(ZONES)}


def _zone_thresholds() -> tuple[float, float, float, float, float]:
    """(map_width, base, lane, mid, river) in map units: the thresholds of ``ZONE_RULES`` times map_width, written for coordinates from the
    map centre (xc = pos_x - map_width / 2, yc = pos_y - map_width / 2; sc = xc + yc, dc = xc - yc): base_blue is sc < -base, base_red sc >
    base; top is xc < -lane or yc > lane.

    (TER-7)의 임계값 T_base, T_lane, T_mid, T_river를 계산한다.
    """
    r = ZONE_RULES
    if not math.isclose(r['base_sum_above'], 2.0 - r['base_sum_below'], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('ZONE_RULES: base_sum_above must be the mirror image of base_sum_below (2 - base_sum_below), '
                         'or the rules are not symmetric')
    w = float(r['norm'])
    return w, w - r['base_sum_below'] * w, w / 2.0 - r['lane'] * w, r['mid_band'] * w, r['river_band'] * w


_W, _T_BASE, _T_LANE, _T_MID, _T_RIVER = _zone_thresholds()


def _positions(*coords):
    """The coordinate arguments as float64 arrays broadcast to one shape. NaN is a missing position and is kept; an infinite value is
    not a position and raises ValueError."""
    out = np.broadcast_arrays(*(np.asarray(c, dtype=np.float64) for c in coords))
    if any(np.isinf(c).any() for c in out):
        raise ValueError('a position must be a finite number, or NaN for a missing position; an infinite value is neither')
    return out


def zone_of(x, y) -> np.ndarray:
    """Index into ``ZONES`` of the position(s) (x, y), int8 of the broadcast shape; -1 where x or y is NaN (a missing position).

    The rules are in the module text; they hold for every point of the plane, also off the map. Keep -1 out of ``ZONE_SWAP``.

    수식
    ----
    (TER-7)  xc = pos_x − map_width/2,  yc = pos_y − map_width/2,  sc = xc + yc,  dc = xc − yc
             위에서 아래로 처음 맞는 줄이 구역이다 (모든 부등호가 엄격하다):
               sc < −T_base → base_blue;   sc > T_base → base_red;   xc < −T_lane ∨ yc > T_lane → top;   yc < −T_lane ∨ xc > T_lane → bot;
               |dc| < T_mid → mid;   |sc| < T_river → river;
               그 밖: (dc < 0, sc < 0) jg_blue_top;  (dc < 0, sc ≥ 0) jg_red_top;  (dc ≥ 0, sc < 0) jg_blue_bot;  (dc ≥ 0, sc ≥ 0) jg_red_bot
             pos_x 또는 pos_y가 NaN이면 −1.
             임계값: T_base = (1 − base_sum_below)·map_width,  T_lane = map_width/2 − lane·map_width,  T_mid = mid_band·map_width,
                     T_river = river_band·map_width
    (TER-9)  ZONE_SWAP[z] = zone_of(map_width − pos_x, map_width − pos_y)   (zone_of(pos_x, pos_y) = z인 격자점에서)
             격자점: pos_x, pos_y ∈ {0, 125, 250, …, map_width}. 같은 z에서 짝이 달라지거나 열 구역의 순열이 아니면 import에서 ValueError.

      기호                  | 뜻                                | 단위 | 범위                  | 코드 변수
      pos_x, pos_y          | 위치 좌표                         | u    | 실수, 또는 NaN        | x, y
      map_width             | 지도 폭(회전 상수)                | u    | 고정                  | `_W` (= `config.W`)
      xc, yc                | 지도 중심에서 잰 좌표             | u    | 실수                  | a, b
      sc, dc                | 중심 좌표의 합과 차               | u    | 실수                  | s, d
      T_base, T_lane, T_mid, T_river | 구역 임계값              | u    | > 0                   | `_T_BASE`, `_T_LANE`, `_T_MID`, `_T_RIVER`
      zone, z               | 구역 번호(`ZONES`의 위치)         | —    | −1, 0, …, 9           | 반환값(int8); z는 −1이 아닌 구역
      ZONE_SWAP[z]          | 180° 회전 뒤의 짝 구역            | —    | 0, …, 9               | `ZONE_SWAP`

    해석
    ----
    식 (TER-7)은 위치를 열 구역 가운데 하나로 나누는 규칙이며, 기지, 라인, 중앙, 강, 정글 순서로 먼저 맞는 것이 이긴다. 지도 중심에서 잰 좌표와 W를 곱한 임계값으로
    평가하므로 거울상은 xc, yc의 부호만 바뀌고 임계선 위의 점도 항상 짝 구역으로 간다 [S1–S2 판정 R11-a]. 180° 회전은 기지와 라인을 서로 보내고 중앙과 강은 제자리에
    두므로 규칙이 대칭이다. 지도 밖의 점도 같은 규칙으로 구역을 받고 NaN은 −1이다. `ZONE_SWAP`은 손으로 쓰지 않고 이 규칙을 격자점에서 읽어 import 때
    유도하며 순열이 아니면 import가 실패한다 [S1–S2 판정 R11-a].

    근거
    ----
    설계서 5.2절(진영 교환), 5.3절(구역 토큰)과 11절 항목 12(정규화 좌표 규칙), S1–S2 판정 R11-a(스왑 상수로 정규화, 짝은 유도한다).
    """
    # (TER-7)의 xc, yc, sc, dc (코드의 a, b, s, d).
    x, y = _positions(x, y)
    a, b = x - _W / 2.0, y - _W / 2.0                    # from the map centre: the team swap negates both
    s, d = a + b, a - b
    top = (a < -_T_LANE) | (b > _T_LANE)
    bot = (b < -_T_LANE) | (a > _T_LANE)
    # 정글은 마지막 규칙이므로 앞의 구역에 걸리지 않은 점에서만 쓰이고, 아래의 select가 위에서 아래의 순서(기지, 라인, 중앙, 강, 정글)를 정한다.
    jungle = np.where(d < 0, np.where(s < 0, _ZONE['jg_blue_top'], _ZONE['jg_red_top']),
                      np.where(s < 0, _ZONE['jg_blue_bot'], _ZONE['jg_red_bot']))
    z = np.select([s < -_T_BASE, s > _T_BASE, top, bot, np.abs(d) < _T_MID, np.abs(s) < _T_RIVER],
                  [_ZONE['base_blue'], _ZONE['base_red'], _ZONE['top'], _ZONE['bot'], _ZONE['mid'], _ZONE['river']], default=jungle)
    z = np.where(np.isnan(a) | np.isnan(b), -1, z)
    return narrow('zone', z, np.int8)


def _derive_zone_swap() -> np.ndarray:
    """The zone that a zone turns into when the map is rotated by 180 degrees, found by applying ``zone_of`` to a lattice of the map
    (0, 125, ..., 14,875 on both axes) and to the mirror image of every lattice point. ValueError unless every zone always meets the
    same partner and the partners form a permutation of the ten zones.

    (TER-9)의 짝을 `zone_of` 문서의 식대로 유도한다.
    """
    k = np.arange(0.0, _W + 1.0, 125.0)
    gx, gy = np.meshgrid(k, k)
    pairs = np.unique(np.stack([zone_of(gx, gy).ravel(), zone_of(_W - gx, _W - gy).ravel()]), axis=1)
    if pairs.shape[1] != len(ZONES) or not np.array_equal(pairs[0], np.arange(len(ZONES))) or len(set(pairs[1].tolist())) != len(ZONES):
        raise ValueError(f'the zone rules are not symmetric under the team swap: (zone, mirrored zone) pairs {pairs.T.tolist()}')
    swap = narrow('ZONE_SWAP', pairs[1], np.int8)
    swap.setflags(write=False)
    return swap


# (TER-9)의 식과 기호는 `zone_of`의 문서에 있다. 이 상수는 그 식으로 import 때 만든다.
ZONE_SWAP = _derive_zone_swap()          # (10,) int8: ZONE_SWAP[zone_of(pos_x, pos_y)] == zone_of(map_width - pos_x, map_width - pos_y)


# ---- occupancy of the S1 positions ------------------------------------------------------------------------------------------------
def _match_rows(s1, match_rows) -> np.ndarray:
    rows = np.asarray(match_rows)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu':
        raise ValueError('match_rows must be a one-dimensional array of integer match numbers (rows of the matches/ arrays)')
    n = len(s1.array('matches/role'))
    if rows.size and (rows.min() < 0 or rows.max() >= n):
        raise ValueError(f'match_rows must lie in 0 .. {n - 1}')
    if len(np.unique(rows)) != rows.size:
        raise ValueError('match_rows lists a match twice')
    return rows.astype(np.int64)


def _rows_of(offsets, m) -> np.ndarray:
    """The table rows (as one index array) of the matches ``m`` from the (n + 1) offsets of the table."""
    starts, counts = offsets[m], offsets[m + 1] - offsets[m]
    first = np.cumsum(counts) - counts
    return np.repeat(starts - first, counts) + np.arange(int(counts.sum()))


def _new_census() -> dict:
    return dict(n=0, nan=0, zero_zero=0, outside_map=0, not_binned=0)


def _tally(census: dict, p: np.ndarray) -> np.ndarray:
    """Add the (n, 2) positions ``p`` to ``census`` and return the ones the histogram bins.

    ``nan``: pos_x or pos_y is NaN. ``zero_zero``: exactly (0, 0). ``outside_map``: not NaN and a coordinate outside [0, map_width]
    (map_width = 14,875, the map side). ``not_binned``: NaN or outside the histogram range [0, 15,000]^2 (both ends included):
    ``np.histogram2d`` drops these [R11-d].

    (TER-1)의 census를 센다.
    """
    nan = np.isnan(p).any(axis=1)
    binned = ((p >= 0.0) & (p <= MAP_LIMIT)).all(axis=1)                 # False for NaN
    census['n'] += len(p)
    census['nan'] += int(nan.sum())
    census['zero_zero'] += int(((p[:, 0] == 0.0) & (p[:, 1] == 0.0)).sum())
    census['outside_map'] += int((~nan & ((p < 0.0) | (p > _W)).any(axis=1)).sum())
    census['not_binned'] += int((~binned).sum())
    return p[binned]


# (TER-1)의 히스토그램 한 장을 센다.
def _histogram(p: np.ndarray) -> np.ndarray:
    h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=GRID100, range=[[0, MAP_LIMIT], [0, MAP_LIMIT]])
    return h


def scan_positions(s1, match_rows) -> tuple[np.ndarray, np.ndarray, dict]:
    """One pass over the S1 store: ``(frames100, kills100, census)`` for the matches ``match_rows`` (rows of the ``matches/`` arrays).

    ``frames100``: (150, 150) float64 counts [ix, iy] of the positions of all regular frames and of the game-end frame; ``kills100``: the
    same for the kills. ``census``: {'regular_frames' | 'game_end_frame' | 'kills': {n, nan, zero_zero, outside_map, not_binned}}, plain
    ints (see ``_tally``). ``s1`` is an S1 store (``array(name)``); it is only read.

    수식
    ----
    (TER-1)  bin(v) = min(⌊v / 100⌋, 149)            (유한한 v ∈ [0, 15,000]; 두 축에 같은 규칙)
             hist[h, h′] = #{ 위치 (pos_x, pos_y) : pos_x, pos_y ∈ [0, 15,000],  bin(pos_x) = h,  bin(pos_y) = h′ }     (h, h′ = 0, …, 149)
             frames100 = hist(정규 프레임 위치) + hist(경기 끝 프레임 위치),   kills100 = hist(킬 위치)
             census: n, nan = [pos_x 또는 pos_y가 NaN], zero_zero = [(pos_x, pos_y) = (0, 0)],
                     outside_map = [NaN이 아니고 pos_x 또는 pos_y가 [0, map_width] 밖(무한대 포함)],
                     not_binned = [NaN이거나 [0, 15,000]² 밖(무한대 포함)]   (프레임 종류마다 따로 센다; 무한대는 거부하지 않고 센다)

      기호                   | 뜻                                   | 단위 | 범위             | 코드 변수
      hist                   | 100 u 칸의 위치 개수                 | 개   | 정수 ≥ 0         | frames100, kills100 (인덱스 [ix, iy])
      h, h′                  | 히스토그램 칸 번호(0부터)            | —    | 0, …, 149        | 코드의 ix, iy
      v                      | 한 축의 유한한 좌표(`np.histogram2d`의 입력) | u | 유한, [0, 15,000]  | pos_x 또는 pos_y
      bin(v)                 | 좌표 v가 든 히스토그램 칸 번호       | —    | 0, …, 149        | `np.histogram2d`의 칸
      n                      | census가 센 위치의 수                | 개   | 정수 ≥ 0         | `_tally`의 census['n']
      pos_x, pos_y           | 위치 좌표                            | u    | 실수, NaN, ±무한대 | `frames/num`의 x, y, `matches/end_xy`, `kills/x, y`
      map_width              | 지도 폭                              | u    | 고정             | `_W`

    해석
    ----
    식 (TER-1)은 선수 위치와 킬 위치를 100 u 칸에 세는 점유 히스토그램이며, 칸 h는 pos_x가 [100 h, 100 h + 100)인 위치를 받고 15,000은 마지막 칸 149에
    들어간다(`np.histogram2d`). NaN이거나 범위 밖이거나 무한대인 위치는 칸에 넣지 않고 census가 센다(`scan_positions`는 무한대를 거부하지 않는다).
    이 계산은 시제품과 같아 시제품과의 동일성 시험이 고정한다 [S1–S2 판정 R11-d]. 센 위치는 호출하는 쪽이 고른 경기의 행에서 오고, FIT 선택은 `build_terrain`이 한다.

    근거
    ----
    S1–S2 판정 R11-d(의심 위치를 따로 센다, 히스토그램은 시제품과 같다)와 S2-9(버리지 않는다), 설계서 5.3절(FIT 위치 점유).
    """
    rows = _match_rows(s1, match_rows)
    frame_off, kill_off = np.asarray(s1.array('matches/frame_off')), np.asarray(s1.array('matches/kill_off'))
    num, end_xy, kx, ky = s1.array('frames/num'), s1.array('matches/end_xy'), s1.array('kills/x'), s1.array('kills/y')
    frames, kills = np.zeros((GRID100, GRID100)), np.zeros((GRID100, GRID100))
    census = {'regular_frames': _new_census(), 'game_end_frame': _new_census(), 'kills': _new_census()}
    # (TER-1): 경기를 묶음으로 읽어 프레임 위치(정규 프레임과 경기 끝 프레임)와 킬 위치를 hist에 더한다. 위치는 float64로 읽는다.
    for lo in range(0, len(rows), _CHUNK_MATCHES):
        m = rows[lo:lo + _CHUNK_MATCHES]
        regular = np.asarray(num[_rows_of(frame_off, m)][..., [_COL_X, _COL_Y]], dtype=np.float64).reshape(-1, 2)
        last = np.asarray(end_xy[m], dtype=np.float64).reshape(-1, 2)
        k_rows = _rows_of(kill_off, m)
        kill_xy = np.stack([np.asarray(kx[k_rows], dtype=np.float64), np.asarray(ky[k_rows], dtype=np.float64)], axis=1)
        frames += _histogram(_tally(census['regular_frames'], regular)) + _histogram(_tally(census['game_end_frame'], last))
        kills += _histogram(_tally(census['kills'], kill_xy))
    return frames, kills, census


def occupancy_100(s1, match_rows) -> tuple[np.ndarray, np.ndarray]:
    """``(frames100, kills100)`` of ``scan_positions``: (150, 150) counts [ix, iy] of the frame positions (regular frames and the game-end
    frame) and of the kill positions of the matches ``match_rows``. For the FIT matches of the S1 run they equal the prototype's
    ``terrain_occupancy.npz`` (``frames``, ``kills``) exactly.

    (TER-1)의 히스토그램 두 장만 돌려주며 census는 버린다.
    """
    frames, kills, _ = scan_positions(s1, match_rows)
    return frames, kills


def _hist100(name: str, a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (GRID100, GRID100):
        raise ValueError(f'{name}: expected a ({GRID100}, {GRID100}) array of counts, got {a.shape}')
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f'{name}: counts must be finite and not negative')
    return a


def walkable(frames100, kills100) -> np.ndarray:
    """(75, 75) bool grid [iy, ix]: a 200 u cell is walkable when the 2 x 2 histogram cells below it hold at least ``MIN_COUNT`` hits of
    frames and kills together. ``np.histogram2d`` indexes [ix, iy]; the result is transposed.

    수식
    ----
    (TER-2)  count[ix, iy] = Σ_{h ∈ {2ix, 2ix+1}} Σ_{h′ ∈ {2iy, 2iy+1}} (frames100 + kills100)[h, h′],
             walk[iy, ix] = [ count[ix, iy] ≥ MIN_COUNT ]

      기호      | 뜻                                      | 단위 | 범위          | 코드 변수
      count     | 200 u 칸 아래 네 히스토그램 칸의 위치 개수 | 개   | 정수 ≥ 0      | pooled
      ix, iy    | 보행 격자의 칸 번호(0부터)              | —    | 0, …, 74      | 인덱스
      h, h′     | 히스토그램 칸 번호(0부터)               | —    | 0, …, 149     | 인덱스
      MIN_COUNT | 보행 칸의 최소 위치 개수                | 개   | 고정          | `MIN_COUNT`
      walk      | 보행 가능 격자 [iy, ix]                 | —    | 참, 거짓      | 반환값

    해석
    ----
    식 (TER-2)는 프레임 위치와 킬 위치를 합친 개수가 임계값 이상인 칸을 보행 가능으로 본다. 칸마다 위치 개수가 두 봉우리로 갈리고 벽이 낮은 봉우리에 모이므로 임계값이
    그 사이에 놓이며, 처음 값이 낮아 벽 대부분이 보행 칸이 된 것을 S2 관문이 고쳤다 [설계서 11절 항목 12, S2 관문 S2G-2]. 히스토그램은 [ix, iy], 보행 격자는 [iy, ix]이므로 결과를
    뒤집어 준다. 입력의 모양과 유한성, 음수가 아님을 검사한다.

    근거
    ----
    설계서 5.3절(FIT 위치 점유에서 만든 보행 격자)과 11절 항목 12(임계값, S2 관문 S2G-2).
    """
    occupancy = _hist100('frames100', frames100) + _hist100('kills100', kills100)
    pooled = occupancy.reshape(GRID, 2, GRID, 2).sum(axis=(1, 3))        # [ix, iy]
    return np.ascontiguousarray(pooled.T >= MIN_COUNT)


# ---- graph, components, geodesic table --------------------------------------------------------------------------------------------
def _check_walk(walk) -> np.ndarray:
    walk = np.asarray(walk)
    if walk.dtype != bool:
        raise TypeError(f'the walkable grid must be a bool array, got {walk.dtype}')
    if walk.shape != (GRID, GRID):
        raise ValueError(f'the walkable grid must have the shape ({GRID}, {GRID}), got {walk.shape}')
    return walk


def _graph(walk: np.ndarray):
    """The walk graph on the 5625 cell numbers as a sparse matrix (each undirected edge once; walls have no edge): four neighbours at 200,
    a diagonal neighbour at 200 sqrt(2) only inside a 2 x 2 block without a wall, which is exactly "both orthogonal neighbours walkable".

    (TER-3)의 간선을 계산한다.
    """
    idx = np.arange(N_CELLS).reshape(GRID, GRID)
    right, down = walk[:, :-1] & walk[:, 1:], walk[:-1, :] & walk[1:, :]
    block = walk[:-1, :-1] & walk[:-1, 1:] & walk[1:, :-1] & walk[1:, 1:]
    edges = ((idx[:, :-1][right], idx[:, 1:][right], CELL), (idx[:-1, :][down], idx[1:, :][down], CELL),
             (idx[:-1, :-1][block], idx[1:, 1:][block], _DIAGONAL), (idx[:-1, 1:][block], idx[1:, :-1][block], _DIAGONAL))
    src = np.concatenate([e[0] for e in edges])
    dst = np.concatenate([e[1] for e in edges])
    weight = np.concatenate([np.full(len(e[0]), e[2]) for e in edges])
    return coo_matrix((weight, (src, dst)), shape=(N_CELLS, N_CELLS)).tocsr()


class Components(NamedTuple):
    labels: np.ndarray       # (75, 75) int32: component number of a walkable cell (numbered by their lowest cell number), -1 on a wall
    sizes: np.ndarray        # (k,) int64: cells per component
    main: np.ndarray         # (75, 75) bool: the largest component (a tie: the one with the lowest cell number)


def components(walk) -> Components:
    """Connected components of the walkable cells, with the edges of the geodesic graph (a diagonal contact is no connection).

    (TER-3)의 간선으로 섬을 가려 가장 큰 성분(같으면 가장 낮은 칸 번호의 성분)을 `snap`의 대상으로 넘긴다 [S1–S2 판정 R11-b].
    """
    walk = _check_walk(walk)
    cells = np.flatnonzero(walk.ravel())
    labels = np.full(N_CELLS, -1, dtype=np.int32)
    if not cells.size:
        return Components(labels.reshape(GRID, GRID), np.zeros(0, dtype=np.int64), np.zeros((GRID, GRID), dtype=bool))
    _, found = connected_components(_graph(walk), directed=False)
    _, first, inverse = np.unique(found[cells], return_index=True, return_inverse=True)       # first: the lowest cell of each component
    labels[cells] = narrow('component labels', np.argsort(np.argsort(first))[inverse], np.int32)
    sizes = np.bincount(labels[cells]).astype(np.int64)
    labels = labels.reshape(GRID, GRID)
    return Components(labels, sizes, labels == int(np.argmax(sizes)))


def _to_uint16(name: str, a) -> np.ndarray:
    """``a`` (rounded distances) as uint16; ValueError naming ``name`` unless every value is in 0 .. 65,534 (65,535 is ``UNREACHABLE``).

    (TER-4)의 범위 검사와 uint16 변환을 계산한다.
    """
    a = np.asarray(a)
    if a.size:
        lo, hi = a.min(), a.max()
        if not (lo >= 0 and hi < UNREACHABLE):                     # also false for NaN
            raise ValueError(f'{name}: values {lo:g} .. {hi:g} do not fit a uint16 below the unreachable marker {UNREACHABLE:,} [S2-4]')
    return a.astype(np.uint16)


def geodesic_table(walk) -> np.ndarray:
    """(5625, 5625) uint16: the length of the shortest path between every two cells in map units, rounded once per path. A wall cell (also
    against itself) and an unreachable pair are 65,535; a reachable distance of 65,535 or more raises ValueError. Dijkstra runs from the
    walkable cells in chunks of sources, so no float64 table of the whole grid is held.

    수식
    ----
    (TER-3)  보행 칸 c, c′ 사이의 간선: 직교 이웃이면 길이 200,  대각 이웃이면 길이 200·√2 (2 × 2 블록의 네 칸이 모두 보행일 때만)
    (TER-4)  table[c, c′] = rint( path(c, c′) )    (c, c′ 모두 보행 칸이고 경로가 있을 때;  0 ≤ 값 < UNREACHABLE이 아니면 ValueError)
             table[c, c′] = UNREACHABLE            (c 또는 c′가 벽, 또는 경로 없음)

      기호        | 뜻                                | 단위 | 범위                 | 코드 변수
      c, c′       | 칸 번호 75·iy + ix                | —    | 0, …, 5,624          | 인덱스
      path(c, c′) | 간선 위 최단 경로 길이            | u    | ≥ 0, 또는 ∞(경로 없음) | dist (scipy dijkstra)
      rint        | 가장 가까운 정수로 반올림(짝수 쪽)  | —    | —                    | `np.rint`
      table       | 측지 거리표                       | u    | 0, …, 65,534; 65,535 | 반환값(uint16)
      UNREACHABLE | 벽이거나 경로 없음을 나타내는 표시 | —    | 65,535               | `UNREACHABLE`

    해석
    ----
    식 (TER-3)은 보행 칸을 노드로 하고 대각 이동은 두 직교 이웃이 모두 보행일 때만 허용하여 모서리를 가로지르지 않게 한다. 식 (TER-4)는 그 그래프의 최단 경로 길이를 경로마다
    한 번 정수로 반올림해 uint16에 두고, 벽 칸(자기 자신과도)과 경로 없는 쌍에는 거리와 구별되는 표시를 둔다. 도달 가능한 거리가 이 표시 이상이면 표시와 섞이므로 거부한다.
    경로는 격자 위의 길이이므로 두 칸 중심의 유클리드 거리보다 짧지 않다. 변환은 `narrow` 대신 자체 범위 검사를 쓴다(영어 본문의 이유).

    근거
    ----
    설계서 11절 항목 12(측지 거리표의 형태), S1–S2 판정 S2-4(좁힘은 범위 검사를 거친다).
    Ref: E. W. Dijkstra (1959), Numerische Mathematik 1: 269-271; doi:10.1007/BF01386390 (모듈 머리 Ref 줄과 같은 글).
    """
    walk = _check_walk(walk)
    graph = _graph(walk)
    table = np.full((N_CELLS, N_CELLS), UNREACHABLE, dtype=np.uint16)
    # (TER-4): 보행 칸마다 다익스트라를 돌리되 한 번에 _CHUNK_SOURCES개씩 하여 전체 float64 표를 잡지 않는다. 경로 없는 쌍(inf)은 반올림하지 않고 표시값을 둔다.
    sources = np.flatnonzero(walk.ravel())
    for lo in range(0, len(sources), _CHUNK_SOURCES):
        rows = sources[lo:lo + _CHUNK_SOURCES]
        dist = dijkstra(graph, directed=False, indices=rows)          # inf where there is no path
        reachable = np.isfinite(dist)
        part = np.full(dist.shape, UNREACHABLE, dtype=np.uint16)
        part[reachable] = _to_uint16('geodesic', np.rint(dist[reachable]))
        table[rows] = part
    return table


def _check_table(table) -> np.ndarray:
    table = np.asarray(table)
    if table.dtype != np.uint16:
        raise TypeError(f'geodesic: the table must be uint16, got {table.dtype}')
    if table.shape != (N_CELLS, N_CELLS):
        raise ValueError(f'geodesic: the table must have the shape ({N_CELLS}, {N_CELLS}), got {table.shape}')
    return table


# ---- files ------------------------------------------------------------------------------------------------------------------------
def _sha(path) -> str:
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _write_npz(path: Path, arrays: dict) -> None:
    """A stored (uncompressed) .npz that ``np.load`` reads, without time stamps: the same arrays give the same bytes."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            zf.writestr(info, buf.getvalue())


def write_terrain_files(out, walk, frames100, kills100, table) -> dict:
    """Write ``terrain.npz`` (walk (75, 75) bool [iy, ix], frames100 and kills100 (150, 150) float64 [ix, iy]) and ``geodesic.npy`` (the
    uint16 table) into the folder ``out`` (created). Returns {file name: SHA-256}. The arrays are checked before anything is written.

    (TER-1), (TER-2), (TER-4)의 결과를 파일로 옮기는 단계이며, 시간 도장을 넣지 않아 같은 배열이 같은 바이트가 된다.
    """
    walk = _check_walk(walk)
    frames100, kills100 = _hist100('frames100', frames100), _hist100('kills100', kills100)
    table = _check_table(table)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_npz(out / 'terrain.npz', {'walk': walk, 'frames100': frames100, 'kills100': kills100})
    np.save(out / 'geodesic.npy', table)
    return {name: _sha(out / name) for name in _FILES}


_PAD = 3                                  # window centres from -3 to 77 cover every window that touches the grid (a window reaches 2 cells)


def _wall_table(walk: np.ndarray) -> np.ndarray:
    """(81, 81) float32: the non-walkable share of the 5 x 5 window around each cell centre from -3 to 77 on both axes (index = cell + 3);
    cells outside the grid count as walls.

    (TER-8)의 벽 비율 표를 만든다.
    """
    size = GRID + 2 * _PAD
    padded = np.zeros((size + 4, size + 4), dtype=np.int64)
    padded[_PAD + 2:_PAD + 2 + GRID, _PAD + 2:_PAD + 2 + GRID] = walk
    walkable_cells = sum(padded[dy:dy + size, dx:dx + size] for dy in range(5) for dx in range(5))
    return narrow('wall_fraction', (25 - walkable_cells) / 25.0, np.float32)


class Terrain:
    """The terrain of a folder written by ``build_terrain`` (``terrain.npz`` and ``geodesic.npy``; TERRAIN.json when it is there).

    ``path`` the folder; ``walk`` (75, 75) bool [iy, ix]; ``main`` its largest component; ``frames100`` and ``kills100`` the (150, 150)
    counts [ix, iy] that made the grid; ``geodesic`` the (5625, 5625) uint16 table as a read-only memmap (opened, not loaded);
    ``manifest`` the parsed TERRAIN.json or None. Loading checks the shape and dtype of the table and that its diagonal (0 on a walkable
    cell, 65,535 on a wall) fits ``walk``; ValueError otherwise, and when no cell is walkable. A Terrain pickles as its path, so a
    worker pool never copies the table. The memory map stays open for the life of the object (Windows does not delete a mapped file).

    폴더를 열 때 표의 모양, dtype, 대각선이 보행 격자와 맞는지 검사하고, 가장 큰 성분의 칸과 칸 중심을 한 번 계산해 둔다 (TER-5).
    """

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path / 'terrain.npz') as z:
            self.walk = _check_walk(z['walk'])
            self.frames100, self.kills100 = _hist100('frames100', z['frames100']), _hist100('kills100', z['kills100'])
        self.geodesic = np.load(self.path / 'geodesic.npy', mmap_mode='r')
        if self.geodesic.dtype != np.uint16 or self.geodesic.shape != (N_CELLS, N_CELLS):
            raise ValueError(f'geodesic.npy: expected a ({N_CELLS}, {N_CELLS}) uint16 table, '
                             f'got {self.geodesic.shape} {self.geodesic.dtype}')
        if not np.array_equal(np.diagonal(self.geodesic), np.where(self.walk.ravel(), 0, UNREACHABLE)):
            raise ValueError('geodesic.npy does not belong to the walkable grid of terrain.npz: its diagonal differs from the grid')
        comps = components(self.walk)
        if not comps.sizes.size:
            raise ValueError('terrain.npz has no walkable cell')
        self.main = comps.main
        self._main_flat = comps.main.ravel()
        self._cells = np.flatnonzero(self._main_flat)                       # ascending: argmin then takes the smaller cell number
        self._cx, self._cy = CELL * (self._cells % GRID) + CELL / 2, CELL * (self._cells // GRID) + CELL / 2
        self._wall = _wall_table(self.walk)
        manifest = self.path / 'TERRAIN.json'
        self.manifest = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else None

    def __reduce__(self):
        return Terrain, (str(self.path),)

    def snap(self, x, y) -> np.ndarray:
        """Cell number (75 iy + ix, int64) of the walkable cell of the largest component whose centre is nearest to each position; a tie
        goes to the smaller cell number. -1 for a missing position (NaN). A position off the map or in a wall is snapped like any other.

        수식
        ----
        (TER-5)  (ix, iy) = (⌊pos_x/200⌋, ⌊pos_y/200⌋),   c₀ = 75·iy + ix
                 snap(pos_x, pos_y) = c₀                              (0 ≤ ix, iy ≤ 74,  pos_x mod 200 ≠ 0,  pos_y mod 200 ≠ 0,  c₀ ∈ main)
                 snap(pos_x, pos_y) = argmin_{c ∈ main} ( (pos_x − cx_c)² + (pos_y − cy_c)², c )     (그 밖; 사전식 최소: 거리가 같으면 작은 칸 번호)
                 snap(pos_x, pos_y) = −1                              (pos_x 또는 pos_y가 NaN)
                 cx_c = 200·(c mod 75) + 100,   cy_c = 200·⌊c / 75⌋ + 100

          기호      | 뜻                                          | 단위 | 범위             | 코드 변수
          pos_x, pos_y | 위치 좌표                                | u    | 실수, 또는 NaN   | x, y
          ix, iy    | 위치가 든 칸의 번호(0부터)                  | —    | 정수             | ix, iy
          c, c₀     | 칸 번호 75·iy + ix                          | —    | 0, …, 5,624      | cell, `_cells`
          main      | 가장 큰 보행 성분의 칸 집합                 | —    | 비지 않음        | `_main_flat`, `_cells`
          cx_c, cy_c | 칸 c의 중심                                | u    | 100, 300, …      | `_cx`, `_cy`

        해석
        ----
        식 (TER-5)는 위치를 가장 큰 성분의 칸 가운데 중심이 가장 가까운 칸에 붙이므로 붙인 두 위치 사이의 측지 거리는 늘 유한하다 [S1–S2 판정 R11-b]. 칸의 안쪽에 있고 그 칸이
        가장 큰 성분에 속하면 그 칸의 중심이 가장 가까우므로 거리 계산을 건너뛰고, 칸의 경계 위의 점은 같은 거리의 칸이 둘이나 넷이어서 계산에 넘긴다. 같은 거리이면 작은 칸
        번호를 고른다. 지도 밖이나 벽 안의 위치도 버리지 않고 같은 규칙으로 붙이고 [S1–S2 판정 S2-9], NaN은 −1이다.

        근거
        ----
        설계서 11절 항목 12(붙이기 규칙), S1–S2 판정 R11-b(성분 제한과 동률 규칙)와 S2-9.
        """
        x, y = _positions(x, y)
        shape = x.shape
        # (TER-5): 아래 fast는 칸 안쪽의 가장 큰 성분 칸을 거리 계산 없이 정하고, 나머지(경계, 벽, 섬, 지도 밖)는 slow에서 중심 거리의 최소로 정한다.
        x, y = x.ravel(), y.ravel()
        out = np.full(x.size, -1, dtype=np.int64)
        known = ~(np.isnan(x) | np.isnan(y))
        with np.errstate(invalid='ignore'):
            ix, iy = np.floor_divide(x, CELL), np.floor_divide(y, CELL)
            on_edge = (np.mod(x, CELL) == 0) | (np.mod(y, CELL) == 0)        # on a cell edge two or four cells are equally near
        inside = known & (ix >= 0) & (ix < GRID) & (iy >= 0) & (iy < GRID)
        cell = np.where(inside, GRID * np.where(inside, iy, 0) + np.where(inside, ix, 0), 0).astype(np.int64)
        fast = inside & ~on_edge & self._main_flat[cell]                      # strictly inside a main-map cell: that cell is the nearest
        out[fast] = cell[fast]
        slow = np.flatnonzero(known & ~fast)
        for lo in range(0, slow.size, _CHUNK_SNAP):
            s = slow[lo:lo + _CHUNK_SNAP]
            d2 = (x[s, None] - self._cx) ** 2 + (y[s, None] - self._cy) ** 2
            out[s] = self._cells[np.argmin(d2, axis=1)]                      # the first minimum: the smaller cell number
        return out.reshape(shape)

    def geo(self, xa, ya, xb, yb) -> np.ndarray:
        """Geodesic distance in map units (float32) between the snapped positions of (xa, ya) and (xb, yb); the arguments broadcast. The
        distance is between the centres of the two cells, so two positions in one cell are 0 apart. NaN where a position is missing.
        ValueError if the table holds 65,535 for a pair: snapping uses the largest component only, so that cannot happen with a table that
        belongs to the grid.

        수식
        ----
        (TER-6)  geo = table[snap(pos_x1, pos_y1), snap(pos_x2, pos_y2)]    (두 snap이 −1이 아닐 때, float32),    geo = NaN   (한 위치라도 snap = −1)

          기호        | 뜻                          | 단위 | 범위               | 코드 변수
          pos_x1, pos_y1, pos_x2, pos_y2 | 두 위치의 좌표 | u | 실수, 또는 NaN  | xa, ya, xb, yb
          table       | 측지 거리표 (TER-4)         | u    | 0, …, 65,534       | `self.geodesic`
          geo         | 두 칸 중심 사이의 측지 거리 | u    | ≥ 0, 또는 NaN      | 반환값(float32)

        해석
        ----
        식 (TER-6)은 두 위치를 각각 붙인 칸 중심 사이의 거리표 값이므로 같은 칸의 두 위치는 0이고, 칸 안의 위치 차이는 반영하지 않는다. 위치가 없으면 NaN이다.
        붙인 칸은 모두 가장 큰 성분에 있어 표가 격자에 속하면 도달 불가 표시가 나올 수 없고, 나오면 표와 격자가 맞지 않는 것이므로 거부한다 [S1–S2 판정 R11-b].
        uint16 값은 float32로 손실 없이 옮겨진다.

        근거
        ----
        S1–S2 판정 R11-b(도달 불가 표시는 거리로 돌려주지 않는다), R12-n(거리는 칸 중심 사이, 단위는 지도 단위).
        """
        a, b = np.broadcast_arrays(self.snap(xa, ya), self.snap(xb, yb))   # snap first: (10, 1) with (1, 10) is 20 snaps, not 200
        shape = a.shape
        a, b = a.ravel(), b.ravel()
        d = np.full(a.size, np.nan, dtype=np.float32)
        ok = (a >= 0) & (b >= 0)
        if ok.any():
            found = np.asarray(self.geodesic[a[ok], b[ok]])
            if (found == UNREACHABLE).any():
                raise ValueError('the geodesic table holds 65,535 (unreachable) for a pair of snapped positions: '
                                 'it does not belong to this walkable grid')
            d[ok] = found                                                    # uint16 to float32 loses nothing
        return d.reshape(shape)

    def zone(self, x, y) -> np.ndarray:
        """``zone_of``: index into ``ZONES`` (int8), -1 for a missing position.

        식 (TER-7)을 `zone_of`로 계산한다.
        """
        return zone_of(x, y)

    def wall_fraction(self, x, y) -> np.ndarray:
        """Float32 in [0, 1]: the share of the 5 x 5 cells around the cell that contains each position that is not walkable (cells outside
        the 75 x 75 grid are walls); NaN for a missing position. The position is not snapped.

        수식
        ----
        (TER-8)  wall(pos_x, pos_y) = ( 25 − Σ_{|shift_x| ≤ 2} Σ_{|shift_y| ≤ 2} walk[iy + shift_y, ix + shift_x] ) / 25,
                 (ix, iy) = (⌊pos_x/200⌋, ⌊pos_y/200⌋)를 [−3, 77]로 자른 값
                 walk = 0   (격자 밖의 칸),    wall = NaN   (pos_x 또는 pos_y가 NaN)
                 wall의 정확한 값은 25분의 몇이고, 표와 반환값에는 float32로 반올림해 저장한다.

          기호      | 뜻                                  | 단위 | 범위            | 코드 변수
          pos_x, pos_y | 위치 좌표                        | u    | 실수, 또는 NaN  | x, y
          ix, iy    | 위치가 든 칸의 번호(0부터)          | —    | 정수            | ix, iy
          shift_x, shift_y | 윈도우 안의 칸 이동          | 칸   | −2, …, 2        | `_wall_table`의 dx, dy
          walk      | 보행 가능 격자(격자 밖은 0)         | —    | 0, 1            | `self.walk`
          wall      | 윈도우 25칸 가운데 보행 불가의 몫   | —    | [0, 1], 또는 NaN | 반환값(float32)

        해석
        ----
        식 (TER-8)은 위치가 든 칸을 중심으로 한 5 × 5 윈도우에서 보행할 수 없는 칸의 비율이고, 격자 밖의 칸은 벽으로 센다. 위치를 보행 칸에 붙이지 않고 든 칸을 그대로 쓰며
        [S1–S2 판정 R11-c], 칸 번호를 자르는 것은 그 바깥에서는 윈도우에 격자의 칸이 하나도 없어 값이 같기 때문이다. 값은 25분의 몇이며 float32로 줄이고 NaN은 NaN이다.

        근거
        ----
        S1–S2 판정 R11-c(윈도우 규칙), 설계서 11절 항목 12(벽 비율).
        """
        x, y = _positions(x, y)
        known = ~(np.isnan(x) | np.isnan(y))
        with np.errstate(invalid='ignore'):
            ix = np.clip(np.floor_divide(x, CELL), -_PAD, GRID + _PAD - 1)       # beyond that a window holds no cell of the grid at all
            iy = np.clip(np.floor_divide(y, CELL), -_PAD, GRID + _PAD - 1)
        # (TER-8): 잘라 낸 칸 번호에 _PAD를 더해 표를 읽고, 위치가 없는 곳은 NaN으로 덮는다.
        ix, iy = np.where(known, ix, 0).astype(np.int64), np.where(known, iy, 0).astype(np.int64)
        return np.where(known, self._wall[iy + _PAD, ix + _PAD], np.float32(np.nan))

    def verify(self) -> dict:
        """Recompute the SHA-256 of ``terrain.npz`` and ``geodesic.npy`` and compare them with TERRAIN.json; returns the hashes, ValueError
        naming the files that differ, FileNotFoundError when the folder has no TERRAIN.json.

        폴더의 두 파일이 만들 때 TERRAIN.json에 기록한 해시와 같은지 확인한다.
        """
        if self.manifest is None:
            raise FileNotFoundError(f'{self.path.name} has no TERRAIN.json to verify against')
        now = {name: _sha(self.path / name) for name in self.manifest['sha256']}
        bad = sorted(name for name in now if now[name] != self.manifest['sha256'][name])
        if bad:
            raise ValueError(f'{bad} differ from the SHA-256 in TERRAIN.json')
        return now


# ---- the build --------------------------------------------------------------------------------------------------------------------
def _lies_inside(path: Path, folder: Path) -> bool:
    r"""Whether ``path`` is the folder ``folder`` (already resolved) or lies inside it, however either is spelled. Two tests, because a
    text comparison of resolved paths misses spellings on Windows:

    1. the resolved ``path`` is ``folder`` or below it (``is_relative_to``): this follows a relative name, ``..``, a symbolic link and a
       junction (a junction that leads into the S1 run lies inside it, also for a name that does not exist yet);
    2. some existing ancestor of the resolved ``path`` (itself included) is the same folder as ``folder`` by identity,
       ``os.path.samefile`` (volume and file index): ``Path.resolve`` leaves the extended-length prefix ``\\?\`` and the administrative
       share of the local machine (``\\localhost\D$\...``, ``\\127.0.0.1\D$\...``) as they are, so test 1 does not see that they name the
       same folder. An ancestor that does not exist (yet), or cannot be read, is not the S1 run.
    """
    here = Path(path).resolve()
    if here.is_relative_to(folder):
        return True
    for ancestor in (here, *here.parents):
        try:
            if os.path.samefile(ancestor, folder):
                return True
        except OSError:
            continue
    return False


def build_terrain(s1_run, run) -> Path:
    r"""Make the terrain of the FIT matches of an S1 run: ``<run>/terrain/{terrain.npz, geodesic.npy, TERRAIN.json}``; returns that folder.

    ``s1_run``: the S1 run folder (or its ``s1`` store folder), read only. ``run``: the folder of the S2 run. Order: the E6 check (no model
    server runs: ``StrataRunning`` otherwise, nothing read or written [S2-1]); refuse a ``run`` that is the S1 run folder or lies inside it
    (ValueError), so that nothing is ever written into an S1 run: the resolved paths are compared, and every existing ancestor of the
    resolved ``run`` is compared with the S1 run folder by identity, which also sees through the ``\\?\`` prefix and an administrative share
    of this machine (``_lies_inside``); refuse an existing ``terrain`` folder (and a ``terrain.tmp`` left by an interrupted build) with
    FileExistsError; open the S1 store (complete, SHA-256 of every file checked) and take the FIT rows (ValueError if there are none);
    compute; write into ``terrain.tmp``, fsync, write TERRAIN.json last and rename the folder, so a ``terrain`` folder is always complete.
    Nothing is printed. TERRAIN.json holds what the module text lists, and no key.

    FIT 경기의 위치에서 지형 폴더를 만드는 진입점이다. S1 실행은 읽기만 하고, 모델 서버가 돌 때와 `terrain` 폴더가 이미 있을 때는 시작하지 않는다.
    `terrain.tmp`에 쓰고 fsync하며 TERRAIN.json을 마지막에 쓴 뒤 이름을 바꾸는 순서로 만들고(영어 본문), 그 밖의 보장은 주장하지 않는다. FIT 한정은 이 함수가
    역할 배열에서 FIT 행을 골라 만든다.

    근거: S1–S2 판정 S2-1(서버가 돌면 중단), R11-f(출처 기록, 덮어쓰기 거부), 설계서 5.3절.
    """
    from c_ept.run_s1 import load_s1, require_no_strata     # imported here: the runner chain stays out of the workers that load Terrain
    from c_ept.store_core import fsync_file
    require_no_strata()
    run, root = Path(run), Path(s1_run)
    s1_folder = (root.parent if (root / 'STORE.json').is_file() else root).resolve()      # the S1 run folder, also when its store is given
    if _lies_inside(run, s1_folder):
        raise ValueError(f'{run} is the S1 run folder or lies inside it: an S1 run is read only, give the folder of the S2 run')
    out, tmp = run / _TERRAIN_DIR, run / _TERRAIN_TMP
    for folder in (out, tmp):
        if folder.exists():
            raise FileExistsError(f'{folder} exists: a terrain is never overwritten (an interrupted build leaves terrain.tmp: remove it)')
    s1 = load_s1(s1_run)
    fit = np.flatnonzero(np.asarray(s1.array('matches/role')) == C.ROLES.index('FIT'))
    if not fit.size:
        raise ValueError('the S1 run has no FIT match: there is nothing to make a terrain from')
    frames, kills, census = scan_positions(s1, fit)
    walk = walkable(frames, kills)
    comps = components(walk)
    table = geodesic_table(walk)
    largest = int(comps.sizes.max()) if comps.sizes.size else 0
    info = {
        'version': 1, 's1_run': s1_folder.name, 's1_files_sha256': runlog.sha256_json(s1.manifest['files']), 'fit_matches': int(fit.size),
        'grid': {'cells': GRID, 'cell_u': int(CELL), 'histogram_cells': GRID100, 'histogram_u': int(MAP_LIMIT // GRID100),
                 'min_count': MIN_COUNT},
        'walkable_cells': int(walk.sum()),
        'components': {'count': int(comps.sizes.size), 'largest_cells': largest, 'outside_largest': int(comps.sizes.sum()) - largest},
        'positions': census,
        'zone_rules': {k: (list(v) if isinstance(v, tuple) else v) for k, v in ZONE_RULES.items()},
        **runlog.code_state(),                                                   # code_commit and code_dirty, as RUN.json records them
    }
    tmp.mkdir(parents=True)
    try:
        info['sha256'] = write_terrain_files(tmp, walk, frames, kills, table)
        text = json.dumps(info, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
        (tmp / 'TERRAIN.json').write_text(text, encoding='utf-8', newline='\n')
        for name in (*_FILES, 'TERRAIN.json'):
            fsync_file(tmp / name)
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out
