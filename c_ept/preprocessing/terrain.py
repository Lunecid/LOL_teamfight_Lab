"""
C-EPT 소환사의 협곡 지형(Terrain) 모듈 (c_ept.terrain):
실제 플레이어 이동 궤적 데이터를 기반으로 보행 가능 격자(Walkable Grid),
맵 전체의 최단 경로 측지선 거리표(Geodesic Distance Table), 구역(Zone) 분류,
그리고 국소 벽 비율(Wall Fraction)을 구축하고 제공합니다.

- scan_positions, walkable: 15,000x15,000 맵을 75x75(200유닛) 격자로 나누고 방문 150회 이상 칸을 보행 가능 노드로 판정
- geodesic_table: 5,625개 노드 간 다익스트라(Dijkstra) 8방향 최단 보행 거리를 사전 계산한 uint16 거리표
- Terrain.snap: 임의의 위치를 가장 가까운 메인 보행 가능 셀로 스냅 (거리 계산 시 유한성 보장)
- Terrain.geo: 두 좌표 사이의 실제 보행 우회 측지선 거리(map units) 반환
- zone_of: 10개 지형 구역(기지, 3개 라인, 강, 4개 정글 사분면) 판별
- Terrain.wall_fraction: 특정 위치 주변 5x5 윈도우 내 벽 비율(0.0~1.0) 계산
"""
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
from c_ept.data.casts import narrow

# -----------------------------------------------------------------------------
# 1. 격자 규격 및 기본 상수
# -----------------------------------------------------------------------------
# 원시 위치 집계용 100 유닛 격자 크기 (150x150 셀)
GRID100 = 150

# 모델 입력용 200 유닛 보행 격자 크기 (75x75 셀, 총 5,625개 셀, cell_idx = 75 * iy + ix)
GRID = 75
CELL = 200.0                     # 한 셀의 가로/세로 길이 (200 맵 유닛)
MAP_LIMIT = 15_000.0             # 격자 좌표 상한 [0, 15,000]^2

# 200 유닛 셀이 보행 가능(Walkable) 노드로 인정받기 위한 최소 통과 횟수 (150회 이상)
MIN_COUNT = 150
N_CELLS = GRID * GRID

# 측지선 거리표에서 도달 불가(벽 또는 경로 없음)를 나타내는 특수 센티넬 값 (65,535)
UNREACHABLE = 65_535

_CHUNK_MATCHES = 1_000           # 저장소 일괄 읽기 매치 단위
_CHUNK_SOURCES = 256             # 다익스트라 일괄 출발지 노드 수
_CHUNK_SNAP = 256                # 스냅 계산 일괄 좌표 수
_COL_X, _COL_Y = C.FRAME_FIELDS.index('x'), C.FRAME_FIELDS.index('y')
_DIAGONAL = CELL * math.sqrt(2.0)
_TERRAIN_DIR, _TERRAIN_TMP = 'terrain', 'terrain.tmp'
_FILES = ('terrain.npz', 'geodesic.npy')

# -----------------------------------------------------------------------------
# 2. 맵 전술 구역(Zone) 정의 및 대칭 규칙
# -----------------------------------------------------------------------------
# 10개 전술 구역 이름 (블루기지, 레드기지, 탑, 미드, 봇, 강가, 4개 정글 사분면)
ZONES = ('base_blue', 'base_red', 'top', 'mid', 'bot', 'river', 'jg_blue_top', 'jg_blue_bot', 'jg_red_top', 'jg_red_bot')

ZONE_RULES = MappingProxyType({
    'norm': C.W,                 # 좌표 정규화 상수 (W = 14,875)
    'base_sum_below': 0.36,      # 블루팀 본진 판정 임계치 (xn + yn < 0.36)
    'base_sum_above': 1.64,      # 레드팀 본진 판정 임계치 (2 - 0.36 점대칭: xn + yn > 1.64)
    'lane': 0.13,                # 탑/봇 외곽 라인 경계 두께
    'mid_band': 0.07,            # 미드 라인 대각선 밴드 폭 (|xn - yn| < 0.07)
    'river_band': 0.07,          # 강가 대각선 밴드 폭 (|xn + yn - 1| < 0.07)
    'priority': ('base', 'lane', 'mid', 'river', 'jungle'),
})
_ZONE = {name: i for i, name in enumerate(ZONES)}


def _zone_thresholds() -> tuple[float, float, float, float, float]:
    """
    맵 중심(W/2, W/2) 기준으로 변환된 구역 판별 임계값들을 맵 유닛 단위로 계산합니다.
    """
    r = ZONE_RULES
    if not math.isclose(r['base_sum_above'], 2.0 - r['base_sum_below'], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('ZONE_RULES: base_sum_above must be the mirror image of base_sum_below (2 - base_sum_below), '
                         'or the rules are not symmetric')
    w = float(r['norm'])
    return w, w - r['base_sum_below'] * w, w / 2.0 - r['lane'] * w, r['mid_band'] * w, r['river_band'] * w


_W, _T_BASE, _T_LANE, _T_MID, _T_RIVER = _zone_thresholds()


def _positions(*coords):
    """
    입력 좌표들을 float64 배열로 브로드캐스팅하고 유효성을 검사합니다.
    결측 위치(NaN)는 허용하되, 무한대(inf)는 에러로 거부합니다.
    """
    out = np.broadcast_arrays(*(np.asarray(c, dtype=np.float64) for c in coords))
    if any(np.isinf(c).any() for c in out):
        raise ValueError('a position must be a finite number, or NaN for a missing position; an infinite value is neither')
    return out


def zone_of(x, y) -> np.ndarray:
    """
    주어진 좌표 (x, y)가 속한 전술 구역(0~9: ZONES)을 판별하여 int8 배열로 반환합니다 (-1: 위치 결측/NaN).

    - 우선순위 평가: 기지(Base) -> 외곽 라인(Top/Bot) -> 미드(Mid) -> 강가(River) -> 정글 사분면(Jungle)
    - 180도 점대칭 회전 시 완벽한 대칭성을 만족하도록 설계되었습니다.
    """
    x, y = _positions(x, y)
    a, b = x - _W / 2.0, y - _W / 2.0                    # 맵 중심 기준 상대 좌표
    s, d = a + b, a - b
    top = (a < -_T_LANE) | (b > _T_LANE)
    bot = (b < -_T_LANE) | (a > _T_LANE)

    # 라인이나 강가에 걸리지 않는 나머지 영역을 4개 정글 사분면으로 분할
    jungle = np.where(d < 0, np.where(s < 0, _ZONE['jg_blue_top'], _ZONE['jg_red_top']),
                      np.where(s < 0, _ZONE['jg_blue_bot'], _ZONE['jg_red_bot']))

    # 우선순위에 따라 조건문 평가
    z = np.select(
        [s < -_T_BASE, s > _T_BASE, top, bot, np.abs(d) < _T_MID, np.abs(s) < _T_RIVER],
        [_ZONE['base_blue'], _ZONE['base_red'], _ZONE['top'], _ZONE['bot'], _ZONE['mid'], _ZONE['river']],
        default=jungle
    )
    z = np.where(np.isnan(a) | np.isnan(b), -1, z)
    return narrow('zone', z, np.int8)


def _derive_zone_swap() -> np.ndarray:
    """
    맵을 180도 회전시켰을 때 10개 구역이 각각 어떤 구역으로 매핑되는지 격자 전수 조사를 통해 자동 도출합니다.
    """
    k = np.arange(0.0, _W + 1.0, 125.0)
    gx, gy = np.meshgrid(k, k)
    pairs = np.unique(np.stack([zone_of(gx, gy).ravel(), zone_of(_W - gx, _W - gy).ravel()]), axis=1)
    if pairs.shape[1] != len(ZONES) or not np.array_equal(pairs[0], np.arange(len(ZONES))) or len(set(pairs[1].tolist())) != len(ZONES):
        raise ValueError(f'the zone rules are not symmetric under the team swap: (zone, mirrored zone) pairs {pairs.T.tolist()}')
    swap = narrow('ZONE_SWAP', pairs[1], np.int8)
    swap.setflags(write=False)
    return swap


ZONE_SWAP = _derive_zone_swap()


# -----------------------------------------------------------------------------
# 3. S1 매치 위치 스캔 및 보행 가능 격자 도출
# -----------------------------------------------------------------------------
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
    starts, counts = offsets[m], offsets[m + 1] - offsets[m]
    first = np.cumsum(counts) - counts
    return np.repeat(starts - first, counts) + np.arange(int(counts.sum()))


def _new_census() -> dict:
    return dict(n=0, nan=0, zero_zero=0, outside_map=0, not_binned=0)


def _tally(census: dict, p: np.ndarray) -> np.ndarray:
    """
    좌표 데이터를 검사하여 정상 좌표, 결측(NaN), 원점(0,0), 맵 외곽 위치 수를 집계하고 히스토그램 범위 내 좌표를 반환합니다.
    """
    nan = np.isnan(p).any(axis=1)
    binned = ((p >= 0.0) & (p <= MAP_LIMIT)).all(axis=1)
    census['n'] += len(p)
    census['nan'] += int(nan.sum())
    census['zero_zero'] += int(((p[:, 0] == 0.0) & (p[:, 1] == 0.0)).sum())
    census['outside_map'] += int((~nan & ((p < 0.0) | (p > _W)).any(axis=1)).sum())
    census['not_binned'] += int((~binned).sum())
    return p[binned]


def _histogram(p: np.ndarray) -> np.ndarray:
    """
    좌표 배열 p에 대해 150x150 2차원 점유 히스토그램을 생성합니다.
    """
    h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=GRID100, range=[[0, MAP_LIMIT], [0, MAP_LIMIT]])
    return h


def scan_positions(s1, match_rows) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    S1 저장소의 대상 매치들을 순회하며 정규 프레임, 종료 프레임, 킬 위치를
    150x150 (100 유닛 단위) 2D 히스토그램으로 집계합니다.
    """
    rows = _match_rows(s1, match_rows)
    frame_off, kill_off = np.asarray(s1.array('matches/frame_off')), np.asarray(s1.array('matches/kill_off'))
    num, end_xy, kx, ky = s1.array('frames/num'), s1.array('matches/end_xy'), s1.array('kills/x'), s1.array('kills/y')
    frames, kills = np.zeros((GRID100, GRID100)), np.zeros((GRID100, GRID100))
    census = {'regular_frames': _new_census(), 'game_end_frame': _new_census(), 'kills': _new_census()}

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
    """
    150x150 히스토그램을 2x2 풀링하여 75x75 격자로 축소하고,
    플레이어 이동 및 킬 방문 횟수가 MIN_COUNT(150회) 이상인 셀을 보행 가능(True) 노드로 판정합니다.
    """
    occupancy = _hist100('frames100', frames100) + _hist100('kills100', kills100)
    pooled = occupancy.reshape(GRID, 2, GRID, 2).sum(axis=(1, 3))        # [ix, iy]
    return np.ascontiguousarray(pooled.T >= MIN_COUNT)


# -----------------------------------------------------------------------------
# 4. 그래프 생성 및 다익스트라 전역 측지 거리표 (Geodesic Distance Table)
# -----------------------------------------------------------------------------
def _check_walk(walk) -> np.ndarray:
    walk = np.asarray(walk)
    if walk.dtype != bool:
        raise TypeError(f'the walkable grid must be a bool array, got {walk.dtype}')
    if walk.shape != (GRID, GRID):
        raise ValueError(f'the walkable grid must have the shape ({GRID}, {GRID}), got {walk.shape}')
    return walk


def _graph(walk: np.ndarray):
    """
    보행 가능 셀들 간의 8방향 희소 그래프를 구축합니다:
    - 직교 인접 셀 간 간선 가중치: 200 유닛
    - 대각선 인접 셀 간 간선 가중치: 200 * sqrt(2) 유닛 (코너 커팅 방지: 2x2 블록 내 네 셀이 모두 보행 가능할 때만 연결)
    """
    idx = np.arange(N_CELLS).reshape(GRID, GRID)
    right, down = walk[:, :-1] & walk[:, 1:], walk[:-1, :] & walk[1:, :]
    block = walk[:-1, :-1] & walk[:-1, 1:] & walk[1:, :-1] & walk[1:, 1:]
    edges = (
        (idx[:, :-1][right], idx[:, 1:][right], CELL),
        (idx[:-1, :][down], idx[1:, :][down], CELL),
        (idx[:-1, :-1][block], idx[1:, 1:][block], _DIAGONAL),
        (idx[:-1, 1:][block], idx[1:, :-1][block], _DIAGONAL)
    )
    src = np.concatenate([e[0] for e in edges])
    dst = np.concatenate([e[1] for e in edges])
    weight = np.concatenate([np.full(len(e[0]), e[2]) for e in edges])
    return coo_matrix((weight, (src, dst)), shape=(N_CELLS, N_CELLS)).tocsr()


class Components(NamedTuple):
    labels: np.ndarray       # (75, 75) int32 컴포넌트 레이블 (-1: 벽/보행 불가)
    sizes: np.ndarray        # 컴포넌트별 셀 개수
    main: np.ndarray         # (75, 75) bool 메인 맵 컴포넌트 마스크


def components(walk) -> Components:
    """
    보행 가능 격자의 연결 성분들을 라벨링하고, 고립된 섬들을 제외한 메인 맵 컴포넌트를 분리합니다.
    """
    walk = _check_walk(walk)
    cells = np.flatnonzero(walk.ravel())
    labels = np.full(N_CELLS, -1, dtype=np.int32)
    if not cells.size:
        return Components(labels.reshape(GRID, GRID), np.zeros(0, dtype=np.int64), np.zeros((GRID, GRID), dtype=bool))
    _, found = connected_components(_graph(walk), directed=False)
    _, first, inverse = np.unique(found[cells], return_index=True, return_inverse=True)
    labels[cells] = narrow('component labels', np.argsort(np.argsort(first))[inverse], np.int32)
    sizes = np.bincount(labels[cells]).astype(np.int64)
    labels = labels.reshape(GRID, GRID)
    return Components(labels, sizes, labels == int(np.argmax(sizes)))


def _to_uint16(name: str, a) -> np.ndarray:
    a = np.asarray(a)
    if a.size:
        lo, hi = a.min(), a.max()
        if not (lo >= 0 and hi < UNREACHABLE):
            raise ValueError(f'{name}: values {lo:g} .. {hi:g} do not fit a uint16 below the unreachable marker {UNREACHABLE:,} [S2-4]')
    return a.astype(np.uint16)


def geodesic_table(walk) -> np.ndarray:
    """
    보행 격자 상의 모든 셀 쌍(5625 x 5625) 간 최단 이동 거리를 다익스트라 알고리즘으로 사전 연산하여
    uint16 거리표로 반환합니다 (도달 불가 및 벽은 65,535 센티넬 값).
    """
    walk = _check_walk(walk)
    graph = _graph(walk)
    table = np.full((N_CELLS, N_CELLS), UNREACHABLE, dtype=np.uint16)

    sources = np.flatnonzero(walk.ravel())
    for lo in range(0, len(sources), _CHUNK_SOURCES):
        rows = sources[lo:lo + _CHUNK_SOURCES]
        dist = dijkstra(graph, directed=False, indices=rows)
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


# -----------------------------------------------------------------------------
# 5. 파일 저장 및 국소 벽 비율(Wall Fraction) 사전 연산
# -----------------------------------------------------------------------------
def _sha(path) -> str:
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _write_npz(path: Path, arrays: dict) -> None:
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            zf.writestr(info, buf.getvalue())


def write_terrain_files(out, walk, frames100, kills100, table) -> dict:
    walk = _check_walk(walk)
    frames100, kills100 = _hist100('frames100', frames100), _hist100('kills100', kills100)
    table = _check_table(table)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_npz(out / 'terrain.npz', {'walk': walk, 'frames100': frames100, 'kills100': kills100})
    np.save(out / 'geodesic.npy', table)
    return {name: _sha(out / name) for name in _FILES}


_PAD = 3


def _wall_table(walk: np.ndarray) -> np.ndarray:
    """
    각 셀 중심 주변 5x5 윈도우(1000x1000 유닛) 내에서 벽(보행 불가)이 차지하는 비율(0.0~1.0)을 사전 계산합니다.
    """
    size = GRID + 2 * _PAD
    padded = np.zeros((size + 4, size + 4), dtype=np.int64)
    padded[_PAD + 2:_PAD + 2 + GRID, _PAD + 2:_PAD + 2 + GRID] = walk
    walkable_cells = sum(padded[dy:dy + size, dx:dx + size] for dy in range(5) for dx in range(5))
    return narrow('wall_fraction', (25 - walkable_cells) / 25.0, np.float32)


# -----------------------------------------------------------------------------
# 6. 지형 쿼리 인터페이스 클래스 (Terrain)
# -----------------------------------------------------------------------------
class Terrain:
    """
    빌드된 지형 데이터(terrain.npz, geodesic.npy)를 메모리 맵(mmap)으로 로드하여 고속 쿼리를 제공하는 클래스입니다.

    - snap(x, y): 임의 좌표를 메인 보행 가능 컴포넌트의 가장 가까운 셀 인덱스(0~5,624)로 투영
    - geo(xa, ya, xb, yb): 두 좌표 사이의 실제 보행 우회 측지선 거리(float32, 맵 유닛) 조회
    - zone(x, y): 좌표가 속한 10개 전술 구역 인덱스(0~9) 반환
    - wall_fraction(x, y): 해당 좌표 주변 5x5 윈도우의 국소 벽 비율(0.0~1.0, float32) 반환
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
        self._cells = np.flatnonzero(self._main_flat)
        self._cx, self._cy = CELL * (self._cells % GRID) + CELL / 2, CELL * (self._cells // GRID) + CELL / 2
        self._wall = _wall_table(self.walk)
        manifest = self.path / 'TERRAIN.json'
        self.manifest = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else None

    def __reduce__(self):
        return Terrain, (str(self.path),)

    def snap(self, x, y) -> np.ndarray:
        """
        임의의 위치 (x, y)를 메인 맵 컴포넌트의 가장 가까운 보행 가능 셀 인덱스(0~5,624)로 스냅합니다 (-1: 결측 NaN).
        """
        x, y = _positions(x, y)
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        out = np.full(x.size, -1, dtype=np.int64)
        known = ~(np.isnan(x) | np.isnan(y))

        with np.errstate(invalid='ignore'):
            ix, iy = np.floor_divide(x, CELL), np.floor_divide(y, CELL)
            on_edge = (np.mod(x, CELL) == 0) | (np.mod(y, CELL) == 0)

        # 이미 메인 맵 보행 셀 내부에 완전히 위치한 좌표는 O(1) 고속 매핑
        inside = known & (ix >= 0) & (ix < GRID) & (iy >= 0) & (iy < GRID)
        cell = np.where(inside, GRID * np.where(inside, iy, 0) + np.where(inside, ix, 0), 0).astype(np.int64)
        fast = inside & ~on_edge & self._main_flat[cell]
        out[fast] = cell[fast]

        # 셀 경계선, 벽 내부, 또는 맵 외곽 좌표는 유클리드 최근접 탐색으로 스냅
        slow = np.flatnonzero(known & ~fast)
        for lo in range(0, slow.size, _CHUNK_SNAP):
            s = slow[lo:lo + _CHUNK_SNAP]
            d2 = (x[s, None] - self._cx) ** 2 + (y[s, None] - self._cy) ** 2
            out[s] = self._cells[np.argmin(d2, axis=1)]
        return out.reshape(shape)

    def geo(self, xa, ya, xb, yb) -> np.ndarray:
        """
        두 위치 (xa, ya)와 (xb, yb) 사이의 실제 보행 우회 측지선 거리를 float32로 반환합니다.
        - 스냅된 셀 중심 간의 다익스트라 최단 경로 거리
        - 결측 위치가 포함된 경우 NaN 반환
        """
        a, b = np.broadcast_arrays(self.snap(xa, ya), self.snap(xb, yb))
        shape = a.shape
        a, b = a.ravel(), b.ravel()
        d = np.full(a.size, np.nan, dtype=np.float32)
        ok = (a >= 0) & (b >= 0)
        if ok.any():
            found = np.asarray(self.geodesic[a[ok], b[ok]])
            if (found == UNREACHABLE).any():
                raise ValueError('the geodesic table holds 65,535 (unreachable) for a pair of snapped positions: '
                                 'it does not belong to this walkable grid')
            d[ok] = found
        return d.reshape(shape)

    def zone(self, x, y) -> np.ndarray:
        """
        좌표의 전술 구역(0~9) 인덱스 반환
        """
        return zone_of(x, y)

    def wall_fraction(self, x, y) -> np.ndarray:
        """
        해당 좌표 주변 5x5 윈도우 내의 벽(보행 불가) 비율(0.0~1.0, float32) 반환
        """
        x, y = _positions(x, y)
        known = ~(np.isnan(x) | np.isnan(y))
        with np.errstate(invalid='ignore'):
            ix = np.clip(np.floor_divide(x, CELL), -_PAD, GRID + _PAD - 1)
            iy = np.clip(np.floor_divide(y, CELL), -_PAD, GRID + _PAD - 1)
        ix, iy = np.where(known, ix, 0).astype(np.int64), np.where(known, iy, 0).astype(np.int64)
        return np.where(known, self._wall[iy + _PAD, ix + _PAD], np.float32(np.nan))

    def verify(self) -> dict:
        """
        저장된 terrain.npz 및 geodesic.npy의 SHA-256 해시를 TERRAIN.json 메타데이터와 대조 검증합니다.
        """
        if self.manifest is None:
            raise FileNotFoundError(f'{self.path.name} has no TERRAIN.json to verify against')
        now = {name: _sha(self.path / name) for name in self.manifest['sha256']}
        bad = sorted(name for name in now if now[name] != self.manifest['sha256'][name])
        if bad:
            raise ValueError(f'{bad} differ from the SHA-256 in TERRAIN.json')
        return now


# -----------------------------------------------------------------------------
# 7. 지형 빌드 엔트리포인트 (build_terrain)
# -----------------------------------------------------------------------------
def _lies_inside(path: Path, folder: Path) -> bool:
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
    """
    S1 실행의 FIT 매치 데이터를 기반으로 지형 아티팩트(<run>/terrain/)를 원자적으로 생성합니다.

    1. 활성 모델 서버 여부 사전 검사 (require_no_strata)
    2. S1 폴더 덮어쓰기 방지 검증
    3. FIT 매치 위치 스캔 -> 보행 격자 도출 -> 다익스트라 측지 거리표 계산
    4. 임시 폴더(terrain.tmp)에 기록 후 원자적 이름 변경(rename)으로 완성
    """
    from c_ept.workflows.run_s1 import load_s1, require_no_strata
    from c_ept.data.store_core import fsync_file

    require_no_strata()
    run, root = Path(run), Path(s1_run)
    s1_folder = (root.parent if (root / 'STORE.json').is_file() else root).resolve()
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
        'version': 1, 's1_run': s1_folder.name, 's1_files_sha256': runlog.sha256_json(s1.manifest['files']),
        'fit_matches': int(fit.size),
        'grid': {
            'cells': GRID, 'cell_u': int(CELL), 'histogram_cells': GRID100,
            'histogram_u': int(MAP_LIMIT // GRID100), 'min_count': MIN_COUNT
        },
        'walkable_cells': int(walk.sum()),
        'components': {'count': int(comps.sizes.size), 'largest_cells': largest, 'outside_largest': int(comps.sizes.sum()) - largest},
        'positions': census,
        'zone_rules': {k: (list(v) if isinstance(v, tuple) else v) for k, v in ZONE_RULES.items()},
        **runlog.code_state(),
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
