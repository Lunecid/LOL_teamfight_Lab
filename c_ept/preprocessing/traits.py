"""
C-EPT 챔피언 특성 및 레벨/아이템 모듈 (c_ept.traits):
선수 10명의 챔피언 고유 특성(19차원 벡터), 레벨별 성장 스탯, 실시간 레벨 시계,
그리고 교전 시점까지의 보유 아이템 상태(22차원 벡터)를 생성합니다.

- stat_at_level, attack_speed_at_level: LoL 공식 성장 계수 factor(n)을 적용한 레벨 n 스탯 계산
- champion_traits: Data Dragon 기반 챔피언 역할군/사거리/자원 및 레벨 반영 19차원 특성 벡터
- LevelIndex, level_at: LEVEL_UP 이벤트를 누적하여 특정 시점(ref_ms)의 선수별 레벨 조회
- ItemIndex: 교전 시점까지 구매한 아이템의 종합 스탯(22개 피처) 재생
"""
from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache

import numpy as np

from c_ept import legacy
from c_ept.data.casts import narrow
from c_ept.data.reader import MatchFormatError, MatchRecord
from c_ept.domain.units import _event_id, _is_int

# -----------------------------------------------------------------------------
# 기본 상수 및 특성 열(Column) 이름 정의
# -----------------------------------------------------------------------------
N_PLAYERS = 10
MAX_LEVEL = 18                  # 소환사의 협곡 레벨 상한 (1~18)

# 19개 챔피언 특성 열: 역할군 태그 6개, 사거리 2개, 자원 타입 4개, 레벨 성장 스탯 6개, 알 수 없는 챔피언 1개
TRAIT_NAMES = (
    'tag_Assassin', 'tag_Fighter', 'tag_Mage', 'tag_Marksman', 'tag_Support', 'tag_Tank',
    'is_ranged', 'attackrange',
    'partype_Mana', 'partype_Energy', 'partype_none', 'partype_other',
    'lvl_healthMax', 'lvl_armor', 'lvl_magicResist', 'lvl_attackDamage', 'lvl_attackSpeed', 'lvl_movementSpeed',
    'champ_unknown'
)
_N_STATIC = TRAIT_NAMES.index('lvl_healthMax')       # 레벨과 무관한 앞 12개 정적 열
_UNKNOWN = TRAIT_NAMES.index('champ_unknown')

_BASE_FIELDS = ('healthMax', 'armor', 'magicResist', 'attackDamage', 'attackSpeed', 'movementSpeed')
_GROWTH_FIELDS = ('healthMax', 'armor', 'magicResist', 'attackDamage', 'attackSpeed_pct')

_LEGACY = None


def _leg():
    """
    해시 고정된 레거시(LEGACY) 아이템/챔피언 모듈을 로드하고 프로세스 수명 동안 캐시합니다.
    """
    global _LEGACY
    if _LEGACY is None:
        _LEGACY = legacy.load()
    return _LEGACY


def _ms(ref_ms, name: str = 'ref_ms') -> int:
    """
    입력 시각이 엄격한 정수 밀리초(ms)인지 검증합니다.
    """
    if isinstance(ref_ms, (bool, np.bool_)) or not isinstance(ref_ms, (int, np.integer)):
        raise TypeError(f'{name} must be a whole number of ms (int), not {type(ref_ms).__name__}')
    return int(ref_ms)


# -----------------------------------------------------------------------------
# 1. 레벨별 챔피언 스탯 성장 공식 (LoL 공식 위키 기반)
# -----------------------------------------------------------------------------
def _growth_factor(level):
    """
    LoL 공식 스탯 성장 계수를 계산합니다: factor(n) = (n - 1) * (0.7025 + 0.0175 * (n - 1))
    - 레벨 1일 때 0.0, 레벨 18일 때 17.0
    - 레벨이 오를수록 레벨당 증가분이 점진적으로 커지는 2차 곡선 성장 구조
    """
    if isinstance(level, (bool, str, bytes)):
        raise TypeError(f'level must be a number, not {type(level).__name__}')
    n = np.asarray(level, dtype=np.float64)
    if not (np.isfinite(n).all() and (n == np.floor(n)).all() and ((n >= 1) & (n <= MAX_LEVEL)).all()):
        raise ValueError(f'level must be a whole number in 1..{MAX_LEVEL}, got {level!r}')
    d = n - 1.0
    factor = d * (0.7025 + 0.0175 * d)
    return float(factor) if factor.ndim == 0 else factor


def stat_at_level(base, growth, level):
    """
    레벨 n에서의 기본 스탯(체력, 방어력, 마법 저항력, 공격력)을 계산합니다:
    stat(n) = base + growth * factor(n)
    """
    return base + growth * _growth_factor(level)


def attack_speed_at_level(base, growth_pct, level):
    """
    레벨 n에서의 공격 속도를 계산합니다:
    AS(n) = base * (1 + growth_pct * factor(n) / 100)
    """
    return base * (1.0 + growth_pct * _growth_factor(level) / 100.0)


# -----------------------------------------------------------------------------
# 2. 실시간 레벨 시계 (LevelIndex & level_at)
# -----------------------------------------------------------------------------
class LevelIndex:
    """
    매치 내의 모든 LEVEL_UP 이벤트를 시간순으로 인덱싱하여,
    임의의 기준 시각 ref_ms에서의 10명 플레이어 레벨을 이진 탐색(binary search)으로 고속 조회합니다.
    """

    def __init__(self, rec: MatchRecord):
        per_player: list[list[tuple[int, int]]] = [[] for _ in range(N_PLAYERS)]
        for e in rec.events:
            if e.get('type') != 'LEVEL_UP':
                continue
            pid = _event_id(rec, e, 'participantId', 1, None, 'a LEVEL_UP')
            per_player[pid - 1].append((e['timestamp'], self._level(rec, e)))

        # 선수별 레벨업 시각과 해당 시점까지의 누적 최고 레벨을 기록
        self._times: list[list[int]] = []
        self._best: list[list[int]] = []
        for events in per_player:
            events.sort(key=lambda te: te[0])
            self._times.append([t for t, _ in events])
            best, top = [], 0
            for _, level in events:
                top = max(top, level)
                best.append(top)
            self._best.append(best)

    @staticmethod
    def _level(rec: MatchRecord, e: dict) -> int:
        """
        LEVEL_UP 이벤트의 level 필드를 엄격하게 검증합니다.
        """
        if 'level' not in e:
            raise MatchFormatError(rec.key, rec.patch, 'a LEVEL_UP has no level')
        v = e['level']
        if not _is_int(v):
            raise MatchFormatError(rec.key, rec.patch, f'a LEVEL_UP has a level of type {type(v).__name__}, not an integer')
        if not 1 <= v <= MAX_LEVEL:
            raise MatchFormatError(rec.key, rec.patch, f'a LEVEL_UP has level {v}, not in 1-{MAX_LEVEL}')
        return v

    def at(self, ref_ms) -> np.ndarray:
        """
        ref_ms 시점 이하에서 일어난 LEVEL_UP 중 가장 높은 레벨을 10명 선수별로 반환합니다 (기본값 1레벨).
        """
        ref = _ms(ref_ms)
        levels = []
        for times, best in zip(self._times, self._best):
            j = bisect_right(times, ref)
            levels.append(best[j - 1] if j else 1)
        return narrow('level_at', np.array(levels, dtype=np.int64), np.int16)


def level_at(rec: MatchRecord, ref_ms) -> np.ndarray:
    """
    rec 매치의 ref_ms 시점 기준 10명 플레이어의 레벨을 (10,) int16 배열로 반환합니다.
    """
    return LevelIndex(rec).at(ref_ms)


# -----------------------------------------------------------------------------
# 3. 챔피언 특성 벡터 생성 (Data Dragon 기반)
# -----------------------------------------------------------------------------
def _as_levels(levels) -> np.ndarray:
    """
    레벨 배열이 (10,) 형상의 1~18 범위 정수인지 검증합니다.
    """
    a = np.asarray(levels)
    if a.shape != (N_PLAYERS,) or a.dtype.kind not in 'iuf':
        raise ValueError(f'levels must be a numeric array of shape ({N_PLAYERS},), got shape {a.shape} and dtype {a.dtype}')
    a = a.astype(np.float64)
    if not (np.isfinite(a).all() and (a == np.floor(a)).all() and ((a >= 1) & (a <= MAX_LEVEL)).all()):
        raise ValueError(f'levels must be whole levels in 1..{MAX_LEVEL} (no NaN), got {a.tolist()}')
    return a


@lru_cache(maxsize=None)
def _table_row(patch: str, champ: int):
    """
    패치 버전과 챔피언 ID에 해당하는 Data Dragon 특성 데이터(정적 속성 12개, 기본 스탯 6개, 성장 스탯 5개)를 캐시 조회합니다.
    """
    ca = _leg().champion_attributes
    v = ca.champion_vector(champ, ca.load_champion_table_v2(patch))
    if v['champ_unknown']:
        return None
    static = tuple(v[n] for n in TRAIT_NAMES[:_N_STATIC])
    base, growth = tuple(v[f'base_{s}'] for s in _BASE_FIELDS), tuple(v[f'growth_{s}'] for s in _GROWTH_FIELDS)
    if not np.isfinite(np.array(static + base + growth)).all():
        raise ValueError(f'the champion table of patch {patch} lacks a value for champion {champ}')
    return static, base, growth


def champion_traits(rec: MatchRecord, levels) -> np.ndarray:
    """
    주어진 레벨(levels)에서의 10명 플레이어 챔피언 특성을 (10, 19) float32 배열로 계산합니다.

    - 1~12번 열: 정적 속성 (역할군 태그, 원거리 여부, 사거리, 자원 타입)
    - 13~18번 열: 레벨 n 반영 스탯 (체력, 방어력, 마저, 공격력, 공속, 이속)
    - 19번 열: champ_unknown 플래그 (정상 챔피언 0.0, 미확인 챔피언 1.0 및 나머지 18개 열 NaN)
    """
    lv = _as_levels(levels)
    rows = [_table_row(rec.patch, int(c)) for c in rec.champ]

    out = np.full((N_PLAYERS, len(TRAIT_NAMES)), np.nan)
    known = np.array([r is not None for r in rows])
    out[~known, _UNKNOWN] = 1.0

    if known.any():
        idx = np.flatnonzero(known)
        static = np.array([rows[i][0] for i in idx])
        base, growth, n = np.array([rows[i][1] for i in idx]), np.array([rows[i][2] for i in idx]), lv[idx]

        # 정적 속성 및 레벨별 스탯 주입
        out[idx, :_N_STATIC] = static
        out[idx, _N_STATIC:_N_STATIC + 4] = stat_at_level(base[:, :4], growth[:, :4], n[:, None])
        out[idx, _N_STATIC + 4] = attack_speed_at_level(base[:, 4], growth[:, 4], n)
        out[idx, _N_STATIC + 5] = base[:, 5]       # 이동 속도는 레벨 성장이 없음
        out[idx, _UNKNOWN] = 0.0

    return narrow('champion_traits', out, np.float32)


# -----------------------------------------------------------------------------
# 4. 아이템 벡터 재생 (Item Replay)
# -----------------------------------------------------------------------------
def item_vector_names() -> tuple:
    """
    아이템 벡터의 22개 속성 열 이름을 반환합니다.
    """
    return _leg().item_state.ITEM_VECTOR_NAMES


@lru_cache(maxsize=None)
def _effect_flags(patch: str):
    """
    해당 패치의 아이템 고유 효과 플래그 셋을 로드합니다.
    """
    item_state = _leg().item_state
    return item_state.load_effect_flags(item_state.load_item_table_v2(patch), patch=patch)


class ItemIndex:
    """
    매치 내 아이템 구매/판매 이벤트를 완벽하게 재생하여,
    ref_ms 시점에 10명 플레이어가 보유한 총 아이템 스탯을 (10, 22) float32 벡터로 산출합니다.
    """

    def __init__(self, rec: MatchRecord):
        item_state = _leg().item_state
        champion_by_pid = {str(i + 1): int(c) for i, c in enumerate(rec.champ)}
        self._names = item_state.ITEM_VECTOR_NAMES
        self._index = item_state.ItemStateIndex(
            rec.events, item_state.load_item_table_v2(rec.patch), exact=True, effect_flags=_effect_flags(rec.patch),
            champion_by_pid=champion_by_pid, runes_by_pid=rec.runes_by_pid, patch=rec.patch
        )

    def vectors(self, ref_ms) -> np.ndarray:
        """
        ref_ms 시점 기준 10명 플레이어의 22개 아이템 스탯 벡터를 (10, 22) float32 배열로 반환합니다.
        """
        per_player = self._index.vectors_at(_ms(ref_ms))
        out = np.array([[per_player[pid][name] for name in self._names] for pid in range(1, N_PLAYERS + 1)], dtype=np.float64)
        if not np.isfinite(out).all():
            raise ValueError('item_vectors: the legacy replay gave a value that is not finite')
        return narrow('item_vectors', out, np.float32)
