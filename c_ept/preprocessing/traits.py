"""Player columns that are not frame values: champion traits (no ids), the level clock and the item vectors of the ten players.

``champion_traits(rec, levels)`` gives each player a row of ``TRAIT_NAMES`` (19 columns, float32): class tags, range, resource type, the
six level-scaled statistics (health, armor, magic resist, attack damage, attack speed, movement speed at the player's level) and the
flag ``champ_unknown``. The champion key is looked up in the Data Dragon table of the match's patch and never enters a row; a champion
that is not in the table (missing, or a mode key) is ``champ_unknown = 1`` with NaN in every other column (NaN is allowed here and
nowhere else in a trait row). The table values are the LEGACY ``champion_vector``; the level scaling is ``stat_at_level`` and
``attack_speed_at_level`` below, valid for the levels 1-18 of Summoner's Rift (anything else raises). The timeline's championStats are
whole numbers: the frame armor of a player at level 2 or more without armor items is floor(lvl_armor) in 3,436 of 3,893 golden
player-frames (88%; magic resist 3,859 of 4,396), and at level 1 it equals the trait, so compare a trait with a frame statistic after
flooring it (checked 2026-10-20).

``level_at(rec, ref_ms)`` is the level clock of the event-clock views: the highest LEVEL_UP level of each player among the events at or
before ``ref_ms`` (1 if none). Frame-clock rows use the level of their frame (``rec.frame_num[k, :, 2]``). At a regular frame the two
agree except when a LEVEL_UP falls on the frame's own millisecond and the frame snapshot does not show it yet.
Counts of 2026-10-20, run at commit 39195c7: none of the 5,820 player-frames of the 20 golden matches (c_ept/checks/measure_levels.py,
run again from the package with the same result; the golden test pins it) and 150 of the 7,573,150 player-frames of all 28,701 S1
matches (0.002%: 150 player-frame cells, each a lag of one level at a LEVEL_UP on the frame's ms; no other kind of difference). The
second count comes from the scratch predecessor of c_ept/checks/sweep_s1.py, which was moved there and not run again from the package.
It counts player-frame cells; the 1,888 LEVEL_UPs that fall on a frame's ms are events, and no rate of cells per LEVEL_UP was measured.
If a pinned count changes: measure again, list the differing cells and their cause (c_ept/checks/explain_mismatch.py); a frame that
lags by one level at a LEVEL_UP on its ms is exempted cell by cell and the new count is pinned; for any other cause stop and report it
[S2-8]. ``LevelIndex(rec)`` is the same clock indexed once per match for many reference times.

``ItemIndex(rec)`` is the exact LEGACY item replay of the match (``ItemStateIndex``), built once per match; ``vectors(ref_ms)`` gives the
(10, 22) float32 item vector of the ten players in the order of LEGACY ``ITEM_VECTOR_NAMES`` (``item_vector_names()``), from the events
at or before ``ref_ms`` only. The item vectors are statistics and counts, not ids. The replay is given the champion keys and the runes of
the record because LEGACY's game rules need them (Viego's possession, Fiddlesticks' trinket, Biscuit Delivery); they do not reach the
output. Summoner spells are not passed: they matter only for patch major 16 (the top lane's role-quest item), which S2 does not read.

The LEGACY modules are verified and imported on first use, and kept for the rest of the process (``legacy.load`` checks the pin of
every file on each call, which is too slow for millions of views); a builder that wants the pins checked calls ``legacy.verify()``.
Every cast to a narrower dtype goes through ``casts.narrow``: it is range-checked and raises on overflow, naming the array; NaN is
allowed only where the design says so, here in the traits of an unknown champion [S2-4].

방법에서의 역할
---------------
이 모듈은 선수 열 명의 프레임 값이 아닌 입력, 곧 챔피언 특성, 레벨 시계, 아이템 벡터를 만든다. 특성은 그 패치의 Data Dragon 표 값이고 챔피언
번호는 행에 들어가지 않는다(패치 불변 입력 [설계서 5.3절]). 레벨 칸 여섯 개는 표의 기본값과 성장값을 위키의 성장 공식으로 현재 레벨에 맞춘 값이다(이동 속도는 기본값 그대로이다).
레벨 시계는 기준 시각까지의 LEVEL_UP 사건에서 선수별 최고 레벨을 읽는다. 아이템 벡터는 LEGACY의 정확한 아이템 재생이 만들며 이 모듈에 식이 없다.
시각 ref는 ms이고, 레벨 n은 1–18의 정수이다.

식 목록
-------
    (TRAIT-1)  성장 계수 factor(n)                                    `stat_at_level` (`_growth_factor`)
    (TRAIT-2)  레벨 n의 체력·방어력·마법 저항력·공격력             `stat_at_level`
    (TRAIT-3)  레벨 n의 공격 속도                                `attack_speed_at_level`
    (TRAIT-4)  특성 행의 열 구성과 알 수 없는 챔피언              `champion_traits`
    (TRAIT-5)  레벨 시계 level_p(ref)                            `LevelIndex.at` (`level_at`)

가정
----
- 레벨의 입력 영역은 1–18의 정수(`MAX_LEVEL`)로 선언되어 있다. `_growth_factor`, `_as_levels`, LEVEL_UP 읽기가 이 영역을 검사하여 NaN, 무한대,
  정수가 아닌 값, 범위 밖의 값을 거부한다.
- 위키 식의 보너스 항은 0으로 둔다. 곧 특성의 레벨 값은 아이템이나 룬이 더해지지 않은 값이다.
- 공격 속도 비율(ratio)은 기본 공격 속도와 같다고 둔다. Data Dragon에 비율이 없기 때문이며, 비율이 다른 소수의 챔피언에서는 위키 식과 다르다.
- 값이 정수로 내림된 프레임 능력치와 비교할 때는 특성을 내림한 뒤 비교한다(모듈 영어 본문의 측정). NaN은 알 수 없는 챔피언의 특성 행에만 있다.
"""
# Ref: LoL Wiki, "Champion statistic" (https://wiki.leagueoflegends.com/en-us/Champion_statistic), accessed 2026-10-20. Section "Primary
#      growth formula": statistic = base + bonus + growth (n - 1)(0.7025 + 0.0175 (n - 1)), here with bonus = 0; the page applies it to
#      health, armor, magic resist and attack damage, and says movement speed and range do not grow. Section "Attack speed growth": total
#      attack speed = AS_base + [AS_bonus + growth (n - 1)(0.7025 + 0.0175 (n - 1))] AS_ratio, growth in percent. Data Dragon has no attack
#      speed ratio; the wiki page "Attack speed" says most champions' base attack speed and ratio are equal, and then the total is
#      AS_base (1 + bonus): the form of spec section 11 item 11 (ratio = base). It differs from the page for the few champions whose
#      ratio differs.
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 11, items 9-11 (frame features, game-rule reuse, level-scaled statistics).
# Ref: LEGACY gameplay/champion_attributes.py::champion_vector (tags, range, resource type, base and growth fields) and
#      gameplay/item_state.py::ItemStateIndex / ITEM_VECTOR_NAMES (the exact item replay), loaded through the hash-pinned c_ept.legacy.
from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache

import numpy as np

from c_ept import legacy
from c_ept.casts import narrow
from c_ept.reader import MatchFormatError, MatchRecord
from c_ept.units import _event_id, _is_int

N_PLAYERS = 10
MAX_LEVEL = 18                  # Summoner's Rift level cap; the growth formula is checked on 1..MAX_LEVEL only
TRAIT_NAMES = ('tag_Assassin', 'tag_Fighter', 'tag_Mage', 'tag_Marksman', 'tag_Support', 'tag_Tank', 'is_ranged', 'attackrange',
               'partype_Mana', 'partype_Energy', 'partype_none', 'partype_other', 'lvl_healthMax', 'lvl_armor', 'lvl_magicResist',
               'lvl_attackDamage', 'lvl_attackSpeed', 'lvl_movementSpeed', 'champ_unknown')
_N_STATIC = TRAIT_NAMES.index('lvl_healthMax')       # the first 12 columns are the table values that do not depend on the level
_UNKNOWN = TRAIT_NAMES.index('champ_unknown')
_BASE_FIELDS = ('healthMax', 'armor', 'magicResist', 'attackDamage', 'attackSpeed', 'movementSpeed')      # LEGACY base_<stat>
_GROWTH_FIELDS = ('healthMax', 'armor', 'magicResist', 'attackDamage', 'attackSpeed_pct')                   # LEGACY growth_<stat>

_LEGACY = None


def _leg():
    """The pinned LEGACY modules: verified by ``legacy.load`` on the first call of this process, then kept."""
    global _LEGACY
    if _LEGACY is None:
        _LEGACY = legacy.load()
    return _LEGACY


def _ms(ref_ms, name: str = 'ref_ms') -> int:
    """``ref_ms`` as a Python int; TypeError unless it is a whole number of milliseconds (an int or a numpy integer; not a bool/float)."""
    if isinstance(ref_ms, (bool, np.bool_)) or not isinstance(ref_ms, (int, np.integer)):
        raise TypeError(f'{name} must be a whole number of ms (int), not {type(ref_ms).__name__}')
    return int(ref_ms)


# ---- level-scaled statistics ------------------------------------------------------------------------------------------------------
def _growth_factor(level):
    """(n - 1)(0.7025 + 0.0175 (n - 1)) for a level or an array of levels; ValueError unless every level is a whole number in 1..18.

    The factor is 0 at level 1 and 17 at level 18, so a statistic grows by ``growth`` times it (the increase from level n - 1 to n is
    growth (0.65 + 0.035 n), the wiki's second form).

    (TRAIT-1)의 factor(n)을 계산한다.
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
    """base + growth (n - 1)(0.7025 + 0.0175 (n - 1)) at level n: health, armor, magic resist and attack damage (also on arrays).

    수식
    ----
    (TRAIT-1)  factor(n) = (n − 1)·(0.7025 + 0.0175·(n − 1))
    (TRAIT-2)  stat(n) = base + growth · factor(n)
    (TRAIT-1), (TRAIT-2) = 설계서 11절 항목 11, LoL 위키 "Champion statistic"의 기본 성장 공식에서 보너스 항을 0으로 둔 꼴이다.

      기호   | 뜻                                          | 단위                         | 범위          | 코드 변수
      n      | 챔피언 레벨                                 | —                            | 1, …, 18      | level
      factor(n) | 성장 계수                                | —                            | 0 (n = 1), 17 (n = 18) | 반환값 of `_growth_factor`
      base   | 표의 기본값(레벨 1)                         | 체력 등 게임 값              | 유한          | base
      growth | 표의 레벨당 성장값                          | 게임 값 / 레벨               | 유한          | growth
      stat   | 레벨 n의 값                                 | 게임 값                      | —             | 반환값

    해석
    ----
    식 (TRAIT-2)는 레벨 n의 능력치를 기본값에 성장값과 성장 계수의 곱을 더한 값으로 놓는다. 성장 계수는 레벨 1에서 0이므로 레벨 1의 값은 기본값과
    같다. 계수가 n에 대한 이차식이어서 레벨이 오를수록 한 단계 증가분이 커지며, 레벨 18에서 성장값의 17배가 된다. 보너스 항은 0이므로
    아이템과 룬의 효과가 없는 값이다. 레벨이 정수 1–18이 아니면 거부한다.

    근거
    ----
    설계서 11절 항목 11(레벨별 능력치 식), 5.3절(패치 불변 입력), S1–S2 계획 Task 9(성장식은 위키 원문 확인 뒤).
    원전: 모듈 머리의 LoL Wiki "Champion statistic" Ref 줄(원래 기록을 가리킨다).
    """
    return base + growth * _growth_factor(level)


def attack_speed_at_level(base, growth_pct, level):
    """base (1 + growth% (n - 1)(0.7025 + 0.0175 (n - 1)) / 100) at level n; ``growth_pct`` is Data Dragon's attack speed growth in percent
    (also on arrays). This is the wiki's total attack speed with the attack speed ratio taken equal to the base attack speed.

    수식
    ----
    (TRAIT-3)  AS(n) = base · (1 + growth_pct · factor(n) / 100)                factor(n)은 (TRAIT-1)

      기호       | 뜻                                | 단위               | 범위     | 코드 변수
      AS(n)      | 레벨 n의 공격 속도                | 표의 공격 속도 단위 | —        | 반환값
      base       | 표의 기본 공격 속도               | 표의 공격 속도 단위 | 유한     | base
      growth_pct | 표의 레벨당 공격 속도 성장(퍼센트) | %                  | 유한     | growth_pct
      n          | 챔피언 레벨                       | —                  | 1, …, 18 | level
      factor(n)  | 성장 계수 (TRAIT-1)               | —                  | 0, …, 17 | `_growth_factor`의 반환값

    해석
    ----
    식 (TRAIT-3)은 공격 속도의 성장분을 기본값에 대한 비율로 보고 기본값에 곱한다. 위키의 총 공격 속도에서 비율을 기본값과 같다고 두고 보너스 항을
    0으로 둔 꼴이다. Data Dragon에 비율이 없어 이 형태를 쓰며, 비율이 다른 소수의 챔피언에서는 위키 값과 다르다 [설계서 11절 항목 11].
    성장 계수는 (TRAIT-1)과 같고 레벨이 정수 1–18이 아니면 거부한다.

    근거
    ----
    설계서 11절 항목 11(공격 속도 식, 비율 = 기본값), S1–S2 계획 Task 9.
    원전: 모듈 머리의 LoL Wiki Ref 줄(원래 기록을 가리킨다).
    """
    return base * (1.0 + growth_pct * _growth_factor(level) / 100.0)


# ---- the level clock --------------------------------------------------------------------------------------------------------------
class LevelIndex:
    """The LEVEL_UP events of one match, indexed once. ``at(ref_ms)`` is the level clock: for each player the highest level among the
    LEVEL_UP events at or before ``ref_ms`` (1 if there is none). A LEVEL_UP with an unusable participantId or level raises
    MatchFormatError (the message names the field, the type or the number and the patch, never the match key).

    선수마다 시각순 사건과 그 앞까지의 최고 레벨을 한 번 만들어 두어, 한 경기의 여러 기준 시각이 식 (TRAIT-5)를 이분 탐색으로 읽게 한다.
    """

    def __init__(self, rec: MatchRecord):
        per_player: list[list[tuple[int, int]]] = [[] for _ in range(N_PLAYERS)]
        for e in rec.events:
            if e.get('type') != 'LEVEL_UP':
                continue
            pid = _event_id(rec, e, 'participantId', 1, None, 'a LEVEL_UP')
            per_player[pid - 1].append((e['timestamp'], self._level(rec, e)))
        # (TRAIT-5): 선수별 사건을 시각순으로 놓고 누적 최대를 미리 구해 두어, at이 이분 탐색 한 번으로 max{lev_e : t_e ≤ ref}를 읽게 한다.
        self._times: list[list[int]] = []
        self._best: list[list[int]] = []                  # running maximum of the levels up to each event
        for events in per_player:
            events.sort(key=lambda te: te[0])             # stable; the record is in time order already
            self._times.append([t for t, _ in events])
            best, top = [], 0
            for _, level in events:
                top = max(top, level)
                best.append(top)
            self._best.append(best)

    @staticmethod
    def _level(rec: MatchRecord, e: dict) -> int:
        if 'level' not in e:
            raise MatchFormatError(rec.key, rec.patch, 'a LEVEL_UP has no level')
        v = e['level']
        if not _is_int(v):
            raise MatchFormatError(rec.key, rec.patch, f'a LEVEL_UP has a level of type {type(v).__name__}, not an integer')
        if not 1 <= v <= MAX_LEVEL:
            raise MatchFormatError(rec.key, rec.patch, f'a LEVEL_UP has level {v}, not in 1-{MAX_LEVEL}')
        return v

    def at(self, ref_ms) -> np.ndarray:
        """(10,) int16 levels of the players 1-10 at ``ref_ms`` (events at ``timestamp <= ref_ms``).

        수식
        ----
        (TRAIT-5)  level_p(ref) = max{ lev_e : pid_e = p + 1,  t_e ≤ ref }     (집합이 비면 1)

          기호        | 뜻                              | 단위 | 범위        | 코드 변수
          p           | 선수 번호 − 1(0부터)            | —    | 0, …, 9     | 반환 배열의 위치
          ref         | 기준 시각                       | ms   | 정수        | ref_ms
          e           | 한 선수의 LEVEL_UP 사건         | —    | —           | per_player[p]의 항목
          pid_e       | 사건 e의 참가자 번호(p + 1과 같은 선수) | — | 1, …, 10    | `participantId`
          t_e, lev_e  | 사건 e의 시각, 올린 레벨        | ms, — | 정수; 1–18 | `_times`, `_best`
          level_p     | 선수 p의 레벨                   | —    | 1, …, 18    | 반환값(int16, `narrow`)

        해석
        ----
        식 (TRAIT-5)는 기준 시각 이전 또는 같은 시각에 일어난 LEVEL_UP 가운데 가장 높은 레벨을 그 시각의 레벨로 읽는다. 시각 경계는 포함이므로
        기준 시각과 같은 ms의 LEVEL_UP이 반영된다. 사건이 없으면 1이다. 누적 최대를 읽으므로 레벨이 내려가는 기록이 있어도 읽은 값은 ref가 커질 때 줄지
        않는다.

        근거
        ----
        S1–S2 계획 Task 9(레벨 시계), S1–S2 판정 S2-8(프레임 레벨과의 같은 ms 차이).
        """
        ref = _ms(ref_ms)
        levels = []
        for times, best in zip(self._times, self._best):
            j = bisect_right(times, ref)
            levels.append(best[j - 1] if j else 1)
        return narrow('level_at', np.array(levels, dtype=np.int64), np.int16)


def level_at(rec: MatchRecord, ref_ms) -> np.ndarray:
    """(10,) int16: the level of each player at ``ref_ms`` by the LEVEL_UP events at or before it (1 if none). It indexes the events on
    every call; for many reference times of one match build a ``LevelIndex`` once.

    식 (TRAIT-5)를 계산한다.
    """
    return LevelIndex(rec).at(ref_ms)


# ---- champion traits --------------------------------------------------------------------------------------------------------------
def _as_levels(levels) -> np.ndarray:
    """``levels`` as (10,) float64 whole levels in 1..18 (an int16 array of ``level_at`` or the float frame values); else ValueError."""
    a = np.asarray(levels)
    if a.shape != (N_PLAYERS,) or a.dtype.kind not in 'iuf':
        raise ValueError(f'levels must be a numeric array of shape ({N_PLAYERS},), got shape {a.shape} and dtype {a.dtype}')
    a = a.astype(np.float64)
    if not (np.isfinite(a).all() and (a == np.floor(a)).all() and ((a >= 1) & (a <= MAX_LEVEL)).all()):
        raise ValueError(f'levels must be whole levels in 1..{MAX_LEVEL} (no NaN), got {a.tolist()}')
    return a


@lru_cache(maxsize=None)
def _table_row(patch: str, champ: int):
    """(the 12 level-free trait values, the 6 base values, the 5 growth values) of a champion in the table of ``patch``; None if the
    champion is not in it. ValueError if the table lacks a number the traits need (a known champion must have every value)."""
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
    """(10, 19) float32 traits (``TRAIT_NAMES``) of the players 1-10 at the given ``levels`` (10 whole levels 1-18).

    A champion that is not in the table of ``rec.patch`` is ``champ_unknown = 1`` with NaN in the other 18 columns; for the others
    ``champ_unknown = 0`` and every value is finite.

    수식
    ----
    (TRAIT-4)  행 p의 열: 처음 12개 = 표 값(레벨 무관);  레벨 칸 = (TRAIT-2)(체력, 방어력, 마법 저항력, 공격력), (TRAIT-3)(공격 속도),
               base(이동 속도: 성장 없음);  champ_unknown = 0.  표에 없는 챔피언: 앞의 18개 열 NaN, champ_unknown = 1.  마지막에 float32로 줄인다.

      기호          | 뜻                          | 단위 | 범위             | 코드 변수
      p             | 선수 번호 − 1(0부터)         | —    | 0, …, 9          | 행 번호
      n             | 선수 p의 레벨                | —    | 1, …, 18         | levels[p]
      champ_unknown | 알 수 없는 챔피언 표시       | —    | 0, 1             | `champ_unknown` 열

    해석
    ----
    식 (TRAIT-4)는 한 선수의 특성 행을 표 값과 레벨 칸으로 채우고, 그 패치의 표에 없는 챔피언(결측이거나 모드 키)은 알 수 없음 표시와 NaN으로 채운다.
    NaN은 특성 행에서만 허용된다 [S1–S2 판정 S2-4]. 챔피언 번호는 표를 찾는 데만 쓰이고 행에는 들어가지 않는다 [설계서 5.3절]. 레벨은 10개의
    정수 1–18이어야 하고 아니면 거부한다. 레벨 칸의 이동 속도는 성장이 없다고 둔다.

    근거
    ----
    설계서 5.3절(패치 불변: 챔피언 ID 임베딩 없음), 11절 항목 9–11, S1–S2 계획 Task 9(표 밖 챔피언 처리).
    """
    lv = _as_levels(levels)
    rows = [_table_row(rec.patch, int(c)) for c in rec.champ]
    # (TRAIT-4): 모든 칸을 NaN으로 시작하여 알 수 없는 챔피언의 행은 champ_unknown만 1이 되고, 아는 챔피언의 행은 아래에서 모든 칸이 유한하게 채워진다.
    out = np.full((N_PLAYERS, len(TRAIT_NAMES)), np.nan)
    known = np.array([r is not None for r in rows])
    out[~known, _UNKNOWN] = 1.0
    if known.any():
        idx = np.flatnonzero(known)
        static = np.array([rows[i][0] for i in idx])
        base, growth, n = np.array([rows[i][1] for i in idx]), np.array([rows[i][2] for i in idx]), lv[idx]
        # 앞 12개 칸은 표 값 그대로이고, 다음 네 칸은 (TRAIT-2), 공격 속도 칸은 (TRAIT-3), 이동 속도 칸은 성장 없는 기본값이다.
        out[idx, :_N_STATIC] = static
        out[idx, _N_STATIC:_N_STATIC + 4] = stat_at_level(base[:, :4], growth[:, :4], n[:, None])    # health, armor, magic resist, damage
        out[idx, _N_STATIC + 4] = attack_speed_at_level(base[:, 4], growth[:, 4], n)
        out[idx, _N_STATIC + 5] = base[:, 5]                                                         # movement speed does not grow
        out[idx, _UNKNOWN] = 0.0
    return narrow('champion_traits', out, np.float32)


# ---- item vectors -----------------------------------------------------------------------------------------------------------------
def item_vector_names() -> tuple:
    """The 22 columns of ``ItemIndex.vectors``: LEGACY ``ITEM_VECTOR_NAMES``, in its order.

    아이템 벡터의 열 이름이며 LEGACY의 순서를 그대로 따른다.
    """
    return _leg().item_state.ITEM_VECTOR_NAMES


@lru_cache(maxsize=None)
def _effect_flags(patch: str):
    """The effect-flag id sets of ``patch``, loaded together with the item table of the patch as LEGACY's own default does: a flagged id
    that the table lacks raises KeyError, so a table that does not fit the flags stops the build."""
    item_state = _leg().item_state
    return item_state.load_effect_flags(item_state.load_item_table_v2(patch), patch=patch)


class ItemIndex:
    """The exact LEGACY item replay of one match (build it once per match); ``vectors(ref_ms)`` is (10, 22) float32.

    Row i is the player with participant id i + 1, columns follow ``item_vector_names()``. Only the events at or before ``ref_ms`` are
    read. Values are finite; a value that is not finite or does not fit float32 raises.

    LEGACY의 정확한 아이템 재생이 만든 능력치와 개수를 기준 시각까지의 사건만으로 읽는 단계이며, 값이 유한하지 않으면 거부한다.
    이 모듈은 그 식을 다시 쓰지 않는다 [설계서 11절 항목 10].
    """

    def __init__(self, rec: MatchRecord):
        item_state = _leg().item_state
        champion_by_pid = {str(i + 1): int(c) for i, c in enumerate(rec.champ)}
        self._names = item_state.ITEM_VECTOR_NAMES
        self._index = item_state.ItemStateIndex(
            rec.events, item_state.load_item_table_v2(rec.patch), exact=True, effect_flags=_effect_flags(rec.patch),
            champion_by_pid=champion_by_pid, runes_by_pid=rec.runes_by_pid, patch=rec.patch)

    def vectors(self, ref_ms) -> np.ndarray:
        per_player = self._index.vectors_at(_ms(ref_ms))
        out = np.array([[per_player[pid][name] for name in self._names] for pid in range(1, N_PLAYERS + 1)], dtype=np.float64)
        if not np.isfinite(out).all():
            raise ValueError('item_vectors: the legacy replay gave a value that is not finite')
        return narrow('item_vectors', out, np.float32)
