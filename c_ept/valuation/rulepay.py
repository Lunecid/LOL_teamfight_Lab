r"""
Game Rule-Based Kill & Assist Payouts / LoL 게임 공식 규칙 기반 킬 및 어시스트 골드 풀 산출기

[English]
Computes exact in-game gold rewards for champion kills and assists to label economic fight outcomes (f0):
  - Motivation: Riot Timeline files record the killer's bounty and shutdown bounty, but omit the assist pool
    distributed to teammates. This module reconstructs the assist pool from official game mechanics.
  - Formula (patches 15.18 - 15.22):
      pool = min(pool_share * recorded_bounty, cap_share * base(victim_level)) * mult(t)
        * pool_share = cap_share = 0.5 (50% bounty share capped at 50% base bounty).
        * mult(t): Early-game scaling: 0.5 until 90s, ramping linearly to 1.0 at 210s.
        * base(level): 300g (levels 1-6), increasing by +10g per level up to 420g at level 18.
  - Rule Verification (`RuleTable`): Manages verified parameter schemas across patch ranges (RULE_PAYOUTS.json).

[한국어 설명]
교전의 경제적 가치 평가($f_0$)를 위해 공식 롤 게임 메커니즘을 바탕으로 킬 및 어시스트 보상 골드를 산출합니다:
  - 목적: 라이엇 타임라인에는 킬러가 획득한 기본 현상금(bounty)과 제압 골드(shutdown bounty)만 기록되어 있고, 어시스트 소환사들에게 분배된 골드 풀은 누락되어 있습니다.
  - 어시스트 풀 계산 수식 (15.18 ~ 15.22 패치):
      $\text{pool} = \min(0.5 \times \text{bounty}, 0.5 \times \text{base}(\text{피해자 레벨})) \times \text{mult}(t)$
        * 초반 감쇠 배수 $\text{mult}(t)$: 90초까지 0.5배, 이후 210초까지 1.0배로 선형 증가.
        * 피해자 레벨별 기본 현상금 $\text{base}(\text{level})$: 1~6레벨 300골드, 7레벨부터 레벨당 +10골드씩 증가하여 18레벨에 최대 420골드.
  - 규칙표 관리(`RuleTable`): 패치별 게임 변경사항을 엄격히 검증하여 신뢰성 있는 보상 테이블(`RULE_PAYOUTS.json`) 제공.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decision P3 (the assist pool of a kill), Task 3, and the rulings R7 and R8 of the S3
#      ledger (the built-in P3 rules; the loader refuses a table that disagrees with them).
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 8 (the label f0: kill gold = bounty + shutdown + assist pool).
# Ref: Riot Games, League of Legends patch notes 14.21 (a positive bounty is a bonus for the killer; a negative bounty is also factored
#      into assist gold) and 25.09 (base kill gold +10 per level from level 7 to a maximum of 420); League of Legends Wiki, "Champion gold
#      bounties" (the assist pool is half of the kill bounty, capped at half of the base bounty, and reduced early in the game: history
#      line V4.20). The checked values, their status and their sources are in RULE_PAYOUTS.json of the s3a run (rules/).
from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = 'rule_payouts_v2'
F0_RULES = ('champion_kill_base_bounty', 'assist_bounty', 'assist_bounty_early_game_multiplier', 'first_blood_bonus')
P3_PATCHES = ('15.18', '15.19', '15.20', '15.21', '15.22')           # the patches of the approved decision P3 (the S3 data)
N_LEVELS = 18

_PATCH = re.compile(r'(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})')       # 'major.minor' as the data folders and Match-V5 write it, canonical


# ---- patches -----------------------------------------------------------------------------------------------------------------------
def _patch_key(patch, what: str = 'a patch') -> tuple:
    """``(major, minor)`` of a patch string such as '15.20'; ValueError for anything else. The message never holds the value."""
    match = _PATCH.fullmatch(patch) if isinstance(patch, str) else None
    if match is None:
        raise ValueError(f'{what} must be a string like "15.20" (major.minor, digits only, no leading zeros)')
    return int(match[1]), int(match[2])


def _fmt(key: tuple) -> str:
    return f'{key[0]}.{key[1]}'


# ---- the values of one patch -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchRules:
    """What f0 needs from the rules of one patch (the field names are those of the table).

    ``base_by_level``: the base bounty of a champion of level 1 .. 18 (a tuple of 18 whole amounts). ``pool_share``, ``cap_share_of_base``:
    the pool is at most ``pool_share`` of the recorded bounty and at most ``cap_share_of_base`` of the base bounty. The early-game
    multiplier is ``start_multiplier`` until ``flat_until_s`` seconds of game time and rises linearly to ``end_multiplier`` at
    ``full_at_s``. ``first_blood_killer_bonus`` is the killer's First Blood gold (not part of the pool; the caller's business) and
    ``first_blood_assist_pool_bonus`` is always 0: a table with another value is refused.

    식 (PAY-1)–(PAY-3)의 매개변수를 한 패치 몫으로 묶는다: base_by_level은 (PAY-1), 배수의 네 값은 (PAY-2), pool_share와 cap_share_of_base는 (PAY-3)에 쓴다.
    """
    base_by_level: tuple
    pool_share: float
    cap_share_of_base: float
    flat_until_s: float
    full_at_s: float
    start_multiplier: float
    end_multiplier: float
    first_blood_killer_bonus: int
    first_blood_assist_pool_bonus: int


@dataclass(frozen=True)
class _Segment:
    """The parsed value of one rule for the patches ``lo`` .. ``hi`` (inclusive; both are ``(major, minor)``)."""
    lo: tuple
    hi: tuple
    value: tuple


def _number(x, where: str, *, lo=None, hi=None, lo_open: bool = False, whole: bool = False):
    """A finite number of the table (never a bool or a text) inside the bounds, as an int when ``whole`` and else a float."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f'{where} must be a finite number')
    try:
        finite = math.isfinite(x)
    except OverflowError:                       # a whole number too large for a float
        finite = False
    if not finite:
        raise ValueError(f'{where} must be a finite number')
    if whole and x != int(x):
        raise ValueError(f'{where} must be a whole number')
    if lo is not None and (x <= lo if lo_open else x < lo):
        raise ValueError(f'{where} must be {">" if lo_open else ">="} {lo}')
    if hi is not None and x > hi:
        raise ValueError(f'{where} must be <= {hi}')
    return int(x) if whole else float(x)


def _need(value, field: str, where: str):
    if not isinstance(value, dict) or field not in value:
        raise ValueError(f'{where}: the value of the entry has no "{field}"')
    return value[field]


def _parse_base(value, where: str) -> tuple:
    levels = _need(value, 'by_level', where)
    if not isinstance(levels, list) or len(levels) != N_LEVELS:
        raise ValueError(f'{where}: by_level must be a list of {N_LEVELS} amounts, one for each level 1 .. {N_LEVELS}')
    return tuple(_number(g, f'{where}: by_level[{i}]', lo=0, lo_open=True, whole=True) for i, g in enumerate(levels))


def _parse_assist(value, where: str) -> tuple:
    return (_number(_need(value, 'pool_share', where), f'{where}: pool_share', lo=0, lo_open=True, hi=1),
            _number(_need(value, 'cap_share_of_base', where), f'{where}: cap_share_of_base', lo=0, lo_open=True, hi=1))


def _parse_multiplier(value, where: str) -> tuple:
    flat = _number(_need(value, 'flat_until_s', where), f'{where}: flat_until_s', lo=0)
    full = _number(_need(value, 'full_at_s', where), f'{where}: full_at_s', lo=0)
    if full <= flat:
        raise ValueError(f'{where}: full_at_s must be later than flat_until_s')
    start = _number(_need(value, 'start_multiplier', where), f'{where}: start_multiplier', lo=0, hi=1)
    end = _number(_need(value, 'end_multiplier', where), f'{where}: end_multiplier', lo=0, hi=1)
    return flat, full, start, end


def _parse_first_blood(value, where: str) -> tuple:
    killer = _number(_need(value, 'killer_bonus', where), f'{where}: killer_bonus', lo=0, whole=True)
    assist = _number(_need(value, 'assist_pool_bonus', where), f'{where}: assist_pool_bonus', lo=0, whole=True)
    if assist != 0:
        raise ValueError(f'{where}: assist_pool_bonus is not 0, but the assist pool has no First Blood term in this module: '
                         'the table cannot be priced correctly')
    return killer, assist


_PARSERS = dict(zip(F0_RULES, (_parse_base, _parse_assist, _parse_multiplier, _parse_first_blood)))


def _entry_range(entry, where: str) -> tuple:
    pair = entry.get('patch_range')
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f'{where}: patch_range must be a pair of patches [first, last]')
    lo, hi = _patch_key(pair[0], f'{where}: patch_range'), _patch_key(pair[1], f'{where}: patch_range')
    if lo > hi:
        raise ValueError(f'{where}: patch_range runs backwards')
    return lo, hi


def _asked_patches(patches) -> set:
    if isinstance(patches, str):
        patches = (patches,)
    keys = {_patch_key(p, 'a requested patch') for p in patches}
    if not keys:
        raise ValueError('patches is empty: ask for at least one patch, or for None')
    return keys


def _check_no_overlap(rid: str, spans) -> None:
    """ValueError when two of the ``(lo, hi)`` spans of rule ``rid`` cover the same patch (sorted neighbours are enough to see it)."""
    ordered = sorted(spans)
    for (_, hi), (lo, _) in zip(ordered, ordered[1:]):
        if lo <= hi:
            raise ValueError(f'{rid}: two entries cover the same patch')


# ---- the table ---------------------------------------------------------------------------------------------------------------------------
class RuleTable:
    """The four f0 rules for a set of patch ranges. Make one with ``RuleTable.load``; the built-in one is ``P3_RULES``.

    ``segments``: {rule id: segments}, every rule of ``F0_RULES``; two segments of one rule must not cover the same patch. ``served``:
    None (the table serves every patch that all four rules have a segment for) or the patches it was loaded for, as ``(major, minor)``
    keys: it serves those and no others, however wide its segments are.
    """

    def __init__(self, segments, served=None):
        if set(segments) != set(F0_RULES):
            raise ValueError('a rule table needs exactly the four f0 rules')
        self._segments = {}
        for rid in F0_RULES:
            _check_no_overlap(rid, [(s.lo, s.hi) for s in segments[rid]])
            self._segments[rid] = tuple(segments[rid])
        self._served = None if served is None else frozenset(served)

    def _lookup(self, key: tuple):
        """The ``PatchRules`` of the patch ``key``, or None when the table does not serve it (it was loaded for other patches, or one of
        the four rules has no entry for it)."""
        if self._served is not None and key not in self._served:
            return None
        found = []
        for rid in F0_RULES:
            hit = [s.value for s in self._segments[rid] if s.lo <= key <= s.hi]
            if not hit:
                return None
            found.append(hit[0])
        base, (share, cap), (flat, full, start, end), (fb_killer, fb_assist) = found
        return PatchRules(base, share, cap, flat, full, start, end, fb_killer, fb_assist)

    def for_patch(self, patch) -> PatchRules:
        """The rules of ``patch`` (a string such as '15.20'); ValueError when the table does not serve the patch.

        표가 다루지 않는 패치에는 값을 만들지 않고 오류를 내어, 규칙이 없는 패치에 가격이 매겨지는 일을 막는다(판정 R10).
        """
        key = _patch_key(patch)
        rules = self._lookup(key)
        if rules is None:
            raise ValueError(f'the rule table does not serve patch {_fmt(key)} (it was loaded for other patches, or one of the four f0 '
                             'rules has no entry for it)')
        return rules

    @classmethod
    def load(cls, path, patches=None) -> RuleTable:
        """Read a ``rule_payouts_v2`` file; refuse it (ValueError) when f0 cannot trust it. See the module text for the refusals.

        ``patches``: None (every f0 entry of the file must be verified, and the table serves every patch they cover), or the patches to
        serve (a string, or any collection of strings): only the entries that cover one of them are read, each of them must be covered
        by all four rules, and the table serves exactly these patches and no others, however wide the entries are. Two entries of one
        rule that cover the same patch refuse the file in both cases, also when neither of them (or only one) is read.

        규칙표 파일에서 f₀의 규칙 네 가지를 읽어, 읽는 항목의 상태가 verified이고 표가 다루는 15.18–15.22 패치의 값이 승인된 결정 P3와 같을 때만 표를 만든다(판정 R8, R10).
        어긋난 표로 가격을 매기면 값이 조용히 틀리므로, 의심스러운 표는 받아들이지 않고 멈춘다.
        """
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
            raise ValueError(f'the rule table must be a "{SCHEMA}" document')
        rules = doc.get('rules')
        if not isinstance(rules, dict):
            raise ValueError('the rule table has no "rules" mapping')
        asked = None if patches is None else _asked_patches(patches)
        segments = {}
        for rid in F0_RULES:
            entries = rules.get(rid)
            if not isinstance(entries, list) or not entries:
                raise ValueError(f'{rid}: the table has no entries for this f0 rule')
            spans = []
            for number, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(f'{rid}[{number}]: an entry must be an object')
                spans.append(_entry_range(entry, f'{rid}[{number}]'))
            _check_no_overlap(rid, spans)                                 # every entry counts, whether it is read or not
            kept = []
            for number, (entry, (lo, hi)) in enumerate(zip(entries, spans)):
                if asked is not None and not any(lo <= key <= hi for key in asked):
                    continue                                              # an entry for other patches: not read, whatever its status
                if entry.get('status') != 'verified':
                    raise ValueError(f'{rid} for patches {_fmt(lo)} .. {_fmt(hi)}: the status is {entry.get("status")!r}, not "verified": '
                                     'f0 uses verified rules only')
                kept.append(_Segment(lo, hi, _PARSERS[rid](entry.get('value'), f'{rid}[{number}]')))
            segments[rid] = tuple(kept)
        if asked is not None:
            for key in sorted(asked):
                for rid in F0_RULES:
                    if not any(s.lo <= key <= s.hi for s in segments[rid]):
                        raise ValueError(f'{rid}: the table has no entry for patch {_fmt(key)}')
        table = cls(segments, served=asked)                               # asked patches: served exactly; None: every patch covered
        for patch in P3_PATCHES:                                          # one source of truth: the file must say what P3 says
            key = _patch_key(patch)
            mine = table._lookup(key)
            if mine is None:
                continue
            approved = P3_RULES._lookup(key)
            differ = [f.name for f in dataclasses.fields(PatchRules) if getattr(mine, f.name) != getattr(approved, f.name)]
            if differ:
                raise ValueError(f'patch {patch}: the table disagrees with the approved decision P3 in {", ".join(differ)}')
        return table


def _p3_segments() -> dict:
    """(PAY-1)의 내장 표와 (PAY-2), (PAY-3)의 내장 매개변수(결정 P3)를 규칙표의 한 구간으로 만든다."""
    lo, hi = _patch_key(P3_PATCHES[0]), _patch_key(P3_PATCHES[-1])
    base = tuple(300 + 10 * max(0, level - 6) for level in range(1, N_LEVELS + 1))     # 300 for levels 1-6, +10 per level from 7, 420 at 18
    return {'champion_kill_base_bounty': (_Segment(lo, hi, base),),
            'assist_bounty': (_Segment(lo, hi, (0.5, 0.5)),),                              # the pool: 0.5 x bounty, at most 0.5 x base
            'assist_bounty_early_game_multiplier': (_Segment(lo, hi, (90.0, 210.0, 0.5, 1.0)),),    # 0.5 until 1:30, 1.0 at 3:30
            'first_blood_bonus': (_Segment(lo, hi, (0, 0)),)}                              # no First Blood gold in 15.18-15.22


P3_RULES = RuleTable(_p3_segments())          # the approved decision P3 as a table; the default of every function below


def _rules(rules) -> RuleTable:
    if rules is None:
        return P3_RULES
    if not isinstance(rules, RuleTable):
        raise TypeError('rules must be a RuleTable (RuleTable.load(path)) or None')
    return rules


# ---- inputs ------------------------------------------------------------------------------------------------------------------------------
def _nonnegative(name: str, x) -> np.ndarray:
    """``x`` as a float64 array of finite numbers that are not negative (ValueError otherwise; the message holds counts, not values)."""
    a = np.asarray(x)
    if a.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be numbers')
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f'{name}: {int((~np.isfinite(a)).sum())} value(s) are not finite')
    if (a < 0).any():
        raise ValueError(f'{name}: {int((a < 0).sum())} value(s) are negative')
    return a


def _whole(name: str, x, lo: int, hi: int | None = None) -> np.ndarray:
    """``x`` as an array of whole numbers in ``lo`` .. ``hi`` (floats that are whole are fine); ValueError otherwise, without the values."""
    a = np.asarray(x)
    if a.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be whole numbers')
    if a.dtype.kind == 'f':
        if not np.isfinite(a).all():
            raise ValueError(f'{name}: {int((~np.isfinite(a)).sum())} value(s) are not finite')
        if (a != np.floor(a)).any():
            raise ValueError(f'{name}: {int((a != np.floor(a)).sum())} value(s) are not whole numbers')
    outside = (a < lo) if hi is None else ((a < lo) | (a > hi))
    if outside.any():
        raise ValueError(f'{name}: {int(outside.sum())} value(s) outside {lo} .. {"infinity" if hi is None else hi}')
    return a


def _patch_array(patch) -> np.ndarray:
    a = np.asarray(patch)
    if a.dtype.kind == 'O' and all(isinstance(p, str) for p in a.ravel()):
        a = a.astype(str)
    if a.dtype.kind != 'U':
        raise ValueError('patch must be a string, or an array of strings, like "15.20"')
    return a


def _common_shape(*arrays) -> tuple:
    shapes = {a.shape for a in arrays if a.ndim}
    if len(shapes) > 1:
        raise ValueError('the arguments must all have one shape (a single value may stand for the whole array)')
    return shapes.pop() if shapes else ()


# ---- the functions -----------------------------------------------------------------------------------------------------------------------
def base_bounty(level, patch, rules=None) -> int:
    """The base bounty of a champion of ``level`` (1 .. 18) on ``patch``, as a whole amount of gold.

    ``rules``: a loaded ``RuleTable``, or None for the built-in P3 rules (15.18-15.22).

    수식
    ----
    (PAY-1)  base(lvl) = by_level[lvl − 1]                   (lvl = 1, …, 18)
             내장 P3 표(15.18–15.22)는 base(lvl) = 300 + 10·max(0, lvl − 6)이다.

      기호        | 뜻                    | 단위 | 범위             | 코드 변수
      lvl         | 챔피언(피해자)의 레벨 | 레벨 | 1, …, 18         | level
      by_level[·] | 패치의 기본 현상금 표 | 골드 | 18개의 양의 정수 | PatchRules.base_by_level
      base(lvl)   | 그 레벨의 기본 현상금 | 골드 | 양의 정수        | 반환값

    해석
    ----
    식 (PAY-1)은 기본 현상금이 레벨마다 정해진 표의 값이라는 규칙이며, 도움 풀의 상한 cap·base(lvl)에 들어간다((PAY-3)). 내장 표는 레벨 6까지 같은 값이고 7부터 레벨마다 같은 양씩
    늘어난다(판정 R8). 레벨이 1–18의 정수가 아니거나 표가 다루지 않는 패치이면 틀린 값이 조용히 나오지 않도록 거부한다(판정 R8).

    근거
    ----
    계획 P3(도움 풀의 상한은 피해자 레벨의 기본 현상금), Task 3(기본 현상금 표), 판정 R8(내장 P3 규칙; 다루지 않는 패치는 오류).
    """
    rules = _rules(rules)
    level = _whole('level', level, 1, N_LEVELS)
    if level.ndim:
        raise ValueError('level must be a single whole number')
    return rules.for_patch(patch).base_by_level[int(level) - 1]


def assist_pools(bounty, victim_level, t_ms, patch, n_assisters, rules=None) -> np.ndarray:
    """The assist pool of each kill, in gold: the array form of ``assist_pool``, as a float64 array of the shape of the arguments.

    ``bounty``: the recorded bounty of the kill (not negative). ``victim_level``: the level of the victim when it died (1 .. 18).
    ``t_ms``: the time of the kill in ms of game time (not negative). ``patch``: one patch string, or an array of them. ``n_assisters``:
    the number of assisters (0 or more; only whether it is at least 1 matters). An argument is an array of one common shape, or a single
    value that stands for the whole array. An unknown patch, a level outside 1 .. 18, a negative or non-finite bounty or time, a
    negative or fractional number of assisters, or arrays of different shapes raise ValueError; so does an argument that is not a number
    (or, for ``patch``, not a string). The inputs are not changed.

    The result is ``min(pool_share x bounty, cap_share_of_base x base(victim_level)) x mult(t)`` where there is at least one assister and
    0.0 elsewhere (see the module text). An execution has no assisters; whether a kill is an execution is for the caller to decide.

    수식
    ----
    (PAY-2)  mult(t) = start + (end − start)·clip((t/1000 − t_flat) / (t_full − t_flat), 0, 1)
    (PAY-3)  pool = min(share·bounty, cap·base(lvl))·mult(t)   (n_assisters ≥ 1),      pool = 0   (n_assisters = 0)
             clip(x, 0, 1) = min(max(x, 0), 1)이다.
             내장 P3 규칙(15.18–15.22)은 share = cap = 0.5, start = 0.5, end = 1.0, t_flat = 90 s, t_full = 210 s이다.

      기호           | 뜻                                                        | 단위       | 범위                | 코드 변수
      t              | 킬 시각(경기 시간; 식에서 1000으로 나누어 s로 쓴다)       | ms         | ≥ 0                 | t_ms
      start, end     | 초반 배수의 처음 값, 끝 값                                | —          | [0, 1]              | start_multiplier, end_multiplier
      t_flat, t_full | 배수가 처음 값에 머무는 끝 시각, 끝 값에 닿는 시각        | s          | 0 ≤ t_flat < t_full | flat_until_s, full_at_s
      mult(t)        | 초반 배수                                                 | —          | start와 end 사이    | (식 안에서만 쓴다)
      bounty         | 킬의 기록 bounty                                          | 골드       | ≥ 0                 | bounty
      share, cap     | bounty에서 풀이 가져가는 몫, 기본 현상금에서 풀의 상한 몫 | —          | (0, 1]              | pool_share, cap_share_of_base
      lvl, base(lvl) | 피해자 레벨, 그 레벨의 기본 현상금((PAY-1))               | 레벨, 골드 | 1–18; 양의 정수     | victim_level
      n_assisters    | 킬의 도움 참가자 수                                       | 명         | 0, 1, …             | n_assisters
      pool           | 킬의 도움 풀(도움 참가자 전체의 합계)                     | 골드       | ≥ 0                 | 반환값

    해석
    ----
    식 (PAY-3)은 도움 참가자 전체에게 지급되는 골드의 합계를 기록 bounty의 share만큼으로 하되 피해자 레벨의 기본 현상금의 cap만큼을 넘지 않게 자르고 초반 배수로 줄인 값이다(계획 P3).
    배수 (PAY-2)는 t_flat까지 start이고 이후 t_full까지 선형으로 변해 end에 닿은 뒤 end를 유지하므로, t가 어디에 있든 start와 end 사이이며 내장 P3에서는 증가한다(계획 P3).
    풀은 합계이므로 도움 참가자 수는
    풀이 있는지만 정하고 크기는 바꾸지 않으며, 참가자가 없으면 풀이 0이다(계획 P3). 다루지 않는 패치나 정의역 밖의 입력은 틀린 가격이 조용히 나오지 않도록 거부한다(판정 R8).
    표의 몫과 배수가 범위 안인지는 이 함수가 아니라 표를 읽을 때 검사한다(모듈 머리의 거부 목록).

    근거
    ----
    계획 P3(도움 풀 식, 초반 배수), Task 3(도움 풀), 판정 R8(내장 P3 규칙; 다루지 않는 패치는 오류), 설계서 8절(f₀의 도움 규칙 골드), 모듈 머리의 Ref 줄(패치 노트와 위키 출처).
    """
    rules = _rules(rules)
    gold = _nonnegative('bounty', bounty)
    level = _whole('victim_level', victim_level, 1, N_LEVELS)
    seconds = _nonnegative('t_ms', t_ms)
    assisted = _whole('n_assisters', n_assisters, 0) >= 1
    names = _patch_array(patch)
    shape = _common_shape(gold, level, seconds, assisted, names)
    size = int(np.prod(shape, dtype=np.int64))
    if names.ndim == 0:
        unique, which = names.reshape(1), np.zeros(size, dtype=np.intp)
    else:
        unique, which = np.unique(names.ravel(), return_inverse=True)
        which = which.ravel()
    rows = [rules.for_patch(str(p)) for p in unique]                       # ValueError for a patch the rules do not serve

    def per_patch(field) -> np.ndarray:
        return np.array([getattr(r, field) for r in rows], dtype=np.float64)[which]

    def flat(a) -> np.ndarray:
        return np.broadcast_to(a, shape).ravel()

    # 식 (PAY-1)의 base(lvl): 킬마다 그 패치의 표에서 피해자 레벨의 값을 찾는다.
    base = np.array([r.base_by_level for r in rows], dtype=np.float64).reshape(len(rows), N_LEVELS)[which, flat(level).astype(np.int64) - 1]
    flat_s, full_s = per_patch('flat_until_s'), per_patch('full_at_s')
    start, end = per_patch('start_multiplier'), per_patch('end_multiplier')
    # 식 (PAY-3)의 min(share·bounty, cap·base(lvl)): bounty가 크면 기본 현상금의 몫이 풀의 상한이 된다.
    pool = np.minimum(per_patch('pool_share') * flat(gold), per_patch('cap_share_of_base') * base)
    # 식 (PAY-2)의 clip 항: 시각이 t_flat 앞이면 0, t_full 뒤면 1이다. 인자는 ms이므로 1000으로 나눈다.
    ramp = np.clip((flat(seconds) / 1000.0 - flat_s) / (full_s - flat_s), 0.0, 1.0)
    # 식 (PAY-3): 도움 참가자가 1명 이상인 킬에만 풀을 주고 나머지는 0이다.
    out = np.where(flat(assisted), pool * (start + (end - start) * ramp), 0.0)
    return out.reshape(shape)


def assist_pool(bounty, victim_level, t_ms, patch, n_assisters, rules=None) -> float:
    """The assist pool of one kill, in gold (the total over all assisters; not rounded): ``assist_pools`` for single values.

    ``assist_pool(300, 6, 150_000, '15.20', 1)`` is 112.5: min(0.5 x 300, 0.5 x 300) x 0.75 at 2:30. With no assister (0) it is 0.0.
    ValueError for the inputs that ``assist_pools`` refuses, and for arrays (use ``assist_pools``).

    식 (PAY-2), (PAY-3)은 assist_pools에 있고, 이 함수는 같은 계산을 단일 값으로 부른다.
    """
    out = assist_pools(bounty, victim_level, t_ms, patch, n_assisters, rules)
    if out.ndim:
        raise ValueError('assist_pool prices one kill: use assist_pools for arrays')
    return float(out)
