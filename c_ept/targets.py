"""Targets of an engagement unit: the kill gold K, the objectives O the unit claims, and the conversions that follow it.

K (``unit_K``, ``unit_exec``). Per unit, from its own kills only (kills of other units in the same window are not in it): for each team
the kills it scored, the recorded bounty, the recorded shutdown bounty and the assisters of those kills; cell = 2 * field + team with
``K_FIELDS`` naming the fields and team 0 blue, 1 red. Only a kill whose killer is a champion (1-10) counts for a team. An execution (the
killer is no champion: minions, tower, monster) pays no team, assisters included, and is counted by its victim's team in ``unit_exec``.

O (``ObjectiveTable``, ``objective_table``, ``unit_O``). One row per ELITE_MONSTER_KILL, BUILDING_KILL and TURRET_PLATE_DESTROYED of the
match, at any time of it, in event order:

    kind    index in ``config.O_KINDS`` (-1: another monster or building); the elder dragon is its own kind
    taker   the team that got it: 0 blue, 1 red, -1 unknown (not an error). A monster has its killerTeamId; a building or plate has the
            teamId of the owner who lost it, so the taker is the other team. A team id is the integer 100 or 200: anything else (absent,
            another number, a float, a bool, a string) is unknown
    x, y, has_pos   position by ``units.event_xy``; NaN and False when the event has none
    cred    bit mask (bit pid - 1) of the credited champions 1-10: the killer and the assisters; a plate has its killer only (its
            assistingParticipantIds are read and must be well formed, but they are not credited)
    unit    index of the unit the objective is attributed to (G3), -1 for none
    sub     tower tier by towerType (0 outer, 1 inner, 2 base, 3 nexus turret) of a tower (BUILDING_KILL with buildingType
            TOWER_BUILDING), -1 for everything else, an inhibitor with a towerType included; O has one tower kind, so the tier is
            kept here for the flow prices (the cell of ``unit_O`` is 2 * kind + taker whatever the tier)

Attribution (G3). An objective with a position, a known kind and a known taker goes to the unit among all units, in scope or not, with
tau <= t < delta whose people (killers, victims and assisters of its kills, ``units.kill_people_mask``) share at least one champion with
``cred``: the unit with the most shared champions, then the one with the nearest kill (smallest Euclidean distance from the objective
to a kill of the unit that has a position; infinite for a unit without one), then the earlier unit. An objective without a champion
credit, without a position, or with an unknown kind or taker is attributed to none. The attribution is made in hindsight: it sees all
kills of every unit, and units whose first kill comes after the objective (tau <= t). So ``units/O_at_f`` and ``units/K_at_f`` are target
splits and never model inputs (see ``unit_K`` and ``unit_O``).

Conversion (G4, ``conversions``). An objective with a known kind and taker that no unit claims is converted to the one unit with the
latest delta among those with delta < t <= min(delta + CONV_MS, T) (the earlier unit on a tie), or to none. Exposure of a unit = the
length of its conversion window cut at the game end: max(0, min(delta + CONV_MS, T) - delta). So every objective is in exactly one of
three places: attributed (unit >= 0), converted (conv_unit >= 0) or rest (the accounting identity).

방법에서의 위치. 이 모듈은 교전 과정 모형(EPT)이 맞히는 교전의 결과 Y = (K, O)와 교전 뒤의 전환을 S1 정답 표로 만든다(설계서 3.2절·3.3절, S1–S2 계획 Task 4). K는 교전
단위 자신의 킬에서 팀별로 센 킬 수·bounty 합·shutdown bounty 합·도움 수이고, O는 [τ, δ) 안에서 일어난 오브젝트 가운데 가져간 챔피언이 단위의 킬 인원과 겹치는 것을
종류와 가져간 팀의 칸으로 센 것이며(귀속, 결정 G3), 전환은 어느 단위에도 귀속되지 않은 오브젝트를 교전이 끝난 뒤 60초 안에서 δ가 가장 늦은 단위 하나에 붙인 것이다
(결정 G4).

용어. 교전: 같은 경기에서 15초와 2,000 지도 단위 이내로 이어진 킬의 묶음이고, 교전 단위는 S1 교전 표의 한 행이다(설계서 2절). f는 단위의 첫 킬 시각, τ = f − 15 s는
교전 시작, δ = max(마지막 킬 + 1 ms, f + 20 s)는 교전 끝, T는 경기 끝이다. 킬 인원: 단위의 킬마다 킬러·피해자·도움 참가자(1–10)이다. 크레딧: 오브젝트를 가져간
챔피언, 곧 킬러와 도움 참가자(포탑 방패는 킬러만)이다. 오브젝트 칸: 오브젝트 종류(9가지)와 가져간 팀의 쌍마다 하나(18칸)이다. 처형: 킬러가 챔피언이 아닌 킬이다.

식 목록. 식 번호는 (모듈 약어-번호) 꼴이며 식과 기호는 docs/superpowers/notes/equation-index.md의 식 색인과 기호 등록부에 올라 있다. j는 교전 단위, i는 오브젝트이다.
j는 교전 표의 0부터 시작하는 행 번호이며 범위는 0, …, 단위 수 − 1이다. i도 오브젝트 표의 0부터 시작하는 행 번호이다. unit(i)와 conv_unit(i)는 −1 또는 해당 교전 행 번호이다.
    (TGT-1)  단위 j의 K        team ∈ {blue, red}마다 킬러가 team인 챔피언 킬의 수, bounty 합, shutdown 합, 도움 참가자 수의 합
    (TGT-2)  단위 j의 처형 수   exec_{j,team} = 킬러가 챔피언이 아니고 피해자가 team인 킬의 수
    (TGT-3)  귀속(G3)          unit(i) = argmax_{j∈cand_i} (ov_ij, −dist_ij),   cand_i = {j : τ_j ≤ t_i < δ_j, ov_ij ≥ 1}   (cand_i = ∅이면 −1)
    (TGT-4)  단위 j의 O        O_{j,c} = #{i : unit(i) = j, 2·kind_i + taker_i = c}
    (TGT-5)  전환(G4)과 노출    conv_unit(i) = argmax_{j∈win_i} δ_j,   win_i = {j : δ_j < t_i ≤ end_j}
                                e_j = max(0, end_j − δ_j),   end_j = min(δ_j + 60 s, T)
    (TGT-6)  회계 항등식        n_all = n_attr + n_conv + n_rest   (귀속과 전환은 겹치지 않는다)

공통 가정.
    (가) 시각은 ms이고 거리는 지도 단위이다. 귀속 창은 [τ_j, δ_j), 전환 창은 (δ_j, min(δ_j + 60 s, T)]이어서 시각이 δ_j인 오브젝트는 단위 j의 두 창에 모두 들지 않는다(노트 02 C·D절).
    (나) 귀속과 단위 구성은 경기의 모든 킬을 본 뒤 정한 사후 값이다. 그래서 K_at_f와 O_at_f는 보고용 목표 쪼개기이고 모형의 입력이 아니다(결정 S1-1).
    (다) 위치나 크레딧이 없는 오브젝트는 귀속하지 않고, 종류나 가져간 팀을 모르는 오브젝트는 귀속도 전환도 하지 않아 나머지로 센다(설계서 11절 4·17).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 3.2 (K and O: credit overlap, then nearest kill; executions pay no team),
#      section 3.3 (conversion window, unique assignment D4), section 8.1 (tower tiers, D7) and section 11, items 3, 4 and 6 (nine O kinds,
#      objectives without a position, every unit is a candidate).
# Ref: D:\LOL_Data\runs\20261019_link_rule_fit_1\objective_participation_probe.py and aftermath_probe.py (the prototype: credited champions
#      against the people of the unit, key (overlap, -distance) with the earlier unit on a tie, taker from killerTeamId / the other team
#      of teamId, conversion window (delta, delta + 60 s] cut at the game end).
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from c_ept import config as C
from c_ept import units
from c_ept.reader import MatchRecord
# The id rule and the participant helpers of the kill table, shared on purpose: kills and objectives parse their ids alike.
from c_ept.units import _ALL_PIDS, _POPCOUNT, _assist_mask, _event_id, _is_int, _is_pid

K_FIELDS = ('kills_b', 'kills_r', 'bounty_b', 'bounty_r', 'shutdown_b', 'shutdown_r', 'assists_b', 'assists_r')
OBJECTIVE_TYPES = ('ELITE_MONSTER_KILL', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED')
_ELITE, _BUILDING, _PLATE = OBJECTIVE_TYPES
N_CELLS = 2 * len(C.O_KINDS)                 # O cell = 2 * kind + taker
if not set(OBJECTIVE_TYPES) <= set(C.EVENT_TYPES):       # an explicit raise, not an assert: asserts vanish under python -O
    raise RuntimeError(f'config.EVENT_TYPES {C.EVENT_TYPES} lacks one of the objective types {OBJECTIVE_TYPES}')

_KIND = {name: i for i, name in enumerate(C.O_KINDS)}
_MONSTER_KIND = {raw: _KIND[name] for raw, name in (('DRAGON', 'dragon'), ('ELDER_DRAGON', 'elder'), ('HORDE', 'grubs'),
                                                    ('RIFTHERALD', 'herald'), ('BARON_NASHOR', 'baron'), ('ATAKHAN', 'atakhan'))}
_BUILDING_KIND = {raw: _KIND[name] for raw, name in (('TOWER_BUILDING', 'tower'), ('INHIBITOR_BUILDING', 'inhibitor'))}
_TOWER_TIER = {'OUTER_TURRET': 0, 'INNER_TURRET': 1, 'BASE_TURRET': 2, 'NEXUS_TURRET': 3}


@dataclass(frozen=True, eq=False)
class ObjectiveTable:
    """The objectives of one match, one row each, in event order; see the module text for the columns.

    열 unit은 (TGT-3)이 정하고, 열 kind와 taker는 (TGT-4)와 (TGT-5)의 칸 2·kind + taker를 정한다.
    """
    t: np.ndarray             # int64, ms
    kind: np.ndarray          # int8, index in config.O_KINDS, -1 other
    taker: np.ndarray         # int8, 0 blue, 1 red, -1 unknown
    x: np.ndarray             # float64, NaN without position
    y: np.ndarray             # float64
    has_pos: np.ndarray       # bool
    cred: np.ndarray          # int16, bit mask (bit pid - 1) of the credited champions
    unit: np.ndarray          # int32, index in the UnitTable, -1 for none
    sub: np.ndarray           # int8, tower tier 0-3, -1 otherwise


# ---- K -----------------------------------------------------------------------------------------------------------------------
def _sides(team) -> np.ndarray:
    """(10,) side of each participant (index pid - 1): 0 blue, 1 red. ValueError unless ``team`` holds 10 ids, all blue or red.

    (TGT-1)의 team(참가자)을 만든다.
    """
    team = np.asarray(team)
    if team.shape != (10,):
        raise ValueError(f'team must hold 10 team ids, got shape {team.shape}')
    red = team == units.RED
    if not np.all(red | (team == units.BLUE)):
        raise ValueError(f'team must hold only the team ids {units.BLUE} and {units.RED}')
    return red.astype(np.int64)


def _unit_rows(k: units.KillTable, comp, upto_ms: int | None) -> np.ndarray:
    """The KillTable rows ``comp`` of a unit, only those with t <= upto_ms when that is given (inclusive).

    (TGT-1)의 시각 조건 t(킬) ≤ upto를 단위의 킬 행에 적용한다.
    """
    rows = np.asarray(comp, dtype=np.intp)
    return rows if upto_ms is None else rows[k.t[rows] <= upto_ms]


def unit_K(k: units.KillTable, comp, team, upto_ms: int | None = None) -> np.ndarray:
    """(8,) int64 in the order of ``K_FIELDS``: the kills, the recorded bounty, the recorded shutdown bounty and the assisters of the
    kills of one unit, by the team of the killer. ``comp`` holds the KillTable rows of the unit, ``team`` the (10,) team ids (index
    pid - 1). Only kills whose killer is a champion (1-10) count, so an execution pays nothing, not even its assisters. With
    ``upto_ms`` only the kills with t <= upto_ms count (inclusive). ValueError unless ``team`` holds 10 blue or red ids.

    ``upto_ms`` = the first kill f gives the S1 column ``K_at_f``. It is a HINDSIGHT target split, never a model input: ``comp`` is the
    unit as it stands at the end (single linkage over all kills), so kills at time f (the first kill and any kill at the same
    millisecond) can belong to the unit only because of kills after f. An information-up-to-f rule would count 8 units differently
    (final S1 review, Important 1).

    수식
    ----
    (TGT-1)  kills_{j,team}    = #{킬 ∈ j : team(killer(킬)) = team}
             bounty_{j,team}   = Σ_{킬 ∈ j, team(killer(킬)) = team} bounty(킬),      shutdown_{j,team}도 같은 꼴
             assists_{j,team}  = Σ_{킬 ∈ j, team(killer(킬)) = team} |assisters(킬)|
             킬 ∈ j: 단위 j의 킬 가운데 킬러가 참가자 1–10이고, upto를 주면 시각이 upto 이하(경계 포함)인 것이다.
             K의 칸 번호는 2·항목 + team이다(항목 kills 0, bounty 1, shutdown 2, assists 3; team blue 0, red 1).
             기록 bounty와 shutdown bounty는 비음수 정수라는 입력 가정이며, 이 함수는 그 조건을 검사하지 않는다.

      기호                               | 뜻                                           | 단위 | 범위          | 코드 변수
      j                                  | 교전 단위 번호(S1 교전 표의 0부터 시작하는 행) | —    | 0, …, 단위 수 − 1 | comp (단위 j의 킬 행)
      team                               | 킬러의 팀(참가자 1–5는 blue, 6–10은 red)     | —    | blue 0, red 1 | team 인자, side
      killer(킬), victim(킬)             | 킬의 킬러, 피해자 번호(킬러 0은 챔피언 아님) | —    | 0–10; 1–10    | k.killer, k.victim
      bounty(킬), shutdown(킬)           | 킬 하나의 기록 bounty, 기록 shutdown bounty  | 골드 | ≥ 0           | k.bounty, k.shutdown
      assisters(킬)                      | 킬의 도움 참가자(1–10)의 집합                | —    | ⊂ {1, …, 10}  | k.amask
      kills_{j,team}, assists_{j,team}   | 킬 수, 도움 참가자 수의 합                   | 개   | 0, 1, …       | out[team], out[6 + team]
      bounty_{j,team}, shutdown_{j,team} | bounty 합, shutdown bounty 합                | 골드 | ≥ 0           | out[2 + team], out[4 + team]
      upto                               | 시각 상한(없으면 제한 없음)                  | ms   | ≥ 0           | upto_ms

    해석
    ----
    식 (TGT-1)은 단위 j의 킬만 킬러의 팀별로 더한 K이고, 같은 창에 있는 다른 단위의 킬은 들어가지 않는다(설계서 3.2절). 킬러가 챔피언인 킬만 세므로 처형은 어느 팀의
    킬·골드·도움에도 들지 않고 (TGT-2)로 따로 센다(설계서 3.2절). 도움은 도움 참가자의 수이며, 도움 풀을 포함한 킬 골드는 K가 아니라 킬마다 bounty + shutdown + 도움 풀로
    정해지고 도움 풀은 (PAY-3)이 계산한다. 도움 참가자가 킬러와 같은 팀인지는 이 식이 확인하지 않고 킬러의 팀 칸에 더한다. upto를 주면 그 시각까지의 킬만 세는 쪼개기이다(모듈
    머리의 공통 가정 (나)).

    근거
    ----
    설계서 3.2절(K: 단위 자신의 킬, 처형은 지급 없음), 결정 G3(K의 정의), S1–S2 계획 Task 4(unit_K), 결정 S1-1(사후 쪼개기).
    """
    side = _sides(team)
    # 식 (TGT-1)의 킬 집합 "킬 ∈ j": 시각 조건을 적용하고 킬러가 챔피언인 킬만 남긴다. 처형은 어느 팀에도 지급하지 않으므로 여기서 빠진다.
    rows = _unit_rows(k, comp, upto_ms)
    killer = k.killer[rows].astype(np.int64)
    scored = _is_pid(killer)
    rows, killer = rows[scored], killer[scored]
    # 식 (TGT-1)의 합 항: 킬마다 킬러의 팀과 bounty, shutdown, 도움 참가자 수를 구한다. 도움은 킬러의 팀 칸에 더하므로 킬러의 팀만 쓴다.
    mine = side[killer - 1]
    bounty, shutdown = k.bounty[rows].astype(np.int64), k.shutdown[rows].astype(np.int64)
    assists = _POPCOUNT[k.amask[rows].astype(np.int64) & _ALL_PIDS].astype(np.int64)      # participants 1-10 only
    # 식 (TGT-1)의 합: team마다 더하여 칸 2·항목 + team에 쓴다.
    out = np.zeros(len(K_FIELDS), dtype=np.int64)
    for s in (0, 1):
        m = mine == s
        out[s], out[2 + s], out[4 + s], out[6 + s] = m.sum(), bounty[m].sum(), shutdown[m].sum(), assists[m].sum()
    return out


def unit_exec(k: units.KillTable, comp, team) -> np.ndarray:
    """(2,) int64: the executions of one unit (the killer is no champion) by the team of the victim, [blue victims, red victims].

    수식
    ----
    (TGT-2)  exec_{j,team} = #{킬 ∈ j : killer(킬) ∉ {1, …, 10}, victim(킬) ∈ {1, …, 10}, team(victim(킬)) = team}
             킬 ∈ j는 단위 j의 모든 킬이다(시각 상한이 없다). 기호는 (TGT-1)과 같고 새 기호는 아래 한 줄이다.

      기호          | 뜻                                      | 단위 | 범위    | 코드 변수
      exec_{j,team} | 단위 j의 처형 가운데 피해자가 team인 수 | 개   | 0, 1, … | 반환값 [blue 피해자, red 피해자]

    해석
    ----
    킬러가 챔피언이 아닌 킬(처형)은 어느 팀의 킬·골드·도움에도 들지 않으므로((TGT-1)) 피해자의 팀별 횟수만 따로 센다(설계서 3.2절).

    근거
    ----
    설계서 3.2절(처형은 어느 팀에도 지급하지 않음), S1–S2 계획 Task 4(unit_exec).
    """
    side = _sides(team)
    rows = np.asarray(comp, dtype=np.intp)
    killer, victim = k.killer[rows].astype(np.int64), k.victim[rows].astype(np.int64)
    lost = side[victim[~_is_pid(killer) & _is_pid(victim)] - 1]
    return np.array([(lost == 0).sum(), (lost == 1).sum()], dtype=np.int64)


# ---- objective table ---------------------------------------------------------------------------------------------------------
def _text(e: dict, name: str) -> str | None:
    """``e[name]`` when it is a string, else None (so that it can be looked up in a dict whatever the file holds)."""
    v = e.get(name)
    return v if isinstance(v, str) else None


def objective_kind(e: dict) -> int:
    """Index in ``config.O_KINDS`` of a raw event, -1 when it is none of the nine kinds.

    ELITE_MONSTER_KILL: monsterType ELDER_DRAGON, or DRAGON with monsterSubType ELDER_DRAGON, is the elder dragon; otherwise DRAGON,
    HORDE (grubs), RIFTHERALD, BARON_NASHOR and ATAKHAN. BUILDING_KILL: buildingType TOWER_BUILDING or INHIBITOR_BUILDING.
    TURRET_PLATE_DESTROYED is a plate.

    원시 사건을 오브젝트 칸 (TGT-4)의 종류 번호 아홉 가지로 옮기는 단계이다. 장로 드래곤은 가치와 규칙이 드래곤과 달라 따로 두고(설계서 11절 3), 아홉 가지가 아닌 사건은 −1로 두어
    귀속과 전환에서 나머지로 센다(설계서 11절 17).
    """
    typ = e.get('type')
    if typ == _ELITE:
        monster = _text(e, 'monsterType')
        if monster == 'DRAGON' and _text(e, 'monsterSubType') == 'ELDER_DRAGON':
            return _KIND['elder']
        return _MONSTER_KIND.get(monster, -1)
    if typ == _BUILDING:
        return _BUILDING_KIND.get(_text(e, 'buildingType'), -1)
    if typ == _PLATE:
        return _KIND['plate']
    return -1


def _taker(e: dict) -> int:
    """0 when blue got the objective, 1 when red did, -1 when the file does not say: the team id is absent or is not the integer 100
    or 200 (a float or a bool that equals one of them is not a team id)."""
    elite = e.get('type') == _ELITE
    team = e.get('killerTeamId' if elite else 'teamId')       # a building or plate: the owner who lost it, so the other team took it
    if not _is_int(team) or team not in (units.BLUE, units.RED):
        return -1
    return int(team == units.RED) if elite else int(team == units.BLUE)


def _tower_tier(e: dict) -> int:
    """0-3 for the towerType of a tower (a BUILDING_KILL with buildingType TOWER_BUILDING), -1 for every other event, building or value."""
    return _TOWER_TIER.get(_text(e, 'towerType'), -1) if objective_kind(e) == _KIND['tower'] else -1


def _attribute(t, valid, x, y, cred, u: units.UnitTable, k: units.KillTable) -> np.ndarray:
    """(n,) int32 attribution of the objectives: the unit index of each one marked ``valid`` that a unit claims, else -1 (see the
    module text for the rule). The units are visited in table order and only a strictly larger key (overlap, -distance) replaces the
    choice, so a tie keeps the earlier unit.

    (TGT-3)의 argmax를 계산한다.
    """
    unit = np.full(len(t), -1, dtype=np.int32)
    todo = np.flatnonzero(valid & (cred != 0))                    # without a champion in the credit nothing can overlap
    if todo.size == 0 or len(u.f) == 0:
        return unit
    # 식 (TGT-3)의 people_j: 단위의 킬마다 킬러·피해자·도움의 비트 마스크를 OR로 모은다. spots는 dist_ij에 쓸 위치 있는 킬의 좌표이다.
    credit = units.kill_people_mask(k)
    people = [int(np.bitwise_or.reduce(credit[g])) for g in u.comp]
    spots = []                                                    # the kills of each unit that have a position
    for g in u.comp:
        kx, ky = k.x[g], k.y[g]
        seen = np.isfinite(kx) & np.isfinite(ky)
        spots.append((kx[seen], ky[seen]))
    # 식 (TGT-3)의 후보 cand_i와 사전식 argmax: 겹침이 클수록, 겹침이 같으면 거리가 가까울수록 이기고, 더 큰 키만 바꾸므로 같으면 앞 단위가 남는다.
    for i in todo:
        best, best_overlap, best_dist = -1, 0, np.inf
        for j in np.flatnonzero((u.tau <= t[i]) & (t[i] < u.delta)):
            overlap = int(_POPCOUNT[int(cred[i]) & people[j]])
            if overlap == 0:
                continue
            kx, ky = spots[j]
            dist = float(np.hypot(kx - x[i], ky - y[i]).min()) if kx.size else np.inf
            if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
                best, best_overlap, best_dist = int(j), overlap, dist
        unit[i] = best
    return unit


def objective_table(rec: MatchRecord, u: units.UnitTable, k: units.KillTable) -> ObjectiveTable:
    """The objectives of a match (ELITE_MONSTER_KILL, BUILDING_KILL, TURRET_PLATE_DESTROYED at any time) with their attribution to the
    units ``u`` built from the kill table ``k`` of the same match (module text for the rules).

    Ids follow the rule of ``kill_table`` (the same functions, with the event named in the message): killerId is absent (no champion)
    or an integer 0-10 (not a bool); assistingParticipantIds is absent, null or a list of integers, those outside 1-10 are ignored. A
    plate is credited to its killer only, but its assists must be well formed as well. Anything else raises MatchFormatError with the
    reason and the patch in the message, never the key. A taker that the file does not say is -1, not an error.

    수식
    ----
    (TGT-3)  cand_i = {j : τ_j ≤ t_i < δ_j, ov_ij ≥ 1},   ov_ij = |cred_i ∩ people_j|,   unit(i) = argmax_{j∈cand_i} (ov_ij, −dist_ij)
             argmax는 사전식 비교이고 같으면 j가 가장 작은(앞) 단위이며, cand_i = ∅이면 unit(i) = −1이다.
             cred_i = {killer_i} ∪ assisters_i (킬러가 챔피언이 아니면 killer_i 없음; 포탑 방패는 {killer_i}만),
             people_j = 단위 j의 킬마다 킬러·피해자·도움 참가자(1–10)의 합집합(두 팀 모두이며 겹침은 팀을 가리지 않는다),
             dist_ij = min_{킬 ∈ j, 위치 있음} ‖pos_i − pos_킬‖ (위치 있는 킬이 없으면 +∞).
             대상은 위치, 종류, 가져간 팀을 알고 cred_i ≠ ∅인 오브젝트 i이다. 후보 j는 범위 밖 단위를 포함한 모든 단위이다.

      기호             | 뜻                                                     | 단위          | 범위              | 코드 변수
      i                | 오브젝트 번호(오브젝트 표의 0부터 시작하는 행)         | —             | 0, …, 오브젝트 수 − 1 | 행 번호 i
      t_i, pos_i       | 오브젝트 i의 시각, 위치                                | ms, 지도 단위 | ≥ 0; 지도 안      | t, (x, y)
      τ_j, δ_j         | 단위 j의 교전 시작, 끝                                 | ms            | τ_j < δ_j         | u.tau, u.delta
      cred_i, people_j | 가져간 챔피언(킬러·도움), 단위 j의 킬 인원             | —             | ⊂ {1, …, 10}      | cred, people (비트 마스크)
      ov_ij, dist_ij   | 겹치는 참가자 수, 오브젝트에서 단위 킬까지의 최소 거리 | 명, 지도 단위 | 0–10; ≥ 0 또는 +∞ | overlap, dist
      unit(i)          | 귀속된 단위(−1은 귀속 없음)                            | —             | −1 또는 0, …, 단위 수 − 1 | unit

    해석
    ----
    식 (TGT-3)은 오브젝트를 시각이 [τ_j, δ_j)에 드는 단위 가운데 가져간 챔피언과 킬 인원이 가장 많이 겹치는 단위에 붙이고, 겹침이 같으면 단위의 킬이 가장 가까운 쪽에,
    그것도 같으면 앞 단위에 붙이는 규칙이다(결정 G3). 겹침을 거리보다 먼저 보는 것은 가져간 챔피언이 그 교전의 킬 인원이었다는 것이 오브젝트와 교전을 잇는 증거이고, 설계서 3.2절에 기록된 FIT 탐색에서 거리 골짜기를
    찾지 못했으므로 참여자 겹침을 우선하고 거리는 동률을 가르는 데 쓴다(설계서 3.2절 근거). 범위 밖 단위까지 후보로 삼는 것은 경기 시작 후 2분 이전의 교전의 오브젝트가 뒤 교전으로 잘못 가는 것을 막기 위해서이다(설계서 11절 6). 위치나 챔피언 기록이 없는 오브젝트와 후보 단위가
    없는 오브젝트는 귀속하지 않고(unit = −1), 종류와 가져간 팀을 알면 전환의 후보가 된다(설계서 3.2절·3.3절, 11절 4). 종류나 가져간 팀을 모르는 오브젝트는 귀속도 전환도 하지
    않는다(설계서 11절 17).

    근거
    ----
    결정 G3(겹침, 가까운 쪽, 챔피언 기록 없으면 안 붙임), 결정 S1-3(팀을 가리지 않는 겹침), 설계서 3.2절·3.3절·11절의 항목 4·6·17, S1–S2 계획 Task 4, 검산 노트 math-control-02 C절(손 계산).
    """
    rows = [e for e in rec.events if e.get('type') in OBJECTIVE_TYPES]
    n = len(rows)
    t, kind, taker, sub = np.empty(n, np.int64), np.empty(n, np.int8), np.empty(n, np.int8), np.empty(n, np.int8)
    x, y, cred = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, np.int16)
    for r, e in enumerate(rows):
        label = f'{e["type"]} event'
        t[r], kind[r], taker[r], sub[r] = e['timestamp'], objective_kind(e), _taker(e), _tower_tier(e)
        x[r], y[r] = units.event_xy(e)
        killer = _event_id(rec, e, 'killerId', 0, 0, label)         # absent: no champion
        assisted = _assist_mask(rec, e, label)                       # parsed for a plate as well (a malformed value raises), not credited
        cred[r] = (1 << (killer - 1) if killer else 0) | (0 if e['type'] == _PLATE else assisted)
    has_pos = ~np.isnan(x)
    unit = _attribute(t, has_pos & (kind >= 0) & (taker >= 0), x, y, cred, u, k)
    return ObjectiveTable(t=t, kind=kind, taker=taker, x=x, y=y, has_pos=has_pos, cred=cred, unit=unit, sub=sub)


def _counted(obj: ObjectiveTable) -> np.ndarray:
    """Which objectives have a cell: a kind of the table and a taker that is blue or red.

    (TGT-4)와 (TGT-5)가 세는 오브젝트(칸이 있는 것)를 고른다.
    """
    return (obj.kind >= 0) & (obj.kind < len(C.O_KINDS)) & (obj.taker >= 0) & (obj.taker <= 1)


def unit_O(obj: ObjectiveTable, unit_index: int, upto_ms: int | None = None) -> np.ndarray:
    """(18,) int64: the objectives attributed to unit ``unit_index``, counted in the cell 2 * kind + taker. With ``upto_ms`` only
    those with t <= upto_ms (inclusive). Objectives with an unknown kind or taker have no cell and are not counted.

    ``upto_ms`` = the first kill f gives the S1 column ``O_at_f``. It is a HINDSIGHT target split, never a model input: the attribution
    (``obj.unit``) uses all kills of a unit, and units that start later, so an objective before f can be attributed to the unit because of
    what happened after f. The final S1 review measured 12,606 objectives in 11,528 units that differ under an information-up-to-f rule
    (Important 1).

    수식
    ----
    (TGT-4)  O_{j,c} = #{i : unit(i) = j, kind_i ∈ {0, …, 8}, taker_i ∈ {0, 1}, 2·kind_i + taker_i = c, t_i ≤ upto}        (c = 0, 1, …, 17)
             j는 unit_index이다. upto를 주지 않으면 시각 조건이 없다. 종류 번호는 dragon 0, elder 1, grubs 2, herald 3, baron 4, atakhan 5,
             tower 6, inhibitor 7, plate 8이다. 나머지 기호는 (TGT-3)과 같다.

      기호    | 뜻                                        | 단위 | 범위              | 코드 변수
      O_{j,c} | 단위 j에 귀속된 오브젝트 가운데 칸 c의 수 | 개   | 0, 1, …           | 반환값 (18,)
      c       | 오브젝트 칸 번호 2·kind + taker           | —    | 0, …, 17          | N_CELLS가 칸의 수
      kind_i  | 오브젝트 i의 종류 번호                    | —    | −1(기타), 0, …, 8 | obj.kind
      taker_i | 오브젝트 i를 가져간 팀(blue 0, red 1)     | —    | −1(모름), 0, 1    | obj.taker
      upto    | 시각 상한(없으면 제한 없음)               | ms   | ≥ 0               | upto_ms

    해석
    ----
    식 (TGT-4)는 단위 j에 귀속된 오브젝트를 종류와 가져간 팀의 칸으로 센 O이고, 모형이 맞히는 결과 Y = (K, O)의 오브젝트 부분이다(설계서 3.2절). 포탑은 단계와 관계없이 한 칸에
    세고 단계는 흐름 가격을 위해 오브젝트 표에 따로 보관한다(설계서 11절 16). 종류나 가져간 팀을 모르는 오브젝트는 칸이 없어 세지 않는다(설계서 11절 17). upto를 주면 그 시각까지의
    오브젝트만 세는 쪼개기이다(모듈 머리의 공통 가정 (나)).

    근거
    ----
    설계서 3.2절(O: 종류 × 팀)과 11절의 항목 3·16·17, 결정 G3, S1–S2 계획 Task 4(unit_O), 결정 S1-1(사후 쪼개기).
    """
    m = (obj.unit == unit_index) & _counted(obj)
    if upto_ms is not None:
        m &= obj.t <= upto_ms
    return np.bincount(2 * obj.kind[m].astype(np.int64) + obj.taker[m], minlength=N_CELLS).astype(np.int64)


def conversions(obj: ObjectiveTable, u: units.UnitTable, T_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(conv, exposure_ms, conv_unit)`` of a match with the game end ``T_ms``.

    ``conv`` (n_units, 18) int64 counts, per unit and cell 2 * kind + taker, the objectives converted to it. ``exposure_ms`` (n_units,)
    int64 is max(0, min(delta + CONV_MS, T_ms) - delta): zero for a unit that ends at or after the game end. ``conv_unit`` (n_obj,)
    int32 gives, per objective, the unit it is converted to, -1 for none. Only an objective that no unit claims (``unit`` -1) and has
    a known kind and taker is converted, to the unit with the latest delta among those with delta < t <= min(delta + CONV_MS, T_ms)
    (the earlier unit on a tie).

    수식
    ----
    (TGT-5)  end_j = min(δ_j + 60 s, T),   e_j = max(0, end_j − δ_j),   win_i = {j : δ_j < t_i ≤ end_j}
             conv_unit(i) = argmax_{j∈win_i} δ_j (같으면 j가 작은 쪽; win_i = ∅이면 −1),   conv_{j,c} = #{i : conv_unit(i) = j, 칸 c}
             대상은 unit(i) = −1이고 종류와 가져간 팀을 아는 오브젝트 i이다. 후보 j는 범위 밖 단위를 포함한 모든 단위이다.
             칸 c는 (TGT-4)와 같은 2·kind_i + taker_i이고, δ_j, t_i, unit(i), kind_i, taker_i의 뜻은 (TGT-3), (TGT-4)와 같다.
    (TGT-6)  n_all = n_attr + n_conv + n_rest,   n_attr = #{i : unit(i) ≥ 0},   n_conv = #{i : conv_unit(i) ≥ 0}
             n_rest = #{i : unit(i) < 0, conv_unit(i) < 0}. n_all은 오브젝트 표의 행 수이고, conv_unit(i) ≥ 0이면 unit(i) = −1이므로 귀속과 전환은 겹치지 않는다.

      기호                          | 뜻                                        | 단위 | 범위              | 코드 변수
      end_j, e_j                    | 단위 j의 전환 창의 끝, 노출(창의 길이)    | ms   | ≥ 0               | end, exposure_ms
      CONV                          | 전환 창의 길이(식의 60 s)                 | ms   | 상수(60,000)      | C.CONV_MS
      T                             | 경기 끝                                   | ms   | > 0               | T_ms
      win_i                         | 오브젝트 i의 전환 후보 단위의 집합        | —    | ⊂ 단위 번호       | window
      conv_unit(i)                  | 오브젝트 i가 전환된 단위(−1은 없음)       | —    | −1 또는 0, …, 단위 수 − 1 | conv_unit
      conv_{j,c}                    | 단위 j에 전환된 오브젝트 가운데 칸 c의 수 | 개   | 0, 1, …           | conv (n_units, 18)
      n_all, n_attr, n_conv, n_rest | 오브젝트 전체, 귀속, 전환, 나머지의 수    | 개   | 0, 1, …           | 식의 개수(코드에서 세지 않는다)

    해석
    ----
    식 (TGT-5)는 귀속되지 않은 오브젝트를 교전이 끝난 뒤 60초 창 (δ_j, end_j] 안에서 δ가 가장 늦은 단위 하나에 붙이고, 노출 e_j는 그 창을 경기 끝 T에서 자른 길이이다(결정 G4,
    설계서 3.3절 D4). δ_j가 T 이상이면 e_j = 0이고 창이 비어 그 단위에는 전환이 붙지 않는다(S1–S2 계획 검토 초점 3). 오브젝트를 단위 하나에만 붙이므로 한 행의 교전 가치를
    더해도 같은 오브젝트를 두 번 세지 않는다(설계서 3.4절 D4). 식 (TGT-6)의 귀속, 전환, 나머지는 서로 겹치지 않고 합이 오브젝트 전체이다(설계서 3.3절 D4). (경기, 묶음,
    가져간 팀) 칸별 개수 검사는 정답 단계의 회계 검사가 한다(판정 R11).

    근거
    ----
    결정 G4(전환 창, 유일 귀속), 설계서 3.3절 D4와 3.4절 D4(회계 항등식, 중복 없음), S1–S2 계획 검토 초점 3, 판정 R11(개수 수준 항등식), 검산 노트 math-control-02 D절(손 계산).
    """
    # 식 (TGT-5)의 end_j와 노출 e_j: 전환 창을 경기 끝 T에서 자르고 길이가 0 아래로 내려가지 않게 한다.
    end = np.minimum(u.delta + C.CONV_MS, T_ms)
    exposure = np.maximum(end - u.delta, 0).astype(np.int64)
    conv = np.zeros((len(u.f), N_CELLS), dtype=np.int64)
    conv_unit = np.full(len(obj.t), -1, dtype=np.int32)
    # 식 (TGT-5)의 후보 win_i와 argmax: 귀속되지 않은 오브젝트만 보고, δ가 가장 늦은 단위 하나에 붙인다.
    for i in np.flatnonzero((obj.unit < 0) & _counted(obj)):
        window = np.flatnonzero((u.delta < obj.t[i]) & (obj.t[i] <= end))
        if window.size:
            j = int(window[np.argmax(u.delta[window])])       # argmax takes the first of equal deltas: the earlier unit
            conv_unit[i] = j
            conv[j, 2 * int(obj.kind[i]) + int(obj.taker[i])] += 1
    return conv, exposure, conv_unit
