"""C-EPT 교전 타깃(Target) 및 결과 라벨 생성 모듈 (c_ept.domain.targets).

교전 단위(Engagement Unit)의 결과 라벨을 계산합니다:
  1. K 타깃: 킬 성과 (팀별 킬 수, 현상금 골드, 제압 골드, 어시스트 수, 처형 수)
  2. O 타깃: 교전 시간 구간 [tau, delta) 내에 획득된 오브젝트 귀속 (9종 × 2개 팀 = 18개 셀)
  3. 전환(Conversion): 교전 종료 후 60초 전환 윈도우 (delta, delta + 60초] 내에 획득된 전과 오브젝트
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from c_ept import config as C
from c_ept.domain import units
from c_ept.data.reader import MatchRecord
# units 모듈에서 참가자 ID 판정 및 비트마스크 연산 헬퍼 재사용
from c_ept.domain.units import _ALL_PIDS, _POPCOUNT, _assist_mask, _event_id, _is_int, _is_pid

# -----------------------------------------------------------------------------
# 상수 및 오브젝트 매핑 정의
# -----------------------------------------------------------------------------
# K 타깃의 8개 열 이름: 블루/레드 킬수, 현상금, 제압골드, 어시스트
K_FIELDS = ('kills_b', 'kills_r', 'bounty_b', 'bounty_r', 'shutdown_b', 'shutdown_r', 'assists_b', 'assists_r')

OBJECTIVE_TYPES = ('ELITE_MONSTER_KILL', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED')
_ELITE, _BUILDING, _PLATE = OBJECTIVE_TYPES

# 오브젝트 분류 셀 총 개수: 9종 × 2개 팀(0: Blue, 1: Red) = 18개 셀 (cell = 2 * kind + taker)
N_CELLS = 2 * len(C.O_KINDS)
if not set(OBJECTIVE_TYPES) <= set(C.EVENT_TYPES):
    raise RuntimeError(f'config.EVENT_TYPES {C.EVENT_TYPES} lacks one of the objective types {OBJECTIVE_TYPES}')

_KIND = {name: i for i, name in enumerate(C.O_KINDS)}
_MONSTER_KIND = {
    raw: _KIND[name] for raw, name in (
        ('DRAGON', 'dragon'), ('ELDER_DRAGON', 'elder'), ('HORDE', 'grubs'),
        ('RIFTHERALD', 'herald'), ('BARON_NASHOR', 'baron'), ('ATAKHAN', 'atakhan')
    )
}
_BUILDING_KIND = {
    raw: _KIND[name] for raw, name in (
        ('TOWER_BUILDING', 'tower'), ('INHIBITOR_BUILDING', 'inhibitor')
    )
}
_TOWER_TIER = {'OUTER_TURRET': 0, 'INNER_TURRET': 1, 'BASE_TURRET': 2, 'NEXUS_TURRET': 3}


# -----------------------------------------------------------------------------
# 오브젝트 테이블 데이터 구조체
# -----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ObjectiveTable:
    """단일 매치에서 발생한 모든 주요 오브젝트 파괴/처치 사건을 담는 테이블:
    - kind: 오브젝트 종류 인덱스 (0~8: config.O_KINDS, -1: 기타)
    - taker: 획득한 팀 (0: Blue, 1: Red, -1: 미상)
    - cred: 처치에 기여한 챔피언들의 10비트 마스크 (킬러 및 어시스트)
    - unit: 이 오브젝트가 직접 귀속된 교전 단위 인덱스 (-1: 교전 중 귀속 없음)
    - sub: 타워 티어 (0: 외곽, 1: 내부, 2: 억제기 타워, 3: 넥서스 타워, -1: 기타)
    """
    t: np.ndarray             # 발생 시각 (밀리초, ms, int64)
    kind: np.ndarray          # 오브젝트 종류 (0~8, int8)
    taker: np.ndarray         # 획득 팀 (0: Blue, 1: Red, int8)
    x: np.ndarray             # X 좌표 (float64)
    y: np.ndarray             # Y 좌표 (float64)
    has_pos: np.ndarray       # 위치 유효 여부 (bool)
    cred: np.ndarray          # 기여 챔피언 비트마스크 (int16)
    unit: np.ndarray          # 귀속 교전 번호 (int32, -1이면 미귀속)
    sub: np.ndarray           # 타워 티어 (0~3, int8)


# -----------------------------------------------------------------------------
# 1. K 타깃 계산: 킬 수, 골드, 어시스트, 처형
# -----------------------------------------------------------------------------
def _sides(team) -> np.ndarray:
    """10명 참가자의 소속 진영을 0(Blue), 1(Red)로 인코딩한 배열을 반환합니다."""
    team = np.asarray(team)
    if team.shape != (10,):
        raise ValueError(f'team must hold 10 team ids, got shape {team.shape}')
    red = team == units.RED
    if not np.all(red | (team == units.BLUE)):
        raise ValueError(f'team must hold only the team ids {units.BLUE} and {units.RED}')
    return red.astype(np.int64)


def _unit_rows(k: units.KillTable, comp, upto_ms: int | None) -> np.ndarray:
    """교전에 속한 킬 행 인덱스 중 지정된 시각(upto_ms) 이하인 행들만 필터링합니다."""
    rows = np.asarray(comp, dtype=np.intp)
    return rows if upto_ms is None else rows[k.t[rows] <= upto_ms]


def unit_K(k: units.KillTable, comp, team, upto_ms: int | None = None) -> np.ndarray:
    """특정 교전 단위에서 발생한 킬 성과(K)를 (8,) 크기의 정수 배열로 계산합니다.

    - 열 구성: [블루킬, 레드킬, 블루현상금, 레드현상금, 블루제압골드, 레드제압골드, 블루어시, 레더어시]
    - 챔피언이 처치한 유효 킬만 팀 성과로 인정되며, 처형은 골드/킬 집계에서 제외됩니다.
    - upto_ms를 전달하면 해당 시각(예: 첫 킬 시점 f)까지의 부분 성과를 계산할 수 있습니다.
    """
    side = _sides(team)
    rows = _unit_rows(k, comp, upto_ms)
    killer = k.killer[rows].astype(np.int64)
    # 킬러가 실제 챔피언(1~10)인 유효 킬만 필터링 (처형 제외)
    scored = _is_pid(killer)
    rows, killer = rows[scored], killer[scored]

    # 킬러의 소속 팀, 현상금, 제압 골드, 어시스트 수 추출
    mine = side[killer - 1]
    bounty, shutdown = k.bounty[rows].astype(np.int64), k.shutdown[rows].astype(np.int64)
    assists = _POPCOUNT[k.amask[rows].astype(np.int64) & _ALL_PIDS].astype(np.int64)

    # 블루팀(0), 레드팀(1)별로 합산하여 8개 칸에 기록
    out = np.zeros(len(K_FIELDS), dtype=np.int64)
    for s in (0, 1):
        m = mine == s
        out[s], out[2 + s], out[4 + s], out[6 + s] = m.sum(), bounty[m].sum(), shutdown[m].sum(), assists[m].sum()
    return out


def unit_exec(k: units.KillTable, comp, team) -> np.ndarray:
    """교전 단위 내에서 발생한 처형(미니언/타워/몬스터에 의한 사망) 횟수를 피해자 팀별로 계산합니다:
    반환값: [블루팀 피해자 수, 레드팀 피해자 수]
    """
    side = _sides(team)
    rows = np.asarray(comp, dtype=np.intp)
    killer, victim = k.killer[rows].astype(np.int64), k.victim[rows].astype(np.int64)
    # 킬러가 챔피언이 아니고 피해자는 챔피언인 경우(처형) 추출
    lost = side[victim[~_is_pid(killer) & _is_pid(victim)] - 1]
    return np.array([(lost == 0).sum(), (lost == 1).sum()], dtype=np.int64)


# -----------------------------------------------------------------------------
# 2. 오브젝트 파싱 및 교전 귀속 (Attribution)
# -----------------------------------------------------------------------------
def _text(e: dict, name: str) -> str | None:
    """딕셔너리에서 문자열 값을 안전하게 조회합니다."""
    v = e.get(name)
    return v if isinstance(v, str) else None


def objective_kind(e: dict) -> int:
    """타임라인 원시 이벤트에서 오브젝트 종류를 0~8 인덱스로 판정합니다:
    드래곤, 장로 드래곤, 공허 유충, 전령, 바론, 아타칸, 타워, 억제기, 포탑 방패 (기타는 -1)
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
    """오브젝트를 획득한 팀을 판정합니다 (0: Blue, 1: Red, -1: 알 수 없음).
    - 몬스터 처치: killerTeamId 기준
    - 타워/억제기/방패 파괴: 파괴된 건물 소유팀(teamId)의 상대 팀이 획득팀이 됨
    """
    elite = e.get('type') == _ELITE
    team = e.get('killerTeamId' if elite else 'teamId')
    if not _is_int(team) or team not in (units.BLUE, units.RED):
        return -1
    return int(team == units.RED) if elite else int(team == units.BLUE)


def _tower_tier(e: dict) -> int:
    """타워 파괴 이벤트인 경우 타워 티어(0: 외곽, 1: 내부, 2: 억제기 타워, 3: 넥서스 타워)를 반환합니다."""
    return _TOWER_TIER.get(_text(e, 'towerType'), -1) if objective_kind(e) == _KIND['tower'] else -1


def _attribute(t, valid, x, y, cred, u: units.UnitTable, k: units.KillTable) -> np.ndarray:
    """각 오브젝트 사건을 가장 적합한 교전 단위(Unit)에 1:1 귀속시킵니다.

    - 후보 조건: 교전의 시간 구간 [tau, delta) 내에 오브젝트가 발생해야 함
    - 우선순위 규칙 (Lexicographical Argmax):
      1. 참여자 겹침(overlap): 오브젝트 기여 챔피언(cred)과 교전 참여자(people) 간 교집합 수 최댓값
      2. 거리 근접도(distance): 오브젝트 좌표와 교전 내 킬 발생 위치들 간의 최소 유클리드 거리
      3. 동률 시 먼저 발생한 교전 우선
    """
    unit = np.full(len(t), -1, dtype=np.int32)
    todo = np.flatnonzero(valid & (cred != 0))
    if todo.size == 0 or len(u.f) == 0:
        return unit

    # 각 교전 단위의 전체 참여자 비트마스크 및 킬 발생 위치 좌표 모음
    credit = units.kill_people_mask(k)
    people = [int(np.bitwise_or.reduce(credit[g])) for g in u.comp]
    spots = []
    for g in u.comp:
        kx, ky = k.x[g], k.y[g]
        seen = np.isfinite(kx) & np.isfinite(ky)
        spots.append((kx[seen], ky[seen]))

    # 각 오브젝트마다 최적의 교전 단위 탐색
    for i in todo:
        best, best_overlap, best_dist = -1, 0, np.inf
        # 시간 창 [tau_j, delta_j) 내에 속하는 교전들만 후보로 검토
        for j in np.flatnonzero((u.tau <= t[i]) & (t[i] < u.delta)):
            overlap = int(_POPCOUNT[int(cred[i]) & people[j]])
            if overlap == 0:
                continue
            kx, ky = spots[j]
            dist = float(np.hypot(kx - x[i], ky - y[i]).min()) if kx.size else np.inf
            # 겹치는 챔피언이 더 많거나, 겹침 수가 같고 거리가 더 가까우면 최적 교전 갱신
            if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
                best, best_overlap, best_dist = int(j), overlap, dist
        unit[i] = best
    return unit


def objective_table(rec: MatchRecord, u: units.UnitTable, k: units.KillTable) -> ObjectiveTable:
    """매치 레코드에서 모든 오브젝트 사건을 추출하고, 교전 단위 테이블(u)에 귀속시켜 ObjectiveTable을 구축합니다."""
    rows = [e for e in rec.events if e.get('type') in OBJECTIVE_TYPES]
    n = len(rows)
    t, kind, taker, sub = np.empty(n, np.int64), np.empty(n, np.int8), np.empty(n, np.int8), np.empty(n, np.int8)
    x, y, cred = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, np.int16)
    for r, e in enumerate(rows):
        label = f'{e["type"]} event'
        t[r], kind[r], taker[r], sub[r] = e['timestamp'], objective_kind(e), _taker(e), _tower_tier(e)
        x[r], y[r] = units.event_xy(e)
        killer = _event_id(rec, e, 'killerId', 0, 0, label)
        assisted = _assist_mask(rec, e, label)
        # 포탑 방패는 킬러만 기여로 인정, 몬스터/건물은 어시스트 포함
        cred[r] = (1 << (killer - 1) if killer else 0) | (0 if e['type'] == _PLATE else assisted)
    has_pos = ~np.isnan(x)
    unit = _attribute(t, has_pos & (kind >= 0) & (taker >= 0), x, y, cred, u, k)
    return ObjectiveTable(t=t, kind=kind, taker=taker, x=x, y=y, has_pos=has_pos, cred=cred, unit=unit, sub=sub)


# -----------------------------------------------------------------------------
# 3. O 타깃 집계 및 사후 전환 (Conversion)
# -----------------------------------------------------------------------------
def _counted(obj: ObjectiveTable) -> np.ndarray:
    """종류(0~8)와 획득 팀(블루/레드)이 명확하여 18개 셀에 집계 가능한 오브젝트 선별"""
    return (obj.kind >= 0) & (obj.kind < len(C.O_KINDS)) & (obj.taker >= 0) & (obj.taker <= 1)


def unit_O(obj: ObjectiveTable, unit_index: int, upto_ms: int | None = None) -> np.ndarray:
    """지정된 교전 단위(unit_index)에 직접 귀속된 오브젝트 수를 (18,) 크기의 배열로 집계합니다:
        cell = 2 * kind + taker (0~17)
    upto_ms를 전달하면 특정 시점(예: 첫 킬 시점 f)까지 발생한 부분 오브젝트만 집계합니다.
    """
    m = (obj.unit == unit_index) & _counted(obj)
    if upto_ms is not None:
        m &= obj.t <= upto_ms
    return np.bincount(2 * obj.kind[m].astype(np.int64) + obj.taker[m], minlength=N_CELLS).astype(np.int64)


def conversions(obj: ObjectiveTable, u: units.UnitTable, T_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """교전 직후 60초 전환 윈도우(delta, min(delta + 60초, T)] 내에서 획득한 오브젝트(전과 전환)를 계산합니다.

    - 대상: 어떤 교전에도 직접 귀속되지 않은 오브젝트 중 종류와 획득팀이 명확한 사건
    - 규칙: 해당 오브젝트 시각 이전에 끝난 교전들 중 가장 최근에 종료된(delta가 가장 늦은) 교전에 1:1 귀속
    - 반환값:
        1. conv: (단위수, 18) 각 교전 단위별로 전환된 18개 셀의 오브젝트 수
        2. exposure_ms: (단위수,) 각 교전의 유효 전환 윈도우 지속 시간 (경기 종료 시점에서 절단됨)
        3. conv_unit: (오브젝트수,) 각 오브젝트가 전환된 교전 인덱스 (-1이면 미전환)
    """
    # 각 교전의 전환 윈도우 종료 시각 계산 (경기 종료 시각 T_ms에서 절단)
    end = np.minimum(u.delta + C.CONV_MS, T_ms)
    exposure = np.maximum(end - u.delta, 0).astype(np.int64)
    conv = np.zeros((len(u.f), N_CELLS), dtype=np.int64)
    conv_unit = np.full(len(obj.t), -1, dtype=np.int32)

    # 직접 귀속되지 않은 오브젝트들을 가장 최근 종료된 교전에 할당
    for i in np.flatnonzero((obj.unit < 0) & _counted(obj)):
        window = np.flatnonzero((u.delta < obj.t[i]) & (obj.t[i] <= end))
        if window.size:
            j = int(window[np.argmax(u.delta[window])])
            conv_unit[i] = j
            conv[j, 2 * int(obj.kind[i]) + int(obj.taker[i])] += 1
    return conv, exposure, conv_unit
