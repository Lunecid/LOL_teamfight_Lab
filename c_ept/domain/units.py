"""
C-EPT 교전 단위(Engagement Unit) 모듈:
경기 내 CHAMPION_KILL 이벤트들을 시공간 거리 기준으로 클러스터링하여 교전(한타) 단위를 정의하고,
교전의 시작/종료 시점, 참여 인원 및 규모, 선제 킬 표지(First-kill mark) 등을 계산합니다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from c_ept import config as C
from c_ept.data.reader import MatchFormatError, MatchRecord, MatchRejected

# -----------------------------------------------------------------------------
# 기본 상수 및 팀/교전 규모(Size) 정의
# -----------------------------------------------------------------------------
# Riot Match-V5 기준 팀 ID (100: 블루팀, 200: 레드팀)
BLUE, RED = 100, 200

# 교전 규모 분류 코드 (T: 대규모 한타, S: 소규모 국지전, ASYM: 비대칭 교전, P: 짤라먹기/단독 킬)
SIZE_T, SIZE_S, SIZE_ASYM, SIZE_P = range(4)
SIZE_NAMES = ('T', 'S', 'ASYM', 'P')

# 팀당 플레이어 슬롯 수 (각 팀 5명, 0~4번 슬롯)
N_SLOTS = 5

# 선제 킬(First-kill mark) 인덱스 레이아웃:
# 0~49: 챔피언 간 첫 킬 (킬러 팀 2개 × 킬러 슬롯 5개 × 피해자 슬롯 5개 = 50종)
# 50: 블루팀 챔피언이 처형당함 (타워/미니언/중립 몬스터 등에 사망)
# 51: 레드팀 챔피언이 처형당함
EXEC_MARK0 = 2 * N_SLOTS * N_SLOTS
if EXEC_MARK0 + 2 != C.N_MARKS:
    raise RuntimeError(f'config.N_MARKS is {C.N_MARKS}, but the mark layout of units.py has {EXEC_MARK0 + 2} marks')

_NAN = float('nan')

# 1~10번 참가자를 나타내는 10비트 마스크 (0b1111111111 = 1023)
_ALL_PIDS = (1 << 10) - 1

# 0~1023 정수의 1비트 개수(참여 인원수)를 빠르게 조회하는 lookup 테이블
_POPCOUNT = np.array([bin(m).count('1') for m in range(1 << 10)], dtype=np.int8)


# -----------------------------------------------------------------------------
# 데이터 구조 정의: 킬 테이블 & 교전 단위 테이블
# -----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class KillTable:
    """
    한 매치에서 발생한 모든 챔피언 킬(CHAMPION_KILL) 이벤트를 시간순으로 저장하는 테이블입니다.
    - killer: 킬러 참가자 ID (1~10: 챔피언, 0: 타워/미니언/몬스터 등에 의한 처형)
    - victim: 피해자 참가자 ID (1~10)
    - amask: 어시스트를 기록한 플레이어들의 10비트 마스크 (bit pid-1)
    - pos_missing: 위치 좌표 (x, y) 누락 여부 플래그
    """
    t: np.ndarray             # 킬 발생 시각 (밀리초, ms int64)
    x: np.ndarray             # 킬 발생 위치 X 좌표 (float64)
    y: np.ndarray             # 킬 발생 위치 Y 좌표 (float64)
    killer: np.ndarray        # 킬러 ID (0~10, int8)
    victim: np.ndarray        # 피해자 ID (1~10, int8)
    amask: np.ndarray         # 어시스트 참가자 10비트 마스크 (int16)
    bounty: np.ndarray        # 획득 현상금 골드 (int32)
    shutdown: np.ndarray      # 제압 골드 (int32)
    pos_missing: np.ndarray   # 위치 누락 여부 불리언 마스크 (bool)


@dataclass(frozen=True, eq=False)
class UnitTable:
    """
    단일 매치에서 추출된 모든 교전(한타) 단위들의 속성을 담는 테이블입니다.
    - comp: 각 교전 단위에 속한 KillTable 행 인덱스들의 배열 리스트
    - f: 교전의 첫 번째 킬 발생 시각 (First kill timestamp, ms)
    - last: 교전의 마지막 킬 발생 시각 (Last kill timestamp, ms)
    - tau: 교전 시작 규칙 앵커 (tau = f - 15초). O(오브젝트)의 귀속 창 [tau, delta)의 시작점을 정의하는 데에만 쓴다 (설계서 §3.1).
           K는 교전 성분 자신의 킬이라 귀속 창이 없으며, 그 킬은 모두 [f, last] ⊂ [tau, delta)에 있다.
           뒤 흐름(전환)의 창 (delta, delta + 60초]는 별개이며 tau는 그 창의 경계가 아니다.
           모델 입력의 snapshot 시점이 아니다: 입력은 프레임 시계의 t0 또는 사건 시계의 f / delta에서 취한다.
    - delta: 교전 종료 및 결과 확정 시점 (delta = max(last + 1ms, f + 20초))
    - nb, nr: 해당 교전에 참여한 블루팀/레드팀 인원수 (킬러, 피해자, 어시스트 포함)
    - size: 교전 규모 분류 코드 (0: T, 1: S, 2: ASYM, 3: P)
    - mark: 교전의 선제 킬 양상 인덱스 (0~51)
    - in_scope: 모델 평가 범위 내에 들어오는 유효 교전인지 여부 (초반 2분 이후 ~ 경기 종료 30초 전)
    - spread: 교전 내 킬들 사이의 최대 거리 (한타의 지리적 분산도, max spatial spread)
    - n_kills: 교전 내 총 킬 수
    """
    comp: list[np.ndarray]    # 교전별 킬 행 인덱스 배열 리스트
    f: np.ndarray             # 첫 킬 시각 (int64, ms)
    last: np.ndarray          # 마지막 킬 시각 (int64, ms)
    tau: np.ndarray           # 교전 직전 기준 시각 tau (int64, ms)
    delta: np.ndarray         # 교전 결과 확정 시각 delta (int64, ms)
    nb: np.ndarray            # 블루팀 고유 참여 인원수 (int8)
    nr: np.ndarray            # 레드팀 고유 참여 인원수 (int8)
    size: np.ndarray          # 교전 규모 분류 코드 (int8; 0: T, 1: S, 2: ASYM, 3: P)
    mark: np.ndarray          # 선제 킬 표지 번호 (int16; 0~51)
    in_scope: np.ndarray      # 분석 대상 포함 여부 불리언 플래그 (bool)
    spread: np.ndarray        # 킬 간 최대 거리 (float32, 맵 유닛)
    n_kills: np.ndarray       # 교전 단위 내 총 킬 수 (int16)


# -----------------------------------------------------------------------------
# 킬 이벤트 파싱 및 테이블 구축 함수
# -----------------------------------------------------------------------------
def event_xy(e: dict) -> tuple[float, float]:
    """
    이벤트 딕셔너리에서 (x, y) 좌표를 안전하게 파싱합니다.
    좌표가 없거나 유효하지 않은 숫자면 (NaN, NaN)을 반환하여 이후 거리 계산 시 제외되도록 합니다.
    """
    pos = e.get('position')
    if isinstance(pos, dict):
        try:
            x, y = float(pos['x']), float(pos['y'])
        except (KeyError, TypeError, ValueError):
            return _NAN, _NAN
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    return _NAN, _NAN


def _is_int(v) -> bool:
    """
    값이 엄격한 정수형(int)인지 검사합니다 (bool 타입 제외).
    """
    return isinstance(v, int) and not isinstance(v, bool)


_KILL_LABEL = 'a CHAMPION_KILL'


def _event_id(rec: MatchRecord, e: dict, name: str, lowest: int, default: int | None, label: str = _KILL_LABEL) -> int:
    """
    이벤트에서 지정된 참가자 ID 필드를 안전하게 검증하여 추출합니다.
    lowest..10 범위의 정수가 아니거나 필수 필드가 없으면 MatchFormatError를 발생시킵니다.
    """
    if name not in e:
        if default is None:
            raise MatchFormatError(rec.key, rec.patch, f'{label} has no {name}')
        return default
    v = e[name]
    if not _is_int(v):
        raise MatchFormatError(rec.key, rec.patch, f'{label} has a {name} of type {type(v).__name__}, not an integer')
    if not lowest <= v <= 10:
        raise MatchFormatError(rec.key, rec.patch, f'{label} has {name} {v}, not in {lowest}-10')
    return v


def _assist_mask(rec: MatchRecord, e: dict, label: str = _KILL_LABEL) -> int:
    """
    어시스트 참가자 목록(assistingParticipantIds)을 10비트 마스크 정수로 변환합니다.
    참가자 pid (1~10)에 대해 (pid - 1)번째 비트를 1로 설정합니다.
    """
    ids = e.get('assistingParticipantIds')
    if ids is None:
        return 0
    if not isinstance(ids, (list, tuple)):
        raise MatchFormatError(rec.key, rec.patch, f'{label} has assistingParticipantIds of type {type(ids).__name__}, not a list')
    mask = 0
    for a in ids:
        if not _is_int(a):
            raise MatchFormatError(rec.key, rec.patch,
                                   f'{label} has an assistingParticipantIds entry of type {type(a).__name__}, not an integer')
        if 1 <= a <= 10:
            mask |= 1 << (a - 1)
    return mask


def kill_table(rec: MatchRecord) -> KillTable:
    """
    매치 레코드에서 CHAMPION_KILL 이벤트들을 추출하여 KillTable 객체를 생성합니다.
    시간순 정렬 상태를 유지하며 킬러, 피해자, 어시스트, 시각, 위치, 현상금을 NumPy 배열로 변환합니다.
    """
    kills = [e for e in rec.events if e.get('type') == 'CHAMPION_KILL']
    n = len(kills)
    t, x, y = np.empty(n, np.int64), np.full(n, _NAN), np.full(n, _NAN)
    killer, victim = np.empty(n, np.int8), np.empty(n, np.int8)
    amask, bounty, shutdown = np.zeros(n, np.int16), np.zeros(n, np.int32), np.zeros(n, np.int32)
    for r, e in enumerate(kills):
        killer[r] = _event_id(rec, e, 'killerId', 0, 0)
        victim[r] = _event_id(rec, e, 'victimId', 1, None)
        amask[r] = _assist_mask(rec, e)
        t[r] = e['timestamp']
        x[r], y[r] = event_xy(e)
        bounty[r], shutdown[r] = int(e.get('bounty') or 0), int(e.get('shutdownBounty') or 0)
    return KillTable(t, x, y, killer, victim, amask, bounty, shutdown, np.isnan(x))


# -----------------------------------------------------------------------------
# 시공간 단일 연결 클러스터링 (교전 단위 묶기)
# -----------------------------------------------------------------------------
def link_components(t, x, y, gap_ms: int = C.G_MS, link_u: float = C.L_U) -> list[np.ndarray]:
    """
    시간과 공간상 가까운 킬들을 하나의 교전 단위(연결 성분)로 묶습니다.

    - 연결 조건 (Single-Linkage):
      두 킬 k, k'의 시간 간격 |t_k - t_k'| <= gap_ms (기본 15초) AND
      두 킬 간의 유클리드 거리 dist(pos_k, pos_k') <= link_u (기본 2000 맵 유닛).
    - 알고리즘:
      시간순 정렬 배열에 대해 two-pointer sweep 기법으로 15초 이내 후보들을 탐색하고,
      Union-Find (경로 압축 포함) 자료구조로 연결 성분을 병합합니다.
    - 위치가 NaN인 킬은 거리 비교가 항상 False가 되므로 단독 교전 단위로 분리됩니다.
    """
    t = np.asarray(t)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.ndim != 1 or x.shape != t.shape or y.shape != t.shape:
        raise ValueError('t, x and y must be 1-d arrays of the same length')
    if t.size > 1 and np.any(t[1:] < t[:-1]):
        raise ValueError('kill times must not decrease')
    n = t.size
    tl, xl, yl = t.tolist(), x.tolist(), y.tolist()

    # 유니온-파인드(Disjoint Set Union) 초기화: 각 킬은 처음에 자기 자신을 부모로 가짐
    parent = list(range(n))

    def find(a: int) -> int:
        """
        경로 압축(Path Compression)을 적용한 루트 노드 탐색
        """
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # 킬 쌍 비교 루프:
    # 시간순으로 정렬되어 있으므로 tl[j] - tl[i] > gap_ms가 되는 순간 이후의 모든 j는 15초를 초과하여 조기 탈출
    reach2 = float(link_u) * float(link_u)
    for i in range(n):
        j = i + 1
        while j < n and tl[j] - tl[i] <= gap_ms:
            dx, dy = xl[i] - xl[j], yl[i] - yl[j]
            # 거리 제곱이 L^2 이하이면 두 킬을 같은 교전 단위로 병합
            if dx * dx + dy * dy <= reach2:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
            j += 1

    # 같은 루트를 공유하는 킬 행 번호들을 그룹으로 취합
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(g, dtype=np.intp) for g in groups.values()]


# -----------------------------------------------------------------------------
# 교전 규모 분류 및 선제 킬 표지(First-Kill Mark) 계산
# -----------------------------------------------------------------------------
def size_class(nb: int, nr: int) -> int:
    """
    양 팀의 교전 참여 인원수(nb: 블루, nr: 레드)에 따라 교전의 규모(Size)를 4단계로 분류합니다.
    - 3 (SIZE_P, Pick): 한쪽 팀이라도 참여자가 2명 미만인 경우 (단독 처치, 짤라먹기)
    - 0 (SIZE_T, Teamfight): 양 팀 모두 4명 이상 참여한 대규모 한타 (4v4, 5v5 등)
    - 1 (SIZE_S, Skirmish): 양 팀 모두 2~3명씩 참여한 대등한 소규모 교전 (2v2, 3v3 등)
    - 2 (SIZE_ASYM, Asymmetric): 한 팀은 2~3명인데 상대 팀은 4명 이상인 인원 불균형 교전
    """
    lo, hi = min(nb, nr), max(nb, nr)
    if lo < 2:
        return SIZE_P
    if lo >= 4:
        return SIZE_T
    return SIZE_S if hi <= 3 else SIZE_ASYM


def first_kill_mark(killer: int, victim: int) -> int:
    """
    교전의 첫 번째 킬(선제 킬)이 어떤 역할/진영 간에 발생했는지를 0~51번 인덱스로 인코딩합니다.

    - 챔피언 간 킬 (killer 1~10, victim 1~10):
      d: 킬러 진영 (0: 블루팀 1~5번, 1: 레드팀 6~10번)
      i: 킬러 슬롯 번호 (0~4)
      j: 피해자 슬롯 번호 (0~4)
      mark = (d * 5 + i) * 5 + j (0 ~ 49)
    - 처형 (killer가 챔피언이 아님: 타워/미니언/몬스터 등에 의한 사망):
      50: 블루팀 참가자가 처형당함
      51: 레드팀 참가자가 처형당함
    """
    killer, victim = int(killer), int(victim)
    if not 1 <= victim <= 10:
        raise ValueError(f'victim must be a participant 1-10, got {victim}')
    if 1 <= killer <= 10:
        d = 0 if killer <= N_SLOTS else 1
        i, j = (killer - 1) % N_SLOTS, (victim - 1) % N_SLOTS
        return (d * N_SLOTS + i) * N_SLOTS + j
    return EXEC_MARK0 + (0 if victim <= N_SLOTS else 1)


# -----------------------------------------------------------------------------
# 보조 연산 함수 (거리 분산도 및 참여자 비트 연산)
# -----------------------------------------------------------------------------
def _spread(x: np.ndarray, y: np.ndarray) -> float:
    """
    교전 단위 내에서 발생한 킬들 간의 최대 유클리드 거리를 계산합니다.
    - 킬이 1개뿐인 경우 거리 분산도는 0.0입니다.
    - 교전이 한 자리에서 일어났는지, 추격전으로 맵에 넓게 퍼졌는지를 나타냅니다.
    """
    if x.size < 2:
        return 0.0
    dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
    return float(np.sqrt((dx * dx + dy * dy).max()))


def _is_pid(a: np.ndarray) -> np.ndarray:
    """
    배열의 각 원소가 유효한 챔피언 참가자 ID(1~10)인지 확인하는 불리언 마스크
    """
    return (a >= 1) & (a <= 10)


def _pid_bits(pid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    유효한 참가자 ID에 대해 1 << (pid - 1) 비트를 켜서 반환합니다.
    """
    return np.where(valid, np.left_shift(1, np.clip(pid - 1, 0, 9)), 0)


def kill_people_mask(k: KillTable) -> np.ndarray:
    """
    각 킬 이벤트마다 관련된 모든 참가자(킬러, 피해자, 어시스트)를 10비트 마스크 정수로 통합합니다.
    같은 플레이어가 여러 역할을 맡더라도 비트 OR 연산을 통해 중복 없이 고유 참여자로 집계됩니다.
    """
    killer, victim = k.killer.astype(np.int64), k.victim.astype(np.int64)
    people = (k.amask.astype(np.int64) & _ALL_PIDS) | _pid_bits(killer, _is_pid(killer)) | _pid_bits(victim, _is_pid(victim))
    return people.astype(np.int16)


# -----------------------------------------------------------------------------
# 메인 빌더: 매치 전체 교전 단위(UnitTable) 생성
# -----------------------------------------------------------------------------
def build_units(k: KillTable, team: np.ndarray, T_ms: int) -> UnitTable:
    """
    킬 테이블(KillTable), 팀 정보, 경기 종료 시각(T_ms)을 바탕으로 전체 교전 단위(UnitTable)를 구축합니다.

    처리 순서:
    1. 같은 팀끼리 처치한 비정상 경기(Teamkill) 오류 검사 및 경기 제외
    2. link_components를 호출하여 킬들을 시공간 교전 성분(comp)으로 클러스터링
    3. 각 교전의 시간 시계(Clock anchors) 계산:
       - f: 첫 킬 시각 (First kill time)
       - last: 마지막 킬 시각 (Last kill time)
       - tau = f - 15,000ms: 교전 시작 규칙 앵커 (O의 귀속 창 [tau, delta)의 시작점이며, 모델 입력 시점이 아님; K는 교전 성분의 킬로
         모두 [f, last] ⊂ [tau, delta)에 있음; 뒤 흐름 전환 창 (delta, delta + 60초]는 별개; 설계서 §3.1)
       - delta = max(last + 1ms, f + 20,000ms): 교전 결과 수렴 시점
    4. 블루팀/레드팀 고유 참여 인원수(nb, nr) 집계 및 교전 규모(size: T/S/ASYM/P) 분류
    5. 선제 킬 표지(mark: 0~51) 및 교전 지리적 분산도(spread) 산출
    6. 유효 분석 범위 필터링 (2분 이후 ~ 경기 종료 30초 전: in_scope)
    """
    team = np.asarray(team)
    if team.shape != (10,):
        raise ValueError(f'team must hold 10 team ids, got shape {team.shape}')

    # 팀킬 검증: 킬러와 피해자가 모두 챔피언인데 같은 팀이면 데이터 오류로 간주하여 경기 제외
    killer, victim = k.killer.astype(np.int64), k.victim.astype(np.int64)
    both = _is_pid(killer) & _is_pid(victim)
    if np.any(team[killer[both] - 1] == team[victim[both] - 1]):
        raise MatchRejected('same_team_kill')

    # 참가자별 팀 마스크 생성 (100번: 블루팀 비트마스크, 200번: 레드팀 비트마스크)
    credit = kill_people_mask(k)
    blue = sum(1 << i for i in range(10) if team[i] == BLUE)
    red = sum(1 << i for i in range(10) if team[i] == RED)

    # 킬들을 시공간 클러스터링하여 교전 단위 리스트 생성
    comp = link_components(k.t, k.x, k.y)
    first = np.array([g[0] for g in comp], dtype=np.intp)
    final = np.array([g[-1] for g in comp], dtype=np.intp)

    # 각 교전의 첫 킬 시각(f)과 마지막 킬 시각(last) 추출
    f, last = k.t[first].astype(np.int64), k.t[final].astype(np.int64)

    # 교전에 속한 모든 킬의 참여자 마스크를 비트 OR 연산으로 합쳐 고유 참여자 산출
    people = np.array([np.bitwise_or.reduce(credit[g]) for g in comp], dtype=np.int64)
    nb, nr = _POPCOUNT[people & blue], _POPCOUNT[people & red]

    # 최종 UnitTable 구조체 생성 및 반환
    return UnitTable(
        comp=comp,
        f=f,
        last=last,
        tau=f - C.TAU_LEAD_MS,                                   # 첫 킬 15초 전 규칙 앵커: 귀속 창 [tau, delta)의 시작 (입력 시점 아님)
        delta=np.maximum(last + 1, f + C.DELTA_AFTER_F_MS),      # 교전 결과 확정 시점 (해상도 컷오프)
        nb=nb,
        nr=nr,
        size=np.array([size_class(int(a), int(b)) for a, b in zip(nb, nr)], dtype=np.int8),
        mark=np.array([first_kill_mark(int(killer[i]), int(victim[i])) for i in first], dtype=np.int16),
        in_scope=(f >= C.SCOPE_FIRST_MS) & (f <= T_ms - C.END_CUT_MS),  # 유효 분석 구간 (2분 ~ 종료 30초 전)
        spread=np.array([_spread(k.x[g], k.y[g]) for g in comp], dtype=np.float32),
        n_kills=np.array([len(g) for g in comp], dtype=np.int16)
    )
