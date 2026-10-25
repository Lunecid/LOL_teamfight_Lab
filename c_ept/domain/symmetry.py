"""C-EPT 진영 대칭성(Team Symmetry) 및 180도 맵 회전 변환 모듈 (c_ept.domain.symmetry).

소환사의 협곡 맵은 점대칭(180도 회전 시 동일한 기하 구조)을 가집니다.
본 모듈은 관측 뷰(Observation View)를 180도 점대칭 회전 (x, y) -> (W - x, W - y)하고
블루팀(Blue)과 레드팀(Red)의 진영을 교환하는 기능을 제공합니다.

이를 통해 모델이 단일 가중치 세트로 블루팀과 레드팀 시점 모두를 공평하게 학습(데이터 증강 및 진영 불변성 확보)할 수 있습니다.

주요 변환 규칙:
  1. swap_participants: 참가자 인덱스(0~4 <-> 5~9) 스왑
  2. swap_actors: 10비트 액터 마스크의 하위 5비트(블루)와 상위 5비트(레드) 스왑
  3. swap_mark: 첫 킬 매치업 표지(0~51)의 진영 반전
  4. swap_K, swap_O: 킬 성과(K) 및 오브젝트 성과(O)의 팀별 열 교환
  5. OBJ_SWAP, LANDMARK_SWAP: 건물 및 오브젝트 상태 열의 탑/봇 대칭 교환 (180도 회전 시 Top <-> Bot)
  6. swap_view: 멀티모달 입력 딕셔너리 전체의 상대 진영 시점 변환 (자기 역치환 swap_view(swap_view(v)) == v 보장)
"""
from __future__ import annotations

import math
import re

import numpy as np

from c_ept import config as C
from c_ept.domain import state
from c_ept.preprocessing import terrain, tokens

# 소환사의 협곡 맵 변 길이 (14,875 유닛)
W = C.W

# 진영 교환 시 10명 플레이어 행 순서: 블루(0~4)와 레드(5~9)를 서로 맞바꿈 [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
PLAYER_ORDER = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
PLAYER_ORDER.setflags(write=False)

# 회전 변환 대상 좌표 열 인덱스 (x, y, 직전 x, 직전 y)
_XY_COLUMNS = [tokens.PLAYER_NUM.index(name) for name in ('x', 'y', 'prev_x', 'prev_y')]
_N_OLD = len(tokens.OLD_COLUMNS)

# 180도 점대칭 회전 시 탑 라인과 봇 라인이 서로 맞바뀜 (미드는 대칭 유지)
_LANE = {'top': 'bot', 'mid': 'mid', 'bot': 'top'}
_LANE_COLUMN = re.compile(r'(inhib|plates)_(top|mid|bot)(_respawn_s)?')


def _integers(name: str, a) -> np.ndarray:
    """배열이 정수형(int/uint)인지 검증합니다."""
    a = np.asarray(a)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{name} must hold whole numbers (int), not {a.dtype}')
    return a


# -----------------------------------------------------------------------------
# 1. 기본 엔티티 스왑: 참가자, 액터 비트, 마크, 타깃(K, O)
# -----------------------------------------------------------------------------
def swap_participants(p) -> np.ndarray:
    """참가자 인덱스(0~9)를 반대 진영의 동일 슬롯 인덱스로 교환합니다: p' = (p + 5) % 10.
    참가자 없음(-1)은 그대로 유지됩니다.
    """
    p = _integers('participants', p)
    if ((p < -1) | (p > 9)).any():
        raise ValueError('participants must be -1 (none) or 0 .. 9')
    return np.where(p >= 0, (p + 5) % 10, p)


def swap_actors(a) -> np.ndarray:
    """10비트 액터 마스크의 하위 5비트(블루팀 0~4번)와 상위 5비트(레드팀 5~9번)를 서로 맞바꿉니다."""
    a = _integers('actors', a)
    if ((a < 0) | (a > 1023)).any():
        raise ValueError('an actor mask holds ten bits: 0 .. 1023')
    return ((a & 31) << 5) | ((a >> 5) & 31)


def swap_mark(m):
    """첫 킬 매치업 표지(0~51)를 반대 진영 시점으로 변환합니다.
    - 0~49 (챔피언 간 첫 킬): 킬러 진영이 반전되어 m' = (m + 25) % 50
    - 50 <-> 51 (처형): 블루팀 사망(50)과 레드팀 사망(51)이 서로 교환됨 (m' = 101 - m)
    """
    a = _integers('mark', m)
    if ((a < 0) | (a >= C.N_MARKS)).any():
        raise ValueError(f'a mark is 0 .. {C.N_MARKS - 1}')
    out = np.where(a < 50, (a + 25) % 50, 101 - a)
    return int(out) if out.ndim == 0 else out


def _pair_swap(name: str, x, n: int) -> np.ndarray:
    """마지막 축의 인접한 (2k, 2k+1) 쌍(블루칸, 레드칸)을 비트 XOR(^ 1)로 서로 맞바꿉니다."""
    x = np.asarray(x)
    if x.ndim == 0 or x.shape[-1] != n:
        raise ValueError(f'{name} must have {n} cells on its last axis, got shape {x.shape}')
    return x[..., np.arange(n) ^ 1]


def swap_K(K) -> np.ndarray:
    """킬 성과 벡터 K (8개 열: 블루/레드 킬수, 현상금, 제압골드, 어시스트)의 팀별 열을 맞바꿉니다."""
    return _pair_swap('K', K, 8)


def swap_O(O) -> np.ndarray:
    """오브젝트 성과 벡터 O (18개 열: 9종 오브젝트 × 블루/레드)의 팀별 열을 맞바꿉니다."""
    return _pair_swap('O', O, 2 * len(C.O_KINDS))


# -----------------------------------------------------------------------------
# 2. 오브젝트 상태(Objective State) 열 치환 정의
# -----------------------------------------------------------------------------
def _partner(name: str) -> str:
    """오브젝트 상태 피처 열 이름이 180도 회전 시 어떤 대응 열로 변환되는지 결정합니다:
    - 'blue_' <-> 'red_' 접두사 교환
    - 탑(top)과 봇(bot) 레인 교차 대칭 적용
    - 중립 몬스터 둥지 및 전역 타이머는 변경 없이 유지
    """
    if not name.startswith(('blue_', 'red_')):
        return name
    side, rest = name.split('_', 1)
    lane = _LANE_COLUMN.fullmatch(rest)
    if lane:
        rest = f'{lane.group(1)}_{_LANE[lane.group(2)]}{lane.group(3) or ""}'
    return f'{"red" if side == "blue" else "blue"}_{rest}'


def _obj_swap() -> np.ndarray:
    """89개 오브젝트 상태 열의 순열(Permutation) 배열을 구축하고 자기 역치환(Involutive) 여부를 검증합니다."""
    names = tuple(state.OBJ_STATE_NAMES)
    index = {name: i for i, name in enumerate(names)}
    partners = [_partner(name) for name in names]
    unknown = sorted(p for p in partners if p not in index)
    if unknown:
        raise ValueError(f'the objective state has no column {unknown} to swap with')
    perm = np.array([index[p] for p in partners], dtype=np.int64)
    if sorted(perm.tolist()) != list(range(len(names))) or not np.array_equal(perm[perm], np.arange(len(names))):
        raise ValueError('the swap of the objective-state columns is not a permutation that is its own inverse')
    perm.setflags(write=False)
    return perm


# 89개 오브젝트 상태 열의 스왑 순열 (읽기 전용)
OBJ_SWAP = _obj_swap()


def _summary_swap() -> np.ndarray:
    """전역 요약(Summary) 벡터 내의 과거 사건 통계 및 오브젝트 상태 열에 대한 순열 배열을 생성합니다."""
    perm = np.arange(len(tokens.SUMMARY))
    old0, obj0 = tokens.SUMMARY.index(tokens.OLD_COLUMNS[0]), tokens.SUMMARY.index(state.OBJ_STATE_NAMES[0])
    perm[old0:old0 + _N_OLD] = old0 + (np.arange(_N_OLD) ^ 1)
    perm[obj0:] = obj0 + OBJ_SWAP
    perm.setflags(write=False)
    return perm


SUMMARY_SWAP = _summary_swap()
_SUMMARY_SWAP = SUMMARY_SWAP

if terrain.ZONES[terrain.ZONE_SWAP[terrain.ZONES.index('top')]] != 'bot':
    raise RuntimeError('the zone rules and the lane rule of OBJ_SWAP disagree: the rotation must turn the top lane into the bottom lane')


# -----------------------------------------------------------------------------
# 3. 랜드마크(건물 및 둥지) 대칭 대응 도출
# -----------------------------------------------------------------------------
_LANDMARK_LANE = {f'{lane.upper()}_LANE': f'{other.upper()}_LANE' for lane, other in _LANE.items()}
_PARTNER_MAX_OFFSET = 200.0


def derive_landmark_swap(landmarks) -> tuple[np.ndarray, float]:
    """소환사의 협곡 건물(타워, 억제기)들을 180도 회전시킨 위치 (W - x, W - y)에 가장 가까운 상대 팀 건물을 찾아 1:1로 짝을 짓습니다.

    - 탑 라인과 봇 라인의 건물이 정확히 교차 대응됩니다 (Top <-> Bot).
    - 넥서스 쌍둥이 타워 2개도 기하학적 최근접 위치에 따라 1:1 대응됩니다.
    - 소유 팀이 없는 중립 몬스터 둥지(드래곤 둥지, 바론 둥지)는 자기 자신이 짝이 됩니다.

    Returns:
        tuple[np.ndarray, float]: (34개 랜드마크 순열 배열, 회전된 건물과 짝 사이의 최대 유클리드 오차 거리)
    """
    def kind_of(j):
        return landmarks[j]['kind'], landmarks[j]['building'], landmarks[j]['tower']

    n = len(landmarks)
    owned = [i for i in range(n) if landmarks[i]['team'] is not None]
    perm, offset = np.arange(n, dtype=np.int64), 0.0

    for i in owned:
        a = landmarks[i]
        rx, ry = W - a['x'], W - a['y']
        mates = [j for j in owned if landmarks[j]['team'] != a['team'] and kind_of(j) == kind_of(i)]
        if not mates:
            raise ValueError(f'landmark {i} has no partner: there is no building of the other team of the same kind and tier')
        best = min(mates, key=lambda j: ((landmarks[j]['x'] - rx) ** 2 + (landmarks[j]['y'] - ry) ** 2, j))
        perm[i] = best
        offset = max(offset, math.hypot(landmarks[best]['x'] - rx, landmarks[best]['y'] - ry))

    if sorted(perm.tolist()) != list(range(n)) or not np.array_equal(perm[perm], np.arange(n)):
        raise ValueError('the swap of the landmark rows is not a permutation that is its own inverse')

    crossed = [i for i in owned if landmarks[perm[i]]['lane'] != _LANDMARK_LANE.get(landmarks[i]['lane'])]
    if crossed:
        raise ValueError(f'the partners of the landmarks {crossed} do not lie in the lane that the rule of OBJ_SWAP gives '
                         '(the rotation turns the top lane into the bottom lane)')

    if offset > _PARTNER_MAX_OFFSET:
        raise ValueError(f'a rotated building lies {offset:,.1f} u from its partner: more than {_PARTNER_MAX_OFFSET:,.0f} u')

    perm.setflags(write=False)
    return perm, offset


LANDMARK_SWAP, LANDMARK_SWAP_OFFSET = derive_landmark_swap(state.LANDMARKS)

if len(LANDMARK_SWAP) != tokens.N_LANDMARKS:
    raise RuntimeError('LANDMARK_SWAP and the views disagree on the number of landmarks')


# -----------------------------------------------------------------------------
# 4. 전체 멀티모달 입력 뷰(View)의 진영 스왑
# -----------------------------------------------------------------------------
def _check_view(v: dict) -> None:
    """입력 뷰 딕셔너리의 키와 텐서 형상이 tokens.VIEW_SPEC 규격과 일치하는지 검증합니다."""
    spec = tokens.VIEW_SPEC
    extra, missing = sorted(set(v) - set(spec)), sorted(set(spec) - set(v))
    if extra:
        raise ValueError(f'swap_view does not know how to swap {extra}: a silent pass-through would leave a team-dependent array unswapped')
    if missing:
        raise ValueError(f'swap_view needs {missing}')
    for key, (shape, _) in spec.items():
        if np.shape(v[key]) != shape:
            raise ValueError(f'{key}: expected shape {shape}, got {np.shape(v[key])}')


def swap_view(v: dict) -> dict:
    """tokens.build_view가 생성한 멀티모달 관측 뷰(v)를 상대 진영 시점으로 180도 회전/치환합니다.

    - 원본 딕셔너리 v는 변경되지 않으며(불변성), 완전히 새로운 배열들로 구성된 딕셔너리를 반환합니다.
    - 정수 좌표계에서 swap_view(swap_view(v)) == v (자기 역치환, Involutive)이 수학적으로 성립합니다.

    변환 내용:
      1. 플레이어: 0~4번(블루)과 5~9번(레드) 행 스왑, 좌표 (x, y) -> (W - x, W - y), 구역(zone) 대칭 변환
      2. 이벤트 토큰: 팀 코드(0 <-> 1) 반전, 액터 비트마스크 스왑, 좌표 회전
      3. 상호작용 엣지: 참여자 i, j 슬롯 교환
      4. 요약 벡터: 과거 이벤트 블루/레드 통계 스왑 및 OBJ_SWAP 적용
      5. 측지선 거리 텐서(geo_pp, geo_pl): 플레이어 순서에 맞춰 행/열 재배열

    Loader 주의사항 (S4 Loader가 처리해야 할 추가 작업):
    swap_view 변환 후의 뷰가 물리적으로 완전히 일관(physically consistent)되려면 다음 사항이 고려되어야 합니다:
    - 34개 랜드마크(landmarks) 행 및 geo_pl의 34개 열은 제자리에 유지되므로, 원래 뷰의 건물 w가 회전된 좌표(W - x, W - y)에 상대 진영 건물로 표시됩니다.
    - S4 loader는 이를 두 가지 방식으로 처리할 수 있습니다:
      1. LANDMARK_SWAP을 적용하여 행과 열을 치환: landmarks[LANDMARK_SWAP], geo_pl[:, LANDMARK_SWAP].
      2. 또는 랜드마크에 행별 고유 식별자를 부여하지 않고(no per-row identity), 소유권(ownership)과 위치(positions)를 반전시킵니다.
    - 학습된 지형 필드(G6c)는 실제 물리적 위치(physical position)에 바인딩되므로, 회전된 뷰의 좌표에 대해 W - x 물리적 역변환 위치에서 샘플링되어야 합니다.
    """
    _check_view(v)

    # 플레이어 행 순서 교환 및 좌표 180도 점대칭 반전
    players = np.asarray(v['players_num'])[PLAYER_ORDER]
    players[:, _XY_COLUMNS] = W - players[:, _XY_COLUMNS]

    # 플레이어 소속 지형 구역(zone) 대칭 매핑
    z = np.asarray(v['players_zone'])
    zone = np.where(z >= 0, terrain.ZONE_SWAP[np.where(z >= 0, z, 0)], z)[PLAYER_ORDER]

    team = np.asarray(v['ev_team'])
    geo_pp = np.asarray(v['geo_pp'])

    return {
        'players_num': players,
        'players_zone': zone,
        'landmarks': np.array(v['landmarks']),
        'ev_type': np.array(v['ev_type']),
        'ev_dt': np.array(v['ev_dt']),
        'ev_team': np.where(team < tokens.TEAM_NONE, team ^ 1, team),   # 블루(0) <-> 레드(1) 비트 반전
        'ev_xy': W - np.asarray(v['ev_xy']),                           # 이벤트 발생 위치 180도 점대칭 회전
        'ev_actors': swap_actors(v['ev_actors']),                      # 관여 챔피언 비트마스크 스왑
        'ev_gold': np.array(v['ev_gold']),
        'ev_n': np.array(v['ev_n']),
        'ev_dropped': np.array(v['ev_dropped']),
        'ed_i': swap_participants(v['ed_i']),                          # 상호작용 엣지 주체 교환
        'ed_j': swap_participants(v['ed_j']),                          # 상호작용 엣지 대상 교환
        'ed_dt': np.array(v['ed_dt']),
        'ed_kind': np.array(v['ed_kind']),
        'ed_n': np.array(v['ed_n']),
        'ed_dropped': np.array(v['ed_dropped']),
        'summary': np.asarray(v['summary'])[SUMMARY_SWAP],             # 요약 통계 및 오브젝트 상태 스왑
        'geo_pp': geo_pp[PLAYER_ORDER][:, PLAYER_ORDER],               # 플레이어 간 측지 거리 행/열 재배열
        'geo_pl': np.asarray(v['geo_pl'])[PLAYER_ORDER],               # 플레이어-랜드마크 거리 행 재배열
    }
